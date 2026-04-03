"""
Chat Manager for the AI Meeting Assistant.

Handles:
- Message formatting with bot label
- Rate limiting (proactive vs reactive)
- Message queue management
- Delivery via meeting bot
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Optional, List
from collections import deque
import structlog

from src.config import config
from src.context.models import AIResponse, MessagePriority, TriggerType
from src.bot.base_bot import BaseMeetingBot

logger = structlog.get_logger()


class ChatManager:
    """
    Manages outgoing chat messages with rate limiting and formatting.
    
    Ensures:
    - Messages are properly formatted with bot label
    - Rate limits are respected for proactive messages
    - High-priority messages get through
    - Message delivery is tracked
    """
    
    def __init__(self, bot: BaseMeetingBot):
        self.bot = bot
        self.bot_label = config.bot.label
        
        # Rate limiting config
        self.min_proactive_interval = config.chat.min_seconds_between_proactive
        self.cooldown_after_user = config.chat.cooldown_after_user_message
        
        # State tracking
        self._last_proactive_time: Optional[datetime] = None
        self._last_user_message_time: Optional[datetime] = None
        self._message_history: deque = deque(maxlen=50)
        self._pending_queue: asyncio.Queue = asyncio.Queue()
        
        # Processing task
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the chat manager background processing."""
        self._running = True
        self._processing_task = asyncio.create_task(self._process_queue())
        logger.info("chat_manager_started")
    
    async def stop(self):
        """Stop the chat manager."""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info("chat_manager_stopped")
    
    async def send(self, response: AIResponse) -> bool:
        """
        Send an AI response to the meeting chat.
        
        Args:
            response: The AI response to send
        
        Returns:
            True if sent, False if blocked by rate limiting
        """
        is_reactive = response.trigger.type in {
            TriggerType.EXPLICIT_COMMAND,
            TriggerType.VOICE_ADDRESS,
        }
        
        # Check rate limiting
        if not is_reactive and not self._can_send_proactive():
            logger.info(
                "proactive_message_rate_limited",
                priority=response.priority.value
            )
            return False

        success = False

        if self._should_send_as_speech(response):
            success = await self.bot.send_speech(self._format_speech(response.text))
            if not success:
                logger.info("speech_delivery_fell_back_to_chat")

        if not success:
            formatted_message = self._format_message(response.text)
            success = await self.bot.send_chat(formatted_message)
        
        if success:
            response.mark_sent()
            self._message_history.append(response)
            
            if not is_reactive:
                self._last_proactive_time = datetime.now()
            
            logger.info(
                "message_sent",
                is_reactive=is_reactive,
                length=len(response.text)
            )
        
        return success
    
    async def queue_message(self, response: AIResponse):
        """Queue a message for sending."""
        await self._pending_queue.put(response)
    
    def notify_user_message(self):
        """
        Call when a user sends a chat message.
        This triggers a cooldown on proactive messages.
        """
        self._last_user_message_time = datetime.now()
    
    def _can_send_proactive(self) -> bool:
        """Check if we can send a proactive message."""
        now = datetime.now()
        
        # Check minimum interval
        if self._last_proactive_time:
            elapsed = (now - self._last_proactive_time).total_seconds()
            if elapsed < self.min_proactive_interval:
                return False
        
        # Check cooldown after user message
        if self._last_user_message_time:
            elapsed = (now - self._last_user_message_time).total_seconds()
            if elapsed < self.cooldown_after_user:
                return False
        
        return True
    
    def _format_message(self, text: str) -> str:
        """Format a message with the bot label."""
        return f"{self.bot_label}: {text}"

    def _format_speech(self, text: str) -> str:
        """Format text for spoken delivery without chat-specific labels."""
        normalized = " ".join((text or "").replace("|", ". ").split())
        if not normalized:
            return ""

        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized) if segment.strip()]
        spoken_parts: List[str] = []
        for sentence in sentences:
            candidate = " ".join(spoken_parts + [sentence]).strip()
            if len(candidate) <= 160 and len(spoken_parts) < 2:
                spoken_parts.append(sentence)
            else:
                break

        spoken_text = " ".join(spoken_parts).strip() or normalized
        if len(spoken_text) > 160:
            spoken_text = spoken_text[:157].rsplit(" ", 1)[0].rstrip(".,!? ") + "..."

        return spoken_text

    def _should_send_as_speech(self, response: AIResponse) -> bool:
        """Direct voice-addressed replies may be spoken on supported bots."""
        return (
            response.trigger.type == TriggerType.VOICE_ADDRESS
            and getattr(self.bot, "supports_speech_output", False) is True
        )
    
    async def _process_queue(self):
        """Background task to process queued messages."""
        while self._running:
            try:
                # Get next message with timeout
                response = await asyncio.wait_for(
                    self._pending_queue.get(), 
                    timeout=1.0
                )
                
                # Send the message
                await self.send(response)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("queue_processing_error", error=str(e))
    
    def get_recent_messages(self, count: int = 10) -> List[AIResponse]:
        """Get recent messages sent by the bot."""
        return list(self._message_history)[-count:]
    
    def get_stats(self) -> dict:
        """Get chat manager statistics."""
        return {
            "total_messages_sent": len(self._message_history),
            "pending_messages": self._pending_queue.qsize(),
            "last_proactive_time": self._last_proactive_time.isoformat() if self._last_proactive_time else None,
            "can_send_proactive": self._can_send_proactive(),
        }
