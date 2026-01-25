"""
Main Orchestrator for the Real-Time AI Meeting Assistant.

This is the central coordinator that:
1. Initializes all components
2. Manages the event-driven data flow
3. Coordinates transcript → context → trigger → LLM → chat pipeline
4. Handles the meeting lifecycle
"""

import asyncio
import argparse
from datetime import datetime
from typing import Optional, List
import structlog

from src.config import config
from src.context import (
    ContextEngine, TranscriptChunk, MeetingContext, 
    Trigger, TriggerType, ChatMessage
)
from src.triggers import TriggerEngine
from src.reasoning import LLMService
from src.transcription import create_stt_service, MockSTTService
from src.chat import ChatManager
from src.bot import create_bot, detect_platform, BaseMeetingBot, BotState

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer() if config.log_level == "DEBUG" else structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class MeetingAssistant:
    """
    The main orchestrator for the AI meeting assistant.
    
    Coordinates all components to provide real-time meeting assistance:
    - Joins meetings via platform-specific bots
    - Streams audio to speech-to-text
    - Maintains live context from transcript
    - Detects trigger conditions
    - Generates and sends chat messages
    """
    
    def __init__(self, meeting_url: str):
        self.meeting_url = meeting_url
        self.platform = detect_platform(meeting_url)
        
        # Initialize components (lazy - actual init in start())
        self.bot: Optional[BaseMeetingBot] = None
        self.stt_service = None
        self.context_engine: Optional[ContextEngine] = None
        self.trigger_engine: Optional[TriggerEngine] = None
        self.llm_service: Optional[LLMService] = None
        self.chat_manager: Optional[ChatManager] = None
        
        # State
        self._running = False
        self._main_loop_task: Optional[asyncio.Task] = None
        self._new_chunks: List[TranscriptChunk] = []
        self._chunk_lock = asyncio.Lock()
        
        logger.info(
            "assistant_created",
            platform=self.platform.value,
            meeting_url=meeting_url
        )
    
    async def start(self):
        """Start the meeting assistant."""
        logger.info("assistant_starting")
        
        try:
            # 1. Initialize LLM service
            self.llm_service = LLMService()
            
            # 2. Initialize meeting bot
            self.bot = create_bot(self.platform.value)
            
            # 3. Initialize chat manager
            self.chat_manager = ChatManager(self.bot)
            await self.chat_manager.start()
            
            # 4. Initialize context engine
            self.context_engine = ContextEngine(
                meeting_id=f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                platform=self.platform.value,
                on_context_updated=self._on_context_updated
            )
            
            # 5. Initialize trigger engine
            self.trigger_engine = TriggerEngine(
                on_trigger=self._on_trigger_fired
            )
            
            # 6. Initialize STT service
            self.stt_service = create_stt_service(
                on_transcript=self._on_transcript_chunk
            )
            
            # 7. Set up bot callbacks
            self.bot.on_audio_chunk(self._on_audio_chunk)
            self.bot.on_chat_message(self._on_chat_message)
            self.bot.on_state_change(self._on_bot_state_change)
            
            # 8. Join the meeting
            joined = await self.bot.join(self.meeting_url)
            if not joined:
                raise Exception("Failed to join meeting")
            
            # 9. Start all services
            await self.stt_service.start()
            await self.context_engine.start()
            await self.trigger_engine.start()
            
            # 10. Start audio capture
            await self.bot.start_audio_capture()
            
            # 11. Start main processing loop
            self._running = True
            self._main_loop_task = asyncio.create_task(self._main_loop())
            
            logger.info("assistant_started")
            
        except Exception as e:
            logger.error("assistant_start_failed", error=str(e))
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the meeting assistant."""
        logger.info("assistant_stopping")
        self._running = False
        
        # Stop main loop
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
        
        # Stop services in reverse order
        if self.bot:
            await self.bot.stop_audio_capture()
            await self.bot.leave()
        
        if self.trigger_engine:
            await self.trigger_engine.stop()
        
        if self.context_engine:
            await self.context_engine.stop()
        
        if self.stt_service:
            await self.stt_service.stop()
        
        if self.chat_manager:
            await self.chat_manager.stop()
        
        logger.info("assistant_stopped")
    
    async def _main_loop(self):
        """
        Main processing loop that coordinates trigger detection.
        
        Runs every ~500ms to check for triggers based on new transcript chunks.
        """
        while self._running:
            try:
                # Get new chunks since last check
                async with self._chunk_lock:
                    chunks_to_process = self._new_chunks.copy()
                    self._new_chunks.clear()
                
                if chunks_to_process and self.context_engine:
                    # Get current context
                    context = self.context_engine.get_context_snapshot()
                    
                    # Check for triggers
                    triggers = await self.trigger_engine.check_for_triggers(
                        context=context,
                        new_chunks=chunks_to_process
                    )
                    
                    # Process triggers
                    for trigger in triggers:
                        await self._process_trigger(trigger, context)
                
                # Sleep for trigger check interval
                await asyncio.sleep(config.trigger.check_interval_ms / 1000.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("main_loop_error", error=str(e))
                await asyncio.sleep(1)
    
    async def _process_trigger(self, trigger: Trigger, context: MeetingContext):
        """Process a fired trigger by generating and sending a response."""
        logger.info(
            "processing_trigger",
            type=trigger.type.value,
            confidence=trigger.confidence,
            reason=trigger.reason
        )
        
        try:
            # Get recent transcript for LLM
            recent_text = context.get_recent_text(minutes=3)
            
            # Generate response
            response = await self.llm_service.generate_response(
                trigger=trigger,
                context=context,
                recent_transcript=recent_text
            )
            
            if response:
                # Send via chat manager
                sent = await self.chat_manager.send(response)
                
                if sent:
                    logger.info(
                        "response_sent",
                        trigger=trigger.type.value,
                        message=response.text[:50]
                    )
            else:
                logger.debug("llm_skipped_response", trigger=trigger.type.value)
                
        except Exception as e:
            logger.error("trigger_processing_error", error=str(e))
    
    def _on_transcript_chunk(self, chunk: TranscriptChunk):
        """Callback when a transcript chunk is received from STT."""
        # Add to context engine
        asyncio.create_task(self.context_engine.add_transcript_chunk(chunk))
        
        # Add to pending chunks for trigger processing
        asyncio.create_task(self._add_chunk(chunk))
    
    async def _add_chunk(self, chunk: TranscriptChunk):
        """Add chunk to pending list (thread-safe)."""
        async with self._chunk_lock:
            self._new_chunks.append(chunk)
    
    def _on_audio_chunk(self, audio_data: bytes):
        """Callback when audio is received from the bot."""
        # Forward to STT service
        asyncio.create_task(self.stt_service.process_audio(audio_data))
    
    def _on_chat_message(self, sender: str, text: str):
        """Callback when a chat message is received."""
        # Notify chat manager of user activity
        self.chat_manager.notify_user_message()
        
        # Add to context
        asyncio.create_task(self.context_engine.add_chat_message(sender, text))
        
        # Check if it's a command for us
        bot_name = config.bot.name.lower()
        if f"@{bot_name}" in text.lower():
            # Process command immediately
            asyncio.create_task(self._process_command(sender, text))
    
    async def _process_command(self, sender: str, text: str):
        """Process a command directed at the bot."""
        if not self.context_engine or not self.trigger_engine:
            return
        
        context = self.context_engine.get_context_snapshot()
        
        # Check for command trigger
        triggers = await self.trigger_engine.check_for_triggers(
            context=context,
            chat_text=text
        )
        
        for trigger in triggers:
            if trigger.type == TriggerType.EXPLICIT_COMMAND:
                await self._process_trigger(trigger, context)
                break
    
    def _on_bot_state_change(self, state: BotState):
        """Callback when bot state changes."""
        logger.info("bot_state_change", state=state.value)
        
        if state == BotState.DISCONNECTED:
            # Meeting ended or we got kicked
            self._running = False
    
    def _on_context_updated(self, context: MeetingContext):
        """Callback when context is updated."""
        # Could be used for debugging or UI updates
        pass


async def run_assistant(meeting_url: str):
    """Run the meeting assistant until interrupted."""
    assistant = MeetingAssistant(meeting_url)
    
    try:
        await assistant.start()
        
        # Keep running until interrupted
        while assistant._running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("interrupted_by_user")
    finally:
        await assistant.stop()


async def run_simulation():
    """
    Run a simulation of the assistant for testing/demo.
    
    Uses mock services to simulate a meeting conversation.
    """
    logger.info("starting_simulation")
    
    # Create mock components
    from src.transcription.streaming_stt import MockSTTService
    
    # Initialize services
    llm_service = LLMService()
    
    context_engine = ContextEngine(
        meeting_id="simulation_001",
        platform="simulation"
    )
    
    trigger_engine = TriggerEngine()
    
    mock_stt = MockSTTService(
        on_transcript=lambda c: asyncio.create_task(context_engine.add_transcript_chunk(c))
    )
    
    # Start services
    await mock_stt.start()
    await context_engine.start()
    await trigger_engine.start()
    
    # Simulate a conversation
    conversation = [
        ("John", "Alright, let's discuss the architecture for the new service."),
        ("Sarah", "I'm worried about latency. Our current system has 200ms response times and users are complaining."),
        ("John", "Good point. We need to keep that in mind."),
        ("Mike", "What about using Kafka for the message queue?"),
        ("John", "Kafka sounds good for scalability."),
        ("Sarah", "But doesn't Kafka add latency? I thought we were trying to reduce that."),
        ("John", "Let's just go with Kafka - it'll solve our scaling problems."),
    ]
    
    print("\n" + "="*60)
    print("🎬 SIMULATION: Real-Time AI Meeting Assistant")
    print("="*60 + "\n")
    
    for speaker, text in conversation:
        # Inject transcript
        await mock_stt.inject_text(text, speaker=speaker)
        print(f"💬 {speaker}: {text}")
        
        # Give context engine time to process
        await asyncio.sleep(0.5)
        
        # Check for triggers
        context = context_engine.get_context_snapshot()
        triggers = await trigger_engine.check_for_triggers(
            context=context,
            new_chunks=[TranscriptChunk(text=text, timestamp=datetime.now(), speaker=speaker)]
        )
        
        for trigger in triggers:
            # Generate response
            recent_text = context.get_recent_text(minutes=5)
            response = await llm_service.generate_response(
                trigger=trigger,
                context=context,
                recent_transcript=recent_text
            )
            
            if response:
                print(f"\n🤖 {config.bot.label}: {response.text}\n")
        
        await asyncio.sleep(1)
    
    # Test explicit command
    print("\n" + "-"*40)
    print("Testing explicit command...")
    print("-"*40 + "\n")
    
    command = "@Alex summarize what we discussed"
    print(f"💬 Sarah: {command}")
    
    triggers = await trigger_engine.check_for_triggers(
        context=context_engine.get_context_snapshot(),
        chat_text=command
    )
    
    for trigger in triggers:
        context = context_engine.get_context_snapshot()
        response = await llm_service.generate_response(
            trigger=trigger,
            context=context,
            recent_transcript=context.get_recent_text(minutes=10)
        )
        
        if response:
            print(f"\n🤖 {config.bot.label}: {response.text}\n")
    
    # Cleanup
    await trigger_engine.stop()
    await context_engine.stop()
    await mock_stt.stop()
    
    print("\n" + "="*60)
    print("✅ Simulation complete!")
    print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-Time AI Meeting Assistant"
    )
    
    parser.add_argument(
        "--meeting-url",
        "-m",
        help="Meeting URL to join (Google Meet, Teams, or Zoom)"
    )
    
    parser.add_argument(
        "--simulate",
        "-s",
        action="store_true",
        help="Run a simulation instead of joining a real meeting"
    )
    
    args = parser.parse_args()
    
    if args.simulate:
        asyncio.run(run_simulation())
    elif args.meeting_url:
        asyncio.run(run_assistant(args.meeting_url))
    else:
        parser.print_help()
        print("\n💡 Try: python -m src.main --simulate")


if __name__ == "__main__":
    main()
