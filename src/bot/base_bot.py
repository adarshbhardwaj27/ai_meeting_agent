"""
Base Meeting Bot interface for the AI Meeting Assistant.

Defines the abstract interface that all platform-specific bots must implement.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional
from enum import Enum
import structlog

logger = structlog.get_logger()


class BotState(Enum):
    """States of the meeting bot."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    IN_MEETING = "in_meeting"
    LEAVING = "leaving"
    ERROR = "error"


class MeetingPlatform(Enum):
    """Supported meeting platforms."""
    MICROSOFT_TEAMS = "teams"
    ZOOM = "zoom"


class BaseMeetingBot(ABC):
    """
    Abstract base class for meeting bots.
    
    Each platform (Teams, Zoom) has its own implementation
    that handles the specifics of joining meetings and sending chat messages.
    """
    
    def __init__(self, bot_name: str = "Alex"):
        self.bot_name = bot_name
        self.state = BotState.DISCONNECTED
        self.meeting_url: Optional[str] = None
        
        # Callbacks
        self._on_chat_message: Optional[Callable[[str, str], None]] = None
        self._on_state_change: Optional[Callable[[BotState], None]] = None
        self._on_participant_joined: Optional[Callable[[str], None]] = None
        self._on_participant_left: Optional[Callable[[str], None]] = None
    
    @property
    @abstractmethod
    def platform(self) -> MeetingPlatform:
        """Return the platform this bot is for."""
        pass
    
    @abstractmethod
    async def join(self, meeting_url: str) -> bool:
        """
        Join a meeting.
        
        Args:
            meeting_url: The meeting URL or ID
        
        Returns:
            True if successfully joined, False otherwise
        """
        pass
    
    @abstractmethod
    async def leave(self):
        """Leave the current meeting."""
        pass
    
    @abstractmethod
    async def send_chat(self, message: str) -> bool:
        """
        Send a chat message to the meeting.
        
        Args:
            message: The message to send
        
        Returns:
            True if sent successfully, False otherwise
        """
        pass

    @property
    def supports_speech_output(self) -> bool:
        """Whether the bot can speak into the meeting."""
        return False

    async def send_speech(self, text: str) -> bool:
        """Send spoken output into the meeting when supported."""
        return False
    
    def on_chat_message(self, callback: Callable[[str, str], None]):
        """Register callback for chat messages (sender, text)."""
        self._on_chat_message = callback
    
    def on_state_change(self, callback: Callable[[BotState], None]):
        """Register callback for state changes."""
        self._on_state_change = callback
    
    def on_participant_joined(self, callback: Callable[[str], None]):
        """Register callback for participant joins."""
        self._on_participant_joined = callback
    
    def on_participant_left(self, callback: Callable[[str], None]):
        """Register callback for participant leaves."""
        self._on_participant_left = callback
    
    def _set_state(self, new_state: BotState):
        """Update state and notify listeners."""
        old_state = self.state
        self.state = new_state
        
        logger.info(
            "bot_state_changed",
            platform=self.platform.value,
            old_state=old_state.value,
            new_state=new_state.value
        )
        
        if self._on_state_change:
            self._on_state_change(new_state)
    
    def _emit_chat_message(self, sender: str, text: str):
        """Emit chat message to callback."""
        logger.debug("chat_received", sender=sender, text=text[:50])
        if self._on_chat_message:
            self._on_chat_message(sender, text)
    
    def _emit_participant_joined(self, name: str):
        """Emit participant joined event."""
        logger.info("participant_joined", name=name)
        if self._on_participant_joined:
            self._on_participant_joined(name)
    
    def _emit_participant_left(self, name: str):
        """Emit participant left event."""
        logger.info("participant_left", name=name)
        if self._on_participant_left:
            self._on_participant_left(name)
