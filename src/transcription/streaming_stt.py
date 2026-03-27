"""
Mock STT Service for testing/simulation.

This module provides a MockSTTService that injects text as if it
were transcribed from speech. Used by the --simulate mode.

The real transcription sources are:
  - Google Meet: CaptionScraper (reads built-in captions from DOM)
  - Teams/Zoom: RecallWebhookServer (receives Recall.ai webhook events)
"""

import asyncio
from datetime import datetime
from typing import Callable, Optional
import structlog

from src.context.models import TranscriptChunk

logger = structlog.get_logger()


class MockSTTService:
    """
    Mock STT service for testing without a real meeting.

    Accepts text input and emits it as TranscriptChunk objects.
    Used by run_simulation() in main.py.
    """

    def __init__(self, on_transcript: Optional[Callable[[TranscriptChunk], None]] = None):
        self.on_transcript = on_transcript
        self._running = False

    async def start(self):
        self._running = True
        logger.info("mock_stt_started")

    async def stop(self):
        self._running = False
        logger.info("mock_stt_stopped")

    async def inject_text(
        self,
        text: str,
        speaker: Optional[str] = None,
        is_final: bool = True,
    ):
        """Inject text as if it was transcribed from speech."""
        chunk = TranscriptChunk(
            text=text,
            timestamp=datetime.now(),
            speaker=speaker,
            is_final=is_final,
            confidence=1.0,
        )
        if self.on_transcript:
            self.on_transcript(chunk)
        logger.debug("mock_transcript_emitted", text=text[:50], speaker=speaker)
