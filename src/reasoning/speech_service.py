"""Speech synthesis service for spoken bot replies."""

from __future__ import annotations

import base64
from typing import Optional

import structlog

from src.config import config

logger = structlog.get_logger()


class SpeechService:
    """Generate MP3 audio for short spoken bot replies."""

    def __init__(self):
        self._client = None
        self._model = config.speech.openai_tts_model
        self._voice = config.speech.openai_tts_voice
        self._speed = config.speech.openai_tts_speed
        self._initialize_client()

    @property
    def available(self) -> bool:
        return self._client is not None

    def _initialize_client(self):
        if not config.speech.enable_voice_replies:
            return

        if not config.llm.openai_api_key:
            logger.info("speech_disabled_no_openai_key")
            return

        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=config.llm.openai_api_key)
            logger.info(
                "speech_service_initialized",
                model=self._model,
                voice=self._voice,
            )
        except ImportError:
            logger.warning("speech_openai_not_installed")

    async def synthesize_mp3(self, text: str) -> Optional[bytes]:
        """Convert response text into MP3 audio bytes."""
        if not self._client:
            return None

        spoken_text = (text or "").strip()
        if not spoken_text:
            return None

        try:
            response = await self._client.audio.speech.create(
                model=self._model,
                voice=self._voice,
                input=spoken_text,
                response_format="mp3",
                speed=self._speed,
            )
            audio_bytes = await response.aread()
            await response.aclose()
            return audio_bytes
        except Exception as e:
            logger.error("speech_synthesis_failed", error=str(e))
            return None

    async def synthesize_mp3_b64(self, text: str) -> Optional[str]:
        """Convert text into base64 MP3 for provider APIs that expect inline audio."""
        audio_bytes = await self.synthesize_mp3(text)
        if not audio_bytes:
            return None
        return base64.b64encode(audio_bytes).decode("ascii")
