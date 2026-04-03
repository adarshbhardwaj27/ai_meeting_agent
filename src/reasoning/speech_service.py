"""Speech synthesis service for spoken bot replies."""

from __future__ import annotations

import base64
from typing import Optional

import httpx
import structlog

from src.config import config

logger = structlog.get_logger()


class SpeechService:
    """Generate MP3 audio for short spoken bot replies."""

    _API_BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(self):
        self._api_key = config.speech.elevenlabs_api_key
        self._model = config.speech.elevenlabs_model_id
        self._voice_id = config.speech.elevenlabs_voice_id
        self._output_format = config.speech.elevenlabs_output_format
        self._latency_mode = config.speech.elevenlabs_latency_mode
        self._initialize_client()

    @property
    def available(self) -> bool:
        return bool(config.speech.enable_voice_replies and self._api_key)

    def _initialize_client(self):
        if not config.speech.enable_voice_replies:
            return

        if not self._api_key:
            logger.info("speech_disabled_no_elevenlabs_key")
            return

        logger.info(
            "speech_service_initialized",
            provider="elevenlabs",
            model=self._model,
            voice_id=self._voice_id,
        )

    async def synthesize_mp3(self, text: str) -> Optional[bytes]:
        """Convert response text into MP3 audio bytes."""
        if not self.available:
            return None

        spoken_text = (text or "").strip()
        if not spoken_text:
            return None

        try:
            params = {"output_format": self._output_format}
            if self._latency_mode >= 0:
                params["optimize_streaming_latency"] = self._latency_mode

            payload = {
                "text": spoken_text,
                "model_id": self._model,
            }
            headers = {
                "xi-api-key": self._api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            }

            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(
                    f"{self._API_BASE_URL}/text-to-speech/{self._voice_id}",
                    params=params,
                    headers=headers,
                    json=payload,
                )

            if response.status_code == 200:
                return response.content

            logger.error(
                "speech_synthesis_failed",
                status=response.status_code,
                error=response.text,
            )
            return None
        except Exception as e:
            logger.error("speech_synthesis_failed", error=str(e))
            return None

    async def synthesize_mp3_b64(self, text: str) -> Optional[str]:
        """Convert text into base64 MP3 for provider APIs that expect inline audio."""
        audio_bytes = await self.synthesize_mp3(text)
        if not audio_bytes:
            return None
        return base64.b64encode(audio_bytes).decode("ascii")
