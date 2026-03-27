"""
Recall.ai Real-Time Transcript Webhook Server.

Recall.ai sends real-time transcription events to a webhook endpoint
as the meeting progresses. This module runs a lightweight aiohttp server
that receives those events, parses them into TranscriptChunk objects,
and feeds them into the existing MeetingAssistant pipeline.

Event format (Recall.ai → our webhook):
    POST /transcript
    {
        "event": "transcript.data",
        "data": {
            "transcript": {
                "original_transcript_id": "...",
                "speaker": "John Smith",
                "speaker_id": 1,
                "words": [
                    {"text": "Hello", "start_time": 1.0, "end_time": 1.5},
                    {"text": "everyone", "start_time": 1.5, "end_time": 2.0}
                ],
                "is_final": true,
                "language": "en"
            }
        }
    }

This replaces CaptionScraper for platforms that use Recall.ai (Teams, Zoom).
The downstream pipeline (ContextEngine → TriggerEngine → LLM) is unchanged.
"""

import asyncio
from datetime import datetime
from typing import Callable, Optional
import structlog

from aiohttp import web

from src.context.models import TranscriptChunk
from src.config import config

logger = structlog.get_logger()


class RecallWebhookServer:
    """
    Lightweight HTTP server that receives real-time transcript events
    from Recall.ai and converts them into TranscriptChunk objects.

    Lifecycle:
        server = RecallWebhookServer(on_transcript=callback, port=8765)
        await server.start()    # starts listening
        ...                     # meeting happens, events arrive
        await server.stop()     # shuts down
    """

    def __init__(
        self,
        on_transcript: Optional[Callable[[TranscriptChunk], None]] = None,
        port: int = None,
        host: str = "0.0.0.0",
    ):
        self.on_transcript = on_transcript
        self.port = port or config.recall.webhook_port
        self.host = host

        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._running = False

        # Stats
        self._events_received = 0
        self._chunks_emitted = 0

    async def start(self):
        """Start the webhook HTTP server."""
        self._app = web.Application()
        self._app.router.add_post("/transcript", self._handle_transcript)
        self._app.router.add_post("/events", self._handle_event)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        self._running = True
        print(f"📡 Recall.ai webhook server listening on http://{self.host}:{self.port}")
        logger.info(
            "recall_webhook_started",
            host=self.host,
            port=self.port,
        )

    async def stop(self):
        """Stop the webhook server."""
        self._running = False
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        logger.info(
            "recall_webhook_stopped",
            events_received=self._events_received,
            chunks_emitted=self._chunks_emitted,
        )

    @property
    def webhook_url(self) -> str:
        """The URL that should be given to Recall.ai for real-time transcription."""
        return f"http://localhost:{self.port}/transcript"

    # ── HTTP Handlers ────────────────────────────────────────────────

    async def _handle_transcript(self, request: web.Request) -> web.Response:
        """
        Handle real-time transcript events from Recall.ai.

        Supports both:
        - transcript.data (final transcription for an utterance)
        - transcript.partial_data (partial/streaming transcription)
        """
        try:
            body = await request.json()
            self._events_received += 1

            event_type = body.get("event", "transcript.data")
            data = body.get("data", body)

            # Extract transcript data — Recall.ai nests it differently
            # depending on the event type
            transcript = data.get("transcript", data)

            if not transcript:
                return web.Response(status=200, text="ok")

            # Build the text from word list or direct text field
            words = transcript.get("words", [])
            if words:
                text = " ".join(w.get("text", "") for w in words).strip()
            else:
                text = transcript.get("text", "").strip()

            if not text:
                return web.Response(status=200, text="ok")

            # Extract speaker info
            speaker = transcript.get("speaker", "Unknown")
            if not speaker or speaker == "":
                speaker = f"Speaker {transcript.get('speaker_id', '?')}"

            is_final = transcript.get("is_final", True)

            # Create and emit TranscriptChunk
            chunk = TranscriptChunk(
                text=text,
                timestamp=datetime.now(),
                speaker=speaker,
                is_final=is_final,
                confidence=0.95,
            )

            if is_final:
                self._chunks_emitted += 1
                print(f"📝 [{speaker}]: {text}")

            if self.on_transcript:
                self.on_transcript(chunk)

            logger.debug(
                "recall_transcript_received",
                event=event_type,
                speaker=speaker,
                text=text[:60],
                is_final=is_final,
            )

            return web.Response(status=200, text="ok")

        except Exception as e:
            logger.error("recall_webhook_handler_error", error=str(e))
            return web.Response(status=500, text=str(e))

    async def _handle_event(self, request: web.Request) -> web.Response:
        """
        Handle general Recall.ai events (bot status changes, chat messages, etc.).

        These are non-transcript events like:
        - bot.status_change
        - bot.chat_message
        - bot.participant_join
        - bot.participant_leave
        """
        try:
            body = await request.json()
            event_type = body.get("event", "unknown")

            logger.info(
                "recall_event_received",
                event=event_type,
                data_keys=list(body.get("data", {}).keys()),
            )

            # Handle specific event types
            if event_type == "bot.status_change":
                status = body.get("data", {}).get("status", {})
                code = status.get("code", "unknown")
                print(f"🔄 Bot status: {code}")

            elif event_type == "bot.chat_message":
                msg_data = body.get("data", {})
                sender = msg_data.get("sender", "Unknown")
                text = msg_data.get("message", "")
                if text:
                    print(f"💬 Chat [{sender}]: {text}")

            return web.Response(status=200, text="ok")

        except Exception as e:
            logger.error("recall_event_handler_error", error=str(e))
            return web.Response(status=500, text=str(e))

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response(
            {
                "status": "ok",
                "events_received": self._events_received,
                "chunks_emitted": self._chunks_emitted,
                "running": self._running,
            }
        )

    def get_stats(self) -> dict:
        """Get webhook server statistics."""
        return {
            "events_received": self._events_received,
            "chunks_emitted": self._chunks_emitted,
            "running": self._running,
            "port": self.port,
        }
