"""
Real-Time Transcript WebSocket Server for Recall.ai.

When a Recall.ai bot is created with `realtime_endpoints`, Recall.ai
connects TO this server as a WebSocket client and pushes transcript
chunks in real-time during the meeting.

Architecture:
  Recall.ai cloud ──(WebSocket)──> [tunnel] ──> this server ──> callback

Requirements:
  - `pip install websockets`
  - A tunnel (cloudflared, bore, localtunnel) to expose the local port
  - Set RECALL_REALTIME_URL in .env to the tunnel's public wss:// URL

If no tunnel is configured, the app falls back to post-meeting polling.
"""

import asyncio
import json
from typing import Callable, Optional, Set
from datetime import datetime
import structlog

logger = structlog.get_logger()


class RealtimeTranscriptServer:
    """
    WebSocket server that receives real-time transcript events from Recall.ai.

    Recall.ai connects as a client and sends JSON messages with transcript
    data (speaker, words, timestamps) as speech is detected in the meeting.
    """

    def __init__(self, port: int = 8766, callback: Optional[Callable] = None):
        self.port = port
        self._callback = callback
        self._server = None
        self._connections: Set = set()
        self._total_chunks = 0

    def set_callback(self, callback: Callable):
        """Set the callback for incoming transcript chunks."""
        self._callback = callback

    async def start(self):
        """Start the WebSocket server."""
        try:
            import websockets

            self._server = await websockets.serve(
                self._handle_connection, "0.0.0.0", self.port
            )
            logger.info("realtime_ws_server_started", port=self.port)
            print(f"🔌 Real-time WebSocket server listening on port {self.port}")

        except ImportError:
            logger.error(
                "websockets_not_installed",
                hint="Run: pip install websockets",
            )
            raise
        except OSError as e:
            logger.error(
                "realtime_ws_server_port_in_use",
                port=self.port,
                error=str(e),
            )
            raise

    async def stop(self):
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info(
                "realtime_ws_server_stopped",
                total_chunks_received=self._total_chunks,
            )
            print(f"🔌 Real-time server stopped ({self._total_chunks} chunks received)")

    async def _handle_connection(self, websocket, path=None):
        """Handle a new WebSocket connection from Recall.ai."""
        self._connections.add(websocket)
        remote = getattr(websocket, "remote_address", "unknown")
        logger.info("realtime_ws_client_connected", remote=remote, path=path)
        print("🔗 Recall.ai connected for real-time transcript!")

        try:
            async for message in websocket:
                await self._process_message(message)
        except Exception as e:
            logger.warning("realtime_ws_connection_error", error=str(e))
        finally:
            self._connections.discard(websocket)
            logger.info("realtime_ws_client_disconnected")

    async def _process_message(self, raw_message: str):
        """
        Parse a transcript event from Recall.ai and invoke callback.

        Recall.ai real-time event format (from docs):
        {
          "event": "transcript.data",
          "data": {
            "data": {
              "words": [{"text": "hello", ...}],
              "participant": {"id": 1, "name": "John", ...},
              "language_code": "en"
            },
            "bot": {"id": "..."},
            "recording": {"id": "..."},
            ...
          }
        }
        """
        try:
            data = json.loads(raw_message)
            event_type = data.get("event", "")

            # Only process transcript events
            if event_type not in ("transcript.data", "transcript.partial_data"):
                logger.debug(
                    "realtime_non_transcript_event",
                    event=event_type,
                    keys=list(data.keys()),
                )
                return

            is_final = event_type == "transcript.data"

            # Navigate: data -> data -> {words, participant}
            inner = data.get("data", {}).get("data", {})
            if not inner:
                logger.warning("realtime_empty_inner_data", event=event_type)
                return

            # Extract speaker name
            participant = inner.get("participant") or {}
            speaker = participant.get("name") or f"Participant {participant.get('id', '?')}"

            # Extract text from words array
            words = inner.get("words", [])
            text = " ".join(w.get("text", "") for w in words).strip()

            if text and self._callback:
                from src.context.models import TranscriptChunk

                now = datetime.now()
                chunk = TranscriptChunk(
                    speaker=speaker,
                    text=text,
                    timestamp=now,
                    is_final=is_final,
                )
                label = "🎤" if is_final else "💬"
                # Show latency from word timestamps if available
                latency_info = ""
                if words and words[0].get("end_timestamp"):
                    end_ts = words[-1].get("end_timestamp", {})
                    if end_ts and end_ts.get("absolute"):
                        try:
                            from datetime import timezone
                            spoken_at = datetime.fromisoformat(end_ts["absolute"].replace("Z", "+00:00"))
                            delivered_at = datetime.now(timezone.utc)
                            latency_sec = (delivered_at - spoken_at).total_seconds()
                            latency_info = f" ⏱️{latency_sec:.1f}s"
                        except Exception:
                            pass
                print(f"{label} [{speaker}]: {text}{latency_info}")
                self._callback(chunk)
                self._total_chunks += 1

        except json.JSONDecodeError:
            logger.warning(
                "realtime_invalid_json",
                message=raw_message[:200],
            )
        except Exception as e:
            logger.warning("realtime_process_error", error=str(e))

    @property
    def is_connected(self) -> bool:
        """Whether Recall.ai is currently connected."""
        return len(self._connections) > 0
