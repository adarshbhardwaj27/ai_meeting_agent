"""
Recall.ai Bot — Platform-agnostic meeting bot via Recall.ai API.

Recall.ai provides a managed service that sends a bot into any meeting
(Teams, Google Meet, Zoom, Webex) and streams real-time transcription
back via webhooks. One REST API call = bot joins + audio + STT + chat.

This replaces platform-specific browser automation (Playwright) for
platforms like Teams where browser-based approaches don't work well.

Docs: https://docs.recall.ai
"""

import asyncio
import atexit
import base64
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Callable
import structlog

from src.bot.base_bot import BaseMeetingBot, BotState, MeetingPlatform
from src.config import config
from src.reasoning.speech_service import SpeechService

logger = structlog.get_logger()

# Recall.ai API base URL (default region: US West 2)
DEFAULT_API_BASE = "https://us-west-2.recall.ai/api/v1"


class RecallBot(BaseMeetingBot):
    """
    Meeting bot powered by Recall.ai's managed infrastructure.

    Instead of launching a browser and automating DOM interactions,
    this bot makes REST API calls to Recall.ai which handles:
    - Joining the meeting as a named participant
    - Real-time audio capture + speech-to-text
    - Chat message send/receive
    - Speaker diarization
    - Recording (optional)

    The real-time transcript arrives via webhook to our local server
    (see src/transcription/recall_webhook.py).
    """

    def __init__(
        self,
        bot_name: str = None,
        platform: MeetingPlatform = MeetingPlatform.MICROSOFT_TEAMS,
    ):
        super().__init__(bot_name or config.bot.name)
        self._platform = platform
        self._bot_id: Optional[str] = None
        self._api_key: Optional[str] = None
        self._api_base: str = DEFAULT_API_BASE
        self._chat_monitor_task: Optional[asyncio.Task] = None
        self._transcript_monitor_task: Optional[asyncio.Task] = None
        self._transcript_callback: Optional[Callable] = None
        self._speech_service = SpeechService()
        self._speech_output_enabled = False

        # Load Recall.ai config
        self._api_key = config.recall.api_key
        if config.recall.api_base_url:
            self._api_base = config.recall.api_base_url

        if not self._api_key:
            logger.warning(
                "recall_api_key_missing",
                hint="Set RECALL_API_KEY in .env to use Recall.ai bot",
            )

    @property
    def platform(self) -> MeetingPlatform:
        return self._platform

    @property
    def supports_speech_output(self) -> bool:
        return self._speech_output_enabled

    def get_page(self):
        """Not applicable for Recall.ai — there is no browser page."""
        return None

    def set_transcript_callback(self, callback: Callable):
        """Set callback for transcript chunks (used in polling mode)."""
        self._transcript_callback = callback

    # ── Join / Leave ─────────────────────────────────────────────────

    async def join(self, meeting_url: str) -> bool:
        """
        Join a meeting via Recall.ai API.

        Sends a POST to /api/v1/bot/ which creates a bot instance
        that joins the meeting, captures audio, and starts transcribing.
        """
        self._set_state(BotState.CONNECTING)
        self.meeting_url = meeting_url

        try:
            import httpx

            headers = {
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "application/json",
            }

            # Build the bot creation payload (Recall.ai v2 API format)
            payload = {
                "meeting_url": meeting_url,
                "bot_name": f"{self.bot_name} (AI)",
                "recording_config": {
                    "transcript": {
                        "provider": {
                            "recallai_streaming": {
                                # prioritize_low_latency = 1-3 sec delay (vs 3-10 min for prioritize_accuracy)
                                # Tradeoff: English only, no spelling/profanity filter
                                "mode": "prioritize_low_latency",
                                "language_code": "en",
                            }
                        },
                    },
                },
                "chat": {
                    "on_bot_join": {
                        "send_to": "everyone",
                        "message": f"👋 Hi! I'm {self.bot_name}, an AI meeting assistant. "
                        f"Type @{self.bot_name} followed by a command to interact with me.",
                    }
                },
            }

            audio_output_config = self._build_automatic_audio_output_config()
            if audio_output_config:
                payload["output_media"] = audio_output_config
                self._speech_output_enabled = True
                logger.info("recall_speech_output_enabled")
            else:
                self._speech_output_enabled = False

            # If a real-time tunnel URL is configured, add realtime_endpoints
            # so Recall.ai pushes transcript chunks via WebSocket during the meeting
            if config.recall.realtime_url:
                payload["recording_config"]["realtime_endpoints"] = [
                    {
                        "type": "websocket",
                        "url": config.recall.realtime_url,
                        "events": ["transcript.data", "transcript.partial_data"],
                    }
                ]
                logger.info(
                    "recall_realtime_enabled",
                    url=config.recall.realtime_url,
                )
            else:
                print("ℹ️  No RECALL_REALTIME_URL — transcript available after meeting ends")
                print("   Set up a tunnel for real-time (see .env for instructions)")

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self._api_base}/bot/",
                    headers=headers,
                    json=payload,
                )

                if response.status_code in (200, 201):
                    data = response.json()
                    self._bot_id = data.get("id")
                    logger.info(
                        "recall_bot_created",
                        bot_id=self._bot_id,
                        meeting_url=meeting_url,
                    )
                    print(f"✅ Recall.ai bot created: {self._bot_id}")

                    # Wait for the bot to actually join the meeting
                    joined = await self._wait_for_join()
                    if joined:
                        self._set_state(BotState.IN_MEETING)
                        print("🚀 Bot has joined the meeting!")

                        # Register cleanup so bot ALWAYS leaves
                        # (even on Ctrl+C, crash, unhandled exception)
                        atexit.register(self._leave_sync)

                        # Start monitoring chat + transcript
                        self._chat_monitor_task = asyncio.create_task(
                            self._monitor_chat()
                        )
                        if self._transcript_callback:
                            self._transcript_monitor_task = asyncio.create_task(
                                self._monitor_transcript()
                            )
                        return True
                    else:
                        self._set_state(BotState.ERROR)
                        return False
                else:
                    error_text = response.text
                    logger.error(
                        "recall_bot_create_failed",
                        status=response.status_code,
                        error=error_text,
                    )
                    print(f"❌ Failed to create bot: {response.status_code} — {error_text}")
                    self._set_state(BotState.ERROR)
                    return False

        except ImportError:
            logger.error("httpx_not_installed", hint="Run: pip install httpx")
            self._set_state(BotState.ERROR)
            return False
        except Exception as e:
            error_detail = f"{type(e).__name__}: {e}" if str(e) else f"{type(e).__name__}: {repr(e)}"
            logger.error("recall_join_failed", error=error_detail)
            print(f"❌ Recall.ai join failed: {error_detail}")
            import traceback
            traceback.print_exc()
            self._set_state(BotState.ERROR)
            return False

    async def leave(self):
        """Leave the meeting by telling Recall.ai to remove the bot."""
        if self.state == BotState.DISCONNECTED:
            return  # Already left

        self._set_state(BotState.LEAVING)

        # Unregister atexit to avoid double-leave
        try:
            atexit.unregister(self._leave_sync)
        except Exception:
            pass

        # Stop monitors
        for task in (self._chat_monitor_task, self._transcript_monitor_task):
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        if not self._bot_id:
            self._set_state(BotState.DISCONNECTED)
            return

        # Use sync httpx — more robust during event loop teardown
        # (async client raises CancelledError on Ctrl+C)
        try:
            import httpx

            headers = {"Authorization": f"Token {self._api_key}"}

            with httpx.Client(timeout=15) as client:
                response = client.post(
                    f"{self._api_base}/bot/{self._bot_id}/leave_call/",
                    headers=headers,
                )
                if response.status_code in (200, 204):
                    logger.info("recall_bot_left", bot_id=self._bot_id)
                    print("👋 Bot left the meeting")
                else:
                    logger.warning(
                        "recall_leave_failed",
                        status=response.status_code,
                        error=response.text,
                    )

        except Exception as e:
            logger.error("recall_leave_error", error=str(e))

        self._bot_id = None
        self._speech_output_enabled = False
        self._set_state(BotState.DISCONNECTED)

    def _leave_sync(self):
        """
        Synchronous cleanup — called by atexit to ensure bot leaves.

        When user presses Ctrl+C, the asyncio event loop may be dead,
        so we can't use async leave(). This uses synchronous httpx.
        """
        if not self._bot_id or self.state == BotState.DISCONNECTED:
            return

        try:
            import httpx

            headers = {"Authorization": f"Token {self._api_key}"}

            with httpx.Client(timeout=10) as client:
                response = client.post(
                    f"{self._api_base}/bot/{self._bot_id}/leave_call/",
                    headers=headers,
                )
                if response.status_code in (200, 204):
                    print("\n👋 Bot left the meeting (cleanup)")
                else:
                    # Bot might already be gone — that's OK
                    print(f"\n⚠️  Bot leave returned {response.status_code} (may already be gone)")

        except Exception as e:
            print(f"\n⚠️  Could not send leave: {e}")

        self._bot_id = None  # Prevent double cleanup
        self._speech_output_enabled = False

    # ── Chat ─────────────────────────────────────────────────────────

    async def send_chat(self, message: str) -> bool:
        """
        Send a chat message into the meeting via Recall.ai API.

        Args:
            message: The text message to send

        Returns:
            True if sent successfully
        """
        if not self._bot_id or self.state != BotState.IN_MEETING:
            return False

        try:
            import httpx

            headers = {
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "application/json",
            }

            payload = {"message": message}

            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{self._api_base}/bot/{self._bot_id}/send_chat_message/",
                    headers=headers,
                    json=payload,
                )
                if response.status_code in (200, 201):
                    logger.info("recall_chat_sent", message=message[:50])
                    return True
                else:
                    logger.error(
                        "recall_chat_failed",
                        status=response.status_code,
                        error=response.text,
                    )
                    return False

        except Exception as e:
            logger.error("recall_send_chat_error", error=str(e))
            return False

    async def send_speech(self, text: str) -> bool:
        """Speak a short reply into the meeting via Recall.ai audio output."""
        if not self.supports_speech_output or not self._bot_id or self.state != BotState.IN_MEETING:
            return False

        b64_audio = await self._speech_service.synthesize_mp3_b64(text)
        if not b64_audio:
            logger.info("recall_speech_skipped_no_audio")
            return False

        try:
            import httpx

            headers = {
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "kind": "mp3",
                "b64_data": b64_audio,
            }

            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(
                    f"{self._api_base}/bot/{self._bot_id}/output_audio/",
                    headers=headers,
                    json=payload,
                )
                if response.status_code in (200, 201, 204):
                    logger.info("recall_speech_sent", text=text[:50])
                    return True

                logger.error(
                    "recall_speech_failed",
                    status=response.status_code,
                    error=response.text,
                )
                return False

        except Exception as e:
            logger.error("recall_send_speech_error", error=str(e))
            return False

    # ── Internal helpers ─────────────────────────────────────────────

    def _build_automatic_audio_output_config(self) -> Optional[dict]:
        """Enable Recall's on-demand audio endpoint with a silent placeholder clip."""
        if not config.speech.enable_voice_replies:
            return None

        if not self._speech_service.available:
            logger.info("recall_speech_disabled_no_tts")
            return None

        silent_audio = self._generate_silent_mp3_b64()
        if not silent_audio:
            logger.info("recall_speech_disabled_no_silent_audio")
            return None

        return {
            "automatic_audio_output": {
                "in_call_recording": {
                    "data": {
                        "kind": "mp3",
                        "b64_data": silent_audio,
                    }
                }
            }
        }

    def _generate_silent_mp3_b64(self) -> Optional[str]:
        """Create a tiny silent MP3 so Recall enables the output_audio endpoint."""
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            logger.warning("recall_speech_disabled_ffmpeg_missing")
            return None

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "silence.mp3"
                command = [
                    ffmpeg_path,
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=channel_layout=mono:sample_rate=24000",
                    "-t",
                    "0.25",
                    "-q:a",
                    "9",
                    "-acodec",
                    "libmp3lame",
                    str(output_path),
                    "-y",
                    "-loglevel",
                    "error",
                ]
                subprocess.run(command, check=True, capture_output=True)
                return base64.b64encode(output_path.read_bytes()).decode("ascii")
        except Exception as e:
            logger.error("recall_silent_audio_generation_failed", error=str(e))
            return None

    async def _wait_for_join(self, timeout: int = 120) -> bool:
        """
        Poll the Recall.ai bot status until it's in the meeting.

        The bot goes through states: ready → joining → in_call_not_recording
        → in_call_recording → ...
        """
        import httpx

        headers = {"Authorization": f"Token {self._api_key}"}
        consecutive_errors = 0
        max_consecutive_errors = 5

        for i in range(timeout):
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(
                        f"{self._api_base}/bot/{self._bot_id}/",
                        headers=headers,
                    )

                if response.status_code == 200:
                    consecutive_errors = 0  # Reset on success
                    data = response.json()
                    status = data.get("status_changes", [])
                    current_status = status[-1]["code"] if status else "unknown"

                    if current_status in (
                        "in_call_not_recording",
                        "in_call_recording",
                        "recording_done",
                    ):
                        logger.info(
                            "recall_bot_joined",
                            status=current_status,
                            waited_seconds=i,
                        )
                        return True

                    if current_status in ("call_ended", "fatal", "done"):
                        logger.error(
                            "recall_bot_failed_to_join",
                            status=current_status,
                        )
                        return False

                    if i % 10 == 0:
                        print(f"⏳ Waiting for bot to join... (status: {current_status})")

            except Exception as e:
                consecutive_errors += 1
                logger.warning(
                    "recall_poll_error",
                    error=f"{type(e).__name__}: {e}",
                    attempt=i,
                    consecutive_errors=consecutive_errors,
                )
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        "recall_wait_for_join_failed",
                        reason=f"{max_consecutive_errors} consecutive poll errors",
                    )
                    return False
                # Don't bail on a single network hiccup — retry

            await asyncio.sleep(2)

        logger.error("recall_join_timeout", timeout=timeout)
        return False

    async def _monitor_chat(self):
        """
        Poll Recall.ai API for new chat messages in the meeting.

        Recall.ai provides chat messages via the bot's transcript endpoint.
        We poll every 2 seconds and emit new messages via _emit_chat_message.
        """
        last_message_count = 0

        while self.state == BotState.IN_MEETING:
            try:
                import httpx

                headers = {"Authorization": f"Token {self._api_key}"}

                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(
                        f"{self._api_base}/bot/{self._bot_id}/chat_messages/",
                        headers=headers,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        messages = data if isinstance(data, list) else data.get("results", [])

                        if len(messages) > last_message_count:
                            for msg in messages[last_message_count:]:
                                sender = msg.get("sender", "Unknown")
                                text = msg.get("message", "") or msg.get("text", "")
                                if sender != self.bot_name and text:
                                    print(f"💬 Chat [{sender}]: {text}")
                                    logger.info(
                                        "recall_chat_received",
                                        sender=sender,
                                        text=text[:80],
                                    )
                                    self._emit_chat_message(sender, text)

                            last_message_count = len(messages)

                await asyncio.sleep(2)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("recall_chat_monitor_error", error=str(e))
                await asyncio.sleep(5)

    async def _monitor_transcript(self):
        """
        Poll Recall.ai for transcript — smart fallback for when real-time
        WebSocket is not configured.

        The transcript download_url is only available AFTER the recording
        finishes (status: "done"). During the meeting it stays null.

        This monitor:
        1. Watches the recording status
        2. When status becomes "done", downloads the full transcript
        3. Processes any new utterances via callback

        If realtime_endpoints are configured, this serves as a safety net
        that picks up anything missed by the WebSocket feed.
        """
        from src.context.models import TranscriptChunk
        from datetime import datetime

        last_count = 0
        recording_id = None
        has_realtime = bool(config.recall.realtime_url)
        already_warned = False
        logger.info(
            "recall_transcript_monitor_started",
            mode="realtime_fallback" if has_realtime else "post_meeting_poll",
        )

        while self.state == BotState.IN_MEETING:
            try:
                import httpx

                headers = {"Authorization": f"Token {self._api_key}"}

                async with httpx.AsyncClient(timeout=10) as client:
                    # Step 1: Get recording_id if we don't have it yet
                    if not recording_id:
                        bot_resp = await client.get(
                            f"{self._api_base}/bot/{self._bot_id}/",
                            headers=headers,
                        )
                        if bot_resp.status_code == 200:
                            bot_data = bot_resp.json()
                            recordings = bot_data.get("recordings", [])
                            if recordings:
                                recording_id = recordings[0].get("id")
                                logger.info("recall_recording_found", recording_id=recording_id)

                    if not recording_id:
                        await asyncio.sleep(5)
                        continue

                    # Step 2: Check transcript status + download_url
                    transcript_resp = await client.get(
                        f"{self._api_base}/transcript/?recording_id={recording_id}",
                        headers=headers,
                    )

                    if transcript_resp.status_code == 200:
                        transcript_data = transcript_resp.json()
                        results = transcript_data.get("results", [])

                        if results:
                            transcript_obj = results[0]
                            status_code = transcript_obj.get("status", {}).get("code", "unknown")
                            download_url = transcript_obj.get("data", {}).get("download_url")

                            # During the meeting: status is "processing", download_url is null
                            if not download_url:
                                if not already_warned and not has_realtime:
                                    logger.info(
                                        "recall_transcript_not_ready",
                                        status=status_code,
                                        hint="Transcript available after meeting ends",
                                    )
                                    already_warned = True
                                await asyncio.sleep(10)
                                continue

                            # download_url is available — recording is done or has partial data
                            dl_resp = await client.get(download_url)
                            if dl_resp.status_code == 200:
                                utterances = dl_resp.json()
                                if not isinstance(utterances, list):
                                    utterances = utterances.get("results", [])

                                if len(utterances) > last_count:
                                    source_label = "📜" if has_realtime else "🎤"
                                    for entry in utterances[last_count:]:
                                        speaker_name = entry.get("participant", {}).get("name")
                                        speaker = speaker_name or entry.get("speaker", "Unknown")
                                        words = entry.get("words", [])
                                        text = (
                                            " ".join(w.get("text", "") for w in words).strip()
                                            if words
                                            else entry.get("text", "")
                                        )

                                        if text and self._transcript_callback:
                                            chunk = TranscriptChunk(
                                                speaker=speaker,
                                                text=text,
                                                timestamp=datetime.now(),
                                                is_final=True,
                                            )
                                            print(f"{source_label} [{speaker}]: {text}")
                                            self._transcript_callback(chunk)

                                    last_count = len(utterances)

                            # If recording is "done", grab final transcript and exit monitor
                            if status_code == "done":
                                logger.info(
                                    "recall_transcript_complete",
                                    total_utterances=last_count,
                                )
                                print(f"✅ Full transcript downloaded ({last_count} utterances)")
                                break

                # Poll less aggressively — transcript isn't available real-time anyway
                await asyncio.sleep(10 if not has_realtime else 30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("recall_transcript_poll_error", error=f"{type(e).__name__}: {e}")
                await asyncio.sleep(10)

        logger.info("recall_transcript_monitor_stopped", total_entries=last_count)

    async def get_bot_status(self) -> dict:
        """Get the current status of the Recall.ai bot (for debugging)."""
        if not self._bot_id:
            return {"status": "no_bot"}

        try:
            import httpx

            headers = {"Authorization": f"Token {self._api_key}"}

            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self._api_base}/bot/{self._bot_id}/",
                    headers=headers,
                )
                if response.status_code == 200:
                    return response.json()

        except Exception as e:
            logger.error("recall_status_error", error=str(e))

        return {"status": "error"}
