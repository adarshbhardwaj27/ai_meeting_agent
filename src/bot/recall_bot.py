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
from typing import Optional, Callable
import structlog

from src.bot.base_bot import BaseMeetingBot, BotState, MeetingPlatform
from src.config import config

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

    def get_page(self):
        """Not applicable for Recall.ai — there is no browser page."""
        return None

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

            # Build the bot creation payload
            payload = {
                "meeting_url": meeting_url,
                "bot_name": f"{self.bot_name} (AI)",
                "transcription_options": {
                    "provider": config.recall.transcription_provider,
                },
                "chat": {
                    "on_bot_join": {
                        "send_to": "everyone",
                        "message": f"👋 Hi! I'm {self.bot_name}, an AI meeting assistant. "
                        f"Type @{self.bot_name} followed by a command to interact with me.",
                    }
                },
            }

            # Add real-time transcript webhook if configured
            webhook_url = config.recall.webhook_url
            if webhook_url:
                payload["real_time_transcription"] = {
                    "destination_url": webhook_url,
                    "partial_results": True,
                }

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

                        # Start monitoring chat messages
                        self._chat_monitor_task = asyncio.create_task(
                            self._monitor_chat()
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
        self._set_state(BotState.LEAVING)

        # Stop chat monitor
        if self._chat_monitor_task:
            self._chat_monitor_task.cancel()
            try:
                await self._chat_monitor_task
            except asyncio.CancelledError:
                pass

        if not self._bot_id:
            self._set_state(BotState.DISCONNECTED)
            return

        try:
            import httpx

            headers = {"Authorization": f"Token {self._api_key}"}

            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
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

        self._set_state(BotState.DISCONNECTED)

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

    # ── Internal helpers ─────────────────────────────────────────────

    async def _wait_for_join(self, timeout: int = 120) -> bool:
        """
        Poll the Recall.ai bot status until it's in the meeting.

        The bot goes through states: ready → joining → in_call_not_recording
        → in_call_recording → ...
        """
        try:
            import httpx

            headers = {"Authorization": f"Token {self._api_key}"}

            for i in range(timeout):
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(
                        f"{self._api_base}/bot/{self._bot_id}/",
                        headers=headers,
                    )

                    if response.status_code == 200:
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

                await asyncio.sleep(1)

            logger.error("recall_join_timeout", timeout=timeout)
            return False

        except Exception as e:
            logger.error("recall_wait_for_join_error", error=str(e))
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
