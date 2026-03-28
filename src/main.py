"""
Main Orchestrator for the Real-Time AI Meeting Assistant.

Architecture (v2 — Multi-platform):
  Google Meet:  Playwright → Built-in Captions → CaptionScraper
  Teams/Zoom:   Recall.ai API → Webhook → RecallWebhookServer
      → ContextEngine → TriggerEngine → LLM → ChatManager → Meeting Chat

Google Meet uses browser automation + caption scraping (free, no API).
Teams and Zoom use Recall.ai managed API (no browser, no Azure setup).
The downstream pipeline is identical regardless of platform.
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
from src.chat import ChatManager
from src.bot import create_bot, detect_platform, BaseMeetingBot, BotState, MeetingPlatform
from src.captions import CaptionScraper

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
        structlog.dev.ConsoleRenderer()
        if config.log_level == "DEBUG"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class MeetingAssistant:
    """
    Main orchestrator for the AI meeting assistant (v2 — multi-platform).

    Pipeline:
      Google Meet: CaptionScraper ──transcript──> ContextEngine
      Teams/Zoom:  RecallWebhook  ──transcript──> ContextEngine
                                                        |
      ContextEngine ──context──> TriggerEngine          |
                                       |                |
      ChatManager <──response── LLMService <──trigger+context──+
    """

    def __init__(self, meeting_url: str):
        self.meeting_url = meeting_url
        self.platform = detect_platform(meeting_url)

        # Components (initialised in start())
        self.bot: Optional[BaseMeetingBot] = None
        self.caption_scraper: Optional[CaptionScraper] = None
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
            meeting_url=meeting_url,
        )

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self):
        """Start the meeting assistant."""
        logger.info("assistant_starting")

        try:
            # 1. LLM service
            self.llm_service = LLMService()

            # 2. Meeting bot (platform-specific)
            self.bot = create_bot(self.platform.value)

            # 3. Chat manager (sends messages via bot)
            self.chat_manager = ChatManager(self.bot)
            await self.chat_manager.start()

            # 4. Context engine (meeting memory)
            self.context_engine = ContextEngine(
                meeting_id=f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                platform=self.platform.value,
                on_context_updated=self._on_context_updated,
            )

            # 5. Trigger engine (decides when to speak)
            self.trigger_engine = TriggerEngine()

            # 6. Bot callbacks
            self.bot.on_chat_message(self._on_chat_message)
            self.bot.on_state_change(self._on_bot_state_change)

            # 7. Join the meeting
            print(f"🔗 Joining meeting: {self.meeting_url}")
            print(f"📋 Platform: {self.platform.value}")

            # ── Platform-specific: start transcript source ───────────
            if self.platform == MeetingPlatform.GOOGLE_MEET:
                # Google Meet: use browser + caption scraping (free)
                self.caption_scraper = CaptionScraper(
                    on_transcript=self._on_transcript_chunk,
                    poll_interval_ms=300,
                    meeting_url=self.meeting_url,
                )

                joined = await self.bot.join(self.meeting_url)
                if not joined:
                    raise Exception("Failed to join meeting")

                page = self.bot.get_page()
                if page:
                    await self.caption_scraper.start(page)
                else:
                    print("⚠️  Could not get browser page for caption scraping")
                    logger.error("no_page_for_captions")

            else:
                # Teams / Zoom: use Recall.ai (managed API + polling)
                # No webhook server needed — bot polls transcript API directly
                self.bot.set_transcript_callback(self._on_transcript_chunk)

                joined = await self.bot.join(self.meeting_url)
                if not joined:
                    raise Exception("Failed to join meeting")

            # ── Common services ──────────────────────────────────────
            await self.context_engine.start()
            await self.trigger_engine.start()

            # Main processing loop
            self._running = True
            self._main_loop_task = asyncio.create_task(self._main_loop())

            source = "captions" if self.platform == MeetingPlatform.GOOGLE_MEET else "Recall.ai"
            print(f"🚀 Meeting assistant is LIVE — listening via {source}")
            logger.info("assistant_started", transcript_source=source)

        except Exception as e:
            logger.error("assistant_start_failed", error=str(e))
            await self.stop()
            raise

    async def stop(self):
        """Stop the meeting assistant gracefully."""
        logger.info("assistant_stopping")
        self._running = False

        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass

        if self.caption_scraper:
            await self.caption_scraper.stop()

        if self.bot:
            await self.bot.leave()

        for svc in (self.trigger_engine, self.context_engine, self.chat_manager):
            if svc:
                await svc.stop()

        logger.info("assistant_stopped")

    # ── Main loop ────────────────────────────────────────────────────

    async def _main_loop(self):
        """Check for triggers every ~500 ms based on new transcript chunks."""
        while self._running:
            try:
                async with self._chunk_lock:
                    chunks_to_process = self._new_chunks.copy()
                    self._new_chunks.clear()

                if chunks_to_process and self.context_engine:
                    context = self.context_engine.get_context_snapshot()

                    triggers = await self.trigger_engine.check_for_triggers(
                        context=context, new_chunks=chunks_to_process
                    )

                    for trigger in triggers:
                        await self._process_trigger(trigger, context)

                await asyncio.sleep(config.trigger.check_interval_ms / 1000.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("main_loop_error", error=str(e))
                await asyncio.sleep(1)

    # ── Trigger processing ───────────────────────────────────────────

    async def _process_trigger(self, trigger: Trigger, context: MeetingContext):
        """Generate an LLM response for a trigger and send it to chat."""
        logger.info(
            "processing_trigger",
            type=trigger.type.value,
            confidence=trigger.confidence,
            reason=trigger.reason,
        )
        try:
            recent_text = context.get_recent_text(minutes=3)

            response = await self.llm_service.generate_response(
                trigger=trigger,
                context=context,
                recent_transcript=recent_text,
            )

            if response:
                sent = await self.chat_manager.send(response)
                if sent:
                    print(f"🤖 Sent: {response.text[:80]}")
                    logger.info(
                        "response_sent",
                        trigger=trigger.type.value,
                        message=response.text[:50],
                    )
            else:
                logger.debug("llm_skipped_response", trigger=trigger.type.value)

        except Exception as e:
            logger.error("trigger_processing_error", error=str(e))

    # ── Callbacks ────────────────────────────────────────────────────

    def _on_transcript_chunk(self, chunk: TranscriptChunk):
        """Called by CaptionScraper when a new caption line appears."""
        asyncio.create_task(self.context_engine.add_transcript_chunk(chunk))
        asyncio.create_task(self._add_chunk(chunk))

    async def _add_chunk(self, chunk: TranscriptChunk):
        async with self._chunk_lock:
            self._new_chunks.append(chunk)

    def _on_chat_message(self, sender: str, text: str):
        """Called when someone sends a chat message in the meeting."""
        self.chat_manager.notify_user_message()
        asyncio.create_task(self.context_engine.add_chat_message(sender, text))

        bot_name = config.bot.name.lower()
        if f"@{bot_name}" in text.lower():
            asyncio.create_task(self._process_command(sender, text))

    async def _process_command(self, sender: str, text: str):
        if not self.context_engine or not self.trigger_engine:
            return
        context = self.context_engine.get_context_snapshot()
        triggers = await self.trigger_engine.check_for_triggers(
            context=context, chat_text=text
        )
        for trigger in triggers:
            if trigger.type == TriggerType.EXPLICIT_COMMAND:
                await self._process_trigger(trigger, context)
                break

    def _on_bot_state_change(self, state: BotState):
        logger.info("bot_state_change", state=state.value)
        if state == BotState.DISCONNECTED:
            self._running = False

    def _on_context_updated(self, context: MeetingContext):
        pass  # Hook for future UI / debugging


# ── Entry points ─────────────────────────────────────────────────────


async def run_assistant(meeting_url: str):
    """Run the meeting assistant until interrupted."""
    assistant = MeetingAssistant(meeting_url)
    try:
        await assistant.start()
        while assistant._running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("interrupted_by_user")
    finally:
        await assistant.stop()


async def run_simulation():
    """
    Run a simulation of the assistant for testing/demo.
    Uses mock transcript data — no meeting needed.
    """
    from src.transcription.streaming_stt import MockSTTService

    logger.info("starting_simulation")
    llm_service = LLMService()

    context_engine = ContextEngine(
        meeting_id="simulation_001", platform="simulation"
    )
    trigger_engine = TriggerEngine()
    mock_stt = MockSTTService(
        on_transcript=lambda c: asyncio.create_task(
            context_engine.add_transcript_chunk(c)
        )
    )

    await mock_stt.start()
    await context_engine.start()
    await trigger_engine.start()

    conversation = [
        ("John", "Alright, let's discuss the architecture for the new service."),
        (
            "Sarah",
            "I'm worried about latency. Our current system has 200ms response times.",
        ),
        ("John", "Good point. We need to keep that in mind."),
        ("Mike", "What about using Kafka for the message queue?"),
        ("John", "Kafka sounds good for scalability."),
        (
            "Sarah",
            "But doesn't Kafka add latency? I thought we were trying to reduce that.",
        ),
        ("John", "Let's just go with Kafka - it'll solve our scaling problems."),
    ]

    print("\n" + "=" * 60)
    print("🎬 SIMULATION: Real-Time AI Meeting Assistant")
    print("=" * 60 + "\n")

    for speaker, text in conversation:
        await mock_stt.inject_text(text, speaker=speaker)
        print(f"💬 {speaker}: {text}")
        await asyncio.sleep(0.5)

        context = context_engine.get_context_snapshot()
        triggers = await trigger_engine.check_for_triggers(
            context=context,
            new_chunks=[
                TranscriptChunk(
                    text=text, timestamp=datetime.now(), speaker=speaker
                )
            ],
        )

        for trigger in triggers:
            recent_text = context.get_recent_text(minutes=5)
            response = await llm_service.generate_response(
                trigger=trigger,
                context=context,
                recent_transcript=recent_text,
            )
            if response:
                print(f"\n🤖 {config.bot.label}: {response.text}\n")

        await asyncio.sleep(1)

    # Test explicit command
    print("\n" + "-" * 40)
    print("Testing explicit command...")
    print("-" * 40 + "\n")

    command = "@Alex summarize what we discussed"
    print(f"💬 Sarah: {command}")

    triggers = await trigger_engine.check_for_triggers(
        context=context_engine.get_context_snapshot(), chat_text=command
    )
    for trigger in triggers:
        context = context_engine.get_context_snapshot()
        response = await llm_service.generate_response(
            trigger=trigger,
            context=context,
            recent_transcript=context.get_recent_text(minutes=10),
        )
        if response:
            print(f"\n🤖 {config.bot.label}: {response.text}\n")

    await trigger_engine.stop()
    await context_engine.stop()
    await mock_stt.stop()

    print("\n" + "=" * 60)
    print("✅ Simulation complete!")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-Time AI Meeting Assistant"
    )

    parser.add_argument(
        "--meeting-url",
        "-m",
        help="Meeting URL (Google Meet, Teams, or Zoom)",
    )

    parser.add_argument(
        "--simulate",
        "-s",
        action="store_true",
        help="Run a simulation (no real meeting needed)",
    )

    args = parser.parse_args()

    if args.simulate:
        asyncio.run(run_simulation())
    elif args.meeting_url:
        asyncio.run(run_assistant(args.meeting_url))
    else:
        parser.print_help()
        print("\n💡 Try: python -m src.main --simulate")
        print("   Or:  python -m src.main -m https://meet.google.com/xxx-yyyy-zzz")
        print("   Or:  python -m src.main -m https://teams.microsoft.com/l/meetup-join/...")


if __name__ == "__main__":
    main()
