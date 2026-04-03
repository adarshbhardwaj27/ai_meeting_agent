"""
Main Orchestrator for the Real-Time AI Meeting Assistant.

Architecture (v2 — Recall.ai platforms):
    Teams/Zoom:   Recall.ai API → WebSocket → RealtimeTranscriptServer
      → ContextEngine → TriggerEngine → LLM → ChatManager → Meeting Chat

Teams and Zoom use Recall.ai managed API (no browser, no Azure setup).
The downstream pipeline is identical regardless of platform.
"""

import asyncio
import argparse
import signal
from datetime import datetime
from typing import Callable, Optional, List, Coroutine, Any
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
from src.transcription.realtime_server import RealtimeTranscriptServer
from src.meeting_logger import MeetingLogger

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
        Main orchestrator for the AI meeting assistant (v2 — Teams/Zoom).

    Pipeline:
            Teams/Zoom:  RealtimeTranscriptServer ──transcript──> ContextEngine
                                                        |
      ContextEngine ──context──> TriggerEngine          |
                                       |                |
      ChatManager <──response── LLMService <──trigger+context──+
    """

    def __init__(
        self,
        meeting_url: str,
        *,
        bot: Optional[BaseMeetingBot] = None,
        llm_service: Optional[LLMService] = None,
    ):
        self.meeting_url = meeting_url
        self.platform = detect_platform(meeting_url)

        # Dependency-injection overrides (for testing)
        self._injected_bot = bot
        self._injected_llm = llm_service

        # Components (initialised in start())
        self.bot: Optional[BaseMeetingBot] = None
        self.realtime_server: Optional[RealtimeTranscriptServer] = None
        self.context_engine: Optional[ContextEngine] = None
        self.trigger_engine: Optional[TriggerEngine] = None
        self.llm_service: Optional[LLMService] = None
        self.chat_manager: Optional[ChatManager] = None

        # State
        self._running = False
        self._main_loop_task: Optional[asyncio.Task] = None
        self._new_chunks: List[TranscriptChunk] = []
        self._chunk_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task] = set()

        # Session logger (append-mode file log for post-meeting analysis)
        self.meeting_logger: Optional[MeetingLogger] = None

        # Observable hooks — set by tests to inspect events
        self.on_trigger_fired: Optional[Callable] = None
        self.on_response_sent: Optional[Callable] = None

        logger.info(
            "assistant_created",
            platform=self.platform.value,
            meeting_url=meeting_url,
        )

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self):
        """Start the meeting assistant."""
        logger.info("assistant_starting")

        # Create session logger (appends to logs/meeting_sessions.log)
        meeting_id = f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        source = "e2e-test" if self._injected_bot else "live"
        self.meeting_logger = MeetingLogger(
            meeting_id=meeting_id,
            source=source,
        )

        try:
            # 1. LLM service (use injected or create real)
            self.llm_service = self._injected_llm or LLMService()

            # 2. Meeting bot (use injected or create real)
            self.bot = self._injected_bot or create_bot(self.platform.value)

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

            # 7. Join the meeting (skip if bot is already in-meeting,
            #    e.g. FakeBot in E2E tests)
            if self.bot.state == BotState.IN_MEETING:
                logger.info("bot_already_in_meeting_skipping_join")
                self.meeting_logger.event("🔗", "Bot already in meeting (injected)")
            else:
                print(f"🔗 Joining meeting: {self.meeting_url}")
                print(f"📋 Platform: {self.platform.value}")
                self.meeting_logger.event("🔗", f"Joining meeting: {self.meeting_url}")
                self.meeting_logger.event("📋", f"Platform: {self.platform.value}")

                # Teams / Zoom: use Recall.ai (managed API)
                # Start real-time WebSocket server if tunnel URL is configured
                if config.recall.realtime_url:
                    try:
                        self.realtime_server = RealtimeTranscriptServer(
                            port=config.recall.realtime_ws_port,
                            callback=self._on_transcript_chunk,
                        )
                        await self.realtime_server.start()
                        print(f"🌐 Tunnel URL: {config.recall.realtime_url}")
                    except Exception as e:
                        logger.error("realtime_server_start_failed", error=str(e))
                        print(f"⚠️  Real-time server failed: {e}")
                        print("   Falling back to post-meeting transcript")
                        self.realtime_server = None

                # Set polling fallback callback
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

            source = "Recall.ai"
            print(f"🚀 Meeting assistant is LIVE — listening via {source}")
            logger.info("assistant_started", transcript_source=source)
            self.meeting_logger.event("🚀", f"Meeting assistant is LIVE — listening via {source}")
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

        await self._cancel_background_tasks()

        if self.realtime_server:
            await self.realtime_server.stop()

        if self.bot:
            await self.bot.leave()

        for svc in (self.trigger_engine, self.context_engine, self.chat_manager):
            if svc:
                await svc.stop()

        # Close session log
        if self.meeting_logger:
            self.meeting_logger.event("🛑", "Meeting assistant stopped")
            self.meeting_logger.close()
            self.meeting_logger = None

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

                elif self.context_engine and self.trigger_engine:
                    # No new chunks, but still need to check if voice buffer
                    # settle timer should fire (speaker went quiet after
                    # saying our name).
                    context = self.context_engine.get_context_snapshot()
                    triggers = await self.trigger_engine.check_for_triggers(
                        context=context, new_chunks=[]
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
        trigger_started_at = datetime.now()
        logger.info(
            "processing_trigger",
            type=trigger.type.value,
            confidence=trigger.confidence,
            reason=trigger.reason,
        )

        # Observable hook — tests use this to record trigger events
        if self.on_trigger_fired:
            self.on_trigger_fired(trigger, context)

        # Log trigger to session file
        if self.meeting_logger:
            self.meeting_logger.trigger_fired(
                trigger_type=trigger.type.value,
                speaker=trigger.reason[:40] if trigger.reason else "",
                text=trigger.relevant_transcript or "",
                confidence=trigger.confidence,
            )

        try:
            recent_text = context.get_recent_text(minutes=3)

            generation_started_at = datetime.now()
            response = await self.llm_service.generate_response(
                trigger=trigger,
                context=context,
                recent_transcript=recent_text,
            )
            generation_elapsed = (datetime.now() - generation_started_at).total_seconds()
            if self.meeting_logger:
                self.meeting_logger.event("⏱️", f"Response generation: {generation_elapsed:.2f}s")

            if response:
                delivery_started_at = datetime.now()
                sent = await self.chat_manager.send(response)
                delivery_elapsed = (datetime.now() - delivery_started_at).total_seconds()
                if sent:
                    total_elapsed = (datetime.now() - trigger_started_at).total_seconds()
                    print(f"🤖 Sent: {response.text[:80]}")
                    logger.info(
                        "response_sent",
                        trigger=trigger.type.value,
                        message=response.text[:50],
                    )
                    if self.meeting_logger:
                        self.meeting_logger.event("⏱️", f"Delivery: {delivery_elapsed:.2f}s")
                        self.meeting_logger.event("⏱️", f"Trigger to sent: {total_elapsed:.2f}s")
                    # Observable hook — tests use this to record responses
                    if self.on_response_sent:
                        self.on_response_sent(response, trigger)
                    # Log response to session file
                    if self.meeting_logger:
                        self.meeting_logger.bot_response(
                            text=response.text,
                            trigger_type=trigger.type.value,
                        )
                elif self.meeting_logger:
                    self.meeting_logger.event("⏱️", f"Delivery attempt: {delivery_elapsed:.2f}s (not sent)")
            else:
                logger.debug("llm_skipped_response", trigger=trigger.type.value)

        except Exception as e:
            logger.error("trigger_processing_error", error=str(e))

    # ── Callbacks ────────────────────────────────────────────────────

    def _on_transcript_chunk(self, chunk: TranscriptChunk):
        """Called when a new transcript chunk arrives from Recall.ai."""
        if not self.context_engine:
            return

        # Log to session file
        if self.meeting_logger:
            self.meeting_logger.transcript(
                speaker=chunk.speaker,
                text=chunk.text,
                is_final=chunk.is_final,
            )
        self._spawn_background_task(
            self.context_engine.add_transcript_chunk(chunk),
            label="context.add_transcript_chunk",
        )
        self._spawn_background_task(
            self._add_chunk(chunk),
            label="assistant.add_chunk",
        )

    async def _add_chunk(self, chunk: TranscriptChunk):
        async with self._chunk_lock:
            self._new_chunks.append(chunk)

    def _spawn_background_task(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        label: str,
    ) -> asyncio.Task:
        """Create a tracked background task so failures are not silent."""
        task = asyncio.create_task(coro, name=label)
        self._background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)
        return task

    def _on_background_task_done(self, task: asyncio.Task):
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as e:
            logger.error(
                "background_task_failed",
                task_name=task.get_name(),
                error=str(e),
            )

    async def _cancel_background_tasks(self):
        """Cancel and await all tracked background work."""
        if not self._background_tasks:
            return

        tasks = list(self._background_tasks)
        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()

    def _on_chat_message(self, sender: str, text: str):
        """Called when someone sends a chat message in the meeting."""
        if not self.chat_manager or not self.context_engine:
            return

        self.chat_manager.notify_user_message()
        self._spawn_background_task(
            self.context_engine.add_chat_message(sender, text),
            label="context.add_chat_message",
        )

        bot_name = config.bot.name.lower()
        if f"@{bot_name}" in text.lower():
            self._spawn_background_task(
                self._process_command(sender, text),
                label="assistant.process_command",
            )

    async def _process_command(self, sender: str, text: str):
        if not self.context_engine or not self.trigger_engine:
            return
        context = self.context_engine.get_context_snapshot()
        triggers = await self.trigger_engine.check_for_triggers(
            context=context, chat_text=text
        )
        for trigger in triggers:
            if trigger.type == TriggerType.EXPLICIT_COMMAND:
                trigger.speaker = sender
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

    # Set up signal handler for graceful shutdown
    shutdown_event = asyncio.Event()

    def _signal_handler():
        print("\n⏹️  Shutting down...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, _signal_handler)
        loop.add_signal_handler(signal.SIGTERM, _signal_handler)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler for SIGINT
        pass

    try:
        await assistant.start()

        # Wait until stopped or signal received
        while assistant._running and not shutdown_event.is_set():
            await asyncio.sleep(1)

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("interrupted_by_user")
    except Exception as e:
        logger.error("assistant_error", error=str(e))
    finally:
        print("🛑 Cleaning up...")
        try:
            await asyncio.wait_for(assistant.stop(), timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            print("⚠️  Cleanup timed out — bot will leave via atexit")
        except Exception as e:
            logger.error("cleanup_error", error=str(e))


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
        help="Meeting URL (Teams or Zoom)",
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
        print("   Or:  python -m src.main -m https://teams.microsoft.com/l/meetup-join/...")
        print("   Or:  python -m src.main -m https://zoom.us/j/123456789")


if __name__ == "__main__":
    main()
