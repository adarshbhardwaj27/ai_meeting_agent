"""
End-to-End Test Runner for the AI Meeting Assistant.

Loads JSON scenario files and replays them through the REAL MeetingAssistant
with REAL timing.  The app doesn't know it's being tested — it receives
transcript chunks and chat messages through the exact same callbacks,
main loop, and trigger pipeline as a live meeting.

What's real:
  • MeetingAssistant              (the production orchestrator)
  • MeetingAssistant._main_loop   (the production polling loop)
  • MeetingAssistant._on_transcript_chunk → _add_chunk  (production callback)
  • MeetingAssistant._on_chat_message → _process_command  (production path)
  • MeetingAssistant._process_trigger   (production LLM → chat path)
  • ContextEngine                 (production meeting memory)
  • TriggerEngine                 (production voice / proactive / command logic)
  • ChatManager                   (production rate-limiting, formatting)

What's faked:
    • FakeBot     — captures outgoing chat instead of sending to Teams/Zoom and explicitly disables speech/TTS
  • MockLLMService — returns instant canned responses (fast, free, deterministic)
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import structlog


def _print(msg: str):
    """Print with safe encoding for Windows terminals."""
    try:
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()
    except UnicodeEncodeError:
        sys.stdout.write(msg.encode("ascii", errors="replace").decode("ascii") + "\n")
        sys.stdout.flush()

from src.config import config
from src.context import TranscriptChunk, MeetingContext, Trigger, TriggerType
from src.context.models import CommandType, AIResponse, MessagePriority
from src.reasoning import LLMService
from src.bot.base_bot import BaseMeetingBot, BotState, MeetingPlatform
from src.main import MeetingAssistant

logger = structlog.get_logger()


# =============================================================================
# Fake Bot — captures outgoing messages instead of sending to real meeting
# =============================================================================

class FakeBot(BaseMeetingBot):
    """
    A bot that doesn't connect to any meeting.

    It records every message the app tries to send so the test runner
    can assert on them.  All other methods are no-ops.
    """

    def __init__(self):
        super().__init__()
        self.sent_messages: List[Dict[str, Any]] = []
        self.state = BotState.IN_MEETING  # pretend we're already in

    @property
    def platform(self) -> MeetingPlatform:
        return MeetingPlatform.MICROSOFT_TEAMS

    @property
    def supports_speech_output(self) -> bool:
        return False

    async def join(self, meeting_url: str) -> bool:
        return True

    async def leave(self):
        self.state = BotState.DISCONNECTED

    async def send_chat(self, message: str) -> bool:
        self.sent_messages.append({
            "text": message,
            "timestamp": datetime.now(),
        })
        _print(f"  🤖 [FakeBot] {message}")
        return True

    async def send_speech(self, text: str) -> bool:
        raise AssertionError("E2E FakeBot must not synthesize or send speech")

    def get_page(self):
        return None

    def set_transcript_callback(self, callback):
        pass


# =============================================================================
# Mock LLM — returns a canned response (fast, free, deterministic)
# =============================================================================

class MockLLMService:
    """
    Drop-in replacement for LLMService that returns instant canned responses.

    Used when the test only cares about trigger detection, not LLM quality.
    """

    async def generate_response(
        self, trigger: Trigger, context: MeetingContext, recent_transcript: str
    ) -> Optional[AIResponse]:
        # Build a short response that echoes the trigger info
        text = f"[Mock] Responding to {trigger.type.value}"
        if trigger.relevant_transcript:
            text += f": {trigger.relevant_transcript[:80]}"

        return AIResponse.create(
            text=text,
            trigger=trigger,
            priority=MessagePriority.HIGH,
        )

    async def summarize_context(
        self, existing_summary: str, chunks: List[TranscriptChunk]
    ) -> Optional[str]:
        if not chunks:
            return existing_summary or None

        latest_points = "; ".join(chunk.text.strip() for chunk in chunks[-3:] if chunk.text.strip())
        if not latest_points:
            return existing_summary or None

        summary = latest_points[:240]
        return (
            f"Summary: {summary}\n"
            "Decisions: None\n"
            "Open Questions: None\n"
            "Risks: None\n"
            "Action Items: None"
        )


# =============================================================================
# Recorded events — what happened during the test
# =============================================================================

@dataclass
class RecordedTrigger:
    """A trigger that fired during the test."""
    trigger: Trigger
    timestamp: datetime


@dataclass
class RecordedResponse:
    """A chat message the bot sent during the test."""
    text: str
    timestamp: datetime


@dataclass
class E2EResult:
    """Full result of running one test scenario."""
    meeting_id: str
    passed: bool = False
    triggers_fired: List[RecordedTrigger] = field(default_factory=list)
    responses_sent: List[RecordedResponse] = field(default_factory=list)
    assertion_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: float = 0.0


# =============================================================================
# E2E Runner — the main engine
# =============================================================================

class E2ERunner:
    """
    Loads a JSON scenario and replays it through the REAL MeetingAssistant.

    The only fakes are the Bot (no real meeting) and optionally the LLM
    (no real OpenAI calls).  Everything else — the main loop, callbacks,
    trigger engine, context engine, chat manager — is production code.

    Usage:
        runner = E2ERunner(use_real_llm=False)
        result = await runner.run_scenario("tests/e2e_scenarios/01_voice_basic.json")
        assert result.passed
    """

    def __init__(self, use_real_llm: bool = False):
        self.use_real_llm = use_real_llm

    async def run_scenario(self, scenario_path: str) -> E2EResult:
        """Run a single test scenario from a JSON file."""
        # Load scenario
        with open(scenario_path) as f:
            scenario = json.load(f)

        meeting_id = scenario.get("meeting_id", Path(scenario_path).stem)
        _print(f"\n{'='*60}")
        _print(f"🧪 E2E: {meeting_id}")
        _print(f"   {scenario.get('description', '')}")
        _print(f"{'='*60}")

        result = E2EResult(meeting_id=meeting_id)
        start_time = datetime.now()

        # ── Build the REAL pipeline with injected fakes ──────────────
        fake_bot = FakeBot()
        llm_service = LLMService() if self.use_real_llm else MockLLMService()

        # Use a fake Teams URL — FakeBot is already IN_MEETING so join is skipped
        assistant = MeetingAssistant(
            "https://teams.microsoft.com/l/meetup-join/e2e-test",
            bot=fake_bot,
            llm_service=llm_service,
        )

        # ── Hook into observable events for assertions ───────────────
        all_triggers: List[RecordedTrigger] = []
        recent_triggers: List[RecordedTrigger] = []

        def on_trigger(trigger: Trigger, context: MeetingContext):
            record = RecordedTrigger(trigger=trigger, timestamp=datetime.now())
            all_triggers.append(record)
            recent_triggers.append(record)
            logger.info(
                "e2e_trigger_fired",
                type=trigger.type.value,
                confidence=trigger.confidence,
                reason=trigger.reason[:60] if trigger.reason else "",
            )

        def on_response(response: AIResponse, trigger: Trigger):
            result.responses_sent.append(RecordedResponse(
                text=response.text,
                timestamp=datetime.now(),
            ))

        # Start the REAL MeetingAssistant
        # (creates ContextEngine, TriggerEngine, ChatManager, wires callbacks,
        #  starts _main_loop — everything except the actual meeting join)
        await assistant.start()

        # Attach hooks AFTER start so they don't interfere with init
        assistant.on_trigger_fired = on_trigger
        assistant.on_response_sent = on_response

        # ── Replay events through the REAL callbacks ─────────────────
        try:
            events = scenario.get("events", [])
            responses_before_assert = 0
            trigger_checkpoint = 0  # index into recent_triggers for "since last assert"

            for i, event in enumerate(events):
                event_type = event.get("type")
                delay_ms = event.get("delay_ms", 0)

                if delay_ms > 0 and event_type not in ("assert", "wait"):
                    await asyncio.sleep(delay_ms / 1000.0)

                if event_type == "transcript":
                    chunk = TranscriptChunk(
                        text=event["text"],
                        timestamp=datetime.now(),
                        speaker=event.get("speaker", "Unknown"),
                        is_final=event.get("is_final", True),
                        confidence=event.get("confidence", 1.0),
                    )

                    label = "🎤" if chunk.is_final else "💬"
                    _print(f"  {label} [{chunk.speaker}]: {chunk.text}")

                    # Feed through the REAL production callback
                    # (same path as Recall.ai WebSocket → _on_transcript_chunk)
                    assistant._on_transcript_chunk(chunk)

                elif event_type == "chat":
                    speaker = event.get("speaker", "Unknown")
                    text = event.get("text", "")
                    _print(f"  💬 Chat [{speaker}]: {text}")

                    # Feed through the REAL production callback
                    # (same path as bot._emit_chat_message → _on_chat_message)
                    assistant._on_chat_message(speaker, text)

                elif event_type == "wait":
                    wait_ms = event.get("delay_ms", 1000)
                    _print(f"  ⏳ Waiting {wait_ms}ms...")
                    await asyncio.sleep(wait_ms / 1000.0)

                elif event_type == "assert":
                    # Give the main loop a couple extra ticks to process
                    await asyncio.sleep(0.3)

                    assertion = self._check_assertion(
                        event, recent_triggers, result.responses_sent,
                        responses_before_assert, trigger_checkpoint
                    )
                    result.assertion_results.append(assertion)

                    status = "✅" if assertion["passed"] else "❌"
                    _print(f"  {status} Assert: {assertion['description']}")
                    if not assertion["passed"]:
                        _print(f"       Detail: {assertion.get('detail', '')}")

                    # Advance checkpoints after each assert so subsequent
                    # assertions only see NEW triggers / responses.
                    trigger_checkpoint = len(recent_triggers)
                    if event.get("condition") in ("response_sent", "response_contains"):
                        responses_before_assert = len(result.responses_sent)

        except Exception as e:
            result.error = str(e)
            logger.error("e2e_scenario_error", error=str(e))

        # ── Cleanup — stop the REAL assistant ────────────────────────
        await assistant.stop()

        result.triggers_fired = all_triggers
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        result.passed = all(a["passed"] for a in result.assertion_results)

        status = "PASSED ✅" if result.passed else "FAILED ❌"
        _print(f"\n  Result: {status}  ({result.duration_seconds:.1f}s)")
        _print(f"  Triggers: {len(all_triggers)}  |  Responses: {len(result.responses_sent)}")
        if result.error:
            _print(f"  Error: {result.error}")

        return result

    # ── Assertion checking ───────────────────────────────────────────

    def _check_assertion(
        self,
        event: Dict,
        recent_triggers: List[RecordedTrigger],
        all_responses: List[RecordedResponse],
        responses_before: int,
        trigger_checkpoint: int = 0,
    ) -> Dict[str, Any]:
        """Check a single assertion event and return result.

        ``trigger_checkpoint`` is the index into *recent_triggers* that
        marks the boundary of "already-checked" triggers.  Assertions
        only inspect triggers **at or after** the checkpoint so that
        ``no_trigger`` can coexist with an earlier ``trigger_fired`` in
        the same scenario.
        """
        condition = event.get("condition", "")
        # Slice to only consider triggers since the last checkpoint
        triggers_since = recent_triggers[trigger_checkpoint:]

        if condition == "trigger_fired":
            return self._assert_trigger_fired(event, triggers_since)

        elif condition == "no_trigger":
            return self._assert_no_trigger(event, triggers_since)

        elif condition == "response_sent":
            new_responses = all_responses[responses_before:]
            return self._assert_response_sent(event, new_responses)

        elif condition == "response_contains":
            new_responses = all_responses[responses_before:]
            return self._assert_response_contains(event, new_responses)

        else:
            return {
                "passed": False,
                "description": f"Unknown condition: {condition}",
                "detail": "",
            }

    def _assert_trigger_fired(
        self, event: Dict, recent_triggers: List[RecordedTrigger]
    ) -> Dict[str, Any]:
        """Assert that at least one trigger of the expected type fired."""
        expected_type = event.get("trigger_type")
        expected_command = event.get("command_type")
        text_contains = event.get("text_contains", "").lower()

        desc_parts = ["trigger_fired"]
        if expected_type:
            desc_parts.append(f"type={expected_type}")
        if expected_command:
            desc_parts.append(f"cmd={expected_command}")
        if text_contains:
            desc_parts.append(f"contains='{text_contains}'")
        description = " ".join(desc_parts)

        if not recent_triggers:
            return {
                "passed": False,
                "description": description,
                "detail": "No triggers fired",
            }

        # Filter by type if specified
        matching = recent_triggers
        if expected_type:
            matching = [
                t for t in matching
                if t.trigger.type.value == expected_type
            ]
            if not matching:
                actual_types = [t.trigger.type.value for t in recent_triggers]
                return {
                    "passed": False,
                    "description": description,
                    "detail": f"Expected type '{expected_type}', got: {actual_types}",
                }

        # Filter by command type if specified
        if expected_command:
            matching = [
                t for t in matching
                if getattr(t.trigger, 'command_type', None)
                and t.trigger.command_type.value == expected_command
            ]
            if not matching:
                return {
                    "passed": False,
                    "description": description,
                    "detail": f"No trigger with command_type='{expected_command}'",
                }

        # Check text_contains if specified
        if text_contains:
            matching = [
                t for t in matching
                if t.trigger.relevant_transcript
                and text_contains in t.trigger.relevant_transcript.lower()
            ]
            if not matching:
                actual_texts = [
                    t.trigger.relevant_transcript[:60] for t in recent_triggers
                    if t.trigger.relevant_transcript
                ]
                return {
                    "passed": False,
                    "description": description,
                    "detail": f"No trigger text contains '{text_contains}'. Got: {actual_texts}",
                }

        # Check speaker if specified
        expected_speaker = event.get("speaker")
        if expected_speaker:
            matching = [
                t for t in matching
                if t.trigger.reason and expected_speaker.lower() in t.trigger.reason.lower()
            ]
            if not matching:
                return {
                    "passed": False,
                    "description": description,
                    "detail": f"No trigger attributed to '{expected_speaker}'",
                }

        return {"passed": True, "description": description, "detail": ""}

    def _assert_no_trigger(
        self, event: Dict, recent_triggers: List[RecordedTrigger]
    ) -> Dict[str, Any]:
        """Assert that no triggers fired."""
        # Filter to voice triggers only if the test is about voice
        trigger_type = event.get("trigger_type")
        if trigger_type:
            filtered = [
                t for t in recent_triggers
                if t.trigger.type.value == trigger_type
            ]
        else:
            filtered = recent_triggers

        if filtered:
            types = [t.trigger.type.value for t in filtered]
            return {
                "passed": False,
                "description": "no_trigger",
                "detail": f"Expected no triggers, but got: {types}",
            }

        return {"passed": True, "description": "no_trigger", "detail": ""}

    def _assert_response_sent(
        self, event: Dict, new_responses: List[RecordedResponse]
    ) -> Dict[str, Any]:
        """Assert that the bot sent at least one response."""
        if not new_responses:
            return {
                "passed": False,
                "description": "response_sent",
                "detail": "No response was sent",
            }
        return {"passed": True, "description": "response_sent", "detail": ""}

    def _assert_response_contains(
        self, event: Dict, new_responses: List[RecordedResponse]
    ) -> Dict[str, Any]:
        """Assert that a bot response contains specific text."""
        expected = event.get("text_contains", "").lower()
        for resp in new_responses:
            if expected in resp.text.lower():
                return {
                    "passed": True,
                    "description": f"response_contains '{expected}'",
                    "detail": "",
                }

        actual = [r.text[:60] for r in new_responses]
        return {
            "passed": False,
            "description": f"response_contains '{expected}'",
            "detail": f"No response contains '{expected}'. Got: {actual}",
        }


# =============================================================================
# CLI — run scenarios from command line
# =============================================================================

async def run_all_scenarios(
    scenario_dir: str = "tests/e2e_scenarios",
    use_real_llm: bool = False,
    tags: Optional[List[str]] = None,
):
    """Discover and run all JSON scenarios in a directory."""
    scenario_dir = Path(scenario_dir)
    if not scenario_dir.exists():
        _print(f"❌ Scenario directory not found: {scenario_dir}")
        return

    files = sorted(scenario_dir.glob("*.json"))
    if not files:
        _print(f"❌ No .json files found in {scenario_dir}")
        return

    # Filter by tags if specified
    scenarios_to_run = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if tags:
            scenario_tags = data.get("tags", [])
            if not any(t in scenario_tags for t in tags):
                continue
        scenarios_to_run.append(f)

    _print(f"\n🧪 E2E Test Suite — {len(scenarios_to_run)} scenarios")
    _print(f"   LLM: {'REAL' if use_real_llm else 'MOCK'}")
    if tags:
        _print(f"   Tags: {tags}")
    _print("")

    runner = E2ERunner(use_real_llm=use_real_llm)
    results: List[E2EResult] = []

    for scenario_path in scenarios_to_run:
        result = await runner.run_scenario(str(scenario_path))
        results.append(result)

    # ── Summary ──────────────────────────────────────────────────────
    _print(f"\n{'='*60}")
    _print("📊 E2E Results Summary")
    _print(f"{'='*60}")

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_time = sum(r.duration_seconds for r in results)

    for r in results:
        status = "✅" if r.passed else "❌"
        _print(f"  {status} {r.meeting_id} ({r.duration_seconds:.1f}s)")
        if not r.passed:
            for a in r.assertion_results:
                if not a["passed"]:
                    _print(f"       ❌ {a['description']}: {a.get('detail', '')}")

    _print(f"\n  {passed} passed, {failed} failed ({total_time:.1f}s total)")
    _print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E2E Test Runner")
    parser.add_argument("--dir", default="tests/e2e_scenarios", help="Scenario directory")
    parser.add_argument("--real-llm", action="store_true", help="Use real LLM (costs money)")
    parser.add_argument("--tags", nargs="*", help="Filter by tags")
    parser.add_argument("--scenario", help="Run a single scenario file")
    args = parser.parse_args()

    if args.scenario:
        runner = E2ERunner(use_real_llm=args.real_llm)
        asyncio.run(runner.run_scenario(args.scenario))
    else:
        asyncio.run(run_all_scenarios(
            scenario_dir=args.dir,
            use_real_llm=args.real_llm,
            tags=args.tags,
        ))
