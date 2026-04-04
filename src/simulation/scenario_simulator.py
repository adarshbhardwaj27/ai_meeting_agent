"""Replay uploaded JSON conversations through the real assistant pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import uuid

import structlog

from src.bot.base_bot import BaseMeetingBot, BotState, MeetingPlatform
from src.context import MeetingContext, TranscriptChunk, Trigger
from src.context.models import AIResponse, MessagePriority
from src.main import MeetingAssistant

logger = structlog.get_logger()


class ScenarioPayloadError(ValueError):
    """Raised when uploaded JSON cannot be converted into replay events."""


class FakeBot(BaseMeetingBot):
    """In-memory bot used for simulation replays."""

    def __init__(self):
        super().__init__()
        self.sent_messages: List[Dict[str, Any]] = []
        self.state = BotState.IN_MEETING

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
        self.sent_messages.append(
            {
                "text": message,
                "timestamp": datetime.now().isoformat(),
            }
        )
        return True

    async def send_speech(self, text: str) -> bool:
        raise AssertionError("Scenario simulator FakeBot must not send speech")

    def get_page(self):
        return None

    def set_transcript_callback(self, callback):
        return None


class MockLLMService:
    """Deterministic response generator for uploaded scenario replay."""

    async def generate_response(
        self,
        trigger: Trigger,
        context: MeetingContext,
        recent_transcript: str,
    ) -> Optional[AIResponse]:
        text = f"[Mock] Responding to {trigger.type.value}"
        if trigger.relevant_transcript:
            text += f": {trigger.relevant_transcript[:120]}"

        return AIResponse.create(
            text=text,
            trigger=trigger,
            priority=MessagePriority.HIGH,
        )

    async def summarize_context(
        self,
        existing_summary: str,
        chunks: List[TranscriptChunk],
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


@dataclass
class SimulationResult:
    meeting_id: str
    description: str
    input_event_count: int
    duration_seconds: float
    trigger_count: int
    response_count: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meeting_id": self.meeting_id,
            "description": self.description,
            "input_event_count": self.input_event_count,
            "duration_seconds": round(self.duration_seconds, 3),
            "trigger_count": self.trigger_count,
            "response_count": self.response_count,
            "error": self.error,
        }


def normalize_scenario_payload(payload: Any, *, default_delay_ms: int = 1200) -> Dict[str, Any]:
    """Normalize uploaded JSON into replay-ready transcript/chat events."""
    if default_delay_ms < 0:
        raise ScenarioPayloadError("default_delay_ms must be non-negative")

    if isinstance(payload, list):
        return _build_from_turns(
            turns=payload,
            meeting_id="uploaded_conversation",
            description="Uploaded conversation replay",
            default_delay_ms=default_delay_ms,
            final_wait_ms=max(1800, default_delay_ms),
        )

    if not isinstance(payload, dict):
        raise ScenarioPayloadError("Uploaded JSON must be an object or a list of conversation turns")

    meeting_id = str(payload.get("meeting_id") or "uploaded_conversation")
    description = str(payload.get("description") or "Uploaded conversation replay")
    final_wait_ms = int(payload.get("final_wait_ms", max(1800, default_delay_ms)))

    if isinstance(payload.get("events"), list):
        normalized_events = []
        for index, raw_event in enumerate(payload["events"]):
            normalized_event = _normalize_existing_event(
                raw_event,
                index=index,
                default_delay_ms=default_delay_ms,
            )
            if normalized_event:
                normalized_events.append(normalized_event)
        return {
            "meeting_id": meeting_id,
            "description": description,
            "events": normalized_events,
            "final_wait_ms": final_wait_ms,
        }

    for key in ("conversation", "transcript", "messages", "turns"):
        if isinstance(payload.get(key), list):
            return _build_from_turns(
                turns=payload[key],
                meeting_id=meeting_id,
                description=description,
                default_delay_ms=default_delay_ms,
                final_wait_ms=final_wait_ms,
            )

    raise ScenarioPayloadError(
        "Uploaded JSON must contain either 'events' or a list under 'conversation', 'transcript', 'messages', or 'turns'"
    )


def _build_from_turns(
    *,
    turns: List[Any],
    meeting_id: str,
    description: str,
    default_delay_ms: int,
    final_wait_ms: int,
) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ScenarioPayloadError(f"Conversation item {index} must be an object")

        if "wait_ms" in turn:
            events.append(
                {
                    "type": "wait",
                    "delay_ms": int(turn.get("wait_ms", 0)),
                }
            )
            continue

        event_type = str(turn.get("type") or "").strip().lower()
        if event_type not in {"transcript", "chat"}:
            event_type = "chat" if turn.get("channel") == "chat" or turn.get("is_chat") else "transcript"

        text = turn.get("text") or turn.get("message")
        if not text:
            raise ScenarioPayloadError(f"Conversation item {index} must include 'text' or 'message'")

        speaker = turn.get("speaker") or turn.get("sender") or "Unknown"
        delay_ms = int(turn.get("delay_ms", turn.get("gap_ms", 0 if index == 0 else default_delay_ms)))
        events.append(
            {
                "type": event_type,
                "speaker": str(speaker),
                "text": str(text),
                "is_final": bool(turn.get("is_final", True)),
                "delay_ms": delay_ms,
            }
        )

    return {
        "meeting_id": meeting_id,
        "description": description,
        "events": events,
        "final_wait_ms": final_wait_ms,
    }


def _normalize_existing_event(event: Any, *, index: int, default_delay_ms: int) -> Optional[Dict[str, Any]]:
    if not isinstance(event, dict):
        raise ScenarioPayloadError(f"Event {index} must be an object")

    event_type = str(event.get("type") or "").strip().lower()
    if event_type == "assert":
        return None

    if event_type == "wait":
        return {
            "type": "wait",
            "delay_ms": int(event.get("delay_ms", event.get("wait_ms", default_delay_ms))),
        }

    if event_type not in {"transcript", "chat"}:
        raise ScenarioPayloadError(
            f"Unsupported event type '{event_type}' in uploaded scenario. Supported types: transcript, chat, wait"
        )

    text = event.get("text")
    if not text:
        raise ScenarioPayloadError(f"Event {index} must include 'text'")

    return {
        "type": event_type,
        "speaker": str(event.get("speaker") or event.get("sender") or "Unknown"),
        "text": str(text),
        "is_final": bool(event.get("is_final", True)),
        "delay_ms": int(event.get("delay_ms", 0 if index == 0 else default_delay_ms)),
    }


@dataclass
class ScenarioSimulator:
    """Replay normalized conversation JSON through the real assistant path."""

    meeting_url: str = "https://teams.microsoft.com/l/meetup-join/scenario-simulator"
    _timeline: List[Dict[str, Any]] = field(default_factory=list, init=False)

    async def run(
        self,
        scenario: Dict[str, Any],
        *,
        playback_speed: float = 1.0,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> SimulationResult:
        if playback_speed <= 0:
            raise ScenarioPayloadError("playback_speed must be greater than zero")

        fake_bot = FakeBot()
        assistant = MeetingAssistant(
            self.meeting_url,
            bot=fake_bot,
            llm_service=MockLLMService(),
        )

        meeting_id = str(scenario.get("meeting_id") or f"uploaded_{uuid.uuid4().hex[:8]}")
        description = str(scenario.get("description") or "Uploaded conversation replay")
        events = list(scenario.get("events") or [])
        final_wait_ms = int(scenario.get("final_wait_ms", 1800))
        started_at = datetime.now()
        trigger_count = 0
        response_count = 0
        run_error: Optional[str] = None

        def emit(kind: str, **payload: Any) -> None:
            relative_seconds = round((datetime.now() - started_at).total_seconds(), 3)
            timeline_event = {
                "kind": kind,
                "relative_seconds": relative_seconds,
                **payload,
            }
            self._timeline.append(timeline_event)
            if on_event:
                on_event(timeline_event)

        def on_trigger(trigger: Trigger, context: MeetingContext):
            nonlocal trigger_count
            trigger_count += 1
            emit(
                "trigger",
                trigger_type=trigger.type.value,
                confidence=round(trigger.confidence, 3),
                reason=trigger.reason,
                relevant_transcript=trigger.relevant_transcript,
            )

        def on_response(response: AIResponse, trigger: Trigger):
            nonlocal response_count
            response_count += 1
            emit(
                "response",
                trigger_type=trigger.type.value,
                text=response.text,
            )

        await assistant.start()
        assistant.on_trigger_fired = on_trigger
        assistant.on_response_sent = on_response
        emit("system", status="started", meeting_id=meeting_id, description=description)

        try:
            for event in events:
                delay_ms = int(event.get("delay_ms", 0))
                if delay_ms > 0:
                    emit("wait", delay_ms=delay_ms)
                    await asyncio.sleep((delay_ms / 1000.0) / playback_speed)

                event_type = event.get("type")
                if event_type == "transcript":
                    chunk = TranscriptChunk(
                        text=str(event.get("text", "")),
                        timestamp=datetime.now(),
                        speaker=str(event.get("speaker") or "Unknown"),
                        is_final=bool(event.get("is_final", True)),
                        confidence=float(event.get("confidence", 1.0)),
                    )
                    emit(
                        "transcript",
                        speaker=chunk.speaker,
                        text=chunk.text,
                        is_final=chunk.is_final,
                    )
                    assistant._on_transcript_chunk(chunk)

                elif event_type == "chat":
                    speaker = str(event.get("speaker") or "Unknown")
                    text = str(event.get("text") or "")
                    emit("chat", speaker=speaker, text=text)
                    assistant._on_chat_message(speaker, text)

                elif event_type == "wait":
                    continue

            if final_wait_ms > 0:
                emit("wait", delay_ms=final_wait_ms, phase="final_settle")
                await asyncio.sleep((final_wait_ms / 1000.0) / playback_speed)

            emit("system", status="completed")

        except Exception as exc:
            run_error = str(exc)
            logger.error("scenario_simulation_failed", error=run_error)
            emit("system", status="failed", error=run_error)
        finally:
            await assistant.stop()

        duration_seconds = (datetime.now() - started_at).total_seconds()
        return SimulationResult(
            meeting_id=meeting_id,
            description=description,
            input_event_count=len(events),
            duration_seconds=duration_seconds,
            trigger_count=trigger_count,
            response_count=response_count,
            error=run_error,
        )