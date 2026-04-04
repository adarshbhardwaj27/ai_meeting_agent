"""Build review datasets from live meeting logs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional


_SESSION_START_RE = re.compile(r"^\s*SESSION START:\s*(?P<value>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*$")
_MEETING_ID_RE = re.compile(r"^\s*Meeting ID:\s*(?P<value>.+?)\s*$")
_SOURCE_RE = re.compile(r"^\s*Source:\s*(?P<value>.+?)\s*$")
_EVENT_LINE_RE = re.compile(r"^\[(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\]\s+(?P<message>.*)$")
_TRANSCRIPT_RE = re.compile(r"^(?P<label>[🎤💬]) \[(?P<speaker>[^\]]+)\]: (?P<text>.*)$")
_TRIGGER_RE = re.compile(
    r'^⚡ Trigger: (?P<trigger_type>[\w_]+)'
    r'(?:  speaker=(?P<speaker>.*?))?'
    r'(?:  text="(?P<text>.*?)")?'
    r'(?:  conf=(?P<confidence>\d+(?:\.\d+)?))?$'
)
_BOT_RESPONSE_RE = re.compile(r'^🤖 Sent(?: \((?P<trigger_type>[\w_]+)\))?: (?P<text>.*)$')


@dataclass(frozen=True)
class LoggedTranscriptTurn:
    speaker: str
    text: str
    is_final: bool
    start_time: float
    end_time: float
    wall_clock: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "is_final": self.is_final,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "wall_clock": self.wall_clock,
        }


@dataclass(frozen=True)
class LiveSessionSummary:
    meeting_id: str
    source: str
    started_at: str
    transcript_turn_count: int
    trigger_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meeting_id": self.meeting_id,
            "source": self.source,
            "started_at": self.started_at,
            "transcript_turn_count": self.transcript_turn_count,
            "trigger_count": self.trigger_count,
        }


@dataclass(frozen=True)
class LiveTriggerReviewExample:
    meeting_id: str
    example_id: str
    trigger_index: int
    event_time: str
    start_time: float
    end_time: float
    predicted_trigger_type: str
    predicted_confidence: float
    predicted_reason: str
    relevant_text: str
    response_text: str
    transcript_window: List[LoggedTranscriptTurn]
    trigger_anchor_index: Optional[int]
    trigger_anchor_time: Optional[str]
    trigger_anchor_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meeting_id": self.meeting_id,
            "example_id": self.example_id,
            "window_index": self.trigger_index,
            "event_time": self.event_time,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "trigger_anchor_index": self.trigger_anchor_index,
            "trigger_anchor_time": self.trigger_anchor_time,
            "trigger_anchor_text": self.trigger_anchor_text,
            "topic": None,
            "transcript_window": [turn.to_dict() for turn in self.transcript_window],
            "weak_signals": {
                "predicted_trigger_type": self.predicted_trigger_type,
                "predicted_confidence": self.predicted_confidence,
                "predicted_reason": self.predicted_reason,
                "relevant_text": self.relevant_text,
                "response_text": self.response_text,
            },
            "model_suggestion": {
                "human_label": "should_trigger",
                "trigger_type": self.predicted_trigger_type,
                "confidence": self.predicted_confidence,
                "reason": self.predicted_reason or self.relevant_text,
                "source": "live_session_log",
            },
        }


@dataclass
class _TriggerEvent:
    occurred_at: datetime
    offset_seconds: float
    trigger_type: str
    reason: str
    text: str
    confidence: float


@dataclass
class _BotResponseEvent:
    occurred_at: datetime
    offset_seconds: float
    trigger_type: str
    text: str


@dataclass
class _ParsedSession:
    meeting_id: str
    source: str
    started_at: datetime
    transcript_turns: List[LoggedTranscriptTurn] = field(default_factory=list)
    trigger_events: List[_TriggerEvent] = field(default_factory=list)
    bot_responses: List[_BotResponseEvent] = field(default_factory=list)


class LiveLogDataset:
    """Expose live meeting logs as reviewable trigger examples."""

    def __init__(
        self,
        log_path: str | Path,
        event_log_path: str | Path | None = None,
        transcript_context_turns: int = 8,
    ):
        self.log_path = Path(log_path)
        self.event_log_path = Path(event_log_path) if event_log_path else self.log_path.with_name("meeting_session_events.jsonl")
        self.transcript_context_turns = transcript_context_turns
        self._sessions: Optional[Dict[str, _ParsedSession]] = None

    def list_sessions(self) -> List[LiveSessionSummary]:
        sessions = self._load_sessions()
        summaries = [
            LiveSessionSummary(
                meeting_id=session.meeting_id,
                source=session.source,
                started_at=session.started_at.isoformat(),
                transcript_turn_count=len(session.transcript_turns),
                trigger_count=len(session.trigger_events),
            )
            for session in sessions.values()
        ]
        return sorted(summaries, key=lambda item: item.started_at, reverse=True)

    def build_reviews(self, meeting_id: str) -> List[LiveTriggerReviewExample]:
        session = self._load_sessions().get(meeting_id)
        if not session:
            return []

        reviews: List[LiveTriggerReviewExample] = []
        final_turns = [turn for turn in session.transcript_turns if turn.is_final]
        source_turns = final_turns or session.transcript_turns

        for index, trigger_event in enumerate(session.trigger_events):
            next_trigger_offset = (
                session.trigger_events[index + 1].offset_seconds
                if index + 1 < len(session.trigger_events)
                else float("inf")
            )
            candidate_turns = [
                turn for turn in source_turns
                if turn.end_time <= trigger_event.offset_seconds
            ]
            transcript_window = candidate_turns[-self.transcript_context_turns:]
            if not transcript_window:
                transcript_window = source_turns[:self.transcript_context_turns]

            trigger_anchor_index = self._find_trigger_anchor_index(transcript_window, trigger_event)
            trigger_anchor_turn = (
                transcript_window[trigger_anchor_index]
                if trigger_anchor_index is not None and 0 <= trigger_anchor_index < len(transcript_window)
                else None
            )

            response_event = next(
                (
                    response for response in session.bot_responses
                    if trigger_event.offset_seconds <= response.offset_seconds < next_trigger_offset
                ),
                None,
            )
            start_time = transcript_window[0].start_time if transcript_window else max(0.0, trigger_event.offset_seconds - 5.0)
            end_time = transcript_window[-1].end_time if transcript_window else trigger_event.offset_seconds
            reviews.append(
                LiveTriggerReviewExample(
                    meeting_id=meeting_id,
                    example_id=f"{meeting_id}__trigger_{index:03d}",
                    trigger_index=index,
                    event_time=trigger_event.occurred_at.isoformat(),
                    start_time=start_time,
                    end_time=end_time,
                    predicted_trigger_type=trigger_event.trigger_type,
                    predicted_confidence=trigger_event.confidence,
                    predicted_reason=trigger_event.reason,
                    relevant_text=trigger_event.text,
                    response_text=response_event.text if response_event else "",
                    transcript_window=transcript_window,
                    trigger_anchor_index=trigger_anchor_index,
                    trigger_anchor_time=trigger_anchor_turn.wall_clock if trigger_anchor_turn else None,
                    trigger_anchor_text=trigger_anchor_turn.text if trigger_anchor_turn else "",
                )
            )

        return reviews

    def _load_sessions(self) -> Dict[str, _ParsedSession]:
        if self._sessions is not None:
            return self._sessions

        sessions = self._load_sessions_from_text_log()
        structured_sessions = self._load_sessions_from_event_log()
        sessions.update(structured_sessions)
        self._sessions = sessions
        return self._sessions

    @classmethod
    def _find_trigger_anchor_index(
        cls,
        transcript_window: List[LoggedTranscriptTurn],
        trigger_event: _TriggerEvent,
    ) -> Optional[int]:
        if not transcript_window:
            return None

        normalized_excerpt = cls._normalize_text(trigger_event.text)
        if normalized_excerpt:
            for index in range(len(transcript_window) - 1, -1, -1):
                normalized_turn = cls._normalize_text(transcript_window[index].text)
                if not normalized_turn:
                    continue
                if normalized_excerpt in normalized_turn or normalized_turn in normalized_excerpt:
                    return index

        latest_before_trigger = None
        for index, turn in enumerate(transcript_window):
            if turn.end_time <= trigger_event.offset_seconds:
                latest_before_trigger = index

        return latest_before_trigger if latest_before_trigger is not None else len(transcript_window) - 1

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _load_sessions_from_event_log(self) -> Dict[str, _ParsedSession]:
        sessions: Dict[str, _ParsedSession] = {}
        if not self.event_log_path.exists():
            return sessions

        for raw_line in self.event_log_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            meeting_id = str(record.get("meeting_id") or "").strip()
            if not meeting_id:
                continue

            started_at_value = str(record.get("session_started_at") or record.get("started_at") or record.get("timestamp"))
            started_at = datetime.fromisoformat(started_at_value)
            session = sessions.setdefault(
                meeting_id,
                _ParsedSession(
                    meeting_id=meeting_id,
                    source=str(record.get("source") or "live"),
                    started_at=started_at,
                ),
            )
            occurred_at = datetime.fromisoformat(str(record.get("timestamp")))
            offset_seconds = max(0.0, (occurred_at - session.started_at).total_seconds())
            event_type = record.get("event_type")

            if event_type == "transcript":
                session.transcript_turns.append(
                    LoggedTranscriptTurn(
                        speaker=str(record.get("speaker") or "Unknown"),
                        text=str(record.get("text") or ""),
                        is_final=bool(record.get("is_final", True)),
                        start_time=offset_seconds,
                        end_time=offset_seconds,
                        wall_clock=occurred_at.strftime("%H:%M:%S.%f")[:-3],
                    )
                )
            elif event_type == "trigger_fired":
                session.trigger_events.append(
                    _TriggerEvent(
                        occurred_at=occurred_at,
                        offset_seconds=offset_seconds,
                        trigger_type=str(record.get("trigger_type") or "unknown"),
                        reason=str(record.get("speaker") or ""),
                        text=str(record.get("text") or ""),
                        confidence=float(record.get("confidence") or 0.0),
                    )
                )
            elif event_type == "bot_response":
                session.bot_responses.append(
                    _BotResponseEvent(
                        occurred_at=occurred_at,
                        offset_seconds=offset_seconds,
                        trigger_type=str(record.get("trigger_type") or ""),
                        text=str(record.get("text") or ""),
                    )
                )

        return sessions

    def _load_sessions_from_text_log(self) -> Dict[str, _ParsedSession]:
        sessions: Dict[str, _ParsedSession] = {}
        if not self.log_path.exists():
            return sessions

        current_started_at: Optional[datetime] = None
        current_meeting_id: Optional[str] = None
        current_source: Optional[str] = None
        current_session: Optional[_ParsedSession] = None

        for raw_line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            start_match = _SESSION_START_RE.match(line)
            if start_match:
                current_started_at = datetime.strptime(start_match.group("value"), "%Y-%m-%d %H:%M:%S")
                current_meeting_id = None
                current_source = None
                current_session = None
                continue

            if current_started_at and not current_meeting_id:
                meeting_match = _MEETING_ID_RE.match(line)
                if meeting_match:
                    current_meeting_id = meeting_match.group("value")
                continue

            if current_started_at and current_meeting_id and not current_source:
                source_match = _SOURCE_RE.match(line)
                if source_match:
                    current_source = source_match.group("value")
                    current_session = _ParsedSession(
                        meeting_id=current_meeting_id,
                        source=current_source,
                        started_at=current_started_at,
                    )
                    sessions[current_meeting_id] = current_session
                continue

            if not current_session:
                continue

            event_match = _EVENT_LINE_RE.match(line)
            if not event_match:
                continue

            occurred_at = datetime.combine(
                current_session.started_at.date(),
                datetime.strptime(event_match.group("time"), "%H:%M:%S.%f").time(),
            )
            if occurred_at < current_session.started_at:
                occurred_at = current_session.started_at
            offset_seconds = max(0.0, (occurred_at - current_session.started_at).total_seconds())
            message = event_match.group("message")

            transcript_match = _TRANSCRIPT_RE.match(message)
            if transcript_match:
                current_session.transcript_turns.append(
                    LoggedTranscriptTurn(
                        speaker=transcript_match.group("speaker"),
                        text=transcript_match.group("text"),
                        is_final=transcript_match.group("label") == "🎤",
                        start_time=offset_seconds,
                        end_time=offset_seconds,
                        wall_clock=event_match.group("time"),
                    )
                )
                continue

            trigger_match = _TRIGGER_RE.match(message)
            if trigger_match:
                current_session.trigger_events.append(
                    _TriggerEvent(
                        occurred_at=occurred_at,
                        offset_seconds=offset_seconds,
                        trigger_type=trigger_match.group("trigger_type"),
                        reason=(trigger_match.group("speaker") or "").strip(),
                        text=(trigger_match.group("text") or "").strip(),
                        confidence=float(trigger_match.group("confidence") or 0.0),
                    )
                )
                continue

            bot_match = _BOT_RESPONSE_RE.match(message)
            if bot_match:
                current_session.bot_responses.append(
                    _BotResponseEvent(
                        occurred_at=occurred_at,
                        offset_seconds=offset_seconds,
                        trigger_type=str(bot_match.group("trigger_type") or ""),
                        text=bot_match.group("text"),
                    )
                )

        return sessions