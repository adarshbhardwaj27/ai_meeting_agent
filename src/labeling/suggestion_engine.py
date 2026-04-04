"""Model-assisted prefill suggestions for labeling meeting windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List

from src.context import MeetingContext, Topic, TranscriptChunk, Trigger
from src.triggers.trigger_engine import TriggerEngine

from .ami_dataset import AmiWindow


@dataclass(frozen=True)
class LabelSuggestion:
    human_label: str
    trigger_type: str
    confidence: float
    reason: str
    source: str
    candidate_trigger_types: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "human_label": self.human_label,
            "trigger_type": self.trigger_type,
            "confidence": round(self.confidence, 6),
            "reason": self.reason,
            "source": self.source,
            "candidate_trigger_types": self.candidate_trigger_types,
        }


class WindowSuggestionEngine:
    """Replay the current trigger stack over a static AMI window.

    Suggestions are only a productivity aid. Human labels remain the source of truth.
    """

    _TRIGGER_PRIORITY = {
        "contradiction": 100,
        "factual_inconsistency": 95,
        "memory_gap": 90,
        "decision_forming": 80,
        "conflict_detected": 78,
        "risk_overlooked": 76,
        "repeated_confusion": 74,
        "collaboration_opportunity": 70,
        "topic_drift": 65,
        "circular_discussion": 60,
    }

    async def suggest_window(self, window: AmiWindow) -> LabelSuggestion:
        engine = TriggerEngine(enable_proactive=True)
        engine.bot_name = "__labeler__"
        engine._voice_mention_patterns = []

        context = self._build_context(window)
        collected_triggers: List[Trigger] = []

        for chunk in context.recent_transcript:
            chunk_context = self._context_up_to_chunk(context, chunk.chunk_id)
            triggers = await engine.check_for_triggers(context=chunk_context, new_chunks=[chunk])
            collected_triggers.extend(triggers)

        if collected_triggers:
            ranked = sorted(
                collected_triggers,
                key=lambda trigger: (
                    self._TRIGGER_PRIORITY.get(trigger.type.value, 0),
                    round(trigger.confidence, 6),
                ),
                reverse=True,
            )
            best = ranked[0]
            return LabelSuggestion(
                human_label="should_trigger",
                trigger_type=best.type.value,
                confidence=best.confidence,
                reason=best.reason,
                source="trigger_engine_replay",
                candidate_trigger_types=sorted({trigger.type.value for trigger in ranked}),
            )

        return LabelSuggestion(
            human_label="should_not_trigger",
            trigger_type="none",
            confidence=0.5,
            reason=(
                "No trigger fired when the current trigger engine replayed this window. "
                "Negative suggestions are weaker than positive ones and still need review."
            ),
            source="trigger_engine_replay",
            candidate_trigger_types=[],
        )

    @staticmethod
    def _build_context(window: AmiWindow) -> MeetingContext:
        now = datetime.now()
        window_start = window.start_time
        meeting_started_at = now - timedelta(seconds=max(1.0, window.end_time - window_start + 5.0))
        current_topic = Topic(name=window.topic, started_at=meeting_started_at) if window.topic else None

        chunks: List[TranscriptChunk] = []
        participants = set()
        open_questions: List[str] = []
        for turn in window.transcript_window:
            timestamp = meeting_started_at + timedelta(seconds=max(0.0, turn.end_time - window_start))
            speaker = f"Speaker {turn.speaker}"
            participants.add(speaker)
            if turn.text.strip().endswith("?"):
                open_questions.append(turn.text.strip())
            chunks.append(
                TranscriptChunk(
                    text=turn.text,
                    timestamp=timestamp,
                    speaker=speaker,
                    is_final=True,
                )
            )

        return MeetingContext(
            meeting_id=window.meeting_id,
            meeting_started_at=meeting_started_at,
            platform="offline_labeler",
            recent_transcript=chunks,
            current_topic=current_topic,
            topics_discussed=[current_topic] if current_topic else [],
            open_questions=open_questions[:4],
            participants=participants,
            active_speaker=chunks[-1].speaker if chunks else None,
            last_updated=chunks[-1].timestamp if chunks else now,
        )

    @staticmethod
    def _context_up_to_chunk(context: MeetingContext, chunk_id: str) -> MeetingContext:
        visible_chunks: List[TranscriptChunk] = []
        for chunk in context.recent_transcript:
            visible_chunks.append(chunk)
            if chunk.chunk_id == chunk_id:
                break

        visible_participants = {chunk.speaker for chunk in visible_chunks if chunk.speaker}
        visible_questions = [chunk.text.strip() for chunk in visible_chunks if chunk.text.strip().endswith("?")]

        return MeetingContext(
            meeting_id=context.meeting_id,
            meeting_started_at=context.meeting_started_at,
            platform=context.platform,
            recent_transcript=visible_chunks,
            historical_summary=context.historical_summary,
            current_topic=context.current_topic,
            topics_discussed=context.topics_discussed,
            concerns_raised=context.concerns_raised,
            decisions_confirmed=context.decisions_confirmed,
            decisions_pending=context.decisions_pending,
            open_questions=visible_questions[:4],
            action_items=context.action_items,
            participants=visible_participants,
            active_speaker=visible_chunks[-1].speaker if visible_chunks else context.active_speaker,
            last_updated=visible_chunks[-1].timestamp if visible_chunks else context.last_updated,
            tokens_in_window=context.tokens_in_window,
        )