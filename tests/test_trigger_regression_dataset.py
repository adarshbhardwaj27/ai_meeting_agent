import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.context.models import Concern, ConcernSeverity, MeetingContext, Topic, TranscriptChunk, TriggerType
from src.triggers.trigger_engine import TriggerEngine


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "labels"
    / "log_batch_discrepancies_20260404_16505_end.jsonl"
)


def _load_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


DATASET_ROWS = _load_rows()
TOPIC_DRIFT_ROWS = [
    row for row in DATASET_ROWS if "topic_drift" in row.get("observed_trigger_types", [])
]
CONTRADICTION_ROWS = [
    row for row in DATASET_ROWS if "contradiction" in row.get("observed_trigger_types", [])
]
COLLABORATION_ROWS = [
    row
    for row in DATASET_ROWS
    if "collaboration_opportunity" in row.get("observed_trigger_types", [])
]
DECISION_ROWS = [
    row for row in DATASET_ROWS if row.get("correct_trigger_type") == "decision_forming"
]


def _row_id(row: dict) -> str:
    return f"{row['meeting_id']}-{row['event_time']}"


def _parse_transcript_line(line: str) -> tuple[str, str]:
    if ":" not in line:
        return "Unknown", line.strip()
    speaker, text = line.split(":", 1)
    return speaker.strip(), text.strip()


def _window_to_chunks(window: list[str]) -> list[TranscriptChunk]:
    now = datetime.now()
    base_time = now - timedelta(seconds=max(len(window), 1))
    chunks: list[TranscriptChunk] = []

    for index, line in enumerate(window):
        speaker, text = _parse_transcript_line(line)
        chunks.append(
            TranscriptChunk(
                text=text,
                timestamp=base_time + timedelta(seconds=index),
                speaker=speaker,
                is_final=True,
            )
        )

    return chunks


def _make_context(meeting_id: str) -> MeetingContext:
    return MeetingContext(
        meeting_id=meeting_id,
        meeting_started_at=datetime.now() - timedelta(minutes=12),
        platform="test",
    )


def _seed_topic(context: MeetingContext, chunks: list[TranscriptChunk]) -> None:
    topic_seed = " ".join(chunk.text for chunk in chunks[:2]).strip()
    context.current_topic = Topic(
        name=(topic_seed[:120] or "Active discussion"),
        started_at=datetime.now() - timedelta(minutes=4),
    )


class TestTriggerRegressionDataset:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("row", TOPIC_DRIFT_ROWS, ids=_row_id)
    async def test_reviewed_topic_drift_rows_do_not_refire(self, row):
        engine = TriggerEngine()
        chunks = _window_to_chunks(row["transcript_window"])
        context = _make_context(f"topic-{row['meeting_id']}")
        _seed_topic(context, chunks)
        context.recent_transcript.extend(chunks)

        triggers = await engine.check_for_triggers(
            context=context,
            new_chunks=[chunks[-1]],
        )

        assert TriggerType.TOPIC_DRIFT not in {trigger.type for trigger in triggers}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("row", CONTRADICTION_ROWS, ids=_row_id)
    async def test_reviewed_contradiction_rows_do_not_refire(self, row):
        engine = TriggerEngine()
        chunks = _window_to_chunks(row["transcript_window"])
        context = _make_context(f"contradiction-{row['meeting_id']}")
        _seed_topic(context, chunks)

        for chunk in chunks[:-1]:
            context.recent_transcript.append(chunk)
            await engine.check_for_triggers(context=context, new_chunks=[chunk])

        context.recent_transcript.append(chunks[-1])
        triggers = await engine.check_for_triggers(
            context=context,
            new_chunks=[chunks[-1]],
        )

        assert TriggerType.CONTRADICTION not in {trigger.type for trigger in triggers}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("row", COLLABORATION_ROWS, ids=_row_id)
    async def test_reviewed_collaboration_rows_do_not_refire(self, row):
        engine = TriggerEngine()
        chunks = _window_to_chunks(row["transcript_window"])
        context = _make_context(f"collab-{row['meeting_id']}")
        _seed_topic(context, chunks)

        concern = Concern.create(
            description=chunks[0].text,
            raised_by=chunks[0].speaker,
            severity=ConcernSeverity.HIGH if "blocking" in chunks[0].text.lower() else ConcernSeverity.MEDIUM,
        )
        concern.raised_at = datetime.now() - timedelta(minutes=6)
        context.concerns_raised.append(concern)
        context.recent_transcript.extend(chunks)

        triggers = await engine.check_for_triggers(
            context=context,
            new_chunks=chunks,
        )

        assert TriggerType.COLLABORATION_OPPORTUNITY not in {trigger.type for trigger in triggers}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("row", DECISION_ROWS, ids=_row_id)
    async def test_reviewed_decision_rows_surface_decision_forming(self, row):
        engine = TriggerEngine()
        chunks = _window_to_chunks(row["transcript_window"])
        context = _make_context(f"decision-{row['meeting_id']}")
        _seed_topic(context, chunks)
        context.recent_transcript.extend(chunks)

        triggers = await engine.check_for_triggers(
            context=context,
            new_chunks=chunks,
        )
        trigger_types = {trigger.type for trigger in triggers}

        assert TriggerType.DECISION_FORMING in trigger_types
        assert TriggerType.TOPIC_DRIFT not in trigger_types
        assert TriggerType.CONTRADICTION not in trigger_types