from pathlib import Path

from src.labeling import LiveLogDataset
from src.meeting_logger import MeetingLogger


def test_live_log_dataset_builds_trigger_reviews_from_text_log(tmp_path: Path):
    log_path = tmp_path / "meeting_sessions.log"
    log_path.write_text(
        """
================================================================================
  SESSION START: 2026-04-04 10:00:00
  Meeting ID:    meeting_live_001
  Source:        live
================================================================================
[10:00:01.000] 🎤 [Alice]: We still do not understand the rollback order.
[10:00:02.000] 🎤 [Bob]: Same here, I am not following the sequence.
[10:00:03.250] ⚡ Trigger: repeated_confusion  speaker=Multiple confusion signals  text="not following the sequence"  conf=0.83
[10:00:03.600] 🤖 Sent (repeated_confusion): Want me to summarize the rollout steps?
[10:00:05.000] 🛑 Meeting assistant stopped
--------------------------------------------------------------------------------
  SESSION END: 2026-04-04 10:00:05  (duration: 5.0s)
--------------------------------------------------------------------------------
""".strip(),
        encoding="utf-8",
    )

    dataset = LiveLogDataset(log_path=log_path, event_log_path=tmp_path / "missing.jsonl")

    sessions = dataset.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].meeting_id == "meeting_live_001"
    assert sessions[0].trigger_count == 1

    reviews = dataset.build_reviews("meeting_live_001")
    assert len(reviews) == 1
    assert reviews[0].predicted_trigger_type == "repeated_confusion"
    assert reviews[0].response_text == "Want me to summarize the rollout steps?"
    assert len(reviews[0].transcript_window) == 2
    assert reviews[0].transcript_window[-1].text == "Same here, I am not following the sequence."
    assert reviews[0].trigger_anchor_index == 1
    assert reviews[0].trigger_anchor_time == "10:00:02.000"
    assert reviews[0].trigger_anchor_text == "Same here, I am not following the sequence."


def test_meeting_logger_writes_structured_event_log(tmp_path: Path):
    log_path = tmp_path / "meeting_sessions.log"
    logger = MeetingLogger(meeting_id="meeting_structured_001", source="live", log_file=str(log_path))

    logger.transcript("Alice", "Can someone explain the rollback order?", is_final=True)
    logger.trigger_fired(
        trigger_type="repeated_confusion",
        speaker="Multiple confusion signals",
        text="Can someone explain the rollback order?",
        confidence=0.77,
    )
    logger.bot_response("Here is the rollback order.", trigger_type="repeated_confusion")
    logger.close()

    event_log_path = tmp_path / "meeting_sessions_events.jsonl"
    assert event_log_path.exists()

    records = [line for line in event_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records
    joined = "\n".join(records)
    assert '"event_type": "session_start"' in joined
    assert '"event_type": "transcript"' in joined
    assert '"event_type": "trigger_fired"' in joined
    assert '"event_type": "bot_response"' in joined
    assert '"event_type": "session_end"' in joined