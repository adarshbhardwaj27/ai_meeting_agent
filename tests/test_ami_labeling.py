from pathlib import Path

from src.labeling import AmiDataset


DATASET_ROOT = Path(__file__).resolve().parents[1] / "docs" / "ami_public_manual_1.6.2"


def test_ami_dataset_lists_meetings():
    dataset = AmiDataset(DATASET_ROOT)

    meetings = dataset.list_meetings()

    assert "ES2002a" in meetings


def test_ami_dataset_builds_readable_windows():
    dataset = AmiDataset(DATASET_ROOT)

    windows = dataset.build_windows("ES2002a", window_size=5, step=2)

    assert windows
    first_window = windows[0]
    assert first_window.meeting_id == "ES2002a"
    assert first_window.transcript_window
    assert any(turn.text for turn in first_window.transcript_window)
    assert all(turn.speaker for turn in first_window.transcript_window)
    assert isinstance(first_window.weak_signals["dialogue_act_counts"], dict)


def test_ami_dataset_merges_short_same_speaker_continuations():
    dataset = AmiDataset(DATASET_ROOT)

    windows = dataset.build_windows("EN2001a", window_size=6, step=3)

    assert windows
    first_window = windows[0]
    rendered_turns = [turn.text for turn in first_window.transcript_window]
    assert any("he said yesterday?" in text for text in rendered_turns)