import pytest

from src.labeling.ami_dataset import AmiTurn, AmiWindow
from src.labeling.suggestion_engine import WindowSuggestionEngine


@pytest.mark.asyncio
async def test_suggestion_engine_prefills_confusion_window():
    engine = WindowSuggestionEngine()
    window = AmiWindow(
        meeting_id="demo",
        example_id="demo_0001",
        window_index=1,
        start_time=0.0,
        end_time=9.0,
        topic="rollout plan",
        transcript_window=[
            AmiTurn(meeting_id="demo", speaker="A", start_time=0.0, end_time=2.0, text="I am a bit lost on the rollback plan."),
            AmiTurn(meeting_id="demo", speaker="B", start_time=2.2, end_time=4.0, text="Same here, I am not following the sequence."),
            AmiTurn(meeting_id="demo", speaker="A", start_time=4.2, end_time=6.5, text="Can we slow down and explain it again?"),
            AmiTurn(meeting_id="demo", speaker="B", start_time=6.7, end_time=9.0, text="We are missing something basic here."),
        ],
        weak_signals={},
    )

    suggestion = await engine.suggest_window(window)

    assert suggestion.human_label == "should_trigger"
    assert suggestion.trigger_type == "repeated_confusion"


@pytest.mark.asyncio
async def test_suggestion_engine_can_prefill_negative_window():
    engine = WindowSuggestionEngine()
    window = AmiWindow(
        meeting_id="demo",
        example_id="demo_0002",
        window_index=2,
        start_time=0.0,
        end_time=8.0,
        topic="standup",
        transcript_window=[
            AmiTurn(meeting_id="demo", speaker="A", start_time=0.0, end_time=2.0, text="I fixed the payment bug and deployed it to staging."),
            AmiTurn(meeting_id="demo", speaker="B", start_time=2.3, end_time=4.0, text="Great, I will verify it after lunch."),
            AmiTurn(meeting_id="demo", speaker="A", start_time=4.2, end_time=6.0, text="I will also update the ticket."),
            AmiTurn(meeting_id="demo", speaker="B", start_time=6.2, end_time=8.0, text="Perfect, let us move to the next item."),
        ],
        weak_signals={},
    )

    suggestion = await engine.suggest_window(window)

    assert suggestion.human_label == "should_not_trigger"
    assert suggestion.trigger_type == "none"