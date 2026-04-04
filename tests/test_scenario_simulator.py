import pytest

from src.simulation import ScenarioSimulator, normalize_scenario_payload


def test_normalize_scenario_payload_from_simple_conversation():
    payload = {
        "meeting_id": "simple_upload",
        "conversation": [
            {"speaker": "John", "text": "The deadline is Friday."},
            {"speaker": "Sarah", "text": "The deadline is not Friday, it is Monday.", "delay_ms": 1500},
        ],
    }

    normalized = normalize_scenario_payload(payload, default_delay_ms=900)

    assert normalized["meeting_id"] == "simple_upload"
    assert normalized["events"][0]["type"] == "transcript"
    assert normalized["events"][0]["delay_ms"] == 0
    assert normalized["events"][1]["type"] == "transcript"
    assert normalized["events"][1]["delay_ms"] == 1500
    assert normalized["final_wait_ms"] >= 1800


def test_normalize_scenario_payload_filters_assert_events():
    payload = {
        "meeting_id": "e2e_style_upload",
        "events": [
            {"type": "transcript", "speaker": "John", "text": "hello", "delay_ms": 0},
            {"type": "assert", "condition": "trigger_fired", "trigger_type": "voice_address"},
            {"type": "wait", "delay_ms": 500},
        ],
    }

    normalized = normalize_scenario_payload(payload, default_delay_ms=700)

    assert len(normalized["events"]) == 2
    assert normalized["events"][0]["type"] == "transcript"
    assert normalized["events"][1]["type"] == "wait"


@pytest.mark.asyncio
async def test_scenario_simulator_replays_uploaded_chat_command():
    payload = {
        "meeting_id": "chat_command_upload",
        "conversation": [
            {"speaker": "Sarah", "text": "We decided to use microservices."},
            {"speaker": "Adarsh", "text": "@Alex summarize what we discussed", "type": "chat", "delay_ms": 0},
        ],
        "final_wait_ms": 1200,
    }
    normalized = normalize_scenario_payload(payload, default_delay_ms=0)
    captured_events = []

    result = await ScenarioSimulator().run(normalized, on_event=captured_events.append)

    assert result.error is None
    assert any(event["kind"] == "chat" for event in captured_events)
    assert any(
        event["kind"] == "trigger" and event["trigger_type"] == "explicit_command"
        for event in captured_events
    )
    assert any(event["kind"] == "response" for event in captured_events)