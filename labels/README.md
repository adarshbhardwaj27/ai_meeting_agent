# E2E Trigger Label Seeds

This folder is a starter set for labeling trigger-model examples from deterministic E2E runs.

Included files:

- `e2e_seed_labels.jsonl` contains real examples captured from sampled E2E scenarios run on 2026-04-04.
- `past_session_seed_labels.jsonl` contains hand-labeled examples extracted from reviewed session logs, including false fires and missed triggers from real conversation excerpts.

Use these label values:

- `should_trigger`: the assistant should intervene and the observed trigger type is correct.
- `should_not_trigger`: the assistant should stay silent.
- `wrong_trigger_type`: the assistant fired, but the trigger type was wrong.
- `missed_trigger`: the assistant should have fired, but no trigger fired.

Recommended row shape:

```json
{
  "source": "e2e",
  "scenario_path": "tests/e2e_scenarios/51_confused_engineers_fast_convo.json",
  "meeting_id": "confused_engineers_fast_convo",
  "label": "should_trigger",
  "correct_trigger_type": "repeated_confusion",
  "observed_trigger_types": ["repeated_confusion"],
  "observed_response_sent": true,
  "transcript_window": ["Speaker: text"],
  "notes": "Short human note"
}
```

How to label the two categories that did not appear in the sampled run:

- `wrong_trigger_type`: set `label` to `wrong_trigger_type`, keep the actual fired trigger in `observed_trigger_types`, and set `correct_trigger_type` to what a human reviewer thinks it should have been.
- `missed_trigger`: set `label` to `missed_trigger`, leave `observed_trigger_types` empty, set `correct_trigger_type` to the expected trigger, and explain why the intervention was warranted.

Useful commands:

```powershell
c:/Users/Adarsh.Bhardwaj/codes/meet_ai_agent/venv/Scripts/python.exe -m tests.e2e_runner --scenario tests/e2e_scenarios/51_confused_engineers_fast_convo.json
c:/Users/Adarsh.Bhardwaj/codes/meet_ai_agent/venv/Scripts/python.exe -m pytest tests/gap_e2e_specs.py -q --tb=short
```

Current note:

- The sampled run showed that every current file under `tests/e2e_gap_scenarios/` passes, so that lane is stale right now. In particular, `54_repeated_confusion_across_turns.json` is no longer a real gap and is a candidate to move into the main E2E suite.

AMI XML labeling workflow:

- Run `c:/Users/Adarsh.Bhardwaj/codes/meet_ai_agent/venv/Scripts/python.exe tools/ami_labeling_app.py`
- Open `http://127.0.0.1:8767`
- Pick a meeting, review each window, and click `Should Trigger`, `Should Not Trigger`, or `Skip / Unsure`
- The labeler now pre-fills each window with the current trigger engine's best suggestion; you can accept it or override it before saving
- Saved outputs are written to `labels/ami_manual_labels.json`, `labels/ami_manual_labels.jsonl`, and `labels/ami_manual_labels.csv`
- The CSV file opens cleanly in Excel if you want spreadsheet review after labeling in the browser

Past meeting trigger review workflow:

- Run your real meeting assistant as usual; after the meeting ends it appends that finished session to `logs/meeting_sessions.log`
- New runs now also append structured event records to `logs/meeting_session_events.jsonl` so completed sessions are easier to review later
- Launch `c:/Users/Adarsh.Bhardwaj/codes/meet_ai_agent/venv/Scripts/python.exe tools/live_log_review_app.py`
- Open `http://127.0.0.1:8768`
- Pick a past session, review each fired trigger with the surrounding transcript, and either keep it, correct the trigger type, or mark that it should not have fired
- Saved outputs are written to `labels/live_trigger_reviews.json`, `labels/live_trigger_reviews.jsonl`, and `labels/live_trigger_reviews.csv`
- This gives you a real-life correction dataset built from actual meetings instead of only AMI or synthetic E2E fixtures