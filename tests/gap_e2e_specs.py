"""
Intentional gap-spec E2E tests.

These scenarios describe behaviors the bot SHOULD support but does NOT yet.
They are intentionally kept outside the default pytest discovery flow by using
the file name ``gap_e2e_specs.py`` instead of ``test_*.py``.

Run explicitly when you want to surface missing capabilities:

    pytest tests/gap_e2e_specs.py -v --tb=short
"""

import json
from pathlib import Path

import pytest

from tests.e2e_runner import E2ERunner


SCENARIO_DIR = Path(__file__).parent / "e2e_gap_scenarios"


def _discover_gap_scenarios():
    scenarios = []
    for file_path in sorted(SCENARIO_DIR.glob("*.json")):
        with open(file_path) as fh:
            data = json.load(fh)
        meeting_id = data.get("meeting_id", file_path.stem)
        scenarios.append(pytest.param(str(file_path), id=meeting_id))
    return scenarios


class TestGapE2EScenarios:
    """Spec tests for missing behaviors that are expected to fail today."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario_path", _discover_gap_scenarios())
    async def test_gap_scenario(self, scenario_path):
        runner = E2ERunner(use_real_llm=False)
        result = await runner.run_scenario(scenario_path)

        failures = [
            f"  {assertion['description']}: {assertion.get('detail', '')}"
            for assertion in result.assertion_results
            if not assertion["passed"]
        ]

        if result.error:
            failures.append(f"  Runtime error: {result.error}")

        assert result.passed, (
            f"Gap spec '{result.meeting_id}' is still failing as expected until the feature is implemented:\n"
            + "\n".join(failures)
        )