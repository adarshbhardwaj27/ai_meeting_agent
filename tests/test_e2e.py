"""
Pytest wrapper for E2E scenario tests.

Discovers all JSON files in tests/e2e_scenarios/ and runs each
as a separate parametrized test case.

Usage:
    pytest tests/test_e2e.py -v                      # all scenarios (mock LLM)
    pytest tests/test_e2e.py -v -k "voice"           # only voice scenarios
    pytest tests/test_e2e.py -v -k "contradiction"   # only intelligence tests
    pytest tests/test_e2e.py -v --real-llm            # use real LLM ($$$)
"""

import asyncio
import json
import pytest
from pathlib import Path

from tests.e2e_runner import E2ERunner


# ── Discover scenarios ───────────────────────────────────────────────

SCENARIO_DIR = Path(__file__).parent / "e2e_scenarios"


def _discover_scenarios():
    """Find all JSON scenario files and return (id, path) pairs."""
    if not SCENARIO_DIR.exists():
        return []

    scenarios = []
    for f in sorted(SCENARIO_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        meeting_id = data.get("meeting_id", f.stem)
        scenarios.append(pytest.param(str(f), id=meeting_id))
    return scenarios


# ── Test class ───────────────────────────────────────────────────────

class TestE2EScenarios:
    """End-to-end tests driven by JSON scenario files."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario_path", _discover_scenarios())
    async def test_scenario(self, scenario_path):
        """Run a single E2E scenario and assert all checks pass."""
        runner = E2ERunner(use_real_llm=False)
        result = await runner.run_scenario(scenario_path)

        # Collect failure details for a clear error message
        failures = [
            f"  {a['description']}: {a.get('detail', '')}"
            for a in result.assertion_results
            if not a["passed"]
        ]

        if result.error:
            failures.append(f"  Runtime error: {result.error}")

        assert result.passed, (
            f"E2E scenario '{result.meeting_id}' failed:\n"
            + "\n".join(failures)
        )
