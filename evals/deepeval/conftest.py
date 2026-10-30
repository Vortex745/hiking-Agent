"""Honest aggregate status for the live DeepEval suite.

DeepEval 4.2.9's final banner can omit a case that errored in the judge and
still print a 100% pass rate. This hook reports the pytest-level counts against
the dataset size so a missing/errored case is visible as not evaluated.
"""

import json
from pathlib import Path


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    dataset = Path(__file__).with_name(".dataset.json")
    try:
        expected = len(json.loads(dataset.read_text(encoding="utf-8")))
    except Exception:
        expected = None
    stats = terminalreporter.stats
    passed = len(stats.get("passed", []))
    failed = len(stats.get("failed", []))
    errors = len(stats.get("error", []))
    terminalreporter.write_line(
        f"LIVE EVAL STATUS: expected={expected} passed={passed} "
        f"failed={failed} errors={errors}"
    )
    if expected is None or failed or errors or passed != expected:
        terminalreporter.write_line(
            "LIVE EVAL NOT CLEAN: not-evaluated cases exist; this run is not a pass."
        )
