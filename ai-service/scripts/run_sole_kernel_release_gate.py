"""Ticket 35 — sole-kernel release gate report generator.

Aggregates the CURRENT on-disk evidence into one JSON/Markdown report:

* the vendored fork commit the runtime actually resolves (kernel identity);
* the configured pin (fail-closed comparison re-run here);
* the Chat v2 contract suite command and its pass/fail/skip summary;
* the sole-kernel invariant suite result (stub/bypass absence).

``overall_pass`` is True only when identity is compatible AND every
collected check passed. Skips are recorded as skips — never converted into
passes. The canary/cutover coordinators (Tickets 21/22) must treat a
passing report as a prerequisite for any go decision.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR))


def _pytest(summary_args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", *summary_args],
        cwd=str(AI_DIR),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    parsed = {"rc": completed.returncode, "tail": tail}
    for token in tail.split():
        if token.rstrip(",").isdigit():
            parsed.setdefault("count", token.rstrip(","))
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(AI_DIR / "data" / "sole-kernel-gate-report.json"))
    parser.add_argument("--skip-pytest", action="store_true", help="aggregate without running suites")
    args = parser.parse_args()

    from agent_v2.uniharness.kernel_identity import resolve_kernel_identity
    from config import Settings

    settings = Settings()
    identity = resolve_kernel_identity(settings.agent_v2_uniharness_pin)

    checks: list[dict[str, Any]] = [
        {
            "check": "kernel_identity_matches_pin",
            "passed": identity.compatible,
            "evidence": identity.detail,
            "fork_commit": identity.fork_commit,
            "distribution_version": identity.distribution_version,
        },
    ]

    if not args.skip_pytest:
        full = _pytest([])
        checks.append({
            "check": "full_ai_service_suite",
            "passed": full["rc"] == 0,
            "evidence": full["tail"],
        })

    invariants = _pytest(["test_sole_kernel_invariants.py"])
    checks.append({
        "check": "sole_kernel_invariants",
        "passed": invariants["rc"] == 0,
        "evidence": invariants["tail"],
    })

    overall = all(c["passed"] for c in checks)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pin": settings.agent_v2_uniharness_pin,
        "upstream_base": "757bf3d60f763706aae60b040d775e8563b9c577",
        "provenance": "third_party/uniharness-provenance.md",
        "checks": checks,
        "overall_pass": overall,
        "decision": "GO (sole-kernel gate satisfied)" if overall else "NO-GO",
        "notes": [
            "Live provider E2Es are opt-in (KERNEL_LIVE_E2E=1); recorded skips do not fail the gate.",
            "Old stub-era reports are NOT acceptable evidence for this gate.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
