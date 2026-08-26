"""Evaluate Chat v2 canary windows from frozen staging evidence.

Ticket 19: internal and 1% canaries.

This runner does not simulate production traffic. It evaluates the
machine-readable evidence produced by the internal and 1% staging
windows and fails closed when a required artifact is missing or when
hard gates are not met.

The script reuses the frozen corpus gate and the acceptance report from
ticket 18, then adds canary-specific checks:
- each window observed at least 24h
- P0/P1 stayed at zero
- 5xx rate stayed under 1%
- run-recovery failure stayed under 0.5%
- internal window includes backup/restore evidence
- internal window includes a force-v1 drill evidence record
- 1% window only counts after internal passed

Missing real evidence is recorded as skipped and makes overall_pass
False. That keeps the report usable in CI while still blocking rollout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from scripts.run_agent_v2_acceptance import (
    ACResult,
    AcceptanceReport,
    REQUIRED_CATEGORIES,
    load_corpus,
)


@dataclass(frozen=True)
class CanaryGateResult:
    gate_id: str
    status: Literal["passed", "failed", "skipped"]
    evidence: str
    reason: str = ""


@dataclass(frozen=True)
class CanaryWindowEvidence:
    cohort: str
    started_at: str
    ended_at: str
    window_hours: float
    p0_count: int
    p1_count: int
    five_xx_rate: float
    recovery_failure_rate: float
    acceptance_report: AcceptanceReport | None = None
    backup_restore: dict[str, Any] = field(default_factory=dict)
    rollback_drill: dict[str, Any] = field(default_factory=dict)
    evidence_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CanaryReport:
    timestamp: str
    corpus_scenario_count: int
    corpus_categories: list[str]
    windows: list[CanaryWindowEvidence]
    gates: list[CanaryGateResult]
    overall_pass: bool
    executed_commands: list[str] = field(default_factory=list)
    evidence_paths: list[str] = field(default_factory=list)
    residual_risks: list[str] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_acceptance_report(payload: dict[str, Any]) -> AcceptanceReport:
    ac_results = [
        ACResult(
            ac_id=str(item.get("ac_id", "")),
            title=str(item.get("title", "")),
            passed=bool(item.get("passed", False)),
            evidence=str(item.get("evidence", "")),
            skipped=bool(item.get("skipped", False)),
            skip_reason=str(item.get("skip_reason", "")),
        )
        for item in payload.get("ac_results", [])
        if isinstance(item, dict)
    ]
    return AcceptanceReport(
        timestamp=str(payload.get("timestamp", "")),
        environment=dict(payload.get("environment", {})),
        ac_results=ac_results,
        overall_pass=bool(payload.get("overall_pass", False)),
        executed_commands=[str(item) for item in payload.get("executed_commands", [])],
        evidence_paths=[str(item) for item in payload.get("evidence_paths", [])],
        residual_risks=[str(item) for item in payload.get("residual_risks", [])],
    )


def _parse_window_evidence(payload: dict[str, Any], *, default_cohort: str) -> CanaryWindowEvidence:
    acceptance_payload = payload.get("acceptance_report")
    acceptance_report = (
        _parse_acceptance_report(acceptance_payload)
        if isinstance(acceptance_payload, dict)
        else None
    )
    return CanaryWindowEvidence(
        cohort=str(payload.get("cohort", default_cohort)),
        started_at=str(payload.get("started_at", "")),
        ended_at=str(payload.get("ended_at", "")),
        window_hours=float(payload.get("window_hours", 0.0)),
        p0_count=int(payload.get("p0_count", 0)),
        p1_count=int(payload.get("p1_count", 0)),
        five_xx_rate=float(payload.get("five_xx_rate", 1.0)),
        recovery_failure_rate=float(payload.get("recovery_failure_rate", 1.0)),
        acceptance_report=acceptance_report,
        backup_restore=dict(payload.get("backup_restore", {})),
        rollback_drill=dict(payload.get("rollback_drill", {})),
        evidence_paths=[str(item) for item in payload.get("evidence_paths", [])],
    )


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _load_window(path: Path, default_cohort: str) -> CanaryWindowEvidence | None:
    payload = _load_json(path)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"invalid canary evidence payload: {path}")
    return _parse_window_evidence(payload, default_cohort=default_cohort)


def _evaluate_acceptance_report(report: AcceptanceReport | None) -> tuple[bool, str, str]:
    if report is None:
        return False, "acceptance report missing", "acceptance report missing"
    if report.environment.get("name") != "staging":
        return False, f"environment={report.environment.get('name', '')}", "environment is not staging"
    if not report.environment.get("versions"):
        return False, "versions missing", "dependency versions missing"
    if not report.executed_commands:
        return False, "executed_commands missing", "acceptance report missing commands"
    if any(ac.skipped for ac in report.ac_results):
        skipped = [ac.ac_id for ac in report.ac_results if ac.skipped]
        return False, f"skipped={skipped}", "acceptance report contains skipped ACs"
    if not report.overall_pass:
        failed = [ac.ac_id for ac in report.ac_results if not ac.passed]
        return False, f"failed={failed}", "acceptance report not overall_pass"
    if not all(ac.passed for ac in report.ac_results):
        failed = [ac.ac_id for ac in report.ac_results if not ac.passed]
        return False, f"failed={failed}", "acceptance report contains failed ACs"
    return True, "acceptance_report_ok", ""


def _evaluate_window(window: CanaryWindowEvidence) -> tuple[CanaryGateResult, list[str]]:
    reasons: list[str] = []
    if window.window_hours < 24:
        return (
            CanaryGateResult(
                gate_id=f"{window.cohort}_window",
                status="failed",
                evidence=f"window_hours={window.window_hours:.2f}",
                reason="window must observe at least 24h",
            ),
            ["window shorter than 24h"],
        )
    if window.p0_count != 0 or window.p1_count != 0:
        return (
            CanaryGateResult(
                gate_id=f"{window.cohort}_window",
                status="failed",
                evidence=f"p0={window.p0_count} p1={window.p1_count}",
                reason="P0/P1 must stay at zero during canary",
            ),
            ["P0/P1 incidents observed"],
        )
    if window.five_xx_rate >= 0.01:
        return (
            CanaryGateResult(
                gate_id=f"{window.cohort}_window",
                status="failed",
                evidence=f"five_xx_rate={window.five_xx_rate:.4f}",
                reason="5xx rate must stay below 1%",
            ),
            ["5xx rate exceeded 1%"],
        )
    if window.recovery_failure_rate >= 0.005:
        return (
            CanaryGateResult(
                gate_id=f"{window.cohort}_window",
                status="failed",
                evidence=f"recovery_failure_rate={window.recovery_failure_rate:.4f}",
                reason="run recovery failure must stay below 0.5%",
            ),
            ["run recovery failure exceeded 0.5%"],
        )

    acceptance_ok, evidence, reason = _evaluate_acceptance_report(window.acceptance_report)
    if not acceptance_ok:
        status = "skipped" if "skipped ACs" in reason or "missing" in reason else "failed"
        return (
            CanaryGateResult(
                gate_id=f"{window.cohort}_window",
                status=status,
                evidence=evidence,
                reason=reason,
            ),
            [reason],
        )

    return (
        CanaryGateResult(
            gate_id=f"{window.cohort}_window",
            status="passed",
            evidence=(
                f"hours={window.window_hours:.2f} p0={window.p0_count} p1={window.p1_count} "
                f"five_xx_rate={window.five_xx_rate:.4f} recovery_failure_rate={window.recovery_failure_rate:.4f}"
            ),
        ),
        reasons,
    )


def _evaluate_backup_restore(window: CanaryWindowEvidence) -> CanaryGateResult:
    backup = window.backup_restore or {}
    paths = [str(item) for item in backup.get("backup_paths", [])]
    manifest_path = str(backup.get("manifest_path", ""))
    if not paths or len(paths) != 3 or not manifest_path:
        return CanaryGateResult(
            gate_id="backup_restore",
            status="skipped" if not paths and not manifest_path else "failed",
            evidence=f"backup_paths={len(paths)} manifest={'set' if manifest_path else 'missing'}",
            reason="three SQLite backups plus a manifest are required",
        )
    if not bool(backup.get("restore_verified", False)):
        return CanaryGateResult(
            gate_id="backup_restore",
            status="failed",
            evidence="restore_verified=false",
            reason="backup restore was not verified",
        )
    if not bool(backup.get("manifest_readable", False)):
        return CanaryGateResult(
            gate_id="backup_restore",
            status="failed",
            evidence="manifest_readable=false",
            reason="backup manifest is not readable",
        )
    for path_text in paths:
        if not Path(path_text).exists():
            return CanaryGateResult(
                gate_id="backup_restore",
                status="failed",
                evidence=f"missing_backup={path_text}",
                reason="backup file missing",
            )
        try:
            conn = sqlite3.connect(path_text)
            try:
                conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
            finally:
                conn.close()
        except Exception as exc:
            return CanaryGateResult(
                gate_id="backup_restore",
                status="failed",
                evidence=f"unreadable_backup={path_text}",
                reason=str(exc),
            )

    try:
        json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return CanaryGateResult(
            gate_id="backup_restore",
            status="failed",
            evidence=f"manifest_path={manifest_path}",
            reason=f"manifest unreadable: {exc}",
        )

    return CanaryGateResult(
        gate_id="backup_restore",
        status="passed",
        evidence=f"backup_paths=3 manifest={manifest_path}",
    )


def _evaluate_rollback_drill(window: CanaryWindowEvidence) -> CanaryGateResult:
    drill = window.rollback_drill or {}
    required_flags = (
        "force_v1_applied",
        "new_requests_routed_to_v1",
        "paused_v2_run_preserved",
        "effectful_v2_run_preserved",
        "restored_internal_v2",
    )
    if not drill:
        return CanaryGateResult(
            gate_id="force_v1_drill",
            status="skipped",
            evidence="rollback drill missing",
            reason="force-v1 drill evidence missing",
        )
    missing = [flag for flag in required_flags if not bool(drill.get(flag, False))]
    if missing:
        return CanaryGateResult(
            gate_id="force_v1_drill",
            status="failed",
            evidence=f"missing_flags={missing}",
            reason="force-v1 drill did not prove rollback safety",
        )
    return CanaryGateResult(
        gate_id="force_v1_drill",
        status="passed",
        evidence="force-v1 preserved paused/effectful runs and routed new traffic to v1",
    )


def evaluate_canary(
    *,
    corpus_path: Path,
    internal_evidence_path: Path,
    one_percent_evidence_path: Path,
    operator: str = "canary-acceptance",
    notes: list[str] | None = None,
) -> CanaryReport:
    timestamp = _now_iso()
    residual_risks = list(notes or [])
    executed_commands = [
        subprocess.list2cmdline(
            [sys.executable, "-m", "scripts.run_agent_v2_canary", "--operator", operator]
        )
    ]
    evidence_paths = [str(corpus_path)]

    scenarios = load_corpus(corpus_path)
    categories = sorted({scenario.category for scenario in scenarios})
    corpus_ok = len(scenarios) >= 50 and REQUIRED_CATEGORIES <= set(categories)
    gates: list[CanaryGateResult] = [
        CanaryGateResult(
            gate_id="corpus",
            status="passed" if corpus_ok else "failed",
            evidence=f"scenarios={len(scenarios)} categories={categories}",
            reason="frozen corpus must contain at least 50 scenarios and every required category",
        )
    ]
    if not corpus_ok:
        residual_risks.append("corpus gate failed: frozen corpus incomplete")

    internal_window = _load_window(internal_evidence_path, "internal")
    if internal_window is None:
        internal_window = CanaryWindowEvidence(
            cohort="internal",
            started_at="",
            ended_at="",
            window_hours=0.0,
            p0_count=0,
            p1_count=0,
            five_xx_rate=1.0,
            recovery_failure_rate=1.0,
            evidence_paths=[str(internal_evidence_path)],
        )
        gates.append(
            CanaryGateResult(
                gate_id="internal_window",
                status="skipped",
                evidence="",
                reason="internal evidence missing",
            )
        )
        residual_risks.append("internal canary evidence missing")
    else:
        evidence_paths.extend([str(internal_evidence_path), *internal_window.evidence_paths])
        internal_gate, internal_risks = _evaluate_window(internal_window)
        gates.append(internal_gate)
        residual_risks.extend(internal_risks)
        if internal_gate.status == "passed":
            backup_gate = _evaluate_backup_restore(internal_window)
            gates.append(backup_gate)
            if backup_gate.status != "passed" and backup_gate.reason:
                residual_risks.append(f"backup_restore: {backup_gate.reason}")
            drill_gate = _evaluate_rollback_drill(internal_window)
            gates.append(drill_gate)
            if drill_gate.status != "passed" and drill_gate.reason:
                residual_risks.append(f"force_v1_drill: {drill_gate.reason}")
        else:
            gates.append(
                CanaryGateResult(
                    gate_id="backup_restore",
                    status="skipped",
                    evidence="",
                    reason="internal window did not pass",
                )
            )
            gates.append(
                CanaryGateResult(
                    gate_id="force_v1_drill",
                    status="skipped",
                    evidence="",
                    reason="internal window did not pass",
                )
            )

    one_percent_window = _load_window(one_percent_evidence_path, "1%")
    if internal_window and internal_window.acceptance_report and any(
        gate.status == "passed" for gate in gates if gate.gate_id == "internal_window"
    ):
        if one_percent_window is None:
            one_percent_window = CanaryWindowEvidence(
                cohort="1%",
                started_at="",
                ended_at="",
                window_hours=0.0,
                p0_count=0,
                p1_count=0,
                five_xx_rate=1.0,
                recovery_failure_rate=1.0,
                evidence_paths=[str(one_percent_evidence_path)],
            )
            gates.append(
                CanaryGateResult(
                    gate_id="one_percent_window",
                    status="skipped",
                    evidence="",
                    reason="1% evidence missing",
                )
            )
            residual_risks.append("1% canary evidence missing")
        else:
            evidence_paths.extend([str(one_percent_evidence_path), *one_percent_window.evidence_paths])
            one_gate, one_risks = _evaluate_window(one_percent_window)
            gates.append(one_gate)
            residual_risks.extend(one_risks)
    else:
        if one_percent_window is None:
            one_percent_window = CanaryWindowEvidence(
                cohort="1%",
                started_at="",
                ended_at="",
                window_hours=0.0,
                p0_count=0,
                p1_count=0,
                five_xx_rate=1.0,
                recovery_failure_rate=1.0,
                evidence_paths=[str(one_percent_evidence_path)],
            )
        gates.append(
            CanaryGateResult(
                gate_id="one_percent_window",
                status="skipped",
                evidence="",
                reason="internal window did not pass",
            )
        )
        residual_risks.append("1% canary was not allowed because internal window did not pass")

    internal_gate = next(gate for gate in gates if gate.gate_id == "internal_window")
    one_gate = next(gate for gate in gates if gate.gate_id == "one_percent_window")
    promotion_ok = (
        internal_gate.status == "passed"
        and one_gate.status == "passed"
        and _parse_iso(internal_window.ended_at) is not None
        and _parse_iso(one_percent_window.started_at) is not None
        and _parse_iso(one_percent_window.started_at) >= _parse_iso(internal_window.ended_at)
    )
    promotion_reason = (
        "1% cohort must start after internal cohort passes and finishes"
        if not promotion_ok
        else ""
    )
    gates.append(
        CanaryGateResult(
            gate_id="promotion",
            status="passed" if promotion_ok else "failed",
            evidence=(
                f"internal_end={internal_window.ended_at} "
                f"one_percent_start={one_percent_window.started_at}"
            ),
            reason=promotion_reason,
        )
    )
    if not promotion_ok:
        residual_risks.append("1% cohort promotion gate failed")

    reporting_ok = bool(executed_commands) and bool(evidence_paths)
    gates.append(
        CanaryGateResult(
            gate_id="reporting",
            status="passed" if reporting_ok else "failed",
            evidence=f"commands={len(executed_commands)} evidence_paths={len(evidence_paths)}",
            reason="report must record commands and evidence paths",
        )
    )

    overall_pass = all(gate.status == "passed" for gate in gates)
    if not overall_pass:
        failed = [gate.gate_id for gate in gates if gate.status == "failed"]
        skipped = [gate.gate_id for gate in gates if gate.status == "skipped"]
        if failed:
            residual_risks.append(f"failed gates: {', '.join(failed)}")
        if skipped:
            residual_risks.append(f"skipped gates: {', '.join(skipped)}")

    return CanaryReport(
        timestamp=timestamp,
        corpus_scenario_count=len(scenarios),
        corpus_categories=categories,
        windows=[internal_window, one_percent_window],
        gates=gates,
        overall_pass=overall_pass,
        executed_commands=executed_commands,
        evidence_paths=evidence_paths,
        residual_risks=residual_risks,
    )


def main(argv: list[str] | None = None) -> int:
    ai_service_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate Chat v2 canary windows")
    parser.add_argument(
        "--operator",
        default="canary-acceptance",
        help="operator label written into the executed command record",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ai_service_dir / "tests" / "fixtures" / "agent_v2_cases.jsonl",
    )
    parser.add_argument(
        "--internal-evidence",
        type=Path,
        default=ai_service_dir / "data" / "missing-internal-canary-evidence.json",
        help="internal cohort evidence JSON",
    )
    parser.add_argument(
        "--one-percent-evidence",
        type=Path,
        default=ai_service_dir / "data" / "missing-one-percent-canary-evidence.json",
        help="1% cohort evidence JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ai_service_dir / "data" / "agent-v2-canary-report.json",
    )
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="additional residual risk note",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    report = evaluate_canary(
        corpus_path=args.corpus,
        internal_evidence_path=args.internal_evidence,
        one_percent_evidence_path=args.one_percent_evidence,
        operator=args.operator,
        notes=args.note,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Chat v2 canary report written to: {args.output}")
    print(f"overall_pass: {report.overall_pass}")
    for gate in report.gates:
        print(f"  {gate.gate_id}: {gate.status}")
    if report.residual_risks:
        print("residual_risks:")
        for risk in report.residual_risks:
            print(f"  - {risk}")
    return 0 if report.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
