"""Ticket 19 — internal and 1% Chat v2 canary coordinator.

This script does not run a live rollout by itself. It stitches together
the evidence that an operator collects during the internal and 1%
windows:

* backup and restore smoke tests for the three v2 SQLite databases
* artifact manifest export and verification
* hard-gate reports for each 24-hour canary window
* explicit force-v1 rollback drill evidence

The goal is to keep the implementation narrow: backup/restore is
validated locally, while the actual 24-hour window observations remain
operator-supplied evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.repositories.memory import MemoryStore
from config import settings
from scripts.run_agent_v2_acceptance import evaluate_acceptance


REQUIRED_SQLITE_DB_SPECS: dict[str, tuple[str, ...]] = {
    "agent_control.sqlite": (
        "runs",
        "cancel_requests",
        "writer_lock",
        "sandbox_leases",
        "artifacts",
        "approvals",
        "effects",
        "run_telemetry",
        "raw_sampling_audit",
    ),
    "chat_checkpoints.sqlite": (
        "conversations",
        "messages",
        "run_events",
    ),
    "agent_memory.sqlite": (
        "subjects",
        "memory_settings",
        "conversation_owners",
        "memories",
        "memory_candidates",
    ),
}

REQUIRED_WINDOW_SECTIONS = (
    "quality",
    "performance",
    "cost",
    "dependency",
    "capacity",
    "privacy",
    "security",
    "recovery",
    "user_visible",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sqlite_integrity_check(path: Path) -> str:
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        if row is None:
            raise RuntimeError(f"integrity_check returned no rows for {path}")
        return str(row[0])
    finally:
        conn.close()


def _sqlite_table_counts(path: Path, tables: tuple[str, ...]) -> dict[str, int]:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        existing = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        missing = [table for table in tables if table not in existing]
        if missing:
            raise RuntimeError(f"{path.name} missing tables: {missing}")
        counts: dict[str, int] = {}
        for table in tables:
            row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
            counts[table] = int(row["count"]) if row is not None else 0
        return counts
    finally:
        conn.close()


def _backup_sqlite_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_conn = sqlite3.connect(str(source))
    target_conn = sqlite3.connect(str(target))
    try:
        source_conn.backup(target_conn)
    finally:
        target_conn.close()
        source_conn.close()


def _copy_tree(source: Path, target: Path) -> None:
    if not source.exists():
        target.mkdir(parents=True, exist_ok=True)
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


@dataclass(frozen=True, slots=True)
class SQLiteBackupCheck:
    database_name: str
    source_path: str
    backup_path: str
    restore_path: str
    source_counts: dict[str, int]
    backup_counts: dict[str, int]
    restore_counts: dict[str, int]
    source_integrity: str
    backup_integrity: str
    restore_integrity: str
    passed: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ArtifactManifestEntry:
    artifact_id: str
    relative_storage_path: str
    public_filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    status: str
    failure_reason: str | None
    file_present: bool
    file_sha256: str | None


@dataclass(frozen=True, slots=True)
class BackupReport:
    timestamp: str
    state_dir: str
    backup_dir: str
    restore_dir: str
    sqlite_checks: list[SQLiteBackupCheck]
    artifact_manifest_path: str
    artifact_entries: list[ArtifactManifestEntry]
    overall_pass: bool
    residual_risks: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CanaryWindowReport:
    cohort: str
    window_hours: float
    hard_gate_evidence_path: str
    hard_gate_report_path: str
    hard_gate_pass: bool
    rollback_drill_path: str
    rollback_pass: bool
    incident_counts: dict[str, int]
    five_xx_rate: float
    run_recovery_failure_rate: float
    section_statuses: dict[str, str]
    status: Literal["passed", "failed", "skipped"]
    residual_risks: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CanaryRolloutReport:
    timestamp: str
    backup: BackupReport
    windows: list[CanaryWindowReport]
    overall_pass: bool
    executed_commands: list[str] = field(default_factory=list)
    evidence_paths: list[str] = field(default_factory=list)
    residual_risks: list[str] = field(default_factory=list)


def _collect_artifact_rows(control_db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(control_db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT artifact_id, run_id, public_filename, mime_type, size_bytes, "
            "sha256, storage_path, status, failure_reason, created_at, expires_at "
            "FROM artifacts ORDER BY created_at, artifact_id"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _build_artifact_manifest(
    *,
    state_dir: Path,
    backup_dir: Path,
    control_db_path: Path,
) -> tuple[Path, list[ArtifactManifestEntry]]:
    artifact_root = state_dir / "artifacts"
    backup_artifact_root = backup_dir / "artifacts"
    _copy_tree(artifact_root, backup_artifact_root)

    entries: list[ArtifactManifestEntry] = []
    for row in _collect_artifact_rows(control_db_path):
        storage_path = Path(str(row["storage_path"]))
        try:
            relative_storage_path = storage_path.relative_to(state_dir).as_posix()
        except ValueError:
            relative_storage_path = storage_path.name
        backup_storage_path = backup_dir / relative_storage_path
        file_present = backup_storage_path.is_file()
        file_sha256 = _sha256_file(backup_storage_path) if file_present else None
        entries.append(
            ArtifactManifestEntry(
                artifact_id=str(row["artifact_id"]),
                relative_storage_path=relative_storage_path,
                public_filename=str(row["public_filename"]),
                mime_type=str(row["mime_type"]),
                size_bytes=int(row["size_bytes"]),
                sha256=str(row["sha256"]),
                status=str(row["status"]),
                failure_reason=row["failure_reason"],
                file_present=file_present,
                file_sha256=file_sha256,
            )
        )

    manifest_path = backup_dir / "artifact-manifest.json"
    _write_json(
        manifest_path,
        {
            "timestamp": _now_iso(),
            "state_dir": str(state_dir),
            "artifact_root": str(backup_artifact_root),
            "artifacts": [asdict(entry) for entry in entries],
        },
    )
    return manifest_path, entries


def _verify_artifact_manifest(
    *,
    manifest_path: Path,
    backup_dir: Path,
) -> list[ArtifactManifestEntry]:
    manifest = _read_json(manifest_path)
    entries: list[ArtifactManifestEntry] = []
    for raw in manifest.get("artifacts", []):
        entry = ArtifactManifestEntry(
            artifact_id=str(raw["artifact_id"]),
            relative_storage_path=str(raw["relative_storage_path"]),
            public_filename=str(raw["public_filename"]),
            mime_type=str(raw["mime_type"]),
            size_bytes=int(raw["size_bytes"]),
            sha256=str(raw["sha256"]),
            status=str(raw["status"]),
            failure_reason=raw.get("failure_reason"),
            file_present=bool(raw["file_present"]),
            file_sha256=raw.get("file_sha256"),
        )
        file_path = backup_dir / entry.relative_storage_path
        if not file_path.is_file():
            raise RuntimeError(
                f"artifact backup missing file: {entry.relative_storage_path}"
            )
        if _sha256_file(file_path) != entry.sha256:
            raise RuntimeError(
                f"artifact backup hash mismatch for {entry.relative_storage_path}"
            )
        entries.append(entry)
    return entries


def _snapshot_sqlite(
    path: Path,
    tables: tuple[str, ...],
) -> tuple[str, dict[str, int]]:
    return _sqlite_integrity_check(path), _sqlite_table_counts(path, tables)


def create_backup_report(
    *,
    state_dir: Path,
    backup_root: Path,
) -> BackupReport:
    timestamp = _now_iso()
    backup_dir = backup_root / timestamp.replace(":", "").replace("+", "_")
    restore_dir = backup_dir / "restored"
    backup_dir.mkdir(parents=True, exist_ok=True)
    restore_dir.mkdir(parents=True, exist_ok=True)

    sqlite_checks: list[SQLiteBackupCheck] = []
    for db_name, tables in REQUIRED_SQLITE_DB_SPECS.items():
        source_path = state_dir / db_name
        if not source_path.is_file():
            raise RuntimeError(f"missing v2 sqlite database: {source_path}")

        backup_path = backup_dir / db_name
        restore_path = restore_dir / db_name
        _backup_sqlite_file(source_path, backup_path)
        shutil.copy2(backup_path, restore_path)

        source_integrity, source_counts = _snapshot_sqlite(source_path, tables)
        backup_integrity, backup_counts = _snapshot_sqlite(backup_path, tables)
        restore_integrity, restore_counts = _snapshot_sqlite(restore_path, tables)

        counts_match = source_counts == backup_counts == restore_counts
        integrity_match = (
            source_integrity == backup_integrity == restore_integrity == "ok"
        )
        passed = counts_match and integrity_match
        reason = "" if passed else "sqlite backup/restore counts or integrity mismatch"
        sqlite_checks.append(
            SQLiteBackupCheck(
                database_name=db_name,
                source_path=str(source_path),
                backup_path=str(backup_path),
                restore_path=str(restore_path),
                source_counts=source_counts,
                backup_counts=backup_counts,
                restore_counts=restore_counts,
                source_integrity=source_integrity,
                backup_integrity=backup_integrity,
                restore_integrity=restore_integrity,
                passed=passed,
                reason=reason,
            )
        )

    control_db_path = backup_dir / "agent_control.sqlite"
    manifest_path, manifest_entries = _build_artifact_manifest(
        state_dir=state_dir,
        backup_dir=backup_dir,
        control_db_path=control_db_path,
    )
    manifest_entries = _verify_artifact_manifest(
        manifest_path=manifest_path,
        backup_dir=backup_dir,
    )

    overall_pass = all(check.passed for check in sqlite_checks)
    residual_risks: list[str] = []
    for check in sqlite_checks:
        if not check.passed:
            residual_risks.append(f"{check.database_name}: {check.reason}")
    if not manifest_entries and not _collect_artifact_rows(control_db_path):
        residual_risks.append("artifact manifest is empty")

    return BackupReport(
        timestamp=timestamp,
        state_dir=str(state_dir),
        backup_dir=str(backup_dir),
        restore_dir=str(restore_dir),
        sqlite_checks=sqlite_checks,
        artifact_manifest_path=str(manifest_path),
        artifact_entries=manifest_entries,
        overall_pass=overall_pass and True,
        residual_risks=residual_risks,
    )


def _validate_window_evidence(
    *,
    evidence: dict[str, Any],
    expected_cohort: str,
) -> tuple[bool, list[str], dict[str, str], dict[str, int], float, float, Path, dict[str, Any]]:
    residual_risks: list[str] = []

    if str(evidence.get("cohort", "")).strip() != expected_cohort:
        residual_risks.append(
            f"cohort mismatch: expected {expected_cohort}, got {evidence.get('cohort')!r}"
        )

    try:
        window_hours = float(evidence.get("window_hours"))
    except (TypeError, ValueError):
        residual_risks.append("window_hours missing or invalid")
        window_hours = 0.0
    if window_hours < 24.0:
        residual_risks.append(f"window shorter than 24h: {window_hours}")

    incident_counts = evidence.get("incident_counts", {})
    if not isinstance(incident_counts, dict):
        residual_risks.append("incident_counts missing or invalid")
        incident_counts = {}
    p0 = int(incident_counts.get("p0", -1))
    p1 = int(incident_counts.get("p1", -1))
    if p0 != 0 or p1 != 0:
        residual_risks.append(f"P0/P1 incidents recorded: p0={p0} p1={p1}")

    try:
        five_xx_rate = float(evidence.get("five_xx_rate"))
    except (TypeError, ValueError):
        residual_risks.append("five_xx_rate missing or invalid")
        five_xx_rate = 1.0
    if five_xx_rate >= 0.01:
        residual_risks.append(f"5xx rate too high: {five_xx_rate}")

    try:
        run_recovery_failure_rate = float(evidence.get("run_recovery_failure_rate"))
    except (TypeError, ValueError):
        residual_risks.append("run_recovery_failure_rate missing or invalid")
        run_recovery_failure_rate = 1.0
    if run_recovery_failure_rate >= 0.005:
        residual_risks.append(
            f"run recovery failure rate too high: {run_recovery_failure_rate}"
        )

    section_statuses: dict[str, str] = {}
    sections = evidence.get("report_sections", {})
    if not isinstance(sections, dict):
        residual_risks.append("report_sections missing or invalid")
        sections = {}
    missing_sections = [name for name in REQUIRED_WINDOW_SECTIONS if name not in sections]
    if missing_sections:
        residual_risks.append(f"missing report sections: {missing_sections}")
    for name in REQUIRED_WINDOW_SECTIONS:
        section = sections.get(name, {})
        status = str(section.get("status", "")).strip().lower()
        section_statuses[name] = status
        if status not in {"passed", "failed", "skipped"}:
            residual_risks.append(f"invalid section status for {name}: {status!r}")

    rollback = evidence.get("rollback_drill", {})
    if not isinstance(rollback, dict):
        residual_risks.append("rollback_drill missing or invalid")
        rollback = {}
    rollback_status = str(rollback.get("status", "")).strip().lower()
    rollback_pass = rollback_status == "passed"
    required_rollback_flags = (
        "new_requests_routed_to_v1",
        "paused_effectful_runs_preserved",
        "v2_restored_after_rollback",
    )
    for key in required_rollback_flags:
        if not bool(rollback.get(key, False)):
            residual_risks.append(f"rollback drill missing or false: {key}")
    if not rollback_pass:
        residual_risks.append(f"rollback drill status={rollback_status!r}")

    hard_gate_evidence_raw = evidence.get("hard_gate_evidence")
    if not hard_gate_evidence_raw:
        residual_risks.append("hard_gate_evidence missing")
        hard_gate_evidence = Path("")
    else:
        hard_gate_evidence = Path(str(hard_gate_evidence_raw))

    return (
        not residual_risks,
        residual_risks,
        section_statuses,
        incident_counts,
        five_xx_rate,
        run_recovery_failure_rate,
        hard_gate_evidence,
        rollback,
    )


def _evaluate_window(
    *,
    cohort: str,
    window_evidence_path: Path,
    backup_passed: bool,
    corpus_path: Path,
    output_dir: Path,
) -> tuple[CanaryWindowReport, Path, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not backup_passed:
        residual = [f"backup verification failed; {cohort} window held at 0%"]
        return (
            CanaryWindowReport(
                cohort=cohort,
                window_hours=0.0,
                hard_gate_evidence_path="",
                hard_gate_report_path="",
                hard_gate_pass=False,
                rollback_drill_path="",
                rollback_pass=False,
                incident_counts={"p0": 0, "p1": 0},
                five_xx_rate=1.0,
                run_recovery_failure_rate=1.0,
                section_statuses={name: "skipped" for name in REQUIRED_WINDOW_SECTIONS},
                status="skipped",
                residual_risks=residual,
            ),
            output_dir / f"{cohort}-window-report.json",
            residual,
        )

    evidence = _read_json(window_evidence_path)
    (
        valid,
        residual_risks,
        section_statuses,
        incident_counts,
        five_xx_rate,
        run_recovery_failure_rate,
        hard_gate_evidence,
        rollback,
    ) = _validate_window_evidence(evidence=evidence, expected_cohort=cohort)

    hard_gate_report = None
    hard_gate_report_path = output_dir / f"{cohort}-hard-gates.json"
    if hard_gate_evidence and hard_gate_evidence.is_file():
        hard_gate_report = evaluate_acceptance(
            corpus_path=corpus_path,
            evidence_path=hard_gate_evidence,
            executed_commands=[
                subprocess.list2cmdline(
                    [
                        sys.executable,
                        "-m",
                        "scripts.run_agent_v2_acceptance",
                        "--corpus",
                        str(corpus_path),
                        "--staging-evidence",
                        str(hard_gate_evidence),
                        "--output",
                        str(hard_gate_report_path),
                    ]
                )
            ],
        )
        _write_json(hard_gate_report_path, asdict(hard_gate_report))
        hard_gate_pass = hard_gate_report.overall_pass
    else:
        hard_gate_pass = False
        if hard_gate_evidence:
            residual_risks.append(f"hard gate evidence missing: {hard_gate_evidence}")
        _write_json(
            hard_gate_report_path,
            {
                "overall_pass": False,
                "residual_risks": ["missing hard gate evidence"],
            },
        )

    rollback_pass = bool(rollback.get("status") == "passed") and valid
    status: Literal["passed", "failed", "skipped"]
    if valid and hard_gate_pass and rollback_pass:
        status = "passed"
    else:
        status = "failed"

    window_report = CanaryWindowReport(
        cohort=cohort,
        window_hours=float(evidence.get("window_hours", 0.0)),
        hard_gate_evidence_path=str(hard_gate_evidence) if hard_gate_evidence else "",
        hard_gate_report_path=str(hard_gate_report_path),
        hard_gate_pass=hard_gate_pass,
        rollback_drill_path=str(window_evidence_path),
        rollback_pass=rollback_pass,
        incident_counts={
            "p0": int(incident_counts.get("p0", 0)),
            "p1": int(incident_counts.get("p1", 0)),
        },
        five_xx_rate=five_xx_rate,
        run_recovery_failure_rate=run_recovery_failure_rate,
        section_statuses=section_statuses,
        status=status,
        residual_risks=residual_risks
        + (
            [] if hard_gate_pass else ["hard gate report did not pass"]
        )
        + ([] if rollback_pass else ["rollback drill did not pass"]),
    )

    return window_report, hard_gate_report_path, window_report.residual_risks


def run_canary_rollout(
    *,
    state_dir: Path,
    backup_root: Path,
    corpus_path: Path,
    internal_window_evidence_path: Path,
    one_percent_window_evidence_path: Path,
    output_dir: Path | None = None,
) -> CanaryRolloutReport:
    backup = create_backup_report(state_dir=state_dir, backup_root=backup_root)
    report_dir = output_dir or Path(backup.backup_dir)

    windows: list[CanaryWindowReport] = []
    residual_risks: list[str] = list(backup.residual_risks)
    evidence_paths = [backup.artifact_manifest_path]
    executed_commands = [
        subprocess.list2cmdline(
            [
                sys.executable,
                "-m",
                "scripts.run_agent_v2_canaries",
                "--state-dir",
                str(state_dir),
                "--backup-root",
                str(backup_root),
                "--corpus",
                str(corpus_path),
                "--internal-window-evidence",
                str(internal_window_evidence_path),
                "--one-percent-window-evidence",
                str(one_percent_window_evidence_path),
            ]
        )
    ]

    internal_report, internal_hard_gate_path, internal_risks = _evaluate_window(
        cohort="internal",
        window_evidence_path=internal_window_evidence_path,
        backup_passed=backup.overall_pass,
        corpus_path=corpus_path,
        output_dir=report_dir,
    )
    windows.append(internal_report)
    residual_risks.extend(internal_risks)
    if internal_hard_gate_path:
        evidence_paths.append(str(internal_hard_gate_path))
    evidence_paths.append(str(internal_window_evidence_path))

    one_percent_report: CanaryWindowReport
    if internal_report.status == "passed":
        one_percent_report, one_percent_hard_gate_path, one_percent_risks = _evaluate_window(
            cohort="1%",
            window_evidence_path=one_percent_window_evidence_path,
            backup_passed=backup.overall_pass,
            corpus_path=corpus_path,
            output_dir=report_dir,
        )
        windows.append(one_percent_report)
        residual_risks.extend(one_percent_risks)
        if one_percent_hard_gate_path:
            evidence_paths.append(str(one_percent_hard_gate_path))
        evidence_paths.append(str(one_percent_window_evidence_path))
    else:
        one_percent_report = CanaryWindowReport(
            cohort="1%",
            window_hours=0.0,
            hard_gate_evidence_path="",
            hard_gate_report_path="",
            hard_gate_pass=False,
            rollback_drill_path="",
            rollback_pass=False,
            incident_counts={"p0": 0, "p1": 0},
            five_xx_rate=1.0,
            run_recovery_failure_rate=1.0,
            section_statuses={name: "skipped" for name in REQUIRED_WINDOW_SECTIONS},
            status="skipped",
            residual_risks=["internal window failed; 1% held at 0%"],
        )
        windows.append(one_percent_report)
        residual_risks.extend(one_percent_report.residual_risks)

    promotion_ok = False
    if internal_report.status == "passed" and one_percent_report.status == "passed":
        internal_window_evidence = _read_json(internal_window_evidence_path)
        one_percent_window_evidence = _read_json(one_percent_window_evidence_path)
        internal_ended_at = _parse_iso(str(internal_window_evidence.get("ended_at", "")))
        one_percent_started_at = _parse_iso(str(one_percent_window_evidence.get("started_at", "")))
        promotion_ok = (
            internal_ended_at is not None
            and one_percent_started_at is not None
            and one_percent_started_at >= internal_ended_at
        )
    if not promotion_ok:
        residual_risks.append("1% cohort promotion gate failed")

    overall_pass = (
        backup.overall_pass
        and all(window.status == "passed" for window in windows)
        and promotion_ok
    )
    if not overall_pass:
        failed_windows = [window.cohort for window in windows if window.status != "passed"]
        if failed_windows:
            residual_risks.append(f"failed or skipped windows: {failed_windows}")

    report = CanaryRolloutReport(
        timestamp=_now_iso(),
        backup=backup,
        windows=windows,
        overall_pass=overall_pass,
        executed_commands=executed_commands,
        evidence_paths=evidence_paths,
        residual_risks=residual_risks,
    )
    _write_json(report_dir / "agent-v2-canary-report.json", asdict(report))
    return report


def main(argv: list[str] | None = None) -> int:
    ai_service_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Ticket 19 Chat v2 canary coordinator")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(settings.agent_v2_state_dir),
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=ai_service_dir / "data" / "canary-backups",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ai_service_dir / "tests" / "fixtures" / "agent_v2_cases.jsonl",
    )
    parser.add_argument(
        "--internal-window-evidence",
        type=Path,
        required=True,
        help="24-hour internal window evidence JSON",
    )
    parser.add_argument(
        "--one-percent-window-evidence",
        type=Path,
        required=True,
        help="24-hour one-percent window evidence JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit final report path",
    )
    args = parser.parse_args(argv)

    report = run_canary_rollout(
        state_dir=args.state_dir,
        backup_root=args.backup_root,
        corpus_path=args.corpus,
        internal_window_evidence_path=args.internal_window_evidence,
        one_percent_window_evidence_path=args.one_percent_window_evidence,
        output_dir=args.output.parent if args.output else None,
    )

    output_path = args.output or Path(report.backup.backup_dir) / "agent-v2-canary-report.json"
    if output_path != Path(report.backup.backup_dir) / "agent-v2-canary-report.json":
        _write_json(output_path, asdict(report))

    print(f"Chat v2 canary report: {output_path}")
    print(f"overall_pass: {report.overall_pass}")
    for window in report.windows:
        print(f"  {window.cohort}: {window.status}")
    return 0 if report.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
