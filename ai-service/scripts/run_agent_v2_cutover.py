"""Ticket 21 — 100% Chat v2 cutover coordinator.

This script does not flip production traffic by itself. It validates the
pre-cutover gates and records the release entry (timestamp, operator,
config, report paths, rollback entry) without secrets.

Reuses the backup/restore validation from canary rollout (ticket 19/20)
and the acceptance gate from ticket 18.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from config import settings
from scripts.run_agent_v2_canaries import BackupReport, create_backup_report


SECRET_FIELD_PATTERNS = (
    "secret",
    "api_key",
    "apikey",
    "token",
    "password",
    "passwd",
    "credential",
    "private_key",
    "auth",
    "access_key",
    "secret_key",
    "session",
    "cookie",
)

REQUIRED_RELEASE_RECORD_FIELDS = (
    "timestamp",
    "operator",
    "config_snapshot",
    "acceptance_report_path",
    "backup_report_path",
    "rollback_drill_path",
    "rollback_entry_path",
    "v1_availability_check",
    "v2_default_check",
    "residual_risks",
    "overall_pass",
)

# AC3: v1 components that must remain available post-cutover.
# Each entry is (path_relative_to_project_root, component_label).
# Resolved via Path(__file__).resolve().parents[2] (project root).
V1_AVAILABILITY_FILES = (
    ("ai-service/api/chat.py", "v1 API"),
    ("ai-service/agent/__init__.py", "v1 Agent runtime"),
    ("ai-service/agent/agent.py", "v1 Agent runtime"),
    ("gateway/handler/chat.go", "Gateway v1 route"),
    ("frontend/src/api/sse.ts", "Frontend v1 parser"),
    ("ai-service/config.py", "Runtime configuration"),
)

# AC3: components that cannot be auto-verified from the source tree
# (e.g., deployed image in registry) — surfaced as residual risks for manual sign-off.
V1_AVAILABILITY_RESIDUAL_RISKS = (
    "v1 deployment image availability — 需人工确认 (registry/CI 检查)",
)

# AC2: v1 conversation migration policy evidence (doc OR config field).
V1_CONVERSATION_POLICY_DOC = "ai-service/docs/chat_v2_cutover.md"
V1_CONVERSATION_POLICY_CONFIG_ATTR = "v1_conversation_policy"

# AC2: Gateway runtime_selector that must not silently flip existing v1 conversations.
# Either runtime_selector.go (dedicated file) OR chat_v2.go (defines RuntimeSelector) is acceptable.
RUNTIME_SELECTOR_FILES = (
    "gateway/handler/runtime_selector.go",
    "gateway/handler/chat_v2.go",
)

V2_DEFAULT_FILES = (
    "api/chat_v2.py",
    "agent_v2/__init__.py",
    "agent_v2/startup.py",
)

REQUIRED_ROLLBACK_ENTRY_FIELDS = (
    "rollback_command",
    "v1_route_target",
    "v2_disable_flag",
    "v2_data_preserved",
)

ROLLBACK_DRILL_REQUIRED_FLAGS = (
    "new_requests_routed_to_v1",
    "paused_effectful_runs_preserved",
    "v2_restored_after_rollback",
)

# AC4: sections that must continue to be recorded at 100% cutover
# with the same fidelity as the 50% canary window.
REQUIRED_METRIC_SECTIONS = (
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


@dataclass(frozen=True, slots=True)
class CutoverPreCheck:
    check_id: str
    status: Literal["passed", "failed", "skipped"]
    evidence: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class CutoverReleaseRecord:
    timestamp: str
    operator: str
    config_snapshot: dict[str, Any]
    acceptance_report_path: str
    backup_report_path: str
    rollback_drill_path: str
    rollback_entry_path: str
    v1_availability_check: CutoverPreCheck
    v2_default_check: CutoverPreCheck
    residual_risks: list[str] = field(default_factory=list)
    overall_pass: bool = False
    # AC4: confirms section-level metrics recording (quality/performance/cost/
    # dependency/capacity/privacy/security/recovery/user_visible) remains in place
    # at 100% cutover with the same fidelity as the 50% canary window.
    metrics_recording_ready: bool = False


@dataclass(frozen=True, slots=True)
class CutoverReport:
    timestamp: str
    pre_checks: list[CutoverPreCheck]
    release_record: CutoverReleaseRecord | None
    overall_pass: bool
    executed_commands: list[str] = field(default_factory=list)
    evidence_paths: list[str] = field(default_factory=list)
    residual_risks: list[str] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _looks_like_secret(key: str) -> bool:
    lowered = key.lower()
    return any(pattern in lowered for pattern in SECRET_FIELD_PATTERNS)


def _scrub_secrets(payload: Any) -> Any:
    """Recursively redact secret-looking keys from a dict/list payload."""
    if isinstance(payload, dict):
        scrubbed: dict[str, Any] = {}
        for key, value in payload.items():
            if _looks_like_secret(str(key)):
                scrubbed[str(key)] = "[REDACTED]"
            else:
                scrubbed[str(key)] = _scrub_secrets(value)
        return scrubbed
    if isinstance(payload, list):
        return [_scrub_secrets(item) for item in payload]
    return payload


def _validate_acceptance_report(report_path: Path) -> CutoverPreCheck:
    if not report_path.is_file():
        return CutoverPreCheck(
            check_id="acceptance_report",
            status="failed",
            evidence=f"path={report_path}",
            reason="acceptance report file missing",
        )
    try:
        payload = _read_json(report_path)
    except (OSError, json.JSONDecodeError) as exc:
        return CutoverPreCheck(
            check_id="acceptance_report",
            status="failed",
            evidence=f"path={report_path}",
            reason=f"acceptance report unreadable: {exc}",
        )
    overall_pass = bool(payload.get("overall_pass", False))
    if not overall_pass:
        gates = payload.get("gates", [])
        failed_gates = [
            gate.get("gate_id", "?")
            for gate in gates
            if isinstance(gate, dict) and gate.get("status") != "passed"
        ]
        return CutoverPreCheck(
            check_id="acceptance_report",
            status="failed",
            evidence=f"overall_pass={overall_pass} failed_gates={failed_gates}",
            reason="acceptance report overall_pass is False",
        )
    return CutoverPreCheck(
        check_id="acceptance_report",
        status="passed",
        evidence=(
            f"overall_pass={overall_pass} "
            f"timestamp={payload.get('timestamp', '')}"
        ),
    )


def _validate_backups(
    state_dir: Path,
    backup_dir: Path,
) -> tuple[CutoverPreCheck, BackupReport | None]:
    try:
        backup_report = create_backup_report(
            state_dir=state_dir, backup_root=backup_dir
        )
    except Exception as exc:
        return (
            CutoverPreCheck(
                check_id="backup_restore",
                status="failed",
                evidence=f"state_dir={state_dir} backup_dir={backup_dir}",
                reason=f"backup report generation failed: {exc}",
            ),
            None,
        )
    if not backup_report.overall_pass:
        return (
            CutoverPreCheck(
                check_id="backup_restore",
                status="failed",
                evidence=(
                    f"backup_dir={backup_report.backup_dir} "
                    f"residual_risks={backup_report.residual_risks}"
                ),
                reason="backup report overall_pass is False",
            ),
            backup_report,
        )
    return (
        CutoverPreCheck(
            check_id="backup_restore",
            status="passed",
            evidence=(
                f"sqlite_checks={len(backup_report.sqlite_checks)} "
                f"artifact_entries={len(backup_report.artifact_entries)} "
                f"manifest_path={backup_report.artifact_manifest_path}"
            ),
        ),
        backup_report,
    )


def _validate_force_v1_drill(canary_report_path: Path) -> CutoverPreCheck:
    if not canary_report_path.is_file():
        return CutoverPreCheck(
            check_id="force_v1_drill",
            status="failed",
            evidence=f"path={canary_report_path}",
            reason="canary report file missing",
        )
    try:
        payload = _read_json(canary_report_path)
    except (OSError, json.JSONDecodeError) as exc:
        return CutoverPreCheck(
            check_id="force_v1_drill",
            status="failed",
            evidence=f"path={canary_report_path}",
            reason=f"canary report unreadable: {exc}",
        )
    windows = payload.get("windows", [])
    if not isinstance(windows, list) or not windows:
        return CutoverPreCheck(
            check_id="force_v1_drill",
            status="failed",
            evidence="windows=[]",
            reason="canary report missing windows list",
        )
    drill_found = False
    drill_passed = False
    drill_evidence_parts: list[str] = []
    for window in windows:
        if not isinstance(window, dict):
            continue
        rollback_pass = bool(window.get("rollback_pass", False))
        rollback_drill_path = str(window.get("rollback_drill_path", ""))
        rollback_drill = window.get("rollback_drill")
        if rollback_drill_path and not isinstance(rollback_drill, dict):
            try:
                drill_payload = _read_json(Path(rollback_drill_path))
                rollback_drill = drill_payload.get("rollback_drill", {})
            except (OSError, json.JSONDecodeError):
                rollback_drill = {}
        if isinstance(rollback_drill, dict) and rollback_drill:
            drill_found = True
            missing_flags = [
                flag
                for flag in ROLLBACK_DRILL_REQUIRED_FLAGS
                if not bool(rollback_drill.get(flag, False))
            ]
            drill_status = str(rollback_drill.get("status", "")).strip().lower()
            if drill_status == "passed" and not missing_flags:
                drill_passed = True
                drill_evidence_parts.append(
                    f"{window.get('cohort', '?')}: status=passed"
                )
                break
            drill_evidence_parts.append(
                f"{window.get('cohort', '?')}: "
                f"status={drill_status} missing={missing_flags}"
            )
        elif rollback_pass:
            drill_found = True
            drill_passed = True
            drill_evidence_parts.append(
                f"{window.get('cohort', '?')}: rollback_pass=True"
            )
            break
    if not drill_found:
        return CutoverPreCheck(
            check_id="force_v1_drill",
            status="failed",
            evidence=f"windows={len(windows)}",
            reason="force-v1 drill evidence not present in any window",
        )
    if not drill_passed:
        return CutoverPreCheck(
            check_id="force_v1_drill",
            status="failed",
            evidence="; ".join(drill_evidence_parts),
            reason="force-v1 drill did not pass in any window",
        )
    return CutoverPreCheck(
        check_id="force_v1_drill",
        status="passed",
        evidence="; ".join(drill_evidence_parts),
    )


def _validate_metrics_recording_readiness(
    canary_report_path: Path,
) -> CutoverPreCheck:
    """AC4: confirm 100% cutover keeps recording the same section-level metrics
    (quality/performance/cost/dependency/capacity/privacy/security/recovery/
    user_visible) and that the hard-gate evidence path is still reachable so
    monitoring is reusable post-cutover."""
    if not canary_report_path.is_file():
        return CutoverPreCheck(
            check_id="metrics_recording_readiness",
            status="failed",
            evidence=f"path={canary_report_path}",
            reason="canary report file missing",
        )
    try:
        payload = _read_json(canary_report_path)
    except (OSError, json.JSONDecodeError) as exc:
        return CutoverPreCheck(
            check_id="metrics_recording_readiness",
            status="failed",
            evidence=f"path={canary_report_path}",
            reason=f"canary report unreadable: {exc}",
        )
    windows = payload.get("windows", [])
    if not isinstance(windows, list) or not windows:
        return CutoverPreCheck(
            check_id="metrics_recording_readiness",
            status="failed",
            evidence="windows=[]",
            reason="canary report missing windows list",
        )
    problems: list[str] = []
    checked_windows = 0
    hard_gate_paths_present = True
    for window in windows:
        if not isinstance(window, dict):
            continue
        checked_windows += 1
        cohort = window.get("cohort", "?")
        section_statuses = window.get("section_statuses", {})
        if not isinstance(section_statuses, dict):
            section_statuses = {}
        missing_sections = [
            name
            for name in REQUIRED_METRIC_SECTIONS
            if name not in section_statuses
        ]
        if missing_sections:
            problems.append(f"{cohort}: missing sections {missing_sections}")
        hard_gate_path = str(window.get("hard_gate_evidence_path", ""))
        if not hard_gate_path or not Path(hard_gate_path).is_file():
            hard_gate_paths_present = False
            problems.append(
                f"{cohort}: hard_gate_evidence_path missing or invalid"
            )
    if problems:
        return CutoverPreCheck(
            check_id="metrics_recording_readiness",
            status="failed",
            evidence=(
                f"checked_windows={checked_windows} "
                f"sections_per_window={len(REQUIRED_METRIC_SECTIONS)} "
                f"hard_gate_paths_present={hard_gate_paths_present}"
            ),
            reason="; ".join(problems),
        )
    return CutoverPreCheck(
        check_id="metrics_recording_readiness",
        status="passed",
        evidence=(
            f"checked_windows={checked_windows} "
            f"sections={len(REQUIRED_METRIC_SECTIONS)} "
            f"hard_gate_paths_present={hard_gate_paths_present}"
        ),
    )


def _validate_v1_availability() -> CutoverPreCheck:
    project_root = Path(__file__).resolve().parents[2]
    missing: list[str] = []
    for relative, _label in V1_AVAILABILITY_FILES:
        absolute = project_root / relative
        if not absolute.exists():
            missing.append(relative)
    if missing:
        return CutoverPreCheck(
            check_id="v1_availability",
            status="failed",
            evidence=f"project_root={project_root}",
            reason=f"v1 components missing: {missing}",
        )
    return CutoverPreCheck(
        check_id="v1_availability",
        status="passed",
        evidence=(
            f"checked={len(V1_AVAILABILITY_FILES)} components "
            f"project_root={project_root}"
        ),
    )


def _validate_v2_default() -> CutoverPreCheck:
    ai_service_dir = Path(__file__).resolve().parent.parent
    project_root = Path(__file__).resolve().parents[2]
    missing: list[str] = []
    for relative in V2_DEFAULT_FILES:
        absolute = ai_service_dir / relative
        if not absolute.exists():
            missing.append(relative)
    if missing:
        return CutoverPreCheck(
            check_id="v2_default",
            status="failed",
            evidence=f"ai_service_dir={ai_service_dir}",
            reason=f"v2 components missing: {missing}",
        )
    v2_enabled = bool(getattr(settings, "agent_v2_enabled", False))
    if not v2_enabled:
        env_value = os.getenv("AGENT_V2_ENABLED", "").strip().lower()
        v2_enabled = env_value in ("1", "true", "yes", "on")
    if not v2_enabled:
        return CutoverPreCheck(
            check_id="v2_default",
            status="failed",
            evidence=f"agent_v2_enabled={v2_enabled}",
            reason="new conversations are not configured to default to v2",
        )
    # AC2: existing v1 conversations must have a documented migration policy
    # and the Gateway runtime_selector must not silently flip them to v2.
    notes: list[str] = []
    policy_doc = project_root / V1_CONVERSATION_POLICY_DOC
    policy_config = bool(
        getattr(settings, V1_CONVERSATION_POLICY_CONFIG_ATTR, "")
    )
    if not policy_doc.is_file() and not policy_config:
        notes.append("v1 conversation 迁移策略需人工确认")
    selector_present = any(
        (project_root / relative).is_file() for relative in RUNTIME_SELECTOR_FILES
    )
    if not selector_present:
        notes.append(
            "gateway runtime_selector missing — 需人工确认无提示状态转换"
        )
    return CutoverPreCheck(
        check_id="v2_default",
        status="passed",
        evidence=(
            f"checked={len(V2_DEFAULT_FILES)} files "
            f"agent_v2_enabled={v2_enabled} "
            f"policy_notes={len(notes)}"
        ),
        reason="; ".join(notes),
    )


def _validate_rollback_entry(rollback_entry_path: Path) -> CutoverPreCheck:
    if not rollback_entry_path.is_file():
        return CutoverPreCheck(
            check_id="rollback_entry",
            status="failed",
            evidence=f"path={rollback_entry_path}",
            reason="rollback entry file missing",
        )
    try:
        payload = _read_json(rollback_entry_path)
    except (OSError, json.JSONDecodeError) as exc:
        return CutoverPreCheck(
            check_id="rollback_entry",
            status="failed",
            evidence=f"path={rollback_entry_path}",
            reason=f"rollback entry unreadable: {exc}",
        )
    missing = [
        field_name
        for field_name in REQUIRED_ROLLBACK_ENTRY_FIELDS
        if not payload.get(field_name)
    ]
    if missing:
        return CutoverPreCheck(
            check_id="rollback_entry",
            status="failed",
            evidence=f"path={rollback_entry_path}",
            reason=f"rollback entry missing fields: {missing}",
        )
    scrubbed = _scrub_secrets(payload)
    if scrubbed != payload:
        return CutoverPreCheck(
            check_id="rollback_entry",
            status="failed",
            evidence=f"path={rollback_entry_path}",
            reason="rollback entry contains secret-looking keys",
        )
    return CutoverPreCheck(
        check_id="rollback_entry",
        status="passed",
        evidence=(
            f"path={rollback_entry_path} "
            f"fields={sorted(payload.keys())}"
        ),
    )


def _build_config_snapshot(
    state_dir: Path,
    ai_service_dir: Path,
) -> dict[str, Any]:
    """Build a sanitized config snapshot for the release record."""
    snapshot: dict[str, Any] = {
        "agent_v2_enabled": bool(getattr(settings, "agent_v2_enabled", False)),
        "agent_v2_uniharness_pin": str(
            getattr(settings, "agent_v2_uniharness_pin", "")
        ),
        "agent_v2_state_dir": str(state_dir),
        "agent_v2_sandbox_type": str(
            getattr(settings, "agent_v2_sandbox_type", "")
        ),
        "agent_v2_cowork_policy": str(
            getattr(settings, "agent_v2_cowork_policy", "")
        ),
        "ai_service_dir": str(ai_service_dir),
        # Record key=value for non-secret env vars and key=[REDACTED] for
        # secret-looking ones so operators can audit the runtime config without
        # leaking material. _scrub_secrets is still applied as a safety net.
        "environment_values": {
            key: (
                "[REDACTED]"
                if _looks_like_secret(key)
                else os.environ[key]
            )
            for key in sorted(os.environ)
            if key.startswith(("AGENT_V2_", "CHAT_V2_"))
        },
    }
    return _scrub_secrets(snapshot)


def _build_release_record(
    *,
    operator: str,
    acceptance_report_path: Path,
    backup_report: BackupReport | None,
    canary_report_path: Path,
    rollback_entry_path: Path,
    v1_availability_check: CutoverPreCheck,
    v2_default_check: CutoverPreCheck,
    config_snapshot: dict[str, Any],
    residual_risks: list[str],
    overall_pass: bool,
    metrics_recording_ready: bool = False,
) -> CutoverReleaseRecord:
    return CutoverReleaseRecord(
        timestamp=_now_iso(),
        operator=operator,
        config_snapshot=_scrub_secrets(config_snapshot),
        acceptance_report_path=str(acceptance_report_path),
        backup_report_path=(
            str(backup_report.backup_dir) if backup_report else ""
        ),
        rollback_drill_path=str(canary_report_path),
        rollback_entry_path=str(rollback_entry_path),
        v1_availability_check=v1_availability_check,
        v2_default_check=v2_default_check,
        residual_risks=list(residual_risks),
        overall_pass=overall_pass,
        metrics_recording_ready=metrics_recording_ready,
    )


def _write_release_record(record: CutoverReleaseRecord, path: Path) -> Path:
    _write_json(path, asdict(record))
    return path


def run_cutover_validation(
    *,
    state_dir: Path,
    acceptance_report_path: Path,
    canary_report_path: Path,
    operator: str,
    rollback_entry_path: Path,
    output: Path | None = None,
    backup_dir: Path | None = None,
) -> CutoverReport:
    ai_service_dir = Path(__file__).resolve().parent.parent
    timestamp = _now_iso()
    pre_checks: list[CutoverPreCheck] = []
    evidence_paths: list[str] = []
    residual_risks: list[str] = []
    cmd_args: list[str] = [
        sys.executable,
        "-m",
        "scripts.run_agent_v2_cutover",
        "--state-dir",
        str(state_dir),
        "--acceptance-report",
        str(acceptance_report_path),
        "--canary-report",
        str(canary_report_path),
        "--operator",
        operator,
        "--rollback-entry",
        str(rollback_entry_path),
    ]
    if output is not None:
        cmd_args.extend(["--output", str(output)])
    if backup_dir is not None:
        cmd_args.extend(["--backup-dir", str(backup_dir)])
    executed_commands: list[str] = [subprocess.list2cmdline(cmd_args)]

    # AC1: acceptance report overall_pass=True
    acceptance_check = _validate_acceptance_report(acceptance_report_path)
    pre_checks.append(acceptance_check)
    evidence_paths.append(str(acceptance_report_path))
    if acceptance_check.status != "passed":
        residual_risks.append(f"acceptance_report: {acceptance_check.reason}")

    # AC1: three SQLite + artifact manifest have recoverable backups
    backup_root = backup_dir or (
        state_dir / f"cutover-backup-{timestamp.replace(':', '').replace('+', '_')}"
    )
    backup_check, backup_report = _validate_backups(state_dir, backup_root)
    pre_checks.append(backup_check)
    if backup_report is not None:
        evidence_paths.append(backup_report.artifact_manifest_path)
    if backup_check.status != "passed":
        residual_risks.append(f"backup_restore: {backup_check.reason}")

    # AC1: force-v1 drill still succeeds (read from canary report)
    drill_check = _validate_force_v1_drill(canary_report_path)
    pre_checks.append(drill_check)
    evidence_paths.append(str(canary_report_path))
    if drill_check.status != "passed":
        residual_risks.append(f"force_v1_drill: {drill_check.reason}")

    # AC3: v1 routes / image / parser still available
    v1_check = _validate_v1_availability()
    pre_checks.append(v1_check)
    if v1_check.status != "passed":
        residual_risks.append(f"v1_availability: {v1_check.reason}")
    # AC3: deployed image / registry availability cannot be auto-verified
    # from the source tree — always surface for manual sign-off.
    residual_risks.extend(V1_AVAILABILITY_RESIDUAL_RISKS)

    # AC2: new conversations default to v2 (config flag ready)
    config_snapshot = _build_config_snapshot(state_dir, ai_service_dir)
    v2_check = _validate_v2_default()
    pre_checks.append(v2_check)
    if v2_check.status != "passed":
        residual_risks.append(f"v2_default: {v2_check.reason}")
    elif v2_check.reason:
        # AC2: passed but with policy/selector notes that need manual sign-off.
        residual_risks.append(f"v2_default: {v2_check.reason}")

    # AC4: 100% cutover keeps recording the same section-level metrics
    # (quality/performance/cost/dependency/capacity/privacy/security/recovery/
    # user_visible) and the hard-gate evidence path is still reachable.
    metrics_check = _validate_metrics_recording_readiness(canary_report_path)
    pre_checks.append(metrics_check)
    if metrics_check.status != "passed":
        residual_risks.append(f"metrics_recording_readiness: {metrics_check.reason}")
    metrics_recording_ready = metrics_check.status == "passed"

    # AC5: rollback entry exists and is usable
    rollback_check = _validate_rollback_entry(rollback_entry_path)
    pre_checks.append(rollback_check)
    evidence_paths.append(str(rollback_entry_path))
    if rollback_check.status != "passed":
        residual_risks.append(f"rollback_entry: {rollback_check.reason}")

    overall_pass = all(check.status == "passed" for check in pre_checks)

    release_record = _build_release_record(
        operator=operator,
        acceptance_report_path=acceptance_report_path,
        backup_report=backup_report,
        canary_report_path=canary_report_path,
        rollback_entry_path=rollback_entry_path,
        v1_availability_check=v1_check,
        v2_default_check=v2_check,
        config_snapshot=config_snapshot,
        residual_risks=residual_risks,
        overall_pass=overall_pass,
        metrics_recording_ready=metrics_recording_ready,
    )

    # AC6: write release record (no secrets)
    output_path = output or (
        ai_service_dir / "data" / "agent-v2-cutover-release-record.json"
    )
    _write_release_record(release_record, output_path)
    evidence_paths.append(str(output_path))

    report = CutoverReport(
        timestamp=timestamp,
        pre_checks=pre_checks,
        release_record=release_record,
        overall_pass=overall_pass,
        executed_commands=executed_commands,
        evidence_paths=evidence_paths,
        residual_risks=residual_risks,
    )
    # Write the full cutover report next to the release record.
    report_path = output_path.parent / "agent-v2-cutover-report.json"
    _write_json(report_path, asdict(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ticket 21 Chat v2 cutover coordinator"
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(getattr(settings, "agent_v2_state_dir", "./agent_state")),
    )
    parser.add_argument(
        "--acceptance-report",
        type=Path,
        required=True,
        help="Path to the latest acceptance report JSON",
    )
    parser.add_argument(
        "--canary-report",
        type=Path,
        required=True,
        help="Path to the canary rollout report JSON (force-v1 drill evidence)",
    )
    parser.add_argument(
        "--operator",
        required=True,
        help="Operator identifier recorded in the release record",
    )
    parser.add_argument(
        "--rollback-entry",
        type=Path,
        required=True,
        help="Path to the rollback entry JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit release record output path",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Optional explicit backup root directory",
    )
    args = parser.parse_args(argv)

    report = run_cutover_validation(
        state_dir=args.state_dir,
        acceptance_report_path=args.acceptance_report,
        canary_report_path=args.canary_report,
        operator=args.operator,
        rollback_entry_path=args.rollback_entry,
        output=args.output,
        backup_dir=args.backup_dir,
    )

    print(f"Chat v2 cutover overall_pass: {report.overall_pass}")
    for check in report.pre_checks:
        print(f"  {check.check_id}: {check.status}")
    if report.residual_risks:
        print("residual_risks:")
        for risk in report.residual_risks:
            print(f"  - {risk}")
    return 0 if report.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
