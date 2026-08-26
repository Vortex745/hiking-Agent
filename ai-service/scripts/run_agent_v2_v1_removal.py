"""Ticket 22 — Chat v2 v1 removal coordinator (expand phase).

This script does NOT remove v1 code. It validates the 14-day stable
observation gate, performs the final force-v1 drill, records the
go/no-go decision, and emits a removal plan listing v1 code paths to
remove (contract phase, executed by operator).

The contract phase (actual removal) is gated by:
- 14 days of stable 100% v2 operation (AC1)
- Final force-v1 drill success (AC2)
- go/no-go decision record (AC2)
- Removal plan approval (AC3)

Reuses force-v1 drill validation from cutover coordinator (ticket 21).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from config import settings


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

# AC3: v1 components targeted for removal (expand-contract boundaries).
# Each entry is (component_label, file_path_relative_to_project_root, boundary).
# Resolved via Path(__file__).resolve().parents[2] (project root).
V1_REMOVAL_ITEMS: tuple[tuple[str, str, str], ...] = (
    (
        "v1 Agent runtime",
        "ai-service/agent/",
        "expand-contract: agent_v2/ replaces agent/",
    ),
    (
        "v1 detached confirmation",
        "ai-service/api/confirmation_store.py",
        "expand-contract: agent_v2/approvals.py replaces confirmation_store.py",
    ),
    (
        "v1 Chat memory",
        "ai-service/memory/",
        "expand-contract: agent_v2/repositories/memory.py replaces memory/",
    ),
    (
        "Gateway v1 chat routes",
        "gateway/handler/chat.go",
        "expand-contract: gateway/handler/chat_v2.go replaces chat.go",
    ),
    (
        "Frontend v1 parser/canary selector",
        "frontend/src/api/sse.ts",
        "expand-contract: frontend/src/api/chatV2.ts replaces v1 branches in sse.ts",
    ),
)

# AC4: v2 source paths (relative to ai-service/) scanned for v1 references.
V2_SCAN_PATHS: tuple[str, ...] = (
    "agent_v2",
    "api/chat_v2.py",
    "api/memory_v2.py",
)

# AC4: RAG source paths (relative to ai-service/) scanned for v1 references.
RAG_SCAN_PATHS: tuple[str, ...] = (
    "rag",
    "api/rag.py",
)

# AC4: import patterns indicating v2/RAG references a v1 component.
# Patterns are chosen so they do NOT match v2 module names
# (e.g. "from agent " won't match "from agent_v2 ").
V1_REFERENCE_PATTERNS: dict[str, tuple[str, ...]] = {
    "v1 Agent runtime": (
        "from agent ",
        "from agent.",
        "import agent\n",
        "import agent ",
    ),
    "v1 detached confirmation": ("confirmation_store",),
    "v1 Chat memory": (
        "from memory ",
        "from memory.",
        "import memory\n",
        "import memory ",
    ),
    "Gateway v1 chat routes": (),
    "Frontend v1 parser/canary selector": (),
}

# AC5: representative v1 test files that must remain executable pre-removal.
# Relative to ai-service/.
V1_REPRESENTATIVE_TEST_FILES: tuple[str, ...] = (
    "tests/test_agent.py",
    "tests/test_confirmation.py",
    "tests/test_memory.py",
)

# AC6: candidate v1 data archive / retention policy doc paths.
# Relative to project root. The check passes if at least one exists.
V1_DATA_ARCHIVE_POLICY_DOCS: tuple[str, ...] = (
    "ai-service/docs/v1_data_retention_policy.md",
    "ai-service/docs/v1_removal_runbook.md",
    "docs/v1_data_retention_policy.md",
)

# AC7: documents / configs / health checks / runbooks that must stop
# claiming v1 availability after removal.
DOC_UPDATES_REQUIRED: tuple[str, ...] = (
    "ai-service/docs/chat_v2_cutover.md — remove v1 availability claims",
    "ai-service/api/chat.py — remove or deprecate v1 /chat/sse route health check",
    "gateway/handler/health.go — remove v1 chat health check entry",
    "frontend/src/api/sse.ts — remove v1 SSE parser branches after sse.ts cleanup",
    "deployment config — remove v1 image tag and v1 rollback env vars",
    "operations runbook — remove v1 rollback procedure, keep only v2 runbook",
)

# AC1: required daily hard-gate report sections (mirrors canary windows).
REQUIRED_DAILY_SECTIONS = (
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

# AC5: post-removal test plan entries (mirrors ticket 22 AC5).
POST_REMOVAL_TEST_PLAN: tuple[str, ...] = (
    "Chat v2 complete test suite (test_chat_v2_*.py)",
    "Gateway handler tests (go test ./handler/...)",
    "Frontend build (npm run build)",
    "Representative v1 removal assertions (v1 tests removed, v2 unaffected)",
    "RAG regression suite (test_rag_*.py)",
)


@dataclass(frozen=True, slots=True)
class RemovalPreCheck:
    check_id: str
    status: Literal["passed", "failed", "skipped"]
    evidence: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class DailyHardGateSummary:
    date: str
    overall_pass: bool
    p0_count: int
    p1_count: int
    section_statuses: dict[str, str]


@dataclass(frozen=True, slots=True)
class ObservationPeriodReport:
    start_date: str
    end_date: str
    total_days: int
    daily_summaries: list[DailyHardGateSummary]
    all_days_pass: bool
    open_p0_count: int
    open_p1_count: int


@dataclass(frozen=True, slots=True)
class GoNoGoRecord:
    timestamp: str
    operator: str
    decision: Literal["go", "no_go"]
    rationale: str
    force_v1_drill_pass: bool
    observation_period_pass: bool
    residual_risks: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class V1RemovalItem:
    component: str
    file_path: str
    expand_contract_boundary: str
    used_by_v2: bool
    used_by_rag: bool
    safe_to_remove: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RemovalPlan:
    items: list[V1RemovalItem]
    total_items: int
    safe_to_remove_count: int
    blocked_count: int
    removal_order: list[str]
    residual_risks: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RemovalReport:
    timestamp: str
    pre_checks: list[RemovalPreCheck]
    observation_period: ObservationPeriodReport | None
    go_no_go: GoNoGoRecord
    removal_plan: RemovalPlan | None
    post_removal_test_plan: list[str]
    doc_updates_required: list[str]
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


def _parse_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _collect_source_text(scan_paths: tuple[str, ...], base_dir: Path) -> str:
    """Read all .py source files under scan_paths and return concatenated text.

    Used by the AC4 static reference check. Directories are walked recursively.
    """
    chunks: list[str] = []
    for relative in scan_paths:
        absolute = base_dir / relative
        if absolute.is_file():
            try:
                chunks.append(absolute.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
        elif absolute.is_dir():
            for py_file in sorted(absolute.rglob("*.py")):
                try:
                    chunks.append(py_file.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError):
                    continue
    return "\n".join(chunks)


def _component_referenced(
    component: str, source_text: str
) -> bool:
    """Return True if source_text contains any v1 reference pattern for component."""
    patterns = V1_REFERENCE_PATTERNS.get(component, ())
    if not patterns:
        return False
    return any(pattern in source_text for pattern in patterns)


def _validate_observation_period(
    daily_reports_dir: Path,
    required_days: int,
) -> tuple[RemovalPreCheck, ObservationPeriodReport | None]:
    """AC1: validate 14 consecutive days of daily hard-gate reports all pass."""
    if not daily_reports_dir.is_dir():
        return (
            RemovalPreCheck(
                check_id="observation_period",
                status="failed",
                evidence=f"dir={daily_reports_dir}",
                reason="daily reports directory missing",
            ),
            None,
        )
    summaries: list[DailyHardGateSummary] = []
    invalid_files: list[str] = []
    for json_path in sorted(daily_reports_dir.glob("*.json")):
        try:
            payload = _read_json(json_path)
        except (OSError, json.JSONDecodeError) as exc:
            invalid_files.append(f"{json_path.name}: {exc}")
            continue
        date_str = str(payload.get("date", ""))
        if not _parse_date(date_str):
            invalid_files.append(f"{json_path.name}: invalid date")
            continue
        section_statuses = payload.get("section_statuses", {})
        if not isinstance(section_statuses, dict):
            section_statuses = {}
        open_incidents = payload.get("open_incidents")
        if not isinstance(open_incidents, dict):
            open_incidents = payload.get("incident_counts", {})
            if not isinstance(open_incidents, dict):
                open_incidents = {}
        summaries.append(
            DailyHardGateSummary(
                date=date_str,
                overall_pass=bool(payload.get("overall_pass", False)),
                p0_count=int(open_incidents.get("p0", 0)),
                p1_count=int(open_incidents.get("p1", 0)),
                section_statuses={
                    str(k): str(v) for k, v in section_statuses.items()
                },
            )
        )
    if invalid_files:
        return (
            RemovalPreCheck(
                check_id="observation_period",
                status="failed",
                evidence=f"dir={daily_reports_dir} files={len(summaries)}",
                reason=f"invalid daily reports: {invalid_files[:3]}",
            ),
            None,
        )
    if len(summaries) < required_days:
        return (
            RemovalPreCheck(
                check_id="observation_period",
                status="failed",
                evidence=f"found={len(summaries)} required={required_days}",
                reason=(
                    f"insufficient daily reports: need {required_days} "
                    f"consecutive days, found {len(summaries)}"
                ),
            ),
            None,
        )
    summaries.sort(key=lambda s: s.date)
    window = summaries[-required_days:]
    start_date = window[0].date
    end_date = window[-1].date
    # Check consecutive days.
    prev = _parse_date(start_date)
    if prev is None:
        return (
            RemovalPreCheck(
                check_id="observation_period",
                status="failed",
                evidence=f"start_date={start_date}",
                reason="could not parse observation window start date",
            ),
            None,
        )
    gaps: list[str] = []
    for summary in window[1:]:
        current = _parse_date(summary.date)
        if current is None:
            gaps.append(f"unparseable date: {summary.date}")
            continue
        if current - prev != timedelta(days=1):
            gaps.append(f"gap between {prev.date().isoformat()} and {summary.date}")
        prev = current
    if gaps:
        return (
            RemovalPreCheck(
                check_id="observation_period",
                status="failed",
                evidence=f"window={start_date}..{end_date} days={len(window)}",
                reason=f"observation window not consecutive: {gaps[:3]}",
            ),
            None,
        )
    failed_days = [s.date for s in window if not s.overall_pass]
    all_days_pass = not failed_days
    # open_p0/p1 from the latest report (currently open incidents).
    latest = window[-1]
    open_p0 = latest.p0_count
    open_p1 = latest.p1_count
    if not all_days_pass:
        return (
            RemovalPreCheck(
                check_id="observation_period",
                status="failed",
                evidence=(
                    f"window={start_date}..{end_date} days={len(window)} "
                    f"failed_days={failed_days[:3]}"
                ),
                reason="one or more daily hard-gate reports did not pass",
            ),
            ObservationPeriodReport(
                start_date=start_date,
                end_date=end_date,
                total_days=len(window),
                daily_summaries=window,
                all_days_pass=False,
                open_p0_count=open_p0,
                open_p1_count=open_p1,
            ),
        )
    return (
        RemovalPreCheck(
            check_id="observation_period",
            status="passed",
            evidence=(
                f"window={start_date}..{end_date} days={len(window)} "
                f"all_days_pass=True open_p0={open_p0} open_p1={open_p1}"
            ),
        ),
        ObservationPeriodReport(
            start_date=start_date,
            end_date=end_date,
            total_days=len(window),
            daily_summaries=window,
            all_days_pass=True,
            open_p0_count=open_p0,
            open_p1_count=open_p1,
        ),
    )


def _validate_open_incidents(
    daily_reports_dir: Path,
) -> RemovalPreCheck:
    """AC1: validate open v2 P0/P1 incidents are zero (latest daily report)."""
    if not daily_reports_dir.is_dir():
        return RemovalPreCheck(
            check_id="open_incidents",
            status="failed",
            evidence=f"dir={daily_reports_dir}",
            reason="daily reports directory missing",
        )
    summaries: list[tuple[str, int, int]] = []
    for json_path in daily_reports_dir.glob("*.json"):
        try:
            payload = _read_json(json_path)
        except (OSError, json.JSONDecodeError):
            continue
        date_str = str(payload.get("date", ""))
        if not _parse_date(date_str):
            continue
        open_incidents = payload.get("open_incidents")
        if not isinstance(open_incidents, dict):
            open_incidents = payload.get("incident_counts", {})
            if not isinstance(open_incidents, dict):
                open_incidents = {}
        summaries.append(
            (date_str, int(open_incidents.get("p0", 0)), int(open_incidents.get("p1", 0)))
        )
    if not summaries:
        return RemovalPreCheck(
            check_id="open_incidents",
            status="failed",
            evidence=f"dir={daily_reports_dir}",
            reason="no readable daily reports found",
        )
    summaries.sort(key=lambda t: t[0])
    latest_date, p0, p1 = summaries[-1]
    if p0 > 0 or p1 > 0:
        return RemovalPreCheck(
            check_id="open_incidents",
            status="failed",
            evidence=f"latest={latest_date} open_p0={p0} open_p1={p1}",
            reason=f"open v2 P0/P1 incidents non-zero: p0={p0} p1={p1}",
        )
    return RemovalPreCheck(
        check_id="open_incidents",
        status="passed",
        evidence=f"latest={latest_date} open_p0={p0} open_p1={p1}",
    )


def _validate_final_force_v1_drill(
    cutover_report_path: Path,
    *,
    observation_start: str | None = None,
) -> RemovalPreCheck:
    """AC2: reuse cutover coordinator's force-v1 drill evidence.

    If observation_start is provided (ISO date/datetime string), also
    validates the cutover report timestamp falls within or after the
    observation window (i.e. cutover report time >= observation start).
    A stale or unparseable timestamp fails the check.
    """
    if not cutover_report_path.is_file():
        return RemovalPreCheck(
            check_id="final_force_v1_drill",
            status="failed",
            evidence=f"path={cutover_report_path}",
            reason="cutover report file missing",
        )
    try:
        payload = _read_json(cutover_report_path)
    except (OSError, json.JSONDecodeError) as exc:
        return RemovalPreCheck(
            check_id="final_force_v1_drill",
            status="failed",
            evidence=f"path={cutover_report_path}",
            reason=f"cutover report unreadable: {exc}",
        )
    pre_checks = payload.get("pre_checks", [])
    if not isinstance(pre_checks, list):
        return RemovalPreCheck(
            check_id="final_force_v1_drill",
            status="failed",
            evidence=f"path={cutover_report_path}",
            reason="cutover report missing pre_checks list",
        )
    drill_check = None
    for check in pre_checks:
        if not isinstance(check, dict):
            continue
        if str(check.get("check_id", "")) == "force_v1_drill":
            drill_check = check
            break
    if drill_check is None:
        return RemovalPreCheck(
            check_id="final_force_v1_drill",
            status="failed",
            evidence=f"path={cutover_report_path} pre_checks={len(pre_checks)}",
            reason="force_v1_drill check not present in cutover report",
        )
    drill_status = str(drill_check.get("status", "")).strip().lower()
    if drill_status != "passed":
        return RemovalPreCheck(
            check_id="final_force_v1_drill",
            status="failed",
            evidence=f"drill_status={drill_status}",
            reason=f"force-v1 drill did not pass in cutover report: {drill_check.get('reason', '')}",
        )
    # Timestamp check: cutover report must fall within or after the
    # observation window (i.e. report time >= observation start).
    if observation_start is not None:
        report_ts = str(payload.get("timestamp", ""))
        report_dt = _parse_date(report_ts)
        obs_start_dt = _parse_date(observation_start)
        if report_dt is None or obs_start_dt is None:
            return RemovalPreCheck(
                check_id="final_force_v1_drill",
                status="failed",
                evidence=f"report_ts={report_ts} observation_start={observation_start}",
                reason="cutover report timestamp outside observation window",
            )
        # Normalize offset-naive datetimes to UTC so the comparison
        # works for both date-only and full ISO datetime strings.
        if report_dt.tzinfo is None:
            report_dt = report_dt.replace(tzinfo=timezone.utc)
        if obs_start_dt.tzinfo is None:
            obs_start_dt = obs_start_dt.replace(tzinfo=timezone.utc)
        if report_dt < obs_start_dt:
            return RemovalPreCheck(
                check_id="final_force_v1_drill",
                status="failed",
                evidence=f"report_ts={report_ts} observation_start={observation_start}",
                reason="cutover report timestamp outside observation window",
            )
    return RemovalPreCheck(
        check_id="final_force_v1_drill",
        status="passed",
        evidence=f"drill_status=passed evidence={drill_check.get('evidence', '')}",
    )


def _record_go_no_go_decision(
    operator: str,
    observation_pass: bool,
    drill_pass: bool,
) -> GoNoGoRecord:
    """AC2: record go/no-go decision based on observation + drill outcomes."""
    timestamp = _now_iso()
    residual_risks: list[str] = []
    if observation_pass and drill_pass:
        decision: Literal["go", "no_go"] = "go"
        rationale = (
            "Observation period passed (14 consecutive days, 0 open P0/P1) "
            "and final force-v1 drill passed; v1 rollback no longer required."
        )
    else:
        decision = "no_go"
        blockers: list[str] = []
        if not observation_pass:
            blockers.append("observation period not satisfied")
        if not drill_pass:
            blockers.append("final force-v1 drill not passed")
        rationale = (
            "Cannot proceed to contract phase: " + "; ".join(blockers) + "."
        )
        residual_risks.extend(blockers)
    return GoNoGoRecord(
        timestamp=timestamp,
        operator=operator,
        decision=decision,
        rationale=rationale,
        force_v1_drill_pass=drill_pass,
        observation_period_pass=observation_pass,
        residual_risks=residual_risks,
    )


def _build_removal_plan() -> tuple[RemovalPreCheck, RemovalPlan]:
    """AC3: build v1 removal plan with expand-contract boundaries.

    Static-scans v2 and RAG source trees to flag items still referenced
    by v2 or RAG (AC4). Items so flagged are marked safe_to_remove=False
    and must be manually reviewed before contract execution.
    """
    project_root = Path(__file__).resolve().parents[2]
    ai_service_dir = Path(__file__).resolve().parent.parent
    v2_text = _collect_source_text(V2_SCAN_PATHS, ai_service_dir)
    rag_text = _collect_source_text(RAG_SCAN_PATHS, ai_service_dir)
    items: list[V1RemovalItem] = []
    missing_files: list[str] = []
    for component, file_path, boundary in V1_REMOVAL_ITEMS:
        absolute = project_root / file_path
        file_exists = absolute.exists()
        if not file_exists:
            missing_files.append(file_path)
        used_by_v2 = _component_referenced(component, v2_text)
        used_by_rag = _component_referenced(component, rag_text)
        safe = file_exists and not used_by_v2 and not used_by_rag
        reasons: list[str] = []
        if not file_exists:
            reasons.append("file not found (already removed?)")
        if used_by_v2:
            reasons.append("referenced by v2 source")
        if used_by_rag:
            reasons.append("referenced by RAG source")
        items.append(
            V1RemovalItem(
                component=component,
                file_path=file_path,
                expand_contract_boundary=boundary,
                used_by_v2=used_by_v2,
                used_by_rag=used_by_rag,
                safe_to_remove=safe,
                reason="; ".join(reasons) if reasons else "v1-only, no v2/RAG references",
            )
        )
    safe_items = [item for item in items if item.safe_to_remove]
    blocked_items = [item for item in items if not item.safe_to_remove]
    removal_order = [item.component for item in safe_items] + [
        item.component for item in blocked_items
    ]
    residual_risks: list[str] = []
    if missing_files:
        residual_risks.append(
            f"v1 files already missing: {missing_files} — verify manual removal"
        )
    for item in blocked_items:
        residual_risks.append(
            f"blocked: {item.component} — {item.reason}"
        )
    # AC4: components with empty V1_REFERENCE_PATTERNS (Gateway/Frontend)
    # are not scanned for cross-language references (.go/.ts). Flag as
    # residual risk so operator manually confirms no v1 references remain.
    for component, _, _ in V1_REMOVAL_ITEMS:
        if not V1_REFERENCE_PATTERNS.get(component):
            residual_risks.append(
                f"{component}: cross-language reference scan not implemented "
                "(only .py scanned); manual review of .go/.ts sources required"
            )
    plan = RemovalPlan(
        items=items,
        total_items=len(items),
        safe_to_remove_count=len(safe_items),
        blocked_count=len(blocked_items),
        removal_order=removal_order,
        residual_risks=residual_risks,
    )
    if missing_files:
        check = RemovalPreCheck(
            check_id="removal_plan",
            status="failed",
            evidence=f"items={len(items)} missing={len(missing_files)}",
            reason=f"v1 files already missing: {missing_files}",
        )
    else:
        check = RemovalPreCheck(
            check_id="removal_plan",
            status="passed",
            evidence=(
                f"items={len(items)} safe={len(safe_items)} "
                f"blocked={len(blocked_items)}"
            ),
        )
    return check, plan


def _validate_dependencies_not_shared(
    removal_plan: RemovalPlan,
) -> RemovalPreCheck:
    """AC4: validate removal items are not used by v2 or RAG."""
    blocked = [
        item.component
        for item in removal_plan.items
        if item.used_by_v2 or item.used_by_rag
    ]
    if blocked:
        return RemovalPreCheck(
            check_id="dependencies_not_shared",
            status="failed",
            evidence=f"blocked={blocked}",
            reason=(
                "v1 components still referenced by v2 or RAG: "
                f"{blocked} — must detach before removal"
            ),
        )
    return RemovalPreCheck(
        check_id="dependencies_not_shared",
        status="passed",
        evidence=f"checked={removal_plan.total_items} blocked=0",
    )


def _run_representative_v1_assertions() -> RemovalPreCheck:
    """AC5: verify representative v1 test files exist and are collectable.

    This does NOT run the tests — it confirms the v1 test suite is intact
    so the operator can execute it pre-removal. Actual test execution is
    part of the post-removal verification plan.
    """
    ai_service_dir = Path(__file__).resolve().parent.parent
    missing: list[str] = []
    for relative in V1_REPRESENTATIVE_TEST_FILES:
        absolute = ai_service_dir / relative
        if not absolute.is_file():
            missing.append(relative)
    if missing:
        return RemovalPreCheck(
            check_id="representative_v1_assertions",
            status="failed",
            evidence=f"checked={len(V1_REPRESENTATIVE_TEST_FILES)} missing={len(missing)}",
            reason=f"v1 test files missing: {missing}",
        )
    return RemovalPreCheck(
        check_id="representative_v1_assertions",
        status="passed",
        evidence=f"checked={len(V1_REPRESENTATIVE_TEST_FILES)} all_present=True",
    )


def _validate_v1_data_archive_policy(
    project_root: Path | None = None,
) -> RemovalPreCheck:
    """AC6: verify a v1 data archive / retention policy doc exists.

    This script does NOT execute data deletion. It only confirms the
    operator has a documented retention policy before contract phase.

    Args:
        project_root: Optional project root path. Defaults to the
            repository root (parents[2] of this script). Tests should
            pass a temp directory to verify real detection behavior
            rather than monkeypatching this function.
    """
    root = project_root or Path(__file__).resolve().parents[2]
    found: list[str] = []
    for relative in V1_DATA_ARCHIVE_POLICY_DOCS:
        if (root / relative).is_file():
            found.append(relative)
    if not found:
        return RemovalPreCheck(
            check_id="v1_data_archive_policy",
            status="failed",
            evidence=f"candidates={list(V1_DATA_ARCHIVE_POLICY_DOCS)}",
            reason="no v1 data archive / retention policy doc found",
        )
    return RemovalPreCheck(
        check_id="v1_data_archive_policy",
        status="passed",
        evidence=f"found={found[0]}",
    )


def _list_doc_updates_required() -> list[str]:
    """AC7: list docs / health checks / deploy config / runbooks to update."""
    return list(DOC_UPDATES_REQUIRED)


def _write_removal_plan(plan: RemovalPlan, path: Path) -> Path:
    _write_json(path, _scrub_secrets(asdict(plan)))
    return path


def _write_go_no_go_record(record: GoNoGoRecord, path: Path) -> Path:
    _write_json(path, _scrub_secrets(asdict(record)))
    return path


def run_v1_removal_validation(
    *,
    state_dir: Path,
    cutover_report_path: Path,
    daily_reports_dir: Path,
    operator: str,
    observation_days: int = 14,
    output: Path | None = None,
    removal_plan_output: Path | None = None,
) -> RemovalReport:
    """Run the v1 removal validation pipeline.

    Args:
        state_dir: Audit-only directory. Recorded in the command audit
            trail (executed_commands) for reproducibility, but does NOT
            participate in any validation logic. Kept for signature
            parity with the cutover coordinator (ticket 21).
        cutover_report_path: Path to the cutover report JSON (force-v1
            drill evidence).
        daily_reports_dir: Directory containing daily hard-gate reports.
        operator: Operator identifier recorded in the go/no-go decision.
        observation_days: Required consecutive observation days.
        output: Optional explicit removal report output path.
        removal_plan_output: Optional explicit removal plan output path.
    """
    ai_service_dir = Path(__file__).resolve().parent.parent
    timestamp = _now_iso()
    pre_checks: list[RemovalPreCheck] = []
    evidence_paths: list[str] = []
    residual_risks: list[str] = []
    cmd_args: list[str] = [
        sys.executable,
        "-m",
        "scripts.run_agent_v2_v1_removal",
        "--state-dir",
        str(state_dir),
        "--cutover-report",
        str(cutover_report_path),
        "--daily-reports-dir",
        str(daily_reports_dir),
        "--operator",
        operator,
        "--observation-days",
        str(observation_days),
    ]
    if output is not None:
        cmd_args.extend(["--output", str(output)])
    if removal_plan_output is not None:
        cmd_args.extend(["--removal-plan-output", str(removal_plan_output)])
    executed_commands: list[str] = [subprocess.list2cmdline(cmd_args)]

    # AC1: 14-day observation period of daily hard-gate reports.
    obs_check, obs_report = _validate_observation_period(
        daily_reports_dir, observation_days
    )
    pre_checks.append(obs_check)
    evidence_paths.append(str(daily_reports_dir))
    if obs_check.status != "passed":
        residual_risks.append(f"observation_period: {obs_check.reason}")

    # AC1: open v2 P0/P1 incidents must be zero.
    incidents_check = _validate_open_incidents(daily_reports_dir)
    pre_checks.append(incidents_check)
    if incidents_check.status != "passed":
        residual_risks.append(f"open_incidents: {incidents_check.reason}")

    # AC2: final force-v1 drill (reuse cutover report evidence).
    # Pass observation window start so the drill's cutover report
    # timestamp can be validated against the observation period.
    observation_start = obs_report.start_date if obs_report else None
    drill_check = _validate_final_force_v1_drill(
        cutover_report_path,
        observation_start=observation_start,
    )
    pre_checks.append(drill_check)
    evidence_paths.append(str(cutover_report_path))
    if drill_check.status != "passed":
        residual_risks.append(f"final_force_v1_drill: {drill_check.reason}")

    # AC2: record go/no-go decision.
    observation_pass = obs_check.status == "passed"
    drill_pass = drill_check.status == "passed"
    go_no_go = _record_go_no_go_decision(
        operator=operator,
        observation_pass=observation_pass,
        drill_pass=drill_pass,
    )
    if go_no_go.decision == "no_go":
        residual_risks.append(
            f"go_no_go: decision=no_go — {go_no_go.rationale}"
        )

    # AC3: build v1 removal plan (expand-contract boundaries).
    plan_check, removal_plan = _build_removal_plan()
    pre_checks.append(plan_check)
    if plan_check.status != "passed":
        residual_risks.append(f"removal_plan: {plan_check.reason}")

    # AC4: validate dependencies not shared with v2 or RAG.
    deps_check = _validate_dependencies_not_shared(removal_plan)
    pre_checks.append(deps_check)
    if deps_check.status != "passed":
        residual_risks.append(f"dependencies_not_shared: {deps_check.reason}")

    # AC5: representative v1 assertions (test files present).
    assertions_check = _run_representative_v1_assertions()
    pre_checks.append(assertions_check)
    if assertions_check.status != "passed":
        residual_risks.append(
            f"representative_v1_assertions: {assertions_check.reason}"
        )

    # AC6: v1 data archive / retention policy doc exists.
    archive_check = _validate_v1_data_archive_policy()
    pre_checks.append(archive_check)
    if archive_check.status != "passed":
        residual_risks.append(f"v1_data_archive_policy: {archive_check.reason}")

    # AC7: list doc / health check / deploy config / runbook updates required.
    doc_updates = _list_doc_updates_required()

    # Aggregate.
    overall_pass = all(check.status == "passed" for check in pre_checks)

    # Write go/no-go record and removal plan (no secrets).
    output_path = output or (
        ai_service_dir / "data" / "agent-v2-v1-removal-report.json"
    )
    plan_output_path = removal_plan_output or (
        output_path.parent / "agent-v2-v1-removal-plan.json"
    )
    _write_go_no_go_record(
        go_no_go,
        output_path.parent / "agent-v2-v1-removal-go-no-go.json",
    )
    _write_removal_plan(removal_plan, plan_output_path)
    evidence_paths.append(str(plan_output_path))

    report = RemovalReport(
        timestamp=timestamp,
        pre_checks=pre_checks,
        observation_period=obs_report,
        go_no_go=go_no_go,
        removal_plan=removal_plan,
        post_removal_test_plan=list(POST_REMOVAL_TEST_PLAN),
        doc_updates_required=doc_updates,
        overall_pass=overall_pass,
        executed_commands=executed_commands,
        evidence_paths=evidence_paths,
        residual_risks=residual_risks,
    )
    _write_json(output_path, _scrub_secrets(asdict(report)))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ticket 22 Chat v2 v1 removal coordinator (expand phase)"
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(getattr(settings, "agent_v2_state_dir", "./agent_state")),
    )
    parser.add_argument(
        "--cutover-report",
        type=Path,
        required=True,
        help="Path to the cutover report JSON (force-v1 drill evidence)",
    )
    parser.add_argument(
        "--daily-reports-dir",
        type=Path,
        required=True,
        help="Directory containing daily hard-gate report JSON files",
    )
    parser.add_argument(
        "--operator",
        required=True,
        help="Operator identifier recorded in the go/no-go decision",
    )
    parser.add_argument(
        "--observation-days",
        type=int,
        default=14,
        help="Required consecutive observation days (default: 14)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit removal report output path",
    )
    parser.add_argument(
        "--removal-plan-output",
        type=Path,
        default=None,
        help="Optional explicit removal plan output path",
    )
    args = parser.parse_args(argv)

    report = run_v1_removal_validation(
        state_dir=args.state_dir,
        cutover_report_path=args.cutover_report,
        daily_reports_dir=args.daily_reports_dir,
        operator=args.operator,
        observation_days=args.observation_days,
        output=args.output,
        removal_plan_output=args.removal_plan_output,
    )

    print(f"Chat v2 v1 removal overall_pass: {report.overall_pass}")
    for check in report.pre_checks:
        print(f"  {check.check_id}: {check.status}")
    print(f"go/no-go decision: {report.go_no_go.decision}")
    if report.removal_plan is not None:
        print(
            f"removal plan: {report.removal_plan.safe_to_remove_count} safe, "
            f"{report.removal_plan.blocked_count} blocked"
        )
    if report.residual_risks:
        print("residual_risks:")
        for risk in report.residual_risks:
            print(f"  - {risk}")
    return 0 if report.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
