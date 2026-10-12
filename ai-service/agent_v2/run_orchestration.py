"""Deep run orchestration module for Chat v2 (Tickets 36/37).

Owns the run orchestration previously spread across the Chat v2
HTTP/SSE adapter: stream lifecycle, exactly-one-terminal guarantee,
persist-before-frame event ordering, capacity lease acquisition/release
(with its 503 trigger), idempotent run creation + duplicate replay,
missing-terminal synthesis on reconnect, cancel observation, chat /
route_plan system prompt assembly, and the kernel turn itself.

The adapter (``api/chat_v2.py``) keeps only its transport duties: HTTP
request parsing, Pydantic validation, SSE frame rendering, and status
code mapping. Every dependency is constructor-injected — kernel entry,
RunRepository, capacity guard, telemetry, memory service, settings — so
the orchestration interface is testable in-process with a fake kernel
and real SQLite under a temp directory, without an ASGI shell and
without module-global monkeypatch points.

Since Ticket 41 the host-side artifact export no longer lives in the
adapter — the production implementation is :func:`agent_v2.artifacts.
export_artifact_to_host`. The ``artifact_export`` constructor hook
remains only as a test seam, as do the sandbox computer factory,
cleanup, and the PermissionGate constructor hooks, which resolve the
adapter's globals lazily at call time so existing ASGI contract-test
monkeypatch points keep working.

Since Ticket 38 the cowork flow AND the approval-decision state machine
(approve / reject / expire / duplicate idempotency) live here end-to-end;
the adapter maps decision outcomes to HTTP status codes.

Since Ticket 39 the artifact flow is driven end-to-end by this module,
and the effect CAS ledger is a SINGLE implementation here
(:func:`drive_effect`): the ``authorized → executing → committed/failed/
outcome_unknown`` transitions happen nowhere else. Since Ticket 41 the
host-side verified export also lives in :mod:`agent_v2.artifacts`
(:func:`agent_v2.artifacts.export_artifact_to_host`); the
``artifact_export`` constructor hook remains only as a test seam.

Since Ticket 40 the RESUME path is also driven end-to-end by this
module (:meth:`RunOrchestrator.stream_resume` + the sandbox
re-execution it schedules), together with the snapshot and event
replay reads (:meth:`RunOrchestrator.load_snapshot` /
:meth:`RunOrchestrator.load_events`). Both the cowork bash and the
artifact resume paths share the unified effect ledger; concurrent
resume keeps the sandbox alive when the CAS winner is still executing.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable

from agent_v2.approvals import (
    canonical_args,
    compute_approval_id,
    compute_expiry,
    is_expired,
)
from agent_v2.artifacts import (
    ArtifactEvidencePersistenceError,
    ArtifactExportError,
)
from agent_v2.computer import (
    Computer,
    ComputerError,
    SandboxCreationError,
    SandboxQuotaError,
    SandboxTimeoutError,
)
from agent_v2.effects import (
    EffectStatus,
    compute_effect_key,
    compute_external_idempotency_key,
    deserialize_result,
    serialize_result,
)
from agent_v2.events import (
    TERMINAL_EVENT_TYPES,
    EventEnvelope,
    EventStream,
    TerminalEventAlreadyEmitted,
)
from agent_v2.memory_tools import extract_sensitive_memory
from agent_v2.observability import CapacityExceeded
from agent_v2.permissions import PermissionDecision, PermissionGate
from agent_v2.plan_workflow import (
    BLOCK_MESSAGE,
    SUPPLEMENT_ARRANGEMENT_KEYS,
    PlanDraftStage,
    RecordedCompleteness,
    classify_plan_request,
    draft_summary,
    evaluate_plan_completeness,
    evaluate_plan_intent,
    missing_plan_fields,
)
from agent_v2.profiles import (
    CapabilityProfile,
    ProfileId,
    UnknownProfileError,
    resolve_profile,
)
from agent_v2.runs import (
    ApprovalRecord,
    RunAlreadyTerminal,
    RunRepository,
    RunSnapshot,
)
from agent_v2.startup import is_cowork_authorized

logger = logging.getLogger("ai-service.run_orchestration")


# ---- Types -----------------------------------------------------------------


@dataclass
class ChatFlowRequest:
    """Transport-neutral chat request.

    The adapter builds this from its Pydantic ``ChatV2Request`` so the
    orchestration module never touches HTTP types. The field set is
    exactly what the flows consume; ``current_location`` carries the
    adapter's validated ``CurrentLocationV2`` object (only
    ``model_dump`` is called on it).

    Ticket 01: ``intent`` is the whitelisted structured plan-book intent
    (already validated at the wire boundary) and ``plan_input`` the
    adapter-validated :class:`agent_v2.plan_workflow.PlanDraftInput`.
    Both must be mirrored here or the flow sees ``None`` for every
    request.

    Ticket 03 adds the third plan input the same way: ``route_selection`` is
    the adapter-validated
    :class:`agent_v2.plan_workflow.RouteSelectionInput` — a reference
    (``comparison_id`` + ``candidate_id``) into an existing comparison
    result. Forgetting it here is a silent bug, not a 422: the wire model
    would accept the field and the flow would never see it.

    Ticket 04 adds the last plan input the same way: ``plan_type`` is the
    adapter-validated draft-type *claim* a client may attach to
    ``plan_generic_draft`` — the only value the state machine accepts there.
    What a draft actually IS is never taken from this field: the flow derives it
    from the route basis the comparison leaf re-read, so forgetting the field
    here would only lose the user's choice, never fake a route plan.
    """

    message: str
    request_id: str
    profile_id: str = "chat"
    conversation_id: str | None = None
    current_location: Any | None = None
    intent: str | None = None
    plan_input: Any | None = None
    route_selection: Any | None = None
    plan_type: str | None = None


# Kernel entry: ``agent_v2.kernel.run_chat_turn`` and test fakes with the
# same keyword signature (returns ``(response_text, audit_entries)``).
KernelTurn = Callable[..., Any]

# Emit closure used inside the orchestrator: build + persist + telemetry,
# returning the envelope so the stream can yield it.
EmitEnvelope = Callable[[str, dict[str, Any]], EventEnvelope]

# Transport-neutral emit accepted by :func:`drive_effect`: the module
# flows pass an :class:`EmitEnvelope`, the adapter resume path (until
# Ticket 40) passes its frame-rendering emit. Only the return value
# differs; the ledger never inspects it.
EmitLike = Callable[[str, dict[str, Any]], Any]

# Sandbox factory injected from the adapter (resolves settings-driven
# sandbox type); kept adapter-provided so existing test monkeypatch
# points keep working (lazy global resolution at call time).
ComputerFactory = Callable[..., Computer]

# Best-effort sandbox cleanup: ``(computer, workspace_path, runs_repo,
# run_id) -> None``. Same rationale as ComputerFactory.
SandboxCleanup = Callable[..., None]

# Host-side artifact export hook: ``(**kwargs) -> dict``. Injected from
# the adapter because the export implementation still lives there
# (Ticket 39 transitional seam; Ticket 41 relocates it).
ArtifactExporter = Callable[..., dict[str, Any]]

# PermissionGate constructor hook. Injected from the adapter so ASGI
# contract tests can keep monkeypatching the adapter global.
GateFactory = Callable[[], Any]


@dataclass(frozen=True, slots=True)
class PlanEntry:
    """Deterministic plan-flow entry decision (Ticket 01).

    ``route`` is the only thing the flow trusts:

    * ``intent`` — the client sent a whitelisted structured intent
      (quick-action button, dialog submit, confirm/cancel).
    * ``open_form`` — the rules found an unambiguous plan-book ask in the
      free text, so the dialog opens (spec AC1).
    * ``ask_confirmation`` — the text is planning-shaped but ambiguous:
      the server asks first and the dialog stays closed (spec AC2).

    ``source`` records whether a structured intent or the rule classifier
    produced the entry, which is what the draft-state event carries for
    audit.
    """

    route: Literal["intent", "open_form", "ask_confirmation"]
    source: Literal["structured_intent", "natural_language"]
    intent: str | None = None


#: Profiles a *structured* plan intent may enter the plan flow from. ``chat``
#: is ticket 01's entry; ``route_plan`` is ticket 03's widening: the route
#: comparison the user selects from is produced by a ``route_plan`` run, so a
#: "select this route into my plan draft" click arrives with that profile.
#: Without it the plan intent would fall through to the kernel, which is the
#: one thing the plan path must never do (spec AC4/AC5).
_PLAN_INTENT_PROFILES: frozenset[str] = frozenset({
    ProfileId.CHAT.value,
    ProfileId.ROUTE_PLAN.value,
})


def _resolve_plan_entry(req: ChatFlowRequest, profile: CapabilityProfile) -> PlanEntry | None:
    """Decide whether this request belongs to the plan flow — rules only.

    Returns ``None`` for everything else. Natural-language entry stays
    ``chat``-only: the plan entry must never hijack a route_plan / rag /
    artifact / cowork run on the strength of a sentence, and it must never be
    decided by a model.

    A *whitelisted structured intent* is honoured on ``chat`` and ``route_plan``
    only (ticket 03). The client cannot use this to skip the kernel for a
    route question: :data:`_PLAN_INTENT_PROFILES` is server-side and the intent
    set is the :data:`agent_v2.plan_workflow.PLAN_INTENTS` whitelist, so any
    other profile_id keeps its normal flow.
    """
    if req.intent and profile.profile_id.value in _PLAN_INTENT_PROFILES:
        return PlanEntry("intent", "structured_intent", req.intent)

    if profile.profile_id != ProfileId.CHAT:
        return None

    outcome = classify_plan_request(req.message).outcome
    if outcome == "clear_plan_intent":
        return PlanEntry("open_form", "natural_language", "plan_create")
    if outcome == "ambiguous":
        return PlanEntry("ask_confirmation", "natural_language")
    return None


@dataclass
class ApprovalDecisionOutcome:
    """Transport-neutral result of an approval decision (Ticket 38).

    ``status`` is one of:

    * ``recorded`` — decision persisted, run resumed (HTTP 200)
    * ``idempotent`` — duplicate of an already-recorded decision with
      the same ``decision_request_id`` (HTTP 200)
    * ``approval_not_found`` (HTTP 404)
    * ``already_resolved`` / ``expired`` / ``run_not_found`` /
      ``run_terminal`` / ``run_not_paused`` / ``sandbox_lease_lost``
      (HTTP 409, ``payload`` carries the public reason + message)

    The payload is the exact public response body minus the
    ``resume_url``, which is an adapter (HTTP surface) concern.
    """

    status: str
    payload: dict[str, Any]


async def _production_kernel_turn(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
    """Real kernel entry; imported lazily as in the previous adapter code."""
    from agent_v2.kernel import run_chat_turn  # heavy import stays lazy

    return await run_chat_turn(**kwargs)


# ---- Terminal-status helpers -----------------------------------------------
#
# Shared with the adapter (artifact / cowork flows and the resume path
# still call them) until Tickets 38-40 fold those flows into this module.


def is_cancelled(runs_repo: RunRepository | None, run_id: str | None) -> bool:
    """Poll the cancel state. Returns False when persistence is off."""
    if runs_repo is None or run_id is None:
        return False
    try:
        return runs_repo.is_cancelled(run_id)
    except Exception:
        logger.exception("chat_v2: is_cancelled check failed; treating as not cancelled")
        return False


def mark_terminal_safe(
    runs_repo: RunRepository | None,
    run_id: str | None,
    terminal: str,
) -> None:
    """Best-effort DB status transition. Swallows RunAlreadyTerminal.

    The stream guarantees exactly one terminal event on the wire; the DB
    status is the audit trail. When the cancel API beat us to the
    transition, ``RunAlreadyTerminal`` is expected and logged, not raised.
    """
    if runs_repo is None or run_id is None:
        return
    try:
        if terminal == "done":
            runs_repo.mark_run_done(run_id)
        elif terminal == "error":
            runs_repo.mark_run_error(run_id)
        # ``cancelled`` is set by the cancel API; the stream does not
        # call mark_run_cancelled because the DB row already reflects it.
    except Exception as exc:
        logger.warning(
            "chat_v2: mark_run_%s failed for run_id=%s: %s",
            terminal,
            run_id,
            exc,
        )


def synthesize_terminal_type(snapshot: RunSnapshot | None) -> str | None:
    """Map a run snapshot's DB status to an SSE terminal event type.

    Used when a reconnect observes stored events without a terminal: the
    run's DB status determines which terminal the client should see.
    Returns ``None`` when the run is still ``running`` (no synthesis
    possible — the client must wait or poll /snapshot).
    """
    if snapshot is None:
        return None
    if snapshot.status == "done":
        return "done"
    if snapshot.status == "error":
        return "error"
    if snapshot.status == "cancelled":
        return "cancelled"
    return None


# ---- Unified effect CAS ledger (Ticket 39) ----------------------------------
#
# The SINGLE implementation of the side-effect ledger state machine:
# ``authorized → executing → committed | failed | outcome_unknown``.
# Before Ticket 39 the resume paths for ``generate_pdf`` and
# ``sandbox_bash`` mirrored this logic in two ~200-line copies inside the
# adapter; both now call :func:`drive_effect` with a tool-specific
# :class:`EffectCopy`. Crash semantics (spec AC3): an effect found in
# ``executing`` with no live executor transitions to ``outcome_unknown``
# and the run terminates — no automatic re-execution, human review
# required. A live executor (recent persisted tool_call) is left alone
# (CAS-loser semantics, Ticket 12 review H1).

# Artifact profile plan filename (was adapter-global ``_ARTIFACT_FILENAME``).
_ARTIFACT_FILENAME = "hiking-plan.pdf"


def _pdf_plan_status_inputs(runs_repo: Any, conversation_id: str) -> dict[str, Any]:
    """The plan side of the PDF cover, read from the rows the plan flow wrote.

    Ticket 08's layout states a status, so this reads only what is on record and
    each read is best-effort — a row that is not there leaves the cover line at
    待核验 instead of becoming a claim:

    * 草案类型 is ``route_plan`` because the book is built from a comparison the
      gate has just re-run (``render_report`` is called before this at both
      sites); it is the one thing the document itself can evidence;
    * the rule verdict counts only while it belongs to the draft on record, so a
      stale row cannot describe a newer plan (ticket 07's discipline);
    * the arrangement record is merged over ``plan_details`` by the document
      builder, which is where an absent field stays 待确认.

    Nothing here reads or implies ``report_allowed`` — eligibility stays
    :func:`route_compare.service.render_report`'s alone. The safety record has no
    row yet (ticket 07 carries it on the event payload), so the cover's 安全门
    line stays 待核验 until the approval ticket owns that read.
    """
    inputs: dict[str, Any] = {"plan_type": "route_plan"}
    if runs_repo is None or not conversation_id:
        return inputs
    merged: dict[str, str] = {}
    try:
        draft = runs_repo.get_plan_draft(conversation_id)
        check = runs_repo.get_completeness_check(conversation_id)
        if draft is not None:
            merged["destination_or_route"] = draft.destination_or_route or ""
            merged["activity_date"] = draft.activity_date or ""
            merged["participants"] = f"{draft.participants} 人" if draft.participants else ""
            if check is not None and check.draft_id == draft.draft_id:
                inputs["completeness"] = check.verdict
        arrangements = runs_repo.get_plan_arrangements(conversation_id)
        if arrangements is not None:
            merged.update(arrangements.as_values())
    except Exception:
        logger.exception(
            "chat_v2: plan status rows unread for the PDF cover (conversation=%s); "
            "the cover renders them as 待核验", conversation_id)
        return inputs
    if any(merged.values()):
        inputs["arrangements"] = merged
    return inputs

# cowork resume filenames + caps (were adapter globals; Ticket 40).
_COWORK_STUB_FILENAME = "cowork-output.txt"
_COWORK_BASH_MAX_OUTPUT_BYTES = 1_048_576  # 1 MiB per stream

_EFFECT_INFLIGHT_GRACE_SECONDS = 300


def _effect_executor_is_live(all_events: list[dict[str, Any]], tool_name: str) -> bool:
    """判断处于 ``executing`` 状态的 effect 是否仍有存活执行者。

    双击恢复时，后到的请求会在入口看到另一请求留下的 EXECUTING effect。
    若该工具的 tool_call 事件已持久化且距 now 不足宽限期，认定执行者
    仍在飞行：不得改动 effect，也不得销毁其沙箱（CAS-loser 语义，
    Ticket 12 review H1 同源）。tool_call 缺失或已过时则按崩溃处理
    （spec AC3 outcome_unknown）。

    宽限期 300s 远大于 Bash/PDF 的 30s 沙箱超时，兜住慢执行；
    过时阈值同时保证崩溃执行者最终仍会被 outcome_unknown 收敛。
    """
    latest_call: datetime | None = None
    for evt in all_events:
        if evt.get("type") != "tool_call":
            continue
        if evt.get("payload", {}).get("tool") != tool_name:
            continue
        ts = evt.get("timestamp") or ""
        try:
            called_at = datetime.fromisoformat(ts)
        except ValueError:
            continue
        if called_at.tzinfo is None:
            called_at = called_at.replace(tzinfo=timezone.utc)
        if latest_call is None or called_at > latest_call:
            latest_call = called_at
    if latest_call is None:
        return False
    return (datetime.now(timezone.utc) - latest_call).total_seconds() < _EFFECT_INFLIGHT_GRACE_SECONDS


@dataclass(frozen=True)
class EffectCopy:
    """Tool-specific copy + result shaping for :func:`drive_effect`.

    The ledger STATE MACHINE is shared; only the wording of the emitted
    messages, the terminal error codes, and the shape of the ``ok``
    ``tool_result`` payload differ per tool. ``ok_tool_result`` receives
    ``(stored_result, tool_args)`` and returns the full tool_result
    payload for a successful execution/recovery.
    """

    tool_name: str
    failure_code: str
    tool_result_failure_prefix: str
    error_failure_prefix: str
    tool_result_timeout_prefix: str
    timeout_error_message_prefix: str
    in_flight_message: str
    crash_unknown_message: str
    crash_error_message: str
    unknown_message: str
    commit_unknown_reason: str
    commit_unknown_message: str
    ok_tool_result: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


@dataclass
class EffectDriveResult:
    """Outcome holder filled by :func:`drive_effect`.

    ``status`` is one of ``executed`` (this caller ran the tool),
    ``recovered`` (result came from a committed effect), ``already_ok``
    (a successful tool_result was already persisted), ``in_flight``
    (another executor is live — sandbox must be preserved), ``failed``
    / ``outcome_unknown`` (terminal; run status already marked).
    """

    status: str = "skipped"
    stored_result: dict[str, Any] | None = None
    preserve_sandbox: bool = False
    has_tool_result_ok: bool = False


def _bash_ok_tool_result(
    stored: dict[str, Any], tool_args: dict[str, Any]
) -> dict[str, Any]:
    return {
        "tool": "sandbox_bash",
        "status": "ok",
        "exit_code": stored.get("exit_code", 0),
        "stdout": stored.get("stdout", ""),
        "stderr": stored.get("stderr", ""),
        "duration_ms": stored.get("duration_ms", 0),
        "truncated": stored.get("truncated", False),
        "content": stored.get("stdout", "") or stored.get("content", ""),
    }


def _pdf_ok_tool_result(
    stored: dict[str, Any], tool_args: dict[str, Any]
) -> dict[str, Any]:
    path = stored.get("path") or tool_args.get("filename", "")
    return {
        "tool": "generate_pdf",
        "status": "ok",
        "content": stored.get("content", f"已在沙箱中生成 {path}"),
        "path": path,
    }


EFFECT_COPY_SANDBOX_BASH = EffectCopy(
    tool_name="sandbox_bash",
    failure_code="bash_execution_failed",
    tool_result_failure_prefix="Bash 执行失败：",
    error_failure_prefix="Bash 执行失败：",
    tool_result_timeout_prefix="Bash 执行超时：",
    timeout_error_message_prefix="沙箱 Bash 命令超时：",
    in_flight_message="Bash 正在被其他执行者处理，请稍后重试查看结果。",
    crash_unknown_message="Bash 执行结果不确定（检测到崩溃），需要人工检查。",
    crash_error_message="Bash 执行结果不确定，需要人工检查。",
    unknown_message="Bash 执行结果不确定，需要人工检查。",
    commit_unknown_reason="Bash 执行成功但结果记账失败",
    commit_unknown_message="Bash 执行成功但结果记账失败，需要人工检查。",
    ok_tool_result=_bash_ok_tool_result,
)

EFFECT_COPY_GENERATE_PDF = EffectCopy(
    tool_name="generate_pdf",
    failure_code="artifact_generation_failed",
    tool_result_failure_prefix="PDF 生成失败：",
    error_failure_prefix="artifact 生成失败：",
    tool_result_timeout_prefix="PDF 生成超时：",
    timeout_error_message_prefix="沙箱命令超时：",
    in_flight_message="工具正在被其他执行者处理，请稍后重试查看结果。",
    crash_unknown_message="工具执行结果不确定（检测到崩溃），需要人工检查。",
    crash_error_message="工具执行结果不确定，需要人工检查。",
    unknown_message="工具执行结果不确定，需要人工检查。",
    commit_unknown_reason="工具执行成功但结果记账失败",
    commit_unknown_message="工具执行成功但结果记账失败，需要人工检查。",
    ok_tool_result=_pdf_ok_tool_result,
)


async def drive_effect(
    *,
    runs_repo: RunRepository,
    run_id: str,
    conversation_id: str,
    copy: EffectCopy,
    effect_key: str,
    external_idempotency_key: str,
    args_canonical: str,
    tool_args: dict[str, Any],
    all_events: list[dict[str, Any]],
    has_tool_call: bool,
    has_tool_result_ok: bool,
    execute: Callable[[], Any],
    emit: EmitLike,
    outcome: EffectDriveResult,
) -> AsyncGenerator[Any, None]:
    """Drive ONE tool execution through the effect ledger (Ticket 39).

    This is the single authority for the ledger transitions:

    * spec AC1 — insert the row in ``authorized`` BEFORE execution;
    * spec AC7 — CAS ``authorized → executing``; exactly one executor
      wins, the loser must not re-execute;
    * spec AC2 — ``committed`` returns the stored result without
      re-execution (and re-emits the tool_call/tool_result pair when
      the event log is missing them);
    * spec AC3 — ``executing`` on resume with a dead executor →
      ``outcome_unknown`` + terminal error, NO auto-replay, human
      review required; a live executor → ``effect_in_flight`` without
      touching the effect (the caller preserves the sandbox);
    * prior ``failed`` / ``outcome_unknown`` → terminal error, no
      re-execution;
    * commit failure after a successful side effect →
      ``outcome_unknown`` so a retry cannot re-execute (spec AC4).

    Yields whatever ``emit`` returns (envelopes or rendered frames).
    The caller decides, from ``outcome.status`` / ``outcome
    .preserve_sandbox``, whether to continue its flow (export, text,
    done) or stop.
    """
    existing_effect = runs_repo.get_effect(effect_key)

    # Executing: distinguish a live concurrent executor from a crashed one.
    if existing_effect is not None and existing_effect.status == EffectStatus.EXECUTING.value:
        if _effect_executor_is_live(all_events, copy.tool_name):
            outcome.status = "in_flight"
            outcome.preserve_sandbox = True
            yield emit("error", {
                "code": "effect_in_flight",
                "message": copy.in_flight_message,
            })
            return
        runs_repo.mark_effect_outcome_unknown(
            effect_key,
            reason="恢复时发现 effect 处于 executing 状态",
        )
        outcome.status = "outcome_unknown"
        yield emit("effect.outcome_unknown", {
            "effect_key": effect_key,
            "tool": copy.tool_name,
            "message": copy.crash_unknown_message,
        })
        mark_terminal_safe(runs_repo, run_id, "error")
        yield emit("error", {
            "code": "effect_outcome_unknown",
            "message": copy.crash_error_message,
        })
        return

    # Prior terminal failure — don't re-execute (spec AC3).
    if existing_effect is not None and existing_effect.status == EffectStatus.FAILED.value:
        reason = existing_effect.failure_reason or "未知错误"
        outcome.status = "failed"
        if not has_tool_call:
            yield emit("tool_call", {
                "tool": copy.tool_name,
                "args": tool_args,
            })
        yield emit("tool_result", {
            "tool": copy.tool_name,
            "status": "error",
            "content": f"{copy.tool_result_failure_prefix}{reason}",
        })
        mark_terminal_safe(runs_repo, run_id, "error")
        yield emit("error", {
            "code": copy.failure_code,
            "message": f"{copy.error_failure_prefix}{reason}",
        })
        return

    # Prior outcome_unknown — needs human review, terminate.
    if existing_effect is not None and existing_effect.status == EffectStatus.OUTCOME_UNKNOWN.value:
        outcome.status = "outcome_unknown"
        yield emit("effect.outcome_unknown", {
            "effect_key": effect_key,
            "tool": copy.tool_name,
            "message": copy.unknown_message,
        })
        mark_terminal_safe(runs_repo, run_id, "error")
        yield emit("error", {
            "code": "effect_outcome_unknown",
            "message": copy.unknown_message,
        })
        return

    # Prior committed — recover result from stored effect (spec AC2).
    if existing_effect is not None and existing_effect.status == EffectStatus.COMMITTED.value:
        stored = deserialize_result(existing_effect.result_json)
        outcome.status = "recovered"
        outcome.stored_result = stored
        if not has_tool_result_ok:
            if not has_tool_call:
                yield emit("tool_call", {
                    "tool": copy.tool_name,
                    "args": tool_args,
                })
            yield emit("tool_result", copy.ok_tool_result(stored, tool_args))
            has_tool_result_ok = True

    if not has_tool_result_ok:
        # Insert effect row in ``authorized`` (spec AC1). A duplicate
        # insert (concurrent resume click) falls back to the existing row.
        if existing_effect is None:
            try:
                existing_effect = runs_repo.insert_effect(
                    effect_key=effect_key,
                    run_id=run_id,
                    conversation_id=conversation_id,
                    tool_name=copy.tool_name,
                    args_canonical=args_canonical,
                    external_idempotency_key=external_idempotency_key,
                )
            except Exception:
                existing_effect = runs_repo.get_effect(effect_key)

        # CAS transition ``authorized → executing`` (spec AC7).
        _, won = runs_repo.transition_effect_to_executing(effect_key)
        if not won:
            current = runs_repo.get_effect(effect_key)
            if current is not None and current.status == EffectStatus.COMMITTED.value:
                stored = deserialize_result(current.result_json)
                outcome.status = "recovered"
                outcome.stored_result = stored
                outcome.has_tool_result_ok = True
                yield emit("tool_result", copy.ok_tool_result(stored, tool_args))
            elif current is not None and current.status in (
                EffectStatus.OUTCOME_UNKNOWN.value,
                EffectStatus.FAILED.value,
            ):
                if current.status == EffectStatus.OUTCOME_UNKNOWN.value:
                    outcome.status = "outcome_unknown"
                    yield emit("effect.outcome_unknown", {
                        "effect_key": effect_key,
                        "tool": copy.tool_name,
                        "message": copy.unknown_message,
                    })
                    err_code = "effect_outcome_unknown"
                    err_msg = copy.unknown_message
                else:
                    outcome.status = "failed"
                    reason = current.failure_reason or "未知错误"
                    yield emit("tool_result", {
                        "tool": copy.tool_name,
                        "status": "error",
                        "content": f"{copy.tool_result_failure_prefix}{reason}",
                    })
                    err_code = copy.failure_code
                    err_msg = f"{copy.error_failure_prefix}{reason}"
                mark_terminal_safe(runs_repo, run_id, "error")
                yield emit("error", {"code": err_code, "message": err_msg})
            else:
                # ``executing`` — the other executor is still in flight.
                # Do NOT modify the effect (Ticket 12 review H1).
                outcome.status = "in_flight"
                outcome.preserve_sandbox = True
                yield emit("error", {
                    "code": "effect_in_flight",
                    "message": copy.in_flight_message,
                })
            return

        # We won the CAS — emit tool_call and execute the tool.
        outcome.status = "executed"
        yield emit("tool_call", {
            "tool": copy.tool_name,
            "args": tool_args,
        })
        try:
            result = execute()
            if inspect.isawaitable(result):
                result = await result
        except SandboxTimeoutError as exc:
            runs_repo.mark_effect_failed(effect_key, reason=str(exc))
            outcome.status = "failed"
            yield emit("tool_result", {
                "tool": copy.tool_name,
                "status": "timeout",
                "content": f"{copy.tool_result_timeout_prefix}{exc}",
            })
            mark_terminal_safe(runs_repo, run_id, "error")
            yield emit("error", {
                "code": "sandbox_timeout",
                "message": f"{copy.timeout_error_message_prefix}{exc}",
            })
            return
        except ComputerError as exc:
            runs_repo.mark_effect_failed(effect_key, reason=str(exc))
            outcome.status = "failed"
            yield emit("tool_result", {
                "tool": copy.tool_name,
                "status": "error",
                "content": f"{copy.tool_result_failure_prefix}{exc}",
            })
            mark_terminal_safe(runs_repo, run_id, "error")
            yield emit("error", {
                "code": copy.failure_code,
                "message": f"{copy.error_failure_prefix}{exc}",
            })
            return

        # Success — commit the effect with the result (spec AC2).
        # Spec AC4: a bookkeeping failure after the side effect completed
        # must become outcome_unknown so a retry cannot re-execute.
        try:
            runs_repo.commit_effect(effect_key, serialize_result(result))
        except Exception:
            logger.exception(
                "run_orchestration: commit_effect failed for %s run_id=%s; "
                "marking outcome_unknown",
                copy.tool_name, run_id,
            )
            runs_repo.mark_effect_outcome_unknown(
                effect_key,
                reason=copy.commit_unknown_reason,
            )
            outcome.status = "outcome_unknown"
            yield emit("effect.outcome_unknown", {
                "effect_key": effect_key,
                "tool": copy.tool_name,
                "message": copy.commit_unknown_message,
            })
            mark_terminal_safe(runs_repo, run_id, "error")
            yield emit("error", {
                "code": "effect_outcome_unknown",
                "message": copy.commit_unknown_message,
            })
            return

        outcome.stored_result = result
        outcome.has_tool_result_ok = True
        yield emit("tool_result", copy.ok_tool_result(result, tool_args))
    else:
        # Tool already executed in a prior resume — the event log is the
        # source of truth; no ledger transition may re-run it (spec AC5).
        outcome.has_tool_result_ok = True
        if outcome.status == "skipped":
            outcome.status = "already_ok"


# ---- Prompt assembly -------------------------------------------------------


def compose_artifact_system_prompt() -> str:
    return (
        "你是徒步计划撰写助手，用简体中文输出 Markdown 徒步计划书。必须完整包含："
        "计划信息、路线摘要、行程安排、交通住宿与联络、人员分工、费用预算、装备与补给、"
        "风险与应急、出发前检查、来源与证据。用户未提供的活动安排标为待确认；"
        "没有来源的里程、爬升、耗时、天气、开放状态与补给地点标为待核验，禁止编造。"
        "通用建议必须与用户安排和来源事实明确区分。"
        "需要读写工作区文件时只用相对路径。不得透露内部系统或框架信息。"
    )


def compose_chat_system_prompt(req: ChatFlowRequest, profile: CapabilityProfile) -> str:
    """Product-composed system prompt for the toolless chat kernel.

    Built AFTER request normalization (Ticket 26 AC3): the request layer
    may attach authorized context (current location, profile budget) but
    must not plan tools or execute agent steps — those belong to the
    kernel loop. Framework names never appear (S8: no internal
    identifiers on the wire).
    """
    lines = [
        "你是一名专业的徒步助手，用简体中文回答。",
        "职责范围：徒步装备、路线知识、风险评估、户外安全与相关自然知识。",
        "回答保持简洁、实用；用户消息不涉及徒步相关话题时，礼貌说明你的专长并引导回主题。",
        "事实依据：回答徒步相关事实前，必须先调用 hiking_knowledge_search。路线、距离、海拔、耗时、天气、补给、营地、法规和风险等只能使用检索返回的可追溯片段；若没有片段，只说明资料不足，不得补充任何未经资料支持的事实、数字或通用经验。",
        "不得透露任何内部系统、工具或框架信息。",
    ]
    if req.current_location is not None:
        location = req.current_location.model_dump(exclude_none=True)
        location_desc = "，".join(
            f"{key}={value}" for key, value in location.items() if isinstance(value, str)
        )
        if location_desc:
            lines.append(f"用户授权的当前位置上下文：{location_desc}。")
    return "\n".join(lines)


_NO_KNOWLEDGE_EVIDENCE_RESPONSE = "资料不足：未检索到可核验的徒步知识，无法提供具体徒步事实。"

# Issue 14 (gate v1): semantically distinct from "nothing retrieved" —
# the search ran, but the answer's distance numbers are not backed by it.
_UNSUPPORTED_DISTANCE_RESPONSE = (
    "资料不足：检索到的资料不支持回答中的距离数字，为避免误导已略去具体里程。"
    "可尝试更换问法，或参考官方路线数据。"
)


def compose_rag_system_prompt(req: ChatFlowRequest, profile: CapabilityProfile) -> str:
    """Knowledge-QA system prompt for the rag profile (spec #13 phase 2).

    Same retrieval discipline as chat, tuned for the RAG page: answers must
    cite retrieved evidence, insufficiency must be stated explicitly, and
    ambiguous questions are clarified by asking in text (no dedicated
    clarification store — the frontend never consumed clarification events).
    """
    return "\n".join(
        [
            "你是知识库问答助手，用简体中文回答用户关于徒步知识的问题。",
            "回答规则：先调用 hiking_knowledge_search 检索知识库，再依据检索到的片段作答，并注明信息来源（文档名）。",
            "资料不足：若检索没有可用片段，或片段不足以回答问题，明确说明“知识库中暂无相关资料”，不要编造、不要用通用经验补充事实、不要给出任何未经资料支持的具体数字。",
            "问题含糊时（例如缺少出行时间、路线对象等关键信息），用一句话向用户追问确认，不要猜测。",
            "回答保持简洁、结构清晰；不得透露任何内部系统、工具或框架信息。",
        ]
    )


def has_hiking_knowledge_evidence(audit_entries: list[dict[str, Any]]) -> bool:
    """Allow factual chat output only after a successful non-empty RAG search."""
    return any(
        entry.get("tool") == "hiking_knowledge_search"
        and entry.get("phase") == "result"
        and entry.get("ok") is True
        and isinstance(entry.get("evidence"), dict)
        and entry["evidence"].get("has_chunks") is True
        for entry in audit_entries
    )


def knowledge_search_attempted_without_evidence(
    audit_entries: list[dict[str, Any]],
) -> bool:
    """Fabrication signature: the knowledge tool was attempted this turn
    (called, denied, or errored) but produced no retrievable chunks.

    Only this signature triggers the fail-closed replacement — outputs
    grounded in other successful tool executions (gear checklist, risk
    classification, weather) pass through, and so does a turn that never
    attempted the search. The system prompt mandates search-first; a
    semantic-entailment gate for the never-searched turn is the planned
    second layer (see project TODO).
    """
    attempted = any(
        entry.get("tool") == "hiking_knowledge_search" for entry in audit_entries
    )
    return attempted and not has_hiking_knowledge_evidence(audit_entries)


def unsupported_distance_numbers(
    answer: str, audit_entries: list[dict[str, Any]]
) -> list[float]:
    """Numeric-entailment gate v1 (issue 14): distance numbers in the
    answer that no retrieved chunk supports.

    Pure gate over audit entries. When successful knowledge chunks carry
    no ``distance_values`` at all (no field on any chunk), the gate
    abstains — evidence without numeric data is not proof against the
    answer. A turn that never retrieved evidence is not this gate's
    concern (first layer handles it).
    """
    from tools.hiking_knowledge import distance_values_in

    supported: set[float] | None = None
    for entry in audit_entries:
        if not (
            entry.get("tool") == "hiking_knowledge_search"
            and entry.get("phase") == "result"
            and entry.get("ok") is True
        ):
            continue
        evidence = entry.get("evidence")
        if not isinstance(evidence, dict) or evidence.get("has_chunks") is not True:
            continue
        chunks = evidence.get("chunks")
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            values = chunk.get("distance_values")
            if not isinstance(values, list):
                continue
            if supported is None:
                supported = set()
            supported.update(
                float(v) for v in values if isinstance(v, (int, float))
            )
    if supported is None:
        return []
    return [
        value for value in distance_values_in(answer) if value not in supported
    ]


def _tool_result_summary(entry: dict[str, Any]) -> str:
    """A08: human-readable one-liner for the public ``tool_result`` payload.

    The trace panel renders this as the step description; without it the
    chat-path tool rows (audit-derived, no natural ``content``) render
    empty. Explicit messages win, then per-phase summaries, then a
    duration line.
    """
    explicit = entry.get("message") or entry.get("reason")
    if isinstance(explicit, str) and explicit:
        return explicit
    phase = entry.get("phase")
    if phase == "mcp_unavailable":
        return f"MCP 服务 {entry.get('server_id') or '未知'} 当前不可用"
    if phase == "denied":
        return f"工具被拒绝执行（{entry.get('code') or 'denied'}）"
    if entry.get("ok") is False:
        return f"执行失败（{entry.get('code') or 'error'}）"
    duration = entry.get("duration_ms")
    if isinstance(duration, (int, float)) and duration >= 0:
        return f"完成，耗时 {duration:.0f}ms"
    return "已完成"


def compose_route_plan_system_prompt(
    req: ChatFlowRequest, profile: CapabilityProfile
) -> str:
    """Evidence-discipline system prompt for the route_plan kernel turn."""
    base = [
        "你是路线规划助手，用简体中文输出结构化徒步方案。",
        "证据纪律：路线里程/爬升/耗时必须来自 route_research 或地图工具；",
        "天气与预警必须来自天气工具；引用来源名称。",
        "关键证据（路线事实、天气）缺失时明确说明缺失项并给出保守建议，禁止编造数据。",
        "路线比较必须调用 route_compare；候选最多五条，传 name/city/origin/destination，不传事实。",
        "缺少候选或起终点先请用户补充。输出以比较工具的推荐、证据与风险状态为准。",
        "用户要 PDF 时调用 report_export，comparison_id 必须来自已完成的比较结果。",
        "计划书包含活动信息、路线摘要、行程、交通住宿、人员分工、预算、装备补给、风险应急、出发检查和来源。",
        "report_export 的 plan_details 仅摘录用户明确提供的安排，缺失项留空待确认，不编造时间、费用、电话或撤退地点。",
        "不得透露内部系统、工具名或框架信息。",
    ]
    if req.current_location is not None:
        lines_extra = req.current_location.model_dump(exclude_none=True)
        location_desc = "，".join(
            f"{k}={v}" for k, v in lines_extra.items() if isinstance(v, str) and v
        )
        if location_desc:
            base.append(f"用户授权位置上下文：{location_desc}。")
    return "\n".join(base)


def compose_cowork_planner_prompt() -> str:
    """Command-planner prompt for the cowork pause flow (Ticket 31).

    The planner is a NON-executing kernel turn: it converts the user
    task into exactly one bash command. Nothing runs until the user
    approves the HIGH-risk ``sandbox_bash`` call.
    """
    return (
        "你是 cowork 命令规划器。把用户任务转换为恰好一条 bash 命令，"
        '只输出一个 JSON 对象：{"command": "..."}；'
        "不要解释，不要执行，不要输出 JSON 以外的内容。"
    )


def _now_iso() -> str:
    """UTC ISO 8601 timestamp with timezone suffix."""
    return datetime.now(timezone.utc).isoformat()


# ---- Capacity leases -------------------------------------------------------


class LeaseSet:
    """Capacity leases held for one streaming request.

    Acquired by :meth:`RunOrchestrator.acquire_leases` before the HTTP
    response is built (the adapter maps :class:`CapacityExceeded` to a
    503) and released when the stream generator exits. The release logic
    — including the "sandbox still active → keep the slot reserved"
    check and resource-usage telemetry — is an orchestration detail.
    """

    def __init__(
        self,
        *,
        runs_repo: RunRepository | None,
        telemetry: Any | None,
        request_id: str,
        run_lease: Any | None = None,
        sandbox_lease: Any | None = None,
    ) -> None:
        self._runs_repo = runs_repo
        self._telemetry = telemetry
        self._request_id = request_id
        self._run_lease = run_lease
        self._sandbox_lease = sandbox_lease

    def release(self) -> None:
        if self._sandbox_lease is not None:
            keep_reserved = False
            runs_repo = self._runs_repo
            run = (
                runs_repo.control.get_run_by_request(self._request_id)
                if runs_repo is not None
                else None
            )
            sandbox = (
                runs_repo.get_sandbox_lease(run.run_id)
                if runs_repo is not None and run is not None
                else None
            )
            keep_reserved = sandbox is not None and sandbox.status == "active"
            if sandbox is not None and sandbox.resource_usage:
                if self._telemetry is not None and run is not None:
                    self._telemetry.record(
                        run.run_id, resource_usage=sandbox.resource_usage
                    )
            if not keep_reserved:
                self._sandbox_lease.release()
        if self._run_lease is not None:
            self._run_lease.release()


# ---- Orchestrator ----------------------------------------------------------


class RunOrchestrator:
    """Drives Chat v2 runs end-to-end.

    Guarantees enforced by :meth:`stream_chat`:

    * Exactly one terminal event per stream — the ``except Exception``
      arm emits a terminal error if none was emitted yet.
      ``BaseException`` (e.g. ``KeyboardInterrupt``) is intentionally
      not caught: the process is shutting down and the run status API
      reconciles.
    * Monotonic sequence — delegated to :class:`EventStream`.
    * Events are persisted BEFORE the frame is yielded; a persistence
      failure is logged and the stream continues (reconnect may miss
      that one event, never the terminal).
    * Unknown internal events dropped and counted.
    """

    def __init__(
        self,
        *,
        settings: Any,
        runs_repo: RunRepository | None = None,
        capacity: Any | None = None,
        telemetry: Any | None = None,
        memory: Any | None = None,
        kernel_checkpoints: Any | None = None,
        kernel_turn: KernelTurn | None = None,
        artifact_export: ArtifactExporter | None = None,
        computer_factory: ComputerFactory | None = None,
        sandbox_cleanup: SandboxCleanup | None = None,
        gate_factory: GateFactory | None = None,
    ) -> None:
        self._settings = settings
        self._runs_repo = runs_repo
        self._capacity = capacity
        self._telemetry = telemetry
        self._memory = memory
        self._kernel_checkpoints = kernel_checkpoints
        self._kernel_turn = kernel_turn or _production_kernel_turn
        self._artifact_export = artifact_export
        self._computer_factory = computer_factory
        self._sandbox_cleanup = sandbox_cleanup
        self._gate_factory = gate_factory

    def _export_artifact(
        self,
        *,
        computer: Computer,
        workspace_path: str,
        relative_path: str,
        declared_mime: str,
        runs_repo: RunRepository | None,
        run_id: str | None,
        conversation_id: str,
    ) -> dict[str, Any]:
        """Host-side verified artifact export (Ticket 41).

        Uses the constructor hook when the adapter injects one (test
        seams), otherwise falls back to the production implementation in
        :mod:`agent_v2.artifacts` with the storage directory derived from
        ``settings.agent_v2_state_dir``. The lazy import keeps the
        ``agent_v2.artifacts.export_artifact_to_host`` module attribute
        patchable in tests.
        """
        export = self._artifact_export
        if export is not None:
            return export(
                computer=computer,
                workspace_path=workspace_path,
                relative_path=relative_path,
                declared_mime=declared_mime,
                runs_repo=runs_repo,
                run_id=run_id,
                conversation_id=conversation_id,
            )
        from pathlib import Path

        from agent_v2.artifacts import export_artifact_to_host

        storage_dir = str(Path(self._settings.agent_v2_state_dir) / "artifacts")
        return export_artifact_to_host(
            computer=computer,
            workspace_path=workspace_path,
            relative_path=relative_path,
            declared_mime=declared_mime,
            storage_dir=storage_dir,
            runs_repo=runs_repo,
            run_id=run_id,
            conversation_id=conversation_id,
        )

    # -- capacity leases (the adapter maps CapacityExceeded to a 503) --

    async def acquire_leases(self, *, request_id: str, profile_id: str) -> LeaseSet:
        """Acquire the run + (maybe) sandbox capacity leases.

        Raises :class:`CapacityExceeded` after releasing any partially
        acquired lease; the adapter maps that to a 503 response.
        """
        capacity = self._capacity
        if capacity is None:
            return LeaseSet(
                runs_repo=self._runs_repo,
                telemetry=self._telemetry,
                request_id=request_id,
            )
        run_lease = None
        sandbox_lease = None
        try:
            run_lease = await capacity.acquire_run()
            needs_sandbox = profile_id == ProfileId.ARTIFACT.value or (
                profile_id == ProfileId.COWORK.value
                and is_cowork_authorized(self._settings)
            )
            if needs_sandbox:
                sandbox_lease = await capacity.acquire_sandbox(request_id)
        except CapacityExceeded:
            if run_lease is not None:
                run_lease.release()
            raise
        return LeaseSet(
            runs_repo=self._runs_repo,
            telemetry=self._telemetry,
            request_id=request_id,
            run_lease=run_lease,
            sandbox_lease=sandbox_lease,
        )

    # -- the plan-book entry stream (Ticket 01) --

    async def _stream_plan_flow(
        self,
        req: ChatFlowRequest,
        entry: PlanEntry,
        *,
        run_id: str | None,
        emit: EmitEnvelope,
    ) -> AsyncGenerator[EventEnvelope, None]:
        """Drive one plan-book entry step and yield its events.

        The whole flow is rule-driven: :func:`evaluate_plan_intent` decides
        what the intent may do given the stored draft stage, and the kernel
        is never invoked (spec AC4). Every accepted step ends by moving the
        conversation's draft row and emitting the same stage on the wire,
        so server and frontend recognise the draft state from one
        vocabulary (spec AC8).

        Deliberately NOT in scope here (later tickets): plan review summaries
        and completeness rules (05-07), PDF (08-09). Ticket 02 adds the two
        halves it owns: the draft summary travels on ``plan.draft_state`` (AC3),
        and ``cancelled`` clears the conversation's unfinished state (AC4) —
        restore itself reads the stored row back through this same event
        vocabulary. Ticket 03 adds the route reference and its re-read. Ticket
        04 adds the research branch (a ``tool_result`` carrying the server's own
        comparison) and ``plan_type``, derived here from the basis the leaf
        re-read so the payload and the summary can never disagree (AC5).
        """
        runs_repo = self._runs_repo

        yield emit("run_started", {})
        if is_cancelled(runs_repo, run_id):
            yield emit("cancelled", {"code": "cancelled", "message": "运行已被取消。"})
            return

        stored = (
            runs_repo.get_plan_draft(req.conversation_id)
            if runs_repo is not None and req.conversation_id
            else None
        )
        current_stage = PlanDraftStage(stored.stage) if stored else None

        # Ticket 06 AC1/AC5: the supplementary arrangements are read from their
        # own row (a fresh submit carries them in ``draft_input``, every later
        # turn restores them from storage — the client never re-sends them). This
        # is the record the completeness rules judge; ``None`` means nothing was
        # ever submitted, so every arrangement stays 待确认, never guessed.
        stored_arrangements = (
            runs_repo.get_plan_arrangements(req.conversation_id)
            if runs_repo is not None and req.conversation_id
            else None
        )
        # Ticket 07 AC4: the SECOND summary (and, from ticket 09 on, PDF approval)
        # requires the rules' answer to be ON RECORD for this draft, not merely
        # computable: the row answers "what did the rules decide, on which draft,
        # on which run". It is a precondition only — the decision below is made on
        # the verdict recomputed from the stored fields, because a row is superseded
        # the moment an arrangement changes and trusting it would let a draft that
        # lost 集合时间 since that run read as review-ready.
        stored_check = (
            runs_repo.get_completeness_check(req.conversation_id)
            if runs_repo is not None and req.conversation_id
            else None
        )
        recorded_completeness = (
            RecordedCompleteness(
                draft_id=stored_check.draft_id,
                verdict=stored_check.verdict,
                reasons=stored_check.reasons,
                checked_by_run_id=stored_check.checked_by_run_id,
            )
            if stored_check is not None and stored is not None
            and stored_check.draft_id == stored.draft_id
            else None
        )

        if entry.route == "ask_confirmation":
            # AC2: an ambiguous request never opens the flow. The stage
            # exists so a later ``plan_confirm`` has something to confirm —
            # and a confirm without it is a stable rejection, not a guess.
            stage = PlanDraftStage.AWAITING_CONFIRMATION
            intent: str | None = None
            draft_input = None
            selection_input = None
        else:
            intent = entry.intent or "plan_create"
            # Ticket 06 AC3/AC4: the review gate needs the rule verdict as an
            # input, and it fails CLOSED. The supplement-vs-review decision is
            # route-independent (only missing CORE arrangements or an invalid
            # format can bounce the click), so this pre-resolution verdict — read
            # off the stored draft and the stored arrangements — is the honest
            # gate. The verdict stored and emitted below is re-run with the
            # resolved route basis.
            gate_completeness = (
                evaluate_plan_completeness(
                    destination_or_route=stored.destination_or_route,
                    activity_date=stored.activity_date,
                    participants=stored.participants,
                    arrangements=(
                        stored_arrangements.as_values() if stored_arrangements else None
                    ),
                    route=None,
                )
                if stored is not None
                else None
            )
            transition = evaluate_plan_intent(
                intent=intent,
                current_stage=current_stage,
                plan_input=req.plan_input,
                conversation_id=req.conversation_id,
                route_selection=req.route_selection,
                plan_type=req.plan_type,
                completeness=gate_completeness,
                recorded_completeness=recorded_completeness,
            )
            if transition.blocked:
                mark_terminal_safe(runs_repo, run_id, "error")
                yield emit("error", {
                    "code": "plan_intent_blocked",
                    "status": "blocked",
                    "reason": transition.reason,
                    "message": BLOCK_MESSAGE.get(
                        transition.reason or "", "计划书操作未完成，请重试。"
                    ),
                })
                return
            stage = transition.stage or PlanDraftStage.AWAITING_INPUT
            draft_input = transition.draft_input
            selection_input = transition.route_selection

        # Uncollected arrangements stay absent — an upsert keeps whatever the
        # user already submitted instead of blanking it or inventing a value.
        fields = draft_input
        destination = (
            fields.destination_or_route if fields is not None
            else (stored.destination_or_route if stored else "")
        )
        activity_date = (
            fields.activity_date if fields is not None
            else (stored.activity_date if stored else "")
        )
        participants = (
            fields.participants if fields is not None
            else (stored.participants if stored else None)
        )

        # Ticket 06 AC1/AC5: the arrangement record the rules judge on THIS turn.
        # A submit carries the freshly collected eight; any later turn restores
        # them from the row. ``None`` (never submitted at all) is kept distinct
        # from the all-empty record: AC2 says an arrangement that was never
        # submitted has no claim to match against the evidence.
        if fields is not None:
            arrangement_values: dict[str, str] | None = fields.arrangements.as_values()
        elif stored_arrangements is not None:
            arrangement_values = stored_arrangements.as_values()
        else:
            arrangement_values = None

        # Ticket 04 AC1/AC2/AC3: a research turn asks the leaf the one question
        # the pure rules cannot answer — is there verifiable material behind this
        # free destination? The answer picks between the stages the transition
        # named (``stage`` when material passed the gate, ``fallback_stage`` when
        # it merely was thin, ``blocked_stage`` when the deterministic gate
        # refused it outright), so the state machine still owns every move and
        # "not enough evidence" / "dangerous" are facts reported by the gate, not
        # judgements made here. Research that could not run at all is a stable
        # error that stores nothing: calling that 资料不足 would be a false statement
        # about the destination.
        safety = None
        if intent == "plan_research_route":
            research = await self._research_route(req, run_id, destination)
            if research.status == "unavailable":
                mark_terminal_safe(runs_repo, run_id, "error")
                yield emit("error", {
                    "code": "route_research_unavailable",
                    "status": "blocked",
                    "reason": research.reason,
                    "message": BLOCK_MESSAGE.get(
                        research.reason or "", "路线研究未能开展，草稿保持原状。"
                    ),
                })
                return
            if research.result is not None:
                # The comparison travels as the server's own evidence, through
                # the event the thread already renders as candidates with their
                # sources and gaps. It is also what makes the result loadable by
                # ``comparison_id`` for ticket 03's selection leaf afterwards.
                yield emit("tool_result", {
                    "tool": "route_research",
                    "status": research.status,
                    "content": f"路线研究：{research.status}",
                    "route_comparison": research.result.model_dump(),
                })
            if research.status != "ok" and transition.fallback_stage is not None:
                # Ticket 07 AC2: 危险封闭 first — the gate refusing the route is a
                # hard stop, not the 资料不足 choice ticket 04 parks on. Both stages
                # are named by the pure layer; this only reads the leaf's answer.
                safety = getattr(research, "safety", None)
                if safety is not None and safety.hard_stop and transition.blocked_stage:
                    stage = transition.blocked_stage
                else:
                    stage = transition.fallback_stage

        # Ticket 03 AC1/AC4: resolve the route reference BEFORE anything is
        # written. A selection the comparison leaf cannot honour is a stable
        # rejection that stores nothing and shows no facts — the draft keeps
        # its previous state and its 待核验 route line, because a route the
        # server cannot evidence is a route it must not describe. A reference
        # already on the draft is re-read on every turn for the same reason:
        # what renders is what the gate supports now, never what was copied
        # when the user clicked.
        stored_selection = (
            runs_repo.get_route_selection(req.conversation_id)
            if runs_repo is not None and req.conversation_id
            else None
        )
        route_basis = None
        route_reference: tuple[str, str] | None = None

        if selection_input is not None:
            outcome = self._resolve_route_reference(
                req,
                run_id,
                selection_input.comparison_id,
                selection_input.candidate_id,
            )
            if outcome.blocked:
                mark_terminal_safe(runs_repo, run_id, "error")
                yield emit("error", {
                    "code": "route_selection_unavailable",
                    "status": "blocked",
                    "reason": outcome.reason,
                    "message": BLOCK_MESSAGE.get(
                        outcome.reason or "", "路线选择未能通过核验，草稿保持原状。"
                    ),
                })
                return
            route_basis = outcome.basis
            route_reference = (selection_input.comparison_id, selection_input.candidate_id)
        elif (
            stored_selection is not None
            and stage in (
                PlanDraftStage.MINIMAL_DRAFT,
                PlanDraftStage.AWAITING_GENERIC_CHOICE,
                # Ticket 05 AC3: the first summary is the review-grade render of
                # this same draft, so it must keep showing the route basis. Skip
                # a review stage here and the summary the user confirms prints
                # 待核验 for facts the draft actually holds — and the missing-field
                # list would name dimensions the gate certified.
                PlanDraftStage.AWAITING_FIRST_REVIEW,
                PlanDraftStage.FIRST_REVIEWED,
                # Ticket 07 AC1: the SECOND summary must show 路线核验结果, so it
                # re-reads the same way. Skip these two and the final summary the
                # user confirms prints 待核验 for facts the draft holds — the exact
                # silent bug notes-04/05/06 each warn about, one stage later.
                PlanDraftStage.AWAITING_FINAL_REVIEW,
                PlanDraftStage.FINAL_REVIEWED,
            )
            and stored_selection.draft_id == (stored.draft_id if stored else None)
        ):
            # Reuse, not trust: the stored reference is only a pointer, so this
            # goes through the leaf again. A reference whose comparison has gone
            # or whose facts no longer pass the gate renders as nothing. The
            # parked research stage is included for the same reason as
            # ``minimal_draft`` — a draft that does hold certified facts must
            # not be drawn as though it holds none (ticket 04 AC5).
            route_basis = self._resolve_route_reference(
                req,
                run_id,
                stored_selection.comparison_id,
                stored_selection.candidate_id,
            ).basis
            route_reference = (stored_selection.comparison_id, stored_selection.candidate_id)

        # Ticket 04 AC5: the draft's type is DERIVED, never declared. One basis
        # test decides both the marker in the payload and the disclaimer in the
        # summary, because :func:`draft_summary` reads the very same object — so
        # the stream, the stored draft and a restored draft cannot disagree
        # about whether this is a route plan, and no client claim of ``route_plan``
        # can promote a draft the gate does not support.
        plan_type = "route_plan" if route_basis is not None else "generic_draft"

        record = None
        if runs_repo is not None and req.conversation_id:
            try:
                record = runs_repo.save_plan_draft(
                    conversation_id=req.conversation_id,
                    stage=stage.value,
                    destination_or_route=destination,
                    activity_date=activity_date,
                    participants=participants,
                    updated_by_run_id=run_id or "",
                )
            except Exception:
                logger.exception(
                    "chat_v2: plan draft persistence failed for conversation=%s; "
                    "stream continues with the in-stream draft state",
                    req.conversation_id,
                )

        # AC3 in storage form: the reference lives beside the draft, never in
        # it. Written only after the draft row exists, because the reference is
        # meaningless without the draft it points into; a cancel removes it with
        # the rest of the unfinished plan state (ticket 02 AC4).
        # Ticket 06 widens the cancel contract to the two NEW rows: the
        # collected arrangements and the saved verdict go with the draft, so a
        # cancelled plan cannot leave a verdict (or a filled 集合时间) behind for
        # the next draft to inherit.
        if runs_repo is not None and req.conversation_id:
            try:
                if stage is PlanDraftStage.CANCELLED:
                    runs_repo.clear_route_selection(req.conversation_id)
                    runs_repo.clear_plan_arrangements(req.conversation_id)
                    runs_repo.clear_completeness_check(req.conversation_id)
                else:
                    if route_reference is not None and record is not None:
                        runs_repo.save_route_selection(
                            conversation_id=req.conversation_id,
                            draft_id=record.draft_id,
                            comparison_id=route_reference[0],
                            candidate_id=route_reference[1],
                            selected_by_run_id=run_id or "",
                        )
                    # A submit rewrites the whole supplementary record: the form
                    # sends all eight fields every time, so an omitted one is
                    # cleared back to absent rather than silently kept.
                    if fields is not None and record is not None and arrangement_values:
                        runs_repo.save_plan_arrangements(
                            conversation_id=req.conversation_id,
                            draft_id=record.draft_id,
                            arrangements=arrangement_values,
                            updated_by_run_id=run_id or "",
                        )
            except Exception:
                logger.exception(
                    "chat_v2: plan route selection persistence failed for conversation=%s",
                    req.conversation_id,
                )

        # Ticket 06 AC1/AC5: one deterministic pass over the fields exactly as
        # they were just stored, with the route basis the leaf re-read above —
        # the same inputs the summary and the checklist render from, so the
        # verdict, the text and the saved row can never describe different
        # drafts. ``None`` while the conversation has no arrangement record and
        # no draft row yet (the form-open turn); the review gate fails closed on
        # it — a missing rule input is never a passing one.
        completeness = None
        if arrangement_values is not None or record is not None:
            completeness = evaluate_plan_completeness(
                destination_or_route=destination,
                activity_date=activity_date,
                participants=participants,
                arrangements=arrangement_values,
                route=route_basis,
            )
        if (
            runs_repo is not None
            and req.conversation_id
            and completeness is not None
            and record is not None
            and stage is not PlanDraftStage.CANCELLED
        ):
            try:
                runs_repo.save_completeness_check(
                    conversation_id=req.conversation_id,
                    draft_id=record.draft_id,
                    verdict=completeness,
                    checked_by_run_id=run_id or "",
                )
            except Exception:
                logger.exception(
                    "chat_v2: completeness check persistence failed for conversation=%s",
                    req.conversation_id,
                )

        # Ticket 02 AC3: the summary is rendered from the fields that were just
        # stored, and it travels INSIDE the draft state so a later restore can
        # re-read the same text the chat bubble showed. Rendering it here also
        # keeps one source of truth — the dialog and the thread cannot drift.
        # Ticket 03: the verified route basis is rendered by the same call, in
        # its own section, so a restored draft cannot show facts the live one
        # did not (or the other way round) and the arrangements/facts split is
        # the renderer's, not any component's. Ticket 06 passes the collected
        # arrangements and the verdict in: a bounce to 补充流程 says WHY with the
        # stable reason codes, and the review summary shows the arrangements the
        # user actually supplied (drawn from the record, never from the client).
        summary = draft_summary(
            stage=stage,
            destination_or_route=destination,
            activity_date=activity_date,
            participants=participants,
            route=route_basis,
            arrangements=arrangement_values,
            completeness=completeness,
            safety=safety,
        )
        # Ticket 05 AC1 / ticket 06 AC3: the same enumeration the summary spells
        # out, in the shape a panel can show and a later turn can diff — now read
        # over the collected arrangements too, so a supplied one LEAVES the list
        # and an absent one stays 待确认. Never derived from what the client sent,
        # never scored.
        missing_fields = missing_plan_fields(
            destination_or_route=destination,
            activity_date=activity_date,
            participants=participants,
            route=route_basis,
            arrangements=arrangement_values,
        )

        # Ticket 06 AC1/AC5: the draft's own record. The eight supplementary
        # arrangements travel inside ``draft`` so the restore UX reads them from
        # the same object it reads the three minimal fields from. The key is
        # present only once the user has actually filled something: an
        # all-absent record keeps the ticket-01 draft shape rather than echoing
        # eight empty strings on every turn.
        draft_obj: dict[str, Any] = {
            "destination_or_route": destination,
            "activity_date": activity_date,
            "participants": participants,
        }
        if arrangement_values and any(v.strip() for v in arrangement_values.values()):
            draft_obj["arrangements"] = dict(arrangement_values)

        yield emit("plan.draft_state", {
            "stage": stage.value,
            "draft_id": record.draft_id if record else "",
            "conversation_id": req.conversation_id or "",
            "source": entry.source,
            "intent": intent or "",
            "summary": summary,
            # Ticket 04 AC5: the explicit marker, next to the same sentence in
            # ``summary``. ``route_plan`` appears only when the leaf handed back
            # a gate-passed basis for this very turn.
            "plan_type": plan_type,
            # The reference only — two ids. No fact, no value, no
            # ``report_allowed`` flag travels to the client from here.
            "route_selection": (
                {"comparison_id": route_reference[0], "candidate_id": route_reference[1]}
                if route_reference else None
            ),
            # Ticket 05 AC1/AC5: the 缺失字段清单. Each entry is a field the draft
            # does NOT hold, marked 待确认 (arrangement) or 待核验 (route fact);
            # an empty list would mean nothing is missing, which is why nothing
            # here ever fills an entry with a value it did not read.
            "missing_fields": [
                {"key": field.key, "label": field.label, "status": field.status}
                for field in missing_fields
            ],
            # Ticket 06 AC1/AC4/AC5: the deterministic verdict, emitted verbatim
            # from the same record that was just stored and that the summary
            # renders — so the wire, the row and the text agree. Codes, keys and
            # labels only: no route fact, no report flag (AC6). Absent until
            # there is a draft for the rules to judge.
            **({"rules": completeness.to_payload()} if completeness is not None else {}),
            # Ticket 07 AC2/AC5: the deterministic gate's refusal, emitted verbatim
            # from the leaf's own record — the stable block code, its readable copy
            # and the gate's reason lines. Absent when the gate refused nothing, so
            # no turn can imply a safety claim the comparison never made; it rides
            # this same event into ``run_events`` for the audit and into the
            # localStorage mirror for the panel after a refresh.
            **({"safety": safety.to_payload()} if safety is not None else {}),
            "draft": draft_obj,
        })

        text = summary
        envelope = emit("text", {"content": text})
        self._persist_plan_reply(req, run_id, envelope, text)
        yield envelope

        mark_terminal_safe(runs_repo, run_id, "done")
        yield emit("done", {})

    def _resolve_route_reference(
        self,
        req: ChatFlowRequest,
        run_id: str | None,
        comparison_id: str,
        candidate_id: str,
    ):
        """Read one draft route reference back through the comparison leaf.

        Deliberately not cached: a fresh :class:`agent_v2.route_tools.RouteSession`
        per call, whose ``load_result`` hits storage and whose
        ``select_candidate`` re-runs the deterministic gate. That is the whole
        point of storing a reference instead of facts (ticket 03), and it is why
        the plan path never needs a copy of a comparison — the kernel stays out
        of this stream, and so does any assumption that a client's ids are
        evidence.
        """
        from agent_v2.route_tools import RouteSession

        session = RouteSession(
            settings=self._settings,
            run_id=run_id,
            audit_sink=None,
            runs_repo=self._runs_repo,
            conversation_id=req.conversation_id,
        )
        return session.select_candidate(comparison_id, candidate_id)

    async def _research_route(
        self,
        req: ChatFlowRequest,
        run_id: str | None,
        destination_or_route: str,
    ):
        """Run one route-research turn through the comparison leaf (ticket 04).

        A fresh :class:`agent_v2.route_tools.RouteSession` like
        :meth:`_resolve_route_reference`: the leaf owns the data mode, the
        collection and — crucially — the verdict, because
        ``compare`` is the path that re-runs
        :func:`route_compare.service.assess_candidate`. This method decides
        nothing about the evidence and calls no model of its own; the ranking
        provider behind ``compare_routes`` is the one the route tool already
        uses, and it only ever ranks candidates that already passed the gate.

        The comparison lands in this run's event log via the emitted
        ``tool_result``, which is what lets the user select one of these
        candidates afterwards: ``load_result`` finds it by ``comparison_id``
        under this conversation and nothing is copied into the draft.
        """
        from agent_v2.route_tools import RouteSession

        session = RouteSession(
            settings=self._settings,
            run_id=run_id,
            audit_sink=None,
            runs_repo=self._runs_repo,
            conversation_id=req.conversation_id,
        )
        return await session.research_destination(destination_or_route)

    def _persist_plan_reply(
        self,
        req: ChatFlowRequest,
        run_id: str | None,
        envelope: EventEnvelope,
        text: str,
    ) -> None:
        """Append the plan reply to conversation history, best-effort.

        Uses the text event's OWN sequence rather than the hardcoded
        ``sequence=2`` the kernel path passes (the context-pack landmine):
        a run can now hold a plan text message and nothing collides.
        """
        runs_repo = self._runs_repo
        if runs_repo is None or run_id is None or not req.conversation_id or not text:
            return
        try:
            runs_repo.checkpoints.append_message(
                conversation_id=req.conversation_id,
                run_id=run_id,
                role="assistant",
                content=text,
                sequence=envelope.sequence,
            )
        except Exception:
            logger.exception(
                "chat_v2: plan assistant message persistence failed for run_id=%s",
                run_id,
            )

    # -- the chat stream --

    async def stream_chat(
        self,
        req: ChatFlowRequest,
        *,
        subject_id: str | None = None,
    ) -> AsyncGenerator[EventEnvelope | str, None]:
        """Drive one run and yield its events.

        Yields :class:`EventEnvelope` for every event — since Ticket 39
        the artifact flow is folded into this module, so the stream no
        longer mixes in pre-rendered adapter frames.
        """
        stream = EventStream(
            request_id=req.request_id,
            conversation_id=req.conversation_id or "",
        )

        # When v2 is disabled or the runtime is the Ticket 03 placeholder
        # (no ``runs`` attribute), ``runs_repo`` is None and persistence
        # is skipped — the stream still produces the same envelope
        # contract, just without durable replay.
        runs_repo = self._runs_repo
        memory = self._memory
        telemetry = self._telemetry
        telemetry_started = time.perf_counter()

        # Idempotent run creation. ``conversation_id`` defaults to a
        # derived value so single-shot runs without a client-supplied id
        # still persist.
        run_id: str | None = None
        persistence_error = False
        conversation_id = req.conversation_id or f"conv-{req.request_id}"
        conversation_forbidden = bool(
            memory is not None
            and subject_id
            and not memory.claim_conversation(subject_id, conversation_id)
        )
        if runs_repo is not None and not conversation_forbidden:
            try:
                run_id, created = runs_repo.create_run_idempotent(
                    request_id=req.request_id,
                    conversation_id=conversation_id,
                    profile_id=req.profile_id,
                    user_message=req.message,
                )
            except Exception:
                # Persistence is the source of truth for request_id
                # idempotency. Continuing in memory after a failed insert
                # could execute the model without a durable dedupe record.
                logger.exception(
                    "chat_v2: create_run_idempotent failed for request_id=%s; "
                    "failing closed",
                    req.request_id,
                )
                run_id = None
                created = False
                persistence_error = True
            else:
                persisted_run = runs_repo.get_run(run_id)
                if (
                    memory is not None
                    and subject_id
                    and persisted_run is not None
                    and not memory.owns_conversation(
                        subject_id, persisted_run.conversation_id
                    )
                ):
                    conversation_forbidden = True
                    run_id = None
                    created = True
                # Propagate the server-generated run_id to every built
                # envelope (spec decision #26). The frontend reads this
                # from the first event to call /snapshot, /cancel, and
                # /events.
                if run_id:
                    stream.run_id = run_id
                    if created and telemetry is not None:
                        telemetry.record(
                            run_id,
                            profile_name=req.profile_id,
                            status="running",
                            model_name=getattr(self._settings, "openai_model", ""),
                        )
                if not created:
                    # Spec AC2: duplicate request_id — replay stored
                    # events and stop. No model call, no tool
                    # re-execution.
                    async for envelope in self._replay_stored_events(
                        stream, runs_repo, run_id, req
                    ):
                        yield envelope
                    return

        # Build the emit closures. They wrap build + persist (+ render
        # for the legacy hooks) so the flow generators below stay linear
        # and persistence-aware.
        def _emit(event_type: str, payload: dict[str, Any]) -> EventEnvelope:
            envelope = stream.build(event_type, payload)
            if runs_repo is not None and run_id is not None:
                try:
                    runs_repo.persist_event(
                        run_id=run_id,
                        sequence=envelope.sequence,
                        event_type=envelope.type,
                        payload=envelope.payload,
                        request_id=req.request_id,
                        conversation_id=req.conversation_id or "",
                        event_id=envelope.event_id,
                        timestamp=envelope.timestamp,
                    )
                except Exception:
                    logger.exception(
                        "chat_v2: persist_event failed for request_id=%s seq=%d; "
                        "stream continues but reconnect may miss this event",
                        req.request_id,
                        envelope.sequence,
                    )
            if telemetry is not None and run_id is not None:
                try:
                    tool_name = payload.get("tool")
                    state = payload.get("status") or payload.get("decision")
                    error_code = payload.get("code") if event_type == "error" else None
                    telemetry.observe_event(
                        run_id,
                        event_type=event_type,
                        duration_ms=(time.perf_counter() - telemetry_started) * 1000,
                        tool_name=tool_name if isinstance(tool_name, str) else None,
                        state=state if isinstance(state, str) else None,
                        error_code=(
                            error_code if isinstance(error_code, str) else None
                        ),
                    )
                except Exception:
                    logger.exception(
                        "chat_v2: telemetry update failed for run_id=%s", run_id
                    )
            return envelope

        try:
            if persistence_error:
                yield _emit("error", {
                    "code": "persistence_unavailable",
                    "message": "运行状态暂时不可用，请稍后使用相同请求重新尝试。",
                })
                return

            if conversation_forbidden:
                yield _emit("error", {
                    "code": "conversation_forbidden",
                    "message": "Conversation not accessible",
                })
                return

            # v2 disabled → terminal error. The Gateway should not route
            # here when v2 is off, but the endpoint defends in depth. No
            # internal events have been processed, so unknown_dropped
            # stays 0.
            if not self._settings.agent_v2_enabled:
                mark_terminal_safe(runs_repo, run_id, "error")
                yield _emit("error", {
                    "code": "v2_disabled",
                    "message": "Chat v2 is disabled (AGENT_V2_ENABLED=false).",
                })
                return

            # Resolve profile (server-defined; client cannot override).
            try:
                profile = resolve_profile(req.profile_id)
            except UnknownProfileError as exc:
                mark_terminal_safe(runs_repo, run_id, "error")
                yield _emit("error", {
                    "code": "unknown_profile",
                    "message": str(exc),
                })
                return

            # Ticket 13 AC2 — cowork authorization gate. The cowork
            # profile is registered but NOT callable by default. Only an
            # explicit operator-set policy of ``trusted`` or ``staging``
            # authorizes the request. Clients cannot upgrade to cowork
            # by sending the profile id — this server-side gate is the
            # authority. The check runs AFTER profile resolution so an
            # unknown profile still returns ``unknown_profile`` (not
            # ``cowork_unauthorized``).
            if profile.profile_id == ProfileId.COWORK and not is_cowork_authorized(
                self._settings
            ):
                mark_terminal_safe(runs_repo, run_id, "error")
                yield _emit("error", {
                    "code": "cowork_unauthorized",
                    "message": (
                        "cowork profile 未授权 (AGENT_V2_COWORK_POLICY=disabled)。"
                        "请联系运维启用 trusted/staging 策略。"
                    ),
                })
                return

            # Model failure scenario: no API key configured. The contract
            # requires this to be a terminal ``error`` event, not an HTTP
            # 500. ``run_started`` is emitted first so the client sees the
            # run begin before the model-unavailable terminal (spec
            # decision #27).
            if not self._settings.openai_api_key:
                yield _emit("run_started", {})
                mark_terminal_safe(runs_repo, run_id, "error")
                yield _emit("error", {
                    "code": "model_unavailable",
                    "message": "主模型未配置 API Key (OPENAI_API_KEY 为空)。",
                })
                return

            # Ticket 01: plan-book entry. Routing here is structured
            # intent + deterministic rules ONLY — the kernel is never
            # consulted, so no model decides whether a plan book is
            # created (spec AC4; ticket 06 forbids external models for
            # plan judgement). Falls through when the rules say this is
            # ordinary chat.
            plan_entry = _resolve_plan_entry(req, profile)
            if plan_entry is not None:
                async for envelope in self._stream_plan_flow(
                    req, plan_entry, run_id=run_id, emit=_emit,
                ):
                    yield envelope
                return

            # Normal flow: run_started → progress → text → done. Only now
            # do we simulate unknown internal events that the adapter
            # would drop — the count reflects events actually seen during
            # real processing, not the pre-flight v2/profile/key gates
            # that never produced events.
            stream.drop_unknown("langchain_chain_start")
            stream.drop_unknown("uniharness_llm_token")

            # Spec decision #27 + AC2: ``run_started`` is a distinct
            # event type signaling the run has begun. It is the first
            # user-visible event in a successful flow; ``progress``
            # follows for ongoing status updates.
            yield _emit("run_started", {})

            yield _emit("progress", {
                "stage": "processing",
                "message": "正在生成回复...",
            })

            # Spec decision #27 + AC2: heartbeat keeps the SSE connection
            # alive during model thinking / tool calls. In the stub stage
            # the flow is fast, but emitting one here exercises the
            # heartbeat path so the frontend reducer and reconnect replay
            # handle it correctly before real LLM integration adds long
            # waits.
            yield _emit("heartbeat", {})

            # Poll for cancellation between major yields. The cancel API
            # transitions the DB row; we observe it and stop the stream
            # with a ``cancelled`` terminal (spec AC5).
            if is_cancelled(runs_repo, run_id):
                yield _emit("cancelled", {
                    "code": "cancelled",
                    "message": "运行已被取消。",
                })
                return

            if profile.profile_id == ProfileId.ROUTE_PLAN:
                # Ticket 29: route_plan is generated by the SAME real
                # UniHarness kernel turn as chat, with server-approved
                # AMap MCP servers wired per registry allowlist and
                # evidence-discipline prompt. No product-side direct
                # route flow exists anymore.
                async for envelope in self._stream_kernel_turn(
                    req, profile, run_id=run_id, emit=_emit,
                ):
                    if is_cancelled(runs_repo, run_id) and not stream.terminal_emitted:
                        yield _emit("cancelled", {
                            "code": "cancelled",
                            "message": "运行已被取消。",
                        })
                        return
                    yield envelope
                if not stream.terminal_emitted:
                    mark_terminal_safe(runs_repo, run_id, "done")
                    yield _emit("done", {})
                return

            if profile.profile_id == ProfileId.RAG:
                # Spec #13 phase 2: the RAG page's knowledge Q&A is a
                # single kernel turn with the knowledge-search tool and
                # the grounding gate. No memory-consent hook (the rag
                # profile has no memory tools). SSE contract matches
                # chat exactly: one `done` terminal after `text`.
                async for envelope in self._stream_kernel_turn(
                    req, profile, run_id=run_id, emit=_emit,
                ):
                    if is_cancelled(runs_repo, run_id) and not stream.terminal_emitted:
                        yield _emit("cancelled", {
                            "code": "cancelled",
                            "message": "运行已被取消。",
                        })
                        return
                    yield envelope
                if not stream.terminal_emitted:
                    if is_cancelled(runs_repo, run_id):
                        yield _emit("cancelled", {
                            "code": "cancelled",
                            "message": "运行已被取消。",
                        })
                        return
                    mark_terminal_safe(runs_repo, run_id, "done")
                    yield _emit("done", {})
                return

            if profile.profile_id == ProfileId.ARTIFACT:
                # Ticket 09: artifact flow, folded into this module by
                # Ticket 39. The sandbox is created and cleaned up inside
                # the flow; cancel/error paths run the cleanup in a
                # ``finally`` block so the sandbox is never leaked.
                async for envelope in self._stream_artifact_flow(
                    req, profile, run_id=run_id, emit=_emit,
                ):
                    yield envelope
                return

            if profile.profile_id == ProfileId.COWORK:
                # Ticket 13: cowork flow. Same sandbox lifecycle as
                # artifact (create → lease → cleanup in finally). The
                # flow plans the command via a non-executing kernel
                # turn, then pauses on the HIGH-risk sandbox_bash
                # approval; cancel/error paths run cleanup in
                # ``finally``. (Ticket 38 folded this flow into the
                # module.)
                async for envelope in self._stream_cowork_flow(
                    req, profile, run_id=run_id, emit=_emit,
                ):
                    yield envelope
                return

            # Ticket 32: sensitive candidates still surface as
            # consent_required events from the request layer (product API
            # owns consent); ordinary save/search are now kernel tool
            # calls inside the kernel turn.
            response_text: str | None = None
            if memory is not None and subject_id and memory.memory_enabled(subject_id):
                sensitive = extract_sensitive_memory(req.message)
                if sensitive is not None:
                    candidate = memory.create_candidate(subject_id, **sensitive)
                    if candidate is not None:
                        yield _emit("memory.consent_required", candidate)
                        response_text = "这条信息需要你逐条确认后才会保存。"

            if response_text is None:
                # Tickets 26/28/29/32: the real UniHarness kernel
                # generates the answer (bounded adapters + the ONE
                # product PermissionGate + registry-approved MCP for
                # route_plan + memory tools for the server-resolved
                # subject). No stub/direct-flow/v1 fallback; kernel
                # failures are stable terminal errors.
                async for envelope in self._stream_kernel_turn(
                    req, profile, run_id=run_id, emit=_emit,
                    memory=memory, subject_id=subject_id,
                ):
                    yield envelope
            else:
                # Memory-hook path produced its own response above.
                yield _emit("text", {"content": response_text})

            if not stream.terminal_emitted:
                if is_cancelled(runs_repo, run_id):
                    yield _emit("cancelled", {
                        "code": "cancelled",
                        "message": "运行已被取消。",
                    })
                    return

                mark_terminal_safe(runs_repo, run_id, "done")
                yield _emit("done", {})

        except TerminalEventAlreadyEmitted:
            # A second terminal was attempted (programmer error). Log and
            # stop; the stream already has its one terminal event.
            logger.error(
                "chat_v2: attempted to emit after terminal for request_id=%s",
                req.request_id,
            )
            return

        except Exception as exc:
            # Unexpected error. Emit a terminal error if the stream is
            # still open; otherwise the previous terminal stands.
            logger.exception("chat_v2: unexpected error for request_id=%s", req.request_id)
            mark_terminal_safe(runs_repo, run_id, "error")
            if not stream.terminal_emitted:
                yield _emit("error", {
                    "code": "internal_error",
                    "message": "内部错误，请重试。",
                })
            return

    # -- cowork flow (Ticket 38, originally Ticket 13) --

    async def _stream_cowork_flow(
        self,
        req: ChatFlowRequest,
        profile: CapabilityProfile,
        *,
        run_id: str | None,
        emit: EmitEnvelope,
    ) -> AsyncGenerator[EventEnvelope, None]:
        """cowork profile flow.

        Event sequence emitted by this flow:

            tool planning (non-executing kernel turn)
              → approval.required   # HIGH risk → user must approve
              → run.paused          # stream terminal; resume via /resume

        After the user's decision is recorded via
        :meth:`resolve_approval_decision`, the resume flow (Ticket 40)
        continues:

            approval.resolved
              → tool_result(sandbox_bash, ok | timeout | error)
              → text
              → done

        Spec AC2: cowork is authorized ONLY when
        ``settings.agent_v2_cowork_policy`` is ``trusted`` or
        ``staging``. The caller (:meth:`stream_chat`) checks this
        BEFORE dispatching here; this flow does not re-check.

        Spec AC3: PermissionGate evaluates ``sandbox_bash`` (HIGH) →
        ``needs_approval``. The approval row + run.paused terminal are
        persisted; the sandbox lease is held until the user decides.

        Spec AC4: the bash command runs inside the sandbox workspace
        with ``env={}``. The host environment is not accessible.

        Spec AC5: cancel observation between yields; cleanup runs in
        ``finally`` so the sandbox is never leaked. Timeout and
        outcome_unknown paths emit terminal ``error`` events.

        Spec AC6: this flow does NOT call ``generate_pdf``; the cowork
        profile inherits the artifact tools but the stub stage only
        exercises ``sandbox_bash``. A real LLM may chain multiple tool
        calls; each will go through its own approval + effect ledger.
        """
        runs_repo = self._runs_repo
        computer = self._computer_factory()
        preserve_sandbox = False

        # Provision sandbox workspace. Fail-closed: any ComputerError → terminal.
        try:
            workspace_path = computer.create_workspace(run_id or req.request_id)
        except SandboxQuotaError as exc:
            mark_terminal_safe(runs_repo, run_id, "error")
            yield emit("error", {
                "code": "sandbox_quota",
                "message": f"沙箱配额不足：{exc}",
            })
            return
        except SandboxCreationError as exc:
            mark_terminal_safe(runs_repo, run_id, "error")
            yield emit("error", {
                "code": "sandbox_creation_failed",
                "message": f"沙箱创建失败：{exc}",
            })
            return
        except ComputerError as exc:
            mark_terminal_safe(runs_repo, run_id, "error")
            yield emit("error", {
                "code": "sandbox_unavailable",
                "message": f"沙箱不可用：{exc}",
            })
            return

        # Persist the lease (best-effort).
        if runs_repo is not None and run_id is not None:
            try:
                runs_repo.create_sandbox_lease(
                    run_id=run_id,
                    sandbox_type=computer.sandbox_type,
                    workspace_path=workspace_path,
                )
            except Exception:
                logger.exception(
                    "run_orchestration: create_sandbox_lease failed for "
                    "run_id=%s; continuing without lease record (workspace=%s)",
                    run_id, workspace_path,
                )

        # Ticket 31: the bash command is planned by the real UniHarness
        # model loop (a non-executing turn — the planner has no shell
        # tool), then the durable-approval pause semantics take over
        # before anything runs inside the sandbox.
        from agent_v2.kernel import KernelModelFailure

        try:
            command_plan, _ = await self._kernel_turn(
                system_prompt=compose_cowork_planner_prompt(),
                message=req.message,
                settings=self._settings,
                run_id=run_id,
                checkpoint_store=self._kernel_checkpoints,
                extra_tools=(),
                kernel_gate=None,
                mcp_servers=None,
                computer=None,
            )
        except KernelModelFailure as exc:
            mark_terminal_safe(runs_repo, run_id, "error")
            yield emit("error", {"code": exc.code, "message": exc.message})
            return

        try:
            command_plan_obj = json.loads(
                command_plan.strip().removeprefix("```json").removesuffix("```").strip()
            )
            bash_command = str(command_plan_obj["command"])
            assert isinstance(bash_command, str) and bash_command.strip()
        except Exception:
            logger.warning(
                "run_orchestration: cowork command plan invalid for request_id=%s",
                req.request_id,
            )
            mark_terminal_safe(runs_repo, run_id, "error")
            yield emit("error", {
                "code": "bash_plan_invalid",
                "message": "无法从内核回合解析出合法命令，任务已终止。",
            })
            return
        bash_args = {"command": bash_command}

        try:
            # Cancel check before the permission gate.
            if is_cancelled(runs_repo, run_id):
                yield emit("cancelled", {
                    "code": "cancelled",
                    "message": "运行已被取消。",
                })
                return

            # Ticket 13 AC3 — PermissionGate check BEFORE the approval.
            # sandbox_bash is HIGH risk → needs_approval. The flow pauses
            # the run and emits run.paused as the stream terminal; the
            # sandbox lease is held until the user decides (spec AC4: the
            # same sandbox must be reused when the user approves).
            gate = PermissionGate()
            verdict = gate.evaluate(profile, "sandbox_bash", bash_args)
            if verdict.decision == PermissionDecision.DENY:
                mark_terminal_safe(runs_repo, run_id, "error")
                yield emit("error", {
                    "code": "tool_denied",
                    "message": f"工具 sandbox_bash 被拒绝：{verdict.reason}",
                })
                return
            if verdict.decision == PermissionDecision.NEEDS_APPROVAL:
                approval_id = compute_approval_id(
                    run_id=run_id or req.request_id,
                    tool_name="sandbox_bash",
                    args=bash_args,
                )
                created_at = _now_iso()
                expires_at = compute_expiry(created_at)
                impact = f"在沙箱内执行 Bash 命令：{bash_command}"
                if runs_repo is not None and run_id is not None:
                    try:
                        runs_repo.insert_approval(
                            approval_id=approval_id,
                            run_id=run_id,
                            conversation_id=req.conversation_id or f"conv-{req.request_id}",
                            tool_name="sandbox_bash",
                            args_canonical=canonical_args(bash_args),
                            risk_level=verdict.risk.value if verdict.risk else "high",
                            impact=impact,
                            created_at=created_at,
                            expires_at=expires_at,
                        )
                    except Exception:
                        logger.exception(
                            "run_orchestration: insert_approval failed for "
                            "run_id=%s; existing row (if any) is authoritative",
                            run_id,
                        )
                yield emit("approval.required", {
                    "approval_id": approval_id,
                    "tool": "sandbox_bash",
                    "args": bash_args,
                    "risk_level": verdict.risk.value if verdict.risk else "high",
                    "impact": impact,
                    "expires_at": expires_at,
                    "message": "需要您的批准才能在沙箱内执行 Bash 命令。",
                })
                yield emit("run.paused", {
                    "code": "awaiting_approval",
                    "approval_id": approval_id,
                    "message": "运行已暂停，等待用户批准高风险 Bash 操作。",
                })
                if runs_repo is not None and run_id is not None:
                    try:
                        runs_repo.mark_run_paused(run_id)
                    except RunAlreadyTerminal as exc:
                        logger.warning(
                            "run_orchestration: mark_run_paused failed for "
                            "run_id=%s: %s",
                            run_id, exc,
                        )
                        yield emit("cancelled", {
                            "code": "cancelled",
                            "message": "运行已被取消。",
                        })
                        return
                preserve_sandbox = True
                return

            # If we reach here, the gate returned ALLOW (unexpected for
            # sandbox_bash at HIGH risk). Treat as internal_error — the
            # approval threshold must not be bypassed.
            mark_terminal_safe(runs_repo, run_id, "error")
            yield emit("error", {
                "code": "internal_error",
                "message": "sandbox_bash 未走审批流程，已拒绝执行。",
            })
            return

        except TerminalEventAlreadyEmitted:
            logger.error(
                "run_orchestration: attempted to emit after terminal for "
                "request_id=%s",
                req.request_id,
            )
            return
        finally:
            if preserve_sandbox:
                logger.info(
                    "run_orchestration: preserving sandbox for paused cowork "
                    "run_id=%s",
                    run_id,
                )
            else:
                self._sandbox_cleanup(computer, workspace_path, runs_repo, run_id)

    # -- artifact flow (Ticket 39, originally Tickets 09 + 10) --

    async def _stream_artifact_flow(
        self,
        req: ChatFlowRequest,
        profile: CapabilityProfile,
        *,
        run_id: str | None,
        emit: EmitEnvelope,
        report_content: str | None = None,
        pdf_document: dict[str, Any] | None = None,
    ) -> AsyncGenerator[EventEnvelope, None]:
        """artifact profile flow.

        Event sequence emitted by this flow:

            tool_call(generate_pdf)
              → tool_result
              → artifact_candidate
              → artifact.available     # Ticket 10: host verified + persisted
              → text
              → done

        The outer :meth:`stream_chat` emits ``run_started`` → ``progress``
        → ``heartbeat`` before dispatching here, so the full wire sequence
        is: run_started → progress → heartbeat → tool_call → tool_result →
        artifact_candidate → artifact.available → text → done.

        Fail-closed contract: any ``ComputerError`` (creation / quota /
        timeout / cleanup) becomes a terminal ``error`` event. The run
        NEVER falls back to host execution.

        Ticket 10 export contract: after ``artifact_candidate`` the host
        builds a manifest, re-reads the bytes, re-derives size + SHA-256,
        and persists the artifact under an opaque id (driven through the
        adapter-injected ``artifact_export`` hook until Ticket 41). On
        success an ``artifact.available`` event is emitted carrying ONLY
        the opaque ``artifact_id`` + public metadata. On failure a
        ``failed`` evidence row is persisted BEFORE the sandbox is
        destroyed (spec AC4) — the cleanup ``finally`` runs only after
        the evidence row is saved.

        Ticket 30: the plan content is generated by the REAL UniHarness
        model loop; the run-scoped product Computer is surfaced to the
        kernel through ProductSandboxBridge so the model can call
        path-guarded sandbox_read/sandbox_write inside the workspace.
        """
        runs_repo = self._runs_repo
        computer = self._computer_factory()

        # Spec AC4: when evidence persistence fails inside the host
        # export, the sandbox MUST be preserved so operators can recover
        # the artifact bytes manually. This flag is set only on that
        # path; the ``finally`` block checks it to skip cleanup. The
        # lease is intentionally leaked — the error event tells the
        # operator which run_id needs investigation.
        preserve_sandbox = False

        # Provision sandbox workspace. Fail-closed: any ComputerError → terminal.
        try:
            workspace_path = computer.create_workspace(run_id or req.request_id)
        except SandboxQuotaError as exc:
            mark_terminal_safe(runs_repo, run_id, "error")
            yield emit("error", {
                "code": "sandbox_quota",
                "message": f"沙箱配额不足：{exc}",
            })
            return
        except SandboxCreationError as exc:
            mark_terminal_safe(runs_repo, run_id, "error")
            yield emit("error", {
                "code": "sandbox_creation_failed",
                "message": f"沙箱创建失败：{exc}",
            })
            return
        except ComputerError as exc:
            mark_terminal_safe(runs_repo, run_id, "error")
            yield emit("error", {
                "code": "sandbox_unavailable",
                "message": f"沙箱不可用：{exc}",
            })
            return

        # Persist the lease (best-effort). Lease persistence failure does not
        # block the run — the workspace path is logged so operators can find
        # the sandbox manually if the lease row is missing.
        if runs_repo is not None and run_id is not None:
            try:
                runs_repo.create_sandbox_lease(
                    run_id=run_id,
                    sandbox_type=computer.sandbox_type,
                    workspace_path=workspace_path,
                )
            except Exception:
                logger.exception(
                    "run_orchestration: create_sandbox_lease failed for "
                    "run_id=%s; continuing without lease record (workspace=%s)",
                    run_id, workspace_path,
                )

        try:
            # Cancel check before tool_call.
            if is_cancelled(runs_repo, run_id):
                yield emit("cancelled", {
                    "code": "cancelled",
                    "message": "运行已被取消。",
                })
                return

            # Ticket 11 — PermissionGate check BEFORE tool_call.
            # Spec AC3: "approval.required 后 graph 持久 interrupt，并以
            # run.paused 结束当前流；工具在批准前没有副作用".
            # The gate evaluates ``generate_pdf`` (HIGH risk) → needs_approval.
            # We persist the approval row, emit ``approval.required`` + the
            # ``run.paused`` stream terminal, mark the run paused, and return
            # WITHOUT destroying the sandbox (spec AC4: the same sandbox must
            # be reused when the user approves).
            gate = self._gate_factory() if self._gate_factory else PermissionGate()
            verdict = gate.evaluate(
                profile, "generate_pdf", {"filename": _ARTIFACT_FILENAME}
            )
            if verdict.decision == PermissionDecision.DENY:
                mark_terminal_safe(runs_repo, run_id, "error")
                yield emit("error", {
                    "code": "tool_denied",
                    "message": f"工具 generate_pdf 被拒绝：{verdict.reason}",
                })
                return
            if verdict.decision == PermissionDecision.NEEDS_APPROVAL:
                approval_id = compute_approval_id(
                    run_id=run_id or req.request_id,
                    tool_name="generate_pdf",
                    args={"filename": _ARTIFACT_FILENAME},
                )
                created_at = _now_iso()
                expires_at = compute_expiry(created_at)
                impact = "生成徒步计划 PDF 并导出到宿主存储"
                if runs_repo is not None and run_id is not None:
                    try:
                        runs_repo.insert_approval(
                            approval_id=approval_id,
                            run_id=run_id,
                            conversation_id=req.conversation_id or f"conv-{req.request_id}",
                            tool_name="generate_pdf",
                            args_canonical=canonical_args({"filename": _ARTIFACT_FILENAME}),
                            risk_level=verdict.risk.value if verdict.risk else "high",
                            impact=impact,
                            created_at=created_at,
                            expires_at=expires_at,
                        )
                    except Exception:
                        # Duplicate insert (same run/tool/args) is expected
                        # when the client reconnects to a paused run. The
                        # existing approval row is authoritative — log and
                        # continue to emit the pause events so the client
                        # sees a consistent terminal.
                        logger.exception(
                            "run_orchestration: insert_approval failed for "
                            "run_id=%s; existing row (if any) is authoritative",
                            run_id,
                        )
                yield emit("approval.required", {
                    "approval_id": approval_id,
                    "tool": "generate_pdf",
                    "args": {"filename": _ARTIFACT_FILENAME},
                    "risk_level": verdict.risk.value if verdict.risk else "high",
                    "impact": impact,
                    "expires_at": expires_at,
                    "message": "需要您的批准才能继续生成 PDF 并导出。",
                })
                yield emit("run.paused", {
                    "code": "awaiting_approval",
                    "approval_id": approval_id,
                    "message": "运行已暂停，等待用户批准高风险操作。",
                })
                # AC3: paused is a stream terminal, not a run terminal. The
                # run status transitions to ``paused`` so the resume endpoint
                # can pick it up; the sandbox lease is intentionally kept.
                if runs_repo is not None and run_id is not None:
                    try:
                        runs_repo.mark_run_paused(run_id)
                    except RunAlreadyTerminal as exc:
                        # Cancel API beat us to a terminal state — emit a
                        # cancelled terminal instead and let the finally
                        # clean up the sandbox (preserve_sandbox stays False).
                        logger.warning(
                            "run_orchestration: mark_run_paused failed for "
                            "run_id=%s: %s",
                            run_id, exc,
                        )
                        yield emit("cancelled", {
                            "code": "cancelled",
                            "message": "运行已被取消。",
                        })
                        return
                preserve_sandbox = True
                return

            # Ticket 30: the hiking-plan content is generated by the REAL
            # UniHarness model loop (see the method docstring).
            from agent_v2.kernel import KernelModelFailure
            from agent_v2.kernel_tools import (
                ProductSandboxBridge,
                SandboxReadTool,
                SandboxWriteTool,
            )

            bridge = ProductSandboxBridge(computer, workspace_path)
            sandbox_audits: list[dict[str, Any]] = []
            sandbox_tools = (
                SandboxReadTool(bridge=bridge, run_id=run_id, audit_sink=sandbox_audits.append),
                SandboxWriteTool(bridge=bridge, run_id=run_id, audit_sink=sandbox_audits.append),
            )
            # In-workspace read/write are kernel-governed self-service ops
            # (hard boundary = workspace chdir + path guard + env isolation);
            # the product policy ladder re-enters at generate_pdf/export below.
            try:
                plan_text, audits = (report_content, []) if report_content is not None else await self._kernel_turn(
                    system_prompt=compose_artifact_system_prompt(),
                    message=req.message,
                    settings=self._settings,
                    run_id=run_id,
                    checkpoint_store=self._kernel_checkpoints,
                    extra_tools=sandbox_tools,
                    kernel_gate=None,
                    mcp_servers=None,
                    computer=bridge,
                )
            except KernelModelFailure as exc:
                mark_terminal_safe(runs_repo, run_id, "error")
                yield emit("error", {"code": exc.code, "message": exc.message})
                return

            for entry in (*audits, *sandbox_audits):
                payload: dict[str, Any] = {"tool": entry.get("tool") or entry.get("server_id")}
                if entry.get("phase") == "denied":
                    payload["status"] = "denied"
                    payload["reason"] = entry.get("code")
                else:
                    payload["status"] = "ok" if entry.get("ok") else str(entry.get("code") or "error")
                    payload["duration_ms"] = entry.get("duration_ms")
                yield emit("tool_result", payload)

            plan_content = plan_text
            if pdf_document is not None:
                # Structured route-comparison reports render through the
                # designed PDF layout; the text report stays for the
                # conversation transcript.
                plan_content = json.dumps(pdf_document, ensure_ascii=False)
            yield emit("tool_call", {
                "tool": "generate_pdf",
                "args": {"filename": _ARTIFACT_FILENAME},
            })

            # Generate PDF inside sandbox. Fail-closed: timeout/error → terminal.
            try:
                relative_path = computer.generate_pdf(
                    workspace_path, plan_content, _ARTIFACT_FILENAME
                )
            except SandboxTimeoutError as exc:
                yield emit("tool_result", {
                    "tool": "generate_pdf",
                    "status": "timeout",
                    "content": f"PDF 生成超时：{exc}",
                })
                mark_terminal_safe(runs_repo, run_id, "error")
                yield emit("error", {
                    "code": "sandbox_timeout",
                    "message": f"沙箱命令超时：{exc}",
                })
                return
            except ComputerError as exc:
                yield emit("tool_result", {
                    "tool": "generate_pdf",
                    "status": "error",
                    "content": f"PDF 生成失败：{exc}",
                })
                mark_terminal_safe(runs_repo, run_id, "error")
                yield emit("error", {
                    "code": "artifact_generation_failed",
                    "message": f"artifact 生成失败：{exc}",
                })
                return

            yield emit("tool_result", {
                "tool": "generate_pdf",
                "status": "ok",
                "content": f"已在沙箱中生成 {relative_path}",
                "path": relative_path,
            })

            # Cancel check before artifact_candidate.
            if is_cancelled(runs_repo, run_id):
                yield emit("cancelled", {
                    "code": "cancelled",
                    "message": "运行已被取消。",
                })
                return

            yield emit("artifact_candidate", {
                "filename": relative_path,
                "mime_type": "application/pdf",
                "sandbox_type": computer.sandbox_type,
                "message": "已生成徒步计划 PDF，等待宿主侧导出。",
            })

            # Ticket 10/41: host-side verified export. Since T41 the
            # production implementation lives in
            # :mod:`agent_v2.artifacts` (:meth:`_export_artifact`); the
            # constructor hook remains only as a test seam. Failure here
            # is terminal — the run does NOT publish a half-built
            # artifact (spec AC4). The evidence row is saved BEFORE the
            # sandbox cleanup runs (in ``finally``).
            conversation_id = req.conversation_id or f"conv-{req.request_id}"
            try:
                export = self._export_artifact(
                    computer=computer,
                    workspace_path=workspace_path,
                    relative_path=relative_path,
                    declared_mime="application/pdf",
                    runs_repo=runs_repo,
                    run_id=run_id,
                    conversation_id=conversation_id,
                )
            except ArtifactEvidencePersistenceError as exc:
                # Spec AC4: evidence row could not be persisted — preserve
                # the sandbox so operators can recover artifact bytes. The
                # lease is intentionally leaked; the error event carries
                # the run_id for investigation.
                preserve_sandbox = True
                mark_terminal_safe(runs_repo, run_id, "error")
                yield emit("error", {
                    "code": "artifact_evidence_persistence_failed",
                    "message": (
                        f"artifact 证据持久化失败，sandbox 已保留以便排查：{exc}。"
                        f"请联系运维处理 run_id={run_id} 的沙箱。"
                    ),
                })
                return
            except ArtifactExportError as exc:
                mark_terminal_safe(runs_repo, run_id, "error")
                yield emit("error", {
                    "code": "artifact_export_failed",
                    "message": f"artifact 导出失败：{exc}",
                })
                return

            # Spec AC3: the artifact.available payload carries ONLY the
            # opaque artifact_id + public metadata. No sandbox path, no
            # host storage_path, no run_id (the envelope already carries
            # run_id for clients that need it).
            yield emit("artifact.available", {
                "artifact_id": export["artifact_id"],
                "public_filename": export["public_filename"],
                "mime_type": export["mime_type"],
                "size_bytes": export["size_bytes"],
                "sha256": export["sha256"],
                "expires_at": export["expires_at"],
                "message": "artifact 已就绪，可通过 /api/v2/artifacts/{artifact_id} 下载。",
            })

            response_text = (
                f"已在独立沙箱中生成徒步计划 PDF（{relative_path}）。"
                f"artifact 已导出，artifact_id={export['artifact_id']}，"
                "可通过 /api/v2/artifacts/{artifact_id} 下载。"
            )
            yield emit("text", {"content": response_text})

            mark_terminal_safe(runs_repo, run_id, "done")
            yield emit("done", {})

        finally:
            # Spec AC6: cleanup on completion OR cancellation OR error.
            # Always runs so the sandbox is never leaked. Failure is recorded
            # on the lease but does not change the terminal already emitted.
            #
            # Spec AC4 exception: when evidence persistence itself failed,
            # ``preserve_sandbox`` is set and the sandbox is NOT destroyed —
            # operators must recover the artifact bytes manually. The lease
            # is intentionally leaked and the run's error event tells the
            # operator which run_id to investigate.
            if preserve_sandbox:
                logger.warning(
                    "run_orchestration: preserving sandbox for run_id=%s "
                    "(evidence persistence failed or awaiting approval); "
                    "lease is intentionally leaked",
                    run_id,
                )
            else:
                self._sandbox_cleanup(computer, workspace_path, runs_repo, run_id)

    # -- resume flow (Ticket 40, originally Ticket 11/13 adapter code) --
    #
    # Spec AC4 + AC7: ``POST /api/v2/chat/resume/{run_id}`` continues a
    # paused run after the user decision. The orchestration module owns
    # the WHOLE flow end-to-end: replay of stored events (with original
    # sequence numbers), the continuation, sandbox re-execution via the
    # injected computer factory, and the shared effect ledger
    # (:func:`drive_effect`). The adapter keeps only HTTP checks (404 /
    # owner / 409-paused), lease→503 mapping, and SSE frame rendering.

    async def acquire_resume_leases(self, *, run_id: str, request_id: str) -> LeaseSet:
        """Acquire the run lease + (maybe) the sandbox capacity lease.

        The sandbox slot is only reserved when the run still holds an
        ``active`` sandbox lease (spec AC4: resume continues the SAME
        sandbox). Raises :class:`CapacityExceeded` after releasing any
        partially acquired lease; the adapter maps that to a 503.
        """
        runs_repo = self._runs_repo
        capacity = self._capacity
        if runs_repo is None or capacity is None:
            return LeaseSet(
                runs_repo=runs_repo,
                telemetry=self._telemetry,
                request_id=request_id,
            )
        run_lease = None
        sandbox_lease = None
        active_sandbox = runs_repo.get_sandbox_lease(run_id)
        try:
            run_lease = await capacity.acquire_run()
            if active_sandbox is not None and active_sandbox.status == "active":
                sandbox_lease = await capacity.acquire_sandbox(request_id)
        except CapacityExceeded:
            if run_lease is not None:
                run_lease.release()
            raise
        return LeaseSet(
            runs_repo=runs_repo,
            telemetry=self._telemetry,
            request_id=request_id,
            run_lease=run_lease,
            sandbox_lease=sandbox_lease,
        )

    # -- snapshot / event reads (Ticket 40) --

    def load_snapshot(self, run_id: str) -> RunSnapshot | None:
        """Read the product-facing run snapshot (or ``None``)."""
        if self._runs_repo is None:
            return None
        return self._runs_repo.get_snapshot(run_id)

    def load_events(self, run_id: str, *, from_seq: int = 0) -> list[dict[str, Any]]:
        """Read stored events with sequence > ``from_seq`` (replay)."""
        if self._runs_repo is None:
            return []
        return self._runs_repo.get_events_since(run_id=run_id, from_seq=from_seq)

    async def stream_resume(
        self,
        *,
        run_id: str,
        from_seq: int = 0,
    ) -> AsyncGenerator[EventEnvelope, None]:
        """Produce the resume event stream — spec AC4, AC5, AC7.

        Replay stored events > ``from_seq`` (with their ORIGINAL sequence
        numbers so the client's ordering is preserved), then (if the run
        is ``running`` and has a resolved approval) emit the
        continuation. The continuation reuses the SAME sandbox lease
        (AC4) and fails closed if the lease is missing (AC7).

        Idempotency (spec AC5): before emitting any new event, the flow
        inspects already-persisted events from prior resume attempts. If
        a step was already completed (e.g. ``tool_result`` with
        ``status=ok`` exists), it is skipped — the tool is NOT
        re-executed. Persisted events are the source of truth for "what
        already happened".

        Yields :class:`EventEnvelope` objects; the adapter renders SSE
        frames.
        """
        runs_repo = self._runs_repo
        telemetry = self._telemetry
        if runs_repo is None:
            yield EventEnvelope(
                version="2", sequence=1, type="error",
                payload={"code": "run_not_found", "message": f"运行 {run_id} 不存在。"},
            )
            return

        telemetry_started = time.perf_counter()
        run = runs_repo.get_run(run_id)
        if run is None:
            yield EventEnvelope(
                version="2", sequence=1, type="error",
                payload={"code": "run_not_found", "message": f"运行 {run_id} 不存在。"},
            )
            return

        # Replay stored events > from_seq (so the client can restore state).
        # We also scan ALL stored events (from_seq=0) to determine which
        # resume steps have already been completed (spec AC5 idempotency).
        replay_events = runs_repo.get_events_since(run_id=run_id, from_seq=from_seq)
        all_events = (
            replay_events if from_seq == 0
            else runs_repo.get_events_since(run_id=run_id, from_seq=0)
        )
        for evt in replay_events:
            yield EventEnvelope(
                version=evt["version"],
                sequence=evt["sequence"],
                type=evt["type"],
                payload=evt["payload"],
                request_id=evt["request_id"],
                conversation_id=evt["conversation_id"],
                run_id=evt.get("run_id", run_id),
                event_id=evt.get("event_id", ""),
                timestamp=evt.get("timestamp", ""),
            )

        # If the run is terminal, the replay is all the client gets. The
        # continuation was already emitted by a prior resume (or the run
        # was cancelled). Do not re-execute tools (spec AC5).
        if run.is_terminal:
            return

        # Set up the EventStream to continue from the last persisted sequence.
        snapshot = runs_repo.get_snapshot(run_id)
        last_seq = snapshot.sequence_cursor if snapshot else 0
        stream = EventStream(
            request_id=run.request_id,
            conversation_id=run.conversation_id,
            run_id=run_id,
        )
        stream.advance_to(last_seq)

        def emit(event_type: str, payload: dict[str, Any]) -> EventEnvelope:
            envelope = stream.build(event_type, payload)
            try:
                runs_repo.persist_event(
                    run_id=run_id,
                    sequence=envelope.sequence,
                    event_type=envelope.type,
                    payload=envelope.payload,
                    request_id=run.request_id,
                    conversation_id=run.conversation_id,
                    event_id=envelope.event_id,
                    timestamp=envelope.timestamp,
                )
            except Exception:
                logger.exception(
                    "run_orchestration: persist_event failed during resume "
                    "for run_id=%s seq=%d",
                    run_id, envelope.sequence,
                )
            if telemetry is not None:
                try:
                    tool_name = payload.get("tool")
                    state = payload.get("status") or payload.get("decision")
                    error_code = payload.get("code") if event_type == "error" else None
                    telemetry.observe_event(
                        run_id,
                        event_type=event_type,
                        duration_ms=(time.perf_counter() - telemetry_started) * 1000,
                        tool_name=tool_name if isinstance(tool_name, str) else None,
                        state=state if isinstance(state, str) else None,
                        error_code=(
                            error_code if isinstance(error_code, str) else None
                        ),
                    )
                except Exception:
                    logger.exception(
                        "run_orchestration: telemetry update failed during "
                        "resume for run_id=%s",
                        run_id,
                    )
            return envelope

        # Look up the resolved approval that authorized this resume.
        approvals = runs_repo.list_approvals_for_run(run_id)
        resolved = next((a for a in approvals if a.is_terminal), None)
        if resolved is None:
            mark_terminal_safe(runs_repo, run_id, "error")
            yield emit("error", {
                "code": "no_resolved_approval",
                "message": "运行已恢复但未找到已解决的审批。",
            })
            return

        # Scan persisted events to determine which resume steps already ran.
        # This is the spec AC5 idempotency guard: if a step's event is already
        # in the log, we skip the step entirely — no tool re-execution.
        stored_types = {evt["type"] for evt in all_events}
        has_approval_resolved = "approval.resolved" in stored_types
        has_tool_call = "tool_call" in stored_types
        # Ticket 13: ``has_tool_result_ok`` is per-tool — the resume flow
        # dispatches on ``resolved.tool_name`` (``generate_pdf`` or
        # ``sandbox_bash``), so the idempotency guard must check the matching
        # tool_result payload. Without this, a sandbox_bash resume would
        # incorrectly skip tool execution because no ``generate_pdf``
        # tool_result exists.
        has_tool_result_ok = any(
            evt["type"] == "tool_result"
            and evt["payload"].get("tool") == resolved.tool_name
            and evt["payload"].get("status") == "ok"
            for evt in all_events
        )
        has_artifact_available = "artifact.available" in stored_types
        has_text = "text" in stored_types

        # If the run was rejected, the only valid terminal is ``done``. If a
        # prior resume already emitted ``approval.resolved`` but the run is
        # still running (mid-stream crash before ``done``), emit ``done`` now
        # without re-emitting ``approval.resolved``.
        if resolved.status == "rejected":
            if not has_approval_resolved:
                yield emit("approval.resolved", {
                    "approval_id": resolved.approval_id,
                    "tool": resolved.tool_name,
                    "decision": "reject",
                    "status": resolved.status,
                    "message": "用户已拒绝操作，运行将结束。",
                })
            mark_terminal_safe(runs_repo, run_id, "done")
            yield emit("done", {
                "code": "rejected",
                "message": "用户已拒绝操作，运行结束。",
            })
            return

        # Approved path — the tool may or may not have executed yet.
        # AC7: reuse the SAME sandbox lease. If the lease is missing or
        # destroyed, fail closed — do NOT create a new sandbox.
        lease = runs_repo.get_sandbox_lease(run_id)
        if lease is None or lease.status != "active":
            if not has_approval_resolved:
                yield emit("approval.resolved", {
                    "approval_id": resolved.approval_id,
                    "tool": resolved.tool_name,
                    "decision": "approve",
                    "status": resolved.status,
                    "message": "用户已批准操作，继续执行。",
                })
            mark_terminal_safe(runs_repo, run_id, "error")
            yield emit("error", {
                "code": "sandbox_lease_lost",
                "message": (
                    "沙箱租赁已丢失或销毁，无法安全恢复运行（spec AC7）。"
                    "请重新发起请求。"
                ),
            })
            return

        # Reconstruct the Computer bound to the leased workspace through
        # the injected factory (the adapter hook resolves the adapter
        # global lazily so contract-test monkeypatch points keep working).
        try:
            computer = self._computer_factory(lease.sandbox_type)
        except ValueError as exc:
            mark_terminal_safe(runs_repo, run_id, "error")
            yield emit("error", {
                "code": "sandbox_type_invalid",
                "message": f"沙箱类型无效：{exc}",
            })
            return

        workspace_path = lease.workspace_path
        preserve_sandbox = False

        # Parse the approved args from the approval record (spec AC5: the
        # tool must execute with the EXACT args the user approved; hardcoded
        # args would bypass the canonical-args tampering check).
        try:
            approved_args = json.loads(resolved.args_canonical) if resolved.args_canonical else {}
        except (ValueError, TypeError):
            approved_args = {}
        artifact_filename = approved_args.get("filename", _ARTIFACT_FILENAME)

        try:
            # Step 1: emit approval.resolved (skip if already persisted).
            if not has_approval_resolved:
                yield emit("approval.resolved", {
                    "approval_id": resolved.approval_id,
                    "tool": resolved.tool_name,
                    "decision": "approve",
                    "status": resolved.status,
                    "message": "用户已批准操作，继续执行。",
                })

            # Ticket 13: dispatch on the approved tool name. ``sandbox_bash``
            # (cowork profile) runs a bash command inside the sandbox and
            # surfaces the captured stdout/stderr/exit_code. ``generate_pdf``
            # (artifact profile) runs the PDF generation + host export flow.
            # Both share the effect ledger + CAS + crash-recovery contract
            # (Ticket 12).
            if resolved.tool_name == "sandbox_bash":
                async for envelope in self._resume_cowork_bash_steps(
                    emit=emit,
                    runs_repo=runs_repo,
                    run_id=run_id,
                    computer=computer,
                    workspace_path=workspace_path,
                    resolved=resolved,
                    approved_args=approved_args,
                    has_tool_call=has_tool_call,
                    has_tool_result_ok=has_tool_result_ok,
                    has_artifact_available=has_artifact_available,
                    has_text=has_text,
                    all_events=all_events,
                    run=run,
                    snapshot=snapshot,
                ):
                    yield envelope
                # 助手因"并发在飞"退出时 effect 仍为 executing —— 保留沙箱与
                # 租约，避免销毁胜者正在使用的运行环境（双击恢复互删修复）。
                _bash_effect = runs_repo.get_effect(
                    compute_effect_key(
                        run_id=run_id, tool_name="sandbox_bash", args=approved_args
                    )
                )
                if (
                    _bash_effect is not None
                    and _bash_effect.status == EffectStatus.EXECUTING.value
                ):
                    preserve_sandbox = True
                return

            # Step 2: execute tool (skip if tool_result ok already persisted).
            # Ticket 39: the effect CAS ledger is a single implementation
            # (:func:`drive_effect`); Ticket 12: the effect row is the
            # authoritative source of truth for tool execution state; the
            # event log check (``has_tool_result_ok``) is a secondary guard.
            effect_key = compute_effect_key(
                run_id=run_id,
                tool_name="generate_pdf",
                args=approved_args,
            )
            external_idempotency_key = compute_external_idempotency_key(
                effect_key=effect_key,
                tool_name="generate_pdf",
            )
            outcome = EffectDriveResult()

            user_message = ""
            for msg in snapshot.messages if snapshot else []:
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            plan_content = user_message
            has_route_report = False
            # A route report is code-rendered from a persisted, verified
            # comparison; approval resume must not replace it with the prompt.
            # The structured document keeps the designed PDF layout on resume.
            for event in reversed(all_events):
                report = event.get("payload", {}).get("route_report")
                if isinstance(report, dict):
                    from route_compare.models import ComparisonResult
                    from route_compare.report_pdf import build_report_document
                    from route_compare.service import render_report
                    report_result = ComparisonResult.model_validate(report)
                    render_report(report_result)
                    plan_content = json.dumps(build_report_document(
                        report_result,
                        **_pdf_plan_status_inputs(runs_repo, run.conversation_id)),
                        ensure_ascii=False)
                    has_route_report = True
                    break

            async def _execute_pdf() -> dict[str, Any]:
                content = plan_content
                if not has_route_report:
                    from agent_v2.kernel import KernelModelFailure
                    from agent_v2.kernel_tools import ProductSandboxBridge
                    try:
                        content, _ = await self._kernel_turn(
                            system_prompt=compose_artifact_system_prompt(),
                            message=user_message, settings=self._settings, run_id=run_id,
                            checkpoint_store=self._kernel_checkpoints,
                            extra_tools=(), kernel_gate=None, mcp_servers=None,
                            computer=ProductSandboxBridge(computer, workspace_path),
                        )
                    except KernelModelFailure as exc:
                        raise ComputerError("计划正文生成失败，请重新生成。") from exc
                generated = computer.generate_pdf(
                    workspace_path, content, artifact_filename
                )
                return {"content": f"已在沙箱中生成 {generated}", "path": generated}

            async for envelope in drive_effect(
                runs_repo=runs_repo,
                run_id=run_id,
                conversation_id=run.conversation_id,
                copy=EFFECT_COPY_GENERATE_PDF,
                effect_key=effect_key,
                external_idempotency_key=external_idempotency_key,
                args_canonical=resolved.args_canonical,
                tool_args={"filename": artifact_filename},
                all_events=all_events,
                has_tool_call=has_tool_call,
                has_tool_result_ok=has_tool_result_ok,
                execute=_execute_pdf,
                emit=emit,
                outcome=outcome,
            ):
                yield envelope
            if outcome.status in ("failed", "outcome_unknown", "in_flight"):
                if outcome.preserve_sandbox:
                    preserve_sandbox = True
                return

            # Recover the relative_path for artifact_candidate + text from the
            # committed/executed effect result, falling back to the stored
            # tool_result event payload.
            if outcome.stored_result is not None:
                relative_path = outcome.stored_result.get("path", artifact_filename)
            else:
                tool_result_evt = next(
                    evt for evt in all_events
                    if evt["type"] == "tool_result"
                    and evt["payload"].get("tool") == "generate_pdf"
                    and evt["payload"].get("status") == "ok"
                )
                relative_path = tool_result_evt["payload"].get("path", artifact_filename)

            # Step 3: cancel check before artifact_candidate.
            if is_cancelled(runs_repo, run_id):
                yield emit("cancelled", {
                    "code": "cancelled",
                    "message": "运行已被取消。",
                })
                return

            # Step 4: artifact_candidate + host export (skip if
            # artifact.available already persisted — the export is a side
            # effect on the host).
            if not has_artifact_available:
                yield emit("artifact_candidate", {
                    "filename": relative_path,
                    "mime_type": "application/pdf",
                    "sandbox_type": computer.sandbox_type,
                    "message": "已生成徒步计划 PDF，等待宿主侧导出。",
                })

                try:
                    export = self._export_artifact(
                        computer=computer,
                        workspace_path=workspace_path,
                        relative_path=relative_path,
                        declared_mime="application/pdf",
                        runs_repo=runs_repo,
                        run_id=run_id,
                        conversation_id=run.conversation_id,
                    )
                except ArtifactEvidencePersistenceError as exc:
                    preserve_sandbox = True
                    mark_terminal_safe(runs_repo, run_id, "error")
                    yield emit("error", {
                        "code": "artifact_evidence_persistence_failed",
                        "message": (
                            f"artifact 证据持久化失败，sandbox 已保留以便排查：{exc}。"
                            f"请联系运维处理 run_id={run_id} 的沙箱。"
                        ),
                    })
                    return
                except ArtifactExportError as exc:
                    mark_terminal_safe(runs_repo, run_id, "error")
                    yield emit("error", {
                        "code": "artifact_export_failed",
                        "message": f"artifact 导出失败：{exc}",
                    })
                    return

                yield emit("artifact.available", {
                    "artifact_id": export["artifact_id"],
                    "public_filename": export["public_filename"],
                    "mime_type": export["mime_type"],
                    "size_bytes": export["size_bytes"],
                    "sha256": export["sha256"],
                    "expires_at": export["expires_at"],
                    "message": "artifact 已就绪，可通过 /api/v2/artifacts/{artifact_id} 下载。",
                })
            else:
                # Export already done — recover artifact_id from stored event
                # so the text event can reference it.
                avail_evt = next(
                    evt for evt in all_events
                    if evt["type"] == "artifact.available"
                )
                export = {
                    "artifact_id": avail_evt["payload"].get("artifact_id", ""),
                    "public_filename": avail_evt["payload"].get("public_filename", ""),
                }

            # Step 5: text response (skip if already persisted).
            if not has_text:
                response_text = (
                    f"已在原沙箱中生成徒步计划 PDF（{relative_path}）。"
                    f"artifact 已导出，artifact_id={export['artifact_id']}，"
                    "可通过 /api/v2/artifacts/{artifact_id} 下载。"
                )
                yield emit("text", {"content": response_text})

            # Step 6: done terminal.
            mark_terminal_safe(runs_repo, run_id, "done")
            yield emit("done", {})

        except TerminalEventAlreadyEmitted:
            logger.error(
                "run_orchestration: attempted to emit after terminal during "
                "resume for run_id=%s",
                run_id,
            )
            return
        except ArtifactEvidencePersistenceError as exc:
            preserve_sandbox = True
            mark_terminal_safe(runs_repo, run_id, "error")
            yield emit("error", {
                "code": "artifact_evidence_persistence_failed",
                "message": (
                    f"artifact 证据持久化失败，sandbox 已保留以便排查：{exc}。"
                    f"请联系运维处理 run_id={run_id} 的沙箱。"
                ),
            })
            return
        except Exception as exc:
            logger.exception(
                "run_orchestration: unexpected error during resume for run_id=%s",
                run_id,
            )
            mark_terminal_safe(runs_repo, run_id, "error")
            if not stream.terminal_emitted:
                yield emit("error", {
                    "code": "internal_error",
                    "message": "恢复运行时发生内部错误。",
                })
            return
        finally:
            if preserve_sandbox:
                logger.warning(
                    "run_orchestration: preserving sandbox for run_id=%s "
                    "during resume (evidence persistence failed); lease is "
                    "intentionally leaked",
                    run_id,
                )
            else:
                self._sandbox_cleanup(computer, workspace_path, runs_repo, run_id)

    async def _resume_cowork_bash_steps(
        self,
        *,
        emit: EmitEnvelope,
        runs_repo: RunRepository,
        run_id: str,
        computer: Computer,
        workspace_path: str,
        resolved: ApprovalRecord,
        approved_args: dict[str, Any],
        has_tool_call: bool,
        has_tool_result_ok: bool,
        has_artifact_available: bool,
        has_text: bool,
        all_events: list[dict[str, Any]],
        run: RunSnapshot,
        snapshot: RunSnapshot | None,
    ) -> AsyncGenerator[EventEnvelope, None]:
        """Resume steps for the cowork ``sandbox_bash`` tool (Ticket 13).

        Mirrors the ``generate_pdf`` resume structure but for bash execution:

        1. Effect ledger: ``authorized`` → CAS ``executing`` → ``committed``
           (or ``failed``/``outcome_unknown``) — via :func:`drive_effect`.
        2. Run the bash command via :meth:`Computer.run_bash` with timeout
           + output cap (spec AC3).
        3. Emit ``tool_call`` + ``tool_result`` events.
        4. Emit ``text`` response (skip if already persisted).
        5. Emit ``done`` terminal.

        Crash recovery (spec AC3): an effect found in ``executing`` state
        transitions to ``outcome_unknown`` and the run terminates — no
        auto-retry. CAS loser in ``executing`` state returns without
        touching the effect (Ticket 12 review H1).
        """
        bash_command = approved_args.get("command", "")
        bash_args = {"command": bash_command}

        effect_key = compute_effect_key(
            run_id=run_id,
            tool_name="sandbox_bash",
            args=approved_args,
        )
        external_idempotency_key = compute_external_idempotency_key(
            effect_key=effect_key,
            tool_name="sandbox_bash",
        )

        outcome = EffectDriveResult()

        def _execute_bash() -> dict[str, Any]:
            bash_result = computer.run_bash(
                workspace_path,
                bash_command,
                timeout_seconds=30.0,
                max_output_bytes=_COWORK_BASH_MAX_OUTPUT_BYTES,
            )
            return {
                "exit_code": bash_result.exit_code,
                "stdout": bash_result.stdout,
                "stderr": bash_result.stderr,
                "duration_ms": bash_result.duration_ms,
                "truncated": bash_result.truncated,
            }

        async for envelope in drive_effect(
            runs_repo=runs_repo,
            run_id=run_id,
            conversation_id=run.conversation_id,
            copy=EFFECT_COPY_SANDBOX_BASH,
            effect_key=effect_key,
            external_idempotency_key=external_idempotency_key,
            args_canonical=resolved.args_canonical,
            tool_args=bash_args,
            all_events=all_events,
            has_tool_call=has_tool_call,
            has_tool_result_ok=has_tool_result_ok,
            execute=_execute_bash,
            emit=emit,
            outcome=outcome,
        ):
            yield envelope
        if outcome.status in ("failed", "outcome_unknown", "in_flight"):
            return

        recovered_result = outcome.stored_result
        if recovered_result is None:
            # Tool already executed in a prior resume — recover the result
            # from the stored tool_result event for the text response.
            tool_result_evt = next(
                evt for evt in all_events
                if evt["type"] == "tool_result"
                and evt["payload"].get("tool") == "sandbox_bash"
                and evt["payload"].get("status") == "ok"
            )
            recovered_result = {
                "stdout": tool_result_evt["payload"].get("stdout", ""),
                "exit_code": tool_result_evt["payload"].get("exit_code", 0),
            }

        # Export the file produced by the approved Bash command through the
        # same verified host boundary as the artifact profile.
        if not has_artifact_available:
            yield emit("artifact_candidate", {
                "filename": _COWORK_STUB_FILENAME,
                "mime_type": "text/plain",
                "sandbox_type": computer.sandbox_type,
                "message": "Bash 任务输出已生成，等待宿主侧导出。",
            })
            try:
                export = self._export_artifact(
                    computer=computer,
                    workspace_path=workspace_path,
                    relative_path=_COWORK_STUB_FILENAME,
                    declared_mime="text/plain",
                    runs_repo=runs_repo,
                    run_id=run_id,
                    conversation_id=run.conversation_id,
                )
            except ArtifactExportError as exc:
                mark_terminal_safe(runs_repo, run_id, "error")
                yield emit("error", {
                    "code": "artifact_export_failed",
                    "message": f"artifact 导出失败：{exc}",
                })
                return
            yield emit("artifact.available", {
                **export,
                "message": "artifact 已就绪，可通过 /api/v2/artifacts/{artifact_id} 下载。",
            })
        else:
            available = next(
                event for event in all_events
                if event["type"] == "artifact.available"
            )
            export = available["payload"]

        # Cancel check before text.
        if is_cancelled(runs_repo, run_id):
            yield emit("cancelled", {
                "code": "cancelled",
                "message": "运行已被取消。",
            })
            return

        # Step 3: text response (skip if already persisted).
        if not has_text:
            stdout_summary = ""
            if recovered_result is not None:
                stdout_summary = recovered_result.get("stdout", "")
            response_text = (
                f"已在沙箱内执行 Bash 命令。输出：\n{stdout_summary}"
                f"artifact_id={export['artifact_id']}"
            )
            yield emit("text", {"content": response_text})

        # Step 4: done terminal.
        mark_terminal_safe(runs_repo, run_id, "done")
        yield emit("done", {})

    # -- approval decision state machine (Ticket 38, originally Ticket 11) --

    def resolve_approval_decision(
        self,
        approval_id: str,
        *,
        decision: str,
        decision_request_id: str,
    ) -> ApprovalDecisionOutcome:
        """Record the user's approve/reject decision — spec AC4, AC5, AC7.

        This is the single authority for approval state transitions
        (Ticket 38): idempotent duplicates, expiry, run-state guards,
        sandbox-lease fail-closed, and the paused → running transition
        all happen here, bound to the original run + tool-call identity
        carried by the persisted approval row. The adapter maps the
        returned outcome to HTTP status codes only.

        The caller (adapter) is responsible for HTTP-level checks that
        precede this method: 422 body validation, 503 persistence
        guard, 404 + conversation-owner authorization (which needs the
        approval row before any state transition).
        """
        runs_repo = self._runs_repo
        if runs_repo is None:
            # The adapter guards persistence availability (503) before
            # calling; fail closed if invoked without it anyway.
            raise RuntimeError("approval decision requires a run repository")

        approval = runs_repo.get_approval(approval_id)
        if approval is None:
            return ApprovalDecisionOutcome(status="approval_not_found", payload={})

        # AC5: idempotent duplicate (same decision_request_id) → return existing
        if (
            approval.decision_request_id is not None
            and approval.decision_request_id == decision_request_id
            and approval.is_terminal
        ):
            expected_status = "approved" if decision == "approve" else "rejected"
            if approval.status != expected_status:
                return ApprovalDecisionOutcome(
                    status="already_resolved",
                    payload={
                        "approval_id": approval_id,
                        "reason": "decision_request_conflict",
                        "current_status": approval.status,
                        "message": "同一审批请求标识不能用于不同的决定。",
                    },
                )
            return ApprovalDecisionOutcome(
                status="idempotent",
                payload=self._decision_payload(approval, idempotent=True),
            )

        # AC5: already resolved with a different decision_request_id → 409
        if approval.is_terminal:
            return ApprovalDecisionOutcome(
                status="already_resolved",
                payload={
                    "approval_id": approval_id,
                    "reason": "already_resolved",
                    "current_status": approval.status,
                    "message": (
                        f"Approval already resolved as {approval.status} with a "
                        "different decision_request_id."
                    ),
                },
            )

        # AC5: expired (TTL elapsed before decision arrived) → 409
        if is_expired(approval.expires_at):
            try:
                runs_repo.mark_approval_expired(approval_id)
            except Exception:
                logger.exception(
                    "run_orchestration: mark_approval_expired failed for %s",
                    approval_id,
                )
            return ApprovalDecisionOutcome(
                status="expired",
                payload={
                    "approval_id": approval_id,
                    "reason": "expired",
                    "message": "审批已过期，请重新发起请求。",
                },
            )

        # Validate run state
        run = runs_repo.get_run(approval.run_id)
        if run is None:
            return ApprovalDecisionOutcome(
                status="run_not_found",
                payload={
                    "approval_id": approval_id,
                    "reason": "run_not_found",
                    "message": "审批关联的运行不存在。",
                },
            )

        if run.is_terminal:
            # The run was cancelled (or hit an error) while paused. The
            # pause path preserved the sandbox lease for the resume; now
            # that the run is terminal, the lease must be cleaned up to
            # avoid a leak (spec AC6). We mark the lease as destroyed
            # with a failure reason; the actual sandbox process is
            # best-effort cleaned by the operator because we no longer
            # have the Computer handle.
            if run.status == "cancelled":
                try:
                    lease = runs_repo.get_sandbox_lease(approval.run_id)
                    if lease is not None and lease.status == "active":
                        runs_repo.destroy_sandbox_lease(
                            run_id=approval.run_id,
                            failure_reason=(
                                f"run cancelled while paused (approval {approval_id})"
                            ),
                        )
                except Exception:
                    logger.exception(
                        "run_orchestration: destroy_sandbox_lease failed for "
                        "run_id=%s during cancelled-while-paused cleanup",
                        approval.run_id,
                    )
            return ApprovalDecisionOutcome(
                status="run_terminal",
                payload={
                    "approval_id": approval_id,
                    "reason": "run_terminal",
                    "current_run_status": run.status,
                    "message": f"运行已终态（{run.status}），无法继续。",
                },
            )

        if run.status != "paused":
            # Run is running/done/error/cancelled — not awaiting approval.
            return ApprovalDecisionOutcome(
                status="run_not_paused",
                payload={
                    "approval_id": approval_id,
                    "reason": "run_not_paused",
                    "current_run_status": run.status,
                    "message": (
                        f"运行状态为 {run.status}，不在等待审批。"
                        "可能 decision 已被处理。"
                    ),
                },
            )

        # AC7: sandbox lease must be active. If the lease is missing or
        # already destroyed, fail closed — do NOT resume into a new
        # sandbox.
        lease = runs_repo.get_sandbox_lease(approval.run_id)
        if lease is None or lease.status != "active":
            return ApprovalDecisionOutcome(
                status="sandbox_lease_lost",
                payload={
                    "approval_id": approval_id,
                    "reason": "sandbox_lease_lost",
                    "message": (
                        "沙箱租赁已丢失或销毁，无法安全恢复运行（spec AC7）。"
                        "请重新发起请求。"
                    ),
                },
            )

        # Persist the decision. resolve_approval raises RunAlreadyTerminal
        # if the approval was resolved by a concurrent request between our
        # ``get_approval`` and this call — surface as 409 already_resolved.
        target_status = "approved" if decision == "approve" else "rejected"
        try:
            resolved = runs_repo.resolve_approval(
                approval_id=approval_id,
                status=target_status,
                decision_request_id=decision_request_id,
            )
        except RunAlreadyTerminal:
            return ApprovalDecisionOutcome(
                status="already_resolved",
                payload={
                    "approval_id": approval_id,
                    "reason": "already_resolved",
                    "message": "审批已被并发请求解决。",
                },
            )

        if resolved is None:
            return ApprovalDecisionOutcome(status="approval_not_found", payload={})

        # Transition run ``paused`` → ``running`` so the resume endpoint
        # can pick it up. If the cancel API beat us, the run is now
        # terminal and the resume endpoint will surface the cancelled
        # state.
        try:
            runs_repo.resume_run(approval.run_id)
        except RunAlreadyTerminal as exc:
            logger.warning(
                "run_orchestration: resume_run failed for run_id=%s: %s",
                approval.run_id, exc,
            )
            return ApprovalDecisionOutcome(
                status="run_terminal",
                payload={
                    "approval_id": approval_id,
                    "reason": "run_terminal",
                    "message": "运行已终态，无法恢复。",
                },
            )

        return ApprovalDecisionOutcome(
            status="recorded",
            payload=self._decision_payload(resolved, idempotent=False),
        )

    @staticmethod
    def _decision_payload(
        approval: ApprovalRecord, *, idempotent: bool
    ) -> dict[str, Any]:
        """Public decision body (minus ``resume_url`` — adapter concern)."""
        return {
            "approval_id": approval.approval_id,
            "run_id": approval.run_id,
            "tool": approval.tool_name,
            "decision": (
                "approve" if approval.status == "approved" else "reject"
            ),
            "status": approval.status,
            "idempotent": idempotent,
        }

    # -- helpers --

    async def _replay_stored_events(
        self,
        stream: EventStream,
        runs_repo: RunRepository,
        run_id: str,
        req: ChatFlowRequest,
    ) -> AsyncGenerator[EventEnvelope, None]:
        stored = runs_repo.get_events_since(run_id=run_id, from_seq=0)
        has_terminal = False
        max_seq = 0
        for evt in stored:
            envelope = EventEnvelope(
                version=evt["version"],
                sequence=evt["sequence"],
                type=evt["type"],
                payload=evt["payload"],
                request_id=evt["request_id"],
                conversation_id=evt["conversation_id"],
                run_id=evt.get("run_id", run_id),
                event_id=evt.get("event_id", ""),
                timestamp=evt.get("timestamp", ""),
            )
            if envelope.sequence > max_seq:
                max_seq = envelope.sequence
            if envelope.type in TERMINAL_EVENT_TYPES:
                has_terminal = True
            yield envelope
        if has_terminal:
            return
        # No terminal in storage (process crashed mid-stream). Synthesize
        # one from the run's terminal state so the client never observes
        # a stream without a terminal. Advance the stream counter past
        # the replayed events so the synthesized terminal sequence is
        # strictly greater than any stored event (spec AC4 monotonicity),
        # and persist it so future reconnects see the terminal.
        snapshot = runs_repo.get_snapshot(run_id)
        terminal_type = synthesize_terminal_type(snapshot)
        if terminal_type is not None:
            stream.advance_to(max_seq)
            terminal_envelope = stream.build(terminal_type, {
                "code": "reconnected",
                "message": "运行已结束，重连时未观察到终端事件。",
            })
            try:
                runs_repo.persist_event(
                    run_id=run_id,
                    sequence=terminal_envelope.sequence,
                    event_type=terminal_envelope.type,
                    payload=terminal_envelope.payload,
                    request_id=req.request_id,
                    conversation_id=req.conversation_id or "",
                )
            except Exception:
                logger.exception(
                    "chat_v2: persist synthesized terminal failed "
                    "for request_id=%s; client still receives the frame",
                    req.request_id,
                )
            yield terminal_envelope

    async def _stream_kernel_turn(
        self,
        req: ChatFlowRequest,
        profile: CapabilityProfile,
        *,
        run_id: str | None,
        emit: EmitEnvelope,
        memory: Any | None = None,
        subject_id: str | None = None,
    ) -> AsyncGenerator[EventEnvelope, None]:
        """Shared T26/T28/T29 kernel turn used by chat and route_plan profiles."""
        from agent_v2.kernel import KernelModelFailure
        from agent_v2.kernel_tools import (
            build_chat_extra_tools,
            build_chat_kernel_gate,
            build_profile_mcp_servers,
            build_subagent_definitions,
        )

        audit_entries: list[dict[str, Any]] = []
        mcp_servers: dict[str, Any] = {}
        mcp_required = getattr(profile, "allowed_mcp_servers", None)
        has_comparison = "route_compare" in profile.allowed_tools
        route_session = None
        if has_comparison:
            from agent_v2.route_tools import RouteSession
            route_session = RouteSession(settings=self._settings, run_id=run_id,
                audit_sink=audit_entries.append, runs_repo=self._runs_repo,
                conversation_id=req.conversation_id)
        # Comparison collects MCP inside its bounded leaf. Mounting raw MCP
        # here would make a failed handshake abort before gaps can be rendered.
        if mcp_required and not has_comparison:
            mcp_servers, mcp_audits = build_profile_mcp_servers(profile, self._settings)
            if not mcp_servers:
                # Every allowlisted evidence source is unavailable → fail
                # closed instead of fabricating a plan (Ticket 29 AC5).
                mark_terminal_safe(self._runs_repo, run_id, "error")
                yield emit("error", {
                    "code": "evidence_unavailable",
                    "message": "关键证据源（地图/天气 MCP）不可用，无法提供可靠路线规划。",
                })
                return
            audit_entries.extend(mcp_audits)

        try:
            extra_tools = build_chat_extra_tools(
                profile,
                run_id=run_id,
                audit_sink=audit_entries.append,
                memory_service=memory,
                subject_id=subject_id,
                settings=self._settings,
                route_session=route_session,
            )
            kernel_gate = build_chat_kernel_gate(
                profile,
                run_id=run_id,
                audit_sink=audit_entries.append,
            ) if extra_tools or mcp_servers else None

            # Ticket 33: fixed bounded subagent definitions — only roles
            # whose tool surface and MCP needs are satisfiable on THIS
            # kernel turn.
            sub_defs, sub_audits = build_subagent_definitions(
                profile,
                self._settings,
                extra_tool_names={t.name for t in extra_tools},
                available_mcp_servers=set(mcp_servers),
            )
            audit_entries.extend(sub_audits)

            system_prompt = (
                compose_route_plan_system_prompt(req, profile)
                if profile.profile_id == ProfileId.ROUTE_PLAN
                else (
                    compose_rag_system_prompt(req, profile)
                    if profile.profile_id == ProfileId.RAG
                    else compose_chat_system_prompt(req, profile)
                )
            )

            response_text, kernel_audit_entries = await self._kernel_turn(
                system_prompt=system_prompt,
                message=req.message,
                settings=self._settings,
                run_id=run_id,
                checkpoint_store=self._kernel_checkpoints,
                extra_tools=tuple(extra_tools),
                kernel_gate=kernel_gate,
                mcp_servers=mcp_servers or None,
                subagent_definitions=sub_defs or None,
            )
            audit_entries.extend(kernel_audit_entries)
        except KernelModelFailure as exc:
            logger.warning(
                "chat_v2: kernel failure code=%s request_id=%s", exc.code, req.request_id
            )
            mark_terminal_safe(self._runs_repo, run_id, "error")
            yield emit("error", {
                "code": exc.code,
                "message": exc.message,
            })
            return

        # Observe cancellation before building/persisting model output.
        # Dropping an already-built envelope in the caller would leave
        # it in replay even though it was never delivered to the client.
        if is_cancelled(self._runs_repo, run_id):
            yield emit("cancelled", {
                "code": "cancelled",
                "message": "运行已被取消。",
            })
            return

        if has_comparison:
            from route_compare.models import ComparisonResult
            from route_compare.service import comparison_text
            comparisons = [entry["route_comparison"] for entry in audit_entries
                           if isinstance(entry.get("route_comparison"), dict)]
            response_text = (comparison_text(ComparisonResult.model_validate(comparisons[-1])) if comparisons else
                "请提供要比较的路线名、所在城市与各路线起终点。尚未完成可核验比较，无法给出可靠推荐。")

        if profile.profile_id in (ProfileId.CHAT, ProfileId.RAG):
            if knowledge_search_attempted_without_evidence(audit_entries):
                response_text = _NO_KNOWLEDGE_EVIDENCE_RESPONSE
            elif has_hiking_knowledge_evidence(audit_entries) and unsupported_distance_numbers(
                response_text, audit_entries
            ):
                response_text = _UNSUPPORTED_DISTANCE_RESPONSE

        # Ticket 28/29 AC6: public evidence of degraded sources, executed
        # and denied tool calls — bound to the run identity, emitted
        # before the final text so sequence stays monotonic with exactly
        # one terminal.
        for entry in audit_entries:
            payload: dict[str, Any] = {"tool": entry.get("tool") or entry.get("server_id")}
            if entry.get("phase") == "call":
                payload["args"] = entry.get("args", {})
                if entry.get("server_id"):
                    payload["server_id"] = entry["server_id"]
                yield emit("tool_call", payload)
                continue
            if entry.get("phase") == "mcp_unavailable":
                payload["status"] = "mcp_server_unavailable"
            elif entry.get("phase") == "denied":
                payload["status"] = "denied"
                payload["reason"] = entry.get("code")
            else:
                payload["status"] = "ok" if entry.get("ok") else str(entry.get("code") or "error")
                payload["duration_ms"] = entry.get("duration_ms")
            # A08: the MCP source keeps its audit field name (server_id).
            # The old ``server`` rename made the field vanish for the
            # frontend, which reads server_id — "which source is down"
            # was lost on the wire.
            if entry.get("server_id"):
                payload["server_id"] = entry.get("server_id")
            # A08: human-readable one-liner so the trace panel shows what
            # each step did instead of an empty row.
            payload["content"] = _tool_result_summary(entry)
            # Spec #13 phase 2 (T05): bounded evidence (has_chunks/sources/
            # chunk summaries) rides on the public tool_result event so the
            # frontend renders the 检索来源 block (A09); the legacy RAG-page
            # adapter was removed in A11.
            if isinstance(entry.get("evidence"), dict):
                payload["evidence"] = entry["evidence"]
            for key in ("route_comparison", "route_report"):
                if isinstance(entry.get(key), dict):
                    payload[key] = entry[key]
            yield emit("tool_result", payload)

        reports = [entry["route_report"] for entry in audit_entries if isinstance(entry.get("route_report"), dict)]
        if reports:
            from route_compare.models import ComparisonResult
            from route_compare.report_pdf import build_report_document
            from route_compare.service import render_report
            report_result = ComparisonResult.model_validate(reports[-1])
            report_content = render_report(report_result)
            async for envelope in self._stream_artifact_flow(req, resolve_profile("artifact"),
                run_id=run_id, emit=emit, report_content=report_content,
                pdf_document=build_report_document(
                    report_result,
                    **_pdf_plan_status_inputs(self._runs_repo, req.conversation_id))):
                yield envelope
            return

        yield emit("text", {"content": response_text})

        # Spec #13 phase 2 (T07): persist the assistant reply so
        # conversation history (RAG page restore, snapshots) sees both
        # sides. Best-effort: history loss must not fail the stream.
        if req.conversation_id and response_text and self._runs_repo is not None:
            try:
                self._runs_repo.checkpoints.append_message(
                    conversation_id=req.conversation_id,
                    run_id=run_id,
                    role="assistant",
                    content=response_text,
                    sequence=2,
                )
            except Exception:
                logger.exception(
                    "chat_v2: assistant message persistence failed for run_id=%s",
                    run_id,
                )


# ---- Paused-run janitor (Ticket 41) -----------------------------------------
#
# 审批暂停的 run 在流结束时保留 CapacityGuard 沙箱槽位等待用户决策
# （guarded_stream 的 keep_reserved 分支）。用户弃用后没有任何正常路径
# 释放该槽位（只有 cancel 和 resume 会释放），``max_sandboxes=10`` 会被
# 逐渐耗尽直到持续 503。janitor 周期回收暂停超过 TTL 的 run。
#
# Ticket 41: the sweep implementation belongs to this orchestration
# module; the startup assembly (``main.py`` lifespan) only wires it up.

_PAUSED_RUN_TTL_SECONDS = 24 * 3600
_PAUSED_RUN_SWEEP_INTERVAL_SECONDS = 600


def sweep_expired_paused_runs(runtime: Any) -> int:
    """回收超时暂停 run 的沙箱租约与容量槽位，返回回收数量。"""
    runs_repo = getattr(runtime, "runs", None)
    capacity = getattr(runtime, "capacity", None)
    if runs_repo is None:
        return 0
    from datetime import timedelta

    cutoff = (
        datetime.now(timezone.utc) - timedelta(seconds=_PAUSED_RUN_TTL_SECONDS)
    ).isoformat()
    reclaimed = 0
    for run in runs_repo.list_paused_runs_updated_before(cutoff):
        # 复核状态：sweep 与用户决策/resume 并发时跳过已不处于 paused 的 run。
        current = runs_repo.get_run(run.run_id)
        if current is None or current.status != "paused":
            continue
        sandbox = runs_repo.get_sandbox_lease(run.run_id)
        if sandbox is not None and sandbox.status == "active":
            try:
                runs_repo.destroy_sandbox_lease(
                    run_id=run.run_id,
                    failure_reason="paused run expired (approval abandoned)",
                )
            except Exception:
                logger.exception(
                    "run_orchestration: paused-run sweep failed to destroy lease for run_id=%s",
                    run.run_id,
                )
                continue
        try:
            runs_repo.mark_run_error(run.run_id)
        except Exception:
            logger.exception(
                "run_orchestration: paused-run sweep failed to mark run_id=%s terminal",
                run.run_id,
            )
            continue
        if capacity is not None:
            capacity.release_sandbox(run.request_id)
        reclaimed += 1
        logger.warning(
            "run_orchestration: reclaimed expired paused run run_id=%s (sandbox slot released)",
            run.run_id,
        )
    return reclaimed


def start_paused_run_sweeper(runtime: Any) -> "asyncio.Task[None]":
    """启动暂停超时回收周期任务（lifespan 装配后调用，关停时 cancel）。"""

    async def _loop() -> None:
        while True:
            await asyncio.sleep(_PAUSED_RUN_SWEEP_INTERVAL_SECONDS)
            try:
                sweep_expired_paused_runs(runtime)
            except Exception:
                logger.exception("run_orchestration: paused-run sweep iteration failed")

    return asyncio.create_task(_loop(), name="paused-run-sweeper")
