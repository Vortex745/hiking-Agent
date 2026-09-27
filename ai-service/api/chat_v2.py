"""Chat v2 SSE endpoint — product-owned HTTP/SSE contract.

This router exposes ``POST /api/v2/chat/sse``. It is the only public
chat surface — the v1 ``/api/v1/chat/sse`` endpoint has been removed.

Ticket 04 delivers the ``chat`` profile contract:

* Strict request model — client may only send ``message``, ``request_id``,
  ``conversation_id``, ``profile_id`` and ``current_location``. Any attempt
  to inject model config, credentials, tool overrides, or MCP servers is
  rejected with HTTP 400 (Pydantic ``extra="forbid"``).
* Versioned event envelope — every frame is ``{"version":"2",...}``. Raw
  LangChain / UniHarness events never reach the wire; the adapter maps or
  drops them.
* Monotonic sequence + exactly one terminal event — enforced by
  :class:`agent_v2.events.EventStream`.
* Unknown internal events dropped and counted — ``unknown_dropped`` is
  attached to the terminal ``done`` payload for monitoring.

Ticket 05 adds the ``route_plan`` profile contract:

* route_plan extends chat with route_research, web_search, web_scraping
  tools plus two server-registered AMap MCP servers (weather + map).
* Intake may use scenario / slots / history / authorized location to
  pick a prompt and prefetch, but it MUST NOT upgrade ``chat`` to
  ``route_plan``. The profile id is a request, not authorization.
* MCP server ids, connection parameters, and tool names all come from
  :mod:`agent_v2.mcp_registry`. Clients cannot add servers or raw tools.
* MCP credentials never appear in sandbox, product events, default logs,
  or telemetry. :class:`McpServerConfig` stores key *names* only.
* When MCP / Web / map / weather is unavailable, the relevant tools are
  marked unavailable and the response explicitly states evidence is
  missing — no fabricated realtime information.

The actual LLM call is not in scope for Tickets 04-05. The stub produces
deterministic responses based on message pattern so contract tests can
verify the envelope without an API key. Real LLM integration arrives
with the UniHarness factory swap in a later ticket.

Ticket 09 adds the ``artifact`` profile contract:

* artifact uses a run-scoped sandbox (E2B in production, LocalNative in
  development) — never the host filesystem. The sandbox is created per
  run, recorded as a lease in ``agent_control.sqlite``, and cleaned up
  on run completion or cancellation (spec AC1, AC6).
* Fail-closed: sandbox creation / quota / timeout / cleanup failures
  raise :class:`ComputerError` and the run terminates with an ``error``
  event. The run NEVER falls back to executing on the host (spec AC5).
* Tool surface is E2B-scoped: ``sandbox_read``/``write``/``edit``/
  ``glob``/``grep`` + ``generate_pdf``. No Bash, no MCP, no host paths
  (spec decision #11, #12, #14, #20).
* The artifact flow emits ``artifact_candidate`` after the PDF is
  generated so the host can export it. ``cancelled`` is observed
  between yields; cleanup runs in ``finally`` so cancel/error paths
  still destroy the sandbox.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from agent_v2.computer import (
    Computer,
    SandboxCleanupError,
    create_computer,
)
from agent_v2.events import (
    EventEnvelope,
)
from agent_v2.observability import CapacityExceeded
from agent_v2.permissions import (
    PermissionGate,
)
from agent_v2.run_orchestration import (
    ChatFlowRequest,
    RunOrchestrator,
)
from agent_v2.runs import (
    RunRepository,
)
from config import settings
from api.memory_v2 import resolve_request_subject, set_subject_cookie
from observability.langfuse_setup import reset_langfuse_session, set_langfuse_session

logger = logging.getLogger("ai-service.chat_v2")

chat_v2_router = APIRouter(prefix="/chat")
# Ticket 10: artifact download endpoint lives under /api/v2/artifacts so
# the URL is stable and decoupled from the chat run lifecycle. The chat
# router owns the run stream; this router owns the published artifact.
artifacts_v2_router = APIRouter(prefix="/artifacts")
# Ticket 11: approval decision endpoint lives under /api/v2/chat so it
# shares the chat router's prefix. The endpoint resumes a paused run
# after the user approves or rejects a high-risk tool call.
# ``approvals_v2_router`` is mounted at /api/v2/approvals so the URL is
# decoupled from any specific run — the approval_id is the opaque
# capability token that authorizes the decision.
approvals_v2_router = APIRouter(prefix="/approvals")


# ---- Paused-run janitor -----------------------------------------------------
#
# Ticket 41: the sweep implementation moved to
# :mod:`agent_v2.run_orchestration` (:func:`sweep_expired_paused_runs` /
# :func:`start_paused_run_sweeper`). The startup assembly (``main.py``)
# imports it from there; the adapter keeps no orchestration logic.
#
# Ticket 41 likewise relocated the host-side verified artifact export to
# :mod:`agent_v2.artifacts` (:func:`export_artifact_to_host` /
# :func:`persist_failed_artifact`); the orchestrator's ``artifact_export``
# hook defaults to it, so the adapter carries no export code either.


# ---- Request model --------------------------------------------------------


class CurrentLocationV2(BaseModel):
    """Optional user location for context-aware responses.

    Field names match v1 ``CurrentLocationPayload`` so the frontend can
    reuse its existing geo capture. All fields are optional — the server
    decides how to use them; clients cannot inject tools via this model.
    """

    model_config = ConfigDict(extra="forbid")

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    province: str = ""
    city: str = ""
    district: str = ""
    adcode: str = ""
    address: str = ""
    source: str = ""


class ChatV2Request(BaseModel):
    """Strict Chat v2 request.

    ``extra="forbid"`` is the contract: any field not listed here is
    rejected with HTTP 400. This is what prevents clients from injecting
    ``model``, ``api_key``, ``tools``, ``mcp_servers``, or any other
    configuration that the server owns.
    """

    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, description="User message text")
    request_id: str = Field(..., min_length=1, description="Client idempotency key")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation id for multi-turn runs",
    )
    profile_id: str = Field(
        default="chat",
        description="Capability profile id. Server-defined; clients cannot add profiles.",
    )
    current_location: Optional[CurrentLocationV2] = None


# ---- Response helpers -----------------------------------------------------


def _sse_frame(envelope: EventEnvelope) -> str:
    """Render an envelope as an SSE ``data:`` frame."""
    return f"data: {envelope.to_json_line()}\n\n"


# ---- Stub response generation --------------------------------------------

# The artifact profile owns a run-scoped sandbox. The sandbox is created
# per run (spec AC1), recorded as a lease in ``agent_control.sqlite``,
# and cleaned up on terminal (spec AC6). Fail-closed (spec AC5): any
# ``ComputerError`` becomes a terminal ``error`` event; the run NEVER
# falls back to executing on the host. Since Ticket 30 the plan content
# is produced by the real UniHarness kernel loop through
# ProductSandboxBridge; the stub generator has been removed.

_ARTIFACT_FILENAME = "hiking-plan.pdf"


def _build_artifact_computer(sandbox_type: str | None = None) -> Computer:
    """Resolve the sandbox Computer for the artifact profile.

    Reads ``settings.agent_v2_sandbox_type`` so the same code path runs
    in production (``remote_e2b``) and development (``local_native``).
    Tests inject failures by monkeypatching this function or
    ``create_computer`` — production code never sets ``failure_mode``.

    Ticket 11: ``sandbox_type`` is optional. The initial flow leaves it
    ``None`` so the factory reads ``settings.agent_v2_sandbox_type``.
    The resume flow passes ``lease.sandbox_type`` so the SAME sandbox
    type is reused (spec AC4: "继续原 ... 原 sandbox"). Tests that
    monkeypatch this function should accept ``*args, **kwargs`` so the
    same injection works for both flows.
    """
    return create_computer(sandbox_type or settings.agent_v2_sandbox_type)


def _cleanup_sandbox(
    computer: Computer,
    workspace_path: str,
    runs_repo: RunRepository | None,
    run_id: str | None,
) -> None:
    """Best-effort sandbox cleanup (spec AC6).

    Always invoked from a ``finally`` block. Cleanup failure is recorded
    on the lease as ``failed`` with the error reason; the run's terminal
    state is NOT changed — the operator must investigate stale sandboxes
    via the lease record.
    """
    failure_reason: str | None = None
    resource_usage: dict[str, Any] | None = None
    try:
        usage = computer.cleanup(workspace_path)
        resource_usage = usage.to_dict()
    except SandboxCleanupError as exc:
        failure_reason = f"cleanup failed: {exc}"
        # Preserve partial usage gathered before the cleanup failure so
        # the lease still records resource metrics (spec AC6: record
        # success, failure, AND resource usage).
        if exc.partial_usage is not None:
            resource_usage = exc.partial_usage.to_dict()
        logger.warning(
            "chat_v2: sandbox cleanup failed for run_id=%s workspace=%s: %s",
            run_id, workspace_path, exc,
        )
    except Exception as exc:
        # Unexpected cleanup error — record as failed but do not raise.
        failure_reason = f"cleanup unexpected error: {exc}"
        logger.exception(
            "chat_v2: sandbox cleanup unexpected error for run_id=%s",
            run_id,
        )

    if runs_repo is not None and run_id is not None:
        try:
            runs_repo.destroy_sandbox_lease(
                run_id=run_id,
                resource_usage=resource_usage,
                failure_reason=failure_reason,
            )
        except Exception:
            logger.exception(
                "chat_v2: destroy_sandbox_lease failed for run_id=%s; "
                "lease record may be stale",
                run_id,
            )


# Ticket 41: ``_persist_failed_artifact`` and ``_export_artifact_to_host``
# moved to :mod:`agent_v2.artifacts` as ``persist_failed_artifact`` /
# ``export_artifact_to_host``; the orchestrator drives them directly.


# ---- cowork profile (Ticket 13) ------------------------------------------
#
# The cowork profile extends artifact with E2B-scoped Bash. Since
# Ticket 38 the cowork flow (planner turn, approval pause) and since
# Ticket 40 the sandbox_bash resume steps live in
# :mod:`agent_v2.run_orchestration`; this adapter only renders SSE.


def _build_run_orchestrator(runtime: Any) -> RunOrchestrator:
    """Resolve the v2 runtime into typed orchestrator dependencies.

    This is the only place the adapter touches application state; the
    orchestrator itself receives constructor-injected dependencies and
    never duck-types through ``app.state`` (Ticket 37).
    """
    return RunOrchestrator(
        settings=settings,
        runs_repo=getattr(runtime, "runs", None) if runtime else None,
        capacity=getattr(runtime, "capacity", None) if runtime else None,
        telemetry=getattr(runtime, "telemetry", None) if runtime else None,
        memory=getattr(runtime, "memory", None) if runtime else None,
        kernel_checkpoints=getattr(runtime, "kernel_checkpoints", None) if runtime else None,
        # Ticket 41: the artifact export is no longer injected — the
        # orchestrator's production default (agent_v2.artifacts
        # .export_artifact_to_host) drives it. The computer/cleanup hooks
        # remain as test seams for the ASGI contract tests.
        computer_factory=lambda *a, **k: _build_artifact_computer(*a, **k),
        sandbox_cleanup=lambda *a, **k: _cleanup_sandbox(*a, **k),
        gate_factory=lambda: PermissionGate(),
    )


# ---- Endpoint -------------------------------------------------------------


@chat_v2_router.post("/sse")
async def chat_v2_sse(req: ChatV2Request, request: Request):
    """Stream a Chat v2 run as SSE.

    Returns 200 with ``text/event-stream`` once the request is accepted.
    Request schema violations (extra fields, missing required fields) are
    rejected by Pydantic with 422 before the stream starts so clients see
    the error synchronously. Business-level rejections (unknown profile,
    v2 disabled, model unavailable) are reported as ``error`` terminal
    events inside the SSE stream itself, not as HTTP errors — this keeps
    the client's terminal-event handling in one place.

    The stream always ends with exactly one terminal event (``error``,
    ``done``, or ``cancelled``). Transport closure is not a terminal —
    clients that observe EOF without a terminal must treat the run as
    ``outcome_unknown`` and reconcile via the run snapshot API (Ticket 06).

    Ticket 06: when v2 persistence is active, the ``request_id`` is the
    idempotency key. A duplicate POST returns the stored event stream
    rather than starting a new run (spec AC2). Clients reconnecting after
    a network drop should use ``GET /events/{run_id}?from_seq=N`` for
    fine-grained replay.

    Ticket 37: run orchestration (stream lifecycle, terminal guarantee,
    persistence ordering, prompt assembly, kernel turn) lives in
    :class:`agent_v2.run_orchestration.RunOrchestrator`; this adapter
    keeps only request parsing, lease→503 mapping, SSE frame rendering,
    and the subject cookie.
    """
    runtime = getattr(request.app.state, "agent_v2_runtime", None)
    orchestrator = _build_run_orchestrator(runtime)
    try:
        leases = await orchestrator.acquire_leases(
            request_id=req.request_id, profile_id=req.profile_id
        )
    except CapacityExceeded as exc:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": f"{exc.resource}_capacity_exceeded",
                    "retryable": True,
                }
            },
        )

    subject = None
    if getattr(runtime, "memory", None) is not None:
        _, subject = resolve_request_subject(request)

    flow_req = ChatFlowRequest(
        message=req.message,
        request_id=req.request_id,
        profile_id=req.profile_id,
        conversation_id=req.conversation_id,
        current_location=req.current_location,
    )

    async def guarded_stream() -> AsyncGenerator[str, None]:
        # Langfuse session 聚合：conversation_id 优先，退回 request_id。
        # 在生成器任务内设置，确保内核 turn 的执行上下文可见。
        session_token = set_langfuse_session(req.conversation_id or req.request_id)
        try:
            async for item in orchestrator.stream_chat(
                flow_req, subject_id=getattr(request.state, "subject_id", None)
            ):
                yield item if isinstance(item, str) else _sse_frame(item)
        finally:
            reset_langfuse_session(session_token)
            leases.release()

    response = StreamingResponse(
        guarded_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable proxy buffering (Gateway)
        },
    )
    if subject is not None:
        set_subject_cookie(response, request, subject)
    return response


def _get_runs_repo(request: Request) -> RunRepository:
    """Resolve the v2 RunRepository or raise 503.

    Centralized so the snapshot/cancel/events endpoints share one guard.
    The 503 is intentional: these endpoints exist only when v2 persistence
    is assembled. A client hitting them when v2 is disabled has stale
    routing information and should back off.
    """
    runtime = getattr(request.app.state, "agent_v2_runtime", None)
    runs_repo = getattr(runtime, "runs", None) if runtime else None
    if runs_repo is None:
        raise HTTPException(
            status_code=503,
            detail="Chat v2 persistence is not available (AGENT_V2_ENABLED=false "
                   "or runtime not assembled).",
        )
    return runs_repo


def _require_conversation_owner(request: Request, conversation_id: str) -> None:
    memory, subject = resolve_request_subject(request)
    if not memory.owns_conversation(subject.subject_id, conversation_id):
        raise HTTPException(status_code=404, detail="Run not found")


@chat_v2_router.get("/snapshot/{run_id}")
async def chat_v2_snapshot(run_id: str, request: Request):
    """Return the run snapshot — spec AC3.

    The snapshot includes conversation messages, run status, sequence
    cursor, and pending tool calls. Raw framework events are NOT exposed
    — the snapshot is product-facing.

    Ticket 40: the read is driven by the orchestration module
    (:meth:`RunOrchestrator.load_snapshot`); the adapter maps a missing
    run to 404.
    """
    _get_runs_repo(request)  # 503 guard when v2 persistence is not assembled
    orchestrator = _build_run_orchestrator(
        getattr(request.app.state, "agent_v2_runtime", None)
    )
    snapshot = orchestrator.load_snapshot(run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Run not found")
    _require_conversation_owner(request, snapshot.conversation_id)
    return {
        "run_id": snapshot.run_id,
        "request_id": snapshot.request_id,
        "conversation_id": snapshot.conversation_id,
        "profile_id": snapshot.profile_id,
        "status": snapshot.status,
        "terminal_type": snapshot.terminal_type,
        "created_at": snapshot.created_at,
        "updated_at": snapshot.updated_at,
        "terminal_at": snapshot.terminal_at,
        "sequence_cursor": snapshot.sequence_cursor,
        "messages": snapshot.messages,
        "pending_actions": snapshot.pending_actions,
    }


class CancelRequest(BaseModel):
    """Body for ``POST /cancel/{run_id}``.

    ``idempotency_key`` is required so duplicate cancel requests (e.g.
    user double-clicks) return the original outcome rather than
    re-attempting the state transition (spec AC5).
    """

    model_config = ConfigDict(extra="forbid")

    idempotency_key: str = Field(..., min_length=1)


@chat_v2_router.post("/cancel/{run_id}")
async def chat_v2_cancel(run_id: str, body: CancelRequest, request: Request):
    """Idempotent run cancel — spec AC5.

    Returns the cancel outcome:

    * ``accepted=true, reason="cancelled"`` — first cancel on a live run
    * ``accepted=false, reason="already_terminal:<type>"`` — run already
      terminated (done/error/cancelled); repeat cancels with a new
      idempotency_key on an already-cancelled run also return this
    * ``accepted=false, reason="unknown_run"`` — run_id not found

    The ``run_id`` path parameter also accepts a ``request_id`` (spec
    AC4): when the user cancels before the first v2 event has delivered
    ``run_id`` to the client, the frontend only knows ``request_id``.
    The endpoint resolves ``request_id`` → ``run_id`` before cancelling.
    """
    runs_repo = _get_runs_repo(request)
    # AC4: resolve request_id → run_id for early cancel. The path
    # parameter is treated as an identifier that may be either.
    resolved_run_id = runs_repo.resolve_run_id(run_id)
    if resolved_run_id is None:
        return JSONResponse(
            status_code=409,
            content={
                "accepted": False,
                "reason": "unknown_run",
                "run_id": run_id,
            },
        )
    run = runs_repo.get_run(resolved_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    _require_conversation_owner(request, run.conversation_id)
    outcome = runs_repo.request_cancel(
        run_id=resolved_run_id,
        idempotency_key=body.idempotency_key,
    )
    if outcome.accepted:
        sandbox = runs_repo.get_sandbox_lease(resolved_run_id)
        if sandbox is not None and sandbox.status == "active":
            try:
                runs_repo.destroy_sandbox_lease(
                    run_id=resolved_run_id,
                    failure_reason="run cancelled",
                )
            except Exception:
                logger.exception(
                    "chat_v2: sandbox cleanup failed for cancelled run_id=%s",
                    resolved_run_id,
                )
            else:
                runtime = getattr(request.app.state, "agent_v2_runtime", None)
                capacity = getattr(runtime, "capacity", None)
                if capacity is not None:
                    capacity.release_sandbox(run.request_id)
    status_code = 200 if outcome.accepted else 409
    return JSONResponse(
        status_code=status_code,
        content={
            "accepted": outcome.accepted,
            "reason": outcome.reason,
            "run_id": outcome.run_id,
        },
    )


@chat_v2_router.get("/events/{run_id}")
async def chat_v2_events(
    run_id: str,
    request: Request,
    from_seq: int = Query(default=0, ge=0, description="Replay events with sequence > from_seq"),
):
    """Replay stored events for reconnect — spec AC4.

    Returns the events with sequence strictly greater than ``from_seq``,
    in ascending order. ``from_seq=0`` returns the full stream. Clients
    use this to reconnect after a network drop without re-executing tools.

    Response shape: ``{"run_id": ..., "from_seq": N, "events": [...]}``
    where each event matches the v2 wire envelope.
    """
    runs_repo = _get_runs_repo(request)
    # Verify the run exists so we return 404 instead of an empty event list
    # for an unknown run_id.
    run = runs_repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    _require_conversation_owner(request, run.conversation_id)
    # Ticket 40: the replay read is driven by the orchestration module.
    orchestrator = _build_run_orchestrator(
        getattr(request.app.state, "agent_v2_runtime", None)
    )
    events = orchestrator.load_events(run_id, from_seq=from_seq)
    return {
        "run_id": run_id,
        "from_seq": from_seq,
        "events": events,
    }


@chat_v2_router.get("/health")
async def chat_v2_health():
    """Lightweight health probe for the v2 chat endpoint.

    Does NOT assemble the runtime or check the LLM key. Use ``/health``
    for full status. This endpoint exists so the Gateway can liveness-check
    the router without triggering a cold start.
    """
    return {
        "status": "ok" if settings.agent_v2_enabled else "disabled",
        "module": "chat_v2",
        "service": "ai-service",
        "uniharness_pin": settings.agent_v2_uniharness_pin,
    }


# ---- Artifact download endpoint (Ticket 10) ------------------------------
#
# Spec AC6: artifact download verifies identity + ownership; not-found,
# expired, incomplete (failed), and cross-subject access all return a
# stable error. Spec AC3: the response never exposes sandbox or host
# paths — the client only sees the opaque ``artifact_id`` (in the URL)
# and the ``public_filename`` (in Content-Disposition).
#
# Auth model (stub stage): the ``artifact_id`` is an unguessable uuid4.
# This is the same capability-token model the other v2 endpoints use
# for ``run_id``. Real auth lands with the UniHarness identity ticket;
# the ``conversation_id`` query param is an optional ownership check
# that the frontend can supply today (the artifact row stores the
# conversation_id; a mismatch returns 403).


@artifacts_v2_router.get("/{artifact_id}")
async def download_artifact(
    artifact_id: str,
    request: Request,
    conversation_id: Optional[str] = Query(
        default=None,
        description="Optional ownership check; the artifact's conversation_id must match",
    ),
):
    """Download a persisted artifact by its opaque id — spec AC3, AC6.

    Returns the file with the original ``Content-Type`` and a
    ``Content-Disposition: attachment; filename="<public_filename>"``
    header. The response body is the exact bytes the host verified
    against the manifest — no transformation, no re-encoding.

    Status codes:

    * 200 — file returned
    * 404 — ``artifact_id`` is not a valid uuid or no row exists
    * 403 — ``conversation_id`` query param was supplied and does not
      match the artifact's conversation (cross-subject access)
    * 410 — artifact row exists but ``status`` is ``expired`` or
      ``failed`` (no downloadable file), OR ``expires_at`` has elapsed
      (runtime fallback for the missing cron sweep), OR the file is
      missing on disk
    * 503 — v2 persistence is not assembled (AGENT_V2_ENABLED=false)

    All 410 responses share the same stable detail message so the
    frontend can render a single "artifact unavailable" state without
    parsing variants. The exact failure reason is logged server-side.
    """
    runs_repo = _get_runs_repo(request)

    # Validate artifact_id is a uuid4. This is a path-traversal guard:
    # an attacker cannot inject ``..`` or separators because uuid.UUID
    # rejects anything outside the hex+dash grammar.
    try:
        uuid.UUID(artifact_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=404, detail="Artifact not found")

    artifact = runs_repo.get_artifact(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    _require_conversation_owner(request, artifact.conversation_id)

    # Optional ownership check. The frontend knows the conversation_id
    # from the chat run; supplying it lets us reject cross-subject access
    # even before real auth lands. A missing conversation_id is allowed
    # — the opaque artifact_id remains the primary capability.
    if (
        conversation_id is not None
        and conversation_id != artifact.conversation_id
    ):
        raise HTTPException(status_code=403, detail="Artifact not accessible")

    if artifact.status == "expired":
        logger.info("chat_v2: artifact %s expired (status flag)", artifact_id)
        raise HTTPException(status_code=410, detail="Artifact unavailable")
    if artifact.status == "failed":
        logger.info(
            "chat_v2: artifact %s has failed status (reason=%s)",
            artifact_id, artifact.failure_reason or "<none>",
        )
        raise HTTPException(status_code=410, detail="Artifact unavailable")

    # Runtime fallback for the missing cron sweep: even if the row's
    # ``status`` is still ``available``, refuse download once
    # ``expires_at`` has elapsed. This closes the window between TTL
    # expiry and the next sweep that mutates ``status`` to ``expired``.
    if artifact.expires_at:
        try:
            expires_dt = datetime.fromisoformat(artifact.expires_at)
            if expires_dt.tzinfo is None:
                # ``fromisoformat`` on naive strings returns naive
                # datetimes; treat them as UTC for the comparison.
                expires_dt = expires_dt.replace(tzinfo=timezone.utc)
            now_dt = datetime.now(timezone.utc)
            if expires_dt < now_dt:
                logger.info(
                    "chat_v2: artifact %s past expires_at (%s < %s); "
                    "denying download (runtime fallback)",
                    artifact_id, artifact.expires_at, now_dt.isoformat(),
                )
                raise HTTPException(
                    status_code=410, detail="Artifact unavailable"
                )
        except HTTPException:
            raise
        except (ValueError, TypeError) as exc:
            logger.warning(
                "chat_v2: artifact %s has unparseable expires_at=%r: %s",
                artifact_id, artifact.expires_at, exc,
            )

    # Defense-in-depth: verify the file still exists on disk. A row
    # without a file (e.g. operator swept the storage dir) returns 410
    # so the frontend treats it as "no longer available" rather than a
    # server error.
    storage_path = artifact.storage_path
    if not storage_path or not Path(storage_path).is_file():
        logger.warning(
            "chat_v2: artifact row %s has missing file at %s",
            artifact_id, storage_path or "<empty>",
        )
        raise HTTPException(
            status_code=410,
            detail="Artifact unavailable",
        )

    # FileResponse streams the file with the original Content-Type and
    # sets Content-Disposition to the public_filename. The storage_path
    # never appears in any response header — only the basename does.
    return FileResponse(
        path=storage_path,
        media_type=artifact.mime_type,
        filename=artifact.public_filename,
    )


# ---- Approval decision endpoint (Ticket 11) ------------------------------
#
# Spec AC4 + AC5: ``POST /api/v2/approvals/{approval_id}/decision`` is
# the single entry point for the user's approve/reject verdict. The
# endpoint:
#
# * Validates the approval exists, is not expired, and is not already
#   resolved with a different decision_request_id (spec AC5 idempotency).
# * Validates the run is ``paused`` (not terminal) and the sandbox lease
#   is still ``active`` (spec AC7: fail-closed when lease is lost).
# * Persists the decision via ``resolve_approval`` — the DB row's
#   ``decision_request_id`` UNIQUE constraint enforces "first decision
#   wins" so the tool executes at most once.
# * Transitions the run ``paused`` → ``running`` so the resume endpoint
#   can pick it up. The response carries the resume URL.
#
# The endpoint is intentionally synchronous: it does NOT emit SSE events.
# The client opens a separate SSE connection to ``/chat/resume/{run_id}``
# to receive the continuation stream. This separates the state mutation
# (decision) from the stream (resume), so a network drop during resume
# does not re-mutate the decision — the client reconnects to resume and
# gets the already-persisted events + any remaining events.


class ApprovalDecisionRequest(BaseModel):
    """Body for ``POST /approvals/{approval_id}/decision``.

    ``decision`` is the user's verdict: ``approve`` lets the tool execute
    (run resumes with the tool call); ``reject`` skips the tool and ends
    the run with ``done`` (no side effect to undo, per spec AC4).

    ``decision_request_id`` is the idempotency key (spec AC5): duplicate
    decisions with the same key return the original outcome rather than
    re-attempting the transition. A second decision with a NEW key on an
    already-resolved approval returns 409.
    """

    model_config = ConfigDict(extra="forbid")

    decision: str = Field(..., description="User verdict: approve or reject")
    decision_request_id: str = Field(
        ..., min_length=1, description="Idempotency key for this decision"
    )


@approvals_v2_router.post("/{approval_id}/decision")
async def approval_decision(
    approval_id: str,
    body: ApprovalDecisionRequest,
    request: Request,
):
    """Record the user's approve/reject decision — spec AC4, AC5, AC7.

    Returns 200 with the resolved approval + resume URL on success. The
    client opens an SSE connection to ``resume_url`` to receive the
    continuation stream (``approval.resolved`` → tool execution → done).

    Status codes:

    * 200 — decision recorded (or idempotent duplicate of the same
      decision with the same ``decision_request_id``)
    * 404 — ``approval_id`` not found
    * 409 — approval already resolved with a different
      ``decision_request_id``; OR approval expired; OR run is terminal
      (done/error/cancelled); OR run is not paused (decision already
      recorded); OR sandbox lease is missing/destroyed (spec AC7)
    * 422 — Pydantic rejected the body (unknown ``decision`` value,
      missing fields, extra fields)
    * 503 — v2 persistence is not assembled

    Since Ticket 38, ALL approval state transitions (idempotency,
    expiry, run guards, sandbox-lease fail-closed, paused → running)
    live in the run orchestration module
    (:meth:`RunOrchestrator.resolve_approval_decision`); this endpoint
    only performs the HTTP-level pre-checks and maps the outcome to a
    status code.
    """
    if body.decision not in ("approve", "reject"):
        raise HTTPException(
            status_code=422,
            detail="decision must be 'approve' or 'reject'",
        )

    runs_repo = _get_runs_repo(request)

    # The conversation-owner check needs the approval row before any
    # state transition happens — HTTP-level authorization stays here.
    approval = runs_repo.get_approval(approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    _require_conversation_owner(request, approval.conversation_id)

    runtime = getattr(request.app.state, "agent_v2_runtime", None)
    orchestrator = _build_run_orchestrator(runtime)
    outcome = orchestrator.resolve_approval_decision(
        approval_id,
        decision=body.decision,
        decision_request_id=body.decision_request_id,
    )

    if outcome.status == "approval_not_found":
        raise HTTPException(status_code=404, detail="Approval not found")
    if outcome.status in ("recorded", "idempotent"):
        payload = dict(outcome.payload)
        payload["resume_url"] = f"/api/v2/chat/resume/{payload['run_id']}"
        return JSONResponse(status_code=200, content=payload)
    return JSONResponse(status_code=409, content=outcome.payload)


# ---- Resume SSE endpoint (Ticket 11) -------------------------------------
#
# Spec AC4 + AC7: ``POST /api/v2/chat/resume/{run_id}`` is the SSE
# endpoint that continues a paused run after the user decision. It:
#
# * Replays stored events with sequence > ``from_seq`` so the client can
#   restore state after a network drop (spec AC4).
# * Validates the run is ``running`` (post-decision) and has a resolved
#   approval — if the run is still ``paused`` the client must POST the
#   decision first.
# * Reuses the SAME sandbox lease recorded when the run started (spec
#   AC4: "继续原 Chat v2 run、原 checkpoint 和原 sandbox"). If the lease
#   is missing or destroyed, the resume fails closed with an ``error``
#   terminal (spec AC7: "sandbox lease 丢失时 fail closed，不在新 sandbox
#   继续").
# * Emits ``approval.resolved`` → (if approved) tool execution →
#   ``artifact.available`` → ``text`` → ``done``; or (if rejected)
#   ``done``. The sandbox is cleaned up in the ``finally`` block.
# * Idempotent: if the run is already terminal, only the replay runs —
#   no new events are emitted. A second resume after completion returns
#   the stored stream without re-executing tools (spec AC5).


@chat_v2_router.post("/resume/{run_id}")
async def chat_v2_resume(
    run_id: str,
    request: Request,
    from_seq: int = Query(
        default=0, ge=0, description="Replay events with sequence > from_seq before continuing"
    ),
):
    """Resume a paused run as SSE — spec AC4, AC7.

    Returns 200 with ``text/event-stream``. The stream replays stored
    events with sequence > ``from_seq`` (so a client that lost state can
    set ``from_seq=0`` to get the full history), then emits the
    continuation: ``approval.resolved`` → tool execution (if approved) →
    terminal.

    Ticket 40: the resume flow itself (replay, continuation, sandbox
    re-execution, effect ledger) is driven end-to-end by
    :meth:`agent_v2.run_orchestration.RunOrchestrator.stream_resume`;
    this adapter keeps only HTTP checks, lease→503 mapping, and SSE
    frame rendering.

    Status codes:

    * 200 — SSE stream started
    * 404 — ``run_id`` not found
    * 409 — run is ``paused`` (POST decision first), or run is terminal
      and no continuation is possible
    * 503 — v2 persistence is not assembled
    """
    runs_repo = _get_runs_repo(request)
    run = runs_repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    _require_conversation_owner(request, run.conversation_id)

    if run.status == "paused":
        return JSONResponse(
            status_code=409,
            content={
                "run_id": run_id,
                "reason": "run_paused",
                "message": "运行处于 paused 状态，请先 POST /api/v2/approvals/{approval_id}/decision。",
            },
        )

    runtime = getattr(request.app.state, "agent_v2_runtime", None)
    orchestrator = _build_run_orchestrator(runtime)
    try:
        leases = await orchestrator.acquire_resume_leases(
            run_id=run_id, request_id=run.request_id
        )
    except CapacityExceeded as exc:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": f"{exc.resource}_capacity_exceeded",
                    "retryable": True,
                }
            },
        )

    async def guarded_resume_stream() -> AsyncGenerator[str, None]:
        try:
            async for item in orchestrator.stream_resume(
                run_id=run_id, from_seq=from_seq
            ):
                yield item if isinstance(item, str) else _sse_frame(item)
        finally:
            leases.release()

    return StreamingResponse(
        guarded_resume_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
