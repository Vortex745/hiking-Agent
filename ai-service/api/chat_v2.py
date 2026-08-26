"""Chat v2 SSE endpoint — product-owned HTTP/SSE contract.

This router exposes ``POST /api/v2/chat/sse``. It is the only public
surface for Chat v2 during the灰度 period; the v1 ``/api/v1/chat/sse``
endpoint remains unchanged and is the rollback target.

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

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from agent_v2.approvals import (
    canonical_args,
    compute_approval_id,
    compute_expiry,
    is_expired,
)
from agent_v2.artifacts import (
    ArtifactEvidencePersistenceError,
    ArtifactExportError,
    ArtifactManifest,
    host_storage_path,
    is_symlink_escape,
    persist_artifact_bytes,
    validate_manifest,
)
from agent_v2.computer import (
    BashResult,
    Computer,
    ComputerError,
    SandboxBashOutputLimitExceeded,
    SandboxCleanupError,
    SandboxCreationError,
    SandboxQuotaError,
    SandboxTimeoutError,
    create_computer,
)
from agent_v2.effects import (
    EffectStatus,
    compute_effect_key,
    compute_external_idempotency_key,
    deserialize_result,
    serialize_result,
)
from agent_v2.events import (
    EventEnvelope,
    EventStream,
    TerminalEventAlreadyEmitted,
    TERMINAL_EVENT_TYPES,
)
from agent_v2.mcp_registry import (
    McpServerConfig,
    UnknownMcpServerError,
    check_mcp_available,
    list_profile_mcp_tools,
    resolve_mcp_server,
)
from agent_v2.memory_tools import (
    asks_for_preferences,
    extract_ordinary_preference,
    extract_sensitive_memory,
)
from agent_v2.observability import CapacityExceeded
from agent_v2.permissions import (
    PermissionDecision,
    PermissionGate,
)
from agent_v2.profiles import (
    CapabilityProfile,
    ProfileId,
    UnknownProfileError,
    resolve_profile,
)
from agent_v2.runs import (
    ApprovalRecord,
    CancelOutcome,
    RunAlreadyTerminal,
    RunRepository,
    RunSnapshot,
)
from agent_v2.startup import V2Runtime, is_cowork_authorized
from config import settings
from api.memory_v2 import resolve_request_subject, set_subject_cookie

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


# Spec AC3 + decision: artifacts persist for 7 days after creation. The
# TTL is intentionally short — artifacts are downloadable summaries, not
# permanent user files. Operators may sweep expired rows + files in a
# separate cron; the endpoint returns 410 once ``expires_at`` elapses.
ARTIFACT_TTL_DAYS = 7


def _artifacts_dir() -> str:
    """Resolve the host-side artifact storage directory.

    Lives under ``settings.agent_v2_state_dir/artifacts`` so artifacts
    share the persistent volume with the SQLite databases. The directory
    is created on demand by :func:`persist_artifact_bytes`.
    """
    return str(Path(settings.agent_v2_state_dir) / "artifacts")


def _now_iso() -> str:
    """UTC ISO 8601 timestamp with timezone suffix."""
    return datetime.now(timezone.utc).isoformat()


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
    from datetime import timedelta

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


# ---- Paused-run janitor -----------------------------------------------------
#
# 审批暂停的 run 在流结束时保留 CapacityGuard 沙箱槽位等待用户决策
# （guarded_stream 的 keep_reserved 分支）。用户弃用后没有任何正常路径
# 释放该槽位（只有 cancel 和 resume 会释放），``max_sandboxes=10`` 会被
# 逐渐耗尽直到持续 503。janitor 周期回收暂停超过 TTL 的 run。

_PAUSED_RUN_TTL_SECONDS = 24 * 3600
_PAUSED_RUN_SWEEP_INTERVAL_SECONDS = 600


def _sweep_expired_paused_runs(runtime: Any) -> int:
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
                    "chat_v2: paused-run sweep failed to destroy lease for run_id=%s",
                    run.run_id,
                )
                continue
        try:
            runs_repo.mark_run_error(run.run_id)
        except Exception:
            logger.exception(
                "chat_v2: paused-run sweep failed to mark run_id=%s terminal",
                run.run_id,
            )
            continue
        if capacity is not None:
            capacity.release_sandbox(run.request_id)
        reclaimed += 1
        logger.warning(
            "chat_v2: reclaimed expired paused run run_id=%s (sandbox slot released)",
            run.run_id,
        )
    return reclaimed


def start_paused_run_sweeper(runtime: Any) -> "asyncio.Task[None]":
    """启动暂停超时回收周期任务（lifespan 装配后调用，关停时 cancel）。"""

    async def _loop() -> None:
        while True:
            await asyncio.sleep(_PAUSED_RUN_SWEEP_INTERVAL_SECONDS)
            try:
                _sweep_expired_paused_runs(runtime)
            except Exception:
                logger.exception("chat_v2: paused-run sweep iteration failed")

    return asyncio.create_task(_loop(), name="paused-run-sweeper")


def _expires_at_iso(created_at: str) -> str:
    """Compute the expiry timestamp ``ARTIFACT_TTL_DAYS`` after ``created_at``.

    Parses ``created_at`` (ISO 8601 with timezone) so the expiry is
    derived from the creation time, not from ``now`` — this keeps the
    TTL stable even if the clock drifts between insert and expiry check.
    """
    created = datetime.fromisoformat(created_at)
    expires = created + _timedelta(days=ARTIFACT_TTL_DAYS)
    return expires.isoformat()


def _timedelta(*, days: int) -> "timedelta":
    # Local import keeps the module header lean.
    from datetime import timedelta

    return timedelta(days=days)


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

# The stub exists so contract tests can verify the envelope without an LLM
# API key. Real LLM integration replaces this when the UniHarness factory
# swaps from the stub to the pinned fork.
_GREETING_MARKERS = ("你好", "您好", "hello", "hi", "嗨", "早上好", "晚上好", "下午好")
_ROUTE_REQUEST_MARKERS = ("推荐路线", "推荐徒步路线", "推荐一条", "规划路线", "去哪徒步")


def _stub_classify(message: str) -> str:
    """Classify the message into a stub scenario.

    Returns one of: ``greeting``, ``slot_followup``, ``normal``.
    The classifier is intentionally simple — it exists to exercise the
    three event-mapping paths the contract requires, not to be a real
    intent router.
    """
    text = message.strip().lower()
    if any(marker in text for marker in _GREETING_MARKERS):
        return "greeting"
    if any(marker in text for marker in _ROUTE_REQUEST_MARKERS):
        # Route request without a destination → slot-followup.
        if not any(
            dest_marker in text
            for dest_marker in ("北京", "上海", "成都", "云南", "四川", "黄山", "泰山")
        ):
            return "slot_followup"
    return "normal"


def _stub_response(message: str, classification: str, profile: CapabilityProfile) -> str:
    """Generate a deterministic stub response for the given classification.

    Real LLM integration replaces this. The stub responses are chosen to
    exercise the event-mapping contract, not to be helpful answers. Tool
    names are NOT exposed to the user — the stub describes capabilities
    in plain Chinese so internal tool identifiers stay server-side.
    """
    if classification == "greeting":
        return "你好！我是徒步助手。可以问我装备、风险或徒步知识相关问题。"
    if classification == "slot_followup":
        return "请告诉我你想去的目的地，我才能推荐合适的路线。"
    # Normal: acknowledge the question with a user-safe description.
    # S8: do NOT mention internal framework names (UniHarness/LangChain) —
    # the user only sees that the assistant received the question; the
    # real LLM integration is server-side and invisible.
    return "已收到你的问题。这是 v2 stub 响应，正式 LLM 集成后将由服务端接管生成。"


# ---- route_plan intake + MCP stub (Ticket 05) ----------------------------
#
# The route_plan intake is a deterministic classifier that decides which
# scenario the run is in. It is NOT a model call — real intent routing
# arrives with the UniHarness LLM integration. The classifier inspects
# the message and the authorized location to choose one of:
#
#   ``route_with_evidence``   — route request + destination + MCP available
#   ``route_no_evidence``     — route request + destination + MCP unavailable
#   ``location_weather``      — weather question + authorized location + MCP
#   ``weather_no_evidence``   — weather question + authorized location, no MCP
#   ``slot_followup_destination`` — route request, destination missing
#   ``slot_followup_location``    — weather question, location missing
#   ``mcp_timeout``           — test marker in message triggers timeout sim
#   ``mcp_oversize``          — test marker in message triggers oversize sim
#   ``normal``                — none of the above
#
# Intake non-escalation: the chat profile never calls this classifier.
# Even if the user sends ``profile_id="chat"`` with a route-like message,
# the chat profile handles it with its own ``_stub_classify`` flow and
# never reaches MCP. This is the contract: profile_id is a request, not
# authorization; the server never upgrades chat → route_plan.

_ROUTE_PLAN_DESTINATION_MARKERS = (
    "北京", "上海", "成都", "云南", "四川", "黄山", "泰山",
    "杭州", "西安", "拉萨", "新疆", "甘肃", "青海",
)
_WEATHER_MARKERS = ("天气", "气温", "下雨", "weather", "温度")
_ROUTE_REQUEST_MARKERS_rp = ("推荐路线", "推荐徒步路线", "推荐一条", "规划路线", "去哪徒步")
_MCP_TIMEOUT_MARKER = "[mcp:timeout]"
_MCP_OVERSIZE_MARKER = "[mcp:oversize]"


def _stub_classify_route_plan(
    message: str,
    has_location: bool,
    mcp_available: bool,
) -> str:
    """Classify a route_plan message into a scenario.

    The classifier is intentionally deterministic so contract tests can
    assert exact event sequences. Real intent routing is the LLM's job
    in later tickets; this stub only proves the event-mapping contract.
    """
    text = message.strip().lower()

    # Test markers take precedence so tests can exercise timeout/oversize
    # without depending on MCP availability or message content.
    if _MCP_TIMEOUT_MARKER in text:
        return "mcp_timeout"
    if _MCP_OVERSIZE_MARKER in text:
        return "mcp_oversize"

    is_weather = any(marker in text for marker in _WEATHER_MARKERS)
    # Route intent matches exact markers OR the looser "推荐...路线" pattern
    # so messages like "推荐北京路线" (destination between 推荐 and 路线) count.
    is_route = (
        any(marker in text for marker in _ROUTE_REQUEST_MARKERS_rp)
        or ("推荐" in text and "路线" in text)
    )
    has_destination = any(
        marker in text for marker in _ROUTE_PLAN_DESTINATION_MARKERS
    )

    if is_weather:
        if not has_location:
            return "slot_followup_location"
        return "location_weather" if mcp_available else "weather_no_evidence"

    if is_route:
        if not has_destination:
            return "slot_followup_destination"
        return "route_with_evidence" if mcp_available else "route_no_evidence"

    return "normal"


def _simulate_mcp_call(
    server_id: str,
    tool_name: str,
    args: dict,
    *,
    mode: str = "ok",
) -> dict:
    """Return a deterministic stub result for an MCP tool call.

    ``mode`` controls the result shape so tests can exercise the four
    contract paths (ok / timeout / oversize / unavailable) without
    spawning a real MCP subprocess.

    The returned dict carries only user-safe content — no credentials,
    no internal server addresses, no env values. The host connector
    expands credentials in-process and never copies them into the
    result; this stub honors the same boundary.
    """
    if mode == "timeout":
        return {
            "status": "timeout",
            "content": f"MCP 调用超时，未获得 {tool_name} 结果。",
        }
    if mode == "oversize":
        return {
            "status": "oversize",
            "content": f"MCP 输出超过上限被截断，{tool_name} 结果不完整。",
        }
    if mode == "unavailable":
        return {
            "status": "unavailable",
            "content": f"MCP 服务 {server_id} 不可用，{tool_name} 未执行。",
        }
    # ok
    if tool_name == "get_current_weather":
        city = args.get("city", "当前位置")
        return {
            "status": "ok",
            "content": f"{city} 当前气温 15°C，晴，风力 2 级。",
        }
    if tool_name == "plan_route":
        destination = args.get("destination", "目的地")
        return {
            "status": "ok",
            "content": f"已规划前往 {destination} 的徒步路线，全程约 12 公里，预计 5 小时。",
        }
    return {"status": "ok", "content": f"{tool_name} 调用成功（stub 响应）。"}


def _stub_response_route_plan(
    scenario: str,
    tool_results: list[dict],
) -> str:
    """Generate the user-visible text for a route_plan scenario.

    The text never fabricates realtime information. When evidence is
    missing (MCP unavailable / timeout / oversize), the response says
    so explicitly so the user knows the recommendation is not grounded.
    """
    if scenario == "slot_followup_destination":
        return "请告诉我你想去的目的地，我才能推荐合适的路线。"
    if scenario == "slot_followup_location":
        return "请授权当前位置或在消息中说明城市，我才能查询天气。"
    if scenario == "mcp_timeout":
        return (
            "天气/地图服务调用超时，未能获取实时证据。"
            "请稍后重试；在此之前我不会编造路线或天气信息。"
        )
    if scenario == "mcp_oversize":
        return (
            "天气/地图服务返回的数据过大被截断，证据不完整。"
            "请稍后重试或缩小查询范围；我不会基于不完整数据编造推荐。"
        )
    if scenario == "route_no_evidence":
        return (
            "地图 MCP 不可用，无法获取路线证据。"
            "请稍后重试；在此之前我不会编造具体路线推荐。"
        )
    if scenario == "weather_no_evidence":
        return (
            "天气 MCP 不可用，无法获取实时天气。"
            "请稍后重试；在此之前我不会编造天气信息。"
        )
    if scenario == "route_with_evidence":
        # Summarize the tool result without exposing internal tool names.
        summary = tool_results[0]["content"] if tool_results else "路线已规划。"
        return f"基于地图证据：{summary} 请根据体能和天气酌情调整。"
    if scenario == "location_weather":
        summary = tool_results[0]["content"] if tool_results else "天气已查询。"
        return f"基于实时天气：{summary}"
    # normal
    return "已收到你的问题。这是 route_plan 模式的 stub 响应，正式 LLM 集成后将由服务端接管生成。"


# ---- SSE generator --------------------------------------------------------


# ``Emit`` is the closure passed to flow generators. It builds the next
# envelope, persists it (when a runs repository is attached), and returns
# the SSE frame. Centralizing the build+persist+frame sequence keeps the
# flow generators focused on event ordering and contract correctness.
Emit = Callable[[str, dict[str, Any]], str]


async def _stream_route_plan_flow(
    req: ChatV2Request,
    stream: EventStream,
    profile: CapabilityProfile,
    emit: Emit,
) -> AsyncGenerator[str, None]:
    """route_plan profile flow (Ticket 05).

    Emits the route_plan event sequence:
    progress → [tool_call → tool_result]? → text → done

    MCP evidence is emitted only when the relevant MCP server is
    available. When unavailable / timeout / oversize, no tool_call is
    emitted for the failure modes that never executed; the text event
    explicitly states evidence is missing so the user knows the
    response is not grounded in realtime data.

    Credential isolation: this function never reads env values directly.
    ``check_mcp_available`` reads env vars but returns only a bool; the
    actual credential expansion happens inside the (stub) MCP connector
    and never reaches the event stream or logs.

    Ticket 06: events are persisted via the ``emit`` closure, which
    wraps :meth:`EventStream.build` + persistence + SSE framing. The
    flow generator stays unaware of whether persistence is active.
    """
    has_location = (
        req.current_location is not None
        and bool(
            req.current_location.city
            or req.current_location.adcode
            or (req.current_location.latitude and req.current_location.longitude)
        )
    )

    # Check MCP availability once for the route_plan profile. The stub
    # uses the AMap API key as the availability signal; real integration
    # would also probe the subprocess / HTTP endpoint.
    amap_config = resolve_mcp_server("amap_weather")
    mcp_available = check_mcp_available(amap_config, settings)

    scenario = _stub_classify_route_plan(
        req.message, has_location=has_location, mcp_available=mcp_available
    )

    tool_results: list[dict] = []

    # Emit tool_call + tool_result when the scenario has MCP evidence.
    # Failure modes (timeout / oversize) still emit tool_call + tool_result
    # because the tool was *attempted* — the result carries the failure
    # status so the user sees that evidence collection failed, not that
    # the tool was silently skipped.
    if scenario == "route_with_evidence":
        args = {"destination": _extract_destination(req.message)}
        yield emit("tool_call", {
            "tool": "plan_route",
            "server_id": "amap_map",
            "args": args,
        })
        result = _simulate_mcp_call("amap_map", "plan_route", args, mode="ok")
        tool_results.append(result)
        yield emit("tool_result", {
            "tool": "plan_route",
            "server_id": "amap_map",
            "status": result["status"],
            "content": result["content"],
        })

    elif scenario == "location_weather":
        city = req.current_location.city if req.current_location else ""
        args = {"city": city or "当前位置"}
        yield emit("tool_call", {
            "tool": "get_current_weather",
            "server_id": "amap_weather",
            "args": args,
        })
        result = _simulate_mcp_call(
            "amap_weather", "get_current_weather", args, mode="ok"
        )
        tool_results.append(result)
        yield emit("tool_result", {
            "tool": "get_current_weather",
            "server_id": "amap_weather",
            "status": result["status"],
            "content": result["content"],
        })

    elif scenario == "mcp_timeout":
        # Simulate a timeout: tool was attempted, result reports timeout.
        # The text event then explicitly says evidence is missing.
        args = {"destination": _extract_destination(req.message) or "测试"}
        yield emit("tool_call", {
            "tool": "plan_route",
            "server_id": "amap_map",
            "args": args,
        })
        result = _simulate_mcp_call("amap_map", "plan_route", args, mode="timeout")
        tool_results.append(result)
        yield emit("tool_result", {
            "tool": "plan_route",
            "server_id": "amap_map",
            "status": result["status"],
            "content": result["content"],
        })

    elif scenario == "mcp_oversize":
        args = {"destination": _extract_destination(req.message) or "测试"}
        yield emit("tool_call", {
            "tool": "plan_route",
            "server_id": "amap_map",
            "args": args,
        })
        result = _simulate_mcp_call("amap_map", "plan_route", args, mode="oversize")
        tool_results.append(result)
        yield emit("tool_result", {
            "tool": "plan_route",
            "server_id": "amap_map",
            "status": result["status"],
            "content": result["content"],
        })

    # route_no_evidence / weather_no_evidence / slot_followup_* / normal:
    # no tool_call emitted. The text event carries the explicit
    # evidence-missing or slot-followup message.

    response_text = _stub_response_route_plan(scenario, tool_results)
    yield emit("text", {"content": response_text})

    yield emit("done", {})


def _extract_destination(message: str) -> str:
    """Pull a destination token from the message for the stub MCP call.

    Real integration uses the LLM's slot-filling output; the stub just
    matches against the known destination markers so the tool_call args
    look realistic in contract tests.
    """
    for marker in _ROUTE_PLAN_DESTINATION_MARKERS:
        if marker in message:
            return marker
    return ""


# ---- artifact profile + sandbox stub (Ticket 09) -------------------------
#
# The artifact profile owns a run-scoped sandbox. The sandbox is created
# per run (spec AC1), recorded as a lease in ``agent_control.sqlite``,
# and cleaned up on terminal (spec AC6). Fail-closed (spec AC5): any
# ``ComputerError`` becomes a terminal ``error`` event; the run NEVER
# falls back to executing on the host.
#
# The stub ``_stub_artifact_plan_content`` produces a deterministic
# hiking plan so contract tests can assert exact event sequences
# without an LLM API key. Real LLM integration replaces this when the
# UniHarness factory swaps from the stub to the pinned fork.

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


def _stub_artifact_plan_content(message: str) -> str:
    """Deterministic hiking plan content for the stub PDF.

    Real LLM integration replaces this with the model's structured plan
    output. The stub content is plain text so the sandbox write is
    byte-stable across runs (contract tests assert exact event payloads).
    """
    destination = _extract_destination(message) or "目的地"
    return (
        f"徒步计划\n"
        f"========\n"
        f"\n"
        f"目的地：{destination}\n"
        f"距离：约 12 公里\n"
        f"预计时长：5 小时\n"
        f"\n"
        f"行程：\n"
        f"1. 起点集合，检查装备\n"
        f"2. 沿步道上行，注意补给\n"
        f"3. 抵达观景点，休息用餐\n"
        f"4. 下撤至终点，清点人员\n"
        f"\n"
        f"风险提示：天气多变，请携带防雨装备。\n"
    )


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


def _persist_failed_artifact(
    runs_repo: RunRepository | None,
    run_id: str | None,
    conversation_id: str,
    artifact_id: str,
    *,
    public_filename: str,
    mime_type: str,
    size_bytes: int,
    sha256: str,
    storage_path: str,
    failure_reason: str,
) -> None:
    """Persist a ``failed`` evidence row (spec AC4).

    Called BEFORE the sandbox is destroyed so the manifest + failure
    reason survive cleanup. Spec AC4 requires "sandbox 不会在证据保存前
    被销毁" — therefore persistence failure here MUST propagate so the
    caller can preserve the sandbox for investigation instead of
    destroying evidence. The caller (``_stream_artifact_flow``) catches
    :class:`ArtifactEvidencePersistenceError` and skips ``_cleanup_sandbox``.
    """
    if runs_repo is None or run_id is None:
        return
    created_at = _now_iso()
    expires_at = _expires_at_iso(created_at)
    try:
        runs_repo.insert_artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            conversation_id=conversation_id,
            public_filename=public_filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            storage_path=storage_path,
            status="failed",
            failure_reason=failure_reason,
            created_at=created_at,
            expires_at=expires_at,
        )
    except Exception:
        logger.exception(
            "chat_v2: failed to persist evidence row for artifact_id=%s; "
            "raising to preserve sandbox per AC4",
            artifact_id,
        )
        raise ArtifactEvidencePersistenceError(
            f"failed to persist evidence row for artifact_id={artifact_id}"
        )


def _export_artifact_to_host(
    *,
    computer: Computer,
    workspace_path: str,
    relative_path: str,
    declared_mime: str,
    runs_repo: RunRepository | None,
    run_id: str | None,
    conversation_id: str,
) -> dict[str, Any]:
    """Build manifest, verify on host, persist artifact, record in DB.

    Returns a dict with the public metadata for the ``artifact.available``
    event payload: ``artifact_id``, ``public_filename``, ``mime_type``,
    ``size_bytes``, ``sha256``, ``expires_at``. Spec AC3: no sandbox or
    host paths in the returned dict.

    Spec AC1 + AC2: the host re-derives size + SHA-256 from the bytes
    it reads via :meth:`Computer.read_artifact_bytes` and rejects any
    mismatch with the sandbox-declared manifest. Absolute paths, ``..``,
    symlink escape, missing file, MIME mismatch, checksum mismatch, and
    files over 25 MB are rejected.

    Spec AC4: on ANY failure (manifest build, byte read, validation,
    host persist, DB insert) a ``failed`` evidence row is persisted
    BEFORE this function returns/raises. The caller's ``finally`` block
    may then safely destroy the sandbox — the evidence survives.

    Raises :class:`ArtifactExportError` on failure. The caller emits a
    terminal ``error`` event; no half-built artifact is published.
    """
    artifact_id = str(uuid.uuid4())

    # 1. Sandbox builds the manifest (declares size + sha + mime).
    try:
        manifest = computer.build_manifest(
            workspace_path, relative_path, declared_mime
        )
    except ComputerError as exc:
        # Manifest build failed before we have any declarative fields.
        # Persist a failed row with the best-effort fields we can derive
        # from the request (relative_path → public_filename).
        _persist_failed_artifact(
            runs_repo, run_id, conversation_id, artifact_id,
            public_filename=Path(relative_path).name,
            mime_type=declared_mime,
            size_bytes=0,
            sha256="",
            storage_path="",
            failure_reason=f"manifest build failed: {exc}",
        )
        raise ArtifactExportError(f"manifest build failed: {exc}") from exc

    # 2. Host reads the bytes back. The host re-derives size + sha from
    # these bytes — the sandbox's declared values are NOT trusted.
    try:
        actual_bytes = computer.read_artifact_bytes(workspace_path, relative_path)
    except ComputerError as exc:
        _persist_failed_artifact(
            runs_repo, run_id, conversation_id, artifact_id,
            public_filename=manifest.public_filename,
            mime_type=manifest.declared_mime,
            size_bytes=manifest.size_bytes,
            sha256=manifest.sha256,
            storage_path="",
            failure_reason=f"artifact read failed: {exc}",
        )
        raise ArtifactExportError(
            f"artifact read failed: {exc}", manifest=manifest
        ) from exc

    # 3. Validate manifest against actual bytes (spec AC1, AC2).
    #    validate_manifest re-derives size + sha256 from actual_bytes
    #    and rejects any mismatch. It also rejects path traversal,
    #    symlink escape (via relative_path validation), MIME mismatch,
    #    and >25 MB files.
    try:
        validate_manifest(manifest, actual_bytes)
    except ArtifactExportError:
        # validate_manifest already populated the manifest on the exc
        # via ArtifactExportError.__init__ when raised from validators
        # that have the manifest in scope. Re-raise after evidence row.
        _persist_failed_artifact(
            runs_repo, run_id, conversation_id, artifact_id,
            public_filename=manifest.public_filename,
            mime_type=manifest.declared_mime,
            size_bytes=manifest.size_bytes,
            sha256=manifest.sha256,
            storage_path="",
            failure_reason="manifest validation failed",
        )
        raise

    # 4. Symlink escape check (defense-in-depth, spec AC2). The
    #    relative_path validator already rejected ``..`` and absolute
    #    paths; this catches symlinks inside the workspace that point
    #    outside it.
    full_path = os.path.join(workspace_path, relative_path)
    if is_symlink_escape(workspace_path, full_path):
        _persist_failed_artifact(
            runs_repo, run_id, conversation_id, artifact_id,
            public_filename=manifest.public_filename,
            mime_type=manifest.declared_mime,
            size_bytes=manifest.size_bytes,
            sha256=manifest.sha256,
            storage_path="",
            failure_reason="symlink escape detected",
        )
        raise ArtifactExportError(
            "symlink escape detected", manifest=manifest
        )

    # 5. Compute host storage path + persist bytes atomically.
    storage_path = host_storage_path(
        _artifacts_dir(), artifact_id, manifest.public_filename
    )
    try:
        persist_artifact_bytes(storage_path, actual_bytes)
    except Exception as exc:
        _persist_failed_artifact(
            runs_repo, run_id, conversation_id, artifact_id,
            public_filename=manifest.public_filename,
            mime_type=manifest.declared_mime,
            size_bytes=manifest.size_bytes,
            sha256=manifest.sha256,
            storage_path=storage_path,
            failure_reason=f"host persist failed: {exc}",
        )
        raise ArtifactExportError(
            f"host persist failed: {exc}", manifest=manifest
        ) from exc

    # 6. Insert the artifact row (status='available'). The row is the
    #    source of truth for the download endpoint — the file on disk
    #    is the blob, the row carries the public metadata + expiry.
    created_at = _now_iso()
    expires_at = _expires_at_iso(created_at)
    if runs_repo is not None and run_id is not None:
        try:
            runs_repo.insert_artifact(
                artifact_id=artifact_id,
                run_id=run_id,
                conversation_id=conversation_id,
                public_filename=manifest.public_filename,
                mime_type=manifest.declared_mime,
                size_bytes=manifest.size_bytes,
                sha256=manifest.sha256,
                storage_path=storage_path,
                status="available",
                failure_reason=None,
                created_at=created_at,
                expires_at=expires_at,
            )
        except Exception as exc:
            # DB insert failed after the file was written — best-effort
            # delete the orphan so storage does not accumulate blobs
            # with no matching row. The run still fails; the operator
            # checks logs for the orphan cleanup.
            try:
                os.remove(storage_path)
            except OSError:
                logger.warning(
                    "chat_v2: orphan artifact file left at %s after DB insert failed",
                    storage_path,
                )
            _persist_failed_artifact(
                runs_repo, run_id, conversation_id, artifact_id,
                public_filename=manifest.public_filename,
                mime_type=manifest.declared_mime,
                size_bytes=manifest.size_bytes,
                sha256=manifest.sha256,
                storage_path="",
                failure_reason=f"artifact DB insert failed: {exc}",
            )
            raise ArtifactExportError(
                f"artifact DB insert failed: {exc}", manifest=manifest
            ) from exc

    return {
        "artifact_id": artifact_id,
        "public_filename": manifest.public_filename,
        "mime_type": manifest.declared_mime,
        "size_bytes": manifest.size_bytes,
        "sha256": manifest.sha256,
        "expires_at": expires_at,
    }


async def _stream_artifact_flow(
    req: ChatV2Request,
    stream: EventStream,
    profile: CapabilityProfile,
    emit: Emit,
    runs_repo: RunRepository | None,
    run_id: str | None,
) -> AsyncGenerator[str, None]:
    """artifact profile flow (Tickets 09 + 10).

    Event sequence emitted by this flow:

        tool_call(generate_pdf)
          → tool_result
          → artifact_candidate
          → artifact.available     # Ticket 10: host verified + persisted
          → text
          → done

    The outer :func:`_stream_chat_v2` emits ``run_started`` → ``progress``
    → ``heartbeat`` before dispatching here, so the full wire sequence is:
    run_started → progress → heartbeat → tool_call → tool_result →
    artifact_candidate → artifact.available → text → done.

    Fail-closed contract: any ``ComputerError`` (creation / quota /
    timeout / cleanup) becomes a terminal ``error`` event. The run
    NEVER falls back to host execution.

    Ticket 10 export contract: after ``artifact_candidate`` the host
    builds a manifest, re-reads the bytes, re-derives size + SHA-256,
    and persists the artifact under an opaque id. On success an
    ``artifact.available`` event is emitted carrying ONLY the opaque
    ``artifact_id`` + public metadata (no sandbox or host paths, spec
    AC3). On failure a ``failed`` evidence row is persisted BEFORE the
    sandbox is destroyed (spec AC4) — the cleanup ``finally`` runs
    only after the evidence row is saved.

    Cancel observation: ``_is_cancelled`` is polled between major
    yields. When cancelled, a ``cancelled`` terminal is emitted and the
    sandbox is cleaned up in the ``finally`` block (spec AC6 requires
    cleanup on completion OR cancellation).

    The sandbox lease is persisted via :meth:`RunRepository.create_sandbox_lease`
    so operators can audit stale sandboxes. Lease persistence failure
    does NOT block the run — the workspace path is still logged.
    """
    computer = _build_artifact_computer()

    # Spec AC4: when evidence persistence fails inside
    # ``_export_artifact_to_host``, the sandbox MUST be preserved so
    # operators can recover the artifact bytes manually. This flag is
    # set only on that path; the ``finally`` block checks it to skip
    # ``_cleanup_sandbox``. The lease is intentionally leaked — the
    # error event tells the operator which run_id needs investigation.
    preserve_sandbox = False

    # Provision sandbox workspace. Fail-closed: any ComputerError → terminal.
    try:
        workspace_path = computer.create_workspace(run_id or req.request_id)
    except SandboxQuotaError as exc:
        yield emit("error", {
            "code": "sandbox_quota",
            "message": f"沙箱配额不足：{exc}",
        })
        _mark_terminal_safe(runs_repo, run_id, "error")
        return
    except SandboxCreationError as exc:
        yield emit("error", {
            "code": "sandbox_creation_failed",
            "message": f"沙箱创建失败：{exc}",
        })
        _mark_terminal_safe(runs_repo, run_id, "error")
        return
    except ComputerError as exc:
        yield emit("error", {
            "code": "sandbox_unavailable",
            "message": f"沙箱不可用：{exc}",
        })
        _mark_terminal_safe(runs_repo, run_id, "error")
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
                "chat_v2: create_sandbox_lease failed for run_id=%s; "
                "continuing without lease record (workspace=%s)",
                run_id, workspace_path,
            )

    try:
        # Cancel check before tool_call.
        if _is_cancelled(runs_repo, run_id):
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
        # be reused when the user approves). The ``preserve_sandbox`` flag
        # instructs the ``finally`` block to skip cleanup.
        gate = PermissionGate()
        verdict = gate.evaluate(
            profile, "generate_pdf", {"filename": _ARTIFACT_FILENAME}
        )
        if verdict.decision == PermissionDecision.DENY:
            yield emit("error", {
                "code": "tool_denied",
                "message": f"工具 generate_pdf 被拒绝：{verdict.reason}",
            })
            _mark_terminal_safe(runs_repo, run_id, "error")
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
                        "chat_v2: insert_approval failed for run_id=%s; "
                        "existing row (if any) is authoritative",
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
                        "chat_v2: mark_run_paused failed for run_id=%s: %s",
                        run_id, exc,
                    )
                    yield emit("cancelled", {
                        "code": "cancelled",
                        "message": "运行已被取消。",
                    })
                    return
            preserve_sandbox = True
            return

        plan_content = _stub_artifact_plan_content(req.message)
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
            yield emit("error", {
                "code": "sandbox_timeout",
                "message": f"沙箱命令超时：{exc}",
            })
            _mark_terminal_safe(runs_repo, run_id, "error")
            return
        except ComputerError as exc:
            yield emit("tool_result", {
                "tool": "generate_pdf",
                "status": "error",
                "content": f"PDF 生成失败：{exc}",
            })
            yield emit("error", {
                "code": "artifact_generation_failed",
                "message": f"artifact 生成失败：{exc}",
            })
            _mark_terminal_safe(runs_repo, run_id, "error")
            return

        yield emit("tool_result", {
            "tool": "generate_pdf",
            "status": "ok",
            "content": f"已在沙箱中生成 {relative_path}",
            "path": relative_path,
        })

        # Cancel check before artifact_candidate.
        if _is_cancelled(runs_repo, run_id):
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

        # Ticket 10: host-side export. The sandbox declared what it
        # produced; the host now re-reads the bytes, re-derives size +
        # SHA-256, validates the manifest, and persists the artifact
        # under an opaque id. Failure here is terminal — the run does
        # NOT publish a half-built artifact (spec AC4). The evidence
        # row is saved BEFORE the sandbox cleanup runs (in ``finally``).
        conversation_id = req.conversation_id or f"conv-{req.request_id}"
        try:
            export = _export_artifact_to_host(
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
            yield emit("error", {
                "code": "artifact_evidence_persistence_failed",
                "message": (
                    f"artifact 证据持久化失败，sandbox 已保留以便排查：{exc}。"
                    f"请联系运维处理 run_id={run_id} 的沙箱。"
                ),
            })
            _mark_terminal_safe(runs_repo, run_id, "error")
            return
        except ArtifactExportError as exc:
            yield emit("error", {
                "code": "artifact_export_failed",
                "message": f"artifact 导出失败：{exc}",
            })
            _mark_terminal_safe(runs_repo, run_id, "error")
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

        yield emit("done", {})
        _mark_terminal_safe(runs_repo, run_id, "done")

    finally:
        # Spec AC6: cleanup on completion OR cancellation OR error.
        # Always runs so the sandbox is never leaked. Failure is recorded
        # on the lease but does not change the terminal already emitted.
        #
        # Spec AC4: when export failed, the evidence row was already
        # persisted inside _export_artifact_to_host BEFORE this finally
        # runs — so destroying the sandbox here does not lose evidence.
        #
        # Spec AC4 exception: when evidence persistence itself failed,
        # ``preserve_sandbox`` is set and the sandbox is NOT destroyed —
        # operators must recover the artifact bytes manually. The lease
        # is intentionally leaked and the run's error event tells the
        # operator which run_id to investigate.
        if preserve_sandbox:
            logger.warning(
                "chat_v2: preserving sandbox for run_id=%s (evidence "
                "persistence failed); lease is intentionally leaked",
                run_id,
            )
        else:
            _cleanup_sandbox(computer, workspace_path, runs_repo, run_id)


# ---- cowork profile (Ticket 13) ------------------------------------------
#
# The cowork profile extends artifact with E2B-scoped Bash. The flow is
# intentionally minimal: it runs ONE sandbox_bash command (the stub
# derives a deterministic command from the user message so contract
# tests can assert exact event sequences), then surfaces the captured
# stdout/stderr + exit code as a tool_result event. Real LLM integration
# replaces the stub command with the model's tool-call args.
#
# Spec AC3: all Bash calls go through PermissionGate (HIGH → approval),
# effect ledger (authorized → executing → committed/failed), timeout
# cap, output cap, and an audit event (tool_result carries exit_code,
# truncated flag, duration_ms).
#
# Spec AC4: the command runs inside the run's sandbox workspace with
# ``env={}`` and ``cwd=workspace_path``; the host environment is not
# accessible. See :meth:`LocalNativeComputer.run_bash` /
# :meth:`RemoteE2BComputer.run_bash`.
#
# Spec AC5: the flow supports approval (run.paused), cancellation
# (observed between yields), timeout (SandboxTimeoutError → error
# terminal), and outcome_unknown (effect ledger crash recovery during
# resume). The resume flow in :func:`_stream_resume_flow` dispatches
# on ``resolved.tool_name`` to handle both ``generate_pdf`` and
# ``sandbox_bash``.
_COWORK_STUB_FILENAME = "cowork-output.txt"


def _stub_cowork_bash_command(message: str) -> str:
    """Derive a deterministic bash command for the cowork stub.

    Real LLM integration replaces this with the model's tool-call args.
    The stub command writes a small file to the workspace and prints
    its contents so contract tests can assert exact stdout + the
    created file path.

    The command intentionally uses only shell built-ins (``echo``,
    ``cat``, ``>``) so the empty-``PATH`` sandbox can execute it
    without reaching host binaries (spec AC4).
    """
    # Extract a simple label from the message; default to "task" so the
    # stub is deterministic for empty messages. Real LLM integration
    # replaces this with the model's structured command output.
    label = "task"
    for marker in ("生成", "create", "list", "build"):
        if marker in message.lower():
            label = marker
            break
    output = f"cowork stub: {label}"
    return (
        f"printf '%s\\n' '{output}' > {_COWORK_STUB_FILENAME} "
        f"&& printf '%s\\n' '{output}'"
    )


async def _stream_cowork_flow(
    req: ChatV2Request,
    stream: EventStream,
    profile: CapabilityProfile,
    emit: Emit,
    runs_repo: RunRepository | None,
    run_id: str | None,
) -> AsyncGenerator[str, None]:
    """cowork profile flow (Ticket 13).

    Event sequence emitted by this flow:

        tool_call(sandbox_bash)
          → approval.required   # HIGH risk → user must approve
          → run.paused          # stream terminal; resume via /resume

    After user approves, :func:`_stream_resume_flow` continues:

        approval.resolved
          → tool_result(sandbox_bash, ok | timeout | error)
          → text
          → done

    Spec AC2: cowork is authorized ONLY when
    ``settings.agent_v2_cowork_policy`` is ``trusted`` or ``staging``.
    The caller (:func:`_stream_chat_v2`) checks this BEFORE dispatching
    here; this flow does not re-check.

    Spec AC3: PermissionGate evaluates ``sandbox_bash`` (HIGH) →
    ``needs_approval``. The approval row + run.paused terminal are
    persisted; the sandbox lease is held until the user decides.

    Spec AC4: the bash command runs inside the sandbox workspace with
    ``env={}`` (see :meth:`Computer.run_bash`). The host environment
    is not accessible.

    Spec AC5: cancel observation between yields; cleanup runs in
    ``finally`` so the sandbox is never leaked. Timeout and
    outcome_unknown paths emit terminal ``error`` events.

    Spec AC6: this flow does NOT call ``generate_pdf``; the cowork
    profile inherits the artifact tools but the stub stage only
    exercises ``sandbox_bash``. A real LLM may chain multiple tool
    calls; each will go through its own approval + effect ledger.
    """
    computer = _build_artifact_computer()
    preserve_sandbox = False

    # Provision sandbox workspace. Fail-closed: any ComputerError → terminal.
    try:
        workspace_path = computer.create_workspace(run_id or req.request_id)
    except SandboxQuotaError as exc:
        yield emit("error", {
            "code": "sandbox_quota",
            "message": f"沙箱配额不足：{exc}",
        })
        _mark_terminal_safe(runs_repo, run_id, "error")
        return
    except SandboxCreationError as exc:
        yield emit("error", {
            "code": "sandbox_creation_failed",
            "message": f"沙箱创建失败：{exc}",
        })
        _mark_terminal_safe(runs_repo, run_id, "error")
        return
    except ComputerError as exc:
        yield emit("error", {
            "code": "sandbox_unavailable",
            "message": f"沙箱不可用：{exc}",
        })
        _mark_terminal_safe(runs_repo, run_id, "error")
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
                "chat_v2: create_sandbox_lease failed for run_id=%s; "
                "continuing without lease record (workspace=%s)",
                run_id, workspace_path,
            )

    # Derive the stub bash command. Real LLM integration replaces this
    # with the model's tool-call args.
    bash_command = _stub_cowork_bash_command(req.message)
    bash_args = {"command": bash_command}

    try:
        # Cancel check before tool_call.
        if _is_cancelled(runs_repo, run_id):
            yield emit("cancelled", {
                "code": "cancelled",
                "message": "运行已被取消。",
            })
            return

        # Ticket 13 AC3 — PermissionGate check BEFORE tool_call.
        # sandbox_bash is HIGH risk → needs_approval. The flow pauses
        # the run and emits run.paused as the stream terminal; the
        # sandbox lease is held until the user decides (spec AC4: the
        # same sandbox must be reused when the user approves).
        gate = PermissionGate()
        verdict = gate.evaluate(profile, "sandbox_bash", bash_args)
        if verdict.decision == PermissionDecision.DENY:
            yield emit("error", {
                "code": "tool_denied",
                "message": f"工具 sandbox_bash 被拒绝：{verdict.reason}",
            })
            _mark_terminal_safe(runs_repo, run_id, "error")
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
                        "chat_v2: insert_approval failed for run_id=%s; "
                        "existing row (if any) is authoritative",
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
                        "chat_v2: mark_run_paused failed for run_id=%s: %s",
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
        yield emit("error", {
            "code": "internal_error",
            "message": "sandbox_bash 未走审批流程，已拒绝执行。",
        })
        _mark_terminal_safe(runs_repo, run_id, "error")
        return

    except TerminalEventAlreadyEmitted:
        logger.error(
            "chat_v2: attempted to emit after terminal for request_id=%s",
            req.request_id,
        )
        return
    finally:
        if preserve_sandbox:
            logger.info(
                "chat_v2: preserving sandbox for paused cowork run_id=%s",
                run_id,
            )
        else:
            _cleanup_sandbox(computer, workspace_path, runs_repo, run_id)


async def _stream_chat_v2(req: ChatV2Request, request: Request) -> AsyncGenerator[str, None]:
    """Produce the v2 SSE stream for a single run.

    Guarantees enforced here:

    * Exactly one terminal event — the ``except Exception`` arm emits a
      terminal error if none was emitted yet. ``BaseException`` (e.g.
      ``KeyboardInterrupt``) is intentionally not caught: the process is
      shutting down and the run status API (Ticket 06) reconciles.
    * Monotonic sequence — delegated to :class:`EventStream`.
    * Unknown internal events dropped and counted — simulated here by
      calling ``drop_unknown`` for representative framework event names.

    Ticket 06 — persistence:

    * When ``app.state.agent_v2_runtime`` carries a :class:`RunRepository`,
      the run is created idempotently on ``request_id`` before any event
      is emitted. Repeats return the stored events and stop (spec AC2).
    * Every emitted event is persisted via :meth:`RunRepository.persist_event`
      so reconnect (``GET /events?from_seq=N``) can replay the stream
      without re-executing tools (spec AC4).
    * The run is marked ``done``/``error`` on terminal. ``cancelled`` is
      observed via polling between yields — the cancel API transitions
      the DB row, and the stream emits a ``cancelled`` terminal and stops.
    """
    stream = EventStream(
        request_id=req.request_id,
        conversation_id=req.conversation_id or "",
    )

    # Resolve the runs repository, if any. When v2 is disabled or the
    # runtime is the Ticket 03 placeholder (no ``runs`` attribute), this
    # is None and persistence is skipped — the stream still produces the
    # same envelope contract, just without durable replay.
    runtime = getattr(request.app.state, "agent_v2_runtime", None)
    runs_repo: RunRepository | None = getattr(runtime, "runs", None) if runtime else None
    memory = getattr(runtime, "memory", None) if runtime else None
    telemetry = getattr(runtime, "telemetry", None) if runtime else None
    telemetry_started = time.perf_counter()
    subject_id = getattr(request.state, "subject_id", None)

    # Idempotent run creation. ``conversation_id`` defaults to a derived
    # value so single-shot runs without a client-supplied id still persist.
    run_id: str | None = None
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
            # Persistence failure must not break the in-memory stream.
            # Log and continue without persistence; the client still gets
            # the SSE contract, just without durable replay for this run.
            logger.exception(
                "chat_v2: create_run_idempotent failed for request_id=%s; "
                "continuing without persistence",
                req.request_id,
            )
            run_id = None
            created = True  # proceed as if fresh
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
            # Propagate the server-generated run_id to every built envelope
            # (spec decision #26). The frontend reads this from the first
            # event to call /snapshot, /cancel, and /events.
            if run_id:
                stream.run_id = run_id
                if created and telemetry is not None:
                    telemetry.record(
                        run_id,
                        profile_name=req.profile_id,
                        status="running",
                        model_name=getattr(settings, "openai_model", ""),
                    )
            if not created:
                # Spec AC2: duplicate request_id — replay stored events
                # and stop. No model call, no tool re-execution.
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
                    yield _sse_frame(envelope)
                if has_terminal:
                    return
                # No terminal in storage (process crashed mid-stream).
                # Synthesize one from the run's terminal state so the
                # client never observes a stream without a terminal.
                # Advance the stream counter past the replayed events so
                # the synthesized terminal sequence is strictly greater
                # than any stored event (spec AC4 monotonicity), and
                # persist it so future reconnects see the terminal.
                snapshot = runs_repo.get_snapshot(run_id)
                terminal_type = _synthesize_terminal_type(snapshot)
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
                    yield _sse_frame(terminal_envelope)
                return

    # Build the emit closure. This wraps build + persist + frame so the
    # flow generators below stay linear and persistence-aware.
    def emit(event_type: str, payload: dict[str, Any]) -> str:
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
        return _sse_frame(envelope)

    try:
        if conversation_forbidden:
            yield emit("error", {
                "code": "conversation_forbidden",
                "message": "Conversation not accessible",
            })
            return

        # v2 disabled → terminal error. The Gateway should not route here
        # when v2 is off, but the endpoint defends in depth. No internal
        # events have been processed, so unknown_dropped stays 0.
        if not settings.agent_v2_enabled:
            yield emit("error", {
                "code": "v2_disabled",
                "message": "Chat v2 is disabled (AGENT_V2_ENABLED=false).",
            })
            _mark_terminal_safe(runs_repo, run_id, "error")
            return

        # Resolve profile (server-defined; client cannot override).
        try:
            profile = resolve_profile(req.profile_id)
        except UnknownProfileError as exc:
            yield emit("error", {
                "code": "unknown_profile",
                "message": str(exc),
            })
            _mark_terminal_safe(runs_repo, run_id, "error")
            return

        # Ticket 13 AC2 — cowork authorization gate. The cowork profile
        # is registered but NOT callable by default. Only an explicit
        # operator-set policy of ``trusted`` or ``staging`` authorizes
        # the request. Clients cannot upgrade to cowork by sending the
        # profile id — this server-side gate is the authority.
        # The check runs AFTER profile resolution so an unknown profile
        # still returns ``unknown_profile`` (not ``cowork_unauthorized``).
        if profile.profile_id == ProfileId.COWORK and not is_cowork_authorized(settings):
            yield emit("error", {
                "code": "cowork_unauthorized",
                "message": (
                    "cowork profile 未授权 (AGENT_V2_COWORK_POLICY=disabled)。"
                    "请联系运维启用 trusted/staging 策略。"
                ),
            })
            _mark_terminal_safe(runs_repo, run_id, "error")
            return

        # Model failure scenario: no API key configured. The contract
        # requires this to be a terminal ``error`` event, not an HTTP 500.
        # ``run_started`` is emitted first so the client sees the run begin
        # before the model-unavailable terminal (spec decision #27).
        if not settings.openai_api_key:
            yield emit("run_started", {})
            yield emit("error", {
                "code": "model_unavailable",
                "message": "主模型未配置 API Key (OPENAI_API_KEY 为空)。",
            })
            _mark_terminal_safe(runs_repo, run_id, "error")
            return

        # Normal flow: run_started → progress → text → done. Only now do
        # we simulate unknown internal events that the adapter would drop
        # — the count reflects events actually seen during real processing,
        # not the pre-flight v2/profile/key gates that never produced events.
        stream.drop_unknown("langchain_chain_start")
        stream.drop_unknown("uniharness_llm_token")

        # Spec decision #27 + AC2: ``run_started`` is a distinct event
        # type signaling the run has begun. It is the first user-visible
        # event in a successful flow; ``progress`` follows for ongoing
        # status updates.
        yield emit("run_started", {})

        yield emit("progress", {
            "stage": "processing",
            "message": "正在生成回复...",
        })

        # Spec decision #27 + AC2: heartbeat keeps the SSE connection
        # alive during model thinking / tool calls. In the stub stage the
        # flow is fast, but emitting one here exercises the heartbeat
        # path so the frontend reducer and reconnect replay handle it
        # correctly before real LLM integration adds long waits.
        yield emit("heartbeat", {})

        # Poll for cancellation between major yields. The cancel API
        # transitions the DB row; we observe it and stop the stream
        # with a ``cancelled`` terminal (spec AC5).
        if _is_cancelled(runs_repo, run_id):
            yield emit("cancelled", {
                "code": "cancelled",
                "message": "运行已被取消。",
            })
            return

        if profile.profile_id == ProfileId.ROUTE_PLAN:
            # Ticket 05: route_plan flow. Intake classifies the scenario
            # and decides whether MCP evidence is needed. The chat profile
            # NEVER enters this branch — intake non-escalation is enforced
            # by dispatching on the resolved profile, not on message content.
            async for frame in _stream_route_plan_flow(req, stream, profile, emit):
                # Check cancel between each yielded frame so a long-running
                # tool_call/tool_result chain can be interrupted promptly.
                if _is_cancelled(runs_repo, run_id) and not stream.terminal_emitted:
                    yield emit("cancelled", {
                        "code": "cancelled",
                        "message": "运行已被取消。",
                    })
                    return
                yield frame
            _mark_terminal_safe(runs_repo, run_id, "done")
            return

        if profile.profile_id == ProfileId.ARTIFACT:
            # Ticket 09: artifact flow. The sandbox is created and cleaned
            # up inside ``_stream_artifact_flow``; cancel/error paths run
            # the cleanup in a ``finally`` block so the sandbox is never
            # leaked. Cancel observation between yields is handled inside
            # the flow because cleanup must still run.
            async for frame in _stream_artifact_flow(
                req, stream, profile, emit, runs_repo, run_id
            ):
                yield frame
            return

        if profile.profile_id == ProfileId.COWORK:
            # Ticket 13: cowork flow. Same sandbox lifecycle as artifact
            # (create → lease → cleanup in finally). The flow runs the
            # sandbox_bash tool via PermissionGate + approval + effect
            # ledger; cancel/error paths run cleanup in ``finally``.
            async for frame in _stream_cowork_flow(
                req, stream, profile, emit, runs_repo, run_id
            ):
                yield frame
            return

        # chat profile flow (Ticket 04) + controlled opt-in memory tools.
        response_text: str | None = None
        if memory is not None and subject_id and memory.memory_enabled(subject_id):
            sensitive = extract_sensitive_memory(req.message)
            if sensitive is not None:
                candidate = memory.create_candidate(subject_id, **sensitive)
                if candidate is not None:
                    yield emit("memory.consent_required", candidate)
                    response_text = "这条信息需要你逐条确认后才会保存。"
            else:
                preference = extract_ordinary_preference(req.message)
            if sensitive is None and preference is not None:
                verdict = PermissionGate().evaluate(
                    profile, "memory_save", {"content": preference}
                )
                if verdict.decision == PermissionDecision.ALLOW:
                    yield emit("tool_call", {
                        "tool": "memory_save",
                        "args": {"content": preference},
                    })
                    memory_id = memory.save_preference(subject_id, preference)
                    yield emit("tool_result", {
                        "tool": "memory_save",
                        "ok": memory_id is not None,
                        "memory_id": memory_id,
                    })
                    response_text = f"已记住你的偏好：{preference}"
            elif asks_for_preferences(req.message):
                verdict = PermissionGate().evaluate(profile, "memory_search", {})
                if verdict.decision == PermissionDecision.ALLOW:
                    yield emit("tool_call", {"tool": "memory_search", "args": {}})
                    memories = memory.list_active_memories(subject_id)
                    yield emit("tool_result", {
                        "tool": "memory_search",
                        "ok": True,
                        "count": len(memories),
                    })
                    response_text = (
                        "你的偏好：" + "；".join(item["content"] for item in memories)
                        if memories
                        else "暂未保存偏好。"
                    )

        if response_text is None:
            classification = _stub_classify(req.message)
            response_text = _stub_response(req.message, classification, profile)

        yield emit("text", {"content": response_text})

        if _is_cancelled(runs_repo, run_id):
            yield emit("cancelled", {
                "code": "cancelled",
                "message": "运行已被取消。",
            })
            return

        yield emit("done", {})
        _mark_terminal_safe(runs_repo, run_id, "done")

    except TerminalEventAlreadyEmitted:
        # A second terminal was attempted (programmer error). Log and stop;
        # the stream already has its one terminal event.
        logger.error(
            "chat_v2: attempted to emit after terminal for request_id=%s",
            req.request_id,
        )
        return

    except Exception as exc:
        # Unexpected error. Emit a terminal error if the stream is still
        # open; otherwise the previous terminal stands.
        logger.exception("chat_v2: unexpected error for request_id=%s", req.request_id)
        if not stream.terminal_emitted:
            yield emit("error", {
                "code": "internal_error",
                "message": "内部错误，请重试。",
            })
        _mark_terminal_safe(runs_repo, run_id, "error")
        return


def _is_cancelled(runs_repo: RunRepository | None, run_id: str | None) -> bool:
    """Poll the cancel state. Returns False when persistence is off."""
    if runs_repo is None or run_id is None:
        return False
    try:
        return runs_repo.is_cancelled(run_id)
    except Exception:
        logger.exception("chat_v2: is_cancelled check failed; treating as not cancelled")
        return False


def _mark_terminal_safe(
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


def _synthesize_terminal_type(snapshot: RunSnapshot | None) -> str | None:
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
    """
    runtime = getattr(request.app.state, "agent_v2_runtime", None)
    capacity = getattr(runtime, "capacity", None)
    run_capacity_lease = None
    sandbox_capacity_lease = None
    if capacity is not None:
        try:
            run_capacity_lease = await capacity.acquire_run()
            needs_sandbox = req.profile_id == ProfileId.ARTIFACT.value or (
                req.profile_id == ProfileId.COWORK.value
                and is_cowork_authorized(settings)
            )
            if needs_sandbox:
                sandbox_capacity_lease = await capacity.acquire_sandbox(
                    req.request_id
                )
        except CapacityExceeded as exc:
            if run_capacity_lease is not None:
                run_capacity_lease.release()
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

    async def guarded_stream() -> AsyncGenerator[str, None]:
        try:
            async for frame in _stream_chat_v2(req, request):
                yield frame
        finally:
            if sandbox_capacity_lease is not None:
                keep_reserved = False
                runs_repo = getattr(runtime, "runs", None)
                run = (
                    runs_repo.control.get_run_by_request(req.request_id)
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
                    telemetry = getattr(runtime, "telemetry", None)
                    if telemetry is not None and run is not None:
                        telemetry.record(
                            run.run_id, resource_usage=sandbox.resource_usage
                        )
                if not keep_reserved:
                    sandbox_capacity_lease.release()
            if run_capacity_lease is not None:
                run_capacity_lease.release()

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
    """
    runs_repo = _get_runs_repo(request)
    snapshot = runs_repo.get_snapshot(run_id)
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
    events = runs_repo.get_events_since(run_id=run_id, from_seq=from_seq)
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
    """
    if body.decision not in ("approve", "reject"):
        raise HTTPException(
            status_code=422,
            detail="decision must be 'approve' or 'reject'",
        )

    runs_repo = _get_runs_repo(request)
    approval = runs_repo.get_approval(approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    _require_conversation_owner(request, approval.conversation_id)

    # AC5: idempotent duplicate (same decision_request_id) → return existing
    if (
        approval.decision_request_id is not None
        and approval.decision_request_id == body.decision_request_id
        and approval.is_terminal
    ):
        return _build_decision_response(approval, runs_repo, idempotent=True)

    # AC5: already resolved with a different decision_request_id → 409
    if approval.is_terminal:
        return JSONResponse(
            status_code=409,
            content={
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
                "chat_v2: mark_approval_expired failed for %s", approval_id
            )
        return JSONResponse(
            status_code=409,
            content={
                "approval_id": approval_id,
                "reason": "expired",
                "message": "审批已过期，请重新发起请求。",
            },
        )

    # Validate run state
    run = runs_repo.get_run(approval.run_id)
    if run is None:
        return JSONResponse(
            status_code=409,
            content={
                "approval_id": approval_id,
                "reason": "run_not_found",
                "message": "审批关联的运行不存在。",
            },
        )

    if run.is_terminal:
        # The run was cancelled (or hit an error) while paused. The
        # pause path set ``preserve_sandbox=True`` to keep the lease for
        # the resume; now that the run is terminal, the lease must be
        # cleaned up to avoid a leak (spec AC6: cleanup on completion OR
        # cancellation). We mark the lease as destroyed with a failure
        # reason; the actual sandbox process is best-effort cleaned by
        # the operator because we no longer have the Computer handle.
        if run.status == "cancelled":
            try:
                lease = runs_repo.get_sandbox_lease(approval.run_id)
                if lease is not None and lease.status == "active":
                    runs_repo.destroy_sandbox_lease(
                        run_id=approval.run_id,
                        failure_reason=f"run cancelled while paused (approval {approval_id})",
                    )
            except Exception:
                logger.exception(
                    "chat_v2: destroy_sandbox_lease failed for run_id=%s "
                    "during cancelled-while-paused cleanup",
                    approval.run_id,
                )
        return JSONResponse(
            status_code=409,
            content={
                "approval_id": approval_id,
                "reason": "run_terminal",
                "current_run_status": run.status,
                "message": f"运行已终态（{run.status}），无法继续。",
            },
        )

    if run.status != "paused":
        # Run is running/done/error/cancelled — not awaiting approval.
        return JSONResponse(
            status_code=409,
            content={
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
    # already destroyed, fail closed — do NOT resume into a new sandbox.
    lease = runs_repo.get_sandbox_lease(approval.run_id)
    if lease is None or lease.status != "active":
        return JSONResponse(
            status_code=409,
            content={
                "approval_id": approval_id,
                "reason": "sandbox_lease_lost",
                "message": (
                    "沙箱租赁已丢失或销毁，无法安全恢复运行（spec AC7）。"
                    "请重新发起请求。"
                ),
            },
        )

    # Persist the decision. resolve_approval raises RunAlreadyTerminal if
    # the approval was resolved by a concurrent request between our
    # ``get_approval`` and this call — surface as 409 already_resolved.
    target_status = "approved" if body.decision == "approve" else "rejected"
    try:
        resolved = runs_repo.resolve_approval(
            approval_id=approval_id,
            status=target_status,
            decision_request_id=body.decision_request_id,
        )
    except RunAlreadyTerminal:
        return JSONResponse(
            status_code=409,
            content={
                "approval_id": approval_id,
                "reason": "already_resolved",
                "message": "审批已被并发请求解决。",
            },
        )

    if resolved is None:
        raise HTTPException(status_code=404, detail="Approval not found")

    # Transition run ``paused`` → ``running`` so the resume endpoint can
    # pick it up. If the cancel API beat us, the run is now terminal and
    # the resume endpoint will surface the cancelled state.
    try:
        runs_repo.resume_run(approval.run_id)
    except RunAlreadyTerminal as exc:
        logger.warning(
            "chat_v2: resume_run failed for run_id=%s: %s",
            approval.run_id, exc,
        )
        return JSONResponse(
            status_code=409,
            content={
                "approval_id": approval_id,
                "reason": "run_terminal",
                "message": "运行已终态，无法恢复。",
            },
        )

    return _build_decision_response(resolved, runs_repo, idempotent=False)


def _build_decision_response(
    approval: ApprovalRecord,
    runs_repo: RunRepository,
    *,
    idempotent: bool,
) -> JSONResponse:
    """Build the 200 response for a successful (or idempotent) decision."""
    return JSONResponse(
        status_code=200,
        content={
            "approval_id": approval.approval_id,
            "run_id": approval.run_id,
            "tool": approval.tool_name,
            "decision": (
                "approve" if approval.status == "approved" else "reject"
            ),
            "status": approval.status,
            "idempotent": idempotent,
            "resume_url": f"/api/v2/chat/resume/{approval.run_id}",
        },
    )


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
    capacity = getattr(runtime, "capacity", None)
    run_capacity_lease = None
    sandbox_capacity_lease = None
    active_sandbox = runs_repo.get_sandbox_lease(run_id)
    if capacity is not None:
        try:
            run_capacity_lease = await capacity.acquire_run()
            if active_sandbox is not None and active_sandbox.status == "active":
                sandbox_capacity_lease = await capacity.acquire_sandbox(
                    run.request_id
                )
        except CapacityExceeded as exc:
            if run_capacity_lease is not None:
                run_capacity_lease.release()
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
            async for frame in _stream_resume_flow(
                run_id=run_id,
                from_seq=from_seq,
                runs_repo=runs_repo,
                telemetry=getattr(runtime, "telemetry", None),
            ):
                yield frame
        finally:
            if sandbox_capacity_lease is not None:
                sandbox = runs_repo.get_sandbox_lease(run_id)
                if sandbox is not None and sandbox.resource_usage:
                    telemetry = getattr(runtime, "telemetry", None)
                    if telemetry is not None:
                        telemetry.record(
                            run_id, resource_usage=sandbox.resource_usage
                        )
                if sandbox is None or sandbox.status != "active":
                    sandbox_capacity_lease.release()
            if run_capacity_lease is not None:
                run_capacity_lease.release()

    return StreamingResponse(
        guarded_resume_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# Bash output cap for the cowork resume flow. Matches the default in
# :meth:`Computer.run_bash`; defined here so tests can monkeypatch the
# cap without re-creating the Computer.
_COWORK_BASH_MAX_OUTPUT_BYTES = 1_048_576  # 1 MiB per stream


async def _resume_cowork_bash_steps(
    *,
    emit: Emit,
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
) -> AsyncGenerator[str, None]:
    """Resume steps for the cowork ``sandbox_bash`` tool (Ticket 13).

    Mirrors the ``generate_pdf`` resume structure but for bash execution:

    1. Effect ledger: ``authorized`` → CAS ``executing`` → ``committed``
       (or ``failed``/``outcome_unknown``).
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
    existing_effect = runs_repo.get_effect(effect_key)

    # Effect 处于 ``executing``：区分并发在飞与已崩溃（spec AC3 的并发盲区）。
    if existing_effect is not None and existing_effect.status == EffectStatus.EXECUTING.value:
        if _effect_executor_is_live(all_events, "sandbox_bash"):
            # 双击恢复：另一执行者仍在飞行。不改动 effect（Ticket 12 H1
            # 同源）；沙箱由调用方在消费完本生成器后检查 effect 状态并保留
            # （见 _stream_resume_flow 中 preserve_sandbox 的赋值）。
            yield emit("error", {
                "code": "effect_in_flight",
                "message": "Bash 正在被其他执行者处理，请稍后重试查看结果。",
            })
            return
        runs_repo.mark_effect_outcome_unknown(
            effect_key,
            reason="恢复时发现 effect 处于 executing 状态",
        )
        yield emit("effect.outcome_unknown", {
            "effect_key": effect_key,
            "tool": "sandbox_bash",
            "message": "Bash 执行结果不确定（检测到崩溃），需要人工检查。",
        })
        yield emit("error", {
            "code": "effect_outcome_unknown",
            "message": "Bash 执行结果不确定，需要人工检查。",
        })
        _mark_terminal_safe(runs_repo, run_id, "error")
        return

    if existing_effect is not None and existing_effect.status == EffectStatus.FAILED.value:
        if not has_tool_call:
            yield emit("tool_call", {
                "tool": "sandbox_bash",
                "args": bash_args,
            })
        yield emit("tool_result", {
            "tool": "sandbox_bash",
            "status": "error",
            "content": f"Bash 执行失败：{existing_effect.failure_reason or '未知错误'}",
        })
        yield emit("error", {
            "code": "bash_execution_failed",
            "message": f"Bash 执行失败：{existing_effect.failure_reason or '未知错误'}",
        })
        _mark_terminal_safe(runs_repo, run_id, "error")
        return

    if existing_effect is not None and existing_effect.status == EffectStatus.OUTCOME_UNKNOWN.value:
        yield emit("effect.outcome_unknown", {
            "effect_key": effect_key,
            "tool": "sandbox_bash",
            "message": "Bash 执行结果不确定，需要人工检查。",
        })
        yield emit("error", {
            "code": "effect_outcome_unknown",
            "message": "Bash 执行结果不确定，需要人工检查。",
        })
        _mark_terminal_safe(runs_repo, run_id, "error")
        return

    # Prior committed — recover result from stored effect (spec AC2).
    recovered_result: dict[str, Any] | None = None
    if existing_effect is not None and existing_effect.status == EffectStatus.COMMITTED.value:
        recovered_result = deserialize_result(existing_effect.result_json)
        if not has_tool_result_ok:
            if not has_tool_call:
                yield emit("tool_call", {
                    "tool": "sandbox_bash",
                    "args": bash_args,
                })
            yield emit("tool_result", {
                "tool": "sandbox_bash",
                "status": "ok",
                "exit_code": recovered_result.get("exit_code", 0),
                "stdout": recovered_result.get("stdout", ""),
                "stderr": recovered_result.get("stderr", ""),
                "duration_ms": recovered_result.get("duration_ms", 0),
                "truncated": recovered_result.get("truncated", False),
                "content": recovered_result.get(
                    "stdout", ""
                ) or recovered_result.get("content", ""),
            })
            has_tool_result_ok = True

    if not has_tool_result_ok:
        # Insert effect row in ``authorized`` (spec AC1).
        if existing_effect is None:
            try:
                existing_effect = runs_repo.insert_effect(
                    effect_key=effect_key,
                    run_id=run_id,
                    conversation_id=run.conversation_id,
                    tool_name="sandbox_bash",
                    args_canonical=resolved.args_canonical,
                    external_idempotency_key=external_idempotency_key,
                )
            except Exception:
                existing_effect = runs_repo.get_effect(effect_key)

        # CAS transition ``authorized`` → ``executing`` (spec AC7).
        _, won = runs_repo.transition_effect_to_executing(effect_key)
        if not won:
            current = runs_repo.get_effect(effect_key)
            if current is not None and current.status == EffectStatus.COMMITTED.value:
                stored = deserialize_result(current.result_json)
                yield emit("tool_result", {
                    "tool": "sandbox_bash",
                    "status": "ok",
                    "exit_code": stored.get("exit_code", 0),
                    "stdout": stored.get("stdout", ""),
                    "stderr": stored.get("stderr", ""),
                    "duration_ms": stored.get("duration_ms", 0),
                    "truncated": stored.get("truncated", False),
                    "content": stored.get("stdout", "") or stored.get("content", ""),
                })
            elif current is not None and current.status in (
                EffectStatus.OUTCOME_UNKNOWN.value,
                EffectStatus.FAILED.value,
            ):
                if current.status == EffectStatus.OUTCOME_UNKNOWN.value:
                    yield emit("effect.outcome_unknown", {
                        "effect_key": effect_key,
                        "tool": "sandbox_bash",
                        "message": "Bash 执行结果不确定，需要人工检查。",
                    })
                    err_code = "effect_outcome_unknown"
                    err_msg = "Bash 执行结果不确定，需要人工检查。"
                else:
                    yield emit("tool_result", {
                        "tool": "sandbox_bash",
                        "status": "error",
                        "content": f"Bash 执行失败：{current.failure_reason or '未知错误'}",
                    })
                    err_code = "bash_execution_failed"
                    err_msg = f"Bash 执行失败：{current.failure_reason or '未知错误'}"
                yield emit("error", {"code": err_code, "message": err_msg})
                _mark_terminal_safe(runs_repo, run_id, "error")
                return
            else:
                # ``executing`` — the other executor is still in flight.
                # Do NOT modify the effect (Ticket 12 review H1).
                yield emit("error", {
                    "code": "effect_in_flight",
                    "message": "Bash 正在被其他执行者处理，请稍后重试查看结果。",
                })
                return
        else:
            # We won the CAS — emit tool_call + execute bash.
            yield emit("tool_call", {
                "tool": "sandbox_bash",
                "args": bash_args,
            })

            try:
                bash_result = computer.run_bash(
                    workspace_path,
                    bash_command,
                    timeout_seconds=30.0,
                    max_output_bytes=_COWORK_BASH_MAX_OUTPUT_BYTES,
                )
            except SandboxTimeoutError as exc:
                runs_repo.mark_effect_failed(effect_key, reason=str(exc))
                yield emit("tool_result", {
                    "tool": "sandbox_bash",
                    "status": "timeout",
                    "content": f"Bash 执行超时：{exc}",
                })
                yield emit("error", {
                    "code": "sandbox_timeout",
                    "message": f"沙箱 Bash 命令超时：{exc}",
                })
                _mark_terminal_safe(runs_repo, run_id, "error")
                return
            except ComputerError as exc:
                runs_repo.mark_effect_failed(effect_key, reason=str(exc))
                yield emit("tool_result", {
                    "tool": "sandbox_bash",
                    "status": "error",
                    "content": f"Bash 执行失败：{exc}",
                })
                yield emit("error", {
                    "code": "bash_execution_failed",
                    "message": f"Bash 执行失败：{exc}",
                })
                _mark_terminal_safe(runs_repo, run_id, "error")
                return

            # Success — commit the effect with the result (spec AC2).
            result_json = serialize_result({
                "exit_code": bash_result.exit_code,
                "stdout": bash_result.stdout,
                "stderr": bash_result.stderr,
                "duration_ms": bash_result.duration_ms,
                "truncated": bash_result.truncated,
            })
            try:
                runs_repo.commit_effect(effect_key, result_json)
            except Exception:
                logger.exception(
                    "chat_v2: commit_effect failed for sandbox_bash run_id=%s; "
                    "marking outcome_unknown",
                    run_id,
                )
                runs_repo.mark_effect_outcome_unknown(
                    effect_key,
                    reason="Bash 执行成功但结果记账失败",
                )
                yield emit("effect.outcome_unknown", {
                    "effect_key": effect_key,
                    "tool": "sandbox_bash",
                    "message": "Bash 执行成功但结果记账失败，需要人工检查。",
                })
                yield emit("error", {
                    "code": "effect_outcome_unknown",
                    "message": "Bash 执行成功但结果记账失败，需要人工检查。",
                })
                _mark_terminal_safe(runs_repo, run_id, "error")
                return

            yield emit("tool_result", {
                "tool": "sandbox_bash",
                "status": "ok",
                "exit_code": bash_result.exit_code,
                "stdout": bash_result.stdout,
                "stderr": bash_result.stderr,
                "duration_ms": bash_result.duration_ms,
                "truncated": bash_result.truncated,
                "content": bash_result.stdout,
            })
            # 供 Step 3 的 text 回复使用：fresh 执行路径此前从不给
            # recovered_result 赋值，导致回复里的 stdout 恒为空。
            recovered_result = {
                "exit_code": bash_result.exit_code,
                "stdout": bash_result.stdout,
                "stderr": bash_result.stderr,
                "duration_ms": bash_result.duration_ms,
                "truncated": bash_result.truncated,
            }
    else:
        # Tool already executed in a prior resume — recover the result
        # from the stored tool_result event for the text response.
        if recovered_result is None:
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
            export = _export_artifact_to_host(
                computer=computer,
                workspace_path=workspace_path,
                relative_path=_COWORK_STUB_FILENAME,
                declared_mime="text/plain",
                runs_repo=runs_repo,
                run_id=run_id,
                conversation_id=run.conversation_id,
            )
        except ArtifactExportError as exc:
            yield emit("error", {
                "code": "artifact_export_failed",
                "message": f"artifact 导出失败：{exc}",
            })
            _mark_terminal_safe(runs_repo, run_id, "error")
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
    if _is_cancelled(runs_repo, run_id):
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
    yield emit("done", {})
    _mark_terminal_safe(runs_repo, run_id, "done")


async def _stream_resume_flow(
    *,
    run_id: str,
    from_seq: int,
    runs_repo: RunRepository,
    telemetry: Any = None,
) -> AsyncGenerator[str, None]:
    """Produce the resume SSE stream — spec AC4, AC5, AC7.

    Replay stored events > ``from_seq``, then (if the run is ``running``
    and has a resolved approval) emit the continuation. The continuation
    reuses the SAME sandbox lease (AC4) and fails closed if the lease is
    missing (AC7).

    Idempotency (spec AC5): before emitting any new event, the flow
    inspects already-persisted events from prior resume attempts. If a
    step was already completed (e.g. ``tool_result`` with ``status=ok``
    exists), it is skipped — the tool is NOT re-executed. This makes the
    resume safe against:

    * Network drops mid-stream — the client reconnects and the flow
      picks up from the last persisted event.
    * Duplicate resume clicks — the second SSE connection sees the
      first attempt's persisted events and skips already-done steps.
    * Process restart — the same logic applies; persisted events are
      the source of truth for "what already happened".

    The flow is therefore event-sourced: each step checks the persisted
    event log before executing, and only emits + persists new events for
    steps that have not yet run.
    """
    telemetry_started = time.perf_counter()
    run = runs_repo.get_run(run_id)
    if run is None:
        yield _sse_frame(EventEnvelope(
            version="2", sequence=1, type="error",
            payload={"code": "run_not_found", "message": f"运行 {run_id} 不存在。"},
        ))
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
        yield _sse_frame(envelope)

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

    def emit(event_type: str, payload: dict[str, Any]) -> str:
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
                "chat_v2: persist_event failed during resume for run_id=%s seq=%d",
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
                    "chat_v2: telemetry update failed during resume for run_id=%s",
                    run_id,
                )
        return _sse_frame(envelope)

    # Look up the resolved approval that authorized this resume.
    approvals = runs_repo.list_approvals_for_run(run_id)
    resolved = next((a for a in approvals if a.is_terminal), None)
    if resolved is None:
        yield emit("error", {
            "code": "no_resolved_approval",
            "message": "运行已恢复但未找到已解决的审批。",
        })
        _mark_terminal_safe(runs_repo, run_id, "error")
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
        yield emit("done", {
            "code": "rejected",
            "message": "用户已拒绝操作，运行结束。",
        })
        _mark_terminal_safe(runs_repo, run_id, "done")
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
        yield emit("error", {
            "code": "sandbox_lease_lost",
            "message": (
                "沙箱租赁已丢失或销毁，无法安全恢复运行（spec AC7）。"
                "请重新发起请求。"
            ),
        })
        _mark_terminal_safe(runs_repo, run_id, "error")
        return

    # Reconstruct the Computer bound to the leased workspace.
    # Ticket 11: route through ``_build_artifact_computer`` (not
    # ``create_computer`` directly) so test failure injections via
    # monkeypatching ``_build_artifact_computer`` apply to the resume
    # flow too.
    try:
        computer = _build_artifact_computer(lease.sandbox_type)
    except ValueError as exc:
        yield emit("error", {
            "code": "sandbox_type_invalid",
            "message": f"沙箱类型无效：{exc}",
        })
        _mark_terminal_safe(runs_repo, run_id, "error")
        return

    workspace_path = lease.workspace_path
    preserve_sandbox = False

    # Parse the approved args from the approval record (spec AC5: the
    # tool must execute with the EXACT args the user approved; hardcoded
    # args would bypass the canonical-args tampering check).
    try:
        import json as _json
        approved_args = _json.loads(resolved.args_canonical) if resolved.args_canonical else {}
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
            async for frame in _resume_cowork_bash_steps(
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
                yield frame
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
        # Ticket 12: effect ledger integration. The effect row is the
        # authoritative source of truth for tool execution state; the
        # event log check (``has_tool_result_ok``) is a secondary guard.
        #
        # Spec AC1: insert effect row in ``authorized`` BEFORE execution.
        # Spec AC2: ``committed`` → return stored result without re-exec.
        # Spec AC3: ``executing`` on resume → crash → ``outcome_unknown``.
        # Spec AC7: CAS ``authorized`` → ``executing`` — one executor wins.
        effect_key = compute_effect_key(
            run_id=run_id,
            tool_name="generate_pdf",
            args=approved_args,
        )
        external_idempotency_key = compute_external_idempotency_key(
            effect_key=effect_key,
            tool_name="generate_pdf",
        )
        existing_effect = runs_repo.get_effect(effect_key)

        # Effect 处于 ``executing``：区分并发在飞与已崩溃（spec AC3 的
        # 并发盲区，与 sandbox_bash 入口检查同源）。
        if existing_effect is not None and existing_effect.status == EffectStatus.EXECUTING.value:
            if _effect_executor_is_live(all_events, "generate_pdf"):
                # 双击恢复：另一执行者仍在飞行。不改动 effect，保留沙箱。
                yield emit("error", {
                    "code": "effect_in_flight",
                    "message": "工具正在被其他执行者处理，请稍后重试查看结果。",
                })
                preserve_sandbox = True
                return
            # 崩溃语义：副作用可能已完成也可能没有，转 outcome_unknown
            # 并终止 —— 不自动重试（spec AC3）。
            runs_repo.mark_effect_outcome_unknown(
                effect_key,
                reason="恢复时发现 effect 处于 executing 状态",
            )
            yield emit("effect.outcome_unknown", {
                "effect_key": effect_key,
                "tool": "generate_pdf",
                "message": "工具执行结果不确定（检测到崩溃），需要人工检查。",
            })
            yield emit("error", {
                "code": "effect_outcome_unknown",
                "message": "工具执行结果不确定，需要人工检查。",
            })
            _mark_terminal_safe(runs_repo, run_id, "error")
            return

        # Prior terminal failure — don't re-execute (spec AC3).
        if existing_effect is not None and existing_effect.status == EffectStatus.FAILED.value:
            yield emit("tool_result", {
                "tool": "generate_pdf",
                "status": "error",
                "content": f"PDF 生成失败：{existing_effect.failure_reason or '未知错误'}",
            })
            yield emit("error", {
                "code": "artifact_generation_failed",
                "message": f"artifact 生成失败：{existing_effect.failure_reason or '未知错误'}",
            })
            _mark_terminal_safe(runs_repo, run_id, "error")
            return

        # Prior outcome_unknown — needs human review, terminate.
        if existing_effect is not None and existing_effect.status == EffectStatus.OUTCOME_UNKNOWN.value:
            yield emit("effect.outcome_unknown", {
                "effect_key": effect_key,
                "tool": "generate_pdf",
                "message": "工具执行结果不确定，需要人工检查。",
            })
            yield emit("error", {
                "code": "effect_outcome_unknown",
                "message": "工具执行结果不确定，需要人工检查。",
            })
            _mark_terminal_safe(runs_repo, run_id, "error")
            return

        # Prior committed — recover path from stored result (spec AC2:
        # 把已有成功结果作为唯一结果返回). If the tool_result event
        # wasn't persisted before the crash, re-emit it from the stored
        # effect result so the client sees a complete event log.
        recovered_path_from_effect: str | None = None
        if existing_effect is not None and existing_effect.status == EffectStatus.COMMITTED.value:
            stored_result = deserialize_result(existing_effect.result_json)
            recovered_path_from_effect = stored_result.get("path", artifact_filename)
            if not has_tool_result_ok:
                if not has_tool_call:
                    yield emit("tool_call", {
                        "tool": "generate_pdf",
                        "args": {"filename": artifact_filename},
                    })
                yield emit("tool_result", {
                    "tool": "generate_pdf",
                    "status": "ok",
                    "content": stored_result.get(
                        "content", f"已在沙箱中生成 {recovered_path_from_effect}"
                    ),
                    "path": recovered_path_from_effect,
                })
                has_tool_result_ok = True

        # AC5: re-executing the tool would violate "重复点击...不会再次
        # 执行工具". If the prior resume already produced a successful
        # tool_result, we skip the tool call and reuse the stored result.
        if not has_tool_result_ok:
            # Insert effect row in ``authorized`` (spec AC1). If it
            # already exists (duplicate resume click), the UNIQUE
            # constraint raises — we fetch the existing row instead.
            if existing_effect is None:
                try:
                    existing_effect = runs_repo.insert_effect(
                        effect_key=effect_key,
                        run_id=run_id,
                        conversation_id=run.conversation_id,
                        tool_name="generate_pdf",
                        args_canonical=resolved.args_canonical,
                        external_idempotency_key=external_idempotency_key,
                    )
                except Exception:
                    existing_effect = runs_repo.get_effect(effect_key)

            # CAS transition ``authorized`` → ``executing`` (spec AC7).
            # Only one concurrent caller wins; the loser must NOT re-execute.
            _, won = runs_repo.transition_effect_to_executing(effect_key)
            if not won:
                # Another executor won. Re-read the effect and handle
                # its new state. Valid new states: ``executing`` (still
                # in-flight) or terminal (committed/failed/outcome_unknown).
                current = runs_repo.get_effect(effect_key)
                if current is not None and current.status == EffectStatus.COMMITTED.value:
                    stored = deserialize_result(current.result_json)
                    relative_path = stored.get("path", artifact_filename)
                    yield emit("tool_result", {
                        "tool": "generate_pdf",
                        "status": "ok",
                        "content": stored.get(
                            "content", f"已在沙箱中生成 {relative_path}"
                        ),
                        "path": relative_path,
                    })
                elif current is not None and current.status in (
                    EffectStatus.OUTCOME_UNKNOWN.value,
                    EffectStatus.FAILED.value,
                ):
                    if current.status == EffectStatus.OUTCOME_UNKNOWN.value:
                        yield emit("effect.outcome_unknown", {
                            "effect_key": effect_key,
                            "tool": "generate_pdf",
                            "message": "工具执行结果不确定，需要人工检查。",
                        })
                        err_code = "effect_outcome_unknown"
                        err_msg = "工具执行结果不确定，需要人工检查。"
                    else:
                        yield emit("tool_result", {
                            "tool": "generate_pdf",
                            "status": "error",
                            "content": f"PDF 生成失败：{current.failure_reason or '未知错误'}",
                        })
                        err_code = "artifact_generation_failed"
                        err_msg = f"artifact 生成失败：{current.failure_reason or '未知错误'}"
                    yield emit("error", {"code": err_code, "message": err_msg})
                    _mark_terminal_safe(runs_repo, run_id, "error")
                    return
                else:
                    # ``executing`` — the other executor is still in
                    # flight. We MUST NOT modify the effect state (the
                    # winner may still commit successfully). Terminating
                    # this resume stream without touching the effect or
                    # the run; the user can retry later to see the
                    # winner's final result (spec AC3 crash semantics
                    # only apply when the prior executor is actually
                    # dead, not merely in-flight on another request).
                    # Ticket 12 review H1: previously this branch called
                    # ``mark_effect_outcome_unknown``, which raced with
                    # the winner's ``commit_effect`` and produced
                    # inconsistent state.
                    yield emit("error", {
                        "code": "effect_in_flight",
                        "message": "工具正在被其他执行者处理，请稍后重试查看结果。",
                    })
                    # 胜者仍在该沙箱内执行 —— 保留沙箱与租约，仅终止本
                    # 流（双击恢复互删修复）。
                    preserve_sandbox = True
                    return
            else:
                # We won the CAS — execute the tool.
                user_message = ""
                for msg in snapshot.messages if snapshot else []:
                    if msg.get("role") == "user":
                        user_message = msg.get("content", "")
                        break

                plan_content = _stub_artifact_plan_content(user_message)
                yield emit("tool_call", {
                    "tool": "generate_pdf",
                    "args": {"filename": artifact_filename},
                })

                try:
                    relative_path = computer.generate_pdf(
                        workspace_path, plan_content, artifact_filename
                    )
                except SandboxTimeoutError as exc:
                    runs_repo.mark_effect_failed(effect_key, reason=str(exc))
                    yield emit("tool_result", {
                        "tool": "generate_pdf",
                        "status": "timeout",
                        "content": f"PDF 生成超时：{exc}",
                    })
                    yield emit("error", {
                        "code": "sandbox_timeout",
                        "message": f"沙箱命令超时：{exc}",
                    })
                    _mark_terminal_safe(runs_repo, run_id, "error")
                    return
                except ComputerError as exc:
                    runs_repo.mark_effect_failed(effect_key, reason=str(exc))
                    yield emit("tool_result", {
                        "tool": "generate_pdf",
                        "status": "error",
                        "content": f"PDF 生成失败：{exc}",
                    })
                    yield emit("error", {
                        "code": "artifact_generation_failed",
                        "message": f"artifact 生成失败：{exc}",
                    })
                    _mark_terminal_safe(runs_repo, run_id, "error")
                    return

                # Success — commit the effect with the result (spec AC2).
                # Spec AC4: if commit fails (external success but local
                # bookkeeping failure), mark outcome_unknown so a retry
                # doesn't re-execute the now-completed side effect.
                result_json = serialize_result({
                    "content": f"已在沙箱中生成 {relative_path}",
                    "path": relative_path,
                })
                try:
                    runs_repo.commit_effect(effect_key, result_json)
                except Exception:
                    logger.exception(
                        "chat_v2: commit_effect failed for run_id=%s; "
                        "marking outcome_unknown",
                        run_id,
                    )
                    runs_repo.mark_effect_outcome_unknown(
                        effect_key,
                        reason="工具执行成功但结果记账失败",
                    )
                    yield emit("effect.outcome_unknown", {
                        "effect_key": effect_key,
                        "tool": "generate_pdf",
                        "message": "工具执行成功但结果记账失败，需要人工检查。",
                    })
                    yield emit("error", {
                        "code": "effect_outcome_unknown",
                        "message": "工具执行成功但结果记账失败，需要人工检查。",
                    })
                    _mark_terminal_safe(runs_repo, run_id, "error")
                    return

                yield emit("tool_result", {
                    "tool": "generate_pdf",
                    "status": "ok",
                    "content": f"已在沙箱中生成 {relative_path}",
                    "path": relative_path,
                })
        else:
            # Tool already executed in a prior resume — recover the
            # relative_path from the stored tool_result payload OR the
            # committed effect result so the artifact_candidate event
            # can carry the correct filename.
            if recovered_path_from_effect is not None:
                relative_path = recovered_path_from_effect
            else:
                tool_result_evt = next(
                    evt for evt in all_events
                    if evt["type"] == "tool_result"
                    and evt["payload"].get("tool") == "generate_pdf"
                    and evt["payload"].get("status") == "ok"
                )
                relative_path = tool_result_evt["payload"].get("path", artifact_filename)

        # Step 3: cancel check before artifact_candidate.
        if _is_cancelled(runs_repo, run_id):
            yield emit("cancelled", {
                "code": "cancelled",
                "message": "运行已被取消。",
            })
            return

        # Step 4: artifact_candidate + host export (skip if artifact.available
        # already persisted — the export is a side effect on the host).
        if not has_artifact_available:
            yield emit("artifact_candidate", {
                "filename": relative_path,
                "mime_type": "application/pdf",
                "sandbox_type": computer.sandbox_type,
                "message": "已生成徒步计划 PDF，等待宿主侧导出。",
            })

            try:
                export = _export_artifact_to_host(
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
                yield emit("error", {
                    "code": "artifact_evidence_persistence_failed",
                    "message": (
                        f"artifact 证据持久化失败，sandbox 已保留以便排查：{exc}。"
                        f"请联系运维处理 run_id={run_id} 的沙箱。"
                    ),
                })
                _mark_terminal_safe(runs_repo, run_id, "error")
                return
            except ArtifactExportError as exc:
                yield emit("error", {
                    "code": "artifact_export_failed",
                    "message": f"artifact 导出失败：{exc}",
                })
                _mark_terminal_safe(runs_repo, run_id, "error")
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
        yield emit("done", {})
        _mark_terminal_safe(runs_repo, run_id, "done")

    except TerminalEventAlreadyEmitted:
        logger.error(
            "chat_v2: attempted to emit after terminal during resume for run_id=%s",
            run_id,
        )
        return
    except ArtifactEvidencePersistenceError as exc:
        preserve_sandbox = True
        yield emit("error", {
            "code": "artifact_evidence_persistence_failed",
            "message": (
                f"artifact 证据持久化失败，sandbox 已保留以便排查：{exc}。"
                f"请联系运维处理 run_id={run_id} 的沙箱。"
            ),
        })
        _mark_terminal_safe(runs_repo, run_id, "error")
        return
    except Exception as exc:
        logger.exception(
            "chat_v2: unexpected error during resume for run_id=%s", run_id
        )
        if not stream.terminal_emitted:
            yield emit("error", {
                "code": "internal_error",
                "message": "恢复运行时发生内部错误。",
            })
        _mark_terminal_safe(runs_repo, run_id, "error")
        return
    finally:
        if preserve_sandbox:
            logger.warning(
                "chat_v2: preserving sandbox for run_id=%s during resume "
                "(evidence persistence failed); lease is intentionally leaked",
                run_id,
            )
        else:
            _cleanup_sandbox(computer, workspace_path, runs_repo, run_id)
