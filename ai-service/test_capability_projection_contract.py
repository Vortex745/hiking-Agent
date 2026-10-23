"""BCDEF ticket 10 — capability/MCP/risk/effect policy projection.

Seam: the pure projection built from a frozen execution policy plus the
orchestration boundary. No real LLM, network or sandbox: kernel turns,
computers, exports and cleanup are injected fakes over tmp SQLite.

The projection is asserted to be deterministic, complete for every tool
source (builtin / skills / sandbox / MCP), independent of registration
order, bound to the run policy hash, and fail-closed on incomplete input.
"""

from __future__ import annotations

import asyncio
import copy
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from agent_v2.capability_projection import (
    CapabilityProjectionError,
    build_capability_projection,
    legacy_capability_projection,
    mcp_capability_id,
)
from agent_v2.effects import compute_effect_identity
from agent_v2.execution_contract import ExecutionErrorCode, policy_hash
from agent_v2.execution_policy import build_execution_policy
from agent_v2.permissions import PermissionDecision, RiskLevel
from agent_v2.profiles import resolve_profile
from agent_v2.run_orchestration import (
    EFFECT_COPY_GENERATE_PDF,
    EffectDriveResult,
    drive_effect,
)
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.runs import RunRepository


def _settings(**overrides: Any) -> SimpleNamespace:
    base = {
        "agent_v2_enabled": True,
        "openai_api_key": "sk-test",
        "openai_model": "test-model",
        "agent_v2_cowork_policy": "disabled",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _policy(profile_id: str = "route_plan", **overrides: Any) -> dict[str, Any]:
    return copy.deepcopy(
        build_execution_policy(resolve_profile(profile_id), _settings(**overrides))
    )


def _projection(profile_id: str = "route_plan", *, run_id: str = "run-1"):
    policy = _policy(profile_id)
    return build_capability_projection(policy, run_id=run_id)


# ---------------------------------------------------------------------------
# Deterministic, complete inventory
# ---------------------------------------------------------------------------


def test_projection_is_deterministic_and_bound_to_policy_hash():
    policy = _policy("route_plan")
    policy["capability"]["skills"] = [
        {"name": "pdf-report", "risk": "medium"},
    ]
    first = build_capability_projection(policy, run_id="run-1")
    second = build_capability_projection(copy.deepcopy(policy), run_id="run-1")

    assert first.policy_hash == policy_hash(policy)
    assert first.projection_hash == second.projection_hash
    assert first.to_payload() == second.to_payload()
    assert first.summary()["policy_hash"] == first.policy_hash
    assert first.summary()["projection_hash"] == first.projection_hash


def test_projection_inventory_covers_builtin_skills_mcp_and_sandbox():
    route_policy = _policy("route_plan")
    route_policy["capability"]["skills"] = [
        {"name": "pdf-report", "risk": "medium"},
    ]
    route = build_capability_projection(route_policy, run_id="run-route")
    ids = {tool.capability_id for tool in route.tools}

    assert "builtin:hiking_knowledge_search" in ids
    assert "builtin:web_search" in ids
    assert "mcp:amap_weather:get_current_weather" in ids
    assert "mcp:amap_map:plan_route" in ids
    assert "skill:pdf-report" in ids
    assert route.tool("mcp__amap_map__plan_route").source == "mcp"
    assert route.tool("hiking_knowledge_search").source == "builtin"

    artifact = _projection("artifact")
    sandbox_ids = {tool.capability_id for tool in artifact.tools}
    assert "sandbox:generate_pdf" in sandbox_ids
    assert "sandbox:sandbox_read" in sandbox_ids
    assert artifact.tool("generate_pdf").source == "sandbox"
    assert artifact.tool("generate_pdf").requires_approval is True
    assert artifact.tool("sandbox_read").requires_approval is False

    summary = route.summary()
    assert summary["capability_count"] == len(route.tools) == len(ids)
    assert summary["capability_ids"] == sorted(ids)
    assert {service["server_id"] for service in summary["mcp_services"]} == {
        "amap_map",
        "amap_weather",
    }
    assert summary["risk_policy"]["approval_threshold"] == "high"
    assert summary["approval_policy"]["requires_approval_at_or_above"] == "high"
    # High-risk tool entries are enumerated under the approval policy.
    assert summary["approval_policy"]["capability_ids"] == [
        tool.capability_id for tool in route.tools if tool.requires_approval
    ]


def test_capability_ids_do_not_depend_on_registration_order():
    policy = _policy("route_plan")
    reordered = copy.deepcopy(policy)
    reordered["capability"]["allowed_tools"].reverse()
    reordered["capability"]["allowed_mcp_servers"].reverse()
    reordered["capability"]["mcp_servers"] = dict(
        reversed(list(reordered["capability"]["mcp_servers"].items()))
    )

    first = build_capability_projection(policy, run_id="run-1")
    second = build_capability_projection(reordered, run_id="run-1")

    assert [tool.capability_id for tool in first.tools] == [
        tool.capability_id for tool in second.tools
    ]
    assert [service.server_id for service in first.mcp_services] == [
        service.server_id for service in second.mcp_services
    ]
    assert first.to_payload()["capabilities"] == second.to_payload()["capabilities"]
    # Policy identity changed with the reordered input, projection identity is
    # explicitly bound to it rather than silently ignoring the difference.
    assert first.policy_hash != second.policy_hash
    assert first.projection_hash != second.projection_hash


# ---------------------------------------------------------------------------
# Authorization: frozen risk + approval policy
# ---------------------------------------------------------------------------


def test_authorization_uses_frozen_risk_and_threshold():
    projection = _projection("route_plan")

    allow = projection.authorize("web_search", {"query": "x"})
    assert allow.decision is PermissionDecision.ALLOW
    assert allow.risk is RiskLevel.MEDIUM
    assert allow.reason == "low_risk_allowed"

    high = projection.authorize("web_scraping", {"url": "https://example.com"})
    assert high.decision is PermissionDecision.NEEDS_APPROVAL
    assert high.risk is RiskLevel.HIGH
    assert high.reason == "high_risk_requires_approval"

    granted = projection.authorize("web_scraping", {"url": "https://example.com"}, approval_granted=True)
    assert granted.decision is PermissionDecision.ALLOW
    assert granted.reason == "approval_granted"

    unknown = projection.authorize("Bash", {"command": "id"})
    assert unknown.decision is PermissionDecision.DENY
    assert unknown.reason == "unknown_tool"

    mcp_ok = projection.authorize("mcp__amap_map__plan_route", {"destination": "杭州"})
    assert mcp_ok.decision is PermissionDecision.ALLOW
    assert mcp_ok.capability_id == "mcp:amap_map:plan_route"

    mcp_rogue = projection.authorize("mcp__amap_map__shell_exec", {})
    assert mcp_rogue.decision is PermissionDecision.DENY
    assert mcp_rogue.reason == "unauthorized_mcp_tool"

    mcp_server = projection.authorize("mcp__evil__read_secrets", {})
    assert mcp_server.decision is PermissionDecision.DENY
    assert mcp_server.reason == "unknown_mcp_server"


def test_legacy_projection_from_profile_keeps_same_authorization_semantics():
    projection = legacy_capability_projection(
        resolve_profile("route_plan"), run_id="run-legacy"
    )
    assert projection.authorize("hiking_knowledge_search").decision is (
        PermissionDecision.ALLOW
    )
    assert projection.authorize("mcp__amap_map__plan_route").decision is (
        PermissionDecision.ALLOW
    )
    assert projection.authorize("mcp__amap_map__shell_exec").decision is (
        PermissionDecision.DENY
    )


# ---------------------------------------------------------------------------
# Lazy MCP discovery
# ---------------------------------------------------------------------------


def test_lazy_mcp_discovery_only_grants_projected_tools():
    projection = _projection("route_plan")

    result = projection.discover_mcp_tools(
        "amap_map", ["plan_route", "shell_exec", "geocode"]
    )

    discovered_ids = {entry["capability_id"] for entry in result.discovered}
    assert discovered_ids == {
        mcp_capability_id("amap_map", "plan_route"),
        mcp_capability_id("amap_map", "geocode"),
    }
    denied = {entry["tool"]: entry for entry in result.denied}
    assert denied["mcp__amap_map__shell_exec"]["reason"] == "not_projected"
    assert denied["mcp__amap_map__shell_exec"]["capability_id"] == (
        mcp_capability_id("amap_map", "shell_exec")
    )
    assert result.policy_hash == projection.policy_hash
    assert result.projection_hash == projection.projection_hash

    # A tool is denied until it exists in the frozen projection, even if
    # the live server advertises it.
    assert projection.authorize("mcp__amap_map__shell_exec", {}).decision is (
        PermissionDecision.DENY
    )
    assert projection.authorize("mcp__amap_map__plan_route", {}).decision is (
        PermissionDecision.ALLOW
    )

    unknown_server = projection.discover_mcp_tools("not_registered", ["x"])
    assert unknown_server.discovered == ()
    assert unknown_server.denied[0]["reason"] == "unknown_mcp_server"


def test_describe_exposes_stable_identity_and_deferred_schema():
    projection = _projection("route_plan")
    described = projection.describe("mcp__amap_map__plan_route")
    assert described["projected"] is True
    assert described["capability_id"] == "mcp:amap_map:plan_route"
    assert described["schema_summary"]["deferred"] is True
    assert described["policy_hash"] == projection.policy_hash

    missing = projection.describe("mcp__amap_map__new_tool")
    assert missing["projected"] is False
    assert missing["capability_id"] == "mcp:amap_map:new_tool"
    assert missing["policy_hash"] == projection.policy_hash


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


def test_missing_risk_for_allowed_tool_fails_closed():
    policy = _policy("chat")
    del policy["effect_policy"]["tool_risk"]["gear_checklist"]

    with pytest.raises(CapabilityProjectionError) as excinfo:
        build_capability_projection(policy, run_id="run-1")
    error = excinfo.value
    assert error.code is ExecutionErrorCode.CAPABILITY_PROJECTION_FAILED
    assert error.code.value == "capability_projection_failed"
    assert error.reason == "missing_risk"
    assert error.recoverable is False
    assert error.to_payload()["code"] == "capability_projection_failed"


def test_missing_mcp_service_detail_fails_closed():
    policy = _policy("route_plan")
    del policy["capability"]["mcp_servers"]["amap_map"]

    with pytest.raises(CapabilityProjectionError) as excinfo:
        build_capability_projection(policy, run_id="run-1")
    assert excinfo.value.reason == "missing_mcp_service"


def test_malformed_capability_fails_closed():
    policy = _policy("chat")
    policy["capability"]["allowed_tools"] = "gear_checklist"

    with pytest.raises(CapabilityProjectionError) as excinfo:
        build_capability_projection(policy, run_id="run-1")
    assert excinfo.value.reason == "invalid_capability"


def test_wrong_policy_hash_fails_closed():
    policy = _policy("chat")
    with pytest.raises(CapabilityProjectionError) as excinfo:
        build_capability_projection(
            policy, run_id="run-1", policy_hash_value="0" * 64
        )
    assert excinfo.value.code is ExecutionErrorCode.POLICY_HASH_MISMATCH
    assert excinfo.value.code.value == "policy_hash_mismatch"
    assert excinfo.value.recoverable is False


# ---------------------------------------------------------------------------
# Effect ledger seam: risk/approval BEFORE the effect claim
# ---------------------------------------------------------------------------


def _repo(tmp_path: Any) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def _make_run(repo: RunRepository, request_id: str = "req-proj") -> str:
    run_id, _created = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=f"conv-{request_id}",
        profile_id="artifact",
        user_message="m",
    )
    return run_id


def _drive_projection(
    repo: RunRepository,
    run_id: str,
    *,
    projection: Any,
    executor: Any,
    approval_granted: bool = False,
    calls: list[tuple[str, dict[str, Any]]] | None = None,
) -> tuple[EffectDriveResult, list[tuple[str, dict[str, Any]]]]:
    """Drive ONE generate_pdf effect through the ledger under ``projection``."""
    events: list[tuple[str, dict[str, Any]]] = (
        calls if calls is not None else []
    )

    def emit(event_type: str, payload: dict[str, Any]) -> Any:
        events.append((event_type, payload))
        return SimpleNamespace(type=event_type, payload=payload)

    tool_args = {"filename": "p.pdf"}
    identity = compute_effect_identity(
        run_id=run_id, tool_name="generate_pdf", args=tool_args
    )
    outcome = EffectDriveResult()

    async def run() -> None:
        async for _ in drive_effect(
            runs_repo=repo,
            run_id=run_id,
            conversation_id=f"conv-{run_id}",
            copy=EFFECT_COPY_GENERATE_PDF,
            effect_key=identity.effect_id,
            external_idempotency_key=identity.idempotency_key,
            args_canonical=json.dumps(
                tool_args, sort_keys=True, separators=(",", ":")
            ),
            tool_args=tool_args,
            all_events=[],
            has_tool_call=False,
            has_tool_result_ok=False,
            execute=executor,
            emit=emit,
            outcome=outcome,
            projection=projection,
            approval_granted=approval_granted,
        ):
            pass

    asyncio.run(run())
    return outcome, events


def test_projection_denial_stops_drive_effect_before_any_effect_row(tmp_path):
    """Ticket 10 AC: risk/approval is evaluated BEFORE the effect claim.

    A tool the frozen projection does not carry must not create an
    executable effect row, must not reach the executor, and must surface
    the stable ``tool_not_authorized`` boundary error.
    """
    repo = _repo(tmp_path)
    run_id = _make_run(repo)
    policy = _policy("artifact")
    policy["capability"]["allowed_tools"].remove("generate_pdf")
    projection = build_capability_projection(policy, run_id=run_id)

    executed: list[int] = []
    outcome, events = _drive_projection(
        repo,
        run_id,
        projection=projection,
        executor=lambda: executed.append(1) or {"content": "x", "path": "p.pdf"},
    )

    assert executed == []
    assert outcome.status == "denied"
    error = next(p for (t, p) in events if t == "error")
    assert error["code"] == "tool_not_authorized"
    assert error["reason"] == "unknown_tool"
    assert error["recoverable"] is False
    # No executable effect row was created, and the run terminated.
    tool_name = "generate_pdf"
    from agent_v2.effects import compute_effect_key

    assert repo.get_effect(
        compute_effect_key(run_id=run_id, tool_name=tool_name, args={"filename": "p.pdf"})
    ) is None
    assert repo.control.get_run(run_id).status == "error"
    repo.close()


def test_high_risk_without_approval_denied_before_claim(tmp_path):
    """generate_pdf is projected HIGH risk → claim is refused until approval."""
    repo = _repo(tmp_path)
    run_id = _make_run(repo)
    projection = _projection("artifact", run_id=run_id)
    assert projection.tool("generate_pdf").requires_approval is True

    executed: list[int] = []
    outcome, events = _drive_projection(
        repo,
        run_id,
        projection=projection,
        executor=lambda: executed.append(1) or {"content": "x", "path": "p.pdf"},
    )
    assert executed == []
    assert outcome.status == "denied"
    error = next(p for (t, p) in events if t == "error")
    assert error["code"] == "tool_not_authorized"
    assert error["reason"] == "high_risk_requires_approval"
    repo.close()


def test_approved_effect_carries_policy_bound_replay_metadata(tmp_path):
    """AC: effect execution is auditable/replayable with run_id + policy hash."""
    repo = _repo(tmp_path)
    run_id = _make_run(repo)
    projection = _projection("artifact", run_id=run_id)

    executed: list[int] = []

    def executor() -> dict[str, Any]:
        executed.append(1)
        return {"content": "已在沙箱中生成 p.pdf", "path": "p.pdf"}

    outcome, events = _drive_projection(
        repo,
        run_id,
        projection=projection,
        executor=executor,
        approval_granted=True,
    )
    assert executed == [1]
    assert outcome.status == "executed"
    tool_call = next(p for (t, p) in events if t == "tool_call")
    assert tool_call["policy_hash"] == projection.policy_hash
    assert tool_call["projection_hash"] == projection.projection_hash
    assert tool_call["capability_id"] == "sandbox:generate_pdf"
    assert tool_call["run_id"] == run_id
    tool_result = next(
        p for (t, p) in events if t == "tool_result" and p.get("status") == "ok"
    )
    assert tool_result["policy_hash"] == projection.policy_hash
    assert tool_result["capability_id"] == "sandbox:generate_pdf"
    repo.close()


def test_duplicate_live_claim_reports_stable_boundary_code(tmp_path):
    """AC: a second executor on a live claim → stable duplicate effect error.

    The public wire code stays ``effect_in_flight`` (client compatible);
    the additive ``boundary_code`` names the shared contract code
    ``duplicate_effect``. No re-execution, no ledger mutation.
    """
    repo = _repo(tmp_path)
    run_id = _make_run(repo)
    tool_args = {"filename": "p.pdf"}
    identity = compute_effect_identity(
        run_id=run_id, tool_name="generate_pdf", args=tool_args
    )
    repo.insert_effect(
        effect_key=identity.effect_id,
        run_id=run_id,
        conversation_id="conv-x",
        tool_name="generate_pdf",
        args_canonical='{"filename":"p.pdf"}',
        external_idempotency_key=identity.idempotency_key,
    )
    _row, won = repo.claim_effect(
        identity.effect_id, owner="worker-1", claim_token="tok-1", ttl_seconds=60
    )
    assert won

    events: list[tuple[str, dict[str, Any]]] = []

    def emit(event_type: str, payload: dict[str, Any]) -> Any:
        events.append((event_type, payload))
        return SimpleNamespace(type=event_type, payload=payload)

    outcome = EffectDriveResult()
    executed: list[int] = []
    all_events = [{
        "type": "tool_call",
        "payload": {"tool": "generate_pdf"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }]

    async def run() -> None:
        async for _ in drive_effect(
            runs_repo=repo,
            run_id=run_id,
            conversation_id="conv-x",
            copy=EFFECT_COPY_GENERATE_PDF,
            effect_key=identity.effect_id,
            external_idempotency_key=identity.idempotency_key,
            args_canonical='{"filename":"p.pdf"}',
            tool_args=tool_args,
            all_events=all_events,
            has_tool_call=True,
            has_tool_result_ok=False,
            execute=lambda: executed.append(1) or {"content": "x", "path": "p.pdf"},
            emit=emit,
            outcome=outcome,
        ):
            pass

    asyncio.run(run())
    assert executed == []
    assert outcome.status == "in_flight"
    error = next(p for (t, p) in events if t == "error")
    assert error["code"] == "effect_in_flight"
    assert error["boundary_code"] == "duplicate_effect"
    current = repo.get_effect(identity.effect_id)
    assert current.status == "executing"  # untouched: still the owner's claim
    repo.close()


# ---------------------------------------------------------------------------
# Snapshot/replay exposure (additive)
# ---------------------------------------------------------------------------


def test_snapshot_exposes_projection_summary_from_frozen_policy(
    tmp_path, monkeypatch
):
    from agent_v2 import permissions as permissions_module
    from agent_v2.persistence import PolicySnapshotWrite

    repo = _repo(tmp_path)
    run_id, _created = repo.create_run_idempotent(
        request_id="req-snap",
        conversation_id="conv-snap",
        profile_id="route_plan",
        user_message="m",
    )
    policy = _policy("route_plan")
    repo.control.insert_policy_snapshot(
        run_id=run_id,
        policy=PolicySnapshotWrite(
            policy_json=json.dumps(policy, sort_keys=True, separators=(",", ":")),
            policy_hash=policy_hash(policy),
            schema_version=policy["schema_version"],
            source_version=policy["source_version"],
            compatibility_json=json.dumps(policy["compatibility"]),
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    )

    snapshot = repo.get_snapshot(run_id)
    summary = snapshot.capability_projection
    assert summary is not None
    assert summary["policy_hash"] == policy_hash(policy)
    assert "builtin:web_search" in summary["capability_ids"]
    assert "mcp:amap_map:plan_route" in summary["capability_ids"]
    assert summary["capability_count"] == len(summary["capability_ids"])
    assert {s["server_id"] for s in summary["mcp_services"]} == {
        "amap_map",
        "amap_weather",
    }
    assert summary["risk_policy"]["approval_threshold"] == "high"
    assert summary["approval_policy"]["requires_approval_at_or_above"] == "high"
    # Existing snapshot fields stay intact — the summary is additive.
    assert snapshot.policy_hash == policy_hash(policy)
    assert snapshot.status == "running"

    # Current global config drift cannot change what the snapshot reports:
    # the summary is derived from the frozen policy, not the live risk map.
    monkeypatch.setitem(
        permissions_module._TOOL_RISK, "web_search", RiskLevel.HIGH
    )
    assert permissions_module.tool_risk_policy()["web_search"] == "high"
    drifted = repo.get_snapshot(run_id).capability_projection
    assert drifted == summary
    assert drifted["risk_policy"]["tool_risk"]["web_search"] == "medium"
    repo.close()


# ---------------------------------------------------------------------------
# Orchestration seam: admitted run + fake kernel over real SQLite
# ---------------------------------------------------------------------------


class _FakeMemory:
    def memory_enabled(self, subject_id): return True

    def claim_conversation(self, s, c): return True

    def owns_conversation(self, s, c): return True

    def create_candidate(self, s, **kw): return kw


def _collect(gen: Any) -> list[Any]:
    async def run() -> list[Any]:
        return [item async for item in gen]

    return asyncio.run(run())


def _orchestrator(tmp_path, kernel_calls: list[dict[str, Any]], **kwargs: Any):
    from agent_v2.run_orchestration import RunOrchestrator

    async def fake_kernel(**kw: Any) -> tuple[str, list[dict[str, Any]]]:
        kernel_calls.append(kw)
        return kw.get("_result") or ("你好，徒步愉快！", [])

    repo = kwargs.pop("runs_repo", None) or _repo(tmp_path)
    orch = RunOrchestrator(
        settings=_settings(),
        runs_repo=repo,
        memory=_FakeMemory(),
        kernel_turn=fake_kernel,
    )
    return orch, repo


def _req(profile_id: str = "chat", request_id: str = "req-t10"):
    from agent_v2.run_orchestration import ChatFlowRequest

    return ChatFlowRequest(
        message="带什么装备？", request_id=request_id, profile_id=profile_id
    )


def test_admission_fails_closed_when_frozen_policy_is_incomplete(
    tmp_path, monkeypatch
):
    """AC: projection failure is terminal before any model/tool work."""
    import agent_v2.run_orchestration as ro

    original = ro.build_execution_policy

    def broken_policy(profile, settings):
        policy = original(profile, settings)
        # Simulates a snapshot whose risk table cannot cover the surface.
        del policy["effect_policy"]["tool_risk"]["gear_checklist"]
        return policy

    monkeypatch.setattr(ro, "build_execution_policy", broken_policy)
    kernel_calls: list[dict[str, Any]] = []
    orch, repo = _orchestrator(tmp_path, kernel_calls)

    items = _collect(orch.stream_chat(_req(), subject_id=SUBJECT_ID))

    assert kernel_calls == []  # no model call under an incomplete projection
    terminal = items[-1]
    assert terminal.type == "error"
    assert terminal.payload["code"] == "capability_projection_failed"
    assert terminal.payload["reason"] == "missing_risk"
    assert terminal.payload["recoverable"] is False
    run = repo.control.get_run(terminal.run_id)
    assert run.status == "error"
    repo.close()


def test_admitted_run_projects_from_frozen_policy_not_live_config(
    tmp_path, monkeypatch
):
    """AC: allowed_tools is a projection input; live profile drift is inert.

    The frozen snapshot omits ``gear_checklist`` while the CURRENT live
    profile still allows it. The admitted run's tool surface must come
    from the frozen projection — a mid-run global config change can
    neither widen nor silently replace it.
    """
    import agent_v2.run_orchestration as ro

    original = ro.build_execution_policy

    def narrowed_policy(profile, settings):
        policy = original(profile, settings)
        policy["capability"]["allowed_tools"] = [
            name for name in policy["capability"]["allowed_tools"]
            if name != "gear_checklist"
        ]
        return policy

    monkeypatch.setattr(ro, "build_execution_policy", narrowed_policy)
    kernel_calls: list[dict[str, Any]] = []
    orch, repo = _orchestrator(tmp_path, kernel_calls)

    items = _collect(orch.stream_chat(_req(request_id="req-t10-drift"), subject_id=SUBJECT_ID))
    assert items[-1].type == "done"

    # The live profile still exposes the tool — but the run does not.
    assert "gear_checklist" in resolve_profile("chat").allowed_tools
    assert kernel_calls
    mounted = {tool.name for tool in kernel_calls[0]["extra_tools"]}
    assert "gear_checklist" not in mounted
    assert "hiking_knowledge_search" in mounted  # rest of the surface survives
    repo.close()


def test_mcp_discovery_events_are_replayable_and_policy_bound(
    tmp_path, monkeypatch
):
    """AC: MCP discovery evidence carries run_id + policy hash additively."""
    import agent_v2.kernel_tools as kt

    def fake_build(profile, settings, projection=None):
        assert projection is not None, "preflight must use the frozen projection"
        return (
            {
                service.server_id: {"transport": service.transport}
                for service in projection.mcp_services
            },
            [],
        )

    monkeypatch.setattr(kt, "build_profile_mcp_servers", fake_build)
    kernel_calls: list[dict[str, Any]] = []
    orch, repo = _orchestrator(tmp_path, kernel_calls)

    items = _collect(
        orch.stream_chat(
            _req(profile_id="route_plan", request_id="req-t10-mcp"),
            subject_id=SUBJECT_ID,
        )
    )

    run_id = items[0].run_id
    snapshot = repo.get_snapshot(run_id)
    discovery = [
        e.payload
        for e in items
        if e.type == "tool_result"
        and str(e.payload.get("tool", "")).endswith("__discovery")
    ]
    assert {d["server_id"] for d in discovery} == {"amap_map", "amap_weather"}
    for entry in discovery:
        assert entry["run_id"] == run_id
        assert entry["policy_hash"] == snapshot.policy_hash
        assert entry["projection_hash"]
        assert all("mcp:" in cid for cid in entry["capability_ids"])
    amap = next(d for d in discovery if d["server_id"] == "amap_map")
    assert "mcp:amap_map:plan_route" in amap["capability_ids"]
    assert "mcp__amap_map__plan_route" in amap["discovered_tools"]
    # Envelope shape unchanged: additive keys inside existing event types.
    assert {e.type for e in items} <= {
        "run_started", "progress", "heartbeat", "text", "tool_result",
        "tool_call", "done", "error",
    }
    repo.close()


SUBJECT_ID = "sub-t10"
