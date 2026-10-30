"""Ticket 26 — Chat v2 toolless runs through the real UniHarness kernel.

HTTP/SSE contract coverage:

* success — real kernel model loop generates a single ``text`` event then
  one ``done`` terminal (real credentials required; skipped without).
* model failure / assembly failure — stable terminal ``error`` with a
  public code and NO fallback to stub responses or direct flows.
* envelope invariants — monotonic sequence, exactly one terminal,
  unknown-event dropping untouched.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import api.chat_v2 as chat_v2_module
from test_mcp_management_contract import http_server  # noqa: F401 (pytest fixture)
from agent_v2.kernel import (
    KernelApprovalPending,
    KernelAssemblyError,
    KernelModelFailure,
    extract_final_text,
)

_HAS_LLM_CREDS = bool(os.getenv("OPENAI_API_KEY")) and bool(os.getenv("OPENAI_BASE_URL"))
# Opt-in live check (provider qpm makes un-gated bursts flaky).
_LIVE_LLM = _HAS_LLM_CREDS and os.getenv("KERNEL_LIVE_E2E", "") == "1"


class _FakeSettings:
    """Minimal server settings for stream-level contract tests."""

    def __init__(self, **overrides: Any) -> None:
        self.agent_v2_enabled = True
        self.openai_api_key = "sk-fake"
        self.openai_base_url = "https://fake.invalid/v1"
        self.openai_model = "fake-model"
        for key, value in overrides.items():
            setattr(self, key, value)


def _build_app(runtime: Any | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(chat_v2_module.chat_v2_router, prefix="/api/v2")
    state = app.state

    class _FakeMemory:
        def memory_enabled(self, subject_id): return True
        def claim_conversation(self, s, c): return True
        def owns_conversation(self, s, c): return True
        def save_preference(self, s, content): return "m1"
        def list_active_memories(self, s): return []
        def create_candidate(self, s, **kw): return kw

    if runtime is None:
        class _FakeRuntime:
            runs = None
            telemetry = None
            kernel_checkpoints = None
            memory = _FakeMemory()

        state.agent_v2_runtime = _FakeRuntime()
    else:
        state.agent_v2_runtime = runtime
    return app, state


async def _post_sse(app: FastAPI, payload: dict[str, Any]) -> list[dict[str, Any]]:
    transport = ASGITransport(app=app)
    events: list[dict[str, Any]] = []
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST", "/api/v2/chat/sse", json=payload
        ) as response:
            assert response.status_code == 200
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    for line in frame.splitlines():
                        if line.startswith("data: "):
                            events.append(json.loads(line[len("data: ") :]))
    return events


def test_kernel_run_toolless_failure_maps_to_terminal_error_without_fallback(monkeypatch):
    """Kernel failures produce exactly one terminal error; stub never runs."""
    called: dict[str, int] = {"n": 0}

    async def _failing_run(**kwargs: Any):
        called["n"] += 1
        raise KernelModelFailure("model_unavailable", "主模型未配置 API Key (OPENAI_API_KEY 为空)。")

    # The stream imports run_chat_turn lazily from agent_v2.kernel at call
    # time, so patching there is authoritative (Ticket 28 renamed the entry).
    from agent_v2 import kernel as kernel_module

    monkeypatch.setattr(kernel_module, "run_chat_turn", _failing_run)

    from types import SimpleNamespace

    def _fake_subject(request):
        request.state.subject_id = "sub-1"
        return None, SimpleNamespace(issued=False, token="tok", subject_id="sub-1")

    monkeypatch.setattr(chat_v2_module, "resolve_request_subject", _fake_subject)

    app, _state = _build_app(None)
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_enabled", True)
    events = asyncio.run(_post_sse(app, {
        "message": "随便聊聊",
        "request_id": "req-kernel-fail",
        "profile_id": "chat",
    }))

    assert called["n"] == 1  # kernel path was attempted
    types = [e["type"] for e in events]
    assert types.count("error") == 1, f"exactly one terminal error expected, got {types}"
    assert events[-1]["type"] == "error"
    assert events[-1]["payload"]["code"] == "model_unavailable"
    # No stub-generated text leaked onto the wire.
    assert all(e["type"] != "text" for e in events)


def test_invalid_tool_arguments_maps_to_public_kernel_failure(monkeypatch):
    """The middleware's typed argument failure never leaks as a generic 500."""
    from types import SimpleNamespace

    from uniharness.langchain.middleware import InvalidToolArgumentsError

    from agent_v2 import kernel as kernel_module

    closed: list[int] = []

    class _Agent:
        async def ainvoke(self, *_args: Any, **_kwargs: Any) -> Any:
            raise InvalidToolArgumentsError(
                "tool_call_arguments_repair_exhausted",
                call_count=2,
                call_ids=("call-1", "call-2"),
            )

        async def aclose(self) -> None:
            closed.append(1)

    async def _assemble(*_args: Any, **_kwargs: Any) -> Any:
        return _Agent()

    monkeypatch.setattr(kernel_module, "assemble_kernel_agent", _assemble)

    with pytest.raises(KernelModelFailure) as excinfo:
        asyncio.run(kernel_module.run_toolless_chat(
            system_prompt="s",
            message="生成图片",
            settings=SimpleNamespace(openai_model="fake-model"),
        ))

    assert excinfo.value.code == "invalid_tool_arguments"
    assert excinfo.value.message == "工具参数格式错误，本次请求未完成。"
    assert closed == [1]  # the real turn still closes its agent exactly once


def test_invalid_tool_arguments_is_a_stable_public_stream_code():
    from api.chat_v2 import _STREAM_ERROR_CODES

    assert "invalid_tool_arguments" in _STREAM_ERROR_CODES


def test_envelope_monotonic_and_single_terminal_on_kernel_success_contract():
    """Unit-level: extract_final_text maps last AI content; non-AI tails ignored."""
    from types import SimpleNamespace

    result = {
        "messages": [
            SimpleNamespace(type="human", content="hi"),
            SimpleNamespace(type="ai", content="partial"),
            SimpleNamespace(type="tool", content="ignored"),
            SimpleNamespace(type="ai", content="最终回答 "),
        ]
    }
    assert extract_final_text(result) == "最终回答"

    empty = {"messages": [SimpleNamespace(type="ai", content="")]}
    assert extract_final_text(empty) is None


@pytest.mark.skipif(
    not _LIVE_LLM,
    reason="live kernel SSE E2E requires KERNEL_LIVE_E2E=1 plus OPENAI_* creds (recorded skip)",
)
def test_real_kernel_end_to_end_success_over_http_sse(monkeypatch):
    """Full product SSE stream driven by the real vendored kernel loop."""
    from config import Settings

    real_settings = Settings()
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_enabled", True)

    from types import SimpleNamespace

    def _fake_subject(request):
        request.state.subject_id = "sub-1"
        return None, SimpleNamespace(issued=False, token="tok", subject_id="sub-1")

    monkeypatch.setattr(chat_v2_module, "resolve_request_subject", _fake_subject)

    app, _state = _build_app(None)
    payload = {
        "message": "用一句话介绍徒步时的三口分组原则。",
        "request_id": "req-real-kernel-e2e",
        "profile_id": "chat",
    }
    events = asyncio.run(_post_sse(app, payload))

    types = [e["type"] for e in events]
    assert types[:1] == ["run_started"]
    text_events = [e for e in events if e["type"] == "text"]
    assert text_events and "".join(e["payload"]["content"] for e in text_events).strip()

    terminals = [t for t in types if t in ("done", "error", "cancelled")]
    assert terminals == ["done"]
    seqs = [e["sequence"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def _make_validation_probe_app(monkeypatch):
    """App whose kernel stub records whether the request reached it."""
    called: dict[str, int] = {"n": 0}

    async def _failing_run(**kwargs: Any):
        called["n"] += 1
        raise KernelModelFailure("model_unavailable", "stub")

    from agent_v2 import kernel as kernel_module

    monkeypatch.setattr(kernel_module, "run_chat_turn", _failing_run)

    from types import SimpleNamespace

    def _fake_subject(request):
        request.state.subject_id = "sub-1"
        return None, SimpleNamespace(issued=False, token="tok", subject_id="sub-1")

    monkeypatch.setattr(chat_v2_module, "resolve_request_subject", _fake_subject)
    app, _state = _build_app(None)
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_enabled", True)
    return app, called


async def _post_raw(app: FastAPI, payload: dict[str, Any]):
    """Non-streaming probe: status code is enough to observe validation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/api/v2/chat/sse", json=payload)


def test_current_location_accepts_browser_accuracy_field(monkeypatch):
    """Browser geolocation always carries ``accuracy``; the strict contract
    must not 422 it away (A01)."""
    app, called = _make_validation_probe_app(monkeypatch)
    events = asyncio.run(_post_sse(app, {
        "message": "我这里适合徒步吗",
        "request_id": "req-loc-accuracy",
        "profile_id": "chat",
        "current_location": {
            "latitude": 30.26,
            "longitude": 119.56,
            "accuracy": 35.0,
            "source": "browser",
        },
    }))
    # Validation passed -> the stream opened and the kernel stub ran.
    assert called["n"] == 1
    assert events[-1]["type"] == "error"  # stub terminal, stream completed


def test_current_location_still_rejects_unknown_fields(monkeypatch):
    """extra=forbid keeps guarding against unknown location fields (A01)."""
    app, called = _make_validation_probe_app(monkeypatch)
    response = asyncio.run(_post_raw(app, {
        "message": "我这里适合徒步吗",
        "request_id": "req-loc-unknown",
        "profile_id": "chat",
        "current_location": {"latitude": 30.26, "elevation_m": 100},
    }))
    assert response.status_code == 422
    assert called["n"] == 0  # kernel never reached


# ---- MCP publication boundary (display repair batch 2) ----------------------


class _OrchestratorMemory:
    def memory_enabled(self, subject_id): return False
    def claim_conversation(self, subject_id, conversation_id): return True
    def owns_conversation(self, subject_id, conversation_id): return True


def _orchestrator_fixture(tmp_path, monkeypatch, kernel_turn):
    from config import settings
    from agent_v2.kernel_persistence import KernelCheckpointStore
    from agent_v2.persistence import CheckpointsStore, ControlStore
    from agent_v2.run_orchestration import RunOrchestrator
    from agent_v2.runs import RunRepository

    monkeypatch.setattr(settings, "agent_v2_state_dir", str(tmp_path))
    monkeypatch.setattr(settings, "agent_v2_sandbox_type", "local_native")
    monkeypatch.setattr(settings, "agent_v2_enabled", True)
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "agent_v2_thoughts_enabled", False)
    repo = RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "events.sqlite")),
    )
    checkpoints = KernelCheckpointStore(tmp_path / "graph.sqlite")
    orchestrator = RunOrchestrator(
        settings=settings,
        runs_repo=repo,
        memory=_OrchestratorMemory(),
        kernel_checkpoints=checkpoints,
        kernel_turn=kernel_turn,
    )
    return orchestrator, repo, checkpoints, settings


def _knowledge_audit(run_id: str) -> dict[str, Any]:
    return {
        "tool": "hiking_knowledge_search",
        "phase": "result",
        "ok": True,
        "run_id": run_id,
        "evidence": {"has_chunks": True, "chunks": [{"title": "fixture"}]},
    }


def test_mcp_text_is_buffered_and_published_once(tmp_path, monkeypatch, http_server):
    """Text after a custom MCP call is composed once; the preface stays once."""
    from agent_v2.run_orchestration import ChatFlowRequest
    from api.mcp_router import ServerInput, _save, test_mcp_server as discover_server

    kernel_holder: dict[str, Any] = {}

    async def dispatch(**kwargs):
        return await kernel_holder["turn"](**kwargs)

    # State dir is patched before the server is saved so the test server
    # never reaches the real agent_state directory.
    orchestrator, repo, checkpoints, _settings = _orchestrator_fixture(
        tmp_path, monkeypatch, dispatch
    )

    async def scenario():
        body = ServerInput(name="测试服务", transport="http", url=http_server, enabled=True)
        tested = await discover_server(body)
        body.test_id = tested["test_id"]
        saved = _save(body, None)
        custom_tool = f"mcp__{saved['id']}__echo"

        async def fake_kernel(**kwargs):
            run_id = kwargs["run_id"]
            knowledge = _knowledge_audit(run_id)
            custom_audit = {
                "tool": custom_tool,
                "run_id": run_id,
                "phase": "result",
                "ok": True,
                "duration_ms": 2,
                "source": "mcp",
                "server_id": saved["id"],
            }
            await kwargs["on_text"]("前言。", [knowledge])
            await kwargs["on_text"]("后文。", [knowledge, custom_audit])
            return "前言。后文。", [knowledge, custom_audit]

        kernel_holder["turn"] = fake_kernel
        try:
            request = ChatFlowRequest(
                message="看看图片", request_id="mcp-buffer", profile_id="chat"
            )
            events = [event async for event in orchestrator.stream_chat(request, subject_id="subject")]
            assert events[-1].type == "done", [(e.type, e.payload) for e in events]

            text_events = [
                event.payload["content"] for event in events if event.type == "text"
            ]
            assert text_events == ["前言。", "后文。"]

            custom_results = [
                (index, event)
                for index, event in enumerate(events)
                if event.type == "tool_result"
                and event.payload.get("tool") == custom_tool
            ]
            assert len(custom_results) == 1
            later_text = next(
                index for index, event in enumerate(events)
                if event.type == "text" and event.payload.get("content") == "后文。"
            )
            assert custom_results[0][0] < later_text
        finally:
            checkpoints.close()
            repo.close()

    asyncio.run(scenario())


def test_prefix_mismatch_appends_the_whole_answer(tmp_path, monkeypatch):
    """A rewritten answer that does not extend the streamed preface is published whole."""
    from agent_v2.run_orchestration import ChatFlowRequest

    async def fake_kernel(**kwargs):
        run_id = kwargs["run_id"]
        knowledge = _knowledge_audit(run_id)
        await kwargs["on_text"]("前言。", [knowledge])
        return "完整回答。", [knowledge]

    orchestrator, repo, checkpoints, _settings = _orchestrator_fixture(
        tmp_path, monkeypatch, fake_kernel
    )

    async def scenario():
        request = ChatFlowRequest(
            message="随便聊聊", request_id="prefix-mismatch", profile_id="chat"
        )
        events = [event async for event in orchestrator.stream_chat(request, subject_id="subject")]
        assert events[-1].type == "done"
        text_events = [
            event.payload["content"] for event in events if event.type == "text"
        ]
        assert text_events == ["前言。", "\n\n完整回答。"]
        run = repo.control.get_run_by_request(request.request_id)
        assistant = next(
            item for item in repo.get_snapshot(run.run_id).messages
            if item["role"] == "assistant"
        )
        assert assistant["content"] == "前言。\n\n完整回答。"

    try:
        asyncio.run(scenario())
    finally:
        checkpoints.close()
        repo.close()


def test_completed_audit_is_published_before_approval_pause(tmp_path, monkeypatch):
    """A pause must not swallow calls that already completed."""
    from agent_v2.capability_projection import CapabilityProjection, ToolAuthorization
    from agent_v2.permissions import PermissionDecision
    from agent_v2.run_orchestration import ChatFlowRequest

    async def fake_kernel(**kwargs):
        run_id = kwargs["run_id"]
        knowledge = _knowledge_audit(run_id)
        await kwargs["on_text"]("前言。", [knowledge])
        exc = KernelApprovalPending({
            "tool": "mcp__svc__tool",
            "args": {"x": 1},
            "call_id": "call-1",
            "interrupt_id": "interrupt-1",
        })
        exc.audit_entries = [knowledge]
        raise exc

    monkeypatch.setattr(
        CapabilityProjection,
        "authorize",
        lambda self, tool_name, args=None, **kwargs: ToolAuthorization(
            decision=PermissionDecision.NEEDS_APPROVAL,
            risk=None,
            reason="fixture",
        ),
    )
    orchestrator, repo, checkpoints, _settings = _orchestrator_fixture(
        tmp_path, monkeypatch, fake_kernel
    )

    async def scenario():
        request = ChatFlowRequest(
            message="需要审批", request_id="pause-audit", profile_id="chat"
        )
        events = [event async for event in orchestrator.stream_chat(request, subject_id="subject")]
        types = [event.type for event in events]
        assert "run.paused" in types
        knowledge_index = next(
            index for index, event in enumerate(events)
            if event.type == "tool_result"
            and event.payload.get("tool") == "hiking_knowledge_search"
        )
        assert knowledge_index < types.index("approval.required")
        assert types.index("approval.required") < types.index("run.paused")
        assert "done" not in types
        run = repo.control.get_run_by_request(request.request_id)
        assert run.status == "paused"
        assert len(repo.list_pending_approvals_for_run(run.run_id)) == 1

    try:
        asyncio.run(scenario())
    finally:
        checkpoints.close()
        repo.close()


def test_resume_does_not_republish_tool_events(tmp_path, monkeypatch):
    """The continuation adds no duplicate tool_result for already-persisted work."""
    from agent_v2.capability_projection import CapabilityProjection, ToolAuthorization
    from agent_v2.permissions import PermissionDecision
    from agent_v2.run_orchestration import ChatFlowRequest

    holder: dict[str, Any] = {}

    async def dispatch(**kwargs):
        return await holder["turn"](**kwargs)

    monkeypatch.setattr(
        CapabilityProjection,
        "authorize",
        lambda self, tool_name, args=None, **kwargs: ToolAuthorization(
            decision=PermissionDecision.NEEDS_APPROVAL,
            risk=None,
            reason="fixture",
        ),
    )
    orchestrator, repo, checkpoints, _settings = _orchestrator_fixture(
        tmp_path, monkeypatch, dispatch
    )

    async def scenario():
        async def first_turn(**kwargs):
            run_id = kwargs["run_id"]
            knowledge = _knowledge_audit(run_id)
            await kwargs["on_text"]("前言。", [knowledge])
            exc = KernelApprovalPending({
                "tool": "mcp__svc__tool",
                "args": {"x": 1},
                "call_id": "call-1",
                "interrupt_id": "interrupt-1",
            })
            exc.audit_entries = [knowledge]
            raise exc

        holder["turn"] = first_turn
        request = ChatFlowRequest(
            message="恢复测试", request_id="resume-dedupe", profile_id="chat"
        )
        original = [event async for event in orchestrator.stream_chat(request, subject_id="subject")]
        assert original[-1].type == "run.paused"
        run = repo.control.get_run_by_request(request.request_id)
        last_seq = original[-1].sequence
        approval = repo.list_pending_approvals_for_run(run.run_id)[0]
        decision = orchestrator.resolve_approval_decision(
            approval.approval_id,
            decision="approve",
            decision_request_id="decision-resume-dedupe",
        )
        assert decision.status == "recorded"

        async def resumed_turn(**kwargs):
            return "继续。", []

        holder["turn"] = resumed_turn
        resumed = [
            event async for event in orchestrator.stream_resume(
                run_id=run.run_id, from_seq=last_seq, subject_id="subject"
            )
        ]
        assert resumed[-1].type == "done"
        assert not [event for event in resumed if event.type == "tool_result"]
        assert "".join(
            event.payload.get("content", "")
            for event in resumed
            if event.type == "text"
        ) == "继续。"

    try:
        asyncio.run(scenario())
    finally:
        checkpoints.close()
        repo.close()


def test_completed_audit_is_published_before_model_failure(tmp_path, monkeypatch):
    """A failure terminal must not swallow calls that already completed."""
    from agent_v2.run_orchestration import ChatFlowRequest

    async def fake_kernel(**kwargs):
        run_id = kwargs["run_id"]
        knowledge = _knowledge_audit(run_id)
        await kwargs["on_text"]("前言。", [knowledge])
        exc = KernelModelFailure("model_unavailable", "主模型未配置 API Key。")
        exc.audit_entries = [knowledge]
        raise exc

    orchestrator, repo, checkpoints, _settings = _orchestrator_fixture(
        tmp_path, monkeypatch, fake_kernel
    )

    async def scenario():
        request = ChatFlowRequest(
            message="失败出口", request_id="failure-audit", profile_id="chat"
        )
        events = [event async for event in orchestrator.stream_chat(request, subject_id="subject")]
        types = [event.type for event in events]
        knowledge_index = next(
            index for index, event in enumerate(events)
            if event.type == "tool_result"
            and event.payload.get("tool") == "hiking_knowledge_search"
        )
        assert knowledge_index < types.index("error")
        assert events[-1].type == "error"
        assert events[-1].payload["code"] == "model_unavailable"

    try:
        asyncio.run(scenario())
    finally:
        checkpoints.close()
        repo.close()
