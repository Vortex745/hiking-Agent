"""Ticket 32 — memory tools through the kernel, server-subject bound."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import api.chat_v2 as chat_v2_module
from agent_v2.kernel_tools import MemorySaveTool, MemorySearchTool, build_chat_extra_tools


class _FakeMemory:
    def __init__(self, *, opted_in: bool = True) -> None:
        self.opted_in = opted_in
        self.saved: list[tuple[str, str]] = []
        self.items = [{"content": "喜欢清晨出发"}, {"content": "重装党"}]

    def memory_enabled(self, subject_id: str) -> bool:
        return self.opted_in

    def claim_conversation(self, subject_id: str, conversation_id: str) -> bool:
        return True

    def owns_conversation(self, subject_id: str, conversation_id: str) -> bool:
        return True

    def save_preference(self, subject_id: str, content: str) -> str | None:
        self.saved.append((subject_id, content))
        return "mem-1"

    def list_active_memories(self, subject_id: str) -> list[dict[str, Any]]:
        return self.items

    def create_candidate(self, subject_id: str, **candidate: str) -> dict[str, Any]:
        return {"candidate_id": "c-1", **candidate}


class _Profile:
    class profile_id:
        value = "chat"

    allowed_tools = ("memory_save", "memory_search")


def _save_tool(memory: Any, subject: str | None):
    audit: list[dict[str, Any]] = []
    return MemorySaveTool(memory=memory, subject_id=subject, run_id="r32", audit_sink=audit.append), audit


def test_memory_save_happy_binds_server_subject():
    memory = _FakeMemory()
    tool, audit = _save_tool(memory, "sub-1")

    result = asyncio.run(tool(content="喜欢清晨出发"))

    assert result.output == "saved:mem-1"
    assert memory.saved == [("sub-1", "喜欢清晨出发")]  # subject from server, never client
    assert audit[-1]["phase"] == "result" and audit[-1]["ok"] is True


def test_memory_save_not_opted_in_rejected():
    memory = _FakeMemory(opted_in=False)
    tool, audit = _save_tool(memory, "sub-1")

    result = asyncio.run(tool(content="普通偏好"))

    assert result.error and "未开启长期记忆" in result.error
    assert memory.saved == []
    assert audit[-1]["code"] == "memory_not_opted_in"


def test_memory_save_sensitive_content_requires_consent_not_tool():
    memory = _FakeMemory()
    tool, audit = _save_tool(memory, "sub-1")

    result = asyncio.run(tool(content="记住我的手机号是13812345678"))

    assert result.error and "敏感" in result.error
    assert memory.saved == []  # nothing persisted without product-API consent
    assert audit[-1]["code"] == "sensitive_requires_consent"


def test_memory_search_lists_only_active_items():
    memory = _FakeMemory()
    tool, _ = _save_tool(memory, "sub-1")
    search = MemorySearchTool(memory=memory, subject_id="sub-1", run_id="r32", audit_sink=None)

    result = asyncio.run(search(query=None))

    assert "喜欢清晨出发" in (result.output or "")
    assert "重装党" in (result.output or "")


def test_memory_tools_without_subject_fail_closed():
    memory = _FakeMemory()
    tool, _ = _save_tool(memory, None)

    result = asyncio.run(tool(content="x"))
    assert result.error and "缺少已认证主体" in result.error


def test_builder_fails_closed_when_memory_allowed_but_subject_missing():
    with pytest.raises(Exception) as excinfo:
        build_chat_extra_tools(_Profile(), run_id="r", audit_sink=None, memory_service=None, subject_id=None)
    assert "fail closed" in str(excinfo.value)


# ---------------------------------------------------------------------------
# SSE: consent event still product-owned; save evidence rides tool_result
# ---------------------------------------------------------------------------


def _app(tmp_path: Path, memory: Any, subject: Any) -> FastAPI:
    app = FastAPI()
    app.include_router(chat_v2_module.chat_v2_router, prefix="/api/v2")

    class _Runtime:
        runs = None
        telemetry = None
        capacity = None
        kernel_checkpoints = None

    rt = _Runtime()
    rt.memory = memory
    app.state.agent_v2_runtime = rt
    return app


async def _post(app: FastAPI, payload: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        async with client.stream("POST", "/api/v2/chat/sse", json=payload) as resp:
            assert resp.status_code == 200
            buf = ""
            async for chunk in resp.aiter_text():
                buf += chunk
                while "\n\n" in buf:
                    frame, buf = buf.split("\n\n", 1)
                    for line in frame.splitlines():
                        if line.startswith("data: "):
                            events.append(json.loads(line[6:]))
    return events


def test_sse_sensitive_message_emits_consent_required_event(monkeypatch, tmp_path: Path):
    memory = _FakeMemory()
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_enabled", True)

    def _fake_subject(request):
        request.state.subject_id = "sub-1"
        return memory, SimpleNamespace(issued=False, token="tok", subject_id="sub-1")

    monkeypatch.setattr(chat_v2_module, "resolve_request_subject", _fake_subject)

    from agent_v2 import kernel as km

    async def fake_turn(**kwargs: Any):
        return "好的，这条信息我不会保存。", []

    monkeypatch.setattr(km, "run_chat_turn", fake_turn)

    events = asyncio.run(_post(_app(tmp_path, memory, None), {
        "message": "记住我的手机号是13812345678，徒步时方便联系",
        "request_id": "req-mem-consent",
        "profile_id": "chat",
    }))

    types = [e["type"] for e in events]
    assert "memory.consent_required" in types
    assert "text" in types and types[-1] == "done"
    # consent path short-circuits the kernel text? No — text comes from kernel
    # turn afterwards; both must coexist with a single terminal.
    assert types.count("done") == 1


def test_sse_memory_save_audit_surfaces_as_tool_result(monkeypatch, tmp_path: Path):
    memory = _FakeMemory()
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_enabled", True)

    def _fake_subject(request):
        request.state.subject_id = "sub-1"
        return memory, SimpleNamespace(issued=False, token="tok", subject_id="sub-1")

    monkeypatch.setattr(chat_v2_module, "resolve_request_subject", _fake_subject)

    from agent_v2 import kernel as km

    async def fake_turn(**kwargs: Any):
        tools = [t.name for t in kwargs.get("extra_tools", ())]
        assert "memory_save" in tools and "memory_search" in tools, tools
        return "已记住你的偏好。", [
            {"tool": "memory_save", "phase": "result", "ok": True, "code": None, "duration_ms": 2},
        ]

    monkeypatch.setattr(km, "run_chat_turn", fake_turn)

    events = asyncio.run(_post(_app(tmp_path, memory, None), {
        "message": "帮我记住我喜欢清晨出发",
        "request_id": "req-mem-save",
        "profile_id": "chat",
    }))

    tool_results = [e for e in events if e["type"] == "tool_result" and e["payload"].get("tool") == "memory_save"]
    assert tool_results and tool_results[0]["payload"]["status"] == "ok"
    assert types_single_terminal(events)


def types_single_terminal(events: list[dict[str, Any]]) -> bool:
    terminals = [e["type"] for e in events if e["type"] in ("done", "error", "cancelled")]
    return terminals == ["done"]
