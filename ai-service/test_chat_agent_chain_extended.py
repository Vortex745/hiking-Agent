"""Agent 模块链路扩展测试。

覆盖范围（与已有 test_chat_sse_agent_chain.py 互补）：
1. SSE 错误路径：无 API key / Agent 异常 / 兜底 done
2. 辅助端点：/chat/sync, /chat/health, /chat/history GET/DELETE, /chat/pending
3. 工具确认链路：needs_confirmation → /chat/confirm 的 confirm/reject/already_resolved/not_found
4. 多轮对话：history 透传给 Agent、累积持久化

设计原则（Karpathy）：
- 最小代码，不引入 TestClient，直接 asyncio.run 调用端点函数。
- FakeAgent/FakeMemory 与既有测试文件保持一致，便于复用与审阅。
- 每个 test 独立 monkeypatch，互不污染。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi import HTTPException

from api import chat
from api.confirmation_store import ConfirmationStore
from api.models import (
    ChatRequest,
    ConfirmRequest,
    CurrentLocationPayload,
)
from agent.tool_names import OPENMANUS_TOOL_NAMES


# ── Fakes ──────────────────────────────────────────────────────────────


class FakeMemory:
    """最小 ChatMemory 实现：记录 add_message 调用，可预设历史。"""

    def __init__(self, messages: list[dict[str, str]] | None = None) -> None:
        self.messages: list[dict[str, str]] = list(messages or [])

    def get_messages(self) -> list[dict[str, str]]:
        return list(self.messages)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def clear(self) -> None:
        self.messages = []


class FakeAgent:
    """记录调用参数的假 Agent；可配置流式事件序列。"""

    invocations: list[dict[str, Any]] = []  # 类变量，跨实例共享以便断言
    stream_events: list[dict[str, Any]] = []
    raise_on_stream: Exception | None = None
    raise_on_aexecute: Exception | None = None

    def __init__(self, *, llm_config: Any = None) -> None:
        self.llm_config = llm_config

    @classmethod
    def reset(cls, *, stream_events: list[dict[str, Any]] | None = None) -> None:
        cls.invocations = []
        cls.stream_events = list(stream_events or [])
        cls.raise_on_stream = None
        cls.raise_on_aexecute = None

    async def aexecute(
        self,
        message: str,
        history: list | None = None,
        scenario: str | None = None,
        current_location: Any = None,
    ) -> dict:
        self.invocations.append({
            "message": message,
            "history": history,
            "scenario": scenario,
            "current_location": current_location,
            "stream": False,
        })
        if self.raise_on_aexecute is not None:
            raise self.raise_on_aexecute
        # 返回最后一条 text 事件作为 output
        for event in reversed(self.stream_events):
            if event.get("type") == "text":
                return {"output": event.get("content", "")}
        return {"output": "ok"}

    async def aexecute_stream(
        self,
        message: str,
        history: list | None = None,
        scenario: str | None = None,
        current_location: Any = None,
    ) -> Any:
        self.invocations.append({
            "message": message,
            "history": history,
            "scenario": scenario,
            "current_location": current_location,
            "stream": True,
        })
        if self.raise_on_stream is not None:
            raise self.raise_on_stream
        for event in self.stream_events:
            yield event


class FakeToolRegistry:
    """记录 execute_tool 调用的假工具注册表。"""

    def __init__(self, result: dict | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._result = result or {"ok": True, "result": "executed"}

    async def execute_tool(self, tool_name: str, args: dict) -> dict:
        self.calls.append((tool_name, args))
        return self._result


# ── Helpers ───────────────────────────────────────────────────────────


async def _drain_sse(response: Any) -> str:
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.encode() if isinstance(chunk, str) else chunk)
    return b"".join(chunks).decode()


def _parse_sse(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(frame.removeprefix("data: "))
        for frame in body.split("\n\n")
        if frame.strip()
    ]


def _apply_common_mocks(monkeypatch, *, memory: FakeMemory, has_key: bool = True) -> None:
    """统一打桩：memory + API key + agent class + store。"""
    monkeypatch.setattr(chat, "_get_memory", lambda _chat_id: memory)
    monkeypatch.setattr(chat, "_has_llm_api_key", lambda _req: has_key)
    monkeypatch.setattr(chat, "_get_agent_class", lambda: FakeAgent)
    monkeypatch.setattr(chat, "get_store", lambda: ConfirmationStore())


# ── 1. SSE 错误路径 ────────────────────────────────────────────────────


class TestChatSseErrorPaths:
    """验证 chat_sse 在异常/边界场景下仍能保持 error→done 的 SSE 契约。"""

    def test_no_api_key_emits_error_and_done(self, monkeypatch) -> None:
        memory = FakeMemory()
        FakeAgent.reset(stream_events=[{"type": "text", "content": "should not run"}])
        _apply_common_mocks(monkeypatch, memory=memory, has_key=False)

        request = ChatRequest(message="hi", chat_id="no-key-chat")
        response = asyncio.run(chat.chat_sse(request))
        body = asyncio.run(_drain_sse(response))
        events = _parse_sse(body)

        assert [e["type"] for e in events] == ["error", "done"]
        assert "API Key" in events[0]["content"]
        # Agent 不应被调用
        assert FakeAgent.invocations == []
        # user 消息仍被记录（chat_sse 入口即 add_message）
        assert memory.messages == [{"role": "user", "content": "hi"}]

    def test_agent_exception_emits_error_and_done(self, monkeypatch) -> None:
        memory = FakeMemory()
        FakeAgent.reset(stream_events=[])
        FakeAgent.raise_on_stream = RuntimeError("boom inside agent")
        _apply_common_mocks(monkeypatch, memory=memory, has_key=True)

        request = ChatRequest(message="trigger", chat_id="exc-chat")
        response = asyncio.run(chat.chat_sse(request))
        body = asyncio.run(_drain_sse(response))
        events = _parse_sse(body)

        assert [e["type"] for e in events] == ["error", "done"]
        assert "boom inside agent" in events[0]["content"]
        # Agent 被调用了一次
        assert len(FakeAgent.invocations) == 1
        assert FakeAgent.invocations[0]["stream"] is True

    def test_missing_done_is_appended(self, monkeypatch) -> None:
        """Agent 流未发 done 时，端点应兜底补 done（SSE 契约要求）。"""
        memory = FakeMemory()
        FakeAgent.reset(stream_events=[
            {"type": "thought", "content": "思考"},
            {"type": "text", "content": "回答"},
        ])
        _apply_common_mocks(monkeypatch, memory=memory, has_key=True)

        request = ChatRequest(message="hi", chat_id="no-done-chat")
        response = asyncio.run(chat.chat_sse(request))
        body = asyncio.run(_drain_sse(response))
        events = _parse_sse(body)

        assert [e["type"] for e in events] == ["thought", "text", "done"]
        # 助手回复仍被持久化
        assert memory.messages[-1] == {"role": "assistant", "content": "回答"}

    def test_persistence_skipped_when_no_text(self, monkeypatch) -> None:
        """只有 thought/tool_call 没有 text 时不应写空 assistant 消息。"""
        memory = FakeMemory()
        FakeAgent.reset(stream_events=[
            {"type": "thought", "content": "thinking only"},
            {"type": "done", "content": ""},
        ])
        _apply_common_mocks(monkeypatch, memory=memory, has_key=True)

        request = ChatRequest(message="hi", chat_id="no-text-chat")
        response = asyncio.run(chat.chat_sse(request))
        asyncio.run(_drain_sse(response))

        # 只有 user 被记录，无 assistant 空消息
        assert memory.messages == [{"role": "user", "content": "hi"}]

    def test_scenario_and_current_location_propagated(self, monkeypatch) -> None:
        memory = FakeMemory()
        FakeAgent.reset(stream_events=[{"type": "text", "content": "ok"}, {"type": "done", "content": ""}])
        _apply_common_mocks(monkeypatch, memory=memory, has_key=True)

        request = ChatRequest(
            message="推荐路线",
            chat_id="loc-chat",
            scenario="route_plan",
            current_location=CurrentLocationPayload(city="杭州", source="test"),
        )
        response = asyncio.run(chat.chat_sse(request))
        asyncio.run(_drain_sse(response))

        assert len(FakeAgent.invocations) == 1
        inv = FakeAgent.invocations[0]
        assert inv["scenario"] == "route_plan"
        assert inv["current_location"]["city"] == "杭州"


# ── 2. 辅助端点 ────────────────────────────────────────────────────────


class TestChatEndpoints:
    """验证 /chat/sync, /chat/health, /chat/history, /chat/pending 的契约。"""

    def test_chat_sync_returns_response_and_persists(self, monkeypatch) -> None:
        memory = FakeMemory()
        FakeAgent.reset(stream_events=[{"type": "text", "content": "sync reply"}])
        _apply_common_mocks(monkeypatch, memory=memory, has_key=True)

        request = ChatRequest(message="hi sync", chat_id="sync-chat")
        result = asyncio.run(chat.chat_sync(request))

        assert isinstance(result, chat.ChatResponse)
        assert result.content == "sync reply"
        assert result.chat_id == "sync-chat"
        # user + assistant 都被持久化
        assert memory.messages == [
            {"role": "user", "content": "hi sync"},
            {"role": "assistant", "content": "sync reply"},
        ]

    def test_chat_sync_503_when_no_api_key(self, monkeypatch) -> None:
        """chat_sync 在写 memory 之前先校验 API key，所以无 key 时 memory 仍为空。"""
        memory = FakeMemory()
        FakeAgent.reset()
        _apply_common_mocks(monkeypatch, memory=memory, has_key=False)

        request = ChatRequest(message="hi", chat_id="no-key-sync")
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(chat.chat_sync(request))
        assert exc_info.value.status_code == 503
        # 关键断言：raise 在 add_message 之前，所以 memory 不会被写入
        assert memory.messages == []
        assert FakeAgent.invocations == []

    def test_chat_sync_500_on_agent_error(self, monkeypatch) -> None:
        """chat_sync 调 aexecute（非 stream），Agent 抛错被捕获转 500。"""
        memory = FakeMemory()
        FakeAgent.reset(stream_events=[])
        FakeAgent.raise_on_aexecute = ValueError("agent failed")
        _apply_common_mocks(monkeypatch, memory=memory, has_key=True)

        request = ChatRequest(message="hi", chat_id="err-sync")
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(chat.chat_sync(request))
        assert exc_info.value.status_code == 500

    def test_chat_health_returns_ok_with_tools_count(self, monkeypatch) -> None:
        # /chat/health 不需要 mock，直接调用即可
        from config import settings

        result = asyncio.run(chat.chat_health())
        assert result["module"] == "agent"
        assert result["service"] == "ai-service"
        assert result["tools"] == len(OPENMANUS_TOOL_NAMES)
        # status 反映 openai_api_key 配置
        expected = "ok" if settings.openai_api_key else "unconfigured"
        assert result["status"] == expected

    def test_chat_history_get_returns_messages(self, monkeypatch) -> None:
        memory = FakeMemory([
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ])
        monkeypatch.setattr(chat, "_get_memory", lambda _chat_id: memory)

        result = asyncio.run(chat.chat_history("hist-chat"))
        assert result["chat_id"] == "hist-chat"
        assert result["messages"] == [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]

    def test_chat_history_delete_clears_memory(self, monkeypatch) -> None:
        memory = FakeMemory([{"role": "user", "content": "to-clear"}])
        monkeypatch.setattr(chat, "_get_memory", lambda _chat_id: memory)

        result = asyncio.run(chat.chat_clear_history("clear-chat"))
        assert result == {"chat_id": "clear-chat", "status": "cleared"}
        assert memory.messages == []

    def test_chat_pending_returns_only_pending(self, monkeypatch) -> None:
        store = ConfirmationStore()
        cid_a = store.add("tool_a", {"x": 1}, "chat-pending", step=0)
        cid_b = store.add("tool_b", {"x": 2}, "chat-pending", step=1)
        store.confirm(cid_b)  # b 不再 pending

        monkeypatch.setattr(chat, "get_store", lambda: store)
        result = asyncio.run(chat.chat_pending("chat-pending"))

        assert result.chat_id == "chat-pending"
        ids = [item.confirmation_id for item in result.pending]
        assert cid_a in ids
        assert cid_b not in ids
        assert all(item.status == "pending" for item in result.pending)


# ── 3. 工具确认链路 ────────────────────────────────────────────────────


class TestChatToolConfirmation:
    """验证 needs_confirmation 元数据附加 + /chat/confirm 决策。"""

    def test_confirm_unknown_id_returns_not_found(self, monkeypatch) -> None:
        store = ConfirmationStore()
        monkeypatch.setattr(chat, "get_store", lambda: store)

        req = ConfirmRequest(confirmation_id="does-not-exist", action="confirm")
        result = asyncio.run(chat.chat_confirm(req))
        assert result.status == "not_found"
        assert result.confirmation_id == "does-not-exist"

    def test_confirm_executes_tool_and_returns_result(self, monkeypatch) -> None:
        store = ConfirmationStore()
        cid = store.add("route_search", {"dest": "杭州"}, "confirm-chat", step=0)

        registry = FakeToolRegistry(result={"ok": True, "result": {"routes": ["r1"]}})
        monkeypatch.setattr(chat, "get_store", lambda: store)
        monkeypatch.setattr(chat, "_get_tool_registry", lambda: registry)

        req = ConfirmRequest(confirmation_id=cid, action="confirm")
        result = asyncio.run(chat.chat_confirm(req))

        assert result.status == "confirmed"
        assert result.tool_name == "route_search"
        assert result.tool_result == {"ok": True, "result": {"routes": ["r1"]}}
        # 工具被调用一次，参数透传
        assert registry.calls == [("route_search", {"dest": "杭州"})]
        # store 状态已更新
        assert store.get(cid).status == "confirmed"

    def test_reject_does_not_execute_tool(self, monkeypatch) -> None:
        store = ConfirmationStore()
        cid = store.add("dangerous_tool", {"x": 1}, "reject-chat", step=0)

        registry = FakeToolRegistry()
        monkeypatch.setattr(chat, "get_store", lambda: store)
        monkeypatch.setattr(chat, "_get_tool_registry", lambda: registry)

        req = ConfirmRequest(confirmation_id=cid, action="reject")
        result = asyncio.run(chat.chat_confirm(req))

        assert result.status == "rejected"
        assert result.tool_result is None
        # 工具不应被调用
        assert registry.calls == []
        assert store.get(cid).status == "rejected"

    def test_confirm_already_resolved(self, monkeypatch) -> None:
        store = ConfirmationStore()
        cid = store.add("tool_x", {}, "resolved-chat", step=0)
        store.confirm(cid)

        registry = FakeToolRegistry()
        monkeypatch.setattr(chat, "get_store", lambda: store)
        monkeypatch.setattr(chat, "_get_tool_registry", lambda: registry)

        req = ConfirmRequest(confirmation_id=cid, action="confirm")
        result = asyncio.run(chat.chat_confirm(req))

        assert result.status == "already_resolved"
        # 工具不应被重复执行
        assert registry.calls == []

    def test_invalid_action_returns_400(self, monkeypatch) -> None:
        store = ConfirmationStore()
        monkeypatch.setattr(chat, "get_store", lambda: store)

        req = ConfirmRequest(confirmation_id="any", action="bogus")
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(chat.chat_confirm(req))
        assert exc_info.value.status_code == 400


# ── 4. 多轮对话历史 ────────────────────────────────────────────────────


class TestChatMultiTurn:
    """验证多轮对话时 history 正确透传与累积持久化。"""

    def test_history_passed_to_agent_on_second_turn(self, monkeypatch) -> None:
        # 模拟首轮已结束后的状态
        memory = FakeMemory([
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
        ])
        FakeAgent.reset(stream_events=[
            {"type": "text", "content": "第二答"},
            {"type": "done", "content": ""},
        ])
        _apply_common_mocks(monkeypatch, memory=memory, has_key=True)

        request = ChatRequest(message="第二问", chat_id="multi-chat")
        response = asyncio.run(chat.chat_sse(request))
        asyncio.run(_drain_sse(response))

        # 验证 Agent 收到了完整 history（首轮 user+assistant）
        assert len(FakeAgent.invocations) == 1
        inv = FakeAgent.invocations[0]
        assert inv["history"] == [
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
        ]
        assert inv["message"] == "第二问"

        # 验证 memory 累积：首轮 2 条 + 本轮 user+assistant = 4 条
        assert memory.messages == [
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
            {"role": "user", "content": "第二问"},
            {"role": "assistant", "content": "第二答"},
        ]

    def test_empty_history_on_first_turn(self, monkeypatch) -> None:
        memory = FakeMemory()  # 空
        FakeAgent.reset(stream_events=[
            {"type": "text", "content": "首答"},
            {"type": "done", "content": ""},
        ])
        _apply_common_mocks(monkeypatch, memory=memory, has_key=True)

        request = ChatRequest(message="首问", chat_id="first-chat")
        response = asyncio.run(chat.chat_sse(request))
        asyncio.run(_drain_sse(response))

        assert FakeAgent.invocations[0]["history"] == []
        assert memory.messages == [
            {"role": "user", "content": "首问"},
            {"role": "assistant", "content": "首答"},
        ]

    def test_three_turns_accumulate_correctly(self, monkeypatch) -> None:
        memory = FakeMemory()
        FakeAgent.reset(stream_events=[
            {"type": "text", "content": "答"},
            {"type": "done", "content": ""},
        ])
        _apply_common_mocks(monkeypatch, memory=memory, has_key=True)

        # 模拟三轮对话
        for turn in ["问1", "问2", "问3"]:
            request = ChatRequest(message=turn, chat_id="three-chat")
            # 每轮重置 stream_events 以保持一致
            FakeAgent.reset(stream_events=[
                {"type": "text", "content": f"答-{turn}"},
                {"type": "done", "content": ""},
            ])
            response = asyncio.run(chat.chat_sse(request))
            asyncio.run(_drain_sse(response))

        # 三轮后 memory 应有 6 条
        assert len(memory.messages) == 6
        assert memory.messages[0] == {"role": "user", "content": "问1"}
        assert memory.messages[1] == {"role": "assistant", "content": "答-问1"}
        assert memory.messages[-2] == {"role": "user", "content": "问3"}
        assert memory.messages[-1] == {"role": "assistant", "content": "答-问3"}

        # 第三轮 Agent 收到的 history 是前 4 条（前两轮）
        assert len(FakeAgent.invocations) == 1  # reset 清空了，最后一轮的调用
        assert len(FakeAgent.invocations[0]["history"]) == 4
