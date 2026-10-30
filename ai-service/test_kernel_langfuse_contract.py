"""Kernel tracing contract tests (ticket 01).

The old SDK callback ingestion path was removed: the kernel must never attach
Langfuse session metadata to the invoke config. A single execution produces a
single OTel trace started at the HTTP boundary; the only callbacks attached are
the tool-span handler from ticket 04.
"""

from types import SimpleNamespace

import pytest

import agent_v2.kernel as kernel_mod
from agent_v2.kernel import run_chat_turn
from langchain_core.messages import AIMessage


@pytest.fixture(autouse=True)
def _no_llm_log(monkeypatch):
    """LLM_LOG 是独立通道：本文件只测 tracing，避免前序测试的全局开关污染。"""
    monkeypatch.setattr(kernel_mod.llm_log, "enabled", lambda: False)


class _FakeAgent:
    def __init__(self) -> None:
        self.captured_config = None

    async def ainvoke(self, inputs, config=None):
        self.captured_config = config
        return {"messages": [AIMessage(content="ok")]}

    async def aclose(self) -> None:
        return None


def _disable_tracing(monkeypatch) -> None:
    monkeypatch.setattr(kernel_mod.tracing, "tracing_enabled", lambda: False)


def test_kernel_does_not_attach_sdk_callbacks(monkeypatch) -> None:
    _disable_tracing(monkeypatch)
    agent = _FakeAgent()

    async def _fake_assemble(config, settings):
        return agent

    monkeypatch.setattr(kernel_mod, "assemble_kernel_agent", _fake_assemble)
    text, _audit = __import__("asyncio").run(
        run_chat_turn(
            system_prompt="sp",
            message="hi",
            settings=SimpleNamespace(),
            run_id="run-1",
        )
    )

    assert text == "ok"
    # Without a checkpointer there is no config at all; with one the config
    # only carries the thread id. No callbacks / session metadata (the old
    # double-trace route) may appear.
    assert agent.captured_config is None


def test_kernel_config_only_carries_thread_id(monkeypatch) -> None:
    _disable_tracing(monkeypatch)
    agent = _FakeAgent()

    async def _fake_assemble(config, settings):
        return agent

    class _Store:
        async def get(self):
            return object()

    monkeypatch.setattr(kernel_mod, "assemble_kernel_agent", _fake_assemble)
    __import__("asyncio").run(
        run_chat_turn(
            system_prompt="sp",
            message="hi",
            settings=SimpleNamespace(),
            run_id="run-1",
            checkpoint_store=_Store(),
        )
    )

    assert agent.captured_config == {
        "configurable": {"thread_id": "v2:chat:run-1"}
    }


def test_kernel_attaches_only_tool_span_handler(monkeypatch) -> None:
    """Tracing on: the only callback is the ticket-04 tool span handler."""
    from observability.callbacks import ToolSpanCallbackHandler

    agent = _FakeAgent()

    async def _fake_assemble(config, settings):
        return agent

    monkeypatch.setattr(kernel_mod, "assemble_kernel_agent", _fake_assemble)
    monkeypatch.setattr(kernel_mod.tracing, "tracing_enabled", lambda: True)
    monkeypatch.setattr(kernel_mod.tracing, "get_tracer", lambda: None)

    __import__("asyncio").run(
        run_chat_turn(
            system_prompt="sp",
            message="hi",
            settings=SimpleNamespace(),
            run_id="run-1",
        )
    )

    callbacks = agent.captured_config["callbacks"]
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], ToolSpanCallbackHandler)
