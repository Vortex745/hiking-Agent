"""Kernel tracing contract tests (ticket 01).

The old SDK callback ingestion path was removed: the kernel must not attach
LangChain callbacks or Langfuse session metadata to the invoke config. A
single execution produces a single OTel trace started at the HTTP boundary.
"""

from types import SimpleNamespace

import agent_v2.kernel as kernel_mod
from agent_v2.kernel import run_chat_turn
from langchain_core.messages import AIMessage


class _FakeAgent:
    def __init__(self) -> None:
        self.captured_config = None

    async def ainvoke(self, inputs, config=None):
        self.captured_config = config
        return {"messages": [AIMessage(content="ok")]}

    async def aclose(self) -> None:
        return None


def test_kernel_does_not_attach_sdk_callbacks(monkeypatch) -> None:
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
