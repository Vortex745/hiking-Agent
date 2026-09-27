"""Kernel Langfuse contract tests.

AC:
1. Langfuse enabled → ``run_chat_turn`` passes callbacks + session metadata
   into ``agent.ainvoke`` config so the whole kernel run tree is traced under
   the request's session id.
2. Langfuse disabled (empty callbacks) → invoke config stays untouched.
"""

from types import SimpleNamespace

import agent_v2.kernel as kernel_mod
from agent_v2.kernel import run_chat_turn
from langchain_core.messages import AIMessage
from observability.langfuse_setup import reset_langfuse_session, set_langfuse_session


class _FakeAgent:
    def __init__(self) -> None:
        self.captured_config = None

    async def ainvoke(self, inputs, config=None):
        self.captured_config = config
        return {"messages": [AIMessage(content="ok")]}

    async def aclose(self) -> None:
        return None


def _run_turn(monkeypatch, callbacks) -> _FakeAgent:
    agent = _FakeAgent()

    async def _fake_assemble(config, settings):
        return agent

    monkeypatch.setattr(kernel_mod, "assemble_kernel_agent", _fake_assemble)
    monkeypatch.setattr(kernel_mod, "get_langchain_callbacks", lambda: callbacks)
    text, _audit = __import__("asyncio").run(
        run_chat_turn(
            system_prompt="sp",
            message="hi",
            settings=SimpleNamespace(),
            run_id="run-1",
        )
    )
    assert text == "ok"
    return agent


def test_invoke_config_carries_callbacks_and_session(monkeypatch) -> None:
    sentinel = object()
    token = set_langfuse_session("conv-42")
    try:
        agent = _run_turn(monkeypatch, [sentinel])
    finally:
        reset_langfuse_session(token)
    config = agent.captured_config
    assert config is not None
    assert config["callbacks"] == [sentinel]
    assert config["metadata"]["langfuse_session_id"] == "conv-42"


def test_disabled_tracing_leaves_config_untouched(monkeypatch) -> None:
    agent = _run_turn(monkeypatch, [])
    assert agent.captured_config is None


def test_session_absent_keeps_callbacks_without_metadata(monkeypatch) -> None:
    sentinel = object()
    agent = _run_turn(monkeypatch, [sentinel])
    config = agent.captured_config
    assert config["callbacks"] == [sentinel]
    assert "metadata" not in config
