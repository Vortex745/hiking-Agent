"""Turn and tool span hierarchy tests (ticket 04)."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import agent_v2.kernel as kernel_mod
from agent_v2.kernel import run_chat_turn
from observability import tracing


class _FakeAgent:
    def __init__(self, *, fail: bool = False, nested_generation: bool = False) -> None:
        self.fail = fail
        self.nested_generation = nested_generation
        self.last_config = None

    async def ainvoke(self, inputs, config=None):
        self.last_config = config
        handler = (config or {}).get("callbacks", [None])[0]
        if handler is not None:
            handler.on_tool_start({"name": "search_docs"}, "q", run_id="tool-1")
            handler.on_tool_end("raw tool output", run_id="tool-1")
        if self.nested_generation:
            with tracing.start_span("chat.generation"):
                pass
        if self.fail:
            raise RuntimeError("model exploded")
        return {"messages": [AIMessage(content="ok")]}

    async def aclose(self) -> None:
        return None


@pytest.fixture()
def spans(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_initialized", True)
    monkeypatch.setattr(tracing, "_shutdown", False)
    monkeypatch.setattr(tracing, "_provider", provider)
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("test"))
    monkeypatch.setattr(tracing, "tracing_enabled", lambda: True)
    return exporter


def _run(monkeypatch, agent: _FakeAgent, *, run_id: str = "run-1"):
    async def _fake_assemble(config, settings):
        return agent

    monkeypatch.setattr(kernel_mod, "assemble_kernel_agent", _fake_assemble)
    return __import__("asyncio").run(
        run_chat_turn(
            system_prompt="sp",
            message="hi",
            settings=SimpleNamespace(),
            run_id=run_id,
        )
    )


def test_turn_span_is_parent_of_tool_and_nested_generation(spans, monkeypatch):
    text, _audit = _run(
        monkeypatch, _FakeAgent(nested_generation=True), run_id="run-42"
    )
    assert text == "ok"

    by_name = {span.name: span for span in spans.get_finished_spans()}
    assert "agent.turn" in by_name
    assert "tool.search_docs" in by_name
    assert "chat.generation" in by_name

    turn = by_name["agent.turn"]
    assert turn.attributes["run.id"] == "run-42"
    assert turn.attributes["gen_ai.operation.name"] == "invoke_agent"

    # One execution → one trace id; derived work stays under the turn.
    for name in ("tool.search_docs", "chat.generation"):
        assert by_name[name].context.trace_id == turn.context.trace_id
        assert by_name[name].parent.span_id == turn.context.span_id

    tool = by_name["tool.search_docs"]
    assert tool.attributes["gen_ai.tool.name"] == "search_docs"
    assert tool.attributes["gen_ai.tool.status"] == "ok"
    # Raw tool output must not be recorded verbatim.
    assert "raw tool output" not in str(dict(tool.attributes))


def test_turn_failure_sets_error_status(spans, monkeypatch):
    with pytest.raises(kernel_mod.KernelModelFailure):
        _run(monkeypatch, _FakeAgent(fail=True))

    turn = [s for s in spans.get_finished_spans() if s.name == "agent.turn"][0]
    assert turn.status.status_code.name == "ERROR"
    assert turn.attributes["error.type"] == "KernelModelFailure"
