"""Generation-node decorator contract tests (ticket 03)."""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult

from observability import tracing
from observability.generation import trace_chat_model


class _FakeModel:
    model_name = "gpt-test"
    temperature = 0
    _llm_type = "openai"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("upstream failed")
        return LLMResult(
            generations=[
                [
                    ChatGeneration(
                        message=AIMessage(content="ok sk-abcdefgh12345678"),
                        generation_info={},
                    )
                ]
            ],
            llm_output={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}},
        )

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("stream failed")
        yield AIMessageChunk(
            content="he",
            usage_metadata={
                "input_tokens": 7,
                "output_tokens": 0,
                "total_tokens": 7,
            },
        )
        yield AIMessageChunk(
            content="llo",
            usage_metadata={
                "input_tokens": 7,
                "output_tokens": 2,
                "total_tokens": 9,
            },
        )


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
    return exporter


@pytest.mark.asyncio
async def test_agenerate_emits_one_generation_span(spans):
    model = trace_chat_model(_FakeModel())
    await model._agenerate([HumanMessage(content="hello")])

    finished = spans.get_finished_spans()
    assert len(finished) == 1
    span = finished[0]
    assert span.name == "chat.generation"
    attrs = dict(span.attributes)
    assert attrs["gen_ai.request.model"] == "gpt-test"
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.request.temperature"] == 0
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 3
    assert "hello" in attrs["gen_ai.input.messages"]
    assert "sk-abcdefgh12345678" not in attrs["gen_ai.output.messages"]
    assert span.context.trace_id != 0


@pytest.mark.asyncio
async def test_wrapping_twice_does_not_double_trace(spans):
    model = _FakeModel()
    trace_chat_model(model)
    trace_chat_model(model)
    await model._agenerate([HumanMessage(content="hi")])
    assert len(spans.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_tool_bound_invocation_keeps_working_and_is_traced(spans):
    from langchain_openai import ChatOpenAI

    class _CannedChat(ChatOpenAI):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(
                generations=[
                    ChatGeneration(message=AIMessage(content="bound-ok"))
                ]
            )

    model = trace_chat_model(
        _CannedChat(model="gpt-test", api_key="sk-test", max_retries=0)
    )
    bound = model.bind_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    result = await bound.ainvoke([HumanMessage(content="hi")], config={"callbacks": []})

    assert result.content == "bound-ok"
    finished = spans.get_finished_spans()
    assert len(finished) == 1
    attrs = dict(finished[0].attributes)
    assert tuple(attrs["gen_ai.request.tool.names"]) == ("search_docs",)
    assert attrs["gen_ai.output.messages"] == "bound-ok"


@pytest.mark.asyncio
async def test_tool_names_are_attached(spans):
    model = trace_chat_model(_FakeModel())
    await model._agenerate(
        [HumanMessage(content="hi")],
        tools=[
            {"type": "function", "function": {"name": "search_docs"}},
            {"type": "function", "function": {"name": "plan_route"}},
        ],
    )
    attrs = dict(spans.get_finished_spans()[0].attributes)
    assert tuple(attrs["gen_ai.request.tool.names"]) == ("search_docs", "plan_route")


@pytest.mark.asyncio
async def test_astream_records_ttft_usage_and_output(spans):
    model = trace_chat_model(_FakeModel())
    chunks = [chunk async for chunk in model._astream([HumanMessage(content="hi")])]

    assert "".join(chunk.content for chunk in chunks) == "hello"
    finished = spans.get_finished_spans()
    assert len(finished) == 1
    attrs = dict(finished[0].attributes)
    assert attrs["gen_ai.response.time_to_first_token_ms"] >= 0
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 2
    assert attrs["gen_ai.output.messages"] == "hello"


@pytest.mark.asyncio
async def test_error_is_recorded_and_reraised(spans):
    model = trace_chat_model(_FakeModel(fail=True))
    with pytest.raises(RuntimeError):
        await model._agenerate([HumanMessage(content="hi")])

    finished = spans.get_finished_spans()
    assert len(finished) == 1
    attrs = dict(finished[0].attributes)
    assert attrs["error.type"] == "RuntimeError"
    assert finished[0].status.status_code.name == "ERROR"


@pytest.mark.asyncio
async def test_missing_usage_is_marked_not_fabricated(spans, monkeypatch):
    class _NoUsageModel(_FakeModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return LLMResult(
                generations=[[ChatGeneration(message=AIMessage(content="ok"))]]
            )

    model = trace_chat_model(_NoUsageModel())
    await model._agenerate([HumanMessage(content="hi")])
    attrs = dict(spans.get_finished_spans()[0].attributes)
    assert attrs["gen_ai.usage.details"] == "missing"
    assert "gen_ai.usage.input_tokens" not in attrs
    assert "gen_ai.usage.output_tokens" not in attrs


def test_sensitive_input_is_redacted(spans):
    import asyncio

    model = trace_chat_model(_FakeModel())
    asyncio.run(
        model._agenerate([HumanMessage(content="Authorization: Bearer super-secret")])
    )
    attrs = dict(spans.get_finished_spans()[0].attributes)
    assert "super-secret" not in attrs["gen_ai.input.messages"]
