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
async def test_astream_handles_chat_generation_chunks(spans):
    """Production ``_astream`` yields ChatGenerationChunk, not AIMessageChunk."""
    from langchain_core.outputs import ChatGenerationChunk

    class _ProductionShapedModel:
        model_name = "gpt-test"
        temperature = 0
        _llm_type = "openai"

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    usage_metadata=None,
                )
            )
            yield ChatGenerationChunk(message=AIMessageChunk(content="hel"))
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="lo",
                    usage_metadata={
                        "input_tokens": 12,
                        "output_tokens": 8,
                        "total_tokens": 20,
                    },
                )
            )

    model = trace_chat_model(_ProductionShapedModel())
    chunks = [chunk async for chunk in model._astream([HumanMessage(content="hi")])]

    assert "".join(chunk.message.content for chunk in chunks) == "hello"
    attrs = dict(spans.get_finished_spans()[0].attributes)
    assert attrs["gen_ai.usage.input_tokens"] == 12
    assert attrs["gen_ai.usage.output_tokens"] == 8
    assert attrs["gen_ai.output.messages"] == "hello"
    assert "gen_ai.response.time_to_first_token_ms" in attrs


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


def test_authorized_location_is_redacted_from_free_text():
    from observability.redaction import scrub_text

    prompt = (
        "用户授权的当前位置上下文：province=浙江省，city=杭州市，"
        "district=西湖区，adcode=330106，address=文三路 123 号 5 幢，"
        "source=browser。坐标 latitude=30.123456，longitude=120.654321。"
    )
    cleaned = scrub_text(prompt)

    assert "文三路" not in cleaned
    assert "123 号" not in cleaned
    assert "30.123456" not in cleaned
    assert "120.654321" not in cleaned
    assert "330106" not in cleaned
    assert "province=浙江省" in cleaned


def test_sensitive_input_is_redacted(spans):
    import asyncio

    model = trace_chat_model(_FakeModel())
    asyncio.run(
        model._agenerate([HumanMessage(content="Authorization: Bearer super-secret")])
    )
    attrs = dict(spans.get_finished_spans()[0].attributes)
    assert "super-secret" not in attrs["gen_ai.input.messages"]


@pytest.mark.asyncio
async def test_client_disconnect_is_not_an_error_and_keeps_usage(spans):
    model = trace_chat_model(_FakeModel())
    stream = model._astream([HumanMessage(content="hi")])
    first = await stream.__anext__()
    assert first.content == "he"
    # Browser stop / connection drop closes the generator early.
    await stream.aclose()

    finished = spans.get_finished_spans()
    assert len(finished) == 1
    attrs = dict(finished[0].attributes)
    assert finished[0].status.status_code.name != "ERROR"
    assert "error.type" not in attrs
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 0
    assert attrs["gen_ai.output.messages"] == "he"


def test_sync_stream_close_is_not_an_error(spans):
    class _SyncStreamModel:
        model_name = "gpt-test"
        temperature = 0
        _llm_type = "openai"

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            yield AIMessageChunk(
                content="he",
                usage_metadata={
                    "input_tokens": 4,
                    "output_tokens": 1,
                    "total_tokens": 5,
                },
            )
            yield AIMessageChunk(content="llo")

    model = trace_chat_model(_SyncStreamModel())
    stream = model._stream([HumanMessage(content="hi")])
    assert next(stream).content == "he"
    stream.close()

    finished = spans.get_finished_spans()
    assert len(finished) == 1
    attrs = dict(finished[0].attributes)
    assert finished[0].status.status_code.name != "ERROR"
    assert "error.type" not in attrs
    assert attrs["gen_ai.usage.input_tokens"] == 4
    assert attrs["gen_ai.output.messages"] == "he"


def test_export_redacts_span_attributes_before_they_leave_the_process():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from observability.export.redaction import RedactingSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(RedactingSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("chat.generation") as span:
        span.set_attribute("cookie", "session=abc123")
        span.set_attribute("address", "文三路 123 号")
        span.set_attribute("note", "Authorization: Bearer supersecret-token")
        span.set_attribute("gen_ai.usage.input_tokens", 7)

    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["cookie"] == "[REDACTED]"
    assert attrs["address"] == "[REDACTED]"
    assert "supersecret-token" not in attrs["note"]
    assert attrs["gen_ai.usage.input_tokens"] == 7


def test_tool_messages_are_not_copied_into_generation_input(spans):
    import asyncio
    import json

    from langchain_core.messages import ToolMessage

    document = "完整检索文档：西湖十景包括苏堤春晓与曲院风荷。"
    model = trace_chat_model(_FakeModel())
    asyncio.run(
        model._agenerate(
            [
                HumanMessage(content="hi"),
                ToolMessage(content=document, tool_call_id="call-1"),
            ]
        )
    )

    raw = dict(spans.get_finished_spans()[0].attributes)["gen_ai.input.messages"]
    assert document not in raw
    messages = json.loads(raw)
    tool = [message for message in messages if message["role"] == "tool"][0]
    assert tool["content"] == "[REDACTED]"
    assert tool["content.length"] == len(document)


def test_denied_key_matching_uses_suffixes_without_hitting_metrics():
    from observability.redaction import is_denied_key

    assert is_denied_key("cookie")
    assert is_denied_key("Authorization")
    assert is_denied_key("user_latitude")
    assert is_denied_key("user-address")
    assert is_denied_key("memory_body")
    assert is_denied_key("session_token")
    assert not is_denied_key("latency")
    assert not is_denied_key("token_count")
    assert not is_denied_key("input_tokens")


def test_structured_json_text_is_redacted_key_by_key():
    from observability.redaction import scrub_text

    cleaned = scrub_text('{"lat": 30.1, "city": "杭州", "adcode": "330106"}')

    assert "30.1" not in cleaned
    assert "330106" not in cleaned
    assert "杭州" in cleaned


def test_key_value_credentials_and_locations_are_redacted():
    from observability.redaction import scrub_text

    assert "30.1" not in scrub_text("user_latitude=30.1")
    assert "dXNlcjpwYXNz" not in scrub_text("Authorization: Basic dXNlcjpwYXNz")
    assert "abc" not in scrub_text("Cookie: session=abc")
    assert "sk-live-9f8e7d6c" not in scrub_text("X-Api-Key: sk-live-9f8e7d6c")
    assert "tok-123456" not in scrub_text("access_token=tok-123456")
    assert "30.1" not in scrub_text('attrs {"lat": 30.1} trailing')
    # Metric names that merely contain a denied word stay readable.
    assert scrub_text("latency=5") == "latency=5"
    assert scrub_text("token_count=3") == "token_count=3"
