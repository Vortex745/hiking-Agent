"""Generation-node decorator for chat models (ticket 03).

The product constructs exactly one chat model (``build_chat_model``). Wrapping
that instance keeps every inference call — normal, streaming, structured
output and tool-bound — under a single ``chat.generation`` span without
relying on LangChain callbacks.

The wrapper is idempotent: wrapping twice does not create two generation
nodes. When no span context is active the span naturally becomes the root.
"""

from __future__ import annotations

import json
import time
from typing import Any

from observability import tracing
from observability.redaction import sanitize_messages, scrub_text

_MAX_TEXT = 4000


def _model_name(model: Any) -> str:
    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or "unknown"
    )


def _provider(model: Any) -> str:
    return str(getattr(model, "_llm_type", None) or "openai")


def _tool_names(tools: Any) -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        if isinstance(tool, dict):
            function = tool.get("function") or {}
            name = function.get("name") or tool.get("name")
        else:
            name = getattr(tool, "name", None)
        if name:
            names.append(str(name))
    return names


def _request_attributes(
    model: Any, messages: Any, kwargs: dict
) -> dict:
    attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.input.messages": scrub_text(
            json.dumps(sanitize_messages(messages), ensure_ascii=False), _MAX_TEXT
        ),
    }
    attrs.update(
        tracing.generation_attributes(
            model=_model_name(model),
            provider=_provider(model),
            temperature=getattr(model, "temperature", None),
            tools=_tool_names(kwargs.get("tools")),
        )
    )
    return attrs


def _first_generation(result: Any) -> Any | None:
    generations = getattr(result, "generations", None) or []
    if not generations:
        return None
    generation = generations[0]
    if isinstance(generation, (list, tuple)):
        generation = generation[0] if generation else None
    return generation


def _usage_from_result(result: Any) -> tuple[int | None, int | None]:
    llm_output = getattr(result, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    if input_tokens is None and output_tokens is None:
        message = getattr(_first_generation(result), "message", None)
        metadata = getattr(message, "usage_metadata", None) or {}
        input_tokens = metadata.get("input_tokens")
        output_tokens = metadata.get("output_tokens")
    return input_tokens, output_tokens


def _text_from_result(result: Any) -> str:
    generation = _first_generation(result)
    if generation is None:
        return ""
    text = getattr(generation, "text", None)
    if text is None:
        text = getattr(getattr(generation, "message", None), "content", "")
    return scrub_text(text or "", _MAX_TEXT)


def _apply_usage(
    span: Any, input_tokens: int | None, output_tokens: int | None
) -> None:
    if input_tokens is None and output_tokens is None:
        span.set_attribute("gen_ai.usage.details", "missing")
        return
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))


def _usage_from_chunk(chunk: Any) -> tuple[int | None, int | None]:
    # Streaming implementations yield ChatGenerationChunk in production
    # (usage lives on ``chunk.message``); lightweight fakes yield the
    # message chunk directly.
    carriers = (chunk, getattr(chunk, "message", None))
    metadata: dict = {}
    response_metadata: dict = {}
    for carrier in carriers:
        if carrier is None:
            continue
        metadata = getattr(carrier, "usage_metadata", None) or {}
        response_metadata = getattr(carrier, "response_metadata", None) or {}
        if metadata:
            break
    input_tokens = metadata.get("input_tokens")
    output_tokens = metadata.get("output_tokens")
    if input_tokens is None and output_tokens is None:
        usage = response_metadata.get("token_usage") or {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
    return input_tokens, output_tokens


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", None)
    if content is None:
        content = getattr(getattr(chunk, "message", None), "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content or "")


def trace_chat_model(model: Any) -> Any:
    """Wrap a chat model instance so each inference call emits a span."""
    if model is None or getattr(model, "_obs_generation_traced", False):
        return model

    original_agenerate = getattr(model, "_agenerate", None)
    original_generate = getattr(model, "_generate", None)
    original_astream = getattr(model, "_astream", None)
    original_stream = getattr(model, "_stream", None)

    if original_agenerate is not None:

        async def traced_agenerate(messages, stop=None, run_manager=None, **kwargs):
            attrs = _request_attributes(model, messages, kwargs)
            with tracing.start_span("chat.generation", attrs) as span:
                try:
                    result = await original_agenerate(
                        messages, stop=stop, run_manager=run_manager, **kwargs
                    )
                except BaseException as exc:
                    tracing.record_exception(span, exc)
                    raise
                input_tokens, output_tokens = _usage_from_result(result)
                _apply_usage(span, input_tokens, output_tokens)
                span.set_attribute("gen_ai.output.messages", _text_from_result(result))
                tracing.mark_ok(span)
                return result

        model._agenerate = traced_agenerate

    if original_generate is not None:

        def traced_generate(messages, stop=None, run_manager=None, **kwargs):
            attrs = _request_attributes(model, messages, kwargs)
            with tracing.start_span("chat.generation", attrs) as span:
                try:
                    result = original_generate(
                        messages, stop=stop, run_manager=run_manager, **kwargs
                    )
                except BaseException as exc:
                    tracing.record_exception(span, exc)
                    raise
                input_tokens, output_tokens = _usage_from_result(result)
                _apply_usage(span, input_tokens, output_tokens)
                span.set_attribute("gen_ai.output.messages", _text_from_result(result))
                tracing.mark_ok(span)
                return result

        model._generate = traced_generate

    if original_astream is not None:

        async def traced_astream(messages, stop=None, run_manager=None, **kwargs):
            attrs = _request_attributes(model, messages, kwargs)
            started = time.perf_counter()
            chunks: list[str] = []
            input_tokens = output_tokens = None
            first_token_ms = None
            with tracing.start_span("chat.generation", attrs) as span:
                try:
                    async for chunk in original_astream(
                        messages, stop=stop, run_manager=run_manager, **kwargs
                    ):
                        text = _chunk_text(chunk)
                        if text and first_token_ms is None:
                            first_token_ms = round(
                                (time.perf_counter() - started) * 1000, 1
                            )
                            span.set_attribute(
                                "gen_ai.response.time_to_first_token_ms",
                                first_token_ms,
                            )
                        if len("".join(chunks)) < _MAX_TEXT:
                            chunks.append(text)
                        chunk_input, chunk_output = _usage_from_chunk(chunk)
                        input_tokens = chunk_input if chunk_input is not None else input_tokens
                        output_tokens = (
                            chunk_output if chunk_output is not None else output_tokens
                        )
                        yield chunk
                except BaseException as exc:
                    tracing.record_exception(span, exc)
                    raise
                _apply_usage(span, input_tokens, output_tokens)
                span.set_attribute(
                    "gen_ai.output.messages", scrub_text("".join(chunks), _MAX_TEXT)
                )
                tracing.mark_ok(span)

        model._astream = traced_astream

    if original_stream is not None:

        def traced_stream(messages, stop=None, run_manager=None, **kwargs):
            attrs = _request_attributes(model, messages, kwargs)
            started = time.perf_counter()
            chunks: list[str] = []
            input_tokens = output_tokens = None
            first_token_ms = None
            with tracing.start_span("chat.generation", attrs) as span:
                try:
                    for chunk in original_stream(
                        messages, stop=stop, run_manager=run_manager, **kwargs
                    ):
                        text = _chunk_text(chunk)
                        if text and first_token_ms is None:
                            first_token_ms = round(
                                (time.perf_counter() - started) * 1000, 1
                            )
                            span.set_attribute(
                                "gen_ai.response.time_to_first_token_ms",
                                first_token_ms,
                            )
                        if len("".join(chunks)) < _MAX_TEXT:
                            chunks.append(text)
                        chunk_input, chunk_output = _usage_from_chunk(chunk)
                        input_tokens = chunk_input if chunk_input is not None else input_tokens
                        output_tokens = (
                            chunk_output if chunk_output is not None else output_tokens
                        )
                        yield chunk
                except BaseException as exc:
                    tracing.record_exception(span, exc)
                    raise
                _apply_usage(span, input_tokens, output_tokens)
                span.set_attribute(
                    "gen_ai.output.messages", scrub_text("".join(chunks), _MAX_TEXT)
                )
                tracing.mark_ok(span)

        model._stream = traced_stream

    model._obs_generation_traced = True
    return model
