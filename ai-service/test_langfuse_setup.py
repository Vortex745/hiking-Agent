"""Tracing facade contract tests (ticket 01).

The SDK callback ingestion path is gone; tracing is OpenTelemetry-based and
must stay optional.
"""

import asyncio

import pytest

from observability import tracing


def _tracer_provider():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


@pytest.fixture(autouse=True)
def _reset_tracing(monkeypatch):
    monkeypatch.setattr(tracing, "_tracer", None)
    monkeypatch.setattr(tracing, "_provider", None)
    monkeypatch.setattr(tracing, "_initialized", True)
    monkeypatch.setattr(tracing, "_shutdown", False)
    yield


def test_disabled_when_keys_missing(monkeypatch):
    monkeypatch.setattr(tracing.settings, "langfuse_enabled", True)
    monkeypatch.setattr(tracing.settings, "langfuse_public_key", "")
    monkeypatch.setattr(tracing.settings, "langfuse_secret_key", "")
    assert tracing.tracing_enabled() is False
    assert tracing.get_tracer() is None
    with tracing.start_span("chat.request") as span:
        span.set_attribute("run.id", "run-1")
    assert tracing.current_trace_id() == ""


def test_disabled_switch_wins_even_with_keys(monkeypatch):
    monkeypatch.setattr(tracing.settings, "langfuse_enabled", False)
    monkeypatch.setattr(tracing.settings, "langfuse_public_key", "pk-test")
    monkeypatch.setattr(tracing.settings, "langfuse_secret_key", "sk-test")
    assert tracing.tracing_enabled() is False
    assert tracing.get_tracer() is None


def test_session_context_scope():
    assert tracing.current_session() is None
    token = tracing.set_session("chat-1")
    try:
        assert tracing.current_session() == "chat-1"
    finally:
        tracing.reset_session(token)
    assert tracing.current_session() is None


def test_root_span_gets_new_trace(monkeypatch):
    provider, _exporter = _tracer_provider()
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("test"))

    with tracing.start_span("parent"):
        parent_trace = tracing.current_trace_id()
        with tracing.start_span("chat.resume", root=True):
            assert tracing.current_trace_id() != parent_trace
    assert parent_trace


def test_flush_exports_batched_spans_and_shutdown_is_idempotent(monkeypatch):
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider, _ = _tracer_provider()
    # Replace the simple processor with a batch processor that only exports
    # on flush, reproducing the "last batch is lost without a flush" case.
    provider._active_span_processor = (
        provider._active_span_processor.__class__()
    )
    provider.add_span_processor(
        BatchSpanProcessor(exporter, schedule_delay_millis=60_000)
    )
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("test"))
    monkeypatch.setattr(tracing, "_provider", provider)

    with tracing.start_span("chat.request") as span:
        span.set_attribute("run.id", "run-1")

    assert exporter.get_finished_spans() == ()
    assert tracing.force_flush(timeout_millis=2000) is True
    names = [span.name for span in exporter.get_finished_spans()]
    assert "chat.request" in names

    tracing.shutdown_tracing()
    tracing.shutdown_tracing()  # repeated close must not raise


def test_exporter_adapter_builds_langfuse_endpoint(monkeypatch):
    from observability.export import langfuse as adapter

    monkeypatch.setattr(tracing.settings, "langfuse_base_url", "http://localhost:3000/")
    monkeypatch.setattr(tracing.settings, "langfuse_public_key", "pk-test")
    monkeypatch.setattr(tracing.settings, "langfuse_secret_key", "sk-test")

    assert adapter.endpoint() == "http://localhost:3000/api/public/otel/v1/traces"
    assert adapter.auth_header().startswith("Basic ")
    assert adapter.session_attributes("chat-1") == {"langfuse.session.id": "chat-1"}
    assert adapter.observation_attributes("generation") == {
        "langfuse.observation.type": "generation"
    }
