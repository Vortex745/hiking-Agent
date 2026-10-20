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
    monkeypatch.setattr(tracing, "_init_failed", False)
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


# ---- Ticket 06: optional / degradable tracing ------------------------------


def _enable(monkeypatch, *, enabled: bool = True) -> None:
    monkeypatch.setattr(tracing, "_initialized", False)
    monkeypatch.setattr(tracing, "_init_failed", False)
    monkeypatch.setattr(tracing.settings, "langfuse_enabled", enabled)
    monkeypatch.setattr(tracing.settings, "langfuse_public_key", "pk-test")
    monkeypatch.setattr(tracing.settings, "langfuse_secret_key", "sk-test")


def test_switch_off_builds_no_exporter(monkeypatch):
    from observability.export import langfuse as adapter

    _enable(monkeypatch, enabled=False)

    def _boom():  # pragma: no cover - only fails the test if called
        raise AssertionError("exporter must not be built when the switch is off")

    monkeypatch.setattr(adapter, "build_exporter", _boom)

    assert tracing.get_tracer() is None
    with tracing.start_span("chat.request") as span:
        span.set_attribute("run.id", "run-1")
    assert tracing.current_trace_id() == ""


def test_init_failure_degrades_to_noop(monkeypatch):
    from observability.export import langfuse as adapter

    _enable(monkeypatch)

    def _boom():
        raise RuntimeError("langfuse unreachable")

    monkeypatch.setattr(adapter, "build_exporter", _boom)

    assert tracing.get_tracer() is None
    # A second call must not retry initialization or raise.
    assert tracing.get_tracer() is None
    with tracing.start_span("chat.request") as span:
        span.set_attribute("run.id", "run-1")
    assert tracing.force_flush(timeout_millis=100) is True


def test_provider_uses_batch_processor_and_registers_exit_flush(monkeypatch):
    import opentelemetry.trace as otel_trace
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from observability.export import langfuse as adapter

    _enable(monkeypatch)
    registered = []
    monkeypatch.setattr(adapter, "build_exporter", InMemorySpanExporter)
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda _p: None)
    monkeypatch.setattr(tracing.atexit, "register", registered.append)

    assert tracing.get_tracer() is not None
    processor = tracing._provider._active_span_processor
    registered_processors = getattr(processor, "_span_processors", processor)
    if isinstance(registered_processors, dict):
        registered_processors = registered_processors.values()
    active = list(registered_processors)
    assert any(isinstance(p, BatchSpanProcessor) for p in active)
    assert not any(isinstance(p, SimpleSpanProcessor) for p in active)
    assert tracing.shutdown_tracing in registered

    tracing.shutdown_tracing()


class _FailingExporter:
    """Span exporter that always raises, simulating Langfuse being down."""

    def export(self, spans):
        raise RuntimeError("langfuse down")

    def shutdown(self) -> None:
        return None


def test_export_failure_never_reaches_business_code(monkeypatch, caplog):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(_FailingExporter(), schedule_delay_millis=60_000)
    )
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("test"))
    monkeypatch.setattr(tracing, "_provider", provider)

    with caplog.at_level("ERROR"):
        with tracing.start_span("chat.request") as span:
            span.set_attribute("run.id", "run-1")

        # Flush must not raise; the export failure is only logged.
        tracing.force_flush(timeout_millis=500)

    assert any(
        "Exception while exporting" in record.message
        for record in caplog.records
    )
    # Business code keeps working after the failed export.
    with tracing.start_span("chat.request"):
        assert tracing.current_trace_id() != ""
    tracing.shutdown_tracing()


def _sampling_provider(ratio: float):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=ParentBased(TraceIdRatioBased(ratio)))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


@pytest.mark.parametrize("ratio,expected", [(0.0, 0), (1.0, 2)])
def test_sampling_ratio_extremes(ratio, expected):
    provider, exporter = _sampling_provider(ratio)
    tracer = provider.get_tracer("test")
    for index in range(2):
        with tracer.start_as_current_span(f"span-{index}"):
            pass
    assert len(exporter.get_finished_spans()) == expected


def test_sampled_parent_keeps_children(monkeypatch):
    """ParentBased: a dropped root decision never orphans sampled children."""
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    provider, exporter = _sampling_provider(0.0)
    tracer = provider.get_tracer("test")

    parent = SpanContext(
        trace_id=0x1234567890ABCDEF1234567890ABCDEF,
        span_id=0x1234567890ABCDEF,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    context = otel_trace.set_span_in_context(NonRecordingSpan(parent))
    with tracer.start_as_current_span("child", context=context):
        pass

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["child"]
    assert spans[0].context.trace_id == parent.trace_id
    assert spans[0].parent.span_id == parent.span_id

