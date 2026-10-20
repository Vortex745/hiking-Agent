"""Tracing facade used by business code.

Business code depends only on this module (plus the neutral ``gen_ai.*``
keys). OpenTelemetry performs collection, context propagation, batching and
OTLP transport; the backend-specific dialect lives exclusively in
``observability.export.langfuse``.

Tracing is optional: when the switch is closed, credentials are missing, or
initialization fails, every helper degrades to a no-op and chat keeps working.
"""

from __future__ import annotations

import atexit
import asyncio
import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from config import settings

logger = logging.getLogger("ai-service.observability")

_session_id: ContextVar[str | None] = ContextVar("observability_session_id", default=None)

_tracer: Any | None = None
_provider: Any | None = None
_initialized = False
_shutdown = False
_init_failed = False
# Initialization happens on the first request thread; concurrent callers must
# not each build an exporter/provider.
_init_lock = threading.Lock()


class _NullSpan:
    """Minimal span stand-in used when tracing is off."""

    def set_attribute(self, _key: str, _value: Any) -> None:
        return None

    def set_attributes(self, _attributes: dict) -> None:
        return None

    def add_event(self, _name: str, _attributes: dict | None = None) -> None:
        return None

    def set_status(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def record_exception(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def update_name(self, _name: str) -> None:
        return None

    def get_span_context(self) -> Any:
        return None

    def end(self) -> None:
        return None


NULL_SPAN = _NullSpan()


def set_session(session_id: str | None):
    """Bind the logical session (Langfuse session) for the current context."""
    return _session_id.set(session_id)


def reset_session(token) -> None:
    _session_id.reset(token)


def current_session() -> str | None:
    return _session_id.get()


def tracing_enabled() -> bool:
    return bool(
        getattr(settings, "langfuse_enabled", False)
        and getattr(settings, "langfuse_public_key", "")
        and getattr(settings, "langfuse_secret_key", "")
    )


def _sample_ratio() -> float:
    try:
        ratio = float(getattr(settings, "observability_sample_ratio", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return min(max(ratio, 0.0), 1.0)


def _ensure_provider() -> None:
    """Initialize the OTel provider once, if tracing is configured."""
    global _tracer, _provider, _initialized, _init_failed
    if _initialized or _init_failed or _shutdown:
        return
    with _init_lock:
        # Double check: the first thread may have finished while we waited.
        if _initialized or _init_failed or _shutdown:
            return
        if not tracing_enabled():
            _initialized = True
            return
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

            from observability.export.langfuse import RESOURCE_ATTRIBUTES, build_exporter
            from observability.export.redaction import RedactingSpanProcessor

            resource = Resource.create(
                {
                    **RESOURCE_ATTRIBUTES,
                    "service.version": getattr(settings, "app_version", "0.1.0"),
                    "deployment.environment": getattr(settings, "environment", "dev"),
                }
            )
            provider = TracerProvider(
                resource=resource,
                sampler=ParentBased(TraceIdRatioBased(_sample_ratio())),
            )
            # Redaction must run before the batch processor queues the span, so
            # the exporter never observes a raw cookie/address/token value.
            provider.add_span_processor(RedactingSpanProcessor())
            # BatchSpanProcessor exports off the request path; failures are logged
            # by OTel and never raised into business code.
            provider.add_span_processor(
                BatchSpanProcessor(
                    build_exporter(),
                    max_queue_size=2048,
                    schedule_delay_millis=1000,
                    max_export_batch_size=512,
                    export_timeout_millis=5000,
                )
            )
            trace.set_tracer_provider(provider)
            _provider = provider
            _tracer = provider.get_tracer("ai-service")
            _initialized = True
            atexit.register(shutdown_tracing)
            logger.info("tracing enabled (sample_ratio=%s)", _sample_ratio())
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning(
                "tracing initialization failed, continuing disabled: %s", exc
            )
            _tracer = None
            _provider = None
            _init_failed = True


def get_tracer() -> Any | None:
    if _shutdown:
        return None
    _ensure_provider()
    if _shutdown or _init_failed:
        return None
    if _tracer is not None:
        return _tracer
    if getattr(settings, "langfuse_enabled", False) and tracing_enabled():
        from opentelemetry import trace

        return trace.get_tracer("ai-service")
    return None


def attributes(extra: dict | None = None) -> dict:
    """Merge neutral attributes with session grouping from the adapter."""
    merged = {key: value for key, value in (extra or {}).items() if value is not None}
    session = _session_id.get()
    if session:
        from observability.export.langfuse import session_attributes

        merged.update(session_attributes(session))
    return merged


def generation_attributes(**fields: Any) -> dict:
    """Neutral entry point for generation fields; dialect stays in the adapter."""
    from observability.export.langfuse import generation_attributes as build

    return build(**fields)


def observation_attributes(kind: str) -> dict:
    from observability.export.langfuse import observation_attributes as build

    return build(kind)


def _publish_trace_id(span: Any) -> None:
    """Put the live trace id on the logging context.

    The request summary line is written after the span ends, so the value is
    deliberately not reset: logs emitted for this request keep correlating
    with the exported trace even when the gateway sent no traceparent.
    """
    try:
        span_context = span.get_span_context()
        if not getattr(span_context, "is_valid", False):
            return None
        from observability.logging_setup import trace_id_var

        trace_id_var.set(format(span_context.trace_id, "032x"))
    except Exception:  # pragma: no cover - defensive
        return None


@contextmanager
def start_span(
    name: str,
    attrs: dict | None = None,
    *,
    root: bool = False,
    context: Any | None = None,
) -> Iterator[Any]:
    """Start a span as the current span and end it on exit.

    ``root=True`` detaches any parent (used by resume, which must always be a
    new trace). When tracing is disabled this yields a no-op span.
    """
    tracer = get_tracer()
    if tracer is None:
        yield NULL_SPAN
        return

    span_context = context
    if span_context is None and root:
        from opentelemetry.context import Context

        span_context = Context()

    try:
        if span_context is not None:
            span = tracer.start_span(
                name, context=span_context, attributes=attributes(attrs)
            )
        else:
            span = tracer.start_span(name, attributes=attributes(attrs))
    except Exception as exc:  # pragma: no cover - exporter/backend failures
        logger.warning("span %s could not start: %s", name, exc)
        yield NULL_SPAN
        return

    from opentelemetry import trace as otel_trace

    _publish_trace_id(span)

    # record_exception=False: business exceptions are sanitized by the caller
    # via ``record_exception``; raw stack traces may contain secrets.
    with otel_trace.use_span(
        span, end_on_exit=True, record_exception=False, set_status_on_exception=False
    ):
        yield span


def current_span() -> Any:
    from opentelemetry import trace

    return trace.get_current_span()


def current_trace_id() -> str:
    span = current_span()
    context = span.get_span_context() if hasattr(span, "get_span_context") else None
    if context is None or not getattr(context, "is_valid", False):
        return ""
    return format(context.trace_id, "032x")


def record_exception(span: Any, exc: BaseException) -> None:
    """Mark the span failed without exporting raw (possibly secret) text."""
    # A closed stream (browser stop/disconnect) is a normal end, not a
    # failure: the span keeps its UNSET status and already-collected usage.
    if isinstance(exc, (GeneratorExit, asyncio.CancelledError)):
        return None
    from observability.redaction import sanitize_error

    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_attribute("error.type", type(exc).__name__)
        span.set_attribute("error.message", sanitize_error(exc))
        span.set_status(Status(StatusCode.ERROR))
    except Exception:  # pragma: no cover - defensive
        return None


def mark_ok(span: Any) -> None:
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.OK))
    except Exception:  # pragma: no cover - defensive
        return None


def extract_traceparent(header: str | None) -> Any | None:
    """Return an OTel context carrying a valid W3C traceparent, else None."""
    if not header:
        return None
    from opentelemetry import trace
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    context = TraceContextTextMapPropagator().extract({"traceparent": header})
    remote = trace.get_current_span(context).get_span_context()
    if remote is None or not getattr(remote, "is_valid", False):
        return None
    # The sampling decision belongs to this service: a client sending
    # ``-00`` only means "I did not sample", not "do not trace me".
    local = SpanContext(
        trace_id=remote.trace_id,
        span_id=remote.span_id,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        trace_state=remote.trace_state,
    )
    return trace.set_span_in_context(NonRecordingSpan(local), context)


def force_flush(timeout_millis: int = 5000) -> bool:
    if _provider is None:
        return True
    try:
        return bool(_provider.force_flush(timeout_millis=timeout_millis))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("tracing flush failed: %s", exc)
        return False


def shutdown_tracing() -> None:
    """Flush and close the provider. Safe to call repeatedly."""
    global _shutdown, _provider
    if _shutdown:
        return
    _shutdown = True
    provider = _provider
    if provider is None:
        return
    try:
        provider.force_flush(timeout_millis=5000)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("tracing flush on shutdown failed: %s", exc)
    try:
        provider.shutdown()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("tracing shutdown failed: %s", exc)
