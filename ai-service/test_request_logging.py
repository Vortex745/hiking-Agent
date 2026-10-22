"""Request correlation + structured logging contract tests (ticket 02)."""

import asyncio
import logging
import re
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

import api.chat_v2 as chat_v2
from agent_v2.events import EventEnvelope
from observability import tracing
from observability.logging_setup import (
    JsonFormatter,
    RequestContextMiddleware,
    configure_logging,
    request_id_var,
)
from observability.request_context import (
    generate_request_id,
    parse_traceparent,
    valid_request_id,
)

VALID_TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,128}$")


@pytest.fixture(autouse=True)
def isolate_llm_log_sink(monkeypatch):
    """Stop ``configure_logging`` from attaching the ambient LLM_LOG file sink.

    ``ai-service/.env`` ships ``LLM_LOG_ENABLED=true``, so a test that calls
    ``configure_logging`` leaves a live handler on ``ai-service.llm_log``. Files
    scheduled afterwards then see ``llm_log.enabled()`` and the kernel adds a
    tool-log callback they never asked for, while fake-model records land in
    the developer's real ``./logs/llm.log``.
    """
    from config import settings
    from observability import llm_log

    saved = (llm_log._handler, llm_log._setup_done, llm_log._broken)
    monkeypatch.setattr(settings, "llm_log_enabled", False, raising=False)
    yield
    llm_log._handler, llm_log._setup_done, llm_log._broken = saved


def build_app(*, spanned: bool = False) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/ping")
    async def ping():
        if spanned:
            # A handler that emits a real span must share its trace id with
            # the request summary line written by the middleware.
            with tracing.start_span("chat.request"):
                return {"ok": True}
        return {"ok": True}

    @app.get("/stream")
    async def stream():
        async def gen():
            yield b"one"
            yield b"two"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/stream-spanned")
    async def stream_spanned(request: Request):
        async def gen():
            with tracing.start_span("chat.request"):
                # Mirrors chat_v2: the body is iterated in a child task under
                # ASGI < 2.4, so the id must ride on the shared scope state.
                request.state.trace_id = tracing.current_trace_id()
                yield b"one"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


@pytest.fixture()
def client():
    return TestClient(build_app())


def test_generated_request_id_format_and_validation():
    generated = generate_request_id()
    assert REQUEST_ID_RE.match(generated)
    assert valid_request_id(generated) == generated
    assert valid_request_id("short") is None
    assert valid_request_id("bad id!") is None
    assert valid_request_id("x" * 200) is None
    # Trailing content (e.g. a stray newline from a header) must not pass.
    assert valid_request_id("abcdefgh\n") is None
    assert valid_request_id("abcdefgh ") is None


def test_valid_request_id_is_preserved_and_echoed(client):
    incoming = "20261020T220000000-9"
    response = client.get("/ping", headers={"X-Request-ID": incoming})
    assert response.headers["X-Request-ID"] == incoming


def test_missing_or_invalid_request_id_is_generated(client):
    response = client.get("/ping", headers={"X-Request-ID": "bad id!"})
    request_id = response.headers["X-Request-ID"]
    assert request_id != "bad id!"
    assert REQUEST_ID_RE.match(request_id)


def test_summary_log_has_pinned_keys(span_exporter, caplog):
    client = TestClient(build_app(spanned=True))
    with caplog.at_level(logging.INFO, logger="ai-service.http"):
        client.get("/ping", headers={"X-Request-ID": "20261020T220000000-11"})
    records = [r for r in caplog.records if r.name == "ai-service.http"]
    assert records
    summary = records[-1]
    assert summary.event == "request.completed"
    assert summary.request_id == "20261020T220000000-11"
    assert summary.method == "GET"
    assert summary.route == "/ping"
    assert summary.status == 200
    assert isinstance(summary.duration_ms, float)
    # The log line carries the trace id of the span the request produced.
    span_trace_id = format(
        span_exporter.get_finished_spans()[0].context.trace_id, "032x"
    )
    assert summary.trace_id == span_trace_id
    assert summary.trace_id != ""


def test_streaming_span_trace_id_reaches_the_summary_line(span_exporter, caplog):
    client = TestClient(build_app())
    with caplog.at_level(logging.INFO, logger="ai-service.http"):
        response = client.get(
            "/stream-spanned", headers={"X-Request-ID": "20261020T220000000-41"}
        )
    assert response.status_code == 200
    summary = [
        r for r in caplog.records if getattr(r, "event", None) == "request.completed"
    ][-1]
    span_trace_id = format(
        span_exporter.get_finished_spans()[0].context.trace_id, "032x"
    )
    assert span_trace_id
    assert summary.trace_id == span_trace_id


def test_non_chat_request_emits_logs_but_no_trace(span_exporter, caplog):
    """Ticket 01: only chat endpoints create spans; other paths log only."""
    with caplog.at_level(logging.INFO, logger="ai-service.http"):
        TestClient(build_app()).get("/ping")

    assert span_exporter.get_finished_spans() == ()
    assert any(
        getattr(record, "event", None) == "request.completed"
        for record in caplog.records
    )


def test_valid_traceparent_sets_trace_id(client, caplog):
    with caplog.at_level(logging.INFO, logger="ai-service.http"):
        client.get("/ping", headers={"traceparent": VALID_TRACEPARENT})
    summary = [r for r in caplog.records if r.name == "ai-service.http"][-1]
    assert summary.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_invalid_traceparent_is_dropped_and_logged(client, caplog):
    seen = {}

    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/echo")
    async def echo(request: Request):
        seen["traceparent"] = request.headers.get("traceparent")
        return {"ok": True}

    with caplog.at_level(logging.WARNING, logger="ai-service.http"):
        response = TestClient(app).get(
            "/echo",
            headers={
                "traceparent": "01-not-hex",
                "X-Request-ID": "20261020T220000000-13",
            },
        )
    assert response.status_code == 200
    assert seen["traceparent"] is None
    warnings = [r for r in caplog.records if r.event == "traceparent.invalid"]
    assert warnings
    assert warnings[-1].request_id == "20261020T220000000-13"
    assert warnings[-1].trace_id == ""


def test_parse_traceparent_rejects_bad_versions_and_zero_ids():
    assert parse_traceparent(VALID_TRACEPARENT) == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert parse_traceparent("ff-" + "a" * 32 + "-" + "b" * 16 + "-01") is None
    assert parse_traceparent("00-" + "0" * 32 + "-" + "b" * 16 + "-01") is None
    assert parse_traceparent("00-" + "a" * 32 + "-" + "0" * 16 + "-01") is None
    assert parse_traceparent("garbage") is None
    assert parse_traceparent(None) is None


def test_streaming_response_is_not_buffered():
    async def app(scope, receive, send):
        await send(
            {"type": "http.response.start", "status": 200, "headers": []}
        )
        await send(
            {"type": "http.response.body", "body": b"first", "more_body": True}
        )
        await first_chunk_sent.wait()
        await send({"type": "http.response.body", "body": b"second"})

    middleware = RequestContextMiddleware(app)
    first_chunk_sent = asyncio.Event()
    release_second = asyncio.Event()
    received = []

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body") == b"first":
            first_chunk_sent.set()
            # Wait until the test confirms it saw the first chunk; if the
            # middleware buffered, this would deadlock.
            await asyncio.wait_for(release_second.wait(), timeout=2)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/stream",
        "headers": [],
    }

    async def run():
        task = asyncio.create_task(middleware(scope, receive, send))
        await asyncio.wait_for(first_chunk_sent.wait(), timeout=2)
        received.append("first")
        release_second.set()
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(run())
    assert received == ["first"]


def test_json_formatter_envelope_is_pinned_but_event_fields_are_conditional():
    """Envelope keys always exist; request fields appear only when they have a value.

    Forcing ``method/route/status/duration_ms`` onto every line as ``""`` made
    ``duration_ms`` a string on business logs and a number on access logs —
    consumers aggregating it would break.
    """
    import json

    formatter = JsonFormatter()
    record = logging.LogRecord(
        "ai-service.http", logging.INFO, __file__, 1, "hello", None, None
    )
    record.request_id = "req-12345678"
    record.trace_id = "abc"
    record.route = "/x"
    record.status = 200
    record.duration_ms = 1.5
    token = request_id_var.set("req-12345678")
    try:
        payload = json.loads(formatter.format(record))
        bare = json.loads(
            formatter.format(
                logging.LogRecord(
                    "ai-service.demo", logging.INFO, __file__, 2, "tick", None, None
                )
            )
        )
    finally:
        request_id_var.reset(token)

    for key in (
        "ts",
        "level",
        "event",
        "logger",
        "service",
        "schema",
        "caller",
        "msg",
        "request_id",
        "trace_id",
    ):
        assert key in payload
    assert payload["service"] == "ai-service"
    assert payload["schema"] == 1
    assert payload["logger"] == record.name
    assert payload["caller"] == "test_request_logging.py:1[None]"
    assert (payload["route"], payload["status"], payload["duration_ms"]) == (
        "/x",
        200,
        1.5,
    )

    for key in ("method", "route", "status", "duration_ms"):
        assert key not in bare, f"{key} must not be pinned as an empty string"


def test_json_formatter_normalizes_missing_correlation_ids():
    import json

    from observability.logging_setup import trace_id_var

    formatter = JsonFormatter()
    record = logging.LogRecord(
        "httpx2", logging.INFO, __file__, 1, "no request context", None, None
    )
    request_token = request_id_var.set(None)
    trace_token = trace_id_var.set(None)
    try:
        payload = json.loads(formatter.format(record))
    finally:
        request_id_var.reset(request_token)
        trace_id_var.reset(trace_token)
    assert payload["request_id"] == ""
    assert payload["trace_id"] == ""


def test_json_formatter_redacts_denied_extra_fields():
    import json

    formatter = JsonFormatter()
    record = logging.LogRecord(
        "ai-service.http", logging.INFO, __file__, 1, "probe", None, None
    )
    record.cookie = "sessionid=abc123"
    record.address = "文三路 123 号"
    record.nested = {"lat": 30.1, "note": "ok"}
    payload = json.loads(formatter.format(record))
    encoded = json.dumps(payload, ensure_ascii=False)

    assert payload["cookie"] == "[REDACTED]"
    assert payload["address"] == "[REDACTED]"
    nested = json.loads(payload["nested"])
    assert nested == {"lat": "[REDACTED]", "note": "ok"}
    for leaked in ("sessionid=abc123", "文三路", "123 号", "30.1"):
        assert leaked not in encoded


def test_json_formatter_includes_redacted_exception_detail():
    import json

    formatter = JsonFormatter()
    try:
        raise RuntimeError("boom Authorization: Bearer supersecret-token")
    except RuntimeError:
        import sys

        record = logging.LogRecord(
            "ai-service.http", logging.ERROR, __file__, 1, "failed", None,
            sys.exc_info(),
        )
        payload = json.loads(formatter.format(record))

    assert "RuntimeError" in payload["exc_info"]
    assert "supersecret-token" not in payload["exc_info"]


def test_configure_logging_is_idempotent():
    configure_logging()
    configure_logging()
    assert logging.getLogger().handlers


# ---- Ticket 02: one correlation key across logs and traces ------------------


@pytest.fixture()
def span_exporter(monkeypatch):
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
    monkeypatch.setattr(tracing, "_init_failed", False)
    monkeypatch.setattr(tracing, "_provider", provider)
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("test"))
    monkeypatch.setattr(tracing, "tracing_enabled", lambda: True)
    return exporter


def test_middleware_publishes_request_id_to_asgi_state(caplog):
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    observed = {}

    @app.get("/state")
    async def state(request: Request):
        observed["request_id"] = getattr(request.state, "request_id", "")
        return {"ok": True}

    with caplog.at_level(logging.INFO, logger="ai-service.http"):
        TestClient(app).get(
            "/state", headers={"X-Request-ID": "20261020T220000000-21"}
        )

    summary = [
        r for r in caplog.records if getattr(r, "event", None) == "request.completed"
    ][-1]
    assert observed["request_id"] == summary.request_id
    assert observed["request_id"] == "20261020T220000000-21"


class _Leases:
    def release(self) -> None:
        return None


class _StreamingOrchestrator:
    def __init__(self) -> None:
        self.leases = _Leases()

    async def acquire_leases(self, **_kwargs):
        return self.leases

    async def acquire_resume_leases(self, **_kwargs):
        return self.leases

    async def stream_chat(self, *_args, **_kwargs):
        yield "data: done\n\n"

    async def stream_resume(self, **_kwargs):
        yield "data: done\n\n"


def _chat_request(request_id: str):
    app = SimpleNamespace(state=SimpleNamespace(agent_v2_runtime=None))
    return SimpleNamespace(
        app=app, headers={}, state=SimpleNamespace(request_id=request_id)
    )


def _drain(response):
    async def _collect():
        return [frame async for frame in response.body_iterator]

    return asyncio.run(_collect())


def test_chat_v2_sse_span_uses_gateway_request_id(span_exporter, monkeypatch):
    monkeypatch.setattr(
        chat_v2, "_build_run_orchestrator", lambda _runtime: _StreamingOrchestrator()
    )
    monkeypatch.setattr(chat_v2, "_sse_frame", lambda item: item)
    request = _chat_request("20261020T220000000-31")
    req = chat_v2.ChatV2Request(message="hi", request_id="client-body-id")

    response = asyncio.run(chat_v2.chat_v2_sse(req, request))
    _drain(response)

    span = [s for s in span_exporter.get_finished_spans() if s.name == "chat.request"][0]
    assert span.attributes["request.id"] == "20261020T220000000-31"
    assert span.attributes["client.request.id"] == "client-body-id"


def test_chat_v2_resume_span_uses_gateway_request_id(span_exporter, monkeypatch):
    monkeypatch.setattr(
        chat_v2, "_build_run_orchestrator", lambda _runtime: _StreamingOrchestrator()
    )
    monkeypatch.setattr(chat_v2, "_sse_frame", lambda item: item)
    monkeypatch.setattr(
        chat_v2,
        "_get_runs_repo",
        lambda _request: SimpleNamespace(
            get_run=lambda _run_id: SimpleNamespace(
                request_id="client-body-id",
                conversation_id="conv-1",
                status="running",
            )
        ),
    )
    monkeypatch.setattr(chat_v2, "_require_conversation_owner", lambda *a, **k: None)
    request = _chat_request("20261020T220000000-32")

    response = asyncio.run(chat_v2.chat_v2_resume("run-1", request, from_seq=0))
    _drain(response)

    span = [s for s in span_exporter.get_finished_spans() if s.name == "chat.resume"][0]
    assert span.attributes["request.id"] == "20261020T220000000-32"
    assert span.attributes["client.request.id"] == "client-body-id"


def _event_stream_client(monkeypatch, events):
    class Orchestrator(_StreamingOrchestrator):
        async def stream_chat(self, *_args, **_kwargs):
            for event in events:
                yield event

        async def stream_resume(self, **_kwargs):
            for event in events:
                yield event

    monkeypatch.setattr(chat_v2, "_build_run_orchestrator", lambda _runtime: Orchestrator())
    monkeypatch.setattr(
        chat_v2,
        "_get_runs_repo",
        lambda _request: SimpleNamespace(
            get_run=lambda _run_id: SimpleNamespace(
                request_id="client-body-id", conversation_id="conv-1", status="running"
            )
        ),
    )
    monkeypatch.setattr(chat_v2, "_require_conversation_owner", lambda *a, **k: None)
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    app.include_router(chat_v2.chat_v2_router, prefix="/api/v2")
    return TestClient(app)


def _post_event_stream(client, endpoint):
    if endpoint == "sse":
        return client.post(
            "/api/v2/chat/sse",
            json={"message": "hi", "request_id": "client-body-id"},
        )
    return client.post("/api/v2/chat/resume/run-1")


@pytest.mark.parametrize("endpoint", ["sse", "resume"])
@pytest.mark.parametrize(
    "event_type,outcome,status",
    [
        ("error", "error", "ERROR"),
        ("done", "done", "OK"),
        ("cancelled", "cancelled", "UNSET"),
        ("run.paused", "paused", "UNSET"),
        ("text", "unknown", "UNSET"),
    ],
)
def test_stream_outcome_reaches_logs_and_root_span(
    span_exporter, monkeypatch, caplog, endpoint, event_type, outcome, status
):
    payload = {"code": "v2_disabled", "message": "private-error-fixture"}
    event = EventEnvelope(version="2", sequence=1, type=event_type, payload=payload)
    client = _event_stream_client(monkeypatch, [event])

    with caplog.at_level(logging.INFO, logger="ai-service.http"):
        response = _post_event_stream(client, endpoint)

    assert response.status_code == 200
    assert "private-error-fixture" in response.text
    summaries = [r for r in caplog.records if getattr(r, "event", None) == "request.completed"]
    failures = [r for r in caplog.records if getattr(r, "event", None) == "request.failed"]
    assert len(summaries) == 1
    assert summaries[0].status == 200
    assert summaries[0].outcome == outcome
    assert len(failures) == (1 if event_type == "error" else 0)
    span_name = "chat.request" if endpoint == "sse" else "chat.resume"
    span = next(s for s in span_exporter.get_finished_spans() if s.name == span_name)
    assert span.status.status_code.name == status
    assert span.attributes["chat.outcome"] == outcome
    assert summaries[0].trace_id == format(span.context.trace_id, "032x")
    if event_type == "error":
        assert summaries[0].error_code == "v2_disabled"
        assert failures[0].error_code == "v2_disabled"
        assert failures[0].error_type == "StreamError"
        assert span.attributes["error.code"] == "v2_disabled"
    for record in summaries + failures:
        assert "private-error-fixture" not in JsonFormatter().format(record)
    assert "private-error-fixture" not in str(dict(span.attributes))


@pytest.mark.parametrize("endpoint", ["sse", "resume"])
@pytest.mark.parametrize(
    "code", [None, 12, ["bad"], {"bad": "value"}, "private_fixture", "x" * 100]
)
def test_stream_error_code_is_allowlisted(
    span_exporter, monkeypatch, caplog, endpoint, code
):
    event = EventEnvelope(
        version="2", sequence=1, type="error", payload={"code": code}
    )
    client = _event_stream_client(monkeypatch, [event])
    with caplog.at_level(logging.INFO, logger="ai-service.http"):
        response = _post_event_stream(client, endpoint)

    assert response.status_code == 200
    failure = next(r for r in caplog.records if getattr(r, "event", None) == "request.failed")
    assert failure.error_code == "stream_error"
    span = span_exporter.get_finished_spans()[0]
    assert span.attributes["error.code"] == "stream_error"
    assert span.status.status_code.name == "ERROR"


@pytest.mark.parametrize("endpoint", ["sse", "resume"])
def test_replayed_pause_does_not_hide_final_outcome(
    span_exporter, monkeypatch, caplog, endpoint
):
    events = [
        EventEnvelope(version="2", sequence=1, type="run.paused", payload={}),
        EventEnvelope(version="2", sequence=2, type="text", payload={"content": "ok"}),
        EventEnvelope(version="2", sequence=3, type="done", payload={}),
    ]
    client = _event_stream_client(monkeypatch, events)
    with caplog.at_level(logging.INFO, logger="ai-service.http"):
        response = _post_event_stream(client, endpoint)

    assert response.status_code == 200
    summary = next(r for r in caplog.records if getattr(r, "event", None) == "request.completed")
    assert summary.outcome == "done"
    assert span_exporter.get_finished_spans()[0].status.status_code.name == "OK"


@pytest.mark.parametrize("endpoint", ["sse", "resume"])
def test_stream_error_cannot_be_overwritten_by_done(
    span_exporter, monkeypatch, caplog, endpoint
):
    events = [
        EventEnvelope(
            version="2", sequence=1, type="error", payload={"code": "v2_disabled"}
        ),
        EventEnvelope(version="2", sequence=2, type="done", payload={}),
    ]
    client = _event_stream_client(monkeypatch, events)
    with caplog.at_level(logging.INFO, logger="ai-service.http"):
        response = _post_event_stream(client, endpoint)

    assert response.status_code == 200
    failure = next(r for r in caplog.records if getattr(r, "event", None) == "request.failed")
    assert failure.outcome == "error"
    assert failure.error_code == "v2_disabled"
    assert span_exporter.get_finished_spans()[0].status.status_code.name == "ERROR"


@pytest.mark.parametrize("endpoint", ["sse", "resume"])
def test_stream_error_is_logged_with_tracing_disabled(monkeypatch, caplog, endpoint):
    monkeypatch.setattr(tracing, "get_tracer", lambda: None)
    event = EventEnvelope(
        version="2", sequence=1, type="error", payload={"code": "v2_disabled"}
    )
    client = _event_stream_client(monkeypatch, [event])
    with caplog.at_level(logging.INFO, logger="ai-service.http"):
        response = _post_event_stream(client, endpoint)

    assert response.status_code == 200
    failure = next(r for r in caplog.records if getattr(r, "event", None) == "request.failed")
    assert failure.outcome == "error"
    assert failure.error_code == "v2_disabled"
    assert failure.trace_id == ""
