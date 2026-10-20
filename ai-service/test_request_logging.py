"""Request correlation + structured logging contract tests (ticket 02)."""

import asyncio
import logging
import re

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

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


def build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    @app.get("/stream")
    async def stream():
        async def gen():
            yield b"one"
            yield b"two"

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


def test_valid_request_id_is_preserved_and_echoed(client):
    incoming = "20261020T220000000-9"
    response = client.get("/ping", headers={"X-Request-ID": incoming})
    assert response.headers["X-Request-ID"] == incoming


def test_missing_or_invalid_request_id_is_generated(client):
    response = client.get("/ping", headers={"X-Request-ID": "bad id!"})
    request_id = response.headers["X-Request-ID"]
    assert request_id != "bad id!"
    assert REQUEST_ID_RE.match(request_id)


def test_summary_log_has_pinned_keys(client, caplog):
    with caplog.at_level(logging.INFO, logger="ai-service.http"):
        client.get("/ping", headers={"X-Request-ID": "20261020T220000000-11"})
    records = [r for r in caplog.records if r.name == "ai-service.http"]
    assert records
    summary = records[-1]
    assert summary.event == "request.completed"
    assert summary.request_id == "20261020T220000000-11"
    assert summary.route == "/ping"
    assert summary.status == 200
    assert isinstance(summary.duration_ms, float)
    assert summary.trace_id == ""


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


def test_json_formatter_emits_pinned_keys():
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
        payload = __import__("json").loads(formatter.format(record))
    finally:
        request_id_var.reset(token)
    for key in (
        "ts",
        "level",
        "event",
        "msg",
        "request_id",
        "trace_id",
        "route",
        "status",
        "duration_ms",
    ):
        assert key in payload
    assert set(payload).issuperset({"ts", "level", "event", "msg"})


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
