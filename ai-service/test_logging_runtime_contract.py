"""Runtime contract tests for the logging system (gaps left by obs 01-08).

The ticket-level suites drive the middleware through ``caplog`` and never let a
record reach an output stream. These tests pin what decides whether logs are
usable in a running service: the failure summary line, formatter selection,
non-ASCII survivability on a Windows console, and the wiring of the real app.
"""

import asyncio
import contextlib
import io
import json
import logging
import re
import sys

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from observability import logging_setup
from observability import tracing
from observability.logging_setup import (
    JsonFormatter,
    RequestContextMiddleware,
    TextFormatter,
    configure_logging,
    request_id_var,
    trace_id_var,
)

ACCESS_LOGGER = "ai-service.http"
VALID_TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


@pytest.fixture(autouse=True)
def isolate_logging(monkeypatch):
    """configure_logging mutates process-wide logging state; restore it per test.

    ``configure_logging`` also attaches the LLM_LOG file sink, and the ambient
    ``ai-service/.env`` ships ``LLM_LOG_ENABLED=true``. Left alone, that handler
    leaks into every later test file (the kernel then attaches a tool-log
    callback where other tests assert "no callbacks") and writes fake-model
    records into the developer's real ``./logs/llm.log``.
    """
    from config import settings
    from observability import llm_log

    root = logging.getLogger()
    saved = (
        list(root.handlers),
        root.level,
        logging.getLogger("uvicorn.access").disabled,
        logging_setup._configured,
    )
    saved_llm_log = (llm_log._handler, llm_log._setup_done, llm_log._broken)
    quiet = {
        name: logging.getLogger(name).level
        for name in ("httpx", "httpx2", "httpcore")
    }
    monkeypatch.setattr(settings, "llm_log_enabled", False, raising=False)
    logging_setup._configured = False
    yield
    handlers, level, access_disabled, configured = saved
    logging_setup._configured = configured
    root.handlers = handlers
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = access_disabled
    for name, quiet_level in quiet.items():
        logging.getLogger(name).setLevel(quiet_level)
    llm_log._handler, llm_log._setup_done, llm_log._broken = saved_llm_log


def capture_stdout(monkeypatch, log_format=None):
    """Bind logging to a buffer, the way uvicorn binds it to the real stdout."""
    from config import settings

    if log_format is not None:
        monkeypatch.setattr(settings, "log_format", log_format, raising=False)
    sink = io.StringIO()
    monkeypatch.setattr(sys, "stdout", sink)
    configure_logging()
    return sink


def json_lines(sink):
    return [json.loads(line) for line in sink.getvalue().splitlines() if line.strip()]


def summary_lines(sink, event="request.completed"):
    return [payload for payload in json_lines(sink) if payload.get("event") == event]


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture()
def access_logs():
    """Collect access-logger records without depending on the root handler."""
    logger = logging.getLogger(ACCESS_LOGGER)
    handler = _Capture()
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def events(records):
    return [getattr(r, "event", None) for r in records]


def app_client(*, status=None, boom=None):
    """A routed app behind the real middleware."""
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/probe")
    async def probe():
        if boom is not None:
            raise boom
        if status is not None:
            return JSONResponse({"ok": False}, status_code=status)
        return {"ok": True}

    return TestClient(app)


# ---- failure path: the "errors always get an extra line" contract -----------


def test_server_error_response_emits_single_error_summary(access_logs):
    """A bare 5xx has no cause; doubling it as request.failed skews level counts."""
    app_client(status=500).get("/probe", headers={"X-Request-ID": "20261020T230000000-01"})

    assert events(access_logs) == ["request.completed"]
    completed = access_logs[0]
    assert completed.status == 500 and completed.outcome == "error"
    assert completed.levelno == logging.ERROR
    assert completed.request_id == "20261020T230000000-01"


def test_raised_exception_emits_failed_line_and_still_propagates(access_logs):
    client = app_client(boom=ValueError("kernel rejected the payload"))

    with pytest.raises(ValueError):
        client.get("/probe", headers={"X-Request-ID": "20261020T230000000-02"})

    assert events(access_logs) == ["request.completed", "request.failed"]
    completed, failed = access_logs
    assert completed.levelno == logging.ERROR
    assert failed.levelno == logging.ERROR
    assert failed.error_type == "ValueError"
    assert "kernel rejected the payload" in failed.error_message
    # The app never responded, so the status stays at the middleware default.
    assert failed.status == 500


def test_failed_line_redacts_credentials_in_error_message(access_logs):
    client = app_client(boom=RuntimeError("refused api_key=sk-abcdefgh12345 for user"))

    with pytest.raises(RuntimeError):
        client.get("/probe")

    failed = [r for r in access_logs if getattr(r, "event", None) == "request.failed"][0]
    assert "sk-abcdefgh12345" not in failed.error_message
    assert "[REDACTED]" in failed.error_message


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422, 429])
def test_client_error_response_is_classified_separately(access_logs, status):
    app_client(status=status).get("/probe")

    assert events(access_logs) == ["request.completed"]
    assert access_logs[-1].status == status
    assert access_logs[-1].outcome == "client_error"
    assert access_logs[-1].levelno == logging.WARNING


def test_successful_request_emits_no_failed_line(access_logs):
    app_client().get("/probe")

    assert events(access_logs) == ["request.completed"]
    assert access_logs[-1].levelno == logging.INFO


def test_failed_line_reaches_stdout_as_redacted_json(monkeypatch):
    sink = capture_stdout(monkeypatch, "json")
    client = app_client(boom=ValueError("cookie: sessionid=supersecret lat=30.1"))

    with pytest.raises(ValueError):
        client.get("/probe", headers={"X-Request-ID": "20261020T230000000-03"})

    failed = summary_lines(sink, "request.failed")
    assert len(failed) == 1
    assert failed[0]["request_id"] == "20261020T230000000-03"
    assert failed[0]["error_type"] == "ValueError"
    assert "error_message" in failed[0] and "error_msg" not in failed[0]
    dump = json.dumps(failed[0], ensure_ascii=False)
    for leaked in ("supersecret", "sessionid", "30.1"):
        assert leaked not in dump


# ---- formatter selection and stdout plumbing --------------------------------


def configure_with(monkeypatch, *, log_format=None, log_level=None):
    from config import settings

    if log_format is not None:
        monkeypatch.setattr(settings, "log_format", log_format, raising=False)
    if log_level is not None:
        monkeypatch.setattr(settings, "log_level", log_level, raising=False)
    configure_logging()
    return logging.getLogger().handlers[0]


def test_configure_logging_selects_text_formatter_when_log_format_text(monkeypatch):
    assert isinstance(configure_with(monkeypatch, log_format="text").formatter, TextFormatter)


def test_configure_logging_selects_json_by_default(monkeypatch):
    assert isinstance(configure_with(monkeypatch, log_format="json").formatter, JsonFormatter)


def test_configure_logging_falls_back_to_json_for_unknown_format(monkeypatch):
    """Spec: an unrecognised LOG_FORMAT must not take structured logging down."""
    assert isinstance(configure_with(monkeypatch, log_format="xml").formatter, JsonFormatter)


def test_configure_logging_applies_level_and_disables_uvicorn_access(monkeypatch):
    handler = configure_with(monkeypatch, log_format="json", log_level="WARNING")
    assert logging.getLogger().level == logging.WARNING
    assert logging.getLogger("uvicorn.access").disabled
    assert handler.stream is sys.stdout


def test_configure_logging_invalid_level_falls_back_to_info(monkeypatch):
    configure_with(monkeypatch, log_format="json", log_level="not-a-level")
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_quiets_third_party_http_clients(monkeypatch):
    """The product stream carries business milestones, not per-call transport noise."""
    configure_with(monkeypatch, log_format="json")
    for name in ("httpx", "httpx2", "httpcore"):
        assert logging.getLogger(name).level == logging.WARNING


def test_stdout_message_strips_ansi_without_collapsing_lines(monkeypatch):
    """Spec §6: no ANSI in non-terminal output; D8: newlines stay inside the value."""
    sink = capture_stdout(monkeypatch, "json")
    logging.getLogger("ai-service.demo").info(
        "\x1b[31mred\x1b[0m first\nsecond\tline"
    )

    raw = [line for line in sink.getvalue().splitlines() if line.strip()]
    assert len(raw) == 1, "one record must stay one physical line"
    payload = json.loads(raw[0])
    assert payload["msg"] == "red first\nsecond\tline"
    assert "\x1b" not in payload["msg"]


def test_configure_logging_is_idempotent_and_never_grows_handlers(monkeypatch):
    handler = configure_with(monkeypatch, log_format="json")
    first = list(logging.getLogger().handlers)
    from config import settings

    monkeypatch.setattr(settings, "log_format", "text", raising=False)

    configure_logging()
    configure_logging()

    assert logging.getLogger().handlers == first
    assert logging.getLogger().handlers[0] is handler
    assert isinstance(handler.formatter, JsonFormatter)


def test_text_formatter_renders_correlation_ids():
    record = logging.LogRecord(
        ACCESS_LOGGER, logging.INFO, __file__, 1, "hello", None, None
    )
    record.request_id = "20261020T230000000-11"
    record.trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    line = TextFormatter().format(record)
    assert "request_id=20261020T230000000-11" in line
    assert "trace_id=4bf92f3577b34da6a3ce929d0e0e4736" in line
    assert line.endswith("hello")


def test_text_formatter_inherits_contextvar_without_explicit_ids():
    record = logging.LogRecord(ACCESS_LOGGER, logging.INFO, __file__, 1, "ctx", None, None)
    request_token = request_id_var.set("20261020T230000000-12")
    trace_token = trace_id_var.set("aaaa")
    try:
        line = TextFormatter().format(record)
    finally:
        request_id_var.reset(request_token)
        trace_id_var.reset(trace_token)
    assert "request_id=20261020T230000000-12" in line
    assert "trace_id=aaaa" in line


def test_text_formatter_uses_dash_without_correlation_context():
    record = logging.LogRecord(ACCESS_LOGGER, logging.INFO, __file__, 1, "orphan", None, None)
    request_token = request_id_var.set(None)
    trace_token = trace_id_var.set(None)
    try:
        line = TextFormatter().format(record)
    finally:
        request_id_var.reset(request_token)
        trace_id_var.reset(trace_token)
    assert "request_id=-" in line
    assert "trace_id=-" in line


def test_text_formatter_output_for_a_real_request(monkeypatch):
    sink = capture_stdout(monkeypatch, "text")
    assert isinstance(logging.getLogger().handlers[0].formatter, TextFormatter)

    app_client().get("/probe", headers={"X-Request-ID": "20261020T230000000-13"})

    lines = [line for line in sink.getvalue().splitlines() if "request completed" in line]
    assert len(lines) == 1
    assert "request_id=20261020T230000000-13" in lines[0]


def test_one_parseable_json_summary_per_request(monkeypatch):
    sink = capture_stdout(monkeypatch, "json")

    app_client().get(
        "/probe",
        headers={"X-Request-ID": "20261020T230000000-14", "traceparent": VALID_TRACEPARENT},
    )

    completed = summary_lines(sink)
    assert len(completed) == 1
    payload = completed[0]
    assert payload["request_id"] == "20261020T230000000-14"
    assert payload["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert payload["route"] == "/probe"
    assert payload["status"] == 200
    assert payload["outcome"] == "ok"
    assert payload["error_code"] == ""
    assert payload["duration_ms"] >= 0
    assert payload["service"] == "ai-service"
    assert payload["schema"] == 1
    assert payload["logger"] == ACCESS_LOGGER
    assert payload["level"] == "INFO"
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", payload["ts"]
    ), payload["ts"]
    assert payload["caller"].startswith("logging_setup.py:") and payload["caller"].endswith("]")


def test_json_formatter_keeps_event_fallback_and_adds_logger():
    """Generic logs keep ``event`` as the logger name; ``logger`` carries it too."""
    record = logging.LogRecord(
        ACCESS_LOGGER, logging.INFO, __file__, 7, "hello", None, None, func="probe"
    )

    payload = json.loads(JsonFormatter().format(record))

    assert payload["event"] == ACCESS_LOGGER
    assert payload["logger"] == ACCESS_LOGGER
    assert payload["caller"].startswith("test_logging_runtime_contract.py:7[probe]")


def test_text_formatter_keeps_control_characters_out_of_the_line():
    record = logging.LogRecord(
        ACCESS_LOGGER, logging.INFO, __file__, 1, "line1\nline2\tend\r", None, None
    )

    line = TextFormatter().format(record)

    assert "\n" not in line and "\r" not in line and "\t" not in line
    assert "line1 line2 end" in line


@pytest.mark.parametrize(
    "key",
    [
        "secret_key",
        "private_key",
        "passwd",
        "ticket",
        "pairing_link",
        "authorization_url",
        "authorization_attempt",
        "assertion",
    ],
)
def test_spec_secret_keys_are_redacted(key):
    record = logging.LogRecord(
        ACCESS_LOGGER, logging.INFO, __file__, 1, "note", None, None
    )
    setattr(record, key, f"sensitive-{key}")

    payload = json.loads(JsonFormatter().format(record))

    assert payload[key] == "[REDACTED]"


# ---- non-ASCII survivability (Windows console / GBK) ------------------------


def ascii_console():
    """A stream that rejects characters the way a Chinese Windows console does."""
    sink = io.BytesIO()
    return sink, io.TextIOWrapper(sink, encoding="ascii", errors="strict")


def test_log_line_survives_a_console_that_cannot_encode_non_ascii(monkeypatch):
    """main.py logs a check mark on startup; losing that line is silent breakage."""
    sink, stream = ascii_console()
    with contextlib.redirect_stdout(stream):
        configure_with(monkeypatch, log_format="json", log_level="INFO")
        logging.getLogger("ai-service").info("chat v2 router loaded \u2713")
        stream.flush()

    payload = json.loads(sink.getvalue().decode("ascii"))
    assert payload["msg"] == "chat v2 router loaded \u2713"


def test_error_detail_line_survives_emoji_in_the_exception_message(monkeypatch):
    """The 500-detail line is the one you need when a request breaks.

    ``request.completed`` (pure ASCII) still reaches the console, so the failure
    leaves no trace in the logs at all.
    """
    sink, stream = ascii_console()
    with contextlib.redirect_stdout(stream):
        configure_with(monkeypatch, log_format="json", log_level="INFO")
        with pytest.raises(ValueError):
            app_client(boom=ValueError("计划生成失败 \U0001f389")).get(
                "/probe", headers={"X-Request-ID": "20261020T230000000-15"}
            )
        stream.flush()

    written = sink.getvalue().decode("ascii")
    assert "request failed" in written, "the failure line was dropped"
    assert "20261020T230000000-15" in written
    failure = next(
        json.loads(line) for line in written.splitlines()
        if json.loads(line).get("event") == "request.failed"
    )
    assert failure["error_type"] == "ValueError"
    assert failure["error_message"] == "计划生成失败 \U0001f389"


@pytest.mark.parametrize("encoding", ["ascii", "gbk", "utf-8"])
@pytest.mark.parametrize("log_format", ["json", "text"])
def test_console_encoding_preserves_records(monkeypatch, encoding, log_format):
    sink = io.BytesIO()
    stream = io.TextIOWrapper(sink, encoding=encoding, errors="strict")
    message = "计划 \u2713 \U0001f389"
    with contextlib.redirect_stdout(stream):
        handler = configure_with(
            monkeypatch, log_format=log_format, log_level="INFO"
        )
        logging.getLogger("ai-service").info(message)
        stream.flush()

    assert handler.stream is stream
    written = sink.getvalue().decode(encoding)
    assert len(written.splitlines()) == 1
    if log_format == "json":
        # JSON escapes must round-trip even on GBK/ASCII (no \U / \x escapes).
        assert json.loads(written)["msg"] == message
    else:
        expected = message.encode(encoding, errors="backslashreplace").decode(encoding)
        assert written.rstrip().endswith(expected)
    if encoding == "utf-8":
        assert message in written


def test_json_console_fallback_keeps_redaction(monkeypatch):
    sink, stream = ascii_console()
    with contextlib.redirect_stdout(stream):
        configure_with(monkeypatch, log_format="json", log_level="INFO")
        logging.getLogger("ai-service").info(
            "计划 \U0001f389 Authorization: Bearer fixture-secret-token",
            extra={"api_key": "fixture-api-key", "detail": {"note": "中文"}},
        )
        stream.flush()

    written = sink.getvalue().decode("ascii")
    payload = json.loads(written)
    assert "计划 \U0001f389" in payload["msg"]
    assert "[REDACTED]" in payload["msg"]
    assert payload["api_key"] == "[REDACTED]"
    assert json.loads(payload["detail"]) == {"note": "中文"}
    assert "fixture-secret-token" not in written
    assert "fixture-api-key" not in written


def test_console_logging_does_not_probe_encoding_before_writing(monkeypatch):
    class WriteOnlyConsole(io.StringIO):
        @property
        def encoding(self):
            raise AssertionError("log writes must not probe encoding")

    sink = WriteOnlyConsole()
    monkeypatch.setattr(sys, "stdout", sink)
    configure_with(monkeypatch, log_format="json", log_level="INFO")
    logging.getLogger("ai-service").info("正常日志 \u2713")
    assert json.loads(sink.getvalue())["msg"] == "正常日志 \u2713"


def test_console_handler_writes_through_when_stream_accepts_the_text(monkeypatch):
    """pytest capture exposes no encoding; emit must not re-encode or drop it."""
    sink = capture_stdout(monkeypatch, "json")
    handler = logging.getLogger().handlers[0]
    assert sink.encoding is None
    record = logging.LogRecord(
        "ai-service", logging.INFO, __file__, 1, "中文 \u2713", None, None
    )

    handler.emit(record)

    written = sink.getvalue()
    assert len(written.splitlines()) == 1
    assert json.loads(written)["msg"] == "中文 \u2713"


def test_console_handler_never_raises_when_the_stream_is_broken():
    """A closed pipe must not turn into a business exception."""
    from observability.logging_setup import _ConsoleStreamHandler

    class BrokenStream:
        encoding = "utf-8"

        def write(self, _text):
            raise OSError("pipe closed")

        def flush(self):
            raise OSError("pipe closed")

    handler = _ConsoleStreamHandler(BrokenStream())
    handler.setFormatter(JsonFormatter())
    failures = []
    handler.handleError = failures.append
    record = logging.LogRecord("ai-service", logging.ERROR, __file__, 1, "boom", None, None)

    handler.emit(record)

    assert len(failures) == 1


# ---- in-band SSE failures must reach the access log -------------------------


def _sse_error_app(monkeypatch, *, code="v2_disabled", message="private-error-fixture"):
    import api.chat_v2 as chat_v2
    from agent_v2.events import EventEnvelope

    class Leases:
        def release(self) -> None:
            return None

    class Orchestrator:
        async def acquire_leases(self, **_kwargs):
            return Leases()

        async def stream_chat(self, *_args, **_kwargs):
            yield EventEnvelope(
                version="2",
                sequence=1,
                type="error",
                payload={"code": code, "message": message},
            )

    monkeypatch.setattr(
        chat_v2, "_build_run_orchestrator", lambda _runtime: Orchestrator()
    )
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    app.state.agent_v2_runtime = None
    app.include_router(chat_v2.chat_v2_router, prefix="/api/v2")
    return TestClient(app)


def test_real_http_sse_error_is_logged_as_failure_on_stdout(monkeypatch):
    """HTTP stays 200 by contract; the access log must still call the run failed."""
    sink = capture_stdout(monkeypatch, "json")
    client = _sse_error_app(monkeypatch)

    response = client.post(
        "/api/v2/chat/sse",
        json={"message": "hi", "request_id": "client-1"},
        headers={"X-Request-ID": "20261021T010000000-1"},
    )

    assert response.status_code == 200
    completed = summary_lines(sink)
    failures = summary_lines(sink, "request.failed")
    assert len(completed) == 1, "an SSE request must still produce one summary line"
    assert completed[0]["status"] == 200
    assert completed[0]["outcome"] == "error"
    assert completed[0]["error_code"] == "v2_disabled"
    assert len(failures) == 1
    assert failures[0]["error_type"] == "StreamError"
    assert failures[0]["error_code"] == "v2_disabled"
    assert failures[0]["request_id"] == "20261021T010000000-1"
    assert "private-error-fixture" not in json.dumps(completed + failures, ensure_ascii=False)


# ---- the real app: main.py wiring ------------------------------------------


def test_main_app_installs_the_request_context_middleware():
    import main

    installed = [m.cls.__name__ for m in main.app.user_middleware]
    assert "RequestContextMiddleware" in installed, (
        "every route must sit behind the correlation middleware"
    )


def test_real_app_health_logs_one_line_matching_the_response_request_id(monkeypatch):
    import main

    client = TestClient(main.app)
    response = client.get(
        "/health",
        headers={"X-Request-ID": "20261020T230000000-16", "traceparent": VALID_TRACEPARENT},
    )
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "20261020T230000000-16"

    sink = capture_stdout(monkeypatch)

    client.get("/health", headers={"X-Request-ID": "20261020T230000000-17"})

    completed = summary_lines(sink)
    assert len(completed) == 1, "one request must produce exactly one summary line"
    assert completed[0]["request_id"] == "20261020T230000000-17"
    assert completed[0]["route"] == "/health"
    assert completed[0]["status"] == 200
    assert completed[0]["duration_ms"] >= 0


def test_logging_keeps_working_when_tracing_is_disabled(monkeypatch, access_logs):
    monkeypatch.setattr(tracing, "_shutdown", True)
    monkeypatch.setattr(tracing, "get_tracer", lambda: None)

    with tracing.start_span("chat.request"):
        response = app_client().get("/probe", headers={"X-Request-ID": "20261020T230000000-18"})

    assert response.status_code == 200
    summary = access_logs[-1]
    assert summary.event == "request.completed"
    assert summary.request_id == "20261020T230000000-18"
    assert summary.trace_id == ""


# ---- non-http scopes --------------------------------------------------------


@pytest.mark.parametrize("scope_type", ["websocket", "lifespan"])
def test_non_http_scope_is_passed_through_without_logging(scope_type, access_logs):
    calls = []

    async def inner(scope, receive, send):
        calls.append(scope["type"])

    async def receive():
        return {"type": scope_type}

    async def send(message):
        return None

    scope = {"type": scope_type, "headers": [], "path": "/ws", "method": "GET"}

    asyncio.run(RequestContextMiddleware(inner)(scope, receive, send))

    assert calls == [scope_type]
    assert access_logs == []
