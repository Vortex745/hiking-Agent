"""Structured stdout logging with request correlation.

Every request emits exactly one JSON summary line after it finishes. Streaming
responses are never buffered: the summary is written when the ASGI app
returns (i.e. after the stream closes). Errors always get an extra line with
the exception type and a redacted message.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from config import settings
from observability.redaction import (
    REDACTED,
    is_denied_key,
    sanitize_mapping,
    scrub_text,
)
from observability.request_context import (
    REQUEST_ID_HEADER,
    TRACEPARENT_HEADER,
    generate_request_id,
    parse_traceparent,
    valid_request_id,
)

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)

SERVICE_NAME = "ai-service"
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _utc_iso_ms(created: float) -> str:
    """One timestamp format for every channel: UTC, millisecond precision, Z."""
    return (
        datetime.fromtimestamp(created, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + "Z"
    )


def _caller(record: logging.LogRecord) -> str:
    return f"{record.filename}:{record.lineno}[{record.funcName}]"[:120]


def _sanitize_line(text: str) -> str:
    """Keep one record on one line in text mode; JSON relies on escaping."""
    return _CONTROL_CHARS.sub(
        " ", str(text).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    )

_PINNED_KEYS = (
    "ts",
    "level",
    "event",
    "msg",
    "request_id",
    "trace_id",
)

# Third-party HTTP clients log every outbound call at INFO. In the product's
# structured stream that is transport noise, not a business milestone.
_QUIET_LOGGERS = ("httpx", "httpx2", "httpcore")

# Instance attributes every LogRecord already carries; never dump them as
# extra fields (they are noise and may include paths).
_STANDARD_ATTRS = set(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _utc_iso_ms(record.created),
            "level": record.levelname,
            "event": record.name,
            "logger": record.name,
            "service": SERVICE_NAME,
            "schema": 1,
            "caller": _caller(record),
            "msg": scrub_text(record.getMessage(), 2000),
            "request_id": getattr(record, "request_id", None)
            or request_id_var.get()
            or "",
            "trace_id": getattr(record, "trace_id", None) or trace_id_var.get() or "",
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            if key in {"ts", "level", "msg"}:
                continue
            if is_denied_key(key):
                payload[key] = REDACTED
                continue
            if isinstance(value, str):
                payload[key] = scrub_text(value)
            elif isinstance(value, (int, float, bool)) or value is None:
                payload[key] = value
            elif isinstance(value, (dict, list, tuple)):
                try:
                    payload[key] = scrub_text(
                        json.dumps(
                            sanitize_mapping(value), ensure_ascii=False, default=str
                        ),
                        2000,
                    )
                except Exception:  # pragma: no cover - defensive
                    payload[key] = REDACTED
        for key in _PINNED_KEYS:
            payload.setdefault(key, "")
        if record.exc_info:
            payload["exc_info"] = scrub_text(
                self.formatException(record.exc_info), 4000
            )
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        request_id = getattr(record, "request_id", None) or request_id_var.get() or "-"
        trace_id = getattr(record, "trace_id", None) or trace_id_var.get() or "-"
        return (
            f"{_utc_iso_ms(record.created)} "
            f"{record.levelname} {record.name} request_id={request_id} "
            f"trace_id={trace_id} {_caller(record)} | {_sanitize_line(record.getMessage())}"
        )


class _ConsoleStreamHandler(logging.StreamHandler):
    """Keep whole records on consoles that cannot encode every character."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            try:
                self.stream.write(message + self.terminator)
            except UnicodeEncodeError as exc:
                if isinstance(self.formatter, JsonFormatter):
                    # JSON requires \u escapes, not backslashreplace's \U / \x.
                    message = json.dumps(json.loads(message), ensure_ascii=True)
                else:
                    message = message.encode(
                        exc.encoding, errors="backslashreplace"
                    ).decode(exc.encoding)
                self.stream.write(message + self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True
    handler = _ConsoleStreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if settings.log_format != "text" else TextFormatter()
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))
    # Access logs are produced by our middleware; don't double-log.
    logging.getLogger("uvicorn.access").disabled = True
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    # Local agent execution log (LLM_LOG). Disabled by default; failures in
    # setup are contained by llm_log itself.
    from observability import llm_log

    llm_log.setup()


class RequestContextMiddleware:
    """Pure ASGI middleware: correlation ids + one-line request summary."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        raw_request_id = headers.get(REQUEST_ID_HEADER.encode(), b"").decode(
            "latin-1"
        )
        request_id = valid_request_id(raw_request_id) or generate_request_id()
        traceparent = headers.get(TRACEPARENT_HEADER.encode(), b"").decode("latin-1")
        trace_id = parse_traceparent(traceparent)
        if traceparent and trace_id is None:
            logging.getLogger("ai-service.http").warning(
                "invalid traceparent dropped",
                extra={
                    "request_id": request_id,
                    "trace_id": "",
                    "event": "traceparent.invalid",
                },
            )
            scope["headers"] = [
                (key, value)
                for key, value in scope.get("headers", [])
                if key.lower() != TRACEPARENT_HEADER.encode()
            ]

        request_token = request_id_var.set(request_id)
        trace_token = trace_id_var.set(trace_id or "")
        # Publish the correlation id on the ASGI scope so handlers (and their
        # spans) use exactly the id that appears in the request summary line.
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status_code = 500
        error: BaseException | None = None

        async def send_with_echo(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                raw_headers = list(message.get("headers") or [])
                raw_headers.append(
                    (REQUEST_ID_HEADER.encode(), request_id.encode("latin-1"))
                )
                message = {**message, "headers": raw_headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_echo)
        except BaseException as exc:
            error = exc
            raise
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            route = scope.get("route")
            route_path = getattr(route, "path", None) or scope.get("path", "")
            # A span started during the request publishes its trace id on the
            # logging context; streaming handlers additionally publish it on
            # the shared ASGI state because their body runs in a child task.
            state_trace_id = (scope.get("state") or {}).get("trace_id", "")
            effective_trace_id = trace_id or state_trace_id or trace_id_var.get() or ""
            state = scope.get("state") or {}
            stream_outcome = state.get("stream_outcome")
            stream_error_code = state.get("stream_error_code", "")
            failed = error is not None or status_code >= 500 or stream_outcome == "error"
            if failed:
                outcome = "error"
            elif status_code >= 400:
                outcome = "client_error"
            else:
                outcome = stream_outcome or "ok"
            if failed:
                level = logging.ERROR
            elif status_code >= 400:
                level = logging.WARNING
            else:
                level = logging.INFO
            method = scope.get("method", "")
            logger = logging.getLogger("ai-service.http")
            logger.log(
                level,
                "request completed",
                extra={
                    "event": "request.completed",
                    "request_id": request_id,
                    "trace_id": effective_trace_id,
                    "method": method,
                    "route": route_path,
                    "status": status_code,
                    "duration_ms": duration_ms,
                    "outcome": outcome,
                    "error_code": stream_error_code,
                },
            )
            # A bare 5xx has no cause to report; only a real exception or an
            # in-band stream error earns the extra failure line.
            if error is not None or stream_outcome == "error":
                logger.error(
                    "request failed",
                    extra={
                        "event": "request.failed",
                        "request_id": request_id,
                        "trace_id": effective_trace_id,
                        "method": method,
                        "route": route_path,
                        "status": status_code,
                        "duration_ms": duration_ms,
                        "outcome": outcome,
                        "error_code": stream_error_code,
                        "error_type": (
                            type(error).__name__ if error is not None
                            else "StreamError" if stream_outcome == "error" else ""
                        ),
                        "error_message": scrub_text(str(error), 500) if error else "",
                    },
                )
            request_id_var.reset(request_token)
            trace_id_var.reset(trace_token)
