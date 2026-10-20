"""Structured stdout logging with request correlation.

Every request emits exactly one JSON summary line after it finishes. Streaming
responses are never buffered: the summary is written when the ASGI app
returns (i.e. after the stream closes). Errors always get an extra line with
the exception type and a redacted message.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

from config import settings
from observability.redaction import scrub_text
from observability.request_context import (
    REQUEST_ID_HEADER,
    TRACEPARENT_HEADER,
    generate_request_id,
    parse_traceparent,
    valid_request_id,
)

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)

_PINNED_KEYS = (
    "ts",
    "level",
    "event",
    "msg",
    "request_id",
    "trace_id",
    "route",
    "status",
    "duration_ms",
)

# Instance attributes every LogRecord already carries; never dump them as
# extra fields (they are noise and may include paths).
_STANDARD_ATTRS = set(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "event": record.name,
            "msg": scrub_text(record.getMessage(), 2000),
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
            "trace_id": getattr(record, "trace_id", None) or trace_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            if key in {"ts", "level", "msg"}:
                continue
            if isinstance(value, str):
                payload[key] = scrub_text(value)
            elif isinstance(value, (int, float, bool)) or value is None:
                payload[key] = value
        for key in _PINNED_KEYS:
            payload.setdefault(key, "")
        if record.exc_info:
            payload["exc_info"] = scrub_text(
                self.formatException(record.exc_info), 4000
            )
        return json.dumps(payload, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        request_id = getattr(record, "request_id", None) or request_id_var.get() or "-"
        trace_id = getattr(record, "trace_id", None) or trace_id_var.get() or "-"
        return (
            f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S%z')} "
            f"{record.levelname} {record.name} request_id={request_id} "
            f"trace_id={trace_id} {record.getMessage()}"
        )


_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if settings.log_format != "text" else TextFormatter()
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))
    # Access logs are produced by our middleware; don't double-log.
    logging.getLogger("uvicorn.access").disabled = True


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
            logger = logging.getLogger("ai-service.http")
            logger.info(
                "request completed",
                extra={
                    "event": "request.completed",
                    "request_id": request_id,
                    "trace_id": trace_id or "",
                    "route": route_path,
                    "status": status_code,
                    "duration_ms": duration_ms,
                },
            )
            if error is not None or status_code >= 500:
                logger.error(
                    "request failed",
                    extra={
                        "event": "request.failed",
                        "request_id": request_id,
                        "trace_id": trace_id or "",
                        "route": route_path,
                        "status": status_code,
                        "duration_ms": duration_ms,
                        "error_type": type(error).__name__ if error else "",
                        "error_msg": scrub_text(str(error), 500) if error else "",
                    },
                )
            request_id_var.reset(request_token)
            trace_id_var.reset(trace_token)
