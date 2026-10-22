"""Local JSONL execution log for agent runs (LLM_LOG).

This is a local debugging channel, not an authoritative record: ``run_events``
and ``run_telemetry`` remain the source of truth. It is disabled by default;
initialization or write failures degrade to one warning and never reach chat.
"""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from config import settings
from observability.redaction import sanitize_mapping, sanitize_messages, scrub_text

SCHEMA_VERSION = 2
SERVICE_NAME = "ai-service"
LOGGER_NAME = "ai-service.llm_log"
MAX_MESSAGE_CHARS = 2000
MAX_MESSAGES = 50
MAX_TOOL_ARGS_CHARS = 1000
MAX_OUTPUT_CHARS = 4000
MAX_FILE_BYTES = 20 * 1024 * 1024
BACKUP_COUNT = 5

logger = logging.getLogger("ai-service.llm_log")
# Degradation warnings must reach stdout, not the JSONL file: a plain-text
# warning inside the file would break the one-JSON-object-per-line contract.
_warn_logger = logging.getLogger("ai-service.observability")

_handler: logging.Handler | None = None
_setup_done = False
_broken = False


@dataclass
class _TurnState:
    run_id: str
    turn_id: str
    started: float
    step: int = 0
    generations: int = 0
    tools: int = 0


_turn: ContextVar[_TurnState | None] = ContextVar("llm_log_turn", default=None)


def enabled() -> bool:
    return _handler is not None and not _broken


def setup(
    *,
    enabled: bool | None = None,
    path: str | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
) -> None:
    """Configure the file handler once. Never raises into the caller."""
    global _handler, _setup_done, _broken
    if _setup_done:
        return
    _setup_done = True
    want = settings.llm_log_enabled if enabled is None else enabled
    if not want:
        return
    target = path or settings.llm_log_path
    try:
        from pathlib import Path

        file_path = Path(target)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_bytes or MAX_FILE_BYTES,
            backupCount=backup_count if backup_count is not None else BACKUP_COUNT,
            encoding="utf-8",
            delay=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _handler = handler
    except Exception as exc:  # pragma: no cover - environment dependent
        _broken = True
        _handler = None
        _warn_logger.warning("llm log disabled: %s", exc)


def close() -> None:
    """Detach the handler. Used by tests and for clean reconfiguration."""
    global _handler, _setup_done, _broken
    handler, _handler = _handler, None
    _setup_done = False
    _broken = False
    if handler is not None:
        try:
            handler.close()
        except Exception:  # pragma: no cover - defensive
            pass
        logger.handlers = []
        logger.propagate = True


def begin_turn(
    *,
    run_id: str,
    model: str = "",
    message_chars: int = 0,
    history_count: int = 0,
) -> Any:
    if not enabled():
        return None
    state = _TurnState(
        run_id=run_id or "",
        turn_id=uuid.uuid4().hex[:12],
        started=time.perf_counter(),
    )
    token = _turn.set(state)
    _write(
        "llm.turn.started",
        "INFO",
        {
            "model": model,
            "message_chars": message_chars,
            "history_count": history_count,
        },
    )
    return token


def end_turn(
    token: Any,
    *,
    status: str,
    final_text: str = "",
    error: BaseException | None = None,
) -> None:
    if token is None:
        return
    state = _turn.get()
    normal_close = _is_normal_close(error)
    if normal_close:
        status = "cancelled"
        error = None
    try:
        if state is not None and enabled():
            level = "ERROR" if status == "error" else "WARNING" if status == "cancelled" else "INFO"
            _write(
                "llm.turn.completed",
                level,
                {
                    "status": status,
                    "duration_ms": _elapsed_ms(state.started),
                    "generations": state.generations,
                    "tools": state.tools,
                    "final_text": scrub_text(final_text or "", MAX_OUTPUT_CHARS),
                    "error_type": type(error).__name__ if error is not None else "",
                },
            )
    finally:
        try:
            _turn.reset(token)
        except ValueError:  # context was copied into another task
            _turn.set(None)


def generation(
    *,
    model: str,
    provider: str = "",
    temperature: Any = None,
    tools: list[str] | None = None,
    messages: Any = None,
    output: str = "",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    ttft_ms: float | None = None,
    duration_ms: float | None = None,
    error: BaseException | None = None,
    agent: str = "main",
) -> None:
    if not enabled():
        return
    state = _turn.get()
    if state is not None:
        state.step += 1
        state.generations += 1
    formatted, total = _format_messages(messages)
    normal_error = None if _is_normal_close(error) else error
    _write(
        "llm.generation",
        "ERROR" if normal_error is not None else "INFO",
        {
            "step": state.step if state is not None else 0,
            "agent": agent,
            "model": _safe(model),
            "provider": _safe(provider),
            "temperature": temperature if isinstance(temperature, (int, float)) else None,
            "tools": list(tools or []),
            "input": {"messages": formatted, "messages_total": total},
            "output": {"text": scrub_text(output or "", MAX_OUTPUT_CHARS)},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            "ttft_ms": ttft_ms,
            "duration_ms": duration_ms,
            "status": "error" if normal_error is not None else "ok",
            "error_type": (
                type(normal_error).__name__ if normal_error is not None else ""
            ),
            "error_message": (
                scrub_text(str(normal_error), 500)
                if normal_error is not None
                else ""
            ),
        },
    )


def tool(
    *,
    tool: str,
    args: Any = None,
    status: str = "ok",
    duration_ms: float | None = None,
    output_chars: int | None = None,
    error: BaseException | None = None,
    error_type: str | None = None,
    langgraph_step: Any = None,
) -> None:
    if not enabled():
        return
    state = _turn.get()
    if state is not None:
        state.tools += 1
    failed = error is not None or bool(error_type) or status == "error"
    _write(
        "llm.tool",
        "ERROR" if failed else "INFO",
        {
            "step": state.step if state is not None else 0,
            "langgraph_step": langgraph_step,
            "tool": _safe(tool),
            "args": _format_args(args),
            "status": status,
            "duration_ms": duration_ms,
            "output_chars": output_chars,
            "error_type": (
                type(error).__name__ if error is not None else (error_type or "")
            ),
        },
    )


def tool_denied(
    *,
    tool: str,
    reason: str,
    args: Any = None,
    run_id: str = "",
) -> None:
    if not enabled():
        return
    state = _turn.get()
    _write(
        "llm.tool.denied",
        "WARNING",
        {
            "run_id": run_id or None,
            "step": state.step if state is not None else 0,
            "tool": _safe(tool),
            "args": _format_args(args),
            "reason": _safe(reason),
        },
    )


def _write(event: str, level: str, fields: dict[str, Any]) -> None:
    global _broken
    if _handler is None or _broken:
        return
    state = _turn.get()
    payload: dict[str, Any] = {
        "ts": _now_iso(),
        "schema": SCHEMA_VERSION,
        "level": level,
        "event": event,
        "logger": LOGGER_NAME,
        "service": SERVICE_NAME,
        "request_id": _request_id(),
        "trace_id": _trace_id(),
        "run_id": fields.pop("run_id", None) or (state.run_id if state else ""),
        "session": _session(),
        "turn_id": state.turn_id if state is not None else "",
    }
    payload.update(fields)
    try:
        logger.info(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        )
    except Exception as exc:  # pragma: no cover - defensive
        _broken = True
        _warn_logger.warning("llm log disabled after write failure: %s", exc)


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    )


def _format_messages(messages: Any) -> tuple[list[dict], int]:
    sanitized = sanitize_messages(messages)
    total = len(sanitized)
    if total > MAX_MESSAGES:
        # Keep the system prompt (first) and the most recent context.
        sanitized = [sanitized[0], *sanitized[-(MAX_MESSAGES - 1) :]]
    formatted: list[dict] = []
    for item in sanitized:
        entry = dict(item)
        content = entry.get("content")
        if isinstance(content, str):
            entry["content"] = scrub_text(content, MAX_MESSAGE_CHARS)
        formatted.append(entry)
    return formatted, total


def _format_args(args: Any) -> str:
    if args is None:
        return ""
    if isinstance(args, str):
        return scrub_text(args, MAX_TOOL_ARGS_CHARS)
    try:
        return scrub_text(
            json.dumps(sanitize_mapping(args), ensure_ascii=False, default=str),
            MAX_TOOL_ARGS_CHARS,
        )
    except Exception:  # pragma: no cover - defensive
        return scrub_text(str(args), MAX_TOOL_ARGS_CHARS)


def _safe(value: Any) -> str:
    return "" if value is None else scrub_text(str(value), 200)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _is_normal_close(error: BaseException | None) -> bool:
    return isinstance(error, (GeneratorExit, asyncio.CancelledError))


def _request_id() -> str:
    try:
        from observability.logging_setup import request_id_var

        return request_id_var.get() or ""
    except Exception:  # pragma: no cover - defensive
        return ""


def _trace_id() -> str:
    try:
        from observability.logging_setup import trace_id_var

        return trace_id_var.get() or ""
    except Exception:  # pragma: no cover - defensive
        return ""


def _session() -> str:
    try:
        from observability import tracing

        return tracing.current_session() or ""
    except Exception:  # pragma: no cover - defensive
        return ""
