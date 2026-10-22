"""LangChain callback handler that emits tool spans only.

Generations are traced by the model decorator (ticket 03). This handler is
deliberately limited to tool start/end/error so the kernel does not regress to
the old "trace everything through callbacks" design.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from observability import llm_log, tracing
from observability.redaction import scrub_text


class ToolSpanCallbackHandler(BaseCallbackHandler):
    """Create one ``tool.<name>`` child span per tool invocation."""

    def __init__(self, parent_span: Any = None) -> None:
        self._parent_span = parent_span
        self._spans: dict[Any, Any] = {}

    def _start_span(self, serialized: dict | None, run_id: Any) -> None:
        tracer = tracing.get_tracer()
        if tracer is None:
            return
        raw_name = ""
        if isinstance(serialized, dict):
            raw_name = str(serialized.get("name") or serialized.get("id") or "")
        name = raw_name.split(".")[-1] or "tool"
        context = None
        if self._parent_span is not None:
            from opentelemetry import trace

            context = trace.set_span_in_context(self._parent_span)
        try:
            self._spans[run_id] = tracer.start_span(
                f"tool.{name}",
                context=context,
                attributes=tracing.attributes(
                    {
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": name,
                    }
                ),
            )
        except Exception:  # pragma: no cover - defensive
            return

    def _finish_span(self, run_id: Any, *, error: BaseException | None = None) -> None:
        span = self._spans.pop(run_id, None)
        if span is None:
            return
        try:
            if error is None:
                span.set_attribute("gen_ai.tool.status", "ok")
                tracing.mark_ok(span)
            else:
                span.set_attribute("gen_ai.tool.status", "error")
                tracing.record_exception(span, error)
            span.end()
        except Exception:  # pragma: no cover - defensive
            return

    # -- sync -----------------------------------------------------------
    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        self._start_span(serialized, kwargs.get("run_id"))

    def on_tool_end(self, output, **kwargs) -> None:
        # Raw tool output is on the reject list; record size only.
        span = self._spans.get(kwargs.get("run_id"))
        if span is not None:
            try:
                span.set_attribute(
                    "gen_ai.tool.output.length",
                    len(scrub_text(str(output), 4000)),
                )
            except Exception:  # pragma: no cover - defensive
                pass
        self._finish_span(kwargs.get("run_id"))

    def on_tool_error(self, error, **kwargs) -> None:
        self._finish_span(kwargs.get("run_id"), error=error)

    # -- async ----------------------------------------------------------
    async def on_tool_start_async(self, serialized, input_str, **kwargs) -> None:
        self.on_tool_start(serialized, input_str, **kwargs)

    async def on_tool_end_async(self, output, **kwargs) -> None:
        self.on_tool_end(output, **kwargs)

    async def on_tool_error_async(self, error, **kwargs) -> None:
        self.on_tool_error(error, **kwargs)


class ToolLogCallbackHandler(BaseCallbackHandler):
    """Write one ``llm.tool`` record per tool invocation.

    Deliberately log-only: spans stay with :class:`ToolSpanCallbackHandler` so
    enabling the local log never changes the exported trace tree.
    """

    def __init__(self) -> None:
        self._started: dict[Any, tuple[str, float, Any, Any]] = {}

    @staticmethod
    def _tool_name(serialized: Any) -> str:
        raw = ""
        if isinstance(serialized, dict):
            raw = str(serialized.get("name") or serialized.get("id") or "")
        return raw.split(".")[-1] or "tool"

    def on_tool_start(self, serialized, input_str=None, **kwargs) -> None:
        if not llm_log.enabled():
            return
        metadata = kwargs.get("metadata") or {}
        args = kwargs.get("inputs")
        if args is None:
            args = input_str
        self._started[kwargs.get("run_id")] = (
            self._tool_name(serialized),
            time.perf_counter(),
            args,
            metadata.get("langgraph_step"),
        )

    def on_tool_end(self, output, **kwargs) -> None:
        entry = self._started.pop(kwargs.get("run_id"), None)
        if entry is None:
            return
        name, started, args, step = entry
        text = _output_text(output)
        # Domain adapters turn timeouts/exceptions into a normal public error
        # result; without this check the failure would look like a success.
        error_type = _domain_error_type(name, text)
        llm_log.tool(
            tool=name,
            args=args,
            status="error" if error_type else "ok",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            output_chars=len(text),
            error_type=error_type,
            langgraph_step=step,
        )

    def on_tool_error(self, error, **kwargs) -> None:
        entry = self._started.pop(kwargs.get("run_id"), None)
        if entry is None:
            return
        name, started, args, step = entry
        llm_log.tool(
            tool=name,
            args=args,
            status="error",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            error=error,
            langgraph_step=step,
        )

    async def on_tool_start_async(self, serialized, input_str=None, **kwargs) -> None:
        self.on_tool_start(serialized, input_str, **kwargs)

    async def on_tool_end_async(self, output, **kwargs) -> None:
        self.on_tool_end(output, **kwargs)

    async def on_tool_error_async(self, error, **kwargs) -> None:
        self.on_tool_error(error, **kwargs)


def _output_text(output: Any) -> str:
    content = getattr(output, "content", output)
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


def _domain_error_type(tool_name: str, text: str) -> str:
    """Stable public error wording from the domain-tool adapter."""
    if f"工具 {tool_name} 执行超时" in text:
        return "ToolTimeout"
    if f"工具 {tool_name} 执行失败" in text:
        return "ToolError"
    # Builtin tools (Agent/task, filesystem, …) report failures as an
    # ``<error>…</error>`` result instead of raising; that is not a success.
    if text.lstrip().startswith("<error>"):
        return "ToolError"
    return ""
