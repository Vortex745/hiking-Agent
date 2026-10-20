"""LangChain callback handler that emits tool spans only.

Generations are traced by the model decorator (ticket 03). This handler is
deliberately limited to tool start/end/error so the kernel does not regress to
the old "trace everything through callbacks" design.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from observability import tracing
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
