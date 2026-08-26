"""OpenManus-style tool collection for local Agent runtimes.

This layer keeps tool exposure and tool execution together. It accepts the
LangChain StructuredTool objects already used by the current Agent module, while
also supporting OpenManus-like tools that expose ``to_param`` and ``execute``.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ToolExecutionResult:
    ok: bool
    output: Any = None
    error: str | None = None
    tool_name: str = ""
    args: dict[str, Any] | None = None

    def __str__(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        if isinstance(self.output, str):
            return self.output
        try:
            return json.dumps(self.output, ensure_ascii=False)
        except TypeError:
            return str(self.output)


class ToolCollection:
    """A small OpenManus-compatible collection around executable tools."""

    def __init__(self, *tools: Any):
        self.tools: tuple[Any, ...] = tuple(tools)
        self.tool_map: dict[str, Any] = {}
        for tool in self.tools:
            name = getattr(tool, "name", None)
            if name:
                self.tool_map[str(name)] = tool

    def __iter__(self):
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    def get_tool(self, name: str) -> Any | None:
        return self.tool_map.get(name)

    def to_langchain_tools(self) -> list[Any]:
        result: list[Any] = []
        for tool in self.tools:
            if hasattr(tool, "to_langchain_tool"):
                result.append(tool.to_langchain_tool())
            else:
                result.append(tool)
        return result

    def to_params(self) -> list[dict[str, Any]]:
        params: list[dict[str, Any]] = []
        for tool in self.tools:
            if hasattr(tool, "to_param"):
                params.append(tool.to_param())
                continue

            name = getattr(tool, "name", "")
            description = getattr(tool, "description", "")
            parameters = self._parameters_for_tool(tool)
            params.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            })
        return params

    def add_tool(self, tool: Any) -> "ToolCollection":
        name = getattr(tool, "name", None)
        if not name or name in self.tool_map:
            return self
        self.tools = (*self.tools, tool)
        self.tool_map[str(name)] = tool
        return self

    def add_tools(self, *tools: Any) -> "ToolCollection":
        for tool in tools:
            self.add_tool(tool)
        return self

    async def execute(self, *, name: str, tool_input: dict[str, Any] | None = None) -> ToolExecutionResult:
        tool = self.tool_map.get(name)
        args = tool_input or {}
        if tool is None:
            return ToolExecutionResult(False, error=f"Tool {name} is invalid", tool_name=name, args=args)

        try:
            output = await self._call_tool(tool, args)
            return ToolExecutionResult(True, output=output, tool_name=name, args=args)
        except Exception as exc:
            return ToolExecutionResult(False, error=str(exc), tool_name=name, args=args)

    async def _call_tool(self, tool: Any, args: dict[str, Any]) -> Any:
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(args)
        if hasattr(tool, "invoke"):
            return tool.invoke(args)
        if hasattr(tool, "execute"):
            result = tool.execute(**args)
            if inspect.isawaitable(result):
                return await result
            return result
        if callable(tool):
            result = tool(**args)
            if inspect.isawaitable(result):
                return await result
            return result
        raise TypeError(f"Tool {getattr(tool, 'name', tool)!r} is not executable")

    @staticmethod
    def _parameters_for_tool(tool: Any) -> dict[str, Any]:
        args_schema = getattr(tool, "args_schema", None)
        if args_schema and hasattr(args_schema, "model_json_schema"):
            return args_schema.model_json_schema()
        args = getattr(tool, "args", None)
        if isinstance(args, dict):
            return {"type": "object", "properties": args}
        return {"type": "object", "properties": {}}
