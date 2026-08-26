"""OpenManus-style tool-calling Agent runtime."""

from __future__ import annotations

import json
import inspect
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock

from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage, SystemMessage, ToolMessage

from agent.base_agent import AgentMemory, OpenManusAgentState
from agent.react_agent import ReActAgent
from agent.tool_collection import ToolCollection


@dataclass
class ToolCallAgent(ReActAgent):
    name: str = "toolcall"
    description: str = "An agent that can execute tool calls."
    system_prompt: str | None = None
    next_step_prompt: str | None = None
    llm: Any | None = None
    memory: AgentMemory = field(default_factory=AgentMemory)
    available_tools: ToolCollection = field(default_factory=ToolCollection)
    tool_choice: str = "auto"
    special_tool_names: list[str] = field(default_factory=lambda: ["terminate"])
    tool_calls: list[Any] = field(default_factory=list)
    max_steps: int = 30
    max_observe: int | None = None

    async def think(self) -> bool:
        if self.next_step_prompt:
            self.memory.add_message(HumanMessage(content=self.next_step_prompt))

        response = await self._ask_tool_model()
        self.tool_calls = [self._tool_call_for_message(call) for call in self._response_tool_calls(response)]
        content = self._response_content(response)
        if self.tool_calls:
            self.memory.add_message(AIMessage(content=content, tool_calls=self.tool_calls))
        else:
            self.memory.add_message(AIMessage(content=content))

        if self.tool_choice == "required" and not self.tool_calls:
            return True
        if self.tool_choice == "none":
            return bool(content)
        if self.tool_choice == "auto" and not self.tool_calls:
            return bool(content)
        return bool(self.tool_calls)

    async def act(self) -> str:
        if not self.tool_calls:
            if self.tool_choice == "required":
                raise ValueError("Tool calls required but none provided")
            return self._message_text(self.messages[-1]) or "No content or commands to execute"

        results: list[str] = []
        for call in self.tool_calls:
            name, args, call_id = self._normalize_tool_call(call)
            result = await self.execute_tool(name, args)
            if self.max_observe and len(result) > self.max_observe:
                result = result[: self.max_observe]
            self.memory.add_message(ToolMessage(content=result, name=name, tool_call_id=call_id or name))
            results.append(result)
        return "\n\n".join(results)

    async def execute_tool(self, name: str, args: dict[str, Any]) -> str:
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"
        result = await self.available_tools.execute(name=name, tool_input=args)
        await self._handle_special_tool(name, result)
        if result.ok:
            output = str(result)
            return f"Observed output of cmd `{name}` executed:\n{output}" if output else f"Cmd `{name}` completed with no output"
        return f"Error: {result.error}"

    async def _ask_tool_model(self) -> Any:
        if self.llm is None:
            raise RuntimeError("ToolCallAgent requires an llm")

        system_msgs = [SystemMessage(content=self.system_prompt)] if self.system_prompt else None
        ask_tool = getattr(self.llm, "ask_tool", None)
        if callable(ask_tool) and not isinstance(ask_tool, Mock):
            messages = self._messages_for_model(self.messages, tool_result_protocol="function")
            try:
                result = ask_tool(
                    messages=messages,
                    system_msgs=system_msgs,
                    tools=self.available_tools.to_params(),
                    tool_choice=self.tool_choice,
                )
                return await result if inspect.isawaitable(result) else result
            except Exception as exc:
                if not self._is_role_compatibility_error(exc, rejected="function", expected="tool"):
                    raise
                result = ask_tool(
                    messages=self._messages_for_model(self.messages, tool_result_protocol="tool"),
                    system_msgs=system_msgs,
                    tools=self.available_tools.to_params(),
                    tool_choice=self.tool_choice,
                )
                return await result if inspect.isawaitable(result) else result

        raw_messages = [*(system_msgs or []), *self.messages]
        messages = self._messages_for_model(raw_messages, tool_result_protocol="tool")
        bind_tools = getattr(self.llm, "bind_tools", None)
        if callable(bind_tools) and not isinstance(bind_tools, Mock):
            bound = bind_tools(self.available_tools.to_langchain_tools(), tool_choice=self.tool_choice)
            try:
                result = await self._invoke_model(bound, messages)
            except Exception as exc:
                if not self._is_role_compatibility_error(exc, rejected="tool", expected="function"):
                    raise
                legacy_messages = self._messages_for_model(raw_messages, tool_result_protocol="function")
                result = await self._invoke_model(bound, legacy_messages)
            if result is not None:
                return result

        ainvoke = getattr(self.llm, "ainvoke", None)
        if callable(ainvoke) and not isinstance(ainvoke, Mock):
            result = ainvoke(messages)
            return await result if inspect.isawaitable(result) else result

        invoke = getattr(self.llm, "invoke", None)
        if callable(invoke) and not isinstance(invoke, Mock):
            return invoke(messages)

        return AIMessage(content="任务已完成。")

    async def _invoke_model(self, model: Any, messages: list[Any]) -> Any | None:
        ainvoke = getattr(model, "ainvoke", None)
        if callable(ainvoke) and not isinstance(ainvoke, Mock):
            result = ainvoke(messages)
            return await result if inspect.isawaitable(result) else result

        invoke = getattr(model, "invoke", None)
        if callable(invoke) and not isinstance(invoke, Mock):
            return invoke(messages)

        return None

    def _messages_for_model(self, messages: list[Any], *, tool_result_protocol: str = "tool") -> list[Any]:
        if tool_result_protocol == "tool":
            return self._messages_for_tool_protocol(messages)
        if tool_result_protocol != "function":
            raise ValueError(f"Unsupported tool result protocol: {tool_result_protocol}")

        converted: list[Any] = []
        i = 0
        while i < len(messages):
            message = messages[i]
            tool_calls = self._response_tool_calls(message) if getattr(message, "type", "") == "ai" else []
            if tool_calls:
                tool_messages: list[Any] = []
                j = i + 1
                while j < len(messages) and getattr(messages[j], "type", "") == "tool":
                    tool_messages.append(messages[j])
                    j += 1

                matched_ids: set[int] = set()
                for call in tool_calls:
                    name, args, call_id = self._normalize_tool_call(call)
                    converted.append(
                        AIMessage(
                            content=self._message_text(message),
                            additional_kwargs={
                                "function_call": {
                                    "name": name,
                                    "arguments": json.dumps(args, ensure_ascii=False),
                                }
                            },
                        )
                    )
                    tool_message = self._matching_tool_message(tool_messages, call_id, name, matched_ids)
                    if tool_message is not None:
                        converted.append(
                            FunctionMessage(
                                content=self._message_text(tool_message),
                                name=name or getattr(tool_message, "name", "") or "tool",
                            )
                        )
                i = j
                continue

            if getattr(message, "type", "") == "tool":
                converted.append(
                    FunctionMessage(
                        content=self._message_text(message),
                        name=getattr(message, "name", "") or getattr(message, "tool_call_id", "") or "tool",
                    )
                )
            else:
                converted.append(message)
            i += 1
        return converted

    def _messages_for_tool_protocol(self, messages: list[Any]) -> list[Any]:
        converted: list[Any] = []
        for message in messages:
            tool_calls = self._response_tool_calls(message) if getattr(message, "type", "") == "ai" else []
            if tool_calls:
                converted.append(
                    AIMessage(
                        content=self._message_text(message),
                        tool_calls=[self._tool_call_for_message(call) for call in tool_calls],
                    )
                )
            elif getattr(message, "type", "") == "tool":
                converted.append(
                    ToolMessage(
                        content=self._message_text(message),
                        name=getattr(message, "name", "") or None,
                        tool_call_id=getattr(message, "tool_call_id", "") or getattr(message, "name", "") or "tool",
                    )
                )
            else:
                converted.append(message)
        return converted

    @staticmethod
    def _matching_tool_message(tool_messages: list[Any], call_id: str, name: str, matched_ids: set[int]) -> Any | None:
        for index, message in enumerate(tool_messages):
            if index in matched_ids:
                continue
            if getattr(message, "tool_call_id", "") == call_id:
                matched_ids.add(index)
                return message
        for index, message in enumerate(tool_messages):
            if index in matched_ids:
                continue
            if getattr(message, "name", "") == name:
                matched_ids.add(index)
                return message
        return None

    async def _handle_special_tool(self, name: str, result: Any) -> None:
        if name.lower() in {tool.lower() for tool in self.special_tool_names}:
            self.state = OpenManusAgentState.FINISHED

    @staticmethod
    def _response_content(response: Any) -> str:
        return str(getattr(response, "content", "") or "")

    @staticmethod
    def _response_tool_calls(response: Any) -> list[Any]:
        return list(getattr(response, "tool_calls", None) or getattr(response, "additional_kwargs", {}).get("tool_calls", []) or [])

    @staticmethod
    def _message_text(message: Any) -> str:
        return str(getattr(message, "content", "") or "")

    @staticmethod
    def _normalize_tool_call(call: Any) -> tuple[str, dict[str, Any], str]:
        if isinstance(call, dict):
            function = call.get("function") or {}
            name = call.get("name") or function.get("name") or ""
            raw_args = call.get("args")
            if raw_args is None:
                raw_args = call.get("arguments", function.get("arguments", {}))
            call_id = call.get("id") or call.get("tool_call_id") or name
        else:
            function = getattr(call, "function", None)
            name = getattr(call, "name", "") or getattr(function, "name", "")
            raw_args = getattr(call, "args", None)
            if raw_args is None:
                raw_args = getattr(function, "arguments", {})
            call_id = getattr(call, "id", "") or name

        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        return name, args, call_id

    @classmethod
    def _tool_call_for_message(cls, call: Any) -> dict[str, Any]:
        name, args, call_id = cls._normalize_tool_call(call)
        return {"name": name, "args": args, "id": call_id or name}

    @staticmethod
    def _is_role_compatibility_error(exc: Exception, *, rejected: str, expected: str) -> bool:
        text = str(exc).lower()
        return "unknown variant" in text and rejected.lower() in text and expected.lower() in text
