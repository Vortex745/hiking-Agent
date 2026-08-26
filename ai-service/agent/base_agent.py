"""OpenManus-style base Agent runtime.

This module provides the reusable step-loop shell used by OpenManus so
specialized agents can share state, memory, max-step handling, and stuck
detection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


class OpenManusAgentState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


@dataclass
class AgentMemory:
    messages: list[BaseMessage] = field(default_factory=list)
    max_messages: int = 100

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def add_messages(self, messages: list[BaseMessage]) -> None:
        for message in messages:
            self.add_message(message)

    def clear(self) -> None:
        self.messages.clear()


@dataclass
class BaseAgent(ABC):
    name: str
    description: str = ""
    system_prompt: str | None = None
    next_step_prompt: str | None = None
    llm: Any | None = None
    memory: AgentMemory = field(default_factory=AgentMemory)
    state: OpenManusAgentState = OpenManusAgentState.IDLE
    max_steps: int = 10
    current_step: int = 0
    duplicate_threshold: int = 2

    @asynccontextmanager
    async def state_context(self, new_state: OpenManusAgentState):
        previous = self.state
        self.state = new_state
        try:
            yield
        except Exception:
            self.state = OpenManusAgentState.ERROR
            raise
        finally:
            if self.state not in {OpenManusAgentState.FINISHED, OpenManusAgentState.ERROR}:
                self.state = previous

    @property
    def messages(self) -> list[BaseMessage]:
        return self.memory.messages

    @messages.setter
    def messages(self, value: list[BaseMessage]) -> None:
        self.memory.messages = value

    def update_memory(self, role: str, content: str, **kwargs: Any) -> None:
        if role == "user":
            self.memory.add_message(HumanMessage(content=content))
        elif role == "system":
            self.memory.add_message(SystemMessage(content=content))
        elif role == "assistant":
            self.memory.add_message(AIMessage(content=content))
        elif role == "tool":
            self.memory.add_message(ToolMessage(content=content, tool_call_id=kwargs.get("tool_call_id", "")))
        else:
            raise ValueError(f"Unsupported message role: {role}")

    async def run(self, request: str | None = None) -> str:
        if self.state != OpenManusAgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")
        if request:
            self.update_memory("user", request)

        results: list[str] = []
        async with self.state_context(OpenManusAgentState.RUNNING):
            while self.current_step < self.max_steps and self.state != OpenManusAgentState.FINISHED:
                self.current_step += 1
                step_result = await self.step()
                if self.is_stuck():
                    self.handle_stuck_state()
                results.append(f"Step {self.current_step}: {step_result}")

            if self.current_step >= self.max_steps and self.state != OpenManusAgentState.FINISHED:
                results.append(f"Terminated: Reached max steps ({self.max_steps})")
                self.state = OpenManusAgentState.IDLE
                self.current_step = 0

        await self.cleanup()
        return "\n".join(results) if results else "No steps executed"

    @abstractmethod
    async def step(self) -> str:
        """Execute a single Agent step."""

    def handle_stuck_state(self) -> None:
        stuck_prompt = (
            "Observed duplicate responses. Consider new strategies and avoid "
            "repeating ineffective paths already attempted."
        )
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt or ''}".strip()

    def is_stuck(self) -> bool:
        if len(self.messages) < 2:
            return False
        last = self.messages[-1]
        content = str(getattr(last, "content", "") or "")
        if not content:
            return False
        duplicate_count = 0
        for message in reversed(self.messages[:-1]):
            if getattr(message, "type", "") == "ai" and getattr(message, "content", None) == content:
                duplicate_count += 1
        return duplicate_count >= self.duplicate_threshold

    async def cleanup(self) -> None:
        return None
