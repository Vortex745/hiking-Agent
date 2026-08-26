"""Agent runtime state tracking.

Provides enums, data structures, and the AgentRunRecord for tracking
a single agent execution from start to finish. Compatible with
task_exit.AgentExitStatus for backward compatibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict


class AgentState(str, Enum):
    """Lifecycle states for a single agent run."""

    PLANNING = "planning"
    EXECUTING = "executing"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    WAITING_FOR_USER = "waiting_for_user"


class ExecutionLane(str, Enum):
    """OpenManus execution lane exposed in Agent metadata."""

    REACT = "react"


class ExitReason(str, Enum):
    """Why the agent run ended."""

    NATURAL_END = "natural_end"
    TERMINATE_TOOL = "terminate_tool"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ERROR = "error"
    CLARIFICATION_NEEDED = "clarification_needed"
    STUCK = "stuck"


class ToolResult(TypedDict):
    """Record of a single tool invocation within an agent run."""

    tool_name: str
    args: dict[str, Any]
    result: Any
    error: str | None
    duration_ms: float
    step_index: int


_VALID_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.PLANNING: {AgentState.EXECUTING, AgentState.WAITING_FOR_USER, AgentState.FAILED},
    AgentState.EXECUTING: {
        AgentState.SYNTHESIZING,
        AgentState.COMPLETED,
        AgentState.WAITING_FOR_USER,
        AgentState.BUDGET_EXHAUSTED,
        AgentState.FAILED,
    },
    AgentState.SYNTHESIZING: {AgentState.COMPLETED, AgentState.FAILED},
    AgentState.WAITING_FOR_USER: {AgentState.EXECUTING, AgentState.COMPLETED, AgentState.FAILED},
}

_TERMINAL_STATES = {AgentState.COMPLETED, AgentState.FAILED, AgentState.BUDGET_EXHAUSTED}

_EXIT_REASON_STATE_MAP: dict[ExitReason, AgentState] = {
    ExitReason.BUDGET_EXHAUSTED: AgentState.BUDGET_EXHAUSTED,
    ExitReason.CLARIFICATION_NEEDED: AgentState.WAITING_FOR_USER,
    ExitReason.ERROR: AgentState.FAILED,
    ExitReason.NATURAL_END: AgentState.COMPLETED,
    ExitReason.TERMINATE_TOOL: AgentState.COMPLETED,
    ExitReason.STUCK: AgentState.FAILED,
}

_AGENT_EXIT_STATUS_MAP: dict[str, tuple[AgentState, ExitReason]] = {
    "completed": (AgentState.COMPLETED, ExitReason.NATURAL_END),
    "waiting_for_user": (AgentState.WAITING_FOR_USER, ExitReason.CLARIFICATION_NEEDED),
    "budget_exhausted": (AgentState.BUDGET_EXHAUSTED, ExitReason.BUDGET_EXHAUSTED),
    "error": (AgentState.FAILED, ExitReason.ERROR),
}


@dataclass
class AgentRunRecord:
    """Tracks a single agent execution from start to finish."""

    state: AgentState = AgentState.PLANNING
    lane: ExecutionLane | None = None
    exit_reason: ExitReason | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    step_count: int = 0
    selected_tools: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    metadata: dict[str, Any] = field(default_factory=dict)

    def transition(self, new_state: AgentState) -> None:
        """Validate and apply state transition."""
        if self.state in _TERMINAL_STATES:
            raise ValueError(f"Cannot transition from terminal state {self.state}")
        if new_state not in _VALID_TRANSITIONS.get(self.state, set()):
            raise ValueError(f"Invalid transition: {self.state} → {new_state}")
        self.state = new_state

    def record_tool_result(self, result: ToolResult) -> None:
        """Append a tool result and update step count."""
        self.tool_results.append(result)
        self.step_count = max(self.step_count, result["step_index"])

    def complete(self, exit_reason: ExitReason, **extra_metadata: Any) -> None:
        """Mark execution as completed with the given exit reason."""
        target_state = _EXIT_REASON_STATE_MAP[exit_reason]
        self.transition(target_state)
        self.exit_reason = exit_reason
        self.metadata.update(extra_metadata)
        # Auto-fill step_count from metadata if provided
        step = extra_metadata.get("step") or extra_metadata.get("current_step")
        if step is not None and self.step_count == 0:
            self.step_count = int(step)

    @property
    def duration_ms(self) -> float:
        """Elapsed time since the record was created, in milliseconds."""
        return (time.monotonic() - self.started_at) * 1000

    def to_metadata(self) -> dict[str, Any]:
        """Serialize for SSE done event metadata."""
        duration_ms = round(self.duration_ms, 1)
        if duration_ms <= 0:
            duration_ms = 0.1
        return {
            "lane": self.lane.value if self.lane else None,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "step_count": self.step_count,
            "tool_count": len(self.tool_results),
            "selected_tools": self.selected_tools,
            "duration_ms": duration_ms,
        }

    @classmethod
    def from_exit_status(cls, status: str, reason: str = "", **kwargs: Any) -> AgentRunRecord:
        """Create a record from existing AgentExitStatus string (backward compat)."""
        state, exit_reason = _AGENT_EXIT_STATUS_MAP.get(status, (AgentState.FAILED, ExitReason.ERROR))
        return cls(state=state, exit_reason=exit_reason, **kwargs)
