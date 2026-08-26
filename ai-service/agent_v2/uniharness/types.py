"""Core types for the UniHarness stub.

Value types (Tool, PermissionDecision, ApprovalRequest, ApprovalDecision) are
frozen so callers cannot mutate them. AgentContext and ToolInvocation are
mutable records (slots-only) because they accumulate runtime state; treat them
as write-once-then-read from outside the runtime. Protocols keep the contract
narrow and replaceable when the real fork is swapped in. No I/O, no side
effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable


class ToolGroup(str, Enum):
    """The five built-in tool groups that ``create_agent`` can disable.

    Names match the spec's ``cli``/``todo``/``task``/``skills``/``web``
    grouping. Disabling a group removes its tools from every derived view.
    """

    CLI = "cli"
    TODO = "todo"
    TASK = "task"
    SKILLS = "skills"
    WEB = "web"


@dataclass(frozen=True, slots=True)
class Tool:
    """A registered tool with its group and approval requirement.

    ``requires_approval`` flags tools whose execution must trigger the
    approval callback (or be rejected when no callback is injected).
    """

    name: str
    description: str
    group: ToolGroup
    requires_approval: bool = False


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    """Result of a PermissionGate evaluation.

    ``allowed=False`` must carry a user-safe reason; raw exception text is
    never exposed through this type.
    """

    allowed: bool
    reason: str = ""


@runtime_checkable
class PermissionGate(Protocol):
    """Product-injected permission evaluator.

    Root Agent and allowed subagents share the same instance so capability
    rules cannot diverge between levels. Implementations must be pure: same
    ``(tool, args, context)`` returns the same decision.
    """

    def evaluate(
        self,
        tool: Tool,
        args: dict[str, Any],
        context: "AgentContext",
    ) -> PermissionDecision: ...


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """Payload passed to ``approval_callback`` for NEEDS_APPROVAL tools."""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    risk_summary: str


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """Result of an approval callback.

    ``approved=False`` with ``outcome_unknown=False`` is a clean reject.
    ``outcome_unknown=True`` is reserved for crash-recovery ledger semantics
    handled in Ticket 12; the stub only emits clean decisions.
    """

    approved: bool
    reason: str = ""
    outcome_unknown: bool = False


ApprovalCallback = Callable[[ApprovalRequest], ApprovalDecision]


class UpstreamApprovalRejected(Exception):
    """Raised when NEEDS_APPROVAL is hit without an injected callback.

    Preserves the upstream "no approval handler = reject" semantic so the
    default ``create_agent()`` behavior is safe by construction.
    """


@dataclass(slots=True)
class AgentContext:
    """Immutable-as-used context produced by ``create_agent``.

    ``tools`` is the canonical tuple; ``tool_map`` and ``enabled_groups``
    are derived views kept in sync by the factory. Callers must not mutate
    fields directly — re-invoke ``create_agent`` with new parameters.
    """

    tools: tuple[Tool, ...]
    enabled_groups: frozenset[ToolGroup]
    permission_gate: PermissionGate | None
    approval_callback: ApprovalCallback | None
    tool_map: dict[str, Tool] = field(default_factory=dict)

    def has_tool(self, name: str) -> bool:
        return name in self.tool_map

    def tool_for(self, name: str) -> Tool | None:
        return self.tool_map.get(name)
