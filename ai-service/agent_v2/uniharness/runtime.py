"""Agent runtime: execution shell, subagent guard, prompt/compaction rebuild.

The runtime is deliberately thin. It exists to prove the three extension
points reach every derived view (tool list, prompt, compaction) and to
enforce the subagent invariant. Real UniHarness owns the LLM loop; this
stub only exposes the contract surface that product code depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import (
    AgentContext,
    ApprovalDecision,
    ApprovalRequest,
    PermissionDecision,
    Tool,
    ToolGroup,
    UpstreamApprovalRejected,
)


def build_prompt(context: AgentContext, *, include_tools: bool = True) -> str:
    """Render the system prompt fragment that lists available tools.

    Disabled groups are absent by construction: ``context.tools`` already
    excludes them. The prompt never mentions tools that are not callable.
    """
    if not include_tools:
        return "You are a helpful agent."
    if not context.tools:
        return "You are a helpful agent with no tools."
    lines = ["You are a helpful agent. Available tools:"]
    for tool in context.tools:
        suffix = " (requires approval)" if tool.requires_approval else ""
        lines.append(f"- {tool.name}: {tool.description}{suffix}")
    return "\n".join(lines)


def rebuild_compaction(context: AgentContext) -> str:
    """Render the compaction summary that lists surviving tools.

    Compaction rebuild must reflect the same tool surface as the prompt.
    If a disabled group left a tool referenced in prior compaction state,
    the rebuild drops it — that is the invariant contract tests assert.
    """
    if not context.tools:
        return "Compacted state: no tools available."
    names = ", ".join(tool.name for tool in context.tools)
    return f"Compacted state: tools=[{names}]"


@dataclass(slots=True)
class ToolInvocation:
    """Record of a single attempted tool call."""

    tool_name: str
    args: dict[str, Any]
    approved: bool = False
    permission_reason: str = ""
    result: Any = None
    rejected: bool = False


class AgentRuntime:
    """Execution shell that wires the AgentContext to tool invocation.

    The runtime is the single place where:

    * PermissionGate is consulted before any tool runs.
    * approval_callback is invoked for NEEDS_APPROVAL tools.
    * Subagent creation is guarded against the Task-group bypass.

    Keeping these concerns here (rather than in the factory) means the
    context stays a pure data object and the runtime owns behavior.
    """

    def __init__(self, context: AgentContext, *, is_subagent: bool = False):
        self.context = context
        self.is_subagent = is_subagent
        self.invocations: list[ToolInvocation] = []

    # -- subagent guard ------------------------------------------------

    def can_create_subagent(self) -> bool:
        """Whether this runtime may dispatch a subagent.

        Two conditions must hold:

        1. This runtime is not itself a subagent (one-level limit).
        2. The Task tool group is enabled in the context.

        The second condition closes the bypass where a caller disables the
        Task tool but tries to invoke ``create_subagent`` directly.
        """
        if self.is_subagent:
            return False
        return ToolGroup.TASK in self.context.enabled_groups

    def create_subagent(self) -> "AgentRuntime":
        """Create a child runtime sharing the parent's gate and callback.

        Raises ``RuntimeError`` if ``can_create_subagent`` is False. The
        child inherits ``permission_gate`` and ``approval_callback`` so
        capability rules cannot diverge between levels.
        """
        if not self.can_create_subagent():
            if self.is_subagent:
                raise RuntimeError("Subagents cannot create lower-level subagents")
            raise RuntimeError("Task tool group is disabled; subagent creation unavailable")
        return AgentRuntime(self.context, is_subagent=True)

    # -- tool invocation -----------------------------------------------

    def evaluate_permission(self, tool: Tool, args: dict[str, Any]) -> PermissionDecision:
        """Consult the injected PermissionGate, else default-allow.

        The default-allow for non-gate runs matches upstream behavior:
        UniHarness does not block tools by default; product policy is
        layered on via the gate. NEEDS_APPROVAL tools still go through
        :meth:`request_approval` regardless of the gate's decision.
        """
        if self.context.permission_gate is None:
            return PermissionDecision(allowed=True)
        return self.context.permission_gate.evaluate(tool, args, self.context)

    def request_approval(self, tool: Tool, args: dict[str, Any], tool_call_id: str) -> ApprovalDecision:
        """Resolve a NEEDS_APPROVAL tool via the injected callback.

        Without a callback, raise :class:`UpstreamApprovalRejected` to
        preserve upstream "no handler = reject" semantics. With a callback,
        the decision is returned as-is; crash-recovery ``outcome_unknown``
        handling is a Ticket 12 concern.
        """
        if self.context.approval_callback is None:
            raise UpstreamApprovalRejected(
                f"Tool {tool.name!r} requires approval but no callback is injected"
            )
        request = ApprovalRequest(
            tool_call_id=tool_call_id,
            tool_name=tool.name,
            args=args,
            risk_summary=tool.description,
        )
        return self.context.approval_callback(request)

    def invoke_tool(self, tool_name: str, args: dict[str, Any], *, tool_call_id: str = "") -> ToolInvocation:
        """Attempt a tool invocation, recording the outcome.

        This method does not execute the tool's business logic; the stub
        has no real tool implementations. It only exercises the permission
        and approval seam so contract tests can assert behavior. Real tool
        execution arrives with the v2 tool adapters in Ticket 04.
        """
        tool = self.context.tool_for(tool_name)
        if tool is None:
            invocation = ToolInvocation(
                tool_name=tool_name,
                args=args,
                rejected=True,
                permission_reason=f"Tool {tool_name!r} not in context",
            )
            self.invocations.append(invocation)
            return invocation

        decision = self.evaluate_permission(tool, args)
        if not decision.allowed:
            invocation = ToolInvocation(
                tool_name=tool_name,
                args=args,
                rejected=True,
                permission_reason=decision.reason,
            )
            self.invocations.append(invocation)
            return invocation

        if tool.requires_approval:
            approval = self.request_approval(tool, args, tool_call_id=tool_call_id)
            if not approval.approved:
                invocation = ToolInvocation(
                    tool_name=tool_name,
                    args=args,
                    rejected=True,
                    permission_reason=approval.reason or "approval denied",
                )
                self.invocations.append(invocation)
                return invocation

        invocation = ToolInvocation(tool_name=tool_name, args=args, approved=tool.requires_approval)
        self.invocations.append(invocation)
        return invocation
