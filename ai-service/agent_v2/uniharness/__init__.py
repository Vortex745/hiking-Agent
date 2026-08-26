"""Local UniHarness narrow-fork stub.

Replaces the project-controlled UniHarness fork (Ticket 01) with an in-tree
minimal implementation that exposes the three approved extension points:

1. ``builtin_tool_groups`` — selectively disable cli/todo/task/skills/web.
2. ``permission_gate`` — inject product-owned capability and risk rules.
3. ``approval_callback`` — route NEEDS_APPROVAL to durable interrupt/resume.

Defaults preserve "upstream" behavior: all tool groups enabled, empty
PermissionGate, and NEEDS_APPROVAL rejected. The stub is intentionally small
so the project can swap to a real pinned fork commit without changing
callers; only this package needs to be replaced.
"""

from __future__ import annotations

from .builtin_tools import DEFAULT_TOOLS_BY_GROUP, all_default_tools, tools_for_groups
from .factory import create_agent, upstream_default_tools
from .runtime import (
    AgentRuntime,
    ToolInvocation,
    build_prompt,
    rebuild_compaction,
)
from .types import (
    AgentContext,
    ApprovalCallback,
    ApprovalDecision,
    ApprovalRequest,
    PermissionDecision,
    PermissionGate,
    Tool,
    ToolGroup,
    UpstreamApprovalRejected,
)

__all__ = [
    "AgentContext",
    "AgentRuntime",
    "ApprovalCallback",
    "ApprovalDecision",
    "ApprovalRequest",
    "DEFAULT_TOOLS_BY_GROUP",
    "PermissionDecision",
    "PermissionGate",
    "Tool",
    "ToolGroup",
    "ToolInvocation",
    "UpstreamApprovalRejected",
    "all_default_tools",
    "build_prompt",
    "create_agent",
    "rebuild_compaction",
    "tools_for_groups",
    "upstream_default_tools",
]
