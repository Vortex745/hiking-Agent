"""Factory for UniHarness Agent contexts.

``create_agent`` is the single entry point that mirrors the pinned upstream
``create_agent`` API plus the three approved extension points. Defaults are
safe: all tool groups enabled, no PermissionGate, no approval callback (so
NEEDS_APPROVAL tools reject).
"""

from __future__ import annotations

from typing import Iterable

from .builtin_tools import all_default_tools, tools_for_groups
from .types import (
    AgentContext,
    ApprovalCallback,
    PermissionGate,
    Tool,
    ToolGroup,
)


def _coerce_enabled_groups(
    builtin_tool_groups: Iterable[ToolGroup] | None,
) -> frozenset[ToolGroup]:
    """Normalize the ``builtin_tool_groups`` argument.

    ``None`` means "all groups enabled" (upstream default). An explicit
    iterable is interpreted as the enabled set; anything not listed is
    disabled. Unknown values (including non-enum strings) are rejected
    loudly so a typo cannot silently widen the tool surface.
    """
    if builtin_tool_groups is None:
        return frozenset(ToolGroup)

    enabled: set[ToolGroup] = set()
    unknown: list[str] = []
    for item in builtin_tool_groups:
        if isinstance(item, ToolGroup):
            enabled.add(item)
        else:
            # Capture non-enum values (typos, strings) for the error message.
            unknown.append(str(item))

    if unknown:
        raise ValueError(f"Unknown tool groups: {sorted(unknown)}")
    return frozenset(enabled)


def create_agent(
    *,
    builtin_tool_groups: Iterable[ToolGroup] | None = None,
    permission_gate: PermissionGate | None = None,
    approval_callback: ApprovalCallback | None = None,
) -> AgentContext:
    """Build an AgentContext with the three narrow extension points applied.

    Parameters mirror the approved fork delta:

    * ``builtin_tool_groups`` — iterable of groups to keep enabled. ``None``
      keeps every group (upstream default). Disabling a group removes its
      tools from ``context.tools``, ``context.tool_map``, prompt rebuilds,
      and compaction rebuilds.
    * ``permission_gate`` — product-owned permission evaluator. ``None``
      leaves permission evaluation to upstream semantics (the runtime
      treats absent gate as "allow" for non-approval tools and "reject"
      for approval tools).
    * ``approval_callback`` — handles NEEDS_APPROVAL tools. ``None`` keeps
      upstream rejection behavior: invoking an approval-requiring tool
      raises :class:`UpstreamApprovalRejected`.

    The returned context is the single source of truth for tool visibility.
    Mutating it directly breaks invariants; re-invoke this factory instead.
    """
    enabled = _coerce_enabled_groups(builtin_tool_groups)
    tools = tools_for_groups(enabled)
    tool_map = {tool.name: tool for tool in tools}

    return AgentContext(
        tools=tools,
        enabled_groups=enabled,
        permission_gate=permission_gate,
        approval_callback=approval_callback,
        tool_map=tool_map,
    )


def upstream_default_tools() -> tuple[Tool, ...]:
    """Public helper for contract tests asserting upstream parity."""
    return all_default_tools()
