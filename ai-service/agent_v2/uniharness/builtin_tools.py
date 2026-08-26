"""Default built-in tool groups for the UniHarness stub.

Mirrors the upstream tool surface so ``create_agent()`` defaults match the
pinned revision. Disabling a group via ``builtin_tool_groups`` removes every
tool listed here from the resulting AgentContext.
"""

from __future__ import annotations

from .types import Tool, ToolGroup


# CLI group — file system and shell surface. Bash/Write require approval.
_CLI_TOOLS: tuple[Tool, ...] = (
    Tool(name="Bash", description="Execute shell commands", group=ToolGroup.CLI, requires_approval=True),
    Tool(name="Read", description="Read a file from the workspace", group=ToolGroup.CLI),
    Tool(name="Write", description="Write a file in the workspace", group=ToolGroup.CLI, requires_approval=True),
    Tool(name="Edit", description="Apply a targeted edit to a file", group=ToolGroup.CLI),
    Tool(name="Glob", description="Find files by glob pattern", group=ToolGroup.CLI),
    Tool(name="Grep", description="Search file contents", group=ToolGroup.CLI),
)

# Todo group — short-horizon task tracking.
_TODO_TOOLS: tuple[Tool, ...] = (
    Tool(name="TodoWrite", description="Update the task todo list", group=ToolGroup.TODO),
)

# Task group — subagent dispatch. Disabling also blocks ``create_subagent``.
_TASK_TOOLS: tuple[Tool, ...] = (
    Tool(name="Task", description="Dispatch a bounded subagent", group=ToolGroup.TASK),
)

# Skills group — declarative skill invocation.
_SKILLS_TOOLS: tuple[Tool, ...] = (
    Tool(name="Skill", description="Invoke a named skill", group=ToolGroup.SKILLS),
)

# Web group — search and fetch.
_WEB_TOOLS: tuple[Tool, ...] = (
    Tool(name="WebSearch", description="Search the web", group=ToolGroup.WEB),
    Tool(name="WebFetch", description="Fetch a URL", group=ToolGroup.WEB),
)


DEFAULT_TOOLS_BY_GROUP: dict[ToolGroup, tuple[Tool, ...]] = {
    ToolGroup.CLI: _CLI_TOOLS,
    ToolGroup.TODO: _TODO_TOOLS,
    ToolGroup.TASK: _TASK_TOOLS,
    ToolGroup.SKILLS: _SKILLS_TOOLS,
    ToolGroup.WEB: _WEB_TOOLS,
}


def all_default_tools() -> tuple[Tool, ...]:
    """Return the full default tool surface in canonical group order."""
    return tuple(
        tool
        for group in ToolGroup
        for tool in DEFAULT_TOOLS_BY_GROUP[group]
    )


def tools_for_groups(enabled: frozenset[ToolGroup]) -> tuple[Tool, ...]:
    """Return only the tools whose group is in ``enabled``.

    Order is stable: groups iterate in ``ToolGroup`` definition order, tools
    iterate in their group tuple order. This stability is what lets contract
    tests assert exact tool lists when a group is disabled.
    """
    return tuple(
        tool
        for group in ToolGroup
        if group in enabled
        for tool in DEFAULT_TOOLS_BY_GROUP[group]
    )
