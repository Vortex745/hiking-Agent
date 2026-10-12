"""Ticket 23 — Bounded one-level subagents for Chat v2.

Subagents are disabled by default. Activation requires:

- ``settings.agent_v2_subagent_enabled = True``
- 14 days of stable post-cutover root-only operation (Ticket 22 gate)

When enabled, the root Agent may dispatch up to 2 fixed-role subagents
per run, with at most 2 running concurrently. Subagents cannot create
sub-subagents (no Agent/Task tools). Raw subagent events do not become
public product events; the root Agent aggregates and owns the final
user-facing state.

This module ships the code components only. Activation is gated by
:func:`agent_v2.startup.is_subagent_authorized`, which reads the
server-side ``agent_v2_subagent_enabled`` flag. The flag defaults to
``False`` so deployments that upgrade to this code remain root-only
until an operator explicitly enables subagents AND the Ticket 22
observation gate has elapsed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agent_v2.uniharness.types import (
    AgentContext,
    PermissionDecision,
    PermissionGate,
    Tool,
    ToolGroup,
)


class SubagentRole(str, Enum):
    """The fixed subagent roles the product supports.

    Clients and models cannot create arbitrary roles — only these three
    server-defined roles may be dispatched. Adding a role requires a
    server-side code change and a new ``SubagentDefinition`` entry in
    :data:`SUBAGENT_DEFINITIONS`.
    """

    RESEARCHER = "researcher"
    ROUTE_PLANNER = "route_planner"
    ARTIFACT_WRITER = "artifact_writer"


# Subagents cannot create lower-level subagents. ``Agent`` and ``Task``
# are the two creation surfaces in UniHarness; both are absent from every
# subagent's tool allowlist and are additionally flagged by
# :func:`is_subagent_creation_tool` so the dispatcher can reject them
# even if a future allowlist accidentally widens.
SUBAGENT_DISALLOWED_TOOLS: tuple[str, ...] = ("Agent", "Task")

# AC3: each run may dispatch at most 2 subagents (lifetime total across
# the run) and at most 2 may run concurrently. These are hard limits —
# the dispatcher rejects any dispatch that would exceed them.
MAX_SUBAGENTS_PER_RUN = 2
MAX_CONCURRENT_SUBAGENTS = 2


@dataclass(frozen=True, slots=True)
class SubagentModelProfile:
    """Server-owned model and budget configuration for a subagent role.

    Mirrors :class:`agent_v2.profiles.ModelProfile`: clients never see
    credentials and cannot override models or limits. Each role has a
    fixed profile assigned at server definition time; the client only
    sends the ``role`` string.
    """

    profile_id: SubagentRole
    main_model: str
    fast_model: str
    context_window: int
    compaction_threshold: int
    output_limit: int


@dataclass(frozen=True, slots=True)
class SubagentDefinition:
    """A fixed subagent role definition.

    ``system_duty`` is a server-owned string that cannot be overridden
    by the client (AC1). ``allowed_tools`` is the product-defined tool
    surface for the role; tools outside this tuple are invisible to the
    subagent (AC2). ``disallowed_tools`` explicitly lists creation tools
    the subagent must never reach — defense in depth alongside the
    allowlist (AC3). ``allowed_mcp_servers`` follows the
    :class:`agent_v2.profiles.CapabilityProfile` semantics: ``None``
    means the role has no MCP concept; a tuple is the server-registered
    server-id allowlist.
    """

    role: SubagentRole
    model_profile: SubagentModelProfile
    system_duty: str
    allowed_tools: tuple[str, ...]
    disallowed_tools: tuple[str, ...] = SUBAGENT_DISALLOWED_TOOLS
    allowed_mcp_servers: tuple[str, ...] | None = None


# ---- Model profiles (server-owned, one per role) -------------------------
#
# Each role gets a fixed ModelProfile. Values mirror the route_plan /
# artifact profiles (deepseek-v4-flash, 64k context, 4k output) since
# subagent tasks are evidence-grounded and need room for tool-call
# overhead. Clients cannot override these values.

_RESEARCHER_MODEL = SubagentModelProfile(
    profile_id=SubagentRole.RESEARCHER,
    main_model="deepseek-v4-flash",
    fast_model="deepseek-v4-flash",
    context_window=64_000,
    compaction_threshold=48_000,
    output_limit=4_096,
)

_ROUTE_PLANNER_MODEL = SubagentModelProfile(
    profile_id=SubagentRole.ROUTE_PLANNER,
    main_model="deepseek-v4-flash",
    fast_model="deepseek-v4-flash",
    context_window=64_000,
    compaction_threshold=48_000,
    output_limit=4_096,
)

_ARTIFACT_WRITER_MODEL = SubagentModelProfile(
    profile_id=SubagentRole.ARTIFACT_WRITER,
    main_model="deepseek-v4-flash",
    fast_model="deepseek-v4-flash",
    context_window=64_000,
    compaction_threshold=48_000,
    output_limit=4_096,
)


# ---- Role definitions (server-fixed, clients cannot create) --------------
#
# The three fixed roles. Tool names match the v1 ``AVAILABLE_TOOL_MAP``
# and the v2 ``CapabilityProfile`` allowlists so subagents reuse the
# existing tool implementations without renaming. MCP server ids match
# the server-side registry in :mod:`agent_v2.mcp_registry`.

_RESEARCHER_DEFINITION = SubagentDefinition(
    role=SubagentRole.RESEARCHER,
    model_profile=_RESEARCHER_MODEL,
    system_duty=(
        "You are a researcher subagent for hiking-related complex research. "
        "You gather evidence from hiking knowledge, route research, web "
        "scraping, and memory. Route research owns candidate search. You do NOT execute shell, write "
        "files, edit files, or call MCP servers. You return structured "
        "findings to the root Agent; the root Agent owns the final "
        "user-facing answer."
    ),
    allowed_tools=(
        "hiking_knowledge_search",
        "route_research",
        "web_scraping",
        "memory_search",
    ),
    disallowed_tools=SUBAGENT_DISALLOWED_TOOLS,
    # researcher has no realtime data needs → no MCP.
    allowed_mcp_servers=None,
)

_ROUTE_PLANNER_DEFINITION = SubagentDefinition(
    role=SubagentRole.ROUTE_PLANNER,
    model_profile=_ROUTE_PLANNER_MODEL,
    system_duty=(
        "You are a route planner subagent. You produce evidence-grounded "
        "route plans using hiking knowledge, route research, gear checklists, "
        "and memory. You may reach the "
        "two AMap MCP servers (weather + map) allowlisted below. You do "
        "NOT execute shell, write files, edit files, or create subagents. "
        "You return a structured plan to the root Agent; the root Agent "
        "owns the final user-facing answer."
    ),
    allowed_tools=(
        "hiking_knowledge_search",
        "route_research",
        "gear_checklist",
        "memory_search",
    ),
    disallowed_tools=SUBAGENT_DISALLOWED_TOOLS,
    # route_planner may reach the two AMap servers registered in the
    # server-side MCP registry. The registry rejects any other id.
    allowed_mcp_servers=("amap_weather", "amap_map"),
)

_ARTIFACT_WRITER_DEFINITION = SubagentDefinition(
    role=SubagentRole.ARTIFACT_WRITER,
    model_profile=_ARTIFACT_WRITER_MODEL,
    system_duty=(
        "You are an artifact writer subagent. You produce structured "
        "documents (PDF reports) inside the run-scoped E2B sandbox using "
        "sandbox_read, sandbox_write, sandbox_edit, sandbox_glob, "
        "sandbox_grep, and generate_pdf. You do NOT execute Bash, reach "
        "MCP servers, or create subagents. You return the produced "
        "artifact path to the root Agent; the root Agent owns the final "
        "user-facing artifact."
    ),
    allowed_tools=(
        "sandbox_read",
        "sandbox_write",
        "sandbox_edit",
        "sandbox_glob",
        "sandbox_grep",
        "generate_pdf",
    ),
    disallowed_tools=SUBAGENT_DISALLOWED_TOOLS,
    # artifact_writer has no realtime data needs → no MCP.
    allowed_mcp_servers=None,
)


SUBAGENT_DEFINITIONS: dict[SubagentRole, SubagentDefinition] = {
    SubagentRole.RESEARCHER: _RESEARCHER_DEFINITION,
    SubagentRole.ROUTE_PLANNER: _ROUTE_PLANNER_DEFINITION,
    SubagentRole.ARTIFACT_WRITER: _ARTIFACT_WRITER_DEFINITION,
}


class UnknownSubagentRoleError(ValueError):
    """Raised when a request asks for a role the product does not define."""


def resolve_subagent_definition(role: SubagentRole | str) -> SubagentDefinition:
    """Return the frozen :class:`SubagentDefinition` for ``role``.

    Accepts the enum or its string value so callers can pass
    ``"researcher"`` without first importing the enum. Unknown values
    raise :class:`UnknownSubagentRoleError` — the dispatcher maps this
    to a rejected dispatch so clients cannot probe for arbitrary roles.
    """
    if isinstance(role, str):
        try:
            role = SubagentRole(role)
        except ValueError as exc:
            raise UnknownSubagentRoleError(
                f"Unknown subagent role: {role!r}. "
                f"Supported: {[r.value for r in SubagentRole]}"
            ) from exc

    definition = SUBAGENT_DEFINITIONS.get(role)
    if definition is None:
        raise UnknownSubagentRoleError(
            f"Role {role!r} is registered but has no definition"
        )
    return definition


def is_subagent_tool_allowed(definition: SubagentDefinition, tool_name: str) -> bool:
    """True iff ``tool_name`` is in the subagent role's allowlist.

    Centralized so the dispatcher, runtime, and tests share the same
    source of truth. A tool not in the list is invisible to the subagent.
    Creation tools (``Agent`` / ``Task``) are never in any allowlist and
    are additionally rejected by :func:`is_subagent_creation_tool` so a
    future allowlist widening cannot accidentally grant subagent
    creation.
    """
    if is_subagent_creation_tool(tool_name):
        return False
    return tool_name in definition.allowed_tools


def is_subagent_creation_tool(tool_name: str) -> bool:
    """True iff ``tool_name`` is a subagent-creation tool.

    Subagents must never create lower-level subagents (AC3). ``Agent``
    and ``Task`` are the two creation surfaces in UniHarness; both are
    rejected regardless of any allowlist.
    """
    return tool_name in SUBAGENT_DISALLOWED_TOOLS


# ---- Runtime permission gate (AC2 + AC3) ---------------------------------


@dataclass(frozen=True, slots=True)
class SubagentPermissionGate:
    """PermissionGate implementation for subagent tool enforcement.

    Enforces AC2: subagent cannot call unauthorized MCP, Bash, or other
    profile tools. Only tools in the SubagentDefinition.allowed_tools
    tuple are permitted; everything else is rejected with a clear reason.

    This gate is injected into the subagent's AgentRuntime so enforcement
    happens at the runtime layer, not just at the dispatcher layer.
    """

    definition: SubagentDefinition

    def evaluate(
        self,
        tool: Tool,
        args: dict[str, Any],
        context: AgentContext,
    ) -> PermissionDecision:
        """Reject tools not in the subagent's allowlist.

        Two checks run in order:

        1. Creation tools (``Agent`` / ``Task``) are rejected first with
           a dedicated reason — AC3 double defense so a future allowlist
           widening cannot accidentally grant subagent creation.
        2. Any other tool outside the role's ``allowed_tools`` tuple is
           rejected as unauthorized (AC2).
        """
        # First check: is this a subagent creation tool? (AC3 double defense)
        if is_subagent_creation_tool(tool.name):
            return PermissionDecision(
                allowed=False,
                reason=f"Subagent cannot create sub-subagents: {tool.name!r} is forbidden",
            )
        # Second check: is this tool in the role's allowlist?
        if not is_subagent_tool_allowed(self.definition, tool.name):
            return PermissionDecision(
                allowed=False,
                reason=f"Tool {tool.name!r} not in {self.definition.role.value} subagent allowlist",
            )
        return PermissionDecision(allowed=True)


# ---- Dispatch records ----------------------------------------------------

# Valid terminal statuses for SubagentDispatch / SubagentResult. The
# dispatcher's complete() method validates result.status against this set
# so a buggy caller cannot put a dispatch into an undefined state.
_VALID_SUBAGENT_STATUSES: frozenset[str] = frozenset(
    {"completed", "failed", "cancelled"}
)

# Valid event_type values for SubagentEvent. record_event() validates
# event_type against this set so the internal event stream stays within
# the documented taxonomy (AC4).
_VALID_SUBAGENT_EVENT_TYPES: frozenset[str] = frozenset(
    {"tool_call", "tool_result", "checkpoint", "approval", "artifact", "memory", "telemetry"}
)


@dataclass(slots=True)
class SubagentDispatch:
    """Record of a single subagent dispatch.

    ``parent_run_id`` and ``parent_subject_id`` link the subagent's
    checkpoint, tools, approval, artifact, memory, and telemetry to the
    parent run and subject (AC5). ``status`` transitions:
    ``running → {completed | failed | cancelled}`` and is terminal.
    ``permission_gate`` carries the AC2 enforcement gate injected into
    the subagent's AgentRuntime at creation time.
    """

    dispatch_id: str
    role: SubagentRole
    parent_run_id: str
    parent_subject_id: str
    started_at: str
    status: str = "running"
    result_summary: str = ""
    permission_gate: SubagentPermissionGate | None = None


@dataclass(frozen=True, slots=True)
class SubagentResult:
    """Result of a completed subagent dispatch.

    ``status`` is one of ``completed`` / ``failed`` / ``cancelled``.
    ``output`` is the subagent's final structured output (string or
    dict); ``tool_invocations`` is the count of tool calls made;
    ``token_usage`` is the subagent's token budget consumption;
    ``evidence_paths`` lists sandbox paths the root Agent may inspect.
    """

    dispatch_id: str
    role: SubagentRole
    status: str
    output: Any = None
    tool_invocations: int = 0
    token_usage: int = 0
    evidence_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SubagentEvent:
    """An internal subagent event.

    ``SubagentEvent`` is NOT a public product event (AC4). The root
    Agent aggregates these into its own user-facing state via
    :meth:`SubagentDispatcher.aggregate_events`; raw subagent events
    never appear on the SSE stream. ``event_type`` is one of
    ``tool_call`` / ``tool_result`` / ``checkpoint`` / ``approval``
    / ``artifact`` / ``memory`` / ``telemetry``.
    """

    dispatch_id: str
    event_type: str
    payload: dict[str, Any]
    sequence: int


class SubagentDispatcher:
    """Manages subagent lifecycle for a single parent run.

    The dispatcher enforces the three bounded-one-level invariants:

    1. Disabled by default — :meth:`can_dispatch` returns ``False``
       unless the dispatcher was constructed with ``enabled=True`` AND
       :meth:`disable` has not been called (AC7). The server-side flag
       ``agent_v2_subagent_enabled`` is the authority; the dispatcher
       only acts on the value injected at construction time.
    2. Fixed roles — only :data:`SUBAGENT_DEFINITIONS` roles may be
       dispatched; arbitrary roles raise
       :class:`UnknownSubagentRoleError` (AC1).
    3. Bounded — at most :data:`MAX_SUBAGENTS_PER_RUN` dispatches per
       run, at most :data:`MAX_CONCURRENT_SUBAGENTS` running at once
       (AC3).

    Subagent events are stored internally and exposed only via
    :meth:`aggregate_events`, which returns a root-usable summary
    (AC4). The raw :class:`SubagentEvent` records never become public
    product events.
    """

    def __init__(
        self,
        parent_run_id: str,
        parent_subject_id: str,
        *,
        enabled: bool = False,
    ) -> None:
        self.parent_run_id = parent_run_id
        self.parent_subject_id = parent_subject_id
        self._enabled = bool(enabled)
        self._dispatches: dict[str, SubagentDispatch] = {}
        self._events: list[SubagentEvent] = []
        self._results: dict[str, SubagentResult] = {}
        self._sequence_counter = 0

    # -- enable / disable (AC7) ----------------------------------------

    def is_enabled(self) -> bool:
        """True iff subagent dispatch is currently enabled."""
        return self._enabled

    def disable(self) -> None:
        """Immediately disable subagent dispatch (AC7).

        In-flight subagents are NOT killed — they run to completion —
        but no new dispatches are accepted. The server-side config
        ``agent_v2_subagent_enabled=False`` causes the next request
        boundary to construct a disabled dispatcher, so this method is
        the in-band escape hatch for runtime operators.
        """
        self._enabled = False

    # -- dispatch lifecycle (AC1 + AC3) --------------------------------

    def can_dispatch(self, role: SubagentRole | str) -> bool:
        """True iff ``role`` may be dispatched right now.

        Four conditions must hold:

        1. The dispatcher is enabled.
        2. The role is a known fixed role (resolved via
           :func:`resolve_subagent_definition` so ``can_dispatch`` and
           :meth:`dispatch` agree on string vs enum handling).
        3. The run has not yet reached ``MAX_CONCURRENT_SUBAGENTS``
           (checked first because it gates "can I open another now").
        4. The run has not yet reached ``MAX_SUBAGENTS_PER_RUN``
           (checked second because it gates lifetime totals).
        """
        if not self._enabled:
            return False
        try:
            resolve_subagent_definition(role)
        except UnknownSubagentRoleError:
            return False
        if self._active_count() >= MAX_CONCURRENT_SUBAGENTS:
            return False
        if len(self._dispatches) >= MAX_SUBAGENTS_PER_RUN:
            return False
        return True

    def dispatch(self, role: SubagentRole, task_description: str) -> SubagentDispatch:
        """Dispatch a subagent for ``role``.

        Raises :class:`UnknownSubagentRoleError` if ``role`` is not a
        fixed role. Raises :class:`RuntimeError` if the dispatcher is
        disabled or the run has reached its dispatch/concurrency limit.

        ``task_description`` is stored on the dispatch record for audit
        and aggregation; it does NOT override the role's
        ``system_duty`` (AC1).

        The concurrency limit is checked before the per-run limit so
        that when both are saturated the operator-facing error names
        the actionable constraint (too many running now) rather than
        the lifetime total.
        """
        if not self._enabled:
            raise RuntimeError("Subagent dispatch is disabled")
        # resolve_subagent_definition validates the role and raises
        # UnknownSubagentRoleError for arbitrary strings/enums.
        definition = resolve_subagent_definition(role)
        if self._active_count() >= MAX_CONCURRENT_SUBAGENTS:
            raise RuntimeError(
                f"Run already has MAX_CONCURRENT_SUBAGENTS="
                f"{MAX_CONCURRENT_SUBAGENTS} active subagents"
            )
        if len(self._dispatches) >= MAX_SUBAGENTS_PER_RUN:
            raise RuntimeError(
                f"Run already dispatched MAX_SUBAGENTS_PER_RUN="
                f"{MAX_SUBAGENTS_PER_RUN} subagents"
            )
        dispatch_id = f"sa-{uuid.uuid4().hex[:16]}"
        started_at = datetime.now(timezone.utc).isoformat()
        gate = SubagentPermissionGate(definition=definition)
        dispatch = SubagentDispatch(
            dispatch_id=dispatch_id,
            role=definition.role,
            parent_run_id=self.parent_run_id,
            parent_subject_id=self.parent_subject_id,
            started_at=started_at,
            status="running",
            result_summary=task_description,
            permission_gate=gate,
        )
        self._dispatches[dispatch_id] = dispatch
        return dispatch

    def complete(self, dispatch_id: str, result: SubagentResult) -> None:
        """Record the completion of a subagent.

        Marks the dispatch as ``completed``/``failed``/``cancelled``
        based on ``result.status``. Unknown ``dispatch_id`` raises
        :class:`KeyError`. Already-completed dispatches are idempotent:
        the first completion wins and later calls are no-ops (this
        preserves the crash-recovery invariant that a subagent's
        terminal state is write-once).
        """
        dispatch = self._dispatches.get(dispatch_id)
        if dispatch is None:
            raise KeyError(f"Unknown dispatch_id: {dispatch_id!r}")
        if dispatch.status != "running":
            # Idempotent: first completion wins.
            return
        if result.status not in _VALID_SUBAGENT_STATUSES:
            raise ValueError(
                f"Invalid subagent status: {result.status!r}. "
                f"Valid: {sorted(_VALID_SUBAGENT_STATUSES)}"
            )
        dispatch.status = result.status
        self._results[dispatch_id] = result

    def active_count(self) -> int:
        """Number of currently-running subagents."""
        return self._active_count()

    def total_dispatched(self) -> int:
        """Total subagents dispatched in this run (lifetime)."""
        return len(self._dispatches)

    def _active_count(self) -> int:
        return sum(1 for d in self._dispatches.values() if d.status == "running")

    # -- subagent runtime creation (AC2 + AC3) -------------------------

    def create_subagent_runtime(self, dispatch_id: str) -> "AgentRuntime":
        """Create an AgentRuntime for the dispatched subagent, with PermissionGate injected.

        approval_callback is deferred to integration time (parent run
        callback injected at chat_v2 integration layer) and gets a
        SubagentPermissionGate that enforces the role's tool allowlist
        (AC2: cannot call unauthorized tools). The Task tool group is
        disabled so the subagent cannot create lower-level subagents
        (AC3: no sub-subagents).

        Raises :class:`KeyError` if ``dispatch_id`` is unknown.
        Raises :class:`RuntimeError` if the dispatch is not in the
        ``running`` status (already completed / failed / cancelled).

        Local imports avoid a circular dependency at module load time:
        ``uniharness.runtime`` and ``uniharness.factory`` are only
        needed when a subagent runtime is actually materialized.
        """
        from agent_v2.uniharness.factory import create_agent
        from agent_v2.uniharness.runtime import AgentRuntime

        dispatch = self._dispatches.get(dispatch_id)
        if dispatch is None:
            raise KeyError(f"Unknown dispatch_id: {dispatch_id!r}")
        if dispatch.status != "running":
            raise RuntimeError(
                f"Dispatch {dispatch_id!r} is not running "
                f"(status={dispatch.status!r})"
            )

        # Task group is excluded so the subagent cannot create
        # sub-subagents (AC3). The gate enforces the role's tool
        # allowlist at the runtime layer (AC2).
        gate = dispatch.permission_gate
        if gate is None:
            # Defensive: dispatch() always sets permission_gate. If we
            # reach here the dispatch record was constructed outside the
            # dispatcher — fail closed by raising.
            raise RuntimeError(
                f"Dispatch {dispatch_id!r} has no permission_gate; "
                f"cannot create subagent runtime"
            )

        context = create_agent(
            builtin_tool_groups={
                ToolGroup.CLI,
                ToolGroup.TODO,
                ToolGroup.SKILLS,
                ToolGroup.WEB,
            },  # No TASK — AC3: no sub-subagents
            permission_gate=gate,
            approval_callback=None,  # Inherited from parent at integration time
        )
        return AgentRuntime(context, is_subagent=True)

    # -- event aggregation (AC4 + AC5) ---------------------------------

    def record_event(
        self, dispatch_id: str, event_type: str, payload: dict[str, Any]
    ) -> SubagentEvent:
        """Record an internal subagent event.

        These events are NOT public product events (AC4). They are
        aggregated by :meth:`aggregate_events` into a root-usable
        summary. Returns the recorded event for caller inspection
        (e.g. telemetry assertions in tests).
        """
        if dispatch_id not in self._dispatches:
            raise KeyError(f"Unknown dispatch_id: {dispatch_id!r}")
        if event_type not in _VALID_SUBAGENT_EVENT_TYPES:
            raise ValueError(
                f"Invalid subagent event type: {event_type!r}. "
                f"Valid: {sorted(_VALID_SUBAGENT_EVENT_TYPES)}"
            )
        self._sequence_counter += 1
        event = SubagentEvent(
            dispatch_id=dispatch_id,
            event_type=event_type,
            payload=dict(payload),
            sequence=self._sequence_counter,
        )
        self._events.append(event)
        return event

    def aggregate_events(self) -> list[dict[str, Any]]:
        """Aggregate subagent events into a root-usable summary (AC4).

        Returns a list of per-dispatch summary dicts. The root Agent
        incorporates these into its own user-facing state; raw
        :class:`SubagentEvent` records never appear on the SSE stream.
        Each summary contains: ``dispatch_id``, ``role``, ``status``,
        ``result_summary``, ``tool_invocations``, ``token_usage``,
        ``evidence_paths``, ``event_count``, ``parent_run_id``,
        ``parent_subject_id`` — enough for the root Agent to own the
        final user-facing state without re-running subagent tools.
        """
        summaries: list[dict[str, Any]] = []
        for dispatch_id, dispatch in self._dispatches.items():
            event_count = sum(
                1 for e in self._events if e.dispatch_id == dispatch_id
            )
            result = self._results.get(dispatch_id)
            summaries.append(
                {
                    "dispatch_id": dispatch_id,
                    "role": dispatch.role.value,
                    "status": dispatch.status,
                    "result_summary": dispatch.result_summary,
                    "tool_invocations": result.tool_invocations if result else 0,
                    "token_usage": result.token_usage if result else 0,
                    "evidence_paths": list(result.evidence_paths) if result else [],
                    "event_count": event_count,
                    "parent_run_id": dispatch.parent_run_id,
                    "parent_subject_id": dispatch.parent_subject_id,
                }
            )
        return summaries
