"""Server-side capability profiles for Chat v2.

A profile bundles three things:

1. ``ModelProfile`` — server-owned model + token budget. The client only
   sends ``profile_id``; it cannot override models, credentials, or limits.
2. ``allowed_tools`` — the product-defined tool surface for the scenario.
   Tools outside this tuple are invisible to the run, even if the
   UniHarness context would otherwise expose them.
3. ``allowed_mcp_servers`` — server-registered MCP server ids the profile
   may reach. ``None`` means "no MCP". Client cannot add servers or raw
   tool names.

Ticket 04 only ships ``chat``. Ticket 05 adds ``route_plan``. The
``resolve_profile`` function is the single entry point so callers never
construct profiles ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProfileId(str, Enum):
    """The capability profiles the product supports.

    ``chat`` — ordinary hiking conversation. Low-risk hiking tools only.
    ``route_plan`` — adds route research, Web, and allowlisted map/weather
    MCP (Ticket 05).
    ``artifact`` — adds E2B-scoped file tools + PDF generation (Ticket 09).
        No Bash, no MCP, no host filesystem access.
    ``cowork`` — extends ``artifact`` with E2B-scoped Bash (Ticket 13).
        Authorized only via explicit server-side trusted/staging policy;
        anonymous production users can never reach this profile.
    ``rag`` — knowledge-base Q&A for the RAG page (spec #13 phase 2).
        Knowledge-search only; the grounding gate applies to this profile
        exactly as it does to ``chat``.
    """

    CHAT = "chat"
    ROUTE_PLAN = "route_plan"
    ARTIFACT = "artifact"
    COWORK = "cowork"
    RAG = "rag"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Server-owned model and budget configuration.

    The client never sees credentials. ``main_model`` is the primary
    generator; ``fast_model`` handles classification / slot detection.
    ``context_window`` and ``compaction_threshold`` are in tokens;
    ``output_limit`` caps generated tokens per turn.
    """

    profile_id: ProfileId
    main_model: str
    fast_model: str
    context_window: int
    compaction_threshold: int
    output_limit: int


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    """A profile bundle: model config + tool/MCP allowlist.

    ``allowed_tools`` is a tuple of tool names (matching UniHarness ``Tool.name``
    or product hiking tool names). ``allowed_mcp_servers`` semantics:

    * ``None`` — the profile has no MCP concept at all (e.g. ``chat``).
    * ``()`` (empty tuple) — MCP is enabled for the profile but the
      server-side registry currently lists zero servers.
    * ``(server_id, ...)`` — MCP is enabled and only the listed server ids
      (from the server-side registry) may be reached.
    """

    profile_id: ProfileId
    model: ModelProfile
    allowed_tools: tuple[str, ...]
    allowed_mcp_servers: tuple[str, ...] | None = None


# ---- Model profiles --------------------------------------------------------

_CHAT_MODEL = ModelProfile(
    profile_id=ProfileId.CHAT,
    main_model="deepseek-v4-flash",
    fast_model="deepseek-v4-flash",
    context_window=32_000,
    compaction_threshold=24_000,
    output_limit=2_048,
)

# route_plan model profile: larger context and output budget than chat
# because route planning generates longer, evidence-grounded answers.
# Values are server-owned; clients cannot override.
_ROUTE_PLAN_MODEL = ModelProfile(
    profile_id=ProfileId.ROUTE_PLAN,
    main_model="deepseek-v4-flash",
    fast_model="deepseek-v4-flash",
    context_window=64_000,
    compaction_threshold=48_000,
    output_limit=4_096,
)

# artifact model profile: same budget as route_plan — artifact tasks
# generate structured documents (PDF reports) that need room for the
# plan body plus tool call overhead. Server-owned; clients cannot override.
_ARTIFACT_MODEL = ModelProfile(
    profile_id=ProfileId.ARTIFACT,
    main_model="deepseek-v4-flash",
    fast_model="deepseek-v4-flash",
    context_window=64_000,
    compaction_threshold=48_000,
    output_limit=4_096,
)


# ---- Chat profile (Ticket 04) ---------------------------------------------
#
# Only low-risk hiking knowledge / gear / risk tools are exposed. The
# following are explicitly NOT in the chat profile:
#   - Computer / Bash / Write / Edit (file or shell side effects)
#   - resource_download (filesystem write)
#   - route_research (reserved for route_plan; chat cannot plan routes)
#   - terminal (system command execution)
#   - All MCP servers (chat has no realtime data needs)
#
# The tool names match the existing v1 ``AVAILABLE_TOOL_MAP`` so v2 can
# reuse the hiking tool implementations without renaming.
_CHAT_ALLOWED_TOOLS: tuple[str, ...] = (
    "hiking_knowledge_search",
    "gear_checklist",
    "risk_assessment",
    "memory_save",
    "memory_search",
)


_CHAT_PROFILE = CapabilityProfile(
    profile_id=ProfileId.CHAT,
    model=_CHAT_MODEL,
    allowed_tools=_CHAT_ALLOWED_TOOLS,
    allowed_mcp_servers=None,
)


# ---- route_plan profile (Ticket 05) ---------------------------------------
#
# route_plan extends chat with route research and controlled Web scraping, and
# two allowlisted AMap MCP servers (weather + map). It does NOT include
# Bash, Write, Edit, or any file-operation tool — those belong to the
# ``artifact`` and ``cowork`` profiles (later tickets).
#
# ``allowed_mcp_servers`` is a tuple of server ids registered in
# :mod:`agent_v2.mcp_registry`. The MCP registry is the single source
# of truth for connection parameters and tool allowlists; the profile
# only declares which servers it may reach.
_ROUTE_PLAN_ALLOWED_TOOLS: tuple[str, ...] = (
    "route_compare",
    "report_export",
    # Narrow hiking surface for route planning. Risk assessment remains a
    # chat-level preliminary hint; route planning uses route_compare's
    # structured evidence and safety gate.
    "hiking_knowledge_search",
    "gear_checklist",
    # …then adds route research and source-page retrieval.
    "route_research",
    "web_scraping",
)

_ROUTE_PLAN_PROFILE = CapabilityProfile(
    profile_id=ProfileId.ROUTE_PLAN,
    model=_ROUTE_PLAN_MODEL,
    allowed_tools=_ROUTE_PLAN_ALLOWED_TOOLS,
    # MCP is enabled for route_plan. Only the two AMap servers are
    # allowlisted; the registry rejects any other id at resolution time.
    allowed_mcp_servers=("amap_weather", "amap_map"),
)


# ---- artifact profile (Ticket 09) -----------------------------------------
#
# Spec decision #11: file_operation is not migrated. The artifact profile
# uses E2B-scoped Read, Write, Edit, Glob, Grep and a PDF/report tool.
# Host paths and delete operations are unavailable.
# Spec decision #12: terminal (Bash) exists only in server-authorized
# ``cowork``; it is never callable from ``chat``, ``route_plan``, or
# ``artifact``.
# Spec decision #14: ``artifact`` adds bounded file/artifact capabilities
# without Bash.
# Spec decision #20: E2B workspace is the only writable execution space;
# the sandbox template pins runtime, artifact libraries, and fonts but
# contains no model, MCP, storage, or provider credentials.
#
# The tool names are E2B-scoped equivalents — they execute inside the
# run's sandbox, not on the host. ``generate_pdf`` is the artifact
# production tool that writes a PDF into the sandbox workspace.
_ARTIFACT_ALLOWED_TOOLS: tuple[str, ...] = (
    # E2B-scoped file tools (no host access, no delete).
    "sandbox_read",
    "sandbox_write",
    "sandbox_edit",
    "sandbox_glob",
    "sandbox_grep",
    # Artifact production tool.
    "generate_pdf",
)

_ARTIFACT_PROFILE = CapabilityProfile(
    profile_id=ProfileId.ARTIFACT,
    model=_ARTIFACT_MODEL,
    allowed_tools=_ARTIFACT_ALLOWED_TOOLS,
    # Spec decision #18: MCP tools are filtered by the effective capability
    # profile. artifact has no realtime data needs → no MCP.
    allowed_mcp_servers=None,
)


# ---- cowork profile (Ticket 13) ------------------------------------------
#
# Spec decision #12: terminal (Bash) exists only in server-authorized
# ``cowork``; it is never callable from ``chat``, ``route_plan``, or
# ``artifact``.
# Spec AC1: cowork inherits artifact's E2B-scoped file tools + PDF
# generation, then adds ``sandbox_bash`` for trusted internal scenarios.
#
# The profile is REGISTERED but NOT ENABLED by default. The startup
# guard (Ticket 13 AC2) checks ``settings.agent_v2_cowork_policy`` and
# rejects requests with ``profile_id="cowork"`` when the policy is
# ``disabled``. Only an explicit operator-set policy of ``trusted`` or
# ``staging`` authorizes the cowork profile. Clients cannot upgrade to
# cowork by merely sending the profile id — the server gate is the
# authority, not the request.
_COWORK_ALLOWED_TOOLS: tuple[str, ...] = (
    # Inherits artifact's E2B-scoped file tools + PDF generation.
    "sandbox_read",
    "sandbox_write",
    "sandbox_edit",
    "sandbox_glob",
    "sandbox_grep",
    "generate_pdf",
    # Adds E2B-scoped Bash for trusted internal scenarios.
    "sandbox_bash",
)

_COWORK_MODEL = ModelProfile(
    profile_id=ProfileId.COWORK,
    main_model="deepseek-v4-flash",
    fast_model="deepseek-v4-flash",
    # cowork tasks may run multi-step shell + file workflows; give them
    # the same budget as artifact. Server-owned; clients cannot override.
    context_window=64_000,
    compaction_threshold=48_000,
    output_limit=4_096,
)

_COWORK_PROFILE = CapabilityProfile(
    profile_id=ProfileId.COWORK,
    model=_COWORK_MODEL,
    allowed_tools=_COWORK_ALLOWED_TOOLS,
    # Spec decision #18: MCP tools are filtered by the effective capability
    # profile. cowork has no realtime data needs → no MCP.
    allowed_mcp_servers=None,
)


# ---- rag profile (spec #13 phase 2) ---------------------------------------
#
# The RAG page's knowledge Q&A runs as a kernel profile. Tool surface is
# the knowledge search alone: retrieval planning / parent back-tracing /
# compression happen inside the tool (embedded retrieve subgraph), and the
# grounding fail-closed gate (same predicate as chat) governs output.
# No memory tools, no MCP, no sandbox.
_RAG_MODEL = ModelProfile(
    profile_id=ProfileId.RAG,
    main_model="deepseek-v4-flash",
    fast_model="deepseek-v4-flash",
    context_window=32_000,
    compaction_threshold=24_000,
    output_limit=2_048,
)

_RAG_ALLOWED_TOOLS: tuple[str, ...] = (
    "hiking_knowledge_search",
)

_RAG_PROFILE = CapabilityProfile(
    profile_id=ProfileId.RAG,
    model=_RAG_MODEL,
    allowed_tools=_RAG_ALLOWED_TOOLS,
    allowed_mcp_servers=None,
)


# Ticket 04 ships ``chat``; Ticket 05 adds ``route_plan``; Ticket 09 adds
# ``artifact``; Ticket 13 adds ``cowork``; spec #13 phase 2 adds ``rag``.
# The enum values are the wire-level ``profile_id`` strings clients send.
_PROFILES: dict[ProfileId, CapabilityProfile] = {
    ProfileId.CHAT: _CHAT_PROFILE,
    ProfileId.ROUTE_PLAN: _ROUTE_PLAN_PROFILE,
    ProfileId.ARTIFACT: _ARTIFACT_PROFILE,
    ProfileId.COWORK: _COWORK_PROFILE,
    ProfileId.RAG: _RAG_PROFILE,
}


class UnknownProfileError(ValueError):
    """Raised when a request asks for a profile the product does not define."""


def resolve_profile(profile_id: ProfileId | str) -> CapabilityProfile:
    """Return the frozen :class:`CapabilityProfile` for ``profile_id``.

    Accepts the enum or its string value so Pydantic / JSON requests can
    pass ``"chat"`` without first importing the enum. Unknown values
    raise :class:`UnknownProfileError` — the SSE endpoint maps this to a
    400 response so clients cannot discover profiles by probing.
    """
    if isinstance(profile_id, str):
        try:
            profile_id = ProfileId(profile_id)
        except ValueError as exc:
            raise UnknownProfileError(
                f"Unknown profile_id: {profile_id!r}. "
                f"Supported: {[p.value for p in ProfileId]}"
            ) from exc

    profile = _PROFILES.get(profile_id)
    if profile is None:
        raise UnknownProfileError(
            f"Profile {profile_id!r} is registered but has no capability bundle"
        )
    return profile


def is_tool_allowed(profile: CapabilityProfile, tool_name: str) -> bool:
    """True iff ``tool_name`` is in the profile's allowlist.

    Centralized so the SSE adapter, the runtime, and tests all share the
    same source of truth. A tool not in the list is invisible to the run,
    regardless of what UniHarness ``AgentContext`` would otherwise expose.
    """
    return tool_name in profile.allowed_tools
