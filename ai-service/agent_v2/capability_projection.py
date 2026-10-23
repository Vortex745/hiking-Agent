"""BCDEF ticket 10 — capability / MCP / risk / effect policy projection.

The projection is the ONE execution-time view of what an admitted run may
do. It is built from the admission-frozen policy snapshot (BCDEF ticket
08/09) and never reads the live profile, risk table, MCP registry grants
or any other mutable global configuration. Every tool source — builtin
product tools, skills, sandbox tools and MCP tools (including schemas
discovered lazily at runtime) — gets a stable capability identity and
passes the same :meth:`CapabilityProjection.authorize` boundary before a
side effect may be claimed.

Identities are name-derived, never registration-order-derived::

    builtin:<tool>              product builtin tool
    sandbox:<tool>              sandbox-scoped tool (incl. generate_pdf)
    skill:<skill>               declarative skill
    mcp:<server>:<tool>         MCP tool (kernel name mcp__<server>__<tool>)

The run policy hash is part of the projection identity. A projection can
be checked against an expected policy hash (:meth:`assert_policy_hash`);
a mismatch is the stable ``policy_hash_mismatch`` failure and never falls
back to live configuration.

Failure is always closed: incomplete capability/risk/MCP input raises
:class:`CapabilityProjectionError` (``capability_projection_failed``)
instead of silently exposing an unconstrained surface.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from agent_v2.execution_contract import (
    BoundaryError,
    ExecutionErrorCode,
    policy_hash,
)
from agent_v2.permissions import (
    PermissionDecision,
    RiskLevel,
    approval_risk_threshold,
    tool_risk_policy,
)

PROJECTION_VERSION = 1

SOURCE_BUILTIN = "builtin"
SOURCE_SANDBOX = "sandbox"
SOURCE_SKILL = "skill"
SOURCE_MCP = "mcp"

_VALID_RISKS = frozenset(level.value for level in RiskLevel)
_RISK_ORDER: dict[str, int] = {
    RiskLevel.LOW.value: 0,
    RiskLevel.MEDIUM.value: 1,
    RiskLevel.HIGH.value: 2,
}

# Stable failure reasons (additive detail next to the stable wire code).
REASON_INVALID_POLICY = "invalid_policy"
REASON_INVALID_CAPABILITY = "invalid_capability"
REASON_MISSING_RISK = "missing_risk"
REASON_INVALID_RISK = "invalid_risk"
REASON_MISSING_MCP_SERVICE = "missing_mcp_service"
REASON_INVALID_MCP_SERVICE = "invalid_mcp_service"
REASON_MCP_TOOL_NOT_IN_SERVICE = "mcp_tool_not_in_service"
REASON_POLICY_HASH_MISMATCH = "policy_hash_mismatch"

NEXT_ACTION_MANUAL = "manual"
NEXT_ACTION_JANITOR = "janitor"


class CapabilityProjectionError(BoundaryError):
    """The frozen policy cannot be projected into a complete capability set.

    ``code`` is stable (``capability_projection_failed`` or
    ``policy_hash_mismatch``); ``reason``/``detail``/``next_action`` make
    the failure explainable from snapshot and replay without parsing
    messages. The failure is never recoverable in the same run: an
    incomplete projection must not degrade into unconstrained execution.
    """

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        detail: str = "",
        next_action: str = NEXT_ACTION_MANUAL,
        code: ExecutionErrorCode = (
            ExecutionErrorCode.CAPABILITY_PROJECTION_FAILED
        ),
    ) -> None:
        super().__init__(code, message, recoverable=False)
        self.reason = reason
        self.detail = detail
        self.next_action = next_action

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code.value,
            "reason": self.reason,
            "recoverable": False,
            "next_action": self.next_action,
            "message": self.message,
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload


class ToolNotAuthorizedError(BoundaryError):
    """A tool call was refused by the frozen capability projection.

    Stable wire code ``tool_not_authorized``; ``reason`` is the gate-level
    reason (``unknown_tool``, ``unauthorized_mcp_tool``, ...). Used by
    effect-bearing paths that must stop before the effect claim.
    """

    def __init__(self, reason: str, tool: str, message: str = "") -> None:
        super().__init__(
            ExecutionErrorCode.TOOL_NOT_AUTHORIZED,
            message or f"tool {tool!r} is not authorized by the frozen policy",
            recoverable=False,
        )
        self.reason = reason
        self.tool = tool

    def to_payload(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "reason": self.reason,
            "tool": self.tool,
            "recoverable": False,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Stable identity helpers
# ---------------------------------------------------------------------------


def builtin_capability_id(tool_name: str) -> str:
    return f"{SOURCE_BUILTIN}:{tool_name}"


def sandbox_capability_id(tool_name: str) -> str:
    return f"{SOURCE_SANDBOX}:{tool_name}"


def skill_capability_id(skill_name: str) -> str:
    return f"{SOURCE_SKILL}:{skill_name}"


def mcp_capability_id(server_id: str, tool_name: str) -> str:
    return f"{SOURCE_MCP}:{server_id}:{tool_name}"


def mcp_kernel_tool_name(server_id: str, tool_name: str) -> str:
    return f"mcp__{server_id}__{tool_name}"


def split_mcp_kernel_tool_name(tool_name: str) -> tuple[str, str] | None:
    """Parse ``mcp__<server>__<tool>`` into ``(server, tool)``."""
    if not tool_name.startswith("mcp__"):
        return None
    rest = tool_name[len("mcp__"):]
    if "__" not in rest:
        return None
    server, tool = rest.split("__", 1)
    if not server or not tool:
        return None
    return server, tool


def classify_tool_source(tool_name: str) -> str:
    """Name-derived source classification (never registration order)."""
    if split_mcp_kernel_tool_name(tool_name) is not None:
        return SOURCE_MCP
    if tool_name == "generate_pdf" or tool_name.startswith("sandbox_"):
        return SOURCE_SANDBOX
    if tool_name.startswith("skill:"):
        return SOURCE_SKILL
    return SOURCE_BUILTIN


# ---------------------------------------------------------------------------
# Projected records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProjectedTool:
    """One authorized capability of the frozen policy."""

    capability_id: str
    tool_name: str
    source: str
    risk: str
    requires_approval: bool
    description: str = ""
    server_id: str = ""
    mcp_tool: str = ""
    schema_summary: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "capability_id": self.capability_id,
            "tool": self.tool_name,
            "source": self.source,
            "risk": self.risk,
            "requires_approval": self.requires_approval,
        }
        if self.server_id:
            payload["server_id"] = self.server_id
        if self.mcp_tool:
            payload["mcp_tool"] = self.mcp_tool
        if self.schema_summary:
            payload["schema_summary"] = dict(self.schema_summary)
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True, slots=True)
class ProjectedMcpService:
    """Frozen MCP service grant: identity, transport and tool allowlist."""

    server_id: str
    tools: tuple[str, ...]
    transport: str = ""
    timeout_seconds: float = 0.0
    max_output_bytes: int = 0
    description: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "server_id": self.server_id,
            "tool_count": len(self.tools),
            "tools": list(self.tools),
        }
        if self.transport:
            payload["transport"] = self.transport
        if self.timeout_seconds:
            payload["timeout_seconds"] = self.timeout_seconds
        if self.max_output_bytes:
            payload["max_output_bytes"] = self.max_output_bytes
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True, slots=True)
class ToolAuthorization:
    """Result of the single tool authorization boundary."""

    decision: PermissionDecision
    risk: RiskLevel | None
    reason: str
    capability_id: str = ""
    tool_name: str = ""
    source: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision is PermissionDecision.ALLOW


@dataclass(frozen=True, slots=True)
class McpDiscoveryResult:
    """Deterministic result of a lazy MCP discovery/describe request.

    ``discovered`` entries already exist in the frozen projection;
    ``denied`` entries were advertised by the runtime server but are not
    part of the admitted capability set. ``missing`` lists projected
    tools the server did not advertise (still authorized, currently
    unavailable) so replay can explain the difference.
    """

    server_id: str
    discovered: tuple[dict[str, Any], ...]
    denied: tuple[dict[str, Any], ...]
    missing: tuple[dict[str, Any], ...]
    policy_hash: str
    projection_hash: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "discovered": [dict(entry) for entry in self.discovered],
            "denied": [dict(entry) for entry in self.denied],
            "missing": [dict(entry) for entry in self.missing],
            "policy_hash": self.policy_hash,
            "projection_hash": self.projection_hash,
        }


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CapabilityProjection:
    """Admission-frozen, deterministic capability/risk/effect projection."""

    run_id: str
    policy_hash: str
    projection_hash: str
    profile_id: str
    tools: tuple[ProjectedTool, ...]
    mcp_services: tuple[ProjectedMcpService, ...]
    approval_threshold: str
    tool_risk: Mapping[str, str]
    tool_groups: tuple[str, ...] = ()
    version: int = PROJECTION_VERSION
    _by_name: Mapping[str, ProjectedTool] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_by_name",
            {tool.tool_name: tool for tool in self.tools},
        )

    # -- identity ---------------------------------------------------------

    @property
    def capability_ids(self) -> tuple[str, ...]:
        return tuple(tool.capability_id for tool in self.tools)

    @property
    def allowed_tool_names(self) -> frozenset[str]:
        return frozenset(tool.tool_name for tool in self.tools)

    def tool(self, tool_name: str) -> ProjectedTool | None:
        return self._by_name.get(tool_name)

    def tool_by_id(self, capability_id: str) -> ProjectedTool | None:
        for tool in self.tools:
            if tool.capability_id == capability_id:
                return tool
        return None

    def mcp_service(self, server_id: str) -> ProjectedMcpService | None:
        for service in self.mcp_services:
            if service.server_id == server_id:
                return service
        return None

    def assert_policy_hash(self, expected_policy_hash: str) -> None:
        """Refuse a projection presented under another run's policy hash."""
        expected = (expected_policy_hash or "").strip().lower()
        actual = self.policy_hash.strip().lower()
        if not expected:
            raise CapabilityProjectionError(
                REASON_POLICY_HASH_MISMATCH,
                "expected policy hash is empty",
                code=ExecutionErrorCode.POLICY_HASH_MISMATCH,
            )
        if expected != actual:
            raise CapabilityProjectionError(
                REASON_POLICY_HASH_MISMATCH,
                f"projection policy hash {actual} does not match expected "
                f"{expected}",
                code=ExecutionErrorCode.POLICY_HASH_MISMATCH,
            )

    # -- authorization ----------------------------------------------------

    def authorize(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None = None,
        *,
        approval_granted: bool = False,
        expected_policy_hash: str | None = None,
    ) -> ToolAuthorization:
        """The ONE tool authorization check for every tool source.

        Unknown capabilities are denied (``unknown_tool`` /
        ``unauthorized_mcp_tool`` / ``unknown_mcp_server``). A capability
        at or above the frozen approval threshold returns
        ``needs_approval`` unless the caller proves a durable approval
        (``approval_granted=True``). ``args`` is reserved for
        parameter-based escalation and never widens the allowlist.
        """
        del args  # parameter-based escalation is a future extension
        if expected_policy_hash is not None:
            self.assert_policy_hash(expected_policy_hash)

        projected = self._by_name.get(tool_name)
        if projected is None:
            parsed = split_mcp_kernel_tool_name(tool_name)
            if parsed is not None:
                server_id, _mcp_tool = parsed
                if self.mcp_service(server_id) is None:
                    return ToolAuthorization(
                        decision=PermissionDecision.DENY,
                        risk=None,
                        reason="unknown_mcp_server",
                        tool_name=tool_name,
                        source=SOURCE_MCP,
                    )
                return ToolAuthorization(
                    decision=PermissionDecision.DENY,
                    risk=None,
                    reason="unauthorized_mcp_tool",
                    capability_id=mcp_capability_id(server_id, _mcp_tool),
                    tool_name=tool_name,
                    source=SOURCE_MCP,
                )
            return ToolAuthorization(
                decision=PermissionDecision.DENY,
                risk=None,
                reason="unknown_tool",
                tool_name=tool_name,
                source=classify_tool_source(tool_name),
            )

        risk = RiskLevel(projected.risk)
        if projected.requires_approval and not approval_granted:
            return ToolAuthorization(
                decision=PermissionDecision.NEEDS_APPROVAL,
                risk=risk,
                reason="high_risk_requires_approval",
                capability_id=projected.capability_id,
                tool_name=tool_name,
                source=projected.source,
            )
        return ToolAuthorization(
            decision=PermissionDecision.ALLOW,
            risk=risk,
            reason="approval_granted" if projected.requires_approval else "low_risk_allowed",
            capability_id=projected.capability_id,
            tool_name=tool_name,
            source=projected.source,
        )

    def require_authorized(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None = None,
        *,
        approval_granted: bool = False,
        expected_policy_hash: str | None = None,
    ) -> ToolAuthorization:
        """Authorize or raise a stable :class:`ToolNotAuthorizedError`.

        Effect-bearing paths use this so a rejected tool stops BEFORE the
        effect claim (no external side effect, no executable ledger row).
        """
        authorization = self.authorize(
            tool_name,
            args,
            approval_granted=approval_granted,
            expected_policy_hash=expected_policy_hash,
        )
        if authorization.decision is PermissionDecision.DENY:
            raise ToolNotAuthorizedError(
                authorization.reason, tool_name
            )
        return authorization

    # -- lazy MCP discovery / describe -----------------------------------

    def describe(self, tool_name: str) -> dict[str, Any]:
        """Auditable description of a (possibly not yet projected) tool."""
        projected = self._by_name.get(tool_name)
        payload: dict[str, Any] = {
            "tool": tool_name,
            "projected": projected is not None,
            "policy_hash": self.policy_hash,
            "projection_hash": self.projection_hash,
            "run_id": self.run_id,
        }
        if projected is not None:
            payload.update(projected.to_payload())
            payload["projected"] = True
            payload["authorized"] = True
            return payload
        parsed = split_mcp_kernel_tool_name(tool_name)
        if parsed is not None:
            server_id, mcp_tool = parsed
            payload.update(
                {
                    "capability_id": mcp_capability_id(server_id, mcp_tool),
                    "source": SOURCE_MCP,
                    "server_id": server_id,
                    "mcp_tool": mcp_tool,
                    "schema_summary": {"kind": "mcp", "deferred": True},
                    "authorized": False,
                    "reason": (
                        "unknown_mcp_server"
                        if self.mcp_service(server_id) is None
                        else "unauthorized_mcp_tool"
                    ),
                }
            )
            return payload
        payload.update(
            {
                "capability_id": classify_tool_source(tool_name)
                + ":"
                + tool_name,
                "source": classify_tool_source(tool_name),
                "authorized": False,
                "reason": "unknown_tool",
            }
        )
        return payload

    def discover_mcp_tools(
        self, server_id: str, advertised_tools: Iterable[str]
    ) -> McpDiscoveryResult:
        """Reconcile runtime-advertised MCP tools with the frozen projection.

        A tool advertised by the live server but absent from the frozen
        projection is DENIED forever for this run: lazy discovery can
        confirm/describe a schema, never widen the capability set.
        """
        service = self.mcp_service(server_id)
        advertised = tuple(
            dict.fromkeys(
                str(tool) for tool in advertised_tools if str(tool).strip()
            )
        )
        if service is None:
            denied = tuple(
                {
                    "tool": mcp_kernel_tool_name(server_id, tool),
                    "capability_id": mcp_capability_id(server_id, tool),
                    "reason": "unknown_mcp_server",
                }
                for tool in advertised
            )
            return McpDiscoveryResult(
                server_id=server_id,
                discovered=(),
                denied=denied,
                missing=(),
                policy_hash=self.policy_hash,
                projection_hash=self.projection_hash,
            )

        projected_tools = {
            tool.mcp_tool: tool
            for tool in self.tools
            if tool.source == SOURCE_MCP and tool.server_id == server_id
        }
        discovered: list[dict[str, Any]] = []
        denied: list[dict[str, Any]] = []
        advertised_set = set(advertised)
        for tool in advertised:
            projected = projected_tools.get(tool)
            if projected is None:
                denied.append(
                    {
                        "tool": mcp_kernel_tool_name(server_id, tool),
                        "capability_id": mcp_capability_id(server_id, tool),
                        "reason": "not_projected",
                    }
                )
                continue
            discovered.append(
                {
                    "tool": projected.tool_name,
                    "capability_id": projected.capability_id,
                    "risk": projected.risk,
                    "requires_approval": projected.requires_approval,
                }
            )
        missing = [
            {
                "tool": projected.tool_name,
                "capability_id": projected.capability_id,
                "reason": "not_advertised",
            }
            for name, projected in sorted(projected_tools.items())
            if name not in advertised_set
        ]
        return McpDiscoveryResult(
            server_id=server_id,
            discovered=tuple(discovered),
            denied=tuple(denied),
            missing=tuple(missing),
            policy_hash=self.policy_hash,
            projection_hash=self.projection_hash,
        )

    # -- event/replay metadata -------------------------------------------

    def event_meta(self, tool_name: str) -> dict[str, Any]:
        """Additive run/policy binding merged into tool_call/tool_result."""
        meta: dict[str, Any] = {
            "run_id": self.run_id,
            "policy_hash": self.policy_hash,
            "projection_hash": self.projection_hash,
        }
        projected = self._by_name.get(tool_name)
        if projected is not None:
            meta["capability_id"] = projected.capability_id
            meta["source"] = projected.source
        return meta

    # -- snapshot/replay projection --------------------------------------

    def to_payload(self) -> dict[str, Any]:
        """Canonical projection content (identity + frozen grants)."""
        return {
            "version": self.version,
            "policy_hash": self.policy_hash,
            "projection_hash": self.projection_hash,
            "profile_id": self.profile_id,
            "approval_threshold": self.approval_threshold,
            "tool_groups": list(self.tool_groups),
            "tool_risk": dict(sorted(self.tool_risk.items())),
            "capabilities": [tool.to_payload() for tool in self.tools],
            "mcp_services": [
                service.to_payload() for service in self.mcp_services
            ],
        }

    def summary(self) -> dict[str, Any]:
        """Additive snapshot/replay summary of the projection."""
        return {
            "version": self.version,
            "policy_hash": self.policy_hash,
            "projection_hash": self.projection_hash,
            "profile_id": self.profile_id,
            "compatible": True,
            "capability_count": len(self.tools),
            "capability_ids": sorted(self.capability_ids),
            "mcp_services": [
                service.to_payload() for service in self.mcp_services
            ],
            "risk_policy": {
                "approval_threshold": self.approval_threshold,
                "tool_risk": dict(sorted(self.tool_risk.items())),
            },
            "approval_policy": {
                "requires_approval_at_or_above": self.approval_threshold,
                "capability_ids": [
                    tool.capability_id
                    for tool in self.tools
                    if tool.requires_approval
                ],
            },
        }


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _projection_error(reason: str, message: str, *, detail: str = "") -> CapabilityProjectionError:
    return CapabilityProjectionError(
        reason, message, detail=detail, next_action=NEXT_ACTION_JANITOR
    )


def _normalize_risk(value: Any, *, label: str) -> str:
    if isinstance(value, RiskLevel):
        return value.value
    text = str(value or "").strip().lower()
    if text not in _VALID_RISKS:
        raise _projection_error(
            REASON_INVALID_RISK,
            f"{label} has an unknown risk level {value!r}",
            detail=label,
        )
    return text


def _normalize_tool_risk_map(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise _projection_error(
            REASON_INVALID_CAPABILITY,
            "effect_policy.tool_risk must be a mapping",
            detail="effect_policy.tool_risk",
        )
    normalized: dict[str, str] = {}
    for name, risk in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise _projection_error(
                REASON_INVALID_RISK,
                "tool_risk keys must be non-empty strings",
                detail="effect_policy.tool_risk",
            )
        normalized[name] = _normalize_risk(risk, label=f"tool_risk.{name}")
    return normalized


def _normalize_skills(raw: Any) -> list[tuple[str, str | None]]:
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise _projection_error(
            REASON_INVALID_CAPABILITY,
            "capability.skills must be a list",
            detail="capability.skills",
        )
    skills: list[tuple[str, str | None]] = []
    for entry in raw:
        if isinstance(entry, str):
            name = entry.strip()
            risk = None
        elif isinstance(entry, Mapping):
            name = str(entry.get("name") or "").strip()
            risk = entry.get("risk")
            risk = None if risk is None else str(risk)
        else:
            raise _projection_error(
                REASON_INVALID_CAPABILITY,
                "skill entries must be strings or mappings",
                detail="capability.skills",
            )
        if not name:
            raise _projection_error(
                REASON_INVALID_CAPABILITY,
                "skill entries require a non-empty name",
                detail="capability.skills",
            )
        skills.append((name, risk))
    return skills


def _mcp_tool_entry_risk(raw_service: Any, tool_name: str) -> str | None:
    """Frozen inline risk for one MCP tool, if the summary carries it."""
    if isinstance(raw_service, Mapping):
        for entry in raw_service.get("tools") or ():
            if isinstance(entry, Mapping) and str(entry.get("name")) == tool_name:
                if entry.get("risk") is not None:
                    return _normalize_risk(
                        entry.get("risk"), label=f"mcp.tool.{tool_name}"
                    )
    return None


def _resolve_mcp_tool_risk(
    server_id: str,
    tool_name: str,
    raw_service: Any,
    tool_risk: Mapping[str, str],
) -> str | None:
    inline = _mcp_tool_entry_risk(raw_service, tool_name)
    if inline is not None:
        return inline
    return (
        tool_risk.get(mcp_kernel_tool_name(server_id, tool_name))
        or tool_risk.get(tool_name)
    )


def _normalize_mcp_service(
    server_id: str, raw: Any, tool_risk: Mapping[str, str]
) -> ProjectedMcpService:
    if not isinstance(raw, Mapping):
        raise _projection_error(
            REASON_MISSING_MCP_SERVICE,
            f"policy has no frozen MCP service summary for {server_id!r}",
            detail=server_id,
        )
    raw_tools = raw.get("tools")
    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes)):
        raise _projection_error(
            REASON_INVALID_MCP_SERVICE,
            f"MCP service {server_id!r} has no tools list",
            detail=server_id,
        )
    tools: list[str] = []
    for entry in raw_tools:
        if isinstance(entry, str):
            name = entry.strip()
            risk = None
        elif isinstance(entry, Mapping):
            name = str(entry.get("name") or "").strip()
            risk = entry.get("risk")
            risk = None if risk is None else str(risk)
        else:
            raise _projection_error(
                REASON_INVALID_MCP_SERVICE,
                f"MCP service {server_id!r} tool entries are malformed",
                detail=server_id,
            )
        if not name:
            raise _projection_error(
                REASON_INVALID_MCP_SERVICE,
                f"MCP service {server_id!r} has an empty tool name",
                detail=server_id,
            )
        # Validate the risk now; the projection builder resolves the same
        # value from the inline summary / frozen risk map.
        full_name = mcp_kernel_tool_name(server_id, name)
        resolved = (
            _normalize_risk(risk, label=f"mcp.{server_id}.{name}")
            if risk is not None
            else tool_risk.get(full_name) or tool_risk.get(name)
        )
        if resolved is None:
            raise _projection_error(
                REASON_MISSING_RISK,
                f"MCP tool {full_name!r} has no frozen risk classification",
                detail=full_name,
            )
        tools.append(name)
    return ProjectedMcpService(
        server_id=server_id,
        tools=tuple(sorted(set(tools))),
        transport=str(raw.get("transport") or ""),
        timeout_seconds=float(raw.get("timeout_seconds") or 0.0),
        max_output_bytes=int(raw.get("max_output_bytes") or 0),
        description=str(raw.get("description") or ""),
    )


def _digest_projection(payload: Mapping[str, Any]) -> str:
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise _projection_error(
            REASON_INVALID_POLICY,
            f"capability projection is not canonicalizable: {exc}",
        ) from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_capability_projection(
    policy: Mapping[str, Any],
    *,
    run_id: str = "",
    policy_hash_value: str | None = None,
) -> CapabilityProjection:
    """Project the frozen policy into stable capabilities + effect policy.

    Pure and deterministic: the same policy always yields the same
    inventory and projection hash. Raises :class:`CapabilityProjectionError`
    (fail closed) for malformed/incomplete input or a policy hash that
    does not match the expected run policy hash.
    """
    if not isinstance(policy, Mapping):
        raise _projection_error(
            REASON_INVALID_POLICY, "execution policy must be a mapping"
        )
    try:
        actual_hash = policy_hash(policy)
    except Exception as exc:
        raise _projection_error(
            REASON_INVALID_POLICY, f"execution policy is not canonical: {exc}"
        ) from exc
    if policy_hash_value is not None:
        expected = str(policy_hash_value or "").strip().lower()
        if expected != actual_hash:
            raise CapabilityProjectionError(
                REASON_POLICY_HASH_MISMATCH,
                f"policy hash {actual_hash} does not match expected {expected}",
                code=ExecutionErrorCode.POLICY_HASH_MISMATCH,
            )

    capability = policy.get("capability")
    if not isinstance(capability, Mapping):
        raise _projection_error(
            REASON_INVALID_CAPABILITY,
            "policy capability block is missing",
            detail="capability",
        )
    raw_tools = capability.get("allowed_tools")
    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes)):
        raise _projection_error(
            REASON_INVALID_CAPABILITY,
            "capability.allowed_tools must be a list",
            detail="capability.allowed_tools",
        )
    allowed_tools: list[str] = []
    for entry in raw_tools:
        if not isinstance(entry, str) or not entry.strip():
            raise _projection_error(
                REASON_INVALID_CAPABILITY,
                "capability.allowed_tools entries must be non-empty strings",
                detail="capability.allowed_tools",
            )
        allowed_tools.append(entry.strip())
    allowed_tools = sorted(set(allowed_tools))

    raw_servers = capability.get("allowed_mcp_servers")
    if raw_servers is not None:
        if not isinstance(raw_servers, Sequence) or isinstance(
            raw_servers, (str, bytes)
        ):
            raise _projection_error(
                REASON_INVALID_CAPABILITY,
                "capability.allowed_mcp_servers must be a list or null",
                detail="capability.allowed_mcp_servers",
            )
        if any(not isinstance(server, str) or not server.strip() for server in raw_servers):
            raise _projection_error(
                REASON_INVALID_CAPABILITY,
                "capability.allowed_mcp_servers entries must be non-empty strings",
                detail="capability.allowed_mcp_servers",
            )
    allowed_servers = sorted(set(raw_servers or ()))

    raw_services = capability.get("mcp_servers") or {}
    if not isinstance(raw_services, Mapping):
        raise _projection_error(
            REASON_INVALID_CAPABILITY,
            "capability.mcp_servers must be a mapping",
            detail="capability.mcp_servers",
        )

    effect_policy = policy.get("effect_policy")
    if not isinstance(effect_policy, Mapping):
        raise _projection_error(
            REASON_INVALID_CAPABILITY,
            "policy effect_policy block is missing",
            detail="effect_policy",
        )
    tool_risk = _normalize_tool_risk_map(effect_policy.get("tool_risk"))
    threshold = _normalize_risk(
        effect_policy.get("approval_risk_threshold"),
        label="effect_policy.approval_risk_threshold",
    )

    services: list[ProjectedMcpService] = []
    for server_id in allowed_servers:
        service = _normalize_mcp_service(
            server_id, raw_services.get(server_id), tool_risk
        )
        services.append(service)

    projected: dict[str, ProjectedTool] = {}

    def add_tool(
        *,
        capability_id: str,
        tool_name: str,
        source: str,
        risk: str,
        description: str = "",
        server_id: str = "",
        mcp_tool: str = "",
        schema_summary: Mapping[str, Any] | None = None,
    ) -> None:
        if capability_id in projected:
            return
        projected[capability_id] = ProjectedTool(
            capability_id=capability_id,
            tool_name=tool_name,
            source=source,
            risk=risk,
            requires_approval=_RISK_ORDER[risk] >= _RISK_ORDER[threshold],
            description=description,
            server_id=server_id,
            mcp_tool=mcp_tool,
            schema_summary=dict(schema_summary or {}),
        )

    mcp_tools_by_server: dict[str, set[str]] = {
        service.server_id: set(service.tools) for service in services
    }

    for tool_name in allowed_tools:
        parsed = split_mcp_kernel_tool_name(tool_name)
        if parsed is not None:
            server_id, mcp_tool = parsed
            if server_id not in mcp_tools_by_server:
                raise _projection_error(
                    REASON_MCP_TOOL_NOT_IN_SERVICE,
                    f"allowed MCP tool {tool_name!r} belongs to a server that "
                    "is not in the frozen MCP grant",
                    detail=tool_name,
                )
            if mcp_tool not in mcp_tools_by_server[server_id]:
                raise _projection_error(
                    REASON_MCP_TOOL_NOT_IN_SERVICE,
                    f"allowed MCP tool {tool_name!r} is outside the frozen "
                    f"server allowlist for {server_id!r}",
                    detail=tool_name,
                )
            risk = tool_risk.get(tool_name) or tool_risk.get(mcp_tool)
            if risk is None:
                raise _projection_error(
                    REASON_MISSING_RISK,
                    f"MCP tool {tool_name!r} has no frozen risk classification",
                    detail=tool_name,
                )
            add_tool(
                capability_id=mcp_capability_id(server_id, mcp_tool),
                tool_name=tool_name,
                source=SOURCE_MCP,
                risk=risk,
                server_id=server_id,
                mcp_tool=mcp_tool,
                schema_summary={"kind": "mcp", "deferred": True},
            )
            continue

        source = classify_tool_source(tool_name)
        risk = tool_risk.get(tool_name)
        if risk is None:
            raise _projection_error(
                REASON_MISSING_RISK,
                f"allowed tool {tool_name!r} has no frozen risk classification",
                detail=tool_name,
            )
        if source == SOURCE_SANDBOX:
            capability_id = sandbox_capability_id(tool_name)
        else:
            capability_id = builtin_capability_id(tool_name)
        add_tool(
            capability_id=capability_id,
            tool_name=tool_name,
            source=source,
            risk=risk,
            schema_summary={"kind": "product_tool"},
        )

    for server_id in allowed_servers:
        service = next(item for item in services if item.server_id == server_id)
        raw_service = raw_services.get(server_id)
        for mcp_tool in service.tools:
            full_name = mcp_kernel_tool_name(server_id, mcp_tool)
            risk = _resolve_mcp_tool_risk(
                server_id, mcp_tool, raw_service, tool_risk
            )
            if risk is None:
                raise _projection_error(
                    REASON_MISSING_RISK,
                    f"MCP tool {full_name!r} has no frozen risk classification",
                    detail=full_name,
                )
            add_tool(
                capability_id=mcp_capability_id(server_id, mcp_tool),
                tool_name=full_name,
                source=SOURCE_MCP,
                risk=risk,
                server_id=server_id,
                mcp_tool=mcp_tool,
                schema_summary={"kind": "mcp", "deferred": True},
            )

    for skill_name, inline_risk in _normalize_skills(capability.get("skills")):
        full_name = f"skill:{skill_name}"
        risk = (
            _normalize_risk(inline_risk, label=f"skill.{skill_name}")
            if inline_risk is not None
            else tool_risk.get(full_name) or tool_risk.get(skill_name)
        )
        if risk is None:
            raise _projection_error(
                REASON_MISSING_RISK,
                f"skill {skill_name!r} has no frozen risk classification",
                detail=full_name,
            )
        add_tool(
            capability_id=skill_capability_id(skill_name),
            tool_name=full_name,
            source=SOURCE_SKILL,
            risk=risk,
            schema_summary={"kind": "skill"},
        )

    raw_groups = capability.get("tool_groups") or ()
    if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
        raise _projection_error(
            REASON_INVALID_CAPABILITY,
            "capability.tool_groups must be a list",
            detail="capability.tool_groups",
        )
    tool_groups = tuple(
        sorted({str(group) for group in raw_groups if str(group).strip()})
    )

    tools = tuple(sorted(projected.values(), key=lambda tool: tool.capability_id))
    services = tuple(sorted(services, key=lambda service: service.server_id))
    profile_id = str(policy.get("profile_id") or "")
    digest_payload = {
        "version": PROJECTION_VERSION,
        "policy_hash": actual_hash,
        "profile_id": profile_id,
        "approval_threshold": threshold,
        "tool_groups": list(tool_groups),
        "tool_risk": dict(sorted(tool_risk.items())),
        "capabilities": [tool.to_payload() for tool in tools],
        "mcp_services": [service.to_payload() for service in services],
    }
    return CapabilityProjection(
        run_id=run_id,
        policy_hash=actual_hash,
        projection_hash=_digest_projection(digest_payload),
        profile_id=profile_id,
        tools=tools,
        mcp_services=services,
        approval_threshold=threshold,
        tool_risk=tool_risk,
        tool_groups=tool_groups,
    )


# ---------------------------------------------------------------------------
# MCP registry freeze helper (used by the admission policy builder)
# ---------------------------------------------------------------------------


def frozen_mcp_server_summary(
    server_ids: Sequence[str] | None,
) -> dict[str, dict[str, Any]]:
    """Snapshot the server-side MCP grant for admission.

    Only identity/transport/bounds/tool allowlist/risk are frozen — never
    credential values. The result is JSON-safe and becomes part of the
    execution policy, so execution never has to consult the live registry
    for authorization.
    """
    if not server_ids:
        return {}
    from agent_v2.mcp_registry import resolve_mcp_server
    from tools.risk_classifier import RiskLevel as ClassifierRisk
    from tools.risk_classifier import classify_tool

    summary: dict[str, dict[str, Any]] = {}
    for server_id in sorted(set(server_ids)):
        config = resolve_mcp_server(server_id)
        summary[server_id] = {
            "transport": config.transport,
            "timeout_seconds": config.timeout_seconds,
            "max_output_bytes": config.max_output_bytes,
            "description": config.description,
            "tools": [
                {
                    "name": tool,
                    "risk": classify_tool(
                        tool, default=ClassifierRisk.HIGH
                    ).value,
                }
                for tool in sorted(config.tool_allowlist)
            ],
        }
    return summary


def legacy_capability_projection(
    profile: Any, *, run_id: str = ""
) -> CapabilityProjection:
    """Build a projection for a caller with no frozen snapshot (test seam).

    Production execution always uses :func:`build_capability_projection`
    over the admitted policy. Direct kernel/gate users without a run (and
    the v2-disabled/toolless test paths) still get one authorization
    boundary built from the profile they were handed.
    """
    from agent_v2.mcp_registry import resolve_mcp_server

    allowed_servers = getattr(profile, "allowed_mcp_servers", None)
    mcp_summary = frozen_mcp_server_summary(allowed_servers)
    risk = tool_risk_policy()
    for server_id, service in mcp_summary.items():
        for entry in service["tools"]:
            risk[mcp_kernel_tool_name(server_id, entry["name"])] = entry["risk"]
    policy = {
        "profile_id": str(
            getattr(getattr(profile, "profile_id", None), "value", "")
            or getattr(profile, "profile_id", "")
        ),
        "capability": {
            "allowed_tools": list(getattr(profile, "allowed_tools", ())),
            "allowed_mcp_servers": (
                None if allowed_servers is None else list(allowed_servers)
            ),
            "mcp_servers": mcp_summary,
            "skills": [],
            "tool_groups": [],
        },
        "effect_policy": {
            "approval_risk_threshold": approval_risk_threshold(),
            "tool_risk": risk,
        },
    }
    # Fail-closed semantics identical to the frozen path: an unclassified
    # tool cannot be silently exposed. Callers that need the legacy
    # "unknown adapter" error still validate their own surface.
    try:
        return build_capability_projection(policy, run_id=run_id)
    except CapabilityProjectionError:
        raise


__all__ = [
    "CapabilityProjection",
    "CapabilityProjectionError",
    "McpDiscoveryResult",
    "PROJECTION_VERSION",
    "ProjectedMcpService",
    "ProjectedTool",
    "REASON_INVALID_CAPABILITY",
    "REASON_INVALID_POLICY",
    "REASON_MISSING_MCP_SERVICE",
    "REASON_MISSING_RISK",
    "REASON_POLICY_HASH_MISMATCH",
    "SOURCE_BUILTIN",
    "SOURCE_MCP",
    "SOURCE_SANDBOX",
    "SOURCE_SKILL",
    "ToolAuthorization",
    "ToolNotAuthorizedError",
    "build_capability_projection",
    "builtin_capability_id",
    "classify_tool_source",
    "frozen_mcp_server_summary",
    "legacy_capability_projection",
    "mcp_capability_id",
    "mcp_kernel_tool_name",
    "sandbox_capability_id",
    "skill_capability_id",
    "split_mcp_kernel_tool_name",
]
