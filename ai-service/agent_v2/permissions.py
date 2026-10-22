"""Ticket 11 — PermissionGate: per-tool risk classification + decision.

The PermissionGate is the single decision point that decides whether a
tool call may proceed, must be denied outright, or requires explicit
user approval before execution. It is consulted by the Chat v2 stream
before any tool side effect happens (spec AC1, AC3).

Contract:

* Inputs: a resolved :class:`CapabilityProfile`, the tool name, and the
  tool arguments dict.
* Output: a :class:`PermissionDecision` of ``allow``, ``deny``, or
  ``needs_approval``.
* Invariants:

  - Unknown tools (not in the profile's allowlist OR not registered in
    the risk table) default to ``deny``. Spec AC1: "未知工具默认拒绝".
  - The gate never executes the tool — it only returns a decision.
  - The decision is deterministic for the same inputs (no randomness,
    no time-based logic) so tests and replay are stable.

Risk classification is server-owned and product-defined. Clients cannot
influence it. The table intentionally lives here (not in profiles) so
adding a new tool risk does not require changing the frozen
:class:`CapabilityProfile` dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from agent_v2.profiles import CapabilityProfile, is_tool_allowed


class RiskLevel(str, Enum):
    """Server-owned risk taxonomy for tool calls.

    ``LOW`` — read-only or knowledge retrieval; no side effects.
    ``MEDIUM`` — has side effects but bounded (e.g. web fetch with SSRF
    guards, file write inside sandbox).
    ``HIGH`` — destructive, externally visible, or hard-to-reverse
    (e.g. PDF generation that produces an artifact, web scraping that
    fetches arbitrary URLs).
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PermissionDecision(str, Enum):
    """The PermissionGate's verdict for a tool call.

    ``ALLOW`` — the tool may execute without user approval.
    ``DENY`` — the tool must not execute; the run terminates with an
    error event carrying the deny reason.
    ``NEEDS_APPROVAL`` — the tool may execute only after the user
    approves via the decision endpoint; the run pauses until then.
    """

    ALLOW = "allow"
    DENY = "deny"
    NEEDS_APPROVAL = "needs_approval"


# Server-owned risk table. Tool names match the names declared in
# :mod:`agent_v2.profiles` (``allowed_tools`` tuples). A tool not listed
# here is treated as ``HIGH`` risk conservatively — but only after the
# profile allowlist check passes. Tools outside the profile allowlist
# are denied outright without consulting the risk table.
#
# The table is intentionally exhaustive for the v2-shipped profiles so
# adding a new tool forces an explicit risk decision here (no silent
# default for registered tools).
_TOOL_RISK: dict[str, RiskLevel] = {
    # ---- chat profile ----
    "hiking_knowledge_search": RiskLevel.LOW,
    "gear_checklist": RiskLevel.LOW,
    "risk_assessment": RiskLevel.LOW,
    "memory_save": RiskLevel.LOW,
    "memory_search": RiskLevel.LOW,
    # ---- route_plan profile ----
    "route_research": RiskLevel.MEDIUM,
    "web_search": RiskLevel.MEDIUM,
    "web_scraping": RiskLevel.HIGH,
    # ---- artifact profile ----
    "sandbox_read": RiskLevel.LOW,
    "sandbox_glob": RiskLevel.LOW,
    "sandbox_grep": RiskLevel.LOW,
    "sandbox_write": RiskLevel.MEDIUM,
    "sandbox_edit": RiskLevel.MEDIUM,
    "generate_pdf": RiskLevel.HIGH,
    # ---- cowork profile (Ticket 13) ----
    # Bash is the highest-risk tool in the v2 surface: arbitrary shell
    # inside the sandbox can read secrets, exhaust resources, or
    # exfiltrate data. Every call MUST go through approval + effect
    # ledger + output cap + audit, regardless of args.
    "sandbox_bash": RiskLevel.HIGH,
}


# Spec AC1: tools at or above this risk level require explicit user
# approval before execution. The threshold is intentionally ``HIGH`` for
# the stub stage so the approval flow is exercised by ``generate_pdf``
# and ``web_scraping`` only; ``MEDIUM`` tools (e.g. ``sandbox_write``)
# proceed without approval to keep the artifact flow usable end-to-end.
# Operators may lower the threshold in a later ticket once the approval
# UX is validated.
_APPROVAL_RISK_THRESHOLD = RiskLevel.HIGH

_RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
}


def approval_risk_threshold() -> str:
    """The server-owned risk level at which approval is required.

    Exposed for the admission-time execution policy snapshot (BCDEF
    ticket 08) so the frozen effect/approval policy stays honest when the
    threshold is tuned.
    """
    return _APPROVAL_RISK_THRESHOLD.value


def tool_risk_policy() -> dict[str, str]:
    """Stable ``{tool_name: risk_level}`` view of the server risk table."""
    return {tool: risk.value for tool, risk in _TOOL_RISK.items()}


@dataclass(frozen=True, slots=True)
class PermissionVerdict:
    """The gate's verdict for a single tool call.

    ``decision`` is the action the stream should take. ``risk`` is the
    classified risk level (``None`` when the tool is unknown and the
    decision is ``DENY``). ``reason`` is a stable machine-readable code
    the stream attaches to the ``approval.required`` or ``error`` event
    payload so the frontend and tests can assert on it.
    """

    decision: PermissionDecision
    risk: RiskLevel | None
    reason: str


class PermissionGate:
    """Decide whether a tool call may proceed.

    The gate is stateless — each call is evaluated independently. State
    (e.g. "user already approved this tool in this run") lives in the
    approvals table (:mod:`agent_v2.approvals`), not here.
    """

    def evaluate(
        self,
        profile: CapabilityProfile,
        tool_name: str,
        args: dict[str, Any] | None = None,
    ) -> PermissionVerdict:
        """Return the verdict for ``tool_name`` under ``profile``.

        Evaluation order (spec AC1):

        1. Profile allowlist — tools not in ``profile.allowed_tools``
           are denied as ``unknown_tool``. This is the first check so
           a misconfigured profile cannot expose tools the product did
           not intend to grant.
        2. Risk table — registered tools look up their risk level. A
           tool in the allowlist but missing from the risk table is
           denied as ``unclassified_tool`` (fail-closed: the operator
           must add an explicit risk entry before the tool can run).
        3. Threshold — tools at or above :data:`_APPROVAL_RISK_THRESHOLD`
           return ``needs_approval``; lower-risk tools return ``allow``.

        ``args`` is accepted for future extension (e.g. parameter-based
        risk escalation like "sandbox_write to a path outside the
        workspace") but is not used in the stub stage. Passing ``None``
        or ``{}`` is fine.
        """
        # 1. Profile allowlist.
        if not is_tool_allowed(profile, tool_name):
            return PermissionVerdict(
                decision=PermissionDecision.DENY,
                risk=None,
                reason="unknown_tool",
            )

        # 2. Risk table lookup.
        risk = _TOOL_RISK.get(tool_name)
        if risk is None:
            # Fail-closed: a tool in the allowlist but missing from the
            # risk table is treated as unclassified. The operator must
            # add an explicit entry before the tool can run.
            return PermissionVerdict(
                decision=PermissionDecision.DENY,
                risk=None,
                reason="unclassified_tool",
            )

        # 3. Threshold check.
        if _RISK_ORDER[risk] >= _RISK_ORDER[_APPROVAL_RISK_THRESHOLD]:
            return PermissionVerdict(
                decision=PermissionDecision.NEEDS_APPROVAL,
                risk=risk,
                reason="high_risk_requires_approval",
            )

        return PermissionVerdict(
            decision=PermissionDecision.ALLOW,
            risk=risk,
            reason="low_risk_allowed",
        )


__all__ = [
    "PermissionDecision",
    "PermissionGate",
    "PermissionVerdict",
    "RiskLevel",
]
