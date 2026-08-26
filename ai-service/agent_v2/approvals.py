"""Ticket 11 — Approval lifecycle: id computation + status + canonical args.

The approvals module owns three concerns:

1. **Canonical argument serialization** — the same tool call (same run,
   same tool, same arguments) must always produce the same approval id.
   We use deterministic JSON with sorted keys so parameter ordering does
   not smuggle in a "different" approval (spec AC2: "approval id 由 run、
   tool call 和 canonical arguments 确定").

2. **Approval id computation** — a uuid5 derived from
   ``(run_id, tool_name, canonical_args)``. The id is deterministic and
   has a UNIQUE constraint in the persistence layer, so a re-insert of
   the same logical approval raises IntegrityError rather than silently
   creating a duplicate row (spec AC5: "重复点击... 不会再次执行工具").

3. **Approval status taxonomy** — pending → approved/rejected/expired.
   ``superseded`` is reserved for future use (e.g. operator force-cancels
   a stuck approval); it is not produced by the stub stage.

The module is pure — no I/O, no DB. Persistence lives in
:mod:`agent_v2.persistence` (the ``approvals`` table) and is accessed
through :class:`agent_v2.runs.RunRepository`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


# Spec decision #29: approvals expire after this many seconds. The TTL
# is short on purpose — paused runs hold sandbox leases, and operators
# should not keep E2B sandboxes alive indefinitely waiting for a user
# decision. The frontend surfaces the expiry in the ``approval.required``
# event payload so the user knows the deadline.
APPROVAL_TTL_SECONDS = 300  # 5 minutes


class ApprovalStatus(str, Enum):
    """Lifecycle states for an approval row.

    ``PENDING`` — the approval is awaiting a user decision. The run is
    paused and the sandbox lease is held.
    ``APPROVED`` — the user approved; the run resumes and the tool
    executes exactly once.
    ``REJECTED`` — the user rejected; the run resumes briefly to emit an
    ``approval.resolved`` event, then terminates with ``done`` (the tool
    did not execute, so there is no side effect to undo).
    ``EXPIRED`` — the TTL elapsed before a decision arrived. The run
    terminates with ``error`` and the sandbox is cleaned up.
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


# Approval states that are terminal — no further decision can mutate them.
TERMINAL_APPROVAL_STATUSES: frozenset[ApprovalStatus] = frozenset(
    {ApprovalStatus.APPROVED, ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED}
)


class ApprovalDecisionKind(str, Enum):
    """The user's decision on an approval.

    ``APPROVE`` — the tool may execute; the run resumes.
    ``REJECT`` — the tool must not execute; the run terminates with
    ``done`` after emitting ``approval.resolved``.
    """

    APPROVE = "approve"
    REJECT = "reject"


def canonical_args(args: dict[str, Any] | None) -> str:
    """Serialize ``args`` deterministically.

    Returns a JSON string with sorted keys, no extra whitespace, and
    ``ensure_ascii=False`` so Chinese content stays readable. ``None``
    normalizes to ``"{}"`` (empty args). The output is the canonical
    form used both for the approval id computation and for the args
    tampering check on the decision endpoint (spec AC5: "参数篡改不会
    再次执行工具").

    Non-JSON-serializable values raise ``TypeError`` — the caller (the
    stream) is expected to validate args before reaching the gate. The
    stub-stage tools all use JSON-safe args.
    """
    if not args:
        return "{}"
    return json.dumps(args, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# Namespace UUID for approval id derivation. Fixed so the same
# ``(run_id, tool_name, canonical_args)`` always produces the same
# approval id across processes and restarts. Generated once with
# ``uuid.uuid5(uuid.NAMESPACE_URL, "uniharness.chat.v2.approvals")``.
_APPROVAL_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "uniharness.chat.v2.approvals")


def compute_approval_id(
    *,
    run_id: str,
    tool_name: str,
    args: dict[str, Any] | None,
) -> str:
    """Compute the deterministic approval id (spec AC2).

    The id is a uuid5 over ``f"{run_id}|{tool_name}|{canonical_args}"``.
    This means:

    * Same run + same tool + same args → same approval id (idempotent).
    * Same run + same tool + different args → different approval id
      (parameter tampering is detected as a new approval, not a replay).
    * Different run + same tool + same args → different approval id
      (approvals do not cross runs).

    The function never raises for valid inputs; ``canonical_args`` may
    raise ``TypeError`` for non-serializable args, but the stream validates
    args before reaching this point.
    """
    canonical = canonical_args(args)
    key = f"{run_id}|{tool_name}|{canonical}"
    return str(uuid.uuid5(_APPROVAL_NAMESPACE, key))


def compute_expiry(
    created_at_iso: str,
    ttl_seconds: int = APPROVAL_TTL_SECONDS,
) -> str:
    """Compute the expiry timestamp for an approval.

    Parses ``created_at_iso`` (UTC ISO 8601) and returns
    ``created_at + ttl_seconds`` as ISO 8601. The TTL is fixed at
    :data:`APPROVAL_TTL_SECONDS` so operators can audit how long a
    paused run can hold a sandbox lease.
    """
    created = datetime.fromisoformat(created_at_iso)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    expires = created + timedelta(seconds=ttl_seconds)
    return expires.isoformat()


def is_expired(
    expires_at_iso: str,
    now: datetime | None = None,
) -> bool:
    """True iff ``expires_at_iso`` is in the past.

    ``now`` defaults to ``datetime.now(timezone.utc)``. Naive datetimes
    are treated as UTC (matching the persistence layer's convention).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    expires = datetime.fromisoformat(expires_at_iso)
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return expires < now


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """A pending approval awaiting user decision.

    The dataclass mirrors the ``approvals`` table row. The
    :class:`RunRepository` returns this type from ``get_approval`` and
    ``list_pending_approvals_for_run`` so the stream and decision
    endpoint share one type.

    ``args_canonical`` is the JSON-serialized canonical form; the
    original args dict is NOT stored separately to avoid ambiguity
    (callers that need the dict parse it back). ``impact`` is a short
    user-facing description of what the tool will do, surfaced in the
    ``approval.required`` event payload (spec AC2: "事件展示工具、风险、
    影响和过期时间").
    """

    approval_id: str
    run_id: str
    conversation_id: str
    tool_name: str
    args_canonical: str
    risk_level: str
    impact: str
    status: str
    created_at: str
    expires_at: str
    resolved_at: str | None
    decision_request_id: str | None


__all__ = [
    "APPROVAL_TTL_SECONDS",
    "ApprovalDecisionKind",
    "ApprovalRequest",
    "ApprovalStatus",
    "TERMINAL_APPROVAL_STATUSES",
    "canonical_args",
    "compute_approval_id",
    "compute_expiry",
    "is_expired",
]
