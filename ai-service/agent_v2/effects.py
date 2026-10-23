"""Ticket 12 — Effect ledger: crash-safe side-effect tracking.

The effects module owns three concerns:

1. **Effect key derivation** — every side-effecting tool call gets a
   deterministic key computed from ``(run_id, tool_name, canonical_args)``.
   The key is a uuid5 in a SEPARATE namespace from approvals so an
   effect key never collides with an approval id, even though they share
   the same inputs. This lets the effect ledger record execution state
   independently of the approval (permission) state (spec AC1: "每个有
   副作用的 tool call 在执行前事务写入唯一 effect key").

2. **Effect status taxonomy** — ``authorized`` → ``executing`` →
   ``committed`` (success, result persisted) | ``outcome_unknown``
   (crash during/after exec, cannot safely retry) | ``failed``
   (definitive failure). The status machine enforces crash semantics:
   a non-idempotent action that crashes in ``executing`` MUST NOT be
   auto-retried (spec AC3).

3. **External idempotency key** — tools that support external
   idempotency (e.g. an API with an ``Idempotency-Key`` header) reuse
   the same key on retry so the external system returns the cached
   result (spec AC2). The ``external_idempotency_key`` is stored on the
   effect row; for local tools (e.g. ``generate_pdf``) it is the effect
   key itself, but the tool is still non-idempotent because the
   external "system" is a local filesystem that does not deduplicate.

The module is pure — no I/O, no DB. Persistence lives in
:mod:`agent_v2.persistence` (the ``effects`` table) and is accessed
through :class:`agent_v2.runs.RunRepository`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from agent_v2.execution_contract import BoundaryError, ExecutionErrorCode


class EffectStatus(str, Enum):
    """Lifecycle states for an effect ledger row (spec AC1, AC3).

    ``AUTHORIZED`` — the tool call has been authorized (approval granted
    or not needed) and is about to execute. The row is written BEFORE
    any side effect so a crash here is safe to retry.

    ``EXECUTING`` — the tool is in flight. A crash in this state means
    we cannot determine whether the side effect completed; for
    non-idempotent tools the row transitions to ``OUTCOME_UNKNOWN`` on
    recovery and must NOT be auto-retried (spec AC3).

    ``COMMITTED`` — the side effect completed and the result is
    persisted in ``result_json``. On retry, the stored result is
    returned as the unique result without re-executing (spec AC2).

    ``OUTCOME_UNKNOWN`` — a crash left the execution result
    indeterminate. The run terminates with ``effect.outcome_unknown``
    and requires human review (spec AC3, AC5). The row is terminal —
    no automatic retry is permitted.

    ``FAILED`` — the tool returned a definitive failure (e.g. sandbox
    timeout). The run terminates with ``error``. The row is terminal;
    a future re-approval would create a NEW effect row with a new
    effect key (different ``authorized_at``).
    """

    AUTHORIZED = "authorized"
    EXECUTING = "executing"
    COMMITTED = "committed"
    OUTCOME_UNKNOWN = "outcome_unknown"
    FAILED = "failed"

    # BCDEF ticket 04 canonical names. ``AUTHORIZED``/``EXECUTING`` are
    # the legacy stored values kept readable; new writers may store the
    # canonical pair. :func:`canonical_effect_status` maps both onto one
    # taxonomy so the state machine has a single vocabulary.
    PENDING = "pending"
    CLAIMED = "claimed"


# Legacy stored status -> canonical ticket-04 status. ``pending`` is the
# durable "authorized, not yet claimed" row and ``claimed`` is the row a
# worker owns (the legacy ``executing``). Terminal values are unchanged.
_EFFECT_STATUS_ALIASES: dict[str, str] = {
    "authorized": EffectStatus.PENDING.value,
    "pending": EffectStatus.PENDING.value,
    "executing": EffectStatus.CLAIMED.value,
    "claimed": EffectStatus.CLAIMED.value,
    "committed": EffectStatus.COMMITTED.value,
    "failed": EffectStatus.FAILED.value,
    "outcome_unknown": EffectStatus.OUTCOME_UNKNOWN.value,
}

# Statuses a worker may claim (CAS source state). Both the legacy and
# the canonical spelling are accepted so old rows stay claimable.
CLAIMABLE_EFFECT_STATUSES: frozenset[str] = frozenset({"authorized", "pending"})

# Statuses that mean "a worker owns this effect". A second claimer must
# never execute; it either returns the committed result or a stable
# conflict/in-flight answer.
CLAIMED_EFFECT_STATUSES: frozenset[str] = frozenset({"executing", "claimed"})


def canonical_effect_status(status: str) -> str:
    """Map a stored effect status onto the canonical taxonomy.

    Unknown values are returned unchanged (fail visible, not silently
    coerced) so a corrupt row cannot be mistaken for a claimable one.
    """
    return _EFFECT_STATUS_ALIASES.get(status, status)


def is_claimable_effect_status(status: str) -> bool:
    return status in CLAIMABLE_EFFECT_STATUSES


def is_claimed_effect_status(status: str) -> bool:
    return status in CLAIMED_EFFECT_STATUSES


# ---- BCDEF ticket 05: unknown-outcome repair state -------------------------
#
# An ``outcome_unknown`` effect is NOT retried automatically. It carries a
# separate repair state so snapshot/replay can explain who is reconciling
# it and what happens next, without changing the effect status taxonomy:
#
#   none → repair_pending → repair_claimed → repaired
#                        ↘              ↘ → repair_pending (failed attempt)
#                         → abandoned (fail closed, stays unknown)
#
# Repair NEVER re-runs the external effect and NEVER calls the model. A
# successful confirm commits the operator-supplied result as the single
# replayable final result; an abandon keeps the effect unknown and the run
# terminal (error).


class EffectRepairStatus(str, Enum):
    NONE = "none"
    PENDING = "repair_pending"
    CLAIMED = "repair_claimed"
    REPAIRED = "repaired"
    ABANDONED = "abandoned"


class EffectRepairAction(str, Enum):
    CONFIRM = "confirm"
    ABANDON = "abandon"


# Repair states that still allow a confirm/abandon decision.
OPEN_REPAIR_STATUSES: frozenset[str] = frozenset(
    {"none", "repair_pending", "repair_claimed"}
)


def repair_next_action(
    effect_status: str,
    repair_status: str,
    *,
    result_known: bool,
) -> str:
    """The operator/janitor next action for an effect row.

    ``none`` — nothing to repair (committed/failed/abandoned);
    ``wait_janitor`` — a worker holds the repair claim;
    ``confirm_known_result`` — the external result is known (publication
    failed) so confirmation can reuse it; ``manual_confirm`` — the
    result is unknown and an operator must supply it (or abandon).
    """
    if effect_status in ("committed", "failed"):
        return "none"
    if repair_status == EffectRepairStatus.ABANDONED.value:
        return "none"
    if repair_status == EffectRepairStatus.CLAIMED.value:
        return "wait_janitor"
    if effect_status == EffectStatus.OUTCOME_UNKNOWN.value:
        return "confirm_known_result" if result_known else "manual_confirm"
    return "none"


def repair_age_seconds(timestamp: str | None, *, now: datetime | None = None) -> float:
    """Seconds since the repair state last changed (0 when unknown).

    Parsing failures surface as 0.0 instead of raising: repair age is
    observability metadata and must never break a snapshot read.
    """
    if not timestamp:
        return 0.0
    try:
        parsed = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return max(0.0, (current - parsed).total_seconds())



# Default claim TTL. The claim token is the fencing credential for the
# commit; the expiry is audit/reconciliation metadata (a crashed claim is
# NEVER re-claimed automatically — recovery transitions to
# ``outcome_unknown`` instead).
DEFAULT_EFFECT_CLAIM_TTL_SECONDS: float = 300.0


@dataclass(frozen=True, slots=True)
class EffectIdentity:
    """Stable identity of one logical side effect.

    ``effect_id`` is derived from ``(run_id, tool_name, canonical args)``
    so a retry of the same logical tool call always maps to the same
    ledger row. ``idempotency_key`` is the stable external key derived
    from the same identity — it is never regenerated on retry.
    """

    effect_id: str
    idempotency_key: str
    run_id: str
    tool_name: str

    def audit_summary(self) -> dict[str, str]:
        """Auditable, content-free summary for events/logs."""
        return {
            "effect_id": self.effect_id,
            "idempotency_key": self.idempotency_key,
            "run_id": self.run_id,
            "tool": self.tool_name,
        }


@dataclass(frozen=True, slots=True)
class EffectClaim:
    """A fenced claim on one effect row.

    Only the holder of ``claim_token`` may commit, fail, or mark the
    effect ``outcome_unknown``. ``lease_id``/``fencing_token`` bind a
    sandbox effect claim to the sandbox active lease that authorized it.
    """

    effect_id: str
    claim_token: str
    owner: str = ""
    lease_id: str = ""
    fencing_token: int = 0

    def __post_init__(self) -> None:
        if not self.effect_id or not self.effect_id.strip():
            raise ValueError("effect claim requires a non-empty effect_id")
        if not self.claim_token or not self.claim_token.strip():
            raise ValueError("effect claim requires a non-empty claim_token")


class EffectNotClaimedError(BoundaryError):
    """Raised when execution is attempted without a winning claim."""

    def __init__(self, message: str = "effect claim required before execution") -> None:
        super().__init__(ExecutionErrorCode.EFFECT_NOT_CLAIMED, message)


def execute_claimed(
    claim: EffectClaim | None,
    execute: Callable[[], Any],
) -> Any:
    """Run ``execute`` only when ``claim`` is a valid fenced claim.

    This is the structural guard for "claim happens BEFORE any external
    side effect": callers cannot reach the executor without the claim
    object returned by the ledger CAS.
    """
    if claim is None:
        raise EffectNotClaimedError()
    return execute()


# Namespace UUID for effect key derivation. SEPARATE from the approvals
# namespace so an effect key never collides with an approval id, even
# though they share the same ``(run_id, tool_name, canonical_args)``
# inputs. This keeps the permission (approval) and execution (effect)
# ledgers decoupled — a tool may be approved but fail to execute, or
# (in future) execute without approval if it is low-risk.
_EFFECT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "uniharness.chat.v2.effects")


def compute_effect_key(
    *,
    run_id: str,
    tool_name: str,
    args: dict[str, Any] | None,
) -> str:
    """Compute the deterministic effect key (spec AC1).

    The key is a uuid5 over ``f"{run_id}|{tool_name}|{canonical_args}"``
    in the effects namespace. Same inputs → same key (idempotent);
    different args → different key (parameter tampering is a new effect,
    not a replay). The canonical args serialization is shared with
    :mod:`agent_v2.approvals` so the effect key and approval id are
    derived from the same canonical form.
    """
    # Local import avoids a circular dependency at module load time —
    # approvals.py is in the same package and imports no effect code.
    from agent_v2.approvals import canonical_args

    canonical = canonical_args(args)
    key = f"{run_id}|{tool_name}|{canonical}"
    return str(uuid.uuid5(_EFFECT_NAMESPACE, key))


def compute_external_idempotency_key(
    *,
    effect_key: str,
    tool_name: str,
) -> str:
    """Compute the external idempotency key for a tool (spec AC2).

    Tools that support external idempotency (e.g. an HTTP API with an
    ``Idempotency-Key`` header) send this key to the external system so
    a retry returns the cached result instead of re-executing. The key
    is derived from the effect key + tool name so it is stable across
    retries within the same logical tool call but distinct across
    different tool calls.

    For local tools (e.g. ``generate_pdf``) this key is stored on the
    effect row for auditability, but the local filesystem does not
    deduplicate — so a crash during ``EXECUTING`` still transitions to
    ``OUTCOME_UNKNOWN`` (spec AC3).
    """
    return str(uuid.uuid5(_EFFECT_NAMESPACE, f"external|{tool_name}|{effect_key}"))


def compute_effect_identity(
    *,
    run_id: str,
    tool_name: str,
    args: dict[str, Any] | None,
) -> EffectIdentity:
    """Return the stable identity bundle for one logical side effect.

    ``effect_id`` is the deterministic ledger key (legacy ``effect_key``)
    and ``idempotency_key`` is the external key. Both derive from the
    same ``(run_id, tool_name, canonical args)`` input, so retries map to
    the SAME effect and never mint a fresh key.
    """
    effect_id = compute_effect_key(run_id=run_id, tool_name=tool_name, args=args)
    return EffectIdentity(
        effect_id=effect_id,
        idempotency_key=compute_external_idempotency_key(
            effect_key=effect_id, tool_name=tool_name
        ),
        run_id=run_id,
        tool_name=tool_name,
    )



def serialize_result(result: dict[str, Any] | None) -> str:
    """Serialize the tool result for durable storage on the effect row.

    The result is stored as JSON with sorted keys so the stored form is
    deterministic. On retry, the stored result is returned as-is (spec
    AC2: "把已有成功结果作为唯一结果返回").
    """
    if not result:
        return "{}"
    return json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def deserialize_result(result_json: str | None) -> dict[str, Any]:
    """Deserialize the stored tool result.

    Returns an empty dict if ``result_json`` is None or empty (e.g. the
    effect crashed before the result was written). Invalid JSON returns
    an empty dict rather than raising — the effect ledger is the
    authority, and a corrupt result row is treated as
    ``OUTCOME_UNKNOWN`` by the caller.
    """
    if not result_json:
        return {}
    try:
        parsed = json.loads(result_json)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, TypeError):
        pass
    return {}
