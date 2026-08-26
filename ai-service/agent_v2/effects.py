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
from enum import Enum
from typing import Any


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
