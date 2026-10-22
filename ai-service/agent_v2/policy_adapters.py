"""BCDEF ticket 09 — versioned execution-policy snapshot adapters.

Every persisted policy snapshot declares a schema version. The current
version (ticket 08) is executed natively; older shapes are lifted by an
explicit, pure, versioned adapter chain before resume. A snapshot that
cannot be lifted safely — unknown version, missing critical field, hash
mismatch or unreadable content — NEVER falls back to the current default
profile/model/budget/capability. The resume path terminates the run with a
stable ``policy_incompatible`` (or ``policy_snapshot_missing``) error and a
``next_action`` of ``janitor``/``manual``; snapshot and replay expose the
reason.

Critical fields (all required after adaptation): ``schema_version``,
``profile_id``, ``model`` (main/fast/context window), ``budget``
(compaction threshold/output limit), ``capability`` (allowed tools + MCP
grant), ``lifecycle`` and ``effect_policy`` (approval risk + tool risk).

Adapters are registered in the ticket-01
:class:`agent_v2.execution_contract.LegacyStateAdapter` registry. The
contract registry rejects duplicate ``(from, to)`` pairs; the module-level
bootstrap here is idempotent, and a foreign adapter already registered for
``0 -> 1`` is left in place (the registry stays the single source of
truth).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from agent_v2.execution_contract import (
    EXECUTION_SCHEMA_VERSION,
    BoundaryError,
    ExecutionErrorCode,
    get_adapter,
    policy_hash,
    register_adapter,
    registered_adapter_pairs,
)
from agent_v2.execution_policy import (
    ResolvedPolicySnapshot,
    profile_from_policy,
)

# The version written by the current policy builder. Kept as its own name
# so adapter code never has to mean "current" by implication.
NATIVE_POLICY_SCHEMA_VERSION: int = EXECUTION_SCHEMA_VERSION
# Pre-ticket-08 rows: no schema_version/compatibility block inside the
# policy JSON, row version recorded as 0.
LEGACY_POLICY_SCHEMA_VERSIONS: frozenset[int] = frozenset({0})
# Versions this runtime can READ (native + adapted). Distinct from
# ``execution_contract.SUPPORTED_SCHEMA_VERSIONS``, which stays the set of
# versions the current writer NATIVELY produces.
READABLE_POLICY_SCHEMA_VERSIONS: frozenset[int] = frozenset(
    {*LEGACY_POLICY_SCHEMA_VERSIONS, NATIVE_POLICY_SCHEMA_VERSION}
)

# Stable machine-readable failure reasons. The wire code stays
# ``policy_incompatible``; ``reason`` is the additive detail.
REASON_SNAPSHOT_MISSING = "snapshot_missing"
REASON_UNSUPPORTED_SCHEMA_VERSION = "unsupported_schema_version"
REASON_MISSING_CRITICAL_FIELD = "missing_critical_field"
REASON_POLICY_HASH_MISMATCH = "policy_hash_mismatch"
REASON_ADAPTER_MISSING = "adapter_missing"
REASON_INVALID_SNAPSHOT = "invalid_snapshot"
REASON_ADAPTER_MUTATED_INPUT = "adapter_mutated_input"

# Follow-up owner for an incompatible run. Unknown versions and hash
# mismatches need a human (upgrade/migration or tamper investigation);
# structurally-missing legacy data is a janitor repair candidate.
NEXT_ACTION_MANUAL = "manual"
NEXT_ACTION_JANITOR = "janitor"

_MISSING = object()


class PolicyCompatibilityError(BoundaryError):
    """A snapshot cannot be adapted to the current execution schema.

    Always carries :data:`ExecutionErrorCode.POLICY_INCOMPATIBLE`; the
    additive ``reason``/``next_action``/``detail`` fields make the failure
    explainable from snapshot and replay without parsing messages.
    """

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        source_schema_version: int | None = None,
        target_schema_version: int = NATIVE_POLICY_SCHEMA_VERSION,
        next_action: str = NEXT_ACTION_MANUAL,
        detail: str = "",
    ) -> None:
        super().__init__(ExecutionErrorCode.POLICY_INCOMPATIBLE, message)
        self.reason = reason
        self.source_schema_version = source_schema_version
        self.target_schema_version = target_schema_version
        self.next_action = next_action
        self.detail = detail

    def to_payload(self) -> dict[str, Any]:
        """Stable additive error payload (existing envelope shape)."""
        payload: dict[str, Any] = {
            "code": ExecutionErrorCode.POLICY_INCOMPATIBLE.value,
            "reason": self.reason,
            "recoverable": False,
            "next_action": self.next_action,
            "target_schema_version": self.target_schema_version,
            "message": self.message,
        }
        if self.source_schema_version is not None:
            payload["source_schema_version"] = self.source_schema_version
        if self.detail:
            payload["detail"] = self.detail
        return payload


@dataclass(frozen=True, slots=True)
class PolicyAdapterStep:
    """One hop of an adapter chain, recorded for audit."""

    from_version: int
    to_version: int
    adapter_id: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "adapter_id": self.adapter_id,
        }


class LegacyPolicySnapshotV0Adapter:
    """Lift a pre-ticket-08 minimal snapshot (version 0) to version 1.

    Pure: deep-copies its input, never reads current configuration and
    never validates against the live profile. Critical-field validation
    happens after the whole chain, so a malformed legacy row fails with a
    precise ``missing_critical_field`` rather than a guessed default.
    """

    from_version = 0
    to_version = NATIVE_POLICY_SCHEMA_VERSION
    adapter_id = "policy_snapshot_v0_to_v1"

    def adapt(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise PolicyCompatibilityError(
                REASON_INVALID_SNAPSHOT,
                "legacy policy snapshot is not a mapping",
                source_schema_version=self.from_version,
            )
        lifted = copy.deepcopy(dict(payload))
        lifted["schema_version"] = self.to_version
        lifted.setdefault("policy_kind", "execution")
        lifted.setdefault("source_version", "agent_v2.execution_policy/legacy-v0")
        compatibility = dict(lifted.get("compatibility") or {})
        compatibility.update(
            {
                "schema_version": self.to_version,
                "policy_kind": lifted["policy_kind"],
                "source_version": lifted["source_version"],
                "adapted_from_schema_version": self.from_version,
                "adapter_id": self.adapter_id,
            }
        )
        lifted["compatibility"] = compatibility
        return lifted


def ensure_default_policy_adapters_registered() -> None:
    """Idempotently bootstrap the built-in adapter chain.

    Safe to call any number of times. If another adapter already owns the
    ``0 -> 1`` pair it is left untouched: the shared registry, not this
    module, is the source of truth (and a duplicate registration attempt
    still raises ``adapter_conflict``).
    """
    if get_adapter(min(LEGACY_POLICY_SCHEMA_VERSIONS), NATIVE_POLICY_SCHEMA_VERSION) is None:
        register_adapter(LegacyPolicySnapshotV0Adapter())


def _decode_policy(record: Any) -> dict[str, Any]:
    raw = getattr(record, "policy_json", "")
    try:
        policy = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise PolicyCompatibilityError(
            REASON_INVALID_SNAPSHOT,
            f"run {getattr(record, 'run_id', '?')} policy snapshot is not valid JSON",
            source_schema_version=_source_version(record),
            next_action=NEXT_ACTION_JANITOR,
            detail=str(exc),
        ) from exc
    if not isinstance(policy, dict):
        raise PolicyCompatibilityError(
            REASON_INVALID_SNAPSHOT,
            f"run {getattr(record, 'run_id', '?')} policy snapshot is not a mapping",
            source_schema_version=_source_version(record),
            next_action=NEXT_ACTION_JANITOR,
        )
    return policy


def _source_version(record: Any) -> int | None:
    value = getattr(record, "schema_version", None)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _assert_content_version(policy: Mapping[str, Any], source_version: int) -> None:
    content_version = policy.get("schema_version", _MISSING)
    if source_version == NATIVE_POLICY_SCHEMA_VERSION:
        if content_version != NATIVE_POLICY_SCHEMA_VERSION:
            raise PolicyCompatibilityError(
                REASON_MISSING_CRITICAL_FIELD,
                "policy content is missing its schema_version",
                source_schema_version=source_version,
                next_action=NEXT_ACTION_JANITOR,
                detail="schema_version",
            )
        return
    if content_version is not _MISSING and content_version != source_version:
        raise PolicyCompatibilityError(
            REASON_INVALID_SNAPSHOT,
            f"policy content schema {content_version!r} contradicts row "
            f"version {source_version!r}",
            source_schema_version=source_version,
            next_action=NEXT_ACTION_JANITOR,
            detail="schema_version",
        )


def _assert_stored_hash(policy: Mapping[str, Any], record: Any, source_version: int) -> str:
    stored = getattr(record, "policy_hash", "") or ""
    stored = str(stored).strip().lower()
    actual = policy_hash(policy)
    if stored and stored != actual:
        raise PolicyCompatibilityError(
            REASON_POLICY_HASH_MISMATCH,
            f"run {getattr(record, 'run_id', '?')} policy hash mismatch",
            source_schema_version=source_version,
            next_action=NEXT_ACTION_MANUAL,
            detail=f"stored={stored}",
        )
    return actual


def _resolve_chain(source_version: int) -> list[Any]:
    """Return registered adapter hops from ``source_version`` to native.

    Deterministic: at each hop the registered adapter with the smallest
    target version (not beyond native) wins, so a chain is stable and
    auditable. A missing hop fails closed as ``adapter_missing``.
    """
    chain: list[Any] = []
    current = source_version
    seen: set[int] = {current}
    while current != NATIVE_POLICY_SCHEMA_VERSION:
        options: list[tuple[int, Any]] = []
        for from_version, to_version in registered_adapter_pairs():
            if from_version != current or to_version > NATIVE_POLICY_SCHEMA_VERSION:
                continue
            adapter = get_adapter(from_version, to_version)
            if adapter is not None:
                options.append((to_version, adapter))
        options.sort(key=lambda item: item[0])
        if not options:
            raise PolicyCompatibilityError(
                REASON_ADAPTER_MISSING,
                f"no registered adapter lifts policy schema {current} to "
                f"{NATIVE_POLICY_SCHEMA_VERSION}",
                source_schema_version=source_version,
                next_action=NEXT_ACTION_MANUAL,
            )
        next_version, adapter = options[0]
        if next_version in seen:
            raise PolicyCompatibilityError(
                REASON_ADAPTER_MISSING,
                f"adapter chain loops at policy schema {current}",
                source_schema_version=source_version,
                next_action=NEXT_ACTION_MANUAL,
            )
        chain.append(adapter)
        seen.add(next_version)
        current = next_version
    return chain


def _append_error(errors: list[str], condition: bool, field: str) -> None:
    if condition:
        errors.append(field)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_critical_fields(
    policy: Mapping[str, Any], *, source_version: int
) -> None:
    """Require every execution-critical field; never fill defaults."""
    errors: list[str] = []
    _append_error(
        errors,
        policy.get("schema_version") != NATIVE_POLICY_SCHEMA_VERSION,
        "schema_version",
    )
    profile_id = policy.get("profile_id")
    _append_error(
        errors,
        not isinstance(profile_id, str) or not profile_id.strip(),
        "profile_id",
    )

    model = policy.get("model")
    if not isinstance(model, Mapping):
        errors.append("model")
    else:
        for key in ("main_model", "fast_model"):
            value = model.get(key)
            _append_error(
                errors,
                not isinstance(value, str) or not value.strip(),
                f"model.{key}",
            )
        _append_error(
            errors,
            not _is_int(model.get("context_window"))
            or int(model["context_window"]) <= 0,
            "model.context_window",
        )

    budget = policy.get("budget")
    if not isinstance(budget, Mapping):
        errors.append("budget")
    else:
        for key in ("compaction_threshold", "output_limit"):
            value = budget.get(key)
            _append_error(
                errors,
                not _is_int(value) or int(value) < 0,
                f"budget.{key}",
            )

    capability = policy.get("capability")
    if not isinstance(capability, Mapping):
        errors.append("capability")
    else:
        tools = capability.get("allowed_tools")
        _append_error(
            errors,
            not isinstance(tools, list)
            or any(not isinstance(tool, str) for tool in tools),
            "capability.allowed_tools",
        )
        if "allowed_mcp_servers" not in capability:
            errors.append("capability.allowed_mcp_servers")
        else:
            mcp = capability["allowed_mcp_servers"]
            _append_error(
                errors,
                mcp is not None
                and (
                    not isinstance(mcp, list)
                    or any(not isinstance(server, str) for server in mcp)
                ),
                "capability.allowed_mcp_servers",
            )

    _append_error(
        errors, not isinstance(policy.get("lifecycle"), Mapping), "lifecycle"
    )

    effect_policy = policy.get("effect_policy")
    if not isinstance(effect_policy, Mapping):
        errors.append("effect_policy")
    else:
        threshold = effect_policy.get("approval_risk_threshold")
        _append_error(
            errors,
            not isinstance(threshold, str) or not threshold.strip(),
            "effect_policy.approval_risk_threshold",
        )
        _append_error(
            errors,
            not isinstance(effect_policy.get("tool_risk"), Mapping),
            "effect_policy.tool_risk",
        )

    if errors:
        detail = ",".join(errors)
        raise PolicyCompatibilityError(
            REASON_MISSING_CRITICAL_FIELD,
            f"policy snapshot is missing critical fields: {detail}",
            source_schema_version=source_version,
            next_action=NEXT_ACTION_JANITOR,
            detail=detail,
        )


def adapt_policy_snapshot(
    record: Any, *, now: str | None = None
) -> ResolvedPolicySnapshot:
    """Adapt one persisted snapshot row to the native policy shape.

    ``record`` is the storage-local ``PolicySnapshotRecord`` (duck-typed
    here to avoid a persistence import cycle). The returned snapshot
    carries the adapted policy + hash + schema version and the full adapter
    provenance. Raises :class:`PolicyCompatibilityError` for unknown
    versions, hash mismatches, missing critical fields and adapter
    failures. Pure: the stored row and its JSON are never mutated.
    """
    source_version = _source_version(record)
    if source_version not in READABLE_POLICY_SCHEMA_VERSIONS:
        raise PolicyCompatibilityError(
            REASON_UNSUPPORTED_SCHEMA_VERSION,
            f"policy schema {source_version!r} is not supported "
            f"(readable: {sorted(READABLE_POLICY_SCHEMA_VERSIONS)})",
            source_schema_version=source_version,
            next_action=NEXT_ACTION_MANUAL,
        )

    policy = _decode_policy(record)
    _assert_content_version(policy, source_version)
    stored_hash = _assert_stored_hash(policy, record, source_version)

    payload: Mapping[str, Any] = policy
    steps: list[PolicyAdapterStep] = []
    for adapter in _resolve_chain(source_version):
        before = copy.deepcopy(dict(payload))
        try:
            adapted = adapter.adapt(payload)
        except PolicyCompatibilityError:
            raise
        except Exception as exc:  # pragma: no cover - defensive adapter host
            raise PolicyCompatibilityError(
                REASON_INVALID_SNAPSHOT,
                f"adapter {adapter.adapter_id} failed: {exc}",
                source_schema_version=source_version,
                next_action=NEXT_ACTION_JANITOR,
                detail=str(exc),
            ) from exc
        if payload != before:  # pragma: no cover - defensive adapter host
            raise PolicyCompatibilityError(
                REASON_ADAPTER_MUTATED_INPUT,
                f"adapter {adapter.adapter_id} mutated its input payload",
                source_schema_version=source_version,
                next_action=NEXT_ACTION_MANUAL,
            )
        if not isinstance(adapted, Mapping):
            raise PolicyCompatibilityError(
                REASON_INVALID_SNAPSHOT,
                f"adapter {adapter.adapter_id} returned a non-mapping",
                source_schema_version=source_version,
                next_action=NEXT_ACTION_JANITOR,
            )
        payload = copy.deepcopy(dict(adapted))
        steps.append(
            PolicyAdapterStep(
                from_version=adapter.from_version,
                to_version=adapter.to_version,
                adapter_id=getattr(adapter, "adapter_id", type(adapter).__name__),
            )
        )

    _validate_critical_fields(payload, source_version=source_version)
    try:
        profile_from_policy(payload)
    except BoundaryError as exc:
        raise PolicyCompatibilityError(
            REASON_MISSING_CRITICAL_FIELD,
            f"policy snapshot cannot be reconstructed: {exc.message}",
            source_schema_version=source_version,
            next_action=NEXT_ACTION_JANITOR,
            detail=exc.message,
        ) from exc

    adapted_hash = policy_hash(payload)
    adapter_id = steps[-1].adapter_id if steps else "native"
    try:
        compatibility = json.loads(getattr(record, "compatibility_json", "") or "{}")
    except (TypeError, ValueError):  # pragma: no cover - defensive read
        compatibility = {}
    if not isinstance(compatibility, dict):
        compatibility = {}

    return ResolvedPolicySnapshot(
        run_id=str(getattr(record, "run_id", "") or ""),
        policy=payload,
        policy_hash=adapted_hash,
        schema_version=NATIVE_POLICY_SCHEMA_VERSION,
        source_version=str(getattr(record, "source_version", "") or ""),
        created_at=str(getattr(record, "created_at", "") or ""),
        compatibility=compatibility,
        source_schema_version=source_version,
        adapter_id=adapter_id,
        adapted_at=now or datetime.now(timezone.utc).isoformat(),
        original_policy_hash=stored_hash,
        adapter_chain=tuple(step.to_payload() for step in steps),
    )


def describe_policy_record(record: Any) -> dict[str, Any]:
    """Non-raising snapshot/replay projection of one policy record.

    Returns ``{"compatible", "policy_provenance", "policy_failure"}``.
    On failure the provenance is ``None`` and ``policy_failure`` carries
    the stable code/reason/next_action. Never raises: snapshot reads must
    stay available even when the policy is unusable.
    """
    try:
        resolved = adapt_policy_snapshot(record)
    except PolicyCompatibilityError as exc:
        return {
            "compatible": False,
            "policy_provenance": None,
            "policy_failure": exc.to_payload(),
        }
    provenance = resolved.provenance_payload()
    provenance["compatible"] = True
    return {
        "compatible": True,
        "policy_provenance": provenance,
        "policy_failure": None,
    }


def missing_snapshot_failure(run_id: str) -> dict[str, Any]:
    """Stable failure projection for a run without any policy snapshot."""
    return {
        "code": ExecutionErrorCode.POLICY_SNAPSHOT_MISSING.value,
        "reason": REASON_SNAPSHOT_MISSING,
        "recoverable": False,
        "next_action": NEXT_ACTION_MANUAL,
        "target_schema_version": NATIVE_POLICY_SCHEMA_VERSION,
        "message": f"run {run_id} has no frozen execution policy snapshot",
    }


def policy_error_payload(exc: BoundaryError) -> dict[str, Any]:
    """Convert any policy-resolution failure to the additive error payload."""
    if isinstance(exc, PolicyCompatibilityError):
        return exc.to_payload()
    code = getattr(exc, "code", None)
    if code is ExecutionErrorCode.POLICY_SNAPSHOT_MISSING:
        return missing_snapshot_failure("")
    return {
        "code": getattr(code, "value", "policy_incompatible"),
        "reason": REASON_INVALID_SNAPSHOT,
        "recoverable": False,
        "next_action": NEXT_ACTION_JANITOR,
        "target_schema_version": NATIVE_POLICY_SCHEMA_VERSION,
        "detail": str(exc),
        "message": str(exc),
    }


ensure_default_policy_adapters_registered()


__all__ = [
    "LEGACY_POLICY_SCHEMA_VERSIONS",
    "LegacyPolicySnapshotV0Adapter",
    "NATIVE_POLICY_SCHEMA_VERSION",
    "NEXT_ACTION_JANITOR",
    "NEXT_ACTION_MANUAL",
    "PolicyAdapterStep",
    "PolicyCompatibilityError",
    "READABLE_POLICY_SCHEMA_VERSIONS",
    "REASON_ADAPTER_MISSING",
    "REASON_INVALID_SNAPSHOT",
    "REASON_MISSING_CRITICAL_FIELD",
    "REASON_POLICY_HASH_MISMATCH",
    "REASON_SNAPSHOT_MISSING",
    "REASON_UNSUPPORTED_SCHEMA_VERSION",
    "adapt_policy_snapshot",
    "describe_policy_record",
    "ensure_default_policy_adapters_registered",
    "missing_snapshot_failure",
    "policy_error_payload",
]
