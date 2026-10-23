"""BCDEF ticket 05 — unknown-outcome reconciliation and repair.

An external effect that may have run but whose ledger/event commit failed
lands in ``outcome_unknown``. That state is terminal for execution: the
model is never re-run, the external effect is never re-executed, and no
success result may be published until the outcome is reconciled.

This module owns the reconciliation decisions:

* ``confirm`` — an operator (or the janitor, from a known result) supplies
  the result. The effect row commits to the SAME effect/run identity and
  exactly one replayable ``tool_result`` plus one ``effect.repaired``
  progress event are published.
* ``abandon`` — fail closed. The effect stays ``outcome_unknown``; only
  the repair state becomes ``abandoned``. No success result is published.

Repair is idempotent: a retried decision returns the original outcome and
never publishes a second result. A repair whose result event cannot be
made durable reverts to ``outcome_unknown``/``repair_pending`` so it can
be retried. Nothing in this module calls the model or an executor.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_v2.effects import (
    OPEN_REPAIR_STATUSES,
    EffectRepairAction,
    EffectRepairStatus,
    EffectStatus,
    repair_next_action,
    serialize_result,
)

# Public shorthand used by the HTTP adapter / janitor / tests.
RepairAction = EffectRepairAction

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent_v2.persistence import EffectRecord
    from agent_v2.runs import RunRepository

logger = logging.getLogger("ai-service.agent_v2.effect_repair")

# Stable public codes returned by the repair seam (HTTP + janitor).
REPAIR_NOT_FOUND = "effect_not_found"
REPAIR_NOT_NEEDED = "repair_not_needed"
REPAIR_CONFLICT = "repair_conflict"
REPAIR_FAILED = "repair_failed"
REPAIR_INVALID_RESULT = "repair_result_invalid"


@dataclass(frozen=True, slots=True)
class RepairOutcome:
    """Result of one repair decision.

    ``status`` is one of ``repaired`` / ``abandoned`` / ``idempotent`` /
    ``conflict`` / ``not_repairable`` / ``not_found`` / ``invalid_result``
    / ``failed``. ``payload`` is the stable public body shared by the HTTP
    adapter and the janitor.
    """

    status: str
    effect: "EffectRecord | None" = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("repaired", "abandoned", "idempotent")


def effect_known_facts(effect: "EffectRecord") -> dict[str, Any]:
    """Facts that are durable regardless of the unknown outcome."""
    return {
        "tool": effect.tool_name,
        "args_canonical": effect.args_canonical,
        "idempotency_key": effect.resolved_idempotency_key,
        "lease_id": effect.lease_id,
        "fencing_token": effect.fencing_token,
        "external_result_known": bool(effect.result_json),
    }


def effect_unknown_part(effect: "EffectRecord") -> dict[str, Any]:
    """The part of the outcome that is still indeterminate."""
    result_known = bool(effect.result_json)
    return {
        "result_known": result_known,
        "publication_failed": (
            effect.status == EffectStatus.OUTCOME_UNKNOWN.value and result_known
        ),
        "reason": effect.failure_reason,
        "requires_human": effect.status == EffectStatus.OUTCOME_UNKNOWN.value,
    }


def _public_payload(
    effect: "EffectRecord",
    *,
    status: str,
    action: str,
    code: str = "",
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result_known = bool(effect.result_json)
    payload: dict[str, Any] = {
        "effect_id": effect.resolved_effect_id,
        "effect_key": effect.effect_key,
        "run_id": effect.run_id,
        "tool": effect.tool_name,
        "action": action,
        "repair_status": effect.repair_status,
        "repair_attempts": effect.repair_attempts,
        "repair_last_error": effect.repair_last_error or effect.failure_reason,
        "repair_next_action": repair_next_action(
            effect.status,
            effect.repair_status,
            result_known=result_known,
        ),
        "status": status,
    }
    if code:
        payload["code"] = code
    if result is not None:
        payload["result"] = result
    return payload


def _run_events(runs_repo: "RunRepository", run_id: str) -> list[dict[str, Any]]:
    return runs_repo.get_events_since(run_id=run_id, from_seq=0)


def _next_sequence(events: list[dict[str, Any]]) -> int:
    return max((int(evt.get("sequence") or 0) for evt in events), default=0) + 1


def _persist(
    runs_repo: "RunRepository",
    effect: "EffectRecord",
    *,
    event_type: str,
    payload: dict[str, Any],
) -> int:
    events = _run_events(runs_repo, effect.run_id)
    run = runs_repo.get_run(effect.run_id)
    request_id = run.request_id if run is not None else ""
    sequence = _next_sequence(events)
    runs_repo.persist_event(
        run_id=effect.run_id,
        sequence=sequence,
        event_type=event_type,
        payload=payload,
        request_id=request_id,
        conversation_id=effect.conversation_id,
    )
    return sequence


def _has_repaired_progress(events: list[dict[str, Any]], effect_id: str) -> bool:
    return any(
        evt.get("type") == "progress"
        and evt.get("payload", {}).get("stage") == "effect.repaired"
        and evt["payload"].get("effect_id") == effect_id
        for evt in events
    )


def _has_success_result(events: list[dict[str, Any]], effect_id: str) -> bool:
    return any(
        evt.get("type") == "tool_result"
        and evt.get("payload", {}).get("effect_id") == effect_id
        and evt["payload"].get("status") == "ok"
        for evt in events
    )


def _has_abandoned_progress(events: list[dict[str, Any]], effect_id: str) -> bool:
    return any(
        evt.get("type") == "progress"
        and evt.get("payload", {}).get("stage") == "effect.repair_abandoned"
        and evt["payload"].get("effect_id") == effect_id
        for evt in events
    )


def _confirmed_result_payload(
    effect: "EffectRecord", result: dict[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tool": effect.tool_name,
        "status": "ok",
        "effect_id": effect.resolved_effect_id,
        "repaired": True,
        "result": result,
    }
    # Surface well-known result keys so replay can render the tool result
    # without re-parsing the nested ``result`` object.
    for key in ("content", "path", "exit_code", "stdout", "stderr"):
        if key in result:
            payload[key] = result[key]
    return payload


def _publish_confirmed(
    runs_repo: "RunRepository",
    effect: "EffectRecord",
    *,
    result: dict[str, Any],
    actor: str,
    idempotency_key: str,
) -> None:
    """Publish exactly one repaired progress + one final result.

    The progress event is written FIRST so a failure between the two
    writes leaves no success result that replay could mistake for the
    final outcome; the retry fills in the missing result.
    """
    events = _run_events(runs_repo, effect.run_id)
    effect_id = effect.resolved_effect_id
    if not _has_repaired_progress(events, effect_id):
        _persist(
            runs_repo,
            effect,
            event_type="progress",
            payload={
                "stage": "effect.repaired",
                "effect_id": effect_id,
                "tool": effect.tool_name,
                "action": EffectRepairAction.CONFIRM.value,
                "actor": actor,
                "idempotency_key": idempotency_key,
                "repair_status": EffectRepairStatus.REPAIRED.value,
                "repair_attempts": effect.repair_attempts,
                "next_action": "none",
            },
        )
        events = _run_events(runs_repo, effect.run_id)
    if not _has_success_result(events, effect_id):
        _persist(
            runs_repo,
            effect,
            event_type="tool_result",
            payload=_confirmed_result_payload(effect, result),
        )


def _publish_abandoned(
    runs_repo: "RunRepository",
    effect: "EffectRecord",
    *,
    actor: str,
    idempotency_key: str,
    reason: str,
) -> None:
    events = _run_events(runs_repo, effect.run_id)
    effect_id = effect.resolved_effect_id
    if _has_abandoned_progress(events, effect_id):
        return
    _persist(
        runs_repo,
        effect,
        event_type="progress",
        payload={
            "stage": "effect.repair_abandoned",
            "effect_id": effect_id,
            "tool": effect.tool_name,
            "action": EffectRepairAction.ABANDON.value,
            "actor": actor,
            "idempotency_key": idempotency_key,
            "reason": reason,
            "repair_status": EffectRepairStatus.ABANDONED.value,
            "repair_attempts": effect.repair_attempts,
            "next_action": "none",
        },
    )


def _stored_result(effect: "EffectRecord") -> dict[str, Any] | None:
    if not effect.result_json:
        return None
    try:
        parsed = json.loads(effect.result_json)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def repair_effect(
    runs_repo: "RunRepository",
    effect_id: str,
    *,
    action: str,
    result: dict[str, Any] | None = None,
    idempotency_key: str,
    actor: str = "operator",
    reason: str = "",
) -> RepairOutcome:
    """Apply one idempotent repair decision to an unknown-outcome effect.

    The original ``effect_key``/``run_id``/approval identity is reused;
    nothing is re-created and no external effect or model turn runs.
    """
    effect = runs_repo.get_effect_by_effect_id(effect_id)
    if effect is None:
        return RepairOutcome(
            status="not_found",
            payload={"code": REPAIR_NOT_FOUND, "effect_id": effect_id},
        )

    if action not in (
        EffectRepairAction.CONFIRM.value,
        EffectRepairAction.ABANDON.value,
    ):
        return RepairOutcome(
            status="invalid_result",
            effect=effect,
            payload={"code": "repair_action_invalid", "effect_id": effect_id},
        )

    # Already repaired — repeat confirm is idempotent; abandon conflicts.
    if effect.status == EffectStatus.COMMITTED.value:
        if action == EffectRepairAction.ABANDON.value:
            return RepairOutcome(
                status="conflict",
                effect=effect,
                payload=_public_payload(
                    effect,
                    status="conflict",
                    action=action,
                    code=REPAIR_CONFLICT,
                ),
            )
        stored = _stored_result(effect)
        if stored is not None:
            try:
                _publish_confirmed(
                    runs_repo,
                    effect,
                    result=stored,
                    actor=actor,
                    idempotency_key=idempotency_key,
                )
            except Exception as exc:
                logger.exception(
                    "effect_repair: repaired-result publication retry failed "
                    "for %s",
                    effect_id,
                )
                return RepairOutcome(
                    status="failed",
                    effect=effect,
                    payload=_public_payload(
                        effect,
                        status="failed",
                        action=action,
                        code=REPAIR_FAILED,
                    ),
                )
        return RepairOutcome(
            status="idempotent",
            effect=effect,
            payload=_public_payload(
                effect,
                status="idempotent",
                action=action,
                result=stored or {},
            ),
        )

    # Already abandoned — repeat abandon is idempotent; confirm conflicts.
    if effect.repair_status == EffectRepairStatus.ABANDONED.value:
        if action == EffectRepairAction.CONFIRM.value:
            return RepairOutcome(
                status="conflict",
                effect=effect,
                payload=_public_payload(
                    effect,
                    status="conflict",
                    action=action,
                    code=REPAIR_CONFLICT,
                ),
            )
        try:
            _publish_abandoned(
                runs_repo,
                effect,
                actor=actor,
                idempotency_key=idempotency_key,
                reason=effect.repair_note or reason,
            )
        except Exception:
            logger.exception(
                "effect_repair: abandoned-event publication retry failed for %s",
                effect_id,
            )
        return RepairOutcome(
            status="idempotent",
            effect=effect,
            payload=_public_payload(
                effect, status="idempotent", action=action
            ),
        )

    if effect.status != EffectStatus.OUTCOME_UNKNOWN.value:
        return RepairOutcome(
            status="not_repairable",
            effect=effect,
            payload=_public_payload(
                effect,
                status="not_repairable",
                action=action,
                code=REPAIR_NOT_NEEDED,
            ),
        )

    if effect.repair_status not in OPEN_REPAIR_STATUSES:
        return RepairOutcome(
            status="not_repairable",
            effect=effect,
            payload=_public_payload(
                effect,
                status="not_repairable",
                action=action,
                code=REPAIR_NOT_NEEDED,
            ),
        )

    if action == EffectRepairAction.CONFIRM.value:
        # A confirmed repair must supply an operator-known result. When a
        # result is already durable (publication failure) the caller may
        # reuse it by passing the stored payload.
        effective_result = result
        if effective_result is None:
            effective_result = _stored_result(effect)
        if not isinstance(effective_result, dict) or not effective_result:
            return RepairOutcome(
                status="invalid_result",
                effect=effect,
                payload=_public_payload(
                    effect,
                    status="invalid_result",
                    action=action,
                    code=REPAIR_INVALID_RESULT,
                ),
            )
        try:
            serialized = serialize_result(effective_result)
        except (TypeError, ValueError):
            return RepairOutcome(
                status="invalid_result",
                effect=effect,
                payload=_public_payload(
                    effect,
                    status="invalid_result",
                    action=action,
                    code=REPAIR_INVALID_RESULT,
                ),
            )

        record, won = runs_repo.complete_effect_repair(
            effect.effect_key,
            serialized,
            request_id=idempotency_key,
            actor=actor,
        )
        if not won:
            current = record or runs_repo.get_effect(effect.effect_key)
            if (
                current is not None
                and current.status == EffectStatus.COMMITTED.value
            ):
                return repair_effect(
                    runs_repo,
                    effect_id,
                    action=action,
                    result=effective_result,
                    idempotency_key=idempotency_key,
                    actor=actor,
                    reason=reason,
                )
            if (
                current is not None
                and current.repair_status
                == EffectRepairStatus.ABANDONED.value
            ):
                return RepairOutcome(
                    status="conflict",
                    effect=current,
                    payload=_public_payload(
                        current,
                        status="conflict",
                        action=action,
                        code=REPAIR_CONFLICT,
                    ),
                )
            return RepairOutcome(
                status="failed",
                effect=current,
                payload=_public_payload(
                    current or effect,
                    status="failed",
                    action=action,
                    code=REPAIR_FAILED,
                ),
            )

        assert record is not None
        try:
            _publish_confirmed(
                runs_repo,
                record,
                result=effective_result,
                actor=actor,
                idempotency_key=idempotency_key,
            )
        except Exception as exc:
            # The final result is not replayable — do not present the
            # repair as complete. Keep the known result but revert to
            # unknown/repair_pending so a retry can republish it.
            logger.exception(
                "effect_repair: result publication failed for %s; reverting",
                effect_id,
            )
            reverted = runs_repo.revert_effect_repair(
                effect.effect_key,
                str(exc) or exc.__class__.__name__,
                request_id=idempotency_key,
            )
            return RepairOutcome(
                status="failed",
                effect=reverted or record,
                payload=_public_payload(
                    reverted or record,
                    status="failed",
                    action=action,
                    code=REPAIR_FAILED,
                ),
            )
        return RepairOutcome(
            status="repaired",
            effect=record,
            payload=_public_payload(
                record,
                status="repaired",
                action=action,
                result=effective_result,
            ),
        )

    # abandon — fail closed, keep the effect unknown, publish no result.
    record, won = runs_repo.abandon_effect_repair(
        effect.effect_key,
        request_id=idempotency_key,
        actor=actor,
        reason=reason,
    )
    if not won:
        current = record or runs_repo.get_effect(effect.effect_key)
        if (
            current is not None
            and current.repair_status == EffectRepairStatus.ABANDONED.value
        ):
            return repair_effect(
                runs_repo,
                effect_id,
                action=action,
                idempotency_key=idempotency_key,
                actor=actor,
                reason=reason,
            )
        if (
            current is not None
            and current.status == EffectStatus.COMMITTED.value
        ):
            return RepairOutcome(
                status="conflict",
                effect=current,
                payload=_public_payload(
                    current,
                    status="conflict",
                    action=action,
                    code=REPAIR_CONFLICT,
                ),
            )
        return RepairOutcome(
            status="failed",
            effect=current,
            payload=_public_payload(
                current or effect,
                status="failed",
                action=action,
                code=REPAIR_FAILED,
            ),
        )

    assert record is not None
    try:
        _publish_abandoned(
            runs_repo,
            record,
            actor=actor,
            idempotency_key=idempotency_key,
            reason=reason,
        )
    except Exception as exc:
        logger.exception(
            "effect_repair: abandon-event publication failed for %s", effect_id
        )
        return RepairOutcome(
            status="failed",
            effect=record,
            payload=_public_payload(
                record,
                status="failed",
                action=action,
                code=REPAIR_FAILED,
            ),
        )
    return RepairOutcome(
        status="abandoned",
        effect=record,
        payload=_public_payload(record, status="abandoned", action=action),
    )


__all__ = [
    "REPAIR_CONFLICT",
    "REPAIR_FAILED",
    "REPAIR_INVALID_RESULT",
    "REPAIR_NOT_FOUND",
    "REPAIR_NOT_NEEDED",
    "RepairAction",
    "RepairOutcome",
    "effect_known_facts",
    "effect_unknown_part",
    "repair_effect",
]
