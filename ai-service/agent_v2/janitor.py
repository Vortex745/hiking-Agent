"""BCDEF ticket 07 — durable janitor store runner.

Cleanup failures and unknown-outcome repairs become durable work items
(:attr:`agent_v2.persistence.JanitorTaskRecord`). This module provides
the idempotent sweep seam :func:`run_pending_janitor_tasks` that a
background loop (ticket 12) — or a test — can call:

* claim is a CAS (``pending``/``failed`` → ``claimed``); only one worker
  processes a task,
* completion/failure is a CAS on the claim (``claimed`` → ``done`` /
  ``failed``) so a task is never completed twice,
* cleanup retries use the ORIGINAL recorded workspace and lease — the
  janitor never calls ``create_workspace`` and never mints a new fencing
  generation to bypass the old lease,
* repair tasks reuse ticket-05 reconciliation: a known result is
  confirmed idempotently; otherwise the janitor only CLAIMS the repair
  for an operator. No model turn and no external effect is executed.

No background daemon is required here.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from agent_v2.effect_repair import repair_effect

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent_v2.persistence import JanitorTaskRecord
    from agent_v2.runs import RunRepository

logger = logging.getLogger("ai-service.agent_v2.janitor")

CleanupHandler = Callable[["RunRepository", dict[str, Any]], None]
RepairHandler = Callable[["RunRepository", "JanitorTaskRecord"], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class JanitorRunResult:
    """Outcome of one claimed janitor task in a sweep."""

    task_id: str
    kind: str
    status: str  # done | failed
    outcome: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def _default_repair(
    runs_repo: "RunRepository", task: "JanitorTaskRecord"
) -> dict[str, Any]:
    """Reconcile one unknown-outcome effect WITHOUT executing anything.

    A known result (e.g. a publication failure) is confirmed through the
    ticket-05 repair seam. When no result is known the janitor only
    claims the repair so an operator can confirm/abandon it.
    """
    evidence = task.evidence
    run_id = task.run_id or str(evidence.get("run_id") or "")
    effect_id = task.effect_id or str(evidence.get("effect_id") or "")
    if not effect_id:
        unknown = [
            effect
            for effect in runs_repo.list_effects_for_run(run_id)
            if effect.status == "outcome_unknown"
        ]
        if len(unknown) != 1:
            raise ValueError(
                "repair task requires an explicit effect_id "
                f"(run {run_id!r} has {len(unknown)} unknown effects)"
            )
        effect_id = unknown[0].resolved_effect_id

    effect = runs_repo.get_effect_by_effect_id(effect_id)
    if effect is None:
        raise LookupError(f"effect {effect_id!r} not found")
    if effect.status == "committed":
        return {
            "effect_id": effect_id,
            "repair_status": effect.repair_status,
            "already_repaired": True,
        }

    known_result = evidence.get("known_result")
    if known_result is None and effect.result_json:
        try:
            parsed = json.loads(effect.result_json)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            known_result = parsed

    if isinstance(known_result, dict) and known_result:
        outcome = repair_effect(
            runs_repo,
            effect_id,
            action="confirm",
            result=known_result,
            idempotency_key=f"janitor:{task.task_id}",
            actor=f"janitor:{task.task_id}",
        )
        if outcome.status not in ("repaired", "idempotent"):
            raise RuntimeError(
                f"janitor repair failed: {outcome.payload.get('code') or outcome.status}"
            )
        return {
            "effect_id": effect_id,
            "repair_status": "repaired",
            "status": outcome.status,
        }

    record, won = runs_repo.claim_effect_repair(
        effect.effect_key, owner=f"janitor:{task.task_id}"
    )
    if record is None:
        raise LookupError(f"effect {effect_id!r} not found")
    return {
        "effect_id": effect_id,
        "repair_status": record.repair_status,
        "next_action": (
            "manual_confirm"
            if record.repair_status == "repair_claimed"
            else "none"
        ),
        "claimed": won,
    }


def _run_cleanup(
    runs_repo: "RunRepository",
    task: "JanitorTaskRecord",
    *,
    cleanup: CleanupHandler | None,
    computer_factory: Callable[[str], Any] | None,
) -> dict[str, Any]:
    evidence = task.evidence
    run_id = task.run_id or str(evidence.get("run_id") or "")
    if not run_id:
        raise ValueError("cleanup task requires a run_id")

    lease = runs_repo.get_sandbox_lease(run_id)
    if lease is not None and lease.cleanup_status == "succeeded":
        return {
            "run_id": run_id,
            "cleanup_status": "succeeded",
            "already_cleaned": True,
        }

    if cleanup is not None:
        cleanup(runs_repo, evidence)
    elif computer_factory is not None:
        if lease is None:
            # Nothing durable to clean — the workspace is already gone.
            return {
                "run_id": run_id,
                "cleanup_status": "succeeded",
                "already_cleaned": True,
            }
        # Reuse the ORIGINAL sandbox identity: no create_workspace, no new
        # fencing token, no second lease. The workspace/lease recorded at
        # failure time IS the retry target.
        computer = computer_factory(lease.sandbox_type)
        computer.cleanup(lease.workspace_path)
        runs_repo.destroy_sandbox_lease(run_id=run_id)
    else:
        raise RuntimeError("cleanup_handler_unavailable")

    if lease is not None:
        runs_repo.record_sandbox_cleanup(run_id, status="succeeded")
    return {
        "run_id": run_id,
        "lease_id": lease.lease_id if lease is not None else task.lease_id,
        "cleanup_status": "succeeded",
        "workspace_path": (
            lease.workspace_path
            if lease is not None
            else evidence.get("workspace_path")
        ),
    }


def run_pending_janitor_tasks(
    runs_repo: "RunRepository",
    *,
    limit: int = 10,
    cleanup: CleanupHandler | None = None,
    computer_factory: Callable[[str], Any] | None = None,
    repair: RepairHandler | None = None,
    claimer: str | None = None,
) -> list[JanitorRunResult]:
    """Claim and execute up to ``limit`` open janitor tasks.

    Idempotent: only CAS-claimed tasks are processed, completed tasks are
    never picked again, and failed tasks are retried on a later sweep.
    Returns one result per processed task (skipped CAS losers excluded).
    """
    owner = claimer or f"janitor:{uuid.uuid4().hex}"
    candidates = runs_repo.list_janitor_tasks(status="pending", limit=limit)
    if len(candidates) < limit:
        candidates.extend(
            runs_repo.list_janitor_tasks(
                status="failed", limit=limit - len(candidates)
            )
        )

    results: list[JanitorRunResult] = []
    for task in candidates:
        claimed, won = runs_repo.claim_janitor_task(task.task_id, owner=owner)
        if not won or claimed is None:
            continue
        try:
            if claimed.kind == "cleanup":
                outcome = _run_cleanup(
                    runs_repo,
                    claimed,
                    cleanup=cleanup,
                    computer_factory=computer_factory,
                )
            elif claimed.kind == "repair":
                handler = repair or _default_repair
                outcome = handler(runs_repo, claimed)
            else:  # pragma: no cover - enqueue validates the kind
                raise ValueError(f"unknown janitor kind: {claimed.kind!r}")
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            logger.exception(
                "janitor: task %s (%s) failed", claimed.task_id, claimed.kind
            )
            runs_repo.fail_janitor_task(claimed.task_id, error, owner=owner)
            results.append(
                JanitorRunResult(
                    task_id=claimed.task_id,
                    kind=claimed.kind,
                    status="failed",
                    error=error,
                )
            )
            continue

        runs_repo.complete_janitor_task(
            claimed.task_id, owner=owner, result=outcome
        )
        results.append(
            JanitorRunResult(
                task_id=claimed.task_id,
                kind=claimed.kind,
                status="done",
                outcome=outcome,
            )
        )
    return results


__all__ = ["JanitorRunResult", "run_pending_janitor_tasks"]
