"""BCDEF ticket 07 — durable janitor store runner.

Cleanup failures and unknown-outcome repairs become durable work items
(:attr:`agent_v2.persistence.JanitorTaskRecord`). This module provides
the idempotent sweep seam :func:`run_pending_janitor_tasks` that a
background loop (ticket 12) — or a test — can call:

* claim is a CAS (``pending``/``failed``, or an EXPIRED ``claimed``, →
  ``claimed``); only one worker processes a task, and a worker that crashes
  is taken over once its claim deadline lapses (BCDEF review S4),
* completion/failure is a CAS on the claim (``claimed`` → ``done`` /
  ``failed``) that also checks the claim owner and its generation, so a
  task is never completed twice and a stale worker cannot write an outcome
  — the sweep then reports ``claim_lost`` instead of ``done``/``failed``,
* cleanup retries use the ORIGINAL recorded workspace and lease — the
  janitor never calls ``create_workspace`` and never mints a new fencing
  generation to bypass the old lease. Before ANY short-circuit or
  physical work it validates the task's recorded original
  lease/workspace/sandbox/generation against the live repository state
  (ALR ticket 06): a misjudged ``failed``+``succeeded`` row with an
  explicit task REALLY retries its original target, the success
  short-circuit additionally requires ``lease=destroyed`` and no
  failure reason, a task whose live row belongs to a successor
  generation is refused with zero physical calls, and success settles
  through the fenced one-transaction settlement while re-failure
  records the real reason and available partial usage,
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
from agent_v2.sandbox_lifecycle import CLEANUP_SUCCEEDED, run_sandbox_cleanup

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent_v2.persistence import JanitorTaskRecord, SandboxLeaseRecord
    from agent_v2.runs import RunRepository

logger = logging.getLogger("ai-service.agent_v2.janitor")

CleanupHandler = Callable[["RunRepository", dict[str, Any]], None]
RepairHandler = Callable[["RunRepository", "JanitorTaskRecord"], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class JanitorRunResult:
    """Outcome of one claimed janitor task in a sweep.

    ``status`` is ``done`` / ``failed`` ONLY when the settlement CAS was
    won, i.e. the durable row really carries that outcome. ``claim_lost``
    means the work ran but this worker no longer owned the claim (another
    worker took the expired item over, or its own deadline lapsed), so
    nothing was written and the reported outcome must not be read as
    persisted (BCDEF review S4).
    """

    task_id: str
    kind: str
    status: str  # done | failed | claim_lost
    outcome: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def _default_repair(
    runs_repo: "RunRepository", task: "JanitorTaskRecord"
) -> dict[str, Any]:
    """Reconcile one unknown-outcome effect WITHOUT executing anything.

    BCDEF review S5: the ledger row is the only source of a repairable
    result. A known result (a lost success-event publication) is confirmed
    through the ticket-05 repair seam FROM THE DURABLE ROW — the janitor
    asserts nothing from its own evidence payload. When no result was
    recorded the janitor only CLAIMS the repair so an operator can
    confirm/abandon it. No model turn and no external effect is executed.
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

    known_result: dict[str, Any] | None = None
    if effect.result_json:
        try:
            parsed = json.loads(effect.result_json)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            known_result = parsed

    if isinstance(known_result, dict) and known_result:
        # No ``result`` payload: the repair seam reads the durable row, so
        # the janitor publishes exactly what the ledger recorded.
        outcome = repair_effect(
            runs_repo,
            effect_id,
            action="confirm",
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


class OriginalTargetConflict(RuntimeError):
    """The live lease row is not the task's original target (ALR 06).

    Raised BEFORE any physical work or durable lease write when the
    task's recorded original lease/workspace/sandbox/generation no
    longer matches the live repository state — a successor generation
    owns the row. The runner surfaces it as a real task failure whose
    ``last_error`` carries the conflict evidence; the successor's
    resources and durable state are never touched.
    """


def _original_target_conflict(
    task: "JanitorTaskRecord",
    evidence: dict[str, Any],
    lease: "SandboxLeaseRecord",
) -> str | None:
    """Validate the task's ORIGINAL target against the live lease row.

    Compares what the task RECORDED (its ``lease_id`` column first, then
    the evidence payload's lease id / workspace / sandbox identity /
    fencing generation) against the live lease. Only recorded fields are
    compared; ``fencing_token`` 0/absent means "not recorded" (legacy
    row). Returns a traceable conflict description, or ``None`` when
    the live lease IS the original target.

    A task that records NO identity at all against a live lease cannot
    be verified and is refused — the janitor never guesses which
    resource to destroy.
    """
    recorded_lease_id = (task.lease_id or "").strip() or str(
        evidence.get("lease_id") or ""
    ).strip()
    recorded_workspace = str(evidence.get("workspace_path") or "").strip()
    recorded_identity = str(evidence.get("sandbox_identity") or "").strip()
    try:
        recorded_token = int(evidence.get("fencing_token"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        recorded_token = 0

    checked = 0
    mismatches: list[str] = []
    if recorded_lease_id:
        checked += 1
        if lease.lease_id != recorded_lease_id:
            mismatches.append(
                f"lease_id task={recorded_lease_id!r} live={lease.lease_id!r}"
            )
    if recorded_workspace:
        checked += 1
        if lease.workspace_path != recorded_workspace:
            mismatches.append(
                "workspace_path task="
                f"{recorded_workspace!r} live={lease.workspace_path!r}"
            )
    if recorded_identity and (lease.sandbox_identity or ""):
        checked += 1
        if (lease.sandbox_identity or "") != recorded_identity:
            mismatches.append(
                "sandbox_identity task="
                f"{recorded_identity!r} live={lease.sandbox_identity!r}"
            )
    if recorded_token:
        checked += 1
        if int(lease.fencing_token or 0) != recorded_token:
            mismatches.append(
                f"fencing_token task={recorded_token} live={lease.fencing_token}"
            )

    if checked == 0:
        return (
            "original_target_conflict: task records no original identity "
            "(lease_id/workspace/fencing_token) to verify against live "
            f"lease {lease.lease_id!r} — refusing to guess the target"
        )
    if mismatches:
        return (
            "original_target_conflict: live lease is not the task's "
            "original target (" + "; ".join(mismatches) + ")"
        )
    return None


def _settle_retry_failure(
    runs_repo: "RunRepository",
    *,
    run_id: str,
    lease_id: str,
    evidence: dict[str, Any],
    failure_reason: str,
    resource_usage: dict[str, Any] | None = None,
) -> None:
    """Record a REAL retry failure through the atomic settlement (ALR 04).

    One control transaction: the idempotent retry task (dedupe reuses
    the row the runner currently owns), the lease's failed release —
    only an ACTIVE lease transitions, so the FIRST release's evidence
    on an already-failed lease is never overwritten — and the
    ``pending_janitor`` cleanup record with the fresh reason. The task
    row itself is failed separately by the runner's claim CAS. A
    bookkeeping fault here is logged, never swallowed into a success.
    """
    try:
        runs_repo.settle_cleanup_failure(
            run_id=run_id,
            lease_id=lease_id,
            evidence=evidence,
            dedupe_key=None,
            resource_usage=resource_usage,
            failure_reason=failure_reason,
        )
    except Exception:
        logger.exception(
            "janitor: settling retry failure for run %s failed", run_id
        )


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

    # ALR ticket 06 — validate the ORIGINAL target FIRST: before the
    # success short-circuit AND before any physical operation. No
    # workspace is created, no model or effect is invoked, and the run's
    # own terminal state is never touched on this path.
    if lease is not None:
        conflict = _original_target_conflict(task, evidence, lease)
        if conflict is not None:
            raise OriginalTargetConflict(conflict)

        # T06-03: the already-cleaned short-circuit is only trustworthy
        # when the ORIGINAL identity matches (validated above) AND the
        # lease itself is destroyed AND the cleanup really succeeded AND
        # no failure reason is left. A misjudged failed+succeeded row
        # does NOT short-circuit — it really retries below.
        if (
            lease.status == "destroyed"
            and lease.cleanup_status == CLEANUP_SUCCEEDED
            and not lease.failure_reason
        ):
            return {
                "run_id": run_id,
                "lease_id": lease.lease_id,
                "cleanup_status": CLEANUP_SUCCEEDED,
                "workspace_path": lease.workspace_path,
                "already_cleaned": True,
            }

    if cleanup is not None:
        try:
            cleanup(runs_repo, evidence)
        except Exception as exc:
            _settle_retry_failure(
                runs_repo,
                run_id=run_id,
                lease_id=task.lease_id or str(evidence.get("lease_id") or ""),
                evidence=evidence,
                failure_reason=f"cleanup retry failed: {exc}",
            )
            raise
        if lease is not None:
            _record, settled = runs_repo.settle_cleanup_success(
                run_id=run_id,
                lease_id=lease.lease_id,
                workspace_path=lease.workspace_path,
                sandbox_identity=lease.sandbox_identity or "",
                fencing_token=lease.fencing_token,
            )
            if not settled:
                # The handler settled the row its own way (legacy seam);
                # keep the cleanup record it may have skipped.
                runs_repo.record_sandbox_cleanup(
                    run_id, status=CLEANUP_SUCCEEDED
                )
    elif computer_factory is not None:
        if lease is None:
            # Nothing durable to clean — the workspace is already gone.
            return {
                "run_id": run_id,
                "cleanup_status": CLEANUP_SUCCEEDED,
                "already_cleaned": True,
            }
        # Reuse the ORIGINAL sandbox identity: no create_workspace, no new
        # fencing token, no second lease. The workspace/lease recorded at
        # failure time IS the retry target (validated above).
        outcome = run_sandbox_cleanup(
            computer_factory(lease.sandbox_type), lease.workspace_path
        )
        if not outcome.succeeded:
            reason = outcome.failure_reason or "cleanup failed"
            _settle_retry_failure(
                runs_repo,
                run_id=run_id,
                lease_id=task.lease_id or str(evidence.get("lease_id") or ""),
                evidence=evidence,
                failure_reason=reason,
                resource_usage=outcome.partial_usage,
            )
            detail = f"cleanup retry failed: {reason}"
            if outcome.partial_usage:
                detail += " (partial_usage: {})".format(
                    json.dumps(outcome.partial_usage, sort_keys=True)
                )
            raise RuntimeError(detail)
        _record, settled = runs_repo.settle_cleanup_success(
            run_id=run_id,
            lease_id=lease.lease_id,
            workspace_path=lease.workspace_path,
            sandbox_identity=lease.sandbox_identity or "",
            fencing_token=lease.fencing_token,
            resource_usage=outcome.usage,
        )
        if not settled:
            # Idempotent duplicate: this generation was already settled
            # (a previous worker's fenced write, or a takeover between
            # validation and settlement). The first settlement's
            # evidence stands; nothing is overwritten.
            logger.info(
                "janitor: fenced success settlement for run %s was a "
                "no-op (already settled or superseded)",
                run_id,
            )
    else:
        raise RuntimeError("cleanup_handler_unavailable")

    return {
        "run_id": run_id,
        "lease_id": lease.lease_id if lease is not None else task.lease_id,
        "cleanup_status": CLEANUP_SUCCEEDED,
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
    """Claim and execute up to ``limit`` claimable janitor tasks.

    Idempotent: only CAS-claimed tasks are processed, completed tasks are
    never picked again, and failed tasks are retried on a later sweep.
    Returns one result per processed task (skipped CAS losers excluded).

    BCDEF review S4 — crash recovery and fencing:

    * candidates are ONE bounded query over CLAIMABLE items: ``pending``,
      ``failed`` and ``claimed`` rows whose deadline has passed, ordered by
      the moment they became claimable. A hung claim therefore outranks a
      stream of newer ``pending`` work instead of starving behind it, and
      the round still processes at most ``limit`` items;
    * settlement presents the claim's owner AND its generation
      (``attempts``). When that CAS is lost the task is reported as
      ``claim_lost`` — never as ``done``/``failed`` — because the durable
      row does not carry this worker's outcome.
    """
    owner = claimer or f"janitor:{uuid.uuid4().hex}"
    candidates = runs_repo.list_janitor_tasks(claimable=True, limit=limit)

    results: list[JanitorRunResult] = []
    for task in candidates:
        claimed, won = runs_repo.claim_janitor_task(task.task_id, owner=owner)
        if not won or claimed is None:
            continue
        attempt = claimed.attempts
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
            _row, persisted = runs_repo.fail_janitor_task(
                claimed.task_id, error, owner=owner, expected_attempt=attempt
            )
            results.append(
                JanitorRunResult(
                    task_id=claimed.task_id,
                    kind=claimed.kind,
                    status="failed" if persisted else "claim_lost",
                    error=error,
                )
            )
            if not persisted:
                logger.error(
                    "janitor: settlement for task %s (%s) was refused — the"
                    " claim was taken over; reporting claim_lost, not failed",
                    claimed.task_id,
                    claimed.kind,
                )
            continue

        _row, persisted = runs_repo.complete_janitor_task(
            claimed.task_id,
            owner=owner,
            expected_attempt=attempt,
            result=outcome,
        )
        if not persisted:
            logger.error(
                "janitor: completion of task %s (%s) was refused — this"
                " worker no longer owns the claim; nothing was persisted",
                claimed.task_id,
                claimed.kind,
            )
            results.append(
                JanitorRunResult(
                    task_id=claimed.task_id,
                    kind=claimed.kind,
                    status="claim_lost",
                    outcome=outcome,
                )
            )
            continue
        results.append(
            JanitorRunResult(
                task_id=claimed.task_id,
                kind=claimed.kind,
                status="done",
                outcome=outcome,
            )
        )
    return results


__all__ = [
    "JanitorRunResult",
    "OriginalTargetConflict",
    "run_pending_janitor_tasks",
]
