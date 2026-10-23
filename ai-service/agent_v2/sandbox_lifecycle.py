"""Sandbox lifecycle: the typed terminal-cleanup contract (ALR ticket 03).

The cleanup chain used to end in an untyped ``None``: the production
adapter swallowed the physical failure, destroyed the lease on its own,
and the orchestrator then recorded ``cleanup_status="succeeded"`` for a
run whose sandbox was never torn down — a false success with no janitor
work item, so the stale sandbox was invisible to every recovery path.

This module owns the fix:

* :class:`SandboxCleanupResult` is the ONE typed outcome a cleanup hook
  may return. It carries the real ``usage`` (success), the preserved
  ``partial_usage`` (failure) and the ``failure_reason``. ``None`` or any
  other shape is a PROTOCOL failure
  (:func:`coerce_cleanup_result`) — never a success.
* :func:`run_sandbox_cleanup` is the production adapter body: it calls
  the Computer, translates ``SandboxCleanupError`` (keeping
  ``partial_usage``) and unexpected exceptions into the typed result, and
  performs NO lease mutation of its own.
* :func:`settle_terminal_cleanup` is the settlement entry for the run's
  CURRENT resource: success destroys the lease with the real usage and
  records ``succeeded``; failure enqueues exactly one idempotent janitor
  retry for the ORIGINAL target, destroys the lease as ``failed`` with
  the preserved partial usage, and records ``pending_janitor``.

Ticket 04 makes the failure settlement atomic. To keep that possible,
every write helper here performs exactly ONE repository call with no
transaction of its own — the transaction boundaries stay where the
repository puts them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from .computer import Computer, SandboxCleanupError

logger = logging.getLogger(__name__)

# Lease-level cleanup statuses (``sandbox_leases.cleanup_status``).
CLEANUP_SUCCEEDED = "succeeded"
CLEANUP_PENDING_JANITOR = "pending_janitor"

# Typed-result statuses (``SandboxCleanupResult.status``).
RESULT_SUCCEEDED = "succeeded"
RESULT_FAILED = "failed"

# The dedupe key shape that pins a cleanup retry to the ORIGINAL run.
CLEANUP_TASK_DEDUPE_KEY = "cleanup:{run_id}"

# SandboxCleanup hooks take (computer, workspace_path, runs_repo, run_id)
# and return a SandboxCleanupResult (legacy hooks returned None — that is
# now a protocol failure, see coerce_cleanup_result).
SandboxCleanupHook = Callable[..., "SandboxCleanupResult"]


def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
    """Normalize a usage value (ResourceUsage | mapping | None) to a dict."""
    if usage is None:
        return None
    to_dict = getattr(usage, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    if isinstance(usage, dict):
        return dict(usage)
    return None


@dataclass(frozen=True, slots=True)
class SandboxCleanupResult:
    """The explicit terminal-cleanup outcome.

    ``status`` is ``succeeded`` or ``failed`` — nothing else. On success
    ``usage`` carries the real resource usage reported by the Computer;
    on failure ``failure_reason`` is required and ``partial_usage``
    preserves whatever usage was gathered before the failure. A result
    without this shape never reaches the ledger: the orchestrator treats
    it as a protocol failure.
    """

    status: str
    usage: dict[str, Any] | None = None
    partial_usage: dict[str, Any] | None = None
    failure_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == RESULT_SUCCEEDED

    @classmethod
    def success(cls, usage: Any) -> "SandboxCleanupResult":
        return cls(status=RESULT_SUCCEEDED, usage=_usage_to_dict(usage))

    @classmethod
    def failure(
        cls,
        reason: Any,
        *,
        partial_usage: Any = None,
    ) -> "SandboxCleanupResult":
        text = str(reason).strip() or "cleanup failed"
        return cls(
            status=RESULT_FAILED,
            partial_usage=_usage_to_dict(partial_usage),
            failure_reason=text,
        )

    @property
    def resource_usage(self) -> dict[str, Any] | None:
        """The usage evidence to persist on the lease for this outcome."""
        return self.usage if self.succeeded else self.partial_usage


def coerce_cleanup_result(value: Any) -> SandboxCleanupResult:
    """Interpret a cleanup-hook return value; ambiguity is a failure.

    Only a :class:`SandboxCleanupResult` is a valid outcome. ``None``
    (the legacy adapter swallow), a dict, or any other shape means the
    hook broke the protocol — the caller registers a failure, it NEVER
    records a success it cannot vouch for.
    """
    if isinstance(value, SandboxCleanupResult):
        return value
    return SandboxCleanupResult.failure(
        "cleanup protocol failure: hook returned "
        f"{type(value).__name__ if value is not None else 'None'} "
        "instead of SandboxCleanupResult"
    )


def run_sandbox_cleanup(
    computer: Computer, workspace_path: str
) -> SandboxCleanupResult:
    """Production adapter: run the physical cleanup, translate honestly.

    ``SandboxCleanupError`` keeps its ``partial_usage``; an unexpected
    exception becomes a failure result too. The adapter performs no
    lease mutation and never lets a failure masquerade as success — the
    settlement entry owns the durable writes.
    """
    try:
        usage = computer.cleanup(workspace_path)
    except SandboxCleanupError as exc:
        # Preserve partial usage gathered before the cleanup failure so
        # the lease still records resource metrics (spec AC6).
        logger.warning(
            "sandbox_lifecycle: cleanup failed for workspace=%s: %s",
            workspace_path,
            exc,
        )
        return SandboxCleanupResult.failure(
            f"cleanup failed: {exc}", partial_usage=exc.partial_usage
        )
    except Exception as exc:
        logger.exception(
            "sandbox_lifecycle: unexpected cleanup error for workspace=%s",
            workspace_path,
        )
        return SandboxCleanupResult.failure(f"cleanup unexpected error: {exc}")
    return SandboxCleanupResult.success(usage)


def build_cleanup_evidence(
    runs_repo: Any,
    *,
    computer: Any,
    workspace_path: str,
    run_id: str,
    outcome: SandboxCleanupResult,
) -> dict[str, Any]:
    """Assemble the janitor evidence for the run's ORIGINAL resource.

    Single-read helper (no writes, no transaction): ticket 06 uses these
    identity fields to verify a retry still targets the original lease.
    """
    try:
        lease = runs_repo.get_sandbox_lease(run_id)
    except Exception:
        logger.exception(
            "sandbox_lifecycle: reading lease for cleanup evidence failed "
            "run_id=%s",
            run_id,
        )
        lease = None
    try:
        effect_ids = [
            effect.resolved_effect_id
            for effect in runs_repo.list_effects_for_run(run_id)
        ]
    except Exception:
        logger.exception(
            "sandbox_lifecycle: listing effects for cleanup evidence failed "
            "run_id=%s",
            run_id,
        )
        effect_ids = []
    return {
        "run_id": run_id,
        "lease_id": lease.lease_id if lease is not None else "",
        "sandbox_type": (
            lease.sandbox_type
            if lease is not None
            else getattr(computer, "sandbox_type", "")
        ),
        "workspace_path": workspace_path,
        "sandbox_identity": lease.sandbox_identity if lease is not None else "",
        "fencing_token": lease.fencing_token if lease is not None else 0,
        "effect_ids": effect_ids,
        "failure_reason": outcome.failure_reason or "cleanup failed",
    }


def enqueue_cleanup_retry(
    runs_repo: Any,
    *,
    run_id: str,
    lease_id: str,
    evidence: dict[str, Any],
) -> tuple[Any, bool]:
    """Enqueue the idempotent janitor retry for the ORIGINAL target.

    Thin, transaction-free pass-through over the repository's deduped
    enqueue (``cleanup:{run_id}``), so ticket 04 can lift this write into
    the atomic settlement without another transaction wrapper.
    """
    return runs_repo.enqueue_janitor_task(
        kind="cleanup",
        run_id=run_id,
        lease_id=lease_id,
        evidence=evidence,
        dedupe_key=CLEANUP_TASK_DEDUPE_KEY.format(run_id=run_id),
    )


def settle_terminal_cleanup(
    runs_repo: Any,
    *,
    computer: Any,
    workspace_path: str,
    run_id: str | None,
    outcome: SandboxCleanupResult,
) -> str | None:
    """Settle the durable records for the run's CURRENT resource.

    Success: destroy the lease carrying the real usage and record
    ``succeeded``. Failure: enqueue one idempotent janitor retry for the
    original target, destroy the lease as ``failed`` with the preserved
    partial usage + failure reason, and record ``pending_janitor``.

    Each step is ONE repository write with no transaction of its own
    (ticket 04 wraps them atomically). Bookkeeping failures are logged
    and never re-raise: the terminal that was already emitted stands.

    Returns the janitor task id on the failure path, else ``None``.
    """
    if runs_repo is None or run_id is None:
        return None
    try:
        if outcome.succeeded:
            runs_repo.destroy_sandbox_lease(
                run_id=run_id,
                resource_usage=outcome.usage,
                failure_reason=None,
            )
            runs_repo.record_sandbox_cleanup(run_id, status=CLEANUP_SUCCEEDED)
            return None

        evidence = build_cleanup_evidence(
            runs_repo,
            computer=computer,
            workspace_path=workspace_path,
            run_id=run_id,
            outcome=outcome,
        )
        task, _created = enqueue_cleanup_retry(
            runs_repo,
            run_id=run_id,
            lease_id=evidence["lease_id"],
            evidence=evidence,
        )
        lease = runs_repo.get_sandbox_lease(run_id)
        if lease is not None and lease.status == "active":
            runs_repo.destroy_sandbox_lease(
                run_id=run_id,
                resource_usage=outcome.partial_usage,
                failure_reason=outcome.failure_reason,
            )
        runs_repo.record_sandbox_cleanup(
            run_id,
            status=CLEANUP_PENDING_JANITOR,
            error=outcome.failure_reason,
            janitor_task_id=task.task_id,
        )
        return task.task_id
    except Exception:
        logger.exception(
            "sandbox_lifecycle: recording cleanup failure / janitor task "
            "failed for run_id=%s",
            run_id,
        )
        return None


def run_terminal_cleanup(
    hook: SandboxCleanupHook,
    computer: Computer,
    workspace_path: str,
    runs_repo: Any,
    run_id: str | None,
) -> None:
    """Idempotent terminal cleanup: hook → typed result → settlement.

    A repeated call after a successful recorded cleanup is a no-op
    BEFORE the hook runs, so the physical teardown happens exactly once.
    A hook that raises is an explicit failure; a hook that returns
    anything but :class:`SandboxCleanupResult` is a protocol failure.
    """
    if runs_repo is not None and run_id is not None:
        try:
            existing = runs_repo.get_sandbox_lease(run_id)
        except Exception:
            existing = None
        if (
            existing is not None
            and existing.cleanup_status == CLEANUP_SUCCEEDED
            and existing.status != "active"
        ):
            return

    try:
        outcome = coerce_cleanup_result(
            hook(computer, workspace_path, runs_repo, run_id)
        )
    except Exception as exc:
        logger.exception(
            "sandbox_lifecycle: terminal cleanup failed for run_id=%s "
            "workspace=%s",
            run_id,
            workspace_path,
        )
        outcome = SandboxCleanupResult.failure(f"cleanup failed: {exc}")

    settle_terminal_cleanup(
        runs_repo,
        computer=computer,
        workspace_path=workspace_path,
        run_id=run_id,
        outcome=outcome,
    )
