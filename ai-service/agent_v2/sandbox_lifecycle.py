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
  CURRENT resource, and it is ATOMIC (ALR ticket 04): success settles
  the identity-matched lease to ``destroyed`` + ``succeeded`` with the
  real usage in ONE fenced repository transaction (a lease the same
  generation had settled ``failed`` is repaired by the same commit);
  failure commits the failed lease (preserved partial usage + failure
  reason), the ``pending_janitor`` cleanup record and exactly one
  idempotent janitor retry for the ORIGINAL target in ONE control
  transaction — together or not at all. The repository owns every
  transaction boundary; this module never wraps a write in another one.
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


def settle_terminal_cleanup(
    runs_repo: Any,
    *,
    computer: Any,
    workspace_path: str,
    run_id: str | None,
    outcome: SandboxCleanupResult,
) -> str | None:
    """Settle the durable records for the run's CURRENT resource — ATOMICALLY.

    Success: the repository's fenced settlement transitions the lease
    whose identity (lease id / workspace / fencing generation) was read
    BEFORE the physical cleanup to ``destroyed`` + ``succeeded`` with the
    real usage — in one transaction. A lease settled ``failed`` by the
    SAME generation is repaired by the same commit (ticket 06's retry
    fence); a successor generation is never touched.

    Failure: the repository's ONE-transaction settlement commits the
    failed lease (preserved partial usage + failure reason), the
    ``pending_janitor`` cleanup record, and exactly one idempotent
    janitor retry for the ORIGINAL target — together or not at all.

    Bookkeeping failures (e.g. a fully unwritable database) are logged
    and never re-raise: the terminal that was already emitted stands, no
    durable success or enqueue is claimed, and the logged failure stays
    available for later explicit handling.

    Returns the janitor task id on the failure path, else ``None``.
    """
    if runs_repo is None or run_id is None:
        return None
    try:
        if outcome.succeeded:
            lease = runs_repo.get_sandbox_lease(run_id)
            if lease is None:
                return None
            if lease.workspace_path != workspace_path:
                # The cleanup succeeded for a workspace this lease does not
                # own — a stale worker after a successor takeover. Settling
                # it would credit the wrong (or someone else's) resource.
                logger.warning(
                    "sandbox_lifecycle: refusing success settlement for "
                    "run_id=%s — cleanup workspace %s is not the leased "
                    "workspace %s",
                    run_id,
                    workspace_path,
                    lease.workspace_path,
                )
                return None
            runs_repo.settle_cleanup_success(
                run_id=run_id,
                lease_id=lease.lease_id,
                workspace_path=lease.workspace_path,
                sandbox_identity=lease.sandbox_identity or "",
                fencing_token=lease.fencing_token,
                resource_usage=outcome.usage,
            )
            return None

        evidence = build_cleanup_evidence(
            runs_repo,
            computer=computer,
            workspace_path=workspace_path,
            run_id=run_id,
            outcome=outcome,
        )
        task, _created, _lease = runs_repo.settle_cleanup_failure(
            run_id=run_id,
            lease_id=evidence["lease_id"],
            evidence=evidence,
            dedupe_key=CLEANUP_TASK_DEDUPE_KEY.format(run_id=run_id),
            resource_usage=outcome.partial_usage,
            failure_reason=outcome.failure_reason,
        )
        return task.task_id if task is not None else None
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
