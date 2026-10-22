"""BCDEF ticket 06 — Sandbox active lease + fencing token.

A sandbox may only execute while a durable ``active`` lease exists for
the run. The lease row carries:

* ``owner_id`` — the worker that last acquired/renewed the lease;
* ``expires_at`` / ``heartbeat_at`` — the TTL heartbeat;
* ``fencing_token`` — monotonically increasing generation. A lease
  re-acquired after release/expiry gets ``token + 1``; a stale worker
  holding an older token is fenced out;
* ``sandbox_identity`` — the concrete sandbox (type + workspace) the
  lease binds.

The module is pure policy: persistence lives in
:mod:`agent_v2.persistence` (``sandbox_leases``) and the CAS methods are
exposed through :class:`agent_v2.runs.RunRepository`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# Heartbeat/TTL used when a caller does not supply one. Long enough for a
# 30s sandbox command plus slack; short enough that a crashed worker's
# lease eventually surfaces as expired instead of blocking forever.
SANDBOX_LEASE_TTL_SECONDS: float = 600.0

# Lease statuses. ``active`` is the only status that authorizes sandbox
# execution. ``destroyed`` is a clean release, ``failed`` is a release
# with a cleanup error, ``expired`` marks a TTL lapse.
SANDBOX_LEASE_STATUSES: frozenset[str] = frozenset(
    {"active", "destroyed", "failed", "expired"}
)


class SandboxLeaseStatus(str, Enum):
    ACTIVE = "active"
    DESTROYED = "destroyed"
    FAILED = "failed"
    EXPIRED = "expired"


class SandboxLeaseLostError(Exception):
    """Stable error raised when a sandbox lease no longer authorizes work.

    ``code`` is one of ``sandbox_lease_lost`` (missing/not active),
    ``lease_expired`` (TTL elapsed), or ``lease_fencing_rejected``
    (stale fencing token after a new lease generation took over).
    """

    SANDBOX_LEASE_LOST = "sandbox_lease_lost"
    LEASE_EXPIRED = "lease_expired"
    LEASE_FENCING_REJECTED = "lease_fencing_rejected"

    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        self.message = message or code
        super().__init__(f"{code}: {self.message}")


def parse_iso(value: str | None) -> datetime | None:
    """Parse a stored UTC ISO timestamp; ``None``/invalid → ``None``."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def lease_is_expired(
    expires_at: str | None,
    now: datetime | None = None,
) -> bool:
    """True iff the lease TTL elapsed.

    A missing ``expires_at`` is a legacy row (pre-ticket-06) and is NOT
    treated as expired — it stays readable and renewable until the first
    heartbeat writes an expiry.
    """
    expiry = parse_iso(expires_at)
    if expiry is None:
        return False
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return expiry <= current


@dataclass(frozen=True, slots=True)
class SandboxLeaseHandle:
    """The fenced identity a worker uses to authorize sandbox work."""

    lease_id: str
    run_id: str
    owner_id: str
    fencing_token: int
    sandbox_identity: str = ""
    expires_at: str | None = None

    def __post_init__(self) -> None:
        if not self.lease_id or not self.run_id:
            raise ValueError("lease handle requires lease_id and run_id")
        if self.fencing_token < 0:
            raise ValueError("fencing_token must be non-negative")


class SandboxLeaseGuard:
    """Read-side fence used by every sandbox execution path.

    ``assert_active`` is called BEFORE any sandbox command and again
    after the external command returns (before publishing success). A
    stale worker (expired lease or superseded fencing token) fails with
    a stable :class:`SandboxLeaseLostError` instead of silently running
    without a lease.
    """

    def __init__(
        self,
        runs_repo: Any,
        *,
        run_id: str,
        fencing_token: int,
        lease_id: str = "",
        owner_id: str = "",
        ttl_seconds: float = SANDBOX_LEASE_TTL_SECONDS,
    ) -> None:
        self._runs_repo = runs_repo
        self.run_id = run_id
        self.fencing_token = int(fencing_token)
        self.lease_id = lease_id
        self.owner_id = owner_id
        self.ttl_seconds = ttl_seconds

    def assert_active(self, now: datetime | None = None) -> Any:
        """Raise unless the run's current lease is active and unexpired."""
        record = self._runs_repo.get_sandbox_lease(self.run_id)
        if record is None or record.status != SandboxLeaseStatus.ACTIVE.value:
            raise SandboxLeaseLostError(
                SandboxLeaseLostError.SANDBOX_LEASE_LOST,
                f"run {self.run_id!r} has no active sandbox lease",
            )
        if lease_is_expired(record.expires_at, now=now):
            raise SandboxLeaseLostError(
                SandboxLeaseLostError.LEASE_EXPIRED,
                f"sandbox lease for run {self.run_id!r} expired",
            )
        if (
            self.fencing_token
            and record.fencing_token
            and record.fencing_token != self.fencing_token
        ):
            raise SandboxLeaseLostError(
                SandboxLeaseLostError.LEASE_FENCING_REJECTED,
                f"stale fencing token {self.fencing_token} for run "
                f"{self.run_id!r} (current {record.fencing_token})",
            )
        return record

    def renew(self, now: datetime | None = None) -> Any:
        """Heartbeat the lease under the CURRENT fencing token.

        Returns the renewed record. An expired lease or a superseded
        token raises :class:`SandboxLeaseLostError` — a stale worker can
        never revive its lease.
        """
        record, renewed = self._runs_repo.renew_sandbox_lease(
            run_id=self.run_id,
            fencing_token=self.fencing_token,
            owner_id=self.owner_id,
            ttl_seconds=self.ttl_seconds,
            now=now,
        )
        if record is not None and record.fencing_token != self.fencing_token:
            raise SandboxLeaseLostError(
                SandboxLeaseLostError.LEASE_FENCING_REJECTED,
                f"stale fencing token {self.fencing_token} for run "
                f"{self.run_id!r}",
            )
        if not renewed:
            raise SandboxLeaseLostError(
                SandboxLeaseLostError.LEASE_EXPIRED,
                f"sandbox lease for run {self.run_id!r} cannot be renewed",
            )
        return record


__all__ = [
    "SANDBOX_LEASE_STATUSES",
    "SANDBOX_LEASE_TTL_SECONDS",
    "SandboxLeaseGuard",
    "SandboxLeaseHandle",
    "SandboxLeaseLostError",
    "SandboxLeaseStatus",
    "lease_is_expired",
    "parse_iso",
]
