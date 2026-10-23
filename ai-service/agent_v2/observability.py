"""Chat v2 observability, capacity, and migration threshold helpers."""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from agent_v2.persistence import ControlStore


logger = logging.getLogger("ai-service.agent_v2.observability")

_RUN_TERMINALS = {"done", "error", "cancelled"}
_RAW_SAMPLING_MINIMUM = timedelta(0)
_MIGRATION_THRESHOLD_WINDOW = timedelta(minutes=15)

# BCDEF spec 07 / ticket 12: the seven boundary metrics an operator must be
# able to see while the unified execution boundary is rolled out. Six are
# :class:`agent_v2.execution_contract.ExecutionErrorCode` values; the last
# one (``repair_age``) is the age of an open unknown-outcome repair, so it
# is reported as a seconds gauge (see :meth:`BoundaryMetrics.snapshot`).
BOUNDARY_METRIC_NAMES: tuple[str, ...] = (
    "admission_failed",
    "outcome_unknown",
    "terminal_persist_failed",
    "lease_expired",
    "cleanup_pending",
    "replay_gap",
    "repair_age",
)

# Alert thresholds for the same seven (BCDEF ticket 12 rollout gate). They
# are deliberately blunt process-lifetime counters: one breach is noise, a
# sustained breach for ``_MIGRATION_THRESHOLD_WINDOW`` escalates through
# :class:`MigrationThresholdMonitor`. ``repair_age`` is in seconds.
BOUNDARY_ALERT_THRESHOLDS: dict[str, float] = {
    "admission_failed": 10.0,
    "outcome_unknown": 5.0,
    "terminal_persist_failed": 1.0,
    "lease_expired": 5.0,
    "cleanup_pending": 5.0,
    "replay_gap": 20.0,
    "repair_age": 900.0,
}


def _now_iso(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class CapacityExceeded(RuntimeError):
    """Raised when a run or sandbox cannot be admitted within budget."""

    def __init__(
        self,
        scope: str,
        message: str | None = None,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message or f"{scope} capacity exhausted")
        self.scope = scope
        self.resource = scope
        self.retryable = True
        self.retry_after_seconds = retry_after_seconds


@dataclass(slots=True)
class _CapacityLease:
    guard: "CapacityGuard"
    scope: Literal["run", "sandbox"]
    key: str | None = None
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.guard._release(self.scope, self.key, self)


class CapacityGuard:
    """Bound active runs and sandboxes with a bounded wait queue."""

    def __init__(
        self,
        *,
        max_runs: int = 20,
        max_sandboxes: int = 10,
        queue_timeout_seconds: float = 60.0,
    ) -> None:
        if max_runs < 0 or max_sandboxes < 0:
            raise ValueError("capacity limits must be non-negative")
        if queue_timeout_seconds < 0:
            raise ValueError("queue_timeout_seconds must be non-negative")
        self.max_runs = max_runs
        self.max_sandboxes = max_sandboxes
        self.queue_timeout_seconds = queue_timeout_seconds
        self._run_sem = asyncio.Semaphore(max_runs) if max_runs > 0 else None
        self._sandbox_sem = (
            asyncio.Semaphore(max_sandboxes) if max_sandboxes > 0 else None
        )
        self._sandbox_leases: dict[str, _CapacityLease] = {}
        self._active_runs = 0
        self._active_sandboxes = 0
        self._run_waiters = 0
        self._sandbox_waiters = 0

    async def acquire_run(self) -> _CapacityLease:
        return await self._acquire("run")

    async def acquire_sandbox(self, key: str | None = None) -> _CapacityLease:
        if key is not None and key in self._sandbox_leases:
            return self._sandbox_leases[key]
        lease = await self._acquire("sandbox")
        lease.key = key
        if key is not None:
            self._sandbox_leases[key] = lease
        return lease

    def release_sandbox(self, key: str) -> None:
        lease = self._sandbox_leases.get(key)
        if lease is not None:
            lease.release()

    async def _acquire(self, scope: Literal["run", "sandbox"]) -> _CapacityLease:
        limit = self.max_runs if scope == "run" else self.max_sandboxes
        if limit <= 0:
            raise CapacityExceeded(scope)

        semaphore = self._run_sem if scope == "run" else self._sandbox_sem
        assert semaphore is not None
        waiting = semaphore.locked()
        waiter_attr = "_run_waiters" if scope == "run" else "_sandbox_waiters"
        max_waiters = max(limit, 1)
        if waiting:
            if getattr(self, waiter_attr) >= max_waiters:
                raise CapacityExceeded(scope)
            setattr(self, waiter_attr, getattr(self, waiter_attr) + 1)
        try:
            await asyncio.wait_for(
                semaphore.acquire(),
                timeout=self.queue_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise CapacityExceeded(
                scope,
                retry_after_seconds=self.queue_timeout_seconds,
            ) from exc
        finally:
            if waiting:
                setattr(self, waiter_attr, getattr(self, waiter_attr) - 1)
        if scope == "run":
            self._active_runs += 1
        else:
            self._active_sandboxes += 1
        return _CapacityLease(self, scope)

    def _release(
        self,
        scope: Literal["run", "sandbox"],
        key: str | None,
        lease: _CapacityLease,
    ) -> None:
        semaphore = self._run_sem if scope == "run" else self._sandbox_sem
        if semaphore is None:
            return
        semaphore.release()
        if scope == "run":
            self._active_runs -= 1
        else:
            self._active_sandboxes -= 1
        if key is not None and self._sandbox_leases.get(key) is lease:
            self._sandbox_leases.pop(key, None)

    def snapshot(self) -> dict[str, int]:
        """Return the number of active runs/sandboxes."""
        return {
            "active_runs": self._active_runs,
            "active_sandboxes": self._active_sandboxes,
        }


class TelemetryStore:
    """Redacted in-memory telemetry for v2 runs."""

    _SAFE_FIELDS = {
        "conversation_id",
        "duration_ms",
        "effect_statuses",
        "e2b_cost_usd",
        "error_code",
        "event_count",
        "input_tokens",
        "last_event_type",
        "model_cost_usd",
        "model_name",
        "output_tokens",
        "profile_name",
        "request_id",
        "resource_usage",
        "started_at",
        "status",
        "tool_names",
        "tool_statuses",
        "approval_statuses",
        "updated_at",
        "ended_at",
    }

    def __init__(self, control: ControlStore | None = None) -> None:
        self._lock = threading.RLock()
        self._control = control
        self._runs: dict[str, dict[str, Any]] = {}
        self._raw_sampling: dict[str, Any] | None = None
        self._raw_sampling_audit: list[dict[str, Any]] = []

    def start_run(
        self,
        run_id: str,
        *,
        profile_name: str,
        model_name: str,
        conversation_id: str | None = None,
        request_id: str | None = None,
        started_at: str | None = None,
    ) -> dict[str, Any]:
        now = started_at or _now_iso()
        return self.record(
            run_id,
            profile_name=profile_name,
            model_name=model_name,
            conversation_id=conversation_id,
            request_id=request_id,
            status="running",
            started_at=now,
            updated_at=now,
            duration_ms=0,
            input_tokens=0,
            output_tokens=0,
            model_cost_usd=0.0,
            e2b_cost_usd=0.0,
            tool_names=[],
            tool_statuses=[],
            approval_statuses=[],
            effect_statuses=[],
            event_count=0,
            last_event_type=None,
            error_code=None,
        )

    def observe_event(
        self,
        run_id: str,
        *,
        event_type: str,
        duration_ms: float,
        tool_name: str | None = None,
        state: str | None = None,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {
            "last_event_type": event_type,
            "updated_at": _now_iso(),
            "duration_ms": duration_ms,
        }

        if event_type in {"run_started", "progress", "heartbeat", "run.paused"}:
            updates["status"] = "paused" if event_type == "run.paused" else "running"
        elif event_type in _RUN_TERMINALS:
            updates["status"] = event_type
            updates["ended_at"] = _now_iso()
        if error_code is not None:
            updates["error_code"] = error_code

        if tool_name:
            updates["tool_names"] = [tool_name]
        if event_type == "tool_result" and state:
            updates["tool_statuses"] = [state]
            if tool_name in {"generate_pdf", "sandbox_bash"}:
                effect_status = "committed" if state == "ok" else "failed"
                updates["effect_statuses"] = [effect_status]
        if event_type == "approval.required":
            updates["approval_statuses"] = ["pending"]
        if event_type == "approval.resolved" and state:
            updates["approval_statuses"] = [state]
        if event_type == "effect.outcome_unknown":
            updates["effect_statuses"] = ["outcome_unknown"]

        return self.record(run_id, **updates)

    def finalize_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        ended_at: str | None = None,
    ) -> dict[str, Any]:
        record = self._runs.get(run_id)
        if record is None:
            return {}
        final_status = status or record.get("status") or "unknown"
        updates: dict[str, Any] = {
            "status": final_status,
            "ended_at": ended_at or _now_iso(),
            "updated_at": ended_at or _now_iso(),
        }
        started_at = _parse_iso(record.get("started_at"))
        ended_dt = _parse_iso(updates["ended_at"])
        if started_at is not None and ended_dt is not None:
            updates["duration_ms"] = max(
                0, int((ended_dt - started_at).total_seconds() * 1000)
            )
        return self.record(run_id, **updates)

    def record(self, run_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            current = copy.deepcopy(self._runs.get(run_id, {}))
            if not current and self._control is not None:
                current = self._control.get_run_telemetry(run_id) or {}
            merged = self._merge_safe_fields(current, fields)
            merged["run_id"] = run_id
            self._runs[run_id] = merged
            if self._control is not None:
                self._control.upsert_run_telemetry(
                    run_id=run_id,
                    profile_name=str(merged.get("profile_name", "")),
                    tool_names=list(merged.get("tool_names", [])),
                    status=str(merged.get("status", "running")),
                    duration_ms=float(merged.get("duration_ms", 0.0)),
                    input_tokens=int(merged.get("input_tokens", 0)),
                    output_tokens=int(merged.get("output_tokens", 0)),
                    model_name=str(merged.get("model_name", "")),
                    model_cost_usd=float(merged.get("model_cost_usd", 0.0)),
                    e2b_cost_usd=float(merged.get("e2b_cost_usd", 0.0)),
                    approval_statuses=list(merged.get("approval_statuses", [])),
                    effect_statuses=list(merged.get("effect_statuses", [])),
                    resource_usage=dict(merged.get("resource_usage", {})),
                    error_code=merged.get("error_code"),
                )
            return copy.deepcopy(merged)

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._runs.get(run_id)
            if record is not None:
                return copy.deepcopy(record)
            if self._control is not None:
                return self._control.get_run_telemetry(run_id)
            return None

    def cost_summary(self) -> dict[str, float | int]:
        with self._lock:
            records = (
                self._control.list_run_telemetry()
                if self._control is not None
                else list(self._runs.values())
            )
        return {
            "run_count": len(records),
            "model_cost_usd": round(
                sum(float(record.get("model_cost_usd", 0.0)) for record in records),
                10,
            ),
            "e2b_cost_usd": round(
                sum(float(record.get("e2b_cost_usd", 0.0)) for record in records),
                10,
            ),
        }

    def enable_raw_sampling(
        self,
        operator: str,
        reason: str,
        expires_at: datetime,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if not operator.strip():
            raise ValueError("operator is required")
        if not reason.strip():
            raise ValueError("reason is required")
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= current:
            raise ValueError("expires_at must be in the future")

        audit = {
            "operator": operator,
            "reason": reason,
            "enabled_at": _now_iso(current),
            "expires_at": _now_iso(expires_at),
            "disabled_at": None,
        }
        with self._lock:
            if self._raw_sampling is not None:
                self._raw_sampling["disabled_at"] = audit["enabled_at"]
                self._raw_sampling["updated_at"] = audit["enabled_at"]
            self._raw_sampling = audit
            self._raw_sampling_audit.append(audit)
            if self._control is not None:
                self._control.insert_raw_sampling_audit(
                    sample_id=str(uuid.uuid4()),
                    operator=operator,
                    reason=reason,
                    enabled_at=audit["enabled_at"],
                    expires_at=audit["expires_at"],
                )
            return copy.deepcopy(audit)

    def active_raw_sampling(
        self,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        with self._lock:
            if self._control is not None:
                now_iso = _now_iso(current)
                self._control.disable_expired_raw_sampling(now_iso)
                return self._control.get_active_raw_sampling(now_iso)
            audit = self._raw_sampling
            if audit is None:
                return None
            expires_at = _parse_iso(audit["expires_at"])
            if expires_at is not None and current >= expires_at:
                audit["disabled_at"] = _now_iso(current)
                self._raw_sampling = None
                return None
            return copy.deepcopy(audit)

    def raw_sampling_audit(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._control is not None:
                return self._control.list_raw_sampling_audit()
            return [copy.deepcopy(item) for item in reversed(self._raw_sampling_audit)]

    def _merge_safe_fields(
        self,
        record: dict[str, Any],
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        merged = copy.deepcopy(record)
        for key, value in fields.items():
            if key not in self._SAFE_FIELDS or value is None:
                continue
            if key in {"tool_names", "tool_statuses", "approval_statuses", "effect_statuses"}:
                merged[key] = self._merge_list(merged.get(key), value)
            elif key == "resource_usage":
                merged[key] = copy.deepcopy(value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def _merge_list(self, current: Any, value: Any) -> list[Any]:
        items = list(current or [])
        if isinstance(value, (list, tuple, set)):
            new_values = list(value)
        else:
            new_values = [value]
        for item in new_values:
            if item not in items:
                items.append(copy.deepcopy(item))
        return items


class MigrationThresholdMonitor:
    """Detect sustained migration thresholds that warrant escalation."""

    def __init__(self) -> None:
        self._first_seen: dict[str, datetime] = {}

    def observe(
        self,
        metrics: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)

        observations = [
            (
                "active_runs",
                float(metrics.get("active_runs", 0) or 0),
                20.0,
                float(metrics.get("active_runs", 0) or 0) > 20.0,
            ),
            (
                "commit_p95_ms",
                float(metrics.get("commit_p95_ms", 0.0) or 0.0),
                100.0,
                float(metrics.get("commit_p95_ms", 0.0) or 0.0) > 100.0,
            ),
            (
                "sqlite_busy_ratio",
                self._busy_ratio(metrics),
                0.005,
                self._busy_ratio(metrics) > 0.005,
            ),
            (
                "second_writer_required",
                bool(metrics.get("second_writer_required", False)),
                True,
                bool(metrics.get("second_writer_required", False)),
            ),
            # BCDEF ticket 12: the seven boundary metrics ride the same
            # sustained-breach rule. A key that is absent from the input
            # never alerts, so existing callers keep their behaviour.
            *self._boundary_observations(metrics),
        ]

        evidence: list[dict[str, Any]] = []
        for metric, value, threshold, exceeded in observations:
            if exceeded:
                first_seen = self._first_seen.setdefault(metric, current)
                if current - first_seen >= _MIGRATION_THRESHOLD_WINDOW:
                    evidence.append(
                        {
                            "metric": metric,
                            "value": value,
                            "threshold": threshold,
                            "first_seen_at": first_seen.isoformat(),
                            "observed_at": current.isoformat(),
                            "sustained_for_seconds": int(
                                (current - first_seen).total_seconds()
                            ),
                        }
                    )
            else:
                self._first_seen.pop(metric, None)
        return evidence

    def _busy_ratio(self, metrics: dict[str, Any]) -> float:
        busy = float(metrics.get("busy_writes", 0) or 0)
        total = float(metrics.get("total_writes", 0) or 0)
        if total <= 0:
            return 0.0
        return busy / total

    @staticmethod
    def _boundary_observations(
        metrics: dict[str, Any],
    ) -> list[tuple[str, float, float, bool]]:
        """Boundary-metric observations for the sustained-breach rule."""
        observations: list[tuple[str, float, float, bool]] = []
        for name in BOUNDARY_METRIC_NAMES:
            if name not in metrics:
                continue
            threshold = BOUNDARY_ALERT_THRESHOLDS[name]
            value = float(metrics.get(name, 0) or 0)
            observations.append((name, value, threshold, value > threshold))
        return observations


class BoundaryMetrics:
    """Process-local counters for the seven BCDEF boundary metrics.

    Spec 07 requires metrics AND alerts for
    ``admission_failed``/``outcome_unknown``/``terminal_persist_failed``/
    ``lease_expired``/``cleanup_pending``/``replay_gap``/``repair_age``.
    Counters here are the emission surface; the durable facts they were
    derived from stay in ``agent_control.sqlite`` (runs, effects,
    sandbox_leases, janitor_tasks), so a restart loses the counts but never
    the evidence — an operator can still reconcile from a snapshot.

    Only metric names, counts, ISO timestamps and opaque run ids are kept:
    no message text, no user content (spec Further Notes).
    """

    def __init__(self, *, thresholds: dict[str, float] | None = None) -> None:
        self._lock = threading.RLock()
        self._counts: dict[str, int] = {name: 0 for name in BOUNDARY_METRIC_NAMES}
        self._last_seen: dict[str, str] = {}
        self._repair_age_max_seconds = 0.0
        self._repair_age_last_seconds = 0.0
        self._repair_age_last_run_id = ""
        self.alert_thresholds: dict[str, float] = dict(
            thresholds or BOUNDARY_ALERT_THRESHOLDS
        )
        self._monitor = MigrationThresholdMonitor()

    def observe(
        self,
        name: str,
        *,
        count: int = 1,
    ) -> int:
        """Add ``count`` observations of one named boundary metric."""
        # ``ExecutionErrorCode`` is a str-Enum: accept the member or its
        # value so emission sites never depend on which one they were given.
        key = str(getattr(name, "value", name))
        if key not in self._counts:
            raise ValueError(f"unknown boundary metric: {name!r}")
        if count < 1:
            raise ValueError("count must be positive")
        with self._lock:
            self._counts[key] += count
            self._last_seen[key] = _now_iso()
            return self._counts[key]

    def observe_repair_age(self, age_seconds: float, *, run_id: str = "") -> float:
        """Record how long an open unknown-outcome repair has been open."""
        with self._lock:
            age = max(0.0, float(age_seconds))
            self._counts["repair_age"] += 1
            self._last_seen["repair_age"] = _now_iso()
            self._repair_age_last_seconds = age
            self._repair_age_last_run_id = run_id
            if age > self._repair_age_max_seconds:
                self._repair_age_max_seconds = age
            return self._repair_age_max_seconds

    def snapshot(self) -> dict[str, Any]:
        """The health/metrics payload: counters, gauges, thresholds, alerts."""
        with self._lock:
            payload: dict[str, Any] = dict(self._counts)
            payload["repair_age_max_seconds"] = round(
                self._repair_age_max_seconds, 3
            )
            payload["repair_age_last_seconds"] = round(
                self._repair_age_last_seconds, 3
            )
            payload["repair_age_last_run_id"] = self._repair_age_last_run_id
            payload["last_seen_at"] = dict(self._last_seen)
            payload["alert_thresholds"] = dict(self.alert_thresholds)
            return payload

    def alerting_metrics(self) -> dict[str, float]:
        """The value set fed to :class:`MigrationThresholdMonitor`.

        ``repair_age`` alerts on the AGE (seconds) of the oldest open
        repair, not on how many times a snapshot was read.
        """
        with self._lock:
            values = {name: float(self._counts[name]) for name in BOUNDARY_METRIC_NAMES}
        values["repair_age"] = self._repair_age_max_seconds
        return values

    def alerts(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        """Sustained boundary breaches (empty while everything is healthy)."""
        return self._monitor.observe(self.alerting_metrics(), now=now)

    def reset(self) -> None:
        with self._lock:
            self._counts = {name: 0 for name in BOUNDARY_METRIC_NAMES}
            self._last_seen.clear()
            self._repair_age_max_seconds = 0.0
            self._repair_age_last_seconds = 0.0
            self._repair_age_last_run_id = ""
            self._monitor = MigrationThresholdMonitor()


_boundary_metrics = BoundaryMetrics()


def get_boundary_metrics() -> BoundaryMetrics:
    """The process-wide boundary metric registry (emission target)."""
    return _boundary_metrics


def reset_boundary_metrics() -> BoundaryMetrics:
    """Zero the process-wide counters (tests, and a deliberate re-rollout)."""
    _boundary_metrics.reset()
    return _boundary_metrics


def observe_boundary_metric(name: str, *, count: int = 1) -> int:
    """Emit one boundary-metric observation; NEVER raises.

    Every call site here sits on a durable write path (admission, effect
    settle, terminal fence, lease CAS, janitor enqueue, replay, snapshot).
    Observability must not turn a successful write into a failed request,
    so a registry problem is logged and swallowed.
    """
    try:
        return _boundary_metrics.observe(name, count=count)
    except Exception:  # pragma: no cover - defensive
        logger.warning("boundary metrics: observation failed for %r", name)
        return 0


def observe_boundary_repair_age(age_seconds: float, *, run_id: str = "") -> float:
    """Emit one open-repair age observation; NEVER raises."""
    try:
        return _boundary_metrics.observe_repair_age(age_seconds, run_id=run_id)
    except Exception:  # pragma: no cover - defensive
        logger.warning("boundary metrics: repair age observation failed")
        return 0.0


def boundary_metrics_snapshot() -> dict[str, Any]:
    """The seven counters + gauges for the health surfaces."""
    return _boundary_metrics.snapshot()


__all__ = [
    "BOUNDARY_ALERT_THRESHOLDS",
    "BOUNDARY_METRIC_NAMES",
    "BoundaryMetrics",
    "CapacityExceeded",
    "CapacityGuard",
    "MigrationThresholdMonitor",
    "TelemetryStore",
    "TelemetryService",
    "boundary_metrics_snapshot",
    "get_boundary_metrics",
    "observe_boundary_metric",
    "observe_boundary_repair_age",
    "reset_boundary_metrics",
]


TelemetryService = TelemetryStore
