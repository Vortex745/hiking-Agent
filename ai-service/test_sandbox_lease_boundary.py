"""BCDEF ticket 06 — sandbox active lease, fencing token and resume gate.

Seam: the ``sandbox_leases`` persistence CAS (through ``RunRepository``),
the :class:`agent_v2.leases.SandboxLeaseGuard`, the sandbox execution
bridge, and the orchestration resume path. Real SQLite under a tmp dir;
the computer is a counting fake (no real sandbox, no network).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from agent_v2.approvals import compute_approval_id, compute_expiry
from agent_v2.execution_contract import policy_hash
from agent_v2.execution_policy import build_execution_policy
from agent_v2.leases import (
    SandboxLeaseGuard,
    SandboxLeaseLostError,
    lease_is_expired,
)
from agent_v2.persistence import (
    CheckpointsStore,
    ControlStore,
    PolicySnapshotWrite,
)
from agent_v2.profiles import resolve_profile
from agent_v2.run_orchestration import RunOrchestrator
from agent_v2.runs import RunRepository
from agent_v2.sandbox_lifecycle import run_sandbox_cleanup


def _repo(tmp_path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def _make_run(repo: RunRepository, request_id: str = "req-lease") -> str:
    run_id, _ = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=f"conv-{request_id}",
        profile_id="cowork",
        user_message="m",
    )
    return run_id


def _past_iso(seconds: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def _expire_by_sql(repo: RunRepository, run_id: str) -> None:
    repo.control._conn.execute(
        "UPDATE sandbox_leases SET expires_at = ? WHERE run_id = ?",
        (_past_iso(), run_id),
    )


# ---- acquire / renew / release / expire CAS ---------------------------------


def test_acquire_creates_one_active_lease_with_fencing_metadata(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo)
        lease, acquired = repo.acquire_sandbox_lease(
            run_id=run_id,
            owner_id="worker-1",
            sandbox_type="local_native",
            workspace_path="/ws/run-lease",
            sandbox_identity="local_native:/ws/run-lease",
        )
        assert acquired is True
        assert lease is not None
        assert lease.status == "active"
        assert lease.owner_id == "worker-1"
        assert lease.fencing_token == 1
        assert lease.sandbox_identity == "local_native:/ws/run-lease"
        assert lease.expires_at is not None
        assert not lease_is_expired(lease.expires_at)
        assert lease.heartbeat_at is not None

        # Duplicate acquire: same generation, same row, no token bump.
        again, acquired_again = repo.acquire_sandbox_lease(
            run_id=run_id,
            owner_id="worker-2",
            sandbox_type="local_native",
            workspace_path="/ws/other",
        )
        assert acquired_again is False
        assert again is not None
        assert again.lease_id == lease.lease_id
        assert again.fencing_token == 1
        assert again.workspace_path == "/ws/run-lease"
        rows = repo.control._conn.execute(
            "SELECT COUNT(*) FROM sandbox_leases WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert rows == 1
    finally:
        repo.close()


def test_concurrent_acquire_yields_one_active_lease(tmp_path):
    shared = tmp_path / "control.sqlite"
    seeder = RunRepository(
        control=ControlStore(str(shared)),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )
    run_id = _make_run(seeder, "req-lease-race")
    seeder.close()

    stores = [ControlStore(str(shared)), ControlStore(str(shared))]
    results: list[tuple[bool, str, int]] = []
    barrier = threading.Barrier(2)

    def _acquire(store: ControlStore, owner: str) -> None:
        barrier.wait()
        lease, acquired = store.acquire_sandbox_lease(
            run_id=run_id,
            owner_id=owner,
            sandbox_type="local_native",
            workspace_path=f"/ws/{owner}",
        )
        results.append((acquired, lease.lease_id if lease else "", lease.fencing_token if lease else -1))

    threads = [
        threading.Thread(target=_acquire, args=(stores[0], "worker-a")),
        threading.Thread(target=_acquire, args=(stores[1], "worker-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert len(results) == 2
    assert sum(1 for acquired, _lid, _tok in results if acquired) == 1
    lease_ids = {lease_id for _a, lease_id, _t in results}
    assert len(lease_ids) == 1  # one active lease row/generation
    conn = sqlite3.connect(str(shared))
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM sandbox_leases WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 1
    for store in stores:
        store.close()


def test_renew_rejects_stale_token_and_expired_lease(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo)
        lease, _ = repo.acquire_sandbox_lease(
            run_id=run_id,
            owner_id="worker-1",
            sandbox_type="local_native",
            workspace_path="/ws",
        )
        assert lease is not None and lease.fencing_token == 1

        renewed, ok = repo.renew_sandbox_lease(
            run_id=run_id, fencing_token=1, owner_id="worker-1b"
        )
        assert ok is True and renewed is not None
        assert renewed.owner_id == "worker-1b"

        stale, stale_ok = repo.renew_sandbox_lease(
            run_id=run_id, fencing_token=99, owner_id="stale"
        )
        assert stale_ok is False
        assert stale is not None and stale.fencing_token == 1
        assert stale.owner_id == "worker-1b"

        # Expired leases cannot be renewed by the old worker.
        _expire_by_sql(repo, run_id)
        expired, expired_ok = repo.renew_sandbox_lease(
            run_id=run_id, fencing_token=1, owner_id="worker-1"
        )
        assert expired_ok is False
        assert expired is not None and expired.is_expired()

        marked = repo.expire_sandbox_lease(run_id)
        assert marked is not None and marked.status == "expired"
        # A NEW sandbox may take over an explicitly expired/released row,
        # but only with a higher fencing generation (old worker fenced).
        takeover, acquired = repo.acquire_sandbox_lease(
            run_id=run_id,
            owner_id="worker-2",
            sandbox_type="local_native",
            workspace_path="/ws",
        )
        assert acquired is True
        assert takeover is not None and takeover.status == "active"
        assert takeover.fencing_token == 2
    finally:
        repo.close()


def test_release_is_idempotent_and_fenced(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo)
        lease, _ = repo.acquire_sandbox_lease(
            run_id=run_id,
            owner_id="worker-1",
            sandbox_type="local_native",
            workspace_path="/ws",
        )
        assert lease is not None

        released, ok = repo.release_sandbox_lease(
            run_id=run_id, fencing_token=1, resource_usage={"files_deleted": 2}
        )
        assert ok is True and released is not None
        assert released.status == "destroyed"
        assert released.resource_usage == {"files_deleted": 2}
        destroyed_at = released.destroyed_at

        # Idempotent: a second release does not mutate the evidence row.
        again, ok_again = repo.release_sandbox_lease(
            run_id=run_id, failure_reason="late cleanup failure"
        )
        assert ok_again is False
        assert again is not None
        assert again.status == "destroyed"
        assert again.destroyed_at == destroyed_at
        assert again.failure_reason is None

        # Re-acquire after release bumps the fencing token (monotonic).
        gen2, acquired = repo.acquire_sandbox_lease(
            run_id=run_id,
            owner_id="worker-2",
            sandbox_type="local_native",
            workspace_path="/ws",
        )
        assert acquired is True
        assert gen2 is not None and gen2.fencing_token == 2

        # A stale worker cannot release the newer generation.
        stale, stale_ok = repo.release_sandbox_lease(
            run_id=run_id, fencing_token=1
        )
        assert stale_ok is False
        assert stale is not None and stale.status == "active"
        assert stale.fencing_token == 2
    finally:
        repo.close()


def test_guard_blocks_expired_and_stale_fences(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo)
        lease, _ = repo.acquire_sandbox_lease(
            run_id=run_id,
            owner_id="worker-1",
            sandbox_type="local_native",
            workspace_path="/ws",
        )
        assert lease is not None and lease.fencing_token == 1

        guard = SandboxLeaseGuard(
            repo, run_id=run_id, fencing_token=1, lease_id=lease.lease_id
        )
        assert guard.assert_active() is not None
        assert guard.renew() is not None

        stale_guard = SandboxLeaseGuard(
            repo, run_id=run_id, fencing_token=99, lease_id=lease.lease_id
        )
        with pytest.raises(SandboxLeaseLostError) as excinfo:
            stale_guard.assert_active()
        assert excinfo.value.code == "lease_fencing_rejected"

        _expire_by_sql(repo, run_id)
        with pytest.raises(SandboxLeaseLostError) as excinfo:
            guard.assert_active()
        assert excinfo.value.code == "lease_expired"

        repo.release_sandbox_lease(run_id=run_id, fencing_token=1)
        with pytest.raises(SandboxLeaseLostError) as excinfo:
            guard.assert_active()
        assert excinfo.value.code == "sandbox_lease_lost"
    finally:
        repo.close()


def test_pre_ticket_schema_rows_stay_readable(tmp_path):
    """Old effects/sandbox_leases tables get additive columns on open."""
    db_path = tmp_path / "old-control.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            request_id TEXT NOT NULL UNIQUE,
            conversation_id TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            status TEXT NOT NULL,
            terminal_type TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            terminal_at TEXT
        );
        CREATE TABLE effects (
            effect_key TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            args_canonical TEXT NOT NULL,
            status TEXT NOT NULL,
            external_idempotency_key TEXT NOT NULL,
            result_json TEXT,
            failure_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            committed_at TEXT,
            UNIQUE (run_id, tool_name, args_canonical)
        );
        CREATE TABLE sandbox_leases (
            lease_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL UNIQUE,
            sandbox_type TEXT NOT NULL,
            status TEXT NOT NULL,
            workspace_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            destroyed_at TEXT,
            resource_usage_json TEXT,
            failure_reason TEXT
        );
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, 'running', NULL, ?, ?, NULL)",
        ("run-old", "req-old", "conv-old", "cowork", now, now),
    )
    conn.execute(
        "INSERT INTO effects VALUES (?, ?, ?, ?, ?, 'authorized', ?, NULL, "
        "NULL, ?, ?, NULL)",
        ("effect-old", "run-old", "conv-old", "sandbox_bash", '{"command":"x"}',
         "ext-old", now, now),
    )
    conn.execute(
        "INSERT INTO sandbox_leases VALUES (?, ?, ?, 'active', ?, ?, NULL, "
        "NULL, NULL)",
        ("lease-old", "run-old", "local_native", "/ws/old", now),
    )
    conn.commit()
    conn.close()

    store = ControlStore(str(db_path))
    try:
        effect = store.get_effect("effect-old")
        assert effect is not None
        assert effect.status == "authorized"
        assert effect.phase == "pending"
        assert effect.resolved_effect_id == "effect-old"
        assert effect.resolved_idempotency_key == "ext-old"
        assert effect.fencing_token == 0

        lease = store.get_sandbox_lease("run-old")
        assert lease is not None
        assert lease.status == "active"
        # BCDEF review B1: the stored pre-ticket-06 default of 0 is not a
        # fenceable generation. The additive migration normalizes legacy
        # lease rows to generation 1 so an unupgraded worker (which can
        # only ever hold 0) is fenced instead of waved through.
        assert lease.fencing_token == 1
        assert lease.expires_at is None
        assert lease.is_expired() is False
        # Legacy lease renews under the normalized generation and gains an
        # expiry.
        renewed, ok = store.renew_sandbox_lease(
            run_id="run-old", fencing_token=1, owner_id="worker"
        )
        assert ok is True and renewed is not None
        assert renewed.expires_at is not None
    finally:
        store.close()


_LEGACY_LEASE_DDL = """
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    conversation_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    status TEXT NOT NULL,
    terminal_type TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    terminal_at TEXT
);
CREATE TABLE sandbox_leases (
    lease_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    sandbox_type TEXT NOT NULL,
    status TEXT NOT NULL,
    workspace_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    destroyed_at TEXT,
    resource_usage_json TEXT,
    failure_reason TEXT
);
"""


def _seed_legacy_lease_db(db_path) -> None:
    """A pre-ticket-06 database: one active lease, no fencing column."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_LEGACY_LEASE_DDL)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, 'running', NULL, ?, ?, NULL)",
        ("run-legacy", "req-legacy", "conv-legacy", "cowork", now, now),
    )
    conn.execute(
        "INSERT INTO sandbox_leases "
        "(lease_id, run_id, sandbox_type, status, workspace_path, created_at) "
        "VALUES ('lease-legacy', 'run-legacy', 'local_native', 'active', "
        "'/ws/legacy', ?)",
        (now,),
    )
    conn.commit()
    conn.close()


def test_legacy_generation_zero_lease_cannot_execute_after_takeover(tmp_path):
    """BCDEF review B1: ``fencing_token == 0`` is a real token, not "unset".

    A worker that read the lease before the upgrade holds generation 0. Once
    a newer generation owns the lease, its guard must be refused — not
    silently waved through because ``0`` is falsy.
    """
    db_path = tmp_path / "legacy-fencing.sqlite"
    _seed_legacy_lease_db(db_path)
    repo = RunRepository(
        control=ControlStore(str(db_path)),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )
    try:
        lease = repo.get_sandbox_lease("run-legacy")
        assert lease is not None
        assert lease.fencing_token == 1  # normalized by the migration

        # The unupgraded worker still holds generation 0.
        zero_guard = SandboxLeaseGuard(
            repo, run_id="run-legacy", fencing_token=0, lease_id=lease.lease_id
        )
        with pytest.raises(SandboxLeaseLostError) as excinfo:
            zero_guard.assert_active()
        assert excinfo.value.code == "lease_fencing_rejected"

        # The current generation runs.
        current_guard = SandboxLeaseGuard(
            repo, run_id="run-legacy", fencing_token=1, lease_id=lease.lease_id
        )
        assert current_guard.assert_active() is not None

        # Expire + takeover: generation 2 owns the lease, so BOTH stale
        # generations are fenced.
        _expire_by_sql(repo, "run-legacy")
        repo.expire_sandbox_lease("run-legacy")
        takeover, acquired = repo.acquire_sandbox_lease(
            run_id="run-legacy",
            owner_id="worker-2",
            sandbox_type="local_native",
            workspace_path="/ws/takeover",
        )
        assert acquired is True
        assert takeover is not None and takeover.fencing_token == 2

        for guard in (zero_guard, current_guard):
            with pytest.raises(SandboxLeaseLostError) as excinfo:
                guard.assert_active()
            assert excinfo.value.code == "lease_fencing_rejected"

        # A stale generation can never revive its lease either.
        with pytest.raises(SandboxLeaseLostError) as excinfo:
            current_guard.renew()
        assert excinfo.value.code == "lease_fencing_rejected"
    finally:
        repo.close()


# ---- sandbox execution bridge ------------------------------------------------


def _bridge(computer: Any, guard: Any):
    from agent_v2.kernel_tools import ProductSandboxBridge

    return ProductSandboxBridge(computer, "/ws", lease_guard=guard)


class _CountingComputer:
    sandbox_type = "local_native"

    def __init__(self) -> None:
        self.commands: list[str] = []

    def run_bash(self, workspace, command, **_kwargs):
        self.commands.append(command)
        return SimpleNamespace(
            exit_code=0, stdout="ok", stderr="", duration_ms=1, truncated=False
        )


def test_bridge_refuses_execution_without_active_lease(tmp_path):
    from uniharness.exceptions import CLIError

    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo)
        computer = _CountingComputer()
        guard = SandboxLeaseGuard(repo, run_id=run_id, fencing_token=1)
        bridge = _bridge(computer, guard)

        async def scenario():
            with pytest.raises(CLIError) as missing:
                await bridge.run("echo hi")
            assert "sandbox_lease_lost" in str(missing.value)
            assert computer.commands == []

            lease, _ = repo.acquire_sandbox_lease(
                run_id=run_id,
                owner_id="worker-1",
                sandbox_type="local_native",
                workspace_path="/ws",
            )
            assert lease is not None
            await bridge.run("echo hi")
            assert computer.commands == ["echo hi"]

            stale = SandboxLeaseGuard(repo, run_id=run_id, fencing_token=99)
            stale_bridge = _bridge(computer, stale)
            with pytest.raises(CLIError) as fenced:
                await stale_bridge.run("echo again")
            assert "lease_fencing_rejected" in str(fenced.value)
            assert computer.commands == ["echo hi"]

        asyncio.run(scenario())
    finally:
        repo.close()


# ---- cowork bash tool lease gate ----------------------------------------------


def test_cowork_bash_tool_refuses_without_active_lease(tmp_path):
    from agent_v2.effects import compute_effect_identity
    from agent_v2.kernel_tools import CoworkBashTool

    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo)
        identity = compute_effect_identity(
            run_id=run_id, tool_name="sandbox_bash", args={"command": "echo hi"}
        )
        repo.insert_effect(
            effect_key=identity.effect_id,
            run_id=run_id,
            conversation_id="conv-lease",
            tool_name="sandbox_bash",
            args_canonical='{"command": "echo hi"}',
            external_idempotency_key=identity.idempotency_key,
        )
        computer = _CountingComputer()

        class _Bridge:
            def __init__(self) -> None:
                self.commands: list[str] = []

            async def run(self, command: str, **_kwargs):
                self.commands.append(command)
                return SimpleNamespace(
                    exit_code=0, stdout="hi", stderr="", duration_ms=1, truncated=False
                )

        bridge = _Bridge()
        guard = SandboxLeaseGuard(repo, run_id=run_id, fencing_token=1)
        tool = CoworkBashTool(
            bridge=bridge,
            run_id=run_id,
            effects=repo,
            audit_sink=lambda _entry: None,
            lease_guard=guard,
        )

        result = asyncio.run(tool(command="echo hi"))

        assert result.error and "lease" in result.error
        assert bridge.commands == []  # no sandbox side effect
        row = repo.get_effect(identity.effect_id)
        assert row is not None and row.status == "authorized"  # never claimed
    finally:
        repo.close()


# ---- orchestration first-stream lease gate ------------------------------------


def test_cowork_first_stream_fails_closed_when_lease_cannot_be_acquired(
    tmp_path, monkeypatch
):
    from agent_v2.run_orchestration import ChatFlowRequest

    repo = _repo(tmp_path)

    def boom(**_kwargs):
        raise sqlite3.OperationalError("lease store unavailable")

    monkeypatch.setattr(repo, "acquire_sandbox_lease", boom)
    computer = _FakeComputer()
    cleanups: list[Any] = []

    async def fake_kernel(**_kwargs):
        return ('{"command": "echo hi"}', [])

    def cleanup(computer_arg, workspace_path, runs_repo_arg, run_id_arg):
        # ALR ticket 03: explicit typed result via the production helper —
        # the settlement, not the hook, owns the durable writes.
        cleanups.append(workspace_path)
        return run_sandbox_cleanup(computer_arg, workspace_path)

    orch = RunOrchestrator(
        settings=_settings(),
        runs_repo=repo,
        kernel_turn=fake_kernel,
        computer_factory=lambda *a, **k: computer,
        sandbox_cleanup=cleanup,
    )
    try:
        req = ChatFlowRequest(
            message="在沙箱里创建一个文件",
            request_id="req-lease-broken",
            profile_id="cowork",
        )

        async def run() -> list[Any]:
            return [event async for event in orch.stream_chat(req)]

        items = asyncio.run(run())

        types = [event.type for event in items]
        assert types[-1] == "error"
        assert items[-1].payload["code"] == "sandbox_lease_unavailable"
        assert "approval.required" not in types  # never reached the gate
        assert computer.bash_calls == []
        assert len(cleanups) == 1  # workspace cleaned, no leaked sandbox
        assert repo.get_sandbox_lease(repo.control.get_run_by_request("req-lease-broken").run_id) is None
    finally:
        repo.close()


# ---- orchestration resume gate ------------------------------------------------
class _FakeComputer:
    sandbox_type = "local_native"

    def __init__(self) -> None:
        self.workspaces: list[str] = []
        self.bash_calls: list[tuple[str, str]] = []
        self.cleaned: list[str] = []

    def create_workspace(self, run_id: str) -> str:
        self.workspaces.append(run_id)
        return f"/ws/{run_id}"

    def run_bash(self, workspace_path, command, **_kwargs):
        self.bash_calls.append((workspace_path, command))
        return SimpleNamespace(
            exit_code=0, stdout="hi", stderr="", duration_ms=1, truncated=False
        )

    def cleanup(self, workspace_path: str) -> dict[str, int]:
        self.cleaned.append(workspace_path)
        return {"files_deleted": 0}


def _settings(**overrides: Any) -> Any:
    base = {
        "agent_v2_enabled": True,
        "openai_api_key": "sk-test",
        "openai_model": "m",
        "agent_v2_cowork_policy": "trusted",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _seed_frozen_policy(repo: RunRepository, run_id: str, profile_id: str = "cowork") -> None:
    """Ticket 09: a resumable run needs its admission-frozen policy snapshot."""
    policy = build_execution_policy(resolve_profile(profile_id), _settings())
    write = PolicySnapshotWrite(
        policy_json=json.dumps(policy, sort_keys=True, separators=(",", ":")),
        policy_hash=policy_hash(policy),
        schema_version=policy["schema_version"],
        source_version=policy["source_version"],
        compatibility_json=json.dumps(policy["compatibility"]),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    repo.control.insert_policy_snapshot(run_id=run_id, policy=write)


def _paused_cowork_state(repo: RunRepository, request_id: str, command: str = "echo hi") -> tuple[str, str]:
    run_id = _make_run(repo, request_id)
    _seed_frozen_policy(repo, run_id)
    repo.acquire_sandbox_lease(
        run_id=run_id,
        owner_id="first-stream",
        sandbox_type="local_native",
        workspace_path=f"/ws/{run_id}",
    )
    approval_id = compute_approval_id(
        run_id=run_id, tool_name="sandbox_bash", args={"command": command}
    )
    repo.insert_approval(
        approval_id=approval_id,
        run_id=run_id,
        conversation_id=f"conv-{request_id}",
        tool_name="sandbox_bash",
        args_canonical=f'{{"command": "{command}"}}',
        risk_level="high",
        impact="sandbox bash",
        created_at=datetime.now(timezone.utc).isoformat(),
        expires_at=compute_expiry(datetime.now(timezone.utc).isoformat()),
    )
    repo.resolve_approval(
        approval_id=approval_id, status="approved", decision_request_id="dec-1"
    )
    return run_id, approval_id


def _resume_orchestrator(repo: RunRepository, computer: _FakeComputer):
    def fake_export(**_kwargs: Any) -> dict[str, Any]:
        return {"artifact_id": "art-lease", "public_filename": "out.txt"}

    def cleanup(comp: Any, workspace_path: str, runs: Any, run_id: Any):
        # ALR ticket 03: explicit typed result via the production helper —
        # the lifecycle settlement destroys the lease (the hook does not).
        return run_sandbox_cleanup(comp, workspace_path)

    return RunOrchestrator(
        settings=_settings(),
        runs_repo=repo,
        artifact_export=fake_export,
        computer_factory=lambda *a, **k: computer,
        sandbox_cleanup=cleanup,
    )


def _resume(orch: RunOrchestrator, run_id: str, from_seq: int = 0) -> list[Any]:
    async def run() -> list[Any]:
        return [event async for event in orch.stream_resume(run_id=run_id, from_seq=from_seq)]

    return asyncio.run(run())


def test_resume_with_expired_lease_fails_closed_without_execution(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id, _approval = _paused_cowork_state(repo, "req-lease-expired")
        _expire_by_sql(repo, run_id)
        computer = _FakeComputer()
        orch = _resume_orchestrator(repo, computer)

        items = _resume(orch, run_id)

        assert items[-1].type == "error"
        assert items[-1].payload["code"] == "lease_expired"
        assert computer.bash_calls == []  # no sandbox side effect
        lease = repo.get_sandbox_lease(run_id)
        assert lease is not None and lease.status != "active"
        assert lease.fencing_token == 1  # no new generation created
        run = repo.get_run(run_id)
        assert run is not None and run.status == "error"
    finally:
        repo.close()


def test_resume_rejects_non_continuable_run_status(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id, _approval = _paused_cowork_state(repo, "req-lease-status")
        repo.control.update_run_status(run_id=run_id, status="admitted")
        computer = _FakeComputer()
        orch = _resume_orchestrator(repo, computer)

        items = _resume(orch, run_id)

        assert items[-1].type == "error"
        assert items[-1].payload["code"] == "run_not_resumable"
        assert computer.bash_calls == []
    finally:
        repo.close()


def test_resume_renews_lease_and_links_effect_claim_to_fencing_token(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id, _approval = _paused_cowork_state(repo, "req-lease-link")
        computer = _FakeComputer()
        orch = _resume_orchestrator(repo, computer)

        items = _resume(orch, run_id)

        assert [event.type for event in items][-1] == "done"
        assert computer.bash_calls == [(f"/ws/{run_id}", "echo hi")]
        lease = repo.get_sandbox_lease(run_id)
        assert lease is not None
        # The lease is released by cleanup; its fencing generation and
        # identity survive for replay/audit.
        assert lease.fencing_token == 1
        effect = repo.list_effects_for_run(run_id)[0]
        assert effect.status == "committed"
        assert effect.lease_id == lease.lease_id
        assert effect.fencing_token == 1

        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.sandbox_lease is not None
        assert snapshot.sandbox_lease["fencing_token"] == 1
        assert snapshot.sandbox_lease["lease_id"] == lease.lease_id
        assert snapshot.sandbox_lease["status"] == "destroyed"
    finally:
        repo.close()


def test_duplicate_resume_does_not_create_second_sandbox_or_lease(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id, _approval = _paused_cowork_state(repo, "req-lease-dup")
        computer = _FakeComputer()
        orch = _resume_orchestrator(repo, computer)

        first = _resume(orch, run_id)
        assert first[-1].type == "done"
        second = _resume(orch, run_id, from_seq=1)

        assert [event.type for event in second][-1] == "done"
        assert computer.bash_calls == [(f"/ws/{run_id}", "echo hi")]  # once
        assert computer.workspaces == []  # resume never creates a sandbox
        rows = repo.control._conn.execute(
            "SELECT COUNT(*) FROM sandbox_leases WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert rows == 1
        lease = repo.get_sandbox_lease(run_id)
        assert lease is not None and lease.fencing_token == 1
    finally:
        repo.close()


def test_cleanup_release_is_idempotent_end_to_end(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id, _approval = _paused_cowork_state(repo, "req-lease-clean")
        computer = _FakeComputer()
        orch = _resume_orchestrator(repo, computer)

        _resume(orch, run_id)
        first = repo.get_sandbox_lease(run_id)
        assert first is not None and first.status == "destroyed"
        destroyed_at = first.destroyed_at

        # A second cleanup (retry after a crash) is a no-op.
        repo.destroy_sandbox_lease(run_id=run_id, failure_reason="late retry")
        again = repo.get_sandbox_lease(run_id)
        assert again is not None
        assert again.status == "destroyed"
        assert again.destroyed_at == destroyed_at
        assert again.failure_reason is None
    finally:
        repo.close()


# ---- ALR ticket 04: atomic settlement, fenced success, dedupe ----------------


def _lease_with_run(repo: RunRepository, request_id: str, workspace: str):
    run_id = _make_run(repo, request_id)
    lease, _acquired = repo.acquire_sandbox_lease(
        run_id=run_id,
        owner_id=f"worker-{request_id}",
        sandbox_type="local_native",
        workspace_path=workspace,
        sandbox_identity=f"local_native:{workspace}",
    )
    assert lease is not None and lease.status == "active"
    return run_id, lease


def test_real_cleanup_success_settles_the_original_failed_lease_to_destroyed(
    tmp_path,
):
    """T04-04: a REAL cleanup success after a settled failure transitions
    the ORIGINAL failed lease — same lease id, same workspace, same
    generation — to ``destroyed`` / ``succeeded`` with the real usage and
    the failure reason cleared, all in the one settlement (the fenced
    transition ticket 06's retry builds on)."""
    from agent_v2.sandbox_lifecycle import (
        SandboxCleanupResult,
        settle_terminal_cleanup,
    )

    repo = _repo(tmp_path)
    try:
        run_id, lease = _lease_with_run(repo, "req-04-fenced", "/ws/fenced")

        # The failure settlement: failed lease + pending_janitor + task.
        failure = SandboxCleanupResult.failure(
            "cleanup failed: workspace locked",
            partial_usage={"files_deleted": 2, "bytes_written": 10},
        )
        task_id = settle_terminal_cleanup(
            repo,
            computer=_FakeComputer(),
            workspace_path="/ws/fenced",
            run_id=run_id,
            outcome=failure,
        )
        assert task_id is not None
        failed = repo.get_sandbox_lease(run_id)
        assert failed is not None
        assert failed.status == "failed"
        assert failed.cleanup_status == "pending_janitor"
        assert failed.failure_reason

        # The REAL success for the SAME original target.
        success = SandboxCleanupResult.success(
            {"duration_ms": 5, "files_deleted": 7, "bytes_written": 128}
        )
        settle_terminal_cleanup(
            repo,
            computer=_FakeComputer(),
            workspace_path="/ws/fenced",
            run_id=run_id,
            outcome=success,
        )

        settled = repo.get_sandbox_lease(run_id)
        assert settled is not None
        assert settled.status == "destroyed"
        assert settled.cleanup_status == "succeeded"
        assert settled.cleanup_error is None
        # The failure reason is cleared and the REAL success usage replaced
        # the preserved partial usage.
        assert settled.failure_reason is None
        assert settled.resource_usage == {
            "duration_ms": 5, "files_deleted": 7, "bytes_written": 128
        }
        # The ORIGINAL lease identity survives — no new generation, no
        # forged token, and the janitor task association stands.
        assert settled.lease_id == lease.lease_id
        assert settled.fencing_token == lease.fencing_token
        assert settled.janitor_task_id == task_id
        rows = repo.control._conn.execute(
            "SELECT COUNT(*) FROM sandbox_leases WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert rows == 1
    finally:
        repo.close()


def test_success_settlement_for_a_stale_target_never_touches_the_successor(
    tmp_path,
):
    """T04-04 (mismatch): a success attempt whose identity does NOT match
    the current lease — a successor generation took the row over, or the
    cleaned workspace is not the leased one — must not modify the
    successor lease and must not forge a new generation."""
    from agent_v2.sandbox_lifecycle import (
        SandboxCleanupResult,
        settle_terminal_cleanup,
    )

    repo = _repo(tmp_path)
    try:
        run_id, original = _lease_with_run(repo, "req-04-stale", "/ws/original")

        failure = SandboxCleanupResult.failure("cleanup failed: locked")
        settle_terminal_cleanup(
            repo,
            computer=_FakeComputer(),
            workspace_path="/ws/original",
            run_id=run_id,
            outcome=failure,
        )

        # A successor generation takes the SAME row over (resume/retry).
        successor, acquired = repo.acquire_sandbox_lease(
            run_id=run_id,
            owner_id="worker-successor",
            sandbox_type="local_native",
            workspace_path="/ws/successor",
            sandbox_identity="local_native:/ws/successor",
        )
        assert acquired is True
        assert successor is not None
        assert successor.status == "active"
        assert successor.fencing_token == original.fencing_token + 1
        assert successor.lease_id != original.lease_id
        # The successor inherits the row-level pending_janitor record from
        # the failed generation (it re-records when IT settles) — but it
        # has no failure of its own and no usage evidence.
        assert successor.failure_reason is None
        assert successor.resource_usage is None

        # (a) The fenced storage primitive with the ORIGINAL identity is
        # refused: the successor row is untouched.
        record, settled = repo.settle_cleanup_success(
            run_id=run_id,
            lease_id=original.lease_id,
            workspace_path="/ws/original",
            sandbox_identity=original.sandbox_identity or "",
            fencing_token=original.fencing_token,
            resource_usage={"duration_ms": 1},
        )
        assert settled is False
        assert record is not None
        assert record.status == "active"
        assert record.fencing_token == successor.fencing_token
        assert record.lease_id == successor.lease_id
        assert record.failure_reason is None  # successor has no failure to clear
        assert record.resource_usage is None
        assert record.cleanup_status != "succeeded"

        # (b) The production lifecycle entry refuses a success for a
        # workspace that is not the leased one — same protection from the
        # orchestration side.
        settle_terminal_cleanup(
            repo,
            computer=_FakeComputer(),
            workspace_path="/ws/original",  # the stale, superseded target
            run_id=run_id,
            outcome=SandboxCleanupResult.success({"duration_ms": 1}),
        )
        after = repo.get_sandbox_lease(run_id)
        assert after is not None
        assert after.status == "active"
        assert after.lease_id == successor.lease_id
        assert after.fencing_token == successor.fencing_token
        assert after.cleanup_status != "succeeded"
        assert after.failure_reason is None
        assert after.resource_usage is None
        # No generation was forged and no second row exists.
        rows = repo.control._conn.execute(
            "SELECT COUNT(*) FROM sandbox_leases WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert rows == 1
    finally:
        repo.close()


def test_duplicate_failure_settlement_keeps_exactly_one_active_dedupe_task(
    tmp_path,
):
    """T04-03: settling the SAME failure repeatedly never produces a second
    ACTIVE dedupe task — before and after the task is claimed — and the
    settled lease facts stay stable (first-release evidence preserved)."""
    from agent_v2.sandbox_lifecycle import (
        SandboxCleanupResult,
        settle_terminal_cleanup,
    )

    repo = _repo(tmp_path)
    try:
        run_id, _lease = _lease_with_run(repo, "req-04-dedupe", "/ws/dedupe")
        failure = SandboxCleanupResult.failure(
            "cleanup failed: locked",
            partial_usage={"files_deleted": 1},
        )

        first = settle_terminal_cleanup(
            repo, computer=_FakeComputer(), workspace_path="/ws/dedupe",
            run_id=run_id, outcome=failure,
        )
        assert first is not None
        assert len(repo.list_janitor_tasks(kind="cleanup")) == 1

        # A duplicate settlement of the SAME failure reuses the task.
        second = settle_terminal_cleanup(
            repo, computer=_FakeComputer(), workspace_path="/ws/dedupe",
            run_id=run_id, outcome=failure,
        )
        assert second == first
        assert len(repo.list_janitor_tasks(kind="cleanup")) == 1
        lease_after = repo.get_sandbox_lease(run_id)
        assert lease_after is not None
        assert lease_after.status == "failed"
        assert lease_after.cleanup_status == "pending_janitor"
        assert lease_after.janitor_task_id == first
        assert lease_after.resource_usage == {"files_deleted": 1}

        # While the task is CLAIMED its active dedupe key still collapses
        # duplicates — and the claim fence itself keeps working.
        claimed, won = repo.claim_janitor_task(first, owner="janitor-04")
        assert won is True and claimed is not None
        third = settle_terminal_cleanup(
            repo, computer=_FakeComputer(), workspace_path="/ws/dedupe",
            run_id=run_id, outcome=failure,
        )
        assert third == first
        tasks = repo.list_janitor_tasks(kind="cleanup")
        assert len(tasks) == 1
        assert tasks[0].status == "claimed"

        _record, released = repo.complete_janitor_task(
            first, owner="janitor-04", expected_attempt=claimed.attempts
        )
        assert released is True

        # Exactly one row total; the done row released its dedupe key, so
        # the NEXT round of the same logical work may open a new task.
        active = repo.control._conn.execute(
            "SELECT COUNT(*) FROM janitor_tasks "
            "WHERE kind = 'cleanup' AND dedupe_key IS NOT NULL"
        ).fetchone()[0]
        assert active == 0
        rows = repo.control._conn.execute(
            "SELECT COUNT(*) FROM janitor_tasks WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert rows == 1
    finally:
        repo.close()

