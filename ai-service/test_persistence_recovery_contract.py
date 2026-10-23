"""harness-p0-p1 Ticket 05 — persistence, run-scope isolation & recovery.

Spec coverage: T07 (run isolation + durable turn/checkpoint/audit/effect/lease
state) and T08 (uniform labels + correlatable run_id, non-empty payloads,
run-scoped retrieval/event data, disconnect-safe single terminal,
exactly-once concurrency).

The contracts pinned here, against REAL production code (RunRepository,
ControlStore/CheckpointsStore, KernelCheckpointStore + a real LangGraph
checkpointer, BoundedDomainTool audit trail, the effect ledger CAS, the
single-writer lock, and the vendored UniHarness TaskRegistry):

* two Chat v2 runs in the SAME conversation never share event sequences,
  effect keys, checkpoint threads, audit sinks or task labels;
* turn/checkpoint/audit/effect/lease state persists to SQLite and is
  fully recovered by a fresh set of stores over the same files (restart);
* the real checkpointer writes at least once per turn and preserves
  structured messages; event appends are monotonic per run and never
  duplicated;
* kernel tasks keep their registry label until terminal, the owner is
  notified on completion, and a run's completion/error removes its labels
  (``cancel_all`` on agent close) — other runs' labels are untouched;
* done/error/cancelled events always carry the correlatable ``run_id``;
* concurrent same-key writes produce exactly one row, exactly one tool
  execution and exactly one external effect; a second writer lock holder
  and a failed renewal fail closed; forged cross-run labels cannot borrow
  another run's authorization and result swaps are refused;
* a client disconnect followed by resume keeps exactly one terminal.

Run (ai-service working directory):
    E:/miniconda3/python.exe -B -m pytest -q test_persistence_recovery_contract.py
"""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

from agent_v2.effects import EffectStatus, compute_effect_key, serialize_result
from agent_v2.kernel_tools import CoworkBashTool
from agent_v2.kernel_persistence import KernelCheckpointStore
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.runs import RunRepository
from agent_v2.sandbox_lifecycle import run_sandbox_cleanup
from agent_v2.uniharness.kernel_identity import (  # noqa: F401  (import side effects)
    VENDORED_UPSTREAM_REVISION,
)

TERMINALS = {"error", "done", "cancelled"}


# ---- helpers -------------------------------------------------------------------


def _repo(tmp_path: Path, *, suffix: str = "") -> RunRepository:
    tag = suffix or "shared"
    return RunRepository(
        control=ControlStore(str(tmp_path / f"control-{tag}.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / f"ckpt-{tag}.sqlite")),
    )


def _make_run(repo: RunRepository, request_id: str, *, conversation_id: str, profile: str = "chat"):
    run_id, _created = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=conversation_id,
        profile_id=profile,
        user_message=f"message-for-{request_id}",
    )
    return run_id


def _thread_rows(db_path: Path, thread_id: str) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread_id,)
        ).fetchone()[0]
    finally:
        conn.close()


# ---- Part 1: run-scope isolation (AC1) -----------------------------------------


def test_two_runs_same_conversation_keep_independent_event_sequences(tmp_path: Path):
    repo = _repo(tmp_path)
    try:
        run_a = _make_run(repo, "req-iso-a", conversation_id="conv-shared")
        run_b = _make_run(repo, "req-iso-b", conversation_id="conv-shared")
        assert run_a != run_b  # same conversation, distinct runs

        for index in range(1, 4):
            repo.persist_event(
                run_id=run_a, sequence=index, event_type="text",
                payload={"run": "A", "index": index},
                request_id="req-iso-a", conversation_id="conv-shared",
            )
            repo.persist_event(
                run_id=run_b, sequence=index, event_type="text",
                payload={"run": "B", "index": index},
                request_id="req-iso-b", conversation_id="conv-shared",
            )

        events_a = repo.get_events_since(run_id=run_a, from_seq=0)
        events_b = repo.get_events_since(run_id=run_b, from_seq=0)
        assert [e["sequence"] for e in events_a] == [1, 2, 3]
        assert [e["sequence"] for e in events_b] == [1, 2, 3]
        # No cross-run leakage: every event carries its own run and payload.
        assert all(e["run_id"] == run_a for e in events_a)
        assert all(e["run_id"] == run_b for e in events_b)
        assert all(e["payload"]["run"] == "A" for e in events_a)
        assert all(e["payload"]["run"] == "B" for e in events_b)
        assert all(e["payload"] for e in events_a + events_b)  # non-empty payloads
    finally:
        repo.close()


def test_run_level_effect_keys_do_not_collide_across_runs(tmp_path: Path):
    repo = _repo(tmp_path)
    try:
        run_a = _make_run(repo, "req-eff-a", conversation_id="conv-shared")
        run_b = _make_run(repo, "req-eff-b", conversation_id="conv-shared")
        args = {"command": "echo shared-command"}
        args_canonical = '{"command": "echo shared-command"}'

        key_a = compute_effect_key(run_id=run_a, tool_name="sandbox_bash", args=args)
        key_b = compute_effect_key(run_id=run_b, tool_name="sandbox_bash", args=args)
        assert key_a != key_b  # run-scoped keys never collide

        for run_id, key in ((run_a, key_a), (run_b, key_b)):
            repo.insert_effect(
                effect_key=key,
                run_id=run_id,
                conversation_id="conv-shared",
                tool_name="sandbox_bash",
                args_canonical=args_canonical,
                external_idempotency_key=f"ext-{key}",
            )

        # Each run retrieves exactly its own authorization.
        found_a = repo.get_effect_by_tool_call(
            run_id=run_a, tool_name="sandbox_bash", args_canonical=args_canonical
        )
        found_b = repo.get_effect_by_tool_call(
            run_id=run_b, tool_name="sandbox_bash", args_canonical=args_canonical
        )
        assert found_a is not None and found_a.effect_key == key_a
        assert found_b is not None and found_b.effect_key == key_b
        assert repo.list_effects_for_run(run_a) == [found_a]
        assert repo.list_effects_for_run(run_b) == [found_b]
    finally:
        repo.close()


def test_checkpoint_threads_isolated_with_real_graph_writes(tmp_path: Path):
    """Real LangGraph checkpointer: two run threads in one DB never mix state."""
    db = tmp_path / "kernel_checkpoints.sqlite"
    store = KernelCheckpointStore(db)

    async def scenario() -> None:
        saver = await store.get()
        graph = _tiny_graph().compile(checkpointer=saver)

        for thread_id, marker in (("v2:chat:run-A", "alpha"), ("v2:chat:run-B", "beta")):
            config = {"configurable": {"thread_id": thread_id}}
            await graph.ainvoke({"messages": [("user", f"turn-1-{marker}")]}, config=config)
            await graph.ainvoke({"messages": [("user", f"turn-2-{marker}")]}, config=config)

        # Structured messages round-trip per thread — no cross-thread bleed.
        for thread_id, marker in (("v2:chat:run-A", "alpha"), ("v2:chat:run-B", "beta")):
            state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
            contents = [str(getattr(m, "content", m)) for m in state.values["messages"]]
            joined = "\n".join(contents)
            assert f"turn-1-{marker}" in joined and f"turn-2-{marker}" in joined
            other = "beta" if marker == "alpha" else "alpha"
            assert other not in joined
            # Every message stays a structured message object (not a blob).
            assert all(hasattr(m, "content") for m in state.values["messages"])

    asyncio.run(scenario())
    store.close()

    assert _thread_rows(db, "v2:chat:run-A") >= 2  # at least one write per turn
    assert _thread_rows(db, "v2:chat:run-B") >= 2
    assert _thread_rows(db, "v2:chat:run-unknown") == 0


def _tiny_graph() -> Any:
    """A minimal real LangGraph graph: echoes an AI message per turn."""
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import MessagesState

    def _record(state: MessagesState) -> MessagesState:
        last = state["messages"][-1]
        return {"messages": [AIMessage(content=f"ack:{getattr(last, 'content', last)}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("record", _record)
    builder.set_entry_point("record")
    builder.add_edge("record", END)
    return builder


# ---- Part 2: run-scoped real code + restart recovery (AC2) ---------------------


# ---- Part 2: run-scoped real code + restart recovery (AC2) ---------------------


def test_audit_entries_are_run_scoped_with_uniform_label_and_payload(tmp_path: Path):
    """Real tool executions record one uniform label + correlatable run_id."""
    repo = _repo(tmp_path)
    try:
        run_a = _make_run(repo, "req-audit-a", conversation_id="conv-audit")
        run_b = _make_run(repo, "req-audit-b", conversation_id="conv-audit")

        class _Bridge:
            def __init__(self) -> None:
                self.commands: list[str] = []

            async def run(self, command: str, *, timeout: float | None = None):
                self.commands.append(command)
                return SimpleNamespace(
                    exit_code=0, stdout="out", stderr="", duration_ms=3, truncated=False
                )

        bridge = _Bridge()
        sink: list[dict[str, Any]] = []
        args = {"command": "echo audit-probe"}
        key = compute_effect_key(run_id=run_a, tool_name="sandbox_bash", args=args)
        repo.insert_effect(
            effect_key=key,
            run_id=run_a,
            conversation_id="conv-audit",
            tool_name="sandbox_bash",
            args_canonical='{"command": "echo audit-probe"}',
            external_idempotency_key="ext-audit",
        )
        tool = CoworkBashTool(bridge=bridge, run_id=run_a, effects=repo, audit_sink=sink.append)

        result = asyncio.run(tool(command="echo audit-probe"))
        assert result.output == "out"
        assert bridge.commands == ["echo audit-probe"]

        assert sink, "audit sink received nothing"
        for entry in sink:
            # Uniform label + correlatable run identity + non-empty payload.
            assert entry["tool"] == "sandbox_bash"
            assert entry["run_id"] == run_a
            assert entry.get("phase") in {"result", "denied"}
            assert entry.get("ok") is True
            assert entry.get("duration_ms") is not None
            assert run_b not in str(entry)

        # A tool bound to the other run never attributes entries to run_a.
        other_sink: list[dict[str, Any]] = []
        tool_b = CoworkBashTool(bridge=bridge, run_id=run_b, effects=repo, audit_sink=other_sink.append)
        denied = asyncio.run(tool_b(command="echo audit-probe"))
        assert denied.error  # no authorization for run_b — fail closed
        assert other_sink and all(e["run_id"] == run_b for e in other_sink)
        assert all(e["run_id"] != run_a for e in other_sink)
    finally:
        repo.close()


def test_turn_state_survives_simulated_restart(tmp_path: Path):
    """Turn/effect/lease/event state persists and recovers on fresh stores."""
    control = tmp_path / "control.sqlite"
    ckpt = tmp_path / "ckpt.sqlite"

    repo1 = RunRepository(
        control=ControlStore(str(control)), checkpoints=CheckpointsStore(str(ckpt))
    )
    run_id = _make_run(repo1, "req-restart", conversation_id="conv-restart", profile="artifact")
    repo1.create_sandbox_lease(
        run_id=run_id, sandbox_type="local_native", workspace_path=f"/ws/{run_id}"
    )
    key = compute_effect_key(run_id=run_id, tool_name="generate_pdf", args={"filename": "p.pdf"})
    repo1.insert_effect(
        effect_key=key,
        run_id=run_id,
        conversation_id="conv-restart",
        tool_name="generate_pdf",
        args_canonical='{"filename": "p.pdf"}',
        external_idempotency_key="ext-restart",
    )
    repo1.persist_event(
        run_id=run_id, sequence=1, event_type="progress",
        payload={"message": "working"},
        request_id="req-restart", conversation_id="conv-restart",
    )
    repo1.mark_run_paused(run_id)
    repo1.close()

    # Simulated restart: brand-new store instances over the SAME files.
    repo2 = RunRepository(
        control=ControlStore(str(control)), checkpoints=CheckpointsStore(str(ckpt))
    )
    try:
        run = repo2.get_run(run_id)
        assert run is not None and run.status == "paused"
        assert repo2.is_paused(run_id)
        effect = repo2.get_effect(key)
        assert effect is not None and effect.status == EffectStatus.AUTHORIZED.value
        lease = repo2.get_sandbox_lease(run_id)
        assert lease is not None and lease.status == "active"
        events = repo2.get_events_since(run_id=run_id, from_seq=0)
        assert [e["type"] for e in events] == ["progress"]
        snapshot = repo2.get_snapshot(run_id)
        assert snapshot is not None
        assert any(m.get("content") == "message-for-req-restart" for m in snapshot.messages)
    finally:
        repo2.close()


# ---- Part 3: per-turn checkpoint writes + no duplicate events (AC3) ------------


def test_event_appends_never_duplicate(tmp_path: Path):
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo, "req-events", conversation_id="conv-events")
        for sequence in (1, 2, 3):
            repo.persist_event(
                run_id=run_id, sequence=sequence, event_type="text",
                payload={"n": sequence},
                request_id="req-events", conversation_id="conv-events",
            )
        # Monotonic, gap-free, exactly once.
        events = repo.get_events_since(run_id=run_id, from_seq=0)
        assert [e["sequence"] for e in events] == [1, 2, 3]
        # A duplicate (run_id, sequence) is a hard error, never a silent append.
        with pytest.raises(sqlite3.IntegrityError):
            repo.persist_event(
                run_id=run_id, sequence=2, event_type="text",
                payload={"n": "duplicate"},
                request_id="req-events", conversation_id="conv-events",
            )
        after = repo.get_events_since(run_id=run_id, from_seq=0)
        assert [e["sequence"] for e in after] == [1, 2, 3]
        assert [e["payload"]["n"] for e in after] == [1, 2, 3]
    finally:
        repo.close()


# ---- Part 4: Run Task labels (AC4) ---------------------------------------------


def test_kernel_tasks_keep_label_and_notify_owner_on_completion():
    """Tasks keep their registry label until terminal; the owner is notified."""
    from uniharness.tasks import TaskRegistry
    from uniharness.types import ToolResult

    registry = TaskRegistry()

    async def _background() -> ToolResult:
        return ToolResult(output="background-done")

    async def scenario() -> None:
        registry.submit("task-bg-1", "run_task", "计划工作流后台任务", _background())
        entry = registry.get("task-bg-1")
        assert entry is not None
        # The label rides along while the task runs.
        assert entry.kind == "run_task"
        assert entry.status == "running"
        await registry.wait("task-bg-1", timeout_ms=5000)
        finished = registry.get("task-bg-1")
        assert finished is not None and finished.status == "completed"
        assert finished.kind == "run_task"  # label survives to terminal
        assert finished.result is not None and finished.result.output == "background-done"

        # Status change notifies the owner via the completion queue.
        completions = registry.drain_completions()
        assert [c.task_id for c in completions] == ["task-bg-1"]
        assert completions[0].kind == "run_task"
        assert completions[0].status == "completed"
        # Drain is idempotent — no duplicate notifications.
        assert registry.drain_completions() == []

    asyncio.run(scenario())


def test_run_completion_removes_task_labels():
    """Agent close (run complete/error) removes the run's task labels."""
    from uniharness.tasks import TaskRegistry
    from uniharness.types import ToolResult

    registry = TaskRegistry()
    registry.register("task-fg-1", "run_task", "计划工作流前台任务")
    registry.complete("task-fg-1", ToolResult(output="ok"))
    assert registry.get("task-fg-1") is not None
    assert registry.get("task-fg-1").kind == "run_task"

    async def scenario() -> None:
        # A still-running background label must be finalized, not orphaned.
        async def _never_ends() -> ToolResult:
            await asyncio.sleep(30)
            return ToolResult(output="late")

        registry.submit("task-bg-2", "run_task", "未完成的后台任务", _never_ends())
        assert registry.get("task-bg-2").status == "running"
        await asyncio.sleep(0)  # let the background task actually start
        await registry.cancel_all()
        entry = registry.get("task-bg-2")
        assert entry is not None and entry.status == "cancelled"

    asyncio.run(scenario())
    # After the run's agent closed, no label still claims to be running.
    running = [e for e in registry._tasks.values() if e.status == "running"]
    assert running == []


def test_concurrent_run_agents_keep_isolated_task_registries():
    """Two concurrent runs' kernels never share a task registry/label scope."""
    from uniharness.langchain.agent import create_agent
    from uniharness.tasks import TaskRegistry
    from uniharness.types import ToolResult

    class _Probe:
        is_running = True

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def run(self, command: str, *, timeout: float | None = None):
            raise AssertionError("resolve_environment=False must not probe")

    async def scenario() -> None:
        agent_a = await create_agent(
            "openai:gpt-4o-mini", _Probe(), skill_paths=(), resolve_environment=False
        )
        agent_b = await create_agent(
            "openai:gpt-4o-mini", _Probe(), skill_paths=(), resolve_environment=False
        )
        try:
            registry_a = agent_a._task_registry
            registry_b = agent_b._task_registry
            assert isinstance(registry_a, TaskRegistry)
            assert registry_a is not registry_b  # isolated label scopes

            async def _long_running() -> ToolResult:
                await asyncio.sleep(30)
                return ToolResult(output="late")

            registry_a.submit("task-in-a", "run_task", "run A 的后台任务", _long_running())
            registry_b.submit("task-in-b", "run_task", "run B 的后台任务", _long_running())

            # Run B cannot see (or borrow) run A's label.
            assert registry_b.get("task-in-a") is None
            assert registry_a.get("task-in-b") is None

            # Closing run A's agent (run complete/error) removes A's live
            # labels; run B's label scope is untouched.
            await agent_a.aclose()
            entry_a = registry_a.get("task-in-a")
            assert entry_a is not None and entry_a.status == "cancelled"
            assert registry_b.get("task-in-b").status == "running"
            assert not [
                e for e in registry_a._tasks.values() if e.status == "running"
            ]
        finally:
            await agent_b.aclose()

    asyncio.run(scenario())


def test_terminal_events_correlate_to_run(tmp_path: Path):
    """done/error/cancelled events always carry the correlatable run_id."""
    from agent_v2.kernel import KernelModelFailure
    from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator

    class _FakeMemory:
        def memory_enabled(self, subject_id): return True
        def claim_conversation(self, s, c): return True
        def owns_conversation(self, s, c): return True
        def create_candidate(self, s, **kw): return kw

    async def _boom(**kwargs: Any):
        raise KernelModelFailure("model_unavailable", "模型不可用")

    repo = _repo(tmp_path)
    orch = RunOrchestrator(
        settings=SimpleNamespace(
            agent_v2_enabled=True, openai_api_key="sk-test", openai_model="m",
            agent_v2_cowork_policy="disabled",
        ),
        runs_repo=repo,
        memory=_FakeMemory(),
        kernel_turn=_boom,
    )
    try:
        req = ChatFlowRequest(message="你好", request_id="req-terminal-corr")

        async def run() -> list[Any]:
            return [e async for e in orch.stream_chat(req, subject_id="sub-1")]

        events = asyncio.run(run())
        terminals = [e for e in events if e.type in TERMINALS]
        assert len(terminals) == 1 and terminals[0].type == "error"
        # Every event — terminal included — correlates to the one run.
        run_ids = {e.run_id for e in events}
        assert len(run_ids) == 1 and None not in run_ids
        stored = repo.get_events_since(run_id=events[0].run_id, from_seq=0)
        assert stored and all(e["run_id"] == events[0].run_id for e in stored)
    finally:
        repo.close()


# ---- Part 5: exactly-once concurrency (AC6) ------------------------------------


def test_concurrent_same_key_effect_insert_yields_exactly_one_row(tmp_path: Path):
    """Concurrent writers racing one effect key: exactly one row, one success."""
    shared = tmp_path / "control-race.sqlite"

    # The run row must exist first (FK) — create it through one repository.
    seeder = RunRepository(
        control=ControlStore(str(shared)),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt-race.sqlite")),
    )
    run_id, _ = seeder.create_run_idempotent(
        request_id="req-race",
        conversation_id="conv-race",
        profile_id="cowork",
        user_message="race",
    )
    seeder.close()

    # Two independent connections (two writers) over the SAME database file.
    store_a = ControlStore(str(shared))
    store_b = ControlStore(str(shared))

    args_canonical = '{"command": "echo race"}'
    key = compute_effect_key(run_id=run_id, tool_name="sandbox_bash", args={"command": "echo race"})

    results: list[str] = []
    barrier = threading.Barrier(2)

    def _insert(store: ControlStore) -> None:
        barrier.wait()
        try:
            store.insert_effect(
                effect_key=key,
                run_id=run_id,
                conversation_id="conv-race",
                tool_name="sandbox_bash",
                args_canonical=args_canonical,
                external_idempotency_key=f"ext-{key}",
            )
            results.append("ok")
        except sqlite3.IntegrityError:
            results.append("rejected")

    threads = [
        threading.Thread(target=_insert, args=(store_a,)),
        threading.Thread(target=_insert, args=(store_b,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert sorted(results) == ["ok", "rejected"], results
    conn = sqlite3.connect(str(shared))
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM effects WHERE effect_key = ?", (key,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 1  # exactly one row survives
    store_a._conn.close()
    store_b._conn.close()


def test_concurrent_effect_execution_runs_exactly_once(tmp_path: Path):
    """Concurrent executors on one authorized effect: one execution, one commit.

    Concurrency is modelled the way production contends: two coroutines on
    one event loop racing the effect-ledger CAS (the DB section between
    ``get_effect`` and ``transition_effect_to_executing`` is synchronous,
    so exactly one caller can win the transition).
    """
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo, "req-exec-once", conversation_id="conv-exec")
        args = {"command": "echo once"}
        key = compute_effect_key(run_id=run_id, tool_name="sandbox_bash", args=args)
        repo.insert_effect(
            effect_key=key,
            run_id=run_id,
            conversation_id="conv-exec",
            tool_name="sandbox_bash",
            args_canonical='{"command": "echo once"}',
            external_idempotency_key="ext-once",
        )

        class _Bridge:
            def __init__(self) -> None:
                self.commands: list[str] = []

            async def run(self, command: str, *, timeout: float | None = None):
                self.commands.append(command)
                # Yield so the second executor actually interleaves with the
                # first one mid-flight (a real race, not sequential replay).
                await asyncio.sleep(0.01)
                return SimpleNamespace(
                    exit_code=0, stdout="once", stderr="", duration_ms=2, truncated=False
                )

        bridge = _Bridge()
        sink: list[dict[str, Any]] = []
        tool = CoworkBashTool(bridge=bridge, run_id=run_id, effects=repo, audit_sink=sink.append)

        async def scenario() -> list[Any]:
            return list(await asyncio.gather(
                tool(command="echo once"), tool(command="echo once")
            ))

        outputs = asyncio.run(scenario())

        assert bridge.commands == ["echo once"], bridge.commands  # exactly one execution
        effect = repo.get_effect(key)
        assert effect is not None and effect.status == EffectStatus.COMMITTED.value
        # The loser is refused — never silently re-executed. It sees the
        # in-flight ``executing`` row and is told to wait for reconciliation.
        denied = [
            e for e in sink
            if e.get("phase") == "denied"
            and e.get("code") in {"effect_race_lost", "outcome_unknown"}
        ]
        assert len(denied) == 1, sink
        assert sum(1 for o in outputs if o is not None and o.output == "once") == 1
        assert any(o is not None and o.error for o in outputs)  # loser got a stable error
    finally:
        repo.close()


def test_recovery_state_task_is_consistently_rejected(tmp_path: Path):
    """An effect already in recovery (outcome_unknown) never auto-replays."""
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo, "req-recovery", conversation_id="conv-recovery")
        args = {"command": "echo recover"}
        key = compute_effect_key(run_id=run_id, tool_name="sandbox_bash", args=args)
        repo.insert_effect(
            effect_key=key,
            run_id=run_id,
            conversation_id="conv-recovery",
            tool_name="sandbox_bash",
            args_canonical='{"command": "echo recover"}',
            external_idempotency_key="ext-recover",
        )

        class _Bridge:
            def __init__(self) -> None:
                self.commands: list[str] = []

            async def run(self, command: str, *, timeout: float | None = None):
                self.commands.append(command)
                return SimpleNamespace(
                    exit_code=0, stdout="ok", stderr="", duration_ms=1, truncated=False
                )

        bridge = _Bridge()
        tool = CoworkBashTool(bridge=bridge, run_id=run_id, effects=repo, audit_sink=None)

        # Simulate a crashed execution: authorized → executing → outcome_unknown.
        repo.transition_effect_to_executing(key)
        repo.mark_effect_outcome_unknown(key, "process died")
        assert repo.get_effect(key).status == EffectStatus.OUTCOME_UNKNOWN.value

        result = asyncio.run(tool(command="echo recover"))
        assert result.error and "outcome_unknown" in result.error
        assert bridge.commands == []  # nothing executed

        # And the same state is consistently rejected on a fresh tool instance.
        tool2 = CoworkBashTool(bridge=bridge, run_id=run_id, effects=repo, audit_sink=None)
        result2 = asyncio.run(tool2(command="echo recover"))
        assert result2.error and "outcome_unknown" in result2.error
        assert bridge.commands == []
    finally:
        repo.close()


def test_writer_lock_denies_second_holder_and_failed_renewal(tmp_path: Path):
    """A second writer (and a failed renewal) fails closed — never co-ownership."""
    store_a = ControlStore(str(tmp_path / "control-lock.sqlite"))
    store_b = ControlStore(str(tmp_path / "control-lock.sqlite"))
    try:
        assert store_a.acquire_writer_lock("writer-a") is True
        # Second holder is denied; the existing owner keeps the lock.
        assert store_b.acquire_writer_lock("writer-b") is False
        # Failed renewal: re-acquiring under a different id still fails closed.
        assert store_a.acquire_writer_lock("writer-b") is False
        row = store_a.get_writer_lock()
        assert row is not None and row["writer_id"] == "writer-a"
        # The owner's own re-acquire is idempotent (renewal success).
        assert store_a.acquire_writer_lock("writer-a") is True
        store_a.release_writer_lock("writer-a")
        assert store_a.get_writer_lock() is None
        # After release a new writer may acquire.
        assert store_b.acquire_writer_lock("writer-b") is True
    finally:
        store_a._conn.close()
        store_b._conn.close()


def test_writer_lock_busy_across_live_processes(tmp_path: Path):
    """The OS-level lock refuses a second live process (single-writer, spec AC7)."""
    from agent_v2.writer_lock import WriterLockBusy, acquire_writer_lock

    state_dir = tmp_path / "state"
    holder_code = (
        "import sys, time;"
        "sys.path.insert(0, sys.argv[1]);"
        "from agent_v2.writer_lock import acquire_writer_lock;"
        "lock = acquire_writer_lock(sys.argv[2]);"
        "print('held', flush=True);"
        "time.sleep(8)"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_code, str(Path(__file__).resolve().parent), str(state_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = holder.stdout.readline() if holder.stdout else ""
        assert "held" in line, f"holder did not acquire: {line}"
        with pytest.raises(WriterLockBusy):
            acquire_writer_lock(str(state_dir))
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_result_swap_across_effect_keys_is_refused(tmp_path: Path):
    """A committed result cannot be swapped onto another effect key."""
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo, "req-swap", conversation_id="conv-swap")
        key_a = compute_effect_key(run_id=run_id, tool_name="sandbox_bash", args={"command": "echo A"})
        key_b = compute_effect_key(run_id=run_id, tool_name="sandbox_bash", args={"command": "echo B"})
        for key, command in ((key_a, "echo A"), (key_b, "echo B")):
            repo.insert_effect(
                effect_key=key,
                run_id=run_id,
                conversation_id="conv-swap",
                tool_name="sandbox_bash",
                args_canonical=f'{{"command": "{command}"}}',
                external_idempotency_key=f"ext-{key}",
            )

        # Legitimately execute + commit B only.
        repo.transition_effect_to_executing(key_b)
        committed = repo.commit_effect(key_b, serialize_result({"stdout": "B-output"}))
        assert committed is not None and committed.status == EffectStatus.COMMITTED.value

        # Swap attempt 1: commit B's result under A's key — A never executed,
        # so the CAS (executing → committed) refuses the illegal transition.
        stolen = repo.commit_effect(key_a, serialize_result({"stdout": "B-output"}))
        assert stolen is not None and stolen.status == EffectStatus.AUTHORIZED.value
        assert repo.get_effect(key_a).status == EffectStatus.AUTHORIZED.value

        # Swap attempt 2: re-committing B with a DIFFERENT result does not
        # overwrite the durable result (no second write).
        again = repo.commit_effect(key_b, serialize_result({"stdout": "swapped"}))
        assert again is not None and again.status == EffectStatus.COMMITTED.value
        import json as _json

        stored = _json.loads(repo.get_effect(key_b).result_json)
        assert stored.get("stdout") == "B-output"
    finally:
        repo.close()


def test_forged_cross_run_label_cannot_borrow_authorization(tmp_path: Path):
    """A forged shared label (foreign run_id) never borrows another run's effect."""
    repo = _repo(tmp_path)
    try:
        run_a = _make_run(repo, "req-forge-a", conversation_id="conv-forge")
        run_b = _make_run(repo, "req-forge-b", conversation_id="conv-forge")
        args = {"command": "echo privileged"}
        key_a = compute_effect_key(run_id=run_a, tool_name="sandbox_bash", args=args)
        repo.insert_effect(
            effect_key=key_a,
            run_id=run_a,
            conversation_id="conv-forge",
            tool_name="sandbox_bash",
            args_canonical='{"command": "echo privileged"}',
            external_idempotency_key="ext-forge",
        )

        class _Bridge:
            def __init__(self) -> None:
                self.commands: list[str] = []

            async def run(self, command: str, *, timeout: float | None = None):
                self.commands.append(command)
                return SimpleNamespace(
                    exit_code=0, stdout="leaked", stderr="", duration_ms=1, truncated=False
                )

        bridge = _Bridge()
        # Run B claims the same command — the key is run-scoped, so there is
        # no authorization to borrow: fail closed before any execution.
        tool_b = CoworkBashTool(bridge=bridge, run_id=run_b, effects=repo, audit_sink=None)
        result = asyncio.run(tool_b(command="echo privileged"))
        assert result.error and "缺少授权记录" in result.error
        assert bridge.commands == []
        assert repo.get_effect_by_tool_call(
            run_id=run_b, tool_name="sandbox_bash",
            args_canonical='{"command": "echo privileged"}',
        ) is None
        # Run A's authorization is untouched.
        assert repo.get_effect(key_a).status == EffectStatus.AUTHORIZED.value
    finally:
        repo.close()


# ---- Part 6: disconnect + single terminal (AC5) --------------------------------


def test_disconnect_then_resume_yields_single_terminal(tmp_path: Path):
    """A client that disconnects mid-stream still ends with exactly one terminal."""
    from agent_v2.computer import LocalNativeComputer
    from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator

    class _FakeMemory:
        def memory_enabled(self, subject_id): return True
        def claim_conversation(self, s, c): return True
        def owns_conversation(self, s, c): return True
        def create_candidate(self, s, **kw): return kw

    async def fake_kernel(**kwargs: Any):
        return ("# 杭州两日徒步计划\n\n正文内容。", [])

    repo = _repo(tmp_path)
    orch = RunOrchestrator(
        settings=SimpleNamespace(
            agent_v2_enabled=True, openai_api_key="sk-test", openai_model="m",
            agent_v2_state_dir=str(tmp_path / "state"),
        ),
        runs_repo=repo,
        memory=_FakeMemory(),
        kernel_turn=fake_kernel,
        computer_factory=lambda *a, **k: LocalNativeComputer(),
        # ALR ticket 03: explicit typed result via the production helper.
        sandbox_cleanup=lambda comp, workspace_path, _runs, _rid: (
            run_sandbox_cleanup(comp, workspace_path)
        ),
        gate_factory=None,
    )
    req = ChatFlowRequest(
        message="规划杭州两日徒步并出 PDF 计划",
        request_id="req-disconnect",
        profile_id="artifact",
    )

    async def consume_until_paused() -> tuple[str, list[Any]]:
        received: list[Any] = []
        stream = orch.stream_chat(req, subject_id="sub-disconnect")
        async for envelope in stream:
            received.append(envelope)
            if envelope.type == "run.paused":
                break  # client disconnects right after the pause is persisted
        await stream.aclose()
        run = repo.control.get_run_by_request("req-disconnect")
        return run.run_id, received

    run_id, partial = asyncio.run(consume_until_paused())
    assert any(e.type == "run.paused" for e in partial)
    assert all(e.run_id == run_id for e in partial)

    # The user approves while the client is away; reconnecting replays the
    # stored events and continues the flow.
    pending = repo.list_pending_approvals_for_run(run_id)
    assert pending, "the pause must have persisted an approval row"
    repo.resolve_approval(
        approval_id=pending[0].approval_id,
        status="approved",
        decision_request_id="dec-disconnect",
    )

    async def resume_all() -> list[Any]:
        return [e async for e in orch.stream_resume(run_id=run_id)]

    resumed = asyncio.run(resume_all())

    types = [e.type for e in resumed]
    terminals = [t for t in types if t in TERMINALS]
    assert terminals.count("done") + terminals.count("error") == 1, types
    # No duplicated tool execution across the disconnect boundary.
    tool_calls = [e for e in resumed if e.type == "tool_call"]
    assert len(tool_calls) == 1
    artifacts = repo.list_artifacts_for_run(run_id)
    assert len(artifacts) <= 1
    assert all(e.run_id == run_id for e in resumed)
