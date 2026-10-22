"""BCDEF ticket 02 — admission fence and idempotent canonical run.

High-level seam: the durable admission operation on ``RunRepository`` plus
the orchestration entry. Tests assert external state (run/message/event
counts, statuses, terminal order, model-call count), never private call
order. Partial-commit failures are injected by replacing a single store
method with a fail-once wrapper.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import api.chat_v2 as chat_v2_module
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.runs import RunRepository
from test_run_orchestration_chat import (
    SUBJECT_ID,
    _FakeMemory,
    _repo,
    _settings,
)


TERMINALS = {"error", "done", "cancelled"}


def _orchestrator(
    repo: RunRepository,
    kernel_calls: list[dict[str, Any]],
    *,
    settings: Any | None = None,
    computer_factory: Any | None = None,
) -> RunOrchestrator:
    async def fake_kernel(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        kernel_calls.append(kwargs)
        return ("好的。", [])

    return RunOrchestrator(
        settings=settings or _settings(),
        runs_repo=repo,
        memory=_FakeMemory(),
        kernel_turn=fake_kernel,
        computer_factory=computer_factory,
    )


def _collect(gen: Any) -> list[Any]:
    async def run() -> list[Any]:
        return [item async for item in gen]

    return asyncio.run(run())


def _stream(orch: RunOrchestrator, request_id: str, **kwargs: Any) -> list[Any]:
    return _collect(
        orch.stream_chat(
            ChatFlowRequest(message="带什么装备？", request_id=request_id, **kwargs),
            subject_id=SUBJECT_ID,
        )
    )


def _fail_once(monkeypatch: Any, owner: Any, name: str, exc: Exception) -> dict[str, int]:
    """Replace ``owner.name`` so the FIRST call raises ``exc``, then delegates."""
    state = {"calls": 0}
    original = getattr(owner, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        state["calls"] += 1
        if state["calls"] == 1:
            raise exc
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)
    return state


# ---- normal admission ------------------------------------------------------


def test_admit_persists_canonical_conversation_message_and_admission_event(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    result = repo.admit(
        request_id="req-adm-1",
        conversation_id=None,
        profile_id="chat",
        user_message="你好",
    )

    assert result.ok
    assert result.created and result.execute
    assert result.conversation_id == "conv-req-adm-1"

    run = repo.control.get_run_by_request("req-adm-1")
    assert run is not None
    assert run.run_id == result.run_id
    assert run.conversation_id == "conv-req-adm-1"
    assert run.status == "admitted"

    messages = repo.checkpoints.get_messages(conversation_id=result.conversation_id)
    assert [(m.role, m.content) for m in messages] == [("user", "你好")]

    events = repo.get_events_since(run_id=result.run_id, from_seq=0)
    assert [e["type"] for e in events] == ["run_started"]
    assert events[0]["conversation_id"] == result.conversation_id
    assert events[0]["payload"]["admission"]["status"] == "admitted"


def test_stream_uses_canonical_conversation_for_every_envelope(tmp_path: Path):
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []
    orch = _orchestrator(repo, calls)

    items = _stream(orch, "req-adm-conv")

    assert [e.type for e in items] == [
        "run_started", "progress", "heartbeat", "text", "done",
    ]
    assert {e.conversation_id for e in items} == {"conv-req-adm-conv"}
    assert [e.sequence for e in items] == [1, 2, 3, 4, 5]
    assert len(calls) == 1

    run = repo.control.get_run_by_request("req-adm-conv")
    assert run is not None and run.status == "done"
    stored = repo.get_events_since(run_id=run.run_id, from_seq=0)
    assert all(e["conversation_id"] == "conv-req-adm-conv" for e in stored)


def test_duplicate_admission_reuses_run_and_never_duplicates_user_message(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    first = repo.admit(
        request_id="req-adm-dup",
        conversation_id=None,
        profile_id="chat",
        user_message="重复",
    )
    second = repo.admit(
        request_id="req-adm-dup",
        conversation_id="ignored-client-id",
        profile_id="chat",
        user_message="重复",
    )

    assert first.run_id == second.run_id
    assert second.conversation_id == first.conversation_id == "conv-req-adm-dup"
    assert second.replayed and not second.execute
    messages = repo.checkpoints.get_messages(conversation_id=first.conversation_id)
    assert [m.content for m in messages] == ["重复"]
    events = repo.get_events_since(run_id=first.run_id, from_seq=0)
    assert [e["type"] for e in events] == ["run_started"]
    assert events[0]["payload"]["admission"]["replayed"] is True


def test_concurrent_admissions_converge_on_one_run_and_message(tmp_path: Path):
    repos = [
        RunRepository(
            control=repo.control,
            checkpoints=repo.checkpoints,
        )
        for repo in [_repo(tmp_path), _repo(tmp_path)]
    ]
    barrier = threading.Barrier(2)
    results: list[Any] = []
    errors: list[BaseException] = []

    def worker(repo: RunRepository) -> None:
        try:
            barrier.wait(timeout=5)
            results.append(
                repo.admit(
                    request_id="req-adm-concurrent",
                    conversation_id=None,
                    profile_id="chat",
                    user_message="并发",
                )
            )
        except BaseException as exc:  # pragma: no cover - failure evidence
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(repo,)) for repo in repos]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert len(results) == 2
    run_ids = {result.run_id for result in results}
    assert len(run_ids) == 1
    assert len([r for r in results if r.execute]) == 1  # exactly one executor

    verifier = repos[0]
    run_id = run_ids.pop()
    run = verifier.get_run(run_id)
    assert run is not None and run.status == "admitted"
    assert len(verifier.get_events_since(run_id=run_id, from_seq=0)) == 1
    messages = verifier.checkpoints.get_messages(conversation_id=run.conversation_id)
    assert [m.content for m in messages] == ["并发"]


# ---- fail-closed write failures -------------------------------------------


def test_run_write_failure_emits_admission_failed_without_model_call(
    tmp_path: Path, monkeypatch,
):
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []

    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("control store unavailable")

    monkeypatch.setattr(repo.control, "admit_run", boom)
    orch = _orchestrator(repo, calls)

    items = _stream(orch, "req-adm-runfail")

    assert [e.type for e in items] == ["error"]
    assert items[0].payload["code"] == "admission_failed"
    assert items[0].payload["recoverable"] is True
    assert calls == []
    assert repo.control.get_run_by_request("req-adm-runfail") is None


def test_conversation_write_failure_fails_closed_and_retry_repairs(
    tmp_path: Path, monkeypatch,
):
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []
    state = _fail_once(
        monkeypatch,
        repo.checkpoints,
        "ensure_conversation",
        RuntimeError("checkpoint store unavailable"),
    )
    orch = _orchestrator(repo, calls)

    first = _stream(orch, "req-adm-convfail")

    assert [e.type for e in first] == ["error"]
    assert first[0].payload["code"] == "admission_failed"
    assert calls == []
    assert state["calls"] == 1
    run = repo.control.get_run_by_request("req-adm-convfail")
    assert run is not None and run.status == "admission_failed"
    assert repo.checkpoints.get_messages(conversation_id=run.conversation_id) == []

    # Same request retry repairs the missing records, then executes once.
    second = _stream(orch, "req-adm-convfail")
    assert [e.type for e in second] == [
        "run_started", "progress", "heartbeat", "text", "done",
    ]
    assert second[0].payload["admission"]["status"] == "repaired"
    assert len(calls) == 1
    messages = repo.checkpoints.get_messages(conversation_id=run.conversation_id)
    assert [m.role for m in messages] == ["user"]

    # A further retry is a pure replay: no second model call.
    third = _stream(orch, "req-adm-convfail")
    assert [e.type for e in third] == [e.type for e in second]
    assert len(calls) == 1
    assert repo.get_run(run.run_id).status == "done"


def test_user_message_write_failure_fails_closed_and_repairs_on_retry(
    tmp_path: Path, monkeypatch,
):
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []
    _fail_once(
        monkeypatch,
        repo.checkpoints,
        "ensure_user_message",
        RuntimeError("message write failed"),
    )
    orch = _orchestrator(repo, calls)

    first = _stream(orch, "req-adm-msgfail")
    assert [e.type for e in first] == ["error"]
    assert first[0].payload["code"] == "admission_failed"
    assert calls == []

    run = repo.control.get_run_by_request("req-adm-msgfail")
    assert run is not None and run.status == "admission_failed"
    assert repo.checkpoints.get_messages(conversation_id=run.conversation_id) == []

    second = _stream(orch, "req-adm-msgfail")
    assert second[-1].type == "done"
    assert len(calls) == 1
    messages = repo.checkpoints.get_messages(conversation_id=run.conversation_id)
    assert [m.content for m in messages] == ["带什么装备？"]


def test_admission_repair_after_admitted_mark_failure_is_replayable(
    tmp_path: Path, monkeypatch,
):
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []
    _fail_once(
        monkeypatch,
        repo.control,
        "mark_run_admitted",
        RuntimeError("status write failed"),
    )
    orch = _orchestrator(repo, calls)

    first = _stream(orch, "req-adm-markfail")
    assert [e.type for e in first] == ["run_started", "error"]
    assert first[1].payload["code"] == "admission_failed"
    assert calls == []
    run = repo.control.get_run_by_request("req-adm-markfail")
    assert run is not None and run.status == "admission_failed"

    stored = repo.get_events_since(run_id=run.run_id, from_seq=0)
    assert [e["type"] for e in stored] == ["run_started", "error"]
    assert stored[1]["payload"]["code"] == "admission_failed"

    second = _stream(orch, "req-adm-markfail")
    assert second[-1].type == "done"
    assert second[0].payload["admission"]["repaired"] is True
    assert len(calls) == 1
    assert repo.get_run(run.run_id).status == "done"


def test_failed_admission_never_creates_a_sandbox(tmp_path: Path, monkeypatch):
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []
    created: list[str] = []

    def factory() -> Any:
        created.append("called")
        raise AssertionError("sandbox must not be created after failed admission")

    _fail_once(
        monkeypatch,
        repo.checkpoints,
        "ensure_conversation",
        RuntimeError("checkpoint store unavailable"),
    )
    orch = _orchestrator(repo, calls, computer_factory=factory)

    items = _stream(orch, "req-adm-sandbox", profile_id="artifact")

    assert [e.type for e in items] == ["error"]
    assert items[0].payload["code"] == "admission_failed"
    assert created == []
    assert calls == []


# ---- replay visibility -----------------------------------------------------


def test_events_endpoint_replays_admission_failure_and_repair(tmp_path: Path, monkeypatch):
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []
    _fail_once(
        monkeypatch,
        repo.control,
        "mark_run_admitted",
        RuntimeError("status write failed"),
    )
    orch = _orchestrator(repo, calls)
    _stream(orch, "req-adm-replay")

    def fake_subject(request: Any) -> Any:
        request.state.subject_id = "sub-1"
        return _FakeMemory(), SimpleNamespace(
            issued=False, token="tok", subject_id="sub-1"
        )

    monkeypatch.setattr(chat_v2_module, "resolve_request_subject", fake_subject)
    app = FastAPI()
    app.include_router(chat_v2_module.chat_v2_router, prefix="/api/v2")
    app.state.agent_v2_runtime = SimpleNamespace(runs=repo, memory=_FakeMemory())

    async def fetch(path: str) -> Any:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)

    failed = asyncio.run(
        fetch("/api/v2/chat/events/-?from_seq=0&request_id=req-adm-replay")
    )
    assert failed.status_code == 200
    body = failed.json()
    assert [e["type"] for e in body["events"]] == ["run_started", "error"]
    assert body["events"][1]["payload"]["code"] == "admission_failed"

    # Repair retry: the admission event now explains the repaired outcome.
    _stream(orch, "req-adm-replay")
    repaired = asyncio.run(
        fetch("/api/v2/chat/events/-?from_seq=0&request_id=req-adm-replay")
    )
    events = repaired.json()["events"]
    assert events[0]["type"] == "run_started"
    assert events[0]["payload"]["admission"]["status"] == "repaired"
    assert [e["type"] for e in events].count("done") == 1
    assert len(calls) == 1


def test_duplicate_terminal_request_replays_admission_event_once(tmp_path: Path):
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []
    orch = _orchestrator(repo, calls)

    first = _stream(orch, "req-adm-terminal")
    second = _stream(orch, "req-adm-terminal")

    assert [e.type for e in second] == [e.type for e in first]
    assert second[0].type == "run_started"
    assert [e.type for e in second].count("done") == 1
    assert len(calls) == 1
