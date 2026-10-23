"""BCDEF boundary for the memory-export path.

The export endpoint creates a durable run without a model turn, so it must
still pass the shared admission fence (run + user message + admission event)
and the terminal fence (terminal event persisted before the status flips).
Real SQLite under a tmp dir; the memory service is a counting fake.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api import memory_v2
from agent_v2.memory_service import SubjectResolution
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.runs import RunRepository


class _Request:
    def __init__(self, runtime) -> None:
        self.app = SimpleNamespace(state=SimpleNamespace(agent_v2_runtime=runtime))
        self.cookies: dict[str, str] = {}
        self.state = SimpleNamespace()
        self.url = SimpleNamespace(scheme="http")


class _FakeMemory:
    def __init__(self) -> None:
        self.claims = 0

    def resolve_subject(self, token=None) -> SubjectResolution:
        return SubjectResolution(subject_id="subj-export", token="tok", issued=False)

    def claim_conversation(self, subject_id: str, conversation_id: str) -> bool:
        self.claims += 1
        return True

    def count_active_memories(self, subject_id: str) -> int:
        return 1

    def list_active_memories(self, subject_id: str, limit=1, offset=0):
        return [{"content": "喜欢清晨出发"}]


def _repo(tmp_path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def _runtime(repo: RunRepository, memory: _FakeMemory):
    return SimpleNamespace(runs=repo, memory=memory)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(
        memory_v2, "settings", SimpleNamespace(agent_v2_state_dir=str(tmp_path))
    )
    repo = _repo(tmp_path)
    memory = _FakeMemory()
    request = _Request(_runtime(repo, memory))
    yield repo, memory, request
    repo.control.close()
    repo.checkpoints.close()


def _events(repo: RunRepository, run_id: str) -> list[tuple[int, str]]:
    return [
        (row[0], row[1])
        for row in repo.checkpoints._conn.execute(
            "SELECT sequence, event_type FROM run_events WHERE run_id = ?"
            " ORDER BY sequence",
            (run_id,),
        )
    ]


def test_export_admits_one_run_and_persists_one_terminal(env):
    repo, _memory, request = env

    response = asyncio.run(memory_v2.export_memories(request))
    body = json.loads(response.body)

    runs = repo.control._conn.execute(
        "SELECT run_id, status, terminal_type, conversation_id FROM runs"
    ).fetchall()
    assert len(runs) == 1
    run_id, status, terminal_type, conversation_id = runs[0]
    assert (status, terminal_type) == ("done", "done")
    assert conversation_id  # canonical conversation from admission

    events = _events(repo, run_id)
    assert events[0] == (1, "run_started")
    terminals = [seq_type for _, seq_type in events if seq_type in ("done", "error", "cancelled")]
    assert terminals == ["done"]
    assert body["artifact_id"]

    artifacts = [
        (row["run_id"], row["status"])
        for row in repo.control._conn.execute(
            "SELECT run_id, status FROM artifacts"
        )
    ]
    assert artifacts == [(run_id, "available")]


def test_export_admission_failure_returns_503_and_writes_nothing(env, monkeypatch):
    repo, _memory, request = env

    monkeypatch.setattr(
        repo,
        "admit",
        lambda **_kwargs: SimpleNamespace(
            run_id="",
            conversation_id="conv-x",
            status="admission_failed",
            error_code="admission_failed",
            recoverable=True,
        ),
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(memory_v2.export_memories(request))

    assert excinfo.value.status_code == 503
    assert repo.control._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert (
        repo.control._conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
    )


def test_export_artifact_write_failure_persists_error_terminal(
    env, monkeypatch
):
    repo, _memory, request = env
    monkeypatch.setattr(
        memory_v2,
        "persist_artifact_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk gone")),
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(memory_v2.export_memories(request))

    assert excinfo.value.status_code == 503
    runs = repo.control._conn.execute(
        "SELECT run_id, status, terminal_type FROM runs"
    ).fetchall()
    assert len(runs) == 1
    run_id, status, terminal_type = runs[0]
    assert (status, terminal_type) == ("error", "error")
    events = _events(repo, run_id)
    assert [event_type for _, event_type in events] == ["run_started", "error"]


def test_export_terminal_persist_failure_is_reported(env, monkeypatch):
    repo, _memory, request = env

    def boom(**_kwargs):
        raise RuntimeError("terminal write failed")

    monkeypatch.setattr(repo, "finalize_run", boom)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(memory_v2.export_memories(request))

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail["code"] == "terminal_persist_failed"
