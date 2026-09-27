"""A05 — snapshot/events accept ``request_id`` fallback (AC4 parity with cancel).

Recovery endpoints previously resolved only a real run_id: a client that
refreshes before the first v2 event delivered the server run_id held only
its ``request_id``, got a 404, and dropped its recovery context while the
run kept executing server-side. Cancel already resolved ``request_id`` →
``run_id`` (spec AC4); snapshot and events now share that semantics:

* the path identifier may be a run_id or a request_id;
* ``?request_id=`` is an additional fallback for clients that cannot put
  the idempotency key in the path.

HTTP/SSE contract coverage (real RunRepository on tmp SQLite):

* direct run_id still resolves (regression);
* path-as-request_id resolves for snapshot and events;
* ``?request_id=`` resolves when the path id is unknown;
* both identifiers unknown → 404.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import api.chat_v2 as chat_v2_module
from agent_v2.runs import CheckpointsStore, ControlStore, RunRepository


def _repo(tmp_path: Path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


class _FakeMemory:
    def owns_conversation(self, subject_id: str, conversation_id: str) -> bool:
        return True


def _build_app(repo: RunRepository) -> FastAPI:
    app = FastAPI()
    app.include_router(chat_v2_module.chat_v2_router, prefix="/api/v2")
    app.state.agent_v2_runtime = SimpleNamespace(runs=repo, memory=_FakeMemory())
    return app


def _make_run(repo: RunRepository) -> tuple[str, str]:
    """Create one run; returns (run_id, request_id)."""
    run_id, _created = repo.create_run_idempotent(
        request_id="req-recovery-1",
        conversation_id="conv-recovery-1",
        profile_id="chat",
        user_message="看看恢复",
    )
    return run_id, "req-recovery-1"


@pytest.fixture()
def _subject(monkeypatch):
    from types import SimpleNamespace

    def _fake_subject(request):
        request.state.subject_id = "sub-1"
        return _FakeMemory(), SimpleNamespace(issued=False, token="tok", subject_id="sub-1")

    monkeypatch.setattr(chat_v2_module, "resolve_request_subject", _fake_subject)


def _get(app: FastAPI, path: str) -> Any:
    async def _run() -> Any:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)

    return asyncio.run(_run())


def test_snapshot_by_direct_run_id_still_works(tmp_path: Path, _subject):
    """Regression: the original run_id path keeps resolving (A05)."""
    repo = _repo(tmp_path)
    run_id, _request_id = _make_run(repo)
    response = _get(_build_app(repo), f"/api/v2/chat/snapshot/{run_id}")
    assert response.status_code == 200
    assert response.json()["run_id"] == run_id


def test_snapshot_resolves_path_request_id(tmp_path: Path, _subject):
    """AC4 parity with cancel: the path identifier may be a request_id (A05)."""
    repo = _repo(tmp_path)
    run_id, request_id = _make_run(repo)
    response = _get(_build_app(repo), f"/api/v2/chat/snapshot/{request_id}")
    assert response.status_code == 200
    assert response.json()["run_id"] == run_id


def test_snapshot_resolves_request_id_query_param(tmp_path: Path, _subject):
    """A05: ?request_id= rescues the refresh-before-first-frame case where
    the client has an empty/unknown path id."""
    repo = _repo(tmp_path)
    run_id, request_id = _make_run(repo)
    response = _get(
        _build_app(repo),
        f"/api/v2/chat/snapshot/-?request_id={request_id}",
    )
    assert response.status_code == 200
    assert response.json()["run_id"] == run_id


def test_events_resolve_request_id_fallback(tmp_path: Path, _subject):
    """A05: events replay resolves the same way and reports the real run_id."""
    repo = _repo(tmp_path)
    run_id, request_id = _make_run(repo)
    response = _get(
        _build_app(repo),
        f"/api/v2/chat/events/-?from_seq=0&request_id={request_id}",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == run_id
    assert body["from_seq"] == 0
    assert isinstance(body["events"], list)


def test_unknown_identifiers_still_404(tmp_path: Path, _subject):
    """A05: resolution failure keeps the 404 contract (no silent empty)."""
    repo = _repo(tmp_path)
    _make_run(repo)
    response = _get(
        _build_app(repo),
        "/api/v2/chat/snapshot/-?request_id=req-does-not-exist",
    )
    assert response.status_code == 404
