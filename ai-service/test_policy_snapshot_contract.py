"""BCDEF ticket 08 — admission-time execution policy snapshot.

The snapshot is asserted through the repository/admission seam and the
orchestration entry: content completeness, hash stability, idempotent
create/read, fail-closed missing/mismatch handling, config-mutation
isolation and one-run-one-snapshot. No real LLM or network.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import api.chat_v2 as chat_v2_module
from agent_v2 import profiles
from agent_v2 import run_orchestration as orchestration_module
from agent_v2.execution_contract import (
    EXECUTION_SCHEMA_VERSION,
    BoundaryError,
    ExecutionErrorCode,
    policy_hash,
)
from agent_v2.execution_policy import build_execution_policy
from agent_v2.profiles import (
    CapabilityProfile,
    ModelProfile,
    ProfileId,
    resolve_profile,
)
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from test_run_orchestration_chat import SUBJECT_ID, _FakeMemory, _repo, _settings


def _build_policy(settings: Any | None = None) -> dict[str, Any]:
    return build_execution_policy(resolve_profile("chat"), settings or _settings())


def _admit(repo: Any, request_id: str, *, settings: Any | None = None):
    return repo.admit(
        request_id=request_id,
        conversation_id=None,
        profile_id="chat",
        user_message="带什么装备？",
        policy=_build_policy(settings),
    )


def _orchestrator(
    repo: Any,
    kernel_calls: list[dict[str, Any]],
    *,
    settings: Any | None = None,
) -> RunOrchestrator:
    async def fake_kernel(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        kernel_calls.append(kwargs)
        return ("好的。", [])

    return RunOrchestrator(
        settings=settings or _settings(),
        runs_repo=repo,
        memory=_FakeMemory(),
        kernel_turn=fake_kernel,
    )


def _stream(orch: RunOrchestrator, request_id: str) -> list[Any]:
    async def run() -> list[Any]:
        return [
            item
            async for item in orch.stream_chat(
                ChatFlowRequest(message="带什么装备？", request_id=request_id),
                subject_id=SUBJECT_ID,
            )
        ]

    return asyncio.run(run())


def _snapshot_count(repo: Any) -> int:
    conn = sqlite3.connect(repo.control.db_path)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM policy_snapshots"
        ).fetchone()[0]
    finally:
        conn.close()


# ---- content + hash -------------------------------------------------------


def test_admission_freezes_complete_policy_with_stable_hash(tmp_path: Path):
    repo = _repo(tmp_path)
    result = _admit(repo, "req-policy-content")

    assert result.ok and result.execute
    assert result.policy_hash
    assert result.policy_schema_version == EXECUTION_SCHEMA_VERSION

    record = repo.get_policy_snapshot(result.run_id)
    assert record is not None
    assert record.policy_hash == result.policy_hash
    assert record.schema_version == EXECUTION_SCHEMA_VERSION
    assert record.source_version
    assert record.created_at

    policy = json.loads(record.policy_json)
    assert policy_hash(policy) == record.policy_hash
    assert policy == result.policy

    # Profile + model identity (name only; never secrets).
    assert policy["profile_id"] == "chat"
    assert policy["model"]["main_model"] == "deepseek-v4-flash"
    assert policy["model"]["provider_model"] == "test-model"
    assert policy["model"]["fast_model"] == "deepseek-v4-flash"
    # Budget.
    assert policy["budget"]["compaction_threshold"] == 24_000
    assert policy["budget"]["output_limit"] == 2_048
    # Capability inputs.
    assert set(policy["capability"]["allowed_tools"]) == {
        "hiking_knowledge_search",
        "gear_checklist",
        "risk_assessment",
        "memory_save",
        "memory_search",
        "focus_map",
    }
    assert policy["capability"]["allowed_mcp_servers"] is None
    assert isinstance(policy["capability"]["skills"], list)
    assert policy["capability"]["subagent_enabled"] is False
    # Lifecycle / effect / approval policy.
    assert policy["lifecycle"]["thoughts_enabled"] is True
    assert policy["effect_policy"]["approval_risk_threshold"] == "high"
    assert policy["effect_policy"]["tool_risk"]["sandbox_bash"] == "high"
    assert policy["approval_policy"]["requires_approval_at_or_above"] == "high"
    # Compatibility metadata + no secret.
    assert policy["compatibility"]["schema_version"] == EXECUTION_SCHEMA_VERSION
    assert EXECUTION_SCHEMA_VERSION in policy["compatibility"][
        "supported_schema_versions"
    ]
    assert "sk-test" not in record.policy_json
    assert "api_key" not in record.policy_json

    resolved = repo.resolve_policy_snapshot(result.run_id)
    assert resolved.policy_hash == record.policy_hash
    assert resolved.schema_version == EXECUTION_SCHEMA_VERSION
    assert resolved.profile().allowed_tools == resolve_profile("chat").allowed_tools

    snapshot = repo.get_snapshot(result.run_id)
    assert snapshot is not None
    assert snapshot.policy_hash == record.policy_hash
    assert snapshot.policy_schema_version == EXECUTION_SCHEMA_VERSION

    events = repo.get_events_since(run_id=result.run_id, from_seq=0)
    admission = events[0]["payload"]["admission"]
    assert admission["policy_hash"] == record.policy_hash
    assert admission["schema_version"] == EXECUTION_SCHEMA_VERSION


def test_snapshot_create_and_read_are_idempotent(tmp_path: Path):
    repo = _repo(tmp_path)
    first = _admit(repo, "req-policy-idem")
    second = _admit(repo, "req-policy-idem")

    assert first.run_id == second.run_id
    assert second.policy_hash == first.policy_hash
    assert _snapshot_count(repo) == 1

    one = repo.resolve_policy_snapshot(first.run_id)
    two = repo.resolve_policy_snapshot(first.run_id)
    assert one.policy_hash == two.policy_hash
    assert one.policy == two.policy

    # A duplicate call never rewrites the frozen content.
    before = repo.get_policy_snapshot(first.run_id)
    assert before is not None
    third = _admit(repo, "req-policy-idem")
    after = repo.get_policy_snapshot(first.run_id)
    assert third.replayed and not third.execute
    assert after is not None
    assert after.policy_json == before.policy_json
    assert after.created_at == before.created_at


def test_missing_snapshot_fails_closed(tmp_path: Path):
    repo = _repo(tmp_path)
    run_id, _created = repo.create_run_idempotent(
        request_id="req-policy-missing",
        conversation_id="conv-policy-missing",
        profile_id="chat",
        user_message="legacy",
    )
    with pytest.raises(BoundaryError) as excinfo:
        repo.resolve_policy_snapshot(run_id)
    assert excinfo.value.code is ExecutionErrorCode.POLICY_SNAPSHOT_MISSING
    assert excinfo.value.code.value == "policy_snapshot_missing"
    assert excinfo.value.recoverable is False


def test_hash_mismatch_fails_closed(tmp_path: Path):
    repo = _repo(tmp_path)
    result = _admit(repo, "req-policy-mismatch")

    conn = sqlite3.connect(repo.control.db_path)
    try:
        conn.execute(
            "UPDATE policy_snapshots SET policy_hash = ? WHERE run_id = ?",
            ("0" * 64, result.run_id),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(BoundaryError) as excinfo:
        repo.resolve_policy_snapshot(result.run_id)
    assert excinfo.value.code is ExecutionErrorCode.POLICY_INCOMPATIBLE


def test_tampered_policy_json_fails_closed(tmp_path: Path):
    repo = _repo(tmp_path)
    result = _admit(repo, "req-policy-tamper")
    record = repo.get_policy_snapshot(result.run_id)
    assert record is not None
    tampered = json.loads(record.policy_json)
    tampered["model"]["main_model"] = "some-other-model"

    conn = sqlite3.connect(repo.control.db_path)
    try:
        conn.execute(
            "UPDATE policy_snapshots SET policy_json = ? WHERE run_id = ?",
            (json.dumps(tampered), result.run_id),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(BoundaryError) as excinfo:
        repo.resolve_policy_snapshot(result.run_id)
    assert excinfo.value.code is ExecutionErrorCode.POLICY_INCOMPATIBLE


# ---- config mutation isolation -------------------------------------------


def test_config_mutation_after_admission_does_not_change_frozen_values(
    tmp_path: Path, monkeypatch,
):
    settings = _settings()
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []
    orch = _orchestrator(repo, calls, settings=settings)

    original_admit = repo.admit
    mutated_profile = CapabilityProfile(
        profile_id=ProfileId.CHAT,
        model=ModelProfile(
            profile_id=ProfileId.CHAT,
            main_model="mutated-model",
            fast_model="mutated-model",
            context_window=1,
            compaction_threshold=1,
            output_limit=1,
        ),
        allowed_tools=(),
        allowed_mcp_servers=None,
    )

    def admit_then_mutate(**kwargs: Any):
        result = original_admit(**kwargs)
        # Hot-update config AFTER the admission snapshot was frozen.
        settings.openai_model = "mutated-model"
        monkeypatch.setattr(
            profiles, "_PROFILES", {**profiles._PROFILES, ProfileId.CHAT: mutated_profile}
        )
        monkeypatch.setattr(
            orchestration_module,
            "resolve_profile",
            lambda _profile_id: (_ for _ in ()).throw(
                AssertionError("execution re-resolved the live profile")
            ),
        )
        return result

    repo.admit = admit_then_mutate  # type: ignore[method-assign]

    items = _stream(orch, "req-policy-freeze")

    assert items[-1].type == "done"
    assert len(calls) == 1
    frozen_settings = calls[0]["settings"]
    assert frozen_settings.openai_model == "test-model"  # admission-time value
    assert calls[0]["compaction_threshold"] == 24_000
    assert calls[0]["extra_tools"]  # frozen capability, not the emptied profile

    run = repo.control.get_run_by_request("req-policy-freeze")
    assert run is not None
    resolved = repo.resolve_policy_snapshot(run.run_id)
    assert resolved.policy["model"]["provider_model"] == "test-model"
    assert "hiking_knowledge_search" in resolved.profile().allowed_tools
    assert resolved.policy["budget"]["compaction_threshold"] == 24_000


def test_execution_fails_closed_when_snapshot_disappears(
    tmp_path: Path, monkeypatch,
):
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []
    orch = _orchestrator(repo, calls)

    original_admit = repo.admit

    def admit_then_break(**kwargs: Any):
        result = original_admit(**kwargs)

        def missing(_run_id: str):
            raise BoundaryError(
                ExecutionErrorCode.POLICY_SNAPSHOT_MISSING, "snapshot gone"
            )

        monkeypatch.setattr(repo, "resolve_policy_snapshot", missing)
        return result

    repo.admit = admit_then_break  # type: ignore[method-assign]

    items = _stream(orch, "req-policy-gone")

    assert [e.type for e in items] == ["run_started", "error"]
    assert items[-1].payload["code"] == "policy_snapshot_missing"
    assert calls == []
    run = repo.control.get_run_by_request("req-policy-gone")
    assert run is not None and run.status == "error"


def test_execution_fails_closed_on_hash_mismatch(tmp_path: Path, monkeypatch):
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []
    orch = _orchestrator(repo, calls)
    original_admit = repo.admit

    def admit_then_break(**kwargs: Any):
        result = original_admit(**kwargs)

        def incompatible(_run_id: str):
            raise BoundaryError(
                ExecutionErrorCode.POLICY_INCOMPATIBLE, "hash mismatch"
            )

        monkeypatch.setattr(repo, "resolve_policy_snapshot", incompatible)
        return result

    repo.admit = admit_then_break  # type: ignore[method-assign]

    items = _stream(orch, "req-policy-incompat")

    assert [e.type for e in items] == ["run_started", "error"]
    assert items[-1].payload["code"] == "policy_incompatible"
    assert calls == []


# ---- snapshot/replay visibility ------------------------------------------


def test_snapshot_endpoint_exposes_policy_identity(tmp_path: Path, monkeypatch):
    repo = _repo(tmp_path)
    result = _admit(repo, "req-policy-api")

    def fake_subject(request: Any) -> Any:
        request.state.subject_id = "sub-1"
        return _FakeMemory(), SimpleNamespace(
            issued=False, token="tok", subject_id="sub-1"
        )

    monkeypatch.setattr(chat_v2_module, "resolve_request_subject", fake_subject)
    app = FastAPI()
    app.include_router(chat_v2_module.chat_v2_router, prefix="/api/v2")
    app.state.agent_v2_runtime = SimpleNamespace(runs=repo, memory=_FakeMemory())

    async def fetch() -> Any:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(
                f"/api/v2/chat/snapshot/{result.run_id}"
            )

    response = asyncio.run(fetch())
    assert response.status_code == 200
    body = response.json()
    assert body["policy_hash"] == result.policy_hash
    assert body["policy_schema_version"] == EXECUTION_SCHEMA_VERSION


def test_replay_events_expose_policy_hash_and_schema(tmp_path: Path):
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []
    orch = _orchestrator(repo, calls)

    first = _stream(orch, "req-policy-replay")
    run = repo.control.get_run_by_request("req-policy-replay")
    assert run is not None
    record = repo.get_policy_snapshot(run.run_id)
    assert record is not None

    replay = _stream(orch, "req-policy-replay")
    assert [e.type for e in replay] == [e.type for e in first]
    admission = replay[0].payload["admission"]
    assert admission["policy_hash"] == record.policy_hash
    assert admission["schema_version"] == EXECUTION_SCHEMA_VERSION
    assert len(calls) == 1


def test_one_run_has_exactly_one_snapshot(tmp_path: Path):
    repo = _repo(tmp_path)
    result = _admit(repo, "req-policy-one")
    for _ in range(3):
        repo.resolve_policy_snapshot(result.run_id)
        repo.get_snapshot(result.run_id)
    assert _snapshot_count(repo) == 1
