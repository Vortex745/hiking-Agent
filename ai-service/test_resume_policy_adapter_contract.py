"""BCDEF ticket 09 — versioned resume adapters and incompatibility pause.

Two seams are asserted:

* the versioned adapter chain itself (pure, idempotent, metadata-bearing,
  duplicate-registration behavior);
* the resume execution boundary (``RunOrchestrator.stream_resume`` and the
  repository snapshot): legacy snapshots resume through the adapted policy,
  incompatible ones terminate as ``error`` BEFORE any model / effect /
  sandbox work, repeated resumes never repeat work, and snapshot/replay
  explain the reason + next action.

No real LLM, network or sandbox: kernel, computer factory, artifact export
and cleanup are injected fakes over a tmp SQLite repository.
"""

from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import api.chat_v2 as chat_v2_module
from agent_v2.approvals import compute_approval_id, compute_expiry
from agent_v2.execution_contract import (
    EXECUTION_SCHEMA_VERSION,
    BoundaryError,
    ContractViolation,
    ExecutionErrorCode,
    ReasonCode,
    policy_hash,
    register_adapter,
)
from agent_v2.execution_policy import build_execution_policy
from agent_v2.persistence import (
    CheckpointsStore,
    ControlStore,
    PolicySnapshotWrite,
)
from agent_v2.profiles import resolve_profile
from agent_v2.run_orchestration import RunOrchestrator
from agent_v2.runs import RunRepository
from agent_v2.policy_adapters import (
    NATIVE_POLICY_SCHEMA_VERSION,
    NEXT_ACTION_JANITOR,
    NEXT_ACTION_MANUAL,
    REASON_MISSING_CRITICAL_FIELD,
    REASON_POLICY_HASH_MISMATCH,
    REASON_UNSUPPORTED_SCHEMA_VERSION,
    LegacyPolicySnapshotV0Adapter,
    PolicyCompatibilityError,
    adapt_policy_snapshot,
    ensure_default_policy_adapters_registered,
)

TERMINALS = {"error", "done", "cancelled", "run.paused"}
LEGACY_VERSION = 0
ADAPTER_ID = "policy_snapshot_v0_to_v1"
FIXED_NOW = "2026-10-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Snapshot shapes
# ---------------------------------------------------------------------------


def _settings(**overrides: Any) -> Any:
    base = {
        "agent_v2_enabled": True,
        "openai_api_key": "sk-test",
        "openai_model": "current-model",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _native_policy(settings: Any | None = None) -> dict[str, Any]:
    return build_execution_policy(resolve_profile("artifact"), settings or _settings())


def _legacy_policy() -> dict[str, Any]:
    """Pre-ticket-08 minimal snapshot: no schema_version/compatibility."""
    return {
        "profile_id": "artifact",
        "model": {
            "provider_model": "legacy-model",
            "main_model": "legacy-main",
            "fast_model": "legacy-fast",
            "context_window": 32_000,
        },
        "budget": {"compaction_threshold": 12_345, "output_limit": 512},
        "capability": {
            "allowed_tools": ["generate_pdf"],
            "allowed_mcp_servers": None,
            "subagent_enabled": False,
        },
        "effect_policy": {
            "approval_risk_threshold": "high",
            "tool_risk": {"generate_pdf": "high"},
        },
        "approval_policy": {"requires_approval_at_or_above": "high"},
        "lifecycle": {
            "thoughts_enabled": False,
            "cowork_policy": "disabled",
        },
    }


def _json(policy: dict[str, Any]) -> str:
    return json.dumps(policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _record(
    policy: dict[str, Any],
    *,
    row_version: int,
    stored_hash: str | None = None,
    source_version: str = "legacy-test",
):
    return SimpleNamespace(
        run_id="run-adapter-1",
        policy_hash=policy_hash(policy) if stored_hash is None else stored_hash,
        schema_version=row_version,
        source_version=source_version,
        policy_json=_json(policy),
        compatibility_json="{}",
        created_at="2026-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Adapter chain seam
# ---------------------------------------------------------------------------


def test_legacy_adapter_is_pure_idempotent_and_reports_provenance():
    original = _legacy_policy()
    untouched = copy.deepcopy(original)
    record = _record(original, row_version=LEGACY_VERSION)

    first = adapt_policy_snapshot(record, now=FIXED_NOW)
    second = adapt_policy_snapshot(record, now=FIXED_NOW)

    # Purity: neither the record payload nor the caller's mapping changed.
    assert original == untouched
    assert json.loads(record.policy_json) == untouched

    # Idempotent: same adapted content and identity on every call.
    assert first.policy == second.policy
    assert first.policy_hash == second.policy_hash
    assert first.adapter_id == second.adapter_id == ADAPTER_ID

    # Version lift + preserved execution values (never current config).
    assert first.schema_version == NATIVE_POLICY_SCHEMA_VERSION
    assert first.source_schema_version == LEGACY_VERSION
    assert first.policy["schema_version"] == NATIVE_POLICY_SCHEMA_VERSION
    assert first.policy["model"]["main_model"] == "legacy-main"
    assert first.policy["budget"]["compaction_threshold"] == 12_345
    assert first.policy["capability"]["allowed_tools"] == ["generate_pdf"]
    assert first.policy["effect_policy"]["tool_risk"] == {"generate_pdf": "high"}

    provenance = first.provenance_payload()
    assert provenance["source_schema_version"] == LEGACY_VERSION
    assert provenance["target_schema_version"] == NATIVE_POLICY_SCHEMA_VERSION
    assert provenance["adapter_id"] == ADAPTER_ID
    assert provenance["adapted_at"] == FIXED_NOW
    assert provenance["adapter_chain"] == [
        {
            "from_version": LEGACY_VERSION,
            "to_version": NATIVE_POLICY_SCHEMA_VERSION,
            "adapter_id": ADAPTER_ID,
        }
    ]
    assert provenance["source_policy_hash"] == record.policy_hash
    assert provenance["target_policy_hash"] == policy_hash(first.policy)


def test_native_snapshot_is_handled_without_translation():
    policy = _native_policy()
    record = _record(policy, row_version=EXECUTION_SCHEMA_VERSION)

    adapted = adapt_policy_snapshot(record, now=FIXED_NOW)

    assert adapted.policy == policy
    assert adapted.policy_hash == record.policy_hash
    assert adapted.schema_version == EXECUTION_SCHEMA_VERSION
    assert adapted.source_schema_version == EXECUTION_SCHEMA_VERSION
    assert adapted.adapter_id == "native"
    assert adapted.provenance_payload()["adapter_chain"] == []


def test_duplicate_adapter_registration_conflicts_and_ensure_is_idempotent():
    ensure_default_policy_adapters_registered()
    ensure_default_policy_adapters_registered()  # idempotent module bootstrap

    with pytest.raises(ContractViolation) as excinfo:
        register_adapter(LegacyPolicySnapshotV0Adapter())
    assert excinfo.value.code is ReasonCode.ADAPTER_CONFLICT


def test_unknown_schema_version_is_a_stable_policy_incompatible_error():
    record = _record(_native_policy(), row_version=99)

    with pytest.raises(PolicyCompatibilityError) as excinfo:
        adapt_policy_snapshot(record)
    error = excinfo.value
    assert error.code is ExecutionErrorCode.POLICY_INCOMPATIBLE
    assert error.reason == REASON_UNSUPPORTED_SCHEMA_VERSION
    assert error.next_action == NEXT_ACTION_MANUAL
    payload = error.to_payload()
    assert payload["code"] == "policy_incompatible"
    assert payload["reason"] == "unsupported_schema_version"
    assert payload["next_action"] == "manual"
    assert payload["recoverable"] is False


def test_missing_critical_field_fails_with_janitor_action():
    policy = _legacy_policy()
    del policy["budget"]
    record = _record(policy, row_version=LEGACY_VERSION)

    with pytest.raises(PolicyCompatibilityError) as excinfo:
        adapt_policy_snapshot(record)
    error = excinfo.value
    assert error.reason == REASON_MISSING_CRITICAL_FIELD
    assert error.next_action == NEXT_ACTION_JANITOR
    assert "budget" in (error.detail or "")
    assert error.to_payload()["next_action"] == "janitor"


def test_policy_hash_mismatch_fails_closed():
    record = _record(
        _native_policy(),
        row_version=EXECUTION_SCHEMA_VERSION,
        stored_hash="0" * 64,
    )

    with pytest.raises(PolicyCompatibilityError) as excinfo:
        adapt_policy_snapshot(record)
    assert excinfo.value.reason == REASON_POLICY_HASH_MISMATCH
    assert excinfo.value.next_action == NEXT_ACTION_MANUAL


def test_native_content_without_schema_version_is_missing_critical_field():
    policy = _native_policy()
    del policy["schema_version"]
    record = _record(policy, row_version=EXECUTION_SCHEMA_VERSION)

    with pytest.raises(PolicyCompatibilityError) as excinfo:
        adapt_policy_snapshot(record)
    assert excinfo.value.reason == REASON_MISSING_CRITICAL_FIELD
    assert "schema_version" in (excinfo.value.detail or "")


def test_unknown_schema_version_is_not_reported_as_missing_snapshot():
    """A stored-but-unsupported version is incompatible, not 'missing'."""
    record = _record(_legacy_policy(), row_version=7)
    with pytest.raises(BoundaryError) as excinfo:
        adapt_policy_snapshot(record)
    assert excinfo.value.code is ExecutionErrorCode.POLICY_INCOMPATIBLE


# ---------------------------------------------------------------------------
# Resume boundary seam
# ---------------------------------------------------------------------------


class _FakeComputer:
    sandbox_type = "local_native"

    def __init__(self) -> None:
        self.cleaned: list[str] = []
        self.pdfs: list[tuple[str, str, str]] = []

    def generate_pdf(self, workspace_path: str, content: str, filename: str) -> str:
        self.pdfs.append((workspace_path, content, filename))
        return filename

    def cleanup(self, workspace_path: str) -> dict[str, int]:
        self.cleaned.append(workspace_path)
        return {"files_deleted": 0}


def _repo(tmp_path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _insert_policy(
    repo: RunRepository,
    run_id: str,
    policy: dict[str, Any],
    *,
    row_version: int,
    stored_hash: str | None = None,
) -> None:
    write = PolicySnapshotWrite(
        policy_json=_json(policy),
        policy_hash=policy_hash(policy) if stored_hash is None else stored_hash,
        schema_version=row_version,
        source_version="test-writer",
        compatibility_json="{}",
        created_at="2026-01-01T00:00:00+00:00",
    )
    repo.control.insert_policy_snapshot(run_id=run_id, policy=write)


def _counters() -> dict[str, Any]:
    return {
        "kernel": 0,
        "sandbox": 0,
        "export": 0,
        "cleanup": 0,
        "kernel_settings": [],
    }


def _orchestrator(
    repo: RunRepository,
    counters: dict[str, Any],
    *,
    openai_model: str = "current-model",
):
    computer = _FakeComputer()

    async def fake_kernel(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        counters["kernel"] += 1
        counters["kernel_settings"].append(kwargs.get("settings"))
        return ("# 徒步计划\n占位正文。", [])

    def fake_export(**kwargs: Any) -> dict[str, Any]:
        counters["export"] += 1
        return {
            "artifact_id": "art-1",
            "public_filename": "hiking-plan.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 10,
            "sha256": "deadbeef",
            "expires_at": "2026-11-01T00:00:00+00:00",
        }

    def fake_factory(*args: Any, **kwargs: Any) -> _FakeComputer:
        counters["sandbox"] += 1
        return computer

    def fake_cleanup(*args: Any, **kwargs: Any) -> None:
        counters["cleanup"] += 1

    return (
        RunOrchestrator(
            settings=_settings(openai_model=openai_model),
            runs_repo=repo,
            kernel_turn=fake_kernel,
            artifact_export=fake_export,
            computer_factory=fake_factory,
            sandbox_cleanup=fake_cleanup,
            gate_factory=None,
        ),
        computer,
    )


def _paused_artifact_run(
    repo: RunRepository,
    request_id: str,
    *,
    policy: dict[str, Any] | None = None,
    row_version: int = EXECUTION_SCHEMA_VERSION,
    stored_hash: str | None = None,
    seed_policy: bool = True,
) -> str:
    run_id, _created = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=f"conv-{request_id}",
        profile_id="artifact",
        user_message="规划杭州两日徒步并出 PDF 计划",
    )
    if seed_policy:
        _insert_policy(
            repo,
            run_id,
            policy if policy is not None else _native_policy(),
            row_version=row_version,
            stored_hash=stored_hash,
        )
    repo.create_sandbox_lease(
        run_id=run_id,
        sandbox_type="local_native",
        workspace_path=f"/ws/{run_id}",
    )
    approval_id = compute_approval_id(
        run_id=run_id, tool_name="generate_pdf", args={"filename": "hiking-plan.pdf"}
    )
    repo.insert_approval(
        approval_id=approval_id,
        run_id=run_id,
        conversation_id=f"conv-{request_id}",
        tool_name="generate_pdf",
        args_canonical='{"filename": "hiking-plan.pdf"}',
        risk_level="high",
        impact="生成徒步计划 PDF 并导出到宿主存储",
        created_at=_now_iso(),
        expires_at=compute_expiry(_now_iso()),
    )
    repo.resolve_approval(
        approval_id=approval_id,
        status="approved",
        decision_request_id=f"dec-{request_id}",
    )
    return run_id


def _stream_resume(orch: RunOrchestrator, run_id: str, from_seq: int = 0) -> list[Any]:
    async def run() -> list[Any]:
        return [
            item
            async for item in orch.stream_resume(run_id=run_id, from_seq=from_seq)
        ]

    return asyncio.run(run())


# ---- native current-version resume ----------------------------------------


def test_native_resume_uses_frozen_policy_not_current_config(tmp_path):
    repo = _repo(tmp_path)
    frozen = _native_policy(_settings(openai_model="frozen-model"))
    run_id = _paused_artifact_run(repo, "req-native", policy=frozen)
    counters = _counters()
    # Current config changes AFTER admission; resume must not see it.
    orch, computer = _orchestrator(repo, counters, openai_model="mutated-model")

    items = _stream_resume(orch, run_id)

    assert items[-1].type == "done"
    assert counters["kernel"] == 1
    assert counters["sandbox"] == 1
    assert counters["export"] == 1
    assert counters["cleanup"] == 1
    assert counters["kernel_settings"][0].openai_model == "frozen-model"
    assert len(computer.pdfs) == 1
    assert all(e.run_id == run_id for e in items)
    assert all(e.conversation_id == "conv-req-native" for e in items)

    snapshot = repo.get_snapshot(run_id)
    assert snapshot is not None
    assert snapshot.policy_hash == policy_hash(frozen)
    assert snapshot.policy_schema_version == EXECUTION_SCHEMA_VERSION
    assert snapshot.policy_failure is None
    assert snapshot.policy_provenance is not None
    assert snapshot.policy_provenance["compatible"] is True
    assert snapshot.policy_provenance["adapter_id"] == "native"
    assert snapshot.policy_provenance["source_schema_version"] == (
        EXECUTION_SCHEMA_VERSION
    )


# ---- legacy snapshot resume through the adapter ---------------------------


def test_legacy_snapshot_resume_adapts_and_preserves_original_mapping(tmp_path):
    repo = _repo(tmp_path)
    legacy = _legacy_policy()
    stored_copy = copy.deepcopy(legacy)
    run_id = _paused_artifact_run(
        repo, "req-legacy", policy=legacy, row_version=LEGACY_VERSION
    )
    record_before = repo.get_policy_snapshot(run_id)
    assert record_before is not None
    counters = _counters()
    orch, computer = _orchestrator(repo, counters, openai_model="current-model")

    items = _stream_resume(orch, run_id)

    types = [e.type for e in items]
    # Adapted provenance is replayable BEFORE the continuation.
    assert types[0] == "progress"
    provenance = items[0].payload
    assert provenance["stage"] == "policy.adapted"
    assert provenance["source_schema_version"] == LEGACY_VERSION
    assert provenance["target_schema_version"] == NATIVE_POLICY_SCHEMA_VERSION
    assert provenance["adapter_id"] == ADAPTER_ID
    assert provenance["adapter_chain"] == [
        {
            "from_version": LEGACY_VERSION,
            "to_version": NATIVE_POLICY_SCHEMA_VERSION,
            "adapter_id": ADAPTER_ID,
        }
    ]
    assert types[-1] == "done"

    # Resume executed with the ADAPTED frozen values, never current config.
    assert counters["kernel"] == 1
    assert counters["sandbox"] == 1
    frozen_settings = counters["kernel_settings"][0]
    assert frozen_settings.openai_model == "legacy-model"
    assert frozen_settings.agent_v2_thoughts_enabled is False
    assert len(computer.pdfs) == 1
    assert all(e.run_id == run_id for e in items)

    # The stored row is untouched — adaptation is a pure projection.
    record_after = repo.get_policy_snapshot(run_id)
    assert record_after is not None
    assert record_after.policy_json == record_before.policy_json
    assert record_after.policy_hash == record_before.policy_hash
    assert record_after.schema_version == LEGACY_VERSION
    assert json.loads(record_after.policy_json) == stored_copy

    # Adapted policy identity is exposed by the repository.
    resolved = repo.resolve_policy_snapshot(run_id)
    assert resolved.schema_version == NATIVE_POLICY_SCHEMA_VERSION
    assert resolved.source_schema_version == LEGACY_VERSION
    assert resolved.adapter_id == ADAPTER_ID
    assert resolved.original_policy_hash == record_before.policy_hash
    assert resolved.policy["model"]["main_model"] == "legacy-main"
    assert resolved.policy["budget"]["compaction_threshold"] == 12_345
    assert resolved.policy["budget"]["output_limit"] == 512
    assert resolved.policy["capability"]["allowed_tools"] == ["generate_pdf"]
    assert resolved.policy["effect_policy"]["approval_risk_threshold"] == "high"

    # Snapshot keeps the stored identity AND the adapter provenance.
    snapshot = repo.get_snapshot(run_id)
    assert snapshot is not None
    assert snapshot.policy_hash == record_before.policy_hash
    assert snapshot.policy_schema_version == LEGACY_VERSION
    assert snapshot.policy_failure is None
    assert snapshot.policy_provenance is not None
    assert snapshot.policy_provenance["source_schema_version"] == LEGACY_VERSION
    assert (
        snapshot.policy_provenance["target_schema_version"]
        == NATIVE_POLICY_SCHEMA_VERSION
    )


# ---- incompatible snapshots: error before any execution -------------------


@pytest.mark.parametrize(
    ("case", "expected_code", "expected_reason", "expected_action"),
    [
        (
            "missing_field",
            "policy_incompatible",
            REASON_MISSING_CRITICAL_FIELD,
            NEXT_ACTION_JANITOR,
        ),
        (
            "unknown_version",
            "policy_incompatible",
            REASON_UNSUPPORTED_SCHEMA_VERSION,
            NEXT_ACTION_MANUAL,
        ),
        (
            "hash_mismatch",
            "policy_incompatible",
            REASON_POLICY_HASH_MISMATCH,
            NEXT_ACTION_MANUAL,
        ),
        (
            "missing_snapshot",
            "policy_snapshot_missing",
            "snapshot_missing",
            NEXT_ACTION_MANUAL,
        ),
    ],
)
def test_incompatible_snapshot_stops_before_model_effect_or_sandbox(
    tmp_path, case, expected_code, expected_reason, expected_action
):
    repo = _repo(tmp_path)
    if case == "missing_field":
        policy = _legacy_policy()
        del policy["budget"]
        run_id = _paused_artifact_run(
            repo, f"req-{case}", policy=policy, row_version=LEGACY_VERSION
        )
    elif case == "unknown_version":
        run_id = _paused_artifact_run(
            repo, f"req-{case}", policy=_native_policy(), row_version=99
        )
    elif case == "hash_mismatch":
        run_id = _paused_artifact_run(
            repo,
            f"req-{case}",
            policy=_native_policy(),
            row_version=EXECUTION_SCHEMA_VERSION,
            stored_hash="0" * 64,
        )
    else:
        run_id = _paused_artifact_run(repo, f"req-{case}", seed_policy=False)

    counters = _counters()
    orch, computer = _orchestrator(repo, counters, openai_model="current-model")

    items = _stream_resume(orch, run_id)

    assert items[-1].type == "error"
    error = items[-1].payload
    assert error["code"] == expected_code
    assert error["reason"] == expected_reason
    assert error["next_action"] == expected_action
    assert error["recoverable"] is False

    # ZERO model / effect / sandbox / export / cleanup work.
    assert counters["kernel"] == 0
    assert counters["sandbox"] == 0
    assert counters["export"] == 0
    assert counters["cleanup"] == 0
    assert computer.pdfs == []
    assert repo.list_effects_for_run(run_id) == []

    run = repo.get_run(run_id)
    assert run is not None and run.status == "error"

    # No current default leaked into the wire.
    assert "current-model" not in json.dumps(
        [e.payload for e in items], ensure_ascii=False
    )

    # Snapshot explains the failure + next action.
    snapshot = repo.get_snapshot(run_id)
    assert snapshot is not None
    assert snapshot.policy_failure is not None
    assert snapshot.policy_failure["reason"] == expected_reason
    assert snapshot.policy_failure["next_action"] == expected_action
    assert snapshot.policy_provenance is None

    # Replay explains the failure + next action.
    stored = repo.get_events_since(run_id=run_id, from_seq=0)
    assert stored[-1]["type"] == "error"
    assert stored[-1]["payload"]["reason"] == expected_reason
    assert stored[-1]["payload"]["next_action"] == expected_action


def test_repeated_incompatible_resume_keeps_work_at_zero(tmp_path):
    repo = _repo(tmp_path)
    policy = _legacy_policy()
    del policy["capability"]
    run_id = _paused_artifact_run(
        repo, "req-repeat-incompat", policy=policy, row_version=LEGACY_VERSION
    )
    counters = _counters()
    orch, _computer = _orchestrator(repo, counters)

    first = _stream_resume(orch, run_id)
    assert first[-1].type == "error"
    first_events = repo.get_events_since(run_id=run_id, from_seq=0)

    second = _stream_resume(orch, run_id)
    third = _stream_resume(orch, run_id, from_seq=0)
    second_events = repo.get_events_since(run_id=run_id, from_seq=0)

    # Replay-only after the first failure: no new events, no new work.
    assert [e.sequence for e in second] == [e["sequence"] for e in first_events]
    assert [e.sequence for e in third] == [e["sequence"] for e in first_events]
    assert len(second_events) == len(first_events)
    assert counters["kernel"] == 0
    assert counters["sandbox"] == 0
    assert counters["export"] == 0
    assert counters["cleanup"] == 0


def test_repeated_legacy_resume_does_not_repeat_model_effect_or_sandbox(tmp_path):
    repo = _repo(tmp_path)
    run_id = _paused_artifact_run(
        repo, "req-repeat-legacy", policy=_legacy_policy(), row_version=LEGACY_VERSION
    )
    counters = _counters()
    orch, computer = _orchestrator(repo, counters)

    first = _stream_resume(orch, run_id)
    assert first[-1].type == "done"
    events_after_first = repo.get_events_since(run_id=run_id, from_seq=0)
    counters_after_first = dict(counters)

    second = _stream_resume(orch, run_id, from_seq=0)
    events_after_second = repo.get_events_since(run_id=run_id, from_seq=0)

    assert second[-1].type == "done"
    assert [e.sequence for e in second] == [e.sequence for e in first]
    assert len(events_after_second) == len(events_after_first)
    assert counters["kernel"] == counters_after_first["kernel"] == 1
    assert counters["sandbox"] == counters_after_first["sandbox"] == 1
    assert counters["export"] == counters_after_first["export"] == 1
    assert counters["cleanup"] == counters_after_first["cleanup"] == 1
    assert len(computer.pdfs) == 1
    assert len([e for e in second if e.type == "artifact.available"]) == 1


# ---- HTTP snapshot exposure (additive) ------------------------------------


class _FakeMemory:
    def owns_conversation(self, subject_id: str, conversation_id: str) -> bool:
        return True


def test_snapshot_endpoint_exposes_adapter_provenance(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    run_id = _paused_artifact_run(
        repo, "req-api-legacy", policy=_legacy_policy(), row_version=LEGACY_VERSION
    )

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
            return await client.get(f"/api/v2/chat/snapshot/{run_id}")

    response = asyncio.run(fetch())
    assert response.status_code == 200
    body = response.json()
    assert body["policy_provenance"]["source_schema_version"] == LEGACY_VERSION
    assert body["policy_provenance"]["target_schema_version"] == (
        NATIVE_POLICY_SCHEMA_VERSION
    )
    assert body["policy_provenance"]["adapter_id"] == ADAPTER_ID
    assert body["policy_failure"] is None
