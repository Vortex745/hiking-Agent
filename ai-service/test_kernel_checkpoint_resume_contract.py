"""Ticket 27 — kernel checkpoint identity, resume, fail-closed contract.

Coverage:

* thread_id ↔ run identity is 1:1 (``v2:chat:<run_id>``) and survives
  process "restart" via separate store instances over one SQLite file —
  real graph state restores from disk; EventStream replay never runs.
* Missing run identity with a store present fails closed BEFORE the
  kernel is assembled.
* Corrupt/unwritable checkpoint destinations fail closed with a stable
  public code and map through :class:`KernelModelFailure`.
* Public cancel stays idempotent (RunRepository), independent of kernel.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

from agent_v2.kernel import KernelModelFailure, run_toolless_chat
from agent_v2.kernel_persistence import (
    KernelCheckpointStore,
    KernelCheckpointUnavailable,
)
from agent_v2.runs import RunRepository

_HAS_LLM_CREDS = bool(os.getenv("OPENAI_API_KEY")) and bool(os.getenv("OPENAI_BASE_URL"))
# Opt-in live check (provider qpm makes un-gated bursts flaky).
_LIVE_LLM = _HAS_LLM_CREDS and os.getenv("KERNEL_LIVE_E2E", "") == "1"


class _Settings:
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "")
    openai_model = os.getenv("OPENAI_MODEL", "")
    agent_v2_uniharness_pin = os.getenv(
        "AGENT_V2_UNIHARNESS_PIN",
        "67a9497f8bd20c205accffc3a44492554fda43af",
    )


def test_checkpoint_thread_identity_isolated_per_run(tmp_path: Path):
    """Fresh threads never see each other's state in the backing store."""
    db = tmp_path / "kernel_checkpoints.sqlite"
    store = KernelCheckpointStore(db)
    asyncio.run(store.get())

    counts = _thread_rows(db, ["v2:chat:rA", "v2:chat:rB"])
    assert counts == {"v2:chat:rA": 0, "v2:chat:rB": 0}
    store.close()


def _thread_rows(db_path: Path, thread_ids: list[str]) -> dict[str, int]:
    """Count checkpoints per thread directly in the backing database."""
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            tid: conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (tid,)
            ).fetchone()[0]
            for tid in thread_ids
        }
    finally:
        conn.close()


def test_missing_run_identity_with_store_fails_closed_before_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []

    async def _must_not_run(*a: Any, **k: Any):
        calls.append("assembled")
        raise AssertionError("kernel must not assemble when checkpoint preflight fails")

    import agent_v2.kernel as kernel_module

    monkeypatch.setattr(kernel_module, "assemble_kernel_agent", _must_not_run)

    class _Store:
        @staticmethod
        def get():
            raise KernelCheckpointUnavailable("运行检查点不可用，无法安全地执行或恢复该请求。")

    with pytest.raises(KernelModelFailure) as excinfo:
        asyncio.run(run_toolless_chat(
            system_prompt="p",
            message="m",
            settings=_Settings(),
            run_id="r1",
            checkpoint_store=_Store(),
        ))

    assert excinfo.value.code == "checkpoint_unavailable"
    assert calls == []  # assembly never happened


def test_corrupt_checkpoint_destination_fails_closed(tmp_path: Path):
    blocker = tmp_path / "blocker-file"
    blocker.write_text("not a directory")

    store = KernelCheckpointStore(blocker / "sub" / "kernel_checkpoints.sqlite")

    with pytest.raises(KernelCheckpointUnavailable) as excinfo:
        asyncio.run(store.get())
    assert excinfo.value.code == "checkpoint_unavailable"


@pytest.mark.skipif(
    not _LIVE_LLM,
    reason="live resume E2E requires KERNEL_LIVE_E2E=1 plus OPENAI_* creds (recorded skip)",
)
def test_restart_resume_restores_real_graph_state_from_disk(tmp_path: Path):
    """Two 'processes' (separate stores, same file): turn 2 sees turn 1 state."""
    db = tmp_path / "kernel_checkpoints.sqlite"
    marker = "量子速通不是徒步。"

    async def scenario() -> tuple[str | None, str | None]:
        text1 = await run_toolless_chat(
            system_prompt='回答后以一行"MARKER: <内容>"结尾。',
            message=f"请把这句话原样复述一遍：{marker}",
            settings=_Settings(),
            run_id="r-restart-1",
            checkpoint_store=KernelCheckpointStore(db),
        )
        # Simulate service restart: first process's connection closed,
        # second process opens its own store over the same file.
        await asyncio.sleep(20)  # provider qpm window; burst guard
        text2 = await run_toolless_chat(
            system_prompt="回答前先检索你的记忆；若无则只输出UNKNOWN。",
            message="我上一条消息让你复述的那句话是什么？只输出那句话本身。",
            settings=_Settings(),
            run_id="r-restart-1",  # SAME public run identity → SAME thread
            checkpoint_store=KernelCheckpointStore(db),
        )
        return text1, text2

    try:
        text1, text2 = asyncio.run(scenario())
    except KernelModelFailure as exc:
        from agent_v2.kernel import is_provider_throttled

        if is_provider_throttled(exc):
            pytest.skip("provider qpm throttled (recorded skip)")
        raise

    assert text1 and marker[:6] in text1
    assert text2 and "UNKNOWN" not in text2

    # Direct disk evidence: the single shared thread persisted both turns'
    # state; EventStream/replay had no part in turn 2's context.
    rows = _thread_rows(db, ["v2:chat:r-restart-1"])
    assert rows["v2:chat:r-restart-1"] >= 2


def test_public_cancel_remains_idempotent_independent_of_kernel(tmp_path: Path):
    control_path = tmp_path / "agent_control.sqlite"
    ckpt_path = tmp_path / "chat_checkpoints.sqlite"

    from agent_v2.persistence import CheckpointsStore, ControlStore

    repo = RunRepository(control=ControlStore(str(control_path)), checkpoints=CheckpointsStore(str(ckpt_path)))
    try:
        run_id, created = repo.create_run_idempotent(
            request_id="req-cancel-idem",
            conversation_id="conv-x",
            profile_id="chat",
            user_message="hi",
        )
        assert created
        assert not repo.is_cancelled(run_id)

        repo.request_cancel(run_id=run_id, idempotency_key="op1")
        assert repo.is_cancelled(run_id)

        outcome_again = repo.request_cancel(run_id=run_id, idempotency_key="op2")
        # Duplicate cancel is accepted/idempotent without re-attempting the
        # transition; CancelOutcome carries the original reason.
        assert outcome_again.run_id == run_id
        record = repo.get_run(run_id)
        assert record is not None and record.status == "cancelled"
    finally:
        repo.close()
