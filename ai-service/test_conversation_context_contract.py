"""Ticket 05 — bounded server-side conversation history (spec T07 + T08).

Defect pinned here: a follow-up turn ("改成下周", "再加一个人") reached the
model with ONLY the current user message — the conversation's stored
constraints never entered the next kernel turn. This file pins the fixed
contract:

* T07 — the recent-completed-history query is server-side, keyed by the
  current run's conversation_id, built only from user/assistant messages of
  COMPLETED runs, excluding the current run and every running / failed /
  cancelled / internal turn, in stable true insertion order (AC1/AC2);
* T07 — the window keeps at most 20 messages AND at most 8000 body chars,
  trimmed from the oldest by COMPLETE turns: a turn is never split, an
  over-long single turn is omitted whole (AC3);
* T07 — the existing snapshot query semantics stay untouched; no history
  behaves as a fresh conversation; a storage failure surfaces as the
  explicit ``context_unavailable`` terminal, never a silent fresh
  conversation (AC4);
* T08 — both kernel entry constructions receive the SAME history and the
  current user message EXACTLY once; thread_id stays run-scoped
  (``v2:chat:<run_id>``) (AC5);
* E2E — date/headcount follow-ups, latest-condition updates, cross-run
  continuation, service restart, char boundaries and concurrent unfinished
  turns; same request_id never duplicates the model call; a foreign subject
  with the same conversation id is rejected; failures produce an explicit
  terminal (AC6);
* T08 — the actual kernel input order is proven with a deterministic agent
  double, and an old un-grounded assistant fact sitting in history can NOT
  bypass Ticket 02's this-turn evidence gate (AC7).

Run (ai-service working directory):
    E:/miniconda3/python.exe -B -m pytest -q test_conversation_context_contract.py
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.runs import RunRepository

TERMINALS = {"error", "done", "cancelled"}

CONVERSATION = "conv-ctx-1"


def _repo(tmp_path: Path, name: str = "") -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / f"control{name}.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / f"ckpt{name}.sqlite")),
    )


def _seed_turn(
    repo: RunRepository,
    *,
    request_id: str,
    user_msg: str,
    assistant_msg: str,
    conversation_id: str = CONVERSATION,
) -> str:
    """Create a run and persist both sides (run stays ``running``)."""
    run_id, _ = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=conversation_id,
        profile_id="chat",
        user_message=user_msg,
    )
    repo.checkpoints.append_message(
        conversation_id=conversation_id,
        run_id=run_id,
        role="assistant",
        content=assistant_msg,
        sequence=2,
    )
    return run_id


def _completed_turn(
    repo: RunRepository,
    *,
    request_id: str,
    user_msg: str,
    assistant_msg: str,
    conversation_id: str = CONVERSATION,
) -> str:
    """Create a run, persist both sides, mark the run ``done``."""
    run_id = _seed_turn(
        repo,
        request_id=request_id,
        user_msg=user_msg,
        assistant_msg=assistant_msg,
        conversation_id=conversation_id,
    )
    repo.mark_run_done(run_id)
    return run_id


# ---- AC1/AC2: only completed user/assistant turns, true insertion order ------


def test_history_returns_only_completed_turns_in_true_insertion_order(tmp_path: Path):
    repo = _repo(tmp_path)
    _completed_turn(
        repo,
        request_id="req-h1",
        user_msg="10月20日出发，两个人",
        assistant_msg="好的，已记录10月20日出发、两人同行。",
    )
    _completed_turn(
        repo,
        request_id="req-h2",
        user_msg="再补充一份装备清单",
        assistant_msg="装备清单如下……",
    )

    history = repo.get_recent_completed_history(conversation_id=CONVERSATION)

    assert history == [
        {"role": "user", "content": "10月20日出发，两个人"},
        {"role": "assistant", "content": "好的，已记录10月20日出发、两人同行。"},
        {"role": "user", "content": "再补充一份装备清单"},
        {"role": "assistant", "content": "装备清单如下……"},
    ]


def test_history_excludes_the_current_run(tmp_path: Path):
    repo = _repo(tmp_path)
    _completed_turn(
        repo,
        request_id="req-old",
        user_msg="上次说10月20日出发",
        assistant_msg="已记录。",
    )
    # The current run is completed here on purpose so the identity rule is
    # exercised on its own — a running current run is covered separately.
    current_run_id = _completed_turn(
        repo,
        request_id="req-now",
        user_msg="改成下周",
        assistant_msg="好的。",
    )

    history = repo.get_recent_completed_history(
        conversation_id=CONVERSATION, current_run_id=current_run_id
    )

    assert [entry["content"] for entry in history] == [
        "上次说10月20日出发",
        "已记录。",
    ]


def test_history_excludes_running_failed_and_cancelled_turns(tmp_path: Path):
    repo = _repo(tmp_path)
    _completed_turn(
        repo,
        request_id="req-ok",
        user_msg="已完成的回合",
        assistant_msg="已完成的回答。",
    )
    running_id, _ = repo.create_run_idempotent(
        request_id="req-running",
        conversation_id=CONVERSATION,
        profile_id="chat",
        user_message="运行中的回合",
    )
    failed_id = _seed_turn(
        repo,
        request_id="req-failed",
        user_msg="失败的回合",
        assistant_msg="失败的回答。",
    )
    repo.mark_run_error(failed_id)
    cancelled_id = _seed_turn(
        repo,
        request_id="req-cancelled",
        user_msg="取消的回合",
        assistant_msg="取消的回答。",
    )
    repo.request_cancel(run_id=cancelled_id, idempotency_key="cancel-ctx-1")

    history = repo.get_recent_completed_history(conversation_id=CONVERSATION)

    contents = [entry["content"] for entry in history]
    assert contents == ["已完成的回合", "已完成的回答。"]
    assert "运行中的回合" not in contents
    assert "失败的回合" not in contents
    assert "取消的回合" not in contents


def test_history_orders_by_true_insertion_even_with_identical_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Stable true insertion order: same ``created_at`` must not reorder."""
    import agent_v2.persistence as persistence_module

    frozen = "2026-10-15T00:00:00+00:00"
    monkeypatch.setattr(persistence_module, "_now_iso", lambda: frozen)
    repo = _repo(tmp_path)
    _completed_turn(
        repo, request_id="req-ts1", user_msg="第一条", assistant_msg="回答一"
    )
    _completed_turn(
        repo, request_id="req-ts2", user_msg="第二条", assistant_msg="回答二"
    )

    history = repo.get_recent_completed_history(conversation_id=CONVERSATION)

    assert [entry["content"] for entry in history] == [
        "第一条",
        "回答一",
        "第二条",
        "回答二",
    ]


def test_history_never_includes_tool_or_approval_messages(tmp_path: Path):
    """Internal tool/approval rows are unreadable through the history seam."""
    repo = _repo(tmp_path)
    run_id = _completed_turn(
        repo,
        request_id="req-internal",
        user_msg="查一下天气",
        assistant_msg="已查询。",
    )
    conn = sqlite3.connect(str(tmp_path / "ckpt.sqlite"), isolation_level=None)
    try:
        conn.execute(
            "INSERT INTO messages (message_id, conversation_id, run_id, role, "
            "content, sequence, created_at) VALUES "
            "('m-tool', ?, ?, 'tool', 'sandbox_bash 内部记录', 1, '2026-10-15T00:00:00+00:00')",
            (CONVERSATION, run_id),
        )
        conn.execute(
            "INSERT INTO messages (message_id, conversation_id, run_id, role, "
            "content, sequence, created_at) VALUES "
            "('m-approval', ?, ?, 'approval', '待批准 Bash', 1, '2026-10-15T00:00:00+00:00')",
            (CONVERSATION, run_id),
        )
    finally:
        conn.close()

    history = repo.get_recent_completed_history(conversation_id=CONVERSATION)

    assert [entry["role"] for entry in history] == ["user", "assistant"]
    assert all("内部记录" not in entry["content"] for entry in history)
    assert all("待批准" not in entry["content"] for entry in history)


def test_history_is_scoped_to_one_conversation(tmp_path: Path):
    repo = _repo(tmp_path)
    _completed_turn(
        repo, request_id="req-c1", user_msg="会话一的约束", assistant_msg="好的。"
    )
    _completed_turn(
        repo,
        request_id="req-c2",
        user_msg="会话二的约束",
        assistant_msg="好的。",
        conversation_id="conv-other",
    )

    history = repo.get_recent_completed_history(conversation_id=CONVERSATION)

    assert [entry["content"] for entry in history] == ["会话一的约束", "好的。"]


# ---- AC3: bounded window, trimmed from the oldest by complete turns ----------


def test_history_window_keeps_at_most_twenty_messages_by_whole_turns(tmp_path: Path):
    repo = _repo(tmp_path)
    for index in range(12):
        _completed_turn(
            repo,
            request_id=f"req-win-{index}",
            user_msg=f"约束{index}",
            assistant_msg=f"回答{index}",
        )

    history = repo.get_recent_completed_history(conversation_id=CONVERSATION)

    assert len(history) == 20
    # Newest 10 complete turns; the oldest 2 turns are gone whole.
    assert history[0] == {"role": "user", "content": "约束2"}
    assert history[1] == {"role": "assistant", "content": "回答2"}
    assert history[-1] == {"role": "assistant", "content": "回答11"}
    assert [entry["content"] for entry in history].count("约束0") == 0
    assert [entry["content"] for entry in history].count("回答1") == 0


def test_history_window_respects_char_budget_by_complete_turns(tmp_path: Path):
    repo = _repo(tmp_path)
    # Each turn is 6000 chars (3000 + 3000). 8000 allows the newest turn
    # (6000) but not a second one (12000), so older turns are omitted whole.
    for index in range(3):
        _completed_turn(
            repo,
            request_id=f"req-char-{index}",
            user_msg=f"U{index}-" + "甲" * 2997,
            assistant_msg=f"A{index}-" + "乙" * 2997,
        )

    history = repo.get_recent_completed_history(conversation_id=CONVERSATION)

    assert len(history) == 2  # exactly the newest complete turn
    assert history[0]["content"].startswith("U2-")
    assert history[1]["content"].startswith("A2-")


def test_history_char_budget_includes_content_exactly_up_to_the_limit(tmp_path: Path):
    repo = _repo(tmp_path)
    # Oldest turn would fit on its own; the newest turn fills the budget to
    # exactly 8000, so the older turn must be dropped whole.
    _completed_turn(
        repo,
        request_id="req-exact-old",
        user_msg="旧" * 4000,
        assistant_msg="旧答" * 2000,
    )
    _completed_turn(
        repo,
        request_id="req-exact-new",
        user_msg="新" * 4000,
        assistant_msg="新答" * 2000,
    )

    history = repo.get_recent_completed_history(conversation_id=CONVERSATION)

    assert len(history) == 2
    assert history[0]["content"] == "新" * 4000
    assert history[1]["content"] == "新答" * 2000


def test_oversized_newest_turn_is_omitted_whole_never_truncated(tmp_path: Path):
    """A single over-long turn is missed as a whole — never half a fact."""
    repo = _repo(tmp_path)
    _completed_turn(
        repo, request_id="req-big-old", user_msg="旧约束", assistant_msg="旧答复"
    )
    _completed_turn(
        repo,
        request_id="req-big-new",
        user_msg="超长" * 4500,  # 9000 chars > 8000
        assistant_msg="好的。",
    )

    history = repo.get_recent_completed_history(conversation_id=CONVERSATION)

    # Nothing of the oversized newest turn (and therefore nothing older,
    # which could only outrank it) enters the window.
    assert history == []


# ---- AC4: snapshot semantics preserved, no history = fresh -------------------


def test_no_completed_history_returns_empty_list(tmp_path: Path):
    repo = _repo(tmp_path)
    repo.create_run_idempotent(
        request_id="req-fresh",
        conversation_id=CONVERSATION,
        profile_id="chat",
        user_message="第一句话",
    )
    assert repo.get_recent_completed_history(conversation_id=CONVERSATION) == []
    assert repo.get_recent_completed_history(conversation_id="") == []


def test_snapshot_query_semantics_stay_unchanged(tmp_path: Path):
    """The UI snapshot keeps its OLDEST-first, complete-conversation, 100-row
    semantics — the new windowed history read must not alter it."""
    repo = _repo(tmp_path)
    for index in range(12):
        _completed_turn(
            repo,
            request_id=f"req-snap-{index}",
            user_msg=f"约束{index}",
            assistant_msg=f"回答{index}",
        )
    run_id = repo.control.get_run_by_request("req-snap-0").run_id

    snapshot = repo.get_snapshot(run_id)

    assert snapshot is not None
    # Snapshot: full conversation from the oldest message (all 24 rows).
    assert len(snapshot.messages) == 24
    assert snapshot.messages[0]["content"] == "约束0"
    assert snapshot.messages[-1]["content"] == "回答11"
    # Windowed history: newest 20 messages only.
    history = repo.get_recent_completed_history(conversation_id=CONVERSATION)
    assert len(history) == 20
    assert history[0]["content"] == "约束2"


def test_completed_history_reads_the_same_after_service_restart(tmp_path: Path):
    repo = _repo(tmp_path)
    _completed_turn(
        repo, request_id="req-rst-1", user_msg="出发日期10月20日", assistant_msg="已记录。"
    )
    _completed_turn(
        repo, request_id="req-rst-2", user_msg="两个人", assistant_msg="已记录人数。"
    )
    expected = repo.get_recent_completed_history(conversation_id=CONVERSATION)
    repo.close()

    reopened = _repo(tmp_path)
    try:
        assert (
            reopened.get_recent_completed_history(conversation_id=CONVERSATION)
            == expected
        )
    finally:
        reopened.close()
