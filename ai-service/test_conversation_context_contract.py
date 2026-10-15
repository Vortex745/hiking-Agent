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


# ---- AC5/AC7: actual kernel input order (deterministic agent double) ---------

HISTORY_FIXTURE = [
    {"role": "user", "content": "10月20日出发，两个人"},
    {"role": "assistant", "content": "好的，已记录10月20日出发、两人同行。"},
]


class _CapturingAgent:
    """Deterministic agent double: records exactly what the kernel sends."""

    def __init__(self) -> None:
        self.inputs: Any = None
        self.config: Any = None

    async def ainvoke(self, inputs: Any, config: Any = None, **kwargs: Any) -> dict[str, Any]:
        self.inputs = inputs
        self.config = config
        from langchain_core.messages import AIMessage

        return {"messages": [AIMessage(content="ok")]}

    async def aclose(self) -> None:
        return None


def _run_through_kernel(
    monkeypatch: pytest.MonkeyPatch,
    *,
    message: str,
    history: Any,
    entry: str = "run_chat_turn",
    run_id: str | None = "run-ctx-1",
    checkpoint_store: Any | None = None,
) -> _CapturingAgent:
    import agent_v2.kernel as kernel_module
    from agent_v2.kernel import run_chat_turn, run_toolless_chat

    agent = _CapturingAgent()

    async def _fake_assemble(config: Any, settings: Any) -> _CapturingAgent:
        return agent

    monkeypatch.setattr(kernel_module, "assemble_kernel_agent", _fake_assemble)
    monkeypatch.setattr(kernel_module, "get_langchain_callbacks", lambda: [])

    kwargs = dict(
        system_prompt="sp",
        message=message,
        settings=SimpleNamespace(),
        run_id=run_id,
        history=history,
    )
    if checkpoint_store is not None:
        kwargs["checkpoint_store"] = checkpoint_store
    if entry == "run_chat_turn":
        asyncio.run(run_chat_turn(**kwargs))
    else:
        asyncio.run(run_toolless_chat(**kwargs))
    return agent


def _inputs_of(agent: _CapturingAgent) -> list[tuple[str, str]]:
    assert agent.inputs is not None
    return list(agent.inputs["messages"])


def test_kernel_input_is_history_then_the_single_current_message(monkeypatch):
    agent = _run_through_kernel(
        monkeypatch, message="改成下周", history=HISTORY_FIXTURE
    )

    assert _inputs_of(agent) == [
        ("user", "10月20日出发，两个人"),
        ("assistant", "好的，已记录10月20日出发、两人同行。"),
        ("user", "改成下周"),
    ]
    # The current user message appears exactly once.
    assert [m for m in _inputs_of(agent) if m == ("user", "改成下周")] == [
        ("user", "改成下周")
    ]


def test_both_kernel_entry_constructions_receive_the_same_history(monkeypatch):
    """Ticket 26 wrapper and the T28 entry share ONE input construction."""
    direct = _run_through_kernel(
        monkeypatch, message="改成下周", history=HISTORY_FIXTURE
    )
    wrapper = _run_through_kernel(
        monkeypatch, message="改成下周", history=HISTORY_FIXTURE, entry="run_toolless_chat"
    )

    assert _inputs_of(direct) == _inputs_of(wrapper)


def test_kernel_without_history_appends_only_the_current_message(monkeypatch):
    agent = _run_through_kernel(monkeypatch, message="你好", history=None)
    assert _inputs_of(agent) == [("user", "你好")]


def test_kernel_history_never_carries_internal_roles(monkeypatch):
    """Defense in depth: a mis-shaped caller cannot smuggle tool rows in."""
    agent = _run_through_kernel(
        monkeypatch,
        message="改成下周",
        history=[
            {"role": "tool", "content": "sandbox_bash 内部记录"},
            {"role": "approval", "content": "待批准 Bash"},
            {"role": "user", "content": "10月20日出发，两个人"},
        ],
    )
    assert _inputs_of(agent) == [
        ("user", "10月20日出发，两个人"),
        ("user", "改成下周"),
    ]


def test_kernel_thread_id_stays_run_scoped_with_history(monkeypatch):
    class _Store:
        async def get(self) -> object:
            return object()

    agent = _run_through_kernel(
        monkeypatch,
        message="改成下周",
        history=HISTORY_FIXTURE,
        run_id="run-thread-a",
        checkpoint_store=_Store(),
    )

    assert agent.config is not None
    assert agent.config["configurable"]["thread_id"] == "v2:chat:run-thread-a"


# ---- AC1/AC4/AC5/AC6: orchestrator reads the history server-side -------------

_EVIDENCE_OK = {
    "tool": "hiking_knowledge_search",
    "phase": "result",
    "ok": True,
    "evidence": {"has_chunks": True, "sources": ["鳌太线攻略.md"]},
}


class _FakeMemory:
    def __init__(self, *, forbidden: set[str] | None = None) -> None:
        self.forbidden = forbidden or set()

    def memory_enabled(self, subject_id: str) -> bool:
        return True

    def claim_conversation(self, subject_id: str, conversation_id: str) -> bool:
        return conversation_id not in self.forbidden

    def owns_conversation(self, subject_id: str, conversation_id: str) -> bool:
        return conversation_id not in self.forbidden

    def create_candidate(self, subject_id: str, **kwargs: Any) -> Any:
        return kwargs


class _SpyHistoryRepo(RunRepository):
    """Real repository plus a reader counter (history-read observability)."""

    def __init__(self, control: ControlStore, checkpoints: CheckpointsStore) -> None:
        super().__init__(control=control, checkpoints=checkpoints)
        self.history_calls = 0

    def get_recent_completed_history(self, **kwargs: Any) -> list[dict[str, str]]:
        self.history_calls += 1
        return super().get_recent_completed_history(**kwargs)


class _BrokenHistoryRepo(RunRepository):
    """Real repository whose history read fails like a storage fault."""

    def get_recent_completed_history(self, **kwargs: Any) -> list[dict[str, str]]:
        raise sqlite3.OperationalError("disk I/O error")


def _settings(**overrides: Any) -> SimpleNamespace:
    base = {
        "agent_v2_enabled": True,
        "openai_api_key": "sk-test",
        "openai_model": "test-model",
        "agent_v2_cowork_policy": "disabled",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _orchestrator(
    repo: RunRepository,
    kernel_calls: list[dict[str, Any]],
    *,
    memory: Any | None = None,
    kernel_result: tuple[str, list[dict[str, Any]]] | None = None,
) -> RunOrchestrator:
    async def fake_kernel(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        kernel_calls.append(kwargs)
        return kernel_result or ("你好，徒步愉快！", [])

    return RunOrchestrator(
        settings=_settings(),
        runs_repo=repo,
        memory=memory or _FakeMemory(),
        kernel_turn=fake_kernel,
    )


def _req(
    message: str, request_id: str, conversation_id: str = CONVERSATION
) -> ChatFlowRequest:
    return ChatFlowRequest(
        message=message, request_id=request_id, conversation_id=conversation_id
    )


def _stream(orch: RunOrchestrator, req: ChatFlowRequest) -> list[Any]:
    async def run() -> list[Any]:
        return [item async for item in orch.stream_chat(req, subject_id="sub-ctx")]

    return asyncio.run(run())


def _types(items: list[Any]) -> list[str]:
    return [item.type for item in items]


def _text_of(items: list[Any]) -> str:
    return next(e.payload["content"] for e in items if e.type == "text")


def test_first_turn_kernel_receives_an_empty_history(tmp_path: Path):
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []

    items = _stream(_orchestrator(repo, calls), _req("你好", "req-first"))

    assert _types(items)[-1] == "done"
    assert calls[0]["history"] == []  # fresh conversation semantics
    assert calls[0]["message"] == "你好"


def test_second_turn_kernel_receives_the_prior_completed_turn_in_order(tmp_path: Path):
    repo = _repo(tmp_path)
    first = _stream(
        _orchestrator(
            repo,
            [],
            kernel_result=("好的，已记录10月20日出发、两人同行。", [_EVIDENCE_OK]),
        ),
        _req("10月20日出发，两个人", "req-turn-1"),
    )
    assert _types(first)[-1] == "done"

    second_calls: list[dict[str, Any]] = []
    second = _stream(
        _orchestrator(repo, second_calls, kernel_result=("已按新日期更新。", [_EVIDENCE_OK])),
        _req("改成下周", "req-turn-2"),
    )

    assert _types(second)[-1] == "done"
    assert second_calls[0]["history"] == [
        {"role": "user", "content": "10月20日出发，两个人"},
        {"role": "assistant", "content": "好的，已记录10月20日出发、两人同行。"},
    ]
    assert second_calls[0]["message"] == "改成下周"


def test_three_turn_continuation_keeps_the_latest_condition_last(tmp_path: Path):
    repo = _repo(tmp_path)
    orch = _orchestrator(repo, [], kernel_result=("收到，已记录。", [_EVIDENCE_OK]))
    assert _types(_stream(orch, _req("出发日期10月20日", "req-c1")))[-1] == "done"
    assert _types(_stream(orch, _req("改成10月27日", "req-c2")))[-1] == "done"

    calls: list[dict[str, Any]] = []
    third = _stream(
        _orchestrator(repo, calls, kernel_result=("好的。", [_EVIDENCE_OK])),
        _req("再加一个人", "req-c3"),
    )

    assert _types(third)[-1] == "done"
    assert [entry["content"] for entry in calls[0]["history"]] == [
        "出发日期10月20日",
        "收到，已记录。",
        "改成10月27日",
        "收到，已记录。",
    ]
    assert calls[0]["message"] == "再加一个人"
    # The current user message is NOT duplicated into the history window.
    assert all(entry["content"] != "再加一个人" for entry in calls[0]["history"])


def test_controlled_double_behavior_changes_with_history(tmp_path: Path):
    """Deterministic double: with history it continues the constraint; on a
    fresh conversation the same request cannot be answered from context."""

    def _make_kernel(calls: list[dict[str, Any]]):
        async def fake_kernel(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
            calls.append(kwargs)
            history_text = "".join(
                entry["content"] for entry in kwargs.get("history") or []
            )
            if "10月20日" in history_text:
                return "沿用此前记录的10月20日出发，仅调整日程。", [_EVIDENCE_OK]
            return "缺少出发日期，无法沿用任何约束。", []

        return fake_kernel

    repo = _repo(tmp_path)
    _stream(
        _orchestrator(repo, [], kernel_result=("已记录10月20日出发。", [_EVIDENCE_OK])),
        _req("10月20日出发", "req-d1"),
    )

    follow_calls: list[dict[str, Any]] = []
    follow = _stream(
        RunOrchestrator(
            settings=_settings(),
            runs_repo=repo,
            memory=_FakeMemory(),
            kernel_turn=_make_kernel(follow_calls),
        ),
        _req("改成下周", "req-d2"),
    )
    assert "沿用此前记录的10月20日" in _text_of(follow)

    fresh_calls: list[dict[str, Any]] = []
    fresh = _stream(
        RunOrchestrator(
            settings=_settings(),
            runs_repo=_repo(tmp_path, "-fresh"),
            memory=_FakeMemory(),
            kernel_turn=_make_kernel(fresh_calls),
        ),
        _req("改成下周", "req-d3", conversation_id="conv-fresh"),
    )
    assert fresh_calls[0]["history"] == []
    assert "沿用" not in _text_of(fresh)
    assert "资料不足" in _text_of(fresh)


def test_old_ungrounded_assistant_fact_in_history_cannot_bypass_evidence_gate(
    tmp_path: Path,
):
    """AC7: history is context, not this-turn evidence."""
    repo = _repo(tmp_path)
    # A completed turn whose stored answer carries an un-grounded route
    # fact (written before Ticket 02's gate existed).
    _completed_turn(
        repo,
        request_id="req-legacy",
        user_msg="这段路线爬升多少",
        assistant_msg="这段路线爬升约1800米。",
    )
    calls: list[dict[str, Any]] = []
    items = _stream(
        _orchestrator(
            repo,
            calls,
            kernel_result=("根据上一轮记录，这段路线爬升约1800米。", []),
        ),
        _req("那爬升呢", "req-follow"),
    )

    # The double DID receive the un-grounded old fact…
    assert calls[0]["history"][-1]["content"] == "这段路线爬升约1800米。"
    # …but it grants no factual authority for THIS turn.
    text = _text_of(items)
    assert "1800" not in text
    assert "资料不足" in text


def test_oversized_prior_turn_stays_out_of_the_next_kernel_history(tmp_path: Path):
    repo = _repo(tmp_path)
    _completed_turn(
        repo,
        request_id="req-huge",
        user_msg="超长约束",
        assistant_msg="超长" * 4500,  # 9000 chars > 8000 window
    )
    calls: list[dict[str, Any]] = []
    items = _stream(
        _orchestrator(repo, calls, kernel_result=("好的。", [_EVIDENCE_OK])),
        _req("改成下周", "req-after-huge"),
    )

    assert _types(items)[-1] == "done"
    assert calls[0]["history"] == []  # omitted whole — ask, don't guess


def test_concurrent_unfinished_turn_is_excluded_from_the_next_kernel_history(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    _completed_turn(
        repo,
        request_id="req-done-before",
        user_msg="10月20日出发",
        assistant_msg="已记录。",
    )
    # Another run of the SAME conversation is still running right now.
    repo.create_run_idempotent(
        request_id="req-concurrent",
        conversation_id=CONVERSATION,
        profile_id="chat",
        user_message="正在进行的回合",
    )

    calls: list[dict[str, Any]] = []
    items = _stream(
        _orchestrator(repo, calls, kernel_result=("好的。", [_EVIDENCE_OK])),
        _req("改成下周", "req-next"),
    )

    assert _types(items)[-1] == "done"
    contents = [entry["content"] for entry in calls[0]["history"]]
    assert contents == ["10月20日出发", "已记录。"]
    assert "正在进行的回合" not in contents
    assert "改成下周" not in contents


def test_duplicate_request_id_does_not_repeat_the_model_call(tmp_path: Path):
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []
    orch = _orchestrator(repo, calls, kernel_result=("已记录。", [_EVIDENCE_OK]))

    first = _stream(orch, _req("10月20日出发", "req-dup-ctx"))
    second = _stream(orch, _req("10月20日出发", "req-dup-ctx"))

    assert len(calls) == 1  # idempotent replay — no second model call
    assert _types(second) == _types(first)


def test_context_unavailable_terminal_when_history_storage_fails(tmp_path: Path):
    normal = _repo(tmp_path)
    _completed_turn(
        normal, request_id="req-pre", user_msg="约束", assistant_msg="答复"
    )
    broken = _BrokenHistoryRepo(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )
    calls: list[dict[str, Any]] = []

    items = _stream(_orchestrator(broken, calls), _req("改成下周", "req-broken"))

    types = _types(items)
    assert types[-1] == "error"
    assert items[-1].payload["code"] == "context_unavailable"
    assert [t for t in types if t in TERMINALS] == ["error"]
    assert calls == []  # no model call without trustworthy history
    run = broken.control.get_run_by_request("req-broken")
    assert run is not None and run.status == "error"


def test_foreign_subject_same_conversation_is_rejected_without_history_read(
    tmp_path: Path,
):
    repo = _SpyHistoryRepo(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )
    _completed_turn(
        repo, request_id="req-owned", user_msg="本主体的约束", assistant_msg="答复"
    )
    calls: list[dict[str, Any]] = []

    items = _stream(
        _orchestrator(repo, calls, memory=_FakeMemory(forbidden={CONVERSATION})),
        _req("改成下周", "req-foreign"),
    )

    assert [item.type for item in items] == ["error"]
    assert items[0].payload["code"] == "conversation_forbidden"
    assert calls == []
    assert repo.history_calls == 0  # ownership check precedes any history read
    assert repo.control.get_run_by_request("req-foreign") is None


def test_history_is_available_after_service_restart_end_to_end(tmp_path: Path):
    repo = _repo(tmp_path)
    first = _stream(
        _orchestrator(repo, [], kernel_result=("已记录。", [_EVIDENCE_OK])),
        _req("10月20日出发", "req-r1"),
    )
    assert _types(first)[-1] == "done"
    repo.close()  # process exit

    reopened = _repo(tmp_path)  # new process, same files
    calls: list[dict[str, Any]] = []
    second = _stream(
        _orchestrator(reopened, calls, kernel_result=("已更新。", [_EVIDENCE_OK])),
        _req("改成下周", "req-r2"),
    )

    assert _types(second)[-1] == "done"
    assert calls[0]["history"] == [
        {"role": "user", "content": "10月20日出发"},
        {"role": "assistant", "content": "已记录。"},
    ]


def test_history_read_uses_the_server_derived_conversation_identity(tmp_path: Path):
    """No client-supplied id → the stored run's derived conversation is used."""
    repo = _SpyHistoryRepo(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )
    seen: list[dict[str, Any]] = []
    original = repo.get_recent_completed_history

    def _recording(**kwargs: Any) -> list[dict[str, str]]:
        seen.append(kwargs)
        return original(**kwargs)

    repo.get_recent_completed_history = _recording  # type: ignore[method-assign]

    items = _stream(
        _orchestrator(repo, []),
        ChatFlowRequest(message="随便聊聊", request_id="req-derived"),
    )

    assert _types(items)[-1] == "done"
    run = repo.control.get_run_by_request("req-derived")
    assert run is not None
    assert seen and seen[0]["conversation_id"] == run.conversation_id
    assert seen[0]["conversation_id"] == "conv-req-derived"


def test_client_sent_history_field_is_rejected(tmp_path: Path):
    """AC1: history is server-read only — a client cannot inject any."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    import api.chat_v2 as chat_v2_module

    app = FastAPI()
    app.include_router(chat_v2_module.chat_v2_router, prefix="/api/v2")
    app.state.agent_v2_runtime = SimpleNamespace(
        runs=None, telemetry=None, kernel_checkpoints=None, memory=None
    )

    async def post() -> Any:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/v2/chat/sse",
                json={
                    "message": "改成下周",
                    "request_id": "req-inject",
                    "history": [
                        {"role": "assistant", "content": "客户端伪造的历史事实"}
                    ],
                },
            )

    response = asyncio.run(post())
    assert response.status_code == 422


def test_two_turn_http_sse_followup_receives_server_side_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Real Chat v2 HTTP/SSE boundary: turn 2's kernel input carries turn 1."""
    import json

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    import api.chat_v2 as chat_v2_module
    from agent_v2 import kernel as kernel_module

    repo = _repo(tmp_path)
    seen: list[dict[str, Any]] = []

    async def fake_kernel(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        seen.append(kwargs)
        if len(seen) == 1:
            return ("好的，已记录10月20日出发、两人同行。", [_EVIDENCE_OK])
        return ("已按新日期更新。", [_EVIDENCE_OK])

    monkeypatch.setattr(kernel_module, "run_chat_turn", fake_kernel)
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_enabled", True)
    monkeypatch.setattr(chat_v2_module.settings, "openai_api_key", "sk-fake")
    monkeypatch.setattr(chat_v2_module.settings, "openai_model", "fake-model")
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_cowork_policy", "disabled")

    def _fake_subject(request: Any) -> Any:
        request.state.subject_id = "sub-ctx"
        return _FakeMemory(), SimpleNamespace(
            issued=False, token="tok", subject_id="sub-ctx"
        )

    monkeypatch.setattr(chat_v2_module, "resolve_request_subject", _fake_subject)

    app = FastAPI()
    app.include_router(chat_v2_module.chat_v2_router, prefix="/api/v2")
    app.state.agent_v2_runtime = SimpleNamespace(
        runs=repo,
        telemetry=None,
        kernel_checkpoints=None,
        memory=_FakeMemory(),
        capacity=None,
    )

    async def collect(payload: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.stream(
                "POST", "/api/v2/chat/sse", json=payload
            ) as response:
                assert response.status_code == 200
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    while "\n\n" in buffer:
                        frame, buffer = buffer.split("\n\n", 1)
                        for line in frame.splitlines():
                            if line.startswith("data: "):
                                events.append(json.loads(line[len("data: "):]))
        return events

    first = asyncio.run(
        collect({
            "message": "10月20日出发，两个人",
            "request_id": "req-http-1",
            "conversation_id": CONVERSATION,
        })
    )
    assert first[-1]["type"] == "done"
    assert seen[0]["history"] == []  # fresh conversation on turn 1

    second = asyncio.run(
        collect({
            "message": "改成下周",
            "request_id": "req-http-2",
            "conversation_id": CONVERSATION,
        })
    )
    assert second[-1]["type"] == "done"
    assert seen[1]["history"] == [
        {"role": "user", "content": "10月20日出发，两个人"},
        {"role": "assistant", "content": "好的，已记录10月20日出发、两人同行。"},
    ]
    assert seen[1]["message"] == "改成下周"
    # Exactly one current-message copy across the whole input contract.
    assert [e for e in seen[1]["history"] if e["content"] == "改成下周"] == []
