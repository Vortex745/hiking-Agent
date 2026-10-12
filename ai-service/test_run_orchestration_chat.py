"""Ticket 37 — run orchestration module interface tests for the chat flow.

The orchestration interface is tested in-process: a fake kernel entry and
real SQLite persistence under a temp directory are injected through the
constructor. No ASGI shell, no module-global monkeypatching, no real LLM
or sandbox credentials.

Behavior asserted (Ticket 36/37 AC):

* Event sequence and terminal semantics of the chat flow are unchanged:
  ``run_started → progress → heartbeat → text → done``, strictly
  monotonic sequence from 1, exactly one terminal per stream.
* Duplicate ``request_id`` replays stored events without re-calling the
  kernel (spec AC2).
* Kernel failures map to stable terminal ``error`` events.
* Pre-flight gates (v2 disabled, unknown profile, model unavailable)
  still produce terminal ``error`` events.
* Capacity lease semantics: exhaustion raises ``CapacityExceeded`` with
  the partial lease released; acquired leases release on stream exit.
* Cancel observation produces the ``cancelled`` terminal.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_v2.kernel import KernelModelFailure
from agent_v2.observability import CapacityExceeded
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator, mark_terminal_safe
from agent_v2.runs import RunRepository
from agent_v2.persistence import CheckpointsStore, ControlStore


TERMINALS = {"error", "done", "cancelled"}

SUBJECT_ID = "sub-1"


class _FakeMemory:
    """Same shape the ASGI contract tests inject for the chat profile.

    The chat profile allows memory_* kernel tools, which fail closed
    without a memory service + server-resolved subject — so the
    orchestration tests inject this fake exactly like production does.
    """

    def memory_enabled(self, subject_id): return True

    def claim_conversation(self, s, c): return True

    def owns_conversation(self, s, c): return True

    def create_candidate(self, s, **kw): return kw


def _repo(tmp_path: Path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


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
    tmp_path: Path,
    kernel_calls: list[dict[str, Any]],
    *,
    settings: Any | None = None,
    runs_repo: RunRepository | None = None,
    capacity: Any | None = None,
    kernel_error: Exception | None = None,
    kernel_result: tuple[str, list[dict[str, Any]]] | None = None,
) -> RunOrchestrator:
    async def fake_kernel(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        kernel_calls.append(kwargs)
        if kernel_error is not None:
            raise kernel_error
        return kernel_result or ("你好，徒步愉快！", [])

    return RunOrchestrator(
        settings=settings or _settings(),
        runs_repo=runs_repo if runs_repo is not None else _repo(tmp_path),
        memory=_FakeMemory(),
        kernel_turn=fake_kernel,
        capacity=capacity,
    )


def _flow_req(request_id: str = "req-37", message: str = "带什么装备？") -> ChatFlowRequest:
    return ChatFlowRequest(message=message, request_id=request_id)


def _collect(gen: Any) -> list[Any]:
    async def run() -> list[Any]:
        return [item async for item in gen]

    return asyncio.run(run())


def _stream(orch: RunOrchestrator, req: ChatFlowRequest) -> list[Any]:
    return _collect(orch.stream_chat(req, subject_id=SUBJECT_ID))


def test_chat_flow_emits_expected_sequence_with_single_done_terminal(tmp_path: Path):
    kernel_calls: list[dict[str, Any]] = []
    orch = _orchestrator(tmp_path, kernel_calls)

    items = _stream(orch, _flow_req())

    types = [item.type for item in items]
    assert types == ["run_started", "progress", "heartbeat", "text", "done"]

    sequences = [item.sequence for item in items]
    assert sequences == [1, 2, 3, 4, 5]

    terminals = [t for t in types if t in TERMINALS]
    assert terminals == ["done"]

    text_events = [e for e in items if e.type == "text"]
    # 窄门语义：本回合未尝试知识检索（无 audit），文本不被 fail-closed 替换。
    assert text_events[0].payload["content"] == "你好，徒步愉快！"

    # Exactly one kernel call, with the composed product system prompt.
    assert len(kernel_calls) == 1
    assert "徒步助手" in kernel_calls[0]["system_prompt"]
    assert "必须先调用 hiking_knowledge_search" in kernel_calls[0]["system_prompt"]
    assert kernel_calls[0]["message"] == "带什么装备？"


def test_chat_flow_replaces_unsupported_factual_output_when_search_has_no_evidence(
    tmp_path: Path,
):
    kernel_calls: list[dict[str, Any]] = []
    orch = _orchestrator(
        tmp_path,
        kernel_calls,
        kernel_result=(
            "鳌太线全程约120公里，冬季水源很少。",
            [{
                "tool": "hiking_knowledge_search",
                "phase": "result",
                "ok": True,
                "evidence": {"has_chunks": False, "sources": []},
            }],
        ),
    )

    items = _stream(orch, _flow_req(message="查鳌太线冬季精确里程"))

    text = next(event.payload["content"] for event in items if event.type == "text")
    assert text == "资料不足：未检索到可核验的徒步知识，无法提供具体徒步事实。"
    assert "120公里" not in text


def test_chat_flow_passes_text_when_model_skips_knowledge_search(tmp_path: Path):
    """窄门边界：模型完全未尝试检索时不拦截。

    fail-closed 只覆盖"编造签名"（检索被尝试但无可用地片段）。
    未尝试检索的回合由 system prompt 的先检索约束约束；
    语义蕴含门是计划中的第二层（见项目 TODO）。
    """
    kernel_calls: list[dict[str, Any]] = []
    orch = _orchestrator(
        tmp_path,
        kernel_calls,
        kernel_result=("这段路线爬升约1800米。", []),
    )

    items = _stream(orch, _flow_req(message="这段路线爬升多少"))

    text = next(event.payload["content"] for event in items if event.type == "text")
    assert text == "这段路线爬升约1800米。"


def test_chat_flow_passes_text_when_other_tool_grounded_output_exists(tmp_path: Path):
    """窄门边界：非知识工具的成功产出（装备清单等）不被拦截。"""
    kernel_calls: list[dict[str, Any]] = []
    orch = _orchestrator(
        tmp_path,
        kernel_calls,
        kernel_result=(
            "装备清单如下：帐篷、睡袋、登山鞋。请按保守原则准备。",
            [{
                "tool": "gear_checklist",
                "phase": "result",
                "ok": True,
            }],
        ),
    )

    items = _stream(orch, _flow_req(message="带什么装备？"))

    text = next(event.payload["content"] for event in items if event.type == "text")
    assert "帐篷" in text


def test_chat_flow_preserves_answer_when_knowledge_evidence_exists(tmp_path: Path):
    kernel_calls: list[dict[str, Any]] = []
    expected = "根据已检索资料，携带离线地图并核对天气预警。"
    orch = _orchestrator(
        tmp_path,
        kernel_calls,
        kernel_result=(
            expected,
            [{
                "tool": "hiking_knowledge_search",
                "phase": "result",
                "ok": True,
                "evidence": {"has_chunks": True, "sources": ["safety-guide.md"]},
            }],
        ),
    )

    items = _stream(orch, _flow_req(message="徒步需要哪些安全准备"))

    text = next(event.payload["content"] for event in items if event.type == "text")
    assert text == expected


def _knowledge_evidence(distance_values: list[float]) -> list[dict[str, Any]]:
    return [{
        "tool": "hiking_knowledge_search",
        "phase": "result",
        "ok": True,
        "evidence": {
            "has_chunks": True,
            "sources": ["trail-guide.md"],
            "chunks": [{
                "title": "trail-guide",
                "source": "trail-guide.md",
                "preview": "全程里程……",
                "distance_values": distance_values,
            }],
        },
    }]


def test_chat_flow_keeps_answer_when_distance_numbers_are_supported(tmp_path: Path):
    kernel_calls: list[dict[str, Any]] = []
    expected = "鳌太线全程约120公里，冬季水源很少。"
    orch = _orchestrator(
        tmp_path,
        kernel_calls,
        kernel_result=(expected, _knowledge_evidence([120.0])),
    )

    items = _stream(orch, _flow_req(message="鳌太线多长"))

    text = next(event.payload["content"] for event in items if event.type == "text")
    assert text == expected


def test_chat_flow_replaces_answer_with_unsupported_distance_number(tmp_path: Path):
    kernel_calls: list[dict[str, Any]] = []
    orch = _orchestrator(
        tmp_path,
        kernel_calls,
        kernel_result=("鳌太线全程约120公里，冬季水源很少。", _knowledge_evidence([55.0])),
    )

    items = _stream(orch, _flow_req(message="鳌太线多长"))

    text = next(event.payload["content"] for event in items if event.type == "text")
    assert "120公里" not in text
    assert "资料不足" in text
    # 语义区别于第一层「未检索到证据」：用户能看出是证据不支持，而非没去查。
    assert "未检索到" not in text


def test_chat_flow_keeps_distance_numbers_when_search_never_attempted(tmp_path: Path):
    """窄门边界不变：从未检索的回合不做数字校验。"""
    kernel_calls: list[dict[str, Any]] = []
    expected = "这段路线全程约15公里。"
    orch = _orchestrator(tmp_path, kernel_calls, kernel_result=(expected, []))

    items = _stream(orch, _flow_req(message="这段路线多长"))

    text = next(event.payload["content"] for event in items if event.type == "text")
    assert text == expected


def test_chat_flow_persists_events_and_marks_run_done(tmp_path: Path):
    repo = _repo(tmp_path)
    kernel_calls: list[dict[str, Any]] = []
    orch = _orchestrator(tmp_path, kernel_calls, runs_repo=repo)

    items = _stream(orch, _flow_req())

    run = repo.control.get_run_by_request("req-37")
    assert run is not None
    snapshot = repo.get_snapshot(run.run_id)
    assert snapshot is not None
    assert snapshot.status == "done"

    stored = repo.get_events_since(run_id=run.run_id, from_seq=0)
    assert [e["type"] for e in stored] == [item.type for item in items]
    assert [e["sequence"] for e in stored] == [item.sequence for item in items]


def test_duplicate_request_id_replays_without_kernel_call(tmp_path: Path):
    kernel_calls: list[dict[str, Any]] = []
    orch = _orchestrator(tmp_path, kernel_calls)

    first = _stream(orch, _flow_req())
    second = _stream(orch, _flow_req())

    assert len(kernel_calls) == 1  # no model re-execution (spec AC2)

    # Replay emits the same event types/sequences as the original stream.
    assert [e.type for e in second] == [e.type for e in first]
    assert [e.sequence for e in second] == [e.sequence for e in first]
    terminals = [e.type for e in second if e.type in TERMINALS]
    assert terminals == ["done"]


def test_mark_terminal_safe_does_not_resurrect_terminal_run(tmp_path: Path):
    """Terminal reset is idempotent (Ticket 41 direct case).

    The DB status is the audit trail for the exactly-one-terminal wire
    guarantee: once a run reached a terminal, a late ``error`` arm (e.g.
    cancel beat the stream, or a race between resume and an exception
    path) must be swallowed — not raise, not overwrite the status.
    """
    repo = _repo(tmp_path)
    run_id, _ = repo.create_run_idempotent(
        request_id="req-term",
        conversation_id="conv-term",
        profile_id="chat",
        user_message="m",
    )
    repo.mark_run_done(run_id)

    # Late error arm after the run is already done — swallowed.
    mark_terminal_safe(repo, run_id, "error")
    run = repo.get_run(run_id)
    assert run is not None
    assert run.status == "done" and run.terminal_type == "done"

    # Pre-persistence stream paths (no repo / no run_id) are no-ops.
    mark_terminal_safe(None, run_id, "error")
    mark_terminal_safe(repo, None, "error")

    # A still-running run DOES get its terminal stamped.
    run_id2, _ = repo.create_run_idempotent(
        request_id="req-term2",
        conversation_id="conv-term2",
        profile_id="chat",
        user_message="m2",
    )
    mark_terminal_safe(repo, run_id2, "error")
    run2 = repo.get_run(run_id2)
    assert run2 is not None and run2.status == "error"


def test_kernel_failure_maps_to_stable_error_terminal(tmp_path: Path):
    kernel_calls: list[dict[str, Any]] = []
    orch = _orchestrator(
        tmp_path,
        kernel_calls,
        kernel_error=KernelModelFailure("model_500", "模型调用失败。"),
    )

    items = _stream(orch, _flow_req())

    types = [e.type for e in items]
    assert types == ["run_started", "progress", "heartbeat", "error"]
    error = items[-1]
    assert error.payload["code"] == "model_500"
    assert len([t for t in types if t in TERMINALS]) == 1


def test_v2_disabled_emits_terminal_error(tmp_path: Path):
    kernel_calls: list[dict[str, Any]] = []
    orch = _orchestrator(
        tmp_path, kernel_calls, settings=_settings(agent_v2_enabled=False)
    )

    items = _stream(orch, _flow_req())

    types = [e.type for e in items]
    assert types == ["error"]
    assert items[0].payload["code"] == "v2_disabled"
    assert kernel_calls == []


def test_unknown_profile_emits_terminal_error(tmp_path: Path):
    kernel_calls: list[dict[str, Any]] = []
    orch = _orchestrator(tmp_path, kernel_calls)

    items = _stream(
        orch,
        ChatFlowRequest(message="x", request_id="req-37p", profile_id="no-such-profile"),
    )

    types = [e.type for e in items]
    assert types == ["error"]
    assert items[0].payload["code"] == "unknown_profile"
    assert kernel_calls == []


def test_missing_api_key_emits_run_started_then_model_unavailable(tmp_path: Path):
    kernel_calls: list[dict[str, Any]] = []
    orch = _orchestrator(tmp_path, kernel_calls, settings=_settings(openai_api_key=""))

    items = _stream(orch, _flow_req())

    types = [e.type for e in items]
    assert types == ["run_started", "error"]
    assert items[-1].payload["code"] == "model_unavailable"
    assert kernel_calls == []


def test_cancel_before_stream_emits_cancelled_terminal(tmp_path: Path):
    repo = _repo(tmp_path)
    kernel_calls: list[dict[str, Any]] = []
    orch = _orchestrator(tmp_path, kernel_calls, runs_repo=repo)

    # Create + cancel the run before the stream starts.
    run_id, _created = repo.create_run_idempotent(
        request_id="req-37c",
        conversation_id="conv-37c",
        profile_id="chat",
        user_message="x",
    )
    outcome = repo.request_cancel(run_id=run_id, idempotency_key="k1")
    assert outcome.accepted

    items = _stream(orch, _flow_req(request_id="req-37c"))

    types = [e.type for e in items]
    assert types[-1] == "cancelled"
    assert len([t for t in types if t in TERMINALS]) == 1
    assert kernel_calls == []  # cancelled before the kernel turn


def test_capacity_exceeded_releases_partial_run_lease(tmp_path: Path):
    kernel_calls: list[dict[str, Any]] = []

    class FakeRunLease:
        def __init__(self) -> None:
            self.released = False

        def release(self) -> None:
            self.released = True

    class FakeCapacity:
        def __init__(self) -> None:
            self.run_lease = FakeRunLease()

        async def acquire_run(self):
            return self.run_lease

        async def acquire_sandbox(self, request_id: str):
            raise CapacityExceeded("sandbox")

    capacity = FakeCapacity()
    orch = _orchestrator(tmp_path, kernel_calls, capacity=capacity)

    with pytest.raises(CapacityExceeded) as excinfo:
        asyncio.run(
            orch.acquire_leases(request_id="req-37x", profile_id="artifact")
        )

    assert excinfo.value.resource == "sandbox"
    assert capacity.run_lease.released  # partial lease must not leak
    assert kernel_calls == []


def test_leases_acquired_and_released_on_stream_exit(tmp_path: Path):
    kernel_calls: list[dict[str, Any]] = []

    class FakeRunLease:
        def __init__(self) -> None:
            self.released = False

        def release(self) -> None:
            self.released = True

    class FakeCapacity:
        def __init__(self) -> None:
            self.run_lease = FakeRunLease()

        async def acquire_run(self):
            return self.run_lease

        async def acquire_sandbox(self, request_id: str):  # pragma: no cover
            raise AssertionError("chat profile must not acquire a sandbox slot")

    capacity = FakeCapacity()
    repo = _repo(tmp_path)
    orch = _orchestrator(tmp_path, kernel_calls, runs_repo=repo, capacity=capacity)

    leases = asyncio.run(orch.acquire_leases(request_id="req-37l", profile_id="chat"))
    assert not capacity.run_lease.released

    items = _stream(orch, _flow_req(request_id="req-37l"))
    assert types_of(items)[-1] == "done"

    leases.release()
    assert capacity.run_lease.released


def types_of(items: list[Any]) -> list[str]:
    return [e.type for e in items]


def test_chat_tool_result_carries_server_id_and_content(tmp_path: Path):
    """A08: audit → tool_result conversion exposes server_id (the old
    ``server`` rename lost the field on the wire) and a human-readable
    content summary so the trace panel shows what each step did."""
    kernel_calls: list[dict[str, Any]] = []
    audits = [
        {"tool": "weather_lookup", "phase": "result", "ok": True, "duration_ms": 1234},
        {
            "phase": "mcp_unavailable",
            "ok": False,
            "code": "mcp_server_unavailable",
            "server_id": "amap",
        },
        {"tool": "file_write", "phase": "denied", "code": "denied_by_gate"},
    ]
    orch = _orchestrator(tmp_path, kernel_calls, kernel_result=("回答", audits))

    items = _stream(orch, _flow_req())
    results = {e.payload.get("tool"): e.payload for e in items if e.type == "tool_result"}

    weather = results["weather_lookup"]
    assert weather["status"] == "ok"
    assert "server" not in weather  # old renamed key must not return
    assert weather["content"] == "完成，耗时 1234ms"

    # The mcp_unavailable audit has no tool name — the payload's tool
    # key falls back to server_id.
    amap = results["amap"]
    assert amap["server_id"] == "amap"
    assert "不可用" in amap["content"]

    denied = results["file_write"]
    assert denied["status"] == "denied"
    assert denied["reason"] == "denied_by_gate"
    assert "拒绝" in denied["content"]
