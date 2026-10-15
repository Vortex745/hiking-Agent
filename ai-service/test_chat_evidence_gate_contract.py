"""Ticket 02 — Chat v2 turn evidence gate (spec T03 + T04).

Defect pinned here: the old grounding gate only fired on the "fabrication
signature" (knowledge search attempted but no chunks). A turn where the
model NEVER searched passed arbitrary factual text straight through — the
project's own old contract asserted that "这段路线爬升约1800米。" reached
the user un-retrieved. This file pins the fixed behavior:

* a free-form factual reply requires THIS-turn successful, non-empty,
  traceable knowledge retrieval (AC1);
* the streamed text event, the final result, and the persisted history
  all carry the SAME gated content — no unsupported content exists in any
  emitted frame, replay frame, or stored answer (AC2);
* bypass attempts are blocked: never-searched turns, generate-then-
  retrieve, evidence belonging to another run, and old-conversation
  references (AC3);
* greetings, insufficient-materials, and missing-parameter
  clarifications use controlled templates (AC4);
* gear_checklist / risk_assessment successes render from CONTROLLED
  fields with basis + limitations; invalid or failed results never
  produce a fake success (AC5);
* one domain-tool success never lets the model attach unsupported route
  numbers, live weather, or other facts (AC6);
* retrieval-grounded progressive replies keep working; cancel/disconnect
  keep exactly one terminal and no late text (AC7).

Run (ai-service working directory):
    E:/miniconda3/python.exe -B -m pytest -q test_chat_evidence_gate_contract.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_v2.run_orchestration import (
    ChatFlowRequest,
    RunOrchestrator,
    has_hiking_knowledge_evidence,
)
from agent_v2.runs import RunRepository
from agent_v2.persistence import CheckpointsStore, ControlStore


TERMINALS = {"error", "done", "cancelled"}

UNSUPPORTED_FACT = "这段路线爬升约1800米。"


class _FakeMemory:
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
    kernel_result: tuple[str, list[dict[str, Any]]] | None = None,
    runs_repo: RunRepository | None = None,
) -> RunOrchestrator:
    async def fake_kernel(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        kernel_calls.append(kwargs)
        return kernel_result or (UNSUPPORTED_FACT, [])

    return RunOrchestrator(
        settings=_settings(),
        runs_repo=runs_repo if runs_repo is not None else _repo(tmp_path),
        memory=_FakeMemory(),
        kernel_turn=fake_kernel,
    )


def _stream(orch: RunOrchestrator, req: ChatFlowRequest) -> list[Any]:
    async def run() -> list[Any]:
        return [item async for item in orch.stream_chat(req, subject_id="sub-gate")]

    return asyncio.run(run())


def _req(message: str, request_id: str, profile_id: str = "chat") -> ChatFlowRequest:
    return ChatFlowRequest(
        message=message, request_id=request_id, profile_id=profile_id,
        conversation_id=f"conv-{request_id}",
    )


def _text_of(items: list[Any]) -> str:
    return next(e.payload["content"] for e in items if e.type == "text")


_EVIDENCE_OK = {
    "tool": "hiking_knowledge_search",
    "phase": "result",
    "ok": True,
    "evidence": {"has_chunks": True, "sources": ["鳌太线攻略.md"]},
}

_GEAR_CONTROLLED = {
    "tool": "gear_checklist",
    "phase": "result",
    "ok": True,
    "controlled": {
        "tool": "gear_checklist",
        "ok": True,
        "days": 2,
        "season": "冬季",
        "experience": "新手",
        "camping": True,
        "checklist": {
            "基础": ["登山鞋", "速干衣裤"],
            "露营": ["帐篷", "睡袋", "防潮垫"],
        },
        "limitations": "装备清单按保守原则生成；具体重量和型号需结合天气、海拔和路线强度调整。",
    },
}

_RISK_CONTROLLED = {
    "tool": "risk_assessment",
    "phase": "result",
    "ok": True,
    "controlled": {
        "tool": "risk_assessment",
        "ok": True,
        "destination": "鳌太线",
        "risk_level": "high",
        "reasons": ["存在暴雨/雷暴/橙色及以上预警等硬风险条件。"],
        "cancel_conditions": ["暴雨橙色及以上预警", "雷暴大风"],
        "alternatives": ["改期", "缩短路线"],
        "recommendation": "建议取消或改期，不建议进入山区、峡谷或长距离路线。",
    },
}


# ---- AC1: no this-turn evidence ⇒ no free factual text -----------------------


def test_never_searched_factual_output_is_blocked(tmp_path: Path):
    """The old "1800 米 passthrough" contract, inverted (AC1/AC3)."""
    items = _stream(
        _orchestrator(tmp_path, []),
        _req("这段路线爬升多少", "req-gate-never"),
    )
    text = _text_of(items)
    assert UNSUPPORTED_FACT not in text
    assert "1800" not in text
    assert "资料不足" in text


def test_empty_retrieval_output_is_blocked(tmp_path: Path):
    items = _stream(
        _orchestrator(tmp_path, [], kernel_result=(
            "鳌太线全程约120公里，冬季水源很少。",
            [{
                "tool": "hiking_knowledge_search",
                "phase": "result",
                "ok": True,
                "evidence": {"has_chunks": False, "sources": []},
            }],
        )),
        _req("查鳌太线冬季精确里程", "req-gate-empty"),
    )
    text = _text_of(items)
    assert "120公里" not in text
    assert "资料不足" in text


def test_failed_retrieval_output_is_blocked(tmp_path: Path):
    items = _stream(
        _orchestrator(tmp_path, [], kernel_result=(
            "全程约120公里。",
            [{
                "tool": "hiking_knowledge_search",
                "phase": "result",
                "ok": False,
                "code": "tool_error",
                "evidence": {"has_chunks": False, "sources": []},
            }],
        )),
        _req("查里程", "req-gate-failed"),
    )
    assert "120公里" not in _text_of(items)


def test_denied_retrieval_output_is_blocked(tmp_path: Path):
    items = _stream(
        _orchestrator(tmp_path, [], kernel_result=(
            "全程约120公里。",
            [{
                "tool": "hiking_knowledge_search",
                "phase": "denied",
                "code": "denied_by_gate",
            }],
        )),
        _req("查里程", "req-gate-denied"),
    )
    assert "120公里" not in _text_of(items)


def test_grounded_reply_passes_through(tmp_path: Path):
    expected = "根据已检索资料，携带离线地图并核对天气预警（来源：鳌太线攻略.md）。"
    items = _stream(
        _orchestrator(tmp_path, [], kernel_result=(expected, [_EVIDENCE_OK])),
        _req("徒步需要哪些安全准备", "req-gate-ok"),
    )
    assert _text_of(items) == expected


# ---- AC2: stream / final / persisted history share the gated content ---------


def test_gated_content_is_what_streams_and_persists(tmp_path: Path):
    repo = _repo(tmp_path)
    items = _stream(
        _orchestrator(tmp_path, [], runs_repo=repo),
        _req("这段路线爬升多少", "req-gate-persist"),
    )
    streamed = _text_of(items)
    assert UNSUPPORTED_FACT not in streamed

    run = repo.control.get_run_by_request("req-gate-persist")
    assert run is not None
    stored = repo.checkpoints.get_messages(conversation_id=run.conversation_id)
    assistant = [m for m in stored if m.role == "assistant"]
    assert assistant, "assistant reply must be persisted"
    assert assistant[-1].content == streamed
    assert UNSUPPORTED_FACT not in assistant[-1].content

    # Replay frames carry the SAME gated content (no late ungrounded text).
    replayed = _stream(
        _orchestrator(tmp_path, [], runs_repo=repo),
        _req("这段路线爬升多少", "req-gate-persist"),
    )
    assert _text_of(replayed) == streamed
    assert [e.sequence for e in replayed] == [e.sequence for e in items]


# ---- AC3: bypass attempts -----------------------------------------------------


def test_generate_then_retrieve_cannot_smuggle_facts(tmp_path: Path):
    """Model emits the claim first and only then searches — the final text
    is still gated on THIS-turn evidence, so an empty search blocks it."""
    items = _stream(
        _orchestrator(tmp_path, [], kernel_result=(
            UNSUPPORTED_FACT,
            [{
                "tool": "hiking_knowledge_search",
                "phase": "result",
                "ok": True,
                "evidence": {"has_chunks": False, "sources": []},
            }],
        )),
        _req("这段路线爬升多少", "req-gate-order"),
    )
    assert UNSUPPORTED_FACT not in _text_of(items)


def test_evidence_from_another_run_does_not_count(tmp_path: Path):
    foreign = dict(_EVIDENCE_OK)
    foreign["run_id"] = "run-someone-else"
    items = _stream(
        _orchestrator(tmp_path, [], kernel_result=(UNSUPPORTED_FACT, [foreign])),
        _req("这段路线爬升多少", "req-gate-foreign"),
    )
    assert UNSUPPORTED_FACT not in _text_of(items)
    # The scoping predicate itself is direct about it.
    assert not has_hiking_knowledge_evidence([foreign], run_id="run-mine")
    assert has_hiking_knowledge_evidence([_EVIDENCE_OK], run_id="run-mine")


def test_old_conversation_reference_is_not_evidence(tmp_path: Path):
    items = _stream(
        _orchestrator(tmp_path, [], kernel_result=(
            "根据上次检索到的资料，这段路线爬升约1800米。",
            [],
        )),
        _req("那爬升呢", "req-gate-history"),
    )
    assert "1800" not in _text_of(items)


# ---- AC4: controlled templates -------------------------------------------------


def test_greeting_uses_controlled_template(tmp_path: Path):
    items = _stream(
        _orchestrator(tmp_path, [], kernel_result=("你好呀，想聊点什么？", [])),
        _req("你好", "req-gate-hello"),
    )
    text = _text_of(items)
    assert "徒步助手" in text
    # The model's free greeting wording is not needed — template only.
    assert "想聊点什么" not in text


def test_clarification_uses_controlled_template(tmp_path: Path):
    items = _stream(
        _orchestrator(tmp_path, [], kernel_result=(
            "请问您的出行时间和路线对象是什么？",
            [],
        )),
        _req("帮我规划一条路线", "req-gate-clarify"),
    )
    text = _text_of(items)
    assert "请补充" in text
    assert "路线" in text


# ---- AC5/AC6: controlled domain results ---------------------------------------


def test_gear_success_renders_controlled_fields_with_limitations(tmp_path: Path):
    items = _stream(
        _orchestrator(tmp_path, [], kernel_result=(
            "装备清单如下：帐篷、睡袋、登山鞋。另外这条路线全程约120公里，建议三天走完。",
            [_GEAR_CONTROLLED],
        )),
        _req("两天冬季露营要带什么", "req-gate-gear"),
    )
    text = _text_of(items)
    # Controlled fields are present…
    assert "登山鞋" in text and "帐篷" in text
    assert "冬季" in text and "新手" in text
    # …basis and limitations are stated…
    assert "gear_checklist" in text
    assert "局限" in text
    # …and the model's unsupported route numbers are gone (AC6).
    assert "120公里" not in text
    assert "三天走完" not in text


def test_risk_success_renders_controlled_fields_with_limitations(tmp_path: Path):
    items = _stream(
        _orchestrator(tmp_path, [], kernel_result=(
            "风险很高。另外实时天气显示明天晴，气温 25 度。",
            [_RISK_CONTROLLED],
        )),
        _req("鳌太线暴雨天能去吗", "req-gate-risk"),
    )
    text = _text_of(items)
    assert "risk_assessment" in text
    assert "high" in text or "高" in text
    assert "取消或改期" in text
    # AC5: basis AND limitations are stated, mirroring the gear rendering.
    assert "局限" in text
    assert "风险评估按保守原则生成" in text
    # Unsupported live weather never rides along (AC6).
    assert "25 度" not in text
    assert "明天晴" not in text


def test_invalid_controlled_result_never_fakes_success(tmp_path: Path):
    invalid_gear = {
        "tool": "gear_checklist",
        "phase": "result",
        "ok": True,
        "controlled": {"tool": "gear_checklist", "ok": False, "code": "invalid_controlled_result"},
    }
    items = _stream(
        _orchestrator(tmp_path, [], kernel_result=(
            "放心，带双登山鞋就够了，这条路线很成熟。",
            [invalid_gear],
        )),
        _req("带什么装备", "req-gate-invalid"),
    )
    text = _text_of(items)
    assert "登山鞋" not in text  # no pseudo checklist rendered
    assert "资料不足" in text


def test_failed_domain_tool_never_fakes_success(tmp_path: Path):
    failed = {"tool": "gear_checklist", "phase": "result", "ok": False, "code": "tool_error"}
    items = _stream(
        _orchestrator(tmp_path, [], kernel_result=(
            "装备清单：帐篷、睡袋。",
            [failed],
        )),
        _req("带什么装备", "req-gate-toolfail"),
    )
    text = _text_of(items)
    assert "帐篷" not in text
    assert "资料不足" in text


def test_domain_tool_success_overrides_even_with_stale_evidence(tmp_path: Path):
    """A gear success plus another run's stale evidence must not launder
    the model's unsupported claims through the free-text path."""
    stale = dict(_EVIDENCE_OK)
    stale["run_id"] = "run-old"
    items = _stream(
        _orchestrator(tmp_path, [], kernel_result=(
            "帐篷、睡袋必备。这条路线全程约120公里。",
            [stale, _GEAR_CONTROLLED],
        )),
        _req("带什么装备", "req-gate-stale"),
    )
    text = _text_of(items)
    assert "120公里" not in text
    assert "帐篷" in text  # controlled rendering still applies


# ---- AC7: grounded progressive replies + terminal discipline ------------------


def test_grounded_turn_keeps_single_done_terminal(tmp_path: Path):
    items = _stream(
        _orchestrator(tmp_path, [], kernel_result=(
            "根据检索：鳌太线最佳窗口为 6-9 月。",
            [_EVIDENCE_OK],
        )),
        _req("鳌太线什么季节走合适？", "req-gate-term"),
    )
    types = [e.type for e in items]
    assert types[-1] == "done"
    assert [t for t in types if t in TERMINALS] == ["done"]


def test_cancelled_turn_has_single_terminal_and_no_late_text(tmp_path: Path):
    repo = _repo(tmp_path)
    run_id, _ = repo.create_run_idempotent(
        request_id="req-gate-cancel",
        conversation_id="conv-gate-cancel",
        profile_id="chat",
        user_message="这段路线爬升多少",
    )
    outcome = repo.request_cancel(run_id=run_id, idempotency_key="k-gate")
    assert outcome.accepted

    items = _stream(
        _orchestrator(tmp_path, [], runs_repo=repo),
        _req("这段路线爬升多少", "req-gate-cancel"),
    )
    types = [e.type for e in items]
    assert types[-1] == "cancelled"
    assert [t for t in types if t in TERMINALS] == ["cancelled"]
    assert "text" not in types  # no late ungrounded text after cancel
