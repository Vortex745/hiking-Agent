"""Ticket 01 — 计划书入口: the plan flow under the run orchestrator.

Seam under test: :meth:`agent_v2.run_orchestration.RunOrchestrator.stream_chat`
driven in-process with a fake kernel + real temp SQLite, the repo's
established style for orchestration ("no ASGI shell, no module-global
monkeypatching"). The fake kernel records every call, which is how these
tests prove the entry path never asks a model to decide anything.

Acceptance criteria proven here:

* AC1 明确计划书意图直接打开计划书弹窗 — a ``plan_create`` intent (and an
  unambiguous natural-language ask) lands on stage ``awaiting_input``.
* AC2 模糊规划意图先请求用户确认 — ``ambiguous`` text parks the draft in
  ``awaiting_confirmation`` and does NOT open the dialog; a later
  ``plan_confirm`` is what opens it.
* AC4 快捷入口直接发送结构化 intent；自然语言只经过确定性规则归一化 —
  neither path calls the kernel.
* AC5 服务端拒绝…与当前计划状态不匹配的 intent — a stable reason in the
  terminal payload.
* AC6 最小字段校验 — a submitted draft becomes a previewable minimal draft;
  a malformed one is rejected.
* AC8 服务端和前端都能识别计划书草稿状态 — the draft row and the
  ``plan.draft_state`` event carry the same stage, and the frontend event
  vocabulary registers the type (a dropped-as-unknown event is a silent
  bug the context pack warns about).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.plan_workflow import PlanDraftInput
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.runs import RunRepository

TERMINALS = {"error", "done", "cancelled"}
CONVERSATION = "conv-plan-01"
SUBJECT_ID = "sub-plan-01"


class _FakeMemory:
    def memory_enabled(self, subject_id): return True

    def claim_conversation(self, s, c): return True

    def owns_conversation(self, s, c): return True


def _repo(tmp_path: Path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        agent_v2_enabled=True,
        openai_api_key="sk-test",
        openai_model="test-model",
        agent_v2_cowork_policy="disabled",
        agent_v2_state_dir="/tmp/unused",
    )


@pytest.fixture
def plan_repo(tmp_path):
    """Repository shared by a whole test, closed at teardown.

    The drive helper must not close it early: several assertions read the
    stored draft state back through the same facade afterwards.
    """
    store = _repo(tmp_path)
    yield store
    store.close()


def _drive(repo: RunRepository, *requests: ChatFlowRequest) -> dict[str, Any]:
    """Drive requests through one orchestrator over the same repository."""
    kernel_calls: list[str] = []

    async def fake_kernel(**kwargs):
        kernel_calls.append(kwargs["message"])
        return "普通回答。", []

    orchestrator = RunOrchestrator(
        settings=_settings(), runs_repo=repo, memory=_FakeMemory(), kernel_turn=fake_kernel
    )

    async def drive() -> list[list[Any]]:
        return [
            [e async for e in orchestrator.stream_chat(req, subject_id=SUBJECT_ID)]
            for req in requests
        ]

    return {"streams": asyncio.run(drive()), "kernel_calls": kernel_calls, "repo": repo}


def _req(request_id: str, *, message: str = "生成行程", intent: str | None = None,
         plan_input: Any | None = None, conversation_id: str | None = CONVERSATION) -> ChatFlowRequest:
    return ChatFlowRequest(
        message=message,
        request_id=request_id,
        conversation_id=conversation_id,
        intent=intent,
        plan_input=plan_input,
    )


def _types(events) -> list[str]:
    return [e.type for e in events]


def _draft_events(events):
    return [e for e in events if e.type == "plan.draft_state"]


def _assert_stream_invariants(events):
    sequences = [e.sequence for e in events]
    assert sequences == sorted(sequences) and len(set(sequences)) == len(sequences)
    assert sequences[0] == 1
    assert [e.type for e in events if e.type in TERMINALS] == [events[-1].type]


def _valid_payload() -> dict:
    return {
        "destination_or_route": "四姑娘山长坪沟",
        "activity_date": "2026-11-08",
        "participants": 12,
    }


# ---- AC1 / AC4: the entry opens the dialog without any model call ---------


def test_structured_create_intent_opens_the_plan_dialog(plan_repo):
    """AC1 + AC4: the quick action's intent opens the dialog, no LLM decides."""
    result = _drive(plan_repo, _req("req-create", intent="plan_create"))
    events = result["streams"][0]

    _assert_stream_invariants(events)
    assert _types(events)[0] == "run_started"
    drafts = _draft_events(events)
    assert [d.payload["stage"] for d in drafts] == ["awaiting_input"]
    assert events[-1].type == "done"
    assert result["kernel_calls"] == []


def test_unambiguous_natural_language_enters_the_plan_flow(plan_repo):
    """AC4: free text is normalised by rules only, then behaves like create."""
    result = _drive(plan_repo, _req("req-nl-clear", message="帮我生成一份徒步计划书"))
    events = result["streams"][0]

    assert [d.payload["stage"] for d in _draft_events(events)] == ["awaiting_input"]
    assert events[-1].type == "done"
    assert result["kernel_calls"] == []


def test_ordinary_chat_still_goes_to_the_kernel(plan_repo):
    """AC4: rules must not hijack a question that is not a plan request."""
    result = _drive(plan_repo, _req("req-chat", message="这条路线爬升多少？", intent=None))
    events = result["streams"][0]

    assert _draft_events(events) == []
    assert result["kernel_calls"] == ["这条路线爬升多少？"]


# ---- AC2: ambiguous intent asks first, confirm then opens ------------------


def test_ambiguous_planning_text_asks_for_confirmation_without_opening(plan_repo):
    """AC2: ambiguous text must NOT open the plan flow."""
    result = _drive(plan_repo, _req("req-amb", message="周末想去爬山，还没定", intent=None))
    events = result["streams"][0]
    drafts = _draft_events(events)

    assert [d.payload["stage"] for d in drafts] == ["awaiting_confirmation"]
    assert all(d.payload["stage"] != "awaiting_input" for d in drafts)
    text = "".join(e.payload.get("content", "") for e in events if e.type == "text")
    assert "计划书" in text and ("是否" in text or "确认" in text)
    assert result["kernel_calls"] == []


def test_confirm_after_an_ambiguous_ask_opens_the_dialog(plan_repo):
    """AC2: only a subsequent ``plan_confirm`` proceeds into the flow."""
    result = _drive(plan_repo, 
        _req("req-amb", message="周末想去爬山，还没定"),
        _req("req-confirm", message="确认", intent="plan_confirm"),
    )
    second = result["streams"][1]

    assert [d.payload["stage"] for d in _draft_events(second)] == ["awaiting_input"]
    assert second[-1].type == "done"
    assert result["kernel_calls"] == []


# ---- AC5: rejections carry a stable machine-readable reason ----------------


def test_confirm_without_a_pending_draft_is_rejected_with_a_stable_reason(plan_repo):
    """AC5: state mismatch never opens anything and is machine-readable."""
    result = _drive(plan_repo, _req("req-confirm-cold", intent="plan_confirm"))
    events = result["streams"][0]

    assert _draft_events(events) == []
    assert events[-1].type == "error"
    assert events[-1].payload["reason"] == "confirm_without_pending_confirmation"
    assert result["kernel_calls"] == []
    assert result["repo"].get_plan_draft(CONVERSATION) is None


def test_regenerate_before_a_draft_exists_is_rejected(plan_repo):
    result = _drive(plan_repo, _req("req-regen-cold", intent="plan_regenerate"))
    events = result["streams"][0]

    assert events[-1].payload["reason"] == "regenerate_without_draft"


def test_intent_without_a_conversation_is_rejected_as_missing_context(plan_repo):
    """AC5: 缺少必要上下文 — the draft is keyed by conversation."""
    result = _drive(plan_repo, _req("req-no-conv", intent="plan_create", conversation_id=None))
    events = result["streams"][0]

    assert events[-1].type == "error"
    assert events[-1].payload["reason"] == "plan_context_required"
    assert result["kernel_calls"] == []


def test_intent_outside_the_whitelist_is_rejected_by_the_flow_too(plan_repo):
    """AC5: in-depth defence — the flow re-checks the whitelist itself."""
    result = _drive(plan_repo, _req("req-teleport", intent="plan_teleport"))
    events = result["streams"][0]

    assert events[-1].payload["reason"] == "unknown_intent"


def test_cancel_closes_the_open_plan_flow(plan_repo):
    result = _drive(plan_repo, 
        _req("req-create", intent="plan_create"),
        _req("req-cancel", message="先不用了", intent="plan_cancel"),
    )
    second = result["streams"][1]

    assert [d.payload["stage"] for d in _draft_events(second)] == ["cancelled"]
    assert second[-1].type == "done"
    assert result["repo"].get_plan_draft(CONVERSATION).stage == "cancelled"


# ---- AC6: the submitted form becomes a previewable minimal draft ----------


def test_submitted_fields_produce_a_previewable_minimal_draft(plan_repo):
    """AC6 + AC8: validated fields land in the draft row and the preview."""
    result = _drive(plan_repo, 
        _req("req-create", intent="plan_create"),
        _req("req-submit", message="提交草稿", intent="plan_create",
             plan_input=PlanDraftInput.model_validate(_valid_payload())),
    )
    second = result["streams"][1]
    drafts = _draft_events(second)

    assert drafts[-1].payload["stage"] == "minimal_draft"
    assert drafts[-1].payload["draft"] == {
        "destination_or_route": "四姑娘山长坪沟",
        "activity_date": "2026-11-08",
        "participants": 12,
    }
    stored = result["repo"].get_plan_draft(CONVERSATION)
    assert (stored.stage, stored.destination_or_route, stored.activity_date,
            stored.participants) == ("minimal_draft", "四姑娘山长坪沟", "2026-11-08", 12)

    preview = "".join(e.payload.get("content", "") for e in second if e.type == "text")
    assert "四姑娘山长坪沟" in preview and "2026-11-08" in preview and "12" in preview
    # Content boundary (docs/plan-pdf-format.md): uncollected arrangements
    # stay 待确认 and unverified facts stay 待核验 — never invented.
    assert "待确认" in preview and "待核验" in preview
    assert "草案" in preview


def test_unvalidated_draft_payload_is_rejected_as_invalid_fields(plan_repo):
    """AC6: a dict that skipped field validation never becomes a draft."""
    result = _drive(plan_repo, 
        _req("req-bad-submit", message="提交草稿", intent="plan_create",
             plan_input={"destination_or_route": "", "activity_date": "下周六"}),
    )
    events = result["streams"][0]

    assert events[-1].type == "error"
    assert events[-1].payload["reason"] == "invalid_draft_fields"
    assert result["repo"].get_plan_draft(CONVERSATION) is None


def test_regenerate_reuses_the_stored_draft_state(plan_repo):
    """AC8 seam for ticket 09: regenerate re-reads the persisted draft."""
    submit = _req("req-submit", message="提交草稿", intent="plan_create",
                  plan_input=PlanDraftInput.model_validate(_valid_payload()))
    result = _drive(plan_repo, submit, _req("req-regen", message="重新生成", intent="plan_regenerate"))
    second = result["streams"][1]

    assert second[-1].type == "done"
    assert _draft_events(second)[-1].payload["stage"] == "minimal_draft"
    assert _draft_events(second)[-1].payload["draft"]["activity_date"] == "2026-11-08"


def test_plan_entry_replays_are_idempotent(plan_repo):
    """A duplicate request_id replays the stored stream, no second draft."""
    first = _req("req-dup", intent="plan_create")
    result = _drive(plan_repo, first, _req("req-dup", intent="plan_create"))
    second = result["streams"][1]

    assert _types(second) == _types(result["streams"][0])
    assert result["repo"].get_plan_draft(CONVERSATION).stage == "awaiting_input"


# ---- AC8: the frontend recognises the same draft state --------------------


def _frontend_root() -> Path:
    return Path(__file__).resolve().parents[1] / "frontend"


def _backend_event_types() -> set[str]:
    """Read the ``EventType`` Literal members out of the events module."""
    source = Path("agent_v2/events.py").read_text(encoding="utf-8")
    block = re.search(r"EventType = Literal\[(.*?)\n\]", source, re.S)
    assert block, "EventType literal block not found"
    return set(re.findall(r'"([^"]+)"', block.group(1)))


def test_plan_draft_event_type_is_registered_on_both_sides():
    """AC8: a new EventType must exist in every frontend mapping layer.

    The context pack's landmine: forgetting one of the frontend layers
    silently drops the event as ``unknown_dropped`` instead of failing
    loudly, so the draft state would never reach the dialog.
    """
    assert "plan.draft_state" in _backend_event_types()

    frontend = _frontend_root()
    wire = (frontend / "src/api/chatV2.ts").read_text(encoding="utf-8")
    reducer = (frontend / "src/api/v2Reducer.ts").read_text(encoding="utf-8")
    page = (frontend / "src/pages/SuperAgent.tsx").read_text(encoding="utf-8")

    assert wire.count("'plan.draft_state'") >= 2, "union + type guard"
    assert "'plan.draft_state'" in reducer, "pure reducer must map the event"
    assert "'plan.draft_state'" in page, "UI side-effect chain must react to it"
