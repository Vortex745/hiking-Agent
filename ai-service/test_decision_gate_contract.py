"""The evidence gate consumes a typed retrieval decision without changing its
historical signature or its deterministic fallbacks."""

from __future__ import annotations

from agent_v2.decisions import RetrievalDecision
from agent_v2.run_orchestration import (
    ChatFlowRequest,
    _CONTROLLED_GREETING_RESPONSE,
    _NO_KNOWLEDGE_EVIDENCE_RESPONSE,
    apply_chat_turn_evidence_gate,
)

MODEL_TEXT = "这是一段模型自由回答。"


def _decision(
    choice: str,
    confidence: float = 0.95,
    *,
    mode: str = "enforce",
    fallback: bool = False,
) -> RetrievalDecision:
    return RetrievalDecision(
        choice=choice,
        confidence=confidence,
        model="Kev-4b",
        latency_ms=100,
        mode=mode,
        fallback=fallback,
    )


def _req(message: str, *, profile: str = "chat", decision=None) -> ChatFlowRequest:
    return ChatFlowRequest(
        message=message,
        request_id="req-decision",
        profile_id=profile,
        conversation_id="conv-decision",
        retrieval_decision=decision,
    )


def _gate(message: str, *, profile: str = "chat", decision=None, audits=None, run_id=None) -> str:
    return apply_chat_turn_evidence_gate(
        MODEL_TEXT,
        audits or [],
        req=_req(message, profile=profile, decision=decision),
        run_id=run_id,
    )


def _evidence(run_id: str = "run-1") -> list[dict]:
    return [
        {
            "tool": "hiking_knowledge_search",
            "phase": "result",
            "ok": True,
            "run_id": run_id,
            "evidence": {"has_chunks": True},
        }
    ]


def test_general_decision_passes_a_hiking_keyword_message():
    text = _gate(
        "户外活动对心理健康有什么好处？",
        decision=_decision("general"),
    )
    assert text == MODEL_TEXT


def test_requires_retrieval_decision_tightens_a_keyword_miss():
    text = _gate(
        "那条线路要走几天？",
        decision=_decision("requires_retrieval"),
    )
    assert text == _NO_KNOWLEDGE_EVIDENCE_RESPONSE


def test_requires_retrieval_decision_tightens_even_without_any_keyword():
    text = _gate(
        "今天心情有点差。",
        decision=_decision("requires_retrieval"),
    )
    assert text == _NO_KNOWLEDGE_EVIDENCE_RESPONSE


def test_low_confidence_falls_back_to_the_keyword_gate():
    text = _gate(
        "户外活动对心理健康有什么好处？",
        decision=_decision("general", confidence=0.5),
    )
    assert text == _NO_KNOWLEDGE_EVIDENCE_RESPONSE


def test_shadow_decision_never_changes_the_gate():
    text = _gate(
        "户外活动对心理健康有什么好处？",
        decision=_decision("general", mode="shadow"),
    )
    assert text == _NO_KNOWLEDGE_EVIDENCE_RESPONSE


def test_fallback_decision_never_changes_the_gate():
    text = _gate(
        "户外活动对心理健康有什么好处？",
        decision=_decision("general", fallback=True),
    )
    assert text == _NO_KNOWLEDGE_EVIDENCE_RESPONSE


def test_rag_strict_ignores_a_general_decision():
    text = _gate(
        "今天心情有点差。",
        profile="rag",
        decision=_decision("general"),
    )
    assert text == _NO_KNOWLEDGE_EVIDENCE_RESPONSE


def test_evidence_still_wins_over_a_strict_decision():
    text = _gate(
        "鳌太线全程多少公里？",
        decision=_decision("requires_retrieval"),
        audits=_evidence(),
        run_id="run-1",
    )
    assert text == MODEL_TEXT


def test_greeting_template_still_wins():
    text = _gate("你好", decision=_decision("general"))
    assert text == _CONTROLLED_GREETING_RESPONSE


def test_no_decision_keeps_todays_behavior():
    assert _gate("户外活动对心理健康有什么好处？") == _NO_KNOWLEDGE_EVIDENCE_RESPONSE
    assert _gate("今天心情有点差。") == MODEL_TEXT


def test_decision_result_rides_on_the_request_not_the_gate_signature():
    req = _req("户外活动对心理健康有什么好处？", decision=_decision("general"))
    assert req.retrieval_decision is not None
    assert (
        apply_chat_turn_evidence_gate(
            MODEL_TEXT, [], req=req, run_id=None
        )
        == MODEL_TEXT
    )
