"""SiliconFlow System One decision contract.

These tests exercise the real client serialization/parsing against a mock
HTTP transport; the upstream service itself is covered by the separate live
probe, not by a fake decision.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx

from agent_v2.decisions import (
    RetrievalDecision,
    SystemOneClient,
    decision_overrides_keywords,
    parse_choice_answer,
    probe_systemone,
    run_retrieval_decision,
)


def _settings(tmp_path: Path, **overrides):
    base = {
        "decision_model_base_url": "https://decision.example/v1",
        "decision_model_api_key": "test-key",
        "decision_model_name": "Kev-4b",
        "decision_model_mode": "enforce",
        "decision_model_timeout_seconds": 2.0,
        "agent_v2_state_dir": str(tmp_path / "state"),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _decision(choice: str, confidence: float = 0.95, *, mode: str = "enforce", fallback: bool = False):
    return RetrievalDecision(
        choice=choice,
        confidence=confidence,
        model="Kev-4b",
        latency_ms=120,
        mode=mode,
        fallback=fallback,
    )


def _response_payload(choice: str = "requires_retrieval", confidence: float = 0.95) -> dict:
    return {
        "model": "Kev-4b",
        "answers": {
            "needs_retrieval": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": 120, "output_tokens": 8},
    }


def test_parse_doc_choice_shape():
    assert parse_choice_answer(
        {"answers": {"needs_retrieval": {"type": "choice", "choice": "general", "confidence": 0.91}}},
        "needs_retrieval",
    ) == ("general", 0.91)


def test_parse_probabilities_when_confidence_missing():
    payload = {
        "answers": {
            "needs_retrieval": {
                "type": "choice",
                "choice": "general",
                "probabilities": {"general": 0.8, "requires_retrieval": 0.2},
            }
        }
    }
    assert parse_choice_answer(payload, "needs_retrieval") == ("general", 0.8)


def test_parse_rejects_unknown_choice_and_missing_answer():
    assert parse_choice_answer(
        {"answers": {"needs_retrieval": {"choice": "banana", "confidence": 0.9}}},
        "needs_retrieval",
    ) is None
    assert parse_choice_answer({"answers": {}}, "needs_retrieval") is None
    assert parse_choice_answer({}, "needs_retrieval") is None


def test_parse_clamps_confidence():
    assert parse_choice_answer(
        {"answers": {"needs_retrieval": {"choice": "general", "confidence": 1.7}}},
        "needs_retrieval",
    ) == ("general", 1.0)
    assert parse_choice_answer(
        {"answers": {"needs_retrieval": {"choice": "general", "confidence": -1}}},
        "needs_retrieval",
    ) == ("general", 0.0)


def test_override_requires_enforce_and_confidence():
    assert decision_overrides_keywords(_decision("general")) is False
    assert decision_overrides_keywords(_decision("requires_retrieval")) is True
    assert decision_overrides_keywords(_decision("uncertain")) is None
    assert decision_overrides_keywords(_decision("general", 0.5)) is None
    assert decision_overrides_keywords(_decision("general", mode="shadow")) is None
    assert decision_overrides_keywords(_decision("general", fallback=True)) is None
    assert decision_overrides_keywords(None) is None


def test_run_decision_serializes_doc_contract_and_logs(tmp_path: Path):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.path == "/v1/systemone"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "Kev-4b"
        assert body["state"]["current_message"] == "鳌太线全程多少公里？"
        assert body["questions"]["needs_retrieval"]["type"] == "choice"
        assert set(body["questions"]["needs_retrieval"]["criteria"]) == {
            "requires_retrieval",
            "general",
            "uncertain",
        }
        return httpx.Response(200, json=_response_payload())

    settings = _settings(tmp_path)
    decision = asyncio.run(
        run_retrieval_decision(
            "鳌太线全程多少公里？",
            [{"role": "user", "content": "上一轮问的是鳌太线"}],
            settings=settings,
            run_id="run-decision-1",
            transport=httpx.MockTransport(handler),
        )
    )
    assert decision is not None
    assert decision.choice == "requires_retrieval"
    assert decision.confidence == 0.95
    assert decision.mode == "enforce"
    assert len(seen) == 1
    lines = (tmp_path / "state" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["run_id"] == "run-decision-1"
    assert record["choice"] == "requires_retrieval"
    assert record["fallback"] is False
    assert "test-key" not in lines[0]


def test_mode_off_makes_no_call_and_writes_nothing(tmp_path: Path):
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json=_response_payload())

    result = asyncio.run(
        run_retrieval_decision(
            "鳌太线全程多少公里？",
            [],
            settings=_settings(tmp_path, decision_model_mode="off"),
            transport=httpx.MockTransport(handler),
        )
    )
    assert result is None
    assert called is False
    assert not (tmp_path / "state" / "decisions.jsonl").exists()


def test_not_configured_falls_back_and_logs(tmp_path: Path):
    result = asyncio.run(
        run_retrieval_decision(
            "鳌太线全程多少公里？",
            [],
            settings=_settings(tmp_path, decision_model_api_key=""),
        )
    )
    assert result is None
    record = json.loads(
        (tmp_path / "state" / "decisions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert record["fallback"] is True
    assert record["error"] == "not_configured"


def test_timeout_falls_back_and_logs(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    result = asyncio.run(
        run_retrieval_decision(
            "鳌太线全程多少公里？",
            [],
            settings=_settings(tmp_path),
            transport=httpx.MockTransport(handler),
        )
    )
    assert result is None
    record = json.loads(
        (tmp_path / "state" / "decisions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert record["fallback"] is True
    assert record["error"] == "timeout"


def test_upstream_auth_failure_falls_back(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"code": 30014, "message": "Token is invalid."})

    result = asyncio.run(
        run_retrieval_decision(
            "鳌太线全程多少公里？",
            [],
            settings=_settings(tmp_path),
            transport=httpx.MockTransport(handler),
        )
    )
    assert result is None
    record = json.loads(
        (tmp_path / "state" / "decisions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert record["error"] == "upstream_auth"


def test_invalid_answer_falls_back(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": {"needs_retrieval": {"choice": "banana"}}})

    result = asyncio.run(
        run_retrieval_decision(
            "鳌太线全程多少公里？",
            [],
            settings=_settings(tmp_path),
            transport=httpx.MockTransport(handler),
        )
    )
    assert result is None
    record = json.loads(
        (tmp_path / "state" / "decisions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert record["error"] == "invalid_answer"


def test_probe_systemone_returns_choice():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answers": {
                    "probe": {"type": "choice", "choice": "ok", "confidence": 0.99}
                }
            },
        )

    latency, choice = asyncio.run(
        probe_systemone(
            base_url="https://decision.example/v1",
            api_key="test-key",
            model="Kev-4b",
            transport=httpx.MockTransport(handler),
        )
    )
    assert latency >= 0
    assert choice == "ok"


def test_await_decision_consumes_a_finished_task_after_a_long_kernel_turn():
    from agent_v2.decisions import await_decision

    async def run():
        task = asyncio.create_task(asyncio.sleep(0.01, result=_decision("general")))
        await asyncio.sleep(0.05)
        return await await_decision(task, budget_seconds=0.001)

    result = asyncio.run(run())
    assert result is not None
    assert result.choice == "general"


def test_await_decision_times_out_an_unfinished_task():
    from agent_v2.decisions import await_decision

    async def run():
        task = asyncio.create_task(asyncio.sleep(1, result=_decision("general")))
        return await await_decision(task, budget_seconds=0.01)

    assert asyncio.run(run()) is None
