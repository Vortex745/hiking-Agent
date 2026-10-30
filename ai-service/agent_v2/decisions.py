"""Jev-style decision endpoints backed by SiliconFlow System One.

The runtime consumes typed choices, never free text.  The first endpoint is
``needs_retrieval``: the evidence gate asks whether the user's CURRENT message
requires this turn's knowledge-base retrieval.  A decision is advisory: a
timeout, an unconfigured key, an invalid answer, or low confidence all fall
back to the existing keyword gate, so a decision outage can never fail a turn.

SiliconFlow hosts the TypeSafe System One contract at
``POST {base_url}/systemone`` with ``{model, state, questions}`` and returns
``{model, answers, usage}``.  Only the choice/confidence fields are used.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from observability.redaction import scrub_text

logger = logging.getLogger("ai-service.decisions")

DECISION_MODES = frozenset({"off", "shadow", "enforce"})
RETRIEVAL_CHOICES = frozenset({"requires_retrieval", "general", "uncertain"})
# Both directions of the keyword override need this much confidence; below it
# the deterministic keyword gate stays authoritative.
CONFIDENCE_THRESHOLD = 0.85
DEFAULT_TIMEOUT_SECONDS = 5.0
# The gate may wait at most this long in total for the parallel decision.
GATE_WAIT_BUDGET_SECONDS = 3.0

_LOG_LOCK = threading.Lock()


class DecisionCallError(RuntimeError):
    """A typed, non-user-visible failure from the decision upstream."""

    def __init__(self, code: str, status: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


@dataclass(frozen=True, slots=True)
class RetrievalDecision:
    choice: str
    confidence: float
    model: str
    latency_ms: int
    mode: str
    fallback: bool = False


def decision_mode(settings: Any) -> str:
    mode = str(getattr(settings, "decision_model_mode", "off") or "off").lower()
    return mode if mode in DECISION_MODES else "off"


def build_needs_retrieval_question() -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": (
            "判断用户当前消息是否需要本轮徒步知识库检索才能可靠回答。"
            "路线、里程、海拔、耗时、天气、补给、营地、法规、风险等可核验徒步事实"
            "需要检索；通用知识、闲聊、通用装备建议不需要检索；无法判断时选 uncertain。"
        ),
        "criteria": {
            "requires_retrieval": "需要本轮知识库中的可核验徒步事实（数字、地点、时间、法规、风险）",
            "general": "通用知识、闲聊或通用装备建议，不需要本轮知识库证据",
            "uncertain": "信息不足，无法判断",
        },
    }


def build_state(message: str, history: list[dict[str, str]] | None) -> dict[str, Any]:
    turns: list[dict[str, str]] = []
    for entry in (history or ())[-2:]:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content:
            turns.append({"role": role, "content": content[:400]})
    return {"current_message": message, "recent_turns": turns}


def parse_choice_answer(
    payload: dict[str, Any], key: str
) -> tuple[str, float] | None:
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        return None
    answer = answers.get(key)
    if not isinstance(answer, dict):
        return None
    choice = answer.get("choice")
    if choice not in RETRIEVAL_CHOICES:
        return None
    confidence = answer.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        probabilities = answer.get("probabilities")
        confidence = (
            probabilities.get(choice)
            if isinstance(probabilities, dict)
            else None
        )
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = 0.0
    return str(choice), max(0.0, min(1.0, float(confidence)))


def decision_overrides_keywords(decision: RetrievalDecision | None) -> bool | None:
    """True=strict, False=pass, None=keep the keyword gate."""
    if decision is None or decision.fallback:
        return None
    if decision.mode != "enforce":
        return None
    if decision.confidence < CONFIDENCE_THRESHOLD:
        return None
    if decision.choice == "requires_retrieval":
        return True
    if decision.choice == "general":
        return False
    return None


class SystemOneClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds
        self._transport = transport

    async def ask(
        self, state: Any, questions: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        if not self._base_url or not self._api_key:
            raise DecisionCallError("not_configured")
        url = f"{self._base_url.rstrip('/')}/systemone"
        payload = {"model": self._model, "state": state, "questions": questions}
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
        except httpx.TimeoutException as exc:
            raise DecisionCallError("timeout") from exc
        except httpx.HTTPError as exc:
            raise DecisionCallError("transport_error") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code in (401, 403):
            raise DecisionCallError("upstream_auth", response.status_code)
        if response.status_code >= 400:
            raise DecisionCallError("upstream_error", response.status_code)
        try:
            data = response.json()
        except ValueError as exc:
            raise DecisionCallError("invalid_json", response.status_code) from exc
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            raise DecisionCallError("invalid_response", response.status_code)
        return data, latency_ms


def _record(
    *,
    settings: Any,
    run_id: str | None,
    mode: str,
    model: str,
    message: str,
    history_len: int,
    choice: str | None = None,
    confidence: float | None = None,
    latency_ms: int | None = None,
    fallback: bool = False,
    error: str | None = None,
) -> None:
    """Append one redacted JSONL record; logging failure never affects a turn."""
    try:
        state_dir = Path(getattr(settings, "agent_v2_state_dir", "./agent_state"))
        state_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id or "",
            "point": "needs_retrieval",
            "mode": mode,
            "model": model,
            "choice": choice,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "fallback": fallback,
            "error": error,
            "history_turns": history_len,
            "message_preview": scrub_text(message or "")[:300],
        }
        line = json.dumps(record, ensure_ascii=False)
        with _LOG_LOCK:
            with (state_dir / "decisions.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        logger.warning("decision record write failed", exc_info=True)


async def run_retrieval_decision(
    message: str,
    history: list[dict[str, str]] | None,
    *,
    settings: Any,
    run_id: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> RetrievalDecision | None:
    """Decide whether THIS message needs knowledge retrieval.

    Returns ``None`` for every non-answer (off, unconfigured, timeout, bad
    payload); the caller then keeps the existing keyword gate.
    """
    mode = decision_mode(settings)
    if mode == "off":
        return None
    base_url = str(getattr(settings, "decision_model_base_url", "") or "")
    api_key = str(getattr(settings, "decision_model_api_key", "") or "")
    model = str(getattr(settings, "decision_model_name", "Kev-4b") or "Kev-4b")
    timeout = float(
        getattr(settings, "decision_model_timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        or DEFAULT_TIMEOUT_SECONDS
    )
    history_len = len(history or ())
    if not base_url or not api_key:
        _record(
            settings=settings, run_id=run_id, mode=mode, model=model,
            message=message, history_len=history_len, fallback=True,
            error="not_configured",
        )
        return None
    client = SystemOneClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout,
        transport=transport,
    )
    try:
        data, latency_ms = await client.ask(
            build_state(message, history),
            {"needs_retrieval": build_needs_retrieval_question()},
        )
        parsed = parse_choice_answer(data, "needs_retrieval")
        if parsed is None:
            raise DecisionCallError("invalid_answer")
        choice, confidence = parsed
        decision = RetrievalDecision(
            choice=choice,
            confidence=confidence,
            model=model,
            latency_ms=latency_ms,
            mode=mode,
        )
        _record(
            settings=settings, run_id=run_id, mode=mode, model=model,
            message=message, history_len=history_len, choice=choice,
            confidence=confidence, latency_ms=latency_ms,
        )
        return decision
    except DecisionCallError as exc:
        _record(
            settings=settings, run_id=run_id, mode=mode, model=model,
            message=message, history_len=history_len, fallback=True,
            error=exc.code,
        )
        logger.info("decision needs_retrieval fallback code=%s", exc.code)
        return None


async def await_decision(
    task: "asyncio.Task[RetrievalDecision | None]",
    *,
    budget_seconds: float = GATE_WAIT_BUDGET_SECONDS,
) -> RetrievalDecision | None:
    """Take a finished decision immediately; otherwise wait up to ``budget``.

    The kernel turn often outlives the decision, so elapsed time must NOT
    count against the wait budget: a decision that completed while the model
    was working is still valid and must be consumed.
    """
    if task.cancelled():
        return None
    if task.done():
        try:
            return task.result()
        except Exception:
            logger.warning("decision task failed (non-fatal)", exc_info=True)
            return None
    try:
        return await asyncio.wait_for(task, timeout=budget_seconds)
    except TimeoutError:
        await asyncio.gather(task, return_exceptions=True)
        logger.info("decision needs_retrieval wait budget exhausted")
        return None
    except Exception:
        logger.warning("decision task failed (non-fatal)", exc_info=True)
        return None


async def probe_systemone(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[int, str]:
    """Connectivity probe for the config page; returns (latency_ms, choice)."""
    client = SystemOneClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )
    data, latency_ms = await client.ask(
        "connectivity test",
        {
            "probe": {
                "type": "choice",
                "instructions": "Is this a connectivity test?",
                "criteria": {"ok": "Yes, this request works", "fail": "No"},
            }
        },
    )
    answer = (data.get("answers") or {}).get("probe")
    choice = answer.get("choice") if isinstance(answer, dict) else None
    if choice not in {"ok", "fail"}:
        raise DecisionCallError("invalid_probe_answer")
    return latency_ms, str(choice)
