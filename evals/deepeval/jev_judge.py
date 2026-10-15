"""DeepEval model adapter for the project's Jev/System One judge.

Score legality (spec T10): a judge answer maps to a score ONLY when it
carries a numeric, finite value inside the judge's declared range
(0..2, normalized to 0..1). Missing / null / wrong-type / non-finite /
out-of-range scores raise :class:`JevJudgeResponseError` — they are NEVER
silently defaulted to a valid zero. A LEGAL zero score is returned as 0.0
and participates in the real threshold decision.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import httpx
from deepeval.models.base_model import DeepEvalBaseLLM

# The Jev score question declares three levels (0 / 1 / 2).
RAW_SCORE_MIN = 0.0
RAW_SCORE_MAX = 2.0


class JevJudgeResponseError(RuntimeError):
    """Judge response cannot produce a legal score (no default-zero合成)."""


class JevJudge(DeepEvalBaseLLM):
    """Translate DeepEval's rubric prompt into a Jev score question."""

    def __init__(self) -> None:
        self.api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
        self.endpoint = os.getenv("JEV_ENDPOINT", "https://api.typesafe.ai/v1/systemone")
        self._jev_model = os.getenv("JEV_MODEL", "jev-latest")
        super().__init__(model=self._jev_model)

    def load_model(self) -> "JevJudge":
        return self

    def get_model_name(self) -> str:
        return self._jev_model

    # ---- 分数合法性 -------------------------------------------------------

    def parse_score_payload(self, raw: Any) -> tuple[float, dict[str, Any]]:
        """解析 Jev 原始响应 → (归一化分数, DeepEval 需要的 JSON 载荷)。

        非法返回（缺失/null/非法类型/非有限/越界）一律抛
        :class:`JevJudgeResponseError`；合法零分正常返回 0.0。
        """
        answer: Any = None
        if isinstance(raw, dict):
            answers = raw.get("answers")
            if isinstance(answers, dict):
                answer = answers.get("score")
        if not isinstance(answer, dict):
            raise JevJudgeResponseError(
                "judge response has no usable 'score' answer object"
            )
        if "score" not in answer:
            raise JevJudgeResponseError("judge response is missing the 'score' field")
        raw_score = answer["score"]
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float, str)):
            raise JevJudgeResponseError(
                f"judge score has an illegal type: {type(raw_score).__name__}"
            )
        try:
            number = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise JevJudgeResponseError(
                f"judge score is not numeric: {raw_score!r}"
            ) from exc
        if not math.isfinite(number):
            raise JevJudgeResponseError(f"judge score is not finite: {raw_score!r}")
        if number < RAW_SCORE_MIN or number > RAW_SCORE_MAX:
            raise JevJudgeResponseError(
                f"judge score {raw_score!r} is outside the declared range "
                f"[{RAW_SCORE_MIN}, {RAW_SCORE_MAX}]"
            )
        score = number / RAW_SCORE_MAX
        reason = str(
            answer.get("reason") or answer.get("explanation") or "Jev returned a score"
        )
        return score, {"score": score, "reason": reason}

    def _request(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY is required for JevJudge")
        body: dict[str, Any] = {
            "model": self._jev_model,
            "state": {"evaluation_prompt": prompt},
            "questions": {
                "score": {
                    "type": "score",
                    "criteria": ["The response fails the criterion", "The response partially meets the criterion", "The response fully meets the criterion"],
                    "instructions": "Evaluate only the supplied evaluation prompt. Do not invent facts. Return a score and concise reason.",
                }
            },
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(self.endpoint, json=body, headers={"Authorization": f"Bearer {self.api_key}"})
            response.raise_for_status()
            raw = response.json()
        score, payload = self.parse_score_payload(raw)
        return json.dumps(payload, ensure_ascii=False)

    def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        return self._request(prompt)

    async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        return self._request(prompt)
