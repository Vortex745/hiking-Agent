"""DeepEval model adapter for the project's Jev/System One judge."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from deepeval.models.base_model import DeepEvalBaseLLM


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
        answer = (raw.get("answers") or {}).get("score") or {}
        score = float(answer.get("score", 0.0)) / 2.0
        reason = str(answer.get("reason") or answer.get("explanation") or "Jev returned a score")
        return json.dumps({"score": max(0.0, min(1.0, score)), "reason": reason}, ensure_ascii=False)

    def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        return self._request(prompt)

    async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        return self._request(prompt)
