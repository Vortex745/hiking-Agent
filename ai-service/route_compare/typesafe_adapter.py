"""Optional TypeSafe System One HTTP adapter; not imported by the default path."""
from __future__ import annotations

from pydantic import Field

from .models import Confidence, Contract, DecisionRequest, DecisionResult
from .providers import Provider, ProviderConfig


class ScoreAnswer(Contract):
    type: str
    score: float = Field(ge=0, le=4)
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float] | None = None
    legend: dict[str, str] | None = None


class TypeSafeProvider(Provider):
    name = "external"
    invalid_response_status = "unavailable"

    def __init__(self, config: ProviderConfig, *, client=None):
        super().__init__(config)
        self.model = config.external_model
        self.client = client

    async def evaluate(self, request: DecisionRequest) -> DecisionResult:
        if not self.config.external_key:
            return self.failure("missing_credentials")
        levels = ["与偏好明显不匹配", "基本不匹配", "部分匹配", "较匹配", "非常匹配"]
        body = {"model": self.model, "state": request.model_dump(), "questions": {
            cid: {"type": "score", "instructions": f"只根据 state 中候选 {cid} 的证据，"
                  "评估它对 preferences 的匹配程度；不得补充事实。",
                  "criteria": levels}
            for cid in request.candidate_ids
        }}
        if self.client is not None:
            raw = await self.client.evaluate(body)
        else:
            import httpx
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.post(self.config.external_url, json=body,
                    headers={"Authorization": "Bearer " + self.config.external_key})
                response.raise_for_status()
                raw = response.json()
        answers = {cid: ScoreAnswer.model_validate(raw["answers"][cid]) for cid in request.candidate_ids}
        if any(answer.type != "score" for answer in answers.values()):
            raise ValueError("wrong answer primitive")
        ranking = sorted(request.candidate_ids, key=lambda cid: -answers[cid].score)
        return DecisionResult(provider=self.name, model=raw["model"], status="ok", ranking=ranking,
                              confidence=Confidence(kind="calibrated",
                                  value=min(a.confidence for a in answers.values())))
