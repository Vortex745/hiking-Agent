"""Public route comparison contracts. Unknown facts remain absent."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Confidence(Contract):
    kind: Literal["heuristic", "calibrated", "unavailable"] = "unavailable"
    value: float | None = Field(default=None, ge=0, le=1)


class DecisionRequest(Contract):
    candidate_ids: list[str] = Field(min_length=1, max_length=5)
    state: dict[str, Any] = Field(default_factory=dict)
    preferences: str = Field(default="优先选择较轻松的路线", max_length=500)


class DecisionResult(Contract):
    provider: str
    model: str | None = None
    status: Literal["ok", "unavailable", "invalid_response"]
    reason: str | None = None
    ranking: list[str] = Field(default_factory=list, max_length=5)
    confidence: Confidence = Field(default_factory=Confidence)


class Source(Contract):
    id: str = Field(min_length=1, max_length=100)
    tool: str = Field(min_length=1, max_length=100)
    url: str | None = Field(default=None, max_length=1000)
    excerpt: str = Field(min_length=1, max_length=2000)
    observed_at: str


FactField = Literal["distance_km", "ascent_m", "duration_hours", "weather", "map", "trail_status", "risk"]


class Fact(Contract):
    value: str | float
    source_id: str
    # A canonical claim copied/converted from a tool's structured field.
    # Search prose is retained as a source, never promoted to a route number.
    source_field: str


class RouteCandidate(Contract):
    id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=100)
    facts: dict[FactField, Fact | None] = Field(default_factory=dict)
    sources: list[Source] = Field(default_factory=list, max_length=12)
    # Independently re-checkable canonical values produced by the adapter.
    evidence_values: dict[str, dict[str, str | float]] = Field(default_factory=dict)
    gaps: list[str] = Field(default_factory=list)
    data_mode: Literal["live", "offline"] = "live"


class RouteQuery(Contract):
    name: str = Field(min_length=1, max_length=100)
    city: str = Field(min_length=1, max_length=80)
    origin: str = Field(default="", max_length=100)
    destination: str = Field(min_length=1, max_length=100)


class CompareParams(Contract):
    candidates: list[RouteQuery] = Field(min_length=1, max_length=5)
    preferences: str = Field(default="优先选择较轻松的路线", max_length=500)


class CandidateAssessment(Contract):
    candidate: RouteCandidate
    evidence_status: Literal["complete", "partial", "empty"]
    risk_status: Literal["clear", "caution", "confirmation_required", "no_go"]
    reasons: list[str]
    required_action: str | None = None


class PlanStage(Contract):
    time: str = Field(max_length=100)
    activity: str = Field(max_length=300)


class BudgetItem(Contract):
    item: str = Field(min_length=1, max_length=100)
    amount_yuan: float = Field(ge=0, le=1000000)


class PlanDetails(Contract):
    """User arrangements, distinct from independently verified route facts."""
    activity_date: str = Field(default="", max_length=100)
    meeting_time: str = Field(default="", max_length=100)
    meeting_point: str = Field(default="", max_length=300)
    participants: int | None = Field(default=None, ge=1, le=10000)
    leader: str = Field(default="", max_length=100)
    transport: str = Field(default="", max_length=500)
    accommodation: str = Field(default="", max_length=500)
    emergency_contact: str = Field(default="", max_length=300)
    retreat_plan: str = Field(default="", max_length=500)
    itinerary: list[PlanStage] = Field(default_factory=list, max_length=30)
    budget: list[BudgetItem] = Field(default_factory=list, max_length=20)


class ComparisonResult(Contract):
    comparison_id: str
    status: Literal["recommended", "confirmation_required", "insufficient_evidence", "no_go", "provider_unavailable"]
    decision: DecisionResult
    candidates: list[CandidateAssessment]
    ranking: list[str] = Field(default_factory=list)
    recommended_id: str | None = None
    alternatives: list[str] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)
    required_action: str | None = None
    generated_at: str
    report_allowed: bool = False
    data_mode: Literal["live", "offline"] = "live"
    plan_details: PlanDetails = Field(default_factory=PlanDetails)

    @model_validator(mode="after")
    def validate_report(self):
        if self.report_allowed and (self.status != "recommended" or not self.recommended_id):
            raise ValueError("A report requires a verified recommendation")
        return self
