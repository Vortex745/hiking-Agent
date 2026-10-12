"""Run-scoped kernel leaves for route comparison, report preparation and the
plan draft's route reference.
"""
from __future__ import annotations

import asyncio
import json

from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from agent_v2.kernel_tools import BoundedDomainTool
from agent_v2.plan_workflow import (
    FACT_DIMENSIONS,
    PlanBlockReason,
    RouteBasis,
    RouteFactLine,
    format_fact_value,
)
from route_compare.data import collect_candidates
from route_compare.fixtures import fixture_candidates
from route_compare.models import (CompareParams, CandidateAssessment, ComparisonResult, Contract,
                                  PlanDetails, RouteCandidate, RouteQuery)
from route_compare.providers import ProviderConfig, select_provider
from route_compare.service import assess_candidate, compare_routes, render_report

AMAP_OPERATIONS = {
    ("amap_map", "geocode"): "maps_geo",
    ("amap_map", "plan_route"): "maps_direction_walking",
    ("amap_weather", "get_current_weather"): "maps_weather",
}


class LiveTools:
    """Uses the existing MCP registry/client and route research leaf, with audit."""
    def __init__(self, settings: Any, audit_sink=None):
        self.settings = settings
        self.audit_sink = audit_sink

    async def call(self, server: str, tool: str, args: dict) -> dict:
        if self.audit_sink:
            self.audit_sink({"phase": "call", "tool": tool, "server_id": server})
        code = "tool_unavailable"
        try:
            if server == "local" and tool == "route_research":
                from tools.hiking_domain import route_research
                async with asyncio.timeout(18):
                    result = await route_research.ainvoke(args)
                cap = 16000
            else:
                from agent_v2.mcp_registry import expand_credentials, resolve_mcp_server
                from uniharness.mcp import McpClient
                cfg = resolve_mcp_server(server)
                remote_tool = AMAP_OPERATIONS.get((server, tool))
                if remote_tool not in cfg.tool_allowlist:
                    raise ValueError("not_allowlisted")
                env, headers = expand_credentials(cfg, self.settings)
                config = ({"type": "stdio", "command": cfg.command_or_url,
                           "args": list(cfg.args), "env": env} if cfg.transport == "stdio" else
                          {"type": cfg.transport, "url": cfg.command_or_url, "headers": headers})
                async with asyncio.timeout(cfg.timeout_seconds):
                    async with McpClient(server, config) as client:
                        name = f"mcp__{server}__{remote_tool}"
                        adapter = next(t for t in client.tools if t.name == name)
                        output = await adapter.execute(adapter.args_schema.model_validate(args))
                        if output.error:
                            raise ValueError("mcp_error")
                        result = json.loads(output.output or "{}")
                cap = cfg.max_output_bytes
            if not isinstance(result, dict) or len(json.dumps(result).encode()) > cap:
                raise ValueError("invalid_or_oversized_output")
            if self.audit_sink:
                self.audit_sink({"phase": "result", "tool": tool, "server_id": server, "ok": True})
            return result
        except TimeoutError:
            code = "tool_timeout"
        except Exception:
            pass
        if self.audit_sink:
            self.audit_sink({"phase": "result", "tool": tool, "server_id": server, "ok": False, "code": code})
        return {"unavailable": code}


def provider_config(settings: Any) -> ProviderConfig:
    return ProviderConfig(provider=getattr(settings, "route_decision_provider", "llm"),
        model=getattr(settings, "openai_model", "deepseek-v4-flash"),
        timeout_seconds=getattr(settings, "route_decision_timeout_seconds", 15.0),
        data_mode=getattr(settings, "route_compare_data_mode", "live"),
        base_url=getattr(settings, "openai_base_url", ""), api_key=getattr(settings, "openai_api_key", ""),
        external_adapter=getattr(settings, "route_decision_external_adapter", "route_compare.typesafe_adapter:TypeSafeProvider"),
        external_model=getattr(settings, "route_decision_external_model", "jev-latest"),
        external_url=getattr(settings, "route_decision_external_url", "https://api.typesafe.ai/v1/systemone"),
        external_key=getattr(settings, "typesafe_api_key", ""))


class ReportParams(Contract):
    comparison_id: str = Field(min_length=1, max_length=150)
    plan_details: PlanDetails | None = None


@dataclass(frozen=True, slots=True)
class RouteSelectionOutcome:
    """Result of resolving a draft's route reference (ticket 03).

    Same ``status``/``reason`` shape :meth:`RouteSession.prepare_report`
    returns, so a refusal is a stable machine-readable reason and never a
    half-filled draft: blocked means the caller renders every route fact as
    待核验, which is exactly spec AC4.
    """

    status: Literal["ok", "blocked"]
    basis: RouteBasis | None = None
    reason: str | None = None

    @property
    def blocked(self) -> bool:
        return self.status == "blocked"


@dataclass(frozen=True, slots=True)
class RouteResearchOutcome:
    """Result of one route-research turn for a draft's free destination (04).

    Same ``status``/``reason`` discipline as :class:`RouteSelectionOutcome`,
    with one extra honest case: ``unavailable`` means research could not run
    at all (no data source configured), which is NOT the same statement as
    ``insufficient`` (it ran and nothing was verifiable). Only ``ok`` may be
    labelled a route plan, and the label comes from the gate's own answer.

    ``result`` is the stored :class:`route_compare.models.ComparisonResult`
    itself — candidates, sources, gaps and all — so the caller persists and
    renders the evidence rather than a paraphrase of it. It is present even for
    ``insufficient``, because "here is what we found and why it does not
    qualify" is the informative answer (spec AC2/AC3).
    """

    status: Literal["ok", "insufficient", "unavailable"]
    result: ComparisonResult | None = None
    reason: str | None = None
    gaps: tuple[str, ...] = ()

    @property
    def insufficient(self) -> bool:
        return self.status != "ok"


def _research_query(destination_or_route: str) -> dict:
    """Turn one free-text destination into a bounded research query.

    ``RouteQuery``'s field caps are the transport's, not the user's: a
    120-character 目的地 or a long route description is a legal draft field
    (:data:`agent_v2.plan_workflow._DESTINATION_MAX`) and simply does not fit a
    search term, so each slot carries the same clipped text. Nothing is
    invented here — no city, no origin, no "nearby" route is guessed to make
    the query look richer, and a clipped term is still the user's own words.
    """
    text = (destination_or_route or "").strip()
    return {
        "name": text[:100],
        "city": text[:80],
        "origin": text[:100],
        "destination": text[:100],
    }


def _route_basis(result: ComparisonResult, assessment: CandidateAssessment,
                 role: Literal["recommended", "alternative"]) -> RouteBasis:
    """Project a re-assessed candidate onto the draft's fact section (AC2).

    Only facts the gate left standing appear, each with the source that backs
    it: ``source_id`` + ``source_field`` come from the fact itself and
    ``source_tool`` / ``observed_at`` from the candidate's own source list. A
    fact whose source cannot be found is dropped rather than printed bare —
    a number without provenance is not a verified fact.
    """
    candidate = assessment.candidate
    sources = {source.id: source for source in candidate.sources}
    lines: list[RouteFactLine] = []
    missing: list[str] = []
    for field, label, _unit in FACT_DIMENSIONS:
        fact = candidate.facts.get(field)
        source = sources.get(fact.source_id) if fact is not None else None
        if fact is None or source is None:
            missing.append(label)
            continue
        lines.append(RouteFactLine(
            field=field,
            label=label,
            display_value=format_fact_value(field, fact.value),
            source_id=fact.source_id,
            source_field=fact.source_field,
            source_tool=source.tool,
            observed_at=source.observed_at,
        ))
    return RouteBasis(
        comparison_id=result.comparison_id,
        candidate_id=candidate.id,
        candidate_name=candidate.name,
        role=role,
        data_mode=result.data_mode,
        facts=tuple(lines),
        missing_labels=tuple(missing),
    )


class RouteSession:
    def __init__(self, *, settings, run_id, audit_sink, runs_repo=None, conversation_id=None,
                 provider=None, tools=None):
        self.config = provider_config(settings)
        self.settings = settings
        self.run_id = run_id or "ephemeral"
        self.provider = provider or select_provider(self.config)
        self.tools = tools or LiveTools(settings, audit_sink)
        self.results: dict[str, ComparisonResult] = {}
        self.runs_repo = runs_repo
        self.conversation_id = conversation_id

    async def compare(self, candidates: list[dict], preferences: str = "优先选择较轻松的路线") -> dict:
        queries = [RouteQuery.model_validate(query) for query in candidates]
        if self.config.data_mode == "offline":
            samples = {c.name: c for c in fixture_candidates(
                getattr(self.settings, "route_compare_demo_scenario", "complete"))}
            collected = [samples.get(q.name) or RouteCandidate(id=f"unknown-{i}", name=q.name,
                         data_mode="offline", gaps=["离线样例未包含该路线"])
                         for i, q in enumerate(queries)]
        elif self.config.data_mode == "live":
            collected = await collect_candidates(queries, self.tools)
        else:
            raise ValueError("unknown_data_mode")
        result = await compare_routes(collected, self.provider, preferences=preferences,
                                      comparison_id=self.run_id + ":" + str(uuid4()))
        self.results[result.comparison_id] = result.model_copy(deep=True)
        return result.model_dump()

    def load_result(self, comparison_id: str) -> ComparisonResult | None:
        if comparison_id in self.results:
            return self.results[comparison_id].model_copy(deep=True)
        if not self.runs_repo or not self.conversation_id or ":" not in comparison_id:
            return None
        run_id = comparison_id.rsplit(":", 1)[0]
        run = self.runs_repo.get_run(run_id)
        if run is None or run.conversation_id != self.conversation_id:
            return None
        for event in reversed(self.runs_repo.get_events_since(run_id=run_id, from_seq=0)):
            raw = event.get("payload", {}).get("route_comparison")
            if isinstance(raw, dict) and raw.get("comparison_id") == comparison_id:
                return ComparisonResult.model_validate(raw)
        return None

    async def prepare_report(self, comparison_id: str, plan_details: dict | None = None) -> dict:
        result = self.load_result(comparison_id)
        if result is None:
            return {"status": "blocked", "reason": "comparison_not_found"}
        if plan_details is not None:
            result.plan_details = PlanDetails.model_validate(plan_details)
        try:
            render_report(result)
        except ValueError:
            return {"status": "blocked", "reason": "evidence_or_safety_gate"}
        return {"status": "ready_for_approval", "comparison_id": comparison_id,
                "route_report": result.model_dump()}

    def select_candidate(self, comparison_id: str, candidate_id: str) -> RouteSelectionOutcome:
        """Resolve a plan draft's route reference into verified facts (ticket 03).

        This is the only way the plan path reaches a comparison, and it is
        called on *every* use rather than once at selection time:
        :meth:`load_result` re-reads and re-validates the stored result, and
        :func:`route_compare.service.assess_candidate` — the same deterministic
        gate :func:`render_report` re-runs before it trusts a stored result —
        re-checks the candidate. So a draft can hold nothing but two ids and
        still cannot carry a fact that its evidence no longer supports.

        Selectable is the recommendation or a listed alternative of a gate
        passed comparison (spec AC1), which is a rule over stored fields, not
        a model call. Report eligibility is untouched here: this leaf makes no
        ``report_allowed`` claim and never renders a report — that stays
        :func:`render_report`'s authority.
        """
        result = self.load_result(comparison_id)
        if result is None:
            return RouteSelectionOutcome("blocked", reason=PlanBlockReason.COMPARISON_NOT_FOUND)

        selectable = ([result.recommended_id] if result.recommended_id else []) + list(
            result.alternatives)
        if result.status != "recommended" or candidate_id not in selectable:
            # No usable comparison, or a candidate the comparison did not offer
            # (a no-go or insufficient-evidence route): AC4 forbids dressing it
            # up as a basis for the plan.
            return RouteSelectionOutcome("blocked", reason=PlanBlockReason.ROUTE_NOT_SELECTABLE)

        stored = next((a for a in result.candidates if a.candidate.id == candidate_id), None)
        if stored is None:
            return RouteSelectionOutcome("blocked", reason=PlanBlockReason.ROUTE_NOT_SELECTABLE)

        checked = assess_candidate(stored.candidate)
        if checked.evidence_status != "complete" or checked.risk_status != "clear":
            return RouteSelectionOutcome("blocked", reason=PlanBlockReason.ROUTE_NOT_SELECTABLE)

        role: Literal["recommended", "alternative"] = (
            "recommended" if candidate_id == result.recommended_id else "alternative")
        return RouteSelectionOutcome("ok", basis=_route_basis(result, checked, role))

    async def research_destination(self, destination_or_route: str) -> RouteResearchOutcome:
        """Try to research the draft's own free destination or route (ticket 04).

        This is a *branch*, not a second gate: it hands one query to the same
        :meth:`compare` the kernel tool uses — ``fixture_candidates`` offline,
        :func:`route_compare.data.collect_candidates` live — and then reads the
        verdict :func:`route_compare.service.compare_routes` already reached
        through :func:`assess_candidate`. A candidate counts as evidence only
        when its facts are source-backed, non-negative and its weather /
        open-status / risk clear the freshness rules; that is the gate's
        business and this leaf does not restate, weaken or override it.

        So AC3 is the *status*, not a retry policy: anything other than
        ``recommended`` (partial evidence, empty evidence, moderate risk, severe
        weather, provider unavailable) is returned as ``insufficient`` together
        with the gaps that produced the answer, and the caller must not write a
        formal route plan from it. Nothing here fills a gap: no derived segment
        time, no estimated price, no search snippet promoted to a fact
        (:data:`route_compare.service.REQUIRED_FACTS` stays the only yardstick).
        """
        query = _research_query(destination_or_route)
        if not query["name"]:
            return RouteResearchOutcome("unavailable", reason=PlanBlockReason.RESEARCH_WITHOUT_DRAFT)

        try:
            raw = await self.compare([query])
        except ValueError:
            # ``unknown_data_mode``: no research source is configured at all.
            # Saying "not enough material" here would be a false statement
            # about the destination, so this comes back as ``unavailable``.
            return RouteResearchOutcome("unavailable", reason=PlanBlockReason.RESEARCH_UNAVAILABLE)

        result = ComparisonResult.model_validate(raw)
        gaps = tuple(dict.fromkeys(
            reason for assessment in result.candidates for reason in assessment.reasons))
        if result.status == "recommended":
            return RouteResearchOutcome("ok", result=result, gaps=gaps)
        return RouteResearchOutcome("insufficient", result=result,
                                    reason=result.status, gaps=gaps)


class RouteCompareTool(BoundedDomainTool):
    name = "route_compare"
    description = "比较最多五条路线。候选只传路线名、城市与起终点；事实由服务器收集。资料不足或危险时拒绝推荐。"
    args_schema = CompareParams
    timeout_seconds = 120.0
    max_output_chars = 40000

    async def execute(self, params):
        self._record({"phase": "call", "args": params.model_dump()})
        return await super().execute(params)

    def _success_audit_fields(self, raw):
        return {"route_comparison": raw}


class ReportExportTool(BoundedDomainTool):
    name = "report_export"
    description = ("按服务器签发的 comparison_id 导出完整徒步计划书。plan_details 只填写用户明确提供的日期、"
                   "集合、人数、交通住宿、行程、预算和应急安排；缺失项留空，禁止猜测或填入路线事实。"
                   "仅通过证据/安全门的结果可进入现有 PDF 审批与 Artifact 生命周期。")
    args_schema = ReportParams

    async def execute(self, params):
        self._record({"phase": "call", "args": params.model_dump()})
        return await super().execute(params)

    def _success_audit_fields(self, raw):
        return {"route_report": raw["route_report"]} if raw.get("route_report") else {}


def build_route_tools(session: RouteSession, *, run_id, audit_sink):
    return [RouteCompareTool(run_id=run_id, audit_sink=audit_sink, delegate=session.compare),
            ReportExportTool(run_id=run_id, audit_sink=audit_sink, delegate=session.prepare_report)]
