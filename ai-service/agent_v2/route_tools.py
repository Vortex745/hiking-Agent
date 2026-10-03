"""Run-scoped kernel leaves for route comparison and report preparation."""
from __future__ import annotations

import asyncio
import json

from typing import Any
from uuid import uuid4

from pydantic import Field

from agent_v2.kernel_tools import BoundedDomainTool
from route_compare.data import collect_candidates
from route_compare.fixtures import fixture_candidates
from route_compare.models import CompareParams, ComparisonResult, Contract, RouteCandidate, RouteQuery
from route_compare.providers import ProviderConfig, select_provider
from route_compare.service import compare_routes, render_report

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

    async def prepare_report(self, comparison_id: str) -> dict:
        result = self.load_result(comparison_id)
        if result is None:
            return {"status": "blocked", "reason": "comparison_not_found"}
        try:
            render_report(result)
        except ValueError:
            return {"status": "blocked", "reason": "evidence_or_safety_gate"}
        return {"status": "ready_for_approval", "comparison_id": comparison_id,
                "route_report": result.model_dump()}


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
    description = "按服务器签发的 comparison_id 请求带来源 PDF。仅通过证据/安全门的结果可进入现有 PDF 审批与 Artifact 生命周期。"
    args_schema = ReportParams

    async def execute(self, params):
        self._record({"phase": "call", "args": params.model_dump()})
        return await super().execute(params)

    def _success_audit_fields(self, raw):
        return {"route_report": raw["route_report"]} if raw.get("route_report") else {}


def build_route_tools(session: RouteSession, *, run_id, audit_sink):
    return [RouteCompareTool(run_id=run_id, audit_sink=audit_sink, delegate=session.compare),
            ReportExportTool(run_id=run_id, audit_sink=audit_sink, delegate=session.prepare_report)]
