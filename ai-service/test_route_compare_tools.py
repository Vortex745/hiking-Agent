"""Kernel tool seam: run-scoped results, source lookup and export eligibility."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agent_v2.kernel_tools import build_chat_extra_tools
from agent_v2.profiles import resolve_profile


class CompleteLiveTools:
    def __init__(self, *, reporttime=None, observed_at=None, weather_observed_at=None):
        self.reporttime = reporttime
        self.observed_at = observed_at
        self.weather_observed_at = weather_observed_at

    async def call(self, server, tool, args):
        if tool == "route_research":
            return {"route_facts": [{"name": "测试路线", "ascent_m": 300,
                    "trail_status": "open", "risk": "low", "observed_at": self.observed_at}]}
        if tool == "geocode":
            return {"geocodes": [{"location": "120.1,30.1"}]}
        if tool == "plan_route":
            return {"route": {"paths": [{"distance": "8000", "duration": "7200"}]}}
        return {"lives": [{"weather": "晴", "reporttime": self.reporttime,
                           "observed_at": self.weather_observed_at}]}


def live_comparison_tool(tools):
    from agent_v2.route_tools import RouteSession
    from route_compare.providers import ProviderConfig, select_provider
    class Model:
        async def ainvoke(self, messages):
            return SimpleNamespace(content='{"ranking":["route-1"],"confidence":0.8}')
    settings = SimpleNamespace(route_compare_data_mode="live", route_decision_provider="llm")
    session = RouteSession(settings=settings, run_id="live-test", audit_sink=None, tools=tools,
                           provider=select_provider(ProviderConfig(), llm=Model()))
    return next(tool for tool in build_chat_extra_tools(resolve_profile("route_plan"),
                run_id="live-test", audit_sink=None, settings=settings, route_session=session)
                if tool.name == "route_compare")


def execute_live_comparison(tools):
    tool = live_comparison_tool(tools)
    output = asyncio.run(tool.execute(tool.args_schema.model_validate({"candidates": [
        {"name": "测试路线", "city": "杭州", "origin": "起点", "destination": "终点"}]})))
    assert not output.error
    return json.loads(output.output)


@pytest.mark.parametrize("reporttime", [None, "2020-01-01 10:00:00"])
def test_live_comparison_blocks_unknown_or_stale_safety_observation_time(reporttime):
    result = execute_live_comparison(CompleteLiveTools(reporttime=reporttime))
    assert result["status"] == "confirmation_required"
    assert result["recommended_id"] is None
    assert not result["report_allowed"]
    assert "安全资料过期或时间未知" in result["candidates"][0]["reasons"]


@pytest.mark.parametrize("stale", [False, True])
def test_live_comparison_uses_amap_observation_time_in_china_timezone(stale):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    weather_time = now - timedelta(days=1) if stale else now
    reporttime = weather_time.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    result = execute_live_comparison(CompleteLiveTools(reporttime=reporttime, observed_at=now.isoformat()))
    assert result["status"] == ("confirmation_required" if stale else "recommended")
    assert result["report_allowed"] is (not stale)


def test_malformed_weather_preserves_map_facts_and_explains_missing_data():
    class Tools(CompleteLiveTools):
        async def call(self, server, tool, args):
            if tool == "get_current_weather":
                return {"weather_data": []}
            return await super().call(server, tool, args)
    result = execute_live_comparison(Tools())
    candidate = result["candidates"][0]
    assert candidate["candidate"]["facts"]["distance_km"]["value"] == 8.0
    assert candidate["candidate"]["facts"]["weather"] is None
    assert candidate["evidence_status"] == "partial"
    assert result["status"] == "insufficient_evidence"
    assert not result["report_allowed"]
    assert any("weather" in reason for reason in candidate["reasons"])


def test_route_compare_consumes_official_amap_mcp_tools_and_forecast_payloads(monkeypatch):
    import sys
    from pydantic import BaseModel, ConfigDict
    from uniharness.types import ToolResult
    from agent_v2.route_tools import LiveTools
    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
    class Tool:
        args_schema = Params
        def __init__(self, server, name, payload):
            self.name = f"mcp__{server}__{name}"
            self.payload = payload
        async def execute(self, params):
            return ToolResult(output=json.dumps(self.payload))
    class Client:
        def __init__(self, server, config):
            self.tools = [
                Tool(server, "maps_geo", {"return": [{"location": "120.1,30.1"}]}),
                Tool(server, "maps_direction_walking", {"route": {"paths": [{"distance": "8000", "duration": "7200"}]}}),
                Tool(server, "maps_weather", {"city": "杭州", "forecasts": [
                    {"date": "2026-10-03", "dayweather": "晴", "nightweather": "多云"}]}),
            ]
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return None
    monkeypatch.setenv("AMAP_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "uniharness.mcp", SimpleNamespace(McpClient=Client))
    class Tools(LiveTools):
        async def call(self, server, tool, args):
            if server == "local":
                return {"route_facts": []}
            return await super().call(server, tool, args)
    result = execute_live_comparison(Tools(SimpleNamespace()))
    candidate = result["candidates"][0]["candidate"]
    assert candidate["facts"]["distance_km"]["value"] == 8.0
    assert candidate["facts"]["duration_hours"]["value"] == 2.0
    assert "2026-10-03" in candidate["facts"]["weather"]["value"]
    assert "晴" in candidate["facts"]["weather"]["value"]
    assert result["status"] == "insufficient_evidence"
    assert not result["report_allowed"]


def offline_settings(**kw):
    return SimpleNamespace(route_compare_data_mode="offline", route_decision_provider="deterministic",
                           route_compare_demo_scenario=kw.get("scenario", "complete"))


def test_route_plan_tools_offer_comparison_and_read_only_export_request():
    audit = []
    tools = build_chat_extra_tools(resolve_profile("route_plan"), run_id="run-demo", audit_sink=audit.append,
                                  settings=offline_settings())
    comparison = next(tool for tool in tools if tool.name == "route_compare")
    report = next(tool for tool in tools if tool.name == "report_export")
    async def flow():
        output = await comparison.execute(comparison.args_schema.model_validate({"candidates": [
            {"name":"演示林间环线", "city":"演示城市", "origin":"起点", "destination":"终点"}]}))
        result = json.loads(output.output)
        exported = await report.execute(report.args_schema.model_validate({"comparison_id": result["comparison_id"]}))
        return result, json.loads(exported.output)
    result, exported = asyncio.run(flow())
    assert result["status"] == "recommended"
    assert exported["status"] == "ready_for_approval"
    assert any(entry.get("route_comparison") for entry in audit)
    assert any(entry.get("route_report") for entry in audit)


def test_route_comparison_sse_replaces_unsupported_model_text_and_has_one_terminal(tmp_path):
    from agent_v2.persistence import CheckpointsStore, ControlStore
    from agent_v2.runs import RunRepository
    from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
    settings = offline_settings()
    settings.agent_v2_enabled = True
    settings.openai_api_key = "test"
    settings.openai_model = "test"
    repo = RunRepository(control=ControlStore(str(tmp_path / "control.sqlite")),
                         checkpoints=CheckpointsStore(str(tmp_path / "checkpoints.sqlite")))
    async def kernel(**kw):
        tool = next(t for t in kw["extra_tools"] if t.name == "route_compare")
        await tool.execute(tool.args_schema.model_validate({"candidates": [
            {"name":"演示林间环线", "city":"演示城市", "origin":"起点", "destination":"终点"}]}))
        return "凭空编造距离99999公里", []
    orchestrator = RunOrchestrator(settings=settings, runs_repo=repo, kernel_turn=kernel)
    async def stream():
        return [event async for event in orchestrator.stream_chat(ChatFlowRequest(
            message="比较演示路线", request_id="route-test", conversation_id="conv-route", profile_id="route_plan"))]
    events = asyncio.run(stream())
    assert [e.type for e in events if e.type in {"done", "error", "cancelled"}] == ["done"], [(e.type, e.payload) for e in events]
    assert any(e.payload.get("route_comparison", {}).get("status") == "recommended" for e in events)
    assert "99999" not in "".join(e.payload.get("content", "") for e in events)
    assert [e.sequence for e in events] == list(range(1, len(events) + 1))


def test_report_preparation_reloads_only_same_conversation_comparisons(tmp_path):
    from agent_v2.route_tools import RouteSession
    from agent_v2.persistence import CheckpointsStore, ControlStore
    from agent_v2.runs import RunRepository
    repo = RunRepository(control=ControlStore(str(tmp_path / "control.sqlite")),
                         checkpoints=CheckpointsStore(str(tmp_path / "checkpoints.sqlite")))
    run_id, _ = repo.create_run_idempotent(request_id="persisted", conversation_id="owner", profile_id="route_plan", user_message="compare")
    session = RouteSession(settings=offline_settings(), run_id=run_id, audit_sink=None)
    queries = [{"name":"演示林间环线", "city":"演示城市", "origin":"起点", "destination":"终点"}]
    result = asyncio.run(session.compare(queries))
    repo.persist_event(run_id=run_id, sequence=1, event_type="tool_result", request_id="persisted",
                       conversation_id="owner", payload={"route_comparison": result})
    good = RouteSession(settings=offline_settings(), run_id="next", audit_sink=None, runs_repo=repo, conversation_id="owner")
    bad = RouteSession(settings=offline_settings(), run_id="other", audit_sink=None, runs_repo=repo, conversation_id="stranger")
    assert asyncio.run(good.prepare_report(result["comparison_id"]))["status"] == "ready_for_approval"
    assert asyncio.run(bad.prepare_report(result["comparison_id"]))["status"] == "blocked"
