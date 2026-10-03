"""Evidence and safety behavior through the public comparison service."""
import asyncio
import pytest

from route_compare.fixtures import fixture_candidates
from route_compare.providers import ProviderConfig, select_provider
from route_compare.service import compare_routes, comparison_text, render_report


def test_complete_offline_evidence_produces_sourced_recommendation_and_alternative():
    result = asyncio.run(compare_routes(fixture_candidates("complete"),
        select_provider(ProviderConfig(provider="deterministic", data_mode="offline"))))
    assert result.status == "recommended"
    assert result.recommended_id == "demo-a"
    assert result.alternatives == ["demo-b"]
    assert result.report_allowed
    assert result.candidates[0].candidate.facts["distance_km"].value == 8.0
    assert result.candidates[0].candidate.sources


@pytest.mark.parametrize("scenario,status", [("partial", "insufficient_evidence"),
                                             ("empty", "insufficient_evidence"),
                                             ("severe_weather", "no_go"),
                                             ("caution", "confirmation_required")])
def test_missing_evidence_or_danger_cannot_be_ranked_or_exported(scenario, status):
    result = asyncio.run(compare_routes(fixture_candidates(scenario),
        select_provider(ProviderConfig(provider="deterministic", data_mode="offline"))))
    assert result.status == status
    assert result.recommended_id is None
    assert not result.report_allowed
    with pytest.raises(ValueError, match="report_blocked"):
        render_report(result)


def test_numeric_claim_not_matching_source_is_removed_from_public_output():
    candidates = fixture_candidates()
    candidates[0].facts["distance_km"].value = 99999.0
    result = asyncio.run(compare_routes(candidates,
        select_provider(ProviderConfig(provider="deterministic", data_mode="offline"))))
    assert result.candidates[0].candidate.facts["distance_km"] is None
    assert "99999" not in result.model_dump_json()
    assert "99999" not in comparison_text(result)
    assert result.recommended_id == "demo-b"


def test_provider_failure_is_not_silently_replaced_by_offline_ordering():
    result = asyncio.run(compare_routes(fixture_candidates(), select_provider(ProviderConfig())))
    assert result.status == "provider_unavailable"
    assert result.recommended_id is None


@pytest.mark.parametrize("weather,status", [("unknown", "confirmation_required"),
                                           (123.0, "confirmation_required"),
                                           ("雷阵雨", "no_go")])
def test_unrecognized_or_dangerous_weather_cannot_be_a_clear_recommendation(weather, status):
    candidates = fixture_candidates()
    for candidate in candidates:
        candidate.data_mode = "live"
        candidate.facts["weather"].value = weather
        candidate.evidence_values[candidate.facts["weather"].source_id]["weather"] = weather
    result = asyncio.run(compare_routes(candidates,
        select_provider(ProviderConfig(provider="deterministic", data_mode="offline"))))
    assert result.status == status
    assert result.recommended_id is None
    assert not result.report_allowed


def test_map_weather_live_adapter_retains_missing_ascent_without_guessing():
    from route_compare.data import collect_candidates
    from route_compare.models import RouteQuery
    class Tools:
        async def call(self, server, tool, args):
            if tool == "route_research":
                return {"search_results": []}
            if tool == "geocode":
                return {"geocodes": [{"location": "120.1,30.1"}]}
            if tool == "plan_route":
                return {"route": {"paths": [{"distance": "8000", "duration": "7200"}]}}
            return {"lives": [{"weather": "晴", "reporttime": "2026-10-03 10:00:00"}]}
    candidates = asyncio.run(collect_candidates([RouteQuery(name="测试路线", city="杭州",
        origin="起点", destination="终点")], Tools()))
    assert candidates[0].facts["distance_km"].value == 8.0
    assert candidates[0].facts.get("ascent_m") is None
    assert candidates[0].sources
    result = asyncio.run(compare_routes(candidates, select_provider(ProviderConfig())))
    assert result.recommended_id is None
