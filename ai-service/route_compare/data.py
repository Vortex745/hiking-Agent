"""Bounded tool collection and lossless, conservative normalization."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from .models import Fact, RouteCandidate, RouteQuery, Source


def _number(value: Any, divisor: float = 1) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        number = float(value) / divisor
        return number if 0 <= number < float("inf") else None
    except (ValueError, TypeError):
        return None


def _object(raw: Any) -> dict:
    return raw if isinstance(raw, dict) else {}


def _first(raw: Any, list_key: str) -> dict:
    values = _object(raw).get(list_key)
    return values[0] if isinstance(values, list) and values and isinstance(values[0], dict) else {}


def _add_values(candidate: RouteCandidate, tool: str, values: dict, *, url=None, observed_at=None):
    values = {k: v for k, v in values.items() if v is not None and isinstance(v, (str, float)) and v != ""}
    if not values:
        return
    sid = f"{candidate.id}-{len(candidate.sources)}"
    source = Source(id=sid, tool=tool, url=url,
                    excerpt=json.dumps(values, ensure_ascii=False)[:2000],
                    observed_at=observed_at if isinstance(observed_at, str) and observed_at else "unknown")
    candidate.sources.append(source)
    candidate.evidence_values[sid] = values
    for field, value in values.items():
        candidate.facts[field] = Fact(value=value, source_id=sid, source_field=field)


def _weather_observed_at(raw: dict) -> str | None:
    if isinstance(raw.get("observed_at"), str) and raw["observed_at"]:
        return raw["observed_at"]
    try:
        # AMap reporttime is China local time, not the tool retrieval time.
        timestamp = datetime.fromisoformat(raw["reporttime"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone(timedelta(hours=8)))
        return timestamp.isoformat()
    except (ValueError, TypeError, KeyError):
        return None


def _weather_values(raw: Any) -> dict:
    current = _first(raw, "lives") or _object(_object(raw).get("weather_data"))
    if current:
        return current
    # Official maps_weather returns a dated day/night forecast and omits
    # its observation timestamp. Keep both the forecast label and unknown
    # freshness; never relabel it as current conditions.
    forecast = _first(raw, "forecasts")
    if all(isinstance(forecast.get(key), str) for key in ("date", "dayweather", "nightweather")):
        return {"weather": f"{forecast['date']}预报：白天{forecast['dayweather']}；夜间{forecast['nightweather']}"}
    return {}


async def collect_candidates(queries: list[RouteQuery], tools: Any) -> list[RouteCandidate]:
    """Inputs are search terms. Model-provided facts/URLs cannot enter this path."""
    if not 1 <= len(queries) <= 5:
        raise ValueError("one_to_five_routes_required")
    async def collect(index: int, query: RouteQuery) -> RouteCandidate:
        candidate = RouteCandidate(id=f"route-{index + 1}", name=query.name)
        research, weather, origin, destination = await asyncio.gather(
            tools.call("local", "route_research", {"destination": query.name}),
            tools.call("amap_weather", "get_current_weather", {"city": query.city}),
            tools.call("amap_map", "geocode", {"address": query.origin, "city": query.city}),
            tools.call("amap_map", "geocode", {"address": query.destination, "city": query.city}),
        )
        start = (_first(origin, "geocodes") or _first(origin, "return")).get("location")
        end = (_first(destination, "geocodes") or _first(destination, "return")).get("location")
        if isinstance(start, str) and isinstance(end, str) and start and end:
            route = await tools.call("amap_map", "plan_route", {"origin": start, "destination": end})
        else:
            route = {"unavailable": "geocode_unavailable"}
        for label, raw in [("路线搜索", research), ("天气", weather), ("地图", route)]:
            if _object(raw).get("unavailable"):
                candidate.gaps.append(label + ": " + str(raw["unavailable"]))
        path = _first(_object(route).get("route"), "paths") or _object(_object(route).get("route_facts"))
        weather_data = _weather_values(weather)
        records = _object(research).get("route_facts")
        record = next((r for r in records if isinstance(r, dict) and r.get("name") == query.name), {}) if isinstance(records, list) else {}
        # Normalize each source independently so one malformed result cannot
        # erase good facts collected by the other tools.
        groups = [
            ("mcp__amap_map__plan_route", {
                "distance_km": _number(path.get("distance_km")) if "distance_km" in path else _number(path.get("distance"), 1000),
                "duration_hours": _number(path.get("duration_hours")) if "duration_hours" in path else _number(path.get("duration"), 3600),
                "ascent_m": _number(path.get("ascent_m")),
                "map": f"https://uri.amap.com/navigation?from={start}&to={end}&mode=walk" if path and start and end else None,
            }, None, None),
            ("mcp__amap_weather__get_current_weather", {
                "weather": weather_data.get("weather"),
            }, None, _weather_observed_at(weather_data)),
            # Search snippets and heuristic stars cannot supply hiking facts.
            ("route_research", {
                "ascent_m": _number(record.get("ascent_m")),
                "trail_status": record.get("trail_status"), "risk": record.get("risk"),
            }, record.get("source_url"), record.get("observed_at")),
        ]
        for tool, values, url, observed_at in groups:
            try:
                _add_values(candidate, tool, values, url=url, observed_at=observed_at)
            except (ValueError, TypeError):
                candidate.gaps.append(tool + ": 返回格式不支持；未补齐任何猜测值")
        return candidate
    return list(await asyncio.gather(*(collect(i, q) for i, q in enumerate(queries))))
