"""Domain tools for hiking planning, gear, and risk."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

from tools.web_search import web_search


logger = logging.getLogger("ai-service.hiking_domain")


def _now_iso() -> str:
    # UTC、微秒精度——与其余 16 处 _now_iso 副本保持同一时区基准。
    # 原实现的 astimezone() 会转成本地时间,导致 queried_at 等
    # 时间戳字段与其他模块混用两种时区,排序/比较错乱。
    return datetime.now(timezone.utc).isoformat()


ROUTE_NAME_RE = re.compile(
    r"([A-Za-z0-9\u4e00-\u9fff·\-]{2,28}"
    r"(?:风景区徒步线|徒步路线|森林公园步道|森林公园|公园步道|登山步道|步道|古道|环线|短线|长线|山|峰|岭|峡谷|公园))"
)


def _extract_route_candidates(search_text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for match in ROUTE_NAME_RE.findall(search_text or ""):
        name = match.strip(" ，,。.!！?？：:；;、")
        if len(name) < 2 or name in seen:
            continue
        seen.add(name)
        candidates.append(name)
    return candidates[:5]


def _route_rating(index: int, name: str, search_text: str) -> str:
    score = 4.4 - index * 0.2
    if any(word in search_text for word in ("成熟", "热门", "热度高", "适合新手", "交通方便")):
        score += 0.2
    if any(word in name for word in ("峡谷", "长线", "峰")):
        score -= 0.2
    score = max(3.6, min(4.8, score))
    return f"{score:.1f}/5"


def _route_reason(name: str, search_text: str) -> str:
    if "新手" in search_text or "短线" in name:
        return "搜索摘要显示强度相对友好，适合作为短途候选。"
    if any(word in search_text for word in ("成熟", "热门", "热度高", "交通方便")):
        return "搜索摘要显示路线成熟度和可达性较好。"
    if any(word in name for word in ("森林公园", "公园", "步道")):
        return "公园/步道型路线通常更适合城市近郊轻量徒步。"
    return "来自搜索摘要的候选路线，出发前仍需核验开放状态、里程和爬升。"


@tool
async def route_research(
    destination: str,
    date: str | None = None,
    days: int | None = None,
    focus: str | None = None,
) -> dict[str, Any]:
    """Search hiking route candidates and return a structured facts envelope."""
    destination = (destination or "").strip()
    if not destination:
        return {
            "ok": False,
            "facts": [],
            "message": "缺少目的地，无法收集路线资料。",
        }

    day_text = f"{days}天" if days else ""
    queries = [
        f"{destination} 徒步 路线 里程 爬升 下撤点",
        f"{destination} {day_text} 徒步 攻略 安全",
    ]
    if date:
        queries.append(f"{destination} {date} 天气 路况 徒步")
    if focus:
        queries.append(f"{destination} 徒步 路线 {focus}")

    search_results: list[dict[str, Any]] = []
    for query in queries[:4]:
        try:
            result = await web_search.ainvoke({"query": query})
        except Exception as e:
            result = f"搜索出错: {str(e)}"
        search_results.append({
            "query": query,
            "result": str(result),
            "queried_at": _now_iso(),
        })

    combined_search_text = "\n".join(item["result"] for item in search_results)
    route_names = _extract_route_candidates(combined_search_text)
    recommended_routes = [
        {
            "name": name,
            "rating": _route_rating(index, name, combined_search_text),
            "reason": _route_reason(name, combined_search_text),
            "source": "web_search",
        }
        for index, name in enumerate(route_names)
    ]

    return {
        "ok": True,
        "destination": destination,
        "date": date or "未指定",
        "days": days,
        "queries": queries,
        "search_results": search_results,
        "recommended_routes": recommended_routes,
        "facts": [
            {
                "field": "资料状态",
                "value": "已调用搜索引擎收集路线摘要；候选路线和星级为基于搜索摘要的初筛，出发前仍需核验官方开放状态、里程、爬升和天气。",
                "source": "route_research",
                "queried_at": _now_iso(),
            }
        ],
        "message": (
            "已完成路线搜索初筛。"
            if recommended_routes
            else "搜索结果没有返回可核验路线名，请换更具体的城市/区域或手动提供候选路线。"
        ),
    }


@tool
async def gear_checklist(
    days: int | None = None,
    season: str | None = None,
    experience: str | None = None,
    camping: bool | None = None,
    gear_level: str | None = None,
) -> dict[str, Any]:
    """Generate a conservative hiking gear checklist."""
    trip_days = days or 1
    is_camping = bool(camping) or trip_days >= 2 or gear_level == "重装"

    items = {
        "基础": ["登山鞋", "速干衣裤", "雨衣/冲锋衣", "头灯", "充电宝", "垃圾袋"],
        "安全": ["急救包", "保温毯", "哨子", "离线地图", "备用食物", "足量饮水"],
        "分层穿衣": ["排汗层", "保暖层", "防风防雨层"],
    }
    if is_camping:
        items["露营"] = ["帐篷", "睡袋", "防潮垫", "炉具/餐具", "保暖帽"]
    if season == "冬季":
        items["冬季"] = ["抓绒/羽绒中层", "手套", "保暖帽", "冰爪按路线核验"]
    if experience == "新手":
        items["新手额外"] = ["结伴出行", "明确下撤点", "不要夜行", "行前共享路线"]

    return {
        "ok": True,
        "days": trip_days,
        "season": season or "未指定",
        "experience": experience or "未指定",
        "camping": is_camping,
        "checklist": items,
        "message": "装备清单按保守原则生成；具体重量和型号需结合天气、海拔和路线强度调整。",
    }


@tool
async def focus_map(name: str) -> dict[str, Any]:
    """Record the single scenic spot that the current turn focuses on."""
    cleaned = (name or "").strip()
    if not cleaned:
        return {"ok": False, "message": "景点名为空，无法定位。"}
    return {"ok": True, "name": cleaned[:80]}


@tool
async def risk_assessment(
    destination: str | None = None,
    weather: str | None = None,
    route: str | None = None,
    experience: str | None = None,
    days: int | None = None,
) -> dict[str, Any]:
    """Assess hiking risk with conservative rule-based guardrails."""
    text = " ".join(x for x in [destination, weather, route, experience] if x)
    reasons: list[str] = []
    level = "low"
    recommendation = "可以继续规划，但要补齐天气、路线和撤退点信息。"

    severe_weather = any(word in text for word in ("暴雨", "雷暴", "橙色预警", "红色预警", "大风预警"))
    if severe_weather:
        level = "high"
        reasons.append("存在暴雨/雷暴/橙色及以上预警等硬风险条件。")
        recommendation = "建议取消或改期，不建议进入山区、峡谷或长距离路线。"

    if experience == "新手" and any(word in text for word in ("高海拔", "夜行", "峡谷", "重装")):
        level = "high" if level == "high" else "medium"
        reasons.append("新手叠加高海拔/夜行/峡谷/重装等复杂因素。")

    if (days or 1) >= 2 and experience == "新手":
        if level == "low":
            level = "medium"
        reasons.append("新手两天及以上行程需要更保守的里程、补给和住宿安排。")

    if not reasons:
        reasons.append("缺少实时天气、海拔、里程和路况证据，风险等级只能保守估计。")

    return {
        "ok": True,
        "destination": destination or "未指定",
        "risk_level": level,
        "reasons": reasons,
        "cancel_conditions": ["暴雨橙色及以上预警", "雷暴大风", "山洪/滑坡风险", "无法确认下撤点"],
        "alternatives": ["改期", "缩短路线", "选择低海拔成熟路线", "改为城市近郊短线"],
        "recommendation": recommendation,
    }
