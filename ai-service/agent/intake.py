"""Request intake for AI Hiking agent.

This layer keeps the first decision deterministic: identify the hiking scenario,
extract stable slots, and surface missing constraints before the ReAct loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class AgentIntent(str, Enum):
    KNOWLEDGE_QA = "knowledge_qa"
    ROUTE_PLAN = "route_plan"
    GEAR_CHECK = "gear_check"
    RISK_ASSESSMENT = "risk_assessment"
    REPORT_EXPORT = "report_export"
    GENERAL_CHAT = "general_chat"


@dataclass
class HikingSlots:
    destination: str | None = None
    date: str | None = None
    days: int | None = None
    origin: str | None = None
    experience: str | None = None
    camping: bool | None = None
    group_size: int | None = None
    gear_level: str | None = None
    season: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "destination": self.destination,
            "date": self.date,
            "days": self.days,
            "origin": self.origin,
            "experience": self.experience,
            "camping": self.camping,
            "group_size": self.group_size,
            "gear_level": self.gear_level,
            "season": self.season,
        }


@dataclass
class CurrentLocation:
    latitude: float | None = None
    longitude: float | None = None
    accuracy: float | None = None
    province: str | None = None
    city: str | None = None
    district: str | None = None
    adcode: str | None = None
    address: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "accuracy": self.accuracy,
            "province": self.province,
            "city": self.city,
            "district": self.district,
            "adcode": self.adcode,
            "address": self.address,
            "source": self.source,
        }

    @property
    def label(self) -> str:
        return (
            self.district
            or self.city
            or self.address
            or self.adcode
            or "当前位置"
        )


@dataclass
class AgentRequestContext:
    raw_query: str
    intent: AgentIntent
    slots: HikingSlots
    current_location: CurrentLocation | None = None
    rewritten_query: str | None = None
    prefetched_tool_results: list[dict[str, object]] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    clarifying_question: str = ""
    scenario: str | None = None
    search_queries: list[str] = field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        return bool(self.missing_slots)


SCENARIO_TO_INTENT = {
    "knowledge_qa": AgentIntent.KNOWLEDGE_QA,
    "route_plan": AgentIntent.ROUTE_PLAN,
    "gear_check": AgentIntent.GEAR_CHECK,
    "risk_assessment": AgentIntent.RISK_ASSESSMENT,
    "report_export": AgentIntent.REPORT_EXPORT,
    "general_chat": AgentIntent.GENERAL_CHAT,
}

CHINESE_NUMBERS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

SLOT_LABELS = {
    "destination": "目的地",
    "date": "出行日期",
    "days": "天数",
    "origin": "出发地",
    "experience": "经验水平",
    "camping": "是否露营",
    "group_size": "团队人数",
    "gear_level": "装备水平",
}

TIME_PREFIXES = (
    "这周末",
    "本周末",
    "这个周末",
    "这周",
    "本周",
    "今天",
    "明天",
    "后天",
    "下周末",
    "下周",
)

DESTINATION_PREFIXES = (
    "我想",
    "想去",
    "计划去",
    "准备去",
    "去",
    "到",
    "前往",
    "帮我看看",
    "帮我查查",
    "帮我查",
    "帮我",
)

GENERIC_DESTINATIONS = {
    "徒步",
    "徒步吗",
    "爬山",
    "登山",
    "远足",
    "户外",
}


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().split())


def _number_value(raw: str | None) -> int | None:
    if not raw:
        return None
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    return CHINESE_NUMBERS.get(raw)


def _parse_days(text: str) -> int | None:
    match = re.search(r"(\d+|[一二两三四五六七八九十])\s*天", text)
    if match:
        return _number_value(match.group(1))
    if "当天往返" in text or "单日" in text or "一天" in text:
        return 1
    return None


def _parse_date(text: str) -> str | None:
    patterns = [
        ("这周末", "本周末"),
        ("本周末", "本周末"),
        ("这个周末", "本周末"),
        ("下周末", "下周末"),
        ("今天", "今天"),
        ("明天", "明天"),
        ("后天", "后天"),
    ]
    for needle, value in patterns:
        if needle in text:
            return value
    match = re.search(r"(\d{1,2}\s*月\s*\d{1,2}\s*[日号])", text)
    if match:
        return re.sub(r"\s+", "", match.group(1))
    return None


def _clean_destination(candidate: str) -> str:
    value = candidate.strip(" ，,。.!！?？：:；;、")
    changed = True
    while changed:
        changed = False
        for prefix in TIME_PREFIXES + DESTINATION_PREFIXES:
            if value.startswith(prefix):
                value = value[len(prefix):].strip(" ，,。.!！?？：:；;、")
                changed = True
    value = re.sub(r"(适合|安全吗|安全|怎么样|咋样|攻略|路线|行程|计划|PDF|Markdown).*$", "", value)
    return value.strip(" ，,。.!！?？：:；;、")


def _parse_destination(text: str) -> str | None:
    destination_patterns = [
        r"([A-Za-z0-9\u4e00-\u9fff·\-]{2,24}周边)",
        r"([A-Za-z0-9\u4e00-\u9fff·\-]{2,24}(?:山|峰|岭|景区|国家公园|公园|古道|峡谷|营地|环线|线路|步道))",
    ]
    for pattern in destination_patterns:
        matches = re.findall(pattern, text)
        if matches:
            destination = _clean_destination(matches[-1])
            if destination and destination not in GENERIC_DESTINATIONS:
                return destination

    match = re.search(
        r"(?:去|前往|到|计划|规划|整理)\s*([A-Za-z0-9\u4e00-\u9fff·\-]{2,18})"
        r"(?:徒步|路线|攻略|行程|两天|一天|适合|安全吗|PDF|Markdown|$)",
        text,
    )
    if match:
        destination = _clean_destination(match.group(1))
        if destination and destination not in GENERIC_DESTINATIONS:
            return destination
    return None


def _parse_origin(text: str) -> str | None:
    match = re.search(r"(?:从|出发地是|我在)([A-Za-z0-9\u4e00-\u9fff·\-]{2,12})(?:出发|过去|去|周边|$)", text)
    if match:
        return _clean_destination(match.group(1))
    return None


def _parse_experience(text: str) -> str | None:
    if any(word in text for word in ("新手", "第一次", "小白", "没经验", "没有经验")):
        return "新手"
    if any(word in text for word in ("有经验", "老手", "经常徒步", "进阶")):
        return "有经验"
    return None


def _parse_group_size(text: str) -> int | None:
    match = re.search(r"(\d+)\s*人", text)
    if match:
        return int(match.group(1))
    return None


def _parse_gear_level(text: str) -> str | None:
    if "轻装" in text:
        return "轻装"
    if "重装" in text:
        return "重装"
    return None


def _parse_season(text: str) -> str | None:
    for season in ("春季", "夏季", "秋季", "冬季", "雨季"):
        if season in text:
            return season
    return None


def _float_value(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _str_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_current_location(raw: CurrentLocation | dict | None) -> CurrentLocation | None:
    if raw is None:
        return None
    if isinstance(raw, CurrentLocation):
        return raw
    if not isinstance(raw, dict):
        return None

    raw_latitude = raw.get("latitude") if raw.get("latitude") is not None else raw.get("lat")
    raw_longitude = raw.get("longitude") if raw.get("longitude") is not None else raw.get("lng")
    location = CurrentLocation(
        latitude=_float_value(raw_latitude),
        longitude=_float_value(raw_longitude),
        accuracy=_float_value(raw.get("accuracy")),
        province=_str_value(raw.get("province")),
        city=_str_value(raw.get("city")),
        district=_str_value(raw.get("district")),
        adcode=_str_value(raw.get("adcode")),
        address=_str_value(raw.get("address") or raw.get("formatted_address")),
        source=_str_value(raw.get("source")),
    )
    if any(value is not None for value in location.to_dict().values()):
        return location
    return None


def _should_use_current_location(text: str, intent: AgentIntent) -> bool:
    if intent not in (AgentIntent.RISK_ASSESSMENT, AgentIntent.ROUTE_PLAN):
        return False
    location_words = ("附近", "周边", "当前位置", "我这里", "我这边", "本地")
    weather_words = ("天气", "适合", "能去", "可以去", "徒步吗", "去徒步")
    route_followup_words = ("推荐路线", "路线推荐", "继续推荐", "需要", "要", "好的", "可以", "安排")
    if any(word in text for word in location_words) or any(word in text for word in weather_words):
        return True
    return intent == AgentIntent.ROUTE_PLAN and any(word in text for word in route_followup_words)


def _detect_intent(text: str, scenario: str | None = None) -> AgentIntent:
    normalized_scenario = (scenario or "").strip().lower()
    if normalized_scenario in SCENARIO_TO_INTENT:
        return SCENARIO_TO_INTENT[normalized_scenario]

    upper_text = text.upper()
    if any(word in upper_text for word in ("PDF", "MARKDOWN")) or any(
        word in text for word in ("导出", "整理成", "保存为", "生成文档", "整理一份")
    ):
        return AgentIntent.REPORT_EXPORT
    if any(word in text for word in ("装备", "带什么", "清单", "穿什么", "背包")):
        return AgentIntent.GEAR_CHECK
    if any(
        word in text
        for word in (
            "适合去吗",
            "适合去徒步",
            "适合徒步",
            "天气适合",
            "能去吗",
            "可不可以去",
            "风险",
            "安全吗",
            "暴雨",
            "预警",
            "下雨后",
        )
    ):
        return AgentIntent.RISK_ASSESSMENT
    if any(word in text for word in ("路线", "攻略", "行程", "规划", "两天一夜", "一日", "单日", "周边徒步")):
        return AgentIntent.ROUTE_PLAN
    if any(word in text for word in ("是什么", "怎么处理", "怎么办", "原理", "失温", "高反", "三层穿衣")):
        return AgentIntent.KNOWLEDGE_QA
    return AgentIntent.GENERAL_CHAT


def _build_missing_slots(intent: AgentIntent, slots: HikingSlots) -> list[str]:
    required: dict[AgentIntent, list[str]] = {
        AgentIntent.ROUTE_PLAN: ["destination"],
        AgentIntent.RISK_ASSESSMENT: ["destination", "date"],
        AgentIntent.REPORT_EXPORT: [],
        AgentIntent.GEAR_CHECK: [],
        AgentIntent.KNOWLEDGE_QA: [],
        AgentIntent.GENERAL_CHAT: [],
    }

    missing: list[str] = []
    slot_values = slots.to_dict()
    for key in required.get(intent, []):
        if not slot_values.get(key):
            missing.append(key)
    return missing


def _build_clarifying_question(missing_slots: list[str]) -> str:
    if not missing_slots:
        return ""
    labels = [SLOT_LABELS.get(slot, slot) for slot in missing_slots]
    if len(labels) == 1:
        target = labels[0]
    elif len(labels) == 2:
        target = f"{labels[0]}和{labels[1]}"
    else:
        target = "、".join(labels[:-1]) + f"和{labels[-1]}"
    return f"请先补充{target}，我再继续规划。"


def _build_search_queries(intent: AgentIntent, slots: HikingSlots) -> list[str]:
    if not slots.destination:
        return []
    destination = slots.destination
    queries = [f"{destination} 徒步 路线 里程 爬升 下撤点"]
    if slots.date:
        queries.append(f"{destination} {slots.date} 天气 徒步 风险")
    if intent in (AgentIntent.GEAR_CHECK, AgentIntent.ROUTE_PLAN, AgentIntent.RISK_ASSESSMENT):
        days = f"{slots.days}天" if slots.days else ""
        queries.append(f"{destination} {days} 徒步 装备 安全")
    return queries


def understand_request(
    message: str,
    scenario: str | None = None,
    current_location: CurrentLocation | dict | None = None,
) -> AgentRequestContext:
    text = _normalize(message)
    location = _normalize_current_location(current_location)
    intent = _detect_intent(text, scenario)
    slots = HikingSlots(
        destination=_parse_destination(text),
        date=_parse_date(text),
        days=_parse_days(text),
        origin=_parse_origin(text),
        experience=_parse_experience(text),
        camping=True if "露营" in text or "扎营" in text else None,
        group_size=_parse_group_size(text),
        gear_level=_parse_gear_level(text),
        season=_parse_season(text),
    )
    if not slots.destination and location and _should_use_current_location(text, intent):
        slots.destination = location.label
    missing_slots = _build_missing_slots(intent, slots)
    return AgentRequestContext(
        raw_query=text,
        intent=intent,
        slots=slots,
        current_location=location,
        missing_slots=missing_slots,
        clarifying_question=_build_clarifying_question(missing_slots),
        scenario=scenario,
        search_queries=_build_search_queries(intent, slots),
    )
