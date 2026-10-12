"""Structured document model for the sourced route-comparison PDF.

``render_report`` text stays the conversation-facing plain text; this module
derives a layout-oriented document dict from the same validated
:class:`ComparisonResult` so the sandbox PDF renderer can produce a designed
document instead of dumping the text line by line. Rendering itself lives in
``agent_v2.computer`` — this module only knows the domain structure and the
Chinese labels for the wire enums.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from .models import ComparisonResult

REPORT_KIND = "route_comparison_report"

_STATUS_LABELS = {
    "recommended": "已核验推荐",
    "confirmation_required": "需要确认",
    "no_go": "不建议出行",
    "blocked": "已拦截",
    "insufficient_evidence": "资料不足",
    "provider_unavailable": "排序服务不可用",
}
_RISK_LABELS = {
    "clear": "资料齐全 · 低风险",
    "caution": "需谨慎",
    "confirmation_required": "需要确认",
    "no_go": "不建议出行",
}
_EVIDENCE_LABELS = {
    "complete": "资料齐全",
    "partial": "资料不全",
    "empty": "无资料",
}
_CONFIDENCE_KIND_LABELS = {
    "heuristic": "启发式估计",
    "calibrated": "校准值",
    "unavailable": "不可用",
}
_FIELD_LABELS = {
    "distance_km": "距离",
    "ascent_m": "累计爬升",
    "duration_hours": "预计耗时",
    "weather": "天气",
    "trail_status": "开放状态",
    "risk": "风险提示",
}
_FIELD_UNITS = {"distance_km": "km", "ascent_m": "m", "duration_hours": "小时"}
_TABLE_FIELDS = ["distance_km", "ascent_m", "duration_hours", "weather", "trail_status"]
_VALUE_LABELS = {
    "trail_status": {"open": "开放", "closed": "关闭", "unknown": "未知"},
    "risk": {"low": "低", "moderate": "中等", "high": "高"},
}

_UNKNOWN = "待核验"


def format_datetime(raw: str) -> str:
    """Use China local time for this Chinese document, retaining unknown timestamps."""
    try:
        parsed = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return raw
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone(timedelta(hours=8)))
        return parsed.strftime("%Y-%m-%d %H:%M") + " (北京时间)"
    return parsed.strftime("%Y-%m-%d %H:%M") + " (时区未知)"


def _format_fact(field: str, value: Any) -> str:
    if value is None:
        return _UNKNOWN
    labels = _VALUE_LABELS.get(field)
    if labels is not None:
        return labels.get(str(value), str(value))
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    unit = _FIELD_UNITS.get(field)
    return f"{text} {unit}" if unit and text != _UNKNOWN else text


def _plan_sections(result: ComparisonResult, route_id: str | None) -> dict[str, Any]:
    """Create deterministic execution sections from verified facts and safe defaults."""
    assessment = next((a for a in result.candidates if a.candidate.id == route_id), None)
    facts = assessment.candidate.facts if assessment else {}
    def value(field: str) -> str:
        return _format_fact(field, facts[field].value if facts.get(field) else None)

    details = result.plan_details
    def user_value(text: Any) -> str:
        return str(text).strip() if text is not None and str(text).strip() else "待确认"
    risk_label = _RISK_LABELS.get(assessment.risk_status, _UNKNOWN) if assessment else _UNKNOWN
    return {
        "plan_metadata": [
            {"label": "活动日期", "value": user_value(details.activity_date)},
            {"label": "集合时间", "value": user_value(details.meeting_time)},
            {"label": "集合地点", "value": user_value(details.meeting_point)},
            {"label": "队伍人数", "value": user_value(details.participants)},
            {"label": "领队", "value": user_value(details.leader)},
            {"label": "安排来源", "value": "用户提供 / 缺失待确认"},
        ],
        "route_overview": [
            {"label": "计划路线", "value": assessment.candidate.name if assessment else _UNKNOWN},
            {"label": "距离", "value": value("distance_km")},
            {"label": "累计爬升", "value": value("ascent_m")},
            {"label": "预计耗时", "value": value("duration_hours")},
            {"label": "天气", "value": value("weather")},
            {"label": "开放状态", "value": value("trail_status")},
            {"label": "风险状态", "value": risk_label},
            {"label": "资料状态", "value": _EVIDENCE_LABELS.get(assessment.evidence_status, _UNKNOWN) if assessment else _UNKNOWN},
        ],
        "itinerary": [
            {"stage": "出发准备", "window": "待确认",
             "focus": "签到、热身、确认天气与开放状态", "checkpoint": "确认人数与撤退方案"},
            {"stage": "主体徒步", "window": "待确认",
             "focus": "结伴行进，保持可沟通距离", "checkpoint": "按体感与环境及时补水和休息"},
            {"stage": "补给与评估", "window": "待确认",
             "focus": "检查体感、天气和剩余补给", "checkpoint": "安全条件恶化时停止前进"},
            {"stage": "返程与收尾", "window": "待确认",
             "focus": "留出下撤余量，离队前清点人员与装备", "checkpoint": "到达终点后报平安"},
        ],
        "user_itinerary": [{"stage": f"安排 {i}", "window": user_value(s.time),
                            "focus": s.activity, "checkpoint": "用户安排，执行前确认"}
                           for i, s in enumerate(details.itinerary, 1)],
        "logistics": [
            {"label": "往返交通", "value": user_value(details.transport)},
            {"label": "住宿 / 营地", "value": user_value(details.accommodation)},
            {"label": "路线地图", "value": value("map")},
            {"label": "紧急联系人", "value": user_value(details.emergency_contact)},
            {"label": "撤退安排", "value": user_value(details.retreat_plan)},
        ],
        "roles": [
            {"role": "领队", "person": user_value(details.leader), "duty": "确认路线与风险，决定折返或取消"},
            {"role": "导航 / 收队", "person": "待确认", "duty": "核对地图、清点人数，避免落单"},
            {"role": "急救 / 联络", "person": "待确认", "duty": "检查急救用品，维护对外联络"},
        ],
        "budget": [{"item": b.item, "amount": f"{b.amount_yuan:g} 元", "basis": "用户提供"} for b in details.budget]
                  or [{"item": category, "amount": "待确认", "basis": "待询价"}
                      for category in ("交通", "住宿 / 门票", "餐食 / 补给", "保险 / 其他")],
        "budget_total": f"{sum(b.amount_yuan for b in details.budget):g} 元" if details.budget else "待确认",
        "equipment": [
            {"category": "安全", "item": "手机、充电宝、头灯、急救包", "priority": "必带", "status": _UNKNOWN},
            {"category": "补给", "item": "饮水与能量补给", "priority": "必带", "status": _UNKNOWN},
            {"category": "防护", "item": "防滑鞋、雨具、防晒与保暖层", "priority": "必带", "status": _UNKNOWN},
            {"category": "导航", "item": "离线地图与纸质 / 截图备份", "priority": "建议", "status": _UNKNOWN},
            {"category": "联络", "item": "紧急联系人、集合点和撤退交通", "priority": "必带", "status": _UNKNOWN},
        ],
        "risk_controls": [
            {"trigger": "雷暴、暴雨、大风或官方关闭", "action": "取消或立即撤退，等待官方解除风险", "severity": "停止出行"},
            {"trigger": "体力、伤情或天气明显恶化", "action": "停止前进，结伴撤退并联系外部支援", "severity": "立即处置"},
            {"trigger": "迷路、失联或超过计划时间", "action": "在安全位置停止行进，定位并联系领队；有危险时求助当地救援", "severity": "立即处置"},
            {"trigger": "当前资料缺失或超过有效时间", "action": "出发前重新查询并完成人工确认", "severity": "出发前拦截"},
        ],
        "checklist": [
            {"item": "天气、官方开放状态和风险资料在出发前重新确认", "owner": "领队", "status": _UNKNOWN},
            {"item": "路线地图已下载，集合点和撤退点已共享", "owner": "领队", "status": _UNKNOWN},
            {"item": "每位成员完成装备、饮水和身体状态自检", "owner": "全员", "status": _UNKNOWN},
            {"item": "紧急联系人知道路线、预计返程时间和报平安方式", "owner": "领队", "status": _UNKNOWN},
            {"item": "垃圾全部带离，遵守当地开放与生态保护规定", "owner": "全员", "status": _UNKNOWN},
        ],
    }


def build_report_document(result: ComparisonResult) -> dict[str, Any]:
    """Derive the structured PDF document from a gate-passed comparison result."""
    names = {a.candidate.id: a.candidate.name for a in result.candidates}
    extra_fields: list[str] = []
    for assessment in result.candidates:
        for field in assessment.candidate.facts:
            if field not in _TABLE_FIELDS and field not in extra_fields and field != "map":
                extra_fields.append(field)

    rows: list[dict[str, Any]] = []
    ordered = ([result.recommended_id] if result.recommended_id else []) + list(result.alternatives)
    ordered += [a.candidate.id for a in result.candidates if a.candidate.id not in ordered]
    for cid in ordered:
        assessment = next(a for a in result.candidates if a.candidate.id == cid)
        facts = assessment.candidate.facts
        cells = [_format_fact(f, facts[f].value if facts.get(f) else None) for f in (*_TABLE_FIELDS, *extra_fields)]
        rows.append({
            "name": assessment.candidate.name,
            "cells": cells,
            "highlight": cid == result.recommended_id,
            "badge": "推荐" if cid == result.recommended_id else "备选",
            "risk": _RISK_LABELS.get(assessment.risk_status, assessment.risk_status),
            "evidence": _EVIDENCE_LABELS.get(assessment.evidence_status, assessment.evidence_status),
        })

    sources = []
    for assessment in result.candidates:
        entries = [{
            "id": s.id,
            "tool": s.tool,
            "url": s.url or "",
            "excerpt": s.excerpt,
            "observed_at": format_datetime(s.observed_at),
        } for s in assessment.candidate.sources]
        sources.append({"candidate": assessment.candidate.name, "entries": entries})

    confidence = result.decision.confidence
    value_text = "—" if confidence.value is None else f"{confidence.value:.2f}"

    selected_id = result.recommended_id or (result.candidates[0].candidate.id if result.candidates else None)
    sections = _plan_sections(result, selected_id)
    return {
        "kind": REPORT_KIND,
        "schema_version": "2.0",
        "document_type": "hiking_plan",
        "title": "AI Hiking 徒步计划书",
        "subtitle": "路线资料已核验 · 活动安排待确认",
        "content_policy": {
            "facts": "仅使用已核验来源；缺失信息显示为待核验",
            "advice": "通用执行建议，出发前需由领队确认",
        },
        "generated_at": format_datetime(result.generated_at),
        "comparison_id": result.comparison_id,
        "status_label": _STATUS_LABELS.get(result.status, result.status),
        "offline_disclaimer": result.data_mode == "offline",
        "recommended": names.get(result.recommended_id, ""),
        "alternatives": [names[cid] for cid in result.alternatives],
        "columns": [_FIELD_LABELS[f] for f in _TABLE_FIELDS] + [f" {_FIELD_LABELS.get(f, f)} " for f in extra_fields],
        "rows": rows,
        # Candidate detail lines from compare_routes repeat the facts already
        # shown in the comparison table; keep only real explanatory notes.
        "notes": [n for n in result.explanation if "=" not in n],
        "required_action": result.required_action,
        "decision": {
            "provider": result.decision.provider,
            "model": result.decision.model or "未调用",
            "confidence_kind": _CONFIDENCE_KIND_LABELS.get(confidence.kind, confidence.kind),
            "confidence_value": value_text,
            "reason": result.decision.reason or "",
        },
        "sources": sources,
        **sections,
    }
