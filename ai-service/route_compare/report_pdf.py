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
from typing import Any, Mapping

from .models import UNCONFIRMED, UNVERIFIED, ComparisonResult

REPORT_KIND = "route_comparison_report"

#: The arrangement labels arrive as an ARGUMENT (:func:`build_report_document`'s
#: ``arrangement_labels``) rather than being read from a module: they are the plan
#: vocabulary the draft collects and the 缺失字段清单 enumerates, and this layer sits
#: BELOW ``agent_v2``, so reaching up for them was the inversion ticket 06's drift
#: guard accidentally created. The caller passes the same record it renders the
#: checklist from, so a field still cannot exist in one place and be missing from the
#: PDF — the rule is now enforced by the data flow instead of by an import.

#: The plan status the cover states, from the two vocabularies that already own
#: these answers (ticket 04's ``plan_type``, ticket 06's verdict). An absent one
#: is :data:`route_compare.models.UNVERIFIED` — a cover never infers a status the
#: flow did not record.
_PLAN_TYPE_LABELS = {"route_plan": "正式路线计划", "generic_draft": "通用草案 · 未经核验路线资料"}
_VERDICT_LABELS = {
    "complete": "安排已齐全",
    "confirmation_required": f"有{UNCONFIRMED}安排",
    "supplement_required": "核心安排待补充",
}
_DATA_MODE_LABELS = {"offline": "离线合成演示数据 · 不可用于真实出行", "live": "服务器实时核验来源"}

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
        return UNVERIFIED
    labels = _VALUE_LABELS.get(field)
    if labels is not None:
        return labels.get(str(value), str(value))
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    unit = _FIELD_UNITS.get(field)
    return f"{text} {unit}" if unit and text != UNVERIFIED else text


def _user_value(text: Any) -> str:
    """One arrangement as the user stated it, or the 待确认 marker and nothing else."""
    return str(text).strip() if text is not None and str(text).strip() else UNCONFIRMED


def arrangement_values(result: ComparisonResult,
                       arrangements: Mapping[str, str] | None,
                       labels: Mapping[str, str]) -> dict[str, str]:
    """The user's own record, nothing else.

    Every user-arrangement field is drawn ONLY from ``arrangements`` — the
    stored record :meth:`agent_v2.runs.RunRepository.get_plan_arrangements`
    merged over the draft row by the export path. A field the user left blank
    stays blank and renders :data:`route_compare.models.UNCONFIRMED` (seam contract
    AC2, ticket 05 AC4).

    ``labels`` is the caller's arrangement enumeration
    (:data:`agent_v2.plan_workflow.ARRANGEMENT_FIELDS`): it decides WHICH keys exist,
    in display order. Keys outside it are dropped rather than drawn.

    This deliberately does NOT seed from ``result.plan_details``: that object is
    what the kernel composed for ``report_export``, so a model string would
    otherwise render as if the user had supplied it — the invention the no-invention
    invariant forbids. ``budget`` stays free text here and is never parsed into an
    amount, so no price and no total is ever produced (AC2). ``result`` is kept in
    the signature because callers pass the gate-checked object through, not because
    anything is read from it here.
    """
    values = {key: "" for key in labels}
    supplied = arrangements or {}
    for key in values:
        value = supplied.get(key)
        if isinstance(value, str) and value.strip():
            values[key] = value.strip()
    return values


def _plan_sections(result: ComparisonResult, route_id: str | None,
                   values: dict[str, str],
                   labels: Mapping[str, str]) -> dict[str, Any]:
    """Create deterministic execution sections from verified facts and safe defaults."""
    assessment = next((a for a in result.candidates if a.candidate.id == route_id), None)
    facts = assessment.candidate.facts if assessment else {}
    def value(field: str) -> str:
        return _format_fact(field, facts[field].value if facts.get(field) else None)

    details = result.plan_details
    def user_value(text: Any) -> str:
        return _user_value(text)
    risk_label = _RISK_LABELS.get(assessment.risk_status, UNVERIFIED) if assessment else UNVERIFIED
    template = [
        {"stage": "出发准备", "window": UNCONFIRMED,
         "focus": "签到、热身、确认天气与开放状态", "checkpoint": "确认人数与撤退方案"},
        {"stage": "主体徒步", "window": UNCONFIRMED,
         "focus": "结伴行进，保持可沟通距离", "checkpoint": "按体感与环境及时补水和休息"},
        {"stage": "补给与评估", "window": UNCONFIRMED,
         "focus": "检查体感、天气和剩余补给", "checkpoint": "安全条件恶化时停止前进"},
        {"stage": "返程与收尾", "window": UNCONFIRMED,
         "focus": "留出下撤余量，离队前清点人员与装备", "checkpoint": "到达终点后报平安"},
    ]
    timeline = ([{"time": _user_value(stage.time), "stage": _user_value(stage.activity),
                 "focus": "用户提供的安排，执行前需确认", "checkpoint": "按约定时间由领队确认"}
                for stage in details.itinerary]
                or [{"time": _user_value(row["window"]), "stage": row["stage"],
                     "focus": row["focus"], "checkpoint": row["checkpoint"]} for row in template])
    #: Risk handling is generic advice, not a verified fact, and says so.
    risk_rows = [
        {"trigger": "雷暴、暴雨、大风或官方关闭", "action": "取消或立即撤退，等待官方解除风险", "severity": "停止出行"},
        {"trigger": "体力、伤情或天气明显恶化", "action": "停止前进，结伴撤退并联系外部支援", "severity": "立即处置"},
        {"trigger": "迷路、失联或超过计划时间", "action": "在安全位置停止行进，定位并联系领队；有危险时求助当地救援", "severity": "立即处置"},
        {"trigger": "当前资料缺失或超过有效时间", "action": "出发前重新查询并完成人工确认", "severity": "出发前拦截"},
    ]
    return {
        "plan_metadata": [{"label": label, "value": user_value(values.get(key, ""))}
                          for key, label in labels.items()],
        "route_overview": [
            {"label": "计划路线", "value": assessment.candidate.name if assessment else UNVERIFIED},
            {"label": "距离", "value": value("distance_km")},
            {"label": "累计爬升", "value": value("ascent_m")},
            {"label": "预计耗时", "value": value("duration_hours")},
            {"label": "天气", "value": value("weather")},
            {"label": "开放状态", "value": value("trail_status")},
            {"label": "风险状态", "value": risk_label},
            {"label": "资料状态", "value": _EVIDENCE_LABELS.get(assessment.evidence_status, UNVERIFIED) if assessment else UNVERIFIED},
        ],
        "itinerary": template,
        "timeline": timeline,
        "user_itinerary": [{"stage": f"安排 {i}", "window": user_value(s.time),
                            "focus": s.activity, "checkpoint": "用户安排，执行前确认"}
                           for i, s in enumerate(details.itinerary, 1)],
        "logistics": [
            {"label": labels["transport"], "value": user_value(values.get("transport"))},
            {"label": labels["accommodation"], "value": user_value(values.get("accommodation"))},
            {"label": "路线地图", "value": value("map")},
            {"label": labels["emergency_contact"],
             "value": user_value(values.get("emergency_contact"))},
            {"label": labels["retreat_plan"], "value": user_value(values.get("retreat_plan"))},
        ],
        "roles": [
            {"role": "领队", "person": user_value(values.get("leader")),
             "duty": "确认路线与风险，决定折返或取消"},
            {"role": "导航 / 收队", "person": UNCONFIRMED, "duty": "核对地图、清点人数，避免落单"},
            {"role": "急救 / 联络", "person": UNCONFIRMED, "duty": "检查急救用品，维护对外联络"},
        ],
        "budget": [{"item": b.item, "amount": f"{b.amount_yuan:g} 元", "basis": "用户提供"} for b in details.budget]
                  or [{"item": category, "amount": UNCONFIRMED, "basis": "待询价"}
                      for category in ("交通", "住宿 / 门票", "餐食 / 补给", "保险 / 其他")],
        # The user's own free-text budget note travels as text: it is never parsed
        # into an amount, so no price and no total is created here (AC2).
        "budget_note": _user_value(values.get("budget")),
        # No server-side total is ever computed. The itemised ``budget`` above is
        # the model's own plan_details, not a user-confirmed figure, and the spec
        # rule (ticket 09 / plan_workflow:192) is that prices are never estimated
        # and a total is never summed out of them. The 合计 therefore stays
        # :data:`UNCONFIRMED` — a number the server cannot stand behind is not
        # printed as one it can.
        "budget_total": UNCONFIRMED,
        "equipment": [
            {"category": "安全", "item": "手机、充电宝、头灯、急救包", "priority": "必带", "status": UNVERIFIED},
            {"category": "补给", "item": "饮水与能量补给", "priority": "必带", "status": UNVERIFIED},
            {"category": "防护", "item": "防滑鞋、雨具、防晒与保暖层", "priority": "必带", "status": UNVERIFIED},
            {"category": "导航", "item": "离线地图与纸质 / 截图备份", "priority": "建议", "status": UNVERIFIED},
            {"category": "联络", "item": "紧急联系人、集合点和撤退交通", "priority": "必带", "status": UNVERIFIED},
        ],
        "risk_controls": risk_rows,
        "risk": {
            "rows": risk_rows,
            "status_label": risk_label,
            "decision": _user_value(result.required_action),
            "note": "以上为通用风险处置建议：触发条件一旦成立即按处置动作执行，出发前由领队确认。",
            "contacts": [
                {"label": labels["emergency_contact"],
                 "value": _user_value(values.get("emergency_contact"))},
                {"label": labels["retreat_plan"],
                 "value": _user_value(values.get("retreat_plan"))},
            ],
        },
        "checklist": [
            {"item": "天气、官方开放状态和风险资料在出发前重新确认", "owner": "领队", "status": UNVERIFIED},
            {"item": "路线地图已下载，集合点和撤退点已共享", "owner": "领队", "status": UNVERIFIED},
            {"item": "每位成员完成装备、饮水和身体状态自检", "owner": "全员", "status": UNVERIFIED},
            {"item": "紧急联系人知道路线、预计返程时间和报平安方式", "owner": "领队", "status": UNVERIFIED},
            {"item": "垃圾全部带离，遵守当地开放与生态保护规定", "owner": "全员", "status": UNVERIFIED},
        ],
    }


def _stated(mapping: Mapping[str, str], key: Any) -> str:
    """A recorded status word as its label, or 待核验 — never a guessed one."""
    if key is None or str(key).strip() == "":
        return UNVERIFIED
    return mapping.get(str(key).strip(), UNVERIFIED)


def _safety_status(safety: Mapping[str, Any] | None) -> str:
    """Ticket 07's refusal, copied from the record it produced.

    Nothing is re-derived here: the message is :data:`SAFETY_MESSAGE` as the gate
    wrote it, and an absent record stays 待核验 rather than becoming a pass.
    """
    if not isinstance(safety, Mapping):
        return UNVERIFIED
    message = str(safety.get("message") or "").strip()
    if safety.get("hard_stop") or not safety.get("formal_route_plan_allowed", True):
        return message or "安全门已拦截本次正式路线计划"
    return message or "本次记录未见安全门拦截 · 出发前仍须重新核验"


def _cover(result: ComparisonResult, status_label: str, plan_type: str | None,
           completeness: str | None, safety: Mapping[str, Any] | None) -> dict[str, Any]:
    """The cover page: what kind of plan this is, and how far it got."""
    return {
        "title": "AI Hiking 徒步计划书",
        "kicker": "徒步计划书 · Hiking Plan Book",
        "heading": "计划状态",
        "status_lines": [
            {"label": "草案类型", "value": _stated(_PLAN_TYPE_LABELS, plan_type)},
            {"label": "审阅结论", "value": _stated(_VERDICT_LABELS, completeness)},
            {"label": "安全门", "value": _safety_status(safety)},
            {"label": "路线核验", "value": status_label},
            {"label": "资料模式", "value": _stated(_DATA_MODE_LABELS, result.data_mode)},
            {"label": "生成时间", "value": format_datetime(result.generated_at)},
            {"label": "计划编号", "value": result.comparison_id},
        ],
        "content_policy": {
            "facts": f"仅使用已核验来源；缺失信息显示为{UNVERIFIED}",
            "advice": "通用执行建议，出发前需由领队确认",
        },
    }


def _summary(result: ComparisonResult, names: dict[str, str], values: dict[str, str],
             labels: Mapping[str, str]) -> dict[str, Any]:
    """The 路线 / 日期 / 人数 summary — the book's answer to 「which outing is this」."""
    route = names.get(result.recommended_id or "", "") or _user_value(values.get("destination_or_route"))
    return {
        "headline": route if route != UNVERIFIED else UNVERIFIED,
        "lines": [
            {"label": "路线", "value": _user_value(route) if route else UNVERIFIED},
            {"label": labels["activity_date"], "value": _user_value(values.get("activity_date"))},
            {"label": labels["participants"], "value": _user_value(values.get("participants"))},
            {"label": labels["meeting_time"], "value": _user_value(values.get("meeting_time"))},
            {"label": labels["meeting_point"], "value": _user_value(values.get("meeting_point"))},
            {"label": labels["leader"], "value": _user_value(values.get("leader"))},
        ],
    }


def build_report_document(result: ComparisonResult, *,
                          arrangement_labels: Mapping[str, str],
                          plan_type: str | None = None,
                          completeness: str | None = None,
                          safety: Mapping[str, Any] | None = None,
                          arrangements: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Derive the structured PDF document from a gate-passed comparison result.

    The five keyword arguments are the plan side of the book, each one a record an
    earlier ticket already produced and this function only reads:
    ``arrangement_labels`` — the caller's arrangement enumeration
    (:data:`agent_v2.plan_workflow.ARRANGEMENT_FIELDS`), REQUIRED because this module
    sits below ``agent_v2`` and must not read up for its own vocabulary;
    ``plan_type`` (ticket 04), ``completeness`` — the rule verdict (ticket 06),
    ``safety`` — the gate's own refusal payload (ticket 07), and ``arrangements`` —
    the stored arrangement record merged over the draft row. None of them is a claim
    about report eligibility: ``render_report`` stays the only authority on that, so a
    ``complete`` verdict renders as 安排已齐全 and nothing more.
    """
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
    values = arrangement_values(result, arrangements, arrangement_labels)
    status_label = _STATUS_LABELS.get(result.status, result.status)
    sections = _plan_sections(result, selected_id, values, arrangement_labels)
    is_route_plan = str(plan_type or "").strip() == "route_plan"
    return {
        "kind": REPORT_KIND,
        "schema_version": "2.0",
        "document_type": "hiking_plan",
        "title": "AI Hiking 徒步计划书",
        # The subtitle is a status claim, so it follows the derived type: a draft
        # without a verified route basis cannot describe itself as 已核验.
        "subtitle": (f"路线资料已核验 · 活动安排{UNCONFIRMED}" if is_route_plan
                     else f"通用草案 · 路线资料{UNVERIFIED}"),
        "plan_type": _stated(_PLAN_TYPE_LABELS, plan_type),
        "completeness_verdict": _stated(_VERDICT_LABELS, completeness),
        "cover": _cover(result, status_label, plan_type, completeness, safety),
        "summary": _summary(result, names, values, arrangement_labels),
        "content_policy": {
            "facts": f"仅使用已核验来源；缺失信息显示为{UNVERIFIED}",
            "advice": "通用执行建议，出发前需由领队确认",
        },
        "generated_at": format_datetime(result.generated_at),
        "comparison_id": result.comparison_id,
        "status_label": status_label,
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
