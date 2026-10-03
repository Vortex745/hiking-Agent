"""Code owns facts, safety, output and export eligibility; providers only rank."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from .models import (CandidateAssessment, ComparisonResult, DecisionRequest,
                     DecisionResult, RouteCandidate)
from .providers import DecisionProvider

REQUIRED_FACTS = ("distance_km", "ascent_m", "duration_hours", "weather", "map", "trail_status", "risk")
SEVERE = ("雷暴", "雷阵雨", "台风", "暴雨", "洪水", "雪崩", "封闭", "thunderstorm", "typhoon",
          "flood", "avalanche", "closed", "wildfire", "暴雪", "大风预警")
CLEAR_WEATHER = {"晴", "多云", "少云", "晴间多云", "阴", "clear", "sunny", "cloudy", "overcast"}


def assess_candidate(candidate: RouteCandidate) -> CandidateAssessment:
    clean = candidate.model_copy(deep=True)
    sources = {source.id: source for source in clean.sources}
    reasons = list(clean.gaps)
    for field in REQUIRED_FACTS:
        fact = clean.facts.get(field)
        supported = (fact is not None and fact.source_id in sources and fact.source_field == field and
                     clean.evidence_values.get(fact.source_id, {}).get(field) == fact.value)
        if supported and field in {"distance_km", "ascent_m", "duration_hours"}:
            supported = isinstance(fact.value, (int, float)) and fact.value >= 0
        if not supported:
            clean.facts[field] = None
            reasons.append(field + ": 资料不足或证据不支持")
    facts = clean.facts
    weather_value = facts["weather"].value if facts.get("weather") else None
    weather = weather_value.strip().lower() if isinstance(weather_value, str) else ""
    trail = str(facts["trail_status"].value).lower() if facts.get("trail_status") else ""
    risk = str(facts["risk"].value).lower() if facts.get("risk") else ""
    risk_status = "clear"
    action = None
    if any(word in weather or word in trail for word in SEVERE) or risk in {"high", "severe", "no_go"}:
        risk_status = "no_go"
        reasons.append("严重天气、封闭路线或高风险条件：禁止推荐")
        action = "取消该路线；等待官方解除风险后重新查询。"
    elif weather not in CLEAR_WEATHER or trail != "open" or risk not in {"low", "moderate"}:
        risk_status = "confirmation_required"
        reasons.append("安全资料缺失或无法判断")
        action = "补充最新天气、官方开放状态和风险资料后重新比较。"
    elif risk == "moderate":
        risk_status = "caution"
        reasons.append("存在中等风险，必须人工确认")
        action = "核对自身经验与风险承受能力，谨慎决定；暂不导出完整报告。"
    # Live weather/open/risk must be recent. Stale or unparsable dates close the gate.
    if risk_status != "no_go" and clean.data_mode == "live":
        try:
            timestamps = [datetime.fromisoformat(sources[facts[f].source_id].observed_at)
                          for f in ("weather", "trail_status", "risk") if facts.get(f)]
            if len(timestamps) < 3 or any(
                not t.tzinfo or not 0 <= (datetime.now(timezone.utc) - t).total_seconds() <= 21600
                for t in timestamps
            ):
                raise ValueError("stale safety data")
        except (ValueError, KeyError, TypeError):
            risk_status = "confirmation_required"
            reasons.append("安全资料过期或时间未知")
            action = "重新获取安全资料。"
    clean.gaps = list(dict.fromkeys(reasons))
    count = sum(facts.get(field) is not None for field in REQUIRED_FACTS)
    return CandidateAssessment(candidate=clean,
        evidence_status="complete" if count == len(REQUIRED_FACTS) else "partial" if count else "empty",
        risk_status=risk_status, reasons=clean.gaps, required_action=action)


async def compare_routes(candidates: list[RouteCandidate], provider: DecisionProvider,
                         *, preferences: str = "优先选择较轻松的路线",
                         comparison_id: str | None = None) -> ComparisonResult:
    if not 1 <= len(candidates) <= 5 or len({c.id for c in candidates}) != len(candidates):
        raise ValueError("Expected one to five uniquely identified routes")
    assessments = [assess_candidate(candidate) for candidate in candidates]
    eligible = [a for a in assessments if a.evidence_status == "complete" and a.risk_status == "clear"]
    status = "insufficient_evidence"
    decision = DecisionResult(provider=getattr(provider, "name", "unknown"), status="unavailable",
                              reason="evidence_or_safety_gate")
    recommended = None
    ranking = []
    if eligible:
        decision = await provider.decide(DecisionRequest(candidate_ids=[a.candidate.id for a in eligible],
            state={a.candidate.id: a.candidate.model_dump() for a in eligible}, preferences=preferences))
        if decision.status == "ok" and (set(decision.ranking) != {a.candidate.id for a in eligible}
                                        or len(decision.ranking) != len(eligible)):
            decision = DecisionResult(provider=decision.provider, model=decision.model,
                                      status="invalid_response", reason="invalid_ranking")
        if decision.status == "ok":
            status, ranking = "recommended", decision.ranking
            recommended = ranking[0]
        else:
            status = "provider_unavailable"
    elif all(a.risk_status == "no_go" for a in assessments):
        status = "no_go"
    elif any(a.evidence_status == "complete" and a.risk_status in {"confirmation_required", "caution"}
             for a in assessments):
        status = "confirmation_required"
    reasons = ["排序依据：用户偏好与已核验资料；高风险或资料不足候选未参与排序。"] if ranking else []
    for a in assessments:
        details = "、".join(f"{field}={fact.value}" for field, fact in a.candidate.facts.items() if fact)
        reasons.append(f"{a.candidate.name}：{details or '无可核验资料'}；{a.risk_status}")
    return ComparisonResult(comparison_id=comparison_id or str(uuid4()), status=status, decision=decision,
        candidates=assessments, ranking=ranking, recommended_id=recommended, alternatives=ranking[1:],
        explanation=reasons, required_action=None if status == "recommended" else "补充资料或处理风险后重新比较。",
        generated_at=datetime.now(timezone.utc).isoformat(), report_allowed=status == "recommended",
        data_mode="offline" if all(c.data_mode == "offline" for c in candidates) else "live")


def comparison_text(result: ComparisonResult) -> str:
    lines = [f"路线比较：{result.status}", f"决策：{result.decision.provider} / {result.decision.model or '未调用'}",
             f"confidence 语义：{result.decision.confidence.kind}"]
    if result.data_mode == "offline":
        lines.append("离线演示：合成数据，不可用于真实出行。")
    if result.recommended_id:
        names = {a.candidate.id: a.candidate.name for a in result.candidates}
        lines.append("推荐：" + names[result.recommended_id])
        lines.append("备选：" + "、".join(names[cid] for cid in result.alternatives))
    lines.extend(result.explanation)
    for assessment in result.candidates:
        lines.extend(assessment.reasons)
        if assessment.required_action:
            lines.append(assessment.required_action)
    if result.decision.reason:
        lines.append("决策状态：" + result.decision.reason)
    return "\n".join(lines)


def render_report(result: ComparisonResult) -> str:
    # Re-check stored data too; a client-supplied report flag has no authority.
    if not result.report_allowed or result.status != "recommended" or result.decision.status != "ok":
        raise ValueError("report_blocked")
    eligible = {a.candidate.id for a in result.candidates
                if (checked := assess_candidate(a.candidate)).evidence_status == "complete"
                and checked.risk_status == "clear"}
    if not result.recommended_id or any(cid not in eligible for cid in result.ranking):
        raise ValueError("report_blocked")
    lines = ["AI Hiking 路线比较报告", "生成时间：" + result.generated_at,
             comparison_text(result), f"confidence 值：{result.decision.confidence.value}"]
    for assessment in result.candidates:
        lines.append("来源 — " + assessment.candidate.name)
        for source in assessment.candidate.sources:
            lines.extend([f"[{source.id}] {source.tool} {source.url or '引用片段'}",
                          source.excerpt, "观测时间：" + source.observed_at])
    return "\n".join(lines)
