"""Artifact format contracts for the development computer implementation."""

from __future__ import annotations

from agent_v2.computer import LocalNativeComputer


def test_local_pdf_is_a_real_pdf(tmp_path) -> None:
    computer = LocalNativeComputer()
    workspace = computer.create_workspace("pdf-test")
    try:
        relative_path = computer.generate_pdf(workspace, "# 徒步计划\n带上雨具", "plan")
        payload = computer.read_artifact_bytes(workspace, relative_path)
        assert relative_path == "plan.pdf"
        assert payload.startswith(b"%PDF-")
    finally:
        computer.cleanup(workspace)


def test_structured_report_pdf_is_a_real_pdf(tmp_path) -> None:
    """The designed route-comparison layout renders from the JSON document protocol."""
    import json

    from route_compare.report_pdf import build_report_document
    from route_compare.service import render_report
    from route_compare.fixtures import fixture_candidates
    from route_compare.models import DecisionResult
    from route_compare.service import compare_routes

    import asyncio

    class OrderProvider:
        name = "deterministic"

        async def decide(self, request):
            return DecisionResult(provider=self.name, model="offline-order-v1", status="ok",
                                  ranking=list(request.candidate_ids))

    result = asyncio.run(compare_routes(
        fixture_candidates("complete"), OrderProvider(), preferences="新手", comparison_id="run:pdf"))
    content = json.dumps(build_report_document(result), ensure_ascii=False)

    computer = LocalNativeComputer()
    workspace = computer.create_workspace("pdf-report")
    try:
        relative_path = computer.generate_pdf(workspace, content, "hiking-plan.pdf")
        payload = computer.read_artifact_bytes(workspace, relative_path)
        assert relative_path == "hiking-plan.pdf"
        assert payload.startswith(b"%PDF-")
        assert b"/Font" in payload
    finally:
        computer.cleanup(workspace)


def test_plan_document_has_complete_sections_and_explicit_unknowns() -> None:
    """The export protocol stays a plan book, even when facts are incomplete."""
    import asyncio

    from route_compare.fixtures import fixture_candidates
    from route_compare.models import DecisionResult
    from route_compare.report_pdf import build_report_document
    from route_compare.service import compare_routes

    class OrderProvider:
        name = "deterministic"

        async def decide(self, request):
            return DecisionResult(provider=self.name, model="offline-order-v1", status="ok",
                                  ranking=list(request.candidate_ids))

    result = asyncio.run(compare_routes(
        fixture_candidates("partial"), OrderProvider(), comparison_id="run:sections"))
    document = build_report_document(result)

    assert document["document_type"] == "hiking_plan"
    assert document["schema_version"] == "2.0"
    for section in ("plan_metadata", "route_overview", "itinerary", "logistics", "roles", "budget",
                    "equipment", "risk_controls", "checklist", "sources"):
        assert document[section], section
    assert any(item["value"] == "待核验" for item in document["route_overview"])
    assert all("status" in item for item in document["checklist"])
    assert document["content_policy"]["facts"].startswith("仅使用已核验来源")
    assert all(row["window"] == "待确认" for row in document["itinerary"])


def test_pdf_plan_pages_have_readable_sections_and_no_orphan_headings() -> None:
    import asyncio
    import json
    import re

    import pytest
    fitz = pytest.importorskip("fitz")
    from agent_v2.computer import _render_pdf_bytes
    from route_compare.fixtures import fixture_candidates
    from route_compare.models import BudgetItem, PlanDetails, PlanStage
    from route_compare.providers import DeterministicProvider, ProviderConfig
    from route_compare.report_pdf import build_report_document
    from route_compare.service import compare_routes

    result = asyncio.run(compare_routes(fixture_candidates(),
        DeterministicProvider(ProviderConfig(data_mode="offline"))))
    result.plan_details = PlanDetails(leader="领队甲", itinerary=[
        PlanStage(time=f"第 {i} 天", activity="集合、装备检查与结伴徒步；" * 12) for i in range(1, 16)],
        budget=[BudgetItem(item="交通", amount_yuan=200)])
    payload = _render_pdf_bytes(json.dumps(build_report_document(result), ensure_ascii=False))
    with fitz.open(stream=payload, filetype="pdf") as pdf:
        texts = [page.get_text() for page in pdf]
        combined = "\n".join(texts)
        for label in ("计划信息", "路线摘要", "行程安排", "装备与补给", "风险与应急", "出发前检查",
                      "候选路线对比", "排序依据与说明", "交通住宿与联络", "人员分工", "费用预算", "来源与证据"):
            assert label in combined, label
        assert "已列预算合计：200 元" in combined
        assert len(pdf) > 2
        for page in pdf:
            body_lines = [line for line in page.get_text().splitlines()
                          if line.strip() and "自动生成" not in line and not re.match(r"第 \d+ 页", line)]
            assert not re.match(r"\d{2} .*", body_lines[-1]), "A heading must stay with its content"
            for block in page.get_text("blocks"):
                assert block[0] >= 40 and block[2] <= page.rect.width - 35
                assert block[1] >= 15 and block[3] <= page.rect.height - 25


def test_user_arrangements_cannot_overwrite_verified_route_facts() -> None:
    import pytest
    from pydantic import ValidationError
    from agent_v2.route_tools import ReportParams

    with pytest.raises(ValidationError):
        ReportParams.model_validate({"comparison_id": "run:plan", "plan_details": {"distance_km": 999}})
    with pytest.raises(ValidationError):
        ReportParams.model_validate({"comparison_id": "run:plan",
                                     "plan_details": {"budget": [{"item": "交通", "amount_yuan": -1}]}})


def test_markdown_plan_renders_tables_and_fills_missing_sections() -> None:
    import pytest
    fitz = pytest.importorskip("fitz")
    from agent_v2.computer import _render_pdf_bytes

    content = "# 徒步计划\n## 行程安排\n| 时间 | 活动 |\n| --- | --- |\n| 08:00 | **集合** |\n\n## 装备\n- 雨具\n- literal <b>标签</b>\n"
    with fitz.open(stream=_render_pdf_bytes(content), filetype="pdf") as pdf:
        text = "\n".join(page.get_text() for page in pdf)
    for label in ("费用预算", "风险与应急", "出发前检查", "人员分工", "来源与证据"):
        assert label in text
    assert "08:00" in text and "集合" in text and "雨具" in text
    assert "| ---" not in text and "**集合**" not in text
    assert "<b>标签</b>" in text
    assert "活动安排草案" in text and "待核验" in text
