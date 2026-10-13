"""Artifact format contracts for the development computer implementation."""

from __future__ import annotations

from pathlib import Path

from agent_v2.computer import LocalNativeComputer


def test_the_renderer_types_no_plan_status_marker_of_its_own() -> None:
    """The 待确认 / 待核验 markers are ``plan_workflow``'s, never the renderer's.

    ``agent_v2/computer.py`` both FILLS these markers as values and COMPARES against
    them (:func:`_report_story`'s ``if note and note != UNCONFIRMED`` budget and
    required-action guards). With a literal on one side and
    :data:`agent_v2.plan_workflow.UNCONFIRMED` on the other, rewording the marker
    silently strands the comparison — the blank note starts rendering as user copy —
    and no runtime test can see a rename that has not happened yet. So the rule is
    pinned at the source level, the same way this repo pins what a browser-only
    claim cannot prove.

    The import direction is part of the claim: ``computer.py`` sits in
    ``agent_v2``, so reading the vocabulary from :mod:`agent_v2.plan_workflow` is the
    correct way down (the reverse edge is what finding 9 removed from
    :mod:`route_compare.report_pdf`).
    """
    import ast
    import inspect

    import agent_v2.computer as computer_module
    from agent_v2 import plan_workflow

    path = Path(inspect.getfile(computer_module))
    tree = ast.parse(path.read_text(encoding="utf-8"))

    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))

    literals = [node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings]
    typed = [text for text in literals if plan_workflow.UNCONFIRMED in text
             or plan_workflow.UNVERIFIED in text]
    assert typed == [], f"placeholders must come from plan_workflow, not as literals: {typed}"

    # …and the shared constants really are imported here, so the check above cannot
    # be satisfied by deleting the markers instead of routing them.
    imports_plan_workflow = any(
        isinstance(node, ast.ImportFrom)
        and (node.module or "").endswith("plan_workflow")
        and {"UNCONFIRMED", "UNVERIFIED"} <= {alias.name for alias in node.names}
        for node in ast.walk(tree))
    assert imports_plan_workflow, (
        "computer.py must read UNCONFIRMED/UNVERIFIED from agent_v2.plan_workflow")


def test_the_blank_budget_note_and_action_are_gated_on_the_shared_marker() -> None:
    """Same rule, observed as behaviour: an absent arrangement is not user copy.

    The document builder writes :data:`UNCONFIRMED` for a budget the user left
    blank; the renderer must recognise THAT marker and not print a 用户预算说明 row
    for it, while a real note still prints both the row and the quote. Asserted
    through the constant rather than the word, so the two sides cannot drift apart
    and stay green.
    """
    import asyncio
    import json

    import pytest
    fitz = pytest.importorskip("fitz")

    from agent_v2.computer import _render_pdf_bytes
    from agent_v2.plan_workflow import UNCONFIRMED
    from route_compare.fixtures import fixture_candidates
    from route_compare.providers import DeterministicProvider, ProviderConfig
    from route_compare.report_pdf import build_report_document
    from route_compare.service import compare_routes

    result = asyncio.run(compare_routes(
        fixture_candidates("complete"),
        DeterministicProvider(ProviderConfig(data_mode="offline")),
        comparison_id="run:marker"))
    document = build_report_document(result)
    # The builder's own answer for a user who never stated a budget: the marker.
    assert document["budget_note"] == UNCONFIRMED

    def text_of(payload: bytes) -> str:
        with fitz.open(stream=payload, filetype="pdf") as pdf:
            return "\n".join(page.get_text() for page in pdf)

    blank = text_of(_render_pdf_bytes(json.dumps(document, ensure_ascii=False)))
    assert "用户预算说明" not in blank, "an absent budget is not user copy"
    assert "出发前必须完成：" not in blank

    stated = dict(document, budget_note="300 元/人，含保险",
                  required_action="出发前重新查询天气与开放状态")
    shown = text_of(_render_pdf_bytes(json.dumps(stated, ensure_ascii=False)))
    assert "用户预算说明" in shown and "预算原文：300 元/人，含保险" in shown
    assert "出发前必须完成：出发前重新查询天气与开放状态" in shown


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
        for label in ("计划摘要", "计划信息", "路线摘要", "行程时间线", "装备与补给", "风险决策区",
                      "出发前检查", "候选路线对比", "排序依据与说明", "交通住宿与联络",
                      "人员分工", "费用预算", "来源与证据"):
            assert label in combined, label
        assert "已列预算合计：待确认" in combined
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
    for label in ("费用预算", "风险决策区", "出发前检查", "人员分工", "来源与证据"):
        assert label in text
    assert "08:00" in text and "集合" in text and "雨具" in text
    assert "| ---" not in text and "**集合**" not in text
    assert "<b>标签</b>" in text
    assert "活动安排草案" in text and "待核验" in text
    # The OLD heading name still routes: 行程安排 is an alias of 行程时间线, so a
    # draft written before the rename cannot have its table filed under 路线摘要.
    timeline = text.split("行程时间线", 1)[1]
    assert "08:00" in timeline.split("交通住宿与联络", 1)[0]
