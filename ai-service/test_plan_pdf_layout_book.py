"""Ticket 08 — the confirmed structured plan renders as a new-layout plan book.

Two public seams are under test, and nothing else:

* :func:`route_compare.report_pdf.build_report_document` — the pure derivation of
  the layout document from a gate-passed ``ComparisonResult`` plus the plan-side
  status the earlier tickets produced (``plan_type``, the rule verdict, the safety
  record, the stored arrangement record).
* :func:`agent_v2.computer._render_pdf_bytes` — the rendered bytes, read back
  through PyMuPDF page by page, because a layout change nobody has seen rendered
  is not done.

The acceptance copy comes from the ticket (封面与计划状态 / 路线、日期、人数摘要 /
时间线、风险决策区、装备和出发前检查 / 候选路线和来源证据可读 / 跨页重复表头 /
无装饰性图片), never from the implementation.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from agent_v2.computer import _render_pdf_bytes
from route_compare.fixtures import fixture_candidates
from route_compare.models import BudgetItem, PlanDetails, PlanStage
from route_compare.providers import DeterministicProvider, ProviderConfig
from route_compare.report_pdf import build_report_document
from route_compare.service import compare_routes

fitz = pytest.importorskip("fitz")


def _result(scenario: str = "complete", plan_details: PlanDetails | None = None):
    """A gate-passed offline comparison, the same way the export flow builds one."""
    result = asyncio.run(compare_routes(
        fixture_candidates(scenario),
        DeterministicProvider(ProviderConfig(data_mode="offline")),
        comparison_id="run:layout"))
    if plan_details is not None:
        result.plan_details = plan_details
    return result


def _pages(document: dict) -> list[str]:
    payload = _render_pdf_bytes(json.dumps(document, ensure_ascii=False))
    with fitz.open(stream=payload, filetype="pdf") as pdf:
        return [page.get_text() for page in pdf]


# ---- AC1: 封面与计划状态 ----------------------------------------------------


def test_the_first_page_is_a_cover_naming_the_plan_status() -> None:
    """A confirmed plan opens on its own cover page carrying the plan status."""
    document = build_report_document(
        _result(), plan_type="route_plan", completeness="confirmation_required")
    pages = _pages(document)

    cover = pages[0]
    assert "AI Hiking 徒步计划书" in cover
    assert "计划状态" in cover
    assert "正式路线计划" in cover
    assert "有待确认安排" in cover
    # The cover is a page of its own: the numbered body starts after it.
    assert "计划摘要" not in cover
    assert "行程时间线" not in cover
    assert "计划摘要" in pages[1]
    assert len(pages) > 1


def test_a_generic_draft_cover_is_never_presented_as_a_route_plan() -> None:
    """``plan_type`` decides the cover claim; a thin draft cannot wear the badge."""
    document = build_report_document(
        _result("partial"), plan_type="generic_draft", completeness="supplement_required")
    cover = _pages(document)[0]

    assert "通用草案" in cover
    assert "正式路线计划" not in cover
    assert "核心安排待补充" in cover


def test_a_cover_without_a_recorded_status_says_so_instead_of_claiming_one() -> None:
    """Absent status stays absent — the cover never infers a verdict."""
    cover = _pages(build_report_document(_result()))[0]

    assert "待核验" in cover
    assert "正式路线计划" not in cover
    assert "安排齐全" not in cover


def test_a_safety_refusal_is_on_the_cover_in_the_gate_own_words() -> None:
    """Ticket 07's refusal travels to the cover, copied from the record."""
    from agent_v2.plan_workflow import SAFETY_MESSAGE, PlanSafetyReason

    safety = {
        "hard_stop": True,
        "codes": [PlanSafetyReason.SEVERE_WEATHER],
        "formal_route_plan_allowed": False,
        "message": SAFETY_MESSAGE[PlanSafetyReason.SEVERE_WEATHER],
        "comparison_id": "run:layout",
        "comparison_status": "no_go",
        "gate_reasons": [],
        "required_action": "",
    }
    cover = _pages(build_report_document(_result("severe_weather"), plan_type="generic_draft",
                                         safety=safety))[0]

    assert SAFETY_MESSAGE[PlanSafetyReason.SEVERE_WEATHER] in cover.replace("\n", "")


# ---- AC2: 路线、日期、人数摘要 ----------------------------------------------


def test_the_summary_page_names_the_route_date_and_party_size() -> None:
    document = build_report_document(
        _result(plan_details=PlanDetails(activity_date="2026-10-18", participants=4,
                                         meeting_time="08:00", leader="领队甲")),
        plan_type="route_plan")
    body = "\n".join(_pages(document))

    assert "计划摘要" in body
    assert "演示林间环线" in body
    assert "2026-10-18" in body
    assert "4 人" in body


def test_a_summary_line_the_user_never_filled_stays_待确认() -> None:
    """The summary shows the gap rather than a derived value (seam contract AC2)."""
    document = build_report_document(_result(), plan_type="route_plan")
    summary = _pages(document)[1]

    assert "待确认" in summary
    assert all(token not in summary for token in ("预计出发", "估算"))


# ---- AC3: 时间线、风险决策区、装备、出发前检查 ------------------------------


def test_the_body_has_a_timeline_a_risk_block_gear_and_a_predeparture_check() -> None:
    body = "\n".join(_pages(build_report_document(_result(), plan_type="route_plan")))

    for heading in ("行程时间线", "风险决策区", "装备与补给", "出发前检查"):
        assert heading in body, heading


def test_the_user_stages_are_the_timeline_and_the_template_says_it_is_a_template() -> None:
    with_stages = build_report_document(
        _result(plan_details=PlanDetails(itinerary=[
            PlanStage(time="08:00", activity="集合、点名与装备检查"),
            PlanStage(time="16:00", activity="清点人员、带离垃圾并报平安")])),
        plan_type="route_plan")
    body = "\n".join(_pages(with_stages))
    assert "08:00" in body and "16:00" in body

    without = _pages(build_report_document(_result(), plan_type="route_plan"))
    assert "通用行程模板" in "\n".join(without)
    # No segment time is ever derived from the total duration (AC2 of the contract).
    assert not re.search(r"行程时间线[\s\S]{0,400}\d{1,2}:\d{2}", "".join(without))


def test_the_risk_block_is_rendered_differently_from_an_ordinary_table() -> None:
    """风险重点 means visual emphasis, measured as colour, not as a heading."""
    document = build_report_document(_result(), plan_type="route_plan")
    payload = _render_pdf_bytes(json.dumps(document, ensure_ascii=False))
    with fitz.open(stream=payload, filetype="pdf") as pdf:
        spans = [span for page in pdf for block in page.get_text("dict")["blocks"]
                 for line in block.get("lines", []) for span in line["spans"]]
        risk = [span for span in spans if "触发条件" in span["text"] or "立即处置" in span["text"]]
        ordinary = [span for span in spans if "检查项" in span["text"] or "负责人" in span["text"]]
        assert risk and ordinary
        assert {span["color"] for span in risk} != {span["color"] for span in ordinary}


# ---- AC4: 候选路线与来源证据保持可读 ---------------------------------------


def test_every_candidate_and_every_source_stays_readable() -> None:
    result = _result()
    document = build_report_document(result, plan_type="route_plan")
    body = "\n".join(_pages(document))

    for assessment in result.candidates:
        assert assessment.candidate.name in body
        for source in assessment.candidate.sources:
            assert source.id in body
            assert source.url in body
    assert "候选路线对比" in body and "来源与证据" in body


def test_appendix_text_stays_inside_the_print_margins() -> None:
    document = build_report_document(
        _result(plan_details=PlanDetails(itinerary=[
            PlanStage(time=f"第 {i} 天", activity="集合、装备检查与结伴徒步；" * 12)
            for i in range(1, 16)])),
        plan_type="route_plan")
    payload = _render_pdf_bytes(json.dumps(document, ensure_ascii=False))
    with fitz.open(stream=payload, filetype="pdf") as pdf:
        for page in pdf:
            for block in page.get_text("blocks"):
                assert block[0] >= 40 and block[2] <= page.rect.width - 35
                assert block[1] >= 15 and block[3] <= page.rect.height - 25


# ---- AC5: 跨页表格重复表头，风险章节不从表格中间开始 ----------------------


def test_a_timeline_that_splits_across_pages_repeats_its_header_row() -> None:
    """The header must reappear on the continuation page, demonstrated by rendering."""
    document = build_report_document(
        _result(plan_details=PlanDetails(itinerary=[
            PlanStage(time=f"{i:02d}:00", activity="结伴徒步、补给与体感检查的长安排段落" * 6)
            for i in range(1, 26)])),
        plan_type="route_plan")
    pages = _pages(document)
    timeline_pages = [page for page in pages if "完成检查" in page]

    assert len(timeline_pages) >= 2, "the sample must actually cross a page boundary"
    for page in timeline_pages:
        assert "时间线" in page or "时间" in page
        assert page.count("完成检查") >= 1


def test_a_candidate_table_that_splits_across_pages_repeats_its_header_row() -> None:
    from route_compare.models import CandidateAssessment

    result = _result()
    result.candidates = [assessment.model_copy(deep=True) for assessment in result.candidates] * 8
    for index, assessment in enumerate(result.candidates):
        assessment.candidate.name = f"演示林间环线 {index}"
    document = build_report_document(result, plan_type="route_plan")
    pages = _pages(document)
    comparison_pages = [page for page in pages if "累计爬升" in page and "开放状态" in page]

    assert len(comparison_pages) >= 2, "the sample must actually cross a page boundary"


#: A risk set with more rows than the template ships, expressed in the same public
#: document protocol the renderer consumes: the AC is about the layout of a risk
#: table that cannot fit the space left on a page, so the sweep needs a tall block.
_EXTRA_RISK_ROWS = [{"trigger": f"风险情景 {index}：天气、体力或路线状态恶化到第 {index} 档",
                     "action": "停止前进，结伴撤退，按预案联系外部支援并清点人员" * 2,
                     "severity": "立即处置"} for index in range(1, 10)]


@pytest.mark.parametrize("padding_stages", [0, 5, 9, 13, 17])
def test_the_risk_block_never_opens_on_a_table_it_cannot_finish(padding_stages: int) -> None:
    """A risk section pushed against a page break moves whole, not mid-table."""
    details = PlanDetails(
        itinerary=[PlanStage(time=f"{i:02d}:00", activity="结伴徒步、补给与体感检查")
                   for i in range(1, padding_stages + 1)],
        budget=[BudgetItem(item=f"演示条目 {i}", amount_yuan=10 * i) for i in range(1, 5)])
    document = build_report_document(_result(plan_details=details), plan_type="route_plan")
    document["risk"]["rows"] = _EXTRA_RISK_ROWS
    pages = _pages(document)
    risk_pages = [page for page in pages if "风险决策区" in page]

    assert risk_pages
    for page in risk_pages:
        flat = page.replace("\n", "")
        assert document["risk"]["rows"][0]["trigger"] in flat
        assert _EXTRA_RISK_ROWS[-1]["trigger"] in flat, "the block must not start mid-table"
        body_lines = [line for line in page.splitlines() if line.strip()]
        assert "风险决策区" not in body_lines[-1]


def test_the_risk_sweep_actually_stressed_a_page_boundary() -> None:
    """Guard for the sweep above: at least one case starts 风险决策区 below the top."""
    started_mid_page = 0
    for padding in (5, 9, 13, 17):
        document = build_report_document(
            _result(plan_details=PlanDetails(itinerary=[
                PlanStage(time=f"{i:02d}:00", activity="结伴徒步、补给与体感检查")
                for i in range(1, padding + 1)])), plan_type="route_plan")
        document["risk"]["rows"] = _EXTRA_RISK_ROWS
        for page in _pages(document):
            if "风险决策区" in page:
                lines = [line for line in page.splitlines() if line.strip()]
                # A running header is line 0; the risk block opening later means the
                # page already carried other content before it moved or was split.
                if lines.index([l for l in lines if "风险决策区" in l][0]) > 2:
                    started_mid_page += 1
    assert started_mid_page, "no case put the risk block against a boundary"


# ---- The cover reads what the plan flow recorded ---------------------------


def test_the_export_flow_covers_with_the_stored_status_and_arrangements(tmp_path) -> None:
    """The document the flow hands the renderer carries the recorded status.

    The cover is only worth having if a real export fills it: the verdict row and
    the arrangement row are read at document time, and the user's free-text budget
    still becomes no number.
    """
    from types import SimpleNamespace

    from agent_v2.computer import LocalNativeComputer
    from agent_v2.memory_service import MemoryService
    from agent_v2.persistence import CheckpointsStore, ControlStore
    from agent_v2.plan_workflow import evaluate_plan_completeness
    from agent_v2.repositories.memory import MemoryStore
    from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
    from agent_v2.runs import RunRepository
    from api.chat_v2 import _cleanup_sandbox

    settings = SimpleNamespace(agent_v2_enabled=True, openai_api_key="test", openai_model="test",
                               agent_v2_state_dir=str(tmp_path), route_compare_data_mode="offline",
                               route_decision_provider="deterministic")
    repo = RunRepository(control=ControlStore(str(tmp_path / "control.sqlite")),
                         checkpoints=CheckpointsStore(str(tmp_path / "checkpoints.sqlite")))
    memory = MemoryService(MemoryStore(str(tmp_path / "memory.sqlite")))
    owner = memory.resolve_subject(None)
    memory.claim_conversation(owner.subject_id, "conv-cover")
    arrangements = {"meeting_time": "07:30", "meeting_point": "山谷停车场", "leader": "领队乙",
                    "transport": "自驾往返", "accommodation": "无", "budget": "全队AA，油费另计",
                    "emergency_contact": "13800000000", "retreat_plan": "原路返回至停车场"}
    draft = repo.save_plan_draft(conversation_id="conv-cover", stage="final_reviewed",
                                 destination_or_route="演示山谷环线", activity_date="2026-10-18",
                                 participants=6)
    repo.save_plan_arrangements(conversation_id="conv-cover", draft_id=draft.draft_id,
                                arrangements=arrangements)
    verdict = evaluate_plan_completeness(destination_or_route="演示山谷环线", activity_date="2026-10-18",
                                         participants=6, arrangements=arrangements)
    repo.save_completeness_check(conversation_id="conv-cover", draft_id=draft.draft_id, verdict=verdict)

    contents: list[str] = []

    class RecordingComputer(LocalNativeComputer):
        def generate_pdf(self, workspace_path, content, filename):
            contents.append(content)
            return super().generate_pdf(workspace_path, content, filename)

    computer = RecordingComputer()

    async def kernel(**kw):
        compare = next(t for t in kw["extra_tools"] if t.name == "route_compare")
        report = next(t for t in kw["extra_tools"] if t.name == "report_export")
        output = await compare.execute(compare.args_schema.model_validate({"candidates": [
            {"name": "演示林间环线", "city": "演示城市", "origin": "起点", "destination": "终点"}]}))
        comparison_id = json.loads(output.output)["comparison_id"]
        await report.execute(report.args_schema.model_validate({"comparison_id": comparison_id}))
        return "模型正文不应进入结构化计划书", []

    orchestrator = RunOrchestrator(settings=settings, runs_repo=repo, memory=memory, kernel_turn=kernel,
                                   computer_factory=lambda *args: computer, sandbox_cleanup=_cleanup_sandbox)

    async def pause():
        return [e async for e in orchestrator.stream_chat(ChatFlowRequest(
            message="导出比较报告", request_id="cover-test", conversation_id="conv-cover",
            profile_id="route_plan"), subject_id=owner.subject_id)]

    events = asyncio.run(pause())
    approval = repo.list_pending_approvals_for_run(events[-1].run_id)[0]
    orchestrator.resolve_approval_decision(approval.approval_id, decision="approve",
                                           decision_request_id="cover-approve")

    async def resume():
        return [e async for e in orchestrator.stream_resume(run_id=events[-1].run_id,
                                                            from_seq=events[-1].sequence)]

    asyncio.run(resume())
    assert contents, "the approved export must reach the renderer"
    document = json.loads(contents[0])
    cover = {line["label"]: line["value"] for line in document["cover"]["status_lines"]}
    metadata = {row["label"]: row["value"] for row in document["plan_metadata"]}

    assert cover["草案类型"] == "正式路线计划"
    assert cover["审阅结论"] != "待核验"
    assert metadata["集合时间"] == "07:30" and metadata["应急联络"] == "13800000000"
    assert metadata["费用预算"] == "全队AA，油费另计"
    # Free text is carried as text: no amount and no total is read out of it.
    assert document["budget_total"] == "待确认"
    assert document["budget_note"] == "全队AA，油费另计"


def test_a_stale_verdict_row_does_not_become_the_cover_status(tmp_path) -> None:
    """The verdict counts only while it belongs to the draft on record."""
    from agent_v2.persistence import CheckpointsStore, ControlStore
    from agent_v2.plan_workflow import evaluate_plan_completeness
    from agent_v2.run_orchestration import _pdf_plan_status_inputs
    from agent_v2.runs import RunRepository

    repo = RunRepository(control=ControlStore(str(tmp_path / "control.sqlite")),
                         checkpoints=CheckpointsStore(str(tmp_path / "checkpoints.sqlite")))
    verdict = evaluate_plan_completeness(destination_or_route="演示山谷", activity_date="2026-10-18",
                                         participants=6, arrangements=None)
    first = repo.save_plan_draft(conversation_id="conv-stale", stage="first_reviewed",
                                 destination_or_route="演示山谷", activity_date="2026-10-18",
                                 participants=6)
    repo.save_completeness_check(conversation_id="conv-stale", draft_id=first.draft_id, verdict=verdict)
    assert _pdf_plan_status_inputs(repo, "conv-stale")["completeness"] == verdict.verdict

    # A verdict written for another draft generation must not describe this one
    # (the row is scoped by ``draft_id``; the flow re-decides, this only reads).
    repo.save_completeness_check(conversation_id="conv-stale", draft_id="another-draft",
                                 verdict=verdict)
    inputs = _pdf_plan_status_inputs(repo, "conv-stale")
    assert "completeness" not in inputs
    assert inputs["plan_type"] == "route_plan"
    assert _pdf_plan_status_inputs(None, "") == {"plan_type": "route_plan"}


# ---- The rename coupling: one vocabulary for both stories ------------------



@pytest.mark.parametrize("heading", ["行程时间线", "行程安排"])
def test_a_timeline_heading_lands_in_the_timeline_section_whichever_name_it_uses(heading: str) -> None:
    """_text_story buckets by substring, so a rename must carry its old name.

    A heading that stopped matching would not error: its content would be filed
    under the previous section, which is how a plan silently loses a table.
    """
    content = (f"# 徒步计划\n## 路线摘要\n里程 8 公里\n\n## {heading}\n"
               "| 时间 | 活动 |\n| --- | --- |\n| 07:30 | 首班集合 |\n")
    text = "\n".join(_pages_text(content))

    timeline_body = text.split("行程时间线", 1)[1].split("交通住宿与联络", 1)[0]
    assert "07:30" in timeline_body and "首班集合" in timeline_body
    assert "07:30" not in text.split("路线摘要", 1)[1].split("行程时间线", 1)[0]


@pytest.mark.parametrize("heading", ["风险决策区", "风险与应急"])
def test_a_risk_heading_lands_in_the_risk_section_whichever_name_it_uses(heading: str) -> None:
    content = f"# 徒步计划\n## {heading}\n- 雷雨立即取消行程\n\n## 出发前检查\n- 检查头灯\n"
    text = "\n".join(_pages_text(content))

    risk_body = text.split("风险决策区", 1)[1].split("出发前检查", 1)[0]
    assert "雷雨立即取消行程" in risk_body


def _pages_text(content: str) -> list[str]:
    with fitz.open(stream=_render_pdf_bytes(content), filetype="pdf") as pdf:
        return [page.get_text() for page in pdf]


# ---- AC6: 无核验素材时不添加装饰性地图或户外图片 --------------------------


def test_the_book_carries_no_decorative_imagery() -> None:
    payload = _render_pdf_bytes(json.dumps(build_report_document(_result(), plan_type="route_plan"),
                                           ensure_ascii=False))
    with fitz.open(stream=payload, filetype="pdf") as pdf:
        assert sum(len(page.get_images(full=True)) for page in pdf) == 0
    assert b"/Subtype /Image" not in payload
    assert b"/DCTDecode" not in payload and b"/JPXDecode" not in payload


def test_an_unverified_map_stays_a_待核验_line_and_not_a_picture() -> None:
    document = build_report_document(_result("partial"), plan_type="generic_draft")
    body = "\n".join(_pages(document))

    assert "路线地图" in body
    assert "待核验" in body
