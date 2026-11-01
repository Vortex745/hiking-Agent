"""Ticket 09 — Run-scoped Computer (sandbox) abstraction.

Spec decisions:
* #19 — Production creates a run-scoped Remote E2B Computer. Development
  may use LocalNative or WSL LocalVM. There is NO automatic host fallback
  in production.
* #20 — E2B workspace is the only writable execution space. The sandbox
  template pins runtime, artifact libraries, and fonts but contains no
  model, MCP, storage, or provider credentials.
* #11 — artifact profile uses E2B-scoped Read/Write/Edit/Glob/Grep; host
  paths and delete operations are unavailable.

Ticket 01 (harness-p0-p1): ``RemoteE2BComputer`` is NOT a stub that mimics
the E2B lifecycle on the host. No live E2B client is wired yet, so the
placeholder now FAILS CLOSED: every operation raises
:class:`SandboxUnavailableError` before any host workspace, process, or
file is touched. There is no automatic fallback to
:class:`LocalNativeComputer` — a run selecting the remote sandbox ends in
a terminal ``sandbox_unavailable`` error, and basic chat plus explicitly
configured trusted ``local_native`` development keep working unchanged.

Real E2B integration (a later ticket) replaces ``RemoteE2BComputer`` with
a live client; the interface and fail-closed contract stay the same.

Ticket 10 adds manifest + byte retrieval so the host can re-verify
sandbox-produced artifacts before publishing them. The host never
trusts sandbox-declared size/sha — it re-derives them from the bytes
read via :meth:`Computer.read_artifact_bytes`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agent_v2.artifacts import ArtifactManifest, compute_sha256
# The two plan status markers belong to the plan vocabulary, not to this renderer:
# :func:`_report_story` both FILLS them as values and COMPARES against them (the
# budget-note and required-action guards), so a literal typed here would be a second
# copy of a word another module owns. ``agent_v2.computer`` → ``agent_v2.plan_workflow``
# is the correct direction (plan_workflow imports nothing), pinned by
# ``test_the_plan_document_layers_type_no_status_marker_of_its_own``.
from agent_v2.plan_workflow import UNCONFIRMED, UNVERIFIED

logger = logging.getLogger("ai-service.agent_v2.computer")


SandboxType = Literal["local_native", "wsl_localvm", "remote_e2b"]


_PDF_FONT_DIRS = [
    r"C:\Windows\Fonts",
    "/usr/share/fonts/truetype",
    "/usr/share/fonts/opentype/noto",
    "/usr/share/fonts",
    str(Path(__file__).resolve().parent.parent / "assets" / "fonts"),
]
_PDF_FONT_PAIRS = [
    # (regular file, bold file or None) — first existing pair wins.
    ("msyh.ttc", "msyhbd.ttc"),
    ("NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"),
    ("NotoSansSC-Regular.otf", "NotoSansSC-Bold.otf"),
    ("simhei.ttf", None),
    ("wqy-microhei.ttc", None),
]

_PDF_BRAND = "AI Hiking"
_PDF_PRIMARY = "#15803d"
_PDF_PRIMARY_DARK = "#14532d"
_PDF_PRIMARY_BG = "#f0fdf4"
_PDF_TEXT = "#1f2937"
_PDF_MUTED = "#6b7280"
_PDF_BORDER = "#e5e7eb"
_PDF_ZEBRA = "#f8fafc"
_PDF_AMBER_BG = "#fffbeb"
_PDF_AMBER_BORDER = "#f59e0b"
_PDF_AMBER_TEXT = "#92400e"
_PDF_LINK = "#1d4ed8"
#: The risk block is the page the leader reads twice, so it is the one block that
#: is not green: a red rule, a red head row and a red panel behind it.
_PDF_RISK = "#b91c1c"
_PDF_RISK_DARK = "#7f1d1d"
_PDF_RISK_BG = "#fef2f2"


def _register_pdf_font() -> tuple[str, str]:
    """Register a Unicode CJK font pair (regular, bold) with usable Latin glyphs.

    STSong-Light renders ASCII with broken spacing (``reco m mended``), so a
    real TTF/TTC (Microsoft YaHei / Noto Sans CJK / SimHei) is preferred; the
    CID font stays as the deterministic fallback for minimal environments.
    Returns ``(regular_font_name, bold_font_name)``.
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont

    override = os.getenv("AI_HIKING_PDF_FONT")
    candidates: list[tuple[Path, Path | None]] = []
    if override:
        override_path = Path(override)
        bold_guess = override_path.with_name(override_path.stem + "-Bold" + override_path.suffix)
        candidates.append((override_path, bold_guess if bold_guess.exists() else None))
    for regular_name, bold_name in _PDF_FONT_PAIRS:
        for base in _PDF_FONT_DIRS:
            regular = Path(base) / regular_name
            if regular.exists():
                bold = (Path(base) / bold_name) if bold_name else None
                candidates.append((regular, bold if bold and bold.exists() else None))
                break

    for regular_path, bold_path in candidates:
        try:
            regular_name = "PdfCJK"
            kwargs: dict[str, Any] = {}
            if regular_path.suffix.lower() == ".ttc":
                kwargs["subfontIndex"] = 0
            pdfmetrics.registerFont(TTFont(regular_name, str(regular_path), **kwargs))
            bold_name = regular_name + "-Bold"
            if bold_path:
                bold_kwargs = {"subfontIndex": 0} if bold_path.suffix.lower() == ".ttc" else {}
                try:
                    pdfmetrics.registerFont(TTFont(bold_name, str(bold_path), **bold_kwargs))
                except Exception:
                    bold_name = regular_name
            else:
                bold_name = regular_name
            pdfmetrics.registerFontFamily(regular_name, normal=regular_name, bold=bold_name,
                                          italic=regular_name, boldItalic=bold_name)
            return regular_name, bold_name
        except Exception:
            continue

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light", "STSong-Light"
    except Exception:
        logger.warning("STSong-Light is unavailable; using Helvetica for PDF text")
        return "Helvetica", "Helvetica"


def _pdf_footer(canvas: Any, doc: Any) -> None:
    """Brand + page number footer on every page."""
    from reportlab.lib.colors import HexColor

    canvas.saveState()
    width, _height = doc.pagesize
    canvas.setStrokeColor(HexColor(_PDF_BORDER))
    canvas.setLineWidth(0.6)
    canvas.line(18 * 2.83465, 14 * 2.83465 + 10, width - 18 * 2.83465, 14 * 2.83465 + 10)
    canvas.setFont(doc._pdf_font_regular, 7.5)
    canvas.setFillColor(HexColor(_PDF_MUTED))
    if canvas.getPageNumber() > 1:
        canvas.drawString(18 * 2.83465, _height - 12 * 2.83465,
                          getattr(doc, "_pdf_title", _PDF_BRAND))
        canvas.drawRightString(width - 18 * 2.83465, _height - 12 * 2.83465, "活动计划 / 资料附录")
    canvas.drawString(18 * 2.83465, 14 * 2.83465, _PDF_BRAND + " · 自动生成，仅供出行参考")
    canvas.drawRightString(width - 18 * 2.83465, 14 * 2.83465, f"第 {canvas.getPageNumber()} 页")
    canvas.restoreState()


#: The plan book's numbered sections, in order: ``(number, label, aliases)``.
#: ONE vocabulary for both stories on purpose — :func:`_text_story` matches a
#: Markdown heading by substring against these aliases, so renaming a label here
#: without keeping the old name as an alias silently files its content under the
#: previous section instead of erroring. The old names stay below as aliases.
PLAN_SECTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("01", "计划摘要", ("计划摘要", "计划概要", "本计划概要")),
    ("02", "计划信息", ("计划信息", "活动信息", "目的地", "基本信息")),
    ("03", "路线摘要", ("路线摘要", "路线概况", "路线规划", "路线信息")),
    ("04", "行程时间线", ("行程时间线", "行程安排", "日程", "行程", "时间安排")),
    ("05", "交通住宿与联络", ("交通住宿与联络", "交通", "住宿", "后勤安排")),
    ("06", "人员分工", ("人员分工", "队伍", "组织分工")),
    ("07", "费用预算", ("费用预算", "预算", "费用")),
    ("08", "装备与补给", ("装备与补给", "装备", "装备清单", "补给")),
    ("09", "风险决策区", ("风险决策区", "风险与应急", "风险", "风险预案", "安全", "应急")),
    ("10", "出发前检查", ("出发前检查", "检查清单", "注意事项", "建议")),
    ("11", "候选路线对比", ("候选路线对比", "候选对比", "候选路线")),
    ("12", "排序依据与说明", ("排序依据与说明", "排序说明", "排序依据")),
    ("13", "来源与证据", ("来源与证据", "来源", "参考资料")),
)

#: The (number, label) pair of one section, looked up BY NAME. Call sites name the
#: section they draw instead of indexing PLAN_SECTIONS, so inserting or reordering a
#: section cannot silently renumber every later heading.
def _plan_section(label: str) -> tuple[str, str]:
    for number, section_label, _aliases in PLAN_SECTIONS:
        if section_label == label:
            return number, section_label
    raise KeyError(label)


#: The copy an ordinary artifact export falls back to when the Markdown draft has
#: nothing under a heading (spec: 补齐缺失章节并标为草案, never invent).
_TEXT_SECTION_DEFAULTS: dict[str, str] = {
    "计划摘要": f"{UNCONFIRMED}：路线、日期与人数摘要需由已确认安排与核验资料组成。",
    "计划信息": f"{UNCONFIRMED}：活动日期、集合时间与地点、人数、活动目标。",
    "路线摘要": f"{UNVERIFIED}：起终点、里程、爬升、耗时、天气与开放状态。",
    "行程时间线": f"{UNCONFIRMED}：逐日安排、途经点、补给与最晚下撤时间。",
    "交通住宿与联络": f"{UNCONFIRMED}：往返交通、住宿、应急联络与撤退交通。",
    "人员分工": f"{UNCONFIRMED}：领队、导航、收队和急救联络负责人。",
    "费用预算": f"{UNCONFIRMED}：交通、住宿门票、餐食补给、保险与费用分摊。",
    "装备与补给": f"{UNCONFIRMED}：防滑鞋、雨具、保暖层、头灯、手机充电宝、急救包、饮水和食品。",
    "风险决策区": "通用建议：恶劣天气或封闭路线取消出行；身体不适或迷路时停止前进，在安全位置联系领队或当地救援。",
    "出发前检查": f"{UNCONFIRMED}：最新天气与开放状态、离线地图、装备自检和对外联络；垃圾全部带离。",
    "候选路线对比": f"{UNVERIFIED}：本草案没有经过服务器核验的候选路线对比表。",
    "排序依据与说明": f"{UNVERIFIED}：草案中的排序不来自证据校验，不得视为已核验结论。",
    "来源与证据": f"{UNVERIFIED}：本草案未完成路线证据校验；请补充可靠资料和查询时间。",
}


def _report_story(doc: Any, data: dict[str, Any]) -> list[Any]:
    """Render the structured report as a complete, evidence-aware plan book."""
    from reportlab.lib import colors
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import (HRFlowable, KeepTogether, PageBreak, Paragraph,
                                    Spacer, Table, TableStyle)
    from xml.sax.saxutils import escape

    regular, bold = doc._pdf_font_regular, doc._pdf_font_bold
    # ReportLab frames reserve six points on either side inside the margins.
    page_width = doc.width - 12
    mm = page_width / 174

    def style(name: str, **kw: Any) -> ParagraphStyle:
        kw.setdefault("fontName", regular)
        kw.setdefault("alignment", TA_LEFT)
        kw.setdefault("wordWrap", "CJK")
        return ParagraphStyle(name, **kw)

    title = style("PlanTitle", fontName=bold, fontSize=20, leading=25, textColor=colors.white)
    subtitle = style("PlanSubtitle", fontSize=10, leading=14, textColor=HexColor("#dcfce7"))
    meta = style("PlanMeta", fontSize=8.2, leading=12, textColor=HexColor("#dcfce7"))
    section = style("PlanSection", fontName=bold, fontSize=12.5, leading=16,
                    textColor=HexColor(_PDF_PRIMARY_DARK), spaceBefore=12, spaceAfter=4, keepWithNext=True)
    risk_section = style("PlanRiskSection", fontName=bold, fontSize=13.5, leading=17,
                         textColor=HexColor(_PDF_RISK), spaceBefore=14, spaceAfter=4, keepWithNext=True)
    body = style("PlanBody", fontSize=9.5, leading=14, textColor=HexColor(_PDF_TEXT))
    small = style("PlanSmall", fontSize=8.2, leading=12, textColor=HexColor(_PDF_MUTED))
    table_head = style("PlanTableHead", fontName=bold, fontSize=8.4, leading=11, textColor=colors.white)
    risk_head = style("PlanRiskHead", fontName=bold, fontSize=8.6, leading=11,
                      textColor=HexColor(_PDF_RISK_DARK))
    table_cell = style("PlanTableCell", fontSize=8.4, leading=12, textColor=HexColor(_PDF_TEXT))
    table_cell_bold = style("PlanTableCellBold", fontName=bold, fontSize=8.6, leading=12, textColor=HexColor(_PDF_TEXT))
    timeline_time = style("PlanTimelineTime", fontName=bold, fontSize=8.6, leading=12,
                          textColor=HexColor(_PDF_PRIMARY_DARK))
    status = style("PlanStatus", fontName=bold, fontSize=15, leading=19, textColor=HexColor(_PDF_PRIMARY))
    status_heading = style("PlanStatusHeading", fontName=bold, fontSize=12, leading=16,
                           textColor=HexColor(_PDF_PRIMARY_DARK))
    risk_title = style("PlanRiskTitle", fontName=bold, fontSize=10.5, leading=14,
                       textColor=HexColor(_PDF_RISK_DARK))
    risk_center = style("PlanRiskCenter", fontName=bold, fontSize=8.4, leading=11, alignment=TA_CENTER,
                        textColor=HexColor(_PDF_RISK_DARK))
    callout = style("PlanCallout", fontName=bold, fontSize=9, leading=13, textColor=HexColor(_PDF_AMBER_TEXT))
    center_cell = style("PlanCenterCell", fontSize=8.2, leading=11, alignment=TA_CENTER, textColor=HexColor(_PDF_TEXT))

    def p(value: Any, style_obj: ParagraphStyle = table_cell) -> Paragraph:
        text = escape(str(value if value not in (None, "") else UNVERIFIED)).replace("\n", "<br/>")
        return Paragraph(text, style_obj)

    def section_title(number: str, label: str, *, tone: str = "default") -> list[Any]:
        risk = tone == "risk"
        rule = HRFlowable(width="100%", thickness=1.4 if risk else 0.8,
                          color=HexColor(_PDF_RISK if risk else _PDF_PRIMARY), spaceAfter=6)
        rule.keepWithNext = True
        return [Paragraph(f"{number}  {escape(label)}", risk_section if risk else section), rule]

    def styled_table(rows: list[list[Any]], widths: list[float], *, header: bool = True,
                     highlight_rows: set[int] | None = None, tone: str = "default") -> Table:
        table = Table(rows, colWidths=[width * mm for width in widths], repeatRows=1 if header else 0,
                      splitByRow=1, splitInRow=1, hAlign="LEFT")
        risk = tone == "risk"
        commands: list[tuple[Any, ...]] = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.45, HexColor(_PDF_BORDER)),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
        if risk:
            commands.append(("BOX", (0, 0), (-1, -1), 1.1, HexColor(_PDF_RISK)))
        if header:
            commands.append(("BACKGROUND", (0, 0), (-1, 0),
                             HexColor(_PDF_RISK_BG if risk else _PDF_PRIMARY)))
        for index in range(1 if header else 0, len(rows)):
            if highlight_rows and index in highlight_rows:
                commands.append(("BACKGROUND", (0, index), (-1, index), HexColor(_PDF_PRIMARY_BG)))
            elif risk:
                commands.append(("BACKGROUND", (0, index), (-1, index), HexColor(_PDF_RISK_BG)))
            elif index % 2 == 0:
                commands.append(("BACKGROUND", (0, index), (-1, index), HexColor(_PDF_ZEBRA)))
        table.setStyle(TableStyle(commands))
        return table

    story: list[Any] = []
    cover = data.get("cover") or {}
    status_lines = cover.get("status_lines") or [
        {"label": "计划状态", "value": str(data.get("status_label") or UNVERIFIED)}]
    generated = "生成时间：" + str(data.get("generated_at", UNVERIFIED))
    cid = str(data.get("comparison_id", ""))
    cid_text = cid[:44] + ("…" if len(cid) > 44 else "")

    # ---- Cover page (page 1, nothing else on it) -------------------------
    cover_head = Table([[Paragraph(escape(str(cover.get("kicker", data.get("title", "徒步计划书")))), meta)],
                        [Paragraph(escape(data.get("title", "AI Hiking 徒步计划书")), title)],
                        [Paragraph(escape(data.get("subtitle", "证据核验版徒步计划书")), subtitle)]],
                       colWidths=[page_width])
    cover_head.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor(_PDF_PRIMARY_DARK)),
        ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, 0), 16), ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("TOPPADDING", (0, 1), (-1, 1), 2), ("BOTTOMPADDING", (0, 1), (-1, 1), 4),
        ("TOPPADDING", (0, 2), (-1, 2), 2), ("BOTTOMPADDING", (0, 2), (-1, 2), 16),
    ]))
    status_rows: list[list[Any]] = [[Paragraph(escape(str(cover.get("heading", "计划状态"))), status_heading)]]
    status_rows += [[Paragraph(escape(str(line.get("label", "")) + "："), table_cell_bold),
                     p(line.get("value", ""))] for line in status_lines]
    status_panel = Table(status_rows, colWidths=[46 * mm, 128 * mm], hAlign="LEFT")
    status_panel.setStyle(TableStyle([
        ("SPAN", (0, 0), (1, 0)),
        ("BOX", (0, 0), (-1, -1), 0.75, HexColor(_PDF_PRIMARY)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, HexColor(_PDF_BORDER)),
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(_PDF_PRIMARY_BG)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([Spacer(1, 26), cover_head, Spacer(1, 14), status_panel])
    if data.get("offline_disclaimer"):
        notice = Table([[Paragraph("离线演示：合成数据，不可用于真实出行。", callout)]], colWidths=[page_width])
        notice.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), HexColor(_PDF_AMBER_BG)),
                                    ("BOX", (0, 0), (-1, -1), 0.65, HexColor(_PDF_AMBER_BORDER)),
                                    ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                                    ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        story.extend([Spacer(1, 10), notice])
    policy = data.get("content_policy") or {}
    story.append(Spacer(1, 18))
    for key in ("facts", "advice"):
        if policy.get(key):
            story.append(Paragraph("·　" + escape(str(policy[key])), small))
    story.append(Spacer(1, 8))
    story.append(Paragraph("签名确认：________________　　确认日期：________________", body))
    story.append(PageBreak())

    # ---- 01 计划摘要 -----------------------------------------------------
    story.extend(section_title(*_plan_section("计划摘要")))
    summary_data = data.get("summary") or {}
    headline = str(summary_data.get("headline") or "").strip()
    if headline:
        story.append(Paragraph(escape(headline), status))
    summary_lines = summary_data.get("lines") or []
    summary_pairs: list[list[Any]] = []
    for index in range(0, len(summary_lines), 2):
        left = summary_lines[index]
        right = summary_lines[index + 1] if index + 1 < len(summary_lines) else {"label": "", "value": ""}
        summary_pairs.append([p(left.get("label", ""), table_cell_bold), p(left.get("value", "")),
                              p(right.get("label", ""), table_cell_bold), p(right.get("value", ""))])
    if summary_pairs:
        story.append(styled_table(summary_pairs, [28, 59, 28, 59], header=False))
        story.append(Spacer(1, 6))
    recommendation = str(data.get("recommended") or "暂未形成可执行推荐")
    decision_rows: list[list[Any]] = [[
        Paragraph(escape(str(data.get("status_label", UNVERIFIED))), status),
        p(f"推荐路线：{recommendation}\n活动安排需逐项确认；路线状态需在出发前重新核验。", body),
    ]]
    if data.get("alternatives"):
        decision_rows.append([Paragraph("备选路线", table_cell_bold),
                              p("、".join(map(str, data["alternatives"])), body)])
    decision = Table(decision_rows, colWidths=[48 * mm, 126 * mm], hAlign="LEFT")
    decision.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.75, HexColor(_PDF_BORDER)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), HexColor(_PDF_PRIMARY_BG)),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(decision)

    # ---- 02 计划信息 + 03 路线摘要 --------------------------------------
    story.extend(section_title(*_plan_section("计划信息")))
    metadata = data.get("plan_metadata", [])
    metadata_rows: list[list[Any]] = []
    for index in range(0, len(metadata), 2):
        left = metadata[index]
        right = metadata[index + 1] if index + 1 < len(metadata) else {"label": "", "value": ""}
        metadata_rows.append([p(left.get("label", ""), table_cell_bold), p(left.get("value", "")),
                              p(right.get("label", ""), table_cell_bold), p(right.get("value", ""))])
    if metadata_rows:
        story.append(styled_table(metadata_rows, [28, 59, 28, 59], header=False))
    story.append(Spacer(1, 4))
    story.extend(section_title(*_plan_section("路线摘要")))
    overview = data.get("route_overview", [])
    overview_rows: list[list[Any]] = []
    for index in range(0, len(overview), 2):
        left = overview[index]
        right = overview[index + 1] if index + 1 < len(overview) else {"label": "", "value": ""}
        overview_rows.append([p(left.get("label", ""), table_cell_bold), p(left.get("value", "")),
                              p(right.get("label", ""), table_cell_bold), p(right.get("value", ""))])
    if overview_rows:
        story.append(styled_table(overview_rows, [28, 59, 28, 59], header=False))
    story.append(Spacer(1, 2))
    story.append(Paragraph(f"事实字段来自路线证据；显示“{UNVERIFIED}”的字段不得在出发前视为已确认。", small))

    # ---- 04 行程时间线 ---------------------------------------------------
    story.extend(section_title(*_plan_section("行程时间线")))
    timeline_note = ("以下按用户提供的安排成时间线，执行前需确认。" if data.get("user_itinerary") else
                     f"以下为通用行程模板，日期、时间与途经点{UNCONFIRMED}；不代表已核验的分段行程。")
    story.append(Paragraph(timeline_note, small))
    story[-1].keepWithNext = True
    timeline = [[p("时间", table_head), p("阶段", table_head),
                 p("执行重点", table_head), p("完成检查", table_head)]]
    timeline += [[p(row.get("time"), timeline_time), p(row.get("stage"), table_cell_bold),
                  p(row.get("focus")), p(row.get("checkpoint"))]
                 for row in data.get("timeline", [])]
    timeline_table = styled_table(timeline, [26, 40, 62, 46])
    # The axis rule down the left of the stage column is what makes this read as
    # a timeline rather than as another four-column grid.
    timeline_table.setStyle(TableStyle([("LINEBEFORE", (1, 1), (1, -1), 1.4, HexColor(_PDF_PRIMARY))]))
    story.append(timeline_table)

    story.extend(section_title(*_plan_section("交通住宿与联络")))
    logistics = [[p(row.get("label"), table_cell_bold), p(row.get("value"))]
                 for row in data.get("logistics", [])]
    if logistics:
        story.append(styled_table(logistics, [35, 139], header=False))
    story.extend(section_title(*_plan_section("人员分工")))
    roles = [[p("岗位", table_head), p("人员", table_head), p("职责建议", table_head)]]
    roles += [[p(row.get("role")), p(row.get("person")), p(row.get("duty"))] for row in data.get("roles", [])]
    story.append(styled_table(roles, [30, 40, 104]))
    story.extend(section_title(*_plan_section("费用预算")))
    budget = [[p("费用项目", table_head), p("金额", table_head), p("依据", table_head)]]
    budget += [[p(row.get("item")), p(row.get("amount")), p(row.get("basis"))] for row in data.get("budget", [])]
    note = str(data.get("budget_note") or "").strip()
    if note and note != UNCONFIRMED:
        # The user's own wording, as text, beside an amount that stays
        # :data:`UNCONFIRMED` — free text is never parsed into a number here. The
        # guard compares against the SAME constant the document builder writes, so
        # rewording the marker moves both sides together instead of turning a blank
        # budget into a 用户预算说明 row.
        budget.append([p("用户预算说明"), p(UNCONFIRMED), p("用户原文")])
    story.append(styled_table(budget, [75, 50, 49]))
    story.extend([Spacer(1, 4),
                  p("已列预算合计：" + str(data.get("budget_total", UNCONFIRMED)), body)])
    if note and note != UNCONFIRMED:
        story.append(p("预算原文：" + note, small))

    # ---- 08 装备与补给 ---------------------------------------------------
    story.extend(section_title(*_plan_section("装备与补给")))
    gear = [[p("类别", table_head), p("物品", table_head), p("优先级", table_head), p("出发前状态", table_head)]]
    gear += [[p(row.get("category"), table_cell_bold), p(row.get("item")), p(row.get("priority"), center_cell),
              p(row.get("status"), center_cell)] for row in data.get("equipment", [])]
    story.append(styled_table(gear, [25, 77, 30, 42]))

    # ---- 09 风险决策区 ---------------------------------------------------
    risk = data.get("risk") or {}
    risk_rows = [[p("触发条件", risk_head), p("处置动作", risk_head), p("级别", risk_head)]]
    risk_rows += [[p(row.get("trigger")), p(row.get("action")), p(row.get("severity"), risk_center)]
                  for row in risk.get("rows", data.get("risk_controls", []))]
    risk_table = styled_table(risk_rows, [53, 82, 39], header=True, tone="risk")
    risk_block: list[Any] = [Paragraph(
        "风险状态：" + escape(str(risk.get("status_label", data.get("status_label", UNVERIFIED)))), risk_title)]
    if risk.get("note"):
        risk_block.append(Paragraph(escape(str(risk["note"])), small))
    risk_block.append(risk_table)
    contacts = risk.get("contacts") or []
    if contacts:
        risk_block.append(Spacer(1, 3))
        risk_block.append(styled_table(
            [[p(str(row.get("label", "")), table_cell_bold), p(row.get("value", ""))] for row in contacts],
            [35, 139], header=False))
    story.append(KeepTogether(section_title(*_plan_section("风险决策区"), tone="risk") + risk_block))
    required = str(data.get("required_action") or "").strip()
    if required and required != UNCONFIRMED:
        action_table = Table([[Paragraph("出发前必须完成：" + escape(required), callout)]], colWidths=[page_width])
        action_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), HexColor(_PDF_AMBER_BG)),
                                          ("BOX", (0, 0), (-1, -1), 0.65, HexColor(_PDF_AMBER_BORDER)),
                                          ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                                          ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        story.extend([Spacer(1, 6), action_table])

    # ---- 10 出发前检查 ---------------------------------------------------
    story.extend(section_title(*_plan_section("出发前检查")))
    checklist = [[p("检查项", table_head), p("负责人", table_head), p("状态", table_head)]]
    checklist += [[p(row.get("item")), p(row.get("owner"), center_cell), p(row.get("status"), center_cell)]
                  for row in data.get("checklist", [])]
    story.append(styled_table(checklist, [107, 31, 36]))

    # ---- 11 候选路线对比 -------------------------------------------------
    rows = data.get("rows", [])
    if rows:
        story.extend(section_title(*_plan_section("候选路线对比")))
        # Use one route per row and short labels so nine columns remain legible.
        columns = ["路线"] + [str(c).strip() for c in data.get("columns", [])] + ["资料", "风险"]
        table_data: list[list[Any]] = [[p(c, table_head) for c in columns]]
        for row in rows:
            name = str(row.get("name", UNVERIFIED)) + (" ★" if row.get("badge") == "推荐" else "")
            table_data.append([p(name, table_cell_bold)] + [p(cell) for cell in row.get("cells", [])] +
                              [p(row.get("evidence")), p(row.get("risk"))])
        remaining = 174 - 34
        col_widths = [34] + [remaining / max(1, len(columns) - 1)] * (len(columns) - 1)
        story.append(styled_table(table_data, col_widths, highlight_rows={i for i, row in enumerate(rows, 1) if row.get("highlight")}))

    notes = [str(n) for n in data.get("notes", []) if n]
    story.extend(section_title(*_plan_section("排序依据与说明")))
    if notes:
        for note in notes:
            story.append(Paragraph("•　" + escape(note), body))
            story.append(Spacer(1, 2))
    else:
        story.append(p(f"仅对证据齐全且通过安全校验的候选排序；具体偏好说明{UNCONFIRMED}。", body))

    # Sources are retained as an appendix and can split across pages.
    source_groups = data.get("sources", [])
    if source_groups:
        story.extend(section_title(*_plan_section("来源与证据")))
        for group in source_groups:
            story.append(Paragraph(escape("来源 — " + str(group.get("candidate", UNVERIFIED))), table_cell_bold))
            for entry in group.get("entries", []):
                url = str(entry.get("url", ""))
                header = f"[{escape(str(entry.get('id', '')))}] {escape(str(entry.get('tool', '')))}"
                if url:
                    header += f'　<font color="{_PDF_LINK}">{escape(url)}</font>'
                source_card = Table([[Paragraph(header, small)],
                                     [Paragraph(escape(str(entry.get("excerpt", UNVERIFIED))), small)],
                                     [Paragraph("观测时间：" + escape(str(entry.get("observed_at", UNVERIFIED))), small)]],
                                    colWidths=[page_width], splitInRow=1)
                source_card.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.45, HexColor(_PDF_BORDER)),
                                                 ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                                                 ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
                story.extend([source_card, Spacer(1, 4)])
    story.append(Spacer(1, 5))
    story.append(Paragraph("内容规范：事实仅来自证据；建议项需由领队在出发前逐项确认。", small))
    story.append(Paragraph("领队确认：________________    确认日期：________________", small))
    return story


def _text_story(doc: Any, content: str) -> list[Any]:
    """Parse Markdown into print elements and retain missing sections as open items."""
    import mistune
    from reportlab.lib import colors
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import PageBreak, Paragraph, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    regular, bold = doc._pdf_font_regular, doc._pdf_font_bold

    def style(name: str, **kw: Any) -> ParagraphStyle:
        return ParagraphStyle(name, alignment=TA_LEFT, wordWrap="CJK", **kw)

    h1 = style("H1", fontName=bold, fontSize=16, leading=22, textColor=colors.white, spaceBefore=6, spaceAfter=4)
    h2 = style("H2", fontName=bold, fontSize=13, leading=18, textColor=HexColor(_PDF_PRIMARY_DARK), spaceBefore=10, spaceAfter=5, keepWithNext=True)
    h3 = style("H3", fontName=bold, fontSize=11.5, leading=16, spaceBefore=6, spaceAfter=2, keepWithNext=True)
    body = style("Body", fontName=regular, fontSize=10, leading=16, textColor=HexColor(_PDF_TEXT))
    bullet = style("Bullet", fontName=regular, fontSize=10, leading=16, leftIndent=12, textColor=HexColor(_PDF_TEXT))
    meta = style("DraftMeta", fontName=regular, fontSize=8.6, leading=12, textColor=HexColor("#dcfce7"))
    table_cell_bold = style("DraftCellBold", fontName=bold, fontSize=9, leading=13,
                            textColor=HexColor(_PDF_TEXT))

    def inline(nodes: list[dict[str, Any]], *, markup: bool = True) -> str:
        parts = []
        for node in nodes:
            kind = node["type"]
            text = escape(node.get("raw", "")) if markup else node.get("raw", "")
            if "children" in node:
                text = inline(node["children"], markup=markup)
            if markup and kind in {"strong", "emphasis"}:
                tag = "b" if kind == "strong" else "i"
                text = f"<{tag}>{text}</{tag}>"
            if kind in {"softbreak", "linebreak"}:
                text = "<br/>" if markup else " "
            parts.append(text)
        return "".join(parts)

    def blocks(nodes: list[dict[str, Any]]) -> list[Any]:
        rendered = []
        for node in nodes:
            kind = node["type"]
            children = node.get("children", [])
            if kind == "heading":
                rendered.append(Paragraph(inline(children), h3))
                rendered[-1].keepWithNext = True
            elif kind in {"paragraph", "block_text"}:
                rendered.append(Paragraph(inline(children) or "&nbsp;", body))
            elif kind == "list":
                for index, item in enumerate(children, 1):
                    label = f"{index}. " if node.get("attrs", {}).get("ordered") else "•　"
                    item_nodes = item.get("children", [])
                    if not item_nodes:
                        continue
                    first, *rest = item_nodes
                    rendered.append(Paragraph(label + inline(first.get("children", [])), bullet))
                    rendered.extend(blocks(rest))
            elif kind == "table":
                head, table_body = children
                rows = [head["children"]] + [row["children"] for row in table_body["children"]]
                cell_style = style("MarkdownCell", fontName=regular, fontSize=9, leading=13)
                cells = [[Paragraph(inline(cell.get("children", [])), cell_style) for cell in row] for row in rows]
                table = Table(cells, colWidths=[(doc.width - 12) / len(rows[0])] * len(rows[0]),
                              repeatRows=1, splitInRow=1)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor(_PDF_PRIMARY_BG)),
                    ("GRID", (0, 0), (-1, -1), 0.45, HexColor(_PDF_BORDER)),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                rendered.append(table)
            elif children:
                rendered.extend(blocks(children))
            elif node.get("raw"):
                rendered.append(Paragraph(escape(node["raw"]).replace("\n", "<br/>"), body))
            if kind != "blank_line":
                rendered.append(Spacer(1, 4))
        return rendered

    # One vocabulary for both stories: the numbered headings and the aliases a
    # Markdown heading is matched against come from PLAN_SECTIONS, so a rename in
    # the plan book cannot leave the text draft matching the OLD name only.
    sections = [(number, label, aliases, _TEXT_SECTION_DEFAULTS[label])
                for number, label, aliases in PLAN_SECTIONS]
    grouped: dict[str, list[dict[str, Any]]] = {label: [] for _n, label, _, _ in sections}
    current = sections[0][1]
    for node in mistune.create_markdown(renderer="ast", plugins=["table"])(content):
        if node["type"] == "heading":
            heading = inline(node.get("children", []), markup=False)
            if node.get("attrs", {}).get("level") == 1 and "计划" in heading:
                continue
            match = next((label for _n, label, aliases, _d in sections if any(
                name in heading for name in (label, *aliases))), None)
            if match:
                current = match
                continue
        grouped[current].append(node)

    # Cover page: an ordinary artifact export has no route evidence behind it, so
    # its status line says 草案 and :data:`UNVERIFIED` rather than claiming a
    # verified plan.
    cover_rows = [
        ("草案类型", f"通用草案 · 路线事实{UNVERIFIED}"),
        ("审阅结论", UNVERIFIED),
        ("安全门", UNVERIFIED),
    ]
    cover_panel = Table([[Paragraph("徒步计划书 · Hiking Plan Book", meta)],
                         [Paragraph("AI Hiking 徒步计划书", h1)],
                         [Paragraph(f"活动安排草案 · 路线事实尚{UNVERIFIED}", meta)]]
                        + [[Paragraph(escape(label), table_cell_bold), Paragraph(value, body)]
                           for label, value in cover_rows],
                        colWidths=[46 * 2.83465, doc.width - 12 - 46 * 2.83465])
    cover_panel.setStyle(TableStyle([
        ("SPAN", (0, 0), (-1, 0)), ("SPAN", (0, 1), (-1, 1)), ("SPAN", (0, 2), (-1, 2)),
        ("BACKGROUND", (0, 0), (-1, 2), HexColor(_PDF_PRIMARY_DARK)),
        ("TEXTCOLOR", (0, 1), (-1, 2), colors.white),
        ("BOX", (3, 3), (-1, -1), 0.75, HexColor(_PDF_PRIMARY)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story = [Spacer(1, 26), cover_panel, PageBreak()]
    for number, label, _aliases, default in sections:
        story.append(Paragraph(f"{number}  {label}", h2))
        section_blocks = blocks(grouped[label])
        story.extend(section_blocks or [Paragraph(default, body), Spacer(1, 4)])
    return story


def _render_pdf_bytes(content: str) -> bytes:
    """Render text or a structured document into a valid PDF.

    Structured route-comparison reports (JSON with ``kind=route_comparison_report``)
    get a designed layout with real heading hierarchy, a comparison table and
    source cards; any other content falls back to markdown-flavoured text.
    ReportLab keeps the development path deterministic and produces bytes that
    PDF readers and downstream artifact checks can actually parse.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate
    except ImportError as exc:
        raise ComputerError(
            "PDF generation requires reportlab; install ai-service/requirements.txt"
        ) from exc

    regular, bold = _register_pdf_font()
    document_data: dict[str, Any] | None = None
    if content.lstrip().startswith("{"):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and parsed.get("kind") == "route_comparison_report":
                document_data = parsed
        except ValueError:
            document_data = None

    output = BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title=document_data.get("title", "AI Hiking 徒步计划书") if document_data else "AI Hiking 徒步计划书",
        author=_PDF_BRAND,
    )
    doc._pdf_font_regular = regular
    doc._pdf_font_bold = bold
    doc._pdf_title = doc.title
    if document_data is not None:
        story = _report_story(doc, document_data)
    else:
        story = _text_story(doc, content)
    doc.build(story, onFirstPage=_pdf_footer, onLaterPages=_pdf_footer)
    return output.getvalue()


# ---- Failure modes (injectable for tests + future E2B mapping) ------------


class ComputerError(Exception):
    """Base for all sandbox failures. Fail-closed: never fall back to host."""


class SandboxCreationError(ComputerError):
    """Sandbox could not be created (e.g. E2B API error, disk full)."""


class SandboxUnavailableError(ComputerError):
    """The selected sandbox backend is a placeholder that refuses to run.

    Ticket 01 (harness-p0-p1): ``remote_e2b`` has no live client yet. The
    placeholder raises this error BEFORE any host workspace, process, or
    file operation — it never mimics the E2B lifecycle on the host, and
    callers must never fall back to a host computer. The public mapping
    is the stable error code ``sandbox_unavailable`` with exactly one
    terminal event per stream; nothing is executed, so no effect is
    committed and no artifact is produced.
    """

    # The ONE source of truth for the public error code: direct adapter
    # calls, the kernel-facing Bridge (which re-raises it as a CLIError
    # carrying this code) and the run orchestration all map to it.
    code = "sandbox_unavailable"


class SandboxQuotaError(ComputerError):
    """Quota exhausted — no new sandboxes can be provisioned."""


class SandboxTimeoutError(ComputerError):
    """A command inside the sandbox timed out."""


class SandboxGenerationError(ComputerError):
    """Artifact generation inside the sandbox failed (Ticket 04).

    Raised when the trusted-local sandbox cannot produce a real artifact
    (e.g. blank body, missing rendering dependency). The caller maps it to
    a terminal failure — no pseudo artifact is ever published.
    """


class SandboxFileNotFoundError(ComputerError):
    """A sandbox file was expected but is missing (e.g. manifest build
    or read_artifact_bytes on a non-existent path). Distinct from
    :class:`SandboxTimeoutError` so callers can classify failures."""


class SandboxBashOutputLimitExceeded(ComputerError):
    """A sandbox bash command produced more output than the configured
    cap (Ticket 13 AC3: 输出上限). The partial output captured before
    the limit is exposed via ``partial_output`` so the audit log can
    record what the command printed before being cut off."""

    def __init__(self, message: str, partial_output: str = "") -> None:
        super().__init__(message)
        self.partial_output = partial_output


class SandboxCleanupError(ComputerError):
    """Cleanup failed — the lease is marked ``failed`` but the run continues
    to its terminal state. The operator must investigate stale sandboxes.

    Carries ``partial_usage`` (best-effort :class:`ResourceUsage` gathered
    before the cleanup failure) so the lease can record resource usage
    even when destruction failed (spec AC6: record success, failure, AND
    resource usage).
    """

    def __init__(self, message: str, partial_usage: "ResourceUsage | None" = None) -> None:
        super().__init__(message)
        self.partial_usage = partial_usage


# ---- Resource usage reported at cleanup -----------------------------------


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Resource metrics gathered at sandbox cleanup (spec AC6).

    All fields are best-effort; the stub reports synthetic values. Real
    E2B integration replaces these with ``usage`` from the E2B API.
    """

    duration_ms: int
    files_written: int
    bytes_written: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "files_written": self.files_written,
            "bytes_written": self.bytes_written,
        }


# ---- Bash result (Ticket 13) ---------------------------------------------


@dataclass(frozen=True, slots=True)
class BashResult:
    """Result of a sandbox bash command (Ticket 13).

    All fields are deterministic so the effect ledger can store the
    result and the resume flow can replay it without re-executing the
    command (spec AC2: 把已有成功结果作为唯一结果返回).

    ``stdout`` and ``stderr`` are truncated to ``max_output_bytes`` per
    stream; the ``truncated`` flag is set when truncation occurred so
    the audit log and the user-facing event can surface it.
    """

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "truncated": self.truncated,
        }


# ---- Computer interface ---------------------------------------------------


class Computer:
    """Abstract sandbox lifecycle + artifact operations.

    Implementations must be fail-closed: any failure raises a
    :class:`ComputerError` subclass. The caller (run stream) maps the
    failure to an ``error`` terminal event; it NEVER retries on the host.
    """

    sandbox_type: SandboxType

    def create_workspace(self, run_id: str) -> str:
        """Provision a workspace for ``run_id`` and return its path.

        Raises :class:`SandboxCreationError` or :class:`SandboxQuotaError`
        on failure. The returned path is opaque to callers — they pass
        it back to ``generate_pdf`` / ``cleanup``.
        """
        raise NotImplementedError

    def generate_pdf(
        self,
        workspace_path: str,
        content: str,
        filename: str,
    ) -> str:
        """Write ``content`` as ``filename`` inside the sandbox workspace.

        Returns the relative path of the generated file (no host prefix).
        Raises :class:`SandboxTimeoutError` if the write exceeds the
        sandbox command timeout.
        """
        raise NotImplementedError

    def build_manifest(
        self,
        workspace_path: str,
        relative_path: str,
        declared_mime: str,
    ) -> ArtifactManifest:
        """Build a sandbox-side manifest for the produced artifact (Ticket 10).

        The sandbox declares the file's relative path, public filename,
        MIME type, size, and SHA-256. The host re-derives size + SHA-256
        from the bytes returned by :meth:`read_artifact_bytes` and
        rejects any mismatch.

        Raises :class:`SandboxTimeoutError` if the checksum command
        exceeds the sandbox timeout.
        """
        raise NotImplementedError

    def read_artifact_bytes(
        self,
        workspace_path: str,
        relative_path: str,
    ) -> bytes:
        """Read the produced artifact's bytes back to the host (Ticket 10).

        The host uses these bytes to re-verify the manifest. The sandbox
        must NOT mutate the file between :meth:`generate_pdf` and this
        call.

        Raises :class:`SandboxTimeoutError` if the read exceeds the
        sandbox timeout.
        """
        raise NotImplementedError

    def run_bash(
        self,
        workspace_path: str,
        command: str,
        *,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 1_048_576,
    ) -> "BashResult":
        """Run a bash command inside the sandbox workspace (Ticket 13).

        Spec AC3: all Bash calls enforce a timeout and an output cap.
        Spec AC4: the command runs with ``cwd=workspace_path`` and an
        EMPTY environment (``env={}``) so host paths, host environment
        variables, and host processes are NOT accessible from inside
        the sandbox. The only path the command can read or write is the
        workspace.

        Args:
            workspace_path: opaque sandbox workspace handle returned by
                :meth:`create_workspace`.
            command: shell command string. Executed via ``bash -c`` on
                the sandbox; no shell expansion is performed by the
                host.
            timeout_seconds: wall-clock cap. Exceeding this raises
                :class:`SandboxTimeoutError`.
            max_output_bytes: per-stream (stdout/stderr) byte cap. The
                captured bytes are truncated to this length and the
                result's ``truncated`` flag is set; the run continues
                rather than failing (the user sees the partial output).

        Returns:
            :class:`BashResult` with exit code, truncated stdout/stderr,
            duration, and a ``truncated`` flag.

        Raises:
            :class:`SandboxTimeoutError`: command exceeded timeout.
            :class:`ComputerError`: sandbox-side execution failure.
        """
        raise NotImplementedError

    def cleanup(self, workspace_path: str) -> ResourceUsage:
        """Destroy the sandbox and return resource usage (spec AC6).

        Raises :class:`SandboxCleanupError` if destruction fails. The
        caller records the failure on the lease but still completes the
        run's terminal transition.
        """
        raise NotImplementedError


# ---- LocalNative (development only) ---------------------------------------


class LocalNativeComputer(Computer):
    """Development sandbox backed by a host temp directory.

    Spec decision #19 allows LocalNative in development. This impl is
    NOT a production fallback — production uses :class:`RemoteE2BComputer`.
    The temp directory is removed on ``cleanup``.
    """

    sandbox_type: SandboxType = "local_native"

    def __init__(self, failure_mode: str | None = None) -> None:
        # ``failure_mode`` lets tests inject failures without monkeypatching.
        # None = no injected failure. Values: "creation", "quota", "timeout",
        # "cleanup". Production code never sets this.
        self._failure_mode = failure_mode
        self._files_written: int = 0
        self._bytes_written: int = 0
        self._created_at: float | None = None

    def create_workspace(self, run_id: str) -> str:
        if self._failure_mode == "creation":
            raise SandboxCreationError("injected creation failure")
        if self._failure_mode == "quota":
            raise SandboxQuotaError("injected quota exhaustion")
        # mkdtemp gives us a host-backed workspace for development. The
        # path is opaque to callers; they never treat it as a host path.
        # Wrap OSError (disk full / permissions) as SandboxCreationError
        # so the fail-closed branch in _stream_artifact_flow maps it to
        # ``sandbox_creation_failed`` instead of a generic internal_error.
        try:
            workspace = tempfile.mkdtemp(prefix=f"v2-sandbox-{run_id}-")
        except OSError as exc:
            raise SandboxCreationError(f"workspace creation failed: {exc}") from exc
        self._created_at = time.monotonic()
        logger.info("LocalNative sandbox created: %s", workspace)
        return workspace

    def generate_pdf(
        self,
        workspace_path: str,
        content: str,
        filename: str,
    ) -> str:
        if self._failure_mode == "timeout":
            raise SandboxTimeoutError("injected command timeout")
        # Ticket 04: real PDF bytes via the shared generic renderer. A
        # blank body or a missing rendering dependency raises
        # PDFRenderingError → mapped by the caller to a terminal failure;
        # a text file with a .pdf suffix is never produced again.
        from agent_v2.pdf_rendering import (
            PDFRenderingError,
            render_markdown_pdf_bytes,
        )

        safe_name = Path(filename).name  # strip any path components
        if not safe_name.endswith(".pdf"):
            safe_name = f"{safe_name}.pdf"
        try:
            document = json.loads(content) if content.lstrip().startswith("{") else None
            if isinstance(document, dict) and document.get("kind") == "route_comparison_report":
                data = _render_pdf_bytes(content)
            else:
                data = render_markdown_pdf_bytes(content, title=Path(safe_name).stem)
        except ValueError:
            data = render_markdown_pdf_bytes(content, title=Path(safe_name).stem)
        except PDFRenderingError as exc:
            raise SandboxGenerationError(f"PDF 渲染失败：{exc}") from exc
        full = Path(workspace_path) / safe_name
        full.write_bytes(data)
        self._files_written += 1
        self._bytes_written += len(data)
        return safe_name

    def build_manifest(
        self,
        workspace_path: str,
        relative_path: str,
        declared_mime: str,
    ) -> ArtifactManifest:
        if self._failure_mode == "timeout":
            raise SandboxTimeoutError("injected command timeout")
        # Sandbox computes its own size + sha256. The host re-derives
        # both from read_artifact_bytes and rejects any mismatch — the
        # sandbox's values are declarative, not authoritative.
        full = Path(workspace_path) / relative_path
        try:
            data = full.read_bytes()
        except FileNotFoundError as exc:
            raise SandboxFileNotFoundError(
                f"manifest read failed for {relative_path!r}: file not found"
            ) from exc
        except OSError as exc:
            raise SandboxTimeoutError(
                f"manifest read failed for {relative_path!r}: {exc}"
            ) from exc
        return ArtifactManifest(
            relative_path=relative_path,
            public_filename=Path(relative_path).name,
            declared_mime=declared_mime,
            size_bytes=len(data),
            sha256=compute_sha256(data),
        )

    def read_artifact_bytes(
        self,
        workspace_path: str,
        relative_path: str,
    ) -> bytes:
        if self._failure_mode == "timeout":
            raise SandboxTimeoutError("injected command timeout")
        full = Path(workspace_path) / relative_path
        try:
            return full.read_bytes()
        except FileNotFoundError as exc:
            raise SandboxFileNotFoundError(
                f"artifact read failed for {relative_path!r}: file not found"
            ) from exc
        except OSError as exc:
            raise SandboxTimeoutError(
                f"artifact read failed for {relative_path!r}: {exc}"
            ) from exc

    def run_bash(
        self,
        workspace_path: str,
        command: str,
        *,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 1_048_576,
    ) -> "BashResult":
        """Run ``command`` inside the LocalNative sandbox workspace.

        Ticket 13 AC4 isolation guarantees:

        * ``cwd=workspace_path`` — the command's working directory is
          the sandbox workspace; relative paths resolve there.
        * ``env={}`` — the host environment is NOT inherited. ``HOME``
          points at the workspace and ``PATH`` is empty so
          shells that expect ``$HOME`` (e.g. ``bash -c`` reading
          ``~/.bashrc``) do not crash while commands cannot resolve host
          binaries. Cowork commands must use shell built-ins.
        * Subprocess is invoked with ``shell=False`` and explicit
          ``["bash", "-c", command]`` so the host does not interpret
          metacharacters.

        Failure injection: ``failure_mode="timeout"`` raises
        :class:`SandboxTimeoutError` without running the command; this
        lets tests exercise the timeout path deterministically.
        """
        if self._failure_mode == "timeout":
            raise SandboxTimeoutError("injected bash timeout")
        # Local import keeps the module header lean.
        import subprocess

        start = time.monotonic()
        # Empty env + HOME=workspace enforces AC4: no host env leak.
        sandbox_env = {"HOME": workspace_path, "PATH": ""}
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=workspace_path,
                env=sandbox_env,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxTimeoutError(
                f"bash command exceeded {timeout_seconds}s timeout"
            ) from exc
        except OSError as exc:
            raise ComputerError(f"bash execution failed: {exc}") from exc

        duration_ms = int((time.monotonic() - start) * 1000)
        stdout_bytes = proc.stdout or b""
        stderr_bytes = proc.stderr or b""
        truncated = (
            len(stdout_bytes) > max_output_bytes
            or len(stderr_bytes) > max_output_bytes
        )
        stdout_text = stdout_bytes[:max_output_bytes].decode(
            "utf-8", errors="replace"
        )
        stderr_text = stderr_bytes[:max_output_bytes].decode(
            "utf-8", errors="replace"
        )
        return BashResult(
            exit_code=proc.returncode,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_ms=duration_ms,
            truncated=truncated,
        )

    def cleanup(self, workspace_path: str) -> ResourceUsage:
        duration_ms = int((time.monotonic() - (self._created_at or 0)) * 1000)
        usage = ResourceUsage(
            duration_ms=duration_ms,
            files_written=self._files_written,
            bytes_written=self._bytes_written,
        )
        if self._failure_mode == "cleanup":
            # Don't delete the dir; raise so the lease is marked failed.
            # Attach partial_usage so AC6 records resource usage even on
            # cleanup failure.
            raise SandboxCleanupError(
                "injected cleanup failure", partial_usage=usage
            )
        try:
            shutil.rmtree(workspace_path, ignore_errors=False)
        except OSError as exc:
            raise SandboxCleanupError(str(exc), partial_usage=usage) from exc
        logger.info("LocalNative sandbox destroyed: %s", workspace_path)
        return usage


# ---- RemoteE2B (production; placeholder until a live client lands) ------------

_REFUSAL_MESSAGE = (
    "远程沙箱（remote_e2b）尚未接入：已拒绝执行。"
    "未创建宿主工作区、未启动任何进程、未写入任何文件；"
    "基础聊天与明确配置的受信 local_native 不受影响。"
)


class RemoteE2BComputer(Computer):
    """Production sandbox placeholder that FAILS CLOSED (Ticket 01).

    No live E2B client is wired. The previous stub mimicked the E2B
    lifecycle on the HOST (a temp directory plus ``bash -c``), which meant
    model-written commands executed on the host machine whenever
    ``remote_e2b`` — the production default — was selected. That behavior
    is removed: every operation now raises :class:`SandboxUnavailableError`
    BEFORE any host access, and no automatic fallback to
    :class:`LocalNativeComputer` exists.

    When the real E2B integration lands, it replaces this class body with
    the SDK client; the interface, the fail-closed taxonomy, and the
    ``sandbox_unavailable`` public mapping stay the same.

    ``failure_mode`` is still accepted so the factory signature and test
    injection points stay compatible; it no longer changes any behavior
    because every operation refuses regardless.
    """

    sandbox_type: SandboxType = "remote_e2b"

    def __init__(self, failure_mode: str | None = None) -> None:
        # ``failure_mode`` is accepted for factory-signature compatibility
        # and intentionally NOT stored: every operation refuses regardless,
        # so there is no injectable failure behavior to read back.
        _ = failure_mode

    def create_workspace(self, run_id: str) -> str:
        raise SandboxUnavailableError(_REFUSAL_MESSAGE)

    def generate_pdf(
        self,
        workspace_path: str,
        content: str,
        filename: str,
    ) -> str:
        raise SandboxUnavailableError(_REFUSAL_MESSAGE)

    def build_manifest(
        self,
        workspace_path: str,
        relative_path: str,
        declared_mime: str,
    ) -> ArtifactManifest:
        raise SandboxUnavailableError(_REFUSAL_MESSAGE)

    def read_artifact_bytes(
        self,
        workspace_path: str,
        relative_path: str,
    ) -> bytes:
        raise SandboxUnavailableError(_REFUSAL_MESSAGE)

    def run_bash(
        self,
        workspace_path: str,
        command: str,
        *,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 1_048_576,
    ) -> "BashResult":
        raise SandboxUnavailableError(_REFUSAL_MESSAGE)

    def cleanup(self, workspace_path: str) -> ResourceUsage:
        raise SandboxUnavailableError(_REFUSAL_MESSAGE)


# ---- Factory --------------------------------------------------------------


def create_computer(
    sandbox_type: SandboxType = "remote_e2b",
    *,
    failure_mode: str | None = None,
) -> Computer:
    """Return a :class:`Computer` for ``sandbox_type``.

    Spec decision #19: production uses ``remote_e2b``; development may
    select ``local_native`` or ``wsl_localvm``. The factory never falls
    back from ``remote_e2b`` to a host computer — an E2B failure fails
    the run (fail-closed).

    ``wsl_localvm`` currently maps to :class:`LocalNativeComputer` because
    WSL integration is not wired in the stub stage. Real WSL support
    lands with the LocalVM ticket.
    """
    if sandbox_type == "remote_e2b":
        return RemoteE2BComputer(failure_mode=failure_mode)
    if sandbox_type in ("local_native", "wsl_localvm"):
        return LocalNativeComputer(failure_mode=failure_mode)
    raise ValueError(f"invalid sandbox_type: {sandbox_type!r}")
