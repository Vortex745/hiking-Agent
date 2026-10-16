"""harness-p0-p1 Ticket 04 — generic Markdown → PDF byte rendering (spec T05).

Extracted from the historical plan-book renderer (kept generic on purpose):
CJK font registration with a deterministic fallback chain, A4 layout with
page-number footer, heading/paragraph/list/table story building, and PDF
byte production. Route-comparison reports, forms, and the full plan-book
workflow are deliberately NOT part of this module.

Hard rules enforced here:

* the body is TEXT — every character is XML-escaped before it reaches
  ReportLab, so HTML in the draft is displayed literally and never
  executed; links and images are never fetched or embedded;
* a blank / whitespace-only body is rejected — a PDF is never produced
  "successfully" just because a header could be written;
* the rendered document always carries the route-facts-pending notice —
  a generic draft must never look like a verified safety conclusion;
* rendering requires reportlab; a missing dependency raises
  :class:`PDFRenderingError` instead of silently degrading.

Parse/verify counterpart: :func:`agent_v2.artifacts.validate_pdf_bytes`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("ai-service.agent_v2.pdf_rendering")

#: Shown on the cover and in every page footer: a generic draft's route
#: facts are NOT server-verified and must not be read as a safety verdict.
ROUTE_FACTS_NOTICE = "路线事实待核验"

PDF_BRAND = "AI Hiking"

_PDF_PRIMARY = "#15803d"
_PDF_PRIMARY_DARK = "#14532d"
_PDF_PRIMARY_BG = "#f0fdf4"
_PDF_TEXT = "#1f2937"
_PDF_MUTED = "#6b7280"
_PDF_BORDER = "#e5e7eb"

_FONT_DIRS = [
    r"C:\Windows\Fonts",
    "/usr/share/fonts/truetype",
    "/usr/share/fonts/opentype/noto",
    "/usr/share/fonts",
    str(Path(__file__).resolve().parent.parent / "assets" / "fonts"),
]

#: (regular file, bold file or None) — first existing pair wins.
_FONT_PAIRS = [
    ("msyh.ttc", "msyhbd.ttc"),
    ("NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"),
    ("NotoSansSC-Regular.otf", "NotoSansSC-Bold.otf"),
    ("simhei.ttf", None),
    ("wqy-microhei.ttc", None),
]


class PDFRenderingError(Exception):
    """PDF rendering failed (blank body, missing dependency, bad layout)."""


def register_pdf_font() -> tuple[str, str]:
    """Register a Unicode CJK font pair and return ``(regular, bold)`` names.

    STSong-Light renders ASCII with broken spacing, so a real TTF/TTC
    (Microsoft YaHei / Noto Sans CJK / SimHei / WenQuanYi) is preferred;
    the CID font stays as the deterministic fallback for minimal font
    environments, and Helvetica is the last resort so rendering never
    hard-fails on font availability alone.
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont

    candidates: list[tuple[Path, Path | None]] = []
    override = os.getenv("AI_HIKING_PDF_FONT")
    if override:
        override_path = Path(override)
        bold_guess = override_path.with_name(
            override_path.stem + "-Bold" + override_path.suffix
        )
        candidates.append((override_path, bold_guess if bold_guess.exists() else None))
    for regular_name, bold_name in _FONT_PAIRS:
        for base in _FONT_DIRS:
            regular = Path(base) / regular_name
            if regular.exists():
                bold = (Path(base) / bold_name) if bold_name else None
                candidates.append((regular, bold if bold and bold.exists() else None))
                break

    for regular_path, bold_path in candidates:
        try:
            kwargs: dict[str, Any] = {}
            if regular_path.suffix.lower() == ".ttc":
                kwargs["subfontIndex"] = 0
            pdfmetrics.registerFont(TTFont("PdfCJK", str(regular_path), **kwargs))
            bold = "PdfCJK"
            if bold_path is not None:
                bold_kwargs = (
                    {"subfontIndex": 0} if bold_path.suffix.lower() == ".ttc" else {}
                )
                try:
                    pdfmetrics.registerFont(
                        TTFont("PdfCJK-Bold", str(bold_path), **bold_kwargs)
                    )
                    bold = "PdfCJK-Bold"
                except Exception:
                    logger.info("pdf_rendering: bold CJK font unavailable; reusing regular")
            pdfmetrics.registerFontFamily(
                "PdfCJK", normal="PdfCJK", bold=bold, italic="PdfCJK", boldItalic=bold
            )
            return "PdfCJK", bold
        except Exception:
            logger.info("pdf_rendering: font candidate failed: %s", regular_path)
            continue

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light", "STSong-Light"
    except Exception:
        logger.warning("pdf_rendering: STSong-Light unavailable; using Helvetica")
        return "Helvetica", "Helvetica"


def _pdf_footer(canvas: Any, doc: Any) -> None:
    """Brand + route-facts notice + page number on every page."""
    from reportlab.lib.colors import HexColor

    canvas.saveState()
    width, height = doc.pagesize
    canvas.setStrokeColor(HexColor(_PDF_BORDER))
    canvas.setLineWidth(0.6)
    canvas.line(18 * 2.83465, 14 * 2.83465 + 10, width - 18 * 2.83465, 14 * 2.83465 + 10)
    canvas.setFont(doc._pdf_font_regular, 7.5)
    canvas.setFillColor(HexColor(_PDF_MUTED))
    canvas.drawString(
        18 * 2.83465, 14 * 2.83465,
        f"{PDF_BRAND} · 自动生成草案，仅供出行参考 · {ROUTE_FACTS_NOTICE}",
    )
    canvas.drawRightString(
        width - 18 * 2.83465, 14 * 2.83465, f"第 {canvas.getPageNumber()} 页"
    )
    canvas.restoreState()


def _markdown_story(doc: Any, content: str) -> list[Any]:
    """Build the ReportLab story from Markdown text.

    Inline markup is re-emitted as Paragraph mini-HTML from ESCAPED raw
    text only; nothing from the draft is interpreted as markup, so
    scripts, links and image references stay literal text.
    """
    import mistune
    from reportlab.lib import colors
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    regular, bold = doc._pdf_font_regular, doc._pdf_font_bold

    def style(name: str, **kw: Any) -> ParagraphStyle:
        kw.setdefault("fontName", regular)
        kw.setdefault("alignment", TA_LEFT)
        kw.setdefault("wordWrap", "CJK")
        return ParagraphStyle(name, **kw)

    h1 = style("H1", fontName=bold, fontSize=15, leading=21,
               textColor=HexColor(_PDF_PRIMARY_DARK), spaceBefore=4, spaceAfter=6)
    h2 = style("H2", fontName=bold, fontSize=12.5, leading=17,
               textColor=HexColor(_PDF_PRIMARY_DARK), spaceBefore=10, spaceAfter=4,
               keepWithNext=True)
    h3 = style("H3", fontName=bold, fontSize=11, leading=15,
               spaceBefore=6, spaceAfter=2, keepWithNext=True)
    body = style("Body", fontSize=10, leading=16, textColor=HexColor(_PDF_TEXT))
    bullet = style("Bullet", fontSize=10, leading=16, leftIndent=12,
                   textColor=HexColor(_PDF_TEXT))
    cell = style("Cell", fontSize=9, leading=13, textColor=HexColor(_PDF_TEXT))

    def inline(nodes: list[dict[str, Any]], *, markup: bool = True) -> str:
        parts: list[str] = []
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
        rendered: list[Any] = []
        for node in nodes:
            kind = node["type"]
            children = node.get("children", [])
            if kind == "heading":
                level = node.get("attrs", {}).get("level", 3)
                heading_style = h1 if level <= 1 else (h2 if level == 2 else h3)
                rendered.append(Paragraph(inline(children) or "&nbsp;", heading_style))
                rendered[-1].keepWithNext = True
            elif kind in {"paragraph", "block_text"}:
                rendered.append(Paragraph(inline(children) or "&nbsp;", body))
            elif kind == "list":
                ordered = bool(node.get("attrs", {}).get("ordered"))
                for index, item in enumerate(children, 1):
                    label = f"{index}. " if ordered else "•　"
                    item_nodes = item.get("children", [])
                    if not item_nodes:
                        continue
                    first, *rest = item_nodes
                    rendered.append(
                        Paragraph(label + inline(first.get("children", [])), bullet)
                    )
                    rendered.extend(blocks(rest))
            elif kind == "table":
                head, table_body = children
                rows = [head["children"]] + [
                    row["children"] for row in table_body["children"]
                ]
                column_count = max(len(row) for row in rows)
                cells = [
                    [
                        Paragraph(inline(cell_node.get("children", [])), cell)
                        for cell_node in (row + [[]] * (column_count - len(row)))
                    ]
                    for row in rows
                ]
                table = Table(
                    cells,
                    colWidths=[(doc.width - 12) / column_count] * column_count,
                    repeatRows=1,
                    splitInRow=1,
                    hAlign="LEFT",
                )
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor(_PDF_PRIMARY_BG)),
                    ("GRID", (0, 0), (-1, -1), 0.45, HexColor(_PDF_BORDER)),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                rendered.append(table)
            elif children:
                rendered.extend(blocks(children))
            elif node.get("raw"):
                rendered.append(Paragraph(escape(node["raw"]).replace("\n", "<br/>"), body))
            if kind != "blank_line":
                rendered.append(Spacer(1, 4))
        return rendered

    return blocks(mistune.create_markdown(renderer="ast", plugins=["table"])(content))


def render_markdown_pdf_bytes(content: str, *, title: str = "徒步计划草案") -> bytes:
    """Render ``content`` (Markdown draft) into real PDF bytes.

    Raises :class:`PDFRenderingError` for a blank body or a missing
    rendering dependency — callers must treat that as a generation
    failure, never as a successful export.
    """
    if not content or not content.strip():
        raise PDFRenderingError("正文为空：拒绝生成 PDF（空白正文不构成可导出草案）。")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise PDFRenderingError(
            "PDF generation requires reportlab; install ai-service/requirements.txt"
        ) from exc

    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import ParagraphStyle
    from xml.sax.saxutils import escape

    regular, bold = register_pdf_font()
    from io import BytesIO

    output = BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title=title,
        author=PDF_BRAND,
    )
    doc._pdf_font_regular = regular
    doc._pdf_font_bold = bold

    notice_style = ParagraphStyle(
        "RouteFactsNotice",
        fontName=regular,
        fontSize=8.6,
        leading=12,
        wordWrap="CJK",
        textColor=HexColor(_PDF_MUTED),
    )
    story: list[Any] = [
        Paragraph(escape(title), ParagraphStyle(
            "DocTitle", fontName=bold, fontSize=15, leading=21,
            wordWrap="CJK", textColor=HexColor(_PDF_PRIMARY_DARK),
        )),
        Spacer(1, 2),
        Paragraph(
            f"{ROUTE_FACTS_NOTICE}：本草案由通用排版生成，其中的路线里程、爬升、耗时、"
            "天气与开放状态未经服务器核验，不得作为安全结论；出发前请以官方与实时来源为准。",
            notice_style,
        ),
        Spacer(1, 8),
        *_markdown_story(doc, content),
    ]
    doc.build(story, onFirstPage=_pdf_footer, onLaterPages=_pdf_footer)
    return output.getvalue()


__all__ = [
    "PDFRenderingError",
    "ROUTE_FACTS_NOTICE",
    "register_pdf_font",
    "render_markdown_pdf_bytes",
]
