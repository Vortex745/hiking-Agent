import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, ListFlowable, ListItem
from langchain_core.tools import tool
from runtime_paths import runtime_dir

logger = logging.getLogger("ai-service.pdf")

WORKSPACE_DIR = runtime_dir("WORKSPACE_DIR", "workspace")

_CJK_FONT_NAME: str | None = None
_CJK_FONT_LOADED: bool = False

# Priority-ordered candidate Chinese font filenames to search for.
_CJK_CANDIDATES: list[str] = [
    "simhei.ttf",              # 黑体 (Windows)
    "msyh.ttc",                # 微软雅黑 (Windows)
    "NotoSansSC-VF.ttf",       # Noto Sans SC (cross-platform)
    "simsun.ttc",              # 宋体 (Windows)
    "simsunb.ttf",             # 宋体扩展 (Windows)
    "NotoSerifSC-VF.ttf",      # Noto Serif SC (cross-platform)
]

_WINDOWS_FONT_DIR: str = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
_LINUX_FONT_DIRS: list[str] = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
]


def _find_cjk_font_path() -> str | None:
    """Search for an available CJK font file on the system."""
    search_dirs: list[str] = [_WINDOWS_FONT_DIR] + _LINUX_FONT_DIRS

    for candidate in _CJK_CANDIDATES:
        for base_dir in search_dirs:
            # Walk subdirectories (e.g. /usr/share/fonts/truetype/...)
            for root, _dirs, files in os.walk(base_dir):
                if candidate in files:
                    font_path = os.path.join(root, candidate)
                    if os.path.isfile(font_path) and os.path.getsize(font_path) > 0:
                        return font_path
                # Also try exact name match
                exact_path = os.path.join(root, candidate)
                if os.path.isfile(exact_path) and os.path.getsize(exact_path) > 0:
                    return exact_path

    return None


def _ensure_cjk_font() -> None:
    """Load and register a CJK font into reportlab (idempotent)."""
    global _CJK_FONT_NAME, _CJK_FONT_LOADED
    if _CJK_FONT_LOADED:
        return

    font_path = _find_cjk_font_path()
    if font_path is None:
        logger.warning("No CJK font found on system; Chinese characters may render as tofu")
        _CJK_FONT_LOADED = True
        return

    try:
        # Use a stable registered name regardless of which font file we found
        _CJK_FONT_NAME = "CJKFont"
        pdfmetrics.registerFont(TTFont(_CJK_FONT_NAME, font_path))
        _CJK_FONT_LOADED = True
        logger.info("Registered CJK font: %s → %s", font_path, _CJK_FONT_NAME)
    except Exception as e:
        logger.warning("Failed to register CJK font %s: %s", font_path, e)
        _CJK_FONT_LOADED = True


def _get_cjk_font_name() -> str | None:
    """Return the registered CJK font name, or None if not available."""
    return _CJK_FONT_NAME


def artifact_metadata(path: Path, *, title: str, kind: str, mime_type: str) -> dict[str, Any]:
    workspace = WORKSPACE_DIR.resolve()
    resolved = path.resolve()
    relative_path = resolved.relative_to(workspace).as_posix()
    return {
        "kind": kind,
        "title": title,
        "filename": resolved.name,
        "relative_path": relative_path,
        "download_url": f"/api/v1/artifacts/{quote(relative_path, safe='/')}",
        "mime_type": mime_type,
        "size_bytes": resolved.stat().st_size,
    }


# ---------------------------------------------------------------------------
# Markdown → reportlab rich elements
# ---------------------------------------------------------------------------

_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC = re.compile(r"\*(.+?)\*")
_MD_INLINE_CODE = re.compile(r"`([^`]+)`")


def _md_to_xml(text: str) -> str:
    """Convert inline Markdown formatting to reportlab XML tags."""
    # Bold
    text = _MD_BOLD.sub(r"<b>\1</b>", text)
    # Italic (after bold to avoid conflict)
    text = _MD_ITALIC.sub(r"<i>\1</i>", text)
    # Inline code
    text = _MD_INLINE_CODE.sub(r'<font face="Courier" size="10">\1</font>', text)
    return text


def _markdown_to_elements(content: str) -> list:
    """Parse Markdown content into a list of reportlab flowable elements.

    Supports:
      - # / ## / ### headings
      - **bold** and *italic* inline formatting
      - - or * bullet list items
      - --- horizontal rules
      - Plain text paragraphs
      - Empty lines → spacers
    """
    _ensure_cjk_font()
    cjk = _get_cjk_font_name()
    font = cjk or "Helvetica"

    styles = getSampleStyleSheet()

    h1_style = ParagraphStyle(
        "MD_h1", parent=styles["Heading1"],
        fontName=font, fontSize=16, leading=22, spaceBefore=16, spaceAfter=8,
        textColor=HexColor("#1a1a1a"),
    )
    h2_style = ParagraphStyle(
        "MD_h2", parent=styles["Heading2"],
        fontName=font, fontSize=14, leading=20, spaceBefore=14, spaceAfter=6,
        textColor=HexColor("#333333"),
    )
    h3_style = ParagraphStyle(
        "MD_h3", parent=styles["Heading3"],
        fontName=font, fontSize=12, leading=17, spaceBefore=12, spaceAfter=4,
        textColor=HexColor("#555555"),
    )
    body_style = ParagraphStyle(
        "MD_body", parent=styles["Normal"],
        fontName=font, fontSize=11, leading=17, spaceAfter=8,
    )
    bullet_style = ParagraphStyle(
        "MD_bullet", parent=body_style,
        leftIndent=24, bulletIndent=12,
        spaceBefore=2, spaceAfter=2,
    )

    elements: list = []
    # Collect consecutive bullet lines into groups
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Empty line
        if not line:
            elements.append(Spacer(1, 6))
            i += 1
            continue

        # Horizontal rule
        if line in ("---", "***", "___", "* * *") or re.match(r"^-{3,}$", line):
            elements.append(Spacer(1, 4))
            elements.append(HRFlowable(
                width="100%", thickness=0.5,
                color=HexColor("#cccccc"), spaceAfter=8,
            ))
            i += 1
            continue

        # Heading 1
        if line.startswith("# ") and not line.startswith("## "):
            text = _md_to_xml(line[2:])
            elements.append(Paragraph(text, h1_style))
            i += 1
            continue

        # Heading 2
        if line.startswith("## "):
            text = _md_to_xml(line[3:])
            elements.append(Paragraph(text, h2_style))
            i += 1
            continue

        # Heading 3
        if line.startswith("### "):
            text = _md_to_xml(line[4:])
            elements.append(Paragraph(text, h3_style))
            i += 1
            continue

        # Bullet list group: consecutive lines starting with "- " or "* "
        if re.match(r"^[-*] ", line):
            bullet_lines = []
            while i < len(lines) and re.match(r"^[-*] ", lines[i].strip()):
                raw = re.sub(r"^[-*] ", "", lines[i].strip())
                bullet_lines.append(_md_to_xml(raw))
                i += 1

            for bl in bullet_lines:
                elements.append(Paragraph(f"• {bl}", bullet_style))
            continue

        # Default: body paragraph
        elements.append(Paragraph(_md_to_xml(line), body_style))
        i += 1

    return elements


@tool
async def generate_pdf(title: str, content: str) -> dict[str, Any]:
    """Generate a PDF document from Markdown text content.

    Supports headings (# ## ###), **bold**, *italic*, bullet lists (- *),
    horizontal rules (---), and plain text.

    Args:
        title: The title of the PDF document
        content: The main text content (Markdown format)
    """
    safe_filename = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:50]
    filepath = WORKSPACE_DIR / f"{safe_filename}.pdf"

    try:
        # Ensure WORKSPACE_DIR exists (e.g. when monkeypatched to a missing path at runtime)
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

        # Ensure a CJK font is registered for Chinese text
        _ensure_cjk_font()
        cjk_font = _get_cjk_font_name()

        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            title=title,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontName=cjk_font or styles["Title"].fontName,
            fontSize=20,
            leading=28,
            spaceAfter=24,
        )

        elements = []
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 6))

        # Parse Markdown content into rich elements
        elements.extend(_markdown_to_elements(content))

        doc.build(elements)
        artifact = artifact_metadata(
            filepath,
            title=title,
            kind="pdf",
            mime_type="application/pdf",
        )
        return {
            "ok": True,
            "message": f"PDF 已生成：{artifact['filename']}",
            "artifact": artifact,
        }
    except Exception as e:
        return {"ok": False, "message": f"PDF 生成失败：{str(e)}"}
