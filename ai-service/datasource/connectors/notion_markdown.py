"""Pure Notion block-tree to Markdown renderer.

The connector attaches child blocks under a ``children`` key. This module never
performs network calls and deliberately omits expiring image/file URLs.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

_LIST_INDENT = "  "


def render_rich_text(items: Iterable[Mapping[str, Any]] | None) -> str:
    parts: list[str] = []
    for item in items or []:
        text = str(item.get("plain_text") or "")
        if not text:
            continue
        if item.get("type") == "equation":
            expression = (item.get("equation") or {}).get("expression")
            if expression:
                text = str(expression)
        annotations = item.get("annotations") or {}
        if annotations.get("code"):
            text = f"`{text}`"
        if annotations.get("bold"):
            text = f"**{text}**"
        if annotations.get("italic"):
            text = f"*{text}*"
        if annotations.get("strikethrough"):
            text = f"~~{text}~~"
        href = item.get("href") or ((item.get("text") or {}).get("link") or {}).get("url")
        if href:
            text = f"[{text}]({href})"
        parts.append(text)
    return "".join(parts)


def render_blocks(blocks: Iterable[Mapping[str, Any]] | None, depth: int = 0) -> str:
    rendered = [_render_block(block, depth) for block in blocks or []]
    return "\n\n".join(part for part in rendered if part)


def _render_block(block: Mapping[str, Any], depth: int) -> str:
    block_type = str(block.get("type") or "")
    data = block.get(block_type) or {}
    children = block.get("children") or []
    text = render_rich_text(data.get("rich_text") or [])
    indent = _LIST_INDENT * depth

    if block_type == "paragraph":
        return _with_nested(text, children, depth)
    if block_type in {"heading_1", "heading_2", "heading_3"}:
        level = {"heading_1": 1, "heading_2": 2, "heading_3": 3}[block_type]
        return f"{'#' * level} {text}".rstrip()
    if block_type == "bulleted_list_item":
        return f"{indent}- {text}{_nested(children, depth + 1)}"
    if block_type == "numbered_list_item":
        return f"{indent}1. {text}{_nested(children, depth + 1)}"
    if block_type == "to_do":
        checked = "x" if data.get("checked") else " "
        return f"{indent}- [{checked}] {text}{_nested(children, depth + 1)}"
    if block_type == "quote":
        body = _with_nested(text, children, depth)
        return "\n".join(f"> {line}" for line in body.splitlines())
    if block_type == "callout":
        body = _with_nested(text, children, depth)
        return "\n".join(f"> {line}" for line in body.splitlines())
    if block_type == "toggle":
        return f"{indent}- {text}{_nested(children, depth + 1)}"
    if block_type == "code":
        language = str(data.get("language") or "")
        return f"```{language}\n{text}\n```"
    if block_type == "divider":
        return "---"
    if block_type == "table":
        return _render_table(children)
    if block_type in {"column_list", "column", "synced_block"}:
        return _nested(children, depth).lstrip("\n")
    if block_type == "image":
        caption = render_rich_text(data.get("caption") or []) or "图片"
        return f"[图片: {caption}]"
    if block_type == "child_page":
        return f"[子页面: {data.get('title') or 'Untitled'}]"
    if block_type in {"file", "pdf", "video", "audio"}:
        name = data.get("name") or data.get("title") or block_type
        return f"[附件: {name}]"
    if block_type in {"bookmark", "link_preview"}:
        url = data.get("url") or ""
        return f"[链接]({url})" if url else ""
    return ""


def _nested(children: Iterable[Mapping[str, Any]], depth: int) -> str:
    rendered = [_render_block(child, depth) for child in children or []]
    body = "\n".join(part for part in rendered if part)
    return f"\n{body}" if body else ""


def _with_nested(text: str, children: Iterable[Mapping[str, Any]], depth: int) -> str:
    nested = _nested(children, depth)
    if text and nested:
        return f"{text}{nested}"
    return text or nested.lstrip("\n")


def _render_table(rows: Iterable[Mapping[str, Any]]) -> str:
    table_rows: list[list[str]] = []
    for row in rows or []:
        data = row.get("table_row") or {}
        cells = data.get("cells") or []
        table_rows.append([render_rich_text(cell) or " " for cell in cells])
    if not table_rows:
        return ""
    width = max(len(row) for row in table_rows)
    normalized = [row + [" "] * (width - len(row)) for row in table_rows]
    lines = [
        "| " + " | ".join(normalized[0]) + " |",
        "|" + "|".join(["---"] * width) + "|",
    ]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
