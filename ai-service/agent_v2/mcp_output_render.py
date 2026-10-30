"""Compose committed custom-MCP outputs into the assistant reply.

Custom MCP tools answer with a JSON envelope. Image tools describe the same
asset several ways (``big_url``/``preview_url``/``origin_url``), so a naive
URL walk turns one picture into several previews and may request a protected
original. This module parses each result ONCE, treats the version fields of
one object as one image, and composes the final reply around the model's own
Markdown: known objects are rewritten to their public preview, repeated
references collapse, and objects the model omitted are appended. Plain-text
tool output keeps its previous behaviour; JSON-only envelopes stay in the
tool result and are never dumped into the reply.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence
from urllib.parse import urlsplit

_IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".bmp",
)
_MAX_IMAGES = 12
# Fallback only: when the model showed no asset at all, a few previews prove
# the tool result; dumping the whole result list is noise.
_MAX_APPENDED_IMAGES = 3

# Lower number wins when one object carries several versions of the asset.
_DISPLAY_PRIORITY = {
    "big_url": 0,
    "preview_url": 1,
    "image": 2,
    "thumbnail": 3,
    "thumbnail_url": 3,
    "thumb": 4,
    "thumb_url": 4,
    "cover": 5,
    "cover_url": 5,
    "image_url": 6,
    "image_urls": 7,
    "url": 8,
}
_ORIGIN_KEYS = frozenset({"origin_url", "download_url"})
_IDENTIFIER_KEYS = ("ai_detail_id", "pid")
_MARKDOWN_TOKEN = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")


@dataclass
class McpImage:
    """One asset, regardless of how many URL versions describe it."""

    service: str
    identifiers: dict[str, Any] = field(default_factory=dict)
    aliases: set[str] = field(default_factory=set)
    origin_urls: set[str] = field(default_factory=set)
    display_candidates: list[tuple[int, str]] = field(default_factory=list)
    title: str | None = None

    @property
    def display_url(self) -> str | None:
        if not self.display_candidates:
            return None
        return min(
            enumerate(self.display_candidates),
            key=lambda item: (item[1][0], item[0]),
        )[1][1]

    @property
    def alt(self) -> str:
        if not isinstance(self.title, str):
            return ""
        cleaned = self.title.replace("[", "").replace("]", "").replace("\n", " ").strip()
        return cleaned[:60]


def _is_http_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _has_image_extension(value: str) -> bool:
    return urlsplit(value).path.lower().endswith(_IMAGE_EXTENSIONS)


def _looks_like_image_url(value: str, key: str | None) -> bool:
    if not _is_http_url(value):
        return False
    if key in _DISPLAY_PRIORITY:
        # ``url`` is generic: only an image extension makes it a picture.
        return key != "url" or _has_image_extension(value)
    return _has_image_extension(value)


def _merge_image(target: McpImage, source: McpImage) -> None:
    target.aliases.update(source.aliases)
    target.origin_urls.update(source.origin_urls)
    for priority, url in source.display_candidates:
        if all(existing != url for _, existing in target.display_candidates):
            target.display_candidates.append((priority, url))
    if not target.identifiers:
        target.identifiers = dict(source.identifiers)
    if not target.title and source.title:
        target.title = source.title


def _merge_images(raw_images: list[McpImage]) -> list[McpImage]:
    merged: list[McpImage] = []
    by_identifier: dict[tuple[str, str, str], McpImage] = {}
    for raw in raw_images:
        identifier_key = next(
            (key for key in _IDENTIFIER_KEYS if raw.identifiers.get(key) not in (None, "")),
            None,
        )
        target: McpImage | None = None
        if identifier_key is not None:
            key = (raw.service, identifier_key, str(raw.identifiers[identifier_key]))
            target = by_identifier.get(key)
        else:
            # Without an identifier only an exactly shared known alias proves
            # two records are the same asset.
            target = next(
                (
                    candidate
                    for candidate in merged
                    if candidate.service == raw.service
                    and not candidate.identifiers
                    and candidate.aliases & raw.aliases
                ),
                None,
            )
        if target is None:
            target = raw
            merged.append(target)
            if identifier_key is not None:
                by_identifier[key] = target
        else:
            _merge_image(target, raw)
    return merged


def _walk_images(
    value: Any,
    key: str | None,
    title: str | None,
    service: str,
    found: list[McpImage],
) -> None:
    if isinstance(value, dict):
        local_title = value.get("title") or value.get("description") or title
        candidates: list[tuple[int, str]] = []
        origins: set[str] = set()
        aliases: set[str] = set()
        identifiers: dict[str, Any] = {}
        for identifier in _IDENTIFIER_KEYS:
            identifier_value = value.get(identifier)
            if identifier_value not in (None, "") and isinstance(
                identifier_value, (str, int)
            ):
                identifiers[identifier] = identifier_value
        for child_key, child in value.items():
            if not isinstance(child, str):
                continue
            if child_key in _ORIGIN_KEYS:
                if _is_http_url(child):
                    origins.add(child)
                    aliases.add(child)
                continue
            if _looks_like_image_url(child, child_key):
                candidates.append((_DISPLAY_PRIORITY.get(child_key, 8), child))
                aliases.add(child)
        if candidates or origins:
            found.append(
                McpImage(
                    service=service,
                    identifiers=identifiers,
                    aliases=aliases,
                    origin_urls=origins,
                    display_candidates=candidates,
                    title=local_title if isinstance(local_title, str) else None,
                )
            )
            return
        for child_key, child in value.items():
            _walk_images(child, child_key, local_title, service, found)
        return
    if isinstance(value, list):
        for item in value:
            _walk_images(item, key, title, service, found)
        return
    if isinstance(value, str) and _looks_like_image_url(value, key):
        found.append(
            McpImage(
                service=service,
                aliases={value},
                display_candidates=[(_DISPLAY_PRIORITY.get(key or "", 8), value)],
                title=title,
            )
        )


def _markdown_images(text: str, service: str) -> list[McpImage]:
    images: list[McpImage] = []
    for match in _MARKDOWN_TOKEN.finditer(text):
        if not match.group(1):
            continue
        url = match.group(3)
        if not _is_http_url(url):
            continue
        images.append(
            McpImage(
                service=service,
                aliases={url},
                display_candidates=[(_DISPLAY_PRIORITY["url"], url)],
                title=match.group(2) or None,
            )
        )
    return images


def _normalize_outputs(
    outputs: Sequence[str | tuple[str, str]],
) -> Iterator[tuple[str, str]]:
    for output in outputs:
        if isinstance(output, tuple):
            service, text = output
        else:
            service, text = "", output
        if isinstance(text, str) and text.strip():
            yield service, text


def _collect(
    outputs: Sequence[str | tuple[str, str]],
) -> tuple[list[McpImage], list[str]]:
    raw_images: list[McpImage] = []
    text_blocks: list[str] = []
    for service, output in _normalize_outputs(outputs):
        try:
            payload = json.loads(output)
        except (TypeError, ValueError):
            images = _markdown_images(output, service)
            if images:
                raw_images.extend(images)
            else:
                text_blocks.append(output.strip())
            continue
        _walk_images(payload, None, None, service, raw_images)
    return _merge_images(raw_images), text_blocks


def extract_mcp_images(
    outputs: Sequence[str | tuple[str, str]],
    *,
    max_images: int = _MAX_IMAGES,
) -> list[McpImage]:
    """Return the distinct images described by one run's MCP outputs."""
    return _collect(outputs)[0][:max_images]


def extract_image_urls(output: str) -> list[tuple[str, str]]:
    """Return ``[(display_url, alt)]`` for one MCP output."""
    images = extract_mcp_images([output])
    return [
        (image.display_url, _image_alt(image, index))
        for index, image in enumerate(images)
        if image.display_url is not None
    ]


def _image_alt(image: McpImage, index: int) -> str:
    return image.alt or f"MCP 图片 {index + 1}"


def _render_images(images: Sequence[McpImage]) -> str:
    lines = [
        f"![{_image_alt(image, index)}]({image.display_url})"
        for index, image in enumerate(images)
        if image.display_url is not None
    ]
    return "\n".join(lines)


def render_mcp_outputs(outputs: Sequence[str | tuple[str, str]]) -> str:
    """Render only the reply fragments contributed by MCP outputs."""
    images, text_blocks = _collect(outputs)
    parts = list(text_blocks)
    rendered_images = _render_images(images[: _MAX_IMAGES])
    if rendered_images:
        parts.append(rendered_images)
    return "\n\n".join(parts)


def _is_display_alias(image: McpImage, url: str) -> bool:
    return any(candidate == url for _, candidate in image.display_candidates)


def compose_mcp_reply(
    model_text: str,
    outputs: Sequence[str | tuple[str, str]],
    *,
    max_images: int = _MAX_IMAGES,
    prior_text: str = "",
) -> str:
    """Merge MCP images into the model's Markdown reply.

    A link to a public alias (``big_url``/``preview_url``) is a picture the
    model chose to show, so it becomes an inline image at that position.
    Links to a protected origin/download keep their link semantics.
    Unreferenced assets are only appended when the model showed none at all,
    and then at most ``_MAX_APPENDED_IMAGES``; once the reply references any
    asset, its selection wins over the raw result list.
    """
    all_images, text_blocks = _collect(outputs)
    images = all_images[:max_images]
    alias_map: dict[str, McpImage] = {}
    # Resolve every alias, not only the first page: the model may pick an
    # asset deeper in a 36-item search result and still expect it inline.
    for image in all_images:
        if image.display_url is None:
            continue
        for alias in image.aliases:
            alias_map.setdefault(alias, image)

    seen: set[int] = set()
    referenced: set[int] = set()
    # A resumed segment must not append assets the pre-pause reply already
    # showed; prior text is evidence only and is never rewritten.
    for prior_match in _MARKDOWN_TOKEN.finditer(prior_text):
        prior_image = alias_map.get(prior_match.group(3))
        if prior_image is None or prior_image.display_url is None:
            continue
        referenced.add(id(prior_image))
        if prior_match.group(1) or _is_display_alias(prior_image, prior_match.group(3)):
            seen.add(id(prior_image))

    def replace(match: re.Match[str]) -> str:
        bang, alt, url = match.group(1), match.group(2), match.group(3)
        image = alias_map.get(url)
        if image is None or image.display_url is None:
            return match.group(0)
        referenced.add(id(image))
        if bang or _is_display_alias(image, url):
            if id(image) in seen:
                return ""
            seen.add(id(image))
            return f"![{alt or image.alt or 'MCP 图片'}]({image.display_url})"
        return f"[{alt}]({image.display_url})"

    merged_text = _MARKDOWN_TOKEN.sub(replace, model_text or "")
    appended: list[str] = []
    if not referenced:
        for image in images:
            if image.display_url is None:
                continue
            if len(appended) >= _MAX_APPENDED_IMAGES:
                break
            appended.append(f"![{_image_alt(image, len(appended))}]({image.display_url})")

    parts: list[str] = []
    if merged_text.strip():
        parts.append(merged_text.strip())
    parts.extend(text_blocks)
    parts.extend(appended)
    return "\n\n".join(part for part in parts if part)
