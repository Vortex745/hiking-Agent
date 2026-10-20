"""Redaction helpers shared by logs and traces.

The project's reject list (see the observability spec) must never reach
stdout logs or exported spans: coordinates, addresses, memory bodies, full
retrieved documents, raw tool output, cookies and auth values. Values are
truncated rather than dropped so diagnostics stay useful.
"""

from __future__ import annotations

import json
import re

REDACTED = "[REDACTED]"
TOOL_OUTPUT_LIMIT = 4000
DOCUMENT_LIMIT = 160
DOCUMENT_COUNT_LIMIT = 25
_MAX_STRUCTURED_DEPTH = 3

_SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9._-]{8,}"),
    re.compile(r"\bpk-lf-[A-Za-z0-9._-]+"),
    re.compile(r"\bsk-lf-[A-Za-z0-9._-]+"),
    # Header-shaped credentials: keep the header name, drop the value.
    re.compile(r"(?i)\bauthorization\s*:\s*(?:basic|bearer)\s+[^\s,;]+"),
    re.compile(r"(?i)\bcookie\s*:\s*[^\n;]+"),
    re.compile(r"(?i)\bx-api-key\s*:\s*[^\s,;]+"),
    # Token/key assignments, including suffixed names (``access_token=``).
    re.compile(r"(?i)\b[\w-]*(?:access_token|refresh_token|id_token)\s*[=:]\s*[^\s,;&]+"),
    re.compile(r"(?i)\b[\w-]*(?:api[_-]?key|apikey|secret|password)\s*[=:]\s*[^\s,;&]+"),
    re.compile(r"(?i)\b[\w-]*token\s*[=:]\s*[^\s,;&]+"),
)

# Location values may be embedded in free text (the product puts the
# authorized location into a system prompt), so key=value pairs must be
# scrubbed even when the message is not a structured mapping. Addresses
# contain spaces, so the value terminates at a line break or punctuation.
# The optional prefix keeps ``user_latitude=`` covered while leaving
# metrics such as ``latency=`` untouched (the separator must follow).
_LOCATION_PATTERNS = (
    re.compile(
        r"(?i)\b[\w-]*(?:latitude|longitude|adcode|address|addr|lat|lng|lon)"
        r"\s*[=:]\s*[^\n，,；;。]+"
    ),
)

# JSON fragments that are not valid JSON on their own (e.g. truncated
# payloads) still need key-based redaction.
_QUOTED_KEY_PATTERN = re.compile(
    r"""(?P<quote>["'])(?P<key>lat|lng|lon|latitude|longitude|adcode|address|addr|"""
    r"""memory|memory_body|token|secret|password|cookie|authorization)"""
    r"""(?P=quote)\s*:\s*"""
    r"""(?P<value>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^,}\]\n]+)""",
    re.IGNORECASE,
)

_DENIED_KEYS = {
    "authorization",
    "cookie",
    "set_cookie",
    "x_api_key",
    "api_key",
    "apikey",
    "secret",
    "password",
    "token",
    "access_token",
    "refresh_token",
    "id_token",
    "latitude",
    "longitude",
    "lat",
    "lng",
    "lon",
    "adcode",
    "address",
    "addr",
    "memory",
    "memory_body",
}

# Suffixed variants (``user_latitude``, ``session_token``) are just as
# sensitive as the bare key. Bare metrics such as ``latency``/``token_count``
# never end with these suffixes, so they stay readable.
_DENIED_KEY_SUFFIXES = (
    "_lat",
    "_lng",
    "_lon",
    "_latitude",
    "_longitude",
    "_adcode",
    "_address",
    "_addr",
    "_memory",
    "_token",
    "_secret",
    "_password",
    "_cookie",
    "_authorization",
)


def is_denied_key(key) -> bool:
    """True when a mapping key belongs to the reject list."""
    text = str(key).strip().lower().replace("-", "_")
    if text in _DENIED_KEYS:
        return True
    return text.endswith(_DENIED_KEY_SUFFIXES)


def scrub_text(value: str, limit: int = TOOL_OUTPUT_LIMIT, *, _depth: int = 0) -> str:
    """Redact credentials/location values and truncate to ``limit`` chars."""
    text = "" if value is None else str(value)
    stripped = text.strip()
    if _depth < _MAX_STRUCTURED_DEPTH and stripped[:1] in {"{", "["}:
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None:
            text = json.dumps(
                sanitize_mapping(parsed, depth=_depth), ensure_ascii=False
            )
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    for pattern in _LOCATION_PATTERNS:
        text = pattern.sub(REDACTED, text)
    text = _QUOTED_KEY_PATTERN.sub(
        lambda match: f"{match.group('quote')}{match.group('key')}"
        f"{match.group('quote')}: \"{REDACTED}\"",
        text,
    )
    if limit and len(text) > limit:
        return text[:limit] + f"...[truncated {len(text) - limit} chars]"
    return text


def sanitize_mapping(value, *, depth: int = 0):
    """Recursively drop denied keys and scrub string values."""
    if depth > 4:
        return REDACTED
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            key_text = str(key)
            if is_denied_key(key_text):
                cleaned[key_text] = REDACTED
            else:
                cleaned[key_text] = sanitize_mapping(item, depth=depth + 1)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [sanitize_mapping(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        return scrub_text(value, _depth=depth + 1)
    return value


def sanitize_messages(messages) -> list[dict]:
    """Represent chat messages for span attributes without leaking secrets."""
    result = []
    for message in messages or []:
        role = getattr(message, "type", None) or getattr(message, "role", "unknown")
        content = getattr(message, "content", message)
        if str(role) == "tool":
            # Tool messages carry full retrieved documents; only their size
            # may leave the process.
            result.append(
                {
                    "role": "tool",
                    "content": REDACTED,
                    "content.length": len(str(content or "")),
                }
            )
            continue
        result.append({"role": str(role), "content": scrub_text(content)})
    return result


def sanitize_error(exc: BaseException, limit: int = 500) -> str:
    return f"{type(exc).__name__}: {scrub_text(str(exc), limit)}"


def sanitize_documents(documents) -> list[dict]:
    """Bound retrieved documents to the spec's 25 x 160 character budget."""
    bounded = []
    for document in list(documents or [])[:DOCUMENT_COUNT_LIMIT]:
        if isinstance(document, dict):
            text = document.get("content", document.get("page_content", ""))
        else:
            text = getattr(document, "page_content", str(document))
        bounded.append({"content": scrub_text(text, DOCUMENT_LIMIT)})
    return bounded
