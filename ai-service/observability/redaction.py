"""Redaction helpers shared by logs and traces.

The project's reject list (see the observability spec) must never reach
stdout logs or exported spans: coordinates, addresses, memory bodies, full
retrieved documents, raw tool output, cookies and auth values. Values are
truncated rather than dropped so diagnostics stay useful.
"""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"
TOOL_OUTPUT_LIMIT = 4000
DOCUMENT_LIMIT = 160
DOCUMENT_COUNT_LIMIT = 25

_SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9._-]{8,}"),
    re.compile(r"\bpk-lf-[A-Za-z0-9._-]+"),
    re.compile(r"\bsk-lf-[A-Za-z0-9._-]+"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*[=:]\s*[^\s,;]+"),
)

# Location values may be embedded in free text (the product puts the
# authorized location into a system prompt), so key=value pairs must be
# scrubbed even when the message is not a structured mapping. Addresses
# contain spaces, so the value terminates at a line break or punctuation.
_LOCATION_PATTERNS = (
    re.compile(
        r"(?i)\b(latitude|longitude|lat|lng|adcode|address)\s*[=:]\s*[^\n，,；;。]+"
    ),
)

_DENIED_KEYS = {
    "authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api_key",
    "apikey",
    "secret",
    "password",
    "token",
    "latitude",
    "longitude",
    "lat",
    "lng",
    "address",
    "memory",
    "memory_body",
}


def scrub_text(value: str, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    """Redact credentials/location values and truncate to ``limit`` chars."""
    text = "" if value is None else str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    for pattern in _LOCATION_PATTERNS:
        text = pattern.sub(REDACTED, text)
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
            if key_text.lower() in _DENIED_KEYS:
                cleaned[key_text] = REDACTED
            else:
                cleaned[key_text] = sanitize_mapping(item, depth=depth + 1)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [sanitize_mapping(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        return scrub_text(value)
    return value


def sanitize_messages(messages) -> list[dict]:
    """Represent chat messages for span attributes without leaking secrets."""
    result = []
    for message in messages or []:
        role = getattr(message, "type", None) or getattr(message, "role", "unknown")
        content = getattr(message, "content", message)
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
