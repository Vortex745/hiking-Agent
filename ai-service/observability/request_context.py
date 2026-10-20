"""Request correlation primitives shared by middleware, logs and traces.

Request ids are generated at the gateway edge and validated at every hop;
traceparent follows the W3C Trace Context format and is never invented.
"""

from __future__ import annotations

import itertools
import re
from datetime import datetime, timezone

REQUEST_ID_HEADER = "x-request-id"
TRACEPARENT_HEADER = "traceparent"

# Gateway-friendly charset and bounded length.
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,128}$")
_TRACEPARENT_RE = re.compile(
    r"^(?P<version>[0-9a-f]{2})-(?P<trace>[0-9a-f]{32})-(?P<span>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})$"
)

_counter = itertools.count(1)


def generate_request_id() -> str:
    """Timestamp + atomic sequence, matching the gateway's format."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{stamp}-{next(_counter)}"


def valid_request_id(value: str | None) -> str | None:
    if not value:
        return None
    return value if _REQUEST_ID_RE.match(value) else None


def parse_traceparent(value: str | None) -> str | None:
    """Return the trace id when ``value`` is a valid W3C traceparent."""
    if not value:
        return None
    match = _TRACEPARENT_RE.match(value.strip())
    if not match:
        return None
    if match.group("version") == "ff":
        return None
    if match.group("trace") == "0" * 32 or match.group("span") == "0" * 16:
        return None
    return match.group("trace")
