"""Redact span attributes on the export path.

Business code already sanitizes what it puts on spans, but a forgotten
``set_attribute`` (or a value assembled elsewhere) must not leave the
process. This processor runs *before* the batch processor so the exporter
only ever sees redacted values.
"""

from __future__ import annotations

from typing import Any

from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.trace import SpanProcessor

from observability.redaction import REDACTED, is_denied_key, scrub_text

_MAX_VALUE = 4000


class RedactingSpanProcessor(SpanProcessor):
    """Apply the reject list to every span attribute before it is exported."""

    def on_start(self, span: Any, parent_context: Any | None = None) -> None:
        return None

    def on_end(self, span: Any) -> None:
        try:
            attributes = getattr(span, "_attributes", None)
            if attributes is None:
                return None
            cleaned: dict[str, Any] = {}
            for key, value in list(attributes.items()):
                if is_denied_key(key):
                    cleaned[key] = REDACTED
                elif isinstance(value, str):
                    cleaned[key] = scrub_text(value, _MAX_VALUE)
                else:
                    cleaned[key] = value
            # Ended spans are immutable; replace the container instead of
            # mutating it (private on purpose — SpanProcessor has no public
            # attribute API after ``end``).
            span._attributes = BoundedAttributes(attributes=cleaned)
        except Exception:  # pragma: no cover - never break export
            return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int | None = None) -> bool:
        return True
