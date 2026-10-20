"""Deprecated import location.

Tracing moved to :mod:`observability.tracing` (OpenTelemetry facade). This
module only forwards the session-context helpers so older imports keep
working; the previous SDK callback ingestion path has been removed.
"""

from observability.tracing import (  # noqa: F401
    reset_session as reset_langfuse_session,
    set_session as set_langfuse_session,
)
