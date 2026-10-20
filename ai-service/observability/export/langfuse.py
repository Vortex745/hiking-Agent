"""Langfuse OTLP export adapter.

This module is the ONLY place allowed to reference Langfuse package names or
the ``langfuse.*`` attribute dialect. Business code and the tracing facade use
neutral ``gen_ai.*`` / product keys and go through the small helper functions
below, so swapping the backend only touches this file.
"""

from __future__ import annotations

import base64

from config import settings

# Langfuse self-hosted / cloud OTLP trace ingestion path.
OTLP_TRACES_PATH = "/api/public/otel/v1/traces"

# Neutral resource attributes translated for this backend.
RESOURCE_ATTRIBUTES = {
    "service.name": "ai-service",
}


def _base_url() -> str:
    return (settings.langfuse_base_url or "").rstrip("/")


def endpoint() -> str:
    return f"{_base_url()}{OTLP_TRACES_PATH}"


def auth_header() -> str:
    raw = f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def build_exporter():
    """Build the OTLP/HTTP span exporter for this backend.

    Imported lazily so a deployment without tracing never pays the import cost.
    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    return OTLPSpanExporter(
        endpoint=endpoint(),
        headers={"Authorization": auth_header()},
        timeout=10,
    )


def session_attributes(session_id: str) -> dict:
    """Backend dialect for grouping executions into a session."""
    return {"langfuse.session.id": session_id}


def trace_attributes(name: str) -> dict:
    """Backend dialect for the trace display name."""
    return {"langfuse.trace.name": name}


def observation_attributes(kind: str) -> dict:
    """Backend dialect for span observation type (generation/tool/span)."""
    return {"langfuse.observation.type": kind}


def generation_attributes(
    *,
    model: str,
    provider: str,
    temperature: float | None = None,
    tools: list[str] | None = None,
    usage_input: int | None = None,
    usage_output: int | None = None,
    usage_missing: bool = False,
    time_to_first_token_ms: float | None = None,
) -> dict:
    """Backend dialect for generation fields Langfuse reads directly."""
    attrs: dict = {
        "langfuse.observation.type": "generation",
        "langfuse.observation.model.name": model,
        "gen_ai.request.model": model,
        "gen_ai.system": provider,
    }
    if temperature is not None:
        attrs["gen_ai.request.temperature"] = temperature
    if tools:
        attrs["gen_ai.request.tool.names"] = tools
    if usage_missing:
        attrs["langfuse.observation.usage_details"] = None
    else:
        if usage_input is not None:
            attrs["gen_ai.usage.input_tokens"] = usage_input
        if usage_output is not None:
            attrs["gen_ai.usage.output_tokens"] = usage_output
    if time_to_first_token_ms is not None:
        attrs["gen_ai.response.time_to_first_token_ms"] = time_to_first_token_ms
    return attrs
