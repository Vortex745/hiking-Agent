"""Log-only session lifecycle audit endpoint consumed by the frontend."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("ai-service.session_lifecycle")

session_lifecycle_router = APIRouter()

_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"


class SessionLifecycleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: Literal[
        "session.created",
        "session.restored",
        "session.closed",
        "session.deleted",
        "session.pruned",
        "session.snapshot.failed",
    ]
    reason: Literal[
        "new_session",
        "switch",
        "window_exit",
        "tab_hidden",
        "delete",
        "startup_prune",
        "quota_exceeded",
        "storage_unavailable",
    ]
    session_id: str = Field(
        ..., min_length=1, max_length=128, pattern=_ID_PATTERN
    )
    conversation_id: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=_ID_PATTERN
    )
    snapshot_id: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=_ID_PATTERN
    )
    superseded_snapshot_id: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=_ID_PATTERN
    )
    outgoing_session_id: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=_ID_PATTERN
    )
    outgoing_snapshot_id: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=_ID_PATTERN
    )
    destroyed_count: int = Field(default=0, ge=0, le=1000)
    snapshot_bytes: int | None = Field(default=None, ge=0, le=10_000_000)


@session_lifecycle_router.post("/sessions/lifecycle", status_code=204)
async def record_session_lifecycle(
    payload: SessionLifecycleRequest, request: Request
) -> Response:
    logger.info(
        "session lifecycle %s",
        payload.event,
        extra={
            "event": payload.event,
            "session_id": payload.session_id,
            "conversation_id": payload.conversation_id,
            "snapshot_id": payload.snapshot_id,
            "superseded_snapshot_id": payload.superseded_snapshot_id,
            "outgoing_session_id": payload.outgoing_session_id,
            "outgoing_snapshot_id": payload.outgoing_snapshot_id,
            "snapshot_bytes": payload.snapshot_bytes,
            "destroyed_count": payload.destroyed_count,
            "subject_id": str(getattr(request.state, "subject_id", "") or ""),
        },
    )
    return Response(status_code=204, headers={"Cache-Control": "no-store"})
