"""Anonymous subject and opt-in memory APIs for Chat v2."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from agent_v2.artifacts import compute_sha256, host_storage_path, persist_artifact_bytes
from agent_v2.events import EventStream
from agent_v2.memory_service import (
    SUBJECT_COOKIE_NAME,
    MemoryService,
    SubjectResolution,
)
from config import settings


memory_v2_router = APIRouter()
SUBJECT_COOKIE_MAX_AGE = 365 * 24 * 60 * 60


class MemorySettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class CandidateDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]
    decision_request_id: str = Field(..., min_length=1)


class MemoryUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1, max_length=4000)
    category: str = Field(..., min_length=1, max_length=64)
    ttl_days: int | None = Field(default=None, ge=1, le=3650)


def get_memory_service(request: Request) -> MemoryService:
    runtime = getattr(request.app.state, "agent_v2_runtime", None)
    memory = getattr(runtime, "memory", None) if runtime else None
    if memory is None:
        raise HTTPException(status_code=503, detail="Chat v2 memory is unavailable")
    return memory


def resolve_request_subject(
    request: Request,
) -> tuple[MemoryService, SubjectResolution]:
    memory = get_memory_service(request)
    resolution = memory.resolve_subject(request.cookies.get(SUBJECT_COOKIE_NAME))
    request.state.subject_id = resolution.subject_id
    return memory, resolution


def set_subject_cookie(
    response: Response,
    request: Request,
    resolution: SubjectResolution,
) -> None:
    if not resolution.issued:
        return
    response.set_cookie(
        key=SUBJECT_COOKIE_NAME,
        value=resolution.token,
        max_age=SUBJECT_COOKIE_MAX_AGE,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        path="/",
    )


def _finish_export_run(
    runs,
    *,
    run_id: str,
    request_id: str,
    conversation_id: str,
    terminal_type: str,
    payload: dict,
) -> None:
    """Close an export run through the shared assistant/terminal fence.

    The export has no model turn, but its run is durable state: the terminal
    event must be persisted before the status flips, so a partial write can
    never leave a ``done`` run without a replayable terminal.
    """
    snapshot = runs.get_snapshot(run_id)
    stream = EventStream(
        request_id=request_id,
        conversation_id=conversation_id,
        run_id=run_id,
    )
    stream.advance_to(snapshot.sequence_cursor if snapshot else 0)
    envelope = stream.build(terminal_type, payload)
    try:
        runs.finalize_run(
            run_id=run_id,
            request_id=request_id,
            terminal_type=envelope.type,
            payload=envelope.payload,
            sequence=envelope.sequence,
            event_id=envelope.event_id,
            timestamp=envelope.timestamp,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "terminal_persist_failed",
                "message": "导出终态未能落库，请稍后重试。",
            },
        ) from exc


def subject_json(
    request: Request,
    resolution: SubjectResolution,
    content: dict,
    *,
    status_code: int = 200,
) -> JSONResponse:
    response = JSONResponse(status_code=status_code, content=content)
    set_subject_cookie(response, request, resolution)
    return response


@memory_v2_router.get("/memory/settings")
async def get_memory_settings(request: Request):
    memory, subject = resolve_request_subject(request)
    return subject_json(
        request,
        subject,
        {
            "subject_id": subject.subject_id,
            "enabled": memory.memory_enabled(subject.subject_id),
        },
    )


@memory_v2_router.put("/memory/settings")
async def update_memory_settings(body: MemorySettingsRequest, request: Request):
    memory, subject = resolve_request_subject(request)
    memory.set_memory_enabled(subject.subject_id, body.enabled)
    return subject_json(
        request,
        subject,
        {"subject_id": subject.subject_id, "enabled": body.enabled},
    )


@memory_v2_router.get("/memories")
async def list_memories(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    memory, subject = resolve_request_subject(request)
    return subject_json(
        request,
        subject,
        {
            "items": memory.list_active_memories(
                subject.subject_id, limit=limit, offset=offset
            ),
            "total": memory.count_active_memories(subject.subject_id),
            "limit": limit,
            "offset": offset,
        },
    )


@memory_v2_router.post("/memory/candidates/{candidate_id}/decision")
async def decide_memory_candidate(
    candidate_id: str, body: CandidateDecisionRequest, request: Request
):
    memory, subject = resolve_request_subject(request)
    try:
        decided = memory.decide_candidate(
            subject.subject_id,
            candidate_id,
            body.decision,
            body.decision_request_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if decided is None:
        raise HTTPException(status_code=404, detail="Memory candidate not found")
    result, idempotent = decided
    return subject_json(request, subject, {**result, "idempotent": idempotent})


@memory_v2_router.patch("/memories/{memory_id}")
async def update_memory(
    memory_id: str, body: MemoryUpdateRequest, request: Request
):
    memory, subject = resolve_request_subject(request)
    updated = memory.update_memory(
        subject.subject_id,
        memory_id,
        content=body.content,
        category=body.category,
        ttl_days=body.ttl_days,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return subject_json(request, subject, updated)


@memory_v2_router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str, request: Request):
    memory, subject = resolve_request_subject(request)
    if not memory.delete_memory(subject.subject_id, memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return subject_json(request, subject, {"deleted": True})


@memory_v2_router.delete("/memories")
async def clear_memories(
    request: Request, confirm: bool = Query(default=False)
):
    if not confirm:
        raise HTTPException(status_code=409, detail="Confirmation required")
    memory, subject = resolve_request_subject(request)
    deleted = memory.delete_all_memories(subject.subject_id)
    return subject_json(request, subject, {"deleted": deleted})


@memory_v2_router.get("/memories/export")
async def export_memories(request: Request):
    memory, subject = resolve_request_subject(request)
    runtime = getattr(request.app.state, "agent_v2_runtime", None)
    runs = getattr(runtime, "runs", None) if runtime else None
    if runs is None:
        raise HTTPException(status_code=503, detail="Chat v2 persistence unavailable")

    total = memory.count_active_memories(subject.subject_id)
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "memories": memory.list_active_memories(
            subject.subject_id, limit=max(total, 1), offset=0
        ),
    }
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    artifact_id = str(uuid.uuid4())
    conversation_id = f"memory-export-{uuid.uuid4()}"
    request_id = f"memory-export-{uuid.uuid4()}"
    if not memory.claim_conversation(subject.subject_id, conversation_id):
        raise HTTPException(status_code=409, detail="Export conversation conflict")
    admission = runs.admit(
        request_id=request_id,
        conversation_id=conversation_id,
        profile_id="chat",
        user_message="Export active memories",
    )
    if not admission.run_id:
        raise HTTPException(
            status_code=503,
            detail={
                "code": admission.error_code or "admission_failed",
                "message": "导出运行未完成持久准入，请稍后重试。",
            },
        )
    run_id = admission.run_id
    created_at = datetime.now(timezone.utc).isoformat()
    expires_at = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    storage_path = host_storage_path(
        str(Path(settings.agent_v2_state_dir) / "artifacts"),
        artifact_id,
        "memories.json",
    )
    try:
        persist_artifact_bytes(storage_path, data)
        runs.insert_artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            conversation_id=conversation_id,
            public_filename="memories.json",
            mime_type="application/json",
            size_bytes=len(data),
            sha256=compute_sha256(data),
            storage_path=storage_path,
            status="available",
            failure_reason=None,
            created_at=created_at,
            expires_at=expires_at,
        )
    except Exception:
        _finish_export_run(
            runs,
            run_id=run_id,
            request_id=request_id,
            conversation_id=conversation_id,
            terminal_type="error",
            payload={
                "code": "memory_export_failed",
                "message": "导出产物未能完整落库。",
                "recoverable": True,
            },
        )
        raise HTTPException(status_code=503, detail="Export artifact unavailable")
    _finish_export_run(
        runs,
        run_id=run_id,
        request_id=request_id,
        conversation_id=conversation_id,
        terminal_type="done",
        payload={"artifact_id": artifact_id, "expires_at": expires_at},
    )
    return subject_json(
        request,
        subject,
        {"artifact_id": artifact_id, "expires_at": expires_at},
    )


@memory_v2_router.post("/subject/rotate")
async def rotate_subject(request: Request):
    memory, subject = resolve_request_subject(request)
    rotated = memory.rotate_subject(subject.subject_id)
    return subject_json(
        request,
        rotated,
        {"subject_id": rotated.subject_id, "rotated": True},
    )
