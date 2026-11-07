"""Session lifecycle audit contract.

The endpoint is log-only. It accepts opaque lifecycle metadata and rejects
anything that could carry conversation content.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))

from api.session_lifecycle import session_lifecycle_router

LOGGER_NAME = "ai-service.session_lifecycle"


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(session_lifecycle_router, prefix="/api/v2")
    return TestClient(app)


def _payload(**overrides):
    payload = {
        "event": "session.created",
        "reason": "new_session",
        "session_id": "session-abc123",
        "conversation_id": "chat-abc123",
        "snapshot_id": "snap-abc123",
        "superseded_snapshot_id": "snap-old",
        "outgoing_session_id": "session-old",
        "outgoing_snapshot_id": "snap-old",
        "snapshot_bytes": 1200,
        "destroyed_count": 1,
    }
    payload.update(overrides)
    return payload


def test_valid_event_is_logged_without_content_fields(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    response = _client().post("/api/v2/sessions/lifecycle", json=_payload())

    assert response.status_code == 204
    assert response.content == b""
    records = [record for record in caplog.records if record.name == LOGGER_NAME]
    assert len(records) == 1
    record = records[0]
    assert record.event == "session.created"
    assert record.session_id == "session-abc123"
    assert record.conversation_id == "chat-abc123"
    assert record.snapshot_id == "snap-abc123"
    assert record.superseded_snapshot_id == "snap-old"
    assert record.outgoing_session_id == "session-old"
    assert record.outgoing_snapshot_id == "snap-old"
    assert record.snapshot_bytes == 1200
    assert record.destroyed_count == 1
    for forbidden in ("title", "content", "messages", "user_message"):
        assert not hasattr(record, forbidden)


@pytest.mark.parametrize(
    "event",
    [
        "session.created",
        "session.restored",
        "session.closed",
        "session.deleted",
        "session.pruned",
        "session.snapshot.failed",
    ],
)
def test_all_declared_events_are_accepted(event):
    response = _client().post(
        "/api/v2/sessions/lifecycle",
        json=_payload(event=event),
    )

    assert response.status_code == 204


def test_minimal_payload_is_accepted():
    response = _client().post(
        "/api/v2/sessions/lifecycle",
        json={
            "event": "session.closed",
            "reason": "window_exit",
            "session_id": "session-abc123",
        },
    )

    assert response.status_code == 204


@pytest.mark.parametrize(
    "payload",
    [
        _payload(event="session.unknown"),
        _payload(reason="unknown"),
        _payload(session_id="bad id"),
        _payload(session_id="bad\nid"),
        _payload(conversation_id="bad/id"),
        _payload(destroyed_count=1001),
        _payload(snapshot_bytes=10_000_001),
        {**_payload(), "title": "secret"},
        {**_payload(), "messages": [{"role": "user", "content": "secret"}]},
    ],
)
def test_invalid_payload_is_rejected(payload):
    response = _client().post("/api/v2/sessions/lifecycle", json=payload)

    assert response.status_code == 422
