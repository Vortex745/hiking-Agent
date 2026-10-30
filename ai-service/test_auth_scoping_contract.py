"""账号身份复用现有 subject：会话所有权与匿名 cookie 不串号。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

from agent_v2.memory_service import MemoryService
from agent_v2.repositories.memory import MemoryStore
from auth.service import AuthService
from auth.store import AuthStore


def _service(tmp_path: Path) -> tuple[AuthService, MemoryService]:
    memory_store = MemoryStore(str(tmp_path / "agent_memory.sqlite"))
    auth = AuthService(
        AuthStore(str(tmp_path / "auth.sqlite")),
        memory_store,
        secret="test-secret-0123456789abcdef0123456789",
        ttl_seconds=3600,
    )
    return auth, MemoryService(memory_store)


def test_two_accounts_get_distinct_subjects_and_conversations_do_not_cross(
    tmp_path: Path,
) -> None:
    auth, memory = _service(tmp_path)
    alice = auth.register("Alice1", "password123").user
    bob = auth.register("Bob123", "password123").user

    assert alice.subject_id != bob.subject_id
    assert memory.claim_conversation(alice.subject_id, "chat-alice")
    assert memory.owns_conversation(alice.subject_id, "chat-alice")
    assert not memory.owns_conversation(bob.subject_id, "chat-alice")


def test_logged_in_subject_wins_over_anonymous_cookie(tmp_path: Path) -> None:
    from api.memory_v2 import resolve_request_subject

    auth, memory = _service(tmp_path)
    user = auth.register("Alice1", "password123").user

    request = SimpleNamespace(
        cookies={"hiking_subject": "anonymous-token"},
        state=SimpleNamespace(),
        app=SimpleNamespace(
            state=SimpleNamespace(agent_v2_runtime=SimpleNamespace(memory=memory))
        ),
    )
    request.state.subject_id = user.subject_id

    resolved_memory, resolution = resolve_request_subject(request)
    assert resolved_memory is memory
    assert resolution.subject_id == user.subject_id
    assert resolution.issued is False
