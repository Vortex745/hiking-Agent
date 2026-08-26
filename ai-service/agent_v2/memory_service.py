"""Host-side long-term memory service for Chat v2."""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass

from agent_v2.repositories.memory import MemoryStore


SUBJECT_COOKIE_NAME = "hiking_subject"


def generate_subject_token() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()


def is_valid_subject_token(token: str | None) -> bool:
    if not token:
        return False
    try:
        decoded = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
    except (ValueError, TypeError):
        return False
    return len(decoded) == 32 and len(token) == 43


def hash_subject_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SubjectResolution:
    subject_id: str
    token: str
    issued: bool


class MemoryService:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def close(self) -> None:
        self.store.close()

    def resolve_subject(self, token: str | None) -> SubjectResolution:
        issued = not is_valid_subject_token(token)
        token = generate_subject_token() if issued else str(token)
        subject_id = self.store.get_or_create_subject(hash_subject_token(token))
        return SubjectResolution(subject_id, token, issued)

    def rotate_subject(self, subject_id: str) -> SubjectResolution:
        token = generate_subject_token()
        self.store.rotate_token(subject_id, hash_subject_token(token))
        return SubjectResolution(subject_id, token, True)

    def memory_enabled(self, subject_id: str) -> bool:
        return self.store.memory_enabled(subject_id)

    def set_memory_enabled(self, subject_id: str, enabled: bool) -> None:
        self.store.set_memory_enabled(subject_id, enabled)

    def claim_conversation(self, subject_id: str, conversation_id: str) -> bool:
        return self.store.claim_conversation(subject_id, conversation_id)

    def owns_conversation(self, subject_id: str, conversation_id: str) -> bool:
        return self.store.owns_conversation(subject_id, conversation_id)

    def save_preference(self, subject_id: str, content: str) -> str | None:
        return self.store.save_preference(subject_id, content)

    def list_active_memories(
        self, subject_id: str, *, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        return self.store.list_active_memories(subject_id, limit=limit, offset=offset)

    def count_active_memories(self, subject_id: str) -> int:
        return self.store.count_active_memories(subject_id)

    def create_candidate(self, subject_id: str, **candidate: str) -> dict | None:
        return self.store.create_candidate(subject_id, **candidate)

    def decide_candidate(
        self,
        subject_id: str,
        candidate_id: str,
        decision: str,
        decision_request_id: str,
    ) -> tuple[dict, bool] | None:
        return self.store.decide_candidate(
            subject_id, candidate_id, decision, decision_request_id
        )

    def update_memory(
        self,
        subject_id: str,
        memory_id: str,
        *,
        content: str,
        category: str,
        ttl_days: int | None,
    ) -> dict | None:
        return self.store.update_memory(
            subject_id,
            memory_id,
            content=content,
            category=category,
            ttl_days=ttl_days,
        )

    def delete_memory(self, subject_id: str, memory_id: str) -> bool:
        return self.store.delete_memory(subject_id, memory_id)

    def delete_all_memories(self, subject_id: str) -> int:
        return self.store.delete_all_memories(subject_id)
