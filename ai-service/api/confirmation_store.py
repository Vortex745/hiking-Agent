import time
import uuid
import threading
import logging
from dataclasses import asdict, dataclass
from typing import Optional

logger = logging.getLogger("ai-service.confirmation")
REDIS_CONFIRMATION_STORE_KEY = "agent:confirmations:pending"


@dataclass
class PendingConfirmation:
    """一条待用户确认的工具调用记录。"""

    confirmation_id: str
    tool_name: str
    args: dict
    chat_id: str
    step: int
    status: str = "pending"  # pending | confirmed | rejected
    created_at: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()


class ConfirmationStore:
    """内存存储，管理所有待用户确认的工具调用。

    提供添加、查询、确认/拒绝操作，支持按 chat_id 查询。
    线程安全（threading.Lock），适用于多 worker 场景。
    """

    def __init__(self, persistent_store=None):
        self._lock = threading.Lock()
        self._data: dict[str, PendingConfirmation] = {}
        self._persistent_store = persistent_store
        self._load()

    def _load(self) -> None:
        if self._persistent_store is None:
            return
        try:
            raw = self._persistent_store.get_json(REDIS_CONFIRMATION_STORE_KEY) or {}
            self._data = {
                cid: PendingConfirmation(**record)
                for cid, record in raw.items()
                if isinstance(record, dict)
            }
        except Exception as exc:
            logger.warning("Failed to load confirmation store from Redis: %s", exc)
            self._data = {}

    def _save(self) -> None:
        if self._persistent_store is None:
            return
        try:
            self._persistent_store.set_json(
                REDIS_CONFIRMATION_STORE_KEY,
                {cid: asdict(record) for cid, record in self._data.items()},
            )
        except Exception as exc:
            logger.warning("Failed to save confirmation store to Redis: %s", exc)

    def add(
        self,
        tool_name: str,
        args: dict,
        chat_id: str,
        step: int,
        *,
        confirmation_id: Optional[str] = None,
    ) -> str:
        """添加一条待确认记录，返回 confirmation_id。"""
        cid = confirmation_id or str(uuid.uuid4())
        with self._lock:
            self._data[cid] = PendingConfirmation(
                confirmation_id=cid,
                tool_name=tool_name,
                args=args,
                chat_id=chat_id,
                step=step,
            )
            self._save()
        return cid

    def get(self, confirmation_id: str) -> Optional[PendingConfirmation]:
        """按 ID 查询确认记录。"""
        with self._lock:
            return self._data.get(confirmation_id)

    def confirm(self, confirmation_id: str) -> bool:
        """标记为已确认。返回 False 表示记录不存在或状态异常。"""
        with self._lock:
            rec = self._data.get(confirmation_id)
            if rec is None or rec.status != "pending":
                return False
            rec.status = "confirmed"
            self._save()
            return True

    def reject(self, confirmation_id: str) -> bool:
        """标记为已拒绝。返回 False 表示记录不存在或状态异常。"""
        with self._lock:
            rec = self._data.get(confirmation_id)
            if rec is None or rec.status != "pending":
                return False
            rec.status = "rejected"
            self._save()
            return True

    def get_pending_by_chat(self, chat_id: str) -> list[PendingConfirmation]:
        """获取某次对话中所有待确认的记录。"""
        with self._lock:
            return [
                rec
                for rec in self._data.values()
                if rec.chat_id == chat_id and rec.status == "pending"
            ]

    def cleanup_expired(self, max_age: float = 3600) -> int:
        """清理超过 max_age 秒的过期记录。返回清理数量。"""
        now = time.time()
        with self._lock:
            expired = [
                cid
                for cid, rec in self._data.items()
                if now - rec.created_at > max_age
            ]
            for cid in expired:
                del self._data[cid]
            self._save()
        return len(expired)


# ── 模块级单例 ──────────────────────────────────────────────
_store: Optional[ConfirmationStore] = None


def get_store() -> ConfirmationStore:
    """获取全局唯一的 ConfirmationStore 实例。"""
    global _store
    if _store is None:
        persistent_store = None
        try:
            from config import settings
            from memory.redis_memory import RedisCommandStore

            if settings.redis_configured:
                persistent_store = RedisCommandStore()
        except Exception as exc:
            logger.warning("ConfirmationStore Redis persistence unavailable: %s", exc)
        _store = ConfirmationStore(persistent_store=persistent_store)
    return _store
