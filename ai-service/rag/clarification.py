"""票据 08：澄清双请求 + SSE 契约的澄清状态管理。

按 sse-clarification-contract 冻结结论实现：
- clarification_id = uuid5(NAMESPACE_URL, chat_id + ':' + sha256(question))
- 5 分钟过期（以 clarification_needed 事件的 expires_at 为准，不依赖进程内存）
- 重复首请求返回同 clarification_id（幂等）
- 取消后标记 cancelled，后续 resume 返回 410 Gone
- 持久化到 SQLite（data/clarifications.sqlite），服务重启后恢复仍成功

本模块只提供状态管理，不涉及 SSE 事件生成（由 api/clarification.py 承接）
和 LangGraph interrupt 触发（由 graph.py 的 request_clarification 节点承接）。
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from runtime_paths import runtime_dir

# 默认过期时间（秒）
DEFAULT_CLARIFICATION_TTL_SECONDS = 300  # 5 分钟

# clarification_id 命名空间（uuid5 需要）
_CLARIFICATION_NAMESPACE = uuid.NAMESPACE_URL


def clarification_id_for(chat_id: str, question: str) -> str:
    """计算 clarification_id（幂等键）。

    按 sse-clarification-contract 冻结结论：
    `clarification_id = uuid5(NAMESPACE_URL, chat_id + ':' + sha256(question))`

    同 chat_id + 同 question 重复提交首请求 → 返回同 clarification_id（幂等）

    Args:
        chat_id: 聊天会话标识
        question: 用户原始问题

    Returns:
        clarification_id 字符串（uuid5 格式）
    """
    question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
    name = f"{chat_id}:{question_hash}"
    return str(uuid.uuid5(_CLARIFICATION_NAMESPACE, name))


def _now() -> datetime:
    """当前 UTC 时间（带时区）。"""
    return datetime.now(timezone.utc)


def _parse_iso8601(s: str) -> datetime:
    """解析 ISO 8601 时间字符串（带时区）。"""
    return datetime.fromisoformat(s)


def _to_iso8601(dt: datetime) -> str:
    """格式化为 ISO 8601 时间字符串（带时区）。"""
    return dt.isoformat()


@dataclass
class ClarificationStatus:
    """澄清状态记录。

    Attributes:
        clarification_id: 幂等键（uuid5）
        chat_id: 聊天会话标识
        question: 用户原始问题
        questions: 澄清问题列表（推送给用户的选择题/补充题）
        expires_at: 过期时间（ISO 8601，首请求发出时间 + 5min）
        status: 状态（active/cancelled/expired/resumed）
        created_at: 创建时间（ISO 8601）
    """

    clarification_id: str
    chat_id: str
    question: str
    questions: list[str]
    expires_at: str  # ISO 8601
    status: str = "active"  # active | cancelled | expired | resumed
    created_at: str = field(default_factory=lambda: _to_iso8601(_now()))

    def is_expired(self, now: datetime | None = None) -> bool:
        """检查是否已过期。"""
        now = now or _now()
        return _parse_iso8601(self.expires_at) <= now

    def to_dict(self) -> dict[str, Any]:
        """转为字典（用于 SSE 事件 metadata）。"""
        return {
            "clarification_id": self.clarification_id,
            "chat_id": self.chat_id,
            "question": self.question,
            "questions": list(self.questions),
            "expires_at": self.expires_at,
            "status": self.status,
            "created_at": self.created_at,
        }


class ClarificationStore:
    """澄清状态存储（内存字典 + SQLite 持久化）。

    线程安全（内存字典用 lock 保护，SQLite 用 check_same_thread=False）。
    服务重启后从 SQLite 加载，恢复仍成功（不依赖进程内存）。

    持久化策略：
    - 内存字典为热路径（快速查询）
    - SQLite 为持久化备份（服务重启后加载）
    - 每次写操作同步到 SQLite（write-through）
    """

    def __init__(self, db_path: Path | None = None):
        """初始化澄清状态存储。

        Args:
            db_path: SQLite 文件路径；None 时使用默认 data/clarifications.sqlite
        """
        if db_path is None:
            db_path = self._default_db_path()
        self._db_path = str(db_path)
        self._db_path_obj = db_path
        self._lock = threading.RLock()
        self._store: dict[str, ClarificationStatus] = {}
        # 初始化 SQLite + 加载已有数据
        self._init_db()
        self._load_from_db()

    @staticmethod
    def _default_db_path() -> Path:
        """默认 SQLite 文件路径：data/clarifications.sqlite。"""
        return runtime_dir("CLARIFICATIONS_DIR", "data") / "clarifications.sqlite"

    def _init_db(self) -> None:
        """初始化 SQLite 表结构。"""
        db_path_obj = Path(self._db_path)
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS clarifications (
                    clarification_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    questions TEXT NOT NULL,  -- JSON array
                    expires_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _load_from_db(self) -> None:
        """从 SQLite 加载所有澄清记录到内存。"""
        import json

        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT clarification_id, chat_id, question, questions, "
                    "expires_at, status, created_at FROM clarifications"
                ).fetchall()
            for row in rows:
                cid, chat_id, question, questions_json, expires_at, status, created_at = row
                self._store[cid] = ClarificationStatus(
                    clarification_id=cid,
                    chat_id=chat_id,
                    question=question,
                    questions=json.loads(questions_json),
                    expires_at=expires_at,
                    status=status,
                    created_at=created_at,
                )

    def _persist(self, status: ClarificationStatus) -> None:
        """将单条记录写入 SQLite（upsert）。失败时抛异常，调用方负责回滚内存。"""
        import json

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO clarifications
                (clarification_id, chat_id, question, questions, expires_at, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    status.clarification_id,
                    status.chat_id,
                    status.question,
                    json.dumps(status.questions, ensure_ascii=False),
                    status.expires_at,
                    status.status,
                    status.created_at,
                ),
            )
            conn.commit()

    def create(
        self,
        chat_id: str,
        question: str,
        questions: list[str],
        ttl_seconds: int = DEFAULT_CLARIFICATION_TTL_SECONDS,
    ) -> ClarificationStatus:
        """创建澄清记录（幂等）。

        同 chat_id + 同 question 重复调用 → 返回同 clarification_id 的已有记录
        （不重新计算 expires_at，保留首次创建时间）。

        Args:
            chat_id: 聊天会话标识
            question: 用户原始问题
            questions: 澄清问题列表
            ttl_seconds: 过期时间（秒），默认 300（5 分钟）

        Returns:
            ClarificationStatus 记录
        """
        cid = clarification_id_for(chat_id, question)
        with self._lock:
            existing = self._store.get(cid)
            if existing is not None:
                # 幂等：返回已有记录（不更新 expires_at）
                return existing
            now = _now()
            expires_at = _to_iso8601(now + timedelta(seconds=ttl_seconds))
            status = ClarificationStatus(
                clarification_id=cid,
                chat_id=chat_id,
                question=question,
                questions=list(questions),
                expires_at=expires_at,
                status="active",
            )
            # 先持久化，成功后再写入内存（避免 disk full/corruption 时内存与 SQLite 不一致）
            self._persist(status)
            self._store[cid] = status
            return status

    def get(self, clarification_id: str) -> ClarificationStatus | None:
        """获取澄清记录。"""
        with self._lock:
            return self._store.get(clarification_id)

    def cancel(self, clarification_id: str) -> ClarificationStatus | None:
        """标记澄清为已取消（仅 active 状态可取消）。

        取消后后续 resume 返回 410 Gone + {error: "clarification_cancelled"}。
        状态机：仅 active 可 → cancelled；已 resumed/expired/cancelled 的记录保持原状态不变
        （防止非法转换破坏已完成对话或被恶意/误发请求篡改）。

        Args:
            clarification_id: 澄清幂等键

        Returns:
            更新后的 ClarificationStatus，或 None（记录不存在）
        """
        with self._lock:
            status = self._store.get(clarification_id)
            if status is None:
                return None
            if status.status != "active":
                # 已 resumed/cancelled/expired 的记录拒绝转换，返回当前状态
                return status
            original_status = status.status
            status.status = "cancelled"
            try:
                self._persist(status)
            except Exception:
                # 回滚内存状态，保持与 SQLite 一致
                status.status = original_status
                raise
            return status

    def mark_resumed(self, clarification_id: str) -> ClarificationStatus | None:
        """标记澄清为已恢复（仅 active 状态可恢复，CAS 原子语义）。

        状态机：仅 active 可 → resumed；已 resumed/cancelled/expired 的记录保持原状态不变。
        API 层应在调用前用 is_resumed/is_cancelled/is_expired 预检，但即便并发
        两个 resume 都通过预检，本方法的 lock + status 校验仍能保证只有一个成功。

        Args:
            clarification_id: 澄清幂等键

        Returns:
            更新后的 ClarificationStatus（status=='resumed' 表示本次成功，
            其他状态表示因并发/状态机约束未转换），或 None（记录不存在）
        """
        with self._lock:
            status = self._store.get(clarification_id)
            if status is None:
                return None
            if status.status != "active":
                return status
            original_status = status.status
            status.status = "resumed"
            try:
                self._persist(status)
            except Exception:
                # 回滚内存状态，保持与 SQLite 一致
                status.status = original_status
                raise
            return status

    def expire_active_for_chat(self, chat_id: str, except_clarification_id: str | None = None) -> None:
        """将指定 chat_id 下所有 active clarification 标记为 expired。

        用于实现契约§5："用户在澄清期间发新问题：视为放弃澄清，旧 clarification_id 自动过期"。
        调用时机：首请求触发新澄清时，将同 chat_id 下其他 active clarification 过期掉。

        Args:
            chat_id: 聊天会话标识
            except_clarification_id: 排除的 clarification_id（通常是刚创建的新 clarification）
        """
        with self._lock:
            for cid, status in list(self._store.items()):
                if status.chat_id != chat_id:
                    continue
                if cid == except_clarification_id:
                    continue
                if status.status != "active":
                    continue
                original_status = status.status
                status.status = "expired"
                try:
                    self._persist(status)
                except Exception:
                    status.status = original_status
                    raise

    def is_active(self, clarification_id: str) -> bool:
        """检查澄清是否仍可恢复（active 且未过期）。"""
        with self._lock:
            status = self._store.get(clarification_id)
            if status is None:
                return False
            if status.status != "active":
                return False
            if status.is_expired():
                return False
            return True

    def is_expired(self, clarification_id: str) -> bool:
        """检查澄清是否已过期。"""
        with self._lock:
            status = self._store.get(clarification_id)
            if status is None:
                return False
            return status.is_expired()

    def is_cancelled(self, clarification_id: str) -> bool:
        """检查澄清是否已取消。"""
        with self._lock:
            status = self._store.get(clarification_id)
            if status is None:
                return False
            return status.status == "cancelled"

    def is_resumed(self, clarification_id: str) -> bool:
        """检查澄清是否已恢复（用于幂等去重）。"""
        with self._lock:
            status = self._store.get(clarification_id)
            if status is None:
                return False
            return status.status == "resumed"

    def clear(self) -> None:
        """清空所有澄清记录（测试用）。"""
        with self._lock:
            self._store.clear()
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("DELETE FROM clarifications")
                conn.commit()


# 全局单例（懒加载）
_global_store: ClarificationStore | None = None
_global_store_lock = threading.Lock()


def get_clarification_store() -> ClarificationStore:
    """获取全局 ClarificationStore 单例。

    单例模式避免重复初始化 SQLite 连接。
    服务重启后单例会重新加载 SQLite 数据，恢复仍成功。
    """
    global _global_store
    if _global_store is None:
        with _global_store_lock:
            if _global_store is None:
                _global_store = ClarificationStore()
    return _global_store


def reset_clarification_store(store: ClarificationStore | None = None) -> None:
    """重置全局 ClarificationStore（测试用）。

    Args:
        store: 指定的 store 实例；None 时重置为 None（下次 get 重新创建）
    """
    global _global_store
    with _global_store_lock:
        _global_store = store


__all__ = [
    "DEFAULT_CLARIFICATION_TTL_SECONDS",
    "ClarificationStatus",
    "ClarificationStore",
    "clarification_id_for",
    "get_clarification_store",
    "reset_clarification_store",
]
