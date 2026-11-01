"""账号服务：注册、登录、JWT 校验，以及登录账号的稳定 subject。"""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass

from agent_v2.repositories.memory import MemoryStore

from auth.rules import validate_login_id, validate_password
from auth.security import (
    create_token,
    decode_token,
    hash_password,
    verify_password,
)
from auth.store import AuthStore, DuplicateLoginIdError

# 未知 ID 也执行一次同成本校验，避免用响应时间枚举账号。
_DUMMY_PASSWORD_HASH = hash_password("not-a-real-account-password")


@dataclass(frozen=True, slots=True)
class AuthUser:
    user_id: str
    login_id: str
    subject_id: str
    created_at: str


@dataclass(frozen=True, slots=True)
class AuthSession:
    user: AuthUser
    token: str


class AuthError(Exception):
    status_code = 400
    code = "auth_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DuplicateLoginId(AuthError):
    status_code = 409
    code = "duplicate_login_id"


class InvalidCredentials(AuthError):
    status_code = 401
    code = "invalid_credentials"


class InvalidToken(AuthError):
    status_code = 401
    code = "unauthenticated"


def _user_from_row(row: dict) -> AuthUser:
    return AuthUser(
        user_id=row["user_id"],
        login_id=row["login_id"],
        subject_id=row["subject_id"],
        created_at=row["created_at"],
    )


def _subject_token_hash(user_id: str) -> str:
    """登录账号的 subject 由 user_id 决定，重启/换浏览器都不漂移。"""
    return hashlib.sha256(f"user:{user_id}".encode("utf-8")).hexdigest()


class AuthService:
    def __init__(
        self,
        store: AuthStore,
        memory_store: MemoryStore,
        *,
        secret: str,
        ttl_seconds: int,
    ) -> None:
        self.store = store
        self.memory_store = memory_store
        self.secret = secret
        self.ttl_seconds = int(ttl_seconds)
        self._register_lock = threading.Lock()

    def close(self) -> None:
        self.store.close()
        self.memory_store.close()

    def register(self, login_id: str, password: str) -> AuthSession:
        validate_login_id(login_id)
        validate_password(password)
        with self._register_lock:
            if self.store.get_by_login_id(login_id) is not None:
                raise DuplicateLoginId("该登录 ID 已被使用（区分大小写）")
            user_id = uuid.uuid4().hex
            subject_id = self.memory_store.get_or_create_subject(
                _subject_token_hash(user_id)
            )
            try:
                row = self.store.create_user(
                    user_id=user_id,
                    login_id=login_id,
                    password_hash=hash_password(password),
                    subject_id=subject_id,
                )
            except DuplicateLoginIdError as exc:
                raise DuplicateLoginId("该登录 ID 已被使用（区分大小写）") from exc
            self.store.touch_login(user_id)
        return self._session(_user_from_row(row))

    def login(self, login_id: str, password: str) -> AuthSession:
        row = self.store.get_by_login_id(login_id)
        encoded = row["password_hash"] if row is not None else _DUMMY_PASSWORD_HASH
        valid = verify_password(password, encoded)
        if row is None or not valid:
            raise InvalidCredentials("登录 ID 或密码错误")
        self.store.touch_login(row["user_id"])
        return self._session(_user_from_row(row))

    def authenticate(self, token: str) -> AuthUser:
        try:
            payload = decode_token(token, self.secret)
        except ValueError as exc:
            raise InvalidToken("登录状态已失效，请重新登录") from exc
        row = self.store.get_by_id(payload["sub"])
        if row is None:
            raise InvalidToken("登录状态已失效，请重新登录")
        return _user_from_row(row)

    def get_user(self, user_id: str) -> AuthUser | None:
        row = self.store.get_by_id(user_id)
        return _user_from_row(row) if row else None

    def _session(self, user: AuthUser) -> AuthSession:
        token = create_token(
            user_id=user.user_id,
            subject_id=user.subject_id,
            secret=self.secret,
            ttl_seconds=self.ttl_seconds,
        )
        return AuthSession(user=user, token=token)
