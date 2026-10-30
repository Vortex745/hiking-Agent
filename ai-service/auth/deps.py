"""认证依赖：从 Cookie 或 Bearer 中解析登录账号。"""

from __future__ import annotations

from fastapi import HTTPException, Request

from auth.service import AuthError, AuthUser, InvalidToken

COOKIE_NAME = "hiking_auth"


def extract_token(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        return token or None
    return request.cookies.get(COOKIE_NAME)


def _unauthorized(message: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail={"code": "unauthenticated", "message": message},
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(request: Request) -> AuthUser:
    service = getattr(request.app.state, "auth_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="登录服务不可用")
    token = extract_token(request)
    if not token:
        raise _unauthorized("请先登录")
    try:
        user = service.authenticate(token)
    except (AuthError, InvalidToken) as exc:
        raise _unauthorized(getattr(exc, "message", "登录状态已失效，请重新登录")) from exc
    request.state.user_id = user.user_id
    request.state.login_id = user.login_id
    request.state.subject_id = user.subject_id
    return user


def get_optional_user(request: Request) -> AuthUser | None:
    token = extract_token(request)
    service = getattr(request.app.state, "auth_service", None)
    if not token or service is None:
        return None
    try:
        user = service.authenticate(token)
    except AuthError:
        return None
    request.state.user_id = user.user_id
    request.state.login_id = user.login_id
    request.state.subject_id = user.subject_id
    return user


def current_user_id(request: Request) -> str:
    """端点内读取账号标识；主应用未通过鉴权时拒绝，裸挂载测试退化为空 owner。"""
    user_id = str(getattr(request.state, "user_id", "") or "")
    if user_id:
        return user_id
    if getattr(request.app.state, "auth_service", None) is not None:
        raise _unauthorized("请先登录")
    return ""
