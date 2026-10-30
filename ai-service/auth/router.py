"""注册、登录、退出与当前账号。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from auth.deps import COOKIE_NAME, get_current_user
from auth.rules import validate_login_id, validate_password
from auth.service import (
    AuthService,
    AuthUser,
    DuplicateLoginId,
    InvalidCredentials,
)

auth_router = APIRouter(prefix="/auth")


class CredentialsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    login_id: str = Field(..., min_length=2, max_length=6)
    password: str = Field(..., min_length=8, max_length=15)

    @field_validator("login_id")
    @classmethod
    def _login_id(cls, value: str) -> str:
        return validate_login_id(value)

    @field_validator("password")
    @classmethod
    def _password(cls, value: str) -> str:
        return validate_password(value)


class AuthUserResponse(BaseModel):
    user_id: str
    login_id: str
    created_at: str


class AuthResponse(AuthUserResponse):
    access_token: str
    token_type: str = "bearer"


def _service(request: Request) -> AuthService:
    service = getattr(request.app.state, "auth_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="登录服务不可用")
    return service


def _set_cookie(request: Request, response: Response, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=_service(request).ttl_seconds,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        path="/",
    )


def _payload(session) -> AuthResponse:
    user: AuthUser = session.user
    return AuthResponse(
        user_id=user.user_id,
        login_id=user.login_id,
        created_at=user.created_at,
        access_token=session.token,
    )


@auth_router.post("/register", response_model=AuthResponse, status_code=201)
def register(
    body: CredentialsRequest, request: Request, response: Response
) -> AuthResponse:
    try:
        session = _service(request).register(body.login_id, body.password)
    except DuplicateLoginId as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _set_cookie(request, response, session.token)
    return _payload(session)


@auth_router.post("/login", response_model=AuthResponse)
def login(
    body: CredentialsRequest, request: Request, response: Response
) -> AuthResponse:
    try:
        session = _service(request).login(body.login_id, body.password)
    except InvalidCredentials as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    _set_cookie(request, response, session.token)
    return _payload(session)


@auth_router.post("/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@auth_router.get("/me", response_model=AuthUserResponse)
def me(user: AuthUser = Depends(get_current_user)) -> AuthUserResponse:
    return AuthUserResponse(
        user_id=user.user_id,
        login_id=user.login_id,
        created_at=user.created_at,
    )
