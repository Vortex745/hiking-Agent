"""登录 ID / 密码规则，路由与账号服务共用同一份。"""

from __future__ import annotations

import re

LOGIN_ID_RE = re.compile(r"^[A-Za-z0-9_]{2,6}$")
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 15


def validate_login_id(value: str) -> str:
    if not isinstance(value, str) or not LOGIN_ID_RE.fullmatch(value):
        raise ValueError("登录 ID 需为 2-6 位字母、数字或下划线")
    return value


def validate_password(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("密码格式不正确")
    if not PASSWORD_MIN_LENGTH <= len(value) <= PASSWORD_MAX_LENGTH:
        raise ValueError("密码需为 8-15 个字符")
    if any(char.isspace() for char in value):
        raise ValueError("密码不能包含空白字符")
    return value
