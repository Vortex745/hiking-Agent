"""账号级模型配置：账号自持（未配置即为空），API Key 加密落盘。"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping

import config
from auth.paths import user_config_path
from datasource.errors import CredentialsUnreadable
from datasource.secrets import SecretBox

logger = logging.getLogger("ai-service.auth.user_config")

_LOCK = threading.RLock()
_SECRET_BOXES: dict[str, SecretBox] = {}
_VERSION = 1
_SECTION_FIELDS = config.section_field_settings_attrs()
_SECTIONS = tuple(_SECTION_FIELDS)
_DEFAULT_EMBEDDING_DIMENSIONS = 1536


def _secret_box(settings) -> SecretBox:
    key_path = Path(settings.rag_sqlite_db_path).with_name("user_config.key")
    cache_key = str(key_path)
    with _LOCK:
        box = _SECRET_BOXES.get(cache_key)
        if box is None:
            box = SecretBox.from_settings("", key_path)
            _SECRET_BOXES[cache_key] = box
        return box


def _read(settings, owner_id: str) -> dict[str, dict[str, Any]]:
    path = user_config_path(settings, owner_id)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("unreadable user config for %s; treating sections as unconfigured", owner_id)
        return {}
    sections = data.get("sections") if isinstance(data, dict) else None
    if not isinstance(sections, dict):
        return {}
    return {
        section: dict(fields)
        for section, fields in sections.items()
        if section in _SECTION_FIELDS and isinstance(fields, dict)
    }


def _write(settings, owner_id: str, sections: Mapping[str, Mapping[str, Any]]) -> None:
    path = user_config_path(settings, owner_id)
    if not sections:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("could not remove empty user config %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                {"version": _VERSION, "sections": sections},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def section_sources(settings, owner_id: str) -> dict[str, str]:
    overrides = _read(settings, owner_id)
    return {
        section: "account" if overrides.get(section) else "unset"
        for section in _SECTIONS
    }


def _section_defaults(settings) -> dict[str, Any]:
    """非敏感默认值；账号未配置时模型/地址/密钥一律为空。

    ``embedding_dimensions`` 取 Settings 类默认值而不是进程值：账号
    不能从 .env 继承 ``EMBEDDING_DIMENSIONS``。
    """
    return {
        "dimensions": getattr(
            type(settings), "embedding_dimensions", _DEFAULT_EMBEDDING_DIMENSIONS
        ),
        "mode": "off",
    }


def effective_settings(settings, owner_id: str) -> Any:
    """返回账号实际生效的 settings 副本。

    登录账号完全自持：模型 / Embedding / Rerank / 决策四段只读账号
    覆盖文件，未配置就是空，绝不回落到系统 .env。非模型字段（路径、
    沙箱、开关等）仍从进程 settings 继承。
    """
    effective = copy.copy(settings)
    if not owner_id:
        return effective

    # 内置 MCP 凭据只从账号自己的 mcp_builtin.json 取；system 的
    # AMAP_API_KEY 不进入账号作用域，避免日后被顺手读到。
    if hasattr(effective, "amap_api_key"):
        effective.amap_api_key = ""

    overrides = _read(settings, owner_id)
    defaults = _section_defaults(settings)
    for section, attrs in _SECTION_FIELDS.items():
        fields = overrides.get(section, {})
        for field, attr in attrs.items():
            if field == "api_key":
                continue
            if field in defaults:
                setattr(effective, attr, fields.get(field, defaults[field]))
            else:
                setattr(effective, attr, fields.get(field, ""))
        key = ""
        stored = fields.get("api_key")
        if stored:
            try:
                key = _secret_box(settings).decrypt(str(stored)).get("api_key", "")
            except (CredentialsUnreadable, ValueError):
                logger.warning(
                    "user api key unreadable section=%s user=%s",
                    section,
                    owner_id,
                )
        setattr(effective, attrs["api_key"], key)
    return effective


def save_overrides(
    settings, owner_id: str, updates: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """写入账号自己的配置；base_url 与 model 都为空表示清空该段。

    ``api_key`` 为空串表示保持账号已存值；清空分项会连同已存 key 一起删除。
    """
    with _LOCK:
        overrides = _read(settings, owner_id)
        changed: list[str] = []
        for section, fields in updates.items():
            attrs = _SECTION_FIELDS.get(section)
            if attrs is None:
                continue
            base_url = str(fields.get("base_url") or "").strip()
            model = str(fields.get("model") or "").strip()
            if not base_url and not model:
                if overrides.pop(section, None) is not None:
                    changed.append(f"{section}.cleared")
                continue

            current = dict(overrides.get(section, {}))
            current["base_url"] = base_url
            current["model"] = model
            changed.extend([f"{section}.base_url", f"{section}.model"])

            if "dimensions" in fields and "dimensions" in attrs:
                current["dimensions"] = int(fields["dimensions"])
                changed.append(f"{section}.dimensions")
            if "mode" in fields and "mode" in attrs:
                mode = str(fields.get("mode") or "off")
                if mode == "off":
                    current.pop("mode", None)
                else:
                    current["mode"] = mode
                changed.append(f"{section}.mode")

            api_key = fields.get("api_key")
            if isinstance(api_key, str) and api_key.strip():
                current["api_key"] = _secret_box(settings).encrypt(
                    {"api_key": api_key}
                )
                changed.append(f"{section}.api_key")

            overrides[section] = current
        _write(settings, owner_id, overrides)
        return changed


def user_config_updated_at(settings, owner_id: str) -> str | None:
    """账号配置文件最后修改时间（UTC ISO8601）；不存在为 None。"""
    from datetime import datetime, timezone

    try:
        mtime = user_config_path(settings, owner_id).stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def reset_overrides(settings, owner_id: str) -> None:
    with _LOCK:
        _write(settings, owner_id, {})
