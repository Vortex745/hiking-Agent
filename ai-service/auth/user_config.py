"""账号级模型配置覆盖层：只存与系统默认不同的字段，API Key 加密落盘。"""

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
        logger.warning("unreadable user config for %s; falling back to system defaults", owner_id)
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
        section: "account" if overrides.get(section) else "system"
        for section in _SECTIONS
    }


def effective_settings(settings, owner_id: str) -> Any:
    """返回账号实际生效的 settings 副本；未覆盖字段继承系统默认。"""
    effective = copy.copy(settings)
    overrides = _read(settings, owner_id)
    for section, fields in overrides.items():
        attrs = _SECTION_FIELDS[section]
        base_attr = attrs.get("base_url")
        key_attr = attrs.get("api_key")
        base_changed = bool(
            base_attr
            and "base_url" in fields
            and fields["base_url"] != getattr(settings, base_attr, None)
        )
        account_key_applied = False
        for field, value in fields.items():
            attr = attrs.get(field)
            if attr is None:
                continue
            if field == "api_key":
                try:
                    decrypted = _secret_box(settings).decrypt(str(value))
                except (CredentialsUnreadable, ValueError):
                    logger.warning(
                        "user api key unreadable section=%s user=%s",
                        section,
                        owner_id,
                    )
                    if base_changed and key_attr:
                        setattr(effective, key_attr, "")
                    continue
                value = decrypted.get("api_key", "")
                account_key_applied = True
            setattr(effective, attr, value)
        # 账号把 base_url 指向别的主机时，绝不能把系统密钥转发过去。
        if base_changed and key_attr and not account_key_applied:
            setattr(effective, key_attr, "")
    return effective


def save_overrides(
    settings, owner_id: str, updates: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """只写入与系统默认不同的字段；api_key 为空串表示保持账号已存值。"""
    with _LOCK:
        overrides = _read(settings, owner_id)
        changed: list[str] = []
        for section, fields in updates.items():
            attrs = _SECTION_FIELDS.get(section)
            if attrs is None:
                continue
            current = dict(overrides.get(section, {}))
            for field, value in fields.items():
                attr = attrs.get(field)
                if attr is None:
                    continue
                if field == "api_key":
                    if value is None:
                        continue
                    if isinstance(value, str) and not value.strip():
                        always = current.get("api_key")
                        if always:
                            changed.append(f"{section}.api_key")
                        continue
                    current["api_key"] = _secret_box(settings).encrypt({"api_key": str(value)})
                    changed.append(f"{section}.api_key")
                    continue
                default = getattr(settings, attr, None)
                if value == default:
                    current.pop(field, None)
                else:
                    current[field] = value
                    changed.append(f"{section}.{field}")
            if current:
                overrides[section] = current
            else:
                overrides.pop(section, None)
        _write(settings, owner_id, overrides)
        return changed


def reset_overrides(settings, owner_id: str) -> None:
    with _LOCK:
        _write(settings, owner_id, {})
