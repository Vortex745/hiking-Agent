"""Controlled memory-tool input parsing used by the Chat v2 stub."""

from __future__ import annotations

import re


_SENSITIVE_MARKERS = (
    "位置", "住址", "地址", "健康", "疾病", "过敏", "行程", "日期", "时间",
    "联系人", "电话", "手机", "微信", "location", "health", "schedule", "contact",
)

_REMEMBER_PREFIXES = ("请记住", "记住", "remember ")


def _remembered_value(message: str) -> str | None:
    text = message.strip()
    lowered = text.lower()
    for prefix in _REMEMBER_PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip(" ：:") or None
    return None


def extract_sensitive_memory(message: str) -> dict[str, str] | None:
    value = _remembered_value(message)
    if value is None:
        return None
    lowered = value.lower()
    if any(marker in lowered for marker in ("电话", "手机", "微信", "联系人", "contact")) \
            or re.search(r"(?<!\d)1\d{10}(?!\d)", value):
        category = "contact"
    elif any(marker in lowered for marker in ("健康", "疾病", "过敏", "病史", "health")):
        category = "health"
    elif (
        any(marker in lowered for marker in ("行程", "出发时间", "旅行日期", "schedule"))
        or re.search(r"\d{1,2}月\d{1,2}[日号].*(?:去|到|徒步|旅行)", value)
    ):
        category = "trip"
    elif any(marker in lowered for marker in ("住在", "住址", "地址", "位置", "location")):
        category = "location"
    else:
        return None
    return {
        "category": category,
        "content": value,
        "preview": value[:80],
        "sensitivity": "sensitive",
    }


def extract_ordinary_preference(message: str) -> str | None:
    value = _remembered_value(message)
    if value is None:
        return None
    lowered = value.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        return None
    return value


def asks_for_preferences(message: str) -> bool:
    lowered = message.strip().lower()
    return "偏好" in lowered or "preference" in lowered
