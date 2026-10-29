"""Local-time helpers for cron matching and persisted timestamps."""

from __future__ import annotations

from datetime import datetime


def now() -> datetime:
    return datetime.now().astimezone()


def now_iso() -> str:
    return now().isoformat()


def parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


def iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.astimezone()
    return value.isoformat()
