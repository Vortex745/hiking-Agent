"""Five-field cron parsing and next-run calculation.

Only the subset the product needs is supported: ``*``, single values,
ranges, lists and ``/step``. Day-of-week is 0-6 with Sunday=0. When both
day-of-month and day-of-week are restricted, the standard Vixie cron rule
applies: either may match.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

_MAX_SEARCH_MINUTES = 366 * 24 * 60


@dataclass(frozen=True)
class _Field:
    low: int
    high: int
    values: frozenset[int]
    wildcard: bool

    def matches(self, value: int) -> bool:
        return value in self.values


def _parse_field(text: str, low: int, high: int) -> _Field:
    text = text.strip()
    if not text:
        raise ValueError("empty cron field")
    wildcard = text == "*"
    values: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"invalid cron field: {text!r}")
        step = 1
        base = part
        if "/" in part:
            base, step_text = part.split("/", 1)
            step = int(step_text)
            if step <= 0:
                raise ValueError(f"cron step must be positive: {part!r}")
        if base == "*":
            start, end = low, high
        elif "-" in base:
            start_text, end_text = base.split("-", 1)
            start, end = int(start_text), int(end_text)
        else:
            start = end = int(base)
        if start < low or end > high or start > end:
            raise ValueError(f"cron value out of range {low}-{high}: {part!r}")
        values.update(range(start, end + 1, step))
    return _Field(low=low, high=high, values=frozenset(values), wildcard=wildcard)


@dataclass(frozen=True)
class CronSchedule:
    minute: _Field
    hour: _Field
    day_of_month: _Field
    month: _Field
    day_of_week: _Field

    @classmethod
    def parse(cls, expression: str) -> "CronSchedule":
        fields = expression.split()
        if len(fields) != 5:
            raise ValueError("cron 表达式必须包含 5 个字段：分 时 日 月 周")
        return cls(
            minute=_parse_field(fields[0], 0, 59),
            hour=_parse_field(fields[1], 0, 23),
            day_of_month=_parse_field(fields[2], 1, 31),
            month=_parse_field(fields[3], 1, 12),
            day_of_week=_parse_field(fields[4], 0, 6),
        )

    def matches(self, moment: datetime) -> bool:
        if not (
            self.minute.matches(moment.minute)
            and self.hour.matches(moment.hour)
            and self.month.matches(moment.month)
        ):
            return False
        dom_ok = self.day_of_month.matches(moment.day)
        # Python Monday=0; cron Sunday=0.
        dow_ok = self.day_of_week.matches((moment.weekday() + 1) % 7)
        if self.day_of_month.wildcard or self.day_of_week.wildcard:
            return dom_ok and dow_ok
        return dom_ok or dow_ok

    def next_after(self, moment: datetime) -> datetime:
        candidate = moment.replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(_MAX_SEARCH_MINUTES):
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        raise ValueError(f"cron expression never matches within a year: {self!r}")


def validate_cron(expression: str) -> None:
    if not expression.strip():
        raise ValueError("cron 表达式为空；留空表示仅手动同步")
    CronSchedule.parse(expression)


def next_run_at(expression: str, after: datetime) -> datetime:
    return CronSchedule.parse(expression).next_after(after)
