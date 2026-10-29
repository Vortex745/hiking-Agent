"""Small token bucket used by connectors with per-minute quotas."""

from __future__ import annotations

import time
from typing import Callable


class TokenBucket:
    def __init__(
        self,
        rate_per_minute: int,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._rate = max(1.0, float(rate_per_minute))
        self._tokens = 1.0
        self._last = clock_fn()
        self._sleep = sleep_fn
        self._clock = clock_fn

    def acquire(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._tokens = min(self._rate, self._tokens + elapsed * self._rate / 60.0)
        self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return
        wait = (1.0 - self._tokens) * 60.0 / self._rate
        self._sleep(wait)
        self._tokens = 0.0
        self._last = self._clock()
