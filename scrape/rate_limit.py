"""Token-bucket rate limiter for API-Football (and friends).

The free plan caps at 10 requests/minute. Earlier we had a fixed 0.25s
``_throttle()`` between requests on each ``ApiFootballClient`` instance —
that's two problems:

  1. Per-instance state means two clients in the same process can each fire
     10 req/min, doubling the effective rate and hitting 429.
  2. A fixed interval doesn't model the "10 req per rolling 60-second window"
     constraint well. The bucket lets us burst up to 10 in a few seconds (the
     common case for one batch operation), then makes us wait — which matches
     how the provider actually counts.

Token bucket semantics:
  * ``capacity`` tokens are available at startup
  * tokens refill at ``refill_per_minute`` per minute, continuously
  * ``acquire(n)`` blocks until ``n`` tokens are available, then consumes them
  * ``drain()`` forces the bucket to empty — used after a 429 response so the
    next ``acquire`` waits a full refill window before letting another
    request through

Clock + sleep are injectable so unit tests can run deterministically without
real ``time.sleep`` calls.
"""
from __future__ import annotations

import threading
import time
from typing import Callable


class TokenBucket:
    """Thread-safe token bucket. Inject ``clock``/``sleep`` for tests."""

    def __init__(
        self,
        *,
        capacity: int,
        refill_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if capacity < 0 or refill_per_minute < 0:
            raise ValueError("capacity and refill_per_minute must be ≥0")
        self.capacity = capacity
        self.refill_per_minute = refill_per_minute
        self._tokens = float(capacity)
        self._clock = clock
        self._sleep = sleep
        self._last_refill = clock()
        self._lock = threading.Lock()

    @property
    def disabled(self) -> bool:
        """A bucket with refill_per_minute=0 is a no-op (used in tests)."""
        return self.refill_per_minute <= 0 or self.capacity <= 0

    def acquire(self, n: int = 1) -> float:
        """Block until ``n`` tokens are available, consume them, return seconds waited."""
        if self.disabled:
            return 0.0
        total_waited = 0.0
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= n:
                    self._tokens -= n
                    return total_waited
                deficit = n - self._tokens
                # Tokens-per-second = refill_per_minute / 60.
                wait_s = deficit / (self.refill_per_minute / 60.0)
            self._sleep(wait_s)
            total_waited += wait_s

    def drain(self) -> None:
        """Empty the bucket — call after a 429 to enforce a full refill window."""
        with self._lock:
            self._tokens = 0.0
            self._last_refill = self._clock()

    def available(self) -> float:
        """Best-effort current token count. For diagnostics, not control flow."""
        with self._lock:
            self._refill_locked()
            return self._tokens

    def _refill_locked(self) -> None:
        now = self._clock()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        added = elapsed * (self.refill_per_minute / 60.0)
        if added <= 0:
            return
        self._tokens = min(float(self.capacity), self._tokens + added)
        self._last_refill = now


# Module-level bucket shared across all ApiFootballClient instances in the
# same process. Initialized lazily so tests can swap it without monkeypatching
# every client.
_GLOBAL_BUCKET: TokenBucket | None = None
_BUCKET_INIT_LOCK = threading.Lock()


def get_api_football_bucket() -> TokenBucket:
    global _GLOBAL_BUCKET
    if _GLOBAL_BUCKET is not None:
        return _GLOBAL_BUCKET
    with _BUCKET_INIT_LOCK:
        if _GLOBAL_BUCKET is None:
            from config.settings import get_settings
            settings = get_settings()
            rate = max(0, int(settings.api_football_rate_per_min))
            _GLOBAL_BUCKET = TokenBucket(capacity=rate, refill_per_minute=rate)
    return _GLOBAL_BUCKET


def reset_api_football_bucket(bucket: TokenBucket | None = None) -> None:
    """Replace the module-level bucket — for tests."""
    global _GLOBAL_BUCKET
    with _BUCKET_INIT_LOCK:
        _GLOBAL_BUCKET = bucket
