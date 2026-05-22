"""Tests for the API-Football token-bucket rate limiter."""
from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from scrape.rate_limit import TokenBucket


class FakeClock:
    """Deterministic clock + sleep that advances on demand."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        # Recording the sleep but actually advancing time is what makes the
        # bucket's "block until tokens available" loop terminate.
        self.slept.append(seconds)
        self.now += seconds

    @property
    def total_slept(self) -> float:
        return sum(self.slept)


# ---------------------------------------------------------------------------
# Basic mechanics
# ---------------------------------------------------------------------------

def test_acquire_does_not_block_when_bucket_is_full() -> None:
    clk = FakeClock()
    bucket = TokenBucket(capacity=10, refill_per_minute=10, clock=clk.time, sleep=clk.sleep)
    for _ in range(10):
        waited = bucket.acquire()
        assert waited == 0.0
    assert clk.total_slept == 0.0


def test_acquire_blocks_when_bucket_is_empty() -> None:
    clk = FakeClock()
    bucket = TokenBucket(capacity=10, refill_per_minute=10, clock=clk.time, sleep=clk.sleep)
    # Drain the bucket by burning 10 tokens with no time passing.
    for _ in range(10):
        bucket.acquire()
    # The 11th call must wait at least 6s (one token at 10/min = 1 per 6s).
    waited = bucket.acquire()
    assert waited == pytest.approx(6.0, rel=1e-3)


def test_bucket_refills_over_time() -> None:
    clk = FakeClock()
    bucket = TokenBucket(capacity=10, refill_per_minute=10, clock=clk.time, sleep=clk.sleep)
    # Burn all 10 tokens.
    for _ in range(10):
        bucket.acquire()
    # Advance the clock 30s — that's 5 tokens at 10/min.
    clk.now += 30.0
    # Five consecutive acquires should not block.
    for _ in range(5):
        assert bucket.acquire() == 0.0
    # The sixth should block briefly (less than one full refill).
    waited = bucket.acquire()
    assert waited > 0


def test_drain_forces_empty() -> None:
    clk = FakeClock()
    bucket = TokenBucket(capacity=10, refill_per_minute=10, clock=clk.time, sleep=clk.sleep)
    bucket.drain()
    assert bucket.available() == pytest.approx(0.0, abs=1e-6)
    # And the next acquire blocks a full 6s.
    waited = bucket.acquire()
    assert waited == pytest.approx(6.0, rel=1e-3)


def test_capacity_caps_refill() -> None:
    """Even after sitting idle for an hour, the bucket can't hold more than ``capacity``."""
    clk = FakeClock()
    bucket = TokenBucket(capacity=10, refill_per_minute=10, clock=clk.time, sleep=clk.sleep)
    bucket.drain()
    clk.now += 3600.0  # 1h of refill at 10/min would be 600 tokens — capped at 10.
    for _ in range(10):
        assert bucket.acquire() == 0.0
    # 11th call must block.
    assert bucket.acquire() > 0


def test_disabled_bucket_is_a_noop() -> None:
    clk = FakeClock()
    bucket = TokenBucket(capacity=0, refill_per_minute=0, clock=clk.time, sleep=clk.sleep)
    assert bucket.disabled
    for _ in range(100):
        assert bucket.acquire() == 0.0
    assert clk.total_slept == 0.0


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        TokenBucket(capacity=-1, refill_per_minute=10)
    with pytest.raises(ValueError):
        TokenBucket(capacity=10, refill_per_minute=-1)


# ---------------------------------------------------------------------------
# ApiFootballClient 429 handling — uses module-level bucket
# ---------------------------------------------------------------------------

def _fake_response(status_code: int, json_body: Any | None = None):
    resp = mock.Mock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    if 400 <= status_code < 600:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=mock.Mock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def client_with_disabled_bucket(monkeypatch):
    """Build an ApiFootballClient with rate limiting disabled + zero backoff sleep."""
    from scrape import api_football
    from scrape.rate_limit import TokenBucket, reset_api_football_bucket

    monkeypatch.setenv("FOOTBALL_API_KEY", "x" * 32)
    reset_api_football_bucket(TokenBucket(capacity=0, refill_per_minute=0))
    client = api_football.ApiFootballClient()
    client._backoff_s = 0.0  # don't actually sleep in tests
    yield client
    reset_api_football_bucket(None)


def test_client_returns_payload_on_success(client_with_disabled_bucket, monkeypatch) -> None:
    payload = {"response": [{"x": 1}], "errors": {}}
    monkeypatch.setattr(
        "scrape.api_football.httpx.get",
        mock.Mock(return_value=_fake_response(200, payload)),
    )
    got = client_with_disabled_bucket.get("/fixtures")
    assert got == payload


def test_client_retries_once_on_http_429(client_with_disabled_bucket, monkeypatch) -> None:
    """First call returns 429, second call (after backoff) returns 200."""
    good = _fake_response(200, {"response": [], "errors": {}})
    bad = _fake_response(429)
    mock_get = mock.Mock(side_effect=[bad, good])
    monkeypatch.setattr("scrape.api_football.httpx.get", mock_get)

    result = client_with_disabled_bucket.get("/fixtures")
    assert result == {"response": [], "errors": {}}
    assert mock_get.call_count == 2


def test_client_retries_once_on_soft_rate_limit_body(client_with_disabled_bucket, monkeypatch) -> None:
    """200 OK with errors body containing 'rateLimit' → drain + retry."""
    soft = _fake_response(200, {"errors": {"rateLimit": "Too many requests..."}})
    good = _fake_response(200, {"response": [{"x": 1}], "errors": {}})
    mock_get = mock.Mock(side_effect=[soft, good])
    monkeypatch.setattr("scrape.api_football.httpx.get", mock_get)

    result = client_with_disabled_bucket.get("/fixtures")
    assert result["response"] == [{"x": 1}]
    assert mock_get.call_count == 2


def test_client_raises_on_persistent_soft_rate_limit(client_with_disabled_bucket, monkeypatch) -> None:
    """Two soft rate-limit responses in a row → RuntimeError surfaces."""
    soft = _fake_response(200, {"errors": {"rateLimit": "Too many requests..."}})
    mock_get = mock.Mock(side_effect=[soft, soft])
    monkeypatch.setattr("scrape.api_football.httpx.get", mock_get)

    with pytest.raises(RuntimeError, match="still rate-limited"):
        client_with_disabled_bucket.get("/fixtures")
    assert mock_get.call_count == 2


def test_client_does_not_retry_non_rate_limit_errors(client_with_disabled_bucket, monkeypatch) -> None:
    """An ``errors`` body that isn't about rate limits should NOT be retried."""
    bad = _fake_response(200, {"errors": {"plan": "Free plans do not have access"}})
    mock_get = mock.Mock(return_value=bad)
    monkeypatch.setattr("scrape.api_football.httpx.get", mock_get)

    with pytest.raises(RuntimeError, match="returned errors"):
        client_with_disabled_bucket.get("/fixtures")
    assert mock_get.call_count == 1  # no retry


def test_is_rate_limit_error_detects_various_shapes() -> None:
    from scrape.api_football import _is_rate_limit_error

    assert _is_rate_limit_error({"rateLimit": "..."})
    assert _is_rate_limit_error({"requests": "..."})
    assert _is_rate_limit_error([{"rateLimit": "..."}])
    assert _is_rate_limit_error({"error": "your rate limit is 10 / minute"})
    assert not _is_rate_limit_error({"plan": "free plan only"})
    assert not _is_rate_limit_error({})
    assert not _is_rate_limit_error([])


def test_is_daily_quota_error_distinguishes_from_per_minute() -> None:
    from scrape.api_football import _is_daily_quota_error

    # Daily-quota wording (matched)
    assert _is_daily_quota_error({"requests": "You have reached the request limit for the day."})
    assert _is_daily_quota_error({"requests": "Daily limit exceeded"})
    # Per-minute wording (NOT matched — we don't want to circuit-break for these)
    assert not _is_daily_quota_error({"rateLimit": "Too many requests. Your rate limit is 10 / minute"})
    assert not _is_daily_quota_error({})
    assert not _is_daily_quota_error(None)


def test_daily_quota_sets_circuit_breaker_on_soft_rate_limit(client_with_disabled_bucket, monkeypatch) -> None:
    """Soft rate-limit with daily-quota wording → flag set, no retry, short-circuit future calls."""
    from scrape import api_football

    monkeypatch.setattr(api_football, "reset_daily_quota_flag", api_football.reset_daily_quota_flag)
    api_football.reset_daily_quota_flag()

    quota_body = {"errors": {"requests": "You have reached the request limit for the day."}}
    mock_get = mock.Mock(return_value=_fake_response(200, quota_body))
    monkeypatch.setattr("scrape.api_football.httpx.get", mock_get)

    # First call: hits quota → raises, sets flag.
    with pytest.raises(RuntimeError, match="daily quota exhausted"):
        client_with_disabled_bucket.get("/fixtures")
    assert mock_get.call_count == 1  # NO retry — we know retrying would fail
    assert api_football.is_daily_quota_exhausted() is True

    # Second call: short-circuits without touching the network.
    with pytest.raises(RuntimeError, match="daily quota exhausted"):
        client_with_disabled_bucket.get("/leagues")
    assert mock_get.call_count == 1  # still 1 — no second network call

    # Cleanup for the rest of the suite.
    api_football.reset_daily_quota_flag()


def test_daily_quota_set_on_hard_429_with_quota_body(client_with_disabled_bucket, monkeypatch) -> None:
    """HTTP 429 whose JSON body has daily-quota wording also trips the breaker."""
    from scrape import api_football

    api_football.reset_daily_quota_flag()
    quota_body = {"errors": {"requests": "Daily limit reached"}}
    bad = _fake_response(429, quota_body)
    mock_get = mock.Mock(return_value=bad)
    monkeypatch.setattr("scrape.api_football.httpx.get", mock_get)

    with pytest.raises(RuntimeError, match="daily quota exhausted"):
        client_with_disabled_bucket.get("/fixtures")
    assert api_football.is_daily_quota_exhausted() is True
    api_football.reset_daily_quota_flag()


def test_per_minute_rate_limit_does_not_trip_daily_breaker(client_with_disabled_bucket, monkeypatch) -> None:
    """A per-minute rate-limit must NOT set the daily-quota flag — it's recoverable."""
    from scrape import api_football

    api_football.reset_daily_quota_flag()
    soft = _fake_response(200, {"errors": {"rateLimit": "Too many requests. Your rate limit is 10 / minute"}})
    good = _fake_response(200, {"response": [{"x": 1}], "errors": {}})
    mock_get = mock.Mock(side_effect=[soft, good])
    monkeypatch.setattr("scrape.api_football.httpx.get", mock_get)

    result = client_with_disabled_bucket.get("/fixtures")
    assert result["response"] == [{"x": 1}]
    assert api_football.is_daily_quota_exhausted() is False
    api_football.reset_daily_quota_flag()
