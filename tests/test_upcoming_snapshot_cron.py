"""Tests for the periodic /upcoming snapshot cron (R31).

The scheduled job exists so history.jsonl keeps growing even when nobody
hits the web UI. Without it the audit module starves — see Round 30's
"by_league empty" observation: 5 leagues × 1 prediction each, nothing meets
the ≥3 threshold for per-league rollup.
"""
from __future__ import annotations

import os
from unittest import mock

import pytest


def test_settings_parse_upcoming_snapshot_env_vars(monkeypatch):
    """All three knobs (enable / hours / days_ahead) read from env, with defaults."""
    from config.settings import get_settings

    # Custom values
    monkeypatch.setenv("FOOTBALL_PREDICTOR_UPCOMING_SNAPSHOT", "false")
    monkeypatch.setenv("FOOTBALL_PREDICTOR_UPCOMING_SNAPSHOT_HOURS", "7,19")
    monkeypatch.setenv("FOOTBALL_PREDICTOR_UPCOMING_SNAPSHOT_DAYS", "21")
    s = get_settings()
    assert s.upcoming_snapshot_enabled is False
    assert s.upcoming_snapshot_hours == "7,19"
    assert s.upcoming_snapshot_days_ahead == 21


def test_settings_have_sane_defaults(monkeypatch):
    """Defaults: enabled, 4x daily (9/15/21/3), 14-day horizon."""
    from config.settings import get_settings

    for v in (
        "FOOTBALL_PREDICTOR_UPCOMING_SNAPSHOT",
        "FOOTBALL_PREDICTOR_UPCOMING_SNAPSHOT_HOURS",
        "FOOTBALL_PREDICTOR_UPCOMING_SNAPSHOT_DAYS",
    ):
        monkeypatch.delenv(v, raising=False)
    s = get_settings()
    assert s.upcoming_snapshot_enabled is True
    assert s.upcoming_snapshot_hours == "9,15,21,3"
    assert s.upcoming_snapshot_days_ahead == 14


def test_scheduled_snapshot_swallows_inner_errors(monkeypatch):
    """If ``_compute_upcoming_payload`` raises, the scheduled wrapper must NOT
    propagate — otherwise one bad day kills the whole scheduler loop."""
    import predict as predict_module

    monkeypatch.setattr(
        predict_module, "_compute_upcoming_payload",
        mock.Mock(side_effect=RuntimeError("simulated downstream failure")),
    )
    # Should NOT raise
    predict_module._scheduled_upcoming_snapshot()
    # And the inner function was actually called (i.e. we didn't short-circuit
    # away from the work — we tried, then ate the error)
    assert predict_module._compute_upcoming_payload.called


def test_scheduled_snapshot_calls_compute_upcoming_with_all_leagues(monkeypatch):
    """The cron should request the full league set, not a random subset."""
    import predict as predict_module
    from scrape.registry import LeagueRegistry

    captured = {}

    def fake_compute(*, league_keys, days_ahead, include_predictions):
        captured["league_keys"] = league_keys
        captured["days_ahead"] = days_ahead
        captured["include_predictions"] = include_predictions
        return {"fixtures": []}

    monkeypatch.setattr(predict_module, "_compute_upcoming_payload", fake_compute)
    predict_module._scheduled_upcoming_snapshot()

    expected = sorted({league.key for league in LeagueRegistry().all()})
    assert captured["league_keys"] == expected
    assert captured["include_predictions"] is True
    # days_ahead defaults to the settings value, currently 14
    assert isinstance(captured["days_ahead"], int)
    assert captured["days_ahead"] >= 1


class _FakeScheduler:
    """Minimal stand-in for APScheduler — captures job IDs, never actually runs."""
    def __init__(self, **_):
        self.added_ids: list[str] = []
    def add_job(self, *args, **kwargs):
        self.added_ids.append(kwargs.get("id", "?"))
    def start(self):
        pass


def _patched_settings(predict_module, **overrides):
    """Return a new frozen AppSettings with the given overrides — for tests
    that need to flip ``upcoming_snapshot_enabled`` etc. without mutating the
    real (frozen) instance."""
    from dataclasses import replace
    return replace(predict_module.SETTINGS, **overrides)


def test_start_scheduler_registers_upcoming_snapshot_job(monkeypatch):
    """When ``upcoming_snapshot_enabled`` is True, ``_start_scheduler`` adds a
    job with id 'periodic_upcoming_snapshot'."""
    import predict as predict_module

    fake_scheduler = _FakeScheduler()
    monkeypatch.setattr(predict_module, "BackgroundScheduler", lambda **_: fake_scheduler)
    monkeypatch.setattr(
        predict_module, "SETTINGS",
        _patched_settings(predict_module, upcoming_snapshot_enabled=True),
    )

    predict_module._start_scheduler()
    assert "periodic_upcoming_snapshot" in fake_scheduler.added_ids


def test_start_scheduler_skips_snapshot_job_when_disabled(monkeypatch):
    """When the env flag is off, no snapshot job is registered."""
    import predict as predict_module

    fake_scheduler = _FakeScheduler()
    monkeypatch.setattr(predict_module, "BackgroundScheduler", lambda **_: fake_scheduler)
    monkeypatch.setattr(
        predict_module, "SETTINGS",
        _patched_settings(predict_module, upcoming_snapshot_enabled=False),
    )

    predict_module._start_scheduler()
    assert "periodic_upcoming_snapshot" not in fake_scheduler.added_ids
    # But the always-on daily job is still there
    assert "daily_incremental_update" in fake_scheduler.added_ids
