"""Tests for the audit-vs-backtest comparison module."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from data.audit_vs_backtest import (
    _delta,
    _trim_audit_row,
    _trim_backtest_summary,
    compare_audit_to_backtest,
)
from data.database import Database


# ---------------------------------------------------------------------------
# Pure-helper unit tests (no DB)
# ---------------------------------------------------------------------------

def test_trim_audit_row_keeps_only_headline_fields():
    full_row = {
        "league_key": "pl", "n": 50, "accuracy": 0.6,
        "brier": 0.5, "rps": 0.2,
        "exact_score_accuracy": 0.1, "mean_goal_distance": 1.5,
        "extra_field_we_dont_show": "junk",
    }
    out = _trim_audit_row(full_row)
    assert "extra_field_we_dont_show" not in out
    assert out["n"] == 50
    assert out["accuracy"] == 0.6


def test_trim_backtest_summary_renames_brier_score_to_brier():
    bt_summary = {
        "n_predictions": 1689, "accuracy": 0.527,
        "brier_score": 0.585, "rps": 0.203,
        "exact_score_accuracy": 0.104, "mean_goal_distance": 1.94,
    }
    out = _trim_backtest_summary(bt_summary)
    # Note the rename: backtest summary uses "brier_score", audit uses "brier"
    assert out["brier"] == 0.585
    assert out["n"] == 1689


def test_delta_returns_backtest_minus_audit():
    audit = {"accuracy": 0.4, "brier": 0.6, "rps": 0.25, "exact_score_accuracy": 0.08}
    bt = {"accuracy": 0.55, "brier_score": 0.5, "rps": 0.2, "exact_score_accuracy": 0.11}
    delta = _delta(audit, bt)
    assert delta["accuracy"] == pytest.approx(0.15)
    assert delta["brier"] == pytest.approx(-0.1)
    assert delta["rps"] == pytest.approx(-0.05)
    assert delta["exact_score_accuracy"] == pytest.approx(0.03)


def test_delta_returns_none_for_missing_metrics():
    audit = {"accuracy": 0.4, "brier": None, "rps": 0.25, "exact_score_accuracy": None}
    bt = {"accuracy": 0.55, "brier_score": 0.5, "rps": None, "exact_score_accuracy": 0.11}
    delta = _delta(audit, bt)
    assert delta["accuracy"] == pytest.approx(0.15)
    assert delta["brier"] is None        # audit missing
    assert delta["rps"] is None          # backtest missing
    assert delta["exact_score_accuracy"] is None  # audit missing


# ---------------------------------------------------------------------------
# Integration: compare_audit_to_backtest end-to-end with mocked DB + backtest
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_db():
    return mock.Mock(spec=Database)


def test_compare_empty_audit_returns_empty_rows(fake_db):
    out = compare_audit_to_backtest(fake_db, [])
    assert out == {"n_leagues_compared": 0, "n_leagues_skipped": 0, "rows": [], "skipped": []}


def test_compare_one_league_joins_audit_and_backtest(fake_db, monkeypatch):
    audit_by_league = [
        {"league_key": "pl", "n": 8, "accuracy": 0.5, "brier": 0.6, "rps": 0.21,
         "exact_score_accuracy": 0.125, "mean_goal_distance": 1.5},
    ]
    # Mock the inner backtest call so we don't run a real walk-forward
    fake_summary = {
        "n_predictions": 1689, "accuracy": 0.527,
        "brier_score": 0.585, "rps": 0.203,
        "exact_score_accuracy": 0.104, "mean_goal_distance": 1.94,
    }
    monkeypatch.setattr(
        "data.audit_vs_backtest._run_single_league_backtest",
        mock.Mock(return_value=fake_summary),
    )
    out = compare_audit_to_backtest(fake_db, audit_by_league)
    assert out["n_leagues_compared"] == 1
    row = out["rows"][0]
    assert row["league_key"] == "pl"
    assert row["audit"]["n"] == 8
    assert row["audit"]["accuracy"] == 0.5
    assert row["backtest"]["n"] == 1689
    assert row["backtest"]["accuracy"] == 0.527
    # backtest beats audit on accuracy → positive Δ
    assert row["delta"]["accuracy"] == pytest.approx(0.027)
    # backtest is slightly better on brier (lower=better) → negative Δ
    assert row["delta"]["brier"] == pytest.approx(-0.015)


def test_compare_skips_league_when_backtest_unavailable(fake_db, monkeypatch):
    audit_by_league = [
        {"league_key": "tiny_league", "n": 3, "accuracy": 0.33, "brier": 0.7, "rps": 0.3},
    ]
    # Backtest returns None → league can't be backtested
    monkeypatch.setattr(
        "data.audit_vs_backtest._run_single_league_backtest",
        mock.Mock(return_value=None),
    )
    out = compare_audit_to_backtest(fake_db, audit_by_league)
    assert out["n_leagues_compared"] == 0
    assert out["n_leagues_skipped"] == 1
    assert out["skipped"][0]["league_key"] == "tiny_league"
    assert out["skipped"][0]["reason"] == "backtest_unavailable"


def test_compare_caps_at_max_leagues(fake_db, monkeypatch):
    """Running 30 backtests on one request would be a multi-minute response.
    Cap at ``max_leagues`` (default 12), sorted by sample size descending."""
    audit_by_league = [
        {"league_key": f"league_{i:02d}", "n": i, "accuracy": 0.5, "brier": 0.5, "rps": 0.2}
        for i in range(15)
    ]
    monkeypatch.setattr(
        "data.audit_vs_backtest._run_single_league_backtest",
        mock.Mock(return_value={"n_predictions": 100, "accuracy": 0.5, "brier_score": 0.5, "rps": 0.2}),
    )
    out = compare_audit_to_backtest(fake_db, audit_by_league, max_leagues=5)
    assert out["n_leagues_compared"] == 5
    # The 5 biggest by ``n`` should win: 14, 13, 12, 11, 10
    league_keys = [r["league_key"] for r in out["rows"]]
    assert league_keys == ["league_14", "league_13", "league_12", "league_11", "league_10"]


def test_compare_drops_audit_rows_without_league_key(fake_db, monkeypatch):
    """A malformed audit row (no league_key) shouldn't crash or produce a
    junk entry."""
    audit_by_league = [
        {"n": 5, "accuracy": 0.5, "brier": 0.5, "rps": 0.2},  # missing league_key
        {"league_key": "pl", "n": 10, "accuracy": 0.6, "brier": 0.4, "rps": 0.18},
    ]
    monkeypatch.setattr(
        "data.audit_vs_backtest._run_single_league_backtest",
        mock.Mock(return_value={"n_predictions": 100, "accuracy": 0.5, "brier_score": 0.5, "rps": 0.2}),
    )
    out = compare_audit_to_backtest(fake_db, audit_by_league)
    assert out["n_leagues_compared"] == 1
    assert out["rows"][0]["league_key"] == "pl"


def test_compare_swallows_backtest_exceptions(fake_db, monkeypatch):
    """If the inner backtest blows up unexpectedly, we just skip that league."""
    audit_by_league = [
        {"league_key": "pl", "n": 5, "accuracy": 0.5, "brier": 0.5, "rps": 0.2},
    ]
    # The wrapper catches ValueError/RuntimeError already; verify behavior end-to-end
    fake_db.fetch_matches = mock.Mock(side_effect=RuntimeError("boom"))
    out = compare_audit_to_backtest(fake_db, audit_by_league)
    # The whole pipeline shouldn't raise — league just goes to skipped
    assert out["n_leagues_compared"] == 0
    assert out["n_leagues_skipped"] == 1
