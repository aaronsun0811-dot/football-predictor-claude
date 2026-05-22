"""Tests for models/diagnostics — calibration curve, ECE, confidence ladder.

Synthetic tests so we don't depend on the SQLite DB or live data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.diagnostics import (
    DEFAULT_BIN_EDGES,
    build_diagnostics,
    calibration_curve,
    confidence_ladder,
    expected_calibration_error,
)


def _make_predictions(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_perfectly_calibrated_model_has_near_zero_ece() -> None:
    """A model that always assigns p to the actual outcome with frequency p."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(2000):
        p_home = rng.uniform(0.2, 0.6)
        p_away = rng.uniform(0.2, 0.5 - p_home + 0.5)
        p_away = min(p_away, 1 - p_home - 0.05)
        p_draw = 1 - p_home - p_away
        # Pick the actual outcome with the predicted probability — perfect calibration.
        actual = rng.choice(
            ["home_win", "draw", "away_win"],
            p=[p_home, p_draw, p_away],
        )
        rows.append({
            "home_win": p_home,
            "draw": p_draw,
            "away_win": p_away,
            "actual": actual,
        })
    frame = _make_predictions(rows)
    ece = expected_calibration_error(frame)
    # With 2000 samples, sampling noise alone gives ECE ~0.02. Anything under
    # 0.05 means we're not introducing systematic miscalibration.
    assert all(value < 0.05 for value in ece.values()), ece


def test_overconfident_model_shows_negative_calibration_gap() -> None:
    """If actual frequency is lower than predicted, calibration row reflects it."""
    rows = []
    # 100 matches where the model says home=0.8 but only 50% actually happen.
    for i in range(100):
        rows.append({
            "home_win": 0.8,
            "draw": 0.1,
            "away_win": 0.1,
            "actual": "home_win" if i < 50 else "away_win",
        })
    frame = _make_predictions(rows)
    curve = calibration_curve(frame)
    home_bin = next(b for b in curve["home_win"] if b["bin_low"] == 0.8)
    assert home_bin["n"] == 100
    assert abs(home_bin["mean_predicted"] - 0.8) < 1e-9
    assert abs(home_bin["observed_frequency"] - 0.5) < 1e-9


def test_confidence_ladder_buckets_top_pick_only() -> None:
    """Confidence ladder uses max(probs), not the actual outcome's prob."""
    rows = [
        # High-confidence home pick that's correct
        {"home_win": 0.85, "draw": 0.10, "away_win": 0.05, "actual": "home_win"},
        # High-confidence home pick that's wrong
        {"home_win": 0.85, "draw": 0.10, "away_win": 0.05, "actual": "away_win"},
        # Low-confidence pick that's correct
        {"home_win": 0.40, "draw": 0.35, "away_win": 0.25, "actual": "home_win"},
    ]
    frame = _make_predictions(rows)
    ladder = confidence_ladder(frame)
    high_band = next(b for b in ladder if b["band_low"] == 0.85)
    assert high_band["n"] == 2
    assert high_band["accuracy"] == 0.5
    low_band = next(b for b in ladder if b["band_low"] == 0.34)
    assert low_band["n"] == 1
    assert low_band["accuracy"] == 1.0


def test_build_diagnostics_handles_empty_frame() -> None:
    out = build_diagnostics(pd.DataFrame())
    assert out["confidence_ladder"] == []
    assert out["expected_calibration_error"] == {
        "home_win": 0.0,
        "draw": 0.0,
        "away_win": 0.0,
    }


def test_calibration_includes_zero_count_bins_for_completeness() -> None:
    """Empty bins still appear in the result so chart axes stay stable."""
    rows = [
        {"home_win": 0.45, "draw": 0.30, "away_win": 0.25, "actual": "home_win"},
    ]
    frame = _make_predictions(rows)
    curve = calibration_curve(frame)
    # Should have one bin per (bin_low, bin_high) pair regardless of data.
    expected_bins = len(DEFAULT_BIN_EDGES) - 1
    for outcome in ("home_win", "draw", "away_win"):
        assert len(curve[outcome]) == expected_bins
