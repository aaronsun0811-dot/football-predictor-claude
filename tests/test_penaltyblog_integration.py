"""Tests for the penaltyblog wrapper + Shin implied + RPS metric.

These confirm that the ideas borrowed from MatchOracle (RPS) and
penaltyblog (multiple goal models, Shin probabilities) integrate cleanly
without depending on the live SQLite DB or network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.backtest import summarize_predictions
from models.implied_probs import implied_probabilities
from models.penaltyblog_models import (
    MODEL_FACTORIES,
    fit_and_predict,
)


def _synthetic_matches(n: int = 200, seed: int = 0) -> pd.DataFrame:
    import random
    rng = random.Random(seed)
    teams = ["A", "B", "C", "D", "E"]
    rows = []
    for i in range(n):
        h, a = rng.sample(teams, 2)
        rows.append({
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "home_team": h,
            "away_team": a,
            "home_goals": rng.choices([0, 1, 2, 3], weights=[15, 35, 30, 20])[0],
            "away_goals": rng.choices([0, 1, 2, 3], weights=[20, 35, 30, 15])[0],
        })
    return pd.DataFrame(rows)


# ---------------------------- penaltyblog wrappers ----------------------------


@pytest.mark.parametrize("model", ["dixon_coles", "bivariate_poisson", "poisson"])
def test_each_penaltyblog_model_returns_valid_probabilities(model: str) -> None:
    matches = _synthetic_matches(n=180)
    pred = fit_and_predict(matches, "A", "B", model=model)
    p = pred.home_win + pred.draw + pred.away_win
    assert 0.99 <= p <= 1.01, f"{model}: probabilities sum to {p}"
    assert pred.expected_home_goals > 0
    assert pred.expected_away_goals > 0
    assert len(pred.most_likely_scores) == 5


def test_penaltyblog_unknown_model_raises() -> None:
    matches = _synthetic_matches()
    with pytest.raises(ValueError, match="Unknown model"):
        fit_and_predict(matches, "A", "B", model="not_a_real_model")


def test_penaltyblog_factories_include_expected_models() -> None:
    expected = {"dixon_coles", "bivariate_poisson", "poisson",
                "negative_binomial", "zero_inflated_poisson"}
    assert expected <= set(MODEL_FACTORIES)


# ---------------------------- Shin implied probabilities ----------------------------


def test_shin_method_returns_positive_probabilities_summing_to_one() -> None:
    res = implied_probabilities(2.10, 3.40, 3.80, method="shin")
    total = res.home_win + res.draw + res.away_win
    assert 0.99 <= total <= 1.01
    assert res.margin > 0
    # All four supported methods should agree to within 1pp on tight odds.
    methods = {m: implied_probabilities(2.10, 3.40, 3.80, method=m)
               for m in ["multiplicative", "shin", "power", "additive"]}
    for outcome in ["home_win", "draw", "away_win"]:
        values = [getattr(methods[m], outcome) for m in methods]
        assert max(values) - min(values) < 0.01, (outcome, methods)


def test_implied_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        implied_probabilities(2.0, 3.5, 4.0, method="banana")


# ---------------------------- RPS in backtest summary ----------------------------


def test_rps_added_to_summary() -> None:
    """RPS is the headline metric in MatchOracle. We compute it ourselves
    so the core summary doesn't depend on penaltyblog being installed."""
    predictions = pd.DataFrame([
        {"home_win": 0.7, "draw": 0.2, "away_win": 0.1, "actual": "home_win", "correct": True},
        {"home_win": 0.4, "draw": 0.3, "away_win": 0.3, "actual": "draw", "correct": False},
        {"home_win": 0.2, "draw": 0.2, "away_win": 0.6, "actual": "away_win", "correct": True},
    ])
    summary = summarize_predictions(predictions)
    assert "rps" in summary
    assert summary["rps"] > 0
    assert summary["rps"] < 1
    # Sanity: a perfectly confident correct prediction should have RPS ~0
    perfect = pd.DataFrame([
        {"home_win": 1.0, "draw": 0.0, "away_win": 0.0, "actual": "home_win", "correct": True},
    ])
    perfect_summary = summarize_predictions(perfect)
    assert perfect_summary["rps"] == pytest.approx(0.0, abs=1e-9)


def test_rps_penalizes_distant_misses_more_than_near_ones() -> None:
    """Key property of RPS: predicting away when home actually happens
    is worse than predicting draw when home actually happens."""
    near_miss = pd.DataFrame([
        {"home_win": 0.0, "draw": 1.0, "away_win": 0.0, "actual": "home_win", "correct": False},
    ])
    far_miss = pd.DataFrame([
        {"home_win": 0.0, "draw": 0.0, "away_win": 1.0, "actual": "home_win", "correct": False},
    ])
    near_rps = summarize_predictions(near_miss)["rps"]
    far_rps = summarize_predictions(far_miss)["rps"]
    assert far_rps > near_rps, "RPS must penalize ordered errors"


def test_rps_summary_handles_empty_frame() -> None:
    summary = summarize_predictions(pd.DataFrame())
    assert summary["rps"] is None
