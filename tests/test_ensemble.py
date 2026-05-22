"""Tests for the ensemble + market-fused predictor."""
from __future__ import annotations

import pandas as pd
import pytest

from models.ensemble import EnsembleConfig, fit, predict_match


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


def test_ensemble_fits_all_default_members() -> None:
    matches = _synthetic_matches()
    ens = fit(matches)
    # Default config has 3 members.
    assert set(ens.members) == {"dixon_coles_elo", "dixon_coles", "bivariate_poisson"}
    assert ens.training_rows == len(matches)


def test_ensemble_probabilities_sum_to_one_and_match_member_average() -> None:
    matches = _synthetic_matches()
    ens = fit(matches)
    out = predict_match(ens, "A", "B")
    p = out["probabilities"]
    total = p["home_win"] + p["draw"] + p["away_win"]
    assert 0.99 <= total <= 1.01

    # With equal weights, ensemble probabilities should equal the unweighted
    # mean of the per-member probabilities.
    contribs = out["contributions"]
    if len(contribs) >= 2:
        mean_h = sum(c["home_win"] for c in contribs) / len(contribs)
        assert abs(p["home_win"] - mean_h) < 1e-9


def test_market_fusion_pulls_toward_market_probabilities() -> None:
    matches = _synthetic_matches()
    ens = fit(matches, config=EnsembleConfig(market_fusion_weight=0.5))

    # An extreme market view: 90% home, 5% draw, 5% away
    market = {"home_win": 0.9, "draw": 0.05, "away_win": 0.05}
    fused = predict_match(ens, "A", "B", market_implied=market)
    pure = predict_match(ens, "A", "B", market_implied=None)

    # Fused probability of home should be strictly closer to 0.9 than pure.
    assert abs(fused["probabilities"]["home_win"] - 0.9) < abs(pure["probabilities"]["home_win"] - 0.9)


def test_market_fusion_weight_zero_ignores_market() -> None:
    matches = _synthetic_matches()
    ens = fit(matches, config=EnsembleConfig(market_fusion_weight=0.0))
    market = {"home_win": 0.99, "draw": 0.005, "away_win": 0.005}
    out = predict_match(ens, "A", "B", market_implied=market)
    # Should equal the pure model ensemble even with extreme market input.
    bare = predict_match(ens, "A", "B", market_implied=None)
    assert out["probabilities"] == pytest.approx(bare["probabilities"])


def test_ensemble_skips_members_that_cant_predict_unknown_teams() -> None:
    """penaltyblog models raise KeyError on teams not in training; our
    home-grown dixon_coles_elo gracefully returns a neutral baseline. The
    ensemble should drop the raising members and use whatever remains."""
    matches = _synthetic_matches()
    ens = fit(matches)
    out = predict_match(ens, "Atlantis FC", "Krypton XI")
    # dixon_coles_elo will silently predict; pb members will be skipped.
    assert "dixon_coles_elo" in out["members_used"]
    assert "dixon_coles" not in out["members_used"]
    assert "bivariate_poisson" not in out["members_used"]
    total = sum(out["probabilities"].values())
    assert 0.99 <= total <= 1.01


def test_ensemble_weighted_average_respects_weights() -> None:
    """If we weight one member heavily, the ensemble should match it."""
    matches = _synthetic_matches()
    cfg = EnsembleConfig(
        members=("dixon_coles_elo", "dixon_coles"),
        weights=(99.0, 1.0),  # 99% weight on dixon_coles_elo
    )
    ens = fit(matches, config=cfg)
    out = predict_match(ens, "A", "B")
    elo_member = next(c for c in out["contributions"] if c["name"] == "dixon_coles_elo")
    # Ensemble should be ~99% the dixon_coles_elo prediction.
    assert abs(out["probabilities"]["home_win"] - elo_member["home_win"]) < 0.05
