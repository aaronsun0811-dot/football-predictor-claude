"""Tests for the ROI simulator + odds backfill.

Synthetic data only — no DB or network dependency.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from models.roi_simulator import ROIConfig, _best_value_bet, simulate_roi
from scrape.odds_backfill import parse_one


# ----------------------------- _best_value_bet -----------------------------


def _basic_config(**overrides):
    defaults = dict(
        min_train_matches=80,
        refit_every=1,
        min_edge=0.05,
        min_ev=0.05,
        kelly_multiplier=0.5,
        max_kelly_fraction=0.05,
        starting_bankroll=100.0,
    )
    defaults.update(overrides)
    return ROIConfig(**defaults)


def test_best_value_bet_picks_highest_ev_outcome() -> None:
    probs = {"home_win": 0.55, "draw": 0.25, "away_win": 0.20}
    odds = {"home_win": 2.20, "draw": 3.50, "away_win": 4.50}  # implied ~45/29/22 after norm
    chosen = _best_value_bet(probs, odds, _basic_config())
    assert chosen is not None
    assert chosen["outcome"] == "home_win"
    assert chosen["ev"] > 0
    assert chosen["kelly_fraction"] > 0


def test_best_value_bet_returns_none_when_no_edge() -> None:
    probs = {"home_win": 0.45, "draw": 0.30, "away_win": 0.25}
    # Market matches model probabilities with a small overround.
    odds = {"home_win": 2.18, "draw": 3.27, "away_win": 3.92}
    assert _best_value_bet(probs, odds, _basic_config(min_edge=0.05)) is None


def test_best_value_bet_respects_max_kelly_cap() -> None:
    # Huge edge: model 80% vs odds 4.0 → full Kelly would be ~0.73.
    probs = {"home_win": 0.80, "draw": 0.10, "away_win": 0.10}
    odds = {"home_win": 4.0, "draw": 6.0, "away_win": 6.0}
    cfg = _basic_config(max_kelly_fraction=0.05)
    chosen = _best_value_bet(probs, odds, cfg)
    assert chosen is not None
    assert chosen["kelly_fraction"] <= 0.05 + 1e-9


# ----------------------------- simulate_roi -----------------------------


def _build_synthetic_dataset(n_matches: int = 200, *, seed: int = 0):
    """Two teams, slight home edge, fair coin draws to keep the test fast."""
    import random

    rng = random.Random(seed)
    teams = ["A", "B", "C", "D"]
    rows = []
    odds_rows = []
    start = datetime(2024, 8, 1)
    for i in range(n_matches):
        home, away = rng.sample(teams, 2)
        # Simple Poisson-ish goal generator
        home_goals = rng.choices([0, 1, 2, 3], weights=[15, 35, 30, 20])[0]
        away_goals = rng.choices([0, 1, 2, 3], weights=[20, 35, 30, 15])[0]
        d = (start + timedelta(days=i * 2)).date()
        rows.append({
            "date": d,
            "home_team": home,
            "away_team": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
        })
        # Odds with a 5% overround
        odds_rows.append({
            "date": d,
            "home_team": home,
            "away_team": away,
            "odds_home": 2.20,
            "odds_draw": 3.40,
            "odds_away": 3.20,
        })
    return pd.DataFrame(rows), pd.DataFrame(odds_rows)


def test_simulate_roi_returns_summary_and_curve() -> None:
    matches, odds = _build_synthetic_dataset(n_matches=180)
    cfg = _basic_config(
        min_train_matches=80,
        refit_every=20,
        min_edge=0.0,
        min_ev=0.0,
    )
    result = simulate_roi(matches, odds, config=cfg)
    assert "n_bets" in result.summary
    assert "ending_bankroll" in result.summary
    assert len(result.bankroll_curve) == result.summary["n_bets"]
    if result.summary["n_bets"] > 0:
        # Curve should start near (but not exactly at) starting bankroll
        first_bankroll = float(result.bankroll_curve.iloc[0]["bankroll"])
        assert first_bankroll > 0
        # Stake fraction stayed within the Kelly cap
        max_stake_pct = (result.bets["stake"] / cfg.starting_bankroll).max()
        assert max_stake_pct <= cfg.max_kelly_fraction * 1.5  # loose bound


def test_simulate_roi_raises_when_no_odds_overlap() -> None:
    matches, odds = _build_synthetic_dataset(n_matches=120)
    # Shift odds dates so the join produces nothing
    odds = odds.assign(date=pd.to_datetime(odds["date"]) + pd.Timedelta(days=10000))
    odds["date"] = odds["date"].dt.date
    with pytest.raises(ValueError):
        simulate_roi(matches, odds)


def test_simulate_roi_handles_zero_bankroll_gracefully() -> None:
    """With aggressive Kelly + bad luck the bankroll can hit zero. Don't crash."""
    matches, odds = _build_synthetic_dataset(n_matches=200, seed=7)
    cfg = _basic_config(
        min_train_matches=60,
        refit_every=10,
        min_edge=0.0,
        min_ev=0.0,
        max_kelly_fraction=0.5,  # very aggressive
        kelly_multiplier=1.0,
    )
    result = simulate_roi(matches, odds, config=cfg)
    # Summary should always be present, even after a wipe
    assert result.summary["ending_bankroll"] >= 0


# ----------------------------- odds_backfill -----------------------------


def test_parse_one_extracts_b365_closing_odds_when_available(tmp_path: Path) -> None:
    csv_path = tmp_path / "E0_test.csv"
    csv_path.write_text(
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365CH,B365CD,B365CA\n"
        "E0,12/08/24,20:00,Arsenal,Chelsea,2,1,H,1.85,3.60,4.50\n"
        "E0,13/08/24,20:00,Liverpool,Spurs,3,2,H,1.70,3.80,5.00\n"
    )
    out = parse_one(csv_path, league_key="premier_league")
    assert len(out) == 2
    assert (out["league_key"] == "premier_league").all()
    assert (out["source"] == "b365_closing").all()
    assert out.iloc[0]["odds_home"] == 1.85
    assert out.iloc[1]["odds_away"] == 5.00


def test_parse_one_falls_back_to_opening_odds(tmp_path: Path) -> None:
    csv_path = tmp_path / "E0_test.csv"
    csv_path.write_text(
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
        "E0,12/08/24,20:00,Arsenal,Chelsea,2,1,H,1.85,3.60,4.50\n"
    )
    out = parse_one(csv_path, league_key="premier_league")
    assert len(out) == 1
    assert (out["source"] == "b365_opening").all()


def test_parse_one_drops_rows_without_complete_odds(tmp_path: Path) -> None:
    csv_path = tmp_path / "E0_test.csv"
    csv_path.write_text(
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365CH,B365CD,B365CA\n"
        "E0,12/08/24,20:00,Arsenal,Chelsea,2,1,H,1.85,3.60,4.50\n"
        "E0,13/08/24,20:00,Liverpool,Spurs,3,2,H,,,\n"
    )
    out = parse_one(csv_path, league_key="premier_league")
    assert len(out) == 1
