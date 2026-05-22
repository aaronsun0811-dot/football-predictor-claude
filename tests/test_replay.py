"""Tests for the match replay module."""
from __future__ import annotations

import random
from datetime import date, timedelta

import pandas as pd
import pytest

from models.replay import _outcome, rank_surprises, replay_match


def _synthetic_league(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    teams = ["A", "B", "C", "D", "E"]
    rows = []
    start = pd.Timestamp("2024-01-01")
    for i in range(n):
        h, a = rng.sample(teams, 2)
        rows.append({
            "date": start + pd.Timedelta(days=i),
            "league_key": "test_league",
            "home_team": h,
            "away_team": a,
            "home_goals": rng.choices([0, 1, 2, 3], weights=[15, 35, 30, 20])[0],
            "away_goals": rng.choices([0, 1, 2, 3], weights=[20, 35, 30, 15])[0],
        })
    return pd.DataFrame(rows)


def test_replay_match_returns_full_result_structure() -> None:
    league = _synthetic_league(n=200, seed=0)
    target = league.iloc[150]
    out = replay_match(
        target_date=target["date"],
        home_team=target["home_team"],
        away_team=target["away_team"],
        league_matches=league,
    )
    # Probabilities sum to ~1
    total = out.predicted_home_win + out.predicted_draw + out.predicted_away_win
    assert 0.99 <= total <= 1.01
    # All metrics non-negative
    assert out.log_loss >= 0
    assert 0 <= out.rps <= 1
    assert 0 <= out.brier <= 2
    # Training rows = matches strictly before target.
    assert out.training_rows <= 150


def test_replay_match_actual_outcome_matches_dataframe() -> None:
    league = _synthetic_league()
    target = league.iloc[100]
    out = replay_match(
        target_date=target["date"],
        home_team=target["home_team"],
        away_team=target["away_team"],
        league_matches=league,
    )
    assert out.actual_home_goals == int(target["home_goals"])
    assert out.actual_away_goals == int(target["away_goals"])
    assert out.actual_outcome == _outcome(
        int(target["home_goals"]), int(target["away_goals"]),
    )


def test_replay_match_raises_when_match_not_in_data() -> None:
    league = _synthetic_league()
    with pytest.raises(KeyError, match="No match found"):
        replay_match(
            target_date=pd.Timestamp("2024-01-15"),
            home_team="Atlantis FC",
            away_team="Krypton XI",
            league_matches=league,
        )


def test_replay_match_raises_when_too_few_prior_matches() -> None:
    league = _synthetic_league(n=200)
    early = league.iloc[10]  # only ~10 prior matches
    with pytest.raises(ValueError, match="Need at least 50"):
        replay_match(
            target_date=early["date"],
            home_team=early["home_team"],
            away_team=early["away_team"],
            league_matches=league,
        )


def test_replay_match_no_leakage_training_set_excludes_target() -> None:
    """The training set must never include the target match itself."""
    league = _synthetic_league(n=200)
    target = league.iloc[120]
    out = replay_match(
        target_date=target["date"],
        home_team=target["home_team"],
        away_team=target["away_team"],
        league_matches=league,
    )
    # 120 matches happened strictly before this one's date (or earlier
    # same-date). The cutoff is strict < target_date so training_rows
    # may equal 120 or less (multiple matches on the same day).
    matches_before = (pd.to_datetime(league["date"]) < target["date"]).sum()
    assert out.training_rows <= int(matches_before)


def test_rank_surprises_returns_both_lists() -> None:
    league = _synthetic_league(n=350)
    out = rank_surprises(
        league_matches=league,
        league_key="test_league",
        min_train_matches=80,
        refit_every=20,
    )
    assert "biggest_upsets" in out
    assert "best_calls" in out
    # When data is plentiful, both lists should have entries.
    assert len(out["biggest_upsets"]) > 0
    assert len(out["best_calls"]) > 0
    # Best calls must all be correct predictions.
    for call in out["best_calls"]:
        assert call["correct"] is True


def test_rank_surprises_upsets_sorted_ascending_by_p_actual() -> None:
    league = _synthetic_league(n=400)
    out = rank_surprises(
        league_matches=league,
        league_key="test_league",
        min_train_matches=100,
        refit_every=30,
    )
    p_values = [r["p_actual"] for r in out["biggest_upsets"]]
    assert p_values == sorted(p_values), "biggest_upsets must be sorted lowest-p first"


def test_rank_surprises_handles_too_small_league() -> None:
    league = _synthetic_league(n=100)
    out = rank_surprises(
        league_matches=league,
        league_key="test_league",
        min_train_matches=200,  # impossible
    )
    assert out["biggest_upsets"] == []
    assert out["best_calls"] == []


def test_outcome_helper() -> None:
    assert _outcome(2, 1) == "home_win"
    assert _outcome(0, 0) == "draw"
    assert _outcome(1, 3) == "away_win"
