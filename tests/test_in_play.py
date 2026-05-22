"""Tests for the in-play (live) match prediction module."""
from __future__ import annotations

import pytest

from models.in_play import InPlayConfig, predict_in_play


# Helper: typical EPL match xG (slight home favorite)
HOME_XG = 1.5
AWAY_XG = 1.2


def test_zero_minute_zero_score_matches_pre_match_intuition() -> None:
    """At 0' with 0-0, in-play prediction should resemble pre-match."""
    r = predict_in_play(
        home_team="A", away_team="B",
        pre_match_xg_home=HOME_XG, pre_match_xg_away=AWAY_XG,
        current_home=0, current_away=0, minute_elapsed=0,
    )
    # Probabilities sum to 1 and are reasonable.
    total = r.home_win + r.draw + r.away_win
    assert 0.99 <= total <= 1.01
    # Slight home favorite given the xG split.
    assert r.home_win > r.away_win
    # Expected final goals = pre-match xG when nothing has happened.
    assert r.expected_home_final == pytest.approx(HOME_XG, abs=1e-9)
    assert r.expected_away_final == pytest.approx(AWAY_XG, abs=1e-9)


def test_ninety_minute_locks_in_current_score() -> None:
    """At 90', the current score is the final score."""
    r = predict_in_play(
        home_team="A", away_team="B",
        pre_match_xg_home=HOME_XG, pre_match_xg_away=AWAY_XG,
        current_home=2, current_away=1, minute_elapsed=90,
    )
    assert r.minutes_remaining == 0
    assert r.home_win == pytest.approx(1.0, abs=1e-6)
    assert r.draw == pytest.approx(0.0, abs=1e-6)
    assert r.away_win == pytest.approx(0.0, abs=1e-6)


def test_late_lead_dominates_probability() -> None:
    """1-0 at 88' should give the home team ~95%+ win probability."""
    r = predict_in_play(
        home_team="A", away_team="B",
        pre_match_xg_home=HOME_XG, pre_match_xg_away=AWAY_XG,
        current_home=1, current_away=0, minute_elapsed=88,
    )
    assert r.home_win > 0.93
    assert r.away_win < 0.01


def test_chasing_team_gets_higher_multiplier() -> None:
    """Trailing side's remaining xG should be boosted by chasing_multiplier."""
    r = predict_in_play(
        home_team="A", away_team="B",
        pre_match_xg_home=HOME_XG, pre_match_xg_away=AWAY_XG,
        current_home=0, current_away=1, minute_elapsed=60,
    )
    # Home is trailing → home multiplier should be the chasing one.
    assert r.home_state_multiplier > r.away_state_multiplier
    assert r.home_state_multiplier == pytest.approx(InPlayConfig().chasing_multiplier)
    assert r.away_state_multiplier == pytest.approx(InPlayConfig().leading_multiplier)


def test_tied_score_keeps_neutral_multipliers() -> None:
    r = predict_in_play(
        home_team="A", away_team="B",
        pre_match_xg_home=HOME_XG, pre_match_xg_away=AWAY_XG,
        current_home=1, current_away=1, minute_elapsed=45,
    )
    assert r.home_state_multiplier == 1.0
    assert r.away_state_multiplier == 1.0


def test_minute_elapsed_above_90_is_clamped() -> None:
    """If a caller passes 95', treat as 90' (regulation lock-in)."""
    r = predict_in_play(
        home_team="A", away_team="B",
        pre_match_xg_home=HOME_XG, pre_match_xg_away=AWAY_XG,
        current_home=0, current_away=0, minute_elapsed=95,
    )
    assert r.minute_elapsed == 90
    assert r.minutes_remaining == 0


def test_negative_score_raises() -> None:
    with pytest.raises(ValueError):
        predict_in_play(
            home_team="A", away_team="B",
            pre_match_xg_home=1.0, pre_match_xg_away=1.0,
            current_home=-1, current_away=0, minute_elapsed=10,
        )


def test_zero_pre_match_xg_raises() -> None:
    """Predict logic doesn't make sense for a team that's expected to score zero."""
    with pytest.raises(ValueError):
        predict_in_play(
            home_team="A", away_team="B",
            pre_match_xg_home=0.0, pre_match_xg_away=1.0,
            current_home=0, current_away=0, minute_elapsed=10,
        )


def test_most_likely_scores_include_current_score_as_floor() -> None:
    """All listed scorelines should be >= current score."""
    r = predict_in_play(
        home_team="A", away_team="B",
        pre_match_xg_home=HOME_XG, pre_match_xg_away=AWAY_XG,
        current_home=1, current_away=2, minute_elapsed=70,
    )
    for s in r.most_likely_final_scores:
        assert s["home_goals"] >= 1
        assert s["away_goals"] >= 2


def test_zero_zero_at_eighty_inflates_draw_probability() -> None:
    """The longer 0-0 stays, the more likely it stays 0-0."""
    r60 = predict_in_play(
        home_team="A", away_team="B",
        pre_match_xg_home=HOME_XG, pre_match_xg_away=AWAY_XG,
        current_home=0, current_away=0, minute_elapsed=60,
    )
    r80 = predict_in_play(
        home_team="A", away_team="B",
        pre_match_xg_home=HOME_XG, pre_match_xg_away=AWAY_XG,
        current_home=0, current_away=0, minute_elapsed=80,
    )
    assert r80.draw > r60.draw


def test_to_dict_round_trip_has_expected_keys() -> None:
    r = predict_in_play(
        home_team="A", away_team="B",
        pre_match_xg_home=HOME_XG, pre_match_xg_away=AWAY_XG,
        current_home=1, current_away=1, minute_elapsed=45,
    )
    out = r.to_dict()
    assert set(out["probabilities"]) == {"home_win", "draw", "away_win"}
    assert out["minutes_remaining"] == 45
    assert "most_likely_final_scores" in out
    assert len(out["most_likely_final_scores"]) <= 5
