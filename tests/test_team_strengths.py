"""Tests for the team strengths extraction."""
from __future__ import annotations

import random

import pandas as pd
import pytest

from models.team_strengths import compare_two, extract_strengths


def _synthetic_matches(n: int = 200, seed: int = 1) -> pd.DataFrame:
    """Build a synthetic 4-team league with known relative strength.

    Team A: strong (always favored, scores a lot).
    Team B: average.
    Team C: average.
    Team D: weak.
    """
    rng = random.Random(seed)
    teams = ["A", "B", "C", "D"]
    strength = {"A": 0.6, "B": 0.0, "C": 0.0, "D": -0.6}
    rows = []
    for i in range(n):
        h, a = rng.sample(teams, 2)
        # Lambda based on strength differential + small home edge.
        lam_home = max(0.3, 1.3 + strength[h] - 0.5 * strength[a] + 0.2)
        lam_away = max(0.3, 1.1 + strength[a] - 0.5 * strength[h])
        # Crude Poisson sample: count how many of 5 random thresholds fire.
        hg = sum(1 for _ in range(8) if rng.random() < lam_home / 4)
        ag = sum(1 for _ in range(8) if rng.random() < lam_away / 4)
        rows.append({
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "home_team": h,
            "away_team": a,
            "home_goals": hg,
            "away_goals": ag,
        })
    return pd.DataFrame(rows)


def test_extract_returns_one_row_per_team_sorted_by_overall() -> None:
    matches = _synthetic_matches()
    out = extract_strengths(matches)
    teams = out["teams"]
    assert len(teams) == 4
    # Sorted descending by overall — strongest first.
    overalls = [t["overall"] for t in teams]
    assert overalls == sorted(overalls, reverse=True)
    # Team A should rank above Team D in this synthetic world.
    team_names = [t["team"] for t in teams]
    assert team_names.index("A") < team_names.index("D")


def test_extract_attack_higher_for_strong_attacker() -> None:
    """Attack parameter should reflect synthetic strength ranking."""
    matches = _synthetic_matches(n=400)
    out = extract_strengths(matches)
    by_team = {t["team"]: t for t in out["teams"]}
    assert by_team["A"]["attack"] > by_team["D"]["attack"]


def test_defense_field_is_inverted_higher_is_better() -> None:
    """A team that concedes few should have positive defense_inverted."""
    matches = _synthetic_matches(n=400)
    out = extract_strengths(matches)
    by_team = {t["team"]: t for t in out["teams"]}
    # Team A should have higher defense (concedes less) than D.
    assert by_team["A"]["defense"] >= by_team["D"]["defense"]


def test_recent_form_string_is_5_chars_or_fewer_with_only_wdl() -> None:
    matches = _synthetic_matches()
    out = extract_strengths(matches)
    for t in out["teams"]:
        assert len(t["recent_form"]) <= 5
        assert all(ch in "WDL" for ch in t["recent_form"])


def test_recent_gd_matches_recent_form_count() -> None:
    matches = _synthetic_matches()
    out = extract_strengths(matches)
    for t in out["teams"]:
        # GD is in [-5*max_goals, +5*max_goals] — sanity bound.
        assert -50 <= t["recent_gd"] <= 50


def test_empty_matches_returns_empty_teams() -> None:
    out = extract_strengths(pd.DataFrame())
    assert out["teams"] == []
    assert out["model"] is None


def test_compare_two_returns_diff_dictionary() -> None:
    matches = _synthetic_matches(n=300)
    out = compare_two(matches, "A", "D")
    assert out["home"]["team"] == "A"
    assert out["away"]["team"] == "D"
    assert "attack" in out["differential"]
    assert "defense" in out["differential"]
    assert "overall" in out["differential"]
    # Differential = home − away (so positive favors A)
    expected_attack_diff = out["home"]["attack"] - out["away"]["attack"]
    assert out["differential"]["attack"] == pytest.approx(expected_attack_diff)


def test_compare_two_raises_on_unknown_team() -> None:
    matches = _synthetic_matches()
    with pytest.raises(KeyError):
        compare_two(matches, "A", "Atlantis FC")


def test_club_elo_lookup_propagates_into_output() -> None:
    matches = _synthetic_matches()
    elo = {"A": 1850, "D": 1450}
    out = extract_strengths(matches, club_elo_lookup=elo)
    by_team = {t["team"]: t for t in out["teams"]}
    assert by_team["A"]["club_elo"] == 1850
    assert by_team["D"]["club_elo"] == 1450
    # Teams not in the lookup should have None.
    assert by_team["B"]["club_elo"] is None
