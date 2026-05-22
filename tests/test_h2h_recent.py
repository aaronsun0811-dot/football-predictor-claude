"""Tests for the H2H last-N lookup attached to /predict (R38).

Each /predict response now carries up to 5 previous meetings between the two
teams, re-oriented to the current prediction's home-team perspective so the UI
can render colored W/D/L chips without needing to think about home/away flips.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from data.database import Database


@pytest.fixture
def db_with_h2h(tmp_path: Path) -> Database:
    """Seed a DB with 6 Arsenal-vs-Chelsea matches spread over time."""
    db = Database(tmp_path / "h2h.sqlite3")
    db.init()
    today = date.today()
    rows = pd.DataFrame([
        # Recent first (we'll sort by date desc when querying)
        {"date": today - timedelta(days=30),  "home_team": "Arsenal", "away_team": "Chelsea", "home_goals": 2, "away_goals": 1},
        {"date": today - timedelta(days=120), "home_team": "Chelsea", "away_team": "Arsenal", "home_goals": 0, "away_goals": 1},
        {"date": today - timedelta(days=200), "home_team": "Arsenal", "away_team": "Chelsea", "home_goals": 1, "away_goals": 1},
        {"date": today - timedelta(days=400), "home_team": "Chelsea", "away_team": "Arsenal", "home_goals": 2, "away_goals": 0},
        {"date": today - timedelta(days=500), "home_team": "Arsenal", "away_team": "Chelsea", "home_goals": 3, "away_goals": 2},
        # 6th should NOT come back when we ask for 5
        {"date": today - timedelta(days=800), "home_team": "Chelsea", "away_team": "Arsenal", "home_goals": 1, "away_goals": 1},
    ])
    db.upsert_matches(rows, source="test", league_key="premier_league", league_name="PL")
    return db


def test_h2h_returns_last_n_matches_only(db_with_h2h):
    from predict import _h2h_last_n
    results = _h2h_last_n(db_with_h2h, "Arsenal", "Chelsea", n=5)
    assert len(results) == 5


def test_h2h_results_sorted_newest_first(db_with_h2h):
    from predict import _h2h_last_n
    results = _h2h_last_n(db_with_h2h, "Arsenal", "Chelsea", n=5)
    dates = [r["date"] for r in results]
    assert dates == sorted(dates, reverse=True), f"expected newest first, got {dates}"


def test_h2h_orients_outcome_to_current_home(db_with_h2h):
    """When Arsenal is the CURRENT home and the past match was Arsenal-home → W
    means Arsenal won. When the past match was Chelsea-home → still oriented
    to Arsenal's outcome, so Arsenal winning while away still reads 'W'."""
    from predict import _h2h_last_n
    results = _h2h_last_n(db_with_h2h, "Arsenal", "Chelsea", n=5)
    by_date = {r["date"]: r for r in results}
    # Find the Chelsea-home Arsenal-2-Chelsea-0 game (90d ago in our seed):
    # actually the 120d game: Chelsea 0, Arsenal 1 (away win for Arsenal)
    away_win = [r for r in results
                if r["home_team"] == "Chelsea" and r["away_team"] == "Arsenal"
                and r["home_goals"] == 0 and r["away_goals"] == 1][0]
    assert away_win["outcome_for_home"] == "W"  # Arsenal won despite being away
    assert away_win["venue_for_current_home"] == "A"
    assert away_win["score_for_current_home"] == "1-0"  # Arsenal's goals first


def test_h2h_handles_swapped_query(db_with_h2h):
    """Asking with Chelsea as home should orient outcomes to Chelsea's POV."""
    from predict import _h2h_last_n
    results = _h2h_last_n(db_with_h2h, "Chelsea", "Arsenal", n=5)
    # Same set of matches, but now WINs/LOSSes flip
    arsenal_results = _h2h_last_n(db_with_h2h, "Arsenal", "Chelsea", n=5)
    # Map by date
    chelsea_by_date = {r["date"]: r for r in results}
    arsenal_by_date = {r["date"]: r for r in arsenal_results}
    for d in chelsea_by_date:
        c_out = chelsea_by_date[d]["outcome_for_home"]
        a_out = arsenal_by_date[d]["outcome_for_home"]
        # W ↔ L flip; D stays D
        if a_out == "W":
            assert c_out == "L"
        elif a_out == "L":
            assert c_out == "W"
        else:
            assert c_out == "D"


def test_h2h_empty_for_unseen_pair(db_with_h2h):
    """No previous matches → empty list, not an error."""
    from predict import _h2h_last_n
    results = _h2h_last_n(db_with_h2h, "Real Madrid", "Galatasaray", n=5)
    assert results == []


def test_h2h_score_format_uses_current_home_perspective(db_with_h2h):
    """Score string always reads 'currentHomeGoals-currentAwayGoals'."""
    from predict import _h2h_last_n
    results = _h2h_last_n(db_with_h2h, "Arsenal", "Chelsea", n=5)
    for r in results:
        # Score is "X-Y". Current home (Arsenal) goals come first.
        x, y = r["score_for_current_home"].split("-")
        x, y = int(x), int(y)
        if r["home_team"] == "Arsenal":
            assert (x, y) == (r["home_goals"], r["away_goals"])
        else:
            # Past meeting had Chelsea at home; we flip the display.
            assert (x, y) == (r["away_goals"], r["home_goals"])
