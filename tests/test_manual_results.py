"""Tests for the manual-result entry flow."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from data.database import Database
from data.manual_results import list_recent_manual_results, submit_manual_result


@pytest.fixture
def fresh_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "manual.sqlite3")
    db.init()
    return db


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_submit_valid_result_inserts_row(fresh_db: Database) -> None:
    today = date.today()
    yesterday = today - timedelta(days=1)
    report = submit_manual_result(
        fresh_db,
        league="premier_league",
        date=yesterday,
        home_team="Arsenal",
        away_team="Chelsea",
        home_goals=2,
        away_goals=1,
        today=today,
    )
    assert report["inserted"] == 1
    assert report["league_key"] == "premier_league"
    assert report["score"] == "2-1"
    assert report["result"] == "H"

    # Round-trip: row exists with source='manual'
    matches = fresh_db.fetch_matches(league_key="premier_league")
    assert len(matches) == 1
    assert matches.iloc[0]["source"] == "manual"


def test_submit_accepts_chinese_alias(fresh_db: Database) -> None:
    today = date.today()
    report = submit_manual_result(
        fresh_db,
        league="英超",
        date=today - timedelta(days=1),
        home_team="Liverpool",
        away_team="Tottenham",
        home_goals=3,
        away_goals=3,
        today=today,
    )
    assert report["league_key"] == "premier_league"
    assert report["result"] == "D"


def test_submit_canonicalizes_team_names(fresh_db: Database) -> None:
    """'Real Madrid CF' should land as 'Real Madrid' (canonical)."""
    report = submit_manual_result(
        fresh_db,
        league="la_liga",
        date=date.today() - timedelta(days=1),
        home_team="Real Madrid CF",
        away_team="Barcelona",
        home_goals=2,
        away_goals=2,
        today=date.today(),
    )
    assert report["home_team"] == "Real Madrid"


def test_submit_derives_season_from_date(fresh_db: Database) -> None:
    # August 2025 → 2025/26 season → season = "2025"
    report = submit_manual_result(
        fresh_db,
        league="premier_league",
        date=date(2025, 8, 20),
        home_team="Arsenal",
        away_team="Chelsea",
        home_goals=1,
        away_goals=0,
        today=date(2025, 8, 25),
    )
    assert report["season"] == "2025"

    # February 2026 still belongs to the 2025/26 season → "2025"
    report2 = submit_manual_result(
        fresh_db,
        league="premier_league",
        date=date(2026, 2, 10),
        home_team="Liverpool",
        away_team="Spurs",
        home_goals=0,
        away_goals=2,
        today=date(2026, 2, 15),
    )
    assert report2["season"] == "2025"


def test_submit_derives_calendar_year_season_for_csl(fresh_db: Database) -> None:
    """CSL (China) is calendar-year, not Aug-Jun."""
    report = submit_manual_result(
        fresh_db,
        league="chinese_super_league",
        date=date(2026, 5, 1),
        home_team="Shanghai Port",
        away_team="Beijing Guoan",
        home_goals=1,
        away_goals=1,
        today=date(2026, 5, 10),
    )
    assert report["season"] == "2026"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_unknown_league_raises(fresh_db: Database) -> None:
    with pytest.raises(ValueError, match="Unknown league"):
        submit_manual_result(
            fresh_db,
            league="not_a_real_league",
            date=date.today() - timedelta(days=1),
            home_team="A", away_team="B",
            home_goals=1, away_goals=0,
            today=date.today(),
        )


def test_future_date_raises(fresh_db: Database) -> None:
    today = date.today()
    with pytest.raises(ValueError, match="strictly before today"):
        submit_manual_result(
            fresh_db,
            league="premier_league",
            date=today + timedelta(days=1),
            home_team="A", away_team="B",
            home_goals=1, away_goals=0,
            today=today,
        )


def test_today_match_is_rejected(fresh_db: Database) -> None:
    """A match on the audit cutoff day is still "live" — reject it."""
    today = date.today()
    with pytest.raises(ValueError, match="strictly before today"):
        submit_manual_result(
            fresh_db,
            league="premier_league",
            date=today,
            home_team="A", away_team="B",
            home_goals=1, away_goals=0,
            today=today,
        )


def test_negative_score_raises(fresh_db: Database) -> None:
    with pytest.raises(ValueError, match="home_goals"):
        submit_manual_result(
            fresh_db,
            league="premier_league",
            date=date.today() - timedelta(days=1),
            home_team="A", away_team="B",
            home_goals=-1, away_goals=0,
            today=date.today(),
        )


def test_absurd_score_raises(fresh_db: Database) -> None:
    with pytest.raises(ValueError, match=r"\[0, 30\]"):
        submit_manual_result(
            fresh_db,
            league="premier_league",
            date=date.today() - timedelta(days=1),
            home_team="A", away_team="B",
            home_goals=100, away_goals=0,
            today=date.today(),
        )


def test_same_team_both_sides_raises(fresh_db: Database) -> None:
    with pytest.raises(ValueError, match="must differ"):
        submit_manual_result(
            fresh_db,
            league="premier_league",
            date=date.today() - timedelta(days=1),
            home_team="Arsenal", away_team="Arsenal",
            home_goals=1, away_goals=0,
            today=date.today(),
        )


# ---------------------------------------------------------------------------
# Upsert behavior: re-submitting same fixture updates, doesn't duplicate
# ---------------------------------------------------------------------------

def test_resubmit_same_fixture_updates_in_place(fresh_db: Database) -> None:
    today = date.today()
    yesterday = today - timedelta(days=1)
    submit_manual_result(
        fresh_db, league="premier_league", date=yesterday,
        home_team="Arsenal", away_team="Chelsea",
        home_goals=2, away_goals=1, today=today,
    )
    # User realizes the score was wrong, re-submits
    submit_manual_result(
        fresh_db, league="premier_league", date=yesterday,
        home_team="Arsenal", away_team="Chelsea",
        home_goals=3, away_goals=1, today=today,
    )
    matches = fresh_db.fetch_matches(league_key="premier_league")
    assert len(matches) == 1
    assert int(matches.iloc[0]["home_goals"]) == 3


# ---------------------------------------------------------------------------
# Listing endpoint
# ---------------------------------------------------------------------------

def test_list_recent_manual_results(fresh_db: Database) -> None:
    today = date.today()
    submit_manual_result(
        fresh_db, league="premier_league", date=today - timedelta(days=1),
        home_team="Arsenal", away_team="Chelsea",
        home_goals=2, away_goals=1, today=today,
    )
    submit_manual_result(
        fresh_db, league="la_liga", date=today - timedelta(days=2),
        home_team="Real Madrid", away_team="Barcelona",
        home_goals=1, away_goals=1, today=today,
    )
    rows = list_recent_manual_results(fresh_db, limit=10)
    assert len(rows) == 2
    keys = {r["league_key"] for r in rows}
    assert keys == {"premier_league", "la_liga"}


def test_list_empty_when_no_manual_results(fresh_db: Database) -> None:
    assert list_recent_manual_results(fresh_db) == []
