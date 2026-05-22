"""Tests for the /upcoming fixture-date cross-check."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from data.fixture_date_check import attach_date_warnings, cross_check_dates


# ---------------------------------------------------------------------------
# cross_check_dates — the core lookup
# ---------------------------------------------------------------------------

def _fdorg_frame(rows):
    """Build a fd.org-shaped DataFrame for mocking ``fetch_matches``."""
    return pd.DataFrame([
        {
            "date": pd.to_datetime(r["date"]).date(),
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "home_goals": None, "away_goals": None,
            "status": r.get("status", "SCHEDULED"),
            "matchday": None, "stage": None,
        }
        for r in rows
    ])


@pytest.fixture
def fixtures_df():
    """A typical /upcoming fixture set."""
    return pd.DataFrame([
        {"date": "2026-05-18", "league_key": "premier_league",
         "home_team": "Arsenal", "away_team": "Burnley"},
        {"date": "2026-05-17", "league_key": "premier_league",
         "home_team": "Brentford", "away_team": "Crystal Palace"},
        # A league fd.org doesn't cover
        {"date": "2026-05-18", "league_key": "mls",
         "home_team": "LAFC", "away_team": "Galaxy"},
    ])


def test_no_api_key_returns_empty_dict(fixtures_df, monkeypatch, tmp_path):
    for var in ("FOOTBALL_DATA_ORG_KEY", "FOOTBALL_DATA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    result = cross_check_dates(fixtures_df, cache_dir=tmp_path)
    assert result == {}


def test_empty_fixtures_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake")
    result = cross_check_dates(pd.DataFrame(), cache_dir=tmp_path)
    assert result == {}


def test_warning_when_fdorg_date_differs(fixtures_df, monkeypatch, tmp_path):
    """fd.org says Arsenal vs Burnley is on 5-24, TSDB has it on 5-18 → warning."""
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake")
    fdorg_scheduled = _fdorg_frame([
        {"date": "2026-05-24", "home_team": "Arsenal", "away_team": "Burnley"},
    ])
    with mock.patch(
        "scrape.football_data_org.FootballDataOrgClient.fetch_matches",
        return_value=fdorg_scheduled,
    ):
        result = cross_check_dates(fixtures_df, cache_dir=tmp_path,
                                   today=date(2026, 5, 19))
    key = ("premier_league", "Arsenal", "Burnley")
    assert key in result
    assert result[key]["match_in_fdorg"] is True
    assert result[key]["fdorg_date"] == "2026-05-24"
    assert result[key]["days_off"] == 6  # +6 = fd.org is later


def test_confirmed_when_fdorg_date_matches(fixtures_df, monkeypatch, tmp_path):
    """fd.org and TSDB agree → days_off=0."""
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake")
    fdorg_scheduled = _fdorg_frame([
        {"date": "2026-05-18", "home_team": "Arsenal", "away_team": "Burnley"},
        {"date": "2026-05-17", "home_team": "Brentford", "away_team": "Crystal Palace"},
    ])
    with mock.patch(
        "scrape.football_data_org.FootballDataOrgClient.fetch_matches",
        return_value=fdorg_scheduled,
    ):
        result = cross_check_dates(fixtures_df, cache_dir=tmp_path,
                                   today=date(2026, 5, 19))
    assert result[("premier_league", "Arsenal", "Burnley")]["days_off"] == 0
    assert result[("premier_league", "Brentford", "Crystal Palace")]["days_off"] == 0


def test_flipped_home_away_still_matched(fixtures_df, monkeypatch, tmp_path):
    """If fd.org has the same teams but home/away flipped, mark flipped=True."""
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake")
    fdorg_scheduled = _fdorg_frame([
        # Home/away reversed vs our fixture
        {"date": "2026-05-18", "home_team": "Burnley", "away_team": "Arsenal"},
    ])
    with mock.patch(
        "scrape.football_data_org.FootballDataOrgClient.fetch_matches",
        return_value=fdorg_scheduled,
    ):
        result = cross_check_dates(fixtures_df, cache_dir=tmp_path,
                                   today=date(2026, 5, 19))
    arsenal_burnley = result[("premier_league", "Arsenal", "Burnley")]
    assert arsenal_burnley["match_in_fdorg"] is True
    assert arsenal_burnley["flipped"] is True


def test_leagues_outside_fdorg_skipped_silently(fixtures_df, monkeypatch, tmp_path):
    """MLS is not in LEAGUE_KEY_TO_CODE — should never appear in the warnings dict."""
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake")
    with mock.patch(
        "scrape.football_data_org.FootballDataOrgClient.fetch_matches",
        return_value=_fdorg_frame([]),
    ):
        result = cross_check_dates(fixtures_df, cache_dir=tmp_path,
                                   today=date(2026, 5, 19))
    assert ("mls", "LAFC", "Galaxy") not in result


def test_fdorg_error_returns_empty(fixtures_df, monkeypatch, tmp_path):
    """Network/auth errors on fd.org should never blow up /upcoming."""
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake")
    with mock.patch(
        "scrape.football_data_org.FootballDataOrgClient.fetch_matches",
        side_effect=RuntimeError("boom"),
    ):
        result = cross_check_dates(fixtures_df, cache_dir=tmp_path,
                                   today=date(2026, 5, 19))
    # No warnings collected, but also no crash
    assert isinstance(result, dict)


def test_unknown_when_fdorg_has_no_record_for_pair(fixtures_df, monkeypatch, tmp_path):
    """fd.org returns SCHEDULED for the league but doesn't list this specific
    pair → match_in_fdorg=False (e.g. cup match, or fixture already finished)."""
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake")
    fdorg_scheduled = _fdorg_frame([
        # PL but a different fixture
        {"date": "2026-05-22", "home_team": "Liverpool", "away_team": "Tottenham"},
    ])
    with mock.patch(
        "scrape.football_data_org.FootballDataOrgClient.fetch_matches",
        return_value=fdorg_scheduled,
    ):
        result = cross_check_dates(fixtures_df, cache_dir=tmp_path,
                                   today=date(2026, 5, 19))
    key = ("premier_league", "Arsenal", "Burnley")
    assert key in result
    assert result[key]["match_in_fdorg"] is False
    assert result[key]["fdorg_date"] is None


# ---------------------------------------------------------------------------
# attach_date_warnings — the UI shape
# ---------------------------------------------------------------------------

def test_attach_marks_confirmed_for_zero_days_off():
    fixtures = [{"league_key": "premier_league", "home_team": "Arsenal", "away_team": "Burnley"}]
    warnings = {("premier_league", "Arsenal", "Burnley"): {
        "match_in_fdorg": True, "fdorg_date": "2026-05-18",
        "days_off": 0, "flipped": False,
    }}
    attach_date_warnings(fixtures, warnings)
    assert fixtures[0]["date_check"]["status"] == "confirmed"


def test_attach_marks_warning_for_nonzero_days_off():
    fixtures = [{"league_key": "premier_league", "home_team": "Arsenal", "away_team": "Burnley"}]
    warnings = {("premier_league", "Arsenal", "Burnley"): {
        "match_in_fdorg": True, "fdorg_date": "2026-05-24",
        "days_off": 6, "flipped": False,
    }}
    attach_date_warnings(fixtures, warnings)
    assert fixtures[0]["date_check"]["status"] == "warning"
    assert fixtures[0]["date_check"]["days_off"] == 6
    assert fixtures[0]["date_check"]["fdorg_date"] == "2026-05-24"


def test_attach_marks_not_covered_for_non_fdorg_leagues():
    fixtures = [{"league_key": "mls", "home_team": "LAFC", "away_team": "Galaxy"}]
    attach_date_warnings(fixtures, {})
    assert fixtures[0]["date_check"]["status"] == "not_covered"


def test_attach_marks_unknown_when_pair_missing():
    fixtures = [{"league_key": "premier_league", "home_team": "Arsenal", "away_team": "Burnley"}]
    # warnings dict has the league but doesn't include this pair
    attach_date_warnings(fixtures, {})
    assert fixtures[0]["date_check"]["status"] == "unknown"


def test_attach_marks_unknown_when_fdorg_lacks_match():
    fixtures = [{"league_key": "premier_league", "home_team": "Arsenal", "away_team": "Burnley"}]
    warnings = {("premier_league", "Arsenal", "Burnley"): {
        "match_in_fdorg": False, "fdorg_date": None,
        "days_off": None, "flipped": False,
    }}
    attach_date_warnings(fixtures, warnings)
    assert fixtures[0]["date_check"]["status"] == "unknown"
