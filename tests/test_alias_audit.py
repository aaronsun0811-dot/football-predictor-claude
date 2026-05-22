"""Tests for the alias-audit / unmatched-fixture diagnostic."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from data import history_store
from data.alias_audit import find_unmatched_fixtures
from data.database import Database


@pytest.fixture
def isolated_history(tmp_path: Path, monkeypatch):
    """Redirect the history store to a temp dir so each test is hermetic."""
    legacy = tmp_path / "history.jsonl"
    shard_dir = tmp_path / "history"
    monkeypatch.setattr(history_store, "LEGACY_PATH", legacy)
    monkeypatch.setattr(history_store, "SHARD_DIR", shard_dir)
    return {"legacy": legacy, "shard_dir": shard_dir, "tmp": tmp_path}


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "audit.sqlite3")
    d.init()
    return d


def _write_history(shard_dir: Path, *rows: dict) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    # Use the month of the first row's date_taken (or today) for the shard name
    when = datetime.now(timezone.utc)
    shard = shard_dir / f"{when:%Y-%m}.jsonl"
    with shard.open("a") as f:
        for r in rows:
            r.setdefault("taken_at", when.isoformat())
            f.write(json.dumps(r) + "\n")


def _seed_db_match(db, *, league_key, dt, home, away, home_goals=1, away_goals=0):
    db.upsert_matches(
        pd.DataFrame([{
            "date": dt, "home_team": home, "away_team": away,
            "home_goals": home_goals, "away_goals": away_goals,
        }]),
        source="test", league_key=league_key, league_name=league_key,
    )


# ---------------------------------------------------------------------------
# Happy path — matched fixtures don't show up as unmatched
# ---------------------------------------------------------------------------

def test_matched_fixture_does_not_appear_in_unmatched(isolated_history, db) -> None:
    today = date(2026, 5, 19)
    yesterday = today - timedelta(days=1)
    _write_history(isolated_history["shard_dir"], {
        "date": yesterday.isoformat(), "league_key": "premier_league",
        "home_team": "Arsenal", "away_team": "Chelsea",
        "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
    })
    _seed_db_match(db, league_key="premier_league", dt=yesterday,
                   home="Arsenal", away="Chelsea")

    report = find_unmatched_fixtures(db, today=today)
    assert report["n_matched"] == 1
    assert report["n_unmatched"] == 0
    assert report["unmatched"] == []


# ---------------------------------------------------------------------------
# Reason: likely_date_mismatch — same team pair on a nearby date
# ---------------------------------------------------------------------------

def test_nearby_date_match_classified_as_date_mismatch(isolated_history, db) -> None:
    """The Arsenal-vs-Burnley case: prediction date is 5-18, actual match is 5-24."""
    today = date(2026, 5, 26)
    pred_date = date(2026, 5, 18)
    actual_date = date(2026, 5, 24)

    _write_history(isolated_history["shard_dir"], {
        "date": pred_date.isoformat(), "league_key": "premier_league",
        "home_team": "Arsenal", "away_team": "Burnley",
        "home_win": 0.9, "draw": 0.08, "away_win": 0.02,
    })
    _seed_db_match(db, league_key="premier_league", dt=actual_date,
                   home="Arsenal", away="Burnley",
                   home_goals=3, away_goals=1)

    report = find_unmatched_fixtures(db, today=today, nearby_window_days=7)
    assert report["n_unmatched"] == 1
    row = report["unmatched"][0]
    assert row["reason"] == "likely_date_mismatch"
    assert row["nearby_match"] is not None
    assert row["nearby_match"]["ds"] == actual_date.isoformat()
    assert row["nearby_match"]["days_off"] == 6
    assert row["nearby_match"]["flipped"] is False


def test_nearby_date_with_flipped_home_away_still_detected(isolated_history, db) -> None:
    """TheSportsDB sometimes flips home/away vs api-football. The detector
    checks both orderings so a flipped DB row is still considered the same fixture."""
    today = date(2026, 5, 26)
    pred_date = date(2026, 5, 18)
    actual_date = date(2026, 5, 20)
    _write_history(isolated_history["shard_dir"], {
        "date": pred_date.isoformat(), "league_key": "premier_league",
        "home_team": "Arsenal", "away_team": "Burnley",
        "home_win": 0.6, "draw": 0.25, "away_win": 0.15,
    })
    # DB has flipped order
    _seed_db_match(db, league_key="premier_league", dt=actual_date,
                   home="Burnley", away="Arsenal",
                   home_goals=1, away_goals=2)

    report = find_unmatched_fixtures(db, today=today, nearby_window_days=7)
    assert report["n_unmatched"] == 1
    row = report["unmatched"][0]
    assert row["reason"] == "likely_date_mismatch"
    assert row["nearby_match"]["flipped"] is True
    assert row["nearby_match"]["home_team"] == "Burnley"


def test_nearby_match_outside_window_not_classified_as_mismatch(isolated_history, db) -> None:
    """A 30-day-old match between the same teams is NOT the prediction's target —
    it's the reverse fixture from earlier in the season."""
    today = date(2026, 5, 26)
    pred_date = date(2026, 5, 18)
    far_date = date(2026, 1, 5)  # too far back
    _write_history(isolated_history["shard_dir"], {
        "date": pred_date.isoformat(), "league_key": "premier_league",
        "home_team": "Arsenal", "away_team": "Burnley",
        "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
    })
    _seed_db_match(db, league_key="premier_league", dt=far_date,
                   home="Arsenal", away="Burnley")

    report = find_unmatched_fixtures(db, today=today, nearby_window_days=7)
    row = report["unmatched"][0]
    assert row["reason"] != "likely_date_mismatch"
    assert row.get("nearby_match") is None


def test_closest_nearby_date_wins_over_distant_one(isolated_history, db) -> None:
    """When two same-team-pair matches are in window (e.g. cup + league),
    pick the chronologically nearest to the prediction's date."""
    today = date(2026, 6, 1)
    pred_date = date(2026, 5, 18)
    close = date(2026, 5, 20)   # 2 days off
    far = date(2026, 5, 25)     # 7 days off
    _write_history(isolated_history["shard_dir"], {
        "date": pred_date.isoformat(), "league_key": "premier_league",
        "home_team": "A", "away_team": "B",
        "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
    })
    _seed_db_match(db, league_key="premier_league", dt=close, home="A", away="B")
    _seed_db_match(db, league_key="premier_league", dt=far,   home="A", away="B")

    report = find_unmatched_fixtures(db, today=today, nearby_window_days=14)
    row = report["unmatched"][0]
    assert row["nearby_match"]["ds"] == close.isoformat()
    assert row["nearby_match"]["days_off"] == 2


# ---------------------------------------------------------------------------
# Reason: no_db_matches_on_date
# ---------------------------------------------------------------------------

def test_no_db_data_classified_correctly(isolated_history, db) -> None:
    """League has no DB rows at all → no_db_matches_on_date, no nearby match."""
    today = date(2026, 5, 19)
    yesterday = today - timedelta(days=1)
    _write_history(isolated_history["shard_dir"], {
        "date": yesterday.isoformat(), "league_key": "mls",
        "home_team": "LAFC", "away_team": "Galaxy",
        "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
    })
    # DB is empty for MLS
    report = find_unmatched_fixtures(db, today=today)
    row = report["unmatched"][0]
    assert row["reason"] == "no_db_matches_on_date"
    assert row.get("nearby_match") is None


# ---------------------------------------------------------------------------
# Reason taxonomy: by_reason rollup
# ---------------------------------------------------------------------------

def test_by_reason_counts_in_report(isolated_history, db) -> None:
    today = date(2026, 6, 1)
    # 1× likely_date_mismatch
    _write_history(isolated_history["shard_dir"], {
        "date": "2026-05-18", "league_key": "premier_league",
        "home_team": "Arsenal", "away_team": "Burnley",
        "home_win": 0.7, "draw": 0.2, "away_win": 0.1,
    })
    _seed_db_match(db, league_key="premier_league", dt=date(2026, 5, 24),
                   home="Arsenal", away="Burnley")
    # 1× no_db_matches_on_date
    _write_history(isolated_history["shard_dir"], {
        "date": "2026-05-18", "league_key": "mls",
        "home_team": "LAFC", "away_team": "Galaxy",
        "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
    })
    report = find_unmatched_fixtures(db, today=today, nearby_window_days=7)
    assert report["n_unmatched"] == 2
    assert report["by_reason"] == {
        "likely_date_mismatch": 1,
        "no_db_matches_on_date": 1,
    }


# ---------------------------------------------------------------------------
# Honors days_back filter
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Post-hoc fd.org fallback (check_fdorg=True)
# ---------------------------------------------------------------------------

def _fdorg_finished_frame(rows):
    """Build a fd.org-shaped DataFrame for mocking ``fetch_matches``."""
    import pandas as _pd
    from datetime import date as _date
    return _pd.DataFrame([{
        "date": r["date"] if isinstance(r["date"], _date) else _date.fromisoformat(r["date"]),
        "home_team": r["home_team"], "away_team": r["away_team"],
        "home_goals": r.get("home_goals", 1),
        "away_goals": r.get("away_goals", 0),
        "status": "FINISHED",
        "matchday": None, "stage": None,
    } for r in rows])


def test_check_fdorg_upgrades_reason_when_fdorg_has_nearby_match(isolated_history, db, monkeypatch, tmp_path) -> None:
    """The Arsenal-vs-Burnley case: local DB has nothing, but fd.org has the
    pair on a nearby date. The fdorg fallback upgrades the reason to
    ``likely_date_mismatch`` with fd.org as the source.
    """
    from unittest import mock as _mock
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake")
    today = date(2026, 5, 30)
    pred_date = date(2026, 5, 18)
    actual_date = date(2026, 5, 24)
    _write_history(isolated_history["shard_dir"], {
        "date": pred_date.isoformat(), "league_key": "premier_league",
        "home_team": "Arsenal", "away_team": "Burnley",
        "home_win": 0.7, "draw": 0.2, "away_win": 0.1,
    })
    # Local DB has nothing for PL on or around 5-18

    fdorg_response = _fdorg_finished_frame([
        {"date": actual_date, "home_team": "Arsenal", "away_team": "Burnley",
         "home_goals": 3, "away_goals": 1},
    ])
    with _mock.patch(
        "scrape.football_data_org.FootballDataOrgClient.fetch_matches",
        return_value=fdorg_response,
    ):
        report = find_unmatched_fixtures(
            db, today=today, nearby_window_days=7,
            check_fdorg=True, cache_dir=tmp_path,
        )

    assert report["n_unmatched"] == 1
    row = report["unmatched"][0]
    assert row["reason"] == "likely_date_mismatch"
    assert row["nearby_match"] is not None
    assert row["nearby_match"]["source"] == "football-data.org"
    assert row["nearby_match"]["ds"] == actual_date.isoformat()
    assert row["nearby_match"]["days_off"] == 6
    assert row["nearby_match"]["home_goals"] == 3


def test_check_fdorg_disabled_by_default(isolated_history, db, monkeypatch, tmp_path) -> None:
    """Without --check-fdorg, fd.org should never be queried."""
    from unittest import mock as _mock
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake")
    today = date(2026, 5, 30)
    _write_history(isolated_history["shard_dir"], {
        "date": "2026-05-18", "league_key": "premier_league",
        "home_team": "Arsenal", "away_team": "Burnley",
        "home_win": 0.7, "draw": 0.2, "away_win": 0.1,
    })

    with _mock.patch(
        "scrape.football_data_org.FootballDataOrgClient.fetch_matches",
    ) as mocked:
        find_unmatched_fixtures(db, today=today)  # check_fdorg=False default
    mocked.assert_not_called()


def test_check_fdorg_skips_leagues_not_in_fdorg(isolated_history, db, monkeypatch, tmp_path) -> None:
    """An MLS unresolved fixture must NOT trigger an fd.org call — fd.org
    doesn't cover MLS, so it'd be wasted quota."""
    from unittest import mock as _mock
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake")
    _write_history(isolated_history["shard_dir"], {
        "date": "2026-05-18", "league_key": "mls",
        "home_team": "LAFC", "away_team": "Galaxy",
        "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
    })

    with _mock.patch(
        "scrape.football_data_org.FootballDataOrgClient.fetch_matches",
    ) as mocked:
        find_unmatched_fixtures(
            db, today=date(2026, 5, 30),
            check_fdorg=True, cache_dir=tmp_path,
        )
    # MLS isn't in fd.org's catalog, so nothing should be queried
    mocked.assert_not_called()


def test_check_fdorg_silent_failure_doesnt_break_report(isolated_history, db, monkeypatch, tmp_path) -> None:
    """fd.org rate-limit / 403 / network errors leave rows unchanged."""
    from unittest import mock as _mock
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake")
    _write_history(isolated_history["shard_dir"], {
        "date": "2026-05-18", "league_key": "premier_league",
        "home_team": "Arsenal", "away_team": "Burnley",
        "home_win": 0.7, "draw": 0.2, "away_win": 0.1,
    })
    with _mock.patch(
        "scrape.football_data_org.FootballDataOrgClient.fetch_matches",
        side_effect=RuntimeError("boom"),
    ):
        report = find_unmatched_fixtures(
            db, today=date(2026, 5, 30),
            check_fdorg=True, cache_dir=tmp_path,
        )
    # Original reason preserved, no crash
    assert report["n_unmatched"] == 1
    row = report["unmatched"][0]
    assert row["reason"] == "no_db_matches_on_date"
    assert row.get("nearby_match") is None


def test_check_fdorg_no_key_silently_skips(isolated_history, db, monkeypatch, tmp_path) -> None:
    """Without an API key, the fdorg path is a no-op (not an error)."""
    for var in ("FOOTBALL_DATA_ORG_KEY", "FOOTBALL_DATA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    _write_history(isolated_history["shard_dir"], {
        "date": "2026-05-18", "league_key": "premier_league",
        "home_team": "Arsenal", "away_team": "Burnley",
        "home_win": 0.7, "draw": 0.2, "away_win": 0.1,
    })
    report = find_unmatched_fixtures(
        db, today=date(2026, 5, 30),
        check_fdorg=True, cache_dir=tmp_path,
    )
    assert report["n_unmatched"] == 1
    # Reason unchanged, no crash
    assert report["unmatched"][0]["reason"] == "no_db_matches_on_date"


def test_days_back_filters_old_history(isolated_history, db) -> None:
    today = date(2026, 5, 19)
    # Fixture from 60 days ago should be excluded with days_back=14
    _write_history(isolated_history["shard_dir"], {
        "date": (today - timedelta(days=60)).isoformat(),
        "league_key": "premier_league",
        "home_team": "A", "away_team": "B",
        "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
    })
    report = find_unmatched_fixtures(db, today=today, days_back=14)
    assert report["n_past_fixtures"] == 0
    assert report["n_unmatched"] == 0
