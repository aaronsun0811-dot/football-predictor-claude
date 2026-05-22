"""Tests for data/result_backfill.py — uses mocked HTTP clients, no real network."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd
import pytest

from data.database import Database
from data.result_backfill import _season_for, backfill_recent_results


# ---------------------------------------------------------------------------
# _season_for — pure helper, easy to assert on
# ---------------------------------------------------------------------------

def _fake_league(key: str, *, country: str | None = None, api_football_id: int | None = 1):
    """Build a duck-typed League object good enough for the helpers."""
    return mock.Mock(key=key, country=country, api_football_id=api_football_id, name=key)


def test_season_for_calendar_year_league_returns_year() -> None:
    league = _fake_league("j1_league", country="JPN")
    # March 2024 → 2024 (calendar season).
    assert _season_for(league, date(2024, 3, 15)) == 2024


def test_season_for_european_league_uses_split_year() -> None:
    league = _fake_league("premier_league", country="ENG")
    # August 2024 starts the 2024/25 season → 2024 (also within free-plan cap).
    assert _season_for(league, date(2024, 8, 15)) == 2024
    # February 2024 still belongs to 2023/24 season → 2023.
    assert _season_for(league, date(2024, 2, 1)) == 2023


def test_season_for_clamps_to_free_plan_ceiling() -> None:
    league = _fake_league("premier_league", country="ENG")
    # Spring 2027 → 2026 split-year, but free plan tops at 2024.
    assert _season_for(league, date(2027, 3, 1)) <= 2024
    # Spring 2026 also clamps (would naturally be 2025, but free plan blocks it).
    assert _season_for(league, date(2026, 2, 1)) <= 2024


# ---------------------------------------------------------------------------
# backfill_recent_results — high-level routing tests with mocks
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "bf.sqlite3")
    db.init()
    return db


def _fake_fdorg_frame(matchday: date) -> pd.DataFrame:
    """A single finished match in the football-data.org shape."""
    return pd.DataFrame([{
        "date": matchday,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "home_goals": 2,
        "away_goals": 1,
        "status": "FINISHED",
        "matchday": 38,
        "stage": "REGULAR_SEASON",
    }])


def _disable_all_clients(monkeypatch) -> None:
    """Block ALL three data-source clients so no real HTTP fires.

    Note: ``monkeypatch.delenv`` is not enough on its own — ``get_settings()``
    re-loads ``.env`` via dotenv with ``override=False``, which can re-populate
    a key we just deleted. Patching ``client_from_env`` is the only sure way
    to disarm the API-Football route in tests.

    The fd.co.uk CSV tier needs no API key, so deleting env vars doesn't disarm
    it — we patch ``fdcouk_fetch_season`` to a no-op too.
    """
    for var in ("FOOTBALL_DATA_ORG_KEY", "FOOTBALL_DATA_API_KEY",
                "FOOTBALL_API_KEY", "API_FOOTBALL_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        "data.result_backfill.client_from_env",
        lambda: None,
    )
    monkeypatch.setattr(
        "data.result_backfill.fdcouk_fetch_season",
        lambda *args, **kwargs: pd.DataFrame(),
    )


def test_returns_report_shape_even_with_no_keys(empty_db, monkeypatch) -> None:
    """No env keys + fd.co.uk patched empty → no inserts, no errors, well-formed dict.

    fd.co.uk doesn't need an API key so it's "reached" for every league with a
    ``football_data_code`` (yielding 0 inserts because the patch returns empty).
    Leagues without any registered code stay ``skipped: no_source``.
    """
    _disable_all_clients(monkeypatch)
    report = backfill_recent_results(empty_db, days_back=7)
    assert "generated_at" in report
    assert "window" in report
    assert "totals" in report
    assert "leagues" in report
    assert report["totals"]["inserted"] == 0
    assert report["totals"]["errors"] == 0
    # Every league is reported. The ones with a fd.co.uk code show up as reached;
    # the rest (MLS, J1, K1, continental cups...) are skipped no_source.
    skipped = [r for r in report["leagues"] if r.get("skipped") == "no_source"]
    reached = [r for r in report["leagues"] if r.get("source")]
    assert len(skipped) + len(reached) == len(report["leagues"])
    assert all(r.get("inserted") == 0 for r in reached)


def test_window_respects_days_back(empty_db, monkeypatch) -> None:
    _disable_all_clients(monkeypatch)
    today = date(2026, 5, 18)
    report = backfill_recent_results(empty_db, days_back=10, as_of=today)
    assert report["window"]["from"] == (today - timedelta(days=10)).isoformat()
    assert report["window"]["to"] == today.isoformat()


def test_fdorg_route_used_when_key_present(empty_db, monkeypatch, tmp_path) -> None:
    """When FOOTBALL_DATA_ORG_KEY is set, leagues in LEAGUE_KEY_TO_CODE get fd.org."""
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake-key-for-test")
    monkeypatch.delenv("FOOTBALL_API_KEY", raising=False)
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    # Disable downstream tiers to keep the test offline for leagues without fd.org code.
    monkeypatch.setattr("data.result_backfill.client_from_env", lambda: None)
    monkeypatch.setattr(
        "data.result_backfill.fdcouk_fetch_season",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    today = date(2026, 5, 18)
    fake_frame = _fake_fdorg_frame(today - timedelta(days=1))

    with mock.patch(
        "scrape.football_data_org.FootballDataOrgClient.fetch_matches",
        return_value=fake_frame,
    ) as patched:
        report = backfill_recent_results(
            empty_db, days_back=7, cache_dir=tmp_path, as_of=today,
        )
    assert patched.called
    # At least one league was reached via football-data.org
    fdorg_rows = [r for r in report["leagues"] if r.get("source") == "football-data.org"]
    assert fdorg_rows, "expected football-data.org to be used for at least one league"
    # Premier League is in LEAGUE_KEY_TO_CODE → should have been called
    pl = next((r for r in fdorg_rows if r["league_key"] == "premier_league"), None)
    assert pl is not None
    assert pl["inserted"] >= 1


def test_fdorg_error_falls_through_to_api_football(empty_db, monkeypatch, tmp_path) -> None:
    """If fd.org throws on a league, the report records the error."""
    from scrape.football_data_org import FootballDataOrgError

    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake-key")
    monkeypatch.delenv("FOOTBALL_API_KEY", raising=False)
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.setattr("data.result_backfill.client_from_env", lambda: None)
    monkeypatch.setattr(
        "data.result_backfill.fdcouk_fetch_season",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    with mock.patch(
        "scrape.football_data_org.FootballDataOrgClient.fetch_matches",
        side_effect=FootballDataOrgError("403 Forbidden"),
    ):
        report = backfill_recent_results(
            empty_db, days_back=7, cache_dir=tmp_path, as_of=date(2026, 5, 18),
        )
    err_rows = [r for r in report["leagues"] if r.get("error")]
    assert err_rows
    assert report["totals"]["errors"] >= 1


def test_empty_fdorg_response_does_not_keyerror(empty_db, monkeypatch, tmp_path) -> None:
    """Regression: a league with no matches in the window used to KeyError on the
    dropna inside ``_matches_to_frame`` because the empty DataFrame had no columns.
    """
    from scrape.football_data_org import _matches_to_frame

    # Direct unit check: empty list → empty frame with expected columns, no crash.
    empty = _matches_to_frame([])
    assert empty.empty
    for col in ("date", "home_team", "away_team", "home_goals", "away_goals"):
        assert col in empty.columns

    # Integration: backfill across all leagues with mock returning empty for all.
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake-key")
    monkeypatch.setattr("data.result_backfill.client_from_env", lambda: None)
    monkeypatch.setattr(
        "data.result_backfill.fdcouk_fetch_season",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    with mock.patch(
        "scrape.football_data_org.FootballDataOrgClient.fetch_matches",
        return_value=empty,
    ):
        report = backfill_recent_results(
            empty_db, days_back=3, cache_dir=tmp_path, as_of=date(2026, 5, 18),
        )
    # No errors should have surfaced from empty frames.
    fdorg_errors = [
        r for r in report["leagues"]
        if r.get("source") == "football-data.org" and r.get("error")
    ]
    assert not fdorg_errors, f"unexpected fd.org errors: {fdorg_errors}"


def test_progress_callback_fires_once_per_league(empty_db, monkeypatch, tmp_path) -> None:
    """Each league row appended to the report should also pass through the callback."""
    _disable_all_clients(monkeypatch)
    seen: list[dict[str, Any]] = []
    report = backfill_recent_results(
        empty_db, days_back=3, cache_dir=tmp_path, as_of=date(2026, 5, 18),
        progress_callback=seen.append,
        persist_report=False,
    )
    assert len(seen) == len(report["leagues"])
    # Every callback row matches its corresponding leagues entry.
    for cb_row, lr_row in zip(seen, report["leagues"]):
        assert cb_row == lr_row


def test_progress_callback_exception_is_swallowed(empty_db, monkeypatch, tmp_path) -> None:
    """A buggy callback must NOT kill the batch."""
    _disable_all_clients(monkeypatch)

    def explosive(_row):
        raise RuntimeError("bug in user code")

    # Should not raise
    report = backfill_recent_results(
        empty_db, days_back=3, cache_dir=tmp_path, as_of=date(2026, 5, 18),
        progress_callback=explosive, persist_report=False,
    )
    assert len(report["leagues"]) > 0  # batch still completed


def test_report_persisted_to_reports_dir(empty_db, monkeypatch, tmp_path) -> None:
    _disable_all_clients(monkeypatch)
    reports_dir = tmp_path / "reports"
    today = date(2026, 5, 18)
    backfill_recent_results(
        empty_db, days_back=3, cache_dir=tmp_path, as_of=today,
        reports_dir=reports_dir,
    )
    expected = reports_dir / "2026-05-18.json"
    assert expected.exists()
    body = json.loads(expected.read_text())
    # No inserts (fd.co.uk patched empty, no other sources reachable)
    assert body["totals"]["inserted"] == 0
    assert "started_at" in body
    assert "duration_s" in body


def test_report_persistence_can_be_disabled(empty_db, monkeypatch, tmp_path) -> None:
    _disable_all_clients(monkeypatch)
    reports_dir = tmp_path / "reports"
    backfill_recent_results(
        empty_db, days_back=3, cache_dir=tmp_path, as_of=date(2026, 5, 18),
        reports_dir=reports_dir, persist_report=False,
    )
    assert not reports_dir.exists() or not any(reports_dir.iterdir())


def test_load_latest_report_returns_newest(tmp_path) -> None:
    from data.result_backfill import load_latest_report, save_report

    save_report({"totals": {"inserted": 1}}, reports_dir=tmp_path, today=date(2026, 5, 10))
    save_report({"totals": {"inserted": 2}}, reports_dir=tmp_path, today=date(2026, 5, 17))
    save_report({"totals": {"inserted": 3}}, reports_dir=tmp_path, today=date(2026, 5, 15))
    latest = load_latest_report(reports_dir=tmp_path)
    assert latest["totals"]["inserted"] == 2  # 2026-05-17 file


def test_load_latest_report_returns_none_when_no_reports(tmp_path) -> None:
    from data.result_backfill import load_latest_report
    # Empty dir
    assert load_latest_report(reports_dir=tmp_path) is None
    # Non-existent dir
    assert load_latest_report(reports_dir=tmp_path / "nope") is None


def test_load_recent_reports_skips_corrupt_files(tmp_path) -> None:
    from data.result_backfill import load_recent_reports, save_report

    save_report({"x": 1}, reports_dir=tmp_path, today=date(2026, 5, 17))
    (tmp_path / "2026-05-16.json").write_text("not-json-at-all")
    save_report({"x": 2}, reports_dir=tmp_path, today=date(2026, 5, 15))

    reports = load_recent_reports(reports_dir=tmp_path, limit=10)
    assert [r["x"] for r in reports] == [1, 2]  # corrupt 5-16 dropped


def test_load_recent_reports_respects_limit(tmp_path) -> None:
    from data.result_backfill import load_recent_reports, save_report

    for d in (10, 11, 12, 13, 14, 15):
        save_report({"d": d}, reports_dir=tmp_path, today=date(2026, 5, d))
    reports = load_recent_reports(reports_dir=tmp_path, limit=3)
    # Newest 3 = 15, 14, 13
    assert [r["d"] for r in reports] == [15, 14, 13]


def test_fdcouk_tier_used_for_leagues_without_fdorg_code(empty_db, monkeypatch, tmp_path) -> None:
    """A league that fd.org doesn't cover but fd.co.uk does (Belgian Pro, B1) —
    the router should fall through to the fd.co.uk CSV tier instead of going
    straight to API-Football and burning quota.
    """
    monkeypatch.delenv("FOOTBALL_DATA_ORG_KEY", raising=False)
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    monkeypatch.setattr("data.result_backfill.client_from_env", lambda: None)

    today = date(2026, 5, 18)
    fake_csv_frame = pd.DataFrame([
        # In-window
        {"date": today - timedelta(days=1), "home_team": "Anderlecht",
         "away_team": "Club Brugge", "home_goals": 1, "away_goals": 2,
         "league_code": "B1", "result": "A"},
        # Outside window — should be filtered
        {"date": today - timedelta(days=60), "home_team": "Anderlecht",
         "away_team": "Genk", "home_goals": 0, "away_goals": 0,
         "league_code": "B1", "result": "D"},
    ])
    with mock.patch(
        "data.result_backfill.fdcouk_fetch_season",
        return_value=fake_csv_frame,
    ) as patched:
        report = backfill_recent_results(
            empty_db, days_back=7, cache_dir=tmp_path, as_of=today,
            persist_report=False,
        )

    # fd.co.uk was called for at least one league (Belgian Pro at minimum)
    assert patched.called
    fdcouk_rows = [r for r in report["leagues"]
                   if r.get("source") == "football-data.co.uk"]
    assert fdcouk_rows, "expected fd.co.uk to be used for at least one league"
    bp = next((r for r in fdcouk_rows if r["league_key"] == "belgian_pro"), None)
    assert bp is not None, f"belgian_pro should route to fd.co.uk; got: {fdcouk_rows}"
    # Only the 1 in-window match was inserted, not the 60-day-old one.
    assert bp["inserted"] == 1
    assert bp["fetched"] == 1


def test_fdcouk_tier_skipped_for_leagues_without_code(empty_db, monkeypatch, tmp_path) -> None:
    """A league with ``football_data_code: null`` (e.g. MLS) should NOT trigger
    the fd.co.uk fetch — saves an unnecessary HTTP roundtrip."""
    monkeypatch.delenv("FOOTBALL_DATA_ORG_KEY", raising=False)
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    monkeypatch.setattr("data.result_backfill.client_from_env", lambda: None)

    with mock.patch("data.result_backfill.fdcouk_fetch_season") as patched:
        backfill_recent_results(
            empty_db, days_back=3, cache_dir=tmp_path,
            as_of=date(2026, 5, 18), persist_report=False,
        )
    # MLS / J1 / K1 etc. have no fd.co.uk code, so the helper should only have
    # been called for the leagues that DO have a code.
    called_codes = {call.args[0] for call in patched.call_args_list}
    # We expect at least one European code present and MLS code absent.
    assert "B1" in called_codes
    assert "MLS" not in called_codes  # would be nonsense — no such fd.co.uk code


def test_fdcouk_error_falls_through_to_api_football(empty_db, monkeypatch, tmp_path) -> None:
    """If fd.co.uk throws (e.g. 404), the league falls through to api-football."""
    monkeypatch.delenv("FOOTBALL_DATA_ORG_KEY", raising=False)
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    monkeypatch.setattr("data.result_backfill.client_from_env", lambda: None)

    with mock.patch(
        "data.result_backfill.fdcouk_fetch_season",
        side_effect=RuntimeError("HTTP 404"),
    ):
        report = backfill_recent_results(
            empty_db, days_back=3, cache_dir=tmp_path,
            as_of=date(2026, 5, 18), persist_report=False,
        )
    err_rows = [r for r in report["leagues"]
                if r.get("source") == "football-data.co.uk" and r.get("error")]
    assert err_rows, "expected fd.co.uk errors to be recorded"
    assert report["totals"]["errors"] >= 1


def test_inserted_matches_land_in_db(empty_db, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake-key")
    monkeypatch.delenv("FOOTBALL_API_KEY", raising=False)
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.setattr("data.result_backfill.client_from_env", lambda: None)
    monkeypatch.setattr(
        "data.result_backfill.fdcouk_fetch_season",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    today = date(2026, 5, 18)
    fake_frame = _fake_fdorg_frame(today - timedelta(days=2))

    with mock.patch(
        "scrape.football_data_org.FootballDataOrgClient.fetch_matches",
        return_value=fake_frame,
    ):
        backfill_recent_results(
            empty_db, days_back=7, cache_dir=tmp_path, as_of=today,
        )

    # The mocked frame was returned for every league with a fd.org code, so
    # there should be at least one row in the matches table.
    with empty_db.engine.begin() as conn:
        n = pd.read_sql("SELECT COUNT(*) AS n FROM matches", conn)["n"].iloc[0]
    assert n >= 1
