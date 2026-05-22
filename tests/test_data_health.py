"""Tests for the data-health collector — uses temp DB, no real data needed."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from data.data_health import (
    _bucket,
    _format_age,
    _human_bytes,
    _mask_secret,
    api_key_health,
    build_health_report,
    cache_health,
    per_league_health,
    per_source_health,
)
from data.database import Database


@pytest.fixture
def populated_db(tmp_path: Path) -> Database:
    """Seed a tiny DB with two sources × two leagues so health logic has data."""
    db = Database(tmp_path / "test.sqlite3")
    db.init()
    today = date.today()
    rows_a = pd.DataFrame([
        {"date": today - timedelta(days=2), "home_team": "A", "away_team": "B",
         "home_goals": 1, "away_goals": 0},
        {"date": today - timedelta(days=5), "home_team": "B", "away_team": "C",
         "home_goals": 2, "away_goals": 2},
    ])
    db.upsert_matches(rows_a, source="src_one", league_key="league_alpha", league_name=None)
    rows_b = pd.DataFrame([
        {"date": today - timedelta(days=200), "home_team": "X", "away_team": "Y",
         "home_goals": 0, "away_goals": 1},
    ])
    db.upsert_matches(rows_b, source="src_two", league_key="league_beta", league_name=None)
    return db


def test_per_league_health_lists_each_league_once(populated_db: Database) -> None:
    rows = per_league_health(populated_db)
    keys = {r["league_key"] for r in rows}
    assert keys == {"league_alpha", "league_beta"}


def test_per_league_health_flags_fresh_vs_stale(populated_db: Database) -> None:
    rows = {r["league_key"]: r for r in per_league_health(populated_db)}
    assert rows["league_alpha"]["freshness"] == "fresh"
    # 200-day old data → "very_stale" bucket
    assert rows["league_beta"]["freshness"] == "very_stale"


def test_per_league_health_match_counts(populated_db: Database) -> None:
    rows = {r["league_key"]: r for r in per_league_health(populated_db)}
    assert rows["league_alpha"]["match_count"] == 2
    assert rows["league_beta"]["match_count"] == 1


def test_per_source_health_groups_correctly(populated_db: Database) -> None:
    rows = per_source_health(populated_db)
    by_src = {r["source"]: r for r in rows}
    assert set(by_src) == {"src_one", "src_two"}
    assert by_src["src_one"]["match_count"] == 2
    assert "league_alpha" in by_src["src_one"]["leagues"]


def test_bucket_thresholds() -> None:
    assert _bucket(0) == "fresh"
    assert _bucket(7) == "fresh"
    assert _bucket(8) == "recent"
    assert _bucket(30) == "recent"
    assert _bucket(31) == "stale"
    assert _bucket(180) == "stale"
    assert _bucket(181) == "very_stale"


def test_format_age_units() -> None:
    assert _format_age(30) == "30s"
    assert _format_age(120) == "2m"
    assert _format_age(3600) == "1.0h"
    assert _format_age(86400 * 2) == "2.0d"


def test_human_bytes() -> None:
    assert _human_bytes(0) == "0B"
    assert _human_bytes(512) == "512B"
    assert _human_bytes(2048) == "2.0KB"
    assert _human_bytes(int(5e6)) == "4.8MB"


def test_mask_secret() -> None:
    assert _mask_secret(None) is None
    assert _mask_secret("short") == "***"
    assert _mask_secret("a-32-char-secret-abcdefghij1234") == "a-32…1234"


def test_api_key_health_reports_missing_when_unset(monkeypatch) -> None:
    for var in ("FOOTBALL_DATA_ORG_KEY", "API_FOOTBALL_KEY", "FOOTBALL_API_KEY",
                "TSDB_API_KEY", "THESPORTSDB_API_KEY", "THESPORTSDB_KEY"):
        monkeypatch.delenv(var, raising=False)
    rows = api_key_health()
    assert all(not r["configured"] for r in rows)


def test_api_key_health_picks_up_set_value(monkeypatch) -> None:
    monkeypatch.setenv("API_FOOTBALL_KEY", "x" * 40)
    rows = {r["env_var"]: r for r in api_key_health()}
    af = rows["API_FOOTBALL_KEY"]
    assert af["configured"] is True
    assert af["masked"].startswith("xxxx")
    assert af["found_via"] == "API_FOOTBALL_KEY"


def test_cache_health_returns_one_row_per_known_dir() -> None:
    rows = cache_health()
    labels = {r["label"] for r in rows}
    assert "football-data.co.uk CSVs" in labels
    assert "API-Football" in labels


def test_build_health_report_assembles_all_sections(populated_db: Database) -> None:
    report = build_health_report(populated_db)
    assert "totals" in report
    assert "per_league" in report
    assert "per_source" in report
    assert "api_keys" in report
    assert "caches" in report
    assert report["totals"]["total_matches"] == 3
    assert report["totals"]["league_count"] == 2
    assert report["totals"]["source_count"] == 2
