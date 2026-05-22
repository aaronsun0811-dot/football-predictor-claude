"""Tests for the team-badge URL extraction added to the /upcoming scraper.

The scraper passes through TheSportsDB's per-event badge fields so the matchup
cards on the homepage can render real club crests instead of monogram circles.
Empty-string badges (which TSDB returns instead of null for unset fields) get
coerced to None so the UI fallback path triggers cleanly.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from scrape.upcoming_fixtures import _clean_url, fetch_upcoming


# ---------------------------------------------------------------------------
# _clean_url helper
# ---------------------------------------------------------------------------

def test_clean_url_returns_none_for_empty_string():
    """TheSportsDB sends ``""`` for unset badge URLs — we want ``None`` so the
    UI's ``v-if`` can branch to the monogram fallback."""
    assert _clean_url("") is None
    assert _clean_url("   ") is None
    assert _clean_url(None) is None


def test_clean_url_strips_whitespace_around_real_urls():
    assert _clean_url(" https://example.com/badge.png ") == "https://example.com/badge.png"
    assert _clean_url("https://example.com/badge.png") == "https://example.com/badge.png"


# ---------------------------------------------------------------------------
# fetch_upcoming carries badge URLs through to the DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture
def tsdb_cache(tmp_path: Path) -> Path:
    """Build a fresh on-disk TSDB cache so fetch_upcoming reads from it rather
    than hitting the network. The league_key 'premier_league' maps to TSDB id
    4328 in LEAGUE_KEY_TO_TSDB_ID — keep that in sync if the registry changes."""
    cache_dir = tmp_path / "tsdb"
    cache_dir.mkdir()
    payload = {
        "events": [
            # Fixture with full badges
            {
                "dateEvent": (date.today() + timedelta(days=2)).isoformat(),
                "strHomeTeam": "Arsenal",
                "strAwayTeam": "Chelsea",
                "idEvent": "12345",
                "strVenue": "Emirates",
                "strTime": "15:00:00",
                "strStatus": "NS",
                "strHomeTeamBadge": "https://example.com/arsenal.png",
                "strAwayTeamBadge": "https://example.com/chelsea.png",
            },
            # Fixture with empty-string badges → should coerce to None
            {
                "dateEvent": (date.today() + timedelta(days=3)).isoformat(),
                "strHomeTeam": "Liverpool",
                "strAwayTeam": "Tottenham",
                "idEvent": "12346",
                "strHomeTeamBadge": "",
                "strAwayTeamBadge": "   ",
            },
            # Fixture with missing badge fields entirely → also None
            {
                "dateEvent": (date.today() + timedelta(days=4)).isoformat(),
                "strHomeTeam": "Man City",
                "strAwayTeam": "Newcastle",
                "idEvent": "12347",
            },
        ]
    }
    cache_path = cache_dir / "tsdb_upcoming_4328_free.json"
    cache_path.write_text(json.dumps(payload))
    return cache_dir


def test_fetch_upcoming_passes_through_badge_urls(tsdb_cache, monkeypatch):
    # Force the v1 (free) endpoint code path so the cache filename matches
    monkeypatch.delenv("TSDB_API_KEY", raising=False)
    monkeypatch.delenv("THESPORTSDB_API_KEY", raising=False)
    monkeypatch.delenv("THESPORTSDB_KEY", raising=False)

    df = fetch_upcoming("premier_league", cache_dir=tsdb_cache, days_ahead=7)

    assert not df.empty
    by_home = {r["home_team"]: r for r in df.to_dict(orient="records")}

    # Real URLs come through unchanged
    assert by_home["Arsenal"]["home_badge_url"] == "https://example.com/arsenal.png"
    assert by_home["Arsenal"]["away_badge_url"] == "https://example.com/chelsea.png"

    # Empty/whitespace badges normalized to a falsy value (None or NaN —
    # pandas coerces None to NaN inside a mixed-string column, but both
    # serialize to JSON null and both fail the UI's truthy check, which is
    # what matters end-to-end).
    def _is_missing(v):
        return v is None or (isinstance(v, float) and pd.isna(v))

    assert _is_missing(by_home["Liverpool"]["home_badge_url"])
    assert _is_missing(by_home["Liverpool"]["away_badge_url"])
    assert _is_missing(by_home["Man City"]["home_badge_url"])
    assert _is_missing(by_home["Man City"]["away_badge_url"])


def test_fetch_upcoming_columns_include_badge_fields(tsdb_cache, monkeypatch):
    """Regression: callers (predict.py::_compute_upcoming_payload) use
    ``row.get("home_badge_url")``. The column must exist on the DataFrame even
    when no event has a badge — otherwise downstream ``row.get`` returns the
    pandas default which is fine, but column existence still matters for any
    .copy() / dtype paths."""
    monkeypatch.delenv("TSDB_API_KEY", raising=False)
    monkeypatch.delenv("THESPORTSDB_API_KEY", raising=False)
    df = fetch_upcoming("premier_league", cache_dir=tsdb_cache, days_ahead=7)
    assert "home_badge_url" in df.columns
    assert "away_badge_url" in df.columns
