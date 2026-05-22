"""Tests for the per-team badge lookup + cache."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "tsdb-teams"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# fetch_team_badge — disk cache + TSDB API
# ---------------------------------------------------------------------------

def test_fetch_returns_badge_url_from_tsdb(cache_dir, monkeypatch):
    """A successful TSDB search yields a badge URL and caches it on disk."""
    from scrape import team_badges

    fake_payload = {
        "teams": [
            # ``strTeam`` is now checked against the query so we don't return
            # an unrelated club. The mock has to match what we asked for.
            {"strSport": "Soccer", "strTeam": "Arsenal",
             "strBadge": "https://r2.thesportsdb.com/x/arsenal.png"},
        ]
    }
    fake_response = mock.Mock()
    fake_response.json.return_value = fake_payload
    fake_response.raise_for_status.return_value = None
    monkeypatch.setattr(team_badges.httpx, "get", lambda *a, **kw: fake_response)
    # Skip the 500ms polite sleep so the test is instant
    monkeypatch.setattr(team_badges.time, "sleep", lambda _: None)

    url = team_badges.fetch_team_badge("Arsenal", cache_dir=cache_dir)
    assert url == "https://r2.thesportsdb.com/x/arsenal.png"

    # And cached to disk
    cache_files = list(cache_dir.iterdir())
    assert len(cache_files) == 1
    cached = json.loads(cache_files[0].read_text())
    assert cached["badge_url"] == "https://r2.thesportsdb.com/x/arsenal.png"
    assert cached["canonical"] == "Arsenal"


def test_fetch_returns_cached_value_without_http(cache_dir, monkeypatch):
    """Second call for the same team skips the network entirely."""
    from scrape import team_badges

    # Pre-seed cache
    (cache_dir / "arsenal.json").write_text(json.dumps({
        "canonical": "Arsenal",
        "badge_url": "https://example.com/cached.png",
        "fetched_at": 0,
    }))

    # Any HTTP call would explode — we shouldn't make one
    fake_get = mock.Mock(side_effect=AssertionError("should not call HTTP"))
    monkeypatch.setattr(team_badges.httpx, "get", fake_get)

    url = team_badges.fetch_team_badge("Arsenal", cache_dir=cache_dir)
    assert url == "https://example.com/cached.png"
    fake_get.assert_not_called()


def test_fetch_caches_negative_results(cache_dir, monkeypatch):
    """A team TSDB doesn't know → cache the miss so we don't retry forever."""
    from scrape import team_badges

    fake_response = mock.Mock()
    fake_response.json.return_value = {"teams": None}
    fake_response.raise_for_status.return_value = None
    monkeypatch.setattr(team_badges.httpx, "get", lambda *a, **kw: fake_response)
    monkeypatch.setattr(team_badges.time, "sleep", lambda _: None)

    url = team_badges.fetch_team_badge("Some Obscure Team", cache_dir=cache_dir)
    assert url is None

    # Cache file IS written, with badge_url=null
    files = list(cache_dir.iterdir())
    assert len(files) == 1
    body = json.loads(files[0].read_text())
    assert body["badge_url"] is None


def test_fetch_skips_non_soccer_results(cache_dir, monkeypatch):
    """TSDB's free search returns all sports. Pick the soccer entry, ignore
    Cricket/American Football/Basketball teams with the same name."""
    from scrape import team_badges

    # All three entries share the queried name (TSDB's behavior — the API
    # filters by name across all sports). Only the Soccer one should win.
    fake_payload = {
        "teams": [
            {"strSport": "American Football", "strTeam": "Arsenal",
             "strBadge": "https://example.com/nfl.png"},
            {"strSport": "Cricket", "strTeam": "Arsenal",
             "strBadge": "https://example.com/cricket.png"},
            {"strSport": "Soccer", "strTeam": "Arsenal",
             "strBadge": "https://example.com/soccer.png"},
        ]
    }
    fake_response = mock.Mock()
    fake_response.json.return_value = fake_payload
    fake_response.raise_for_status.return_value = None
    monkeypatch.setattr(team_badges.httpx, "get", lambda *a, **kw: fake_response)
    monkeypatch.setattr(team_badges.time, "sleep", lambda _: None)

    url = team_badges.fetch_team_badge("Arsenal", cache_dir=cache_dir)
    assert url == "https://example.com/soccer.png"


def test_fetch_handles_network_errors(cache_dir, monkeypatch):
    """Network/HTTP errors return ``None`` and DON'T crash."""
    from scrape import team_badges
    import httpx

    def boom(*a, **kw):
        raise httpx.ConnectError("network unreachable")

    monkeypatch.setattr(team_badges.httpx, "get", boom)
    monkeypatch.setattr(team_badges.time, "sleep", lambda _: None)
    url = team_badges.fetch_team_badge("Arsenal", cache_dir=cache_dir)
    assert url is None


def test_empty_team_name_returns_none(cache_dir):
    from scrape.team_badges import fetch_team_badge
    assert fetch_team_badge("", cache_dir=cache_dir) is None
    assert fetch_team_badge(None, cache_dir=cache_dir) is None


# ---------------------------------------------------------------------------
# attach_badges — DataFrame mutation
# ---------------------------------------------------------------------------

def test_attach_badges_fills_missing_urls(cache_dir, monkeypatch):
    from scrape import team_badges

    df = pd.DataFrame([{
        "home_team": "Arsenal", "away_team": "Chelsea",
        "home_badge_url": None, "away_badge_url": None,
    }])
    # Pre-seed disk cache so attach_badges doesn't need the network
    (cache_dir / "arsenal.json").write_text(json.dumps({"badge_url": "https://x/a.png"}))
    (cache_dir / "chelsea.json").write_text(json.dumps({"badge_url": "https://x/c.png"}))

    team_badges.attach_badges(df, cache_dir=cache_dir)

    assert df.iloc[0]["home_badge_url"] == "https://x/a.png"
    assert df.iloc[0]["away_badge_url"] == "https://x/c.png"


def test_attach_badges_preserves_already_valid_urls(cache_dir, monkeypatch):
    """If a URL already ends in .png/.jpg/.svg, don't re-fetch."""
    from scrape import team_badges

    df = pd.DataFrame([{
        "home_team": "Arsenal", "away_team": "Chelsea",
        "home_badge_url": "https://existing.com/arsenal.png",
        "away_badge_url": "https://existing.com/chelsea.svg",
    }])
    # Any HTTP call would fail — verify we don't make any
    fake_get = mock.Mock(side_effect=AssertionError("should not call HTTP"))
    monkeypatch.setattr(team_badges.httpx, "get", fake_get)

    team_badges.attach_badges(df, cache_dir=cache_dir)
    assert df.iloc[0]["home_badge_url"] == "https://existing.com/arsenal.png"
    assert df.iloc[0]["away_badge_url"] == "https://existing.com/chelsea.svg"


def test_attach_badges_rejects_broken_stub_urls(cache_dir, monkeypatch):
    """The broken stub URLs from eventsnextleague.php (no .png extension) get
    replaced with the proper looked-up URL."""
    from scrape import team_badges

    df = pd.DataFrame([{
        "home_team": "Arsenal", "away_team": "Chelsea",
        # These are the broken-stub form (no extension)
        "home_badge_url": "https://r2.thesportsdb.com/x/xrxtrq144",
        "away_badge_url": "https://r2.thesportsdb.com/x/0ynlvb177",
    }])
    (cache_dir / "arsenal.json").write_text(json.dumps({"badge_url": "https://good.png"}))
    (cache_dir / "chelsea.json").write_text(json.dumps({"badge_url": "https://good2.png"}))

    team_badges.attach_badges(df, cache_dir=cache_dir)
    # Broken stubs got replaced with valid .png URLs from the cache
    assert df.iloc[0]["home_badge_url"] == "https://good.png"
    assert df.iloc[0]["away_badge_url"] == "https://good2.png"


def test_attach_badges_handles_empty_dataframe(cache_dir):
    """No-op on empty input, no errors."""
    from scrape.team_badges import attach_badges
    df = pd.DataFrame()
    attach_badges(df, cache_dir=cache_dir)  # should not raise


# ---------------------------------------------------------------------------
# Name-validation: don't trust TSDB returning Arsenal for every query
# ---------------------------------------------------------------------------

def test_fetch_rejects_mismatched_team_name(cache_dir, monkeypatch):
    """The free-tier searchteams.php has been observed to return Arsenal for
    every query. When the returned ``strTeam`` clearly doesn't match what we
    asked for, treat it as 'no match' rather than serving the wrong crest."""
    from scrape import team_badges

    # Asked for Chelsea, TSDB returns Arsenal (the actual bug)
    fake_payload = {
        "teams": [
            {"strSport": "Soccer", "strTeam": "Arsenal",
             "strBadge": "https://example.com/arsenal.png"},
        ]
    }
    fake_response = mock.Mock()
    fake_response.json.return_value = fake_payload
    fake_response.raise_for_status.return_value = None
    monkeypatch.setattr(team_badges.httpx, "get", lambda *a, **kw: fake_response)
    monkeypatch.setattr(team_badges.time, "sleep", lambda _: None)

    url = team_badges.fetch_team_badge("Chelsea", cache_dir=cache_dir)
    assert url is None  # Arsenal's badge rejected — names don't match Chelsea


def test_fetch_accepts_alias_style_name_matches(cache_dir, monkeypatch):
    """'Real Madrid' asked, TSDB returns 'Real Madrid CF' — should accept."""
    from scrape import team_badges

    fake_payload = {
        "teams": [
            {"strSport": "Soccer", "strTeam": "Real Madrid CF",
             "strBadge": "https://example.com/real-madrid.png"},
        ]
    }
    fake_response = mock.Mock()
    fake_response.json.return_value = fake_payload
    fake_response.raise_for_status.return_value = None
    monkeypatch.setattr(team_badges.httpx, "get", lambda *a, **kw: fake_response)
    monkeypatch.setattr(team_badges.time, "sleep", lambda _: None)

    url = team_badges.fetch_team_badge("Real Madrid", cache_dir=cache_dir)
    assert url == "https://example.com/real-madrid.png"


# ---------------------------------------------------------------------------
# populate_cache_from_events — harvest from /upcoming event payloads
# ---------------------------------------------------------------------------

def test_populate_cache_writes_per_team_files(cache_dir):
    """When /upcoming runs, every team with a valid event-side badge URL
    gets a cache entry, keyed by canonical name. This is the ONLY reliable
    way to get correct badges since searchteams.php is broken."""
    from scrape import team_badges

    df = pd.DataFrame([{
        "home_team": "Arsenal", "away_team": "Chelsea",
        "home_badge_url": "https://r2/arsenal.png",
        "away_badge_url": "https://r2/chelsea.png",
    }, {
        "home_team": "Liverpool", "away_team": "Arsenal",  # dup home
        "home_badge_url": "https://r2/liverpool.png",
        "away_badge_url": "https://r2/arsenal.png",
    }])
    written = team_badges.populate_cache_from_events(df, cache_dir=cache_dir)
    # 3 unique teams (Arsenal de-duped), each written once
    assert written == 3
    files = {p.name for p in cache_dir.iterdir()}
    assert files == {"arsenal.json", "chelsea.json", "liverpool.json"}
    # And each cache has the right URL
    import json
    arsenal = json.loads((cache_dir / "arsenal.json").read_text())
    assert arsenal["badge_url"] == "https://r2/arsenal.png"
    assert arsenal["source"] == "tsdb_event"


def test_populate_cache_skips_invalid_urls(cache_dir):
    """Rows with empty / non-.png URLs don't populate the cache."""
    from scrape.team_badges import populate_cache_from_events
    df = pd.DataFrame([{
        "home_team": "Arsenal", "away_team": "Chelsea",
        "home_badge_url": None,
        "away_badge_url": "",
    }])
    written = populate_cache_from_events(df, cache_dir=cache_dir)
    assert written == 0


# ---------------------------------------------------------------------------
# warm_from_api_football — fill gaps fd.org doesn't cover
# ---------------------------------------------------------------------------

def test_warm_from_api_football_no_client_returns_empty(cache_dir, monkeypatch):
    """No API key configured → graceful exit, no crash."""
    from scrape import team_badges
    monkeypatch.setattr(
        "scrape.api_football.client_from_env",
        lambda: None,
    )
    report = team_badges.warm_from_api_football(cache_dir=cache_dir)
    assert report["total_cached"] == 0
    assert "no_api_football_key" in report["errors"]


def test_warm_from_api_football_caches_team_logos(cache_dir, monkeypatch):
    """Each team in api-football's /teams response → one cache entry."""
    from scrape import team_badges, api_football as af_mod

    # Mock client + quota state
    af_mod.reset_daily_quota_flag()
    fake_client = mock.Mock()
    fake_client.get = mock.Mock(return_value={"response": [
        {"team": {"id": 33, "name": "Los Angeles FC", "logo": "https://media.api-sports.io/lafc.png"}},
        {"team": {"id": 34, "name": "Vissel Kobe", "logo": "https://media.api-sports.io/vissel.png"}},
    ]})
    monkeypatch.setattr("scrape.api_football.client_from_env", lambda: fake_client)

    report = team_badges.warm_from_api_football(
        cache_dir=cache_dir, league_keys=["mls"],
    )
    assert report["total_cached"] == 2
    files = {p.stem for p in cache_dir.iterdir()}
    assert "lafc" in files  # "Los Angeles FC" canonicalizes to LAFC via aliases
    # The other one should be cached as the canonical "Vissel Kobe"
    assert "vissel-kobe" in files


def test_warm_from_api_football_skips_when_cache_exists(cache_dir, monkeypatch):
    """Default: don't overwrite an existing cache entry (fd.org wins ties)."""
    from scrape import team_badges, api_football as af_mod

    af_mod.reset_daily_quota_flag()
    # Pre-seed cache with an "fd.org-sourced" Arsenal
    (cache_dir / "arsenal.json").write_text(json.dumps({
        "canonical": "Arsenal", "badge_url": "https://crests.football-data.org/57.png",
        "source": "football-data.org",
    }))
    fake_client = mock.Mock()
    fake_client.get = mock.Mock(return_value={"response": [
        {"team": {"id": 42, "name": "Arsenal", "logo": "https://media.api-sports.io/arsenal-af.png"}},
    ]})
    monkeypatch.setattr("scrape.api_football.client_from_env", lambda: fake_client)

    report = team_badges.warm_from_api_football(
        cache_dir=cache_dir, league_keys=["premier_league"], skip_if_cached=True,
    )
    assert report["total_cached"] == 0  # didn't overwrite
    # Cache still has fd.org's URL
    body = json.loads((cache_dir / "arsenal.json").read_text())
    assert body["source"] == "football-data.org"


def test_warm_from_api_football_overwrites_when_skip_disabled(cache_dir, monkeypatch):
    """With ``skip_if_cached=False``, api-football wins — use sparingly."""
    from scrape import team_badges, api_football as af_mod

    af_mod.reset_daily_quota_flag()
    (cache_dir / "arsenal.json").write_text(json.dumps({
        "badge_url": "https://crests.football-data.org/57.png", "source": "football-data.org",
    }))
    fake_client = mock.Mock()
    fake_client.get = mock.Mock(return_value={"response": [
        {"team": {"id": 42, "name": "Arsenal", "logo": "https://media.api-sports.io/arsenal-af.png"}},
    ]})
    monkeypatch.setattr("scrape.api_football.client_from_env", lambda: fake_client)

    report = team_badges.warm_from_api_football(
        cache_dir=cache_dir, league_keys=["premier_league"], skip_if_cached=False,
    )
    assert report["total_cached"] == 1
    body = json.loads((cache_dir / "arsenal.json").read_text())
    assert body["source"] == "api-football"


def test_warm_from_api_football_bails_when_daily_quota_exhausted(cache_dir, monkeypatch):
    """Quota already burned by other ops → don't even try. Save a wasted call."""
    from scrape import team_badges, api_football as af_mod

    fake_client = mock.Mock()
    monkeypatch.setattr("scrape.api_football.client_from_env", lambda: fake_client)
    monkeypatch.setattr("scrape.api_football.is_daily_quota_exhausted", lambda: True)

    report = team_badges.warm_from_api_football(cache_dir=cache_dir, league_keys=["mls"])
    assert report["total_cached"] == 0
    assert "daily_quota_exhausted_before_start" in report["errors"]
    fake_client.get.assert_not_called()


def test_warm_from_api_football_stops_mid_run_on_quota(cache_dir, monkeypatch):
    """If quota gets exhausted while iterating leagues, stop cleanly."""
    from scrape import team_badges, api_football as af_mod

    af_mod.reset_daily_quota_flag()
    fake_client = mock.Mock()
    fake_client.get = mock.Mock(return_value={"response": [
        {"team": {"id": 1, "name": "Some Team", "logo": "https://x.com/team.png"}},
    ]})

    # Track call count to simulate quota exhaustion mid-run
    quota_calls = {"n": 0}
    def fake_quota_check():
        quota_calls["n"] += 1
        # First check (entry) returns False. Second league iteration True.
        return quota_calls["n"] >= 3

    monkeypatch.setattr("scrape.api_football.client_from_env", lambda: fake_client)
    monkeypatch.setattr("scrape.api_football.is_daily_quota_exhausted", fake_quota_check)

    report = team_badges.warm_from_api_football(
        cache_dir=cache_dir, league_keys=["mls", "j1", "saudi_pro"],
    )
    # At least one league got cached; later ones bailed
    assert report["total_cached"] >= 1
    assert any("daily_quota_exhausted_mid_run" in e for e in report["errors"])


# ---------------------------------------------------------------------------
# warm_from_fdorg — bulk cache warming from football-data.org
# ---------------------------------------------------------------------------

def test_warm_from_fdorg_returns_empty_without_api_key(cache_dir, monkeypatch):
    from scrape import team_badges
    for var in ("FOOTBALL_DATA_ORG_KEY", "FOOTBALL_DATA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    report = team_badges.warm_from_fdorg(cache_dir=cache_dir)
    assert report["total_cached"] == 0
    assert report["errors"] == ["no_api_key"]


def test_warm_from_fdorg_caches_each_team_with_crest(cache_dir, monkeypatch):
    """Each team in the fd.org response → one cache file with the crest URL."""
    from scrape import team_badges
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake-key")
    monkeypatch.setattr(team_badges.time, "sleep", lambda _: None)

    def fake_get(url, **kw):
        resp = mock.Mock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        # Return different teams per league code, mirroring the real shape
        if "/PL/teams" in url:
            resp.json.return_value = {"teams": [
                {"name": "Arsenal FC", "shortName": "Arsenal",
                 "crest": "https://crests.football-data.org/57.png"},
                {"name": "Chelsea FC", "shortName": "Chelsea",
                 "crest": "https://crests.football-data.org/61.png"},
            ]}
        else:
            resp.json.return_value = {"teams": []}
        return resp

    monkeypatch.setattr(team_badges.httpx, "get", fake_get)

    report = team_badges.warm_from_fdorg(
        cache_dir=cache_dir, league_keys=["premier_league"],
    )
    assert report["total_cached"] == 2
    assert report["by_league"]["premier_league"] == 2
    # Cache files exist for Arsenal and Chelsea
    assert (cache_dir / "arsenal.json").exists()
    assert (cache_dir / "chelsea.json").exists()
    body = json.loads((cache_dir / "arsenal.json").read_text())
    assert body["badge_url"] == "https://crests.football-data.org/57.png"
    assert body["source"] == "football-data.org"


def test_warm_from_fdorg_skips_teams_with_invalid_crests(cache_dir, monkeypatch):
    """Teams with empty / non-.png crest URLs are skipped, not cached."""
    from scrape import team_badges
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake-key")
    monkeypatch.setattr(team_badges.time, "sleep", lambda _: None)

    def fake_get(url, **kw):
        resp = mock.Mock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"teams": [
            {"name": "Good Team", "crest": "https://example.com/good.png"},
            {"name": "Empty Crest Team", "crest": ""},
            {"name": "No Crest Team"},
            {"name": "Bad Format Team", "crest": "https://example.com/team"},  # no extension
        ]}
        return resp

    monkeypatch.setattr(team_badges.httpx, "get", fake_get)
    report = team_badges.warm_from_fdorg(
        cache_dir=cache_dir, league_keys=["premier_league"],
    )
    assert report["total_cached"] == 1


def test_warm_from_fdorg_handles_403_forbidden(cache_dir, monkeypatch):
    """A 403 (key doesn't have access to that league) is logged as an error
    but doesn't kill the whole warm."""
    from scrape import team_badges
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake-key")
    monkeypatch.setattr(team_badges.time, "sleep", lambda _: None)

    def fake_get(url, **kw):
        resp = mock.Mock()
        resp.status_code = 403
        return resp

    monkeypatch.setattr(team_badges.httpx, "get", fake_get)
    report = team_badges.warm_from_fdorg(
        cache_dir=cache_dir, league_keys=["chinese_super_league"],
    )
    assert report["total_cached"] == 0
    assert any("forbidden" in e for e in report["errors"])


def test_warm_from_fdorg_dedupes_teams_across_leagues(cache_dir, monkeypatch):
    """Real Madrid in La Liga AND Champions League → only one cache file
    (idempotent, deterministic content)."""
    from scrape import team_badges
    monkeypatch.setenv("FOOTBALL_DATA_ORG_KEY", "fake-key")
    monkeypatch.setattr(team_badges.time, "sleep", lambda _: None)

    same_team = [{"name": "Real Madrid CF", "crest": "https://crests.football-data.org/86.png"}]

    def fake_get(url, **kw):
        resp = mock.Mock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"teams": same_team}
        return resp

    monkeypatch.setattr(team_badges.httpx, "get", fake_get)
    report = team_badges.warm_from_fdorg(
        cache_dir=cache_dir,
        league_keys=["la_liga", "champions_league"],
    )
    # Real Madrid counted ONCE despite appearing in two competitions
    assert report["total_cached"] == 1
    # Both leagues processed (no errors)
    assert report["by_league"] == {"la_liga": 1, "champions_league": 0}


def test_populate_cache_canonicalizes_keys(cache_dir):
    """'Brentford FC' → 'Brentford' (canonical) — cache key matches what
    /predict will later look up by canonical name."""
    from scrape.team_badges import populate_cache_from_events
    df = pd.DataFrame([{
        "home_team": "Brentford FC", "away_team": "Crystal Palace FC",
        "home_badge_url": "https://r2/brentford.png",
        "away_badge_url": "https://r2/crystal-palace.png",
    }])
    populate_cache_from_events(df, cache_dir=cache_dir)
    files = {p.name for p in cache_dir.iterdir()}
    # Filename safe-translation: lowercase + non-alnum → "-"
    assert "brentford.json" in files
    assert "crystal-palace.json" in files
