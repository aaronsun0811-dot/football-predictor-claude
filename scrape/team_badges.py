"""Look up team-crest URLs from TheSportsDB and cache them forever.

The badge URLs that come back on /eventsnextleague.php events are unreliable
stubs (often missing the .png extension or pointing at expired CDN paths).
The proper source is the per-team ``searchteams.php?t=<name>`` endpoint —
that one returns the canonical CDN URL like
``https://r2.thesportsdb.com/images/media/team/badge/uyhbfe1612467038.png``.

Strategy:
  * One lookup per unique team name, then cached on disk **forever** (badges
    change once a decade at most — definitely not faster than the schedule).
  * Cache key = canonicalized team name (so "Real Madrid CF" and "Real Madrid"
    share the same crest after R23's canonicalize cleanup).
  * Cache miss → one HTTP call, ~200ms.
  * Failures (404, empty response, network error) cached as "no badge found"
    so we don't re-hammer the API for teams that don't exist in TSDB.

Cost on first warm-up: ~22 unique teams across ~10 fixtures × 1 request =
~50s at TSDB's free 30/min rate. After that: instant.
"""
from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Any

import httpx

from data.team_normalize import canonicalize


TSDB_SEARCH_URL = "https://www.thesportsdb.com/api/v1/json/3/searchteams.php"
TIMEOUT_S = 10.0
_LAST_REQUEST_AT = 0.0
_MIN_INTERVAL_S = 0.5  # 30 req/min → 2s gap is plenty safe; 0.5s gives headroom


def fetch_team_badge(
    team_name: str,
    *,
    cache_dir: Path,
) -> str | None:
    """Return the canonical badge URL for a team, or ``None`` if TSDB has no record.

    Disk-cached forever (per canonical name). Network failures return ``None``
    AND cache the negative so we don't retry on every page load.

    Args:
        team_name: any spelling — will be canonicalized for cache lookup.
        cache_dir: where to persist the JSON-per-team cache files.
    """
    if not team_name:
        return None
    canon = canonicalize(team_name) or team_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_safe_filename(canon)}.json"

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            return cached.get("badge_url")
        except (OSError, json.JSONDecodeError):
            pass  # corrupted cache → re-fetch

    badge_url = _query_tsdb(canon)
    try:
        cache_path.write_text(json.dumps({
            "canonical": canon,
            "badge_url": badge_url,  # explicitly null on miss → cached as miss
            "fetched_at": time.time(),
        }))
    except OSError:
        pass  # disk full or read-only → still return what we got
    return badge_url


def _query_tsdb(canonical_name: str) -> str | None:
    """One TSDB request. Returns the badge URL string, or ``None`` on any failure.

    Polite throttle: at most one request every ``_MIN_INTERVAL_S`` seconds
    across all callers — TSDB free tier nominally allows 30/min but
    aggressive batching gets you 429'd in practice.
    """
    global _LAST_REQUEST_AT
    elapsed = time.monotonic() - _LAST_REQUEST_AT
    if elapsed < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - elapsed)
    _LAST_REQUEST_AT = time.monotonic()

    try:
        response = httpx.get(
            TSDB_SEARCH_URL,
            params={"t": canonical_name},
            timeout=TIMEOUT_S,
            headers={"User-Agent": "football-predictor/0.1"},
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None

    teams = payload.get("teams") or []
    if not teams:
        return None

    # TSDB returns all matches — pick the first soccer team. We also defend
    # against a known free-tier breakage: searchteams.php has been observed
    # to return Arsenal for EVERY query (any team name routes to the same
    # cached response). So validate that the returned ``strTeam`` actually
    # resembles what we asked for. If not, treat it as "no match" — let the
    # cache fall back to None and the UI to its monogram, rather than
    # serving an unrelated club's crest.
    needle = canonical_name.lower().strip()
    for team in teams:
        sport = (team.get("strSport") or "").lower()
        if sport != "soccer":
            continue
        matched_name = (team.get("strTeam") or "").lower().strip()
        if not _names_plausibly_match(needle, matched_name):
            continue
        badge = team.get("strBadge")
        if badge and badge.strip():
            return badge.strip()
    return None


def _names_plausibly_match(asked: str, returned: str) -> bool:
    """Cheap fuzzy check that TSDB's returned team is the team we asked for.

    Catches the free-tier bug where every query returns Arsenal. We accept:
      * exact match
      * one is a prefix/suffix of the other (e.g. "Real Madrid" ↔ "Real Madrid CF")
      * shared word count ≥ 1 (handles "FC Bayern München" ↔ "Bayern Munich")

    Errs on the side of accepting — overly strict means we lose real matches
    for teams TSDB knows by a different long name.
    """
    if not asked or not returned:
        return False
    if asked == returned:
        return True
    if asked in returned or returned in asked:
        return True
    asked_words = set(asked.split())
    returned_words = set(returned.split())
    # Strip 1-2 letter filler words (FC, AC, CF, de, la, ...)
    asked_words = {w for w in asked_words if len(w) >= 3}
    returned_words = {w for w in returned_words if len(w) >= 3}
    return bool(asked_words & returned_words)


def warm_from_fdorg(
    *,
    cache_dir: Path,
    api_key: str | None = None,
    league_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Bulk-populate the team-badge cache from football-data.org's
    ``/competitions/{code}/teams`` endpoint.

    The TSDB free tier's ``searchteams.php`` AND ``lookup_all_teams.php`` are
    both degraded (return Arsenal / League One teams for any query). fd.org's
    per-competition teams endpoint, by contrast, returns the full team list
    for each league with a working PNG crest per team. One API call per league.

    Coverage: every league in ``scrape.football_data_org.LEAGUE_KEY_TO_CODE``
    that the user's free fd.org key has access to (top European + Brazilian +
    CSL + UCL/Europa/Libertadores + EC + WC). MLS / J1 / K1 / Argentine /
    Liga MX are NOT covered — those still rely on /upcoming-event harvesting.

    Returns a report ``{leagues, total_cached, by_league, errors}``. Failures
    per league are recorded but never raise.
    """
    import os  # noqa: PLC0415
    from scrape.football_data_org import (  # noqa: PLC0415
        LEAGUE_KEY_TO_CODE as FDORG_CODES,
        BASE_URL as FDORG_BASE_URL,
    )

    api_key = api_key or os.environ.get("FOOTBALL_DATA_ORG_KEY") or os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        return {
            "leagues": 0, "total_cached": 0, "by_league": {}, "errors": ["no_api_key"],
        }

    targets = league_keys or list(FDORG_CODES.keys())
    cache_dir.mkdir(parents=True, exist_ok=True)
    by_league: dict[str, int] = {}
    errors: list[str] = []
    total_cached = 0
    seen_canon: set[str] = set()

    for league_key in targets:
        code = FDORG_CODES.get(league_key)
        if not code:
            continue

        # Polite throttle (fd.org free tier = 10/min, this gives 0.5s gaps)
        elapsed = time.monotonic() - _LAST_REQUEST_AT
        if elapsed < _MIN_INTERVAL_S:
            time.sleep(_MIN_INTERVAL_S - elapsed)
        _set_last_request_at()

        try:
            resp = httpx.get(
                f"{FDORG_BASE_URL}/competitions/{code}/teams",
                headers={"X-Auth-Token": api_key, "User-Agent": "football-predictor/0.1"},
                timeout=TIMEOUT_S,
            )
            if resp.status_code == 429:
                errors.append(f"{league_key}: rate_limit")
                # Pause longer and continue with next league
                time.sleep(60)
                continue
            if resp.status_code == 403:
                errors.append(f"{league_key}: forbidden_for_key")
                continue
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            errors.append(f"{league_key}: {type(exc).__name__}: {str(exc)[:60]}")
            continue

        teams = payload.get("teams") or []
        league_count = 0
        for team in teams:
            name = team.get("name") or team.get("shortName")
            crest = team.get("crest")
            if not name or not crest:
                continue
            # Validate crest looks like a real URL (fd.org sometimes returns "")
            if not crest.lower().rstrip("/").endswith((".png", ".jpg", ".jpeg", ".svg", ".webp")):
                continue

            canon = canonicalize(name) or name
            if canon in seen_canon:
                # A team in multiple competitions (Real Madrid in La Liga AND
                # Champions League) — first write wins, that's fine since
                # fd.org returns the same crest URL for the same club.
                continue
            seen_canon.add(canon)

            cache_path = cache_dir / f"{_safe_filename(canon)}.json"
            try:
                cache_path.write_text(json.dumps({
                    "canonical": canon,
                    "badge_url": crest,
                    "fetched_at": time.time(),
                    "source": "football-data.org",
                    "league_seen_in": league_key,
                    "fdorg_name": name,
                    "fdorg_short_name": team.get("shortName"),
                }))
                league_count += 1
                total_cached += 1
            except OSError as exc:
                errors.append(f"{league_key}/{canon}: {exc}")
        by_league[league_key] = league_count

    return {
        "leagues": len(by_league),
        "total_cached": total_cached,
        "by_league": by_league,
        "errors": errors,
    }


def _set_last_request_at() -> None:
    """Bump the module-level throttle marker — shared with ``_query_tsdb``."""
    global _LAST_REQUEST_AT
    _LAST_REQUEST_AT = time.monotonic()


def warm_from_api_football(
    *,
    cache_dir: Path,
    league_keys: list[str] | None = None,
    skip_if_cached: bool = True,
) -> dict[str, Any]:
    """Bulk-populate the team-badge cache from API-Football.

    Complements ``warm_from_fdorg`` — covers leagues fd.org's free tier
    doesn't include (MLS, J1, K1, Saudi Pro League, Liga MX, Argentine
    Primera, plus continental cups like Copa Sudamericana / AFC Champions).
    One request per league via ``/teams?league=<id>&season=<season>``.

    ``skip_if_cached=True`` (default) preserves fd.org's higher-quality crests
    when a team has already been warmed there. Pass ``False`` to overwrite —
    useful only if you know the fd.org URL went stale.

    Cost: ~10 leagues × 6s (api-football is bucket-paced 10/min on free) =
    ~60s. Plus the daily-quota circuit breaker bails early if API-Football
    has been used up by other operations.
    """
    from scrape.api_football import (  # noqa: PLC0415
        ApiFootballClient,
        client_from_env,
        is_daily_quota_exhausted,
    )
    from scrape.registry import LeagueRegistry  # noqa: PLC0415
    from data.result_backfill import _season_for  # noqa: PLC0415

    af_client: ApiFootballClient | None = client_from_env()
    if af_client is None:
        return {
            "leagues": 0, "total_cached": 0, "by_league": {},
            "errors": ["no_api_football_key"],
        }
    if is_daily_quota_exhausted():
        return {
            "leagues": 0, "total_cached": 0, "by_league": {},
            "errors": ["daily_quota_exhausted_before_start"],
        }

    registry = LeagueRegistry()
    if league_keys:
        leagues = [registry.leagues[k] for k in league_keys if k in registry.leagues]
    else:
        leagues = [lg for lg in registry.all() if lg.api_football_id is not None]

    cache_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today()

    by_league: dict[str, int] = {}
    errors: list[str] = []
    total_cached = 0

    for league in leagues:
        if is_daily_quota_exhausted():
            errors.append(f"{league.key}: daily_quota_exhausted_mid_run")
            break  # no point continuing — every call would short-circuit

        season = _season_for(league, today)
        try:
            payload = af_client.get(
                "/teams",
                params={"league": int(league.api_football_id), "season": int(season)},
            )
        except Exception as exc:  # noqa: BLE001 — soft fail per league
            errors.append(f"{league.key}: {type(exc).__name__}: {str(exc)[:80]}")
            continue

        teams = payload.get("response") or []
        league_count = 0
        for entry in teams:
            t = entry.get("team") or {}
            name = t.get("name")
            logo = t.get("logo")
            if not name or not logo:
                continue
            if not logo.lower().rstrip("/").endswith((".png", ".jpg", ".jpeg", ".svg", ".webp")):
                continue

            canon = canonicalize(name) or name
            cache_path = cache_dir / f"{_safe_filename(canon)}.json"

            if skip_if_cached and cache_path.exists():
                # Don't overwrite — fd.org's crest is usually nicer-looking
                # and we want stable URLs (fd.org's `crests.football-data.org`
                # ids are stable; api-football's `media.api-sports.io` may
                # rotate paths).
                continue

            try:
                cache_path.write_text(json.dumps({
                    "canonical": canon,
                    "badge_url": logo,
                    "fetched_at": time.time(),
                    "source": "api-football",
                    "league_seen_in": league.key,
                    "api_football_team_id": t.get("id"),
                    "api_football_name": name,
                }))
                league_count += 1
                total_cached += 1
            except OSError as exc:
                errors.append(f"{league.key}/{canon}: {exc}")
        by_league[league.key] = league_count

    return {
        "leagues": len(by_league),
        "total_cached": total_cached,
        "by_league": by_league,
        "errors": errors,
    }


def populate_cache_from_events(
    fixtures_df,
    *,
    cache_dir: Path,
) -> int:
    """Write per-team badge cache entries from TSDB event payloads.

    TheSportsDB's ``eventsnextleague.php`` endpoint returns proper per-team
    badge URLs directly on each event (``strHomeTeamBadge`` / ``strAwayTeamBadge``).
    Those URLs are accurate per team — unlike the broken ``searchteams.php``
    fallback which returns the same Arsenal-shaped response for every query.

    We harvest those event-side URLs and seed the per-team cache so the later
    /predict path (which only has team names) finds the right crest without
    needing a buggy search lookup.

    Returns the number of cache entries written. Idempotent: re-running with
    the same events overwrites the cache files with the same content.
    """
    if fixtures_df.empty:
        return 0
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _ok(url):
        if not isinstance(url, str) or not url:
            return False
        u = url.lower().rstrip("/")
        return u.endswith((".png", ".jpg", ".jpeg", ".svg", ".webp"))

    written = 0
    seen_canonicals: set[str] = set()
    for row in fixtures_df.itertuples(index=False):
        for team, badge in (
            (getattr(row, "home_team", None), getattr(row, "home_badge_url", None)),
            (getattr(row, "away_team", None), getattr(row, "away_badge_url", None)),
        ):
            if not team or not _ok(badge):
                continue
            canon = canonicalize(team) or team
            if canon in seen_canonicals:
                continue
            seen_canonicals.add(canon)
            cache_path = cache_dir / f"{_safe_filename(canon)}.json"
            try:
                cache_path.write_text(json.dumps({
                    "canonical": canon,
                    "badge_url": badge,
                    "fetched_at": time.time(),
                    "source": "tsdb_event",  # distinguish from searchteams.php hits
                }))
                written += 1
            except OSError:
                pass
    return written


def attach_badges(
    fixtures_df,
    *,
    cache_dir: Path,
) -> None:
    """Walk a fixtures DataFrame, look up + assign ``home_badge_url`` /
    ``away_badge_url`` from the badge cache.

    Mutates the DataFrame in place. Skips teams whose badge field is already
    a working URL (i.e., not from the broken event-stub path — heuristic: must
    end in ``.png`` / ``.jpg`` / ``.svg``).
    """
    if fixtures_df.empty:
        return
    import pandas as _pd

    def _looks_valid(url):
        if not isinstance(url, str) or not url:
            return False
        lower = url.lower().rstrip("/")
        return lower.endswith((".png", ".jpg", ".jpeg", ".svg", ".webp"))

    teams_needing_lookup = set()
    for row in fixtures_df.itertuples(index=False):
        for team, current in (
            (getattr(row, "home_team", None), getattr(row, "home_badge_url", None)),
            (getattr(row, "away_team", None), getattr(row, "away_badge_url", None)),
        ):
            if team and not _looks_valid(current):
                teams_needing_lookup.add(team)

    badge_for: dict[str, str | None] = {}
    for team in teams_needing_lookup:
        badge_for[team] = fetch_team_badge(team, cache_dir=cache_dir)

    def _resolve(team_name, current):
        if _looks_valid(current):
            return current
        if team_name and team_name in badge_for:
            return badge_for[team_name]
        return None

    fixtures_df["home_badge_url"] = [
        _resolve(row.home_team, getattr(row, "home_badge_url", None))
        for row in fixtures_df.itertuples(index=False)
    ]
    fixtures_df["away_badge_url"] = [
        _resolve(row.away_team, getattr(row, "away_badge_url", None))
        for row in fixtures_df.itertuples(index=False)
    ]


def _safe_filename(name: str) -> str:
    """Make a team name filesystem-safe (lowercase, alphanumeric + dashes only)."""
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")
