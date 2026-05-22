"""Pull match fixtures from TheSportsDB.

Two tiers:

  * **Free**  — `/api/v1/json/3/<endpoint>`. Heavily throttled: `eventsnextleague`
    returns just 1-2 fixtures per league, `eventsseason` returns 15 events max.
  * **Patreon $2/month** — set ``TSDB_API_KEY`` in env. Switches to
    `/api/v2/json/<KEY>/<endpoint>`: full seasons, full upcoming schedule,
    higher rate limit (~60/min).

Same code path either way — we just swap the base URL based on whether
the key is set.

Maps our internal ``league_key`` → TheSportsDB ``idLeague``. Verified manually
2026-05.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import httpx
import pandas as pd


FREE_BASE = "https://www.thesportsdb.com/api/v1/json/3"
PATREON_BASE_TMPL = "https://www.thesportsdb.com/api/v2/json/{key}"
TIMEOUT_S = 15.0
CACHE_TTL_HOURS = 6.0


def _tsdb_base() -> tuple[str, bool]:
    """Return (base_url, is_patreon). Use Patreon if TSDB_API_KEY is set."""
    key = (
        os.environ.get("TSDB_API_KEY")
        or os.environ.get("THESPORTSDB_API_KEY")
        or os.environ.get("THESPORTSDB_KEY")
    )
    if key:
        return PATREON_BASE_TMPL.format(key=key.strip()), True
    return FREE_BASE, False


def is_patreon_key_set() -> bool:
    """Public helper for the UI / CLI to advertise the upgrade."""
    return _tsdb_base()[1]


# Backward-compat alias — the rest of the module used to reference BASE directly.
BASE = FREE_BASE


# TheSportsDB league IDs for leagues we have models trained for.
LEAGUE_KEY_TO_TSDB_ID: dict[str, int] = {
    # Asian top-flight
    "chinese_super_league": 4359,
    "j1": 4633,
    # k1 (Korean K League 1) — not found in their free catalog
    # American
    "mls": 4346,
    # European top-5 (already have other sources for upcoming)
    "premier_league": 4328,
    "championship": 4329,
    "la_liga": 4335,
    "serie_a": 4332,
    "bundesliga": 4331,
    "ligue_1": 4334,
    "eredivisie": 4337,
    "belgian_pro": 4338,
}


def _read_json_lenient(text: str) -> dict[str, Any]:
    """TheSportsDB occasionally returns control chars; lenient parse."""
    return json.loads(text, strict=False)


def fetch_upcoming(
    league_key: str,
    *,
    cache_dir: Path,
    days_ahead: int = 14,
) -> pd.DataFrame:
    """Return upcoming fixtures for a league as a DataFrame.

    Columns: ``date``, ``home_team``, ``away_team``, ``league_key``,
    ``tsdb_event_id``, ``venue``, ``status``.
    """
    tsdb_id = LEAGUE_KEY_TO_TSDB_ID.get(league_key)
    if tsdb_id is None:
        return _empty()

    base, is_patreon = _tsdb_base()
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "patreon" if is_patreon else "free"
    cache_path = cache_dir / f"tsdb_upcoming_{tsdb_id}_{suffix}.json"
    if _cache_fresh(cache_path):
        payload = _read_json_lenient(cache_path.read_text())
    else:
        # v2 endpoint is /schedule/next/league/<id> (richer); v1 is the legacy php
        url = (
            f"{base}/schedule/next/league/{tsdb_id}"
            if is_patreon
            else f"{base}/eventsnextleague.php?id={tsdb_id}"
        )
        try:
            response = httpx.get(url, timeout=TIMEOUT_S)
            response.raise_for_status()
            payload = _read_json_lenient(response.text)
            # v2 wraps the list under "schedule"; v1 under "events". Normalize.
            if "schedule" in payload and "events" not in payload:
                payload["events"] = payload["schedule"]
            cache_path.write_text(json.dumps(payload))
        except (httpx.HTTPError, ValueError) as exc:
            print(f"[tsdb] {league_key}: {exc}")
            return _empty()

    events = payload.get("events") or []
    today = date.today()
    horizon = today + pd.Timedelta(days=days_ahead).to_pytimedelta()
    rows = []
    for e in events:
        d = _parse_date(e.get("dateEvent"))
        if d is None:
            continue
        if d < today or d > horizon:
            continue
        rows.append({
            "date": d,
            "home_team": e.get("strHomeTeam"),
            "away_team": e.get("strAwayTeam"),
            "league_key": league_key,
            "tsdb_event_id": e.get("idEvent"),
            "venue": e.get("strVenue"),
            "time_utc": e.get("strTime"),
            "status": e.get("strStatus") or "NS",
            # TheSportsDB returns badge URLs on each event. Empty strings are
            # common — coerce to None so the UI can ``v-if`` cleanly. Two
            # possible field names: ``strHomeTeamBadge`` (v2 schedule API) and
            # ``strThumbHome`` / fallbacks (older v1 payloads). We try the
            # canonical one first.
            "home_badge_url": _clean_url(e.get("strHomeTeamBadge") or e.get("strThumbHome")),
            "away_badge_url": _clean_url(e.get("strAwayTeamBadge") or e.get("strThumbAway")),
        })
    if not rows:
        return _empty()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _clean_url(value: Any) -> str | None:
    """Treat empty/whitespace strings as missing — TheSportsDB returns ``""``
    instead of null for unset image fields, which would otherwise render as
    broken images in the UI."""
    if not value:
        return None
    s = str(value).strip()
    return s if s else None


def fetch_upcoming_multi(
    league_keys: Iterable[str],
    *,
    cache_dir: Path,
    days_ahead: int = 14,
) -> pd.DataFrame:
    """Concatenate upcoming fixtures across many leagues."""
    frames = [fetch_upcoming(k, cache_dir=cache_dir, days_ahead=days_ahead)
              for k in league_keys]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def _cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600
    return age_h < CACHE_TTL_HOURS


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def fetch_full_season(
    league_key: str,
    season: str | int,
    *,
    cache_dir: Path,
    require_patreon: bool = True,
) -> pd.DataFrame:
    """Pull a whole league-season's fixtures (finished + upcoming).

    Free tier caps responses to ~15 events per season — useless for backfill.
    Patreon ($2/mo) returns the full ~240-380 events.

    Set ``require_patreon=False`` if you want to use the free tier anyway
    (returns the slice; for sampling/dev only).

    Returns columns: date, home_team, away_team, home_goals, away_goals,
    league_key, season, status, venue.
    """
    tsdb_id = LEAGUE_KEY_TO_TSDB_ID.get(league_key)
    if tsdb_id is None:
        return _empty_full()

    base, is_patreon = _tsdb_base()
    if require_patreon and not is_patreon:
        raise RuntimeError(
            "fetch_full_season needs a Patreon TSDB_API_KEY. "
            "Sign up: https://www.patreon.com/thedatadb"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    season_str = str(season)
    cache_path = cache_dir / f"tsdb_season_{tsdb_id}_{season_str}.json"
    if cache_path.exists() and not _is_current_season(season_str):
        payload = _read_json_lenient(cache_path.read_text())
    else:
        # v2: /schedule/league/<id>/<season>   v1: /eventsseason.php?id=X&s=YYYY
        url = (
            f"{base}/schedule/league/{tsdb_id}/{season_str}"
            if is_patreon
            else f"{base}/eventsseason.php?id={tsdb_id}&s={season_str}"
        )
        response = httpx.get(url, timeout=TIMEOUT_S)
        response.raise_for_status()
        payload = _read_json_lenient(response.text)
        if "schedule" in payload and "events" not in payload:
            payload["events"] = payload["schedule"]
        cache_path.write_text(json.dumps(payload))

    rows = []
    for e in (payload.get("events") or []):
        d = _parse_date(e.get("dateEvent"))
        if d is None:
            continue
        hg = _to_int(e.get("intHomeScore"))
        ag = _to_int(e.get("intAwayScore"))
        rows.append({
            "date": d,
            "home_team": e.get("strHomeTeam"),
            "away_team": e.get("strAwayTeam"),
            "home_goals": hg,
            "away_goals": ag,
            "league_key": league_key,
            "season": season_str,
            "status": e.get("strStatus") or "NS",
            "venue": e.get("strVenue"),
        })
    if not rows:
        return _empty_full()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def upsert_season_into_db(
    db,
    league_key: str,
    seasons: Iterable[str | int],
    *,
    cache_dir: Path,
) -> int:
    """Pull each season + upsert finished matches into the matches table."""
    total = 0
    for season in seasons:
        frame = fetch_full_season(league_key, season, cache_dir=cache_dir)
        # Only keep matches with scores (for training).
        frame = frame.dropna(subset=["home_goals", "away_goals"])
        if frame.empty:
            continue
        frame["home_goals"] = frame["home_goals"].astype(int)
        frame["away_goals"] = frame["away_goals"].astype(int)
        total += db.upsert_matches(
            frame, source="thesportsdb",
            league_key=league_key, league_name=None,
        )
    return total


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_current_season(season: str) -> bool:
    """Bypass cache for the current (still-being-played) season."""
    today = date.today()
    if season.isdigit():
        year = int(season)
        return year == today.year or year == today.year - 1
    return False


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "date", "home_team", "away_team", "league_key",
        "tsdb_event_id", "venue", "time_utc", "status",
    ])


def _empty_full() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "date", "home_team", "away_team", "home_goals", "away_goals",
        "league_key", "season", "status", "venue",
    ])
