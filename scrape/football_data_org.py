"""football-data.org client (free tier).

Sign-up: <https://www.football-data.org/client/register> — instant email
delivery of an API key. Free tier rate-limit: 10 requests/minute, with
access to TIER_ONE competitions. Set ``FOOTBALL_DATA_ORG_KEY`` in env to use.

Free-tier league coverage we care about:

  PL    English Premier League
  PD    Spanish La Liga (Primera División)
  BL1   German Bundesliga
  FL1   French Ligue 1
  SA    Italian Serie A
  PPL   Portuguese Primeira Liga
  DED   Dutch Eredivisie
  ELC   English Championship
  BSA   Brazilian Série A             ← fills our biggest gap
  CL    UEFA Champions League         ← real UCL data instead of cross-league fit
  EL    UEFA Europa League
  EC    European Championship (Euro)
  WC    FIFA World Cup
  CLI   Copa Libertadores             ← South American club data
  CSL   Chinese Super League          ← Tier 3 league !

Use this when our existing football-data.co.uk CSV pipeline doesn't cover
a league (mainly Tier 3 + continental cups + intercontinental events).
"""
from __future__ import annotations

import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import httpx
import pandas as pd


BASE_URL = "https://api.football-data.org/v4"
USER_AGENT = "football-predictor/0.1 (+https://github.com/)"
TIMEOUT_S = 30.0
# Free-tier rate limit is 10/min — we leave headroom.
MIN_INTERVAL_S = 6.5


# Map our internal league_key → football-data.org competition code.
# Codes verified against /competitions on 2026-05-17.
LEAGUE_KEY_TO_CODE: dict[str, str] = {
    # Top-flight European (duplicate coverage with football-data.co.uk — useful
    # for cross-checking but normally skipped to avoid double-counting)
    "premier_league": "PL",
    "la_liga": "PD",
    "bundesliga": "BL1",
    "ligue_1": "FL1",
    "serie_a": "SA",
    "primeira": "PPL",
    "eredivisie": "DED",
    "championship": "ELC",
    # Leagues where this is the ONLY free source we have
    "brasileirao": "BSA",
    "chinese_super_league": "CSL",
    # Continental club cups
    "champions_league": "CL",
    "europa_league": "EL",
    "copa_libertadores": "CLI",
    # National-team cups
    "euro": "EC",
    "world_cup": "WC",
}


class FootballDataOrgError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("FOOTBALL_DATA_ORG_KEY") or os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        raise FootballDataOrgError(
            "FOOTBALL_DATA_ORG_KEY is not set. Get a free key at "
            "https://www.football-data.org/client/register (30 seconds) "
            "and `export FOOTBALL_DATA_ORG_KEY=your_key`."
        )
    return key


class FootballDataOrgClient:
    """Minimal client with polite rate-limiting and an on-disk cache."""

    def __init__(self, *, cache_dir: Path, api_key: str | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or _api_key()
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        delta = time.time() - self._last_request_at
        if delta < MIN_INTERVAL_S:
            time.sleep(MIN_INTERVAL_S - delta)
        self._last_request_at = time.time()

    def _get_json(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._throttle()
        url = f"{BASE_URL}/{path.lstrip('/')}"
        headers = {"X-Auth-Token": self.api_key, "User-Agent": USER_AGENT}
        response = httpx.get(url, headers=headers, params=params or {}, timeout=TIMEOUT_S)
        if response.status_code == 429:
            raise FootballDataOrgError("Rate limited (10/min on free tier). Slow down.")
        if response.status_code == 403:
            raise FootballDataOrgError(
                f"403 Forbidden — your free key probably can't access this competition. "
                f"URL: {url}"
            )
        response.raise_for_status()
        return response.json()

    def list_competitions(self) -> pd.DataFrame:
        """Public catalog — works WITHOUT an API key (and a bad key is rejected
        with 400, so we send the request bare)."""
        self._throttle()
        response = httpx.get(
            f"{BASE_URL}/competitions",
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json()
        rows = [
            {
                "code": c.get("code"),
                "name": c.get("name"),
                "area": c.get("area", {}).get("name"),
                "plan": c.get("plan"),
                "id": c.get("id"),
            }
            for c in payload.get("competitions", [])
        ]
        return pd.DataFrame(rows)

    def fetch_matches(
        self,
        competition_code: str,
        *,
        season_start_year: int | None = None,
        status: str = "FINISHED",
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> pd.DataFrame:
        """Fetch matches for a competition.

        ``season_start_year=2024`` gets the 2024/25 season. ``status='SCHEDULED'``
        gets upcoming fixtures. Cached to ``data/cache/football-data-org/``.
        """
        cache_key = f"{competition_code}_{season_start_year or 'all'}_{status}.json"
        cache_path = self.cache_dir / cache_key
        if cache_path.exists() and self._cache_fresh(cache_path, status):
            payload = self._read_cache(cache_path)
        else:
            params: dict[str, Any] = {"status": status}
            if season_start_year is not None:
                params["season"] = season_start_year
            if date_from is not None:
                params["dateFrom"] = date_from.isoformat()
            if date_to is not None:
                params["dateTo"] = date_to.isoformat()
            payload = self._get_json(
                f"competitions/{competition_code}/matches", params=params,
            )
            self._write_cache(cache_path, payload)

        return _matches_to_frame(payload.get("matches", []))

    def _cache_fresh(self, path: Path, status: str) -> bool:
        # Finished-match caches are immutable; refresh scheduled ones daily.
        if status == "FINISHED":
            return True
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        return age_hours < 24

    @staticmethod
    def _read_cache(path: Path) -> dict[str, Any]:
        import json
        return json.loads(path.read_text())

    @staticmethod
    def _write_cache(path: Path, payload: dict[str, Any]) -> None:
        import json
        path.write_text(json.dumps(payload))


def _matches_to_frame(matches: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for m in matches:
        score = m.get("score", {}).get("fullTime", {}) or {}
        rows.append({
            "date": _parse_date(m.get("utcDate")),
            "home_team": (m.get("homeTeam") or {}).get("name"),
            "away_team": (m.get("awayTeam") or {}).get("name"),
            "home_goals": score.get("home"),
            "away_goals": score.get("away"),
            "status": m.get("status"),
            "matchday": m.get("matchday"),
            "stage": m.get("stage"),
        })
    if not rows:
        # No matches in window → empty frame with the expected columns so
        # downstream `.dropna(subset=...)` calls don't KeyError.
        return pd.DataFrame(columns=[
            "date", "home_team", "away_team", "home_goals", "away_goals",
            "status", "matchday", "stage",
        ])
    frame = pd.DataFrame(rows)
    frame = frame.dropna(subset=["date", "home_team", "away_team"])
    # Only keep matches with scores for training; scheduled ones for fixture views.
    return frame.reset_index(drop=True)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def upsert_matches_into_db(
    db,
    competition_code: str,
    league_key: str,
    *,
    cache_dir: Path,
    seasons: Iterable[int] | None = None,
    api_key: str | None = None,
) -> int:
    """Convenience: fetch + upsert into our SQLite matches table."""
    client = FootballDataOrgClient(cache_dir=cache_dir, api_key=api_key)
    seasons = list(seasons) if seasons is not None else [date.today().year - 1, date.today().year]
    inserted = 0
    for season in seasons:
        try:
            frame = client.fetch_matches(competition_code, season_start_year=season)
        except FootballDataOrgError as exc:
            print(f"[football-data.org] {competition_code} {season}: {exc}")
            continue
        # Drop matches without scores (scheduled / postponed).
        frame = frame.dropna(subset=["home_goals", "away_goals"])
        if frame.empty:
            continue
        frame["home_goals"] = frame["home_goals"].astype(int)
        frame["away_goals"] = frame["away_goals"].astype(int)
        frame["league_key"] = league_key
        inserted += db.upsert_matches(
            frame, source="football-data.org",
            league_key=league_key, league_name=None,
        )
    return inserted
