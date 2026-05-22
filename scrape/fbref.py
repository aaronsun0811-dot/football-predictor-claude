"""FBref xG enrichment.

FBref publishes per-match xG for matches in competitions it covers (most of
the top European leagues + a few Americas/Asia leagues). We scrape the
"Scores & Fixtures" page for a given competition season.

URL pattern:
  https://fbref.com/en/comps/<id>/<season>/schedule/

The page contains a results table with columns including ``xG`` and ``xG.1``.
Pandas can read it directly via ``read_html`` after a small bit of cleanup.

This scraper is intentionally lenient: if FBref blocks (their rate limit is
aggressive — 10 requests per minute when not signed in), we fall back to
returning an empty frame so the rest of the pipeline still works without xG.
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd

USER_AGENT = "Mozilla/5.0 (compatible; football-predictor/0.1)"
TIMEOUT_S = 60.0
BASE = "https://fbref.com/en/comps"
# Polite delay between requests. FBref bans IPs that ignore this.
INTER_REQUEST_DELAY_S = 6.5


def _season_token(season_start_year: int) -> str:
    return f"{season_start_year}-{season_start_year + 1}"


def fetch_schedule_xg(
    fbref_id: int,
    season_start_year: int,
    *,
    cache_dir: Path,
) -> pd.DataFrame:
    """Return a minimal frame of (date, home, away, home_xg, away_xg, home_g, away_g)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    season = _season_token(season_start_year)
    cache_html = cache_dir / f"fbref_{fbref_id}_{season}.html"

    if not cache_html.exists() or _season_is_in_progress(season_start_year):
        url = f"{BASE}/{fbref_id}/{season}/schedule/"
        try:
            response = httpx.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
                timeout=TIMEOUT_S,
            )
            response.raise_for_status()
            cache_html.write_bytes(response.content)
            time.sleep(INTER_REQUEST_DELAY_S)
        except httpx.HTTPError as exc:
            print(f"[fbref] {fbref_id} {season}: {exc}")
            if not cache_html.exists():
                return _empty()

    return _parse_schedule_html(cache_html)


def fetch_recent_xg(
    fbref_id: int,
    *,
    years_back: int,
    cache_dir: Path,
    current_season_start_year: int | None = None,
) -> pd.DataFrame:
    """Fetch recent FBref xG schedules and concatenate them."""
    current = current_season_start_year if current_season_start_year is not None else _current_season_start_year()
    season_years = range(current - years_back + 1, current + 1)
    frames = [
        fetch_schedule_xg(fbref_id, season_start_year=year, cache_dir=cache_dir)
        for year in season_years
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def _season_is_in_progress(season_start_year: int) -> bool:
    today = date.today()
    return date(season_start_year, 7, 1) <= today < date(season_start_year + 1, 7, 1)


def _current_season_start_year() -> int:
    today = date.today()
    return today.year if today.month >= 7 else today.year - 1


def _parse_schedule_html(path: Path) -> pd.DataFrame:
    try:
        tables = pd.read_html(path)
    except ValueError:
        return _empty()
    # The schedule page can carry several tables (overall + per-stage).
    # We want the largest one with both team and xG columns.
    candidates = []
    for tbl in tables:
        cols = {str(c) for c in tbl.columns}
        if {"Home", "Away"}.issubset(cols) and ("xG" in cols or "Score" in cols):
            candidates.append(tbl)
    if not candidates:
        return _empty()
    candidates.sort(key=len, reverse=True)
    raw = candidates[0]
    frame = pd.DataFrame()
    frame["date"] = pd.to_datetime(raw.get("Date"), errors="coerce")
    frame["home_team"] = raw.get("Home")
    frame["away_team"] = raw.get("Away")
    if "Score" in raw.columns:
        score = raw["Score"].astype(str).str.split("[–—-]", regex=True, expand=True)
        frame["home_goals"] = pd.to_numeric(score[0], errors="coerce")
        frame["away_goals"] = pd.to_numeric(score[1] if score.shape[1] > 1 else None, errors="coerce")
    else:
        frame["home_goals"] = None
        frame["away_goals"] = None
    if "xG" in raw.columns:
        frame["home_xg"] = pd.to_numeric(raw["xG"], errors="coerce")
    if "xG.1" in raw.columns:
        frame["away_xg"] = pd.to_numeric(raw["xG.1"], errors="coerce")
    frame = frame.dropna(subset=["date", "home_team", "away_team"]).reset_index(drop=True)
    return frame


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "home_xg",
            "away_xg",
        ]
    )
