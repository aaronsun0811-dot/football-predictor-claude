"""football-data.co.uk historical results.

The site publishes per-season CSV files for each European league at:

  https://www.football-data.co.uk/mmz4281/<season>/<code>.csv

Where <season> looks like ``2425`` for 2024/25 and <code> is the league code
(E0, E1, SP1, ...). Files contain Div, Date, HomeTeam, AwayTeam, FTHG, FTAG,
FTR, plus a host of betting columns we ignore.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281"
USER_AGENT = "football-predictor/0.1 (research; +contact via repo)"
TIMEOUT_S = 60.0

CORE_COLUMNS = {
    "Div": "league_code",
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
}


def season_token(season_start_year: int) -> str:
    """Convert a starting year (e.g. 2024) to the URL token ``2425``."""
    if not 1990 <= season_start_year <= 2100:
        raise ValueError(f"Implausible season start: {season_start_year}")
    end_year = season_start_year + 1
    return f"{season_start_year % 100:02d}{end_year % 100:02d}"


def recent_seasons(years_back: int = 5, *, today: date | None = None) -> list[int]:
    """Return season start years to cover at least ``years_back`` years.

    European leagues run Aug-May. After August we include the new season as
    in-progress; before that we wait — football-data updates weekly.
    """
    today = today or date.today()
    current_start = today.year if today.month >= 8 else today.year - 1
    return list(range(current_start - years_back + 1, current_start + 1))


def fetch_season(code: str, season_start_year: int, *, cache_dir: Path) -> pd.DataFrame:
    """Fetch one season for one league. Cached on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = season_token(season_start_year)
    target = cache_dir / f"{code}_{token}.csv"

    if not target.exists() or _is_in_progress(season_start_year):
        url = f"{BASE_URL}/{token}/{code}.csv"
        response = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
        if response.status_code == 404:
            # Season not yet published — return empty.
            return _empty_results_frame(code)
        response.raise_for_status()
        target.write_bytes(response.content)

    return _read_results_csv(target, code)


def fetch_recent(code: str, *, years_back: int = 5, cache_dir: Path) -> pd.DataFrame:
    """Concatenate the last ``years_back`` seasons for one league."""
    frames = []
    for year in recent_seasons(years_back):
        try:
            frame = fetch_season(code, year, cache_dir=cache_dir)
            if not frame.empty:
                frames.append(frame)
        except httpx.HTTPError as exc:
            print(f"[football-data] {code} {year}: {exc}")
    if not frames:
        return _empty_results_frame(code)
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def fetch_many(codes: Iterable[str], *, years_back: int = 5, cache_dir: Path) -> pd.DataFrame:
    """Fetch multiple leagues and stack into one frame."""
    frames = [fetch_recent(code, years_back=years_back, cache_dir=cache_dir) for code in codes]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty_results_frame("")
    return pd.concat(frames, ignore_index=True)


def _is_in_progress(season_start_year: int) -> bool:
    today = date.today()
    season_end = date(season_start_year + 1, 7, 1)
    season_start = date(season_start_year, 7, 1)
    return season_start <= today < season_end


def _empty_results_frame(code: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=["league_code", "date", "home_team", "away_team", "home_goals", "away_goals", "result"]
    )


def _read_results_csv(path: Path, code: str) -> pd.DataFrame:
    # The site occasionally embeds stray bytes; let pandas be lenient.
    raw = pd.read_csv(path, encoding_errors="replace", on_bad_lines="skip")
    available = [col for col in CORE_COLUMNS if col in raw.columns]
    if not available:
        return _empty_results_frame(code)
    frame = raw[available].rename(columns=CORE_COLUMNS).copy()
    frame = frame.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    # football-data dates are usually dd/mm/yy or dd/mm/yyyy.
    frame["date"] = pd.to_datetime(frame["date"], dayfirst=True, errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame["home_goals"] = frame["home_goals"].astype(int)
    frame["away_goals"] = frame["away_goals"].astype(int)
    frame["league_code"] = frame.get("league_code", code)
    return frame.reset_index(drop=True)
