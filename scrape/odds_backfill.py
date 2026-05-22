"""Backfill historical bookmaker closing odds from football-data.co.uk CSVs.

We already cache the CSVs under ``data/cache/football-data/``. The original
match parser only kept FTHG/FTAG/FTR. This module re-reads the same files
and pulls Bet365 closing prices into a separate ``match_odds`` table without
touching the matches table.

Bet365 closing odds are columns ``B365CH`` (home), ``B365CD`` (draw),
``B365CA`` (away). When closing odds are missing (older seasons) we fall
back to opening odds ``B365H/B365D/B365A``.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from data.database import Database, MatchOdds
from scrape.registry import LeagueRegistry

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "football-data"
CSV_PATTERN = re.compile(r"^([A-Z][A-Z0-9]{1,2})_(\d{4})\.csv$")
# football-data codes: 2-char (E0, D1, F1, ...) AND 3-char (SP1, SP2).


def parse_one(path: Path, *, league_key: str) -> pd.DataFrame:
    """Read one CSV and return a normalized odds frame."""
    raw = pd.read_csv(path, encoding_errors="replace", on_bad_lines="skip")
    if "Date" not in raw.columns or "HomeTeam" not in raw.columns:
        return _empty()

    # Build the frame anchored to raw's index so the scalar `league_key`
    # broadcasts to every row instead of only the first.
    frame = pd.DataFrame(index=raw.index)
    frame["league_key"] = league_key
    frame["date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce").dt.date
    frame["home_team"] = raw["HomeTeam"].astype(str)
    frame["away_team"] = raw["AwayTeam"].astype(str)

    # Prefer closing odds; fall back to opening if closing absent.
    for col_close, col_open, target in [
        ("B365CH", "B365H", "odds_home"),
        ("B365CD", "B365D", "odds_draw"),
        ("B365CA", "B365A", "odds_away"),
    ]:
        if col_close in raw.columns:
            frame[target] = pd.to_numeric(raw[col_close], errors="coerce")
        elif col_open in raw.columns:
            frame[target] = pd.to_numeric(raw[col_open], errors="coerce")
        else:
            frame[target] = None

    frame["source"] = "b365_closing" if "B365CH" in raw.columns else "b365_opening"
    frame = frame.dropna(subset=["date", "home_team", "away_team"])
    frame = frame.dropna(subset=["odds_home", "odds_draw", "odds_away"])
    return frame.reset_index(drop=True)


def discover_files(cache_dir: Path = CACHE_DIR) -> list[tuple[str, Path]]:
    """Return (football_data_code, path) pairs from the cache directory."""
    out = []
    for path in sorted(cache_dir.glob("*.csv")):
        match = CSV_PATTERN.match(path.name)
        if match:
            out.append((match.group(1), path))
    return out


def upsert_odds(db: Database, frame: pd.DataFrame) -> int:
    """Bulk upsert into match_odds. Returns number of rows touched."""
    if frame.empty:
        return 0
    rows = frame.to_dict(orient="records")
    stmt = sqlite_insert(MatchOdds.__table__).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["league_key", "date", "home_team", "away_team", "source"],
        set_={
            "odds_home": stmt.excluded.odds_home,
            "odds_draw": stmt.excluded.odds_draw,
            "odds_away": stmt.excluded.odds_away,
        },
    )
    with db.engine.begin() as conn:
        result = conn.execute(stmt)
    return result.rowcount or len(rows)


def backfill_all(db: Database, *, cache_dir: Path = CACHE_DIR) -> dict[str, int]:
    """Backfill every cached CSV into match_odds. Returns per-league counts."""
    db.init()
    registry = LeagueRegistry()
    code_to_key = {
        league.football_data_code: league.key
        for league in registry.all()
        if league.football_data_code
    }
    counts: dict[str, int] = {}
    for code, path in discover_files(cache_dir):
        league_key = code_to_key.get(code)
        if not league_key:
            continue
        frame = parse_one(path, league_key=league_key)
        n = upsert_odds(db, frame)
        counts[league_key] = counts.get(league_key, 0) + n
    return counts


def fetch_odds_frame(db: Database, *, league_key: str | None = None) -> pd.DataFrame:
    """Read odds for joining against matches frames."""
    query = "SELECT league_key, date, home_team, away_team, source, odds_home, odds_draw, odds_away FROM match_odds"
    params: tuple = ()
    if league_key:
        query += " WHERE league_key = ?"
        params = (league_key,)
    with db.engine.begin() as conn:
        frame = pd.read_sql(query, conn, params=params, parse_dates=["date"])
    return frame


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["league_key", "date", "home_team", "away_team",
                 "source", "odds_home", "odds_draw", "odds_away"]
    )
