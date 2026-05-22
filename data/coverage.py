from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data.database import DEFAULT_DB_PATH, Database
from scrape.registry import LeagueRegistry


@dataclass(frozen=True)
class CoverageConfig:
    ready_match_threshold: int = 80
    sparse_match_threshold: int = 1


def build_coverage_report(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    config: CoverageConfig | None = None,
    registry: LeagueRegistry | None = None,
) -> dict[str, Any]:
    config = config or CoverageConfig()
    registry = registry or LeagueRegistry()
    db = Database(db_path)
    db.init()

    matches = db.fetch_matches(deduplicate=True)
    raw_matches = db.fetch_matches(deduplicate=False)
    player_stats = _read_optional_table(db, "player_season_stats")

    rows: list[dict[str, Any]] = []
    for league in registry.all():
        league_matches = _filter_league(matches, league.key)
        league_raw = _filter_league(raw_matches, league.key)
        league_players = _filter_league(player_stats, league.key)
        match_count = int(len(league_matches))
        status = _coverage_status(match_count, config=config)
        rows.append(
            {
                "key": league.key,
                "name": league.name,
                "country": league.country,
                "tier": league.tier,
                "status": status,
                "match_count": match_count,
                "raw_match_rows": int(len(league_raw)),
                "xg_match_rows": _xg_count(league_matches),
                "team_count": _team_count(league_matches),
                "player_stat_rows": int(len(league_players)),
                "earliest_match": _date_or_none(league_matches, "min"),
                "latest_match": _date_or_none(league_matches, "max"),
                "sources": _source_counts(league_raw),
                "football_data_code": league.football_data_code,
                "api_football_id": league.api_football_id,
                "needs_external_provider": league.football_data_code is None,
                "note": league.note,
            }
        )

    loaded = [row for row in rows if row["match_count"] > 0]
    return {
        "summary": {
            "configured_leagues": len(rows),
            "loaded_leagues": len(loaded),
            "empty_leagues": len(rows) - len(loaded),
            "total_matches": int(len(matches)),
            "total_player_stat_rows": int(len(player_stats)),
        },
        "leagues": sorted(rows, key=lambda row: (row["tier"], row["name"])),
    }


def _read_optional_table(db: Database, table: str) -> pd.DataFrame:
    try:
        return pd.read_sql_table(table, db.engine)
    except ValueError:
        return pd.DataFrame()


def _filter_league(frame: pd.DataFrame, league_key: str) -> pd.DataFrame:
    if frame.empty or "league_key" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["league_key"] == league_key].copy()


def _coverage_status(match_count: int, *, config: CoverageConfig) -> str:
    if match_count >= config.ready_match_threshold:
        return "ready"
    if match_count >= config.sparse_match_threshold:
        return "sparse"
    return "empty"


def _team_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return len(set(frame["home_team"].dropna().astype(str)) | set(frame["away_team"].dropna().astype(str)))


def _xg_count(frame: pd.DataFrame) -> int:
    if frame.empty or not {"home_xg", "away_xg"}.issubset(frame.columns):
        return 0
    return int((frame["home_xg"].notna() & frame["away_xg"].notna()).sum())


def _date_or_none(frame: pd.DataFrame, mode: str) -> str | None:
    if frame.empty:
        return None
    value = frame["date"].min() if mode == "min" else frame["date"].max()
    return str(value)


def _source_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "source" not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame["source"].value_counts().items()}
