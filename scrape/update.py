from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable
import re
import unicodedata

import pandas as pd

from data.database import Database, DEFAULT_DB_PATH
from scrape.registry import League, LeagueRegistry
from . import api_football, clubelo, eloratings, fbref, football_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"


@dataclass(frozen=True)
class UpdateReport:
    source: str
    league_key: str | None
    rows_changed: int
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "league_key": self.league_key,
            "rows_changed": self.rows_changed,
            "message": self.message,
        }


class IncrementalUpdater:
    def __init__(
        self,
        *,
        db_path: str | Path = DEFAULT_DB_PATH,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        registry: LeagueRegistry | None = None,
    ) -> None:
        self.db = Database(db_path)
        self.db.init()
        self.cache_dir = Path(cache_dir)
        self.registry = registry or LeagueRegistry()

    def update_all(
        self,
        *,
        leagues: Iterable[str] | None = None,
        years_back: int = 5,
        include_ratings: bool = True,
        include_api_football: bool = False,
        include_players: bool = False,
        include_fbref_xg: bool = False,
    ) -> list[UpdateReport]:
        targets = self._resolve_leagues(leagues)
        reports: list[UpdateReport] = []
        reports.extend(
            self.update_results(
                targets,
                years_back=years_back,
                include_fbref_xg=include_fbref_xg,
            )
        )
        if include_api_football:
            reports.extend(self.update_api_football_results(targets, years_back=years_back))
        if include_api_football and include_players:
            reports.extend(self.update_api_football_players(targets))
        if include_ratings:
            reports.append(self.update_club_elos())
            reports.append(self.update_national_team_elos())

        self.db.mark_update_success(
            "daily_incremental_update",
            meta={"reports": [report.to_dict() for report in reports]},
        )
        return reports

    def update_api_football_players(
        self,
        leagues: Iterable[League],
        *,
        season: int | None = None,
    ) -> list[UpdateReport]:
        client = api_football.client_from_env()
        if client is None:
            return [
                UpdateReport(
                    source="api-football-players",
                    league_key=None,
                    rows_changed=0,
                    message="Skipped: FOOTBALL_API_KEY/API_FOOTBALL_KEY is not set.",
                )
            ]
        reports: list[UpdateReport] = []
        for league in leagues:
            if league.api_football_id is None:
                continue
            target_season = season or api_football.recent_api_seasons(league, years_back=1)[0]
            try:
                frame = api_football.fetch_league_player_stats(
                    league,
                    season=target_season,
                    cache_dir=self.cache_dir / "api-football",
                    client=client,
                )
                changed = self.db.upsert_player_stats(
                    frame,
                    source="api-football",
                    league_key=league.key,
                    season=str(target_season),
                )
                reports.append(
                    UpdateReport(
                        source="api-football-players",
                        league_key=league.key,
                        rows_changed=changed,
                        message=f"Updated player season stats for {target_season}.",
                    )
                )
            except Exception as exc:
                self.db.mark_update_error(f"api-football-players:{league.key}", exc)
                reports.append(
                    UpdateReport(
                        source="api-football-players",
                        league_key=league.key,
                        rows_changed=0,
                        message=f"Update failed: {exc}",
                    )
                )
        return reports

    def update_api_football_results(
        self,
        leagues: Iterable[League],
        *,
        years_back: int = 5,
    ) -> list[UpdateReport]:
        client = api_football.client_from_env()
        if client is None:
            return [
                UpdateReport(
                    source="api-football",
                    league_key=None,
                    rows_changed=0,
                    message="Skipped: FOOTBALL_API_KEY/API_FOOTBALL_KEY is not set.",
                )
            ]

        reports: list[UpdateReport] = []
        for league in leagues:
            if league.api_football_id is None:
                reports.append(
                    UpdateReport(
                        source="api-football",
                        league_key=league.key,
                        rows_changed=0,
                        message="No API-Football league id configured.",
                    )
                )
                continue
            try:
                seasons = api_football.recent_api_seasons(league, years_back=years_back)
                frame = api_football.fetch_league_results(
                    league,
                    seasons=seasons,
                    cache_dir=self.cache_dir / "api-football",
                    client=client,
                )
                changed = self.db.upsert_matches(
                    frame,
                    source="api-football",
                    league_key=league.key,
                    league_name=league.name,
                )
                reports.append(
                    UpdateReport(
                        source="api-football",
                        league_key=league.key,
                        rows_changed=changed,
                        message="Updated API-Football fixtures.",
                    )
                )
            except Exception as exc:
                self.db.mark_update_error(f"api-football:{league.key}", exc)
                reports.append(
                    UpdateReport(
                        source="api-football",
                        league_key=league.key,
                        rows_changed=0,
                        message=f"Update failed: {exc}",
                    )
                )
        return reports

    def update_results(
        self,
        leagues: Iterable[League],
        *,
        years_back: int = 5,
        include_fbref_xg: bool = False,
    ) -> list[UpdateReport]:
        reports: list[UpdateReport] = []
        for league in leagues:
            if not league.football_data_code:
                reports.append(
                    UpdateReport(
                        source="football-data.co.uk",
                        league_key=league.key,
                        rows_changed=0,
                        message="No football-data.co.uk code configured; external provider required.",
                    )
                )
                continue

            try:
                frame = football_data.fetch_recent(
                    league.football_data_code,
                    years_back=years_back,
                    cache_dir=self.cache_dir / "football-data",
                )
                if frame.empty:
                    changed = 0
                    xg_rows_merged = 0
                else:
                    xg_rows_merged = 0
                    if include_fbref_xg and league.fbref_id is not None:
                        xg = fbref.fetch_recent_xg(
                            int(league.fbref_id),
                            years_back=years_back,
                            cache_dir=self.cache_dir / "fbref",
                        )
                        frame = merge_xg(frame, xg)
                        if {"home_xg", "away_xg"}.issubset(frame.columns):
                            xg_rows_merged = int((frame["home_xg"].notna() & frame["away_xg"].notna()).sum())
                    frame["league_key"] = league.key
                    frame["league_name"] = league.name
                    changed = self.db.upsert_matches(
                        frame,
                        source="football-data.co.uk",
                        league_key=league.key,
                        league_name=league.name,
                    )
                reports.append(
                    UpdateReport(
                        source="football-data.co.uk",
                        league_key=league.key,
                        rows_changed=changed,
                        message=(
                            f"Updated results; FBref xG rows merged: {xg_rows_merged}."
                            if include_fbref_xg and league.fbref_id is not None
                            else "Updated results."
                        ),
                    )
                )
            except Exception as exc:
                self.db.mark_update_error(f"results:{league.key}", exc)
                reports.append(
                    UpdateReport(
                        source="football-data.co.uk",
                        league_key=league.key,
                        rows_changed=0,
                        message=f"Update failed: {exc}",
                    )
                )
        return reports

    def update_club_elos(self, *, on: date | None = None) -> UpdateReport:
        try:
            snapshot = clubelo.fetch_snapshot(on=on, cache_dir=self.cache_dir / "clubelo")
            changed = self.db.upsert_ratings(
                snapshot.rename(columns={"Club": "team", "Elo": "elo", "Rank": "rank"}),
                scope="club",
                rating_date=on or date.today(),
            )
            return UpdateReport("clubelo", None, changed, "Updated club Elo ratings.")
        except Exception as exc:
            self.db.mark_update_error("ratings:clubelo", exc)
            return UpdateReport("clubelo", None, 0, f"Update failed: {exc}")

    def update_national_team_elos(self) -> UpdateReport:
        try:
            snapshot = eloratings.fetch_world_ratings(cache_dir=self.cache_dir / "eloratings")
            changed = self.db.upsert_ratings(
                snapshot,
                scope="national",
                rating_date=date.today(),
            )
            return UpdateReport("eloratings.net", None, changed, "Updated national-team Elo ratings.")
        except Exception as exc:
            self.db.mark_update_error("ratings:eloratings", exc)
            return UpdateReport("eloratings.net", None, 0, f"Update failed: {exc}")

    def _resolve_leagues(self, leagues: Iterable[str] | None) -> list[League]:
        if leagues is None:
            return self.registry.all()
        return [self.registry.get(value) for value in leagues]


def run_daily_update(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    leagues: Iterable[str] | None = None,
    years_back: int = 5,
    include_api_football: bool = False,
    include_players: bool = False,
    include_fbref_xg: bool = False,
) -> list[dict[str, object]]:
    updater = IncrementalUpdater(db_path=db_path, cache_dir=cache_dir)
    return [
        report.to_dict()
        for report in updater.update_all(
            leagues=leagues,
            years_back=years_back,
            include_api_football=include_api_football,
            include_players=include_players,
            include_fbref_xg=include_fbref_xg,
        )
    ]


TEAM_ALIASES = {
    "ac milan": "milan",
    "athletic bilbao": "athletic club",
    "man united": "manchester utd",
    "man utd": "manchester utd",
    "man city": "manchester city",
    "newcastle": "newcastle utd",
    "nottingham forest": "nottm forest",
    "nottm forest": "nottm forest",
    "paris sg": "paris sg",
    "paris s g": "paris sg",
    "psg": "paris sg",
    "tottenham": "tottenham hotspur",
    "west brom": "west bromwich albion",
    "west ham": "west ham united",
    "wolves": "wolverhampton wanderers",
}


def merge_xg(left, right):
    """Best-effort FBref xG merge onto result rows.

    FBref and football-data.co.uk do not always use identical team names, so
    merge on date + normalized home/away names instead of raw strings.
    """
    if left.empty or right.empty:
        return left
    if not {"home_xg", "away_xg"}.issubset(right.columns):
        return left

    result_rows = left.copy()
    xg_rows = right.copy()
    result_rows["_date_key"] = _date_key(result_rows["date"])
    xg_rows["_date_key"] = _date_key(xg_rows["date"])
    for frame in (result_rows, xg_rows):
        frame["_home_key"] = frame["home_team"].map(_team_key)
        frame["_away_key"] = frame["away_team"].map(_team_key)

    xg_rows = (
        xg_rows.dropna(subset=["home_xg", "away_xg"])
        .sort_values("_date_key")
        .drop_duplicates(subset=["_date_key", "_home_key", "_away_key"], keep="last")
    )
    enriched = result_rows.merge(
        xg_rows[["_date_key", "_home_key", "_away_key", "home_xg", "away_xg"]],
        on=["_date_key", "_home_key", "_away_key"],
        how="left",
        suffixes=("", "_fbref"),
    )
    for column in ("home_xg", "away_xg"):
        fbref_column = f"{column}_fbref"
        if fbref_column in enriched.columns:
            if column in left.columns:
                enriched[column] = enriched[column].combine_first(enriched[fbref_column])
            else:
                enriched[column] = enriched[fbref_column]
    return enriched.drop(
        columns=[
            "_date_key",
            "_home_key",
            "_away_key",
            "home_xg_fbref",
            "away_xg_fbref",
        ],
        errors="ignore",
    )


def _date_key(values) -> object:
    return pd.to_datetime(values, errors="coerce").dt.date


def _team_key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = " ".join(text.split())
    return TEAM_ALIASES.get(text, text)
