from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd
import numpy as np
from sqlalchemy import Date, DateTime, Float, Integer, String, UniqueConstraint, create_engine, delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.types import JSON


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "football.sqlite3"


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint(
            "source",
            "league_key",
            "season",
            "date",
            "home_team",
            "away_team",
            name="uq_match_identity",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(64), default="manual", index=True)
    league_key: Mapped[str] = mapped_column(String(64), index=True)
    league_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    season: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    home_team: Mapped[str] = mapped_column(String(128), index=True)
    away_team: Mapped[str] = mapped_column(String(128), index=True)
    home_goals: Mapped[int] = mapped_column(Integer)
    away_goals: Mapped[int] = mapped_column(Integer)
    result: Mapped[str | None] = mapped_column(String(1), nullable=True)
    home_xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    home_elo: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_elo: Mapped[float | None] = mapped_column(Float, nullable=True)
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    neutral_site: Mapped[bool | None] = mapped_column(Integer, nullable=True)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        onupdate=utc_now,
    )


class Rating(Base):
    __tablename__ = "ratings"
    __table_args__ = (
        UniqueConstraint("scope", "team", "rating_date", name="uq_rating_identity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scope: Mapped[str] = mapped_column(String(32), index=True)
    team: Mapped[str] = mapped_column(String(128), index=True)
    country: Mapped[str | None] = mapped_column(String(32), nullable=True)
    rating_date: Mapped[date] = mapped_column(Date, index=True)
    elo: Mapped[float] = mapped_column(Float)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class Player(Base):
    __tablename__ = "players"
    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_player_identity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    external_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    birth_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    nationality: Mapped[str | None] = mapped_column(String(64), nullable=True)
    position: Mapped[str | None] = mapped_column(String(64), nullable=True)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        onupdate=utc_now,
    )


class PlayerSeasonStat(Base):
    __tablename__ = "player_season_stats"
    __table_args__ = (
        UniqueConstraint(
            "source",
            "player_external_id",
            "team",
            "league_key",
            "season",
            name="uq_player_season_stat_identity",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    player_external_id: Mapped[str] = mapped_column(String(64), index=True)
    player_name: Mapped[str] = mapped_column(String(128), index=True)
    team: Mapped[str] = mapped_column(String(128), index=True)
    league_key: Mapped[str] = mapped_column(String(64), index=True)
    season: Mapped[str] = mapped_column(String(32), index=True)
    position: Mapped[str | None] = mapped_column(String(64), nullable=True)
    appearances: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lineups: Mapped[int | None] = mapped_column(Integer, nullable=True)
    minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    assists: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        onupdate=utc_now,
    )


class UpdateState(Base):
    __tablename__ = "update_state"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class MatchOdds(Base):
    """Bookmaker closing odds per match.

    Used by the ROI simulator to evaluate the value-finder strategy against
    real Bet365 closing prices, which are the most informative public market
    benchmark for football.
    """
    __tablename__ = "match_odds"
    __table_args__ = (
        UniqueConstraint(
            "league_key", "date", "home_team", "away_team", "source",
            name="uq_match_odds_identity",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league_key: Mapped[str] = mapped_column(String(64), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    home_team: Mapped[str] = mapped_column(String(128), index=True)
    away_team: Mapped[str] = mapped_column(String(128), index=True)
    # 'b365_closing' (Bet365 close), 'b365_opening', 'pinnacle', etc.
    source: Mapped[str] = mapped_column(String(32), default="b365_closing", index=True)
    odds_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    odds_draw: Mapped[float | None] = mapped_column(Float, nullable=True)
    odds_away: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Database:
    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.engine = create_engine_for_path(self.path)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        from data.schema import ensure_schema

        ensure_schema(self.engine, Base.metadata)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_matches(
        self,
        rows: pd.DataFrame | Sequence[dict[str, Any]],
        *,
        source: str,
        league_key: str,
        league_name: str | None = None,
    ) -> int:
        frame = pd.DataFrame(rows).copy()
        if frame.empty:
            return 0
        frame = _normalize_match_frame(
            frame,
            source=source,
            league_key=league_key,
            league_name=league_name,
        )

        changed = 0
        with self.session() as session:
            for payload in frame.to_dict(orient="records"):
                existing = session.execute(
                    select(Match).where(
                        Match.source == payload["source"],
                        Match.league_key == payload["league_key"],
                        Match.season == payload["season"],
                        Match.date == payload["date"],
                        Match.home_team == payload["home_team"],
                        Match.away_team == payload["away_team"],
                    )
                ).scalar_one_or_none()
                if existing is None:
                    session.add(Match(**payload))
                    changed += 1
                else:
                    for key, value in payload.items():
                        if key in {"home_xg", "away_xg"} and value is None and getattr(existing, key) is not None:
                            continue
                        setattr(existing, key, value)
                    changed += 1
        return changed

    def upsert_ratings(
        self,
        rows: pd.DataFrame | Sequence[dict[str, Any]],
        *,
        scope: str,
        rating_date: date | None = None,
    ) -> int:
        frame = pd.DataFrame(rows).copy()
        if frame.empty:
            return 0
        rating_date = rating_date or date.today()
        frame = _normalize_rating_frame(frame, scope=scope, rating_date=rating_date)

        changed = 0
        with self.session() as session:
            for payload in frame.to_dict(orient="records"):
                existing = session.execute(
                    select(Rating).where(
                        Rating.scope == payload["scope"],
                        Rating.team == payload["team"],
                        Rating.rating_date == payload["rating_date"],
                    )
                ).scalar_one_or_none()
                if existing is None:
                    session.add(Rating(**payload))
                    changed += 1
                else:
                    for key, value in payload.items():
                        setattr(existing, key, value)
                    changed += 1
        return changed

    def upsert_player_stats(
        self,
        rows: pd.DataFrame | Sequence[dict[str, Any]],
        *,
        source: str,
        league_key: str,
        season: str,
    ) -> int:
        frame = pd.DataFrame(rows).copy()
        if frame.empty:
            return 0
        frame = _normalize_player_stat_frame(
            frame,
            source=source,
            league_key=league_key,
            season=season,
        )

        changed = 0
        with self.session() as session:
            for payload in frame.to_dict(orient="records"):
                player_payload = payload.pop("_player")
                player = session.execute(
                    select(Player).where(
                        Player.source == player_payload["source"],
                        Player.external_id == player_payload["external_id"],
                    )
                ).scalar_one_or_none()
                if player is None:
                    session.add(Player(**player_payload))
                else:
                    for key, value in player_payload.items():
                        setattr(player, key, value)

                stat = session.execute(
                    select(PlayerSeasonStat).where(
                        PlayerSeasonStat.source == payload["source"],
                        PlayerSeasonStat.player_external_id == payload["player_external_id"],
                        PlayerSeasonStat.team == payload["team"],
                        PlayerSeasonStat.league_key == payload["league_key"],
                        PlayerSeasonStat.season == payload["season"],
                    )
                ).scalar_one_or_none()
                if stat is None:
                    session.add(PlayerSeasonStat(**payload))
                    changed += 1
                else:
                    for key, value in payload.items():
                        setattr(stat, key, value)
                    changed += 1
        return changed

    def fetch_matches(
        self,
        *,
        league_key: str | None = None,
        source: str | None = None,
        since: date | None = None,
        until: date | None = None,
        deduplicate: bool = True,
    ) -> pd.DataFrame:
        with self.session() as session:
            stmt = select(Match)
            if league_key:
                stmt = stmt.where(Match.league_key == league_key)
            if source:
                stmt = stmt.where(Match.source == source)
            if since:
                stmt = stmt.where(Match.date >= since)
            if until:
                stmt = stmt.where(Match.date <= until)
            stmt = stmt.order_by(Match.date)
            rows = session.execute(stmt).scalars().all()
        frame = _matches_to_frame(rows)
        if deduplicate and not frame.empty:
            frame = _deduplicate_matches(frame)
        return frame

    def latest_rating(
        self,
        team: str,
        *,
        scope: str,
        on_or_before: date | None = None,
    ) -> float | None:
        on_or_before = on_or_before or date.today()
        with self.session() as session:
            stmt = (
                select(Rating)
                .where(Rating.scope == scope)
                .where(Rating.team == team)
                .where(Rating.rating_date <= on_or_before)
                .order_by(Rating.rating_date.desc())
                .limit(1)
            )
            row = session.execute(stmt).scalar_one_or_none()
        return None if row is None else float(row.elo)

    def mark_update_success(self, key: str, *, meta: dict[str, Any] | None = None) -> None:
        self._set_update_state(key, last_error=None, meta=meta)

    def mark_update_error(self, key: str, error: Exception | str) -> None:
        self._set_update_state(key, last_error=str(error), meta=None)

    def export_csv(self, table: str, output_path: str | Path) -> Path:
        if table not in _TABLES:
            raise ValueError(f"table must be one of: {', '.join(sorted(_TABLES))}")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.read_sql_table(table, self.engine)
        frame.to_csv(output, index=False)
        return output

    def reset_table(self, table: str) -> int:
        model = _TABLES.get(table)
        if model is None:
            raise ValueError(f"table must be one of: {', '.join(sorted(_TABLES))}")
        with self.session() as session:
            result = session.execute(delete(model))
            return int(result.rowcount or 0)

    def _set_update_state(
        self,
        key: str,
        *,
        last_error: str | None,
        meta: dict[str, Any] | None,
    ) -> None:
        with self.session() as session:
            row = session.get(UpdateState, key)
            if row is None:
                row = UpdateState(key=key)
                session.add(row)
            row.last_success_at = None if last_error else utc_now()
            row.last_error = last_error
            row.meta = meta


def create_engine_for_path(path: str | Path) -> Engine:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", future=True)


def init_database(path: str | Path = DEFAULT_DB_PATH) -> Database:
    db = Database(path)
    db.init()
    return db


_TABLES = {
    "matches": Match,
    "ratings": Rating,
    "players": Player,
    "player_season_stats": PlayerSeasonStat,
    "update_state": UpdateState,
}


def _normalize_match_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    league_key: str,
    league_name: str | None,
) -> pd.DataFrame:
    required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing match columns: {sorted(missing)}")

    normalized = pd.DataFrame(index=frame.index)
    normalized["source"] = _series_or_default(frame, "source", source)
    normalized["league_key"] = _series_or_default(frame, "league_key", league_key)
    normalized["league_name"] = _series_or_default(frame, "league_name", league_name)
    normalized["date"] = pd.to_datetime(frame["date"]).dt.date
    normalized["season"] = _series_or_default(frame, "season", None)
    normalized["season"] = normalized["season"].fillna(normalized["date"].map(_season_from_date))
    # Canonicalize team names at the boundary so multi-source data merges
    # into one team identity instead of fragmenting ("Real Madrid CF" vs
    # "Real Madrid" etc). See config/team_aliases.yaml.
    from data.team_normalize import canonicalize as _canon
    normalized["home_team"] = frame["home_team"].astype(str).map(_canon)
    normalized["away_team"] = frame["away_team"].astype(str).map(_canon)
    normalized["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce").astype(int)
    normalized["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce").astype(int)
    normalized["result"] = _series_or_default(frame, "result", None)
    normalized["home_xg"] = _optional_float(frame, "home_xg")
    normalized["away_xg"] = _optional_float(frame, "away_xg")
    normalized["home_elo"] = _optional_float(frame, "home_elo")
    normalized["away_elo"] = _optional_float(frame, "away_elo")
    normalized["stage"] = _series_or_default(frame, "stage", None)
    normalized["neutral_site"] = _series_or_default(frame, "neutral_site", None)
    normalized["raw"] = frame.apply(lambda row: _json_safe_dict(row.to_dict()), axis=1)
    return normalized.where(pd.notna(normalized), None)


def _normalize_rating_frame(
    frame: pd.DataFrame,
    *,
    scope: str,
    rating_date: date,
) -> pd.DataFrame:
    team_col = "team" if "team" in frame.columns else "Club"
    elo_col = "elo" if "elo" in frame.columns else "Elo"
    if team_col not in frame.columns or elo_col not in frame.columns:
        raise ValueError("Rating rows need team/Club and elo/Elo columns.")

    normalized = pd.DataFrame(index=frame.index)
    normalized["scope"] = scope
    normalized["team"] = frame[team_col].astype(str)
    normalized["country"] = _series_or_default(frame, "country", None)
    if normalized["country"].isna().all() and "Country" in frame.columns:
        normalized["country"] = frame["Country"]
    normalized["rating_date"] = rating_date
    normalized["elo"] = pd.to_numeric(frame[elo_col], errors="coerce")
    rank_values = _series_or_default(frame, "rank", None)
    if rank_values.isna().all() and "Rank" in frame.columns:
        rank_values = frame["Rank"]
    normalized["rank"] = pd.to_numeric(rank_values, errors="coerce")
    normalized["rank"] = normalized["rank"].astype("Int64").astype(object)
    normalized["raw"] = frame.apply(lambda row: _json_safe_dict(row.to_dict()), axis=1)
    normalized = normalized.dropna(subset=["team", "elo"])
    return normalized.where(pd.notna(normalized), None)


def _normalize_player_stat_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    league_key: str,
    season: str,
) -> pd.DataFrame:
    required = {"player_external_id", "player_name", "team"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing player stat columns: {sorted(missing)}")

    normalized = pd.DataFrame(index=frame.index)
    normalized["source"] = _series_or_default(frame, "source", source)
    normalized["player_external_id"] = frame["player_external_id"].astype(str)
    normalized["player_name"] = frame["player_name"].astype(str)
    normalized["team"] = frame["team"].astype(str)
    normalized["league_key"] = _series_or_default(frame, "league_key", league_key)
    normalized["season"] = _series_or_default(frame, "season", season).astype(str)
    normalized["position"] = _series_or_default(frame, "position", None)
    normalized["appearances"] = _optional_int(frame, "appearances")
    normalized["lineups"] = _optional_int(frame, "lineups")
    normalized["minutes"] = _optional_int(frame, "minutes")
    normalized["goals"] = _optional_int(frame, "goals")
    normalized["assists"] = _optional_int(frame, "assists")
    normalized["rating"] = _optional_float(frame, "rating")
    normalized["raw"] = frame.apply(lambda row: _json_safe_dict(row.to_dict()), axis=1)

    players = pd.DataFrame(index=frame.index)
    players["source"] = normalized["source"]
    players["external_id"] = normalized["player_external_id"]
    players["name"] = normalized["player_name"]
    players["birth_date"] = _optional_date(frame, "birth_date")
    players["age"] = _optional_int(frame, "age")
    players["nationality"] = _series_or_default(frame, "nationality", None)
    players["position"] = normalized["position"]
    players["raw"] = normalized["raw"]
    normalized["_player"] = players.where(pd.notna(players), None).to_dict(orient="records")
    return normalized.where(pd.notna(normalized), None)


def _matches_to_frame(rows: Iterable[Match]) -> pd.DataFrame:
    records = [
        {
            "date": row.date,
            "source": row.source,
            "league_key": row.league_key,
            "league_name": row.league_name,
            "season": row.season,
            "home_team": row.home_team,
            "away_team": row.away_team,
            "home_goals": row.home_goals,
            "away_goals": row.away_goals,
            "result": row.result,
            "home_xg": row.home_xg,
            "away_xg": row.away_xg,
            "home_elo": row.home_elo,
            "away_elo": row.away_elo,
            "stage": row.stage,
            "neutral_site": row.neutral_site,
        }
        for row in rows
    ]
    return pd.DataFrame.from_records(records)


def _deduplicate_matches(frame: pd.DataFrame) -> pd.DataFrame:
    source_priority = {
        "api-football": 3,
        "football-data.co.uk": 2,
        "manual": 1,
    }
    working = frame.copy()
    working["_source_priority"] = working["source"].map(source_priority).fillna(0)
    working["_xg_priority"] = (
        working.get("home_xg").notna() & working.get("away_xg").notna()
        if {"home_xg", "away_xg"}.issubset(working.columns)
        else False
    )
    working = working.sort_values(["date", "_xg_priority", "_source_priority"])
    return (
        working.drop_duplicates(
            subset=["league_key", "date", "home_team", "away_team"],
            keep="last",
        )
        .drop(columns=["_source_priority", "_xg_priority"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def _optional_float(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _optional_int(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce").astype("Int64").astype(object)
    return values.where(pd.notna(values), None)


def _optional_date(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index)
    return pd.to_datetime(frame[column], errors="coerce").dt.date


def _series_or_default(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _season_from_date(value: date) -> str:
    start = value.year if value.month >= 7 else value.year - 1
    return f"{start}/{start + 1}"


def _json_safe_dict(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            cleaned[str(key)] = None
        elif isinstance(value, (dict, list, tuple)):
            cleaned[str(key)] = _json_safe_value(value)
        elif _is_missing(value):
            cleaned[str(key)] = None
        elif isinstance(value, (pd.Timestamp, datetime)):
            cleaned[str(key)] = value.isoformat()
        elif isinstance(value, date):
            cleaned[str(key)] = value.isoformat()
        elif isinstance(value, np.generic):
            cleaned[str(key)] = value.item()
        else:
            cleaned[str(key)] = value
    return cleaned


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(inner) for inner in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if _is_missing(value):
        return None
    return value


def _is_missing(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (np.ndarray, list, tuple)):
        return False
    return bool(missing)
