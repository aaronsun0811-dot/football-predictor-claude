"""User-entered match results: the "patch" mechanism.

When none of the three automated sources (football-data.co.uk, fd.org,
api-football) cover a fixture — or when they cover it wrong — the user can
file a manual result. These rows land in the ``matches`` table with
``source="manual"`` and sit at the **top** of every scoring-field priority
list in ``data/source_resolver.py``, so they override any automated value
for the same canonical fixture.

This is the only mechanism for fixing bad-data issues without editing the
DB by hand. Validations are kept tight on purpose:
  * date must be in the past (no "manual predictions")
  * scores must be 0–30 (typo guard)
  * league must resolve through the registry
  * team names get canonicalized at write time, so "Real Madrid CF" and
    "Real Madrid" can't both end up as separate manual entries
"""
from __future__ import annotations

from datetime import date as date_cls
from typing import Any

import pandas as pd

from data.database import Database
from data.team_normalize import canonicalize
from scrape.registry import LeagueRegistry


MAX_GOALS = 30  # If a real match was 30-0, we'll re-think the cap.


def submit_manual_result(
    db: Database,
    *,
    league: str,
    date: date_cls,
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    season: str | None = None,
    neutral_site: bool | None = None,
    stage: str | None = None,
    registry: LeagueRegistry | None = None,
    today: date_cls | None = None,
) -> dict[str, Any]:
    """Upsert one user-entered match result. Returns a small report dict.

    Validates aggressively and raises ``ValueError`` on bad input — the
    FastAPI route surfaces that as a 400 to the caller.
    """
    registry = registry or LeagueRegistry()
    today = today or date_cls.today()

    # --- league ---
    try:
        league_key = registry.normalize(league)
    except KeyError as exc:
        raise ValueError(f"Unknown league: {league!r}") from exc
    league_obj = registry.get(league_key)

    # --- date (no future matches as "results") ---
    if date >= today:
        raise ValueError(
            f"manual result date must be strictly before today ({today.isoformat()}); "
            f"got {date.isoformat()}"
        )

    # --- score sanity ---
    for label, value in (("home_goals", home_goals), ("away_goals", away_goals)):
        if not isinstance(value, int) or value < 0 or value > MAX_GOALS:
            raise ValueError(f"{label} must be int in [0, {MAX_GOALS}]; got {value!r}")

    # --- team names → canonical ---
    canon_home = canonicalize(home_team) or home_team
    canon_away = canonicalize(away_team) or away_team
    if not canon_home.strip() or not canon_away.strip():
        raise ValueError("home_team and away_team must be non-empty")
    if canon_home == canon_away:
        raise ValueError("home_team and away_team must differ")

    # --- season (auto-derive if not given) ---
    if not season:
        season = _derive_season(date, league_obj)

    # --- result outcome string (consistent with our other sources) ---
    if home_goals > away_goals:
        result = "H"
    elif home_goals < away_goals:
        result = "A"
    else:
        result = "D"

    # Build a single-row frame in the same shape upsert_matches expects.
    frame = pd.DataFrame([{
        "date": date,
        "home_team": canon_home,
        "away_team": canon_away,
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
        "result": result,
        "season": season,
        "stage": stage,
        "neutral_site": neutral_site,
    }])

    inserted = db.upsert_matches(
        frame,
        source="manual",
        league_key=league_key,
        league_name=league_obj.name,
    )

    return {
        "inserted": int(inserted),
        "league_key": league_key,
        "season": season,
        "home_team": canon_home,
        "away_team": canon_away,
        "score": f"{home_goals}-{away_goals}",
        "result": result,
    }


def _derive_season(when: date_cls, league_obj) -> str:
    """Pick a season string consistent with how our other scrapers label it.

    For calendar-year leagues (USA / Asia / South America) → ``"YYYY"``.
    For Aug–Jun leagues (Europe + most others) → ``"YYYY"`` of the *start*
    year, so a match on 2026-02-15 returns ``"2025"`` (= 2025/26 season).
    """
    calendar_countries = {"USA", "BRA", "ARG", "JPN", "KOR", "CHN"}
    if (
        (league_obj.country or "") in calendar_countries
        or league_obj.key == "liga_mx"
    ):
        return str(when.year)
    return str(when.year if when.month >= 7 else when.year - 1)


def list_recent_manual_results(db: Database, *, limit: int = 50) -> list[dict[str, Any]]:
    """Newest first, for the dashboard table."""
    with db.engine.begin() as conn:
        frame = pd.read_sql(
            "SELECT date, league_key, home_team, away_team, home_goals, away_goals, "
            "updated_at FROM matches WHERE source = 'manual' "
            "ORDER BY updated_at DESC LIMIT :n",
            conn, params={"n": int(limit)},
        )
    if frame.empty:
        return []
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    if "updated_at" in frame.columns:
        frame["updated_at"] = frame["updated_at"].astype(str)
    return frame.to_dict(orient="records")
