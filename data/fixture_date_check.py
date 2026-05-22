"""Proactive cross-check of TheSportsDB upcoming dates against football-data.org.

TheSportsDB's free tier sometimes carries stale fixture dates — see the
Arsenal-vs-Burnley case in round 23/24. By the time the audit module catches
it (post-match), we've already made and logged predictions for the wrong day.

This module runs at /upcoming computation time: for each fixture from TSDB,
look up the same team pair in fd.org's SCHEDULED matches. If fd.org has a
*different* date, flag it. The user sees a warning chip on the prediction card
BEFORE acting on it.

Cost: one fd.org request per league_key the first time per day, then cached
24h via the existing ``FootballDataOrgClient`` cache. Behind the rate-limit
bucket so it stays polite. Failures are swallowed — date warnings are
nice-to-have, never the reason /upcoming breaks.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any

import pandas as pd

from data.team_normalize import canonicalize
from scrape.football_data_org import (
    LEAGUE_KEY_TO_CODE as FDORG_CODES,
    FootballDataOrgClient,
)


def cross_check_dates(
    fixtures_df: pd.DataFrame,
    *,
    cache_dir: Path,
    api_key: str | None = None,
    today: datetime.date | None = None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """For each TSDB fixture, look up fd.org's scheduled date for the same pair.

    Returns a lookup keyed by ``(league_key, canon_home, canon_away)`` →
    ``{match_in_fdorg, fdorg_date, days_off, fdorg_status}``. Fixtures whose
    league isn't covered by fd.org (MLS, J1, K1, ...) are simply absent from
    the result — callers should treat "missing" as "no opinion."

    Never raises. fd.org rate-limit / 403 / network errors → empty result.
    """
    if fixtures_df.empty:
        return {}

    today = today or datetime.date.today()
    api_key = api_key or os.environ.get("FOOTBALL_DATA_ORG_KEY") or os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        return {}  # no key → no cross-check, that's fine

    # Only attempt leagues that (a) appear in our fixtures AND (b) fd.org covers.
    leagues_in_fixtures = set(fixtures_df["league_key"].astype(str).unique())
    relevant_leagues = {lg: code for lg, code in FDORG_CODES.items() if lg in leagues_in_fixtures}
    if not relevant_leagues:
        return {}

    try:
        client = FootballDataOrgClient(cache_dir=cache_dir, api_key=api_key)
    except Exception:  # noqa: BLE001 — diagnostic
        return {}

    # For each relevant league, fetch SCHEDULED matches once. fd.org caches for
    # 24h, so this is a single round-trip per league per day at worst.
    fdorg_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for league_key, code in relevant_leagues.items():
        try:
            scheduled = client.fetch_matches(
                code, status="SCHEDULED",
                date_from=today,
                date_to=today + datetime.timedelta(days=30),
            )
        except Exception:  # noqa: BLE001 — soft fail per league
            continue
        if scheduled.empty:
            continue
        for row in scheduled.itertuples(index=False):
            h = canonicalize(str(row.home_team)) or str(row.home_team)
            a = canonicalize(str(row.away_team)) or str(row.away_team)
            fdorg_date = pd.to_datetime(row.date).date()
            fdorg_index[(league_key, h, a)] = {
                "fdorg_date": fdorg_date.isoformat(),
                "fdorg_status": str(getattr(row, "status", "SCHEDULED")),
            }

    # Compare each input fixture against fd.org's view
    warnings: dict[tuple[str, str, str], dict[str, Any]] = {}
    for fx in fixtures_df.itertuples(index=False):
        league = str(fx.league_key)
        if league not in relevant_leagues:
            continue
        canon_h = canonicalize(str(fx.home_team)) or str(fx.home_team)
        canon_a = canonicalize(str(fx.away_team)) or str(fx.away_team)
        try:
            tsdb_date = pd.to_datetime(fx.date).date()
        except (ValueError, TypeError):
            continue

        # Try exact match first, then home/away swapped (sources sometimes flip)
        hit = fdorg_index.get((league, canon_h, canon_a))
        flipped = False
        if hit is None:
            hit = fdorg_index.get((league, canon_a, canon_h))
            flipped = hit is not None
        if hit is None:
            # fd.org doesn't have this fixture as SCHEDULED in our 30d window.
            # Could be: cup match fd.org doesn't track, fixture already
            # finished, or fd.org just lags. Mark as "unknown" so the UI can
            # show absence-of-confirmation differently from "actively
            # contradicted."
            warnings[(league, canon_h, canon_a)] = {
                "match_in_fdorg": False,
                "fdorg_date": None,
                "days_off": None,
                "flipped": False,
            }
            continue

        fdorg_date = datetime.date.fromisoformat(hit["fdorg_date"])
        days_off = (fdorg_date - tsdb_date).days
        warnings[(league, canon_h, canon_a)] = {
            "match_in_fdorg": True,
            "fdorg_date": hit["fdorg_date"],
            "fdorg_status": hit["fdorg_status"],
            "days_off": int(days_off),  # signed: +N = fd.org is later, -N = earlier
            "flipped": flipped,
        }
    return warnings


def attach_date_warnings(
    fixtures: list[dict[str, Any]],
    warnings: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    """Mutate fixtures in place, attaching ``date_check`` blocks.

    ``date_check`` shape:
      * ``status: "confirmed"``  — fd.org agrees on the date
      * ``status: "warning"``    — fd.org has a different date (TSDB stale)
      * ``status: "unknown"``    — fd.org has no entry for this pair
      * ``status: "not_covered"`` — league isn't in fd.org's catalog

    The UI shows a small chip only for ``warning`` cases — confirmed and
    unknown are silent (not actionable).
    """
    for fx in fixtures:
        league = str(fx.get("league_key", ""))
        canon_h = canonicalize(str(fx.get("home_team", ""))) or fx.get("home_team")
        canon_a = canonicalize(str(fx.get("away_team", ""))) or fx.get("away_team")

        if league not in FDORG_CODES:
            fx["date_check"] = {"status": "not_covered"}
            continue

        warning = warnings.get((league, canon_h, canon_a))
        if warning is None:
            fx["date_check"] = {"status": "unknown"}
            continue

        if not warning["match_in_fdorg"]:
            fx["date_check"] = {"status": "unknown"}
            continue

        if warning["days_off"] == 0:
            fx["date_check"] = {
                "status": "confirmed",
                "fdorg_date": warning["fdorg_date"],
            }
        else:
            fx["date_check"] = {
                "status": "warning",
                "fdorg_date": warning["fdorg_date"],
                "days_off": warning["days_off"],
                "flipped": warning.get("flipped", False),
            }
