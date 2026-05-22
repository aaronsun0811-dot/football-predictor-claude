"""Pull recent finished matches into the local DB.

The full season refresh in ``scrape/update.py`` is expensive (per-season CSV
pulls, many API requests, FBref scraping). For day-to-day operations all we
actually need is "what happened in the last N days" — that's the slice the
prediction-audit module needs to score yesterday's calls.

This module is the lightweight version. For each league we know how to ask,
it pulls FINISHED matches in ``[today - days_back, today]`` and upserts them
into the ``matches`` table.

Routing per league:

  * If a league has a ``football_data_org`` competition code AND
    ``FOOTBALL_DATA_ORG_KEY`` is set → use that (10-req/min free tier, fast,
    no season-locked).
  * Else if the league has an ``api_football_id`` AND an API-Football key is
    configured → use that.
  * Else skip (we'd need football-data.co.uk CSV, which is full-season and
    handled by the heavier daily refresh).

Returns a report dict, never raises — per-league failures are logged and
attached to the response so the dashboard can surface them.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from data.database import Database
from scrape.api_football import (
    ApiFootballClient,
    client_from_env,
    fixtures_to_matches,
)
from scrape.football_data import fetch_season as fdcouk_fetch_season
from scrape.football_data_org import (
    LEAGUE_KEY_TO_CODE as FDORG_CODES,
    FootballDataOrgClient,
    FootballDataOrgError,
)
from scrape.registry import League, LeagueRegistry


# Per-run reports get persisted to disk so /data-health can show
# "last backfill: 4.2s ago, +72 matches, 2 errors" without re-running.
# One file per UTC day; later runs that day overwrite. Keep ~30 days.
REPORTS_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "backfill-reports"
REPORTS_KEEP_DAYS = 30


ProgressCallback = Callable[[dict[str, Any]], None]
"""Called once per league with the per-league row that was just appended.

The row dict matches what ends up in ``report["leagues"]``: at minimum
``{league_key, source, inserted}`` plus optional ``error`` / ``fetched`` /
``skipped`` keys. The callback is best-effort — exceptions inside it are
swallowed so a logging glitch can't crash the whole batch.
"""


def backfill_recent_results(
    db: Database,
    *,
    days_back: int = 7,
    registry: LeagueRegistry | None = None,
    cache_dir: Path | None = None,
    as_of: date | None = None,
    progress_callback: ProgressCallback | None = None,
    persist_report: bool = True,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Pull finished results from the last ``days_back`` days for every league we can reach.

    Failures (missing key, 403, rate limit) are recorded per-league and the
    function continues — never raises. The caller (CLI / scheduler / dashboard)
    decides what to do with the report.

    ``progress_callback`` (optional) is invoked once per league as soon as that
    league's per-source attempt is decided. Useful for the CLI to print live
    progress instead of staring at a blank screen for 5+ minutes.

    ``persist_report=True`` writes the final report to ``reports_dir`` (default
    ``data/cache/backfill-reports/<YYYY-MM-DD>.json``) so the dashboard can
    read it back later.
    """
    registry = registry or LeagueRegistry()
    as_of = as_of or date.today()
    date_from = as_of - timedelta(days=int(days_back))
    cache_dir = Path(cache_dir) if cache_dir else (
        Path(__file__).resolve().parents[1] / "data" / "cache"
    )
    started_at = datetime.utcnow()
    started_monotonic = time.monotonic()

    leagues_report: list[dict[str, Any]] = []
    total_inserted = 0
    total_errors = 0

    def _emit(row: dict[str, Any]) -> None:
        """Append the row to the running report and notify the callback."""
        leagues_report.append(row)
        if progress_callback is not None:
            try:
                progress_callback(row)
            except Exception:  # noqa: BLE001 — never let logging crash the batch
                pass

    fdorg_key = os.environ.get("FOOTBALL_DATA_ORG_KEY") or os.environ.get("FOOTBALL_DATA_API_KEY")

    fdorg_client = None
    if fdorg_key:
        try:
            fdorg_client = FootballDataOrgClient(
                cache_dir=cache_dir / "football-data-org",
                api_key=fdorg_key,
            )
        except Exception as exc:  # noqa: BLE001 — diagnostic, non-fatal
            fdorg_client = None
            _emit({
                "league_key": None,
                "source": "football-data.org",
                "error": f"client init failed: {exc}",
                "inserted": 0,
            })

    # API-Football client (only built if we have a key)
    af_client: ApiFootballClient | None
    try:
        af_client = client_from_env()
    except Exception:  # noqa: BLE001 — diagnostic
        af_client = None

    for league in registry.all():
        # Try football-data.org first (cheaper, no season constraint).
        if fdorg_client is not None and league.key in FDORG_CODES:
            code = FDORG_CODES[league.key]
            try:
                frame = fdorg_client.fetch_matches(
                    code,
                    status="FINISHED",
                    date_from=date_from,
                    date_to=as_of,
                )
                frame = frame.dropna(subset=["home_goals", "away_goals"]) if not frame.empty else frame
                if not frame.empty:
                    frame["home_goals"] = frame["home_goals"].astype(int)
                    frame["away_goals"] = frame["away_goals"].astype(int)
                    inserted = db.upsert_matches(
                        frame,
                        source="football-data.org",
                        league_key=league.key,
                        league_name=league.name,
                    )
                else:
                    inserted = 0
                total_inserted += inserted
                _emit({
                    "league_key": league.key,
                    "source": "football-data.org",
                    "inserted": int(inserted),
                    "fetched": int(len(frame)),
                })
                continue
            except FootballDataOrgError as exc:
                total_errors += 1
                _emit({
                    "league_key": league.key,
                    "source": "football-data.org",
                    "error": str(exc)[:160],
                    "inserted": 0,
                })
                # Fall through to API-Football
            except Exception as exc:  # noqa: BLE001 — diagnostic
                total_errors += 1
                _emit({
                    "league_key": league.key,
                    "source": "football-data.org",
                    "error": f"{type(exc).__name__}: {exc}"[:160],
                    "inserted": 0,
                })

        # football-data.co.uk CSV — second tier. Free, no API key, no rate limit.
        # Covers leagues fd.org doesn't (Belgian Pro / English lower divisions /
        # Bundesliga 2 / Serie B / Ligue 2 / Segunda) at the cost of slightly
        # noisier team names. Pulled at the season level (cached) then filtered
        # to our window, so the per-call cost is one cheap CSV download for the
        # current season plus zero for finished ones.
        if league.football_data_code:
            try:
                inserted, fetched = _backfill_via_fdcouk(
                    league,
                    db=db,
                    cache_dir=cache_dir,
                    date_from=date_from,
                    as_of=as_of,
                )
                total_inserted += inserted
                _emit({
                    "league_key": league.key,
                    "source": "football-data.co.uk",
                    "inserted": int(inserted),
                    "fetched": int(fetched),
                })
                continue
            except Exception as exc:  # noqa: BLE001 — diagnostic
                total_errors += 1
                _emit({
                    "league_key": league.key,
                    "source": "football-data.co.uk",
                    "error": f"{type(exc).__name__}: {exc}"[:160],
                    "inserted": 0,
                })
                # Fall through to API-Football

        # API-Football fallback (or primary, for leagues without fd.org code).
        if af_client is not None and league.api_football_id is not None:
            try:
                # Pick a season that overlaps our date window. For Aug-Jun
                # leagues, "from_date.year" lands in the correct season-key.
                season = _season_for(league, date_from)
                fixtures = af_client.fetch_fixtures(
                    league_id=int(league.api_football_id),
                    season=int(season),
                    from_date=date_from,
                    to_date=as_of,
                )
                frame = fixtures_to_matches(fixtures, league=league)
                if not frame.empty:
                    inserted = db.upsert_matches(
                        frame,
                        source="api-football",
                        league_key=league.key,
                        league_name=league.name,
                    )
                else:
                    inserted = 0
                total_inserted += inserted
                _emit({
                    "league_key": league.key,
                    "source": "api-football",
                    "inserted": int(inserted),
                    "fetched": int(len(frame)),
                })
                continue
            except Exception as exc:  # noqa: BLE001 — diagnostic
                total_errors += 1
                _emit({
                    "league_key": league.key,
                    "source": "api-football",
                    "error": f"{type(exc).__name__}: {exc}"[:160],
                    "inserted": 0,
                })
                continue

        # No reachable source for this league.
        _emit({
            "league_key": league.key,
            "source": None,
            "skipped": "no_source",
            "inserted": 0,
        })

    finished_at = datetime.utcnow()
    duration_s = round(time.monotonic() - started_monotonic, 2)
    report = {
        "generated_at": finished_at.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_s": duration_s,
        "window": {"from": date_from.isoformat(), "to": as_of.isoformat()},
        "totals": {
            "leagues_reached": sum(1 for r in leagues_report if r.get("source")),
            "inserted": int(total_inserted),
            "errors": int(total_errors),
        },
        "leagues": leagues_report,
    }
    if persist_report:
        try:
            # The report filename uses ``as_of`` so a backfill run "as of
            # May 18" lands in ``2026-05-18.json`` even if the wall clock
            # has rolled into May 19 by the time we save. Makes tests
            # deterministic across midnight and aligns the file with the
            # audit-window reference date rather than the literal save time.
            save_report(report, reports_dir=reports_dir, today=as_of)
        except Exception:  # noqa: BLE001 — persistence is best-effort
            pass
    return report


def save_report(
    report: dict[str, Any],
    *,
    reports_dir: Path | None = None,
    today: date | None = None,
) -> Path:
    """Persist one run's report. One file per day (later wins). Returns the path written."""
    dest_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    day = today or date.today()
    path = dest_dir / f"{day.isoformat()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    _prune_old_reports(dest_dir)
    return path


def load_latest_report(reports_dir: Path | None = None) -> dict[str, Any] | None:
    """Return the most recent report or ``None`` if no reports exist."""
    reports = load_recent_reports(reports_dir=reports_dir, limit=1)
    return reports[0] if reports else None


def load_recent_reports(
    *,
    reports_dir: Path | None = None,
    limit: int = 14,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` recent reports, newest first.

    Malformed files are silently skipped so one bad file can't break the
    whole dashboard.
    """
    src_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    if not src_dir.exists():
        return []
    paths = sorted(
        (p for p in src_dir.iterdir() if p.suffix == ".json"),
        key=lambda p: p.name,
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for path in paths[: max(0, int(limit))]:
        try:
            out.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _prune_old_reports(reports_dir: Path) -> None:
    """Keep ~30 days of reports. Older files get deleted."""
    try:
        paths = sorted(
            (p for p in reports_dir.iterdir() if p.suffix == ".json"),
            key=lambda p: p.name,
            reverse=True,
        )
    except OSError:
        return
    for stale in paths[REPORTS_KEEP_DAYS:]:
        try:
            stale.unlink()
        except OSError:
            continue


def _backfill_via_fdcouk(
    league: League,
    *,
    db: Database,
    cache_dir: Path,
    date_from: date,
    as_of: date,
) -> tuple[int, int]:
    """Pull the current season's CSV, filter to ``[date_from, as_of]``, upsert.

    Returns ``(inserted, fetched)`` — fetched is the number of rows that landed
    inside the window, inserted is the number of rows touched by upsert. They
    should be equal in practice; the distinction matches the other tiers'
    reporting shape.
    """
    code = league.football_data_code
    if not code:
        return 0, 0

    # Heuristic: pick the season whose calendar contains ``date_from``. For
    # European Aug-May leagues this means months 1-7 → previous year, 8-12 →
    # current year. fd.co.uk's CSV for an in-progress season auto-refreshes
    # because the scraper re-downloads when ``_is_in_progress`` is true.
    if date_from.month >= 8:
        season_start = date_from.year
    else:
        season_start = date_from.year - 1

    cache_subdir = cache_dir / "football-data"
    frame = fdcouk_fetch_season(code, season_start, cache_dir=cache_subdir)
    if frame.empty:
        return 0, 0

    # Window filter
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    in_window = frame[(frame["date"] >= date_from) & (frame["date"] <= as_of)]
    if in_window.empty:
        return 0, 0

    inserted = db.upsert_matches(
        in_window,
        source="football-data.co.uk",
        league_key=league.key,
        league_name=league.name,
    )
    return int(inserted), int(len(in_window))


def _season_for(league, ref_date: date) -> int:
    """Pick the API-Football season key whose calendar overlaps ``ref_date``.

    * Calendar-year leagues (USA/Brazil/Argentina/Japan/Korea/China/Liga MX) →
      simply ``ref_date.year``.
    * Aug-Jun (Europe) → year that the season *started*, so spring 2026 returns
      2025 (= 2025/26 season).

    We clamp to the API-Football free-plan ceiling (2024) so calls don't 403.
    """
    from scrape.api_football import FREE_PLAN_MAX_SEASON

    calendar_year_countries = {"USA", "BRA", "ARG", "JPN", "KOR", "CHN"}
    if league.country in calendar_year_countries or league.key == "liga_mx":
        season = ref_date.year
    else:
        season = ref_date.year if ref_date.month >= 7 else ref_date.year - 1
    # Clamp to free-plan ceiling; if user has a paid plan they can pass seasons
    # explicitly via the higher-level update path.
    return min(season, FREE_PLAN_MAX_SEASON)
