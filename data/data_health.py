"""Aggregate "what data does this install actually have?" report.

Pulls together 4 views in one snapshot:

  1. **Per-league** — match count, date range, source distribution, freshness
  2. **Per-source** — total matches, leagues touched, last write
  3. **API keys** — which are configured, masked previews, quota where available
  4. **Caches** — TheSportsDB / football-data.org / API-Football disk caches
     with file count, total bytes, age of newest entry

The dashboard tab reads ``GET /data-health`` and renders the four sections.
"""
from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from data.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-4:]}"


def _format_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def per_league_health(db: Database) -> list[dict[str, Any]]:
    """One row per league_key, with match counts and freshness."""
    with db.engine.begin() as conn:
        frame = pd.read_sql(
            "SELECT league_key, source, date FROM matches",
            conn, parse_dates=["date"],
        )
    if frame.empty:
        return []

    today = pd.Timestamp(date.today())
    rows: list[dict[str, Any]] = []
    for league_key, sub in frame.groupby("league_key"):
        latest = sub["date"].max()
        earliest = sub["date"].min()
        days_stale = int((today - latest).days)
        sources = sub.source.value_counts().to_dict()
        rows.append({
            "league_key": str(league_key),
            "match_count": int(len(sub)),
            "earliest": str(earliest.date()) if pd.notna(earliest) else None,
            "latest": str(latest.date()) if pd.notna(latest) else None,
            "days_stale": days_stale,
            "freshness": _bucket(days_stale),
            "sources": sources,
            "primary_source": max(sources, key=sources.get) if sources else None,
        })
    rows.sort(key=lambda r: (r["days_stale"], -r["match_count"]))
    return rows


def _bucket(days: int) -> str:
    if days <= 7:
        return "fresh"       # updated within the week
    if days <= 30:
        return "recent"
    if days <= 180:
        return "stale"
    return "very_stale"      # >6 months — data is historical only


def per_source_health(db: Database) -> list[dict[str, Any]]:
    """One row per data source: total matches, leagues covered."""
    with db.engine.begin() as conn:
        frame = pd.read_sql(
            "SELECT source, league_key, date, updated_at FROM matches",
            conn, parse_dates=["date", "updated_at"],
        )
    if frame.empty:
        return []
    out: list[dict[str, Any]] = []
    for source, sub in frame.groupby("source"):
        out.append({
            "source": str(source),
            "match_count": int(len(sub)),
            "leagues": sorted(set(sub.league_key.astype(str).tolist())),
            "latest_match": str(sub["date"].max().date()) if pd.notna(sub["date"].max()) else None,
            "latest_write": str(sub["updated_at"].max()) if "updated_at" in sub.columns and pd.notna(sub["updated_at"].max()) else None,
        })
    out.sort(key=lambda r: -r["match_count"])
    return out


def api_key_health() -> list[dict[str, Any]]:
    """Which API keys are set + masked previews."""
    keys = [
        ("FOOTBALL_DATA_ORG_KEY", "football-data.org",
         "https://www.football-data.org/client/register"),
        ("API_FOOTBALL_KEY", "API-Football",
         "https://dashboard.api-football.com/register"),
        ("TSDB_API_KEY", "TheSportsDB Premium",
         "https://www.patreon.com/thedatadb"),
    ]
    out = []
    for env_name, display, signup in keys:
        # Honor common alternate env-var names too.
        candidates = [env_name]
        if env_name == "API_FOOTBALL_KEY":
            candidates.append("FOOTBALL_API_KEY")
        if env_name == "TSDB_API_KEY":
            candidates.extend(["THESPORTSDB_API_KEY", "THESPORTSDB_KEY"])
        value = None
        used_name = None
        for n in candidates:
            v = os.environ.get(n)
            if v:
                value = v
                used_name = n
                break
        out.append({
            "display_name": display,
            "env_var": env_name,
            "configured": bool(value),
            "masked": _mask_secret(value),
            "found_via": used_name,
            "signup_url": signup,
        })
    return out


CACHE_DIRS = {
    "football-data.co.uk CSVs": "data/cache/football-data",
    "football-data.org JSON":    "data/cache/football-data-org",
    "API-Football":              "data/cache/api-football",
    "TheSportsDB":               "data/cache/tsdb",
    "TheSportsDB seasons":       "data/cache/tsdb-season",
    "FBref xG":                  "data/cache/fbref",
    "ClubElo":                   "data/cache/clubelo",
    "eloratings.net":            "data/cache/eloratings",
    "Upcoming bundle":           "data/cache/upcoming",
}


def cache_health() -> list[dict[str, Any]]:
    """Disk-cache footprint per source."""
    out = []
    now = time.time()
    for label, rel in CACHE_DIRS.items():
        path = PROJECT_ROOT / rel
        if not path.exists():
            out.append({
                "label": label,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": False,
                "file_count": 0,
                "size_bytes": 0,
                "newest_age_seconds": None,
                "newest_age_human": None,
            })
            continue
        files = list(path.rglob("*"))
        files = [f for f in files if f.is_file()]
        size = sum(f.stat().st_size for f in files)
        newest = max((f.stat().st_mtime for f in files), default=0)
        age = (now - newest) if newest else None
        out.append({
            "label": label,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "exists": True,
            "file_count": len(files),
            "size_bytes": int(size),
            "size_human": _human_bytes(size),
            "newest_age_seconds": age,
            "newest_age_human": _format_age(age) if age is not None else None,
        })
    return out


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def build_health_report(db: Database) -> dict[str, Any]:
    """One-stop snapshot for the web tab."""
    per_league = per_league_health(db)
    per_source = per_source_health(db)

    # Real-world accuracy: how did past /upcoming predictions actually do?
    # Best-effort; failures shouldn't break the dashboard.
    audit: dict[str, Any] = {"n_resolved": 0, "accuracy": None}
    try:
        from models.prediction_audit import audit_summary
        matches = pd.read_sql_table("matches", db.engine, columns=["date", "home_team", "away_team", "home_goals", "away_goals"])
        audit = audit_summary(matches)
    except Exception as exc:  # noqa: BLE001 — non-fatal
        audit = {"n_resolved": 0, "accuracy": None, "error": str(exc)[:120]}

    # Unmatched-fixture diagnostic. Attached as ``audit.unmatched`` so the
    # dashboard can show "5 predictions can't resolve — 3 are date mismatches
    # (TSDB), 2 are missing-league data" without spinning up the CLI.
    #
    # ``check_fdorg=True`` enables the post-hoc fd.org fallback: for leagues
    # fd.org covers but our local DB is missing the match, query fd.org's
    # FINISHED list to catch "TSDB stale date + we missed the backfill" cases.
    # Cheap because fd.org's per-league SCHEDULED cache is shared with the
    # /upcoming cross-check and refreshes only every 24h.
    try:
        from data.alias_audit import find_unmatched_fixtures
        unmatched_report = find_unmatched_fixtures(
            db, days_back=30,
            check_fdorg=True,
            cache_dir=PROJECT_ROOT / "data" / "cache" / "football-data-org",
        )
        audit["unmatched"] = {
            "n_unmatched": unmatched_report["n_unmatched"],
            "n_matched": unmatched_report["n_matched"],
            "by_reason": unmatched_report.get("by_reason", {}),
            "sample": unmatched_report["unmatched"][:5],
        }
    except Exception as exc:  # noqa: BLE001 — non-fatal
        audit["unmatched"] = {"error": str(exc)[:120]}

    # Cross-source overlap / conflict scan: today this should be 0, the routers
    # keep sources disjoint by design. If a number > 0 ever shows up here, that
    # means a future change accidentally pulls the same league from two sources.
    overlaps_section: dict[str, Any] = {"overlap_count": 0, "score_conflict_count": 0}
    try:
        from data.source_resolver import overlap_summary
        all_matches = pd.read_sql_table(
            "matches", db.engine,
            columns=["source", "league_key", "date", "home_team", "away_team",
                     "home_goals", "away_goals"],
        )
        overlaps_section = overlap_summary(all_matches)
    except Exception as exc:  # noqa: BLE001 — non-fatal
        overlaps_section = {
            "overlap_count": 0,
            "score_conflict_count": 0,
            "error": str(exc)[:120],
        }

    # Observability for the daily backfill cron. Without this the only signal
    # that the 06:15 job ran is "did the DB grow" — opaque, slow to debug.
    last_backfill: dict[str, Any] | None = None
    try:
        from data.result_backfill import load_latest_report
        latest = load_latest_report()
        if latest is not None:
            last_backfill = _trim_backfill_report(latest)
    except Exception as exc:  # noqa: BLE001 — non-fatal
        last_backfill = {"error": str(exc)[:120]}

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "totals": {
            "total_matches": sum(r["match_count"] for r in per_league),
            "league_count": len(per_league),
            "source_count": len(per_source),
            "fresh_leagues": sum(1 for r in per_league if r["freshness"] == "fresh"),
            "stale_leagues": sum(1 for r in per_league if r["freshness"] in ("stale", "very_stale")),
        },
        "per_league": per_league,
        "per_source": per_source,
        "api_keys": api_key_health(),
        "caches": cache_health(),
        "audit": audit,
        "overlaps": overlaps_section,
        "last_backfill": last_backfill,
    }


def _trim_backfill_report(report: dict[str, Any]) -> dict[str, Any]:
    """Strip the heavy per-league list down to a summary + the rows that matter.

    The dashboard only needs: totals, when, how long, plus the rows that
    inserted something OR errored. Skipped 'no_source' leagues are noise.
    """
    leagues = report.get("leagues", []) or []
    interesting = [
        r for r in leagues
        if r.get("inserted", 0) or r.get("error") or r.get("skipped") not in (None, "no_source")
    ]
    return {
        "started_at": report.get("started_at"),
        "finished_at": report.get("finished_at"),
        "duration_s": report.get("duration_s"),
        "window": report.get("window"),
        "totals": report.get("totals"),
        # Up to 30 most-relevant league rows. ~95% of a typical run.
        "league_rows": interesting[:30],
        "total_league_rows": len(leagues),
    }
