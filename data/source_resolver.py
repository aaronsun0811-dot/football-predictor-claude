"""Cross-source overlap detector + field-priority resolver.

Today this module is **mostly defensive**: the three data sources
(football-data.co.uk, football-data.org, api-football) are kept disjoint by the
backfill router in ``data/result_backfill.py``, so overlapping rows are rare.

But "rare" is not "never." If a future change accidentally pulls the same
league from two sources, this module surfaces the overlap on the dashboard
before it silently corrupts the training set (double-counted matches → biased
team strength estimates).

The module also declares a **field-priority table** — for the day overlaps
do appear and we need to pick a winning value per field. It is a pure
function: no DB writes, no schema changes. Higher layers (data-health,
prediction-audit, training pipeline) consume the output if/when they need to.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from data.team_normalize import canonicalize


# ---------------------------------------------------------------------------
# Field priority declarations
# ---------------------------------------------------------------------------
# When two sources disagree on a field, the first source in the list wins.
# Sources not listed for a field are ignored for that field — for closing odds
# in particular, only football-data.co.uk has them, so the list is unary.
#
# Rationale:
#   * football-data.org has a small editorial team that manually verifies
#     scores before publishing. Highest trust for scores / stage / matchday.
#   * api-football pulls from sports data feeds within minutes of FT — faster
#     than fd.org, but occasionally has feed errors. Trusted second.
#   * football-data.co.uk is a volunteer-maintained CSV. Largest volume, but
#     historical typos exist. Trusted last for scores; trusted exclusively
#     for closing-odds columns (no other source carries them).
FIELD_PRIORITY: dict[str, list[str]] = {
    # "manual" entries are user-entered corrections / fills (see POST /manual-result).
    # They sit at the top of every score-related priority list because the user
    # explicitly disagreed with — or filled in for — every automated source.
    "home_goals":    ["manual", "football-data.org", "api-football", "football-data.co.uk"],
    "away_goals":    ["manual", "football-data.org", "api-football", "football-data.co.uk"],
    "result":        ["manual", "football-data.org", "api-football", "football-data.co.uk"],
    "stage":         ["manual", "football-data.org", "api-football", "football-data.co.uk"],
    "matchday":      ["manual", "football-data.org", "api-football", "football-data.co.uk"],
    # xG and odds: niche columns owned by specific sources.
    "home_xg":       ["fbref", "api-football"],
    "away_xg":       ["fbref", "api-football"],
    "b365_home":     ["football-data.co.uk"],
    "b365_draw":     ["football-data.co.uk"],
    "b365_away":     ["football-data.co.uk"],
}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _canonical_key(frame: pd.DataFrame) -> pd.Series:
    """Build a stable per-fixture key: ``date|canonical_home|canonical_away``."""
    h = frame["home_team"].astype(str).map(canonicalize)
    a = frame["away_team"].astype(str).map(canonicalize)
    d = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return d + "|" + h + "|" + a


def find_overlaps(matches: pd.DataFrame) -> pd.DataFrame:
    """Return every canonical fixture present in ≥2 sources.

    Output columns: ``key, date, home_team, away_team, sources, n_sources, rows``
    where ``rows`` is the list of row-dicts (one per source) for that fixture.
    """
    if matches.empty or "source" not in matches.columns:
        return _empty_overlap_frame()
    work = matches.copy()
    work["_key"] = _canonical_key(work)
    grouped = work.groupby("_key")
    overlap_rows: list[dict[str, Any]] = []
    for key, sub in grouped:
        sources = sorted(sub["source"].astype(str).unique())
        if len(sources) < 2:
            continue
        first = sub.iloc[0]
        overlap_rows.append({
            "key": key,
            "date": pd.to_datetime(first["date"]).strftime("%Y-%m-%d"),
            "home_team": canonicalize(str(first["home_team"])) or str(first["home_team"]),
            "away_team": canonicalize(str(first["away_team"])) or str(first["away_team"]),
            "league_key": str(first.get("league_key", "")),
            "sources": sources,
            "n_sources": len(sources),
            "rows": sub.drop(columns=["_key"]).to_dict(orient="records"),
        })
    if not overlap_rows:
        return _empty_overlap_frame()
    return pd.DataFrame(overlap_rows)


def find_score_conflicts(matches: pd.DataFrame) -> pd.DataFrame:
    """Return overlapping fixtures where sources disagree on the final score.

    A row is a conflict if either ``home_goals`` or ``away_goals`` has >1
    distinct non-null value across the sources reporting that fixture.

    Output columns: ``key, date, home_team, away_team, league_key,
    home_goals_by_source, away_goals_by_source, winning_score, winning_source``
    """
    overlaps = find_overlaps(matches)
    if overlaps.empty:
        return _empty_conflict_frame()

    conflict_rows: list[dict[str, Any]] = []
    for ov in overlaps.to_dict(orient="records"):
        rows = ov["rows"]
        h_by_src = {r["source"]: r.get("home_goals") for r in rows
                    if r.get("home_goals") is not None}
        a_by_src = {r["source"]: r.get("away_goals") for r in rows
                    if r.get("away_goals") is not None}
        h_vals = {v for v in h_by_src.values()}
        a_vals = {v for v in a_by_src.values()}
        if len(h_vals) <= 1 and len(a_vals) <= 1:
            continue

        # Apply priority to pick the "winning" score.
        win_src, win_h = _pick_with_priority(h_by_src, FIELD_PRIORITY["home_goals"])
        _,       win_a = _pick_with_priority(a_by_src, FIELD_PRIORITY["away_goals"])
        conflict_rows.append({
            "key": ov["key"],
            "date": ov["date"],
            "home_team": ov["home_team"],
            "away_team": ov["away_team"],
            "league_key": ov["league_key"],
            "home_goals_by_source": h_by_src,
            "away_goals_by_source": a_by_src,
            "winning_score": (
                f"{int(win_h)}-{int(win_a)}"
                if (win_h is not None and win_a is not None)
                else None
            ),
            "winning_source": win_src,
        })
    if not conflict_rows:
        return _empty_conflict_frame()
    return pd.DataFrame(conflict_rows)


def pick_winning_value(
    values_by_source: dict[str, Any],
    field: str,
    *,
    priority: dict[str, list[str]] | None = None,
) -> tuple[str | None, Any]:
    """Choose the winning value for ``field`` from a ``{source: value}`` map.

    Returns ``(winning_source, winning_value)``. If no priority entry exists
    for the field, falls back to "first non-null source seen" (deterministic
    by dict insertion order). Useful as a building block when a future
    feature pipeline needs to collapse multi-source rows into one.
    """
    priority = priority or FIELD_PRIORITY
    return _pick_with_priority(values_by_source, priority.get(field, []))


def _pick_with_priority(
    values_by_source: dict[str, Any],
    ranked: list[str],
) -> tuple[str | None, Any]:
    # Drop nulls so a higher-priority source with None doesn't shadow a
    # lower-priority source that actually has the value.
    non_null = {s: v for s, v in values_by_source.items() if v is not None}
    if not non_null:
        return (None, None)
    for src in ranked:
        if src in non_null:
            return (src, non_null[src])
    # No priority match → return first available (deterministic by dict order).
    first_src, first_val = next(iter(non_null.items()))
    return (first_src, first_val)


# ---------------------------------------------------------------------------
# Summary builder — what the dashboard surfaces
# ---------------------------------------------------------------------------

def overlap_summary(matches: pd.DataFrame) -> dict[str, Any]:
    """High-level snapshot for the data-health dashboard."""
    overlaps = find_overlaps(matches)
    conflicts = find_score_conflicts(matches)
    return {
        "overlap_count": int(len(overlaps)),
        "score_conflict_count": int(len(conflicts)),
        # Top 10 conflicts, sorted newest-first — the few that matter.
        "top_conflicts": (
            conflicts.sort_values("date", ascending=False).head(10).to_dict(orient="records")
            if not conflicts.empty else []
        ),
        # Top 10 overlapping fixtures without score conflict (informational only —
        # might still be worth a glance to make sure router stays disjoint).
        "top_overlaps_no_conflict": _overlaps_minus_conflicts(overlaps, conflicts),
    }


def _overlaps_minus_conflicts(
    overlaps: pd.DataFrame,
    conflicts: pd.DataFrame,
) -> list[dict[str, Any]]:
    if overlaps.empty:
        return []
    conflict_keys = set(conflicts["key"]) if not conflicts.empty else set()
    benign = overlaps[~overlaps["key"].isin(conflict_keys)]
    if benign.empty:
        return []
    # Drop the heavy ``rows`` payload for the summary view.
    light = benign.drop(columns=["rows"]).sort_values("date", ascending=False).head(10)
    return light.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Empty-frame builders — keep column shape stable for callers
# ---------------------------------------------------------------------------

def _empty_overlap_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "key", "date", "home_team", "away_team", "league_key",
        "sources", "n_sources", "rows",
    ])


def _empty_conflict_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "key", "date", "home_team", "away_team", "league_key",
        "home_goals_by_source", "away_goals_by_source",
        "winning_score", "winning_source",
    ])
