"""Cross-source name-mismatch detector.

For every past-dated prediction in ``history.jsonl``, look up the corresponding
match in the DB. When no canonical row matches, look for fuzzy candidates in
the same league + date and rank them by team-name similarity. The output is a
list of suggested aliases the user can add to ``config/team_aliases.yaml``.

Without this, the audit module silently under-counts resolved predictions —
"why is n_resolved only 1?" turns into hours of manual SQL spelunking. With it,
``python predict.py find-unmatched-fixtures`` produces a punch list in 2s.
"""
from __future__ import annotations

import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from data.database import Database
from data.history_store import iter_all_rows
from data.team_normalize import canonicalize


SIMILARITY_THRESHOLD = 0.55  # min ratio to be considered a fuzzy candidate


def find_unmatched_fixtures(
    db: Database,
    *,
    days_back: int | None = None,
    today: datetime.date | None = None,
    max_candidates_per_fixture: int = 3,
    nearby_window_days: int = 7,
    check_fdorg: bool = False,
    cache_dir: "Path | None" = None,
) -> dict[str, Any]:
    """Walk history.jsonl, find past-dated fixtures whose canonical (date, home, away)
    has no DB match. Return a report listing each unmatched fixture + best candidates.

    The ``reason`` field categorizes WHY the fixture couldn't be resolved:

      * ``name_mismatch`` — same-day candidates with similar (but not equal)
        team names exist in DB. Fix by adding an alias to ``team_aliases.yaml``.
      * ``likely_date_mismatch`` — the SAME team pair appears in DB on a date
        within ±``nearby_window_days``. The prediction's date was probably
        wrong (TheSportsDB upcoming-fixtures can be stale by several days).
        The ``nearby_match`` field has the actual date + score.
      * ``no_db_matches_on_date`` — no DB candidates at all (league probably
        wasn't backfilled for that day, e.g. api-football quota exhausted).
      * ``no_close_match`` — same-day candidates exist but none similar enough
        to be confidently the same fixture.

    Args:
        db: connected Database
        days_back: only consider fixtures from the last N days (None = all)
        today: override "today" for reproducible tests
        max_candidates_per_fixture: cap the candidate list per fixture
        nearby_window_days: ± days to scan for "same team pair, wrong date"
        check_fdorg: if True, for fixtures still unresolved after the local
            DB check, query football-data.org's FINISHED list to see if the
            match happened on a nearby date that we just didn't pull. Catches
            "TheSportsDB stale date + our backfill missed it" cases. Costs
            ~1 fd.org call per league with unresolved fixtures, cached 24h.
        cache_dir: cache directory for fd.org. Required when ``check_fdorg=True``.
    """
    today = today or datetime.date.today()
    cutoff = (today - datetime.timedelta(days=int(days_back))) if days_back else None

    # Gather past-dated unique fixtures from history
    past_fixtures: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in iter_all_rows():
        date_str = row.get("date")
        if not date_str:
            continue
        try:
            fx_date = datetime.date.fromisoformat(date_str)
        except ValueError:
            continue
        if fx_date >= today:
            continue
        if cutoff and fx_date < cutoff:
            continue
        h_raw = str(row.get("home_team") or "")
        a_raw = str(row.get("away_team") or "")
        canon_h = canonicalize(h_raw) or h_raw
        canon_a = canonicalize(a_raw) or a_raw
        key = (date_str, canon_h, canon_a)
        if key not in past_fixtures:
            past_fixtures[key] = {
                "league_key": row.get("league_key"),
                "raw_home": h_raw, "raw_away": a_raw,
            }

    # Load all DB matches (canonicalized)
    m = pd.read_sql_table(
        "matches", db.engine,
        columns=["league_key", "date", "home_team", "away_team", "source"],
    )
    m["ds"] = pd.to_datetime(m["date"]).dt.strftime("%Y-%m-%d")
    m["h_canon"] = m["home_team"].astype(str).map(canonicalize)
    m["a_canon"] = m["away_team"].astype(str).map(canonicalize)
    db_lookup = set(zip(m["ds"], m["h_canon"], m["a_canon"]))

    matched = sum(1 for k in past_fixtures if k in db_lookup)

    # Pre-compute a fast (h_canon, a_canon) → list-of-dates index, scoped to the
    # leagues we'll be querying, so the nearby-date scan stays O(unmatched)
    # instead of O(unmatched × all_matches).
    teampair_index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in m.itertuples(index=False):
        key = (str(row.league_key), str(row.h_canon), str(row.a_canon))
        teampair_index.setdefault(key, []).append({
            "ds": row.ds, "home_team": row.home_team,
            "away_team": row.away_team, "source": row.source,
        })

    unmatched_rows: list[dict[str, Any]] = []
    for (d, ch, ca), meta in past_fixtures.items():
        if (d, ch, ca) in db_lookup:
            continue

        # Look for fuzzy candidates: same league, same date.
        candidates_df = m[(m["league_key"] == meta["league_key"]) & (m["ds"] == d)]
        scored: list[dict[str, Any]] = []
        for row in candidates_df.itertuples(index=False):
            sim_h = _sim(ch, row.h_canon)
            sim_a = _sim(ca, row.a_canon)
            score = max(sim_h, sim_a)
            if score < SIMILARITY_THRESHOLD:
                continue
            suggestion = _suggest_alias(
                ch, ca, row.h_canon, row.a_canon,
                row.home_team, row.away_team,
            )
            scored.append({
                "source": row.source,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "h_canon": row.h_canon,
                "a_canon": row.a_canon,
                "similarity": round(float(score), 3),
                "suggested_alias": suggestion,
            })
        scored.sort(key=lambda r: -r["similarity"])

        # Same team pair on a NEARBY date? That's a TheSportsDB-date-stale
        # signature. Check both orderings (home/away) since the upcoming feed
        # sometimes flips them.
        nearby = _find_nearby_match(
            teampair_index, meta["league_key"], ch, ca, d,
            window_days=nearby_window_days,
        )

        # Pick the most informative reason. Order matters:
        # 1. Same-day name candidate (most actionable) wins over nearby date.
        # 2. Nearby date wins over "nothing".
        # 3. Empty case.
        if scored:
            reason = "name_mismatch"
        elif nearby:
            reason = "likely_date_mismatch"
        elif candidates_df.empty:
            reason = "no_db_matches_on_date"
        else:
            reason = "no_close_match"

        unmatched_rows.append({
            "date": d,
            "league_key": meta["league_key"],
            "raw_home": meta["raw_home"],
            "raw_away": meta["raw_away"],
            "canon_home": ch,
            "canon_away": ca,
            "candidates": scored[:max_candidates_per_fixture],
            "nearby_match": nearby,
            "reason": reason,
        })

    # Optional post-hoc check: for fixtures still labeled "no_db_matches_on_date"
    # or "no_close_match", ask fd.org directly. If fd.org has the same team pair
    # on a nearby date that we just hadn't pulled into our DB, upgrade the
    # reason to ``likely_date_mismatch`` and attach the fd.org evidence.
    # This catches the most useful pattern: TSDB had wrong date AND our
    # backfill missed the league that day.
    if check_fdorg:
        _enrich_with_fdorg(
            unmatched_rows,
            cache_dir=cache_dir,
            today=today,
            window_days=nearby_window_days,
        )

    return {
        "n_past_fixtures": len(past_fixtures),
        "n_matched": matched,
        "n_unmatched": len(unmatched_rows),
        "by_reason": _count_by_reason(unmatched_rows),
        "unmatched": unmatched_rows,
    }


def _count_by_reason(unmatched_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in unmatched_rows:
        r = row.get("reason", "unknown")
        counts[r] = counts.get(r, 0) + 1
    return counts


def _enrich_with_fdorg(
    unmatched_rows: list[dict[str, Any]],
    *,
    cache_dir: Path | None,
    today: datetime.date,
    window_days: int,
) -> None:
    """For each unresolved fixture in an fd.org-covered league, query fd.org's
    FINISHED matches and try to find the same team pair on a nearby date.

    Mutates rows in place: when we find a hit, upgrades ``reason`` to
    ``likely_date_mismatch`` and sets ``nearby_match`` with ``source: "fd.org"``.

    Never raises. fd.org failures (no key, 403, 429, network) silently leave
    the rows unchanged — this is enrichment, not a hard requirement.
    """
    import os  # noqa: PLC0415 — kept local; only this path needs it
    from scrape.football_data_org import (  # noqa: PLC0415 — heavy import
        LEAGUE_KEY_TO_CODE as FDORG_CODES,
        FootballDataOrgClient,
    )

    # Only the fixtures where local DB had nothing useful are worth checking
    # against fd.org. Skip ``name_mismatch`` (already actionable via alias)
    # and existing ``likely_date_mismatch`` (already resolved locally).
    candidates = [
        r for r in unmatched_rows
        if r["reason"] in ("no_db_matches_on_date", "no_close_match")
        and str(r["league_key"]) in FDORG_CODES
    ]
    if not candidates:
        return

    api_key = os.environ.get("FOOTBALL_DATA_ORG_KEY") or os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        return
    if cache_dir is None:
        # Fall back to a reasonable default but warn via early-exit if path
        # doesn't exist — caller should pass cache_dir.
        return
    try:
        client = FootballDataOrgClient(cache_dir=cache_dir, api_key=api_key)
    except Exception:  # noqa: BLE001 — diagnostic
        return

    # Group candidates by league_key so we hit fd.org once per league at most.
    # ``date_from`` is the earliest predicted-date in the league minus the window;
    # ``date_to`` is the latest plus the window. Together they form a window we
    # know all candidates' nearby-date checks fall inside.
    by_league: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_league.setdefault(str(row["league_key"]), []).append(row)

    for league_key, league_rows in by_league.items():
        try:
            dates = [datetime.date.fromisoformat(r["date"]) for r in league_rows]
        except ValueError:
            continue
        date_from = min(dates) - datetime.timedelta(days=window_days)
        date_to = min(today, max(dates) + datetime.timedelta(days=window_days))
        try:
            fdorg_finished = client.fetch_matches(
                FDORG_CODES[league_key], status="FINISHED",
                date_from=date_from, date_to=date_to,
            )
        except Exception:  # noqa: BLE001 — soft fail per league
            continue
        if fdorg_finished.empty:
            continue

        # Build (canon_home, canon_away) → list of {date, ...} for fast lookup
        fdorg_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in fdorg_finished.itertuples(index=False):
            h = canonicalize(str(r.home_team)) or str(r.home_team)
            a = canonicalize(str(r.away_team)) or str(r.away_team)
            entry = {
                "ds": pd.to_datetime(r.date).date().isoformat(),
                "home_team": str(r.home_team),
                "away_team": str(r.away_team),
                "home_goals": _to_int_or_none(getattr(r, "home_goals", None)),
                "away_goals": _to_int_or_none(getattr(r, "away_goals", None)),
            }
            fdorg_index.setdefault((h, a), []).append(entry)

        # For each unresolved row in this league, scan fd.org for nearby
        for row in league_rows:
            ch, ca = row["canon_home"], row["canon_away"]
            pred_date = datetime.date.fromisoformat(row["date"])
            # Try both orderings
            for h, a, flipped in [(ch, ca, False), (ca, ch, True)]:
                entries = fdorg_index.get((h, a), [])
                best: tuple[int, dict[str, Any]] | None = None
                for e in entries:
                    try:
                        e_date = datetime.date.fromisoformat(e["ds"])
                    except ValueError:
                        continue
                    days_off = abs((e_date - pred_date).days)
                    if days_off == 0 or days_off > window_days:
                        continue
                    if best is None or days_off < best[0]:
                        best = (days_off, e)
                if best:
                    days_off, e = best
                    row["nearby_match"] = {
                        "ds": e["ds"],
                        "days_off": days_off,
                        "home_team": e["home_team"],
                        "away_team": e["away_team"],
                        "home_goals": e["home_goals"],
                        "away_goals": e["away_goals"],
                        "source": "football-data.org",
                        "flipped": flipped,
                    }
                    row["reason"] = "likely_date_mismatch"
                    break  # don't check flipped if exact already matched


def _to_int_or_none(x: Any) -> int | None:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _find_nearby_match(
    teampair_index: dict[tuple[str, str, str], list[dict[str, Any]]],
    league_key: Any,
    canon_home: str,
    canon_away: str,
    target_date: str,
    *,
    window_days: int,
) -> dict[str, Any] | None:
    """Look up DB rows for (league, home, away) — and (league, away, home) to
    catch ordering flips — within ±``window_days`` of ``target_date``.

    Returns the closest-by-date match as ``{ds, days_off, home_team,
    away_team, source, flipped}`` or ``None``. "Closest" because if there
    are multiple matches (a real league has two meetings per season), we
    want the one chronologically nearest to what was predicted.
    """
    try:
        target = datetime.date.fromisoformat(target_date)
    except ValueError:
        return None

    # Try both orderings — TheSportsDB and api-football sometimes disagree
    # on which team is "home" for a given fixture.
    keys = [
        (str(league_key), canon_home, canon_away, False),
        (str(league_key), canon_away, canon_home, True),
    ]
    best: tuple[int, dict[str, Any], bool] | None = None
    for lg, h, a, flipped in keys:
        rows = teampair_index.get((lg, h, a), [])
        for row in rows:
            try:
                row_date = datetime.date.fromisoformat(row["ds"])
            except ValueError:
                continue
            days_off = abs((row_date - target).days)
            if days_off == 0 or days_off > window_days:
                continue  # 0 = exact match (already handled), out-of-window = skip
            if best is None or days_off < best[0]:
                best = (days_off, row, flipped)
    if best is None:
        return None
    days_off, row, flipped = best
    return {
        "ds": row["ds"],
        "days_off": days_off,
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "source": row["source"],
        "flipped": flipped,
    }


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _suggest_alias(
    pred_h: str, pred_a: str,
    db_h_canon: str, db_a_canon: str,
    db_h_raw: str, db_a_raw: str,
) -> str | None:
    """If exactly one team-name pair differs, that's our likely alias gap.

    Output format: ``"DB_NAME ↔ PRED_NAME"`` so the user can paste it into
    the YAML as an alias entry.
    """
    h_match = pred_h.lower() == db_h_canon.lower()
    a_match = pred_a.lower() == db_a_canon.lower()
    if h_match and not a_match:
        return f'"{db_a_raw}" ↔ "{pred_a}"'
    if a_match and not h_match:
        return f'"{db_h_raw}" ↔ "{pred_h}"'
    if not h_match and not a_match:
        return f'home: "{db_h_raw}" ↔ "{pred_h}";  away: "{db_a_raw}" ↔ "{pred_a}"'
    return None  # both match → not actually an alias issue
