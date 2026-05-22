"""Append-only history of /upcoming predictions, for trend-vs-N-hours-ago.

Each time the /upcoming endpoint computes a fresh batch, we append every
fixture+probability+timestamp to the prediction-history log.

When we later serve /upcoming, we look up the snapshot closest to N hours
ago for the same (date, home, away) tuple and compute the percentage-point
delta. The frontend renders this as a small "↑ +3pp" / "↓ −2pp" chip.

Storage layout is owned by ``data/history_store.py`` (month-sharded JSONL).
This module is just the prediction-flow adapter on top of it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from data.history_store import (
    LEGACY_PATH,
    SHARD_DIR,
    append_rows,
    iter_all_rows,
    iter_recent_rows,
)

# Re-export for back-compat. Tests and external scripts that imported
# ``HISTORY_PATH`` from this module will still see the legacy single-file path
# (where the old data lives). All new writes go through ``append_rows`` and
# land in the current-month shard inside ``SHARD_DIR``.
HISTORY_PATH = LEGACY_PATH


def append_snapshot(fixtures: Iterable[dict[str, Any]], *, taken_at: datetime | None = None) -> int:
    """Append one row per fixture to the current-month shard. Returns rows written.

    Each row includes:
      * The outcome distribution (``home_win/draw/away_win``)
      * The single most-likely scoreline (``predicted_home_goals/predicted_away_goals/predicted_score_prob``)
        — captured at prediction time so the audit module can compare it against
        the actual final score weeks later, even if the model gets retrained in
        between. Without this, post-hoc "几比几" comparison would be impossible.
      * The ``date_check`` status (if present on the fixture). Persisting this
        means the audit module can later tell whether a stale/wrong fixture
        date is to blame for an unresolved prediction — see Round 25/26 for
        the upstream date-mismatch detection. Without persisting, the model
        gets blamed for the data source's date-quality bug.

    Skips fixtures without a 'prediction.probabilities' block.
    """
    taken_at = taken_at or datetime.now(timezone.utc)
    ts = taken_at.isoformat()
    rows = []
    for fx in fixtures:
        pred = fx.get("prediction") or {}
        probs = pred.get("probabilities")
        if not probs:
            continue
        # most_likely_scores is sorted by probability descending. The first
        # entry is the argmax of the score matrix — i.e. "the single scoreline
        # the model would commit to if forced to pick one."
        top_score = (pred.get("most_likely_scores") or [None])[0]
        row = {
            "taken_at": ts,
            "date": fx.get("date"),
            "league_key": fx.get("league_key"),
            "home_team": fx.get("home_team"),
            "away_team": fx.get("away_team"),
            "home_win": probs.get("home_win"),
            "draw": probs.get("draw"),
            "away_win": probs.get("away_win"),
        }
        if top_score:
            row["predicted_home_goals"] = top_score.get("home_goals")
            row["predicted_away_goals"] = top_score.get("away_goals")
            row["predicted_score_prob"] = top_score.get("probability")
        # Persist the date-check status if /upcoming attached one. Only the
        # ``status`` + ``fdorg_date`` + ``days_off`` fields are kept — the
        # downstream audit doesn't need the full block.
        dc = fx.get("date_check") or {}
        if dc.get("status") in ("warning", "confirmed", "unknown", "not_covered"):
            row["date_check_status"] = dc["status"]
            if dc.get("status") == "warning":
                row["date_check_fdorg_date"] = dc.get("fdorg_date")
                row["date_check_days_off"] = dc.get("days_off")
        rows.append(row)
    return append_rows(rows, taken_at=taken_at)


def lookup_deltas(
    fixtures: Iterable[dict[str, Any]],
    *,
    hours_ago: float = 3.0,
    tolerance_hours: float = 1.5,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """For each current fixture, find the snapshot closest to ``hours_ago`` ago.

    Returns ``{(date, home, away) → {past_probabilities, delta, taken_at, age_hours}}``.
    A fixture with no usable historical snapshot is omitted from the returned dict.
    The "tolerance" defines how loose we are: a 6h-ago lookup with tol=2 will
    accept any snapshot in [4h, 8h] old.
    """
    now = datetime.now(timezone.utc)
    target = now - timedelta(hours=hours_ago)
    min_ts = target - timedelta(hours=tolerance_hours)
    max_ts = target + timedelta(hours=tolerance_hours)
    fixture_keys = {
        (fx.get("date"), fx.get("home_team"), fx.get("away_team"))
        for fx in fixtures
    }
    if not fixture_keys:
        return {}

    # Hot path: only scan shards from the month containing ``min_ts`` onward.
    # Falls back to the legacy file too (`iter_recent_rows` always includes it).
    best: dict[tuple[str, str, str], tuple[float, dict[str, Any]]] = {}
    for row in iter_recent_rows(since=min_ts):
        key = (row.get("date"), row.get("home_team"), row.get("away_team"))
        if key not in fixture_keys:
            continue
        try:
            ts = datetime.fromisoformat(row["taken_at"])
        except (KeyError, ValueError):
            continue
        if not (min_ts <= ts <= max_ts):
            continue
        distance = abs((ts - target).total_seconds())
        prev = best.get(key)
        if prev is None or distance < prev[0]:
            best[key] = (distance, row)

    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, (_, row) in best.items():
        ts = datetime.fromisoformat(row["taken_at"])
        result[key] = {
            "past_probabilities": {
                "home_win": row.get("home_win"),
                "draw": row.get("draw"),
                "away_win": row.get("away_win"),
            },
            "taken_at": row["taken_at"],
            "age_hours": round((now - ts).total_seconds() / 3600, 2),
        }
    return result


def attach_deltas(
    fixtures: list[dict[str, Any]],
    *,
    hours_ago: float = 3.0,
    tolerance_hours: float = 1.5,
) -> None:
    """Mutate fixtures in place, attaching `delta_vs_3h` blocks where available."""
    deltas = lookup_deltas(fixtures, hours_ago=hours_ago, tolerance_hours=tolerance_hours)
    for fx in fixtures:
        key = (fx.get("date"), fx.get("home_team"), fx.get("away_team"))
        ref = deltas.get(key)
        pred = fx.get("prediction") or {}
        probs = pred.get("probabilities")
        if not ref or not probs:
            pred["delta_vs_3h"] = None
            continue
        past = ref["past_probabilities"]
        delta = {
            outcome: (probs.get(outcome) or 0) - (past.get(outcome) or 0)
            for outcome in ("home_win", "draw", "away_win")
        }
        pred["delta_vs_3h"] = {
            "delta": delta,                  # percentage-POINT deltas (e.g. +0.03 = +3pp)
            "past_probabilities": past,
            "taken_at": ref["taken_at"],
            "age_hours": ref["age_hours"],
        }


def history_size() -> int:
    """Total bytes across legacy file + all shards. For dashboard display."""
    total = 0
    if LEGACY_PATH.exists():
        total += LEGACY_PATH.stat().st_size
    if SHARD_DIR.exists():
        for path in SHARD_DIR.iterdir():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total
