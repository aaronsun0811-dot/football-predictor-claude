"""Reality-check the model: join predicted fixtures with what actually happened.

We've been logging every /upcoming snapshot to ``data/cache/upcoming/history.jsonl``
since round 6 (~"6 小时前 vs 现在" feature). Each line is
``{taken_at, date, league_key, home_team, away_team, home_win, draw, away_win}``.

If a fixture's date is in the past AND we have its actual result in the
``matches`` table, we can mark the prediction as resolved and compute:
  * was the model's top pick correct?
  * Brier / log-loss / RPS for this specific prediction
  * what was the prob it assigned to the actual outcome?

Aggregating those gives a **real-world accuracy** number, very different from
the synthetic backtest (which trains and predicts on the same closed dataset).
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from data.history_store import LEGACY_PATH, SHARD_DIR, iter_all_rows
from data.team_normalize import canonicalize


# Back-compat: external callers (and the test suite) patch ``HISTORY_PATH``
# to redirect reads. We still honor that — when patched, ``_read_history``
# reads from the patched path instead of going through the shard store.
HISTORY_PATH = LEGACY_PATH
OUTCOMES = ("home_win", "draw", "away_win")


def _outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def _read_history() -> list[dict[str, Any]]:
    """Read every prediction row across legacy file + month shards.

    If a caller (typically tests) has patched ``HISTORY_PATH`` to point at a
    custom file, we read ONLY that file. This preserves the original
    single-file semantics for tests that fixture up a tmp_path.
    """
    if HISTORY_PATH != LEGACY_PATH:
        # Patched — fall back to single-file mode for back-compat with tests.
        if not HISTORY_PATH.exists():
            return []
        rows: list[dict[str, Any]] = []
        with HISTORY_PATH.open() as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows
    # Production path: walk legacy file + every month shard.
    return list(iter_all_rows(legacy_path=LEGACY_PATH, shard_dir=SHARD_DIR))


def _earliest_prediction_per_fixture(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Keep the FIRST prediction per (date, home, away) — that's the "pre-match" call.

    Subsequent snapshots are noisier (line moves, model drift). The earliest
    one is what a user would have acted on.
    """
    first: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("date"), row.get("home_team"), row.get("away_team"))
        if any(v is None for v in key):
            continue
        existing = first.get(key)
        if existing is None or row.get("taken_at", "") < existing.get("taken_at", ""):
            first[key] = row
    return first


def resolve_predictions(
    matches_frame: pd.DataFrame,
    *,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Join history.jsonl predictions with actual results.

    Returns one row per RESOLVED prediction (fixture date in the past AND
    we have a result for it). Columns:
      league_key, match_date, home_team, away_team, actual_score,
      actual_outcome, predicted_outcome, p_actual, p_predicted_outcome,
      correct, brier, log_loss, rps, taken_at.
    """
    history = _read_history()
    if not history:
        return _empty_resolved()
    if matches_frame.empty:
        return _empty_resolved()
    as_of = as_of or date.today()

    # Build a lookup over the matches table. Use canonicalized names so
    # predicted-vs-actual aliasing works.
    m = matches_frame.copy()
    m["date_str"] = pd.to_datetime(m["date"]).dt.strftime("%Y-%m-%d")
    m["home_norm"] = m["home_team"].astype(str).map(canonicalize)
    m["away_norm"] = m["away_team"].astype(str).map(canonicalize)
    result_lookup: dict[tuple[str, str, str], tuple[int, int]] = {}
    for row in m.itertuples(index=False):
        result_lookup[(row.date_str, row.home_norm, row.away_norm)] = (
            int(row.home_goals), int(row.away_goals),
        )

    earliest = _earliest_prediction_per_fixture(history)
    resolved = []
    for (date_str, home, away), pred in earliest.items():
        # Only audit completed matches (strict <today; today's games still live).
        try:
            fx_date = datetime.fromisoformat(str(date_str)).date()
        except ValueError:
            continue
        if fx_date >= as_of:
            continue

        canon_home = canonicalize(home) or home
        canon_away = canonicalize(away) or away
        actual = result_lookup.get((date_str, canon_home, canon_away))
        if actual is None:
            # No result in our DB → can't audit yet.
            continue

        hg, ag = actual
        actual_outcome = _outcome(hg, ag)
        probs = np.array([
            pred.get("home_win", 0) or 0,
            pred.get("draw", 0) or 0,
            pred.get("away_win", 0) or 0,
        ], dtype=float)
        if probs.sum() <= 0:
            continue
        probs = probs / probs.sum()

        predicted_idx = int(np.argmax(probs))
        predicted_outcome = OUTCOMES[predicted_idx]
        actual_idx = OUTCOMES.index(actual_outcome)
        p_actual = float(probs[actual_idx])

        one_hot = np.zeros(3)
        one_hot[actual_idx] = 1.0
        brier = float(np.sum((probs - one_hot) ** 2))
        log_loss = float(-np.log(max(p_actual, 1e-12)))
        cum_p = np.cumsum(probs)
        cum_a = np.cumsum(one_hot)
        rps = float(np.sum((cum_p[:-1] - cum_a[:-1]) ** 2) / 2.0)

        # Score-level audit: only available for snapshots taken AFTER round 18
        # (when we started persisting predicted scoreline). Older rows are
        # marked exact_score_correct=None and excluded from the score accuracy
        # rollup so we don't penalise the model for our own history-schema gap.
        pred_h = pred.get("predicted_home_goals")
        pred_a = pred.get("predicted_away_goals")
        if pred_h is not None and pred_a is not None:
            pred_h_int = int(pred_h)
            pred_a_int = int(pred_a)
            predicted_score = f"{pred_h_int}-{pred_a_int}"
            exact_score_correct = (pred_h_int == hg and pred_a_int == ag)
            # L1 distance in goals — "we said 2-1, actual 1-1" → distance 1.
            goal_distance = abs(pred_h_int - hg) + abs(pred_a_int - ag)
            predicted_score_prob = pred.get("predicted_score_prob")
        else:
            predicted_score = None
            exact_score_correct = None
            goal_distance = None
            predicted_score_prob = None

        # Date-check status persisted at /upcoming time. A "warning" here
        # means the prediction was made for a date the data source itself
        # was unsure about — useful context when judging the model's accuracy.
        date_check_status = pred.get("date_check_status")
        had_date_warning = bool(date_check_status == "warning")

        resolved.append({
            "league_key": pred.get("league_key"),
            "match_date": str(fx_date),
            "home_team": canon_home,
            "away_team": canon_away,
            "actual_score": f"{hg}-{ag}",
            "actual_outcome": actual_outcome,
            "predicted_outcome": predicted_outcome,
            "predicted_score": predicted_score,
            "predicted_score_prob": predicted_score_prob,
            "exact_score_correct": exact_score_correct,
            "goal_distance": goal_distance,
            "p_actual": p_actual,
            "p_predicted_outcome": float(probs[predicted_idx]),
            "correct": predicted_outcome == actual_outcome,
            "brier": brier,
            "log_loss": log_loss,
            "rps": rps,
            "taken_at": pred.get("taken_at"),
            "date_check_status": date_check_status,
            "had_date_warning": had_date_warning,
        })

    if not resolved:
        return _empty_resolved()
    return pd.DataFrame(resolved).sort_values("match_date", ascending=False).reset_index(drop=True)


def _empty_resolved() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "league_key", "match_date", "home_team", "away_team", "actual_score",
        "actual_outcome", "predicted_outcome", "predicted_score",
        "predicted_score_prob", "exact_score_correct", "goal_distance",
        "p_actual", "p_predicted_outcome",
        "correct", "brier", "log_loss", "rps", "taken_at",
        "date_check_status", "had_date_warning",
    ])


def summarize_resolved(resolved: pd.DataFrame) -> dict[str, Any]:
    """Aggregate accuracy / Brier / RPS / score-level metrics over resolved predictions."""
    if resolved.empty:
        return {
            "n_resolved": 0, "accuracy": None, "brier": None,
            "log_loss": None, "rps": None,
            "n_scored": 0, "exact_score_accuracy": None, "mean_goal_distance": None,
            "by_league": [],
        }
    summary = {
        "n_resolved": int(len(resolved)),
        "accuracy": float(resolved["correct"].mean()),
        "brier": float(resolved["brier"].mean()),
        "log_loss": float(resolved["log_loss"].mean()),
        "rps": float(resolved["rps"].mean()),
    }

    # Score-level rollup: only over predictions that actually carry a stored
    # predicted scoreline. Old snapshots (pre-round-18) have None here.
    scored = resolved[resolved["exact_score_correct"].notna()] if "exact_score_correct" in resolved.columns else resolved.iloc[0:0]
    summary["n_scored"] = int(len(scored))
    if not scored.empty:
        summary["exact_score_accuracy"] = float(scored["exact_score_correct"].mean())
        summary["mean_goal_distance"] = float(scored["goal_distance"].mean())
    else:
        summary["exact_score_accuracy"] = None
        summary["mean_goal_distance"] = None

    # Date-warning rollup: predictions made when fd.org disagreed with TSDB
    # on the fixture date. These shouldn't count toward "the model is bad" —
    # the data source was already flagged as uncertain. Show alongside the
    # honest accuracy number so the user can mentally discount it.
    if "had_date_warning" in resolved.columns:
        summary["n_with_date_warning"] = int(resolved["had_date_warning"].sum())
    else:
        summary["n_with_date_warning"] = 0
    # Per-league breakdown — only show leagues with >=3 resolved predictions.
    # Matched to /backtest's ``summary.by_league`` (R29) in shape, so the UI
    # can render both panels with the same columns and the user can directly
    # compare "real /upcoming accuracy" vs "synthetic walk-forward accuracy"
    # per league. The kicker: when audit ≪ backtest for a league, you've found
    # genuine data drift (model fit a closed historical dataset but doesn't
    # generalize to today's live fixtures).
    by_league = []
    for league, sub in resolved.groupby("league_key"):
        if len(sub) < 3:
            continue
        entry: dict[str, Any] = {
            "league_key": str(league),
            "n": int(len(sub)),
            "accuracy": float(sub["correct"].mean()),
            "brier": float(sub["brier"].mean()),
            "rps": float(sub["rps"].mean()),
        }
        # Score-level rollup, defensive against legacy rows. Same logic as
        # the global score-metrics block above.
        if "exact_score_correct" in sub.columns:
            scored_sub = sub[sub["exact_score_correct"].notna()]
        else:
            scored_sub = sub.iloc[0:0]
        if not scored_sub.empty:
            entry["n_scored"] = int(len(scored_sub))
            entry["exact_score_accuracy"] = float(
                scored_sub["exact_score_correct"].astype(bool).mean()
            )
            entry["mean_goal_distance"] = float(scored_sub["goal_distance"].mean())
        else:
            entry["n_scored"] = 0
            entry["exact_score_accuracy"] = None
            entry["mean_goal_distance"] = None
        by_league.append(entry)
    by_league.sort(key=lambda r: -r["n"])
    summary["by_league"] = by_league
    return summary


def audit_summary(matches_frame: pd.DataFrame) -> dict[str, Any]:
    """High-level: resolve everything we can + return summary + sample bets."""
    resolved = resolve_predictions(matches_frame)
    summary = summarize_resolved(resolved)
    # 5 most-confident-correct + 5 most-surprising (lowest p_actual).
    # Both lists now also carry the predicted scoreline so the UI can show
    # "we said 2-1, actual was 1-1" at a glance.
    sample_cols_base = [
        "match_date", "league_key", "home_team", "away_team",
        "actual_score", "predicted_score", "predicted_outcome",
        "exact_score_correct", "goal_distance",
        "had_date_warning",
    ]
    if not resolved.empty:
        best_cols = sample_cols_base + ["p_predicted_outcome"]
        summary["best_calls"] = (
            resolved[resolved["correct"]]
            .sort_values("p_predicted_outcome", ascending=False)
            .head(5)
            [best_cols]
            .to_dict(orient="records")
        )
        worst_cols = sample_cols_base + ["p_actual"]
        summary["worst_misses"] = (
            resolved.sort_values("p_actual")
            .head(5)
            [worst_cols]
            .to_dict(orient="records")
        )
        # NEW: 5 most surprising EXACT-SCORE hits — when we actually nailed it,
        # which scorelines did we get right? Useful sanity check.
        scored = resolved[resolved["exact_score_correct"] == True]  # noqa: E712
        if not scored.empty:
            summary["exact_score_hits"] = (
                scored.sort_values("predicted_score_prob", ascending=True)  # least-likely-but-still-right first
                .head(5)
                [sample_cols_base + ["predicted_score_prob"]]
                .to_dict(orient="records")
            )
        else:
            summary["exact_score_hits"] = []
    else:
        summary["best_calls"] = []
        summary["worst_misses"] = []
        summary["exact_score_hits"] = []
    return summary
