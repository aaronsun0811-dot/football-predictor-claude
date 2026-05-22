"""Compare audit (real /upcoming → actual results) vs backtest (synthetic
walk-forward on historical data) per league.

When the same model predicts the same league two ways — once on live future
fixtures and once on the closed historical sample — divergence between the
two is a measurable, interpretable signal:

  * ``Δaccuracy > 0`` (backtest beats audit) — likely **data drift**. The
    model fit a historical distribution that doesn't generalize to today's
    live fixtures. Examples: rule changes (handball, VAR), promoted teams
    not yet well-modelled, mid-season manager changes.
  * ``Δaccuracy < 0`` (audit beats backtest) — usually small-sample
    noise. Audit has tens of resolved predictions, backtest has thousands.
    Treat with caution and wait for more audit data.
  * ``Δaccuracy ≈ 0`` — the model generalizes; backtest is a useful proxy.

This module just produces the comparison rows. The UI decides how to render.

Cost: one backtest per league = 5-30 seconds depending on league size. The
endpoint that wraps this caches the result for 24h on disk; matches table
only changes once a day from the backfill cron anyway.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from data.database import Database
from models.backtest import BacktestConfig, backtest_dixon_coles


def compare_audit_to_backtest(
    db: Database,
    audit_by_league: list[dict[str, Any]],
    *,
    backtest_min_train: int = 200,
    backtest_refit_every: int = 50,
    max_leagues: int = 12,
) -> dict[str, Any]:
    """Run one quick backtest per league in ``audit_by_league`` and align the
    results.

    Returns:
        ``{
          "n_leagues_compared": int,
          "n_leagues_skipped": int,
          "rows": [
            {"league_key", "audit": {...}, "backtest": {...},
             "delta": {"accuracy", "brier", "rps", "exact_score_accuracy"}}
          ],
          "skipped": [{"league_key", "reason"}, ...],
        }``

    A league shows up in ``rows`` only when BOTH sides have data. Anything
    that can't be backtested (not enough matches, fit failed entirely, etc.)
    goes in ``skipped`` so the user sees the gap rather than wondering why
    their league is missing.
    """
    if not audit_by_league:
        return {"n_leagues_compared": 0, "n_leagues_skipped": 0, "rows": [], "skipped": []}

    # Sort audit rows by sample size desc and cap to ``max_leagues`` — running
    # 30 separate backtests would be a multi-minute request.
    audit_sorted = sorted(audit_by_league, key=lambda r: -r.get("n", 0))[:max_leagues]

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for audit_row in audit_sorted:
        league_key = audit_row.get("league_key")
        if not league_key:
            continue

        bt_summary = _run_single_league_backtest(
            db, league_key,
            min_train_matches=backtest_min_train,
            refit_every=backtest_refit_every,
        )
        if bt_summary is None:
            skipped.append({"league_key": league_key, "reason": "backtest_unavailable"})
            continue

        rows.append({
            "league_key": league_key,
            "audit": _trim_audit_row(audit_row),
            "backtest": _trim_backtest_summary(bt_summary),
            "delta": _delta(audit_row, bt_summary),
        })

    return {
        "n_leagues_compared": len(rows),
        "n_leagues_skipped": len(skipped),
        "rows": rows,
        "skipped": skipped,
    }


def _run_single_league_backtest(
    db: Database,
    league_key: str,
    *,
    min_train_matches: int,
    refit_every: int,
) -> dict[str, Any] | None:
    """Walk-forward one league. Returns the backtest summary dict, or ``None``
    if the league simply can't be backtested (too few matches, fit failure,
    DB read error, etc.).

    Failures here are non-fatal at the caller's level — they just mean this
    one league doesn't make it into the comparison. The blanket ``Exception``
    catch covers the fetch + fit + summary pipeline; we'd rather skip one
    league than crash the whole comparison endpoint.
    """
    try:
        matches = db.fetch_matches(league_key=league_key)
        if matches is None or matches.empty or len(matches) <= min_train_matches + 5:
            return None
        result = backtest_dixon_coles(
            matches,
            config=BacktestConfig(
                min_train_matches=min_train_matches,
                refit_every=refit_every,
            ),
        )
        return result.summary
    except Exception:  # noqa: BLE001 — soft fail per league, on purpose
        return None


# ---------------------------------------------------------------------------
# Per-side trimming + delta calculation
# ---------------------------------------------------------------------------

def _trim_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    """Just the fields the UI table actually shows."""
    return {
        "n": int(row.get("n", 0)),
        "accuracy": _maybe_float(row.get("accuracy")),
        "brier": _maybe_float(row.get("brier")),
        "rps": _maybe_float(row.get("rps")),
        "exact_score_accuracy": _maybe_float(row.get("exact_score_accuracy")),
        "mean_goal_distance": _maybe_float(row.get("mean_goal_distance")),
    }


def _trim_backtest_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Backtest's headline numbers in the same shape as the audit row."""
    return {
        "n": int(summary.get("n_predictions", 0)),
        "accuracy": _maybe_float(summary.get("accuracy")),
        "brier": _maybe_float(summary.get("brier_score")),
        "rps": _maybe_float(summary.get("rps")),
        "exact_score_accuracy": _maybe_float(summary.get("exact_score_accuracy")),
        "mean_goal_distance": _maybe_float(summary.get("mean_goal_distance")),
    }


def _delta(audit_row: dict[str, Any], bt_summary: dict[str, Any]) -> dict[str, Any]:
    """``backtest - audit`` per metric. Positive ``Δaccuracy`` means backtest
    is doing better, which we treat as the data-drift signal.

    For ``brier`` and ``rps`` lower is better, so positive Δ means backtest
    is WORSE — keep the convention ``Δ = backtest - audit`` and let the UI
    interpret directionality.
    """
    def diff(a, b):
        a_val = audit_row.get(a) if a is not None else None
        b_val = bt_summary.get(b) if b is not None else None
        if a_val is None or b_val is None:
            return None
        try:
            return float(b_val) - float(a_val)
        except (TypeError, ValueError):
            return None
    return {
        # backtest summary uses "brier_score", audit uses "brier"
        "accuracy": diff("accuracy", "accuracy"),
        "brier": diff("brier", "brier_score"),
        "rps": diff("rps", "rps"),
        "exact_score_accuracy": diff("exact_score_accuracy", "exact_score_accuracy"),
    }


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN guard
        return None
    return result
