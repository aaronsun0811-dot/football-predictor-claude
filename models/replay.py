"""Per-match replay — "what did the model predict for this game, before it
happened, and what actually happened?"

Different from the aggregate backtest: pick one specific historical match
and rewind the model to the day before. The fit uses only matches strictly
before the target's date, so the prediction is leakage-free.

Use cases:
  * Browse the season — "Liverpool 4-3 Newcastle — did the model see it coming?"
  * Find the model's most-surprising calls (it said 5% and that's what
    actually happened) and most-confident hits (90% and they happened)
  * Validate a single result vs the headline accuracy number
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from models.dixon_coles import DixonColesConfig, DixonColesModel


OUTCOMES = ("home_win", "draw", "away_win")


@dataclass(frozen=True)
class MatchReplay:
    match_date: date
    league_key: str
    home_team: str
    away_team: str
    actual_home_goals: int
    actual_away_goals: int
    actual_outcome: str  # home_win / draw / away_win

    # Model predictions made on the cutoff date
    predicted_home_win: float
    predicted_draw: float
    predicted_away_win: float
    predicted_outcome: str  # the model's most-likely outcome
    correct: bool

    # Per-match "surprise" — log-loss of just this prediction. Higher = worse.
    log_loss: float
    # Ranked probability score for this single match (0..1; lower = better).
    rps: float
    # Brier on this single match.
    brier: float

    training_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_date": str(self.match_date),
            "league_key": self.league_key,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "actual": {
                "home_goals": self.actual_home_goals,
                "away_goals": self.actual_away_goals,
                "outcome": self.actual_outcome,
            },
            "prediction": {
                "home_win": self.predicted_home_win,
                "draw": self.predicted_draw,
                "away_win": self.predicted_away_win,
                "most_likely": self.predicted_outcome,
                "correct": self.correct,
            },
            "scores": {
                "log_loss": self.log_loss,
                "rps": self.rps,
                "brier": self.brier,
            },
            "training_rows": self.training_rows,
        }


def replay_match(
    *,
    target_date: date | pd.Timestamp,
    home_team: str,
    away_team: str,
    league_matches: pd.DataFrame,
    lookback_days: int = 730,
    home_advantage: float = 0.22,
    max_goals: int = 8,
    optimizer_maxiter: int = 2500,
) -> MatchReplay:
    """Refit Dixon-Coles up to (but not including) ``target_date`` and predict.

    ``league_matches`` should be the league's complete match frame; the
    function slices it internally to enforce no leakage.
    """
    target_ts = pd.Timestamp(target_date)
    frame = league_matches.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    # Locate the actual match in the data so we can compare.
    actual_row = frame[
        (frame["date"] == target_ts) &
        (frame["home_team"] == home_team) &
        (frame["away_team"] == away_team)
    ]
    if actual_row.empty:
        raise KeyError(
            f"No match found: {home_team} vs {away_team} on {target_ts.date()}. "
            "Check team-name spelling against /teams output."
        )
    actual = actual_row.iloc[0]
    actual_outcome = _outcome(int(actual["home_goals"]), int(actual["away_goals"]))

    # Train on everything strictly before the match date.
    train = frame[frame["date"] < target_ts].copy()
    if len(train) < 50:
        raise ValueError(
            f"Only {len(train)} matches available before {target_ts.date()}. "
            "Need at least 50 for a stable fit."
        )

    model = DixonColesModel(
        DixonColesConfig(
            home_advantage=home_advantage,
            max_goals=max_goals,
            optimizer_maxiter=optimizer_maxiter,
            lookback_days=lookback_days,
        )
    ).fit(train, as_of=train["date"].max())

    prediction = model.predict_match(home_team, away_team, max_goals=max_goals)

    probs = {
        "home_win": prediction.home_win,
        "draw": prediction.draw,
        "away_win": prediction.away_win,
    }
    most_likely = max(OUTCOMES, key=lambda o: probs[o])

    actual_idx = OUTCOMES.index(actual_outcome)
    p_actual = max(min(probs[actual_outcome], 1.0 - 1e-12), 1e-12)
    log_loss = float(-np.log(p_actual))
    brier = float(sum((probs[o] - (1 if o == actual_outcome else 0)) ** 2 for o in OUTCOMES))

    # RPS for one observation, 3 ordered outcomes:
    cum_p = np.cumsum([probs[o] for o in OUTCOMES])
    cum_a = np.cumsum([1 if i == actual_idx else 0 for i in range(3)])
    rps = float(np.sum((cum_p[:-1] - cum_a[:-1]) ** 2) / (3 - 1))

    return MatchReplay(
        match_date=actual["date"].date(),
        league_key=str(actual.get("league_key", "")),
        home_team=home_team,
        away_team=away_team,
        actual_home_goals=int(actual["home_goals"]),
        actual_away_goals=int(actual["away_goals"]),
        actual_outcome=actual_outcome,
        predicted_home_win=probs["home_win"],
        predicted_draw=probs["draw"],
        predicted_away_win=probs["away_win"],
        predicted_outcome=most_likely,
        correct=(most_likely == actual_outcome),
        log_loss=log_loss,
        rps=rps,
        brier=brier,
        training_rows=int(model.training_rows_),
    )


def rank_surprises(
    *,
    league_matches: pd.DataFrame,
    league_key: str,
    sample_size: int = 50,
    refit_every: int = 25,
    lookback_days: int = 730,
    optimizer_maxiter: int = 2500,
    min_train_matches: int = 200,
) -> dict[str, list[dict[str, Any]]]:
    """Compute per-match surprise scores across a league's history.

    Returns two ranked lists:
      * "biggest_upsets" — actual happened but model gave it tiny probability
      * "best_calls"     — model said 70%+ on the actual outcome AND was correct

    To avoid blowing CPU we refit only every ``refit_every`` matches; predict
    the intermediate matches using the last fit.
    """
    frame = league_matches.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    if len(frame) < min_train_matches + 50:
        return {"biggest_upsets": [], "best_calls": []}

    rows: list[dict[str, Any]] = []
    model: DixonColesModel | None = None
    last_fit_idx = -1

    for idx in range(min_train_matches, len(frame)):
        if model is None or idx - last_fit_idx >= refit_every:
            train = frame.iloc[:idx].copy()
            try:
                model = DixonColesModel(
                    DixonColesConfig(
                        max_goals=8,
                        optimizer_maxiter=optimizer_maxiter,
                        lookback_days=lookback_days,
                    )
                ).fit(train, as_of=train["date"].max())
                last_fit_idx = idx
            except RuntimeError:
                if model is None:
                    continue
                continue

        target = frame.iloc[idx]
        try:
            pred = model.predict_match(
                str(target["home_team"]),
                str(target["away_team"]),
                max_goals=8,
            )
        except (KeyError, ValueError):
            continue

        actual_outcome = _outcome(int(target["home_goals"]), int(target["away_goals"]))
        probs = {
            "home_win": pred.home_win,
            "draw": pred.draw,
            "away_win": pred.away_win,
        }
        most_likely = max(OUTCOMES, key=lambda o: probs[o])
        p_actual = probs[actual_outcome]
        confidence = max(probs.values())

        rows.append({
            "match_date": str(target["date"].date()),
            "league_key": league_key,
            "home_team": str(target["home_team"]),
            "away_team": str(target["away_team"]),
            "actual_score": f"{int(target['home_goals'])}-{int(target['away_goals'])}",
            "actual_outcome": actual_outcome,
            "probabilities": probs,
            "predicted_outcome": most_likely,
            "p_actual": p_actual,
            "p_model_pick": confidence,
            "correct": most_likely == actual_outcome,
        })

    if not rows:
        return {"biggest_upsets": [], "best_calls": []}

    # Biggest upsets: actual outcome got lowest predicted probability.
    upsets = sorted(rows, key=lambda r: r["p_actual"])[:15]
    # Best calls: most confident correct calls (high p on actual outcome).
    correct = [r for r in rows if r["correct"]]
    best = sorted(correct, key=lambda r: -r["p_actual"])[:15]

    return {
        "biggest_upsets": upsets,
        "best_calls": best,
        "total_evaluated": len(rows),
    }


def _outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"
