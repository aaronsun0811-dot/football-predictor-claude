"""Wrappers around penaltyblog's goal models for use in the prediction pipeline.

Why bother: our home-grown ``DixonColesModel`` blends an Elo prior into the
goal-rate prediction. On EPL, that pushes Arsenal vs Chelsea (Elo gap ~210)
to **67% / 22% / 11%** while penaltyblog's pure Dixon-Coles on the same data
says **47% / 25% / 28%**. Market closing odds tend to land near the
penaltyblog number — i.e. our Elo correction is over-confident, which is
the root cause of the ROI sim losing money.

Exposing penaltyblog's models gives the user (and the ROI sim) a sharper
baseline to compare against.

Models exposed:
  * ``dixon_coles``         — penaltyblog.models.DixonColesGoalModel
  * ``bivariate_poisson``   — Karlis & Ntzoufras correlated bivariate Poisson
  * ``poisson``             — vanilla independent Poisson (baseline)

All return a uniform ``PenaltyBlogPrediction`` with the W/D/L probabilities
and an expected-goals tuple.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import penaltyblog as pb

REQUIRED_COLUMNS = {"home_team", "away_team", "home_goals", "away_goals"}

MODEL_FACTORIES: dict[str, Any] = {
    "dixon_coles": pb.models.DixonColesGoalModel,
    "bivariate_poisson": pb.models.BivariatePoissonGoalModel,
    "poisson": pb.models.PoissonGoalsModel,
    "negative_binomial": pb.models.NegativeBinomialGoalModel,
    "zero_inflated_poisson": pb.models.ZeroInflatedPoissonGoalsModel,
}


@dataclass(frozen=True)
class PenaltyBlogPrediction:
    home_team: str
    away_team: str
    home_win: float
    draw: float
    away_win: float
    expected_home_goals: float | None
    expected_away_goals: float | None
    score_matrix: list[list[float]]
    most_likely_scores: list[dict[str, float | int]]

    def to_dict(self, *, neutral_site: bool = False, knockout: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "probabilities": {
                "home_win": self.home_win,
                "draw": self.draw,
                "away_win": self.away_win,
            },
            "expected_goals": {
                "home": self.expected_home_goals,
                "away": self.expected_away_goals,
            },
            "score_matrix": self.score_matrix,
            "most_likely_scores": self.most_likely_scores,
            "neutral_site": neutral_site,
            "knockout": knockout,
        }
        if knockout:
            # Equivalent of our knockout split.
            home_advance = self.home_win + self.draw * 0.5
            away_advance = self.away_win + self.draw * 0.5
            out["advancement_probabilities"] = {"home": home_advance, "away": away_advance}
        return out


def fit_and_predict(
    matches: pd.DataFrame | Sequence[Mapping[str, Any]],
    home_team: str,
    away_team: str,
    *,
    model: str = "dixon_coles",
    max_goals: int = 8,
    time_decay_xi: float | None = None,
) -> PenaltyBlogPrediction:
    """Fit ``model`` on ``matches`` and predict ``home_team`` vs ``away_team``.

    ``time_decay_xi`` enables Dixon-Coles-style time weighting (default off).
    A typical value is ``0.0019`` (Dixon-Coles' original suggestion).
    """
    if model not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{model}'. Pick from {sorted(MODEL_FACTORIES)}.")
    frame = _prepare(matches)

    weights = None
    if time_decay_xi:
        weights = pb.models.dixon_coles_weights(frame["date"], xi=time_decay_xi)

    factory = MODEL_FACTORIES[model]
    instance = factory(
        goals_home=np.array(frame["home_goals"].values, dtype=np.int64),
        goals_away=np.array(frame["away_goals"].values, dtype=np.int64),
        teams_home=np.array(frame["home_team"].values, dtype=str),
        teams_away=np.array(frame["away_team"].values, dtype=str),
        weights=weights,
    )
    instance.fit()

    grid = instance.predict(home_team, away_team, max_goals=max_goals)
    matrix = np.asarray(grid.grid)
    matrix = matrix / matrix.sum()

    home_win = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())

    # Expected goals from the marginal distributions of the score matrix.
    home_marginal = matrix.sum(axis=1)
    away_marginal = matrix.sum(axis=0)
    goals = np.arange(matrix.shape[0])
    expected_home = float((goals * home_marginal).sum())
    expected_away = float((goals * away_marginal).sum())

    most_likely = _top_scores(matrix, n=5)

    return PenaltyBlogPrediction(
        home_team=home_team,
        away_team=away_team,
        home_win=home_win,
        draw=draw,
        away_win=away_win,
        expected_home_goals=expected_home,
        expected_away_goals=expected_away,
        score_matrix=matrix.tolist(),
        most_likely_scores=most_likely,
    )


def _prepare(matches: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(matches).copy()
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.dropna(subset=list(REQUIRED_COLUMNS))
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date").reset_index(drop=True)
    frame["home_goals"] = frame["home_goals"].astype(int)
    frame["away_goals"] = frame["away_goals"].astype(int)
    frame["home_team"] = frame["home_team"].astype(str)
    frame["away_team"] = frame["away_team"].astype(str)
    return frame


def _top_scores(matrix: np.ndarray, *, n: int = 5) -> list[dict[str, float | int]]:
    flat = np.argsort(matrix.ravel())[::-1][:n]
    out = []
    for idx in flat:
        h, a = np.unravel_index(idx, matrix.shape)
        out.append({
            "home_goals": int(h),
            "away_goals": int(a),
            "probability": float(matrix[h, a]),
        })
    return out
