"""Ensemble + market-fused predictors.

Inspired by MatchOracle's 5-layer stacking architecture. We can't realistically
ship XGBoost/LightGBM/CatBoost meta-learners overnight, but two simpler
ensembles capture most of the gain:

  * **probability_average** — straight mean of the constituent probabilities.
    Sanity check: averaging Dixon-Coles, Bivariate Poisson, and our DC+Elo
    blend reduces variance from any single model's quirks.

  * **market_fused** — weighted blend with the bookmaker-implied probability
    (de-vigged via Shin). Mathematically: any positive weight on a perfectly
    efficient market guarantees you can't lose to that market. Empirically,
    market closing odds are highly efficient, so even a small fusion weight
    tames our model's over-confidence and shrinks ROI losses.

The constituents and weights are configurable. ``fit_constituent_models``
shares a single training pool across models so we don't pay 3x the fit cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

OUTCOMES = ("home_win", "draw", "away_win")


@dataclass(frozen=True)
class EnsembleConfig:
    members: tuple[str, ...] = (
        "dixon_coles_elo",
        "dixon_coles",
        "bivariate_poisson",
    )
    # Equal weights by default. The weights are normalized inside `predict`.
    weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    # 0.0 = pure model average. 0.5 = half model, half market. 1.0 = pure market.
    market_fusion_weight: float = 0.0


@dataclass
class FittedEnsemble:
    config: EnsembleConfig
    members: dict[str, Any]  # name → fitted model (heterogeneous)
    home_advantage: float
    max_goals: int
    training_rows: int


def fit(
    matches: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    config: EnsembleConfig | None = None,
    home_advantage: float = 0.22,
    max_goals: int = 8,
    optimizer_maxiter: int = 2500,
) -> FittedEnsemble:
    """Fit every constituent model on the same training pool."""
    config = config or EnsembleConfig()
    frame = pd.DataFrame(matches).copy()
    fitted: dict[str, Any] = {}
    for name in config.members:
        fitted[name] = _fit_member(
            name, frame,
            home_advantage=home_advantage,
            max_goals=max_goals,
            optimizer_maxiter=optimizer_maxiter,
        )
    return FittedEnsemble(
        config=config,
        members=fitted,
        home_advantage=home_advantage,
        max_goals=max_goals,
        training_rows=int(len(frame)),
    )


def predict_match(
    ensemble: FittedEnsemble,
    home_team: str,
    away_team: str,
    *,
    home_elo: float | None = None,
    away_elo: float | None = None,
    market_implied: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Predict W/D/L probabilities by weighted averaging the constituents."""
    probs_by_member: dict[str, np.ndarray] = {}
    contributions: list[dict[str, Any]] = []
    for name, model in ensemble.members.items():
        try:
            p = _predict_member(
                name, model,
                home_team=home_team, away_team=away_team,
                home_elo=home_elo, away_elo=away_elo,
                max_goals=ensemble.max_goals,
            )
        except (KeyError, ValueError):
            # If a model hasn't seen the team in training, skip it.
            continue
        probs_by_member[name] = p
        contributions.append({
            "name": name,
            "home_win": float(p[0]), "draw": float(p[1]), "away_win": float(p[2]),
        })

    if not probs_by_member:
        raise ValueError(
            f"No member of the ensemble could predict {home_team} vs {away_team}. "
            "Both teams must appear in the training pool of at least one model."
        )

    name_to_weight = dict(zip(ensemble.config.members, ensemble.config.weights))
    weighted = np.zeros(3, dtype=float)
    total_weight = 0.0
    for name, p in probs_by_member.items():
        w = name_to_weight.get(name, 1.0)
        weighted += w * p
        total_weight += w
    if total_weight == 0:
        raise ValueError("Ensemble weights sum to zero.")
    model_probs = weighted / total_weight

    fusion = ensemble.config.market_fusion_weight
    fused = model_probs.copy()
    if fusion > 0 and market_implied is not None:
        market_vec = np.array([
            market_implied.get("home_win", 0.0),
            market_implied.get("draw", 0.0),
            market_implied.get("away_win", 0.0),
        ], dtype=float)
        if market_vec.sum() > 0:
            market_vec = market_vec / market_vec.sum()
            fused = (1 - fusion) * model_probs + fusion * market_vec

    return {
        "probabilities": {
            "home_win": float(fused[0]),
            "draw": float(fused[1]),
            "away_win": float(fused[2]),
        },
        "model_probabilities": {
            "home_win": float(model_probs[0]),
            "draw": float(model_probs[1]),
            "away_win": float(model_probs[2]),
        },
        "contributions": contributions,
        "members_used": list(probs_by_member.keys()),
        "market_fusion_weight": fusion,
        "market_implied_used": fusion > 0 and market_implied is not None,
    }


def _fit_member(
    name: str,
    frame: pd.DataFrame,
    *,
    home_advantage: float,
    max_goals: int,
    optimizer_maxiter: int,
) -> Any:
    if name == "dixon_coles_elo":
        from models.dixon_coles import DixonColesConfig, DixonColesModel

        return DixonColesModel(
            DixonColesConfig(
                home_advantage=home_advantage,
                max_goals=max_goals,
                optimizer_maxiter=optimizer_maxiter,
            )
        ).fit(frame, as_of=frame["date"].max())

    # penaltyblog models share a constructor signature.
    import penaltyblog as pb

    pb_factories = {
        "dixon_coles": pb.models.DixonColesGoalModel,
        "bivariate_poisson": pb.models.BivariatePoissonGoalModel,
        "poisson": pb.models.PoissonGoalsModel,
    }
    factory = pb_factories.get(name)
    if factory is None:
        raise ValueError(f"Unknown ensemble member '{name}'.")

    instance = factory(
        goals_home=np.array(frame["home_goals"].values, dtype=np.int64),
        goals_away=np.array(frame["away_goals"].values, dtype=np.int64),
        teams_home=np.array(frame["home_team"].values, dtype=str),
        teams_away=np.array(frame["away_team"].values, dtype=str),
    )
    instance.fit()
    return instance


def _predict_member(
    name: str,
    model: Any,
    *,
    home_team: str,
    away_team: str,
    home_elo: float | None,
    away_elo: float | None,
    max_goals: int,
) -> np.ndarray:
    if name == "dixon_coles_elo":
        result = model.predict_match(
            home_team, away_team,
            home_elo=home_elo, away_elo=away_elo,
            max_goals=max_goals,
        )
        return np.array([result.home_win, result.draw, result.away_win], dtype=float)
    grid = model.predict(home_team, away_team, max_goals=max_goals)
    return np.array([grid.home_win, grid.draw, grid.away_win], dtype=float)
