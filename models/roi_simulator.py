"""ROI simulator: walk-forward value-betting against Bet365 closing odds.

The Value Finder tab shows EV for a single match. This module asks the harder
question: **if you'd actually placed those bets historically, what would your
bankroll look like?**

Method:
  1. Walk-forward fit Dixon-Coles (same as the backtest).
  2. For each match, compute model probabilities and look up Bet365 closing
     odds from the ``match_odds`` table.
  3. Apply the value-betting filter (edge > min_edge AND EV > min_ev).
  4. Stake a fixed fraction of bankroll using fractional Kelly.
  5. Track bankroll over time.

The results frame has one row per bet placed — date, outcome bet on, stake,
odds, hit/miss, profit. The summary aggregates: total bets, hit rate,
total return, ROI, Sharpe ratio, max drawdown.

Notes on the math:
  - "Edge" = model_prob − overround-normalized_implied_prob.
  - "EV per unit stake" = p * (odds - 1) - (1 - p).
  - Kelly fraction = max(0, (p * odds - 1) / (odds - 1)).
  - Fractional Kelly multiplier (default 0.5) makes the strategy more robust
    when model probabilities are themselves noisy.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from models.dixon_coles import DixonColesConfig, DixonColesModel
from models.elo import EloConfig, attach_pre_match_elos


OUTCOMES = ("home_win", "draw", "away_win")
GOAL_DIFF = {"home_win": 1, "draw": 0, "away_win": -1}


@dataclass(frozen=True)
class ROIConfig:
    min_train_matches: int = 200
    refit_every: int = 25
    min_edge: float = 0.03           # require model − implied >= 3 percentage points
    min_ev: float = 0.02             # require EV >= 2% per stake
    max_kelly_fraction: float = 0.05 # never stake more than 5% on one bet
    kelly_multiplier: float = 0.5    # half-Kelly
    starting_bankroll: float = 100.0
    optimizer_maxiter: int = 2500
    home_advantage: float = 0.22
    lookback_days: int = 730
    max_goals: int = 8
    # Implied-probability extraction method. "shin" (Shin 1993) is the
    # field standard for de-vigging bookmaker odds; "multiplicative" is
    # the naive sum-to-1 normalization. The choice subtly shifts the
    # implied probabilities and therefore the value finder's "edge".
    implied_method: str = "shin"
    # Goal model. Options:
    #   "dixon_coles_elo": Elo-blended MLE Dixon-Coles (default, accurate
    #                      but over-confident on large Elo gaps).
    #   "bayesian":        Hierarchical Bayesian Dixon-Coles. Slower
    #                      (~30x per refit) but better-calibrated. Does
    #                      NOT currently use Elo or xG; pure team-level
    #                      shrinkage. See models/dixon_coles_bayes.py.
    #   penaltyblog family: "dixon_coles", "bivariate_poisson", "poisson",
    #                       "negative_binomial", "zero_inflated_poisson"
    #   ensemble: "ensemble" (3-model avg), "market_fused" (blend with
    #             bookmaker implied probs)
    model: str = "dixon_coles_elo"


@dataclass(frozen=True)
class ROIResult:
    summary: dict[str, Any]
    bets: pd.DataFrame
    bankroll_curve: pd.DataFrame  # one row per bet: date, bankroll

    def to_dict(self, include_bets: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "summary": self.summary,
            "bankroll_curve": self.bankroll_curve.to_dict(orient="records"),
        }
        if include_bets:
            out["bets"] = self.bets.to_dict(orient="records")
        return out


def simulate_roi(
    matches: pd.DataFrame | Sequence[Mapping[str, Any]],
    odds: pd.DataFrame,
    *,
    config: ROIConfig | None = None,
    elo_config: EloConfig | None = None,
) -> ROIResult:
    """Run a walk-forward ROI simulation."""
    config = config or ROIConfig()
    frame = _prepare_matches(matches)
    if len(frame) <= config.min_train_matches:
        raise ValueError(
            f"Need more than {config.min_train_matches} matches for ROI sim; got {len(frame)}."
        )

    # Join odds onto matches by (date, home, away). Inner join — we can only
    # bet matches with odds available.
    odds_renamed = odds[["date", "home_team", "away_team", "odds_home", "odds_draw", "odds_away"]].copy()
    odds_renamed["date"] = pd.to_datetime(odds_renamed["date"])
    frame = frame.merge(odds_renamed, on=["date", "home_team", "away_team"], how="inner")
    if frame.empty:
        raise ValueError("No matches matched against the odds frame. Check league_key alignment.")
    if len(frame) <= config.min_train_matches:
        raise ValueError(
            f"After joining odds, only {len(frame)} matches remain — not enough to backtest."
        )
    frame = attach_pre_match_elos(frame, config=elo_config)

    bankroll = config.starting_bankroll
    bets: list[dict[str, Any]] = []
    bankroll_curve: list[dict[str, Any]] = []
    model: DixonColesModel | None = None
    last_fit_index = -1
    skipped_refits = 0

    use_penaltyblog = config.model in {
        "dixon_coles", "bivariate_poisson", "poisson",
        "negative_binomial", "zero_inflated_poisson",
    }
    use_ensemble = config.model in {"ensemble", "market_fused"}
    use_bayesian = config.model == "bayesian"
    market_fusion_weight = 0.5 if config.model == "market_fused" else 0.0

    for idx in range(config.min_train_matches, len(frame)):
        if model is None or idx - last_fit_index >= config.refit_every:
            train = frame.iloc[:idx].copy()
            try:
                if use_ensemble:
                    from models.ensemble import EnsembleConfig as EC, fit as ens_fit
                    model = ens_fit(
                        train,
                        config=EC(market_fusion_weight=market_fusion_weight),
                        home_advantage=config.home_advantage,
                        max_goals=config.max_goals,
                        optimizer_maxiter=config.optimizer_maxiter,
                    )
                elif use_penaltyblog:
                    model = _fit_penaltyblog(train, config.model)
                elif use_bayesian:
                    from models.dixon_coles_bayes import (  # noqa: PLC0415
                        BayesianDCConfig,
                        DixonColesBayesianModel,
                    )
                    bayes_cfg = BayesianDCConfig(
                        n_tune=1000, n_draws=1000, chains=2,
                        progressbar=False,
                    )
                    model = DixonColesBayesianModel(bayes_cfg).fit(train)
                else:
                    model = DixonColesModel(
                        DixonColesConfig(
                            lookback_days=config.lookback_days,
                            home_advantage=config.home_advantage,
                            max_goals=config.max_goals,
                            optimizer_maxiter=config.optimizer_maxiter,
                        )
                    ).fit(train, as_of=train["date"].max())
                last_fit_index = idx
            except RuntimeError:
                skipped_refits += 1
                if model is None:
                    raise
                continue
            except KeyError:
                # penaltyblog raises KeyError if asked to predict an
                # unfit team. Skip refits that hit this.
                skipped_refits += 1
                if model is None:
                    raise
                continue

        target = frame.iloc[idx]
        if use_ensemble:
            from models.ensemble import predict_match as ens_predict
            try:
                # For market_fused mode, feed the implied probs of THIS match
                # so the model sees the market view of the match it's about
                # to bet on (it then bets only if it disagrees with that view).
                market_implied = None
                if market_fusion_weight > 0:
                    market_implied = _implied_probability_lookup(
                        {
                            "home_win": float(target["odds_home"]),
                            "draw": float(target["odds_draw"]),
                            "away_win": float(target["odds_away"]),
                        },
                        method=config.implied_method,
                    )
                res = ens_predict(
                    model,
                    str(target["home_team"]),
                    str(target["away_team"]),
                    home_elo=float(target["home_elo"]),
                    away_elo=float(target["away_elo"]),
                    market_implied=market_implied,
                )
                probs = res["probabilities"]
            except (KeyError, ValueError):
                continue
        elif use_penaltyblog:
            try:
                grid = model.predict(
                    str(target["home_team"]),
                    str(target["away_team"]),
                    max_goals=config.max_goals,
                )
                probs = {
                    "home_win": float(grid.home_win),
                    "draw": float(grid.draw),
                    "away_win": float(grid.away_win),
                }
            except (KeyError, ValueError):
                continue
        elif use_bayesian:
            # Bayesian model.predict_match returns None for unseen teams.
            # Mirror the penaltyblog "skip and continue" behavior.
            pred = model.predict_match(
                str(target["home_team"]),
                str(target["away_team"]),
                max_goals=config.max_goals,
            )
            if pred is None:
                continue
            probs = {
                "home_win": float(pred["home_win"]),
                "draw": float(pred["draw"]),
                "away_win": float(pred["away_win"]),
            }
        else:
            prediction = model.predict_match(
                str(target["home_team"]),
                str(target["away_team"]),
                home_elo=float(target["home_elo"]),
                away_elo=float(target["away_elo"]),
                max_goals=config.max_goals,
            )
            probs = {
                "home_win": prediction.home_win,
                "draw": prediction.draw,
                "away_win": prediction.away_win,
            }
        odds_dict = {
            "home_win": float(target["odds_home"]),
            "draw": float(target["odds_draw"]),
            "away_win": float(target["odds_away"]),
        }
        actual = _actual_outcome(int(target["home_goals"]), int(target["away_goals"]))

        # Pick the best value bet across the three outcomes.
        candidate = _best_value_bet(probs, odds_dict, config)
        if candidate is None:
            continue

        outcome = candidate["outcome"]
        stake = bankroll * candidate["kelly_fraction"]
        won = outcome == actual
        profit = stake * (odds_dict[outcome] - 1) if won else -stake
        bankroll = bankroll + profit
        if bankroll <= 0:
            # Bankrupt — record the wipe and stop.
            bankroll = 0
            bets.append({
                "date": target["date"].date() if hasattr(target["date"], "date") else target["date"],
                "league_key": target.get("league_key"),
                "home_team": target["home_team"],
                "away_team": target["away_team"],
                "bet_on": outcome,
                "model_prob": probs[outcome],
                "implied_prob": candidate["implied_prob"],
                "edge": candidate["edge"],
                "ev": candidate["ev"],
                "kelly_fraction": candidate["kelly_fraction"],
                "stake": stake,
                "odds": odds_dict[outcome],
                "actual": actual,
                "won": won,
                "profit": profit,
                "bankroll_after": 0.0,
            })
            bankroll_curve.append({"date": str(target["date"].date()), "bankroll": 0.0, "n_bets": len(bets)})
            break

        bets.append({
            "date": target["date"].date() if hasattr(target["date"], "date") else target["date"],
            "league_key": target.get("league_key"),
            "home_team": target["home_team"],
            "away_team": target["away_team"],
            "bet_on": outcome,
            "model_prob": probs[outcome],
            "implied_prob": candidate["implied_prob"],
            "edge": candidate["edge"],
            "ev": candidate["ev"],
            "kelly_fraction": candidate["kelly_fraction"],
            "stake": stake,
            "odds": odds_dict[outcome],
            "actual": actual,
            "won": bool(won),
            "profit": profit,
            "bankroll_after": bankroll,
        })
        bankroll_curve.append({
            "date": str(target["date"].date() if hasattr(target["date"], "date") else target["date"]),
            "bankroll": float(bankroll),
            "n_bets": len(bets),
        })

    bets_frame = pd.DataFrame(bets)
    curve_frame = pd.DataFrame(bankroll_curve)
    summary = _summarize(bets_frame, curve_frame, config, n_matches_eligible=len(frame) - config.min_train_matches)
    summary["skipped_refits"] = skipped_refits
    return ROIResult(summary=summary, bets=bets_frame, bankroll_curve=curve_frame)


def _fit_penaltyblog(train: pd.DataFrame, model_name: str):
    """Fit a penaltyblog goal model on a chronological slice."""
    import numpy as np

    import penaltyblog as pb

    factories = {
        "dixon_coles": pb.models.DixonColesGoalModel,
        "bivariate_poisson": pb.models.BivariatePoissonGoalModel,
        "poisson": pb.models.PoissonGoalsModel,
        "negative_binomial": pb.models.NegativeBinomialGoalModel,
        "zero_inflated_poisson": pb.models.ZeroInflatedPoissonGoalsModel,
    }
    factory = factories[model_name]
    instance = factory(
        goals_home=np.array(train["home_goals"].values, dtype=np.int64),
        goals_away=np.array(train["away_goals"].values, dtype=np.int64),
        teams_home=np.array(train["home_team"].values, dtype=str),
        teams_away=np.array(train["away_team"].values, dtype=str),
    )
    instance.fit()
    return instance


def _implied_probability_lookup(
    odds: dict[str, float],
    *,
    method: str = "shin",
) -> dict[str, float]:
    """Return de-vigged implied probabilities keyed by outcome.

    Uses penaltyblog's `calculate_implied` when method != 'multiplicative'
    so we get Shin / power / additive treatment for free. Falls back to the
    naive overround normalization if the import isn't available — keeps the
    module testable without penaltyblog.
    """
    def _multiplicative() -> dict[str, float]:
        overround = sum(1 / o for o in odds.values())
        return {outcome: (1 / odds[outcome]) / overround for outcome in OUTCOMES}

    if method == "multiplicative":
        return _multiplicative()
    try:
        from models.implied_probs import implied_probabilities
    except ImportError:
        return _multiplicative()
    try:
        res = implied_probabilities(
            odds["home_win"], odds["draw"], odds["away_win"], method=method,
        )
    except (ValueError, RuntimeError):
        # Shin's method needs a positive overround and can fail when odds
        # are exotic (very low margin or extreme longshots). Fall back to
        # naive normalization rather than aborting the simulation.
        return _multiplicative()
    return {"home_win": res.home_win, "draw": res.draw, "away_win": res.away_win}


def _best_value_bet(
    probs: dict[str, float],
    odds: dict[str, float],
    config: ROIConfig,
) -> dict[str, Any] | None:
    implied_lookup = _implied_probability_lookup(odds, method=config.implied_method)
    best: dict[str, Any] | None = None
    for outcome in OUTCOMES:
        p = probs[outcome]
        o = odds[outcome]
        implied = implied_lookup[outcome]
        edge = p - implied
        ev = p * (o - 1) - (1 - p)
        if edge < config.min_edge or ev < config.min_ev:
            continue
        kelly_full = max(0.0, (p * o - 1) / (o - 1))
        kelly_fraction = min(
            config.max_kelly_fraction,
            kelly_full * config.kelly_multiplier,
        )
        if kelly_fraction <= 0:
            continue
        candidate = {
            "outcome": outcome,
            "implied_prob": implied,
            "edge": edge,
            "ev": ev,
            "kelly_fraction": kelly_fraction,
        }
        if best is None or ev > best["ev"]:
            best = candidate
    return best


def _summarize(
    bets: pd.DataFrame,
    curve: pd.DataFrame,
    config: ROIConfig,
    *,
    n_matches_eligible: int,
) -> dict[str, Any]:
    if bets.empty:
        return {
            "n_bets": 0,
            "n_matches_eligible": n_matches_eligible,
            "bet_rate": 0.0,
            "total_staked": 0.0,
            "total_profit": 0.0,
            "roi": 0.0,
            "hit_rate": 0.0,
            "starting_bankroll": config.starting_bankroll,
            "ending_bankroll": config.starting_bankroll,
            "max_drawdown_pct": 0.0,
            "max_bankroll": config.starting_bankroll,
            "min_bankroll": config.starting_bankroll,
        }
    total_staked = float(bets["stake"].sum())
    total_profit = float(bets["profit"].sum())
    hit_rate = float(bets["won"].mean())
    roi = total_profit / total_staked if total_staked else 0.0
    bankroll_series = curve["bankroll"].astype(float)
    running_max = bankroll_series.cummax()
    drawdown = (bankroll_series - running_max) / running_max
    max_drawdown_pct = float(-drawdown.min()) if not drawdown.empty else 0.0
    return {
        "n_bets": int(len(bets)),
        "n_matches_eligible": n_matches_eligible,
        "bet_rate": float(len(bets) / max(n_matches_eligible, 1)),
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": roi,
        "hit_rate": hit_rate,
        "starting_bankroll": config.starting_bankroll,
        "ending_bankroll": float(bankroll_series.iloc[-1]),
        "max_drawdown_pct": max_drawdown_pct,
        "max_bankroll": float(bankroll_series.max()),
        "min_bankroll": float(bankroll_series.min()),
    }


def _prepare_matches(matches: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(matches).copy()
    required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing match columns for ROI sim: {sorted(missing)}")
    frame = frame.dropna(subset=list(required))
    frame["date"] = pd.to_datetime(frame["date"])
    frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce").astype(int)
    frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce").astype(int)
    return frame.sort_values("date").reset_index(drop=True)


def _actual_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"
