from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from models.dixon_coles import DixonColesConfig, DixonColesModel
from models.elo import EloConfig, attach_pre_match_elos


OUTCOMES = ("home_win", "draw", "away_win")


@dataclass(frozen=True)
class BacktestConfig:
    min_train_matches: int = 80
    lookback_days: int = 730
    max_goals: int = 8
    optimizer_maxiter: int = 2500
    # 700 was too low for 24-team leagues (Championship, League One, Segunda)
    # whose later walk-forward fits push past it. 2500 matches the production
    # default in DixonColesConfig with headroom for the largest training sets.
    home_advantage: float = 0.22
    refit_every: int = 1
    xg_blend_weight: float = 0.35
    # Ablation knob: 0 silences the Elo prior entirely (the model becomes pure
    # team-attack/defense Dixon-Coles with no rating-based shrinkage). Default
    # 0.10 matches DixonColesConfig.elo_weight — the production setting.
    # Set to 0.0 in /diagnostics/ablation to measure how much Elo contributes.
    elo_weight: float = 0.10


@dataclass(frozen=True)
class BacktestResult:
    summary: dict[str, Any]
    predictions: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "predictions": self.predictions.to_dict(orient="records"),
        }


def backtest_dixon_coles(
    matches: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    config: BacktestConfig | None = None,
    elo_config: EloConfig | None = None,
) -> BacktestResult:
    config = config or BacktestConfig()
    frame = _prepare_matches(matches)
    if len(frame) <= config.min_train_matches:
        raise ValueError(
            f"Need more than {config.min_train_matches} matches for backtest; got {len(frame)}."
        )

    frame = attach_pre_match_elos(frame, config=elo_config)
    rows: list[dict[str, Any]] = []
    model: DixonColesModel | None = None
    last_fit_index = -1

    skipped_refits = 0
    refit_attempts = 0  # Total times we tried to fit (including the failed ones)
    # Failed-refit history: which dates in the walk did fitting blow up?
    # Useful when investigating why a particular accuracy dip happened.
    failed_refit_log: list[dict[str, Any]] = []
    for idx in range(config.min_train_matches, len(frame)):
        attempted_fit_here = False
        fit_failed_here = False
        if model is None or idx - last_fit_index >= config.refit_every:
            attempted_fit_here = True
            refit_attempts += 1
            train = frame.iloc[:idx].copy()
            try:
                model = DixonColesModel(
                    DixonColesConfig(
                        lookback_days=config.lookback_days,
                        home_advantage=config.home_advantage,
                        max_goals=config.max_goals,
                        optimizer_maxiter=config.optimizer_maxiter,
                        xg_blend_weight=config.xg_blend_weight,
                        elo_weight=config.elo_weight,
                    )
                ).fit(train, as_of=train["date"].max())
                last_fit_index = idx
            except RuntimeError as exc:
                # L-BFGS-B hit the eval limit. Keep the previous model and
                # carry on so one bad fit doesn't abort the whole backtest.
                # First-iteration failure is fatal because we have no model yet.
                skipped_refits += 1
                fit_failed_here = True
                # Record enough to debug later — date + training-set size +
                # short error message. Capped to a sample so a runaway failure
                # loop can't bloat the response.
                if len(failed_refit_log) < 50:
                    failed_refit_log.append({
                        "match_index": idx,
                        "as_of": str(frame.iloc[idx - 1]["date"].date()
                                     if hasattr(frame.iloc[idx - 1]["date"], "date")
                                     else frame.iloc[idx - 1]["date"]),
                        "train_size": idx,
                        "error": str(exc)[:120],
                    })
                if model is None:
                    raise
                continue

        target = frame.iloc[idx]
        # ``model_age_matches`` = how many actual played matches happened
        # between the most recent successful fit and this prediction. A fresh
        # fit gives 0; a long staleness gap (e.g. a series of failed refits in
        # the middle of the season) blows this up. Use it to filter the
        # accuracy rollup to "fresh-fit predictions only" if desired.
        model_age_matches = idx - last_fit_index
        prediction = model.predict_match(
            str(target["home_team"]),
            str(target["away_team"]),
            home_elo=float(target["home_elo"]),
            away_elo=float(target["away_elo"]),
            max_goals=config.max_goals,
        )
        actual = _actual_outcome(int(target["home_goals"]), int(target["away_goals"]))
        probabilities = {
            "home_win": prediction.home_win,
            "draw": prediction.draw,
            "away_win": prediction.away_win,
        }
        predicted = max(OUTCOMES, key=lambda outcome: probabilities[outcome])

        # Score-level prediction: same definition as /audit (Round 19) — the
        # argmax of the score matrix. Lets us measure "how often is the actual
        # scoreline exactly what the model considered most likely" and "how
        # far off is the predicted score on average."
        actual_hg, actual_ag = int(target["home_goals"]), int(target["away_goals"])
        top_score = (prediction.most_likely_scores or [None])[0]
        if top_score is not None:
            pred_hg = int(top_score["home_goals"])
            pred_ag = int(top_score["away_goals"])
            exact_score_correct = (pred_hg == actual_hg and pred_ag == actual_ag)
            goal_distance = abs(pred_hg - actual_hg) + abs(pred_ag - actual_ag)
            predicted_score = f"{pred_hg}-{pred_ag}"
            predicted_score_prob = float(top_score["probability"])
        else:
            pred_hg = pred_ag = None
            exact_score_correct = None
            goal_distance = None
            predicted_score = None
            predicted_score_prob = None

        rows.append(
            {
                "date": target["date"].date() if hasattr(target["date"], "date") else target["date"],
                "league_key": target.get("league_key"),
                "home_team": target["home_team"],
                "away_team": target["away_team"],
                "home_goals": actual_hg,
                "away_goals": actual_ag,
                "actual": actual,
                "predicted": predicted,
                "correct": predicted == actual,
                "home_win": prediction.home_win,
                "draw": prediction.draw,
                "away_win": prediction.away_win,
                "expected_home_goals": prediction.expected_home_goals,
                "expected_away_goals": prediction.expected_away_goals,
                "xg_training_rows": model.xg_training_rows_,
                "model_age_matches": int(model_age_matches),
                "fit_attempted_here": bool(attempted_fit_here),
                "fit_failed_here": bool(fit_failed_here),
                "predicted_score": predicted_score,
                "predicted_score_prob": predicted_score_prob,
                "exact_score_correct": exact_score_correct,
                "goal_distance": goal_distance,
            }
        )

    predictions = pd.DataFrame(rows)
    summary = summarize_predictions(predictions)
    summary["n_train_min"] = config.min_train_matches
    summary["n_matches_total"] = len(frame)
    summary["xg_blend_weight"] = config.xg_blend_weight

    # Fit-health diagnostics. Previously only ``skipped_refits`` was exposed —
    # but a count without context can't tell you whether 5 failures over 1000
    # matches matters or whether the failures clustered in one season corner.
    summary["fit_health"] = _summarize_fit_health(
        predictions,
        refit_attempts=refit_attempts,
        skipped_refits=skipped_refits,
        failed_refit_log=failed_refit_log,
    )
    return BacktestResult(summary=summary, predictions=predictions)


def backtest_bayesian_dc(
    matches: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Walk-forward backtest using the hierarchical Bayesian DC.

    Output schema matches ``backtest_dixon_coles`` exactly so downstream
    metric code (summarize_predictions, fit-health, per-league rollup)
    doesn't care which model produced the predictions. Differences vs MLE:

      - No Elo signal. The Bayesian model is a pure team-attack/defense
        hierarchical fit; Elo isn't in scope for this first iteration.
        ``home_elo`` / ``away_elo`` columns get filled with None.
      - No xG blending. Same reason.
      - No fit retries on transient failures (NUTS has its own
        convergence handling). Skipped/failed-fit counts reflect that.
      - ``model_age_matches`` is still tracked the same way as MLE so
        the staleness rollup keeps working.

    Cost: ~30x slower per refit than MLE. Use a larger ``refit_every``
    (50+) for full-league backtests to keep total wall time tractable.
    """
    from models.dixon_coles_bayes import (  # noqa: PLC0415 — lazy import
        BayesianDCConfig,
        DixonColesBayesianModel,
    )

    config = config or BacktestConfig()
    frame = _prepare_matches(matches)
    if len(frame) <= config.min_train_matches:
        raise ValueError(
            f"Need more than {config.min_train_matches} matches for backtest; got {len(frame)}."
        )

    bayes_cfg = BayesianDCConfig(
        n_tune=1000, n_draws=1000, chains=2,
        progressbar=False,
    )

    rows: list[dict[str, Any]] = []
    model: DixonColesBayesianModel | None = None
    last_fit_index = -1

    skipped_refits = 0
    refit_attempts = 0
    failed_refit_log: list[dict[str, Any]] = []

    for idx in range(config.min_train_matches, len(frame)):
        attempted_fit_here = False
        fit_failed_here = False
        if model is None or idx - last_fit_index >= config.refit_every:
            attempted_fit_here = True
            refit_attempts += 1
            train = frame.iloc[:idx].copy()
            try:
                model = DixonColesBayesianModel(bayes_cfg).fit(train)
                last_fit_index = idx
            except Exception as exc:  # noqa: BLE001
                skipped_refits += 1
                fit_failed_here = True
                if len(failed_refit_log) < 50:
                    failed_refit_log.append({
                        "match_index": idx,
                        "as_of": str(frame.iloc[idx - 1]["date"].date()
                                     if hasattr(frame.iloc[idx - 1]["date"], "date")
                                     else frame.iloc[idx - 1]["date"]),
                        "train_size": idx,
                        "error": str(exc)[:120],
                    })
                if model is None:
                    raise

        target = frame.iloc[idx]
        model_age_matches = idx - last_fit_index
        pred = model.predict_match(
            str(target["home_team"]),
            str(target["away_team"]),
            max_goals=config.max_goals,
        )
        if pred is None:
            # Unseen team — Bayesian can't predict it. Skip this row entirely,
            # mirroring how MLE backtest just wouldn't produce a row.
            continue

        actual_hg, actual_ag = int(target["home_goals"]), int(target["away_goals"])
        actual = _actual_outcome(actual_hg, actual_ag)
        probs = {
            "home_win": pred["home_win"],
            "draw": pred["draw"],
            "away_win": pred["away_win"],
        }
        predicted = max(OUTCOMES, key=lambda o: probs[o])

        # Score matrix from Bayesian → most-likely scoreline (mirrors MLE shape).
        score_matrix = pred["score_matrix"]
        flat_idx = int(score_matrix.argmax())
        pred_hg, pred_ag = divmod(flat_idx, score_matrix.shape[1])
        predicted_score = f"{pred_hg}-{pred_ag}"
        predicted_score_prob = float(score_matrix[pred_hg, pred_ag])
        exact_score_correct = (pred_hg == actual_hg and pred_ag == actual_ag)
        goal_distance = abs(pred_hg - actual_hg) + abs(pred_ag - actual_ag)

        rows.append({
            "date": target["date"].date() if hasattr(target["date"], "date") else target["date"],
            "league_key": target.get("league_key"),
            "home_team": target["home_team"],
            "away_team": target["away_team"],
            "home_goals": actual_hg,
            "away_goals": actual_ag,
            "actual": actual,
            "predicted": predicted,
            "correct": predicted == actual,
            "home_win": pred["home_win"],
            "draw": pred["draw"],
            "away_win": pred["away_win"],
            "expected_home_goals": pred["expected_home_goals"],
            "expected_away_goals": pred["expected_away_goals"],
            # Bayesian doesn't currently track per-match xG sample count
            "xg_training_rows": None,
            "model_age_matches": int(model_age_matches),
            "fit_attempted_here": bool(attempted_fit_here),
            "fit_failed_here": bool(fit_failed_here),
            "predicted_score": predicted_score,
            "predicted_score_prob": predicted_score_prob,
            "exact_score_correct": exact_score_correct,
            "goal_distance": goal_distance,
        })

    predictions = pd.DataFrame(rows)
    summary = summarize_predictions(predictions)
    summary["n_train_min"] = config.min_train_matches
    summary["n_matches_total"] = len(frame)
    summary["model"] = "dixon_coles_bayes"
    summary["fit_health"] = _summarize_fit_health(
        predictions,
        refit_attempts=refit_attempts,
        skipped_refits=skipped_refits,
        failed_refit_log=failed_refit_log,
    )
    return BacktestResult(summary=summary, predictions=predictions)


def _summarize_fit_health(
    predictions: pd.DataFrame,
    *,
    refit_attempts: int,
    skipped_refits: int,
    failed_refit_log: list[dict[str, Any]],
) -> dict[str, Any]:
    """How fresh was the model when each prediction was made? Roll it up so
    users can read 'fitting worked most of the time' or 'half the predictions
    were on a >20-matches-stale model'."""
    if predictions.empty:
        return {
            "refit_attempts": refit_attempts,
            "skipped_refits": skipped_refits,
            "refits_succeeded": refit_attempts - skipped_refits,
            "pct_with_fresh_model": None,
            "max_model_staleness": None,
            "mean_model_staleness": None,
            "failed_refits": failed_refit_log,
        }
    age_col = predictions["model_age_matches"] if "model_age_matches" in predictions.columns else pd.Series(dtype=int)
    return {
        "refit_attempts": refit_attempts,
        "skipped_refits": skipped_refits,
        "refits_succeeded": refit_attempts - skipped_refits,
        # "fresh" = made on the same iteration a refit succeeded (age == 0)
        "pct_with_fresh_model": (
            float((age_col == 0).mean()) if not age_col.empty else None
        ),
        "max_model_staleness": int(age_col.max()) if not age_col.empty else None,
        "mean_model_staleness": float(age_col.mean()) if not age_col.empty else None,
        # Limited to first 50 failures so a runaway loop can't bloat the
        # response. Kept for debug, never used in roll-up math.
        "failed_refits": failed_refit_log,
    }


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {
            "n_predictions": 0,
            "accuracy": None,
            "brier_score": None,
            "log_loss": None,
            "rps": None,
            "n_scored": 0,
            "exact_score_accuracy": None,
            "mean_goal_distance": None,
        }

    probs = predictions[list(OUTCOMES)].to_numpy(dtype=float)
    actual_idx = np.array([OUTCOMES.index(value) for value in predictions["actual"]])
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(predictions)), actual_idx] = 1.0
    chosen = probs[np.arange(len(predictions)), actual_idx]
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    log_loss = -np.mean(np.log(np.clip(chosen, 1e-12, 1.0)))

    # Ranked Probability Score — preferred 3-way metric in football modelling
    # because it weighs ordered errors (Constantinou & Fenton 2012). Lower=better.
    # We compute it ourselves rather than depending on penaltyblog so the
    # core summary doesn't have a hard import. The formula is:
    #   RPS = (1 / (k - 1)) * Σᵢ (Σⱼ≤ᵢ (pⱼ - oⱼ))²
    # where k is the number of ordered outcome categories.
    cum_probs = np.cumsum(probs, axis=1)
    cum_actual = np.cumsum(one_hot, axis=1)
    # Sum the squared cumulative differences for the first k-1 categories
    # (the last cumulative sum is always 1 for both).
    rps = float(np.mean(np.sum((cum_probs[:, :-1] - cum_actual[:, :-1]) ** 2, axis=1)) /
                (probs.shape[1] - 1))

    summary = {
        "n_predictions": int(len(predictions)),
        "accuracy": float(predictions["correct"].mean()),
        "brier_score": float(brier),
        "log_loss": float(log_loss),
        "rps": rps,
        "home_win_rate": float((predictions["actual"] == "home_win").mean()),
        "draw_rate": float((predictions["actual"] == "draw").mean()),
        "away_win_rate": float((predictions["actual"] == "away_win").mean()),
    }

    # Score-level rollup — only over rows that actually carry a predicted
    # scoreline (always true for the post-R28 walk-forward, but defensive in
    # case ``summarize_predictions`` ever sees pre-R28 data).
    if "exact_score_correct" in predictions.columns:
        scored = predictions[predictions["exact_score_correct"].notna()]
    else:
        scored = predictions.iloc[0:0]
    summary["n_scored"] = int(len(scored))
    if not scored.empty:
        summary["exact_score_accuracy"] = float(scored["exact_score_correct"].astype(bool).mean())
        summary["mean_goal_distance"] = float(scored["goal_distance"].mean())
    else:
        summary["exact_score_accuracy"] = None
        summary["mean_goal_distance"] = None

    # Per-league breakdown. For single-league backtests this is just one row
    # (matches the headline). For multi-league it lets the user see which
    # league the model handles best/worst — e.g. Brier 0.55 in PL vs 0.62 in
    # Liga MX is a real signal about where the prior holds.
    summary["by_league"] = _per_league_breakdown(predictions)

    return summary


def _per_league_breakdown(predictions: pd.DataFrame, *, min_n: int = 30) -> list[dict[str, Any]]:
    """Group by ``league_key`` and report headline metrics per league.

    Sorted by sample-size descending. Leagues with fewer than ``min_n`` predictions
    are dropped — small-sample-noise outliers would mislead more than they inform.
    """
    if "league_key" not in predictions.columns:
        return []
    out: list[dict[str, Any]] = []
    for league_key, sub in predictions.groupby("league_key", dropna=True):
        if pd.isna(league_key) or len(sub) < min_n:
            continue
        probs = sub[list(OUTCOMES)].to_numpy(dtype=float)
        actual_idx = np.array([OUTCOMES.index(value) for value in sub["actual"]])
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(sub)), actual_idx] = 1.0
        chosen = probs[np.arange(len(sub)), actual_idx]
        brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
        log_loss = float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0))))
        cum_p = np.cumsum(probs, axis=1)
        cum_a = np.cumsum(one_hot, axis=1)
        rps = float(
            np.mean(np.sum((cum_p[:, :-1] - cum_a[:, :-1]) ** 2, axis=1))
            / (probs.shape[1] - 1)
        )
        entry = {
            "league_key": str(league_key),
            "n": int(len(sub)),
            "accuracy": float(sub["correct"].astype(bool).mean()),
            "brier": brier,
            "log_loss": log_loss,
            "rps": rps,
        }
        # Score-level metrics — defensive same way as the rollup above.
        if "exact_score_correct" in sub.columns:
            scored_sub = sub[sub["exact_score_correct"].notna()]
        else:
            scored_sub = sub.iloc[0:0]
        if not scored_sub.empty:
            entry["exact_score_accuracy"] = float(
                scored_sub["exact_score_correct"].astype(bool).mean()
            )
            entry["mean_goal_distance"] = float(scored_sub["goal_distance"].mean())
        else:
            entry["exact_score_accuracy"] = None
            entry["mean_goal_distance"] = None
        out.append(entry)
    out.sort(key=lambda r: -r["n"])
    return out


def _prepare_matches(matches: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(matches).copy()
    required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing match columns for backtest: {sorted(missing)}")
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
