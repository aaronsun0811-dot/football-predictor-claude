"""Walk-forward backtest: MLE Dixon-Coles vs Hierarchical Bayesian DC.

Same matches, same train/test split policy, same metrics. Runs both
models in parallel and prints accuracy / Brier / log-loss / ECE side-by-
side. The question this answers: is the Bayesian shrinkage actually
buying us anything that justifies the 30-100x fit cost?

Usage:
    python scripts/compare_dc_vs_bayes.py --league premier_league \
        --min-train 300 --refit-every 20 --max-tests 100
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.database import Database  # noqa: E402
from models.dixon_coles import DixonColesConfig, DixonColesModel  # noqa: E402
from models.dixon_coles_bayes import BayesianDCConfig, DixonColesBayesianModel  # noqa: E402
from models.elo import attach_pre_match_elos  # noqa: E402


def _outcome(home_g: int, away_g: int) -> str:
    if home_g > away_g: return "home_win"
    if home_g < away_g: return "away_win"
    return "draw"


def _metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    df = pd.DataFrame(rows)
    n = len(df)
    correct = (df["pred_outcome"] == df["actual"]).mean()
    # Brier (multi-class)
    one_hot = np.zeros((n, 3))
    label_idx = {"home_win": 0, "draw": 1, "away_win": 2}
    for i, a in enumerate(df["actual"]):
        one_hot[i, label_idx[a]] = 1
    probs = df[["p_home", "p_draw", "p_away"]].values
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    # Log-loss: probability assigned to the actual outcome
    chosen = np.array([probs[i, label_idx[a]] for i, a in enumerate(df["actual"])])
    log_loss = float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0))))
    # ECE (10 equal-width bins on the max-prob)
    max_prob = probs.max(axis=1)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (max_prob >= lo) & (max_prob < hi)
        if not mask.any():
            continue
        acc_in_bin = (df["pred_outcome"][mask] == df["actual"][mask]).mean()
        conf_in_bin = max_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc_in_bin - conf_in_bin)
    return {
        "n": n,
        "accuracy": float(correct),
        "brier_score": brier,
        "log_loss": log_loss,
        "ece": float(ece),
    }


def run_walk_forward(
    matches: pd.DataFrame,
    model_factory,
    *,
    min_train: int,
    refit_every: int,
    max_tests: int | None,
    needs_elo: bool,
):
    """Generic walk-forward loop. model_factory returns a fresh model
    given training matches; the returned object must have predict_match()."""
    frame = matches.sort_values("date").reset_index(drop=True).copy()
    if needs_elo:
        frame = attach_pre_match_elos(frame)
    rows = []
    model = None
    last_fit = -1
    end_idx = len(frame) if max_tests is None else min(len(frame), min_train + max_tests)
    for idx in range(min_train, end_idx):
        if model is None or idx - last_fit >= refit_every:
            train = frame.iloc[:idx].copy()
            try:
                model = model_factory(train)
                last_fit = idx
            except Exception as exc:
                print(f"  fit failed at idx={idx}: {exc}", file=sys.stderr)
                continue
        row = frame.iloc[idx]
        if pd.isna(row["home_goals"]) or pd.isna(row["away_goals"]):
            continue
        pred = model.predict_match(row["home_team"], row["away_team"])
        if pred is None:
            continue
        # Both model APIs return either dict or dataclass-ish object
        if hasattr(pred, "home_win"):
            p_h, p_d, p_a = pred.home_win, pred.draw, pred.away_win
        else:
            p_h, p_d, p_a = pred["home_win"], pred["draw"], pred["away_win"]
        probs = (p_h, p_d, p_a)
        labels = ("home_win", "draw", "away_win")
        pred_outcome = labels[int(np.argmax(probs))]
        actual = _outcome(int(row["home_goals"]), int(row["away_goals"]))
        rows.append({
            "p_home": p_h, "p_draw": p_d, "p_away": p_a,
            "pred_outcome": pred_outcome, "actual": actual,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="premier_league")
    ap.add_argument("--min-train", type=int, default=300)
    ap.add_argument("--refit-every", type=int, default=20)
    ap.add_argument("--max-tests", type=int, default=200,
                    help="Cap walk-forward iterations. None for unbounded; Bayesian is slow.")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    db = Database(PROJECT_ROOT / "data/football.sqlite3")
    matches = db.fetch_matches(league_key=args.league)
    print(f"League: {args.league}, total matches: {len(matches)}")

    # MLE factory
    def mle_factory(train):
        return DixonColesModel(DixonColesConfig()).fit(train, as_of=train["date"].max())

    # Bayesian factory — quicker config for backtest (fewer draws ok for point estimates)
    bayes_cfg = BayesianDCConfig(n_tune=300, n_draws=300, chains=2)
    def bayes_factory(train):
        return DixonColesBayesianModel(bayes_cfg).fit(train)

    print("\n=== Running MLE walk-forward ===")
    t0 = time.time()
    mle_rows = run_walk_forward(matches, mle_factory,
                                 min_train=args.min_train,
                                 refit_every=args.refit_every,
                                 max_tests=args.max_tests,
                                 needs_elo=True)
    mle_time = time.time() - t0
    print(f"Done in {mle_time:.1f}s")

    print("\n=== Running Bayesian walk-forward ===")
    t0 = time.time()
    bayes_rows = run_walk_forward(matches, bayes_factory,
                                   min_train=args.min_train,
                                   refit_every=args.refit_every,
                                   max_tests=args.max_tests,
                                   needs_elo=False)
    bayes_time = time.time() - t0
    print(f"Done in {bayes_time:.1f}s")

    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD")
    print("=" * 60)
    mle_m = _metrics(mle_rows)
    bayes_m = _metrics(bayes_rows)
    print(f"\n{'metric':<15s}  {'MLE':>12s}  {'Bayesian':>12s}  {'Δ':>12s}")
    print("-" * 60)
    for k in ["n", "accuracy", "brier_score", "log_loss", "ece"]:
        m1, m2 = mle_m.get(k), bayes_m.get(k)
        if isinstance(m1, float):
            delta = m2 - m1
            sign = "better" if (k == "accuracy" and delta > 0) or (k != "accuracy" and delta < 0) else "worse"
            print(f"{k:<15s}  {m1:>12.4f}  {m2:>12.4f}  {delta:>+12.4f} ({sign if k != 'n' else ''})")
        else:
            print(f"{k:<15s}  {m1:>12}  {m2:>12}")
    print(f"\nFit time: MLE {mle_time:.1f}s, Bayesian {bayes_time:.1f}s "
          f"({bayes_time/max(mle_time,0.1):.1f}x slower)")


if __name__ == "__main__":
    main()
