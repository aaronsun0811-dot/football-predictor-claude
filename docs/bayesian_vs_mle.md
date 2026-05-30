# Bayesian DC vs MLE DC — head-to-head

Verified walk-forward backtest on the Premier League. Both models given the
**same** features (Elo + xG) so the comparison isolates the modeling approach,
not the feature set.

## Setup

- League: `premier_league`
- `min_train_matches=600`, `refit_every=25`
- 308 walk-forward test predictions
- xG coverage at time of run: ~50% of the 2024-25 season (API-Football backfill
  in progress; see `deploy/com.aaronsun.football-predictor-claude.api-xg-backfill.plist`)
- Command:
  ```python
  backtest_payload(BacktestRequest(
      league="premier_league", min_train_matches=600,
      refit_every=25, model_choice="mle" | "bayesian"))
  ```

## Result (verified 2026-05)

| metric        | MLE+Elo+xG | Bayes+Elo+xG | winner          |
|---------------|------------|--------------|-----------------|
| accuracy      | 52.27%     | **52.60%**   | Bayes +0.33pp   |
| Brier         | 0.5839     | **0.5807**   | Bayes (lower)   |
| log loss      | 0.9811     | **0.9785**   | Bayes (lower)   |
| ECE           | 0.0719     | **0.0389**   | Bayes −46%      |
| pred-draw rate| 6.8%       | **21.4%**    | Bayes (actual 25.6%) |

**Bayesian wins on every metric.**

## Why this matters

1. **Draw prediction fixed.** The chronic Dixon-Coles failure — never picking
   draws because the draw probability is split out from a single peak — is
   largely resolved. MLE predicts draws 6.8% of the time when reality is 25.6%.
   Bayesian gets to 21.4%, because the hierarchical shrinkage + learned Elo
   coefficient produce flatter, better-calibrated probability vectors rather
   than over-confident home/away peaks.

2. **Calibration nearly halved (ECE 0.0719 → 0.0389).** 0.0389 is right at the
   textbook "excellent calibration" threshold (<0.04). Well-calibrated
   probabilities are the prerequisite for the value-pick rule (see
   `models/prediction_audit.py::_value_pick`) to fire meaningfully.

3. **Contrast with yesterday's Bayesian-ONLY result.** When the Bayesian model
   had no Elo and no xG, it *lost* to MLE+Elo on Brier/log-loss (the calibration
   win was real but it gave up the Elo signal). The lesson held: the models must
   share features for the comparison to be fair. With Elo + xG integrated, the
   Bayesian advantage is unambiguous.

## What changed in the model

`models/dixon_coles_bayes.py` (commit after the MVP):
- **Elo as a learned covariate** — `elo_coef * (home_elo − away_elo)/scale`
  added to the log-rate, with a `Normal(0, 0.5)` prior. Data learns the
  coefficient (~0.081 on EPL) rather than the MLE's hardcoded 0.10.
- **xG as a second noisy likelihood** — `Normal(latent_rate, 0.75)` on matches
  where xG is present. Continuous, no rounding to fake integer goals like the
  MLE blend. Partial coverage handled gracefully (rows without xG just don't
  contribute the term).

## Open questions / next

- **ROI sim**: does the calibration win translate to a less-negative ROI?
  Yesterday Bayesian-only was −22%; need to re-run ROI with the Elo+xG Bayesian.
- **Full xG coverage**: re-run this head-to-head once EPL hits 100% xG (a few
  more days of the daily backfill). Expect the gap to widen.
- **Cost**: Bayesian fit is ~30x slower than MLE (~20s vs <1s per refit). Fine
  for backtest/analysis; the live `/predict` path would need fit caching before
  it could serve Bayesian predictions interactively.
- **Tier-3 leagues**: the shrinkage advantage should be *largest* where MLE
  overfits most — small-sample leagues (Saudi, J1, MLS). Untested so far
  because those leagues lack the xG + odds for a clean comparison.
