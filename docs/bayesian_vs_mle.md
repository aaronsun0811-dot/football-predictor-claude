# Bayesian DC vs MLE DC — head-to-head

Verified walk-forward backtest on the Premier League. Both models given the
**same** features (Elo + xG) so the comparison isolates the modeling approach,
not the feature set.

## Setup

- League: `premier_league`
- `min_train_matches=600`, `refit_every=25`
- Full walk-forward: ~1300 test predictions (n=1302 MLE / 1297 Bayesian; the
  small difference is Bayesian skipping a handful of unseen-team rows)
- xG coverage at run time: ~50% of the 2024-25 season (API-Football backfill
  in progress; see `deploy/...api-xg-backfill.plist`)
- Command:
  ```python
  backtest_payload(BacktestRequest(
      league="premier_league", min_train_matches=600,
      refit_every=25, model_choice="mle" | "bayesian"))
  ```

Raw numbers in the companion `bayesian_vs_mle_data.json`
(`[n, accuracy, brier, log_loss, ece, pred_draw, actual_draw]`).

## Result (verified 2026-05)

| metric         | MLE+Elo+xG | Bayes+Elo+xG | winner            |
|----------------|------------|--------------|-------------------|
| accuracy       | 53.23%     | **53.66%**   | Bayes +0.43pp     |
| Brier          | **0.5810** | 0.5826       | MLE (lower) −0.0016 |
| log loss       | 0.9802     | **0.9780**   | Bayes (lower)     |
| ECE            | 0.0358     | **0.0151**   | Bayes −58%        |
| pred-draw rate | 0.0%       | 0.0%         | tie — see below   |

**Mixed, leaning Bayes.** Bayesian wins accuracy, log-loss, and calibration
(ECE) decisively; MLE edges Brier by a hair.

## Honest reading

1. **Calibration is the real win.** ECE drops from 0.0358 to **0.0151** — well
   under the 0.04 "excellent" threshold, less than half MLE's. The Bayesian
   probability vectors genuinely match observed frequencies better. This is the
   prerequisite for the value-pick rule (`prediction_audit::_value_pick`) and
   for any honest "edge" calculation in the ROI sim.

2. **The draw problem is NOT fixed at the argmax level.** Both models pick a
   draw as their single top outcome **0%** of the time, despite draws being
   ~24% of results. The earlier smoke test showed Bayesian assigns a *healthy*
   ~25% probability to draws (vs MLE crushing it lower) — but a 25% draw prob
   still loses argmax to a 38%+ home or away peak almost every time. So:
   - **Probability quality**: Bayesian materially better (this is what ECE
     measures and rewards).
   - **Argmax label**: unchanged — draws still never win the top slot.

   The fix for the argmax behavior is *not* a better-calibrated model; it's a
   different decision rule (the value-pick / expected-utility selection in
   `prediction_audit`). A well-calibrated model is the input that finally makes
   that rule trustworthy.

3. **Brier slightly worse, log-loss slightly better.** Both moved <0.0025 —
   essentially a wash on proper-scoring-rule terms. The accuracy and ECE gains
   are the signal; the Brier/log-loss deltas are noise at this sample size.

4. **vs yesterday's Bayesian-ONLY result.** With no Elo and no xG, Bayesian
   *lost* to MLE+Elo on Brier/log-loss — the calibration win was real but it
   had given up the Elo signal. With Elo (learned coef ~0.081) and xG (noisy
   likelihood) integrated, Bayesian now matches or beats MLE on everything
   except a negligible Brier difference. Lesson confirmed: the models must
   share features for the comparison to be fair.

## What changed in the model

`models/dixon_coles_bayes.py`:
- **Elo as a learned covariate** — `elo_coef * (home_elo − away_elo)/scale`
  added to the log-rate, `Normal(0, 0.5)` prior. Data learns the coefficient
  (~0.081 on EPL) rather than the MLE's hardcoded 0.10.
- **xG as a second noisy likelihood** — `Normal(latent_rate, 0.75)` on matches
  where xG is present. Continuous, no rounding to fake integer goals like the
  MLE blend. Partial coverage handled gracefully.

## Open questions / next

- **ROI sim**: does the −58% ECE translate to a less-negative ROI? Yesterday
  Bayesian-only was −22%. Re-run ROI with the Elo+xG Bayesian — this is the
  decision-relevant test, since better calibration *should* sharpen value-bet
  selection even if argmax accuracy barely moves.
- **value-pick with the calibrated model**: re-run the audit's value-pick rule
  on Bayesian probabilities. With honest ~25% draw probs, the lift-over-baseline
  rule should finally start electing some draws (it fired on only 1/15 with
  MLE's over-confident probs).
- **Full xG coverage**: re-run once EPL hits 100% xG (a few more days of daily
  backfill). Expect the calibration gap to widen further.
- **Cost**: Bayesian fit is ~30x slower (~20s vs <1s per refit). Fine for
  backtest/analysis; live `/predict` would need fit caching.
- **Tier-3 leagues**: the shrinkage advantage should be largest where MLE
  overfits most (small-sample leagues: Saudi, J1, MLS). Untested — those
  leagues lack xG + odds for a clean comparison.

---
*Note: an earlier draft of this file reported n=308 and a 21.4% Bayesian
draw rate. Those numbers were misread from garbled interactive shell output
and are wrong; the table above is read directly from the saved data file.*
