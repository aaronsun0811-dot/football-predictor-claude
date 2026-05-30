# ROI head-to-head — calibration does NOT buy ROI

Decision-relevant test: does the Bayesian model's calibration win (ECE −58%,
see `bayesian_vs_mle.md`) translate into a better betting ROI? **No. The
opposite.** This result overturned the working assumption of the prior few days.

## Setup

- League: `premier_league`, `min_train_matches=600`, `refit_every=25`
- Walk-forward value-betting vs Bet365 closing odds, half-Kelly, Shin de-vig
- xG coverage ~50% of 2024-25 (backfill ongoing)
- Numbers read from saved file `roi_head_to_head_data.json`, not shell stdout
  (this session had unreliable terminal display)

## Result (verified 2026-05)

| model            | ROI      | bets | bet-rate | hit-rate | ending (start=100) | max DD |
|------------------|----------|------|----------|----------|--------------------|--------|
| **MLE+Elo+xG**   | **−3.38%** | 572  | 73.4%    | 35.3%    | 53.96              | 74%    |
| Bayesian+Elo+xG  | −31.09%  | 689  | 88.4%    | 20.3%    | **0.06** (busted)  | 99.9%  |
| Market-Fused     | −12.01%  | 319  | 40.9%    | 29.8%    | 34.70              | 71%    |

## The counterintuitive finding

The **better-calibrated** model (Bayesian, ECE 0.0151) produced the **worst**
ROI by far — it essentially went bankrupt (100 → 0.06). Meanwhile MLE+Elo+xG,
with worse calibration (ECE 0.0358), produced the best ROI ever recorded in
this project: **−3.38%**, up from the old −17% (MLE+Elo, no xG) and −15.4%
(Market-Fused) baselines.

### Why calibration ≠ profitability

1. **Flat probabilities look like value everywhere.** Bayesian's hierarchical
   shrinkage pulls probabilities toward league means, making them flatter. A
   flatter model disagrees with the sharp market more often, so it thinks it
   sees an edge constantly — bet-rate 88.4% vs MLE's 73.4%.

2. **But it's not sharper than the market.** Good calibration means "when it
   says 25%, it happens 25% of the time" — internal self-consistency. Value
   betting requires being *more accurate than the bookmaker*, which is a
   different and much higher bar. Bayesian's hit-rate collapsed to 20.3% (vs
   MLE 35.3%): the bets it placed were mostly cases where it disagreed with the
   market and the market was right.

3. **Calibration and profitability are different objectives.** A model can be
   beautifully calibrated and still lose money fast if its disagreements with
   the market are noise rather than signal. The −58% ECE win is real for
   *prediction* tasks (the audit accuracy panel, honest probability display);
   it is actively harmful for *betting* when the flatness manufactures false
   edges.

## What actually moved ROI: xG

The headline that matters for the project's stated goal:

- MLE + Elo (no xG), from the older shootout: **−17.3%**
- MLE + Elo + xG (this run): **−3.38%**

Adding xG to the MLE model closed most of the gap to break-even. **xG is the
real lever**, not the Bayesian machinery. This is consistent with the ablation
finding (xG net-positive on accuracy + calibration) and now extends it to ROI.

Caveat: xG coverage is only ~50% of the season here. The −3.38% may shift
(either direction) once coverage hits 100%. Re-run required before trusting the
exact figure — but the *direction* (xG materially helps ROI) is robust.

## Revised roadmap implication

- **Drop the assumption that Bayesian is the ROI path.** It's a better
  *predictor* (accuracy, calibration) but a worse *bettor*. Keep it for the
  prediction/audit surfaces if anywhere, not for ROI.
- **Double down on xG.** Get EPL to 100% coverage, then extend xG backfill to
  other leagues with odds (La Liga, Bundesliga, Serie A). Re-run ROI per league.
- **Bayesian-only-for-Tier-3 hypothesis still untested.** The shrinkage should
  help most where MLE overfits (small-sample leagues). But those leagues also
  lack odds, so "ROI" can't be measured there — only accuracy/calibration. The
  value of Bayesian there would be better *predictions*, not betting.
- **Best ROI config to date: MLE + Elo + xG, −3.38%.** Still negative (the
  market is efficient + vig), but the closest this project has come to
  break-even. Profitability would still require information the market lacks
  (injuries, lineups) — out of scope.
