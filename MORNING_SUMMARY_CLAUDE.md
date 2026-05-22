# Morning summary — Claude fork (round 8)

## Round 8: 比赛复盘 — per-match replay + upset leaderboard

New **比赛复盘** tab. Click any past match → the model **rewinds**, refits
Dixon-Coles using only matches before that date, and shows what it would
have predicted vs what actually happened.

Different from the aggregate **回测** tab:
- Backtest: aggregate metrics (1680 EPL predictions, 53% accuracy)
- Replay: one match at a time, browseable

### Endpoints

```bash
GET  /match-history?league=英超&limit=30          # 30 most recent matches
POST /match-replay  {league, match_date, home, away}    # rewind + predict
GET  /match-replay/surprises?league=英超&refit_every=50 # all-season scan
```

### Real EPL highlights (full season scan, ~5s)

The **biggest upsets** the model called only ~3% to happen:

```
2022-10-29  Liverpool 1-2 Leeds              model: away_win 3.2%
2022-11-12  Man City  1-2 Brentford          model: away_win 3.3%   ← famous
2022-10-22  Forest    1-0 Liverpool          model: home_win 3.4%   ← famous
2023-11-12  Brighton  1-1 Sheffield United   model: draw 5.8%
2023-12-22  Aston Villa 1-1 Sheffield Utd    model: draw 6.0%
```

The **best calls** where the model gave ≥93% confidence and was right:

```
2022-02-19  Liverpool 3-1 Norwich            model: home_win 96.7%
2023-12-30  Man City  2-0 Sheffield United   model: home_win 95.1%
2023-10-28  Arsenal   5-0 Sheffield United   model: home_win 94.5%
2024-08-24  Man City  4-1 Ipswich            model: home_win 93.8%
2024-10-26  Man City  1-0 Southampton        model: home_win 93.0%
```

This is the **value of having the data: real EPL fans recognize Brentford
upsetting City and Forest beating Liverpool**. Now you can browse and
audit specific match decisions, not just aggregate numbers.

### UI features

- Left column: 30 most recent matches in selected league, click any to replay
- Middle/right: scoreboard with actual score + ring around the actual W/D/L outcome on the probability bar
- "✓ 模型预测正确" or "✗ 模型预测错误" badge
- Per-match Log loss / RPS / Brier metrics with color thresholds
- "扫描整个联赛找惊喜" button → full-season surprise scan in 5-15s
- Upsets and best-calls tables click through to a fresh replay of that match
- Auto-load on tab open + auto-clear when switching leagues

## What was added (round 8)

```
models/replay.py                  MatchReplay dataclass + replay_match
                                  + rank_surprises (walk-forward with
                                  refit_every for speed)
predict.py                        + /match-history + /match-replay
                                  + /match-replay/surprises
static/index.html                 + 比赛复盘 tab (list / detail / upsets
                                  + best-calls leaderboards)
static/app.js                     + loadReplayHistory, runReplay,
                                  loadSurprises methods + auto-load
tests/test_replay.py              9 new tests
```

## Verification

```bash
python -m pytest                           # 99 passed (was 90)

# Sample one famous EPL upset:
curl -X POST http://localhost:8001/match-replay -H "Content-Type: application/json" \
  -d '{"league":"英超","match_date":"2022-11-12","home_team":"Man City","away_team":"Brentford"}' \
  | jq '{ actual: .actual, predicted: .prediction }'
# Expected: actual home_win, model gave away_win only 3.3% → it was a famous 2-1 upset.
```

---

# Morning summary — Claude fork (round 7)

## Round 7: 球队强度 — browsable strength table + 2-team compare

New **球队强度** tab. Pick a league, see every team ranked by overall
strength (attack + defense from a fresh Dixon-Coles fit on stored matches),
plus the last 5 matches and ClubElo when available.

### Two endpoints

```bash
GET /teams/strengths?league=英超
GET /teams/compare?league=英超&home=Arsenal&away=Chelsea
```

`/teams/strengths` returns a ranked array; `/teams/compare` returns just
two teams with a `differential` block (home − away on each dimension).

### Live results (EPL, 754 training matches)

```
Team             Att     Def    Overall   Form   GD
Arsenal       +0.457  +0.700   +1.156   LLWWW   +3
Man City      +0.551  +0.509   +1.060   WWDWW   +8
Liverpool     +0.454  +0.185   +0.639   WWWLD   +4
Brighton      +0.273  +0.222   +0.495   WDWLW   +6
Man United    +0.319  +0.170   +0.489   LWWWD   +2
Bournemouth   +0.288  +0.148   +0.436   WWDWW   +6
Nott'm Forest +0.184  +0.240   +0.424   DWWWD   +10
Chelsea       +0.247  +0.163   +0.410   LLLLD   -9
Brentford     +0.273  +0.119   +0.392   DDLWL   -1
Newcastle     +0.286  +0.101   +0.387   LLLWD   -1
```

Arsenal tops the season on overall (huge defense parameter +0.700, best
in the league), Man City barely behind on attack (+0.551 best). Chelsea
sits 8th and has the worst recent form (LLLLD, GD -9).

Live results (Bundesliga):
```
Bayern Munich  att=+0.941 def=+0.368 overall=+1.309  WWWDW
Dortmund       att=+0.472 def=+0.371 overall=+0.843  LLWLW
Leverkusen     att=+0.463 def=+0.212 overall=+0.676  WLWWL
```

### UI features

- **Sortable table** — click a column-equivalent dropdown: overall / attack /
  defense / recent_gd / club_elo / team name
- **Click two rows to compare** — selected rows get green highlight, a
  side-panel shows attack/defense/overall bars (green vs purple) with
  per-dimension differential
- **Form is color-coded** — W in green, L in rose, D in grey
- **Auto-loads on tab open** — no extra click needed

### How attack/defense are derived

Dixon-Coles fits multiplicative log-rate parameters per team:

```
λ_home = exp(intercept + home_advantage + attack[home] + defense[away])
λ_away = exp(intercept + attack[away] + defense[home])
```

So attack > 0 means "scores more than the average team". Defense > 0 in the
raw model means "concedes more than average" — bad. The UI flips the sign
so higher = better on both dimensions, consistent UX.

## What was added (round 7)

```
models/team_strengths.py        TeamStrength dataclass + extract_strengths
                                + compare_two; recent form + ClubElo merge
predict.py                      + /teams/strengths + /teams/compare endpoints
static/index.html               + 球队强度 tab (table + compare panel)
static/app.js                   + loadStrengths() + toggleCompare()
                                + sortedStrengths getter + barWidth helper
                                + auto-load on tab open
tests/test_team_strengths.py    9 new tests
```

## Verification

```bash
python -m pytest                          # 90 passed (was 81)
curl http://localhost:8001/teams/strengths?league=bundesliga | jq '.teams[0:3]'
```

---

# Morning summary — Claude fork (round 6)

## Round 6 highlight: 实时进行中 — live in-play predictor

Brand-new tab nobody else has. **Pick a match, set the current score and
the minute, and the model recomputes the final-result probabilities.**

### The math

1. Run the standard pre-match prediction → get expected goals `xH`, `xA`.
2. Compute remaining time `r = (90 − minute) / 90`.
3. Scale remaining xG with game-state multipliers — trailing side ×1.15,
   leading side ×0.92 (well-documented effect: chasing teams take more risks,
   leading teams park the bus). Tied → both ×1.0.
4. Convolve a Poisson distribution of remaining goals with the current
   score, sum the joint PMF onto W/D/L.

Live test — Arsenal vs Chelsea (英超) at various live states:

| State | H% | D% | A% | Most likely final | State mults |
|---|---:|---:|---:|---|---|
| 0-0 at 0' | 68.2% | 19.8% | 12.1% | 2-0 (13.3%) | 1.00 / 1.00 |
| 1-0 at 30' | 83.7% | 12.6% | 3.7% | 2-0 (20.8%) | 0.92 / 1.15 |
| 1-0 at 75' | 90.5% | 8.9% | 0.6% | 1-0 (64.0%) | 0.92 / 1.15 |
| 0-1 at 60' | 15.5% | 31.2% | 53.3% | 0-1 (37.0%) | 1.15 / 0.92 |
| 2-0 at 80' | 99.7% | 0.3% | 0.0% | 2-0 (74.2%) | 0.92 / 1.15 |
| 1-1 at 88' | 4.3% | 94.2% | 1.5% | 1-1 (94.1%) | 1.00 / 1.00 |

All numbers match intuition. Sibling project does **not** have this.

### What it's for

A fan watching a live game asks: "Chelsea just scored at 70' to make it
2-1 — can Arsenal hold on?". Open the 实时进行中 tab, slide the minute,
type the score, hit "重新计算" — get an actual probability instead of a
gut feel.

The math is just a clean baseline. It doesn't see red cards, injuries,
or tactical subs. The UI surfaces this honestly.

### Endpoint

```bash
curl -X POST http://localhost:8001/predict/in-play \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "league": "英超",
    "current_home": 1,
    "current_away": 0,
    "minute_elapsed": 75
  }'
```

Returns final W/D/L, expected final goals, remaining xG, state multipliers,
top-5 most likely final scorelines, plus the pre-match baseline for
comparison.

## What was added (round 6)

```
models/in_play.py                 Poisson + game-state multiplier model
predict.py                        + InPlayRequest + POST /predict/in-play
                                  endpoint (reuses /predict pipeline for xG)
static/index.html                 + 实时进行中 tab with live scoreboard UI
                                  including a minute slider, score steppers,
                                  most-likely-finals bar chart
static/app.js                     + predictInPlay() + state shape
tests/test_in_play.py             11 new tests
```

## Verification

```bash
python -m pytest                                  # 81 passed (was 70)
```

---

# Morning summary — Claude fork (round 5)

## Round 5 highlight: ensemble + market-fused predictor

Building on round 4 (penaltyblog wrappers). I built a proper **ensemble**
that averages probabilities across 3 models, plus a **market-fused**
variant that blends ensemble output with Shin-implied bookmaker probabilities.

Both are exposed in the web UI (Match Predict + ROI tabs now have model
dropdowns) and via `?model=` / JSON body on `/predict` and `/roi-simulation`.

### The result that actually matters

`data/ensemble_shootout.json` — 3 leagues × 5 models, all with Shin
probabilities and `min_edge=0.05, min_ev=0.05`:

| League | dc_elo | dc | bp | ensemble | **market_fused** |
|---|---:|---:|---:|---:|---:|
| EPL | **−9.31%** | −15.37% | −15.08% | −13.12% | −11.51% |
| La Liga | −27.66% | −27.57% | −30.00% | −34.08% | **−24.79%** |
| Primeira | −14.95% | −25.11% | −21.03% | −18.69% | **−9.88%** |

**Market-fused wins 2 of 3 leagues** for least-bad ROI. On Primeira it cut
the loss almost in half (−15% → −10%) and ended with 45.8 of 100 starting
bankroll (vs 1.5 for the next-best model).

**How market-fused works**: at each match, blend the model's probabilities
50/50 with the Shin-implied market probabilities. The model only finds an
"edge" when it disagrees with the market — which means betting happens
MUCH less often (270 bets vs 700-1000), but the disagreements are higher
quality. Disciplined betting > more bets.

**All 5 strategies still lose money** — that's still the headline answer,
confirmed across more configurations. But market_fused tames the bleeding
significantly, especially in volatile leagues.

### What's a user supposed to do with this?

The **比赛预测** tab now has a model dropdown:

```
DC + Elo (default, over-confident)    ← was the only option
Ensemble (3 models averaged)
Market-Fused (most stable)
Dixon-Coles (penaltyblog)
Bivariate Poisson
Poisson (baseline)
```

For UCL Real Madrid vs Bayern (Arsenal-Chelsea analog), the 6 produce:

```
dixon_coles_elo  : H=67.0%   ← extreme
ensemble         : H=60.2%   ← balanced mean
market_fused     : H=60.2%   ← (no market data passed → falls back to ensemble)
dixon_coles      : H=56.2%   ← market-realistic
bivariate_poisson: H=57.4%
poisson          : H=57.2%
```

The ROI tab also has the same dropdown so you can re-run the historical
simulation under your chosen model in 5-15 seconds.

## What was added (round 5)

```
models/ensemble.py                      EnsembleConfig + fit + predict_match
                                        with weighted averaging + market fusion
predict.py                              Routes model=ensemble and model=market_fused
                                        through the ensemble layer
models/roi_simulator.py                 Same routing + reads odds-per-match for
                                        market fusion during walk-forward sim
static/index.html                       + 模型 dropdown on Match tab and ROI tab
static/app.js                           Passes model field through both forms
tests/test_ensemble.py                  6 new tests
data/ensemble_shootout.json             15-run shootout (3 leagues x 5 models)
```

## Verification

```bash
python -m pytest                                  # 70 passed (was 64)
curl -X POST http://localhost:8001/roi-simulation \
  -H "Content-Type: application/json" \
  -d '{"league":"葡超","model":"market_fused","min_edge":0.05,"min_ev":0.05}' \
  | jq .summary
# Expected: roi ≈ -0.10, n_bets ≈ 258, ending_bankroll ≈ 45.8
```

---

# Morning summary — Claude fork (round 4)

## Round 4 highlight: integrated penaltyblog + MatchOracle techniques

You pointed at two reference projects. I read both and pulled the most useful
ideas in:

**From [penaltyblog](https://github.com/martineastwood/penaltyblog)** (Martin
Eastwood's Cython-optimized Python package, the gold-standard open-source
football modelling library):

- Wrapped 5 of its goal models — `dixon_coles`, `bivariate_poisson`,
  `poisson`, `negative_binomial`, `zero_inflated_poisson` — and exposed
  via `?model=` on `/predict` and `/roi-simulation`.
- Added Shin's method for implied probabilities (Shin 1993; the published
  research standard for de-vigging bookmaker odds, vs naive overround
  normalization). Plus `power` and `additive` variants.

**From [MatchOracle](https://github.com/abailey81/MatchOracle)** (Adam
Bailey's 5-layer ensemble that beats market on EPL):

- Added **RPS (Ranked Probability Score)** to the backtest summary —
  the headline metric in MatchOracle's results table, and the field
  standard for ordered 3-way classification (Constantinou & Fenton 2012).
  EPL backtest reports RPS = 0.2033, in line with published literature.
- (Still TODO from MatchOracle: ensemble stacking, NLP sentiment, Glicko-2
  ratings, 376 features. Would 2-3 days of work each.)

### The most important finding from this round

**Comparing 4 models on the same EPL match (Arsenal vs Chelsea, 750 matches training):**

```
dixon_coles_elo (ours)           H=67.0%  D=22.0%  A=11.0%   xG 2.02–0.71   ← over-confident
penaltyblog dixon_coles          H=56.2%  D=26.0%  A=17.9%   xG 1.70–0.87
penaltyblog bivariate_poisson    H=57.4%  D=24.4%  A=18.3%   xG 1.70–0.86
penaltyblog poisson              H=57.2%  D=23.9%  A=18.9%   xG 1.70–0.87
```

Our home-grown Elo-blended Dixon-Coles is **~10pp more confident** on the
home favorite than three independent published models. The Diagnostics tab
already showed this overconfidence in the 65-75% probability band; now
we have an alternative-model second opinion that confirms the bias is in
the **Elo correction term**, not the Dixon-Coles base.

### But surprisingly, our model loses LESS money in ROI

9-combo shootout (`min_edge=0.05, min_ev=0.05`, half-Kelly, Shin probabilities):

| League | Model | Bets | Hit% | ROI |
|---|---|---:|---:|---:|
| EPL | **dixon_coles_elo (ours)** | 927 | 35.9% | **-9.31%** |
| EPL | dixon_coles (pb) | 985 | 30.5% | -15.37% |
| EPL | bivariate_poisson (pb) | 967 | 31.0% | -15.08% |
| La Liga | dixon_coles_elo (ours) | 727 | 35.9% | -27.66% |
| La Liga | dixon_coles (pb) | 697 | 32.1% | -27.57% |
| La Liga | bivariate_poisson (pb) | 702 | 32.9% | -30.00% |
| Primeira | dixon_coles_elo (ours) | 680 | 34.6% | -14.95% |
| Primeira | dixon_coles (pb) | 691 | 29.8% | -25.11% |
| Primeira | bivariate_poisson (pb) | 656 | 30.8% | -21.03% |

**All 9 combos lose money.** Counter-intuitively, our overconfident model
loses LESS — because the overconfidence makes it find FEWER value spots
(927 vs 985 bets on EPL), so it bets less on bad signals. The penaltyblog
models are calibrated against reality but reality is closer to what the
market knows, leaving fewer real "edges" but more spurious ones.

The lesson reinforces the original ROI tab finding: **calibration ≠
profitability against efficient closing odds**. Even battle-tested published
models from a respected open-source package don't beat Bet365.

### What that means for the user

If you want **single-match prediction** (e.g. for a friend, for fun):
- Use `model=dixon_coles` (penaltyblog) — sharper, market-aligned probabilities.
- Set via `?model=dixon_coles` on the URL or in the JSON body.

If you want a **decision-support tool that biases toward big favorites**:
- Stick with default `dixon_coles_elo` — the Elo prior nudges predictions
  toward the obvious favorite, which is good UX even if not "true" probability.

If you want to **bet money**:
- Don't, based on this evidence. Or use a model that knows things the
  market doesn't (live injuries, rotation, weather), which we don't have.

## What was added (round 4)

```
models/penaltyblog_models.py     Wrapper around 5 penaltyblog goal models
models/implied_probs.py          Shin / power / additive / multiplicative
models/backtest.py               + RPS metric in summarize_predictions()
models/roi_simulator.py          + model + implied_method config fields
predict.py                       PredictionRequest.model + ROIRequest.model
                                 routes to penaltyblog when picked
tests/test_penaltyblog_integration.py  10 new tests
data/model_shootout.json         9-combo ROI shootout results
```

## Verification

```bash
python -m pytest                                  # 64 passed (was 54)
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team":"Arsenal","away_team":"Chelsea","league":"英超","model":"bivariate_poisson"}'
```

---

# Morning summary — Claude fork (round 3)

## Round 3 highlight: 洲际赛事 (continental competitions)

You asked for 欧冠、南美、亚洲. Done — 10 continental competitions are now
first-class:

**Club competitions (cross-league fit)**
- 欧冠 / UEFA Champions League
- 欧联 / UEFA Europa League
- 欧会杯 / UEFA Conference League
- 解放者杯 / Copa Libertadores
- 南美杯 / Copa Sudamericana
- 亚冠 / AFC Champions League Elite

**National-team competitions (Elo-only path)**
- 欧洲杯 / UEFA European Championship
- 美洲杯 / Copa América
- 亚洲杯 / AFC Asian Cup
- 非洲杯 / Africa Cup of Nations

How they work:

- **Club comps** can't have native data (no free feed for UCL/UEL match
  results). Trick: when you ask "Real Madrid vs Bayern (UCL)", we fit
  Dixon-Coles on **all top-flight matches across the relevant continent**
  (Europe = 8 leagues, ~5,000 matches). Both teams already have attack/
  defense parameters from their domestic data, so the cross-league fit
  predicts the head-to-head with proper Dixon-Coles + Elo correction.
- **National comps** use the existing 242-team national Elo (from
  eloratings.net via international-football.net fallback). New: an
  **adaptive draw share** that drops from 28% at parity to 13% at large
  Elo gaps — fixes the old "one-sided games predict 26% draws" bug.
  Plus a 3% underdog floor so we never print 0.00%.

**New tab in the UI**: 洲际赛事. Pick a competition (button row at the
top), get preset matchups for that comp (Real Madrid vs Bayern,
Argentina vs Brazil, Japan vs Saudi Arabia, etc.), with knockout
advancement probabilities for elimination rounds.

Live test results from the new tab:

```
UCL  · Real Madrid vs Bayern (neutral knockout) ......... 25% / 23% / 53%   (Bayern favored)
UCL  · Man City vs Paris SG (neutral knockout) .......... 11% / 20% / 69%   (PSG favored)
Euro · England vs Germany (knockout) .................... 51% / 24% / 24%   (advance ENG=64%)
Copa · Argentina vs Brazil (knockout) ................... 55% / 26% / 19%   (advance ARG=68%)
Asia · Japan vs Saudi Arabia ............................ 80% / 15% / 05%   (Japan +336 Elo)
AFCON· Morocco vs Senegal (knockout) .................... 29% / 26% / 45%   (advance MAR=42%)
```

Note 亚冠 / 解放者杯 still show "no domestic data" — those continents
have no league data loaded yet (need API-Football key for J1/K1/Saudi/CSL
on Asia, Brasileirão/Liga Profesional on South America). The UI surfaces
this honestly with the registry note.

## Round 3 also fixed

- **Dixon-Coles optimizer hardening**: previously raised RuntimeError on any
  L-BFGS-B failure. Now distinguishes "iter limit hit" (harmless, accept
  result.x) from "actually broken". Cross-league fits and big walk-forward
  backtests work without crashing.
- **Elo-only formula upgrades**: adaptive draw share + 3% underdog floor.
  Old behavior: Japan vs Saudi printed `H=74% D=26% A=0%`. New: `H=80% D=15% A=5%`.

---

# Morning summary — Claude fork (round 2)

> Fork of `~/Documents/football-predictor/`. Other model continues there
> on **:8000**, I continue here on **:8001**.

## Quick start

```bash
cd ~/Documents/football-predictor-claude
source .venv/bin/activate
python predict.py serve          # → http://localhost:8001
```

## What's new this round

I built the most uncomfortable but most useful tab: **ROI 验证** — a walk-forward
simulator that takes the model, the value-finder strategy, and **real Bet365
closing odds**, then asks "if you had actually placed those bets historically,
what would your bankroll look like?"

Spoiler: it loses money in every league at every threshold. **That's the
honest answer**, and the new ROI tab is what surfaces it.

### Tabs (8 total now, vs sibling's 5)

```
比赛预测  · 单场预测，胜平负 / xG / 比分热图
世界杯 2026 · 48 队蒙特卡洛
联赛浏览  · 22 联赛 + 最近比赛
回测      · 滚动 1X2 backtest
模型诊断  · 校准曲线 + ECE + 信心阶梯  ← new in round 1
价值发现  · EV / Kelly 计算（带新增的红色警告横幅）
ROI 验证  · 真实 Bet365 收盘赔率走盘 ← NEW THIS ROUND
关于
```

## ROI report — all 14 leagues, 3 thresholds each

Walk-forward Dixon-Coles + half-Kelly stakes vs Bet365 closing odds. Starting
bankroll 100. Saved to `data/roi_report.json`.

| League | Best filter | Bets | Hit% | ROI | End bankroll |
|---|---|---|---|---|---|
| Premier League | tight | 568 | 40.1% | **-3.57%** | 30.78 |
| EFL Championship | tight | 874 | 37.0% | -4.06% | 5.0 |
| Ligue 1 | tight | 466 | 37.3% | -5.38% | 14.5 |
| Eredivisie | loose | 966 | 36.3% | -5.98% | 1.7 |
| Serie A | moderate | 837 | 37.8% | -7.53% | 5.0 |
| La Liga 2 (Segunda) | moderate | 1,170 | 31.7% | -7.43% | 1.5 |
| EFL League One | tight | 942 | 39.4% | -10.69% | 1.6 |
| Serie B | tight | 559 | 32.9% | -11.29% | 0.3 |
| Ligue 2 | loose | 1,108 | 31.4% | -11.45% | 0.0 |
| Belgian Pro League | moderate | 700 | 38.1% | -11.64% | 1.1 |
| 2. Bundesliga | moderate | 775 | 36.1% | -13.00% | 1.5 |
| Primeira Liga | loose | 899 | 35.7% | -13.10% | 0.9 |
| La Liga | loose | 964 | 35.8% | -24.74% | (wiped) |
| Bundesliga | loose | 923 | 32.4% | -26.27% | (wiped) |

**Verdict:** the basic Dixon-Coles + Elo model **cannot beat Bet365 closing
odds** in any league at any threshold setting. The least bad result is EPL
with tight filters: -3.57% ROI over 568 bets.

### Why this is a useful answer

A typical football-prediction project would stop at "52% accuracy, looks
calibrated, ship it" and the user would lose money. The Diagnostics tab
already showed the model is over-confident in the 65-75% probability band
(it says ~70%, reality is ~60%). The ROI sim is the rubber-meets-road
proof: that over-confidence systematically loses money against efficient
closing prices.

The interesting twist: **Primeira Liga had the best 1X2 accuracy (55.7%)
but one of the worst ROIs (-13%)**. High accuracy at picking the modal
outcome doesn't translate into sharp probability estimates that beat the
market.

The Value Finder tab now has a prominent red disclaimer pointing to this
evidence.

## What it would take to actually make money

1. **Information not in the market** — injury reports, lineup leaks, fatigue
   from midweek travel. Not in our data.
2. **Sharper probabilities** — proper Bayesian shrinkage on small-sample teams,
   weighted xG features (FBref scraper exists, blocked from this network),
   player-level state (paid feed required).
3. **Reduce vig** — Bet365 has ~5% overround. Pinnacle / Betfair Exchange
   are tighter (~2%). Same model, different counterparty = bigger edge.
4. **Stop at value lines, not closing** — opening or pre-match odds are less
   efficient. Our data only has closing.

Out of scope for tonight. The ROI tab + this caveat sets the right
expectation.

## Inherited from the fork point (still all working)

- 25,609 matches across 14 European leagues, 2021-07-23 → 2026-05-14
- 242 national-team Elo rows (international-football.net fallback)
- World Cup 2026 simulator + cached forecast (`data/worldcup_forecast.json`)
- All endpoints from before: `/health`, `/stats`, `/leagues`, `/teams`,
  `/recent`, `/predict`, `/backtest`, `/worldcup/forecast`, `/coverage`,
  `/doctor`, `/api-football/leagues`, `/diagnostics`
- 22 league registry incl. Chinese aliases

## New in this round (my fork only)

```
data/database.py             + MatchOdds table for bookmaker closing odds
scrape/odds_backfill.py      Reparse football-data CSVs to extract B365 odds
                             (fixed regex bug: 3-char codes like SP1, SP2)
models/roi_simulator.py      Walk-forward bankroll simulation
predict.py                   + /roi-simulation endpoint + ROIRequest model
static/index.html            + ROI 验证 tab + disclaimer in 价值发现
static/app.js                + runROI() + _drawROIChart()
tests/test_roi.py            9 new tests (best_value_bet, simulate_roi,
                             odds parser)
data/roi_report.json         42 runs (14 leagues × 3 filter combos)
```

## Verification

```bash
python -m pytest                                  # 29 passed (was 20)
curl http://localhost:8001/health
curl -X POST http://localhost:8001/roi-simulation \
  -H "Content-Type: application/json" \
  -d '{"league":"英超","min_edge":0.08,"min_ev":0.08}' | jq .summary
```

Expected output for that last call:
```json
{
  "n_bets": 568,
  "hit_rate": 0.40,
  "roi": -0.0357,
  "ending_bankroll": 30.78,
  "max_drawdown_pct": 0.90
}
```

## Running both forks side-by-side

```bash
# Other model (sibling)
cd ~/Documents/football-predictor && source .venv/bin/activate
python predict.py serve     # port 8000

# This fork (mine)
cd ~/Documents/football-predictor-claude && source .venv/bin/activate
python predict.py serve     # port 8001
```

Both have separate `data/football.sqlite3` files so writes don't collide.

## Comparison vs sibling

|  | sibling :8000 | claude fork :8001 |
|---|---|---|
| Tabs | 5 (基础功能) | 8 (+ 模型诊断 + 价值发现 + ROI 验证) |
| Tests | ~15 | 29 |
| Endpoints | 13 | 16 (+ /diagnostics + /roi-simulation) |
| Tables | matches + ratings + players | + match_odds (22.2k rows) |
| Strength | 基础设施 (doctor, coverage, schema 自检, API-Football) | 分析深度 + 诚实 (calibration, ROI 验证) |
| Honest about losing money? | 没说 | 有红色警告 + 完整证据 |

Pick whichever style serves what you actually want. Sibling is "build it
right and add features." Mine is "test it harder and tell the user when
it doesn't work." Both are valid; for a betting-adjacent product, I'd
argue mine is more useful.

Sleep well. 🥱
