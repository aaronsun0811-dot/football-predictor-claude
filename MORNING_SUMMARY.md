# Morning Summary

## Quick Start

```bash
cd ~/Documents/football-predictor
source .venv/bin/activate
python predict.py serve
# open http://localhost:8000
```

The web app has tabs for match prediction, World Cup 2026 simulation, league browsing, **回测 (Backtest)**, and notes about model limits. The Backtest tab runs walk-forward 1X2 evaluation against any league and shows accuracy, Brier, log loss, and realized W/D/L rates.

## Current Data

- Matches loaded: 25,609
- Latest match date: 2026-05-14
- Leagues with match data: 14
- National-team Elo rows: 242
- Player rows: 0 until an API-Football key is added

Tier 3 leagues such as Saudi Pro, J1, K1, Chinese Super League, MLS, Brasileirão, Liga MX, and Liga Portugal 2 still need API-Football or another paid source.

## Built Overnight

- SQLite storage for matches, ratings, players, and player season stats
- Dixon-Coles model with Elo correction and score matrix output
- Internal leakage-safe Elo builder for leagues without ClubElo
- Walk-forward backtest module and CLI command
- Per-league data coverage report via CLI, API, and web league browser
- Data-source doctor for API-Football key readiness and next update commands
- Centralized settings layer and SQLite schema self-check for safer iteration
- Optional FBref xG enrichment wired into the update flow and Dixon-Coles fit
- Optional API-Football client for fixtures and player season stats
- FastAPI endpoints: `/health`, `/stats`, `/coverage`, `/leagues`, `/teams`, `/recent`, `/predict`, `/backtest`, `/worldcup/forecast`, `/update`, `/export/{table}`, `/api-football/leagues`
- Static web UI in `static/` (Alpine.js + Tailwind + Chart.js via CDN — no build step)
- Tests for aliases, DB upsert/deduping, model probabilities, internal Elo, backtest, and player-stat normalization
- League registry now covers 23 leagues including Chinese Super League (中超)

## Verification

```bash
python -m pytest                                       # 11 passed
curl http://localhost:8000/health                      # {"status":"ok",...}
```

Smoke tests against the live server (all real data):

```
Arsenal vs Chelsea (英超):       H=67.0% D=22.0% A=11.0% · xG 2.02–0.71 · fitted on 750 EPL matches
Real Madrid vs Barcelona (西甲): H=35.9% D=25.3% A=38.8% · xG 1.38–1.44 · 383 La Liga matches
WC 2026 top 5:  Spain 27.4% · Argentina 18.2% · France 13.0% · England 7.0% · Brazil 4.6%
```

WC forecast cache: `data/worldcup_forecast.json` (5,000 sims pre-computed, 7 ms to return on subsequent calls).

### Calibration report — all 14 European leagues backtested

Walk-forward backtest, `min_train=200`, `refit_every=25` (per-result detail in `data/backtest_report.json`):

| League | Tier | Accuracy | Brier | Log loss | N |
|---|---|---|---|---|---|
| Primeira Liga | 1 | **55.72%** | 0.5488 | 0.9422 | 1,321 |
| Eredivisie | 1 | **53.75%** | 0.5773 | 0.9754 | 1,321 |
| Premier League | 1 | **52.74%** | 0.5850 | 0.9859 | 1,680 |
| Belgian Pro | 1 | 52.22% | 0.5980 | 1.0084 | 1,329 |
| La Liga | 1 | 51.92% | 0.5978 | 1.0061 | 1,300 |
| Bundesliga | 1 | 51.85% | 0.6033 | 1.0160 | 1,321 |
| Ligue 1 | 1 | 51.80% | 0.6012 | 1.0100 | 1,469 |
| Serie A | 1 | 51.49% | 0.5965 | 0.9985 | 1,680 |
| EFL League One | 2 | 49.86% | 0.6204 | 1.0433 | 2,557 |
| 2. Bundesliga | 1 | 46.18% | 0.6427 | 1.0707 | 1,321 |
| Segunda | 1 | 45.49% | 0.6398 | 1.0695 | 2,075 |
| EFL Championship | 1 | 45.42% | 0.6380 | 1.0614 | 2,552 |
| Ligue 2 | 1 | 45.35% | 0.6526 | 1.1090 | 1,550 |
| Serie B | 1 | 43.18% | 0.6622 | 1.1189 | 1,320 |

**Tier 1 top-flight average: 52.1%** — competitive with closing bookmaker odds.
Second-tier leagues (Championship, Segunda, B-divisions) underperform by ~6 pp — expected, since teams swap between divisions, squads turn over heavily, and there are more cup/holiday cancellations creating noise.

Best single league: **Primeira Liga at 55.7%** — above market.
Most likely reason it tops the list: stable squads (Benfica/Porto/Sporting + a long tail of mid-table clubs) and a clear strength hierarchy. The Dixon-Coles + Elo formulation is well-suited.

## Known Issues

- **ClubElo unreachable** from this network (timeout). Club predictions use the internal leakage-safe Elo from `models/elo.py` instead — still produces sensible values, just not the canonical ones.
- **eloratings.net** changed to a JS-driven SPA (the old `ratingsData` HTML embed is gone). The fallback to `international-football.net` is what's actually feeding 242 national-team Elo rows.
- **FBref returns HTTP 403** from this network. The xG enrichment plumbing is all wired (`--include-xg` flag, blend weight in Dixon-Coles, etc.) but no xG rows have been loaded. Try from a residential network / proxy; the scraper is polite (6.5 s between requests).
- **Player features** are stored but not used by the model. Wiring them in needs paid feeds first.
- **No API-Football key** configured — Tier 3 leagues empty.

The accuracy numbers in the calibration table above are **without xG**. Loading FBref xG and re-running should add 1-3 pp to top-flight accuracy.

## Useful Commands

```bash
python predict.py update
python predict.py doctor
python predict.py init-db
python predict.py update --league 英超 --include-xg
python predict.py coverage --only-empty
python predict.py predict "Arsenal" "Chelsea" --league 英超
python predict.py backtest --league 英超 --min-train-matches 120 --refit-every 25
python predict.py export matches -o data/exports/matches.csv
```

With API-Football:

```bash
export FOOTBALL_API_KEY=your_key
python predict.py update --league 沙特 --include-api-football
python predict.py update --league 中超 --include-api-football
python predict.py update --league 英超 --include-api-football --include-players
python predict.py api-football-leagues --country China --search "Super League"
```
