# football-predictor

[English](README.md) · [中文](README.zh.md)

Team-level football match predictor. Dixon-Coles bivariate Poisson with Elo
prior correction, fed by free public data sources. The web UI is fully
bilingual (English / 中文). Three surfaces:

- **Web UI** — `python predict.py serve` → open <http://localhost:8001>
  (single-page app: match predict / World Cup / leagues / about; no build step)
- **CLI** — `python predict.py predict "Arsenal" "Chelsea" --league premier_league`
- **HTTP API** — `python predict.py serve` exposes FastAPI on :8001

The current prediction model is team-level. Player tables are included so paid
feeds can store squad age, minutes, starts, goals, assists, and ratings, but
the first production model deliberately stays team-first because it is easier
to calibrate and harder to overfit.

Read [ARCHITECTURE.md](ARCHITECTURE.md) for the project boundaries, data
contract, and foundation rules.

## Coverage

| Tier | Data | Leagues |
|------|------|---------|
| **1** | Results + ClubElo (+ optional xG) | EPL, Championship, La Liga, Segunda, Serie A/B, Bundesliga 1/2, Ligue 1/2, Eredivisie, Primeira, Belgian Pro |
| **2** | Results + ClubElo | EFL League One |
| **3** | Results only or partial | Saudi Pro, J1, K1, Chinese Super League, MLS, Brasileirão, Liga Argentina, Liga MX, Liga Portugal 2 |

Tier 3 leagues have no free ClubElo or xG. Predictions still work via the
goals-only Dixon-Coles fit and the internal Elo builder, but calibration is
roughly 5-8 pp worse than Tier 1.

API-Football support is optional. If `FOOTBALL_API_KEY` or `API_FOOTBALL_KEY`
is set, the updater can fetch fixtures and current-season player stats for
configured leagues.

**World Cup 2026** uses national-team Elo from eloratings.net. Full 48-team
Monte Carlo (12 groups × 4 + best 8 thirds → R32 → final).

## Install

```bash
cd ~/Documents/football-predictor
python3.11 -m venv .venv         # Python 3.10+ required for scipy>=1.14
source .venv/bin/activate
pip install -r requirements.txt
python predict.py init-db
```

## Fetch data

```bash
# Everything (5y results + today's club Elo + national Elo):
python predict.py update

# Just one league. The `--league` flag accepts league keys directly,
# or Chinese aliases (see the "Chinese aliases" section below).
python predict.py update --league premier_league
python predict.py update --league saudi_pro
python predict.py update --league chinese_super_league

# Optional API-Football path:
cp .env.example .env
# edit FOOTBALL_API_KEY
python predict.py doctor --live
python predict.py update --league saudi_pro --include-api-football
python predict.py update --league chinese_super_league --include-api-football
python predict.py update --league premier_league --include-api-football --include-players

# xG enrichment (see "xG sources" section below for details — the FBref
# path has been blocked, API-Football is the working alternative):
python predict.py backfill-api-xg --league premier_league --season 2024 --limit 90

# Verify API-Football league IDs before relying on them:
python predict.py api-football-leagues --country China --search "Super League"
```

Data lands in `data/football.sqlite3`. Cache CSVs land under `data/cache/`.

The HTTP API also exposes `POST /update` (background task), `GET /leagues`,
`GET /coverage`, `GET /doctor`, `POST /backtest`, and
`GET /export/{matches|ratings|players|player_season_stats|update_state}`.

## xG sources

The Dixon-Coles model accepts per-match expected goals (`home_xg`, `away_xg`).
Two scrapers exist; one currently doesn't work:

| Source | Path | Status |
|--------|------|--------|
| API-Football | `predict.py backfill-api-xg --league <key> --season <year>` | **Working** |
| FBref | `predict.py update --league <key> --include-xg` | **Blocked** — FBref returns 403 to our IP |

**Recommended**: use `backfill-api-xg`. API-Football's `/fixtures/statistics`
endpoint exposes per-team `expected_goals`. One HTTP request per fixture, so a
full EPL season (~380 fixtures) takes ~4 days at the free-tier 100-req/day
quota — or one shot on a paid plan. A launchd plist
(`deploy/com.aaronsun.football-predictor-claude.api-xg-backfill.plist`) fires
a 90-fixture batch daily at CST 08:01 to make this hands-off.

The FBref scraper code still lives in `scrape/fbref.py` and is unchanged, but
FBref now blocks our IP at the network layer (403 on every request, including
the homepage; not a UA or rate-limit issue). Re-enabling it would need either
a residential-proxy service or a headless-browser bypass — neither is in
scope. To verify whether xG is actually populated for a league, hit
`/diagnostics/ablation` and look for the `silent_features` warning array.

## Chinese aliases

The `--league` flag accepts Chinese names in place of league keys. Useful when
you don't want to remember English slugs.

| Alias        | League key             |
|--------------|------------------------|
| `英超`       | `premier_league`       |
| `西甲`       | `la_liga`              |
| `德甲`       | `bundesliga`           |
| `意甲`       | `serie_a`              |
| `法甲`       | `ligue_1`              |
| `中超`       | `chinese_super_league` |
| `沙特`       | `saudi_pro`            |
| `世界杯`     | `world_cup`            |

Full list in `scrape/registry.py::EXTRA_ALIASES`. The web UI's league dropdown
shows both forms automatically based on the language toggle.

## Predict a match

```bash
# Club match. League is optional but improves the fit.
python predict.py predict "Arsenal" "Chelsea" --league premier_league

# Neutral venue, knockout (advancement probability instead of pure draw).
python predict.py predict "Real Madrid" "Bayern Munich" \
  --neutral-site --stage "quarter-final"

# International (uses national Elo).
python predict.py predict "Brazil" "Argentina" --league world_cup
```

Output (CLI): JSON with `probabilities {home_win, draw, away_win}`,
`expected_goals`, `most_likely_scores`, full `score_matrix`, and the
training metadata.

When ClubElo/national Elo is missing, the service builds a leakage-safe
internal Elo from the historical matches already in SQLite.

## Backtest

```bash
# Walk-forward test on one league.
python predict.py backtest --league premier_league --min-train-matches 120 --refit-every 5
python predict.py backtest --league premier_league --include-predictions  # verbose

# See which leagues currently have usable data.
python predict.py coverage
python predict.py coverage --only-empty
python predict.py doctor
python predict.py doctor --live

# Export raw tables for notebooks.
python predict.py export matches -o data/exports/matches.csv
python predict.py export player_season_stats -o data/exports/player_season_stats.csv
```

Backtest output includes 1X2 accuracy, multi-class Brier score, multi-class
log loss, and realized home/draw/away rates.

## World Cup 2026

```bash
# Synthetic top-48 draw (deterministic).
python worldcup.py --n-sims 20000

# With the real draw once it's published. Example draw.json:
# { "A": ["United States", "Mexico", "Egypt", "Iran"], "B": [...] }
python worldcup.py --groups data/wc2026_draw.json
```

Outputs R16 / QF / SF / final / champion probabilities per team.
Names must match `eloratings.net` spellings ("United States", "South Korea").

## Run as a service

```bash
python predict.py serve --port 8001
```

Open <http://localhost:8001> for the web UI, or hit the JSON endpoints directly:

| Method | Path                          | Purpose                                    |
|--------|-------------------------------|--------------------------------------------|
| GET    | `/health`                     | Liveness + scheduler status                |
| GET    | `/stats`                      | Match / team / Elo counts                  |
| GET    | `/coverage`                   | Per-league data coverage report            |
| GET    | `/doctor`                     | Data-source readiness and next commands    |
| GET    | `/leagues`                    | League registry                            |
| GET    | `/teams?league=<key>`         | Distinct teams (used by autocomplete)      |
| GET    | `/recent?league=<key>&limit=` | Most recent matches                        |
| POST   | `/predict`                    | Single-match Dixon-Coles + Elo prediction  |
| GET    | `/worldcup/forecast?n_sims=`  | World Cup 2026 Monte Carlo (cached)        |
| POST   | `/backtest`                   | Rolling-origin backtest of the model       |
| POST   | `/update`                     | Trigger an incremental refresh (background)|
| GET    | `/export/{table}`             | CSV dump of SQLite tables                  |
| GET    | `/api-football/leagues`       | API-Football league ID discovery           |

The service also starts an APScheduler job that runs `update_all` daily at
03:30 Asia/Shanghai. Disable with `FOOTBALL_PREDICTOR_ENABLE_SCHEDULER=false`.
Set `FOOTBALL_PREDICTOR_DAILY_API_FOOTBALL=true` to also pull API-Football
fixtures in the daily run. Keep this off unless your API quota is comfortable.
`FOOTBALL_PREDICTOR_DAILY_FBREF_XG=true` used to enrich the daily run with
FBref xG. FBref has since blocked our IP — see the "xG sources" section above.
Use `predict.py backfill-api-xg` instead.

`python predict.py doctor` should stay green before adding model features. It
checks the SQLite schema, configured data sources, coverage gaps, and the next
commands to run.

## Test

```bash
python -m pytest
```

The tests cover Chinese league aliases, SQLite upsert/de-duplication,
Dixon-Coles probability normalization, xG blending, FBref xG merge behavior,
internal Elo generation, walk-forward backtest metrics, and API-Football
player-stat normalization.

## What it can't do

- **Predict individual player performance.** Storage exists, but the current
  model does not yet use player features.
- **Automatically account for injuries, suspensions, manager changes.** Strength ratings
  catch up with a 4-6 game lag.
- **Beat closing bookmaker lines consistently.** A good Dixon-Coles model
  is ~52-55% accurate on 3-class outcomes for top leagues — the same
  ballpark as efficient markets. Use probabilities to find value, not certainty.

## Files

```
config/leagues.yaml         League registry: codes, tiers, Chinese aliases.
data/database.py            SQLAlchemy ORM (matches, ratings, players, update state).
models/elo.py               Internal leakage-safe Elo builder.
models/backtest.py          Walk-forward 1X2 backtest metrics.
models/dixon_coles.py       Dixon-Coles + Elo-adjusted bivariate Poisson.
scrape/registry.py          LeagueRegistry + EXTRA_ALIASES (Chinese aliases).
scrape/update.py            IncrementalUpdater orchestrator.
scrape/api_football.py      Optional API-Football fixtures + player stats.
scrape/clubelo.py           ClubElo daily snapshot.
scrape/football_data.py     Historical results (football-data.co.uk).
scrape/eloratings.py        National-team Elo.
scrape/fbref.py             Optional FBref xG enrichment (rate-limited).
predict.py                  FastAPI app + typer CLI (update / predict / serve / export).
worldcup.py                 48-team Monte Carlo simulator.
```

## Politeness

- ClubElo cached per day; never refetched the same day.
- football-data.co.uk seasons cached after first fetch; only the in-progress
  season is re-pulled.
- FBref aggressively rate-limits (10 req/min). The scraper sleeps 6.5 s
  between calls and skips silently if blocked.

Don't run `update` in a tight loop. Once per day is enough.
