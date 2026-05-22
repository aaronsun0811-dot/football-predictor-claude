# football-predictor Architecture

This project is intentionally built as a data pipeline plus service, not a
single script. The goal is to keep the foundation stable while data sources
and model features evolve.

## Layers

- `config/`: runtime settings and league registry.
- `scrape/`: source adapters. Each adapter normalizes external data into the
  internal match/player shape.
- `data/`: persistence, coverage reports, schema checks, and doctor reports.
- `models/`: prediction, internal Elo, and walk-forward backtesting.
- `predict.py`: FastAPI service and Typer CLI entrypoint.
- `static/`: no-build web UI for local operation and inspection.

## Data Contract

All match-like sources must normalize to:

- `date`
- `league_key`
- `home_team`
- `away_team`
- `home_goals`
- `away_goals`
- optional `home_xg`, `away_xg`, `home_elo`, `away_elo`

`league_key` must always come from `LeagueRegistry`; aliases such as `英超`,
`沙特`, and `中超` are accepted only at the edges.

## Foundation Rules

- Settings are centralized in `config/settings.py`.
- SQLite schema is checked through `data/schema.py` every time `Database.init()`
  runs. Safe nullable columns are added automatically for old local databases.
- `python predict.py doctor` is the first command to run when something feels
  wrong. It checks schema, API-Football readiness, coverage gaps, and next
  commands.
- Prediction code must not call external APIs. It only reads normalized local
  data from SQLite.
- Backtests must be walk-forward: no future match is allowed in training data.

## Operational Baseline

```bash
python predict.py doctor
python predict.py update
python predict.py backtest --league 英超 --min-train-matches 120 --refit-every 25
python predict.py serve --port 8000
```

For paid data:

```bash
cp .env.example .env
# set FOOTBALL_API_KEY
python predict.py doctor --live
python predict.py update --league 中超 --include-api-football --years-back 3
```
