# Fork: football-predictor-claude

This directory is a **fork** of `~/Documents/football-predictor/` taken at
**2026-05-16 01:33** so two AI agents can build the same product
independently for comparison.

- **`~/Documents/football-predictor/`** — the original. Another model
  continues working there. Server defaults to **:8000**.
- **`~/Documents/football-predictor-claude/`** — this fork. I (Claude)
  continue here exclusively. Server defaults to **:8001**.

Both started from the same code, same SQLite DB (25,609 matches, 242
national-team Elo rows), same web UI. From this point on the codebases
diverge — that's the whole point.

## How to run both side-by-side

```bash
# Other agent's version
cd ~/Documents/football-predictor && source .venv/bin/activate
python predict.py serve      # → http://localhost:8000

# This fork (mine)
cd ~/Documents/football-predictor-claude
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python predict.py serve      # → http://localhost:8001
```

Each directory has its own SQLite DB (`data/football.sqlite3`), so the two
servers don't write-conflict.

## What's the same right now

Everything copied at the fork point: the SQLite DB, all 14 league
backtests, the WC forecast cache, the 23-league registry, the static
web UI, the FastAPI endpoints, the 11 passing tests.

## What I plan to differentiate

(These are direction commitments; specifics evolve as I work.)

1. **Calibration plot tab** — reliability diagram in the web UI: bin
   model probabilities, plot actual frequency vs predicted. Lets you
   see whether the model is over- or under-confident in each range.
2. **Value finder** — given my prediction and a user-entered bookmaker
   odds, compute Kelly stake + expected value. Read-only suggestion,
   no betting integration.
3. **Per-league accuracy time series** — accuracy by season, so you can
   see whether the model degrades after roster changes (e.g.,
   Bayern-dominated Bundesliga vs current mess).
4. **Honest defaults** — the model's accuracy floor on second-tier
   leagues (43-50%) is real. I'll mark them in the UI rather than
   pretend uniform quality.

The other model is going harder on infrastructure (coverage reports,
doctor command, schema self-checks, API-Football wiring). Both
directions are valid; the user wanted to see which produces a more
useful product.
