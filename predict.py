from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import typer
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import get_settings
from data.coverage import build_coverage_report
from data.database import DEFAULT_DB_PATH, init_database
from data.doctor import build_doctor_report
from models.backtest import BacktestConfig, backtest_dixon_coles
from models.diagnostics import build_diagnostics
from models.roi_simulator import ROIConfig, simulate_roi
from scrape.odds_backfill import fetch_odds_frame
from models.dixon_coles import DixonColesConfig, DixonColesModel
from models.elo import latest_elos
from models.ensemble import EnsembleConfig
from models.ensemble import fit as ensemble_fit
from models.ensemble import predict_match as ensemble_predict
from models.in_play import InPlayConfig, predict_in_play
from models.fixture_history import append_snapshot, attach_deltas
from models.replay import rank_surprises, replay_match
from models.totals import derive_totals
from scrape.upcoming_fixtures import LEAGUE_KEY_TO_TSDB_ID, fetch_upcoming_multi
from models.team_strengths import compare_two as compare_team_strengths
from models.team_strengths import extract_strengths
from models.implied_probs import implied_probabilities
from models.penaltyblog_models import (
    MODEL_FACTORIES as PB_MODEL_FACTORIES,
    fit_and_predict as pb_fit_and_predict,
)
from scrape import api_football
from scrape.registry import LeagueRegistry
from scrape.update import DEFAULT_CACHE_DIR, IncrementalUpdater, run_daily_update


SETTINGS = get_settings()
PROJECT_ROOT = SETTINGS.project_root
DEFAULT_DB_PATH = SETTINGS.db_path
DEFAULT_CACHE_DIR = SETTINGS.cache_dir
EXPORT_DIR = SETTINGS.export_dir
STATIC_DIR = SETTINGS.static_dir
WORLDCUP_CACHE_PATH = SETTINGS.worldcup_cache_path


class PredictionRequest(BaseModel):
    home_team: str = Field(..., examples=["Arsenal"])
    away_team: str = Field(..., examples=["Liverpool"])
    league: str | None = Field(None, examples=["英超"])
    home_elo: float | None = None
    away_elo: float | None = None
    neutral_site: bool | None = None
    stage: str | None = Field(None, examples=["group"])
    knockout: bool = False
    max_goals: int = Field(8, ge=4, le=15)
    as_of: date | None = None
    home_advantage: float = 0.22
    xg_blend_weight: float = Field(0.35, ge=0.0, le=1.0)
    # Model picker — defaults to our home-grown DC + Elo blend. The penaltyblog
    # variants are battle-tested implementations from a published Python
    # package; on EPL their Arsenal vs Chelsea prediction is much closer to
    # market closing odds than ours, suggesting our Elo correction is too
    # aggressive. Try ``bivariate_poisson`` for the best non-Elo baseline.
    model: str = Field(
        "dixon_coles_elo",
        description="One of: dixon_coles_elo (default) | dixon_coles | bivariate_poisson | poisson",
    )


class UpdateRequest(BaseModel):
    leagues: list[str] | None = None
    years_back: int = Field(5, ge=1, le=10)
    include_ratings: bool = True
    include_api_football: bool = False
    include_players: bool = False
    include_fbref_xg: bool = False


class InPlayRequest(BaseModel):
    home_team: str = Field(..., examples=["Arsenal"])
    away_team: str = Field(..., examples=["Chelsea"])
    league: str | None = Field(None, examples=["英超"])
    current_home: int = Field(..., ge=0, le=15)
    current_away: int = Field(..., ge=0, le=15)
    minute_elapsed: int = Field(..., ge=0, le=120)
    neutral_site: bool | None = None
    model: str = Field(
        "dixon_coles_elo",
        description="Goal model used for the pre-match xG. Same options as /predict.",
    )
    chasing_multiplier: float = Field(
        1.15, ge=0.5, le=2.0,
        description="Multiplier on remaining xG for the trailing team (default 1.15).",
    )
    leading_multiplier: float = Field(
        0.92, ge=0.5, le=2.0,
        description="Multiplier on remaining xG for the leading team (default 0.92).",
    )


class ROIRequest(BaseModel):
    league: str | None = Field(None, examples=["英超"])
    min_train_matches: int = Field(200, ge=50)
    refit_every: int = Field(25, ge=1)
    min_edge: float = Field(0.05, ge=0.0, le=0.5,
        description="Minimum (model_prob − implied_prob) to bet. 0.05 = 5pp.")
    min_ev: float = Field(0.05, ge=0.0, le=1.0,
        description="Minimum expected value per unit stake.")
    kelly_multiplier: float = Field(0.5, ge=0.0, le=1.0,
        description="Fraction of full Kelly to stake. 0.5 = half-Kelly (recommended).")
    max_kelly_fraction: float = Field(0.05, ge=0.005, le=0.5,
        description="Cap on bankroll fraction per bet (safety).")
    starting_bankroll: float = Field(100.0, ge=1.0)
    include_bets: bool = Field(False, description="Include the per-bet array in the response.")
    model: str = Field(
        "dixon_coles_elo",
        description="Goal model. dixon_coles_elo (default), or penaltyblog: dixon_coles, bivariate_poisson, poisson.",
    )
    implied_method: str = Field(
        "shin",
        description="Implied-probability extraction. shin (default) | multiplicative | power | additive.",
    )


class ManualResultRequest(BaseModel):
    """User-entered result for a fixture none of our sources covered (or got wrong)."""
    model_config = {"populate_by_name": True}
    league: str = Field(..., examples=["k_league_2", "英超"])
    # Field is named ``match_date`` internally to avoid shadowing the imported
    # ``date`` type (Pydantic v2 evaluates annotations even with
    # ``from __future__ import annotations``). The JSON wire format still
    # accepts ``"date"`` via the alias.
    match_date: date = Field(..., alias="date", examples=["2026-05-17"])
    home_team: str = Field(..., min_length=1)
    away_team: str = Field(..., min_length=1)
    home_goals: int = Field(..., ge=0, le=30)
    away_goals: int = Field(..., ge=0, le=30)
    season: str | None = None
    neutral_site: bool | None = None
    stage: str | None = None


class BacktestRequest(BaseModel):
    league: str | None = Field(None, examples=["英超"])
    source: str | None = None
    since: date | None = None
    until: date | None = None
    min_train_matches: int = Field(80, ge=20)
    max_goals: int = Field(8, ge=4, le=15)
    refit_every: int = Field(1, ge=1)
    xg_blend_weight: float = Field(0.35, ge=0.0, le=1.0)
    # Strength of the Elo prior correction. 0 = pure team-attack/defense DC.
    # Default 0.10 matches DixonColesConfig. The /diagnostics/ablation endpoint
    # sets this to 0 to measure how much Elo contributes to model quality.
    elo_weight: float = Field(0.10, ge=0.0, le=1.0)
    summary_only: bool = Field(
        False,
        description="If true, drop the per-match predictions array from the response.",
    )


class AppState:
    scheduler: BackgroundScheduler | None = None


state = AppState()
cli = typer.Typer(help="football-predictor Web service and CLI.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_database(DEFAULT_DB_PATH)
    if _scheduler_enabled():
        state.scheduler = _start_scheduler()

    # Pre-warm the /upcoming cache in a background thread so the tab loads
    # instantly the first time. Skips if a fresh cache file already exists.
    import threading
    def _prewarm() -> None:
        for days in (3, 7, 14, 30):
            try:
                if _read_upcoming_cache(days) is not None:
                    continue
                payload = _compute_upcoming_payload(
                    league_keys=sorted(LEAGUE_KEY_TO_TSDB_ID.keys()),
                    days_ahead=days,
                    include_predictions=True,
                )
                _write_upcoming_cache(days, payload)
                print(f"[upcoming] pre-warmed cache for days_ahead={days}: {payload['fixture_count']} fixtures")
            except Exception as exc:  # noqa: BLE001 — best-effort
                print(f"[upcoming] pre-warm days_ahead={days} failed: {exc}")
    threading.Thread(target=_prewarm, daemon=True, name="upcoming-prewarm").start()

    try:
        yield
    finally:
        if state.scheduler:
            state.scheduler.shutdown(wait=False)


app = FastAPI(
    title="football-predictor",
    version="0.1.0",
    description="Dixon-Coles + Elo football match prediction service.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "database": str(DEFAULT_DB_PATH),
        "scheduler": bool(state.scheduler and state.scheduler.running),
    }


@app.get("/leagues")
def leagues() -> dict[str, Any]:
    registry = LeagueRegistry()
    return {
        "leagues": [
            {
                "key": league.key,
                "name": league.name,
                "country": league.country,
                "tier": league.tier,
                "football_data_code": league.football_data_code,
                "api_football_id": league.api_football_id,
                "note": league.note,
            }
            for league in registry.all()
        ],
        "worldcup_2026": registry.worldcup,
    }


@app.post("/predict")
def predict_endpoint(payload: PredictionRequest) -> dict[str, Any]:
    try:
        return predict_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/update")
def update_endpoint(payload: UpdateRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    background_tasks.add_task(
        _run_update_job,
        payload.leagues,
        payload.years_back,
        payload.include_ratings,
        payload.include_api_football,
        payload.include_players,
        payload.include_fbref_xg,
    )
    return {"status": "queued", "message": "Incremental update started in background."}


@app.post("/backtest")
def backtest_endpoint(payload: BacktestRequest) -> dict[str, Any]:
    try:
        return backtest_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/roi-simulation")
def roi_simulation_endpoint(payload: ROIRequest) -> dict[str, Any]:
    """Walk-forward value-betting against Bet365 closing odds.

    Trains Dixon-Coles on prior matches, places half-Kelly stakes when
    edge + EV exceed the configured thresholds, and tracks bankroll. Tells
    the truth about whether the model's "value finds" actually make money.
    """
    db = init_database(DEFAULT_DB_PATH)
    registry = LeagueRegistry()
    league_key = _normalize_optional_league(payload.league, registry)
    if league_key is None:
        raise HTTPException(status_code=400, detail="Pick a specific league for ROI simulation.")

    matches = db.fetch_matches(league_key=league_key)
    odds = fetch_odds_frame(db, league_key=league_key)
    if matches.empty:
        raise HTTPException(status_code=400, detail=f"No matches for league '{league_key}'.")
    if odds.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No bookmaker odds for '{league_key}'. Run odds_backfill.backfill_all() first.",
        )

    config = ROIConfig(
        min_train_matches=payload.min_train_matches,
        refit_every=payload.refit_every,
        min_edge=payload.min_edge,
        min_ev=payload.min_ev,
        kelly_multiplier=payload.kelly_multiplier,
        max_kelly_fraction=payload.max_kelly_fraction,
        starting_bankroll=payload.starting_bankroll,
        implied_method=payload.implied_method,
        model=payload.model,
    )
    try:
        result = simulate_roi(matches, odds, config=config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        **result.to_dict(include_bets=payload.include_bets),
        "request": {
            "league": payload.league,
            "min_edge": payload.min_edge,
            "min_ev": payload.min_ev,
            "kelly_multiplier": payload.kelly_multiplier,
        },
    }


@app.post("/predict/in-play")
def predict_in_play_endpoint(payload: InPlayRequest) -> dict[str, Any]:
    """Recompute full-time probabilities given the current score and minute.

    Internally: runs a pre-match prediction to get xG, then forwards to the
    in-play module which scales remaining xG by (time remaining / 90) and
    applies a chasing/leading multiplier on the trailing/leading side.
    """
    # 1. Reuse the pre-match pipeline to get xG. We force max_goals so the
    # downstream Poisson grid stays small.
    base_request = PredictionRequest(
        home_team=payload.home_team,
        away_team=payload.away_team,
        league=payload.league,
        neutral_site=payload.neutral_site,
        model=payload.model,
        max_goals=8,
    )
    try:
        pre = predict_payload(base_request)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    xg = pre.get("expected_goals", {})
    pre_xh = xg.get("home")
    pre_xa = xg.get("away")
    if pre_xh is None or pre_xa is None:
        # National-team Elo path doesn't expose xG. Derive crude xG from the
        # match probabilities: avg 2.6 total goals split by win probability.
        probs = pre["probabilities"]
        skew = probs["home_win"] - probs["away_win"]
        pre_xh = max(0.3, 1.3 * (1 + 0.5 * skew))
        pre_xa = max(0.3, 1.3 * (1 - 0.5 * skew))

    cfg = InPlayConfig(
        chasing_multiplier=payload.chasing_multiplier,
        leading_multiplier=payload.leading_multiplier,
    )
    try:
        result = predict_in_play(
            home_team=payload.home_team,
            away_team=payload.away_team,
            pre_match_xg_home=float(pre_xh),
            pre_match_xg_away=float(pre_xa),
            current_home=payload.current_home,
            current_away=payload.current_away,
            minute_elapsed=payload.minute_elapsed,
            config=cfg,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response = result.to_dict()
    response["pre_match"] = {
        "expected_goals": {"home": float(pre_xh), "away": float(pre_xa)},
        "probabilities": pre["probabilities"],
        "model": pre["model"]["type"],
    }
    return response


@app.post("/diagnostics/ablation")
def diagnostics_ablation_endpoint(payload: BacktestRequest) -> dict[str, Any]:
    """Feature-ablation diagnostics: re-run the backtest with one prior turned
    off at a time, return per-config metrics + delta vs full.

    This is the data-driven version of "is Elo actually helping?". For each
    of the four configurations (full / no Elo / no xG / neither), we run the
    same walk-forward backtest and report accuracy, Brier, log-loss, and ECE.
    The deltas vs the full configuration show, in plain numbers, how much
    each prior contributes to model quality.

    Cost: 4× backtest runtime, ~2-3 minutes for a typical Tier-1 league.
    Heavier than the regular /diagnostics — call it deliberately, not on
    every UI render.
    """
    from models.diagnostics import expected_calibration_error  # noqa: PLC0415

    league_norm = _normalize_optional_league(payload.league, LeagueRegistry())
    if not league_norm:
        raise HTTPException(status_code=400, detail="--league is required for ablation; "
            "running across all leagues would take 30+ minutes.")

    # Four configurations to compare. We always include the full one first
    # so the UI can compute deltas client-side if it wants.
    configs = [
        ("full",         {"elo_weight": 0.10, "xg_blend_weight": payload.xg_blend_weight}),
        ("no_elo",       {"elo_weight": 0.00, "xg_blend_weight": payload.xg_blend_weight}),
        ("no_xg",        {"elo_weight": 0.10, "xg_blend_weight": 0.00}),
        ("no_elo_no_xg", {"elo_weight": 0.00, "xg_blend_weight": 0.00}),
    ]

    results = []
    for name, overrides in configs:
        req = payload.model_copy(update={**overrides, "summary_only": False})
        try:
            run = backtest_payload(req)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        summary = run["summary"]
        predictions = pd.DataFrame(run["predictions"])
        ece = expected_calibration_error(predictions)
        ece_values = [v for v in ece.values() if v is not None] if ece else []
        ece_mean = sum(ece_values) / len(ece_values) if ece_values else None
        results.append({
            "config": name,
            "metrics": {
                "accuracy": summary.get("accuracy"),
                "brier_score": summary.get("brier_score"),
                "log_loss": summary.get("log_loss"),
                "ece_mean": ece_mean,
                "sample_size": summary.get("sample_size"),
            },
            "config_params": overrides,
        })

    # Add delta-vs-full to each non-full row so the UI doesn't need math.
    full = results[0]["metrics"]
    for r in results[1:]:
        m = r["metrics"]
        r["delta_vs_full"] = {
            "accuracy_pp": (m["accuracy"] - full["accuracy"]) * 100
                            if m["accuracy"] is not None and full["accuracy"] is not None else None,
            "brier": (m["brier_score"] - full["brier_score"])
                            if m["brier_score"] is not None and full["brier_score"] is not None else None,
            "log_loss": (m["log_loss"] - full["log_loss"])
                            if m["log_loss"] is not None and full["log_loss"] is not None else None,
            "ece_mean": (m["ece_mean"] - full["ece_mean"])
                            if m["ece_mean"] is not None and full["ece_mean"] is not None else None,
        }

    # Detect the silent-feature trap: if no_xg matches full exactly, it means
    # there's no xG data in the matches table for this league (so flipping
    # xg_blend_weight off changed nothing). Surface this — the alternative is
    # the user staring at a misleading "+0.000 brier delta" thinking xG is
    # neutral, when really it's never been pulled.
    silent_features = []
    full_m = results[0]["metrics"]
    for r in results[1:]:
        m = r["metrics"]
        is_identical = (
            m["accuracy"] == full_m["accuracy"]
            and m["brier_score"] == full_m["brier_score"]
            and m["log_loss"] == full_m["log_loss"]
        )
        if is_identical:
            # The ablation knob did nothing. Either the data isn't there or
            # the feature isn't actually plumbed through.
            if r["config"] == "no_xg":
                silent_features.append({
                    "feature": "xg",
                    "explanation": (
                        f"no_xg run produced identical metrics to full, meaning the "
                        f"matches table has no populated home_xg/away_xg for {league_norm}. "
                        "Run `python predict.py update --league " + (payload.league or league_norm) +
                        " --include-xg` to backfill FBref xG, then re-run this ablation."
                    ),
                })
            elif r["config"] == "no_elo":
                silent_features.append({
                    "feature": "elo",
                    "explanation": (
                        "no_elo run produced identical metrics to full. This is unusual — "
                        "Elo should always contribute SOMETHING. Check that pre_match_elos "
                        "were attached (look at the predictions response for "
                        "home_elo/away_elo fields)."
                    ),
                })

    return {
        "league": league_norm,
        "configurations": results,
        "silent_features": silent_features,
        "request": {
            "league": payload.league,
            "min_train_matches": payload.min_train_matches,
            "refit_every": payload.refit_every,
        },
    }


@app.post("/diagnostics")
def diagnostics_endpoint(payload: BacktestRequest) -> dict[str, Any]:
    """Run a backtest and return calibration + confidence-ladder diagnostics.

    Reuses the BacktestRequest schema. Returns the summary block from the
    backtest plus the diagnostics bundle (calibration_curve per outcome,
    confidence_ladder, expected_calibration_error). Predictions are dropped
    so the response stays small.
    """
    try:
        # Force summary_only off internally so we can compute diagnostics from
        # predictions; we strip predictions from the response below.
        request = payload.model_copy(update={"summary_only": False})
        result = backtest_payload(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    predictions = pd.DataFrame(result["predictions"])
    diagnostics = build_diagnostics(predictions)
    return {
        "summary": result["summary"],
        "diagnostics": diagnostics,
        "request": {
            "league": payload.league,
            "min_train_matches": payload.min_train_matches,
            "refit_every": payload.refit_every,
        },
    }


@app.get("/export/{table}")
def export_endpoint(
    table: str,
    filename: str | None = Query(None, description="Optional CSV filename."),
) -> FileResponse:
    db = init_database(DEFAULT_DB_PATH)
    filename = filename or f"{table}.csv"
    output = EXPORT_DIR / filename
    try:
        path = db.export_csv(table, output)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.get("/teams")
def teams_endpoint(league: str | None = Query(None, description="League key or alias.")) -> dict[str, Any]:
    """Distinct team names — used by the web UI's autocomplete."""
    db = init_database(DEFAULT_DB_PATH)
    registry = LeagueRegistry()
    league_key: str | None = None
    if league:
        try:
            league_key = registry.normalize(league)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    matches = db.fetch_matches(league_key=league_key)
    teams: set[str] = set()
    if not matches.empty:
        teams.update(matches["home_team"].dropna().astype(str).tolist())
        teams.update(matches["away_team"].dropna().astype(str).tolist())

    # National teams come from the ratings table.
    if league_key in (None, "world_cup"):
        import pandas as pd
        ratings = pd.read_sql_table("ratings", db.engine)
        if not ratings.empty:
            ratings = ratings[ratings["scope"] == "national"]
            teams.update(ratings["team"].dropna().astype(str).tolist())

    return {
        "league": league_key,
        "count": len(teams),
        "teams": sorted(teams),
    }


@app.get("/teams/strengths")
def team_strengths_endpoint(
    league: str = Query(..., description="League key or alias (required)."),
    lookback_days: int = Query(730, ge=180, le=2000),
) -> dict[str, Any]:
    """Fit Dixon-Coles on a league and return per-team attack/defense ratings.

    Drives the 球队强度 tab. Cached by SQLite query + DC fit, ~1-3s per league.
    """
    db = init_database(DEFAULT_DB_PATH)
    registry = LeagueRegistry()
    try:
        league_key = registry.normalize(league)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    league_obj = registry.leagues.get(league_key)

    matches = db.fetch_matches(league_key=league_key)
    if matches.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No matches for league '{league_key}'. Run `python predict.py update` first.",
        )

    # Pull latest club Elo for each team (best-effort, may be empty).
    import pandas as pd
    ratings = pd.read_sql_table("ratings", db.engine)
    elo_lookup: dict[str, float] = {}
    if not ratings.empty:
        club_ratings = ratings[ratings["scope"] == "club"]
        if not club_ratings.empty:
            club_ratings = club_ratings.sort_values("rating_date")
            elo_lookup = club_ratings.groupby("team").tail(1).set_index("team")["elo"].to_dict()

    out = extract_strengths(
        matches,
        lookback_days=lookback_days,
        club_elo_lookup=elo_lookup,
    )
    return {
        "league": league_key,
        "league_name": league_obj.name if league_obj else league_key,
        **out,
    }


@app.get("/teams/compare")
def team_compare_endpoint(
    league: str = Query(..., description="League key or alias."),
    home: str = Query(..., description="Home team name."),
    away: str = Query(..., description="Away team name."),
) -> dict[str, Any]:
    """Side-by-side strength comparison for two teams in a league."""
    db = init_database(DEFAULT_DB_PATH)
    registry = LeagueRegistry()
    try:
        league_key = registry.normalize(league)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    matches = db.fetch_matches(league_key=league_key)
    if matches.empty:
        raise HTTPException(status_code=400, detail=f"No matches for league '{league_key}'.")

    try:
        return compare_team_strengths(matches, home, away)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/match-history")
def match_history_endpoint(
    league: str = Query(..., description="League key or alias."),
    limit: int = Query(40, ge=5, le=200, description="Most recent N matches."),
) -> dict[str, Any]:
    """Return the most recent completed matches in a league for the replay UI."""
    db = init_database(DEFAULT_DB_PATH)
    registry = LeagueRegistry()
    try:
        league_key = registry.normalize(league)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    matches = db.fetch_matches(league_key=league_key)
    if matches.empty:
        return {"league": league_key, "matches": []}
    recent = matches.sort_values("date", ascending=False).head(limit).copy()
    recent["date"] = recent["date"].astype(str)
    return {
        "league": league_key,
        "matches": recent[
            ["date", "home_team", "away_team", "home_goals", "away_goals"]
        ].to_dict(orient="records"),
    }


@app.post("/match-replay")
def match_replay_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Refit Dixon-Coles up to (not including) a target match date, then predict.

    Body: {"league": "英超", "match_date": "2026-05-13",
           "home_team": "Man City", "away_team": "Crystal Palace"}
    """
    db = init_database(DEFAULT_DB_PATH)
    registry = LeagueRegistry()
    try:
        league_key = registry.normalize(payload["league"])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    matches = db.fetch_matches(league_key=league_key)
    if matches.empty:
        raise HTTPException(status_code=400, detail=f"No matches for league '{league_key}'.")
    try:
        match_date = pd.Timestamp(payload["match_date"]).date()
        result = replay_match(
            target_date=match_date,
            home_team=payload["home_team"],
            away_team=payload["away_team"],
            league_matches=matches,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload_out = result.to_dict()
    payload_out["league_key"] = league_key
    return payload_out


@app.get("/match-replay/surprises")
def match_surprises_endpoint(
    league: str = Query(..., description="League key or alias."),
    refit_every: int = Query(25, ge=5, le=200),
) -> dict[str, Any]:
    """Rank a league's matches by surprise: biggest upsets + best calls.

    Walk-forward Dixon-Coles, refitting every ``refit_every`` matches.
    Returns the model's lowest-probability outcomes that actually happened
    (upsets) and its highest-confidence correct predictions (best calls).
    """
    db = init_database(DEFAULT_DB_PATH)
    registry = LeagueRegistry()
    try:
        league_key = registry.normalize(league)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    matches = db.fetch_matches(league_key=league_key)
    if matches.empty:
        raise HTTPException(status_code=400, detail=f"No matches for league '{league_key}'.")
    result = rank_surprises(
        league_matches=matches,
        league_key=league_key,
        refit_every=refit_every,
    )
    return {"league": league_key, **result}


@app.get("/worldcup/forecast")
def worldcup_forecast(
    n_sims: int = Query(5000, ge=200, le=50000),
    top: int = Query(20, ge=8, le=48),
    seed: int = Query(42),
    fresh: bool = Query(False, description="Skip cache and recompute."),
) -> dict[str, Any]:
    """Run (or return cached) WC 2026 Monte Carlo.

    Cached payloads are keyed by n_sims+seed and refreshed when ``fresh=true``.
    First call with default params takes ~5 seconds; subsequent calls return
    instantly from the JSON cache on disk.
    """
    from worldcup import load_groups, simulate as wc_simulate

    cache_key = f"{n_sims}_{seed}"
    if not fresh and WORLDCUP_CACHE_PATH.exists():
        cached = json.loads(WORLDCUP_CACHE_PATH.read_text())
        if cached.get("cache_key") == cache_key:
            return {**cached, "from_cache": True, "top": top, "table": cached["table"][:top]}

    db = init_database(DEFAULT_DB_PATH)
    try:
        groups = load_groups(None, db)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    result = wc_simulate(groups, n_sims=n_sims, seed=seed)
    table_records = result["table"].to_dict(orient="records")
    payload = {
        "cache_key": cache_key,
        "n_sims": result["n_sims"],
        "seed": seed,
        "computed_at": datetime.now(UTC).isoformat(),
        "table": table_records,
    }
    WORLDCUP_CACHE_PATH.write_text(json.dumps(payload, default=str))
    return {**payload, "from_cache": False, "top": top, "table": table_records[:top]}


UPCOMING_CACHE_TTL_S = 3600  # 1 hour


def _compute_upcoming_payload(
    *,
    league_keys: list[str],
    days_ahead: int,
    include_predictions: bool,
) -> dict[str, Any]:
    """The slow path: scrape TheSportsDB + run model for each fixture. Cached by callers."""
    cache_dir = Path(__file__).resolve().parent / "data" / "cache" / "tsdb"
    fixtures_df = fetch_upcoming_multi(
        league_keys, cache_dir=cache_dir, days_ahead=days_ahead,
    )

    if fixtures_df.empty:
        return {
            "queried_leagues": league_keys,
            "days_ahead": days_ahead,
            "fixture_count": 0,
            "fixtures": [],
            "note": "No upcoming fixtures returned from TheSportsDB. Free tier is tight (often 1 event/league).",
            "computed_at": datetime.utcnow().isoformat(),
        }

    # Resolve team-crest URLs. The /upcoming events from TSDB usually carry
    # a working per-team badge URL — we (a) try them first, and (b) harvest
    # them into the per-team cache so the /predict hero card (which has no
    # event payload, just team names) can hit a warm cache and render the
    # correct crest. Without (b), /predict would fall through to TSDB's
    # ``searchteams.php`` which (free-tier bug) returns Arsenal for every
    # query, painting every team's hero with the wrong crest.
    try:
        from scrape.team_badges import attach_badges, populate_cache_from_events
        badge_cache_dir = PROJECT_ROOT / "data" / "cache" / "tsdb-teams"
        attach_badges(fixtures_df, cache_dir=badge_cache_dir)
        populate_cache_from_events(fixtures_df, cache_dir=badge_cache_dir)
    except Exception as exc:  # noqa: BLE001 — non-fatal
        print(f"[upcoming] badge attach failed: {exc}")

    # Cross-check dates against fd.org's SCHEDULED view. Catches TheSportsDB
    # stale-date errors at /upcoming time, BEFORE we make predictions for the
    # wrong day. Best-effort: if fd.org is down or unconfigured, warnings just
    # don't appear and /upcoming still works.
    date_warnings: dict[tuple[str, str, str], dict[str, Any]] = {}
    try:
        from data.fixture_date_check import cross_check_dates
        date_warnings = cross_check_dates(
            fixtures_df, cache_dir=PROJECT_ROOT / "data" / "cache" / "football-data-org",
        )
    except Exception as exc:  # noqa: BLE001 — non-fatal
        print(f"[upcoming] date cross-check failed: {exc}")

    fixtures = []
    for row in fixtures_df.to_dict(orient="records"):
        item: dict[str, Any] = {
            "date": str(row["date"]),
            "league_key": row["league_key"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "venue": row.get("venue"),
            "time_utc": row.get("time_utc"),
            "status": row.get("status"),
            # Crests from TheSportsDB. The UI uses these when present and
            # falls back to a colored monogram circle when missing/broken.
            "home_badge_url": row.get("home_badge_url"),
            "away_badge_url": row.get("away_badge_url"),
        }
        if include_predictions:
            try:
                pred_req = PredictionRequest(
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    league=row["league_key"],
                )
                pred = predict_payload(pred_req)
                item["prediction"] = {
                    "probabilities": pred.get("probabilities"),
                    "expected_goals": pred.get("expected_goals"),
                    "most_likely_scores": (pred.get("most_likely_scores") or [])[:3],
                    "totals": derive_totals(pred.get("score_matrix")),
                    "model_type": pred.get("model", {}).get("type"),
                    "training_rows": pred.get("model", {}).get("training_rows"),
                }
            except (ValueError, KeyError) as exc:
                item["prediction"] = {"error": str(exc)[:120]}
        fixtures.append(item)

    # Attach per-fixture date-check chips (fd.org confirmation / warning)
    try:
        from data.fixture_date_check import attach_date_warnings
        attach_date_warnings(fixtures, date_warnings)
    except Exception as exc:  # noqa: BLE001 — non-fatal
        print(f"[upcoming] date warning attach failed: {exc}")

    # Append a snapshot to the rolling history log so we can compute "vs 6h ago"
    # deltas on later reads. Cheap (one JSONL line per fixture).
    try:
        append_snapshot(fixtures)
    except Exception as exc:  # noqa: BLE001 — non-fatal
        print(f"[upcoming] history append failed: {exc}")

    return {
        "queried_leagues": league_keys,
        "days_ahead": days_ahead,
        "fixture_count": len(fixtures),
        "fixtures": fixtures,
        "computed_at": datetime.utcnow().isoformat(),
    }


def _upcoming_cache_path(days_ahead: int) -> Path:
    """Disk cache key: just days_ahead (we always query all known leagues)."""
    return PROJECT_ROOT / "data" / "cache" / "upcoming" / f"upcoming_{days_ahead}d.json"


def _read_upcoming_cache(days_ahead: int) -> dict[str, Any] | None:
    path = _upcoming_cache_path(days_ahead)
    if not path.exists():
        return None
    age_s = time.time() - path.stat().st_mtime
    if age_s > UPCOMING_CACHE_TTL_S:
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_upcoming_cache(days_ahead: int, payload: dict[str, Any]) -> None:
    path = _upcoming_cache_path(days_ahead)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@app.get("/upcoming")
def upcoming_endpoint(
    leagues: str | None = Query(
        None,
        description="Comma-separated league keys. Default: all leagues we have TSDB IDs for.",
    ),
    days_ahead: int = Query(7, ge=1, le=30),
    include_predictions: bool = Query(True),
    fresh: bool = Query(False, description="Skip cache; recompute."),
) -> dict[str, Any]:
    """Upcoming fixtures + per-match predictions.

    Disk-cached for 1 hour (or pre-warmed by the server's lifespan task), so
    repeat hits are <10ms. Pass ``fresh=true`` to force recomputation.
    """
    registry = LeagueRegistry()
    if leagues:
        league_keys = []
        for raw in leagues.split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                league_keys.append(registry.normalize(raw))
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
    else:
        league_keys = sorted(LEAGUE_KEY_TO_TSDB_ID.keys())

    # Cached path only when using the default league set (the common case).
    use_cache = (not fresh and not leagues and include_predictions)
    if use_cache:
        cached = _read_upcoming_cache(days_ahead)
        if cached is not None:
            cached["from_cache"] = True
            # Recompute deltas on every read — they depend on "now", not on
            # when the cache was written.
            attach_deltas(cached.get("fixtures", []))
            return cached

    payload = _compute_upcoming_payload(
        league_keys=league_keys,
        days_ahead=days_ahead,
        include_predictions=include_predictions,
    )
    payload["from_cache"] = False
    if use_cache:
        _write_upcoming_cache(days_ahead, payload)
    attach_deltas(payload.get("fixtures", []))
    return payload


@app.get("/recent")
def recent_matches(
    league: str | None = Query(None, description="League key or alias."),
    limit: int = Query(20, ge=1, le=200),
) -> dict[str, Any]:
    """Most recent matches — for the Leagues page."""
    db = init_database(DEFAULT_DB_PATH)
    registry = LeagueRegistry()
    league_key: str | None = None
    if league:
        try:
            league_key = registry.normalize(league)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    matches = db.fetch_matches(league_key=league_key)
    if matches.empty:
        return {"league": league_key, "count": 0, "matches": []}
    matches = matches.sort_values("date", ascending=False).head(limit)
    matches = matches.assign(date=matches["date"].astype(str))
    return {
        "league": league_key,
        "count": len(matches),
        "matches": matches[
            ["date", "league_key", "league_name", "home_team", "away_team", "home_goals", "away_goals"]
        ].to_dict(orient="records"),
    }


@app.get("/data-health")
def data_health_endpoint() -> dict[str, Any]:
    """One-stop snapshot of all data sources, caches, and API key status.

    Used by the 数据健康 (Data Health) tab. Fast — pure DB+filesystem reads, no
    external calls.
    """
    from data.data_health import build_health_report

    db = init_database(DEFAULT_DB_PATH)
    return build_health_report(db)


@app.get("/audit/vs-backtest")
def audit_vs_backtest_endpoint(
    fresh: bool = Query(False, description="Skip the 24h disk cache."),
    max_leagues: int = Query(12, ge=1, le=30, description="Cap how many leagues to backtest."),
) -> dict[str, Any]:
    """For each league in audit's by_league, run a quick backtest and align
    the headline metrics. Result is cached on disk for 24h since the matches
    table only changes once per day from the backfill cron.

    The point: ``Δ = backtest - audit`` per metric. Large positive Δaccuracy
    means the model fit the historical distribution but doesn't generalize
    to live fixtures (= data drift). Δ ≈ 0 means backtest is a useful proxy
    for real-world performance.
    """
    cache_path = PROJECT_ROOT / "data" / "cache" / "audit-vs-backtest.json"
    if not fresh and cache_path.exists():
        age_s = time.time() - cache_path.stat().st_mtime
        if age_s < 86_400:  # 24h
            try:
                return {**json.loads(cache_path.read_text()), "from_cache": True}
            except (OSError, json.JSONDecodeError):
                pass  # corrupted cache → fall through to fresh compute

    import pandas as pd  # noqa: PLC0415 — already in path
    from data.audit_vs_backtest import compare_audit_to_backtest
    from data.database import Database
    from models.prediction_audit import audit_summary

    db = Database(DEFAULT_DB_PATH)
    matches = pd.read_sql_table(
        "matches", db.engine,
        columns=["date", "home_team", "away_team", "home_goals", "away_goals"],
    )
    audit = audit_summary(matches)
    payload = compare_audit_to_backtest(
        db, audit.get("by_league", []), max_leagues=max_leagues,
    )
    payload["generated_at"] = datetime.utcnow().isoformat()
    payload["from_cache"] = False
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, default=str))
    except OSError:
        pass
    return payload


@app.post("/manual-result")
def manual_result_endpoint(payload: ManualResultRequest) -> dict[str, Any]:
    """Patch one fixture into the matches table with ``source="manual"``.

    For results that none of our automated sources cover, or that a source
    got wrong. ``source="manual"`` sits at the top of the field-priority
    table in ``data/source_resolver.py`` — so a manual entry will override
    any conflicting automated score for the same canonical fixture.
    """
    from data.manual_results import submit_manual_result

    db = init_database(DEFAULT_DB_PATH)
    try:
        return submit_manual_result(
            db,
            league=payload.league,
            date=payload.match_date,
            home_team=payload.home_team,
            away_team=payload.away_team,
            home_goals=payload.home_goals,
            away_goals=payload.away_goals,
            season=payload.season,
            neutral_site=payload.neutral_site,
            stage=payload.stage,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/manual-results")
def manual_results_list_endpoint(limit: int = 50) -> dict[str, Any]:
    """Recent manually-entered results — for the UI table."""
    from data.manual_results import list_recent_manual_results

    db = init_database(DEFAULT_DB_PATH)
    return {"results": list_recent_manual_results(db, limit=max(1, min(limit, 500)))}


@app.get("/stats")
def stats_endpoint() -> dict[str, Any]:
    """Headline counts for the landing page."""
    import pandas as pd

    db = init_database(DEFAULT_DB_PATH)
    matches = db.fetch_matches()
    ratings = pd.read_sql_table("ratings", db.engine) if matches is not None else pd.DataFrame()
    leagues_n = int(matches["league_key"].nunique()) if not matches.empty else 0
    teams_set: set[str] = set()
    if not matches.empty:
        teams_set.update(matches["home_team"].astype(str))
        teams_set.update(matches["away_team"].astype(str))

    earliest = latest = None
    if not matches.empty:
        earliest = str(matches["date"].min())
        latest = str(matches["date"].max())

    club_elo_n = national_elo_n = 0
    if not ratings.empty:
        club_elo_n = int((ratings["scope"] == "club").sum())
        national_elo_n = int((ratings["scope"] == "national").sum())

    return {
        "matches": int(len(matches)) if matches is not None else 0,
        "teams": len(teams_set),
        "leagues_with_data": leagues_n,
        "earliest_match": earliest,
        "latest_match": latest,
        "club_elo_rows": club_elo_n,
        "national_elo_rows": national_elo_n,
    }


@app.get("/coverage")
def coverage_endpoint() -> dict[str, Any]:
    return build_coverage_report(db_path=DEFAULT_DB_PATH)


@app.get("/doctor")
def doctor_endpoint(
    live: bool = Query(False, description="If true, probe API-Football with the configured key."),
) -> dict[str, Any]:
    return build_doctor_report(db_path=DEFAULT_DB_PATH, live_api=live)


@app.get("/api-football/leagues")
def api_football_leagues_endpoint(
    country: str | None = Query(None, description="Country name, e.g. China."),
    search: str | None = Query(None, description="Search term, e.g. Super League."),
    league_id: int | None = Query(None, description="Known API-Football league id."),
    season: int | None = Query(None, description="Optional season year."),
) -> dict[str, Any]:
    try:
        frame = api_football.discover_leagues(
            country=country,
            search=search,
            league_id=league_id,
            season=season,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "count": len(frame),
        "leagues": frame.drop(columns=["raw"], errors="ignore").to_dict(orient="records"),
    }


# Mount the web UI last so /api routes resolve first.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def predict_payload(payload: PredictionRequest) -> dict[str, Any]:
    # Canonicalize team-name inputs so users typing source-specific variants
    # ("Real Madrid CF", "Man City", "曼联") all resolve to the trained team.
    from data.team_normalize import canonicalize as _canon_team
    payload = payload.model_copy(update={
        "home_team": _canon_team(payload.home_team) or payload.home_team,
        "away_team": _canon_team(payload.away_team) or payload.away_team,
    })

    db = init_database(DEFAULT_DB_PATH)
    registry = LeagueRegistry()
    league_key = _normalize_optional_league(payload.league, registry)
    league_obj = registry.leagues.get(league_key) if league_key else None
    is_worldcup = league_key == "world_cup"

    # Continental club competition (UCL/Europa/Libertadores/AFC CL/...) — fit
    # Dixon-Coles on every domestic match from the same continent. This is
    # the trick that makes "Real Madrid vs Bayern" predictable even though
    # we don't have a single UCL fixture in our DB.
    is_continental_club = bool(league_obj and league_obj.is_continental_club)
    is_continental_national = bool(league_obj and league_obj.is_continental_national)

    if is_continental_club:
        continent = league_obj.continent
        sibling_keys = _domestic_keys_for_continent(registry, continent)
        cross_league_pool = True
    elif is_worldcup or is_continental_national:
        sibling_keys = None
        cross_league_pool = False
    else:
        sibling_keys = None
        cross_league_pool = False

    # Continental + WC default to neutral knockout. Honor explicit override.
    auto_neutral = is_worldcup or is_continental_club or is_continental_national
    neutral_site = payload.neutral_site if payload.neutral_site is not None else auto_neutral
    knockout = payload.knockout or _is_knockout_stage(payload.stage) or (
        league_obj.knockout if league_obj else False
    )

    as_of = payload.as_of or date.today()
    since = as_of - timedelta(days=730)

    if is_worldcup or is_continental_national:
        # National-team comp — Elo-only path; we don't have a fitted DC for nations.
        return _predict_national_only(db, payload, league_key, league_obj, neutral_site, knockout, as_of)

    if cross_league_pool and sibling_keys:
        # PREFER the competition's OWN data if we have it (e.g. football-data.org
        # gave us 502 actual UCL matches — much better than cross-league fit).
        # Fall back to the continent-pool only when the comp has no native data.
        # Use TOTAL own-data count (not the 2y window) — continental cups play
        # ~100 matches/year and a 2y window may artificially undercount.
        own_total = db.fetch_matches(league_key=league_key)
        if len(own_total) >= 100:
            # Use ALL of the own data — continental cups are small enough that
            # 3-5 years of history is well below memory limits.
            matches = own_total
            cross_league_pool = False  # downgrade so response metadata is honest
        else:
            frames = [
                db.fetch_matches(league_key=k, since=since, until=as_of)
                for k in sibling_keys
            ]
            non_empty = [f for f in frames if not f.empty]
            if not non_empty:
                return _metadata_only_prediction(
                    db, payload, league_key,
                    reason="no_continental_pool_data",
                    detail=(
                        f"No domestic matches available on {league_obj.continent} to fit "
                        f"{league_obj.name}. Run `python predict.py update` first."
                    ),
                )
            matches = pd.concat(non_empty, ignore_index=True)
    else:
        matches = db.fetch_matches(league_key=league_key, since=since, until=as_of)

    if matches.empty:
        # No model fit possible, but we can still render the matchup card
        # with team badges + H2H (which queries the *full* matches table,
        # not just this league). Returning a partial-but-useful response is
        # better UX than a 400.
        return _metadata_only_prediction(
            db, payload, league_key,
            reason="no_training_data",
            detail=(
                "No historical matches in DB for this league. "
                "Run `python predict.py backfill-results` or load matches."
            ),
        )

    scope = "club"
    home_elo = payload.home_elo
    away_elo = payload.away_elo
    if home_elo is None:
        home_elo = db.latest_rating(payload.home_team, scope=scope, on_or_before=as_of)
    if away_elo is None:
        away_elo = db.latest_rating(payload.away_team, scope=scope, on_or_before=as_of)
    if home_elo is None or away_elo is None:
        fallback_elos = latest_elos(matches)
        home_elo = home_elo if home_elo is not None else fallback_elos.get(payload.home_team)
        away_elo = away_elo if away_elo is not None else fallback_elos.get(payload.away_team)

    # Decide training window. Default is 2 years, but two cases force wider:
    #  (1) continental cup using its own data — cups play sparsely, 2y often <50 matches
    #  (2) league whose latest match is >6 months stale (e.g., Tier-3 leagues we
    #      can only pull historical seasons for via API-Football Free) — using
    #      a 2y window from today would exclude all of it.
    # Need an UNFILTERED probe (the `matches` frame above is already 2y-windowed
    # so its max date would always be near today even when staleness is the issue).
    fit_lookback_days = 730
    if league_key:
        all_for_league = db.fetch_matches(league_key=league_key)
        if not all_for_league.empty:
            latest_overall = pd.to_datetime(all_for_league["date"]).max()
            days_stale = (pd.Timestamp(as_of) - latest_overall).days
            if (is_continental_club and not cross_league_pool) or days_stale > 180:
                fit_lookback_days = 1825  # 5 years — use full history
                # Re-fetch matches with the wider window so the training set actually gets it.
                if not cross_league_pool:
                    matches = all_for_league

    # Branch on model selection. Default is our home-grown DC + Elo blend;
    # other choices are penaltyblog's published implementations (no Elo prior).
    if payload.model in {"ensemble", "market_fused"}:
        fusion = 0.5 if payload.model == "market_fused" else 0.0
        cfg = EnsembleConfig(market_fusion_weight=fusion)
        ens = ensemble_fit(matches, config=cfg, max_goals=payload.max_goals,
                           home_advantage=payload.home_advantage,
                           optimizer_maxiter=6000 if cross_league_pool else 2000)
        ens_result = ensemble_predict(
            ens, payload.home_team, payload.away_team,
            home_elo=home_elo, away_elo=away_elo,
            market_implied=None,  # No odds passed through this endpoint; fusion is moot here.
        )
        probs = ens_result["probabilities"]
        response = {
            "home_team": payload.home_team,
            "away_team": payload.away_team,
            "probabilities": probs,
            "expected_goals": {"home": None, "away": None},
            "score_matrix": [],
            "most_likely_scores": [],
            "neutral_site": neutral_site,
            "knockout": knockout,
            "model": {
                "type": f"ensemble({'+market' if fusion > 0 else 'models_only'})",
                "training_rows": ens.training_rows,
                "as_of": str(as_of),
                "league_key": league_key,
                "elo_scope": scope,
                "home_elo": home_elo,
                "away_elo": away_elo,
                "members_used": ens_result["members_used"],
                "member_contributions": ens_result["contributions"],
                "model_probabilities": ens_result["model_probabilities"],
                "market_fusion_weight": fusion,
                "continental": is_continental_club,
                "continental_pool": sibling_keys if cross_league_pool else None,
            },
        }
        if knockout:
            response["advancement_probabilities"] = {
                "home": probs["home_win"] + probs["draw"] * 0.5,
                "away": probs["away_win"] + probs["draw"] * 0.5,
            }
        return response

    if payload.model in PB_MODEL_FACTORIES:
        pb_pred = pb_fit_and_predict(
            matches,
            payload.home_team,
            payload.away_team,
            model=payload.model,
            max_goals=payload.max_goals,
        )
        response = pb_pred.to_dict(neutral_site=neutral_site, knockout=knockout)
        response["model"] = {
            "type": f"penaltyblog.{payload.model}",
            "training_rows": int(len(matches)),
            "as_of": str(as_of),
            "league_key": league_key,
            "elo_scope": scope,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "continental": is_continental_club,
            "continental_pool": sibling_keys if cross_league_pool else None,
        }
        return response

    # Cross-league fits have way more teams (Europe ≈ 280) → more parameters
    # → optimizer needs more iterations. The default 2000 is fine for a single
    # league but blows up for the continental pool.
    optimizer_maxiter = 6000 if cross_league_pool else 2000
    model = DixonColesModel(
        DixonColesConfig(
            home_advantage=payload.home_advantage,
            max_goals=payload.max_goals,
            xg_blend_weight=payload.xg_blend_weight,
            optimizer_maxiter=optimizer_maxiter,
            lookback_days=fit_lookback_days,
        )
    ).fit(matches, as_of=as_of)

    result = model.predict_match(
        payload.home_team,
        payload.away_team,
        home_elo=home_elo,
        away_elo=away_elo,
        neutral_site=neutral_site,
        knockout=knockout,
        max_goals=payload.max_goals,
    )
    response = result.to_dict()
    response["prediction_available"] = True
    response["model"] = {
        "type": "dixon_coles_elo_cross_league" if cross_league_pool else "dixon_coles_elo",
        "training_rows": model.training_rows_,
        "as_of": str(as_of),
        "league_key": league_key,
        "elo_scope": scope,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "xg_training_rows": model.xg_training_rows_,
        "xg_blend_weight": payload.xg_blend_weight,
        "continental": is_continental_club,
        "continental_pool": sibling_keys if cross_league_pool else None,
    }
    _attach_badges_and_h2h(response, db, payload.home_team, payload.away_team)
    return response


# --- Continental & national-team helpers -----------------------------------

# Maps a continent code to the domestic-league keys whose matches make up
# the cross-league fitting pool for continental club competitions on that
# continent.
#
# Europe is restricted to top-flight leagues only. Including the second
# divisions roughly doubles the team count (~280 → ~560) and the L-BFGS-B
# optimizer struggles to converge on that many parameters within a
# reasonable time. UCL/Europa/Conference participants are overwhelmingly
# top-flight clubs anyway, so we lose almost nothing.
_DOMESTIC_BY_CONTINENT = {
    "europe": [
        "premier_league",
        "la_liga",
        "serie_a",
        "bundesliga",
        "ligue_1",
        "eredivisie",
        "primeira",
        "belgian_pro",
    ],
    "south_america": ["brasileirao", "primera_argentina"],
    "asia": ["j1", "k1", "saudi_pro", "chinese_super_league"],
    "north_america": ["mls", "liga_mx"],
    "oceania": [],   # OFC has no league data; predictions fall back to national Elo.
    "africa": [],  # No African league data yet.
}


def _metadata_only_prediction(
    db: Any,
    payload: "PredictionRequest",
    league_key: str | None,
    *,
    reason: str,
    detail: str,
) -> dict[str, Any]:
    """Build a /predict response when we can't actually fit the model — usually
    because the requested league has no matches in our DB.

    Why this exists: a 400 error closes the door on the UI rendering anything.
    But we DO know enough to show a useful card: team identities (canonical
    names), crest URLs from the warm cache, and any H2H history we have from
    OTHER leagues these teams have played in. So instead of failing hard, we
    return a 200 with ``prediction_available: false`` and a clear reason so
    the UI can show the matchup card minus the prediction probabilities.

    The shape mirrors a successful prediction so existing UI code that reads
    ``result.home_team``, ``result.probabilities`` etc. doesn't crash —
    probabilities are just ``None`` instead of a number.
    """
    response: dict[str, Any] = {
        "home_team": payload.home_team,
        "away_team": payload.away_team,
        "probabilities": None,
        "expected_goals": None,
        "most_likely_scores": [],
        "score_matrix": [],
        "prediction_available": False,
        "unavailable_reason": reason,
        "unavailable_detail": detail,
        "model": {
            "type": "metadata_only",
            "training_rows": 0,
            "league_key": league_key,
        },
    }
    _attach_badges_and_h2h(response, db, payload.home_team, payload.away_team)
    return response


def _attach_badges_and_h2h(
    response: dict[str, Any],
    db: Any,
    home_team: str,
    away_team: str,
) -> None:
    """Enrich a /predict response in place with team badges + H2H history.

    Both additions are pure UI features — they don't affect the model's math.
    Failures are non-fatal: missing badge URL or empty H2H list, never an
    exception. The /predict response stays usable even if both lookups fail.

    * Badges come from ``scrape/team_badges.py``'s disk-cached TSDB lookup
      (added in R36 for the fixture board; reused here for the manual hero
      card so both surfaces look the same).
    * H2H pulls the last 5 matches between these two teams from the local
      ``matches`` table, any home/away order. Each entry carries the
      perspective-of-current-home outcome (W/D/L) so the UI can render the
      colored result chips.
    """
    # Badges — best-effort
    try:
        from scrape.team_badges import fetch_team_badge
        cache_dir = PROJECT_ROOT / "data" / "cache" / "tsdb-teams"
        response["home_badge_url"] = fetch_team_badge(home_team, cache_dir=cache_dir)
        response["away_badge_url"] = fetch_team_badge(away_team, cache_dir=cache_dir)
    except Exception:  # noqa: BLE001 — non-fatal
        response.setdefault("home_badge_url", None)
        response.setdefault("away_badge_url", None)

    # H2H — last 5 between these teams
    try:
        response["h2h_recent"] = _h2h_last_n(db, home_team, away_team, n=5)
    except Exception:  # noqa: BLE001 — non-fatal
        response["h2h_recent"] = []


def _h2h_last_n(db: Any, home_team: str, away_team: str, *, n: int = 5) -> list[dict[str, Any]]:
    """Return the most recent ``n`` matches between ``home_team`` and ``away_team``
    (any direction). Each row includes the result from the perspective of the
    CURRENT prediction's home team — i.e. if Arsenal-the-home-this-week played
    away in a past meeting and won, that row gets ``outcome_for_home="W"``.

    Empty list when no matches found (cup ties between teams that never play
    domestically, brand-new fixtures, etc.).
    """
    import pandas as pd  # noqa: PLC0415

    matches = pd.read_sql(
        "SELECT date, home_team, away_team, home_goals, away_goals, league_key, source "
        "FROM matches "
        "WHERE ((home_team = :h1 AND away_team = :a1) OR (home_team = :h2 AND away_team = :a2)) "
        "  AND home_goals IS NOT NULL AND away_goals IS NOT NULL "
        "ORDER BY date DESC LIMIT :n",
        db.engine,
        params={"h1": home_team, "a1": away_team, "h2": away_team, "a2": home_team, "n": int(n)},
    )
    if matches.empty:
        return []

    out: list[dict[str, Any]] = []
    for row in matches.itertuples(index=False):
        hg, ag = int(row.home_goals), int(row.away_goals)
        # Was the current prediction's home team the home side in this past
        # match? Re-orient the result to the current home's perspective.
        if row.home_team == home_team:
            outcome = "W" if hg > ag else ("L" if hg < ag else "D")
            display = f"{hg}-{ag}"
        else:
            outcome = "W" if ag > hg else ("L" if ag < hg else "D")
            display = f"{ag}-{hg}"  # show from current home's POV (their goals first)
        out.append({
            "date": str(row.date),
            "home_team": row.home_team,
            "away_team": row.away_team,
            "home_goals": hg,
            "away_goals": ag,
            "outcome_for_home": outcome,
            "score_for_current_home": display,
            "venue_for_current_home": "H" if row.home_team == home_team else "A",
            "league_key": row.league_key,
        })
    return out


def _domestic_keys_for_continent(registry: LeagueRegistry, continent: str | None) -> list[str]:
    """Return the registered league keys on a continent (filters to those in the registry)."""
    if not continent:
        return []
    candidates = _DOMESTIC_BY_CONTINENT.get(continent, [])
    return [k for k in candidates if k in registry.leagues]


def _predict_national_only(
    db: Database,
    payload: PredictionRequest,
    league_key: str | None,
    league_obj: Any,
    neutral_site: bool,
    knockout: bool,
    as_of: date,
) -> dict[str, Any]:
    """Elo-only prediction for World Cup / Euro / Copa America / Asian Cup / AFCON.

    No Dixon-Coles fit on national-team data (we don't store match-level
    international results). Uses the classic Elo win-expectancy with a draw
    share calibrated against 20 years of international football (~26%).
    """
    home_elo = payload.home_elo or db.latest_rating(payload.home_team, scope="national", on_or_before=as_of)
    away_elo = payload.away_elo or db.latest_rating(payload.away_team, scope="national", on_or_before=as_of)
    if home_elo is None or away_elo is None:
        missing = payload.home_team if home_elo is None else payload.away_team
        raise ValueError(
            f"Missing national-team Elo for '{missing}'. Run `python predict.py update` "
            "to refresh national ratings (or check the team name spelling)."
        )

    # Elo expectancy → W/D/L. Two refinements vs naive draw_share=0.26:
    #
    # 1. Draw share shrinks as the Elo gap grows. With ~26% baseline for
    #    even matches, a 300+ Elo mismatch realistically draws closer to
    #    15-18% (one-sided games rarely end level). Linear interpolation.
    # 2. Once the favored team's expectancy passes ~85%, the underdog gets
    #    a hard floor of ~3% so we don't print 0.00% on dramatic mismatches.
    HOME_BOOST = 0 if neutral_site else 65
    SCALE = 400.0
    diff = (home_elo + HOME_BOOST) - away_elo
    expected_home = 1.0 / (1 + 10 ** (-diff / SCALE))

    # Adaptive draw share: 0.28 at parity, 0.13 at |diff| >= 400.
    abs_diff = abs(diff)
    draw_share = max(0.13, 0.28 - 0.000375 * abs_diff)

    raw_home = expected_home - draw_share / 2
    raw_away = (1 - expected_home) - draw_share / 2
    # Apply 3% floor for the underdog, then renormalize to sum to 1 - draw_share.
    UNDERDOG_FLOOR = 0.03
    home_p = max(raw_home, UNDERDOG_FLOOR if raw_home < raw_away else 0)
    away_p = max(raw_away, UNDERDOG_FLOOR if raw_away < raw_home else 0)
    total = home_p + away_p
    if total > 0:
        scale = (1 - draw_share) / total
        home_p *= scale
        away_p *= scale

    home_advance = away_advance = None
    if knockout:
        # ET + penalties: collapse the draw probability to advancement,
        # weighted slightly toward the higher-Elo side.
        home_advance = home_p + draw_share * (0.5 + 0.05 * diff / SCALE)
        home_advance = max(min(home_advance, 1.0), 0.0)
        away_advance = 1.0 - home_advance

    response = {
        "home_team": payload.home_team,
        "away_team": payload.away_team,
        "probabilities": {"home_win": home_p, "draw": draw_share, "away_win": away_p},
        "expected_goals": {"home": None, "away": None},
        "most_likely_scores": [],
        "score_matrix": [],
        "neutral_site": neutral_site,
        "knockout": knockout,
        "prediction_available": True,
        "advancement_probabilities": (
            {"home": home_advance, "away": away_advance} if knockout else None
        ),
        "model": {
            "type": "elo_only_national",
            "training_rows": 0,
            "as_of": str(as_of),
            "league_key": league_key,
            "elo_scope": "national",
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": diff,
            "home_boost": HOME_BOOST,
            "draw_share": draw_share,
            "continental": bool(league_obj and league_obj.is_continental_national),
        },
    }
    # Same enrichment as the club path so the UI rendering is symmetric.
    # National team "crests" aren't in TSDB's clubs catalog so badges are
    # usually None — that's fine, the UI falls back to monogram circles.
    _attach_badges_and_h2h(response, db, payload.home_team, payload.away_team)
    return response


def backtest_payload(payload: BacktestRequest) -> dict[str, Any]:
    db = init_database(DEFAULT_DB_PATH)
    registry = LeagueRegistry()
    league_key = _normalize_optional_league(payload.league, registry)
    matches = db.fetch_matches(
        league_key=league_key,
        source=payload.source,
        since=payload.since,
        until=payload.until,
    )
    if matches.empty:
        raise ValueError("No matches found for backtest. Run `python predict.py update` first.")
    result = backtest_dixon_coles(
        matches,
        config=BacktestConfig(
            min_train_matches=payload.min_train_matches,
            max_goals=payload.max_goals,
            refit_every=payload.refit_every,
            xg_blend_weight=payload.xg_blend_weight,
            elo_weight=payload.elo_weight,
        ),
    )
    response = result.to_dict()
    if payload.summary_only:
        response.pop("predictions", None)
    return response


@cli.command()
def init_db(
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="SQLite database path."),
) -> None:
    db = init_database(db_path)
    typer.echo(f"Initialized database: {db.path}")


@cli.command("normalize-teams")
def normalize_teams_cmd(
    dry_run: bool = typer.Option(False, help="Show planned renames without writing."),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="SQLite database path."),
) -> None:
    """Apply team_aliases.yaml to existing matches in the DB.

    Useful after seeding the alias config — fixes already-loaded rows so the
    Dixon-Coles model sees "Real Madrid CF" and "Real Madrid" as the same team.
    Run with ``--dry-run`` first to preview.
    """
    from data.team_normalize import canonicalize

    db = init_database(db_path)
    with db.engine.begin() as conn:
        matches = pd.read_sql("SELECT id, home_team, away_team FROM matches", conn)
    if matches.empty:
        typer.echo("No matches in DB.")
        return

    changes: list[tuple[int, str, str, str, str]] = []
    for row in matches.itertuples(index=False):
        new_home = canonicalize(row.home_team) or row.home_team
        new_away = canonicalize(row.away_team) or row.away_team
        if new_home != row.home_team or new_away != row.away_team:
            changes.append((row.id, row.home_team, new_home, row.away_team, new_away))

    if not changes:
        typer.echo("All team names already canonical. Nothing to do.")
        return

    typer.echo(f"Will rename {len(changes)} match rows:")
    # Summarize unique rename pairs
    pairs: dict[tuple[str, str], int] = {}
    for _, oh, nh, oa, na in changes:
        if oh != nh:
            pairs[(oh, nh)] = pairs.get((oh, nh), 0) + 1
        if oa != na:
            pairs[(oa, na)] = pairs.get((oa, na), 0) + 1
    for (old, new), n in sorted(pairs.items(), key=lambda x: -x[1]):
        typer.echo(f"  {n:>4d}  {old!r:<30s} -> {new!r}")

    if dry_run:
        typer.echo("(dry-run, no changes applied)")
        return

    # Apply.
    with db.engine.begin() as conn:
        for match_id, _, new_home, _, new_away in changes:
            conn.exec_driver_sql(
                "UPDATE matches SET home_team = ?, away_team = ? WHERE id = ?",
                (new_home, new_away, match_id),
            )
    typer.echo(f"Updated {len(changes)} rows.")


@cli.command()
def update(
    leagues: list[str] | None = typer.Option(None, "--league", "-l", help="League key/name/alias."),
    years_back: int = typer.Option(5, min=1, max=10, help="How many recent seasons to refresh."),
    include_ratings: bool = typer.Option(True, help="Also refresh club and national Elo ratings."),
    include_api_football: bool = typer.Option(False, help="Also fetch API-Football fixtures if API key is set."),
    include_players: bool = typer.Option(False, help="Also fetch current-season API-Football player stats."),
    include_xg: bool = typer.Option(False, help="Also enrich football-data results with FBref xG when available."),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="SQLite database path."),
) -> None:
    updater = IncrementalUpdater(db_path=db_path, cache_dir=DEFAULT_CACHE_DIR)
    reports = updater.update_all(
        leagues=leagues,
        years_back=years_back,
        include_ratings=include_ratings,
        include_api_football=include_api_football,
        include_players=include_players,
        include_fbref_xg=include_xg,
    )
    typer.echo(json.dumps([report.to_dict() for report in reports], ensure_ascii=False, indent=2))


@cli.command("warm-team-badges")
def warm_team_badges_cmd(
    league: str | None = typer.Option(None, "--league", "-l", help="Warm only this league key. Default: all leagues from both sources."),
    fdorg_only: bool = typer.Option(False, "--fdorg-only", help="Skip the api-football pass (e.g. when daily quota is tight)."),
    af_only: bool = typer.Option(False, "--api-football-only", help="Skip the fd.org pass."),
) -> None:
    """Bulk-populate the team-badge disk cache from BOTH football-data.org
    and API-Football.

    Why this exists: TheSportsDB's free-tier ``searchteams.php`` returns
    Arsenal for every query (their API is degraded). Manual /predict for a
    team that's not currently in /upcoming would render with the wrong crest
    until that team's badge gets cached. This CLI warms the cache up-front:

      * fd.org covers European top divisions + Brazilian + continental UCL/
        Europa/Libertadores + EC + WC (~280 teams).
      * api-football fills the gaps fd.org doesn't reach: MLS, J1, K1,
        Saudi Pro, Liga MX, Argentine Primera, AFC Champions League, Conf./
        Sudamericana, plus any league with an ``api_football_id``.

    fd.org wins ties — its ``crests.football-data.org/<id>.png`` URLs are
    stable and SVG-quality; api-football's URLs occasionally rotate. The
    second pass only writes cache entries for teams not already there.

    Cost: ~90s fd.org + ~60s api-football = ~2.5min total. Cached forever —
    re-run only after a competition catalog change or when adding a new league.
    """
    from scrape.team_badges import warm_from_fdorg, warm_from_api_football

    cache_dir = PROJECT_ROOT / "data" / "cache" / "tsdb-teams"
    league_keys = [league] if league else None
    combined = {"fdorg": None, "api_football": None}

    if not af_only:
        typer.echo(f"== Pass 1: warming from football-data.org ==")
        typer.echo(f"   cache_dir: {cache_dir}")
        report = warm_from_fdorg(cache_dir=cache_dir, league_keys=league_keys)
        combined["fdorg"] = report
        typer.echo(
            f"   → {report['total_cached']} teams across {report['leagues']} leagues"
            + (f" · {len(report['errors'])} errors" if report["errors"] else "")
        )

    if not fdorg_only:
        typer.echo(f"\n== Pass 2: filling gaps from API-Football ==")
        report = warm_from_api_football(cache_dir=cache_dir, league_keys=league_keys)
        combined["api_football"] = report
        typer.echo(
            f"   → {report['total_cached']} additional teams across {report['leagues']} leagues"
            + (f" · {len(report['errors'])} errors" if report["errors"] else "")
        )

    typer.echo("\n" + json.dumps(combined, ensure_ascii=False, indent=2, default=str))


@cli.command("find-unmatched-fixtures")
def find_unmatched_fixtures_cmd(
    days: int | None = typer.Option(None, "--days", "-d", help="Only inspect history from the last N days (default: all)."),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="SQLite database path."),
    format: str = typer.Option("text", "--format", "-f", help="'text' (human-readable) or 'json'."),
    check_fdorg: bool = typer.Option(False, "--check-fdorg", help="For unresolved fd.org-covered leagues, also query football-data.org directly to catch 'TSDB date wrong + backfill missed it' cases."),
) -> None:
    """Find history predictions that have no matching DB result, with fuzzy candidates.

    Use this when the audit panel under-reports — it shows exactly which
    fixtures aren't joining and why. For each unmatched prediction, looks at
    same-league + same-day DB rows and suggests likely aliases.

    With ``--check-fdorg``, also queries football-data.org's FINISHED list as a
    post-hoc fallback for fixtures the local DB doesn't have. Catches TSDB
    date errors even when backfill missed the league.

    Example output:
        2026-05-17 | ligue_1 | Brest vs Angers
          ★ (football-data.org) Stade Brestois vs Angers  [sim=1.00]
            → suggested alias: "Stade Brestois" ↔ "Brest"
    """
    from data.alias_audit import find_unmatched_fixtures
    from data.database import Database

    db = Database(db_path)
    report = find_unmatched_fixtures(
        db, days_back=days,
        check_fdorg=check_fdorg,
        cache_dir=PROJECT_ROOT / "data" / "cache" / "football-data-org",
    )

    if format == "json":
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return

    # Text format
    by_reason = report.get("by_reason") or {}
    typer.echo(
        f"Past-dated fixtures in history: {report['n_past_fixtures']}\n"
        f"  ✓ matched in DB: {report['n_matched']}\n"
        f"  ✗ unmatched:     {report['n_unmatched']}"
    )
    if by_reason:
        for reason, n in sorted(by_reason.items(), key=lambda x: -x[1]):
            typer.echo(f"      · {reason}: {n}")
    typer.echo("")

    for u in report["unmatched"]:
        typer.echo(f"\n{u['date']} | {u['league_key']} | {u['raw_home']} vs {u['raw_away']}")
        typer.echo(f"  canonicalized → {u['canon_home']!r} vs {u['canon_away']!r}")

        if u.get("nearby_match"):
            n = u["nearby_match"]
            flip_note = " (home/away flipped)" if n.get("flipped") else ""
            sign = "+" if n["ds"] > u["date"] else "−"
            typer.echo(
                f"  ⚠ likely date mismatch — same team pair on {n['ds']} "
                f"({sign}{n['days_off']}d){flip_note}"
            )
            typer.echo(
                f"     DB has: ({n['source']}) {n['home_team']!r} vs {n['away_team']!r}"
            )
            typer.echo(
                f"     → TheSportsDB upcoming snapshot probably had a stale date; "
                f"prediction was made for the wrong day."
            )
            continue

        if not u["candidates"]:
            typer.echo(f"  ✗ {u['reason']} — no DB candidates (league probably wasn't backfilled)")
            continue
        for c in u["candidates"]:
            marker = "★" if c["similarity"] > 0.85 else " "
            typer.echo(
                f"  {marker} ({c['source']}) {c['home_team']!r} vs {c['away_team']!r}  "
                f"[sim={c['similarity']:.2f}]"
            )
            if c["suggested_alias"]:
                typer.echo(f"     → suggested alias: {c['suggested_alias']}")


@cli.command("manual-result")
def manual_result_cmd(
    league: str = typer.Option(..., "--league", "-l", help="League key, name, or alias (e.g. '英超', 'premier_league')."),
    match_date: str = typer.Option(..., "--date", "-d", help="Match date YYYY-MM-DD."),
    home: str = typer.Option(..., "--home", help="Home team name (canonicalized on insert)."),
    away: str = typer.Option(..., "--away", help="Away team name (canonicalized on insert)."),
    score: str = typer.Option(..., "--score", help="Final score 'H-A', e.g. '2-1'."),
    season: str | None = typer.Option(None, "--season", help="Override season (otherwise derived from date)."),
    neutral: bool = typer.Option(False, "--neutral", help="Match was at a neutral venue."),
    stage: str | None = typer.Option(None, "--stage", help="Optional stage label, e.g. 'group', 'semifinal'."),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="SQLite database path."),
) -> None:
    """Patch one fixture into the matches table as ``source="manual"``.

    For leagues/matches none of the automated sources cover — or fixing one
    they got wrong. Manual entries beat all automated sources in the conflict
    resolver (see data/source_resolver.py).

    Example:
        python predict.py manual-result --league k_league_2 \\
            --date 2026-05-17 --home Suwon --away Busan --score 2-1
    """
    from data.database import Database
    from data.manual_results import submit_manual_result
    from datetime import date as _date

    try:
        d = _date.fromisoformat(match_date)
    except ValueError as exc:
        raise typer.BadParameter(f"--date must be YYYY-MM-DD; got {match_date!r}") from exc
    try:
        h_str, a_str = score.split("-", 1)
        hg, ag = int(h_str.strip()), int(a_str.strip())
    except (ValueError, AttributeError) as exc:
        raise typer.BadParameter(f"--score must look like '2-1'; got {score!r}") from exc

    db = Database(db_path)
    try:
        report = submit_manual_result(
            db,
            league=league,
            date=d,
            home_team=home,
            away_team=away,
            home_goals=hg,
            away_goals=ag,
            season=season,
            neutral_site=neutral or None,
            stage=stage,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@cli.command("backfill-results")
def backfill_results_cmd(
    days: int = typer.Option(7, min=1, max=60, help="Look back this many days for finished matches."),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="SQLite database path."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress per-league progress lines."),
) -> None:
    """Lightweight daily refresh: pull the last N days of FINISHED matches.

    Routes per league: football-data.org first (cheaper, no season constraint),
    then API-Football fallback. Skips leagues with no reachable source.
    The prediction-audit module consumes the output of this — without it
    the audit panel stays empty even after games finish.

    By default, prints one progress line per league to **stderr** while running,
    so `--quiet` suppresses them and JSON output on stdout stays pipe-safe.
    """
    import sys
    from data.database import Database
    from data.result_backfill import backfill_recent_results

    def _print_progress(row: dict[str, Any]) -> None:
        if quiet:
            return
        league = row.get("league_key", "?")
        source = row.get("source") or "—"
        if row.get("error"):
            tag, detail = "ERR ", str(row["error"])[:80]
        elif row.get("skipped"):
            tag, detail = "SKIP", f"({row['skipped']})"
        else:
            inserted = row.get("inserted", 0)
            fetched = row.get("fetched", 0)
            tag = "✓   " if inserted else "·   "
            detail = f"{inserted:+d} ({fetched} fetched)" if fetched else f"{inserted:+d}"
        print(f"  {tag} {league:<28} {source:<20} {detail}", file=sys.stderr, flush=True)

    db = Database(db_path)
    report = backfill_recent_results(
        db, days_back=days, progress_callback=_print_progress,
    )
    if not quiet:
        t = report["totals"]
        print(
            f"\n→ {t['leagues_reached']} leagues reached · "
            f"{t['inserted']} matches inserted · "
            f"{t['errors']} errors · "
            f"{report['duration_s']}s",
            file=sys.stderr, flush=True,
        )
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2, default=str))


@cli.command("backfill-api-xg")
def backfill_api_xg_cmd(
    league: str = typer.Option(..., help="League key or Chinese alias (英超, 西甲, etc.)."),
    season: int = typer.Option(..., help="Season start year, e.g. 2024 for 2024-25 season."),
    limit: int = typer.Option(
        None,
        help="Cap how many fixtures to enrich. API-Football free tier is 100/day; "
             "EPL season is 380 fixtures so plan accordingly. None = enrich all that "
             "are missing xG, stopping early if the daily quota gets exhausted.",
    ),
    skip_if_present: bool = typer.Option(
        True,
        help="If True, only hit the API for fixtures where home_xg IS NULL. Defaults to True.",
    ),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="SQLite database path."),
) -> None:
    """Backfill home_xg / away_xg from API-Football's /fixtures/statistics.

    Why this exists: the FBref xG scraper has been silently 403'd by FBref's
    anti-bot for an unknown but extended period. As a result, no league has
    populated home_xg/away_xg fields, and the xG-blending feature has been
    a no-op in production. Discovered via /diagnostics/ablation's
    silent_features warning (see commit f608e22).

    API-Football has the same data under a different name: each fixture's
    /fixtures/statistics response contains an ``expected_goals`` stat per team.
    One request per fixture. EPL one season = ~380 requests = 4 days of free
    quota. Paid tier removes the constraint.

    Idempotent: defaults to skipping fixtures that already have xG. Pass
    --no-skip-if-present to overwrite.
    """
    from scrape.api_football import client_from_env, is_daily_quota_exhausted  # noqa: PLC0415
    from sqlalchemy import text  # noqa: PLC0415

    db = init_database(db_path)
    registry = LeagueRegistry()
    league_key = _normalize_optional_league(league, registry)
    if not league_key:
        typer.echo(f"Unknown league: {league}", err=True)
        raise typer.Exit(1)
    lg = registry.leagues.get(league_key)
    if not lg or lg.api_football_id is None:
        typer.echo(f"League '{league_key}' has no api_football_id configured.", err=True)
        raise typer.Exit(1)

    af = client_from_env()
    if af is None:
        typer.echo("No API-Football key configured (set API_FOOTBALL_KEY or FOOTBALL_API_KEY).", err=True)
        raise typer.Exit(1)
    if is_daily_quota_exhausted():
        typer.echo("API-Football daily quota already exhausted. Wait until tomorrow.", err=True)
        raise typer.Exit(1)

    # 1. Pull the fixture list for this league/season from API-Football. We
    #    need the API-Football fixture IDs to call /fixtures/statistics.
    typer.echo(f"Fetching {league_key} {season}-{season+1} fixture list…", err=True)
    fixtures = af.fetch_fixtures(league_id=int(lg.api_football_id), season=season)
    typer.echo(f"  → {len(fixtures)} fixtures", err=True)
    if not fixtures:
        raise typer.Exit(0)

    # 2. For each fixture, look up our matches row by (date, home_team, away_team)
    #    and check if xG is already filled. We canonicalize team names so
    #    api-football's "Manchester United" matches our normalized form.
    from data.team_normalize import canonicalize  # noqa: PLC0415

    needs_xg: list[dict[str, Any]] = []
    with db.engine.begin() as conn:
        for fx in fixtures:
            if fx.get("fixture", {}).get("status", {}).get("short") != "FT":
                continue  # only enrich finished fixtures
            home = fx["teams"]["home"]["name"]
            away = fx["teams"]["away"]["name"]
            d = fx["fixture"]["date"][:10]  # YYYY-MM-DD
            home_canon = canonicalize(home) or home
            away_canon = canonicalize(away) or away
            row = conn.execute(text(
                "SELECT id, home_xg, away_xg FROM matches "
                "WHERE date = :d AND league_key = :lg "
                "AND (home_team = :h OR home_team = :hc) "
                "AND (away_team = :a OR away_team = :ac) "
                "LIMIT 1"
            ), {"d": d, "lg": league_key, "h": home, "hc": home_canon,
                 "a": away, "ac": away_canon}).fetchone()
            if row is None:
                continue
            if skip_if_present and row[1] is not None and row[2] is not None:
                continue
            needs_xg.append({
                "fx_id": fx["fixture"]["id"],
                "match_id": row[0],
                "label": f"{d} {home} vs {away}",
            })

    typer.echo(f"  → {len(needs_xg)} fixtures need xG enrichment", err=True)
    if limit:
        needs_xg = needs_xg[:limit]
        typer.echo(f"  → limited to first {limit}", err=True)
    if not needs_xg:
        typer.echo("Nothing to do.", err=True)
        raise typer.Exit(0)

    # 3. For each, hit /fixtures/statistics and UPDATE the row.
    n_updated = 0
    n_skipped = 0
    n_quota_hit = 0
    for i, job in enumerate(needs_xg, 1):
        if is_daily_quota_exhausted():
            n_quota_hit = len(needs_xg) - i + 1
            typer.echo(f"  daily quota exhausted at fixture {i}, stopping early", err=True)
            break
        try:
            hxg, axg = af.fetch_fixture_xg(fixture_id=job["fx_id"])
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  [{i}/{len(needs_xg)}] {job['label']}: ERR {exc}", err=True)
            n_skipped += 1
            continue
        if hxg is None and axg is None:
            n_skipped += 1
            continue
        with db.engine.begin() as conn:
            conn.execute(text(
                "UPDATE matches SET home_xg = COALESCE(:hx, home_xg), "
                "away_xg = COALESCE(:ax, away_xg) WHERE id = :id"
            ), {"hx": hxg, "ax": axg, "id": job["match_id"]})
        n_updated += 1
        if i % 10 == 0:
            typer.echo(f"  [{i}/{len(needs_xg)}] {job['label']}: home_xg={hxg} away_xg={axg}", err=True)

    typer.echo(json.dumps({
        "league": league_key, "season": season,
        "fixtures_inspected": len(fixtures),
        "needed_xg": len(needs_xg),
        "updated": n_updated, "skipped_no_data": n_skipped,
        "stopped_at_quota": n_quota_hit,
    }, indent=2))


@cli.command("snapshot-upcoming")
def snapshot_upcoming_cmd(
    days_ahead: int = typer.Option(
        None,
        help="Override SETTINGS.upcoming_snapshot_days_ahead. Default uses the env-configured value (14).",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print the row-count summary."),
) -> None:
    """Force one /upcoming snapshot from the command line.

    Why this exists: the in-process APScheduler runs inside the serve process,
    so when the FastAPI server isn't running, NO snapshots fire and the audit
    history starves. This CLI is the same code path the scheduler calls — but
    it can be triggered by launchd/cron independently of whether the web
    server is up. Re-uses the on-disk caches, so back-to-back calls are cheap.

    Wire it into launchd (`~/Library/LaunchAgents/`) to get reliable cron-style
    snapshots that survive server restarts and crashes.
    """
    from data.history_store import SHARD_DIR  # noqa: PLC0415

    registry = LeagueRegistry()
    league_keys = sorted({league.key for league in registry.all()})
    days = days_ahead if days_ahead is not None else SETTINGS.upcoming_snapshot_days_ahead

    if not quiet:
        typer.echo(f"snapshot-upcoming: leagues={len(league_keys)} days_ahead={days}", err=True)

    # Mirrors _scheduled_upcoming_snapshot but with progress + return code.
    try:
        payload = _compute_upcoming_payload(
            league_keys=league_keys,
            days_ahead=days,
            include_predictions=True,
        )
    except Exception as exc:  # noqa: BLE001 — return non-zero so cron sees failure
        typer.echo(f"snapshot-upcoming FAILED: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1)

    fixture_count = payload.get("fixture_count", 0)
    typer.echo(
        json.dumps(
            {
                "ok": True,
                "fixture_count": fixture_count,
                "days_ahead": days,
                "leagues_queried": len(league_keys),
                "history_dir": str(SHARD_DIR),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@cli.command("coverage")
def coverage_cmd(
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="SQLite database path."),
    only_empty: bool = typer.Option(False, help="Show only leagues with no matches."),
) -> None:
    report = build_coverage_report(db_path=db_path)
    leagues = report["leagues"]
    if only_empty:
        leagues = [league for league in leagues if league["status"] == "empty"]
    typer.echo(
        json.dumps(
            {"summary": report["summary"], "leagues": leagues},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


@cli.command("doctor")
def doctor_cmd(
    live: bool = typer.Option(False, help="Probe API-Football with the configured key."),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="SQLite database path."),
) -> None:
    typer.echo(
        json.dumps(
            build_doctor_report(db_path=db_path, live_api=live),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


@cli.command("api-football-leagues")
def api_football_leagues_cmd(
    country: str | None = typer.Option(None, help="Country name, e.g. China."),
    search: str | None = typer.Option(None, help="Search term, e.g. Super League."),
    league_id: int | None = typer.Option(None, "--league-id", help="Known API-Football league id."),
    season: int | None = typer.Option(None, help="Optional season year."),
) -> None:
    frame = api_football.discover_leagues(
        country=country,
        search=search,
        league_id=league_id,
        season=season,
    )
    typer.echo(
        json.dumps(
            frame.drop(columns=["raw"], errors="ignore").to_dict(orient="records"),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


@cli.command("backtest")
def backtest_cmd(
    league: str | None = typer.Option(None, "--league", "-l"),
    source: str | None = typer.Option(None, help="Optional data source filter."),
    since: str | None = typer.Option(None, help="YYYY-MM-DD"),
    until: str | None = typer.Option(None, help="YYYY-MM-DD"),
    min_train_matches: int = typer.Option(80, min=20),
    refit_every: int = typer.Option(1, min=1),
    max_goals: int = typer.Option(8, min=4, max=15),
    xg_blend_weight: float = typer.Option(0.35, min=0.0, max=1.0),
    include_predictions: bool = typer.Option(False, help="Print every match prediction, not just summary."),
) -> None:
    payload = BacktestRequest(
        league=league,
        source=source,
        since=date.fromisoformat(since) if since else None,
        until=date.fromisoformat(until) if until else None,
        min_train_matches=min_train_matches,
        refit_every=refit_every,
        max_goals=max_goals,
        xg_blend_weight=xg_blend_weight,
    )
    result = backtest_payload(payload)
    output = result if include_predictions else result["summary"]
    typer.echo(json.dumps(output, ensure_ascii=False, indent=2, default=str))


@cli.command()
def predict(
    home_team: str = typer.Argument(...),
    away_team: str = typer.Argument(...),
    league: str | None = typer.Option(None, "--league", "-l"),
    home_elo: float | None = typer.Option(None),
    away_elo: float | None = typer.Option(None),
    neutral_site: bool | None = typer.Option(None),
    stage: str | None = typer.Option(None),
    knockout: bool = typer.Option(False),
    max_goals: int = typer.Option(8, min=4, max=15),
    xg_blend_weight: float = typer.Option(0.35, min=0.0, max=1.0),
    as_of: str | None = typer.Option(None, help="YYYY-MM-DD"),
) -> None:
    payload = PredictionRequest(
        home_team=home_team,
        away_team=away_team,
        league=league,
        home_elo=home_elo,
        away_elo=away_elo,
        neutral_site=neutral_site,
        stage=stage,
        knockout=knockout,
        max_goals=max_goals,
        xg_blend_weight=xg_blend_weight,
        as_of=date.fromisoformat(as_of) if as_of else None,
    )
    typer.echo(json.dumps(predict_payload(payload), ensure_ascii=False, indent=2))


@cli.command()
def export(
    table: str = typer.Argument(..., help="matches, ratings, players, player_season_stats, or update_state."),
    output: Path | None = typer.Option(None, "--output", "-o"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="SQLite database path."),
) -> None:
    db = init_database(db_path)
    output = output or EXPORT_DIR / f"{table}.csv"
    path = db.export_csv(table, output)
    typer.echo(f"Exported {table}: {path}")


@cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8001, help="Bind port (fork defaults to 8001 to avoid clashing with sibling on 8000)."),
    reload: bool = typer.Option(False, help="Reload on code changes."),
) -> None:
    uvicorn.run("predict:app", host=host, port=port, reload=reload)


def _run_update_job(
    leagues: list[str] | None = None,
    years_back: int = 5,
    include_ratings: bool = True,
    include_api_football: bool = False,
    include_players: bool = False,
    include_fbref_xg: bool = False,
) -> None:
    updater = IncrementalUpdater(db_path=DEFAULT_DB_PATH, cache_dir=DEFAULT_CACHE_DIR)
    updater.update_all(
        leagues=leagues,
        years_back=years_back,
        include_ratings=include_ratings,
        include_api_football=include_api_football,
        include_players=include_players,
        include_fbref_xg=include_fbref_xg,
    )


def _start_scheduler() -> BackgroundScheduler:
    timezone = ZoneInfo(SETTINGS.timezone)
    scheduler = BackgroundScheduler(timezone=timezone)
    scheduler.add_job(
        run_daily_update,
        trigger="cron",
        hour=SETTINGS.update_hour,
        minute=SETTINGS.update_minute,
        id="daily_incremental_update",
        replace_existing=True,
        kwargs={
            "db_path": DEFAULT_DB_PATH,
            "cache_dir": DEFAULT_CACHE_DIR,
            "include_api_football": _daily_api_football_enabled(),
            "include_players": False,
            "include_fbref_xg": _daily_fbref_xg_enabled(),
        },
    )
    if SETTINGS.results_backfill_enabled:
        scheduler.add_job(
            _scheduled_results_backfill,
            trigger="cron",
            hour=SETTINGS.results_backfill_hour,
            minute=SETTINGS.results_backfill_minute,
            id="daily_results_backfill",
            replace_existing=True,
        )
    if SETTINGS.upcoming_snapshot_enabled:
        scheduler.add_job(
            _scheduled_upcoming_snapshot,
            trigger="cron",
            hour=SETTINGS.upcoming_snapshot_hours,  # "9,15,21,3" → cron-multi-hour
            minute=0,
            id="periodic_upcoming_snapshot",
            replace_existing=True,
        )
    scheduler.start()
    return scheduler


def _scheduled_results_backfill() -> None:
    """Wrapper so the scheduler can call backfill without needing kwargs."""
    from data.database import Database
    from data.result_backfill import backfill_recent_results

    db = Database(DEFAULT_DB_PATH)
    try:
        backfill_recent_results(
            db,
            days_back=SETTINGS.results_backfill_days,
            cache_dir=DEFAULT_CACHE_DIR,
        )
    except Exception:  # noqa: BLE001 — never let a scheduled job crash the loop
        import traceback
        traceback.print_exc()


def _scheduled_upcoming_snapshot() -> None:
    """Periodic /upcoming computation so audit data accumulates unattended.

    Without this, history.jsonl only grows when somebody opens the web UI. Audit
    starvation is the failure mode — we want it self-feeding.

    Reuses ``_compute_upcoming_payload`` (the same code path the HTTP endpoint
    runs), which already appends every fixture to history.jsonl via
    ``append_snapshot``. Caching is preserved: TSDB has 6h TTL, fd.org SCHEDULED
    has 24h, so this job is cheap once warmed up.
    """
    registry = LeagueRegistry()
    league_keys = sorted({league.key for league in registry.all()})
    try:
        _compute_upcoming_payload(
            league_keys=league_keys,
            days_ahead=SETTINGS.upcoming_snapshot_days_ahead,
            include_predictions=True,
        )
    except Exception:  # noqa: BLE001 — never let a scheduled job crash the loop
        import traceback
        traceback.print_exc()


def _scheduler_enabled() -> bool:
    return SETTINGS.scheduler_enabled


def _daily_api_football_enabled() -> bool:
    return SETTINGS.daily_api_football


def _daily_fbref_xg_enabled() -> bool:
    return SETTINGS.daily_fbref_xg


def _normalize_optional_league(value: str | None, registry: LeagueRegistry) -> str | None:
    if value is None:
        return None
    return registry.normalize(value)


def _is_knockout_stage(stage: str | None) -> bool:
    if not stage:
        return False
    normalized = stage.strip().casefold()
    return normalized in {
        "round_of_32",
        "round of 32",
        "round_of_16",
        "round of 16",
        "last 16",
        "quarter_final",
        "quarter-final",
        "semifinal",
        "semi-final",
        "third_place",
        "third-place",
        "final",
        "淘汰赛",
        "决赛",
        "半决赛",
        "四分之一决赛",
        "八分之一决赛",
    }


if __name__ == "__main__":
    cli()
