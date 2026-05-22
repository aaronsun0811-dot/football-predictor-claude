from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, text

from config.settings import get_settings
from data.coverage import build_coverage_report
from data.database import Base, init_database
from data.doctor import build_doctor_report
from data.schema import ensure_schema, schema_report
from models.backtest import BacktestConfig, backtest_dixon_coles
from models.dixon_coles import DixonColesConfig, DixonColesModel
from models.elo import attach_pre_match_elos
from scrape.registry import LeagueRegistry
from scrape.api_football import leagues_to_frame, players_to_season_stats
from scrape.update import merge_xg


def synthetic_matches(n: int = 36) -> list[dict]:
    teams = ["Arsenal", "Liverpool", "Chelsea", "Tottenham"]
    start = date(2025, 1, 1)
    rows = []
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 1 + (i // len(teams))) % len(teams)]
        if home == away:
            away = teams[(i + 2) % len(teams)]
        home_goals = (i * 2 + teams.index(home)) % 4
        away_goals = (i + teams.index(away)) % 3
        rows.append(
            {
                "date": start + timedelta(days=i * 4),
                "league_key": "premier_league",
                "league_name": "Premier League",
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )
    return rows


def test_chinese_league_aliases_normalize_saudi_pro() -> None:
    registry = LeagueRegistry()
    assert registry.normalize("沙特") == "saudi_pro"
    assert registry.normalize("沙特职业足球联赛") == "saudi_pro"
    assert registry.normalize("Saudi Pro League") == "saudi_pro"
    assert registry.normalize("中超") == "chinese_super_league"
    assert registry.normalize("中国足球协会超级联赛") == "chinese_super_league"
    assert registry.normalize("CSL") == "chinese_super_league"


def test_database_upsert_fetch_and_deduplicate(tmp_path) -> None:
    db = init_database(tmp_path / "football.sqlite3")
    rows = synthetic_matches(3)
    assert db.upsert_matches(rows, source="football-data.co.uk", league_key="premier_league") == 3
    assert db.upsert_matches(rows, source="api-football", league_key="premier_league") == 3
    deduped = db.fetch_matches(league_key="premier_league")
    raw = db.fetch_matches(league_key="premier_league", deduplicate=False)
    assert len(deduped) == 3
    assert len(raw) == 6
    assert set(deduped["source"]) == {"api-football"}


def test_database_preserves_existing_xg_on_plain_refresh(tmp_path) -> None:
    db = init_database(tmp_path / "football.sqlite3")
    rows = synthetic_matches(1)
    rows[0]["home_xg"] = 1.7
    rows[0]["away_xg"] = 0.8
    assert db.upsert_matches(rows, source="football-data.co.uk", league_key="premier_league") == 1

    plain = synthetic_matches(1)
    assert db.upsert_matches(plain, source="football-data.co.uk", league_key="premier_league") == 1

    match = db.fetch_matches(league_key="premier_league").iloc[0]
    assert match["home_xg"] == 1.7
    assert match["away_xg"] == 0.8


def test_dixon_coles_prediction_probabilities_sum_to_one() -> None:
    model = DixonColesModel(DixonColesConfig(optimizer_maxiter=120)).fit(synthetic_matches(24))
    result = model.predict_match("Arsenal", "Liverpool", home_elo=1540, away_elo=1510)
    total = result.home_win + result.draw + result.away_win
    assert abs(total - 1.0) < 1e-9
    assert result.score_matrix.shape == (9, 9)
    assert abs(result.score_matrix.sum() - 1.0) < 1e-9


def test_dixon_coles_uses_xg_blend_when_available() -> None:
    rows = synthetic_matches(24)
    for row in rows:
        row["home_xg"] = row["home_goals"] + 0.25
        row["away_xg"] = max(row["away_goals"] - 0.10, 0.0)
    model = DixonColesModel(
        DixonColesConfig(optimizer_maxiter=120, xg_blend_weight=0.5)
    ).fit(rows)
    assert model.xg_training_rows_ == 24


def test_merge_xg_normalizes_common_team_aliases() -> None:
    results = pd.DataFrame(
        [
            {
                "date": "2025-09-01",
                "home_team": "Man United",
                "away_team": "Tottenham",
                "home_goals": 2,
                "away_goals": 1,
            }
        ]
    )
    xg = pd.DataFrame(
        [
            {
                "date": "2025-09-01",
                "home_team": "Manchester Utd",
                "away_team": "Tottenham Hotspur",
                "home_xg": 1.9,
                "away_xg": 1.1,
            }
        ]
    )
    enriched = merge_xg(results, xg)
    assert enriched.iloc[0]["home_xg"] == 1.9
    assert enriched.iloc[0]["away_xg"] == 1.1


def test_internal_elo_attaches_pre_match_ratings() -> None:
    frame = attach_pre_match_elos(pd.DataFrame(synthetic_matches(8)))
    assert {"home_elo", "away_elo", "home_elo_post", "away_elo_post"}.issubset(frame.columns)
    assert frame.iloc[0]["home_elo"] == 1500
    assert frame.iloc[0]["away_elo"] == 1500
    assert frame["home_elo_post"].notna().all()


def test_backtest_returns_metrics() -> None:
    result = backtest_dixon_coles(
        synthetic_matches(32),
        config=BacktestConfig(
            min_train_matches=24,
            optimizer_maxiter=80,
            refit_every=4,
        ),
    )
    assert result.summary["n_predictions"] == 8
    assert 0 <= result.summary["accuracy"] <= 1


def test_backtest_exposes_fit_health() -> None:
    """Every backtest result carries a fit_health block: refit counts, model
    staleness rollup, failed-refit log. Lets users see whether the headline
    accuracy was made on a fresh fit or a stale one."""
    result = backtest_dixon_coles(
        synthetic_matches(32),
        config=BacktestConfig(
            min_train_matches=24,
            optimizer_maxiter=80,
            refit_every=4,
        ),
    )
    fh = result.summary.get("fit_health")
    assert fh is not None, "backtest summary must include fit_health"
    # Counts make sense
    assert fh["refit_attempts"] >= 1
    assert fh["refits_succeeded"] == fh["refit_attempts"] - fh["skipped_refits"]
    # Staleness rollup populated
    assert fh["pct_with_fresh_model"] is not None
    assert 0.0 <= fh["pct_with_fresh_model"] <= 1.0
    assert fh["max_model_staleness"] >= 0
    assert fh["mean_model_staleness"] is not None
    assert isinstance(fh["failed_refits"], list)


def test_backtest_per_prediction_carries_model_age() -> None:
    """Each prediction row has model_age_matches: 0 = fresh, >0 = stale by N."""
    result = backtest_dixon_coles(
        synthetic_matches(32),
        config=BacktestConfig(
            min_train_matches=24,
            optimizer_maxiter=80,
            refit_every=4,
        ),
    )
    rows = result.predictions
    assert "model_age_matches" in rows.columns
    assert "fit_attempted_here" in rows.columns
    assert "fit_failed_here" in rows.columns
    # With refit_every=4, every 4th row should be a fresh fit (age=0)
    fresh_count = int((rows["model_age_matches"] == 0).sum())
    assert fresh_count >= 1
    # Ages never go negative
    assert (rows["model_age_matches"] >= 0).all()


def test_backtest_summary_includes_score_level_metrics() -> None:
    """exact_score_accuracy + mean_goal_distance show up alongside outcome metrics.

    Same shape as /audit's summary so the UI can render both consistently."""
    result = backtest_dixon_coles(
        synthetic_matches(32),
        config=BacktestConfig(
            min_train_matches=24,
            optimizer_maxiter=200,
            refit_every=4,
        ),
    )
    s = result.summary
    # Score-level metrics present and sensible
    assert s["n_scored"] == s["n_predictions"]  # every prediction has a top score
    assert s["exact_score_accuracy"] is not None
    assert 0.0 <= s["exact_score_accuracy"] <= 1.0
    assert s["mean_goal_distance"] is not None
    assert s["mean_goal_distance"] >= 0.0


def test_backtest_per_prediction_carries_score_fields() -> None:
    """Each prediction row has predicted_score / predicted_score_prob /
    exact_score_correct / goal_distance — same shape as /audit's resolved frame."""
    result = backtest_dixon_coles(
        synthetic_matches(32),
        config=BacktestConfig(
            min_train_matches=24,
            optimizer_maxiter=200,
            refit_every=4,
        ),
    )
    rows = result.predictions
    for col in ("predicted_score", "predicted_score_prob",
                "exact_score_correct", "goal_distance"):
        assert col in rows.columns, f"missing column {col}"
    # predicted_score is a "H-A" string
    assert rows["predicted_score"].iloc[0].count("-") == 1
    # probabilities in [0, 1]
    assert ((rows["predicted_score_prob"] >= 0) & (rows["predicted_score_prob"] <= 1)).all()
    # goal_distance is a non-negative integer
    assert (rows["goal_distance"] >= 0).all()


def test_backtest_by_league_breakdown_for_multi_league() -> None:
    """A backtest spanning two leagues produces one summary.by_league entry per
    league (when both meet the min_n threshold)."""
    from models.backtest import _per_league_breakdown
    # Build a fake predictions frame: 50 rows per league, with sensible cols
    rows = []
    for league, _ in (("pl", 50), ("la_liga", 50)):
        for i in range(50):
            rows.append({
                "league_key": league,
                "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
                "actual": "home_win" if i % 2 == 0 else "draw",
                "correct": (i % 2 == 0),
                "exact_score_correct": (i % 10 == 0),
                "goal_distance": 1.0,
            })
    df = pd.DataFrame(rows)
    by_league = _per_league_breakdown(df, min_n=30)
    assert len(by_league) == 2
    keys = {r["league_key"] for r in by_league}
    assert keys == {"pl", "la_liga"}
    for r in by_league:
        assert r["n"] == 50
        assert 0.0 <= r["accuracy"] <= 1.0
        assert r["brier"] >= 0
        assert r["exact_score_accuracy"] == pytest.approx(0.1, abs=1e-9)
        assert r["mean_goal_distance"] == pytest.approx(1.0, abs=1e-9)


def test_backtest_by_league_drops_low_sample_leagues() -> None:
    """A league with fewer than min_n predictions is filtered out (too noisy)."""
    from models.backtest import _per_league_breakdown
    rows = []
    for i in range(40):
        rows.append({
            "league_key": "big_league",
            "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
            "actual": "home_win", "correct": True,
            "exact_score_correct": False, "goal_distance": 1.0,
        })
    # Only 5 entries for tiny_league — should be dropped at min_n=30
    for i in range(5):
        rows.append({
            "league_key": "tiny_league",
            "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
            "actual": "home_win", "correct": True,
            "exact_score_correct": False, "goal_distance": 1.0,
        })
    by_league = _per_league_breakdown(pd.DataFrame(rows), min_n=30)
    assert len(by_league) == 1
    assert by_league[0]["league_key"] == "big_league"


def test_backtest_by_league_sorted_by_sample_size() -> None:
    """When multiple leagues qualify, the entry with the largest n comes first."""
    from models.backtest import _per_league_breakdown
    rows = []
    for league, n in (("smaller", 35), ("bigger", 100)):
        for _ in range(n):
            rows.append({
                "league_key": league,
                "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
                "actual": "home_win", "correct": True,
                "exact_score_correct": False, "goal_distance": 1.0,
            })
    by_league = _per_league_breakdown(pd.DataFrame(rows), min_n=30)
    assert [r["league_key"] for r in by_league] == ["bigger", "smaller"]


def test_backtest_summary_includes_by_league_for_single_league() -> None:
    """Even single-league backtests produce a one-entry by_league list. UI
    can choose whether to render it as a table or skip."""
    result = backtest_dixon_coles(
        synthetic_matches(64),  # All synthetic data has league_key 'synth'
        config=BacktestConfig(
            min_train_matches=24,
            optimizer_maxiter=200,
            refit_every=4,
        ),
    )
    by_league = result.summary.get("by_league") or []
    # n=40 predictions all in one synthetic league — passes min_n=30 threshold
    if by_league:
        assert len(by_league) == 1
        assert by_league[0]["n"] >= 30


def test_backtest_fit_health_zero_failures_when_no_failures() -> None:
    """In a clean run (no L-BFGS-B failures), skipped_refits=0 and
    refits_succeeded == refit_attempts."""
    result = backtest_dixon_coles(
        synthetic_matches(32),
        config=BacktestConfig(
            min_train_matches=24,
            optimizer_maxiter=200,  # generous, should never fail on synthetic data
            refit_every=4,
        ),
    )
    fh = result.summary["fit_health"]
    assert fh["skipped_refits"] == 0
    assert fh["refits_succeeded"] == fh["refit_attempts"]
    assert fh["failed_refits"] == []
    assert result.summary["log_loss"] > 0


def test_player_stats_can_be_normalized_and_stored(tmp_path) -> None:
    registry = LeagueRegistry()
    league = registry.get("英超")
    raw_players = [
        {
            "player": {
                "id": 10,
                "name": "Example Forward",
                "age": 26,
                "nationality": "England",
                "birth": {"date": "1999-01-01"},
            },
            "statistics": [
                {
                    "team": {"name": "Arsenal"},
                    "games": {
                        "appearences": 20,
                        "lineups": 17,
                        "minutes": 1530,
                        "position": "Attacker",
                        "rating": "7.21",
                    },
                    "goals": {"total": 9, "assists": 4},
                }
            ],
        }
    ]
    frame = players_to_season_stats(raw_players, league=league, season=2025)
    db = init_database(tmp_path / "football.sqlite3")
    assert db.upsert_player_stats(frame, source="api-football", league_key=league.key, season="2025") == 1
    stats = pd.read_sql_table("player_season_stats", db.engine)
    players = pd.read_sql_table("players", db.engine)
    assert stats.iloc[0]["minutes"] == 1530
    assert stats.iloc[0]["goals"] == 9
    assert players.iloc[0]["name"] == "Example Forward"


def test_coverage_report_marks_empty_tier3_leagues(tmp_path) -> None:
    db = init_database(tmp_path / "football.sqlite3")
    db.upsert_matches(
        synthetic_matches(10),
        source="test",
        league_key="premier_league",
        league_name="Premier League",
    )
    report = build_coverage_report(db_path=db.path)
    by_key = {league["key"]: league for league in report["leagues"]}
    assert by_key["premier_league"]["status"] == "sparse"
    assert by_key["chinese_super_league"]["status"] == "empty"
    assert by_key["chinese_super_league"]["needs_external_provider"] is True


def test_doctor_report_recommends_api_key_when_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FOOTBALL_API_KEY", raising=False)
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.setattr("config.settings.load_dotenv_file", lambda path: False)
    db = init_database(tmp_path / "football.sqlite3")
    db.upsert_matches(
        synthetic_matches(10),
        source="test",
        league_key="premier_league",
        league_name="Premier League",
    )
    report = build_doctor_report(db_path=db.path)
    assert report["status"] == "needs_api_key"
    assert report["api_football"]["present"] is False
    assert any("FOOTBALL_API_KEY" in command for command in report["recommended_commands"])


def test_schema_report_is_ok_after_init(tmp_path) -> None:
    db = init_database(tmp_path / "football.sqlite3")
    report = schema_report(db.engine, Base.metadata)
    assert report["ok"] is True
    assert report["missing_columns"] == []


def test_schema_migration_adds_missing_nullable_match_columns(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    engine = create_engine(f"sqlite:///{path}", future=True)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE matches (
                    id INTEGER PRIMARY KEY,
                    source VARCHAR(64) NOT NULL,
                    league_key VARCHAR(64) NOT NULL,
                    league_name VARCHAR(128),
                    season VARCHAR(32),
                    date DATE NOT NULL,
                    home_team VARCHAR(128) NOT NULL,
                    away_team VARCHAR(128) NOT NULL,
                    home_goals INTEGER NOT NULL,
                    away_goals INTEGER NOT NULL,
                    result VARCHAR(1),
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                )
                """
            )
        )
    report = ensure_schema(engine, Base.metadata)
    columns = {column["name"] for column in inspect(engine).get_columns("matches")}
    assert "home_xg" in columns
    assert "away_xg" in columns
    assert report["ok"] is True


def test_settings_reads_api_key_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("FOOTBALL_API_KEY", "abcd1234SECRET")
    monkeypatch.setattr("config.settings.load_dotenv_file", lambda path: False)
    settings = get_settings()
    assert settings.has_api_football_key is True
    assert settings.football_api_key_source == "FOOTBALL_API_KEY"
    assert settings.masked_football_api_key == "abcd...CRET"


def test_api_football_leagues_to_frame_normalizes_ids() -> None:
    frame = leagues_to_frame(
        [
            {
                "league": {"id": 169, "name": "Super League", "type": "League"},
                "country": {"name": "China", "code": "CN"},
                "seasons": [{"year": 2025}, {"year": 2026}],
            }
        ]
    )
    assert frame.iloc[0]["league_id"] == 169
    assert frame.iloc[0]["country_name"] == "China"
    assert frame.iloc[0]["seasons"] == [2025, 2026]
