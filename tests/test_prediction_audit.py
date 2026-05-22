"""Tests for the prediction-audit module."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def history_with_resolution(tmp_path: Path, monkeypatch):
    """Seed a history.jsonl with 3 predictions + a matching matches frame."""
    hist_path = tmp_path / "history.jsonl"
    today = date.today()
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=7)
    tomorrow = today + timedelta(days=1)

    base_ts = datetime.now(timezone.utc).isoformat()
    rows = [
        # Resolved + correct: model picked home, actual home win
        {"taken_at": base_ts, "date": yesterday.isoformat(),
         "league_key": "test_league", "home_team": "A", "away_team": "B",
         "home_win": 0.7, "draw": 0.2, "away_win": 0.1},
        # Resolved + wrong: model picked away, actual home win
        {"taken_at": base_ts, "date": week_ago.isoformat(),
         "league_key": "test_league", "home_team": "C", "away_team": "D",
         "home_win": 0.2, "draw": 0.2, "away_win": 0.6},
        # Future fixture — should be IGNORED
        {"taken_at": base_ts, "date": tomorrow.isoformat(),
         "league_key": "test_league", "home_team": "E", "away_team": "F",
         "home_win": 0.5, "draw": 0.3, "away_win": 0.2},
    ]
    with hist_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Patch the module's HISTORY_PATH constant to point at our tmp file.
    import models.prediction_audit as audit_module
    monkeypatch.setattr(audit_module, "HISTORY_PATH", hist_path)

    # Matches frame with results for the two past fixtures.
    matches = pd.DataFrame([
        {"date": yesterday, "home_team": "A", "away_team": "B",
         "home_goals": 2, "away_goals": 1},  # A wins → model was right
        {"date": week_ago, "home_team": "C", "away_team": "D",
         "home_goals": 1, "away_goals": 0},  # C wins → model picked D, wrong
    ])
    return matches


def test_resolve_predictions_skips_future_fixtures(history_with_resolution):
    from models.prediction_audit import resolve_predictions
    resolved = resolve_predictions(history_with_resolution)
    # 2 past fixtures had results; the future one should be excluded.
    assert len(resolved) == 2
    teams = set(zip(resolved["home_team"], resolved["away_team"]))
    assert ("E", "F") not in teams


def test_resolve_predictions_marks_correctness(history_with_resolution):
    from models.prediction_audit import resolve_predictions
    resolved = resolve_predictions(history_with_resolution)
    by_home = {r["home_team"]: r for r in resolved.to_dict(orient="records")}
    # A vs B: model picked home, actual home → correct
    assert by_home["A"]["predicted_outcome"] == "home_win"
    assert by_home["A"]["correct"] is True
    # C vs D: model picked away, actual home → wrong
    assert by_home["C"]["predicted_outcome"] == "away_win"
    assert by_home["C"]["correct"] is False


def test_resolve_predictions_p_actual_is_assigned_correctly(history_with_resolution):
    from models.prediction_audit import resolve_predictions
    resolved = resolve_predictions(history_with_resolution)
    by_home = {r["home_team"]: r for r in resolved.to_dict(orient="records")}
    # A vs B: actual home → p_actual = home_win = 0.7
    assert by_home["A"]["p_actual"] == pytest.approx(0.7)
    # C vs D: actual home → p_actual = home_win = 0.2 (model gave away 0.6)
    assert by_home["C"]["p_actual"] == pytest.approx(0.2)


def test_summarize_resolved_computes_accuracy(history_with_resolution):
    from models.prediction_audit import resolve_predictions, summarize_resolved
    resolved = resolve_predictions(history_with_resolution)
    summary = summarize_resolved(resolved)
    # 1 of 2 correct
    assert summary["n_resolved"] == 2
    assert summary["accuracy"] == pytest.approx(0.5)
    # Brier is non-negative and bounded
    assert summary["brier"] > 0
    assert summary["rps"] > 0


def test_audit_summary_includes_best_and_worst(history_with_resolution):
    from models.prediction_audit import audit_summary
    summary = audit_summary(history_with_resolution)
    assert summary["n_resolved"] == 2
    # 1 correct → best_calls has 1 entry
    assert len(summary["best_calls"]) == 1
    # 2 resolved total → worst_misses has up to 2
    assert len(summary["worst_misses"]) == 2


def test_summary_by_league_includes_score_metrics(tmp_path, monkeypatch):
    """audit summary.by_league carries exact_score_accuracy + mean_goal_distance,
    matching backtest's summary.by_league shape (R29)."""
    from models.prediction_audit import resolve_predictions, summarize_resolved
    import models.prediction_audit as audit_module

    hist_path = tmp_path / "history.jsonl"
    yesterday = date.today() - timedelta(days=1)
    rows = []
    # PL: 3 resolved with predicted scores — 1 exact hit, 2 misses
    for i, (h, a, hg, ag, ph, pa) in enumerate([
        ("A", "B", 2, 1, 2, 1),  # exact hit
        ("C", "D", 1, 0, 2, 0),  # off by 1
        ("E", "F", 0, 0, 1, 1),  # off by 2
    ]):
        rows.append({
            "taken_at": datetime.now(timezone.utc).isoformat(),
            "date": yesterday.isoformat(), "league_key": "pl",
            "home_team": h, "away_team": a,
            "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
            "predicted_home_goals": ph, "predicted_away_goals": pa,
            "predicted_score_prob": 0.1,
        })
    # La Liga: 3 resolved without predicted scores (legacy rows)
    for h, a in [("X", "Y"), ("X2", "Y2"), ("X3", "Y3")]:
        rows.append({
            "taken_at": datetime.now(timezone.utc).isoformat(),
            "date": yesterday.isoformat(), "league_key": "la_liga",
            "home_team": h, "away_team": a,
            "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
        })
    with hist_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    monkeypatch.setattr(audit_module, "HISTORY_PATH", hist_path)

    matches = pd.DataFrame([
        {"date": yesterday, "home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 1},
        {"date": yesterday, "home_team": "C", "away_team": "D", "home_goals": 1, "away_goals": 0},
        {"date": yesterday, "home_team": "E", "away_team": "F", "home_goals": 0, "away_goals": 0},
        {"date": yesterday, "home_team": "X", "away_team": "Y", "home_goals": 1, "away_goals": 1},
        {"date": yesterday, "home_team": "X2", "away_team": "Y2", "home_goals": 1, "away_goals": 1},
        {"date": yesterday, "home_team": "X3", "away_team": "Y3", "home_goals": 1, "away_goals": 1},
    ])
    summary = summarize_resolved(resolve_predictions(matches))

    by_league = {r["league_key"]: r for r in summary["by_league"]}
    assert set(by_league) == {"pl", "la_liga"}

    pl = by_league["pl"]
    assert pl["n"] == 3
    assert pl["n_scored"] == 3
    assert pl["exact_score_accuracy"] == pytest.approx(1 / 3)
    # Goal distances: 0 (exact), 1 (2-0 vs 1-0), 2 (1-1 vs 0-0). Mean = 1.0
    assert pl["mean_goal_distance"] == pytest.approx(1.0)

    laliga = by_league["la_liga"]
    assert laliga["n"] == 3
    assert laliga["n_scored"] == 0
    assert laliga["exact_score_accuracy"] is None
    assert laliga["mean_goal_distance"] is None


def test_resolve_predictions_handles_no_history(tmp_path, monkeypatch):
    import models.prediction_audit as audit_module
    monkeypatch.setattr(audit_module, "HISTORY_PATH", tmp_path / "nonexistent.jsonl")
    resolved = audit_module.resolve_predictions(pd.DataFrame())
    assert len(resolved) == 0


def test_summarize_resolved_handles_empty_frame():
    from models.prediction_audit import summarize_resolved
    summary = summarize_resolved(pd.DataFrame(columns=["correct", "brier", "rps", "log_loss"]))
    assert summary["n_resolved"] == 0
    assert summary["accuracy"] is None


def test_resolve_attaches_score_metrics_when_history_has_them(tmp_path, monkeypatch):
    """A history row with predicted_home_goals/away_goals → audit can score it."""
    import models.prediction_audit as audit_module
    hist_path = tmp_path / "history.jsonl"
    yesterday = date.today() - timedelta(days=1)
    with hist_path.open("w") as f:
        # Three rows, all with predicted scores:
        # 1) exact hit (2-1 vs 2-1) — outcome correct + score correct
        f.write(json.dumps({
            "taken_at": datetime.now(timezone.utc).isoformat(),
            "date": yesterday.isoformat(), "league_key": "pl",
            "home_team": "A", "away_team": "B",
            "home_win": 0.55, "draw": 0.25, "away_win": 0.20,
            "predicted_home_goals": 2, "predicted_away_goals": 1,
            "predicted_score_prob": 0.10,
        }) + "\n")
        # 2) outcome correct, score off by 1 (predicted 1-0, actual 2-0)
        f.write(json.dumps({
            "taken_at": datetime.now(timezone.utc).isoformat(),
            "date": yesterday.isoformat(), "league_key": "pl",
            "home_team": "C", "away_team": "D",
            "home_win": 0.50, "draw": 0.30, "away_win": 0.20,
            "predicted_home_goals": 1, "predicted_away_goals": 0,
            "predicted_score_prob": 0.12,
        }) + "\n")
    monkeypatch.setattr(audit_module, "HISTORY_PATH", hist_path)

    matches = pd.DataFrame([
        {"date": yesterday, "home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 1},
        {"date": yesterday, "home_team": "C", "away_team": "D", "home_goals": 2, "away_goals": 0},
    ])
    resolved = audit_module.resolve_predictions(matches)
    assert len(resolved) == 2
    by_home = {r["home_team"]: r for r in resolved.to_dict(orient="records")}
    # Exact hit
    assert by_home["A"]["predicted_score"] == "2-1"
    assert bool(by_home["A"]["exact_score_correct"]) is True
    assert by_home["A"]["goal_distance"] == 0
    # One-goal miss
    assert by_home["C"]["predicted_score"] == "1-0"
    assert bool(by_home["C"]["exact_score_correct"]) is False
    assert by_home["C"]["goal_distance"] == 1


def test_resolve_handles_legacy_rows_without_predicted_scores(tmp_path, monkeypatch):
    """Old snapshots predate the schema change → score fields are None, but
    outcome metrics still work."""
    import models.prediction_audit as audit_module
    hist_path = tmp_path / "history.jsonl"
    yesterday = date.today() - timedelta(days=1)
    with hist_path.open("w") as f:
        f.write(json.dumps({
            "taken_at": datetime.now(timezone.utc).isoformat(),
            "date": yesterday.isoformat(), "league_key": "pl",
            "home_team": "A", "away_team": "B",
            "home_win": 0.7, "draw": 0.2, "away_win": 0.1,
            # No predicted_home_goals / predicted_away_goals
        }) + "\n")
    monkeypatch.setattr(audit_module, "HISTORY_PATH", hist_path)

    matches = pd.DataFrame([
        {"date": yesterday, "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0},
    ])
    resolved = audit_module.resolve_predictions(matches)
    assert len(resolved) == 1
    row = resolved.iloc[0]
    assert row["predicted_outcome"] == "home_win"
    assert bool(row["correct"]) is True
    # Score fields gracefully None
    assert row["predicted_score"] is None
    assert pd.isna(row["exact_score_correct"]) or row["exact_score_correct"] is None
    assert pd.isna(row["goal_distance"]) or row["goal_distance"] is None


def test_summarize_resolved_computes_score_metrics(tmp_path, monkeypatch):
    """exact_score_accuracy + mean_goal_distance over scored rows only."""
    from models.prediction_audit import resolve_predictions, summarize_resolved
    import models.prediction_audit as audit_module
    hist_path = tmp_path / "history.jsonl"
    yesterday = date.today() - timedelta(days=1)
    rows = [
        # Exact hit
        {"date": yesterday.isoformat(), "league_key": "pl", "home_team": "A", "away_team": "B",
         "home_win": 0.6, "draw": 0.25, "away_win": 0.15,
         "predicted_home_goals": 2, "predicted_away_goals": 1, "predicted_score_prob": 0.1},
        # Distance 2 (1-0 vs 3-1)
        {"date": yesterday.isoformat(), "league_key": "pl", "home_team": "C", "away_team": "D",
         "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
         "predicted_home_goals": 1, "predicted_away_goals": 0, "predicted_score_prob": 0.12},
    ]
    with hist_path.open("w") as f:
        for r in rows:
            r["taken_at"] = datetime.now(timezone.utc).isoformat()
            f.write(json.dumps(r) + "\n")
    monkeypatch.setattr(audit_module, "HISTORY_PATH", hist_path)

    matches = pd.DataFrame([
        {"date": yesterday, "home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 1},
        {"date": yesterday, "home_team": "C", "away_team": "D", "home_goals": 3, "away_goals": 1},
    ])
    resolved = resolve_predictions(matches)
    summary = summarize_resolved(resolved)
    assert summary["n_scored"] == 2
    assert summary["exact_score_accuracy"] == pytest.approx(0.5)  # 1 of 2
    assert summary["mean_goal_distance"] == pytest.approx(1.5)    # (0 + 3) / 2 — 1-0 vs 3-1 = |1-3|+|0-1|=3


def test_audit_summary_includes_exact_score_hits(tmp_path, monkeypatch):
    """When ANY exact-score hits occur, summary["exact_score_hits"] is populated."""
    from models.prediction_audit import audit_summary
    import models.prediction_audit as audit_module
    hist_path = tmp_path / "history.jsonl"
    yesterday = date.today() - timedelta(days=1)
    with hist_path.open("w") as f:
        f.write(json.dumps({
            "taken_at": datetime.now(timezone.utc).isoformat(),
            "date": yesterday.isoformat(), "league_key": "pl",
            "home_team": "A", "away_team": "B",
            "home_win": 0.6, "draw": 0.25, "away_win": 0.15,
            "predicted_home_goals": 2, "predicted_away_goals": 1, "predicted_score_prob": 0.10,
        }) + "\n")
    monkeypatch.setattr(audit_module, "HISTORY_PATH", hist_path)
    matches = pd.DataFrame([
        {"date": yesterday, "home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 1},
    ])
    summary = audit_summary(matches)
    assert len(summary["exact_score_hits"]) == 1
    hit = summary["exact_score_hits"][0]
    assert hit["predicted_score"] == "2-1"
    assert hit["actual_score"] == "2-1"


def test_resolved_row_carries_date_check_status(tmp_path, monkeypatch):
    """A history row with date_check_status='warning' shows up in the
    resolved frame as had_date_warning=True."""
    import models.prediction_audit as audit_module
    hist_path = tmp_path / "history.jsonl"
    yesterday = date.today() - timedelta(days=1)
    with hist_path.open("w") as f:
        # Row 1: had a date warning at /upcoming time
        f.write(json.dumps({
            "taken_at": datetime.now(timezone.utc).isoformat(),
            "date": yesterday.isoformat(), "league_key": "premier_league",
            "home_team": "A", "away_team": "B",
            "home_win": 0.7, "draw": 0.2, "away_win": 0.1,
            "date_check_status": "warning",
            "date_check_fdorg_date": yesterday.isoformat(),
            "date_check_days_off": 2,
        }) + "\n")
        # Row 2: no warning
        f.write(json.dumps({
            "taken_at": datetime.now(timezone.utc).isoformat(),
            "date": yesterday.isoformat(), "league_key": "premier_league",
            "home_team": "C", "away_team": "D",
            "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
            "date_check_status": "confirmed",
        }) + "\n")
    monkeypatch.setattr(audit_module, "HISTORY_PATH", hist_path)

    matches = pd.DataFrame([
        {"date": yesterday, "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0},
        {"date": yesterday, "home_team": "C", "away_team": "D", "home_goals": 0, "away_goals": 1},
    ])
    resolved = audit_module.resolve_predictions(matches)
    by_home = {r["home_team"]: r for r in resolved.to_dict(orient="records")}
    assert bool(by_home["A"]["had_date_warning"]) is True
    assert by_home["A"]["date_check_status"] == "warning"
    assert bool(by_home["C"]["had_date_warning"]) is False
    assert by_home["C"]["date_check_status"] == "confirmed"


def test_summarize_resolved_counts_predictions_with_date_warning(tmp_path, monkeypatch):
    from models.prediction_audit import resolve_predictions, summarize_resolved
    import models.prediction_audit as audit_module
    hist_path = tmp_path / "history.jsonl"
    yesterday = date.today() - timedelta(days=1)
    rows = [
        {"date": yesterday.isoformat(), "league_key": "pl",
         "home_team": "A", "away_team": "B",
         "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
         "date_check_status": "warning"},
        {"date": yesterday.isoformat(), "league_key": "pl",
         "home_team": "C", "away_team": "D",
         "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
         "date_check_status": "confirmed"},
        # No date_check_status at all (legacy row)
        {"date": yesterday.isoformat(), "league_key": "pl",
         "home_team": "E", "away_team": "F",
         "home_win": 0.5, "draw": 0.3, "away_win": 0.2},
    ]
    with hist_path.open("w") as f:
        for r in rows:
            r["taken_at"] = datetime.now(timezone.utc).isoformat()
            f.write(json.dumps(r) + "\n")
    monkeypatch.setattr(audit_module, "HISTORY_PATH", hist_path)

    matches = pd.DataFrame([
        {"date": yesterday, "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0},
        {"date": yesterday, "home_team": "C", "away_team": "D", "home_goals": 1, "away_goals": 0},
        {"date": yesterday, "home_team": "E", "away_team": "F", "home_goals": 1, "away_goals": 0},
    ])
    resolved = resolve_predictions(matches)
    summary = summarize_resolved(resolved)
    assert summary["n_resolved"] == 3
    # Only one had a warning explicitly. confirmed + missing both → no warning.
    assert summary["n_with_date_warning"] == 1


def test_legacy_rows_without_date_check_get_zero_count(tmp_path, monkeypatch):
    """All-legacy data should produce n_with_date_warning=0, not crash."""
    from models.prediction_audit import resolve_predictions, summarize_resolved
    import models.prediction_audit as audit_module
    hist_path = tmp_path / "history.jsonl"
    yesterday = date.today() - timedelta(days=1)
    with hist_path.open("w") as f:
        f.write(json.dumps({
            "taken_at": datetime.now(timezone.utc).isoformat(),
            "date": yesterday.isoformat(), "league_key": "pl",
            "home_team": "A", "away_team": "B",
            "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
        }) + "\n")
    monkeypatch.setattr(audit_module, "HISTORY_PATH", hist_path)
    matches = pd.DataFrame([
        {"date": yesterday, "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0},
    ])
    summary = summarize_resolved(resolve_predictions(matches))
    assert summary["n_with_date_warning"] == 0


def test_resolve_uses_canonicalized_names(tmp_path, monkeypatch):
    """If history has 'Real Madrid CF' and matches has 'Real Madrid', they should match."""
    import models.prediction_audit as audit_module
    hist_path = tmp_path / "history.jsonl"
    yesterday = date.today() - timedelta(days=1)
    with hist_path.open("w") as f:
        f.write(json.dumps({
            "taken_at": datetime.now(timezone.utc).isoformat(),
            "date": yesterday.isoformat(),
            "league_key": "ucl",
            "home_team": "Real Madrid CF",   # un-normalized
            "away_team": "FC Bayern München",
            "home_win": 0.5, "draw": 0.25, "away_win": 0.25,
        }) + "\n")
    monkeypatch.setattr(audit_module, "HISTORY_PATH", hist_path)

    matches = pd.DataFrame([{
        "date": yesterday,
        "home_team": "Real Madrid",   # canonical
        "away_team": "Bayern Munich",
        "home_goals": 2,
        "away_goals": 1,
    }])
    resolved = audit_module.resolve_predictions(matches)
    assert len(resolved) == 1
    assert bool(resolved.iloc[0]["correct"]) is True
