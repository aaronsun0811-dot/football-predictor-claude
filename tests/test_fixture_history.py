"""Tests for the /upcoming prediction-history adapter (`models/fixture_history`).

The adapter sits on top of ``data/history_store`` (which has its own tests).
What we cover here is the *prediction-flow-specific* behavior:

* ``append_snapshot`` builds the right row shape and skips fixtures missing
  the ``prediction.probabilities`` block
* ``lookup_deltas`` finds the snapshot closest to N hours ago, within the
  tolerance window, and only for matching ``(date, home, away)`` triples
* ``attach_deltas`` mutates fixtures in place, computing percentage-point
  deltas and gracefully degrading when no historical snapshot is found
* ``history_size`` reports total bytes across legacy + all shards
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from data import history_store
from models import fixture_history


@pytest.fixture
def isolated_history(tmp_path: Path, monkeypatch):
    """Redirect both the store and the adapter at a temp dir.

    ``fixture_history`` re-binds ``LEGACY_PATH`` / ``SHARD_DIR`` at import time,
    so patching only ``history_store`` isn't enough — ``history_size`` reads the
    adapter-module bindings. We patch both.
    """
    legacy = tmp_path / "history.jsonl"
    shard_dir = tmp_path / "history"
    monkeypatch.setattr(history_store, "LEGACY_PATH", legacy)
    monkeypatch.setattr(history_store, "SHARD_DIR", shard_dir)
    monkeypatch.setattr(fixture_history, "HISTORY_PATH", legacy)
    monkeypatch.setattr(fixture_history, "LEGACY_PATH", legacy)
    monkeypatch.setattr(fixture_history, "SHARD_DIR", shard_dir)
    return {"legacy": legacy, "shard_dir": shard_dir}


class _FakeDateTime:
    """Stand-in for ``datetime`` that returns a fixed ``now()`` value.

    Subclassing the real ``datetime`` doesn't work (it's a C type and class
    attributes can't be set), so we just expose the two staticmethods the
    module actually calls: ``now`` and ``fromisoformat``.
    """

    def __init__(self, fixed: datetime) -> None:
        self.fixed = fixed

    def now(self, tz=None):
        return self.fixed

    @staticmethod
    def fromisoformat(value: str):
        return datetime.fromisoformat(value)


def _read_shard_lines(shard_dir: Path) -> list[dict]:
    out: list[dict] = []
    for path in sorted(shard_dir.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# append_snapshot
# ---------------------------------------------------------------------------

def test_append_snapshot_writes_one_row_per_fixture(isolated_history) -> None:
    fixtures = [
        {
            "date": "2026-05-20", "league_key": "pl",
            "home_team": "Arsenal", "away_team": "Chelsea",
            "prediction": {"probabilities": {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}},
        },
        {
            "date": "2026-05-20", "league_key": "pl",
            "home_team": "Liverpool", "away_team": "Spurs",
            "prediction": {"probabilities": {"home_win": 0.6, "draw": 0.25, "away_win": 0.15}},
        },
    ]
    when = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    written = fixture_history.append_snapshot(fixtures, taken_at=when)
    assert written == 2

    rows = _read_shard_lines(isolated_history["shard_dir"])
    assert len(rows) == 2
    by_home = {r["home_team"]: r for r in rows}
    assert by_home["Arsenal"]["home_win"] == 0.5
    assert by_home["Arsenal"]["taken_at"] == when.isoformat()
    assert by_home["Arsenal"]["league_key"] == "pl"


def test_append_snapshot_skips_fixtures_without_probabilities(isolated_history) -> None:
    fixtures = [
        {"date": "2026-05-20", "home_team": "A", "away_team": "B"},
        {"date": "2026-05-20", "home_team": "C", "away_team": "D",
         "prediction": {}},
        {"date": "2026-05-20", "home_team": "E", "away_team": "F",
         "prediction": {"probabilities": None}},
        {"date": "2026-05-20", "home_team": "G", "away_team": "H",
         "prediction": {"probabilities": {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}}},
    ]
    written = fixture_history.append_snapshot(fixtures)
    assert written == 1
    rows = _read_shard_lines(isolated_history["shard_dir"])
    assert len(rows) == 1
    assert rows[0]["home_team"] == "G"


def test_append_snapshot_persists_date_check_warning(isolated_history) -> None:
    """A fixture with a date_check warning gets the status/days_off persisted
    so the audit module can later say 'this prediction shouldn't count toward
    accuracy — the data source itself flagged the date as suspect'."""
    fixtures = [{
        "date": "2026-05-18", "league_key": "premier_league",
        "home_team": "Arsenal", "away_team": "Burnley",
        "prediction": {"probabilities": {"home_win": 0.6, "draw": 0.25, "away_win": 0.15}},
        "date_check": {
            "status": "warning",
            "fdorg_date": "2026-05-24",
            "days_off": 6,
            "flipped": False,
        },
    }]
    fixture_history.append_snapshot(fixtures, taken_at=datetime(2026, 5, 18, 10, tzinfo=timezone.utc))
    rows = _read_shard_lines(isolated_history["shard_dir"])
    assert len(rows) == 1
    r = rows[0]
    assert r["date_check_status"] == "warning"
    assert r["date_check_fdorg_date"] == "2026-05-24"
    assert r["date_check_days_off"] == 6


def test_append_snapshot_persists_confirmed_status_without_extra_fields(isolated_history) -> None:
    """Confirmed/unknown/not_covered should store only the status, no extras."""
    fixtures = [{
        "date": "2026-05-18", "league_key": "premier_league",
        "home_team": "A", "away_team": "B",
        "prediction": {"probabilities": {"home_win": 0.4, "draw": 0.3, "away_win": 0.3}},
        "date_check": {"status": "confirmed", "fdorg_date": "2026-05-18"},
    }]
    fixture_history.append_snapshot(fixtures)
    r = _read_shard_lines(isolated_history["shard_dir"])[0]
    assert r["date_check_status"] == "confirmed"
    assert "date_check_fdorg_date" not in r  # only on warning
    assert "date_check_days_off" not in r


def test_append_snapshot_no_date_check_field_is_fine(isolated_history) -> None:
    """Old fixture payloads pre-round-25 had no date_check at all — must still work."""
    fixtures = [{
        "date": "2026-05-18", "league_key": "premier_league",
        "home_team": "A", "away_team": "B",
        "prediction": {"probabilities": {"home_win": 0.4, "draw": 0.3, "away_win": 0.3}},
        # no date_check key
    }]
    fixture_history.append_snapshot(fixtures)
    r = _read_shard_lines(isolated_history["shard_dir"])[0]
    assert "date_check_status" not in r


def test_append_snapshot_lands_in_correct_month_shard(isolated_history) -> None:
    """Two snapshots in different months → two different shard files."""
    fx = [{
        "date": "2026-05-20", "home_team": "A", "away_team": "B",
        "prediction": {"probabilities": {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}},
    }]
    fixture_history.append_snapshot(fx, taken_at=datetime(2026, 4, 30, 23, 0, tzinfo=timezone.utc))
    fixture_history.append_snapshot(fx, taken_at=datetime(2026, 5, 1, 0, 30, tzinfo=timezone.utc))
    assert (isolated_history["shard_dir"] / "2026-04.jsonl").exists()
    assert (isolated_history["shard_dir"] / "2026-05.jsonl").exists()


# ---------------------------------------------------------------------------
# lookup_deltas — uses datetime.now() internally, so we patch it
# ---------------------------------------------------------------------------

def _seed_snapshot(
    shard_dir: Path,
    *,
    taken_at: datetime,
    date_str: str,
    home: str,
    away: str,
    home_win: float = 0.5,
    draw: float = 0.3,
    away_win: float = 0.2,
) -> None:
    """Write a single snapshot row directly to the correct shard."""
    month = taken_at.strftime("%Y-%m")
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"{month}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps({
            "taken_at": taken_at.isoformat(),
            "date": date_str,
            "league_key": "pl",
            "home_team": home,
            "away_team": away,
            "home_win": home_win, "draw": draw, "away_win": away_win,
        }) + "\n")


def test_lookup_deltas_finds_snapshot_inside_tolerance(isolated_history, monkeypatch) -> None:
    """A snapshot ~3h old should be picked up by a 3h-ago lookup."""
    now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(fixture_history, "datetime", _FakeDateTime(now))

    _seed_snapshot(
        isolated_history["shard_dir"],
        taken_at=now - timedelta(hours=3, minutes=10),
        date_str="2026-05-20",
        home="Arsenal", away="Chelsea",
        home_win=0.45, draw=0.30, away_win=0.25,
    )
    fixtures = [{
        "date": "2026-05-20", "home_team": "Arsenal", "away_team": "Chelsea",
    }]
    out = fixture_history.lookup_deltas(fixtures, hours_ago=3.0, tolerance_hours=1.5)
    key = ("2026-05-20", "Arsenal", "Chelsea")
    assert key in out
    assert out[key]["past_probabilities"]["home_win"] == 0.45
    assert out[key]["age_hours"] == pytest.approx(3.17, abs=0.05)


def test_lookup_deltas_picks_closest_snapshot(isolated_history, monkeypatch) -> None:
    """Multiple snapshots in the window → the one closest to ``hours_ago`` wins."""
    now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(fixture_history, "datetime", _FakeDateTime(now))

    # 4h ago (further from 3h target) — should be ignored
    _seed_snapshot(
        isolated_history["shard_dir"],
        taken_at=now - timedelta(hours=4),
        date_str="2026-05-20", home="A", away="B",
        home_win=0.1, draw=0.1, away_win=0.8,
    )
    # 3h05m ago (closer to 3h target) — should win
    _seed_snapshot(
        isolated_history["shard_dir"],
        taken_at=now - timedelta(hours=3, minutes=5),
        date_str="2026-05-20", home="A", away="B",
        home_win=0.5, draw=0.3, away_win=0.2,
    )
    out = fixture_history.lookup_deltas(
        [{"date": "2026-05-20", "home_team": "A", "away_team": "B"}],
        hours_ago=3.0, tolerance_hours=1.5,
    )
    assert out[("2026-05-20", "A", "B")]["past_probabilities"]["home_win"] == 0.5


def test_lookup_deltas_ignores_snapshots_outside_tolerance(isolated_history, monkeypatch) -> None:
    """6h-old snapshot vs 3h-ago lookup with tol=1.5 → out of [1.5h, 4.5h] window."""
    now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(fixture_history, "datetime", _FakeDateTime(now))

    _seed_snapshot(
        isolated_history["shard_dir"],
        taken_at=now - timedelta(hours=6),
        date_str="2026-05-20", home="A", away="B",
    )
    out = fixture_history.lookup_deltas(
        [{"date": "2026-05-20", "home_team": "A", "away_team": "B"}],
        hours_ago=3.0, tolerance_hours=1.5,
    )
    assert out == {}


def test_lookup_deltas_ignores_unmatched_fixtures(isolated_history, monkeypatch) -> None:
    now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(fixture_history, "datetime", _FakeDateTime(now))

    _seed_snapshot(
        isolated_history["shard_dir"],
        taken_at=now - timedelta(hours=3),
        date_str="2026-05-20", home="A", away="B",
    )
    # Different fixture — no historical match
    out = fixture_history.lookup_deltas(
        [{"date": "2026-05-21", "home_team": "X", "away_team": "Y"}],
    )
    assert out == {}


def test_lookup_deltas_empty_fixtures_returns_empty(isolated_history) -> None:
    assert fixture_history.lookup_deltas([]) == {}


# ---------------------------------------------------------------------------
# attach_deltas
# ---------------------------------------------------------------------------

def test_attach_deltas_computes_percentage_point_diffs(isolated_history, monkeypatch) -> None:
    now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(fixture_history, "datetime", _FakeDateTime(now))
    _seed_snapshot(
        isolated_history["shard_dir"],
        taken_at=now - timedelta(hours=3),
        date_str="2026-05-20", home="A", away="B",
        home_win=0.40, draw=0.30, away_win=0.30,
    )
    fixtures = [{
        "date": "2026-05-20", "home_team": "A", "away_team": "B",
        "prediction": {"probabilities": {"home_win": 0.50, "draw": 0.30, "away_win": 0.20}},
    }]
    fixture_history.attach_deltas(fixtures, hours_ago=3.0, tolerance_hours=1.5)
    delta = fixtures[0]["prediction"]["delta_vs_3h"]
    assert delta is not None
    # home_win went from 0.40 → 0.50, so +0.10
    assert delta["delta"]["home_win"] == pytest.approx(0.10, abs=1e-6)
    assert delta["delta"]["draw"] == pytest.approx(0.0, abs=1e-6)
    assert delta["delta"]["away_win"] == pytest.approx(-0.10, abs=1e-6)
    assert delta["past_probabilities"]["home_win"] == 0.40


def test_attach_deltas_sets_none_when_no_history(isolated_history, monkeypatch) -> None:
    now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(fixture_history, "datetime", _FakeDateTime(now))
    fixtures = [{
        "date": "2026-05-20", "home_team": "A", "away_team": "B",
        "prediction": {"probabilities": {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}},
    }]
    fixture_history.attach_deltas(fixtures)
    assert fixtures[0]["prediction"]["delta_vs_3h"] is None


def test_attach_deltas_handles_fixture_without_prediction(isolated_history, monkeypatch) -> None:
    """A fixture without a prediction shouldn't crash; ``delta_vs_3h`` is just None."""
    now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(fixture_history, "datetime", _FakeDateTime(now))
    fixtures = [{"date": "2026-05-20", "home_team": "A", "away_team": "B"}]
    # Should not raise.
    fixture_history.attach_deltas(fixtures)
    pred = fixtures[0].get("prediction") or {}
    assert pred.get("delta_vs_3h") is None


# ---------------------------------------------------------------------------
# history_size
# ---------------------------------------------------------------------------

def test_history_size_sums_legacy_and_shards(isolated_history) -> None:
    # Empty initially
    assert fixture_history.history_size() == 0

    # Seed legacy
    isolated_history["legacy"].write_text("x" * 100)
    assert fixture_history.history_size() == 100

    # Seed a shard
    isolated_history["shard_dir"].mkdir(parents=True, exist_ok=True)
    (isolated_history["shard_dir"] / "2026-05.jsonl").write_text("y" * 50)
    assert fixture_history.history_size() == 150


