"""Tests for the month-sharded prediction-history store."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from data import history_store


@pytest.fixture
def tmp_store(tmp_path: Path, monkeypatch):
    """Redirect the store to a temp dir for each test."""
    legacy = tmp_path / "history.jsonl"
    shard_dir = tmp_path / "history"
    monkeypatch.setattr(history_store, "LEGACY_PATH", legacy)
    monkeypatch.setattr(history_store, "SHARD_DIR", shard_dir)
    monkeypatch.setattr(history_store, "HISTORY_ROOT", tmp_path)
    return {"root": tmp_path, "legacy": legacy, "shard_dir": shard_dir}


# ---------------------------------------------------------------------------
# append_rows — writes go to the current-month shard
# ---------------------------------------------------------------------------

def test_append_rows_creates_shard_dir_and_file(tmp_store) -> None:
    when = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
    written = history_store.append_rows(
        [{"taken_at": when.isoformat(), "x": 1}],
        taken_at=when,
        shard_dir=tmp_store["shard_dir"],
    )
    assert written == 1
    assert (tmp_store["shard_dir"] / "2026-05.jsonl").exists()


def test_append_rows_routes_by_month(tmp_store) -> None:
    may = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    june = datetime(2026, 6, 1, 1, 0, tzinfo=timezone.utc)
    history_store.append_rows([{"x": "may1"}], taken_at=may, shard_dir=tmp_store["shard_dir"])
    history_store.append_rows([{"x": "may2"}], taken_at=may, shard_dir=tmp_store["shard_dir"])
    history_store.append_rows([{"x": "jun1"}], taken_at=june, shard_dir=tmp_store["shard_dir"])

    may_shard = tmp_store["shard_dir"] / "2026-05.jsonl"
    jun_shard = tmp_store["shard_dir"] / "2026-06.jsonl"
    assert len(may_shard.read_text().splitlines()) == 2
    assert len(jun_shard.read_text().splitlines()) == 1


def test_append_rows_appends_within_a_month(tmp_store) -> None:
    when = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
    history_store.append_rows([{"a": 1}], taken_at=when, shard_dir=tmp_store["shard_dir"])
    history_store.append_rows([{"b": 2}, {"c": 3}], taken_at=when, shard_dir=tmp_store["shard_dir"])
    shard = tmp_store["shard_dir"] / "2026-05.jsonl"
    assert len(shard.read_text().splitlines()) == 3


# ---------------------------------------------------------------------------
# iter_all_rows — every shard + legacy file
# ---------------------------------------------------------------------------

def test_iter_all_rows_walks_legacy_and_shards(tmp_store) -> None:
    tmp_store["legacy"].write_text(json.dumps({"src": "legacy"}) + "\n")
    apr = datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)
    may = datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc)
    history_store.append_rows([{"src": "apr"}], taken_at=apr, shard_dir=tmp_store["shard_dir"])
    history_store.append_rows([{"src": "may"}], taken_at=may, shard_dir=tmp_store["shard_dir"])

    rows = list(history_store.iter_all_rows(
        legacy_path=tmp_store["legacy"],
        shard_dir=tmp_store["shard_dir"],
    ))
    sources = [r["src"] for r in rows]
    # Legacy first, then chronological shards (apr → may)
    assert sources == ["legacy", "apr", "may"]


def test_iter_all_rows_skips_malformed_lines(tmp_store) -> None:
    tmp_store["legacy"].write_text(
        json.dumps({"ok": 1}) + "\n"
        + "not-json-at-all\n"
        + json.dumps({"ok": 2}) + "\n"
    )
    rows = list(history_store.iter_all_rows(
        legacy_path=tmp_store["legacy"],
        shard_dir=tmp_store["shard_dir"],
    ))
    assert [r["ok"] for r in rows] == [1, 2]


def test_iter_all_rows_handles_no_files(tmp_store) -> None:
    rows = list(history_store.iter_all_rows(
        legacy_path=tmp_store["legacy"],
        shard_dir=tmp_store["shard_dir"],
    ))
    assert rows == []


# ---------------------------------------------------------------------------
# iter_recent_rows — month filter for the hot path
# ---------------------------------------------------------------------------

def test_iter_recent_rows_skips_older_shards(tmp_store) -> None:
    feb = datetime(2026, 2, 5, 12, 0, tzinfo=timezone.utc)
    apr = datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)
    may = datetime(2026, 5, 5, 12, 0, tzinfo=timezone.utc)
    history_store.append_rows([{"src": "feb"}], taken_at=feb, shard_dir=tmp_store["shard_dir"])
    history_store.append_rows([{"src": "apr"}], taken_at=apr, shard_dir=tmp_store["shard_dir"])
    history_store.append_rows([{"src": "may"}], taken_at=may, shard_dir=tmp_store["shard_dir"])

    # Looking back from May, asking for "since April": Feb is skipped, others included.
    rows = list(history_store.iter_recent_rows(
        since=apr,
        legacy_path=tmp_store["legacy"],
        shard_dir=tmp_store["shard_dir"],
    ))
    assert [r["src"] for r in rows] == ["apr", "may"]


def test_iter_recent_rows_includes_same_month(tmp_store) -> None:
    """Boundary: ``since`` mid-May still picks up that month's shard."""
    may1 = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    may20 = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    history_store.append_rows([{"src": "early_may"}], taken_at=may1, shard_dir=tmp_store["shard_dir"])
    rows = list(history_store.iter_recent_rows(
        since=may20,
        legacy_path=tmp_store["legacy"],
        shard_dir=tmp_store["shard_dir"],
    ))
    assert [r["src"] for r in rows] == ["early_may"]


def test_iter_recent_rows_includes_legacy(tmp_store) -> None:
    """Legacy file is read regardless of `since` — we can't know its month range."""
    tmp_store["legacy"].write_text(json.dumps({"src": "legacy"}) + "\n")
    rows = list(history_store.iter_recent_rows(
        since=datetime(2026, 12, 1, tzinfo=timezone.utc),
        legacy_path=tmp_store["legacy"],
        shard_dir=tmp_store["shard_dir"],
    ))
    assert [r["src"] for r in rows] == ["legacy"]


# ---------------------------------------------------------------------------
# migrate_legacy_file — one-shot split
# ---------------------------------------------------------------------------

def test_migrate_splits_legacy_into_monthly_shards(tmp_store) -> None:
    tmp_store["legacy"].write_text(
        json.dumps({"taken_at": "2026-04-15T10:00:00+00:00", "n": 1}) + "\n"
        + json.dumps({"taken_at": "2026-04-22T10:00:00+00:00", "n": 2}) + "\n"
        + json.dumps({"taken_at": "2026-05-01T00:30:00+00:00", "n": 3}) + "\n"
    )
    report = history_store.migrate_legacy_file(
        legacy_path=tmp_store["legacy"],
        shard_dir=tmp_store["shard_dir"],
    )
    assert report["rows_read"] == 3
    assert report["rows_by_month"] == {"2026-04": 2, "2026-05": 1}
    assert report["malformed"] == 0
    assert report["legacy_moved"] is True
    # Legacy is gone (renamed), shards exist.
    assert not tmp_store["legacy"].exists()
    assert (tmp_store["shard_dir"] / "2026-04.jsonl").exists()
    assert (tmp_store["shard_dir"] / "2026-05.jsonl").exists()


def test_migrate_handles_no_legacy_file(tmp_store) -> None:
    report = history_store.migrate_legacy_file(
        legacy_path=tmp_store["legacy"],
        shard_dir=tmp_store["shard_dir"],
    )
    assert report["legacy_moved"] is False
    assert report.get("skipped") == "no_legacy_file"


def test_migrate_counts_malformed_lines(tmp_store) -> None:
    tmp_store["legacy"].write_text(
        json.dumps({"taken_at": "2026-04-15T10:00:00+00:00", "n": 1}) + "\n"
        + "garbage\n"
        + json.dumps({"no_taken_at": True}) + "\n"
    )
    report = history_store.migrate_legacy_file(
        legacy_path=tmp_store["legacy"],
        shard_dir=tmp_store["shard_dir"],
    )
    assert report["rows_read"] == 3
    assert report["rows_by_month"] == {"2026-04": 1}
    assert report["malformed"] == 2
