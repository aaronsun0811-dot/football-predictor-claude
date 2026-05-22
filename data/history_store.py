"""Append-only prediction-history store, sharded by month.

Layout::

    data/cache/upcoming/
        history.jsonl                 ← legacy single file (read-only, kept for compat)
        history/
            2026-05.jsonl             ← writes go here, one shard per month
            2026-04.jsonl
            ...

Why shards:

* The legacy single ``history.jsonl`` grows unbounded — at ~10 fixtures × 1
  snapshot/hour × 24 = 240 rows/day, after a year it's ~90k lines. Still
  small, but reads scan the whole file every time and there's no natural
  pruning unit (you can't delete "predictions from April 2026" without
  rewriting the file).
* Month shards keep the hot path (delta-vs-3h-ago lookups) reading only
  the current month, and let us archive / compress / drop old months
  cheaply.

Read order: legacy file first, then every shard sorted by filename. Callers
that only need recent rows can pass ``since=`` to skip old shards entirely.

Public API:
    * ``append_rows(rows, *, taken_at)`` — write to the current-month shard
    * ``iter_all_rows()`` — yield every row across legacy + all shards
    * ``iter_recent_rows(since)`` — yield rows from shards covering ``since``+
    * ``migrate_legacy_file()`` — one-shot split of the legacy file into shards
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


HISTORY_ROOT = Path(__file__).resolve().parents[1] / "data" / "cache" / "upcoming"
LEGACY_PATH = HISTORY_ROOT / "history.jsonl"
SHARD_DIR = HISTORY_ROOT / "history"
SHARD_PATTERN = re.compile(r"^(\d{4})-(\d{2})\.jsonl$")


def _month_key(when: datetime) -> str:
    return when.strftime("%Y-%m")


def _shard_path_for(when: datetime, *, shard_dir: Path | None = None) -> Path:
    shard_dir = shard_dir or SHARD_DIR
    return shard_dir / f"{_month_key(when)}.jsonl"


def _list_shards(shard_dir: Path | None = None) -> list[Path]:
    """Return every ``YYYY-MM.jsonl`` shard, sorted oldest→newest."""
    shard_dir = shard_dir or SHARD_DIR
    if not shard_dir.exists():
        return []
    found = [p for p in shard_dir.iterdir() if SHARD_PATTERN.match(p.name)]
    return sorted(found, key=lambda p: p.name)


def _shards_since(since: datetime, *, shard_dir: Path | None = None) -> list[Path]:
    """Return shards whose month is ≥ ``since``'s month."""
    cutoff = _month_key(since)
    return [p for p in _list_shards(shard_dir) if p.stem >= cutoff]


def append_rows(
    rows: Iterable[dict[str, Any]],
    *,
    taken_at: datetime | None = None,
    shard_dir: Path | None = None,
) -> int:
    """Append rows to the current-month shard. Returns rows written.

    Each row is serialized as one JSON object per line. The ``taken_at``
    timestamp picks the shard — defaults to "now (UTC)".
    """
    taken_at = taken_at or datetime.now(timezone.utc)
    shard_dir = shard_dir or SHARD_DIR
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = _shard_path_for(taken_at, shard_dir=shard_dir)

    written = 0
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def iter_all_rows(
    *,
    legacy_path: Path | None = None,
    shard_dir: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield every row from the legacy file + every shard, in chronological order.

    The shards are sorted oldest→newest. The legacy file is read first because
    it predates the shard layout and represents the oldest data. Malformed
    JSON lines are silently skipped so a corrupt row can't break readers.
    """
    legacy_path = legacy_path if legacy_path is not None else LEGACY_PATH
    shard_dir = shard_dir if shard_dir is not None else SHARD_DIR
    sources: list[Path] = []
    if legacy_path.exists():
        sources.append(legacy_path)
    sources.extend(_list_shards(shard_dir))

    for path in sources:
        try:
            with path.open() as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            # File got deleted between listdir and open — fine, skip.
            continue


def iter_recent_rows(
    *,
    since: datetime,
    legacy_path: Path | None = None,
    shard_dir: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield rows from shards whose month is ≥ ``since``'s month.

    Faster than ``iter_all_rows()`` for the hot path (delta-vs-Nh-ago lookups).
    Conservatively includes the legacy file too — its content might span any
    month and there's no cheap way to tell from outside.
    """
    legacy_path = legacy_path if legacy_path is not None else LEGACY_PATH
    shard_dir = shard_dir if shard_dir is not None else SHARD_DIR
    sources: list[Path] = []
    if legacy_path.exists():
        sources.append(legacy_path)
    sources.extend(_shards_since(since, shard_dir=shard_dir))

    for path in sources:
        try:
            with path.open() as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue


def migrate_legacy_file(
    *,
    legacy_path: Path | None = None,
    shard_dir: Path | None = None,
    archive_suffix: str = ".migrated",
) -> dict[str, Any]:
    """Split the legacy ``history.jsonl`` into per-month shards.

    Idempotent-ish: appends to existing shards rather than overwriting, so if
    you re-run after partial failure you get duplicate rows (acceptable — the
    audit module already de-dupes by earliest prediction per fixture).

    After splitting, the legacy file is renamed to ``history.jsonl.migrated``
    so a future run sees no legacy file and skips this step. Original content
    is preserved on disk in case the migration needs to be undone manually.

    Returns a report: ``{rows_read, rows_by_month, malformed, legacy_moved}``.
    """
    legacy_path = legacy_path if legacy_path is not None else LEGACY_PATH
    shard_dir = shard_dir if shard_dir is not None else SHARD_DIR
    if not legacy_path.exists():
        return {
            "rows_read": 0, "rows_by_month": {}, "malformed": 0,
            "legacy_moved": False, "skipped": "no_legacy_file",
        }
    shard_dir.mkdir(parents=True, exist_ok=True)

    rows_by_month: dict[str, list[str]] = {}
    rows_read = 0
    malformed = 0
    with legacy_path.open() as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            rows_read += 1
            try:
                row = json.loads(line_stripped)
                ts_str = row.get("taken_at")
                if not ts_str:
                    malformed += 1
                    continue
                ts = datetime.fromisoformat(ts_str)
                month = _month_key(ts)
                rows_by_month.setdefault(month, []).append(line_stripped + "\n")
            except (json.JSONDecodeError, ValueError, TypeError):
                malformed += 1
                continue

    for month, lines in rows_by_month.items():
        path = shard_dir / f"{month}.jsonl"
        with path.open("a") as f:
            f.writelines(lines)

    # Park the original out of the way so re-runs don't double-write.
    archived = legacy_path.with_suffix(legacy_path.suffix + archive_suffix)
    legacy_path.rename(archived)

    return {
        "rows_read": rows_read,
        "rows_by_month": {m: len(v) for m, v in rows_by_month.items()},
        "malformed": malformed,
        "legacy_moved": True,
        "archived_path": str(archived),
    }
