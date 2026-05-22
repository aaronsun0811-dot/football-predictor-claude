"""ClubElo CSV API client.

ClubElo (http://clubelo.com/) publishes daily Elo ratings for ~600 European
clubs as plain CSV. Two endpoints we care about:

  http://api.clubelo.com/<YYYY-MM-DD>      -> ratings snapshot for all clubs
  http://api.clubelo.com/<ClubName>        -> full history for a single club

The API is documented at http://clubelo.com/API. Be polite: cache snapshots
on disk and never refetch the same day twice.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import httpx
import pandas as pd

CLUBELO_BASE = "http://api.clubelo.com"
USER_AGENT = "football-predictor/0.1 (research; +contact via repo)"
TIMEOUT_S = 30.0


def _cache_path(cache_dir: Path, on: date) -> Path:
    return cache_dir / f"clubelo_{on.isoformat()}.csv"


def fetch_snapshot(on: date | None = None, *, cache_dir: Path) -> pd.DataFrame:
    """Return the ClubElo ratings snapshot for ``on`` (default: today).

    Cached on disk; re-reads cache if present.
    """
    on = on or date.today()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = _cache_path(cache_dir, on)
    if target.exists():
        return _read_snapshot_csv(target)

    url = f"{CLUBELO_BASE}/{on.isoformat()}"
    response = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
    response.raise_for_status()
    target.write_bytes(response.content)
    return _read_snapshot_csv(target)


def fetch_club_history(club: str, *, cache_dir: Path) -> pd.DataFrame:
    """Return Elo history for ``club`` (ClubElo's slug, e.g. ``ManCity``)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"clubelo_history_{club}.csv"

    url = f"{CLUBELO_BASE}/{club}"
    response = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
    response.raise_for_status()
    target.write_bytes(response.content)
    history = pd.read_csv(target)
    if "From" in history.columns:
        history["From"] = pd.to_datetime(history["From"])
    if "To" in history.columns:
        history["To"] = pd.to_datetime(history["To"])
    return history


def _read_snapshot_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    # Columns: Rank,Club,Country,Level,Elo,From,To
    if "From" in frame.columns:
        frame["From"] = pd.to_datetime(frame["From"])
    if "To" in frame.columns:
        frame["To"] = pd.to_datetime(frame["To"])
    frame["fetched_at"] = datetime.utcnow()
    return frame


def filter_to_countries(snapshot: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    """Restrict a ClubElo snapshot to teams whose ``Country`` is in ``countries``.

    ClubElo uses ISO-style country codes (ENG, ESP, ITA, GER, FRA, POR, NED,
    BEL, ...). These match what we store in ``config/leagues.yaml``.
    """
    return snapshot[snapshot["Country"].isin(countries)].reset_index(drop=True)
