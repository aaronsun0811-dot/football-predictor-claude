"""National team Elo from eloratings.net.

The site renders the world ranking server-side at https://www.eloratings.net/ .
We parse the embedded ``ratingsData`` JS array — it's been stable for years.
Robust scrape:

  1. GET the main page.
  2. Extract the ``ratingsData`` blob via a regex.
  3. Parse as JSON.

Field positions in the blob: [team_code, team_name, rating, rank, ...].
"""
from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd

USER_AGENT = "Mozilla/5.0 (compatible; football-predictor/0.1)"
TIMEOUT_S = 30.0
RATINGS_URL = "https://www.eloratings.net/"
RATINGS_RE = re.compile(r"ratingsData\s*=\s*(\[\[.*?\]\]);", re.DOTALL)
INTERNATIONAL_FOOTBALL_URL = "https://www.international-football.net/elo-ratings-table"


def fetch_world_ratings(*, cache_dir: Path, force_refresh: bool = False) -> pd.DataFrame:
    """Return today's national-team Elo table sorted by rating descending."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"eloratings_{date.today().isoformat()}.csv"
    if target.exists() and not force_refresh:
        return pd.read_csv(target)

    try:
        frame = _fetch_from_eloratings()
    except Exception as primary_error:
        frame = _fetch_from_international_football()
        frame["fallback_reason"] = str(primary_error)
    frame["fetched_at"] = datetime.now(UTC).isoformat()
    frame.to_csv(target, index=False)
    return frame


def find_team(frame: pd.DataFrame, name: str) -> dict | None:
    """Best-effort lookup. Tries exact name then case-insensitive contains."""
    exact = frame[frame["team"].str.casefold() == name.casefold()]
    if not exact.empty:
        return exact.iloc[0].to_dict()
    partial = frame[frame["team"].str.contains(name, case=False, na=False)]
    if not partial.empty:
        return partial.iloc[0].to_dict()
    return None


def _fetch_from_eloratings() -> pd.DataFrame:
    response = httpx.get(
        RATINGS_URL,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
        timeout=TIMEOUT_S,
    )
    response.raise_for_status()
    match = RATINGS_RE.search(response.text)
    if not match:
        raise RuntimeError(
            "eloratings.net layout changed: ratingsData array not found. "
            "Check the HTML structure and update RATINGS_RE."
        )
    payload = json.loads(match.group(1))
    rows = []
    for row in payload:
        if len(row) < 4:
            continue
        rows.append(
            {
                "team_code": row[0],
                "team": row[1],
                "elo": float(row[2]) if row[2] is not None else None,
                "rank": int(row[3]) if row[3] is not None else None,
                "source_url": RATINGS_URL,
            }
        )
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def _fetch_from_international_football(on: date | None = None) -> pd.DataFrame:
    on = on or date.today()
    url = (
        f"{INTERNATIONAL_FOOTBALL_URL}"
        f"?confed=&day={on.day:02d}&month={on.month:02d}&old-team=&year={on.year}"
    )
    tables = pd.read_html(url)
    rows = []
    for table in tables:
        if table.shape[1] < 4:
            continue
        candidate = table.iloc[:, [0, 2, 3]].copy()
        candidate.columns = ["rank", "team", "elo"]
        candidate["rank"] = pd.to_numeric(candidate["rank"], errors="coerce")
        candidate["elo"] = pd.to_numeric(candidate["elo"], errors="coerce")
        candidate = candidate.dropna(subset=["rank", "team", "elo"])
        for row in candidate.itertuples(index=False):
            rows.append(
                {
                    "team_code": None,
                    "team": str(row.team),
                    "elo": float(row.elo),
                    "rank": int(row.rank),
                    "source_url": url,
                }
            )
    if not rows:
        raise RuntimeError("international-football.net Elo table parse returned no rows.")
    return pd.DataFrame(rows).drop_duplicates("team").sort_values("rank").reset_index(drop=True)
