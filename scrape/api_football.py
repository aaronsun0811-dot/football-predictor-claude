from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import httpx
import pandas as pd

from config.settings import get_settings, load_dotenv_file
from scrape.rate_limit import get_api_football_bucket
from scrape.registry import League


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
FINISHED_STATUSES = {"FT", "AET", "PEN"}
USER_AGENT = "football-predictor/0.1"

# The provider returns 200 OK with a body like {"errors": {"rateLimit": "..."}}
# when we trip a soft cap. These substrings identify those.
_RATE_LIMIT_ERROR_KEYS = ("ratelimit", "requests")

# Process-local flag: once we've hit the daily quota (vs per-minute rate),
# every subsequent request is doomed and would just waste 60s on retry. This
# flag short-circuits future ``get()`` calls until the process restarts (the
# scheduler will respawn for the next day's cron firing).
_DAILY_QUOTA_EXHAUSTED = False


def is_daily_quota_exhausted() -> bool:
    return _DAILY_QUOTA_EXHAUSTED


def reset_daily_quota_flag() -> None:
    """For tests, and for the rare case a user manually wants to retry."""
    global _DAILY_QUOTA_EXHAUSTED
    _DAILY_QUOTA_EXHAUSTED = False


@dataclass(frozen=True)
class ApiFootballConfig:
    api_key: str
    base_url: str = API_FOOTBALL_BASE_URL
    timeout_s: float = 45.0
    min_interval_s: float = 0.25


class ApiFootballClient:
    """Minimal API-Football client.

    Set FOOTBALL_API_KEY or API_FOOTBALL_KEY in the environment. The direct
    API-Sports endpoint uses the ``x-apisports-key`` header.
    """

    def __init__(self, config: ApiFootballConfig | None = None) -> None:
        settings = get_settings()
        self.config = config or ApiFootballConfig(
            api_key=_read_api_key(),
            base_url=settings.api_football_base_url,
            timeout_s=settings.api_football_timeout_s,
            min_interval_s=settings.api_football_min_interval_s,
        )
        self._last_request_at = 0.0
        # Cache the 429-backoff seconds on the instance so tests can override.
        self._backoff_s = float(settings.api_football_429_backoff_s)

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Authenticated GET with token-bucket rate limiting + 429 retry + daily-quota circuit breaker.

        Failure modes handled:
          * HTTP 429 — hard rate-limit response → drain bucket + backoff + retry once
          * HTTP 200 with ``{"errors": {"rateLimit": "..."}}`` — soft cap → same path
          * Daily-quota exhaustion (e.g. "You have reached the request limit for the day"
            on the free plan's 100/day cap) → set process-local circuit breaker so the
            rest of the batch fails fast instead of burning 60s per doomed retry

        After the circuit breaker fires, every subsequent call raises immediately
        until the process restarts (the scheduler respawns for the next cron firing,
        which is the next day's window).
        """
        global _DAILY_QUOTA_EXHAUSTED
        if _DAILY_QUOTA_EXHAUSTED:
            raise RuntimeError(
                "API-Football daily quota exhausted — try again after 00:00 UTC."
            )

        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        bucket = get_api_football_bucket()
        last_error_detail = ""
        for attempt in range(2):
            self._throttle()       # legacy floor — typically a no-op now
            bucket.acquire()
            response = httpx.get(
                url,
                params=params or {},
                headers={
                    "x-apisports-key": self.config.api_key,
                    "User-Agent": USER_AGENT,
                },
                timeout=self.config.timeout_s,
            )
            if response.status_code == 429:
                last_error_detail = f"HTTP 429 (attempt {attempt+1})"
                # Best-effort: check the body for daily-quota wording before
                # deciding whether retrying is even worth it.
                try:
                    body = response.json()
                except Exception:  # noqa: BLE001 — non-json 429 is fine, still retry
                    body = {}
                if _is_daily_quota_error(body.get("errors")):
                    _DAILY_QUOTA_EXHAUSTED = True
                    raise RuntimeError(
                        f"API-Football daily quota exhausted ({body.get('errors')})"
                    )
                if attempt == 0:
                    bucket.drain()
                    time.sleep(self._backoff_s)
                    continue
                response.raise_for_status()

            response.raise_for_status()
            payload = response.json()
            errors = payload.get("errors")
            if errors and _is_rate_limit_error(errors):
                # Check daily-quota BEFORE retrying — retry is pointless.
                if _is_daily_quota_error(errors):
                    _DAILY_QUOTA_EXHAUSTED = True
                    raise RuntimeError(
                        f"API-Football daily quota exhausted ({errors})"
                    )
                last_error_detail = f"soft rate-limit: {errors}"
                if attempt == 0:
                    bucket.drain()
                    time.sleep(self._backoff_s)
                    continue
                raise RuntimeError(
                    f"API-Football still rate-limited after retry ({errors})"
                )
            if errors:
                # Non-rate-limit errors are caller bugs (bad params, etc.) —
                # don't retry, surface immediately.
                raise RuntimeError(f"API-Football returned errors: {errors}")
            return payload

        raise RuntimeError(f"API-Football request failed after retries: {last_error_detail}")

    def fetch_fixtures(
        self,
        *,
        league_id: int,
        season: int,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"league": league_id, "season": season}
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()
        payload = self.get("/fixtures", params=params)
        return list(payload.get("response") or [])

    def fetch_leagues(
        self,
        *,
        country: str | None = None,
        search: str | None = None,
        league_id: int | None = None,
        season: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if country:
            params["country"] = country
        if search:
            params["search"] = search
        if league_id is not None:
            params["id"] = league_id
        if season is not None:
            params["season"] = season
        payload = self.get("/leagues", params=params)
        return list(payload.get("response") or [])

    def fetch_squad(self, *, team_id: int) -> list[dict[str, Any]]:
        payload = self.get("/players/squads", params={"team": team_id})
        return list(payload.get("response") or [])

    def fetch_players(
        self,
        *,
        league_id: int,
        season: int,
        page: int = 1,
    ) -> dict[str, Any]:
        return self.get("/players", params={"league": league_id, "season": season, "page": page})

    def fetch_all_players(
        self,
        *,
        league_id: int,
        season: int,
        max_pages: int = 50,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        page = 1
        while page <= max_pages:
            payload = self.fetch_players(league_id=league_id, season=season, page=page)
            rows.extend(payload.get("response") or [])
            paging = payload.get("paging") or {}
            if int(paging.get("current") or page) >= int(paging.get("total") or page):
                break
            page += 1
        return rows

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.config.min_interval_s - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()


def _is_daily_quota_error(errors: Any) -> bool:
    """Match the free-plan 100/day exhaustion message specifically.

    Typical body: ``{"errors": {"requests": "You have reached the request limit
    for the day."}}``. Per-minute caps say "rate" not "day".
    """
    if not errors:
        return False
    import json as _json
    text = _json.dumps(errors).lower()
    # Daily-quota messages always mention "day" or a plan cap phrase. Per-minute
    # messages say "10 / minute" or "too many requests" without those tokens.
    return (
        "day" in text
        or "reached the request limit" in text
        or "daily limit" in text
    )


def _is_rate_limit_error(errors: Any) -> bool:
    """API-Football returns errors as either a dict or a list, with various keys.

    Examples seen in the wild:
      * ``{"rateLimit": "Too many requests..."}``
      * ``{"requests": "Account daily limit reached"}``
      * ``[{"rateLimit": "..."}]`` (rare)
    """
    def _check(obj: Any) -> bool:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() in _RATE_LIMIT_ERROR_KEYS:
                    return True
                # Some responses tuck the message under a generic key
                if isinstance(v, str) and "rate" in v.lower() and "limit" in v.lower():
                    return True
        elif isinstance(obj, list):
            return any(_check(item) for item in obj)
        return False
    return _check(errors)


def client_from_env() -> ApiFootballClient | None:
    try:
        return ApiFootballClient()
    except RuntimeError:
        return None


def load_dotenv_if_present(path: str | Path | None = None) -> bool:
    env_path = Path(path) if path is not None else PROJECT_ROOT / ".env"
    return load_dotenv_file(env_path)


def api_key_status() -> dict[str, Any]:
    settings = get_settings()
    return {
        "present": settings.has_api_football_key,
        "source": settings.football_api_key_source,
        "masked": settings.masked_football_api_key,
        "dotenv_loaded": settings.dotenv_loaded,
    }


def discover_leagues(
    *,
    country: str | None = None,
    search: str | None = None,
    league_id: int | None = None,
    season: int | None = None,
    client: ApiFootballClient | None = None,
) -> pd.DataFrame:
    client = client or ApiFootballClient()
    rows = client.fetch_leagues(
        country=country,
        search=search,
        league_id=league_id,
        season=season,
    )
    return leagues_to_frame(rows)


def leagues_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        league = row.get("league") or {}
        country = row.get("country") or {}
        seasons = row.get("seasons") or []
        normalized.append(
            {
                "league_id": league.get("id"),
                "league_name": league.get("name"),
                "league_type": league.get("type"),
                "country_name": country.get("name"),
                "country_code": country.get("code"),
                "seasons": [season.get("year") for season in seasons if season.get("year")],
                "raw": row,
            }
        )
    if not normalized:
        return _empty_leagues()
    return pd.DataFrame(normalized)


def fetch_league_results(
    league: League,
    *,
    seasons: Iterable[int],
    cache_dir: Path,
    client: ApiFootballClient | None = None,
) -> pd.DataFrame:
    if league.api_football_id is None:
        return _empty()
    client = client or ApiFootballClient()
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for season in seasons:
        cache_path = cache_dir / f"api_football_{league.key}_{season}.json"
        if cache_path.exists() and not _season_may_be_active(season, league):
            fixtures = json.loads(cache_path.read_text())
        else:
            fixtures = client.fetch_fixtures(
                league_id=int(league.api_football_id),
                season=int(season),
            )
            cache_path.write_text(json.dumps(fixtures))
        frame = fixtures_to_matches(fixtures, league=league)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return _empty()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def fetch_league_player_stats(
    league: League,
    *,
    season: int,
    cache_dir: Path,
    client: ApiFootballClient | None = None,
) -> pd.DataFrame:
    if league.api_football_id is None:
        return _empty_player_stats()
    client = client or ApiFootballClient()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"api_football_players_{league.key}_{season}.json"
    if cache_path.exists() and not _season_may_be_active(season, league):
        payload = json.loads(cache_path.read_text())
    else:
        payload = client.fetch_all_players(
            league_id=int(league.api_football_id),
            season=int(season),
        )
        cache_path.write_text(json.dumps(payload))
    return players_to_season_stats(payload, league=league, season=season)


def fixtures_to_matches(fixtures: list[dict[str, Any]], *, league: League) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in fixtures:
        fixture = item.get("fixture") or {}
        status = (fixture.get("status") or {}).get("short")
        goals = item.get("goals") or {}
        teams = item.get("teams") or {}
        if status not in FINISHED_STATUSES:
            continue
        if goals.get("home") is None or goals.get("away") is None:
            continue

        home_team = (teams.get("home") or {}).get("name")
        away_team = (teams.get("away") or {}).get("name")
        if not home_team or not away_team:
            continue

        fixture_date = pd.to_datetime(fixture.get("date"), utc=True, errors="coerce")
        if pd.isna(fixture_date):
            continue

        rows.append(
            {
                "source": "api-football",
                "league_key": league.key,
                "league_name": league.name,
                "season": str((item.get("league") or {}).get("season") or ""),
                "date": fixture_date.date(),
                "home_team": str(home_team),
                "away_team": str(away_team),
                "home_goals": int(goals["home"]),
                "away_goals": int(goals["away"]),
                "result": _result(int(goals["home"]), int(goals["away"])),
                "stage": (item.get("league") or {}).get("round"),
                "neutral_site": None,
                "raw": item,
            }
        )
    return pd.DataFrame(rows)


def players_to_season_stats(
    rows: list[dict[str, Any]],
    *,
    league: League,
    season: int,
) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        player = row.get("player") or {}
        player_id = player.get("id")
        player_name = player.get("name")
        if player_id is None or not player_name:
            continue
        birth = player.get("birth") or {}
        for stat in row.get("statistics") or []:
            team = stat.get("team") or {}
            games = stat.get("games") or {}
            goals = stat.get("goals") or {}
            if not team.get("name"):
                continue
            normalized.append(
                {
                    "source": "api-football",
                    "player_external_id": str(player_id),
                    "player_name": str(player_name),
                    "birth_date": birth.get("date"),
                    "age": player.get("age"),
                    "nationality": player.get("nationality"),
                    "team": team.get("name"),
                    "league_key": league.key,
                    "season": str(season),
                    "position": games.get("position"),
                    "appearances": games.get("appearences") or games.get("appearances"),
                    "lineups": games.get("lineups"),
                    "minutes": games.get("minutes"),
                    "goals": goals.get("total"),
                    "assists": goals.get("assists"),
                    "rating": games.get("rating"),
                    "raw": row,
                }
            )
    return pd.DataFrame(normalized)


def recent_api_seasons(league: League, *, years_back: int = 5, today: date | None = None) -> list[int]:
    today = today or date.today()
    if league.country in {"USA", "BRA", "ARG", "JPN", "KOR", "CHN"}:
        current = today.year
    elif league.key == "liga_mx":
        current = today.year
    else:
        current = today.year if today.month >= 7 else today.year - 1
    return list(range(current - years_back + 1, current + 1))


def _read_api_key() -> str:
    settings = get_settings()
    if not settings.football_api_key:
        raise RuntimeError("Set FOOTBALL_API_KEY or API_FOOTBALL_KEY to use API-Football.")
    return settings.football_api_key


def _season_may_be_active(season: int, league: League) -> bool:
    today = date.today()
    if league.country in {"USA", "BRA", "ARG", "JPN", "KOR", "CHN"}:
        return season == today.year
    start = date(season, 7, 1)
    end = date(season + 1, 7, 1)
    return start <= today < end


def _result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source",
            "league_key",
            "league_name",
            "season",
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
            "stage",
            "neutral_site",
            "raw",
        ]
    )


def _empty_player_stats() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source",
            "player_external_id",
            "player_name",
            "birth_date",
            "age",
            "nationality",
            "team",
            "league_key",
            "season",
            "position",
            "appearances",
            "lineups",
            "minutes",
            "goals",
            "assists",
            "rating",
            "raw",
        ]
    )


# API-Football Free plan only allows historical seasons 2022/2023/2024 (as of
# 2026-05). Current and previous seasons are gated to paid plans. Pin to the
# free-tier max by default; callers can override via ``seasons=``.
FREE_PLAN_MAX_SEASON = 2024
FREE_PLAN_MIN_SEASON = 2022


# Mirror football_data_org's convenience API so the user can swap freely
# between sources with the same call shape.
def upsert_matches_into_db(
    db,
    league_key: str,
    *,
    cache_dir: Path,
    years_back: int = 3,
    api_key: str | None = None,
    seasons: Iterable[int] | None = None,
) -> int:
    """Fetch ``years_back`` seasons of ``league_key`` and upsert into matches table.

    Requires ``FOOTBALL_API_KEY`` (or ``API_FOOTBALL_KEY``) env var, OR an
    explicit ``api_key`` argument. One API request per season (~3 per call).

    On the Free plan the API only exposes seasons 2022-2024 — newer ones return
    "Free plans do not have access". By default we clamp to that range.

    Returns the number of match rows touched.
    """
    from scrape.registry import LeagueRegistry

    league = LeagueRegistry().leagues.get(league_key)
    if league is None or league.api_football_id is None:
        raise ValueError(
            f"League '{league_key}' has no api_football_id. Check config/leagues.yaml."
        )

    if api_key is not None:
        client = ApiFootballClient(ApiFootballConfig(api_key=api_key))
    else:
        client = ApiFootballClient()

    if seasons is None:
        # Default: clamp to Free-plan-accessible range.
        end = FREE_PLAN_MAX_SEASON
        start = max(FREE_PLAN_MIN_SEASON, end - years_back + 1)
        season_list = list(range(start, end + 1))
    else:
        season_list = list(seasons)

    frame = fetch_league_results(
        league, seasons=season_list, cache_dir=cache_dir, client=client,
    )
    if frame.empty:
        return 0
    return db.upsert_matches(
        frame, source="api-football",
        league_key=league_key, league_name=league.name,
    )


def quota_status(*, api_key: str | None = None) -> dict[str, Any]:
    """One-call /status that returns your daily quota use. Costs 1 request."""
    if api_key is not None:
        client = ApiFootballClient(ApiFootballConfig(api_key=api_key))
    else:
        client = ApiFootballClient()
    payload = client.get("/status")
    resp = payload.get("response") or {}
    requests_info = resp.get("requests") or {}
    return {
        "account_email": (resp.get("account") or {}).get("email"),
        "plan": (resp.get("subscription") or {}).get("plan"),
        "active_until": (resp.get("subscription") or {}).get("end"),
        "requests_today": requests_info.get("current"),
        "requests_limit_day": requests_info.get("limit_day"),
    }


def _empty_leagues() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "league_id",
            "league_name",
            "league_type",
            "country_name",
            "country_code",
            "seasons",
            "raw",
        ]
    )
