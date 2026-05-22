from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppSettings:
    project_root: Path
    db_path: Path
    cache_dir: Path
    export_dir: Path
    static_dir: Path
    worldcup_cache_path: Path
    timezone: str
    scheduler_enabled: bool
    update_hour: int
    update_minute: int
    daily_api_football: bool
    daily_fbref_xg: bool
    results_backfill_enabled: bool
    results_backfill_hour: int
    results_backfill_minute: int
    results_backfill_days: int
    # Periodic /upcoming snapshot — runs the prediction pipeline + appends to
    # history.jsonl on a schedule, so the audit module has data to evaluate
    # even when nobody hits the endpoint by hand.
    upcoming_snapshot_enabled: bool
    upcoming_snapshot_hours: str   # comma-separated cron hour list, e.g. "9,15,21,3"
    upcoming_snapshot_days_ahead: int
    football_api_key: str | None
    football_api_key_source: str | None
    api_football_base_url: str
    api_football_timeout_s: float
    api_football_min_interval_s: float
    # Token-bucket rate limiter: capacity = refill rate per minute.
    # Free plan = 10/min. Paid plans (Pro=300/min, Mega=7500/min) can override
    # via env. Set to 0 to disable the limiter (e.g. for tests with mocked HTTP).
    api_football_rate_per_min: int
    api_football_429_backoff_s: float
    dotenv_path: Path
    dotenv_loaded: bool

    @property
    def has_api_football_key(self) -> bool:
        return bool(self.football_api_key)

    @property
    def masked_football_api_key(self) -> str | None:
        return mask_secret(self.football_api_key)


def get_settings(*, env_file: str | Path | None = None) -> AppSettings:
    dotenv_path = Path(env_file) if env_file is not None else PROJECT_ROOT / ".env"
    dotenv_loaded = load_dotenv_file(dotenv_path)
    api_key_source, api_key = _first_env_value("FOOTBALL_API_KEY", "API_FOOTBALL_KEY")

    return AppSettings(
        project_root=PROJECT_ROOT,
        db_path=_path_env("FOOTBALL_PREDICTOR_DB_PATH", PROJECT_ROOT / "data" / "football.sqlite3"),
        cache_dir=_path_env("FOOTBALL_PREDICTOR_CACHE_DIR", PROJECT_ROOT / "data" / "cache"),
        export_dir=_path_env("FOOTBALL_PREDICTOR_EXPORT_DIR", PROJECT_ROOT / "data" / "exports"),
        static_dir=_path_env("FOOTBALL_PREDICTOR_STATIC_DIR", PROJECT_ROOT / "static"),
        worldcup_cache_path=_path_env(
            "FOOTBALL_PREDICTOR_WORLDCUP_CACHE",
            PROJECT_ROOT / "data" / "worldcup_forecast.json",
        ),
        timezone=os.getenv("FOOTBALL_PREDICTOR_TZ", "Asia/Shanghai"),
        scheduler_enabled=_bool_env("FOOTBALL_PREDICTOR_ENABLE_SCHEDULER", True),
        update_hour=_int_env("FOOTBALL_PREDICTOR_UPDATE_HOUR", 3),
        update_minute=_int_env("FOOTBALL_PREDICTOR_UPDATE_MINUTE", 30),
        daily_api_football=_bool_env("FOOTBALL_PREDICTOR_DAILY_API_FOOTBALL", False),
        daily_fbref_xg=_bool_env("FOOTBALL_PREDICTOR_DAILY_FBREF_XG", False),
        # Lightweight "last-N-days finished results" refresh. Cheaper than the
        # full season pull; needed by the prediction-audit module.
        results_backfill_enabled=_bool_env("FOOTBALL_PREDICTOR_RESULTS_BACKFILL", True),
        results_backfill_hour=_int_env("FOOTBALL_PREDICTOR_RESULTS_BACKFILL_HOUR", 6),
        results_backfill_minute=_int_env("FOOTBALL_PREDICTOR_RESULTS_BACKFILL_MINUTE", 15),
        results_backfill_days=_int_env("FOOTBALL_PREDICTOR_RESULTS_BACKFILL_DAYS", 7),
        # Periodic /upcoming snapshots — keep history.jsonl growing even when
        # the UI isn't being hit. Default cadence: 4x daily at 09/15/21/03 local.
        upcoming_snapshot_enabled=_bool_env("FOOTBALL_PREDICTOR_UPCOMING_SNAPSHOT", True),
        upcoming_snapshot_hours=os.getenv("FOOTBALL_PREDICTOR_UPCOMING_SNAPSHOT_HOURS", "9,15,21,3"),
        upcoming_snapshot_days_ahead=_int_env("FOOTBALL_PREDICTOR_UPCOMING_SNAPSHOT_DAYS", 14),
        football_api_key=api_key,
        football_api_key_source=api_key_source,
        api_football_base_url=os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io"),
        api_football_timeout_s=_float_env("API_FOOTBALL_TIMEOUT_S", 45.0),
        api_football_min_interval_s=_float_env("API_FOOTBALL_MIN_INTERVAL_S", 0.0),
        api_football_rate_per_min=_int_env("API_FOOTBALL_RATE_PER_MIN", 10),
        api_football_429_backoff_s=_float_env("API_FOOTBALL_429_BACKOFF_S", 60.0),
        dotenv_path=dotenv_path,
        dotenv_loaded=dotenv_loaded,
    )


def load_dotenv_file(path: str | Path) -> bool:
    env_path = Path(path)
    if not env_path.exists():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    load_dotenv(env_path, override=False)
    return True


def mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _first_env_value(*names: str) -> tuple[str | None, str | None]:
    for name in names:
        value = os.getenv(name)
        if value:
            return name, value
    return None, None


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser() if value else default
