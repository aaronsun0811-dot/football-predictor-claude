from __future__ import annotations

from pathlib import Path
from typing import Any

from data.coverage import build_coverage_report
from data.database import DEFAULT_DB_PATH, Base, Database
from data.schema import schema_report
from scrape import api_football


def build_doctor_report(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    live_api: bool = False,
) -> dict[str, Any]:
    db = Database(db_path)
    db.init()
    schema = schema_report(db.engine, Base.metadata)
    coverage = build_coverage_report(db_path=db_path)
    leagues = coverage["leagues"]
    api_key = api_football.api_key_status()
    api_probe = _probe_api_football() if live_api and api_key["present"] else {
        "attempted": False,
        "ok": None,
        "message": "Skipped. Use --live with FOOTBALL_API_KEY set to test API-Football.",
    }

    empty_api_leagues = [
        _league_brief(row)
        for row in leagues
        if row["status"] == "empty" and row.get("api_football_id") is not None
    ]
    sparse_leagues = [
        _league_brief(row)
        for row in leagues
        if row["status"] == "sparse"
    ]
    xg_rows = sum(int(row.get("xg_match_rows") or 0) for row in leagues)

    return {
        "status": _overall_status(
            coverage=coverage,
            schema_ok=bool(schema["ok"]),
            api_key_present=bool(api_key["present"]),
            empty_api_leagues=empty_api_leagues,
        ),
        "api_football": {
            **api_key,
            "probe": api_probe,
        },
        "database": {
            **coverage["summary"],
            "xg_match_rows": xg_rows,
            "schema_ok": schema["ok"],
        },
        "schema": schema,
        "leagues": {
            "empty_api_leagues": empty_api_leagues,
            "sparse_leagues": sparse_leagues,
        },
        "recommended_commands": _recommended_commands(
            schema_ok=bool(schema["ok"]),
            api_key_present=bool(api_key["present"]),
            empty_api_leagues=empty_api_leagues,
            xg_rows=xg_rows,
        ),
    }


def _probe_api_football() -> dict[str, Any]:
    try:
        frame = api_football.discover_leagues(league_id=39)
    except Exception as exc:
        return {
            "attempted": True,
            "ok": False,
            "message": str(exc),
        }
    return {
        "attempted": True,
        "ok": not frame.empty,
        "message": "API-Football responded.",
        "sample_count": int(len(frame)),
    }


def _league_brief(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": row["key"],
        "name": row["name"],
        "tier": row["tier"],
        "api_football_id": row.get("api_football_id"),
        "match_count": row.get("match_count", 0),
        "xg_match_rows": row.get("xg_match_rows", 0),
        "latest_match": row.get("latest_match"),
    }


def _overall_status(
    *,
    coverage: dict[str, Any],
    schema_ok: bool,
    api_key_present: bool,
    empty_api_leagues: list[dict[str, Any]],
) -> str:
    # Schema problems are foundational. Report them before data-source gaps.
    if not schema_ok:
        return "schema_problem"
    if empty_api_leagues and not api_key_present:
        return "needs_api_key"
    if coverage["summary"]["empty_leagues"]:
        return "needs_data_update"
    return "ready"


def _recommended_commands(
    *,
    schema_ok: bool,
    api_key_present: bool,
    empty_api_leagues: list[dict[str, Any]],
    xg_rows: int,
) -> list[str]:
    commands: list[str] = []
    if not schema_ok:
        commands.append("python predict.py init-db")
    if not api_key_present:
        commands.extend(
            [
                "cp .env.example .env",
                "在 .env 里填写 FOOTBALL_API_KEY=你的_API_Football_key",
                "python predict.py doctor --live",
            ]
        )
    else:
        commands.append("python predict.py doctor --live")
        for league in empty_api_leagues[:5]:
            commands.append(
                f"python predict.py update --league {league['key']} --include-api-football --years-back 3"
            )
        commands.append("python predict.py update --league 英超 --include-api-football --include-players")

    if xg_rows == 0:
        commands.append("python predict.py update --league 英超 --include-xg --no-include-ratings")
    return commands
