from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloConfig:
    initial_rating: float = 1500.0
    k_factor: float = 22.0
    home_advantage: float = 65.0
    scale: float = 400.0
    min_goal_multiplier: float = 1.0
    max_goal_multiplier: float = 2.2


def expected_score(
    rating_a: float,
    rating_b: float,
    *,
    scale: float = 400.0,
) -> float:
    return float(1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / scale)))


def attach_pre_match_elos(
    matches: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    config: EloConfig | None = None,
) -> pd.DataFrame:
    """Attach leakage-safe pre-match Elo columns to a chronological result frame."""
    config = config or EloConfig()
    frame = _prepare_matches(matches)
    ratings: dict[str, float] = {}
    home_elos: list[float] = []
    away_elos: list[float] = []
    home_post: list[float] = []
    away_post: list[float] = []

    for row in frame.itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        home_rating = ratings.get(home, config.initial_rating)
        away_rating = ratings.get(away, config.initial_rating)
        home_elos.append(home_rating)
        away_elos.append(away_rating)

        actual = _actual_home_score(int(row.home_goals), int(row.away_goals))
        expected = expected_score(
            home_rating + config.home_advantage,
            away_rating,
            scale=config.scale,
        )
        multiplier = _goal_multiplier(
            abs(int(row.home_goals) - int(row.away_goals)),
            config=config,
        )
        delta = config.k_factor * multiplier * (actual - expected)
        ratings[home] = home_rating + delta
        ratings[away] = away_rating - delta
        home_post.append(ratings[home])
        away_post.append(ratings[away])

    enriched = frame.copy()
    enriched["home_elo"] = home_elos
    enriched["away_elo"] = away_elos
    enriched["home_elo_post"] = home_post
    enriched["away_elo_post"] = away_post
    return enriched


def latest_elos(
    matches: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    config: EloConfig | None = None,
) -> dict[str, float]:
    frame = attach_pre_match_elos(matches, config=config)
    ratings: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        ratings[str(row.home_team)] = float(row.home_elo_post)
        ratings[str(row.away_team)] = float(row.away_elo_post)
    return ratings


def _prepare_matches(matches: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(matches).copy()
    required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing match columns for Elo: {sorted(missing)}")
    frame = frame.dropna(subset=list(required))
    frame["date"] = pd.to_datetime(frame["date"])
    frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce").astype(int)
    frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce").astype(int)
    return frame.sort_values("date").reset_index(drop=True)


def _actual_home_score(home_goals: int, away_goals: int) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def _goal_multiplier(goal_diff: int, *, config: EloConfig) -> float:
    if goal_diff <= 1:
        return config.min_goal_multiplier
    multiplier = 1.0 + np.log1p(goal_diff - 1)
    return float(np.clip(multiplier, config.min_goal_multiplier, config.max_goal_multiplier))
