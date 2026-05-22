"""Per-team strength extraction.

Fits Dixon-Coles on a league's matches and exposes each team's:

  * **attack**   — log-multiplier on expected goals scored
  * **defense**  — log-multiplier on expected goals conceded
  * **overall**  — attack − defense (positive = strong both ways)
  * **club_elo** — latest ClubElo if available (else internal Elo)
  * **recent_form**  — last 5 matches: GD points and W/D/L sequence
  * **home_advantage** — same for all teams in a league, included once

These are read by the 球队强度 (Team Strengths) tab so a user can sort by
attack or defense, eyeball who's actually strong, and put two teams side
by side.

Note on the sign convention: Dixon-Coles uses attack > 0 ⇒ scores MORE than
average, defense > 0 ⇒ concedes MORE than average. For UX we flip defense
so that "higher = better" both for attack and defense.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from models.dixon_coles import DixonColesConfig, DixonColesModel


@dataclass(frozen=True)
class TeamStrength:
    team: str
    attack: float           # raw DC parameter (higher = scores more)
    defense_inverted: float # flipped so higher = concedes less
    overall: float          # attack + defense_inverted
    matches_in_window: int
    recent_form: str        # "WWDLW" style
    recent_gd: int          # goal difference over recent_form window
    last_5_dates: list[str]
    home_elo: float | None  # club Elo if available (latest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "attack": self.attack,
            "defense": self.defense_inverted,
            "overall": self.overall,
            "matches_in_window": self.matches_in_window,
            "recent_form": self.recent_form,
            "recent_gd": self.recent_gd,
            "last_5_dates": self.last_5_dates,
            "club_elo": self.home_elo,
        }


def extract_strengths(
    matches: pd.DataFrame,
    *,
    lookback_days: int = 730,
    home_advantage: float = 0.22,
    max_goals: int = 8,
    optimizer_maxiter: int = 2500,
    club_elo_lookup: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fit Dixon-Coles on ``matches`` and return per-team strengths sorted by overall."""
    if matches.empty:
        return {"teams": [], "model": None}

    frame = _prepare(matches)
    model = DixonColesModel(
        DixonColesConfig(
            home_advantage=home_advantage,
            max_goals=max_goals,
            optimizer_maxiter=optimizer_maxiter,
            lookback_days=lookback_days,
        )
    ).fit(frame)

    if model.attack_ is None or model.defense_ is None:
        return {"teams": [], "model": None}

    # Recent form per team (most recent N matches)
    form_window = 5
    team_form = _per_team_recent_form(frame, window=form_window)
    appearances = _per_team_match_counts(frame)

    rows: list[TeamStrength] = []
    for team, idx in model.team_to_idx_.items():
        attack = float(model.attack_[idx])
        defense_raw = float(model.defense_[idx])
        # Defense lower = better at preventing goals. Flip the sign for UX.
        defense_inverted = -defense_raw
        form = team_form.get(team, {"sequence": "", "gd": 0, "dates": []})
        rows.append(TeamStrength(
            team=team,
            attack=attack,
            defense_inverted=defense_inverted,
            overall=attack + defense_inverted,
            matches_in_window=int(appearances.get(team, 0)),
            recent_form=form["sequence"],
            recent_gd=form["gd"],
            last_5_dates=form["dates"],
            home_elo=(club_elo_lookup or {}).get(team),
        ))

    rows.sort(key=lambda r: -r.overall)
    return {
        "teams": [r.to_dict() for r in rows],
        "model": {
            "intercept": model.intercept_,
            "home_advantage": model.home_advantage_,
            "rho": model.rho_,
            "training_rows": model.training_rows_,
            "as_of": str(model.as_of_.date() if model.as_of_ is not None else date.today()),
        },
    }


def compare_two(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience: extract strengths and pull out just two teams + a diff."""
    out = extract_strengths(matches, **kwargs)
    by_team = {row["team"]: row for row in out["teams"]}
    if home_team not in by_team or away_team not in by_team:
        missing = home_team if home_team not in by_team else away_team
        raise KeyError(f"Team '{missing}' not in this league's fitted strengths.")
    home = by_team[home_team]
    away = by_team[away_team]
    return {
        "home": home,
        "away": away,
        "differential": {
            "attack": home["attack"] - away["attack"],
            "defense": home["defense"] - away["defense"],
            "overall": home["overall"] - away["overall"],
        },
        "model": out["model"],
    }


def _prepare(matches: pd.DataFrame) -> pd.DataFrame:
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


def _per_team_recent_form(frame: pd.DataFrame, *, window: int = 5) -> dict[str, dict[str, Any]]:
    """Return {team: {"sequence": "WWDLW", "gd": -1, "dates": [...]}}."""
    out: dict[str, dict[str, Any]] = {}
    # Build per-team timeline of (date, result, gd_for_team)
    timelines: dict[str, list[tuple[pd.Timestamp, str, int]]] = {}
    for row in frame.itertuples(index=False):
        h, a, hg, ag, d = row.home_team, row.away_team, row.home_goals, row.away_goals, row.date
        if hg > ag:
            home_res, away_res = "W", "L"
        elif hg < ag:
            home_res, away_res = "L", "W"
        else:
            home_res, away_res = "D", "D"
        timelines.setdefault(h, []).append((d, home_res, hg - ag))
        timelines.setdefault(a, []).append((d, away_res, ag - hg))

    for team, events in timelines.items():
        events.sort(key=lambda x: x[0])
        recent = events[-window:]
        out[team] = {
            "sequence": "".join(r for _, r, _ in recent),
            "gd": sum(g for _, _, g in recent),
            "dates": [str(d.date()) for d, _, _ in recent],
        }
    return out


def _per_team_match_counts(frame: pd.DataFrame) -> dict[str, int]:
    home = frame["home_team"].value_counts()
    away = frame["away_team"].value_counts()
    return (home.add(away, fill_value=0)).astype(int).to_dict()
