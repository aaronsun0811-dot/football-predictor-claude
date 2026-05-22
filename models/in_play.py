"""In-play (live) match prediction.

Given the **current score** at the **current minute**, what's the probability
distribution of the final result?

Method:

1. Get pre-match expected goals from a fitted model (Dixon-Coles / Bivariate
   Poisson / etc.). Call these ``xH``, ``xA``.

2. Compute the fraction of the match remaining:
   ``r = max(0, (regulation_minutes - minute_elapsed) / regulation_minutes)``.
   We use 90 minutes as regulation by default; injury time is small enough
   to fold into the Poisson noise.

3. Expected remaining goals for each side scale linearly with remaining time:
   ``xH_rem = xH * r * game_state_multiplier_home``
   ``xA_rem = xA * r * game_state_multiplier_away``

4. The **game-state multiplier** captures a well-documented effect: the team
   currently TRAILING tends to score more than its baseline rate (more risk),
   while the LEADING team scores less (parking the bus). Effect sizes in the
   literature are ~10-20%. We use 1.15 for the trailing side and 0.92 for
   the leading side; tied scores get 1.0 each. The multipliers are tunable.

5. Final goals distribution = current_goals + Poisson(xH_rem) for each side,
   computed as a 2D PMF up to ``max_remaining_goals`` per side.

6. Final probabilities (home win / draw / away win) come from collapsing
   the joint distribution.

This is "what's the score going to be at full time given what's happened
so far". It does NOT model:
  - Red cards (drastic xG shift)
  - Substitutions, injuries
  - Big derby tactical adjustments

But for a quick "Arsenal 1-0 Chelsea at 60' — how likely to hold on?"
calculator, it's surprisingly good and fully grounded in the same goal
model used for the pre-match prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import gammaln


REGULATION_MINUTES = 90


@dataclass(frozen=True)
class InPlayConfig:
    regulation_minutes: int = REGULATION_MINUTES
    # Multiplier on a team's xG-remaining when that team is trailing.
    chasing_multiplier: float = 1.15
    # Multiplier on a team's xG-remaining when that team is leading.
    leading_multiplier: float = 0.92
    # Highest number of remaining goals to enumerate per side (the rest is
    # truncated; with xG_rem rarely above 2, max=8 captures 99.9%).
    max_remaining_goals: int = 8


@dataclass(frozen=True)
class InPlayPrediction:
    home_team: str
    away_team: str
    current_home: int
    current_away: int
    minute_elapsed: int
    minutes_remaining: int

    # Final-result probabilities (full time, including current score).
    home_win: float
    draw: float
    away_win: float

    # Expected final goals (current + remaining).
    expected_home_final: float
    expected_away_final: float
    expected_home_remaining: float
    expected_away_remaining: float

    # Multiplier applied to each side based on game state.
    home_state_multiplier: float
    away_state_multiplier: float

    # Top 5 most-likely final scorelines.
    most_likely_final_scores: list[dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "current_score": {
                "home": self.current_home,
                "away": self.current_away,
            },
            "minute_elapsed": self.minute_elapsed,
            "minutes_remaining": self.minutes_remaining,
            "probabilities": {
                "home_win": self.home_win,
                "draw": self.draw,
                "away_win": self.away_win,
            },
            "expected_final_goals": {
                "home": self.expected_home_final,
                "away": self.expected_away_final,
            },
            "expected_remaining_goals": {
                "home": self.expected_home_remaining,
                "away": self.expected_away_remaining,
            },
            "game_state_multipliers": {
                "home": self.home_state_multiplier,
                "away": self.away_state_multiplier,
            },
            "most_likely_final_scores": self.most_likely_final_scores,
        }


def predict_in_play(
    *,
    home_team: str,
    away_team: str,
    pre_match_xg_home: float,
    pre_match_xg_away: float,
    current_home: int,
    current_away: int,
    minute_elapsed: int,
    config: InPlayConfig | None = None,
) -> InPlayPrediction:
    """Recompute full-time probabilities given live state.

    ``pre_match_xg_home`` / ``pre_match_xg_away`` are the expected goals
    BEFORE the match started, from the fitted model. The function does not
    do model fitting itself — it accepts those as inputs so callers can
    use whichever model they like (our DC+Elo, penaltyblog DC, ensemble...).
    """
    config = config or InPlayConfig()
    if minute_elapsed < 0:
        raise ValueError("minute_elapsed must be >= 0")
    if current_home < 0 or current_away < 0:
        raise ValueError("Current scores must be non-negative")
    if pre_match_xg_home <= 0 or pre_match_xg_away <= 0:
        raise ValueError("Pre-match xG must be positive")

    clipped_minute = min(minute_elapsed, config.regulation_minutes)
    minutes_remaining = max(0, config.regulation_minutes - clipped_minute)
    r = minutes_remaining / config.regulation_minutes

    if current_home > current_away:
        home_mult, away_mult = config.leading_multiplier, config.chasing_multiplier
    elif current_home < current_away:
        home_mult, away_mult = config.chasing_multiplier, config.leading_multiplier
    else:
        home_mult = away_mult = 1.0

    xh_rem = max(pre_match_xg_home * r * home_mult, 1e-9)
    xa_rem = max(pre_match_xg_away * r * away_mult, 1e-9)

    home_pmf = _poisson_pmf(xh_rem, config.max_remaining_goals)
    away_pmf = _poisson_pmf(xa_rem, config.max_remaining_goals)
    joint = np.outer(home_pmf, away_pmf)
    joint = joint / joint.sum()  # renormalize against tail truncation

    home_win, draw, away_win = _aggregate_outcomes(
        joint, current_home, current_away,
    )

    top_scores = _top_final_scores(
        joint, current_home, current_away, n=5,
    )

    return InPlayPrediction(
        home_team=home_team,
        away_team=away_team,
        current_home=current_home,
        current_away=current_away,
        minute_elapsed=clipped_minute,
        minutes_remaining=minutes_remaining,
        home_win=float(home_win),
        draw=float(draw),
        away_win=float(away_win),
        expected_home_final=float(current_home + xh_rem),
        expected_away_final=float(current_away + xa_rem),
        expected_home_remaining=float(xh_rem),
        expected_away_remaining=float(xa_rem),
        home_state_multiplier=float(home_mult),
        away_state_multiplier=float(away_mult),
        most_likely_final_scores=top_scores,
    )


def _poisson_pmf(lam: float, max_k: int) -> np.ndarray:
    """Discrete Poisson PMF as a numpy vector (k=0..max_k)."""
    k = np.arange(max_k + 1, dtype=float)
    log_pmf = k * np.log(lam) - lam - gammaln(k + 1.0)
    return np.exp(log_pmf)


def _aggregate_outcomes(
    joint_remaining: np.ndarray,
    current_home: int,
    current_away: int,
) -> tuple[float, float, float]:
    """Collapse the remaining-goals joint PMF onto W/D/L given current score."""
    max_k = joint_remaining.shape[0] - 1
    h_remain = np.arange(max_k + 1).reshape(-1, 1)
    a_remain = np.arange(max_k + 1).reshape(1, -1)
    final_home = current_home + h_remain
    final_away = current_away + a_remain
    home_mask = final_home > final_away
    away_mask = final_home < final_away
    draw_mask = final_home == final_away
    return (
        float(joint_remaining[home_mask].sum()),
        float(joint_remaining[draw_mask].sum()),
        float(joint_remaining[away_mask].sum()),
    )


def _top_final_scores(
    joint_remaining: np.ndarray,
    current_home: int,
    current_away: int,
    *,
    n: int = 5,
) -> list[dict[str, float | int]]:
    flat = np.argsort(joint_remaining.ravel())[::-1][:n]
    out = []
    for idx in flat:
        h_add, a_add = np.unravel_index(idx, joint_remaining.shape)
        out.append({
            "home_goals": int(current_home + h_add),
            "away_goals": int(current_away + a_add),
            "probability": float(joint_remaining[h_add, a_add]),
        })
    return out
