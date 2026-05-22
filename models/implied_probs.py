"""Implied-probability extraction from bookmaker odds.

Wraps penaltyblog's `calculate_implied`. The naive way to convert odds to
probabilities is `1/odds` then divide each by the sum (multiplicative
normalization). That assumes the bookmaker spreads the overround (vig)
uniformly across all outcomes. In reality bookmakers apply different
margins to favorites vs longshots — Shin's method explicitly models this.

Methods supported (penaltyblog 1.9):

  * ``multiplicative`` (default in many places) — divide each by sum
  * ``shin`` — assumes proportion ``z`` of bets are from "insiders" with
    perfect information; solves for ``z`` and unnormalized probs
  * ``power``    — overround applied as a power
  * ``additive`` — overround spread additively

For typical match odds with low overround (3-5%) all methods agree to
within ~1 pp. The differences matter for big handicaps and longshots.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import penaltyblog as pb


@dataclass(frozen=True)
class ImpliedResult:
    method: str
    margin: float            # the bookmaker overround (vig)
    home_win: float
    draw: float
    away_win: float

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "margin": self.margin,
            "probabilities": {
                "home_win": self.home_win,
                "draw": self.draw,
                "away_win": self.away_win,
            },
        }


def implied_probabilities(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    *,
    method: str = "shin",
) -> ImpliedResult:
    """Extract de-vigged probabilities from 1X2 decimal odds.

    Default is Shin because it's the most commonly cited fair-method in the
    research literature (Strumbelj 2014, Shin 1993).
    """
    if method not in {"multiplicative", "shin", "power", "additive"}:
        raise ValueError(f"Unknown implied-prob method '{method}'.")
    res = pb.implied.calculate_implied([odds_home, odds_draw, odds_away], method=method)
    p = list(res.probabilities)
    return ImpliedResult(
        method=method,
        margin=float(res.margin),
        home_win=float(p[0]),
        draw=float(p[1]),
        away_win=float(p[2]),
    )
