"""Over/Under and Both-Teams-To-Score derivations from a score-probability matrix.

Given a Dixon-Coles ``score_matrix`` (2D probability grid: home goals × away
goals), compute:

  * ``expected_total`` — mean of home + away goals
  * Over-line probabilities at 0.5 / 1.5 / 2.5 / 3.5 / 4.5
  * BTTS — both teams to score (= 1 − P(any zero))

These are the standard ancillary markets sports books offer.
"""
from __future__ import annotations

from typing import Any

import numpy as np


OVER_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)


def derive_totals(score_matrix: list[list[float]] | np.ndarray | None) -> dict[str, Any]:
    """Compute O/U + BTTS + expected total from a score-prob matrix."""
    if score_matrix is None:
        return {"available": False}
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.size == 0:
        return {"available": False}
    # Normalize defensively (the Dixon-Coles τ correction can leave it slightly off 1.0)
    total_mass = matrix.sum()
    if total_mass <= 0:
        return {"available": False}
    matrix = matrix / total_mass

    n_home, n_away = matrix.shape
    home_grid, away_grid = np.indices((n_home, n_away))
    totals = home_grid + away_grid

    expected_total = float((matrix * totals).sum())

    # Over X.5 = P(total >= ceil(X.5)) = P(total > X)
    over_probs = {}
    for line in OVER_LINES:
        threshold = int(np.floor(line)) + 1  # 0.5 → 1, 1.5 → 2, etc.
        over_p = float(matrix[totals >= threshold].sum())
        over_probs[f"over_{line}"] = over_p
        over_probs[f"under_{line}"] = 1.0 - over_p

    # BTTS = P(home >= 1) * P(away >= 1) WOULD be wrong (assumes independence).
    # Correct way from the joint: sum cells where both home and away ≥ 1.
    btts_mask = (home_grid >= 1) & (away_grid >= 1)
    btts_yes = float(matrix[btts_mask].sum())

    return {
        "available": True,
        "expected_total": expected_total,
        "lines": over_probs,
        "btts_yes": btts_yes,
        "btts_no": 1.0 - btts_yes,
    }
