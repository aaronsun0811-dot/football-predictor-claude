"""Backtest diagnostics: calibration curve + confidence ladder.

These are the classic "is the model actually good?" diagnostics — much more
informative than headline accuracy alone.

* **Calibration curve** — bin predictions by predicted probability and
  measure realized frequency. A perfectly calibrated model lies on y=x;
  bins above y=x mean the model is under-confident in that range,
  below means over-confident. Computed separately per outcome
  (home_win / draw / away_win) so you can see, e.g., that the model
  systematically under-predicts draws.

* **Confidence ladder** — split predictions by the max probability the
  model assigned to its top pick, into bands [50-60%, 60-70%, 70-80%,
  80-90%, 90-100%]. Then report accuracy in each band. A good model
  earns its confidence: when it says 80%, it's right ~80% of the time.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

OUTCOMES = ("home_win", "draw", "away_win")
DEFAULT_BIN_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DEFAULT_CONFIDENCE_BANDS = (
    (0.34, 0.45),
    (0.45, 0.55),
    (0.55, 0.65),
    (0.65, 0.75),
    (0.75, 0.85),
    (0.85, 1.01),
)


def calibration_curve(
    predictions: pd.DataFrame,
    *,
    bin_edges: tuple[float, ...] = DEFAULT_BIN_EDGES,
) -> dict[str, list[dict[str, Any]]]:
    """Bin each outcome's predicted probabilities and measure realized frequency.

    Returns a dict mapping outcome → list of bins, each bin a record with:
      bin_low, bin_high, n, mean_predicted, observed_frequency.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    for outcome in OUTCOMES:
        probs = predictions[outcome].to_numpy(dtype=float)
        actual = (predictions["actual"] == outcome).to_numpy(dtype=int)
        rows = []
        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            # Include the right edge in the last bin so 1.0 doesn't fall off.
            if high >= 1.0:
                mask = (probs >= low) & (probs <= high)
            else:
                mask = (probs >= low) & (probs < high)
            n = int(mask.sum())
            if n == 0:
                rows.append({
                    "bin_low": round(float(low), 3),
                    "bin_high": round(float(high), 3),
                    "n": 0,
                    "mean_predicted": None,
                    "observed_frequency": None,
                })
                continue
            rows.append({
                "bin_low": round(float(low), 3),
                "bin_high": round(float(high), 3),
                "n": n,
                "mean_predicted": float(probs[mask].mean()),
                "observed_frequency": float(actual[mask].mean()),
            })
        result[outcome] = rows
    return result


def confidence_ladder(
    predictions: pd.DataFrame,
    *,
    bands: tuple[tuple[float, float], ...] = DEFAULT_CONFIDENCE_BANDS,
) -> list[dict[str, Any]]:
    """Group predictions by the model's top-pick confidence and report accuracy."""
    probs = predictions[list(OUTCOMES)].to_numpy(dtype=float)
    top_probs = probs.max(axis=1)
    top_idx = probs.argmax(axis=1)
    predicted = np.array([OUTCOMES[i] for i in top_idx])
    actual = predictions["actual"].to_numpy()
    correct = predicted == actual

    rows = []
    for low, high in bands:
        mask = (top_probs >= low) & (top_probs < high)
        n = int(mask.sum())
        rows.append({
            "band_low": round(float(low), 3),
            "band_high": round(float(min(high, 1.0)), 3),
            "n": n,
            "accuracy": float(correct[mask].mean()) if n else None,
            "mean_top_prob": float(top_probs[mask].mean()) if n else None,
        })
    return rows


def expected_calibration_error(
    predictions: pd.DataFrame,
    *,
    bin_edges: tuple[float, ...] = DEFAULT_BIN_EDGES,
) -> dict[str, float]:
    """One-number summary of calibration quality per outcome.

    ECE = Σᵢ (nᵢ / N) · |observedᵢ − predictedᵢ|
    Lower is better. ~0.02-0.04 is excellent, >0.08 is poorly calibrated.
    """
    summary: dict[str, float] = {}
    n_total = len(predictions)
    if n_total == 0:
        return {outcome: 0.0 for outcome in OUTCOMES}
    for outcome in OUTCOMES:
        probs = predictions[outcome].to_numpy(dtype=float)
        actual = (predictions["actual"] == outcome).to_numpy(dtype=int)
        weighted_error = 0.0
        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            if high >= 1.0:
                mask = (probs >= low) & (probs <= high)
            else:
                mask = (probs >= low) & (probs < high)
            n = int(mask.sum())
            if n == 0:
                continue
            weighted_error += (n / n_total) * abs(actual[mask].mean() - probs[mask].mean())
        summary[outcome] = float(weighted_error)
    return summary


def build_diagnostics(predictions: pd.DataFrame) -> dict[str, Any]:
    """One-stop diagnostics bundle for the web UI."""
    if predictions.empty:
        return {
            "calibration_curve": {outcome: [] for outcome in OUTCOMES},
            "confidence_ladder": [],
            "expected_calibration_error": {outcome: 0.0 for outcome in OUTCOMES},
        }
    return {
        "calibration_curve": calibration_curve(predictions),
        "confidence_ladder": confidence_ladder(predictions),
        "expected_calibration_error": expected_calibration_error(predictions),
    }
