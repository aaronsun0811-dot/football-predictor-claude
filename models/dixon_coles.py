from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
}


@dataclass(frozen=True)
class DixonColesConfig:
    lookback_days: int = 730
    time_decay_xi: float = 0.00325
    home_advantage: float = 0.22
    fit_home_advantage: bool = True
    max_goals: int = 8
    rho_bounds: tuple[float, float] = (-0.20, 0.20)
    attack_bounds: tuple[float, float] = (-3.0, 3.0)
    defense_bounds: tuple[float, float] = (-3.0, 3.0)
    intercept_bounds: tuple[float, float] = (-3.0, 3.0)
    home_advantage_bounds: tuple[float, float] = (-1.0, 1.0)
    elo_weight: float = 0.10
    elo_scale: float = 400.0
    elo_gap_threshold: float = 180.0
    elo_extreme_weight: float = 0.06
    elo_max_adjustment: float = 0.35
    xg_blend_weight: float = 0.35
    optimizer_maxiter: int = 2000


@dataclass(frozen=True)
class PredictionResult:
    home_team: str
    away_team: str
    home_win: float
    draw: float
    away_win: float
    expected_home_goals: float
    expected_away_goals: float
    score_matrix: np.ndarray
    most_likely_scores: list[dict[str, float | int]]
    neutral_site: bool = False
    knockout: bool = False
    home_advance: float | None = None
    away_advance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "probabilities": {
                "home_win": self.home_win,
                "draw": self.draw,
                "away_win": self.away_win,
            },
            "expected_goals": {
                "home": self.expected_home_goals,
                "away": self.expected_away_goals,
            },
            "most_likely_scores": self.most_likely_scores,
            "score_matrix": self.score_matrix.tolist(),
            "neutral_site": self.neutral_site,
            "knockout": self.knockout,
        }
        if self.knockout:
            payload["advancement_probabilities"] = {
                "home": self.home_advance,
                "away": self.away_advance,
            }
        return payload


class DixonColesModel:
    """Dixon-Coles football score model with Elo-based prior correction."""

    def __init__(self, config: DixonColesConfig | None = None) -> None:
        self.config = config or DixonColesConfig()
        self.teams_: list[str] = []
        self.team_to_idx_: dict[str, int] = {}
        self.attack_: np.ndarray | None = None
        self.defense_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.home_advantage_: float | None = None
        self.rho_: float | None = None
        self.as_of_: pd.Timestamp | None = None
        self.training_rows_: int = 0
        self.xg_training_rows_: int = 0
        self.fit_result_: Any | None = None

    def fit(
        self,
        matches: pd.DataFrame | Sequence[Mapping[str, Any]],
        *,
        as_of: date | datetime | str | pd.Timestamp | None = None,
    ) -> DixonColesModel:
        frame = self._prepare_matches(matches, as_of=as_of)
        self.as_of_ = pd.Timestamp(as_of) if as_of is not None else frame["date"].max()
        self.training_rows_ = len(frame)

        teams = sorted(set(frame["home_team"]) | set(frame["away_team"]))
        if len(teams) < 2:
            raise ValueError("At least two teams are required to fit the model.")

        self.teams_ = teams
        self.team_to_idx_ = {team: idx for idx, team in enumerate(teams)}
        n_teams = len(teams)

        home_idx = frame["home_team"].map(self.team_to_idx_).to_numpy(dtype=np.int64)
        away_idx = frame["away_team"].map(self.team_to_idx_).to_numpy(dtype=np.int64)
        home_goals = frame["home_goals"].to_numpy(dtype=np.int64)
        away_goals = frame["away_goals"].to_numpy(dtype=np.int64)
        home_targets = frame["_home_goal_target"].to_numpy(dtype=float)
        away_targets = frame["_away_goal_target"].to_numpy(dtype=float)
        weights = self._time_decay_weights(frame["date"], self.as_of_)
        elo_adjustments = self._elo_adjustment_vector(frame)

        avg_goals = max((home_targets.sum() + away_targets.sum()) / (2 * len(frame)), 0.20)
        initial = self._initial_params(n_teams, intercept=np.log(avg_goals))
        bounds = self._param_bounds(n_teams)

        def objective(params: np.ndarray) -> float:
            intercept, attack, defense, home_advantage, rho = self._unpack_params(params, n_teams)
            home_log_rate = (
                intercept
                + home_advantage
                + attack[home_idx]
                + defense[away_idx]
                + elo_adjustments
            )
            away_log_rate = (
                intercept
                + attack[away_idx]
                + defense[home_idx]
                - elo_adjustments
            )
            home_rate = np.exp(np.clip(home_log_rate, -8.0, 5.0))
            away_rate = np.exp(np.clip(away_log_rate, -8.0, 5.0))
            tau = self._tau_correction(home_goals, away_goals, home_rate, away_rate, rho)
            tau = np.clip(tau, 1e-12, None)
            ll_home = self._poisson_log_pmf(home_targets, home_rate)
            ll_away = self._poisson_log_pmf(away_targets, away_rate)
            log_likelihood = weights * (ll_home + ll_away + np.log(tau))
            return float(-np.sum(log_likelihood))

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.config.optimizer_maxiter},
        )
        if not result.success:
            # L-BFGS-B has two common "failure" modes: (a) it actually got
            # stuck and returned garbage, or (b) it just hit maxiter while
            # still improving the loss. (b) is harmless for prediction —
            # the parameters at the last iterate are perfectly usable.
            #
            # For very large fits (cross-league pools, big walk-forward
            # backtests) (b) is expected. Tolerate it rather than abort.
            message = str(result.message).upper() if result.message else ""
            iter_limit_hit = (
                "EXCEEDS LIMIT" in message
                or "ITERATION LIMIT" in message
                or "MAXIMUM NUMBER OF ITERATIONS" in message
            )
            if not iter_limit_hit:
                raise RuntimeError(f"Dixon-Coles optimization failed: {result.message}")
            # else: fall through and use result.x as our best estimate.

        intercept, attack, defense, home_advantage, rho = self._unpack_params(result.x, n_teams)
        self.intercept_ = float(intercept)
        self.attack_ = attack
        self.defense_ = defense
        self.home_advantage_ = float(home_advantage)
        self.rho_ = float(rho)
        self.fit_result_ = result
        return self

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        *,
        home_elo: float | None = None,
        away_elo: float | None = None,
        neutral_site: bool = False,
        knockout: bool = False,
        max_goals: int | None = None,
    ) -> PredictionResult:
        self._require_fitted()
        score_cap = max_goals if max_goals is not None else self.config.max_goals
        home_rate, away_rate = self.expected_goals(
            home_team,
            away_team,
            home_elo=home_elo,
            away_elo=away_elo,
            neutral_site=neutral_site,
        )
        matrix = self.score_matrix(home_rate, away_rate, max_goals=score_cap)
        home_win = float(np.tril(matrix, k=-1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, k=1).sum())
        likely_scores = self._most_likely_scores(matrix)

        home_advance: float | None = None
        away_advance: float | None = None
        if knockout:
            # Placeholder for World Cup knockout logic. Until extra-time and
            # penalty models are added, split the 90-minute draw probability.
            home_advance = home_win + draw * 0.5
            away_advance = away_win + draw * 0.5

        return PredictionResult(
            home_team=home_team,
            away_team=away_team,
            home_win=home_win,
            draw=draw,
            away_win=away_win,
            expected_home_goals=float(home_rate),
            expected_away_goals=float(away_rate),
            score_matrix=matrix,
            most_likely_scores=likely_scores,
            neutral_site=neutral_site,
            knockout=knockout,
            home_advance=home_advance,
            away_advance=away_advance,
        )

    def expected_goals(
        self,
        home_team: str,
        away_team: str,
        *,
        home_elo: float | None = None,
        away_elo: float | None = None,
        neutral_site: bool = False,
    ) -> tuple[float, float]:
        self._require_fitted()
        assert self.intercept_ is not None
        assert self.attack_ is not None
        assert self.defense_ is not None
        assert self.home_advantage_ is not None

        home_attack = self._team_value(self.attack_, home_team)
        away_attack = self._team_value(self.attack_, away_team)
        home_defense = self._team_value(self.defense_, home_team)
        away_defense = self._team_value(self.defense_, away_team)
        home_advantage = 0.0 if neutral_site else self.home_advantage_
        elo_adjustment = self.elo_log_goal_adjustment(home_elo, away_elo)

        home_log_rate = (
            self.intercept_
            + home_advantage
            + home_attack
            + away_defense
            + elo_adjustment
        )
        away_log_rate = (
            self.intercept_
            + away_attack
            + home_defense
            - elo_adjustment
        )
        return (
            float(np.exp(np.clip(home_log_rate, -8.0, 5.0))),
            float(np.exp(np.clip(away_log_rate, -8.0, 5.0))),
        )

    def score_matrix(
        self,
        expected_home_goals: float,
        expected_away_goals: float,
        *,
        max_goals: int | None = None,
    ) -> np.ndarray:
        self._require_fitted()
        assert self.rho_ is not None
        score_cap = max_goals if max_goals is not None else self.config.max_goals
        goals = np.arange(score_cap + 1, dtype=np.int64)
        home_pmf = np.exp(self._poisson_log_pmf(goals, expected_home_goals))
        away_pmf = np.exp(self._poisson_log_pmf(goals, expected_away_goals))
        matrix = np.outer(home_pmf, away_pmf)

        home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")
        tau = self._tau_correction(
            home_grid,
            away_grid,
            expected_home_goals,
            expected_away_goals,
            self.rho_,
        )
        matrix = matrix * np.clip(tau, 1e-12, None)
        matrix = matrix / matrix.sum()
        return matrix

    def score_matrix_frame(
        self,
        result_or_matrix: PredictionResult | np.ndarray,
    ) -> pd.DataFrame:
        matrix = (
            result_or_matrix.score_matrix
            if isinstance(result_or_matrix, PredictionResult)
            else result_or_matrix
        )
        index = [f"H{goal}" for goal in range(matrix.shape[0])]
        columns = [f"A{goal}" for goal in range(matrix.shape[1])]
        return pd.DataFrame(matrix, index=index, columns=columns)

    def elo_log_goal_adjustment(
        self,
        home_elo: float | None,
        away_elo: float | None,
    ) -> float:
        if home_elo is None or away_elo is None:
            return 0.0

        diff = float(home_elo) - float(away_elo)
        sign = 1.0 if diff >= 0 else -1.0
        abs_diff = abs(diff)
        base = self.config.elo_weight * diff / self.config.elo_scale

        if abs_diff <= self.config.elo_gap_threshold:
            correction = base
        else:
            excess_ratio = (abs_diff - self.config.elo_gap_threshold) / max(
                self.config.elo_gap_threshold,
                1.0,
            )
            extreme_bonus = (
                sign
                * self.config.elo_extreme_weight
                * np.log1p(excess_ratio)
            )
            correction = base + extreme_bonus

        return float(
            np.clip(
                correction,
                -self.config.elo_max_adjustment,
                self.config.elo_max_adjustment,
            )
        )

    def _prepare_matches(
        self,
        matches: pd.DataFrame | Sequence[Mapping[str, Any]],
        *,
        as_of: date | datetime | str | pd.Timestamp | None,
    ) -> pd.DataFrame:
        frame = pd.DataFrame(matches).copy()
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required match columns: {sorted(missing)}")

        frame = frame.dropna(subset=list(REQUIRED_COLUMNS))
        frame["date"] = pd.to_datetime(frame["date"], utc=False)
        frame["home_team"] = frame["home_team"].astype(str)
        frame["away_team"] = frame["away_team"].astype(str)
        frame["home_goals"] = frame["home_goals"].astype(int)
        frame["away_goals"] = frame["away_goals"].astype(int)
        frame["_home_goal_target"] = frame["home_goals"].astype(float)
        frame["_away_goal_target"] = frame["away_goals"].astype(float)
        frame["_has_xg"] = False

        if self.config.xg_blend_weight > 0 and {"home_xg", "away_xg"}.issubset(frame.columns):
            weight = float(np.clip(self.config.xg_blend_weight, 0.0, 1.0))
            home_xg = pd.to_numeric(frame["home_xg"], errors="coerce")
            away_xg = pd.to_numeric(frame["away_xg"], errors="coerce")
            has_xg = home_xg.notna() & away_xg.notna()
            frame.loc[has_xg, "_home_goal_target"] = (
                (1.0 - weight) * frame.loc[has_xg, "home_goals"].astype(float)
                + weight * home_xg.loc[has_xg].astype(float)
            )
            frame.loc[has_xg, "_away_goal_target"] = (
                (1.0 - weight) * frame.loc[has_xg, "away_goals"].astype(float)
                + weight * away_xg.loc[has_xg].astype(float)
            )
            frame.loc[has_xg, "_has_xg"] = True

        cutoff_as_of = pd.Timestamp(as_of) if as_of is not None else frame["date"].max()
        cutoff_start = cutoff_as_of - timedelta(days=self.config.lookback_days)
        frame = frame[(frame["date"] >= cutoff_start) & (frame["date"] <= cutoff_as_of)]
        frame = frame.sort_values("date").reset_index(drop=True)
        if frame.empty:
            raise ValueError("No matches available after lookback filtering.")
        self.xg_training_rows_ = int(frame["_has_xg"].sum())
        return frame

    def _initial_params(self, n_teams: int, *, intercept: float) -> np.ndarray:
        attack = np.zeros(n_teams, dtype=float)
        defense = np.zeros(n_teams, dtype=float)
        parts = [np.array([intercept]), attack, defense]
        if self.config.fit_home_advantage:
            parts.append(np.array([self.config.home_advantage]))
        parts.append(np.array([-0.05]))
        return np.concatenate(parts)

    def _param_bounds(self, n_teams: int) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = [self.config.intercept_bounds]
        bounds.extend([self.config.attack_bounds] * n_teams)
        bounds.extend([self.config.defense_bounds] * n_teams)
        if self.config.fit_home_advantage:
            bounds.append(self.config.home_advantage_bounds)
        bounds.append(self.config.rho_bounds)
        return bounds

    def _unpack_params(
        self,
        params: np.ndarray,
        n_teams: int,
    ) -> tuple[float, np.ndarray, np.ndarray, float, float]:
        cursor = 0
        intercept = float(params[cursor])
        cursor += 1
        attack = params[cursor : cursor + n_teams]
        cursor += n_teams
        defense = params[cursor : cursor + n_teams]
        cursor += n_teams

        attack = attack - attack.mean()
        defense = defense - defense.mean()

        if self.config.fit_home_advantage:
            home_advantage = float(params[cursor])
            cursor += 1
        else:
            home_advantage = float(self.config.home_advantage)

        rho = float(params[cursor])
        return intercept, attack, defense, home_advantage, rho

    def _time_decay_weights(
        self,
        dates: pd.Series,
        as_of: pd.Timestamp,
    ) -> np.ndarray:
        age_days = (as_of - dates).dt.days.clip(lower=0).to_numpy(dtype=float)
        return np.exp(-self.config.time_decay_xi * age_days)

    def _elo_adjustment_vector(self, frame: pd.DataFrame) -> np.ndarray:
        if "home_elo" not in frame.columns or "away_elo" not in frame.columns:
            return np.zeros(len(frame), dtype=float)
        return np.array(
            [
                self.elo_log_goal_adjustment(row.home_elo, row.away_elo)
                for row in frame[["home_elo", "away_elo"]].itertuples(index=False)
            ],
            dtype=float,
        )

    def _team_value(self, values: np.ndarray, team: str) -> float:
        idx = self.team_to_idx_.get(team)
        if idx is None:
            return 0.0
        return float(values[idx])

    @staticmethod
    def _poisson_log_pmf(goals: np.ndarray | int, rate: np.ndarray | float) -> np.ndarray:
        goals_array = np.asarray(goals, dtype=float)
        rate_array = np.asarray(rate, dtype=float)
        rate_array = np.clip(rate_array, 1e-12, None)
        return goals_array * np.log(rate_array) - rate_array - gammaln(goals_array + 1.0)

    @staticmethod
    def _tau_correction(
        home_goals: np.ndarray | int,
        away_goals: np.ndarray | int,
        home_rate: np.ndarray | float,
        away_rate: np.ndarray | float,
        rho: float,
    ) -> np.ndarray:
        hg, ag, hr, ar = np.broadcast_arrays(
            np.asarray(home_goals),
            np.asarray(away_goals),
            np.asarray(home_rate, dtype=float),
            np.asarray(away_rate, dtype=float),
        )
        tau = np.ones(hg.shape, dtype=float)

        mask_00 = (hg == 0) & (ag == 0)
        mask_01 = (hg == 0) & (ag == 1)
        mask_10 = (hg == 1) & (ag == 0)
        mask_11 = (hg == 1) & (ag == 1)

        tau[mask_00] = 1.0 - hr[mask_00] * ar[mask_00] * rho
        tau[mask_01] = 1.0 + hr[mask_01] * rho
        tau[mask_10] = 1.0 + ar[mask_10] * rho
        tau[mask_11] = 1.0 - rho
        return tau

    def _most_likely_scores(
        self,
        matrix: np.ndarray,
        *,
        top_n: int = 5,
    ) -> list[dict[str, float | int]]:
        flat_indices = np.argsort(matrix.ravel())[::-1][:top_n]
        scores: list[dict[str, float | int]] = []
        for flat_idx in flat_indices:
            home_goals, away_goals = np.unravel_index(flat_idx, matrix.shape)
            scores.append(
                {
                    "home_goals": int(home_goals),
                    "away_goals": int(away_goals),
                    "probability": float(matrix[home_goals, away_goals]),
                }
            )
        return scores

    def _require_fitted(self) -> None:
        if (
            self.attack_ is None
            or self.defense_ is None
            or self.intercept_ is None
            or self.home_advantage_ is None
            or self.rho_ is None
        ):
            raise RuntimeError("DixonColesModel must be fitted before prediction.")
