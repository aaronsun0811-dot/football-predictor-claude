"""Hierarchical Bayesian Dixon-Coles.

Why we need this on top of the MLE version (`dixon_coles.py`)
-------------------------------------------------------------
The MLE Dixon-Coles estimates each team's attack/defense parameter
*independently* by maximum likelihood. For Tier 1 leagues with ~380
matches per season per league × ~38 matches per team, MLE estimates are
reasonably stable. For Tier 3 leagues (Saudi Pro, J1, MLS, etc.) with
sometimes <300 matches per season and brand-new promoted teams that
played only a handful of games, MLE estimates become noisy — a team
that won its first 5 games gets an absurd attack parameter that the
model then trusts blindly.

The standard fix is hierarchical Bayesian shrinkage: each team's
attack/defense is drawn from a league-level Normal prior whose variance
is itself learned from the data. Small-sample teams get shrunk toward
the league mean (zero); large-sample teams can drift away from it. This
is one of the most well-established improvements to Dixon-Coles in the
literature (Baio & Blangiardo 2010, ghurault/football-prediction in
Stan, etc.) and the route we identified in the project's "real
ROI-moving improvements" shortlist (see ROI commit message in 6696e3e).

Scope of THIS module
--------------------
First iteration is deliberately minimal:
- Poisson factor model with hierarchical attack/defense priors.
- Single home_advantage parameter (no per-team home advantage).
- NO Dixon-Coles low-score τ correction yet. The MLE version has it;
  we can add later once the base model proves itself.
- NO xG blending. The MLE version blends xG into "fake goals"; we
  defer that integration until basic NUTS convergence is stable.
- NO Elo prior. The MLE version uses Elo as a goal-adjustment add-on;
  the whole point of Bayesian is to LEARN those priors from data, so
  we deliberately omit Elo here for a head-to-head comparison.
- Predictions use posterior MEAN as point estimates (could use
  posterior predictive samples for full uncertainty propagation
  later — that's the next iteration).

Cost
----
NUTS sampling for ~20 teams + 40 team params + 4 global params is
small. On a full EPL season (1500 matches × multi-season backlog),
fitting takes ~30-60s with default settings (1000 tune + 1000 draws,
2 chains). Roughly 30-100x slower than MLE, which is why this is
opt-in via a backtest config flag rather than the default.

Imports `pymc` lazily inside `fit()` so the module can be imported
without the dependency installed — useful for testing scaffolding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BayesianDCConfig:
    """Sampler + prior hyperparameters for the Bayesian DC.

    Defaults chosen to match the MLE production setup where possible
    (home_advantage_mu=0.25 mirrors the MLE default of 0.22 with some
    width to absorb league variation).
    """
    # Prior on home advantage. Mean roughly matches MLE default; sigma
    # wide enough to let leagues with extreme HA (e.g. Brazilian
    # Serie A) drift.
    home_advantage_mu: float = 0.25
    home_advantage_sigma: float = 0.15

    # Half-Normal scale on the league-level sigma_attack / sigma_defense
    # hyperparams. A scale of 1.0 is loose; the model will shrink it
    # tighter when there's enough data to support it.
    sigma_attack_scale: float = 1.0
    sigma_defense_scale: float = 1.0

    # Wide-ish intercept prior. Most leagues sit around log(1.4) ≈ 0.34
    # for home expected goals; sigma 1.0 covers anything reasonable.
    intercept_mu: float = 0.0
    intercept_sigma: float = 1.0

    # NUTS sampler config. 1000 tune + 1000 draws is the PyMC default
    # and works for this small problem.
    n_tune: int = 1000
    n_draws: int = 1000
    chains: int = 2
    target_accept: float = 0.9

    random_seed: int = 42
    progressbar: bool = False

    # Cap how many matches we fit on. Walk-forward backtest will keep
    # adding history; past ~3000 matches the marginal accuracy is
    # negligible and the fit time grows linearly. None = no cap.
    max_training_matches: int | None = 3000


class DixonColesBayesianModel:
    """Hierarchical Bayesian fit of the Dixon-Coles Poisson model.

    Usage mirrors MLE DixonColesModel for drop-in comparison in the
    backtest harness:

        m = DixonColesBayesianModel().fit(matches)
        prediction = m.predict_match("Arsenal", "Chelsea")

    The fitted ``posterior`` attribute exposes posterior means for
    every parameter — that's what predict_match uses. The raw
    InferenceData object is in ``trace`` for users that want full
    uncertainty (e.g. posterior-predictive draws downstream).
    """

    def __init__(self, config: BayesianDCConfig | None = None) -> None:
        self.config = config or BayesianDCConfig()
        # Filled by fit():
        self.team_index: dict[str, int] | None = None
        self.team_names: list[str] | None = None
        self.posterior: dict[str, Any] | None = None
        self.trace: Any | None = None  # arviz.InferenceData

    def fit(self, matches: pd.DataFrame) -> "DixonColesBayesianModel":
        """Fit the model. ``matches`` needs columns:
        date, home_team, away_team, home_goals, away_goals.
        Score columns must be non-null integers; rows where they aren't
        get dropped.
        """
        import pymc as pm  # noqa: PLC0415 — lazy import

        # Clean the training frame
        df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)

        if self.config.max_training_matches and len(df) > self.config.max_training_matches:
            # Most-recent matches carry more signal; keep the tail.
            df = df.sort_values("date").tail(self.config.max_training_matches)

        if df.empty:
            raise ValueError("No matches available to fit Bayesian DC.")

        # Build team index
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.team_names = teams
        self.team_index = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        home_idx = df["home_team"].map(self.team_index).values
        away_idx = df["away_team"].map(self.team_index).values
        home_goals = df["home_goals"].values
        away_goals = df["away_goals"].values

        cfg = self.config
        with pm.Model():
            # League-level hyperpriors. HalfNormal because sigma must be > 0.
            sigma_attack = pm.HalfNormal("sigma_attack", sigma=cfg.sigma_attack_scale)
            sigma_defense = pm.HalfNormal("sigma_defense", sigma=cfg.sigma_defense_scale)

            # Team-level parameters drawn from league prior. This is the
            # shrinkage: small-sample teams have their posterior pulled
            # toward 0 (the league mean).
            attack = pm.Normal("attack", mu=0, sigma=sigma_attack, shape=n_teams)
            defense = pm.Normal("defense", mu=0, sigma=sigma_defense, shape=n_teams)

            # Globals
            home_advantage = pm.Normal(
                "home_advantage",
                mu=cfg.home_advantage_mu,
                sigma=cfg.home_advantage_sigma,
            )
            intercept = pm.Normal(
                "intercept", mu=cfg.intercept_mu, sigma=cfg.intercept_sigma
            )

            # Linear predictors on log scale.
            log_lambda_home = (
                intercept + home_advantage + attack[home_idx] - defense[away_idx]
            )
            log_lambda_away = intercept + attack[away_idx] - defense[home_idx]

            # Independent Poisson likelihoods. (Adding the Dixon-Coles
            # τ low-score correction is deferred to a follow-up.)
            pm.Poisson("home_obs", mu=pm.math.exp(log_lambda_home), observed=home_goals)
            pm.Poisson("away_obs", mu=pm.math.exp(log_lambda_away), observed=away_goals)

            self.trace = pm.sample(
                draws=cfg.n_draws,
                tune=cfg.n_tune,
                chains=cfg.chains,
                target_accept=cfg.target_accept,
                random_seed=cfg.random_seed,
                progressbar=cfg.progressbar,
                return_inferencedata=True,
            )

        # Extract posterior means as point estimates for prediction.
        post = self.trace.posterior
        self.posterior = {
            "attack": np.asarray(post["attack"].mean(dim=["chain", "draw"]).values),
            "defense": np.asarray(post["defense"].mean(dim=["chain", "draw"]).values),
            "home_advantage": float(post["home_advantage"].mean()),
            "intercept": float(post["intercept"].mean()),
            "sigma_attack": float(post["sigma_attack"].mean()),
            "sigma_defense": float(post["sigma_defense"].mean()),
            "n_teams": n_teams,
            "n_matches": int(len(df)),
        }
        return self

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        *,
        max_goals: int = 8,
    ) -> Mapping[str, Any] | None:
        """Predict score matrix + outcome probabilities.

        Returns None if either team wasn't seen during training. Callers
        should fall back to the MLE model or to a metadata-only response
        in that case.
        """
        if self.posterior is None or self.team_index is None:
            raise RuntimeError("Model not fitted yet — call .fit() first.")

        from scipy.stats import poisson  # noqa: PLC0415

        h_idx = self.team_index.get(home_team)
        a_idx = self.team_index.get(away_team)
        if h_idx is None or a_idx is None:
            return None

        p = self.posterior
        log_lambda_h = (
            p["intercept"] + p["home_advantage"]
            + p["attack"][h_idx] - p["defense"][a_idx]
        )
        log_lambda_a = (
            p["intercept"] + p["attack"][a_idx] - p["defense"][h_idx]
        )
        lambda_h = float(np.exp(log_lambda_h))
        lambda_a = float(np.exp(log_lambda_a))

        ks = np.arange(max_goals + 1)
        p_h = poisson.pmf(ks, lambda_h)
        p_a = poisson.pmf(ks, lambda_a)
        score_matrix = np.outer(p_h, p_a)
        # Renormalize for the goal-cap truncation.
        score_matrix = score_matrix / score_matrix.sum()

        return {
            "home_team": home_team,
            "away_team": away_team,
            "expected_home_goals": lambda_h,
            "expected_away_goals": lambda_a,
            "score_matrix": score_matrix,
            "home_win": float(np.tril(score_matrix, -1).sum()),
            "draw": float(np.trace(score_matrix)),
            "away_win": float(np.triu(score_matrix, 1).sum()),
        }

    def diagnostic_summary(self) -> dict[str, Any]:
        """Quick health check on the sampler output. Returns max Rhat,
        min ESS, and divergence count — the three numbers that matter
        for "did this fit?"
        """
        if self.trace is None:
            raise RuntimeError("Not fitted.")
        import arviz as az  # noqa: PLC0415

        summary = az.summary(self.trace, var_names=["sigma_attack", "sigma_defense",
                                                     "home_advantage", "intercept"])
        max_rhat = float(summary["r_hat"].max())
        min_ess = float(summary["ess_bulk"].min())
        try:
            n_divergent = int(self.trace.sample_stats["diverging"].sum())
        except (KeyError, AttributeError):
            n_divergent = 0
        return {
            "max_rhat": max_rhat,
            "min_ess_bulk": min_ess,
            "n_divergent_transitions": n_divergent,
            "ok": max_rhat < 1.05 and min_ess > 100 and n_divergent < 10,
        }
