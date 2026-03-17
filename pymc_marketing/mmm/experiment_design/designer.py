#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""ExperimentDesigner: posterior-aware experiment design for lift tests.

Recommends which marketing experiment to run — which channel, at what spend
level, for how long — based on a fitted MMM's posterior uncertainty about
channel response functions. The v1 scope is national-level experiments on
multiple spend channels, analysed via Interrupted Time Series (ITS).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import norm

from pymc_marketing.mmm.experiment_design.recommendation import (
    ExperimentRecommendation,
    ExperimentRecommendations,
    _format_rationale,
)

if TYPE_CHECKING:
    import matplotlib.axes
    from arviz import InferenceData

_SUPPORTED_SATURATION = {"LogisticSaturation"}
_SUPPORTED_ADSTOCK = {"GeometricAdstock"}

_DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "assurance": 0.5,
    "cost_efficiency": 0.5,
}


class ExperimentDesigner:
    """Posterior-aware experiment designer for marketing lift tests.

    Consumes a fitted MMM and recommends which experiment to run based on
    posterior uncertainty. Supports national-level experiments analysed via
    Interrupted Time Series (ITS).

    .. note::

        Assurance values are conditional on the MMM being reasonably
        well-identified. If a channel's effect is confounded (e.g., spend
        correlates strongly with seasonal demand or another channel), the
        posterior may be confidently wrong, producing misleadingly high
        assurance. Always review model diagnostics (posterior predictive
        checks, prior sensitivity, spend correlation) before acting on
        recommendations. High spend correlation is flagged automatically
        in each recommendation's rationale.

    Parameters
    ----------
    mmm : MMM
        A fitted ``pymc_marketing.mmm.multidimensional.MMM`` instance
        (or the legacy ``pymc_marketing.mmm.MMM``). Must have been fitted
        (``mmm.idata`` is not ``None``).

    Raises
    ------
    ValueError
        If the MMM has not been fitted.
    NotImplementedError
        If ``adstock_first=False`` (not supported in v1).
    NotImplementedError
        If the saturation or adstock type is not supported.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.mmm import MMM
        from pymc_marketing.mmm.experiment_design import ExperimentDesigner

        mmm = MMM(...)
        mmm.fit(X, y)

        designer = ExperimentDesigner(mmm)
        recommendations = designer.recommend(
            spend_changes=[0.1, 0.2, 0.3, -0.5, -1.0],
            durations=[4, 6, 8, 12],
        )
    """

    def __init__(self, mmm: Any) -> None:
        if mmm.idata is None:
            raise ValueError(
                "MMM must be fitted (call mmm.fit()) before creating "
                "an ExperimentDesigner."
            )

        if hasattr(mmm, "adstock_first") and not mmm.adstock_first:
            raise NotImplementedError(
                "adstock_first=False is not supported in v1 of "
                "ExperimentDesigner. Only adstock_first=True "
                "(adstock applied before saturation) is supported."
            )

        sat_type = type(mmm.saturation).__name__
        ads_type = type(mmm.adstock).__name__
        if sat_type not in _SUPPORTED_SATURATION:
            raise NotImplementedError(
                f"Saturation type '{sat_type}' is not supported. "
                f"Supported: {_SUPPORTED_SATURATION}"
            )
        if ads_type not in _SUPPORTED_ADSTOCK:
            raise NotImplementedError(
                f"Adstock type '{ads_type}' is not supported. "
                f"Supported: {_SUPPORTED_ADSTOCK}"
            )

        posterior = mmm.idata.posterior.stack(sample=("chain", "draw"))
        sat_var_map = mmm.saturation.variable_mapping
        ads_var_map = mmm.adstock.variable_mapping

        posterior_samples: dict[str, dict[str, np.ndarray]] = {}
        channel_dim = self._find_channel_dim(posterior, sat_var_map)

        channel_columns = list(mmm.channel_columns)
        for channel in channel_columns:
            sel = {channel_dim: channel} if channel_dim else {}
            posterior_samples[channel] = {
                "lam": posterior[sat_var_map["lam"]]
                .sel(**sel)
                .values.astype(np.float64),
                "beta": posterior[sat_var_map["beta"]]
                .sel(**sel)
                .values.astype(np.float64),
                "alpha": posterior[ads_var_map["alpha"]]
                .sel(**sel)
                .values.astype(np.float64),
            }

        n_recent = min(8, len(mmm.X))
        channel_spend = mmm.X[mmm.channel_columns].tail(n_recent).mean()
        current_spend = {ch: float(channel_spend[ch]) for ch in channel_columns}

        residual_std, residual_autocorr = self._compute_residual_std(mmm)

        try:
            spend_correlation: pd.DataFrame | None = mmm.X[mmm.channel_columns].corr()
        except Exception:
            warnings.warn(
                "Could not compute spend correlation matrix. "
                "Correlation-based diagnostics will be unavailable.",
                stacklevel=2,
            )
            spend_correlation = None

        self._init_common(
            saturation_type=sat_type,
            adstock_type=ads_type,
            channel_columns=channel_columns,
            l_max=int(mmm.adstock.l_max),
            normalize=bool(mmm.adstock.normalize),
            posterior_samples=posterior_samples,
            current_spend=current_spend,
            residual_std=residual_std,
            residual_autocorr=residual_autocorr,
            spend_correlation=spend_correlation,
        )

    def _init_common(
        self,
        *,
        saturation_type: str,
        adstock_type: str,
        channel_columns: list[str],
        l_max: int,
        normalize: bool,
        posterior_samples: dict[str, dict[str, np.ndarray]],
        current_spend: dict[str, float],
        residual_std: float,
        residual_autocorr: float,
        spend_correlation: pd.DataFrame | None,
    ) -> None:
        """Set all instance attributes from pre-extracted data.

        Both ``__init__`` and ``from_idata`` funnel through this method
        so that new attributes only need to be added in one place.
        """
        self._saturation_type = saturation_type
        self._adstock_type = adstock_type
        self.channel_columns = channel_columns
        self.l_max = l_max
        self.normalize = normalize
        self._posterior_samples = posterior_samples
        self.n_draws: int = len(next(iter(posterior_samples.values()))["lam"])
        self._current_spend = current_spend
        self._residual_std = residual_std
        self._residual_autocorr = residual_autocorr
        self._spend_correlation = spend_correlation

    @staticmethod
    def _find_channel_dim(posterior: Any, sat_var_map: dict[str, str]) -> str | None:
        """Identify the channel dimension name in the posterior."""
        first_var = next(iter(sat_var_map.values()))
        if first_var not in posterior:
            return None
        dims = list(posterior[first_var].dims)
        dims.remove("sample")
        if not dims:
            return None
        return dims[0]

    @staticmethod
    def _compute_residual_std(mmm: Any) -> tuple[float, float]:
        """Compute per-week residual standard deviation and autocorrelation.

        Returns
        -------
        tuple[float, float]
            ``(residual_std, residual_autocorr)``.
        """
        try:
            y_pred = mmm.predict(mmm.X)
            y_actual = np.asarray(mmm.y).ravel()
            y_pred_flat = np.asarray(y_pred).ravel()
            if len(y_actual) != len(y_pred_flat):
                y_pred_flat = y_pred_flat[: len(y_actual)]
            residuals = y_actual - y_pred_flat
            std = float(np.std(residuals))
            if len(residuals) > 2:
                autocorr = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
            else:
                autocorr = 0.0
            return std, autocorr
        except Exception:
            warnings.warn(
                "Could not compute residuals from mmm.predict(). "
                "Using default residual_std=1.0.",
                stacklevel=2,
            )
            return 1.0, 0.0

    def _effective_sigma(self, T: int) -> float:
        """Compute measurement noise on the cumulative scale with AR(1) correction.

        When residuals are positively autocorrelated, the IID formula
        ``sigma * sqrt(T)`` underestimates the true cumulative noise.
        The AR(1) correction inflates the variance by ``(1 + rho) / (1 - rho)``
        (the large-T approximation for an AR(1) process).

        Parameters
        ----------
        T : int
            Experiment duration in weeks.

        Returns
        -------
        float
            Corrected cumulative measurement noise.
        """
        rho = np.clip(self._residual_autocorr, 0.0, 0.99)
        correction = (1.0 + rho) / (1.0 - rho) if rho > 0.0 else 1.0
        return self._residual_std * np.sqrt(T * correction)

    @classmethod
    def from_idata(
        cls,
        idata: InferenceData,
        saturation: str = "logistic",
        adstock: str = "geometric",
    ) -> ExperimentDesigner:
        """Create an ExperimentDesigner from a saved InferenceData fixture.

        This constructor is useful for demos and testing when a fitted MMM
        object is not available. The InferenceData must contain posterior
        samples and metadata in ``constant_data``.

        Parameters
        ----------
        idata : InferenceData
            An ArviZ InferenceData containing posterior samples with
            variables ``saturation_lam``, ``saturation_beta``, and
            ``adstock_alpha`` (with a ``channel`` dimension), plus
            ``constant_data`` with ``current_weekly_spend``,
            ``residual_std``, ``l_max``, and ``normalize``.
        saturation : str
            Saturation function type. Currently only ``"logistic"``
            is supported.
        adstock : str
            Adstock function type. Currently only ``"geometric"``
            is supported.

        Returns
        -------
        ExperimentDesigner
            A configured designer ready for ``recommend()``.
        """
        if saturation not in ("logistic",):
            raise NotImplementedError(
                f"Saturation '{saturation}' not supported. Use 'logistic'."
            )
        if adstock not in ("geometric",):
            raise NotImplementedError(
                f"Adstock '{adstock}' not supported. Use 'geometric'."
            )

        instance = cls.__new__(cls)

        posterior = idata.posterior.stack(sample=("chain", "draw"))
        channels = list(posterior["saturation_lam"].coords["channel"].values)

        posterior_samples: dict[str, dict[str, np.ndarray]] = {}
        for channel in channels:
            posterior_samples[channel] = {
                "lam": posterior["saturation_lam"]
                .sel(channel=channel)
                .values.astype(np.float64),
                "beta": posterior["saturation_beta"]
                .sel(channel=channel)
                .values.astype(np.float64),
                "alpha": posterior["adstock_alpha"]
                .sel(channel=channel)
                .values.astype(np.float64),
            }

        cd = idata.constant_data
        spend = cd["current_weekly_spend"]
        current_spend = {ch: float(spend.sel(channel=ch).values) for ch in channels}

        if "spend_correlation" in cd:
            corr_vals = cd["spend_correlation"].values
            spend_correlation: pd.DataFrame | None = pd.DataFrame(
                corr_vals, index=channels, columns=channels
            )
        else:
            spend_correlation = None

        instance._init_common(
            saturation_type="LogisticSaturation",
            adstock_type="GeometricAdstock",
            channel_columns=channels,
            l_max=int(cd["l_max"].values),
            normalize=bool(cd["normalize"].values),
            posterior_samples=posterior_samples,
            current_spend=current_spend,
            residual_std=float(cd["residual_std"].values),
            residual_autocorr=(
                float(cd["residual_autocorr"].values)
                if "residual_autocorr" in cd
                else 0.0
            ),
            spend_correlation=spend_correlation,
        )

        return instance

    # ------------------------------------------------------------------
    # Saturation and adstock helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eval_saturation(
        x: np.ndarray | float,
        lam: np.ndarray | float,
        beta: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate the logistic saturation response in pure numpy.

        Uses the same formula as
        :func:`pymc_marketing.mmm.transformers.logistic_saturation`
        but avoids a PyTensor round-trip so callers can pass raw arrays.
        """
        x = np.asarray(x)
        lam = np.asarray(lam)
        sat = (1.0 - np.exp(-lam * x)) / (1.0 + np.exp(-lam * x))
        return np.asarray(beta) * sat

    def _compute_steady_state_spend(
        self,
        x_current: float,
        alpha_samples: np.ndarray,
    ) -> np.ndarray:
        """Compute steady-state adstocked spend per posterior draw.

        Parameters
        ----------
        x_current : float
            Current raw weekly spend (in model scale).
        alpha_samples : np.ndarray
            Posterior draws for adstock alpha, shape ``(n_draws,)``.

        Returns
        -------
        np.ndarray
            Steady-state adstocked spend, shape ``(n_draws,)``.
        """
        if self.normalize:
            return np.full_like(alpha_samples, x_current)
        else:
            alpha_safe = np.clip(alpha_samples, 1e-6, 1 - 1e-6)
            S = (1.0 - alpha_safe**self.l_max) / (1.0 - alpha_safe)
            return x_current * S

    def _ramp_fractions_matrix(
        self,
        alpha_samples: np.ndarray,
        T_active: int,
    ) -> np.ndarray:
        """Compute per-draw, per-week adstock ramp fractions in [0, 1].

        This is the single source of truth for ramp fraction computation.
        Both ``_compute_ramp_fraction`` and ``plot_adstock_ramp`` delegate
        to this method.

        Parameters
        ----------
        alpha_samples : np.ndarray
            Posterior draws for adstock alpha, shape ``(n_draws,)``.
        T_active : int
            Experiment duration in weeks.

        Returns
        -------
        np.ndarray
            Ramp fractions, shape ``(n_draws, T_active)``, values in [0, 1].
        """
        alpha_safe = np.clip(alpha_samples, 1e-6, 1 - 1e-6)
        t_arr = np.arange(T_active)
        partial_sums = (1.0 - alpha_safe[:, None] ** (t_arr[None, :] + 1)) / (
            1.0 - alpha_safe[:, None]
        )
        S = (1.0 - alpha_safe**self.l_max) / (1.0 - alpha_safe)
        return partial_sums / S[:, None]

    def _compute_adstock_ramp(
        self,
        alpha_samples: np.ndarray,
        T_active: int,
    ) -> np.ndarray:
        """Compute adstock ramp weights for each week of the experiment.

        Parameters
        ----------
        alpha_samples : np.ndarray
            Posterior draws for adstock alpha, shape ``(n_draws,)``.
        T_active : int
            Experiment duration in weeks.

        Returns
        -------
        np.ndarray
            Ramp weights, shape ``(n_draws, T_active)``.
            For normalised adstock these are fractions in [0, 1];
            for unnormalised they are raw partial sums.
        """
        if self.normalize:
            return self._ramp_fractions_matrix(alpha_samples, T_active)

        alpha_safe = np.clip(alpha_samples, 1e-6, 1 - 1e-6)
        t_arr = np.arange(T_active)
        return (1.0 - alpha_safe[:, None] ** (t_arr[None, :] + 1)) / (
            1.0 - alpha_safe[:, None]
        )

    def _predict_lift(
        self,
        channel: str,
        delta_x: float,
        T_active: int,
    ) -> np.ndarray:
        """Predict total cumulative lift for a candidate experiment.

        Parameters
        ----------
        channel : str
            Channel name.
        delta_x : float
            Absolute weekly spend change (in model scale).
        T_active : int
            Experiment duration in weeks.

        Returns
        -------
        np.ndarray
            Predicted total lift per posterior draw, shape ``(n_draws,)``.
        """
        params = self._posterior_samples[channel]
        lam = params["lam"]
        beta = params["beta"]
        alpha = params["alpha"]
        x_current = self._current_spend[channel]
        x_ss = self._compute_steady_state_spend(x_current, alpha)
        ramp = self._compute_adstock_ramp(alpha, T_active)

        effective_spend = x_ss[:, None] + delta_x * ramp
        baseline = self._eval_saturation(x_ss, lam, beta)
        weekly_lift = (
            self._eval_saturation(effective_spend, lam[:, None], beta[:, None])
            - baseline[:, None]
        )
        return weekly_lift.sum(axis=1)

    def _compute_assurance(
        self,
        predicted_lift: np.ndarray,
        sigma_d: float,
        significance_level: float = 0.05,
    ) -> float:
        """Compute posterior-predictive power (Bayesian assurance).

        Parameters
        ----------
        predicted_lift : np.ndarray
            Predicted lift per posterior draw, shape ``(n_draws,)``.
        sigma_d : float
            Measurement noise (cumulative scale).
        significance_level : float
            Significance level for two-sided test.

        Returns
        -------
        float
            Assurance (expected power over the posterior).
        """
        z_alpha = norm.ppf(1.0 - significance_level / 2.0)
        ncp = np.abs(predicted_lift) / sigma_d
        per_draw_power = 1.0 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
        return float(np.mean(per_draw_power))

    def _compute_ramp_fraction(self, alpha_samples: np.ndarray, T_active: int) -> float:
        """Compute the mean adstock ramp fraction over the experiment.

        This is the ratio of time-averaged lift to steady-state lift,
        averaged over posterior draws.
        """
        return float(np.mean(self._ramp_fractions_matrix(alpha_samples, T_active)))

    # ------------------------------------------------------------------
    # Channel-level diagnostics (shared by scoring and plotting)
    # ------------------------------------------------------------------

    def _channel_metrics(self) -> dict[str, dict[str, float]]:
        """Compute per-channel diagnostic metrics.

        Returns a dict mapping channel name to a dict with keys:
        ``hdi_width``, ``mean_correlation``, ``gradient``, ``mean_alpha``.
        """
        result: dict[str, dict[str, float]] = {}
        for ch in self.channel_columns:
            p = self._posterior_samples[ch]

            hdi_lam = az.hdi(p["lam"], hdi_prob=0.94)
            hdi_beta = az.hdi(p["beta"], hdi_prob=0.94)
            lam_w = float(hdi_lam[1] - hdi_lam[0])
            beta_w = float(hdi_beta[1] - hdi_beta[0])

            mean_corr = 0.0
            if self._spend_correlation is not None:
                corr_df = self._spend_correlation
                others = [
                    abs(float(corr_df.loc[ch, o]))  # type: ignore[arg-type]
                    for o in self.channel_columns
                    if o != ch
                ]
                mean_corr = float(np.mean(others)) if others else 0.0

            x_ss = self._current_spend[ch]
            lam_mean = float(np.mean(p["lam"]))
            beta_mean = float(np.mean(p["beta"]))
            h = max(x_ss * 0.01, 1e-6)
            grad = float(
                (
                    self._eval_saturation(x_ss + h, lam_mean, beta_mean)
                    - self._eval_saturation(x_ss, lam_mean, beta_mean)
                )
                / h
            )

            result[ch] = {
                "hdi_width": lam_w * beta_w,
                "mean_correlation": mean_corr,
                "gradient": grad,
                "mean_alpha": float(np.mean(p["alpha"])),
            }
        return result

    def _get_uncertainty_ranks(self) -> dict[str, int]:
        """Rank channels by posterior uncertainty (1 = most uncertain)."""
        metrics = self._channel_metrics()
        ranked = sorted(metrics, key=lambda ch: metrics[ch]["hdi_width"], reverse=True)
        return {ch: rank + 1 for rank, ch in enumerate(ranked)}

    def _get_correlation_info(self, channel: str) -> tuple[str, float] | None:
        """Get correlation description and value for a channel.

        Returns
        -------
        tuple[str, float] | None
            ``(description, max_correlation_value)`` or ``None`` if
            spend correlation data is unavailable.
        """
        if self._spend_correlation is None:
            return None

        max_corr: float = 0.0
        max_other: str | None = None
        corr_df = self._spend_correlation
        for other in self.channel_columns:
            if other == channel:
                continue
            corr = abs(float(corr_df.loc[channel, other]))  # type: ignore[arg-type]
            if corr > max_corr:
                max_corr = corr
                max_other = other

        if max_other is None:
            return None
        desc = f"high spend correlation with {max_other} (r = {max_corr:.2f})"
        return desc, max_corr

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _min_max_normalize(values: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1]. Returns 0.5 if range is zero."""
        vmin, vmax = values.min(), values.max()
        if vmax - vmin < 1e-10:
            return np.full_like(values, 0.5)
        return (values - vmin) / (vmax - vmin)

    def _compute_scores(
        self,
        assurances: np.ndarray,
        net_costs: np.ndarray,
        score_weights: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Compute weighted composite scores for candidates.

        Scoring uses two dimensions: ``assurance`` (will the experiment
        detect the effect?) and ``cost_efficiency`` (assurance per unit
        of spend disruption).
        """
        weights = dict(_DEFAULT_SCORE_WEIGHTS)
        if score_weights is not None:
            weights.update(score_weights)

        known_dims = {"assurance", "cost_efficiency"}
        weights = {k: v for k, v in weights.items() if k in known_dims}

        total_w = sum(weights.values())
        if total_w <= 0:
            return np.zeros(len(assurances))
        weights = {k: v / total_w for k, v in weights.items()}

        scores = np.zeros(len(assurances))
        if weights.get("assurance", 0) > 0:
            scores += weights["assurance"] * self._min_max_normalize(assurances)
        if weights.get("cost_efficiency", 0) > 0:
            abs_cost = np.abs(net_costs)
            cost_eff = np.where(abs_cost > 1e-10, assurances / abs_cost, 0.0)
            scores += weights["cost_efficiency"] * self._min_max_normalize(cost_eff)

        return scores

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        spend_changes: list[float] | None = None,
        durations: list[int] | None = None,
        min_snr: float = 2.0,
        significance_level: float = 0.05,
        score_weights: dict[str, float] | None = None,
    ) -> ExperimentRecommendations:
        """Recommend experiments across all channels.

        Evaluates a grid of candidate designs (channel x spend change x
        duration) and returns a ranked collection of recommendations.

        Parameters
        ----------
        spend_changes : list[float] | None
            Fractional per-week spend changes. E.g. ``[0.2, -0.5, -1.0]``
            means +20%, -50%, and go-dark. Defaults to
            ``[0.1, 0.2, 0.3, 0.5, -0.2, -0.5, -1.0]``.
        durations : list[int] | None
            Experiment durations in weeks. Defaults to ``[4, 6, 8, 12]``.
        min_snr : float
            Minimum signal-to-noise ratio to include in results.
        significance_level : float
            Significance level for the two-sided power calculation.
        score_weights : dict[str, float] | None
            Custom weights for scoring dimensions. By default, experiments
            are ranked by ``"assurance"`` (will the experiment detect the
            effect?) and ``"cost_efficiency"`` (assurance per unit of spend
            disruption).

        Returns
        -------
        ExperimentRecommendations
            Recommendations sorted by score (descending).
        """
        if spend_changes is None:
            spend_changes = [0.1, 0.2, 0.3, 0.5, -0.2, -0.5, -1.0]
        if durations is None:
            durations = [4, 6, 8, 12]

        candidate_data = self._evaluate_candidates(
            spend_changes, durations, min_snr, significance_level
        )
        null_candidates = self._identify_null_candidates()

        if not candidate_data:
            return ExperimentRecommendations(
                [], null_confirmation_candidates=null_candidates
            )

        assurances = np.array([c["assurance"] for c in candidate_data])
        net_costs = np.array([c["net_cost"] for c in candidate_data])
        scores = self._compute_scores(assurances, net_costs, score_weights)

        uncertainty_ranks = self._get_uncertainty_ranks()

        candidates: list[ExperimentRecommendation] = []
        for i, data in enumerate(candidate_data):
            score = float(scores[i])
            corr_result = self._get_correlation_info(data["channel"])
            corr_desc = corr_result[0] if corr_result else None
            corr_value = corr_result[1] if corr_result else None

            rec = ExperimentRecommendation(
                **data,
                score=score,
                rationale="",
            )
            rec.rationale = _format_rationale(
                rec,
                uncertainty_rank=uncertainty_ranks.get(rec.channel),
                correlation_info=corr_desc,
                max_correlation=corr_value,
            )
            candidates.append(rec)

        candidates.sort(key=lambda r: r.score, reverse=True)
        return ExperimentRecommendations(
            candidates, null_confirmation_candidates=null_candidates
        )

    def _evaluate_candidates(
        self,
        spend_changes: list[float],
        durations: list[int],
        min_snr: float,
        significance_level: float,
    ) -> list[dict[str, Any]]:
        """Evaluate the candidate grid and return raw metric dicts.

        Each dict contains all fields needed for an
        ``ExperimentRecommendation`` except ``score`` and ``rationale``.
        """
        candidate_data: list[dict[str, Any]] = []

        for channel in self.channel_columns:
            x_current = self._current_spend[channel]
            alpha_samples = self._posterior_samples[channel]["alpha"]

            for frac in spend_changes:
                delta_x = frac * x_current

                for T in durations:
                    predicted_lift = self._predict_lift(channel, delta_x, T)
                    expected_lift = float(np.mean(predicted_lift))
                    hdi_arr = az.hdi(predicted_lift, hdi_prob=0.94)
                    lift_hdi = (float(hdi_arr[0]), float(hdi_arr[1]))
                    sigma_d = self._effective_sigma(T)
                    snr = expected_lift / sigma_d if sigma_d > 0 else 0.0

                    if abs(snr) < min_snr:
                        continue

                    assurance = self._compute_assurance(
                        predicted_lift, sigma_d, significance_level
                    )
                    ramp_frac = self._compute_ramp_fraction(alpha_samples, T)
                    net_cost = delta_x * T

                    candidate_data.append(
                        {
                            "channel": channel,
                            "spend_change_frac": frac,
                            "spend_change_abs": delta_x,
                            "duration_weeks": T,
                            "expected_lift": expected_lift,
                            "expected_lift_hdi": lift_hdi,
                            "snr": snr,
                            "assurance": assurance,
                            "adstock_ramp_fraction": ramp_frac,
                            "net_cost": net_cost,
                        }
                    )

        return candidate_data

    def _identify_null_candidates(self) -> list[str]:
        """Identify channels whose posterior contribution is near zero.

        A channel is flagged when the posterior mean of its absolute
        contribution at current spend is less than one residual standard
        deviation, indicating the model believes the channel has
        negligible effect.

        Returns
        -------
        list[str]
            Channel names suitable for null-confirmation testing.
        """
        null_candidates: list[str] = []
        for ch in self.channel_columns:
            params = self._posterior_samples[ch]
            x_current = self._current_spend[ch]
            x_ss = self._compute_steady_state_spend(x_current, params["alpha"])
            contribution = np.abs(
                self._eval_saturation(x_ss, params["lam"], params["beta"])
            )
            mean_contribution = float(np.mean(contribution))
            if mean_contribution < self._residual_std:
                null_candidates.append(ch)
        return null_candidates

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_channel_diagnostics(
        self,
        colors: dict[str, str] | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, np.ndarray]:
        """Plot per-channel diagnostic summary.

        Shows posterior HDI width, mean spend correlation, saturation
        gradient at current operating point, and posterior mean adstock
        alpha for each channel.

        Parameters
        ----------
        colors : dict[str, str] | None
            Mapping of channel name to matplotlib color string.
            Falls back to ``C0``, ``C1``, ... when ``None``.
        figsize : tuple[float, float] | None
            ``(width, height)`` for the figure. Defaults to
            ``(10, 4)``.

        Returns
        -------
        tuple[Figure, ndarray of Axes]
        """
        channels = self.channel_columns
        if colors is None:
            colors = {ch: f"C{i}" for i, ch in enumerate(channels)}
        if figsize is None:
            figsize = (10, 4)

        metrics = self._channel_metrics()

        n_metrics = 4
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        x_pos = np.arange(len(channels))
        bar_colors = [colors[ch] for ch in channels]

        titles = [
            "Posterior HDI Width\n(lam × beta)",
            "Mean |Spend Correlation|",
            "Saturation Gradient\n(at operating point)",
            "Adstock α\n(posterior mean)",
        ]
        value_keys = ["hdi_width", "mean_correlation", "gradient", "mean_alpha"]

        for ax_i, title, key in zip(axes, titles, value_keys, strict=True):
            vals = [metrics[ch][key] for ch in channels]
            ax_i.bar(x_pos, vals, color=bar_colors)
            ax_i.set_xticks(x_pos)
            ax_i.set_xticklabels(channels, rotation=45, ha="right")
            ax_i.set_title(title)

        fig.tight_layout()
        return fig, axes

    def plot_power_cost(
        self,
        recommendations: ExperimentRecommendations | Sequence[ExperimentRecommendation],
        colors: dict[str, str] | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, plt.Axes]:
        """Scatter plot of assurance vs. absolute net cost.

        Points are coloured by channel and shaped by spend direction.

        Parameters
        ----------
        recommendations : ExperimentRecommendations | Sequence[ExperimentRecommendation]
            Output of :meth:`recommend`.
        colors : dict[str, str] | None
            Mapping of channel name to matplotlib color string.
            Falls back to ``C0``, ``C1``, ... when ``None``.
        figsize : tuple[float, float] | None
            ``(width, height)`` for the figure. Defaults to
            ``(10, 4)``.

        Returns
        -------
        tuple[Figure, Axes]
        """
        if figsize is None:
            figsize = (10, 4)
        fig, ax = plt.subplots(figsize=figsize)

        if colors is None:
            colors = {ch: f"C{i}" for i, ch in enumerate(self.channel_columns)}
        markers = {
            "increase": "^",
            "decrease": "v",
            "go-dark": "X",
        }

        for rec in recommendations:
            if rec.spend_change_frac == -1.0:
                direction = "go-dark"
            elif rec.spend_change_frac > 0:
                direction = "increase"
            else:
                direction = "decrease"

            ax.scatter(
                abs(rec.net_cost),
                rec.assurance,
                c=colors.get(rec.channel, "C0"),
                marker=markers[direction],
                s=60,
                alpha=0.7,
            )

        for ch, color in colors.items():
            ax.scatter([], [], c=color, label=ch, s=60)
        for name, marker in markers.items():
            ax.scatter([], [], c="gray", marker=marker, label=name, s=60)

        ax.set_xlabel("|Net Cost| (model-scale units)")
        ax.set_ylabel("Assurance (posterior-predictive power)")
        ax.set_title("Experiment Assurance vs. Cost")
        ax.legend(loc="best", fontsize=8)

        fig.tight_layout()
        return fig, ax

    def plot_lift_distributions(
        self,
        channel: str,
        spend_changes: list[float] | None = None,
        durations: list[int] | None = None,
        color: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, np.ndarray]:
        """Grid of lift posterior distributions for one channel.

        Rows = spend changes, columns = durations.

        Parameters
        ----------
        channel : str
            Channel name.
        spend_changes : list[float] | None
            Fractional spend changes. Defaults to
            ``[0.2, 0.5, -0.5, -1.0]``.
        durations : list[int] | None
            Durations in weeks. Defaults to ``[4, 6, 8, 12]``.
        color : str | None
            Matplotlib color for this channel's histograms and HDI
            bands.  Falls back to ``"C0"`` when ``None``.
        figsize : tuple[float, float] | None
            ``(width, height)`` for the figure. Defaults to
            ``(10, 3 * n_rows)``.

        Returns
        -------
        tuple[Figure, ndarray of Axes]
        """
        if spend_changes is None:
            spend_changes = [0.2, 0.5, -0.5, -1.0]
        if durations is None:
            durations = [4, 6, 8, 12]
        if color is None:
            color = "C0"

        n_rows, n_cols = len(spend_changes), len(durations)
        if figsize is None:
            figsize = (10, 3 * n_rows)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize, sharex=False, sharey=False
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        x_current = self._current_spend[channel]

        for i, frac in enumerate(spend_changes):
            for j, T in enumerate(durations):
                ax_ij = axes[i, j]
                delta_x = frac * x_current
                lift = self._predict_lift(channel, delta_x, T)
                hdi = az.hdi(lift, hdi_prob=0.94)

                ax_ij.hist(lift, bins=50, density=True, alpha=0.6, color=color)
                ax_ij.axvline(0, color="black", linestyle="--", linewidth=0.8)
                ax_ij.axvspan(hdi[0], hdi[1], alpha=0.15, color=color)
                ax_ij.set_title(f"Δ={frac * 100:+.0f}%, T={T}w", fontsize=9)
                if j == 0:
                    ax_ij.set_ylabel("Density")
                if i == n_rows - 1:
                    ax_ij.set_xlabel("Total Lift")

        fig.suptitle(f"Lift Distributions — {channel}", fontsize=12, y=1.02)
        fig.tight_layout()
        return fig, axes

    def plot_saturation_curve(
        self,
        channel: str,
        n_samples: int = 500,
        spend_levels: list[float] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        color: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, plt.Axes]:
        """Plot the saturation curve with posterior uncertainty.

        Shows the posterior mean curve with a 94% HDI band.

        Parameters
        ----------
        channel : str
            Channel name.
        n_samples : int
            Number of posterior draws to subsample for the HDI band.
        spend_levels : list[float] | None
            Optional fractional spend levels to mark. E.g. ``[0.2, 0.5]``
            marks +20% and +50% of current spend.
        ax : matplotlib.axes.Axes | None
            Pre-existing axes to draw on. If ``None`` a new figure is created.
        color : str | None
            Matplotlib color for this channel's curve and band.
            Falls back to ``"C0"`` when ``None``.
        figsize : tuple[float, float] | None
            ``(width, height)`` when creating a new figure. Defaults to
            ``(10, 4)``.

        Returns
        -------
        tuple[Figure, Axes]
        """
        if color is None:
            color = "C0"
        if ax is None:
            if figsize is None:
                figsize = (10, 4)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        params = self._posterior_samples[channel]

        x_current = self._current_spend[channel]
        x_max = x_current * 2.5
        n_grid = 200
        x_grid = np.linspace(0, x_max, n_grid)

        rng = np.random.default_rng(42)
        idx = rng.choice(self.n_draws, size=min(n_samples, self.n_draws), replace=False)

        y_all = self._eval_saturation(
            x_grid[None, :], params["lam"][idx, None], params["beta"][idx, None]
        )

        y_mean = y_all.mean(axis=0)
        y_lo = np.percentile(y_all, 3.0, axis=0)
        y_hi = np.percentile(y_all, 97.0, axis=0)

        ax.fill_between(x_grid, y_lo, y_hi, color=color, alpha=0.15)
        ax.plot(x_grid, y_mean, color=color, linewidth=2, label="Posterior mean")

        ax.axvline(
            x_current,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Current spend ({x_current:.2f})",
        )

        if spend_levels:
            for frac in spend_levels:
                x_new = x_current * (1 + frac)
                ax.axvline(
                    x_new,
                    color="C1",
                    linestyle=":",
                    linewidth=1,
                    label=f"+{frac * 100:.0f}%",
                )

        ax.set_xlabel("Adstocked Spend (model scale)")
        ax.set_ylabel("Response")
        ax.set_title(f"Saturation Curve — {channel}")
        ax.legend(fontsize=8)

        fig.tight_layout()
        return fig, ax

    def plot_adstock_ramp(
        self,
        max_weeks: int = 16,
        colors: dict[str, str] | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, plt.Axes]:
        """Plot adstock ramp fraction vs. experiment duration.

        One line per channel showing how quickly the effect approaches
        its steady-state value, with uncertainty bands.

        Parameters
        ----------
        max_weeks : int
            Maximum duration to plot.
        colors : dict[str, str] | None
            Mapping of channel name to matplotlib color string.
            Falls back to ``C0``, ``C1``, ... when ``None``.
        figsize : tuple[float, float] | None
            ``(width, height)`` for the figure. Defaults to
            ``(10, 4)``.

        Returns
        -------
        tuple[Figure, Axes]
        """
        if colors is None:
            colors = {ch: f"C{i}" for i, ch in enumerate(self.channel_columns)}
        if figsize is None:
            figsize = (10, 4)
        fig, ax = plt.subplots(figsize=figsize)
        weeks = np.arange(1, max_weeks + 1)

        for ch in self.channel_columns:
            c = colors[ch]
            alpha = self._posterior_samples[ch]["alpha"]
            ramp_matrix = self._ramp_fractions_matrix(alpha, max_weeks)
            cumsum = np.cumsum(ramp_matrix, axis=1)
            ramp_fracs = cumsum / weeks[None, :]

            mean_ramp = ramp_fracs.mean(axis=0)
            low = np.percentile(ramp_fracs, 5.5, axis=0)
            high = np.percentile(ramp_fracs, 94.5, axis=0)

            ax.plot(weeks, mean_ramp, color=c, label=ch)
            ax.fill_between(weeks, low, high, color=c, alpha=0.15)

        ax.set_xlabel("Experiment Duration (weeks)")
        ax.set_ylabel("Adstock Ramp Fraction")
        ax.set_title("Adstock Ramp-up by Channel")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        return fig, ax
