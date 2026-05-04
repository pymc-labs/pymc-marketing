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

The designer uses the MMM's own computational graph to evaluate channel
contributions, so it automatically supports any adstock/saturation
combination without reimplementing the transformation formulas.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
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

_DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "assurance": 0.5,
    "cost_efficiency": 0.5,
}


def _build_eval_fn_from_model(
    model: Any,
    idata: Any,
    n_channels: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a compiled function that evaluates channel contributions.

    Uses ``extract_response_distribution`` to trace the model's graph for
    ``channel_contribution``, replace free RVs with posterior samples,
    and compile a fast evaluation function.

    Parameters
    ----------
    model : pm.Model
        The fitted PyMC model containing a ``channel_contribution`` variable.
    idata : InferenceData
        Posterior samples.
    n_channels : int
        Number of channels.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A function ``f(spend_data) -> contributions`` where:
        - ``spend_data`` has shape ``(T, n_channels)``
        - ``contributions`` has shape ``(n_samples, T, n_channels)``
    """
    import pytensor.xtensor as ptx
    from pytensor.compile.function import function
    from pytensor.graph.replace import clone_replace

    from pymc_marketing.pytensor_utils import extract_response_distribution

    response_graph = extract_response_distribution(
        pymc_model=model,
        idata=idata,
        response_variable="channel_contribution",
    )

    data_var = model["channel_data"]

    dynamic_input = ptx.xtensor(
        name="channel_data_input",
        dtype=str(data_var.dtype),
        shape=(None, n_channels),
        dims=("date", "channel"),
    )

    [new_graph] = clone_replace([response_graph], replace={data_var: dynamic_input})
    compiled = function([dynamic_input], new_graph)
    return compiled


class ExperimentDesigner:
    """Posterior-aware experiment designer for marketing lift tests.

    Consumes a fitted MMM and recommends which experiment to run based on
    posterior uncertainty. Supports national-level experiments analysed via
    Interrupted Time Series (ITS).

    The designer uses the model's own computational graph (via
    ``extract_response_distribution``) to evaluate channel contributions,
    so it works with any adstock/saturation combination automatically.

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

        channel_columns = list(mmm.channel_columns)
        n_channels = len(channel_columns)

        eval_fn = _build_eval_fn_from_model(mmm.model, mmm.idata, n_channels)

        data_var = mmm.model["channel_data"]
        data_values = data_var.get_value()
        n_recent = min(8, data_values.shape[0])
        current_spend_arr = np.mean(data_values[-n_recent:], axis=0)
        current_spend = {
            ch: float(current_spend_arr[i]) for i, ch in enumerate(channel_columns)
        }

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
            eval_fn=eval_fn,
            channel_columns=channel_columns,
            l_max=int(mmm.adstock.l_max),
            normalize=bool(mmm.adstock.normalize),
            posterior=mmm.idata.posterior,
            current_spend=current_spend,
            residual_std=residual_std,
            residual_autocorr=residual_autocorr,
            spend_correlation=spend_correlation,
        )

    def _init_common(
        self,
        *,
        eval_fn: Callable[[np.ndarray], np.ndarray],
        channel_columns: list[str],
        l_max: int,
        normalize: bool,
        posterior: Any,
        current_spend: dict[str, float],
        residual_std: float,
        residual_autocorr: float,
        spend_correlation: pd.DataFrame | None,
    ) -> None:
        """Set all instance attributes from pre-extracted data.

        Both ``__init__`` and ``from_idata`` funnel through this method
        so that new attributes only need to be added in one place.
        """
        self._eval_fn = eval_fn
        self.channel_columns = channel_columns
        self.l_max = l_max
        self.normalize = normalize
        self._posterior = posterior
        self.n_draws: int = posterior.sizes["chain"] * posterior.sizes["draw"]
        self._current_spend = current_spend
        self._residual_std = residual_std
        self._residual_autocorr = residual_autocorr
        self._spend_correlation = spend_correlation

    @staticmethod
    def _compute_residual_std(mmm: Any) -> tuple[float, float]:
        """Compute per-week residual standard deviation and autocorrelation.

        Uses ``mmm.predict(mmm.X)`` rather than posterior predictive sampling
        because (a) ``predict`` is the public API for point predictions,
        (b) it avoids duplicating training data that the model already holds,
        and (c) the residual standard deviation only needs a point estimate
        of the model fit, not the full posterior predictive distribution.

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

        This constructor builds a lightweight PyMC model using the
        specified transformation classes, then compiles a graph-based
        evaluation function — the same approach as the main constructor.

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
        import pymc as pm
        import pymc.dims as pmd

        from pymc_marketing.mmm.components.adstock import GeometricAdstock
        from pymc_marketing.mmm.components.saturation import LogisticSaturation

        _SATURATION_MAP: dict[str, type] = {"logistic": LogisticSaturation}
        _ADSTOCK_MAP: dict[str, type] = {"geometric": GeometricAdstock}

        if saturation not in _SATURATION_MAP:
            raise NotImplementedError(
                f"Saturation '{saturation}' not supported. "
                f"Supported: {list(_SATURATION_MAP)}"
            )
        if adstock not in _ADSTOCK_MAP:
            raise NotImplementedError(
                f"Adstock '{adstock}' not supported. Supported: {list(_ADSTOCK_MAP)}"
            )

        instance = cls.__new__(cls)

        posterior = idata.posterior
        stacked = posterior.stack(sample=("chain", "draw"))

        first_param_var = next(iter(stacked.data_vars))
        if "channel" in stacked[first_param_var].dims:
            channels = list(stacked[first_param_var].coords["channel"].values)
        else:
            channels = list(stacked.coords.get("channel", []))

        cd = idata.constant_data
        l_max = int(cd["l_max"].values)
        normalize = bool(cd["normalize"].values)
        n_channels = len(channels)
        T_dummy = l_max + 4

        adstock_obj = _ADSTOCK_MAP[adstock](l_max=l_max, normalize=normalize)
        saturation_obj = _SATURATION_MAP[saturation]()

        adstock_obj.set_dims_for_all_priors(("channel",))
        saturation_obj.set_dims_for_all_priors(("channel",))

        with pm.Model(
            coords={
                "channel": channels,
                "date": np.arange(T_dummy),
            }
        ) as model:
            channel_data = pmd.Data(
                "channel_data",
                value=np.zeros((T_dummy, n_channels)),
                dims=("date", "channel"),
            )
            adstocked = adstock_obj.apply(x=channel_data, core_dim="date")
            contribution = saturation_obj.apply(x=adstocked, core_dim="date")
            pmd.Deterministic(
                "channel_contribution",
                contribution,
                dims=("date", "channel"),
            )

        eval_fn = _build_eval_fn_from_model(model, idata, n_channels)

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
            eval_fn=eval_fn,
            channel_columns=channels,
            l_max=l_max,
            normalize=normalize,
            posterior=posterior,
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
    # Graph-based evaluation
    # ------------------------------------------------------------------

    def _eval_contributions(self, spend_data: np.ndarray) -> np.ndarray:
        """Evaluate channel contributions using the compiled model graph.

        Parameters
        ----------
        spend_data : np.ndarray
            Spend time series of shape ``(T, n_channels)`` in model scale.

        Returns
        -------
        np.ndarray
            Contributions of shape ``(n_samples, T, n_channels)``.
        """
        return np.asarray(self._eval_fn(spend_data.astype(np.float64)))

    def _make_spend_array(self, T: int) -> np.ndarray:
        """Create a baseline spend array filled with current spend.

        Parameters
        ----------
        T : int
            Time dimension length.

        Returns
        -------
        np.ndarray
            Spend array of shape ``(T, n_channels)``.
        """
        n_channels = len(self.channel_columns)
        spend = np.zeros((T, n_channels))
        for i, ch in enumerate(self.channel_columns):
            spend[:, i] = self._current_spend[ch]
        return spend

    # ------------------------------------------------------------------
    # Ramp fraction helpers (graph-based, adstock-agnostic)
    # ------------------------------------------------------------------

    def _steady_state_per_week_lift(self, channel: str, delta_x: float) -> np.ndarray:
        """Per-draw steady-state per-week lift via the compiled graph.

        Runs a long experiment (``4 * l_max`` treatment weeks) and returns
        the per-week contribution difference at the last time step, where
        adstock has fully built up regardless of the adstock type.

        Parameters
        ----------
        channel : str
            Channel name.
        delta_x : float
            Absolute weekly spend change (model scale).

        Returns
        -------
        np.ndarray
            Shape ``(n_draws,)``.
        """
        ch_idx = self.channel_columns.index(channel)
        T_long = max(4 * self.l_max, 32)
        T_total = self.l_max + T_long

        baseline = self._make_spend_array(T_total)
        treatment = baseline.copy()
        treatment[self.l_max :, ch_idx] += delta_x

        baseline_contrib = self._eval_contributions(baseline)
        treatment_contrib = self._eval_contributions(treatment)
        return treatment_contrib[:, -1, ch_idx] - baseline_contrib[:, -1, ch_idx]

    def _compute_graph_ramp_fraction(
        self,
        predicted_lift: np.ndarray,
        T_active: int,
        ss_per_week: np.ndarray,
    ) -> float:
        """Compute ramp fraction as average per-week lift / steady-state per-week lift.

        Parameters
        ----------
        predicted_lift : np.ndarray
            Total cumulative lift over ``T_active`` weeks, shape ``(n_draws,)``.
        T_active : int
            Experiment duration in weeks.
        ss_per_week : np.ndarray
            Steady-state per-week lift, shape ``(n_draws,)``.

        Returns
        -------
        float
            Ramp fraction in ``[0, 1]``.
        """
        avg_per_week = predicted_lift / T_active
        ramp = avg_per_week / (ss_per_week + 1e-12)
        return float(np.clip(np.mean(ramp), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Lift prediction (graph-based)
    # ------------------------------------------------------------------

    def _predict_lift(
        self,
        channel: str,
        delta_x: float,
        T_active: int,
    ) -> np.ndarray:
        """Predict total cumulative lift for a candidate experiment.

        Constructs baseline and treatment spend time series and evaluates
        them through the model's compiled computational graph. No manual
        reimplementation of saturation or adstock formulas is needed.

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
        ch_idx = self.channel_columns.index(channel)
        T_total = self.l_max + T_active

        baseline = self._make_spend_array(T_total)
        treatment = baseline.copy()
        treatment[self.l_max :, ch_idx] += delta_x

        baseline_contrib = self._eval_contributions(baseline)
        treatment_contrib = self._eval_contributions(treatment)

        lift = (
            treatment_contrib[:, self.l_max :, ch_idx]
            - baseline_contrib[:, self.l_max :, ch_idx]
        ).sum(axis=1)
        return lift

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

    # ------------------------------------------------------------------
    # Channel-level diagnostics (shared by scoring and plotting)
    # ------------------------------------------------------------------

    def _channel_metrics(self) -> dict[str, dict[str, float]]:
        """Compute per-channel diagnostic metrics.

        Returns a dict mapping channel name to a dict with keys:
        ``hdi_width``, ``mean_correlation``, ``gradient``, ``ramp_at_lmax``.
        """
        result: dict[str, dict[str, float]] = {}

        for ch in self.channel_columns:
            hdi_width = self._compute_response_hdi_width(ch)

            mean_corr = 0.0
            if self._spend_correlation is not None:
                corr_df = self._spend_correlation
                others = [
                    abs(float(corr_df.loc[ch, o]))  # type: ignore[arg-type]
                    for o in self.channel_columns
                    if o != ch
                ]
                mean_corr = float(np.mean(others)) if others else 0.0

            grad = self._compute_gradient(ch)

            x_current = self._current_spend[ch]
            small_delta = 0.01 * x_current
            if small_delta > 0:
                ss = self._steady_state_per_week_lift(ch, small_delta)
                lift = self._predict_lift(ch, small_delta, self.l_max)
                ramp = float(np.clip(np.mean((lift / self.l_max) / (ss + 1e-12)), 0, 1))
            else:
                ramp = 0.0

            result[ch] = {
                "hdi_width": hdi_width,
                "mean_correlation": mean_corr,
                "gradient": grad,
                "ramp_at_lmax": ramp,
            }
        return result

    def _compute_response_hdi_width(self, channel: str) -> float:
        """Compute HDI width of channel contribution at current spend."""
        ch_idx = self.channel_columns.index(channel)
        spend = self._make_spend_array(self.l_max + 1)
        contrib = self._eval_contributions(spend)
        values = contrib[:, -1, ch_idx]
        hdi = az.hdi(values, hdi_prob=0.94)
        return float(hdi[1] - hdi[0])

    def _compute_gradient(self, channel: str) -> float:
        """Compute numerical gradient of the response function."""
        ch_idx = self.channel_columns.index(channel)
        x_current = self._current_spend[channel]
        h = max(x_current * 0.01, 1e-6)

        spend_lo = self._make_spend_array(self.l_max + 1)
        spend_hi = spend_lo.copy()
        spend_hi[:, ch_idx] += h

        contrib_lo = self._eval_contributions(spend_lo)
        contrib_hi = self._eval_contributions(spend_hi)

        response_lo = float(np.mean(contrib_lo[:, -1, ch_idx]))
        response_hi = float(np.mean(contrib_hi[:, -1, ch_idx]))
        return (response_hi - response_lo) / h

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
        """Evaluate the candidate grid and return raw metric dicts."""
        candidate_data: list[dict[str, Any]] = []

        for channel in self.channel_columns:
            x_current = self._current_spend[channel]

            for frac in spend_changes:
                delta_x = frac * x_current
                ss_per_week = self._steady_state_per_week_lift(channel, delta_x)

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
                    ramp_frac = self._compute_graph_ramp_fraction(
                        predicted_lift, T, ss_per_week
                    )
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
        deviation.
        """
        null_candidates: list[str] = []
        spend = self._make_spend_array(self.l_max + 1)
        contrib = self._eval_contributions(spend)

        for i, ch in enumerate(self.channel_columns):
            mean_contribution = float(np.mean(np.abs(contrib[:, -1, i])))
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
        gradient at current operating point, and adstock ramp fraction
        at ``l_max`` weeks for each channel.

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
            "Response HDI Width",
            "Mean |Spend Correlation|",
            "Response Gradient\n(at operating point)",
            f"Ramp @ {self.l_max}w\n(fraction of steady state)",
        ]
        value_keys = ["hdi_width", "mean_correlation", "gradient", "ramp_at_lmax"]

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
                ax_ij.set_title(f"\u0394={frac * 100:+.0f}%, T={T}w", fontsize=9)
                if j == 0:
                    ax_ij.set_ylabel("Density")
                if i == n_rows - 1:
                    ax_ij.set_xlabel("Total Lift")

        fig.suptitle(f"Lift Distributions \u2014 {channel}", fontsize=12, y=1.02)
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

        Shows the posterior mean curve with a 94% HDI band, evaluated
        through the model's computational graph.

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

        ch_idx = self.channel_columns.index(channel)
        x_current = self._current_spend[channel]
        x_max = x_current * 2.5
        n_grid = 200
        x_grid = np.linspace(0, x_max, n_grid)

        T_eval = self.l_max + 1
        y_all = np.zeros((self.n_draws, n_grid))
        for g_idx, x_val in enumerate(x_grid):
            spend = self._make_spend_array(T_eval)
            spend[:, ch_idx] = x_val
            contrib = self._eval_contributions(spend)
            y_all[:, g_idx] = contrib[:, -1, ch_idx]

        rng = np.random.default_rng(42)
        idx = rng.choice(self.n_draws, size=min(n_samples, self.n_draws), replace=False)
        y_sub = y_all[idx]

        y_mean = y_sub.mean(axis=0)
        y_lo = np.percentile(y_sub, 3.0, axis=0)
        y_hi = np.percentile(y_sub, 97.0, axis=0)

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

        ax.set_xlabel("Spend (model scale)")
        ax.set_ylabel("Response")
        ax.set_title(f"Saturation Curve \u2014 {channel}")
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
            ch_idx = self.channel_columns.index(ch)
            x_current = self._current_spend[ch]
            small_delta = 0.01 * x_current
            if small_delta <= 0:
                continue

            T_ss = max(max_weeks, 4 * self.l_max)
            T_total = self.l_max + T_ss

            baseline = self._make_spend_array(T_total)
            treatment = baseline.copy()
            treatment[self.l_max :, ch_idx] += small_delta

            baseline_contrib = self._eval_contributions(baseline)
            treatment_contrib = self._eval_contributions(treatment)

            ss_per_week = (
                treatment_contrib[:, -1, ch_idx] - baseline_contrib[:, -1, ch_idx]
            )

            diff = (
                treatment_contrib[:, self.l_max : self.l_max + max_weeks, ch_idx]
                - baseline_contrib[:, self.l_max : self.l_max + max_weeks, ch_idx]
            )
            cumulative_lift = np.cumsum(diff, axis=1)

            avg_per_week = cumulative_lift / weeks[None, :]
            ramp_fracs = avg_per_week / (ss_per_week[:, None] + 1e-12)
            ramp_fracs = np.clip(ramp_fracs, 0, 1)

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
