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

The designer evaluates the fitted MMM's channel contributions directly from
its computational graph using trailing adstock followed by saturation.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import arviz as az
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import norm
from xarray import DataTree

from pymc_marketing.mmm.experiment_design import _plotting
from pymc_marketing.mmm.experiment_design._evaluation import (
    ContributionGraph,
    fitted_residual_statistics,
    validate_observed_mean,
)
from pymc_marketing.mmm.experiment_design.fixture import _validate_transformations
from pymc_marketing.mmm.experiment_design.recommendation import (
    ExperimentRecommendation,
    ExperimentRecommendations,
    _format_rationale,
)
from pymc_marketing.mmm.transformers import ConvMode
from pymc_marketing.names.mmm import Dim, Var

if TYPE_CHECKING:
    from pymc_marketing.mmm.mmm import MMM

_DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "assurance": 0.5,
    "cost_efficiency": 0.5,
}


class ExperimentDesigner:
    """Posterior-aware experiment designer for marketing lift tests.

    Consumes a fitted MMM and recommends which experiment to run based on
    posterior uncertainty. Supports national-level experiments analysed via
    Interrupted Time Series (ITS).

    The designer uses the model's own computational graph (via
    ``extract_response_distribution``) to evaluate channel contributions,
    with trailing adstock applied before saturation.

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
        A fitted national-level MMM with regular weekly observations, an
        identity link, time-invariant media effects, trailing adstock, and
        a Normal, Laplace, or StudentT likelihood with a defined mean. The
        lift calculation covers the model's direct channel contributions.
        Spend changes and costs use the units supplied to the MMM. Lift and
        measurement noise use the MMM's scaled target units.
    spend_lookback : int
        Number of most recent observations used to estimate current spend.
        If the fit has fewer observations, all are used.

    Raises
    ------
    ValueError
        If the MMM has not been fitted, its observations are not weekly, or
        its likelihood has no supported conditional mean equal to ``mu``.
    NotImplementedError
        If the model includes panel dimensions, a non-identity link,
        time-varying media effects, additional ``mu_effects``, non-trailing
        adstock, or ``adstock_first=False``.

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

    def __init__(self, mmm: MMM, *, spend_lookback: int = 8) -> None:
        idata = mmm.idata
        if idata is None:
            raise ValueError(
                "MMM must be fitted (call mmm.fit()) before creating "
                "an ExperimentDesigner."
            )

        if mmm.dims or mmm.link != "identity":
            raise NotImplementedError(
                "ExperimentDesigner requires a national-level MMM with an identity link."
            )
        if mmm.time_varying_media:
            raise NotImplementedError(
                "ExperimentDesigner requires time-invariant media effects."
            )
        if mmm.mu_effects:
            raise NotImplementedError(
                "ExperimentDesigner cannot evaluate additional mu_effects. "
                "Its counterfactual covers direct channel contributions only."
            )
        if mmm.adstock.mode != ConvMode.After:
            raise NotImplementedError(
                "ExperimentDesigner requires trailing adstock (ConvMode.After)."
            )
        if spend_lookback < 1:
            raise ValueError("spend_lookback must be positive.")
        if not mmm.adstock_first:
            raise NotImplementedError(
                "adstock_first=False is not supported in v1 of "
                "ExperimentDesigner. Only adstock_first=True "
                "(adstock applied before saturation) is supported."
            )

        # A reloaded model follows the posterior's coordinate order.
        channel_columns = list(mmm.model.coords[Dim.CHANNEL])
        posterior = idata.posterior.dataset.sel({Dim.CHANNEL: channel_columns})
        validate_observed_mean(mmm, posterior)

        spend = idata.constant_data[Var.CHANNEL_DATA].sel(
            {Dim.CHANNEL: channel_columns}
        )
        current_spend = dict(
            zip(
                channel_columns,
                spend.isel({Dim.DATE: slice(-spend_lookback, None)})
                .mean(Dim.DATE)
                .values.tolist(),
                strict=True,
            )
        )
        spend_correlation = spend.transpose(Dim.DATE, Dim.CHANNEL).to_pandas().corr()
        residual_std, residual_autocorr = fitted_residual_statistics(
            mmm, idata.observed_data[mmm.output_var], posterior
        )
        eval_fn = ContributionGraph(mmm.model, posterior, len(channel_columns))

        self._init_common(
            eval_fn=eval_fn,
            channel_columns=channel_columns,
            l_max=int(mmm.adstock.l_max),
            normalize=bool(mmm.adstock.normalize),
            current_spend=current_spend,
            residual_std=residual_std,
            residual_autocorr=residual_autocorr,
            spend_correlation=spend_correlation,
        )

    def _init_common(
        self,
        *,
        eval_fn: ContributionGraph,
        channel_columns: list[str],
        l_max: int,
        normalize: bool,
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
        self.n_draws = eval_fn.n_draws
        self._current_spend = current_spend
        self._residual_std = residual_std
        self._residual_autocorr = residual_autocorr
        self._spend_correlation = spend_correlation

    def _effective_sigma(self, T: int) -> float:
        """Return cumulative noise using the finite-horizon AR(1) variance.

        The lag-1 estimate is clamped to [0, 0.99] to avoid lowering the
        noise floor from an estimated negative correlation.
        """
        rho = np.clip(self._residual_autocorr, 0.0, 0.99)
        lags = np.arange(1, T)
        variance_factor = T + 2 * np.sum((T - lags) * rho**lags)
        return float(self._residual_std * np.sqrt(variance_factor))

    @classmethod
    def from_idata(
        cls,
        idata: DataTree,
        saturation: str = "logistic",
        adstock: str = "geometric",
    ) -> ExperimentDesigner:
        """Create an ExperimentDesigner from a saved DataTree fixture.

        This constructor builds a lightweight PyMC model using the
        specified transformation classes, then compiles a graph-based
        evaluation function. Spend changes and costs use the fixture's stored
        spend scale, typically normalized spend. Lift and noise use its stored
        target scale. Generate a fixture with the desired ``spend_lookback``
        to change its operating point.

        Parameters
        ----------
        idata : DataTree
            An xarray DataTree containing posterior samples with
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
        _validate_transformations(saturation, adstock)

        instance = cls.__new__(cls)

        posterior = idata.posterior.dataset
        channels = posterior.coords[Dim.CHANNEL].values.tolist()

        cd = idata.constant_data
        l_max = int(cd["l_max"].values)
        normalize = bool(cd["normalize"].values)
        eval_fn = ContributionGraph.from_fixture(posterior, channels, l_max, normalize)

        spend = cd["current_weekly_spend"]
        current_spend = {ch: float(spend.sel(channel=ch).values) for ch in channels}

        if "spend_correlation" in cd:
            spend_correlation: pd.DataFrame | None = (
                cd["spend_correlation"]
                .to_pandas()
                .reindex(index=channels, columns=channels)
            )
        else:
            spend_correlation = None

        instance._init_common(
            eval_fn=eval_fn,
            channel_columns=channels,
            l_max=l_max,
            normalize=normalize,
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

    def _steady_state_per_week_lift(
        self,
        channel: str,
        delta_x: float,
        *,
        baselines: dict[int, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Per-draw steady-state per-week lift via the compiled graph.

        Runs a long experiment (``4 * l_max`` treatment weeks) and returns
        the per-week contribution difference at the last time step, where
        adstock has fully built up regardless of the adstock type.

        Parameters
        ----------
        channel : str
            Channel name.
        delta_x : float
            Absolute weekly spend change in the designer's input units.
        baselines : dict[int, np.ndarray] | None
            Optional call-local cache of baseline contributions by horizon.

        Returns
        -------
        np.ndarray
            Shape ``(n_draws,)``.
        """
        ch_idx = self.channel_columns.index(channel)
        T_long = max(4 * self.l_max, 32)
        return self._lift_path(ch_idx, delta_x, T_long, baselines)[:, -1]

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
        denominator = float(np.mean(ss_per_week))
        if abs(denominator) <= 1e-12:
            return 0.0
        return float(np.clip(np.mean(avg_per_week) / denominator, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Lift prediction (graph-based)
    # ------------------------------------------------------------------

    def _predict_lift(
        self,
        channel: str,
        delta_x: float,
        T_active: int,
        *,
        baselines: dict[int, np.ndarray] | None = None,
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
            Absolute weekly spend change in the designer's input units.
        T_active : int
            Experiment duration in weeks.
        baselines : dict[int, np.ndarray] | None
            Optional call-local cache of baseline contributions by horizon.

        Returns
        -------
        np.ndarray
            Predicted total lift per posterior draw, shape ``(n_draws,)``.
        """
        ch_idx = self.channel_columns.index(channel)
        return self._lift_path(ch_idx, delta_x, T_active, baselines).sum(axis=1)

    def _lift_path(
        self,
        channel_index: int,
        delta_x: float,
        duration: int,
        baselines: dict[int, np.ndarray] | None,
    ) -> np.ndarray:
        total = self.l_max + duration
        spend = self._make_spend_array(total)
        baseline = None if baselines is None else baselines.get(total)
        if baseline is None:
            baseline = self._eval_fn(spend)[:, self.l_max :, :].copy()
            if baselines is not None:
                baselines[total] = baseline

        spend[self.l_max :, channel_index] += delta_x
        treatment = self._eval_fn(spend)[:, self.l_max :, channel_index]
        return treatment - baseline[:, :, channel_index]

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
        if sigma_d == 0:
            return float(
                np.mean(np.where(predicted_lift == 0, significance_level, 1.0))
            )
        ncp = np.abs(predicted_lift) / sigma_d
        per_draw_power = norm.sf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
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
        hdi_widths = self._response_hdi_widths()

        for ch in self.channel_columns:
            mean_corr = 0.0
            if self._spend_correlation is not None:
                others = (
                    self._spend_correlation.loc[ch]
                    .drop(index=ch)
                    .dropna()
                    .abs()
                    .to_numpy()
                )
                mean_corr = float(others.mean()) if others.size else 0.0

            grad = self._compute_gradient(ch)

            x_current = self._current_spend[ch]
            small_delta = 0.01 * x_current
            if small_delta > 0:
                ss = self._steady_state_per_week_lift(ch, small_delta)
                lift = self._predict_lift(ch, small_delta, self.l_max)
                ramp = self._compute_graph_ramp_fraction(lift, self.l_max, ss)
            else:
                ramp = 0.0

            result[ch] = {
                "hdi_width": hdi_widths[ch],
                "mean_correlation": mean_corr,
                "gradient": grad,
                "ramp_at_lmax": ramp,
            }
        return result

    def _response_hdi_widths(self) -> dict[str, float]:
        """Compute current response uncertainty for every channel in one call."""
        contributions = self._eval_fn(self._make_spend_array(self.l_max + 1))
        widths = {}
        for i, channel in enumerate(self.channel_columns):
            lower, upper = az.hdi(contributions[:, -1, i], prob=0.94)
            widths[channel] = float(upper - lower)
        return widths

    def _compute_gradient(self, channel: str) -> float:
        """Compute numerical gradient of the response function."""
        ch_idx = self.channel_columns.index(channel)
        x_current = self._current_spend[channel]
        h = max(x_current * 0.01, 1e-6)

        spend_lo = self._make_spend_array(self.l_max + 1)
        spend_hi = spend_lo.copy()
        spend_hi[:, ch_idx] += h

        contrib_lo = self._eval_fn(spend_lo)
        contrib_hi = self._eval_fn(spend_hi)

        response_lo = float(np.mean(contrib_lo[:, -1, ch_idx]))
        response_hi = float(np.mean(contrib_hi[:, -1, ch_idx]))
        return (response_hi - response_lo) / h

    def _get_uncertainty_ranks(self) -> dict[str, int]:
        """Rank channels by posterior uncertainty (1 = most uncertain)."""
        widths = self._response_hdi_widths()
        ranked = sorted(widths, key=widths.__getitem__, reverse=True)
        return {ch: rank + 1 for rank, ch in enumerate(ranked)}

    def _strongest_spend_correlation(self, channel: str) -> tuple[str, float] | None:
        """Get the other channel and magnitude of the strongest correlation."""
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

        return (max_other, max_corr) if max_other is not None else None

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
        weights = _DEFAULT_SCORE_WEIGHTS if score_weights is None else score_weights
        unknown = weights.keys() - _DEFAULT_SCORE_WEIGHTS.keys()
        if unknown:
            raise ValueError(f"Unknown score weights: {', '.join(sorted(unknown))}.")
        if not all(np.isfinite(w) and w >= 0 for w in weights.values()):
            raise ValueError("Score weights must be finite and nonnegative.")

        total = sum(weights.values())
        if total <= 0 or not np.isfinite(total):
            raise ValueError("Score weights must have a finite, positive sum.")

        scores = np.zeros(len(assurances))
        if not len(scores):
            return scores
        assurance_weight = weights.get("assurance", 0.0)
        if assurance_weight:
            scores += assurance_weight / total * self._min_max_normalize(assurances)
        efficiency_weight = weights.get("cost_efficiency", 0.0)
        if efficiency_weight:
            efficiency = assurances / np.abs(net_costs)
            scores += efficiency_weight / total * self._min_max_normalize(efficiency)

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
        A percentage change on a channel with zero current spend leaves
        the spend unchanged, so the grid skips it along with 0% changes.

        Parameters
        ----------
        spend_changes : list[float] | None
            Fractional per-week spend changes. E.g. ``[0.2, -0.5, -1.0]``
            means +20%, -50%, and go-dark. Values must be finite and at
            least -1. Defaults to
            ``[0.1, 0.2, 0.3, 0.5, -0.2, -0.5, -1.0]``.
        durations : list[int] | None
            Experiment durations in weeks. Defaults to ``[4, 6, 8, 12]``.
        min_snr : float
            Minimum signal-to-noise ratio to include in results.
        significance_level : float
            Significance level for the two-sided power calculation.
        score_weights : dict[str, float] | None
            Weights for ``"assurance"`` and ``"cost_efficiency"``. Custom
            weights replace the defaults, and omitted dimensions receive
            zero weight. If None, the dimensions are weighted equally.

        Returns
        -------
        ExperimentRecommendations
            Recommendations sorted by score (descending).
        """
        if spend_changes is None:
            spend_changes = [0.1, 0.2, 0.3, 0.5, -0.2, -0.5, -1.0]
        if durations is None:
            durations = [4, 6, 8, 12]

        if not all(np.isfinite(change) and change >= -1 for change in spend_changes):
            raise ValueError("Spend changes must be finite and at least -1 (go dark).")
        if any(duration < 1 for duration in durations):
            raise ValueError("Experiment durations must be positive.")
        if not 0 < significance_level < 1:
            raise ValueError("significance_level must be between 0 and 1.")
        candidate_data, best_snr = self._evaluate_candidates(
            spend_changes, durations, min_snr, significance_level
        )
        null_candidates = self._identify_null_candidates()

        assurances = np.array([c["assurance"] for c in candidate_data])
        net_costs = np.array([c["net_cost"] for c in candidate_data])
        scores = self._compute_scores(assurances, net_costs, score_weights)

        if not candidate_data:
            if best_snr is not None:
                warnings.warn(
                    f"No candidates met min_snr={min_snr:g}. "
                    f"Best absolute SNR was {best_snr:.3g}.",
                    stacklevel=2,
                )
            return ExperimentRecommendations(
                [], null_confirmation_candidates=null_candidates
            )

        uncertainty_ranks = self._get_uncertainty_ranks()

        candidates: list[ExperimentRecommendation] = []
        for i, data in enumerate(candidate_data):
            score = float(scores[i])
            rec = ExperimentRecommendation(
                **data,
                score=score,
                rationale="",
            )
            rec.rationale = _format_rationale(
                rec,
                uncertainty_rank=uncertainty_ranks.get(rec.channel),
                correlation=self._strongest_spend_correlation(rec.channel),
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
    ) -> tuple[list[dict[str, Any]], float | None]:
        """Evaluate candidates and track the best absolute SNR before filtering."""
        candidate_data: list[dict[str, Any]] = []
        best_snr: float | None = None
        baselines: dict[int, np.ndarray] = {}

        for channel in self.channel_columns:
            x_current = self._current_spend[channel]

            for frac in spend_changes:
                delta_x = frac * x_current
                if delta_x == 0:
                    continue
                ss_per_week = self._steady_state_per_week_lift(
                    channel, delta_x, baselines=baselines
                )

                for T in durations:
                    predicted_lift = self._predict_lift(
                        channel, delta_x, T, baselines=baselines
                    )
                    expected_lift = float(np.mean(predicted_lift))
                    sigma_d = self._effective_sigma(T)
                    if sigma_d > 0:
                        snr = expected_lift / sigma_d
                    else:
                        snr = (
                            float(np.copysign(np.inf, expected_lift))
                            if expected_lift
                            else 0.0
                        )
                    best_snr = max(best_snr or 0.0, abs(snr))
                    if abs(snr) < min_snr:
                        continue

                    lower, upper = az.hdi(predicted_lift, prob=0.94)
                    lift_hdi = (float(lower), float(upper))
                    assurance = self._compute_assurance(
                        predicted_lift, sigma_d, significance_level
                    )
                    ramp_frac = self._compute_graph_ramp_fraction(
                        predicted_lift, T, ss_per_week
                    )
                    net_cost = delta_x * T

                    candidate_data.append(
                        dict(
                            channel=channel,
                            spend_change_frac=frac,
                            spend_change_abs=delta_x,
                            duration_weeks=T,
                            expected_lift=expected_lift,
                            expected_lift_hdi=lift_hdi,
                            snr=snr,
                            assurance=assurance,
                            adstock_ramp_fraction=ramp_frac,
                            net_cost=net_cost,
                        )
                    )

        return candidate_data, best_snr

    def _identify_null_candidates(self) -> list[str]:
        """Identify channels whose posterior contribution is near zero.

        A channel is flagged when the posterior mean of its absolute
        contribution at current spend is less than one residual standard
        deviation.
        """
        null_candidates: list[str] = []
        spend = self._make_spend_array(self.l_max + 1)
        contrib = self._eval_fn(spend)

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
        return _plotting.plot_channel_diagnostics(
            self.channel_columns,
            self._channel_metrics(),
            self.l_max,
            colors,
            figsize,
        )

    def plot_power_cost(
        self,
        recommendations: ExperimentRecommendations | Sequence[ExperimentRecommendation],
        colors: dict[str, str] | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, Axes]:
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
        return _plotting.plot_power_cost(
            self.channel_columns,
            recommendations,
            colors,
            figsize,
        )

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
        return _plotting.plot_lift_distributions(
            channel,
            self._current_spend[channel],
            self._predict_lift,
            spend_changes,
            durations,
            color,
            figsize,
        )

    def plot_saturation_curve(
        self,
        channel: str,
        spend_levels: list[float] | None = None,
        ax: Axes | None = None,
        color: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot the saturation curve with posterior uncertainty.

        Shows the posterior mean curve with a central 94% interval, evaluated
        through the model's computational graph.

        Parameters
        ----------
        channel : str
            Channel name.
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
        return _plotting.plot_saturation_curve(
            channel,
            self.channel_columns.index(channel),
            self._current_spend[channel],
            self._make_spend_array(self.l_max + 1),
            self._eval_fn.steady_responses,
            spend_levels,
            ax,
            color,
            figsize,
        )

    def plot_adstock_ramp(
        self,
        max_weeks: int = 16,
        colors: dict[str, str] | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, Axes]:
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
        return _plotting.plot_adstock_ramp(
            self.channel_columns,
            self._current_spend,
            self.l_max,
            self._lift_path,
            max_weeks,
            colors,
            figsize,
        )
