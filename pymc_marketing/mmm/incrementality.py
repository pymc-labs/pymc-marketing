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
"""Incrementality and counterfactual analysis for Marketing Mix Models.

This module provides functionality to compute incremental channel contributions
using counterfactual analysis, accounting for adstock carryover effects.

The core approach compares actual performance with counterfactual scenarios
(e.g., zero spend) to measure what incremental value each channel created.
This properly handles adstock carryover where spend at time t affects
outcomes at t, t+1, ..., t+l_max.

Examples
--------
Compute quarterly ROAS with carryover effects:

>>> roas = mmm.incrementality.contribution_over_spend(
...     frequency="quarterly",
...     period_start="2024-01-01",
...     period_end="2024-12-31",
... )

Compute marginal ROAS (return on next dollar):

>>> mroas = mmm.incrementality.marginal_contribution_over_spend(
...     frequency="quarterly",
... )

References
----------
Google MMM Paper: https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf
Section 3.2.2, Formula (10)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import arviz as az
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import xarray as xr
from pytensor import function
from pytensor.graph import vectorize_graph

from pymc_marketing.data.idata.schema import Frequency
from pymc_marketing.mmm.utils import _convert_frequency_to_timedelta
from pymc_marketing.pytensor_utils import extract_response_distribution

if TYPE_CHECKING:
    from numpy.random import Generator, RandomState

    from pymc_marketing.mmm.multidimensional import MMM


class Incrementality:
    """Incrementality and counterfactual analysis for MMM models.

    Computes incremental channel contributions by comparing predictions with
    actual spend vs. counterfactual (perturbed) spend, accounting for
    adstock carryover effects.

    This class uses vectorized graph evaluation to efficiently compute
    incrementality across all channels and posterior samples simultaneously.

    Parameters
    ----------
    model : MMM
        Fitted MMM model instance
    idata : az.InferenceData
        InferenceData containing posterior samples and fit data

    Attributes
    ----------
    model : MMM
        The fitted MMM model
    idata : az.InferenceData
        Posterior samples and fit data
    data : MMMIDataWrapper
        Data wrapper for accessing model data

    Examples
    --------
    Access via model property:

    >>> incr = mmm.incrementality

    Compute quarterly ROAS:

    >>> roas = incr.contribution_over_spend(frequency="quarterly")

    Compute monthly CAC:

    >>> cac = incr.spend_over_contribution(frequency="monthly")
    """

    def __init__(self, model: MMM, idata: az.InferenceData):
        self.model = model
        self.idata = idata
        self.data = model.data

    def compute_incremental_contribution(
        self,
        frequency: Frequency,
        period_start: str | pd.Timestamp | None = None,
        period_end: str | pd.Timestamp | None = None,
        include_carryin: bool = True,
        include_carryout: bool = True,
        original_scale: bool = True,
        num_samples: int | None = None,
        random_state: RandomState | Generator | None = None,
        counterfactual_spend_factor: float = 0.0,
    ) -> xr.DataArray:
        """Compute incremental channel contributions using counterfactual analysis.

        This is the core incrementality function that determines what incremental
        value each channel created during a time period, accounting for adstock
        carryover effects.

        The computation compares two scenarios whose meaning depends on
        ``counterfactual_spend_factor``:

        **Total incrementality** (``counterfactual_spend_factor=0.0``, the default):
            The counterfactual zeroes out channel spend. The resulting incremental
            contribution is the numerator of the **total ROAS**.

        **Marginal incrementality** (e.g., ``counterfactual_spend_factor=1.01``):
            The counterfactual scales spend by a small factor. The resulting
            incremental contribution is the numerator of the **marginal ROAS**.

        Parameters
        ----------
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}
            Time aggregation frequency. "original" uses data's native frequency.
            "all_time" returns single value across entire period.
        period_start : str or pd.Timestamp, optional
            Start date for evaluation window. If None, uses start of fitted data.
        period_end : str or pd.Timestamp, optional
            End date for evaluation window. If None, uses end of fitted data.
        include_carryin : bool, default=True
            Include impact of pre-period channel spend via adstock carryover.
            When True, prepends last l_max observations to capture historical
            effects that carry into the evaluation period.
        include_carryout : bool, default=True
            Include impact of evaluation period spend that carries into post-period.
            When True, extends evaluation window by l_max periods to capture
            trailing adstock effects.
        original_scale : bool, default=True
            Return contributions in original scale of target variable.
        num_samples : int or None, optional
            Number of posterior samples to use. If None, all samples are used.
            If less than total available (chain × draw), a random subset is drawn.
        random_state : RandomState or Generator or None, optional
            Random state for reproducible subsampling.
            Only used when ``num_samples`` is not None.
        counterfactual_spend_factor : float, default=0.0
            Multiplicative factor applied to channel spend in the counterfactual
            scenario.

            - ``0.0`` (default): Zeroes out channel spend → **total** incremental
              contribution (classic on/off counterfactual).
            - ``1.01``: Scales spend to 101% of actual → **marginal** incremental
              contribution (response to 1% spend increase). Used for marginal ROAS.
            - Any value ≥ 0: Supported. Values > 1 measure upside of *more* spend;
              values in (0, 1) measure cost of *less* spend.

        Returns
        -------
        xr.DataArray
            Incremental contributions with dimensions:

            - (sample, date, channel, *custom_dims) when frequency != "all_time"
            - (sample, channel, *custom_dims) when frequency == "all_time"

            For models with hierarchical dimensions like dims=("country",),
            output has shape (sample, date, channel, country).

            **Sign convention**: The result is always ``Y(perturbed) - Y(actual)``
            when ``factor > 1`` and ``Y(actual) - Y(counterfactual)`` when
            ``factor < 1`` (including 0). This means total incrementality
            (factor=0) and marginal incrementality (factor=1.01) are both
            positive for channels with positive effect.

        Raises
        ------
        ValueError
            If frequency is invalid, period dates are outside fitted data range,
            or counterfactual_spend_factor is negative.

        References
        ----------
        Google MMM Paper: https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf
        Section 3.2.2, Formula (10)

        Examples
        --------
        Compute quarterly incremental contributions:

        >>> incremental = mmm.incrementality.compute_incremental_contribution(
        ...     frequency="quarterly",
        ...     period_start="2024-01-01",
        ...     period_end="2024-12-31",
        ... )

        Mean contribution per channel per quarter:

        >>> incremental.mean(dim="sample")
        <xarray.DataArray (date: 4, channel: 3)>
        ...

        Total annual contribution (all_time):

        >>> annual = mmm.incrementality.compute_incremental_contribution(
        ...     frequency="all_time",
        ...     period_start="2024-01-01",
        ...     period_end="2024-12-31",
        ... )

        Quarterly marginal incrementality (1% spend increase):

        >>> marginal = mmm.incrementality.compute_incremental_contribution(
        ...     frequency="quarterly",
        ...     counterfactual_spend_factor=1.01,
        ... )
        """
        # ── 0. Validate inputs ──────────────────────────────────────────────
        if counterfactual_spend_factor < 0:
            raise ValueError(
                f"counterfactual_spend_factor must be >= 0, got {counterfactual_spend_factor}"
            )

        # ── 1. Subsample posterior if needed ────────────────────────────────
        # isel returns a lightweight view, no deep copy
        draw_selection = self._subsample_posterior_draws(num_samples, random_state)

        # ── 2. Extract response distribution (batched over samples) ─────────
        response_graph = extract_response_distribution(
            pymc_model=self.model.model,
            idata=self.idata.isel(draw=draw_selection),
            response_variable="channel_contribution",
        )
        # Shape: (sample, date, channel, *custom_dims)

        # ── 3. Get fit_data and determine period bounds ─────────────────────
        fit_data = self.idata.fit_data
        dates = pd.to_datetime(fit_data.coords["date"].values)

        period_start_ts: pd.Timestamp = (
            dates[0] if period_start is None else pd.to_datetime(period_start)
        )
        period_end_ts: pd.Timestamp = (
            dates[-1] if period_end is None else pd.to_datetime(period_end)
        )

        # ── 4. Create period groups based on frequency ──────────────────────
        periods = self._create_period_groups(period_start_ts, period_end_ts, frequency)
        # Returns: [(t0_1, t1_1), (t0_2, t1_2), ...] or [(t0, t1)] for "all_time"

        # ── 5. Get l_max for carryover calculations ─────────────────────────
        l_max = self.model.adstock.l_max
        inferred_freq: str | None = pd.infer_freq(dates)
        if inferred_freq is None:
            raise ValueError(
                "Could not infer frequency from the date index. "
                "Ensure the fitted data has a regular date frequency."
            )
        freq: str = inferred_freq

        # ── 6. Create batched scenarios for ALL periods at once ─────────────
        all_scenarios = []
        period_labels = []

        for _period_idx, (t0, t1) in enumerate(periods):
            # Determine data window with carryover padding
            if include_carryin:
                data_start = t0 - _convert_frequency_to_timedelta(l_max, freq)
            else:
                data_start = t0

            if include_carryout:
                data_end = t1 + _convert_frequency_to_timedelta(l_max, freq)
            else:
                data_end = t1

            # Extract data window
            fit_window = fit_data.sel(date=slice(data_start, data_end))
            base_data = fit_window[self.model.channel_columns].to_array(dim="channel")
            # Shape: (channel, date_window, *custom_dims)

            # Create counterfactual by applying the spend factor during [t0, t1]
            counterfactual = base_data.copy(deep=True)
            eval_mask = (fit_window.date >= t0) & (fit_window.date <= t1)
            counterfactual[:, eval_mask] = (
                base_data[:, eval_mask] * counterfactual_spend_factor
            )
            # factor=0.0 → zeroes out (total incrementality)
            # factor=1.01 → scales to 101% (marginal incrementality)

            # Stack: [baseline, counterfactual] → 2 scenarios per period
            # Transpose to match channel_data layout: (date, *custom_dims, channel)
            channel_data_dims = ("date", *self.model.dims, "channel")
            all_scenarios.append(base_data.transpose(*channel_data_dims).values)
            all_scenarios.append(counterfactual.transpose(*channel_data_dims).values)

            period_labels.append(t1)  # Period-end date convention

        # ── 7. Batch all scenarios ──────────────────────────────────────────
        n_periods = len(periods)

        # ── 8. Compile vectorized evaluator (once) ──────────────────────────
        # Inlined from SensitivityAnalysis pattern; will extract to
        # counterfactual_core.py in Phase 3
        data_shared = self.model.model["channel_data"]

        # Cast scenario data to match the shared variable's dtype to avoid
        # int32/int64 mismatches (numpy defaults to int64 but pytensor shared
        # variables may use int32).
        scenario_array = np.stack(all_scenarios, axis=0).astype(data_shared.dtype)
        # Shape: (n_periods * 2, date_window, channel, *custom_dims)

        batched_input = pt.tensor(
            name="channel_data_batched",
            dtype=data_shared.dtype,
            shape=(None, *data_shared.type.shape),
        )

        batched_graph = vectorize_graph(
            response_graph, replace={data_shared: batched_input}
        )

        evaluator = function([batched_input], batched_graph)

        # ── 9. Evaluate ALL scenarios at once ───────────────────────────────
        all_predictions = evaluator(scenario_array)
        # Shape: (n_periods * 2, n_samples, date_window, channel, *custom_dims)

        # ── 10. Compute incremental contributions per period ────────────────
        results = []

        for period_idx in range(n_periods):
            baseline_idx = period_idx * 2
            counter_idx = period_idx * 2 + 1

            baseline_pred = all_predictions[
                baseline_idx
            ]  # (n_samples, date_window, channel, *dims)
            counter_pred = all_predictions[counter_idx]

            # Sign convention:
            # factor > 1 → Y(perturbed) - Y(actual)    (marginal: how much more?)
            # factor < 1 → Y(actual) - Y(counterfactual) (total: how much did it create?)
            if counterfactual_spend_factor > 1.0:
                incremental = counter_pred - baseline_pred
            else:
                incremental = baseline_pred - counter_pred
            # Shape: (n_samples, date_window, channel, *custom_dims)

            # Sum over evaluation window (including carryout if enabled)
            t0, t1 = periods[period_idx]
            period_dates = pd.to_datetime(
                fit_data.sel(
                    date=slice(
                        t0 - _convert_frequency_to_timedelta(l_max, freq)
                        if include_carryin
                        else t0,
                        t1 + _convert_frequency_to_timedelta(l_max, freq)
                        if include_carryout
                        else t1,
                    )
                )
                .coords["date"]
                .values
            )

            if include_carryout:
                carryout_end = t1 + _convert_frequency_to_timedelta(l_max, freq)
                eval_dates = (period_dates >= t0) & (period_dates <= carryout_end)
            else:
                eval_dates = (period_dates >= t0) & (period_dates <= t1)

            total_incremental = incremental[:, eval_dates].sum(axis=1)
            # Shape: (n_samples, channel, *custom_dims)

            # Add period coordinate
            period_label = period_labels[period_idx]
            total_incremental_da = xr.DataArray(
                total_incremental,
                dims=("sample", "channel", *self.model.dims),
                coords={
                    "sample": np.arange(total_incremental.shape[0]),
                    "channel": self.model.channel_columns,
                    **{dim: fit_data.coords[dim].values for dim in self.model.dims},
                },
            )
            total_incremental_da = total_incremental_da.assign_coords(
                date=period_label
            ).expand_dims("date")

            results.append(total_incremental_da)

        # ── 11. Concatenate all periods ─────────────────────────────────────
        if frequency == "all_time":
            # Single period, no date dimension
            result = results[0].squeeze("date", drop=True)
        else:
            result = xr.concat(results, dim="date")

        # ── 12. Ensure standard dimension order ─────────────────────────────
        dim_order = ["sample", "date", "channel", *self.model.dims]
        if frequency == "all_time":
            dim_order.remove("date")
        result = result.transpose(*dim_order)

        # ── 13. Scale to original if needed ─────────────────────────────────
        if original_scale:
            target_scale = self.data.get_target_scale()
            result = result * target_scale

        return result

    def _create_period_groups(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        frequency: Frequency,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Create list of (period_start, period_end) tuples for given frequency.

        Parameters
        ----------
        start : pd.Timestamp
            Start of overall date range
        end : pd.Timestamp
            End of overall date range
        frequency : Frequency
            Time aggregation frequency

        Returns
        -------
        list of tuple
            List of (period_start, period_end) pairs. For "all_time", returns
            single tuple. For "original", returns one tuple per date. For other
            frequencies, returns tuples aligned to period boundaries (week-end,
            month-end, etc.).
        """
        if frequency == "all_time":
            return [(start, end)]

        if frequency == "original":
            # One tuple per date in the data's native frequency
            dates = pd.date_range(
                start,
                end,
                freq=pd.infer_freq(self.idata.fit_data.date.values),
            )
            return [(d, d) for d in dates]

        # Map frequency to pandas period code
        freq_map = {
            "weekly": "W",
            "monthly": "M",
            "quarterly": "Q",
            "yearly": "Y",
        }

        dates = pd.date_range(start, end, freq="D")
        periods = dates.to_period(freq_map[frequency])
        unique_periods = periods.unique()

        period_ranges = []
        for period in unique_periods:
            period_start = period.to_timestamp()
            period_end = period.to_timestamp(how="end")  # Period-end date

            # Clip to requested range
            period_start = max(period_start, start)
            period_end = min(period_end, end)

            period_ranges.append((period_start, period_end))

        return period_ranges

    def _subsample_posterior_draws(
        self,
        num_samples: int | None,
        random_state: RandomState | Generator | None,
    ) -> np.ndarray | slice:
        """Get draw indices for posterior subsampling.

        Parameters
        ----------
        num_samples : int or None
            Number of samples to select. If None, returns all draws.
        random_state : RandomState or Generator or None
            Random state for reproducibility

        Returns
        -------
        np.ndarray or slice
            Integer array for isel(draw=...) indexing, or slice(None) for all
        """
        if num_samples is None:
            return slice(None)

        total_draws = self.idata.posterior.dims["draw"]
        if num_samples >= total_draws:
            return slice(None)

        # Use numpy random state
        rng: RandomState | Generator
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, (int, np.integer)):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state

        # Sample without replacement
        indices = rng.choice(total_draws, size=num_samples, replace=False)
        return np.sort(indices)

    def _aggregate_spend(
        self,
        frequency: Frequency,
        period_start: str | pd.Timestamp | None = None,
        period_end: str | pd.Timestamp | None = None,
    ) -> xr.DataArray:
        """Aggregate channel spend by frequency over a date range.

        Delegates to self.data (MMMIDataWrapper) for date filtering and time
        aggregation.

        Parameters
        ----------
        frequency : Frequency
            Time aggregation frequency
        period_start, period_end : str or pd.Timestamp, optional
            Date range. If None, uses full fitted data range.

        Returns
        -------
        xr.DataArray
            Aggregated spend with dims (date, channel, *custom_dims) or
            (channel, *custom_dims) for "all_time"
        """
        # 1. Filter to date range
        data = self.data.filter_dates(period_start, period_end)

        # 2. Aggregate over time (no-op for "original")
        if frequency != "original":
            data = data.aggregate_time(period=frequency, method="sum")

        # 3. Return spend with channel dimension
        return data.get_channel_spend()
