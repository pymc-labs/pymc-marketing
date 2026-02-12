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
r"""Incrementality and counterfactual analysis for Marketing Mix Models.

This module provides functionality to compute **incremental channel
contributions** using counterfactual analysis, properly accounting for
adstock carryover effects.

Concept
-------
Incrementality measures the *causal* impact of a marketing channel by
comparing two scenarios:

1. **Actual**: the model prediction with real spend data.
2. **Counterfactual**: the model prediction with spend removed or perturbed.

The difference between these two predictions is the **incremental
contribution** of that channel.  Because MMMs include adstock
transformations, spend at time *t* affects outcomes at
*t, t + 1, ..., t + l_max*.  A naïve element-wise comparison ignores this
temporal attribution; this module handles it correctly by extending the
evaluation window to capture both carry-in and carry-out effects.

**Total incrementality** (zero-out counterfactual):

.. math::

    \Delta Y_m = \sum_{t=t_0}^{t_1 + L - 1}
        \bigl[\hat{Y}_t(x;\,\Omega)
            - \hat{Y}_t(x^{\text{cf}};\,\Omega)\bigr]

where the counterfactual spend zeroes out only the evaluation period:

.. math::

    x^{\text{cf}}_{s,m} =
    \begin{cases}
        0        & s \in [t_0,\, t_1] \\
        x_{s,m}  & \text{otherwise}
    \end{cases}

**Marginal incrementality** (small perturbation):

.. math::

    \delta Y_m = \sum_{t=t_0}^{t_1 + L - 1}
        \bigl[\hat{Y}_t(\tilde{x};\,\Omega)
            - \hat{Y}_t(x;\,\Omega)\bigr]

where the perturbed spend scales only the evaluation period:

.. math::

    \tilde{x}_{s,m} =
    \begin{cases}
        \alpha\, x_{s,m}  & s \in [t_0,\, t_1] \\
        x_{s,m}           & \text{otherwise}
    \end{cases}

Here *m* is the channel, *x* the spend vector, *L* the adstock window
length (``l_max``), *Ω* the posterior parameter samples, and
*α* the ``counterfactual_spend_factor``.  Spend **outside**
:math:`[t_0, t_1]` is always kept at its actual value so that adstock
carry-in is correctly accounted for.

Incrementality is a **general-purpose building block**.  Dividing
incremental contribution by spend gives **ROAS** (Return on Ad Spend) when
the model's target variable is revenue; taking the reciprocal
(spend / contribution) gives **CAC** (Customer Acquisition Cost) when the
target is customer count.  The same logic applies to any target variable.

Examples
--------
Compute quarterly incremental contributions:

.. code-block:: python

    incremental = mmm.incrementality.compute_incremental_contribution(
        frequency="quarterly",
        period_start="2024-01-01",
        period_end="2024-12-31",
    )

Compute quarterly ROAS (when target variable is revenue):

.. code-block:: python

    roas = mmm.incrementality.contribution_over_spend(
        frequency="quarterly",
        period_start="2024-01-01",
        period_end="2024-12-31",
    )

Compute monthly CAC (when target variable is customer count):

.. code-block:: python

    cac = mmm.incrementality.spend_over_contribution(
        frequency="monthly",
    )

Compute marginal ROAS (return on next dollar):

.. code-block:: python

    mroas = mmm.incrementality.marginal_contribution_over_spend(
        frequency="quarterly",
    )

References
----------
Google MMM Paper: https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf
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

from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper
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
    adstock carryover effects.  See the :mod:`module docstring
    <pymc_marketing.mmm.incrementality>` for the full mathematical
    formulation and design rationale.

    Parameters
    ----------
    model : MMM
        Fitted MMM model instance.
    idata : az.InferenceData
        InferenceData containing posterior samples and fit data.

    Attributes
    ----------
    model : MMM
        The fitted MMM model.
    idata : az.InferenceData
        Posterior samples and fit data.
    data : MMMIDataWrapper
        Data wrapper for accessing model data.

    Examples
    --------
    >>> incr = mmm.incrementality
    >>> roas = incr.contribution_over_spend(frequency="quarterly")
    >>> cac = incr.spend_over_contribution(frequency="monthly")
    """

    def __init__(self, model: MMM, idata: az.InferenceData):
        self.model = model
        self.idata = idata
        self.data = MMMIDataWrapper.from_mmm(model, idata)

    def compute_incremental_contribution(
        self,
        frequency: Frequency,
        period_start: str | pd.Timestamp | None = None,
        period_end: str | pd.Timestamp | None = None,
        include_carryover: bool = True,
        original_scale: bool = True,
        num_samples: int | None = None,
        random_state: RandomState | Generator | None = None,
        counterfactual_spend_factor: float = 0.0,
    ) -> xr.DataArray:
        """Compute incremental channel contributions using counterfactual analysis.

        Core incrementality function.  Compares the model's prediction under
        actual spend with its prediction under a counterfactual spend
        scenario, properly accounting for adstock carryover.  See the
        :mod:`module docstring <pymc_marketing.mmm.incrementality>` for the
        full mathematical formulation.

        Parameters
        ----------
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}
            Time aggregation frequency. ``"original"`` uses data's native
            frequency. ``"all_time"`` returns a single value across the entire
            period.
        period_start : str or pd.Timestamp, optional
            Start date for evaluation window. If None, uses start of fitted data.
        period_end : str or pd.Timestamp, optional
            End date for evaluation window. If None, uses end of fitted data.
        include_carryover : bool, default=True
            Include adstock carryover effects.  When True, prepends ``l_max``
            observations before the period to capture historical effects
            carrying into the evaluation period, and extends the evaluation
            window by ``l_max`` periods to capture trailing adstock effects
            from spend during the period.
        original_scale : bool, default=True
            Return contributions in original scale of the target variable.
        num_samples : int or None, optional
            Number of posterior samples to use. If None, all samples are used.
            If less than total available (chain × draw), a random subset is
            drawn.
        random_state : RandomState or Generator or None, optional
            Random state for reproducible subsampling.
            Only used when ``num_samples`` is not None.
        counterfactual_spend_factor : float, default=0.0
            Multiplicative factor *α* applied to channel spend in the
            counterfactual scenario.

            - ``0.0`` (default): Zeroes out channel spend → **total**
              incremental contribution (classic on/off counterfactual).
            - ``1.01``: Scales spend to 101% of actual → **marginal**
              incremental contribution (response to a 1 % spend increase).
            - Any value ≥ 0: Supported.  Values > 1 measure the upside of
              *more* spend; values in (0, 1) measure the cost of *less* spend.

        Returns
        -------
        xr.DataArray
            Incremental contributions with dimensions:

            - ``(sample, date, channel, *custom_dims)`` when
              ``frequency != "all_time"``
            - ``(sample, channel, *custom_dims)`` when
              ``frequency == "all_time"``

            For models with hierarchical dimensions like ``dims=("country",)``,
            output has shape ``(sample, date, channel, country)``.

            **Sign convention**: The result is always
            ``Y(perturbed) − Y(actual)`` when *α > 1* and
            ``Y(actual) − Y(counterfactual)`` when *α < 1* (including 0).
            Both total and marginal incrementality are therefore positive for
            channels with a positive effect.

        Raises
        ------
        ValueError
            If frequency is invalid, period dates are outside fitted data
            range, or ``counterfactual_spend_factor`` is negative.

        References
        ----------
        Google MMM Paper:
        https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf


        Examples
        --------
        Compute quarterly incremental contributions:

        .. code-block:: python

            incremental = mmm.incrementality.compute_incremental_contribution(
                frequency="quarterly",
                period_start="2024-01-01",
                period_end="2024-12-31",
            )

        Mean contribution per channel per quarter:

        .. code-block:: python

            incremental.mean(dim="sample")

        Total annual contribution (all_time):

        .. code-block:: python

            annual = mmm.incrementality.compute_incremental_contribution(
                frequency="all_time",
                period_start="2024-01-01",
                period_end="2024-12-31",
            )

        Quarterly marginal incrementality (1 % spend increase):

        .. code-block:: python

            marginal = mmm.incrementality.compute_incremental_contribution(
                frequency="quarterly",
                counterfactual_spend_factor=1.01,
            )

        """
        # Validate inputs
        if counterfactual_spend_factor < 0:
            raise ValueError(
                f"counterfactual_spend_factor must be >= 0, got {counterfactual_spend_factor}"
            )

        # Subsample posterior if needed
        # isel returns a lightweight view, no deep copy
        draw_selection = self._subsample_posterior_draws(num_samples, random_state)

        # Extract response distribution (batched over samples)
        response_graph = extract_response_distribution(
            pymc_model=self.model.model,
            idata=self.idata.isel(draw=draw_selection),
            response_variable="channel_contribution",
        )
        # Shape: (sample, date, channel, *custom_dims)

        # Determine period bounds
        dates = self.data.dates

        period_start_ts: pd.Timestamp = (
            dates[0] if period_start is None else pd.to_datetime(period_start)
        )
        period_end_ts: pd.Timestamp = (
            dates[-1] if period_end is None else pd.to_datetime(period_end)
        )

        # Create period groups based on frequency
        periods = self._create_period_groups(period_start_ts, period_end_ts, frequency)
        # Returns: [(t0_1, t1_1), (t0_2, t1_2), ...] or [(t0, t1)] for "all_time"

        # Get l_max for carryover calculations
        l_max = self.model.adstock.l_max
        inferred_freq: str | None = pd.infer_freq(dates)
        if inferred_freq is None:
            raise ValueError(
                "Could not infer frequency from the date index. "
                "Ensure the fitted data has a regular date frequency."
            )
        freq: str = inferred_freq

        # Compile vectorized evaluator (once, reused for both)
        data_shared = self.model.model["channel_data"]
        batched_input = pt.tensor(
            name="channel_data_batched",
            dtype=data_shared.dtype,
            shape=(None, *data_shared.type.shape),
        )
        batched_graph = vectorize_graph(
            response_graph, replace={data_shared: batched_input}
        )
        evaluator = function([batched_input], batched_graph)

        # Evaluate baseline on full dataset (once)
        fit_data = self.idata.fit_data
        baseline_array = self.data.get_channel_spend().values
        baseline_pred = evaluator(baseline_array[np.newaxis].astype(data_shared.dtype))[
            0
        ]
        # Shape: (n_samples, n_dates, *non_date_dims)

        # Determine actual axis ordering from the model's channel_contribution
        # variable. The PyTensor graph preserves the model's dim order, which
        # may have custom dims (e.g. "country") before "channel".
        cc_dims = list(
            self.model.model.named_vars_to_dims.get("channel_contribution", ())
        )
        non_date_dims = [d for d in cc_dims if d != "date"]

        # Build zero-padded counterfactual windows
        # Each counterfactual window covers [t0 - l_max, t1 + l_max] to
        # capture carry-in history (for correct adstock) and carry-out
        # effects.  At dataset boundaries, positions outside the data are
        # zero-padded (= no spend), and all windows are right-padded to a
        # uniform max size so they can be stacked for batched evaluation.
        n_periods = len(periods)
        period_labels = []
        freq_td = _convert_frequency_to_timedelta(1, freq)
        extra_shape = baseline_array.shape[1:]  # (channel, *custom_dims)

        # First pass: collect per-period window metadata
        window_infos: list[dict] = []
        for t0, t1 in periods:
            # Always include l_max carry-in for correct adstock context.
            # Always include l_max carry-out so carryover effects are
            # captured (eval mask controls what gets summed).
            ideal_start = t0 - _convert_frequency_to_timedelta(l_max, freq)
            ideal_end = t1 + _convert_frequency_to_timedelta(l_max, freq)

            # Actual dates from the dataset that fall in the ideal window
            in_window = (dates >= ideal_start) & (dates <= ideal_end)
            actual_dates = dates[in_window]
            n_actual = int(in_window.sum())

            # Left-padding: ideal positions before the first actual date
            # (represents "no spend" before the dataset)
            left_pad = 0
            if n_actual > 0 and actual_dates[0] > ideal_start:
                left_pad = round((actual_dates[0] - ideal_start) / freq_td)

            # Right-padding: ideal positions after the last actual date
            # (represents "no spend" after the dataset)
            right_pad = 0
            if n_actual > 0 and actual_dates[-1] < ideal_end:
                right_pad = round((ideal_end - actual_dates[-1]) / freq_td)

            window_infos.append(
                {
                    "left_pad": left_pad,
                    "right_pad": right_pad,
                    "n_actual": n_actual,
                    "in_window": in_window,
                    "actual_dates": actual_dates,
                }
            )

        max_window = max(
            w["left_pad"] + w["n_actual"] + w["right_pad"] for w in window_infos
        )

        # Second pass: build zero-padded counterfactual arrays
        cf_scenarios: list[np.ndarray] = []
        # Per-period boolean eval masks over the padded window (only True
        # at actual-data positions within the eval date range, ensuring
        # consistency with the baseline eval which also covers only actual
        # dates).
        cf_eval_masks: list[np.ndarray] = []

        for (t0, t1), info in zip(periods, window_infos, strict=True):
            # Allocate zero-padded window
            padded = np.zeros((max_window, *extra_shape), dtype=data_shared.dtype)

            # Place actual data at correct offset
            start_pos = info["left_pad"]
            end_pos = start_pos + info["n_actual"]
            padded[start_pos:end_pos] = baseline_array[info["in_window"]].astype(
                data_shared.dtype
            )

            # Apply counterfactual factor to [t0, t1] only
            if info["n_actual"] > 0:
                target_in_actual = (info["actual_dates"] >= t0) & (
                    info["actual_dates"] <= t1
                )
                target_offsets = np.where(target_in_actual)[0] + start_pos
                padded[target_offsets] = (
                    padded[target_offsets] * counterfactual_spend_factor
                )

            cf_scenarios.append(padded)
            period_labels.append(t1)

            # Eval mask: actual-data positions in [t0, carryout_end]
            cf_mask = np.zeros(max_window, dtype=bool)
            if info["n_actual"] > 0:
                if include_carryover:
                    carryout_end = t1 + _convert_frequency_to_timedelta(l_max, freq)
                    eval_in_actual = (info["actual_dates"] >= t0) & (
                        info["actual_dates"] <= carryout_end
                    )
                else:
                    eval_in_actual = (info["actual_dates"] >= t0) & (
                        info["actual_dates"] <= t1
                    )
                eval_offsets = np.where(eval_in_actual)[0] + start_pos
                cf_mask[eval_offsets] = True
            cf_eval_masks.append(cf_mask)

        # Evaluate all counterfactuals at once
        cf_array = np.stack(cf_scenarios, axis=0)
        # Shape: (n_periods, max_window, channel, *custom_dims)

        cf_predictions = evaluator(cf_array)
        # Shape: (n_periods, n_samples, max_window, channel, *custom_dims)

        # Compute incremental contributions per period
        results = []

        for period_idx in range(n_periods):
            t0, t1 = periods[period_idx]

            # Baseline: sum over eval dates from full-dataset prediction
            if include_carryover:
                carryout_end = t1 + _convert_frequency_to_timedelta(l_max, freq)
                bl_eval_mask = (dates >= t0) & (dates <= carryout_end)
            else:
                bl_eval_mask = (dates >= t0) & (dates <= t1)

            baseline_sum = baseline_pred[:, bl_eval_mask].sum(axis=1)

            # Counterfactual: sum over matching actual-data positions
            cf_mask = cf_eval_masks[period_idx]
            cf_sum = cf_predictions[period_idx][:, cf_mask].sum(axis=1)

            # Sign convention:
            # factor > 1 → Y(perturbed) - Y(actual)    (marginal)
            # factor < 1 → Y(actual) - Y(counterfactual) (total)
            if counterfactual_spend_factor > 1.0:
                total_incremental = cf_sum - baseline_sum
            else:
                total_incremental = baseline_sum - cf_sum
            # Shape: (n_samples, channel, *custom_dims)

            # Add period coordinate
            period_label = period_labels[period_idx]
            total_incremental_da = xr.DataArray(
                total_incremental,
                dims=("sample", *non_date_dims),
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

        # Concatenate all periods
        if frequency == "all_time":
            # Single period, no date dimension
            result = results[0].squeeze("date", drop=True)
        else:
            result = xr.concat(results, dim="date")

        # Ensure standard dimension order
        dim_order = ["sample", "date", "channel", *self.model.dims]
        if frequency == "all_time":
            dim_order.remove("date")
        result = result.transpose(*dim_order)

        # Scale to original if needed
        if original_scale:
            target_scale = self.data.get_target_scale()
            result = result * target_scale

        return result

    def contribution_over_spend(
        self,
        frequency: Frequency,
        period_start: str | pd.Timestamp | None = None,
        period_end: str | pd.Timestamp | None = None,
        include_carryover: bool = True,
        num_samples: int | None = None,
        random_state: RandomState | Generator | None = None,
    ) -> xr.DataArray:
        """Compute incremental contribution per unit of spend.

        Wraps :meth:`compute_incremental_contribution` (with
        ``counterfactual_spend_factor=0``) and divides by total spend.
        The interpretation depends on the model's target variable --
        e.g. **ROAS** when the target is revenue, **customers per dollar**
        when the target is acquisitions.

        Parameters
        ----------
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}
            Time aggregation frequency.
        period_start, period_end : str or pd.Timestamp, optional
            Date range for computation.
        include_carryover : bool, default=True
            Include adstock carryover effects.
        num_samples : int or None, optional
            Number of posterior samples to use. If None, all samples are used.
        random_state : RandomState or Generator or None, optional
            Random state for reproducible subsampling.

        Returns
        -------
        xr.DataArray
            Contribution per unit spend with dimensions
            ``(sample, date, channel, *custom_dims)``.
            Zero spend results in NaN for that channel/period.

        Examples
        --------
        >>> roas = mmm.incrementality.contribution_over_spend(
        ...     frequency="quarterly",
        ...     period_start="2024-01-01",
        ...     period_end="2024-12-31",
        ... )
        """
        incremental = self.compute_incremental_contribution(
            frequency=frequency,
            period_start=period_start,
            period_end=period_end,
            include_carryover=include_carryover,
            original_scale=True,
            num_samples=num_samples,
            random_state=random_state,
            counterfactual_spend_factor=0.0,
        )

        spend = self._aggregate_spend(frequency, period_start, period_end)
        spend_safe = xr.where(spend == 0, np.nan, spend)

        return incremental / spend_safe

    def spend_over_contribution(
        self,
        frequency: Frequency,
        period_start: str | pd.Timestamp | None = None,
        period_end: str | pd.Timestamp | None = None,
        include_carryover: bool = True,
        num_samples: int | None = None,
        random_state: RandomState | Generator | None = None,
    ) -> xr.DataArray:
        """Compute spend per unit of incremental contribution.

        Reciprocal of :meth:`contribution_over_spend`.  The interpretation
        depends on the model's target variable -- e.g. **CAC** (Customer
        Acquisition Cost) when the target is customer count

        Parameters
        ----------
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}
            Time aggregation frequency.
        period_start, period_end : str or pd.Timestamp, optional
            Date range for computation.
        include_carryover : bool, default=True
            Include adstock carryover effects.
        num_samples : int or None, optional
            Number of posterior samples to use. If None, all samples are used.
        random_state : RandomState or Generator or None, optional
            Random state for reproducible subsampling.

        Returns
        -------
        xr.DataArray
            Spend per unit contribution with dimensions
            ``(sample, date, channel, *custom_dims)``.
            Zero contribution results in Inf; zero spend results in NaN.

        Examples
        --------
        >>> cac = mmm.incrementality.spend_over_contribution(
        ...     frequency="monthly",
        ... )
        """
        ratio = self.contribution_over_spend(
            frequency=frequency,
            period_start=period_start,
            period_end=period_end,
            include_carryover=include_carryover,
            num_samples=num_samples,
            random_state=random_state,
        )

        return 1.0 / ratio

    def marginal_contribution_over_spend(
        self,
        frequency: Frequency,
        period_start: str | pd.Timestamp | None = None,
        period_end: str | pd.Timestamp | None = None,
        include_carryover: bool = True,
        num_samples: int | None = None,
        random_state: RandomState | Generator | None = None,
        spend_increase_pct: float = 0.01,
    ) -> xr.DataArray:
        """Compute marginal contribution per additional unit of spend.

        Unlike :meth:`contribution_over_spend` which measures **total**
        efficiency (zero-out counterfactual), this method measures the
        **marginal** efficiency at the current spend level -- i.e. the slope
        of the response curve at the current operating point.  This captures
        diminishing returns: a heavily invested channel may have a low
        marginal efficiency even if its total efficiency is high.  See the
        :mod:`module docstring <pymc_marketing.mmm.incrementality>` for the
        marginal incrementality formula.

        Parameters
        ----------
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}
            Time aggregation frequency.
        period_start, period_end : str or pd.Timestamp, optional
            Date range for computation.
        include_carryover : bool, default=True
            Include adstock carryover effects.
        num_samples : int or None, optional
            Number of posterior samples to use. If None, all samples are used.
        random_state : RandomState or Generator or None, optional
            Random state for reproducible subsampling.
        spend_increase_pct : float, default=0.01
            Fractional spend increase for the perturbation (default 1 %).
            Must be > 0.  Smaller values give a closer approximation to the
            true derivative but may suffer from numerical noise.

        Returns
        -------
        xr.DataArray
            Marginal contribution per unit spend with dimensions
            ``(sample, date, channel, *custom_dims)``.
            Zero spend results in NaN for that channel/period.

        Raises
        ------
        ValueError
            If ``spend_increase_pct <= 0``.

        Examples
        --------
        >>> mroas = mmm.incrementality.marginal_contribution_over_spend(
        ...     frequency="quarterly",
        ...     period_start="2024-01-01",
        ...     period_end="2024-12-31",
        ... )
        """
        if spend_increase_pct <= 0:
            raise ValueError(
                f"spend_increase_pct must be > 0, got {spend_increase_pct}"
            )

        factor = 1.0 + spend_increase_pct

        marginal_contribution = self.compute_incremental_contribution(
            frequency=frequency,
            period_start=period_start,
            period_end=period_end,
            include_carryover=include_carryover,
            original_scale=True,
            num_samples=num_samples,
            random_state=random_state,
            counterfactual_spend_factor=factor,
        )

        spend = self._aggregate_spend(frequency, period_start, period_end)

        # Denominator is the *incremental* spend: pct * total_spend
        incremental_spend = spend_increase_pct * spend
        incremental_spend_safe = xr.where(
            incremental_spend == 0, np.nan, incremental_spend
        )

        return marginal_contribution / incremental_spend_safe

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
            period_end = period.to_timestamp(how="end").normalize()

            # Clip start to the requested range (needed when the user
            # passes a start date inside a period)
            period_start = max(period_start, start)

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
