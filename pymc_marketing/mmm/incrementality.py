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
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

Compute quarterly ROAS (when target variable is revenue):

.. code-block:: python

    roas = mmm.incrementality.contribution_over_spend(
        frequency="quarterly",
        start_date="2024-01-01",
        end_date="2024-12-31",
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
from pandas.tseries.offsets import BaseOffset
from pydantic import ConfigDict, validate_call
from pytensor import function
from pytensor.graph import vectorize_graph

from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper
from pymc_marketing.data.idata.schema import Frequency
from pymc_marketing.data.idata.utils import subsample_draws
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
    idata : az.InferenceData, optional
        InferenceData containing posterior samples and fit data.
        Mutually exclusive with ``data``.
    data : MMMIDataWrapper, optional
        Pre-built data wrapper. When provided, ``idata`` is taken from
        ``data.idata`` and no re-wrapping occurs.
        Mutually exclusive with ``idata``.

    Attributes
    ----------
    model : MMM
        The fitted MMM model.
    idata : az.InferenceData
        Posterior samples and fit data.
    data : MMMIDataWrapper
        Data wrapper for accessing model data.

    Raises
    ------
    ValueError
        If both ``idata`` and ``data`` are provided, or neither is.

    Examples
    --------
    >>> incr = mmm.incrementality
    >>> roas = incr.contribution_over_spend(frequency="quarterly")
    >>> cac = incr.spend_over_contribution(frequency="monthly")
    """

    def __init__(
        self,
        model: MMM,
        idata: az.InferenceData | None = None,
        data: MMMIDataWrapper | None = None,
    ):
        if idata is not None and data is not None:
            raise ValueError("Provide either 'idata' or 'data', not both.")
        if idata is None and data is None:
            raise ValueError("Provide either 'idata' or 'data'.")

        self.model = model
        if data is not None:
            self.data = data
            self.idata = data.idata
        else:
            self.idata = idata
            self.data = MMMIDataWrapper.from_mmm(model, idata)

        # Validate that idata dimensions/coordinates match the model
        in_model_not_idata, in_idata_not_model = self.data.compare_coords(model)
        if in_idata_not_model:
            dim_name = next(iter(in_idata_not_model))
            raise ValueError(
                f"The idata contains unknown coordinate values for dimension "
                f"'{dim_name}': {sorted(in_idata_not_model[dim_name])}. "
                "The model's saturation and adstock parameters are fitted per "
                "dimension value, so incrementality cannot be computed on "
                "aggregated data. Compute incrementality on the original data "
                "first, then aggregate the results."
            )
        if in_model_not_idata:
            dim_name = next(iter(in_model_not_idata))
            channel_spend = self.data.get_channel_spend()
            if dim_name not in channel_spend.dims:
                raise ValueError(
                    f"The idata is missing dimension '{dim_name}' expected by "
                    f"the model. Use a list value in filter_dims to preserve "
                    f'the dimension (e.g. {dim_name}=["US"]).'
                )

    def compute_incremental_contribution(
        self,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        include_carryover: bool = True,
        num_samples: int | None = None,
        random_state: RandomState | Generator | None = None,
        counterfactual_spend_factor: float = 0.0,
    ) -> xr.DataArray:
        """Compute incremental channel contributions using counterfactual analysis.

        Core incrementality function.  Compares the model's prediction under
        actual spend with its prediction under a counterfactual spend
        scenario, properly accounting for adstock carryover.  Results are
        always returned in the original scale of the target variable.
        See the :mod:`module docstring <pymc_marketing.mmm.incrementality>`
        for the full mathematical formulation.

        Parameters
        ----------
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}
            Time aggregation frequency. ``"original"`` uses data's native
            frequency. ``"all_time"`` returns a single value across the entire
            period.
        start_date : str or pd.Timestamp, optional
            Start date for evaluation window. If None, uses start of fitted data.
        end_date : str or pd.Timestamp, optional
            End date for evaluation window. If None, uses end of fitted data.
        include_carryover : bool, default=True
            Include adstock carryover effects.  When True, prepends ``l_max``
            observations before the period to capture historical effects
            carrying into the evaluation period, and extends the evaluation
            window by ``l_max`` periods to capture trailing adstock effects
            from spend during the period.
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
            Incremental contributions in original scale with dimensions:

            - ``(chain, draw, date, channel, *custom_dims)`` when
              ``frequency != "all_time"``
            - ``(chain, draw, channel, *custom_dims)`` when
              ``frequency == "all_time"``

            For models with hierarchical dimensions like ``dims=("country",)``,
            output has shape ``(chain, draw, date, channel, country)``.

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
                start_date="2024-01-01",
                end_date="2024-12-31",
            )

        Mean contribution per channel per quarter:

        .. code-block:: python

            incremental.mean(dim=["chain", "draw"])

        Total annual contribution (all_time):

        .. code-block:: python

            annual = mmm.incrementality.compute_incremental_contribution(
                frequency="all_time",
                start_date="2024-01-01",
                end_date="2024-12-31",
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

        # Validate and parse dates
        start_date_ts, end_date_ts = self._validate_input(start_date, end_date)

        # Subsample posterior if needed (correctly across chain x draw)
        posterior_sub = subsample_draws(
            self.idata.posterior, num_samples=num_samples, random_state=random_state
        )
        n_chains = posterior_sub.sizes["chain"]
        n_draws = posterior_sub.sizes["draw"]

        # Extract response distribution (batched over samples)
        response_graph = extract_response_distribution(
            pymc_model=self.model.model,
            idata=az.InferenceData(posterior=posterior_sub),
            response_variable="channel_contribution",
        )
        # Shape: (sample, date, channel, *custom_dims) where sample = chain x draw

        # Create period groups based on frequency
        dates = self.data.dates
        periods = self._create_period_groups(start_date_ts, end_date_ts, frequency)

        # Get l_max for carryover calculations
        l_max = self.model.adstock.l_max
        inferred_freq: str | None = pd.infer_freq(dates)
        if inferred_freq is None:
            raise ValueError(
                "Could not infer frequency from the date index. "
                "Ensure the fitted data has a regular date frequency."
            )
        freq: str = inferred_freq
        freq_offset = pd.tseries.frequencies.to_offset(freq)

        # Compile vectorized evaluator (once, reused for both)
        # Use float64 for evaluation to avoid integer truncation when
        # counterfactual_spend_factor produces fractional values (e.g. 1.01).
        data_shared = self.model.model["channel_data"]
        eval_dtype = "float64"
        batched_input = pt.tensor(
            name="channel_data_batched",
            dtype=eval_dtype,
            shape=(None, *data_shared.type.shape),
        )
        replace_dict: dict = {data_shared: batched_input}
        func_inputs: list = [batched_input]

        # When time_varying_media (or time_varying_intercept) is enabled,
        # the response graph contains HSGP-based latent variables that
        # depend on the ``time_index`` shared variable (shape n_dates).
        # We must replace it alongside ``channel_data`` so that both
        # tensors have a consistent date dimension (max_window).
        has_time_index = "time_index" in self.model.model.named_vars
        if has_time_index:
            time_shared = self.model.model["time_index"]
            batched_time = pt.tensor(
                name="time_index_batched",
                dtype=time_shared.dtype,
                shape=(None, *time_shared.type.shape),
            )
            replace_dict[time_shared] = batched_time
            func_inputs.append(batched_time)

        batched_graph = vectorize_graph(response_graph, replace=replace_dict)
        evaluator = function(func_inputs, batched_graph)

        # Evaluate baseline on full dataset (once)
        baseline_array = self.data.get_channel_spend().values
        baseline_eval_args: list[np.ndarray] = [
            baseline_array[np.newaxis].astype(eval_dtype)
        ]
        if has_time_index:
            baseline_eval_args.append(
                np.arange(len(dates))[np.newaxis].astype(time_shared.dtype)
            )
        baseline_pred = evaluator(*baseline_eval_args)[0]
        # Shape: (n_samples, n_dates, *non_date_dims)

        # Determine actual axis ordering from the model's channel_contribution
        # variable. The PyTensor graph preserves the model's dim order, which
        # may have custom dims (e.g. "country") before "channel".
        cc_dims = list(
            self.model.model.named_vars_to_dims.get("channel_contribution", ())
        )
        non_date_dims = [d for d in cc_dims if d != "date"]
        extra_shape = baseline_array.shape[1:]  # (channel, *custom_dims)

        # Compute, for each period, the required window metadata including
        # the start/end indices into `dates` and any necessary left/right padding,
        # and determine the maximum window length across all periods.
        # Output:
        #   - window_infos: list of dicts (one per period), each containing
        #       info about the window's bounds and padding for use in evaluation
        #   - max_window: int, the maximum window length needed for counterfactual evaluation
        window_infos, max_window = self._compute_window_metadata(
            periods, dates, l_max, freq_offset, freq
        )

        # Build counterfactual scenarios:
        #   - cf_array: array of counterfactual channel_data with shaped (n_periods, max_window, channel, *custom_dims)
        #   - cf_eval_masks: period-specific boolean masks indicating which
        #     max_window indices correspond to the period's actual dates
        #   - period_labels: labels (dates or ranges) identifying each incrementality window/period
        cf_array, cf_eval_masks, period_labels = self._build_counterfactual_scenarios(
            periods=periods,
            window_infos=window_infos,
            max_window=max_window,
            baseline_array=baseline_array,
            counterfactual_spend_factor=counterfactual_spend_factor,
            include_carryover=include_carryover,
            l_max=l_max,
            freq_offset=freq_offset,
            extra_shape=extra_shape,
            dtype=eval_dtype,
        )

        # Evaluate all counterfactuals at once
        cf_eval_args: list[np.ndarray] = [cf_array]
        if has_time_index:
            cf_eval_args.append(
                self._build_time_index_array(
                    window_infos=window_infos,
                    dates=dates,
                    max_window=max_window,
                    dtype=time_shared.dtype,
                )
            )
        cf_predictions = evaluator(*cf_eval_args)
        # Shape: (n_periods, n_samples, max_window, channel, *custom_dims)

        # Assemble results
        return self._compute_period_increments(
            periods=periods,
            period_labels=period_labels,
            baseline_pred=baseline_pred,
            cf_predictions=cf_predictions,
            cf_eval_masks=cf_eval_masks,
            dates=dates,
            include_carryover=include_carryover,
            l_max=l_max,
            freq_offset=freq_offset,
            counterfactual_spend_factor=counterfactual_spend_factor,
            non_date_dims=non_date_dims,
            frequency=frequency,
            n_chains=n_chains,
            n_draws=n_draws,
        )

    def _validate_input(
        self,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Parse and validate input dates against the fitted data range.

        Parameters
        ----------
        start_date : str or pd.Timestamp or None
            Start date. If None, uses start of fitted data.
        end_date : str or pd.Timestamp or None
            End date. If None, uses end of fitted data.

        Returns
        -------
        tuple of (pd.Timestamp, pd.Timestamp)
            Validated ``(start_date, end_date)``.

        Raises
        ------
        ValueError
            If dates are outside fitted data range or start > end.
        """
        dates = self.data.dates
        data_start = dates[0]
        data_end = dates[-1]

        start_date_ts: pd.Timestamp = (
            data_start if start_date is None else pd.to_datetime(start_date)
        )
        end_date_ts: pd.Timestamp = (
            data_end if end_date is None else pd.to_datetime(end_date)
        )

        if start_date_ts < data_start:
            raise ValueError(
                f"start_date '{start_date_ts.date()}' is before fitted data "
                f"start '{data_start.date()}'."
            )
        if end_date_ts > data_end:
            raise ValueError(
                f"end_date '{end_date_ts.date()}' is after fitted data "
                f"end '{data_end.date()}'."
            )
        if start_date_ts > end_date_ts:
            raise ValueError(
                f"start_date '{start_date_ts.date()}' is after "
                f"end_date '{end_date_ts.date()}'."
            )

        return start_date_ts, end_date_ts

    @staticmethod
    def _compute_window_metadata(
        periods: list[tuple[pd.Timestamp, pd.Timestamp]],
        dates: pd.DatetimeIndex,
        l_max: int,
        freq_offset: BaseOffset,
        freq: str,
    ) -> tuple[list[dict], int]:
        """Compute per-period window metadata for counterfactual evaluation.

        For each period, determines the ideal window
        ``[t0 - l_max, t1 + l_max]``, finds actual dates in that window,
        and computes left/right padding for positions that fall outside the
        dataset.

        Parameters
        ----------
        periods : list of (pd.Timestamp, pd.Timestamp)
            Period ``(start, end)`` pairs.
        dates : pd.DatetimeIndex
            All dates from the fitted data.
        l_max : int
            Adstock maximum lag.
        freq_offset : pd.DateOffset
            Calendar-aware frequency offset.
        freq : str
            Pandas frequency string for ``date_range``.

        Returns
        -------
        tuple of (list[dict], int)
            ``(window_infos, max_window)`` where each dict has keys:
            ``left_pad``, ``right_pad``, ``n_actual``, ``in_window``,
            ``actual_dates``.
        """
        window_infos: list[dict] = []
        for t0, t1 in periods:
            # Always include l_max carry-in for correct adstock context.
            # Always include l_max carry-out so carryover effects are
            # captured (eval mask controls what gets summed).
            ideal_start = t0 - l_max * freq_offset
            ideal_end = t1 + l_max * freq_offset

            # Actual dates from the dataset that fall in the ideal window
            in_window = (dates >= ideal_start) & (dates <= ideal_end)
            actual_dates = dates[in_window]
            n_actual = int(in_window.sum())

            # Left-padding: count frequency steps between ideal_start and
            # first actual date (represents "no spend" before the dataset).
            # Uses date_range counting instead of timedelta division to
            # handle variable-length calendar periods (months, years).
            left_pad = 0
            if n_actual > 0 and actual_dates[0] > ideal_start:
                left_pad = (
                    len(pd.date_range(ideal_start, actual_dates[0], freq=freq)) - 1
                )

            # Right-padding: count frequency steps between last actual date
            # and ideal_end (represents "no spend" after the dataset).
            right_pad = 0
            if n_actual > 0 and actual_dates[-1] < ideal_end:
                right_pad = (
                    len(pd.date_range(actual_dates[-1], ideal_end, freq=freq)) - 1
                )

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

        return window_infos, max_window

    @staticmethod
    def _build_counterfactual_scenarios(
        periods: list[tuple[pd.Timestamp, pd.Timestamp]],
        window_infos: list[dict],
        max_window: int,
        baseline_array: np.ndarray,
        counterfactual_spend_factor: float,
        include_carryover: bool,
        l_max: int,
        freq_offset: BaseOffset,
        extra_shape: tuple,
        dtype: str,
    ) -> tuple[np.ndarray, list[np.ndarray], list[pd.Timestamp]]:
        """Build zero-padded counterfactual arrays for batched evaluation.

        Each counterfactual window covers ``[t0 - l_max, t1 + l_max]`` to
        capture carry-in history (for correct adstock) and carry-out
        effects.  At dataset boundaries, positions outside the data are
        zero-padded (= no spend), and all windows are right-padded to a
        uniform max size so they can be stacked for batched evaluation.

        Parameters
        ----------
        periods : list of (pd.Timestamp, pd.Timestamp)
            Period ``(start, end)`` pairs.
        window_infos : list of dict
            Per-period metadata from :meth:`_compute_window_metadata`.
        max_window : int
            Maximum padded window size across all periods.
        baseline_array : np.ndarray
            Actual channel spend data, shape ``(n_dates, channel, *custom_dims)``.
        counterfactual_spend_factor : float
            Multiplicative factor for counterfactual spend.
        include_carryover : bool
            Whether to include carryover effects in eval mask.
        l_max : int
            Adstock maximum lag.
        freq_offset : pd.DateOffset
            Calendar-aware frequency offset.
        extra_shape : tuple
            Shape of non-date dimensions ``(channel, *custom_dims)``.
        dtype : str
            NumPy dtype for the output array.

        Returns
        -------
        tuple of (np.ndarray, list[np.ndarray], list[pd.Timestamp])
            ``(cf_array, cf_eval_masks, period_labels)`` where:

            - ``cf_array``: shape ``(n_periods, max_window, *extra_shape)``
            - ``cf_eval_masks``: per-period boolean masks over padded window
            - ``period_labels``: end date of each period
        """
        cf_scenarios: list[np.ndarray] = []
        # Per-period boolean eval masks over the padded window (only True
        # at actual-data positions within the eval date range, ensuring
        # consistency with the baseline eval which also covers only actual
        # dates).
        cf_eval_masks: list[np.ndarray] = []
        period_labels: list[pd.Timestamp] = []

        for (t0, t1), info in zip(periods, window_infos, strict=True):
            # Allocate zero-padded window
            padded = np.zeros((max_window, *extra_shape), dtype=dtype)

            # Place actual data at correct offset
            start_pos = info["left_pad"]
            end_pos = start_pos + info["n_actual"]
            padded[start_pos:end_pos] = baseline_array[info["in_window"]].astype(dtype)

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
                    carryout_end = t1 + l_max * freq_offset
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

        cf_array = np.stack(cf_scenarios, axis=0)
        return cf_array, cf_eval_masks, period_labels

    @staticmethod
    def _build_time_index_array(
        window_infos: list[dict],
        dates: pd.DatetimeIndex,
        max_window: int,
        dtype: str,
    ) -> np.ndarray:
        """Build batched time_index arrays for counterfactual windows.

        Each window needs a corresponding ``time_index`` array so that
        HSGP-based latent variables (e.g. ``media_temporal_latent_multiplier``)
        evaluate their basis functions at the correct temporal positions.

        Parameters
        ----------
        window_infos : list of dict
            Per-period metadata from :meth:`_compute_window_metadata`.
        dates : pd.DatetimeIndex
            All dates from the fitted data.
        max_window : int
            Maximum padded window size across all periods.
        dtype : str
            NumPy dtype for the output array.

        Returns
        -------
        np.ndarray
            Time index array of shape ``(n_periods, max_window)``, where
            each row contains sequential integer indices corresponding to
            the temporal positions in the window.  Indices may extend
            beyond ``[0, n_dates)`` for boundary padding.
        """
        time_index_scenarios: list[np.ndarray] = []
        for info in window_infos:
            if info["n_actual"] > 0:
                first_actual_idx = int(np.searchsorted(dates, info["actual_dates"][0]))
                start_idx = first_actual_idx - info["left_pad"]
            else:
                start_idx = 0
            window_time_index = np.arange(
                start_idx, start_idx + max_window, dtype=dtype
            )
            time_index_scenarios.append(window_time_index)
        return np.stack(time_index_scenarios, axis=0)

    def _compute_period_increments(
        self,
        periods: list[tuple[pd.Timestamp, pd.Timestamp]],
        period_labels: list[pd.Timestamp],
        baseline_pred: np.ndarray,
        cf_predictions: np.ndarray,
        cf_eval_masks: list[np.ndarray],
        dates: pd.DatetimeIndex,
        include_carryover: bool,
        l_max: int,
        freq_offset: BaseOffset,
        counterfactual_spend_factor: float,
        non_date_dims: list[str],
        frequency: Frequency,
        n_chains: int,
        n_draws: int,
    ) -> xr.DataArray:
        """Assemble per-period incremental results into a single DataArray.

        For each period, sums baseline and counterfactual predictions over
        the evaluation window, computes the difference (with appropriate
        sign convention), reshapes the flattened sample dimension back to
        ``(chain, draw)``, and concatenates all periods into a single
        ``xr.DataArray`` in original scale.

        Parameters
        ----------
        periods : list of (pd.Timestamp, pd.Timestamp)
            Period ``(start, end)`` pairs.
        period_labels : list of pd.Timestamp
            End date label for each period.
        baseline_pred : np.ndarray
            Baseline predictions, shape ``(n_samples, n_dates, *non_date_dims)``.
        cf_predictions : np.ndarray
            Counterfactual predictions, shape
            ``(n_periods, n_samples, max_window, *non_date_dims)``.
        cf_eval_masks : list of np.ndarray
            Per-period boolean masks over the padded window.
        dates : pd.DatetimeIndex
            All dates from the fitted data.
        include_carryover : bool
            Whether to include carryover effects in eval mask.
        l_max : int
            Adstock maximum lag.
        freq_offset : pd.DateOffset
            Calendar-aware frequency offset.
        counterfactual_spend_factor : float
            Multiplicative factor used for sign convention.
        non_date_dims : list of str
            Dimension names excluding ``"date"`` (e.g. ``["channel"]`` or
            ``["channel", "country"]``).
        frequency : Frequency
            Time aggregation frequency.
        n_chains : int
            Number of MCMC chains in the posterior.
        n_draws : int
            Number of draws per chain.

        Returns
        -------
        xr.DataArray
            Incremental contributions in original scale with dimensions
            ``(chain, draw, date, channel, *custom_dims)`` or
            ``(chain, draw, channel, *custom_dims)`` for ``"all_time"``.
        """
        fit_data = self.idata.fit_data
        n_periods = len(periods)
        results = []

        for period_idx in range(n_periods):
            t0, t1 = periods[period_idx]

            # Baseline: sum over eval dates from full-dataset prediction
            if include_carryover:
                carryout_end = t1 + l_max * freq_offset
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
            # Shape: (n_samples, *non_date_dims) where n_samples = n_chains * n_draws

            # Reshape flattened sample → (chain, draw) to preserve MCMC structure
            reshaped = total_incremental.reshape(
                n_chains, n_draws, *total_incremental.shape[1:]
            )

            # Add period coordinate
            period_label = period_labels[period_idx]
            total_incremental_da = xr.DataArray(
                reshaped,
                dims=("chain", "draw", *non_date_dims),
                coords={
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
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
        dim_order = ["chain", "draw", "date", "channel", *self.model.dims]
        if frequency == "all_time":
            dim_order.remove("date")
        result = result.transpose(*dim_order)

        # Always apply original scale
        target_scale = self.data.get_target_scale()
        result = result * target_scale

        return result

    # ==================== Convenience Methods ====================

    def contribution_over_spend(
        self,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
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
        start_date, end_date : str or pd.Timestamp, optional
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
            ``(chain, draw, date, channel, *custom_dims)``.
            Zero spend results in NaN for that channel/period.

        Examples
        --------
        >>> roas = mmm.incrementality.contribution_over_spend(
        ...     frequency="quarterly",
        ...     start_date="2024-01-01",
        ...     end_date="2024-12-31",
        ... )
        """
        incremental = self.compute_incremental_contribution(
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            include_carryover=include_carryover,
            num_samples=num_samples,
            random_state=random_state,
            counterfactual_spend_factor=0.0,
        )

        spend = self._aggregate_spend(frequency, start_date, end_date)
        spend_safe = xr.where(spend == 0, np.nan, spend)

        return incremental / spend_safe

    def spend_over_contribution(
        self,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
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
        start_date, end_date : str or pd.Timestamp, optional
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
            ``(chain, draw, date, channel, *custom_dims)``.
            Zero contribution results in Inf; zero spend results in NaN.

        Examples
        --------
        >>> cac = mmm.incrementality.spend_over_contribution(
        ...     frequency="monthly",
        ... )
        """
        ratio = self.contribution_over_spend(
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            include_carryover=include_carryover,
            num_samples=num_samples,
            random_state=random_state,
        )

        return 1.0 / ratio

    def marginal_contribution_over_spend(
        self,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
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
        start_date, end_date : str or pd.Timestamp, optional
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
            ``(chain, draw, date, channel, *custom_dims)``.
            Zero spend results in NaN for that channel/period.

        Raises
        ------
        ValueError
            If ``spend_increase_pct <= 0``.

        Examples
        --------
        >>> mroas = mmm.incrementality.marginal_contribution_over_spend(
        ...     frequency="quarterly",
        ...     start_date="2024-01-01",
        ...     end_date="2024-12-31",
        ... )
        """
        if spend_increase_pct <= 0:
            raise ValueError(
                f"spend_increase_pct must be > 0, got {spend_increase_pct}"
            )

        factor = 1.0 + spend_increase_pct

        marginal_contribution = self.compute_incremental_contribution(
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            include_carryover=include_carryover,
            num_samples=num_samples,
            random_state=random_state,
            counterfactual_spend_factor=factor,
        )

        spend = self._aggregate_spend(frequency, start_date, end_date)

        # Denominator is the *incremental* spend: pct * total_spend
        incremental_spend = spend_increase_pct * spend
        incremental_spend_safe = xr.where(
            incremental_spend == 0, np.nan, incremental_spend
        )

        return marginal_contribution / incremental_spend_safe

    # ==================== Period & Subsampling Helpers ====================

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
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
                freq=pd.infer_freq(self.data.dates),
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

        # Validate that end aligns with a period boundary.
        last_period_boundary = unique_periods[-1].to_timestamp(how="end").normalize()
        if end < last_period_boundary:
            data_last_date = self.data.dates[-1]
            if end != data_last_date:
                raise ValueError(
                    f"end_date ({end.strftime('%Y-%m-%d')}) falls in the "
                    f"middle of a {frequency} period that ends on "
                    f"{last_period_boundary.strftime('%Y-%m-%d')}. "
                    f"Use an end_date that aligns with a {frequency} "
                    f"boundary, or omit end_date to use the last date "
                    f"of the fitted data "
                    f"({data_last_date.strftime('%Y-%m-%d')})."
                )

        period_ranges = []
        for period in unique_periods:
            period_start = period.to_timestamp()
            period_end = period.to_timestamp(how="end").normalize()

            # Clip start to the requested range (needed when the user
            # passes a start date inside a period)
            period_start = max(period_start, start)

            period_ranges.append((period_start, period_end))

        return period_ranges

    def _aggregate_spend(
        self,
        frequency: Frequency,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> xr.DataArray:
        """Aggregate channel spend by frequency over a date range.

        Delegates to self.data (MMMIDataWrapper) for date filtering and time
        aggregation.

        Parameters
        ----------
        frequency : Frequency
            Time aggregation frequency
        start_date, end_date : str or pd.Timestamp, optional
            Date range. If None, uses full fitted data range.

        Returns
        -------
        xr.DataArray
            Aggregated spend with dims (date, channel, *custom_dims) or
            (channel, *custom_dims) for "all_time"
        """
        # 1. Filter to date range
        data = self.data.filter_dates(start_date, end_date)

        # 2. Aggregate over time (no-op for "original")
        if frequency != "original":
            data = data.aggregate_time(period=frequency, method="sum")

        # 3. Return spend with channel dimension
        return data.get_channel_spend()
