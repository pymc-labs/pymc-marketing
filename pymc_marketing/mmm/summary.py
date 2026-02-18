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
"""Summary DataFrame generation for MMM models.

Access via MMM model:
    >>> df = mmm.summary.contributions()
    >>> df = mmm.summary.roas()

Or create factory directly:
    >>> from pymc_marketing.mmm.summary import MMMSummaryFactory
    >>> factory = MMMSummaryFactory(mmm.data, model=mmm)
    >>> df = factory.contributions()

Key Features:
    - output_format parameter: Choose between Pandas and Polars DataFrames
    - frequency parameter: View data at different aggregation levels
    - HDI computation: Configurable probability levels for uncertainty
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import validate_call
from pymc.util import RandomState

from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper
from pymc_marketing.data.idata.schema import Frequency
from pymc_marketing.data.idata.utils import aggregate_idata_dims
from pymc_marketing.mmm.incrementality import Incrementality

# Type aliases
OutputFormat = Literal["pandas", "polars"]

# Union type for return values
DataFrameType = pd.DataFrame  # Will be Union[pd.DataFrame, pl.DataFrame] at runtime

# Public API
__all__ = [
    "DataFrameType",
    "Frequency",
    "MMMSummaryFactory",
    "OutputFormat",
]


@dataclass(frozen=True)
class MMMSummaryFactory:
    """Factory for creating summary DataFrames from MMM data.

    Provides a convenient interface for generating summary DataFrames
    with shared default settings. Accepts data wrapper (required) and
    optionally the MMM model to access transformations.

    The factory is immutable (frozen dataclass). To create a factory with
    different settings, instantiate a new one directly.

    Parameters
    ----------
    data : MMMIDataWrapper
        Data wrapper containing idata and schema (required)
    model : MMM, optional
        Fitted MMM model with transformations (saturation, adstock).
        Required for saturation_curves() and adstock_curves() methods.
    hdi_probs : sequence of float, optional
        Default HDI probability levels (default: (0.94,)).
        Accepts list or tuple; stored internally as tuple.
    output_format : {"pandas", "polars"}, default "pandas"
        Default output DataFrame format

    Examples
    --------
    >>> # With data only (for most summaries)
    >>> factory = MMMSummaryFactory(mmm.data)
    >>> contributions_df = factory.contributions()
    >>>
    >>> # With model (for transformation curves)
    >>> factory = MMMSummaryFactory(mmm.data, model=mmm)
    >>> saturation_df = factory.saturation_curves()
    >>>
    >>> # Via model property (recommended - includes model automatically)
    >>> factory = mmm.summary
    >>> saturation_df = factory.saturation_curves()
    >>>
    >>> # Create new factory with different settings (direct instantiation)
    >>> polars_factory = MMMSummaryFactory(
    ...     mmm.data, model=mmm, output_format="polars", hdi_probs=[0.80, 0.94]
    ... )
    >>> df = polars_factory.contributions()  # Uses configured defaults
    """

    data: MMMIDataWrapper
    model: Any | None = None  # MMM type, but avoid circular import
    hdi_probs: Sequence[float] = (0.94,)
    output_format: OutputFormat = "pandas"
    validate_data: bool = True

    def __post_init__(self) -> None:
        """Validate data and convert hdi_probs to tuple for immutability."""
        # Convert hdi_probs to tuple if passed as list (for immutability)
        if not isinstance(self.hdi_probs, tuple):
            object.__setattr__(self, "hdi_probs", tuple(self.hdi_probs))

        # Validate data structure at initialization (early fail)
        if (
            self.validate_data
            and hasattr(self.data, "validate_or_raise")
            and self.data.schema is not None
        ):
            self.data.validate_or_raise()

        # Validate HDI probs at init time
        self._validate_hdi_probs(self.hdi_probs)

    # ==================== Private Helper Methods ====================

    def _validate_hdi_probs(self, hdi_probs: Sequence[float]) -> None:
        """Validate HDI probability values are in valid range.

        Parameters
        ----------
        hdi_probs : sequence of float
            HDI probability levels to validate

        Raises
        ------
        ValueError
            If any probability is not in range (0, 1)
        """
        for prob in hdi_probs:
            if not 0 < prob < 1:
                raise ValueError(
                    f"HDI probability must be between 0 and 1 (exclusive), got {prob}. "
                    "Use values like 0.94 for 94% HDI, not percentages like 94."
                )

    def _convert_output(
        self, df: pd.DataFrame, output_format: OutputFormat | None = None
    ) -> DataFrameType:
        """Convert Pandas DataFrame to requested output format.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (always Pandas from internal computation)
        output_format : {"pandas", "polars"} or None
            Desired output format. If None, uses factory default.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            DataFrame in requested format

        Raises
        ------
        ImportError
            If output_format="polars" but polars is not installed
        ValueError
            If output_format is not recognized
        """
        fmt = output_format if output_format is not None else self.output_format
        if fmt == "pandas":
            return df
        elif fmt == "polars":
            try:
                import polars as pl
            except ImportError:
                raise ImportError(
                    "Polars is required for output_format='polars'. "
                    "Install it with: pip install pymc-marketing[polars]"
                )
            return pl.from_pandas(df)
        else:
            raise ValueError(
                f"Unknown output_format: {fmt!r}. Use 'pandas' or 'polars'."
            )

    def _compute_summary_stats_with_hdi(
        self,
        data: xr.DataArray,
        hdi_probs: Sequence[float],
    ) -> pd.DataFrame:
        """Convert xarray to DataFrame with summary stats and HDI.

        Core transformation function that:
        1. Computes mean and median across MCMC samples
        2. Computes HDI bounds for each probability level
        3. Returns structured DataFrame with absolute HDI bounds

        Parameters
        ----------
        data : xr.DataArray
            Must have 'chain' and 'draw' dimensions OR a single 'sample' dimension.
            The sample dimensions are auto-detected based on which dimensions are present.
        hdi_probs : list of float
            HDI probability levels (e.g., [0.80, 0.94])

        Returns
        -------
        pd.DataFrame
            With columns: <dimensions>, mean, median, abs_error_{prob}_lower,
            abs_error_{prob}_upper

        Notes
        -----
        - Always returns Pandas internally
        - Conversion to other formats happens at public API boundary
        - Uses az.stats.hdi() for HDI computation when data has chain/draw dims
        - Uses quantile-based HDI for data with sample dimension
        """
        # Auto-detect sample dimensions based on what's present in the data
        if "chain" in data.dims and "draw" in data.dims:
            sample_dims = ["chain", "draw"]
            use_az_hdi = True
        elif "sample" in data.dims:
            sample_dims = ["sample"]
            use_az_hdi = False
        else:
            raise ValueError(
                f"Data must have either ('chain', 'draw') or 'sample' dimensions. "
                f"Found dimensions: {list(data.dims)}"
            )

        # Determine the index columns (all dims except sample dimensions)
        index_cols = [d for d in data.dims if d not in sample_dims]

        # Rename the DataArray to avoid conflicts with coordinates
        # (az.hdi fails if name matches a coordinate name)
        var_name = "_values"
        data = data.rename(var_name)

        # Compute point estimates
        mean_ = data.mean(dim=sample_dims)
        median_ = data.median(dim=sample_dims)

        # Compute HDI for each probability level
        hdi_results = {}
        for hdi_prob in hdi_probs:
            prob_str = str(int(hdi_prob * 100))

            if use_az_hdi:
                # Use az.hdi when we have chain/draw dimensions
                hdi_dataset = az.hdi(data, hdi_prob=hdi_prob)
                hdi_da = hdi_dataset[var_name]
                # Drop the 'hdi' coordinate after selection to avoid conflicts
                hdi_lower = hdi_da.sel(hdi="lower").drop_vars("hdi", errors="ignore")
                hdi_upper = hdi_da.sel(hdi="higher").drop_vars("hdi", errors="ignore")
            else:
                # Use quantile-based HDI for single sample dimension
                # Symmetric HDI: take (1-prob)/2 and 1-(1-prob)/2 quantiles
                alpha = 1 - hdi_prob
                lower_q = alpha / 2
                upper_q = 1 - alpha / 2
                hdi_lower = data.quantile(lower_q, dim=sample_dims).drop_vars(
                    "quantile", errors="ignore"
                )
                hdi_upper = data.quantile(upper_q, dim=sample_dims).drop_vars(
                    "quantile", errors="ignore"
                )

            hdi_results[f"abs_error_{prob_str}_lower"] = hdi_lower
            hdi_results[f"abs_error_{prob_str}_upper"] = hdi_upper

        # Build a single Dataset with all results and convert to DataFrame
        result_dict = {"mean": mean_, "median": median_, **hdi_results}
        result_ds = xr.Dataset(result_dict)

        # Convert to DataFrame - this preserves coordinate values
        df = result_ds.to_dataframe().reset_index()

        # Ensure coordinate columns have correct order
        other_cols = [c for c in df.columns if c not in index_cols]
        df = df[index_cols + other_cols]

        return df

    def _prepare_data_and_hdi(
        self,
        hdi_probs: Sequence[float] | None = None,
        frequency: Frequency | None = None,
        output_format: OutputFormat | None = None,
    ) -> tuple[MMMIDataWrapper, Sequence[float], OutputFormat]:
        """Prepare data, resolve defaults, and validate.

        This is the main "resolve defaults" method that should be called
        at the start of each public summary method. It:
        1. Resolves hdi_probs default from self.hdi_probs
        2. Resolves output_format default from self.output_format
        3. Validates hdi_probs
        4. Aggregates data by frequency if specified

        Parameters
        ----------
        hdi_probs : sequence of float or None
            HDI probability levels (None uses factory default)
        frequency : Frequency or None
            Time aggregation period (None or "original" means no aggregation)
        output_format : OutputFormat or None
            Output format (None uses factory default)

        Returns
        -------
        tuple[MMMIDataWrapper, Sequence[float], OutputFormat]
            (prepared_data, effective_hdi_probs, effective_output_format)
        """
        effective_hdi_probs = hdi_probs if hdi_probs is not None else self.hdi_probs
        effective_output_format = (
            output_format if output_format is not None else self.output_format
        )

        self._validate_hdi_probs(effective_hdi_probs)

        data = self.data
        if frequency is not None and frequency != "original":
            data = data.aggregate_time(frequency)

        return data, effective_hdi_probs, effective_output_format

    def _require_model(self, method_name: str) -> None:
        """Raise helpful error if model is required but not provided."""
        if self.model is None:
            raise ValueError(
                f"{method_name} requires model to access transformations. "
                f"Use MMMSummaryFactory(data, model=mmm) or mmm.summary instead."
            )

    def posterior_predictive(
        self,
        hdi_probs: Sequence[float] | None = None,
        frequency: Frequency | None = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Create posterior predictive summary DataFrame.

        Computes mean, median, and HDI bounds for posterior predictive samples,
        along with observed values for comparison.

        Parameters
        ----------
        hdi_probs : sequence of float, optional
            HDI probability levels (default: uses factory default)
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}, optional
            Time aggregation period (default: None, no aggregation)
        output_format : {"pandas", "polars"}, optional
            Output DataFrame format (default: uses factory default)

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Summary DataFrame with columns:

            - date: Time index
            - mean: Mean of posterior predictive samples
            - median: Median of posterior predictive samples
            - observed: Observed target values
            - abs_error_{prob}_lower: HDI lower bound for each prob
            - abs_error_{prob}_upper: HDI upper bound for each prob

        Examples
        --------
        >>> df = mmm.summary.posterior_predictive()
        >>> df = mmm.summary.posterior_predictive(frequency="monthly")
        >>> df = mmm.summary.posterior_predictive(hdi_probs=[0.80, 0.94])
        """
        # Resolve all defaults in one call
        data, hdi_probs, output_format = self._prepare_data_and_hdi(
            hdi_probs, frequency, output_format
        )

        # Get posterior predictive samples
        if hasattr(data.idata, "posterior_predictive"):
            pp_samples = data.to_original_scale(data.idata.posterior_predictive["y"])
        else:
            raise AttributeError("No posterior predictive samples found in idata")

        # Get observed data
        observed = data.get_target(original_scale=True)

        # Compute summary stats with HDI
        df = self._compute_summary_stats_with_hdi(pp_samples, hdi_probs)

        # Add observed values
        observed_df = observed.to_dataframe(name="observed").reset_index()
        merge_keys = ["date", *data.custom_dims]
        df = df.merge(observed_df, on=merge_keys, how="left")

        return self._convert_output(df, output_format)

    @validate_call
    def contributions(
        self,
        hdi_probs: Sequence[float] | None = None,
        component: Literal[
            "channel", "channels", "control", "controls", "seasonality", "baseline"
        ] = "channel",
        frequency: Frequency | None = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Create contribution summary DataFrame.

        Computes mean, median, and HDI bounds for contribution samples
        for the specified component type.

        Parameters
        ----------
        hdi_probs : sequence of float, optional
            HDI probability levels (default: uses factory default)
        component : {"channel", "control", "seasonality", "baseline"}, default "channel"
            Which contribution component to summarize
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}, optional
            Time aggregation period (default: None, no aggregation)
        output_format : {"pandas", "polars"}, optional
            Output DataFrame format (default: uses factory default)

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Summary DataFrame with columns:

            - date: Time index
            - channel/control: Component identifier
            - mean: Mean contribution
            - median: Median contribution
            - abs_error_{prob}_lower: HDI lower bound for each prob
            - abs_error_{prob}_upper: HDI upper bound for each prob

        Examples
        --------
        >>> df = mmm.summary.contributions()
        >>> df = mmm.summary.contributions(component="control")
        >>> df = mmm.summary.contributions(frequency="monthly", hdi_probs=[0.80, 0.94])

        Notes
        -----
        Expects validated data. Call `data.validate_or_raise()` if you've
        modified the underlying idata before calling this method.
        """
        # Resolve all defaults in one call
        data, hdi_probs, output_format = self._prepare_data_and_hdi(
            hdi_probs, frequency, output_format
        )

        if component == "control":
            component = "controls"

        # Get contributions via Component 1 (handles scaling)
        if component.startswith("channel"):
            component_data = data.get_channel_contributions(original_scale=True)
        else:
            contributions = data.get_contributions(
                original_scale=True,
                include_baseline=(component == "baseline"),
                include_controls=(component == "controls"),
                include_seasonality=(component == "seasonality"),
            )

            if component not in contributions.data_vars:
                raise ValueError(f"No {component} contributions found in model")
            component_data = contributions[component]

        # Compute summary stats with HDI
        df = self._compute_summary_stats_with_hdi(component_data, hdi_probs)

        return self._convert_output(df, output_format)

    def roas(
        self,
        hdi_probs: Sequence[float] | None = None,
        frequency: Frequency | None = None,
        method: Literal["incremental", "elementwise"] = "incremental",
        include_carryover: bool = True,
        num_samples: int | None = None,
        random_state: RandomState | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        aggregate_dims: dict[str, Any] | list[dict[str, Any]] | None = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Create ROAS (Return on Ad Spend) summary DataFrame.

        Computes ROAS = contribution / spend for each channel with
        mean, median, and HDI bounds.

        Parameters
        ----------
        hdi_probs : sequence of float, optional
            HDI probability levels (default: uses factory default)
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}, optional
            Time aggregation period (default: None, no aggregation)
        method : {"incremental", "elementwise"}, default "incremental"
            Method for computing ROAS:

            - **incremental** (recommended): Uses counterfactual analysis
              accounting for adstock carryover effects. Computes the true
              incremental return on ad spend using Google MMM Formula 10.
              Requires model to be provided (e.g. via ``mmm.summary``).

            - **elementwise**: Simple element-wise division of contributions
              by spend. Does NOT account for carryover effects. Useful for
              daily efficiency tracking but not true incrementality.
              Works with data-only factory.
        include_carryover : bool, default True
            Include adstock carryover effects. Only used when
            ``method="incremental"``.
        num_samples : int or None, optional
            Number of posterior samples to use. If None, all samples are used.
            Only used when ``method="incremental"``.
        random_state : int, np.random.Generator, np.random.RandomState, or None, optional
            Random state for reproducible subsampling. Only used when
            ``method="incremental"`` and ``num_samples`` is not None.
        start_date : str or pd.Timestamp, optional
            Start date for the evaluation window.  For
            ``method="incremental"`` this is passed to
            :meth:`~pymc_marketing.mmm.incrementality.Incrementality.contribution_over_spend`.
            Spend *before* this date still influences ROAS through adstock
            carryover effects (the counterfactual analysis automatically
            includes the necessary carry-in context).
            For ``method="elementwise"`` the ROAS result is filtered to
            dates on or after this value.
        end_date : str or pd.Timestamp, optional
            End date for the evaluation window.  For
            ``method="incremental"`` this is passed to
            :meth:`~pymc_marketing.mmm.incrementality.Incrementality.contribution_over_spend`.
            Spend *during* the window continues to generate returns *after*
            this date through adstock carryover; those trailing effects are
            included in the ROAS calculation.
            For ``method="elementwise"`` the ROAS result is filtered to
            dates on or before this value.
        aggregate_dims : dict or list[dict], optional
            Post-computation dimension aggregation. Accepts either a single
            dict or a list of dicts. Each dict contains keyword arguments
            that are passed directly to
            :func:`~pymc_marketing.data.idata.utils.aggregate_idata_dims`:

            - ``dim`` (str): Dimension to aggregate (e.g. ``"channel"``).
            - ``values`` (list[str]): Coordinate values to aggregate.
            - ``new_label`` (str): Label for the aggregated value.
            - ``method`` (str, optional): ``"sum"`` (default) or ``"mean"``.

            When a **list** is provided, each element is applied
            sequentially, which allows multiple aggregations on the same
            dimension (e.g. grouping different sets of channels).

            Single dict example::

                aggregate_dims = {
                    "dim": "channel",
                    "values": ["Facebook", "Instagram"],
                    "new_label": "Social",
                    "method": "sum",
                }

            List of dicts example::

                aggregate_dims = [
                    {
                        "dim": "channel",
                        "values": ["Facebook", "Instagram"],
                        "new_label": "Social",
                    },
                    {
                        "dim": "channel",
                        "values": ["Google", "Bing"],
                        "new_label": "Search",
                    },
                ]

        output_format : {"pandas", "polars"}, optional
            Output DataFrame format (default: uses factory default)

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Summary DataFrame with columns:

            - date: Time index
            - channel: Channel name
            - mean: Mean ROAS
            - median: Median ROAS
            - abs_error_{prob}_lower: HDI lower bound for each prob
            - abs_error_{prob}_upper: HDI upper bound for each prob

        Examples
        --------
        >>> df = mmm.summary.roas()
        >>> df = mmm.summary.roas(frequency="monthly")
        >>> df = mmm.summary.roas(method="incremental", include_carryover=True)
        >>> df = factory.roas(method="elementwise")
        >>> df = mmm.summary.roas(
        ...     start_date="2024-01-01",
        ...     end_date="2024-06-30",
        ...     frequency="quarterly",
        ... )
        >>> df = mmm.summary.roas(
        ...     aggregate_dims={
        ...         "dim": "channel",
        ...         "values": ["channel_1", "channel_2"],
        ...         "new_label": "combined",
        ...     }
        ... )
        >>> df = mmm.summary.roas(
        ...     aggregate_dims=[
        ...         {
        ...             "dim": "channel",
        ...             "values": ["ch_1", "ch_2"],
        ...             "new_label": "group_A",
        ...         },
        ...         {
        ...             "dim": "channel",
        ...             "values": ["ch_3", "ch_4"],
        ...             "new_label": "group_B",
        ...         },
        ...     ]
        ... )

        """
        if method == "elementwise":
            incremental_only_params = {
                "include_carryover": include_carryover != True,  # noqa: E712
                "num_samples": num_samples is not None,
                "random_state": random_state is not None,
            }
            for name, was_set in incremental_only_params.items():
                if was_set:
                    raise ValueError(
                        f"parameter {name} is only supported with method='incremental', "
                        f"either remove it or use method='incremental'."
                    )

        # Resolve all defaults in one call
        data, hdi_probs, output_format = self._prepare_data_and_hdi(
            hdi_probs, frequency, output_format
        )

        if method == "incremental":
            self._require_model("roas with method='incremental'")
            if self.model is None:
                raise RuntimeError("Model should not be None after _require_model")
            incr = Incrementality(model=self.model, data=self.data)
            roas = incr.contribution_over_spend(
                frequency=frequency or "original",
                period_start=start_date,
                period_end=end_date,
                include_carryover=include_carryover,
                num_samples=num_samples,
                random_state=random_state,
            )
        elif method == "elementwise":
            data = data.filter_dates(start_date=start_date, end_date=end_date)
            roas = data.get_elementwise_roas(original_scale=True)
        else:
            raise ValueError(
                f"method must be 'incremental' or 'elementwise', got {method!r}"
            )

        if aggregate_dims is not None:
            if isinstance(aggregate_dims, dict):
                aggregate_dims = [aggregate_dims]
            roas = self._apply_aggregate_dims(roas, aggregate_dims)

        # Compute summary stats with HDI
        df = self._compute_summary_stats_with_hdi(roas, hdi_probs)

        return self._convert_output(df, output_format)

    @staticmethod
    def _apply_aggregate_dims(
        data: xr.DataArray,
        aggregate_dims: list[dict[str, Any]],
    ) -> xr.DataArray:
        """Aggregate dimensions on an xr.DataArray via idata utilities.

        Converts the DataArray to an ``az.InferenceData`` object, applies
        :func:`~pymc_marketing.data.idata.utils.aggregate_idata_dims` for
        each entry, and extracts the result back.

        Parameters
        ----------
        data : xr.DataArray
            Data with ``(chain, draw, ...)`` dimensions.
        aggregate_dims : list[dict[str, Any]]
            List of aggregation specs. Each dict contains keyword arguments
            passed directly to ``aggregate_idata_dims`` (``dim``,
            ``values``, ``new_label``, optional ``method``). Specs are
            applied sequentially, so multiple aggregations on the same
            dimension are supported.

        Returns
        -------
        xr.DataArray
            Data with the requested dimensions aggregated.
        """
        var_name = "roas"
        idata = az.InferenceData(roas=data.to_dataset(name=var_name))
        for agg_spec in aggregate_dims:
            idata = aggregate_idata_dims(idata, **agg_spec)
        return idata.roas[var_name]

    def channel_spend(
        self,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Create channel spend DataFrame (raw data, no HDI).

        Returns the raw spend values per channel and date without
        any statistical aggregation.

        Parameters
        ----------
        output_format : {"pandas", "polars"}, optional
            Output DataFrame format (default: uses factory default)

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            DataFrame with columns:

            - date: Time index
            - channel: Channel name
            - channel_data: Spend value

        Examples
        --------
        >>> df = mmm.summary.channel_spend()
        >>> df = mmm.summary.channel_spend(output_format="polars")
        """
        effective_output_format = (
            output_format if output_format is not None else self.output_format
        )

        spend = self.data.get_channel_spend()
        df = spend.to_dataframe(name="channel_data").reset_index()

        return self._convert_output(df, effective_output_format)

    def saturation_curves(
        self,
        hdi_probs: Sequence[float] | None = None,
        output_format: OutputFormat | None = None,
        data: MMMIDataWrapper | None = None,
        max_value: float = 1.0,
        num_points: int = 100,
        num_samples: int | None = None,
        random_state: RandomState | None = None,
        original_scale: bool = True,
    ) -> DataFrameType:
        """Create saturation curves summary DataFrame.

        Samples saturation response curves from the posterior distribution
        using the model's sample_saturation_curve() method, then computes
        summary statistics (mean, median, HDI).

        Supports multi-dimensional data with custom_dims (e.g., country, region).
        When custom dimensions are present, curves are generated for each
        combination of channel and custom dimension values.

        Requires model to be provided (has saturation transformation).

        Parameters
        ----------
        hdi_probs : sequence of float, optional
            HDI probability levels (default: uses factory default)
        output_format : {"pandas", "polars"}, optional
            Output DataFrame format (default: uses factory default)
        data : MMMIDataWrapper or None, optional
            Optional data wrapper to use for sampling curves. If None (default),
            uses self.data. This allows sampling curves from a different
            InferenceData, such as from a subset of samples or another model.
        max_value : float, default 1.0
            Maximum value for the curve x-axis, in scaled space (consistent with
            model internals). This represents the maximum spend level in scaled
            units. To convert from original scale, divide by channel_scale:
            ``max_scaled = original_max / mmm.data.get_channel_scale().mean()``
        num_points : int, default 100
            Number of points between 0 and max_value to evaluate the curve at.
            Higher values give smoother curves but take longer.
        num_samples : int or None, optional
            Number of posterior samples to use for generating curves. By default
            None (use all posterior samples for accurate HDI). Using fewer samples
            speeds up computation and reduces memory usage while still capturing
            posterior uncertainty.
        random_state : int, np.random.Generator, or None, optional
            Random state for reproducible subsampling. Can be an integer seed,
            a numpy Generator instance, or None for non-reproducible sampling.
            Only used when num_samples is not None and less than total available
            samples.
        original_scale : bool, default True
            Whether to return curve y-values in original scale. If True (default),
            y-axis values (contribution) are multiplied by target_scale to convert
            from scaled to original units. If False, values remain in scaled space
            as used internally by the model.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Summary DataFrame with columns:

            - x: Input value (spend level, in scaled space)
            - channel: Channel name
            - <custom_dims>: One column for each custom dimension (e.g., country)
            - mean: Mean saturation response
            - median: Median saturation response
            - abs_error_{prob}_lower: HDI lower bound for each prob
            - abs_error_{prob}_upper: HDI upper bound for each prob

        Examples
        --------
        >>> df = mmm.summary.saturation_curves()
        >>> df = mmm.summary.saturation_curves(num_points=200)
        >>> df = mmm.summary.saturation_curves(hdi_probs=[0.80, 0.94])
        >>> df = mmm.summary.saturation_curves(max_value=2.0, num_samples=500)

        See Also
        --------
        MMM.sample_saturation_curve : Underlying method for sampling curves
        adstock_curves : For adstock curves
        """
        self._require_model("saturation_curves")
        # Type narrowing: _require_model ensures model is not None
        if self.model is None:
            raise RuntimeError(
                "Model should not be None after _require_model"
            )  # pragma: no cover
        model = self.model

        # Resolve defaults
        effective_hdi_probs = hdi_probs if hdi_probs is not None else self.hdi_probs
        effective_output_format = (
            output_format if output_format is not None else self.output_format
        )
        effective_data = data if data is not None else self.data

        # Validate HDI probs
        self._validate_hdi_probs(effective_hdi_probs)

        # Delegate to MMM.sample_saturation_curve()
        # Returns DataArray with dims: (x, channel, sample) or (x, *custom_dims, channel, sample)
        curve_samples = model.sample_saturation_curve(
            max_value=max_value,
            num_points=num_points,
            num_samples=num_samples,
            random_state=random_state,
            original_scale=original_scale,
            idata=effective_data.idata,
        )

        # Compute summary statistics across 'sample' dimension
        # The sample_saturation_curve returns DataArray with 'sample' dim
        df = self._compute_summary_stats_with_hdi(curve_samples, effective_hdi_probs)

        return self._convert_output(df, effective_output_format)

    def adstock_curves(
        self,
        hdi_probs: Sequence[float] | None = None,
        output_format: OutputFormat | None = None,
        data: MMMIDataWrapper | None = None,
        amount: float = 1.0,
        num_samples: int | None = None,
        random_state: RandomState | None = None,
    ) -> DataFrameType:
        """Create adstock curves summary DataFrame.

        Delegates to MMM.sample_adstock_curve() to sample adstock weight curves
        from the posterior distribution, then computes summary statistics
        (mean, median, HDI).

        Requires model to be provided (has adstock transformation).

        Parameters
        ----------
        hdi_probs : sequence of float, optional
            HDI probability levels (default: uses factory default)
        output_format : {"pandas", "polars"}, optional
            Output DataFrame format (default: uses factory default)
        data : MMMIDataWrapper or None, optional
            Optional data wrapper to use for sampling curves. If None (default),
            uses self.data. This allows sampling curves from a different
            InferenceData, such as from a subset of samples or another model.
        amount : float, default 1.0
            Amount to apply the adstock transformation to. This represents an
            impulse of spend at time 0, and the curve shows how this effect
            decays over subsequent time periods.
        num_samples : int or None, optional
            Number of posterior samples to use for generating curves. By default
            None (use all posterior samples for accurate HDI). Using fewer samples
            speeds up computation and reduces memory usage while still capturing
            posterior uncertainty.
        random_state : int, np.random.Generator, or None, optional
            Random state for reproducible subsampling. Can be an integer seed,
            a numpy Generator instance, or None for non-reproducible sampling.
            Only used when num_samples is not None and less than total available
            samples.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Summary DataFrame with columns:

            - time since exposure: Lag period (0 to l_max from the adstock transformation)
            - channel: Channel name
            - <custom_dims>: One column for each custom dimension (e.g., country)
            - mean: Mean adstock weight
            - median: Median adstock weight
            - abs_error_{prob}_lower: HDI lower bound for each prob
            - abs_error_{prob}_upper: HDI upper bound for each prob

        Examples
        --------
        >>> df = mmm.summary.adstock_curves()
        >>> df = mmm.summary.adstock_curves(amount=100.0)
        >>> df = mmm.summary.adstock_curves(hdi_probs=[0.80, 0.94])
        >>> df = mmm.summary.adstock_curves(num_samples=500, random_state=42)

        See Also
        --------
        MMM.sample_adstock_curve : Underlying method for sampling curves
        saturation_curves : For saturation curves
        """
        self._require_model("adstock_curves")
        # Type narrowing: _require_model ensures model is not None
        if self.model is None:
            raise RuntimeError(
                "Model should not be None after _require_model"
            )  # pragma: no cover
        model = self.model

        # Resolve defaults
        effective_hdi_probs = hdi_probs if hdi_probs is not None else self.hdi_probs
        effective_output_format = (
            output_format if output_format is not None else self.output_format
        )
        effective_data = data if data is not None else self.data

        # Validate HDI probs
        self._validate_hdi_probs(effective_hdi_probs)

        # Delegate to MMM.sample_adstock_curve()
        curve_samples = model.sample_adstock_curve(
            amount=amount,
            num_samples=num_samples,
            random_state=random_state,
            idata=effective_data.idata,
        )

        # Compute summary statistics across 'sample' dimension
        df = self._compute_summary_stats_with_hdi(curve_samples, effective_hdi_probs)

        return self._convert_output(df, effective_output_format)

    def total_contribution(
        self,
        hdi_probs: Sequence[float] | None = None,
        frequency: Frequency | None = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Create total contribution summary (all effects combined).

        Summarizes contributions by component type (channel, control, etc.),
        summing across individual components within each type.

        Parameters
        ----------
        hdi_probs : sequence of float, optional
            HDI probability levels (default: uses factory default)
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}, optional
            Time aggregation period (default: None, no aggregation)
        output_format : {"pandas", "polars"}, optional
            Output DataFrame format (default: uses factory default)

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Summary DataFrame with columns:

            - date: Time index
            - component: Effect type ("channel", "control", "seasonality", "baseline")
            - mean: Mean total contribution
            - median: Median total contribution
            - abs_error_{prob}_lower: HDI lower bound for each prob
            - abs_error_{prob}_upper: HDI upper bound for each prob

        Examples
        --------
        >>> df = mmm.summary.total_contribution()
        >>> df = mmm.summary.total_contribution(frequency="monthly")
        >>> df = mmm.summary.total_contribution(hdi_probs=[0.80, 0.94])

        See Also
        --------
        contributions : For per-channel/control contributions
        """
        # Resolve all defaults in one call
        data, hdi_probs, output_format = self._prepare_data_and_hdi(
            hdi_probs, frequency, output_format
        )

        # Get all contributions
        contributions = data.get_contributions(original_scale=True)

        all_dfs = []

        for component_name in contributions.data_vars:
            component_data = contributions[component_name]
            # Sum across the component dimension if present (e.g., sum across channels)
            component_dims = list(component_data.dims)
            sum_dims = [d for d in component_dims if d not in ["chain", "draw", "date"]]

            if sum_dims:
                summed_data = component_data.sum(dim=sum_dims)
            else:
                summed_data = component_data

            # Compute summary stats
            df = self._compute_summary_stats_with_hdi(summed_data, hdi_probs)
            df["component"] = component_name
            all_dfs.append(df)

        if not all_dfs:
            # Return empty DataFrame with correct schema
            return self._convert_output(
                pd.DataFrame(columns=["date", "component", "mean", "median"]),
                output_format,
            )

        result_df = pd.concat(all_dfs, ignore_index=True)
        return self._convert_output(result_df, output_format)

    def change_over_time(
        self,
        hdi_probs: Sequence[float] | None = None,
        frequency: Frequency | None = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Create change over time summary with per-date percentage changes.

        Computes percentage change in contributions between consecutive time periods:
        (value[t] - value[t-1]) / value[t-1] * 100 for each date.

        Parameters
        ----------
        hdi_probs : sequence of float, optional
            HDI probability levels (default: uses factory default)
        frequency : {"original", "weekly", "monthly", "quarterly", "yearly"}, optional
            Aggregate to time frequency before computing changes.
            Use "original" or None for no aggregation. Cannot use "all_time"
            (change over time requires date dimension).
        output_format : {"pandas", "polars"}, optional
            Output DataFrame format (default: uses factory default)

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Summary DataFrame with columns:

            - date: Date (excluding first date which has no previous)
            - channel: Channel name
            - pct_change_mean: Mean percentage change
            - pct_change_median: Median percentage change
            - abs_error_{prob}_lower: HDI lower bound for each prob
            - abs_error_{prob}_upper: HDI upper bound for each prob

        Raises
        ------
        ValueError
            If data has no date dimension (e.g., after "all_time" aggregation)

        Examples
        --------
        >>> df = mmm.summary.change_over_time()
        >>> df = mmm.summary.change_over_time(frequency="monthly")
        >>> df = mmm.summary.change_over_time(hdi_probs=[0.80, 0.94])
        """
        # Resolve all defaults in one call
        data, hdi_probs, output_format = self._prepare_data_and_hdi(
            hdi_probs, frequency, output_format
        )

        # Get contributions (chain, draw, date, channel)
        contributions = data.get_channel_contributions(original_scale=True)

        # Check for date dimension
        if "date" not in contributions.dims:
            raise ValueError(
                "change_over_time requires date dimension. "
                "Data may have been aggregated with frequency='all_time', "
                "which removes the date dimension. Use a different frequency "
                "or call on unaggregated data."
            )

        # Compute percentage change using xarray operations
        # Formula: (value[t] - value[t-1]) / value[t-1] * 100
        shifted = contributions.shift(date=1)
        diff = contributions.diff("date")

        # Handle division by zero (set to NaN)
        # Use xr.where to replace zeros with NaN before division
        shifted_safe = xr.where(shifted == 0, np.nan, shifted)
        pct_change = (diff / shifted_safe) * 100

        # Note: diff("date") already drops the first date (no previous value),
        # and xarray automatically aligns coordinates when dividing, so pct_change
        # will have dates[1:] (one fewer than input)

        # Compute summary statistics using existing helper
        df = self._compute_summary_stats_with_hdi(pct_change, hdi_probs)

        # Rename columns to match expected schema
        df = df.rename(
            columns={
                "mean": "pct_change_mean",
                "median": "pct_change_median",
            }
        )

        return self._convert_output(df, output_format)
