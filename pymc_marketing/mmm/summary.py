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
"""Summary DataFrame generation for MMM models (Component 2).

This module transforms MMMIDataWrapper (Component 1) into structured
DataFrames with summary statistics for plotting (Component 3).

Key Features:
- output_format parameter: Choose between Pandas and Polars DataFrames
- frequency parameter: View data at different aggregation levels
- HDI computation: Configurable probability levels for uncertainty

Examples
--------
>>> from pymc_marketing.mmm import MMM
>>> from pymc_marketing.mmm.summary import MMMSummaryFactory
>>>
>>> # Fit model
>>> mmm = MMM(...)
>>> mmm.fit(X, y)
>>>
>>> # Get summary as Pandas (default) - via model property
>>> df = mmm.summary.contributions()
>>>
>>> # Get summary as Polars - via factory directly
>>> factory = MMMSummaryFactory(mmm.data)
>>> df = factory.configure(output_format="polars").contributions()
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, Literal, ParamSpec, TypeVar, cast

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper

P = ParamSpec("P")
R = TypeVar("R")

try:
    import polars as pl

    POLARS_INSTALLED = True
except ImportError:
    POLARS_INSTALLED = False
    pl = None  # type: ignore[assignment]

# Type aliases
# Maps to Component 1's aggregate_time(period) - "original" means no aggregation
Frequency = Literal["original", "weekly", "monthly", "quarterly", "yearly", "all_time"]
OutputFormat = Literal["pandas", "polars"]

# Union type for return values
DataFrameType = pd.DataFrame  # Will be Union[pd.DataFrame, pl.DataFrame] at runtime


# ==================== Validation Decorator ====================


def validate_mmm_data(func: Callable[P, R]) -> Callable[P, R]:
    """Validate MMMIDataWrapper before function execution.

    Calls wrapper.validate_or_raise() to ensure idata structure is valid
    before processing. This catches missing variables or incorrect structure
    early with clear error messages.

    The first parameter of the decorated function must be a MMMIDataWrapper.

    Examples
    --------
    >>> @validate_mmm_data
    ... def create_summary(data: MMMIDataWrapper, ...):
    ...     # data is validated before this runs
    ...     contributions = data.get_contributions()
    ...     ...

    Notes
    -----
    For functions that can receive data without a date dimension (e.g., after
    "all_time" aggregation), validation still checks idata structure but
    functions should handle missing dimensions gracefully.
    """

    @wraps(func)
    def wrapper(data: Any, *args: P.args, **kwargs: P.kwargs) -> R:
        # Validate idata structure if schema is available
        if hasattr(data, "validate_or_raise") and data.schema is not None:
            data.validate_or_raise()

        return func(data, *args, **kwargs)

    return cast(Callable[P, R], wrapper)


# ==================== Internal Helper Functions ====================


def _validate_hdi_probs(hdi_probs: list[float]) -> None:
    """Validate HDI probability values are in valid range.

    Parameters
    ----------
    hdi_probs : list of float
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


def _convert_output(df: pd.DataFrame, output_format: OutputFormat) -> DataFrameType:
    """Convert Pandas DataFrame to requested output format.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (always Pandas from internal computation)
    output_format : {"pandas", "polars"}
        Desired output format

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
    if output_format == "pandas":
        return df
    elif output_format == "polars":
        if not POLARS_INSTALLED:
            raise ImportError(
                "Polars is required for output_format='polars'. "
                "Install it with: pip install pymc-marketing[polars]"
            )
        return pl.from_pandas(df)
    else:
        raise ValueError(
            f"Unknown output_format: {output_format!r}. Use 'pandas' or 'polars'."
        )


def _compute_summary_stats_with_hdi(
    data: xr.DataArray,
    hdi_probs: list[float],
    sample_dim: str | None = None,
) -> pd.DataFrame:
    """Convert xarray to DataFrame with summary stats and HDI.

    Core transformation function that:
    1. Computes mean and median across MCMC samples
    2. Computes HDI bounds for each probability level
    3. Returns structured DataFrame with absolute HDI bounds

    Parameters
    ----------
    data : xr.DataArray
        Must have 'chain' and 'draw' dimensions OR a single 'sample' dimension
    hdi_probs : list of float
        HDI probability levels (e.g., [0.80, 0.94])
    sample_dim : str, optional
        Name of the sample dimension. If None, assumes ['chain', 'draw'].
        Use 'sample' for curve data from sample_*_curve() methods.

    Returns
    -------
    pd.DataFrame
        With columns: <dimensions>, mean, median, abs_error_{prob}_lower, abs_error_{prob}_upper

    Notes
    -----
    - Always returns Pandas internally
    - Conversion to other formats happens at public API boundary
    - Uses az.stats.hdi() for HDI computation when data has chain/draw dims
    - Uses quantile-based HDI for data with sample dimension
    """
    # Determine sample dimensions
    if sample_dim is None:
        sample_dims = ["chain", "draw"]
        use_az_hdi = True
    else:
        sample_dims = [sample_dim]
        use_az_hdi = False

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
    data: MMMIDataWrapper,
    hdi_probs: list[float] | None,
    frequency: Frequency | None,
) -> tuple[MMMIDataWrapper, list[float]]:
    """Prepare data and HDI probs with defaults and aggregation.

    Parameters
    ----------
    data : MMMIDataWrapper
        Input data wrapper
    hdi_probs : list of float or None
        HDI probability levels (None uses default [0.94])
    frequency : Frequency or None
        Time aggregation period (None or "original" means no aggregation)

    Returns
    -------
    tuple[MMMIDataWrapper, list[float]]
        Prepared data and validated HDI probs
    """
    if hdi_probs is None:
        hdi_probs = [0.94]
    else:
        _validate_hdi_probs(hdi_probs)

    if frequency is not None and frequency != "original":
        data = data.aggregate_time(frequency)

    return data, hdi_probs


# ==================== Factory Functions ====================


@validate_mmm_data
def create_posterior_predictive_summary(
    data: MMMIDataWrapper,
    hdi_probs: list[float] | None = None,
    frequency: Frequency | None = None,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Create posterior predictive summary DataFrame.

    Computes mean, median, and HDI bounds for posterior predictive samples,
    along with observed values for comparison.

    Parameters
    ----------
    data : MMMIDataWrapper
        Data wrapper from Component 1
    hdi_probs : list of float, optional
        HDI probability levels (default: [0.94])
    frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}, optional
        Time aggregation period (default: None, no aggregation)
    output_format : {"pandas", "polars"}, default "pandas"
        Output DataFrame format

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
    """
    data, hdi_probs = _prepare_data_and_hdi(data, hdi_probs, frequency)

    # Get posterior predictive samples
    if hasattr(data.idata, "posterior_predictive"):
        pp_samples = data.idata.posterior_predictive["y"]
    else:
        raise AttributeError("No posterior predictive samples found in idata")

    # Get observed data
    observed = data.get_target(original_scale=True)

    # Compute summary stats with HDI
    df = _compute_summary_stats_with_hdi(pp_samples, hdi_probs)

    # Add observed values
    observed_df = observed.to_dataframe(name="observed").reset_index()
    merge_keys = ["date", *data.custom_dims]
    df = df.merge(observed_df, on=merge_keys, how="left")

    return _convert_output(df, output_format)


@validate_mmm_data
def create_channel_spend_dataframe(
    data: MMMIDataWrapper,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Create channel spend DataFrame (raw data, no HDI).

    Parameters
    ----------
    data : MMMIDataWrapper
        Data wrapper from Component 1
    output_format : {"pandas", "polars"}, default "pandas"
        Output DataFrame format

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        DataFrame with columns:
        - date: Time index
        - channel: Channel name
        - channel_data: Spend value
    """
    spend = data.get_channel_spend()
    df = spend.to_dataframe(name="channel_data").reset_index()

    return _convert_output(df, output_format)


def create_saturation_curves(
    model: Any,  # MMM type, avoid circular import
    hdi_probs: list[float] | None = None,
    n_points: int = 100,
    output_format: OutputFormat = "pandas",
    data: MMMIDataWrapper | None = None,
) -> DataFrameType:
    """Create saturation curves summary DataFrame.

    Samples saturation response curves from the posterior distribution
    using the model's sample_saturation_curve() method, then computes
    summary statistics (mean, median, HDI).

    Supports multi-dimensional data with custom_dims (e.g., country, region).
    When custom dimensions are present, curves are generated for each
    combination of channel and custom dimension values.

    Parameters
    ----------
    model : MMM
        Fitted MMM model with saturation transformation
    hdi_probs : list of float, optional
        HDI probability levels (default: [0.94])
    n_points : int, default 100
        Number of points to sample along the x-axis
    output_format : {"pandas", "polars"}, default "pandas"
        Output DataFrame format
    data : MMMIDataWrapper or None, optional
        Optional data wrapper to use for sampling curves. If None (default),
        uses model.data. This allows sampling curves from a different
        InferenceData, such as from a subset of samples or another model.

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
    >>> # Basic usage
    >>> df = create_saturation_curves(mmm)
    >>>
    >>> # With more resolution
    >>> df = create_saturation_curves(mmm, n_points=200)
    >>>
    >>> # Via factory
    >>> df = mmm.summary.saturation_curves(n_points=50)
    >>>
    >>> # With custom data wrapper
    >>> df = create_saturation_curves(mmm, data=custom_data_wrapper)

    See Also
    --------
    MMM.sample_saturation_curve : Underlying method for sampling curves
    create_adstock_curves : For adstock curves
    """
    if hdi_probs is None:
        hdi_probs = [0.94]
    else:
        _validate_hdi_probs(hdi_probs)

    # Use provided data wrapper or fall back to model.data
    if data is None:
        data = model.data

    # Determine max_value for x range based on channel spend
    # Use scaled space consistent with sample_saturation_curve()
    spend = data.get_channel_spend()
    max_spend = float(spend.max())

    # Get channel scale to convert to scaled space
    try:
        channel_scale = data.get_channel_scale()
        max_scaled = max_spend / float(channel_scale.mean())
    except (ValueError, AttributeError):
        # If no channel_scale, use max_spend as-is
        max_scaled = max_spend

    # Extend beyond max to show saturation behavior
    max_value = max_scaled * 1.5 if max_scaled > 0 else 1.0

    # Delegate to MMM.sample_saturation_curve()
    # Returns DataArray with dims: (x, channel, sample) or (x, *custom_dims, channel, sample)
    try:
        curve_samples = model.sample_saturation_curve(
            max_value=max_value,
            num_points=n_points,
            num_samples=None,  # Use all posterior samples for accurate HDI
            original_scale=True,  # Return in original scale
            idata=data.idata,  # Pass the idata from the data wrapper
        )
    except Exception as e:
        raise ValueError(f"Failed to sample saturation curves: {e}") from e

    # Compute summary statistics across 'sample' dimension
    # The sample_saturation_curve returns DataArray with 'sample' dim
    df = _compute_summary_stats_with_hdi(curve_samples, hdi_probs, sample_dim="sample")

    # Ensure proper column ordering
    # Detect custom dimensions from model
    custom_dims = list(model.dims) if hasattr(model, "dims") and model.dims else []

    # Standard columns: x, channel, [custom_dims...], mean, median, HDI columns
    index_cols = ["x", "channel", *custom_dims]
    stat_cols = ["mean", "median"] + [c for c in df.columns if "abs_error" in c]

    # Reorder columns (keeping any unexpected columns at the end)
    all_expected = index_cols + stat_cols
    other_cols = [c for c in df.columns if c not in all_expected]
    df = df[[c for c in all_expected if c in df.columns] + other_cols]

    return _convert_output(df, output_format)


def create_adstock_curves(
    model: Any,  # MMM type, avoid circular import
    hdi_probs: list[float] | None = None,
    max_lag: int = 20,
    output_format: OutputFormat = "pandas",
    data: MMMIDataWrapper | None = None,
) -> DataFrameType:
    """Create adstock curves summary DataFrame.

    Delegates to MMM.sample_adstock_curve() to sample adstock weight curves
    from the posterior distribution, then computes summary statistics.

    Parameters
    ----------
    model : MMM
        Fitted MMM model with adstock transformation
    hdi_probs : list of float, optional
        HDI probability levels (default: [0.94])
    max_lag : int, default 20
        Maximum lag periods to include in output
    output_format : {"pandas", "polars"}, default "pandas"
        Output DataFrame format
    data : MMMIDataWrapper or None, optional
        Optional data wrapper to use for sampling curves. If None (default),
        uses model.data. This allows sampling curves from a different
        InferenceData, such as from a subset of samples or another model.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Summary DataFrame with columns:
        - time: Lag period (0 to max_lag)
        - channel: Channel name
        - <custom_dims>: One column for each custom dimension (e.g., country)
        - mean: Mean adstock weight
        - median: Median adstock weight
        - abs_error_{prob}_lower: HDI lower bound for each prob
        - abs_error_{prob}_upper: HDI upper bound for each prob

    Examples
    --------
    >>> # Basic usage
    >>> df = create_adstock_curves(mmm)
    >>>
    >>> # With custom data wrapper
    >>> df = create_adstock_curves(mmm, data=custom_data_wrapper)

    See Also
    --------
    MMM.sample_adstock_curve : Underlying method for sampling curves
    create_saturation_curves : For saturation curves
    """
    if hdi_probs is None:
        hdi_probs = [0.94]
    else:
        _validate_hdi_probs(hdi_probs)

    # Use provided data wrapper or fall back to model.data
    if data is None:
        data = model.data

    # Delegate to MMM.sample_adstock_curve()
    try:
        curve_samples = model.sample_adstock_curve(
            amount=1.0,
            num_samples=None,  # Use all posterior samples for accurate HDI
            idata=data.idata,  # Pass the idata from the data wrapper
        )
    except Exception as e:
        raise ValueError(f"Failed to sample adstock curves: {e}") from e

    # Rename time dimension to 'time' for output
    # Handle both "time_since_exposure" and "time since exposure" (with spaces)
    for dim_name in ["time_since_exposure", "time since exposure"]:
        if dim_name in curve_samples.dims:
            curve_samples = curve_samples.rename({dim_name: "time"})
            break

    # Trim to max_lag if needed
    if max_lag is not None and "time" in curve_samples.dims:
        actual_max_lag = min(max_lag, len(curve_samples["time"]) - 1)
        curve_samples = curve_samples.isel(time=slice(0, actual_max_lag + 1))

    # Compute summary statistics across 'sample' dimension
    df = _compute_summary_stats_with_hdi(curve_samples, hdi_probs, sample_dim="sample")

    # Ensure proper column ordering
    custom_dims = list(model.dims) if hasattr(model, "dims") and model.dims else []
    index_cols = ["time", "channel", *custom_dims]
    stat_cols = ["mean", "median"] + [c for c in df.columns if "abs_error" in c]

    all_expected = index_cols + stat_cols
    other_cols = [c for c in df.columns if c not in all_expected]
    df = df[[c for c in all_expected if c in df.columns] + other_cols]

    return _convert_output(df, output_format)


@validate_mmm_data
def create_total_contribution_summary(
    data: MMMIDataWrapper,
    hdi_probs: list[float] | None = None,
    frequency: Frequency | None = None,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Create total contribution summary (all effects combined).

    Summarizes contributions by component type (channel, control, etc.).

    Parameters
    ----------
    data : MMMIDataWrapper
        Data wrapper from Component 1
    hdi_probs : list of float, optional
        HDI probability levels (default: [0.94])
    frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}, optional
        Time aggregation period (default: None, no aggregation)
    output_format : {"pandas", "polars"}, default "pandas"
        Output DataFrame format

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
    """
    data, hdi_probs = _prepare_data_and_hdi(data, hdi_probs, frequency)

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
        df = _compute_summary_stats_with_hdi(summed_data, hdi_probs)
        df["component"] = component_name
        all_dfs.append(df)

    if not all_dfs:
        # Return empty DataFrame with correct schema
        return _convert_output(
            pd.DataFrame(columns=["date", "component", "mean", "median"]),
            output_format,
        )

    result_df = pd.concat(all_dfs, ignore_index=True)
    return _convert_output(result_df, output_format)


@validate_mmm_data
def create_change_over_time_summary(
    data: MMMIDataWrapper,
    hdi_probs: list[float] | None = None,
    frequency: Frequency | None = None,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Create change over time summary with per-date percentage changes.

    Computes percentage change in contributions between consecutive time periods:
    (value[t] - value[t-1]) / value[t-1] * 100 for each date.

    Parameters
    ----------
    data : MMMIDataWrapper
        Data wrapper from Component 1
    hdi_probs : list of float, optional
        HDI probability levels (default: [0.94])
    frequency : {"original", "weekly", "monthly", "quarterly", "yearly"}, optional
        Aggregate to time frequency before computing changes.
        Use "original" or None for no aggregation. Cannot use "all_time"
        (change over time requires date dimension).
    output_format : {"pandas", "polars"}, default "pandas"
        Output DataFrame format

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
    >>> # Basic usage
    >>> df = create_change_over_time_summary(mmm.data)
    >>>
    >>> # With monthly aggregation first
    >>> df = create_change_over_time_summary(mmm.data, frequency="monthly")
    >>>
    >>> # Via factory
    >>> df = mmm.summary.change_over_time()
    """
    # Validate and prepare
    data, hdi_probs = _prepare_data_and_hdi(data, hdi_probs, frequency)

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
    df = _compute_summary_stats_with_hdi(pct_change, hdi_probs)

    # Rename columns to match expected schema
    df = df.rename(
        columns={
            "mean": "pct_change_mean",
            "median": "pct_change_median",
        }
    )

    return _convert_output(df, output_format)


# ==================== Factory Class ====================


class MMMSummaryFactory:
    """Factory for creating summary DataFrames from MMM data.

    Provides a convenient interface for generating summary DataFrames
    with shared default settings. Accepts data wrapper (required) and
    optionally the MMM model to access transformations.

    The factory is immutable - use :meth:`configure` to create a new
    factory with different settings.

    Parameters
    ----------
    data : MMMIDataWrapper
        Data wrapper containing idata and schema (required)
    model : MMM, optional
        Fitted MMM model with transformations (saturation, adstock).
        Required for saturation_curves() and adstock_curves() methods.
    hdi_probs : list of float, optional
        Default HDI probability levels (default: [0.94])
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
    >>> # Configure factory with new defaults (returns new instance)
    >>> polars_factory = factory.configure(
    ...     output_format="polars", hdi_probs=[0.80, 0.94]
    ... )
    >>> df = polars_factory.contributions()  # Uses configured defaults
    """

    def __init__(
        self,
        data: MMMIDataWrapper,
        model: Any | None = None,  # MMM type, but avoid circular import
        hdi_probs: list[float] | None = None,
        output_format: OutputFormat = "pandas",
    ):
        # Validate data structure at initialization (early fail)
        if hasattr(data, "validate_or_raise") and data.schema is not None:
            data.validate_or_raise()

        self._data = data
        self._model = model
        self._hdi_probs = hdi_probs if hdi_probs is not None else [0.94]
        self._output_format = output_format

        # Validate HDI probs at init time (uses class method)
        self._validate_hdi_probs(self._hdi_probs)

    # ==================== Private Helper Methods ====================

    def _validate_hdi_probs(self, hdi_probs: list[float]) -> None:
        """Validate HDI probability values are in valid range.

        Parameters
        ----------
        hdi_probs : list of float
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
        fmt = output_format if output_format is not None else self._output_format
        if fmt == "pandas":
            return df
        elif fmt == "polars":
            if not POLARS_INSTALLED:
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
        hdi_probs: list[float],
        sample_dim: str | None = None,
    ) -> pd.DataFrame:
        """Convert xarray to DataFrame with summary stats and HDI.

        Core transformation function that:
        1. Computes mean and median across MCMC samples
        2. Computes HDI bounds for each probability level
        3. Returns structured DataFrame with absolute HDI bounds

        Parameters
        ----------
        data : xr.DataArray
            Must have 'chain' and 'draw' dimensions OR a single 'sample' dimension
        hdi_probs : list of float
            HDI probability levels (e.g., [0.80, 0.94])
        sample_dim : str, optional
            Name of the sample dimension. If None, assumes ['chain', 'draw'].
            Use 'sample' for curve data from sample_*_curve() methods.

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
        # Determine sample dimensions
        if sample_dim is None:
            sample_dims = ["chain", "draw"]
            use_az_hdi = True
        else:
            sample_dims = [sample_dim]
            use_az_hdi = False

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
        hdi_probs: list[float] | None = None,
        frequency: Frequency | None = None,
        output_format: OutputFormat | None = None,
    ) -> tuple[MMMIDataWrapper, list[float], OutputFormat]:
        """Prepare data, resolve defaults, and validate.

        This is the main "resolve defaults" method that should be called
        at the start of each public summary method. It:
        1. Resolves hdi_probs default from self._hdi_probs
        2. Resolves output_format default from self._output_format
        3. Validates hdi_probs
        4. Aggregates data by frequency if specified

        Parameters
        ----------
        hdi_probs : list of float or None
            HDI probability levels (None uses factory default)
        frequency : Frequency or None
            Time aggregation period (None or "original" means no aggregation)
        output_format : OutputFormat or None
            Output format (None uses factory default)

        Returns
        -------
        tuple[MMMIDataWrapper, list[float], OutputFormat]
            (prepared_data, effective_hdi_probs, effective_output_format)
        """
        effective_hdi_probs = hdi_probs if hdi_probs is not None else self._hdi_probs
        effective_output_format = (
            output_format if output_format is not None else self._output_format
        )

        self._validate_hdi_probs(effective_hdi_probs)

        data = self._data
        if frequency is not None and frequency != "original":
            data = data.aggregate_time(frequency)

        return data, effective_hdi_probs, effective_output_format

    @property
    def data(self) -> MMMIDataWrapper:
        """Data wrapper containing idata and schema."""
        return self._data

    @property
    def model(self) -> Any | None:
        """Fitted MMM model (None if not provided)."""
        return self._model

    @property
    def hdi_probs(self) -> list[float]:
        """Default HDI probability levels."""
        return self._hdi_probs

    @property
    def output_format(self) -> OutputFormat:
        """Default output DataFrame format."""
        return self._output_format

    def configure(
        self,
        hdi_probs: list[float] | None = None,
        output_format: OutputFormat | None = None,
    ) -> MMMSummaryFactory:
        """Create a new factory with updated configuration.

        Returns a new MMMSummaryFactory instance with the specified settings,
        keeping all other settings from this factory. This allows for an
        immutable configuration pattern.

        Parameters
        ----------
        hdi_probs : list of float, optional
            New HDI probability levels. If None, keeps current setting.
        output_format : {"pandas", "polars"}, optional
            New output DataFrame format. If None, keeps current setting.

        Returns
        -------
        MMMSummaryFactory
            New factory instance with updated configuration

        Examples
        --------
        >>> # Start with default factory
        >>> factory = mmm.summary
        >>>
        >>> # Create new factory with polars output
        >>> polars_factory = factory.configure(output_format="polars")
        >>>
        >>> # Create new factory with custom HDI and polars
        >>> custom_factory = factory.configure(
        ...     hdi_probs=[0.80, 0.94], output_format="polars"
        ... )
        >>>
        >>> # Chain configurations
        >>> result = mmm.summary.configure(output_format="polars").contributions()
        """
        return MMMSummaryFactory(
            data=self._data,
            model=self._model,
            hdi_probs=hdi_probs if hdi_probs is not None else self._hdi_probs,
            output_format=output_format
            if output_format is not None
            else self._output_format,
        )

    def _require_model(self, method_name: str) -> None:
        """Raise helpful error if model is required but not provided."""
        if self.model is None:
            raise ValueError(
                f"{method_name} requires model to access transformations. "
                f"Use MMMSummaryFactory(data, model=mmm) or mmm.summary instead."
            )

    def posterior_predictive(
        self,
        hdi_probs: list[float] | None = None,
        frequency: Frequency | None = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Create posterior predictive summary DataFrame.

        Computes mean, median, and HDI bounds for posterior predictive samples,
        along with observed values for comparison.

        Parameters
        ----------
        hdi_probs : list of float, optional
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

        See Also
        --------
        create_posterior_predictive_summary : Standalone function
        """
        return create_posterior_predictive_summary(
            self.data,
            hdi_probs=hdi_probs if hdi_probs is not None else self.hdi_probs,
            frequency=frequency,
            output_format=output_format
            if output_format is not None
            else self.output_format,
        )

    def contributions(
        self,
        hdi_probs: list[float] | None = None,
        component: Literal["channel", "control", "seasonality", "baseline"] = "channel",
        frequency: Frequency | None = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Create contribution summary DataFrame.

        Computes mean, median, and HDI bounds for contribution samples
        for the specified component type.

        Parameters
        ----------
        hdi_probs : list of float, optional
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

        # Get contributions via Component 1 (handles scaling)
        if component == "channel":
            component_data = data.get_channel_contributions(original_scale=True)
        else:
            contributions = data.get_contributions(
                original_scale=True,
                include_baseline=(component == "baseline"),
                include_controls=(component == "control"),
                include_seasonality=(component == "seasonality"),
            )

            if component not in contributions:
                raise ValueError(f"No {component} contributions found in model")
            component_data = contributions[component]

        # Compute summary stats with HDI
        df = self._compute_summary_stats_with_hdi(component_data, hdi_probs)

        return self._convert_output(df, output_format)

    def roas(
        self,
        hdi_probs: list[float] | None = None,
        frequency: Frequency | None = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Create ROAS (Return on Ad Spend) summary DataFrame.

        Computes ROAS = contribution / spend for each channel with
        mean, median, and HDI bounds.

        Parameters
        ----------
        hdi_probs : list of float, optional
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
            - channel: Channel name
            - mean: Mean ROAS
            - median: Median ROAS
            - abs_error_{prob}_lower: HDI lower bound for each prob
            - abs_error_{prob}_upper: HDI upper bound for each prob

        Examples
        --------
        >>> df = mmm.summary.roas()
        >>> df = mmm.summary.roas(frequency="monthly")
        >>> df = mmm.summary.roas(hdi_probs=[0.80, 0.94])

        See Also
        --------
        create_roas_summary : Standalone function
        """
        # Resolve all defaults in one call
        data, hdi_probs, output_format = self._prepare_data_and_hdi(
            hdi_probs, frequency, output_format
        )

        # Get channel contributions and spend
        contributions = data.get_channel_contributions(original_scale=True)
        spend = data.get_channel_spend()

        # Compute ROAS = contribution / spend
        # Need to broadcast spend to match contributions dimensions
        # spend has dims (date, channel), contributions has (chain, draw, date, channel)
        # xarray handles broadcasting automatically

        # Handle zero spend - use xr.where to avoid division by zero
        spend_with_epsilon = xr.where(spend == 0, np.nan, spend)
        roas = contributions / spend_with_epsilon

        # Compute summary stats with HDI
        df = self._compute_summary_stats_with_hdi(roas, hdi_probs)

        return self._convert_output(df, output_format)

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

        See Also
        --------
        create_channel_spend_dataframe : Standalone function
        """
        return create_channel_spend_dataframe(
            self.data,
            output_format=output_format
            if output_format is not None
            else self.output_format,
        )

    def saturation_curves(
        self,
        hdi_probs: list[float] | None = None,
        n_points: int = 100,
        output_format: OutputFormat | None = None,
        data: MMMIDataWrapper | None = None,
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
        hdi_probs : list of float, optional
            HDI probability levels (default: uses factory default)
        n_points : int, default 100
            Number of points to sample along the x-axis
        output_format : {"pandas", "polars"}, optional
            Output DataFrame format (default: uses factory default)
        data : MMMIDataWrapper or None, optional
            Optional data wrapper to use for sampling curves. If None (default),
            uses self.data. This allows sampling curves from a different
            InferenceData, such as from a subset of samples or another model.

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
        >>> df = mmm.summary.saturation_curves(n_points=200)
        >>> df = mmm.summary.saturation_curves(hdi_probs=[0.80, 0.94])

        See Also
        --------
        create_saturation_curves : Standalone function
        MMM.sample_saturation_curve : Underlying method for sampling curves
        adstock_curves : For adstock curves
        """
        self._require_model("saturation_curves")
        return create_saturation_curves(
            self.model,
            hdi_probs=hdi_probs if hdi_probs is not None else self.hdi_probs,
            n_points=n_points,
            output_format=output_format
            if output_format is not None
            else self.output_format,
            data=data if data is not None else self.data,
        )

    def adstock_curves(
        self,
        hdi_probs: list[float] | None = None,
        max_lag: int = 20,
        output_format: OutputFormat | None = None,
        data: MMMIDataWrapper | None = None,
    ) -> DataFrameType:
        """Create adstock curves summary DataFrame.

        Delegates to MMM.sample_adstock_curve() to sample adstock weight curves
        from the posterior distribution, then computes summary statistics
        (mean, median, HDI).

        Requires model to be provided (has adstock transformation).

        Parameters
        ----------
        hdi_probs : list of float, optional
            HDI probability levels (default: uses factory default)
        max_lag : int, default 20
            Maximum lag periods to include in output
        output_format : {"pandas", "polars"}, optional
            Output DataFrame format (default: uses factory default)
        data : MMMIDataWrapper or None, optional
            Optional data wrapper to use for sampling curves. If None (default),
            uses self.data. This allows sampling curves from a different
            InferenceData, such as from a subset of samples or another model.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Summary DataFrame with columns:

            - time: Lag period (0 to max_lag)
            - channel: Channel name
            - <custom_dims>: One column for each custom dimension (e.g., country)
            - mean: Mean adstock weight
            - median: Median adstock weight
            - abs_error_{prob}_lower: HDI lower bound for each prob
            - abs_error_{prob}_upper: HDI upper bound for each prob

        Examples
        --------
        >>> df = mmm.summary.adstock_curves()
        >>> df = mmm.summary.adstock_curves(max_lag=30)
        >>> df = mmm.summary.adstock_curves(hdi_probs=[0.80, 0.94])

        See Also
        --------
        create_adstock_curves : Standalone function
        MMM.sample_adstock_curve : Underlying method for sampling curves
        saturation_curves : For saturation curves
        """
        self._require_model("adstock_curves")
        return create_adstock_curves(
            self.model,
            hdi_probs=hdi_probs if hdi_probs is not None else self.hdi_probs,
            max_lag=max_lag,
            output_format=output_format
            if output_format is not None
            else self.output_format,
            data=data if data is not None else self.data,
        )

    def total_contribution(
        self,
        hdi_probs: list[float] | None = None,
        frequency: Frequency | None = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Create total contribution summary (all effects combined).

        Summarizes contributions by component type (channel, control, etc.),
        summing across individual components within each type.

        Parameters
        ----------
        hdi_probs : list of float, optional
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
        create_total_contribution_summary : Standalone function
        contributions : For per-channel/control contributions
        """
        return create_total_contribution_summary(
            self.data,
            hdi_probs=hdi_probs if hdi_probs is not None else self.hdi_probs,
            frequency=frequency,
            output_format=output_format
            if output_format is not None
            else self.output_format,
        )

    def change_over_time(
        self,
        hdi_probs: list[float] | None = None,
        frequency: Frequency | None = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Create change over time summary with per-date percentage changes.

        Computes percentage change in contributions between consecutive time periods:
        (value[t] - value[t-1]) / value[t-1] * 100 for each date.

        Parameters
        ----------
        hdi_probs : list of float, optional
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

        See Also
        --------
        create_change_over_time_summary : Standalone function
        """
        return create_change_over_time_summary(
            self.data,
            hdi_probs=hdi_probs if hdi_probs is not None else self.hdi_probs,
            frequency=frequency,
            output_format=output_format
            if output_format is not None
            else self.output_format,
        )
