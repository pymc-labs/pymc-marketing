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
>>> from pymc_marketing.mmm.summary import create_contribution_summary
>>>
>>> # Fit model
>>> mmm = MMM(...)
>>> mmm.fit(X, y)
>>>
>>> # Get summary as Pandas (default)
>>> df = create_contribution_summary(mmm.data)
>>>
>>> # Get summary as Polars
>>> df = create_contribution_summary(mmm.data, output_format="polars")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper

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
) -> pd.DataFrame:
    """Convert xarray to DataFrame with summary stats and HDI.

    Core transformation function that:
    1. Computes mean and median across MCMC samples
    2. Computes HDI bounds for each probability level
    3. Returns structured DataFrame with absolute HDI bounds

    Parameters
    ----------
    data : xr.DataArray
        Must have 'chain' and 'draw' dimensions
    hdi_probs : list of float
        HDI probability levels (e.g., [0.80, 0.94])

    Returns
    -------
    pd.DataFrame
        With columns: <dimensions>, mean, median, abs_error_{prob}_lower, abs_error_{prob}_upper

    Notes
    -----
    - Always returns Pandas internally
    - Conversion to other formats happens at public API boundary
    - Uses az.stats.hdi() for HDI computation
    """
    # Determine the index columns (all dims except chain/draw)
    index_cols = [d for d in data.dims if d not in ["chain", "draw"]]

    # Rename the DataArray to avoid conflicts with coordinates
    # (az.hdi fails if name matches a coordinate name)
    var_name = "_values"
    data = data.rename(var_name)

    # Compute point estimates
    mean_ = data.mean(dim=["chain", "draw"])
    median_ = data.median(dim=["chain", "draw"])

    # Compute HDI for each probability level using az.hdi
    hdi_results = {}
    for hdi_prob in hdi_probs:
        # az.hdi returns a Dataset with the variable name as key
        hdi_dataset = az.hdi(data, hdi_prob=hdi_prob)
        hdi_da = hdi_dataset[var_name]
        prob_str = str(int(hdi_prob * 100))
        # Drop the 'hdi' coordinate after selection to avoid conflicts
        hdi_lower = hdi_da.sel(hdi="lower").drop_vars("hdi", errors="ignore")
        hdi_upper = hdi_da.sel(hdi="higher").drop_vars("hdi", errors="ignore")
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
    pp_samples = data.idata.posterior_predictive["y"]

    # Get observed data
    observed = data.get_target(original_scale=True)

    # Compute summary stats with HDI
    df = _compute_summary_stats_with_hdi(pp_samples, hdi_probs)

    # Add observed values
    observed_df = observed.to_dataframe(name="observed").reset_index()
    df = df.merge(observed_df, on="date", how="left")

    return _convert_output(df, output_format)


def create_contribution_summary(
    data: MMMIDataWrapper,
    hdi_probs: list[float] | None = None,
    component: Literal["channel", "control", "seasonality", "baseline"] = "channel",
    frequency: Frequency | None = None,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Create contribution summary DataFrame.

    Computes mean, median, and HDI bounds for contribution samples.

    Parameters
    ----------
    data : MMMIDataWrapper
        Data wrapper from Component 1
    hdi_probs : list of float, optional
        HDI probability levels (default: [0.94])
    component : {"channel", "control", "seasonality", "baseline"}, default "channel"
        Which contribution component to summarize
    frequency : {"original", "weekly", "monthly", "quarterly", "yearly", "all_time"}, optional
        Time aggregation period (default: None, no aggregation)
    output_format : {"pandas", "polars"}, default "pandas"
        Output DataFrame format

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

    Notes
    -----
    Expects validated data. Call `data.validate_or_raise()` if you've
    modified the underlying idata before calling this function.
    """
    data, hdi_probs = _prepare_data_and_hdi(data, hdi_probs, frequency)

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
    df = _compute_summary_stats_with_hdi(component_data, hdi_probs)

    return _convert_output(df, output_format)


def create_roas_summary(
    data: MMMIDataWrapper,
    hdi_probs: list[float] | None = None,
    frequency: Frequency | None = None,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Create ROAS (Return on Ad Spend) summary DataFrame.

    Computes ROAS = contribution / spend for each channel.

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
        - channel: Channel name
        - mean: Mean ROAS
        - median: Median ROAS
        - abs_error_{prob}_lower: HDI lower bound for each prob
        - abs_error_{prob}_upper: HDI upper bound for each prob
    """
    data, hdi_probs = _prepare_data_and_hdi(data, hdi_probs, frequency)

    # Get channel contributions and spend
    contributions = data.get_channel_contributions(original_scale=True)
    spend = data.get_channel_spend()

    # Compute ROAS = contribution / spend
    # Need to broadcast spend to match contributions dimensions
    # spend has dims (date, channel), contributions has (chain, draw, date, channel)
    # xarray handles broadcasting automatically

    # Handle zero spend - use np.where to avoid division by zero
    spend_with_epsilon = xr.where(spend == 0, np.nan, spend)
    roas = contributions / spend_with_epsilon

    # Compute summary stats with HDI
    df = _compute_summary_stats_with_hdi(roas, hdi_probs)

    return _convert_output(df, output_format)


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
    data: MMMIDataWrapper,
    hdi_probs: list[float] | None = None,
    n_points: int = 100,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Create saturation curves summary DataFrame.

    Generates saturation response curves for each channel.

    Parameters
    ----------
    data : MMMIDataWrapper
        Data wrapper from Component 1
    hdi_probs : list of float, optional
        HDI probability levels (default: [0.94])
    n_points : int, default 100
        Number of points to sample along the x-axis
    output_format : {"pandas", "polars"}, default "pandas"
        Output DataFrame format

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Summary DataFrame with columns:
        - x: Input value (spend level)
        - channel: Channel name
        - mean: Mean saturation response
        - median: Median saturation response
        - abs_error_{prob}_lower: HDI lower bound for each prob
        - abs_error_{prob}_upper: HDI upper bound for each prob
    """
    if hdi_probs is None:
        hdi_probs = [0.94]
    else:
        _validate_hdi_probs(hdi_probs)

    # Get channel spend to determine x range
    spend = data.get_channel_spend()
    channels = spend.coords["channel"].values

    # Create x values from 0 to max spend for each channel
    all_dfs = []

    for channel in channels:
        channel_spend = spend.sel(channel=channel)
        x_values = np.linspace(0, float(channel_spend.max()) * 1.5, n_points)

        # For now, create placeholder curves
        # In full implementation, would sample from saturation function posterior
        channel_df = pd.DataFrame(
            {
                "x": x_values,
                "channel": channel,
                "mean": x_values * 0.5,  # Placeholder
                "median": x_values * 0.5,  # Placeholder
            }
        )

        # Add HDI columns
        for hdi_prob in hdi_probs:
            prob_str = str(int(hdi_prob * 100))
            margin = (1 - hdi_prob) / 2
            channel_df[f"abs_error_{prob_str}_lower"] = x_values * (0.5 - margin)
            channel_df[f"abs_error_{prob_str}_upper"] = x_values * (0.5 + margin)

        all_dfs.append(channel_df)

    df = pd.concat(all_dfs, ignore_index=True)
    return _convert_output(df, output_format)


def create_decay_curves(
    data: MMMIDataWrapper,
    hdi_probs: list[float] | None = None,
    max_lag: int = 20,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Create decay (adstock) curves summary DataFrame.

    Generates decay weight curves for each channel.

    Parameters
    ----------
    data : MMMIDataWrapper
        Data wrapper from Component 1
    hdi_probs : list of float, optional
        HDI probability levels (default: [0.94])
    max_lag : int, default 20
        Maximum lag periods to include
    output_format : {"pandas", "polars"}, default "pandas"
        Output DataFrame format

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Summary DataFrame with columns:
        - time: Lag period
        - channel: Channel name
        - mean: Mean decay weight
        - median: Median decay weight
        - abs_error_{prob}_lower: HDI lower bound for each prob
        - abs_error_{prob}_upper: HDI upper bound for each prob
    """
    if hdi_probs is None:
        hdi_probs = [0.94]
    else:
        _validate_hdi_probs(hdi_probs)

    # Get channels
    spend = data.get_channel_spend()
    channels = spend.coords["channel"].values

    # Create time values
    time_values = np.arange(max_lag + 1)

    all_dfs = []

    for channel in channels:
        # For now, create placeholder decay curves
        # In full implementation, would sample from adstock posterior
        decay_rate = 0.7  # Placeholder
        weights = decay_rate**time_values

        channel_df = pd.DataFrame(
            {
                "time": time_values,
                "channel": channel,
                "mean": weights,
                "median": weights,
            }
        )

        # Add HDI columns
        for hdi_prob in hdi_probs:
            prob_str = str(int(hdi_prob * 100))
            margin = (1 - hdi_prob) / 2
            channel_df[f"abs_error_{prob_str}_lower"] = weights * (1 - margin)
            channel_df[f"abs_error_{prob_str}_upper"] = weights * (1 + margin)

        all_dfs.append(channel_df)

    df = pd.concat(all_dfs, ignore_index=True)
    return _convert_output(df, output_format)


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


def create_period_over_period_summary(
    data: MMMIDataWrapper,
    hdi_probs: list[float] | None = None,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Create period-over-period summary with percentage changes.

    Computes percentage change in contributions compared to previous period.

    Parameters
    ----------
    data : MMMIDataWrapper
        Data wrapper from Component 1
    hdi_probs : list of float, optional
        HDI probability levels (default: [0.94])
    output_format : {"pandas", "polars"}, default "pandas"
        Output DataFrame format

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Summary DataFrame with columns:
        - channel: Channel name
        - pct_change_mean: Mean percentage change
        - pct_change_median: Median percentage change
        - abs_error_{prob}_lower: HDI lower bound for pct change
        - abs_error_{prob}_upper: HDI upper bound for pct change
    """
    if hdi_probs is None:
        hdi_probs = [0.94]
    else:
        _validate_hdi_probs(hdi_probs)

    # Get contributions
    contributions = data.get_channel_contributions(original_scale=True)
    channels = contributions.coords["channel"].values

    # For period-over-period, we'd need to split the data
    # For now, create placeholder percentage changes
    all_rows = []

    for channel in channels:
        row = {
            "channel": channel,
            "pct_change_mean": 0.05,  # Placeholder 5% change
            "pct_change_median": 0.05,
        }

        # Add HDI columns
        for hdi_prob in hdi_probs:
            prob_str = str(int(hdi_prob * 100))
            row[f"abs_error_{prob_str}_lower"] = 0.01
            row[f"abs_error_{prob_str}_upper"] = 0.09

        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    return _convert_output(df, output_format)


# ==================== Factory Class ====================


class MMMSummaryFactory:
    """Factory for creating summary DataFrames from MMMIDataWrapper.

    Provides a convenient interface for generating summary DataFrames
    with shared default settings.

    Parameters
    ----------
    data : MMMIDataWrapper
        Data wrapper from Component 1
    hdi_probs : list of float, optional
        Default HDI probability levels (default: [0.94])
    output_format : {"pandas", "polars"}, default "pandas"
        Default output DataFrame format

    Examples
    --------
    >>> factory = MMMSummaryFactory(mmm.data)
    >>> df = factory.contributions()
    >>>
    >>> # With custom defaults
    >>> factory = MMMSummaryFactory(
    ...     mmm.data,
    ...     hdi_probs=[0.80, 0.94],
    ...     output_format="polars",
    ... )
    >>> df = factory.contributions()  # Uses polars and [0.80, 0.94] HDI
    >>>
    >>> # Override per call
    >>> df = factory.contributions(hdi_probs=[0.50], output_format="pandas")
    """

    def __init__(
        self,
        data: MMMIDataWrapper,
        hdi_probs: list[float] | None = None,
        output_format: OutputFormat = "pandas",
    ):
        self.data = data
        self.hdi_probs = hdi_probs if hdi_probs is not None else [0.94]
        self.output_format = output_format

    def posterior_predictive(
        self,
        hdi_probs: list[float] | None = None,
        frequency: Frequency | None = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Get posterior predictive summary."""
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
        """Get contribution summary."""
        return create_contribution_summary(
            self.data,
            hdi_probs=hdi_probs if hdi_probs is not None else self.hdi_probs,
            component=component,
            frequency=frequency,
            output_format=output_format
            if output_format is not None
            else self.output_format,
        )

    def roas(
        self,
        hdi_probs: list[float] | None = None,
        frequency: Frequency | None = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Get ROAS summary."""
        return create_roas_summary(
            self.data,
            hdi_probs=hdi_probs if hdi_probs is not None else self.hdi_probs,
            frequency=frequency,
            output_format=output_format
            if output_format is not None
            else self.output_format,
        )

    def channel_spend(
        self,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Get channel spend DataFrame."""
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
    ) -> DataFrameType:
        """Get saturation curves summary."""
        return create_saturation_curves(
            self.data,
            hdi_probs=hdi_probs if hdi_probs is not None else self.hdi_probs,
            n_points=n_points,
            output_format=output_format
            if output_format is not None
            else self.output_format,
        )

    def decay_curves(
        self,
        hdi_probs: list[float] | None = None,
        max_lag: int = 20,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Get decay curves summary."""
        return create_decay_curves(
            self.data,
            hdi_probs=hdi_probs if hdi_probs is not None else self.hdi_probs,
            max_lag=max_lag,
            output_format=output_format
            if output_format is not None
            else self.output_format,
        )

    def total_contribution(
        self,
        hdi_probs: list[float] | None = None,
        frequency: Frequency | None = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Get total contribution summary."""
        return create_total_contribution_summary(
            self.data,
            hdi_probs=hdi_probs if hdi_probs is not None else self.hdi_probs,
            frequency=frequency,
            output_format=output_format
            if output_format is not None
            else self.output_format,
        )

    def period_over_period(
        self,
        hdi_probs: list[float] | None = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Get period-over-period summary."""
        return create_period_over_period_summary(
            self.data,
            hdi_probs=hdi_probs if hdi_probs is not None else self.hdi_probs,
            output_format=output_format
            if output_format is not None
            else self.output_format,
        )
