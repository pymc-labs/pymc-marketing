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
"""Shared computation helpers for MMM summary factories."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

import arviz as az
import pandas as pd
import xarray as xr

from pymc_marketing.data.idata import MMMIDataWrapper
from pymc_marketing.data.idata.utils import (
    get_posterior_predictive,
    get_prior,
    get_prior_predictive,
)
from pymc_marketing.mmm.xarray_utils import (
    _apply_aggregation,
    _ensure_chain_draw_dims,
    _select_dims,
)

OutputFormat = Literal["pandas", "polars"]
DataFrameType = pd.DataFrame  # Union[pd.DataFrame, pl.DataFrame] at runtime

__all__ = [
    "DataFrameType",
    "OutputFormat",
    "StatsHelper",
    "compute_channel_shares",
    "compute_residuals",
    "compute_summary_stats_with_hdi",
    "compute_waterfall_components",
    "convert_output",
    "dataframe_to_json_records",
    "get_channel_x_data",
    "get_prior_for_plot",
    "prepare_sensitivity_data",
]


def compute_residuals(data: MMMIDataWrapper) -> xr.DataArray:
    """Compute residuals as target minus posterior predictions.

    Parameters
    ----------
    data : MMMIDataWrapper
        Wrapper holding idata with posterior_predictive and constant_data.

    Returns
    -------
    xr.DataArray
        Residuals named ``residuals`` with same dims as ``y_original_scale``.

    Raises
    ------
    ValueError
        If ``y_original_scale`` is missing from posterior_predictive.
    """
    pp_var = "y_original_scale"
    pp_ds = get_posterior_predictive(data.idata)
    if pp_var not in pp_ds:
        raise ValueError(
            f"Variable '{pp_var}' not found in posterior_predictive. "
            f"Available: {list(pp_ds.data_vars)}"
        )
    predictions = pp_ds[pp_var]
    target = data.get_target(original_scale=True)
    residuals = target - predictions
    residuals.name = "residuals"
    return residuals


def get_prior_for_plot(data: MMMIDataWrapper, original_scale: bool) -> xr.Dataset:
    """Return the correct idata group for prior predictive plotting.

    Parameters
    ----------
    data : MMMIDataWrapper
        Wrapper holding the model's DataTree.
    original_scale : bool
        If True, return ``idata.prior`` (contains ``y_original_scale``).
        If False, return ``idata.prior_predictive`` (contains ``y``).

    Returns
    -------
    xr.Dataset
    """
    if original_scale:
        return get_prior(data.idata)
    return get_prior_predictive(data.idata)


def compute_waterfall_components(
    data: MMMIDataWrapper,
    dims: dict[str, Any] | None = None,
    original_scale: bool = True,
) -> xr.DataArray:
    """Sum contributions over date per component for waterfall summaries.

    Mirrors :meth:`~pymc_marketing.mmm.plotting.decomposition.DecompositionPlots.waterfall`
    but preserves ``chain`` and ``draw`` dimensions.

    Parameters
    ----------
    data : MMMIDataWrapper
        Fitted model data wrapper.
    dims : dict, optional
        Dimension filters passed to :func:`_select_dims`.
    original_scale : bool, default True
        Whether to use original-scale contributions.

    Returns
    -------
    xr.DataArray
        Values with dims ``(chain, draw, component[, custom_dims])``.
    """
    contributions_ds = data.get_contributions(original_scale=original_scale)
    component_arrays: list[xr.DataArray] = []

    for ds_key, coord_dim in [
        ("baseline", None),
        ("channels", "channel"),
        ("controls", "control"),
        ("seasonality", None),
    ]:
        if ds_key not in contributions_ds:
            continue
        da = _select_dims(contributions_ds[ds_key], dims, allow_missing=True)
        sum_dims = [d for d in ("date",) if d in da.dims]
        if coord_dim is not None:
            for val in da.coords[coord_dim].values:
                component_da = da.sel({coord_dim: val}, drop=True)
                if sum_dims:
                    component_da = component_da.sum(dim=sum_dims)
                component_arrays.append(component_da.expand_dims(component=[str(val)]))
        else:
            component_da = da.sum(dim=sum_dims) if sum_dims else da
            component_arrays.append(component_da.expand_dims(component=[ds_key]))

    if not component_arrays:
        raise ValueError("No contribution data found for waterfall components.")

    return xr.concat(component_arrays, dim="component")


def compute_channel_shares(channel_contributions: xr.DataArray) -> xr.DataArray:
    """Compute each channel's share of total channel contribution.

    Parameters
    ----------
    channel_contributions : xr.DataArray
        Channel contributions, typically with dims
        ``(chain, draw, date, channel[, custom_dims])``.

    Returns
    -------
    xr.DataArray
        Channel shares with ``date`` summed out.
    """
    summed = channel_contributions.sum(dim="date")
    total = summed.sum(dim="channel")
    shares = summed / total
    shares.name = "channel_share"
    return shares


def get_channel_x_data(
    data: MMMIDataWrapper, apply_cost_per_unit: bool
) -> xr.DataArray:
    """Return channel spend or raw channel data based on cost-per-unit flag."""
    if apply_cost_per_unit:
        return data.get_channel_spend()
    return data.get_channel_data()


def prepare_sensitivity_data(
    sa_da: xr.DataArray,
    data: MMMIDataWrapper,
    dims: dict[str, Any] | None = None,
    aggregation: dict[str, str | list[str]] | None = None,
    x_sweep_axis: Literal["relative", "absolute"] = "relative",
    apply_cost_per_unit: bool = True,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Prepare sensitivity data for summary tables (steps 1-5 of sensitivity plots).

    Parameters
    ----------
    sa_da : xr.DataArray
        Raw sensitivity analysis values with a ``sweep`` dimension.
    data : MMMIDataWrapper
        Model data wrapper for spend/data scaling.
    dims : dict, optional
        Dimension filters.
    aggregation : dict, optional
        Aggregation spec, e.g. ``{"sum": "channel"}``.
    x_sweep_axis : {"relative", "absolute"}, default "relative"
        Whether x-values are sweep multipliers or absolute spend/data.
    apply_cost_per_unit : bool, default True
        When ``x_sweep_axis="absolute"``, use spend (True) or channel data (False).

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        ``(sweep_x, sa_da_processed)`` ready for HDI summarization.
    """
    sa_da = _apply_aggregation(sa_da, aggregation)
    sa_da = _select_dims(sa_da, dims)
    sa_da = _ensure_chain_draw_dims(sa_da)

    sweep_coords = sa_da.coords["sweep"]
    if x_sweep_axis == "relative":
        sweep_x = sweep_coords
    else:
        if apply_cost_per_unit:
            channel_scale = data.get_channel_spend().sum("date")
        else:
            channel_scale = data.get_channel_data().sum("date")

        if "channel" in channel_scale.dims and "channel" not in sa_da.dims:
            channel_scale = channel_scale.sum("channel")

        sweep_x = sweep_coords * channel_scale

    return sweep_x, sa_da


class StatsHelper:
    """Lightweight helper for HDI summary stats and output conversion."""

    def validate_hdi_probs(self, hdi_probs: Sequence[float]) -> None:
        """Validate HDI probability values are in range (0, 1)."""
        for prob in hdi_probs:
            if not 0 < prob < 1:
                raise ValueError(
                    f"HDI probability must be between 0 and 1 (exclusive), got {prob}. "
                    "Use values like 0.94 for 94% HDI, not percentages like 94."
                )

    def convert_output(
        self, df: pd.DataFrame, output_format: OutputFormat = "pandas"
    ) -> DataFrameType:
        """Convert Pandas DataFrame to requested output format."""
        if output_format == "pandas":
            return df
        if output_format == "polars":
            try:
                import polars as pl
            except ImportError as exc:
                raise ImportError(
                    "Polars is required for output_format='polars'. "
                    "Install it with: pip install pymc-marketing[polars]"
                ) from exc
            return pl.from_pandas(df)
        raise ValueError(
            f"Unknown output_format: {output_format!r}. Use 'pandas' or 'polars'."
        )

    def compute_summary_stats_with_hdi(
        self,
        data: xr.DataArray,
        hdi_probs: Sequence[float],
    ) -> pd.DataFrame:
        """Convert xarray to DataFrame with mean, median, and HDI bounds."""
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

        index_cols = [d for d in data.dims if d not in sample_dims]
        var_name = "_values"
        data = data.rename(var_name)

        mean_ = data.mean(dim=sample_dims)
        median_ = data.median(dim=sample_dims)

        hdi_results = {}
        for hdi_prob in hdi_probs:
            prob_str = str(int(hdi_prob * 100))

            if use_az_hdi:
                hdi_da = az.hdi(data, prob=hdi_prob)
                hdi_lower = hdi_da.sel(ci_bound="lower").drop_vars(
                    "ci_bound", errors="ignore"
                )
                hdi_upper = hdi_da.sel(ci_bound="upper").drop_vars(
                    "ci_bound", errors="ignore"
                )
            else:
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

        result_dict = {"mean": mean_, "median": median_, **hdi_results}
        result_ds = xr.Dataset(result_dict)
        df = result_ds.to_dataframe().reset_index()

        other_cols = [c for c in df.columns if c not in index_cols]
        return df[index_cols + other_cols]


_stats_helper = StatsHelper()


def compute_summary_stats_with_hdi(
    data: xr.DataArray,
    hdi_probs: Sequence[float],
) -> pd.DataFrame:
    """Module-level wrapper around :meth:`StatsHelper.compute_summary_stats_with_hdi`."""
    return _stats_helper.compute_summary_stats_with_hdi(data, hdi_probs)


def convert_output(
    df: pd.DataFrame, output_format: OutputFormat = "pandas"
) -> DataFrameType:
    """Module-level wrapper around :meth:`StatsHelper.convert_output`."""
    return _stats_helper.convert_output(df, output_format)


def dataframe_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a summary DataFrame to JSON-serializable records.

    Normalizes datetime columns to ISO strings and numpy scalars to native
    Python types so the result can be passed directly to ``json.dumps``.
    """
    return json.loads(df.to_json(orient="records", date_format="iso"))
