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
"""Summary DataFrame generation for sensitivity analysis results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd
import xarray as xr

from pymc_marketing.data.idata import MMMIDataWrapper
from pymc_marketing.mmm.summary_helpers import (
    DataFrameType,
    OutputFormat,
    StatsHelper,
    compute_summary_stats_with_hdi,
    convert_output,
    prepare_sensitivity_data,
)

__all__ = [
    "sensitivity_analysis",
    "sensitivity_marginal",
    "sensitivity_uplift",
]

_stats = StatsHelper()


def _add_sweep_x_column(
    df: pd.DataFrame,
    sweep_x: xr.DataArray,
    x_sweep_axis: Literal["relative", "absolute"],
) -> pd.DataFrame:
    """Add sweep_x column aligned with summary rows."""
    if x_sweep_axis == "relative" or sweep_x.ndim == 1:
        if "sweep" in df.columns:
            mapping = {
                float(s): float(sweep_x.sel(sweep=s).item())
                for s in sweep_x.coords["sweep"].values
            }
            df["sweep_x"] = df["sweep"].map(mapping)
        return df

    sweep_x_df = sweep_x.to_dataframe(name="sweep_x").reset_index()
    merge_keys = [c for c in sweep_x_df.columns if c != "sweep_x"]
    return df.merge(sweep_x_df, on=merge_keys, how="left")


def _sensitivity_summary(
    data: MMMIDataWrapper,
    sa_var: str,
    hdi_probs: Sequence[float],
    dims: dict[str, Any] | None,
    aggregation: dict[str, str | list[str]] | None,
    x_sweep_axis: Literal["relative", "absolute"],
    apply_cost_per_unit: bool,
    output_format: OutputFormat,
    missing_message: str,
) -> DataFrameType:
    _stats.validate_hdi_probs(hdi_probs)

    if not hasattr(data.idata, "sensitivity_analysis"):
        raise ValueError(
            "idata has no 'sensitivity_analysis' group. "
            "Run SensitivityAnalysis with extend_idata=True first."
        )
    sa_group = data.idata.sensitivity_analysis
    if sa_var not in sa_group:
        raise ValueError(missing_message)

    sweep_x, sa_da = prepare_sensitivity_data(
        sa_da=sa_group[sa_var],
        data=data,
        dims=dims,
        aggregation=aggregation,
        x_sweep_axis=x_sweep_axis,
        apply_cost_per_unit=apply_cost_per_unit,
    )
    sa_da.name = sa_var

    df = compute_summary_stats_with_hdi(sa_da, hdi_probs)
    df = _add_sweep_x_column(df, sweep_x, x_sweep_axis)

    return convert_output(df, output_format)


def sensitivity_analysis(
    data: MMMIDataWrapper,
    hdi_probs: Sequence[float] = (0.94,),
    dims: dict[str, Any] | None = None,
    aggregation: dict[str, str | list[str]] | None = None,
    x_sweep_axis: Literal["relative", "absolute"] = "relative",
    apply_cost_per_unit: bool = True,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Summarize raw sensitivity sweep results (``sensitivity_analysis['x']``).

    Parameters
    ----------
    data : MMMIDataWrapper
        Fitted model data wrapper with sensitivity analysis results.
    hdi_probs : sequence of float, default (0.94,)
        HDI probability levels.
    dims : dict, optional
        Dimension filters.
    aggregation : dict, optional
        Aggregation spec before summarization.
    x_sweep_axis : {"relative", "absolute"}, default "relative"
        Sweep axis interpretation for ``sweep_x`` column.
    apply_cost_per_unit : bool, default True
        Use spend for absolute x-axis when applicable.
    output_format : {"pandas", "polars"}, default "pandas"
        Output DataFrame format.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Summary with sweep, mean, median, HDI, and ``sweep_x`` columns.
    """
    return _sensitivity_summary(
        data=data,
        sa_var="x",
        hdi_probs=hdi_probs,
        dims=dims,
        aggregation=aggregation,
        x_sweep_axis=x_sweep_axis,
        apply_cost_per_unit=apply_cost_per_unit,
        output_format=output_format,
        missing_message=(
            "'x' not found in idata.sensitivity_analysis. "
            "Run SensitivityAnalysis.run_sweep() to populate it."
        ),
    )


def sensitivity_uplift(
    data: MMMIDataWrapper,
    hdi_probs: Sequence[float] = (0.94,),
    dims: dict[str, Any] | None = None,
    aggregation: dict[str, str | list[str]] | None = None,
    x_sweep_axis: Literal["relative", "absolute"] = "relative",
    apply_cost_per_unit: bool = True,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Summarize uplift curves (``sensitivity_analysis['uplift_curve']``).

    Parameters
    ----------
    data : MMMIDataWrapper
        Fitted model data wrapper with uplift curve results.
    hdi_probs : sequence of float, default (0.94,)
        HDI probability levels.
    dims : dict, optional
        Dimension filters.
    aggregation : dict, optional
        Aggregation spec before summarization.
    x_sweep_axis : {"relative", "absolute"}, default "relative"
        Sweep axis interpretation for ``sweep_x`` column.
    apply_cost_per_unit : bool, default True
        Use spend for absolute x-axis when applicable.
    output_format : {"pandas", "polars"}, default "pandas"
        Output DataFrame format.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Summary with sweep, mean, median, HDI, and ``sweep_x`` columns.
    """
    return _sensitivity_summary(
        data=data,
        sa_var="uplift_curve",
        hdi_probs=hdi_probs,
        dims=dims,
        aggregation=aggregation,
        x_sweep_axis=x_sweep_axis,
        apply_cost_per_unit=apply_cost_per_unit,
        output_format=output_format,
        missing_message=(
            "'uplift_curve' not found in idata.sensitivity_analysis. "
            "Run SensitivityAnalysis.compute_uplift_curve_respect_to_base() first."
        ),
    )


def sensitivity_marginal(
    data: MMMIDataWrapper,
    hdi_probs: Sequence[float] = (0.94,),
    dims: dict[str, Any] | None = None,
    aggregation: dict[str, str | list[str]] | None = None,
    x_sweep_axis: Literal["relative", "absolute"] = "relative",
    apply_cost_per_unit: bool = True,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Summarize marginal effects (``sensitivity_analysis['marginal_effects']``).

    Parameters
    ----------
    data : MMMIDataWrapper
        Fitted model data wrapper with marginal effects results.
    hdi_probs : sequence of float, default (0.94,)
        HDI probability levels.
    dims : dict, optional
        Dimension filters.
    aggregation : dict, optional
        Aggregation spec before summarization.
    x_sweep_axis : {"relative", "absolute"}, default "relative"
        Sweep axis interpretation for ``sweep_x`` column.
    apply_cost_per_unit : bool, default True
        Use spend for absolute x-axis when applicable.
    output_format : {"pandas", "polars"}, default "pandas"
        Output DataFrame format.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Summary with sweep, mean, median, HDI, and ``sweep_x`` columns.
    """
    return _sensitivity_summary(
        data=data,
        sa_var="marginal_effects",
        hdi_probs=hdi_probs,
        dims=dims,
        aggregation=aggregation,
        x_sweep_axis=x_sweep_axis,
        apply_cost_per_unit=apply_cost_per_unit,
        output_format=output_format,
        missing_message=(
            "'marginal_effects' not found in idata.sensitivity_analysis. "
            "Run SensitivityAnalysis.compute_marginal_effects() first."
        ),
    )
