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
"""Summary DataFrame generation for budget allocation results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import xarray as xr

from pymc_marketing.mmm.plotting._helpers import (
    _ensure_chain_draw_dims,
    _select_dims,
)
from pymc_marketing.mmm.summary_helpers import (
    DataFrameType,
    OutputFormat,
    StatsHelper,
    compute_summary_stats_with_hdi,
    convert_output,
)

__all__ = [
    "BudgetSummaryFactory",
    "prepare_allocation_roas",
    "prepare_contribution_over_time",
]

_stats = StatsHelper()


def prepare_allocation_roas(
    samples: xr.Dataset,
    dims: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Prepare per-channel ROAS DataArray from budget allocation samples."""
    if "channel_contribution_original_scale" not in samples:
        raise ValueError(
            "Expected 'channel_contribution_original_scale' variable in samples, "
            "but none found."
        )
    if "allocation" not in samples:
        raise ValueError("Expected 'allocation' variable in samples, but none found.")
    if "channel" not in samples.dims:
        raise ValueError("Expected 'channel' dimension in samples, but none found.")
    if "total_allocation" not in samples:
        raise ValueError(
            "Expected 'total_allocation' variable in samples, but none found."
        )

    roas_da = (
        samples["channel_contribution_original_scale"].sum("date")
        / samples["total_allocation"]
    )
    roas_da.name = "roas"
    roas_da = _select_dims(roas_da, dims)
    return _ensure_chain_draw_dims(roas_da)


def prepare_contribution_over_time(
    samples: xr.Dataset,
    dims: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Prepare channel contribution time series from budget allocation samples."""
    da = _select_dims(samples["channel_contribution_original_scale"], dims)
    return _ensure_chain_draw_dims(da)


class BudgetSummaryFactory:
    """Factory for creating summary DataFrames from budget allocation samples.

    Stateless namespace mirroring :class:`~pymc_marketing.mmm.plotting.budget.BudgetPlots`
    but returning tabular summaries with HDI statistics.
    """

    @staticmethod
    def allocation_roas(
        samples: xr.Dataset,
        hdi_probs: Sequence[float] = (0.94,),
        dims: dict[str, Any] | None = None,
        output_format: OutputFormat = "pandas",
    ) -> DataFrameType:
        """Summarize per-channel ROAS from an optimised budget allocation.

        Parameters
        ----------
        samples : xr.Dataset
            Output of ``sample_response_distribution(...)`` or equivalent.
            Must contain ``channel_contribution_original_scale``, ``allocation``,
            and ``total_allocation``.
        hdi_probs : sequence of float, default (0.94,)
            HDI probability levels.
        dims : dict, optional
            Dimension filters, e.g. ``{"geo": ["CA"]}``.
        output_format : {"pandas", "polars"}, default "pandas"
            Output DataFrame format.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Summary with channel, mean, median, and HDI columns.
        """
        _stats.validate_hdi_probs(hdi_probs)

        roas_da = prepare_allocation_roas(samples, dims=dims)

        df = compute_summary_stats_with_hdi(roas_da, hdi_probs)
        return convert_output(df, output_format)

    @staticmethod
    def contribution_over_time(
        samples: xr.Dataset,
        hdi_probs: Sequence[float] = (0.94,),
        dims: dict[str, Any] | None = None,
        output_format: OutputFormat = "pandas",
    ) -> DataFrameType:
        """Summarize channel contributions over time from budget allocation.

        Parameters
        ----------
        samples : xr.Dataset
            Output of ``allocate_budget_to_maximize_response(...)`` or equivalent.
            Must contain ``channel_contribution_original_scale`` with ``channel``,
            ``date``, and ``sample`` or ``(chain, draw)`` dimensions.
        hdi_probs : sequence of float, default (0.94,)
            HDI probability levels.
        dims : dict, optional
            Dimension filters.
        output_format : {"pandas", "polars"}, default "pandas"
            Output DataFrame format.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Summary with date, channel, mean, median, and HDI columns.
        """
        for dim in ("channel", "date"):
            if dim not in samples.dims:
                raise ValueError(
                    f"Expected '{dim}' dimension in samples, but none found."
                )
        has_sample_dim = "sample" in samples.dims or (
            "chain" in samples.dims and "draw" in samples.dims
        )
        if not has_sample_dim:
            raise ValueError(
                "Expected 'sample' or ('chain', 'draw') dimensions in samples."
            )
        if "channel_contribution_original_scale" not in samples.data_vars:
            raise ValueError(
                "Expected 'channel_contribution_original_scale' variable in samples, "
                "but none found."
            )

        _stats.validate_hdi_probs(hdi_probs)

        da = prepare_contribution_over_time(samples, dims=dims)

        df = compute_summary_stats_with_hdi(da, hdi_probs)
        return convert_output(df, output_format)
