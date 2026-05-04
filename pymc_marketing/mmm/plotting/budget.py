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
"""Budget allocation plots namespace."""

from __future__ import annotations

from typing import Any

import arviz_plots as azp
import xarray as xr
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pymc_marketing.mmm.plotting._helpers import (
    _ensure_chain_draw_dims,
    _extract_matplotlib_result,
    _plot_timeseries_channel,
    _process_plot_params,
    _select_dims,
)


class BudgetPlots:
    """Budget allocation plots.

    Stateless namespace — all data is supplied via the ``samples`` argument
    on each method call.  Obtain an instance via ``mmm.plot.budget``.

    Methods
    -------
    allocation_roas : Forest plot of per-channel ROAS distributions.
    contribution_over_time : Time-series of channel contributions from an
        optimised budget allocation.
    """

    def allocation_roas(
        self,
        samples: xr.Dataset,
        dims: dict[str, Any] | None = None,
        hdi_prob: float = 0.94,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Forest plot of per-channel ROAS from an optimised budget allocation.

        One row per channel; x-axis is ROAS; thick bar = 50% HDI, thin bar =
        ``hdi_prob`` HDI; point = median; vertical reference line at x=1 marks
        break-even (ROAS < 1 means a money-losing channel at this allocation).

        Parameters
        ----------
        samples : xr.Dataset
            Output of ``mmm.allocate_budget_to_maximize_response(...)`` or
            equivalent.  Must contain:

            - ``channel_contribution_original_scale``
              (dims: ``sample`` or ``(chain, draw)``, ``date``, ``channel``, ...)
            - ``allocation`` (dims: ``channel``, ...)
        dims : dict, optional
            Dimension filters, e.g. ``{"geo": ["CA"]}``.
        hdi_prob : float, default 0.94
            Probability mass for the outer HDI bar.
        figsize : tuple, optional
            Injected into ``figure_kwargs``.
        backend : str, optional
            Rendering backend.  Non-matplotlib requires ``return_as_pc=True``.
        return_as_pc : bool, default False
            Return the ``PlotCollection`` instead of ``(Figure, NDArray[Axes])``.
        **pc_kwargs
            Forwarded to ``azp.plot_forest()``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection
        """
        if "channel_contribution_original_scale" not in samples:
            raise ValueError(
                "Expected 'channel_contribution_original_scale' variable in samples, "
                "but none found."
            )
        if "allocation" not in samples:
            raise ValueError(
                "Expected 'allocation' variable in samples, but none found."
            )
        if "channel" not in samples.dims:
            raise ValueError("Expected 'channel' dimension in samples, but none found.")

        pc_kwargs = _process_plot_params(
            figsize=figsize,
            backend=backend,
            return_as_pc=return_as_pc,
            **pc_kwargs,
        )

        roas_da = (
            samples["channel_contribution_original_scale"].sum("date")
            / samples["allocation"]
        )
        roas_da.name = "roas"

        roas_da = _select_dims(roas_da, dims)
        roas_da = _ensure_chain_draw_dims(roas_da)

        pc = azp.plot_forest(
            roas_da.to_dataset(),
            combined=True,
            ci_kind="hdi",
            ci_probs=(0.5, hdi_prob),
            labels=set(roas_da.dims) - set(["chain", "draw"]),
            backend=backend,
            **pc_kwargs,
        )
        azp.add_lines(pc, 1.0, orientation="vertical")

        return _extract_matplotlib_result(pc, return_as_pc)

    def contribution_over_time(
        self,
        samples: xr.Dataset,
        dims: dict[str, Any] | None = None,
        hdi_prob: float = 0.85,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        line_kwargs: dict[str, Any] | None = None,
        hdi_kwargs: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Time-series of channel contributions from an optimised budget allocation.

        Creates one panel per extra-dimension combination (e.g. one per geo).
        Each panel shows a mean line and HDI band per channel.

        Parameters
        ----------
        samples : xr.Dataset
            Output of ``mmm.allocate_budget_to_maximize_response(...)`` or
            equivalent.  Must have:

            - a variable whose name contains ``"channel_contribution"``
              (dims: ``sample``, ``date``, ``channel``, ...)
        dims : dict, optional
            Dimension filters, e.g. ``{"geo": ["CA"]}``.
        hdi_prob : float, default 0.85
            HDI probability mass.
        figsize : tuple, optional
            Injected into ``figure_kwargs``.
        backend : str, optional
            Rendering backend.  Non-matplotlib requires ``return_as_pc=True``.
        return_as_pc : bool, default False
            Return the ``PlotCollection`` instead of ``(Figure, NDArray[Axes])``.
        line_kwargs : dict, optional
            Extra kwargs forwarded to ``azp.visuals.line_xy``.
        hdi_kwargs : dict, optional
            Extra kwargs forwarded to ``azp.visuals.fill_between_y``.
        **pc_kwargs
            Forwarded to ``PlotCollection.wrap()``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection
        """
        for dim in ("channel", "date", "sample"):
            if dim not in samples.dims:
                raise ValueError(
                    f"Expected '{dim}' dimension in samples, but none found."
                )
        if "channel_contribution_original_scale" not in samples.data_vars:
            raise ValueError(
                "Expected a variable containing 'channel_contribution' in samples, "
                "but none found."
            )

        pc_kwargs = _process_plot_params(
            figsize=figsize,
            backend=backend,
            return_as_pc=return_as_pc,
            **pc_kwargs,
        )

        contrib_var = next(
            v for v in samples.data_vars if "channel_contribution_original_scale" in v
        )
        da = _select_dims(samples[contrib_var], dims)

        extra_dims = [
            d
            for d in da.dims
            if d not in {"channel", "date", "sample", "chain", "draw"}
        ]
        da = _ensure_chain_draw_dims(da)
        ds = da.to_dataset(name="contribution")

        pc = _plot_timeseries_channel(
            ds=ds,
            sample_dims=["chain", "draw"],
            color_dim="channel",
            extra_dims=extra_dims,
            hdi_prob=hdi_prob,
            backend=backend,
            line_kwargs=line_kwargs,
            hdi_kwargs=hdi_kwargs,
            **pc_kwargs,
        )

        return _extract_matplotlib_result(pc, return_as_pc)
