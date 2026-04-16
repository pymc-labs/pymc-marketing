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
"""Sensitivity namespace — sensitivity analysis plots.

:class:`SensitivityPlots` provides three methods to visualise the results of a
sensitivity sweep run via :class:`~pymc_marketing.mmm.SensitivityAnalysis`:

* :meth:`~SensitivityPlots.analysis`  — raw effect curves from the input sweep
* :meth:`~SensitivityPlots.uplift`    — uplift relative to a baseline (with reference lines)
* :meth:`~SensitivityPlots.marginal`  — marginal effects along the sweep

The class is normally accessed through the ``mmm.plots.sensitivity`` shortcut on a
fitted :class:`~pymc_marketing.mmm.multidimensional.MMM` instance, but it can also be constructed
directly from any :class:`~pymc_marketing.data.idata.MMMIDataWrapper`.

Examples
--------

.. code-block:: python

    # sensitivity analysis
    sweeps = np.linspace(0.1, 2.0, 100)
    mmm.sensitivity.run_sweep(
        sweep_values=sweeps,
        var_input="channel_data",
        var_names="channel_contribution",
        extend_idata=True,
    )

    sp = mmm.plots.sensitivity
    fig, axes = sp.analysis()

    # uplift curve
    ref = mmm.idata.posterior.channel_contribution.sum(["channel", "date"]).mean(
        ["chain", "draw"]
    )
    mmm.sensitivity.compute_uplift_curve_respect_to_base(
        results=mmm.idata.sensitivity_analysis["x"],
        ref=ref,
        extend_idata=True,
    )

    fig, axes = sp.uplift(aggregation={"sum": "channel"}, figsize=(10, 4))


    # marginal contribution curve
    mmm.sensitivity.compute_marginal_effects(
        results=mmm.idata.sensitivity_analysis["uplift_curve"],
        extend_idata=True,
    )

    fig, axes = sp.marginal(aggregation={"sum": "channel"})
"""

from __future__ import annotations

from typing import Any, Literal

import arviz as az
import arviz_plots as azp
import xarray as xr
from arviz_base.labels import DimCoordLabeller, NoVarLabeller, mix_labellers
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pymc_marketing.data.idata import MMMIDataWrapper
from pymc_marketing.mmm.plotting._helpers import (
    _ensure_chain_draw_dims,
    _extract_matplotlib_result,
    _process_plot_params,
    _select_dims,
)


class SensitivityPlots:
    """Sensitivity analysis plots (effect, uplift, and marginal curves).

    Wraps the three sensitivity-analysis plotting methods previously in the
    monolithic ``MMMPlotSuite``.  All three methods share the same signature
    and delegate to :meth:`_sensitivity_plot` for rendering.

    Parameters
    ----------
    data : MMMIDataWrapper
        Validated wrapper around the fitted model's ``InferenceData``.
    """

    def __init__(self, data: MMMIDataWrapper) -> None:
        self._data = data

    def analysis(
        self,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        aggregation: dict[str, str | list[str]] | None = None,
        x_sweep_axis: Literal["relative", "absolute"] = "relative",
        apply_cost_per_unit: bool = True,
        hdi_prob: float = 0.94,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        line_kwargs: dict[str, Any] | None = None,
        hdi_kwargs: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Plot sensitivity analysis sweep results (``idata.sensitivity_analysis["x"]``).

        Parameters
        ----------
        idata : az.InferenceData, optional
            Override instance data.  When provided, an ``MMMIDataWrapper`` is
            constructed from this ``idata`` and used for this call only.
        dims : dict, optional
            Dimension filters, e.g. ``{"channel": ["tv", "radio"]}``.
        aggregation : dict, optional
            Aggregation to apply before plotting, e.g.
            ``{"sum": "channel"}`` or ``{"mean": ["channel"]}``.
        x_sweep_axis : {"relative", "absolute"}, default "relative"
            ``"relative"`` plots sweep multipliers on the x-axis.
            ``"absolute"`` scales multipliers by total channel spend/data.
        apply_cost_per_unit : bool, default True
            When ``x_sweep_axis="absolute"``, use spend (True) or raw channel
            data (False) for x-axis scaling.
        hdi_prob : float, default 0.94
            Credible interval probability for the HDI band.
        figsize : tuple[float, float], optional
            Convenience shorthand injected into ``figure_kwargs``.
        backend : str, optional
            Rendering backend (``"matplotlib"``, ``"plotly"``, ``"bokeh"``).
        return_as_pc : bool, default False
            If True, return the ``PlotCollection`` instead of the matplotlib tuple.
        line_kwargs : dict, optional
            Extra keyword arguments forwarded to the mean line visual.
        hdi_kwargs : dict, optional
            Extra keyword arguments forwarded to the HDI band visual.
        **pc_kwargs
            Forwarded to ``PlotCollection.grid()``.  Use ``cols=`` / ``rows=``
            to override the default panel layout.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection
        """
        data = (
            MMMIDataWrapper(idata, schema=self._data.schema)
            if idata is not None
            else self._data
        )
        if not hasattr(data.idata, "sensitivity_analysis"):
            raise ValueError(
                "idata has no 'sensitivity_analysis' group. "
                "Run SensitivityAnalysis.run_sweep() with extend_idata=True first."
            )
        sa_group = data.idata.sensitivity_analysis
        if "x" not in sa_group:
            raise ValueError(
                "'x' not found in idata.sensitivity_analysis. "
                "Run SensitivityAnalysis.run_sweep() to populate it."
            )
        return self._sensitivity_plot(
            sa_da=sa_group["x"],
            data=data,
            ylabel="Effect",
            dims=dims,
            aggregation=aggregation,
            x_sweep_axis=x_sweep_axis,
            apply_cost_per_unit=apply_cost_per_unit,
            hdi_prob=hdi_prob,
            figsize=figsize,
            backend=backend,
            return_as_pc=return_as_pc,
            line_kwargs=line_kwargs,
            hdi_kwargs=hdi_kwargs,
            **pc_kwargs,
        )

    def uplift(
        self,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        aggregation: dict[str, str | list[str]] | None = None,
        x_sweep_axis: Literal["relative", "absolute"] = "relative",
        apply_cost_per_unit: bool = True,
        hdi_prob: float = 0.94,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        line_kwargs: dict[str, Any] | None = None,
        hdi_kwargs: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Plot uplift curves (``idata.sensitivity_analysis["uplift_curve"]``).

        Parameters
        ----------
        idata : az.InferenceData, optional
            Override instance data.
        dims : dict, optional
            Dimension filters.
        aggregation : dict, optional
            Aggregation to apply before plotting.
        x_sweep_axis : {"relative", "absolute"}, default "relative"
            ``"relative"`` plots sweep multipliers; ``"absolute"`` scales by
            total channel spend/data.
        apply_cost_per_unit : bool, default True
            When ``x_sweep_axis="absolute"``, use spend (True) or raw channel
            data (False).
        hdi_prob : float, default 0.94
            Credible interval probability for the HDI band.
        figsize : tuple[float, float], optional
            Convenience shorthand for figure size.
        backend : str, optional
            Rendering backend.
        return_as_pc : bool, default False
            Return ``PlotCollection`` instead of matplotlib tuple.
        line_kwargs : dict, optional
            Extra keyword arguments for the mean line visual.
        hdi_kwargs : dict, optional
            Extra keyword arguments for the HDI band visual.
        **pc_kwargs
            Forwarded to ``PlotCollection.grid()``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection
        """
        data = (
            MMMIDataWrapper(idata, schema=self._data.schema)
            if idata is not None
            else self._data
        )
        if not hasattr(data.idata, "sensitivity_analysis"):
            raise ValueError(
                "idata has no 'sensitivity_analysis' group. "
                "Run SensitivityAnalysis and compute_uplift_curve_respect_to_base() first."
            )
        sa_group = data.idata.sensitivity_analysis
        if "uplift_curve" not in sa_group:
            raise ValueError(
                "'uplift_curve' not found in idata.sensitivity_analysis. "
                "Run SensitivityAnalysis.compute_uplift_curve_respect_to_base() first."
            )
        # Validate backend compatibility before calling _sensitivity_plot
        _process_plot_params(
            figsize=figsize,
            backend=backend,
            return_as_pc=return_as_pc,
        )

        pc = self._sensitivity_plot(
            sa_da=sa_group["uplift_curve"],
            data=data,
            ylabel="Uplift",
            dims=dims,
            aggregation=aggregation,
            x_sweep_axis=x_sweep_axis,
            apply_cost_per_unit=apply_cost_per_unit,
            hdi_prob=hdi_prob,
            figsize=figsize,
            backend=backend,
            return_as_pc=True,
            line_kwargs=line_kwargs,
            hdi_kwargs=hdi_kwargs,
            **pc_kwargs,
        )
        # Add reference lines at appropriate positions
        if x_sweep_axis == "relative":
            ref_x = 1.0
        else:
            # In absolute mode, baseline is at total spend/data (multiplier=1.0)
            if apply_cost_per_unit:
                channel_scale = data.get_channel_spend().sum("date")
            else:
                channel_scale = data.get_channel_data().sum("date")
            ref_x = channel_scale

        azp.add_lines(
            pc, ref_x, orientation="vertical", visuals={"ref_line": {"zorder": 2}}
        )
        azp.add_lines(pc, 0.0, orientation="horizontal")
        return _extract_matplotlib_result(pc, return_as_pc)

    def marginal(
        self,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        aggregation: dict[str, str | list[str]] | None = None,
        x_sweep_axis: Literal["relative", "absolute"] = "relative",
        apply_cost_per_unit: bool = True,
        hdi_prob: float = 0.94,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        line_kwargs: dict[str, Any] | None = None,
        hdi_kwargs: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Plot marginal effects (``idata.sensitivity_analysis["marginal_effects"]``).

        Parameters
        ----------
        idata : az.InferenceData, optional
            Override instance data.
        dims : dict, optional
            Dimension filters.
        aggregation : dict, optional
            Aggregation to apply before plotting.
        x_sweep_axis : {"relative", "absolute"}, default "relative"
            ``"relative"`` plots sweep multipliers; ``"absolute"`` scales by
            total channel spend/data.
        apply_cost_per_unit : bool, default True
            When ``x_sweep_axis="absolute"``, use spend (True) or raw channel
            data (False).
        hdi_prob : float, default 0.94
            Credible interval probability for the HDI band.
        figsize : tuple[float, float], optional
            Convenience shorthand for figure size.
        backend : str, optional
            Rendering backend.
        return_as_pc : bool, default False
            Return ``PlotCollection`` instead of matplotlib tuple.
        line_kwargs : dict, optional
            Extra keyword arguments for the mean line visual.
        hdi_kwargs : dict, optional
            Extra keyword arguments for the HDI band visual.
        **pc_kwargs
            Forwarded to ``PlotCollection.grid()``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection
        """
        data = (
            MMMIDataWrapper(idata, schema=self._data.schema)
            if idata is not None
            else self._data
        )
        if not hasattr(data.idata, "sensitivity_analysis"):
            raise ValueError(
                "idata has no 'sensitivity_analysis' group. "
                "Run SensitivityAnalysis and compute_marginal_effects() first."
            )
        sa_group = data.idata.sensitivity_analysis
        if "marginal_effects" not in sa_group:
            raise ValueError(
                "'marginal_effects' not found in idata.sensitivity_analysis. "
                "Run SensitivityAnalysis.compute_marginal_effects() first."
            )
        return self._sensitivity_plot(
            sa_da=sa_group["marginal_effects"],
            data=data,
            ylabel="Marginal Effect",
            dims=dims,
            aggregation=aggregation,
            x_sweep_axis=x_sweep_axis,
            apply_cost_per_unit=apply_cost_per_unit,
            hdi_prob=hdi_prob,
            figsize=figsize,
            backend=backend,
            return_as_pc=return_as_pc,
            line_kwargs=line_kwargs,
            hdi_kwargs=hdi_kwargs,
            **pc_kwargs,
        )

    def _sensitivity_plot(
        self,
        sa_da: xr.DataArray,
        data: MMMIDataWrapper,
        ylabel: str,
        dims: dict[str, Any] | None = None,
        aggregation: dict[str, str | list[str]] | None = None,
        x_sweep_axis: Literal["relative", "absolute"] = "relative",
        apply_cost_per_unit: bool = True,
        hdi_prob: float = 0.94,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        line_kwargs: dict[str, Any] | None = None,
        hdi_kwargs: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        pc_kwargs = _process_plot_params(
            figsize=figsize,
            backend=backend,
            return_as_pc=return_as_pc,
            **pc_kwargs,
        )

        # Step 1: Apply aggregation
        if aggregation:
            for op, dim_spec in aggregation.items():
                dims_list = [dim_spec] if isinstance(dim_spec, str) else list(dim_spec)
                if op == "sum":
                    sa_da = sa_da.sum(dim=dims_list)
                elif op == "mean":
                    sa_da = sa_da.mean(dim=dims_list)
                else:
                    raise ValueError(
                        f"Unknown aggregation operation '{op}'. "
                        "Supported operations: 'sum', 'mean'."
                    )

        # Step 2: Apply dimension filtering
        sa_da = _select_dims(sa_da, dims)

        # Step 3: Reshape sample → (chain, draw)
        sa_da = _ensure_chain_draw_dims(sa_da)

        # Step 4: Determine faceting and hue
        cols = pc_kwargs.pop("cols", list(data.custom_dims))
        rows = pc_kwargs.pop("rows", [])
        hue_dims = [
            d for d in sa_da.dims if d not in {"chain", "draw", "sweep", *cols, *rows}
        ]

        # Step 5: Compute sweep x-values
        sweep_coords = sa_da.coords["sweep"]
        if x_sweep_axis == "relative":
            sweep_x = sweep_coords
        else:
            if apply_cost_per_unit:
                channel_scale = data.get_channel_spend().sum("date")
            else:
                channel_scale = data.get_channel_data().sum("date")

            # If channel dimension was aggregated away, aggregate channel_scale too
            if "channel" in channel_scale.dims and "channel" not in sa_da.dims:
                channel_scale = channel_scale.sum("channel")

            sweep_x = sweep_coords * channel_scale

        # Step 6: Build PlotCollection
        sa_ds = sa_da.to_dataset(name="sensitivity")
        pc = PlotCollection.grid(
            sa_ds,
            rows=rows,
            cols=cols,
            aes={"color": hue_dims} if hue_dims else {},
            backend=backend,
            **pc_kwargs,
        )

        # Step 7: HDI band
        hdi_da = sa_ds.azstats.hdi(hdi_prob)
        pc.map(
            azp.visuals.fill_between_y,
            x=sweep_x,
            y_bottom=hdi_da.sel(ci_bound="lower"),
            y_top=hdi_da.sel(ci_bound="upper"),
            **{"alpha": 0.2, **(hdi_kwargs or {})},
        )

        # Step 8: Mean line
        mean_da = sa_ds.mean(dim=["chain", "draw"])
        pc.map(azp.visuals.line_xy, x=sweep_x, y=mean_da, **(line_kwargs or {}))

        # Step 9: Axis labels and title
        if x_sweep_axis == "relative":
            xlabel = "Sweep Multiplier"
        elif apply_cost_per_unit:
            xlabel = "Spend"
        else:
            xlabel = "Channel Data"
        pc.map(azp.visuals.labelled_x, text=xlabel, ignore_aes={"color"})
        pc.map(azp.visuals.labelled_y, text=ylabel, ignore_aes={"color"})
        pc.map(
            azp.visuals.labelled_title,
            subset_info=True,
            labeller=mix_labellers((NoVarLabeller, DimCoordLabeller))(),
            ignore_aes={"color"},
        )

        # Step 10: Legend
        if hue_dims:
            pc.add_legend(hue_dims[0])

        # Step 11: Return
        return _extract_matplotlib_result(pc, return_as_pc)
