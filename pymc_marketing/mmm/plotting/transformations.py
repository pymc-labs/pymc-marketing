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
"""Transformations namespace — saturation scatter and curve plots."""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np
import xarray as xr
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pymc_marketing.data.idata import MMMIDataWrapper
from pymc_marketing.mmm.plotting._helpers import (
    _dims_to_sel_kwargs,
    _extract_matplotlib_result,
    _process_plot_params,
    _validate_dims,
    channel_color_map,
)

# ============================================================================
# Module-level helpers (I.4 — no nested functions)
# ============================================================================


def _x_axis_label(data: MMMIDataWrapper, apply_cost_per_unit: bool) -> str:
    """Return the x-axis label based on cost-per-unit availability."""
    if apply_cost_per_unit and data.cost_per_unit is not None:
        return "Spend"
    return "Channel Data (X)"


def _get_channel_x_data(
    data: MMMIDataWrapper, apply_cost_per_unit: bool
) -> xr.DataArray:
    """Return channel spend or raw channel data based on cost-per-unit flag."""
    if apply_cost_per_unit:
        return data.get_channel_spend()
    return data.get_channel_data()


def _get_visual_kwargs(
    visuals: dict[str, Any] | None,
    key: str,
    defaults: dict[str, Any],
) -> dict[str, Any] | None:
    """Merge visual kwargs with defaults.  Returns None if the visual is disabled."""
    if visuals is None:
        return defaults
    visual = visuals.get(key)
    if visual is False:
        return None
    if visual is None:
        return defaults
    return {**defaults, **visual}


def _iter_panels(
    axes_da: xr.DataArray,
    dataset: xr.Dataset,
) -> list[tuple[Axes, xr.Dataset]]:
    """Yield (axes, panel_data) for each panel in the PlotCollection."""
    panels = []
    for idx in np.ndindex(axes_da.shape):
        ax = axes_da.values[idx]
        coord_dict = {
            dim: axes_da.coords[dim].values[i]
            for dim, i in zip(axes_da.dims, idx, strict=True)
        }
        panel_data = dataset.sel(**coord_dict)
        panels.append((ax, panel_data))
    return panels


def _scatter_visual(
    panel_ds: xr.Dataset,
    target: Axes,
    *,
    color_map: dict[str, str],
    scatter_kwargs: dict[str, Any],
) -> None:
    """Draw scatter points on one panel."""
    ch = str(panel_ds.coords["channel"].item()) if "channel" in panel_ds.coords else ""
    color = color_map.get(ch, "C0")
    x = panel_ds["x"].values.flatten()
    y = panel_ds["y"].values.flatten()
    target.scatter(x, y, color=color, **scatter_kwargs)


# ============================================================================
# Namespace class
# ============================================================================


class TransformationPlots:
    """Channel transformation plots (saturation scatter and curves).

    Parameters
    ----------
    data : MMMIDataWrapper
        Validated wrapper around the fitted model's ``InferenceData``.
    """

    def __init__(self, data: MMMIDataWrapper) -> None:
        self._data = data

    # ------------------------------------------------------------------ #
    # saturation_scatterplot
    # ------------------------------------------------------------------ #

    def saturation_scatterplot(
        self,
        original_scale: bool = True,
        apply_cost_per_unit: bool = True,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        plot_collection: PlotCollection | None = None,
        backend: str | None = None,
        visuals: dict[str, Any] | None = None,
        aes_by_visuals: dict[str, list[str]] | None = None,
        return_as_pc: bool = False,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Scatter plot of channel spend/data vs. mean channel contributions.

        Creates one panel per channel (and per custom dimension like ``country``
        or ``geo``).  Each point is one date observation.

        Parameters
        ----------
        original_scale : bool, default True
            If True, plot contributions in original (un-scaled) units.
        apply_cost_per_unit : bool, default True
            If True and cost-per-unit data is available, the x-axis shows
            spend.  If False, shows raw channel data.
        idata : az.InferenceData, optional
            Override instance data.  When provided, an ``MMMIDataWrapper``
            is constructed from this ``idata`` and used for all access.
        dims : dict, optional
            Dimension filters, e.g. ``{"country": "US"}`` or
            ``{"channel": ["tv", "radio"]}``.
        figsize : tuple[float, float], optional
            Convenience shorthand injected into ``figure_kwargs``.
        plot_collection : PlotCollection, optional
            Plot onto an existing ``PlotCollection``.
        backend : str, optional
            Rendering backend (``"matplotlib"``, ``"plotly"``, ``"bokeh"``).
        visuals : dict, optional
            Element-level customization.  Keys: ``"scatter"`` (dict of
            scatter kwargs, or ``False`` to disable).
        aes_by_visuals : dict, optional
            Aesthetic mapping per visual element.
        return_as_pc : bool, default False
            If True, return the ``PlotCollection`` instead of the
            matplotlib tuple.
        **pc_kwargs
            Forwarded to ``PlotCollection.wrap()``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection
        """
        data = MMMIDataWrapper(idata) if idata is not None else self._data

        pc_kwargs = _process_plot_params(
            figsize=figsize,
            plot_collection=plot_collection,
            backend=backend,
            return_as_pc=return_as_pc,
            **pc_kwargs,
        )

        contributions = data.get_channel_contributions(original_scale=original_scale)
        mean_contrib = contributions.mean(dim=["chain", "draw"])

        x_data = _get_channel_x_data(data, apply_cost_per_unit)
        x_data, mean_contrib = xr.broadcast(x_data, mean_contrib)

        scatter_ds = xr.Dataset({"x": x_data, "y": mean_contrib})

        _validate_dims(scatter_ds, dims)

        if dims:
            sel_kwargs = _dims_to_sel_kwargs(dims)
            scatter_ds = scatter_ds.sel(**sel_kwargs)

        facet_dims = [d for d in scatter_ds["x"].dims if d != "date"]

        if plot_collection is not None:
            pc = plot_collection
        else:
            pc = PlotCollection.wrap(
                scatter_ds,
                cols=facet_dims if facet_dims else None,
                backend=backend,
                **pc_kwargs,
            )

        scatter_kw = _get_visual_kwargs(visuals, "scatter", {"alpha": 0.8, "s": 20})
        if scatter_kw is not None:
            colors = channel_color_map(data.channels)
            axes_da = pc.viz.ds["plot"]
            for ax, panel_data in _iter_panels(axes_da, scatter_ds):
                _scatter_visual(
                    panel_data, ax, color_map=colors, scatter_kwargs=scatter_kw
                )

        x_label = _x_axis_label(data, apply_cost_per_unit)
        for ax in np.array(pc.viz.ds["plot"].values).flat:
            ax.set_xlabel(x_label)
            ax.set_ylabel("Channel Contributions")

        return _extract_matplotlib_result(pc, return_as_pc)
