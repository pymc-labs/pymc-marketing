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


def _iter_panels_da(
    axes_da: xr.DataArray,
    da: xr.DataArray,
) -> list[tuple[Axes, xr.DataArray]]:
    """Yield (axes, panel_slice) for each panel using a DataArray."""
    panels = []
    for idx in np.ndindex(axes_da.shape):
        ax = axes_da.values[idx]
        coord_dict = {
            dim: axes_da.coords[dim].values[i]
            for dim, i in zip(axes_da.dims, idx, strict=True)
        }
        da_sel = {k: v for k, v in coord_dict.items() if k in da.dims}
        panel_slice = da.sel(**da_sel) if da_sel else da
        panels.append((ax, panel_slice))
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


def _sample_curves_visual(
    da: xr.DataArray,
    target: Axes,
    *,
    x_vals: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    color: str,
    line_kwargs: dict[str, Any],
) -> None:
    """Draw random posterior sample curves on one panel."""
    if n_samples <= 0:
        return

    stacked = da.stack(sample=("chain", "draw"))
    n_total = stacked.sizes["sample"]
    n_draw = min(n_samples, n_total)
    indices = rng.choice(n_total, size=n_draw, replace=False)

    for i in indices:
        y_vals = stacked.isel(sample=i).values
        target.plot(x_vals, y_vals, color=color, **line_kwargs)


def _hdi_band_visual(
    da: xr.DataArray,
    target: Axes,
    *,
    x_vals: np.ndarray,
    hdi_prob: float,
    color: str,
    fill_kwargs: dict[str, Any],
) -> None:
    """Draw HDI band on one panel."""
    stacked = da.stack(sample=("chain", "draw"))
    vals = stacked.transpose("x", "sample").values

    alpha = (1 - hdi_prob) / 2
    low = np.quantile(vals, alpha, axis=1)
    high = np.quantile(vals, 1 - alpha, axis=1)

    target.fill_between(x_vals, low, high, color=color, **fill_kwargs)


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

    # ------------------------------------------------------------------ #
    # saturation_curves
    # ------------------------------------------------------------------ #

    def saturation_curves(
        self,
        curve: xr.DataArray,
        original_scale: bool = True,
        n_samples: int = 10,
        hdi_prob: float = 0.94,
        random_seed: np.random.Generator | None = None,
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
        """Overlay saturation curves with posterior sample lines and HDI bands.

        Renders a scatter plot of observed data, posterior sample curves,
        and a credible interval band for each channel panel.

        Parameters
        ----------
        curve : xr.DataArray
            Posterior-predictive saturation curves, typically from
            ``mmm.saturation.sample_curve(...)``.
            Expected dims: ``(chain, draw, channel, [custom_dims], x)``.
        original_scale : bool, default True
            If True, scale both curve y-values and x-axis to original units.
        n_samples : int, default 10
            Number of posterior sample curves to draw per panel.
            Set to 0 to disable sample curves.
        hdi_prob : float, default 0.94
            Credible interval probability for the HDI band.
        random_seed : np.random.Generator, optional
            RNG for reproducible sample selection.
        apply_cost_per_unit : bool, default True
            If True and cost-per-unit data is available, the x-axis shows spend.
        idata : az.InferenceData, optional
            Override instance data.
        dims : dict, optional
            Dimension filters.
        figsize : tuple[float, float], optional
            Convenience shorthand for figure size.
        plot_collection : PlotCollection, optional
            Plot onto an existing ``PlotCollection``.
        backend : str, optional
            Rendering backend.
        visuals : dict, optional
            Element-level customization. Keys:
            ``"scatter"`` (observed data points),
            ``"samples"`` (posterior sample curves),
            ``"hdi"`` (HDI band).
            Set a key to ``False`` to disable that element.
        aes_by_visuals : dict, optional
            Aesthetic mapping per visual element.
        return_as_pc : bool, default False
            Return ``PlotCollection`` instead of matplotlib tuple.
        **pc_kwargs
            Forwarded to ``PlotCollection.wrap()``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection
        """
        data = MMMIDataWrapper(idata) if idata is not None else self._data
        rng = random_seed if random_seed is not None else np.random.default_rng()

        pc_kwargs = _process_plot_params(
            figsize=figsize,
            plot_collection=plot_collection,
            backend=backend,
            return_as_pc=return_as_pc,
            **pc_kwargs,
        )

        contributions = data.get_channel_contributions(original_scale=original_scale)
        mean_contrib = contributions.mean(dim=["chain", "draw"])
        x_obs = _get_channel_x_data(data, apply_cost_per_unit)
        x_obs, mean_contrib = xr.broadcast(x_obs, mean_contrib)
        scatter_ds = xr.Dataset({"x": x_obs, "y": mean_contrib})

        if original_scale:
            target_scale = data.get_target_scale()
            curve_data = curve * target_scale

            channel_scale = data.get_channel_scale()
            if apply_cost_per_unit:
                avg_cpu = data.get_avg_cost_per_unit()
                x_scale = channel_scale * avg_cpu
            else:
                x_scale = channel_scale
        else:
            curve_data = curve
            x_scale = None

        _validate_dims(scatter_ds, dims)

        if dims:
            sel_kwargs = _dims_to_sel_kwargs(dims)
            scatter_ds = scatter_ds.sel(**sel_kwargs)
            curve_sel = {k: v for k, v in sel_kwargs.items() if k in curve_data.dims}
            if curve_sel:
                curve_data = curve_data.sel(**curve_sel)
            if x_scale is not None:
                xscale_sel = {k: v for k, v in sel_kwargs.items() if k in x_scale.dims}
                if xscale_sel:
                    x_scale = x_scale.sel(**xscale_sel)

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

        colors = channel_color_map(data.channels)
        axes_da = pc.viz.ds["plot"]

        hdi_kw = _get_visual_kwargs(visuals, "hdi", {"alpha": 0.2})
        samples_kw = _get_visual_kwargs(
            visuals, "samples", {"linewidth": 0.5, "alpha": 0.4}
        )
        scatter_kw = _get_visual_kwargs(visuals, "scatter", {"alpha": 0.8, "s": 20})

        for ax, curve_panel in _iter_panels_da(axes_da, curve_data):
            ch = (
                str(curve_panel.coords["channel"].item())
                if "channel" in curve_panel.coords
                else ""
            )
            color = colors.get(ch, "C0")

            raw_x = curve_panel.coords["x"].values
            if x_scale is not None:
                ch_scale_sel = {
                    k: curve_panel.coords[k].item()
                    for k in x_scale.dims
                    if k in curve_panel.coords
                }
                scale_val = float(x_scale.sel(**ch_scale_sel).item())
                panel_x = raw_x * scale_val
            else:
                panel_x = raw_x

            if hdi_kw is not None:
                _hdi_band_visual(
                    curve_panel,
                    ax,
                    x_vals=panel_x,
                    hdi_prob=hdi_prob,
                    color=color,
                    fill_kwargs=hdi_kw,
                )

            if samples_kw is not None and n_samples > 0:
                _sample_curves_visual(
                    curve_panel,
                    ax,
                    x_vals=panel_x,
                    n_samples=n_samples,
                    rng=rng,
                    color=color,
                    line_kwargs=samples_kw,
                )

        if scatter_kw is not None:
            for ax, panel_data in _iter_panels(axes_da, scatter_ds):
                _scatter_visual(
                    panel_data, ax, color_map=colors, scatter_kwargs=scatter_kw
                )

        x_label = _x_axis_label(data, apply_cost_per_unit)
        for ax in np.array(pc.viz.ds["plot"].values).flat:
            ax.set_xlabel(x_label)
            ax.set_ylabel("Channel Contributions")

        return _extract_matplotlib_result(pc, return_as_pc)
