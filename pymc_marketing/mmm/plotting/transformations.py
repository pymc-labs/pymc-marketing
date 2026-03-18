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
import arviz_plots as azp
import numpy as np
import xarray as xr
from arviz_base.labels import DimCoordLabeller, NoVarLabeller, mix_labellers
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pymc_marketing.data.idata import MMMIDataWrapper
from pymc_marketing.mmm.plotting._helpers import (
    _extract_matplotlib_result,
    _process_plot_params,
    _select_dims,
)


def _x_axis_label(data: MMMIDataWrapper, apply_cost_per_unit: bool) -> str:
    """Return the x-axis label based on cost-per-unit availability."""
    if apply_cost_per_unit and data.cost_per_unit is not None:
        return "Spend"
    return "Channel Data"


def _get_channel_x_data(
    data: MMMIDataWrapper, apply_cost_per_unit: bool
) -> xr.DataArray:
    """Return channel spend or raw channel data based on cost-per-unit flag."""
    if apply_cost_per_unit:
        return data.get_channel_spend()
    return data.get_channel_data()


class TransformationPlots:
    """Channel transformation plots (saturation scatter and curves).

    Parameters
    ----------
    data : MMMIDataWrapper
        Validated wrapper around the fitted model's ``InferenceData``.
    """

    def __init__(self, data: MMMIDataWrapper) -> None:
        self._data = data

    def saturation_scatterplot(
        self,
        original_scale: bool = True,
        apply_cost_per_unit: bool = True,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
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
        backend : str, optional
            Rendering backend (``"matplotlib"``, ``"plotly"``, ``"bokeh"``).
        return_as_pc : bool, default False
            If True, return the ``PlotCollection`` instead of the
            matplotlib tuple.
        **pc_kwargs
            Forwarded to ``PlotCollection.wrap()``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection
        """
        # TODO: decide how to validate!!!
        data = MMMIDataWrapper(idata) if idata is not None else self._data

        pc_kwargs = _process_plot_params(
            figsize=figsize,
            backend=backend,
            return_as_pc=return_as_pc,
            **pc_kwargs,
        )

        contributions = data.get_channel_contributions(original_scale=original_scale)
        mean_contrib = contributions.mean(dim=["chain", "draw"])

        x_data = _get_channel_x_data(data, apply_cost_per_unit)
        scatter_ds = xr.Dataset({"x": x_data, "y": mean_contrib})

        scatter_ds = _select_dims(scatter_ds, dims)

        pc = PlotCollection.grid(
            scatter_ds[["y"]],
            cols=data.custom_dims,
            backend=backend,
            rows=["channel"],
            aes={"color": ["channel"]},
            **pc_kwargs,
        )

        pc.map(
            azp.visuals.scatter_xy,
            x=scatter_ds["x"],
        )

        pc.map(
            azp.visuals.labelled_x,
            text=_x_axis_label(data, apply_cost_per_unit),
            ignore_aes={"color"},
        )

        pc.map(
            azp.visuals.labelled_y,
            text="Channel Contributions",
            ignore_aes={"color"},
        )

        pc.map(
            azp.visuals.labelled_title,
            subset_info=True,
            labeller=mix_labellers((NoVarLabeller, DimCoordLabeller))(),
            ignore_aes={"color"},
        )

        return _extract_matplotlib_result(pc, return_as_pc)

    def saturation_curves(
        self,
        curves: xr.DataArray,
        original_scale: bool = True,
        n_samples: int = 10,
        hdi_prob: float = 0.94,
        random_seed: np.random.Generator | None = None,
        apply_cost_per_unit: bool = True,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Overlay saturation curves with posterior sample lines and HDI bands.

        Renders a scatter plot of observed data, posterior sample curves,
        and a credible interval band for each channel panel.

        Parameters
        ----------
        curves : xr.DataArray
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

        pc: PlotCollection = self.saturation_scatterplot(
            original_scale=original_scale,
            apply_cost_per_unit=apply_cost_per_unit,
            idata=idata,
            dims=dims,
            figsize=figsize,
            backend=backend,
            return_as_pc=True,
            **pc_kwargs,
        )

        pc_kwargs = _process_plot_params(
            figsize=figsize,
            backend=backend,
            return_as_pc=return_as_pc,
            **pc_kwargs,
        )

        if original_scale:
            # update y values to original scale
            # TODO: what happens if curves are already in original scale?
            target_scale = data.get_target_scale()
            curve_data = curves * target_scale

            x_scale = data.get_channel_scale()
            if apply_cost_per_unit:
                x_scale *= data.get_avg_cost_per_unit()

            x_data = curve_data["x"].copy()
            curve_data["x"] = range(len(x_data))
            spend_data: xr.DataArray = x_data * x_scale
        else:
            curve_data = curves

        curve_data = _select_dims(curve_data, dims)

        # add the hdi bands
        if hdi_prob is not None:
            hdi = curve_data.azstats.hdi(hdi_prob)
            print(hdi.sizes)
            pc.map(
                azp.visuals.fill_between_y,
                x=spend_data,
                y_bottom=hdi.sel(ci_bound="lower"),
                y_top=hdi.sel(ci_bound="upper"),
                alpha=0.2,
            )

        mean_curve = curve_data.mean(dim=["chain", "draw"])
        pc.map(azp.visuals.line_xy, x=spend_data, y=mean_curve)

        if n_samples > 0:
            # sample the curves
            rng = random_seed if random_seed is not None else np.random.default_rng()
            stacked = curve_data.stack(sample=("chain", "draw"))
            idx = rng.choice(stacked.sizes["sample"], size=n_samples, replace=False)
            sampled_curves = stacked.isel(sample=idx).to_dataset(name="y")
            sampled_curves["x"] = spend_data

            # plot the sampled curves
            for i in range(n_samples):
                pc.map(
                    azp.visuals.line_xy,
                    x=spend_data,
                    y=sampled_curves.isel(sample=i),
                    alpha=0.3,
                )

        return _extract_matplotlib_result(pc, return_as_pc)
