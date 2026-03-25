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

import warnings
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

_SCALED_SPACE_MAX_THRESHOLD = 10.0


def _ensure_chain_draw_dims(curves: xr.DataArray) -> xr.DataArray:
    """Ensure curves have ``(chain, draw)`` dimensions for ArviZ compatibility.

    Curves from ``mmm.sample_saturation_curve()`` have a flat ``sample``
    dimension, while ``mmm.saturation.sample_curve(params)`` returns
    ``(chain, draw)``.  Downstream code (HDI, mean, stacking) requires
    ``(chain, draw)`` — this function bridges the gap.

    Handles three input formats:

    * ``(chain, draw, ...)`` — returned as-is (copy).
    * ``sample`` as a MultiIndex over ``(chain, draw)`` — unstacked.
    * ``sample`` as a plain integer index — expanded to
      ``chain=0, draw=0..N-1``.
    """
    if "chain" in curves.dims and "draw" in curves.dims:
        return curves.copy()

    if "sample" not in curves.dims:
        raise ValueError(
            "Curves must have either ('chain', 'draw') or 'sample' dimensions. "
            f"Got: {list(curves.dims)}"
        )

    # MultiIndex sample (chain/draw are non-dim coords) — just unstack
    if "chain" in curves.coords and "draw" in curves.coords:
        return curves.unstack("sample")

    # Plain integer sample — promote to single-chain (chain=0)
    n_samples = curves.sizes["sample"]
    return (
        curves.assign_coords(chain=("sample", np.zeros(n_samples, dtype=int)))
        .assign_coords(draw=("sample", np.arange(n_samples)))
        .set_index(sample=["chain", "draw"])
        .unstack("sample")
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
        scatter_kwargs: dict[str, Any] | None = None,
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
        scatter_kwargs : dict, optional
            Extra keyword arguments forwarded to the scatter visual
            (``azp.visuals.scatter_xy``).
        **pc_kwargs
            Forwarded to ``PlotCollection.wrap()``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection
        """
        data = (
            MMMIDataWrapper(idata, schema=self._data.schema)
            if idata is not None
            else self._data
        )

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
            **{"alpha": 0.8, **(scatter_kwargs or {})},
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
        hdi_prob: float | None = 0.94,
        random_seed: np.random.Generator | None = None,
        apply_cost_per_unit: bool = True,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        scatter_kwargs: dict[str, Any] | None = None,
        hdi_kwargs: dict[str, Any] | None = None,
        mean_curve_kwargs: dict[str, Any] | None = None,
        sample_curves_kwargs: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Overlay saturation curves with posterior sample lines and HDI bands.

        Renders a scatter plot of observed data, posterior sample curves,
        and a credible interval band for each channel panel.

        The ``curves`` y-values are plotted **as-is** — this function does
        **not** rescale them.  Pass curves whose scale matches the
        ``original_scale`` flag so they align with the scatter plot:

        * ``original_scale=True`` (default) — pass curves generated with
          ``mmm.sample_saturation_curve(original_scale=True)`` so that
          y-values are already in original (un-scaled) units.
        * ``original_scale=False`` — pass curves from
          ``mmm.saturation.sample_curve(...)`` or
          ``mmm.sample_saturation_curve(original_scale=False)`` (model-
          internal scaled space).

        A heuristic warning is emitted when the curve magnitude appears
        inconsistent with the requested scale.

        Parameters
        ----------
        curves : xr.DataArray
            Posterior-predictive saturation curves.  Typical sources:

            * ``mmm.sample_saturation_curve(original_scale=True)`` — curves
              already in original scale (use with ``original_scale=True``).
            * ``mmm.sample_saturation_curve(original_scale=False)`` or
              ``mmm.saturation.sample_curve(params)`` — curves in scaled
              space (use with ``original_scale=False``).

            Expected dims: ``(chain, draw, channel, [custom_dims], x)``.
        original_scale : bool, default True
            Controls the scatter-plot scale (contributions).  The caller
            must ensure ``curves`` are in the matching scale.
        n_samples : int, default 10
            Number of posterior sample curves to draw per panel.
            Set to 0 to disable sample curves.
        hdi_prob : float or None, default 0.94
            Credible interval probability for the HDI band.
            Set to None to disable HDI band rendering.
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
        backend : str, optional
            Rendering backend.
        return_as_pc : bool, default False
            Return ``PlotCollection`` instead of matplotlib tuple.
        scatter_kwargs : dict, optional
            Extra keyword arguments forwarded to the scatter visual
            (``azp.visuals.scatter_xy``).
        hdi_kwargs : dict, optional
            Extra keyword arguments forwarded to the HDI band visual
            (``azp.visuals.fill_between_y``).
        mean_curve_kwargs : dict, optional
            Extra keyword arguments forwarded to the mean curve visual
            (``azp.visuals.line_xy``).
        sample_curves_kwargs : dict, optional
            Extra keyword arguments forwarded to each sample curve visual
            (``azp.visuals.line_xy``).
        **pc_kwargs
            Forwarded to ``PlotCollection.wrap()``.

        Returns
        -------
        tuple[Figure, NDArray[Axes]] or PlotCollection
        """
        data = (
            MMMIDataWrapper(idata, schema=self._data.schema)
            if idata is not None
            else self._data
        )

        _process_plot_params(
            figsize=figsize,
            backend=backend,
            return_as_pc=return_as_pc,
            **pc_kwargs,
        )

        pc: PlotCollection = self.saturation_scatterplot(
            original_scale=original_scale,
            apply_cost_per_unit=apply_cost_per_unit,
            idata=idata,
            dims=dims,
            figsize=figsize,
            backend=backend,
            return_as_pc=True,
            scatter_kwargs=scatter_kwargs,
            **pc_kwargs,
        )

        curves = _ensure_chain_draw_dims(curves)

        curve_max = float(curves.max())
        looks_scaled = curve_max < _SCALED_SPACE_MAX_THRESHOLD
        if original_scale == looks_scaled:
            expected = "original" if original_scale else "scaled"
            actual = "scaled" if looks_scaled else "original"
            warnings.warn(
                f"original_scale={original_scale} but curves.max()="
                f"{curve_max:.4g} suggests the curves are in {actual} "
                f"space, not {expected}. Pass curves generated with "
                f"mmm.sample_saturation_curve(original_scale={original_scale}).",
                UserWarning,
                stacklevel=2,
            )

        # get values for x-axis (spend)
        x_scale = data.get_channel_scale()
        if apply_cost_per_unit:
            x_scale *= data.get_avg_cost_per_unit()
        x_data = curves["x"].copy()
        curves["x"] = range(len(x_data))
        spend_data: xr.DataArray = x_data * x_scale

        # select dimensions
        curves = _select_dims(curves, dims)
        spend_data = _select_dims(spend_data, dims)

        # plot the hdi band
        if hdi_prob is not None:
            hdi = curves.azstats.hdi(hdi_prob)
            pc.map(
                azp.visuals.fill_between_y,
                x=spend_data,
                y_bottom=hdi.sel(ci_bound="lower"),
                y_top=hdi.sel(ci_bound="upper"),
                **{"alpha": 0.2, **(hdi_kwargs or {})},
            )

        # plot the mean curve
        mean_curve = curves.mean(dim=["chain", "draw"])
        pc.map(
            azp.visuals.line_xy,
            x=spend_data,
            y=mean_curve,
            **(mean_curve_kwargs or {}),
        )

        if n_samples > 0:
            # sample the curves
            rng = random_seed if random_seed is not None else np.random.default_rng()
            stacked = curves.stack(sample=("chain", "draw"))
            n_samples = min(n_samples, stacked.sizes["sample"])
            idx = rng.choice(stacked.sizes["sample"], size=n_samples, replace=False)
            sampled_curves = stacked.isel(sample=idx).to_dataset(name="y")
            sampled_curves["x"] = spend_data

            # plot the sampled curves
            for i in range(n_samples):
                pc.map(
                    azp.visuals.line_xy,
                    x=spend_data,
                    y=sampled_curves.isel(sample=i),
                    **{"alpha": 0.3, **(sample_curves_kwargs or {})},
                )

        return _extract_matplotlib_result(pc, return_as_pc)
