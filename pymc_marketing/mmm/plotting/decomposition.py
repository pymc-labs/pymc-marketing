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
"""Decomposition namespace — contribution waterfall and time-series plots."""

from __future__ import annotations

import itertools
import warnings
from typing import Any, Literal

import arviz as az
import arviz_plots as azp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
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


class DecompositionPlots:
    """Decomposition plots for fitted MMM models.

    Provides three methods to visualize how the model decomposes the target:

    - ``contributions_over_time`` — Time-series of each contribution type with HDI.
    - ``waterfall``              — Horizontal waterfall chart of mean contributions.
    - ``channel_share_hdi``     — Forest plot of each channel's share of total response.

    Parameters
    ----------
    data : MMMIDataWrapper
        Validated wrapper around the fitted model's InferenceData.
    """

    def __init__(self, data: MMMIDataWrapper) -> None:
        self._data = data

    def contributions_over_time(
        self,
        include: list[Literal["channels", "baseline", "controls", "seasonality"]]
        | None = None,
        hdi_prob: float = 0.94,
        original_scale: bool = True,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        line_kwargs: dict[str, Any] | None = None,
        hdi_kwargs: dict[str, Any] | None = None,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Plot time-series contributions for selected contribution types with HDI bands.

        Creates one panel per extra-dimension combination (e.g. one per geo for
        geo-segmented models). Each panel overlays one mean line and HDI band per
        contribution type.

        Parameters
        ----------
        include : list of {"channels", "baseline", "controls", "seasonality"}, optional
            Which contribution types to plot. ``None`` means all available.
        hdi_prob : float, default 0.94
            Probability mass for the HDI band.
        original_scale : bool, default True
            Whether to return contributions in original scale.
        idata : az.InferenceData, optional
            Override instance data for this call only.
        dims : dict[str, Any], optional
            Subset dimensions, e.g. ``{"geo": ["CA"]}``.
        figsize : tuple[float, float], optional
            Injected into ``figure_kwargs``.
        backend : str, optional
            Rendering backend. Non-matplotlib requires ``return_as_pc=True``.
        return_as_pc : bool, default False
            If True, return the ``PlotCollection`` instead of ``(Figure, NDArray[Axes])``.
        line_kwargs : dict, optional
            Extra kwargs forwarded to ``azp.visuals.line_xy`` for every mean line.
        hdi_kwargs : dict, optional
            Extra kwargs forwarded to ``azp.visuals.fill_between_y`` for every HDI band.
        **pc_kwargs
            Forwarded to ``PlotCollection.wrap()``. Use ``col_wrap`` to override the
            default single-column layout.

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

        all_keys: set[str] = {"channels", "baseline", "controls", "seasonality"}
        include_set: set[str] = set(include) if include is not None else all_keys
        invalid = include_set - all_keys
        if invalid:
            raise ValueError(
                f"Unknown contribution type(s): {invalid}. Valid options: {all_keys}"
            )

        contributions_ds = data.get_contributions(
            original_scale=original_scale,
            include_baseline="baseline" in include_set,
            include_controls="controls" in include_set,
            include_seasonality="seasonality" in include_set,
        )
        if "channels" not in include_set:
            contributions_ds = contributions_ds.drop_vars("channels", errors="ignore")

        extra_dims = list(data.custom_dims)
        keep_dims = {"date", "chain", "draw"} | set(extra_dims)

        # Collapse model-specific dims (e.g. channel, control) into the time axis
        reduced: dict[str, xr.DataArray] = {}
        for key in contributions_ds.data_vars:
            da = contributions_ds[key]
            to_sum = [d for d in da.dims if d not in keep_dims]
            if to_sum:
                warnings.warn(
                    f"contributions_over_time: summing over dimension(s) {to_sum} "
                    f"for contribution '{key}'.",
                    UserWarning,
                    stacklevel=2,
                )
                da = da.sum(dim=to_sum)
            da = _select_dims(da, dims)
            reduced[key] = da

        if not reduced:
            raise ValueError(
                "No contribution data found after filtering. "
                "Check that the model has the requested contribution types."
            )

        first_da = next(iter(reduced.values()))
        dates = first_da.coords["date"].values

        layout_ds = (
            first_da.mean(dim=("chain", "draw"))
            .isel(date=0, drop=True)
            .to_dataset(name="_layout")
        )
        pc_kwargs.setdefault("col_wrap", 1)
        pc = PlotCollection.wrap(
            layout_ds,
            cols=extra_dims,
            backend=backend,
            **pc_kwargs,
        )

        for i, (label, da) in enumerate(reduced.items()):
            mean_da = da.mean(dim=("chain", "draw"))
            hdi_da = da.azstats.hdi(hdi_prob)
            color = f"C{i}"

            pc.map(
                azp.visuals.fill_between_y,
                x=dates,
                y_bottom=hdi_da.sel(ci_bound="lower"),
                y_top=hdi_da.sel(ci_bound="upper"),
                **{"alpha": 0.2, "color": color, **(hdi_kwargs or {})},
            )
            pc.map(
                azp.visuals.line_xy,
                x=dates,
                y=mean_da,
                **{"label": label, "color": color, **(line_kwargs or {})},
            )

        pc.map(azp.visuals.labelled_x, text="Date", ignore_aes={"color"})
        pc.map(azp.visuals.labelled_y, text="Contribution", ignore_aes={"color"})
        pc.map(azp.visuals.labelled_title, subset_info=True, ignore_aes={"color"})

        return _extract_matplotlib_result(pc, return_as_pc)

    def waterfall(
        self,
        original_scale: bool = True,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        bar_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure, NDArray[Axes]]:
        """Horizontal waterfall chart showing mean contribution per component.

        One subplot per extra-dimension combination (e.g. per geo). Each subplot
        shows how each contribution type (baseline, channels, controls, seasonality)
        builds up to the total.

        Parameters
        ----------
        original_scale : bool, default True
            Whether to plot contributions in original scale.
        idata : az.InferenceData, optional
            Override instance data for this call only.
        dims : dict[str, Any], optional
            Subset dimensions, e.g. ``{"geo": ["CA"]}``.
        figsize : tuple[float, float], optional
            Passed to ``plt.subplots()``.
        bar_kwargs : dict, optional
            Extra kwargs forwarded to ``ax.barh()``. Cannot conflict with
            positional arguments (``y``, ``width``, ``left``).

        Returns
        -------
        tuple[Figure, NDArray[Axes]]
        """
        data = (
            MMMIDataWrapper(idata, schema=self._data.schema)
            if idata is not None
            else self._data
        )

        contributions_ds = data.get_contributions(original_scale=original_scale)
        extra_dims = list(data.custom_dims)
        keep_dims = {"date", "chain", "draw"} | set(extra_dims)

        # Reduce each DataArray to (chain, draw, date[, extra_dims]) then take mean
        # Result: scalar or (extra_dims,) DataArray per contribution type
        means: dict[str, xr.DataArray] = {}
        for key in contributions_ds.data_vars:
            da = contributions_ds[key]
            to_sum = [d for d in da.dims if d not in keep_dims]
            if to_sum:
                da = da.sum(dim=to_sum)
            da = _select_dims(da, dims)
            means[key] = da.mean(dim=("chain", "draw", "date"))

        # Determine subplot combos
        if extra_dims:
            coord_values = [
                means[next(iter(means))].coords[d].values for d in extra_dims
            ]
            combos = list(itertools.product(*coord_values))
        else:
            combos = [()]

        n_panels = len(combos)
        fig, axes_raw = plt.subplots(
            1, n_panels, figsize=figsize or (6 * n_panels, 4), squeeze=False
        )
        axes_flat = axes_raw.flatten()

        reserved_keys = {"y", "width", "left"}
        if bar_kwargs:
            conflict = reserved_keys & set(bar_kwargs.keys())
            if conflict:
                raise ValueError(
                    f"bar_kwargs keys conflict with positional bar arguments: {conflict}. "
                    "Do not pass 'y', 'width', or 'left' in bar_kwargs."
                )
        safe_bar_kwargs = {"height": 0.5, **(bar_kwargs or {})}

        ordered_keys = [
            k for k in ["baseline", "channels", "controls", "seasonality"] if k in means
        ]

        for panel_idx, combo in enumerate(combos):
            ax = axes_flat[panel_idx]
            sel_kwargs = dict(zip(extra_dims, combo, strict=True))

            # Extract scalar values using positional indexing
            values: dict[str, float] = {}
            for key in ordered_keys:
                da = means[key]
                if sel_kwargs:
                    da = da.sel(**{k: [v] for k, v in sel_kwargs.items()}).squeeze()
                values[key] = float(da.values)

            total = sum(values.values())
            components = [*list(values.items()), ("total", total)]

            running = 0.0
            for bar_idx, (label, val) in enumerate(components):
                if label == "total":
                    color = "grey"
                    left = 0.0
                    width = val
                else:
                    color = "green" if val >= 0 else "red"
                    left = running
                    width = val
                    running += val

                ax.barh(
                    y=bar_idx,
                    width=width,
                    left=left,
                    color=color,
                    **safe_bar_kwargs,
                )
                pct = 100 * val / total if total != 0 else 0.0
                ax.text(
                    left + width / 2,
                    bar_idx,
                    f"{val:.1f} ({pct:.1f}%)",
                    va="center",
                    ha="center",
                    fontsize=8,
                )

            ax.set_yticks(range(len(components)))
            ax.set_yticklabels([c[0] for c in components])
            title = (
                " | ".join(f"{k}={v}" for k, v in sel_kwargs.items())
                if sel_kwargs
                else ""
            )
            if title:
                ax.set_title(title)
            ax.axvline(0, color="black", linewidth=0.8)

        fig.tight_layout()
        return fig, np.atleast_1d(np.array(axes_flat))

    def channel_share_hdi(
        self,
        hdi_prob: float = 0.94,
        idata: az.InferenceData | None = None,
        dims: dict[str, Any] | None = None,
        figsize: tuple[float, float] | None = None,
        backend: str | None = None,
        return_as_pc: bool = False,
        **pc_kwargs,
    ) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
        """Forest plot of each channel's share of total channel contribution.

        Computes each channel's contribution as a fraction of total channel
        contribution (summed over dates), then plots the HDI for each channel.

        Parameters
        ----------
        hdi_prob : float, default 0.94
            HDI probability mass.
        idata : az.InferenceData, optional
            Override instance data for this call only.
        dims : dict[str, Any], optional
            Subset dimensions, e.g. ``{"geo": ["CA"]}``.
        figsize : tuple[float, float], optional
            Injected into ``figure_kwargs``.
        backend : str, optional
            Rendering backend. Non-matplotlib requires ``return_as_pc=True``.
        return_as_pc : bool, default False
            If True, return the ``PlotCollection``.
        **pc_kwargs
            Forwarded to ``azp.plot_forest()``.

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

        # (chain, draw, date, channel[, extra_dims])
        channel_contributions = data.get_channel_contributions(original_scale=True)
        channel_contributions = _select_dims(channel_contributions, dims)

        # Sum over date → (chain, draw, channel[, extra_dims])
        summed = channel_contributions.sum(dim="date")

        # Compute share per channel
        total = summed.sum(dim="channel")
        shares = summed / total
        shares.name = "channel_share"

        share_ds = shares.to_dataset(name="channel_share")

        pc = azp.plot_forest(
            share_ds,
            var_names=["channel_share"],
            combined=True,
            ci_kind="hdi",
            ci_probs=(0.5, hdi_prob),
            backend=backend,
            **pc_kwargs,
        )
        return _extract_matplotlib_result(pc, return_as_pc)
