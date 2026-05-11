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
    _plot_timeseries_channel,
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

    @staticmethod
    def _plot_waterfall_panel(
        ax: Axes,
        entries: list[tuple[str, float]],
        bar_kwargs: dict,
    ) -> None:
        """Draw a single waterfall panel onto *ax*.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes to draw on.
        entries : list of (label, value)
            Ordered contribution components. A "total" bar is appended automatically.
        bar_kwargs : dict
            Extra kwargs forwarded to ``ax.barh()``.
        """
        total = sum(v for _, v in entries)
        components = [*entries, ("total", total)]

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
                **bar_kwargs,
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
        ax.axvline(0, color="black", linewidth=0.8)

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

        # Find date coordinate from any contribution that has a date dim.
        # Fall back to the raw posterior coordinate so baseline-only plots work.
        dates_coord = next(
            (
                contributions_ds[k].coords["date"]
                for k in contributions_ds.data_vars
                if "date" in contributions_ds[k].dims
            ),
            None,
        )
        if dates_coord is None:
            posterior = data.idata.posterior
            if "date" in posterior.coords:
                dates_coord = posterior.coords["date"]

        # Build flat entries: each entry has dims (chain, draw, date[, extra_dims])
        # so the rendering loop below is unchanged.
        entries_ds = xr.Dataset()

        if "channels" in contributions_ds:
            ch_da = contributions_ds["channels"]
            for ch in ch_da.coords["channel"].values:
                entries_ds[f"channel={ch}"] = _select_dims(
                    ch_da.sel(channel=ch), dims, allow_missing=True
                )

        if "baseline" in contributions_ds:
            bl_da = contributions_ds["baseline"]
            # baseline has no date dim — broadcast it over the date axis
            bl_broadcast = (
                bl_da.expand_dims({"date": dates_coord})
                if dates_coord is not None
                else bl_da
            )
            entries_ds["baseline"] = _select_dims(
                bl_broadcast, dims, allow_missing=True
            )

        if "controls" in contributions_ds:
            ctrl_da = contributions_ds["controls"]
            # sum over the control dim → single time-series
            entries_ds["controls"] = _select_dims(
                ctrl_da.sum(dim="control"), dims, allow_missing=True
            )

        if "seasonality" in contributions_ds:
            seas_da = contributions_ds["seasonality"]
            entries_ds["seasonality"] = _select_dims(seas_da, dims, allow_missing=True)

        if not entries_ds:
            raise ValueError(
                "No contribution data found after filtering. "
                "Check that the model has the requested contribution types."
            )

        # Turn Dataset to array.
        entries = entries_ds.to_array(dim="component").to_dataset(name="contribution")

        pc = _plot_timeseries_channel(
            entries,
            sample_dims=["chain", "draw"],
            color_dim="component",
            extra_dims=extra_dims,
            hdi_prob=hdi_prob,
            backend=backend,
            line_kwargs=line_kwargs,
            hdi_kwargs=hdi_kwargs,
            **pc_kwargs,
        )

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

        # Build entries: (label, xr.DataArray) where DataArray has dims (extra_dims,) or scalar
        entries: list[tuple[str, xr.DataArray]] = []

        for ds_key, coord_dim in [
            ("baseline", None),
            ("channels", "channel"),
            ("controls", "control"),
            ("seasonality", None),
        ]:
            if ds_key not in contributions_ds:
                continue
            da = _select_dims(contributions_ds[ds_key], dims, allow_missing=True)
            # mean over whichever of chain/draw/date are present (baseline has no date dim)
            mean_dims = [d for d in ("chain", "draw", "date") if d in da.dims]
            if coord_dim is not None:
                for val in da.coords[coord_dim].values:
                    entries.append(
                        (str(val), da.sel({coord_dim: val}).mean(dim=mean_dims))
                    )
            else:
                entries.append((ds_key, da.mean(dim=mean_dims)))

        # Determine panel combos from extra dims
        if extra_dims:
            # Find an entry that has all extra dimensions
            ref_da = None
            for _, da in entries:
                if all(d in da.coords for d in extra_dims):
                    ref_da = da
                    break

            if ref_da is None:
                # Fallback: use constant_data coordinates
                ref_da = data.idata.constant_data

            coord_values = [ref_da.coords[d].values for d in extra_dims]
            combos = list(itertools.product(*coord_values))
        else:
            combos = [()]

        n_panels = len(combos)
        fig, axes_raw = plt.subplots(
            n_panels, 1, figsize=figsize or (6, 4 * n_panels), squeeze=False
        )
        axes_flat = axes_raw.flatten()

        reserved_keys = {"y", "width", "left", "color"}
        if bar_kwargs:
            conflict = reserved_keys & set(bar_kwargs.keys())
            if conflict:
                raise ValueError(
                    f"bar_kwargs keys conflict with positional bar arguments: {conflict}. "
                    "Do not pass 'y', 'width', 'left', or 'color' in bar_kwargs."
                )
        safe_bar_kwargs = {"height": 0.5, **(bar_kwargs or {})}

        for panel_idx, combo in enumerate(combos):
            ax = axes_flat[panel_idx]
            sel_kwargs = dict(zip(extra_dims, combo, strict=True))

            # Extract scalar (label, float) for this panel
            panel_entries: list[tuple[str, float]] = []
            for label, da in entries:
                if sel_kwargs:
                    # Only select dimensions that exist in this DataArray
                    sel_dims = {k: [v] for k, v in sel_kwargs.items() if k in da.dims}
                    if sel_dims:
                        da = da.sel(**sel_dims).squeeze()
                panel_entries.append((label, float(da.values)))

            title = (
                " | ".join(f"{k}={v}" for k, v in sel_kwargs.items())
                if sel_kwargs
                else ""
            )
            if title:
                ax.set_title(title)

            self._plot_waterfall_panel(ax, panel_entries, safe_bar_kwargs)

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
            combined=True,
            ci_kind="hdi",
            ci_probs=(0.5, hdi_prob),
            labels=set(share_ds.dims) - set(["chain", "draw"]),
            backend=backend,
            **pc_kwargs,
        )
        return _extract_matplotlib_result(pc, return_as_pc)
