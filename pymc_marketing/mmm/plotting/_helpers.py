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
"""Shared helpers for MMMPlotSuite namespace classes."""

from __future__ import annotations

import warnings
from typing import Any

import arviz_plots as azp
import numpy as np
import xarray as xr
from arviz_base.labels import DimCoordLabeller, NoVarLabeller, mix_labellers
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pymc_marketing.mmm.xarray_utils import (
    _apply_aggregation,
    _dims_to_sel_kwargs,
    _ensure_chain_draw_dims,
    _select_dims,
    _validate_dims,
)

__all__ = [
    "_apply_aggregation",
    "_dims_to_sel_kwargs",
    "_ensure_chain_draw_dims",
    "_extract_matplotlib_result",
    "_plot_timeseries_channel",
    "_process_plot_params",
    "_select_dims",
    "_validate_dims",
]


def _process_plot_params(
    figsize: tuple[float, float] | None,
    backend: str | None,
    return_as_pc: bool,
    **pc_kwargs,
) -> dict:
    """Validate and normalize the standard customization parameters.

    Enforces the parameter interaction rules from the figure customization
    design doc.  Returns the (possibly modified) ``pc_kwargs`` dict ready to
    be forwarded to ``PlotCollection.wrap()`` or ``.grid()``.

    Parameters
    ----------
    figsize : tuple[float, float] or None
        Convenience shorthand injected into ``figure_kwargs``.
    backend : str or None
        Rendering backend (``"matplotlib"``, ``"plotly"``, ``"bokeh"``).
    return_as_pc : bool
        If False and ``backend`` is not matplotlib/None, raises.
    **pc_kwargs
        Forwarded to ``PlotCollection.wrap()`` / ``.grid()``.

    Returns
    -------
    dict
        Cleaned ``pc_kwargs``.
    """
    if not return_as_pc and backend is not None and backend != "matplotlib":
        raise ValueError(
            f"backend='{backend}' requires return_as_pc=True. "
            "Non-matplotlib backends cannot return (Figure, NDArray[Axes])."
        )

    if figsize is not None:
        fig_kwargs = pc_kwargs.pop("figure_kwargs", {})
        if "figsize" in fig_kwargs:
            warnings.warn(
                "figsize parameter overrides figure_kwargs['figsize'].",
                UserWarning,
                stacklevel=2,
            )
        fig_kwargs["figsize"] = figsize
        pc_kwargs["figure_kwargs"] = fig_kwargs

    return pc_kwargs


def _extract_matplotlib_result(
    pc: PlotCollection,
    return_as_pc: bool,
) -> tuple[Figure, NDArray[Axes]] | PlotCollection:
    """Convert a ``PlotCollection`` to ``(Figure, NDArray[Axes])`` or return as-is.

    Parameters
    ----------
    pc : PlotCollection
        The plot collection to extract from.
    return_as_pc : bool
        If True, return the ``PlotCollection`` directly.

    Returns
    -------
    tuple[Figure, NDArray[Axes]] or PlotCollection
        Standard matplotlib tuple when ``return_as_pc=False``,
        otherwise the original ``PlotCollection``.
    """
    if return_as_pc:
        return pc
    fig = pc.viz.ds["figure"].item()
    axes = np.atleast_1d(np.array(fig.get_axes()))
    return fig, axes


def _plot_timeseries_channel(
    ds: xr.Dataset,
    sample_dims: list[str],
    color_dim: str,
    extra_dims: list[str],
    hdi_prob: float,
    backend: str | None,
    line_kwargs: dict[str, Any] | None,
    hdi_kwargs: dict[str, Any] | None,
    facet_color_dim: bool = False,
    **pc_kwargs,
) -> PlotCollection:
    """Render a time-series Dataset as one line+HDI band per ``color_dim`` value.

    Parameters
    ----------
    ds : xr.Dataset
        Data with a single variable and dims including ``date``, ``color_dim``,
        and zero or more dims in ``extra_dims``.  Sample dims must be
        ``(chain, draw)`` — use :func:`_ensure_chain_draw_dims` on the source
        DataArray before building the Dataset if the raw data has a ``sample``
        dimension.
    sample_dims : list of str
        Dimensions to reduce for the mean line (e.g. ``["chain", "draw"]``).
    color_dim : str
        Dimension mapped to the colour aesthetic (e.g. ``"channel"`` or
        ``"component"``).
    extra_dims : list of str
        Additional dimensions used to create facet panels (e.g. ``["geo"]``).
    hdi_prob : float
        HDI probability mass.
    backend : str or None
        Rendering backend.
    line_kwargs, hdi_kwargs : dict or None
        Extra kwargs forwarded to line and HDI visuals respectively.
    facet_color_dim : bool, default False
        If True, also facet by ``color_dim`` so each of its values is drawn in
        its own panel instead of overlaid in a single panel coloured by
        ``color_dim``. Faceting combines with ``extra_dims``, so the panel
        count is ``len(color_dim) * prod(len(extra_dims))``. When True the
        per-colour legend is omitted because each panel's title already
        identifies its ``color_dim`` value.
    **pc_kwargs
        Forwarded to ``PlotCollection.wrap()``.

    Returns
    -------
    PlotCollection
    """
    pc_kwargs.setdefault("col_wrap", 1)
    cols = [color_dim, *extra_dims] if facet_color_dim else list(extra_dims)
    pc = PlotCollection.wrap(
        ds,
        cols=cols,
        backend=backend,
        aes={"color": [color_dim]},
        **pc_kwargs,
    )

    hdi_da = ds.azstats.hdi(prob=hdi_prob)

    pc.map(
        azp.visuals.fill_between_y,
        x=ds.date,
        y_bottom=hdi_da.sel(ci_bound="lower"),
        y_top=hdi_da.sel(ci_bound="upper"),
        **{"alpha": 0.2, **(hdi_kwargs or {})},
    )
    pc.map(
        azp.visuals.line_xy,
        x=ds.date,
        y=ds.mean(dim=sample_dims),
        **(line_kwargs or {}),
    )

    pc.map(azp.visuals.labelled_x, text="Date", ignore_aes={"color"})
    pc.map(azp.visuals.labelled_y, text="Contribution", ignore_aes={"color"})
    pc.map(
        azp.visuals.labelled_title,
        subset_info=True,
        labeller=mix_labellers((NoVarLabeller, DimCoordLabeller))(),
        ignore_aes={"color"},
    )
    if not facet_color_dim:
        pc.add_legend(color_dim)

    return pc
