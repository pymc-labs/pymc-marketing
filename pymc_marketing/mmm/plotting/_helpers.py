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


def _dims_to_sel_kwargs(
    dims: dict[str, Any] | None,
) -> dict[str, Any]:
    """Convert scalar dim values to single-element lists for ``.sel()``.

    When filtering xarray data with ``.sel()``, scalar values drop the
    dimension.  Wrapping scalars in a list preserves the dimension as
    size-1 so ``PlotCollection`` can still facet on it.

    Parameters
    ----------
    dims : dict or None
        Mapping of dimension name → value(s).

    Returns
    -------
    dict
        Same mapping with scalar values wrapped in single-element lists.
    """
    if not dims:
        return {}
    return {
        k: v if isinstance(v, (list, tuple, np.ndarray)) else [v]
        for k, v in dims.items()
    }


def _select_dims[XarrayT: (xr.Dataset, xr.DataArray)](
    data: XarrayT,
    dims: dict[str, Any] | None,
    allow_missing: bool = False,
) -> XarrayT:
    """Validate dimension filters and apply ``.sel()`` in one step.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        The xarray object to filter.
    dims : dict or None
        Dimension name → value(s).  ``None`` or empty is a no-op.
    allow_missing : bool, default False
        If True, silently ignore dimension keys in *dims* that are not
        present in *data*.  If False (default), raise ValueError for
        unknown dimensions.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Filtered object (same type as *data*).  Dimensions are preserved
        as size-1 (scalars are wrapped in lists) so downstream faceting
        still works.

    Raises
    ------
    ValueError
        If a key in *dims* is not a dimension of *data* (when
        ``allow_missing=False``), or a value is not present in the
        corresponding coordinate.
    """
    if not dims:
        return data

    if allow_missing:
        filtered_dims = {k: v for k, v in dims.items() if k in data.dims}
        if not filtered_dims:
            return data
    else:
        filtered_dims = dims

    _validate_dims(data, filtered_dims)
    sel_kwargs = _dims_to_sel_kwargs(filtered_dims)
    return data.sel(**sel_kwargs)


def _validate_dims(
    dataset: xr.Dataset | xr.DataArray,
    dims: dict[str, Any] | None,
) -> None:
    """Validate that ``dims`` keys and values exist in ``dataset`` coordinates.

    Parameters
    ----------
    dataset : xr.Dataset or xr.DataArray
        The xarray object whose coordinates are checked.
    dims : dict or None
        Mapping of dimension name → value(s) to validate.
        Values may be scalars, lists, tuples, or numpy arrays.

    Raises
    ------
    ValueError
        If a dimension name is not in ``dataset.dims`` or a value
        is not present in the corresponding coordinate.
    """
    if not dims:
        return

    all_dims = list(dataset.dims)
    for key, val in dims.items():
        if key not in all_dims:
            raise ValueError(
                f"Dimension '{key}' not found in dataset dimensions. "
                f"Available: {all_dims}"
            )
        valid_values = dataset.coords[key].values
        values = val if isinstance(val, (list, tuple, np.ndarray)) else [val]
        for v in values:
            if v not in valid_values:
                raise ValueError(
                    f"Value '{v}' not found in dimension '{key}'. "
                    f"Available: {list(valid_values)}"
                )


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


def _apply_aggregation(
    da: xr.DataArray,
    aggregation: dict[str, str | list[str]] | None,
) -> xr.DataArray:
    """Apply a single aggregation operation to *da*.

    Parameters
    ----------
    da : xr.DataArray
        Data to aggregate.
    aggregation : dict or None
        A mapping with exactly one entry: ``{op: dim_spec}`` where *op*
        is ``"sum"`` or ``"mean"`` and *dim_spec* is a dimension name or
        list of dimension names.  ``None`` or an empty dict is a no-op.

    Returns
    -------
    xr.DataArray
        Aggregated data, or *da* unchanged when *aggregation* is falsy.

    Raises
    ------
    ValueError
        If *aggregation* contains more than one entry or an unsupported
        operation.
    """
    if not aggregation:
        return da

    if len(aggregation) > 1:
        raise ValueError(
            f"Only a single aggregation operation is supported, "
            f"got {len(aggregation)}: {list(aggregation)}."
        )

    op, dim_spec = next(iter(aggregation.items()))
    dims_list = [dim_spec] if isinstance(dim_spec, str) else list(dim_spec)

    if op == "sum":
        return da.sum(dim=dims_list)
    if op == "mean":
        return da.mean(dim=dims_list)
    raise ValueError(f"Unknown aggregation operation '{op}'. Supported: 'sum', 'mean'.")


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
    **pc_kwargs
        Forwarded to ``PlotCollection.wrap()``.

    Returns
    -------
    PlotCollection
    """
    pc_kwargs.setdefault("col_wrap", 1)
    pc = PlotCollection.wrap(
        ds,
        cols=extra_dims,
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
    pc.add_legend(color_dim)

    return pc


#: Coordinate labels for the ``metric`` facet of the budget-allocation plot.
ALLOCATION_METRIC_LABELS: tuple[str, str] = (
    "Allocated Spend",
    "Channel Contribution",
)


def _build_allocation_metric_dataset(
    samples: xr.Dataset,
    dims: dict[str, Any] | None = None,
) -> tuple[xr.Dataset, list[str]]:
    """Assemble a metric-faceted dataset comparing spend and contribution.

    Combines total allocated spend and total channel contribution (summed
    over ``date``) into a single ``value`` variable with a new ``metric``
    dimension whose coordinates are :data:`ALLOCATION_METRIC_LABELS`.  This
    lets a single :class:`~arviz_plots.PlotCollection` render both quantities
    as independent-scale facet columns.

    The channel contribution keeps its sample dimensions (``chain``/``draw``)
    so downstream HDI computation works; the deterministic allocation is
    broadcast across those sample dimensions so the two metrics concatenate
    cleanly (its HDI collapses to a point).

    Parameters
    ----------
    samples : xr.Dataset
        Must contain ``channel_contribution_original_scale``
        (dims: ``sample`` or ``(chain, draw)``, ``date``, ``channel``, ...)
        and ``total_allocation`` (dims: ``channel``, ...).
    dims : dict, optional
        Dimension filters, e.g. ``{"geo": ["CA"]}``.

    Returns
    -------
    ds : xr.Dataset
        Single-variable (``value``) dataset with dims
        ``(metric, chain, draw, channel, *extra_dims)``.
    extra_dims : list of str
        Faceting dimensions beyond ``chain``/``draw``/``channel``/``metric``
        (e.g. ``["geo"]``), used as plot rows.
    """
    contribution = _select_dims(samples["channel_contribution_original_scale"], dims)
    contribution = _ensure_chain_draw_dims(contribution)
    contribution_total = contribution.sum("date")

    total_allocation = _select_dims(
        samples["total_allocation"], dims, allow_missing=True
    )
    total_allocation = total_allocation.broadcast_like(contribution_total)

    merged = xr.concat(
        [
            total_allocation.rename("value"),
            contribution_total.rename("value"),
        ],
        dim="metric",
    ).assign_coords(metric=list(ALLOCATION_METRIC_LABELS))
    ds = merged.to_dataset(name="value")

    extra_dims = [
        d for d in ds["value"].dims if d not in {"chain", "draw", "channel", "metric"}
    ]
    return ds, extra_dims


def _plot_allocation_comparison(
    ds: xr.Dataset,
    extra_dims: list[str],
    hdi_prob: float,
    backend: str | None,
    point_kwargs: dict[str, Any] | None = None,
    hdi_kwargs: dict[str, Any] | None = None,
    **pc_kwargs,
) -> PlotCollection:
    """Render the metric-faceted spend-vs-contribution comparison.

    Produces one facet column per metric (``Allocated Spend`` and
    ``Channel Contribution``) with independent y-scales, and one facet row
    per combination of ``extra_dims``.  Each channel is drawn as a median
    point with a vertical HDI whisker (the allocation whisker is degenerate
    because spend is deterministic).

    Parameters
    ----------
    ds : xr.Dataset
        Output of :func:`_build_allocation_metric_dataset` — a ``value``
        variable with dims ``(metric, chain, draw, channel, *extra_dims)``.
    extra_dims : list of str
        Dimensions used as facet rows (e.g. ``["geo"]``).  Empty for no rows.
    hdi_prob : float
        HDI probability mass for the whisker.
    backend : str or None
        Rendering backend.
    point_kwargs : dict, optional
        Extra kwargs forwarded to ``azp.visuals.scatter_xy`` (median point).
    hdi_kwargs : dict, optional
        Extra kwargs forwarded to ``azp.visuals.line_xy`` (HDI whisker).
    **pc_kwargs
        Forwarded to ``PlotCollection.grid()``.

    Returns
    -------
    PlotCollection
    """
    channels = ds["channel"].values
    positions = np.arange(len(channels))
    x_idx = xr.DataArray(positions, dims="channel", coords={"channel": channels})
    x_whisker = xr.concat([x_idx, x_idx], dim="ci_bound").assign_coords(
        ci_bound=["lower", "upper"]
    )

    median = ds["value"].median(dim=["chain", "draw"])
    hdi = ds.azstats.hdi(prob=hdi_prob)["value"]
    # Scalar-per-facet data so tick labels are set once per subplot.
    tick_data = ds["value"].mean(dim=["chain", "draw", "channel"]).to_dataset()

    pc = PlotCollection.grid(
        ds,
        cols=["metric"],
        rows=extra_dims or None,
        backend=backend,
        aes={"color": ["channel"]},
        **pc_kwargs,
    )

    pc.map(
        azp.visuals.line_xy,
        "whisker",
        x=x_whisker,
        y=hdi,
        **(hdi_kwargs or {}),
    )
    pc.map(
        azp.visuals.scatter_xy,
        "point",
        x=x_idx,
        y=median,
        **(point_kwargs or {}),
    )
    pc.map(
        azp.visuals.set_xticks,
        "ticks",
        data=tick_data,
        values=positions.tolist(),
        labels=[str(channel) for channel in channels],
        ignore_aes={"color"},
        store_artist=False,
    )
    pc.map(azp.visuals.labelled_y, text="Value", ignore_aes={"color"})
    pc.map(
        azp.visuals.labelled_title,
        subset_info=True,
        labeller=mix_labellers((NoVarLabeller, DimCoordLabeller))(),
        ignore_aes={"color"},
    )
    pc.add_legend("channel")

    return pc
