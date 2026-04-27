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

import numpy as np
import xarray as xr
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
