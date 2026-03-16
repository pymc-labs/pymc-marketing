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
from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr
from arviz_plots import PlotCollection

MATPLOTLIB_CYCLE_SIZE = 10


def _validate_dims(
    dataset: xr.Dataset,
    dims: dict[str, Any] | None,
) -> None:
    """Validate that ``dims`` keys and values exist in ``dataset`` coordinates.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset whose coordinates are checked.
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


def channel_color_map(channels: Sequence[str]) -> dict[str, str]:
    """Return a deterministic channel → matplotlib color mapping.

    Uses the default matplotlib color cycle (``C0``–``C9``), wrapping
    for more than 10 channels.

    Parameters
    ----------
    channels : sequence of str
        Ordered channel names.

    Returns
    -------
    dict[str, str]
        ``{channel_name: "C<index>"}``, preserving input order.
    """
    return {ch: f"C{i % MATPLOTLIB_CYCLE_SIZE}" for i, ch in enumerate(channels)}


def _process_plot_params(
    figsize: tuple[float, float] | None,
    plot_collection: PlotCollection | None,
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
    plot_collection : PlotCollection or None
        When provided, ``figsize`` and layout kwargs are ignored.
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
    if plot_collection is not None and figsize is not None:
        warnings.warn(
            "figsize is ignored when plot_collection is provided.",
            UserWarning,
            stacklevel=2,
        )

    if not return_as_pc and backend is not None and backend != "matplotlib":
        raise ValueError(
            f"backend='{backend}' requires return_as_pc=True. "
            "Non-matplotlib backends cannot return (Figure, NDArray[Axes])."
        )

    if figsize is not None and plot_collection is None:
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
