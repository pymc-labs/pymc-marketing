#   Copyright 2024 The PyMC Labs Developers
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
import warnings
from collections.abc import Generator, MutableMapping, Sequence
from itertools import product
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr

from pymc_marketing.mmm.utils import drop_scalar_coords

Values = Sequence[Any] | npt.NDArray[Any]
Coords = dict[str, Values]


def get_plot_coords(coords: Coords, non_grid_names: set[str]) -> Coords:
    plot_coord_names = list(key for key in coords.keys() if key not in non_grid_names)
    return {name: np.array(coords[name]) for name in plot_coord_names}


def get_total_coord_size(coords: Coords) -> int:
    total_size: int = (
        1 if coords == {} else np.prod([len(values) for values in coords.values()])  # type: ignore
    )
    if total_size >= 12:
        warnings.warn("Large number of coordinates!", stacklevel=2)

    return total_size


def set_subplot_kwargs_defaults(
    subplot_kwargs: MutableMapping[str, Any],
    total_size: int,
) -> None:
    if "ncols" in subplot_kwargs and "nrows" in subplot_kwargs:
        raise ValueError("Only specify one")

    if "ncols" not in subplot_kwargs and "nrows" not in subplot_kwargs:
        subplot_kwargs["ncols"] = total_size

    if "ncols" in subplot_kwargs:
        subplot_kwargs["nrows"] = total_size // subplot_kwargs["ncols"]
    elif "nrows" in subplot_kwargs:
        subplot_kwargs["ncols"] = total_size // subplot_kwargs["nrows"]


def selections(
    coords: Coords,
) -> Generator[dict[str, Any], None, None]:
    """Helper to create generator of selections."""
    coord_names = coords.keys()
    for values in product(*coords.values()):
        yield {name: value for name, value in zip(coord_names, values, strict=True)}


def plot_hdi(
    curve: xr.DataArray,
    non_grid_names: set[str],
    hdi_kwargs: dict | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    axes: npt.NDArray[plt.Axes] | None = None,
) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    """Plot hdi of the curve across coords.

    Parameters
    ----------
    curve : xr.DataArray
        Curve to plot
    non_grid_names : set[str]
        The names to exclude from the grid. chain and draw are
        excluded automatically
    n : int, optional
        Number of samples to plot
    rng : np.random.Generator, optional
        Random number generator
    axes : npt.NDArray[plt.Axes], optional
        Axes to plot on
    subplot_kwargs : dict, optional
        Additional kwargs to while creating the fig and axes
    plot_kwargs : dict, optional
        Kwargs for the plot function

    Returns
    -------
    tuple[plt.Figure, npt.NDArray[plt.Axes]]
        Figure and the axes

    """
    curve = drop_scalar_coords(curve)

    hdi_kwargs = hdi_kwargs or {}
    conf = az.hdi(curve, **hdi_kwargs)[curve.name]

    plot_coords = get_plot_coords(
        conf.coords,
        non_grid_names=non_grid_names.union({"hdi"}),
    )
    total_size = get_total_coord_size(plot_coords)

    if axes is None:
        subplot_kwargs = subplot_kwargs or {}
        subplot_kwargs = {**{"sharey": True, "sharex": True}, **subplot_kwargs}
        set_subplot_kwargs_defaults(subplot_kwargs, total_size)
        fig, axes = plt.subplots(**subplot_kwargs)
    else:
        fig = plt.gcf()

    plot_kwargs = plot_kwargs or {}
    plot_kwargs = {**{"alpha": 0.25}, **plot_kwargs}

    for i, (ax, sel) in enumerate(
        zip(np.ravel(axes), selections(plot_coords), strict=False)
    ):
        color = f"C{i}"
        df_conf = conf.sel(sel).to_series().unstack()

        ax.fill_between(
            x=df_conf.index,
            y1=df_conf["lower"],
            y2=df_conf["higher"],
            color=color,
            **plot_kwargs,
        )
        title = ", ".join(f"{name}={value}" for name, value in sel.items())
        ax.set_title(title)

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    return fig, axes


def random_samples(
    rng: np.random.Generator,
    n: int,
    n_chains: int,
    n_draws: int,
) -> list[tuple[int, int]]:
    combinations = list(product(range(n_chains), range(n_draws)))

    return [
        tuple(pair) for pair in rng.choice(combinations, size=n, replace=False).tolist()
    ]


def plot_samples(
    curve: xr.DataArray,
    non_grid_names: set[str],
    n: int = 10,
    rng: np.random.Generator | None = None,
    axes: npt.NDArray[plt.Axes] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    """Plot n samples of the curve across coords.

    Parameters
    ----------
    curve : xr.DataArray
        Curve to plot
    non_grid_names : set[str]
        The names to exclude from the grid. chain and draw are
        excluded automatically
    n : int, optional
        Number of samples to plot
    rng : np.random.Generator, optional
        Random number generator
    axes : npt.NDArray[plt.Axes], optional
        Axes to plot on
    subplot_kwargs : dict, optional
        Additional kwargs to while creating the fig and axes
    plot_kwargs : dict, optional
        Kwargs for the plot function

    Returns
    -------
    tuple[plt.Figure, npt.NDArray[plt.Axes]]
        Figure and the axes

    """
    curve = drop_scalar_coords(curve)

    plot_coords = get_plot_coords(
        curve.coords,
        non_grid_names=non_grid_names.union({"chain", "draw"}),
    )
    total_size = get_total_coord_size(plot_coords)

    if axes is None:
        subplot_kwargs = subplot_kwargs or {}
        subplot_kwargs = {**{"sharey": True, "sharex": True}, **subplot_kwargs}
        set_subplot_kwargs_defaults(subplot_kwargs, total_size)
        fig, axes = plt.subplots(**subplot_kwargs)
    else:
        fig = plt.gcf()

    plot_kwargs = plot_kwargs or {}
    plot_kwargs = {
        **{"alpha": 0.3, "legend": False},
        **plot_kwargs,
    }

    rng = rng or np.random.default_rng()
    idx = random_samples(
        rng, n=n, n_chains=curve.sizes["chain"], n_draws=curve.sizes["draw"]
    )

    for i, (ax, sel) in enumerate(
        zip(np.ravel(axes), selections(plot_coords), strict=False)
    ):
        color = f"C{i}"

        df_curve = curve.sel(sel).to_series().unstack()
        df_sample = df_curve.loc[idx, :]

        df_sample.T.plot(ax=ax, color=color, **plot_kwargs)
        title = ", ".join(f"{name}={value}" for name, value in sel.items())
        ax.set_title(title)

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    return fig, axes


def plot_curve(
    curve: xr.DataArray,
    non_grid_names: set[str],
    subplot_kwargs: dict | None = None,
    sample_kwargs: dict | None = None,
    hdi_kwargs: dict | None = None,
) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    """Plot HDI with samples of the curve across coords.

    Parameters
    ----------
    curve : xr.DataArray
        Curve to plot
    non_grid_names : set[str]
        The names to exclude from the grid. HDI and samples both
        have defaults of hdi and chain, draw, respectively
    subplot_kwargs : dict, optional
        Addtional kwargs to while creating the fig and axes
    sample_kwargs : dict, optional
        Kwargs for the :func:`plot_curve` function
    hdi_kwargs : dict, optional
        Kwargs for the :func:`plot_hdi` function

    Returns
    -------
    tuple[plt.Figure, npt.NDArray[plt.Axes]]
        Figure and the axes

    """
    curve = drop_scalar_coords(curve)

    hdi_kwargs = hdi_kwargs or {}
    sample_kwargs = sample_kwargs or {}

    if "subplot_kwargs" not in sample_kwargs:
        sample_kwargs["subplot_kwargs"] = subplot_kwargs

    fig, axes = plot_samples(
        curve,
        non_grid_names=non_grid_names,
        **sample_kwargs,
    )
    fig, axes = plot_hdi(
        curve,
        non_grid_names=non_grid_names,
        axes=axes,
        **hdi_kwargs,
    )

    return fig, axes
