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

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr

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
    conf: xr.DataArray,
    non_grid_names: set[str],
    axes: npt.NDArray[plt.Axes] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
    plot_coords = get_plot_coords(conf.coords, non_grid_names=non_grid_names)
    total_size = get_total_coord_size(plot_coords)

    if axes is None:
        subplot_kwargs = subplot_kwargs or {}
        subplot_kwargs = {**{"sharey": True, "sharex": True}, **subplot_kwargs}
        set_subplot_kwargs_defaults(subplot_kwargs, total_size)
        fig, axes = plt.subplots(**subplot_kwargs)
    else:
        fig = plt.gcf()

    plot_kwargs = plot_kwargs or {}
    plot_kwargs = {**{"alpha": 0.3}, **plot_kwargs}

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
    plot_coords = get_plot_coords(curve.coords, non_grid_names=non_grid_names)
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
