#   Copyright 2022 - 2025 The PyMC Labs Developers
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
"""Plot distributions stored in xarray.DataArray across coordinates.

Used to plot the prior and posterior of the various MMM components.

See the :func:`plot_curve` function for more information.

"""

import warnings
from collections.abc import Callable, Generator, Iterable, MutableMapping, Sequence
from itertools import product, repeat
from typing import Any, Concatenate, ParamSpec, cast

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

Values = Sequence[Any] | npt.NDArray[Any]
Coords = dict[str, Values]


def get_plot_coords(coords: Coords, non_grid_names: set[str]) -> Coords:
    """Get the plot coordinates.

    Parameters
    ----------
    coords : Coords
        The coordinates to get the plot coordinates from.
    non_grid_names : set[str]
        The names to exclude from the grid.

    Returns
    -------
    Coords
        The plot coordinates.

    """
    plot_coord_names = list(key for key in coords.keys() if key not in non_grid_names)
    return {name: np.array(coords[name]) for name in plot_coord_names}


def drop_scalar_coords(curve: xr.DataArray) -> xr.DataArray:
    """Remove scalar coordinates from an xarray DataArray.

    This function identifies and removes scalar coordinates from the given
    DataArray. Scalar coordinates are those with a single value that are
    not part of the DataArray's indexes. The function returns a new DataArray
    with the scalar coordinates removed.

    Parameters
    ----------
    curve : xr.DataArray
        The input DataArray from which scalar coordinates will be removed.

    Returns
    -------
    xr.DataArray
        A new DataArray with the identified scalar coordinates removed.

    """
    scalar_coords_to_drop = []
    for coord, values in curve.coords.items():
        if values.size == 1 and coord not in curve.indexes:
            scalar_coords_to_drop.append(coord)

    return curve.reset_coords(scalar_coords_to_drop, drop=True)


def get_total_coord_size(coords: Coords) -> int:
    """Get the total size of the coordinates.

    Parameters
    ----------
    coords : Coords
        The coordinates to get the total size of.

    Returns
    -------
    int
        The total size of the coordinates.

    """
    total_size: int = (
        1 if coords == {} else np.prod([len(values) for values in coords.values()])  # type: ignore
    )
    if total_size >= 12:
        warnings.warn("Large number of coordinates!", stacklevel=2)

    return total_size


def create_legend_handles(
    colors: Iterable[str],
    alpha: float = 0.5,
    line: bool = True,
    patch: bool = True,
) -> list[Line2D | Patch | tuple[Line2D, Patch]]:
    """Create the legend handles for the given colors.

    Parameters
    ----------
    colors : Iterable[str]
        The colors to create the legend handles.
    alpha : float, optional
        The alpha value for the patches, by default 0.5.
    line : bool, optional
        Whether to include the line, by default True.
    patch : bool, optional
        Whether to include the patch, by default True.

    Returns
    -------
    list[Line2D | Patch | tuple[Line2D, Patch]]
        The legend handles.

    """
    if not line and not patch:
        raise ValueError("At least one of line or patch must be True")

    def create_handle(
        color: str, alpha: float
    ) -> Line2D | Patch | tuple[Line2D, Patch]:
        if line and patch:
            return Line2D([0], [0], color=color), Patch(color=color, alpha=alpha)

        if line:
            return Line2D([0], [0], color=color)

        return Patch(color=color, alpha=alpha)

    return [create_handle(color, alpha) for color in colors]


def set_subplot_kwargs_defaults(
    subplot_kwargs: MutableMapping[str, Any],
    total_size: int,
) -> None:
    """Set the defaults for the subplot kwargs.

    Parameters
    ----------
    subplot_kwargs : MutableMapping[str, Any]
        The subplot kwargs to set the defaults for.
    total_size : int
        The total size of the coordinates.

    Raises
    ------
    ValueError
        If both `ncols` and `nrows` are specified.

    """
    if "ncols" in subplot_kwargs and "nrows" in subplot_kwargs:
        raise ValueError("Only specify one")

    if "ncols" not in subplot_kwargs and "nrows" not in subplot_kwargs:
        subplot_kwargs["ncols"] = total_size

    if "ncols" in subplot_kwargs:
        subplot_kwargs["nrows"] = total_size // subplot_kwargs["ncols"]
    elif "nrows" in subplot_kwargs:
        subplot_kwargs["ncols"] = total_size // subplot_kwargs["nrows"]


Selection = dict[str, Any]


def selections(
    coords: Coords,
) -> Generator[Selection, None, None]:
    """Create generator of selections.

    Parameters
    ----------
    coords : Coords
        The coordinates to create the selections from.

    Yields
    ------
    dict[str, Any]
        The selections.

    """
    coord_names = coords.keys()
    for values in product(*coords.values()):
        yield {name: value for name, value in zip(coord_names, values, strict=True)}


P = ParamSpec("P")
GetPlotData = Callable[[xr.DataArray], xr.DataArray]
MakeSelection = Callable[[xr.DataArray, Selection], pd.DataFrame]
PlotSelection = Callable[Concatenate[pd.DataFrame, Axes, str, P], Axes]


def _get_sample_plot_data(data):
    return data


def _create_make_sample_selection(
    rng,
    n: int,
    n_chains: int,
    n_draws: int,
) -> MakeSelection:
    rng = rng or np.random.default_rng()
    idx = random_samples(
        rng,
        n=n,
        n_chains=n_chains,
        n_draws=n_draws,
    )

    def make_sample_selection(data, sel):
        return data.sel(sel).to_series().unstack().loc[idx, :].T

    return make_sample_selection


def _plot_sample_selection(df, ax: Axes, color: str, **plot_kwargs) -> Axes:
    return df.plot(ax=ax, color=color, **plot_kwargs)


def _create_get_hdi_plot_data(hdi_kwargs) -> GetPlotData:
    def get_plot_data(data: xr.DataArray) -> xr.DataArray:
        hdi: xr.Dataset = az.hdi(data, **hdi_kwargs)
        return hdi[data.name]

    return get_plot_data


def _make_hdi_selection(data: xr.DataArray, sel: dict[str, Any]) -> pd.DataFrame:
    return data.sel(sel).to_series().unstack()


def _plot_hdi_selection(
    df: pd.DataFrame,
    ax: Axes,
    color: str,
    **plot_kwargs,
) -> Axes:
    ax.fill_between(
        x=df.index,
        y1=df["lower"],
        y2=df["higher"],
        color=color,
        **plot_kwargs,
    )
    return ax


SelToString = Callable[[Selection], str]


def random_samples(
    rng: np.random.Generator,
    n: int,
    n_chains: int,
    n_draws: int,
) -> list[tuple[int, int]]:
    """Generate random samples from the chains and draws.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
    n : int
        Number of samples to generate
    n_chains : int
        Number of chains
    n_draws : int
        Number of draws

    Returns
    -------
    list[tuple[int, int]]
        The random samples

    """
    combinations = list(product(range(n_chains), range(n_draws)))

    return [
        tuple(pair) for pair in list(rng.choice(combinations, size=n, replace=False))
    ]


def generate_colors(n: int, start: int = 0) -> list[str]:
    """Generate list of colors.

    Parameters
    ----------
    n : int
        Number of colors to generate
    start : int, optional
        Starting index, by default 0

    Returns
    -------
    list[str]
        List of colors

    Examples
    --------
    Generate 5 colors starting from index 1

    .. code-block:: python

        colors = generate_colors(5, start=1)
        print(colors)
        # ['C1', 'C2', 'C3', 'C4', 'C5']

    """
    return [f"C{i}" for i in range(start, start + n)]


def _plot_across_coord(
    curve: xr.DataArray,
    non_grid_names: set[str],
    get_plot_data: GetPlotData,
    make_selection: MakeSelection,
    plot_selection: PlotSelection,
    subplot_kwargs: dict | None = None,
    axes: npt.NDArray[Axes] | None = None,
    same_axes: bool = False,
    colors: Iterable[str] | None = None,
    legend: bool = False,
    plot_kwargs: dict[str, Any] | None = None,
    patch: bool = True,
    line: bool = True,
    sel_to_string: SelToString | None = None,
) -> tuple[plt.Figure, npt.NDArray[Axes]]:
    """Plot data array across coords.

    Commonality used for the `plot_samples` and `plot_hdi` functions.
    Differences depending on the `get_plot_data`, `make_selection` and
    `plot_selection` functions passed.

    Allows for plotting each coordinate combination on a separate axis
    or on the same axis.

    """
    if sel_to_string is None:

        def sel_to_string(sel):
            return ", ".join(f"{key}={value}" for key, value in sel.items())

    curve = drop_scalar_coords(curve)

    data = get_plot_data(curve)

    plot_coords = get_plot_coords(
        data.coords,
        non_grid_names=non_grid_names.union({"chain", "draw", "hdi"}),
    )
    total_size = get_total_coord_size(plot_coords)

    if axes is None and not same_axes:
        subplot_kwargs = subplot_kwargs or {}
        subplot_kwargs = {**{"sharey": True, "sharex": True}, **subplot_kwargs}
        set_subplot_kwargs_defaults(subplot_kwargs, total_size)
        fig, axes = plt.subplots(**subplot_kwargs)
        axes_iter = np.ravel(axes)
        return_axes = axes

        create_title = sel_to_string

        create_legend_label = None
    elif axes is not None and same_axes:
        fig = plt.gcf()
        axes_iter = repeat(axes[0], total_size)  # type: ignore
        return_axes = np.array([axes]) if not isinstance(axes, np.ndarray) else axes

        def create_title(sel):
            return ""

        create_legend_label = sel_to_string

    elif axes is None and same_axes:
        fig, ax = plt.subplots(ncols=1, nrows=1)
        axes_iter = repeat(ax, total_size)  # type: ignore
        return_axes = np.array([ax])

        def create_title(sel):
            return ""

        create_legend_label = sel_to_string
    else:
        fig = plt.gcf()
        axes_iter = np.ravel(axes)  # type: ignore
        return_axes = np.array([axes]) if not isinstance(axes, np.ndarray) else axes

        create_title = sel_to_string  # type: ignore

        create_legend_label = None

    colors = cast(Iterable[str], colors or generate_colors(n=total_size, start=0))

    for color, ax, sel in zip(colors, axes_iter, selections(plot_coords), strict=False):
        ax = data.pipe(make_selection, sel=sel).pipe(
            plot_selection,
            ax=ax,
            color=color,
            **plot_kwargs,
        )
        title = create_title(sel)
        ax.set_title(title)

    if same_axes and legend and create_legend_label is not None:
        handles = create_legend_handles(colors, patch=patch, line=line)
        labels = [create_legend_label(sel) for sel in selections(plot_coords)]
        ax.legend(handles=handles, labels=labels)

    return fig, return_axes


def plot_hdi(
    curve: xr.DataArray,
    non_grid_names: str | set[str],
    hdi_prob: float | None = None,
    hdi_kwargs: dict | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    axes: npt.NDArray[Axes] | None = None,
    same_axes: bool = False,
    colors: Iterable[str] | None = None,
    legend: bool = False,
    sel_to_string: SelToString | None = None,
) -> tuple[plt.Figure, npt.NDArray[Axes]]:
    """Plot hdi of the curve across coords.

    Parameters
    ----------
    curve : xr.DataArray
        Curve to plot
    non_grid_names : str | set[str]
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
    hdi_kwargs = hdi_kwargs or {}
    hdi_kwargs = {**dict(hdi_prob=hdi_prob), **hdi_kwargs}
    get_plot_data = _create_get_hdi_plot_data(hdi_kwargs)
    make_selection = _make_hdi_selection
    plot_selection = _plot_hdi_selection

    if isinstance(non_grid_names, str):
        non_grid_names = {non_grid_names}

    plot_kwargs = plot_kwargs or {}
    plot_kwargs = {**{"alpha": 0.25}, **plot_kwargs}

    return _plot_across_coord(
        curve=curve,
        non_grid_names=non_grid_names,
        get_plot_data=get_plot_data,
        make_selection=make_selection,
        plot_selection=plot_selection,
        subplot_kwargs=subplot_kwargs,
        same_axes=same_axes,
        axes=axes,
        colors=colors,
        legend=legend,
        plot_kwargs=plot_kwargs,
        patch=True,
        line=False,
        sel_to_string=sel_to_string,
    )


def plot_samples(
    curve: xr.DataArray,
    non_grid_names: str | set[str],
    n: int = 10,
    rng: np.random.Generator | None = None,
    axes: npt.NDArray[Axes] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    same_axes: bool = False,
    colors: Iterable[str] | None = None,
    legend: bool = False,
    sel_to_string: SelToString | None = None,
) -> tuple[plt.Figure, npt.NDArray[Axes]]:
    """Plot n samples of the curve across coords.

    Parameters
    ----------
    curve : xr.DataArray
        Curve to plot
    non_grid_names : str | set[str]
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
    same_axes : bool
        All of the plots in the same axis

    Returns
    -------
    tuple[plt.Figure, npt.NDArray[plt.Axes]]
        Figure and the axes

    """
    get_plot_data = _get_sample_plot_data

    if isinstance(non_grid_names, str):
        non_grid_names = {non_grid_names}

    n_chains = curve.sizes["chain"]
    n_draws = curve.sizes["draw"]
    make_selection = _create_make_sample_selection(
        rng=rng,
        n=n,
        n_chains=n_chains,
        n_draws=n_draws,
    )
    plot_selection = _plot_sample_selection

    plot_kwargs = plot_kwargs or {}
    plot_kwargs = {
        **{"alpha": 0.3, "legend": False},
        **plot_kwargs,
    }

    return _plot_across_coord(
        curve=curve,
        non_grid_names=non_grid_names,
        get_plot_data=get_plot_data,
        make_selection=make_selection,
        plot_selection=plot_selection,
        subplot_kwargs=subplot_kwargs,
        plot_kwargs=plot_kwargs,
        same_axes=same_axes,
        axes=axes,
        colors=colors,
        legend=legend,
        patch=False,
        line=True,
        sel_to_string=sel_to_string,
    )


def plot_curve(
    curve: xr.DataArray,
    non_grid_names: str | set[str],
    n_samples: int = 10,
    hdi_probs: float | list[float] | None = None,
    random_seed: np.random.Generator | None = None,
    subplot_kwargs: dict | None = None,
    sample_kwargs: dict | None = None,
    hdi_kwargs: dict | None = None,
    axes: npt.NDArray[Axes] | None = None,
    same_axes: bool = False,
    colors: Iterable[str] | None = None,
    legend: bool | None = None,
    sel_to_string: SelToString | None = None,
) -> tuple[plt.Figure, npt.NDArray[Axes]]:
    """Plot HDI with samples of the curve across coords.

    Parameters
    ----------
    curve : xr.DataArray
        Curve to plot
    non_grid_names : str | set[str]
        The names to exclude from the grid. HDI and samples both
        have defaults of hdi and chain, draw, respectively
    n_samples : int, optional
        Number of samples
    hdi_probs : float | list[float], optional
        HDI probabilities. Defaults to None which uses arviz default for
        stats.ci_prob which is 94%
    random_seed : np.random.Generator, optional
        Random number generator. Defaults to None which uses
        np.random.default_rng()
    subplot_kwargs : dict, optional
        Additional kwargs to while creating the fig and axes
    sample_kwargs : dict, optional
        Kwargs for the :func:`plot_samples` function
    hdi_kwargs : dict, optional
        Kwargs for the :func:`plot_hdi` function
    same_axes : bool
        If all of the plots are on the same axis
    colors : Iterable[str], optional
        Colors for the plots
    legend : bool, optional
        If to include a legend. Defaults to True if same_axes
    sel_to_string : Callable[[Selection], str], optional
        Function to convert selection to a string. Defaults to
        ", ".join(f"{key}={value}" for key, value in sel.items())

    Returns
    -------
    tuple[plt.Figure, npt.NDArray[plt.Axes]]
        Figure and the axes

    Examples
    --------
    Plot prior for arbitrary Deterministic in PyMC model

    .. plot::
        :include-source: True
        :context: reset

        import numpy as np
        import pandas as pd

        import pymc as pm

        import matplotlib.pyplot as plt

        from pymc_marketing.plot import plot_curve

        seed = sum(map(ord, "Arbitrary curve"))
        rng = np.random.default_rng(seed)

        dates = pd.date_range("2024-01-01", periods=52, freq="W")

        coords = {"date": dates, "product": ["A", "B"]}
        with pm.Model(coords=coords) as model:
            data = pm.Normal(
                "data",
                mu=[-0.5, 0.5],
                sigma=1,
                dims=("date", "product"),
            )
            cumsum = pm.Deterministic(
                "cumsum",
                data.cumsum(axis=0),
                dims=("date", "product"),
            )
            idata = pm.sample_prior_predictive(random_seed=rng)

        curve = idata.prior["cumsum"]

        fig, axes = plot_curve(
            curve,
            "date",
            subplot_kwargs={"figsize": (15, 5)},
            random_seed=rng,
        )
        plt.show()

    Choose the HDI intervals and number of samples

    .. plot::
        :include-source: True
        :context: reset

        fig, axes = plot_curve(
            curve,
            "date",
            n_samples=3,
            hdi_probs=[0.5, 0.95],
            random_seed=rng,
        )
        fig.suptitle("Same data but fewer lines and more HDIs")
        plt.show()

    Plot same curve on same axes with custom colors

    .. plot::
        :include-source: True
        :context: close-figs

        colors = ["red", "blue"]
        fig, axes = plot_curve(
            curve,
            "date",
            same_axes=True,
            colors=colors,
            random_seed=rng,
        )
        axes[0].set(title="Same data but on same axes and custom colors")
        plt.show()

    """
    curve = drop_scalar_coords(curve)

    hdi_probs = hdi_probs or None
    if not isinstance(hdi_probs, list):
        hdi_probs = [hdi_probs]  # type: ignore

    hdi_kwargs = hdi_kwargs or {}
    sample_kwargs = sample_kwargs or {}

    sample_kwargs = {**dict(n=n_samples, rng=random_seed), **sample_kwargs}

    if "subplot_kwargs" not in sample_kwargs:
        sample_kwargs["subplot_kwargs"] = subplot_kwargs

    if "axes" not in sample_kwargs:
        sample_kwargs["axes"] = axes

    if same_axes:
        sample_kwargs["same_axes"] = True
        sample_kwargs["legend"] = False
        hdi_kwargs["same_axes"] = True
        hdi_kwargs["legend"] = legend if isinstance(legend, bool) else True

    if colors is not None:
        sample_kwargs["colors"] = colors
        hdi_kwargs["colors"] = colors

    if sel_to_string is not None:
        sample_kwargs["sel_to_string"] = sel_to_string
        hdi_kwargs["sel_to_string"] = sel_to_string

    fig, axes = plot_samples(
        curve,
        non_grid_names=non_grid_names,
        **sample_kwargs,
    )
    for hdi_prob in hdi_probs:
        fig, axes = plot_hdi(
            curve,
            hdi_prob=hdi_prob,
            non_grid_names=non_grid_names,
            axes=axes,
            **hdi_kwargs,
        )

    return fig, axes
