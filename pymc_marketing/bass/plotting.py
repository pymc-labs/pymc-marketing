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
"""Plotting functions for the Bass diffusion model.

Each function takes a fitted :class:`~pymc_marketing.bass.model.BassModel`
and returns a ``(Figure, ndarray of Axes)`` tuple, following the convention
of :func:`pymc_marketing.plot.plot_curve`. Multi-product models plot one
subplot per product; pass ``product`` to select a single one.

The functions are also exposed as ``plot_*`` methods on
:class:`~pymc_marketing.bass.model.BassModel`.
"""

from typing import TYPE_CHECKING, Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from pymc_marketing.plot import plot_curve

if TYPE_CHECKING:
    from pymc_marketing.bass.model import BassModel

__all__ = [
    "plot_adoption_curve",
    "plot_cumulative",
    "plot_decomposition",
    "plot_peak",
]


def _select_product(da: xr.DataArray, product: str | None) -> xr.DataArray:
    if product is None:
        return da
    if "product" not in da.dims:
        raise ValueError(
            "The model has no 'product' dimension. "
            "Remove the product argument for single-product models."
        )
    return da.sel(product=product)


def _get_observed(model: "BassModel", product: str | None) -> xr.DataArray | None:
    idata = model.idata
    if idata is None or "fit_data" not in idata or "observed" not in idata.fit_data:
        return None
    return _select_product(idata.fit_data["observed"], product)


def _overlay_observed(
    observed: xr.DataArray,
    axes: npt.NDArray[Axes],
    **plot_kwargs: Any,
) -> None:
    """Plot observed data on each axes, one product per axes."""
    plot_kwargs = {"color": "black", **plot_kwargs}
    t = observed.coords["T"].values
    if "product" in observed.dims:
        for ax, product in zip(
            axes.flat, observed.coords["product"].values, strict=False
        ):
            ax.plot(t, observed.sel(product=product).values, **plot_kwargs)
    else:
        for ax in axes.flat:
            ax.plot(t, observed.values, **plot_kwargs)


def plot_adoption_curve(
    model: "BassModel",
    product: str | None = None,
    n_samples: int = 10,
    hdi_probs: float | list[float] | None = None,
    random_seed: np.random.Generator | None = None,
    subplot_kwargs: dict | None = None,
    axes: npt.NDArray[Axes] | None = None,
) -> tuple[plt.Figure, npt.NDArray[Axes]]:
    """Plot the posterior adoption curve with the observed data.

    Shows posterior samples and HDI of the ``adopters`` deterministic
    (new adopters per period) over time, with the observed counts in black.

    Parameters
    ----------
    model : BassModel
        A fitted Bass model.
    product : str, optional
        Plot a single product of a multi-product model. Default plots
        one subplot per product.
    n_samples : int, optional
        Number of posterior sample curves to draw. Default is 10.
    hdi_probs : float or list of float, optional
        HDI probabilities. Defaults to the ArviZ default (0.94).
    random_seed : np.random.Generator, optional
        Random number generator for sample selection.
    subplot_kwargs : dict, optional
        Additional kwargs for the subplot creation, e.g. ``figsize``.
    axes : ndarray of Axes, optional
        Existing axes to plot on.

    Returns
    -------
    tuple[Figure, ndarray of Axes]
        Figure and the axes.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from pymc_marketing.bass import BassModel

        model = BassModel()
        model.fit(data=adoption_counts)

        fig, axes = model.plot_adoption_curve()
    """
    posterior = model.posterior
    curve = _select_product(posterior["adopters"], product)

    fig, axes = plot_curve(
        curve,
        {"T"},
        n_samples=n_samples,
        hdi_probs=hdi_probs,
        random_seed=random_seed,
        subplot_kwargs=subplot_kwargs,
        axes=axes,
    )

    observed = _get_observed(model, product)
    if observed is not None:
        _overlay_observed(observed, axes)

    for ax in axes.flat:
        ax.set_xlabel("T")
        ax.set_ylabel("New adopters")

    return fig, axes


def plot_cumulative(
    model: "BassModel",
    product: str | None = None,
    n_samples: int = 10,
    hdi_probs: float | list[float] | None = None,
    random_seed: np.random.Generator | None = None,
    subplot_kwargs: dict | None = None,
    axes: npt.NDArray[Axes] | None = None,
) -> tuple[plt.Figure, npt.NDArray[Axes]]:
    """Plot the cumulative adoption S-curve with the observed data.

    Shows posterior samples and HDI of the cumulative sum of the
    ``adopters`` deterministic over time, with the observed cumulative
    counts in black.

    Parameters
    ----------
    model : BassModel
        A fitted Bass model.
    product : str, optional
        Plot a single product of a multi-product model. Default plots
        one subplot per product.
    n_samples : int, optional
        Number of posterior sample curves to draw. Default is 10.
    hdi_probs : float or list of float, optional
        HDI probabilities. Defaults to the ArviZ default (0.94).
    random_seed : np.random.Generator, optional
        Random number generator for sample selection.
    subplot_kwargs : dict, optional
        Additional kwargs for the subplot creation, e.g. ``figsize``.
    axes : ndarray of Axes, optional
        Existing axes to plot on.

    Returns
    -------
    tuple[Figure, ndarray of Axes]
        Figure and the axes.
    """
    posterior = model.posterior
    curve = _select_product(posterior["adopters"], product).cumsum(dim="T")

    fig, axes = plot_curve(
        curve,
        {"T"},
        n_samples=n_samples,
        hdi_probs=hdi_probs,
        random_seed=random_seed,
        subplot_kwargs=subplot_kwargs,
        axes=axes,
    )

    observed = _get_observed(model, product)
    if observed is not None:
        _overlay_observed(observed.cumsum(dim="T"), axes)

    for ax in axes.flat:
        ax.set_xlabel("T")
        ax.set_ylabel("Cumulative adopters")

    return fig, axes


def plot_decomposition(
    model: "BassModel",
    product: str | None = None,
    n_samples: int = 10,
    hdi_probs: float | list[float] | None = None,
    random_seed: np.random.Generator | None = None,
    subplot_kwargs: dict | None = None,
    axes: npt.NDArray[Axes] | None = None,
) -> tuple[plt.Figure, npt.NDArray[Axes]]:
    """Plot the adoption decomposition into innovators and imitators.

    Innovators and imitators (new adopters per period) are drawn on the
    left y-axis. Cumulative adoption and the observed cumulative counts
    are drawn on a twin right y-axis, since they are on a much larger
    scale and would otherwise dwarf the per-period curves.

    Parameters
    ----------
    model : BassModel
        A fitted Bass model.
    product : str, optional
        Plot a single product of a multi-product model. Default plots
        one subplot per product.
    n_samples : int, optional
        Number of posterior sample curves to draw. Default is 10.
    hdi_probs : float or list of float, optional
        HDI probabilities. Defaults to the ArviZ default (0.94).
    random_seed : np.random.Generator, optional
        Random number generator for sample selection.
    subplot_kwargs : dict, optional
        Additional kwargs for the subplot creation, e.g. ``figsize``.
    axes : ndarray of Axes, optional
        Existing axes to plot on.

    Returns
    -------
    tuple[Figure, ndarray of Axes]
        Figure and the primary (left) axes. The twin (right) axes are
        accessible through ``fig.axes``.
    """
    posterior = model.posterior
    innovators = _select_product(posterior["innovators"], product)
    imitators = _select_product(posterior["imitators"], product)
    cumulative = _select_product(posterior["adopters"], product).cumsum(dim="T")

    n_axes = innovators.coords["product"].size if "product" in innovators.dims else 1
    fig, axes = plot_curve(
        innovators,
        {"T"},
        n_samples=n_samples,
        hdi_probs=hdi_probs,
        random_seed=random_seed,
        subplot_kwargs=subplot_kwargs,
        axes=axes,
        colors=n_axes * ["C1"],
        legend=False,
    )
    plot_curve(
        imitators,
        {"T"},
        n_samples=n_samples,
        hdi_probs=hdi_probs,
        random_seed=random_seed,
        axes=axes,
        colors=n_axes * ["C2"],
        legend=False,
    )

    twin_axes = np.array([ax.twinx() for ax in axes.flat]).reshape(axes.shape)
    plot_curve(
        cumulative,
        {"T"},
        n_samples=n_samples,
        hdi_probs=hdi_probs,
        random_seed=random_seed,
        axes=twin_axes,
        colors=n_axes * ["C0"],
        legend=False,
    )

    observed = _get_observed(model, product)
    if observed is not None:
        _overlay_observed(observed.cumsum(dim="T"), twin_axes, linestyle="--")

    for ax, twin_ax in zip(axes.flat, twin_axes.flat, strict=True):
        ax.set_xlabel("T")
        ax.set_ylabel("New adopters per period")
        twin_ax.set_ylabel("Cumulative adopters")

    handles = [
        Line2D([], [], color="C1", label="innovators"),
        Line2D([], [], color="C2", label="imitators"),
        Line2D([], [], color="C0", label="cumulative adoption"),
    ]
    if observed is not None:
        handles.append(
            Line2D([], [], color="black", linestyle="--", label="observed cumulative")
        )
    axes.flat[0].legend(handles=handles, loc="upper left")

    return fig, axes


def plot_peak(
    model: "BassModel",
    product: str | None = None,
    **plot_posterior_kwargs: Any,
) -> tuple[plt.Figure, npt.NDArray[Axes]]:
    """Plot the posterior distribution of the peak adoption time.

    Parameters
    ----------
    model : BassModel
        A fitted Bass model.
    product : str, optional
        Plot a single product of a multi-product model. Default plots
        one subplot per product.
    **plot_posterior_kwargs
        Additional kwargs forwarded to :func:`arviz.plot_posterior`.

    Returns
    -------
    tuple[Figure, ndarray of Axes]
        Figure and the axes.
    """
    posterior = model.posterior
    peak = _select_product(posterior["peak"], product)

    axes = np.atleast_1d(az.plot_posterior(peak, **plot_posterior_kwargs))
    fig = axes.flat[0].get_figure()

    return fig, axes
