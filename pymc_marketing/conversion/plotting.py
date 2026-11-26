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
"""Plots for the conversion models."""

from collections.abc import Hashable
from itertools import product

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from pymc_marketing.decision import _sample_dims
from pymc_marketing.names.conversion import Dim

__all__ = ["plot_funnel"]


def plot_funnel(
    samples: xr.DataArray,
    *,
    hue: str | None = None,
    hdi_prob: float = 0.94,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, npt.NDArray[Axes]]:
    """Plot a conversion by step as horizontal bars with highest density intervals.

    The steps run from top to bottom. A bar is the posterior mean and the line
    on it the highest density interval.

    Parameters
    ----------
    samples : xarray.DataArray
        Posterior draws with the dimensions ``("chain", "draw")`` or
        ``"sample"``, the dimension ``step`` and any number of group
        dimensions, for example the output of
        :meth:`~pymc_marketing.conversion.funnel.FunnelModel.cumulative_conversion`.
        Select the labels to show with ``.sel`` before plotting.
    hue : str, optional
        Group dimension whose labels get one colour each. Defaults to the first
        group dimension. Every combination of labels of the other group
        dimensions gets a panel of its own.
    hdi_prob : float, default 0.94
        Probability mass of the highest density interval.
    figsize : tuple of float, optional
        Size of the figure. By default it grows with the number of panels and
        bars.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    numpy.ndarray of matplotlib.axes.Axes
        One Axes per panel, in a one-dimensional array.

    Raises
    ------
    ValueError
        If ``samples`` has no sample dimensions or no ``step`` dimension, or
        if ``hue`` is not one of its group dimensions.

    Examples
    --------
    Cumulative conversion of two variants, for two of the segments:

    .. code-block:: python

        from pymc_marketing.conversion.plotting import plot_funnel

        draws = model.cumulative_conversion().sel(segment=["new", "returning"])
        fig, axes = plot_funnel(draws, hue="variant")
    """
    sample_dims = _sample_dims(samples)
    if Dim.STEP not in samples.dims:
        raise ValueError(f"samples need a {Dim.STEP!r} dimension; got {samples.dims}.")
    group_dims = [dim for dim in samples.dims if dim not in (*sample_dims, Dim.STEP)]
    if hue is not None and hue not in group_dims:
        raise ValueError(
            f"{hue!r} is not a group dimension of samples; got {group_dims}."
        )
    hue_dim: Hashable | None = hue if hue is not None else next(iter(group_dims), None)
    panel_dims = [dim for dim in group_dims if dim != hue_dim]

    mean = samples.mean(sample_dims)
    hdi = az.hdi(samples, prob=hdi_prob, dim=list(sample_dims))

    steps = samples[Dim.STEP].values
    hue_labels = [None] if hue_dim is None else list(samples[hue_dim].values)
    panels = [
        dict(zip(panel_dims, labels, strict=True))
        for labels in product(*(samples[dim].values for dim in panel_dims))
    ]
    n_cols = min(len(panels), 3)
    n_rows = -(-len(panels) // n_cols)
    if figsize is None:
        figsize = (
            8.0 if n_cols == 1 else 6.0 * n_cols,
            n_rows * max(3.5, 0.45 * len(steps) * len(hue_labels) + 1.5),
        )
    fig, grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    axes = grid.ravel()
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    axes = axes[: len(panels)]

    positions = np.arange(len(steps))
    height = 0.8 / len(hue_labels)
    for ax, panel in zip(axes, panels, strict=True):
        for i, label in enumerate(hue_labels):
            cell = panel if hue_dim is None else {**panel, hue_dim: label}
            centres = positions - 0.4 + height * (i + 0.5)
            ax.barh(
                centres,
                mean.sel(cell).values,
                height=height,
                color=f"C{i}",
                alpha=0.8,
                label=None if hue_dim is None else str(label),
            )
            ax.hlines(
                centres,
                hdi.sel(cell).sel(ci_bound="lower").values,
                hdi.sel(cell).sel(ci_bound="upper").values,
                color="black",
                linewidth=1.5,
            )
        ax.set_yticks(positions, labels=[str(step) for step in steps])
        ax.set_title(", ".join(f"{dim} = {label}" for dim, label in panel.items()))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_xlim(left=0)
    axes[0].invert_yaxis()
    if hue_dim is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title=str(hue_dim), loc="outside right upper")
    return fig, axes
