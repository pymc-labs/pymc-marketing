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
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from pymc_marketing.clv import BetaGeoModel, ParetoNBDModel

__all__ = [
    "plot_customer_exposure",
    "plot_frequency_recency_matrix",
    "plot_probability_alive_matrix",
]


def plot_customer_exposure(
    df: pd.DataFrame,
    linewidth: float | None = None,
    size: float | None = None,
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    padding: float = 0.25,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the recency and T of DataFrame of customers.

    Plots customers as horizontal lines with markers representing their recency and T starting.
    Order is the same as the DataFrame and plotted from the bottom up.

    The lines are colored by recency and T.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with columns "recency" and "T" representing the recency and age of customers.
    linewidth : float, optional
        The width of the horizontal lines in the plot.
    size : float, optional
        The size of the markers in the plot.
    labels : Sequence[str], optional
        A sequence of labels for the legend. Default is ["Recency", "T"].
    colors : Sequence[str], optional
        A sequence of colors for the legend. Default is ["C0", "C1"].
    padding : float, optional
        The padding around the plot. Default is 0.25.
    ax : plt.Axes, optional
        A matplotlib axes instance to plot on. If None, a new figure and axes is created.

    Returns
    -------
    plt.Axes
        The matplotlib axes instance.

    Examples
    --------
    Plot customer exposure

    .. code-block:: python

        df = pd.DataFrame({
            "recency": [0, 1, 2, 3, 4],
            "T": [5, 5, 5, 5, 5]
        })

        plot_customer_exposure(df)

    Plot customer exposure ordered by recency and T

    .. code-block:: python

        (
            df
            .sort_values(["recency", "T"])
            .pipe(plot_customer_exposure)
        )

    Plot exposure for only those with time until last purchase is less than 3

    .. code-block:: python

        (
            df
            .query("T - recency < 3")
            .pipe(plot_customer_exposure)
        )

    """
    if padding < 0:
        raise ValueError("padding must be non-negative")

    if size is not None and size < 0:
        raise ValueError("size must be non-negative")

    if linewidth is not None and linewidth < 0:
        raise ValueError("linewidth must be non-negative")

    if ax is None:
        ax = plt.gca()

    n = len(df)
    customer_idx = np.arange(1, n + 1)

    recency = df["recency"].to_numpy()
    T = df["T"].to_numpy()

    if colors is None:
        colors = ["C0", "C1"]

    if len(colors) != 2:
        raise ValueError("colors must be a sequence of length 2")

    recency_color, T_color = colors

    ax.hlines(
        y=customer_idx, xmin=0, xmax=recency, linewidth=linewidth, color=recency_color
    )
    ax.hlines(y=customer_idx, xmin=recency, xmax=T, linewidth=linewidth, color=T_color)

    ax.scatter(x=recency, y=customer_idx, linewidth=linewidth, s=size, c=recency_color)
    ax.scatter(x=T, y=customer_idx, linewidth=linewidth, s=size, c=T_color)

    ax.set(
        xlabel="Time since first purchase",
        ylabel="Customer",
        xlim=(0 - padding, T.max() + padding),
        ylim=(1 - padding, n + padding),
        title="Customer Exposure",
    )

    if labels is None:
        labels = ["Recency", "T"]

    if len(labels) != 2:
        raise ValueError("labels must be a sequence of length 2")

    recency_label, T_label = labels

    legend_elements = [
        Line2D([0], [0], color=recency_color, label=recency_label),
        Line2D([0], [0], color=T_color, label=T_label),
    ]

    ax.legend(handles=legend_elements, loc="best")

    return ax


def _create_frequency_recency_meshes(
    max_frequency: int,
    max_recency: int,
) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.arange(max_frequency + 1)
    recency = np.arange(max_recency + 1)
    mesh_frequency, mesh_recency = np.meshgrid(frequency, recency)

    return mesh_frequency, mesh_recency


def plot_frequency_recency_matrix(
    model: BetaGeoModel | ParetoNBDModel,
    t=1,
    max_frequency: int | None = None,
    max_recency: int | None = None,
    title: str | None = None,
    xlabel: str = "Customer's Historical Frequency",
    ylabel: str = "Customer's Recency",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot recency frequency matrix as heatmap.
    Plot a figure of expected transactions in T next units of time by a customer's frequency and recency.

    Parameters
    ----------
    model: CLV model
        A fitted CLV model.
    t: float, optional
        Next units of time to make predictions for
    max_frequency: int, optional
        The maximum frequency to plot. Default is max observed frequency.
    max_recency: int, optional
        The maximum recency to plot. This also determines the age of the customer.
        Default to max observed age.
    title: str, optional
        Figure title
    xlabel: str, optional
        Figure xlabel
    ylabel: str, optional
        Figure ylabel
    ax: plt.Axes, optional
        A matplotlib axes instance. Creates new axes instance by default.
    kwargs
        Passed into the matplotlib.imshow command.

    Returns
    -------
    axes: matplotlib.AxesSubplot
    """
    if max_frequency is None:
        max_frequency = int(model.data["frequency"].max())

    if max_recency is None:
        max_recency = int(model.data["recency"].max())

    mesh_frequency, mesh_recency = _create_frequency_recency_meshes(
        max_frequency=max_frequency,
        max_recency=max_recency,
    )

    # FIXME: This is a hotfix for ParetoNBDModel, as it has a different API from BetaGeoModel
    #  We should harmonize them!
    if isinstance(model, ParetoNBDModel):
        transaction_data = pd.DataFrame(
            {
                "customer_id": np.arange(mesh_recency.size),  # placeholder
                "frequency": mesh_frequency.ravel(),
                "recency": mesh_recency.ravel(),
                "T": max_recency,
            }
        )

        Z = (
            model.expected_purchases(
                data=transaction_data,
                future_t=t,
            )
            .mean(("draw", "chain"))
            .values.reshape(mesh_recency.shape)
        )
    else:
        Z = (
            model.expected_num_purchases(
                customer_id=np.arange(mesh_recency.size),  # placeholder
                frequency=mesh_frequency.ravel(),
                recency=mesh_recency.ravel(),
                T=max_recency,
                t=t,
            )
            .mean(("draw", "chain"))
            .values.reshape(mesh_recency.shape)
        )

    if ax is None:
        ax = plt.subplot(111)

    pcm = ax.imshow(Z, **kwargs)
    if title is None:
        title = (
            "Expected Number of Future Purchases for {} Unit{} of Time,".format(
                t, "s"[t == 1 :]
            )
            + "\nby Frequency and Recency of a Customer"
        )

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )

    force_aspect(ax)

    # plot colorbar beside matrix
    plt.colorbar(pcm, ax=ax)

    return ax


def plot_probability_alive_matrix(
    model: BetaGeoModel | ParetoNBDModel,
    max_frequency: int | None = None,
    max_recency: int | None = None,
    title: str = "Probability Customer is Alive,\nby Frequency and Recency of a Customer",
    xlabel: str = "Customer's Historical Frequency",
    ylabel: str = "Customer's Recency",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot probability alive matrix as heatmap.
    Plot a figure of the probability a customer is alive based on their
    frequency and recency.

    Parameters
    ----------
    model: CLV model
        A fitted CLV model.
    max_frequency: int, optional
        The maximum frequency to plot. Default is max observed frequency.
    max_recency: int, optional
        The maximum recency to plot. This also determines the age of the customer.
        Default to max observed age.
    title: str, optional
        Figure title
    xlabel: str, optional
        Figure xlabel
    ylabel: str, optional
        Figure ylabel
    ax: plt.Axes, optional
        A matplotlib axes instance. Creates new axes instance by default.
    kwargs
        Passed into the matplotlib.imshow command.

    Returns
    -------
    axes: matplotlib.AxesSubplot
    """

    if max_frequency is None:
        max_frequency = int(model.data["frequency"].max())

    if max_recency is None:
        max_recency = int(model.data["recency"].max())

    mesh_frequency, mesh_recency = _create_frequency_recency_meshes(
        max_frequency=max_frequency,
        max_recency=max_recency,
    )
    # FIXME: This is a hotfix for ParetoNBDModel, as it has a different API from BetaGeoModel
    #  We should harmonize them!
    if isinstance(model, ParetoNBDModel):
        transaction_data = pd.DataFrame(
            {
                "customer_id": np.arange(mesh_recency.size),  # placeholder
                "frequency": mesh_frequency.ravel(),
                "recency": mesh_recency.ravel(),
                "T": max_recency,
            }
        )

        Z = (
            model.expected_probability_alive(
                data=transaction_data,
                future_t=0,  # TODO: This can be a function parameter in the case of ParetoNBDModel
            )
            .mean(("draw", "chain"))
            .values.reshape(mesh_recency.shape)
        )
    else:
        Z = (
            model.expected_probability_alive(
                customer_id=np.arange(mesh_recency.size),  # placeholder
                frequency=mesh_frequency.ravel(),
                recency=mesh_recency.ravel(),
                T=max_recency,  # type: ignore
            )
            .mean(("draw", "chain"))
            .values.reshape(mesh_recency.shape)
        )

    interpolation = kwargs.pop("interpolation", "none")

    if ax is None:
        ax = plt.subplot(111)

    pcm = ax.imshow(Z, interpolation=interpolation, **kwargs)

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )
    force_aspect(ax)

    # plot colorbar beside matrix
    plt.colorbar(pcm, ax=ax)

    return ax


def force_aspect(ax: plt.Axes, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)
