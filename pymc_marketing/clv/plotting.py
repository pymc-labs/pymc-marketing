from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

__all__ = [
    "plot_customer_exposure",
    "plot_frequency_recency_matrix",
    "plot_probability_alive_matrix",
]


def plot_customer_exposure(
    df: pd.DataFrame,
    linewidth: Optional[float] = None,
    size: Optional[float] = None,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    padding: float = 0.25,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot the recency and T of DataFrame of customers.

    Plots customers as horizontal lines with markers representing their recency and T starting.
    Order is the same as the DataFrame and plotted from the bottom up.

    The lines are colored by recency and T.

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame with columns "recency" and "T" representing the recency and age of customers.
    linewidth: float, optional
        The width of the horizontal lines in the plot. Default is 0.1.
    size: float, optional
        The size of the markers in the plot. Default is 10.
    labels: Sequence[str], optional
        A sequence of labels for the legend. Default is ["Recency", "T"].
    colors: Sequence[str], optional
        A sequence of colors for the legend. Default is ["C0", "C1"].
    padding: float, optional
        The padding around the plot. Default is 0.25.
    ax: plt.Axes, optional
        A matplotlib axes object to plot on. If None, a new figure and axes is created.

    Returns
    -------
    ax: plt.Axes

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
    if ax is None:
        ax = plt.gca()

    n = len(df)
    customer_idx = np.arange(1, n + 1)

    recency = df["recency"].to_numpy()
    T = df["T"].to_numpy()

    if colors is None:
        recency_color, T_color = ["C0", "C1"]
    else:
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

    legend_elements = [
        Line2D([0], [0], color=recency_color, label=labels[0]),
        Line2D([0], [0], color=T_color, label=labels[1]),
    ]

    ax.legend(handles=legend_elements, loc="best")

    return ax


def plot_frequency_recency_matrix(
    model,
    t=1,
    max_frequency=None,
    max_recency=None,
    title=None,
    xlabel="Historical Frequency",
    ylabel="Recency",
    **kwargs,
) -> plt.Axes:
    """
    Plot recency frequency matrix as heatmap.
    Plot a figure of expected transactions in T next units of time by a customer's frequency and recency.
    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model.
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
    kwargs
        Passed into the matplotlib.imshow command.
    Returns
    -------
    axes: matplotlib.AxesSubplot
    """

    if max_frequency is None:
        max_frequency = int(model.frequency.max())

    if max_recency is None:
        max_recency = int(model.recency.max())

    frequency = np.arange(max_frequency + 1)
    recency = np.arange(max_recency + 1)
    mesh_frequency, mesh_recency = np.meshgrid(frequency, recency)
    Z = (
        model.expected_num_purchases(
            customer_id=np.arange(mesh_recency.size),  # placeholder
            t=t,
            frequency=mesh_frequency.ravel(),
            recency=mesh_recency.ravel(),
            T=max_recency,
        )
        .mean(("draw", "chain"))
        .values.reshape(mesh_recency.shape)
    )

    ax = plt.subplot(111)
    pcm = ax.imshow(Z, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is None:
        title = (
            "Expected Number of Future Purchases for {} Unit{} of Time,".format(
                t, "s"[t == 1 :]
            )
            + "\nby Frequency and Recency of a Customer"
        )
    plt.title(title)

    force_aspect(ax)

    # plot colorbar beside matrix
    plt.colorbar(pcm, ax=ax)

    return ax


def plot_probability_alive_matrix(
    model,
    max_frequency=None,
    max_recency=None,
    title="Probability Customer is Alive,\nby Frequency and Recency of a Customer",
    xlabel="Customer's Historical Frequency",
    ylabel="Customer's Recency",
    **kwargs,
) -> plt.Axes:
    """
    Plot probability alive matrix as heatmap.
    Plot a figure of the probability a customer is alive based on their
    frequency and recency.
    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model.
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
    kwargs
        Passed into the matplotlib.imshow command.
    Returns
    -------
    axes: matplotlib.AxesSubplot
    """

    if max_frequency is None:
        max_frequency = int(model.frequency.max())

    if max_recency is None:
        max_recency = int(model.recency.max())

    frequency = np.arange(max_frequency + 1)
    recency = np.arange(max_recency + 1)
    mesh_frequency, mesh_recency = np.meshgrid(frequency, recency)
    Z = (
        model.expected_probability_alive(
            customer_id=np.arange(mesh_recency.size),  # placeholder
            frequency=mesh_frequency.ravel(),
            recency=mesh_recency.ravel(),
            T=max_recency,
        )
        .mean(("draw", "chain"))
        .values.reshape(mesh_recency.shape)
    )

    interpolation = kwargs.pop("interpolation", "none")

    ax = plt.subplot(111)
    pcm = ax.imshow(Z, interpolation=interpolation, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    force_aspect(ax)

    # plot colorbar beside matrix
    plt.colorbar(pcm, ax=ax)

    return ax


def force_aspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


if __name__ == "__main__":
    file = "./datasets/clv_quickstart.csv"

    df = pd.read_csv(file)

    (
        df.query("T - recency < 5")
        .sample(n=100)
        .sort_values(["recency", "T"])
        .pipe(plot_customer_exposure)
    )
    plt.show()
