import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_frequency_recency_matrix",
    "plot_probability_alive_matrix",
]


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
