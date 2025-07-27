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
"""Plotting functions for the CLV module."""

import warnings
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from matplotlib.lines import Line2D

from pymc_marketing.clv import BetaGeoModel, ParetoNBDModel
from pymc_marketing.clv.utils import _expected_cumulative_transactions

__all__ = [
    "plot_customer_exposure",
    "plot_expected_purchases_over_time",
    "plot_expected_purchases_ppc",
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

        df = pd.DataFrame({"recency": [0, 1, 2, 3, 4], "T": [5, 5, 5, 5, 5]})

        plot_customer_exposure(df)

    Plot customer exposure ordered by recency and T

    .. code-block:: python

        (df.sort_values(["recency", "T"]).pipe(plot_customer_exposure))

    Plot exposure for only those with time until last purchase is less than 3

    .. code-block:: python

        (df.query("T - recency < 3").pipe(plot_customer_exposure))

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
    future_t: int = 1,
    max_frequency: int | None = None,
    max_recency: int | None = None,
    title: str | None = None,
    xlabel: str = "Customer's Historical Frequency",
    ylabel: str = "Customer's Recency",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot expected purchases in *future_t* time periods as a heatmap based on customer population *frequency* and *recency*.

    Parameters
    ----------
    model: CLV model
        A fitted CLV model.
    future_t: float, optional
        Future time periods over which to run predictions.
    max_frequency: int, optional
        The maximum *frequency* to plot. Defaults to max observed *frequency*.
    max_recency: int, optional
        The maximum *recency* to plot. This also determines the age of the customer. Defaults to max observed *recency*.
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

    """  # noqa: E501
    if max_frequency is None:
        max_frequency = int(model.data["frequency"].max())

    if max_recency is None:
        max_recency = int(model.data["recency"].max())

    mesh_frequency, mesh_recency = _create_frequency_recency_meshes(
        max_frequency=max_frequency,
        max_recency=max_recency,
    )

    # create dataframe for model input
    transaction_data = pd.DataFrame(
        {
            "customer_id": np.arange(mesh_recency.size),  # placeholder
            "frequency": mesh_frequency.ravel(),
            "recency": mesh_recency.ravel(),
            "T": max_recency,
        }
    )

    # run model predictions to create heatmap values
    Z = (
        model.expected_purchases(
            data=transaction_data,
            future_t=future_t,
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
                future_t, "s"[future_t == 1 :]
            )
            + "\nby Frequency and Recency of a Customer"
        )

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )

    _force_aspect(ax)

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
    """Plot probability alive matrix as a heatmap based on customer population *frequency* and *recency*.

    Parameters
    ----------
    model: CLV model
        A fitted CLV model.
    max_frequency: int, optional
        The maximum *frequency* to plot. Defaults to max observed *frequency*.
    max_recency: int, optional
        The maximum *recency* to plot. This also determines the age of the customer. Defaults to max observed *recency*.
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

    # create dataframe for model input
    transaction_data = pd.DataFrame(
        {
            "customer_id": np.arange(mesh_recency.size),  # placeholder
            "frequency": mesh_frequency.ravel(),
            "recency": mesh_recency.ravel(),
            "T": max_recency,
        }
    )

    # run model predictions to create heatmap values
    Z = (
        model.expected_probability_alive(data=transaction_data)
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
    _force_aspect(ax)

    # plot colorbar beside matrix
    plt.colorbar(pcm, ax=ax)

    return ax


def plot_expected_purchases_over_time(
    model,
    purchase_history: pd.DataFrame,
    customer_id_col: str,
    datetime_col: str,
    t: int,
    plot_cumulative: bool = True,
    t_start_eval: int | None = None,
    datetime_format: str | None = None,
    time_unit: str = "D",
    time_scaler: float | None = 1,
    sort_purchases: bool | None = True,
    set_index_date: bool | None = False,
    title: str | None = None,
    xlabel: str = "Time Periods",
    ylabel: str = "Purchases",
    ax: plt.Axes | None = None,
    t_unobserved: int | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot actual and expected purchases over time for a fitted ``BetaGeoModel`` or ``ParetoNBDModel``.

    This function is based on the formulation on page 8 of [1]_. Specifically, we take only customers who have made
    their first purchase before the specified number of ``t`` time periods, and run
    ``expected_purchases_new_customer()`` for all remaining time periods. Results can be either cumulative or
    incremental.

    Adapted from the legacy ``lifetimes`` library:
    https://github.com/CamDavidsonPilon/lifetimes/blob/master/lifetimes/plotting.py#L392

    Parameters
    ----------
    model :
        A fitted ``BetaGeoModel`` or ``ParetoNBDModel``.
    purchase_history : ~pandas.DataFrame
        A Pandas DataFrame containing *customer_id_col* and *datetime_col*.
    customer_id_col : string
        Column in the *purchases* DataFrame denoting the *customer_id*.
    datetime_col :  string
        Column in the *purchases* DataFrame denoting datetimes purchase were made.
    t : int
        Number of time units since earliest purchase to include in plot.
    plot_cumulative : bool
        Default: *True*
        Plot cumulative purchases over time. Set to *False* to plot incremental purchases.
    t_start_eval : int, optional
        If testing model on unobserved data, specify number of time units in training data to add an indicator for
        the start of the testing period.
    datetime_format : string, optional
        A string that represents the timestamp format. Useful if Pandas doesn't recognize the provided format.
    time_unit : string, optional
        Time granularity for study.
        Default: 'D' for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    time_scaler : int, optional
        Default: 1. Scales *recency* & *T* to a different time granularity.
        This is useful for datasets spanning many years, and running predictions in different time scales.
    sort_purchases : bool, optional
        Default: *True*
        If *purchase_history* DataFrame is already sorted in chronological order,
        set to *False* to improve computational efficiency.
    set_index_date : bool, optional
        Set to True to return a dataframe with a datetime index.
    title : str, optional
        Figure title
    xlabel : str, optional
        Figure xlabel
    ylabel : str, optional
        Figure ylabel
    ax : matplotlib.Axes, optional
        A matplotlib Axes instance. Creates new axes instance by default.
    kwargs
        Additional arguments to pass into the pandas.DataFrame.plot command.

    Returns
    -------
    axes: matplotlib.AxesSubplot

    References
    ----------
    .. [1] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005),
    A Note on Implementing the Pareto/NBD Model in MATLAB.
    http://brucehardie.com/notes/008/
    """
    if ax is None:
        ax = plt.subplot(111)

    df_cum_purchases = _expected_cumulative_transactions(
        model=model,
        transactions=purchase_history,
        customer_id_col=customer_id_col,
        datetime_col=datetime_col,
        t=t,
        datetime_format=datetime_format,
        time_unit=time_unit,
        time_scaler=time_scaler,
        sort_transactions=sort_purchases,
        set_index_date=set_index_date,
    )

    if not plot_cumulative:
        df_cum_purchases = df_cum_purchases.diff()
        if title is None:
            title = "Tracking Incremental Transactions"
    else:
        if title is None:
            title = "Tracking Cumulative Transactions"

    # TODO: After utility func supports xarrays, refactor this for matplotlib API.
    ax = df_cum_purchases.plot(ax=ax, title=title, **kwargs)

    if t_unobserved:
        warnings.warn(
            "t_unobserved is deprecated and will be removed in a future release. "
            "Use t_start_eval instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        t_start_eval = t_unobserved

    if t_start_eval:
        if set_index_date:
            x_vline = df_cum_purchases.index[int(t_start_eval)]
        else:
            x_vline = t_start_eval
        ax.axvline(x=x_vline, color="r", linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_expected_purchases_ppc(
    model,
    ppc: str = "posterior",
    max_purchases: int = 10,
    samples: int = 1000,
    random_seed: int = 45,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot a prior or posterior predictive check for the customer purchase frequency distribution.

    ``ParetoNBDModel``, ``BetaGeoBetaBinomModel``, ``BetaGeoModel`` and ``ModifiedBetaGeoModel`` are supported.

    Adapted from legacy ``lifetimes`` library:
    https://github.com/CamDavidsonPilon/lifetimes/blob/master/lifetimes/plotting.py#L25

    Parameters
    ----------
    model : CLVModel
        Prior predictive checks can be performed before or after a model is fit.
        Posterior predictive checks require a fitted model.
    ppc : string, optional
        Type of predictive check to perform. Options are 'prior' or 'posterior'; defaults to 'posterior'.
    max_purchases : int, optional
        Cutoff for bars of purchase counts to plot. Default is 10.
    samples : int, optional
        Number of samples to draw for prior predictive checks. This is not used for posterior predictive checks.
    random_seed : int, optional
        Random seed to fix sampling results
    ax : matplotlib.Axes, optional
        A matplotlib Axes instance. Creates new axes instance by default.
    **kwargs
        Additional arguments to pass into the pandas.DataFrame.plot command.

    Returns
    -------
    axes : matplotlib.AxesSubplot
    """
    if ax is None:
        ax = plt.subplot(111)

    match ppc:
        case "prior":
            # build model if it has not been fit yet
            model.build_model()

            prior_idata = pm.sample_prior_predictive(
                draws=samples,
                model=model.model,
                random_seed=random_seed,
            )

            # obs_var must be retrieved from prior_idata if model has not been fit
            obs_freq = prior_idata.observed_data["recency_frequency"].sel(
                obs_var="frequency"
            )
            ppc_freq = prior_idata.prior_predictive["recency_frequency"].sel(
                obs_var="frequency"
            )
            title = "Prior Predictive Check for Customer Frequency"
        case "posterior":
            obs_freq = model.idata.observed_data["recency_frequency"].sel(
                obs_var="frequency"
            )
            # Keep samples at 1 here because (chain * draw * customer) samples are already being drawn
            ppc_freq = model.distribution_new_customer_recency_frequency(
                random_seed=random_seed,
                n_samples=1,
            ).sel(obs_var="frequency")
            title = "Posterior Predictive Check for Customer Frequency"
        case _:
            raise NameError("Specify 'prior' or 'posterior' for 'ppc' parameter.")

    # convert estimated and observed xarrays into dataframes for plotting
    estimated = ppc_freq.to_dataframe().value_counts(normalize=True).sort_index()
    observed = obs_freq.to_dataframe().value_counts(normalize=True).sort_index()

    # PPC histogram plot
    ax = pd.DataFrame(
        {
            "Estimated": estimated.reset_index()["proportion"].head(max_purchases),
            "Observed": observed.reset_index()["proportion"].head(max_purchases),
        },
    ).plot(
        kind="bar",
        ax=ax,
        title=title,
        xlabel="Repeat Purchases",
        ylabel="% of Customer Population",
        rot=0.0,
        **kwargs,
    )
    return ax


def _force_aspect(ax: plt.Axes, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)
