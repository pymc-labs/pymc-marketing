import arviz as az
import matplotlib.pyplot as plt
import xarray as xr


def plot_hdi_func(x, Y, ax=None):
    """Plot the posterior mean and 95% and 50% CI's of a univariate function
    Parameters
    ----------
    x : vector
        Xarray object of x values. Size (dim)
    Y : array of ints
        Xarray object of y values. Size (chain, draw, dim)
    """

    if ax is None:
        _, ax = plt.subplots()

    quantiles = Y.quantile(
        (0.025, 0.25, 0.5, 0.75, 0.975), dim=("chain", "draw")
    ).transpose()

    az.plot_hdi(
        x,
        hdi_data=quantiles.sel(quantile=[0.025, 0.975]),
        ax=ax,
        fill_kwargs={"label": "95% ETI", "alpha": 0.25},
    )
    az.plot_hdi(
        x,
        hdi_data=quantiles.sel(quantile=[0.25, 0.75]),
        ax=ax,
        fill_kwargs={"label": "50% ETI", "alpha": 0.5},
    )
    ax.plot(x, quantiles.sel(quantile=0.5), label="Median")
    ax.legend()
    return ax


def plot_survival_function_fixed_theta(
    theta_samples, max_time=50, θtrue=None, ax=None, legend=True, data_horizon=None
):
    """Plot the survival function for a fixed churn rate `theta`"""
    if ax is None:
        _, ax = plt.subplots()
    time = xr.DataArray(range(0, max_time), dims="time")
    r = 1 - theta_samples
    S = r ** time
    plot_hdi_func(time, S, ax=ax)
    if θtrue is not None:
        ax.plot(time, (1 - θtrue) ** time, "k--", label="true")
    if data_horizon is not None:
        ax.axvline(
            x=data_horizon, color="k", ls=":", label="data horizon",
        )
    if legend:
        ax.legend()
    ax.set(
        title=r"Survival function - constant churn rate $\theta$",
        xlabel="customer lifetime, $t$",
        ylabel="S(t)",
    )
    return ax
