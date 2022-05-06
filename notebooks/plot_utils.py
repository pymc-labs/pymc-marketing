import arviz as az
import matplotlib.pyplot as plt


def plot_xY(x, Y, ax=None):
    """Plot the posterior mean and 95% and 50% CI's from a given set of x and Y values.
    x: is a vector
    Y: is an xarrray of size (chain, draw, dim)
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
        fill_kwargs={"label": "95% ETI"},
    )
    az.plot_hdi(
        x,
        hdi_data=quantiles.sel(quantile=[0.25, 0.75]),
        ax=ax,
        fill_kwargs={"label": "50% ETI"},
    )
    ax.plot(x, quantiles.sel(quantile=0.5), label="Median")
    ax.legend()

    return ax
