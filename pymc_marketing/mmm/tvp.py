from typing import Optional

import pymc as pm
from pytensor.tensor import softplus


def time_varying_prior(
    name: str,
    X: pm.Deterministic,
    X_mid: int | float,
    dims: tuple[str, str] | str,
    m: int,
    L: int | float,
    eta_lam: float = 1,
    ls_mu: float = 5,
    ls_sigma: float = 5,
    cov_func: Optional[pm.gp.cov.Covariance] = None,
) -> pm.Deterministic:
    """Time varying prior, based the Hilbert Space Gaussian Process (HSGP).

    Parameters
    ----------
    name : str
        Name of the prior.
    X : 1d array-like of int or float
        Time points.
    X_mid : int or float
        Midpoint of the time points.
    dims : tuple of str or str
        Dimensions of the prior.
    m : int
        Number of basis functions.
    L : int
        Extent of basis functions. Set this to reflect the expected range of
        in+out-of-sample data (considering that time-indices are zero-centered)
    eta_lam : float
        Exponential prior for the variance.
    ls_mu : float
        Mean of the inverse gamma prior for the lengthscale.
    ls_sigma : float
        Standard deviation of the inverse gamma prior for the lengthscale.
    cov_func : pm.gp.cov.Prod
        Covariance function.

    Returns
    -------
    pm.Deterministic
        Time-varying prior.
    """  # noqa: W605

    with pm.modelcontext(None) as model:
        if cov_func is None:
            eta = pm.Exponential(f"eta_{name}", lam=eta_lam)
            ls = pm.InverseGamma(f"ls_{name}", mu=ls_mu, sigma=ls_sigma)
            cov_func = eta**2 * pm.gp.cov.Matern52(1, ls=ls)

        hsgp_size: int | tuple[int, int] = m
        if type(dims) is tuple:
            n_columns = len(model.coords[dims[1]])
            hsgp_size = (n_columns, m)

        gp = pm.gp.HSGP(m=[m], L=[L], cov_func=cov_func)
        phi, sqrt_psd = gp.prior_linearized(Xs=X[:, None] - X_mid)
        hsgp_coefs = pm.Normal(f"_hsgp_coefs_{name}", size=hsgp_size)
        f = softplus(phi @ (hsgp_coefs * sqrt_psd).T)
        return pm.Deterministic(name, f, dims=dims)
