import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor import softplus


def time_varying_prior(
    name: str,
    X: pt.sharedvar.TensorSharedVariable,
    dims: tuple[str, str] | str,
    X_mid: int | float | None = None,
    m: int = 200,
    L: int | float | None = None,
    eta_lam: float = 1,
    ls_mu: float = 5,
    ls_sigma: float = 5,
    cov_func: pm.gp.cov.Covariance | None = None,
) -> pt.TensorVariable:
    """Time varying prior, based the Hilbert Space Gaussian Process (HSGP).

    For more information see [pymc.gp.HSGP](https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.HSGP.html).

    Parameters
    ----------
    name : str
        Name of the prior and associated variables.
    X : 1d array-like of int or float
        Time points.
    X_mid : int or float
        Midpoint of the time points.
    dims : tuple of str or str
        Dimensions of the prior. If a tuple, the first element is the name of
        the time dimension, and the second may be any other dimension, across
        which independent time varying priors for each coordinate are desired
        (e.g. channels).
    m : int
        Number of basis functions.
    L : int
        Extent of basis functions. Set this to reflect the expected range of
        in+out-of-sample data (considering that time-indices are zero-centered).
        Default is `X_mid * 2` (identical to `c=2` in HSGP).
    eta_lam : float
        Exponential prior for the variance.
    ls_mu : float
        Mean of the inverse gamma prior for the lengthscale.
    ls_sigma : float
        Standard deviation of the inverse gamma prior for the lengthscale.
    cov_func : pm.gp.cov.Covariance
        Covariance function.

    Returns
    -------
    pm.Deterministic
        Time-varying prior.

    References
    ----------
    -   Ruitort-Mayol, G., and Anderson, M., and Solin, A., and Vehtari, A. (2022). Practical
        Hilbert Space Approximate Bayesian Gaussian Processes for Probabilistic Programming

    -   Solin, A., Sarkka, S. (2019) Hilbert Space Methods for Reduced-Rank Gaussian Process
        Regression.
    """

    if X_mid is None:
        X_mid = float(X.mean().eval())
    if L is None:
        L = X_mid * 2

    with pm.modelcontext(None) as model:
        if cov_func is None:
            eta = pm.Exponential(f"{name}_eta", lam=eta_lam)
            ls = pm.InverseGamma(f"{name}_ls", mu=ls_mu, sigma=ls_sigma)
            cov_func = eta**2 * pm.gp.cov.Matern52(1, ls=ls)

        model.add_coord("m", np.arange(m))  # type: ignore
        hsgp_dims: str | tuple[str, str] = "m"
        if isinstance(dims, tuple):
            hsgp_dims = (dims[0], "m")

        gp = pm.gp.HSGP(m=[m], L=[L], cov_func=cov_func)
        phi, sqrt_psd = gp.prior_linearized(Xs=X[:, None] - X_mid)
        hsgp_coefs = pm.Normal(f"{name}_hsgp_coefs", dims=hsgp_dims)
        f = softplus(phi @ (hsgp_coefs * sqrt_psd).T)
        return pm.Deterministic(name, f, dims=dims)
