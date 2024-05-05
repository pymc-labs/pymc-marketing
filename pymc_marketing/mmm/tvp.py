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
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from pymc_marketing.constants import DAYS_IN_YEAR


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
    """Time varying prior, based on the Hilbert Space Gaussian Process (HSGP).

    For more information see `pymc.gp.HSGP <https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.HSGP.html>`_.

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
    pt.TensorVariable
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

    model = pm.modelcontext(None)

    if cov_func is None:
        eta = pm.Exponential(f"{name}_eta", lam=eta_lam)
        ls = pm.InverseGamma(f"{name}_ls", mu=ls_mu, sigma=ls_sigma)
        cov_func = eta**2 * pm.gp.cov.Matern52(1, ls=ls)

    model.add_coord("m", np.arange(m))  # type: ignore
    hsgp_dims: str | tuple[str, str] = "m"
    if isinstance(dims, tuple):
        hsgp_dims = (dims[1], "m")

    gp = pm.gp.HSGP(m=[m], L=[L], cov_func=cov_func)
    phi, sqrt_psd = gp.prior_linearized(Xs=X[:, None] - X_mid)
    hsgp_coefs = pm.Normal(f"{name}_hsgp_coefs", dims=hsgp_dims)
    f = phi @ (hsgp_coefs * sqrt_psd).T
    centered_f = f - f.mean(axis=0) + 1
    return pm.Deterministic(name, centered_f, dims=dims)


def create_time_varying_intercept(
    time_index: pt.sharedvar.TensorSharedVariable,
    time_index_mid: int,
    time_resolution: int,
    intercept_dist: pm.Distribution,
    model_config: dict,
) -> pt.TensorVariable:
    """Create time-varying intercept.

    Parameters
    ----------
    time_index : 1d array-like of int
        Time points.
    time_index_mid : int
        Midpoint of the time points.
    time_resolution : int
        Time resolution.
    model_config : dict
        Model configuration.
    """

    with pm.modelcontext(None):
        if model_config["intercept_tvp_kwargs"]["L"] is None:
            model_config["intercept_tvp_kwargs"]["L"] = (
                time_index_mid + DAYS_IN_YEAR / time_resolution
            )
        if model_config["intercept_tvp_kwargs"]["ls_mu"] is None:
            model_config["intercept_tvp_kwargs"]["ls_mu"] = (
                DAYS_IN_YEAR / time_resolution * 2
            )

        multiplier = time_varying_prior(
            name="intercept_time_varying_multiplier",
            X=time_index,
            dims="date",
            X_mid=time_index_mid,
            **model_config["intercept_tvp_kwargs"],
        )
        intercept_base = intercept_dist(
            name="intercept_base", **model_config["intercept"]["kwargs"]
        )
        return pm.Deterministic(
            name="intercept",
            var=intercept_base * multiplier,
            dims="date",
        )


def infer_time_index(
    date_series_new: pd.Series, date_series: pd.Series, time_resolution: int
) -> npt.NDArray[np.int_]:
    """Infer the time-index given a new dataset.

    Infers the time-indices by calculating the number of days since the first date in the dataset.
    """
    return (date_series_new - date_series[0]).dt.days.values // time_resolution
