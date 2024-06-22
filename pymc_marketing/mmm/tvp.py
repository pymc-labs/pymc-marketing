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


def pc_prior_1d(name, alpha=0.1, lower=1.0):
    """
    One dimensional PC prior for GP lengthscales, parameterized by tail probability:
    p(lengthscale < lower) = alpha.
    """
    lam_ell = -np.log(alpha) * (1.0 / pt.sqrt(lower))
    ell_inv = pm.Weibull(name=f"{name}_inv_", alpha=0.5, beta=1.0 / pt.square(lam_ell))
    return pm.Deterministic(f"{name}", 1 / ell_inv)


def approx_hsgp_hyperparams(x, x_center, lengthscale_range: list[float], cov_func: str):
    """Utility function that uses heuristics to recommend minimum `m` and `c` values,
    based on recommendations from Ruitort-Mayol et. al.

    In practice, you need to choose `c` large enough to handle the largest lengthscales,
    and `m` large enough to accommodate the smallest lengthscales.

    NB: These recommendations are based on a one-dimensional GP.

    Parameters
    ----------
    x : ArrayLike
        The x values the HSGP will be evaluated over.
    lengthscale_range : List[float]
        The range of the lengthscales. Should be a list with two elements [lengthscale_min, lengthscale_max].
    cov_func : str
        The covariance function to use. Supported options are "expquad", "matern52", and "matern32".

    Returns
    -------
    - `m` : int
        Number of basis vectors. Increasing it helps approximate smaller lengthscales, but increases computational cost.
    - `c` : float
        Scaling factor such that L = c * S, where L is the boundary of the approximation.
        Increasing it helps approximate larger lengthscales, but may require increasing m.

    Raises
    ------
    ValueError
        If either `x_range` or `lengthscale_range` is not in the correct order.

    References
    ----------
    - Ruitort-Mayol, G., Anderson, M., Solin, A., Vehtari, A. (2022).
    Practical Hilbert Space Approximate Bayesian Gaussian Processes for Probabilistic Programming
    """
    if lengthscale_range[0] >= lengthscale_range[1]:
        raise ValueError("One of the boundaries out of order")

    Xs = x - x_center
    S = np.max(np.abs(Xs), axis=0)

    if cov_func.lower() == "expquad":
        a1, a2 = 3.2, 1.75

    elif cov_func.lower() == "matern52":
        a1, a2 = 4.1, 2.65

    elif cov_func.lower() == "matern32":
        a1, a2 = 4.5, 3.42

    else:
        raise ValueError(
            "Unsupported covariance function. Supported options are 'expquad', 'matern52', and 'matern32'."
        )

    c = max(a1 * (lengthscale_range[1] / S), 1.2)
    m = int(a2 * c / (lengthscale_range[0] / S))

    return m, c


def time_varying_prior(
    name: str,
    X: pt.sharedvar.TensorSharedVariable,
    dims: tuple[str, str] | str,
    ls_lower: float = 1,
    ls_upper: float | None = None,
    ls_mass: float = 0.9,
    eta_upper: float = 1.0,
    eta_mass: float = 0.05,
    centered: bool = False,
    drop_first: bool = False,
    m: int = None,
    L: int | float | None = None,
    X_mid: int | float | None = None,
    cov_func: str | None = "Matern52",
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
    ls_mu : float
        Mean of the inverse gamma prior for the lengthscale.
    ls_sigma : float
        Standard deviation of the inverse gamma prior for the lengthscale.
    ls_mass: float
        Mass of prior below ls_lower.  Must be a value between 0 and 1.
        Default is 0.05.
    ls_lower: float
        Lower limit of the lengthscale.
    eta_upper: float
        Upper bound on your expectation for the magnitude of the time varying parameter's
        variation.  Default value is 1.
    eta_mass: float:
        Mass of prior above eta_upper.  Default is 0.05.
    m : int
        Number of basis functions.
    L : int
        Extent of basis functions. Set this to reflect the expected range of
        in+out-of-sample data (considering that time-indices are zero-centered).
        Default is `X_mid * 2` (identical to `c=2` in HSGP).
    centered : bool
        Whether to use the centered or non-centered parameterization.
    drop_first: bool
        Default False.
    cov_func : str
        Covariance function name, must be one of `Matern52`, `Matern32` or `Expquad`.

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

    model = pm.modelcontext(None)

    eta = pm.Exponential(f"{name}_eta", lam=-np.log(eta_mass) / eta_upper)

    if ls_upper is None:
        ls = pc_prior_1d(f"{name}_ls", alpha=1.0 - ls_mass, lower=ls_lower)
        ls_upper = 2 * X_mid
    else:
        # Note: this function sometimes fails, how to handle?
        # If not using PC prior, should user set ls_mu and ls_sigma instead?
        ls_params = pm.find_constrained_prior(
            pm.InverseGamma,
            lower=ls_lower,
            upper=ls_upper,
            mass=ls_mass,
            init_guess={"alpha": 2, "beta": 1},
        )
        ls = pm.InverseGamma("f{name}_ls", **ls_params)

    lengthscale_range = [ls_lower, ls_upper]
    m, c = approx_hsgp_hyperparams(
        X.eval(), X_mid, lengthscale_range, cov_func=cov_func
    )
    L = c * X_mid

    model.add_coord("m", np.arange(m - 1))  # type: ignore
    hsgp_dims: str | tuple[str, str] = "m"
    if isinstance(dims, tuple):
        hsgp_dims = (dims[1], "m")

    cov_funcs = {
        "expquad": pm.gp.cov.ExpQuad,
        "matern52": pm.gp.cov.Matern52,
        "matern32": pm.gp.cov.Matern32,
    }
    cov_func = eta**2 * cov_funcs[cov_func.lower()](input_dim=1, ls=ls)

    gp = pm.gp.HSGP(m=[m], L=[L], cov_func=cov_func, drop_first=True)
    phi, sqrt_psd = gp.prior_linearized(Xs=X[:, None] - X_mid)

    if centered:
        hsgp_coefs = pm.Normal(f"{name}_hsgp_coefs", sigma=sqrt_psd, dims=hsgp_dims)
        f = phi @ hsgp_coefs
    else:
        hsgp_coefs = pm.Normal(f"{name}_hsgp_coefs", dims=hsgp_dims)
        f = phi @ (hsgp_coefs * sqrt_psd)

    return pm.Deterministic(name, f, dims=dims)


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
        # if model_config["intercept_tvp_kwargs"]["L"] is None:
        #    model_config["intercept_tvp_kwargs"]["L"] = (
        #        time_index_mid + DAYS_IN_YEAR / time_resolution
        #    )
        # if model_config["intercept_tvp_kwargs"]["ls_mu"] is None:
        #    model_config["intercept_tvp_kwargs"]["ls_mu"] = (
        #        DAYS_IN_YEAR / time_resolution * 2
        #    )

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
            var=pt.exp(intercept_base + multiplier),
            dims="date",
        )


def infer_time_index(
    date_series_new: pd.Series, date_series: pd.Series, time_resolution: int
) -> npt.NDArray[np.int_]:
    """Infer the time-index given a new dataset.

    Infers the time-indices by calculating the number of days since the first date in the dataset.
    """
    return (date_series_new - date_series[0]).dt.days.values // time_resolution
