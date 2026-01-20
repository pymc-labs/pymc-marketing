#   Copyright 2022 - 2026 The PyMC Labs Developers
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
"""Time Varying Gaussian Process Multiplier for Marketing Mix Modeling (MMM).

Designed to model time-varying effects in marketing mix models (MMM).

This module provides a time-varying Gaussian Process (GP) multiplier,
using the Hilbert Space Gaussian Process (HSGP) approximation.

Examples
--------
Create a basic PyMC model using the time-varying GP multiplier:

.. code-block:: python

    import numpy as np
    import pymc as pm
    import pandas as pd

    from pymc_marketing.hsgp_kwargs import HSGPKwargs
    from pymc_marketing.mmm.tvp import (
        create_time_varying_gp_multiplier,
        infer_time_index,
    )

    # Generate example data
    np.random.seed(0)
    dates = pd.Series(pd.date_range(start="2020-01-01", periods=365))
    sales = np.random.normal(100, 10, size=len(dates))

    # Infer time index
    time_index = infer_time_index(dates, dates, time_resolution=5)

    # Define model configuration
    hsgp_kwargs = HSGPKwargs(
        m=200,
        L=None,
        eta_lam=1,
        ls_mu=10,
        ls_sigma=5,
        cov_func=None,
    )

    coords = {"time": dates}
    with pm.Model(coords=coords) as model:
        # Shared time index variable
        time_index_shared = pm.Data("time_index", time_index)

        # Base parameter
        base_sales = pm.Normal("base_sales", mu=100, sigma=10)

        # Time-varying GP multiplier
        varying_coefficient = create_time_varying_gp_multiplier(
            name="sales",
            dims="time",
            time_index=time_index_shared,
            time_index_mid=int(len(dates) / 2),
            time_resolution=5,
            hsgp_kwargs=hsgp_kwargs,
        )

        # Final sales parameter
        sales_estimated = base_sales * varying_coefficient

        # Likelihood
        pm.Normal("obs", mu=sales_estimated, sigma=10, observed=sales)

    # Sample from the model
    with model:
        trace = pm.sample()

    # Plot results
    import matplotlib.pyplot as plt

    pm.plot_trace(trace, var_names=["base_sales"])
    plt.show()

"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pymc.distributions.shape_utils import Dims
from pymc_extras.prior import Prior
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.constants import DAYS_IN_YEAR
from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.mmm.dims import XTensorLike
from pymc_marketing.mmm.hsgp import CovFunc, SoftPlusHSGP

__all__ = [
    "HSGP_KWARGS_DISTINCTIVE_KEYS",
    "create_hsgp_from_config",
    "create_time_varying_gp_multiplier",
    "infer_time_index",
    "is_hsgp_kwargs_format",
    "time_varying_prior",
]

# Keys unique to HSGPKwargs format (not present in parameterize_from_data)
# m and L are HSGPKwargs-only keys that parameterize_from_data computes internally
HSGP_KWARGS_DISTINCTIVE_KEYS: frozenset[str] = frozenset(
    {"m", "L", "eta_lam", "ls_mu", "ls_sigma"}
)


def is_hsgp_kwargs_format(config: dict[str, Any]) -> bool:
    """Check if a dictionary uses HSGPKwargs-style keys.

    This function detects whether a configuration dictionary follows the
    HSGPKwargs format (with keys like ``eta_lam``, ``ls_mu``, ``ls_sigma``) or
    the ``parameterize_from_data`` format (with keys like ``ls_lower``, ``ls_upper``).

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary to check.

    Returns
    -------
    bool
        True if the config contains HSGPKwargs-style keys, False otherwise.

    Examples
    --------
    HSGPKwargs format detection:

    >>> is_hsgp_kwargs_format({"m": 200, "eta_lam": 1.0, "ls_mu": 5.0})
    True

    parameterize_from_data format detection:

    >>> is_hsgp_kwargs_format({"ls_lower": 0.3, "ls_upper": 2.0})
    False

    """
    return bool(config.keys() & HSGP_KWARGS_DISTINCTIVE_KEYS)


def _create_hsgp_instance(
    X,
    X_mid,
    dims: Dims,
    hsgp_kwargs: HSGPKwargs,
) -> SoftPlusHSGP:
    X = as_xtensor(X)
    eta = Prior("Exponential", lam=float(hsgp_kwargs.eta_lam))
    ls = Prior(
        "InverseGamma", mu=float(hsgp_kwargs.ls_mu), sigma=float(hsgp_kwargs.ls_sigma)
    )
    cov_func = (
        hsgp_kwargs.cov_func
        if isinstance(hsgp_kwargs.cov_func, CovFunc)
        else CovFunc.Matern52
    )

    if X_mid is None:
        X_mid = float(X.mean().eval())

    if hsgp_kwargs.L is None:
        L = X_mid * 2
    else:
        L = hsgp_kwargs.L

    return SoftPlusHSGP(
        eta=eta,
        ls=ls,
        m=hsgp_kwargs.m,
        L=L,
        cov_func=cov_func,
        X=X,
        X_mid=X_mid,
        centered=False,
        dims=dims,
        drop_first=False,
    )


def create_hsgp_from_config(
    X: XTensorLike,
    dims: Dims,
    config: HSGPKwargs | dict[str, Any],
    X_mid: int | float | None = None,
) -> SoftPlusHSGP:
    """Create a SoftPlusHSGP instance from either HSGPKwargs or dict config.

    This function provides a unified interface for creating SoftPlusHSGP
    instances, supporting both the legacy HSGPKwargs format and the newer
    parameterize_from_data format.

    Parameters
    ----------
    X : XTensorLike
        Time index or input data for the HSGP.
    dims : Dims
        Dimensions for the HSGP variable (e.g., ``("date",)`` or ``("date", "channel")``).
    config : HSGPKwargs | dict[str, Any]
        Configuration for the HSGP. Can be:

        - An ``HSGPKwargs`` instance
        - A dict with HSGPKwargs keys (``m``, ``L``, ``eta_lam``, ``ls_mu``, ``ls_sigma``, ``cov_func``)
        - A dict with parameterize_from_data keys (``ls_lower``, ``ls_upper``, etc.)

    X_mid : int | float | None, optional
        Midpoint of the time index. If None, computed from X.

    Returns
    -------
    SoftPlusHSGP
        Configured SoftPlusHSGP instance ready for use.

    Raises
    ------
    TypeError
        If config is neither HSGPKwargs, dict, nor a valid format.

    See Also
    --------
    pymc_marketing.hsgp_kwargs.HSGPKwargs : Legacy configuration format.
    pymc_marketing.mmm.hsgp.SoftPlusHSGP.parameterize_from_data : New configuration format.

    Examples
    --------
    Using HSGPKwargs instance:

    >>> from pymc_marketing.hsgp_kwargs import HSGPKwargs
    >>> config = HSGPKwargs(m=200, eta_lam=1.0, ls_mu=5.0, ls_sigma=10.0)
    >>> hsgp = create_hsgp_from_config(X=np.arange(52), dims="date", config=config)

    Using parameterize_from_data format dict:

    >>> config = {"ls_lower": 0.3, "ls_upper": 2.0}
    >>> hsgp = create_hsgp_from_config(X=np.arange(52), dims="date", config=config)

    """
    # Case 1: HSGPKwargs instance -> use existing _create_hsgp_instance
    if isinstance(config, HSGPKwargs):
        return _create_hsgp_instance(X=X, X_mid=X_mid, dims=dims, hsgp_kwargs=config)

    # Case 2: Must be a dict
    if not isinstance(config, dict):
        raise TypeError(
            f"config must be HSGPKwargs or dict, got {type(config).__name__}"
        )

    # Case 3: Dict with HSGPKwargs-style keys -> convert to HSGPKwargs first
    if is_hsgp_kwargs_format(config):
        hsgp_kwargs = HSGPKwargs.model_validate(config)
        return _create_hsgp_instance(
            X=X, X_mid=X_mid, dims=dims, hsgp_kwargs=hsgp_kwargs
        )

    # Case 4: Dict with parameterize_from_data keys -> use SoftPlusHSGP.parameterize_from_data
    # Filter out keys that are explicitly passed to avoid "multiple values" error
    reserved_keys = {"X", "dims", "X_mid"}
    filtered_config = {k: v for k, v in config.items() if k not in reserved_keys}
    return SoftPlusHSGP.parameterize_from_data(
        X=X, dims=dims, X_mid=X_mid, **filtered_config
    )


def time_varying_prior(
    name: str,
    X: XTensorVariable,
    dims: Dims,
    X_mid: int | float | None = None,
    hsgp_kwargs: HSGPKwargs | None = None,
) -> XTensorVariable:
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
    hsgp_kwargs : HSGPKwargs
        Keyword arguments for the Hilbert Space Gaussian Process. By default it is None,
        in which case the default parameters are used. See `HSGPKwargs` for more information.

    Returns
    -------
    XTensorVariable
        Time-varying prior.

    References
    ----------
    -   Ruitort-Mayol, G., and Anderson, M., and Solin, A., and Vehtari, A. (2022). Practical
        Hilbert Space Approximate Bayesian Gaussian Processes for Probabilistic Programming

    -   Solin, A., Sarkka, S. (2019) Hilbert Space Methods for Reduced-Rank Gaussian Process
        Regression.

    """
    if hsgp_kwargs is None:
        hsgp_kwargs = HSGPKwargs()

    hsgp = _create_hsgp_instance(
        X=X,
        X_mid=X_mid,
        dims=dims,
        hsgp_kwargs=hsgp_kwargs,
    )
    return hsgp.create_variable(name, xdist=True)


def create_time_varying_gp_multiplier(
    name: str,
    dims: Dims,
    time_index: XTensorVariable,
    time_index_mid: int,
    time_resolution: int,
    hsgp_kwargs: HSGPKwargs,
) -> XTensorVariable:
    """Create a time-varying Gaussian Process multiplier.

    Create a time-varying Gaussian Process multiplier based on the provided parameters.

    Parameters
    ----------
    name : str
        Name of the Gaussian Process multiplier.
    dims : tuple[str, str] | str
        Dimensions for the multiplier.
    time_index : XTensorVariable
        Shared variable containing time points.
    time_index_mid : int
        Midpoint of the time points.
    time_resolution : int
        Resolution of time points.
    hsgp_kwargs : HSGPKwargs
        Keyword arguments for the Hilbert Space Gaussian Process (HSGP) component.

    Returns
    -------
    XTensorVariable
        Time-varying Gaussian Process multiplier for a given variable.

    """
    if hsgp_kwargs.L is None:
        hsgp_kwargs.L = time_index_mid + DAYS_IN_YEAR / time_resolution
    if hsgp_kwargs.ls_mu is None:
        hsgp_kwargs.ls_mu = DAYS_IN_YEAR / time_resolution * 2

    return time_varying_prior(
        name=f"{name}_temporal_latent_multiplier",
        X=time_index,
        X_mid=time_index_mid,
        dims=dims,
        hsgp_kwargs=hsgp_kwargs,
    )


def infer_time_index(
    date_series_new: pd.Series,
    date_series: pd.Series,
    time_resolution: int,
) -> npt.NDArray[np.int_]:
    """Infer the time-index given a new dataset.

    Infers the time-indices by calculating the number of days since the first date in the dataset.

    Parameters
    ----------
    date_series_new : pd.Series
        New date series.
    date_series : pd.Series
        Original date series.
    time_resolution : int
        Resolution of time points in days.

    Returns
    -------
    np.ndarray
        Time index.

    """
    return (date_series_new - date_series.iloc[0]).dt.days.values // time_resolution
