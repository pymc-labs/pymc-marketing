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
"""Class to store and validate keyword argument for the Hilbert Space Gaussian Process (HSGP) components."""

from typing import Annotated

import pymc as pm
from pydantic import BaseModel, Field, InstanceOf

from pymc_marketing.deserialize import register_deserialization


class HSGPKwargs(BaseModel):
    """HSGP keyword arguments for the time-varying prior.

    See [1]_ and [2]_ for the theoretical background on the Hilbert Space Gaussian Process (HSGP).
    See , [6]_ for a practical guide through the method using code examples.
    See the :class:`~pymc.gp.HSGP` class for more information on the Hilbert Space Gaussian Process in PyMC.
    We also recommend the following resources for a more practical introduction to HSGP: [3]_, [4]_, [5]_.

    References
    ----------
    .. [1] Solin, A., Sarkka, S. (2019) Hilbert Space Methods for Reduced-Rank Gaussian Process Regression.
    .. [2] Ruitort-Mayol, G., and Anderson, M., and Solin, A., and Vehtari, A. (2022). Practical Hilbert Space Approximate Bayesian Gaussian Processes for Probabilistic Programming.
    .. [3] PyMC Example Gallery: `"Gaussian Processes: HSGP Reference & First Steps" <https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Basic.html>`_.
    .. [4] PyMC Example Gallery: `"Gaussian Processes: HSGP Advanced Usage" <https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Advanced.html>`_.
    .. [5] PyMC Example Gallery: `"Baby Births Modelling with HSGPs" <https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Births.html>`_.
    .. [6] Orduz, J. `"A Conceptual and Practical Introduction to Hilbert Space GPs Approximation Methods" <https://juanitorduz.github.io/hsgp_intro/>`_.

    Parameters
    ----------
    m : int
        Number of basis functions. Default is 200.
    L : float, optional
        Extent of basis functions. Set this to reflect the expected range of in+out-of-sample data
        (considering that time-indices are zero-centered).Default is `X_mid * 2` (identical to `c=2` in HSGP).
        By default it is None.
    eta_lam : float
        Exponential prior for the variance. Default is 1.
    ls_mu : float
        Mean of the inverse gamma prior for the lengthscale. Default is 5.
    ls_sigma : float
        Standard deviation of the inverse gamma prior for the lengthscale. Default is 5.
    cov_func : ~pymc.gp.cov.Covariance, optional
        Gaussian process Covariance function. By default it is None.
    """  # noqa E501

    m: int = Field(200, description="Number of basis functions")
    L: (
        Annotated[
            float,
            Field(
                gt=0,
                description="""
                Extent of basis functions. Set this to reflect the expected range of in+out-of-sample data
                (considering that time-indices are zero-centered).Default is `X_mid * 2` (identical to `c=2` in HSGP)
                """,
            ),
        ]
        | None
    ) = None
    eta_lam: float = Field(1, gt=0, description="Exponential prior for the variance")
    ls_mu: float = Field(
        5, gt=0, description="Mean of the inverse gamma prior for the lengthscale"
    )
    ls_sigma: float = Field(
        5,
        gt=0,
        description="Standard deviation of the inverse gamma prior for the lengthscale",
    )
    cov_func: InstanceOf[pm.gp.cov.Covariance] | str | None = Field(
        None, description="Gaussian process Covariance function"
    )


def _is_hsgp_kwargs(data) -> bool:
    return isinstance(data, dict) and data.keys() == {
        "m",
        "L",
        "eta_lam",
        "ls_mu",
        "ls_sigma",
        "cov_func",
    }


register_deserialization(
    is_type=_is_hsgp_kwargs,
    deserialize=lambda data: HSGPKwargs.model_validate(data),
)
