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
"""Class to store and validate keyword argument for the Hilbert Space Gaussian Process (HSGP) components."""

from enum import Enum
from typing import Annotated

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pydantic import BaseModel, Field, InstanceOf, model_validator
from pymc.distributions.shape_utils import Dims
from typing_extensions import Self

from pymc_marketing.prior import Prior


def pc_prior_1d(alpha: float = 0.1, lower: float = 1.0) -> Prior:
    """
    One dimensional PC prior for GP lengthscales, parameterized by tail probability:
    p(lengthscale < lower) = alpha.
    """
    lam_ell = -np.log(alpha) * (1.0 / np.sqrt(lower))

    return Prior(
        "Weibull",
        alpha=0.5,
        beta=1.0 / np.square(lam_ell),
        transform="reciprocal",
    )


def approx_hsgp_hyperparams(
    x,
    x_center,
    lengthscale_range: list[float],
    cov_func: str,
) -> tuple[int, float]:
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


class CovFunc(str, Enum):
    ExpQuad = "expquad"
    Matern52 = "matern52"
    Matern32 = "matern32"


class HSGP(BaseModel, extra="allow"):  # type: ignore
    """HSGP configuration.

    Examples
    --------
    HSGP with a Matern52 covariance function

    .. code-block:: python

        import pandas as pd

        import pymc as pm
        import numpy as np

        import matplotlib.pyplot as plt

        from pymc_marketing.hsgp_kwargs import HSGP
        from pymc_marketing.mmm.plot import plot_curve

        hsgp = HSGP(
            ls_lower=1,
            ls_upper=15,
            drop_first=True,
            centered=False,
            cov_func="matern52",
            dims="time",
        )

        n = 52
        X = np.arange(n)
        hsgp.register_data(X)

        dates = pd.date_range("2022-01-01", periods=n, freq="W-MON")
        coords = {
            "time": dates,
        }
        with pm.Model(coords=coords) as model:
            f = hsgp.create_variable("f")

            prior = pm.sample_prior_predictive().prior

        plot_curve(prior["f"], {"time"})
        plt.show()

    New data predictions with HSGP

    .. code-block:: python

        import pandas as pd

        import pymc as pm
        import numpy as np

        import matplotlib.pyplot as plt

        from pymc_marketing.hsgp_kwargs import HSGP
        from pymc_marketing.mmm.plot import plot_curve

        hsgp = HSGP(
            ls_lower=1,
            ls_upper=15,
            drop_first=True,
            centered=True,
            cov_func="matern52",
            dims=("time", "channel"),
        )

        n = 52
        X = np.arange(n)

        dates = pd.date_range("2022-01-01", periods=n, freq="W-MON")
        coords = {
            "time": dates,
            "channel": ["A", "B"]
        }
        with pm.Model(coords=coords) as model:
            data = pm.Data("data", X, dims="time")
            f = (
                hsgp
                .register_data(data)
                .create_variable("f")
            )
            prior = pm.sample_prior_predictive().prior

        with model:
            pm.set_data(
                {
                    "data": np.arange(n, n + 10),
                },
                coords={"time": pd.date_range("2023-01-01", periods=10, freq="W-MON")},
            )
            post = pm.sample_posterior_predictive(prior, var_names=["f"])

        chain, draw = 0, 50
        ax = prior["f"].loc[chain, draw].to_series().unstack().plot()
        post.posterior_predictive["f"].loc[chain, draw].to_series().unstack().plot(ax=ax, linestyle="--")
        plt.show()

    """

    ls_lower: float = Field(1.0, gt=0, description="Lower bound for the lengthscales")
    ls_upper: float | None = Field(
        None, gt=0, description="Upper bound for the lengthscales"
    )
    ls_mass: float = Field(0.90, gt=0, lt=1, description="Mass of the lengthscales")
    eta_upper: float = Field(1.0, gt=0, description="Upper bound for the variance")
    eta_mass: float = Field(0.05, gt=0, lt=1, description="Mass of the variance")
    centered: bool = Field(False, description="Whether the model is centered or not")
    drop_first: bool = Field(
        True, description="Whether to drop the first basis function"
    )
    X: InstanceOf[pt.TensorVariable] | None = Field(
        None,
        description="The data to be used in the model",
        exclude=True,
    )
    X_mid: float | None = Field(None, description="The mean of the data")
    cov_func: CovFunc = Field(CovFunc.ExpQuad, description="The covariance function")

    @model_validator(mode="after")
    def _check_lower_below_upper(self) -> Self:
        if self.ls_upper is not None and self.ls_lower >= self.ls_upper:
            raise ValueError("Lower bound must be below the upper bound")

        return self

    @property
    def eta(self) -> Prior:
        return Prior(
            "Exponential",
            lam=-np.log(self.eta_mass) / self.eta_upper,
        )

    @property
    def ls(self) -> Prior:
        if self.ls_upper is None:
            return pc_prior_1d(alpha=1.0 - self.ls_mass, lower=self.ls_lower)

        return Prior(
            "InverseGamma",
            alpha=2,
            beta=1,
        ).constrain(lower=self.ls_lower, upper=self.ls_upper, mass=self.ls_mass)

    def register_data(self, X: pt.TensorLike):
        """Register the data to be used in the model.

        Parameters
        ----------
        X : tensor_like
            The data to be used in the model.

        Returns
        -------
        self : HSGP
            The object itself.

        """

        self.X = pt.as_tensor_variable(X)

        return self

    def create_variable(self, name: str) -> pt.TensorVariable:
        """Create a variable from HSGP configuration.

        Parameters
        ----------
        name : str
            The name of the variable.

        Returns
        -------
        pt.TensorVariable
            The variable created from the HSGP configuration.

        """

        if self.X is None:
            raise ValueError("The data must be registered before creating a variable.")

        if self.X_mid is None:
            self.X_mid = float(self.X.mean().eval())

        if self.ls_upper is None:
            ls_upper = 2 * self.X_mid
        else:
            ls_upper = self.ls_upper

        lengthscale_range = [self.ls_lower, ls_upper]
        m, c = approx_hsgp_hyperparams(
            self.X.eval(),
            self.X_mid,
            lengthscale_range,
            cov_func=self.cov_func,
        )
        L = c * self.X_mid

        model = pm.modelcontext(None)
        coord_name: str = f"{name}_m"
        model.add_coord(coord_name, np.arange(m - 1))

        hsgp_dims: Dims
        if isinstance(self.dims, tuple):
            hsgp_dims = (self.dims[1], coord_name)
        else:
            hsgp_dims = coord_name

        cov_funcs = {
            "expquad": pm.gp.cov.ExpQuad,
            "matern52": pm.gp.cov.Matern52,
            "matern32": pm.gp.cov.Matern32,
        }
        eta = self.eta.create_variable(f"{name}_eta")
        ls = self.ls.create_variable(f"{name}_ls")

        cov_func = eta**2 * cov_funcs[self.cov_func.lower()](input_dim=1, ls=ls)

        gp = pm.gp.HSGP(m=[m], L=[L], cov_func=cov_func, drop_first=self.drop_first)
        phi, sqrt_psd = gp.prior_linearized(
            self.X[:, None] - self.X_mid,
        )

        hsgp_coefs = Prior(
            "Normal", mu=0, sigma=sqrt_psd, dims=hsgp_dims, centered=self.centered
        ).create_variable(f"{name}_hsgp_coefs")
        f = phi @ hsgp_coefs.T
        return pm.Deterministic(name, f, dims=self.dims)


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
    cov_func: InstanceOf[pm.gp.cov.Covariance] | None = Field(
        None, description="Gaussian process Covariance function"
    )
