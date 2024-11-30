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

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel, Field, InstanceOf, model_validator
from pymc.distributions.shape_utils import Dims
from pytensor.tensor import TensorLike
from pytensor.tensor.variable import TensorVariable
from typing_extensions import Self

from pymc_marketing.plot import SelToString, plot_curve
from pymc_marketing.prior import Prior, create_dim_handler


def pc_prior_1d(alpha: float = 0.1, lower: float = 1.0) -> Prior:
    R"""Create a one-dimensional PC prior for GP lengthscale.

    The prior is defined with the following property:

    .. math::

        P[\ell < \text{lower}] = \alpha

    Where :math:`\ell` is the lengthscale

    Parameters
    ----------
    alpha : float
        Tail probability.
    lower : float
        Lower bound for the lengthscale.

    Returns
    -------
    Prior
        Weibull prior for the lengthscale with the given properties.
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
    lengthscale_range: tuple[float, float],
    cov_func: str,
) -> tuple[int, float]:
    """Use heuristics for minimum `m` and `c` values.

    Based on recommendations from Ruitort-Mayol et. al.

    In practice, you need to choose `c` large enough to handle the largest
    lengthscales, and `m` large enough to accommodate the smallest lengthscales.

    NOTE: These recommendations are based on a one-dimensional GP.

    Parameters
    ----------
    x : tensor_like
        The x values the HSGP will be evaluated over.
    x_center : tensor_like
        The center of the data.
    lengthscale_range : tuple[float, float]
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
    lengthscale_min, lengthscale_max = lengthscale_range
    if lengthscale_min >= lengthscale_max:
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

    c = max(a1 * (lengthscale_max / S), 1.2)
    m = int(a2 * c / (lengthscale_min / S))

    return m, c


class CovFunc(str, Enum):
    """Supported covariance functions for the HSGP model."""

    ExpQuad = "expquad"
    Matern52 = "matern52"
    Matern32 = "matern32"


class HSGP(BaseModel, extra="allow"):  # type: ignore
    """HSGP component for the time-varying prior.

    Examples
    --------
    HSGP with default configuration:

    .. plot::
        :include-source: True
        :context: reset

        import numpy as np
        import pandas as pd

        import matplotlib.pyplot as plt

        from pymc_marketing.hsgp_kwargs import HSGP

        seed = sum(map(ord, "Out of the box GP"))
        rng = np.random.default_rng(seed)

        hsgp = HSGP(dims="time")

        n = 52
        X = np.arange(n)
        hsgp.register_data(X)

        dates = pd.date_range("2022-01-01", periods=n, freq="W-MON")
        coords = {
            "time": dates,
        }
        prior = hsgp.sample_prior(coords=coords, random_seed=rng)
        hsgp.plot_curve(prior["f"])
        plt.show()

    HSGP with different covariance function

    .. plot::
        :include-source: True
        :context: reset

        import numpy as np
        import pandas as pd

        import matplotlib.pyplot as plt

        from pymc_marketing.hsgp_kwargs import HSGP

        seed = sum(map(ord, "Change of covariance function"))
        rng = np.random.default_rng(seed)

        hsgp = HSGP(
            cov_func="matern52",
            dims="time",
        )

        n = 52
        X = np.arange(n)

        dates = pd.date_range("2022-01-01", periods=n, freq="W-MON")
        coords = {
            "time": dates,
        }
        hsgp.register_data(X)
        prior = hsgp.sample_prior(coords=coords, random_seed=rng)
        hsgp.plot_curve(prior["f"])
        plt.show()

    New data predictions with HSGP

    .. plot::
        :include-source: True
        :context: reset

        import numpy as np
        import pandas as pd

        import pymc as pm

        import matplotlib.pyplot as plt

        from pymc_marketing.hsgp_kwargs import HSGP

        seed = sum(map(ord, "New data predictions"))
        rng = np.random.default_rng(seed)

        hsgp = HSGP(dims=("time", "channel"))

        n = 52
        X = np.arange(n)

        dates = pd.date_range("2022-01-01", periods=n, freq="W-MON")
        coords = {"time": dates, "channel": ["A", "B"]}
        with pm.Model(coords=coords) as model:
            data = pm.Data("data", X, dims="time")
            f = hsgp.register_data(data).create_variable("f")
            idata = pm.sample_prior_predictive(random_seed=rng)

        prior = idata.prior

        n_new = 10
        X_new = np.arange(n, n + n_new)
        new_dates = pd.date_range("2023-01-01", periods=n_new, freq="W-MON")

        with model:
            pm.set_data(
                new_data={
                    "data": X_new,
                },
                coords={"time": new_dates},
            )
            post = pm.sample_posterior_predictive(prior, var_names=["f"], random_seed=rng)

        chain, draw = 0, 50
        colors = ["C0", "C1"]


        def get_sample(curve):
            return curve.loc[chain, draw].to_series().unstack()


        ax = prior["f"].pipe(get_sample).plot(color=colors)
        post.posterior_predictive["f"].pipe(get_sample).plot(
            ax=ax, color=colors, linestyle="--", legend=False
        )
        ax.set(xlabel="time", ylabel="f", title="New data predictions")
        plt.show()

    Higher dimensional HSGP

    .. plot::
        :include-source: True
        :context: reset

        import numpy as np
        import pymc as pm

        import matplotlib.pyplot as plt

        from pymc_marketing.hsgp_kwargs import HSGP

        seed = sum(map(ord, "Higher dimensional HSGP"))
        rng = np.random.default_rng(seed)

        hsgp = HSGP(dims=("time", "channel", "product"))

        n = 52
        X = np.arange(n)
        hsgp.register_data(X)

        coords = {
            "time": range(n),
            "channel": ["A", "B"],
            "product": ["X", "Y", "Z"],
        }
        prior = hsgp.sample_prior(coords=coords, random_seed=rng)
        curve = prior["f"]
        fig, _ = hsgp.plot_curve(
            curve,
            subplot_kwargs={"figsize": (12, 8), "ncols": 3},
        )
        fig.suptitle("Higher dimensional HSGP prior")
        plt.show()

    """

    ls_lower: float = Field(1.0, gt=0, description="Lower bound for the lengthscales")
    ls_upper: float | None = Field(
        None,
        gt=0,
        description="Upper bound for the lengthscales",
    )
    ls_mass: float = Field(0.90, gt=0, lt=1, description="Mass of the lengthscales")
    eta_upper: float = Field(1.0, gt=0, description="Upper bound for the variance")
    eta_mass: float = Field(0.05, gt=0, lt=1, description="Mass of the variance")
    centered: bool = Field(False, description="Whether the model is centered or not")
    drop_first: bool = Field(
        True, description="Whether to drop the first basis function"
    )
    X: InstanceOf[TensorVariable] | None = Field(
        None,
        description="The data to be used in the model",
        exclude=True,
    )
    X_mid: float | None = Field(None, description="The mean of the data")
    cov_func: CovFunc = Field(CovFunc.ExpQuad, description="The covariance function")
    dims: Dims = Field(..., description="The dimensions of the variable")

    @model_validator(mode="after")
    def _check_lower_below_upper(self) -> Self:
        if self.ls_upper is not None and self.ls_lower >= self.ls_upper:
            raise ValueError("Lower bound must be below the upper bound")

        return self

    @model_validator(mode="after")
    def _dim_is_at_least_one(self) -> Self:
        if isinstance(self.dims, str):
            self.dims = (self.dims,)
            return self

        if isinstance(self.dims, tuple) and len(self.dims) < 1:
            raise ValueError("At least one dimension is required")

        if any(not isinstance(dim, str) for dim in self.dims):
            raise ValueError("All dimensions must be strings")

        return self

    @property
    def eta(self) -> Prior:
        """The prior for the variance."""
        return Prior(
            "Exponential",
            lam=-np.log(self.eta_mass) / self.eta_upper,
        )

    @property
    def ls(self) -> Prior:
        """The prior for the lengthscales."""
        if self.ls_upper is None:
            return pc_prior_1d(alpha=1.0 - self.ls_mass, lower=self.ls_lower)

        return Prior(
            "InverseGamma",
        ).constrain(lower=self.ls_lower, upper=self.ls_upper, mass=self.ls_mass)

    def register_data(self, X: TensorLike) -> Self:
        """Register the data to be used in the model.

        To be used before creating a variable but not for out-of-sample prediction.
        For out-of-sample prediction, use `pm.Data` and `pm.set_data`.

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

    def sample_prior(
        self,
        coords: dict | None = None,
        **sample_prior_predictive_kwargs,
    ) -> xr.Dataset:
        """Sample from the prior distribution.

        Parameters
        ----------
        coords : dict, optional
            The coordinates for the prior. By default it is None.
        sample_prior_predictive_kwargs
            Additional keyword arguments for `pm.sample_prior_predictive`.

        Returns
        -------
        xr.Dataset
            The prior distribution.

        """
        coords = coords or {}
        with pm.Model(coords=coords) as model:
            self.create_variable("f")

        return pm.sample_prior_predictive(
            model=model,
            **sample_prior_predictive_kwargs,
        ).prior

    def plot_curve(
        self,
        curve: xr.DataArray,
        subplot_kwargs: dict | None = None,
        sample_kwargs: dict | None = None,
        hdi_kwargs: dict | None = None,
        axes: npt.NDArray[Axes] | None = None,
        same_axes: bool = False,
        colors: Iterable[str] | None = None,
        legend: bool | None = None,
        sel_to_string: SelToString | None = None,
    ) -> tuple[Figure, npt.NDArray[Axes]]:
        """Plot the curve from the prior.

        Parameters
        ----------
        curve : xr.DataArray
            Curve to plot
        subplot_kwargs : dict, optional
            Additional kwargs to while creating the fig and axes
        sample_kwargs : dict, optional
            Kwargs for the :func:`plot_samples` function
        hdi_kwargs : dict, optional
            Kwargs for the :func:`plot_hdi` function
        same_axes : bool
            If all of the plots are on the same axis
        colors : Iterable[str], optional
            Colors for the plots
        legend : bool, optional
            If to include a legend. Defaults to True if same_axes
        sel_to_string : Callable[[Selection], str], optional
            Function to convert selection to a string. Defaults to
            ", ".join(f"{key}={value}" for key, value in sel.items())

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]
            Figure and the axes

        """
        first_dim: str = self.dims if isinstance(self.dims, str) else self.dims[0]
        return plot_curve(
            curve,
            non_grid_names={first_dim},
            subplot_kwargs=subplot_kwargs,
            sample_kwargs=sample_kwargs,
            hdi_kwargs=hdi_kwargs,
            axes=axes,
            same_axes=same_axes,
            colors=colors,
            legend=legend,
            sel_to_string=sel_to_string,
        )

    def create_variable(self, name: str) -> TensorVariable:
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

        m, c = approx_hsgp_hyperparams(
            self.X.eval(),
            self.X_mid,
            lengthscale_range=(self.ls_lower, ls_upper),
            cov_func=self.cov_func,
        )
        L = c * self.X_mid

        model = pm.modelcontext(None)
        coord_name: str = f"{name}_m"
        model.add_coord(coord_name, np.arange(m - 1))

        first_dim, *rest_dims = self.dims
        hsgp_dims: Dims = (*rest_dims, coord_name)

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
            "Normal",
            mu=0,
            sigma=sqrt_psd,
            dims=hsgp_dims,
            centered=self.centered,
        ).create_variable(f"{name}_hsgp_coefs")
        # (date, m-1) and (*rest_dims, m-1) -> (date, *rest_dims)
        if len(rest_dims) <= 1:
            f = phi @ hsgp_coefs.T
        else:
            result_dims = (first_dim, coord_name, *rest_dims)
            dim_handler = create_dim_handler(desired_dims=result_dims)
            f = (
                dim_handler(phi, (first_dim, coord_name))
                * dim_handler(
                    hsgp_coefs,
                    hsgp_dims,
                )
            ).sum(axis=1)
        return pm.Deterministic(name, f, dims=self.dims)

    def to_dict(self) -> dict:
        """Convert the object to a dictionary.

        Returns
        -------
        dict
            The object as a dictionary.

        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data) -> HSGP:
        """Create an object from a dictionary.

        Parameters
        ----------
        data : dict
            The data to create the object from.

        Returns
        -------
        HSGP
            The object created from the data.

        """
        return cls(**data)


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
