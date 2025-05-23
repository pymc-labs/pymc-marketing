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
"""HSGP components."""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel, Field, InstanceOf, model_validator, validate_call
from pymc.distributions.shape_utils import Dims
from pytensor.tensor import TensorLike
from pytensor.tensor.variable import TensorVariable
from typing_extensions import Self

from pymc_marketing.plot import SelToString, plot_curve
from pymc_marketing.prior import Prior, _get_transform, create_dim_handler


@validate_call
def create_complexity_penalizing_prior(
    *,
    alpha: float = Field(0.1, gt=0, lt=1),
    lower: float = Field(1.0, gt=0),
) -> Prior:
    R"""Create prior that penalizes complexity for GP lengthscale.

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

    References
    ----------
    .. [1] Geir-Arne Fuglstad, Daniel Simpson, Finn Lindgren, Håvard Rue (2015).

    """
    lam_ell = -pt.log(alpha) * (1.0 / pt.sqrt(lower))

    return Prior(
        "Weibull",
        alpha=0.5,
        beta=1.0 / pt.square(lam_ell),
        transform="reciprocal",
    )


def create_constrained_inverse_gamma_prior(
    *,
    upper: float,
    lower: float = 1.0,
    mass: float = 0.9,
) -> Prior:
    """Create a lengthscale prior for the HSGP model.

    Parameters
    ----------
    upper : float
        Upper bound for the lengthscale.
    lower : float
        Lower bound for the lengthscale. Default is 1.0.
    mass : float
        Mass of the lengthscales. Default is 0.9 or 90%.

    Returns
    -------
    Prior
        The prior for the lengthscale.

    """
    return Prior(
        "InverseGamma",
    ).constrain(lower=lower, upper=upper, mass=mass)


def approx_hsgp_hyperparams(
    x,
    x_center,
    lengthscale_range: tuple[float, float],
    cov_func: Literal["expquad", "matern32", "matern52"],
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
    cov_func : Literal["expquad", "matern32", "matern52"]
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
    .. [1] Ruitort-Mayol, G., Anderson, M., Solin, A., Vehtari, A. (2022).
    Practical Hilbert Space Approximate Bayesian Gaussian Processes for Probabilistic Programming
    """
    lengthscale_min, lengthscale_max = lengthscale_range
    if lengthscale_min >= lengthscale_max:
        raise ValueError(
            "The boundaries are out of order. {lengthscale_min} should be less than {lengthscale_max}"
        )

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


def create_eta_prior(mass: float = 0.05, upper: float = 1.0) -> Prior:
    """Create prior for the variance.

    Parameters
    ----------
    mass : float
        Mass of the variance. Default is 0.05 or 5%.
    upper : float
        Upper bound for the variance. Default is 1.0.

    """
    return Prior(
        "Exponential",
        lam=-pt.log(mass) / upper,
    )


def create_m_and_L_recommendations(
    X: np.ndarray,
    X_mid: float,
    ls_lower: float = 1.0,
    ls_upper: float | None = None,
    cov_func: CovFunc = CovFunc.ExpQuad,
) -> tuple[int, float]:
    """Create recommendations for the number of basis functions based on the data.

    Parameters
    ----------
    X : np.ndarray
        The data.
    X_mid : float
        The mean of the data.
    ls_lower : float
        Lower bound for the lengthscale. Default is 1.0.
    ls_upper : float, optional
        Upper bound for the lengthscale. Default is None.
    cov_func : CovFunc
        The covariance function. Default is CovFunc.ExpQuad.

    Returns
    -------
    tuple[int, float]
        The number of basis functions and the boundary of the approximation.

    """
    if ls_upper is None:
        ls_upper = 2 * X_mid
    else:
        ls_upper = ls_upper

    m, c = approx_hsgp_hyperparams(
        X,
        X_mid,
        lengthscale_range=(ls_lower, ls_upper),
        cov_func=cast(Literal["expquad", "matern32", "matern52"], cov_func),
    )
    L = c * X_mid

    return m, L


class HSGPBase(BaseModel):
    """Shared logic between HSGP and HSGPPeriodic."""

    m: int = Field(..., description="Number of basis functions")
    X: InstanceOf[TensorVariable] | InstanceOf[np.ndarray] | None = Field(
        None,
        description="The data to be used in the model",
        exclude=True,
    )
    X_mid: float | None = Field(None, description="The mean of the training data")
    dims: Dims = Field(..., description="The dimensions of the variable")
    transform: str | None = Field(
        None,
        description=(
            "Optional transformation for the variable. "
            "Must be registered or from either pytensor.tensor "
            "or pymc.math namespaces."
        ),
    )
    demeaned_basis: bool = Field(
        False,
        description="Whether each basis has its mean subtracted from it.",
    )

    @model_validator(mode="after")
    def _transform_is_valid(self) -> Self:
        if self.transform is None:
            return self

        # Not storing but will get again later
        _get_transform(self.transform)
        return self

    @staticmethod
    def deterministics_to_replace(name: str) -> list[str]:
        """Name of the Deterministic variables that are replaced with pm.Flat for out-of-sample.

        This is required for out-of-sample predictions for some HSGP variables. Replacing with
        pm.Flat will sample from the values learned in the training data.

        """
        return []

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
        Self
            The object with the data registered.

        """
        self.X = pt.as_tensor_variable(X)

        return self

    @model_validator(mode="after")
    def _register_user_input_X(self) -> Self:
        if self.X is None:
            return self

        return self.register_data(self.X)

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

    def create_variable(self, name: str) -> TensorVariable:
        """Create a variable from configuration."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Convert the object to a dictionary.

        Returns
        -------
        dict
            The object as a dictionary.

        """
        data = self.model_dump()

        def handle_prior(value):
            return value if not hasattr(value, "to_dict") else value.to_dict()

        return {key: handle_prior(value) for key, value in data.items()}

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
        n_samples: int = 10,
        hdi_probs: float | list[float] | None = None,
        random_seed: np.random.Generator | None = None,
        subplot_kwargs: dict | None = None,
        sample_kwargs: dict | None = None,
        hdi_kwargs: dict | None = None,
        axes: npt.NDArray[Axes] | None = None,
        same_axes: bool = False,
        colors: Iterable[str] | None = None,
        legend: bool | None = None,
        sel_to_string: SelToString | None = None,
    ) -> tuple[Figure, npt.NDArray[Axes]]:
        """Plot the curve.

        Parameters
        ----------
        curve : xr.DataArray
            Curve to plot
        n_samples : int, optional
            Number of samples
        hdi_probs : float | list[float], optional
            HDI probabilities. Defaults to None which uses arviz default for
            stats.ci_prob which is 94%
        random_seed : int | random number generator, optional
            Random number generator. Defaults to None
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
        first_dim: str = cast(tuple[str, ...], self.dims)[0]
        return plot_curve(
            curve,
            non_grid_names={first_dim},
            n_samples=n_samples,
            hdi_probs=hdi_probs,
            random_seed=random_seed,
            subplot_kwargs=subplot_kwargs,
            sample_kwargs=sample_kwargs,
            hdi_kwargs=hdi_kwargs,
            axes=axes,
            same_axes=same_axes,
            colors=colors,
            legend=legend,
            sel_to_string=sel_to_string,
        )


class HSGP(HSGPBase):
    """HSGP component.

    Examples
    --------
    Literature recommended HSGP configuration:

    .. plot::
        :include-source: True
        :context: reset

        import numpy as np
        import pandas as pd

        import matplotlib.pyplot as plt

        from pymc_marketing.mmm import HSGP

        seed = sum(map(ord, "Out of the box GP"))
        rng = np.random.default_rng(seed)

        n = 52
        X = np.arange(n)

        hsgp = HSGP.parameterize_from_data(
            X=X,
            dims="time",
        )

        dates = pd.date_range("2022-01-01", periods=n, freq="W-MON")
        coords = {
            "time": dates,
        }
        prior = hsgp.sample_prior(coords=coords, random_seed=rng)
        curve = prior["f"]
        hsgp.plot_curve(curve, random_seed=rng)
        plt.show()

    Using a demeaned basis to remove "intercept" effect of first basis:

    .. plot::
        :include-source: True
        :context: reset

        import numpy as np
        import pandas as pd
        import xarray as xr

        import matplotlib.pyplot as plt

        from pymc_marketing.mmm import HSGP
        from pymc_marketing.plot import plot_curve

        seed = sum(map(ord, "Out of the box GP"))
        rng = np.random.default_rng(seed)

        n = 52
        X = np.arange(n)

        kwargs = dict(X=X, ls=25, eta=1, dims="time", m=200, L=150, drop_first=False)

        hsgp = HSGP(demeaned_basis=False, **kwargs)
        hsgp_demeaned = HSGP(demeaned_basis=True, **kwargs)

        dates = pd.date_range("2022-01-01", periods=n, freq="W-MON")
        coords = {"time": dates}


        def sample_curve(hsgp):
            return hsgp.sample_prior(coords=coords, random_seed=rng)["f"]


        non_demeaned = sample_curve(hsgp).rename("False")
        demeaned = sample_curve(hsgp_demeaned).rename("True")

        combined = xr.merge([non_demeaned, demeaned]).to_array("demeaned")
        _, axes = combined.pipe(plot_curve, "time", same_axes=True)
        axes[0].set(title="Demeaned the intercepty first basis")
        plt.show()


    HSGP with different covariance function

    .. plot::
        :include-source: True
        :context: reset

        import numpy as np
        import pandas as pd

        import matplotlib.pyplot as plt

        from pymc_marketing.mmm import HSGP

        seed = sum(map(ord, "Change of covariance function"))
        rng = np.random.default_rng(seed)

        n = 52
        X = np.arange(n)

        hsgp = HSGP.parameterize_from_data(
            X=X,
            cov_func="matern32",
            dims="time",
        )

        dates = pd.date_range("2022-01-01", periods=n, freq="W-MON")
        coords = {
            "time": dates,
        }
        prior = hsgp.sample_prior(coords=coords, random_seed=rng)
        curve = prior["f"]
        hsgp.plot_curve(curve, random_seed=rng)
        plt.show()

    HSGP with different link function via `transform` argument

    .. note::

        The `transform` parameter must be registered or from either `pytensor.tensor`
        or `pymc.math` namespaces. See the :func:`pymc_marketing.prior.register_tensor_transform`

    .. plot::
        :include-source: True
        :context: reset

        import numpy as np
        import pandas as pd

        import matplotlib.pyplot as plt

        from pymc_marketing.mmm import HSGP

        seed = sum(map(ord, "Change of covariance function"))
        rng = np.random.default_rng(seed)

        n = 52
        X = np.arange(n)

        hsgp = HSGP.parameterize_from_data(
            X=X,
            dims="time",
            transform="sigmoid",
        )

        dates = pd.date_range("2022-01-01", periods=n, freq="W-MON")
        coords = {
            "time": dates,
        }
        prior = hsgp.sample_prior(coords=coords, random_seed=rng)
        curve = prior["f"]
        hsgp.plot_curve(curve, random_seed=rng)
        plt.show()

    New data predictions with HSGP

    .. plot::
        :include-source: True
        :context: reset

        import numpy as np
        import pandas as pd

        import pymc as pm

        import matplotlib.pyplot as plt

        from pymc_marketing.mmm import HSGP
        from pymc_marketing.prior import Prior

        seed = sum(map(ord, "New data predictions"))
        rng = np.random.default_rng(seed)

        eta = Prior("Exponential", lam=1)
        ls = Prior("InverseGamma", alpha=2, beta=1)
        hsgp = HSGP(
            eta=eta,
            ls=ls,
            m=20,
            L=150,
            dims=("time", "channel"),
        )

        n = 52
        X = np.arange(n)

        dates = pd.date_range("2022-01-01", periods=n, freq="W-MON")
        coords = {"time": dates, "channel": ["A", "B"]}
        with pm.Model(coords=coords) as model:
            data = pm.Data("data", X, dims="time")
            hsgp.register_data(data).create_variable("f")
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
            post = pm.sample_posterior_predictive(
                prior,
                var_names=["f"],
                random_seed=rng,
            )

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

        from pymc_marketing.mmm import HSGP

        seed = sum(map(ord, "Higher dimensional HSGP"))
        rng = np.random.default_rng(seed)

        n = 52
        X = np.arange(n)

        hsgp = HSGP.parameterize_from_data(
            X=X,
            dims=("time", "channel", "product"),
        )

        coords = {
            "time": range(n),
            "channel": ["A", "B"],
            "product": ["X", "Y", "Z"],
        }
        prior = hsgp.sample_prior(coords=coords, random_seed=rng)
        curve = prior["f"]
        fig, _ = hsgp.plot_curve(
            curve,
            random_seed=rng,
            subplot_kwargs={"figsize": (12, 8), "ncols": 3},
        )
        fig.suptitle("Higher dimensional HSGP prior")
        plt.show()

    """

    ls: InstanceOf[Prior] | float = Field(..., description="Prior for the lengthscales")
    eta: InstanceOf[Prior] | float = Field(..., description="Prior for the variance")
    L: float = Field(..., gt=0, description="Extent of basis functions")
    centered: bool = Field(False, description="Whether the model is centered or not")
    drop_first: bool = Field(
        True, description="Whether to drop the first basis function"
    )
    cov_func: CovFunc = Field(CovFunc.ExpQuad, description="The covariance function")

    @model_validator(mode="after")
    def _ls_is_scalar_prior(self) -> Self:
        if not isinstance(self.ls, Prior):
            return self

        if self.ls.dims != ():
            raise ValueError("The lengthscale prior must be scalar random variable.")

        return self

    @model_validator(mode="after")
    def _eta_is_scalar_prior(self) -> Self:
        if not isinstance(self.eta, Prior):
            return self

        if self.eta.dims != ():
            raise ValueError("The eta prior must be scalar random variable.")

        return self

    @classmethod
    def parameterize_from_data(
        cls,
        X: TensorLike | np.ndarray,
        dims: Dims,
        X_mid: float | None = None,
        eta_mass: float = 0.05,
        eta_upper: float = 1.0,
        ls_lower: float = 1.0,
        ls_upper: float | None = None,
        ls_mass: float = 0.9,
        cov_func: CovFunc = CovFunc.ExpQuad,
        centered: bool = False,
        drop_first: bool = True,
        transform: str | None = None,
        demeaned_basis: bool = False,
    ) -> HSGP:
        """Create a HSGP informed by the data with literature-based recommendations."""
        eta = create_eta_prior(mass=eta_mass, upper=eta_upper)
        if ls_upper is None:
            ls = create_complexity_penalizing_prior(
                lower=ls_lower,
                alpha=ls_mass,
            )
        else:
            ls = create_constrained_inverse_gamma_prior(
                lower=ls_lower,
                upper=ls_upper,
                mass=ls_mass,
            )

        numeric_X = None
        if isinstance(X, np.ndarray):
            numeric_X = np.asarray(X)
        elif isinstance(X, TensorVariable):
            numeric_X = X.get_value(borrow=False)
        else:
            raise ValueError(
                "X must be a NumPy array (or list) or a TensorVariable. "
                "If it's a plain symbolic tensor, you must manually specify m, L."
            )

        numeric_X = np.asarray(numeric_X)
        if X_mid is None:
            X_mid = float(numeric_X.mean())

        m, L = create_m_and_L_recommendations(
            numeric_X,
            X_mid,
            ls_lower=ls_lower,
            ls_upper=ls_upper,
            cov_func=cov_func,
        )

        return cls(
            ls=ls,
            eta=eta,
            m=m,
            L=L,
            X=X,  # store the original reference (even if symbolic)
            X_mid=X_mid,
            cov_func=cov_func,
            dims=dims,
            centered=centered,
            drop_first=drop_first,
            transform=transform,
            demeaned_basis=demeaned_basis,
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

        def add_suffix(suffix: str) -> str:
            if self.transform is None:
                return f"{name}_{suffix}"

            return f"{name}_raw_{suffix}"

        if self.X is None:
            raise ValueError("The data must be registered before creating a variable.")

        if self.X_mid is None:
            self.X_mid = float(self.X.mean().eval())

        model = pm.modelcontext(None)
        coord_name: str = add_suffix("m")
        model.add_coord(
            coord_name,
            np.arange(self.m - 1 if self.drop_first else self.m),
        )

        cov_funcs = {
            "expquad": pm.gp.cov.ExpQuad,
            "matern52": pm.gp.cov.Matern52,
            "matern32": pm.gp.cov.Matern32,
        }
        eta = (
            self.eta
            if not hasattr(self.eta, "create_variable")
            else self.eta.create_variable(add_suffix("eta"))
        )
        ls = (
            self.ls
            if not hasattr(self.ls, "create_variable")
            else self.ls.create_variable(add_suffix("ls"))
        )

        cov_func = eta**2 * cov_funcs[self.cov_func.lower()](input_dim=1, ls=ls)

        gp = pm.gp.HSGP(
            m=[self.m],
            L=[self.L],
            cov_func=cov_func,
            drop_first=self.drop_first,
        )
        phi, sqrt_psd = gp.prior_linearized(
            self.X[:, None] - self.X_mid,
        )
        if self.demeaned_basis:
            phi = phi - phi.mean(axis=0).eval()

        first_dim, *rest_dims = self.dims
        hsgp_dims: Dims = (*rest_dims, coord_name)
        hsgp_coefs = Prior(
            "Normal",
            mu=0,
            sigma=sqrt_psd,
            dims=hsgp_dims,
            centered=self.centered,
        ).create_variable(add_suffix("hsgp_coefs"))
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

        if self.transform is not None:
            raw_name = f"{name}_raw"
            f = pm.Deterministic(raw_name, f, dims=self.dims)

            transform = _get_transform(self.transform)
            f = pm.Deterministic(name, transform(f), dims=self.dims)
        else:
            f = pm.Deterministic(name, f, dims=self.dims)

        return f

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
        for key in ["eta", "ls"]:
            if isinstance(data[key], dict):
                data[key] = Prior.from_dict(data[key])

        return cls(**data)


class PeriodicCovFunc(str, Enum):
    """Supported covariance functions for the HSGP model."""

    Periodic = "periodic"


class HSGPPeriodic(HSGPBase):
    """HSGP component for periodic data.

    Examples
    --------
    HSGPPeriodic with default configuration:

    .. plot::
        :include-source: True
        :context: reset

        import numpy as np
        import pandas as pd

        import matplotlib.pyplot as plt

        from pymc_marketing.mmm import HSGPPeriodic
        from pymc_marketing.prior import Prior

        seed = sum(map(ord, "Periodic GP"))
        rng = np.random.default_rng(seed)

        n = 52 * 3
        dates = pd.date_range("2023-01-01", periods=n, freq="W-MON")
        X = np.arange(n)
        coords = {
            "time": dates,
        }
        scale = Prior("HalfNormal", sigma=1)
        ls = Prior("InverseGamma", alpha=2, beta=1)

        hsgp = HSGPPeriodic(
            scale=scale,
            m=20,
            cov_func="periodic",
            ls=ls,
            period=52,
            dims="time",
        )
        hsgp.register_data(X)

        prior = hsgp.sample_prior(coords=coords, random_seed=rng)
        curve = prior["f"]
        fig, axes = hsgp.plot_curve(
            curve,
            n_samples=3,
            random_seed=rng,
        )
        ax = axes[0]
        ax.set(xlabel="Date", ylabel="f", title="HSGP with period of 52 weeks")
        plt.show()

    HSGPPeriodic with link function via `transform` argument

    .. note::

        The `transform` parameter must be registered or from either `pytensor.tensor`
        or `pymc.math` namespaces. See the :func:`pymc_marketing.prior.register_tensor_transform`

    .. plot::
        :include-source: True
        :context: reset

        import numpy as np
        import pandas as pd

        import matplotlib.pyplot as plt

        from pymc_marketing.mmm import HSGPPeriodic
        from pymc_marketing.prior import Prior

        seed = sum(map(ord, "Periodic GP"))
        rng = np.random.default_rng(seed)

        n = 52 * 3
        dates = pd.date_range("2023-01-01", periods=n, freq="W-MON")
        X = np.arange(n)
        coords = {
            "time": dates,
        }
        scale = Prior("Gamma", mu=0.25, sigma=0.1)
        ls = Prior("InverseGamma", alpha=2, beta=1)

        hsgp = HSGPPeriodic(
            scale=scale,
            m=20,
            cov_func="periodic",
            ls=ls,
            period=52,
            dims="time",
            transform="exp",
        )
        hsgp.register_data(X)

        prior = hsgp.sample_prior(coords=coords, random_seed=rng)
        curve = prior["f"]
        fig, axes = hsgp.plot_curve(
            curve,
            n_samples=3,
            random_seed=rng,
        )
        ax = axes[0]
        ax.set(xlabel="Date", ylabel="f", title="HSGP with period of 52 weeks")
        plt.show()

    Demeaned basis for HSGPPeriodic

    .. plot::
        :include-source: True
        :context: reset

        import numpy as np
        import pandas as pd
        import xarray as xr

        import matplotlib.pyplot as plt

        from pymc_marketing.mmm import HSGPPeriodic
        from pymc_marketing.plot import plot_curve

        seed = sum(map(ord, "Periodic GP"))
        rng = np.random.default_rng(seed)

        scale = 0.25
        ls = 1
        kwargs = dict(ls=ls, scale=scale, period=52, cov_func="periodic", dims="time", m=20)

        n = 52 * 3
        dates = pd.date_range("2023-01-01", periods=n, freq="W-MON")
        X = np.arange(n)
        coords = {"time": dates}

        hsgp = HSGPPeriodic(demeaned_basis=False, **kwargs).register_data(X)
        hsgp_demeaned = HSGPPeriodic(demeaned_basis=True, **kwargs).register_data(X)


        def sample_curve(hsgp):
            return hsgp.sample_prior(coords=coords, random_seed=rng)["f"]


        non_demeaned = sample_curve(hsgp).rename("False")
        demeaned = sample_curve(hsgp_demeaned).rename("True")

        combined = xr.merge([non_demeaned, demeaned]).to_array("demeaned")
        _, axes = combined.pipe(plot_curve, "time", same_axes=True)
        axes[0].set(title="Demeaned the intercepty first basis")
        plt.show()

    Higher dimensional HSGPPeriodic with periodic data

    .. plot::
        :include-source: True
        :context: reset

        import numpy as np
        import pandas as pd

        import pymc as pm

        import matplotlib.pyplot as plt

        from pymc_marketing.mmm import HSGPPeriodic
        from pymc_marketing.prior import Prior

        seed = sum(map(ord, "Higher dimensional HSGP with periodic data"))
        rng = np.random.default_rng(seed)

        n = 52 * 3
        dates = pd.date_range("2023-01-01", periods=n, freq="W-MON")
        X = np.arange(n)

        scale = Prior("HalfNormal", sigma=1)
        ls = Prior("InverseGamma", alpha=2, beta=1)

        hsgp = HSGPPeriodic(
            X=X,
            scale=scale,
            ls=ls,
            m=20,
            cov_func="periodic",
            period=52,
            dims=("time", "channel", "product"),
        )

        coords = {
            "time": dates,
            "channel": ["A", "B"],
            "product": ["X", "Y", "Z"],
        }
        prior = hsgp.sample_prior(coords=coords, random_seed=rng)
        curve = prior["f"]
        fig, axes = hsgp.plot_curve(
            curve,
            n_samples=3,
            random_seed=rng,
            subplot_kwargs={"figsize": (12, 8), "ncols": 3},
        )
        plt.show()

    """

    ls: InstanceOf[Prior] | float = Field(..., description="Prior for the lengthscale")
    scale: InstanceOf[Prior] | float = Field(..., description="Prior for the scale")
    cov_func: PeriodicCovFunc = Field(
        PeriodicCovFunc.Periodic,
        description="The covariance function",
    )
    period: float = Field(..., description="The period of the function")

    @model_validator(mode="after")
    def _ls_is_scalar_prior(self) -> Self:
        if not isinstance(self.ls, Prior):
            return self

        if self.ls.dims != ():
            raise ValueError("The lengthscale prior must be scalar random variable.")

        return self

    @model_validator(mode="after")
    def _scale_is_scalar_prior(self) -> Self:
        if not isinstance(self.scale, Prior):
            return self

        if self.scale.dims != ():
            raise ValueError("The scale prior must be scalar random variable.")

        return self

    def create_variable(self, name: str) -> TensorVariable:
        """Create HSGP variable.

        Parameters
        ----------
        name : str
            The name of the variable.

        Returns
        -------
        pt.TensorVariable
            The variable created from the HSGP configuration.

        """

        def add_suffix(suffix: str) -> str:
            if self.transform is None:
                return f"{name}_{suffix}"

            return f"{name}_raw_{suffix}"

        if self.X is None:
            raise ValueError("The data must be registered before creating a variable.")

        if self.X_mid is None:
            self.X_mid = float(self.X.mean().eval())

        scale = (
            self.scale
            if not hasattr(self.scale, "create_variable")
            else self.scale.create_variable(add_suffix("scale"))
        )

        ls = (
            self.ls
            if not hasattr(self.ls, "create_variable")
            else self.ls.create_variable(add_suffix("ls"))
        )
        cov_func = pm.gp.cov.Periodic(1, period=self.period, ls=ls)

        gp = pm.gp.HSGPPeriodic(m=self.m, scale=scale, cov_func=cov_func)

        (phi_cos, phi_sin), psd = gp.prior_linearized(
            self.X[:, None] - self.X_mid,
        )
        if self.demeaned_basis:
            phi_cos = phi_cos - phi_cos.mean(axis=0).eval()
            phi_sin = phi_sin - phi_sin.mean(axis=0).eval()

        model = pm.modelcontext(None)
        coord_name: str = add_suffix("m")
        model.add_coord(coord_name, np.arange((self.m * 2) - 1))
        first_dim, *rest_dims = self.dims
        hsgp_dims: Dims = (*rest_dims, coord_name)
        sigma = pt.concatenate([psd, psd[..., 1:]])
        hsgp_coefs = Prior(
            "Normal",
            mu=0,
            sigma=sigma,
            dims=hsgp_dims,
            centered=False,
        ).create_variable(add_suffix("hsgp_coefs"))
        phi = pt.concatenate([phi_cos, phi_sin[..., 1:]], axis=1)

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

        if self.transform is not None:
            raw_name = f"{name}_raw"
            f = pm.Deterministic(raw_name, f, dims=self.dims)

            transform = _get_transform(self.transform)
            f = pm.Deterministic(name, transform(f), dims=self.dims)
        else:
            f = pm.Deterministic(name, f, dims=self.dims)

        return f

    @classmethod
    def from_dict(cls, data) -> HSGPPeriodic:
        """Create an object from a dictionary.

        Parameters
        ----------
        data : dict
            The data to create the object from.

        Returns
        -------
        HSGPPeriodic
            The object created from the data.

        """
        for key in ["scale", "ls"]:
            if isinstance(data[key], dict):
                data[key] = Prior.from_dict(data[key])

        return cls(**data)


class SoftPlusHSGP(HSGP):
    """HSGP with softplus transformation.

    The use of the softplus transformation centers the data
    around 1 and keeps the values positive.

    Examples
    --------
    Literature recommended SoftPlusHSGP configuration:

    .. plot::
        :include-source: True
        :context: reset

        import numpy as np
        import pandas as pd

        import matplotlib.pyplot as plt

        from pymc_marketing.mmm import SoftPlusHSGP

        seed = sum(map(ord, "Out of the box GP"))
        rng = np.random.default_rng(seed)

        n = 52
        X = np.arange(n)

        hsgp = SoftPlusHSGP.parameterize_from_data(
            X=X,
            dims="time",
        )

        dates = pd.date_range("2022-01-01", periods=n, freq="W-MON")
        coords = {
            "time": dates,
        }
        prior = hsgp.sample_prior(coords=coords, random_seed=rng)
        curve = prior["f"]
        hsgp.plot_curve(curve, random_seed=rng)
        plt.show()

    New data predictions with SoftPlusHSGP

    .. note::

        In making the predictions, the model needs to be transformed in order to
        keep the data centered around 1. There is a helper function
        :func:`pymc_marketing.model_graph.deterministics_to_flat` that can be used
        to transform the model upon out of sample predictions.

        This transformation is automatically handled in the provided MMMs.

    .. plot::
        :include-source: True
        :context: reset

        import numpy as np
        import pandas as pd

        import pymc as pm

        import matplotlib.pyplot as plt

        from pymc_marketing.mmm import SoftPlusHSGP
        from pymc_marketing.model_graph import deterministics_to_flat
        from pymc_marketing.prior import Prior

        seed = sum(map(ord, "New data predictions with SoftPlusHSGP"))
        rng = np.random.default_rng(seed)

        eta = Prior("Exponential", lam=1)
        ls = Prior("InverseGamma", alpha=2, beta=1)
        hsgp = SoftPlusHSGP(
            eta=eta,
            ls=ls,
            m=20,
            L=150,
            dims=("time", "channel"),
        )

        n = 52
        X = np.arange(n)

        channels = ["A", "B", "C"]
        dates = pd.date_range("2022-01-01", periods=n, freq="W-MON")
        coords = {"time": dates, "channel": channels}
        with pm.Model(coords=coords) as model:
            data = pm.Data("data", X, dims="time")
            hsgp.register_data(data).create_variable("f")
            idata = pm.sample_prior_predictive(random_seed=rng)

        prior = idata.prior

        n_new = 10
        X_new = np.arange(n - 1 , n + n_new)
        last_date = dates[-1]
        new_dates = pd.date_range(last_date, periods=n_new + 1, freq="W-MON")

        with deterministics_to_flat(model, hsgp.deterministics_to_replace("f")):
            pm.set_data(
                new_data={
                    "data": X_new,
                },
                coords={"time": new_dates},
            )
            post = pm.sample_posterior_predictive(
                prior,
                var_names=["f"],
                random_seed=rng,
            )

        chain, draw = 0, rng.choice(prior.sizes["draw"])
        colors = [f"C{i}" for i in range(len(channels))]


        def get_sample(curve):
            return curve.loc[chain, draw].to_series().unstack()


        ax = prior["f"].pipe(get_sample).plot(color=colors)
        post.posterior_predictive["f"].pipe(get_sample).plot(
            ax=ax, color=colors, linestyle="--", legend=False
        )
        ax.set(xlabel="time", ylabel="f", title="New data predictions")
        plt.show()
    """

    @staticmethod
    def deterministics_to_replace(name: str) -> list[str]:
        """Name of the Deterministic variables that are replaced with pm.Flat for out-of-sample.

        This is required for in order to keep out-of-sample predictions use mean of 1.0
        from the training set. Without this, the training and test data will not be
        continuous, showing a jump from the training data to the test data.

        """
        return [f"{name}_f_mean"]

    def create_variable(self, name: str) -> TensorVariable:
        """Create the variable."""
        f = super().create_variable(f"{name}_raw")
        f = pt.softplus(f)

        _, *f_mean_dims = self.dims
        f_mean_name = f"{name}_f_mean"
        f_mean = pm.Deterministic(f_mean_name, f.mean(axis=0), dims=f_mean_dims)

        centered_f = f - f_mean + 1
        return pm.Deterministic(name, centered_f, dims=self.dims)
