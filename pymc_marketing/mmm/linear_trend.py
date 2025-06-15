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
"""Linear trend using change points.

Examples
--------
Define a linear trend with 8 changepoints:

.. code-block:: python

    from pymc_marketing.mmm import LinearTrend

    trend = LinearTrend(n_changepoints=8)

Sample the prior for the trend parameters and curve:

.. code-block:: python

    import numpy as np

    seed = sum(map(ord, "Linear Trend"))
    rng = np.random.default_rng(seed)

    prior = trend.sample_prior(random_seed=rng)
    curve = trend.sample_curve(prior)

Plot the curve samples:

.. code-block:: python

    _, axes = trend.plot_curve(curve, random_seed=rng)
    ax = axes[0]
    ax.set(
        xlabel="Time",
        ylabel="Trend",
        title=f"Linear Trend with {trend.n_changepoints} Change Points",
    )

.. image:: /_static/linear-trend-prior.png
    :alt: LinearTrend prior

"""

from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, InstanceOf, model_validator
from pymc.distributions.shape_utils import Dims
from pytensor.tensor.variable import TensorVariable
from typing_extensions import Self

from pymc_marketing.plot import SelToString, plot_curve
from pymc_marketing.prior import Prior, create_dim_handler


class LinearTrend(BaseModel):
    r"""LinearTrend class.

    Linear trend component using change points. The trend is defined as:

    .. math::

        f(t) = k + \sum_{m=0}^{M-1} \delta_m I(t > s_m)

    where:

    - :math:`t \ge 0`,
    - :math:`k` is the base intercept,
    - :math:`\delta_m` is the change in the trend at change point :math:`m`,
    - :math:`I` is the indicator function,
    - :math:`s_m` is the change point.

    The change points are defined as:

    .. math::

            s_m = \frac{m}{M-1} T, 0 \le m \le M-1

    where :math:`M` is the number of change points (:math:`M>1`)
    and :math:`T` is the time of the last observed data point.

    The priors for the trend parameters are:

    - :math:`k \sim \text{Normal}(0, 0.05)`
    - :math:`\delta_m \sim \text{Laplace}(0, 0.25)`

    Parameters
    ----------
    priors : dict[str, Prior], optional
        Dictionary with the priors for the trend parameters. The
        dictionary must have 'delta' key. If `include_intercept` is
        True, the 'k' key is also required. By default None, or
        the default priors.
    dims : Dims, optional
        Dimensions of the parameters, by default None or empty.
    n_changepoints : int, optional
        Number of changepoints, by default 10.
    include_intercept : bool, optional
        Include an intercept in the trend, by default False

    Examples
    --------
    Linear trend with 10 changepoints:

    .. code-block:: python

        from pymc_marketing.mmm import LinearTrend

        trend = LinearTrend(n_changepoints=10)

    Use the trend in a model:

    .. code-block:: python

        import pymc as pm
        import numpy as np

        import pandas as pd

        n_years = 3
        n_dates = 52 * n_years
        first_date = "2020-01-01"
        dates = pd.date_range(first_date, periods=n_dates, freq="W-MON")
        dayofyear = dates.dayofyear.to_numpy()
        t = (dates - dates[0]).days.to_numpy()
        t = t / 365.25

        coords = {"date": dates}
        with pm.Model(coords=coords) as model:
            intercept = pm.Normal("intercept", mu=0, sigma=1)
            mu = intercept + trend.apply(t)

            sigma = pm.Gamma("sigma", mu=0.1, sigma=0.025)

            pm.Normal("obs", mu=mu, sigma=sigma, dims="date")

    Hierarchical LinearTrend via hierarchical prior:

    .. code-block:: python

        from pymc_marketing.prior import Prior

        hierarchical_delta = Prior(
            "Laplace",
            mu=Prior("Normal", dims="changepoint"),
            b=Prior("HalfNormal", dims="changepoint"),
            dims=("changepoint", "geo"),
        )
        priors = dict(delta=hierarchical_delta)

        hierarchical_trend = LinearTrend(
            priors=priors,
            n_changepoints=10,
            dims="geo",
        )

    Sample the hierarchical trend:

    .. code-block:: python

        seed = sum(map(ord, "Hierarchical LinearTrend"))
        rng = np.random.default_rng(seed)

        coords = {"geo": ["A", "B"]}
        prior = hierarchical_trend.sample_prior(
            coords=coords,
            random_seed=rng,
        )
        curve = hierarchical_trend.sample_curve(prior)

    Plot the curve HDI and samples:

    .. code-block:: python

        fig, axes = hierarchical_trend.plot_curve(
            curve,
            n_samples=3,
            random_seed=rng,
        )
        fig.suptitle("Hierarchical Linear Trend")
        axes[0].set(ylabel="Trend", xlabel="Time")
        axes[1].set(xlabel="Time")

    .. image:: /_static/hierarchical-linear-trend-prior.png
        :alt: Hierarchical LinearTrend prior

    References
    ----------
    Adapted from MBrouns/timeseers package:
        https://github.com/MBrouns/timeseers/blob/master/src/timeseers/linear_trend.py

    """

    priors: InstanceOf[dict[str, Prior]] = Field(
        None,
        description="Priors for the trend parameters.",
    )
    dims: tuple[str, ...] | InstanceOf[Dims] | str | None = Field(
        None,
        description="The additional dimensions for the trend.",
    )
    n_changepoints: int = Field(
        10,
        description="Number of changepoints.",
        ge=1,
    )
    include_intercept: bool = Field(
        False,
        description="Include an intercept in the trend.",
    )
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _dims_is_tuple(self) -> Self:
        dims = self.dims
        if isinstance(dims, str):
            self.dims = (dims,)

        self.dims: tuple[str] = self.dims or ()

        return self

    @model_validator(mode="after")
    def _priors_are_set(self) -> Self:
        self.priors = self.priors or self.default_priors.copy()

        return self

    @model_validator(mode="after")
    def _check_parameters(self) -> Self:
        required_parameters = set(self.default_priors.keys())
        if set(self.priors.keys()) > required_parameters:
            msg = f"Invalid priors. The required parameters are {required_parameters}."
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def _check_dims_are_subsets(self) -> Self:
        allowed_dims = {"changepoint"}.union(cast(Dims, self.dims))

        if not all(set(prior.dims) <= allowed_dims for prior in self.priors.values()):
            msg = "Invalid dimensions in the priors."
            raise ValueError(msg)

        return self

    @property
    def default_priors(self) -> dict[str, Prior]:
        """Default priors for the trend parameters.

        Returns
        -------
        dict[str, Prior]
            Dictionary with the default priors.

        """
        priors = {
            "delta": Prior(
                "Laplace",
                mu=0,
                b=0.25,
                dims="changepoint",
            ),
        }
        if self.include_intercept:
            priors["k"] = Prior("Normal", mu=0, sigma=0.05)

        return priors

    @property
    def non_broadcastable_dims(self) -> tuple[str, ...]:
        """Get the dimensions of the trend that are not just broadcastable.

        Returns
        -------
        tuple[str, ...]
            Tuple with the dimensions of the trend.

        """
        dims = set()
        for prior in self.priors.values():
            dims.update(prior.dims)

        dims = dims.difference({"changepoint"})

        return tuple(dim for dim in cast(tuple[str, ...], self.dims) if dim in dims)

    def apply(self, t: pt.TensorLike) -> TensorVariable:
        """Create the linear trend for the given x values.

        Parameters
        ----------
        t : pt.TensorLike
            1D array of strictly increasing time values for the trend starting from 0.

        Returns
        -------
        pt.TensorVariable
            TensorVariable with the trend values.

        """
        dims = cast(Dims, self.dims)
        model = pm.modelcontext(None)
        model.add_coord("changepoint", range(self.n_changepoints))
        DUMMY_DIM = "DATE"
        out_dims = (DUMMY_DIM, "changepoint", *dims)
        dim_handler = create_dim_handler(desired_dims=out_dims)

        # (changepoints, )
        s = pt.linspace(0, pt.max(t).eval(), self.n_changepoints)
        s.type.shape = (self.n_changepoints,)
        s = dim_handler(
            s,
            ("changepoint",),
        )
        # (dates, changepoints)
        A = (dim_handler(t, (DUMMY_DIM,)) > s) * 1.0

        delta_dist = self.priors["delta"]
        delta = dim_handler(
            delta_dist.create_variable("delta"),
            delta_dist.dims,
        )

        k_dim_handler = create_dim_handler((DUMMY_DIM, *dims))

        first = (A * delta).sum(axis=1) * k_dim_handler(t, (DUMMY_DIM,))

        if self.include_intercept:
            # (additional_groups)
            k_dist = self.priors["k"]
            k = k_dim_handler(
                k_dist.create_variable("k"),
                k_dist.dims,
            )
            first += k

        gamma = -s * delta
        second = (A * gamma).sum(axis=1)

        return first + second

    def sample_prior(
        self,
        coords=None,
        **sample_prior_predictive_kwargs,
    ) -> xr.Dataset:
        """Sample the prior for the parameters used in the trend.

        Parameters
        ----------
        coords : dict, optional
            Coordinates in the priors, by default includes the changepoints.
        sample_prior_predictive_kwargs : dict, optional
            Keyword arguments for the `pm.sample_prior_predictive` function.

        Returns
        -------
        xr.Dataset
            Dataset with the prior samples.

        """
        coords = coords or {}
        coords["changepoint"] = range(self.n_changepoints)
        with pm.Model(coords=coords):
            for key, param in self.priors.items():
                param.create_variable(key)

            return pm.sample_prior_predictive(**sample_prior_predictive_kwargs).prior

    def sample_curve(
        self,
        parameters: xr.Dataset,
        max_value: float = 1.0,
    ) -> xr.DataArray:
        """Sample the curve given parameters.

        Parameters
        ----------
        parameters : xr.Dataset
            Dataset with the parameters to condition on. Would be
            either the prior or the posterior.

        Returns
        -------
        xr.DataArray
            DataArray with the curve samples.

        """
        t = np.linspace(0, max_value, 100)
        coords: dict[str, Any] = {"t": t}
        for name in self.priors.keys():
            for key, values in parameters[name].coords.items():
                if key in {"chain", "draw"}:
                    continue

                coords[key] = values.to_numpy()

        with pm.Model(coords=coords):
            name = "trend"
            pm.Deterministic(
                name,
                self.apply(t),
                dims=("t", *cast(Dims, self.dims)),
            )

            return pm.sample_posterior_predictive(
                parameters,
                var_names=[name],
            ).posterior_predictive[name]

    def plot_curve(
        self,
        curve: xr.DataArray,
        n_samples: int = 10,
        hdi_probs: float | list[float] | None = None,
        random_seed: np.random.Generator | None = None,
        subplot_kwargs: dict | None = None,
        sample_kwargs: dict | None = None,
        hdi_kwargs: dict | None = None,
        include_changepoints: bool = True,
        axes: npt.NDArray[Axes] | None = None,
        same_axes: bool = False,
        colors: Iterable[str] | None = None,
        legend: bool | None = None,
        sel_to_string: SelToString | None = None,
    ) -> tuple[Figure, npt.NDArray[Axes]]:
        """Plot the curve samples from the trend.

        Parameters
        ----------
        curve : xr.DataArray
            DataArray with the curve samples.
        n_samples : int, optional
            Number of samples
        hdi_probs : float | list[float], optional
            HDI probabilities. Defaults to None which uses arviz default for
            stats.ci_prob which is 94%
        random_seed : int | random number generator, optional
            Random number generator. Defaults to None
        subplot_kwargs : dict, optional
            Keyword arguments for the subplots, by default None.
        sample_kwargs : dict, optional
            Keyword arguments for the samples, by default None.
        hdi_kwargs : dict, optional
            Keyword arguments for the HDI, by default None.
        include_changepoints : bool, optional
            Include the change points in the plot, by default True.
        axes : npt.NDArray[plt.Axes], optional
            Axes to plot the curve, by default None.
        same_axes : bool, optional
            Use the same axes for the samples, by default False.
        colors : Iterable[str], optional
            Colors for the samples, by default None.
        legend : bool, optional
            Include a legend in the plot, by default None.
        sel_to_string : SelToString, optional
            Function to convert the selection to a string, by default None.

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]
            Tuple with the figure and the axes.

        """
        fig, axes = plot_curve(
            curve,
            {"t"},
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

        if not include_changepoints:
            return fig, axes

        max_value = curve.coords["t"].max().item()

        for ax in np.ravel(axes):
            for i in range(0, self.n_changepoints):
                ax.axvline(
                    max_value * i / (self.n_changepoints - 1),
                    color="gray",
                    linestyle="--",
                )

        return fig, axes
