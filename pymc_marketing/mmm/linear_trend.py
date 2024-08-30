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
"""Linear trend using change points.

Examples
--------
Define a linear trend with 8 changepoints:

.. code-block:: python

    from pymc_marketing.mmm import LinearTrend

    trend = LinearTrend(n_changepoints=8)

Sample the prior for the trend parameters and curve:

.. code-block:: python

    prior = trend.sample_prior()
    curve = trend.sample_curve(prior)

Plot the curve samples:

.. code-block:: python

    import numpy as np

    seed = sum(map(ord, "Linear Trend"))
    rng = np.random.default_rng(seed)

    _, axes = trend.plot_curve(curve, sample_kwargs={"rng": rng})
    ax = axes[0]
    ax.set(
        xlabel="Time",
        ylabel="Trend",
        title=f"Linear Trend with {trend.n_changepoints} Change Points",
    )

.. image:: /_static/linear-trend-prior.png
    :alt: LinearTrend prior

"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc.distributions.shape_utils import Dims

from pymc_marketing.mmm.plot import plot_curve
from pymc_marketing.prior import Prior, create_dim_handler


class LinearTrend:
    r"""LinearTrend class.

    Linear trend component using change points. The trend is defined as:

    .. math::

        f(t) = k + \sum_{m=1}^{M} \delta_m I(t > s_m)

    where:

    - :math:`k` is the base trend,
    - :math:`\delta_m` is the change in the trend at change point :math:`m`,
    - :math:`I` is the indicator function,
    - :math:`s_m` is the change point.

    The change points are defined as:

    .. math::

            s_m = \frac{m}{M+1} \max(t)

        where :math:`M` is the number of change points.

    The priors for the trend parameters are:

    - :math:`k \sim \text{Normal}(0, 0.05)`
    - :math:`\delta_m \sim \text{Laplace}(0, 0.25)`

    Adapted from MBrouns/timeseers package:
    https://github.com/MBrouns/timeseers/blob/master/src/timeseers/linear_trend.py

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

        coords = {
            "date": dates,
        }
        with pm.Model(coords=coords) as model:
            intercept = pm.Normal("intercept", mu=0, sigma=1)
            mu = intercept + trend.apply(t)

            sigma = pm.Gamma("sigma", mu=0.1, sigma=0.025)

            pm.Normal("obs", mu=mu, sigma=sigma, dims="date")

    """

    def __init__(
        self,
        priors: dict[str, Prior] | None = None,
        dims: Dims | None = None,
        n_changepoints: int = 10,
        include_intercept: bool = False,
    ) -> None:
        self.n_changepoints = n_changepoints
        self.include_intercept = include_intercept
        self.priors: dict[str, Prior] = priors or self.default_priors.copy()
        self.dims: Dims = dims or ()

        self._check_parameters()

    def _check_parameters(self) -> None:
        required_parameters = set(self.default_priors.keys())
        if set(self.priors.keys()) < required_parameters:
            msg = f"Invalid priors. The required parameters are {required_parameters}."
            raise ValueError(msg)

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

    def apply(self, t: pt.TensorLike) -> pt.TensorVariable:
        """Create the linear trend for the given x values.

        Parameters
        ----------
        t : pt.TensorLike
            Input values for the trend.

        Returns
        -------
        pt.TensorVariable
            TensorVariable with the trend values.

        """
        model = pm.modelcontext(None)
        model.add_coord("changepoint", range(self.n_changepoints))
        DUMMY_DIM = "DATE"
        out_dims = (DUMMY_DIM, "changepoint", *self.dims)
        dim_handler = create_dim_handler(desired_dims=out_dims)

        # (changepoints, )
        s = pt.linspace(0, pt.max(t), self.n_changepoints)
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

        k_dim_handler = create_dim_handler((DUMMY_DIM, *self.dims))

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

    def sample_prior(self, coords=None) -> xr.Dataset:
        """Sample the prior for the parameters used in the trend.

        Parameters
        ----------
        coords : dict, optional
            Coordinates in the priors, by default includes the changepoints.

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

            return pm.sample_prior_predictive().prior

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
        coords: dict[str, Any] = {
            "t": t,
        }
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
                dims=("t", *self.dims),
            )

            return pm.sample_posterior_predictive(
                parameters,
                var_names=[name],
            ).posterior_predictive[name]

    def plot_curve(
        self,
        curve: xr.DataArray,
        subplot_kwargs: dict | None = None,
        sample_kwargs: dict | None = None,
        hdi_kwargs: dict | None = None,
        include_change_points: bool = True,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
        """Plot the curve samples from the trend.

        Parameters
        ----------
        curve : xr.DataArray
            DataArray with the curve samples.
        subplot_kwargs : dict, optional
            Keyword arguments for the subplots, by default None.
        sample_kwargs : dict, optional
            Keyword arguments for the samples, by default None.
        hdi_kwargs : dict, optional
            Keyword arguments for the HDI, by default None.
        include_change_points : bool, optional
            Include the change points in the plot, by default True.

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]
            Tuple with the figure and the axes.

        """
        fig, axes = plot_curve(
            curve,
            {"t"},
            subplot_kwargs=subplot_kwargs,
            sample_kwargs=sample_kwargs,
            hdi_kwargs=hdi_kwargs,
        )

        if not include_change_points:
            return fig, axes

        max_value = curve.coords["t"].max().item()

        for ax in np.ravel(axes):
            for i in range(1, self.n_changepoints + 1):
                # Need to add 1 to the number of changepoints
                ax.axvline(
                    max_value * i / (self.n_changepoints + 1),
                    color="gray",
                    linestyle="--",
                )

        return fig, axes
