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
"""Fourier seasonality transformations.

This modules provides Fourier seasonality transformations for use in
Marketing Mix Models. The Fourier seasonality is a set of sine and cosine
functions that can be used to model periodic patterns in the data.

There are two types of Fourier seasonality transformations available:

- Yearly Fourier: A yearly seasonality with a period of 365 days
- Monthly Fourier: A monthly seasonality with a period of 30 days

Examples
--------
Use yearly fourier seasonality for custom Marketing Mix Model.

.. code-block:: python

    import pandas as pd
    import numpy as np

    import pymc as pm

    from pymc_marketing.mmm import YearlyFourier

    yearly = YearlyFourier(n_order=3)

    dates = pd.date_range("2023-01-01", periods=52, freq="W-MON")

    dayofyear = dates.dayofyear.to_numpy()
    with pm.Model() as model:
        fourier_trend = yearly.apply(dayofyear)

Plot the prior fourier seasonality trend.

For more control over the plot, the `sample_full_period` method can be used to
generate the underlying trend for each sample.

.. code-block:: python

    import matplotlib.pyplot as plt

    prior = yearly.sample_prior()
    curve = yearly.sample_full_period(prior)
    yearly.plot_full_period(curve)
    plt.show()

Change the prior distribution of the fourier seasonality.

.. code-block:: python

    from pymc_marketing.mmm import YearlyFourier
    from pymc_marketing.prior import Prior

    prior = Prior("Normal", mu=0, sigma=0.10)
    yearly = YearlyFourier(n_order=6, prior=prior)

Even make it hierarchical...

.. code-block:: python

    from pymc_marketing.mmm import YearlyFourier
    from pymc_marketing.prior import Prior

    # "fourier" is the default prefix!
    prior = Prior(
        "Laplace",
        mu=Prior("Normal", dims="fourier"),
        b=Prior("HalfNormal", sigma=0.1, dims="fourier"),
        dims=("fourier", "hierarchy"),
    )
    yearly = YearlyFourier(n_order=3, prior=prior)

All the plotting will still work! Just pass any coords.

.. code-block:: python

    import matplotlib.pyplot as plt

    coords = {"hierarchy": ["A", "B", "C"]}
    prior = yearly.sample_prior(coords=coords)
    curve = yearly.sample_full_period(prior)
    yearly.plot_full_period(curve)
    plt.show()

Out of sample predictions with fourier seasonality.

.. code-block:: python

    import pandas as pd
    import pymc as pm

    from pymc_marketing.mmm import YearlyFourier

    periods = 52 * 3
    dates = pd.date_range("2022-01-01", periods=periods, freq="W-MON")

    training_dates = dates[:52 * 2]
    testing_dates = dates[52 * 2:]

    yearly = YearlyFourier(n_order=3)

    coords = {
        "date": training_dates,
    }
    with pm.Model(coords=coords) as model:
        dayofyear = pm.Data(
            "dayofyear",
            training_dates.dayofyear.to_numpy(),
            dims="date",
        )

        trend = pm.Deterministic(
            "trend",
            yearly.apply(dayofyear),
            dims="date",
        )

        idata = pm.sample_prior_predictive().prior

    with model:
        pm.set_data(
            {"dayofyear": testing_dates.dayofyear.to_numpy()},
            coords={"date": testing_dates},
        )

        out_of_sample = pm.sample_posterior_predictive(
            idata,
            var_names=["trend"],
        ).posterior_predictive["trend"]


Use yearly and monthly fourier seasonality together.

By default, the prefix of the fourier seasonality is set to "fourier". However,
the prefix can be changed upon initialization in order to avoid variable name
conflicts.

.. code-block:: python

    from pymc_marketing.mmm import (
        MonthlyFourier,
        YearlyFourier,
    )

    yearly = YearlyFourier(n_order=6, prefix="yearly")
    monthly = MonthlyFourier(n_order=3, prefix="monthly")

    dates = pd.date_range("2023-01-01", periods=52, freq="W-MON")
    dayofyear = dates.dayofyear.to_numpy()

    coords = {
        "date": dates,
    }

    with pm.Model(coords=coords) as model:
        yearly_trend = yearly.apply(dayofyear)
        monthly_trend = monthly.apply(dayofyear)

        trend = pm.Deterministic(
            "trend",
            yearly_trend + monthly_trend,
            dims="date",
        )

    with model:
        prior_samples = pm.sample_prior_predictive().prior

"""

from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from pymc_marketing.constants import DAYS_IN_MONTH, DAYS_IN_YEAR
from pymc_marketing.mmm.plot import (
    plot_hdi,
    plot_samples,
)
from pymc_marketing.prior import Prior, create_dim_handler


def generate_fourier_modes(
    periods: pt.TensorLike,
    n_order: int,
) -> pt.TensorVariable:
    """Create fourier modes for a given period.

    Parameters
    ----------
    periods : pt.TensorLike
        Periods to generate fourier modes for.
    n_order : int
        Number of fourier modes to generate.

    Returns
    -------
    pt.TensorVariable
        Fourier modes.

    """
    multiples = pt.arange(1, n_order + 1)
    x = 2 * pt.pi * periods

    values = x[:, None] * multiples

    return pt.concatenate(
        [
            pt.sin(values),
            pt.cos(values),
        ],
        axis=1,
    )


class FourierBase:
    """Base class for Fourier seasonality transformations.

    Parameters
    ----------
    n_order : int
        Number of fourier modes to use.
    prefix : str, optional
        Alternative prefix for the fourier seasonality, by default None or
        "fourier"
    prior : dict[str, str | dict[str, float]], optional
        Prior distribution for the fourier seasonality beta parameters, by
        default None

    Attributes
    ----------
    days_in_period : float
        Number of days in a period.
    prefix : str
        Name of model coordinates and parameter prefix
    default_prior : dict[str, str | dict[str, float]]
        Default prior distribution for the fourier seasonality
        beta parameters.
    """

    days_in_period: float
    prefix: str = "fourier"

    default_prior = Prior("Laplace", mu=0, b=1)

    def __init__(
        self,
        n_order: int,
        prefix: str | None = None,
        prior: Prior | None = None,
    ) -> None:
        self.n_order = n_order
        self.prefix = prefix or self.prefix
        self.prior = prior or self.default_prior

        if not self.prior.dims:
            self.prior = self.prior.deepcopy()
            self.prior.dims = self.prefix

        if self.prefix not in self.prior.dims:
            raise ValueError(f"Prior distribution must have dimension {self.prefix}")

    @property
    def nodes(self) -> list[str]:
        """Fourier node names for model coordinates."""
        return [
            f"{func}_{i}" for func in ["sin", "cos"] for i in range(1, self.n_order + 1)
        ]

    @property
    def variable_name(self) -> str:
        return f"{self.prefix}_beta"

    def apply(self, dayofyear: pt.TensorLike) -> pt.TensorVariable:
        """Apply fourier seasonality to day of year.

        Must be used within a PyMC model context.

        Parameters
        ----------
        dayofyear : pt.TensorLike
            Day of year.

        Returns
        -------
        pt.TensorVariable
            Fourier seasonality

        """
        periods = dayofyear / self.days_in_period

        model = pm.modelcontext(None)
        model.add_coord(self.prefix, self.nodes)

        beta = self.prior.create_variable(self.variable_name)

        fourier_modes = generate_fourier_modes(periods=periods, n_order=self.n_order)
        if self.prior.dims == (self.prefix,):
            return fourier_modes @ beta

        DUMMY_DIM = "DATE"

        prefix_idx = self.prior.dims.index(self.prefix)
        result_dims = (DUMMY_DIM, *self.prior.dims)
        dim_handler = create_dim_handler(result_dims)

        return (
            dim_handler(fourier_modes, (DUMMY_DIM, self.prefix))
            * dim_handler(beta, self.prior.dims)
        ).sum(axis=prefix_idx + 1)

    def sample_prior(self, coords: dict | None = None, **kwargs) -> xr.Dataset:
        coords = coords or {}
        coords[self.prefix] = self.nodes
        return self.prior.sample_prior(coords=coords, name=self.variable_name)

    def sample_full_period(
        self, parameters: az.InferenceData | xr.Dataset
    ) -> xr.DataArray:
        """Create full period of the fourier seasonality.

        Parameters
        ----------
        parameters : az.InferenceData | xr.Dataset
            Inference data or dataset containing the fourier parameters.
            Can be posterior or prior.

        Returns
        -------
        xr.DataArray
            Full period of the fourier seasonality.

        """
        full_period = np.arange(self.days_in_period + 1)
        coords = {
            "day": full_period,
        }
        for key, values in parameters[self.variable_name].coords.items():
            if key in {"chain", "draw", self.prefix}:
                continue
            coords[key] = values.to_numpy()

        with pm.Model(coords=coords):
            name = f"{self.prefix}_trend"
            pm.Deterministic(
                name,
                self.apply(dayofyear=full_period),
                dims=tuple(coords.keys()),
            )

            return pm.sample_posterior_predictive(
                parameters,
                var_names=[name],
            ).posterior_predictive[name]

    def plot_full_period(
        self,
        curve: xr.DataArray,
        subplot_kwargs: dict | None = None,
        sample_kwargs: dict | None = None,
        hdi_kwargs: dict | None = None,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
        hdi_kwargs = hdi_kwargs or {}
        sample_kwargs = sample_kwargs or {}

        if "subplot_kwargs" not in hdi_kwargs:
            hdi_kwargs["subplot_kwargs"] = subplot_kwargs

        fig, axes = self.plot_full_period_hdi(curve, **hdi_kwargs)
        fig, axes = self.plot_full_period_samples(curve, axes=axes, **sample_kwargs)

        return fig, axes

    def plot_full_period_hdi(
        self,
        samples: xr.DataArray,
        hdi_kwargs: dict | None = None,
        axes: npt.NDArray[plt.Axes] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
        """Plot full period of the fourier seasonality.

        Parameters
        ----------
        parameters : az.InferenceData | xr.Dataset
            Inference data or dataset containing the fourier parameters.
            Can be posterior or prior.
        hdi_prob : float, optional
            HDI probability, by default 0.95
        ax : plt.Axes, optional
            Matplotlib axes, by default None
        **kwargs
            Additional keyword arguments for `fill_between`.

        Returns
        -------
        plt.Axes
            Modified matplotlib axes.

        """

        hdi_kwargs = hdi_kwargs or {}
        conf = az.hdi(samples, **hdi_kwargs)[f"{self.prefix}_trend"]

        return plot_hdi(
            conf,
            non_grid_names={"day", "hdi"},
            axes=axes,
            subplot_kwargs=subplot_kwargs,
            plot_kwargs=plot_kwargs,
        )

    def plot_full_period_samples(
        self,
        samples: xr.DataArray,
        rng=None,
        axes: npt.NDArray[plt.Axes] | None = None,
        n: int = 10,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
        return plot_samples(
            samples,
            non_grid_names={"chain", "draw", "day"},
            n=n,
            rng=rng,
            axes=axes,
            subplot_kwargs=subplot_kwargs,
            plot_kwargs=plot_kwargs,
        )


class YearlyFourier(FourierBase):
    """Yearly fourier seasonality."""

    days_in_period = DAYS_IN_YEAR


class MonthlyFourier(FourierBase):
    """Monthly fourier seasonality."""

    days_in_period = DAYS_IN_MONTH
