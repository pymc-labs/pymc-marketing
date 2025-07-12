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
"""Fourier seasonality transformations.

This modules provides Fourier seasonality transformations for use in
Marketing Mix Models. The Fourier seasonality is a set of sine and cosine
functions that can be used to model periodic patterns in the data.

There are two types of Fourier seasonality transformations available:

- Yearly Fourier: A yearly seasonality with a period of 365.25 days
- Monthly Fourier: A monthly seasonality with a period of 365.25 / 12 days
- Weekly Fourier: A weekly seasonality with a period of 7 days

.. plot::
    :context: close-figs

    import matplotlib.pyplot as plt
    import numpy as np
    import arviz as az
    from pymc_marketing.mmm import YearlyFourier
    from pymc_marketing.prior import Prior

    plt.style.use('arviz-darkgrid')

    prior = Prior(
        "Normal",
        mu=[0, 0, -1, 0],
        sigma=Prior("Gamma", mu=0.10, sigma=0.1, dims="fourier"),
        dims=("hierarchy", "fourier"),
    )
    yearly = YearlyFourier(n_order=2, prior=prior)
    coords = {"hierarchy": ["A", "B"]}
    prior = yearly.sample_prior(coords=coords)
    curve = yearly.sample_curve(prior)
    fig, _ = yearly.plot_curve(curve, subplot_kwargs={"ncols": 1})
    fig.suptitle("Yearly Fourier Seasonality")
    plt.show()

Examples
--------
Use yearly fourier seasonality for custom Marketing Mix Model.

.. code-block:: python

    import pandas as pd
    import pymc as pm

    from pymc_marketing.mmm import YearlyFourier

    yearly = YearlyFourier(n_order=3)

    dates = pd.date_range("2023-01-01", periods=52, freq="W-MON")

    dayofyear = dates.dayofyear.to_numpy()

    with pm.Model() as model:
        fourier_trend = yearly.apply(dayofyear)

Plot the prior fourier seasonality trend.

.. code-block:: python

    import matplotlib.pyplot as plt

    prior = yearly.sample_prior()
    curve = yearly.sample_curve(prior)
    yearly.plot_curve(curve)
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
    curve = yearly.sample_curve(prior)
    yearly.plot_curve(curve)
    plt.show()

Out of sample predictions with fourier seasonality by changing the day of year
used in the model.

.. code-block:: python

    import pandas as pd
    import pymc as pm

    from pymc_marketing.mmm import YearlyFourier

    periods = 52 * 3
    dates = pd.date_range("2022-01-01", periods=periods, freq="W-MON")

    training_dates = dates[: 52 * 2]
    testing_dates = dates[52 * 2 :]

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

    import pandas as pd
    import pymc as pm

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

import datetime
from abc import abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    field_serializer,
    model_validator,
)
from typing_extensions import Self

from pymc_marketing.constants import DAYS_IN_MONTH, DAYS_IN_WEEK, DAYS_IN_YEAR
from pymc_marketing.deserialize import deserialize, register_deserialization
from pymc_marketing.plot import SelToString, plot_curve, plot_hdi, plot_samples
from pymc_marketing.prior import Prior, VariableFactory, create_dim_handler

X_NAME: str = "day"
NON_GRID_NAMES: frozenset[str] = frozenset({X_NAME})


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


class FourierBase(BaseModel):
    """Base class for Fourier seasonality transformations.

    Parameters
    ----------
    n_order : int
        Number of fourier modes to use.
    days_in_period : float
        Number of days in a period.
    prefix : str, optional
        Alternative prefix for the fourier seasonality, by default None or
        "fourier"
    prior : Prior | VariableFactory, optional
        Prior distribution or VariableFactory for the fourier seasonality beta parameters, by
        default `Prior("Laplace", mu=0, b=1)`
    variable_name : str, optional
        Name of the variable that multiplies the fourier modes. By default None,
        in which case it is set to the `{prefix}_beta`.

    """

    n_order: int = Field(..., gt=0)
    days_in_period: float = Field(..., gt=0)
    prefix: str = Field("fourier")
    prior: InstanceOf[Prior] | InstanceOf[VariableFactory] = Field(
        Prior("Laplace", mu=0, b=1)
    )
    variable_name: str | None = Field(None)
    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        """Model post initialization for a Pydantic model."""
        if self.variable_name is None:
            self.variable_name = f"{self.prefix}_beta"

        if not self.prior.dims and isinstance(self.prior, Prior):
            self.prior = self.prior.deepcopy()
            self.prior.dims = self.prefix
        elif not self.prior.dims:
            self.prior.dims = self.prefix

    @model_validator(mode="after")
    def _check_variable_name(self) -> Self:
        if self.variable_name == self.prefix:
            raise ValueError("Variable name cannot be the same as the prefix")
        return self

    @model_validator(mode="after")
    def _check_prior_has_right_dimensions(self) -> Self:
        if self.prefix not in self.prior.dims:
            raise ValueError(f"Prior distribution must have dimension {self.prefix}")
        return self

    @field_serializer("prior", when_used="json")
    def serialize_prior(prior: Any) -> dict[str, Any]:
        """Serialize the prior distribution.

        Parameters
        ----------
        prior : VariableFactory | Prior
            The prior distribution to serialize.

        Returns
        -------
        dict[str, Any]
            The serialized prior distribution.

        """
        return prior.to_dict()

    @property
    def nodes(self) -> list[str]:
        """Fourier node names for model coordinates."""
        return [
            f"{func}_{i}" for func in ["sin", "cos"] for i in range(1, self.n_order + 1)
        ]

    def get_default_start_date(
        self,
        start_date: str | datetime.datetime | None = None,
    ) -> str | datetime.datetime:
        """Get the start date for the Fourier curve.

        If `start_date` is provided, validate its type.
        Otherwise, provide the default start date based on the subclass implementation.

        Parameters
        ----------
        start_date : str or datetime.datetime, optional
            Provided start date. Can be a string or a datetime object.

        Returns
        -------
        str or datetime.datetime
            The validated start date.

        Raises
        ------
        TypeError
            If `start_date` is neither a string nor a datetime object.
        """
        if start_date is None:
            return self._get_default_start_date()
        elif isinstance(start_date, str) | isinstance(start_date, datetime.datetime):
            return start_date
        else:
            raise TypeError(
                "start_date must be a datetime.datetime object, a string, or None"
            )

    @abstractmethod
    def _get_default_start_date(self) -> datetime.datetime:
        """Provide the default start date. Must be implemented by subclasses.

        Returns
        -------
        datetime.datetime
            The default start date.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _get_days_in_period(self, dates: pd.DatetimeIndex) -> pd.Index:
        """Return the relevant day within the characteristic periodicity.

        Returns
        -------
        int or float
            The relevant period within the characteristic periodicity
        """
        pass

    def apply(
        self,
        dayofperiod: pt.TensorLike,
        result_callback: Callable[[pt.TensorVariable], None] | None = None,
    ) -> pt.TensorVariable:
        """Apply fourier seasonality to day of year.

        Must be used within a PyMC model context.

        Parameters
        ----------
        dayofperiod : pt.TensorLike
            Day of year or weekday
        result_callback : Callable[[pt.TensorVariable], None], optional
            Callback function to apply to the result, by default None

        Returns
        -------
        pt.TensorVariable
            Fourier seasonality

        Examples
        --------
        Save off the result before summing through the prefix dimension.

        .. code-block:: python

            import pandas as pd

            import pymc as pm

            from pymc_marketing.mmm import YearlyFourier

            fourier = YearlyFourier(n_order=3)


            def callback(result):
                pm.Deterministic("fourier_trend", result, dims=("date", "fourier"))


            dates = pd.date_range("2023-01-01", periods=52, freq="W-MON")

            coords = {
                "date": dates,
            }
            with pm.Model(coords=coords) as model:
                dayofyear = dates.dayofyear.to_numpy()
                fourier.apply(dayofyear, result_callback=callback)

        """
        periods = dayofperiod / self.days_in_period

        model = pm.modelcontext(None)
        model.add_coord(self.prefix, self.nodes)

        beta = self.prior.create_variable(self.variable_name)

        fourier_modes = generate_fourier_modes(periods=periods, n_order=self.n_order)

        DUMMY_DIM = "DATE"

        prefix_idx = self.prior.dims.index(self.prefix)
        result_dims = (DUMMY_DIM, *self.prior.dims)
        dim_handler = create_dim_handler(result_dims)

        result = dim_handler(fourier_modes, (DUMMY_DIM, self.prefix)) * dim_handler(
            beta, self.prior.dims
        )
        if result_callback is not None:
            result_callback(result)

        return result.sum(axis=prefix_idx + 1)

    def sample_prior(self, coords: dict | None = None, **kwargs) -> xr.Dataset:
        """Sample the prior distributions.

        Parameters
        ----------
        coords : dict, optional
            Coordinates for the prior distribution, by default None
        kwargs
            Additional keywords for sample_prior_predictive

        Returns
        -------
        xr.Dataset
            Prior distribution.

        """
        coords = coords or {}
        coords[self.prefix] = self.nodes
        return self.prior.sample_prior(coords=coords, name=self.variable_name, **kwargs)

    def sample_curve(
        self,
        parameters: az.InferenceData | xr.Dataset,
        use_dates: bool = False,
        start_date: str | datetime.datetime | None = None,
    ) -> xr.DataArray:
        """Create full period of the Fourier seasonality.

        Parameters
        ----------
        parameters : az.InferenceData | xr.Dataset
            Inference data or dataset containing the Fourier parameters.
            Can be posterior or prior.
        use_dates : bool, optional
            If True, use datetime coordinates for the x-axis. Defaults to False.
        start_date : datetime.datetime, optional
            Starting date for the Fourier curve. If not provided and use_dates is True,
            it will be derived from the current year or month. Defaults to None.

        Returns
        -------
        xr.DataArray
            Full period of the Fourier seasonality.

        """
        full_period = np.arange(self.days_in_period + 1)

        coords = {}
        if use_dates:
            start_date = self.get_default_start_date(start_date=start_date)
            date_range = pd.date_range(
                start=start_date,
                periods=int(np.ceil(self.days_in_period) + 1),
                freq="D",
            )
            coords["date"] = date_range.to_numpy()
            dayofperiod = self._get_days_in_period(date_range).to_numpy()

        else:
            coords["day"] = full_period
            dayofperiod = full_period

        for key, values in parameters[self.variable_name].coords.items():
            if key in {"chain", "draw", self.prefix}:
                continue
            coords[key] = values.to_numpy()

        with pm.Model(coords=coords):
            name = f"{self.prefix}_trend"
            pm.Deterministic(
                name,
                self.apply(dayofperiod=dayofperiod),
                dims=tuple(coords.keys()),
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
        axes: npt.NDArray[plt.Axes] | None = None,
        same_axes: bool = False,
        colors: Iterable[str] | None = None,
        legend: bool | None = None,
        sel_to_string: SelToString | None = None,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
        """Plot the seasonality for one full period.

        Parameters
        ----------
        curve : xr.DataArray
            Sampled full period of the fourier seasonality.
        n_samples : int, optional
            Number of samples
        hdi_probs : float | list[float], optional
            HDI probabilities. Defaults to None which uses arviz default for
            stats.ci_prob which is 94%
        random_seed : int | random number generator, optional
            Random number generator. Defaults to None
        subplot_kwargs : dict, optional
            Keyword arguments for the subplot, by default None
        sample_kwargs : dict, optional
            Keyword arguments for the plot_full_period_samples method, by default None
        hdi_kwargs : dict, optional
            Keyword arguments for the plot_full_period_hdi method, by default None
        axes : npt.NDArray[plt.Axes], optional
            Matplotlib axes, by default None
        same_axes : bool, optional
            Use the same axes for all plots, by default False
        colors : Iterable[str], optional
            Colors for the different plots, by default None
        legend : bool, optional
            Show the legend, by default None
        sel_to_string : SelToString, optional
            Function to convert the selection to a string, by default None

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]
            Matplotlib figure and axes.

        """
        if "date" in curve.coords:
            x_coord_name = "date"
        elif "day" in curve.coords:
            x_coord_name = "day"
        else:
            raise ValueError("Curve must have either 'day' or 'date' as a coordinate")

        return plot_curve(
            curve,
            non_grid_names={x_coord_name},
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

    def plot_curve_hdi(
        self,
        curve: xr.DataArray,
        hdi_kwargs: dict | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        axes: npt.NDArray[plt.Axes] | None = None,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
        """Plot full period of the fourier seasonality.

        Parameters
        ----------
        curve : xr.DataArray
            The curve to plot.
        hdi_kwargs : dict, optional
            Keyword arguments for the az.hdi function. Defaults to None.
        plot_kwargs : dict, optional
            Keyword arguments for the fill_between function. Defaults to None.
        subplot_kwargs : dict, optional
            Keyword arguments for plt.subplots
        axes : npt.NDArray[plt.Axes], optional
            The exact axes to plot on. Overrides any subplot_kwargs

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]

        """
        if "date" in curve.coords:
            x_coord_name = "date"
        elif "day" in curve.coords:
            x_coord_name = "day"
        else:
            raise ValueError("Curve must have either 'day' or 'date' as a coordinate")

        return plot_hdi(
            curve,
            non_grid_names={x_coord_name},
            hdi_kwargs=hdi_kwargs,
            subplot_kwargs=subplot_kwargs,
            plot_kwargs=plot_kwargs,
            axes=axes,
        )

    def plot_curve_samples(
        self,
        curve: xr.DataArray,
        n: int = 10,
        rng: np.random.Generator | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        axes: npt.NDArray[plt.Axes] | None = None,
    ) -> tuple[plt.Figure, npt.NDArray[plt.Axes]]:
        """Plot samples from the curve.

        Parameters
        ----------
        curve : xr.DataArray
            Samples from the curve.
        n : int, optional
            Number of samples to plot, by default 10
        rng : np.random.Generator, optional
            Random number generator, by default None
        plot_kwargs : dict, optional
            Keyword arguments for the plot function, by default None
        subplot_kwargs : dict, optional
            Keyword arguments for the subplot, by default None
        axes : npt.NDArray[plt.Axes], optional
            Matplotlib axes, by default None

        Returns
        -------
        tuple[plt.Figure, npt.NDArray[plt.Axes]]
            Matplotlib figure and axes.

        """
        if "date" in curve.coords:
            x_coord_name = "date"
        elif "day" in curve.coords:
            x_coord_name = "day"
        else:
            raise ValueError("Curve must have either 'day' or 'date' as a coordinate")

        return plot_samples(
            curve,
            non_grid_names={x_coord_name},
            n=n,
            rng=rng,
            axes=axes,
            subplot_kwargs=subplot_kwargs,
            plot_kwargs=plot_kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the Fourier seasonality.

        Returns
        -------
        dict[str, Any]
            Serialized Fourier seasonality

        """
        return {
            "class": self.__class__.__name__,
            "data": self.model_dump(mode="json"),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize the Fourier seasonality.

        Parameters
        ----------
        data : dict[str, Any]
            Serialized Fourier seasonality

        Returns
        -------
        FourierBase
            Deserialized Fourier seasonality

        """
        data = data["data"]
        data["prior"] = deserialize(data["prior"])
        return cls(**data)


class YearlyFourier(FourierBase):
    """Yearly fourier seasonality.

    .. plot::
        :context: close-figs

        import arviz as az
        import matplotlib.pyplot as plt
        import numpy as np

        from pymc_marketing.mmm import YearlyFourier
        from pymc_marketing.prior import Prior

        az.style.use("arviz-white")

        seed = sum(map(ord, "Yearly"))
        rng = np.random.default_rng(seed)

        mu = np.array([0, 0, -1, 0])
        b = 0.15
        dist = Prior("Laplace", mu=mu, b=b, dims="fourier")
        yearly = YearlyFourier(n_order=2, prior=dist)
        prior = yearly.sample_prior(random_seed=rng)
        curve = yearly.sample_curve(prior)

        _, axes = yearly.plot_curve(curve)
        axes[0].set(title="Yearly Fourier Seasonality")
        plt.show()

    n_order : int
        Number of fourier modes to use.
    prefix : str, optional
        Alternative prefix for the fourier seasonality, by default None or
        "fourier"
    prior : Prior | VariableFactory, optional
        Prior distribution or VariableFactory for the fourier seasonality beta parameters, by
        default `Prior("Laplace", mu=0, b=1)`
    name : str, optional
        Name of the variable that multiplies the fourier modes, by default None
    variable_name : str, optional
        Name of the variable that multiplies the fourier modes, by default None

    """

    days_in_period: float = DAYS_IN_YEAR

    def _get_default_start_date(self) -> datetime.datetime:
        """Get the default start date for yearly seasonality.

        Returns January 1st of the current year.

        """
        current_year = datetime.datetime.now().year
        return datetime.datetime(year=current_year, month=1, day=1)

    def _get_days_in_period(self, dates: pd.DatetimeIndex) -> pd.Index:
        """Return the dayofyear within the yearly periodicity.

        Returns
        -------
        int or float
            The relevant period within the characteristic periodicity
        """
        return dates.dayofyear


class MonthlyFourier(FourierBase):
    """Monthly fourier seasonality.

    .. plot::
        :context: close-figs

        import arviz as az
        import matplotlib.pyplot as plt
        import numpy as np

        from pymc_marketing.mmm import MonthlyFourier
        from pymc_marketing.prior import Prior

        az.style.use("arviz-white")

        seed = sum(map(ord, "Monthly"))
        rng = np.random.default_rng(seed)

        mu = np.array([0, 0, 0.5, 0])
        b = 0.075
        dist = Prior("Laplace", mu=mu, b=b, dims="fourier")
        monthly = MonthlyFourier(n_order=2, prior=dist)
        prior = monthly.sample_prior(samples=100)
        curve = monthly.sample_curve(prior)

        _, axes = monthly.plot_curve(curve)
        axes[0].set(title="Monthly Fourier Seasonality")
        plt.show()

    n_order : int
        Number of fourier modes to use.
    prefix : str, optional
        Alternative prefix for the fourier seasonality, by default None or
        "fourier"
    prior : Prior | VariableFactory, optional
        Prior distribution or VariableFactory for the fourier seasonality beta parameters, by
        default `Prior("Laplace", mu=0, b=1)`
    name : str, optional
        Name of the variable that multiplies the fourier modes, by default None
    variable_name : str, optional
        Name of the variable that multiplies the fourier modes, by default None

    """

    days_in_period: float = DAYS_IN_MONTH

    def _get_default_start_date(self) -> datetime.datetime:
        """Get the default start date for monthly seasonality.

        Returns the first day of the current month.
        """
        now = datetime.datetime.now()
        return datetime.datetime(year=now.year, month=now.month, day=1)

    def _get_days_in_period(self, dates: pd.DatetimeIndex) -> pd.Index:
        """Return the dayofyear within the yearly periodicity.

        Returns
        -------
        int or float
            The relevant period within the characteristic periodicity
        """
        return dates.dayofyear


class WeeklyFourier(FourierBase):
    """Weekly fourier seasonality.

    .. plot::
        :context: close-figs

        import arviz as az
        import matplotlib.pyplot as plt
        import numpy as np

        from pymc_marketing.mmm import WeeklyFourier
        from pymc_marketing.prior import Prior

        az.style.use("arviz-white")

        seed = sum(map(ord, "Weekly"))
        rng = np.random.default_rng(seed)

        mu = np.array([0, 0, 0.5, 0])
        b = 0.075
        dist = Prior("Laplace", mu=mu, b=b, dims="fourier")
        weekly = WeeklyFourier(n_order=2, prior=dist)
        prior = weekly.sample_prior(samples=100)
        curve = weekly.sample_curve(prior)

        _, axes = weekly.plot_curve(curve)
        axes[0].set(title="Weekly Fourier Seasonality")
        plt.show()

    n_order : int
        Number of fourier modes to use.
    prefix : str, optional
        Alternative prefix for the fourier seasonality, by default None or
        "fourier"
    prior : Prior | VariableFactory, optional
        Prior distribution or VariableFactory for the fourier seasonality beta parameters, by
        default `Prior("Laplace", mu=0, b=1)`
    name : str, optional
        Name of the variable that multiplies the fourier modes, by default None
    variable_name : str, optional
        Name of the variable that multiplies the fourier modes, by default None

    """

    days_in_period: float = DAYS_IN_WEEK

    def _get_default_start_date(self) -> datetime.datetime:
        """Get the default start date for weekly seasonality.

        Returns the first day of the current month.
        """
        now = datetime.datetime.now()
        return datetime.datetime.fromisocalendar(
            year=now.year, week=now.isocalendar().week, day=1
        )

    def _get_days_in_period(self, dates: pd.DatetimeIndex) -> pd.Index:
        """Return the weekday within the weekly periodicity.

        Returns
        -------
        int or float
            The relevant period within the characteristic periodicity
        """
        return dates.weekday


def _is_yearly_fourier(data: Any) -> bool:
    return data.get("class") == "YearlyFourier"


def _is_monthly_fourier(data: Any) -> bool:
    return data.get("class") == "MonthlyFourier"


def _is_weekly_fourier(data: Any) -> bool:
    return data.get("class") == "WeeklyFourier"


register_deserialization(
    is_type=_is_yearly_fourier,
    deserialize=lambda data: YearlyFourier.from_dict(data),
)

register_deserialization(
    is_type=_is_monthly_fourier,
    deserialize=lambda data: MonthlyFourier.from_dict(data),
)

register_deserialization(
    is_type=_is_weekly_fourier, deserialize=lambda data: WeeklyFourier.from_dict(data)
)
