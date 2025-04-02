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
"""Additive effects for the multidimensional Marketing Mix Model."""

from typing import Protocol

import pandas as pd
import pymc as pm
import xarray as xr
from pytensor import tensor as pt

from pymc_marketing.mmm.events import EventEffect, days_from_reference
from pymc_marketing.mmm.fourier import FourierBase
from pymc_marketing.mmm.linear_trend import LinearTrend
from pymc_marketing.prior import create_dim_handler


class MMM(Protocol):
    """Protocol MMM."""

    @property
    def dims(self) -> tuple[str, ...]:
        """The additional dimensions of the MMM target."""

    @property
    def model(self) -> pm.Model:
        """The PyMC model."""


class MuEffect(Protocol):
    """Protocol for arbitrary additive mu effect."""

    def create_data(self, mmm: MMM) -> None:
        """Create the required data in the model."""

    def create_effect(self, mmm: MMM) -> pt.TensorVariable:
        """Create the additive effect in the model."""

    def set_data(self, mmm: MMM, model: pm.Model, X: xr.Dataset) -> None:
        """Set the data for new predictions."""


class FourierEffect:
    """Fourier seasonality additive effect for MMM."""

    def __init__(self, fourier: FourierBase):
        """Initialize the Fourier effect.

        Parameters
        ----------
        fourier : FourierBase

        """
        self.fourier = fourier

    def create_data(self, mmm: MMM) -> None:
        """Create the required data in the model.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance
        """
        model = mmm.model

        # Get dates from model coordinates
        dates = pd.to_datetime(model.coords["date"])

        # Add weekday data to the model
        pm.Data(
            f"{self.fourier.prefix}_day",
            self.fourier._get_days_in_period(dates).to_numpy(),
            dims="date",
        )

    def create_effect(self, mmm: MMM) -> pt.TensorVariable:
        """Create the Fourier effect in the model.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance

        Returns
        -------
        pt.TensorVariable
            The Fourier effect
        """
        model = mmm.model

        # Apply the Fourier transformation to data
        day_data = model[f"{self.fourier.prefix}_day"]
        fourier_effect = self.fourier.apply(day_data)

        # Create a deterministic variable for the effect
        dims = (dim for dim in mmm.dims if dim in self.fourier.prior.dims)
        fourier_dims = ("date", *dims)
        fourier_effect_det = pm.Deterministic(
            f"{self.fourier.prefix}_effect",
            fourier_effect,
            dims=fourier_dims,
        )

        # Handle dimensions for the MMM model
        dim_handler = create_dim_handler(("date", *mmm.dims))
        return dim_handler(fourier_effect_det, fourier_dims)

    def set_data(self, mmm: MMM, model: pm.Model, X: xr.Dataset) -> None:
        """Set the data for new predictions.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance
        model : pm.Model
            The PyMC model
        X : xr.Dataset
            The dataset for prediction
        """
        # Get dates from the new dataset
        new_dates = pd.to_datetime(model.coords["date"])

        # Update the data
        new_data = {
            f"{self.fourier.prefix}_day": self.fourier._get_days_in_period(
                new_dates
            ).to_numpy()
        }
        pm.set_data(new_data=new_data, model=model)


class LinearTrendEffect:
    """Wrapper for LinearTrend to use with MMM's MuEffect protocol.

    This class adapts the LinearTrend component to be used as an additive effect
    in the MMM model.

    Parameters
    ----------
    trend : LinearTrend
        The LinearTrend instance to wrap.
    prefix : str
        The prefix to use for variables in the model.

    Examples
    --------
    Out of sample predictions:

    .. note::

        No new changepoints are used for the out of sample predictions. The trend
        effect is linearly extrapolated from the last changepoint.

    .. plot::
        :include-source: True
        :context: reset

        import pandas as pd
        import numpy as np

        import matplotlib.pyplot as plt

        import pymc as pm

        from pymc_marketing.mmm.linear_trend import LinearTrend
        from pymc_marketing.mmm.additive_effect import LinearTrendEffect

        seed = sum(map(ord, "LinearTrend out of sample"))
        rng = np.random.default_rng(seed)


        class MockMMM:
            pass


        dates = pd.date_range("2025-01-01", periods=52, freq="W")
        coords = {"date": dates}
        model = pm.Model(coords=coords)

        mock_mmm = MockMMM()
        mock_mmm.dims = ()
        mock_mmm.model = model

        effect = LinearTrendEffect(
            trend=LinearTrend(n_changepoints=8),
            prefix="trend",
        )

        with mock_mmm.model:
            effect.create_data(mock_mmm)
            pm.Deterministic(
                "effect",
                effect.create_effect(mock_mmm),
                dims="date",
            )

            idata = pm.sample_prior_predictive(random_seed=rng)

        idata["posterior"] = idata.prior

        n_new = 10 + 1
        new_dates = pd.date_range(
            dates.max(),
            periods=n_new,
            freq="W",
        )


        with mock_mmm.model:
            mock_mmm.model.set_dim("date", n_new, new_dates)

            effect.set_data(mock_mmm, mock_mmm.model, None)

            pm.sample_posterior_predictive(
                idata,
                var_names=["effect"],
                random_seed=rng,
                extend_inferencedata=True,
            )

        draw = rng.choice(range(idata.posterior.sizes["draw"]))
        sel = dict(chain=0, draw=draw)

        before = idata.posterior.effect.sel(sel).to_series()
        after = idata.posterior_predictive.effect.sel(sel).to_series()

        ax = before.plot(color="C0")
        after.plot(color="C0", linestyle="dashed", ax=ax)
        plt.show()

    """

    def __init__(self, trend: LinearTrend, prefix: str):
        self.trend = trend
        self.prefix = prefix
        self.linear_trend_first_date: pd.Timestamp

    def create_data(self, mmm: MMM) -> None:
        """Create the required data in the model.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.
        """
        model: pm.Model = mmm.model

        # Create time index data (normalized between 0 and 1)
        dates = pd.to_datetime(model.coords["date"])
        self.linear_trend_first_date = dates[0]
        t = (dates - self.linear_trend_first_date).days.astype(float)

        pm.Data(f"{self.prefix}_t", t, dims="date")

    def create_effect(self, mmm: MMM) -> pt.TensorVariable:
        """Create the trend effect in the model.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.

        Returns
        -------
        pt.TensorVariable
            The trend effect in the model.
        """
        model: pm.Model = mmm.model

        # Get the time data
        t = model[f"{self.prefix}_t"]
        t_max = t.max().eval()
        t = t / t_max if t_max > 0 else t

        # Apply the trend
        trend_effect = self.trend.apply(t)

        # Create deterministic for the trend effect
        trend_dims = ("date", *self.trend.dims)  # type: ignore
        trend_effect = pm.Deterministic(
            f"{self.prefix}_effect",
            trend_effect,
            dims=trend_dims,
        )

        # Return the trend effect
        dim_handler = create_dim_handler(("date", *mmm.dims))
        return dim_handler(trend_effect, trend_dims)

    def set_data(self, mmm: MMM, model: pm.Model, X: xr.Dataset) -> None:
        """Set the data for new predictions.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.
        model : pm.Model
            The PyMC model.
        X : xr.Dataset
            The dataset for prediction.
        """
        # Create normalized time index for new data
        new_dates = pd.to_datetime(model.coords["date"])
        t = (new_dates - self.linear_trend_first_date).days.astype(float)

        # Update the data
        pm.set_data({f"{self.prefix}_t": t}, model=model)


def create_event_mu_effect(
    df_events: pd.DataFrame,
    prefix: str,
    effect: EventEffect,
) -> MuEffect:
    """Create an event effect for the MMM.

    This class has the ability to create data and mean effects for the MMM model.

    Parameters
    ----------
    df_events : pd.DataFrame
        The DataFrame containing the event data.
            * `name`: name of the event. Used as the model coordinates.
            * `start_date`: start date of the event
            * `end_date`: end date of the event
    prefix : str
        The prefix to use for the event effect and associated variables.
    effect : EventEffect
        The event effect to apply.

    Returns
    -------
    MuEffect
        The event effect which is used in the MMM.

    """
    if missing_columns := set(["start_date", "end_date", "name"]).difference(
        df_events.columns,
    ):
        raise ValueError(f"Columns {missing_columns} are missing in df_events.")

    effect.basis.prefix = prefix

    reference_date = "2025-01-01"
    start_dates = pd.to_datetime(df_events["start_date"])
    end_dates = pd.to_datetime(df_events["end_date"])

    class Effect:
        """Event effect class for the MMM."""

        def create_data(self, mmm: MMM) -> None:
            """Create the required data in the model.

            Parameters
            ----------
            mmm : MMM
                The MMM model instance.

            """
            model: pm.Model = mmm.model

            model_dates = pd.to_datetime(model.coords["date"])

            model.add_coord(prefix, df_events["name"].to_numpy())

            if "days" not in model:
                pm.Data(
                    "days",
                    days_from_reference(model_dates, reference_date),
                    dims="date",
                )

            pm.Data(
                f"{prefix}_start_diff",
                days_from_reference(start_dates, reference_date),
                dims=prefix,
            )
            pm.Data(
                f"{prefix}_end_diff",
                days_from_reference(end_dates, reference_date),
                dims=prefix,
            )

        def create_effect(self, mmm: MMM) -> pt.TensorVariable:
            """Create the event effect in the model.

            Parameters
            ----------
            mmm : MMM
                The MMM model instance.

            Returns
            -------
            pt.TensorVariable
                The average event effect in the model.

            """
            model: pm.Model = mmm.model

            s_ref = model["days"][:, None] - model[f"{prefix}_start_diff"]
            e_ref = model["days"][:, None] - model[f"{prefix}_end_diff"]

            def create_basis_matrix(s_ref, e_ref):
                return pt.where(
                    (s_ref >= 0) & (e_ref <= 0),
                    0,
                    pt.where(pt.abs(s_ref) < pt.abs(e_ref), s_ref, e_ref),
                )

            X = create_basis_matrix(s_ref, e_ref)
            event_effect = effect.apply(X, name=prefix)

            total_effect = pm.Deterministic(
                f"{prefix}_total_effect",
                event_effect.sum(axis=1),
                dims="date",
            )

            dim_handler = create_dim_handler(("date", *mmm.dims))
            return dim_handler(total_effect, "date")

        def set_data(self, mmm: MMM, model: pm.Model, X: xr.Dataset) -> None:
            """Set the data for new predictions."""
            new_dates = pd.to_datetime(model.coords["date"])

            new_data = {
                "days": days_from_reference(new_dates, reference_date),
            }
            pm.set_data(new_data=new_data, model=model)

    return Effect()
