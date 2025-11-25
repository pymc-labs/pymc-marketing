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
"""Additive effects for the multidimensional Marketing Mix Model.

Example of a custom additive effect
--------

1. Custom negative-effect component (added as a MuEffect)

.. code-block:: python

    import numpy as np
    import pandas as pd
    import pymc as pm
    from pymc_extras.prior import create_dim_handler

    # A simple custom effect that penalizes certain dates/segments with a
    # negative-only coefficient. This is not a "control" in the MMM sense, so
    # give it a different name/prefix to avoid clashing with built-in controls.
    class PenaltyEffect:
        '''Example MuEffect that applies a negative coefficient to a user-specified pattern.
        '''

        def __init__(self, name: str, penalty_provider):
            self.name = name
            self.penalty_provider = penalty_provider

        def create_data(self, mmm):
            # Produce penalty values aligned with model dates (and optional extra dims)
            dates = pd.to_datetime(mmm.model.coords["date"])
            penalty = self.penalty_provider(dates)
            pm.Data(f"{self.name}_penalty", penalty, dims=("date", *mmm.dims))

        def create_effect(self, mmm):
            model = mmm.model
            penalty = model[f"{self.name}_penalty"]  # dims: (date, *mmm.dims)

            # Negative-only coefficient per extra dims, broadcast over date
            coef = pm.TruncatedNormal(f"{self.name}_coef", mu=-0.5, sigma=-0.05, lower=-1.0, upper=0.0, dims=mmm.dims)

            dim_handler = create_dim_handler(("date", *mmm.dims))
            effect = pm.Deterministic(
                f"{self.name}_effect_contribution",
                dim_handler(coef, mmm.dims) * penalty,
                dims=("date", *mmm.dims),
            )
            return effect  # Must have dims ("date", *mmm.dims)

        def set_data(self, mmm, model, X):
            # Update to future dates during posterior predictive
            dates = pd.to_datetime(model.coords["date"])
            penalty = self.penalty_provider(dates)
            pm.set_data({f"{self.name}_penalty": penalty}, model=model)

    Usage
    -----
    # Example weekend penalty (Sat/Sun = 1, else 0), applied per geo if present
    weekend_penalty = PenaltyEffect(
        name="brand_penalty",
        penalty_provider=lambda dates: pd.Series(dates)
        .dt.dayofweek.isin([5, 6])
        .astype(float)
        .to_numpy()[:, None]  # if mmm.dims == ("geo",), broadcast over geo
    )

    # Build your MMM as usual (with channels, etc.), then add the effect before build/fit:
    # mmm = MMM(...)
    # mmm.mu_effects.append(weekend_penalty)
    # mmm.build_model(X, y)
    # mmm.fit(X, y, ...)
    # At prediction time, the effect updates itself via set_data.

How it works
------------
- Mu effects follow a simple protocol: ``create_data(mmm)``, ``create_effect(mmm)``,
  and ``set_data(mmm, model, X)``.
- During ``MMM.build_model(...)``, each effect’s ``create_data`` is called first to
  introduce any needed ``pm.Data``. Then ``create_effect`` must return a tensor with
  dims ("date", *mmm.dims) that is added additively to the model mean.
- During posterior predictive, ``set_data`` is called with the cloned PyMC model
  and the new coordinates; update any ``pm.Data`` you created using ``pm.set_data``.

Tips for custom components
--------------------------
- Use unique variable prefixes to avoid name clashes with built-in pieces like
  controls. Do not call your component "control"; choose a distinct name/prefix.
- Follow the patterns used by the provided effects in this module (e.g.,
  `FourierEffect`, `LinearTrendEffect`, `EventAdditiveEffect`):
  - In `create_data`, derive and register any required inputs into the model.
  - In `create_effect`, construct PyTensor expressions and return a contribution
    with dims ("date", *mmm.dims). If you need broadcasting, use
    `pymc_extras.prior.create_dim_handler` as shown above.
  - In `set_data`, update the data variables when dates/dims change.
"""

from typing import Any, Protocol

import pandas as pd
import pymc as pm
import xarray as xr
from pydantic import BaseModel, InstanceOf
from pymc_extras.prior import create_dim_handler
from pytensor import tensor as pt

from pymc_marketing.mmm.events import EventEffect, days_from_reference
from pymc_marketing.mmm.fourier import FourierBase
from pymc_marketing.mmm.linear_trend import LinearTrend
from pymc_marketing.mmm.utils import create_index


class Model(Protocol):
    """Protocol MMM."""

    @property
    def dims(self) -> tuple[str, ...]:
        """The additional dimensions of the MMM target."""

    @property
    def model(self) -> pm.Model:
        """The PyMC model."""


class MuEffect(Protocol):
    """Protocol for arbitrary additive mu effect."""

    def create_data(self, mmm: Model) -> None:
        """Create the required data in the model."""

    def create_effect(self, mmm: Model) -> pt.TensorVariable:
        """Create the additive effect in the model."""

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
        """Set the data for new predictions."""


class FourierEffect:
    """Fourier seasonality additive effect for MMM."""

    def __init__(self, fourier: FourierBase, date_dim_name: str = "date"):
        """Initialize the Fourier effect.

        Parameters
        ----------
        fourier : FourierBase
            The FourierBase instance to use for the effect.
        date_dim_name : str, optional
            The name of the date dimension in the model, by default "date".

        """
        self.fourier = fourier
        self.date_dim_name: str = date_dim_name

    def create_data(self, mmm: Model) -> None:
        """Create the required data in the model.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance
        """
        model = mmm.model

        # Get dates from model coordinates
        dates = pd.to_datetime(model.coords[self.date_dim_name])

        # Add weekday data to the model
        pm.Data(
            f"{self.fourier.prefix}_day",
            self.fourier._get_days_in_period(dates).to_numpy(),
            dims=self.date_dim_name,
        )

    def create_effect(self, mmm: Model) -> pt.TensorVariable:
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

        # Store the unsummed basis components (including the internal fourier mode dim)
        # so users can inspect individual sine/cos contributions if desired.
        def create_deterministic(x: pt.TensorVariable) -> None:
            pm.Deterministic(
                f"{self.fourier.prefix}_components",
                x,
                dims=(self.date_dim_name, *self.fourier.prior.dims),
            )

        # Call apply to create the components deterministic (unsummed basis * betas)
        _ = self.fourier.apply(day_data, result_callback=create_deterministic)

        # Retrieve the components deterministic just created
        components_var = model[f"{self.fourier.prefix}_components"]
        component_dims = model.named_vars_to_dims[components_var.name]
        # Identify axis of the fourier prefix dimension and collapse it
        prefix_axis = component_dims.index(self.fourier.prefix)
        collapsed = components_var.sum(axis=prefix_axis)

        # Determine final dims order consistent with MMM dims
        dims = tuple(dim for dim in mmm.dims if dim in self.fourier.prior.dims)
        fourier_dims = (self.date_dim_name, *dims)

        fourier_contribution = pm.Deterministic(
            f"{self.fourier.prefix}_contribution",
            collapsed,
            dims=fourier_dims,
        )

        # Broadcast to full MMM dims ordering
        dim_handler = create_dim_handler((self.date_dim_name, *mmm.dims))
        return dim_handler(fourier_contribution, fourier_dims)

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
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
        new_dates = pd.to_datetime(model.coords[self.date_dim_name])

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

    def __init__(self, trend: LinearTrend, prefix: str, date_dim_name: str = "date"):
        self.trend = trend
        self.prefix = prefix
        self.linear_trend_first_date: pd.Timestamp
        self.date_dim_name: str = date_dim_name

    def create_data(self, mmm: Model) -> None:
        """Create the required data in the model.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.
        """
        model: pm.Model = mmm.model

        # Create time index data (normalized between 0 and 1)
        dates = pd.to_datetime(model.coords[self.date_dim_name])
        self.linear_trend_first_date = dates[0]
        t = (dates - self.linear_trend_first_date).days.astype(float)

        pm.Data(f"{self.prefix}_t", t, dims=self.date_dim_name)

    def create_effect(self, mmm: Model) -> pt.TensorVariable:
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
        trend_dims = (self.date_dim_name, *self.trend.dims)  # type: ignore
        trend_non_broadcastable_dims = (
            self.date_dim_name,
            *self.trend.non_broadcastable_dims,
        )
        trend_effect = pm.Deterministic(
            f"{self.prefix}_effect_contribution",
            trend_effect[create_index(trend_dims, trend_non_broadcastable_dims)],
            dims=trend_non_broadcastable_dims,
        )

        # Return the trend effect
        dim_handler = create_dim_handler((self.date_dim_name, *mmm.dims))
        return dim_handler(trend_effect, trend_non_broadcastable_dims)

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
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
        new_dates = pd.to_datetime(model.coords[self.date_dim_name])
        t = (new_dates - self.linear_trend_first_date).days.astype(float)

        # Update the data
        pm.set_data({f"{self.prefix}_t": t}, model=model)


class EventAdditiveEffect(BaseModel):
    """Event effect class for the MMM.

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
    reference_date : str
        The arbitrary reference date to calculate distance from events in days. Default
        is "2025-01-01".
    date_dim_name : str
        The name of the date dimension in the model. Default is "date".

    """

    df_events: InstanceOf[pd.DataFrame]
    prefix: str
    effect: EventEffect
    reference_date: str = "2025-01-01"
    date_dim_name: str = "date"

    def model_post_init(self, context: Any, /) -> None:
        """Post initialization of the model."""
        if missing_columns := set(["start_date", "end_date", "name"]).difference(
            self.df_events.columns
        ):
            raise ValueError(f"Columns {missing_columns} are missing in df_events.")

        self.effect.basis.prefix = self.prefix

    @property
    def start_dates(self) -> pd.Series:
        """The start dates of the events."""
        return pd.to_datetime(self.df_events["start_date"])

    @property
    def end_dates(self) -> pd.Series:
        """The end dates of the events."""
        return pd.to_datetime(self.df_events["end_date"])

    def create_data(self, mmm: Model) -> None:
        """Create the required data in the model.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.

        """
        model: pm.Model = mmm.model

        model_dates = pd.to_datetime(model.coords[self.date_dim_name])

        model.add_coord(self.prefix, self.df_events["name"].to_numpy())

        if "days" not in model:
            pm.Data(
                "days",
                days_from_reference(model_dates, self.reference_date),
                dims=self.date_dim_name,
            )

        pm.Data(
            f"{self.prefix}_start_diff",
            days_from_reference(self.start_dates, self.reference_date),
            dims=self.prefix,
        )
        pm.Data(
            f"{self.prefix}_end_diff",
            days_from_reference(self.end_dates, self.reference_date),
            dims=self.prefix,
        )

    def create_effect(self, mmm: Model) -> pt.TensorVariable:
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

        start_ref = model["days"][:, None] - model[f"{self.prefix}_start_diff"]
        end_ref = model["days"][:, None] - model[f"{self.prefix}_end_diff"]

        def create_basis_matrix(start_ref, end_ref):
            return pt.where(
                (start_ref >= 0) & (end_ref <= 0),
                0,
                pt.where(pt.abs(start_ref) < pt.abs(end_ref), start_ref, end_ref),
            )

        X = create_basis_matrix(start_ref, end_ref)
        event_effect = self.effect.apply(X, name=self.prefix)

        total_effect = pm.Deterministic(
            f"{self.prefix}_total_effect",
            event_effect.sum(axis=1),
            dims=self.date_dim_name,
        )

        dim_handler = create_dim_handler((self.date_dim_name, *mmm.dims))
        return dim_handler(total_effect, self.date_dim_name)

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
        """Set the data for new predictions."""
        new_dates = pd.to_datetime(model.coords[self.date_dim_name])

        new_data = {
            "days": days_from_reference(new_dates, self.reference_date),
        }
        pm.set_data(new_data=new_data, model=model)
