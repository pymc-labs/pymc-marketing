#   Copyright 2022 - 2026 The PyMC Labs Developers
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
-----------------------------------

1. Custom negative-effect component (added as a MuEffect)

.. code-block:: python

    import numpy as np
    import pandas as pd
    import pymc as pm
    import pymc.dims as pmd
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
            dates = safe_to_datetime(mmm.model.coords["date"], "date")
            penalty = self.penalty_provider(dates)
            pmd.Data(f"{self.name}_penalty", penalty, dims=("date", *mmm.dims))

        def create_effect(self, mmm):
            model = mmm.model
            penalty = model[f"{self.name}_penalty"]  # dims: (date, *mmm.dims)

            # Negative-only coefficient per extra dims, broadcast over date
            coef = pmd.TruncatedNormal(f"{self.name}_coef", mu=-0.5, sigma=-0.05, lower=-1.0, upper=0.0, dims=mmm.dims)

            dim_handler = create_dim_handler(("date", *mmm.dims))
            effect = pmd.Deterministic(
                f"{self.name}_effect_contribution",
                dim_handler(coef, mmm.dims) * penalty,
                dims=("date", *mmm.dims),
            )
            return effect  # Must have dims ("date", *mmm.dims)

        def set_data(self, mmm, model, X):
            # Update to future dates during posterior predictive
            dates = safe_to_datetime(model.coords["date"], "date")
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
    # mmm.add_mu_effect(weekend_penalty)
    # mmm.build_model(X, y)
    # mmm.fit(X, y, ...)
    # At prediction time, the effect updates itself via set_data.

How it works
------------
- Mu effects follow a simple protocol: ``create_data(mmm)``, ``create_effect(mmm)``,
  and ``set_data(mmm, model, X)``.
- During ``MMM.build_model(...)``, each effect's ``create_data`` is called first to
  introduce any needed ``pmd.Data``. Then ``create_effect`` must return a tensor with
  dims ``("date", *mmm.dims)`` that is added additively to the model mean.
- During posterior predictive, ``set_data`` is called with the cloned PyMC model
  and the new coordinates; update any ``pmd.Data`` you created using ``pm.set_data``.

Tips for custom components
--------------------------
- Use unique variable prefixes to avoid name clashes with built-in pieces like
  controls. Do not call your component "control"; choose a distinct name/prefix.
- Follow the patterns used by the provided effects in this module (e.g.,
  ``FourierEffect``, ``LinearTrendEffect``, ``EventAdditiveEffect``):

  - In ``create_data``, derive and register any required inputs into the model.
  - In ``create_effect``, construct PyTensor expressions and return a contribution
    with dims ``("date", *mmm.dims)``. If you need broadcasting, use
    ``pymc_extras.prior.create_dim_handler`` as shown above.
  - In ``set_data``, update the data variables when dates/dims change.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy.typing as npt
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.xtensor as ptx
import xarray as xr
from pydantic import Field, InstanceOf
from pymc_extras.prior import Prior
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm.events import EventEffect, days_from_reference
from pymc_marketing.mmm.fourier import FourierBase
from pymc_marketing.mmm.linear_trend import LinearTrend
from pymc_marketing.mmm.link import LinkFunction
from pymc_marketing.mmm.validating import _validate_non_numeric_dtype
from pymc_marketing.serialization import SerializableBaseModel


def safe_to_datetime(
    coords_values: pd.Series | pd.Index | list | tuple | pd.DatetimeIndex | npt.NDArray,
    coord_name: str = "date",
    validate_non_numeric: bool = True,
) -> pd.DatetimeIndex:
    """Safely convert coordinates to datetime, with validation.

    This function prevents the issue where numeric values (e.g., [0, 1, 2, 3])
    get incorrectly converted to dates starting from January 1st 1970 with
    nanosecond intervals.

    Parameters
    ----------
    coords_values : pd.Series | pd.Index | list | tuple | pd.DatetimeIndex | npt.NDArray
        The coordinate values to convert to datetime
    coord_name : str, optional
        The name of the coordinate dimension (default: "date")
    validate_non_numeric : bool, optional
        Whether to validate that values are not numeric dtype. Set to False
        when intentionally converting numeric time indices. Default: True

    Returns
    -------
    pd.DatetimeIndex
        The converted datetime index

    Raises
    ------
    ValueError
        If the coordinate values have numeric dtype and validate_non_numeric is True

    Examples
    --------
    >>> # Good usage - string dates
    >>> safe_to_datetime(["2024-01-01", "2024-01-02"])

    >>> # Good usage - already datetime
    >>> safe_to_datetime(pd.to_datetime(["2024-01-01", "2024-01-02"]))

    >>> # Raises error - numeric values with validation
    >>> safe_to_datetime([0, 1, 2, 3])  # Raises ValueError

    >>> # Allowed - numeric time indices with validation disabled
    >>> safe_to_datetime([0, 1, 2, 3], validate_non_numeric=False)
    """
    # Convert to pandas Series/Index for dtype checking
    if isinstance(coords_values, pd.DatetimeIndex):
        # Already datetime, return as-is
        return coords_values

    # Validate that values are not numeric dtype (if requested)
    if validate_non_numeric:
        _validate_non_numeric_dtype(coords_values, f"Coordinate '{coord_name}'")

    result = pd.to_datetime(coords_values)
    # Ensure we always return DatetimeIndex, not Series
    if isinstance(result, pd.Series):
        return pd.DatetimeIndex(result)
    return result


def _get_datetime_coords(
    coords: pd.Index | npt.NDArray,
    coord_name: str,
) -> pd.DatetimeIndex:
    """Get datetime coordinates with automatic validation logic.

    Automatically skips numeric validation for non-date coordinate names
    (e.g., 'time'), allowing numeric indices for customer choice models.

    Parameters
    ----------
    coords : pd.Index | npt.NDArray
        The coordinate values from the model
    coord_name : str
        The name of the coordinate dimension

    Returns
    -------
    pd.DatetimeIndex
        The converted datetime index
    """
    # Skip validation for non-date coordinates (e.g., numeric "time" indices)
    validate = coord_name == "date"
    return safe_to_datetime(coords, coord_name, validate_non_numeric=validate)


class Model(Protocol):
    """Protocol MMM."""

    @property
    def dims(self) -> tuple[str, ...]:
        """The additional dimensions of the MMM target."""

    @property
    def model(self) -> pm.Model:
        """The PyMC model."""


class MuEffect(SerializableBaseModel, ABC):
    """Abstract base class for arbitrary additive mu effects.

    All mu_effects must inherit from this Pydantic BaseModel to ensure proper
    serialization and deserialization when saving/loading MMM models.
    """

    @abstractmethod
    def create_data(self, mmm: Model) -> None:
        """Create the required data in the model."""

    @abstractmethod
    def create_effect(self, mmm: Model) -> XTensorVariable:
        """Create the additive effect in the model."""

    @abstractmethod
    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
        """Set the data for new predictions."""

    @property
    def contribution_var_name(self) -> str:
        """Name of the posterior deterministic holding this effect's contribution.

        Used by :meth:`MMM.compute_counterfactual_contributions_dataset` to
        locate the effect's linear-predictor contribution and include it in
        the decomposition.  The default assumes the effect registers
        ``f"{self.prefix}_effect_contribution"`` (the convention used by
        :class:`LinearTrendEffect` and :class:`EventEffect`); effects that
        register a different name must override this property.

        Raises
        ------
        NotImplementedError
            If the effect has no ``prefix`` attribute and does not override
            this property.
        """
        prefix = getattr(self, "prefix", None)
        if prefix is None:
            raise NotImplementedError(
                f"{type(self).__name__} must define 'contribution_var_name'."
            )
        return f"{prefix}_effect_contribution"

    def idata_groups(self) -> dict[str, xr.Dataset]:
        """Return supplementary data groups to store in DataTree.

        Override in subclasses that need to persist large DataFrames or
        other non-JSON-serializable data alongside the model.

        Each entry is stored as a top-level group in the DataTree
        netCDF file during ``save()`` and is available to custom
        deserializers via ``DeserializationContext(idata=...)``.

        Returns
        -------
        dict[str, xr.Dataset]
            Group name to xarray Dataset mapping.
        """
        return {}


class OptimizableMuEffect(MuEffect, ABC):
    """A :class:`MuEffect` whose own lever can be optimized jointly with media budgets.

    The lever is the ``pm.Data`` node named by :attr:`lever_var_name`
    (registered in :meth:`MuEffect.create_data` with a single non-date dim;
    defaults to ``f"{prefix}_data"``).
    ``MMM.budget_optimizer`` translates each optimizable effect into an
    ``optimizable_vars`` entry -- the node name plus the effect's native
    :attr:`lever_bounds` -- and the
    :class:`~pymc_marketing.mmm.budget_optimizer.BudgetOptimizer` co-optimizes
    that variable purely by name on the model graph, alongside the media
    budgets. The optimal values come back in ``result.optimized_vars``.

    The lever never enters the media budget-sum constraint; its economic cost
    (if any) belongs in the model response. Pair the optimization with
    ``response_variable="total_response_original_scale"`` so the effect's
    contribution enters the objective.
    """

    @property
    def lever_bounds(self) -> list[tuple[float | None, float | None]] | None:
        """Per-lever ``(low, high)`` bounds in the lever's native units.

        ``None`` (default) leaves the lever unbounded. Effects with a naturally
        bounded lever (e.g. a discount fraction) should override this with one
        ``(low, high)`` tuple per lever entry, in the lever dim's coordinate
        order.
        """
        return None

    @property
    def lever_var_name(self) -> str:
        """Name of the ``pm.Data`` node the optimizer substitutes for this effect.

        The default assumes the effect registers ``f"{self.prefix}_data"`` in
        :meth:`MuEffect.create_data`; effects that register a different lever
        node (or have no ``prefix`` attribute) must override this property.

        Raises
        ------
        NotImplementedError
            If the effect has no ``prefix`` attribute and does not override
            this property.
        """
        prefix = getattr(self, "prefix", None)
        if prefix is None:
            raise NotImplementedError(
                f"{type(self).__name__} must define 'lever_var_name'."
            )
        return f"{prefix}_data"


class FourierEffect(MuEffect):
    """Fourier seasonality additive effect for MMM."""

    fourier: InstanceOf[FourierBase]
    date_dim_name: str = Field("date")

    @property
    def contribution_var_name(self) -> str:
        """Fourier effects register ``f"{fourier.prefix}_contribution"``."""
        return f"{self.fourier.prefix}_contribution"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict. ``__type__`` is injected by the registry wrapper."""
        return {
            "fourier": self.fourier.to_dict(),
            "date_dim_name": self.date_dim_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FourierEffect":
        """Reconstruct from a dict, using registry for nested Fourier type."""
        from pymc_marketing.serialization import serialization

        work = {k: v for k, v in data.items() if k != "__type__"}
        fourier_data = work["fourier"]
        if "__type__" in fourier_data:
            fourier = serialization.deserialize(fourier_data)
        else:
            from pymc_extras.deserialize import deserialize

            fourier = deserialize(fourier_data)
        return cls(fourier=fourier, date_dim_name=work.get("date_dim_name", "date"))

    def create_data(self, mmm: Model) -> None:
        """Create the required data in the model.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance
        """
        model = mmm.model

        # Get dates from model coordinates
        dates = _get_datetime_coords(
            model.coords[self.date_dim_name], self.date_dim_name
        )

        # Add weekday data to the model
        pmd.Data(
            f"{self.fourier.prefix}_day",
            self.fourier._get_days_in_period(dates).to_numpy(),
            dims=self.date_dim_name,
        )

    def create_effect(self, mmm: Model) -> XTensorVariable:
        """Create the Fourier effect in the model.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance

        Returns
        -------
        XTensorVariable
            The Fourier effect
        """
        model = mmm.model

        # Apply the Fourier transformation to data
        day_data = model[f"{self.fourier.prefix}_day"]

        # Call apply to create the components deterministic (unsummed basis * betas)
        fourier_dim = self.fourier.prefix
        fourier_components = pmd.Deterministic(
            f"{self.fourier.prefix}_components",
            self.fourier.apply(day_data, sum=False).transpose(
                self.date_dim_name, ..., fourier_dim
            ),
        )

        return pmd.Deterministic(
            f"{self.fourier.prefix}_contribution",
            fourier_components.sum(dim=fourier_dim),
        )

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
        new_dates = _get_datetime_coords(
            model.coords[self.date_dim_name], self.date_dim_name
        )

        # Update the data
        new_data = {
            f"{self.fourier.prefix}_day": self.fourier._get_days_in_period(
                new_dates
            ).to_numpy()
        }
        pm.set_data(new_data=new_data, model=model)


class LinearTrendEffect(MuEffect):
    """Wrapper for LinearTrend to use with MMM's MuEffect protocol.

    This class adapts the LinearTrend component to be used as an additive effect
    in the MMM model.

    Parameters
    ----------
    trend : LinearTrend
        The LinearTrend instance to wrap.
    prefix : str
        The prefix to use for variables in the model.
    date_dim_name : str
        The name of the date dimension in the model.

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
        import pymc.dims as pmd

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
            pmd.Deterministic(
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

        before = idata.posterior["effect"].sel(sel).to_series()
        after = idata.posterior_predictive["effect"].sel(sel).to_series()

        ax = before.plot(color="C0")
        after.plot(color="C0", linestyle="dashed", ax=ax)
        plt.show()

    """

    trend: InstanceOf[LinearTrend]
    prefix: str
    date_dim_name: str = Field("date")
    linear_trend_first_date: Any = Field(default=None, exclude=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict. ``__type__`` is injected by the registry wrapper."""
        return {
            "trend": self.trend.to_dict(),
            "prefix": self.prefix,
            "date_dim_name": self.date_dim_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinearTrendEffect":
        """Reconstruct from a dict, using registry for nested LinearTrend."""
        from pymc_marketing.serialization import serialization

        work = {k: v for k, v in data.items() if k != "__type__"}
        trend_data = work["trend"]
        if "__type__" in trend_data:
            trend = serialization.deserialize(trend_data)
        else:
            from pymc_extras.deserialize import deserialize

            trend_dict = trend_data.copy()
            if trend_dict.get("priors"):
                trend_dict["priors"] = {
                    k: deserialize(v) for k, v in trend_dict["priors"].items()
                }
            trend = LinearTrend.model_validate(trend_dict)
        return cls(
            trend=trend,
            prefix=work["prefix"],
            date_dim_name=work.get("date_dim_name", "date"),
        )

    def create_data(self, mmm: Model) -> None:
        """Create the required data in the model.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.
        """
        model: pm.Model = mmm.model

        # Create time index data (normalized between 0 and 1)
        dates = _get_datetime_coords(
            model.coords[self.date_dim_name], self.date_dim_name
        )
        self.linear_trend_first_date = dates[0]
        t = (dates - self.linear_trend_first_date).days.astype(float)

        pmd.Data(f"{self.prefix}_t", t, dims=self.date_dim_name)

    def create_effect(self, mmm: Model) -> XTensorVariable:
        """Create the trend effect in the model.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.

        Returns
        -------
        XTensorVariable
            The trend effect in the model.
        """
        model: pm.Model = mmm.model

        # Get the time data
        t_name = f"{self.prefix}_t"
        t = model[t_name]

        t_max = t.max()
        t = t / ptx.math.switch(t_max > 0, t_max, 1)
        trend_effect = self.trend.apply(t)

        return pmd.Deterministic(
            f"{self.prefix}_effect_contribution",
            trend_effect,
        )

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
        new_dates = _get_datetime_coords(
            model.coords[self.date_dim_name], self.date_dim_name
        )
        t = (new_dates - self.linear_trend_first_date).days.astype(float)

        # Update the data
        pm.set_data({f"{self.prefix}_t": t}, model=model)


class EventAdditiveEffect(MuEffect):
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict with ``__type__`` key.

        The ``df_events`` DataFrame is NOT included in the dict; instead a
        ``df_events_group`` key stores the idata group path where it lives.
        """
        return {
            "prefix": self.prefix,
            "reference_date": self.reference_date,
            "date_dim_name": self.date_dim_name,
            "effect": self.effect.to_dict(),
            "df_events_group": f"supplementary_data_{self.prefix}",
        }

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

        model_dates = _get_datetime_coords(
            model.coords[self.date_dim_name], self.date_dim_name
        )

        model.add_coord(self.prefix, self.df_events["name"].to_numpy())

        if "days" not in model:
            pmd.Data(
                "days",
                days_from_reference(model_dates, self.reference_date),
                dims=self.date_dim_name,
            )

        pmd.Data(
            f"{self.prefix}_start_diff",
            days_from_reference(self.start_dates, self.reference_date),
            dims=self.prefix,
        )
        pmd.Data(
            f"{self.prefix}_end_diff",
            days_from_reference(self.end_dates, self.reference_date),
            dims=self.prefix,
        )

    def create_effect(self, mmm: Model) -> XTensorVariable:
        """Create the event effect in the model.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.

        Returns
        -------
        XTensorVariable
            The average event effect in the model.

        """
        model: pm.Model = mmm.model

        days = model["days"]
        start_ref = days - model[f"{self.prefix}_start_diff"]
        end_ref = days - model[f"{self.prefix}_end_diff"]

        def create_basis_matrix(start_ref, end_ref):
            return ptx.math.where(
                (start_ref >= 0) & (end_ref <= 0),
                0,
                ptx.math.where(
                    ptx.math.abs(start_ref) < ptx.math.abs(end_ref), start_ref, end_ref
                ),
            )

        X = create_basis_matrix(start_ref, end_ref)
        event_effect = self.effect.apply(X, name=self.prefix)

        return pmd.Deterministic(
            f"{self.prefix}_total_effect",
            event_effect.sum(dim=self.prefix),
        )

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
        """Set the data for new predictions."""
        new_dates = _get_datetime_coords(
            model.coords[self.date_dim_name], self.date_dim_name
        )

        new_data = {
            "days": days_from_reference(new_dates, self.reference_date),
        }
        pm.set_data(new_data=new_data, model=model)

    def idata_groups(self) -> dict[str, xr.Dataset]:
        """Return the events DataFrame as a supplementary idata group."""
        return {
            f"supplementary_data_{self.prefix}": xr.Dataset.from_dataframe(
                self.df_events.reset_index(drop=True)
            ),
        }


def _deserialize_event_additive_effect(
    data: dict[str, Any],
    context: Any,
) -> EventAdditiveEffect:
    from pymc_marketing.serialization import SerializationError, serialization

    group_name = data["df_events_group"]

    if context is None or context.idata is None:
        raise SerializationError(
            f"Cannot deserialize EventAdditiveEffect: no DataTree "
            f"provided. The df_events DataFrame is stored in idata group "
            f"'{group_name}' and requires a DeserializationContext with idata."
        )

    try:
        ds = context.idata[group_name]
        if hasattr(ds, "dataset"):
            ds = ds.dataset
        df_events = ds.to_dataframe().reset_index()
    except (KeyError, AttributeError) as e:
        raise SerializationError(
            f"Cannot read supplementary data group '{group_name}' from "
            f"InferenceData: {e}"
        ) from e

    effect_data = data["effect"]
    if "__type__" in effect_data:
        effect = serialization.deserialize(effect_data)
    else:
        effect = EventEffect.from_dict(effect_data.get("data", effect_data))

    return EventAdditiveEffect(
        df_events=df_events,
        effect=effect,
        prefix=data["prefix"],
        reference_date=data.get("reference_date", "2025-01-01"),
        date_dim_name=data.get("date_dim_name", "date"),
    )


def _register_event_additive_effect() -> None:
    from pymc_marketing.serialization import serialization

    serialization.register(
        f"{EventAdditiveEffect.__module__}.{EventAdditiveEffect.__qualname__}",
        EventAdditiveEffect,
        deserializer=_deserialize_event_additive_effect,
    )


_register_event_additive_effect()


class DiscountedEventEffect(OptimizableMuEffect):
    r"""Promotional event effect with a discount-depth lever (price × volume).

    Each event window (e.g. Black Friday, Summer Sale) has a known date range.
    The optimizer decides *how deep* to discount for each event.  The lever is
    the **discount fraction** :math:`d_k \in [0, 1]` (e.g. ``0.20`` for a 20 %
    discount), optimized directly in native units between
    :attr:`discount_min` and :attr:`discount_max`.

    **Economics.** During event *k* the discount reprices *every* unit sold —
    baseline and incremental alike — so revenue decomposes as
    price :math:`\times` volume: a price factor :math:`(1 - d_k)` and a volume
    uplift with log-linear elasticity :math:`\beta_k \ln(1 + d_k)`.  The effect
    is expressed per link so both are units-coherent:

    * ``link="log"`` (``mu`` is log-scale) — purely additive in log space,
      no baseline access needed:

      .. math::

          \Delta\mu_t = w_{tk} \left[ \ln(1 - d_k)
                        + \beta_k \ln(1 + d_k) \right]

      so :math:`\exp(\mu)` composes it multiplicatively:
      :math:`y_t = \text{baseline}_t (1 - d_k)(1 + d_k)^{\beta_k}`.
      The optimum is event-specific,
      :math:`d_k^* = (\beta_k - 1)/(\beta_k + 1)` for :math:`\beta_k > 1`
      (else 0).

    * ``link="identity"`` — the **same multiplier** applied to the model's
      own baseline :math:`\mu^{\text{base}}_t` (everything in :math:`\mu`
      except the optimizable repricing effects themselves; see below):

      .. math::

          \Delta\mu_t = w_{tk} \, \mu^{\text{base}}_t
              \left[ (1 - d_k)(1 + d_k)^{\beta_k} - 1 \right]

    where :math:`w_{tk}` is the 0/1 event-window indicator.  Both links give
    :math:`\Delta\mu(0) = 0`, drive in-window revenue to zero at
    :math:`d_k = 1` (100 % discount), and — because the multiplier is
    identical — share the same standalone optimum
    :math:`d_k^* = (\beta_k - 1)/(\beta_k + 1)`.

    **Cost semantics.** No external cost term is imposed: the price-retention
    factor :math:`(1 - d_k)` *is* the give-back, applied to all units as part
    of the revenue response itself.  The lever is therefore optimized in
    native ``[0, 1]`` units with its own :attr:`lever_bounds` and does **not**
    enter the media budget sum constraint.

    **Optimizer integration.** The lever is the node named by
    :attr:`lever_var_name` (``f"{prefix}_data"``); ``MMM.budget_optimizer``
    translates it into an ``optimizable_vars`` entry and the optimizer
    co-optimizes it purely by name on the model graph.  Use
    ``response_variable="total_response_original_scale"`` so the discount's
    contribution (and, under the log link, its media amplification) enters the
    objective.

    **Caveats.**

    * **The lift curve is functional form, not data, away from the observed
      depth.** Each event typically has a single historical
      ``discount_pct``, so :math:`\beta_k \ln(1 + d_k)` is exactly the
      event's fitted in-window level shift re-expressed — the shape of
      :math:`(1 - d)(1 + d)^{\beta_k}` at any *other* depth follows from the
      assumed parametric form, not from observed variation in depth.  Treat
      prescribed depths as model-based extrapolations.
    * Under the identity link the baseline that gets repriced is
      ``MMM._mu_baseline`` — everything in :math:`\mu` **except** effects
      applied at or after the optimizable-effects stage.  Non-optimizable
      mu-effects (e.g. :class:`FourierEffect`, :class:`LinearTrendEffect`)
      are included; other :class:`OptimizableMuEffect` instances are not
      (two discount effects never reprice each other).  Under the log link
      everything in :math:`\mu` composes multiplicatively, including
      concurrent media.
    * Because media flows through the repricing multiplier during event
      windows, ``total_media_contribution_original_scale`` no longer
      captures media's full marginal effect when this effect is present —
      use ``total_response_original_scale`` as the optimization objective.
    * Event windows that share a **model date** are rejected at build time
      under the identity link (both events would reprice the same
      :math:`\mu^{\text{base}}_t`); under the log link shared dates simply
      sum in log space, i.e. the multipliers stack multiplicatively.
    * Under the log link, ``discount_max`` (and any historical
      ``discount_pct``) must be strictly below 1: :math:`\ln(1 - d)` is
      :math:`-\infty` at :math:`d = 1`.
    * For events outside the training window, :math:`\beta_k` is prior-driven
      (there are no in-sample dates to inform it) — treat prescriptions for
      such events as prior predictive, not fitted.
    * Per-event contributions are registered as
      ``f"{prefix}_event_contribution"``; an original-scale version exists
      only after
      ``mmm.add_original_scale_contribution_variable([..., f"{prefix}_event_contribution"])``.

    Parameters
    ----------
    df_events : pd.DataFrame
        One row per promotional event with columns:

        * ``name``         — unique event identifier (model coord).
        * ``start_date``   — first active date (inclusive).
        * ``end_date``     — last active date (inclusive).
        * ``discount_pct`` — historical discount fraction ∈ [0, 1] (optional;
          defaults to 0).  Used to initialise the data variable during fitting.
    prefix : str
        Prefix for all PyMC variables registered by this effect.  The per-event
        beta variable is named ``f"{prefix}_beta"``.
    beta_prior : Prior, optional
        Prior for the per-event lift coefficient :math:`\beta_k`.  Defaults to
        ``Prior("HalfNormal", sigma=1)``.  ``dims`` is set to ``(prefix,)``.
    date_dim_name : str, optional
        Name of the date coordinate in the PyMC model.  Defaults to ``"date"``.
    discount_min : float, optional
        Minimum allowed discount fraction (0–1).  Lower bound of the optimized
        lever.  Defaults to ``0.0``.
    discount_max : float, optional
        Maximum allowed discount fraction (0–1).  Upper bound of the optimized
        lever.  Defaults to ``1.0``.

    Examples
    --------
    .. code-block:: python

        import pandas as pd
        from pymc_marketing.mmm import MMM, DiscountedEventEffect

        df_events = pd.DataFrame(
            {
                "name": ["black_friday", "summer_sale"],
                "start_date": ["2024-11-29", "2024-12-02"],
                "end_date": ["2024-12-02", "2024-07-07"],
                "discount_pct": [0.30, 0.20],
            }
        )

        effect = DiscountedEventEffect(df_events=df_events, prefix="promo")

        mmm = MMM(...)
        mmm.add_mu_effect(effect)
        mmm.build_model(X, y)
    """

    df_events: InstanceOf[pd.DataFrame]
    prefix: str
    beta_prior: InstanceOf[Prior] = Field(
        default_factory=lambda: Prior("HalfNormal", sigma=1)
    )
    date_dim_name: str = "date"
    discount_min: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum allowed discount fraction (0–1); lower bound of the lever.",
    )
    discount_max: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum allowed discount fraction (0–1); upper bound of the lever.",
    )

    def model_post_init(self, context: Any, /) -> None:
        """Validate required columns and discount-bound consistency."""
        required = {"name", "start_date", "end_date"}
        if missing := required.difference(self.df_events.columns):
            raise ValueError(
                f"df_events is missing required columns: {missing}. "
                f"Got: {list(self.df_events.columns)}"
            )

        if self.discount_min > self.discount_max:
            raise ValueError(
                f"discount_min ({self.discount_min}) must be <= discount_max "
                f"({self.discount_max})."
            )

        if "discount_pct" in self.df_events.columns:
            pct_vals = self.df_events["discount_pct"].fillna(0.0)
            if (pct_vals < 0).any() or (pct_vals > 1).any():
                raise ValueError(
                    "df_events['discount_pct'] must be in [0, 1]. Got values "
                    f"outside range: {pct_vals[~pct_vals.between(0, 1)].tolist()}"
                )

    # ------------------------------------------------------------------
    # OptimizableMuEffect contract -- the lever is the f"{prefix}_data" node;
    # only the native bounds are effect-specific.
    # ------------------------------------------------------------------

    @property
    def lever_bounds(self) -> list[tuple[float | None, float | None]]:
        """Per-event ``(discount_min, discount_max)`` bounds, in native units."""
        n_events = len(self.df_events)
        return [(float(self.discount_min), float(self.discount_max))] * n_events

    # ------------------------------------------------------------------
    # MuEffect contract
    # ------------------------------------------------------------------

    @staticmethod
    def _events_sharing_model_dates(window: xr.DataArray) -> list[str]:
        """Return event names active on at least one shared model date.

        Calendar-overlapping windows that resolve to disjoint model dates
        (e.g. short windows on weekly data) are fine; only events whose
        resolved 0/1 masks are simultaneously active on some model date are
        returned.
        """
        prefix_dim = window.dims[-1]
        shared = window.where(window.sum(prefix_dim) > 1, 0.0)
        return [
            str(name)
            for name in window.coords[prefix_dim].values
            if float(shared.sel({prefix_dim: name}).sum()) > 0
        ]

    def _window_mask(self, dates) -> xr.DataArray:
        """Return a ``(date, prefix)`` 0/1 event-window indicator DataArray.

        Entry ``(t, k)`` is ``1.0`` when ``dates[t]`` falls within event ``k``'s
        ``[start_date, end_date]`` window (inclusive), else ``0.0``.  Both the
        dates and the windows are known at build time, so the mask is a static
        data node — no per-date symbolic arithmetic or reference anchor needed.
        The ``date`` dim is positional (no coord) so it aligns with the target.
        """
        d = pd.DatetimeIndex(dates).to_numpy()[:, None]
        starts = pd.to_datetime(self.df_events["start_date"]).to_numpy()[None, :]
        ends = pd.to_datetime(self.df_events["end_date"]).to_numpy()[None, :]
        mask = ((d >= starts) & (d <= ends)).astype("float64")
        return xr.DataArray(
            mask,
            dims=(self.date_dim_name, self.prefix),
            coords={self.prefix: self.df_events["name"].to_numpy()},
        )

    def create_data(self, mmm: Model) -> None:
        """Register the event-window indicator and discount-fraction lever nodes.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.

        Raises
        ------
        ValueError
            Under the identity link, if two event windows are active on the
            same model date (both would reprice the same baseline —
            double-counting; only ``link="log"`` composes shared dates).
            Under the log link, if ``discount_max`` or any historical
            ``discount_pct`` is >= 1 (``ln(1 - d)`` is ``-inf`` at ``d=1``).
        """
        link = getattr(mmm, "link", LinkFunction.IDENTITY)

        if link == LinkFunction.LOG:
            pct_vals = (
                self.df_events.get(
                    "discount_pct", pd.Series(0.0, index=self.df_events.index)
                )
                .fillna(0.0)
                .to_numpy(dtype="float64")
            )
            if self.discount_max >= 1.0 or (pct_vals >= 1.0).any():
                raise ValueError(
                    f"DiscountedEventEffect('{self.prefix}') with link='log': "
                    "discount_max and all historical discount_pct values must "
                    "be strictly below 1 (ln(1 - d) is -inf at d = 1)."
                )

        model: pm.Model = mmm.model

        model_dates = _get_datetime_coords(
            model.coords[self.date_dim_name], self.date_dim_name
        )
        model.add_coord(self.prefix, self.df_events["name"].to_numpy())

        window = self._window_mask(model_dates)
        if link != LinkFunction.LOG and (
            shared := self._events_sharing_model_dates(window)
        ):
            raise ValueError(
                f"DiscountedEventEffect('{self.prefix}'): events {shared} are "
                "active on the same model date, which double-counts the "
                "baseline repricing under the identity link. Merge the "
                'overlapping events, or use link="log", where shared dates '
                "stack multiplicatively."
            )

        pmd.Data(
            f"{self.prefix}_window",
            window,
            dims=(self.date_dim_name, self.prefix),
        )
        # Discount fraction per event -- the lever node, static (prefix,). The
        # window mask restricts where it contributes; the optimizer substitutes
        # this node by name (an `optimizable_vars` entry).
        discount_per_event = xr.DataArray(
            self.df_events.get(
                "discount_pct", pd.Series(0.0, index=self.df_events.index)
            )
            .fillna(0.0)
            .to_numpy(dtype="float64"),
            dims=(self.prefix,),
            coords={self.prefix: self.df_events["name"].to_numpy()},
        )
        pmd.Data(
            f"{self.prefix}_data",
            discount_per_event,
            dims=(self.prefix,),
        )

    def create_effect(self, mmm: Model) -> XTensorVariable:
        r"""Build the discount contribution via the price × volume revenue model.

        Link-aware (see the class docstring for the economics):

        .. math::

            \Delta\mu_t = \begin{cases}
                \sum_k w_{tk} \left[ \ln(1 - d_k)
                    + \beta_k \ln(1 + d_k) \right]
                    & \text{link} = \log \\
                \mu^{\text{base}}_t \sum_k w_{tk}
                    \left[ (1 - d_k)(1 + d_k)^{\beta_k} - 1 \right]
                    & \text{link} = \text{identity}
            \end{cases}

        Returns
        -------
        XTensorVariable
            Dims ``("date", *mmm.dims)`` — the additive contribution to the
            model mean.

        Raises
        ------
        ValueError
            Under the identity link, if the MMM instance does not expose the
            symbolic baseline ``_mu_baseline`` (set by ``MMM.build_model``
            just before the mu-effects loop).
        """
        model: pm.Model = mmm.model

        window_mask = model[f"{self.prefix}_window"]  # (date, prefix), 0/1

        # The lever node itself, static (prefix,). During optimization the
        # BudgetOptimizer substitutes it with this variable's segment of the
        # flat decision vector (an `optimizable_vars` entry).
        discount_pct = model[f"{self.prefix}_data"]
        log_pct = ptx.math.log1p(discount_pct)  # ln(1 + d)

        beta_prior = self.beta_prior.deepcopy()
        beta_prior.dims = (self.prefix,)
        beta = beta_prior.create_variable(
            f"{self.prefix}_beta", xdist=True
        )  # (prefix,)

        link = getattr(mmm, "link", LinkFunction.IDENTITY)
        if link == LinkFunction.LOG:
            # Additive in log space: exp(mu) composes the price factor (1 - d)
            # and the volume uplift (1 + d)^beta multiplicatively.
            lift = ptx.math.log1p(-discount_pct) + beta * log_pct  # (prefix,)
            contributions = lift * window_mask  # (date, prefix)
        else:
            mu_baseline = getattr(mmm, "_mu_baseline", None)
            if mu_baseline is None:
                raise ValueError(
                    f"DiscountedEventEffect('{self.prefix}') under the identity "
                    "link needs the model's symbolic baseline, but the MMM "
                    "instance has no '_mu_baseline' attribute. Use a standard "
                    "MMM (which sets it in build_model), or link='log' (which "
                    "needs no baseline)."
                )
            # Exact repricing multiplier on the baseline:
            # (1-d)(1+d)^beta - 1 -- identical to the log link's factor, so
            # both links share the same standalone optimum. Zero at d=0 and
            # -1 (in-window revenue -> 0) at d=1.
            multiplier = (1.0 - discount_pct) * ptx.math.exp(beta * log_pct) - 1.0
            # (date, prefix, *mmm.dims): baseline repriced on event dates only.
            contributions = mu_baseline * (multiplier * window_mask)

        # Per-event contributions, tracked for attribution.
        pmd.Deterministic(
            f"{self.prefix}_event_contribution",
            contributions.transpose(self.date_dim_name, ...),
        )

        signal = contributions.sum(dim=self.prefix)  # (date, *mmm.dims)
        return pmd.Deterministic(
            self.contribution_var_name,
            signal,
        )

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
        """Recompute the event-window indicator for a new date range.

        ``f"{prefix}_data"`` is left untouched: it stays at its historical
        values for plain posterior predictive; write optimised depths onto a
        model copy with ``pm.set_data`` to sample under an optimal plan.
        """
        new_dates = _get_datetime_coords(
            model.coords[self.date_dim_name], self.date_dim_name
        )
        pm.set_data(
            {f"{self.prefix}_window": self._window_mask(new_dates)},
            model=model,
        )

    def idata_groups(self) -> dict[str, xr.Dataset]:
        """Return ``df_events`` as a supplementary idata group."""
        return {
            f"supplementary_data_{self.prefix}": xr.Dataset.from_dataframe(
                self.df_events.reset_index(drop=True)
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dict (``df_events`` stored via idata group)."""
        return {
            "prefix": self.prefix,
            "beta_prior": self.beta_prior.to_dict(),
            "date_dim_name": self.date_dim_name,
            "discount_min": self.discount_min,
            "discount_max": self.discount_max,
            "df_events_group": f"supplementary_data_{self.prefix}",
        }


def _deserialize_discounted_event_effect(
    data: dict[str, Any],
    context: Any,
) -> "DiscountedEventEffect":
    """Deserialize a DiscountedEventEffect, reading df_events from an idata group."""
    from pymc_marketing.serialization import SerializationError

    group_name = data["df_events_group"]

    if context is None or context.idata is None:
        raise SerializationError(
            f"Cannot deserialize DiscountedEventEffect: no InferenceData "
            f"provided. The df_events DataFrame is stored in idata group "
            f"'{group_name}' and requires a DeserializationContext with idata."
        )

    try:
        ds = context.idata[group_name]
        if hasattr(ds, "dataset"):
            ds = ds.dataset
        df_events = ds.to_dataframe().reset_index()
    except (KeyError, AttributeError) as e:
        raise SerializationError(
            f"Cannot read supplementary data group '{group_name}' from "
            f"InferenceData: {e}"
        ) from e

    beta_prior_data = data.get("beta_prior")
    if beta_prior_data is not None:
        beta_prior = Prior.from_dict(beta_prior_data)
    else:
        beta_prior = Prior("HalfNormal", sigma=1)

    return DiscountedEventEffect(
        df_events=df_events,
        prefix=data["prefix"],
        beta_prior=beta_prior,
        date_dim_name=data.get("date_dim_name", "date"),
        discount_min=data.get("discount_min", 0.0),
        discount_max=data.get("discount_max", 1.0),
    )


def _register_discounted_event_effect() -> None:
    from pymc_marketing.serialization import serialization

    serialization.register(
        f"{DiscountedEventEffect.__module__}.{DiscountedEventEffect.__qualname__}",
        DiscountedEventEffect,
        deserializer=_deserialize_discounted_event_effect,
    )


_register_discounted_event_effect()
