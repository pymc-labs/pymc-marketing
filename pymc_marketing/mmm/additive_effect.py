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

import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.xtensor as ptx
import xarray as xr
from pydantic import Field, InstanceOf, PrivateAttr
from pymc_extras.prior import Prior
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm.events import EventEffect, days_from_reference
from pymc_marketing.mmm.fourier import FourierBase
from pymc_marketing.mmm.linear_trend import LinearTrend
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
    """Abstract base class for spend-driven mu effects that participate in budget optimization.

    Extends :class:`MuEffect` with the contract required by
    :class:`~pymc_marketing.mmm.budget_optimizer.BudgetOptimizer` to include
    this effect's channels in the joint budget allocation problem.

    The optimizer discovers optimizable effects at construction time via
    ``isinstance(effect, OptimizableMuEffect)``, reads their channel names
    from the PyMC model coordinate ``pymc_model.coords[effect.budget_dim]``
    (populated by :meth:`create_data`), then calls
    :meth:`replace_for_optimization` to inject spend-derived tensors into the
    PyMC model graph via ``pm.do()``.

    Subclasses must implement :meth:`replace_for_optimization` and the three
    :class:`MuEffect` abstract methods.  The conversion from spend to native
    model units (e.g. impressions → reach / frequency) belongs inside
    :meth:`replace_for_optimization`; the optimizer's flat variable is always
    in monetary units (spend) so that all channel types share a common currency
    and the total-budget constraint is meaningful across the full portfolio.

    :attr:`budget_dim` defaults to the subclass ``prefix`` field (same pattern
    as :attr:`~MuEffect.contribution_var_name`); subclasses without a ``prefix``
    must override it explicitly.

    .. note::
        **Normalized-scale contract**: the optimizer works entirely in the
        *model's internal (normalized) scale*.
        ``total_media_contribution_original_scale`` is already in the original
        scale; effect contributions (e.g. ``promo_effect_contribution``) are
        in the normalized scale and are multiplied by ``target_scale`` before
        being added to the objective.  Subclasses do **not** need to rescale
        inside ``replace_for_optimization``; the optimizer handles the
        conversion automatically.

    Notes
    -----
    :class:`FourierEffect`, :class:`LinearTrendEffect`, and
    :class:`EventAdditiveEffect` are **not** spend-driven and therefore do not
    inherit this class.  Their ``pm.Data`` nodes are frozen during optimization,
    not replaced by free variables.

    Examples
    --------
    Minimal subclass skeleton::

        class MySpendEffect(OptimizableMuEffect):
            prefix: str = "my_effect"
            cost_per_unit: float

            # budget_dim defaults to self.prefix → "my_effect"

            def replace_for_optimization(
                self,
                budget_slice,  # XTensorVariable with dims=(budget_dim,)
                num_periods,
                budget_distribution,
            ) -> dict[str, XTensorVariable]:
                impressions = budget_slice / self.cost_per_unit
                return {"my_channel_data": impressions}

            # ... plus the three MuEffect abstract methods
    """

    @property
    def budget_dim(self) -> str:
        """Name of the PyMC model dimension that indexes this effect's budget channels.

        The optimizer uses this to wrap the symbolic budget slice into an
        :class:`~pytensor.xtensor.type.XTensorVariable` with the correct dim
        before passing it to :meth:`replace_for_optimization`, so that XTensor
        broadcasting works correctly inside the implementation.

        Defaults to the subclass ``prefix`` field.  Subclasses without a
        ``prefix`` attribute must override this property.

        Returns
        -------
        str
            Dimension name (e.g. ``"promo"``, ``"rf_channel"``).
        """
        prefix = getattr(self, "prefix", None)
        if prefix is None:
            raise NotImplementedError(
                f"{type(self).__name__} must define 'budget_dim'."
            )
        return prefix

    @abstractmethod
    def replace_for_optimization(
        self,
        budget_slice: XTensorVariable,
        num_periods: int,
        budget_distribution: XTensorVariable | None,
    ) -> dict[str, XTensorVariable]:
        """Return model-graph replacements for ``pm.do()`` injection.

        Called by the budget optimizer after the standard ``channel_data``
        node has already been replaced.  The returned dict is merged with
        replacements from other optimizable effects and passed to a single
        ``pm.do()`` call.

        Parameters
        ----------
        budget_slice : XTensorVariable
            Tensor of shape ``(len(optimizable_channel_names),)`` with
            ``dims=(budget_dim,)``, holding this effect's spend values in
            monetary units.  This is a symbolic slice of the optimizer's
            combined flat budget variable — gradients flow through it back
            to the optimizer.
        num_periods : int
            Number of optimization periods (dates).
        budget_distribution : XTensorVariable or None
            Optional time-distribution weights of shape ``(num_periods,)``.
            When ``None`` spend is distributed uniformly across periods.

        Returns
        -------
        dict[str, XTensorVariable]
            Mapping of ``{pm_data_variable_name: replacement_tensor}`` to be
            passed to ``pm.do()``.  Each replacement tensor must be compatible
            in shape and dtype with the ``pm.Data`` node it replaces.
        """

    @property
    @abstractmethod
    def budget_channel_names(self) -> list[str]:
        """Names of the budget channels contributed by this effect.

        Must match the coordinate values registered under :attr:`budget_dim`
        inside :meth:`create_data` (i.e. the same list as
        ``pymc_model.coords[self.budget_dim]`` after the model is built).
        Used by
        :meth:`~pymc_marketing.mmm.mmm.BudgetOptimizerWrapper.sample_response_distribution`
        to route each per-channel budget back to the effect for posterior
        predictive sampling.

        Returns
        -------
        list[str]
            Ordered list of channel/event names, consistent with
            ``pymc_model.coords[self.budget_dim]``.
        """

    @abstractmethod
    def set_budget_for_sampling(
        self,
        budget_per_item: npt.NDArray,
        model: pm.Model,
    ) -> None:
        """Update PyMC shared data to reflect *budget_per_item* for sampling.

        Called by
        :meth:`~pymc_marketing.mmm.mmm.BudgetOptimizerWrapper.sample_response_distribution`
        to apply the optimised budget to the *original* model before it is
        cloned inside ``sample_posterior_predictive``.  The clone inherits the
        updated shared-variable values, so sampling sees the right parameters
        without any changes to :class:`~pymc_marketing.mmm.mmm.MMM`.

        Parameters
        ----------
        budget_per_item : ndarray of float, shape (n,)
            Monetary budget per channel/event, in the same order as
            :attr:`budget_channel_names`.
        model : pm.Model
            The PyMC model on which ``pm.set_data`` should be called.
        """

    @property
    def channel_contribution_var_name(self) -> str | None:
        """Name of the per-channel contribution ``pmd.Deterministic``, or ``None``.

        When not ``None``,
        :meth:`~pymc_marketing.mmm.mmm.BudgetOptimizerWrapper.sample_response_distribution`
        automatically includes the corresponding ``*_original_scale`` variable in
        ``var_names`` (if :meth:`~pymc_marketing.mmm.mmm.MMM.add_original_scale_contribution_variable`
        has been called for it), enabling attribution plots for effect channels
        alongside media channels.

        The default is ``None`` (no per-channel tracking).  Subclasses that
        register a per-channel contribution ``pmd.Deterministic`` in
        :meth:`create_effect` should override this property.

        Returns
        -------
        str or None
        """
        return None


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


class DiscountedEventEffect(OptimizableMuEffect):
    r"""Promotional event effect with a log-log discount-depth lever.

    Each event window (e.g. Black Friday, Summer Sale) has a known date range.
    The optimizer decides *how deep* to discount for each event.  Deeper
    discounts drive larger revenue lifts, with diminishing returns captured by
    the :math:`\ln(1+d)` transformation, but shallower margins captured by the
    :math:`(1-d)` price-retention factor and the :math:`-d \cdot r_k` baseline
    cost term.

    The lever is the **discount percentage** :math:`d_k \in [0, 1]` (e.g.
    ``0.20`` for a 20 % discount).  The model is a full revenue-retention
    specification:

    .. math::

        \text{lift}_k = \beta_k \cdot \ln(1 + d_k) \cdot (1 - d_k) - d_k \cdot r_k

        \text{contribution}_t = \sum_k \text{lift}_k
                                 \cdot \mathbf{1}[t \in W_k]

    :math:`\beta_k` is a per-event positive scalar inferred from the data.
    The :math:`\ln(1 + d_k) \cdot (1 - d_k)` term is hump-shaped: the
    logarithm captures volume uplift (more customers shop during deeper
    discounts) while :math:`(1-d_k)` captures price retention (at 100 %
    discount the business collects nothing per unit sold).  The additional
    :math:`-d_k \cdot r_k` term deducts the margin forgone on the *existing*
    customer base: a fraction :math:`d_k` of the average baseline per-period
    revenue :math:`r_k` is sacrificed to fund the discount.

    Together these produce a hump-shaped curve with an interior optimum whose
    depth shifts with :math:`r_k / \beta_k`:

    * At :math:`d = 0`: no discount, zero lift.
    * At :math:`d = 1`: 100 % discount — the :math:`\beta` term vanishes and
      the :math:`-r_k` cost dominates, driving total per-period revenue to
      approximately zero (baseline ≈ :math:`r_k` by construction).

    The optimizer shares a single budget pool across media channels and
    promotional events.  Each euro allocated to event :math:`k` becomes
    ``discount_pct = budget / event_revenue``, where ``event_revenue`` is
    automatically computed as the sum of the observed target ``y`` over the
    event window.  The budget constraint carries the cost; there is no
    separate cost term in the forward model.

    Parameters
    ----------
    df_events : pd.DataFrame
        One row per promotional event with columns:

        * ``name``            — unique event identifier (model coord).
        * ``start_date``      — first date the promotion is active (inclusive).
        * ``end_date``        — last date the promotion is active (inclusive).
        * ``discount_pct``    — historical discount percentage ∈ [0, 1] (e.g.
          ``0.20`` for 20 % off).  Used to initialise the data variable during
          fitting.  Defaults to zero when absent.
    prefix : str
        Prefix for all PyMC variables registered by this effect.  The
        per-event beta variable is named ``f"{prefix}_beta"``.
    beta_prior : Prior, optional
        Prior distribution for the per-event lift coefficient :math:`\beta_k`.
        Defaults to ``Prior("HalfNormal", sigma=1)``.  The ``dims`` argument
        is set automatically to ``(prefix,)`` so each event gets its own
        independent draw.
    date_dim_name : str
        Name of the date coordinate in the PyMC model.  Defaults to
        ``"date"``.
    discount_min : float, optional
        Minimum allowed discount fraction (0–1).  Budget lower bound per
        event is ``discount_min × event_revenue``.  Defaults to ``0.0``
        (no lower restriction).
    discount_max : float, optional
        Maximum allowed discount fraction (0–1).  Budget upper bound per
        event is ``discount_max × event_revenue``.  Defaults to ``1.0``
        (full event revenue).  Set this to, e.g., ``0.35`` to cap the
        optimizer at a 35 % discount across all events.

    Examples
    --------
    Model two promotional events and optimise discount depth:

    .. code-block:: python

        import pandas as pd
        from pymc_marketing.mmm import MMM, DiscountedEventEffect

        df_events = pd.DataFrame(
            {
                "name": ["black_friday", "summer_sale"],
                "start_date": ["2024-11-29", "2024-07-01"],
                "end_date": ["2024-12-02", "2024-07-07"],
                "discount_pct": [0.30, 0.20],
            }
        )

        effect = DiscountedEventEffect(df_events=df_events, prefix="promo")

        mmm = MMM(...)
        mmm.add_mu_effect(effect)
        mmm.build_model(X, y)

    Notes
    -----
    During inference ``discount_pct`` (from ``df_events``) is registered as a
    :func:`pymc.dims.Data` node so it can be swapped during optimisation.
    ``event_revenue`` and ``revenue_per_period`` are computed automatically by
    summing (or averaging) the observed target ``y`` over each event's date
    window.  ``revenue_per_period`` is normalised by ``target_scale`` and
    registered as a :func:`pymc.dims.Data` node so it participates correctly
    in the model's internal (scaled) computation graph.

    The window membership matrix ``(date × event)`` is computed from
    start/end dates at build time and stored as a :func:`pymc.dims.Data` node
    so it updates correctly during posterior predictive on new date ranges.
    """

    df_events: InstanceOf[pd.DataFrame]
    prefix: str
    beta_prior: InstanceOf[Prior] = Field(
        default_factory=lambda: Prior("HalfNormal", sigma=1)
    )
    date_dim_name: str = "date"
    reference_date: str = "2025-01-01"
    discount_min: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum allowed discount fraction (0–1). "
            "Budget lower bound per event is ``discount_min × event_revenue``."
        ),
    )
    discount_max: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum allowed discount fraction (0–1). "
            "Budget upper bound per event is ``discount_max × event_revenue``."
        ),
    )

    _event_revenue: npt.NDArray[np.float64] = PrivateAttr(default=None)
    _revenue_per_period: npt.NDArray[np.float64] = PrivateAttr(default=None)

    def model_post_init(self, context: Any, /) -> None:
        """Validate required columns and discount bound consistency."""
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
                    "df_events['discount_pct'] must be in [0, 1]. "
                    f"Got values outside range: {pct_vals[~pct_vals.between(0, 1)].tolist()}"
                )

    # ------------------------------------------------------------------
    # OptimizableMuEffect contract
    # ------------------------------------------------------------------

    @property
    def budget_bounds(self) -> list[tuple[float, float]]:
        """Per-event bounds for the monetary discount budget.

        Lower bound is ``discount_min × event_revenue``; upper bound is
        ``discount_max × event_revenue``.  Requires :meth:`create_data` to
        have been called (i.e. the MMM model must have been built first).
        """
        if self._event_revenue is None:
            raise RuntimeError(
                "event_revenue has not been computed yet. "
                "Call mmm.build_model(X, y) before accessing budget_bounds."
            )
        return [
            (self.discount_min * float(rev), self.discount_max * float(rev))
            for rev in self._event_revenue
        ]

    def replace_for_optimization(
        self,
        budget_slice: XTensorVariable,
        num_periods: int,
        budget_distribution: XTensorVariable | None,
    ) -> dict[str, XTensorVariable]:
        """Convert a monetary budget slice to a discount percentage.

        The optimizer allocates monetary amounts (€) per event.  This method
        divides by ``event_revenue`` to recover the implied discount percentage,
        which is what the model data variable ``{prefix}_discount_pct`` expects.

        Parameters
        ----------
        budget_slice : XTensorVariable
            Shape ``(n_events,)`` with ``dims=(budget_dim,)``.  Each entry is
            the monetary discount budget allocated to that event (same currency
            as ``y`` and media channels).
        num_periods : int
            Number of optimisation periods (unused — discount is event-level).
        budget_distribution : XTensorVariable or None
            Unused — discount is allocated per event, not per period.

        Returns
        -------
        dict
            ``{f"{prefix}_discount_pct": pct}`` where ``pct`` has
            ``dims=(prefix,)`` and values bounded to ``[0, 1]`` by the
            optimizer's ``budget_bounds``.
        """
        event_revenue = ptx.as_xtensor(
            self._event_revenue,
            dims=(self.prefix,),
        )
        return {
            f"{self.prefix}_discount_pct": budget_slice
            / ptx.math.switch(event_revenue > 0, event_revenue, 1.0)
        }

    # ------------------------------------------------------------------
    # MuEffect contract
    # ------------------------------------------------------------------

    def _compute_event_revenue(
        self, mmm: Model
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute per-event revenue and revenue-per-period from the observed target.

        Sums ``y`` over each event's date window to obtain the total event
        revenue, then divides by the number of in-sample periods covered by
        the window to obtain the average revenue per period.

        Parameters
        ----------
        mmm : Model
            The MMM instance; must expose ``y``, ``X``, and ``date_column``
            attributes (set by :meth:`~MMM.build_model`).

        Returns
        -------
        event_revenue : ndarray of float64, shape (n_events,)
            Total observed target ``y`` summed over each event window.
        revenue_per_period : ndarray of float64, shape (n_events,)
            ``event_revenue / n_periods_k``, where ``n_periods_k`` is the
            number of in-sample dates that fall within event *k*'s window.
            Zero for events with no in-sample dates.
        """
        y = getattr(mmm, "y", None)
        X = getattr(mmm, "X", None)
        date_column = getattr(mmm, "date_column", None)

        if y is None or X is None or date_column is None:
            raise ValueError(
                "Cannot compute event_revenue: the MMM instance does not "
                "expose 'y', 'X', or 'date_column'. "
                "Use a standard MMM instance and call build_model(X, y) first."
            )

        dates = pd.to_datetime(X[date_column])
        y_arr = np.asarray(y)
        revenues = []
        per_period = []
        for _, row in self.df_events.iterrows():
            mask = (dates >= pd.Timestamp(row["start_date"])) & (
                dates <= pd.Timestamp(row["end_date"])
            )
            ev_rev = float(y_arr[mask.values].sum())
            n_periods = int(mask.sum())
            revenues.append(ev_rev)
            per_period.append(ev_rev / n_periods if n_periods > 0 else 0.0)
        return np.array(revenues, dtype="float64"), np.array(
            per_period, dtype="float64"
        )

    def create_data(self, mmm: Model) -> None:
        """Register date offsets, discount percentage, and revenue-per-period as ``pmd.Data`` nodes.

        Also computes and caches per-event revenue and per-period revenue from
        the observed target ``y`` by summing over each event's date window.

        ``revenue_per_period`` is normalized by ``target_scale`` so it lives
        in the same (model-internal) scale as ``beta`` and the predicted mean.

        Follows the same pattern as :class:`EventAdditiveEffect`: a shared
        ``days`` node (created once per model) plus per-effect ``start_diff``
        and ``end_diff`` offsets drive the window membership symbolically
        inside :meth:`create_effect`, so no separate ``window_mask`` array
        is needed.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.
        """
        self._event_revenue, self._revenue_per_period = self._compute_event_revenue(mmm)

        # Normalise to model-internal scale so the beta*(-d*r_k) term is on the
        # same scale as the rest of the linear predictor.  We access .scalers
        # via getattr because the Model protocol does not expose it (it is an
        # MMM-specific attribute), keeping the protocol clean.
        _scalers = getattr(mmm, "scalers", None)
        target_scale = float(_scalers._target) if _scalers is not None else 1.0
        r_per_period_normalized = self._revenue_per_period / (
            target_scale if target_scale > 0 else 1.0
        )

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
            days_from_reference(
                pd.to_datetime(self.df_events["start_date"]), self.reference_date
            ),
            dims=(self.prefix,),
        )
        pmd.Data(
            f"{self.prefix}_end_diff",
            days_from_reference(
                pd.to_datetime(self.df_events["end_date"]), self.reference_date
            ),
            dims=(self.prefix,),
        )

        pmd.Data(
            f"{self.prefix}_discount_pct",
            self.df_events.get(
                "discount_pct", pd.Series(0.0, index=self.df_events.index)
            )
            .fillna(0.0)
            .to_numpy(dtype="float64"),
            dims=(self.prefix,),
        )

        pmd.Data(
            f"{self.prefix}_revenue_per_period",
            r_per_period_normalized.astype("float64"),
            dims=(self.prefix,),
        )

    def create_effect(self, mmm: Model) -> XTensorVariable:
        r"""Build the discount contribution via the full revenue-retention model.

        The contribution is :math:`\beta_k \cdot \ln(1+d_k) \cdot (1-d_k) - d_k \cdot r_k`:

        * :math:`\ln(1+d_k)` — volume uplift (log-linear elasticity).
        * :math:`(1-d_k)` — price retention; at 100 % discount the business
          collects nothing, so the :math:`\beta` term returns to zero.
        * :math:`-d_k \cdot r_k` — cost of discounting existing customers:
          a fraction :math:`d_k` of the baseline per-period revenue :math:`r_k`
          is forgone as margin.  At :math:`d=1` this exactly cancels the
          remaining baseline, driving total revenue per period to zero.

        .. math::

            \text{lift}_k = \beta_k \cdot \ln(1 + d_k) \cdot (1 - d_k)
                            - d_k \cdot r_k

        where :math:`r_k = \text{event\_revenue}_k / n\_periods_k` is the
        average observed revenue per in-sample period during event *k*,
        normalised to the model's internal (scaled) units.

        The function has a hump shape whose peak shifts with :math:`r_k / \beta_k`:
        events with low :math:`r_k / \beta_k` favour deeper discounts; events
        with high :math:`r_k / \beta_k` favour shallower discounts.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.

        Returns
        -------
        XTensorVariable
            Shape ``(date,)`` — the additive contribution to the model mean.
        """
        model: pm.Model = mmm.model

        days = model["days"]  # (date,)
        start_ref = days - model[f"{self.prefix}_start_diff"]  # (date, prefix)
        end_ref = days - model[f"{self.prefix}_end_diff"]  # (date, prefix)

        # Boolean window: event k is active on date t when it has started and
        # not yet ended.  XTensor broadcasts (date,) x (prefix,) -> (date, prefix).
        window_mask = ptx.math.cast(
            (start_ref >= 0) & (end_ref <= 0), "float64"
        )  # (date, prefix)

        discount_pct = model[f"{self.prefix}_discount_pct"]  # (prefix,)

        # Log-log transformation: ln(1 + d) maps [0, 1] → [0, ln(2)] ≈ [0, 0.69]
        log_pct = ptx.math.log1p(discount_pct)  # (prefix,)

        # Per-event lift coefficient — one beta per event via dims
        beta_prior = self.beta_prior.deepcopy()
        beta_prior.dims = (self.prefix,)
        beta = beta_prior.create_variable(f"{self.prefix}_beta", xdist=True)

        # Average revenue per period in model-internal (normalized) scale.
        # Registered as a pmd.Data node in create_data.
        r_per_period = model[f"{self.prefix}_revenue_per_period"]  # (prefix,)

        # Full revenue-retention lift:
        #   beta*ln(1+d)*(1-d)  -- hump-shaped volume x price-retention term
        #   -d*r_k              -- cost of discounting existing customers (margin forgone)
        # At d=0: lift=0.  At d=1: beta*ln(2)*0 - r_k = -r_k (revenue -> 0).
        lift = (
            beta * log_pct * (1.0 - discount_pct) - discount_pct * r_per_period
        )  # (prefix,)

        # Per-event contributions (date, prefix): tracked for posterior predictive
        # attribution.  Dim name matches self.prefix so the coord values are the
        # event names (e.g. ["back_to_school", "black_friday", "summer_sale"]).
        contributions = lift * window_mask  # (date, prefix) — computed once
        pmd.Deterministic(
            self.channel_contribution_var_name,
            contributions,
        )

        # Sum active lifts across events for each date: (date,)
        signal = contributions.sum(dim=self.prefix)

        return pmd.Deterministic(
            self.contribution_var_name,
            signal,
        )

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
        """Update the shared ``days`` node for a new date range.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.
        model : pm.Model
            The (possibly cloned) PyMC model.
        X : xr.Dataset
            The new dataset (unused directly; dates come from model coords).
        """
        new_dates = _get_datetime_coords(
            model.coords[self.date_dim_name], self.date_dim_name
        )
        pm.set_data(
            {"days": days_from_reference(new_dates, self.reference_date)},
            model=model,
        )
        # discount_pct stays at historical values for plain posterior predictive;
        # BudgetOptimizerWrapper overrides it via set_budget_for_sampling when
        # a budget allocation is available.

    # ------------------------------------------------------------------
    # OptimizableMuEffect sampling contract
    # ------------------------------------------------------------------

    @property
    def budget_channel_names(self) -> list[str]:
        """Event names, consistent with the ``prefix`` coord set in ``create_data``.

        Returns
        -------
        list[str]
            One entry per promotional event, in ``df_events`` row order.
        """
        return self.df_events["name"].tolist()

    def set_budget_for_sampling(
        self,
        budget_per_item: npt.NDArray,
        model: pm.Model,
    ) -> None:
        """Convert per-event monetary budgets to ``discount_pct`` and inject into model.

        Mirrors the symbolic conversion in :meth:`replace_for_optimization` but
        operates on concrete values so that ``pm.sample_posterior_predictive``
        sees the optimised discount levels rather than the historical ones.

        Parameters
        ----------
        budget_per_item : ndarray of float, shape (n_events,)
            Monetary budget per event (same order as :attr:`budget_channel_names`).
        model : pm.Model
            The PyMC model on which ``pm.set_data`` will be called.
        """
        key = f"{self.prefix}_discount_pct"
        safe_revenue = np.where(self._event_revenue > 0, self._event_revenue, 1.0)
        discount_pct = budget_per_item / safe_revenue
        pm.set_data(
            {key: discount_pct.astype("float64")},
            model=model,
        )

    @property
    def channel_contribution_var_name(self) -> str:
        """Name of the per-event contribution ``pmd.Deterministic``.

        Registered by :meth:`create_effect` as ``f"{self.prefix}_channel_contribution"``.
        Enables :meth:`~pymc_marketing.mmm.mmm.BudgetOptimizerWrapper.sample_response_distribution`
        to include per-event attribution in the returned dataset when
        :meth:`~pymc_marketing.mmm.mmm.MMM.add_original_scale_contribution_variable`
        has been called for this variable.

        Returns
        -------
        str
            Variable name, e.g. ``"events_channel_contribution"``.
        """
        return f"{self.prefix}_channel_contribution"

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
            "reference_date": self.reference_date,
            "beta_prior": self.beta_prior.to_dict(),
            "date_dim_name": self.date_dim_name,
            "discount_min": self.discount_min,
            "discount_max": self.discount_max,
            "df_events_group": f"supplementary_data_{self.prefix}",
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
        df_events = ds.to_dataframe().reset_index()
    except (KeyError, AttributeError) as e:
        raise SerializationError(
            f"Cannot read supplementary data group '{group_name}' from "
            f"InferenceData: {e}"
        ) from e

    beta_prior_data = data.get("beta_prior")
    if beta_prior_data is not None:
        from pymc_extras.prior import Prior as _Prior

        beta_prior = _Prior.from_dict(beta_prior_data)
    else:
        beta_prior = Prior("HalfNormal", sigma=1)

    return DiscountedEventEffect(
        df_events=df_events,
        prefix=data["prefix"],
        beta_prior=beta_prior,
        reference_date=data.get("reference_date", "2025-01-01"),
        date_dim_name=data.get("date_dim_name", "date"),
        discount_min=data.get("discount_min", 0.0),
        discount_max=data.get("discount_max", 1.0),
    )


def _register_event_additive_effect() -> None:
    from pymc_marketing.serialization import serialization

    serialization.register(
        f"{EventAdditiveEffect.__module__}.{EventAdditiveEffect.__qualname__}",
        EventAdditiveEffect,
        deserializer=_deserialize_event_additive_effect,
    )


def _register_discounted_event_effect() -> None:
    from pymc_marketing.serialization import serialization

    serialization.register(
        f"{DiscountedEventEffect.__module__}.{DiscountedEventEffect.__qualname__}",
        DiscountedEventEffect,
        deserializer=_deserialize_discounted_event_effect,
    )


_register_event_additive_effect()
_register_discounted_event_effect()
