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
r"""Budget model: a control function for demand-chasing media spend.

An MMM regresses sales on spend. When budgets are set with an eye on demand
the analyst cannot see -- a planner's unrecorded forecast, say -- spend and the
sales error share a cause, and the history credits each channel with the demand
its budget followed. No observed adjustment set closes that backdoor path.

:class:`BudgetModelEffect` models how budgets are set, jointly with sales. Each
channel's spend gets its own equation on the observed demand drivers, and
whatever those drivers do not explain is the channel's *budget surprise*
:math:`v_{c,t}`:

.. math::

    \text{spend}_{c,t} = \underbrace{a_c + \psi_c^\top z_t
        + \phi_c^\top f_t}_{m_{c,t}} + v_{c,t},
    \qquad v_{c,t} \sim \mathcal{N}(0, s_c^2)

where :math:`z_t` are the (standardised) MMM controls and :math:`f_t` the same
annual Fourier basis the sales equation uses. The sales mean then gains a
control-function term

.. math::

    \mu_t = \text{MMM}_t + \sum_c \gamma_c\, v_{c,t},

which carries the demand information that rides along with a budget surprise,
so the channel response no longer has to absorb it [1]_ [2]_ [3]_ [4]_.

The control function removes confounding but uses up variation: once
:math:`v_{c,t}` enters the sales equation, what is left in spend to identify the
response curve is :math:`m_{c,t}`, built from the same drivers that already
enter sales. The response is then identified only by functional form (the
nonlinearity of saturation and adstock). A lift test supplies the missing
excluded variation. Its spend change is set by design, independently of
demand, so it is subtracted *before* the surprise is computed and the
designed change never counts as a surprise. The test periods then enter the
sales likelihood as ordinary data with their own counterfactual. Use
:func:`lift_test_design` to describe the experiments.

What may enter the spend equation
---------------------------------
Any regressor in the spend equation that is *not* in the sales equation acts
as excluded variation (an instrument) for the response curve, and is only
valid if it moves spend without affecting sales in any other way. The
defaults therefore keep the spend equation's regressors inside the sales
equation's: the MMM's controls, and a Fourier basis of the same order as the
MMM's yearly seasonality. Seasonal spend is the textbook case: if sales had no
seasonal terms, seasonal budget moves would become an instrument and the
response curve would be identified from seasonal demand, the very confounding
this component exists to remove. Variables that genuinely satisfy the
exclusion restriction, such as media-cost shocks or an internal budget
calendar, can be passed as ``instruments``. A recorded demand forecast is
*not* an instrument: it predicts sales directly, and belongs in the MMM's
``control_columns``, where both equations see it.

Designs act on contemporaneous spend. With adstock, the first weeks of a test
still carry stock from the endogenous weeks before it, and the weeks after it
carry the designed change, so tests should last several adstock half-lives.
A ``"shift"`` design assumes the recorded change was fully realised; if spend
hit a floor or a minimum commitment, record the realised change or use
``"set"``.

Interventions are never surprises
---------------------------------
The surprise is computed from a stored, factual copy of chosen spend, not from
``channel_data``. Counterfactuals that intervene on ``channel_data`` -- the
budget optimizer, :class:`~pymc_marketing.mmm.incrementality.Incrementality`,
or :meth:`MMM.sample_posterior_predictive` on modified spend -- therefore move
the media response but hold the control-function term at the demand that
actually happened. On dates outside the training data the surprise is zero
by default, its expectation for spend that has not yet been chosen; see
``surprise_out_of_sample`` for scoring held-out weeks whose spend was chosen.

Exogeneity diagnostic
---------------------
:math:`\gamma_c = 0` is the exogenous-spend special case, so the posterior of
:math:`\gamma_c` is a control-function check of exogeneity in the spirit of
Durbin-Wu-Hausman [2]_ [3]_: it measures how much weight the assumption was
carrying. See :meth:`BudgetModelEffect.exogeneity_summary`. Two caveats apply.

* The check is conditional on a correctly specified sales equation. The
  surprise is part of spend, so :math:`\gamma_c v_{c,t}` and the channel's
  response curve compete for the same variation, and any misspecification of
  the curve (adstock form, ``l_max``, saturation family, a missing trend)
  leaks into :math:`\gamma_c` as a linear correction.
* It has power only where there is excluded variation. Without a lift-test
  design for a channel, :math:`\gamma_c` is identified by functional form
  and the priors alone, can lean away from zero when spend is exogenous, and
  should not be read as a test.

Spend and sales are fit jointly, which is efficient when both equations are
correctly specified, but lets the sales likelihood pull the spend equation
towards surprises that fit sales residuals when they are not. Comparing the
spend-equation posterior with a fit that fixes :math:`\gamma` near zero is a
cheap check.

References
----------
.. [1] Manchanda, P., Rossi, P. E., & Chintagunta, P. K. (2004). Response
   modeling with nonrandom marketing-mix variables. *Journal of Marketing
   Research*, 41(4), 467-478.
.. [2] Rivers, D., & Vuong, Q. H. (1988). Limited information estimators and
   exogeneity tests for simultaneous probit models. *Journal of Econometrics*,
   39(3), 347-366.
.. [3] Wooldridge, J. M. (2015). Control function methods in applied
   econometrics. *Journal of Human Resources*, 50(2), 420-445.
.. [4] Petrin, A., & Train, K. (2010). A control function approach to
   endogeneity in consumer choice models. *Journal of Marketing Research*,
   47(1), 3-13.

Examples
--------
A four-week TV cut of 15 spend units per week relative to business as usual,
and a two-week geo holdout for Digital:

.. code-block:: python

    import pandas as pd

    from pymc_marketing.mmm import MMM, GeometricAdstock, MichaelisMentenSaturation
    from pymc_marketing.mmm.budget_model import BudgetModelEffect

    design = pd.DataFrame(
        {
            "channel": ["tv", "digital"],
            "start_date": ["2024-01-01", "2024-03-04"],
            "end_date": ["2024-01-22", "2024-03-11"],
            "mode": ["shift", "set"],
            "delta_x": [-15.0, None],
        }
    )

    mmm = MMM(
        date_column="date",
        channel_columns=["tv", "digital"],
        control_columns=["inflation", "unemployment"],
        yearly_seasonality=4,
        adstock=GeometricAdstock(l_max=8),
        saturation=MichaelisMentenSaturation(),
    )
    budget = BudgetModelEffect(design=design, surprise_lags=1)
    mmm.add_mu_effect(budget)
    mmm.fit(X, y)

    budget.exogeneity_summary(mmm)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.xtensor as ptx
import xarray as xr
from pydantic import Field, InstanceOf, field_validator
from pymc_extras.prior import Prior, VariableFactory
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm.additive_effect import (
    FourierEffect,
    LinearTrendEffect,
    Model,
    MuEffect,
    _get_datetime_coords,
)
from pymc_marketing.mmm.fourier import (
    DAYS_IN_YEAR,
    YearlyFourier,
    generate_fourier_modes,
)
from pymc_marketing.serialization import serialization

__all__ = ["BudgetModelEffect", "exogeneity_summary", "lift_test_design"]

DesignMode = Literal["shift", "set"]

_DESIGN_REQUIRED_COLUMNS = ("channel", "start_date", "end_date")


class FittedModel(Model, Protocol):
    """The parts of a fitted MMM that :func:`exogeneity_summary` reads."""

    @property
    def idata(self) -> xr.DataTree | None:
        """The inference data, with a posterior once the model is fitted."""

    @property
    def scalers(self) -> xr.Dataset:
        """Scales of the target and channels."""

    @property
    def mu_effects(self) -> list[MuEffect]:
        """The additive effects the model was built with."""


def lift_test_design(
    df: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex | npt.ArrayLike,
    channels: list[str],
    dim_coords: dict[str, Any] | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Convert a table of lift-test designs into spend-design arrays.

    Each row describes one experiment on one channel (and one cell of any extra
    model dimension) over the inclusive window ``[start_date, end_date]``.

    Two kinds of design are supported:

    ``"shift"``
        Spend was moved by a known amount ``delta_x`` per period *relative to
        what would otherwise have been chosen* (e.g. "business as usual minus
        15"). The chosen spend in those periods is still demand-driven, so it
        is recovered as ``observed - delta_x`` and keeps entering the budget
        model; only the designed change is excluded from the surprise.
    ``"set"``
        Spend was *set* by the design, as in a holdout at zero or a fixed test
        level. The chosen spend is then unobserved but also irrelevant: spend
        in those cells carries no demand information, so the surprise and the
        spend likelihood are masked out there.

    Parameters
    ----------
    df : pd.DataFrame
        One row per experiment cell with columns:

        * ``channel``: channel name, one of ``channels``.
        * one column per key of ``dim_coords`` (e.g. ``geo``).
        * ``start_date``, ``end_date``: inclusive window of the design. A
          window may run past either end of ``dates`` (a test still running at
          the data cutoff, say); only the model dates inside it are used.
        * ``mode`` (optional): ``"shift"`` (default) or ``"set"``.
        * ``delta_x``: per-period spend change in original units. Required
          for ``"shift"`` rows, ignored for ``"set"`` rows.
    dates : pd.DatetimeIndex or array-like
        Model dates.
    channels : list[str]
        Channels, in the order the arrays should use.
    dim_coords : dict[str, array-like], optional
        Coordinates of any extra model dimensions, e.g. ``{"geo": [...]}``.

    Returns
    -------
    design_shift : xr.DataArray
        Dims ``("date", *dim_coords, "channel")``. Known designed spend change,
        zero outside ``"shift"`` windows.
    design_holdout : xr.DataArray
        Boolean, same dims. ``True`` inside ``"set"`` windows.

    Raises
    ------
    ValueError
        If a column is missing, a channel or dimension value is unknown, a
        window is reversed or contains no model dates, a ``"shift"`` row has
        no ``delta_x``, or two windows overlap on the same cell.
    """
    dim_coords = dict(dim_coords or {})
    dims = tuple(dim_coords)
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))

    coords: dict[str, Any] = {"date": dates, **dim_coords, "channel": list(channels)}
    shape = tuple(len(v) for v in coords.values())
    shift = xr.DataArray(np.zeros(shape), dims=tuple(coords), coords=coords)
    holdout = xr.DataArray(
        np.zeros(shape, dtype=bool), dims=tuple(coords), coords=coords
    )
    touched = xr.zeros_like(holdout)

    missing = [c for c in (*_DESIGN_REQUIRED_COLUMNS, *dims) if c not in df.columns]
    if missing:
        raise ValueError(f"Design is missing required columns: {missing}.")

    for idx, row in df.reset_index(drop=True).iterrows():
        mode = row.get("mode", "shift")
        if pd.isna(mode):
            mode = "shift"
        if mode not in ("shift", "set"):
            raise ValueError(f"Row {idx}: mode must be 'shift' or 'set', got {mode!r}.")
        if row["channel"] not in channels:
            raise ValueError(
                f"Row {idx}: channel {row['channel']!r} is not one of {list(channels)}."
            )
        cell: dict[str, Any] = {"channel": row["channel"]}
        for dim in dims:
            if row[dim] not in list(dim_coords[dim]):
                raise ValueError(
                    f"Row {idx}: {dim} value {row[dim]!r} is not in the model coords."
                )
            cell[dim] = row[dim]

        start, end = pd.Timestamp(row["start_date"]), pd.Timestamp(row["end_date"])
        if start > end:
            raise ValueError(f"Row {idx}: start_date {start} is after end_date {end}.")
        in_window = (dates >= start) & (dates <= end)
        if not in_window.any():
            raise ValueError(
                f"Row {idx}: window [{start.date()}, {end.date()}] contains no model dates."
            )
        window_dates = dates[in_window]

        target = {**cell, "date": window_dates}
        if touched.loc[target].any():
            raise ValueError(
                f"Row {idx}: design window overlaps an earlier row on {cell}."
            )
        touched.loc[target] = True

        if mode == "shift":
            delta_x = row.get("delta_x", np.nan)
            if pd.isna(delta_x):
                raise ValueError(f"Row {idx}: 'shift' designs require delta_x.")
            shift.loc[target] = float(delta_x)
        else:
            holdout.loc[target] = True

    return shift, holdout


def _standardise_stats(
    da: xr.DataArray, keep: str
) -> tuple[xr.DataArray, xr.DataArray]:
    """Mean and standard deviation over every dim except ``keep``."""
    reduce_dims = [d for d in da.dims if d != keep]
    mean = da.mean(reduce_dims)
    std = da.std(reduce_dims)
    std = xr.where(std > 0, std, 1.0)
    return mean, std


def _sales_fourier_order(mmm: Model) -> int:
    """Order of the annual Fourier seasonality in the sales equation."""
    orders = [int(getattr(mmm, "yearly_seasonality", None) or 0)]
    orders += [
        effect.fourier.n_order
        for effect in getattr(mmm, "mu_effects", [])
        if isinstance(effect, FourierEffect)
        and isinstance(effect.fourier, YearlyFourier)
    ]
    return max(orders)


def _has_sales_trend(mmm: Model) -> bool:
    """Whether the sales equation has a time trend of its own."""
    return bool(getattr(mmm, "time_varying_intercept", False)) or any(
        isinstance(effect, LinearTrendEffect)
        for effect in getattr(mmm, "mu_effects", [])
    )


def _date_first(values: xr.DataArray, dims: tuple[str, ...]) -> xr.DataArray:
    """Order ``values`` as the model does: ``date``, then any of ``dims`` it has."""
    return values.transpose("date", *[d for d in dims if d in values.dims])


def _time_index(
    dates: pd.DatetimeIndex, training_dates: pd.DatetimeIndex
) -> np.ndarray:
    """Days since the training start, as a fraction of the training span."""
    span = max(float((training_dates.max() - training_dates.min()).days), 1.0)
    return ((dates - training_dates.min()).days / span).to_numpy()


@dataclass(frozen=True)
class _TrainingData:
    """What the effect derives from the MMM's training data.

    It is recomputed from the MMM whenever it is needed rather than cached on
    the effect, so one instance can serve several models, and copies and
    pickles of a fitted MMM keep working.
    """

    # Observed spend in original units, dims ("date", *dims, "channel").
    spend: xr.DataArray
    # Designed spend change (zero outside "shift" windows), same dims.
    shift: xr.DataArray
    # True where a "set" design fixed spend, same dims.
    holdout: xr.DataArray
    # Largest absolute chosen spend per channel and cell, dims (*dims, "channel").
    spend_scale: xr.DataArray
    fourier_order: int
    sales_fourier_order: int
    # Mean and standard deviation of each control, or None without controls.
    control_stats: tuple[xr.DataArray, xr.DataArray] | None
    # Mean and standard deviation of each instrument.
    instrument_stats: dict[str, tuple[float, float]]

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.spend.coords["date"].values)

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(str(d) for d in self.spend.dims if d not in ("date", "channel"))

    @property
    def channels(self) -> list[str]:
        return list(self.spend.coords["channel"].values)


class BudgetModelEffect(MuEffect):
    r"""Joint budget model with a control-function correction for endogenous spend.

    Adds a spend equation for each modelled channel and the control-function
    term :math:`\sum_c \gamma_c v_{c,t}` to the MMM mean. See the module
    docstring for the model and its identification logic.

    Parameters
    ----------
    prefix : str, default "budget"
        Prefix for every variable, data node and coordinate this effect
        creates. The channel coordinate is ``f"{prefix}_channel"``, so custom
        priors must use that dim name rather than ``"channel"``.
    channels : list[str], optional
        Channels whose budgets are modelled. Defaults to all MMM channels.
    use_controls : bool, default True
        Whether the spend equations depend on the MMM's control columns.
        Controls are standardised with their training mean and standard
        deviation before entering the spend equation.
    fourier_order : int, optional
        Order of the annual Fourier basis in the spend equations. Defaults to
        the order of the sales equation's annual seasonality, from the MMM's
        ``yearly_seasonality`` or a ``FourierEffect`` with a ``YearlyFourier``
        (``0`` when it has none), so the spend equation's seasonal terms are
        also in the sales equation. A higher order warns: the extra terms
        would act as instruments.
    trend : bool, default False
        Add a linear time trend to the spend equations, for budgets that grow
        over the years. Enable it only if the sales equation also has a trend
        (a :class:`~pymc_marketing.mmm.additive_effect.LinearTrendEffect` or a
        time-varying intercept); otherwise the trend is an instrument, and the
        effect warns. A trend column among the MMM's controls is already in
        both equations and needs no ``trend=True``.
    instruments : list[str], optional
        Variables that move budgets but, by assumption, affect sales only
        through spend: columns of a ``pd.DataFrame`` ``X`` (or data variables
        of an ``xr.Dataset``), such as media-cost shocks or an internal budget
        calendar. They enter the spend equation only, so they must satisfy
        the exclusion restriction. Recorded demand proxies (a demand forecast)
        and recorded outcomes (lagged sales) do not; put demand proxies in the
        MMM's ``control_columns`` instead. Each must vary over ``date`` and
        have no dims beyond the model's own. Standardised like the controls.
        Needed for prediction dates only with
        ``surprise_out_of_sample="observed"``.
    design : pd.DataFrame, optional
        Lift-test designs, in the format of :func:`lift_test_design`, with no
        other columns (the design is saved with the model). Without a design
        the effect is identified by functional form only. Rows whose window
        contains none of the model dates (e.g. in an early cross-validation
        fold), or whose channel is not modelled, are dropped with a warning.
    surprise_lags : int, default 0
        Number of lagged surprises in the control function, which becomes
        :math:`\sum_c \sum_{l=0}^{L} \gamma_{c,l} v_{c,t-l}`. Persistent demand
        shocks carry information from past budget surprises into this week's
        sales, which a contemporaneous control function cannot absorb.
        Lagged surprises compete with the adstock carryover of past spend, so
        use a small number. The lagged coefficients are stored as
        ``f"{prefix}_gamma_lag"`` and reported by :meth:`exogeneity_summary`.
        Each lag stores a dense date-by-date shift matrix in the model and in
        saved files: negligible for weekly data, about 17 MB per lag for four
        years of daily data. On new dates the lagged terms see only surprises
        within the prediction data; predict with
        ``include_last_observations=True`` to carry the last training
        surprises into the first forecast periods.
    surprise_out_of_sample : {"zero", "observed"}, default "zero"
        Surprise on dates outside the training data. ``"zero"``, its
        expectation, suits scenario planning and forecasting future spend.
        ``"observed"`` computes the surprise from the spend in the prediction
        data, which suits evaluating held-out weeks whose spend was actually
        chosen: with ``"zero"`` the budget model forgoes the demand
        information that realised spend carries, and scores worse than a plain
        MMM on such weeks even when it is closer to the causal truth. Any
        ``design`` rows on those dates are applied as in training (shifts
        subtracted, holdouts masked), and ``instruments`` must be supplied.
        This is a setting of the model, so keep a separate model with the
        default for scenarios; :meth:`MMM.budget_optimizer` warns when it
        runs on a model with ``"observed"``.
    spend_intercept_prior : Prior, optional
        Prior for the spend-equation intercept. Spend is divided by its
        per-channel maximum, and the defaults are weakly informative on that
        scale.
    spend_control_prior : Prior, optional
        Prior for the spend-equation coefficients on the standardised controls.
    spend_fourier_prior : Prior, optional
        Prior for the spend-equation Fourier coefficients.
    spend_trend_prior : Prior, optional
        Prior for the spend-equation trend coefficient (per unit of the
        training period).
    spend_instrument_prior : Prior, optional
        Prior for the spend-equation coefficients on the standardised
        instruments.
    spend_sigma_prior : Prior, optional
        Prior for the scale of the budget surprise.
    gamma_prior : Prior, optional
        Prior for the control-function coefficient, on the MMM's scaled-target
        per scaled-spend scale.
    gamma_lag_prior : Prior, optional
        Prior for the lagged control-function coefficients, with dims
        ``(f"{prefix}_lag", f"{prefix}_channel")`` by default.

    Notes
    -----
    * With extra model dims (e.g. ``geo``) the defaults give each cell its own
      spend intercept and surprise scale, but pool the control, Fourier,
      trend and instrument coefficients and ``gamma`` across cells, so there is one
      :math:`\gamma_c` per channel. Pass priors with the extra dims to unpool
      them. Custom priors may only use the dims ``f"{prefix}_channel"``, the
      model's extra dims, and ``"control"`` or ``f"{prefix}_fourier"`` for the
      control and Fourier coefficients.
    * The spend equation is Gaussian. Flighted channels with many zero weeks
      violate it badly; the effect warns when more than 20% of a modelled
      channel's weeks are zero.
    * The spend equations add a second observed variable,
      ``f"{prefix}_spend"`` (scaled chosen spend, masked to zero with unit
      scale in holdout cells and out of sample). It is sampled alongside
      ``y`` in posterior predictive checks; pass ``var_name="y"`` to model
      comparison tools that read the pointwise log-likelihood.
      ``f"{prefix}_spend_mu"`` and ``f"{prefix}_surprise"`` are stored as
      Deterministics.
    * The control function is linear in the surprises. With the default
      ``surprise_lags=0`` it is also contemporaneous: it assumes this period's
      surprise carries the demand information relevant to this period's
      sales, which persistent demand shocks strain.
    * If lift-test periods enter here as data, do not also pass their summary
      to :meth:`MMM.add_lift_test_measurements`; that counts the experiment
      twice. The lift likelihood remains appropriate for experiments whose
      periods or units are *not* in the MMM data.
    * With ``surprise_out_of_sample="zero"``, out-of-sample predictive
      intervals omit the variance the control function absorbed in-sample,
      roughly :math:`\sum_c \gamma_c^2 s_c^2`, and are somewhat too narrow.
    * With ``link="log"`` the control-function term is additive on the log
      scale.
    """

    prefix: str = "budget"
    channels: list[str] | None = None
    use_controls: bool = True
    fourier_order: int | None = Field(None, ge=0)
    trend: bool = False
    instruments: list[str] = Field(default_factory=list)
    design: InstanceOf[pd.DataFrame] | None = None
    surprise_lags: int = Field(0, ge=0)
    surprise_out_of_sample: Literal["zero", "observed"] = "zero"
    spend_intercept_prior: VariableFactory | None = None
    spend_control_prior: VariableFactory | None = None
    spend_fourier_prior: VariableFactory | None = None
    spend_trend_prior: VariableFactory | None = None
    spend_instrument_prior: VariableFactory | None = None
    spend_sigma_prior: VariableFactory | None = None
    gamma_prior: VariableFactory | None = None
    gamma_lag_prior: VariableFactory | None = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("design")
    @classmethod
    def _validate_design_columns(
        cls, value: pd.DataFrame | None
    ) -> pd.DataFrame | None:
        if value is None:
            return value
        missing = [c for c in _DESIGN_REQUIRED_COLUMNS if c not in value.columns]
        if missing:
            raise ValueError(f"design is missing required columns: {missing}.")
        value = value.reset_index(drop=True).copy()
        for col in ("start_date", "end_date"):
            value[col] = pd.to_datetime(value[col])
        return value

    # ------------------------------------------------------------------ names
    @property
    def channel_dim(self) -> str:
        """Name of the coordinate indexing the modelled channels."""
        return f"{self.prefix}_channel"

    @property
    def fourier_dim(self) -> str:
        """Name of the coordinate indexing the Fourier basis."""
        return f"{self.prefix}_fourier"

    @property
    def lag_dim(self) -> str:
        """Name of the coordinate indexing lagged surprises."""
        return f"{self.prefix}_lag"

    @property
    def source_date_dim(self) -> str:
        """Name of the coordinate the lag operator shifts from."""
        return f"{self.prefix}_source_date"

    @property
    def data_vars(self) -> list[str]:
        """Dataset variables this effect reads (the instruments).

        Exposed so MMM keeps these columns when converting a ``pd.DataFrame``
        ``X``, and so helpers such as ``create_zero_dataset`` carry them into
        prediction and optimization datasets.
        """
        return list(self.instruments)

    # ----------------------------------------------------------- priors
    def _default_priors(self, dims: tuple[str, ...]) -> dict[str, VariableFactory]:
        ch = self.channel_dim
        return {
            "spend_intercept": Prior("Normal", mu=0.5, sigma=0.5, dims=(*dims, ch)),
            "spend_control": Prior("Normal", mu=0, sigma=0.2, dims=("control", ch)),
            "spend_fourier": Prior(
                "Normal", mu=0, sigma=0.2, dims=(self.fourier_dim, ch)
            ),
            "spend_trend": Prior("Normal", mu=0, sigma=0.5, dims=(ch,)),
            "spend_instrument": Prior("Normal", mu=0, sigma=0.2, dims=(ch,)),
            "spend_sigma": Prior("HalfNormal", sigma=0.2, dims=(*dims, ch)),
            "gamma": Prior("Normal", mu=0, sigma=0.5, dims=(ch,)),
            "gamma_lag": Prior("Normal", mu=0, sigma=0.5, dims=(self.lag_dim, ch)),
        }

    def _prior(self, name: str, dims: tuple[str, ...]) -> VariableFactory:
        """Return the user's prior for ``name``, checked, or the default."""
        user = getattr(self, f"{name}_prior")
        if user is None:
            return self._default_priors(dims)[name]
        user_dims = getattr(user, "dims", None) or ()
        user_dims = (user_dims,) if isinstance(user_dims, str) else tuple(user_dims)
        extra = {
            "spend_control": {"control"},
            "spend_fourier": {self.fourier_dim},
            "gamma_lag": {self.lag_dim},
        }
        allowed = {self.channel_dim, *dims, *extra.get(name, set())}
        if not set(user_dims) <= allowed:
            raise ValueError(
                f"{name}_prior has dims {user_dims}, but only {sorted(allowed)} are "
                f"allowed. This effect indexes channels by {self.channel_dim!r}, not "
                "'channel'."
            )
        return user

    # ------------------------------------------------------------- data
    def _training_data(self, mmm: Model, warn: bool = False) -> _TrainingData:
        """Derive spend, designs and standardisation from the MMM's training data.

        ``warn`` reports dropped design rows. It is set only when the model is
        built, so the warning is not repeated on every prediction.
        """
        dataset = mmm.xarray_dataset
        all_channels = list(dataset["_channel"].coords["channel"].values)
        channels = all_channels if self.channels is None else list(self.channels)
        unknown = sorted(set(channels) - set(all_channels))
        if unknown:
            raise ValueError(f"Unknown channels in BudgetModelEffect: {unknown}.")
        spend = (
            dataset["_channel"]
            .sel(channel=channels)
            .transpose("date", *mmm.dims, "channel")
            .astype(float)
        )
        shift, holdout = self._design_for(spend, warn=warn)
        spend_scale = abs((spend - shift).where(~holdout, 0.0)).max("date")

        control_stats = None
        if self.use_controls and "_control" in dataset:
            control_stats = _standardise_stats(dataset["_control"], keep="control")
        instrument_stats = {}
        for name in self.instruments:
            values = self._instrument(mmm, name)
            std = float(values.std())
            instrument_stats[name] = (float(values.mean()), std if std > 0 else 1.0)

        sales_order = _sales_fourier_order(mmm)
        return _TrainingData(
            spend=spend,
            shift=shift,
            holdout=holdout,
            spend_scale=xr.where(spend_scale > 0, spend_scale, 1.0),
            fourier_order=(
                sales_order if self.fourier_order is None else self.fourier_order
            ),
            sales_fourier_order=sales_order,
            control_stats=control_stats,
            instrument_stats=instrument_stats,
        )

    def _design_for(
        self, spend: xr.DataArray, warn: bool = False
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Design shift and holdout arrays aligned with ``spend``.

        Rows whose window contains none of ``spend``'s dates, or whose channel
        is not modelled, say nothing about this data and are dropped.
        """
        if self.design is None:
            zeros = xr.zeros_like(spend)
            return zeros, zeros.astype(bool)
        dates = pd.DatetimeIndex(spend.coords["date"].values)
        channels = list(spend.coords["channel"].values)
        in_window = [
            bool(((dates >= start) & (dates <= end)).any())
            for start, end in zip(
                self.design["start_date"], self.design["end_date"], strict=True
            )
        ]
        keep = self.design["channel"].isin(channels) & np.array(in_window)
        if warn and not keep.all():
            warnings.warn(
                f"Dropping {int((~keep).sum())} BudgetModelEffect design row(s) whose "
                "window contains none of the model dates or whose channel is not "
                "modelled.",
                UserWarning,
                stacklevel=4,
            )
        dim_coords = {
            d: list(spend.coords[d].values)
            for d in spend.dims
            if d not in ("date", "channel")
        }
        shift, holdout = lift_test_design(
            self.design[keep], dates=dates, channels=channels, dim_coords=dim_coords
        )
        return (
            shift.transpose(*spend.dims).assign_coords(spend.coords),
            holdout.transpose(*spend.dims).assign_coords(spend.coords),
        )

    def _surprise_inputs(
        self,
        spend: xr.DataArray,
        shift: xr.DataArray,
        holdout: xr.DataArray,
        spend_scale: xr.DataArray,
    ) -> xr.Dataset:
        """Scaled chosen spend and the activity mask behind the surprise.

        The designed change is removed from spend, and cells whose spend the
        design set are masked, so only budget choices become surprises.
        """
        return xr.Dataset(
            {
                "chosen": (spend - shift) / spend_scale,
                "active": (~holdout).astype(float),
            }
        ).rename({"channel": self.channel_dim})

    def _instrument(self, mmm: Model, name: str) -> xr.DataArray:
        """Return an instrument from the training data, checked against the model dims."""
        if name not in mmm.xarray_dataset:
            raise ValueError(
                f"Instrument {name!r} is not in the training data. Add it as a "
                "column of X."
            )
        values = mmm.xarray_dataset[name]
        if "date" not in values.dims:
            raise ValueError(f"Instrument {name!r} must have a 'date' dim.")
        extra = sorted(set(values.dims) - {"date", *mmm.dims})
        if extra:
            raise ValueError(
                f"Instrument {name!r} has dims {extra} beyond the model's "
                f"{('date', *mmm.dims)}. Aggregate it before passing it."
            )
        return _date_first(values, mmm.dims)

    def _check_design(self, mmm: Model) -> None:
        """Reject design columns the model cannot save and channels it lacks."""
        if self.design is None:
            return
        allowed = {*_DESIGN_REQUIRED_COLUMNS, "mode", "delta_x", *mmm.dims}
        extra = sorted(set(self.design.columns) - allowed)
        if extra:
            raise ValueError(
                f"design has columns {extra} the effect does not use. The design is "
                "saved with the model, so drop them."
            )
        mmm_channels = set(mmm.xarray_dataset["_channel"].coords["channel"].values)
        unknown = sorted(set(self.design["channel"]) - mmm_channels)
        if unknown:
            raise ValueError(f"design has channels {unknown} the MMM does not have.")

    def _warn_about_spend(self, data: _TrainingData) -> None:
        """Warn about spend the Gaussian spend equation or a shift design misreads."""
        in_shift = data.shift != 0
        chosen = data.spend - data.shift
        if bool((in_shift & ((data.spend <= 0) | (chosen < 0))).any()):
            warnings.warn(
                "Some 'shift' design cells have zero observed spend, or negative "
                "spend once delta_x is removed, so the designed change may not have "
                "been fully realised. Record the realised change or use mode='set' "
                "for those cells.",
                UserWarning,
                stacklevel=3,
            )
        zero_share = (data.spend.where(~data.holdout) == 0).mean("date")
        flighted = [
            str(channel)
            for channel in data.channels
            if float(zero_share.sel(channel=channel).max()) > 0.2
        ]
        if flighted:
            warnings.warn(
                f"Channels {flighted} have zero spend in more than 20% of weeks. The "
                "Gaussian spend equation is a poor fit for flighted channels.",
                UserWarning,
                stacklevel=3,
            )

    def _warn_about_instruments(self, mmm: Model, data: _TrainingData) -> None:
        """Warn when a default regressor would act as an unintended instrument."""
        if data.fourier_order > data.sales_fourier_order:
            warnings.warn(
                f"The spend equation's Fourier order ({data.fourier_order}) exceeds "
                f"the sales equation's yearly seasonality "
                f"({data.sales_fourier_order}). The extra seasonal terms act as "
                "instruments, which is invalid if seasonality also moves sales.",
                UserWarning,
                stacklevel=3,
            )
        if self.trend and not _has_sales_trend(mmm):
            warnings.warn(
                "trend=True adds a trend to the spend equation, but the sales "
                "equation has none (no LinearTrendEffect or time-varying "
                "intercept), so the trend acts as an instrument. A trend column "
                "among the controls is already in both equations; drop trend=True "
                "in that case.",
                UserWarning,
                stacklevel=3,
            )

    def create_data(self, mmm: Model) -> None:
        """Register chosen spend, the activity mask, and the spend-equation data.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.
        """
        model = mmm.model
        p = self.prefix
        self._check_design(mmm)
        data = self._training_data(mmm, warn=True)
        self._warn_about_spend(data)
        self._warn_about_instruments(mmm, data)

        n_dates = len(data.dates)
        if self.surprise_lags >= n_dates:
            raise ValueError(
                f"surprise_lags={self.surprise_lags} must be smaller than the number "
                f"of dates ({n_dates})."
            )

        model.add_coord(self.channel_dim, data.channels)
        # The factual inputs to the surprise. The effect never reads channel_data,
        # so interventions on spend leave the control function unchanged.
        factual = self._surprise_inputs(
            data.spend, data.shift, data.holdout, data.spend_scale
        )
        chosen, active = factual["chosen"], factual["active"]
        pmd.Data(f"{p}_chosen_spend", chosen.values, dims=chosen.dims)
        pmd.Data(f"{p}_active", active.values, dims=active.dims)
        if self.surprise_lags > 0:
            model.add_coord(self.lag_dim, list(range(1, self.surprise_lags + 1)))
            model.add_coord(self.source_date_dim, data.dates)
            pmd.Data(
                f"{p}_lag_operator",
                self._lag_operator(n_dates),
                dims=(self.lag_dim, "date", self.source_date_dim),
            )
        if data.fourier_order > 0:
            model.add_coord(
                self.fourier_dim,
                [f"sin_{k}" for k in range(1, data.fourier_order + 1)]
                + [f"cos_{k}" for k in range(1, data.fourier_order + 1)],
            )
            pmd.Data(f"{p}_dayofyear", data.dates.dayofyear.to_numpy(), dims="date")
        if self.trend:
            pmd.Data(f"{p}_time", _time_index(data.dates, data.dates), dims="date")

        # An instrument may already be registered by another effect that reads it.
        for name in self.instruments:
            values = self._instrument(mmm, name)
            existing = model.named_vars.get(name)
            if existing is None:
                pmd.Data(name, values.values, dims=values.dims)
            elif (
                tuple(model.named_vars_to_dims.get(name, ())) != values.dims
                or tuple(existing.get_value(borrow=True).shape) != values.shape
            ):
                raise ValueError(
                    f"Cannot reuse model variable {name!r} as an instrument: its "
                    "dims or shape differ from the instrument's."
                )

    def _lag_operator(self, n_dates: int) -> np.ndarray:
        """Matrices that shift a date series down by 1, ..., ``surprise_lags``."""
        return np.stack(
            [np.eye(n_dates, k=-lag) for lag in range(1, self.surprise_lags + 1)]
        )

    # ----------------------------------------------------------- effect
    def create_effect(self, mmm: Model) -> XTensorVariable:
        """Add the spend equations, their likelihood, and the control function.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.

        Returns
        -------
        XTensorVariable
            The control-function contribution with dims ``("date", *mmm.dims)``.
        """
        model = mmm.model
        p = self.prefix
        ch = self.channel_dim
        data = self._training_data(mmm)
        dims = data.dims

        def prior(name: str, var_name: str) -> XTensorVariable:
            return self._prior(name, dims).create_variable(var_name, xdist=True)

        chosen = model[f"{p}_chosen_spend"]
        active = model[f"{p}_active"]

        # The spend equation's mean m_{c,t}.
        spend_mu = prior("spend_intercept", f"{p}_spend_intercept")
        if data.control_stats is not None and "control_data" in model.named_vars:
            mean, std = data.control_stats
            z = (model["control_data"] - as_xtensor(mean.values, dims=mean.dims)) / (
                as_xtensor(std.values, dims=std.dims)
            )
            coef = prior("spend_control", f"{p}_spend_control_coef")
            spend_mu = spend_mu + (z * coef).sum(dim="control")
        if data.fourier_order > 0:
            modes = generate_fourier_modes(
                periods=model[f"{p}_dayofyear"] / DAYS_IN_YEAR,
                n_order=data.fourier_order,
                fourier_dim=self.fourier_dim,
            )
            coef = prior("spend_fourier", f"{p}_spend_fourier_coef")
            spend_mu = spend_mu + (modes * coef).sum(dim=self.fourier_dim)
        if self.trend:
            coef = prior("spend_trend", f"{p}_spend_trend_coef")
            spend_mu = spend_mu + model[f"{p}_time"] * coef
        for name in self.instruments:
            mean, std = data.instrument_stats[name]
            coef = prior("spend_instrument", f"{p}_instrument_{name}_coef")
            spend_mu = spend_mu + (model[name] - mean) / std * coef
        # An intercept-only equation has no date dim until broadcast against spend.
        spend_mu, _ = ptx.broadcast(spend_mu, chosen)
        spend_mu = pmd.Deterministic(
            f"{p}_spend_mu", spend_mu.transpose("date", *dims, ch)
        )

        # The spend likelihood. Masked cells see observed 0 ~ Normal(0, 1): a
        # constant, so this is exactly the likelihood restricted to active cells.
        sigma = prior("spend_sigma", f"{p}_spend_sigma")
        pmd.Normal(
            f"{p}_spend",
            mu=active * spend_mu,
            sigma=active * sigma + (1 - active),
            observed=active * chosen,
        )

        # The control function sum_l gamma_l v_{t-l}.
        surprise = pmd.Deterministic(
            f"{p}_surprise", (active * (chosen - spend_mu)).transpose("date", *dims, ch)
        )
        control_function = surprise * prior("gamma", f"{p}_gamma")
        if self.surprise_lags > 0:
            gamma_lag = prior("gamma_lag", f"{p}_gamma_lag")
            # Shift with a lag operator, one lag at a time: slicing and
            # concatenating along date, or reducing over a length-one lag dim,
            # produced graphs PyTensor failed to differentiate.
            source = surprise.rename({"date": self.source_date_dim})
            operator = model[f"{p}_lag_operator"]
            for k in range(self.surprise_lags):
                lagged = ptx.dot(
                    source,
                    operator.isel({self.lag_dim: k}),
                    dim=self.source_date_dim,
                )
                control_function = control_function + lagged * gamma_lag.isel(
                    {self.lag_dim: k}
                )

        return pmd.Deterministic(
            self.contribution_var_name,
            control_function.sum(dim=ch).transpose("date", *dims),
        )

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset | None) -> None:
        """Align the factual surprise inputs with new prediction dates.

        On training dates, chosen spend and the activity mask are reindexed
        from the *training* data, never read from ``X``: changing spend in
        ``X`` is an intervention, and interventions are never budget
        surprises. On other dates the surprise is zero, or, with
        ``surprise_out_of_sample="observed"``, computed from the spend in
        ``X`` (see :meth:`_observed_surprise_inputs`). Instruments missing
        from ``X`` keep their training values on training dates and their
        training mean elsewhere, which cannot move a zero surprise; with
        ``"observed"`` they are required on new dates.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.
        model : pm.Model
            The PyMC model, whose ``date`` coordinate is already updated.
        X : xr.Dataset
            The new prediction dataset.
        """
        p = self.prefix
        if f"{p}_chosen_spend" not in model.named_vars:
            raise RuntimeError(
                f"The model has no data for BudgetModelEffect {p!r}. Build the MMM "
                "with this effect before predicting."
            )
        data = self._training_data(mmm)
        new_dates = _get_datetime_coords(model.coords["date"], "date")
        # Factual inputs on training dates; missing (NaN) on new dates.
        inputs = self._surprise_inputs(
            data.spend, data.shift, data.holdout, data.spend_scale
        ).reindex(date=new_dates)
        observed = self.surprise_out_of_sample == "observed" and bool(
            inputs["active"].isnull().any()
        )
        if observed:
            inputs = inputs.fillna(self._observed_surprise_inputs(X, data, new_dates))
        # Otherwise the surprise on new dates is zero: masked, with no spend.
        inputs = inputs.fillna(0.0)

        new_data: dict[str, Any] = {
            f"{p}_chosen_spend": inputs["chosen"].values,
            f"{p}_active": inputs["active"].values,
        }
        if data.fourier_order > 0:
            new_data[f"{p}_dayofyear"] = new_dates.dayofyear.to_numpy()
        if self.trend:
            new_data[f"{p}_time"] = _time_index(new_dates, data.dates)
        for name in self.instruments:
            if X is not None and name in X.data_vars:
                values = _date_first(X[name], mmm.dims)
            elif observed:
                raise ValueError(
                    f"Instrument {name!r} is required on new dates with "
                    "surprise_out_of_sample='observed', because it moves the spend "
                    "equation and hence the surprise."
                )
            else:
                values = (
                    self._instrument(mmm, name)
                    .reindex(date=new_dates)
                    .fillna(data.instrument_stats[name][0])
                )
            new_data[name] = values.values
        coords = None
        if self.surprise_lags > 0:
            new_data[f"{p}_lag_operator"] = self._lag_operator(len(new_dates))
            coords = {self.source_date_dim: new_dates}
        pm.set_data(new_data, coords=coords, model=model)

    def _observed_surprise_inputs(
        self, X: xr.Dataset | None, data: _TrainingData, new_dates: pd.DatetimeIndex
    ) -> xr.Dataset:
        """Surprise inputs that read spend in ``X`` as realised budget choices.

        Used on new dates with ``surprise_out_of_sample="observed"``. A
        lift-test design that falls on these dates is removed exactly as in
        training: designed changes are subtracted and designed holdouts
        masked, so a test in a held-out fold is not scored as a surprise.
        """
        if X is None or "_channel" not in X.data_vars:
            raise ValueError(
                "surprise_out_of_sample='observed' reads spend on new dates from the "
                "prediction data, but it has no channel spend."
            )
        spend = (
            X["_channel"]
            .sel(channel=data.channels)
            .reindex(date=new_dates)
            .transpose("date", *data.dims, "channel")
            .astype(float)
        )
        shift, holdout = self._design_for(spend)
        return self._surprise_inputs(spend, shift, holdout, data.spend_scale)

    # ------------------------------------------------------------ MMM hooks
    def check_scenario_use(self, mmm: Model) -> None:
        """Warn that ``"observed"`` surprises misread scenario spend.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.
        """
        if self.surprise_out_of_sample == "observed":
            warnings.warn(
                f"BudgetModelEffect {self.prefix!r} uses "
                "surprise_out_of_sample='observed', which treats spend on new dates "
                "as realised budget choices. That suits scoring held-out weeks, not "
                "budget scenarios: the scenario's spend would count as a budget "
                "surprise. Use a model with the default 'zero' for optimization.",
                UserWarning,
                stacklevel=3,
            )

    def check_lift_tests(self, mmm: Model, df_lift_test: pd.DataFrame) -> None:
        """Warn when lift tests may duplicate this effect's design.

        The design puts the test periods into the sales likelihood as data, so
        adding the lift summary of the same experiment counts it twice.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.
        df_lift_test : pd.DataFrame
            The lift tests being added.
        """
        if self.design is None:
            return
        keys = ["channel", *mmm.dims]
        tested = set(map(tuple, df_lift_test[keys].astype(str).to_numpy()))
        designed = set(map(tuple, self.design[keys].astype(str).to_numpy()))
        if overlap := sorted(tested & designed):
            warnings.warn(
                f"Lift tests on {overlap} may duplicate the design of "
                f"BudgetModelEffect {self.prefix!r}. The lift table has no dates, "
                "so this matches on channel and dims only. If these are the same "
                "experiments, their periods already enter the sales likelihood as "
                "data and the lift likelihood counts them twice. Use "
                "add_lift_test_measurements only for experiments whose periods or "
                "units are not in the MMM data.",
                UserWarning,
                stacklevel=3,
            )

    # ----------------------------------------------------- diagnostics
    def exogeneity_summary(
        self, mmm: FittedModel, interval_prob: float = 0.94
    ) -> pd.DataFrame:
        """Summarise this effect's control-function coefficients in ``mmm``.

        Equivalent to ``exogeneity_summary(mmm, prefix=self.prefix)``. The
        summary uses the configuration of the effect ``mmm`` was built with,
        so it can be called from any instance with the same prefix, including
        the original object after :meth:`MMM.load`. See
        :func:`exogeneity_summary` for the columns and how to read them.

        Parameters
        ----------
        mmm : MMM
            A fitted MMM containing an effect with this prefix.
        interval_prob : float, default 0.94
            Mass of the equal-tailed posterior interval.

        Returns
        -------
        pd.DataFrame
            One row per channel, lag and cell of any extra dimension.
        """
        return exogeneity_summary(mmm, prefix=self.prefix, interval_prob=interval_prob)

    def _prior_sd(self, mmm: Model, name: str) -> xr.DataArray:
        """Prior standard deviation of a control-function coefficient.

        Exact for a ``Normal`` prior with a fixed ``sigma``; otherwise estimated
        from seeded prior draws so the summary is reproducible. Lags are
        indexed by ``"lag"``.
        """
        factory = self._prior(name, tuple(mmm.dims))
        dims = getattr(factory, "dims", None) or ()
        dims = (dims,) if isinstance(dims, str) else tuple(dims)
        coords = {d: list(mmm.model.coords[d]) for d in dims}
        sigma = getattr(factory, "parameters", {}).get("sigma")
        if getattr(factory, "distribution", None) == "Normal" and isinstance(
            sigma, int | float
        ):
            shape = [len(values) for values in coords.values()]
            sd = xr.DataArray(np.full(shape, float(sigma)), dims=dims, coords=coords)
        else:
            pymc_logger = logging.getLogger("pymc")
            level = pymc_logger.level
            pymc_logger.setLevel(logging.WARNING)
            try:
                draws = factory.sample_prior(
                    coords=coords, name=name, draws=4000, random_seed=0
                )[name]
            finally:
                pymc_logger.setLevel(level)
            sd = draws.std([d for d in draws.dims if d in ("chain", "draw", "sample")])
        return sd.rename({self.lag_dim: "lag"}) if self.lag_dim in sd.dims else sd

    # ---------------------------------------------------- serialization
    @classmethod
    def _prior_fields(cls) -> list[str]:
        return [name for name in cls.model_fields if name.endswith("_prior")]

    def to_dict(self) -> dict[str, Any]:
        """Serialize every field to a JSON-safe dict.

        ``__type__`` is injected by the registry. Only the design and the
        priors need converting; every other field is dumped as is, so a new
        field is saved without further changes here.
        """
        priors = self._prior_fields()
        data = self.model_dump(exclude={"design", *priors})
        data["design"] = None if self.design is None else _design_to_dict(self.design)
        for name in priors:
            prior = getattr(self, name)
            data[name] = None if prior is None else prior.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetModelEffect:
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        fields = {key: value for key, value in data.items() if key in cls.model_fields}
        if fields.get("design") is not None:
            fields["design"] = pd.DataFrame(fields["design"])
        for name in cls._prior_fields():
            fields[name] = _prior_from_dict(fields.get(name))
        return cls(**fields)


def _design_to_dict(design: pd.DataFrame) -> dict[Any, list[Any]]:
    """Design columns as lists of JSON-safe values, with ISO dates and ``None`` for NaN."""
    design = design.copy()
    for col in ("start_date", "end_date"):
        design[col] = pd.to_datetime(design[col]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return design.astype(object).where(design.notna(), None).to_dict(orient="list")


def _prior_from_dict(value: dict[str, Any] | None) -> VariableFactory | None:
    from pymc_extras.deserialize import deserialize

    if value is None:
        return None
    if "__type__" in value:
        return serialization.deserialize(value)
    return deserialize(value)


def exogeneity_summary(
    mmm: FittedModel, prefix: str = "budget", interval_prob: float = 0.94
) -> pd.DataFrame:
    r"""Summarise a budget model's control-function coefficients as an exogeneity check.

    :math:`\gamma_c = 0` is the exogenous-spend case. A posterior for
    :math:`\gamma_c` concentrated away from zero says that unexplained budget
    moves carried demand, so the plain MMM's exogeneity assumption was doing
    work. This is a Bayesian analogue of the control-function
    (Durbin-Wu-Hausman) test, not a formal hypothesis test.

    The check assumes a correctly specified sales equation: because the
    surprise is part of spend, any misspecification of the response curve
    leaks into :math:`\gamma_c`. It has power only with excluded variation.
    For channels without a lift-test design, :math:`\gamma_c` rests on
    functional form and the priors, can lean away from zero when spend is
    exogenous, and the summary flags it.

    With ``surprise_lags > 0`` the summary has one row per lag. Persistent
    demand can load on a lagged coefficient while the contemporaneous one
    stays near zero, so read the rows together. More rows are also more
    chances of a false alarm: with :math:`L` lags, at least one of the
    :math:`L + 1` intervals excludes zero by chance about :math:`1 - p^{L+1}`
    of the time, where :math:`p` is ``interval_prob``.

    Parameters
    ----------
    mmm : MMM
        A fitted MMM containing a :class:`BudgetModelEffect`. The
        configuration is read from that effect.
    prefix : str, default "budget"
        Prefix of the effect to summarise.
    interval_prob : float, default 0.94
        Mass of the equal-tailed posterior interval.

    Returns
    -------
    pd.DataFrame
        One row per channel, lag (``0`` for the contemporaneous coefficient)
        and cell of any extra dimension, with columns ``gamma_mean``,
        ``gamma_lower``, ``gamma_upper`` (scaled units),
        ``gamma_per_spend_unit`` (target units per unexplained unit of spend;
        with ``link="log"``, log-scale change per unit of spend),
        ``prob_positive``, ``prior_sd``, ``posterior_sd``, ``contraction``
        (``1 - posterior_sd / prior_sd``), ``identified_by_design`` and
        ``note``.
    """
    if not 0 < interval_prob < 1:
        raise ValueError(f"interval_prob must be in (0, 1), got {interval_prob}.")
    idata = mmm.idata
    name = f"{prefix}_gamma"
    if idata is None or "posterior" not in idata or name not in idata.posterior:
        raise RuntimeError(f"No posterior for {name!r}; fit the model first.")
    effect = next(
        (
            effect
            for effect in mmm.mu_effects
            if isinstance(effect, BudgetModelEffect) and effect.prefix == prefix
        ),
        None,
    )
    if effect is None:
        raise RuntimeError(f"The MMM has no BudgetModelEffect with prefix {prefix!r}.")

    data = effect._training_data(mmm)
    rename = {"channel": effect.channel_dim}
    sample_dims = ("chain", "draw")

    # Stack the contemporaneous coefficient (lag 0) with any lagged ones.
    gammas = [idata.posterior[name].expand_dims(lag=[0])]
    prior_sds = [effect._prior_sd(mmm, "gamma").expand_dims(lag=[0])]
    if effect.surprise_lags > 0:
        gammas.append(
            idata.posterior[f"{prefix}_gamma_lag"].rename({effect.lag_dim: "lag"})
        )
        prior_sds.append(effect._prior_sd(mmm, "gamma_lag"))
    gamma = xr.concat(gammas, dim="lag")
    tail = (1 - interval_prob) / 2
    per_unit = gamma * mmm.scalers["_target"] / data.spend_scale.rename(rename)

    summary = xr.Dataset(
        {
            "gamma_mean": gamma.mean(sample_dims),
            "gamma_lower": gamma.quantile(tail, dim=sample_dims).drop_vars("quantile"),
            "gamma_upper": gamma.quantile(1 - tail, dim=sample_dims).drop_vars(
                "quantile"
            ),
            "gamma_per_spend_unit": per_unit.mean(sample_dims),
            "prob_positive": (gamma > 0).mean(sample_dims),
            "prior_sd": xr.concat(prior_sds, dim="lag"),
            "posterior_sd": gamma.std(sample_dims),
        }
    )
    summary["contraction"] = 1 - summary["posterior_sd"] / summary["prior_sd"]
    # A gamma shared across a dim is informed by a design anywhere along it.
    identified = ((data.shift != 0) | data.holdout).any("date").rename(rename)
    pooled = [d for d in identified.dims if d not in gamma.dims]
    if pooled:
        identified = identified.any(pooled)
    summary, identified = xr.broadcast(summary, identified)
    summary["identified_by_design"] = identified
    order = [effect.channel_dim, "lag"]
    summary = summary.transpose(*order, ...)

    df = summary.to_dataframe().reset_index()
    df = df[[*order, *[c for c in df.columns if c not in order]]]
    df = df.rename(columns={effect.channel_dim: "channel"})

    def _note(row: pd.Series) -> str:
        notes = []
        if not row["identified_by_design"]:
            notes.append(
                "no lift-test design: gamma rests on functional form and priors, "
                "not a test"
            )
        if row["contraction"] < 0.1:
            notes.append("data barely update the prior on gamma")
        return "; ".join(notes)

    df["note"] = df.apply(_note, axis=1)
    return df
