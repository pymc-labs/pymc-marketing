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
actually happened. On dates outside the training data the surprise is zero,
its expectation for spend that has not yet been chosen.

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
import weakref
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.xtensor as ptx
import xarray as xr
from pydantic import Field, InstanceOf, PrivateAttr, field_validator
from pymc_extras.prior import Prior, VariableFactory
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm.additive_effect import (
    LinearTrendEffect,
    Model,
    MuEffect,
    _get_datetime_coords,
)
from pymc_marketing.mmm.fourier import DAYS_IN_YEAR, generate_fourier_modes
from pymc_marketing.serialization import serialization

__all__ = ["BudgetModelEffect", "lift_test_design"]

DesignMode = Literal["shift", "set"]

_DESIGN_REQUIRED_COLUMNS = ("channel", "start_date", "end_date")


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
        the MMM's ``yearly_seasonality`` (``0`` when it has none), so the
        spend equation's seasonal terms are also in the sales equation. A
        higher order warns: the extra terms would act as instruments.
    trend : bool, default False
        Add a linear time trend to the spend equations, for budgets that grow
        over the years. Enable it only if the sales equation also has a trend
        (a :class:`~pymc_marketing.mmm.additive_effect.LinearTrendEffect` or a
        time-varying intercept); otherwise the trend is an instrument, and the
        effect warns.
    instruments : list[str], optional
        Variables that move budgets but, by assumption, affect sales only
        through spend: columns of a ``pd.DataFrame`` ``X`` (or data variables
        of an ``xr.Dataset``), such as media-cost shocks or an internal budget
        calendar. They enter the spend equation only, so they must satisfy
        the exclusion restriction. Recorded demand proxies (a demand forecast)
        and recorded outcomes (lagged sales) do not; put demand proxies in the
        MMM's ``control_columns`` instead. Each must vary over ``date`` and
        have no dims beyond the model's own. Standardised like the controls.
        Not needed for prediction dates.
    design : pd.DataFrame, optional
        Lift-test designs, in the format of :func:`lift_test_design`. Without
        a design the effect is identified by functional form only. Rows whose
        window contains none of the model dates (e.g. in an early
        cross-validation fold) are dropped with a warning.
    surprise_lags : int, default 0
        Number of lagged surprises in the control function, which becomes
        :math:`\sum_c \sum_{l=0}^{L} \gamma_{c,l} v_{c,t-l}`. Persistent demand
        shocks carry information from past budget surprises into this week's
        sales, which a contemporaneous control function cannot absorb.
        Lagged surprises compete with the adstock carryover of past spend, so
        use a small number. The lagged coefficients are stored as
        ``f"{prefix}_gamma_lag"``; :meth:`exogeneity_summary` reports the
        contemporaneous one.
    surprise_out_of_sample : {"zero", "observed"}, default "zero"
        Surprise on dates outside the training data. ``"zero"``, its
        expectation, suits scenario planning and forecasting future spend.
        ``"observed"`` computes the surprise from the spend in the prediction
        data, which suits evaluating held-out weeks whose spend was actually
        chosen: with ``"zero"`` the budget model forgoes the demand
        information that realised spend carries, and scores worse than a plain
        MMM on such weeks even when it is closer to the causal truth.
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
    * The control function is linear and contemporaneous: it assumes this
      period's surprise carries the demand information relevant to this
      period's sales. Persistent demand shocks strain that assumption.
    * If lift-test periods enter here as data, do not also pass their summary
      to :meth:`MMM.add_lift_test_measurements`; that counts the experiment
      twice. The lift likelihood remains appropriate for experiments whose
      periods or units are *not* in the MMM data.
    * With ``surprise_out_of_sample="zero"``, out-of-sample predictive
      intervals omit the variance the control function absorbed in-sample,
      roughly :math:`\sum_c \gamma_c^2 s_c^2`, and are somewhat too narrow.
    * An instance records the model it was last built for. Use one instance
      per model; after :meth:`MMM.load`, use the effect in
      ``loaded.mu_effects``.
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

    _channels: list[str] = PrivateAttr(default_factory=list)
    _dims: tuple[str, ...] = PrivateAttr(default=())
    _factual: xr.Dataset | None = PrivateAttr(default=None)
    _spend_scale: xr.DataArray | None = PrivateAttr(default=None)
    _identified_by_design: xr.DataArray | None = PrivateAttr(default=None)
    _control_stats: tuple[xr.DataArray, xr.DataArray] | None = PrivateAttr(default=None)
    _instrument_stats: dict[str, tuple[float, float]] = PrivateAttr(
        default_factory=dict
    )
    _fourier_order: int = PrivateAttr(default=0)
    _time_origin: pd.Timestamp = PrivateAttr(default_factory=lambda: pd.Timestamp(0))
    _time_span: float = PrivateAttr(default=1.0)
    _mmm_ref: Any = PrivateAttr(default=None)

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

    def _lag_operator(self, n_dates: int) -> np.ndarray:
        """Matrices that shift a date series down by 1, ..., ``surprise_lags``."""
        return np.stack(
            [np.eye(n_dates, k=-lag) for lag in range(1, self.surprise_lags + 1)]
        )

    @property
    def data_vars(self) -> list[str]:
        """Dataset variables this effect reads (the instruments).

        Exposed so MMM keeps these columns when converting a ``pd.DataFrame``
        ``X``, and so helpers such as ``create_zero_dataset`` carry them into
        prediction and optimization datasets.
        """
        return list(self.instruments)

    def _is_bound_to(self, mmm: Any) -> bool:
        return self._mmm_ref is not None and self._mmm_ref() is mmm

    def _attached(self, mmm: Any) -> BudgetModelEffect:
        """Return the effect ``mmm`` was built with, checked to belong to it."""
        attached = [
            effect
            for effect in getattr(mmm, "mu_effects", [])
            if isinstance(effect, BudgetModelEffect) and effect.prefix == self.prefix
        ]
        if not attached:
            raise RuntimeError(
                f"The MMM has no BudgetModelEffect with prefix {self.prefix!r}."
            )
        effect = attached[0]
        if effect._spend_scale is None or not effect._is_bound_to(mmm):
            raise RuntimeError(
                "The model containing this effect has not been built, or this "
                "effect instance was last built for a different MMM. Use one "
                "instance per model."
            )
        return effect

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

    def _allowed_prior_dims(self, name: str) -> set[str]:
        extra = {
            "spend_control": {"control"},
            "spend_fourier": {self.fourier_dim},
            "gamma_lag": {self.lag_dim},
        }
        return {self.channel_dim, *self._dims, *extra.get(name, set())}

    def _prior(self, name: str) -> VariableFactory:
        user = getattr(self, f"{name}_prior")
        if user is None:
            return self._default_priors(self._dims)[name]
        dims = getattr(user, "dims", None) or ()
        dims = (dims,) if isinstance(dims, str) else tuple(dims)
        allowed = self._allowed_prior_dims(name)
        if not set(dims) <= allowed:
            raise ValueError(
                f"{name}_prior has dims {dims}, but only {sorted(allowed)} are allowed. "
                f"This effect indexes channels by {self.channel_dim!r}, not 'channel'."
            )
        return user

    # ------------------------------------------------------------- data
    def _design_arrays(
        self, dates: pd.DatetimeIndex, dim_coords: dict[str, Any]
    ) -> tuple[xr.DataArray, xr.DataArray]:
        if self.design is None:
            coords: dict[str, Any] = {
                "date": dates,
                **dim_coords,
                "channel": self._channels,
            }
            shape = tuple(len(v) for v in coords.values())
            zeros = xr.DataArray(np.zeros(shape), dims=tuple(coords), coords=coords)
            return zeros, zeros.astype(bool)
        starts = pd.to_datetime(self.design["start_date"])
        ends = pd.to_datetime(self.design["end_date"])
        overlaps = [
            bool(((dates >= start) & (dates <= end)).any())
            for start, end in zip(starts, ends, strict=True)
        ]
        design = self.design.loc[overlaps]
        if len(design) < len(self.design):
            warnings.warn(
                f"Dropping {len(self.design) - len(design)} BudgetModelEffect design "
                "row(s) whose window contains none of the model dates.",
                UserWarning,
                stacklevel=3,
            )
        return lift_test_design(
            design, dates=dates, channels=self._channels, dim_coords=dim_coords
        )

    def create_data(self, mmm: Model) -> None:
        """Register chosen spend, the activity mask, and the instruments.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.
        """
        model = mmm.model
        self._mmm_ref = weakref.ref(mmm)
        self._dims = tuple(mmm.dims)
        all_channels = list(mmm.xarray_dataset["_channel"].coords["channel"].values)
        channels = list(self.channels) if self.channels is not None else all_channels
        unknown = sorted(set(channels) - set(all_channels))
        if unknown:
            raise ValueError(f"Unknown channels in BudgetModelEffect: {unknown}.")
        self._channels = channels

        dates = _get_datetime_coords(model.coords["date"], "date")
        dim_coords = {d: list(model.coords[d]) for d in self._dims}
        shift, holdout = self._design_arrays(dates, dim_coords)

        observed = (
            mmm.xarray_dataset["_channel"]
            .sel(channel=channels)
            .transpose("date", *self._dims, "channel")
            .astype(float)
        )
        shift = shift.transpose(*observed.dims).assign_coords(observed.coords)
        holdout = holdout.transpose(*observed.dims).assign_coords(observed.coords)

        if bool(((shift != 0) & (observed <= 0)).any()):
            warnings.warn(
                "Some 'shift' design cells have zero observed spend, so the designed "
                "change may not have been fully realised. Record the realised change "
                "or use mode='set' for those cells.",
                UserWarning,
                stacklevel=2,
            )
        zero_share = (observed.where(~holdout) == 0).mean("date")
        flighted = [
            str(channel)
            for channel in zero_share["channel"].values
            if float(zero_share.sel(channel=channel).max()) > 0.2
        ]
        if flighted:
            warnings.warn(
                f"Channels {flighted} have zero spend in more than 20% of weeks. The "
                "Gaussian spend equation is a poor fit for flighted channels.",
                UserWarning,
                stacklevel=2,
            )

        chosen = observed - shift
        active = ~holdout
        spend_scale = abs(chosen.where(active, 0.0)).max("date")
        spend_scale = xr.where(spend_scale > 0, spend_scale, 1.0)

        rename = {"channel": self.channel_dim}
        self._spend_scale = spend_scale.rename(rename)
        self._identified_by_design = ((shift != 0) | holdout).any("date").rename(rename)
        self._factual = xr.Dataset(
            {
                "chosen": (chosen / spend_scale).rename(rename),
                "active": active.astype(float).rename(rename),
            }
        )

        model.add_coord(self.channel_dim, channels)
        if self.surprise_lags >= len(dates):
            raise ValueError(
                f"surprise_lags={self.surprise_lags} must be smaller than the number "
                f"of dates ({len(dates)})."
            )
        if self.surprise_lags > 0:
            model.add_coord(self.lag_dim, list(range(1, self.surprise_lags + 1)))
            model.add_coord(self.source_date_dim, dates)
            pmd.Data(
                f"{self.prefix}_lag_operator",
                self._lag_operator(len(dates)),
                dims=(self.lag_dim, "date", self.source_date_dim),
            )
        pmd.Data(
            f"{self.prefix}_chosen_spend",
            self._factual["chosen"].values,
            dims=self._factual["chosen"].dims,
        )
        pmd.Data(
            f"{self.prefix}_active",
            self._factual["active"].values,
            dims=self._factual["active"].dims,
        )

        sales_order = int(getattr(mmm, "yearly_seasonality", None) or 0)
        self._fourier_order = (
            sales_order if self.fourier_order is None else self.fourier_order
        )
        if self._fourier_order > sales_order:
            warnings.warn(
                f"The spend equation's Fourier order ({self._fourier_order}) exceeds "
                f"the sales equation's yearly seasonality ({sales_order}). The extra "
                "seasonal terms act as instruments, which is invalid if seasonality "
                "also moves sales.",
                UserWarning,
                stacklevel=2,
            )
        if self._fourier_order > 0:
            model.add_coord(
                self.fourier_dim,
                [f"sin_{k}" for k in range(1, self._fourier_order + 1)]
                + [f"cos_{k}" for k in range(1, self._fourier_order + 1)],
            )
            pmd.Data(
                f"{self.prefix}_dayofyear",
                dates.dayofyear.to_numpy(),
                dims="date",
            )

        if self.trend:
            has_sales_trend = bool(
                getattr(mmm, "time_varying_intercept", False)
            ) or any(
                isinstance(effect, LinearTrendEffect)
                for effect in getattr(mmm, "mu_effects", [])
            )
            if not has_sales_trend:
                warnings.warn(
                    "trend=True adds a trend to the spend equation, but the sales "
                    "equation has none (no LinearTrendEffect or time-varying "
                    "intercept), so the trend acts as an instrument.",
                    UserWarning,
                    stacklevel=2,
                )
            self._time_origin = dates.min()
            self._time_span = max(float((dates.max() - dates.min()).days), 1.0)
            pmd.Data(f"{self.prefix}_time", self._time_index(dates), dims="date")

        if self.use_controls and "_control" in mmm.xarray_dataset:
            self._control_stats = _standardise_stats(
                mmm.xarray_dataset["_control"], keep="control"
            )
        else:
            self._control_stats = None

        self._instrument_stats = {}
        for var_name in self.instruments:
            if var_name not in mmm.xarray_dataset:
                raise ValueError(
                    f"Instrument {var_name!r} is not in the training data. Add it "
                    "as a column of X."
                )
            da = mmm.xarray_dataset[var_name]
            if "date" not in da.dims:
                raise ValueError(f"Instrument {var_name!r} must have a 'date' dim.")
            extra = sorted(set(da.dims) - {"date", *self._dims})
            if extra:
                raise ValueError(
                    f"Instrument {var_name!r} has dims {extra} beyond the model's "
                    f"{('date', *self._dims)}. Aggregate it before passing it."
                )
            da = da.transpose("date", *[d for d in self._dims if d in da.dims])
            std = float(da.std())
            self._instrument_stats[var_name] = (
                float(da.mean()),
                std if std > 0 else 1.0,
            )
            existing = model.named_vars.get(var_name)
            if existing is None:
                pmd.Data(var_name, da.values, dims=da.dims)
            elif (
                tuple(model.named_vars_to_dims.get(var_name, ())) != da.dims
                or tuple(existing.get_value(borrow=True).shape) != da.shape
            ):
                raise ValueError(
                    f"Cannot reuse model variable {var_name!r} as an instrument: its "
                    "dims or shape differ from the instrument's."
                )

    def _time_index(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Days since the training start, as a fraction of the training span."""
        return ((dates - self._time_origin).days / self._time_span).to_numpy()

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

        chosen = model[f"{p}_chosen_spend"]
        active = model[f"{p}_active"]

        spend_mu = self._prior("spend_intercept").create_variable(
            f"{p}_spend_intercept", xdist=True
        )

        if self._control_stats is not None and "control_data" in model.named_vars:
            mean, std = self._control_stats
            z = (model["control_data"] - as_xtensor(mean.values, dims=mean.dims)) / (
                as_xtensor(std.values, dims=std.dims)
            )
            coef = self._prior("spend_control").create_variable(
                f"{p}_spend_control_coef", xdist=True
            )
            spend_mu = spend_mu + (z * coef).sum(dim="control")

        if self._fourier_order > 0:
            modes = generate_fourier_modes(
                periods=model[f"{p}_dayofyear"] / DAYS_IN_YEAR,
                n_order=self._fourier_order,
                fourier_dim=self.fourier_dim,
            )
            coef = self._prior("spend_fourier").create_variable(
                f"{p}_spend_fourier_coef", xdist=True
            )
            spend_mu = spend_mu + (modes * coef).sum(dim=self.fourier_dim)

        if self.trend:
            coef = self._prior("spend_trend").create_variable(
                f"{p}_spend_trend_coef", xdist=True
            )
            spend_mu = spend_mu + model[f"{p}_time"] * coef

        for var_name in self.instruments:
            mean, std = self._instrument_stats[var_name]
            z = (model[var_name] - mean) / std
            coef = self._prior("spend_instrument").create_variable(
                f"{p}_instrument_{var_name}_coef", xdist=True
            )
            spend_mu = spend_mu + z * coef

        # An intercept-only equation has no date dim until broadcast against spend.
        spend_mu, _ = ptx.broadcast(spend_mu, chosen)
        spend_mu = pmd.Deterministic(
            f"{p}_spend_mu", spend_mu.transpose("date", *self._dims, ch)
        )
        sigma = self._prior("spend_sigma").create_variable(
            f"{p}_spend_sigma", xdist=True
        )

        resid = chosen - spend_mu
        # Masked cells see observed 0 ~ Normal(0, 1): a constant, so this is
        # exactly the spend likelihood restricted to active cells.
        pmd.Normal(
            f"{p}_spend",
            mu=active * spend_mu,
            sigma=active * sigma + (1 - active),
            observed=active * chosen,
        )

        surprise = pmd.Deterministic(
            f"{p}_surprise", (active * resid).transpose("date", *self._dims, ch)
        )
        gamma = self._prior("gamma").create_variable(f"{p}_gamma", xdist=True)
        control_function = surprise * gamma
        if self.surprise_lags > 0:
            gamma_lag = self._prior("gamma_lag").create_variable(
                f"{p}_gamma_lag", xdist=True
            )
            # Shift with a lag operator, one lag at a time: slicing and
            # concatenating along date, or reducing over a length-one lag dim,
            # produced graphs PyTensor failed to differentiate.
            source = surprise.rename({"date": self.source_date_dim})
            operator = model[f"{p}_lag_operator"]
            for lag in range(1, self.surprise_lags + 1):
                lagged = ptx.dot(
                    source,
                    operator.isel({self.lag_dim: lag - 1}),
                    dim=self.source_date_dim,
                )
                control_function = control_function + lagged * gamma_lag.isel(
                    {self.lag_dim: lag - 1}
                )

        return pmd.Deterministic(
            self.contribution_var_name,
            control_function.sum(dim=ch).transpose("date", *self._dims),
        )

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
        """Align the factual surprise inputs with new prediction dates.

        On training dates, chosen spend and the activity mask are reindexed
        from the *training* data, never read from ``X``: changing spend in
        ``X`` is an intervention, and interventions are never budget
        surprises. On other dates the surprise is zero, or, with
        ``surprise_out_of_sample="observed"``, computed from the spend in
        ``X``. Instruments missing from ``X`` are filled with their training
        mean, since they only move the spend equation's mean.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.
        model : pm.Model
            The PyMC model, whose ``date`` coordinate is already updated.
        X : xr.Dataset
            The new prediction dataset.
        """
        if self._factual is None:
            raise RuntimeError(
                "BudgetModelEffect.create_data must run before set_data."
            )
        if not self._is_bound_to(mmm):
            raise RuntimeError(
                "This BudgetModelEffect instance was last built for a different MMM. "
                "Use one instance per model."
            )
        new_dates = _get_datetime_coords(model.coords["date"], "date")
        factual = self._factual.reindex(date=new_dates)
        chosen, active = factual["chosen"], factual["active"]
        if (
            self.surprise_out_of_sample == "observed"
            and X is not None
            and "_channel" in X.data_vars
        ):
            observed = (
                X["_channel"]
                .sel(channel=self._channels)
                .rename({"channel": self.channel_dim})
                .reindex(date=new_dates)
                .transpose(*chosen.dims)
            )
            chosen = chosen.fillna(observed / self._spend_scale)
            active = active.fillna(1.0)
        new_data: dict[str, Any] = {
            f"{self.prefix}_chosen_spend": chosen.fillna(0.0).values,
            f"{self.prefix}_active": active.fillna(0.0).values,
        }
        if self._fourier_order > 0:
            new_data[f"{self.prefix}_dayofyear"] = new_dates.dayofyear.to_numpy()
        if self.trend:
            new_data[f"{self.prefix}_time"] = self._time_index(new_dates)
        for var_name in self.instruments:
            if X is not None and var_name in X.data_vars:
                new_data[var_name] = X[var_name].values
            else:
                shape = (len(new_dates), *model[var_name].type.shape[1:])
                new_data[var_name] = np.full(shape, self._instrument_stats[var_name][0])
        coords = None
        if self.surprise_lags > 0:
            new_data[f"{self.prefix}_lag_operator"] = self._lag_operator(len(new_dates))
            coords = {self.source_date_dim: new_dates}
        pm.set_data(new_data, coords=coords, model=model)

    # ----------------------------------------------------- diagnostics
    def exogeneity_summary(self, mmm: Any, interval_prob: float = 0.94) -> pd.DataFrame:
        r"""Summarise the control-function coefficients as an exogeneity check.

        :math:`\gamma_c = 0` is the exogenous-spend case. A posterior for
        :math:`\gamma_c` concentrated away from zero says that unexplained
        budget moves carried demand, so the plain MMM's exogeneity assumption
        was doing work. This is a Bayesian analogue of the control-function
        (Durbin-Wu-Hausman) test, not a formal hypothesis test.

        The check assumes a correctly specified sales equation: because the
        surprise is part of spend, any misspecification of the response curve
        leaks into :math:`\gamma_c`. It has power only with excluded
        variation. For channels without a lift-test design,
        :math:`\gamma_c` rests on functional form and the priors, can lean
        away from zero when spend is exogenous, and the summary flags it.

        The summary is read from the effect ``mmm`` was built with, so it can
        be called from any instance with the same prefix, including the
        original object after :meth:`MMM.load`.

        Parameters
        ----------
        mmm : MMM
            A fitted MMM containing this effect.
        interval_prob : float, default 0.94
            Mass of the equal-tailed posterior interval.

        Returns
        -------
        pd.DataFrame
            One row per channel (and cell of any extra dimension) with columns
            ``gamma_mean``, ``gamma_lower``, ``gamma_upper`` (scaled units),
            ``gamma_per_spend_unit`` (target units per unexplained unit of
            spend; with ``link="log"``, log-scale change per unit of spend),
            ``prob_positive``, ``prior_sd``, ``posterior_sd``,
            ``contraction`` (``1 - posterior_sd / prior_sd``),
            ``identified_by_design`` and ``note``.
        """
        idata = getattr(mmm, "idata", None)
        name = f"{self.prefix}_gamma"
        if idata is None or "posterior" not in idata or name not in idata.posterior:
            raise RuntimeError(f"No posterior for {name!r}; fit the model first.")
        effect = self._attached(mmm)
        if effect is not self:
            return effect.exogeneity_summary(mmm, interval_prob=interval_prob)
        spend_scale = cast(xr.DataArray, self._spend_scale)

        gamma = idata.posterior[name]
        sample_dims = ("chain", "draw")
        tail = (1 - interval_prob) / 2

        target_scale = mmm.scalers["_target"]
        per_unit = gamma * target_scale / spend_scale

        prior_sd = self._prior_sd(mmm, gamma.isel(chain=0, draw=0, drop=True))
        posterior_sd = gamma.std(sample_dims)

        summary = xr.Dataset(
            {
                "gamma_mean": gamma.mean(sample_dims),
                "gamma_lower": gamma.quantile(tail, dim=sample_dims).drop_vars(
                    "quantile"
                ),
                "gamma_upper": gamma.quantile(1 - tail, dim=sample_dims).drop_vars(
                    "quantile"
                ),
                "gamma_per_spend_unit": per_unit.mean(sample_dims),
                "prob_positive": (gamma > 0).mean(sample_dims),
                "prior_sd": prior_sd,
                "posterior_sd": posterior_sd,
            }
        )
        summary["contraction"] = 1 - summary["posterior_sd"] / summary["prior_sd"]
        # gamma shared across a dim is informed by a design anywhere along it.
        identified = cast(xr.DataArray, self._identified_by_design)
        pooled = [d for d in identified.dims if d not in gamma.dims]
        if pooled:
            identified = identified.any(pooled)
        summary, identified = xr.broadcast(summary, identified)
        summary["identified_by_design"] = identified

        df = summary.to_dataframe().reset_index()
        df = df.rename(columns={self.channel_dim: "channel"})

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

    def _prior_sd(self, mmm: Any, template: xr.DataArray) -> xr.DataArray:
        """Prior standard deviation of gamma.

        Exact for a ``Normal`` prior with a fixed ``sigma``; otherwise estimated
        from seeded prior draws so the summary is reproducible.
        """
        factory = self._prior("gamma")
        sigma = getattr(factory, "parameters", {}).get("sigma")
        if getattr(factory, "distribution", None) == "Normal" and isinstance(
            sigma, int | float
        ):
            return xr.full_like(template, float(sigma), dtype=float)
        dims = getattr(factory, "dims", None) or ()
        dims = (dims,) if isinstance(dims, str) else tuple(dims)
        coords = {d: mmm.model.coords[d] for d in dims}
        pymc_logger = logging.getLogger("pymc")
        level = pymc_logger.level
        pymc_logger.setLevel(logging.WARNING)
        try:
            draws = factory.sample_prior(
                coords=coords, name="gamma", draws=4000, random_seed=0
            )["gamma"]
        finally:
            pymc_logger.setLevel(level)
        return draws.std([d for d in draws.dims if d in ("chain", "draw")])

    # ---------------------------------------------------- serialization
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict. ``__type__`` is injected by the registry."""
        design = None
        if self.design is not None:
            design = self.design.copy()
            for col in ("start_date", "end_date"):
                design[col] = pd.to_datetime(design[col]).dt.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                )
            design = (
                design.astype(object).where(design.notna(), None).to_dict(orient="list")
            )

        def _prior(value: VariableFactory | None) -> dict[str, Any] | None:
            return None if value is None else value.to_dict()

        return {
            "prefix": self.prefix,
            "channels": self.channels,
            "use_controls": self.use_controls,
            "fourier_order": self.fourier_order,
            "trend": self.trend,
            "instruments": list(self.instruments),
            "design": design,
            "surprise_lags": self.surprise_lags,
            "surprise_out_of_sample": self.surprise_out_of_sample,
            "spend_intercept_prior": _prior(self.spend_intercept_prior),
            "spend_control_prior": _prior(self.spend_control_prior),
            "spend_fourier_prior": _prior(self.spend_fourier_prior),
            "spend_trend_prior": _prior(self.spend_trend_prior),
            "spend_instrument_prior": _prior(self.spend_instrument_prior),
            "spend_sigma_prior": _prior(self.spend_sigma_prior),
            "gamma_prior": _prior(self.gamma_prior),
            "gamma_lag_prior": _prior(self.gamma_lag_prior),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetModelEffect:
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        from pymc_extras.deserialize import deserialize

        def _prior(value: dict[str, Any] | None) -> VariableFactory | None:
            if value is None:
                return None
            if "__type__" in value:
                return serialization.deserialize(value)
            return deserialize(value)

        design = data.get("design")
        prior_keys = [
            "spend_intercept_prior",
            "spend_control_prior",
            "spend_fourier_prior",
            "spend_trend_prior",
            "spend_instrument_prior",
            "spend_sigma_prior",
            "gamma_prior",
            "gamma_lag_prior",
        ]
        return cls(
            prefix=data.get("prefix", "budget"),
            channels=data.get("channels"),
            use_controls=data.get("use_controls", True),
            fourier_order=data.get("fourier_order"),
            trend=data.get("trend", False),
            instruments=data.get("instruments", []),
            design=None if design is None else pd.DataFrame(design),
            surprise_lags=data.get("surprise_lags", 0),
            surprise_out_of_sample=data.get("surprise_out_of_sample", "zero"),
            **{key: _prior(data.get(key)) for key in prior_keys},
        )
