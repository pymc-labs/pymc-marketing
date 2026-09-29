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

where :math:`z_t` are the (standardised) MMM controls and :math:`f_t` an annual
Fourier basis. The sales mean then gains a control-function term

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
carrying. See :meth:`BudgetModelEffect.exogeneity_summary`. The check has
power only where there is excluded variation. Without a lift-test design for a
channel, :math:`\gamma_c` is identified by functional form alone, and the
summary says so.

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
    budget = BudgetModelEffect(design=design)
    mmm.add_mu_effect(budget)
    mmm.fit(X, y)

    budget.exogeneity_summary(mmm)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

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

from pymc_marketing.mmm.additive_effect import Model, MuEffect, _get_datetime_coords
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
    fourier_order : int, default 2
        Order of the annual Fourier basis in the spend equations. ``0``
        disables it.
    drivers : list[str], optional
        Extra budget drivers, e.g. a recorded demand forecast: columns of a
        ``pd.DataFrame`` ``X`` (or data variables of an ``xr.Dataset``). Each
        must vary over ``date`` and have no dims beyond the model's own.
        Standardised like the controls. New values must be supplied for
        prediction dates. A driver enters the spend equation but not the sales
        equation, so it acts as an instrument and must be unrelated to the
        sales error: do not pass recorded outcomes such as lagged sales.
    design : pd.DataFrame, optional
        Lift-test designs, in the format of :func:`lift_test_design`. Without
        a design the effect is identified by functional form only.
    spend_intercept_prior : Prior, optional
        Prior for the spend-equation intercept. Spend is divided by its
        per-channel maximum, and the defaults are weakly informative on that
        scale.
    spend_control_prior : Prior, optional
        Prior for the spend-equation coefficients on the standardised controls.
    spend_fourier_prior : Prior, optional
        Prior for the spend-equation Fourier coefficients.
    spend_driver_prior : Prior, optional
        Prior for the spend-equation coefficients on the standardised drivers.
    spend_sigma_prior : Prior, optional
        Prior for the scale of the budget surprise.
    gamma_prior : Prior, optional
        Prior for the control-function coefficient, on the MMM's scaled-target
        per scaled-spend scale.

    Notes
    -----
    * With extra model dims (e.g. ``geo``) the defaults give each cell its own
      spend intercept and surprise scale, but pool the control, Fourier and
      driver coefficients and ``gamma`` across cells, so there is one
      :math:`\gamma_c` per channel. Pass priors with the extra dims to unpool
      them. Custom priors may only use the dims ``f"{prefix}_channel"``, the
      model's extra dims, and ``"control"`` or ``f"{prefix}_fourier"`` for the
      control and Fourier coefficients.
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
    * The surprise is zero on dates outside the training data, its
      expectation. Out-of-sample predictive intervals therefore omit the
      variance the control function absorbed in-sample, roughly
      :math:`\sum_c \gamma_c^2 s_c^2`, and are somewhat too narrow.
    * With ``link="log"`` the control-function term is additive on the log
      scale.
    """

    prefix: str = "budget"
    channels: list[str] | None = None
    use_controls: bool = True
    fourier_order: int = Field(2, ge=0)
    drivers: list[str] = Field(default_factory=list)
    design: InstanceOf[pd.DataFrame] | None = None
    spend_intercept_prior: VariableFactory | None = None
    spend_control_prior: VariableFactory | None = None
    spend_fourier_prior: VariableFactory | None = None
    spend_driver_prior: VariableFactory | None = None
    spend_sigma_prior: VariableFactory | None = None
    gamma_prior: VariableFactory | None = None

    model_config = {"arbitrary_types_allowed": True}

    _channels: list[str] = PrivateAttr(default_factory=list)
    _dims: tuple[str, ...] = PrivateAttr(default=())
    _factual: xr.Dataset | None = PrivateAttr(default=None)
    _spend_scale: xr.DataArray | None = PrivateAttr(default=None)
    _identified_by_design: xr.DataArray | None = PrivateAttr(default=None)
    _control_stats: tuple[xr.DataArray, xr.DataArray] | None = PrivateAttr(default=None)
    _driver_stats: dict[str, tuple[xr.DataArray, xr.DataArray]] = PrivateAttr(
        default_factory=dict
    )

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
    def data_vars(self) -> list[str]:
        """Dataset variables this effect reads (the extra budget drivers).

        Exposed so helpers such as ``create_zero_dataset`` carry the drivers
        into prediction and optimization datasets.
        """
        return list(self.drivers)

    # ----------------------------------------------------------- priors
    def _default_priors(self, dims: tuple[str, ...]) -> dict[str, VariableFactory]:
        ch = self.channel_dim
        return {
            "spend_intercept": Prior("Normal", mu=0.5, sigma=0.5, dims=(*dims, ch)),
            "spend_control": Prior("Normal", mu=0, sigma=0.2, dims=("control", ch)),
            "spend_fourier": Prior(
                "Normal", mu=0, sigma=0.2, dims=(self.fourier_dim, ch)
            ),
            "spend_driver": Prior("Normal", mu=0, sigma=0.2, dims=(ch,)),
            "spend_sigma": Prior("HalfNormal", sigma=0.2, dims=(*dims, ch)),
            "gamma": Prior("Normal", mu=0, sigma=0.5, dims=(ch,)),
        }

    def _allowed_prior_dims(self, name: str) -> set[str]:
        extra = {"spend_control": {"control"}, "spend_fourier": {self.fourier_dim}}
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
        return lift_test_design(
            self.design, dates=dates, channels=self._channels, dim_coords=dim_coords
        )

    def create_data(self, mmm: Model) -> None:
        """Register chosen spend, the activity mask, and the budget drivers.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.
        """
        model = mmm.model
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

        if self.fourier_order > 0:
            model.add_coord(
                self.fourier_dim,
                [f"sin_{k}" for k in range(1, self.fourier_order + 1)]
                + [f"cos_{k}" for k in range(1, self.fourier_order + 1)],
            )
            pmd.Data(
                f"{self.prefix}_dayofyear",
                dates.dayofyear.to_numpy(),
                dims="date",
            )

        if self.use_controls and "_control" in mmm.xarray_dataset:
            self._control_stats = _standardise_stats(
                mmm.xarray_dataset["_control"], keep="control"
            )
        else:
            self._control_stats = None

        self._driver_stats = {}
        for var_name in self.drivers:
            if var_name not in mmm.xarray_dataset:
                raise ValueError(
                    f"Budget driver {var_name!r} is not in the training data. Add it "
                    "as a column of X."
                )
            da = mmm.xarray_dataset[var_name]
            if "date" not in da.dims:
                raise ValueError(f"Budget driver {var_name!r} must have a 'date' dim.")
            extra = sorted(set(da.dims) - {"date", *self._dims})
            if extra:
                raise ValueError(
                    f"Budget driver {var_name!r} has dims {extra} beyond the model's "
                    f"{('date', *self._dims)}. Aggregate it before passing it."
                )
            da = da.transpose("date", *[d for d in self._dims if d in da.dims])
            mean, std = da.mean(), da.std()
            self._driver_stats[var_name] = (mean, xr.where(std > 0, std, 1.0))
            existing = model.named_vars.get(var_name)
            if existing is None:
                pmd.Data(var_name, da.values, dims=da.dims)
            elif (
                tuple(model.named_vars_to_dims.get(var_name, ())) != da.dims
                or tuple(existing.get_value(borrow=True).shape) != da.shape
            ):
                raise ValueError(
                    f"Cannot reuse model variable {var_name!r} as a budget driver: its "
                    "dims or shape differ from the driver's."
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

        if self.fourier_order > 0:
            modes = generate_fourier_modes(
                periods=model[f"{p}_dayofyear"] / DAYS_IN_YEAR,
                n_order=self.fourier_order,
                fourier_dim=self.fourier_dim,
            )
            coef = self._prior("spend_fourier").create_variable(
                f"{p}_spend_fourier_coef", xdist=True
            )
            spend_mu = spend_mu + (modes * coef).sum(dim=self.fourier_dim)

        for var_name in self.drivers:
            mean, std = self._driver_stats[var_name]
            z = (model[var_name] - float(mean)) / float(std)
            coef = self._prior("spend_driver").create_variable(
                f"{p}_driver_{var_name}_coef", xdist=True
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

        return pmd.Deterministic(
            self.contribution_var_name,
            (surprise * gamma).sum(dim=ch).transpose("date", *self._dims),
        )

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
        """Align the factual surprise inputs with new prediction dates.

        Chosen spend and the activity mask are reindexed from the *training*
        data, never read from ``X``: changing spend in ``X`` is an
        intervention, and interventions are never budget surprises. Dates
        outside training have an inactive mask, so their surprise is zero.

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
        new_dates = _get_datetime_coords(model.coords["date"], "date")
        factual = self._factual.reindex(date=new_dates, fill_value=0.0)
        new_data: dict[str, Any] = {
            f"{self.prefix}_chosen_spend": factual["chosen"].values,
            f"{self.prefix}_active": factual["active"].values,
        }
        if self.fourier_order > 0:
            new_data[f"{self.prefix}_dayofyear"] = new_dates.dayofyear.to_numpy()
        for var_name in self.drivers:
            if X is not None and var_name in X.data_vars:
                new_data[var_name] = X[var_name].values
        pm.set_data(new_data, model=model)

    # ----------------------------------------------------- diagnostics
    def exogeneity_summary(self, mmm: Any, interval_prob: float = 0.94) -> pd.DataFrame:
        r"""Summarise the control-function coefficients as an exogeneity check.

        :math:`\gamma_c = 0` is the exogenous-spend case. A posterior for
        :math:`\gamma_c` concentrated away from zero says that unexplained
        budget moves carried demand, so the plain MMM's exogeneity assumption
        was doing work. This is a Bayesian analogue of the control-function
        (Durbin-Wu-Hausman) test, not a formal hypothesis test.

        The check has power only with excluded variation. For channels
        without a lift-test design, :math:`\gamma_c` is identified by the
        nonlinearity of the response alone, and the summary flags it.

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
        if self._spend_scale is None or self._identified_by_design is None:
            raise RuntimeError("The model containing this effect has not been built.")

        gamma = idata.posterior[name]
        sample_dims = ("chain", "draw")
        tail = (1 - interval_prob) / 2

        target_scale = mmm.scalers["_target"]
        per_unit = gamma * target_scale / self._spend_scale

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
        identified = self._identified_by_design
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
                notes.append("identified by functional form only (no lift-test design)")
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
            "drivers": list(self.drivers),
            "design": design,
            "spend_intercept_prior": _prior(self.spend_intercept_prior),
            "spend_control_prior": _prior(self.spend_control_prior),
            "spend_fourier_prior": _prior(self.spend_fourier_prior),
            "spend_driver_prior": _prior(self.spend_driver_prior),
            "spend_sigma_prior": _prior(self.spend_sigma_prior),
            "gamma_prior": _prior(self.gamma_prior),
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
            "spend_driver_prior",
            "spend_sigma_prior",
            "gamma_prior",
        ]
        return cls(
            prefix=data.get("prefix", "budget"),
            channels=data.get("channels"),
            use_controls=data.get("use_controls", True),
            fourier_order=data.get("fourier_order", 2),
            drivers=data.get("drivers", []),
            design=None if design is None else pd.DataFrame(design),
            **{key: _prior(data.get(key)) for key in prior_keys},
        )
