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
"""Spend-dependent price of a delivered media unit, for the budget optimizer.

``cost_per_unit`` answers "what does one unit cost in period *t*". It cannot say that the price of a unit depends
on how much is bought in that period, which is what auction bid-up and rate-card tiers do, and it bites exactly
where budget optimization is used: with a constant price every marginal return the optimizer sees is an upper
bound, and it is too high precisely on the channels it is choosing to grow.

A :class:`PriceResponse` is a monotone map from per-period money to delivered units. It is applied inside the
optimizer's differentiable graph, on unscaled money and before ``channel_scales``, with ``cost_per_unit`` as the
base price. The decision variables, the bounds and every constraint stay in money; the model graph keeps
receiving units.

One parametric form ships, :class:`PowerPriceResponse`, whose ``elasticity=0`` reproduces the constant-price
optimizer operation for operation.

Precondition
------------
The feature is only sound when the model was fitted on **delivery units** (impressions, clicks), or on spend
restated into constant prices. A model fitted on nominal spend has already absorbed part of the price curvature
into its saturation curve, because price and volume moved together historically; a concave price map on top bends
the same curve twice and understates marginal return. The optimizer checks this against the fitted artifact -- the
historical ``cost_per_unit`` table set through :meth:`~pymc_marketing.mmm.mmm.MMM.set_cost_per_unit`, per
channel -- and refuses otherwise. ``assume_delivery_units=True`` with an explicit ``reference_spend`` is the
opt-out for spend deflated outside the library.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Self

import numpy as np
import pytensor.tensor as pt
import pytensor.xtensor as ptx
from pydantic import BaseModel, ConfigDict, Field, InstanceOf, model_validator
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable
from xarray import DataArray

from pymc_marketing.mmm.optimization_variables import align_to_model_coords

__all__ = [
    "PowerPriceResponse",
    "PriceResponse",
    "ResolvedPowerPriceResponse",
    "ResolvedPriceResponse",
]


class ResolvedPriceResponse(ABC):
    """A price response bound to one decision variable's cell layout.

    Built by :meth:`PriceResponse.resolve` and held by
    :class:`~pymc_marketing.mmm.optimization_variables.MediaVariable`. Coefficients are NumPy arrays over ``dims``
    in the model's coordinate order; the three methods build symbolic maps over per-period money.

    Every map takes ``spend``, per-period money as an ``XTensorVariable`` with dims ``(date_dim, *dims)``, and
    ``base_price``, the optimizer's ``cost_per_unit`` tensor over the same dims or ``None`` for a base price of 1.
    """

    dims: tuple[str, ...]
    is_identity: bool
    reference_spend: np.ndarray

    @abstractmethod
    def to_delivery(
        self, spend: XTensorVariable, base_price: XTensorVariable | None = None
    ) -> XTensorVariable:
        """Delivered units bought with ``spend``."""

    @abstractmethod
    def implied_price(
        self, spend: XTensorVariable, base_price: XTensorVariable | None = None
    ) -> XTensorVariable:
        """Average price of a delivered unit at ``spend``: ``spend / to_delivery(spend)``."""

    @abstractmethod
    def implied_marginal_price(
        self, spend: XTensorVariable, base_price: XTensorVariable | None = None
    ) -> XTensorVariable:
        """Money the *next* delivered unit costs at ``spend``: the reciprocal slope of ``to_delivery``."""


class ResolvedPowerPriceResponse(ResolvedPriceResponse):
    r"""The iso-elastic map bound to a layout, with a quadratic floor at low spend.

    Above the floor :math:`s_f`, with :math:`\text{scale} = s_{\text{ref}}^{\gamma}`:

    .. math::

        u(s) = \frac{\text{scale}\, s^{1-\gamma}}{p_0}, \qquad
        p(s) = p_0 \left(\frac{s}{s_{\text{ref}}}\right)^{\gamma}, \qquad
        m(s) = \frac{p(s)}{1 - \gamma}

    Below it :math:`u(s) = (a s + b s^2) / p_0` with :math:`a = (1+\gamma)\,\text{scale}\, s_f^{-\gamma}` and
    :math:`b = -\gamma\,\text{scale}\, s_f^{-\gamma-1}`, which pins :math:`u(0) = 0`, matches value and slope at
    :math:`s_f` (the map is C1), and stays concave and increasing. There :math:`p(s) = p_0 / (a + b s)` and
    :math:`m(s) = p_0 / (a + 2 b s)`; both are continuous at :math:`s_f` and both tend to :math:`p_0 / a` at zero
    spend, so :math:`m = p / (1 - \gamma)` holds on the power branch only.

    The floor is where the slope spread the solver can meet is capped, :math:`u'(0) / u'(s_{\text{ref}}) = M`,
    which gives :math:`s_f / s_{\text{ref}} = (M (1-\gamma)/(1+\gamma))^{-1/\gamma}`. Cells with
    :math:`\gamma = 0` have :math:`s_f = 0`, :math:`a = 1`, :math:`b = 0` and are exactly :math:`s / p_0`.

    Both ``where`` branches receive a clipped input because ``where`` evaluates both: the power branch would have
    an infinite derivative at 0, and the quadratic price branches have a pole in the region where they are not
    selected.
    """

    def __init__(
        self,
        *,
        gamma: np.ndarray,
        reference_spend: np.ndarray,
        max_slope_ratio: float,
        dims: tuple[str, ...],
        label: str,
    ) -> None:
        gamma = np.asarray(gamma, dtype="float64")
        reference_spend = np.asarray(reference_spend, dtype="float64")
        if gamma.shape != reference_spend.shape:
            raise ValueError(
                f"{label}: elasticity has shape {gamma.shape} but reference_spend has shape "
                f"{reference_spend.shape}; both are per cell over {dims}."
            )
        self.dims = tuple(dims)
        self.label = label
        self.gamma = gamma
        self.reference_spend = reference_spend
        self.max_slope_ratio = float(max_slope_ratio)
        self.is_identity = bool(np.all(gamma == 0.0))

        # Closed forms under a double where. A bare np.where(active, f(gamma), 0.0)
        # still evaluates f on the gamma = 0 cells: -1 / gamma and s_f ** -1 warn
        # and produce inf / nan there, and a nan constant in a dead branch is
        # harmless on the C backend but poisons the gradient under JAX.
        active = gamma > 0.0
        safe_gamma = np.where(active, gamma, 1.0)
        inner = self.max_slope_ratio * (1.0 - safe_gamma) / (1.0 + safe_gamma)
        safe_inner = np.where(active, inner, 2.0)
        s_floor = np.where(
            active, reference_spend * safe_inner ** (-1.0 / safe_gamma), 0.0
        )
        safe_floor = np.where(active, s_floor, 1.0)
        scale = reference_spend**gamma
        self.s_floor = s_floor
        self.scale = scale
        self.a = np.where(active, (1.0 + gamma) * scale * safe_floor**-gamma, scale)
        self.b = np.where(active, -gamma * scale * safe_floor ** (-gamma - 1.0), 0.0)

        wide = active & (s_floor > 0.01 * reference_spend)
        if np.any(wide):
            warnings.warn(
                f"{label}: max_slope_ratio={self.max_slope_ratio:g} puts the quadratic floor above 1% of "
                f"the reference spend on cells with elasticity {np.unique(gamma[wide]).tolist()} "
                f"(floor / reference up to {float((s_floor / reference_spend)[wide].max()):.3g}). "
                "At high elasticity a bounded slope spread and a narrow floor region are not both "
                "available. Raise max_slope_ratio to narrow the floor, or accept that the map is "
                "quadratic over that range.",
                UserWarning,
                stacklevel=2,
            )

        self._gamma = self._constant(gamma)
        self._reference = self._constant(reference_spend)
        self._s_floor = self._constant(s_floor)
        self._scale = self._constant(scale)
        self._a = self._constant(self.a)
        self._b = self._constant(self.b)

    def _constant(self, values: np.ndarray) -> XTensorVariable:
        return as_xtensor(pt.constant(values, dtype="float64"), dims=self.dims)

    def _branches(self, spend: XTensorVariable):
        above = spend >= self._s_floor
        s_power = ptx.math.maximum(spend, self._s_floor)
        s_quad = ptx.math.minimum(spend, self._s_floor)
        return above, s_power, s_quad

    def to_delivery(
        self, spend: XTensorVariable, base_price: XTensorVariable | None = None
    ) -> XTensorVariable:
        """Delivered units bought with ``spend``; ``u(0) == 0`` exactly."""
        above, s_power, s_quad = self._branches(spend)
        power = self._scale * s_power ** (1.0 - self._gamma)
        quadratic = self._a * s_quad + self._b * s_quad**2
        units = ptx.math.where(above, power, quadratic)
        return units if base_price is None else units / base_price

    def implied_price(
        self, spend: XTensorVariable, base_price: XTensorVariable | None = None
    ) -> XTensorVariable:
        """Average price of a delivered unit at ``spend``."""
        above, s_power, s_quad = self._branches(spend)
        power = (s_power / self._reference) ** self._gamma
        quadratic = 1.0 / (self._a + self._b * s_quad)
        ratio = ptx.math.where(above, power, quadratic)
        return ratio if base_price is None else ratio * base_price

    def implied_marginal_price(
        self, spend: XTensorVariable, base_price: XTensorVariable | None = None
    ) -> XTensorVariable:
        """Money the next delivered unit costs at ``spend``: ``p / (1 - gamma)`` above the floor."""
        above, s_power, s_quad = self._branches(spend)
        power = (s_power / self._reference) ** self._gamma / (1.0 - self._gamma)
        quadratic = 1.0 / (self._a + 2.0 * self._b * s_quad)
        ratio = ptx.math.where(above, power, quadratic)
        return ratio if base_price is None else ratio * base_price


def _cell_label(index: np.ndarray, template: DataArray) -> tuple:
    """Coordinate labels of one cell from its positional index, for error messages."""
    return tuple(
        template.coords[dim].values.tolist()[int(i)]
        for dim, i in zip(template.dims, index, strict=True)
    )


def _require_labelled(da: DataArray, label: str) -> None:
    """Refuse a DataArray whose dims carry no coordinates.

    ``reindex`` has nothing to align such a dim by and stamps the model's labels on in arrival order, so the same
    values in a different order would resolve to a different map. Same hazard, and same rule, as
    ``BudgetOptimizer._require_labelled_plan``.
    """
    unlabelled = [dim for dim in da.dims if dim not in da.coords]
    if unlabelled:
        raise ValueError(
            f"{label}: dims {unlabelled} carry no coordinates. Alignment would fall back to position, so the "
            "same values in a different order would mean a different thing. Give those dims the model's "
            "coordinate labels."
        )


class PriceResponse(BaseModel, ABC):
    """A monotone map from per-period money to delivered units, declared once and bound per variable.

    The declaration is reusable across models and windows; :meth:`resolve` binds it to one decision variable's
    cell layout and returns the object the optimizer's graph uses. Subclasses ship a closed, invertible
    parametric family so the implied delivery and clearing prices can be reported alongside the allocation.

    Parameters
    ----------
    reference_spend : xarray.DataArray or None
        Per-period money per cell at which the base price applies, over exactly the budget dims, in the units of
        ``result.budgets`` and ``total_budget``. ``None`` lets the optimizer derive it from the fitted model where
        one exists. Every family needs one, and the optimizer reads it before knowing the concrete type, which is
        why it is declared here; subclasses document the derivation and the guard on a supplied value.
    assume_delivery_units : bool
        Attest that the model's channel data is in delivery units (or in spend deflated to constant prices) even
        though no historical ``cost_per_unit`` table prices the channel. Required, together with an explicit
        ``reference_spend``, to apply a non-identity response to channels the fitted artifact cannot vouch for.
        Default ``False``: the optimizer then refuses, because a saturation curve fitted on nominal spend has
        already absorbed part of the price curvature and a concave price map on top would bend it twice.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    reference_spend: InstanceOf[DataArray] | None = Field(
        default=None,
        description=(
            "Per-period money per cell at which the base price applies, over exactly the budget dims; "
            "the units of result.budgets and total_budget. None derives it from the fitted model where "
            "one exists. Subclasses document the derivation and the guard on a supplied value."
        ),
    )

    assume_delivery_units: bool = Field(
        default=False,
        description=(
            "Attest that channel data is in delivery units although no historical cost_per_unit prices "
            "the channel. Requires an explicit reference_spend. See the class docstring for why the "
            "default refuses."
        ),
    )

    @property
    @abstractmethod
    def is_identity(self) -> bool:
        """True when the map is exactly ``spend / base_price`` on every cell."""

    def is_identity_on(
        self,
        *,
        dims: tuple[str, ...],
        coords: Mapping[str, list],
        mask: DataArray,
        date_dim: str,
        label: str = "price_response",
    ) -> bool:
        """Report whether the map is the identity on every *optimized* cell of a layout.

        :meth:`resolve` ignores the declaration outside the mask, so a response that names only masked-out
        cells resolves to the identity even though :attr:`is_identity` is False. The optimizer asks this
        before running its fitted-artifact gate, so a no-op response is not refused over channels it never
        touches. The default answers from the declaration alone; families whose elasticity varies by cell
        override it. ``label`` prefixes any error raised while reading the declaration.
        """
        return self.is_identity

    @abstractmethod
    def resolve(
        self,
        *,
        dims: tuple[str, ...],
        coords: Mapping[str, list],
        mask: DataArray,
        date_dim: str,
        derived_reference: DataArray | None,
        label: str,
        num_periods: int | None = None,
    ) -> ResolvedPriceResponse:
        """Bind the declaration to one variable's cell layout.

        Parameters
        ----------
        dims, coords : tuple[str, ...], Mapping[str, list]
            The variable's budget dims in the model's order and their coordinate labels.
        mask : DataArray
            Boolean mask over ``dims`` selecting the optimized cells.
        date_dim : str
            Name of the date dimension, which no input here may carry.
        derived_reference : DataArray or None
            Per-period money per cell read off the fitted artifact by the optimizer, or ``None`` when there is
            none (a spend variable, an opted-out model). In the units of ``result.budgets``.
        label : str
            Prefix for error messages, naming the variable.
        num_periods : int or None
            Length of the optimization window, when known. Lets a family test the specific hypothesis that a
            supplied reference is a window total rather than a per-period rate.
        """


class PowerPriceResponse(PriceResponse):
    r"""Iso-elastic price: the unit price rises as a power of the money spent in the period.

    .. math::

        p_t(s) = p_{0,t}\left(\frac{s}{s^{\text{ref}}}\right)^{\gamma}, \qquad
        u_t(s) = \frac{s}{p_t(s)} = \frac{s^{1-\gamma}\,(s^{\text{ref}})^{\gamma}}{p_{0,t}}, \qquad 0 \le \gamma < 1

    :math:`p_0` is the optimizer's ``cost_per_unit`` (1 when none is given) and applies at the reference spend;
    :math:`\gamma` is the price elasticity of the buy. ``elasticity=0`` reproduces the constant-price optimizer
    exactly. Below a small floor the map is a quadratic pinned at :math:`u(0) = 0`, so a channel at zero spend
    delivers nothing and has a finite marginal return; see :class:`ResolvedPowerPriceResponse`.

    Parameters
    ----------
    elasticity : float, dict[str, float] or xarray.DataArray
        :math:`\gamma` per cell in ``[0, 1)``. A float applies everywhere. A dict maps coordinate labels of
        exactly one budget dim (usually channels) to values; labels not named default to ``0.0``. A ``DataArray``
        over a subset of the budget dims is aligned to the model's coordinates; it may not carry the date dim,
        because a date-varying ``cost_per_unit`` already covers seasonal price *level* while a date-varying
        elasticity would model seasonal price *sensitivity*, which is not supported.
    reference_spend : xarray.DataArray or None
        Where :math:`p_0` applies: **per-period money per cell, in the units of** ``result.budgets`` **and**
        ``total_budget``, over exactly the budget dims. Default ``None`` derives it from the fitted model as the
        mean of ``constant_data["channel_spend"]`` over the periods each cell was on air (``spend > 0``), so a
        flighted channel is anchored at the level it actually bought at. A supplied value is checked against that
        derived default when one exists; see ``reference_spend_tolerance``. Required for ``spend_vars`` and for
        opted-out models, which have nothing to derive from.
    max_slope_ratio : float
        Cap on :math:`u'(0) / u'(s^{\text{ref}})`, the spread of marginal returns the solver can meet on one
        cell. Sets the floor :math:`s_f / s^{\text{ref}} = (M (1-\gamma)/(1+\gamma))^{-1/\gamma}`; must exceed
        :math:`(1+\gamma)/(1-\gamma)`. Default ``100``. Warns when the resulting floor exceeds 1% of the
        reference, which happens at high elasticity (15.8% at :math:`\gamma = 0.9`).
    reference_spend_tolerance : float
        Largest factor by which a supplied ``reference_spend`` may differ from the derived default on any
        optimized cell before it is rejected. Default ``10``. The guard exists because a reference summed over the
        window instead of per period is off by ``num_periods`` and shifts every price by
        ``num_periods ** elasticity`` with nothing in the output saying so. A typical window of 4 to 13 periods
        sits under the default, so that hypothesis is also tested by name: when the supplied value is within 5%
        of ``num_periods`` times the derived one on every optimized cell, it is refused as a window total.
        Setting ``reference_spend_tolerance`` explicitly (to any value) asserts that the scale is intended and
        skips that check; the generic factor check then applies alone.
    assume_delivery_units : bool
        See :class:`PriceResponse`. Declared there, with ``reference_spend``, because the optimizer reads both
        off any response before knowing its concrete type.

    Notes
    -----
    **Precondition.** Only sound when the model was fitted on delivery units or constant-price spend. The
    optimizer checks each optimized channel against the historical ``cost_per_unit`` table on the fitted model
    (``idata.attrs["cost_per_unit"]``, written by :meth:`~pymc_marketing.mmm.mmm.MMM.set_cost_per_unit` or by
    ``MMM(cost_per_unit=...)``) and refuses unpriced channels unless ``assume_delivery_units=True`` and
    ``reference_spend`` are both given. The ``cost_per_unit`` passed to the *optimizer* is independent of the
    historical one and proves nothing about the fit. The table is a declaration the library takes at face value:
    pricing every channel at 1.0 silences the check without making the fit sound. A merged model
    (:func:`~pymc_marketing.mmm.budget_optimizer.merge_inference_data`) carries no root attrs and therefore always
    needs the opt-out.

    **Extrapolation.** The power law is as confident below the reference as above it: at ``elasticity=0.4``,
    spending 15% of the reference prices a unit at ``0.46 p_0``. Anchor ``reference_spend`` where the base price
    was observed and plan near it; a budget far from history is a statement about the price curve as much as
    about the response curve.

    **Units.** The map acts on per-period money at the model's date granularity, after
    ``budget_distribution_over_period`` has redistributed the total. ``total_budget``, ``result.budgets`` and
    ``reference_spend`` are all per-period quantities; the window total is ``budgets * num_periods``. Pass the
    window's ``cost_per_unit`` too: with the fit in delivery units and no window price, the base price is 1 and
    money is fed to the model as units, which the optimizer warns about.

    **Behaviour change with** :math:`\gamma > 0`. The delivery map is strictly concave, so at equal total spend a
    non-uniform ``budget_distribution_over_period`` buys less delivery than a uniform one: concentrated buying
    clears higher. Economically right, and new for existing users of that argument the moment they set a
    non-zero elasticity. A user who holds a *window* total fixed and shortens the window raises the per-period
    rate and, with it, the price.

    **The elasticity is an input.** The model never observes price, so :math:`\gamma` comes from buying data
    (realized cost against volume, per channel), whose regression has its own endogeneity. Treat it as a
    sensitivity sweep, running ``elasticity=0.0`` (the constant-price baseline, whose result still reports
    ``implied_price == p_0``) beside the values you believe, rather than a point value presented as known.

    **Why not fixed-point iteration.** Solving at an assumed price, repricing from the result and re-solving
    converges to :math:`R'(u_i) / p_i(s_i) = \text{const}` across channels. The true first-order condition is
    :math:`R'(u_i) (1 - \gamma_i) / p_i(s_i) = \text{const}`. With one common :math:`\gamma` the factor is
    absorbed and the fixed point is the optimum; with heterogeneous :math:`\gamma` it over-allocates to the
    high-elasticity channels, which is the biddable-next-to-reserved case this feature exists for. Each reprice
    pass also needs a fresh optimizer, since ``cost_per_unit`` is fixed at construction.

    **Reading the result.** ``result.implied_delivery`` is the delivery per period, in the units the money buys
    -- before ``channel_scales``; the model node receives ``implied_delivery / channel_scales``, which coincides
    for an ``MMM`` (scales are 1). To score the plan's posterior response use
    :meth:`~pymc_marketing.mmm.budget_optimizer.BudgetOptimizer.evaluate_response_distribution`, which runs
    the same graph the solver used, price map included. The deprecated
    :meth:`~pymc_marketing.mmm.mmm.BudgetOptimizerWrapper.sample_response_distribution` takes a date-less
    allocation and broadcasts it over the window, so it needs ``implied_delivery.mean(date_dim)`` and is
    exact only under a uniform ``budget_distribution_over_period``; with a non-uniform one the per-period
    spread is already inside ``implied_delivery`` and must not be applied a second time.
    ``result.implied_marginal_price / result.implied_price == 1 / (1 - elasticity)`` above the floor, so at
    ``elasticity=0.25`` the next unit costs 33% more than the average one.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.mmm import PowerPriceResponse

        mmm.set_cost_per_unit(
            historical_cpu_df
        )  # records that the fit is in delivery units
        optimizer = mmm.budget_optimizer(
            start_date,
            end_date,
            cost_per_unit=window_cpu,  # xarray.DataArray over (date, *budget_dims); see BudgetOptimizer
            price_response=PowerPriceResponse(elasticity={"tv": 0.25, "display": 0.10}),
        )
        result = optimizer.allocate_budget(total_budget=weekly_budget)
        result.budgets, result.implied_price, result.implied_marginal_price
    """

    elasticity: float | dict[str, float] | InstanceOf[DataArray] = 0.0
    max_slope_ratio: float = Field(default=100.0, gt=1.0)
    reference_spend_tolerance: float = Field(default=10.0, gt=1.0)

    def _known_elasticities(self) -> np.ndarray:
        e = self.elasticity
        if isinstance(e, DataArray):
            return np.asarray(e.values, dtype="float64").ravel()
        if isinstance(e, Mapping):
            return np.asarray(list(e.values()), dtype="float64")
        return np.asarray([e], dtype="float64")

    @model_validator(mode="after")
    def _check_domain(self) -> Self:
        values = self._known_elasticities()
        bad = values[~((values >= 0.0) & (values < 1.0))]
        if bad.size:
            raise ValueError(
                f"PowerPriceResponse requires 0 <= elasticity < 1, got {bad.tolist()}. At 1 delivery is "
                "constant in spend; above it more money buys less and the optimizer drives the channel to "
                "its lower bound."
            )
        if values.size:
            g_max = float(values.max())
            minimum = (1.0 + g_max) / (1.0 - g_max)
            if self.max_slope_ratio <= minimum:
                raise ValueError(
                    f"PowerPriceResponse: max_slope_ratio must exceed {minimum:g} for elasticity "
                    f"{g_max:g} (it is (1 + gamma) / (1 - gamma)); got {self.max_slope_ratio:g}. Below "
                    "that the quadratic floor would land above the reference spend."
                )
        return self

    @property
    def is_identity(self) -> bool:
        """True when every elasticity is exactly zero."""
        return bool(np.all(self._known_elasticities() == 0.0))

    def is_identity_on(
        self,
        *,
        dims: tuple[str, ...],
        coords: Mapping[str, list],
        mask: DataArray,
        date_dim: str,
        label: str = "price_response",
    ) -> bool:
        """Report whether every *optimized* cell has zero elasticity; see :meth:`PriceResponse.is_identity_on`."""
        if self.is_identity:
            return True
        dims = tuple(dims)
        template = DataArray(
            np.zeros(tuple(len(coords[d]) for d in dims)),
            dims=dims,
            coords={d: list(coords[d]) for d in dims},
        )
        on = np.asarray(mask.transpose(*dims).values, dtype=bool)
        gamma = self._resolve_elasticity(template, date_dim, label)
        return bool(np.all(gamma[on] == 0.0))

    def resolve(
        self,
        *,
        dims: tuple[str, ...],
        coords: Mapping[str, list],
        mask: DataArray,
        date_dim: str,
        derived_reference: DataArray | None,
        label: str,
        num_periods: int | None = None,
    ) -> ResolvedPowerPriceResponse:
        """Bind to one variable's layout; see :meth:`PriceResponse.resolve`."""
        dims = tuple(dims)
        template = DataArray(
            np.zeros(tuple(len(coords[d]) for d in dims)),
            dims=dims,
            coords={d: list(coords[d]) for d in dims},
        )
        on = np.asarray(mask.transpose(*dims).values, dtype=bool)
        # A masked cell spends exactly nothing whatever its elasticity, so it
        # gets gamma = 0: no wide-floor warning off the sentinel reference, and
        # a decision set whose every cell is at gamma = 0 stays the identity.
        gamma = np.where(on, self._resolve_elasticity(template, date_dim, label), 0.0)

        if not np.any(gamma):
            # The identity on every optimized cell. The reference is never
            # read: at gamma = 0 the floor is 0 and (s / s_ref) ** 0 is 1.
            reference = np.ones(template.shape)
        elif self.reference_spend is not None:
            reference = self._resolve_supplied_reference(
                template, on, derived_reference, date_dim, label, num_periods
            )
        elif derived_reference is not None:
            reference = self._check_reference(
                np.asarray(derived_reference.transpose(*dims).values, dtype="float64"),
                on,
                template,
                label,
                reason="the fitted spend has no on-air period",
            )
        else:
            raise ValueError(
                f"{label}: reference_spend is required -- there is no fitted spend to derive the level at "
                "which the base price applies. Pass reference_spend as per-period money per cell."
            )
        return ResolvedPowerPriceResponse(
            gamma=gamma,
            reference_spend=reference,
            max_slope_ratio=self.max_slope_ratio,
            dims=dims,
            label=label,
        )

    def _resolve_elasticity(
        self, template: DataArray, date_dim: str, label: str
    ) -> np.ndarray:
        dims = template.dims
        e = self.elasticity
        if isinstance(e, DataArray):
            if date_dim in e.dims:
                raise ValueError(
                    f"{label}: elasticity varies over {date_dim!r}. A date-varying cost_per_unit already "
                    "carries the seasonal price level; a date-varying elasticity would model seasonal price "
                    f"sensitivity, which is not supported. Drop the {date_dim!r} dim."
                )
            extra = sorted(set(e.dims) - set(dims))
            if extra:
                raise ValueError(
                    f"{label}: elasticity has dims {extra} that are not budget dims {list(dims)}."
                )
            _require_labelled(e, f"{label}: elasticity")
            aligned = align_to_model_coords(
                e,
                {d: template.coords[d].values.tolist() for d in e.dims},
                label=f"{label}: elasticity",
            )
            full = aligned.broadcast_like(template).transpose(*dims)
        elif isinstance(e, Mapping):
            full = self._elasticity_from_mapping(e, template, label)
        else:
            full = template + float(e)
        gamma = np.asarray(full.values, dtype="float64")
        bad = ~((gamma >= 0.0) & (gamma < 1.0))
        if bad.any():
            raise ValueError(
                f"{label}: requires 0 <= elasticity < 1, got {np.unique(gamma[bad]).tolist()}."
            )
        return gamma

    @staticmethod
    def _elasticity_from_mapping(
        e: Mapping, template: DataArray, label: str
    ) -> DataArray:
        dims = template.dims
        if not e:
            return template.copy()
        keys = set(e)
        labels = {d: template.coords[d].values.tolist() for d in dims}
        owners = [d for d in dims if keys <= set(labels[d])]
        if len(owners) != 1:
            where = "none matches" if not owners else f"they match {owners}"
            raise ValueError(
                f"{label}: elasticity keys {sorted(map(str, keys))} must all be labels of exactly one "
                f"budget dim; {where}. Available labels: {labels}."
            )
        dim = owners[0]
        values = DataArray(
            [float(e.get(lbl, 0.0)) for lbl in labels[dim]],
            dims=(dim,),
            coords={dim: labels[dim]},
        )
        return values.broadcast_like(template).transpose(*dims)

    def _resolve_supplied_reference(
        self,
        template: DataArray,
        on: np.ndarray,
        derived_reference: DataArray | None,
        date_dim: str,
        label: str,
        num_periods: int | None = None,
    ) -> np.ndarray:
        dims = template.dims
        ref = self.reference_spend
        if (
            ref is None
        ):  # pragma: no cover - resolve() only calls this with a supplied reference
            raise ValueError(f"{label}: reference_spend is required here.")
        if set(ref.dims) != set(dims):
            raise ValueError(
                f"{label}: reference_spend must have exactly the budget dims {list(dims)} -- per-period "
                f"money per cell, with no {date_dim!r} dim -- got {list(ref.dims)}."
            )
        _require_labelled(ref, f"{label}: reference_spend")
        aligned = align_to_model_coords(
            ref,
            {d: template.coords[d].values.tolist() for d in dims},
            label=f"{label}: reference_spend",
        ).transpose(*dims)
        values = self._check_reference(
            np.asarray(aligned.values, dtype="float64"),
            on,
            template,
            label,
            reason="reference_spend is not positive and finite",
        )
        if derived_reference is not None:
            expected_all = np.asarray(
                derived_reference.transpose(*dims).values, dtype="float64"
            )
            supplied, expected = values[on], expected_all[on]
            comparable = np.isfinite(expected) & (expected > 0.0)
            if not comparable.all():
                cells = [
                    _cell_label(idx, template)
                    for idx in np.argwhere(on)[~comparable][:5]
                ]
                warnings.warn(
                    f"{label}: reference_spend could not be checked against the fitted spend for cells "
                    f"{cells}, which have no on-air period; the supplied value is used as given there.",
                    UserWarning,
                    stacklevel=3,
                )
            # The mistake this guard exists for is a window total handed over
            # as a per-period rate: off by exactly num_periods, on every cell,
            # and under the generic tolerance for any window shorter than it.
            # Tested by name when the window is known and the user has not
            # asserted the scale by setting the tolerance themselves.
            if (
                num_periods is not None
                and num_periods > 1
                and "reference_spend_tolerance" not in self.model_fields_set
                and comparable.any()
            ):
                scale = supplied[comparable] / expected[comparable]
                if np.all(np.abs(scale / num_periods - 1.0) < 0.05):
                    raise ValueError(
                        f"{label}: reference_spend is num_periods ({num_periods}) times the fitted "
                        "per-period spend on every optimized cell, to within 5%: it looks like a window "
                        "total. reference_spend is per-period money, the units of result.budgets and "
                        f"total_budget -- divide by {num_periods}. If a {num_periods}-fold scale-up on "
                        "every channel is really intended, set reference_spend_tolerance explicitly to "
                        "assert it."
                    )
            ratio = np.where(
                comparable, np.maximum(supplied / expected, expected / supplied), 0.0
            )
            worst = int(np.argmax(ratio))
            if ratio[worst] > self.reference_spend_tolerance:
                cell = _cell_label(np.argwhere(on)[worst], template)
                raise ValueError(
                    f"{label}: reference_spend at cell {cell} is {supplied[worst]:.4g}, but the fitted "
                    f"spend's on-air mean per period is {expected[worst]:.4g} ({ratio[worst]:.3g}x apart; "
                    f"tolerance {self.reference_spend_tolerance:g}x). reference_spend is per-period money, "
                    "the units of result.budgets and total_budget -- a value summed over the window is off "
                    "by num_periods and shifts every price by num_periods ** elasticity. Fix the units, or "
                    "raise reference_spend_tolerance if the difference is intended."
                )
        return values

    @staticmethod
    def _check_reference(
        values: np.ndarray,
        on: np.ndarray,
        template: DataArray,
        label: str,
        *,
        reason: str,
    ) -> np.ndarray:
        bad = on & ~(np.isfinite(values) & (values > 0.0))
        if bad.any():
            cells = [_cell_label(idx, template) for idx in np.argwhere(bad)[:5]]
            more = ", ..." if int(bad.sum()) > 5 else ""
            raise ValueError(
                f"{label}: {reason} for optimized cells {cells}{more}, so there is no spend level to anchor "
                "the base price to. Pass reference_spend explicitly for them."
            )
        # Cells outside the mask scatter to exactly zero spend and deliver
        # nothing whatever the reference is; the coefficients only need to be
        # finite there.
        return np.where(on, values, 1.0)
