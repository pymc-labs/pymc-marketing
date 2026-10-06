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

A :class:`PriceResponse` is a non-decreasing, concave map from per-period money to delivered units. It is applied
inside the optimizer's differentiable graph, on unscaled money and before ``channel_scales``, with ``cost_per_unit``
as the base price. The decision variables, the bounds and every constraint stay in money; the model graph keeps
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
channel -- and refuses otherwise. ``assume_delivery_units=True`` with a ``reference_spend`` for the channels the
table does not price is the opt-out for spend deflated outside the library.
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

from pymc_marketing.mmm.optimization_variables import (
    _reject_unknown_coords,
    align_to_model_coords,
)

__all__ = [
    "PowerPriceResponse",
    "PriceResponse",
    "ResolvedPowerPriceResponse",
    "ResolvedPriceResponse",
]

# Lower bound on s_f / s_ref. The floor is max(inner ** (-1 / gamma), MIN_FLOOR_FRACTION); the
# second term wins below gamma ~ 0.16 at the default max_slope_ratio, where the slope spread
# (1 + gamma) / (1 - gamma) * MIN_FLOOR_FRACTION ** -gamma is below the cap. It also guards the
# underflow below gamma ~ 0.006, where the formula's fraction is 0 and a, b would be inf.
MIN_FLOOR_FRACTION = 1e-12


class ResolvedPriceResponse(ABC):
    """A price response bound to one decision variable's cell layout.

    Built by :meth:`PriceResponse.resolve` and held by
    :class:`~pymc_marketing.mmm.optimization_variables.MediaVariable`. How a family stores its coefficients is its
    own business; what the optimizer reads is per cell over ``dims`` (a family whose parameters vary by date reduces
    over it): ``money_scale``, in money, below ``1e-12 * money_scale`` of which a cell is reported as having bought
    nothing; ``curved``, boolean, where the map bends money (``u'' != 0``), equal to
    :meth:`PriceResponse.curved_cells` on the same layout and read before warning about a cell its bounds pin at
    zero; and ``is_identity``, which keeps the constant-price graph operation for operation.

    Every map takes ``spend``, per-period money as an ``XTensorVariable`` with dims ``(date_dim, *dims)``, and
    ``base_price``, the optimizer's ``cost_per_unit`` tensor over the same dims or ``None`` for a base price of 1.
    A family implements :meth:`to_delivery` and :meth:`implied_marginal_price`; :meth:`implied_price` follows.

    Contract: :meth:`to_delivery` is non-decreasing and concave in money with ``u(0) = 0``, and it and
    :meth:`implied_marginal_price` are finite, in value and gradient, on every cell including those outside the
    mask and those the map leaves unbent. ``where`` evaluates both of its branches, so a ``nan`` or ``inf`` in a
    branch that is not selected is harmless on the C backend and poisons the gradient under JAX.
    :meth:`implied_price` is reported only, never differentiated, and is ``nan`` where nothing is delivered.
    """

    dims: tuple[str, ...]
    is_identity: bool
    money_scale: np.ndarray
    curved: np.ndarray

    @abstractmethod
    def to_delivery(
        self, spend: XTensorVariable, base_price: XTensorVariable | None = None
    ) -> XTensorVariable:
        """Delivered units bought with ``spend``."""

    def implied_price(
        self, spend: XTensorVariable, base_price: XTensorVariable | None = None
    ) -> XTensorVariable:
        """Average price of a delivered unit at ``spend``: ``spend / to_delivery(spend)``.

        ``nan`` where nothing is delivered (zero or negative money): no unit was bought, so none was paid for.
        """
        units = self.to_delivery(spend, base_price)
        bought = units > 0.0
        # Double where: the division is evaluated on every cell, so it gets a safe denominator.
        return ptx.math.where(
            bought, spend / ptx.math.where(bought, units, 1.0), np.nan
        )

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
    :math:`m(s) = p_0 / (a + 2 b s)`; both are continuous at :math:`s_f` and both tend to :math:`p_0 / a` as
    spend falls to zero, so :math:`m = p / (1 - \gamma)` holds on the power branch only. At zero nothing is
    bought, so :math:`p` is ``nan`` there while :math:`m(0) = p_0 / a`.

    The floor is where the slope spread the solver can meet is capped, :math:`u'(0) / u'(s_{\text{ref}}) = M`,
    which gives :math:`s_f / s_{\text{ref}} = \max\big((M (1-\gamma)/(1+\gamma))^{-1/\gamma},\; 10^{-12}\big)`
    with the second term ``MIN_FLOOR_FRACTION``. That term wins below :math:`\gamma \approx 0.16` at the
    default :math:`M = 100`; there the spread is :math:`(1+\gamma)/(1-\gamma)\, 10^{12 \gamma}`, below
    :math:`M` exactly when the clamp binds, so ``max_slope_ratio`` has no effect on those cells. The clamp also
    guards the underflow below :math:`\gamma \approx 0.006`, where the formula's fraction is 0 and :math:`a`,
    :math:`b` would be infinite. Cells with :math:`\gamma = 0` have :math:`s_f = 0`, :math:`a = 1`,
    :math:`b = 0` and are exactly :math:`s / p_0`.

    Both ``where`` branches receive a clipped input because ``where`` evaluates both: the power branch would have
    an infinite derivative at 0, and the quadratic marginal price has a pole in the region where it is not
    selected. Money is clipped at zero first, so :math:`u(s) = 0` for :math:`s < 0`, where the marginal price
    equals its value at zero and the average price is ``nan``.
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
        self.money_scale = reference_spend
        self.curved = gamma > 0.0
        self.max_slope_ratio = float(max_slope_ratio)
        self.is_identity = bool(np.all(gamma == 0.0))
        self.s_floor, self.scale, self.a, self.b = self._power_floor_coefficients(
            gamma, reference_spend, self.max_slope_ratio
        )
        self._warn_wide_floor(
            label, gamma, self.s_floor, reference_spend, self.max_slope_ratio
        )
        self._gamma = self._constant(gamma)
        self._reference = self._constant(reference_spend)
        self._s_floor = self._constant(self.s_floor)
        self._scale = self._constant(self.scale)
        self._a = self._constant(self.a)
        self._b = self._constant(self.b)

    @staticmethod
    def _power_floor_coefficients(
        gamma: np.ndarray, reference_spend: np.ndarray, max_slope_ratio: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Floor location and coefficients ``(s_floor, scale, a, b)`` per cell, floor clamped at ``MIN_FLOOR_FRACTION``.

        Closed forms under a double ``where``: a bare ``np.where(active, f(gamma), 0.0)`` still evaluates ``f``
        on the ``gamma = 0`` cells, where ``-1 / gamma`` and ``s_f ** -1`` warn and produce ``inf``/``nan``. A
        ``nan`` constant in a dead branch is harmless on the C backend but poisons the gradient under JAX.
        """
        active = gamma > 0.0
        safe_gamma = np.where(active, gamma, 1.0)
        inner = max_slope_ratio * (1.0 - safe_gamma) / (1.0 + safe_gamma)
        safe_inner = np.where(active, inner, 2.0)
        # s_f / s_ref = max(inner ** (-1 / gamma), MIN_FLOOR_FRACTION). The clamp binds below
        # gamma ~ 0.16 at M = 100; there u'(0) / u'(s_ref) = (1 + gamma) / (1 - gamma)
        # * MIN_FLOOR_FRACTION ** -gamma, under the cap exactly when the clamp binds. It also
        # keeps a, b finite below gamma ~ 0.006, where the power underflows to 0.
        fraction = np.maximum(safe_inner ** (-1.0 / safe_gamma), MIN_FLOOR_FRACTION)
        s_floor = np.where(active, reference_spend * fraction, 0.0)
        safe_floor = np.where(active, s_floor, 1.0)
        scale = reference_spend**gamma
        a = np.where(active, (1.0 + gamma) * scale * safe_floor**-gamma, scale)
        b = np.where(active, -gamma * scale * safe_floor ** (-gamma - 1.0), 0.0)
        return s_floor, scale, a, b

    @staticmethod
    def _warn_wide_floor(
        label: str,
        gamma: np.ndarray,
        s_floor: np.ndarray,
        reference_spend: np.ndarray,
        max_slope_ratio: float,
    ) -> None:
        """Warn when the quadratic floor reaches above 1% of the reference spend."""
        wide = (gamma > 0.0) & (s_floor > 0.01 * reference_spend)
        if not np.any(wide):
            return
        warnings.warn(
            f"{label}: max_slope_ratio={max_slope_ratio:g} puts the quadratic floor above 1% of "
            f"the reference spend on cells with elasticity {np.unique(gamma[wide]).tolist()} "
            f"(floor / reference up to {float((s_floor / reference_spend)[wide].max()):.3g}). "
            "At high elasticity a bounded slope spread and a narrow floor region are not both "
            "available. Raise max_slope_ratio to narrow the floor, or accept that the map is "
            "quadratic over that range.",
            UserWarning,
            stacklevel=3,
        )

    def _constant(self, values: np.ndarray) -> XTensorVariable:
        return as_xtensor(pt.constant(values, dtype="float64"), dims=self.dims)

    def _branches(self, spend: XTensorVariable):
        # Money below zero buys nothing: SLSQP never evaluates outside the bounds, but a
        # labelled plan handed to evaluate_plan can, and the quadratic extrapolates there.
        # pytensor's ``Maximum`` sends the tie gradient to its first input only, so ``spend``
        # must stay the first argument for ``u'(0)`` to survive.
        spend = ptx.math.maximum(spend, 0.0)
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

    def implied_marginal_price(
        self, spend: XTensorVariable, base_price: XTensorVariable | None = None
    ) -> XTensorVariable:
        """Money the next delivered unit costs at ``spend``: ``p / (1 - gamma)`` above the floor."""
        above, s_power, s_quad = self._branches(spend)
        power = (s_power / self._reference) ** self._gamma / (1.0 - self._gamma)
        quadratic = 1.0 / (self._a + 2.0 * self._b * s_quad)
        ratio = ptx.math.where(above, power, quadratic)
        return ratio if base_price is None else ratio * base_price


class PriceResponse(BaseModel, ABC):
    """A non-decreasing, concave map from per-period money to delivered units, declared once and bound per variable.

    The declaration is reusable across models and windows; :meth:`resolve` binds it to one decision variable's
    cell layout and returns the object the optimizer's graph uses. Subclasses ship a closed, invertible
    parametric family so the implied delivery and clearing prices can be reported alongside the allocation.

    Every map a family resolves to must be non-decreasing and concave in money with ``u(0) = 0`` (see
    :class:`ResolvedPriceResponse`): a convex map, such as a volume discount, makes the allocation non-convex. A
    family refuses a declaration that would break this when it is constructed, as :class:`PowerPriceResponse`
    refuses ``elasticity < 0``.

    The optimizer asks three things of a family before resolving it, and keys its delivery-units gate on the
    answers rather than on the concrete type: where the map *bends* money (:attr:`adds_curvature`,
    :meth:`curved_cells`), whether the user attests that the fitted data are in delivery units
    (:attr:`attests_delivery_units`), and whether the declaration supplies no reference level of its own
    (:attr:`needs_derived_reference`). The gate runs before :meth:`resolve`, because a refusal must name its
    cause rather than the missing reference spend that cause implies. Declarations are frozen: mutating one after
    construction would bypass its validators.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    @property
    @abstractmethod
    def adds_curvature(self) -> bool:
        """True when the map bends money on some cell.

        This is what the delivery-units gate keys on. Writing the composed second derivative as
        :math:`f''(u) u'^2 + f'(u) u''`, it is the :math:`f'(u) u''` term that double-counts a saturation
        curve fitted on nominal spend, so a map with a non-zero ``u''`` needs the fit to be in delivery units
        and a map piecewise linear in money does not: it rescales the axis without bending it.
        """

    def curved_cells(
        self,
        *,
        dims: tuple[str, ...],
        coords: Mapping[str, list],
        mask: DataArray,
        date_dim: str,
        label: str = "price_response",
    ) -> np.ndarray:
        """Boolean per cell of a layout, in ``dims`` order: where the resolved map would bend money.

        Equal to ``resolve(...).curved`` on the same layout, and ``False`` outside the mask, which :meth:`resolve`
        ignores. Answered without a reference spend, so the optimizer can gate a response before resolving it.
        The default is :attr:`adds_curvature` on every optimized cell; a family whose parameters vary by cell
        overrides it. ``label`` prefixes any error raised while reading the declaration.
        """
        _, on = self._layout(dims, coords, mask)
        return on & self.adds_curvature

    @property
    def attests_delivery_units(self) -> bool:
        """The user vouches that the fitted data are in delivery units (or constant-price spend).

        Read only for a response that bends a cell the fitted artifact cannot vouch for. Default ``False``; a
        curved family exposes a field for it.
        """
        return False

    @property
    def needs_derived_reference(self) -> bool:
        """True when the declaration supplies no reference level, so a curved cell can only take a derived one.

        A family that states price *relative* to a level needs one; a family whose schedule is stated in
        absolute money never does. Default ``False``.
        """
        return False

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
            Name of the date dimension, passed so a family can validate its inputs against it. Whether a
            parameter may vary by date is the family's choice: :class:`PowerPriceResponse` refuses it, and a
            family that accepts it takes a ``date_dim`` of length ``num_periods`` aligned by position, as the
            optimizer's ``cost_per_unit`` is.
        derived_reference : DataArray or None
            Per-period money per cell read off the fitted artifact by the optimizer, in the units of
            ``result.budgets``; ``nan`` on cells it cannot vouch for (a channel the historical ``cost_per_unit``
            table does not price, a cell never on air). ``None`` when there is nothing to read (a spend variable,
            a model with no table).
        label : str
            Prefix for error messages, naming the variable.
        num_periods : int or None
            Length of the optimization window, when known: the length of any date axis a family accepts, and
            what lets :class:`PowerPriceResponse` test the hypothesis that a supplied reference is a window total
            rather than a per-period rate.
        """

    @staticmethod
    def _layout(
        dims: tuple[str, ...], coords: Mapping[str, list], mask: DataArray
    ) -> tuple[DataArray, np.ndarray]:
        """Build a zero template over one variable's cell layout and its optimized-cell mask in model order."""
        dims = tuple(dims)
        template = DataArray(
            np.zeros(tuple(len(coords[d]) for d in dims)),
            dims=dims,
            coords={d: list(coords[d]) for d in dims},
        )
        on = np.asarray(mask.transpose(*dims).values, dtype=bool)
        return template, on

    @staticmethod
    def _require_labelled(da: DataArray, label: str) -> None:
        """Refuse a DataArray whose dims carry no coordinates.

        ``reindex`` has nothing to align such a dim by and stamps the model's labels on in arrival order, so
        the same values in a different order would resolve to a different map. Same hazard, and same rule, as
        ``BudgetOptimizer._require_labelled_plan``.
        """
        unlabelled = [dim for dim in da.dims if dim not in da.coords]
        if unlabelled:
            raise ValueError(
                f"{label}: dims {unlabelled} carry no coordinates. Alignment would fall back to position, so "
                "the same values in a different order would mean a different thing. Give those dims the "
                "model's coordinate labels."
            )

    @staticmethod
    def _cell_label(index: np.ndarray, template: DataArray) -> tuple:
        """Coordinate labels of one cell from its positional index, for error messages."""
        return tuple(
            template.coords[dim].values.tolist()[int(i)]
            for dim, i in zip(template.dims, index, strict=True)
        )


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
        over a subset of the budget dims is aligned to the model's coordinates. It may not carry the date dim:
        a date-varying ``cost_per_unit`` already covers seasonal price *level*, and seasonal price
        *sensitivity* is not supported.
    reference_spend : xarray.DataArray or None
        Where :math:`p_0` applies: **per-period money per cell, in the units of** ``result.budgets`` **and**
        ``total_budget``, carrying every budget dim. Default ``None`` derives it from the fitted model as the
        mean of ``constant_data["channel_spend"]`` over the periods each cell was on air (``spend > 0``), so a
        flighted channel is anchored at the level it actually bought at. Only channels the historical
        ``cost_per_unit`` table prices get a derived value: an unpriced channel's ``channel_spend`` is its
        units, not money. A supplied value overrides the default cell by cell: its labels may be partial, cells
        it leaves out (absent labels or ``nan``) keep the derived one, and the cells it gives are checked
        against it where it exists (see ``reference_spend_tolerance``). It must cover every curved cell with no
        derived value: every cell of a spend variable, and the channels an attested model's table does not
        price.
    max_slope_ratio : float
        Cap on :math:`u'(0) / u'(s^{\text{ref}})`, the spread of marginal returns the solver can meet on one
        cell. Sets the floor :math:`s_f / s^{\text{ref}} = \max\big((M (1-\gamma)/(1+\gamma))^{-1/\gamma},\;
        10^{-12}\big)`; must exceed :math:`(1+\gamma)/(1-\gamma)`. Default ``100``. The second term wins below
        :math:`\gamma \approx 0.16` at the default: there the spread is :math:`(1+\gamma)/(1-\gamma)\,
        10^{12 \gamma}`, already under the cap, so this setting has no effect on those cells. Warns when the
        floor exceeds 1% of the reference, which happens at high elasticity. Leave a channel whose bounds pin
        it to zero at ``elasticity=0`` or drop it from ``budgets_to_optimize``: the map is steepest at zero, so
        pricing an immovable channel hands the solver its largest gradient on a variable that cannot move, which
        SLSQP tolerates on some platforms and not on others. ``allocate_budget`` warns when ``budget_bounds``
        does this.
    reference_spend_tolerance : float
        Largest factor by which a supplied ``reference_spend`` may differ from the derived default on any
        curved cell before it is rejected. Default ``10``. The unit error this catches is a window total
        handed over as a per-period rate: off by ``num_periods``, shifting every price by
        ``num_periods ** elasticity``. When the supplied value is ``num_periods`` times the derived one on
        every cell it can be checked on (within 5%), a warning names that hypothesis instead of refusing. A
        window total for a window of ``num_periods <= reference_spend_tolerance`` periods is therefore accepted
        with only the warning; lower the tolerance for short windows if that is a risk.
    assume_delivery_units : bool
        Attest that the node's data are in delivery units (or in spend deflated to constant prices) even
        though no historical ``cost_per_unit`` table prices them. Required, together with a ``reference_spend``
        covering those cells, to bend the price on channels the fitted artifact cannot vouch for and on every
        spend variable, which has no such artifact. Channels the table does price keep their derived reference
        alongside the attested ones, and a cell left at ``elasticity=0`` needs no vouching, since its money
        passes through unbent. Default ``False``: the optimizer then refuses, because a saturation curve fitted
        on nominal spend has already absorbed part of the price curvature and a concave price map on top would
        bend it twice (see :attr:`PriceResponse.adds_curvature`).

    Notes
    -----
    **Precondition.** Only sound when the model was fitted on delivery units or constant-price spend. The
    optimizer checks each channel the response bends against the historical ``cost_per_unit`` table on the
    fitted model (written by :meth:`~pymc_marketing.mmm.mmm.MMM.set_cost_per_unit` or ``MMM(cost_per_unit=...)``)
    and refuses the unpriced ones unless ``assume_delivery_units=True`` and a ``reference_spend`` covering them
    are given; channels left at ``elasticity=0`` need no vouching.
    The ``cost_per_unit`` passed to the *optimizer* is independent of that table and proves nothing about the
    fit. A merged model (:func:`~pymc_marketing.mmm.budget_optimizer.merge_inference_data`) carries no root
    attrs and always needs the opt-out.

    **Below the reference.** The power law is as confident below the reference as above it, so a priced channel
    bought far below its reference looks cheap and can attract budget: anchor ``reference_spend`` where you plan
    to buy (raising ``reference_spend_tolerance`` if that is far from the fitted spend) and read
    ``implied_price`` on every channel before trusting the allocation. A variant flat below the reference is
    #3089.

    **Not a rate schedule.** One smooth curve cannot hold a committed tranche at a contracted rate while the
    increment clears at another: calibrated to that case, its marginal price keeps climbing where the true rate
    is flat. Bracket rates are #3067.

    **Units.** The map acts on per-period money at the model's date granularity, after
    ``budget_distribution_over_period`` has redistributed the total. ``total_budget``, ``result.budgets`` and
    ``reference_spend`` are all per-period quantities; the window total is ``budgets * num_periods``. Without
    a window ``cost_per_unit`` the base price is 1 and money reaches the model as units, which the optimizer
    warns about.

    **Behaviour change with** :math:`\gamma > 0`. The delivery map is strictly concave, so at equal total
    spend a non-uniform ``budget_distribution_over_period`` buys less delivery than a uniform one:
    concentrated buying clears higher. The same holds across windows: for a given per-period plan the window
    length does not move the price, but holding the *window* total fixed, a shorter window spends more per
    period and clears higher.

    **Volume discounts are out of scope.** :math:`\gamma` models bid-up on the incremental money of this plan.
    A negotiated or contract discount belongs in the base price, ``cost_per_unit``. A price that falls with
    volume (:math:`\gamma < 0`) makes delivery convex and the allocation non-convex for SLSQP, so it is
    refused; leave such a channel at ``elasticity=0``.

    **The elasticity is an input.** The model never observes price, so :math:`\gamma` comes from buying data
    with its own endogeneity. Treat it as a sensitivity sweep, running ``elasticity=0.0`` beside the values
    you believe.

    **Why not fixed-point iteration.** Re-solving at repriced constant prices misses the :math:`(1 - \gamma_i)`
    factor of the first-order condition, so it over-allocates to the high-elasticity channels whenever
    elasticities differ.

    **Reading the result.** ``result.implied_delivery`` is per-period delivery before ``channel_scales``.
    Score a plan with
    :meth:`~pymc_marketing.mmm.budget_optimizer.BudgetOptimizer.evaluate_response_distribution`, which runs
    the same graph the solver used, price map included; the deprecated ``sample_response_distribution`` takes
    a date-less allocation, so feed it ``implied_delivery.mean(date_dim)``, exact only under a uniform
    ``budget_distribution_over_period``. Above the floor
    ``implied_marginal_price / implied_price == 1 / (1 - elasticity)``.

    **The price report is sparse.** Both price arrays are ``nan`` wherever no money was spent, including
    periods a ``budget_distribution_over_period`` zeroes out: nothing was bought, so nothing was paid per
    unit. ``implied_delivery`` is ``0.0`` there, and the money identity holds on the spent cells. Reduce with
    ``.mean(skipna=True)``, or weight by delivery: ``budgets * num_periods / implied_delivery.sum(date_dim)``.

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
    reference_spend: InstanceOf[DataArray] | None = Field(
        default=None,
        description=(
            "Per-period money per cell at which the base price applies, carrying every budget dim; the units "
            "of result.budgets and total_budget. None derives it from the fitted model for the channels its "
            "historical cost_per_unit table prices (the on-air mean of constant_data['channel_spend']). A "
            "supplied value overrides that default cell by cell: its labels may be partial, cells it leaves "
            "out (absent or nan) keep the default, "
            "and the cells it gives are guarded against it by reference_spend_tolerance."
        ),
    )
    max_slope_ratio: float = Field(default=100.0, gt=1.0)
    reference_spend_tolerance: float = Field(default=10.0, gt=1.0)
    assume_delivery_units: bool = Field(
        default=False,
        description=(
            "Attest that the node's data are in delivery units although no historical cost_per_unit prices "
            "it. Requires a reference_spend covering the cells it attests for. See the class docstring for "
            "why the default refuses."
        ),
    )

    def _known_elasticities(self) -> np.ndarray:
        e = self.elasticity
        if isinstance(e, DataArray):
            return np.asarray(e.values, dtype="float64").ravel()
        if isinstance(e, Mapping):
            return np.asarray(list(e.values()), dtype="float64")
        return np.asarray([e], dtype="float64")

    @model_validator(mode="after")
    def _check_domain(self) -> Self:
        self._require_valid_elasticity(
            self._known_elasticities(), self.max_slope_ratio, "PowerPriceResponse"
        )
        return self

    @staticmethod
    def _require_valid_elasticity(
        values: np.ndarray, max_slope_ratio: float, label: str
    ) -> None:
        """Refuse an elasticity outside ``[0, 1)``, or one too steep for ``max_slope_ratio``.

        Run at construction and again on the resolved cells: ``frozen`` stops a field being reassigned, but the
        declaration holds the caller's ``DataArray`` or dict, which can still be mutated in place.
        """
        bad = values[~((values >= 0.0) & (values < 1.0))]
        if bad.size:
            raise ValueError(
                f"{label}: requires 0 <= elasticity < 1, got {np.unique(bad).tolist()}. At 1 delivery is "
                "constant in spend; above it more money buys less and the optimizer drives the channel to "
                "its lower bound. Below 0 the price falls with volume, which makes delivery convex: a "
                "negotiated or contract discount belongs in cost_per_unit, the base price."
            )
        if values.size:
            g_max = float(values.max())
            minimum = (1.0 + g_max) / (1.0 - g_max)
            if max_slope_ratio <= minimum:
                raise ValueError(
                    f"{label}: max_slope_ratio must exceed {minimum:g} for elasticity "
                    f"{g_max:g} (it is (1 + gamma) / (1 - gamma)); got {max_slope_ratio:g}. Below "
                    "that the quadratic floor would land above the reference spend."
                )

    @property
    def adds_curvature(self) -> bool:
        """The power law bends money wherever its elasticity is not zero."""
        return bool(np.any(self._known_elasticities() != 0.0))

    @property
    def attests_delivery_units(self) -> bool:
        """See :attr:`assume_delivery_units`."""
        return self.assume_delivery_units

    @property
    def needs_derived_reference(self) -> bool:
        """A relative price needs a level; without a supplied one it has to come from the fitted model."""
        return self.reference_spend is None

    def curved_cells(
        self,
        *,
        dims: tuple[str, ...],
        coords: Mapping[str, list],
        mask: DataArray,
        date_dim: str,
        label: str = "price_response",
    ) -> np.ndarray:
        """Optimized cells with a non-zero elasticity; see :meth:`PriceResponse.curved_cells`."""
        template, on = self._layout(dims, coords, mask)
        return self._optimized_elasticity(template, on, date_dim, label) > 0.0

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
        template, on = self._layout(dims, coords, mask)
        gamma = self._optimized_elasticity(template, on, date_dim, label)
        reference = self._resolve_reference(
            template,
            on,
            gamma > 0.0,
            derived_reference,
            date_dim,
            label,
            num_periods,
        )
        return ResolvedPowerPriceResponse(
            gamma=gamma,
            reference_spend=reference,
            max_slope_ratio=self.max_slope_ratio,
            dims=template.dims,
            label=label,
        )

    def _optimized_elasticity(
        self, template: DataArray, on: np.ndarray, date_dim: str, label: str
    ) -> np.ndarray:
        """Elasticity per cell of ``template``; a masked cell spends exactly nothing, so it gets 0."""
        gamma = np.where(on, self._resolve_elasticity(template, date_dim, label), 0.0)
        self._require_valid_elasticity(gamma, self.max_slope_ratio, label)
        return gamma

    def _resolve_reference(
        self,
        template: DataArray,
        on: np.ndarray,
        needs_reference: np.ndarray,
        derived_reference: DataArray | None,
        date_dim: str,
        label: str,
        num_periods: int | None,
    ) -> np.ndarray:
        """Choose the reference per cell: unread on flat cells, else supplied, else derived."""
        if not needs_reference.any():
            # The identity: the floor is 0 and (s / s_ref) ** 0 is 1, so the reference is never read.
            return np.ones(template.shape)
        derived = (
            None
            if derived_reference is None
            else np.asarray(
                derived_reference.transpose(*template.dims).values, dtype="float64"
            )
        )
        if self.reference_spend is None:
            if derived is None:
                raise ValueError(
                    f"{label}: reference_spend is required -- there is no fitted spend to derive the level "
                    "at which the base price applies. Pass reference_spend as per-period money per cell."
                )
            return self._check_reference(
                derived,
                on,
                needs_reference,
                template,
                label,
                reason="the fitted spend has no on-air period",
            )
        supplied = self._aligned_supplied_reference(
            self.reference_spend, template, date_dim, label
        )
        given = ~np.isnan(supplied)
        # A value that is given must be usable on its own; a missing one falls back below.
        self._check_reference(
            np.where(given, supplied, 1.0),
            on,
            needs_reference,
            template,
            label,
            reason="reference_spend is not positive and finite",
        )
        if derived is None:
            return self._check_reference(
                supplied,
                on,
                needs_reference,
                template,
                label,
                reason=(
                    "there is no reference_spend value (absent label or nan) and no fitted spend to "
                    "derive one from"
                ),
            )
        self._guard_supplied_reference(
            supplied, derived, needs_reference & given, template, label, num_periods
        )
        return self._check_reference(
            np.where(given, supplied, derived),
            on,
            needs_reference,
            template,
            label,
            reason=(
                "there is neither a reference_spend value nor a derived one (the historical "
                "cost_per_unit table does not price the channel, it was never on air, or channel_spend "
                "lacks the cell)"
            ),
        )

    def _resolve_elasticity(
        self, template: DataArray, date_dim: str, label: str
    ) -> np.ndarray:
        """Elasticity per cell of ``template``, whatever form the declaration took."""
        e = self.elasticity
        if isinstance(e, DataArray):
            full = self._elasticity_from_dataarray(e, template, date_dim, label)
        elif isinstance(e, Mapping):
            full = self._elasticity_from_mapping(e, template, label)
        else:
            full = template + float(e)
        return np.asarray(full.values, dtype="float64")

    @staticmethod
    def _elasticity_from_dataarray(
        e: DataArray, template: DataArray, date_dim: str, label: str
    ) -> DataArray:
        """Align a declared array to the layout; refuse a date dim or a non-budget dim."""
        dims = template.dims
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
        PriceResponse._require_labelled(e, f"{label}: elasticity")
        aligned = align_to_model_coords(
            e,
            {d: template.coords[d].values.tolist() for d in e.dims},
            label=f"{label}: elasticity",
        )
        return aligned.broadcast_like(template).transpose(*dims)

    @staticmethod
    def _elasticity_from_mapping(
        e: Mapping, template: DataArray, label: str
    ) -> DataArray:
        """Spread labelled values along the one budget dim that owns every key."""
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

    def _aligned_supplied_reference(
        self, ref: DataArray, template: DataArray, date_dim: str, label: str
    ) -> np.ndarray:
        """Align the declared reference to the layout, ``nan`` on the cells it leaves out.

        Unknown labels are refused, as for every labelled input; missing ones are not, because a cell the
        reference leaves out takes the derived default.
        """
        dims = template.dims
        if set(ref.dims) != set(dims):
            raise ValueError(
                f"{label}: reference_spend must have exactly the budget dims {list(dims)} -- per-period "
                f"money per cell, with no {date_dim!r} dim -- got {list(ref.dims)}."
            )
        self._require_labelled(ref, f"{label}: reference_spend")
        coords = {d: template.coords[d].values.tolist() for d in dims}
        _reject_unknown_coords(ref, coords, label=f"{label}: reference_spend")
        return np.asarray(ref.reindex(coords).transpose(*dims).values, dtype="float64")

    def _guard_supplied_reference(
        self,
        values: np.ndarray,
        expected_all: np.ndarray,
        checked: np.ndarray,
        template: DataArray,
        label: str,
        num_periods: int | None,
    ) -> None:
        """Refuse a unit error: compare the supplied reference with the fitted one on ``checked`` cells.

        ``checked`` holds the curved cells the user gave a value for; only those with a fitted value can be
        compared. The rest (an attested channel the table does not price, a cell never on air) are used as
        given, since nothing fitted is in money there.
        """
        comparable = checked & np.isfinite(expected_all) & (expected_all > 0.0)
        if not comparable.any():
            return
        cells = np.argwhere(comparable)
        supplied, expected = values[comparable], expected_all[comparable]
        # A window total handed over as a per-period rate is off by exactly num_periods
        # on every cell; a warning rather than a refusal, because the match is a heuristic.
        if num_periods is not None and num_periods > 1:
            scale = supplied / expected
            if np.all(np.abs(scale / num_periods - 1.0) < 0.05):
                warnings.warn(
                    f"{label}: reference_spend is num_periods ({num_periods}) times the fitted "
                    "per-period spend on every cell it can be checked on, to within 5%: it looks like a "
                    "window total. reference_spend is per-period money, the units of result.budgets and "
                    f"total_budget -- if so, divide by {num_periods}.",
                    UserWarning,
                    stacklevel=3,
                )
        ratio = np.maximum(supplied / expected, expected / supplied)
        worst = int(np.argmax(ratio))
        if ratio[worst] > self.reference_spend_tolerance:
            cell = self._cell_label(cells[worst], template)
            raise ValueError(
                f"{label}: reference_spend at cell {cell} is {supplied[worst]:.4g}, but the fitted "
                f"spend's on-air mean per period is {expected[worst]:.4g} ({ratio[worst]:.3g}x apart; "
                f"tolerance {self.reference_spend_tolerance:g}x). reference_spend is per-period money, "
                "the units of result.budgets and total_budget -- a value summed over the window is off "
                "by num_periods and shifts every price by num_periods ** elasticity. Fix the units, or "
                "raise reference_spend_tolerance if the difference is intended."
            )

    @staticmethod
    def _check_reference(
        values: np.ndarray,
        on: np.ndarray,
        needs_reference: np.ndarray,
        template: DataArray,
        label: str,
        *,
        reason: str,
    ) -> np.ndarray:
        """Refuse non-positive or non-finite values on the cells that read the reference; neutralise the rest.

        A flat cell never reads its reference (the floor is 0 and ``(s / s_ref) ** 0`` is 1), so only the
        curved cells can refuse. Valid values are kept everywhere they exist; a masked or flat cell with no
        usable value gets a finite sentinel, since its coefficients only need to be finite.
        """
        valid = np.isfinite(values) & (values > 0.0)
        bad = needs_reference & ~valid
        if bad.any():
            cells = [
                PriceResponse._cell_label(idx, template) for idx in np.argwhere(bad)[:5]
            ]
            more = ", ..." if int(bad.sum()) > 5 else ""
            raise ValueError(
                f"{label}: {reason} for optimized cells {cells}{more}, so there is no spend level to anchor "
                "the base price to. Pass reference_spend explicitly for them."
            )
        return np.where(on & valid, values, 1.0)
