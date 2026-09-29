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

# Lower bound on s_f / s_ref. The floor is max(inner ** (-1 / gamma), MIN_FLOOR_FRACTION); the
# second term wins below gamma ~ 0.16 at the default max_slope_ratio, where the slope spread
# (1 + gamma) / (1 - gamma) * MIN_FLOOR_FRACTION ** -gamma is below the cap. It also guards the
# underflow below gamma ~ 0.006, where the formula's fraction is 0 and a, b would be inf.
MIN_FLOOR_FRACTION = 1e-12


class ResolvedPriceResponse(ABC):
    """A price response bound to one decision variable's cell layout.

    Built by :meth:`PriceResponse.resolve` and held by
    :class:`~pymc_marketing.mmm.optimization_variables.MediaVariable`. Coefficients are NumPy arrays over ``dims``
    in the model's coordinate order; the three methods build symbolic maps over per-period money.
    ``money_scale`` is per cell in money: below ``1e-12 * money_scale`` a cell is reported as having bought
    nothing. ``curved`` is boolean per cell: the map bends money there (``u'' != 0``), which is what the
    optimizer checks before warning about a cell its bounds pin at zero.

    Every map takes ``spend``, per-period money as an ``XTensorVariable`` with dims ``(date_dim, *dims)``, and
    ``base_price``, the optimizer's ``cost_per_unit`` tensor over the same dims or ``None`` for a base price of 1.
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
    which gives :math:`s_f / s_{\text{ref}} = \max\big((M (1-\gamma)/(1+\gamma))^{-1/\gamma},\; 10^{-12}\big)`
    with the second term ``MIN_FLOOR_FRACTION``. That term wins below :math:`\gamma \approx 0.16` at the
    default :math:`M = 100`; there the spread is :math:`(1+\gamma)/(1-\gamma)\, 10^{12 \gamma}`, below
    :math:`M` exactly when the clamp binds, so ``max_slope_ratio`` has no effect on those cells. The clamp also
    guards the underflow below :math:`\gamma \approx 0.006`, where the formula's fraction is 0 and :math:`a`,
    :math:`b` would be infinite. Cells with :math:`\gamma = 0` have :math:`s_f = 0`, :math:`a = 1`,
    :math:`b = 0` and are exactly :math:`s / p_0`.

    Both ``where`` branches receive a clipped input because ``where`` evaluates both: the power branch would have
    an infinite derivative at 0, and the quadratic price branches have a pole in the region where they are not
    selected. Money is clipped at zero first, so :math:`u(s) = 0` for :math:`s < 0` and both prices there equal
    their value at zero.
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


class PriceResponse(BaseModel, ABC):
    """A monotone map from per-period money to delivered units, declared once and bound per variable.

    The declaration is reusable across models and windows; :meth:`resolve` binds it to one decision variable's
    cell layout and returns the object the optimizer's graph uses. Subclasses ship a closed, invertible
    parametric family so the implied delivery and clearing prices can be reported alongside the allocation.

    The optimizer asks four things of a family before resolving it, and keys its delivery-units gate on the
    answers rather than on the concrete type: whether the map is the identity (:attr:`is_identity`,
    :meth:`is_identity_on`), whether it *bends* money (:attr:`adds_curvature`, :meth:`adds_curvature_on`),
    whether the user attests that the fitted data are in delivery units (:attr:`attests_delivery_units`),
    and whether resolution needs a reference level read off the fitted model (:attr:`needs_derived_reference`).
    Declarations are frozen: mutating one after construction would bypass its validators.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

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
        touches. The default answers from the declaration alone; families whose parameters vary by cell
        override it. ``label`` prefixes any error raised while reading the declaration.
        """
        return self.is_identity

    @property
    @abstractmethod
    def adds_curvature(self) -> bool:
        """True when the map bends money somewhere it is not the identity.

        This is what the delivery-units gate keys on. Writing the composed second derivative as
        :math:`f''(u) u'^2 + f'(u) u''`, it is the :math:`f'(u) u''` term that double-counts a saturation
        curve fitted on nominal spend, so a map with a non-zero ``u''`` needs the fit to be in delivery units
        and a map piecewise linear in money does not: it rescales the axis without bending it.
        """

    def adds_curvature_on(
        self,
        *,
        dims: tuple[str, ...],
        coords: Mapping[str, list],
        mask: DataArray,
        date_dim: str,
        label: str = "price_response",
    ) -> bool:
        """Report whether the map bends money on some *optimized* cell of a layout.

        Default: :attr:`adds_curvature` and not :meth:`is_identity_on`. Right for any family whose curvature
        is exactly where it is not the identity; a family that is linear on some cells and curved on others
        overrides it.
        """
        return self.adds_curvature and not self.is_identity_on(
            dims=dims, coords=coords, mask=mask, date_dim=date_dim, label=label
        )

    @property
    def attests_delivery_units(self) -> bool:
        """The user vouches that the fitted data are in delivery units (or constant-price spend).

        Read only when :attr:`adds_curvature` is True and the fitted artifact cannot vouch for a cell. Default
        ``False``; a curved family exposes a field for it.
        """
        return False

    @property
    def needs_derived_reference(self) -> bool:
        """True when :meth:`resolve` can only succeed with a ``derived_reference`` from the optimizer.

        A family that states price *relative* to a level needs one unless the user supplied it; a family whose
        schedule is stated in absolute money never does. Default ``False``.
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
        ``total_budget``, over exactly the budget dims. Default ``None`` derives it from the fitted model as the
        mean of ``constant_data["channel_spend"]`` over the periods each cell was on air (``spend > 0``), so a
        flighted channel is anchored at the level it actually bought at. A supplied value is checked against
        that derived default when one exists; see ``reference_spend_tolerance``. Required for ``spend_vars``
        and for opted-out models, which have nothing to derive from.
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
        optimized cell before it is rejected. Default ``10``. The unit error this catches is a window total
        handed over as a per-period rate: off by ``num_periods``, shifting every price by
        ``num_periods ** elasticity``. When the supplied value is ``num_periods`` times the derived one on
        every optimized cell (within 5%), a warning names that hypothesis instead of refusing. A window total
        for a window of ``num_periods <= reference_spend_tolerance`` periods is therefore accepted with only
        the warning; lower the tolerance for short windows if that is a risk.
    assume_delivery_units : bool
        Attest that the node's data are in delivery units (or in spend deflated to constant prices) even
        though no historical ``cost_per_unit`` table prices them. Required, together with an explicit
        ``reference_spend``, to bend the price on channels the fitted artifact cannot vouch for and on every
        spend variable, which has no such artifact; a cell left at ``elasticity=0`` needs no vouching, since
        its money passes through unbent. Default ``False``: the optimizer then refuses, because a saturation
        curve fitted on nominal spend has already absorbed part of the price curvature and a concave price
        map on top would bend it twice (see :attr:`PriceResponse.adds_curvature`).

    Notes
    -----
    **Precondition.** Only sound when the model was fitted on delivery units or constant-price spend. The
    optimizer checks each channel the response bends against the historical ``cost_per_unit`` table on the
    fitted model (written by :meth:`~pymc_marketing.mmm.mmm.MMM.set_cost_per_unit` or ``MMM(cost_per_unit=...)``)
    and refuses the unpriced ones unless ``assume_delivery_units=True`` and ``reference_spend`` are both given;
    channels left at ``elasticity=0`` need no vouching.
    The ``cost_per_unit`` passed to the *optimizer* is independent of that table and proves nothing about the
    fit. A merged model (:func:`~pymc_marketing.mmm.budget_optimizer.merge_inference_data`) carries no root
    attrs and always needs the opt-out.

    **Below the reference.** The power law is as confident below the reference as above it: at
    ``elasticity=0.4``, spending 15% of the reference prices a unit at ``0.46 p_0``, and the marginal unit at
    ``0.77 p_0``. That moves allocations, not only reports. A channel that is worthless at ``p_0`` (measured:
    window price 40x its siblings, zero under a constant price) receives a small budget once it is priced,
    because its first money buys units at a fraction of ``p_0``; and with a total budget below the historical
    spend every priced channel reports a price under ``p_0``, which is the usual planning case when budgets
    are cut. Auction inventory is not symmetric this way -- floor prices and minimum bids hold the price up
    below the reference. So anchor ``reference_spend`` at the level you plan to buy at, not only where
    ``p_0`` was observed (raise ``reference_spend_tolerance`` when that is far from the fitted spend), read
    ``implied_price`` on every channel before trusting the allocation, and sweep ``elasticity`` rather than
    pin it. A variant flat below the reference is #3089.

    **Not a rate schedule.** One smooth curve cannot state a committed tranche at a contracted rate with
    incremental money at another rate. Calibrated to that case, the map is exact only at the anchor: the
    marginal price keeps climbing where the truth is flat (measured 16% high at an increment of half the
    baseline, 30% at a full one) and the committed tranche is repriced. Allocation follows the marginal price,
    so treat those numbers as a bound on the calibration, not a small correction.

    **Units.** The map acts on per-period money at the model's date granularity, after
    ``budget_distribution_over_period`` has redistributed the total. ``total_budget``, ``result.budgets`` and
    ``reference_spend`` are all per-period quantities; the window total is ``budgets * num_periods``. Without
    a window ``cost_per_unit`` the base price is 1 and money reaches the model as units, which the optimizer
    warns about.

    **Behaviour change with** :math:`\gamma > 0`. The delivery map is strictly concave, so at equal total
    spend a non-uniform ``budget_distribution_over_period`` buys less delivery than a uniform one:
    concentrated buying clears higher.

    **The elasticity is an input.** The model never observes price, so :math:`\gamma` comes from buying data
    with its own endogeneity. Treat it as a sensitivity sweep, running ``elasticity=0.0`` beside the values
    you believe.

    **Why not fixed-point iteration.** Solving at an assumed price and repricing from the result converges to
    :math:`R'(u_i) / p_i = \text{const}`; the true first-order condition is
    :math:`R'(u_i)(1 - \gamma_i) / p_i = \text{const}`. One common :math:`\gamma` absorbs the factor;
    heterogeneous :math:`\gamma` over-allocates to the high-elasticity channels, which is the case this
    feature exists for.

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
            "Per-period money per cell at which the base price applies, over exactly the budget dims; "
            "the units of result.budgets and total_budget. None derives it from the fitted model where "
            "one exists (the on-air mean of constant_data['channel_spend']); a supplied value is guarded "
            "against that default by reference_spend_tolerance."
        ),
    )
    max_slope_ratio: float = Field(default=100.0, gt=1.0)
    reference_spend_tolerance: float = Field(default=10.0, gt=1.0)
    assume_delivery_units: bool = Field(
        default=False,
        description=(
            "Attest that the node's data are in delivery units although no historical cost_per_unit prices "
            "it. Requires an explicit reference_spend. See the class docstring for why the default refuses."
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

    @property
    def adds_curvature(self) -> bool:
        """The power law bends money wherever its elasticity is not zero."""
        return not self.is_identity

    @property
    def attests_delivery_units(self) -> bool:
        """See :attr:`assume_delivery_units`."""
        return self.assume_delivery_units

    @property
    def needs_derived_reference(self) -> bool:
        """A relative price needs a level; without a supplied one it has to come from the fitted model."""
        return self.reference_spend is None

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
        template, on = self._layout(dims, coords, mask)
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
        template, on = self._layout(dims, coords, mask)
        # A masked cell spends exactly nothing whatever its elasticity, so it gets gamma = 0.
        gamma = np.where(on, self._resolve_elasticity(template, date_dim, label), 0.0)
        needs_reference = on & (gamma > 0.0)
        reference = self._resolve_reference(
            template,
            on,
            needs_reference,
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
        """Choose the reference source: unread on flat cells, else declared, else derived."""
        if not needs_reference.any():
            # The identity: the floor is 0 and (s / s_ref) ** 0 is 1, so the reference is never read.
            return np.ones(template.shape)
        if self.reference_spend is not None:
            return self._resolve_supplied_reference(
                template,
                on,
                needs_reference,
                derived_reference,
                date_dim,
                label,
                num_periods,
            )
        if derived_reference is not None:
            return self._check_reference(
                np.asarray(
                    derived_reference.transpose(*template.dims).values, dtype="float64"
                ),
                on,
                needs_reference,
                template,
                label,
                reason="the fitted spend has no on-air period",
            )
        raise ValueError(
            f"{label}: reference_spend is required -- there is no fitted spend to derive the level at "
            "which the base price applies. Pass reference_spend as per-period money per cell."
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
        gamma = np.asarray(full.values, dtype="float64")
        bad = ~((gamma >= 0.0) & (gamma < 1.0))
        if bad.any():
            raise ValueError(
                f"{label}: requires 0 <= elasticity < 1, got {np.unique(gamma[bad]).tolist()}."
            )
        return gamma

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

    def _resolve_supplied_reference(
        self,
        template: DataArray,
        on: np.ndarray,
        needs_reference: np.ndarray,
        derived_reference: DataArray | None,
        date_dim: str,
        label: str,
        num_periods: int | None = None,
    ) -> np.ndarray:
        """Validate and align the declared reference, then guard it against the derived one."""
        values = self._aligned_supplied_reference(
            template, on, needs_reference, date_dim, label
        )
        if derived_reference is not None:
            expected = np.asarray(
                derived_reference.transpose(*template.dims).values, dtype="float64"
            )
            self._guard_supplied_reference(
                values, expected, needs_reference, template, label, num_periods
            )
        return values

    def _aligned_supplied_reference(
        self,
        template: DataArray,
        on: np.ndarray,
        needs_reference: np.ndarray,
        date_dim: str,
        label: str,
    ) -> np.ndarray:
        """Align the declared reference to the layout, requiring it positive and finite where it is read."""
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
        self._require_labelled(ref, f"{label}: reference_spend")
        aligned = align_to_model_coords(
            ref,
            {d: template.coords[d].values.tolist() for d in dims},
            label=f"{label}: reference_spend",
        ).transpose(*dims)
        return self._check_reference(
            np.asarray(aligned.values, dtype="float64"),
            on,
            needs_reference,
            template,
            label,
            reason="reference_spend is not positive and finite",
        )

    def _guard_supplied_reference(
        self,
        values: np.ndarray,
        expected_all: np.ndarray,
        needs_reference: np.ndarray,
        template: DataArray,
        label: str,
        num_periods: int | None,
    ) -> None:
        """Compare the declared reference with the fitted one, on the cells that read it, and refuse a unit error."""
        on = needs_reference
        supplied, expected = values[on], expected_all[on]
        comparable = np.isfinite(expected) & (expected > 0.0)
        if not comparable.all():
            cells = [
                self._cell_label(idx, template)
                for idx in np.argwhere(on)[~comparable][:5]
            ]
            warnings.warn(
                f"{label}: reference_spend could not be checked against the fitted spend for cells "
                f"{cells}, which have no on-air period; the supplied value is used as given there.",
                UserWarning,
                stacklevel=4,
            )
        # A window total handed over as a per-period rate is off by exactly num_periods
        # on every cell; a warning rather than a refusal, because the match is a heuristic.
        if num_periods is not None and num_periods > 1 and comparable.any():
            scale = supplied[comparable] / expected[comparable]
            if np.all(np.abs(scale / num_periods - 1.0) < 0.05):
                warnings.warn(
                    f"{label}: reference_spend is num_periods ({num_periods}) times the fitted "
                    "per-period spend on every optimized cell, to within 5%: it looks like a window "
                    "total. reference_spend is per-period money, the units of result.budgets and "
                    f"total_budget -- if so, divide by {num_periods}.",
                    UserWarning,
                    stacklevel=4,
                )
        ratio = np.where(
            comparable, np.maximum(supplied / expected, expected / supplied), 0.0
        )
        worst = int(np.argmax(ratio))
        if ratio[worst] > self.reference_spend_tolerance:
            cell = self._cell_label(np.argwhere(on)[worst], template)
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
