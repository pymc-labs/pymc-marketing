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

A :class:`PriceResponse` is a non-decreasing map from per-period money to delivered units, concave wherever a solve
can reach. It is applied inside the optimizer's differentiable graph, on unscaled money and before
``channel_scales``, with ``cost_per_unit`` as the base price. The decision variables, the bounds and every
constraint stay in money; the model graph keeps receiving units.

Two families ship. :class:`PowerPriceResponse` is one smooth iso-elastic curve per cell, whose ``elasticity=0``
reproduces the constant-price optimizer operation for operation. :class:`TieredPriceResponse` is a schedule of
marginal rates, one per bracket of money, such as a committed rate and a rate for incremental money.

Precondition
------------
The feature is only sound when the model was fitted on **delivery units** (impressions, clicks), or on spend
restated into constant prices. A model fitted on nominal spend has already absorbed part of the price curvature
into its saturation curve, because price and volume moved together historically; a concave price map on top bends
the same curve twice and understates marginal return. The optimizer cannot check this: the fitted model carries
no evidence of the units of its data, and its historical ``cost_per_unit`` table is a declaration that
:meth:`~pymc_marketing.mmm.mmm.MMM.set_cost_per_unit` writes after the fit without touching the model. So a
response that bends money anywhere requires ``assume_delivery_units=True``. The historical table then anchors
``reference_spend`` on the channels it prices; every other curved cell needs a ``reference_spend``.
"""

from __future__ import annotations

import numbers
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Self

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import xarray as xr
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable
from xarray import DataArray

from pymc_marketing.mmm.optimization_variables import (
    align_to_model_coords,
    reject_unknown_coords,
)
from pymc_marketing.mmm.price_schedule import ScheduleRows
from pymc_marketing.names.mmm import Dim

__all__ = [
    "PowerPriceResponse",
    "PriceResponse",
    "ResolvedPowerPriceResponse",
    "ResolvedPriceResponse",
    "ResolvedTieredPriceResponse",
    "TieredPriceResponse",
]

# Lower bound on s_f / s_ref. The floor is max(inner ** (-1 / gamma), MIN_FLOOR_FRACTION); the
# second term wins below gamma ~ 0.16 at the default max_slope_ratio, where the slope spread
# (1 + gamma) / (1 - gamma) * MIN_FLOOR_FRACTION ** -gamma is below the cap. It also guards the
# underflow below gamma ~ 0.006, where the formula's fraction is 0 and a, b would be inf.
MIN_FLOOR_FRACTION = 1e-12

_Schedule = tuple[float, ...]


def _cell_label_for_message(label: tuple) -> object:
    """Return a cell's label for a message: the label itself on a one-dim layout, the tuple otherwise."""
    return label[0] if len(label) == 1 else label


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

    Contract: :meth:`to_delivery` is non-decreasing in money with ``u(0) = 0``, and concave wherever a solve can
    reach: a family whose map has a convex part refuses, in :meth:`conditioned_on`, a solve whose bounds contain it.
    It and :meth:`implied_marginal_price` are finite, in value and gradient, on every cell including those outside
    the mask and those the map leaves unbent. ``where`` evaluates both of its branches, so a ``nan`` or ``inf`` in a
    branch that is not selected is harmless on the C backend and poisons the gradient under JAX.
    :meth:`implied_price` is reported only, never differentiated, and is ``nan`` where nothing is delivered.

    Three methods with defaults tell the optimizer about breakpoints, points where the slope of the map in
    money changes: :meth:`bind_period_weights`, :meth:`conditioned_on` and :meth:`linear_piece_bounds`. A family
    without breakpoints keeps the defaults.
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

    def bind_period_weights(self, weights: XTensorVariable | None) -> None:
        """Receive the money each period gets per unit of budget, before any graph is built.

        Called once by :class:`~pymc_marketing.mmm.optimization_variables.MediaVariable` after
        :meth:`PriceResponse.resolve`. ``weights`` is a symbolic tensor over ``(date_dim, *dims)``: ``num_periods``
        times the ``budget_distribution_over_period`` factor of each period, zero outside the mask; ``None``
        under the uniform spread, where it is 1. A family whose breakpoints are in budget units uses it to place
        the breakpoint of each period at ``breakpoint * weight``. The default ignores it.

        Parameters
        ----------
        weights : XTensorVariable or None
            Money each period gets per unit of budget, over ``(date_dim, *dims)``; ``None`` under the uniform
            spread.
        """
        return None

    @contextmanager
    def conditioned_on(self, low: np.ndarray, high: np.ndarray) -> Iterator[None]:
        """Condition the map for one solve over the box ``[low, high]``, and restore it on exit.

        ``low`` and ``high`` are per cell over ``dims``, in budget units, infinite where unbounded and ``nan``
        outside the mask. Inside the block the map may differ from the declared one in ways a solver needs, for
        example a breakpoint blended where the bounds let the solver cross it. A family whose map has a convex
        part raises here when the bounds contain it. The default changes nothing.

        Parameters
        ----------
        low : np.ndarray
            Lower bound of each cell over ``dims``, in budget units.
        high : np.ndarray
            Upper bound of each cell over ``dims``, in budget units.

        Raises
        ------
        ValueError
            In a family whose map is convex somewhere inside the bounds; never in the default.
        """
        yield

    def linear_piece_bounds(
        self,
        budgets: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        from_below: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per cell, the largest interval around ``budgets`` inside ``[low, high]`` with no breakpoint strictly inside.

        The map is affine in the budget on that interval. A cell on a breakpoint gets the interval above it, or
        the one below where ``from_below`` is set. All arrays are per cell over ``dims``, in budget units. The
        default, for a map with no breakpoints, is ``(low, high)``.

        Parameters
        ----------
        budgets : np.ndarray
            Budget of each cell over ``dims``.
        low : np.ndarray
            Lower bound of each cell over ``dims``.
        high : np.ndarray
            Upper bound of each cell over ``dims``, infinite where unbounded.
        from_below : np.ndarray or None
            Boolean per cell: where set, a cell on a breakpoint gets the piece below it. ``None`` sets it nowhere.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Lower and upper end of the piece of each cell, within ``[low, high]``.
        """
        return low, high


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
        # A flat cell has no floor, so s_power reaches 0 and the gradient of 0 ** 0 is nan;
        # its ratio is 1 at any spend, so it gets a base of 1 and a zero gradient.
        relative = ptx.math.where(self._gamma > 0.0, s_power / self._reference, 1.0)
        power = relative**self._gamma / (1.0 - self._gamma)
        quadratic = 1.0 / (self._a + 2.0 * self._b * s_quad)
        ratio = ptx.math.where(above, power, quadratic)
        return ratio if base_price is None else ratio * base_price


class ResolvedTieredPriceResponse(ResolvedPriceResponse):
    r"""A schedule of marginal rates bound to a layout: piecewise linear in money, with a knot at each breakpoint.

    For one cell with budget :math:`z`, breakpoints :math:`B_1 < \dots < B_K` and one multiplier :math:`r_k` per
    bracket,

    .. math::

        G(z) = \frac{z}{r_0} - \sum_{k=1}^{K} \Delta_k \max(z - B_k, 0), \qquad
        \Delta_k = \frac{1}{r_{k-1}} - \frac{1}{r_k}

    is the budget restated at the base price. In period :math:`t` the cell spends :math:`s_t = w_t z`, with
    :math:`w_t` the weight :meth:`bind_period_weights` received (1 under the uniform spread), and delivers

    .. math::

        u_t = \frac{1}{p_{0,t}} \Big( \frac{s_t}{r_0} - \sum_k \Delta_k \max(s_t - w_t B_k, 0) \Big)

    so the knot of period :math:`t` sits at :math:`w_t B_k`. The marginal price is
    :math:`m_t = p_{0,t} / G'(z)`. A first breakpoint of 0 is folded into the first slope, equal adjacent
    multipliers are no kink, and an unscheduled cell is the identity on the base price.

    Outside a solve every knot is hard and sits ``snap`` (64 ulp of ``floatX``) below its breakpoint, so money
    equal to a breakpoint is in the bracket above and the marginal price is right-continuous: a cell on its
    breakpoint reports the next bracket's rate. The cost is an offset of :math:`\Delta_k B_k \sigma` at and above
    the knot. :meth:`conditioned_on` moves knots off a solve's box and blends the ones strictly inside it with a
    C2 quartic of half-width ``knot_width * B``, for that solve only.

    Parameters
    ----------
    first_slope : np.ndarray
        :math:`1 / r_0` per cell over ``dims``, 1 where the cell is unscheduled.
    edges : np.ndarray
        Shape ``(K, *cells)``: the breakpoint of each kink in budget units, ``nan`` where there is no kink.
    slope_drops : np.ndarray
        Shape ``(K, *cells)``: the fall in slope at each kink, 0 where there is no kink.
    dims : tuple[str, ...]
        The variable's budget dims, in the order of the cell axes.
    coords : Mapping[str, list]
        Coordinate labels of each dim in ``dims``, used to name cells in messages.
    knot_width : float
        Blend half-width relative to the breakpoint, used only inside :meth:`conditioned_on`.
    label : str
        Prefix for error messages, naming the variable.

    Raises
    ------
    ValueError
        If ``edges`` or ``slope_drops`` is not shaped ``(K, *cells)``.

    Attributes
    ----------
    first_slope : np.ndarray
        :math:`1 / r_0` per cell, 1 where the cell is unscheduled.
    edges : np.ndarray
        Shape ``(K, *cells)``: :math:`B_k` where the slope changes, ``nan`` where there is no kink.
    slope_drops : np.ndarray
        Shape ``(K, *cells)``: :math:`\Delta_k`, negative where the rate falls, 0 where there is no kink.
    snap : float
        Relative distance of a hard knot below its breakpoint.
    knot_width : float
        Blend half-width relative to the breakpoint, used only inside :meth:`conditioned_on`.
    """

    def __init__(
        self,
        *,
        first_slope: np.ndarray,
        edges: np.ndarray,
        slope_drops: np.ndarray,
        dims: tuple[str, ...],
        coords: Mapping[str, list],
        knot_width: float,
        label: str,
    ) -> None:
        self.dims = tuple(dims)
        self.coords = {dim: list(coords[dim]) for dim in self.dims}
        self.label = label
        self.first_slope, self.edges, self.slope_drops = (
            self._validated_coefficient_arrays(first_slope, edges, slope_drops, label)
        )
        self.knot_width = float(knot_width)
        # The decision vector is created in floatX, so a knot snapped for floatX is on the right side of any
        # budget the solver hands back. A knot sits 64 ulp of floatX below its breakpoint, so money equal to
        # the breakpoint is in the bracket above whatever rounding it went through: 1.4e-14 in float64,
        # 7.6e-6 in float32.
        self.dtype = pytensor.config.floatX
        self.snap = 64 * float(np.finfo(self.dtype).eps)
        self._kink = np.isfinite(self.edges)
        self._edge = np.where(self._kink, self.edges, 1.0)
        self.is_identity = bool(
            np.all(self.first_slope == 1.0) and not self._kink.any()
        )
        self.curved = np.zeros(self.first_slope.shape, dtype=bool)
        self.money_scale = self._money_scale_from_kinks(self.edges, self._kink)
        self._create_knot_variables()
        self._weights: XTensorVariable | None = None

    @staticmethod
    def _validated_coefficient_arrays(
        first_slope: np.ndarray, edges: np.ndarray, slope_drops: np.ndarray, label: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read the three arrays as float64 and refuse ``edges`` or ``slope_drops`` not shaped ``(K, *cells)``."""
        first_slope = np.asarray(first_slope, dtype="float64")
        edges = np.asarray(edges, dtype="float64")
        slope_drops = np.asarray(slope_drops, dtype="float64")
        expected = (edges.shape[0], *first_slope.shape)
        if edges.shape != expected or slope_drops.shape != expected:
            raise ValueError(
                f"{label}: edges {edges.shape} and slope_drops {slope_drops.shape} must both be "
                f"(K, *cells) with cells {first_slope.shape}."
            )
        return first_slope, edges, slope_drops

    @staticmethod
    def _money_scale_from_kinks(edges: np.ndarray, kink: np.ndarray) -> np.ndarray:
        """Per cell, its smallest kink, else the smallest kink of the schedule, else 1."""
        smallest = np.where(kink, edges, np.inf).min(axis=0, initial=np.inf)
        overall = float(np.min(smallest, initial=np.inf))
        return np.where(
            np.isfinite(smallest), smallest, overall if np.isfinite(overall) else 1.0
        )

    def _create_knot_variables(self) -> None:
        """Create the hard knots and the shared knots and half-widths that :meth:`conditioned_on` moves."""
        # Where there is no kink the knot is a finite sentinel: its drop is 0, and a nan would poison the value
        # and the gradient.
        self._default_knots = np.where(self._kink, self.edges * (1.0 - self.snap), 1.0)
        self._knots = pytensor.shared(
            self._c_contiguous_float64(self._default_knots), name="tiered_knots"
        )
        self._half_widths = pytensor.shared(
            self._c_contiguous_float64(np.zeros(self.edges.shape)),
            name="tiered_half_widths",
        )

    @staticmethod
    def _c_contiguous_float64(values: np.ndarray) -> np.ndarray:
        """Copy into a C-ordered float64 array: a value in another layout makes Numba compile every reader again."""
        return np.array(values, dtype="float64", order="C")

    @classmethod
    def _slopes_and_kinks_from_schedule(
        cls, breakpoints: np.ndarray, multipliers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(first_slope, edges, slope_drops)`` from ``(*cells, K)`` breakpoints and ``(*cells, K + 1)`` multipliers.

        Both are ``nan``-padded, a cell whose multipliers are all ``nan`` is unscheduled, and the inputs are
        valid schedules. A first breakpoint of 0 is folded into the first slope, and equal adjacent multipliers
        carry no kink. ``edges`` and ``slope_drops`` come out as ``(K', *cells)``, without the slots no cell
        uses. Clean under ``simplefilter("error", RuntimeWarning)``.
        """
        b, r = cls._padded_to_common_width(breakpoints, multipliers)
        cells = r.shape[:-1]
        inverse = 1.0 / np.where(np.isnan(r), 1.0, r)
        first_slope = np.where(~np.isnan(r).all(axis=-1), inverse[..., 0], 1.0)
        if b.shape[-1] == 0:
            return first_slope, np.full((0, *cells), np.nan), np.zeros((0, *cells))
        present = ~np.isnan(b) & ~np.isnan(r[..., 1:])
        drops = np.where(present, inverse[..., :-1] - inverse[..., 1:], 0.0)
        first_slope, drops = cls._fold_empty_first_bracket(
            b, present, inverse, first_slope, drops
        )
        edges, slope_drops = cls._drop_unused_kink_slots(
            b, present & (drops != 0.0), drops
        )
        return first_slope, edges, slope_drops

    @staticmethod
    def _padded_to_common_width(
        breakpoints: np.ndarray, multipliers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read both fields as float64 and ``nan``-pad them to ``K`` and ``K + 1`` columns.

        The two fields may be padded to different widths; a valid cell has one more multiplier than kinks.
        """
        b = np.asarray(breakpoints, dtype="float64")
        r = np.asarray(multipliers, dtype="float64")
        cells = r.shape[:-1]
        n_kinks = max(b.shape[-1], r.shape[-1] - 1, 0)
        b = np.concatenate(
            [b, np.full((*cells, n_kinks - b.shape[-1]), np.nan)], axis=-1
        )
        r = np.concatenate(
            [r, np.full((*cells, n_kinks + 1 - r.shape[-1]), np.nan)], axis=-1
        )
        return b, r

    @staticmethod
    def _fold_empty_first_bracket(
        b: np.ndarray,
        present: np.ndarray,
        inverse: np.ndarray,
        first_slope: np.ndarray,
        drops: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Price every unit from the second bracket on where the first breakpoint is 0, and drop its kink."""
        zero_first = present[..., 0] & (b[..., 0] == 0.0)
        first_drop = np.where(zero_first, 0.0, drops[..., 0])
        return (
            np.where(zero_first, inverse[..., 1], first_slope),
            np.concatenate([first_drop[..., None], drops[..., 1:]], axis=-1),
        )

    @staticmethod
    def _drop_unused_kink_slots(
        b: np.ndarray, kink: np.ndarray, drops: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(edges, slope_drops)`` as ``(K', *cells)``, without the bracket slots no cell has a kink in."""
        used = kink.reshape(-1, kink.shape[-1]).any(axis=0)
        edges = np.where(kink, b, np.nan)[..., used]
        slope_drops = np.where(kink, drops, 0.0)[..., used]
        return np.moveaxis(edges, -1, 0), np.moveaxis(slope_drops, -1, 0)

    def bind_period_weights(self, weights: XTensorVariable | None) -> None:
        """Place the knot of each period at ``breakpoint * weight``.

        Parameters
        ----------
        weights : XTensorVariable or None
            Money each period gets per unit of budget, over ``(date_dim, *dims)``, zero outside the mask;
            ``None`` under the uniform spread, where every weight is 1.
        """
        self._weights = weights

    @contextmanager
    def conditioned_on(self, low: np.ndarray, high: np.ndarray) -> Iterator[None]:
        """Condition the knots for one solve over the box ``[low, high]``, and restore the previous ones on exit.

        Per kink, with "at" meaning within ``snap``: a breakpoint at or below ``low`` keeps a hard knot just below
        ``min(B, low)``, so the box sees only the bracket above (a piece whose two ends are one breakpoint takes
        this case, and the solve sees the exact map); one at or above ``high`` gets a hard knot just above
        ``max(B, high)``, so the box sees only the bracket below; one strictly inside is blended over
        ``[B (1 - knot_width), B (1 + knot_width)]``, where the rate rises, and refused where it falls: delivery
        is convex across it, and no local solver can be trusted to find the better side. On exit the knots are
        the ones in place on entry, so a conditioning nested in another gives back the outer one's knots.

        Parameters
        ----------
        low : np.ndarray
            Lower bound of each cell over ``dims``, in budget units, ``nan`` outside the mask.
        high : np.ndarray
            Upper bound of each cell over ``dims``, in budget units, infinite where unbounded and ``nan`` outside
            the mask.

        Raises
        ------
        ValueError
            If the rate falls across a breakpoint strictly inside a cell's bounds.
        """
        shape = self.first_slope.shape
        low = np.broadcast_to(np.asarray(low, dtype="float64"), shape)
        high = np.broadcast_to(np.asarray(high, dtype="float64"), shape)
        below, above, inside = self._classify_kinks_against_box(low, high)
        self._refuse_falling_rate_inside_box(inside)
        knots = self._c_contiguous_float64(self._knots.get_value())
        half_widths = self._c_contiguous_float64(self._half_widths.get_value())
        self._knots.set_value(
            self._c_contiguous_float64(self._knots_off_box(low, high, below, above))
        )
        self._half_widths.set_value(
            self._c_contiguous_float64(
                np.where(inside, self.knot_width * self._edge, 0.0)
            )
        )
        try:
            yield
        finally:
            self._knots.set_value(knots)
            self._half_widths.set_value(half_widths)

    def _classify_kinks_against_box(
        self, low: np.ndarray, high: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per kink, whether its breakpoint is at or below ``low``, at or above ``high``, or strictly inside."""
        # Masked cells have no kink, so their nan bounds select nothing below.
        below = self._kink & (self._edge <= low * (1.0 + self.snap))
        above = self._kink & ~below & (self._edge >= high * (1.0 - self.snap))
        return below, above, self._kink & ~below & ~above

    def _refuse_falling_rate_inside_box(self, inside: np.ndarray) -> None:
        """Refuse a rate that falls across a breakpoint strictly inside a cell's bounds."""
        falling = inside & (self.slope_drops < 0.0)
        if falling.any():
            raise ValueError(
                f"{self.label}: the rate falls across a breakpoint strictly inside the bounds of cells "
                f"{self._format_cell_labels(falling.any(axis=0))}. Delivery is convex there, so each side "
                "of the breakpoint "
                "can hold its own optimum and the solver cannot be trusted to find the better one. Put a bound "
                "on the breakpoint: a lower bound on it optimizes the discounted money above it, an upper bound "
                "on it the money below it, and comparing the two is the choice between the sides. A spend "
                "variable's bounds cannot be set, so a falling schedule on one can be scored with evaluate_plan "
                "and not solved."
            )

    def _knots_off_box(
        self, low: np.ndarray, high: np.ndarray, below: np.ndarray, above: np.ndarray
    ) -> np.ndarray:
        """Hard knots off the box: just below ``min(B, low)`` where ``below``, above ``max(B, high)`` if ``above``."""
        knots = np.where(
            below, np.minimum(self._edge, low) * (1.0 - self.snap), self._default_knots
        )
        return np.where(above, np.maximum(self._edge, high) * (1.0 + self.snap), knots)

    def linear_piece_bounds(
        self,
        budgets: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        from_below: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per cell, the largest interval around ``budgets`` inside ``[low, high]`` with no breakpoint strictly inside.

        A breakpoint within ``snap`` of a bound cuts nothing. A cell on a breakpoint, to within ``snap``, gets
        the piece above it, or the piece below where ``from_below`` is set.

        Parameters
        ----------
        budgets : np.ndarray
            Budget of each cell over ``dims``, in budget units.
        low : np.ndarray
            Lower bound of each cell over ``dims``, in budget units.
        high : np.ndarray
            Upper bound of each cell over ``dims``, in budget units, infinite where unbounded.
        from_below : np.ndarray or None
            Boolean per cell: where set, a cell on a breakpoint gets the piece below it. ``None`` sets it nowhere.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Lower and upper end of the piece of each cell, in budget units, within ``[low, high]``.
        """
        shape = self.first_slope.shape
        budgets = np.broadcast_to(np.asarray(budgets, dtype="float64"), shape)
        low = np.broadcast_to(np.asarray(low, dtype="float64"), shape)
        high = np.broadcast_to(np.asarray(high, dtype="float64"), shape)
        inner = (
            self._kink
            & (self._edge > low * (1.0 + self.snap))
            & (self._edge < high * (1.0 - self.snap))
        )
        at_or_below = self._edge <= budgets * (1.0 + self.snap)
        if from_below is not None:
            strictly_below = self._edge < budgets * (1.0 - self.snap)
            at_or_below = np.where(
                np.broadcast_to(np.asarray(from_below, dtype=bool), shape),
                strictly_below,
                at_or_below,
            )
        piece_low = np.where(inner & at_or_below, self._edge, -np.inf).max(
            axis=0, initial=-np.inf
        )
        piece_high = np.where(inner & ~at_or_below, self._edge, np.inf).min(
            axis=0, initial=np.inf
        )
        return (
            np.asarray(np.maximum(piece_low, low)),
            np.asarray(np.minimum(piece_high, high)),
        )

    def _format_cell_labels(self, where: np.ndarray) -> str:
        """Labels of the first five cells where ``where`` holds, for messages."""
        cells = [
            _cell_label_for_message(
                tuple(
                    self.coords[dim][int(i)]
                    for dim, i in zip(self.dims, index, strict=True)
                )
            )
            for index in np.argwhere(np.asarray(where))[:5]
        ]
        return f"{cells}{', ...' if int(np.sum(where)) > 5 else ''}"

    def _cell_constant(self, values: np.ndarray) -> XTensorVariable:
        """Wrap per-cell values as a float64 constant over ``dims``."""
        return as_xtensor(pt.constant(values, dtype="float64"), dims=self.dims)

    def _weighted_by_period(self, per_cell: XTensorVariable) -> XTensorVariable:
        """Scale a per-cell budget quantity to per-period money by the bound period weights."""
        return per_cell if self._weights is None else per_cell * self._weights

    def _hinge_value_and_slope(
        self, x: XTensorVariable, k: int
    ) -> tuple[XTensorVariable, XTensorVariable]:
        r"""Value and slope of ``max(x, 0)`` at knot ``k``, blended where its half-width is positive.

        The blend :math:`h_w(x) = 2w (t^3 - t^4 / 2)`, :math:`t = \mathrm{clip}((x + w) / 2w, 0, 1)`, equals
        the hinge outside :math:`[-w, w]`, is convex and C2, and exceeds it by at most :math:`3w/16`, at the
        knot. Both ``where`` branches see a finite input: ``safe_w`` is 1 where the knot is hard.
        """
        w = self._weighted_by_period(as_xtensor(self._half_widths[k], dims=self.dims))
        blended = w > 0.0
        safe_w = ptx.math.where(blended, w, 1.0)
        t = ptx.math.clip((x + safe_w) / (2.0 * safe_w), 0.0, 1.0)
        smooth = ptx.math.where(x >= safe_w, x, 2.0 * safe_w * (t**3 - 0.5 * t**4))
        value = ptx.math.where(blended, smooth, ptx.math.maximum(x, 0.0))
        slope = ptx.math.where(
            blended, t * t * (3.0 - 2.0 * t), ptx.math.where(x >= 0.0, 1.0, 0.0)
        )
        return value, slope

    def _delivery_terms(
        self, spend: XTensorVariable, *, slope: bool
    ) -> XTensorVariable:
        """``G`` in per-period money, or its slope in money where ``slope`` is set."""
        if np.finfo(spend.type.dtype).eps > np.finfo(self.dtype).eps:
            raise TypeError(
                f"{self.label}: spend is {spend.type.dtype} but the knots were snapped for {self.dtype}, so "
                "money equal to a breakpoint could land on the wrong side of its knot. Resolve the response "
                "under the floatX the decision vector uses."
            )
        # Money below zero buys nothing. pytensor's Maximum sends the tie gradient to its first input, so
        # spend stays first for the slope at zero to survive.
        s = ptx.math.maximum(spend, 0.0)
        first_slope = self._cell_constant(self.first_slope)
        total = first_slope + 0.0 * s if slope else s * first_slope
        for k in range(self.edges.shape[0]):
            knot = self._weighted_by_period(as_xtensor(self._knots[k], dims=self.dims))
            value, hinge_slope = self._hinge_value_and_slope(s - knot, k)
            total = total - self._cell_constant(self.slope_drops[k]) * (
                hinge_slope if slope else value
            )
        return total

    def to_delivery(
        self, spend: XTensorVariable, base_price: XTensorVariable | None = None
    ) -> XTensorVariable:
        """Delivered units bought with ``spend``; ``u(0) == 0`` exactly.

        Parameters
        ----------
        spend : XTensorVariable
            Per-period money over ``(date_dim, *dims)``.
        base_price : XTensorVariable or None
            Base price per unit over the same dims, ``None`` for a base price of 1.

        Returns
        -------
        XTensorVariable
            Delivered units over ``(date_dim, *dims)``.

        Raises
        ------
        TypeError
            If ``spend`` has a coarser float dtype than the ``floatX`` the knots were snapped for.
        """
        units = self._delivery_terms(spend, slope=False)
        return units if base_price is None else units / base_price

    def implied_marginal_price(
        self, spend: XTensorVariable, base_price: XTensorVariable | None = None
    ) -> XTensorVariable:
        """Money the next delivered unit costs at ``spend``: the base price times the bracket's multiplier.

        Parameters
        ----------
        spend : XTensorVariable
            Per-period money over ``(date_dim, *dims)``.
        base_price : XTensorVariable or None
            Base price per unit over the same dims, ``None`` for a base price of 1.

        Returns
        -------
        XTensorVariable
            Marginal price over ``(date_dim, *dims)``.

        Raises
        ------
        TypeError
            If ``spend`` has a coarser float dtype than the ``floatX`` the knots were snapped for.
        """
        ratio = 1.0 / self._delivery_terms(spend, slope=True)
        return ratio if base_price is None else ratio * base_price


class PriceResponse(BaseModel, ABC):
    """A non-decreasing map from per-period money to delivered units, declared once and bound per variable.

    The declaration is reusable across models and windows; :meth:`resolve` binds it to one decision variable's
    cell layout and returns the object the optimizer's graph uses. Subclasses ship a closed, invertible
    parametric family so the implied delivery and clearing prices can be reported alongside the allocation.

    Every map a family resolves to must be non-decreasing in money with ``u(0) = 0``, and concave wherever a solve
    can reach (see :class:`ResolvedPriceResponse`): a convex map, such as a volume discount, makes the allocation
    non-convex. A family refuses what would break this, either when it is constructed, as
    :class:`PowerPriceResponse` refuses ``elasticity < 0``, or in :meth:`ResolvedPriceResponse.conditioned_on`, as
    :class:`TieredPriceResponse` refuses a solve whose bounds contain a breakpoint where the rate falls. Such a map
    is still useful outside a solve: :meth:`~pymc_marketing.mmm.budget_optimizer.BudgetOptimizer.evaluate_plan`
    scores any plan.

    The optimizer asks three things of a family before resolving it, and keys on the answers rather than on the
    concrete type: where the map *bends* money (:attr:`adds_curvature`, :meth:`curved_cells`), whether the user
    attests that the fitted data are in delivery units (:attr:`attests_delivery_units`, required wherever the
    map bends), and whether the declaration supplies no reference level of its own
    (:attr:`needs_derived_reference`). These are read before :meth:`resolve`, because a refusal must name its
    cause rather than the missing reference spend that cause implies. Declarations are frozen: mutating one after
    construction would bypass its validators.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    @property
    @abstractmethod
    def adds_curvature(self) -> bool:
        """True when the map bends money on some cell, in a way the fitted model may already contain.

        This is what the attestation requirement keys on. Writing the composed second derivative as
        :math:`f''(u) u'^2 + f'(u) u''`, it is the :math:`f'(u) u''` term that double-counts a saturation
        curve fitted on nominal spend. It double-counts when the price-volume relation the map describes was
        already in the fitted history, because the saturation curve absorbed it: a power law in the bid-up of an
        auction is such a relation. A schedule of rates that are new relative to the fit, such as a committed
        rate and a rate for incremental money, describes prices the history did not contain, so a family of
        such schedules answers ``False`` even though its slope changes at each breakpoint.
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
        ignores. Answered without a reference spend, so the optimizer can judge a response before resolving it.
        The default is :attr:`adds_curvature` on every optimized cell; a family whose parameters vary by cell
        overrides it. ``label`` prefixes any error raised while reading the declaration.
        """
        _, on = self._layout(dims, coords, mask)
        return on & self.adds_curvature

    @property
    def attests_delivery_units(self) -> bool:
        """The user attests that the fitted data are in delivery units (or constant-price spend).

        Required of every response that bends a cell, on any node: the fitted model has no evidence of its
        own units, and its historical ``cost_per_unit`` table is a declaration that can be set after the fit.
        Default ``False``; a curved family exposes a field for it.
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
            Per-period money per cell read off the fitted model by the optimizer, in the units of
            ``result.budgets``; ``nan`` on cells with no fitted money (a channel the historical ``cost_per_unit``
            table does not price, a cell never on air). ``None`` when there is nothing to read (a spend variable,
            a custom ``channel_data_var``, a model with no table).
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

    @staticmethod
    def _keyed_by_label_string(value: object, who: str) -> object:
        """Key a dict by the string form of each label, refusing two keys with one string form.

        Channel labels need not be strings (``channel=[1, 2]``), JSON object keys are strings, and the gate
        compares the fitted ``cost_per_unit`` table's columns by their string form; matching on ``str`` serves
        all three. ``who`` names the field in the message. Anything that is not a mapping passes through.
        """
        if not isinstance(value, Mapping):
            return value
        keyed = {str(key): v for key, v in value.items()}
        if len(keyed) != len(value):
            raise ValueError(
                f"{who} keys {list(value)} name the same label more than once: "
                "keys are matched to coordinate labels by their string form."
            )
        return keyed

    @staticmethod
    def _dim_owning_labels(
        keys: set[str], template: DataArray, label: str, what: str
    ) -> str:
        """Return the one budget dim of ``template`` whose labels, as strings, include every key."""
        labels = {d: template.coords[d].values.tolist() for d in template.dims}
        owners = [d for d in template.dims if keys <= {str(lbl) for lbl in labels[d]}]
        if len(owners) != 1:
            where = "none matches" if not owners else f"they match {owners}"
            raise ValueError(
                f"{label}: {what} keys {sorted(map(str, keys))} must all be labels of exactly one "
                f"budget dim; {where}. Available labels: {labels}."
            )
        return str(owners[0])


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
        exactly one budget dim (usually channels) to values; labels not named default to ``0.0``. Keys are
        matched to labels by their string form, as the fitted ``cost_per_unit`` table's columns are, so
        ``{1: 0.3}`` and ``{"1": 0.3}`` both name channel ``1``. A ``DataArray`` over a subset of the budget
        dims is aligned to the model's coordinates. It may not carry the date dim: a date-varying
        ``cost_per_unit`` already covers seasonal price *level*, and seasonal price *sensitivity* is not
        supported.
    reference_spend : xarray.DataArray or None
        Where :math:`p_0` applies: **per-period money per cell, in the units of** ``result.budgets`` **and**
        ``total_budget``, carrying every budget dim. Default ``None`` derives it from the fitted model as the
        mean of ``constant_data["channel_spend"]`` over the periods each cell was on air (``spend > 0``), so a
        flighted channel is anchored at the level it actually bought at. Only channels the historical
        ``cost_per_unit`` table prices get a derived value: an unpriced channel's ``channel_spend`` is its
        units, not money. As :math:`p_0` applies here, anchor it at the spend the window ``cost_per_unit`` was
        quoted at: for a channel whose spend trended over the fit, the whole-fit mean lags today's level, and
        the on-air mean of a trailing window anchors at recent buying (see Examples). A supplied value
        overrides the default cell by cell: its labels may be partial, cells it leaves out (absent labels or
        ``nan``) keep the derived one, and the cells it gives are checked against it where it exists (see
        ``reference_spend_tolerance``). It must cover every curved cell with no derived value: every cell of a
        spend variable or of a custom ``channel_data_var``, and the channels the historical table does not
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
        Attest that the node's data are in delivery units (or in spend deflated to constant prices). Required
        wherever the response bends money, on every node; a cell left at ``elasticity=0`` needs no attestation,
        since its money passes through unbent. Default ``False``: the optimizer then refuses, because a
        saturation curve fitted on nominal spend has already absorbed part of the price curvature and a concave
        price map on top would bend it twice (see :attr:`PriceResponse.adds_curvature`). The fitted model
        cannot establish its own units: its historical ``cost_per_unit`` table is a declaration, written the
        same way by ``MMM(cost_per_unit=...)`` and by :meth:`~pymc_marketing.mmm.mmm.MMM.set_cost_per_unit`
        after the fit, so it does not stand in for this flag. It anchors ``reference_spend`` instead, on the
        channels it prices.

    Notes
    -----
    **Precondition.** Only sound when the model was fitted on delivery units or constant-price spend. The
    optimizer cannot check this and requires ``assume_delivery_units=True`` on every channel the response bends;
    channels left at ``elasticity=0`` need no attestation. The historical ``cost_per_unit`` table on the fitted
    model (written by :meth:`~pymc_marketing.mmm.mmm.MMM.set_cost_per_unit` or ``MMM(cost_per_unit=...)``)
    anchors ``reference_spend`` on the channels it prices; it describes the ``channel_data`` node of ``MMM``
    only, so a custom ``channel_data_var``, a spend variable and any channel the table does not price need a
    ``reference_spend``. The ``cost_per_unit`` passed to the *optimizer* is independent of that table and
    proves nothing about the fit. A merged model (:func:`~pymc_marketing.mmm.budget_optimizer.merge_inference_data`)
    carries no root attrs and so no derived reference.

    **Below the reference.** The power law is as confident below the reference as above it, so a priced channel
    bought far below its reference looks cheap and can attract budget: anchor ``reference_spend`` where you plan
    to buy (raising ``reference_spend_tolerance`` if that is far from the fitted spend) and read
    ``implied_price`` on every channel before trusting the allocation. A variant flat below the reference is
    #3089. The :ref:`cost-per-unit notebook <mmm_cost_per_unit_price_response>` shows this case and the
    precondition end to end.

    **Not a rate schedule.** One smooth curve cannot hold a committed tranche at a contracted rate while the
    increment clears at another: calibrated to that case, its marginal price keeps climbing where the true rate
    is flat. Bracket rates are :class:`TieredPriceResponse`.

    **Units.** The map acts on per-period money at the model's date granularity, after
    ``budget_distribution_over_period`` has redistributed the total. ``total_budget``, ``result.budgets`` and
    ``reference_spend`` are all per-period quantities; the window total is ``budgets * num_periods``. Without
    a window ``cost_per_unit`` the base price is 1 and money reaches the model as units, which the optimizer
    warns about.

    **Behavior change with** :math:`\gamma > 0`. Above the floor each period's price rises with that period's
    money alone, :math:`p_t / p_{0,t} = (s_t / s^{\text{ref}})^{\gamma}`, so concentrated buying clears higher
    relative to the base price. At a base price that is the same in every period, a non-uniform
    ``budget_distribution_over_period`` therefore buys less delivery than a uniform one at equal total spend.
    With a date-varying ``cost_per_unit`` this need not hold: moving money into the cheaper periods can buy
    more and clear lower on average. Across windows, a given per-period plan pays the same price relative to
    the base price whatever the window length. Holding the *window* total fixed, a shorter window spends more
    per period and pays more relative to its base price.

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
    With ``spend_vars``, ``result.delivery_reports[name]`` carries the same arrays plus ``spend`` for every
    priced variable; a spend variable has no ``cost_per_unit``, so its base price is 1 at its
    ``reference_spend``. Score a plan with
    :meth:`~pymc_marketing.mmm.budget_optimizer.BudgetOptimizer.evaluate_response_distribution`, which runs
    the same graph the solver used, price map included. The deprecated ``sample_response_distribution``
    repeats one date-less allocation over the window, so ``implied_delivery.mean(date_dim)`` passed on its
    own matches the delivery the solver used only when that delivery is constant over the window: the
    optimizer's ``budget_distribution_over_period`` is uniform and ``cost_per_unit`` does not vary by date.
    Uniform spend alone is not enough, because a date-varying base price buys different units on different
    dates, and averaging them before adstock and saturation changes the response. By default that method
    also adds input noise and starts from a cold adstock, so it approximates the solver's response rather
    than reproducing it. Above the floor
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
        )  # anchors reference_spend on tv and display
        optimizer = mmm.budget_optimizer(
            start_date,
            end_date,
            cost_per_unit=window_cpu,  # xarray.DataArray over (date, *budget_dims); see BudgetOptimizer
            price_response=PowerPriceResponse(
                elasticity={"tv": 0.25, "display": 0.10},
                assume_delivery_units=True,  # the fit is in impressions, not spend
            ),
        )
        result = optimizer.allocate_budget(total_budget=weekly_budget)
        result.budgets, result.implied_price, result.implied_marginal_price

    Anchor the base price at recent buying instead of the whole fit, here the last 13 weekly periods (a
    quarter), on the priced channels the response bends. A cell off air in that window is ``nan`` and keeps
    the derived default:

    .. code-block:: python

        spend = mmm.idata.constant_data["channel_spend"].isel(date=slice(-13, None))
        recent = spend.where(spend > 0).mean("date").sel(channel=["tv", "display"])
        price_response = PowerPriceResponse(
            elasticity={"tv": 0.25, "display": 0.10},
            reference_spend=recent,
            assume_delivery_units=True,
        )
    """

    elasticity: float | dict[str, float] | InstanceOf[DataArray] = 0.0
    reference_spend: InstanceOf[DataArray] | None = Field(
        default=None,
        description=(
            "Per-period money per cell at which the base price applies, carrying every budget dim; the units "
            "of result.budgets and total_budget. None derives it from the fitted model for the channels its "
            "historical cost_per_unit table prices on MMM's channel_data node (the on-air mean of "
            "constant_data['channel_spend']). A "
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
            "Attest that the node's data are in delivery units (or constant-price spend). Required wherever "
            "the response bends money; the historical cost_per_unit table does not stand in for it. See the "
            "class docstring for why the default refuses."
        ),
    )

    def _known_elasticities(self) -> np.ndarray:
        e = self.elasticity
        if isinstance(e, DataArray):
            return np.asarray(e.values, dtype="float64").ravel()
        if isinstance(e, Mapping):
            return np.asarray(list(e.values()), dtype="float64")
        return np.asarray([e], dtype="float64")

    @field_validator("elasticity", mode="before")
    @classmethod
    def _key_by_label_string(cls, value: object) -> object:
        """Key a dict elasticity by the string form of each label, so non-string labels can be named.

        Channel labels need not be strings (``channel=[1, 2]``), JSON object keys are strings, and the gate
        compares the fitted ``cost_per_unit`` table's columns by their string form; matching on ``str`` serves
        all three.
        """
        return cls._keyed_by_label_string(value, "PowerPriceResponse: elasticity")

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
        e: Mapping[str, float], template: DataArray, label: str
    ) -> DataArray:
        """Spread labeled values along the one budget dim that owns every key, matching labels as strings."""
        dims = template.dims
        if not e:
            return template.copy()
        dim = PriceResponse._dim_owning_labels(set(e), template, label, "elasticity")
        labels = {d: template.coords[d].values.tolist() for d in dims}
        values = DataArray(
            [float(e.get(str(lbl), 0.0)) for lbl in labels[dim]],
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
        reject_unknown_coords(ref, coords, label=f"{label}: reference_spend")
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


class TieredPriceResponse(PriceResponse):
    r"""A schedule of marginal rates: each bracket of money buys units at its own price.

    Money in bracket :math:`i` buys units at ``cost_per_unit * rate_multipliers[i]``, the way a tax schedule
    taxes each bracket of income at its own rate. The rates are marginal, not average: crossing a breakpoint
    reprices the money above it and leaves the money below it alone. For one cell with budget :math:`z`, in the
    units of ``budget_bounds``, ``total_budget`` and ``result.budgets``,

    .. math::

        u(z) = \frac{1}{p_0} \sum_i \frac{\min(z, B_{i+1}) - \min(z, B_i)}{r_i}, \qquad B_0 = 0,\;
        B_{K+1} = \infty

    with :math:`p_0` the base price (``cost_per_unit``, or 1) and :math:`r_i` the multipliers. Delivery is
    increasing and piecewise linear, continuous at every breakpoint, and concave when the multipliers do not
    fall along the brackets. A typical schedule is a committed baseline bought at one rate and incremental money
    at another: ``breakpoints=[committed]``, ``rate_multipliers=[1.0, 1.6]``.

    Parameters
    ----------
    rate_multipliers : sequence of float, dict[str, sequence of float] or xarray.DataArray
        One multiplier of the base price per bracket, ``len(breakpoints) + 1`` per cell. A sequence applies to
        every optimized cell; a bare number is a one-element sequence, a uniform price change. A dict maps
        labels of exactly one budget dim to a sequence, matched by string form as ``elasticity`` keys are, and
        leaves the labels it does not name unscheduled. A ``DataArray`` carries a ``"bracket"`` dim
        (:attr:`~pymc_marketing.names.mmm.Dim.BRACKET`) and a subset of the budget dims; ``nan`` pads a cell
        with fewer brackets and may only trail, and a cell whose multipliers are all ``nan`` is unscheduled.
    breakpoints : sequence of float, dict[str, sequence of float] or xarray.DataArray
        Where each bracket ends, in budget units, strictly ascending and non-negative, in the same form as
        ``rate_multipliers``. Default ``()``: no breakpoint on any cell, whatever form ``rate_multipliers``
        takes. A ``DataArray`` without the bracket dim is one breakpoint per cell, so a committed plan can be
        passed as it is; it names no label that a ``rate_multipliers`` ``DataArray`` leaves out, along the dims
        both have, since the cells of such a label would be unscheduled. A first breakpoint of exactly 0 is an
        empty first bracket, which is what such a plan gives a channel with no commitment. Equal adjacent
        multipliers make a breakpoint with no kink.
    reference_rate : float or None
        The absolute rate a multiplier of 1 stands for, positive and finite, recorded so the declaration says
        what its multipliers are relative to. It changes no number the map computes; it is shown in the repr and
        in the message that refuses a multiplier too far from 1.
    rate_tolerance : float
        Every multiplier must lie in ``[1 / rate_tolerance, rate_tolerance]``. Default 3. Planning rates
        move by tens of percent around the base price, so a multiplier of 8 is an absolute rate typed where a
        ratio was meant, and one of 160 is a percentage. Raise it for a ratio that is really that large.
    knot_width : float
        Half-width of the blend that replaces a breakpoint inside a solve's bounds during the first solve,
        relative to the breakpoint. Default ``1e-2``. It decides how easily that solve converges, not the plan
        that comes back (see Notes).

    Notes
    -----
    **What a multiplier of 1 is.** ``cost_per_unit`` keeps the meaning it has for the constant-price path: the
    level. The multipliers say how the level changes with the money spent. A rate card of 5.5 for the first
    bracket and 8 for the second is written:

    - Fitted on spend, no window ``cost_per_unit`` (a base price of 1): a multiplier of 1 is the fit window's
      average rate, so the card is ``[5.5 / 5, 8 / 5]`` if that average was 5.
    - Fitted on spend, a window ``cost_per_unit``: a multiplier of 1 is ``cost_per_unit`` times that average,
      so the card is ``[1.0, 8 / 5.5]`` with ``cost_per_unit = 5.5 / 5``.
    - Fitted on delivery units, a window ``cost_per_unit``: a multiplier of 1 is ``cost_per_unit``, so the
      card is ``[1.0, 8 / 5.5]`` with ``cost_per_unit = 5.5``.
    - Fitted on delivery units, no window ``cost_per_unit``: money reaches the model as units, and the
      optimizer warns. Pass ``cost_per_unit``.

    On a spend-fitted model the base is the average rate of the fit window: total fitted spend over total
    delivered units. It is not last year's rate and not today's. "TV costs 10% more than last year" is ``[1.1]``
    only if the fit window is last year; with a three-year fit and 4% of yearly media inflation the multiplier is
    about 1.14, and the gap differs by channel. A user with no unit history cannot compute that average and is
    making a judgement call. ``implied_price`` and ``implied_delivery`` are in the units ``cost_per_unit``
    defines: on a spend-fitted model with no window price the price is an index against the fit's average rate
    and the delivery is rate-adjusted money, not impressions.

    **Breakpoints are in budget units**, the number ``budget_bounds`` pins, so a bound and a breakpoint meant to
    coincide are the same float. Under a ``budget_distribution_over_period`` the committed tranche takes the
    plan's time shape: the breakpoint of period :math:`t` is the breakpoint times ``num_periods * f_t``. A
    sequence or a dict gives the same breakpoint to every cell it reaches: on a ``(geo, channel)`` layout
    ``{"tv": [1e6]}`` is one million per period for TV in *each* geo, so a national commitment typed that way is
    applied once per geo. Give commitments as a ``DataArray`` over the cells. Four limits follow from deciding
    one number per cell and pricing it as a multiple of one base price:

    - Prices are per period and a commitment is usually a window total. The two agree when the commitment and
      the increment share one time shape, which the single decision forces; a front-loaded commitment topped
      up evenly is mispriced where the two overlap.
    - "Incremental money only from week 27" cannot be expressed. Optimize the sub-window as its own window.
    - Spending below the first breakpoint prices the unspent commitment as a refund at the committed rate. A
      commitment that cannot be cancelled is a lower bound in ``budget_bounds``, not a price.
    - Every bracket follows the base price's seasonality: a seasonal committed card and a flat open-market rate
      for the increment cannot both be written down.

    **In a solve**, a cell whose best budget is a breakpoint is returned equal to it, bitwise.
    ``allocate_budget`` blends each breakpoint inside the bounds over ``knot_width`` for a first solve, uses that
    solve only to find the linear piece of the schedule each cell is on, and then solves on those pieces, where
    no breakpoint is left to cross, until no cell on a breakpoint would gain by leaving it. When that cannot be
    settled, the best plan found comes back with a warning that it was not settled on its breakpoints.

    **A volume discount**, a multiplier that falls, is accepted. Delivery is convex across that breakpoint, so a
    solve whose bounds contain it is refused, naming the cells: a lower bound on the breakpoint optimizes the
    discounted money above it and an upper bound on it the money below. Any plan can be scored.

    **Not gated on delivery units.** :attr:`adds_curvature` is ``False``, so a schedule needs no
    ``assume_delivery_units`` and works on a model fitted on spend. That holds because the rates are new
    relative to the fit, a new buy or a new strategy, and not because the map is piecewise linear: tiers that
    were already in effect during the fit window were absorbed by the saturation curve, and pricing them again
    double-counts them; that needs a model fitted on delivery units. More than four brackets on a cell warns that
    the schedule looks like a discretized price curve, which is that case.

    **Out of scope.** A tranche that is differently effective per unit. ``implied_price`` is the cost of a unit;
    its effect still comes from the fitted response curve.

    **Scoring a plan.** :meth:`~pymc_marketing.mmm.budget_optimizer.BudgetOptimizer.evaluate_plan` gives the
    objective and gradient of any plan,
    :meth:`~pymc_marketing.mmm.budget_optimizer.BudgetOptimizer.evaluate_response_distribution` its posterior
    response, and :meth:`~pymc_marketing.mmm.budget_optimizer.BudgetOptimizer.delivery_reports` its prices, on
    the schedule exactly as ``allocate_budget`` prices its result. The deprecated
    ``BudgetOptimizerWrapper.sample_response_distribution`` takes neither a ``price_response`` nor a
    ``cost_per_unit``.

    Examples
    --------
    A committed baseline on a spend-fitted model, held by ``budget_bounds`` while the increment is optimized at
    a rate 60% above the base:

    .. code-block:: python

        import xarray as xr

        from pymc_marketing.mmm import TieredPriceResponse

        committed = xr.DataArray(
            [40_000.0, 25_000.0], dims="channel", coords={"channel": ["tv", "radio"]}
        )
        optimizer = mmm.budget_optimizer(
            start_date,
            end_date,
            price_response=TieredPriceResponse(
                breakpoints=committed,
                rate_multipliers=xr.DataArray([1.0, 1.6], dims="bracket"),
            ),
        )
        bounds = xr.DataArray(
            [[40_000.0, 200_000.0], [25_000.0, 200_000.0]],
            dims=("channel", "bound"),
            coords={"channel": ["tv", "radio"], "bound": ["lower", "upper"]},
        )
        result = optimizer.allocate_budget(total_budget=90_000.0, budget_bounds=bounds)
        optimizer.delivery_reports(result.budgets)["channel_data"]
        optimizer.evaluate_plan(result.budgets)

    The same schedule with the default bounds lets the commitment be cut too; a channel left on its
    breakpoint comes back equal to it:

    .. code-block:: python

        result = optimizer.allocate_budget(total_budget=90_000.0)
        (result.budgets == committed).sel(channel="tv")

    A volume discount on TV above 50,000 per period, solved with the lower bound on the breakpoint:

    .. code-block:: python

        discount = TieredPriceResponse(
            breakpoints={"tv": [50_000.0]}, rate_multipliers={"tv": [1.0, 0.85]}
        )
    """

    breakpoints: _Schedule | dict[str, _Schedule] | InstanceOf[DataArray] = ()
    rate_multipliers: _Schedule | dict[str, _Schedule] | InstanceOf[DataArray]
    reference_rate: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    rate_tolerance: float = Field(default=3.0, gt=1.0)
    knot_width: float = Field(default=1e-2, gt=0.0, lt=1.0)

    @field_validator("breakpoints", "rate_multipliers", mode="before")
    @classmethod
    def _schedule_by_label_string(cls, value: object, info: ValidationInfo) -> object:
        """Key a dict by the string form of each label and read a bare number as a one-element sequence.

        Runs before pydantic validates the field, which refuses an integer key for ``dict[str, ...]`` before any
        later validator could see it.
        """
        value = cls._keyed_by_label_string(
            value, f"TieredPriceResponse: {info.field_name}"
        )
        if isinstance(value, Mapping):
            return {
                key: cls._as_one_element_schedule(item) for key, item in value.items()
            }
        return cls._as_one_element_schedule(value)

    @staticmethod
    def _as_one_element_schedule(value: object) -> object:
        """Read a bare number as a one-element sequence; pass anything else through."""
        return (value,) if isinstance(value, numbers.Real) else value

    @staticmethod
    def _row_for_label(value: object, key: str) -> object:
        """Return the row a field gives ``key``: a dict's entry, or the sequence every cell shares."""
        return value[key] if isinstance(value, Mapping) else value

    @staticmethod
    def _schedule_form(value: object) -> str:
        """Name the form a schedule field is given in: ``DataArray``, ``dict`` or ``sequence``."""
        if isinstance(value, DataArray):
            return "DataArray"
        return "dict" if isinstance(value, Mapping) else "sequence"

    @staticmethod
    def _row_as_floats(row: object) -> np.ndarray:
        """Read a sequence as floats; a ``nan`` typed into it is refused as not finite, not read as padding."""
        values = np.asarray(row, dtype="float64").reshape(-1)
        return np.where(np.isnan(values), np.inf, values)

    @staticmethod
    def _pad_rows_with_nan(rows: list[np.ndarray]) -> np.ndarray:
        """Stack rows of different lengths into one array, ``nan``-padded on the right."""
        width = max((row.size for row in rows), default=0)
        out = np.full((len(rows), width), np.nan)
        for i, row in enumerate(rows):
            out[i, : row.size] = row
        return out

    @model_validator(mode="after")
    def _check_schedule(self) -> Self:
        """Refuse an invalid schedule and warn about one with many brackets."""
        label = "TieredPriceResponse"
        self._require_one_form(label)
        if isinstance(self.rate_multipliers, DataArray):
            own, joint = self._own_and_joint_schedule_rows(self.rate_multipliers, label)
        else:
            own, joint = None, self._sequence_schedule_rows(label)
        if own is not None:
            own.refuse_invalid_schedule(self.rate_tolerance, self.reference_rate)
        joint.refuse_invalid_schedule(self.rate_tolerance, self.reference_rate)
        self._warn_many_brackets(joint)
        return self

    def _require_one_form(self, label: str) -> None:
        """Refuse ``breakpoints`` and ``rate_multipliers`` given in different forms or naming different labels."""
        b, r = self.breakpoints, self.rate_multipliers
        b_form, r_form = self._schedule_form(b), self._schedule_form(r)
        no_breakpoints = b_form == "sequence" and len(b) == 0
        if not no_breakpoints and b_form != r_form:
            raise ValueError(
                f"{label}: breakpoints is a {b_form} and rate_multipliers a {r_form}. The two describe one "
                "schedule, so give both in the same form; breakpoints=() goes with any form."
            )
        if b_form == r_form == "dict" and set(b) != set(r):
            raise ValueError(
                f"{label}: breakpoints names {sorted(b)} and rate_multipliers names {sorted(r)}. The two "
                "describe one schedule, so both dicts name the same labels."
            )
        if isinstance(b, DataArray) and isinstance(r, DataArray):
            self._refuse_breakpoint_labels_without_multipliers(b, r, label)

    @staticmethod
    def _refuse_breakpoint_labels_without_multipliers(
        breakpoints: DataArray, multipliers: DataArray, label: str
    ) -> None:
        """Refuse a label ``breakpoints`` names along a dim and ``rate_multipliers`` does not.

        ``rate_multipliers`` leaves a cell it does not name unscheduled, which would drop that cell's breakpoints
        without a word. A dim only one of the two has is broadcast over, so it names nothing to compare; a dim
        without coordinates is refused by :meth:`resolve`.
        """
        for dim in breakpoints.dims:
            if (
                dim == Dim.BRACKET
                or dim not in breakpoints.coords
                or dim not in multipliers.coords
            ):
                continue
            named = set(multipliers.coords[dim].values.tolist())
            missing = [
                lbl
                for lbl in breakpoints.coords[dim].values.tolist()
                if lbl not in named
            ]
            if missing:
                raise ValueError(
                    f"{label}: breakpoints names {missing[:5]} along {dim!r} and rate_multipliers does not. The "
                    "two describe one schedule, so rate_multipliers names every label breakpoints does; nan "
                    "multipliers leave a cell unscheduled."
                )

    def _own_and_joint_schedule_rows(
        self, multipliers: DataArray, label: str
    ) -> tuple[ScheduleRows | None, ScheduleRows]:
        """Rows ``(own, joint)`` of the ``DataArray`` form.

        ``own`` holds the breakpoints on their own cells, ``None`` when there is no breakpoints ``DataArray``, and
        ``joint`` holds both fields on the cells they both describe. ``breakpoints`` is a ``DataArray`` or empty
        here, as :meth:`_require_one_form` checked.
        """
        breakpoints = (
            self.breakpoints if isinstance(self.breakpoints, DataArray) else None
        )
        own = None
        if breakpoints is not None:
            b_rows, b_names = self._rows_from_dataarray(breakpoints)
            own = ScheduleRows(
                breakpoints=b_rows,
                multipliers=np.empty((b_rows.shape[0], 0)),
                name=b_names.__getitem__,
                label=label,
            )
        b_cells, r_cells = self._align_breakpoints_with_multipliers(
            breakpoints, multipliers
        )
        r_rows, names = self._rows_from_dataarray(r_cells)
        joint = ScheduleRows(
            breakpoints=None
            if b_cells is None
            else self._rows_from_dataarray(b_cells)[0],
            multipliers=r_rows,
            name=names.__getitem__,
            label=label,
        )
        return own, joint

    def _sequence_schedule_rows(self, label: str) -> ScheduleRows:
        """Rows of the sequence or dict form, one per label, or one for every cell."""
        b, r = self.breakpoints, self.rate_multipliers
        keys = list(r) if isinstance(r, Mapping) else ["every cell"]
        r_list = [self._row_as_floats(self._row_for_label(r, key)) for key in keys]
        b_list = [self._row_as_floats(self._row_for_label(b, key)) for key in keys]
        empty = [key for key, row in zip(keys, r_list, strict=True) if row.size == 0]
        if empty:
            raise ValueError(
                f"{label}: one multiplier per bracket: rate_multipliers is empty for {empty[:5]}. "
                "A schedule with no breakpoint is one multiplier."
            )
        return ScheduleRows(
            breakpoints=self._pad_rows_with_nan(b_list),
            multipliers=self._pad_rows_with_nan(r_list),
            name=keys.__getitem__,
            label=label,
        )

    @staticmethod
    def _warn_many_brackets(rows: ScheduleRows) -> None:
        """Warn when a cell has more than four brackets, which looks like a discretized price curve."""
        counts = rows.bracket_counts()
        # More than four brackets on one cell looks like a price curve cut into steps (see the Notes).
        many = counts > 4
        if not many.any():
            return
        named = [rows.name(int(i)) for i in np.flatnonzero(many)[:5]]
        warnings.warn(
            f"{rows.label}: {int(counts.max())} brackets on {named}. A schedule with this many brackets looks "
            "like a discretized price curve. If those tiers were already in effect during the fit window, the "
            "saturation curve absorbed them and pricing them again double-counts them: that needs a model "
            "fitted on delivery units and a curved family such as PowerPriceResponse.",
            UserWarning,
            # Raised from a pydantic validator: the frames are this method, the validator, pydantic's
            # constructor, the caller.
            stacklevel=4,
        )

    @staticmethod
    def _rows_from_dataarray(value: DataArray) -> tuple[np.ndarray, list]:
        """Split a schedule ``DataArray`` into rows, one per cell it describes, with ``bracket`` along the columns."""
        if Dim.BRACKET not in value.dims:
            value = value.expand_dims(Dim.BRACKET, axis=-1)
        cell_dims = [dim for dim in value.dims if dim != Dim.BRACKET]
        ordered = value.transpose(*cell_dims, Dim.BRACKET)
        shape = tuple(ordered.sizes[dim] for dim in cell_dims)
        rows = np.asarray(ordered.values, dtype="float64").reshape(
            int(np.prod(shape)), ordered.sizes[Dim.BRACKET]
        )
        if not cell_dims:
            return rows, ["every cell"]
        names = [
            _cell_label_for_message(
                tuple(
                    ordered.coords[dim].values.tolist()[i]
                    if dim in ordered.coords
                    else i
                    for dim, i in zip(cell_dims, index, strict=True)
                )
            )
            for index in np.ndindex(*shape)
        ]
        return rows, names

    @staticmethod
    def _align_breakpoints_with_multipliers(
        breakpoints: DataArray | None, multipliers: DataArray
    ) -> tuple[DataArray | None, DataArray]:
        """Lay two schedule arrays over the cells both describe, so their counts can be compared.

        Without breakpoints every cell has none. Cells only one of them names, and layouts xarray cannot align,
        are left to :meth:`resolve`, which aligns both to the model's cells.
        """
        multipliers = (
            multipliers
            if Dim.BRACKET in multipliers.dims
            else multipliers.expand_dims(Dim.BRACKET, axis=-1)
        )
        if breakpoints is None:
            return (
                multipliers.isel({Dim.BRACKET: slice(0, 0)}),
                multipliers,
            )
        breakpoints = (
            breakpoints
            if Dim.BRACKET in breakpoints.dims
            else breakpoints.expand_dims(Dim.BRACKET, axis=-1)
        )
        try:
            breakpoints, multipliers = xr.align(
                breakpoints, multipliers, join="inner", exclude=[Dim.BRACKET]
            )
            breakpoints, multipliers = xr.broadcast(
                breakpoints, multipliers, exclude=[Dim.BRACKET]
            )
        except ValueError:
            return None, multipliers
        cell_dims = [dim for dim in multipliers.dims if dim != Dim.BRACKET]
        return (
            breakpoints.transpose(*cell_dims, Dim.BRACKET),
            multipliers.transpose(*cell_dims, Dim.BRACKET),
        )

    @property
    def adds_curvature(self) -> bool:
        """A schedule of rates that are new relative to the fit adds no curvature the fit already holds.

        See :attr:`PriceResponse.adds_curvature`: the slope changes at each breakpoint, and what keeps the answer
        ``False`` is that the rates describe prices the fitted history did not contain.
        """
        return False

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
    ) -> ResolvedTieredPriceResponse:
        """Bind the schedule to one variable's cell layout.

        Parameters
        ----------
        dims : tuple[str, ...]
            The variable's budget dims in the model's order.
        coords : Mapping[str, list]
            Coordinate labels of each dim in ``dims``.
        mask : DataArray
            Boolean mask over ``dims`` selecting the optimized cells; a masked cell is left unscheduled.
        date_dim : str
            Name of the date dimension, refused on a ``DataArray`` field: the schedule has no date axis.
        derived_reference : DataArray or None
            Not read: the schedule is stated in money.
        label : str
            Prefix for error messages, naming the variable.
        num_periods : int or None
            Not read: the schedule has no date axis.

        Returns
        -------
        ResolvedTieredPriceResponse
            The schedule's coefficients per cell of the layout.

        Raises
        ------
        ValueError
            If a budget dim is named ``bracket``, if a field names labels or dims the layout does not have or
            carries unlabelled dims, or if the schedule on the layout's cells is invalid.
        """
        template, on = self._layout(dims, coords, mask)
        self._refuse_bracket_dim(template, label)
        breakpoints, multipliers = self._schedule_rows_per_cell(
            template, on, date_dim, label
        )
        ScheduleRows(
            breakpoints=breakpoints,
            multipliers=multipliers,
            name=lambda i: _cell_label_for_message(
                self._cell_label(
                    np.asarray(np.unravel_index(i, template.shape)), template
                )
            ),
            label=label,
        ).refuse_invalid_schedule(self.rate_tolerance, self.reference_rate)
        first_slope, edges, slope_drops = (
            ResolvedTieredPriceResponse._slopes_and_kinks_from_schedule(
                breakpoints.reshape(*template.shape, -1),
                multipliers.reshape(*template.shape, -1),
            )
        )
        return ResolvedTieredPriceResponse(
            first_slope=first_slope,
            edges=edges,
            slope_drops=slope_drops,
            dims=template.dims,
            coords={dim: template.coords[dim].values.tolist() for dim in template.dims},
            knot_width=self.knot_width,
            label=label,
        )

    @staticmethod
    def _refuse_bracket_dim(template: DataArray, label: str) -> None:
        """Refuse a budget dim named ``bracket``: the schedule's own dim has that name."""
        if Dim.BRACKET in template.dims:
            raise ValueError(
                f"{label}: the model has a budget dim named {Dim.BRACKET!r}, and the name is taken by the "
                "schedule's own dim. Rename the model's dim to use a TieredPriceResponse."
            )

    def _schedule_rows_per_cell(
        self, template: DataArray, on: np.ndarray, date_dim: str, label: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Breakpoints and multipliers with one row per cell of ``template``, ``nan`` where masked or unscheduled."""
        multipliers = self._field_per_cell(
            self.rate_multipliers, template, date_dim, label, "rate_multipliers"
        ).reshape(on.size, -1)
        breakpoints = self._field_per_cell(
            self.breakpoints, template, date_dim, label, "breakpoints"
        ).reshape(on.size, -1)
        scheduled = (on.reshape(-1) & ~np.isnan(multipliers).all(axis=1))[:, None]
        return (
            np.where(scheduled, breakpoints, np.nan),
            np.where(scheduled, multipliers, np.nan),
        )

    def _field_per_cell(
        self,
        value: _Schedule | Mapping[str, _Schedule] | DataArray,
        template: DataArray,
        date_dim: str,
        label: str,
        what: str,
    ) -> np.ndarray:
        """One field of the schedule per cell of ``template``, shape ``(*cells, K)``, ``nan`` where absent."""
        if isinstance(value, DataArray):
            return self._align_field_to_layout(value, template, date_dim, label, what)
        if isinstance(value, Mapping):
            if not value:
                return np.full((*template.shape, 0), np.nan)
            dim = self._dim_owning_labels(set(value), template, label, what)
            labels = [str(lbl) for lbl in template.coords[dim].values.tolist()]
            per_label = self._pad_rows_with_nan(
                [
                    self._row_as_floats(value[lbl]) if lbl in value else np.empty(0)
                    for lbl in labels
                ]
            )
            return per_label[np.indices(template.shape)[template.dims.index(dim)]]
        row = self._row_as_floats(value)
        return np.broadcast_to(row, (*template.shape, row.size)).copy()

    def _align_field_to_layout(
        self,
        value: DataArray,
        template: DataArray,
        date_dim: str,
        label: str,
        what: str,
    ) -> np.ndarray:
        """Align a ``DataArray`` field to the layout, ``nan`` on the labels it leaves out."""
        extra = sorted(set(value.dims) - set(template.dims) - {Dim.BRACKET})
        if extra:
            raise ValueError(
                f"{label}: {what} has dims {extra}, which are neither budget dims {list(template.dims)} nor "
                f"{Dim.BRACKET!r}: the schedule has no date axis; the seasonal level goes in cost_per_unit."
                + (f" Drop the {date_dim!r} dim." if date_dim in extra else "")
            )
        cells = (
            value.isel({Dim.BRACKET: 0}, drop=True)
            if Dim.BRACKET in value.dims
            else value
        )
        self._require_labelled(cells, f"{label}: {what}")
        coords = {
            dim: template.coords[dim].values.tolist()
            for dim in value.dims
            if dim != Dim.BRACKET
        }
        reject_unknown_coords(value, coords, label=f"{label}: {what}")
        aligned = value.reindex(coords).astype("float64")
        if Dim.BRACKET not in aligned.dims:
            aligned = aligned.expand_dims(Dim.BRACKET, axis=-1)
        aligned = aligned.drop_vars(Dim.BRACKET, errors="ignore")
        full = aligned.broadcast_like(template).transpose(*template.dims, Dim.BRACKET)
        return np.asarray(full.values, dtype="float64")
