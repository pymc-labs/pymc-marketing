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

import numpy as np
import pytensor.tensor as pt
import pytensor.xtensor as ptx
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable

__all__ = [
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
