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
"""The spend-dependent price map: its shape, its floor, and its declaration."""

import warnings

import numpy as np
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import pytest
from pytensor import function
from pytensor.graph import rewrite_graph
from pytensor.xtensor import as_xtensor

from pymc_marketing.mmm.price_response import ResolvedPowerPriceResponse

# The optimizer lowers before differentiating (budget_optimizer.py:2687-2690);
# xtensor ops have no L_op, so every gradient here goes the same way.
LOWER = ("lower_xtensor", "canonicalize", "stabilize")
DIMS = ("channel",)
GAMMA = np.array([0.0, 0.25, 0.5])
S_REF = np.array([100.0, 100.0, 100.0])
P0 = np.array([3.0, 3.0, 3.0])
M = 100.0


def compile_lowered(output, spend):
    """Compile an xtensor graph the way the optimizer does: lowered, xtensor input kept."""
    return function([spend], rewrite_graph(output.values, include=LOWER))


def slope(u, s, h):
    return (u(s + h) - u(s - h)) / (2 * h)


@pytest.fixture
def resolved() -> ResolvedPowerPriceResponse:
    return ResolvedPowerPriceResponse(
        gamma=GAMMA, reference_spend=S_REF, max_slope_ratio=M, dims=DIMS, label="test"
    )


@pytest.fixture
def spend():
    return ptx.xtensor("spend", shape=(3,), dims=DIMS)


@pytest.fixture
def base_price():
    return as_xtensor(pt.constant(P0), dims=DIMS)


@pytest.fixture
def maps(resolved, spend, base_price):
    return {
        "u": compile_lowered(resolved.to_delivery(spend, base_price), spend),
        "p": compile_lowered(resolved.implied_price(spend, base_price), spend),
        "m": compile_lowered(resolved.implied_marginal_price(spend, base_price), spend),
    }


class TestResolvedPowerPriceResponse:
    def test_mixed_elasticities_resolve_without_warnings_and_finite(self):
        """The floor formulas are undefined at gamma = 0, and a gamma = 0 cell is the
        normal baseline in a sweep. The double-where must keep every coefficient finite
        and warning-free: a nan in a dead branch is harmless on the C backend and poisons
        the gradient under JAX."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            r = ResolvedPowerPriceResponse(
                gamma=GAMMA,
                reference_spend=S_REF,
                max_slope_ratio=M,
                dims=DIMS,
                label="t",
            )
        for name in ("s_floor", "a", "b", "scale"):
            assert np.all(np.isfinite(getattr(r, name))), name
        assert r.s_floor[0] == 0.0 and r.a[0] == 1.0 and r.b[0] == 0.0
        assert not r.is_identity
        np.testing.assert_allclose(
            r.s_floor / S_REF, [0.0, 7.716e-08, 9.0e-04], rtol=1e-3
        )

    def test_zero_spend_delivers_exactly_nothing(self, maps):
        """A tangent-line floor would deliver gamma * u(s_f) for free at zero spend."""
        assert np.all(maps["u"](np.zeros(3)) == 0.0)

    def test_zero_elasticity_cell_is_bitwise_the_constant_price_graph(
        self, maps, spend, base_price
    ):
        """Compared through the same compile pipeline; NumPy differs in the last ulp
        because canonicalize rewrites x / c into x * (1 / c)."""
        old = compile_lowered(spend / base_price, spend)
        s = np.array([7.0, 37.0, 123.456])
        assert maps["u"](s)[0] == old(s)[0]

    def test_closed_form_above_the_floor(self, maps):
        s = np.full(3, 60.0)
        np.testing.assert_allclose(
            maps["u"](s), s ** (1 - GAMMA) * S_REF**GAMMA / P0, rtol=1e-12
        )
        np.testing.assert_allclose(maps["p"](s), P0 * (s / S_REF) ** GAMMA, rtol=1e-12)

    def test_money_identity_holds_in_both_regions(self, maps, resolved):
        """u * p == s is exact by construction above and below the floor."""
        for s in (np.full(3, 60.0), resolved.s_floor * 0.5):
            np.testing.assert_allclose(
                maps["u"](s) * maps["p"](s), s, rtol=1e-12, atol=0.0
            )

    def test_marginal_price_is_the_reciprocal_slope_in_both_regions(
        self, maps, resolved
    ):
        for s in (np.full(3, 60.0), resolved.s_floor * 0.5):
            active = s > 0
            h = np.maximum(s * 1e-6, 1e-12)
            fd = slope(maps["u"], s, h)
            np.testing.assert_allclose(
                maps["m"](s)[active], 1.0 / fd[active], rtol=1e-5
            )

    def test_map_is_c1_and_both_prices_are_continuous_across_the_floor(
        self, maps, resolved
    ):
        """Value, slope, average price and marginal price all match at s_f. The marginal
        tolerance is looser because its own slope near the floor is steep (about 2 per
        money unit at gamma = 0.9): a one-sided step of 1e-7 s_f moves it by ~1e-6
        relative. That is the step, not a kink."""
        act = GAMMA > 0
        sf = resolved.s_floor
        eps = np.maximum(sf * 1e-7, 1e-15)
        np.testing.assert_allclose(
            slope(maps["u"], sf - 2 * eps, eps)[act],
            slope(maps["u"], sf + 2 * eps, eps)[act],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            maps["p"](sf - eps)[act], maps["p"](sf + eps)[act], rtol=1e-6
        )
        np.testing.assert_allclose(
            maps["m"](sf - eps)[act], maps["m"](sf + eps)[act], rtol=1e-5
        )

    def test_marginal_price_relation_to_average_price(self, maps, resolved):
        """m == p / (1 - gamma) is a power-branch statement. At zero spend both prices
        equal p0 / a, and m(0) / m(s_f) == (1 - gamma) / (1 + gamma)."""
        s = np.full(3, 60.0)
        np.testing.assert_allclose(maps["m"](s), maps["p"](s) / (1 - GAMMA), rtol=1e-12)
        zero = np.zeros(3)
        np.testing.assert_allclose(maps["p"](zero), P0 / resolved.a, rtol=1e-12)
        np.testing.assert_allclose(maps["m"](zero), P0 / resolved.a, rtol=1e-12)
        act = GAMMA > 0
        np.testing.assert_allclose(
            (maps["m"](zero) / maps["m"](resolved.s_floor))[act],
            ((1 - GAMMA) / (1 + GAMMA))[act],
            rtol=1e-6,
        )

    def test_gradient_is_finite_at_zero_and_its_spread_is_max_slope_ratio(
        self, resolved, spend, base_price
    ):
        """The knob caps the number SLSQP actually meets: u'(0) / u'(s_ref) == M on every
        active cell, and the gradient is finite at exactly 0.0, which default_bounds allow."""
        objective = rewrite_graph(
            resolved.to_delivery(spend, base_price).sum().values, include=LOWER
        )
        grad = function([spend], pt.grad(objective, spend))
        at_zero, at_ref = grad(np.zeros(3)), grad(S_REF)
        assert np.all(np.isfinite(at_zero))
        act = GAMMA > 0
        np.testing.assert_allclose((at_zero / at_ref)[act], M, rtol=1e-8)
        assert at_zero[0] == at_ref[0]

    def test_map_is_monotone_and_concave(self, maps):
        grid = np.linspace(0.0, 400.0, 2001)
        for i in range(3):
            values = np.array([maps["u"](np.full(3, g))[i] for g in grid])
            first, second = np.diff(values), np.diff(np.diff(values))
            assert np.all(first > 0), f"cell {i} not increasing"
            assert np.all(second <= 1e-12), f"cell {i} not concave"

    def test_wide_floor_warns_at_high_elasticity_only(self):
        """At gamma = 0.9 and M = 100 the floor is 15.8% of the reference: a bounded slope
        spread and a narrow quadratic region are not both available there."""
        with pytest.warns(UserWarning, match="quadratic floor"):
            ResolvedPowerPriceResponse(
                gamma=np.array([0.9]),
                reference_spend=np.array([100.0]),
                max_slope_ratio=M,
                dims=DIMS,
                label="t",
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ResolvedPowerPriceResponse(
                gamma=np.array([0.25]),
                reference_spend=np.array([100.0]),
                max_slope_ratio=M,
                dims=DIMS,
                label="t",
            )
