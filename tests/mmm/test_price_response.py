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

import importlib.util
import warnings

import numpy as np
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import pytest
import xarray as xr
from pydantic import ValidationError
from pytensor import function
from pytensor.graph import rewrite_graph
from pytensor.xtensor import as_xtensor

from pymc_marketing.mmm.price_response import (
    MIN_FLOOR_FRACTION,
    PowerPriceResponse,
    PriceResponse,
    ResolvedPowerPriceResponse,
    TieredPriceResponse,
)

LOWER = ("lower_xtensor", "canonicalize", "stabilize")
DIMS = ("channel",)
DIMS_COORDS = {"channel": ["tv", "radio", "digital"]}
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
        below = np.where(resolved.s_floor > 0.0, resolved.s_floor * 0.5, 7.0)
        for s in (np.full(3, 60.0), below):
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
        """m == p / (1 - gamma) is a power-branch statement. As spend falls to zero both
        prices tend to p0 / a; at zero nothing is bought, so p is nan there while
        m(0) == p0 / a, and m(0) / m(s_f) == (1 - gamma) / (1 + gamma)."""
        s = np.full(3, 60.0)
        np.testing.assert_allclose(maps["m"](s), maps["p"](s) / (1 - GAMMA), rtol=1e-12)
        zero = np.zeros(3)
        np.testing.assert_allclose(
            maps["p"](np.full(3, 1e-12)), P0 / resolved.a, rtol=1e-6
        )
        assert np.all(np.isnan(maps["p"](zero)))
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

    def test_mixed_elasticities_gradient_is_finite_under_jax(
        self, resolved, spend, base_price
    ):
        """The double-where in _power_floor_coefficients exists for this backend: a nan in a
        dead branch is harmless on the C backend and poisons jax.grad. Same numbers on both."""
        pytest.importorskip("jax")
        objective = rewrite_graph(
            resolved.to_delivery(spend, base_price).sum().values, include=LOWER
        )
        gradient = pt.grad(objective, spend)
        c_backend = function([spend], gradient)
        jax_backend = function([spend], gradient, mode="JAX")
        for s in (np.zeros(3), resolved.s_floor, np.full(3, 50.0)):
            np.testing.assert_allclose(jax_backend(s), c_backend(s), rtol=1e-12)
            assert np.all(np.isfinite(jax_backend(s)))

    @pytest.mark.parametrize("mode", [None, "JAX"], ids=["c", "jax"])
    def test_marginal_price_gradient_is_finite_on_every_cell(
        self, resolved, spend, base_price, mode
    ):
        """The contract asks implied_marginal_price for a finite gradient as well. A flat cell
        has no floor, so at zero money (and below it, clipped to zero) its power branch
        differentiated 0 ** 0 and gave nan; its ratio is 1 at any spend, so its gradient is 0."""
        if mode == "JAX":
            pytest.importorskip("jax")
        marginal = resolved.implied_marginal_price(spend, base_price)
        objective = rewrite_graph(marginal.sum().values, include=LOWER)
        gradient = function([spend], pt.grad(objective, spend), mode=mode)

        for s in (np.zeros(3), np.full(3, -1.0), resolved.s_floor, np.full(3, 50.0)):
            at_s = gradient(s)
            assert np.all(np.isfinite(at_s)), s
            assert at_s[0] == 0.0

    @pytest.mark.parametrize("gamma", [1e-6, 5e-3, 1e-2])
    def test_small_elasticity_keeps_the_floor_positive_and_the_gradient_finite(
        self, gamma
    ):
        """The clamp binds at all three gammas (it does below gamma ~ 0.16 at M = 100), and
        below ~ 0.006 inner ** (-1 / gamma) underflows to 0: without the clamp the floor
        vanishes, a and b overflow, and the singularity at zero is back: grad(0) is inf and
        SLSQP fails from a channel started at 0. With the clamp binding the spread is
        (1 + gamma) / (1 - gamma) * MIN_FLOOR_FRACTION ** -gamma, under M exactly then."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            resolved = ResolvedPowerPriceResponse(
                gamma=np.array([gamma]),
                reference_spend=np.array([100.0]),
                max_slope_ratio=M,
                dims=DIMS,
                label="test",
            )
        assert resolved.s_floor[0] > 0.0
        assert np.isfinite(resolved.a).all() and np.isfinite(resolved.b).all()
        spend = ptx.xtensor("spend", shape=(1,), dims=DIMS)
        objective = rewrite_graph(
            resolved.to_delivery(spend).sum().values, include=LOWER
        )
        grad = function([spend], pt.grad(objective, spend))
        at_zero, at_ref = grad(np.zeros(1)), grad(np.array([100.0]))
        assert np.isfinite(at_zero).all()
        assert at_zero[0] <= M * at_ref[0] * (1 + 1e-9)
        np.testing.assert_allclose(
            at_zero[0] / at_ref[0],
            (1 + gamma) / (1 - gamma) * MIN_FLOOR_FRACTION**-gamma,
            rtol=1e-8,
        )
        u = compile_lowered(resolved.to_delivery(spend), spend)
        assert u(np.zeros(1))[0] == 0.0
        np.testing.assert_allclose(u(np.array([100.0]))[0], 100.0, rtol=1e-12)

    def test_map_is_monotone_and_concave(self, maps):
        grid = np.linspace(0.0, 400.0, 2001)
        for i in range(3):
            values = np.array([maps["u"](np.full(3, g))[i] for g in grid])
            first, second = np.diff(values), np.diff(np.diff(values))
            assert np.all(first > 0), f"cell {i} not increasing"
            assert np.all(second <= 1e-12), f"cell {i} not concave"

    def test_negative_money_buys_nothing(self, maps, resolved):
        """SLSQP clips the decision vector to its bounds, but evaluate_plan takes any
        labelled plan. Below zero the quadratic would extrapolate (b < 0 makes -5 money
        deliver about -1.6e7 units on the gamma = 0.25 cell of this fixture); money is
        clipped at zero instead, so u(s < 0) == 0, no unit was paid for (the average price
        is nan) and the marginal price stays at its finite value at zero."""
        s = np.array([-1e-9, -5.0, -100.0])
        assert np.all(maps["u"](s) == 0.0)
        assert np.all(np.isnan(maps["p"](s)))
        np.testing.assert_allclose(maps["m"](s), P0 / resolved.a, rtol=1e-12)

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


def layout(dims=("channel",), coords=None, mask_values=None):
    """What a MediaVariable hands to resolve(): its dims, coords and mask."""
    coords = coords or {"channel": ["tv", "radio", "digital"]}
    shape = tuple(len(coords[d]) for d in dims)
    values = (
        np.ones(shape, dtype=bool)
        if mask_values is None
        else np.asarray(mask_values, dtype=bool)
    )
    mask = xr.DataArray(values, dims=dims, coords=coords)
    return dict(
        dims=dims,
        coords=coords,
        mask=mask,
        date_dim="date",
        label="channel_data: price_response",
    )


def derived(values, coords=None):
    coords = coords or {"channel": ["tv", "radio", "digital"]}
    return xr.DataArray(
        np.asarray(values, dtype=float), dims=tuple(coords), coords=coords
    )


class TestPowerPriceResponseValidation:
    @pytest.mark.parametrize("elasticity", [1.0, 1.5, -0.1])
    def test_elasticity_domain_is_half_open(self, elasticity):
        """At gamma = 1 delivery is constant in spend; above it the map decreases and the
        optimizer drives the channel to its lower bound."""
        with pytest.raises(ValueError, match="0 <= elasticity < 1"):
            PowerPriceResponse(elasticity=elasticity)
        with pytest.raises(ValueError, match="0 <= elasticity < 1"):
            PowerPriceResponse(elasticity={"tv": elasticity})

    def test_max_slope_ratio_must_exceed_the_endpoint_overshoot(self):
        """(1 + g) / (1 - g) is 3 at g = 0.5; at or below it the floor lands above the reference."""
        with pytest.raises(ValueError, match="max_slope_ratio must exceed 3"):
            PowerPriceResponse(elasticity=0.5, max_slope_ratio=3.0)
        PowerPriceResponse(elasticity=0.5, max_slope_ratio=3.01)

    @pytest.mark.parametrize(
        "elasticity, expected",
        [
            (0.0, False),
            ({}, False),
            ({"tv": 0.0}, False),
            ({"tv": 0.1}, True),
            (
                xr.DataArray(
                    [0.0, 0.0], dims=("channel",), coords={"channel": ["tv", "radio"]}
                ),
                False,
            ),
            (0.2, True),
        ],
    )
    def test_curved_only_when_some_elasticity_is_not_zero(self, elasticity, expected):
        assert PowerPriceResponse(elasticity=elasticity).adds_curvature is expected

    def test_an_elasticity_mutated_in_place_is_still_refused_at_resolve(self):
        """frozen stops a field being reassigned, not the caller's DataArray or dict it
        holds being mutated, so resolve checks the domain again on the cells it binds."""
        e = xr.DataArray(
            [0.1, 0.2, 0.3],
            dims=("channel",),
            coords={"channel": ["tv", "radio", "digital"]},
        )
        response = PowerPriceResponse(elasticity=e, reference_spend=derived([1, 1, 1]))
        e[:] = 1.5
        with pytest.raises(ValueError, match="0 <= elasticity < 1"):
            response.resolve(**layout(), derived_reference=None)

    def test_scalar_elasticity_broadcasts_to_every_cell(self):
        resolved = PowerPriceResponse(
            elasticity=0.2, reference_spend=derived([1, 2, 3])
        ).resolve(**layout(), derived_reference=None)
        np.testing.assert_array_equal(resolved.gamma, [0.2, 0.2, 0.2])
        np.testing.assert_array_equal(resolved.reference_spend, [1.0, 2.0, 3.0])

    def test_mapping_keys_must_be_labels_of_exactly_one_dim(self):
        coords = {"geo": ["US", "UK"], "channel": ["tv", "radio"]}
        two_d = layout(dims=("geo", "channel"), coords=coords)
        reference = derived(np.full((2, 2), 50.0), coords=coords)

        resolved = PowerPriceResponse(
            elasticity={"tv": 0.3}, reference_spend=reference
        ).resolve(**two_d, derived_reference=None)
        np.testing.assert_array_equal(resolved.gamma, [[0.3, 0.0], [0.3, 0.0]])

        with pytest.raises(ValueError, match="none matches"):
            PowerPriceResponse(
                elasticity={"nowhere": 0.3}, reference_spend=reference
            ).resolve(**two_d, derived_reference=None)
        with pytest.raises(ValueError, match="none matches"):
            PowerPriceResponse(
                elasticity={"tv": 0.3, "US": 0.1}, reference_spend=reference
            ).resolve(**two_d, derived_reference=None)
        clashing = {"geo": ["US", "tv"], "channel": ["tv", "radio"]}
        with pytest.raises(ValueError, match=r"they match \['geo', 'channel'\]"):
            PowerPriceResponse(
                elasticity={"tv": 0.3},
                reference_spend=derived(np.full((2, 2), 50.0), coords=clashing),
            ).resolve(
                **layout(dims=("geo", "channel"), coords=clashing),
                derived_reference=None,
            )

    def test_mapping_keys_match_non_string_labels_by_their_string_form(self):
        """Channel labels need not be strings. A dict is matched the way the fitted cost_per_unit
        table is, by string form, so an integer key and its string both name the integer label;
        two keys naming one label are refused rather than one of them silently winning."""
        coords = {"channel": [1, 2, 3]}
        reference = derived([50.0, 50.0, 50.0], coords=coords)
        for elasticity in ({1: 0.3}, {"1": 0.3}):
            resolved = PowerPriceResponse(
                elasticity=elasticity, reference_spend=reference
            ).resolve(**layout(coords=coords), derived_reference=None)
            np.testing.assert_array_equal(resolved.gamma, [0.3, 0.0, 0.0])
        with pytest.raises(ValidationError, match="same label more than once"):
            PowerPriceResponse(elasticity={1: 0.3, "1": 0.2})

    def test_dataarray_elasticity_with_a_date_dim_is_rejected(self):
        e = xr.DataArray(
            np.zeros((2, 3)),
            dims=("date", "channel"),
            coords={"date": [0, 1], "channel": ["tv", "radio", "digital"]},
        )
        with pytest.raises(ValueError, match="seasonal price"):
            PowerPriceResponse(
                elasticity=e, reference_spend=derived([1, 1, 1])
            ).resolve(**layout(), derived_reference=None)
        with pytest.raises(
            ValueError, match="channel_data: price_response: elasticity varies"
        ):
            PowerPriceResponse(elasticity=e + 0.2).curved_cells(**layout())

    def test_dataarray_elasticity_with_an_unknown_label_is_rejected(self):
        e = xr.DataArray(
            [0.1, 0.2], dims=("channel",), coords={"channel": ["tv", "print"]}
        )
        with pytest.raises(ValueError, match="coordinates the model does not have"):
            PowerPriceResponse(
                elasticity=e, reference_spend=derived([1, 1, 1])
            ).resolve(**layout(), derived_reference=None)

    def test_supplied_reference_must_carry_exactly_the_budget_dims(self):
        with_date = xr.DataArray(
            np.ones((2, 3)),
            dims=("date", "channel"),
            coords={"date": [0, 1], "channel": ["tv", "radio", "digital"]},
        )
        with pytest.raises(ValueError, match="per-period money per cell"):
            PowerPriceResponse(elasticity=0.2, reference_spend=with_date).resolve(
                **layout(), derived_reference=None
            )

    def test_a_reference_that_is_num_periods_times_the_fitted_spend_is_named_as_a_window_total(
        self,
    ):
        """The mistake the tolerance guard exists for is a window total mistaken for a
        per-period rate, off by exactly num_periods -- and a typical window of 4 to 13
        periods sits under the 10x default. Given num_periods, the hypothesis is tested
        by name on every optimized cell, independently of the tolerance. It is a
        heuristic (5%, every cell), so it warns rather than refuses: a genuine
        num_periods-fold plan is not blocked, and the generic factor check still refuses
        the real unit error on any window longer than the tolerance."""
        fitted = derived([100.0, 200.0, 50.0])
        with pytest.warns(UserWarning, match=r"num_periods \(4\).*window total"):
            PowerPriceResponse(elasticity=0.2, reference_spend=fitted * 4).resolve(
                **layout(), derived_reference=fitted, num_periods=4
            )
        with pytest.warns(UserWarning, match="window total"):
            PowerPriceResponse(elasticity=0.2, reference_spend=fitted * 4.1).resolve(
                **layout(), derived_reference=fitted, num_periods=4
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mixed = derived([400.0, 200.0, 200.0])
            PowerPriceResponse(elasticity=0.2, reference_spend=mixed).resolve(
                **layout(), derived_reference=fitted, num_periods=4
            )
            PowerPriceResponse(elasticity=0.2, reference_spend=fitted * 4).resolve(
                **layout(), derived_reference=fitted
            )

    def test_supplied_reference_far_from_the_derived_one_is_rejected_naming_both(self):
        """A reference summed over a 52-week window is off by 52x and would shift every
        price by 52 ** gamma with nothing in the output saying so."""
        fitted = derived([100.0, 100.0, 100.0])
        with pytest.raises(
            ValueError, match=r"5200.*\b100\b.*52x apart.*per-period money"
        ):
            PowerPriceResponse(elasticity=0.2, reference_spend=fitted * 52).resolve(
                **layout(), derived_reference=fitted
            )
        PowerPriceResponse(elasticity=0.2, reference_spend=fitted * 4).resolve(
            **layout(), derived_reference=fitted
        )
        with pytest.raises(ValueError, match="4x apart"):
            PowerPriceResponse(
                elasticity=0.2,
                reference_spend=fitted * 4,
                reference_spend_tolerance=2.0,
            ).resolve(**layout(), derived_reference=fitted)

    def test_derived_reference_missing_on_an_optimized_cell_is_rejected(self):
        """A cell with no on-air history has no price level to anchor to; outside the
        mask it does not matter and is filled with a finite sentinel."""
        with pytest.raises(ValueError, match=r"no on-air period.*\('radio',\)"):
            PowerPriceResponse(elasticity=0.2).resolve(
                **layout(), derived_reference=derived([100.0, np.nan, 100.0])
            )
        resolved = PowerPriceResponse(elasticity=0.2).resolve(
            **layout(mask_values=[True, False, True]),
            derived_reference=derived([100.0, np.nan, 100.0]),
        )
        np.testing.assert_array_equal(resolved.reference_spend, [100.0, 1.0, 100.0])

    def test_the_reference_is_only_required_where_the_map_bends(self):
        """A flat cell never reads its reference, so a dark channel at elasticity 0
        cannot block the curved siblings; a sentinel keeps its coefficients finite."""
        elasticity = xr.DataArray(
            [0.2, 0.0, 0.2],
            dims=("channel",),
            coords={"channel": ["tv", "radio", "digital"]},
        )
        resolved = PowerPriceResponse(elasticity=elasticity).resolve(
            **layout(), derived_reference=derived([100.0, np.nan, 100.0])
        )
        np.testing.assert_array_equal(resolved.reference_spend, [100.0, 1.0, 100.0])

    def test_a_supplied_reference_is_only_guarded_where_the_map_bends(self):
        """The tolerance guard reads the cells whose price the reference anchors; a
        flat cell's value is never read, so it cannot be 52x wrong."""
        elasticity = xr.DataArray(
            [0.2, 0.0, 0.2],
            dims=("channel",),
            coords={"channel": ["tv", "radio", "digital"]},
        )
        fitted = derived([100.0, 100.0, 100.0])
        supplied = derived([100.0, 5200.0, 100.0])
        PowerPriceResponse(elasticity=elasticity, reference_spend=supplied).resolve(
            **layout(), derived_reference=fitted
        )
        with pytest.raises(ValueError, match="52x apart"):
            PowerPriceResponse(elasticity=0.2, reference_spend=supplied).resolve(
                **layout(), derived_reference=fitted
            )

    def test_identity_needs_no_reference_of_any_kind(self):
        resolved = PowerPriceResponse(elasticity=0.0).resolve(
            **layout(), derived_reference=None
        )
        assert resolved.is_identity
        np.testing.assert_array_equal(resolved.reference_spend, [1.0, 1.0, 1.0])

    def test_non_identity_without_any_reference_is_rejected(self):
        with pytest.raises(ValueError, match="reference_spend is required"):
            PowerPriceResponse(elasticity=0.2).resolve(
                **layout(), derived_reference=None
            )

    def test_coordinate_less_dims_are_refused_not_consumed_positionally(self):
        """`reindex` has nothing to align an unlabelled dim by and stamps the model's labels on
        in arrival order -- the #3038 hazard, one level down. Both inputs are checked."""
        with pytest.raises(ValueError, match="carry no coordinates"):
            PowerPriceResponse(
                elasticity=xr.DataArray([0.1, 0.2, 0.3], dims=("channel",)),
                reference_spend=derived([1, 1, 1]),
            ).resolve(**layout(), derived_reference=None)
        with pytest.raises(ValueError, match="carry no coordinates"):
            PowerPriceResponse(
                elasticity=0.2,
                reference_spend=xr.DataArray([1.0, 1.0, 1.0], dims=("channel",)),
            ).resolve(**layout(), derived_reference=None)

    def test_elasticity_outside_the_mask_is_ignored(self):
        """A masked cell spends nothing, so its elasticity must neither warn (the wide-floor
        check would fire on the sentinel reference) nor stop the resolved map being the identity."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            resolved = PowerPriceResponse(elasticity={"radio": 0.9}).resolve(
                **layout(mask_values=[True, False, True]),
                derived_reference=derived([100.0, 100.0, 100.0]),
            )
        assert resolved.is_identity
        np.testing.assert_array_equal(resolved.gamma, [0.0, 0.0, 0.0])

    def test_tolerance_guard_compares_only_where_a_fitted_value_exists(self):
        """np.argmax lands on a NaN and `nan > tol` is False, so an unguarded comparison would
        pass silently. A cell with no fitted value (an attested channel the table does not
        price, a cell never on air) keeps the supplied value as given, and the others are
        still guarded."""
        fitted = derived([100.0, np.nan, 100.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            resolved = PowerPriceResponse(
                elasticity=0.2, reference_spend=derived([100.0, 5.0, 100.0])
            ).resolve(**layout(), derived_reference=fitted)
        np.testing.assert_array_equal(resolved.reference_spend, [100.0, 5.0, 100.0])
        with pytest.raises(ValueError, match="52x apart"):
            PowerPriceResponse(
                elasticity=0.2, reference_spend=derived([5200.0, 5.0, 100.0])
            ).resolve(**layout(), derived_reference=fitted)

    def test_a_supplied_reference_overrides_the_derived_one_cell_by_cell(self):
        """Cells reference_spend leaves out (absent labels or nan) keep the derived default,
        and the cells it gives are guarded against it. A curved cell with neither is refused
        with every possible cause, not only a flighting pattern; unknown labels are still
        refused rather than dropped."""
        fitted = derived([100.0, np.nan, 300.0])
        expected = [100.0, 40.0, 300.0]
        for partial in (
            xr.DataArray([40.0], dims=("channel",), coords={"channel": ["radio"]}),
            derived([np.nan, 40.0, np.nan]),
        ):
            resolved = PowerPriceResponse(
                elasticity=0.2, reference_spend=partial
            ).resolve(**layout(), derived_reference=fitted)
            np.testing.assert_array_equal(resolved.reference_spend, expected)
        with pytest.raises(ValueError, match="52x apart"):
            PowerPriceResponse(
                elasticity=0.2,
                reference_spend=xr.DataArray(
                    [5200.0, 40.0],
                    dims=("channel",),
                    coords={"channel": ["tv", "radio"]},
                ),
            ).resolve(**layout(), derived_reference=fitted)
        with pytest.raises(
            ValueError, match=r"neither a reference_spend value nor a derived one"
        ) as info:
            PowerPriceResponse(
                elasticity=0.2,
                reference_spend=xr.DataArray(
                    [100.0], dims=("channel",), coords={"channel": ["tv"]}
                ),
            ).resolve(**layout(), derived_reference=fitted)
        assert "('radio',)" in str(info.value) and "does not price" in str(info.value)
        with pytest.raises(ValueError, match="coordinates the model does not have"):
            PowerPriceResponse(
                elasticity=0.2,
                reference_spend=xr.DataArray(
                    [1.0], dims=("channel",), coords={"channel": ["print"]}
                ),
            ).resolve(**layout(), derived_reference=fitted)

    @pytest.mark.parametrize("bad", [-5.0, 0.0, np.inf])
    def test_a_given_reference_that_is_unusable_is_refused_not_replaced(self, bad):
        """Only a missing value falls back to the derived default; a value the user does give
        must be usable, or a typo would silently become the fitted mean."""
        with pytest.raises(ValueError, match=r"not positive and finite.*\('tv',\)"):
            PowerPriceResponse(
                elasticity=0.2, reference_spend=derived([bad, 40.0, np.nan])
            ).resolve(**layout(), derived_reference=derived([100.0, np.nan, 300.0]))

    def test_a_partial_reference_with_nothing_to_derive_names_the_gap(self):
        """A spend variable, or an attested model with no table, has no fitted reference:
        the cells a partial reference leaves out are refused, without blaming a table."""
        with pytest.raises(
            ValueError, match=r"no reference_spend value.*no fitted spend"
        ) as info:
            PowerPriceResponse(
                elasticity=0.2,
                reference_spend=xr.DataArray(
                    [100.0], dims=("channel",), coords={"channel": ["tv"]}
                ),
            ).resolve(**layout(), derived_reference=None)
        assert "('radio',)" in str(info.value) and "('tv',)" not in str(info.value)
        assert "cost_per_unit" not in str(info.value)

    def test_public_import(self):
        from pymc_marketing.mmm import PowerPriceResponse as exported
        from pymc_marketing.mmm import PriceResponse

        assert exported is PowerPriceResponse and issubclass(exported, PriceResponse)


class TestPriceResponseContract:
    """What the optimizer asks of any family, so a bracket schedule (#3067) can answer
    differently from the power law without touching the optimizer."""

    def test_power_specific_fields_live_on_power_only(self):
        assert "reference_spend" not in PriceResponse.model_fields
        assert "assume_delivery_units" not in PriceResponse.model_fields
        assert "reference_spend" in PowerPriceResponse.model_fields
        assert "assume_delivery_units" in PowerPriceResponse.model_fields

    @pytest.mark.parametrize(
        "elasticity, mask_values",
        [
            (0.3, None),
            (0.0, None),
            ({"radio": 0.3}, None),
            ({"radio": 0.3}, [True, False, True]),
            (
                xr.DataArray(
                    [0.2, 0.0, 0.4],
                    dims=("channel",),
                    coords={"channel": ["tv", "radio", "digital"]},
                ),
                [True, True, False],
            ),
        ],
    )
    def test_curved_cells_is_what_resolve_bends(self, elasticity, mask_values):
        """The gate reads curved_cells before resolving; the pinned-cell warning reads the
        resolved map's curved. They must agree, masked-out cells included: a response that
        bends only a masked-out cell is curved as a declaration and flat on that layout."""
        response = PowerPriceResponse(elasticity=elasticity)
        on_layout = layout(mask_values=mask_values)
        resolved = response.resolve(
            **on_layout, derived_reference=derived([100.0, 100.0, 100.0])
        )
        np.testing.assert_array_equal(
            response.curved_cells(**on_layout), resolved.curved
        )

    def test_attestation_and_reference_hooks_read_the_power_fields(self):
        bare = PowerPriceResponse(elasticity=0.3)
        assert bare.attests_delivery_units is False
        assert bare.needs_derived_reference is True
        attested = PowerPriceResponse(
            elasticity=0.3,
            assume_delivery_units=True,
            reference_spend=derived([1, 2, 3]),
        )
        assert attested.attests_delivery_units is True
        assert attested.needs_derived_reference is False

    def test_declarations_are_frozen(self):
        """A declaration is reusable across models and windows; mutating it after
        construction would bypass _check_domain."""
        response = PowerPriceResponse(elasticity=0.3)
        with pytest.raises(ValidationError):
            response.elasticity = 1.5

    def test_resolved_map_exposes_a_money_scale(self):
        resolved = PowerPriceResponse(
            elasticity=0.2, reference_spend=derived([10.0, 20.0, 30.0])
        ).resolve(**layout(), derived_reference=None)
        np.testing.assert_array_equal(resolved.money_scale, resolved.reference_spend)
        np.testing.assert_array_equal(resolved.curved, resolved.gamma > 0.0)

    def test_a_family_without_breakpoints_gets_neutral_solve_hooks(self, spend):
        """The optimizer conditions every map for a solve and asks it for linear pieces.
        A map with no breakpoints is affine on its whole box, so its piece is the box,
        conditioning changes nothing, and period weights do not reach its graph."""
        resolved = PowerPriceResponse(
            elasticity=0.2, reference_spend=derived([10.0, 20.0, 30.0])
        ).resolve(**layout(), derived_reference=None)
        before = compile_lowered(resolved.to_delivery(spend), spend)(
            np.array([5.0, 15.0, 40.0])
        )
        low, high = np.array([0.0, 1.0, 2.0]), np.array([10.0, np.inf, 30.0])
        piece_low, piece_high = resolved.linear_piece_bounds(
            np.array([3.0, 4.0, 5.0]), low, high
        )
        np.testing.assert_array_equal(piece_low, low)
        np.testing.assert_array_equal(piece_high, high)
        with resolved.conditioned_on(low, high):
            during = compile_lowered(resolved.to_delivery(spend), spend)(
                np.array([5.0, 15.0, 40.0])
            )
        np.testing.assert_array_equal(during, before)
        weights = as_xtensor(pt.constant(np.ones((4, 3))), dims=("date", "channel"))
        assert resolved.bind_period_weights(weights) is None
        after = compile_lowered(resolved.to_delivery(spend), spend)(
            np.array([5.0, 15.0, 40.0])
        )
        np.testing.assert_array_equal(after, before)


def tiered_closed_form(z, breakpoints, multipliers):
    """The issue's formula in budget units: sum_i (min(z, B_{i+1}) - min(z, B_i)) / r_i."""
    edges = np.concatenate([[0.0], np.asarray(breakpoints, float), [np.inf]])
    z = np.asarray(z, float)[..., None]
    return (
        (np.minimum(z, edges[1:]) - np.minimum(z, edges[:-1]))
        / np.asarray(multipliers, float)
    ).sum(-1)


def cells(n):
    """A one-dim layout of ``n`` cells, so one compiled call evaluates a whole grid."""
    return layout(dims=("cell",), coords={"cell": list(range(n))})


def tiered_maps(declaration, n, *, base_price=None, mode=None):
    """Resolve on ``n`` cells and compile u, p, m the way the optimizer lowers them."""
    resolved = declaration.resolve(**cells(n), derived_reference=None)
    spend = ptx.xtensor("spend", shape=(n,), dims=("cell",))
    p0 = (
        None
        if base_price is None
        else as_xtensor(pt.constant(np.full(n, base_price)), dims=("cell",))
    )

    def compiled(output):
        return function([spend], rewrite_graph(output.values, include=LOWER), mode=mode)

    return (
        resolved,
        spend,
        compiled(resolved.to_delivery(spend, p0)),
        compiled(resolved.implied_price(spend, p0)),
        compiled(resolved.implied_marginal_price(spend, p0)),
    )


SCHEDULES = [
    pytest.param([100.0, 400.0], [1.1, 1.6, 2.5], id="three-brackets"),
    pytest.param([1e6], [1.0, 0.8], id="falling"),
    pytest.param([100.0, 400.0], [1.0, 0.7, 1.4], id="mixed"),
]


def bracket_grid(breakpoints):
    """Points well inside every bracket, and the bracket index of each."""
    edges = np.concatenate([[0.0], breakpoints, [3.0 * breakpoints[-1]]])
    fractions = np.array([0.01, 0.3, 0.5, 0.7, 0.99])
    z = (edges[:-1, None] + fractions * np.diff(edges)[:, None]).ravel()
    return z, np.repeat(np.arange(len(edges) - 1), fractions.size)


class TestResolvedTieredPriceResponse:
    @pytest.mark.parametrize("breakpoints, multipliers", SCHEDULES)
    def test_closed_form_on_every_bracket(self, breakpoints, multipliers):
        """Delivery is the issue's formula inside every bracket and with no jump at a
        breakpoint approached from either side, and the marginal price is the bracket's
        rate, for rising, falling and mixed multipliers."""
        b = np.asarray(breakpoints)
        inside, bracket = bracket_grid(b)
        z = np.concatenate([inside, b * (1.0 - 1e-12), b, b * (1.0 + 1e-12)])
        _, _, u, _, m = tiered_maps(
            TieredPriceResponse(breakpoints=breakpoints, rate_multipliers=multipliers),
            z.size,
            base_price=5.0,
        )
        delivery = u(z)
        expected = tiered_closed_form(z, breakpoints, multipliers) / 5.0
        np.testing.assert_allclose(delivery, expected, rtol=1e-10)
        np.testing.assert_allclose(
            m(z)[: inside.size], 5.0 * np.asarray(multipliers)[bracket], rtol=1e-12
        )
        assert np.all(np.diff(delivery[: inside.size]) > 0.0)

    def test_the_issue_table(self):
        """Reproduces the two-bracket truth of #3067 that PowerPriceResponse misses by 16% to
        51% in marginal price: a baseline rate of 5, an incremental rate of 8 above 1e6. The
        committed tranche keeps its rate, so at +100% the average price is 6.15, not 6.48."""
        spends = 1e6 * np.array([1.0, 1.1, 1.25, 1.5, 2.0, 3.0])
        _, _, u, p, m = tiered_maps(
            TieredPriceResponse(breakpoints=[1e6], rate_multipliers=[1.0, 1.6]),
            spends.size,
            base_price=5.0,
        )
        np.testing.assert_allclose(
            u(spends),
            [200_000, 212_500, 231_250, 262_500, 325_000, 450_000],
            rtol=1e-10,
        )
        np.testing.assert_allclose(m(spends), 8.0, rtol=1e-12)
        np.testing.assert_allclose(p(spends)[4], 2e6 / 325_000, rtol=1e-10)

    def test_zero_and_negative_money(self):
        _, _, u, p, m = tiered_maps(
            TieredPriceResponse(breakpoints=[100.0], rate_multipliers=[1.2, 1.6]),
            2,
            base_price=5.0,
        )
        s = np.array([0.0, -5.0])
        assert np.all(u(s) == 0.0)
        assert np.all(np.isnan(p(s)))
        np.testing.assert_allclose(m(s), 5.0 * 1.2, rtol=1e-12)

    @pytest.mark.parametrize("mode", [None, "JAX"], ids=["default", "jax"])
    def test_marginal_price_is_right_continuous_at_the_breakpoint(self, mode):
        """A plan on its breakpoint must report the next bracket's rate as the price of its
        next unit, and the solver's gradient there must be the slope above."""
        if mode == "JAX":
            pytest.importorskip("jax")
        resolved, spend, _, _, m = tiered_maps(
            TieredPriceResponse(breakpoints=[1e6], rate_multipliers=[1.0, 1.6]),
            1,
            base_price=5.0,
            mode=mode,
        )
        at = np.array([1e6])
        assert m(at)[0] == pytest.approx(8.0, rel=1e-12)
        p0 = as_xtensor(pt.constant(np.array([5.0])), dims=("cell",))
        objective = rewrite_graph(
            resolved.to_delivery(spend, p0).sum().values, include=LOWER
        )
        gradient = function([spend], pt.grad(objective, spend), mode=mode)
        assert gradient(at)[0] * 5.0 == pytest.approx(1.0 / 1.6, rel=1e-12)

    def test_unscheduled_and_unit_cells_are_bitwise_the_constant_price_graph(
        self, spend, base_price
    ):
        """A cell the schedule leaves out, and one at [1.0], must cost exactly what the
        constant-price path costs, so adding a schedule elsewhere moves nothing there."""
        resolved = TieredPriceResponse(
            breakpoints={"tv": [100.0], "digital": []},
            rate_multipliers={"tv": [1.0, 1.6], "digital": [1.0]},
        ).resolve(**layout(), derived_reference=None)
        with_base = compile_lowered(resolved.to_delivery(spend, base_price), spend)
        without = compile_lowered(resolved.to_delivery(spend), spend)
        constant = compile_lowered(spend / base_price, spend)
        draws = np.random.default_rng(3067).uniform(0.0, 500.0, size=(2000, 3))
        for s in draws:
            assert np.array_equal(with_base(s)[1:], constant(s)[1:])
            assert np.array_equal(without(s)[1:], s[1:])
        assert (
            TieredPriceResponse(rate_multipliers=[1.0])
            .resolve(**layout(), derived_reference=None)
            .is_identity
        )

    def test_padded_zero_first_and_equal_adjacent_cells_resolve_cleanly(self):
        """A committed plan gives a channel with no commitment a zero first breakpoint, and
        equal multipliers are a breakpoint with no kink: neither may warn or carry a knot."""
        names = {"channel": ["tv", "radio", "digital"]}
        breakpoints = xr.DataArray(
            [[100.0, np.nan], [0.0, 100.0], [100.0, 200.0]],
            dims=("channel", "bracket"),
            coords=names,
        )
        multipliers = xr.DataArray(
            [[1.0, 1.6, np.nan], [1.0, 1.5, 2.0], [1.0, 1.0, 2.0]],
            dims=("channel", "bracket"),
            coords=names,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            resolved = TieredPriceResponse(
                breakpoints=breakpoints, rate_multipliers=multipliers
            ).resolve(**layout(), derived_reference=None)
        np.testing.assert_array_equal(
            np.isnan(resolved.edges), [[False, True, True], [True, False, False]]
        )
        np.testing.assert_allclose(resolved.first_slope, [1.0, 1 / 1.5, 1.0])
        assert resolved.edges[1, 1] == 100.0 and resolved.edges[1, 2] == 200.0
        spend = ptx.xtensor("s", shape=(3,), dims=DIMS)
        m = compile_lowered(resolved.implied_marginal_price(spend), spend)
        np.testing.assert_allclose(m(np.full(3, 150.0)), [1.6, 2.0, 1.0])

    def test_the_blend(self):
        """Inside a solve's box the knot is a C2 quartic: exact outside the window, short by
        3/16 of drop * w at the knot, concave and increasing; after the solve the knot is
        hard again. C2 means the curvature is zero at both edges of the window (grid points
        100 and 300), where a C1 blend would jump by about half the peak curvature. A JAX
        function compiled before the solve, as the optimizer's is, reads the blend inside it,
        with a finite gradient, and the hard knot after it."""
        b, w, p0 = 1e6, 1e3, 5.0
        drop = 1.0 - 1.0 / 1.6
        grid = np.linspace(b - 2 * w, b + 2 * w, 401)
        resolved, spend, u, _, _ = tiered_maps(
            TieredPriceResponse(
                breakpoints=[b], rate_multipliers=[1.0, 1.6], knot_width=w / b
            ),
            grid.size,
            base_price=p0,
        )
        exact = tiered_closed_form(grid, [b], [1.0, 1.6]) / p0
        low, high = np.zeros(grid.size), np.full(grid.size, 3 * b)
        on_jax = importlib.util.find_spec("jax") is not None
        if on_jax:
            objective = rewrite_graph(
                resolved.to_delivery(spend).sum().values, include=LOWER
            )
            jax_u = function([spend], objective, mode="JAX")
            jax_grad = function([spend], pt.grad(objective, spend), mode="JAX")
        with resolved.conditioned_on(low, high):
            blended = u(grid)
            edges = u(np.array([b - w, b + w] + [b] * (grid.size - 2)))
            if on_jax:
                np.testing.assert_allclose(jax_u(grid) / p0, blended.sum(), rtol=1e-12)
                assert np.all(np.isfinite(jax_grad(grid)))
        if on_jax:
            np.testing.assert_allclose(jax_u(grid) / p0, exact.sum(), rtol=1e-12)
        outside = np.abs(grid - b) >= w
        np.testing.assert_allclose(blended[outside], exact[outside], rtol=1e-12)
        np.testing.assert_allclose(
            edges[:2],
            tiered_closed_form([b - w, b + w], [b], [1.0, 1.6]) / p0,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            exact[200] - blended[200], 3 / 16 * drop * w / p0, rtol=1e-6
        )
        first, second = np.diff(blended), np.diff(blended, 2)
        assert np.all(first > 0.0)
        assert np.all(second <= 1e-6 * np.abs(second).max())
        peak = np.abs(second).max()
        assert abs(second[99]) < 1e-2 * peak
        assert abs(second[299]) < 1e-2 * peak
        np.testing.assert_allclose(u(grid), exact, rtol=1e-12)


class TestTieredPriceResponseDeclaration:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            pytest.param(
                dict(breakpoints=[100.0], rate_multipliers=[1.0, -1.6]),
                "must be positive and finite",
                id="negative-multiplier",
            ),
            pytest.param(
                dict(breakpoints=[100.0], rate_multipliers=[1.0, np.nan]),
                "must be positive and finite",
                id="nan-multiplier",
            ),
            pytest.param(
                dict(breakpoints=[-1.0], rate_multipliers=[1.0, 1.6]),
                "non-negative, finite and strictly ascending",
                id="negative-breakpoint",
            ),
            pytest.param(
                dict(breakpoints=[200.0, 100.0], rate_multipliers=[1.0, 1.2, 1.6]),
                "non-negative, finite and strictly ascending",
                id="descending",
            ),
            pytest.param(
                dict(
                    breakpoints=xr.DataArray([np.nan, 100.0], dims="bracket"),
                    rate_multipliers=xr.DataArray([1.0, 1.2, 1.6], dims="bracket"),
                ),
                "nan may only trail",
                id="leading-nan",
            ),
            pytest.param(
                dict(breakpoints=[100.0], rate_multipliers=[1.0, 1.2, 1.6]),
                "one multiplier per bracket",
                id="count",
            ),
            pytest.param(
                dict(breakpoints=(), rate_multipliers=()),
                "one multiplier per bracket",
                id="empty",
            ),
            pytest.param(
                dict(breakpoints=[100.0], rate_multipliers={"tv": [1.0, 1.6]}),
                "describe one schedule",
                id="forms",
            ),
            pytest.param(
                dict(
                    breakpoints={"tv": [100.0]},
                    rate_multipliers={"radio": [1.0, 1.6]},
                ),
                "describe one schedule",
                id="labels",
            ),
            pytest.param(
                dict(rate_multipliers={1: [1.0], "1": [1.1]}),
                "name the same label more than once",
                id="duplicate-keys",
            ),
            pytest.param(
                dict(rate_multipliers=[1.0], reference_rate=-5.0),
                "greater than 0",
                id="reference-negative",
            ),
            pytest.param(
                dict(rate_multipliers=[1.0], reference_rate=float("inf")),
                "finite number",
                id="reference-infinite",
            ),
            pytest.param(
                dict(
                    breakpoints=xr.DataArray(
                        [100.0, 200.0, 300.0], dims="channel", coords=DIMS_COORDS
                    ),
                    rate_multipliers=xr.DataArray(
                        [[1.0, 1.6, 2.0]] * 3,
                        dims=("channel", "bracket"),
                        coords=DIMS_COORDS,
                    ),
                ),
                "one multiplier per bracket",
                id="array-count",
            ),
            pytest.param(
                dict(
                    breakpoints=(),
                    rate_multipliers=xr.DataArray([1.0, 1.6], dims="bracket"),
                ),
                "one multiplier per bracket",
                id="array-count-no-breakpoints",
            ),
            pytest.param(
                dict(
                    breakpoints=xr.DataArray(
                        [10.0, 20.0],
                        dims="channel",
                        coords={"channel": ["tv", "radio"]},
                    ),
                    rate_multipliers=xr.DataArray(
                        [[1.0, 1.6]],
                        dims=("channel", "bracket"),
                        coords={"channel": ["tv"]},
                    ),
                ),
                r"breakpoints names \['radio'\] along 'channel'.*describe one schedule",
                id="array-labels",
            ),
        ],
    )
    def test_refusals_when_declared(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            TieredPriceResponse(**kwargs)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            pytest.param(
                dict(rate_multipliers={"podcast": [1.0]}),
                "must all be labels of exactly one budget dim",
                id="unknown-dict-label",
            ),
            pytest.param(
                dict(
                    rate_multipliers=xr.DataArray(
                        [1.0], dims="channel", coords={"channel": ["podcast"]}
                    )
                ),
                "coordinates the model does not have",
                id="unknown-array-label",
            ),
            pytest.param(
                dict(rate_multipliers=xr.DataArray([1.0, 1.0, 1.0], dims="channel")),
                "carry no coordinates",
                id="unlabelled",
            ),
            pytest.param(
                dict(
                    rate_multipliers=xr.DataArray(
                        [1.0, 1.1], dims="date", coords={"date": [0, 1]}
                    )
                ),
                "the schedule has no date axis",
                id="date-dim",
            ),
        ],
    )
    def test_refusals_at_resolve(self, kwargs, match):
        declaration = TieredPriceResponse(**kwargs)
        with pytest.raises(ValueError, match=match):
            declaration.resolve(**layout(), derived_reference=None)

    @pytest.mark.parametrize(
        "breakpoints, multipliers, tolerance, refused",
        [
            pytest.param([1e6], [1.0, 160.0], 3.0, True, id="percentage"),
            pytest.param([1e6], [5.5, 8.0], 3.0, True, id="both-absolute"),
            pytest.param([1e6], [1.0, 2.5], 3.0, False, id="absolute-2.5"),
            pytest.param([1e6], [1.0, 4.0], 5.0, False, id="four-times-raised"),
        ],
    )
    def test_rate_tolerance_table(self, breakpoints, multipliers, tolerance, refused):
        """Every multiplier is checked, so a percentage or an absolute rate is caught in
        whichever bracket it is; one within the tolerance passes, which is the check's limit."""
        kwargs = dict(
            breakpoints=breakpoints,
            rate_multipliers=multipliers,
            rate_tolerance=tolerance,
        )
        if refused:
            with pytest.raises(ValueError, match="too far from 1 to be a ratio"):
                TieredPriceResponse(**kwargs)
        else:
            TieredPriceResponse(**kwargs)

    def test_tolerance_refusal_shows_the_division(self):
        with pytest.raises(ValueError, match=r"8 / 5 = 1\.6"):
            TieredPriceResponse(
                breakpoints=[1e6], rate_multipliers=[1.0, 8.0], reference_rate=5.0
            )
        multipliers = xr.DataArray(
            [[1.0, 1.1], [1.0, 8.0]],
            dims=("channel", "bracket"),
            coords={"channel": ["tv", "radio"]},
        )
        with pytest.raises(ValueError, match=r"8 / 5 = 1\.6"):
            TieredPriceResponse(
                breakpoints=xr.DataArray([1e6], dims="bracket"),
                rate_multipliers=multipliers,
                reference_rate=5.0,
            )

    def test_many_brackets_warn(self):
        with pytest.warns(
            UserWarning, match="looks like a discretized price curve"
        ) as record:
            TieredPriceResponse(
                breakpoints=[1.0, 2.0, 3.0, 4.0],
                rate_multipliers=[1.0, 1.1, 1.2, 1.3, 1.4],
            )
        assert record[0].filename == __file__
        with pytest.warns(
            UserWarning, match="looks like a discretized price curve"
        ) as record:
            TieredPriceResponse.model_validate(
                {
                    "breakpoints": [1.0, 2.0, 3.0, 4.0],
                    "rate_multipliers": [1.0, 1.1, 1.2, 1.3, 1.4],
                }
            )
        assert record[0].filename == __file__
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TieredPriceResponse(
                breakpoints=[1.0, 2.0, 3.0], rate_multipliers=[1.0, 1.1, 1.2, 1.3]
            )

    def test_the_three_forms_resolve_to_the_same_arrays(self):
        """A sequence, a dict naming every channel, and a DataArray typed with the literal
        bracket dim name are one schedule; a reference_rate is a record and changes none
        of it."""
        sequence = TieredPriceResponse(
            breakpoints=[100.0], rate_multipliers=[1.0, 1.6], reference_rate=5.0
        )
        by_label = TieredPriceResponse(
            breakpoints={c: [100.0] for c in DIMS_COORDS["channel"]},
            rate_multipliers={c: [1.0, 1.6] for c in DIMS_COORDS["channel"]},
        )
        array = TieredPriceResponse(
            breakpoints=xr.DataArray([100.0], dims="bracket"),
            rate_multipliers=xr.DataArray([1.0, 1.6], dims="bracket"),
        )
        resolved = [
            d.resolve(**layout(), derived_reference=None)
            for d in (sequence, by_label, array)
        ]
        for other in resolved[1:]:
            np.testing.assert_array_equal(other.edges, resolved[0].edges)
            np.testing.assert_array_equal(other.slope_drops, resolved[0].slope_drops)
            np.testing.assert_array_equal(other.first_slope, resolved[0].first_slope)

    def test_a_bare_number_is_a_one_element_schedule(self):
        resolved = TieredPriceResponse(rate_multipliers=1.1).resolve(
            **layout(), derived_reference=None
        )
        np.testing.assert_allclose(resolved.first_slope, 1 / 1.1)
        assert resolved.edges.shape[0] == 0
        by_label = TieredPriceResponse(
            breakpoints={"tv": 100.0}, rate_multipliers={"tv": [1.0, 1.6]}
        )
        np.testing.assert_array_equal(
            by_label.resolve(**layout(), derived_reference=None).edges[0],
            [100.0, np.nan, np.nan],
        )

    def test_breakpoints_without_a_bracket_dim_is_one_per_cell(self):
        """A committed plan passes as it is."""
        committed = xr.DataArray([40.0, 25.0, 0.0], dims="channel", coords=DIMS_COORDS)
        resolved = TieredPriceResponse(
            breakpoints=committed,
            rate_multipliers=xr.DataArray([1.0, 1.6], dims="bracket"),
        ).resolve(**layout(), derived_reference=None)
        np.testing.assert_array_equal(resolved.edges[0, :2], [40.0, 25.0])
        assert np.isnan(resolved.edges[0, 2])
        np.testing.assert_allclose(resolved.first_slope, [1.0, 1.0, 1 / 1.6])

    def test_a_schedule_mutated_in_place_is_refused_at_resolve(self):
        """The declaration holds the caller's dict or DataArray, so resolve checks it again,
        with the same messages."""
        by_label = TieredPriceResponse(
            breakpoints={"tv": [100.0]},
            rate_multipliers={"tv": [1.0, 1.6]},
            reference_rate=5.0,
        )
        by_label.rate_multipliers["tv"] = (1.0, 8.0)
        with pytest.raises(ValueError, match=r"8 / 5 = 1\.6"):
            by_label.resolve(**layout(), derived_reference=None)
        multipliers = xr.DataArray([1.0, 1.6], dims="bracket")
        array = TieredPriceResponse(
            breakpoints=xr.DataArray([100.0], dims="bracket"),
            rate_multipliers=multipliers,
            reference_rate=5.0,
        )
        multipliers.values[1] = -1.0
        with pytest.raises(ValueError, match="must be positive and finite"):
            array.resolve(**layout(), derived_reference=None)

    def test_a_budget_dim_named_bracket_is_refused(self):
        coords = {"bracket": ["a", "b"]}
        with pytest.raises(ValueError, match="taken by the schedule's own dim"):
            TieredPriceResponse(rate_multipliers=[1.1]).resolve(
                **layout(dims=("bracket",), coords=coords), derived_reference=None
            )

    def test_integer_labels_are_matched_by_string_form(self):
        coords = {"channel": [1, 2]}
        declaration = TieredPriceResponse(
            breakpoints={1: [100.0]}, rate_multipliers={1: [1.0, 1.6]}
        )
        resolved = declaration.resolve(**layout(coords=coords), derived_reference=None)
        np.testing.assert_array_equal(resolved.edges[0], [100.0, np.nan])


def two_bracket(b=1e6, multipliers=(1.0, 1.6), **kwargs):
    return TieredPriceResponse(
        breakpoints=[b], rate_multipliers=list(multipliers), **kwargs
    ).resolve(**layout(dims=("cell",), coords={"cell": [0]}), derived_reference=None)


class TestTieredSolveHooks:
    def test_conditioned_on_moves_blends_and_restores(self):
        """A knot strictly inside the box is blended; one on or beyond a bound is moved off
        the box, or kept where it already is; on exit, and on an exception, the declared
        map comes back."""
        b = 1e6
        resolved = two_bracket(b)
        snap, width = resolved.snap, resolved.knot_width
        default = b * (1.0 - snap)
        ulp = np.spacing(b)
        cases = [
            ((0.0, 3 * b), default, width * b),
            ((b, 3 * b), default, 0.0),
            ((0.0, b), b * (1.0 + snap), 0.0),
            ((b + ulp, 3 * b), default, 0.0),
            ((b - ulp, 3 * b), (b - ulp) * (1.0 - snap), 0.0),
            ((b * (1 - 1e-9), 3 * b), default, width * b),
            ((b, b), default, 0.0),
        ]
        for (low, high), knot, half_width in cases:
            with resolved.conditioned_on(np.array([low]), np.array([high])):
                assert resolved._knots.get_value()[0, 0] == pytest.approx(
                    knot, rel=1e-15
                ), (low, high)
                assert resolved._half_widths.get_value()[0, 0] == half_width
            assert resolved._knots.get_value()[0, 0] == default
            assert resolved._half_widths.get_value()[0, 0] == 0.0
        with pytest.raises(RuntimeError, match="inside"):
            with resolved.conditioned_on(np.array([0.0]), np.array([3 * b])):
                raise RuntimeError("inside")
        assert resolved._knots.get_value()[0, 0] == default
        assert resolved._half_widths.get_value()[0, 0] == 0.0

    def test_a_nested_conditioning_restores_the_outer_block(self):
        """Leaving an inner conditioned_on gives back the outer block's knots, not the
        declared map; leaving the outer one gives back the declared map."""
        b = 1e6
        resolved = two_bracket(b)
        default = resolved._knots.get_value().copy()

        def state():
            return (
                resolved._knots.get_value().copy(),
                resolved._half_widths.get_value().copy(),
            )

        with resolved.conditioned_on(np.array([0.0]), np.array([3 * b])):
            outer = state()
            with resolved.conditioned_on(np.array([0.0]), np.array([b])):
                inner = state()
            assert not np.array_equal(inner[0], outer[0])
            assert not np.array_equal(inner[1], outer[1])
            after_inner = state()
            np.testing.assert_array_equal(after_inner[0], outer[0])
            np.testing.assert_array_equal(after_inner[1], outer[1])
        np.testing.assert_array_equal(resolved._knots.get_value(), default)
        np.testing.assert_array_equal(resolved._half_widths.get_value(), 0.0)

    def test_a_tie_with_a_bound_is_within_the_snap(self):
        """The table's "at a bound" is within the snap: a bound one ulp either side of the
        breakpoint is a tie, so the box sees one bracket and nothing is blended."""
        b = 1e6
        resolved = two_bracket(b)
        for bound in (b - np.spacing(b), b + np.spacing(b)):
            with resolved.conditioned_on(np.array([0.0]), np.array([bound])):
                assert resolved._half_widths.get_value()[0, 0] == 0.0
                assert resolved._knots.get_value()[0, 0] > bound

    def test_linear_piece_bounds(self):
        resolved = TieredPriceResponse(
            breakpoints=[100.0, 400.0], rate_multipliers=[1.0, 1.2, 1.6]
        ).resolve(
            **layout(dims=("cell",), coords={"cell": [0]}), derived_reference=None
        )

        def piece(budget, low=0.0, high=1000.0, from_below=None):
            lo, hi = resolved.linear_piece_bounds(
                np.array([budget]),
                np.array([low]),
                np.array([high]),
                None if from_below is None else np.array([from_below]),
            )
            return float(lo[0]), float(hi[0])

        assert piece(50.0) == (0.0, 100.0)
        assert piece(250.0) == (100.0, 400.0)
        assert piece(700.0) == (400.0, 1000.0)
        assert piece(250.0, 150.0, 300.0) == (150.0, 300.0)
        for budget in (100.0, np.nextafter(100.0, 0.0), np.nextafter(100.0, 200.0)):
            assert piece(budget) == (100.0, 400.0)
            assert piece(budget, from_below=True) == (0.0, 100.0)
        assert piece(100.0, low=100.0 * (1 + 1e-15)) == (100.0 * (1 + 1e-15), 400.0)

    def test_a_falling_breakpoint_inside_the_bounds_is_refused(self):
        """Delivery is convex across a breakpoint where the rate falls: a solve that can
        cross it is refused, and one whose box ends on it is not."""
        b = 1e6
        resolved = TieredPriceResponse(
            breakpoints=[b], rate_multipliers=[1.0, 0.8]
        ).resolve(**layout(mask_values=[True, False, False]), derived_reference=None)
        boxes = {
            "inside": ((0.0, 3 * b), True),
            "just inside": ((b * (1 - 1e-9), 3 * b), True),
            "lower bound on it": ((b, 3 * b), False),
            "upper bound on it": ((0.0, b), False),
            "lower bound one ulp off": ((b - np.spacing(b), 3 * b), False),
        }
        for name, ((low, high), refused) in boxes.items():
            lows, highs = (
                np.array([low, np.nan, np.nan]),
                np.array([high, np.nan, np.nan]),
            )
            if refused:
                with pytest.raises(
                    ValueError, match="the rate falls across a breakpoint"
                ) as info:
                    with resolved.conditioned_on(lows, highs):
                        pass
                assert "['tv']" in str(info.value), name
            else:
                with resolved.conditioned_on(lows, highs):
                    assert not resolved._half_widths.get_value().any(), name
        rising_then_falling = TieredPriceResponse(
            breakpoints=[100.0, 400.0], rate_multipliers=[1.0, 1.5, 1.2]
        ).resolve(
            **layout(dims=("cell",), coords={"cell": [0]}), derived_reference=None
        )
        with rising_then_falling.conditioned_on(np.array([0.0]), np.array([400.0])):
            widths = rising_then_falling._half_widths.get_value()[:, 0]
        assert widths[0] > 0.0 and widths[1] == 0.0
