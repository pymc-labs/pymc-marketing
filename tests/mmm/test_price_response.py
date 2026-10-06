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
)

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
