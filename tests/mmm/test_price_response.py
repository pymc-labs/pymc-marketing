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
from pytensor import function
from pytensor.graph import rewrite_graph
from pytensor.xtensor import as_xtensor

from pymc_marketing.mmm.price_response import (
    PowerPriceResponse,
    ResolvedPowerPriceResponse,
)

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
            (0.0, True),
            ({}, True),
            ({"tv": 0.0}, True),
            ({"tv": 0.1}, False),
            (
                xr.DataArray(
                    [0.0, 0.0], dims=("channel",), coords={"channel": ["tv", "radio"]}
                ),
                True,
            ),
            (0.2, False),
        ],
    )
    def test_is_identity_only_when_every_elasticity_is_zero(self, elasticity, expected):
        assert PowerPriceResponse(elasticity=elasticity).is_identity is expected

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
        # A label that exists in two dims is genuinely ambiguous.
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
        # The gate asks is_identity_on before resolve, so a malformed elasticity
        # must report the variable-qualified label from there as well. (An
        # all-zero declaration short-circuits as the identity before its dims
        # are read; resolve still raises for it, with the same label.)
        with pytest.raises(
            ValueError, match="channel_data: price_response: elasticity varies"
        ):
            PowerPriceResponse(elasticity=e + 0.2).is_identity_on(
                dims=("channel",),
                coords={"channel": ["tv", "radio", "digital"]},
                mask=layout()["mask"],
                date_dim="date",
                label="channel_data: price_response",
            )

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
        by name on every optimized cell, independently of the tolerance. Setting
        reference_spend_tolerance explicitly asserts the scale is intended and skips it."""
        fitted = derived([100.0, 200.0, 50.0])
        with pytest.raises(ValueError, match=r"num_periods \(4\).*window total"):
            PowerPriceResponse(elasticity=0.2, reference_spend=fitted * 4).resolve(
                **layout(), derived_reference=fitted, num_periods=4
            )
        # Within a few percent still reads as the same mistake.
        with pytest.raises(ValueError, match="window total"):
            PowerPriceResponse(elasticity=0.2, reference_spend=fitted * 4.1).resolve(
                **layout(), derived_reference=fitted, num_periods=4
            )
        # Not on every cell: not that mistake, and 4x is inside the generic tolerance.
        mixed = derived([400.0, 200.0, 200.0])
        PowerPriceResponse(elasticity=0.2, reference_spend=mixed).resolve(
            **layout(), derived_reference=fitted, num_periods=4
        )
        # An explicit tolerance is the assertion that the scale is meant.
        PowerPriceResponse(
            elasticity=0.2, reference_spend=fitted * 4, reference_spend_tolerance=10.0
        ).resolve(**layout(), derived_reference=fitted, num_periods=4)
        # Without num_periods there is no hypothesis to test.
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
        # 4x is inside the default 10x tolerance and outside a 2x one.
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

    def test_tolerance_guard_skips_cells_with_no_derived_value_and_says_so(self):
        """np.argmax lands on a NaN and `nan > tol` is False, so an unguarded comparison would
        pass silently. The guard compares where the fitted spend exists and reports the rest."""
        fitted = derived([100.0, np.nan, 100.0])
        with pytest.warns(UserWarning, match=r"could not be checked.*\('radio',\)"):
            PowerPriceResponse(
                elasticity=0.2, reference_spend=derived([100.0, 5.0, 100.0])
            ).resolve(**layout(), derived_reference=fitted)
        # The guard still fires on the comparable cells; the uncheckable one is reported, not hidden.
        with pytest.warns(UserWarning, match="could not be checked"):
            with pytest.raises(ValueError, match="52x apart"):
                PowerPriceResponse(
                    elasticity=0.2, reference_spend=derived([5200.0, 5.0, 100.0])
                ).resolve(**layout(), derived_reference=fitted)

    def test_public_import(self):
        from pymc_marketing.mmm import PowerPriceResponse as exported
        from pymc_marketing.mmm import PriceResponse

        assert exported is PowerPriceResponse and issubclass(exported, PriceResponse)
