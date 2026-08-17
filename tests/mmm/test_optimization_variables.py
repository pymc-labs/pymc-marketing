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
import numpy as np
import pymc as pm
import pytest
import xarray as xr

from pymc_marketing.mmm.optimization_variables import (
    FLAT_DIM,
    LeverVariable,
    MediaVariable,
    OptimizationVariable,
    OptimizationVariables,
)


def make_media_variable(
    rng: np.random.Generator,
    dims: tuple[str, ...] = ("channel",),
    sizes: tuple[int, ...] = (3,),
    partial_mask: bool = False,
    name: str = "channel_data",
) -> MediaVariable:
    coords = {
        dim: [f"{dim}_{i}" for i in range(size)]
        for dim, size in zip(dims, sizes, strict=True)
    }
    mask_values = np.ones(sizes, dtype=bool)
    if partial_mask:
        flat = mask_values.reshape(-1)
        off = rng.choice(flat.size, size=flat.size // 2, replace=False)
        flat[off] = False
        mask_values = flat.reshape(sizes)
    mask = xr.DataArray(mask_values, dims=dims, coords=coords)
    return MediaVariable(
        name=name,
        mask=mask,
        num_periods=4,
        adstock_periods=2,
        channel_scales=1.0,
        dtype="float64",
    )


@pytest.mark.parametrize(
    "dims, sizes, partial_mask",
    [
        (("channel",), (3,), False),
        (("channel",), (5,), True),
        (("channel", "geo"), (3, 2), False),
        (("channel", "geo"), (4, 3), True),
        (("geo", "channel"), (2, 4), True),
    ],
    ids=["1d_full", "1d_masked", "2d_full", "2d_masked", "2d_masked_geo_first"],
)
def test_pack_unpack_round_trip(dims, sizes, partial_mask):
    """pack(unpack(x)) == x and unpack(pack(da)) == da on optimized cells."""
    rng = np.random.default_rng(42)
    variable = make_media_variable(
        rng, dims=dims, sizes=sizes, partial_mask=partial_mask
    )
    opt_vars = OptimizationVariables([variable])

    # flat -> labelled -> flat
    x = rng.uniform(0.0, 100.0, size=opt_vars.size)
    assert np.array_equal(opt_vars.pack(opt_vars.unpack(x)), x)

    # labelled -> flat -> labelled: non-optimized cells come back as zero
    da = opt_vars.unpack(rng.uniform(0.0, 100.0, size=opt_vars.size))["channel_data"]
    round_tripped = opt_vars.unpack(opt_vars.pack({"channel_data": da}))["channel_data"]
    xr.testing.assert_equal(round_tripped, da)


def test_slices_tile_the_flat_vector():
    """Variable slices are contiguous, non-overlapping, and cover [0, size)."""
    rng = np.random.default_rng(0)
    variable = make_media_variable(rng, dims=("channel", "geo"), sizes=(3, 2))
    opt_vars = OptimizationVariables([variable])

    covered = np.zeros(opt_vars.size, dtype=int)
    for flat_slice in opt_vars.slices.values():
        covered[flat_slice] += 1
    assert np.array_equal(covered, np.ones(opt_vars.size, dtype=int))


def test_pack_bare_dataarray_single_variable():
    rng = np.random.default_rng(1)
    variable = make_media_variable(rng)
    opt_vars = OptimizationVariables([variable])

    da = opt_vars.unpack(rng.uniform(size=opt_vars.size))["channel_data"]
    assert np.array_equal(opt_vars.pack(da), opt_vars.pack({"channel_data": da}))


def test_pack_transposed_and_reordered_dataarray():
    """pack() is order-invariant: transposed dims and shuffled coords work."""
    rng = np.random.default_rng(2)
    variable = make_media_variable(rng, dims=("channel", "geo"), sizes=(3, 2))
    opt_vars = OptimizationVariables([variable])

    x = rng.uniform(size=opt_vars.size)
    da = opt_vars.unpack(x)["channel_data"]
    scrambled = da.transpose("geo", "channel").sel(
        channel=list(reversed(da.coords["channel"].values))
    )
    assert np.array_equal(opt_vars.pack(scrambled), x)


def test_pack_missing_dim_raises():
    rng = np.random.default_rng(3)
    variable = make_media_variable(rng, dims=("channel", "geo"), sizes=(3, 2))
    opt_vars = OptimizationVariables([variable])

    da = xr.DataArray(
        np.ones(3), dims=("channel",), coords={"channel": variable.coords["channel"]}
    )
    with pytest.raises(ValueError, match="missing required dims"):
        opt_vars.pack(da)


def test_pack_missing_coords_raises():
    rng = np.random.default_rng(4)
    variable = make_media_variable(rng)
    opt_vars = OptimizationVariables([variable])

    da = xr.DataArray(
        np.ones(2),
        dims=("channel",),
        coords={"channel": ["channel_0", "not_a_channel"]},
    )
    with pytest.raises(ValueError, match="values missing"):
        opt_vars.pack(da)


def test_unpack_wrong_size_raises():
    rng = np.random.default_rng(5)
    opt_vars = OptimizationVariables([make_media_variable(rng)])
    with pytest.raises(ValueError, match="expected shape"):
        opt_vars.unpack(np.ones(opt_vars.size + 1))


def test_default_x0_and_bounds():
    rng = np.random.default_rng(6)
    variable = make_media_variable(rng, sizes=(4,))
    opt_vars = OptimizationVariables([variable])

    x0 = opt_vars.x0(total_budget=100.0)
    assert x0.shape == (4,)
    assert np.allclose(x0, 25.0)

    bounds = opt_vars.bounds(total_budget=100.0)
    assert bounds == [(0.0, 100.0)] * 4

    override = [(0.0, 10.0)] * 4
    assert opt_vars.bounds(100.0, overrides={"channel_data": override}) == override


def test_bounds_override_wrong_length_raises():
    rng = np.random.default_rng(7)
    opt_vars = OptimizationVariables([make_media_variable(rng, sizes=(4,))])
    with pytest.raises(ValueError, match="expected 4 bounds pairs"):
        opt_vars.bounds(100.0, overrides={"channel_data": [(0.0, 1.0)]})


def test_duplicate_variable_names_raise():
    rng = np.random.default_rng(8)
    variables = [make_media_variable(rng), make_media_variable(rng)]
    with pytest.raises(ValueError, match="Duplicate variable names"):
        OptimizationVariables(variables)


def test_substitutions_one_entry_per_variable():
    rng = np.random.default_rng(9)
    opt_vars = OptimizationVariables([make_media_variable(rng)])
    subs = opt_vars.substitutions()
    assert set(subs) == {"channel_data"}


def test_non_float_dtype_raises():
    mask = xr.DataArray(
        np.ones(2, dtype=bool), dims=("channel",), coords={"channel": ["a", "b"]}
    )
    with pytest.raises(ValueError, match="float type"):
        MediaVariable(
            name="channel_data",
            mask=mask,
            num_periods=4,
            adstock_periods=2,
            channel_scales=1.0,
            dtype="int64",
        )


def test_pack_extra_dim_raises():
    rng = np.random.default_rng(13)
    variable = make_media_variable(rng, sizes=(3,))
    opt_vars = OptimizationVariables([variable])
    da = opt_vars.unpack(np.arange(3.0))["channel_data"].expand_dims(geo=["G1"])
    with pytest.raises(ValueError, match="unexpected dims"):
        opt_vars.pack({"channel_data": da})


def test_pack_unknown_variable_name_raises():
    """A typo'd key is rejected rather than silently ignored.

    ``pack`` already rejects missing variables and unexpected dims; an extra
    key used to pass silently, so a warm start written against a mistyped name
    would have been dropped without a word.
    """
    rng = np.random.default_rng(14)
    variable = make_media_variable(rng, sizes=(3,))
    opt_vars = OptimizationVariables([variable])
    da = opt_vars.unpack(np.arange(3.0))["channel_data"]

    with pytest.raises(ValueError, match="unknown variables"):
        opt_vars.pack({"channel_data": da, "chanel_data": da})


def test_empty_mask_raises_at_construction():
    """A mask selecting nothing is rejected, not left to divide by zero later."""
    mask = xr.DataArray(
        np.zeros(2, dtype=bool), dims=("channel",), coords={"channel": ["a", "b"]}
    )
    with pytest.raises(ValueError, match="selects no cells"):
        MediaVariable(
            name="channel_data",
            mask=mask,
            num_periods=4,
            adstock_periods=2,
            channel_scales=1.0,
            dtype="float64",
        )


@pytest.mark.parametrize(
    "with_distribution", [False, True], ids=["uniform", "temporal"]
)
def test_channel_scales_applied_on_both_spreading_paths(with_distribution):
    """channel_scales must reach the model tensor however budgets are spread.

    The temporal branch used to build its result from the raw flat vector, so
    the scales division in to_model never reached it and the substituted data
    stayed in monetary units.
    """
    import pytensor.xtensor as ptx
    from pytensor import function

    num_periods, scales = 2, np.array([1.0, 10.0])
    mask = xr.DataArray(
        np.ones(2, dtype=bool), dims=("channel",), coords={"channel": ["c1", "c2"]}
    )
    distribution = (
        ptx.xtensor_constant(
            np.full((num_periods, 2), 1.0 / num_periods),
            dims=("date", FLAT_DIM),
        )
        if with_distribution
        else None
    )
    variable = MediaVariable(
        name="channel_data",
        mask=mask,
        num_periods=num_periods,
        adstock_periods=0,
        channel_scales=scales,
        dtype="float64",
        budget_distribution_over_period_tensor=distribution,
    )
    z = ptx.xtensor("z", shape=(2,), dims=(FLAT_DIM,))
    spend = np.array([100.0, 100.0])
    out = function([z], variable.to_model(z).values, on_unused_input="ignore")(spend)

    # Budgets are per-period rates, so every period carries the full value,
    # divided by that channel's scale. A uniform distribution reproduces the
    # default spreading exactly, so both paths must agree.
    np.testing.assert_allclose(out[0], spend / scales)


def make_lever_variable(
    bounds=((0.05, 0.45), (0.05, 0.45)),
    initial=(0.30, 0.20),
    name: str = "promo_data",
) -> LeverVariable:
    return LeverVariable(
        name=name,
        dim="promo",
        coords=["spring", "fall"],
        bounds=list(bounds) if bounds is not None else None,
        initial_value=np.asarray(initial),
    )


def test_lever_pack_unpack_round_trip():
    lever = make_lever_variable()
    x = np.array([0.11, 0.42])
    da = lever.unpack(x)
    assert da.dims == ("promo",)
    np.testing.assert_array_equal(lever.pack(da), x)
    # order invariant: packing follows the lever's coords, not the input's
    np.testing.assert_array_equal(lever.pack(da.sel(promo=["fall", "spring"])), x)


def test_lever_warm_start_clipped_to_bounds():
    np.testing.assert_allclose(
        make_lever_variable(initial=(0.80, 0.01)).default_x0(100.0), [0.45, 0.05]
    )
    # an unbounded lever starts at its raw model value
    np.testing.assert_allclose(
        make_lever_variable(bounds=None, initial=(0.80, 0.01)).default_x0(100.0),
        [0.80, 0.01],
    )


def test_lever_default_bounds():
    assert make_lever_variable().default_bounds(100.0) == [(0.05, 0.45)] * 2
    assert make_lever_variable(bounds=None).default_bounds(100.0) == [(None, None)] * 2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"bounds": [(0.0, 1.0)]}, "bounds pairs"),
        ({"initial": (0.3,)}, "expected 2"),
    ],
    ids=["bounds_length", "initial_length"],
)
def test_lever_validation_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_lever_variable(**kwargs)


def test_lever_pack_rejects_unknown_coords():
    lever = make_lever_variable()
    da = xr.DataArray(
        np.zeros(2), dims=("promo",), coords={"promo": ["spring", "typo"]}
    )
    with pytest.raises(ValueError, match="coordinates the model does not have"):
        lever.pack(da)


def test_media_and_lever_layout_exercises_the_isel_path():
    """Two variables: the isel branch of variable_slice runs and stays correct.

    With a single variable the slice covers the whole flat vector and
    ``variable_slice`` short-circuits to ``self.flat``, so the ``isel`` branch
    every additional variable depends on never executes.
    """
    from pytensor import function

    rng = np.random.default_rng(11)
    media = make_media_variable(rng, sizes=(3,))
    lever = make_lever_variable()
    opt_vars = OptimizationVariables([media, lever])

    assert opt_vars.slices == {
        "channel_data": slice(0, 3),
        "promo_data": slice(3, 5),
    }
    assert opt_vars.size == 5

    media_slice = opt_vars.variable_slice("channel_data")
    lever_slice = opt_vars.variable_slice("promo_data")
    assert media_slice is not opt_vars.flat  # no short-circuit with two variables
    assert lever_slice.type.shape == (2,)

    # to_model on the isel'd slice reads the lever's own entries.
    x = np.array([10.0, 20.0, 30.0, 0.25, 0.75])
    got = function(
        [opt_vars.flat], lever.to_model(lever_slice).values, on_unused_input="ignore"
    )(x)
    np.testing.assert_allclose(got, [0.25, 0.75])

    # x0 and bounds concatenate in variable order.
    np.testing.assert_allclose(opt_vars.x0(90.0), [30.0, 30.0, 30.0, 0.30, 0.20])
    assert opt_vars.bounds(90.0) == [(0.0, 90.0)] * 3 + [(0.05, 0.45)] * 2

    # pack/unpack round-trips across both variables.
    unpacked = opt_vars.unpack(x)
    assert set(unpacked) == {"channel_data", "promo_data"}
    np.testing.assert_array_equal(opt_vars.pack(unpacked), x)
    assert set(opt_vars.substitutions()) == {"channel_data", "promo_data"}


class _SpendVariable(OptimizationVariable):
    """Test-only stand-in for a second monetary variable.

    Reach-and-frequency budgets and funnel spend are money, so unlike a lever
    they have to come out of the shared pot. This is the minimum such variable:
    a per-entry spend spread over the periods.
    """

    def __init__(self, name: str, coords: list, num_periods: int):
        self.name = name
        self.dims = ("rf_channel",)
        self.coords = {"rf_channel": list(coords)}
        self.num_periods = num_periods
        self.flat_dim = FLAT_DIM

    @property
    def size(self) -> int:
        return len(self.coords["rf_channel"])

    def to_model(self, z):
        return z.rename({self.flat_dim: "rf_channel"}).expand_dims(
            date=self.num_periods
        )

    def unpack(self, x):
        return xr.DataArray(
            np.asarray(x, dtype=float), dims=self.dims, coords=self.coords
        )

    def pack(self, da):
        return da.reindex(self.coords).transpose(*self.dims).values

    def default_x0(self, total_budget):
        return np.full(self.size, total_budget / self.size)

    def default_bounds(self, total_budget):
        return [(0.0, float(total_budget))] * self.size

    def budget_contribution(self, z):
        """Unlike a lever, this one spends from the pot."""
        return z.rename({self.flat_dim: "rf_channel"})


def test_media_is_the_only_budget_contributor_by_default():
    rng = np.random.default_rng(20)
    opt_vars = OptimizationVariables(
        [make_media_variable(rng, sizes=(3,)), make_lever_variable()]
    )
    # A lever is optimized in its own units, so it never draws from the pot.
    assert len(opt_vars.budget_contributions()) == 1


def test_a_second_monetary_variable_shares_the_budget():
    """Two spending variables total together, which is what the sum constraint uses."""
    from pytensor import function

    rng = np.random.default_rng(21)
    media = make_media_variable(rng, sizes=(2,))
    spend = _SpendVariable("rf_data", ["rf1"], num_periods=media.num_periods)
    opt_vars = OptimizationVariables([media, spend])

    contributions = opt_vars.budget_contributions()
    assert len(contributions) == 2

    total = contributions[0].sum()
    for contribution in contributions[1:]:
        total = total + contribution.sum()
    got = function([opt_vars.flat], total.values, on_unused_input="ignore")(
        np.array([30.0, 20.0, 50.0])
    )
    np.testing.assert_allclose(got, 100.0)


@pytest.fixture(scope="module")
def lever_model() -> pm.Model:
    """A model with a one-dim lever node, a date-varying node and an unnamed one."""
    coords = {"promo": ["p1", "p2"], "date": [0, 1, 2]}
    with pm.Model(coords=coords) as model:
        pm.Data("promo_data", np.array([0.1, 0.2]), dims="promo")
        pm.Data("daily_data", np.zeros(3), dims="date")
        pm.Data("both_data", np.zeros((3, 2)), dims=("date", "promo"))
    return model


def test_from_model_reads_dim_coords_and_value(lever_model):
    """The factory needs only a name: the rest is already in the model."""
    lever = LeverVariable.from_model(lever_model, "promo_data", [(0.0, 0.5)] * 2)

    assert lever.name == "promo_data"
    assert lever.dim == "promo"
    assert lever.coords == {"promo": ["p1", "p2"]}
    assert lever.size == 2
    np.testing.assert_allclose(lever.initial_value, [0.1, 0.2])
    assert lever.default_bounds(total_budget=1.0) == [(0.0, 0.5)] * 2


def test_from_model_leaves_bounds_optional(lever_model):
    lever = LeverVariable.from_model(lever_model, "promo_data")
    assert lever.default_bounds(total_budget=1.0) == [(None, None)] * 2


@pytest.mark.parametrize(
    "name, match",
    [
        ("not_a_node", "not a variable with named dims"),
        ("daily_data", "must have exactly one dim"),
        ("both_data", "must have exactly one dim"),
    ],
    ids=["unknown", "date_varying", "multidim"],
)
def test_from_model_rejects_nodes_that_cannot_be_levers(lever_model, name, match):
    with pytest.raises(ValueError, match=match):
        LeverVariable.from_model(lever_model, name, date_dim="date")
