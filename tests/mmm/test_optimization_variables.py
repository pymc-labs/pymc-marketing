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
import pytest
import xarray as xr

from pymc_marketing.mmm.optimization_variables import (
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


def make_lever_variable(
    bounds=((0.05, 0.45), (0.05, 0.45)),
    initial=(0.30, 0.20),
) -> LeverVariable:
    return LeverVariable(
        name="promo_data",
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
    assert list(da.coords["promo"].values) == ["spring", "fall"]
    np.testing.assert_array_equal(lever.pack(da), x)
    # order-invariant pack
    np.testing.assert_array_equal(lever.pack(da.sel(promo=["fall", "spring"])), x)


def test_lever_warm_start_clipped_to_bounds():
    lever = make_lever_variable(initial=(0.80, 0.01))
    np.testing.assert_allclose(lever.default_x0(100.0), [0.45, 0.05])
    # unbounded lever warm-starts at the raw model value
    free = make_lever_variable(bounds=None, initial=(0.80, 0.01))
    np.testing.assert_allclose(free.default_x0(100.0), [0.80, 0.01])


def test_lever_default_bounds():
    lever = make_lever_variable()
    assert lever.default_bounds(100.0) == [(0.05, 0.45), (0.05, 0.45)]
    free = make_lever_variable(bounds=None)
    assert free.default_bounds(100.0) == [(None, None)] * 2


def test_lever_validation_raises():
    with pytest.raises(ValueError, match="bounds pairs"):
        make_lever_variable(bounds=[(0.0, 1.0)])
    with pytest.raises(ValueError, match="entries, expected"):
        make_lever_variable(initial=(0.3,))


def test_media_and_lever_space_layout():
    """Media head + lever tail: slices tile, x0 and bounds concatenate."""
    rng = np.random.default_rng(10)
    media = make_media_variable(rng, sizes=(3,))
    lever = make_lever_variable()
    opt_vars = OptimizationVariables([media, lever])

    assert opt_vars.slices["channel_data"] == slice(0, 3)
    assert opt_vars.slices["promo_data"] == slice(3, 5)
    assert opt_vars.size == 5

    x0 = opt_vars.x0(total_budget=90.0)
    np.testing.assert_allclose(x0[:3], 30.0)
    np.testing.assert_allclose(x0[3:], [0.30, 0.20])

    bounds = opt_vars.bounds(total_budget=90.0)
    assert bounds[:3] == [(0.0, 90.0)] * 3
    assert bounds[3:] == [(0.05, 0.45), (0.05, 0.45)]

    unpacked = opt_vars.unpack(np.array([10.0, 30.0, 50.0, 0.1, 0.2]))
    assert set(unpacked) == {"channel_data", "promo_data"}
    np.testing.assert_allclose(unpacked["promo_data"].values, [0.1, 0.2])


class _ScalarVariable(OptimizationVariable):
    """Minimal second variable: its flat slice is the model tensor as-is."""

    def __init__(self, name: str, coords: list, flat_dim: str = "budgets_flat"):
        self.name = name
        self.dims = ("lever",)
        self.coords = {"lever": list(coords)}
        self.flat_dim = flat_dim

    @property
    def size(self) -> int:
        return len(self.coords["lever"])

    def to_model(self, z):
        return z.rename({self.flat_dim: "lever"})

    def unpack(self, x):
        return xr.DataArray(
            np.asarray(x, dtype=float), dims=self.dims, coords=self.coords
        )

    def pack(self, da):
        return da.reindex(self.coords).transpose(*self.dims).values

    def default_x0(self, total_budget):
        return np.full(self.size, 0.5)

    def default_bounds(self, total_budget):
        return [(0.0, 1.0)] * self.size


def test_multi_variable_layout_and_isel_path():
    """Two variables: the isel branch of variable_slice runs and stays correct.

    With one variable the slice covers the whole flat vector and
    `variable_slice` short-circuits to `self.flat`, so the `isel` branch every
    additional variable depends on never executes.
    """
    from pytensor import function

    rng = np.random.default_rng(11)
    media = make_media_variable(rng, sizes=(3,))
    lever = _ScalarVariable("lever_data", ["a", "b"])
    opt_vars = OptimizationVariables([media, lever])

    # Layout: contiguous, tiling, media first.
    assert opt_vars.slices == {
        "channel_data": slice(0, 3),
        "lever_data": slice(3, 5),
    }
    assert opt_vars.size == 5

    # The isel branch is what the second variable takes.
    media_slice = opt_vars.variable_slice("channel_data")
    lever_slice = opt_vars.variable_slice("lever_data")
    assert media_slice is not opt_vars.flat  # no short-circuit with 2 variables
    assert lever_slice.type.shape == (2,)

    # to_model on the isel'd slice reads the right entries of the flat vector.
    x = np.array([10.0, 20.0, 30.0, 0.25, 0.75])
    lever_tensor = function(
        [opt_vars.flat], lever.to_model(lever_slice).values, on_unused_input="ignore"
    )(x)
    np.testing.assert_allclose(lever_tensor, [0.25, 0.75])

    # x0 and bounds concatenate in variable order.
    np.testing.assert_allclose(opt_vars.x0(90.0), [30.0, 30.0, 30.0, 0.5, 0.5])
    assert opt_vars.bounds(90.0) == [(0.0, 90.0)] * 3 + [(0.0, 1.0)] * 2

    # pack/unpack round-trip across both variables.
    unpacked = opt_vars.unpack(x)
    assert set(unpacked) == {"channel_data", "lever_data"}
    np.testing.assert_allclose(unpacked["lever_data"].values, [0.25, 0.75])
    np.testing.assert_array_equal(opt_vars.pack(unpacked), x)

    # substitutions has one entry per variable.
    assert set(opt_vars.substitutions()) == {"channel_data", "lever_data"}


def test_container_rejects_mismatched_flat_dim():
    """A variable naming a different flat dim is rejected, not silently renamed.

    Renaming would turn an internally coherent variable (its own tensors keyed
    on its own name) into an incoherent graph that only fails later, in the
    temporal-distribution branch.
    """
    rng = np.random.default_rng(12)
    media = make_media_variable(rng, sizes=(2,))
    lever = _ScalarVariable("lever_data", ["a"], flat_dim="something_else")
    with pytest.raises(ValueError, match="does not match the container"):
        OptimizationVariables([media, lever], flat_dim="budgets_flat")


def test_pack_extra_dim_raises():
    rng = np.random.default_rng(13)
    variable = make_media_variable(rng, sizes=(3,))
    opt_vars = OptimizationVariables([variable])
    da = opt_vars.unpack(np.arange(3.0))["channel_data"].expand_dims(geo=["G1"])
    with pytest.raises(ValueError, match="unexpected dims"):
        opt_vars.pack({"channel_data": da})
