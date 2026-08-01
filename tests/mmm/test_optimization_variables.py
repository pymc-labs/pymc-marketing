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
    MediaVariable,
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
