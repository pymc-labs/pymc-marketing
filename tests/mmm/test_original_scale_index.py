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
"""Unit tests for OriginalScaleIndex."""

import numpy as np
import pytest
import xarray as xr
from xarray.indexes import PandasIndex

from pymc_marketing.mmm.original_scale_index import OriginalScaleIndex


@pytest.fixture()
def x() -> np.ndarray:
    return np.linspace(0, 1, 5)


@pytest.fixture()
def channels() -> np.ndarray:
    return np.array(["TV", "Radio"])


@pytest.fixture()
def channel_scale(channels) -> xr.DataArray:
    return xr.DataArray(
        [5000.0, 1200.0],
        dims=["channel"],
        coords={"channel": channels},
    )


@pytest.fixture()
def curve(x, channels, channel_scale) -> xr.DataArray:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(1, 20, len(channels), len(x)))
    da = xr.DataArray(
        data,
        dims=["chain", "draw", "channel", "x"],
        coords={"chain": [0], "draw": np.arange(20), "channel": channels, "x": x},
    )
    return da.drop_indexes(["x", "channel"]).set_xindex(
        ["x", "channel"], OriginalScaleIndex, channel_scale=channel_scale
    )


def test_set_xindex_creates_original_scale_index(curve) -> None:
    assert isinstance(curve.xindexes["channel"], OriginalScaleIndex)
    assert isinstance(curve.xindexes["x"], OriginalScaleIndex)
    assert curve.xindexes["x"] is curve.xindexes["channel"]


def test_x_original_property(curve, x, channel_scale) -> None:
    idx = curve.xindexes["channel"]
    # x_original should have dims (x, channel) with original-domain values
    xo = idx.x_original
    assert set(xo.dims) == {"x", "channel"}
    np.testing.assert_allclose(xo.sel(channel="TV").values, x * 5000.0)
    np.testing.assert_allclose(xo.sel(channel="Radio").values, x * 1200.0)


@pytest.mark.parametrize("channel,expected_scale", [("TV", 5000.0), ("Radio", 1200.0)])
def test_sel_channel_returns_original_domain_x(
    curve, x, channel, expected_scale
) -> None:
    result = curve.sel(channel=channel)
    np.testing.assert_allclose(result.coords["x"].values, x * expected_scale)


def test_sel_channel_removes_channel_dim(curve) -> None:
    result = curve.sel(channel="TV")
    assert "channel" not in result.dims


def test_sel_channel_x_index_is_pandas_index_after_selection(curve) -> None:
    result = curve.sel(channel="TV")
    assert isinstance(result.xindexes["x"], PandasIndex)


def test_sel_unknown_channel_raises(curve) -> None:
    with pytest.raises(KeyError, match="not found"):
        curve.sel(channel="Unknown")


def test_isel_x_slice_returns_new_original_scale_index(curve) -> None:
    result = curve.isel(x=slice(0, 3))
    assert isinstance(result.xindexes["channel"], OriginalScaleIndex)
    assert result.sizes["x"] == 3


def test_isel_channel_scalar_returns_pandas_index(curve, x) -> None:
    result = curve.isel(channel=0)
    assert "channel" not in result.dims
    expected_x = x * 5000.0  # TV is index 0
    np.testing.assert_allclose(result.coords["x"].values, expected_x)


def test_equals_same_index(x, channel_scale) -> None:
    idx1 = OriginalScaleIndex._from_x_and_scale(x, channel_scale)
    idx2 = OriginalScaleIndex._from_x_and_scale(x.copy(), channel_scale.copy())
    assert idx1.equals(idx2)


def test_equals_different_scales(x, channels) -> None:
    scale1 = xr.DataArray(
        [5000.0, 1200.0], dims=["channel"], coords={"channel": channels}
    )
    scale2 = xr.DataArray(
        [9999.0, 1200.0], dims=["channel"], coords={"channel": channels}
    )
    idx1 = OriginalScaleIndex._from_x_and_scale(x, scale1)
    idx2 = OriginalScaleIndex._from_x_and_scale(x, scale2)
    assert not idx1.equals(idx2)


def test_repr_contains_dims_and_n_x(curve) -> None:
    r = repr(curve.xindexes["channel"])
    assert "channel" in r
    assert "5" in r  # n_x=5


def test_az_hdi_preserves_original_scale_index(curve) -> None:
    az = pytest.importorskip("arviz")
    hdi = az.hdi(curve)
    assert isinstance(hdi.xindexes["channel"], OriginalScaleIndex)


def test_az_hdi_sel_channel_gives_original_domain_x(curve, x) -> None:
    az = pytest.importorskip("arviz")
    hdi = az.hdi(curve)
    scales = {"TV": 5000.0, "Radio": 1200.0}
    for ch, scale in scales.items():
        hdi_ch = hdi.sel(channel=ch)
        np.testing.assert_allclose(hdi_ch.coords["x"].values, x * scale)


def test_to_series_uses_original_domain_x(curve, x) -> None:
    tv = curve.isel(chain=0).sel(channel="TV")
    series = tv.to_series()
    x_in_series = series.index.get_level_values("x")
    expected = x * 5000.0
    np.testing.assert_allclose(np.sort(np.unique(x_in_series)), np.sort(expected))


def test_from_variables_raises_without_x(channels, channel_scale) -> None:
    variables = {"channel": xr.Variable("channel", channels)}
    with pytest.raises(ValueError, match="requires an 'x' variable"):
        OriginalScaleIndex.from_variables(
            variables, options={"channel_scale": channel_scale}
        )


def test_from_variables_raises_without_channel_scale(x) -> None:
    variables = {"x": xr.Variable("x", x)}
    with pytest.raises(ValueError, match="requires 'channel_scale'"):
        OriginalScaleIndex.from_variables(variables, options={})


def test_panel_2d_scale_sel_both_dims_gives_original_domain_x(x) -> None:
    """Panel model: channel_scale has dims ('country', 'channel')."""
    countries = np.array(["US", "UK"])
    channels = np.array(["TV", "Radio"])
    scale_data = np.array([[5000.0, 1200.0], [3000.0, 800.0]])
    channel_scale = xr.DataArray(
        scale_data,
        dims=["country", "channel"],
        coords={"country": countries, "channel": channels},
    )
    da = xr.DataArray(
        np.ones((len(countries), len(channels), len(x))),
        dims=["country", "channel", "x"],
        coords={"country": countries, "channel": channels, "x": x},
    )
    da = da.drop_indexes(["x", "channel", "country"]).set_xindex(
        ["x", "channel", "country"], OriginalScaleIndex, channel_scale=channel_scale
    )

    result = da.sel(channel="TV", country="US")
    np.testing.assert_allclose(result.coords["x"].values, x * 5000.0)

    result_uk_radio = da.sel(channel="Radio", country="UK")
    np.testing.assert_allclose(result_uk_radio.coords["x"].values, x * 800.0)


def test_panel_2d_scale_partial_sel_returns_original_scale_index(x) -> None:
    """Selecting only one dim of a 2-D scale preserves the index for the remaining dim."""
    countries = np.array(["US", "UK"])
    channels = np.array(["TV", "Radio"])
    scale_data = np.array([[5000.0, 1200.0], [3000.0, 800.0]])
    channel_scale = xr.DataArray(
        scale_data,
        dims=["country", "channel"],
        coords={"country": countries, "channel": channels},
    )
    da = xr.DataArray(
        np.ones((len(countries), len(channels), len(x))),
        dims=["country", "channel", "x"],
        coords={"country": countries, "channel": channels, "x": x},
    )
    da = da.drop_indexes(["x", "channel", "country"]).set_xindex(
        ["x", "channel", "country"], OriginalScaleIndex, channel_scale=channel_scale
    )

    partial = da.sel(channel="TV")
    assert "country" in partial.dims
    assert isinstance(partial.xindexes.get("x"), OriginalScaleIndex)

    result = partial.sel(country="US")
    np.testing.assert_allclose(result.coords["x"].values, x * 5000.0)
