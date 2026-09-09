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
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.experimental._data import compute_scales, normalize_data
from pymc_marketing.mmm.scaling import DataDerivedScaling, FixedScaling, Scaling


def test_dataframe_retains_named_raw_variables_and_sorts_dates() -> None:
    frame = pd.DataFrame(
        {
            "week": ["2025-01-02", "2025-01-01", "2025-01-02", "2025-01-01"],
            "geo": ["west", "east", "east", "west"],
            "tv": [20.0, 3.0, 7.0, 10.0],
            "revenue": [200.0, 30.0, 70.0, 100.0],
            "temperature": [4.0, 11.0, 12.0, 5.0],
        }
    )
    ds = normalize_data(
        frame, date_column="week", target_column="revenue", dims=("geo",)
    )
    np.testing.assert_array_equal(
        ds.date.values, pd.date_range("2025-01-01", periods=2).values
    )
    np.testing.assert_array_equal(ds.geo.values, ["west", "east"])
    np.testing.assert_array_equal(ds.tv.values, [[10, 3], [20, 7]])
    np.testing.assert_array_equal(ds.revenue.values, [[100, 30], [200, 70]])
    np.testing.assert_array_equal(ds.temperature.values, [[5, 11], [4, 12]])


def test_canonical_channels_split_by_label_without_losing_custom_coordinates() -> None:
    ds = xr.Dataset(
        {
            "_channel": (("channel", "date"), [[4.0, 2.0], [10.0, 5.0]]),
            "sales": ("date", [40.0, 20.0]),
            "features": (("date", "feature"), [[1.0, 2.0], [3.0, 4.0]]),
        },
        coords={
            "date": pd.to_datetime(["2025-01-02", "2025-01-01"]),
            "channel": ["search", "tv"],
            "feature": ["holiday", "price"],
        },
    )
    normalized = normalize_data(ds, date_column="date", target_column="sales", dims=())
    np.testing.assert_array_equal(normalized.search.values, [2, 4])
    np.testing.assert_array_equal(normalized.tv.values, [5, 10])
    xr.testing.assert_equal(normalized.features, ds.features.sortby("date"))
    xr.testing.assert_equal(normalized.sales, ds.sales.sortby("date"))
    xr.testing.assert_equal(normalized.channel, ds.channel)


def test_xarray_response_reorders_named_dates_and_geographies() -> None:
    ds = xr.Dataset(
        {"tv": (("date", "geo"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={
            "date": pd.date_range("2025-01-01", periods=2),
            "geo": ["west", "east"],
        },
    )
    response = xr.DataArray(
        [[40.0, 20.0], [30.0, 10.0]],
        dims=("geo", "date"),
        coords={"geo": ["east", "west"], "date": ds.date.values[::-1]},
    )
    normalized = normalize_data(
        ds, response, date_column="date", target_column="orders", dims=("geo",)
    )
    np.testing.assert_array_equal(normalized.orders.values, [[10, 20], [30, 40]])


def test_series_response_follows_dataframe_row_labels_not_position() -> None:
    frame = pd.DataFrame(
        {"date": ["2025-01-02", "2025-01-01"], "tv": [2.0, 1.0]}, index=["b", "a"]
    )
    response = pd.Series([10.0, 20.0], index=["a", "b"])
    normalized = normalize_data(
        frame, response, date_column="date", target_column="orders", dims=()
    )
    np.testing.assert_array_equal(normalized.orders.values, [10, 20])


def test_array_response_follows_original_panel_rows() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-01", "2025-01-02", "2025-01-01"],
            "geo": ["west", "east", "east", "west"],
            "tv": [2.0, 3.0, 4.0, 1.0],
        }
    )
    normalized = normalize_data(
        frame,
        np.array([20.0, 30.0, 40.0, 10.0]),
        date_column="date",
        target_column="orders",
        dims=("geo",),
    )
    np.testing.assert_array_equal(normalized.orders.values, [[10, 30], [20, 40]])


@pytest.mark.parametrize("bad_date", ["not-a-date", None, float("nan")])
def test_invalid_dates_are_rejected(bad_date: object) -> None:
    frame = pd.DataFrame({"date": ["2025-01-01", bad_date], "tv": [1.0, 2.0]})
    with pytest.raises(ValueError):
        normalize_data(frame, date_column="date", target_column="y", dims=())


def test_missing_panel_cells_are_not_filled_with_zero() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
            "geo": ["east", "west", "east"],
            "tv": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="missing panel cells"):
        normalize_data(frame, date_column="date", target_column="y", dims=("geo",))


@pytest.mark.parametrize("as_dataset", [False, True])
def test_duplicate_dates_are_rejected(as_dataset: bool) -> None:
    frame = pd.DataFrame({"date": ["2025-01-01", "2025-01-01"], "tv": [1.0, 2.0]})
    data = (
        xr.Dataset(
            {"tv": ("date", [1.0, 2.0])}, coords={"date": pd.to_datetime(frame.date)}
        )
        if as_dataset
        else frame
    )
    with pytest.raises(ValueError):
        normalize_data(data, date_column="date", target_column="y", dims=())


def test_response_coordinate_mismatch_is_not_silently_intersected() -> None:
    ds = xr.Dataset(
        {"tv": ("date", [1.0, 2.0])},
        coords={"date": pd.date_range("2025-01-01", periods=2)},
    )
    response = xr.DataArray(
        [10.0, 20.0],
        dims="date",
        coords={"date": pd.date_range("2025-01-02", periods=2)},
    )
    with pytest.raises(ValueError):
        normalize_data(
            ds, response, date_column="date", target_column="orders", dims=()
        )


def test_missing_values_remain_missing_in_unused_named_inputs() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "unused": [np.nan, 1.0],
            "orders": [2.0, 4.0],
        }
    )
    ds = normalize_data(frame, date_column="date", target_column="orders", dims=())
    assert np.isnan(ds.unused.values[0])
    scales = compute_scales(ds, channels=(), target_column="orders", dims=())
    assert scales.target_scale.item() == 4.0


def test_signed_reductions_and_zero_divisors_remain_usable_for_future_data() -> None:
    ds = xr.Dataset(
        {
            "negative": ("date", [-8.0, -4.0]),
            "zero": ("date", [0.0, 0.0]),
            "orders": ("date", [-2.0, 2.0]),
        },
        coords={"date": pd.date_range("2025-01-01", periods=2)},
    )
    scales = compute_scales(
        ds,
        channels=("negative", "zero"),
        target_column="orders",
        dims=(),
        scaling={"target": DataDerivedScaling(method="mean", dims=())},
    )
    np.testing.assert_allclose(
        scales.channel_scale.sel(channel=["negative", "zero"]), [-4, 1]
    )
    assert scales.target_scale.item() == 1.0
    future = xr.DataArray(
        [8.0, 3.0], dims="channel", coords={"channel": ["negative", "zero"]}
    )
    np.testing.assert_allclose(future / scales.channel_scale, [-2.0, 3.0])


def test_fixed_dataarray_scales_align_channel_and_geo_labels() -> None:
    ds = xr.Dataset(
        {
            "tv": (("date", "geo"), [[10.0, 20.0], [30.0, 40.0]]),
            "search": (("date", "geo"), [[5.0, 10.0], [15.0, 20.0]]),
        },
        coords={
            "date": pd.date_range("2025-01-01", periods=2),
            "geo": ["west", "east"],
        },
    )
    scale = xr.DataArray(
        [[20.0, 5.0], [40.0, 10.0]],
        dims=("channel", "geo"),
        coords={"channel": ["search", "tv"], "geo": ["east", "west"]},
    )
    result = compute_scales(
        ds,
        channels=("tv", "search"),
        target_column=None,
        dims=("geo",),
        scaling={"channel": FixedScaling(dims=(), value=scale)},
    )
    np.testing.assert_allclose(
        result.channel_scale.transpose("geo", "channel"), [[10, 5], [40, 20]]
    )
    xr.testing.assert_equal(
        scale,
        xr.DataArray(
            [[20.0, 5.0], [40.0, 10.0]],
            dims=("channel", "geo"),
            coords={"channel": ["search", "tv"], "geo": ["east", "west"]},
        ),
    )


def test_fixed_dict_scales_follow_labels_and_preserve_configuration() -> None:
    ds = xr.Dataset(
        {"tv": ("date", [2.0]), "search": ("date", [3.0])},
        coords={"date": pd.date_range("2025-01-01", periods=1)},
    )
    config = {
        "channel": {
            "method": "fixed",
            "dims": (),
            "value": {"search": 30.0, "tv": 20.0},
        }
    }
    result = compute_scales(
        ds, channels=("tv", "search"), target_column=None, dims=(), scaling=config
    )
    np.testing.assert_allclose(result.channel_scale, [20, 30])
    assert config["channel"]["method"] == "fixed"


@pytest.mark.parametrize(
    "labels", [["tv", "unknown"], ["tv"], ["tv", "search", "extra"]]
)
def test_fixed_scale_labels_must_match_exactly(labels: list[str]) -> None:
    ds = xr.Dataset(
        {"tv": ("date", [1.0]), "search": ("date", [2.0])},
        coords={"date": pd.date_range("2025-01-01", periods=1)},
    )
    scale = xr.DataArray(
        np.ones(len(labels)), dims="channel", coords={"channel": labels}
    )
    with pytest.raises(ValueError):
        compute_scales(
            ds,
            channels=("tv", "search"),
            target_column=None,
            dims=(),
            scaling={"channel": FixedScaling(dims=(), value=scale)},
        )


@pytest.mark.parametrize(
    "value",
    [
        float("inf"),
        {"tv": float("inf")},
        xr.DataArray([float("inf")], dims="channel", coords={"channel": ["tv"]}),
    ],
)
def test_infinite_fixed_divisors_are_rejected(value: object) -> None:
    ds = xr.Dataset(
        {"tv": ("date", [1.0])}, coords={"date": pd.date_range("2025-01-01", periods=1)}
    )
    with pytest.raises(ValueError, match="finite"):
        compute_scales(
            ds,
            channels=("tv",),
            target_column=None,
            dims=(),
            scaling={"channel": FixedScaling(dims=(), value=value)},
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_used_training_data_must_be_finite_even_with_fixed_scaling(
    value: float,
) -> None:
    ds = xr.Dataset(
        {"tv": ("date", [1.0, value])},
        coords={"date": pd.date_range("2025-01-01", periods=2)},
    )
    with pytest.raises(ValueError, match="finite"):
        compute_scales(
            ds,
            channels=("tv",),
            target_column=None,
            dims=(),
            scaling={"channel": FixedScaling(dims=(), value=2.0)},
        )


def test_unused_scaling_has_no_data_requirements() -> None:
    ds = xr.Dataset(coords={"date": pd.date_range("2025-01-01", periods=2)})
    config = Scaling(
        channel=FixedScaling(dims=(), value=1.0),
        target=FixedScaling(dims=(), value=1.0),
    )
    xr.testing.assert_identical(
        compute_scales(ds, channels=(), target_column=None, dims=(), scaling=config),
        xr.Dataset(),
    )
