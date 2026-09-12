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
"""Labeled dataset validation contracts for the experimental MMM."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.experimental._data import _align_labels, validate_dataset


def test_non_dataset_inputs_are_rejected():
    frame = pd.DataFrame(
        {"date": pd.date_range("2025-01-01", periods=2), "y": [1.0, 2.0]}
    )
    with pytest.raises(TypeError):
        validate_dataset(frame)


@pytest.mark.parametrize(
    "ds",
    [
        xr.Dataset({"y": ("geo", [1.0, 2.0])}),
        xr.Dataset({"y": ("geo", np.empty(0))}, coords={"geo": np.empty(0, dtype=str)}),
        xr.Dataset({"y": ("geo", [1.0, 2.0])}, coords={"geo": ["a", "a"]}),
        xr.Dataset({"y": ("geo", [1.0, 2.0])}, coords={"geo": [1.0, np.nan]}),
    ],
    ids=["unlabeled", "empty", "duplicate-label", "missing-label"],
)
def test_dimensions_need_unique_nonmissing_labels(ds):
    with pytest.raises(ValueError, match="geo"):
        validate_dataset(ds)


def test_string_dates_become_a_datetime_index_in_the_given_order():
    labels = ["2025-01-01", "2025-01-08", "2025-01-15"]
    ds = xr.Dataset({"y": ("date", [1.0, 2.0, 3.0])}, coords={"date": labels})
    validated = validate_dataset(ds)
    assert isinstance(validated.indexes["date"], pd.DatetimeIndex)
    np.testing.assert_array_equal(validated.date.values, pd.to_datetime(labels).values)
    np.testing.assert_array_equal(validated.y.values, ds.y.values)


@pytest.mark.parametrize(
    "dates",
    [
        [0, 1],
        [1.7e9, 1.8e9],
        ["2025-01-01", "not-a-date"],
        ["2025-01-08", "2025-01-01"],
        [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
        pd.date_range("2025-01-01", periods=2, tz="UTC"),
    ],
    ids=["integers", "floats", "unparseable", "decreasing", "duplicate", "tz-aware"],
)
def test_invalid_date_coordinates_are_rejected(dates):
    ds = xr.Dataset({"y": ("date", [1.0, 2.0])}, coords={"date": dates})
    with pytest.raises(ValueError):
        validate_dataset(ds)


def test_datetime_dates_and_non_date_dimensions_pass_through_unchanged():
    ds = xr.Dataset(
        {"y": (("date", "geo", "channel"), np.arange(8.0).reshape(2, 2, 2))},
        coords={
            "date": pd.date_range("2025-01-01", periods=2, freq="W"),
            "geo": [10, 20],
            "channel": ["tv", "radio"],
        },
    )
    xr.testing.assert_identical(validate_dataset(ds), ds)


def test_align_labels_restores_reference_order_on_every_dimension():
    reference = xr.Dataset(coords={"geo": ["a", "b", "c"], "channel": ["tv", "radio"]})
    array = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dims=("geo", "channel"),
        coords={"geo": ["c", "b", "a"], "channel": ["radio", "tv"]},
    )
    aligned = _align_labels(array, reference)
    assert aligned.dims == ("geo", "channel")
    np.testing.assert_array_equal(aligned.geo.values, ["a", "b", "c"])
    np.testing.assert_array_equal(aligned.channel.values, ["tv", "radio"])
    np.testing.assert_array_equal(aligned.values, [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]])


@pytest.mark.parametrize(
    "array",
    [
        xr.DataArray([1.0, 2.0], dims="geo"),
        xr.DataArray([1.0, 2.0], dims="region", coords={"region": ["a", "b"]}),
        xr.DataArray([1.0, 2.0], dims="geo", coords={"geo": ["a", "x"]}),
        xr.DataArray([1.0, 2.0], dims="geo", coords={"geo": ["a", "a"]}),
        xr.DataArray([1.0], dims="geo", coords={"geo": ["a"]}),
        xr.DataArray([1.0, 2.0, 3.0], dims="geo", coords={"geo": ["a", "b", "c"]}),
    ],
    ids=[
        "unlabeled",
        "unknown-dimension",
        "unknown-label",
        "duplicate-label",
        "subset",
        "superset",
    ],
)
def test_align_labels_requires_the_reference_label_set(array):
    reference = xr.Dataset(coords={"geo": ["a", "b"]})
    with pytest.raises(ValueError):
        _align_labels(array, reference)
