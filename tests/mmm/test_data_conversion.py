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
"""Tests for MMM data conversion utilities."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.data_conversion import to_mmm_dataset


class TestToMmmDatasetXInputTypes:
    """to_mmm_dataset handles all supported X input types."""

    def test_dataframe_x_with_series_y(self):
        X = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5, freq="W"),
                "channel_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        y = pd.Series([10, 20, 30, 40, 50], name="target")
        ds = to_mmm_dataset(X, y, date_column="date", channel_columns=["channel_1"])
        assert "_channel" in ds.data_vars
        assert "_target" in ds.data_vars
        assert "date" in ds.coords

    def test_x_as_dataarray(self):
        da = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=("date", "channel"),
            coords={
                "date": pd.date_range("2023-01-01", periods=2, freq="W"),
                "channel": ["tv", "digital"],
            },
            name="media",
        )
        ds = to_mmm_dataset(da, date_column="date", channel_columns=["tv", "digital"])
        assert "_channel" in ds.data_vars
        assert ds.sizes["channel"] == 2

    def test_x_as_dataarray_without_name_defaults_to_media(self):
        da = xr.DataArray(
            [[1]],
            dims=("date", "channel"),
            coords={
                "date": pd.date_range("2023-01-01", periods=1, freq="W"),
                "channel": ["tv"],
            },
        )
        ds = to_mmm_dataset(da, date_column="date", channel_columns=["tv"])
        assert "_channel" in ds.data_vars

    def test_x_as_dataset_renames_media_to_channel(self):
        ds_in = xr.Dataset(
            {"media": xr.DataArray([[1]], dims=("date", "channel"))},
            coords={
                "date": pd.date_range("2023-01-01", periods=1, freq="W"),
                "channel": ["tv"],
            },
        )
        ds_out = to_mmm_dataset(ds_in, date_column="date", channel_columns=["tv"])
        assert "_channel" in ds_out.data_vars
        assert "media" not in ds_out.data_vars

    def test_x_as_dataset_renames_control_to_control(self):
        ds_in = xr.Dataset(
            {
                "_channel": xr.DataArray([[1]], dims=("date", "channel")),
                "control": xr.DataArray([1.0], dims=("date",)),
            },
            coords={
                "date": pd.date_range("2023-01-01", periods=1, freq="W"),
                "channel": ["tv"],
            },
        )
        ds_out = to_mmm_dataset(
            ds_in,
            date_column="date",
            channel_columns=["tv"],
            control_columns=["price"],
        )
        assert "_control" in ds_out.data_vars
        assert "control" not in ds_out.data_vars


class TestToMmmDatasetYInputTypes:
    """to_mmm_dataset handles all supported y input types."""

    def test_y_as_dataframe(self):
        X = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="W"),
                "channel_1": [1.0, 2.0, 3.0],
            }
        )
        y = pd.DataFrame({"target": [10, 20, 30]})
        ds = to_mmm_dataset(X, y, date_column="date", channel_columns=["channel_1"])
        assert "_target" in ds.data_vars
        assert ds["_target"].values.tolist() == [10, 20, 30]

    def test_y_as_numpy_array(self):
        X = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="W"),
                "channel_1": [1.0, 2.0, 3.0],
            }
        )
        y = np.array([10.0, 20.0, 30.0])
        ds = to_mmm_dataset(X, y, date_column="date", channel_columns=["channel_1"])
        assert "_target" in ds.data_vars
        assert ds["_target"].values.tolist() == [10.0, 20.0, 30.0]

    def test_y_as_xarray_dataarray(self):
        X = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="W"),
                "channel_1": [1.0, 2.0, 3.0],
            }
        )
        y = xr.DataArray([10.0, 20.0, 30.0], dims=("date",))
        ds = to_mmm_dataset(X, y, date_column="date", channel_columns=["channel_1"])
        assert "_target" in ds.data_vars

    def test_y_is_none(self):
        X = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="W"),
                "channel_1": [1.0, 2.0, 3.0],
            }
        )
        ds = to_mmm_dataset(X, date_column="date", channel_columns=["channel_1"])
        assert "_target" not in ds.data_vars

    def test_y_as_dataframe_multi_column_raises(self):
        X = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=2, freq="W"),
                "channel_1": [1.0, 2.0],
            }
        )
        y = pd.DataFrame({"a": [10, 20], "b": [30, 40]})
        with pytest.raises(ValueError, match="must have exactly one column"):
            to_mmm_dataset(X, y, date_column="date", channel_columns=["channel_1"])


class TestToMmmDatasetWithDims:
    """to_mmm_dataset handles extra dimension (panel) columns."""

    def test_panel_data_with_extra_dim(self):
        dates = pd.date_range("2023-01-01", periods=2, freq="W")
        X = pd.DataFrame(
            {
                "date": list(dates) * 2,
                "country": ["US", "US", "UK", "UK"],
                "channel_1": [1.0, 2.0, 3.0, 4.0],
            }
        )
        y = pd.Series([10, 20, 30, 40])
        ds = to_mmm_dataset(
            X,
            y,
            date_column="date",
            dims=("country",),
            channel_columns=["channel_1"],
        )
        assert "_channel" in ds.data_vars
        assert "country" in ds.coords
        assert ds.sizes["country"] == 2

    def test_with_control_columns(self):
        X = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="W"),
                "channel_1": [1.0, 2.0, 3.0],
                "price_index": [1.0, 1.1, 0.9],
            }
        )
        y = pd.Series([10, 20, 30])
        ds = to_mmm_dataset(
            X,
            y,
            date_column="date",
            channel_columns=["channel_1"],
            control_columns=["price_index"],
        )
        assert "_control" in ds.data_vars
        assert "control" in ds.coords

    def test_dataframe_with_extra_columns_deduplicates_date(self):
        X = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-01", "2023-01-08"],
                "market": ["US", "UK", "US"],
                "channel_1": [1.0, 2.0, 3.0],
            }
        )
        y = pd.Series([10, 20, 30])
        ds = to_mmm_dataset(
            X,
            y,
            date_column="date",
            channel_columns=["channel_1"],
        )
        assert "_target" in ds.data_vars
        assert ds.sizes["date"] == 2  # deduplicated


class TestToMmmDatasetDateCoercion:
    """to_mmm_dataset coerces date columns to datetime for DataFrame inputs."""

    def test_string_dates_are_coerced(self):
        X = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-08", "2023-01-15"],
                "channel_1": [1.0, 2.0, 3.0],
            }
        )
        y = pd.Series([10, 20, 30])
        ds = to_mmm_dataset(X, y, date_column="date", channel_columns=["channel_1"])
        assert np.issubdtype(ds.coords["date"].values.dtype, np.datetime64)

    def test_non_standard_date_format(self):
        X = pd.DataFrame(
            {
                "date": ["01/01/2023", "01/08/2023", "01/15/2023"],
                "channel_1": [1.0, 2.0, 3.0],
            }
        )
        y = pd.Series([10, 20, 30])
        ds = to_mmm_dataset(X, y, date_column="date", channel_columns=["channel_1"])
        assert np.issubdtype(ds.coords["date"].values.dtype, np.datetime64)
