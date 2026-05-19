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
"""Tests for :func:`to_bass_dataset`."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.bass.data import to_bass_dataset


class TestFromXarray:
    def test_with_T_coord(self):
        ds = xr.Dataset(
            {"observed": ("T", [10, 20, 30])},
            coords={"T": [0, 5, 10]},
        )
        result = to_bass_dataset(ds)
        assert result is ds
        assert list(result.coords["T"].values) == [0, 5, 10]

    def test_without_T_coord(self):
        ds = xr.Dataset({"observed": ("T", [10, 20, 30])})
        result = to_bass_dataset(ds)
        assert "T" in result.coords
        assert list(result.coords["T"].values) == [0, 1, 2]

    def test_with_additional_coords(self):
        ds = xr.Dataset(
            {"observed": (("T", "product"), np.ones((3, 2)))},
            coords={"T": [0, 1, 2], "product": ["A", "B"]},
        )
        result = to_bass_dataset(ds)
        assert "product" in result.coords
        assert list(result.coords["product"].values) == ["A", "B"]


class TestFromDataFrame:
    def test_with_observed_column(self):
        df = pd.DataFrame({"observed": [10, 20, 30]})
        result = to_bass_dataset(df)
        assert "observed" in result.data_vars
        assert list(result["observed"].values) == [10, 20, 30]
        assert list(result.coords["T"].values) == [0, 1, 2]

    def test_with_observed_and_T_columns(self):
        df = pd.DataFrame({"T": [5, 10, 15], "observed": [10, 20, 30]})
        result = to_bass_dataset(df)
        assert list(result.coords["T"].values) == [5, 10, 15]
        assert list(result["observed"].values) == [10, 20, 30]

    def test_with_observed_and_extra_numeric_vars(self):
        df = pd.DataFrame(
            {
                "observed": [10, 20, 30],
                "covariate": [0.5, 0.7, 0.9],
                "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            }
        )
        result = to_bass_dataset(df)
        assert "covariate" in result.data_vars
        assert list(result["covariate"].values) == [0.5, 0.7, 0.9]
        # date is non-numeric → not carried over
        assert "date" not in result.data_vars

    def test_wide_multiple_products(self):
        df = pd.DataFrame(
            {
                "product_A": [10, 25, 40],
                "product_B": [15, 30, 45],
            }
        )
        result = to_bass_dataset(df)
        assert result["observed"].dims == ("T", "product")
        assert list(result.coords["product"].values) == ["product_A", "product_B"]
        assert result.sizes["T"] == 3
        assert result.sizes["product"] == 2

    def test_wide_with_T_column(self):
        df = pd.DataFrame(
            {
                "T": [0, 5, 10],
                "product_A": [10, 25, 40],
                "product_B": [15, 30, 45],
            }
        )
        result = to_bass_dataset(df)
        assert list(result.coords["T"].values) == [0, 5, 10]
        assert list(result.coords["product"].values) == ["product_A", "product_B"]

    def test_no_numeric_columns_raises(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-02-01"])})
        with pytest.raises(ValueError, match="no valid data columns"):
            to_bass_dataset(df)


class TestFromSeries:
    def test_simple(self):
        s = pd.Series([10, 20, 30], name="observed")
        result = to_bass_dataset(s)
        assert list(result["observed"].values) == [10, 20, 30]
        assert list(result.coords["T"].values) == [0, 1, 2]


class TestFromArray:
    def test_1d(self):
        arr = np.array([10, 20, 30])
        result = to_bass_dataset(arr)
        assert list(result["observed"].values) == [10, 20, 30]
        assert list(result.coords["T"].values) == [0, 1, 2]

    def test_2d(self):
        arr = np.array([[10, 20], [30, 40], [50, 60]])
        result = to_bass_dataset(arr)
        assert result["observed"].shape == (3, 2)
        assert list(result.coords["product"].values) == ["P0", "P1"]
        assert list(result.coords["T"].values) == [0, 1, 2]

    def test_3d_raises(self):
        arr = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="3 dimensions"):
            to_bass_dataset(arr)


class TestInvalidType:
    def test_raises(self):
        with pytest.raises(TypeError, match="Unsupported data type"):
            to_bass_dataset("not valid data")
