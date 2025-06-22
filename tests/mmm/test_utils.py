#   Copyright 2022 - 2025 The PyMC Labs Developers
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
from sklearn.preprocessing import MaxAbsScaler

from pymc_marketing.mmm.utils import (
    add_noise_to_channel_allocation,
    apply_sklearn_transformer_across_dim,
    create_index,
    create_new_spend_data,
    create_zero_dataset,
    transform_1d_array,
)


@pytest.fixture
def mock_method():
    def _mock_method(x):
        if x.ndim != 2:
            raise ValueError("x must be 2-dimensional")

        return x * 2

    return _mock_method


@pytest.fixture
def create_mock_mmm_return_data():
    def _create_mock_mm_return_data(combined: bool) -> xr.DataArray:
        dates = pd.date_range(start="2020-01-01", end="2020-01-31", freq="W-MON")
        data = xr.DataArray(
            np.ones(shape=(1, 3, len(dates), 2)),
            coords={
                "chain": [1],
                "draw": [1, 2, 3],
                "date": dates,
                "channel": ["channel1", "channel2"],
            },
        )

        if combined:
            data = data.stack(sample=("chain", "draw"))

        return data

    return _create_mock_mm_return_data


@pytest.mark.parametrize("combined", [True, False])
def test_apply_sklearn_function_across_dim(
    mock_method, create_mock_mmm_return_data, combined: bool
) -> None:
    # Data that would be returned from a MMM model
    data = create_mock_mmm_return_data(combined=combined)
    result = apply_sklearn_transformer_across_dim(
        data,
        mock_method,
        dim_name="date",
    )

    xr.testing.assert_allclose(result, data * 2)


@pytest.mark.parametrize("constructor", [pd.Series, np.array])
def test_transform_1d_array(constructor):
    transform = MaxAbsScaler()
    y = constructor([1, 2, 3, 4, 5])
    transform.fit(np.array(y)[:, None])
    expected = np.array([1, 2, 3, 4, 5]) / 5
    result = transform_1d_array(transform.transform, y)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "spend, adstock_max_lag, one_time, spend_leading_up, expected_result",
    [
        (
            [1, 2],
            2,
            True,
            None,
            [[0, 0], [0, 0], [1, 2], [0, 0], [0, 0]],
        ),
        (
            [1, 2],
            2,
            False,
            None,
            [[0, 0], [0, 0], [1, 2], [1, 2], [1, 2]],
        ),
        (
            [1, 2],
            2,
            True,
            [3, 4],
            [[3, 4], [3, 4], [1, 2], [0, 0], [0, 0]],
        ),
    ],
)
def test_create_new_spend_data(
    spend, adstock_max_lag, one_time, spend_leading_up, expected_result
) -> None:
    spend = np.array(spend)
    if spend_leading_up is not None:
        spend_leading_up = np.array(spend_leading_up)
    new_spend_data = create_new_spend_data(
        spend, adstock_max_lag, one_time, spend_leading_up
    )

    np.testing.assert_allclose(
        new_spend_data,
        np.array(expected_result),
    )


def test_create_new_spend_data_value_errors() -> None:
    with pytest.raises(
        ValueError, match="spend_leading_up must be the same length as the spend"
    ):
        create_new_spend_data(
            spend=np.array([1, 2]),
            adstock_max_lag=2,
            one_time=True,
            spend_leading_up=np.array([3, 4, 5]),
        )


def test_add_noise_to_channel_allocation():
    # Create a simple DataFrame with channel data
    df = pd.DataFrame(
        {
            "channel1": [10, 20, 30, 40, 50],
            "channel2": [5, 10, 15, 20, 25],
            "target": [100, 200, 300, 400, 500],
        }
    )

    channels = ["channel1", "channel2"]

    # Test with fixed seed for reproducibility
    result = add_noise_to_channel_allocation(df, channels, rel_std=0.1, seed=42)

    # Check that the DataFrame was not modified in place
    pd.testing.assert_frame_equal(
        df,
        pd.DataFrame(
            {
                "channel1": [10, 20, 30, 40, 50],
                "channel2": [5, 10, 15, 20, 25],
                "target": [100, 200, 300, 400, 500],
            }
        ),
    )

    # Check that noise was added (values changed)
    assert not np.allclose(df[channels].values, result[channels].values)

    # Check that non-channel columns remain unchanged
    np.testing.assert_array_equal(df["target"].values, result["target"].values)

    # Check that noise is centered around original values (approximately)
    # The means should be close with small sample size
    assert np.abs(df["channel1"].mean() - result["channel1"].mean()) < 5
    assert np.abs(df["channel2"].mean() - result["channel2"].mean()) < 3

    # Test no negative values
    assert (result[channels] >= 0).all().all(), "No negative values in channels"


class FakeMMM:
    def __init__(self):
        # Create a simple dataset
        dates = pd.date_range("2022-01-01", "2022-01-31", freq="D")
        self.X = pd.DataFrame(
            {
                "date": dates,
                "region": ["A"] * 15 + ["B"] * 16,
                "channel1": np.random.rand(31) * 10,
                "channel2": np.random.rand(31) * 5,
                "control1": np.random.rand(31),
                "extra_col": np.ones(31),
            }
        )
        self.date_column = "date"
        self.channel_columns = ["channel1", "channel2"]
        self.control_columns = ["control1"]
        self.dims = ["region"]


def test_create_zero_dataset():
    # Create a fake model
    model = FakeMMM()

    # Test basic zero dataset
    start_date = "2022-02-01"
    end_date = "2022-02-10"
    result = create_zero_dataset(model, start_date, end_date)

    # Check results
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10 * 2  # 10 days by 2 regions
    assert list(result.columns) == list(model.X.columns)
    assert np.all(result[model.channel_columns] == 0)
    assert np.all(result[model.control_columns] == 0)

    # Test with channel_xr
    # Create a simple xarray Dataset with channel values
    region_coords = np.array(["A", "B"])
    channel_values = xr.Dataset(
        data_vars={
            "channel1": (["region"], np.array([5.0, 7.0])),
        },
        coords={"region": region_coords},
    )

    result_with_channels = create_zero_dataset(
        model, start_date, end_date, channel_values
    )

    # Check results
    assert np.all(
        result_with_channels.loc[result_with_channels.region == "A", "channel1"] == 5.0
    )
    assert np.all(
        result_with_channels.loc[result_with_channels.region == "B", "channel1"] == 7.0
    )
    assert np.all(result_with_channels["channel2"] == 0)  # Not provided in channel_xr


@pytest.mark.parametrize(
    "dims, take, expected_result",
    [
        pytest.param(
            ("date",),
            ("date",),
            (slice(None),),
            id="empty-slice",
        ),
        pytest.param(
            ("date", "product", "geo"),
            ("date", "geo"),
            (slice(None), 0, slice(None)),
            id="drop-product",
        ),
        pytest.param(
            ("date", "product", "geo"),
            ("date",),
            (slice(None), 0, 0),
            id="drop-both",
        ),
    ],
)
def test_create_index(dims, take, expected_result):
    assert create_index(dims, take) == expected_result
