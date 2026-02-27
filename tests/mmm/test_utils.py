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

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.preprocessing import MaxAbsScaler

from pymc_marketing.mmm.utils import (
    _convert_frequency_to_timedelta,
    add_noise_to_channel_allocation,
    apply_sklearn_transformer_across_dim,
    build_contributions,
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
        ValueError, match=r"spend_leading_up must be the same length as the spend"
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

        # Add a fake adstock object with l_max attribute
        class FakeAdstock:
            l_max = 7  # Example adstock lag

        self.adstock = FakeAdstock()


def test_create_zero_dataset():
    # Create a fake model
    model = FakeMMM()

    # Test basic zero dataset
    start_date = "2022-02-01"
    end_date = "2022-02-10"
    result = create_zero_dataset(model, start_date, end_date)

    # Check results
    assert isinstance(result, pd.DataFrame)
    # With l_max=7, the function adds 7 days to the end date
    # So we get 17 days (Feb 1 to Feb 17) * 2 regions = 34 rows
    assert len(result) == 17 * 2  # 17 days by 2 regions
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


class TestConvertFrequencyToTimedelta:
    """Test cases for _convert_frequency_to_timedelta function."""

    @pytest.mark.parametrize(
        "periods, freq, expected",
        [
            # Daily frequencies
            (1, "D", pd.Timedelta(days=1)),
            (7, "D", pd.Timedelta(days=7)),
            (0, "D", pd.Timedelta(days=0)),
            # Weekly frequencies
            (1, "W", pd.Timedelta(weeks=1)),
            (4, "W", pd.Timedelta(weeks=4)),
            (1, "W-MON", pd.Timedelta(weeks=1)),  # Complex frequency string
            (2, "W-SUN", pd.Timedelta(weeks=2)),
            # Monthly frequencies (approximated as 30 days)
            (1, "M", pd.Timedelta(days=30)),
            (3, "M", pd.Timedelta(days=90)),
            (12, "M", pd.Timedelta(days=360)),
            # Yearly frequencies (approximated as 365 days)
            (1, "Y", pd.Timedelta(days=365)),
            (2, "Y", pd.Timedelta(days=730)),
            # Hourly frequencies
            (1, "H", pd.Timedelta(hours=1)),
            (24, "H", pd.Timedelta(hours=24)),
            # Minute frequencies
            (1, "T", pd.Timedelta(minutes=1)),
            (60, "T", pd.Timedelta(minutes=60)),
            # Second frequencies
            (1, "S", pd.Timedelta(seconds=1)),
            (3600, "S", pd.Timedelta(seconds=3600)),
        ],
    )
    def test_supported_frequencies(self, periods, freq, expected):
        """Test conversion of supported frequency strings."""
        result = _convert_frequency_to_timedelta(periods, freq)
        assert result == expected

    def test_unrecognized_frequency_with_warning(self):
        """Test that unrecognized frequencies default to weeks and issue a warning."""
        with pytest.warns(
            UserWarning, match=r"Unrecognized frequency 'XYZ'. Defaulting to weeks."
        ):
            result = _convert_frequency_to_timedelta(2, "XYZ")
            expected = pd.Timedelta(weeks=2)
            assert result == expected

    def test_single_character_frequencies(self):
        """Test single character frequency strings."""
        assert _convert_frequency_to_timedelta(5, "D") == pd.Timedelta(days=5)
        assert _convert_frequency_to_timedelta(3, "W") == pd.Timedelta(weeks=3)
        assert _convert_frequency_to_timedelta(2, "M") == pd.Timedelta(days=60)

    def test_complex_frequency_strings(self):
        """Test that complex frequency strings are parsed correctly."""
        # Should extract base frequency 'W' from 'W-MON', 'W-TUE', etc.
        assert _convert_frequency_to_timedelta(1, "W-MON") == pd.Timedelta(weeks=1)
        assert _convert_frequency_to_timedelta(1, "W-TUE") == pd.Timedelta(weeks=1)
        assert _convert_frequency_to_timedelta(1, "W-WED") == pd.Timedelta(weeks=1)
        assert _convert_frequency_to_timedelta(1, "W-THU") == pd.Timedelta(weeks=1)
        assert _convert_frequency_to_timedelta(1, "W-FRI") == pd.Timedelta(weeks=1)
        assert _convert_frequency_to_timedelta(1, "W-SAT") == pd.Timedelta(weeks=1)
        assert _convert_frequency_to_timedelta(1, "W-SUN") == pd.Timedelta(weeks=1)

    def test_edge_cases(self):
        """Test edge cases like zero periods and negative periods."""
        assert _convert_frequency_to_timedelta(0, "D") == pd.Timedelta(days=0)
        assert _convert_frequency_to_timedelta(0, "W") == pd.Timedelta(weeks=0)


class TestCreateZeroDataset:
    """Extended test cases for create_zero_dataset function."""

    def test_create_zero_dataset_basic(self):
        """Test basic functionality of create_zero_dataset."""
        model = FakeMMM()
        start_date = "2022-02-01"
        end_date = "2022-02-10"

        result = create_zero_dataset(model, start_date, end_date)

        # Check basic properties
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(model.X.columns)
        assert np.all(result[model.channel_columns] == 0)
        assert np.all(result[model.control_columns] == 0)

        # Check that we have data for both regions
        assert set(result["region"].unique()) == {"A", "B"}

    def test_create_zero_dataset_with_channel_xr_dataset(self):
        """Test create_zero_dataset with channel_xr as xarray Dataset."""
        model = FakeMMM()
        start_date = "2022-02-01"
        end_date = "2022-02-10"

        # Create channel values as Dataset
        channel_values = xr.Dataset(
            data_vars={
                "channel1": (["region"], np.array([5.0, 7.0])),
                "channel2": (["region"], np.array([3.0, 4.0])),
            },
            coords={"region": np.array(["A", "B"])},
        )

        result = create_zero_dataset(model, start_date, end_date, channel_values)

        # Check channel values are set correctly
        assert np.all(result.loc[result.region == "A", "channel1"] == 5.0)
        assert np.all(result.loc[result.region == "B", "channel1"] == 7.0)
        assert np.all(result.loc[result.region == "A", "channel2"] == 3.0)
        assert np.all(result.loc[result.region == "B", "channel2"] == 4.0)

    def test_create_zero_dataset_with_channel_xr_dataarray(self):
        """Test create_zero_dataset with channel_xr as xarray DataArray."""
        model = FakeMMM()
        start_date = "2022-02-01"
        end_date = "2022-02-10"

        # Create channel values as DataArray
        channel_array = xr.DataArray(
            data=np.array([10.0, 12.0]),
            dims=["region"],
            coords={"region": np.array(["A", "B"])},
            name="channel1",
        )

        result = create_zero_dataset(model, start_date, end_date, channel_array)

        # Check channel values are set correctly
        assert np.all(result.loc[result.region == "A", "channel1"] == 10.0)
        assert np.all(result.loc[result.region == "B", "channel1"] == 12.0)
        assert np.all(result["channel2"] == 0)  # Not provided

    def test_create_zero_dataset_include_carryover_false(self):
        """Test create_zero_dataset with include_carryover=False."""
        model = FakeMMM()
        start_date = "2022-02-01"
        end_date = "2022-02-10"

        result_without_carryover = create_zero_dataset(
            model, start_date, end_date, include_carryover=False
        )
        result_with_carryover = create_zero_dataset(
            model, start_date, end_date, include_carryover=True
        )

        # Without carryover should have fewer rows (no l_max extension)
        assert len(result_without_carryover) < len(result_with_carryover)

        # Without carryover: 10 days * 2 regions = 20 rows
        assert len(result_without_carryover) == 10 * 2

        # With carryover: (10 + 7) days * 2 regions = 34 rows
        assert len(result_with_carryover) == 17 * 2

    def test_create_zero_dataset_with_timestamps(self):
        """Test create_zero_dataset with pd.Timestamp inputs."""
        model = FakeMMM()
        start_date = pd.Timestamp("2022-02-01")
        end_date = pd.Timestamp("2022-02-10")

        result = create_zero_dataset(model, start_date, end_date)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_create_zero_dataset_missing_channels_warning(self):
        """Test warning when channel_xr doesn't supply all channels."""
        model = FakeMMM()
        start_date = "2022-02-01"
        end_date = "2022-02-10"

        # Only provide values for channel1, not channel2
        channel_values = xr.Dataset(
            data_vars={"channel1": (["region"], np.array([5.0, 7.0]))},
            coords={"region": np.array(["A", "B"])},
        )

        with pytest.warns(
            UserWarning, match=r"does not supply values for \['channel2'\]"
        ):
            result = create_zero_dataset(model, start_date, end_date, channel_values)

        # channel2 should still be 0
        assert np.all(result["channel2"] == 0)

    def test_create_zero_dataset_error_cases(self):
        """Test error cases for create_zero_dataset."""
        model = FakeMMM()
        start_date = "2022-02-01"
        end_date = "2022-02-10"

        # Test invalid channel_xr type
        with pytest.raises(TypeError, match=r"must be an xarray Dataset or DataArray"):
            create_zero_dataset(model, start_date, end_date, channel_xr="invalid")

        # Test channel_xr with invalid variables
        invalid_channel_xr = xr.Dataset(
            data_vars={"invalid_channel": (["region"], np.array([5.0, 7.0]))},
            coords={"region": np.array(["A", "B"])},
        )
        with pytest.raises(ValueError, match=r"contains variables not in"):
            create_zero_dataset(model, start_date, end_date, invalid_channel_xr)

        # Test channel_xr with invalid dimensions
        invalid_dims_xr = xr.Dataset(
            data_vars={"channel1": (["invalid_dim"], np.array([5.0, 7.0]))},
            coords={"invalid_dim": np.array(["A", "B"])},
        )
        with pytest.raises(ValueError, match=r"uses dims that are not recognised"):
            create_zero_dataset(model, start_date, end_date, invalid_dims_xr)

        # Test channel_xr with date dimension (not allowed)
        date_dim_xr = xr.Dataset(
            data_vars={
                "channel1": (["region", "date"], np.array([[5.0], [7.0]])),
            },
            coords={
                "region": np.array(["A", "B"]),
                "date": pd.date_range("2022-01-01", periods=1),
            },
        )
        # The date dimension check is caught by the unrecognized dims check first
        with pytest.raises(
            ValueError, match=r"uses dims that are not recognised model dims"
        ):
            create_zero_dataset(model, start_date, end_date, date_dim_xr)

    def test_create_zero_dataset_channel_xr_includes_date_specific_error(self):
        """Ensure we hit the explicit date-dimension error when date is an allowed model dim."""

        class FakeMMM_DateDim:
            def __init__(self):
                dates = pd.date_range("2022-01-01", "2022-01-10", freq="D")
                self.X = pd.DataFrame(
                    {
                        "date": dates,
                        "channel1": np.random.rand(10) * 10,
                        "channel2": np.random.rand(10) * 5,
                    }
                )
                self.date_column = "date"
                self.channel_columns = ["channel1", "channel2"]
                self.control_columns = []
                # Include 'date' as a model dim so the invalid-dims check passes,
                # and we can assert on the specific date-dimension error.
                self.dims = ["date"]

                class FakeAdstock:
                    l_max = 1

                self.adstock = FakeAdstock()

        model = FakeMMM_DateDim()
        start_date = "2022-02-01"
        end_date = "2022-02-03"

        channel_with_date = xr.Dataset(
            data_vars={
                "channel1": ("date", np.array([1.0, 2.0])),
            },
            coords={"date": pd.date_range("2022-01-01", periods=2, freq="D")},
        )

        with pytest.raises(
            ValueError, match=r"`channel_xr` must NOT include the date dimension\."
        ):
            create_zero_dataset(model, start_date, end_date, channel_with_date)

    def test_create_zero_dataset_no_dims(self):
        """Test create_zero_dataset with a model that has no dimensions."""

        class FakeMMM_NoDims:
            def __init__(self):
                dates = pd.date_range("2022-01-01", "2022-01-10", freq="D")
                self.X = pd.DataFrame(
                    {
                        "date": dates,
                        "channel1": np.random.rand(10) * 10,
                        "channel2": np.random.rand(10) * 5,
                    }
                )
                self.date_column = "date"
                self.channel_columns = ["channel1", "channel2"]
                self.control_columns = []
                self.dims = []  # No dimensions

                class FakeAdstock:
                    l_max = 3

                self.adstock = FakeAdstock()

        model = FakeMMM_NoDims()
        start_date = "2022-02-01"
        end_date = "2022-02-05"

        result = create_zero_dataset(model, start_date, end_date)

        # Should have (5 + 3) days = 8 rows (no cross-join with dimensions)
        assert len(result) == 8
        assert "region" not in result.columns

    def test_create_zero_dataset_empty_date_range_error(self):
        """Test error when generated date range is empty."""
        model = FakeMMM()
        # Invalid date range (end before start)
        start_date = "2022-02-10"
        end_date = "2022-02-01"

        with pytest.raises(ValueError, match=r"Generated date range is empty"):
            create_zero_dataset(model, start_date, end_date)

    def test_create_zero_dataset_channel_xr_no_dims_all_channels(self):
        """Channel-only allocation: channel_xr is a 0-dim Dataset with per-channel scalars."""

        class FakeMMM_NoDims:
            def __init__(self):
                dates = pd.date_range("2022-01-01", "2022-01-10", freq="D")
                self.X = pd.DataFrame(
                    {
                        "date": dates,
                        "channel1": np.random.rand(10) * 10,
                        "channel2": np.random.rand(10) * 5,
                    }
                )
                self.date_column = "date"
                self.channel_columns = ["channel1", "channel2"]
                self.control_columns = []
                self.dims = []  # No dimensions

                class FakeAdstock:
                    l_max = 3

                self.adstock = FakeAdstock()

        model = FakeMMM_NoDims()
        start_date = "2022-02-01"
        end_date = "2022-02-05"

        # 0-dim Dataset: variables are channels with scalar values
        channel_values = xr.Dataset(
            data_vars={
                "channel1": 100.0,
                "channel2": 200.0,
            }
        )

        result = create_zero_dataset(model, start_date, end_date, channel_values)

        # (5 + 3) days = 8 rows
        assert len(result) == 8
        assert np.all(result["channel1"] == 100.0)
        assert np.all(result["channel2"] == 200.0)

    def test_create_zero_dataset_channel_xr_no_dims_missing_channel(self):
        """Channel-only allocation with missing channel var should warn and leave others at 0."""

        class FakeMMM_NoDims:
            def __init__(self):
                dates = pd.date_range("2022-01-01", "2022-01-10", freq="D")
                self.X = pd.DataFrame(
                    {
                        "date": dates,
                        "channel1": np.random.rand(10) * 10,
                        "channel2": np.random.rand(10) * 5,
                    }
                )
                self.date_column = "date"
                self.channel_columns = ["channel1", "channel2"]
                self.control_columns = []
                self.dims = []

                class FakeAdstock:
                    l_max = 2

                self.adstock = FakeAdstock()

        model = FakeMMM_NoDims()
        start_date = "2022-02-01"
        end_date = "2022-02-03"

        # Provide only one channel as scalar variable in 0-dim Dataset
        channel_values = xr.Dataset(
            data_vars={
                "channel1": 50.0,
            }
        )

        with pytest.warns(
            UserWarning, match=r"does not supply values for \['channel2'\]"
        ):
            result = create_zero_dataset(model, start_date, end_date, channel_values)

        # (3 + 2) days = 5 rows
        assert len(result) == 5
        assert np.all(result["channel1"] == 50.0)
        assert np.all(result["channel2"] == 0.0)


class TestBuildContributions:
    """Test cases for build_contributions function."""

    @pytest.fixture
    def mock_idata_simple(self):
        """Create simple InferenceData for testing build_contributions."""
        dates = pd.date_range("2025-01-01", periods=10, freq="W-MON")
        channels = ["C1", "C2"]

        posterior = xr.Dataset(
            {
                "intercept_contribution": xr.DataArray(
                    np.random.normal(size=(2, 50, 10)),
                    dims=("chain", "draw", "date"),
                    coords={
                        "chain": [0, 1],
                        "draw": np.arange(50),
                        "date": dates,
                    },
                ),
                "channel_contribution": xr.DataArray(
                    np.random.normal(size=(2, 50, 10, 2)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={
                        "chain": [0, 1],
                        "draw": np.arange(50),
                        "date": dates,
                        "channel": channels,
                    },
                ),
            }
        )
        idata = az.InferenceData(posterior=posterior)
        return idata

    @pytest.fixture
    def mock_idata_multidim(self):
        """Create InferenceData with multiple dimensions."""
        dates = pd.date_range("2025-01-01", periods=5, freq="W-MON")
        channels = ["C1", "C2"]
        geos = ["US", "UK"]

        posterior = xr.Dataset(
            {
                "intercept_contribution": xr.DataArray(
                    np.random.normal(size=(2, 30, 5, 2)),
                    dims=("chain", "draw", "date", "geo"),
                    coords={
                        "chain": [0, 1],
                        "draw": np.arange(30),
                        "date": dates,
                        "geo": geos,
                    },
                ),
                "channel_contribution": xr.DataArray(
                    np.random.normal(size=(2, 30, 5, 2, 2)),
                    dims=("chain", "draw", "date", "channel", "geo"),
                    coords={
                        "chain": [0, 1],
                        "draw": np.arange(30),
                        "date": dates,
                        "channel": channels,
                        "geo": geos,
                    },
                ),
            }
        )
        idata = az.InferenceData(posterior=posterior)
        return idata

    def test_build_contributions_basic(self, mock_idata_simple):
        """Test basic functionality of build_contributions."""
        df = build_contributions(
            idata=mock_idata_simple,
            var=["intercept_contribution", "channel_contribution"],
            agg="mean",
        )

        # Check it returns a DataFrame
        assert isinstance(df, pd.DataFrame)

        # Should have date column
        assert "date" in df.columns

        # Should have expanded channel columns
        assert "channel__C1" in df.columns
        assert "channel__C2" in df.columns

        # Should have intercept column (renamed from intercept_contribution)
        assert "intercept" in df.columns

        # Check that we have 10 rows (one per date)
        assert len(df) == 10

    def test_build_contributions_with_median(self, mock_idata_simple):
        """Test build_contributions with median aggregation."""
        df = build_contributions(
            idata=mock_idata_simple,
            var=["intercept_contribution"],
            agg="median",
        )

        assert isinstance(df, pd.DataFrame)
        assert "intercept" in df.columns
        assert len(df) == 10

    def test_build_contributions_multidimensional(self, mock_idata_multidim):
        """Test build_contributions with multiple dimensions."""
        df = build_contributions(
            idata=mock_idata_multidim,
            var=["intercept_contribution", "channel_contribution"],
            agg="mean",
        )

        # Should have both date and geo columns
        assert "date" in df.columns
        assert "geo" in df.columns

        # Should have expanded channel columns
        assert "channel__C1" in df.columns
        assert "channel__C2" in df.columns

        # Should have intercept
        assert "intercept" in df.columns

        # Check that we have 10 rows (5 dates * 2 geos)
        assert len(df) == 10

        # Check geo is categorical
        assert df["geo"].dtype.name == "category"

    def test_build_contributions_custom_dims(self, mock_idata_simple):
        """Test build_contributions with custom dimension parameters."""
        df = build_contributions(
            idata=mock_idata_simple,
            var=["channel_contribution"],
            agg="mean",
            agg_dims=("chain", "draw"),
            index_dims=("date",),
            expand_dims=("channel",),
        )

        assert isinstance(df, pd.DataFrame)
        assert "channel__C1" in df.columns
        assert "channel__C2" in df.columns

    def test_build_contributions_missing_variables(self, mock_idata_simple):
        """Test that build_contributions raises error for missing variables."""
        with pytest.raises(
            ValueError,
            match=r"None of the requested variables .* are present in idata.posterior",
        ):
            build_contributions(
                idata=mock_idata_simple,
                var=["nonexistent_variable"],
                agg="mean",
            )

    def test_build_contributions_partial_variables(self, mock_idata_simple):
        """Test build_contributions with some valid and some invalid variables."""
        # Should work with only the valid variable
        df = build_contributions(
            idata=mock_idata_simple,
            var=["intercept_contribution", "nonexistent"],
            agg="mean",
        )

        # Should have the intercept column
        assert "intercept" in df.columns
        assert len(df) == 10

    def test_build_contributions_no_category_cast(self, mock_idata_multidim):
        """Test build_contributions without casting to category."""
        df = build_contributions(
            idata=mock_idata_multidim,
            var=["intercept_contribution"],
            agg="mean",
            cast_regular_to_category=False,
        )

        # Geo should not be categorical
        assert df["geo"].dtype.name != "category"

    def test_build_contributions_custom_aggregation(self, mock_idata_simple):
        """Test build_contributions with a custom aggregation function."""

        def custom_agg(data, axis):
            # Custom aggregation: 75th percentile
            return np.quantile(data, 0.75, axis=axis)

        df = build_contributions(
            idata=mock_idata_simple,
            var=["intercept_contribution"],
            agg=custom_agg,
        )

        assert isinstance(df, pd.DataFrame)
        assert "intercept" in df.columns
        assert len(df) == 10
