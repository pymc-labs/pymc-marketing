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
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr
from pytensor import function
from sklearn.preprocessing import MaxAbsScaler

from pymc_marketing.mmm.utils import (
    _calculate_roas_distribution_for_allocation,
    _compute_quantile,
    _convert_frequency_to_timedelta,
    _covariance_matrix,
    add_noise_to_channel_allocation,
    adjusted_value_at_risk_score,
    apply_sklearn_transformer_across_dim,
    average_response,
    build_contributions,
    conditional_value_at_risk,
    create_index,
    create_new_spend_data,
    create_zero_dataset,
    mean_tightness_score,
    portfolio_entropy,
    raroc,
    sharpe_ratio,
    tail_distance,
    transform_1d_array,
    value_at_risk,
)

rng: np.random.Generator = np.random.default_rng(seed=42)

EXPECTED_RESULTS = {
    "avg_response": 5.5,
    "tail_dist": 4.5,
    "mean_tight_score": 0.591,
    "var_95": 1.45,
    "cvar_95": 1.0,
    "sharpe": 1.81327,
    "raroc_value": 0.00099891,
    "adjusted_var": 2.26,
    "entropy": 2.15181,
}


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


@pytest.fixture
def test_data():
    """
    Fixture to generate consistent test data for all tests.
    """
    samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    budgets = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    return pt.as_tensor_variable(samples), pt.as_tensor_variable(budgets)


def test_mean_tightness_score(test_data):
    samples, budgets = test_data
    result = mean_tightness_score(0.5, 0.75)(samples, budgets).eval()
    np.testing.assert_almost_equal(
        result,
        EXPECTED_RESULTS["mean_tight_score"],
        decimal=3,
        err_msg=f"Mean Tightness Score mismatch: {result} != {EXPECTED_RESULTS['mean_tight_score']}",
    )


def test_value_at_risk(test_data):
    samples, budgets = test_data
    result = value_at_risk(0.95)(samples, budgets).eval()
    np.testing.assert_almost_equal(
        result,
        EXPECTED_RESULTS["var_95"],
        decimal=3,
        err_msg=f"Value at Risk mismatch: {result} != {EXPECTED_RESULTS['var_95']}",
    )


def test_conditional_value_at_risk(test_data):
    samples, budgets = test_data
    result = conditional_value_at_risk(0.95)(samples, budgets).eval()
    np.testing.assert_almost_equal(
        result,
        EXPECTED_RESULTS["cvar_95"],
        decimal=3,
        err_msg=f"Conditional Value at Risk mismatch: {result} != {EXPECTED_RESULTS['cvar_95']}",
    )


def test_sharpe_ratio(test_data):
    samples, budgets = test_data
    result = sharpe_ratio(0.01)(samples, budgets).eval()
    np.testing.assert_almost_equal(
        result,
        EXPECTED_RESULTS["sharpe"],
        decimal=3,
        err_msg=f"Sharpe Ratio mismatch: {result} != {EXPECTED_RESULTS['sharpe']}",
    )


def test_raroc(test_data):
    samples, budgets = test_data
    result = raroc(0.01)(samples, budgets).eval()
    np.testing.assert_almost_equal(
        result,
        EXPECTED_RESULTS["raroc_value"],
        decimal=3,
        err_msg=f"RAROC mismatch: {result} != {EXPECTED_RESULTS['raroc_value']}",
    )


def test_adjusted_value_at_risk_score(test_data):
    samples, budgets = test_data
    result = adjusted_value_at_risk_score(0.95, 0.8)(samples, budgets).eval()
    np.testing.assert_almost_equal(
        result,
        EXPECTED_RESULTS["adjusted_var"],
        decimal=3,
        err_msg=f"Adjusted Value at Risk mismatch: {result} != {EXPECTED_RESULTS['adjusted_var']}",
    )


def test_portfolio_entropy(test_data):
    samples, budgets = test_data
    result = portfolio_entropy(samples, budgets).eval()
    np.testing.assert_almost_equal(
        result,
        EXPECTED_RESULTS["entropy"],
        decimal=3,
        err_msg=f"Portfolio Entropy mismatch: {result} != {EXPECTED_RESULTS['entropy']}",
    )


@pytest.mark.parametrize(
    "mean1, std1, mean2, std2, expected_order",
    [
        (
            100,
            30,
            100,
            50,
            "greater",
        ),  # Expect greater tail distance for higher std deviation
        (
            100,
            30,
            100,
            10,
            "smaller",
        ),  # Expect smaller tail distance for lower std deviation
    ],
)
def test_tail_distance(mean1, std1, mean2, std2, expected_order):
    # Generate samples for both distributions
    samples1 = pt.as_tensor(
        pm.draw(pm.Normal.dist(mu=mean1, sigma=std1, size=100), random_seed=rng)
    )
    samples2 = pt.as_tensor(
        pm.draw(pm.Normal.dist(mu=mean2, sigma=std2, size=100), random_seed=rng)
    )

    # Calculate tail distances
    tail_distance_func = tail_distance(confidence_level=0.75)
    tail_distance1 = tail_distance_func(samples1, None).eval()
    tail_distance2 = tail_distance_func(samples2, None).eval()

    # Check that the tail distance is greater for the higher std deviation
    if expected_order == "greater":
        assert tail_distance2 > tail_distance1, (
            f"Expected tail distance to be greater for std={std2}, but got {tail_distance2} <= {tail_distance1}"
        )
    elif expected_order == "smaller":
        assert tail_distance1 > tail_distance2, (
            f"Expected tail distance to be greater for std={std1}, but got {tail_distance1} <= {tail_distance2}"
        )


@pytest.mark.parametrize(
    "mean1, std1, mean2, std2, alpha, expected_relation",
    [
        (
            100,
            30,
            120,
            60,
            0.9,
            "lower_std",
        ),  # With high alpha, lower std should dominate
        (
            100,
            30,
            120,
            60,
            0.1,
            "higher_mean",
        ),  # With low alpha, lower std still dominates due to normalization
    ],
)
def test_compare_mean_tightness_score(
    mean1, std1, mean2, std2, alpha, expected_relation
):
    # Generate samples for both distributions
    samples1 = pt.as_tensor(
        pm.draw(pm.Normal.dist(mu=mean1, sigma=std1, size=100), random_seed=rng)
    )
    samples2 = pt.as_tensor(
        pm.draw(pm.Normal.dist(mu=mean2, sigma=std2, size=100), random_seed=rng)
    )

    # Calculate mean tightness scores
    mean_tightness_score_func = mean_tightness_score(alpha=alpha, confidence_level=0.75)
    score1 = mean_tightness_score_func(samples1, None).eval()
    score2 = mean_tightness_score_func(samples2, None).eval()

    # Assertions based on actual behavior of the normalized formula
    # With the normalized mean tightness score, lower std tends to dominate
    # because the score gets closer to 1 with less tail distance
    if expected_relation == "higher_mean":
        # Even with low alpha, lower std distribution scores higher due to normalization
        assert score1 > score2, (
            f"Expected score for std={std1} to be higher due to normalization, but got {score1} <= {score2}"
        )
    elif expected_relation == "lower_std":
        assert score1 > score2, (
            f"Expected score for std={std1} to be higher, but got {score1} <= {score2}"
        )


@pytest.mark.parametrize(
    "data, quantile",
    [
        ([1, 2, 3, 4, 5], 0.25),
        ([1, 2, 3, 4, 5], 0.5),
        ([1, 2, 3, 4, 5], 0.75),
        ([10, 20, 30, 40, 50], 0.1),
        ([10, 20, 30, 40, 50], 0.9),
        ([-5, -1, 0, 1, 5], 0.5),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 0.33),
        ([100], 0.5),  # Single-element edge case
        ([1, 2], 0.5),  # Small array edge case
    ],
)
def test_compute_quantile_matches_numpy(data, quantile):
    # Convert data to NumPy array
    np_data = np.array(data)

    # Define symbolic variable for input
    pt_data = pt.vector("pt_data")  # Symbolic variable for 1D input data

    # Compile the PyTensor quantile function
    pt_quantile_func = function([pt_data], _compute_quantile(pt_data, quantile))

    # Compute results
    pytensor_result = pt_quantile_func(np_data)  # Pass NumPy array here
    numpy_result = np.quantile(np_data, quantile)

    # Assert the results are close
    np.testing.assert_allclose(
        pytensor_result,
        numpy_result,
        rtol=1e-3,
        atol=1e-8,
        err_msg=f"Mismatch for data={data} and quantile={quantile}",
    )


@pytest.mark.parametrize(
    "data",
    [
        np.array([[1, 2], [3, 4], [5, 6]]),  # Small test case
        np.random.rand(100, 10),  # Random large dataset
        np.array([[1, 1], [1, 1], [1, 1]]),  # Identical columns (zero variance)
        np.array([[1], [2], [3]]),  # Single-column case
    ],
)
def test_covariance_matrix_matches_numpy(data):
    # Define symbolic variable for input
    pt_data = pt.matrix("pt_data")  # Symbolic variable for 2D input data

    # Compile the PyTensor covariance matrix function
    pt_cov_func = function([pt_data], _covariance_matrix(pt_data))

    # Compute results
    pytensor_result = pt_cov_func(data)  # Pass NumPy array directly
    numpy_result = np.cov(data, rowvar=False)

    # Assert the results are close
    np.testing.assert_allclose(
        pytensor_result,
        numpy_result,
        rtol=1e-5,
        atol=1e-8,
        err_msg=f"Mismatch for input data:\n{data}",
    )


# Test Cases
@pytest.mark.parametrize(
    "data",
    [
        np.array([10, 20, 30, 40, 50]),  # Small dataset
        pm.draw(
            pm.Normal.dist(mu=10, sigma=5, size=50), random_seed=rng
        ),  # PyMC generated samples
        np.linspace(1, 100, 100),  # Linearly spaced values
        np.array([]),  # Empty array corner case
    ],
)
def test_compute_quantile(data):
    if data.size == 0:
        with pytest.raises(Exception, match=r".*"):
            _compute_quantile(pt.as_tensor_variable(data), 0.95).eval()
    else:
        pytensor_quantile = _compute_quantile(pt.as_tensor_variable(data), 0.95).eval()
        numpy_quantile = np.quantile(data, 0.95)
        np.testing.assert_allclose(
            pytensor_quantile,
            numpy_quantile,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Quantile mismatch",
        )


@pytest.mark.parametrize(
    "samples, budgets",
    [
        (
            pm.draw(pm.Normal.dist(mu=10, sigma=2, size=100), random_seed=rng),
            pm.draw(pm.Normal.dist(mu=300, sigma=50, size=100), random_seed=rng),
        ),
        (np.array([1, 2, 3]), np.array([100, 200, 300])),  # Simple case
    ],
)
def test_roas_distribution(samples, budgets):
    pt_samples = pt.as_tensor_variable(samples)
    pt_budgets = pt.as_tensor_variable(budgets)

    pytensor_roas = _calculate_roas_distribution_for_allocation(
        pt_samples, pt_budgets
    ).eval()
    numpy_roas = samples / np.sum(budgets)
    np.testing.assert_allclose(
        pytensor_roas, numpy_roas, rtol=1e-5, atol=1e-8, err_msg="ROAS mismatch"
    )


@pytest.mark.parametrize(
    "samples, budgets, func",
    [
        (
            pm.draw(pm.Normal.dist(mu=100, sigma=20, size=100), random_seed=rng),
            pm.draw(pm.Normal.dist(mu=1000, sigma=100, size=100), random_seed=rng),
            average_response,
        ),
        (
            pm.draw(pm.Normal.dist(mu=100, sigma=20, size=100), random_seed=rng),
            pm.draw(pm.Normal.dist(mu=1000, sigma=100, size=100), random_seed=rng),
            sharpe_ratio(0.01),
        ),
        (
            pm.draw(pm.Normal.dist(mu=100, sigma=20, size=100), random_seed=rng),
            pm.draw(pm.Normal.dist(mu=1000, sigma=100, size=100), random_seed=rng),
            raroc(0.01),
        ),
        (
            pm.draw(pm.Normal.dist(mu=100, sigma=20, size=100), random_seed=rng),
            pm.draw(pm.Normal.dist(mu=1000, sigma=100, size=100), random_seed=rng),
            portfolio_entropy,
        ),
    ],
)
def test_general_functions(samples, budgets, func):
    """
    Test utility functions for general behavior.
    """
    pt_samples = pt.as_tensor_variable(samples)
    pt_budgets = pt.as_tensor_variable(budgets)

    try:
        pytensor_result = func(pt_samples, pt_budgets).eval()
        assert pytensor_result is not None, "Function returned None"
    except Exception as e:
        pytest.fail(f"Function {func.__name__} raised an unexpected exception: {e!s}")


@pytest.mark.parametrize(
    "confidence_level",
    [
        0.0,
        1.0,
    ],
)
def test_value_at_risk_invalid_confidence_level(confidence_level, test_data):
    samples, budgets = test_data
    with pytest.raises(ValueError, match=r"Confidence level must be between 0 and 1."):
        value_at_risk(confidence_level)(samples, budgets).eval()


@pytest.mark.parametrize(
    "confidence_level",
    [
        0.0,
        1.0,
    ],
)
def test_conditional_value_at_risk_invalid_confidence_level(
    confidence_level, test_data
):
    samples, budgets = test_data
    with pytest.raises(ValueError, match=r"Confidence level must be between 0 and 1."):
        conditional_value_at_risk(confidence_level)(samples, budgets).eval()


@pytest.mark.parametrize(
    "confidence_level",
    [
        0.0,
        1.0,
    ],
)
def test_tail_distance_invalid_confidence_level(confidence_level, test_data):
    samples, budgets = test_data
    with pytest.raises(ValueError, match=r"Confidence level must be between 0 and 1."):
        tail_distance(confidence_level)(samples, budgets).eval()


@pytest.mark.parametrize(
    "confidence_level",
    [
        0.0,
        1.0,
    ],
)
def test_mean_tightness_score_invalid_confidence_level(confidence_level, test_data):
    samples, budgets = test_data
    with pytest.raises(ValueError, match=r"Confidence level must be between 0 and 1."):
        mean_tightness_score(alpha=0.5, confidence_level=confidence_level)(
            samples, budgets
        ).eval()


@pytest.mark.parametrize(
    "confidence_level",
    [
        0.0,
        1.0,
    ],
)
def test_adjusted_value_at_risk_score_invalid_confidence_level(
    confidence_level, test_data
):
    samples, budgets = test_data
    with pytest.raises(ValueError, match=r"Confidence level must be between 0 and 1."):
        adjusted_value_at_risk_score(
            confidence_level=confidence_level, risk_aversion=0.8
        )(samples, budgets).eval()


@pytest.mark.parametrize(
    "risk_aversion",
    [
        -0.1,
        1.1,
    ],
)
def test_adjusted_value_at_risk_score_invalid_risk_aversion(risk_aversion, test_data):
    samples, budgets = test_data
    with pytest.raises(
        ValueError, match=r"Risk aversion parameter must be between 0 and 1."
    ):
        adjusted_value_at_risk_score(
            confidence_level=0.95, risk_aversion=risk_aversion
        )(samples, budgets).eval()
