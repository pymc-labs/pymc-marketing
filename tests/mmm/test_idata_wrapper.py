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
"""Tests for InferenceData wrapper and utility functions."""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper
from pymc_marketing.data.idata.schema import MMMIdataSchema
from pymc_marketing.data.idata.utils import (
    aggregate_idata_dims,
    aggregate_idata_time,
    filter_idata_by_dates,
    filter_idata_by_dims,
)

# Seed for reproducibility
SEED = sum(map(ord, "idata_wrapper_tests"))
rng = np.random.default_rng(seed=SEED)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def multidim_idata() -> az.InferenceData:
    """Create InferenceData with custom dimensions (country)."""
    dates = pd.date_range("2024-01-01", periods=52, freq="W")
    channels = ["TV", "Radio", "Facebook", "Instagram", "Social"]
    countries = ["US", "UK", "DE"]

    return az.InferenceData(
        constant_data=xr.Dataset(
            {
                "channel_data": xr.DataArray(
                    rng.uniform(0, 100, size=(52, 3, 5)),
                    dims=("date", "country", "channel"),
                    coords={"date": dates, "country": countries, "channel": channels},
                ),
                "target_data": xr.DataArray(
                    rng.uniform(100, 1000, size=(52, 3)),
                    dims=("date", "country"),
                    coords={"date": dates, "country": countries},
                ),
                "channel_scale": xr.DataArray(
                    rng.uniform(50, 150, size=(3, 5)),
                    dims=("country", "channel"),
                    coords={"country": countries, "channel": channels},
                ),
                "target_scale": xr.DataArray(
                    [500.0, 550.0, 480.0],
                    dims=("country",),
                    coords={"country": countries},
                ),
                "dayofyear": xr.DataArray(
                    np.arange(1, len(dates) + 1),  # Day of year values
                    dims=("date",),
                    coords={"date": dates},
                ),
            }
        ),
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(size=(2, 10, 52, 3, 5)),
                    dims=("chain", "draw", "date", "country", "channel"),
                    coords={"date": dates, "country": countries, "channel": channels},
                ),
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 52, 3)),
                    dims=("chain", "draw", "date", "country"),
                    coords={"date": dates, "country": countries},
                ),
            }
        ),
        fit_data=xr.Dataset(
            {
                "TV": xr.DataArray(
                    rng.uniform(0, 100, size=(52, 3)),
                    dims=("date", "country"),
                    coords={"date": dates, "country": countries},
                ),
                "Radio": xr.DataArray(
                    rng.uniform(0, 100, size=(52, 3)),
                    dims=("date", "country"),
                    coords={"date": dates, "country": countries},
                ),
                "target": xr.DataArray(
                    rng.uniform(100, 1000, size=(52, 3)),
                    dims=("date", "country"),
                    coords={"date": dates, "country": countries},
                ),
            }
        ),
    )


@pytest.fixture(scope="module")
def idata_with_original_scale(multidim_idata) -> az.InferenceData:
    """Create InferenceData with _original_scale variables already computed."""
    idata = multidim_idata.copy()

    # Add _original_scale variable
    channel_contrib_orig = (
        idata.posterior.channel_contribution * idata.constant_data.target_scale
    )
    idata.posterior["channel_contribution_original_scale"] = channel_contrib_orig

    return idata


@pytest.fixture(scope="module")
def idata_with_all_contributions() -> az.InferenceData:
    """Create InferenceData with all contribution types (baseline, controls, seasonality) and custom dims."""
    dates = pd.date_range("2024-01-01", periods=52, freq="W")
    channels = ["TV", "Radio"]
    controls = ["price", "promotion"]
    countries = ["US", "UK", "DE"]

    return az.InferenceData(
        constant_data=xr.Dataset(
            {
                "channel_data": xr.DataArray(
                    rng.uniform(0, 100, size=(52, 3, 2)),
                    dims=("date", "country", "channel"),
                    coords={"date": dates, "country": countries, "channel": channels},
                ),
                "target_data": xr.DataArray(
                    rng.uniform(100, 1000, size=(52, 3)),
                    dims=("date", "country"),
                    coords={"date": dates, "country": countries},
                ),
                "control_data_": xr.DataArray(
                    rng.uniform(0, 10, size=(52, 3, 2)),
                    dims=("date", "country", "control"),
                    coords={"date": dates, "country": countries, "control": controls},
                ),
                "channel_scale": xr.DataArray(
                    rng.uniform(50, 150, size=(3, 2)),
                    dims=("country", "channel"),
                    coords={"country": countries, "channel": channels},
                ),
                "target_scale": xr.DataArray(
                    [500.0, 550.0, 480.0],
                    dims=("country",),
                    coords={"country": countries},
                ),
            }
        ),
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(size=(2, 10, 52, 3, 2)),
                    dims=("chain", "draw", "date", "country", "channel"),
                    coords={"date": dates, "country": countries, "channel": channels},
                ),
                "intercept_contribution": xr.DataArray(
                    rng.normal(size=(2, 10, 52, 3)),
                    dims=("chain", "draw", "date", "country"),
                    coords={"date": dates, "country": countries},
                ),
                "control_contribution": xr.DataArray(
                    rng.normal(size=(2, 10, 52, 3, 2)),
                    dims=("chain", "draw", "date", "country", "control"),
                    coords={"date": dates, "country": countries, "control": controls},
                ),
                "yearly_seasonality_contribution": xr.DataArray(
                    rng.normal(size=(2, 10, 52, 3)),
                    dims=("chain", "draw", "date", "country"),
                    coords={"date": dates, "country": countries},
                ),
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 52, 3)),
                    dims=("chain", "draw", "date", "country"),
                    coords={"date": dates, "country": countries},
                ),
            }
        ),
        fit_data=xr.Dataset(
            {
                "TV": xr.DataArray(
                    rng.uniform(0, 100, size=(52, 3)),
                    dims=("date", "country"),
                    coords={"date": dates, "country": countries},
                ),
                "Radio": xr.DataArray(
                    rng.uniform(0, 100, size=(52, 3)),
                    dims=("date", "country"),
                    coords={"date": dates, "country": countries},
                ),
                "target": xr.DataArray(
                    rng.uniform(100, 1000, size=(52, 3)),
                    dims=("date", "country"),
                    coords={"date": dates, "country": countries},
                ),
            }
        ),
    )


@pytest.fixture
def basic_schema():
    """Create basic MMMIdataSchema for testing."""
    return MMMIdataSchema.from_model_config(
        custom_dims=(),
        has_controls=False,
        has_seasonality=False,
        time_varying=False,
    )


@pytest.fixture
def fitted_mmm(multidim_idata):
    """Create mock fitted MMM object for integration tests with custom dims."""

    # Mock MMM object with .data property that returns wrapper
    class MockMMM:
        def __init__(self, idata):
            self.idata = idata
            self.dims = ("country",)
            self.control_columns = None
            self.yearly_seasonality = None
            self.time_varying_intercept = False
            self.time_varying_media = False

        @property
        def data(self):
            """Get data wrapper for InferenceData access and manipulation.

            Returns a fresh wrapper on each access.
            """
            from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper
            from pymc_marketing.data.idata.schema import MMMIdataSchema

            schema = MMMIdataSchema.from_model_config(
                custom_dims=self.dims if hasattr(self, "dims") and self.dims else (),
                has_controls=self.control_columns is not None,
                has_seasonality=self.yearly_seasonality is not None,
                time_varying=(
                    getattr(self, "time_varying_intercept", False)
                    or getattr(self, "time_varying_media", False)
                ),
            )
            return MMMIDataWrapper(self.idata, schema=schema, validate_on_init=False)

    return MockMMM(multidim_idata)


@pytest.fixture
def fitted_mmm_with_controls(idata_with_all_contributions):
    """Create mock fitted MMM with controls and custom dims for integration tests."""

    class MockMMM:
        def __init__(self, idata):
            self.idata = idata
            self.dims = ("country",)
            self.control_columns = ["price", "promotion"]
            self.yearly_seasonality = True
            self.time_varying_intercept = False
            self.time_varying_media = False

        @property
        def data(self):
            """Get data wrapper for InferenceData access and manipulation.

            Returns a fresh wrapper on each access.
            """
            from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper
            from pymc_marketing.data.idata.schema import MMMIdataSchema

            schema = MMMIdataSchema.from_model_config(
                custom_dims=self.dims if hasattr(self, "dims") and self.dims else (),
                has_controls=self.control_columns is not None,
                has_seasonality=self.yearly_seasonality is not None,
                time_varying=(
                    getattr(self, "time_varying_intercept", False)
                    or getattr(self, "time_varying_media", False)
                ),
            )
            return MMMIDataWrapper(self.idata, schema=schema, validate_on_init=False)

    return MockMMM(idata_with_all_contributions)


# ============================================================================
# Category 1: Utility Function Tests - Date Filtering
# ============================================================================


def test_filter_idata_by_dates_filters_all_groups(multidim_idata):
    """Test that filter_idata_by_dates filters all groups with date dimension."""
    # Arrange
    start_date = "2024-03-01"
    end_date = "2024-06-30"

    # Act
    filtered = filter_idata_by_dates(multidim_idata, start_date, end_date)

    # Assert - Verify all groups with date dimension were filtered
    assert hasattr(filtered, "posterior")
    assert hasattr(filtered, "constant_data")

    # Check date ranges
    filtered_dates = filtered.posterior.coords["date"].values
    assert pd.Timestamp(filtered_dates[0]) >= pd.Timestamp(start_date)
    assert pd.Timestamp(filtered_dates[-1]) <= pd.Timestamp(end_date)

    # Verify other dimensions unchanged
    xr.testing.assert_equal(
        filtered.posterior.coords["channel"], multidim_idata.posterior.coords["channel"]
    )


def test_filter_idata_by_dates_returns_new_instance(multidim_idata):
    """Test that filtering returns new InferenceData without modifying original."""
    # Arrange
    # Store original size
    original_size = multidim_idata.posterior.sizes["date"]
    original_dates = multidim_idata.posterior.coords["date"].values.copy()

    # Act
    filtered = filter_idata_by_dates(multidim_idata, "2024-03-01", "2024-06-30")

    # Assert - Original unchanged
    assert multidim_idata.posterior.sizes["date"] == original_size
    np.testing.assert_array_equal(
        multidim_idata.posterior.coords["date"].values, original_dates
    )

    # Assert - Filtered is different
    assert filtered is not multidim_idata
    assert filtered.posterior.sizes["date"] < original_size


def test_filter_idata_by_dates_with_none_returns_original(multidim_idata):
    """Test that None dates return original InferenceData."""
    # Arrange & Act
    result = filter_idata_by_dates(multidim_idata, None, None)

    # Assert - Same object returned (no filtering needed)
    assert result is multidim_idata


def test_filter_idata_by_dates_with_only_start_date(multidim_idata):
    """Test filtering with only start_date specified."""
    # Arrange & Act
    start_date = "2024-03-01"
    filtered = filter_idata_by_dates(multidim_idata, start_date=start_date)

    # Assert
    filtered_dates = filtered.posterior.coords["date"].values
    assert pd.Timestamp(filtered_dates[0]) >= pd.Timestamp(start_date)

    # Last date should be same as original (no end filter)
    original_last = multidim_idata.posterior.coords["date"].values[-1]
    assert pd.Timestamp(filtered_dates[-1]) == pd.Timestamp(original_last)


# ============================================================================
# Category 2: Utility Function Tests - Dimension Filtering
# ============================================================================


def test_filter_idata_by_dims_filters_single_channel(multidim_idata):
    """Test filtering to a single channel."""
    # Arrange & Act
    filtered = filter_idata_by_dims(multidim_idata, channel="TV")

    # Assert - Dimension should be dropped when filtering to single value
    assert "channel" not in filtered.posterior.dims

    # Other dimensions unchanged
    assert filtered.posterior.sizes["date"] == multidim_idata.posterior.sizes["date"]


def test_filter_idata_by_dims_filters_multiple_channels(multidim_idata):
    """Test filtering to multiple channels."""
    # Arrange & Act
    channels = ["TV", "Radio"]
    filtered = filter_idata_by_dims(multidim_idata, channel=channels)

    # Assert
    assert filtered.posterior.sizes["channel"] == 2
    assert list(filtered.posterior.coords["channel"].values) == channels


def test_filter_idata_by_dims_returns_new_instance(multidim_idata):
    """Test that dimension filtering returns new InferenceData."""
    # Arrange
    original_channels = multidim_idata.posterior.coords["channel"].values.copy()

    # Act
    filtered = filter_idata_by_dims(multidim_idata, channel="TV")

    # Assert - Original unchanged
    np.testing.assert_array_equal(
        multidim_idata.posterior.coords["channel"].values, original_channels
    )

    # Assert - Filtered is different
    assert filtered is not multidim_idata


def test_filter_idata_by_dims_with_custom_dimension(multidim_idata):
    """Test filtering by custom dimension like country."""
    # Arrange & Act
    filtered = filter_idata_by_dims(multidim_idata, country="US")

    # Assert - Dimension should be dropped when filtering to single value
    assert "country" not in filtered.posterior.dims


def test_filter_idata_by_dims_with_empty_kwargs_returns_original(multidim_idata):
    """Test that empty dim_filters returns original InferenceData."""
    # Arrange & Act
    result = filter_idata_by_dims(multidim_idata)

    # Assert - Same object returned
    assert result is multidim_idata


def test_filter_idata_by_dims_raises_on_nonexistent_dimension(multidim_idata):
    """Test that filtering by nonexistent dimension raises ValueError."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="Dimension 'nonexistent_dim' not found"):
        filter_idata_by_dims(multidim_idata, nonexistent_dim="value")


# ============================================================================
# Category 3: Utility Function Tests - Time Aggregation
# ============================================================================


def test_aggregate_idata_time_monthly_sum(multidim_idata):
    """Test monthly aggregation with sum method."""
    # Arrange & Act
    monthly = aggregate_idata_time(multidim_idata, period="monthly", method="sum")

    # Assert - Should have fewer dates (52 weeks -> ~12 months)
    assert monthly.posterior.sizes["date"] < multidim_idata.posterior.sizes["date"]
    assert monthly.posterior.sizes["date"] <= 12

    # Assert - Other dimensions unchanged
    assert (
        monthly.posterior.sizes["channel"] == multidim_idata.posterior.sizes["channel"]
    )

    # Assert - Coordinates preserved (except date is aggregated)
    xr.testing.assert_equal(
        monthly.posterior.coords["channel"], multidim_idata.posterior.coords["channel"]
    )


def test_aggregate_idata_time_monthly_mean(multidim_idata):
    """Test monthly aggregation with mean method."""
    # Arrange & Act
    monthly = aggregate_idata_time(multidim_idata, period="monthly", method="mean")

    # Assert - Aggregated to monthly
    assert monthly.posterior.sizes["date"] <= 12


def test_aggregate_idata_time_all_time_removes_date_dim(multidim_idata):
    """Test that all_time aggregation removes date dimension."""
    # Arrange & Act
    total = aggregate_idata_time(multidim_idata, period="all_time", method="sum")

    # Assert - Date dimension removed
    assert "date" not in total.posterior.dims

    # Assert - Other dimensions preserved
    assert "channel" in total.posterior.dims
    assert total.posterior.sizes["channel"] == multidim_idata.posterior.sizes["channel"]


def test_aggregate_idata_time_returns_new_instance(multidim_idata):
    """Test that aggregation returns new InferenceData."""
    # Arrange
    original_size = multidim_idata.posterior.sizes["date"]

    # Act
    aggregated = aggregate_idata_time(multidim_idata, period="monthly")

    # Assert - Original unchanged
    assert multidim_idata.posterior.sizes["date"] == original_size

    # Assert - Aggregated is different
    assert aggregated is not multidim_idata


@pytest.mark.parametrize(
    "period,expected_max_size",
    [
        ("weekly", 52),
        ("monthly", 12),
        ("quarterly", 4),
        ("yearly", 1),
    ],
    ids=["weekly", "monthly", "quarterly", "yearly"],
)
def test_aggregate_idata_time_parametrized_periods(
    multidim_idata, period, expected_max_size
):
    """Test aggregation with different time periods."""
    # Arrange & Act
    aggregated = aggregate_idata_time(multidim_idata, period=period, method="sum")

    # Assert
    assert aggregated.posterior.sizes["date"] <= expected_max_size


# ============================================================================
# Category 4: Utility Function Tests - Dimension Aggregation
# ============================================================================


def test_aggregate_idata_dims_combines_channels(multidim_idata):
    """Test combining multiple channels into one."""
    # Arrange & Act
    # Note: Aggregating Facebook, Instagram, Social into "SocialMedia"
    # (We can't use "Social" as new_label since it already exists as a channel)
    combined = aggregate_idata_dims(
        multidim_idata,
        dim="channel",
        values=["Facebook", "Instagram", "Social"],
        new_label="SocialMedia",
        method="sum",
    )

    # Assert - SocialMedia channel added
    assert "SocialMedia" in combined.posterior.coords["channel"].values

    # Assert - Original channels removed
    assert "Facebook" not in combined.posterior.coords["channel"].values
    assert "Instagram" not in combined.posterior.coords["channel"].values
    assert "Social" not in combined.posterior.coords["channel"].values

    # Assert - Other channels preserved
    assert "TV" in combined.posterior.coords["channel"].values


def test_aggregate_idata_dims_with_mean_method(multidim_idata):
    """Test dimension aggregation with mean method."""
    # Arrange & Act
    combined = aggregate_idata_dims(
        multidim_idata,
        dim="channel",
        values=["TV", "Radio"],
        new_label="Traditional",
        method="mean",
    )

    # Assert
    assert "Traditional" in combined.posterior.coords["channel"].values


def test_aggregate_idata_dims_returns_new_instance(multidim_idata):
    """Test that dimension aggregation returns new InferenceData."""
    # Arrange
    original_channels = multidim_idata.posterior.coords["channel"].values.copy()

    # Act
    combined = aggregate_idata_dims(
        multidim_idata,
        dim="channel",
        values=["TV", "Radio"],
        new_label="Traditional",
        method="sum",
    )

    # Assert - Original unchanged
    np.testing.assert_array_equal(
        multidim_idata.posterior.coords["channel"].values, original_channels
    )

    # Assert - Combined is different
    assert combined is not multidim_idata


def test_aggregate_idata_dims_raises_on_nonexistent_dimension(multidim_idata):
    """Test that aggregating nonexistent dimension raises ValueError."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="Dimension 'nonexistent_dim' not found"):
        aggregate_idata_dims(
            multidim_idata,
            dim="nonexistent_dim",
            values=["a", "b"],
            new_label="combined",
            method="sum",
        )


def test_aggregate_idata_dims_raises_on_conflicting_new_label(multidim_idata):
    """Test that new_label conflicting with existing coordinate raises ValueError.

    When aggregating Facebook and Instagram with new_label="TV", but TV already
    exists as a channel that isn't being aggregated, this should raise an error
    to prevent duplicate coordinate labels.
    """
    # Arrange & Act & Assert
    with pytest.raises(
        ValueError, match="new_label 'TV' conflicts with existing coordinate value"
    ):
        aggregate_idata_dims(
            multidim_idata,
            dim="channel",
            values=["Facebook", "Instagram"],
            new_label="TV",  # TV already exists as a channel!
            method="sum",
        )


# ============================================================================
# Category 5: Wrapper Initialization Tests
# ============================================================================


def test_wrapper_init_without_schema(multidim_idata):
    """Test creating wrapper without schema validation."""
    # Arrange & Act
    wrapper = MMMIDataWrapper(multidim_idata, schema=None, validate_on_init=False)

    # Assert
    assert wrapper.idata is multidim_idata
    assert wrapper.schema is None


def test_wrapper_init_with_schema_validates(multidim_idata):
    """Test that wrapper validates on init when schema provided."""
    # Arrange - Create schema that matches multidim_idata (has custom dims)
    schema = MMMIdataSchema.from_model_config(
        custom_dims=("country",),
        has_controls=False,
        has_seasonality=False,
        time_varying=False,
    )

    # Act
    wrapper = MMMIDataWrapper(multidim_idata, schema=schema, validate_on_init=True)

    # Assert - No exception raised (validation passed)
    assert wrapper.schema is schema


def test_wrapper_init_caches_instance(multidim_idata):
    """Test that wrapper has cache for expensive operations."""
    # Arrange & Act
    wrapper = MMMIDataWrapper(multidim_idata)

    # Assert
    assert hasattr(wrapper, "_cache")
    assert isinstance(wrapper._cache, dict)


# ============================================================================
# Category 6: Data Access Method Tests
# ============================================================================


def test_get_target_returns_target_data(multidim_idata):
    """Test that get_target returns observed target values."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    target = wrapper.get_target(original_scale=True)

    # Assert
    assert isinstance(target, xr.DataArray)
    assert "date" in target.dims
    assert target.sizes["date"] == multidim_idata.constant_data.sizes["date"]

    # Should match constant_data.target_data
    xr.testing.assert_equal(target, multidim_idata.constant_data.target_data)


def test_get_target_scaled_divides_by_target_scale(multidim_idata):
    """Test that get_target with original_scale=False returns scaled values."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    target_scaled = wrapper.get_target(original_scale=False)
    target_original = wrapper.get_target(original_scale=True)
    target_scale = multidim_idata.constant_data.target_scale

    # Assert - Scaled version should equal original / scale
    expected_scaled = target_original / target_scale
    xr.testing.assert_allclose(target_scaled, expected_scaled)


def test_get_channel_spend_returns_channel_data(multidim_idata):
    """Test that get_channel_spend returns channel spend data."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    channel_spend = wrapper.get_channel_spend()

    # Assert
    assert isinstance(channel_spend, xr.DataArray)
    assert "date" in channel_spend.dims
    assert "channel" in channel_spend.dims

    # Should match constant_data.channel_data
    xr.testing.assert_equal(channel_spend, multidim_idata.constant_data.channel_data)


def test_get_channel_contributions_returns_dataarray(multidim_idata):
    """Test that get_channel_contributions returns DataArray with correct dims."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    channel_contrib = wrapper.get_channel_contributions(original_scale=True)

    # Assert
    assert isinstance(channel_contrib, xr.DataArray)
    assert "chain" in channel_contrib.dims
    assert "draw" in channel_contrib.dims
    assert "date" in channel_contrib.dims
    assert "channel" in channel_contrib.dims


def test_get_channel_contributions_with_original_scale(multidim_idata):
    """Test that get_channel_contributions computes original scale correctly."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    channel_contrib_original = wrapper.get_channel_contributions(original_scale=True)
    channel_contrib_scaled = wrapper.get_channel_contributions(original_scale=False)

    # Assert - Original scale should equal scaled * target_scale
    target_scale = multidim_idata.constant_data.target_scale
    # Align coordinates before multiplication to ensure coordinate consistency
    channel_contrib_scaled_aligned, target_scale_aligned = xr.align(
        channel_contrib_scaled, target_scale, join="exact"
    )
    expected_original = channel_contrib_scaled_aligned * target_scale_aligned
    # Align both arrays before comparison to ensure coordinate consistency
    channel_contrib_original_aligned, expected_original_aligned = xr.align(
        channel_contrib_original, expected_original, join="exact"
    )
    # Compare values using numpy since xarray's assert_allclose is strict about
    # coordinate object equality, even when values match
    np.testing.assert_allclose(
        channel_contrib_original_aligned.values,
        expected_original_aligned.values,
    )


def test_get_channel_contributions_uses_existing_original_scale_variable(
    idata_with_original_scale,
):
    """Test that get_channel_contributions uses existing _original_scale variable."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_with_original_scale)

    # Act
    channel_contrib = wrapper.get_channel_contributions(original_scale=True)

    # Assert - Should return existing variable directly
    expected = idata_with_original_scale.posterior.channel_contribution_original_scale
    # Align both arrays before comparison to ensure coordinate consistency
    channel_contrib_aligned, expected_aligned = xr.align(
        channel_contrib, expected, join="exact"
    )
    # Compare values using numpy since xarray's assert_equal is strict about
    # coordinate object equality, even when values match
    np.testing.assert_array_equal(
        channel_contrib_aligned.values,
        expected_aligned.values,
    )


def test_get_contributions_returns_dataset(multidim_idata):
    """Test that get_contributions returns Dataset with contribution variables."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    contributions = wrapper.get_contributions(original_scale=True)

    # Assert
    assert isinstance(contributions, xr.Dataset)
    assert "channels" in contributions

    # Should have channel contribution variable
    assert "date" in contributions["channels"].dims
    assert "channel" in contributions["channels"].dims


def test_get_contributions_uses_original_scale_variable_if_exists(
    idata_with_original_scale,
):
    """Test that get_contributions uses existing _original_scale variable."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_with_original_scale)

    # Act
    contributions = wrapper.get_contributions(original_scale=True)

    # Assert - Should use existing variable
    expected = idata_with_original_scale.posterior.channel_contribution_original_scale
    # Align both arrays before comparison to ensure coordinate consistency
    contributions_channel_aligned, expected_aligned = xr.align(
        contributions["channels"], expected, join="exact"
    )
    # Compare values using numpy since xarray's assert_equal is strict about
    # coordinate object equality, even when values match
    np.testing.assert_array_equal(
        contributions_channel_aligned.values,
        expected_aligned.values,
    )


def test_get_contributions_computes_original_scale_on_the_fly(multidim_idata):
    """Test that get_contributions computes original scale if needed."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    contributions = wrapper.get_contributions(original_scale=True)

    # Assert - Should compute on-the-fly
    channel_contrib = multidim_idata.posterior.channel_contribution
    target_scale = multidim_idata.constant_data.target_scale
    # Align coordinates before multiplication to ensure coordinate consistency
    channel_contrib_aligned, target_scale_aligned = xr.align(
        channel_contrib, target_scale, join="exact"
    )
    expected = channel_contrib_aligned * target_scale_aligned
    # Align both arrays before comparison to ensure coordinate consistency
    contributions_channel_aligned, expected_aligned = xr.align(
        contributions["channels"], expected, join="exact"
    )
    # Compare values using numpy since xarray's assert_allclose is strict about
    # coordinate object equality, even when values match
    np.testing.assert_allclose(
        contributions_channel_aligned.values,
        expected_aligned.values,
    )


@pytest.mark.parametrize(
    "include_baseline,include_controls,include_seasonality",
    [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ],
    ids=["all", "no_baseline", "no_controls", "no_seasonality"],
)
def test_get_contributions_with_options(
    idata_with_all_contributions,
    include_baseline,
    include_controls,
    include_seasonality,
):
    """Test that get_contributions respects include options."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_with_all_contributions)

    # Act
    contributions = wrapper.get_contributions(
        original_scale=True,
        include_baseline=include_baseline,
        include_controls=include_controls,
        include_seasonality=include_seasonality,
    )

    # Assert
    assert isinstance(contributions, xr.Dataset)
    assert "channels" in contributions.data_vars  # Always included

    if include_baseline:
        assert "baseline" in contributions.data_vars
    else:
        assert "baseline" not in contributions.data_vars

    if include_controls:
        assert "controls" in contributions.data_vars
    else:
        assert "controls" not in contributions.data_vars

    if include_seasonality:
        assert "seasonality" in contributions.data_vars
    else:
        assert "seasonality" not in contributions.data_vars


# ============================================================================
# Category 7: Scaling Operation Tests
# ============================================================================


def test_to_original_scale_with_string_variable_name(multidim_idata):
    """Test to_original_scale with variable name as string."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    original_scale = wrapper.to_original_scale("channel_contribution")

    # Assert
    assert isinstance(original_scale, xr.DataArray)

    # Should equal scaled * target_scale
    channel_contrib = multidim_idata.posterior.channel_contribution
    target_scale = multidim_idata.constant_data.target_scale
    # Align coordinates before multiplication to ensure coordinate consistency
    channel_contrib_aligned, target_scale_aligned = xr.align(
        channel_contrib, target_scale, join="exact"
    )
    expected = channel_contrib_aligned * target_scale_aligned
    # Align both arrays before comparison to ensure coordinate consistency
    original_scale_aligned, expected_aligned = xr.align(
        original_scale, expected, join="exact"
    )
    # Compare values using numpy since xarray's assert_allclose is strict about
    # coordinate object equality, even when values match
    np.testing.assert_allclose(
        original_scale_aligned.values,
        expected_aligned.values,
    )


def test_to_original_scale_with_dataarray(multidim_idata):
    """Test to_original_scale with DataArray."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)
    scaled_data = multidim_idata.posterior.channel_contribution

    # Act
    original_scale = wrapper.to_original_scale(scaled_data)

    # Assert
    expected = scaled_data * multidim_idata.constant_data.target_scale
    xr.testing.assert_allclose(original_scale, expected)


def test_to_original_scale_uses_existing_variable_if_present(idata_with_original_scale):
    """Test that to_original_scale returns existing _original_scale variable."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_with_original_scale)

    # Act
    original_scale = wrapper.to_original_scale("channel_contribution")

    # Assert - Should return existing variable, not compute
    xr.testing.assert_equal(
        original_scale,
        idata_with_original_scale.posterior.channel_contribution_original_scale,
    )


def test_to_original_scale_with_original_scale_variable_name(idata_with_original_scale):
    """Test to_original_scale with _original_scale variable name returns as-is."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_with_original_scale)

    # Act
    original_scale = wrapper.to_original_scale("channel_contribution_original_scale")

    # Assert - Should return variable directly without transformation
    xr.testing.assert_equal(
        original_scale,
        idata_with_original_scale.posterior.channel_contribution_original_scale,
    )


def test_to_scaled_with_original_scale_variable_name(idata_with_original_scale):
    """Test to_scaled with _original_scale variable name."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_with_original_scale)

    # Act
    scaled = wrapper.to_scaled("channel_contribution_original_scale")

    # Assert - Should return base scaled variable
    xr.testing.assert_equal(
        scaled, idata_with_original_scale.posterior.channel_contribution
    )


def test_to_scaled_with_scaled_variable_name(idata_with_original_scale):
    """Test to_scaled with scaled variable name."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_with_original_scale)

    # Act
    scaled = wrapper.to_scaled("channel_contribution")

    # Assert - Should return base scaled variable
    xr.testing.assert_equal(
        scaled, idata_with_original_scale.posterior.channel_contribution
    )


def test_to_scaled_with_dataarray(multidim_idata):
    """Test to_scaled with DataArray."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Create original scale data
    original_data = (
        multidim_idata.posterior.channel_contribution
        * multidim_idata.constant_data.target_scale
    )

    # Act
    scaled = wrapper.to_scaled(original_data)

    # Assert
    target_scale = multidim_idata.constant_data.target_scale
    # Align coordinates before division to ensure coordinate consistency
    original_data_aligned, target_scale_aligned = xr.align(
        original_data, target_scale, join="exact"
    )
    expected = original_data_aligned / target_scale_aligned
    # Align both arrays before comparison to ensure coordinate consistency
    scaled_aligned, expected_aligned = xr.align(scaled, expected, join="exact")
    # Compare values using numpy since xarray's assert_allclose is strict about
    # coordinate object equality, even when values match
    np.testing.assert_allclose(
        scaled_aligned.values,
        expected_aligned.values,
    )


# ============================================================================
# Category 8: Filtering Method Tests (Delegation)
# ============================================================================


def test_filter_dates_delegates_to_utility(multidim_idata):
    """Test that filter_dates delegates to standalone utility."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    filtered_wrapper = wrapper.filter_dates("2024-03-01", "2024-06-30")

    # Assert - Returns new wrapper
    assert isinstance(filtered_wrapper, MMMIDataWrapper)
    assert filtered_wrapper is not wrapper

    # Assert - idata is filtered
    assert (
        filtered_wrapper.idata.posterior.sizes["date"]
        < wrapper.idata.posterior.sizes["date"]
    )


def test_filter_dates_preserves_schema(multidim_idata, basic_schema):
    """Test that filter_dates preserves schema in new wrapper."""
    # Arrange
    wrapper = MMMIDataWrapper(
        multidim_idata, schema=basic_schema, validate_on_init=False
    )

    # Act
    filtered_wrapper = wrapper.filter_dates("2024-03-01", "2024-06-30")

    # Assert
    assert filtered_wrapper.schema is basic_schema


def test_filter_dates_no_validation_on_filtered_wrapper(multidim_idata, basic_schema):
    """Test that filtered wrapper doesn't re-validate on init."""
    # Arrange
    wrapper = MMMIDataWrapper(
        multidim_idata, schema=basic_schema, validate_on_init=False
    )

    # Act - Should not raise even if filtered idata doesn't match schema
    filtered_wrapper = wrapper.filter_dates("2024-03-01", "2024-06-30")

    # Assert - New wrapper created without validation
    assert filtered_wrapper is not wrapper


def test_filter_dims_delegates_to_utility(multidim_idata):
    """Test that filter_dims delegates to standalone utility."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    filtered_wrapper = wrapper.filter_dims(channel="TV")

    # Assert
    assert isinstance(filtered_wrapper, MMMIDataWrapper)
    assert filtered_wrapper is not wrapper
    # Dimension should be dropped when filtering to single value
    assert "channel" not in filtered_wrapper.idata.posterior.dims


def test_filter_dims_with_multiple_filters(multidim_idata):
    """Test filtering by multiple dimensions simultaneously."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    filtered_wrapper = wrapper.filter_dims(channel="TV", country="US")

    # Assert - Both dimensions should be dropped when filtering to single values
    assert "channel" not in filtered_wrapper.idata.posterior.dims
    assert "country" not in filtered_wrapper.idata.posterior.dims


@pytest.mark.parametrize(
    "dim_filters, expect_schema_none",
    [
        pytest.param(
            {"country": "US"},
            True,
            id="scalar_filter_drops_dim",
        ),
        pytest.param(
            {"country": ["US", "UK"]},
            False,
            id="list_filter_preserves_dim",
        ),
        pytest.param(
            {"country": "US", "channel": ["TV", "Radio"]},
            True,
            id="mixed_filters_any_scalar_drops",
        ),
    ],
)
def test_filter_dims_schema_nullification(
    multidim_idata, dim_filters, expect_schema_none
):
    """Test that filter_dims nullifies schema only when a dimension is dropped."""
    # Arrange - Create schema matching multidim_idata (has custom dims)
    schema = MMMIdataSchema.from_model_config(
        custom_dims=("country",),
        has_controls=False,
        has_seasonality=False,
        time_varying=False,
    )
    wrapper = MMMIDataWrapper(multidim_idata, schema=schema, validate_on_init=False)

    # Act
    filtered = wrapper.filter_dims(**dim_filters)

    # Assert
    if expect_schema_none:
        assert filtered.schema is None
    else:
        assert filtered.schema is not None


# ============================================================================
# Category 9: Aggregation Method Tests (Delegation)
# ============================================================================


def test_aggregate_time_delegates_to_utility(multidim_idata):
    """Test that aggregate_time delegates to standalone utility."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    monthly_wrapper = wrapper.aggregate_time(period="monthly", method="sum")

    # Assert
    assert isinstance(monthly_wrapper, MMMIDataWrapper)
    assert monthly_wrapper is not wrapper
    assert monthly_wrapper.idata.posterior.sizes["date"] <= 12


def test_aggregate_time_all_time_sets_schema_none(multidim_idata):
    """Test that all_time aggregation sets schema=None."""
    # Arrange - Use schema that matches multidim_idata (has custom dims)
    schema = MMMIdataSchema.from_model_config(
        custom_dims=("country",),
        has_controls=False,
        has_seasonality=False,
        time_varying=False,
    )
    wrapper = MMMIDataWrapper(multidim_idata, schema=schema, validate_on_init=False)

    # Act
    total_wrapper = wrapper.aggregate_time(period="all_time", method="sum")

    # Assert - Schema set to None (date dimension removed)
    assert total_wrapper.schema is None
    assert "date" not in total_wrapper.idata.posterior.dims


def test_aggregate_dims_delegates_to_utility(multidim_idata):
    """Test that aggregate_dims delegates to standalone utility."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    combined_wrapper = wrapper.aggregate_dims(
        dim="channel",
        values=["Facebook", "Instagram"],
        new_label="SocialCombined",
        method="sum",
    )

    # Assert
    assert isinstance(combined_wrapper, MMMIDataWrapper)
    assert combined_wrapper is not wrapper
    assert "SocialCombined" in combined_wrapper.idata.posterior.coords["channel"].values


# ============================================================================
# Category 10: Summary Statistics Tests
# ============================================================================


def test_compute_posterior_summary_returns_dataframe(multidim_idata):
    """Test that compute_posterior_summary returns DataFrame."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    summary = wrapper.compute_posterior_summary("channel_contribution", hdi_prob=0.94)

    # Assert
    assert isinstance(summary, pd.DataFrame)

    # Should have standard arviz.summary columns
    assert "mean" in summary.columns
    assert "sd" in summary.columns
    assert "hdi_3%" in summary.columns
    assert "hdi_97%" in summary.columns


def test_compute_posterior_summary_with_original_scale(multidim_idata):
    """Test compute_posterior_summary with original_scale=True."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    summary = wrapper.compute_posterior_summary(
        "channel_contribution", original_scale=True
    )

    # Assert - Values should be larger (unscaled)
    assert isinstance(summary, pd.DataFrame)
    # Mean should be approximately equal to scaled mean * target_scale
    # (This is a basic sanity check, exact values depend on data)


# ============================================================================
# Category 11: Validation Method Tests
# ============================================================================


def test_validate_returns_error_list(multidim_idata):
    """Test that validate returns list of validation errors."""
    # Arrange - Use schema that matches multidim_idata (has custom dims)
    schema = MMMIdataSchema.from_model_config(
        custom_dims=("country",),
        has_controls=False,
        has_seasonality=False,
        time_varying=False,
    )
    wrapper = MMMIDataWrapper(multidim_idata, schema=schema, validate_on_init=False)

    # Act
    errors = wrapper.validate()

    # Assert
    assert isinstance(errors, list)
    # For valid idata, should be empty
    assert errors == []


def test_validate_or_raise_raises_on_error(multidim_idata):
    """Test that validate_or_raise raises ValueError on validation failure."""
    # Arrange - Create schema that won't match idata (wrong dims)
    schema = MMMIdataSchema.from_model_config(
        custom_dims=("nonexistent_dim",),  # Wrong dims
        has_controls=False,
        has_seasonality=False,
        time_varying=False,
    )
    wrapper = MMMIDataWrapper(multidim_idata, schema=schema, validate_on_init=False)

    # Act & Assert
    with pytest.raises(ValueError, match="idata validation failed"):
        wrapper.validate_or_raise()


def test_validate_or_raise_silent_on_success(multidim_idata):
    """Test that validate_or_raise returns None on success."""
    # Arrange - Create schema that matches idata
    schema = MMMIdataSchema.from_model_config(
        custom_dims=("country",),
        has_controls=False,
        has_seasonality=False,
        time_varying=False,
    )
    wrapper = MMMIDataWrapper(multidim_idata, schema=schema, validate_on_init=False)

    # Act
    result = wrapper.validate_or_raise()

    # Assert
    assert result is None


def test_validate_or_raise_raises_when_no_schema(multidim_idata):
    """Test that validate_or_raise raises when schema is None."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata, schema=None)

    # Act & Assert
    with pytest.raises(ValueError, match="No schema provided"):
        wrapper.validate_or_raise()


# ============================================================================
# Category 12: Property Tests
# ============================================================================


def test_dates_property_returns_datetimeindex(multidim_idata):
    """Test that dates property returns pandas DatetimeIndex."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    dates = wrapper.dates

    # Assert
    assert isinstance(dates, pd.DatetimeIndex)
    assert len(dates) == multidim_idata.constant_data.sizes["date"]


def test_channels_property_returns_list(multidim_idata):
    """Test that channels property returns list."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    channels = wrapper.channels

    # Assert
    assert isinstance(channels, list)
    assert len(channels) == multidim_idata.constant_data.sizes["channel"]
    assert channels == multidim_idata.constant_data.coords["channel"].values.tolist()


def test_custom_dims_property_returns_custom_dimensions(multidim_idata):
    """Test that custom_dims property returns custom dimension names."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    custom_dims = wrapper.custom_dims

    # Assert
    assert isinstance(custom_dims, list)

    # Should include custom dims like "country", but not standard ones
    standard_dims = {"date", "channel", "control", "fourier_mode", "chain", "draw"}
    for dim in custom_dims:
        assert dim not in standard_dims


# ============================================================================
# Category 13: MMM Integration Tests
# ============================================================================


def test_mmm_data_property_returns_wrapper(fitted_mmm):
    """Test that MMM.data property returns wrapper."""
    # Arrange & Act
    wrapper = fitted_mmm.data

    # Assert
    assert isinstance(wrapper, MMMIDataWrapper)
    assert wrapper.idata is fitted_mmm.idata


def test_mmm_data_property_returns_fresh_wrapper(fitted_mmm):
    """Test that .data property returns fresh wrapper on each access."""
    # Arrange & Act
    wrapper1 = fitted_mmm.data
    wrapper2 = fitted_mmm.data

    # Assert - Different instances returned (no caching)
    assert wrapper1 is not wrapper2

    # But both wrap the same idata
    assert wrapper1.idata is wrapper2.idata


def test_mmm_data_property_creates_schema_from_config(fitted_mmm_with_controls):
    """Test that .data creates schema from model config."""
    # Arrange & Act
    wrapper = fitted_mmm_with_controls.data

    # Assert - Schema created with appropriate config
    assert wrapper.schema is not None

    # Schema should reflect model config (has_controls=True)
    control_var_schema = wrapper.schema.groups["constant_data"].variables.get(
        "control_data_"
    )
    assert control_var_schema is not None


def test_mmm_data_property_does_not_validate_on_every_access(fitted_mmm):
    """Test that .data property doesn't validate on every access."""
    # Arrange & Act
    wrapper = fitted_mmm.data

    # Assert - validate_on_init was False (no exception raised)
    # We can't directly check the init parameter, but if validation
    # happened every time, this would be slow/fail on invalid data
    assert wrapper is not None


def test_compare_coords_returns_empty_when_compatible(multidim_idata):
    """Compatible idata returns empty dicts (no coord mismatches)."""
    from types import SimpleNamespace

    channels = list(multidim_idata.constant_data.channel.values)
    countries = list(multidim_idata.constant_data.country.values)

    wrapper = MMMIDataWrapper(multidim_idata, validate_on_init=False)
    mmm_stub = SimpleNamespace(
        model=SimpleNamespace(
            named_vars_to_dims={"channel_data": ("date", "country", "channel")},
            coords={"country": countries, "channel": channels},
        )
    )
    in_model_not_idata, in_idata_not_model = wrapper.compare_coords(mmm_stub)
    assert in_model_not_idata == {}
    assert in_idata_not_model == {}


def test_compare_coords_detects_aggregated_labels(multidim_idata):
    """Aggregated dimension labels not in model coords are detected."""
    from types import SimpleNamespace

    channels = list(multidim_idata.constant_data.channel.values)
    countries = list(multidim_idata.constant_data.country.values)
    dates = multidim_idata.constant_data.date.values

    # Build fresh idata with an aggregated country label ("All")
    aggregated_idata = az.InferenceData(
        constant_data=xr.Dataset(
            {
                "channel_data": xr.DataArray(
                    multidim_idata.constant_data.channel_data.isel(country=[0]).values,
                    dims=("date", "country", "channel"),
                    coords={
                        "date": dates,
                        "country": ["All"],
                        "channel": channels,
                    },
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(aggregated_idata, validate_on_init=False)
    mmm_stub = SimpleNamespace(
        model=SimpleNamespace(
            named_vars_to_dims={"channel_data": ("date", "country", "channel")},
            coords={"country": countries, "channel": channels},
        )
    )
    in_model_not_idata, in_idata_not_model = wrapper.compare_coords(mmm_stub)
    assert "country" in in_model_not_idata
    assert in_model_not_idata["country"] == set(countries)
    assert "country" in in_idata_not_model
    assert in_idata_not_model["country"] == {"All"}


# ============================================================================
# Category 14: Additional Coverage Tests - Error Paths and Edge Cases
# ============================================================================


def test_filter_idata_by_dates_preserves_groups_without_date_dim():
    """Test that groups without date dimension are preserved unchanged."""
    # Arrange - Create idata with a group that has no date dimension
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
        # sample_stats typically doesn't have date dimension
        sample_stats=xr.Dataset(
            {
                "lp": xr.DataArray(
                    rng.normal(size=(2, 10)),
                    dims=("chain", "draw"),
                ),
            }
        ),
    )

    # Act
    filtered = filter_idata_by_dates(idata, "2024-02-01", "2024-03-01")

    # Assert - sample_stats preserved unchanged
    assert hasattr(filtered, "sample_stats")
    assert "date" not in filtered.sample_stats.dims
    xr.testing.assert_equal(filtered.sample_stats.lp, idata.sample_stats.lp)


def test_aggregate_idata_time_all_time_mean_method():
    """Test all_time aggregation with mean method."""
    # Arrange
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    np.ones((2, 10, 10)) * 5.0,  # All 5s for predictable mean
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )

    # Act
    aggregated = aggregate_idata_time(idata, period="all_time", method="mean")

    # Assert - date dimension removed
    assert "date" not in aggregated.posterior.dims

    # Mean of 5s should be 5
    np.testing.assert_allclose(aggregated.posterior.mu.values, 5.0)


def test_aggregate_idata_time_all_time_preserves_groups_without_date():
    """Test that all_time aggregation preserves groups without date dimension."""
    # Arrange
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
        sample_stats=xr.Dataset(
            {
                "lp": xr.DataArray(
                    rng.normal(size=(2, 10)),
                    dims=("chain", "draw"),
                ),
            }
        ),
    )

    # Act
    aggregated = aggregate_idata_time(idata, period="all_time", method="sum")

    # Assert - sample_stats preserved unchanged
    xr.testing.assert_equal(aggregated.sample_stats.lp, idata.sample_stats.lp)


def test_aggregate_idata_time_all_time_unknown_method_raises():
    """Test that unknown aggregation method raises ValueError for all_time."""
    # Arrange
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )

    # Act & Assert
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        aggregate_idata_time(idata, period="all_time", method="invalid")


def test_aggregate_idata_time_periodic_preserves_groups_without_date():
    """Test that periodic aggregation preserves groups without date dimension."""
    # Arrange
    dates = pd.date_range("2024-01-01", periods=52, freq="W")

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 52)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
        sample_stats=xr.Dataset(
            {
                "lp": xr.DataArray(
                    rng.normal(size=(2, 10)),
                    dims=("chain", "draw"),
                ),
            }
        ),
    )

    # Act
    aggregated = aggregate_idata_time(idata, period="monthly", method="sum")

    # Assert - sample_stats preserved unchanged
    xr.testing.assert_equal(aggregated.sample_stats.lp, idata.sample_stats.lp)


def test_aggregate_idata_time_periodic_unknown_method_raises():
    """Test that unknown aggregation method raises ValueError for periodic."""
    # Arrange
    dates = pd.date_range("2024-01-01", periods=52, freq="W")

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 52)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )

    # Act & Assert
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        aggregate_idata_time(idata, period="monthly", method="invalid")


def test_aggregate_idata_dims_unknown_method_raises():
    """Test that unknown aggregation method raises ValueError in aggregate_idata_dims."""
    # Arrange
    dates = pd.date_range("2024-01-01", periods=10, freq="W")
    channels = ["TV", "Radio"]

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(size=(2, 10, 10, 2)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={"date": dates, "channel": channels},
                ),
            }
        ),
    )

    # Act & Assert
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        aggregate_idata_dims(
            idata,
            dim="channel",
            values=["TV", "Radio"],
            new_label="All",
            method="invalid",
        )


def test_aggregate_idata_dims_all_values_aggregated():
    """Test aggregating all values in a dimension (no other_values)."""
    # Arrange
    dates = pd.date_range("2024-01-01", periods=10, freq="W")
    channels = ["TV", "Radio"]

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(size=(2, 10, 10, 2)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={"date": dates, "channel": channels},
                ),
            }
        ),
    )

    # Act - Aggregate ALL channels into one
    combined = aggregate_idata_dims(
        idata,
        dim="channel",
        values=["TV", "Radio"],  # All channels
        new_label="All",
        method="sum",
    )

    # Assert - Only "All" channel should exist
    assert list(combined.posterior.coords["channel"].values) == ["All"]
    assert combined.posterior.sizes["channel"] == 1


def test_aggregate_idata_dims_preserves_dims_of_variables_without_aggregated_dim(
    multidim_idata,
):
    """Test that variables without the aggregated dimension keep their original dims."""
    # Arrange - Add a variable with only ('date',) dims to the existing fixture
    idata = multidim_idata

    # Sanity check - dayofyear should have only ('date',) dims before aggregation
    assert idata.constant_data.dayofyear.dims == ("date",)

    # Act - Aggregate US and UK into combined_region
    aggregated = aggregate_idata_dims(
        idata,
        dim="country",
        values=["US", "UK"],
        new_label="combined_region",
        method="sum",
    )

    # Assert - dayofyear should STILL have only ('date',) dims, NOT ('country', 'date')
    assert aggregated.constant_data.dayofyear.dims == ("date",), (
        f"Expected dayofyear dims to be ('date',), "
        f"but got {aggregated.constant_data.dayofyear.dims}. "
        f"Variables without the aggregated dimension should keep their original dims."
    )

    # Also verify the data values are preserved correctly
    np.testing.assert_array_equal(
        aggregated.constant_data.dayofyear.values,
        idata.constant_data.dayofyear.values,
    )


def test_get_target_raises_when_target_data_missing():
    """Test that get_target raises when constant_data/target_data is missing."""
    # Arrange - Create idata without constant_data/target_data
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata)

    # Act & Assert
    with pytest.raises(ValueError, match="Target data not found in constant_data"):
        wrapper.get_target()


def test_get_channel_spend_raises_when_channel_data_missing():
    """Test that get_channel_spend raises when channel_data is missing."""
    # Arrange - Create idata without constant_data/channel_data
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata)

    # Act & Assert
    with pytest.raises(ValueError, match="Channel data not found in constant_data"):
        wrapper.get_channel_spend()


def test_get_contributions_baseline_scaled(idata_with_all_contributions):
    """Test get_contributions with baseline in scaled space (original_scale=False)."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_with_all_contributions)

    # Act
    contributions = wrapper.get_contributions(
        original_scale=False,
        include_baseline=True,
        include_controls=False,
        include_seasonality=False,
    )

    # Assert
    assert "baseline" in contributions
    # Baseline should be the raw value (not multiplied by target_scale)
    xr.testing.assert_equal(
        contributions["baseline"],
        idata_with_all_contributions.posterior.intercept_contribution,
    )


def test_get_contributions_controls_with_original_scale_variable():
    """Test get_contributions uses existing control_contribution_original_scale."""
    # Arrange - Create idata with control_contribution_original_scale
    dates = pd.date_range("2024-01-01", periods=10, freq="W")
    channels = ["TV", "Radio"]
    controls = ["price"]

    # Create control_contribution_original_scale with distinct values
    control_orig_scale_values = rng.normal(size=(2, 10, 10, 1)) * 999

    idata = az.InferenceData(
        constant_data=xr.Dataset(
            {
                "channel_data": xr.DataArray(
                    rng.uniform(0, 100, size=(10, 2)),
                    dims=("date", "channel"),
                    coords={"date": dates, "channel": channels},
                ),
                "target_scale": xr.DataArray(100.0),
            }
        ),
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(size=(2, 10, 10, 2)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={"date": dates, "channel": channels},
                ),
                "control_contribution": xr.DataArray(
                    rng.normal(size=(2, 10, 10, 1)),
                    dims=("chain", "draw", "date", "control"),
                    coords={"date": dates, "control": controls},
                ),
                "control_contribution_original_scale": xr.DataArray(
                    control_orig_scale_values,
                    dims=("chain", "draw", "date", "control"),
                    coords={"date": dates, "control": controls},
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata)

    # Act
    contributions = wrapper.get_contributions(
        original_scale=True,
        include_baseline=False,
        include_controls=True,
        include_seasonality=False,
    )

    # Assert - Should use existing original scale variable
    # Compare values directly since xr.testing.assert_equal is strict about coordinate objects
    np.testing.assert_array_equal(
        contributions["controls"].values,
        idata.posterior.control_contribution_original_scale.values,
    )


def test_get_contributions_controls_scaled(idata_with_all_contributions):
    """Test get_contributions with controls in scaled space (original_scale=False)."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_with_all_contributions)

    # Act
    contributions = wrapper.get_contributions(
        original_scale=False,
        include_baseline=False,
        include_controls=True,
        include_seasonality=False,
    )

    # Assert
    assert "controls" in contributions
    # Control should be the raw value (not multiplied by target_scale)
    # Compare values directly since xr.testing.assert_equal is strict about coordinate objects
    np.testing.assert_array_equal(
        contributions["controls"].values,
        idata_with_all_contributions.posterior.control_contribution.values,
    )


def test_get_contributions_seasonality_with_original_scale_variable():
    """Test get_contributions uses existing yearly_seasonality_contribution_original_scale."""
    # Arrange - Create idata with yearly_seasonality_contribution_original_scale
    dates = pd.date_range("2024-01-01", periods=10, freq="W")
    channels = ["TV", "Radio"]

    idata = az.InferenceData(
        constant_data=xr.Dataset(
            {
                "channel_data": xr.DataArray(
                    rng.uniform(0, 100, size=(10, 2)),
                    dims=("date", "channel"),
                    coords={"date": dates, "channel": channels},
                ),
                "target_scale": xr.DataArray(100.0),
            }
        ),
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(size=(2, 10, 10, 2)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={"date": dates, "channel": channels},
                ),
                "yearly_seasonality_contribution": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
                "yearly_seasonality_contribution_original_scale": xr.DataArray(
                    rng.normal(size=(2, 10, 10)) * 999,  # Distinct values
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata)

    # Act
    contributions = wrapper.get_contributions(
        original_scale=True,
        include_baseline=False,
        include_controls=False,
        include_seasonality=True,
    )

    # Assert - Should use existing original scale variable
    xr.testing.assert_equal(
        contributions["seasonality"],
        idata.posterior.yearly_seasonality_contribution_original_scale,
    )


def test_get_contributions_seasonality_scaled(idata_with_all_contributions):
    """Test get_contributions with seasonality in scaled space (original_scale=False)."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_with_all_contributions)

    # Act
    contributions = wrapper.get_contributions(
        original_scale=False,
        include_baseline=False,
        include_controls=False,
        include_seasonality=True,
    )

    # Assert
    assert "seasonality" in contributions
    # Seasonality should be the raw value (not multiplied by target_scale)
    xr.testing.assert_equal(
        contributions["seasonality"],
        idata_with_all_contributions.posterior.yearly_seasonality_contribution,
    )


def test_to_original_scale_raises_when_original_scale_var_not_found():
    """Test to_original_scale raises when _original_scale variable doesn't exist."""
    # Arrange - Create idata without the _original_scale variable
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    idata = az.InferenceData(
        constant_data=xr.Dataset(
            {
                "target_scale": xr.DataArray(100.0),
            }
        ),
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata)

    # Act & Assert - Request a var that ends with _original_scale but doesn't exist
    with pytest.raises(ValueError, match="not found in posterior"):
        wrapper.to_original_scale("nonexistent_original_scale")


def test_to_original_scale_raises_when_var_not_found():
    """Test to_original_scale raises when variable doesn't exist."""
    # Arrange
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    idata = az.InferenceData(
        constant_data=xr.Dataset(
            {
                "target_scale": xr.DataArray(100.0),
            }
        ),
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata)

    # Act & Assert
    with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
        wrapper.to_original_scale("nonexistent")


def test_to_scaled_raises_when_base_var_not_found():
    """Test to_scaled raises when base variable for _original_scale doesn't exist."""
    # Arrange
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    idata = az.InferenceData(
        constant_data=xr.Dataset(
            {
                "target_scale": xr.DataArray(100.0),
            }
        ),
        posterior=xr.Dataset(
            {
                # Has nonexistent_original_scale but NOT 'nonexistent'
                "nonexistent_original_scale": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata)

    # Act & Assert - Looking for base var 'nonexistent' which doesn't exist
    with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
        wrapper.to_scaled("nonexistent_original_scale")


def test_to_scaled_raises_when_var_not_found():
    """Test to_scaled raises when variable doesn't exist."""
    # Arrange
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    idata = az.InferenceData(
        constant_data=xr.Dataset(
            {
                "target_scale": xr.DataArray(100.0),
            }
        ),
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata)

    # Act & Assert
    with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
        wrapper.to_scaled("nonexistent")


def test_compute_posterior_summary_scaled_raises_when_var_not_found():
    """Test compute_posterior_summary with original_scale=False raises when var not found."""
    # Arrange
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    idata = az.InferenceData(
        constant_data=xr.Dataset(
            {
                "target_scale": xr.DataArray(100.0),
            }
        ),
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata)

    # Act & Assert
    with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
        wrapper.compute_posterior_summary("nonexistent", original_scale=False)


def test_compute_posterior_summary_scaled_returns_dataframe(multidim_idata):
    """Test compute_posterior_summary with original_scale=False returns DataFrame."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act - Request summary in scaled space (not original scale)
    summary = wrapper.compute_posterior_summary(
        "channel_contribution",
        hdi_prob=0.94,
        original_scale=False,
    )

    # Assert
    assert isinstance(summary, pd.DataFrame)
    assert "mean" in summary.columns
    assert "sd" in summary.columns


def test_validate_raises_when_no_schema():
    """Test validate raises when no schema provided."""
    # Arrange
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata, schema=None)

    # Act & Assert
    with pytest.raises(ValueError, match="No schema provided"):
        wrapper.validate()


def test_dates_property_from_posterior_when_no_constant_data():
    """Test dates property falls back to posterior when constant_data missing."""
    # Arrange
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata)

    # Act
    result_dates = wrapper.dates

    # Assert
    assert isinstance(result_dates, pd.DatetimeIndex)
    assert len(result_dates) == 10
    np.testing.assert_array_equal(result_dates, dates)


def test_dates_property_raises_when_no_date_coord():
    """Test dates property raises when no date coordinate found."""
    # Arrange - Create idata with no date dimension
    idata = az.InferenceData(
        sample_stats=xr.Dataset(
            {
                "lp": xr.DataArray(
                    rng.normal(size=(2, 10)),
                    dims=("chain", "draw"),
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata)

    # Act & Assert
    with pytest.raises(ValueError, match="Could not find date coordinate"):
        _ = wrapper.dates


def test_channels_property_from_posterior_when_no_constant_data():
    """Test channels property falls back to posterior when constant_data missing."""
    # Arrange
    dates = pd.date_range("2024-01-01", periods=10, freq="W")
    channels = ["TV", "Radio", "Facebook"]

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(size=(2, 10, 10, 3)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={"date": dates, "channel": channels},
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata)

    # Act
    result_channels = wrapper.channels

    # Assert
    assert isinstance(result_channels, list)
    assert result_channels == channels


def test_channels_property_raises_when_no_channel_coord():
    """Test channels property raises when no channel coordinate found."""
    # Arrange - Create idata with no channel dimension (neither constant_data nor posterior)
    idata = az.InferenceData(
        sample_stats=xr.Dataset(
            {
                "lp": xr.DataArray(
                    rng.normal(size=(2, 10)),
                    dims=("chain", "draw"),
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata)

    # Act & Assert
    with pytest.raises(ValueError, match="Could not find channel coordinate"):
        _ = wrapper.channels


def test_custom_dims_returns_empty_when_no_constant_data():
    """Test custom_dims returns empty list when no constant_data."""
    # Arrange
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata)

    # Act
    custom_dims = wrapper.custom_dims

    # Assert
    assert custom_dims == []


def test_filter_dates_wrapper_returns_self_when_none(multidim_idata):
    """Test filter_dates wrapper method returns self when both dates are None."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    result = wrapper.filter_dates(None, None)

    # Assert - Same wrapper returned
    assert result is wrapper


def test_filter_dims_wrapper_returns_self_when_no_filters(multidim_idata):
    """Test filter_dims wrapper method returns self when no filters provided."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    result = wrapper.filter_dims()

    # Assert - Same wrapper returned
    assert result is wrapper


# ============================================================================
# Category 15: Scale Accessor Method Tests
# ============================================================================


def test_get_channel_scale_returns_scale_array(multidim_idata):
    """Test that get_channel_scale returns channel scale DataArray."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    channel_scale = wrapper.get_channel_scale()

    # Assert
    assert isinstance(channel_scale, xr.DataArray)
    xr.testing.assert_equal(channel_scale, multidim_idata.constant_data.channel_scale)


def test_get_channel_scale_raises_when_missing():
    """Test that get_channel_scale raises when channel_scale is missing."""
    # Arrange - Create idata without channel_scale
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    idata = az.InferenceData(
        constant_data=xr.Dataset(
            {
                "target_data": xr.DataArray(
                    rng.uniform(100, 1000, size=(10,)),
                    dims=("date",),
                    coords={"date": dates},
                ),
                # channel_scale intentionally missing
            }
        ),
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata)

    # Act & Assert
    with pytest.raises(ValueError, match="channel_scale not found in constant_data"):
        wrapper.get_channel_scale()


def test_get_target_scale_returns_scale_array(multidim_idata):
    """Test that get_target_scale returns target scale DataArray."""
    # Arrange
    wrapper = MMMIDataWrapper(multidim_idata)

    # Act
    target_scale = wrapper.get_target_scale()

    # Assert
    assert isinstance(target_scale, xr.DataArray)
    xr.testing.assert_equal(target_scale, multidim_idata.constant_data.target_scale)


def test_get_target_scale_raises_when_missing():
    """Test that get_target_scale raises when target_scale is missing."""
    # Arrange - Create idata without target_scale
    dates = pd.date_range("2024-01-01", periods=10, freq="W")

    idata = az.InferenceData(
        constant_data=xr.Dataset(
            {
                "target_data": xr.DataArray(
                    rng.uniform(100, 1000, size=(10,)),
                    dims=("date",),
                    coords={"date": dates},
                ),
                # target_scale intentionally missing
            }
        ),
        posterior=xr.Dataset(
            {
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )

    wrapper = MMMIDataWrapper(idata)

    # Act & Assert
    with pytest.raises(ValueError, match="target_scale not found in constant_data"):
        wrapper.get_target_scale()


# ============================================================================
# Category 16: Target Scale Missing Error Tests
# ============================================================================


@pytest.fixture
def idata_without_target_scale():
    """Create InferenceData with target_data but without target_scale."""
    dates = pd.date_range("2024-01-01", periods=10, freq="W")
    channels = ["TV", "Radio"]

    return az.InferenceData(
        constant_data=xr.Dataset(
            {
                "channel_data": xr.DataArray(
                    rng.uniform(0, 100, size=(10, 2)),
                    dims=("date", "channel"),
                    coords={"date": dates, "channel": channels},
                ),
                "target_data": xr.DataArray(
                    rng.uniform(100, 1000, size=(10,)),
                    dims=("date",),
                    coords={"date": dates},
                ),
                # Note: target_scale is intentionally missing!
            }
        ),
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(size=(2, 10, 10, 2)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={"date": dates, "channel": channels},
                ),
                "mu": xr.DataArray(
                    rng.normal(size=(2, 10, 10)),
                    dims=("chain", "draw", "date"),
                    coords={"date": dates},
                ),
            }
        ),
    )


def test_get_target_scaled_raises_valueerror_when_target_scale_missing(
    idata_without_target_scale,
):
    """Test that get_target raises ValueError (not AttributeError) when target_scale is missing."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_without_target_scale)

    # Act & Assert - Should raise ValueError with helpful message
    with pytest.raises(ValueError, match="target_scale not found in constant_data"):
        wrapper.get_target(original_scale=False)


def test_get_target_original_scale_works_without_target_scale(
    idata_without_target_scale,
):
    """Test that get_target with original_scale=True works even without target_scale."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_without_target_scale)

    # Act - Should work since we don't need target_scale for original_scale=True
    result = wrapper.get_target(original_scale=True)

    # Assert
    assert isinstance(result, xr.DataArray)


def test_get_contributions_raises_valueerror_when_target_scale_missing(
    idata_without_target_scale,
):
    """Test that get_contributions raises ValueError when target_scale is missing and needed."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_without_target_scale)

    # Act & Assert - Should raise ValueError with helpful message
    with pytest.raises(ValueError, match="target_scale not found in constant_data"):
        wrapper.get_contributions(original_scale=True)


def test_get_contributions_scaled_works_without_target_scale(
    idata_without_target_scale,
):
    """Test that get_contributions with original_scale=False works without target_scale."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_without_target_scale)

    # Act - Should work since we don't need target_scale for original_scale=False
    result = wrapper.get_contributions(original_scale=False)

    # Assert
    assert isinstance(result, xr.Dataset)
    assert "channel" in result


def test_to_original_scale_raises_valueerror_when_target_scale_missing(
    idata_without_target_scale,
):
    """Test that to_original_scale raises ValueError (not AttributeError) when target_scale missing."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_without_target_scale)

    # Act & Assert - Should raise ValueError with helpful message
    with pytest.raises(ValueError, match="target_scale not found in constant_data"):
        wrapper.to_original_scale("mu")


def test_to_original_scale_dataarray_raises_valueerror_when_target_scale_missing(
    idata_without_target_scale,
):
    """Test that to_original_scale with DataArray raises ValueError when target_scale missing."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_without_target_scale)
    data = idata_without_target_scale.posterior.mu

    # Act & Assert - Should raise ValueError with helpful message
    with pytest.raises(ValueError, match="target_scale not found in constant_data"):
        wrapper.to_original_scale(data)


def test_to_scaled_dataarray_raises_valueerror_when_target_scale_missing(
    idata_without_target_scale,
):
    """Test that to_scaled with DataArray raises ValueError when target_scale missing."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_without_target_scale)
    data = idata_without_target_scale.posterior.mu

    # Act & Assert - Should raise ValueError with helpful message
    with pytest.raises(ValueError, match="target_scale not found in constant_data"):
        wrapper.to_scaled(data)


def test_to_scaled_string_works_without_target_scale(idata_without_target_scale):
    """Test that to_scaled with string variable works without target_scale."""
    # Arrange
    wrapper = MMMIDataWrapper(idata_without_target_scale)

    # Act - Should work since we're just returning the posterior variable
    result = wrapper.to_scaled("mu")

    # Assert
    assert isinstance(result, xr.DataArray)
    xr.testing.assert_equal(result, idata_without_target_scale.posterior.mu)
