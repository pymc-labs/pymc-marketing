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
"""Tests for MMM Summary DataFrame generation (Component 2).

This module tests Component 2 of the MMM Data & Plotting Framework:
factory functions that transform MMMIDataWrapper (Component 1) into
summary DataFrames with HDI statistics.

The tests follow a TDD approach - tests define the expected API and behavior.
"""

import importlib.util

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.idata_wrapper import MMMIDataWrapper

# Seed for reproducibility
SEED = sum(map(ord, "summary_tests"))
rng = np.random.default_rng(seed=SEED)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def simple_dates():
    """Date range for testing (52 weeks)."""
    return pd.date_range("2024-01-01", periods=52, freq="W")


@pytest.fixture(scope="module")
def simple_channels():
    """Channel list for testing."""
    return ["TV", "Radio", "Social"]


@pytest.fixture
def mock_mmm_idata_wrapper(simple_dates, simple_channels):
    """Mock MMMIDataWrapper with complete data for testing.

    Creates a real MMMIDataWrapper with mock InferenceData.
    The wrapper provides the full Component 1 API.
    """
    local_rng = np.random.default_rng(seed=42)

    # Create mock InferenceData with required structure
    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    local_rng.normal(loc=1000, scale=100, size=(2, 10, 52, 3)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={"date": simple_dates, "channel": simple_channels},
                ),
                "mu": xr.DataArray(
                    local_rng.normal(loc=5000, scale=200, size=(2, 10, 52)),
                    dims=("chain", "draw", "date"),
                    coords={"date": simple_dates},
                ),
            }
        ),
        posterior_predictive=xr.Dataset(
            {
                "y": xr.DataArray(
                    local_rng.normal(loc=5000, scale=200, size=(2, 10, 52)),
                    dims=("chain", "draw", "date"),
                    coords={"date": simple_dates},
                ),
            }
        ),
        fit_data=xr.Dataset(
            {
                "target": xr.DataArray(
                    local_rng.uniform(4000, 6000, size=52),
                    dims=("date",),
                    coords={"date": simple_dates},
                ),
            }
        ),
        constant_data=xr.Dataset(
            {
                "channel_data": xr.DataArray(
                    local_rng.uniform(0, 100, size=(52, 3)),
                    dims=("date", "channel"),
                    coords={"date": simple_dates, "channel": simple_channels},
                ),
                "channel_scale": xr.DataArray(
                    [100.0, 50.0, 75.0],
                    dims=("channel",),
                    coords={"channel": simple_channels},
                ),
                "target_scale": xr.DataArray(500.0),
                "target_data": xr.DataArray(
                    local_rng.uniform(4000, 6000, size=52),
                    dims=("date",),
                    coords={"date": simple_dates},
                ),
            }
        ),
    )

    # Create MMMIDataWrapper without schema validation
    return MMMIDataWrapper(idata, schema=None, validate_on_init=False)


@pytest.fixture
def mock_mmm_idata_wrapper_with_zero_spend(simple_dates):
    """Mock MMMIDataWrapper with zero spend channel for ROAS testing."""
    local_rng = np.random.default_rng(seed=43)
    channels = ["TV", "Radio", "zero_spend_channel"]

    # Create channel_data with one channel having zero spend
    channel_data = local_rng.uniform(0, 100, size=(52, 3))
    channel_data[:, 2] = 0.0  # Zero spend for third channel

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    local_rng.normal(loc=1000, scale=100, size=(2, 10, 52, 3)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={"date": simple_dates, "channel": channels},
                ),
            }
        ),
        constant_data=xr.Dataset(
            {
                "channel_data": xr.DataArray(
                    channel_data,
                    dims=("date", "channel"),
                    coords={"date": simple_dates, "channel": channels},
                ),
                "channel_scale": xr.DataArray(
                    [100.0, 50.0, 1.0],
                    dims=("channel",),
                    coords={"channel": channels},
                ),
                "target_scale": xr.DataArray(500.0),
                "target_data": xr.DataArray(
                    local_rng.uniform(4000, 6000, size=52),
                    dims=("date",),
                    coords={"date": simple_dates},
                ),
            }
        ),
    )

    return MMMIDataWrapper(idata, schema=None, validate_on_init=False)


# ============================================================================
# Category 1: Factory Function Existence & Signatures
# ============================================================================


class TestFactoryFunctionExistence:
    """Test that all factory functions exist and are importable."""

    def test_factory_functions_importable(self):
        """Test that all factory functions are importable from summary module."""
        from pymc_marketing.mmm.summary import (
            create_channel_spend_dataframe,
            create_contribution_summary,
            create_decay_curves,
            create_period_over_period_summary,
            create_posterior_predictive_summary,
            create_roas_summary,
            create_saturation_curves,
            create_total_contribution_summary,
        )

        # Assert all are callable
        assert callable(create_posterior_predictive_summary)
        assert callable(create_contribution_summary)
        assert callable(create_roas_summary)
        assert callable(create_channel_spend_dataframe)
        assert callable(create_saturation_curves)
        assert callable(create_decay_curves)
        assert callable(create_total_contribution_summary)
        assert callable(create_period_over_period_summary)

    def test_factory_class_importable(self):
        """Test that MMMSummaryFactory class is importable."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        assert MMMSummaryFactory is not None
        # Should be a class
        assert isinstance(MMMSummaryFactory, type)


# ============================================================================
# Category 2: DataFrame Schema Validation
# ============================================================================


class TestDataFrameSchemas:
    """Test that returned DataFrames have correct structure and columns."""

    def test_posterior_predictive_summary_schema(self, mock_mmm_idata_wrapper):
        """Test posterior predictive summary returns DataFrame with correct schema."""
        from pymc_marketing.mmm.summary import create_posterior_predictive_summary

        # Act
        df = create_posterior_predictive_summary(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.94],
        )

        # Assert - required columns
        required_columns = {"date", "mean", "median", "observed"}
        assert required_columns.issubset(set(df.columns)), (
            f"Missing required columns. Expected {required_columns}, got {set(df.columns)}"
        )

        # Assert - HDI columns for 0.94
        assert "abs_error_94_lower" in df.columns, "Missing HDI lower bound column"
        assert "abs_error_94_upper" in df.columns, "Missing HDI upper bound column"

        # Assert - correct dtypes
        assert pd.api.types.is_datetime64_any_dtype(df["date"]), (
            f"date column should be datetime64, got {df['date'].dtype}"
        )
        assert pd.api.types.is_float_dtype(df["mean"]), (
            f"mean column should be float, got {df['mean'].dtype}"
        )
        assert pd.api.types.is_float_dtype(df["median"]), (
            f"median column should be float, got {df['median'].dtype}"
        )

    def test_contribution_summary_schema(self, mock_mmm_idata_wrapper):
        """Test contribution summary returns DataFrame with correct schema."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Act
        df = create_contribution_summary(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.80, 0.94],
            component="channel",
        )

        # Assert - required columns
        required_columns = {"date", "channel", "mean", "median"}
        assert required_columns.issubset(set(df.columns)), (
            f"Missing required columns. Expected {required_columns}, got {set(df.columns)}"
        )

        # Assert - HDI columns for both probabilities
        assert "abs_error_80_lower" in df.columns, "Missing 80% HDI lower"
        assert "abs_error_80_upper" in df.columns, "Missing 80% HDI upper"
        assert "abs_error_94_lower" in df.columns, "Missing 94% HDI lower"
        assert "abs_error_94_upper" in df.columns, "Missing 94% HDI upper"

        # Assert - channel values are present (either string or the original coordinate type)
        assert len(df["channel"].unique()) > 0, "channel column should have values"

    def test_roas_summary_schema(self, mock_mmm_idata_wrapper):
        """Test ROAS summary returns DataFrame with correct schema."""
        from pymc_marketing.mmm.summary import create_roas_summary

        # Act
        df = create_roas_summary(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.94],
        )

        # Assert - required columns
        required_columns = {"date", "channel", "mean", "median"}
        assert required_columns.issubset(set(df.columns)), (
            f"Missing required columns for ROAS. Got: {set(df.columns)}"
        )

        # Assert - HDI columns
        assert "abs_error_94_lower" in df.columns
        assert "abs_error_94_upper" in df.columns

    def test_channel_spend_dataframe_schema(self, mock_mmm_idata_wrapper):
        """Test channel spend DataFrame has correct schema (no HDI columns)."""
        from pymc_marketing.mmm.summary import create_channel_spend_dataframe

        # Act
        df = create_channel_spend_dataframe(data=mock_mmm_idata_wrapper)

        # Assert - required columns
        required_columns = {"date", "channel", "channel_data"}
        assert set(df.columns) == required_columns, (
            f"Expected exactly {required_columns}, got {set(df.columns)}"
        )

        # Assert - NO HDI columns (this is raw data)
        hdi_columns = [col for col in df.columns if "abs_error" in col]
        assert len(hdi_columns) == 0, (
            f"Channel spend should not have HDI columns, found: {hdi_columns}"
        )


# ============================================================================
# Category 3: Output Format (Pandas vs Polars)
# ============================================================================


class TestOutputFormats:
    """Test output format parameter correctly controls DataFrame type."""

    def test_default_output_is_pandas(self, mock_mmm_idata_wrapper):
        """Test that default output format is pandas DataFrame."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Act - no output_format specified
        df = create_contribution_summary(data=mock_mmm_idata_wrapper)

        # Assert
        assert isinstance(df, pd.DataFrame), (
            f"Default output should be pandas DataFrame, got {type(df)}"
        )

    def test_pandas_output_format_explicit(self, mock_mmm_idata_wrapper):
        """Test that output_format='pandas' returns pandas DataFrame."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Act
        df = create_contribution_summary(
            data=mock_mmm_idata_wrapper,
            output_format="pandas",
        )

        # Assert
        assert isinstance(df, pd.DataFrame), (
            f"output_format='pandas' should return pandas DataFrame, got {type(df)}"
        )

    @pytest.mark.skipif(
        not importlib.util.find_spec("polars"),
        reason="Polars not installed",
    )
    def test_polars_output_format_when_installed(self, mock_mmm_idata_wrapper):
        """Test that output_format='polars' returns polars DataFrame when installed."""
        import polars as pl

        from pymc_marketing.mmm.summary import create_contribution_summary

        # Act
        df = create_contribution_summary(
            data=mock_mmm_idata_wrapper,
            output_format="polars",
        )

        # Assert
        assert isinstance(df, pl.DataFrame), (
            f"output_format='polars' should return polars DataFrame, got {type(df)}"
        )

        # Verify it has the same columns as pandas version
        df_pandas = create_contribution_summary(
            data=mock_mmm_idata_wrapper,
            output_format="pandas",
        )
        assert set(df.columns) == set(df_pandas.columns), (
            "Polars and Pandas versions should have same columns"
        )

    @pytest.mark.skipif(
        importlib.util.find_spec("polars") is not None,
        reason="Test requires Polars NOT installed",
    )
    def test_polars_output_raises_when_not_installed(self, mock_mmm_idata_wrapper):
        """Test that requesting Polars without it installed raises helpful error."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Act & Assert
        with pytest.raises(ImportError, match=r"Polars is required.*polars"):
            create_contribution_summary(
                data=mock_mmm_idata_wrapper,
                output_format="polars",
            )

    def test_invalid_output_format_raises(self, mock_mmm_idata_wrapper):
        """Test that invalid output_format raises ValueError."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Act & Assert
        with pytest.raises(ValueError, match=r"Unknown output_format.*spark"):
            create_contribution_summary(
                data=mock_mmm_idata_wrapper,
                output_format="spark",  # Invalid
            )

    @pytest.mark.parametrize(
        "factory_function_name",
        [
            "create_posterior_predictive_summary",
            "create_contribution_summary",
            "create_roas_summary",
            "create_channel_spend_dataframe",
            "create_saturation_curves",
            "create_decay_curves",
            "create_total_contribution_summary",
            "create_period_over_period_summary",
        ],
    )
    def test_all_factory_functions_support_output_format(
        self, mock_mmm_idata_wrapper, factory_function_name
    ):
        """Test that all factory functions accept output_format parameter."""
        from pymc_marketing.mmm import summary

        factory_func = getattr(summary, factory_function_name)

        # Act - should not raise TypeError for unexpected keyword argument
        try:
            df = factory_func(data=mock_mmm_idata_wrapper, output_format="pandas")
            # Assert it returned something
            assert df is not None, f"{factory_function_name} returned None"
        except TypeError as e:
            if "output_format" in str(e):
                pytest.fail(
                    f"{factory_function_name} does not accept output_format parameter: {e}"
                )
            raise


# ============================================================================
# Category 4: HDI Computation Correctness
# ============================================================================


class TestHDIComputation:
    """Test HDI bounds are computed correctly."""

    def test_hdi_bounds_contain_median(self, mock_mmm_idata_wrapper):
        """Test that HDI bounds contain the median value."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Act
        df = create_contribution_summary(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.94],
        )

        # Assert - HDI contains median
        assert (df["abs_error_94_lower"] <= df["median"]).all(), (
            "HDI lower bound should be <= median"
        )
        assert (df["median"] <= df["abs_error_94_upper"]).all(), (
            "median should be <= HDI upper bound"
        )

    def test_wider_hdi_contains_narrower_hdi(self, mock_mmm_idata_wrapper):
        """Test that wider HDI intervals contain narrower intervals on average."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Act - request two HDI levels
        df = create_contribution_summary(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.80, 0.94],  # 94% is wider than 80%
        )

        # Assert - 94% interval width is >= 80% interval width on average
        # (individual rows may vary due to sampling, but overall pattern should hold)
        width_94 = df["abs_error_94_upper"] - df["abs_error_94_lower"]
        width_80 = df["abs_error_80_upper"] - df["abs_error_80_lower"]

        # On average, 94% HDI should be wider than 80% HDI
        assert width_94.mean() >= width_80.mean(), (
            f"94% HDI should be wider on average. "
            f"Got 94% width={width_94.mean():.2f}, 80% width={width_80.mean():.2f}"
        )

    def test_multiple_hdi_probs_in_same_dataframe(self, mock_mmm_idata_wrapper):
        """Test that multiple HDI probabilities create all expected columns."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Act
        df = create_contribution_summary(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.80, 0.90, 0.94],
        )

        # Assert - all HDI columns exist
        expected_hdi_columns = {
            "abs_error_80_lower",
            "abs_error_80_upper",
            "abs_error_90_lower",
            "abs_error_90_upper",
            "abs_error_94_lower",
            "abs_error_94_upper",
        }
        assert expected_hdi_columns.issubset(set(df.columns)), (
            f"Missing HDI columns. Expected {expected_hdi_columns}, got {set(df.columns)}"
        )


# ============================================================================
# Category 5: Data Transformation (xarray → DataFrame)
# ============================================================================


class TestDataTransformation:
    """Test xarray to DataFrame conversion preserves data correctly."""

    def test_mcmc_dimensions_collapsed(self, mock_mmm_idata_wrapper):
        """Test that chain and draw dimensions are collapsed in output."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Arrange - verify input has chain/draw dimensions
        contributions = mock_mmm_idata_wrapper.get_channel_contributions()
        assert "chain" in contributions.dims, "Test data should have chain dimension"
        assert "draw" in contributions.dims, "Test data should have draw dimension"

        # Act
        df = create_contribution_summary(data=mock_mmm_idata_wrapper)

        # Assert - output should NOT have chain/draw columns
        assert "chain" not in df.columns, (
            "Output DataFrame should not have chain column (should be collapsed)"
        )
        assert "draw" not in df.columns, (
            "Output DataFrame should not have draw column (should be collapsed)"
        )

    def test_data_dimensions_preserved(self, mock_mmm_idata_wrapper):
        """Test that data dimensions (date, channel) are preserved as columns."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Act
        df = create_contribution_summary(data=mock_mmm_idata_wrapper)

        # Assert - data dimensions should be columns
        assert "date" in df.columns, "date dimension should become DataFrame column"
        assert "channel" in df.columns, (
            "channel dimension should become DataFrame column"
        )

        # Assert - DataFrame should have one row per (date, channel) combination
        # Verify this by checking that date x channel count equals total rows
        expected_rows = df["date"].nunique() * df["channel"].nunique()
        assert len(df) == expected_rows, (
            f"Expected {expected_rows} rows (date x channel), got {len(df)}"
        )
        # Also verify we have reasonable number of rows (52 dates x 3 channels = 156)
        assert len(df) > 0, "DataFrame should not be empty"

    def test_coordinates_become_values(self, mock_mmm_idata_wrapper, simple_channels):
        """Test that xarray coordinates become DataFrame column values."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Act
        df = create_contribution_summary(data=mock_mmm_idata_wrapper)

        # Assert - DataFrame should contain the expected number of unique channels
        # (the fixture has 3 channels: TV, Radio, Social)
        assert len(df["channel"].unique()) == len(simple_channels), (
            f"Expected {len(simple_channels)} unique channels, "
            f"got {len(df['channel'].unique())}"
        )

    def test_mean_and_median_computed_correctly(self, mock_mmm_idata_wrapper):
        """Test that mean and median are computed correctly from MCMC samples."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Arrange - get raw xarray data
        contributions = mock_mmm_idata_wrapper.get_channel_contributions()

        # Compute expected mean/median for first date-channel combination
        first_date = contributions.coords["date"].values[0]
        first_channel_idx = 0  # First channel index
        # Use isel for positional indexing
        samples = contributions.isel(date=0, channel=first_channel_idx).values
        expected_mean = np.mean(samples)
        expected_median = np.median(samples)

        # Act
        df = create_contribution_summary(data=mock_mmm_idata_wrapper)

        # Get actual values from DataFrame - filter by date and first channel
        # Use the first row for the first date (dates are sorted)
        first_date_df = df[df["date"] == pd.Timestamp(first_date)]
        # Get the first channel row
        first_row = first_date_df.iloc[0]
        actual_mean = first_row["mean"]
        actual_median = first_row["median"]

        # Assert
        assert np.isclose(actual_mean, expected_mean, rtol=1e-6), (
            f"Expected mean {expected_mean}, got {actual_mean}"
        )
        assert np.isclose(actual_median, expected_median, rtol=1e-6), (
            f"Expected median {expected_median}, got {actual_median}"
        )


# ============================================================================
# Category 6: Integration with Component 1 (MMMIDataWrapper)
# ============================================================================


class TestComponent1Integration:
    """Test Component 2 correctly consumes Component 1's filtered/aggregated data."""

    def test_filtered_dates_produce_filtered_dataframe(self, mock_mmm_idata_wrapper):
        """Test that Component 1 date filtering is preserved in output."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Arrange - filter to specific date range
        start_date = "2024-02-01"
        end_date = "2024-03-31"
        filtered_data = mock_mmm_idata_wrapper.filter_dates(start_date, end_date)

        # Act
        df = create_contribution_summary(data=filtered_data)

        # Assert - DataFrame should only contain filtered dates
        assert df["date"].min() >= pd.Timestamp(start_date), (
            f"DataFrame contains dates before {start_date}"
        )
        assert df["date"].max() <= pd.Timestamp(end_date), (
            f"DataFrame contains dates after {end_date}"
        )

    def test_filtered_channels_produce_filtered_dataframe(self, mock_mmm_idata_wrapper):
        """Test that Component 1 channel filtering is preserved in output."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Arrange - filter to specific channels (first 2 channels)
        all_channels = mock_mmm_idata_wrapper.channels
        selected_channels = all_channels[:2]
        filtered_data = mock_mmm_idata_wrapper.filter_dims(channel=selected_channels)

        # Act
        df = create_contribution_summary(data=filtered_data)

        # Assert - DataFrame should have fewer unique channels than original
        original_df = create_contribution_summary(data=mock_mmm_idata_wrapper)
        assert len(df["channel"].unique()) == len(selected_channels), (
            f"Expected {len(selected_channels)} unique channels, "
            f"got {len(df['channel'].unique())}"
        )
        assert len(df["channel"].unique()) < len(original_df["channel"].unique()), (
            "Filtered DataFrame should have fewer channels than original"
        )

    def test_aggregated_time_produces_aggregated_dataframe(
        self, mock_mmm_idata_wrapper
    ):
        """Test that Component 1 time aggregation is preserved in output."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Arrange - aggregate to monthly
        aggregated_data = mock_mmm_idata_wrapper.aggregate_time("monthly")

        # Act
        df = create_contribution_summary(data=aggregated_data)

        # Assert - fewer dates than original (due to aggregation)
        original_df = create_contribution_summary(data=mock_mmm_idata_wrapper)
        assert df["date"].nunique() < original_df["date"].nunique(), (
            "Aggregated DataFrame should have fewer unique dates"
        )

    def test_frequency_parameter_delegates_to_component1(self, mock_mmm_idata_wrapper):
        """Test that frequency parameter triggers time aggregation via Component 1."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Act - use frequency parameter (Component 2 API)
        df_monthly = create_contribution_summary(
            data=mock_mmm_idata_wrapper,
            frequency="monthly",
        )
        df_original = create_contribution_summary(
            data=mock_mmm_idata_wrapper,
            frequency="original",  # No aggregation
        )

        # Assert - monthly should have fewer dates
        assert df_monthly["date"].nunique() < df_original["date"].nunique(), (
            "frequency='monthly' should produce fewer unique dates than 'original'"
        )


# ============================================================================
# Category 7: MMMSummaryFactory Class
# ============================================================================


class TestMMMSummaryFactory:
    """Test MMMSummaryFactory convenience wrapper."""

    def test_factory_class_initialization(self, mock_mmm_idata_wrapper):
        """Test that MMMSummaryFactory can be instantiated."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        # Act
        factory = MMMSummaryFactory(data=mock_mmm_idata_wrapper)

        # Assert
        assert factory is not None
        assert factory.data is mock_mmm_idata_wrapper
        assert factory.hdi_probs == [0.94], "Default hdi_probs should be [0.94]"
        assert factory.output_format == "pandas", (
            "Default output_format should be 'pandas'"
        )

    def test_factory_class_custom_defaults(self, mock_mmm_idata_wrapper):
        """Test that MMMSummaryFactory stores custom defaults."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        # Act
        factory = MMMSummaryFactory(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.80, 0.94],
            output_format="pandas",  # Use pandas since polars might not be installed
        )

        # Assert
        assert factory.hdi_probs == [0.80, 0.94]
        assert factory.output_format == "pandas"

    def test_factory_methods_exist(self, mock_mmm_idata_wrapper):
        """Test that MMMSummaryFactory has all expected methods."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        factory = MMMSummaryFactory(data=mock_mmm_idata_wrapper)

        # Assert all methods exist and are callable
        assert hasattr(factory, "posterior_predictive")
        assert callable(factory.posterior_predictive)

        assert hasattr(factory, "contributions")
        assert callable(factory.contributions)

        assert hasattr(factory, "roas")
        assert callable(factory.roas)

        assert hasattr(factory, "channel_spend")
        assert callable(factory.channel_spend)

        assert hasattr(factory, "saturation_curves")
        assert callable(factory.saturation_curves)

        assert hasattr(factory, "decay_curves")
        assert callable(factory.decay_curves)

        assert hasattr(factory, "total_contribution")
        assert callable(factory.total_contribution)

        assert hasattr(factory, "period_over_period")
        assert callable(factory.period_over_period)

    def test_factory_methods_use_stored_defaults(self, mock_mmm_idata_wrapper):
        """Test that factory methods use stored defaults."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        # Arrange - create factory with custom defaults
        factory = MMMSummaryFactory(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.80, 0.94],
            output_format="pandas",
        )

        # Act - call method without overrides
        df = factory.contributions()

        # Assert - should use factory defaults
        assert isinstance(df, pd.DataFrame), (
            "Factory method should use stored output_format='pandas'"
        )
        assert "abs_error_80_lower" in df.columns, (
            "Factory method should use stored hdi_probs=[0.80, 0.94]"
        )
        assert "abs_error_94_lower" in df.columns, (
            "Factory method should use stored hdi_probs=[0.80, 0.94]"
        )

    def test_factory_methods_allow_overrides(self, mock_mmm_idata_wrapper):
        """Test that factory methods allow overriding defaults per call."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        # Arrange - create factory with defaults
        factory = MMMSummaryFactory(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.80],
            output_format="pandas",
        )

        # Act - override to different hdi_probs for this call
        df = factory.contributions(
            hdi_probs=[0.94],
            output_format="pandas",
        )

        # Assert - should use overridden values
        assert isinstance(df, pd.DataFrame), (
            "Override output_format='pandas' should return pandas DataFrame"
        )
        assert "abs_error_94_lower" in df.columns, (
            "Override hdi_probs=[0.94] should be used"
        )
        assert "abs_error_80_lower" not in df.columns, (
            "Should not use factory default hdi_probs=[0.80]"
        )

    def test_factory_delegates_to_factory_functions(
        self, mock_mmm_idata_wrapper, monkeypatch
    ):
        """Test that factory methods delegate to factory functions."""
        from pymc_marketing.mmm import summary
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        # Arrange - mock the factory function
        mock_called = {"called": False, "data": None, "kwargs": None}

        def mock_create_contribution_summary(data, **kwargs):
            mock_called["called"] = True
            mock_called["data"] = data
            mock_called["kwargs"] = kwargs
            return pd.DataFrame({"channel": ["TV"], "mean": [100.0]})

        monkeypatch.setattr(
            summary, "create_contribution_summary", mock_create_contribution_summary
        )

        # Act
        factory = MMMSummaryFactory(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.94],
        )
        factory.contributions()

        # Assert - factory function was called with correct arguments
        assert mock_called["called"], "Factory function should be called"
        assert mock_called["data"] is mock_mmm_idata_wrapper
        assert mock_called["kwargs"]["hdi_probs"] == [0.94]


# ============================================================================
# Category 8: Additional Summary Functions
# ============================================================================


class TestAdditionalSummaryFunctions:
    """Test additional summary functions beyond the core 4."""

    def test_saturation_curves_schema(self, mock_mmm_idata_wrapper):
        """Test saturation curves summary returns DataFrame with correct schema."""
        from pymc_marketing.mmm.summary import create_saturation_curves

        # Act
        df = create_saturation_curves(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.94],
        )

        # Assert - required columns
        required_columns = {"x", "channel", "mean", "median"}
        assert required_columns.issubset(set(df.columns)), (
            f"Missing required columns. Expected {required_columns}, got {set(df.columns)}"
        )

        # Assert - HDI columns
        assert "abs_error_94_lower" in df.columns
        assert "abs_error_94_upper" in df.columns

        # Assert - x values are numeric
        assert pd.api.types.is_numeric_dtype(df["x"]), (
            f"x column should be numeric, got {df['x'].dtype}"
        )

    def test_decay_curves_schema(self, mock_mmm_idata_wrapper):
        """Test decay curves summary returns DataFrame with correct schema."""
        from pymc_marketing.mmm.summary import create_decay_curves

        # Act
        df = create_decay_curves(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.94],
        )

        # Assert - required columns
        required_columns = {"time", "channel", "mean", "median"}
        assert required_columns.issubset(set(df.columns)), (
            f"Missing required columns. Expected {required_columns}, got {set(df.columns)}"
        )

        # Assert - time values are numeric (lag periods)
        assert pd.api.types.is_numeric_dtype(df["time"]), (
            f"time column should be numeric, got {df['time'].dtype}"
        )

    def test_total_contribution_summary_schema(self, mock_mmm_idata_wrapper):
        """Test total contribution summary combines all effect types."""
        from pymc_marketing.mmm.summary import create_total_contribution_summary

        # Act
        df = create_total_contribution_summary(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.94],
        )

        # Assert - required columns (even if empty, should have correct schema)
        required_columns = {"date", "component", "mean", "median"}
        assert required_columns.issubset(set(df.columns)), (
            f"Missing required columns. Expected {required_columns}, got {set(df.columns)}"
        )

        # If DataFrame has data, it should have valid component values
        if len(df) > 0:
            components = set(df["component"].unique())
            assert len(components) > 0, "Should have at least one component type"

    def test_period_over_period_summary_schema(self, mock_mmm_idata_wrapper):
        """Test period-over-period summary returns percentage changes."""
        from pymc_marketing.mmm.summary import create_period_over_period_summary

        # Act
        df = create_period_over_period_summary(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.94],
        )

        # Assert - required columns
        required_columns = {"channel", "pct_change_mean", "pct_change_median"}
        assert required_columns.issubset(set(df.columns)), (
            f"Missing required columns. Expected {required_columns}, got {set(df.columns)}"
        )


# ============================================================================
# Category 9: Edge Cases & Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_date_range_raises_or_returns_empty(self, mock_mmm_idata_wrapper):
        """Test behavior when filtered data has no dates."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Arrange - filter to impossible date range
        filtered_data = mock_mmm_idata_wrapper.filter_dates("2030-01-01", "2030-01-02")

        # Act & Assert - should either raise or return empty DataFrame
        try:
            df = create_contribution_summary(data=filtered_data)
            # If returns DataFrame, it should be empty
            assert len(df) == 0, "Empty date range should produce empty DataFrame"
        except ValueError as e:
            # Or it can raise a clear error
            assert "empty" in str(e).lower() or "no data" in str(e).lower()

    @pytest.mark.parametrize(
        "invalid_hdi_prob",
        [1.5, 0.0, -0.5, 100],
        ids=["greater_than_1", "zero", "negative", "percentage_format"],
    )
    def test_invalid_hdi_prob_raises(self, mock_mmm_idata_wrapper, invalid_hdi_prob):
        """Test that invalid HDI probabilities raise ValueError."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Act & Assert
        with pytest.raises(ValueError, match=r"[Hh][Dd][Ii]|probability"):
            create_contribution_summary(
                data=mock_mmm_idata_wrapper,
                hdi_probs=[invalid_hdi_prob],
            )

    def test_division_by_zero_in_roas_handled(
        self, mock_mmm_idata_wrapper_with_zero_spend
    ):
        """Test that ROAS computation handles zero spend without errors."""
        from pymc_marketing.mmm.summary import create_roas_summary

        # Act - should not raise ZeroDivisionError
        df = create_roas_summary(data=mock_mmm_idata_wrapper_with_zero_spend)

        # Assert - rows with zero spend should have ROAS = 0, NaN, or inf
        zero_spend_rows = df[df["channel"] == "zero_spend_channel"]
        assert (
            zero_spend_rows["mean"].isna().all()
            or (zero_spend_rows["mean"] == 0).all()
            or np.isinf(zero_spend_rows["mean"]).all()
        ), "Zero spend should produce NaN, 0, or inf ROAS, not error"

    def test_wrapper_without_schema_still_works(self, mock_mmm_idata_wrapper):
        """Test that summary functions work when wrapper has no schema."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Arrange - wrapper was created with schema=None
        assert mock_mmm_idata_wrapper.schema is None, (
            "Test fixture should have no schema"
        )

        # Act - should not raise
        df = create_contribution_summary(data=mock_mmm_idata_wrapper)

        # Assert - got valid DataFrame
        assert len(df) > 0, "Should return non-empty DataFrame"
        assert "channel" in df.columns
        assert "mean" in df.columns
