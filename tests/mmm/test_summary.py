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

from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper

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
            "create_total_contribution_summary",
            "create_period_over_period_summary",
        ],
    )
    def test_all_factory_functions_support_output_format(
        self, mock_mmm_idata_wrapper, factory_function_name
    ):
        """Test that data-only factory functions accept output_format parameter.

        Note: create_saturation_curves and create_adstock_curves are excluded
        as they require a fitted model, not just a data wrapper.
        """
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

    @pytest.mark.parametrize(
        "factory_function_name",
        [
            "create_saturation_curves",
            "create_adstock_curves",
        ],
    )
    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_model_factory_functions_support_output_format(
        self, factory_function_name, fitted_mmm, request
    ):
        """Test that model-requiring factory functions accept output_format parameter."""
        from pymc_marketing.mmm import summary

        mmm = request.getfixturevalue(fitted_mmm)
        factory_func = getattr(summary, factory_function_name)

        # Act - these functions take model as first arg
        try:
            df = factory_func(model=mmm, output_format="pandas")
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

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_saturation_curves_schema(self, fitted_mmm, request):
        """Test saturation curves summary returns DataFrame with correct schema."""
        from pymc_marketing.mmm.summary import create_saturation_curves

        mmm = request.getfixturevalue(fitted_mmm)
        # Act - saturation curves require a fitted model
        df = create_saturation_curves(
            model=mmm,
            hdi_probs=[0.94],
        )

        # Assert - required columns (channel may not be present with default priors)
        required_columns = {"x", "mean", "median"}
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

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_decay_curves_schema(self, fitted_mmm, request):
        """Test decay curves summary returns DataFrame with correct schema."""
        from pymc_marketing.mmm.summary import create_decay_curves

        mmm = request.getfixturevalue(fitted_mmm)
        # Act - decay curves require a fitted model
        df = create_decay_curves(
            model=mmm,
            hdi_probs=[0.94],
        )

        # Assert - required columns (channel may not be present with default priors)
        required_columns = {"time", "mean", "median"}
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


# ============================================================================
# Category 10: MMMSummaryFactory Method Coverage
# ============================================================================


class TestMMMSummaryFactoryMethodCoverage:
    """Test all MMMSummaryFactory methods to ensure full coverage."""

    def test_factory_posterior_predictive_method(self, mock_mmm_idata_wrapper):
        """Test that factory posterior_predictive method works correctly."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        factory = MMMSummaryFactory(data=mock_mmm_idata_wrapper, hdi_probs=[0.94])

        # Act
        df = factory.posterior_predictive()

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns
        assert "mean" in df.columns
        assert "observed" in df.columns
        assert "abs_error_94_lower" in df.columns

    def test_factory_roas_method(self, mock_mmm_idata_wrapper):
        """Test that factory roas method works correctly."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        factory = MMMSummaryFactory(data=mock_mmm_idata_wrapper, hdi_probs=[0.94])

        # Act
        df = factory.roas()

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns
        assert "channel" in df.columns
        assert "mean" in df.columns
        assert "abs_error_94_lower" in df.columns

    def test_factory_channel_spend_method(self, mock_mmm_idata_wrapper):
        """Test that factory channel_spend method works correctly."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        factory = MMMSummaryFactory(data=mock_mmm_idata_wrapper)

        # Act
        df = factory.channel_spend()

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns
        assert "channel" in df.columns
        assert "channel_data" in df.columns

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_factory_saturation_curves_method(self, fitted_mmm, request):
        """Test that factory saturation_curves method works correctly."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        mmm = request.getfixturevalue(fitted_mmm)
        # saturation_curves requires model
        factory = MMMSummaryFactory(data=mmm.data, model=mmm, hdi_probs=[0.94])

        # Act
        df = factory.saturation_curves(n_points=50)

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert "x" in df.columns
        assert "mean" in df.columns

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_factory_decay_curves_method(self, fitted_mmm, request):
        """Test that factory decay_curves method works correctly."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        # decay_curves requires model
        mmm = request.getfixturevalue(fitted_mmm)
        factory = MMMSummaryFactory(data=mmm.data, model=mmm, hdi_probs=[0.94])

        # Act
        df = factory.decay_curves(max_lag=10)

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert "time" in df.columns
        assert "mean" in df.columns

    def test_factory_total_contribution_method(self, mock_mmm_idata_wrapper):
        """Test that factory total_contribution method works correctly."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        factory = MMMSummaryFactory(data=mock_mmm_idata_wrapper, hdi_probs=[0.94])

        # Act
        df = factory.total_contribution()

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns
        assert "component" in df.columns
        assert "mean" in df.columns

    def test_factory_period_over_period_method(self, mock_mmm_idata_wrapper):
        """Test that factory period_over_period method works correctly."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        factory = MMMSummaryFactory(data=mock_mmm_idata_wrapper, hdi_probs=[0.94])

        # Act
        df = factory.period_over_period()

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert "channel" in df.columns
        assert "pct_change_mean" in df.columns

    def test_factory_methods_with_frequency_parameter(self, mock_mmm_idata_wrapper):
        """Test factory methods with frequency parameter."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        factory = MMMSummaryFactory(data=mock_mmm_idata_wrapper)

        # Test methods that support frequency
        df_pp = factory.posterior_predictive(frequency="monthly")
        df_roas = factory.roas(frequency="monthly")
        df_total = factory.total_contribution(frequency="monthly")

        # All should return DataFrames with fewer unique dates than original
        original_pp = factory.posterior_predictive()

        assert df_pp["date"].nunique() < original_pp["date"].nunique()
        assert isinstance(df_roas, pd.DataFrame)
        assert isinstance(df_total, pd.DataFrame)

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_factory_methods_with_output_format_override(
        self, mock_mmm_idata_wrapper, fitted_mmm, request
    ):
        """Test that factory methods correctly override output_format."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        mmm = request.getfixturevalue(fitted_mmm)
        # Data-only factory for methods that don't need model
        data_factory = MMMSummaryFactory(
            data=mock_mmm_idata_wrapper,
            output_format="pandas",
        )

        # All data-only methods should return pandas when explicitly overridden
        assert isinstance(
            data_factory.posterior_predictive(output_format="pandas"), pd.DataFrame
        )
        assert isinstance(data_factory.roas(output_format="pandas"), pd.DataFrame)
        assert isinstance(
            data_factory.channel_spend(output_format="pandas"), pd.DataFrame
        )
        assert isinstance(
            data_factory.total_contribution(output_format="pandas"), pd.DataFrame
        )
        assert isinstance(
            data_factory.period_over_period(output_format="pandas"), pd.DataFrame
        )

        # Factory with model for curve methods
        model_factory = MMMSummaryFactory(
            data=mmm.data,
            model=mmm,
            output_format="pandas",
        )

        # Curve methods require model
        assert isinstance(
            model_factory.saturation_curves(output_format="pandas"), pd.DataFrame
        )
        assert isinstance(
            model_factory.decay_curves(output_format="pandas"), pd.DataFrame
        )


# ============================================================================
# Category 11: Non-Channel Component Coverage
# ============================================================================


@pytest.fixture
def mock_mmm_idata_wrapper_with_controls(simple_dates, simple_channels):
    """Mock MMMIDataWrapper with control variables for testing non-channel components."""
    local_rng = np.random.default_rng(seed=44)
    controls = ["price", "promo"]

    # Create mock InferenceData with control contributions
    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    local_rng.normal(loc=1000, scale=100, size=(2, 10, 52, 3)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={"date": simple_dates, "channel": simple_channels},
                ),
                "control_contribution": xr.DataArray(
                    local_rng.normal(loc=200, scale=50, size=(2, 10, 52, 2)),
                    dims=("chain", "draw", "date", "control"),
                    coords={"date": simple_dates, "control": controls},
                ),
                "intercept": xr.DataArray(
                    local_rng.normal(loc=3000, scale=100, size=(2, 10)),
                    dims=("chain", "draw"),
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
                "control_data": xr.DataArray(
                    local_rng.uniform(0, 10, size=(52, 2)),
                    dims=("date", "control"),
                    coords={"date": simple_dates, "control": controls},
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


class TestNonChannelComponents:
    """Test contribution summary for non-channel components."""

    def test_contribution_summary_control_component(
        self, mock_mmm_idata_wrapper_with_controls
    ):
        """Test contribution summary for control component."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Act
        df = create_contribution_summary(
            data=mock_mmm_idata_wrapper_with_controls,
            component="control",
            hdi_probs=[0.94],
        )

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert "control" in df.columns
        assert "mean" in df.columns
        assert "date" in df.columns
        assert len(df["control"].unique()) == 2  # price and promo

    def test_contribution_summary_baseline_component_raises_when_missing(
        self, mock_mmm_idata_wrapper_with_controls
    ):
        """Test contribution summary for baseline component raises when not present."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # The fixture doesn't have a proper baseline contribution
        # so this should raise ValueError
        with pytest.raises(ValueError, match=r"No baseline contributions found"):
            create_contribution_summary(
                data=mock_mmm_idata_wrapper_with_controls,
                component="baseline",
                hdi_probs=[0.94],
            )

    def test_contribution_summary_missing_component_raises(
        self, mock_mmm_idata_wrapper
    ):
        """Test that requesting missing component raises ValueError."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # The mock fixture doesn't have control_contribution
        with pytest.raises(ValueError, match=r"No control contributions found"):
            create_contribution_summary(
                data=mock_mmm_idata_wrapper,
                component="control",
            )

    def test_total_contribution_summary_with_multiple_components(
        self, mock_mmm_idata_wrapper
    ):
        """Test total contribution summary includes multiple component types."""
        from pymc_marketing.mmm.summary import create_total_contribution_summary

        # Act - use the basic fixture which has channel contributions
        df = create_total_contribution_summary(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.94],
        )

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert "component" in df.columns
        # Should have at least channel contributions
        if len(df) > 0:
            # All rows should have mean/median computed
            assert not df["mean"].isna().any()
            assert "date" in df.columns


# ============================================================================
# Category 12: Additional Edge Cases and Path Coverage
# ============================================================================


@pytest.fixture
def mock_mmm_idata_wrapper_with_total_contributions(simple_dates, simple_channels):
    """Mock MMMIDataWrapper with contributions that work with create_total_contribution_summary."""
    local_rng = np.random.default_rng(seed=45)

    # Create mock InferenceData
    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    local_rng.normal(loc=1000, scale=100, size=(2, 10, 52, 3)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={"date": simple_dates, "channel": simple_channels},
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
            }
        ),
    )

    return MMMIDataWrapper(idata, schema=None, validate_on_init=False)


class TestAdditionalPathCoverage:
    """Additional tests for complete path coverage."""

    def test_total_contribution_summary_with_mocked_contributions(
        self, mock_mmm_idata_wrapper_with_total_contributions, simple_dates
    ):
        """Test total contribution summary when contributions have proper data_vars."""
        from pymc_marketing.mmm.summary import create_total_contribution_summary

        # Create mock contributions Dataset with proper data_vars
        # The issue is that when key name matches dimension name, xarray puts it in coords
        # So we need to patch get_contributions to return a properly named Dataset
        local_rng = np.random.default_rng(seed=46)

        mock_contributions = xr.Dataset(
            {
                # Use different names from dimensions to ensure they're data_vars
                "channel_contribution": xr.DataArray(
                    local_rng.normal(loc=1000, scale=100, size=(2, 10, 52, 3)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={"date": simple_dates, "channel": ["TV", "Radio", "Social"]},
                ),
            }
        )

        # Patch get_contributions
        wrapper = mock_mmm_idata_wrapper_with_total_contributions
        original_get_contributions = wrapper.get_contributions

        def mock_get_contributions(*args, **kwargs):
            return mock_contributions

        wrapper.get_contributions = mock_get_contributions

        # Act
        df = create_total_contribution_summary(data=wrapper, hdi_probs=[0.94])

        # Restore
        wrapper.get_contributions = original_get_contributions

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "component" in df.columns
        assert "mean" in df.columns
        assert "date" in df.columns
        # Should have channel_contribution as a component
        assert "channel_contribution" in df["component"].values

    def test_hdi_probs_default_none(self, mock_mmm_idata_wrapper):
        """Test that hdi_probs=None uses default [0.94]."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Act - explicitly pass None for hdi_probs
        df = create_contribution_summary(
            data=mock_mmm_idata_wrapper,
            hdi_probs=None,  # Should default to [0.94]
        )

        # Assert - should have 94% HDI columns
        assert "abs_error_94_lower" in df.columns
        assert "abs_error_94_upper" in df.columns

    def test_frequency_none_means_original(self, mock_mmm_idata_wrapper):
        """Test that frequency=None means no aggregation (same as 'original')."""
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Act
        df_none = create_contribution_summary(
            data=mock_mmm_idata_wrapper,
            frequency=None,
        )
        df_original = create_contribution_summary(
            data=mock_mmm_idata_wrapper,
            frequency="original",
        )

        # Assert - same number of dates
        assert df_none["date"].nunique() == df_original["date"].nunique()

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    @pytest.mark.parametrize("n_points", [25, 50, 100])
    def test_saturation_curves_custom_n_points(self, fitted_mmm, n_points, request):
        """Test saturation curves with custom n_points parameter."""
        from pymc_marketing.mmm.summary import create_saturation_curves

        mmm = request.getfixturevalue(fitted_mmm)
        # Act - saturation curves require a fitted model
        df = create_saturation_curves(
            model=mmm,
            n_points=n_points,
            hdi_probs=[0.80],
        )

        # Assert - should have n_points rows per channel (and per custom dim if present)
        if "channel" in df.columns:
            n_channels = df["channel"].nunique()
        else:
            n_channels = 1

        # Check for custom dimensions (e.g., country in panel model)
        excluded_cols = [
            "x",
            "mean",
            "median",
            "abs_error_80_lower",
            "abs_error_80_upper",
        ]
        custom_dims = [col for col in df.columns if col not in excluded_cols]
        if custom_dims:
            n_custom = 1
            for dim in custom_dims:
                n_custom *= df[dim].nunique()
        else:
            n_custom = 1

        expected_rows = n_points * n_channels * n_custom
        assert len(df) == expected_rows, (
            f"Expected {expected_rows} rows ({n_points} x {n_channels} x {n_custom}), "
            f"got {len(df)}"
        )
        assert "abs_error_80_lower" in df.columns

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_decay_curves_custom_max_lag(self, fitted_mmm, request):
        """Test decay curves with custom max_lag parameter."""
        from pymc_marketing.mmm.summary import create_decay_curves

        mmm = request.getfixturevalue(fitted_mmm)
        # Act - decay curves require a fitted model
        max_lag = 3
        df = create_decay_curves(
            model=mmm,
            max_lag=max_lag,
            hdi_probs=[0.80],
        )

        # Assert - should have lags 0 to max_lag (max_lag + 1 values) per channel
        assert df["time"].max() == max_lag
        assert len(df["time"].unique()) == max_lag + 1
        assert "abs_error_80_lower" in df.columns

    def test_period_over_period_with_multiple_hdi(self, mock_mmm_idata_wrapper):
        """Test period over period summary with multiple HDI probs."""
        from pymc_marketing.mmm.summary import create_period_over_period_summary

        # Act
        df = create_period_over_period_summary(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.80, 0.94],
        )

        # Assert
        assert "abs_error_80_lower" in df.columns
        assert "abs_error_94_lower" in df.columns

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_default_hdi_probs_applied_in_all_functions(
        self, mock_mmm_idata_wrapper, fitted_mmm, request
    ):
        """Test that default HDI probs [0.94] are applied when None is passed."""
        from pymc_marketing.mmm.summary import (
            create_adstock_curves,
            create_period_over_period_summary,
            create_saturation_curves,
        )

        mmm = request.getfixturevalue(fitted_mmm)
        # Saturation and adstock curves require a fitted model
        df_sat = create_saturation_curves(mmm, hdi_probs=None)
        assert "abs_error_94_lower" in df_sat.columns

        df_adstock = create_adstock_curves(mmm, hdi_probs=None)
        assert "abs_error_94_lower" in df_adstock.columns

        # Period over period uses data wrapper
        df_pop = create_period_over_period_summary(
            mock_mmm_idata_wrapper, hdi_probs=None
        )
        assert "abs_error_94_lower" in df_pop.columns


# ============================================================================
# Category 13: TDD Extensions - Validation Decorator
# ============================================================================


class TestValidationDecorator:
    """Test validation decorator is applied to all factory functions."""

    @pytest.mark.parametrize(
        "factory_func_name",
        [
            "create_posterior_predictive_summary",
            "create_contribution_summary",
            "create_roas_summary",
            "create_channel_spend_dataframe",
            "create_total_contribution_summary",
        ],
    )
    def test_factory_function_calls_validate(
        self, factory_func_name, mock_mmm_idata_wrapper
    ):
        """Test that factory function validates idata before processing."""
        from unittest.mock import Mock

        from pymc_marketing.mmm import summary

        factory_func = getattr(summary, factory_func_name)

        # Create a mock wrapper that tracks validate_or_raise calls
        mock_wrapper = Mock(spec=MMMIDataWrapper)
        mock_wrapper.idata = mock_mmm_idata_wrapper.idata
        mock_wrapper.schema = Mock()
        mock_wrapper.validate_or_raise = Mock()

        # Mock the wrapper methods to return valid data
        mock_wrapper.get_target = Mock(
            return_value=mock_mmm_idata_wrapper.get_target(original_scale=True)
        )
        mock_wrapper.get_channel_contributions = Mock(
            return_value=mock_mmm_idata_wrapper.get_channel_contributions(
                original_scale=True
            )
        )
        mock_wrapper.get_channel_spend = Mock(
            return_value=mock_mmm_idata_wrapper.get_channel_spend()
        )
        mock_wrapper.get_contributions = Mock(
            return_value=mock_mmm_idata_wrapper.idata.posterior
        )
        mock_wrapper.channels = mock_mmm_idata_wrapper.channels

        try:
            factory_func(mock_wrapper)
        except Exception:  # noqa: S110
            # We only care if validate_or_raise was called, not if the function succeeds
            pass

        assert mock_wrapper.validate_or_raise.called, (
            f"{factory_func_name} should call validate_or_raise before processing"
        )

    def test_validation_fails_with_invalid_idata(self):
        """Test that invalid idata raises validation error."""
        from pymc_marketing.data.idata import MMMIdataSchema, MMMIDataWrapper
        from pymc_marketing.mmm.summary import create_contribution_summary

        # Create wrapper with invalid idata (missing required variables)
        invalid_idata = az.InferenceData(
            posterior=xr.Dataset()  # Empty - missing channel_contribution
        )

        schema = MMMIdataSchema.from_model_config(
            custom_dims=(),
            has_controls=False,
            has_seasonality=False,
            time_varying=False,
        )

        wrapper = MMMIDataWrapper(invalid_idata, schema=schema, validate_on_init=False)

        # Act & Assert
        with pytest.raises(ValueError, match="idata validation failed"):
            create_contribution_summary(wrapper)


# ============================================================================
# Category 14: TDD Extensions - Factory Refactoring
# ============================================================================


class TestMMMSummaryFactoryRefactoring:
    """Test MMMSummaryFactory refactoring to accept optional model parameter."""

    def test_factory_accepts_data_only(self, mock_mmm_idata_wrapper):
        """Test that MMMSummaryFactory works with data only."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        # Act
        factory = MMMSummaryFactory(mock_mmm_idata_wrapper)

        # Assert
        assert factory.data is mock_mmm_idata_wrapper
        assert factory.model is None

        # Data-only functions should work
        df = factory.contributions()
        assert df is not None

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_factory_accepts_data_and_model(self, fitted_mmm, request):
        """Test that MMMSummaryFactory accepts both data and model."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        mmm = request.getfixturevalue(fitted_mmm)
        # Get the data wrapper once to use for factory
        data = mmm.data

        # Act
        factory = MMMSummaryFactory(data, model=mmm)

        # Assert - data should be the same instance we passed
        assert factory.data is data
        assert factory.model is mmm
        assert hasattr(factory.model, "saturation")
        assert hasattr(factory.model, "adstock")

    def test_factory_errors_when_model_required_but_missing(
        self, mock_mmm_idata_wrapper
    ):
        """Test that functions needing model raise helpful error when model is None."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        factory = MMMSummaryFactory(mock_mmm_idata_wrapper)  # No model

        # Act & Assert - saturation_curves needs model
        with pytest.raises(
            ValueError,
            match=r"saturation_curves requires model.*MMMSummaryFactory\(data, model=mmm\)",
        ):
            factory.saturation_curves()

        # Act & Assert - adstock_curves needs model
        with pytest.raises(
            ValueError,
            match=r"adstock_curves requires model.*MMMSummaryFactory\(data, model=mmm\)",
        ):
            factory.adstock_curves()


# ============================================================================
# Category 15: TDD Extensions - Saturation & Adstock Curves (Parameterized)
# ============================================================================


class TestSaturationAndAdstockCurves:
    """Test saturation and adstock curves implementation (shared behavior)."""

    @pytest.mark.parametrize(
        "method_name,sample_method_name",
        [
            ("saturation_curves", "sample_saturation_curve"),
            ("adstock_curves", "sample_adstock_curve"),
        ],
        ids=["saturation", "adstock"],
    )
    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_factory_method_uses_model_sample_method(
        self, method_name, sample_method_name, fitted_mmm, request
    ):
        """Test that factory method delegates to MMM.sample_*_curve() method."""
        from unittest.mock import patch

        from pymc_marketing.mmm.summary import MMMSummaryFactory

        mmm = request.getfixturevalue(fitted_mmm)
        factory = MMMSummaryFactory(mmm.data, model=mmm)

        with patch.object(
            mmm, sample_method_name, wraps=getattr(mmm, sample_method_name)
        ) as mock_sample:
            # Act
            try:
                getattr(factory, method_name)()
            except Exception:  # noqa: S110
                # Method might fail if not implemented yet, but we're checking if it was called
                pass

            # Assert - sample method was called
            assert mock_sample.called, (
                f"{method_name} should delegate to MMM.{sample_method_name}()"
            )

    @pytest.mark.parametrize(
        "method_name",
        ["saturation_curves", "adstock_curves"],
        ids=["saturation", "adstock"],
    )
    def test_curves_preserve_custom_dims(self, panel_fitted_mmm, method_name):
        """Test that curve functions preserve custom dimensions."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        factory = MMMSummaryFactory(panel_fitted_mmm.data, model=panel_fitted_mmm)

        # Act
        df = getattr(factory, method_name)()

        # Assert - custom dimension column should be present
        assert "country" in df.columns, (
            f"Output of {method_name} should include 'country' column for panel MMM"
        )

        # Verify all countries are represented
        expected_countries = ["US", "UK"]  # From panel_mmm_data fixture
        actual_countries = sorted(df["country"].unique())
        assert actual_countries == sorted(expected_countries), (
            f"Expected countries {expected_countries}, got {actual_countries}"
        )

    @pytest.mark.parametrize(
        "method_name,x_col",
        [
            ("saturation_curves", "x"),
            ("adstock_curves", "time"),
        ],
        ids=["saturation", "adstock"],
    )
    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_curves_return_summary_stats(self, method_name, x_col, fitted_mmm, request):
        """Test that curve functions return summary statistics."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        mmm = request.getfixturevalue(fitted_mmm)
        factory = MMMSummaryFactory(mmm.data, model=mmm)

        # Act
        df = getattr(factory, method_name)(hdi_probs=[0.94])

        # Assert - required columns exist
        required_cols = {x_col, "mean", "median"}
        assert required_cols.issubset(set(df.columns)), (
            f"Missing required columns. Expected {required_cols}, got {set(df.columns)}"
        )

        # Assert - HDI columns exist
        assert "abs_error_94_lower" in df.columns
        assert "abs_error_94_upper" in df.columns

        # Assert - HDI bounds contain median
        for _, row in df.iterrows():
            lower = row["abs_error_94_lower"]
            upper = row["abs_error_94_upper"]
            median = row["median"]
            assert lower <= median <= upper, (
                f"Median {median} should be within HDI bounds [{lower}, {upper}]"
            )

    @pytest.mark.parametrize(
        "method_name",
        ["saturation_curves", "adstock_curves"],
        ids=["saturation", "adstock"],
    )
    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_curves_hdi_bounds_vary(self, method_name, fitted_mmm, request):
        """Test that curve HDI bounds vary (not constant placeholder values)."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        mmm = request.getfixturevalue(fitted_mmm)
        factory = MMMSummaryFactory(mmm.data, model=mmm)

        # Act
        df = getattr(factory, method_name)()

        # Assert - HDI bounds should vary
        lower_col = next(c for c in df.columns if "lower" in c)
        assert df[lower_col].std() > 0, (
            "HDI bounds should vary, got constant values (placeholder?)"
        )


# ============================================================================
# Category 16: TDD Extensions - Saturation Curves (Specific Tests)
# ============================================================================


class TestSaturationCurvesSpecific:
    """Test saturation-specific behavior."""

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_saturation_curves_are_increasing(self, fitted_mmm, request):
        """Test that saturation curves show increasing pattern."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        mmm = request.getfixturevalue(fitted_mmm)
        factory = MMMSummaryFactory(mmm.data, model=mmm)

        # Act
        df = factory.saturation_curves()
        if "channel" not in df.columns:
            df["channel"] = "channel"

        # Assert - curves should be increasing
        # Group by all dimension columns (channel, country, etc.)
        excluded_cols = ["x", "mean", "median"]
        dim_cols = [
            col
            for col in df.columns
            if col not in excluded_cols and not col.startswith("abs_error")
        ]
        if not dim_cols:
            dim_cols = ["channel"]

        for dims, group_df in df.groupby(dim_cols):
            group_sorted = group_df.sort_values("x")
            # Mean should increase with x (saturation property)
            assert group_sorted["mean"].iloc[-1] > group_sorted["mean"].iloc[0], (
                f"Saturation curve for {dims} should be increasing"
            )


# ============================================================================
# Category 17: TDD Extensions - Adstock Curves (Specific Tests)
# ============================================================================


class TestAdstockCurvesSpecific:
    """Test adstock-specific behavior."""

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_adstock_curves_are_decreasing(self, fitted_mmm, request):
        """Test that adstock curves show decreasing pattern over time."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        mmm = request.getfixturevalue(fitted_mmm)
        factory = MMMSummaryFactory(mmm.data, model=mmm)

        # Act
        df = factory.adstock_curves()

        if "channel" not in df.columns:
            df["channel"] = "channel"

        # Assert - adstock curves should show decreasing pattern
        # Group by all dimension columns (channel, country, etc.)
        excluded_cols = ["time", "mean", "median"]
        dim_cols = [
            col
            for col in df.columns
            if col not in excluded_cols and not col.startswith("abs_error")
        ]
        if not dim_cols:
            dim_cols = ["channel"]

        for dims, group_df in df.groupby(dim_cols):
            group_sorted = group_df.sort_values("time")
            # Mean should decrease over time (adstock property)
            first_value = group_sorted["mean"].iloc[0]
            last_value = group_sorted["mean"].iloc[-1]

            assert first_value > last_value, (
                f"Adstock curve for {dims} should decrease over time. "
                f"First: {first_value}, Last: {last_value}"
            )

            # First time point should be highest (immediate effect)
            assert group_sorted["mean"].iloc[0] == group_sorted["mean"].max(), (
                "Adstock effect should be strongest at time=0"
            )

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_adstock_curves_respects_max_lag(self, fitted_mmm, request):
        """Test that max_lag parameter controls adstock curve length."""
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        mmm = request.getfixturevalue(fitted_mmm)
        factory = MMMSummaryFactory(mmm.data, model=mmm)
        max_lag = 3

        # Act
        df = factory.adstock_curves(max_lag=max_lag)

        # Assert - rows = (max_lag + 1) x n_channels x n_custom_dims
        if "channel" in df.columns:
            n_channels = df["channel"].nunique()
        else:
            n_channels = 1
            df["channel"] = "channel"

        # Check for custom dimensions (e.g., country in panel model)
        excluded_cols = ["time", "channel", "mean", "median"]
        custom_dims = [
            col
            for col in df.columns
            if col not in excluded_cols and not col.startswith("abs_error")
        ]
        n_custom = 1
        for dim in custom_dims:
            n_custom *= df[dim].nunique()

        expected_rows = (max_lag + 1) * n_channels * n_custom

        assert len(df) == expected_rows, (
            f"Expected {expected_rows} rows "
            f"({max_lag + 1} lags x {n_channels} channels x {n_custom} custom dims), "
            f"got {len(df)}"
        )

        # Verify time values for each combination of dimensions
        excluded_cols = ["time", "mean", "median"]
        dim_cols = [
            col
            for col in df.columns
            if col not in excluded_cols and not col.startswith("abs_error")
        ]
        for dims, subset in df.groupby(dim_cols):
            time_values = sorted(subset["time"].unique())
            assert time_values == list(range(max_lag + 1)), (
                f"Time values for {dims} should be 0 to {max_lag}, got {time_values}"
            )


# ============================================================================
# Category 18: TDD Extensions - Change Over Time Implementation
# ============================================================================


class TestChangeOverTimeImplementation:
    """Test change over time (renamed from period_over_period) implementation."""

    def test_change_over_time_function_exists(self):
        """Test that create_change_over_time_summary exists."""
        from pymc_marketing.mmm.summary import create_change_over_time_summary

        assert callable(create_change_over_time_summary)

    def test_period_over_period_deprecated(self, mock_mmm_idata_wrapper):
        """Test that old period_over_period name is deprecated."""
        import warnings

        from pymc_marketing.mmm.summary import create_period_over_period_summary

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Call the deprecated function
            try:
                create_period_over_period_summary(mock_mmm_idata_wrapper)
            except Exception:  # noqa: S110
                # Function might fail if not fully implemented
                pass

            # Check for deprecation warning
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0, "Should raise DeprecationWarning"
            assert "create_change_over_time_summary" in str(
                deprecation_warnings[0].message
            )

    def test_change_over_time_output_schema(self, mock_mmm_idata_wrapper):
        """Test that change_over_time returns correct column schema."""
        from pymc_marketing.mmm.summary import create_change_over_time_summary

        # Act
        df = create_change_over_time_summary(mock_mmm_idata_wrapper, hdi_probs=[0.94])

        # Assert - required columns
        required_columns = {
            "date",
            "channel",
            "pct_change_mean",
            "pct_change_median",
        }
        assert required_columns.issubset(set(df.columns)), (
            f"Missing required columns. Expected {required_columns}, got {set(df.columns)}"
        )

        # Assert - HDI columns
        assert "abs_error_94_lower" in df.columns
        assert "abs_error_94_upper" in df.columns

        # Assert - fewer dates than original (first date excluded)
        df_dates = sorted(df["date"].unique())
        all_dates = sorted(mock_mmm_idata_wrapper.idata.posterior["date"].values)

        # Output should have one fewer date than input
        assert len(df_dates) == len(all_dates) - 1, (
            f"Expected {len(all_dates) - 1} dates (first excluded), got {len(df_dates)}"
        )

        # First output date should match second input date
        # Convert to same type for comparison
        first_output_date = pd.Timestamp(df_dates[0])
        second_input_date = pd.Timestamp(all_dates[1])
        assert first_output_date == second_input_date, (
            f"First output date {first_output_date} should equal second input date {second_input_date}"
        )

    def test_change_over_time_requires_date_dimension(self, mock_mmm_idata_wrapper):
        """Test that change_over_time raises error when date dimension is missing."""
        from pymc_marketing.mmm.summary import create_change_over_time_summary

        # Aggregate to all_time (removes date dimension)
        aggregated_data = mock_mmm_idata_wrapper.aggregate_time("all_time")

        # Act & Assert
        with pytest.raises(
            ValueError, match=r"change_over_time requires date dimension.*all_time"
        ):
            create_change_over_time_summary(aggregated_data)


# ============================================================================
# Category 19: TDD Extensions - MMM.summary Property Integration
# ============================================================================


class TestMMMSummaryProperty:
    """Test MMM model has summary property."""

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_mmm_has_summary_property(self, fitted_mmm, request):
        """Test that MMM model has summary property."""
        mmm = request.getfixturevalue(fitted_mmm)
        # Assert - property exists
        assert hasattr(mmm, "summary"), "MMM model should have .summary property"

        # Access property
        summary = mmm.summary

        # Assert - returns factory
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        assert isinstance(summary, MMMSummaryFactory), (
            f"summary property should return MMMSummaryFactory, got {type(summary)}"
        )

    @pytest.mark.parametrize(
        "method_name",
        [
            "posterior_predictive",
            "contributions",
            "roas",
            "channel_spend",
            "saturation_curves",
            "adstock_curves",
            "total_contribution",
            "change_over_time",
        ],
    )
    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_summary_property_has_method(self, method_name, fitted_mmm, request):
        """Test that summary property provides access to expected method."""
        mmm = request.getfixturevalue(fitted_mmm)
        summary = mmm.summary

        assert hasattr(summary, method_name), (
            f"summary should have {method_name} method"
        )
        assert callable(getattr(summary, method_name)), (
            f"summary.{method_name} should be callable"
        )

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_summary_property_returns_fresh_factory(self, fitted_mmm, request):
        """Test that summary property returns fresh factory on each access."""
        mmm = request.getfixturevalue(fitted_mmm)
        # Access property twice
        summary1 = mmm.summary
        summary2 = mmm.summary

        # Assert - different instances (following .data property pattern)
        assert summary1 is not summary2, (
            "summary property should return fresh factory on each access"
        )

        # But both should reference the same model
        assert summary1.model is mmm
        assert summary2.model is mmm

    def test_summary_validates_idata_exists(self):
        """Test that summary property validates idata exists."""
        from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
        from pymc_marketing.mmm.multidimensional import MMM

        # Create unfitted model (no idata) - using multidimensional.MMM which has summary property
        mmm = MMM(
            adstock=GeometricAdstock(l_max=10),
            saturation=LogisticSaturation(),
            date_column="date",
            target_column="target",  # Required for multidimensional MMM
            channel_columns=["TV", "Radio"],
        )

        # Act & Assert - accessing summary should raise error
        with pytest.raises(ValueError, match=r"idata does not exist"):
            _ = mmm.summary

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_summary_property_usage_example(self, fitted_mmm, request):
        """Test realistic usage of summary property."""
        mmm = request.getfixturevalue(fitted_mmm)
        # Get contributions summary
        contributions_df = mmm.summary.contributions()

        assert contributions_df is not None
        assert len(contributions_df) > 0
        assert "channel" in contributions_df.columns

        # Get ROAS summary
        roas_df = mmm.summary.roas()

        assert roas_df is not None
        assert len(roas_df) > 0

        # Get saturation curves
        saturation_df = mmm.summary.saturation_curves()

        assert saturation_df is not None
        assert len(saturation_df) > 0
        assert "x" in saturation_df.columns
