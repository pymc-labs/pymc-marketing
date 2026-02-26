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

CONSOLIDATED VERSION: Reduced from ~2300 lines to ~1300 lines by removing
redundant tests while maintaining full coverage.
"""

import importlib.util
from unittest.mock import Mock

import arviz as az
import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from pymc_marketing.data.idata import MMMIdataSchema, MMMIDataWrapper
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.mmm.summary import MMMSummaryFactory

# Seed for reproducibility
SEED = sum(map(ord, "summary_tests"))
rng = np.random.default_rng(seed=SEED)


# =============================================================================
# Fixtures
# =============================================================================


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
    """Mock MMMIDataWrapper with complete data for testing."""
    local_rng = np.random.default_rng(seed=42)

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

    return MMMIDataWrapper(idata, schema=None, validate_on_init=False)


@pytest.fixture
def mock_mmm_idata_wrapper_with_zero_spend(simple_dates):
    """Mock MMMIDataWrapper with zero spend channel for ROAS testing."""
    local_rng = np.random.default_rng(seed=43)
    channels = ["TV", "Radio", "zero_spend_channel"]

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


@pytest.fixture
def mock_panel_idata_wrapper(simple_dates, simple_channels):
    """Mock MMMIDataWrapper with panel data (custom dimensions) for testing."""
    local_rng = np.random.default_rng(seed=44)

    countries = ["US", "UK"]
    n_dates = len(simple_dates)
    n_channels = len(simple_channels)
    n_countries = len(countries)

    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    local_rng.normal(
                        loc=1000,
                        scale=100,
                        size=(2, 10, n_dates, n_countries, n_channels),
                    ),
                    dims=("chain", "draw", "date", "country", "channel"),
                    coords={
                        "date": simple_dates,
                        "country": countries,
                        "channel": simple_channels,
                    },
                ),
                "mu": xr.DataArray(
                    local_rng.normal(
                        loc=5000, scale=200, size=(2, 10, n_dates, n_countries)
                    ),
                    dims=("chain", "draw", "date", "country"),
                    coords={"date": simple_dates, "country": countries},
                ),
            }
        ),
        posterior_predictive=xr.Dataset(
            {
                "y": xr.DataArray(
                    local_rng.normal(
                        loc=5000, scale=200, size=(2, 10, n_dates, n_countries)
                    ),
                    dims=("chain", "draw", "date", "country"),
                    coords={"date": simple_dates, "country": countries},
                ),
            }
        ),
        fit_data=xr.Dataset(
            {
                "target": xr.DataArray(
                    local_rng.uniform(4000, 6000, size=(n_dates, n_countries)),
                    dims=("date", "country"),
                    coords={"date": simple_dates, "country": countries},
                ),
            }
        ),
        constant_data=xr.Dataset(
            {
                "channel_data": xr.DataArray(
                    local_rng.uniform(0, 100, size=(n_dates, n_countries, n_channels)),
                    dims=("date", "country", "channel"),
                    coords={
                        "date": simple_dates,
                        "country": countries,
                        "channel": simple_channels,
                    },
                ),
                "channel_scale": xr.DataArray(
                    [100.0, 50.0, 75.0],
                    dims=("channel",),
                    coords={"channel": simple_channels},
                ),
                "target_scale": xr.DataArray(500.0),
                "target_data": xr.DataArray(
                    local_rng.uniform(4000, 6000, size=(n_dates, n_countries)),
                    dims=("date", "country"),
                    coords={"date": simple_dates, "country": countries},
                ),
            }
        ),
    )

    return MMMIDataWrapper(idata, schema=None, validate_on_init=False)


@pytest.fixture
def mock_mmm_idata_wrapper_with_controls(simple_dates, simple_channels):
    """Mock MMMIDataWrapper with control variables for testing non-channel components."""
    local_rng = np.random.default_rng(seed=44)
    controls = ["price", "promo"]

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


# =============================================================================
# DataFrame Schema Tests
# =============================================================================


class TestDataFrameSchemas:
    """Test that returned DataFrames have correct structure and columns."""

    def test_posterior_predictive_summary_schema(self, mock_mmm_idata_wrapper):
        """Test posterior predictive summary returns DataFrame with correct schema."""
        df = MMMSummaryFactory(mock_mmm_idata_wrapper).posterior_predictive(
            hdi_probs=[0.94],
        )

        required_columns = {"date", "mean", "median", "observed"}
        assert required_columns.issubset(set(df.columns))
        assert "abs_error_94_lower" in df.columns
        assert "abs_error_94_upper" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert pd.api.types.is_float_dtype(df["mean"])

    def test_posterior_predictive_summary_panel_model(self, mock_panel_idata_wrapper):
        """Test posterior predictive summary correctly handles panel models."""
        df = MMMSummaryFactory(mock_panel_idata_wrapper).posterior_predictive(
            hdi_probs=[0.94],
        )

        pp_samples = mock_panel_idata_wrapper.idata.posterior_predictive["y"]
        expected_rows = pp_samples.sizes["date"] * pp_samples.sizes["country"]

        # Should have exactly date x country rows
        assert len(df) == expected_rows
        assert "country" in df.columns
        # Each (date, country) combination should appear exactly once
        date_country_counts = df.groupby(["date", "country"]).size()
        assert all(date_country_counts == 1)

    def test_contribution_summary_schema(self, mock_mmm_idata_wrapper):
        """Test contribution summary returns DataFrame with correct schema."""
        df = MMMSummaryFactory(mock_mmm_idata_wrapper).contributions(
            hdi_probs=[0.80, 0.94],
            component="channel",
        )

        required_columns = {"date", "channel", "mean", "median"}
        assert required_columns.issubset(set(df.columns))
        assert "abs_error_80_lower" in df.columns
        assert "abs_error_94_lower" in df.columns
        assert len(df["channel"].unique()) > 0

    def test_contribution_channel_column_contains_channel_names(
        self, mock_mmm_idata_wrapper, simple_channels
    ):
        """Test that channel column contains actual channel names, not integer indices"""

        df = MMMSummaryFactory(mock_mmm_idata_wrapper).contributions()

        # Channel column should contain string names, not integers
        channel_values = df["channel"].unique()
        assert set(channel_values) == set(simple_channels), (
            f"Expected channel names {simple_channels}, but got {list(channel_values)}. "
            "Channel coordinate values may have been replaced with integer indices."
        )

    def test_channel_spend_dataframe_schema(self, mock_mmm_idata_wrapper):
        """Test channel spend DataFrame has correct schema (no HDI columns)."""
        df = MMMSummaryFactory(mock_mmm_idata_wrapper).channel_spend()

        required_columns = {"date", "channel", "channel_data"}
        assert set(df.columns) == required_columns
        # NO HDI columns for raw data
        hdi_columns = [col for col in df.columns if "abs_error" in col]
        assert len(hdi_columns) == 0

    def test_total_contribution_summary_schema(self, mock_mmm_idata_wrapper):
        """Test total contribution summary has correct schema."""
        df = MMMSummaryFactory(mock_mmm_idata_wrapper).total_contribution(
            hdi_probs=[0.94],
        )

        required_columns = {"date", "component", "mean", "median"}
        assert required_columns.issubset(set(df.columns))

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_saturation_curves_schema(self, fitted_mmm, request):
        """Test saturation curves summary returns DataFrame with correct schema."""
        mmm = request.getfixturevalue(fitted_mmm)
        factory = MMMSummaryFactory(mmm.data, model=mmm)
        df = factory.saturation_curves(hdi_probs=[0.94])

        required_columns = {"x", "mean", "median"}
        assert required_columns.issubset(set(df.columns))
        assert "abs_error_94_lower" in df.columns
        assert pd.api.types.is_numeric_dtype(df["x"])

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_adstock_curves_schema(self, fitted_mmm, request):
        """Test adstock curves summary returns DataFrame with correct schema."""
        mmm = request.getfixturevalue(fitted_mmm)
        factory = MMMSummaryFactory(mmm.data, model=mmm)
        df = factory.adstock_curves(hdi_probs=[0.94])

        required_columns = {"time since exposure", "mean", "median"}
        assert required_columns.issubset(set(df.columns))
        assert pd.api.types.is_numeric_dtype(df["time since exposure"])

    def test_change_over_time_output_schema(self, mock_mmm_idata_wrapper):
        """Test that change_over_time returns correct column schema."""
        df = MMMSummaryFactory(mock_mmm_idata_wrapper).change_over_time(
            hdi_probs=[0.94]
        )

        required_columns = {"date", "channel", "pct_change_mean", "pct_change_median"}
        assert required_columns.issubset(set(df.columns))
        assert "abs_error_94_lower" in df.columns

        # Output should have one fewer date than input (first date excluded)
        all_dates = mock_mmm_idata_wrapper.idata.posterior["date"].values
        assert df["date"].nunique() == len(all_dates) - 1


# =============================================================================
# Output Format Tests
# =============================================================================


class TestOutputFormats:
    """Test output format parameter correctly controls DataFrame type."""

    def test_invalid_output_format_raises(self, mock_mmm_idata_wrapper):
        """Test that invalid output_format raises ValueError.

        Note: The @validate_call decorator on contributions() catches invalid
        output_format values via Pydantic validation before our custom error handler.
        """
        with pytest.raises(ValueError, match=r"'pandas' or 'polars'"):
            MMMSummaryFactory(mock_mmm_idata_wrapper).contributions(
                output_format="spark"
            )

    @pytest.mark.parametrize(
        "method_name",
        [
            "posterior_predictive",
            "channel_spend",
            "total_contribution",
            "contributions",
            "roas",
            "change_over_time",
        ],
    )
    def test_data_factory_methods_support_output_format(
        self, mock_mmm_idata_wrapper, method_name
    ):
        """Test that data-only factory methods accept output_format parameter."""
        factory = MMMSummaryFactory(mock_mmm_idata_wrapper)
        method = getattr(factory, method_name)

        # Act - should not raise TypeError for unexpected keyword argument
        # roas with data-only factory requires method="elementwise"
        kwargs = {"output_format": "pandas"}
        if method_name == "roas":
            kwargs["method"] = "elementwise"
        df = method(**kwargs)

        # Assert it returned a pandas DataFrame
        assert df is not None, f"{method_name} returned None"
        assert isinstance(df, pd.DataFrame), (
            f"{method_name} should return pandas DataFrame"
        )

    @pytest.mark.parametrize(
        "method_name",
        ["saturation_curves", "adstock_curves"],
    )
    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_model_factory_methods_support_output_format(
        self, method_name, fitted_mmm, request
    ):
        """Test that model-requiring factory methods accept output_format parameter."""
        mmm = request.getfixturevalue(fitted_mmm)
        factory = MMMSummaryFactory(mmm.data, model=mmm)
        method = getattr(factory, method_name)

        # Act
        df = method(output_format="pandas")

        # Assert it returned a pandas DataFrame
        assert df is not None, f"{method_name} returned None"
        assert isinstance(df, pd.DataFrame), (
            f"{method_name} should return pandas DataFrame"
        )

    @pytest.mark.skipif(
        not importlib.util.find_spec("polars"),
        reason="Polars not installed",
    )
    @pytest.mark.parametrize(
        "method_name",
        [
            "posterior_predictive",
            "channel_spend",
            "total_contribution",
            "contributions",
            "roas",
            "change_over_time",
        ],
    )
    def test_data_factory_methods_support_polars_output(
        self, mock_mmm_idata_wrapper, method_name
    ):
        """Test that data-only factory methods return polars when requested."""
        factory = MMMSummaryFactory(mock_mmm_idata_wrapper)
        method = getattr(factory, method_name)

        # Act - roas with data-only factory requires method="elementwise"
        kwargs = {"output_format": "polars"}
        if method_name == "roas":
            kwargs["method"] = "elementwise"
        df = method(**kwargs)

        # Assert it returned a polars DataFrame
        assert df is not None, f"{method_name} returned None"
        assert isinstance(df, pl.DataFrame), (
            f"{method_name} should return polars DataFrame when requested"
        )

    @pytest.mark.skipif(
        not importlib.util.find_spec("polars"),
        reason="Polars not installed",
    )
    @pytest.mark.parametrize(
        "method_name",
        ["saturation_curves", "adstock_curves"],
    )
    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_model_factory_methods_support_polars_output(
        self, method_name, fitted_mmm, request
    ):
        """Test that model-requiring factory methods return polars when requested."""
        mmm = request.getfixturevalue(fitted_mmm)
        factory = MMMSummaryFactory(mmm.data, model=mmm)
        method = getattr(factory, method_name)

        # Act
        df = method(output_format="polars")

        # Assert it returned a polars DataFrame
        assert df is not None, f"{method_name} returned None"
        assert isinstance(df, pl.DataFrame), (
            f"{method_name} should return polars DataFrame when requested"
        )


# =============================================================================
# HDI Computation Tests
# =============================================================================


class TestHDIComputation:
    """Test HDI bounds are computed correctly."""

    def test_hdi_bounds_contain_median(self, mock_mmm_idata_wrapper):
        """Test that HDI bounds contain the median value."""
        df = MMMSummaryFactory(mock_mmm_idata_wrapper).contributions(hdi_probs=[0.94])

        assert (df["abs_error_94_lower"] <= df["median"]).all()
        assert (df["median"] <= df["abs_error_94_upper"]).all()

    def test_wider_hdi_contains_narrower_hdi(self, mock_mmm_idata_wrapper):
        """Test that wider HDI intervals contain narrower intervals on average."""
        df = MMMSummaryFactory(mock_mmm_idata_wrapper).contributions(
            hdi_probs=[0.80, 0.94],
        )

        width_94 = df["abs_error_94_upper"] - df["abs_error_94_lower"]
        width_80 = df["abs_error_80_upper"] - df["abs_error_80_lower"]

        assert width_94.mean() >= width_80.mean()

    def test_multiple_hdi_probs_in_same_dataframe(self, mock_mmm_idata_wrapper):
        """Test that multiple HDI probabilities create all expected columns."""
        df = MMMSummaryFactory(mock_mmm_idata_wrapper).contributions(
            hdi_probs=[0.80, 0.90, 0.94],
        )

        expected_hdi_columns = {
            "abs_error_80_lower",
            "abs_error_80_upper",
            "abs_error_90_lower",
            "abs_error_90_upper",
            "abs_error_94_lower",
            "abs_error_94_upper",
        }
        assert expected_hdi_columns.issubset(set(df.columns))

    @pytest.mark.parametrize(
        "invalid_hdi_prob",
        [1.5, 0.0, -0.5, 100],
        ids=["greater_than_1", "zero", "negative", "percentage_format"],
    )
    def test_invalid_hdi_prob_raises(self, mock_mmm_idata_wrapper, invalid_hdi_prob):
        """Test that invalid HDI probabilities raise ValueError."""
        with pytest.raises(ValueError, match=r"[Hh][Dd][Ii]|probability"):
            MMMSummaryFactory(mock_mmm_idata_wrapper, hdi_probs=[invalid_hdi_prob])

    def test_empty_hdi_probs_produces_no_hdi_columns(self, mock_mmm_idata_wrapper):
        """Test that empty hdi_probs list produces no HDI columns."""
        df = MMMSummaryFactory(mock_mmm_idata_wrapper, hdi_probs=[]).contributions()

        assert "mean" in df.columns
        assert "median" in df.columns
        hdi_columns = [col for col in df.columns if "abs_error" in col]
        assert len(hdi_columns) == 0


# =============================================================================
# Data Transformation Tests
# =============================================================================


class TestDataTransformation:
    """Test xarray to DataFrame conversion preserves data correctly."""

    def test_mcmc_dimensions_collapsed(self, mock_mmm_idata_wrapper):
        """Test that chain and draw dimensions are collapsed in output."""
        contributions = mock_mmm_idata_wrapper.get_channel_contributions()
        assert "chain" in contributions.dims
        assert "draw" in contributions.dims

        df = MMMSummaryFactory(mock_mmm_idata_wrapper).contributions()

        assert "chain" not in df.columns
        assert "draw" not in df.columns

    def test_data_dimensions_preserved(self, mock_mmm_idata_wrapper):
        """Test that data dimensions (date, channel) are preserved as columns."""
        df = MMMSummaryFactory(mock_mmm_idata_wrapper).contributions()

        assert "date" in df.columns
        assert "channel" in df.columns
        expected_rows = df["date"].nunique() * df["channel"].nunique()
        assert len(df) == expected_rows

    def test_mean_and_median_computed_correctly(self, mock_mmm_idata_wrapper):
        """Test that mean and median are computed correctly from MCMC samples."""
        contributions = mock_mmm_idata_wrapper.get_channel_contributions()
        samples = contributions.isel(date=0, channel=0).values
        expected_mean = np.mean(samples)
        expected_median = np.median(samples)

        df = MMMSummaryFactory(mock_mmm_idata_wrapper).contributions()
        first_row = df.iloc[0]

        assert np.isclose(first_row["mean"], expected_mean, rtol=1e-6)
        assert np.isclose(first_row["median"], expected_median, rtol=1e-6)


# =============================================================================
# Component 1 Integration Tests
# =============================================================================


class TestComponent1Integration:
    """Test Component 2 correctly consumes Component 1's filtered/aggregated data."""

    def test_filtered_dates_produce_filtered_dataframe(self, mock_mmm_idata_wrapper):
        """Test that date filtering is preserved in output."""
        start_date = "2024-02-01"
        end_date = "2024-03-31"
        filtered_data = mock_mmm_idata_wrapper.filter_dates(start_date, end_date)

        df = MMMSummaryFactory(filtered_data).contributions()

        assert df["date"].min() >= pd.Timestamp(start_date)
        assert df["date"].max() <= pd.Timestamp(end_date)

    def test_filtered_channels_produce_filtered_dataframe(self, mock_mmm_idata_wrapper):
        """Test that channel filtering is preserved in output."""
        all_channels = mock_mmm_idata_wrapper.channels
        selected_channels = all_channels[:2]
        filtered_data = mock_mmm_idata_wrapper.filter_dims(channel=selected_channels)

        df = MMMSummaryFactory(filtered_data).contributions()

        assert len(df["channel"].unique()) == len(selected_channels)

    def test_frequency_parameter_aggregates_data(self, mock_mmm_idata_wrapper):
        """Test that frequency parameter triggers time aggregation."""
        df_monthly = MMMSummaryFactory(mock_mmm_idata_wrapper).contributions(
            frequency="monthly",
        )
        df_original = MMMSummaryFactory(mock_mmm_idata_wrapper).contributions(
            frequency="original",
        )

        assert df_monthly["date"].nunique() < df_original["date"].nunique()

    def test_change_over_time_requires_date_dimension(self, mock_mmm_idata_wrapper):
        """Test that change_over_time raises error when date dimension is missing."""
        aggregated_data = mock_mmm_idata_wrapper.aggregate_time("all_time")

        with pytest.raises(
            ValueError, match=r"change_over_time requires date dimension.*all_time"
        ):
            MMMSummaryFactory(aggregated_data).change_over_time()


# =============================================================================
# MMMSummaryFactory Tests
# =============================================================================


class TestMMMSummaryFactory:
    """Test MMMSummaryFactory class behavior."""

    def test_factory_class_initialization(self, mock_mmm_idata_wrapper):
        """Test that MMMSummaryFactory can be instantiated with defaults."""
        factory = MMMSummaryFactory(data=mock_mmm_idata_wrapper)

        assert factory.data is mock_mmm_idata_wrapper
        assert factory.hdi_probs == (0.94,)
        assert factory.output_format == "pandas"
        assert factory.model is None

    def test_factory_validates_data_at_init(self, mock_mmm_idata_wrapper):
        """Test that MMMSummaryFactory validates data structure at initialization."""
        mock_wrapper = Mock(spec=MMMIDataWrapper)
        mock_wrapper.schema = Mock()
        mock_wrapper.validate_or_raise = Mock()

        MMMSummaryFactory(data=mock_wrapper)

        mock_wrapper.validate_or_raise.assert_called_once()

    def test_factory_skips_validation_when_no_schema(self, mock_mmm_idata_wrapper):
        """Test that validation is skipped when schema is None."""
        mock_wrapper = Mock(spec=MMMIDataWrapper)
        mock_wrapper.schema = None
        mock_wrapper.validate_or_raise = Mock()

        MMMSummaryFactory(data=mock_wrapper)

        mock_wrapper.validate_or_raise.assert_not_called()

    def test_factory_raises_with_invalid_data_structure(self):
        """Test that MMMSummaryFactory raises early with invalid data structure."""
        invalid_idata = az.InferenceData(posterior=xr.Dataset())

        schema = MMMIdataSchema.from_model_config(
            custom_dims=(),
            has_controls=False,
            has_seasonality=False,
            time_varying=False,
        )

        wrapper = MMMIDataWrapper(invalid_idata, schema=schema, validate_on_init=False)

        with pytest.raises(ValueError, match="idata validation failed"):
            MMMSummaryFactory(data=wrapper)

    def test_factory_methods_use_stored_defaults(self, mock_mmm_idata_wrapper):
        """Test that factory methods use stored defaults."""
        factory = MMMSummaryFactory(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.80, 0.94],
            output_format="pandas",
        )

        df = factory.contributions()

        assert isinstance(df, pd.DataFrame)
        assert "abs_error_80_lower" in df.columns
        assert "abs_error_94_lower" in df.columns

    def test_factory_methods_allow_overrides(self, mock_mmm_idata_wrapper):
        """Test that factory methods allow overriding defaults per call."""
        factory = MMMSummaryFactory(
            data=mock_mmm_idata_wrapper,
            hdi_probs=[0.80],
        )

        df = factory.contributions(hdi_probs=[0.94])

        assert "abs_error_94_lower" in df.columns
        assert "abs_error_80_lower" not in df.columns


# =============================================================================
# Model-Requiring Methods Tests
# =============================================================================


class TestModelRequiringMethods:
    """Test methods that require a fitted model."""

    def test_factory_errors_when_model_required_but_missing(
        self, mock_mmm_idata_wrapper
    ):
        """Test that functions needing model raise helpful error when model is None."""
        factory = MMMSummaryFactory(mock_mmm_idata_wrapper)

        with pytest.raises(
            ValueError,
            match=r"saturation_curves requires model.*MMMSummaryFactory\(data, model=mmm\)",
        ):
            factory.saturation_curves()

        with pytest.raises(
            ValueError,
            match=r"adstock_curves requires model.*MMMSummaryFactory\(data, model=mmm\)",
        ):
            factory.adstock_curves()

        with pytest.raises(
            ValueError,
            match=r"roas with method='incremental' requires model",
        ):
            factory.roas(method="incremental")

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_factory_accepts_data_and_model(self, fitted_mmm, request):
        """Test that MMMSummaryFactory accepts both data and model."""
        mmm = request.getfixturevalue(fitted_mmm)
        data = mmm.data

        factory = MMMSummaryFactory(data, model=mmm)

        assert factory.data is data
        assert factory.model is mmm
        assert hasattr(factory.model, "saturation")
        assert hasattr(factory.model, "adstock")


# =============================================================================
# Curve Behavior Tests
# =============================================================================


class TestCurveBehavior:
    """Test saturation and adstock curve behavior."""

    @pytest.mark.parametrize("method_name", ["saturation_curves", "adstock_curves"])
    def test_curves_preserve_custom_dims(self, panel_fitted_mmm, method_name):
        """Test that curve functions preserve custom dimensions."""
        factory = MMMSummaryFactory(panel_fitted_mmm.data, model=panel_fitted_mmm)

        df = getattr(factory, method_name)()

        assert "country" in df.columns
        expected_countries = ["US", "UK"]
        assert sorted(df["country"].unique()) == sorted(expected_countries)

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_saturation_curves_are_increasing(self, fitted_mmm, request):
        """Test that saturation curves show increasing pattern."""
        mmm = request.getfixturevalue(fitted_mmm)
        factory = MMMSummaryFactory(mmm.data, model=mmm)

        df = factory.saturation_curves()
        if "channel" not in df.columns:
            df["channel"] = "channel"

        excluded_cols = ["x", "mean", "median"]
        dim_cols = [
            col
            for col in df.columns
            if col not in excluded_cols and not col.startswith("abs_error")
        ]
        if not dim_cols:
            dim_cols = ["channel"]

        for _dims, group_df in df.groupby(dim_cols):
            group_sorted = group_df.sort_values("x")
            assert group_sorted["mean"].iloc[-1] > group_sorted["mean"].iloc[0]

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_adstock_curves_are_decreasing(self, fitted_mmm, request):
        """Test that adstock curves show decreasing pattern over time."""
        mmm = request.getfixturevalue(fitted_mmm)
        factory = MMMSummaryFactory(mmm.data, model=mmm)

        df = factory.adstock_curves()
        if "channel" not in df.columns:
            df["channel"] = "channel"

        excluded_cols = ["time since exposure", "mean", "median"]
        dim_cols = [
            col
            for col in df.columns
            if col not in excluded_cols and not col.startswith("abs_error")
        ]
        if not dim_cols:
            dim_cols = ["channel"]

        for _dims, group_df in df.groupby(dim_cols):
            group_sorted = group_df.sort_values("time since exposure")
            assert group_sorted["mean"].iloc[0] > group_sorted["mean"].iloc[-1]
            assert group_sorted["mean"].iloc[0] == group_sorted["mean"].max()

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    @pytest.mark.parametrize("num_points", [25, 50, 100])
    def test_saturation_curves_custom_num_points(self, fitted_mmm, num_points, request):
        """Test saturation curves with custom num_points parameter."""
        mmm = request.getfixturevalue(fitted_mmm)
        factory = MMMSummaryFactory(mmm.data, model=mmm)
        df = factory.saturation_curves(num_points=num_points, hdi_probs=[0.80])

        if "channel" in df.columns:
            n_channels = df["channel"].nunique()
        else:
            n_channels = 1

        # Count custom dimensions
        excluded_cols = [
            "x",
            "channel",
            "mean",
            "median",
            "abs_error_80_lower",
            "abs_error_80_upper",
        ]
        custom_dims = [col for col in df.columns if col not in excluded_cols]
        n_custom = 1
        for dim in custom_dims:
            n_custom *= df[dim].nunique()

        expected_rows = num_points * n_channels * n_custom
        assert len(df) == expected_rows


# =============================================================================
# Non-Channel Components Tests
# =============================================================================


class TestNonChannelComponents:
    """Test contribution summary for non-channel components."""

    def test_contribution_summary_control_component(
        self, mock_mmm_idata_wrapper_with_controls
    ):
        """Test contribution summary for control component."""
        df = MMMSummaryFactory(mock_mmm_idata_wrapper_with_controls).contributions(
            component="control",
            hdi_probs=[0.94],
        )

        assert isinstance(df, pd.DataFrame)
        assert "control" in df.columns
        assert len(df["control"].unique()) == 2  # price and promo

    def test_contribution_summary_missing_component_raises(
        self, mock_mmm_idata_wrapper
    ):
        """Test that requesting missing component raises ValueError."""
        with pytest.raises(ValueError, match=r"No controls contributions found"):
            MMMSummaryFactory(mock_mmm_idata_wrapper).contributions(component="control")


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_date_range_raises_or_returns_empty(self, mock_mmm_idata_wrapper):
        """Test behavior when filtered data has no dates."""
        filtered_data = mock_mmm_idata_wrapper.filter_dates("2030-01-01", "2030-01-02")

        try:
            df = MMMSummaryFactory(filtered_data).contributions()
            assert len(df) == 0
        except ValueError as e:
            assert "empty" in str(e).lower() or "no data" in str(e).lower()

    def test_division_by_zero_in_roas_handled(
        self, mock_mmm_idata_wrapper_with_zero_spend
    ):
        """Test that ROAS computation handles zero spend without errors."""
        df = MMMSummaryFactory(mock_mmm_idata_wrapper_with_zero_spend).roas(
            method="elementwise"
        )

        zero_spend_rows = df[df["channel"] == "zero_spend_channel"]
        assert (
            zero_spend_rows["mean"].isna().all()
            or (zero_spend_rows["mean"] == 0).all()
            or np.isinf(zero_spend_rows["mean"]).all()
        )


# =============================================================================
# ROAS Method Tests
# =============================================================================


class TestROASMethods:
    """Test ROAS method parameter (incremental vs elementwise)."""

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_roas_incremental_method_with_model(self, fitted_mmm, request):
        """Test that method='incremental' works when model is provided."""
        mmm = request.getfixturevalue(fitted_mmm)
        df = mmm.summary.roas(
            method="incremental", frequency="original", num_samples=10
        )

        required_columns = {"date", "channel", "mean", "median"}
        assert required_columns.issubset(set(df.columns))
        assert len(df) > 0

    def test_roas_elementwise_method_without_model(self, mock_mmm_idata_wrapper):
        """Test that method='elementwise' works with data-only factory."""
        df = MMMSummaryFactory(mock_mmm_idata_wrapper).roas(method="elementwise")

        required_columns = {"date", "channel", "mean", "median"}
        assert required_columns.issubset(set(df.columns))
        assert len(df) > 0

    def test_roas_elementwise_with_date_filter(self, mock_mmm_idata_wrapper):
        """Test that start_date/end_date filter elementwise ROAS results."""
        factory = MMMSummaryFactory(mock_mmm_idata_wrapper)

        df_full = factory.roas(method="elementwise")

        dates = mock_mmm_idata_wrapper.dates
        mid = dates[len(dates) // 2]

        df_filtered = factory.roas(
            method="elementwise",
            start_date=str(dates[0].date()),
            end_date=str(mid.date()),
        )

        assert len(df_filtered) > 0
        assert len(df_filtered) < len(df_full)
        assert df_filtered["date"].max() <= mid

    def test_roas_elementwise_with_start_date_only(self, mock_mmm_idata_wrapper):
        """Test that only start_date filters elementwise ROAS results."""
        factory = MMMSummaryFactory(mock_mmm_idata_wrapper)

        dates = mock_mmm_idata_wrapper.dates
        mid = dates[len(dates) // 2]

        df_filtered = factory.roas(method="elementwise", start_date=str(mid.date()))

        assert len(df_filtered) > 0
        assert df_filtered["date"].min() >= mid

    def test_roas_elementwise_with_end_date_only(self, mock_mmm_idata_wrapper):
        """Test that only end_date filters elementwise ROAS results."""
        factory = MMMSummaryFactory(mock_mmm_idata_wrapper)

        dates = mock_mmm_idata_wrapper.dates
        mid = dates[len(dates) // 2]

        df_filtered = factory.roas(method="elementwise", end_date=str(mid.date()))

        assert len(df_filtered) > 0
        assert df_filtered["date"].max() <= mid

    def test_roas_invalid_method_raises(self, mock_mmm_idata_wrapper):
        """Test that invalid method value raises ValueError."""
        factory = MMMSummaryFactory(mock_mmm_idata_wrapper)

        with pytest.raises(
            ValueError,
            match=r"method must be 'incremental' or 'elementwise'",
        ):
            factory.roas(method="invalid")

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_roas_incremental_and_elementwise_both_produce_valid_output(
        self, fitted_mmm, request
    ):
        """Integration test: both methods produce valid ROAS DataFrames."""
        mmm = request.getfixturevalue(fitted_mmm)

        df_incremental = mmm.summary.roas(
            method="incremental", frequency="monthly", num_samples=10
        )
        df_elementwise = mmm.summary.roas(method="elementwise", frequency="monthly")

        for df in (df_incremental, df_elementwise):
            assert "mean" in df.columns
            assert "median" in df.columns
            assert "channel" in df.columns
            assert len(df) > 0
            # ROAS values should be non-negative or NaN (for zero spend)
            valid_means = df["mean"].dropna()
            assert (valid_means >= 0).all() or len(valid_means) == 0

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_roas_start_end_date_filters_output(self, fitted_mmm, request):
        """Test that start_date and end_date restrict the ROAS evaluation window."""
        mmm = request.getfixturevalue(fitted_mmm)

        dates = mmm.data.dates
        mid = dates[len(dates) // 2]
        start = dates[0]

        df_full = mmm.summary.roas(
            method="incremental", frequency="original", num_samples=10
        )
        df_partial = mmm.summary.roas(
            method="incremental",
            frequency="original",
            start_date=str(start.date()),
            end_date=str(mid.date()),
            num_samples=10,
        )

        assert len(df_partial) > 0
        assert len(df_partial) < len(df_full)
        assert df_partial["date"].max() <= mid

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_roas_start_end_date_with_frequency(self, fitted_mmm, request):
        """Test start/end_date with a non-original frequency."""
        mmm = request.getfixturevalue(fitted_mmm)

        df = mmm.summary.roas(
            method="incremental",
            frequency="all_time",
            start_date=str(mmm.data.dates[0].date()),
            end_date=str(mmm.data.dates[-1].date()),
            num_samples=10,
        )

        assert len(df) > 0
        assert "channel" in df.columns


# =============================================================================
# MMM.summary Property Tests
# =============================================================================


class TestMMMSummaryProperty:
    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_summary_property_returns_fresh_factory(self, fitted_mmm, request):
        """Test that summary property returns fresh factory on each access."""
        mmm = request.getfixturevalue(fitted_mmm)

        summary1 = mmm.summary
        summary2 = mmm.summary

        assert summary1 is not summary2
        assert summary1.model is mmm
        assert summary2.model is mmm

    def test_summary_validates_idata_exists(self):
        """Test that summary property validates idata exists."""
        mmm = MMM(
            adstock=GeometricAdstock(l_max=10),
            saturation=LogisticSaturation(),
            date_column="date",
            target_column="target",
            channel_columns=["TV", "Radio"],
        )

        with pytest.raises(ValueError, match=r"idata does not exist"):
            _ = mmm.summary

    @pytest.mark.parametrize("fitted_mmm", ["simple_fitted_mmm", "panel_fitted_mmm"])
    def test_summary_property_usage_example(self, fitted_mmm, request):
        """Test realistic usage of summary property."""
        mmm = request.getfixturevalue(fitted_mmm)

        contributions_df = mmm.summary.contributions()
        assert len(contributions_df) > 0
        assert "channel" in contributions_df.columns

        roas_df = mmm.summary.roas(method="incremental", num_samples=10)
        assert len(roas_df) > 0

        saturation_df = mmm.summary.saturation_curves()
        assert len(saturation_df) > 0
        assert "x" in saturation_df.columns
