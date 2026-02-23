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
"""Tests for cost_per_unit feature"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer
from pymc_marketing.mmm.multidimensional import (
    MMM,
    MultiDimensionalBudgetOptimizerWrapper,
)

SEED = sum(map(ord, "cost_per_unit_tests"))
rng = np.random.default_rng(seed=SEED)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def dates():
    return pd.date_range("2024-01-01", periods=4, freq="W-MON")


@pytest.fixture
def channels():
    return ["TV", "Radio"]


@pytest.fixture
def countries():
    return ["US", "UK", "DE"]


@pytest.fixture
def simple_idata(dates, channels):
    """InferenceData without custom dims."""
    n_dates = len(dates)
    n_channels = len(channels)
    return az.InferenceData(
        constant_data=xr.Dataset(
            {
                "channel_data": xr.DataArray(
                    rng.uniform(100, 1000, size=(n_dates, n_channels)),
                    dims=("date", "channel"),
                    coords={"date": dates, "channel": channels},
                ),
                "target_data": xr.DataArray(
                    rng.uniform(500, 2000, size=(n_dates,)),
                    dims=("date",),
                    coords={"date": dates},
                ),
                "channel_scale": xr.DataArray(
                    [500.0, 300.0],
                    dims=("channel",),
                    coords={"channel": channels},
                ),
                "target_scale": xr.DataArray(1000.0),
            }
        ),
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(0.5, 0.1, size=(2, 50, n_dates, n_channels)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={
                        "chain": [0, 1],
                        "draw": np.arange(50),
                        "date": dates,
                        "channel": channels,
                    },
                ),
            }
        ),
    )


@pytest.fixture
def multidim_idata(dates, channels, countries):
    """InferenceData with custom dims (country)."""
    n_dates = len(dates)
    n_channels = len(channels)
    n_countries = len(countries)
    return az.InferenceData(
        constant_data=xr.Dataset(
            {
                "channel_data": xr.DataArray(
                    rng.uniform(100, 1000, size=(n_dates, n_countries, n_channels)),
                    dims=("date", "country", "channel"),
                    coords={
                        "date": dates,
                        "country": countries,
                        "channel": channels,
                    },
                ),
                "target_data": xr.DataArray(
                    rng.uniform(500, 2000, size=(n_dates, n_countries)),
                    dims=("date", "country"),
                    coords={"date": dates, "country": countries},
                ),
                "channel_scale": xr.DataArray(
                    rng.uniform(200, 600, size=(n_countries, n_channels)),
                    dims=("country", "channel"),
                    coords={"country": countries, "channel": channels},
                ),
                "target_scale": xr.DataArray(
                    [1000.0, 900.0, 800.0],
                    dims=("country",),
                    coords={"country": countries},
                ),
            }
        ),
        posterior=xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(
                        0.5,
                        0.1,
                        size=(2, 50, n_dates, n_countries, n_channels),
                    ),
                    dims=("chain", "draw", "date", "country", "channel"),
                    coords={
                        "chain": [0, 1],
                        "draw": np.arange(50),
                        "date": dates,
                        "country": countries,
                        "channel": channels,
                    },
                ),
            }
        ),
    )


# ============================================================================
# _parse_cost_per_unit_df() Unit Tests
# ============================================================================


class TestParseCostPerUnitDf:
    """Tests for the static _parse_cost_per_unit_df method."""

    def test_parse_single_channel_no_custom_dims(self, dates, channels):
        df = pd.DataFrame({"date": dates, "TV": [0.01, 0.02, 0.015, 0.012]})
        result = MMM._parse_cost_per_unit_df(df, channels=channels, dates=dates)

        assert result.dims == ("date", "channel")
        assert list(result.coords["channel"].values) == channels
        np.testing.assert_array_equal(
            result.sel(channel="TV").values, [0.01, 0.02, 0.015, 0.012]
        )
        np.testing.assert_array_equal(
            result.sel(channel="Radio").values, [1.0, 1.0, 1.0, 1.0]
        )

    def test_parse_all_channels_no_custom_dims(self, dates, channels):
        df = pd.DataFrame(
            {
                "date": dates,
                "TV": [0.01, 0.02, 0.015, 0.012],
                "Radio": [0.05, 0.06, 0.055, 0.052],
            }
        )
        result = MMM._parse_cost_per_unit_df(df, channels=channels, dates=dates)

        assert result.dims == ("date", "channel")
        np.testing.assert_array_equal(
            result.sel(channel="TV").values, [0.01, 0.02, 0.015, 0.012]
        )
        np.testing.assert_array_equal(
            result.sel(channel="Radio").values, [0.05, 0.06, 0.055, 0.052]
        )

    def test_parse_single_channel_with_custom_dims(self, dates, channels, countries):
        n_dates = len(dates)
        n_countries = len(countries)
        df = pd.DataFrame(
            {
                "date": np.repeat(dates, n_countries),
                "country": countries * n_dates,
                "TV": [0.01, 0.02, 0.015] * n_dates,
            }
        )
        result = MMM._parse_cost_per_unit_df(
            df,
            channels=channels,
            dates=dates,
            custom_dims=("country",),
            custom_dim_coords={"country": np.array(countries)},
        )

        assert result.dims == ("date", "country", "channel")
        np.testing.assert_array_equal(
            result.sel(channel="Radio").values,
            np.ones((n_dates, n_countries)),
        )
        assert result.sel(date=dates[0], country="US", channel="TV").item() == 0.01
        assert result.sel(date=dates[0], country="UK", channel="TV").item() == 0.02

    def test_parse_all_channels_with_custom_dims(self, dates, channels, countries):
        n_dates = len(dates)
        n_countries = len(countries)
        df = pd.DataFrame(
            {
                "date": np.repeat(dates, n_countries),
                "country": countries * n_dates,
                "TV": [0.01, 0.02, 0.015] * n_dates,
                "Radio": [0.05, 0.06, 0.055] * n_dates,
            }
        )
        result = MMM._parse_cost_per_unit_df(
            df,
            channels=channels,
            dates=dates,
            custom_dims=("country",),
            custom_dim_coords={"country": np.array(countries)},
        )

        assert result.dims == ("date", "country", "channel")
        assert result.sel(date=dates[0], country="US", channel="TV").item() == 0.01
        assert result.sel(date=dates[0], country="US", channel="Radio").item() == 0.05

    def test_parse_missing_date_column_raises(self, dates, channels):
        df = pd.DataFrame({"TV": [0.01, 0.02, 0.015, 0.012]})
        with pytest.raises(ValueError, match="must contain a 'date' column"):
            MMM._parse_cost_per_unit_df(df, channels=channels, dates=dates)

    def test_parse_missing_custom_dim_column_raises(self, dates, channels, countries):
        df = pd.DataFrame({"date": dates, "TV": [0.01, 0.02, 0.015, 0.012]})
        with pytest.raises(ValueError, match="missing dim columns"):
            MMM._parse_cost_per_unit_df(
                df,
                channels=channels,
                dates=dates,
                custom_dims=("country",),
                custom_dim_coords={"country": np.array(countries)},
            )

    def test_parse_unknown_channel_raises(self, dates, channels):
        df = pd.DataFrame(
            {
                "date": dates,
                "TV": [0.01, 0.02, 0.015, 0.012],
                "Unknown": [0.1, 0.2, 0.3, 0.4],
            }
        )
        with pytest.raises(ValueError, match="unknown channels"):
            MMM._parse_cost_per_unit_df(df, channels=channels, dates=dates)

    def test_parse_no_channel_columns_raises(self, dates, channels):
        df = pd.DataFrame({"date": dates})
        with pytest.raises(ValueError, match="no channel columns"):
            MMM._parse_cost_per_unit_df(df, channels=channels, dates=dates)

    def test_parse_negative_values_raises(self, dates, channels):
        df = pd.DataFrame({"date": dates, "TV": [0.01, -0.02, 0.015, 0.012]})
        with pytest.raises(ValueError, match="must be positive"):
            MMM._parse_cost_per_unit_df(df, channels=channels, dates=dates)

    def test_parse_zero_values_raises(self, dates, channels):
        df = pd.DataFrame({"date": dates, "TV": [0.01, 0.0, 0.015, 0.012]})
        with pytest.raises(ValueError, match="must be positive"):
            MMM._parse_cost_per_unit_df(df, channels=channels, dates=dates)

    def test_parse_reindex_nan_raises(self, channels):
        model_dates = pd.date_range("2024-01-01", periods=4, freq="W-MON")
        df_dates = pd.date_range("2024-02-01", periods=4, freq="W-MON")
        df = pd.DataFrame({"date": df_dates, "TV": [0.01, 0.02, 0.015, 0.012]})
        with pytest.raises(ValueError, match="reindex produced NaN"):
            MMM._parse_cost_per_unit_df(df, channels=channels, dates=model_dates)


# ============================================================================
# Wrapper Tests: get_channel_spend() and cost_per_unit property
# ============================================================================


class TestWrapperCostPerUnit:
    """Tests for the MMMIDataWrapper cost_per_unit support."""

    def test_get_channel_spend_without_cost_per_unit(self, simple_idata):
        wrapper = MMMIDataWrapper(simple_idata)
        result = wrapper.get_channel_spend()
        xr.testing.assert_equal(result, simple_idata.constant_data.channel_data)

    def test_cost_per_unit_property_none_when_absent(self, simple_idata):
        wrapper = MMMIDataWrapper(simple_idata)
        assert wrapper.cost_per_unit is None

    def test_cost_per_unit_property_returns_data(self, simple_idata, dates, channels):
        cpu = xr.DataArray(
            rng.uniform(0.01, 0.1, size=(len(dates), len(channels))),
            dims=("date", "channel"),
            coords={"date": dates, "channel": channels},
        )
        channel_data = simple_idata.constant_data.channel_data
        simple_idata.constant_data["channel_spend"] = channel_data * cpu
        wrapper = MMMIDataWrapper(simple_idata)

        computed_cpu = wrapper.cost_per_unit
        xr.testing.assert_allclose(computed_cpu, cpu)

    def test_get_channel_spend_with_cost_per_unit(self, simple_idata, dates, channels):
        cpu = xr.DataArray(
            [[0.01, 0.05]] * len(dates),
            dims=("date", "channel"),
            coords={"date": dates, "channel": channels},
        )
        channel_data = simple_idata.constant_data.channel_data
        simple_idata.constant_data["channel_spend"] = channel_data * cpu
        wrapper = MMMIDataWrapper(simple_idata)

        spend = wrapper.get_channel_spend()
        raw = wrapper.get_channel_data()

        expected = channel_data * cpu
        xr.testing.assert_equal(spend, expected)
        xr.testing.assert_equal(raw, channel_data)

    def test_get_channel_spend_default_applies_conversion(
        self, simple_idata, dates, channels
    ):
        cpu = xr.DataArray(
            [[2.0, 3.0]] * len(dates),
            dims=("date", "channel"),
            coords={"date": dates, "channel": channels},
        )
        channel_data = simple_idata.constant_data.channel_data
        simple_idata.constant_data["channel_spend"] = channel_data * cpu
        wrapper = MMMIDataWrapper(simple_idata)

        spend = wrapper.get_channel_spend()
        expected = channel_data * cpu
        xr.testing.assert_equal(spend, expected)

    def test_get_channel_spend_with_cost_per_unit_multidim(
        self, multidim_idata, dates, channels, countries
    ):
        cpu = xr.DataArray(
            np.full((len(dates), len(countries), len(channels)), 0.02),
            dims=("date", "country", "channel"),
            coords={
                "date": dates,
                "country": countries,
                "channel": channels,
            },
        )
        channel_data = multidim_idata.constant_data.channel_data
        multidim_idata.constant_data["channel_spend"] = channel_data * cpu
        wrapper = MMMIDataWrapper(multidim_idata)

        spend = wrapper.get_channel_spend()
        expected = channel_data * cpu
        xr.testing.assert_equal(spend, expected)

    def test_elementwise_roas_uses_converted_spend(self, simple_idata, dates, channels):
        cpu = xr.DataArray(
            [[2.0, 3.0]] * len(dates),
            dims=("date", "channel"),
            coords={"date": dates, "channel": channels},
        )
        channel_data = simple_idata.constant_data.channel_data
        simple_idata.constant_data["channel_spend"] = channel_data * cpu
        wrapper = MMMIDataWrapper(simple_idata)

        roas = wrapper.get_elementwise_roas()
        contributions = wrapper.get_channel_contributions(original_scale=True)
        spend = wrapper.get_channel_spend()
        spend_safe = xr.where(spend == 0, np.nan, spend)
        expected_roas = contributions / spend_safe
        xr.testing.assert_equal(roas, expected_roas)

    def test_get_channel_data_returns_raw(self, simple_idata, dates, channels):
        """get_channel_data() always returns raw data, even when channel_spend exists."""
        channel_data = simple_idata.constant_data.channel_data.copy(deep=True)
        simple_idata.constant_data["channel_spend"] = channel_data * 0.01
        wrapper = MMMIDataWrapper(simple_idata)

        raw = wrapper.get_channel_data()
        xr.testing.assert_equal(raw, channel_data)

    def test_get_channel_data_without_channel_spend(self, simple_idata):
        """get_channel_data() works when no channel_spend exists."""
        wrapper = MMMIDataWrapper(simple_idata)
        raw = wrapper.get_channel_data()
        xr.testing.assert_equal(raw, simple_idata.constant_data.channel_data)

    def test_get_channel_data_missing_raises(self):
        """get_channel_data() raises ValueError when channel_data absent."""
        empty_idata = az.InferenceData(constant_data=xr.Dataset())
        wrapper = MMMIDataWrapper(empty_idata)
        with pytest.raises(ValueError, match="Channel data not found"):
            wrapper.get_channel_data()


class TestBudgetOptimizerCostPerUnitValidation:
    """Tests for _validate_and_process_cost_per_unit on BudgetOptimizer."""

    @pytest.fixture
    def optimizer_instance(self):
        """Bare BudgetOptimizer instance just for calling the validation method."""

        class Stub:
            pass

        stub = Stub()
        stub._validate_and_process_cost_per_unit = (
            BudgetOptimizer._validate_and_process_cost_per_unit.__get__(stub)
        )
        return stub

    def test_none_returns_none(self, optimizer_instance):
        result = optimizer_instance._validate_and_process_cost_per_unit(
            cost_per_unit=None, num_periods=4, budget_dims=["channel"]
        )
        assert result is None

    def test_wrong_dims_raises(self, optimizer_instance):
        cpu = xr.DataArray(
            np.ones((4, 2)),
            dims=("date", "channel"),
            coords={"date": range(4), "channel": ["TV", "Radio"]},
        )
        with pytest.raises(ValueError, match="must have dims"):
            optimizer_instance._validate_and_process_cost_per_unit(
                cost_per_unit=cpu,
                num_periods=4,
                budget_dims=["channel", "geo"],
            )

    def test_wrong_date_length_raises(self, optimizer_instance):
        cpu = xr.DataArray(
            np.ones((3, 2)),
            dims=("date", "channel"),
            coords={"date": range(3), "channel": ["TV", "Radio"]},
        )
        with pytest.raises(ValueError, match="date dimension must have length"):
            optimizer_instance._validate_and_process_cost_per_unit(
                cost_per_unit=cpu,
                num_periods=5,
                budget_dims=["channel"],
            )

    def test_non_positive_values_raises(self, optimizer_instance):
        cpu = xr.DataArray(
            np.array([[0.01, -0.05], [0.01, 0.05]]),
            dims=("date", "channel"),
            coords={"date": range(2), "channel": ["TV", "Radio"]},
        )
        with pytest.raises(ValueError, match="must be positive"):
            optimizer_instance._validate_and_process_cost_per_unit(
                cost_per_unit=cpu,
                num_periods=2,
                budget_dims=["channel"],
            )

    def test_zero_values_raises(self, optimizer_instance):
        cpu = xr.DataArray(
            np.array([[0.0, 0.05], [0.01, 0.05]]),
            dims=("date", "channel"),
            coords={"date": range(2), "channel": ["TV", "Radio"]},
        )
        with pytest.raises(ValueError, match="must be positive"):
            optimizer_instance._validate_and_process_cost_per_unit(
                cost_per_unit=cpu,
                num_periods=2,
                budget_dims=["channel"],
            )

    def test_valid_returns_tensor(self, optimizer_instance):
        import pytensor.tensor as pt

        cpu = xr.DataArray(
            np.array([[0.01, 0.05], [0.012, 0.055]]),
            dims=("date", "channel"),
            coords={"date": range(2), "channel": ["TV", "Radio"]},
        )
        result = optimizer_instance._validate_and_process_cost_per_unit(
            cost_per_unit=cpu,
            num_periods=2,
            budget_dims=["channel"],
        )
        assert isinstance(result, pt.TensorConstant)
        np.testing.assert_array_almost_equal(
            result.data, np.array([[0.01, 0.05], [0.012, 0.055]])
        )

    def test_valid_with_multiple_budget_dims(self, optimizer_instance):
        import pytensor.tensor as pt

        cpu = xr.DataArray(
            np.full((3, 2, 2), 0.01),
            dims=("date", "channel", "geo"),
            coords={
                "date": range(3),
                "channel": ["TV", "Radio"],
                "geo": ["US", "UK"],
            },
        )
        result = optimizer_instance._validate_and_process_cost_per_unit(
            cost_per_unit=cpu,
            num_periods=3,
            budget_dims=["channel", "geo"],
        )
        assert isinstance(result, pt.TensorConstant)
        assert result.data.shape == (3, 2, 2)


# ============================================================================
# Integration Tests: set_cost_per_unit on fitted models (Tests 12-13)
# ============================================================================


class TestSetCostPerUnit:
    """Tests for set_cost_per_unit on fitted MMM models."""

    def test_set_cost_per_unit_post_hoc(self, simple_fitted_mmm):
        """Test 12: Set cost_per_unit after fitting and verify conversion."""
        mmm = simple_fitted_mmm
        dates = mmm.data.dates

        assert mmm.data.cost_per_unit is None
        raw_data = mmm.data.get_channel_data()

        cpu_df = pd.DataFrame(
            {
                "date": dates,
                "channel_1": [0.01] * len(dates),
                "channel_2": [0.02] * len(dates),
            }
        )
        mmm.set_cost_per_unit(cpu_df)

        assert mmm.data.cost_per_unit is not None
        assert "channel_spend" in mmm.idata.constant_data

        spend = mmm.data.get_channel_spend()
        expected_spend = raw_data * mmm.data.cost_per_unit
        xr.testing.assert_allclose(spend, expected_spend)

        # channel_3 (missing from cpu_df) defaults to 1.0
        np.testing.assert_allclose(
            mmm.data.cost_per_unit.sel(channel="channel_3").values,
            np.ones(len(dates)),
        )

    def test_set_cost_per_unit_overwrite_protection(self, simple_fitted_mmm):
        """Test 13: Overwrite protection and override."""
        mmm = simple_fitted_mmm
        dates = mmm.data.dates

        cpu_df = pd.DataFrame(
            {
                "date": dates,
                "channel_1": [0.01] * len(dates),
            }
        )
        mmm.set_cost_per_unit(cpu_df)

        with pytest.raises(ValueError, match="channel_spend exists"):
            mmm.set_cost_per_unit(cpu_df)

        new_cpu_df = pd.DataFrame(
            {
                "date": dates,
                "channel_1": [0.05] * len(dates),
            }
        )
        mmm.set_cost_per_unit(new_cpu_df, overwrite=True)

        np.testing.assert_array_almost_equal(
            mmm.data.cost_per_unit.sel(channel="channel_1").values,
            [0.05] * len(dates),
        )

    def test_set_cost_per_unit_before_fit_raises(self):
        """Setting cost_per_unit on unfitted model raises RuntimeError."""
        mmm = MMM(
            channel_columns=["channel_1", "channel_2"],
            date_column="date",
            target_column="target",
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        cpu_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4, freq="W-MON"),
                "channel_1": [0.01] * 4,
            }
        )
        with pytest.raises(RuntimeError, match="must be fitted"):
            mmm.set_cost_per_unit(cpu_df)


# ============================================================================
# Integration Tests: BudgetOptimizer cost_per_unit (Tests 15-16, 18-19)
# ============================================================================


class TestBudgetOptimizerCostPerUnitIntegration:
    """Tests 15-16, 18-19: Budget optimizer with cost_per_unit."""

    RESPONSE_VAR = "total_media_contribution_original_scale"

    @pytest.fixture
    def budget_mmm_setup(self, simple_fitted_mmm):
        """Build a multidimensional MMM wrapper for budget optimizer tests."""
        mmm = simple_fitted_mmm
        wrapper = MultiDimensionalBudgetOptimizerWrapper(
            model=mmm, start_date="2025-01-06", end_date="2025-02-03"
        )
        return wrapper, mmm.channel_columns

    def test_budget_optimizer_with_constant_cost_per_unit(self, budget_mmm_setup):
        """Test 15: Constant cost_per_unit (same rate every period)."""
        wrapper, channel_columns = budget_mmm_setup
        num_periods = wrapper.num_periods

        cpu = xr.DataArray(
            np.full((num_periods, len(channel_columns)), 0.01),
            dims=("date", "channel"),
            coords={"date": range(num_periods), "channel": channel_columns},
        )

        with pytest.warns(UserWarning, match="Using default equality constraint"):
            optimizer = BudgetOptimizer(
                model=wrapper,
                num_periods=num_periods,
                cost_per_unit=cpu,
                response_variable=self.RESPONSE_VAR,
            )

        budget_bounds = {ch: (0.0, 500.0) for ch in channel_columns}
        optimal_budgets, opt_result = optimizer.allocate_budget(
            total_budget=1000.0,
            budget_bounds=budget_bounds,
        )

        assert optimal_budgets is not None
        assert opt_result.success or opt_result.fun < 0

    def test_budget_optimizer_with_date_varying_cost_per_unit(self, budget_mmm_setup):
        """Test 16: Date-varying cost_per_unit (different rate per period)."""
        wrapper, channel_columns = budget_mmm_setup
        num_periods = wrapper.num_periods

        base_rates = np.linspace(0.01, 0.06, len(channel_columns))
        varying_values = np.array(
            [base_rates * (1 + 0.1 * t) for t in range(num_periods)]
        )
        cpu = xr.DataArray(
            varying_values,
            dims=("date", "channel"),
            coords={"date": range(num_periods), "channel": channel_columns},
        )

        with pytest.warns(UserWarning, match="Using default equality constraint"):
            optimizer = BudgetOptimizer(
                model=wrapper,
                num_periods=num_periods,
                cost_per_unit=cpu,
                response_variable=self.RESPONSE_VAR,
            )

        budget_bounds = {ch: (0.0, 500.0) for ch in channel_columns}
        optimal_budgets, opt_result = optimizer.allocate_budget(
            total_budget=1000.0,
            budget_bounds=budget_bounds,
        )

        assert optimal_budgets is not None
        assert opt_result.success or opt_result.fun < 0

    def test_optimize_budget_cost_per_unit_dataframe_input(self, panel_fitted_mmm):
        """Test 18: Pass pd.DataFrame to optimize_budget(), verify parsing."""
        mmm = panel_fitted_mmm
        start_date = "2025-01-06"
        end_date = "2025-02-03"

        wrapper = MultiDimensionalBudgetOptimizerWrapper(
            model=mmm, start_date=start_date, end_date=end_date
        )

        opt_dates = pd.date_range(start_date, periods=wrapper.num_periods, freq="W-MON")
        channels = mmm.channel_columns
        countries = mmm.idata.constant_data.coords["country"].values

        cpu_df = pd.DataFrame(
            {
                "date": np.repeat(opt_dates, len(countries)),
                "country": list(countries) * wrapper.num_periods,
                "channel_1": [0.01] * (wrapper.num_periods * len(countries)),
                "channel_2": [0.02] * (wrapper.num_periods * len(countries)),
            }
        )

        budget_bounds = xr.DataArray(
            np.array([[[0, 1000]] * len(channels)] * len(countries)),
            coords=[countries, channels, ["low", "high"]],
            dims=["country", "channel", "bound"],
        )

        allocation, _opt_result = wrapper.optimize_budget(
            budget=1000,
            budget_bounds=budget_bounds,
            cost_per_unit=cpu_df,
        )

        assert allocation is not None
        assert set(allocation.dims) == set((*mmm.dims, "channel"))

    def test_budget_optimizer_cost_per_unit_with_distribution(self, budget_mmm_setup):
        """Test 19: Combined budget_distribution_over_period + cost_per_unit."""
        wrapper, channel_columns = budget_mmm_setup
        num_periods = wrapper.num_periods

        cpu = xr.DataArray(
            np.full((num_periods, len(channel_columns)), 0.01),
            dims=("date", "channel"),
            coords={"date": range(num_periods), "channel": channel_columns},
        )

        weights = np.linspace(num_periods, 1, num_periods)
        weights /= weights.sum()
        dist_values = np.column_stack([weights] * len(channel_columns))
        distribution = xr.DataArray(
            dist_values,
            dims=("date", "channel"),
            coords={"date": range(num_periods), "channel": channel_columns},
        )

        optimizer = BudgetOptimizer(
            model=wrapper,
            num_periods=num_periods,
            cost_per_unit=cpu,
            budget_distribution_over_period=distribution,
            response_variable=self.RESPONSE_VAR,
        )

        budget_bounds = {ch: (0.0, 500.0) for ch in channel_columns}
        optimal_budgets, opt_result = optimizer.allocate_budget(
            total_budget=1000.0,
            budget_bounds=budget_bounds,
        )

        assert optimal_budgets is not None
        assert opt_result.success or opt_result.fun < 0


# ============================================================================
# Serialization Roundtrip Tests (Tests 20-22)
# ============================================================================


class TestSerializationRoundtrip:
    """Tests 20-22: Save/load with cost_per_unit."""

    def test_save_load_with_cost_per_unit(self, simple_fitted_mmm, tmp_path):
        """Test 20: Model with cost_per_unit saves and loads correctly."""
        mmm = simple_fitted_mmm
        dates = mmm.data.dates

        cpu_df = pd.DataFrame(
            {
                "date": dates,
                "channel_1": [0.01] * len(dates),
                "channel_2": [0.02] * len(dates),
            }
        )
        mmm.set_cost_per_unit(cpu_df)

        original_cpu = mmm.data.cost_per_unit.copy(deep=True)
        original_spend = mmm.data.get_channel_spend()

        fname = str(tmp_path / "model_with_cpu.pm")
        mmm.save(fname)
        loaded = MMM.load(fname, check=False)

        assert loaded.data.cost_per_unit is not None
        xr.testing.assert_equal(loaded.data.cost_per_unit, original_cpu)

        loaded_spend = loaded.data.get_channel_spend()
        xr.testing.assert_allclose(loaded_spend, original_spend)

    def test_load_old_model_without_cost_per_unit(self, simple_fitted_mmm, tmp_path):
        """Test 21: Model without cost_per_unit loads correctly, supports post-hoc."""
        mmm = simple_fitted_mmm

        fname = str(tmp_path / "model_no_cpu.pm")
        mmm.save(fname)
        loaded = MMM.load(fname, check=False)

        assert loaded.data.cost_per_unit is None
        raw_spend = loaded.data.get_channel_spend()
        xr.testing.assert_equal(raw_spend, loaded.idata.constant_data.channel_data)

        dates = loaded.data.dates
        cpu_df = pd.DataFrame(
            {
                "date": dates,
                "channel_1": [0.01] * len(dates),
            }
        )
        loaded.set_cost_per_unit(cpu_df)

        assert loaded.data.cost_per_unit is not None
        converted_spend = loaded.data.get_channel_spend()
        assert not converted_spend.equals(raw_spend)

    def test_save_load_set_cost_per_unit_posthoc(self, simple_fitted_mmm, tmp_path):
        """Test 22: Fit -> save -> load -> set_cost_per_unit -> save -> load."""
        mmm = simple_fitted_mmm

        fname1 = str(tmp_path / "model_v1.pm")
        mmm.save(fname1)
        loaded1 = MMM.load(fname1, check=False)

        dates = loaded1.data.dates
        cpu_df = pd.DataFrame(
            {
                "date": dates,
                "channel_1": [0.01] * len(dates),
                "channel_2": [0.02] * len(dates),
            }
        )
        loaded1.set_cost_per_unit(cpu_df)

        fname2 = str(tmp_path / "model_v2.pm")
        loaded1.save(fname2)
        loaded2 = MMM.load(fname2, check=False)

        assert loaded2.data.cost_per_unit is not None
        xr.testing.assert_equal(
            loaded2.data.cost_per_unit,
            loaded1.data.cost_per_unit,
        )

        xr.testing.assert_allclose(
            loaded2.data.get_channel_spend(),
            loaded1.data.get_channel_spend(),
        )


# ============================================================================
# Incrementality Integration Tests (Tests 20a-20b)
# ============================================================================


class TestIncrementalityCostPerUnit:
    """Tests 20a-20b: Incrementality integration with cost_per_unit."""

    def test_incremental_contribution_uses_raw_data_with_cost_per_unit(
        self, simple_fitted_mmm
    ):
        """Test 20a: Incremental contributions use raw data, not converted."""
        mmm = simple_fitted_mmm

        result_no_cpu = mmm.incrementality.compute_incremental_contribution(
            frequency="all_time"
        )

        dates = mmm.data.dates
        cpu_df = pd.DataFrame(
            {
                "date": dates,
                "channel_1": [0.01] * len(dates),
                "channel_2": [0.02] * len(dates),
            }
        )
        mmm.set_cost_per_unit(cpu_df)

        result_with_cpu = mmm.incrementality.compute_incremental_contribution(
            frequency="all_time"
        )

        xr.testing.assert_allclose(result_with_cpu, result_no_cpu)

    def test_roas_via_incrementality_with_cost_per_unit(self, simple_fitted_mmm):
        """Test 20b: ROAS via incrementality uses converted spend."""
        mmm = simple_fitted_mmm

        roas_no_cpu = mmm.incrementality.contribution_over_spend(frequency="all_time")

        dates = mmm.data.dates
        cpu_factor = 0.5
        cpu_df = pd.DataFrame(
            {
                "date": dates,
                "channel_1": [cpu_factor] * len(dates),
                "channel_2": [cpu_factor] * len(dates),
                "channel_3": [cpu_factor] * len(dates),
            }
        )
        mmm.set_cost_per_unit(cpu_df)

        roas_with_cpu = mmm.incrementality.contribution_over_spend(frequency="all_time")

        # ROAS_no_cpu = contributions / raw_spend
        # ROAS_with_cpu = contributions / (raw_spend * cpu_factor)
        # So: ROAS_with_cpu * cpu_factor = ROAS_no_cpu
        scaled_roas = roas_with_cpu * cpu_factor
        mask = np.isfinite(scaled_roas) & np.isfinite(roas_no_cpu)
        np.testing.assert_allclose(
            scaled_roas.values[mask],
            roas_no_cpu.values[mask],
            rtol=1e-5,
        )

    def test_aggregate_spend_correct_with_cost_per_unit(self, simple_fitted_mmm):
        """Regression test: time aggregation of spend must equal
        sum(channel_data * cost_per_unit), not sum(channel_data) * sum(cost_per_unit).
        """
        mmm = simple_fitted_mmm
        dates = mmm.data.dates

        cpu_values = np.linspace(0.01, 0.05, len(dates))
        cpu_df = pd.DataFrame(
            {
                "date": dates,
                "channel_1": cpu_values,
                "channel_2": cpu_values * 2,
            }
        )
        mmm.set_cost_per_unit(cpu_df)

        raw_data = mmm.data.get_channel_data()
        cpu_array = mmm.data.cost_per_unit
        expected_total_spend = (raw_data * cpu_array).sum(dim="date")

        actual_total_spend = mmm.incrementality._aggregate_spend(frequency="all_time")

        xr.testing.assert_allclose(actual_total_spend, expected_total_spend)
