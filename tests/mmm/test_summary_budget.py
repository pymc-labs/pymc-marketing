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
"""Tests for BudgetSummaryFactory summary DataFrames."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.summary_budget import BudgetSummaryFactory

SEED = sum(map(ord, "budget_summary_tests"))
rng = np.random.default_rng(seed=SEED)


@pytest.fixture
def budget_allocation_samples():
    """Minimal budget allocation samples for ROAS summary tests."""
    channels = ["TV", "Radio"]
    dates = pd.date_range("2024-01-01", periods=8, freq="W")
    n_chains, n_draws = 2, 5

    return xr.Dataset(
        {
            "channel_contribution_original_scale": xr.DataArray(
                rng.uniform(
                    100, 500, size=(n_chains, n_draws, len(dates), len(channels))
                ),
                dims=("chain", "draw", "date", "channel"),
                coords={
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "date": dates,
                    "channel": channels,
                },
            ),
            "allocation": xr.DataArray(
                rng.uniform(50, 200, size=(n_chains, n_draws, len(channels))),
                dims=("chain", "draw", "channel"),
                coords={
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "channel": channels,
                },
            ),
            "total_allocation": xr.DataArray(
                rng.uniform(200, 400, size=(n_chains, n_draws)),
                dims=("chain", "draw"),
                coords={
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                },
            ),
        }
    )


@pytest.fixture
def budget_contribution_samples():
    """Minimal budget allocation samples for contribution-over-time tests."""
    channels = ["TV", "Radio"]
    dates = pd.date_range("2024-01-01", periods=8, freq="W")
    n_chains, n_draws = 2, 5

    return xr.Dataset(
        {
            "channel_contribution_original_scale": xr.DataArray(
                rng.uniform(
                    50, 300, size=(n_chains, n_draws, len(dates), len(channels))
                ),
                dims=("chain", "draw", "date", "channel"),
                coords={
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "date": dates,
                    "channel": channels,
                },
            ),
        }
    )


class TestBudgetSummarySchemas:
    """Test BudgetSummaryFactory DataFrame schemas."""

    def test_allocation_roas_schema(self, budget_allocation_samples):
        """Test allocation_roas returns DataFrame with correct schema."""
        df = BudgetSummaryFactory.allocation_roas(
            budget_allocation_samples,
            hdi_probs=[0.94],
        )

        required_columns = {"channel", "mean", "median"}
        assert required_columns.issubset(set(df.columns))
        assert "abs_error_94_lower" in df.columns
        assert "abs_error_94_upper" in df.columns
        assert len(df) == budget_allocation_samples.sizes["channel"]

    def test_contribution_over_time_schema(self, budget_contribution_samples):
        """Test contribution_over_time returns DataFrame with correct schema."""
        df = BudgetSummaryFactory.contribution_over_time(
            budget_contribution_samples,
            hdi_probs=[0.94],
        )

        required_columns = {"date", "channel", "mean", "median"}
        assert required_columns.issubset(set(df.columns))
        assert "abs_error_94_lower" in df.columns
        assert "abs_error_94_upper" in df.columns
        expected_rows = (
            budget_contribution_samples.sizes["date"]
            * budget_contribution_samples.sizes["channel"]
        )
        assert len(df) == expected_rows
