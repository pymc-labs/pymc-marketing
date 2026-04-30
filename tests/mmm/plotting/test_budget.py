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
"""Tests for pymc_marketing.mmm.plotting.budget.BudgetPlots."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from arviz_plots import PlotCollection
from matplotlib.figure import Figure

matplotlib.use("Agg")

SEED = sum(map(ord, "BudgetPlots tests"))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def channels():
    return ["tv", "radio", "social"]


@pytest.fixture(scope="module")
def simple_allocation_samples(channels) -> xr.Dataset:
    """xr.Dataset with channel_contribution_original_scale + allocation, no extra dims."""
    rng = np.random.default_rng(SEED)
    n_sample, n_date = 80, 20
    dates = np.arange(n_date)
    return xr.Dataset(
        {
            "channel_contribution_original_scale": xr.DataArray(
                rng.uniform(100, 500, (n_sample, n_date, len(channels))),
                dims=("sample", "date", "channel"),
                coords={
                    "sample": np.arange(n_sample),
                    "date": dates,
                    "channel": channels,
                },
            ),
            "allocation": xr.DataArray(
                rng.uniform(1000, 5000, len(channels)),
                dims=("channel",),
                coords={"channel": channels},
            ),
        }
    )


@pytest.fixture(scope="module")
def panel_allocation_samples(channels) -> xr.Dataset:
    """xr.Dataset with geo extra dim for panel tests."""
    rng = np.random.default_rng(SEED + 1)
    n_sample, n_date = 80, 20
    geos = ["CA", "NY"]
    dates = np.arange(n_date)
    return xr.Dataset(
        {
            "channel_contribution_original_scale": xr.DataArray(
                rng.uniform(100, 500, (n_sample, n_date, len(geos), len(channels))),
                dims=("sample", "date", "geo", "channel"),
                coords={
                    "sample": np.arange(n_sample),
                    "date": dates,
                    "geo": geos,
                    "channel": channels,
                },
            ),
            "allocation": xr.DataArray(
                rng.uniform(1000, 5000, (len(geos), len(channels))),
                dims=("geo", "channel"),
                coords={"geo": geos, "channel": channels},
            ),
        }
    )


@pytest.fixture(scope="module")
def simple_contribution_samples(channels) -> xr.Dataset:
    """xr.Dataset with channel_contribution_original_scale for contribution_over_time."""
    rng = np.random.default_rng(SEED + 2)
    n_sample, n_date = 80, 20
    dates = np.arange(n_date)
    return xr.Dataset(
        {
            "channel_contribution_original_scale": xr.DataArray(
                rng.uniform(0, 100, (n_sample, n_date, len(channels))),
                dims=("sample", "date", "channel"),
                coords={
                    "sample": np.arange(n_sample),
                    "date": dates,
                    "channel": channels,
                },
            ),
        }
    )


@pytest.fixture(scope="module")
def panel_contribution_samples(channels) -> xr.Dataset:
    """xr.Dataset with geo extra dim for contribution panel tests."""
    rng = np.random.default_rng(SEED + 3)
    n_sample, n_date = 80, 20
    geos = ["CA", "NY"]
    dates = np.arange(n_date)
    return xr.Dataset(
        {
            "channel_contribution_original_scale": xr.DataArray(
                rng.uniform(0, 100, (n_sample, n_date, len(geos), len(channels))),
                dims=("sample", "date", "geo", "channel"),
                coords={
                    "sample": np.arange(n_sample),
                    "date": dates,
                    "geo": geos,
                    "channel": channels,
                },
            ),
        }
    )


class TestAllocationRoas:
    def test_returns_figure_and_axes(self, simple_allocation_samples):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        fig, axes = BudgetPlots().allocation_roas(simple_allocation_samples)
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.ndim >= 1

    def test_returns_plot_collection_when_requested(self, simple_allocation_samples):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        result = BudgetPlots().allocation_roas(
            simple_allocation_samples, return_as_pc=True
        )
        assert isinstance(result, PlotCollection)

    def test_missing_channel_contribution_raises(self):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        bad_samples = xr.Dataset(
            {
                "allocation": xr.DataArray(
                    [1000.0, 2000.0],
                    dims=("channel",),
                    coords={"channel": ["tv", "radio"]},
                )
            }
        )
        with pytest.raises(ValueError, match="channel_contribution_original_scale"):
            BudgetPlots().allocation_roas(bad_samples)

    def test_missing_allocation_raises(self):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        rng = np.random.default_rng(SEED)
        bad_samples = xr.Dataset(
            {
                "channel_contribution_original_scale": xr.DataArray(
                    rng.uniform(0, 1, (10, 5, 2)),
                    dims=("sample", "date", "channel"),
                    coords={"channel": ["tv", "radio"]},
                )
            }
        )
        with pytest.raises(ValueError, match="allocation"):
            BudgetPlots().allocation_roas(bad_samples)

    def test_missing_channel_dim_raises(self):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        rng = np.random.default_rng(SEED)
        bad_samples = xr.Dataset(
            {
                "channel_contribution_original_scale": xr.DataArray(
                    rng.uniform(0, 1, (10, 5)),
                    dims=("sample", "date"),
                ),
                "allocation": xr.DataArray([1000.0], dims=("x",)),
            }
        )
        with pytest.raises(ValueError, match="channel"):
            BudgetPlots().allocation_roas(bad_samples)

    def test_dims_subsetting(self, panel_allocation_samples):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        fig, _axes = BudgetPlots().allocation_roas(
            panel_allocation_samples, dims={"geo": ["CA"]}
        )
        assert isinstance(fig, Figure)

    def test_hdi_prob_accepted(self, simple_allocation_samples):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        fig, _axes = BudgetPlots().allocation_roas(
            simple_allocation_samples, hdi_prob=0.89
        )
        assert isinstance(fig, Figure)

    def test_figsize_accepted(self, simple_allocation_samples):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        fig, _axes = BudgetPlots().allocation_roas(
            simple_allocation_samples, figsize=(8, 4)
        )
        assert isinstance(fig, Figure)


class TestContributionOverTime:
    def test_returns_figure_and_axes(self, simple_contribution_samples):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        fig, axes = BudgetPlots().contribution_over_time(simple_contribution_samples)
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.ndim >= 1

    def test_returns_plot_collection_when_requested(self, simple_contribution_samples):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        result = BudgetPlots().contribution_over_time(
            simple_contribution_samples, return_as_pc=True
        )
        assert isinstance(result, PlotCollection)

    def test_missing_channel_dim_raises(self):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        bad = xr.Dataset(
            {
                "channel_contribution_original_scale": xr.DataArray(
                    np.ones((10, 5)),
                    dims=("sample", "date"),
                )
            }
        )
        with pytest.raises(ValueError, match="channel"):
            BudgetPlots().contribution_over_time(bad)

    def test_missing_date_dim_raises(self):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        bad = xr.Dataset(
            {
                "channel_contribution_original_scale": xr.DataArray(
                    np.ones((10, 3)),
                    dims=("sample", "channel"),
                    coords={"channel": ["tv", "radio", "social"]},
                )
            }
        )
        with pytest.raises(ValueError, match="date"):
            BudgetPlots().contribution_over_time(bad)

    def test_missing_sample_dim_raises(self):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        bad = xr.Dataset(
            {
                "channel_contribution_original_scale": xr.DataArray(
                    np.ones((5, 3)),
                    dims=("date", "channel"),
                    coords={"channel": ["tv", "radio", "social"]},
                )
            }
        )
        with pytest.raises(ValueError, match="sample"):
            BudgetPlots().contribution_over_time(bad)

    def test_missing_channel_contribution_var_raises(self):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        bad = xr.Dataset(
            {
                "other_var": xr.DataArray(
                    np.ones((10, 5, 3)),
                    dims=("sample", "date", "channel"),
                    coords={"channel": ["tv", "radio", "social"]},
                )
            }
        )
        with pytest.raises(ValueError, match="channel_contribution"):
            BudgetPlots().contribution_over_time(bad)

    def test_panel_model_creates_one_panel_per_geo(self, panel_contribution_samples):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        _fig, axes = BudgetPlots().contribution_over_time(panel_contribution_samples)
        assert len(axes) == 2

    def test_dims_subsetting(self, panel_contribution_samples):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        fig, _axes = BudgetPlots().contribution_over_time(
            panel_contribution_samples, dims={"geo": ["CA"]}
        )
        assert isinstance(fig, Figure)

    def test_each_channel_has_own_line(self, simple_contribution_samples, channels):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        _fig, axes = BudgetPlots().contribution_over_time(simple_contribution_samples)
        ax = axes[0]
        data_lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 1]
        assert len(data_lines) == len(channels), (
            f"Expected {len(channels)} lines (one per channel), got {len(data_lines)}"
        )

    def test_hdi_prob_accepted(self, simple_contribution_samples):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        fig, _axes = BudgetPlots().contribution_over_time(
            simple_contribution_samples, hdi_prob=0.89
        )
        assert isinstance(fig, Figure)

    def test_figsize_accepted(self, simple_contribution_samples):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        fig, _axes = BudgetPlots().contribution_over_time(
            simple_contribution_samples, figsize=(10, 4)
        )
        assert isinstance(fig, Figure)


class TestImports:
    def test_budget_plots_importable_from_plotting_package(self):
        from pymc_marketing.mmm.plotting import BudgetPlots

        assert callable(BudgetPlots)

    def test_mmmplotsuite_has_budget_property(self):
        from pymc_marketing.mmm.plot import MMMPlotSuite
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        suite = MMMPlotSuite.__new__(MMMPlotSuite)
        budget = suite.budget
        assert isinstance(budget, BudgetPlots)
