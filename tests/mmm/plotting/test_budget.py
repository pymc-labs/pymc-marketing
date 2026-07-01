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

from pymc_marketing.mmm.plotting.budget import BudgetPlots

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
    contrib = rng.uniform(100, 500, (n_sample, n_date, len(channels)))
    alloc = rng.uniform(1000, 5000, len(channels))
    return xr.Dataset(
        {
            "channel_contribution_original_scale": xr.DataArray(
                contrib,
                dims=("sample", "date", "channel"),
                coords={
                    "sample": np.arange(n_sample),
                    "date": dates,
                    "channel": channels,
                },
            ),
            "allocation": xr.DataArray(
                alloc,
                dims=("channel",),
                coords={"channel": channels},
            ),
            "total_allocation": xr.DataArray(
                alloc * n_date,
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
    contrib = rng.uniform(100, 500, (n_sample, n_date, len(geos), len(channels)))
    alloc = rng.uniform(1000, 5000, (len(geos), len(channels)))
    return xr.Dataset(
        {
            "channel_contribution_original_scale": xr.DataArray(
                contrib,
                dims=("sample", "date", "geo", "channel"),
                coords={
                    "sample": np.arange(n_sample),
                    "date": dates,
                    "geo": geos,
                    "channel": channels,
                },
            ),
            "allocation": xr.DataArray(
                alloc,
                dims=("geo", "channel"),
                coords={"geo": geos, "channel": channels},
            ),
            "total_allocation": xr.DataArray(
                alloc * n_date,
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

    def test_missing_total_allocation_raises(self):
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        rng = np.random.default_rng(SEED)
        channels = ["tv", "radio"]
        bad_samples = xr.Dataset(
            {
                "channel_contribution_original_scale": xr.DataArray(
                    rng.uniform(0, 1, (10, 5, 2)),
                    dims=("sample", "date", "channel"),
                    coords={"channel": channels},
                ),
                "allocation": xr.DataArray(
                    [1000.0, 2000.0],
                    dims=("channel",),
                    coords={"channel": channels},
                ),
            }
        )
        with pytest.raises(ValueError, match="total_allocation"):
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

    def test_panel_titles_include_geo_coords(self, panel_contribution_samples):
        """Regression: facet panels must be titled with their geo coordinate."""
        from pymc_marketing.mmm.plotting.budget import BudgetPlots

        _fig, axes = BudgetPlots().contribution_over_time(panel_contribution_samples)
        titles = [ax.get_title() for ax in axes.ravel() if ax.lines]
        joined = " ".join(titles)
        assert "CA" in joined and "NY" in joined, (
            f"Expected per-panel geo titles, got {titles!r}"
        )

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


def _axis_by_metric(fig, metric: str):
    """Return the first axis whose title mentions ``metric``."""
    return next(ax for ax in fig.get_axes() if metric in ax.get_title())


class TestBudgetAllocation:
    def test_returns_figure_and_axes(self, simple_allocation_samples):
        fig, axes = BudgetPlots().budget_allocation(simple_allocation_samples)
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.ndim >= 1

    def test_returns_plot_collection_when_requested(self, simple_allocation_samples):
        result = BudgetPlots().budget_allocation(
            simple_allocation_samples, return_as_pc=True
        )
        assert isinstance(result, PlotCollection)

    def test_simple_has_two_metric_panels(self, simple_allocation_samples):
        fig, _axes = BudgetPlots().budget_allocation(simple_allocation_samples)
        assert len(fig.get_axes()) == 2

    def test_panel_model_creates_two_panels_per_geo(self, panel_allocation_samples):
        fig, _axes = BudgetPlots().budget_allocation(panel_allocation_samples)
        # 2 metrics x 2 geos
        assert len(fig.get_axes()) == 4

    def test_metric_panels_have_independent_scales(self, simple_allocation_samples):
        """Spend and contribution live on different y-scales (core design)."""
        fig, _axes = BudgetPlots().budget_allocation(simple_allocation_samples)
        spend = _axis_by_metric(fig, "Allocated Spend")
        contribution = _axis_by_metric(fig, "Channel Contribution")
        # total_allocation = allocation * n_date >> contribution summed over date
        assert spend.get_ylim()[1] > contribution.get_ylim()[1]

    def test_point_and_whisker_per_channel(self, simple_allocation_samples, channels):
        fig, _axes = BudgetPlots().budget_allocation(simple_allocation_samples)
        ax = fig.get_axes()[0]
        assert len(ax.get_lines()) == len(channels)
        assert len(ax.collections) == len(channels)

    def test_allocation_whisker_is_degenerate(self, simple_allocation_samples):
        """Deterministic spend -> zero-height whisker; contribution has HDI."""
        fig, _axes = BudgetPlots().budget_allocation(simple_allocation_samples)
        spend = _axis_by_metric(fig, "Allocated Spend")
        contribution = _axis_by_metric(fig, "Channel Contribution")
        for line in spend.get_lines():
            ydata = line.get_ydata()
            assert np.isclose(ydata[0], ydata[-1])
        assert any(
            not np.isclose(line.get_ydata()[0], line.get_ydata()[-1])
            for line in contribution.get_lines()
        )

    def test_xticklabels_are_channel_names(self, simple_allocation_samples, channels):
        fig, _axes = BudgetPlots().budget_allocation(simple_allocation_samples)
        ax = fig.get_axes()[0]
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert labels == list(channels)

    def test_dims_subsetting(self, panel_allocation_samples):
        fig, _axes = BudgetPlots().budget_allocation(
            panel_allocation_samples, dims={"geo": ["CA"]}
        )
        assert len(fig.get_axes()) == 2

    def test_hdi_prob_accepted(self, simple_allocation_samples):
        fig, _axes = BudgetPlots().budget_allocation(
            simple_allocation_samples, hdi_prob=0.89
        )
        assert isinstance(fig, Figure)

    def test_figsize_accepted(self, simple_allocation_samples):
        fig, _axes = BudgetPlots().budget_allocation(
            simple_allocation_samples, figsize=(10, 5)
        )
        assert isinstance(fig, Figure)

    def test_point_and_hdi_kwargs_forwarded(self, simple_allocation_samples):
        fig, _axes = BudgetPlots().budget_allocation(
            simple_allocation_samples,
            point_kwargs={"marker": "s"},
            hdi_kwargs={"linewidth": 3.0},
        )
        assert isinstance(fig, Figure)

    def test_missing_channel_contribution_raises(self):
        bad_samples = xr.Dataset(
            {
                "total_allocation": xr.DataArray(
                    [1000.0, 2000.0],
                    dims=("channel",),
                    coords={"channel": ["tv", "radio"]},
                )
            }
        )
        with pytest.raises(ValueError, match="channel_contribution_original_scale"):
            BudgetPlots().budget_allocation(bad_samples)

    def test_missing_total_allocation_raises(self):
        rng = np.random.default_rng(SEED)
        channels = ["tv", "radio"]
        bad_samples = xr.Dataset(
            {
                "channel_contribution_original_scale": xr.DataArray(
                    rng.uniform(0, 1, (10, 5, 2)),
                    dims=("sample", "date", "channel"),
                    coords={"channel": channels},
                )
            }
        )
        with pytest.raises(ValueError, match="total_allocation"):
            BudgetPlots().budget_allocation(bad_samples)

    def test_missing_channel_dim_raises(self):
        rng = np.random.default_rng(SEED)
        bad_samples = xr.Dataset(
            {
                "channel_contribution_original_scale": xr.DataArray(
                    rng.uniform(0, 1, (10, 5)),
                    dims=("sample", "date"),
                ),
                "total_allocation": xr.DataArray([1000.0], dims=("x",)),
            }
        )
        with pytest.raises(ValueError, match="channel"):
            BudgetPlots().budget_allocation(bad_samples)

    def test_backend_without_return_as_pc_raises(self, simple_allocation_samples):
        with pytest.raises(ValueError, match="return_as_pc=True"):
            BudgetPlots().budget_allocation(simple_allocation_samples, backend="plotly")
