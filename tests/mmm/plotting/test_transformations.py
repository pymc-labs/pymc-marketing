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
"""Tests for pymc_marketing.mmm.plotting.transformations."""

from __future__ import annotations

import arviz as az
import matplotlib
import numpy as np
import pytest
import xarray as xr
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pymc_marketing.data.idata import MMMIDataWrapper
from pymc_marketing.mmm.plotting.transformations import TransformationPlots

matplotlib.use("Agg")

SEED = sum(map(ord, "TransformationPlots tests"))
RNG = np.random.default_rng(SEED)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def simple_idata() -> az.InferenceData:
    """InferenceData with dims (chain, draw, date, channel) — no custom dims."""
    rng = np.random.default_rng(SEED)
    dates = np.arange(20)
    channels = ["tv", "radio", "social"]

    posterior = xr.Dataset(
        {
            "channel_contribution": xr.DataArray(
                rng.normal(size=(2, 50, 20, 3)),
                dims=("chain", "draw", "date", "channel"),
                coords={
                    "chain": np.arange(2),
                    "draw": np.arange(50),
                    "date": dates,
                    "channel": channels,
                },
            ),
            "channel_contribution_original_scale": xr.DataArray(
                rng.normal(size=(2, 50, 20, 3)) * 100,
                dims=("chain", "draw", "date", "channel"),
                coords={
                    "chain": np.arange(2),
                    "draw": np.arange(50),
                    "date": dates,
                    "channel": channels,
                },
            ),
        }
    )

    constant_data = xr.Dataset(
        {
            "channel_data": xr.DataArray(
                rng.uniform(0, 10, size=(20, 3)),
                dims=("date", "channel"),
                coords={"date": dates, "channel": channels},
            ),
            "channel_scale": xr.DataArray(
                [100.0, 150.0, 200.0],
                dims=("channel",),
                coords={"channel": channels},
            ),
            "target_scale": xr.DataArray(1000.0),
        }
    )

    return az.InferenceData(posterior=posterior, constant_data=constant_data)


@pytest.fixture(scope="module")
def panel_idata() -> az.InferenceData:
    """InferenceData with custom dim 'country' — (chain, draw, date, channel, country)."""
    rng = np.random.default_rng(SEED + 1)
    dates = np.arange(15)
    channels = ["tv", "radio"]
    countries = ["US", "UK"]

    posterior = xr.Dataset(
        {
            "channel_contribution": xr.DataArray(
                rng.normal(size=(2, 30, 15, 2, 2)),
                dims=("chain", "draw", "date", "channel", "country"),
                coords={
                    "chain": np.arange(2),
                    "draw": np.arange(30),
                    "date": dates,
                    "channel": channels,
                    "country": countries,
                },
            ),
            "channel_contribution_original_scale": xr.DataArray(
                rng.normal(size=(2, 30, 15, 2, 2)) * 100,
                dims=("chain", "draw", "date", "channel", "country"),
                coords={
                    "chain": np.arange(2),
                    "draw": np.arange(30),
                    "date": dates,
                    "channel": channels,
                    "country": countries,
                },
            ),
        }
    )

    constant_data = xr.Dataset(
        {
            "channel_data": xr.DataArray(
                rng.uniform(0, 10, size=(15, 2, 2)),
                dims=("date", "channel", "country"),
                coords={"date": dates, "channel": channels, "country": countries},
            ),
            "channel_spend": xr.DataArray(
                rng.uniform(10, 100, size=(15, 2, 2)),
                dims=("date", "channel", "country"),
                coords={"date": dates, "channel": channels, "country": countries},
            ),
            "channel_scale": xr.DataArray(
                [[100.0, 150.0], [120.0, 180.0]],
                dims=("country", "channel"),
                coords={"country": countries, "channel": channels},
            ),
            "target_scale": xr.DataArray(1000.0),
        }
    )

    return az.InferenceData(posterior=posterior, constant_data=constant_data)


@pytest.fixture(scope="module")
def simple_data(simple_idata) -> MMMIDataWrapper:
    return MMMIDataWrapper(simple_idata, validate_on_init=False)


@pytest.fixture(scope="module")
def panel_data(panel_idata) -> MMMIDataWrapper:
    return MMMIDataWrapper(panel_idata, validate_on_init=False)


@pytest.fixture(scope="module")
def simple_plots(simple_data) -> TransformationPlots:
    return TransformationPlots(simple_data)


@pytest.fixture(scope="module")
def panel_plots(panel_data) -> TransformationPlots:
    return TransformationPlots(panel_data)


# ============================================================================
# saturation_scatterplot tests
# ============================================================================


class TestSaturationScatterplotBasic:
    def test_returns_figure_and_axes(self, simple_plots):
        fig, axes = simple_plots.saturation_scatterplot()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(a, Axes) for a in axes.flat)

    def test_original_scale_true_is_default(self, simple_plots):
        fig, _axes = simple_plots.saturation_scatterplot()
        assert isinstance(fig, Figure)

    def test_original_scale_false(self, simple_plots):
        fig, _axes = simple_plots.saturation_scatterplot(original_scale=False)
        assert isinstance(fig, Figure)

    def test_axes_count_matches_channels(self, simple_plots, simple_data):
        _, axes = simple_plots.saturation_scatterplot()
        assert axes.size == len(simple_data.channels)

    def test_panels_have_scatter_data(self, simple_plots):
        _, axes = simple_plots.saturation_scatterplot()
        for ax in axes.flat:
            collections = ax.collections
            assert len(collections) > 0, "Each panel should have scatter points"


class TestSaturationScatterplotDims:
    def test_single_dim_value(self, panel_plots):
        _, axes = panel_plots.saturation_scatterplot(dims={"country": "US"})
        assert axes.size == 2  # 2 channels x 1 country

    def test_list_dim_value(self, panel_plots):
        _, axes = panel_plots.saturation_scatterplot(dims={"country": ["US", "UK"]})
        assert axes.size == 4  # 2 channels x 2 countries

    def test_channel_subsetting(self, panel_plots):
        _, axes = panel_plots.saturation_scatterplot(
            dims={"channel": "tv", "country": "US"}
        )
        assert axes.size == 1  # 1 channel x 1 country

    def test_invalid_dim_name_raises(self, simple_plots):
        with pytest.raises(ValueError, match="Dimension 'region' not found"):
            simple_plots.saturation_scatterplot(dims={"region": "US"})

    def test_invalid_dim_value_raises(self, panel_plots):
        with pytest.raises(ValueError, match="Value 'FR' not found"):
            panel_plots.saturation_scatterplot(dims={"country": "FR"})


class TestSaturationScatterplotCustomization:
    def test_return_as_pc_true(self, simple_plots):
        result = simple_plots.saturation_scatterplot(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_custom_figsize(self, simple_plots):
        fig, _ = simple_plots.saturation_scatterplot(figsize=(20, 10))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(20, abs=1)
        assert h == pytest.approx(10, abs=1)

    def test_non_matplotlib_backend_without_return_as_pc_raises(self, simple_plots):
        with pytest.raises(ValueError, match="return_as_pc=True"):
            simple_plots.saturation_scatterplot(backend="plotly")


class TestSaturationScatterplotIdataOverride:
    def test_idata_override_uses_different_data(self, simple_plots, panel_idata):
        """When passing idata override, the method uses data from that idata."""
        _, axes = simple_plots.saturation_scatterplot(idata=panel_idata)
        assert axes.size == 4  # panel_idata has 2 channels x 2 countries

    def test_idata_override_does_not_mutate_self_data(
        self, simple_plots, simple_data, panel_idata
    ):
        simple_plots.saturation_scatterplot(idata=panel_idata)
        assert simple_plots._data is simple_data


class TestSaturationScatterplotLabels:
    def test_ylabel_is_contributions(self, simple_plots):
        _, axes = simple_plots.saturation_scatterplot()
        for ax in axes.flat:
            assert "Contribution" in ax.get_ylabel()

    def test_xlabel_without_cost_per_unit(self, simple_plots):
        _, axes = simple_plots.saturation_scatterplot(apply_cost_per_unit=False)
        for ax in axes.flat:
            assert "Channel Data" in ax.get_xlabel()

    def test_xlabel_with_cost_per_unit_and_spend(self, panel_plots):
        _, axes = panel_plots.saturation_scatterplot(apply_cost_per_unit=True)
        for ax in axes.flat:
            assert "Spend" in ax.get_xlabel()


class TestSaturationScatterplotCostPerUnit:
    def test_apply_cost_per_unit_true(self, panel_plots):
        fig, _axes = panel_plots.saturation_scatterplot(apply_cost_per_unit=True)
        assert isinstance(fig, Figure)

    def test_apply_cost_per_unit_false(self, panel_plots):
        fig, _axes = panel_plots.saturation_scatterplot(apply_cost_per_unit=False)
        assert isinstance(fig, Figure)
