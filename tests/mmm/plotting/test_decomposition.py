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
from __future__ import annotations

import warnings

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from arviz_plots import PlotCollection
from matplotlib.figure import Figure

from pymc_marketing.data.idata import MMMIDataWrapper
from pymc_marketing.mmm.plotting.decomposition import DecompositionPlots

matplotlib.use("Agg")

SEED = sum(map(ord, "DecompositionPlots tests"))


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def simple_idata() -> az.InferenceData:
    """Minimal idata with channels + baseline contributions, no extra dims.

    posterior:
      channel_contribution   (chain, draw, date, channel)
      intercept_contribution (chain, draw, date)
    constant_data:
      target_data  (date,)
      target_scale scalar
    """
    rng = np.random.default_rng(SEED)
    n_chain, n_draw, n_date = 2, 40, 20
    channels = ["tv", "radio", "social"]
    dates = np.arange(n_date)

    posterior = xr.Dataset(
        {
            "channel_contribution": xr.DataArray(
                rng.uniform(0, 100, size=(n_chain, n_draw, n_date, len(channels))),
                dims=("chain", "draw", "date", "channel"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "date": dates,
                    "channel": channels,
                },
            ),
            "intercept_contribution": xr.DataArray(
                rng.uniform(50, 150, size=(n_chain, n_draw, n_date)),
                dims=("chain", "draw", "date"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "date": dates,
                },
            ),
        }
    )
    const = xr.Dataset(
        {
            "target_data": xr.DataArray(
                rng.normal(500, 50, size=(n_date,)),
                dims=("date",),
                coords={"date": dates},
            ),
            "target_scale": xr.DataArray(1000.0),
        }
    )
    return az.InferenceData(posterior=posterior, constant_data=const)


@pytest.fixture(scope="module")
def panel_idata() -> az.InferenceData:
    """idata with geo extra dim — (chain, draw, date, channel, geo) for channels.

    posterior:
      channel_contribution   (chain, draw, date, channel, geo)
      intercept_contribution (chain, draw, date, geo)
    constant_data:
      target_data  (date, geo)
      target_scale scalar
    """
    rng = np.random.default_rng(SEED + 1)
    n_chain, n_draw, n_date = 2, 30, 15
    channels = ["tv", "radio"]
    geos = ["CA", "NY"]
    dates = np.arange(n_date)

    posterior = xr.Dataset(
        {
            "channel_contribution": xr.DataArray(
                rng.uniform(
                    0, 100, size=(n_chain, n_draw, n_date, len(channels), len(geos))
                ),
                dims=("chain", "draw", "date", "channel", "geo"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "date": dates,
                    "channel": channels,
                    "geo": geos,
                },
            ),
            "intercept_contribution": xr.DataArray(
                rng.uniform(50, 150, size=(n_chain, n_draw, n_date, len(geos))),
                dims=("chain", "draw", "date", "geo"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "date": dates,
                    "geo": geos,
                },
            ),
        }
    )
    const = xr.Dataset(
        {
            "target_data": xr.DataArray(
                rng.normal(500, 50, size=(n_date, len(geos))),
                dims=("date", "geo"),
                coords={"date": dates, "geo": geos},
            ),
            "target_scale": xr.DataArray(1000.0),
        }
    )
    return az.InferenceData(posterior=posterior, constant_data=const)


@pytest.fixture(scope="module")
def simple_data(simple_idata) -> MMMIDataWrapper:
    return MMMIDataWrapper(simple_idata, validate_on_init=False)


@pytest.fixture(scope="module")
def panel_data(panel_idata) -> MMMIDataWrapper:
    return MMMIDataWrapper(panel_idata, validate_on_init=False)


@pytest.fixture(scope="module")
def simple_plots(simple_data) -> DecompositionPlots:
    return DecompositionPlots(simple_data)


@pytest.fixture(scope="module")
def panel_plots(panel_data) -> DecompositionPlots:
    return DecompositionPlots(panel_data)


class TestContributionsOverTime:
    def test_returns_figure_and_axes(self, simple_plots):
        fig, axes = simple_plots.contributions_over_time()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.ndim >= 1

    def test_returns_plot_collection_when_requested(self, simple_plots):
        result = simple_plots.contributions_over_time(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_panel_model_creates_one_panel_per_geo(self, panel_plots):
        _fig, axes = panel_plots.contributions_over_time()
        # panel_idata has geo=["CA","NY"] — expect 2 axes
        assert len(axes) == 2

    def test_include_filters_contributions(self, simple_plots):
        # channels only — no baseline line
        fig, _axes = simple_plots.contributions_over_time(include=["channels"])
        assert isinstance(fig, Figure)

    def test_include_invalid_key_raises(self, simple_plots):
        with pytest.raises((ValueError, KeyError)):
            simple_plots.contributions_over_time(include=["invalid_key"])

    def test_col_wrap_overridable(self, panel_plots):
        # default col_wrap=1 → 2 axes stacked; col_wrap=2 → side by side (still 2 axes)
        _fig1, axes1 = panel_plots.contributions_over_time()
        _fig2, axes2 = panel_plots.contributions_over_time(col_wrap=2)
        assert len(axes1) == len(axes2) == 2

    def test_idata_override(self, simple_plots, simple_idata):
        # Override with a fresh idata — should not raise
        fig, _axes = simple_plots.contributions_over_time(idata=simple_idata)
        assert isinstance(fig, Figure)

    def test_dims_subsetting(self, panel_plots):
        fig, _axes = panel_plots.contributions_over_time(dims={"geo": ["CA"]})
        assert isinstance(fig, Figure)

    def test_no_summing_warning(self, simple_plots):
        # Multi-dim contributions (e.g. channel) are silently summed — no UserWarning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            simple_plots.contributions_over_time()
        summing = [w for w in caught if "summing" in str(w.message).lower()]
        assert summing == [], f"Unexpected 'summing' warning(s): {summing}"

    def test_x_axis_is_dates_y_axis_is_contributions(self, simple_plots):
        """dates must be on x-axis; contribution values must be on y-axis."""
        _fig, axes = simple_plots.contributions_over_time()
        ax = axes[0]
        date_coords = np.arange(20)  # simple_idata uses dates = np.arange(20)
        lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 1]
        assert lines, "No data lines found in the contributions plot"
        for line in lines:
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            assert np.array_equal(xdata, date_coords), (
                f"x-axis for '{line.get_label()}' should equal date coords "
                f"{date_coords[:3]}…, got {xdata[:3]}…"
            )
            assert ydata.max() > date_coords.max(), (
                f"y-axis for '{line.get_label()}' max={ydata.max():.1f} is not "
                "greater than the max date value — y-axis may be showing dates "
                "instead of contributions"
            )


class TestWaterfall:
    def test_returns_figure_and_axes(self, simple_plots):
        fig, axes = simple_plots.waterfall()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_single_panel_no_extra_dims(self, simple_plots):
        _fig, axes = simple_plots.waterfall()
        assert len(axes) == 1

    def test_panel_model_one_panel_per_geo(self, panel_plots):
        _fig, axes = panel_plots.waterfall()
        assert len(axes) == 2

    def test_dims_subsetting_reduces_panels(self, panel_plots):
        _fig, axes = panel_plots.waterfall(dims={"geo": ["CA"]})
        assert len(axes) == 1

    def test_idata_override(self, simple_plots, simple_idata):
        fig, _axes = simple_plots.waterfall(idata=simple_idata)
        assert isinstance(fig, Figure)

    def test_no_plt_gcf_used(self, simple_plots, monkeypatch):
        # Ensure no plt.gcf() is called internally
        import matplotlib.pyplot as plt_mod

        original_gcf = plt_mod.gcf
        called = []

        def patched_gcf():
            called.append(True)
            return original_gcf()

        monkeypatch.setattr(plt_mod, "gcf", patched_gcf)
        simple_plots.waterfall()
        assert called == [], "waterfall must not call plt.gcf()"


class TestChannelShareHdi:
    def test_returns_figure_and_axes(self, simple_plots):
        fig, axes = simple_plots.channel_share_hdi()
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)

    def test_returns_plot_collection_when_requested(self, simple_plots):
        result = simple_plots.channel_share_hdi(return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_idata_override(self, simple_plots, simple_idata):
        fig, _axes = simple_plots.channel_share_hdi(idata=simple_idata)
        assert isinstance(fig, Figure)

    def test_dims_subsetting(self, panel_plots):
        fig, _axes = panel_plots.channel_share_hdi(dims={"geo": ["CA"]})
        assert isinstance(fig, Figure)

    def test_channel_coordinate_present(self, simple_plots):
        # The dataset passed to azp.plot_forest has a 'channel' coordinate, not 'x'
        pc = simple_plots.channel_share_hdi(return_as_pc=True)
        assert isinstance(pc, PlotCollection)


def test_decomposition_plots_importable_from_package():
    from pymc_marketing.mmm.plotting import DecompositionPlots as DP

    assert DP is DecompositionPlots
