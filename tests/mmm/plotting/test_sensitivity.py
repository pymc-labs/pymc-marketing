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
"""Tests for pymc_marketing.mmm.plotting.sensitivity."""

from __future__ import annotations

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pymc_marketing.data.idata import MMMIDataWrapper
from pymc_marketing.mmm.plotting.sensitivity import SensitivityPlots

matplotlib.use("Agg")

SEED = sum(map(ord, "SensitivityPlots tests"))


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to prevent memory warnings."""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def simple_sa_idata() -> az.InferenceData:
    """InferenceData with a sensitivity_analysis group.

    Dims: (sample=50, sweep=11, channel=3).
    Sweep coords: np.linspace(0.5, 1.5, 11).
    """
    rng = np.random.default_rng(SEED)
    channels = ["tv", "radio", "social"]
    n_sample = 50
    n_sweep = 11
    sweep_coords = np.linspace(0.5, 1.5, n_sweep)
    shape = (n_sample, n_sweep, 3)
    coords = {
        "sample": np.arange(n_sample),
        "sweep": sweep_coords,
        "channel": channels,
    }

    sa_ds = xr.Dataset(
        {
            "x": xr.DataArray(
                rng.normal(size=shape),
                dims=("sample", "sweep", "channel"),
                coords=coords,
            ),
            "uplift_curve": xr.DataArray(
                rng.normal(size=shape),
                dims=("sample", "sweep", "channel"),
                coords=coords,
            ),
            "marginal_effects": xr.DataArray(
                rng.normal(size=shape),
                dims=("sample", "sweep", "channel"),
                coords=coords,
            ),
        }
    )

    constant_data = xr.Dataset(
        {
            "channel_data": xr.DataArray(
                rng.uniform(0, 10, size=(20, 3)),
                dims=("date", "channel"),
                coords={"date": np.arange(20), "channel": channels},
            ),
            "channel_spend": xr.DataArray(
                rng.uniform(100, 1000, size=(20, 3)),
                dims=("date", "channel"),
                coords={"date": np.arange(20), "channel": channels},
            ),
        }
    )

    return az.InferenceData(sensitivity_analysis=sa_ds, constant_data=constant_data)


@pytest.fixture(scope="module")
def sensitivity_plots(simple_sa_idata) -> SensitivityPlots:
    data = MMMIDataWrapper(simple_sa_idata, validate_on_init=False)
    return SensitivityPlots(data)


def test_analysis_returns_tuple(sensitivity_plots):
    fig, axes = sensitivity_plots.analysis()
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)
    assert all(isinstance(a, Axes) for a in axes.flat)


def test_analysis_return_as_pc(sensitivity_plots):
    result = sensitivity_plots.analysis(return_as_pc=True)
    assert isinstance(result, PlotCollection)


def test_uplift_reads_correct_key(sensitivity_plots):
    fig, axes = sensitivity_plots.uplift()
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)


def test_marginal_reads_correct_key(sensitivity_plots):
    fig, axes = sensitivity_plots.marginal()
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray)


def test_missing_sa_group_raises(simple_sa_idata):
    idata_no_sa = az.InferenceData(constant_data=simple_sa_idata.constant_data)
    data = MMMIDataWrapper(idata_no_sa, validate_on_init=False)
    plots = SensitivityPlots(data)
    with pytest.raises(ValueError, match="sensitivity_analysis"):
        plots.analysis()


def test_missing_key_raises(simple_sa_idata):
    sa_ds = xr.Dataset({"x": simple_sa_idata.sensitivity_analysis["x"]})
    idata_partial = az.InferenceData(
        sensitivity_analysis=sa_ds,
        constant_data=simple_sa_idata.constant_data,
    )
    data = MMMIDataWrapper(idata_partial, validate_on_init=False)
    plots = SensitivityPlots(data)

    with pytest.raises(ValueError, match="uplift_curve"):
        plots.uplift()

    with pytest.raises(ValueError, match="marginal_effects"):
        plots.marginal()


def test_dims_filtering(sensitivity_plots):
    # Without filtering: 3 channels as hue → 3 mean lines in the single panel
    _, axes_full = sensitivity_plots.analysis()
    lines_full = axes_full.flat[0].get_lines()

    # With filter: 1 channel → 1 mean line
    _, axes_filtered = sensitivity_plots.analysis(dims={"channel": ["tv"]})
    lines_filtered = axes_filtered.flat[0].get_lines()

    assert len(lines_filtered) < len(lines_full)


def test_aggregation_str(sensitivity_plots):
    # Summing over channel removes the hue dim → 1 mean line per panel
    _, axes = sensitivity_plots.analysis(aggregation={"sum": "channel"})
    lines = axes.flat[0].get_lines()
    assert len(lines) == 1


def test_aggregation_list(sensitivity_plots):
    # str and list forms should produce the same result
    _, axes_str = sensitivity_plots.analysis(aggregation={"sum": "channel"})
    _, axes_list = sensitivity_plots.analysis(aggregation={"sum": ["channel"]})
    assert len(axes_str.flat[0].get_lines()) == len(axes_list.flat[0].get_lines())


def test_idata_override(sensitivity_plots, simple_sa_idata):
    rng = np.random.default_rng(42)
    channels = ["tv", "radio", "social"]
    sweep_coords = np.linspace(0.5, 1.5, 11)
    coords = {"sample": np.arange(50), "sweep": sweep_coords, "channel": channels}

    new_sa_ds = xr.Dataset(
        {
            "x": xr.DataArray(
                rng.normal(0, 5, size=(50, 11, 3)),
                dims=("sample", "sweep", "channel"),
                coords=coords,
            )
        }
    )
    new_idata = az.InferenceData(
        sensitivity_analysis=new_sa_ds,
        constant_data=simple_sa_idata.constant_data,
    )

    fig, _ = sensitivity_plots.analysis(idata=new_idata)
    assert isinstance(fig, Figure)
    # self._data must be unchanged
    assert sensitivity_plots._data.idata is simple_sa_idata


def test_sweep_x_values_relative(sensitivity_plots):
    sweep_coords = np.linspace(0.5, 1.5, 11)
    _, axes = sensitivity_plots.analysis(x_sweep_axis="relative")
    ax = axes.flat[0]
    for line in ax.get_lines():
        np.testing.assert_allclose(line.get_xdata(), sweep_coords, rtol=1e-5)


def test_x_sweep_axis_absolute(sensitivity_plots, simple_sa_idata):
    channels = ["tv", "radio", "social"]
    sweep_coords = np.linspace(0.5, 1.5, 11)
    channel_spend = simple_sa_idata.constant_data.channel_spend  # (date, channel)

    _, axes = sensitivity_plots.analysis(
        x_sweep_axis="absolute", apply_cost_per_unit=True
    )
    ax = axes.flat[0]
    lines = ax.get_lines()

    # Lines are drawn in channel coordinate order (tv, radio, social)
    for i, ch in enumerate(channels):
        expected_x = sweep_coords * float(channel_spend.sel(channel=ch).sum("date"))
        np.testing.assert_allclose(lines[i].get_xdata(), expected_x, rtol=1e-4)


def test_custom_rows_cols_in_pc_kwargs(sensitivity_plots):
    # cols=["channel"] makes channel a panel dimension → 3 panels, no hue
    _, axes = sensitivity_plots.analysis(cols=["channel"])
    assert axes.size == 3


def test_n_lines_equals_hue_cardinality(sensitivity_plots):
    # Default layout: no custom_dims → cols=[], channel is hue → 3 mean lines
    _, axes = sensitivity_plots.analysis()
    ax = axes.flat[0]
    lines = ax.get_lines()
    assert len(lines) == 3


def test_mean_line_values(sensitivity_plots, simple_sa_idata):
    channels = ["tv", "radio", "social"]
    sa_da = simple_sa_idata.sensitivity_analysis["x"]
    # mean over sample is equivalent to mean over (chain, draw) after conversion
    expected_mean = sa_da.mean("sample")  # (sweep, channel)

    _, axes = sensitivity_plots.analysis(x_sweep_axis="relative")
    ax = axes.flat[0]
    lines = ax.get_lines()

    for i, ch in enumerate(channels):
        np.testing.assert_allclose(
            lines[i].get_ydata(),
            expected_mean.sel(channel=ch).values,
            rtol=1e-5,
        )


def test_uplift_has_vertical_line_at_one(sensitivity_plots):
    _, axes = sensitivity_plots.uplift()
    ax = axes.flat[0]
    vlines = [line for line in ax.get_lines() if list(line.get_xdata()) == [1.0, 1.0]]
    assert vlines, "Expected a vertical reference line at x=1.0"


def test_uplift_has_horizontal_line_at_zero(sensitivity_plots):
    _, axes = sensitivity_plots.uplift()
    ax = axes.flat[0]
    hlines = [line for line in ax.get_lines() if list(line.get_ydata()) == [0.0, 0.0]]
    assert hlines, "Expected a horizontal reference line at y=0.0"


def test_hdi_band_values(sensitivity_plots, simple_sa_idata):
    n_channels = 3

    _, axes = sensitivity_plots.analysis(x_sweep_axis="relative", hdi_prob=0.94)
    ax = axes.flat[0]
    lines = ax.get_lines()
    polys = [c for c in ax.collections if "Poly" in type(c).__name__]

    # One PolyCollection per channel (fill_between per hue value)
    assert len(polys) == n_channels

    # Each mean line must fall within its HDI band
    for line, poly in zip(lines, polys, strict=False):
        y_mean = line.get_ydata()
        n = len(y_mean)
        verts = poly.get_paths()[0].vertices
        y_lower = verts[1 : n + 1, 1]
        y_upper = verts[n + 2 : 2 * n + 2, 1][::-1]
        np.testing.assert_array_less(y_lower - 1e-6, y_mean)
        np.testing.assert_array_less(y_mean, y_upper + 1e-6)
