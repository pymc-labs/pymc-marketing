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

import warnings

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
from pymc_marketing.mmm.plotting.transformations import (
    _SCALED_SPACE_MAX_THRESHOLD,
    TransformationPlots,
    _ensure_chain_draw_dims,
)

matplotlib.use("Agg")

SEED = sum(map(ord, "TransformationPlots tests"))
RNG = np.random.default_rng(SEED)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to prevent memory warnings."""
    yield
    plt.close("all")


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


# ============================================================================
# Additional fixtures for saturation_curves
# ============================================================================


def _make_simple_curve(rng, scale: float = 1.0) -> xr.DataArray:
    """Build a simple saturation curve; multiply y-values by *scale*."""
    x_values = np.linspace(0, 1, 50)
    channels = ["tv", "radio", "social"]

    data = np.empty((2, 50, 3, 50))
    for ci in range(2):
        for di in range(50):
            for c in range(3):
                data[ci, di, c, :] = x_values / (1 + x_values) + rng.normal(
                    0, 0.01, size=50
                )
    data *= scale

    return xr.DataArray(
        data,
        dims=("chain", "draw", "channel", "x"),
        coords={
            "chain": np.arange(2),
            "draw": np.arange(50),
            "channel": channels,
            "x": x_values,
        },
    )


def _make_panel_curve(rng, scale: float = 1.0) -> xr.DataArray:
    """Build a panel saturation curve; multiply y-values by *scale*."""
    x_values = np.linspace(0, 1, 40)
    channels = ["tv", "radio"]
    countries = ["US", "UK"]

    data = np.empty((2, 30, 2, 2, 40))
    for ci in range(2):
        for di in range(30):
            for ch in range(2):
                for co in range(2):
                    data[ci, di, ch, co, :] = x_values / (1 + x_values) + rng.normal(
                        0, 0.01, size=40
                    )
    data *= scale

    return xr.DataArray(
        data,
        dims=("chain", "draw", "channel", "country", "x"),
        coords={
            "chain": np.arange(2),
            "draw": np.arange(30),
            "channel": channels,
            "country": countries,
            "x": x_values,
        },
    )


ORIGINAL_SCALE_FACTOR = 1000.0


@pytest.fixture(scope="module")
def simple_curve() -> xr.DataArray:
    """Saturation curve in original scale (y >> threshold)."""
    return _make_simple_curve(
        np.random.default_rng(SEED + 10), scale=ORIGINAL_SCALE_FACTOR
    )


@pytest.fixture(scope="module")
def simple_curve_scaled() -> xr.DataArray:
    """Saturation curve in model-internal scaled space (y < threshold)."""
    return _make_simple_curve(np.random.default_rng(SEED + 10))


@pytest.fixture(scope="module")
def panel_curve() -> xr.DataArray:
    """Panel saturation curve in original scale (y >> threshold)."""
    return _make_panel_curve(
        np.random.default_rng(SEED + 11), scale=ORIGINAL_SCALE_FACTOR
    )


@pytest.fixture(scope="module")
def panel_curve_scaled() -> xr.DataArray:
    """Panel saturation curve in model-internal scaled space (y < threshold)."""
    return _make_panel_curve(np.random.default_rng(SEED + 11))


# ============================================================================
# saturation_curves tests
# ============================================================================


class TestSaturationCurvesBasic:
    def test_returns_figure_and_axes(self, simple_plots, simple_curve):
        fig, axes = simple_plots.saturation_curves(curves=simple_curve)
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(a, Axes) for a in axes.flat)

    def test_original_scale_true_is_default(self, simple_plots, simple_curve):
        fig, _axes = simple_plots.saturation_curves(curves=simple_curve)
        assert isinstance(fig, Figure)

    def test_original_scale_false(self, simple_plots, simple_curve_scaled):
        fig, _axes = simple_plots.saturation_curves(
            curves=simple_curve_scaled, original_scale=False
        )
        assert isinstance(fig, Figure)

    def test_axes_count_matches_channels(self, simple_plots, simple_data, simple_curve):
        _, axes = simple_plots.saturation_curves(curves=simple_curve)
        assert axes.size == len(simple_data.channels)


class TestSaturationCurvesHDI:
    def test_hdi_band_drawn(self, simple_plots, simple_curve):
        """HDI band should add a fill_between poly collection to axes."""
        _, axes = simple_plots.saturation_curves(
            curves=simple_curve, hdi_prob=0.94, n_samples=0
        )
        for ax in axes.flat:
            polys = [c for c in ax.collections if "Poly" in type(c).__name__]
            assert len(polys) > 0, "HDI fill_between should be present"

    def test_custom_hdi_prob(self, simple_plots, simple_curve):
        fig, _axes = simple_plots.saturation_curves(curves=simple_curve, hdi_prob=0.50)
        assert isinstance(fig, Figure)


class TestSaturationCurvesSamples:
    def test_sample_curves_drawn(self, simple_plots, simple_curve):
        """Sample curves should add Line2D objects to axes."""
        _, axes = simple_plots.saturation_curves(curves=simple_curve, n_samples=5)
        for ax in axes.flat:
            lines = ax.get_lines()
            assert len(lines) >= 5, "At least n_samples lines per panel"

    def test_n_samples_zero_draws_only_mean_line(self, simple_plots, simple_curve):
        _, axes = simple_plots.saturation_curves(curves=simple_curve, n_samples=0)
        for ax in axes.flat:
            lines = ax.get_lines()
            assert len(lines) == 1, "Only the mean curve line should be present"

    def test_random_seed_reproducible(self, simple_plots, simple_curve):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        _, axes1 = simple_plots.saturation_curves(
            curves=simple_curve, n_samples=3, random_seed=rng1
        )
        _, axes2 = simple_plots.saturation_curves(
            curves=simple_curve, n_samples=3, random_seed=rng2
        )
        for a1, a2 in zip(axes1.flat, axes2.flat, strict=True):
            lines1 = [line.get_ydata() for line in a1.get_lines()]
            lines2 = [line.get_ydata() for line in a2.get_lines()]
            for l1, l2 in zip(lines1, lines2, strict=True):
                np.testing.assert_array_equal(l1, l2)


class TestSaturationCurvesDims:
    def test_single_dim_value(self, panel_plots, panel_curve):
        _, axes = panel_plots.saturation_curves(
            curves=panel_curve, dims={"country": "US"}, n_samples=2
        )
        assert axes.size == 2  # 2 channels x 1 country

    def test_list_dim_value(self, panel_plots, panel_curve):
        _, axes = panel_plots.saturation_curves(
            curves=panel_curve, dims={"country": ["US", "UK"]}, n_samples=2
        )
        assert axes.size == 4  # 2 channels x 2 countries


class TestSaturationCurvesCustomization:
    def test_return_as_pc_true(self, simple_plots, simple_curve):
        result = simple_plots.saturation_curves(curves=simple_curve, return_as_pc=True)
        assert isinstance(result, PlotCollection)

    def test_custom_figsize(self, simple_plots, simple_curve):
        fig, _axes = simple_plots.saturation_curves(
            curves=simple_curve, figsize=(20, 10)
        )
        w, h = fig.get_size_inches()
        assert w == pytest.approx(20, abs=1)
        assert h == pytest.approx(10, abs=1)


class TestSaturationCurvesIdataOverride:
    def test_idata_override(self, simple_plots, panel_idata, panel_curve):
        _, axes = simple_plots.saturation_curves(
            curves=panel_curve, idata=panel_idata, n_samples=2
        )
        assert axes.size == 4  # panel_idata has 2 channels x 2 countries


class TestSaturationCurvesLabels:
    def test_ylabel_is_contributions(self, simple_plots, simple_curve):
        _, axes = simple_plots.saturation_curves(curves=simple_curve)
        for ax in axes.flat:
            assert "Contribution" in ax.get_ylabel()

    def test_xlabel_without_cost_per_unit(self, simple_plots, simple_curve):
        _, axes = simple_plots.saturation_curves(
            curves=simple_curve, apply_cost_per_unit=False
        )
        for ax in axes.flat:
            assert "Channel Data" in ax.get_xlabel()


class TestSaturationCurvesScaleWarning:
    """Heuristic warnings when curve magnitude doesn't match original_scale."""

    def test_warns_original_scale_true_with_scaled_curves(
        self, simple_plots, simple_curve_scaled
    ):
        assert float(simple_curve_scaled.max()) < _SCALED_SPACE_MAX_THRESHOLD
        with pytest.warns(UserWarning, match="original_scale=True"):
            simple_plots.saturation_curves(
                curves=simple_curve_scaled, original_scale=True
            )

    def test_warns_original_scale_false_with_original_scale_curves(
        self, simple_plots, simple_curve
    ):
        assert float(simple_curve.max()) >= _SCALED_SPACE_MAX_THRESHOLD
        with pytest.warns(UserWarning, match="original_scale=False"):
            simple_plots.saturation_curves(curves=simple_curve, original_scale=False)

    def test_no_warning_when_scales_match_original(self, simple_plots, simple_curve):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            simple_plots.saturation_curves(curves=simple_curve, original_scale=True)

    def test_no_warning_when_scales_match_scaled(
        self, simple_plots, simple_curve_scaled
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            simple_plots.saturation_curves(
                curves=simple_curve_scaled, original_scale=False
            )


# ============================================================================
# Helpers & fixtures for sample-dimensioned curves
# (as returned by mmm.sample_saturation_curve())
# ============================================================================


def _make_simple_sample_curve(rng, scale: float = 1.0) -> xr.DataArray:
    """Build a curve with a flat ``sample`` dim (like mmm.sample_saturation_curve)."""
    x_values = np.linspace(0, 1, 50)
    channels = ["tv", "radio", "social"]
    n_samples = 100

    data = np.empty((n_samples, 3, 50))
    for si in range(n_samples):
        for c in range(3):
            data[si, c, :] = x_values / (1 + x_values) + rng.normal(0, 0.01, size=50)
    data *= scale

    return xr.DataArray(
        data,
        dims=("sample", "channel", "x"),
        coords={
            "sample": np.arange(n_samples),
            "channel": channels,
            "x": x_values,
        },
    )


def _make_panel_sample_curve(rng, scale: float = 1.0) -> xr.DataArray:
    """Build a panel curve with a flat ``sample`` dim."""
    x_values = np.linspace(0, 1, 40)
    channels = ["tv", "radio"]
    countries = ["US", "UK"]
    n_samples = 60

    data = np.empty((n_samples, 2, 2, 40))
    for si in range(n_samples):
        for ch in range(2):
            for co in range(2):
                data[si, ch, co, :] = x_values / (1 + x_values) + rng.normal(
                    0, 0.01, size=40
                )
    data *= scale

    return xr.DataArray(
        data,
        dims=("sample", "channel", "country", "x"),
        coords={
            "sample": np.arange(n_samples),
            "channel": channels,
            "country": countries,
            "x": x_values,
        },
    )


@pytest.fixture(scope="module")
def simple_sample_curve() -> xr.DataArray:
    return _make_simple_sample_curve(
        np.random.default_rng(SEED + 20), scale=ORIGINAL_SCALE_FACTOR
    )


@pytest.fixture(scope="module")
def simple_sample_curve_scaled() -> xr.DataArray:
    return _make_simple_sample_curve(np.random.default_rng(SEED + 20))


@pytest.fixture(scope="module")
def panel_sample_curve() -> xr.DataArray:
    return _make_panel_sample_curve(
        np.random.default_rng(SEED + 21), scale=ORIGINAL_SCALE_FACTOR
    )


# ============================================================================
# _normalize_curve_dims tests
# ============================================================================


class TestEnsureChainDrawDims:
    def test_chain_draw_input_unchanged(self, simple_curve):
        result = _ensure_chain_draw_dims(simple_curve)
        assert "chain" in result.dims
        assert "draw" in result.dims
        assert "sample" not in result.dims

    def test_sample_input_converted(self, simple_sample_curve):
        result = _ensure_chain_draw_dims(simple_sample_curve)
        assert "chain" in result.dims
        assert "draw" in result.dims
        assert "sample" not in result.dims
        assert result.sizes["draw"] == simple_sample_curve.sizes["sample"]

    def test_sample_values_preserved(self, simple_sample_curve):
        result = _ensure_chain_draw_dims(simple_sample_curve)
        original_mean = float(simple_sample_curve.mean())
        converted_mean = float(result.mean())
        assert original_mean == pytest.approx(converted_mean, rel=1e-10)

    def test_multiindex_sample_unstacked(self):
        """Test curves with sample as MultiIndex over (chain, draw)."""
        chains = [0, 0, 1, 1]
        draws = [0, 1, 0, 1]
        x_values = np.linspace(0, 1, 10)
        channels = ["tv", "radio"]

        data = np.random.randn(4, 2, 10)
        da = xr.DataArray(
            data,
            dims=("sample", "channel", "x"),
            coords={
                "sample": np.arange(4),
                "chain": ("sample", chains),
                "draw": ("sample", draws),
                "channel": channels,
                "x": x_values,
            },
        )
        # Create MultiIndex by setting index
        da = da.set_index(sample=["chain", "draw"])

        result = _ensure_chain_draw_dims(da)
        assert "chain" in result.dims
        assert "draw" in result.dims
        assert "sample" not in result.dims
        assert result.sizes["chain"] == 2
        assert result.sizes["draw"] == 2

    def test_raises_on_unknown_dims(self):
        bad = xr.DataArray(
            np.zeros((3, 4)),
            dims=("foo", "bar"),
        )
        with pytest.raises(ValueError, match=r"'chain', 'draw'.*or 'sample'"):
            _ensure_chain_draw_dims(bad)


# ============================================================================
# saturation_curves with sample-dimensioned curves
# ============================================================================


class TestSaturationCurvesWithSampleDim:
    """Tests for curves produced by mmm.sample_saturation_curve()."""

    def test_returns_figure_and_axes(self, simple_plots, simple_sample_curve):
        fig, axes = simple_plots.saturation_curves(curves=simple_sample_curve)
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert all(isinstance(a, Axes) for a in axes.flat)

    def test_axes_count_matches_channels(
        self, simple_plots, simple_data, simple_sample_curve
    ):
        _, axes = simple_plots.saturation_curves(curves=simple_sample_curve)
        assert axes.size == len(simple_data.channels)

    def test_hdi_band_drawn(self, simple_plots, simple_sample_curve):
        _, axes = simple_plots.saturation_curves(
            curves=simple_sample_curve, hdi_prob=0.94, n_samples=0
        )
        for ax in axes.flat:
            polys = [c for c in ax.collections if "Poly" in type(c).__name__]
            assert len(polys) > 0, "HDI fill_between should be present"

    def test_sample_curves_drawn(self, simple_plots, simple_sample_curve):
        _, axes = simple_plots.saturation_curves(
            curves=simple_sample_curve, n_samples=5
        )
        for ax in axes.flat:
            lines = ax.get_lines()
            assert len(lines) >= 5

    def test_n_samples_zero_draws_only_mean_line(
        self, simple_plots, simple_sample_curve
    ):
        _, axes = simple_plots.saturation_curves(
            curves=simple_sample_curve, n_samples=0
        )
        for ax in axes.flat:
            lines = ax.get_lines()
            assert len(lines) == 1

    def test_panel_with_sample_dim(self, panel_plots, panel_sample_curve):
        _, axes = panel_plots.saturation_curves(curves=panel_sample_curve, n_samples=2)
        assert axes.size == 4  # 2 channels x 2 countries

    def test_dims_filtering_with_sample_dim(self, panel_plots, panel_sample_curve):
        _, axes = panel_plots.saturation_curves(
            curves=panel_sample_curve, dims={"country": "US"}, n_samples=2
        )
        assert axes.size == 2  # 2 channels x 1 country

    def test_original_scale_false(self, simple_plots, simple_sample_curve_scaled):
        fig, _axes = simple_plots.saturation_curves(
            curves=simple_sample_curve_scaled, original_scale=False
        )
        assert isinstance(fig, Figure)

    def test_return_as_pc_true(self, simple_plots, simple_sample_curve):
        result = simple_plots.saturation_curves(
            curves=simple_sample_curve, return_as_pc=True
        )
        assert isinstance(result, PlotCollection)
