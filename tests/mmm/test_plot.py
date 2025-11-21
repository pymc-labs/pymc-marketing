#   Copyright 2022 - 2025 The PyMC Labs Developers
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
#
"""Tests for new MMMPlotSuite with multi-backend support (arviz_plots-based).

This file tests the new arviz_plots-based MMMPlotSuite that supports
matplotlib, plotly, and bokeh backends.

For tests of the legacy matplotlib-only suite, see test_legacy_plot.py.

Test Organization:
- Parametrized backend tests: Each plotting method tested with all backends
- Backend behavior tests: Config override, invalid backends
- Data parameter tests: Explicit data parameter functionality
- Integration tests: Multiple plots, backend switching

.. versionadded:: 0.18.0
   New test suite for arviz_plots-based MMMPlotSuite.
"""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.plot import MMMPlotSuite


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def mock_idata() -> az.InferenceData:
    seed = sum(map(ord, "Fake posterior"))
    rng = np.random.default_rng(seed)
    normal = rng.normal

    dates = pd.date_range("2025-01-01", periods=52, freq="W-MON")
    return az.InferenceData(
        posterior=xr.Dataset(
            {
                "intercept": xr.DataArray(
                    normal(size=(4, 100, 52, 3)),
                    dims=("chain", "draw", "date", "country"),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(100),
                        "date": dates,
                        "country": ["A", "B", "C"],
                    },
                ),
                "linear_trend": xr.DataArray(
                    normal(size=(4, 100, 52, 3)),
                    dims=("chain", "draw", "date", "country"),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(100),
                        "date": dates,
                        "country": ["A", "B", "C"],
                    },
                ),
            }
        )
    )


@pytest.fixture(scope="module")
def mock_idata_with_sensitivity(mock_idata):
    # Copy the mock_idata so we don't mutate the shared fixture
    idata = mock_idata.copy()
    n_sample, n_sweep = 40, 5
    sweep = np.linspace(0.5, 1.5, n_sweep)
    regions = ["A", "B"]

    samples = xr.DataArray(
        np.random.normal(0, 1, size=(n_sample, n_sweep, len(regions))),
        dims=("sample", "sweep", "region"),
        coords={
            "sample": np.arange(n_sample),
            "sweep": sweep,
            "region": regions,
        },
        name="x",
    )

    marginal_effects = xr.DataArray(
        np.random.normal(0, 1, size=(n_sample, n_sweep, len(regions))),
        dims=("sample", "sweep", "region"),
        coords={
            "sample": np.arange(n_sample),
            "sweep": sweep,
            "region": regions,
        },
        name="marginal_effects",
    )

    uplift_curve = xr.DataArray(
        np.random.normal(0, 1, size=(n_sample, n_sweep, len(regions))),
        dims=("sample", "sweep", "region"),
        coords={
            "sample": np.arange(n_sample),
            "sweep": sweep,
            "region": regions,
        },
        name="uplift_curve",
    )

    sensitivity_analysis = xr.Dataset(
        {
            "x": samples,
            "marginal_effects": marginal_effects,
            "uplift_curve": uplift_curve,
        },
        coords={"sweep": sweep, "region": regions},
        attrs={"sweep_type": "multiplicative", "var_names": "test_var"},
    )

    idata.sensitivity_analysis = sensitivity_analysis
    return idata


@pytest.fixture(scope="module")
def mock_suite(mock_idata):
    """Fixture to create a mock MMMPlotSuite with a mocked posterior."""
    return MMMPlotSuite(idata=mock_idata)


@pytest.fixture(scope="module")
def mock_suite_with_sensitivity(mock_idata_with_sensitivity):
    """Fixture to create a mock MMMPlotSuite with sensitivity analysis."""
    return MMMPlotSuite(idata=mock_idata_with_sensitivity)


@pytest.fixture(scope="module")
def mock_idata_with_constant_data() -> az.InferenceData:
    """Create mock InferenceData with constant_data and posterior for saturation tests."""
    seed = sum(map(ord, "Saturation tests"))
    rng = np.random.default_rng(seed)
    normal = rng.normal

    dates = pd.date_range("2025-01-01", periods=52, freq="W-MON")
    channels = ["channel_1", "channel_2"]
    countries = ["A", "B"]

    # Create posterior data
    posterior = xr.Dataset(
        {
            "channel_contribution": xr.DataArray(
                normal(size=(4, 100, 52, 2, 2)),
                dims=("chain", "draw", "date", "channel", "country"),
                coords={
                    "chain": np.arange(4),
                    "draw": np.arange(100),
                    "date": dates,
                    "channel": channels,
                    "country": countries,
                },
            ),
            "channel_contribution_original_scale": xr.DataArray(
                normal(size=(4, 100, 52, 2, 2)) * 100,  # scaled up for original scale
                dims=("chain", "draw", "date", "channel", "country"),
                coords={
                    "chain": np.arange(4),
                    "draw": np.arange(100),
                    "date": dates,
                    "channel": channels,
                    "country": countries,
                },
            ),
        }
    )

    # Create constant_data
    constant_data = xr.Dataset(
        {
            "channel_data": xr.DataArray(
                rng.uniform(0, 10, size=(52, 2, 2)),
                dims=("date", "channel", "country"),
                coords={
                    "date": dates,
                    "channel": channels,
                    "country": countries,
                },
            ),
            "channel_scale": xr.DataArray(
                [[100.0, 200.0], [150.0, 250.0]],
                dims=("country", "channel"),
                coords={"country": countries, "channel": channels},
            ),
            "target_scale": xr.DataArray(
                [1000.0],
                dims="target",
                coords={"target": ["y"]},
            ),
        }
    )

    return az.InferenceData(posterior=posterior, constant_data=constant_data)


@pytest.fixture(scope="module")
def mock_suite_with_constant_data(mock_idata_with_constant_data):
    """Fixture to create a MMMPlotSuite with constant_data for saturation tests."""
    return MMMPlotSuite(idata=mock_idata_with_constant_data)


@pytest.fixture(scope="module")
def mock_saturation_curve() -> xr.DataArray:
    """Create mock saturation curve data for testing saturation_curves method."""
    seed = sum(map(ord, "Saturation curve"))
    rng = np.random.default_rng(seed)

    # Create curve data with typical saturation curve shape
    x_values = np.linspace(0, 1, 100)
    channels = ["channel_1", "channel_2"]
    countries = ["A", "B"]

    curve_data = []
    for _ in range(4):  # chains
        for _ in range(100):  # draws
            for _ in channels:
                for _ in countries:
                    # Simple saturation curve: y = x / (1 + x)
                    y_values = x_values / (1 + x_values) + rng.normal(
                        0, 0.01, size=x_values.shape
                    )
                    curve_data.append(y_values)

    curve_array = np.array(curve_data).reshape(
        4, 100, len(channels), len(countries), len(x_values)
    )

    return xr.DataArray(
        curve_array,
        dims=("chain", "draw", "channel", "country", "x"),
        coords={
            "chain": np.arange(4),
            "draw": np.arange(100),
            "channel": channels,
            "country": countries,
            "x": x_values,
        },
    )


# =============================================================================
# Basic Functionality Tests
# =============================================================================


def test_contributions_over_time_expand_dims(mock_suite: MMMPlotSuite):
    from arviz_plots import PlotCollection

    pc = mock_suite.contributions_over_time(
        var=[
            "intercept",
            "linear_trend",
        ]
    )

    assert isinstance(pc, PlotCollection)
    assert hasattr(pc, "backend")
    assert hasattr(pc, "show")


# =============================================================================
# Comprehensive Backend Tests (Milestone 3)
# =============================================================================
# These tests verify that all plotting methods work correctly across all
# supported backends (matplotlib, plotly, bokeh).
# =============================================================================


class TestPosteriorPredictiveBackends:
    """Test posterior_predictive method across all backends."""

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_posterior_predictive_all_backends(self, mock_suite, backend):
        """Test posterior_predictive works with all backends."""
        from arviz_plots import PlotCollection

        # Create idata with posterior_predictive
        idata = mock_suite.idata.copy()
        rng = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=52, freq="W")
        idata.posterior_predictive = xr.Dataset(
            {
                "y": xr.DataArray(
                    rng.normal(size=(4, 100, 52)),
                    dims=("chain", "draw", "date"),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(100),
                        "date": dates,
                    },
                )
            }
        )
        suite = MMMPlotSuite(idata=idata)

        pc = suite.posterior_predictive(backend=backend)

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )


class TestContributionsOverTimeBackends:
    """Test contributions_over_time method across all backends."""

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_contributions_over_time_all_backends(self, mock_suite, backend):
        """Test contributions_over_time works with all backends."""
        from arviz_plots import PlotCollection

        pc = mock_suite.contributions_over_time(var=["intercept"], backend=backend)

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )


class TestSaturationPlotBackends:
    """Test saturation plot methods across all backends."""

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_saturation_scatterplot_all_backends(
        self, mock_suite_with_constant_data, backend
    ):
        """Test saturation_scatterplot works with all backends."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_constant_data.saturation_scatterplot(backend=backend)

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_saturation_curves_all_backends(
        self, mock_suite_with_constant_data, mock_saturation_curve, backend
    ):
        """Test saturation_curves works with all backends."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve, backend=backend, n_samples=3
        )

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )


class TestBudgetAllocationBackends:
    """Test budget allocation methods across all backends."""

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_budget_allocation_roas_all_backends(self, mock_suite, backend):
        """Test budget_allocation_roas works with all backends."""
        from arviz_plots import PlotCollection

        # Create proper allocation samples with required variables and dimensions
        rng = np.random.default_rng(42)
        channels = ["TV", "Radio", "Digital"]
        dates = pd.date_range("2025-01-01", periods=52, freq="W")
        samples = xr.Dataset(
            {
                "channel_contribution_original_scale": xr.DataArray(
                    rng.normal(loc=1000, scale=100, size=(100, 52, 3)),
                    dims=("sample", "date", "channel"),
                    coords={
                        "sample": np.arange(100),
                        "date": dates,
                        "channel": channels,
                    },
                ),
                "allocation": xr.DataArray(
                    rng.uniform(100, 1000, size=(3,)),
                    dims=("channel",),
                    coords={"channel": channels},
                ),
            }
        )

        pc = mock_suite.budget_allocation_roas(samples=samples, backend=backend)

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_allocated_contribution_by_channel_over_time_all_backends(
        self, mock_suite, backend
    ):
        """Test allocated_contribution_by_channel_over_time works with all backends."""
        from arviz_plots import PlotCollection

        # Create proper samples with 'sample', 'date', and 'channel' dimensions
        rng = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=52, freq="W")
        channels = ["TV", "Radio", "Digital"]
        samples = xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(size=(100, 52, 3)),
                    dims=("sample", "date", "channel"),
                    coords={
                        "sample": np.arange(100),
                        "date": dates,
                        "channel": channels,
                    },
                )
            }
        )

        pc = mock_suite.allocated_contribution_by_channel_over_time(
            samples=samples, backend=backend
        )

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )


class TestSensitivityAnalysisBackends:
    """Test sensitivity analysis methods across all backends."""

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_sensitivity_analysis_all_backends(
        self, mock_suite_with_sensitivity, backend
    ):
        """Test sensitivity_analysis works with all backends."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_sensitivity.sensitivity_analysis(backend=backend)

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_uplift_curve_all_backends(self, mock_suite_with_sensitivity, backend):
        """Test uplift_curve works with all backends."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_sensitivity.uplift_curve(backend=backend)

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
    def test_marginal_curve_all_backends(self, mock_suite_with_sensitivity, backend):
        """Test marginal_curve works with all backends."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_sensitivity.marginal_curve(backend=backend)

        assert isinstance(pc, PlotCollection), (
            f"Expected PlotCollection for backend {backend}, got {type(pc)}"
        )


class TestBackendBehavior:
    """Test backend configuration and override behavior."""

    def test_backend_overrides_global_config(self, mock_suite):
        """Test that method backend parameter overrides global config."""
        from arviz_plots import PlotCollection

        from pymc_marketing.mmm import mmm_config

        original = mmm_config.get("plot.backend", "matplotlib")

        try:
            # Set global to matplotlib
            mmm_config["plot.backend"] = "matplotlib"

            # Override with plotly
            pc_plotly = mock_suite.contributions_over_time(
                var=["intercept"], backend="plotly"
            )
            assert isinstance(pc_plotly, PlotCollection)

            # Default should still be matplotlib
            pc_default = mock_suite.contributions_over_time(var=["intercept"])
            assert isinstance(pc_default, PlotCollection)

        finally:
            mmm_config["plot.backend"] = original

    @pytest.mark.parametrize("config_backend", ["matplotlib", "plotly", "bokeh"])
    def test_backend_parameter_none_uses_config(self, mock_suite, config_backend):
        """Test that backend=None uses global config."""
        from arviz_plots import PlotCollection

        from pymc_marketing.mmm import mmm_config

        original = mmm_config.get("plot.backend", "matplotlib")

        try:
            mmm_config["plot.backend"] = config_backend

            pc = mock_suite.contributions_over_time(
                var=["intercept"], backend=None  # Explicitly None
            )

            assert isinstance(pc, PlotCollection)

        finally:
            mmm_config["plot.backend"] = original

    def test_invalid_backend_warning(self, mock_suite):
        """Test that invalid backend shows warning."""
        import warnings

        # Invalid backend should warn but still attempt to create plot
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This might fail or succeed depending on arviz_plots behavior
            # The important thing is that a warning was issued
            try:
                _pc = mock_suite.contributions_over_time(
                    var=["intercept"], backend="invalid_backend"
                )
                # If it succeeds, just check warning was issued
                assert any("backend" in str(warning.message).lower() for warning in w)
            except Exception:
                # If it fails, that's also acceptable
                # The warning should have been issued before the error
                assert any("backend" in str(warning.message).lower() for warning in w)


class TestDataParameters:
    """Test explicit data parameter functionality."""

    def test_contributions_over_time_with_explicit_data(self, mock_posterior_data):
        """Test contributions_over_time accepts explicit data parameter."""
        from arviz_plots import PlotCollection

        # Create suite without idata
        suite = MMMPlotSuite(idata=None)

        # Should work with explicit data parameter
        pc = suite.contributions_over_time(var=["intercept"], data=mock_posterior_data)

        assert isinstance(pc, PlotCollection)

    def test_saturation_scatterplot_with_explicit_data(
        self, mock_constant_data, mock_posterior_data
    ):
        """Test saturation_scatterplot accepts explicit data parameters."""
        from arviz_plots import PlotCollection

        suite = MMMPlotSuite(idata=None)

        # Create a small posterior for testing
        posterior_data = xr.Dataset(
            {
                "channel_contribution": mock_posterior_data["intercept"].isel(
                    country=0, drop=True
                )
            }
        )

        pc = suite.saturation_scatterplot(
            constant_data=mock_constant_data, posterior_data=posterior_data
        )

        assert isinstance(pc, PlotCollection)


class TestIntegration:
    """Integration tests for multiple plots and backend switching."""

    def test_multiple_plots_same_suite_instance(self, mock_suite_with_constant_data):
        """Test that same suite instance can create multiple plots."""
        from arviz_plots import PlotCollection

        suite = mock_suite_with_constant_data

        # Create multiple different plots
        pc1 = suite.contributions_over_time(var=["channel_contribution"])
        pc2 = suite.saturation_scatterplot()

        assert isinstance(pc1, PlotCollection)
        assert isinstance(pc2, PlotCollection)

        # All should be independent PlotCollection objects
        assert pc1 is not pc2

    def test_backend_switching_same_method(self, mock_suite):
        """Test that backends can be switched for same method."""
        from arviz_plots import PlotCollection

        suite = mock_suite

        # Create same plot with different backends
        pc_mpl = suite.contributions_over_time(var=["intercept"], backend="matplotlib")
        pc_plotly = suite.contributions_over_time(var=["intercept"], backend="plotly")
        pc_bokeh = suite.contributions_over_time(var=["intercept"], backend="bokeh")

        assert isinstance(pc_mpl, PlotCollection)
        assert isinstance(pc_plotly, PlotCollection)
        assert isinstance(pc_bokeh, PlotCollection)
