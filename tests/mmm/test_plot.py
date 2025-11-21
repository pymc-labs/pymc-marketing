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
                var=["intercept"],
                backend=None,  # Explicitly None
            )

            assert isinstance(pc, PlotCollection)

        finally:
            mmm_config["plot.backend"] = original

    def test_invalid_backend_raises_error(self, mock_suite):
        """Test that invalid backend raises an appropriate error."""
        # Invalid backend should raise an error (arviz_plots behavior)
        with pytest.raises((ModuleNotFoundError, ImportError, ValueError)):
            _pc = mock_suite.contributions_over_time(
                var=["intercept"], backend="invalid_backend"
            )


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

        # Create posterior data with channel_contribution matching constant_data channels
        rng = np.random.default_rng(42)
        n_channels = len(mock_constant_data.coords["channel"])
        posterior_data = xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    rng.normal(size=(4, 100, 52, n_channels)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(100),
                        "date": mock_constant_data.coords["date"],
                        "channel": mock_constant_data.coords["channel"],
                    },
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


# =============================================================================
# Validation Error Tests
# =============================================================================


class TestValidationErrors:
    """Test validation and error handling."""

    def test_posterior_predictive_invalid_hdi_prob(self, mock_suite):
        """Test that invalid hdi_prob raises ValueError."""
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

        with pytest.raises(ValueError, match="HDI probability must be between 0 and 1"):
            suite.posterior_predictive(hdi_prob=1.5)

        with pytest.raises(ValueError, match="HDI probability must be between 0 and 1"):
            suite.posterior_predictive(hdi_prob=0.0)

    def test_contributions_over_time_invalid_hdi_prob(self, mock_suite):
        """Test that invalid hdi_prob raises ValueError."""
        with pytest.raises(ValueError, match="HDI probability must be between 0 and 1"):
            mock_suite.contributions_over_time(var=["intercept"], hdi_prob=2.0)

    def test_contributions_over_time_missing_variable(self, mock_suite):
        """Test that missing variable raises ValueError."""
        with pytest.raises(ValueError, match="not found in data"):
            mock_suite.contributions_over_time(var=["nonexistent_var"])

    def test_posterior_predictive_no_data(self):
        """Test that missing posterior_predictive data raises ValueError."""
        suite = MMMPlotSuite(idata=None)

        with pytest.raises(ValueError, match="No posterior_predictive data found"):
            suite.posterior_predictive()

    def test_contributions_over_time_no_posterior(self):
        """Test that missing posterior data raises ValueError."""
        suite = MMMPlotSuite(idata=None)

        with pytest.raises(ValueError, match="No posterior data found"):
            suite.contributions_over_time(var=["intercept"])

    def test_saturation_scatterplot_no_constant_data(self):
        """Test that missing constant_data raises ValueError."""
        suite = MMMPlotSuite(idata=None)

        with pytest.raises(ValueError, match="No constant data found"):
            suite.saturation_scatterplot()

    def test_saturation_scatterplot_missing_channel_data(self, mock_posterior_data):
        """Test that missing channel_data variable raises ValueError."""
        suite = MMMPlotSuite(idata=None)

        # Create constant_data without channel_data
        constant_data = xr.Dataset({"other_var": xr.DataArray([1, 2, 3])})

        with pytest.raises(ValueError, match="'channel_data' variable not found"):
            suite.saturation_scatterplot(
                constant_data=constant_data, posterior_data=mock_posterior_data
            )

    def test_saturation_scatterplot_missing_channel_contribution(
        self, mock_constant_data
    ):
        """Test that missing channel_contribution raises ValueError."""
        suite = MMMPlotSuite(idata=None)

        # Create posterior without channel_contribution
        posterior = xr.Dataset({"other_var": xr.DataArray([1, 2, 3])})

        with pytest.raises(ValueError, match=r"No posterior\.channel_contribution"):
            suite.saturation_scatterplot(
                constant_data=mock_constant_data, posterior_data=posterior
            )

    def test_saturation_curves_missing_x_dimension(self, mock_suite_with_constant_data):
        """Test that curve without 'x' dimension raises ValueError."""
        # Create curve without 'x' dimension
        bad_curve = xr.DataArray(
            np.random.rand(10, 2),
            dims=("time", "channel"),
            coords={"time": np.arange(10), "channel": ["A", "B"]},
        )

        with pytest.raises(ValueError, match="curve must have an 'x' dimension"):
            mock_suite_with_constant_data.saturation_curves(curve=bad_curve)

    def test_saturation_curves_missing_channel_dimension(
        self, mock_suite_with_constant_data
    ):
        """Test that curve without 'channel' dimension raises ValueError."""
        # Create curve without 'channel' dimension
        bad_curve = xr.DataArray(
            np.random.rand(10, 20),
            dims=("time", "x"),
            coords={"time": np.arange(10), "x": np.linspace(0, 1, 20)},
        )

        with pytest.raises(ValueError, match="curve must have a 'channel' dimension"):
            mock_suite_with_constant_data.saturation_curves(curve=bad_curve)

    def test_budget_allocation_roas_missing_channel_dim(self, mock_suite):
        """Test that samples without channel dimension raises ValueError."""
        # Create samples without channel dimension
        samples = xr.Dataset({"some_var": xr.DataArray([1, 2, 3])})

        with pytest.raises(ValueError, match="Expected 'channel' dimension"):
            mock_suite.budget_allocation_roas(samples=samples)

    def test_budget_allocation_roas_missing_contribution(self, mock_suite):
        """Test that samples without contribution variable raises ValueError."""
        # Create samples with channel but missing contribution
        samples = xr.Dataset(
            {
                "other_var": xr.DataArray(
                    [1, 2, 3], dims=("channel",), coords={"channel": ["A", "B", "C"]}
                )
            }
        )

        with pytest.raises(
            ValueError,
            match="Expected a variable containing 'channel_contribution_original_scale'",
        ):
            mock_suite.budget_allocation_roas(samples=samples)

    def test_budget_allocation_roas_missing_allocation(self, mock_suite):
        """Test that samples without allocation raises ValueError."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=52, freq="W")
        channels = ["A", "B", "C"]

        # Create samples with contribution but missing allocation
        samples = xr.Dataset(
            {
                "channel_contribution_original_scale": xr.DataArray(
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

        with pytest.raises(ValueError, match="Expected 'allocation' variable"):
            mock_suite.budget_allocation_roas(samples=samples)

    def test_allocated_contribution_missing_channel(self, mock_suite):
        """Test that samples without channel dimension raises ValueError."""
        samples = xr.Dataset({"some_var": xr.DataArray([1, 2, 3])})

        with pytest.raises(ValueError, match="Expected 'channel' dimension"):
            mock_suite.allocated_contribution_by_channel_over_time(samples=samples)

    def test_allocated_contribution_missing_date(self, mock_suite):
        """Test that samples without date dimension raises ValueError."""
        samples = xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    [[1, 2], [3, 4]],
                    dims=("sample", "channel"),
                    coords={"sample": [0, 1], "channel": ["A", "B"]},
                )
            }
        )

        with pytest.raises(ValueError, match="Expected 'date' dimension"):
            mock_suite.allocated_contribution_by_channel_over_time(samples=samples)

    def test_allocated_contribution_missing_sample(self, mock_suite):
        """Test that samples without sample dimension raises ValueError."""
        dates = pd.date_range("2025-01-01", periods=10, freq="W")
        samples = xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    [[1, 2], [3, 4]],
                    dims=("date", "channel"),
                    coords={"date": dates[:2], "channel": ["A", "B"]},
                )
            }
        )

        with pytest.raises(ValueError, match="Expected 'sample' dimension"):
            mock_suite.allocated_contribution_by_channel_over_time(samples=samples)

    def test_allocated_contribution_missing_contribution_var(self, mock_suite):
        """Test that samples without channel_contribution variable raises ValueError."""
        dates = pd.date_range("2025-01-01", periods=10, freq="W")
        samples = xr.Dataset(
            {
                "other_var": xr.DataArray(
                    [[[1, 2]]],
                    dims=("sample", "date", "channel"),
                    coords={"sample": [0], "date": dates[:1], "channel": ["A", "B"]},
                )
            }
        )

        with pytest.raises(
            ValueError, match="Expected a variable containing 'channel_contribution'"
        ):
            mock_suite.allocated_contribution_by_channel_over_time(samples=samples)

    def test_sensitivity_analysis_invalid_dimensions(self, mock_suite):
        """Test that data without required dimensions raises ValueError."""
        # Create data without required dimensions
        bad_data = xr.DataArray(
            np.random.rand(10, 20), dims=("time", "space"), name="x"
        )

        with pytest.raises(ValueError, match="Data must have dimensions"):
            mock_suite._sensitivity_analysis_plot(data=bad_data)

    def test_sensitivity_analysis_no_data(self):
        """Test that missing sensitivity_analysis group raises ValueError."""
        suite = MMMPlotSuite(idata=None)

        with pytest.raises(ValueError, match="No sensitivity analysis results found"):
            suite.sensitivity_analysis()

    def test_uplift_curve_missing_data(self):
        """Test that missing uplift_curve raises ValueError."""
        # Create idata with sensitivity_analysis but without uplift_curve
        rng = np.random.default_rng(42)
        idata = az.InferenceData(
            posterior=xr.Dataset(
                {"intercept": xr.DataArray(rng.normal(size=(4, 100)))}
            ),
            sensitivity_analysis=xr.Dataset(
                {
                    "x": xr.DataArray(
                        rng.normal(size=(100, 20)),
                        dims=("sample", "sweep"),
                        coords={
                            "sample": np.arange(100),
                            "sweep": np.linspace(0, 1, 20),
                        },
                    )
                }
            ),
        )
        suite = MMMPlotSuite(idata=idata)

        with pytest.raises(ValueError, match="Expected 'uplift_curve'"):
            suite.uplift_curve()

    def test_marginal_curve_missing_data(self):
        """Test that missing marginal_effects raises ValueError."""
        # Create idata with sensitivity_analysis but without marginal_effects
        rng = np.random.default_rng(42)
        idata = az.InferenceData(
            posterior=xr.Dataset(
                {"intercept": xr.DataArray(rng.normal(size=(4, 100)))}
            ),
            sensitivity_analysis=xr.Dataset(
                {
                    "x": xr.DataArray(
                        rng.normal(size=(100, 20)),
                        dims=("sample", "sweep"),
                        coords={
                            "sample": np.arange(100),
                            "sweep": np.linspace(0, 1, 20),
                        },
                    )
                }
            ),
        )
        suite = MMMPlotSuite(idata=idata)

        with pytest.raises(ValueError, match="Expected 'marginal_effects'"):
            suite.marginal_curve()

    def test_get_additional_dim_combinations_missing_variable(self, mock_suite):
        """Test that missing variable in dataset raises ValueError."""
        with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
            mock_suite._get_additional_dim_combinations(
                data=mock_suite.idata.posterior,
                variable="nonexistent",
                ignored_dims={"chain", "draw"},
            )

    def test_validate_dims_invalid_dimension(self, mock_suite):
        """Test that invalid dimension raises ValueError."""
        with pytest.raises(ValueError, match="Dimension 'invalid_dim' not found"):
            mock_suite._validate_dims(
                dims={"invalid_dim": "A"}, all_dims=["chain", "draw", "country"]
            )

    def test_validate_dims_invalid_value(self, mock_suite):
        """Test that invalid dimension value raises ValueError."""
        with pytest.raises(ValueError, match="Value 'Z' not found in dimension"):
            mock_suite._validate_dims(
                dims={"country": "Z"}, all_dims=["chain", "draw", "country"]
            )

    def test_validate_dims_invalid_list_value(self, mock_suite):
        """Test that invalid value in list raises ValueError."""
        with pytest.raises(ValueError, match="Value 'Z' not found in dimension"):
            mock_suite._validate_dims(
                dims={"country": ["A", "Z"]}, all_dims=["chain", "draw", "country"]
            )


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_contributions_over_time_with_dims_filtering(self, mock_suite):
        """Test contributions_over_time with dims parameter."""
        from arviz_plots import PlotCollection

        # Filter to specific country
        pc = mock_suite.contributions_over_time(
            var=["intercept"], dims={"country": "A"}
        )
        assert isinstance(pc, PlotCollection)

    def test_contributions_over_time_with_list_dims(self, mock_suite):
        """Test contributions_over_time with list-valued dims."""
        from arviz_plots import PlotCollection

        # Filter to multiple countries
        pc = mock_suite.contributions_over_time(
            var=["intercept"], dims={"country": ["A", "B"]}
        )
        assert isinstance(pc, PlotCollection)

    def test_saturation_scatterplot_with_dims_single_value(
        self, mock_suite_with_constant_data
    ):
        """Test saturation_scatterplot with single-value dims."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_constant_data.saturation_scatterplot(dims={"country": "A"})
        assert isinstance(pc, PlotCollection)

    def test_saturation_scatterplot_with_dims_list(self, mock_suite_with_constant_data):
        """Test saturation_scatterplot with list-valued dims."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_constant_data.saturation_scatterplot(
            dims={"country": ["A", "B"]}
        )
        assert isinstance(pc, PlotCollection)

    def test_saturation_curves_with_hdi_probs_float(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test saturation_curves with float hdi_probs."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve, hdi_probs=0.9, n_samples=3
        )
        assert isinstance(pc, PlotCollection)

    def test_saturation_curves_with_hdi_probs_list(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test saturation_curves with list of hdi_probs."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve, hdi_probs=[0.5, 0.9], n_samples=3
        )
        assert isinstance(pc, PlotCollection)

    def test_saturation_curves_with_hdi_probs_tuple(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test saturation_curves with tuple of hdi_probs."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve, hdi_probs=(0.5, 0.9), n_samples=3
        )
        assert isinstance(pc, PlotCollection)

    def test_saturation_curves_with_hdi_probs_array(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test saturation_curves with numpy array of hdi_probs."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_constant_data.saturation_curves(
            curve=mock_saturation_curve,
            hdi_probs=np.array([0.5, 0.9]),
            n_samples=3,
        )
        assert isinstance(pc, PlotCollection)

    def test_budget_allocation_roas_with_dims_to_group_by_string(self, mock_suite):
        """Test budget_allocation_roas with dims_to_group_by as string."""
        from arviz_plots import PlotCollection

        rng = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=52, freq="W")
        channels = ["TV", "Radio", "Digital"]
        regions = ["East", "West"]

        samples = xr.Dataset(
            {
                "channel_contribution_original_scale": xr.DataArray(
                    rng.normal(loc=1000, scale=100, size=(100, 52, 3, 2)),
                    dims=("sample", "date", "channel", "region"),
                    coords={
                        "sample": np.arange(100),
                        "date": dates,
                        "channel": channels,
                        "region": regions,
                    },
                ),
                "allocation": xr.DataArray(
                    rng.uniform(100, 1000, size=(3, 2)),
                    dims=("channel", "region"),
                    coords={"channel": channels, "region": regions},
                ),
            }
        )

        pc = mock_suite.budget_allocation_roas(
            samples=samples, dims_to_group_by="region"
        )
        assert isinstance(pc, PlotCollection)

    def test_budget_allocation_roas_with_dims_to_group_by_list(self, mock_suite):
        """Test budget_allocation_roas with dims_to_group_by as list."""
        from arviz_plots import PlotCollection

        rng = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=52, freq="W")
        channels = ["TV", "Radio"]
        regions = ["East", "West"]

        samples = xr.Dataset(
            {
                "channel_contribution_original_scale": xr.DataArray(
                    rng.normal(loc=1000, scale=100, size=(100, 52, 2, 2)),
                    dims=("sample", "date", "channel", "region"),
                    coords={
                        "sample": np.arange(100),
                        "date": dates,
                        "channel": channels,
                        "region": regions,
                    },
                ),
                "allocation": xr.DataArray(
                    rng.uniform(100, 1000, size=(2, 2)),
                    dims=("channel", "region"),
                    coords={"channel": channels, "region": regions},
                ),
            }
        )

        pc = mock_suite.budget_allocation_roas(
            samples=samples, dims_to_group_by=["channel", "region"]
        )
        assert isinstance(pc, PlotCollection)

    def test_sensitivity_analysis_with_aggregation_sum(self, mock_sensitivity_data):
        """Test sensitivity_analysis_plot with sum aggregation."""
        from arviz_plots import PlotCollection

        # Add a dimension to aggregate over
        data_with_dim = xr.Dataset(
            {
                "x": xr.DataArray(
                    np.random.rand(100, 20, 3),
                    dims=("sample", "sweep", "channel"),
                    coords={
                        "sample": np.arange(100),
                        "sweep": np.linspace(0, 1, 20),
                        "channel": ["A", "B", "C"],
                    },
                )
            }
        )

        suite = MMMPlotSuite(idata=None)
        pc = suite._sensitivity_analysis_plot(
            data=data_with_dim, aggregation={"sum": ("channel",)}
        )
        assert isinstance(pc, PlotCollection)

    def test_sensitivity_analysis_with_aggregation_mean(self, mock_sensitivity_data):
        """Test sensitivity_analysis_plot with mean aggregation."""
        from arviz_plots import PlotCollection

        data_with_dim = xr.Dataset(
            {
                "x": xr.DataArray(
                    np.random.rand(100, 20, 3),
                    dims=("sample", "sweep", "channel"),
                    coords={
                        "sample": np.arange(100),
                        "sweep": np.linspace(0, 1, 20),
                        "channel": ["A", "B", "C"],
                    },
                )
            }
        )

        suite = MMMPlotSuite(idata=None)
        pc = suite._sensitivity_analysis_plot(
            data=data_with_dim, aggregation={"mean": ("channel",)}
        )
        assert isinstance(pc, PlotCollection)

    def test_sensitivity_analysis_with_aggregation_median(self, mock_sensitivity_data):
        """Test sensitivity_analysis_plot with median aggregation."""
        from arviz_plots import PlotCollection

        data_with_dim = xr.Dataset(
            {
                "x": xr.DataArray(
                    np.random.rand(100, 20, 3),
                    dims=("sample", "sweep", "channel"),
                    coords={
                        "sample": np.arange(100),
                        "sweep": np.linspace(0, 1, 20),
                        "channel": ["A", "B", "C"],
                    },
                )
            }
        )

        suite = MMMPlotSuite(idata=None)
        pc = suite._sensitivity_analysis_plot(
            data=data_with_dim, aggregation={"median": ("channel",)}
        )
        assert isinstance(pc, PlotCollection)

    def test_uplift_curve_with_dataset_containing_uplift_curve(self):
        """Test uplift_curve when data is Dataset with uplift_curve variable."""
        from arviz_plots import PlotCollection

        rng = np.random.default_rng(42)
        data = xr.Dataset(
            {
                "uplift_curve": xr.DataArray(
                    rng.normal(size=(100, 20)),
                    dims=("sample", "sweep"),
                    coords={
                        "sample": np.arange(100),
                        "sweep": np.linspace(0, 1, 20),
                    },
                )
            }
        )

        suite = MMMPlotSuite(idata=None)
        pc = suite.uplift_curve(data=data)
        assert isinstance(pc, PlotCollection)

    def test_uplift_curve_with_dataset_containing_x(self):
        """Test uplift_curve when data is Dataset with x variable."""
        from arviz_plots import PlotCollection

        rng = np.random.default_rng(42)
        data = xr.Dataset(
            {
                "x": xr.DataArray(
                    rng.normal(size=(100, 20)),
                    dims=("sample", "sweep"),
                    coords={
                        "sample": np.arange(100),
                        "sweep": np.linspace(0, 1, 20),
                    },
                )
            }
        )

        suite = MMMPlotSuite(idata=None)
        pc = suite.uplift_curve(data=data)
        assert isinstance(pc, PlotCollection)

    def test_marginal_curve_with_dataset_containing_marginal_effects(self):
        """Test marginal_curve when data is Dataset with marginal_effects variable."""
        from arviz_plots import PlotCollection

        rng = np.random.default_rng(42)
        data = xr.Dataset(
            {
                "marginal_effects": xr.DataArray(
                    rng.normal(size=(100, 20)),
                    dims=("sample", "sweep"),
                    coords={
                        "sample": np.arange(100),
                        "sweep": np.linspace(0, 1, 20),
                    },
                )
            }
        )

        suite = MMMPlotSuite(idata=None)
        pc = suite.marginal_curve(data=data)
        assert isinstance(pc, PlotCollection)

    def test_marginal_curve_with_dataset_containing_x(self):
        """Test marginal_curve when data is Dataset with x variable."""
        from arviz_plots import PlotCollection

        rng = np.random.default_rng(42)
        data = xr.Dataset(
            {
                "x": xr.DataArray(
                    rng.normal(size=(100, 20)),
                    dims=("sample", "sweep"),
                    coords={
                        "sample": np.arange(100),
                        "sweep": np.linspace(0, 1, 20),
                    },
                )
            }
        )

        suite = MMMPlotSuite(idata=None)
        pc = suite.marginal_curve(data=data)
        assert isinstance(pc, PlotCollection)


# =============================================================================
# Original Scale Tests
# =============================================================================


class TestOriginalScale:
    """Test original_scale parameter functionality."""

    def test_saturation_scatterplot_original_scale_true(
        self, mock_suite_with_constant_data
    ):
        """Test saturation_scatterplot with original_scale=True."""
        from arviz_plots import PlotCollection

        pc = mock_suite_with_constant_data.saturation_scatterplot(original_scale=True)
        assert isinstance(pc, PlotCollection)

    def test_saturation_scatterplot_original_scale_missing_variable(
        self, mock_constant_data
    ):
        """Test that original_scale=True without variable raises ValueError."""
        suite = MMMPlotSuite(idata=None)

        # Create posterior without channel_contribution_original_scale
        posterior = xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    np.random.rand(4, 100, 52, 3),
                    dims=("chain", "draw", "date", "channel"),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(100),
                        "date": mock_constant_data.coords["date"],
                        "channel": mock_constant_data.coords["channel"],
                    },
                )
            }
        )

        with pytest.raises(
            ValueError, match=r"No posterior\.channel_contribution_original_scale"
        ):
            suite.saturation_scatterplot(
                original_scale=True,
                constant_data=mock_constant_data,
                posterior_data=posterior,
            )

    def test_saturation_curves_original_scale_missing_variable(
        self, mock_constant_data, mock_saturation_curve
    ):
        """Test that original_scale=True without variable raises ValueError."""
        suite = MMMPlotSuite(idata=None)

        # Create posterior without channel_contribution_original_scale
        posterior = xr.Dataset(
            {
                "channel_contribution": xr.DataArray(
                    np.random.rand(4, 100, 52, 3),
                    dims=("chain", "draw", "date", "channel"),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(100),
                        "date": mock_constant_data.coords["date"],
                        "channel": mock_constant_data.coords["channel"],
                    },
                )
            }
        )

        with pytest.raises(ValueError, match=r"No posterior\.channel_contribution"):
            suite.saturation_curves(
                curve=mock_saturation_curve,
                original_scale=True,
                constant_data=mock_constant_data,
                posterior_data=posterior,
            )


# =============================================================================
# Deprecated Method Tests
# =============================================================================


class TestDeprecatedMethods:
    """Test deprecated methods raise appropriate errors."""

    def test_budget_allocation_raises_not_implemented(self, mock_suite):
        """Test that budget_allocation() raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match=r"budget_allocation.*removed"):
            mock_suite.budget_allocation()


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestAdditionalCoverage:
    """Additional tests to reach >95% coverage."""

    def test_posterior_predictive_with_explicit_idata(self):
        """Test posterior_predictive with explicit idata parameter."""
        from arviz_plots import PlotCollection

        rng = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=52, freq="W")

        # Create posterior_predictive dataset
        pp_data = xr.Dataset(
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

        # Create suite without idata
        suite = MMMPlotSuite(idata=None)

        # Should work with explicit idata parameter
        pc = suite.posterior_predictive(var="y", idata=pp_data)
        assert isinstance(pc, PlotCollection)

    def test_saturation_curves_with_invalid_hdi_probs_type(
        self, mock_suite_with_constant_data, mock_saturation_curve
    ):
        """Test that invalid hdi_probs type raises TypeError."""
        with pytest.raises(TypeError, match="hdi_probs must be a float"):
            mock_suite_with_constant_data.saturation_curves(
                curve=mock_saturation_curve, hdi_probs={"invalid": "type"}
            )

    def test_uplift_curve_with_dataset_missing_both_variables(self):
        """Test uplift_curve when Dataset has neither uplift_curve nor x."""
        rng = np.random.default_rng(42)
        data = xr.Dataset(
            {
                "other_var": xr.DataArray(
                    rng.normal(size=(100, 20)),
                    dims=("sample", "sweep"),
                    coords={
                        "sample": np.arange(100),
                        "sweep": np.linspace(0, 1, 20),
                    },
                )
            }
        )

        suite = MMMPlotSuite(idata=None)
        with pytest.raises(ValueError, match="must contain 'uplift_curve' or 'x'"):
            suite.uplift_curve(data=data)

    def test_marginal_curve_with_dataset_missing_both_variables(self):
        """Test marginal_curve when Dataset has neither marginal_effects nor x."""
        rng = np.random.default_rng(42)
        data = xr.Dataset(
            {
                "other_var": xr.DataArray(
                    rng.normal(size=(100, 20)),
                    dims=("sample", "sweep"),
                    coords={
                        "sample": np.arange(100),
                        "sweep": np.linspace(0, 1, 20),
                    },
                )
            }
        )

        suite = MMMPlotSuite(idata=None)
        with pytest.raises(ValueError, match="must contain 'marginal_effects' or 'x'"):
            suite.marginal_curve(data=data)

    def test_sensitivity_analysis_with_aggregation_no_matching_dims(self):
        """Test sensitivity_analysis_plot with aggregation but no matching dims."""
        from arviz_plots import PlotCollection

        # Create data without the dimension to aggregate
        data = xr.Dataset(
            {
                "x": xr.DataArray(
                    np.random.rand(100, 20),
                    dims=("sample", "sweep"),
                    coords={
                        "sample": np.arange(100),
                        "sweep": np.linspace(0, 1, 20),
                    },
                )
            }
        )

        suite = MMMPlotSuite(idata=None)
        # Should work even though "channel" doesn't exist in data
        pc = suite._sensitivity_analysis_plot(
            data=data, aggregation={"sum": ("channel",)}
        )
        assert isinstance(pc, PlotCollection)
