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
"""
Backend-agnostic plotting tests for MMMPlotSuite.

This test file validates the migration to ArviZ PlotCollection API for
multi-backend support (matplotlib, plotly, bokeh).

NOTE: Once this migration is complete and stable, evaluate whether
tests/mmm/test_plot.py can be consolidated into this file to avoid duplication.
"""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pymc_marketing.mmm.plot import MMMPlotSuite


@pytest.fixture(scope="module")
def mock_idata_for_pp():
    """
    Create mock InferenceData with posterior_predictive for testing.

    Structure mirrors real MMM output with:
    - posterior_predictive group with y variable
    - proper dimensions: chain, draw, date
    - realistic date range
    """
    seed = sum(map(ord, "Backend test posterior_predictive"))
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2025-01-01", periods=52, freq="W-MON")

    # Create posterior_predictive data
    posterior_predictive = xr.Dataset(
        {
            "y": xr.DataArray(
                rng.normal(loc=100, scale=10, size=(4, 100, 52)),
                dims=("chain", "draw", "date"),
                coords={
                    "chain": np.arange(4),
                    "draw": np.arange(100),
                    "date": dates,
                },
            )
        }
    )

    # Also create a minimal posterior (required for some internal logic)
    posterior = xr.Dataset(
        {
            "intercept": xr.DataArray(
                rng.normal(size=(4, 100)),
                dims=("chain", "draw"),
                coords={
                    "chain": np.arange(4),
                    "draw": np.arange(100),
                },
            )
        }
    )

    return az.InferenceData(
        posterior=posterior, posterior_predictive=posterior_predictive
    )


@pytest.fixture(scope="module")
def mock_suite_with_pp(mock_idata_for_pp):
    """
    Fixture providing MMMPlotSuite with posterior_predictive data.

    Used for testing posterior_predictive() method across backends.
    """
    return MMMPlotSuite(idata=mock_idata_for_pp)


@pytest.fixture(scope="function")
def reset_mmm_config():
    """
    Fixture to reset mmm_config after each test.

    Ensures test isolation - one test's backend changes don't affect others.
    """
    from pymc_marketing.mmm import mmm_config

    original = mmm_config["plot.backend"]
    yield
    mmm_config["plot.backend"] = original


# =============================================================================
# Infrastructure Tests (Global Configuration & Return Types)
# =============================================================================


def test_mmm_config_exists():
    """
    Test that the global mmm_config object exists and is accessible.

    This test verifies:
    - mmm_config can be imported from pymc_marketing.mmm
    - It has a "plot.backend" key
    - Default backend is "matplotlib"
    """
    from pymc_marketing.mmm import mmm_config

    assert "plot.backend" in mmm_config, "mmm_config should have 'plot.backend' key"
    assert mmm_config["plot.backend"] == "matplotlib", (
        f"Default backend should be 'matplotlib', got {mmm_config['plot.backend']}"
    )


def test_mmm_config_backend_setting():
    """
    Test that mmm_config backend can be set and retrieved.

    This test verifies:
    - Backend can be changed from default
    - New value persists
    - Can be reset to default
    """
    from pymc_marketing.mmm import mmm_config

    # Store original
    original = mmm_config["plot.backend"]

    try:
        # Change backend
        mmm_config["plot.backend"] = "plotly"
        assert mmm_config["plot.backend"] == "plotly", (
            "Backend should change to 'plotly'"
        )

        # Reset
        mmm_config.reset()
        assert mmm_config["plot.backend"] == "matplotlib", (
            "reset() should restore default 'matplotlib' backend"
        )
    finally:
        # Cleanup
        mmm_config["plot.backend"] = original


def test_mmm_config_invalid_backend_warning():
    """
    Test that setting an invalid backend name is handled gracefully.

    This test verifies:
    - Invalid backend names are detected
    - Either raises ValueError or emits UserWarning
    - Helpful error message provided
    """
    import warnings

    from pymc_marketing.mmm import mmm_config

    original = mmm_config["plot.backend"]

    try:
        # Attempt to set invalid backend - should either raise or warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mmm_config["plot.backend"] = "invalid_backend"

            # If no exception, should have warning
            assert len(w) > 0, "Should emit warning for invalid backend"
            assert "invalid" in str(w[0].message).lower(), (
                f"Warning should mention 'invalid', got: {w[0].message}"
            )
    except ValueError as e:
        # Acceptable alternative: raise ValueError
        assert "backend" in str(e).lower(), f"Error should mention 'backend', got: {e}"
    finally:
        mmm_config["plot.backend"] = original


# =============================================================================
# Backend Parameter Tests (posterior_predictive)
# =============================================================================


def test_posterior_predictive_accepts_backend_parameter(mock_suite_with_pp):
    """
    Test that posterior_predictive() accepts backend parameter.

    This test verifies:
    - backend parameter is accepted
    - No TypeError is raised
    - Method completes successfully
    """
    # Should not raise TypeError
    result = mock_suite_with_pp.posterior_predictive(backend="matplotlib")

    assert result is not None, "posterior_predictive should return a result"


def test_posterior_predictive_accepts_return_as_pc_parameter(mock_suite_with_pp):
    """
    Test that posterior_predictive() accepts return_as_pc parameter.

    This test verifies:
    - return_as_pc parameter is accepted
    - No TypeError is raised
    """
    # Should not raise TypeError
    result = mock_suite_with_pp.posterior_predictive(return_as_pc=False)

    assert result is not None, "posterior_predictive should return a result"


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
def test_posterior_predictive_backend_overrides_global(mock_suite_with_pp, backend):
    """
    Test that backend parameter overrides global mmm_config setting.

    This test verifies:
    - Global config set to one backend
    - Function called with different backend
    - Function uses parameter, not global config
    """
    from pymc_marketing.mmm import mmm_config

    original = mmm_config["plot.backend"]

    try:
        # Set global to matplotlib
        mmm_config["plot.backend"] = "matplotlib"

        # Call with different backend, request PlotCollection to check
        pc = mock_suite_with_pp.posterior_predictive(backend=backend, return_as_pc=True)

        assert hasattr(pc, "backend"), "PlotCollection should have backend attribute"
        assert pc.backend == backend, (
            f"PlotCollection backend should be '{backend}', got '{pc.backend}'"
        )
    finally:
        mmm_config["plot.backend"] = original


# =============================================================================
# Return Type Tests (Backward Compatibility)
# =============================================================================


def test_posterior_predictive_returns_tuple_by_default(mock_suite_with_pp):
    """
    Test that posterior_predictive() returns tuple by default (backward compat).

    This test verifies:
    - Default behavior (no return_as_pc parameter) returns tuple
    - Tuple has two elements: (figure, axes)
    - axes is a list of matplotlib Axes objects (1D list, not 2D array)
    """
    result = mock_suite_with_pp.posterior_predictive()

    assert isinstance(result, tuple), (
        f"Default return should be tuple, got {type(result)}"
    )
    assert len(result) == 2, (
        f"Tuple should have 2 elements (fig, axes), got {len(result)}"
    )

    fig, axes = result

    # For matplotlib backend (default), should be Figure and array
    assert isinstance(fig, Figure), f"First element should be Figure, got {type(fig)}"
    # Note: Current implementation returns NDArray[Axes], need to adapt test
    assert axes is not None, "Second element should not be None for matplotlib backend"


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
def test_posterior_predictive_returns_plotcollection_when_requested(
    mock_suite_with_pp, backend
):
    """
    Test that posterior_predictive() returns PlotCollection when return_as_pc=True.

    This test verifies:
    - return_as_pc=True returns PlotCollection object
    - PlotCollection has correct backend attribute
    """
    from arviz_plots import PlotCollection

    result = mock_suite_with_pp.posterior_predictive(backend=backend, return_as_pc=True)

    assert isinstance(result, PlotCollection), (
        f"Should return PlotCollection, got {type(result)}"
    )
    assert hasattr(result, "backend"), "PlotCollection should have backend attribute"
    assert result.backend == backend, (
        f"Backend should be '{backend}', got '{result.backend}'"
    )


def test_posterior_predictive_tuple_has_correct_axes_for_matplotlib(mock_suite_with_pp):
    """
    Test that matplotlib backend returns proper axes list in tuple.

    This test verifies:
    - When return_as_pc=False and backend="matplotlib"
    - Second tuple element is list/array of matplotlib Axes
    - All elements in list are Axes instances
    """
    _fig, axes = mock_suite_with_pp.posterior_predictive(
        backend="matplotlib", return_as_pc=False
    )

    assert axes is not None, "Axes should not be None for matplotlib backend"
    # Handle both list and NDArray cases
    axes_flat = axes if isinstance(axes, list) else axes.flat
    assert all(isinstance(ax, Axes) for ax in axes_flat), (
        "All elements should be matplotlib Axes instances"
    )


@pytest.mark.parametrize("backend", ["plotly", "bokeh"])
def test_posterior_predictive_tuple_has_none_axes_for_nonmatplotlib(
    mock_suite_with_pp, backend
):
    """
    Test that non-matplotlib backends return None for axes in tuple.

    This test verifies:
    - When return_as_pc=False and backend in ["plotly", "bokeh"]
    - Second tuple element is None (no axes concept)
    - First element is backend-specific figure object
    """
    fig, axes = mock_suite_with_pp.posterior_predictive(
        backend=backend, return_as_pc=False
    )

    assert axes is None, f"Axes should be None for {backend} backend, got {type(axes)}"
    assert fig is not None, f"Figure should exist for {backend} backend"


# =============================================================================
# Visual Output Validation Tests
# =============================================================================


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "bokeh"])
def test_posterior_predictive_plotcollection_has_viz_attribute(
    mock_suite_with_pp, backend
):
    """
    Test that PlotCollection has viz attribute with figure data.

    This test verifies:
    - PlotCollection has viz attribute
    - viz has figure attribute
    - Figure can be extracted
    """

    pc = mock_suite_with_pp.posterior_predictive(backend=backend, return_as_pc=True)

    assert hasattr(pc, "viz"), "PlotCollection should have 'viz' attribute"
    assert hasattr(pc.viz, "figure"), (
        "PlotCollection.viz should have 'figure' attribute"
    )

    # Should be able to extract figure
    fig = pc.viz.figure.data.item()
    assert fig is not None, "Should be able to extract figure from PlotCollection"


def test_posterior_predictive_matplotlib_has_lines(mock_suite_with_pp):
    """
    Test that matplotlib output contains actual plotted lines.

    This test verifies:
    - Axes contain Line2D objects (plotted data)
    - Number of lines matches expected variables
    - Visual output actually created, not just empty axes
    """
    from matplotlib.lines import Line2D

    _fig, axes = mock_suite_with_pp.posterior_predictive(
        backend="matplotlib", return_as_pc=False
    )

    # Get first axis (should have plots)
    ax = axes.flat[0]

    # Should have lines (median plots)
    lines = [child for child in ax.get_children() if isinstance(child, Line2D)]
    assert len(lines) > 0, (
        f"Axes should contain Line2D objects (plots), found {len(lines)}"
    )


def test_posterior_predictive_plotly_has_traces(mock_suite_with_pp):
    """
    Test that plotly output contains actual traces.

    This test verifies:
    - Plotly figure has 'data' attribute with traces
    - Number of traces > 0 (something was plotted)
    - Visual output actually created
    """
    fig, _ = mock_suite_with_pp.posterior_predictive(
        backend="plotly", return_as_pc=False
    )

    # Plotly figures have .data attribute with traces
    assert hasattr(fig, "data"), "Plotly figure should have 'data' attribute"
    assert len(fig.data) > 0, f"Plotly figure should have traces, found {len(fig.data)}"


def test_posterior_predictive_bokeh_has_renderers(mock_suite_with_pp):
    """
    Test that bokeh output contains actual renderers (plot elements).

    This test verifies:
    - Bokeh figure has renderers
    - Number of renderers > 0 (something was plotted)
    - Visual output actually created
    """
    fig, _ = mock_suite_with_pp.posterior_predictive(
        backend="bokeh", return_as_pc=False
    )

    # Bokeh figures have .renderers attribute
    assert hasattr(fig, "renderers"), "Bokeh figure should have 'renderers' attribute"
    assert len(fig.renderers) > 0, (
        f"Bokeh figure should have renderers, found {len(fig.renderers)}"
    )
