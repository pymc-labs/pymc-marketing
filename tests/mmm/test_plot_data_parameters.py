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
"""Tests for data parameter standardization across plotting methods."""

import arviz_plots
import pytest
import xarray as xr

from pymc_marketing.mmm.plot import MMMPlotSuite


def test_contributions_over_time_accepts_data_parameter(mock_posterior_data):
    """Test that contributions_over_time accepts data parameter."""
    # Create suite without idata
    suite = MMMPlotSuite(idata=None)

    # Should work with explicit data parameter
    pc = suite.contributions_over_time(var=["intercept"], data=mock_posterior_data)

    assert isinstance(pc, arviz_plots.PlotCollection)


def test_contributions_over_time_data_parameter_fallback(mock_idata_with_posterior):
    """Test that contributions_over_time falls back to self.idata.posterior."""
    suite = MMMPlotSuite(idata=mock_idata_with_posterior)

    # Should work without data parameter (fallback)
    pc = suite.contributions_over_time(var=["intercept"])

    assert isinstance(pc, arviz_plots.PlotCollection)


def test_contributions_over_time_no_data_raises_clear_error():
    """Test clear error when no data available."""
    suite = MMMPlotSuite(idata=None)

    with pytest.raises(
        ValueError, match=r"No posterior data found.*and no 'data' argument provided"
    ):
        suite.contributions_over_time(var=["intercept"])


def test_saturation_scatterplot_accepts_data_parameters(
    mock_constant_data, mock_posterior_data
):
    """Test saturation_scatterplot accepts data parameters."""
    import numpy as np

    # Need to add channel_contribution to mock_posterior_data
    # Replicate the data across the channel dimension (3 channels)
    intercept_values = mock_posterior_data["intercept"].values
    channel_contrib_values = np.repeat(intercept_values[:, :, :, np.newaxis], 3, axis=3)

    mock_posterior_data["channel_contribution"] = xr.DataArray(
        channel_contrib_values,
        dims=("chain", "draw", "date", "channel"),
        coords={
            **{k: v for k, v in mock_posterior_data.coords.items()},
            "channel": ["TV", "Radio", "Digital"],
        },
    )

    suite = MMMPlotSuite(idata=None)

    pc = suite.saturation_scatterplot(
        constant_data=mock_constant_data, posterior_data=mock_posterior_data
    )

    assert isinstance(pc, arviz_plots.PlotCollection)


def test_sensitivity_analysis_plot_requires_data_parameter(mock_sensitivity_data):
    """Test _sensitivity_analysis_plot requires data parameter (no fallback)."""
    suite = MMMPlotSuite(idata=None)

    # Should work with data parameter
    pc = suite._sensitivity_analysis_plot(data=mock_sensitivity_data)

    assert isinstance(pc, arviz_plots.PlotCollection)


def test_sensitivity_analysis_plot_no_fallback_to_self_idata(
    mock_idata_with_sensitivity,
):
    """Test _sensitivity_analysis_plot doesn't use self.idata even if available."""
    suite = MMMPlotSuite(idata=mock_idata_with_sensitivity)

    # Should raise error even though self.idata has sensitivity_analysis
    with pytest.raises(TypeError, match=r"missing.*required.*argument.*data"):
        suite._sensitivity_analysis_plot()


def test_uplift_curve_passes_data_to_helper_no_monkey_patch(
    mock_idata_with_uplift_curve,
):
    """Test uplift_curve passes data directly, no monkey-patching."""
    suite = MMMPlotSuite(idata=mock_idata_with_uplift_curve)

    # Store original idata reference
    original_idata = suite.idata
    original_sa_group = original_idata.sensitivity_analysis

    # Call uplift_curve
    pc = suite.uplift_curve()

    # Verify no monkey-patching occurred
    assert suite.idata is original_idata
    assert suite.idata.sensitivity_analysis is original_sa_group
    assert isinstance(pc, arviz_plots.PlotCollection)
