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
"""Regression tests for legacy plot suite."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pymc_marketing.mmm.legacy_plot import LegacyMMMPlotSuite


def test_legacy_suite_all_methods_exist():
    """Test all legacy suite methods still exist after rename."""
    expected_methods = [
        "posterior_predictive",
        "contributions_over_time",
        "saturation_scatterplot",
        "saturation_curves",
        "saturation_curves_scatter",  # Deprecated but still in legacy
        "budget_allocation",
        "allocated_contribution_by_channel_over_time",
        "sensitivity_analysis",
        "uplift_curve",
        "marginal_curve",
    ]

    for method_name in expected_methods:
        assert hasattr(LegacyMMMPlotSuite, method_name), (
            f"LegacyMMMPlotSuite missing method: {method_name}"
        )


def test_legacy_suite_returns_tuple(mock_idata_for_legacy):
    """Test legacy suite returns tuple, not PlotCollection."""
    suite = LegacyMMMPlotSuite(idata=mock_idata_for_legacy)
    result = suite.posterior_predictive()

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], Figure)
    # result[1] can be Axes or ndarray of Axes
    if isinstance(result[1], np.ndarray):
        assert all(isinstance(ax, Axes) for ax in result[1].flat)
    else:
        assert isinstance(result[1], Axes)
