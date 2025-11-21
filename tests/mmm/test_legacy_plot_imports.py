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
"""Tests for legacy plot module renaming."""

import pytest


def test_legacy_plot_module_exists():
    """Test that legacy_plot module exists and can be imported."""
    try:
        from pymc_marketing.mmm import legacy_plot

        assert hasattr(legacy_plot, "LegacyMMMPlotSuite")
    except ImportError as e:
        pytest.fail(f"Failed to import legacy_plot: {e}")


def test_legacy_class_name():
    """Test that legacy suite class is named LegacyMMMPlotSuite."""
    from pymc_marketing.mmm.legacy_plot import LegacyMMMPlotSuite

    assert LegacyMMMPlotSuite.__name__ == "LegacyMMMPlotSuite"
