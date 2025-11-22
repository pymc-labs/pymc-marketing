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
"""Compatibility tests for plot suite version switching."""

import warnings

import numpy as np
import pytest
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class TestVersionSwitching:
    """Test that mmm_config['plot.use_v2'] controls which suite is returned."""

    def test_use_v2_false_returns_legacy_suite(self, mock_mmm):
        """Test that use_v2=False returns LegacyMMMPlotSuite."""
        from pymc_marketing.mmm import mmm_config
        from pymc_marketing.mmm.legacy_plot import LegacyMMMPlotSuite

        original = mmm_config.get("plot.use_v2", False)
        try:
            mmm_config["plot.use_v2"] = False

            with pytest.warns(FutureWarning, match="deprecated in v0.20.0"):
                plot_suite = mock_mmm.plot

            assert isinstance(plot_suite, LegacyMMMPlotSuite)
            assert plot_suite.__class__.__name__ == "LegacyMMMPlotSuite"
        finally:
            mmm_config["plot.use_v2"] = original

    def test_use_v2_true_returns_new_suite(self, mock_mmm):
        """Test that use_v2=True returns MMMPlotSuite."""
        from pymc_marketing.mmm import mmm_config
        from pymc_marketing.mmm.legacy_plot import LegacyMMMPlotSuite
        from pymc_marketing.mmm.plot import MMMPlotSuite

        original = mmm_config.get("plot.use_v2", False)
        try:
            mmm_config["plot.use_v2"] = True

            # Should not warn
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                plot_suite = mock_mmm.plot

            assert isinstance(plot_suite, MMMPlotSuite)
            assert not isinstance(plot_suite, LegacyMMMPlotSuite)
            assert plot_suite.__class__.__name__ == "MMMPlotSuite"
        finally:
            mmm_config["plot.use_v2"] = original

    def test_default_is_legacy_suite(self, mock_mmm):
        """Test that default behavior uses legacy suite (backward compatible)."""
        from pymc_marketing.mmm import mmm_config
        from pymc_marketing.mmm.legacy_plot import LegacyMMMPlotSuite

        # Ensure default state
        if "plot.use_v2" in mmm_config:
            del mmm_config["plot.use_v2"]

        with pytest.warns(FutureWarning):
            plot_suite = mock_mmm.plot

        assert isinstance(plot_suite, LegacyMMMPlotSuite)

    def test_config_flag_persists_across_calls(self, mock_mmm):
        """Test that setting config flag affects all subsequent calls."""
        from pymc_marketing.mmm import mmm_config
        from pymc_marketing.mmm.plot import MMMPlotSuite

        original = mmm_config.get("plot.use_v2", False)
        try:
            # Set once
            mmm_config["plot.use_v2"] = True

            # Multiple calls should all use new suite
            plot_suite1 = mock_mmm.plot
            plot_suite2 = mock_mmm.plot
            plot_suite3 = mock_mmm.plot

            assert isinstance(plot_suite1, MMMPlotSuite)
            assert isinstance(plot_suite2, MMMPlotSuite)
            assert isinstance(plot_suite3, MMMPlotSuite)
        finally:
            mmm_config["plot.use_v2"] = original

    def test_switching_between_v2_true_and_false(self, mock_mmm):
        """Test that switching from use_v2=True to False and back works correctly."""
        from pymc_marketing.mmm import mmm_config
        from pymc_marketing.mmm.legacy_plot import LegacyMMMPlotSuite
        from pymc_marketing.mmm.plot import MMMPlotSuite

        original = mmm_config.get("plot.use_v2", False)
        try:
            # Start with use_v2 = True
            mmm_config["plot.use_v2"] = True

            # Should return new suite without warnings
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                plot_suite_v2 = mock_mmm.plot

            assert isinstance(plot_suite_v2, MMMPlotSuite)

            # Switch to use_v2 = False
            mmm_config["plot.use_v2"] = False

            # Should return legacy suite with deprecation warning
            with pytest.warns(FutureWarning, match="deprecated in v0.20.0"):
                plot_suite_legacy = mock_mmm.plot

            assert isinstance(plot_suite_legacy, LegacyMMMPlotSuite)

            # Switch back to use_v2 = True
            mmm_config["plot.use_v2"] = True

            # Should return new suite again without warnings
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                plot_suite_v2_again = mock_mmm.plot

            assert isinstance(plot_suite_v2_again, MMMPlotSuite)
        finally:
            mmm_config["plot.use_v2"] = original


class TestDeprecationWarnings:
    """Test deprecation warnings shown correctly with helpful information."""

    def test_deprecation_warning_shown_by_default(self, mock_mmm):
        """Test that deprecation warning is shown when using legacy suite."""
        from pymc_marketing.mmm import mmm_config

        original_use_v2 = mmm_config.get("plot.use_v2", False)
        original_warnings = mmm_config.get("plot.show_warnings", True)

        try:
            mmm_config["plot.use_v2"] = False
            mmm_config["plot.show_warnings"] = True

            with pytest.warns(FutureWarning, match=r"deprecated in v0\.20\.0"):
                plot_suite = mock_mmm.plot

            assert plot_suite is not None
        finally:
            mmm_config["plot.use_v2"] = original_use_v2
            mmm_config["plot.show_warnings"] = original_warnings

    def test_deprecation_warning_suppressible(self, mock_mmm):
        """Test that deprecation warning can be suppressed."""
        from pymc_marketing.mmm import mmm_config

        original_use_v2 = mmm_config.get("plot.use_v2", False)
        original_warnings = mmm_config.get("plot.show_warnings", True)

        try:
            mmm_config["plot.use_v2"] = False
            mmm_config["plot.show_warnings"] = False

            # Should not warn
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # Turn warnings into errors
                plot_suite = mock_mmm.plot

            assert plot_suite is not None
        finally:
            mmm_config["plot.use_v2"] = original_use_v2
            mmm_config["plot.show_warnings"] = original_warnings

    def test_warning_message_includes_migration_info(self, mock_mmm):
        """Test that warning provides clear migration instructions."""
        from pymc_marketing.mmm import mmm_config

        original_use_v2 = mmm_config.get("plot.use_v2", False)

        try:
            mmm_config["plot.use_v2"] = False

            with pytest.warns(FutureWarning) as warning_list:
                _ = mock_mmm.plot

            warning_msg = str(warning_list[0].message)

            # Check for key information
            assert "v0.20.0" in warning_msg, "Should mention removal version"
            assert "plot.use_v2" in warning_msg, "Should show how to enable v2"
            assert "True" in warning_msg, "Should show value to set"
            assert any(
                word in warning_msg.lower()
                for word in ["migration", "guide", "documentation", "docs"]
            ), "Should reference migration guide"
        finally:
            mmm_config["plot.use_v2"] = original_use_v2

    def test_no_warning_when_using_new_suite(self, mock_mmm):
        """Test that no warning shown when using new suite."""
        from pymc_marketing.mmm import mmm_config

        original = mmm_config.get("plot.use_v2", False)

        try:
            mmm_config["plot.use_v2"] = True

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                plot_suite = mock_mmm.plot

            assert plot_suite is not None
        finally:
            mmm_config["plot.use_v2"] = original


class TestReturnTypeCompatibility:
    """Test both suites return correct, expected types."""

    def test_legacy_suite_returns_tuple(self, mock_mmm_fitted):
        """Test legacy suite returns (Figure, Axes) tuple."""
        from pymc_marketing.mmm import mmm_config

        original = mmm_config.get("plot.use_v2", False)

        try:
            mmm_config["plot.use_v2"] = False

            with pytest.warns(FutureWarning):
                plot_suite = mock_mmm_fitted.plot
                result = plot_suite.posterior_predictive()

            assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
            assert len(result) == 2, f"Expected 2-tuple, got length {len(result)}"
            assert isinstance(result[0], Figure), (
                f"Expected Figure, got {type(result[0])}"
            )

            # result[1] can be Axes or ndarray of Axes
            if isinstance(result[1], np.ndarray):
                assert all(isinstance(ax, Axes) for ax in result[1].flat)
            else:
                assert isinstance(result[1], Axes)
        finally:
            mmm_config["plot.use_v2"] = original

    def test_new_suite_returns_plot_collection(self, mock_mmm_fitted):
        """Test new suite returns PlotCollection."""
        from pymc_marketing.mmm import mmm_config

        original = mmm_config.get("plot.use_v2", False)

        try:
            mmm_config["plot.use_v2"] = True

            plot_suite = mock_mmm_fitted.plot
            result = plot_suite.posterior_predictive()

            assert isinstance(result, PlotCollection), (
                f"Expected PlotCollection, got {type(result)}"
            )
            assert hasattr(result, "backend"), (
                "PlotCollection should have backend attribute"
            )
            assert hasattr(result, "show"), "PlotCollection should have show method"
        finally:
            mmm_config["plot.use_v2"] = original

    def test_both_suites_produce_valid_plots(self, mock_mmm_fitted):
        """Test that both suites can successfully create plots."""
        from pymc_marketing.mmm import mmm_config

        original = mmm_config.get("plot.use_v2", False)

        try:
            # Legacy suite
            mmm_config["plot.use_v2"] = False
            with pytest.warns(FutureWarning):
                legacy_result = mock_mmm_fitted.plot.contributions_over_time(
                    var=["intercept"]
                )
            assert legacy_result is not None

            # New suite
            mmm_config["plot.use_v2"] = True
            new_result = mock_mmm_fitted.plot.contributions_over_time(var=["intercept"])
            assert new_result is not None
        finally:
            mmm_config["plot.use_v2"] = original


class TestDeprecatedMethodRemoval:
    """Test saturation_curves_scatter() removed from new suite but kept in legacy."""

    def test_saturation_curves_scatter_removed_from_new_suite(self, mock_mmm_fitted):
        """Test saturation_curves_scatter removed from new MMMPlotSuite."""
        from pymc_marketing.mmm import mmm_config

        original = mmm_config.get("plot.use_v2", False)

        try:
            mmm_config["plot.use_v2"] = True
            plot_suite = mock_mmm_fitted.plot

            assert not hasattr(plot_suite, "saturation_curves_scatter"), (
                "saturation_curves_scatter should not exist in new MMMPlotSuite"
            )
        finally:
            mmm_config["plot.use_v2"] = original

    def test_saturation_curves_scatter_exists_in_legacy_suite(self, mock_mmm_fitted):
        """Test saturation_curves_scatter still exists in LegacyMMMPlotSuite."""
        from pymc_marketing.mmm import mmm_config

        original = mmm_config.get("plot.use_v2", False)

        try:
            mmm_config["plot.use_v2"] = False

            with pytest.warns(FutureWarning):
                plot_suite = mock_mmm_fitted.plot

            assert hasattr(plot_suite, "saturation_curves_scatter"), (
                "saturation_curves_scatter should exist in LegacyMMMPlotSuite"
            )
        finally:
            mmm_config["plot.use_v2"] = original


class TestMissingMethods:
    """Test methods that exist in one suite but not the other handle gracefully."""

    def test_budget_allocation_exists_in_legacy_suite(
        self, mock_mmm_fitted, mock_allocation_samples
    ):
        """Test that budget_allocation() works in legacy suite."""
        from pymc_marketing.mmm import mmm_config

        original = mmm_config.get("plot.use_v2", False)

        try:
            mmm_config["plot.use_v2"] = False

            with pytest.warns(FutureWarning):
                plot_suite = mock_mmm_fitted.plot

            # Should work (not raise AttributeError)
            result = plot_suite.budget_allocation(samples=mock_allocation_samples)
            assert isinstance(result, tuple)
            assert len(result) == 2
        finally:
            mmm_config["plot.use_v2"] = original

    def test_budget_allocation_raises_in_new_suite(self, mock_mmm_fitted):
        """Test that budget_allocation() raises helpful error in new suite."""
        from pymc_marketing.mmm import mmm_config

        original = mmm_config.get("plot.use_v2", False)

        try:
            mmm_config["plot.use_v2"] = True
            plot_suite = mock_mmm_fitted.plot

            with pytest.raises(NotImplementedError, match="removed in MMMPlotSuite v2"):
                plot_suite.budget_allocation(samples=None)
        finally:
            mmm_config["plot.use_v2"] = original

    def test_budget_allocation_roas_exists_in_new_suite(self, mock_mmm_fitted):
        """Test that budget_allocation_roas() exists in new suite."""
        from pymc_marketing.mmm import mmm_config

        original = mmm_config.get("plot.use_v2", False)

        try:
            mmm_config["plot.use_v2"] = True
            plot_suite = mock_mmm_fitted.plot

            # Just check that the method exists (not AttributeError)
            assert hasattr(plot_suite, "budget_allocation_roas"), (
                "budget_allocation_roas should exist in new MMMPlotSuite"
            )
            assert callable(plot_suite.budget_allocation_roas), (
                "budget_allocation_roas should be callable"
            )
        finally:
            mmm_config["plot.use_v2"] = original

    def test_budget_allocation_roas_missing_in_legacy_suite(self, mock_mmm_fitted):
        """Test that budget_allocation_roas() doesn't exist in legacy suite."""
        from pymc_marketing.mmm import mmm_config

        original = mmm_config.get("plot.use_v2", False)

        try:
            mmm_config["plot.use_v2"] = False

            with pytest.warns(FutureWarning):
                plot_suite = mock_mmm_fitted.plot

            with pytest.raises(AttributeError):
                plot_suite.budget_allocation_roas(samples=None)
        finally:
            mmm_config["plot.use_v2"] = original


class TestConfigValidation:
    """Test MMMConfig key validation."""

    def test_invalid_key_warns_but_allows_setting(self):
        """Test that setting an invalid config key warns but still sets the value."""
        from pymc_marketing.mmm import mmm_config

        # Store original state
        original_invalid = mmm_config.get("invalid.key", None)
        try:
            # Try to set an invalid key
            with pytest.warns(UserWarning, match="Invalid config key"):
                mmm_config["invalid.key"] = "some_value"

            # Verify the warning message contains valid keys
            with pytest.warns(UserWarning) as warning_list:
                mmm_config["another.invalid.key"] = "another_value"

            warning_msg = str(warning_list[0].message)
            assert "Invalid config key" in warning_msg
            assert "another.invalid.key" in warning_msg
            assert "plot.backend" in warning_msg or "plot.show_warnings" in warning_msg

            # Verify the invalid key was still set (allows setting but warns)
            assert mmm_config["invalid.key"] == "some_value"
            assert mmm_config["another.invalid.key"] == "another_value"
        finally:
            # Clean up invalid keys
            if "invalid.key" in mmm_config:
                del mmm_config["invalid.key"]
            if "another.invalid.key" in mmm_config:
                del mmm_config["another.invalid.key"]
            if original_invalid is not None:
                mmm_config["invalid.key"] = original_invalid

    def test_valid_keys_do_not_warn(self):
        """Test that setting valid config keys does not warn."""
        from pymc_marketing.mmm import mmm_config

        original_backend = mmm_config.get("plot.backend", "matplotlib")
        original_use_v2 = mmm_config.get("plot.use_v2", False)
        original_warnings = mmm_config.get("plot.show_warnings", True)

        try:
            # Setting valid keys should not warn
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                mmm_config["plot.backend"] = "plotly"
                mmm_config["plot.use_v2"] = True
                mmm_config["plot.show_warnings"] = False

            # Verify values were set
            assert mmm_config["plot.backend"] == "plotly"
            assert mmm_config["plot.use_v2"] is True
            assert mmm_config["plot.show_warnings"] is False
        finally:
            mmm_config["plot.backend"] = original_backend
            mmm_config["plot.use_v2"] = original_use_v2
            mmm_config["plot.show_warnings"] = original_warnings

    def test_reset_restores_defaults(self):
        """Test that reset() restores all configuration to default values."""
        from pymc_marketing.mmm import mmm_config

        # Store original state
        original_backend = mmm_config.get("plot.backend", "matplotlib")
        original_use_v2 = mmm_config.get("plot.use_v2", False)
        original_warnings = mmm_config.get("plot.show_warnings", True)

        try:
            # Change all config values
            mmm_config["plot.backend"] = "plotly"
            mmm_config["plot.use_v2"] = True
            mmm_config["plot.show_warnings"] = False

            # Verify they were changed
            assert mmm_config["plot.backend"] == "plotly"
            assert mmm_config["plot.use_v2"] is True
            assert mmm_config["plot.show_warnings"] is False

            # Reset to defaults
            mmm_config.reset()

            # Verify all values are back to defaults
            assert mmm_config["plot.backend"] == "matplotlib"
            assert mmm_config["plot.use_v2"] is False
            assert mmm_config["plot.show_warnings"] is True

            # Verify reset clears any invalid keys that were set
            mmm_config["invalid.key"] = "test"
            assert "invalid.key" in mmm_config
            mmm_config.reset()
            assert "invalid.key" not in mmm_config
        finally:
            # Restore original state
            mmm_config["plot.backend"] = original_backend
            mmm_config["plot.use_v2"] = original_use_v2
            mmm_config["plot.show_warnings"] = original_warnings

    def test_invalid_backend_warns_but_allows_setting(self):
        """Test that setting an invalid backend warns but still sets the value."""
        from pymc_marketing.mmm import mmm_config

        original_backend = mmm_config.get("plot.backend", "matplotlib")

        try:
            # Try to set an invalid backend
            with pytest.warns(UserWarning, match="Invalid backend"):
                mmm_config["plot.backend"] = "invalid_backend"

            # Verify the warning message contains valid backends
            with pytest.warns(UserWarning) as warning_list:
                mmm_config["plot.backend"] = "another_invalid"

            warning_msg = str(warning_list[0].message)
            assert "Invalid backend" in warning_msg
            assert "another_invalid" in warning_msg
            assert (
                "matplotlib" in warning_msg
                or "plotly" in warning_msg
                or "bokeh" in warning_msg
            )

            # Verify the invalid backend was still set (allows setting but warns)
            assert mmm_config["plot.backend"] == "another_invalid"
        finally:
            mmm_config["plot.backend"] = original_backend

    def test_valid_backends_do_not_warn(self):
        """Test that setting valid backend values does not warn."""
        from pymc_marketing.mmm import mmm_config

        original_backend = mmm_config.get("plot.backend", "matplotlib")

        try:
            # Setting valid backends should not warn
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                mmm_config["plot.backend"] = "matplotlib"
                mmm_config["plot.backend"] = "plotly"
                mmm_config["plot.backend"] = "bokeh"

            # Verify values were set
            assert mmm_config["plot.backend"] == "bokeh"
        finally:
            mmm_config["plot.backend"] = original_backend
