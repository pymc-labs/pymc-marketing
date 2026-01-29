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
"""Tests for MMMPlotlyFactory - Interactive Plotly plotting."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import pytest

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.mmm.plot_interactive import MMMPlotlyFactory
from pymc_marketing.mmm.summary import MMMSummaryFactory


class TestMMMPlotlyFactoryContributions:
    """Tests for MMMPlotlyFactory.contributions() method."""

    def _create_simple_mock_summary(self, df=None):
        """Helper to create mock summary with simple data."""
        if df is None:
            df = pd.DataFrame(
                {
                    "channel": ["TV", "Radio", "Social"],
                    "mean": [100.0, 200.0, 300.0],
                    "median": [100.0, 200.0, 300.0],
                    "abs_error_94_lower": [90.0, 190.0, 290.0],
                    "abs_error_94_upper": [110.0, 210.0, 310.0],
                }
            )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.contributions.return_value = df
        mock_summary.data = Mock(custom_dims=[])
        return mock_summary

    def test_contributions_returns_plotly_figure_with_error_bars(self):
        """Test that contributions() returns a Plotly Figure object."""
        # Arrange
        mock_summary = self._create_simple_mock_summary()
        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.contributions()

        # Assert
        assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"

        # Assert that error bars are added
        bar_trace = fig.data[0]
        assert bar_trace.error_y is not None, (
            "Bar chart should have error_y for HDI upper bounds"
        )
        assert hasattr(bar_trace.error_y, "array"), (
            "error_y should have array attribute"
        )

    def test_contributions_with_custom_hdi_prob(self):
        """Test that custom hdi_prob is passed to summary factory."""
        # Arrange
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.contributions.return_value = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "channel": ["TV"] * 3,
                "mean": [100.0, 200.0, 300.0],
                "median": [100.0, 200.0, 300.0],
                "abs_error_80_lower": [90.0, 190.0, 290.0],
                "abs_error_80_upper": [110.0, 210.0, 310.0],
            }
        )
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        factory.contributions(hdi_prob=0.80, color="date")

        # Assert
        mock_summary.contributions.assert_called_once()
        call_kwargs = mock_summary.contributions.call_args[1]
        assert call_kwargs["hdi_probs"] == [0.80], (
            f"Expected hdi_probs=[0.80], got {call_kwargs['hdi_probs']}"
        )

    def test_contributions_accepts_polars_dataframe(self):
        """Test that contributions() handles Polars DataFrames via Narwhals."""
        # Arrange
        df_polars = pl.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "channel": ["TV", "Radio", "Social"],
                "mean": [100.0, 200.0, 300.0],
                "median": [100.0, 200.0, 300.0],
                "abs_error_94_lower": [90.0, 190.0, 290.0],
                "abs_error_94_upper": [110.0, 210.0, 310.0],
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.contributions.return_value = df_polars
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.contributions(color="date")

        # Assert
        assert isinstance(fig, go.Figure), "Should return Figure for Polars input"


class TestMMMPlotlyFactoryHDI:
    """Tests for HDI error bar handling."""

    def test_contributions_converts_absolute_to_relative_errors(self):
        """Test that absolute HDI bounds are converted to relative errors."""
        # Arrange
        df = pd.DataFrame(
            {
                "channel": ["TV", "Radio"],
                "mean": [100.0, 200.0],
                "median": [100.0, 200.0],
                "abs_error_94_lower": [90.0, 190.0],  # 10 below each mean
                "abs_error_94_upper": [115.0, 220.0],  # 15 above TV, 20 above Radio
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.contributions.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.contributions(hdi_prob=0.94)

        # Assert - Check that error bars have correct relative values
        bar_trace = fig.data[0]
        assert bar_trace.error_y is not None, (
            "Bar chart should have error_y for HDI upper bounds"
        )

        # Verify upper errors: absolute_upper - mean
        # TV: 115 - 100 = 15, Radio: 220 - 200 = 20
        np.testing.assert_array_almost_equal(
            bar_trace.error_y.array,
            [15.0, 20.0],
            err_msg="Upper error bars should be [15.0, 20.0]",
        )

        # Verify lower errors: mean - absolute_lower
        # TV: 100 - 90 = 10, Radio: 200 - 190 = 10
        np.testing.assert_array_almost_equal(
            bar_trace.error_y.arrayminus,
            [10.0, 10.0],
            err_msg="Lower error bars should be [10.0, 10.0]",
        )

    def test_contributions_no_error_bars_when_hdi_prob_is_none(self):
        """Test that no error bars are added when hdi_prob=None."""
        # Arrange
        df = pd.DataFrame(
            {
                "channel": ["TV", "Radio"],
                "mean": [100.0, 200.0],
                "median": [100.0, 200.0],
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.contributions.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.contributions(hdi_prob=None)

        # Assert
        bar_trace = fig.data[0]
        # Plotly always creates an error_y object, but array should be None when no errors
        assert bar_trace.error_y.array is None, (
            "Bar chart should not have error bar data when hdi_prob=None"
        )


class TestMMMPlotlyFactoryAutoFaceting:
    """Tests for automatic faceting based on custom dimensions."""

    def test_auto_faceting_disabled_when_auto_facet_false(self):
        """Test that auto-faceting is disabled when auto_facet=False."""
        # Arrange
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4),
                "country": ["US", "US", "UK", "UK"],
                "channel": ["TV", "Radio"] * 2,
                "mean": [100.0, 200.0, 150.0, 250.0],
                "median": [100.0, 200.0, 150.0, 250.0],
                "abs_error_94_lower": [90.0, 190.0, 140.0, 240.0],
                "abs_error_94_upper": [110.0, 210.0, 160.0, 260.0],
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.contributions.return_value = df
        mock_summary.data = Mock(custom_dims=["country"])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.contributions(color="date")

        # Assert - No faceting means single subplot
        # Plotly creates subplots when faceting is used
        assert fig.layout.xaxis.domain is not None, (
            "Figure should not have faceted layout when auto_facet=False"
        )

    def test_auto_faceting_applies_row_and_col_for_two_dimensions(self):
        """Test that auto-faceting uses facet_row and facet_col for multiple dimensions."""
        # Arrange
        df = pd.DataFrame(
            {
                "country": ["US"] * 4 + ["UK"] * 4,
                "region": ["North", "South"] * 4,
                "channel": ["TV", "Radio"] * 4,
                "mean": list(range(100, 900, 100)),
                "median": list(range(100, 900, 100)),
                "abs_error_94_lower": list(range(90, 890, 100)),
                "abs_error_94_upper": list(range(110, 910, 100)),
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.contributions.return_value = df
        mock_summary.data = Mock(custom_dims=["country", "region"])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=True)

        # Act
        fig = factory.contributions()

        # Assert - 2x2 grid should have 4 subplot annotations
        assert len(fig.layout.annotations) >= 4, (
            "2D faceted figure should have at least 4 subplot annotations"
        )

    def test_contributions_component_control(self):
        """Test that component='control' plots control contributions."""
        df = pd.DataFrame(
            {
                "control": ["GDP", "Seasonality"],
                "mean": [100.0, 200.0],
                "median": [100.0, 200.0],
                "abs_error_94_lower": [90.0, 190.0],
                "abs_error_94_upper": [110.0, 210.0],
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.contributions.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        factory.contributions(component="control")

        # Assert
        call_kwargs = mock_summary.contributions.call_args[1]
        assert call_kwargs["component"] == "control"


class TestMMMPlotInteractiveProperty:
    """Tests for mmm.plot_interactive property integration."""

    def test_plot_interactive_property_exists(self, simple_fitted_mmm):
        """Test that plot_interactive property is accessible."""
        # Act
        plot_interactive = simple_fitted_mmm.plot_interactive

        # Assert
        assert plot_interactive is not None, (
            "plot_interactive property should not be None"
        )

    def test_plot_interactive_returns_factory_instance(self, simple_fitted_mmm):
        """Test that plot_interactive returns MMMPlotlyFactory."""
        # Act
        factory = simple_fitted_mmm.plot_interactive

        # Assert
        assert isinstance(factory, MMMPlotlyFactory), (
            f"Expected MMMPlotlyFactory, got {type(factory)}"
        )

    def test_plot_interactive_requires_fitted_model(self):
        """Test that plot_interactive raises error if model not fitted."""
        # Arrange - Create unfitted model
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        # Act & Assert
        with pytest.raises(ValueError, match="idata does not exist"):
            _ = mmm.plot_interactive

    def test_plot_interactive_has_summary_access(self, simple_fitted_mmm):
        """Test that factory has summary attribute."""
        # Act
        factory = simple_fitted_mmm.plot_interactive

        # Assert
        assert hasattr(factory, "summary"), "Factory should have summary attribute"
        assert factory.summary is not None, "Factory summary should not be None"


# ============================================================================
# Phase 2 Tests: Posterior Predictive Plotting
# ============================================================================


class TestMMMPlotlyFactoryPosteriorPredictive:
    """Tests for MMMPlotlyFactory.posterior_predictive() method."""

    def test_posterior_predictive_returns_figure(self):
        """Test that posterior_predictive() returns a Plotly Figure."""
        # Arrange
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "mean": np.random.randn(10) * 100 + 1000,
                "median": np.random.randn(10) * 100 + 1000,
                "observed": np.random.randn(10) * 100 + 1000,
                "abs_error_94_lower": np.random.randn(10) * 50 + 950,
                "abs_error_94_upper": np.random.randn(10) * 50 + 1050,
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.posterior_predictive.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.posterior_predictive()

        # Assert
        assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"

    def test_posterior_predictive_has_two_traces(self):
        """Test that figure contains predicted and observed traces."""
        # Arrange
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "mean": [1000.0] * 10,
                "median": [1000.0] * 10,
                "observed": [900.0] * 10,
                "abs_error_94_lower": [950.0] * 10,
                "abs_error_94_upper": [1050.0] * 10,
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.posterior_predictive.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.posterior_predictive()

        # Assert
        # Should have at least 2 traces: Predicted line, Observed line
        assert len(fig.data) >= 2, (
            f"Expected at least 2 traces (Predicted + Observed), got {len(fig.data)}"
        )

        # Check trace names
        trace_names = [trace.name for trace in fig.data]
        assert "Predicted" in trace_names, "Should have 'Predicted' trace"
        assert "Observed" in trace_names, "Should have 'Observed' trace"

    def test_posterior_predictive_adds_hdi_band(self):
        """Test that HDI band is added as a filled trace."""
        # Arrange
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "mean": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
                "median": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
                "observed": [950.0, 1050.0, 1150.0, 1250.0, 1350.0],
                "abs_error_94_lower": [900.0, 1000.0, 1100.0, 1200.0, 1300.0],
                "abs_error_94_upper": [1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.posterior_predictive.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.posterior_predictive(hdi_prob=0.94)

        # Assert
        # Find HDI trace (filled area with 'toself')
        hdi_traces = [t for t in fig.data if t.fill == "toself"]
        assert len(hdi_traces) >= 1, "Should have at least one HDI band trace"

        hdi_trace = hdi_traces[0]
        assert "94" in hdi_trace.name or "HDI" in hdi_trace.name, (
            f"HDI trace name should mention HDI or probability, got: {hdi_trace.name}"
        )

    def test_posterior_predictive_no_hdi_when_none(self):
        """Test that no HDI band is added when hdi_prob=None."""
        # Arrange
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "mean": [1000.0] * 5,
                "median": [1000.0] * 5,
                "observed": [950.0] * 5,
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.posterior_predictive.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.posterior_predictive(hdi_prob=None)

        # Assert
        # Should only have line traces, no filled areas
        filled_traces = [t for t in fig.data if t.fill == "toself"]
        assert len(filled_traces) == 0, (
            f"Should have no HDI bands when hdi_prob=None, got {len(filled_traces)}"
        )

    def test_posterior_predictive_with_custom_dimensions(self):
        """Test posterior predictive with faceting by country."""
        # Arrange
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4).repeat(2),
                "country": ["US", "UK"] * 4,
                "mean": np.random.randn(8) * 100 + 1000,
                "median": np.random.randn(8) * 100 + 1000,
                "observed": np.random.randn(8) * 100 + 1000,
                "abs_error_94_lower": np.random.randn(8) * 50 + 950,
                "abs_error_94_upper": np.random.randn(8) * 50 + 1050,
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.posterior_predictive.return_value = df
        mock_summary.data = Mock(custom_dims=["country"])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=True)

        # Act
        fig = factory.posterior_predictive()

        # Assert
        # Should have faceted layout (annotations for subplot titles)
        assert len(fig.layout.annotations) >= 2, (
            f"Should have annotations for 2 countries, got {len(fig.layout.annotations)}"
        )


# ============================================================================
# Phase 2 Tests: ROAS Plotting
# ============================================================================


class TestMMMPlotlyFactoryROAS:
    """Tests for MMMPlotlyFactory.roas() method."""

    def test_roas_returns_figure(self):
        """Test that roas() returns a Plotly Figure."""
        # Arrange
        df = pd.DataFrame(
            {
                "channel": ["TV", "Radio", "Social"],
                "mean": [2.5, 3.0, 1.8],
                "median": [2.4, 3.1, 1.9],
                "abs_error_94_lower": [2.0, 2.5, 1.5],
                "abs_error_94_upper": [3.0, 3.5, 2.1],
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.roas.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.roas()

        # Assert
        assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"

    def test_roas_has_error_bars(self):
        """Test that ROAS plot includes HDI error bars."""
        # Arrange
        df = pd.DataFrame(
            {
                "channel": ["TV", "Radio"],
                "mean": [2.5, 3.0],
                "median": [2.4, 3.1],
                "abs_error_94_lower": [2.0, 2.5],
                "abs_error_94_upper": [3.0, 3.5],
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.roas.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.roas(hdi_prob=0.94)

        # Assert
        bar_trace = fig.data[0]
        assert bar_trace.error_y is not None, (
            "Bar chart should have error_y for HDI bounds"
        )
        assert bar_trace.error_y.array is not None, (
            "error_y should have array attribute"
        )

    def test_roas_accepts_polars(self):
        """Test that roas() handles Polars DataFrames."""
        # Arrange
        df_polars = pl.DataFrame(
            {
                "channel": ["TV", "Radio", "Social"],
                "mean": [2.5, 3.0, 1.8],
                "median": [2.4, 3.1, 1.9],
                "abs_error_94_lower": [2.0, 2.5, 1.5],
                "abs_error_94_upper": [3.0, 3.5, 2.1],
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.roas.return_value = df_polars
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.roas()

        # Assert
        assert isinstance(fig, go.Figure), "Should return Figure for Polars input"


# ============================================================================
# Phase 2 Tests: Saturation Curves Plotting
# ============================================================================


class TestMMMPlotlyFactorySaturationCurves:
    """Tests for MMMPlotlyFactory.saturation_curves() method."""

    def test_saturation_curves_returns_figure(self):
        """Test that saturation_curves() returns a Plotly Figure."""
        # Arrange
        df = pd.DataFrame(
            {
                "x": np.linspace(0, 1, 50).tolist() * 2,
                "channel": ["TV"] * 50 + ["Radio"] * 50,
                "mean": np.random.rand(100),
                "median": np.random.rand(100),
                "abs_error_94_lower": np.random.rand(100) * 0.8,
                "abs_error_94_upper": np.random.rand(100) * 1.2,
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.saturation_curves.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.saturation_curves()

        # Assert
        assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"

    def test_saturation_curves_line_plots_per_channel(self):
        """Test that saturation curves shows one line per channel."""
        # Arrange
        x_vals = np.linspace(0, 1, 20)
        df = pd.DataFrame(
            {
                "x": np.tile(x_vals, 2),
                "channel": ["TV"] * 20 + ["Radio"] * 20,
                "mean": np.concatenate([x_vals * 0.8, x_vals * 0.6]),
                "median": np.concatenate([x_vals * 0.8, x_vals * 0.6]),
                "abs_error_94_lower": np.concatenate([x_vals * 0.7, x_vals * 0.5]),
                "abs_error_94_upper": np.concatenate([x_vals * 0.9, x_vals * 0.7]),
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.saturation_curves.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.saturation_curves()

        # Assert
        # Should have at least 2 line traces (one per channel)
        line_traces = [t for t in fig.data if t.mode == "lines" and t.fill != "toself"]
        assert len(line_traces) >= 2, (
            f"Should have at least 2 line traces (TV, Radio), got {len(line_traces)}"
        )

    def test_saturation_curves_hdi_bands(self):
        """Test that HDI bands are added for each channel."""
        # Arrange
        x_vals = np.linspace(0, 1, 10)
        df = pd.DataFrame(
            {
                "x": np.tile(x_vals, 2),
                "channel": ["TV"] * 10 + ["Radio"] * 10,
                "mean": np.concatenate([x_vals * 0.8, x_vals * 0.6]),
                "median": np.concatenate([x_vals * 0.8, x_vals * 0.6]),
                "abs_error_94_lower": np.concatenate([x_vals * 0.7, x_vals * 0.5]),
                "abs_error_94_upper": np.concatenate([x_vals * 0.9, x_vals * 0.7]),
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.saturation_curves.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.saturation_curves(hdi_prob=0.94)

        # Assert
        # Should have filled traces for HDI bands
        hdi_traces = [t for t in fig.data if t.fill == "toself"]
        assert len(hdi_traces) >= 2, (
            f"Should have HDI bands for both channels, got {len(hdi_traces)}"
        )

    def test_saturation_curves_with_custom_dimensions(self):
        """Test saturation curves with faceting by country."""
        # Arrange
        x_vals = np.linspace(0, 1, 10)
        df = pd.DataFrame(
            {
                "x": np.tile(x_vals, 4),
                "channel": (["TV"] * 10 + ["Radio"] * 10) * 2,
                "country": ["US"] * 20 + ["UK"] * 20,
                "mean": np.random.rand(40),
                "median": np.random.rand(40),
                "abs_error_94_lower": np.random.rand(40) * 0.8,
                "abs_error_94_upper": np.random.rand(40) * 1.2,
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.saturation_curves.return_value = df
        mock_summary.data = Mock(custom_dims=["country"])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=True)

        # Act
        fig = factory.saturation_curves()

        # Assert
        # Should have faceted layout
        assert len(fig.layout.annotations) >= 2, (
            f"Should have annotations for 2 countries, got {len(fig.layout.annotations)}"
        )


# ============================================================================
# Phase 2 Tests: Adstock Curves Plotting
# ============================================================================


class TestMMMPlotlyFactoryAdstockCurves:
    """Tests for MMMPlotlyFactory.adstock_curves() method."""

    def test_adstock_curves_returns_figure(self):
        """Test that adstock_curves() returns a Plotly Figure."""
        # Arrange
        df = pd.DataFrame(
            {
                "time since exposure": np.tile(np.arange(10), 2),
                "channel": ["TV"] * 10 + ["Radio"] * 10,
                "mean": np.random.rand(20),
                "median": np.random.rand(20),
                "abs_error_94_lower": np.random.rand(20) * 0.8,
                "abs_error_94_upper": np.random.rand(20) * 1.2,
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.adstock_curves.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.adstock_curves()

        # Assert
        assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"

    def test_adstock_curves_hdi_bands(self):
        """Test that HDI bands are added for adstock curves."""
        # Arrange
        lags = np.arange(5)
        df = pd.DataFrame(
            {
                "time since exposure": np.tile(lags, 2),
                "channel": ["TV"] * 5 + ["Radio"] * 5,
                "mean": np.random.rand(10),
                "median": np.random.rand(10),
                "abs_error_94_lower": np.random.rand(10) * 0.8,
                "abs_error_94_upper": np.random.rand(10) * 1.2,
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.adstock_curves.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary, auto_facet=False)

        # Act
        fig = factory.adstock_curves(hdi_prob=0.94)

        # Assert
        hdi_traces = [t for t in fig.data if t.fill == "toself"]
        assert len(hdi_traces) >= 2, (
            f"Should have HDI bands for both channels, got {len(hdi_traces)}"
        )
