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
import xarray as xr

from pymc_marketing.mmm.plot_interactive import MMMPlotlyFactory
from pymc_marketing.mmm.summary import MMMSummaryFactory


def _create_saturation_mock_summary(
    df: pd.DataFrame,
    custom_dims: list[str],
    channels: list[str] | None = None,
) -> Mock:
    """Create mock summary for saturation_curves tests with proper channel scale.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to return from saturation_curves()
    custom_dims : list[str]
        Custom dimensions (e.g., ["geo"], ["geo", "brand"])
    channels : list[str], optional
        Channel names. If None, extracted from df["channel"].unique()

    Returns
    -------
    Mock
        Properly configured mock summary
    """
    mock_summary = Mock(spec=MMMSummaryFactory)
    mock_summary.saturation_curves.return_value = df
    mock_summary.data = Mock(custom_dims=custom_dims)

    # Extract channels from data if not provided
    if channels is None:
        channels = df["channel"].unique().tolist()

    # Build channel_scale DataArray with correct dimensions
    if len(custom_dims) == 0:
        # Simple case: just channels
        channel_scale = xr.DataArray(
            [1.0] * len(channels),
            dims=["channel"],
            coords={"channel": channels},
        )
    elif len(custom_dims) == 1:
        # One custom dim: (channel, custom_dim)
        coords = df[custom_dims[0]].unique().tolist()
        channel_scale = xr.DataArray(
            np.ones((len(channels), len(coords))),
            dims=["channel", custom_dims[0]],
            coords={"channel": channels, custom_dims[0]: coords},
        )
    else:
        # Two custom dims: (channel, dim1, dim2)
        coords1 = df[custom_dims[0]].unique().tolist()
        coords2 = df[custom_dims[1]].unique().tolist()
        channel_scale = xr.DataArray(
            np.ones((len(channels), len(coords1), len(coords2))),
            dims=["channel", custom_dims[0], custom_dims[1]],
            coords={
                "channel": channels,
                custom_dims[0]: coords1,
                custom_dims[1]: coords2,
            },
        )

    mock_summary.data.get_channel_scale.return_value = channel_scale
    return mock_summary


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
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.contributions(auto_facet=False)

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

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        factory.contributions(hdi_prob=0.80, color="date", auto_facet=False)

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

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.contributions(color="date", auto_facet=False)

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

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.contributions(hdi_prob=0.94, auto_facet=False)

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

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.contributions(hdi_prob=None, auto_facet=False)

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

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.contributions(color="date", auto_facet=False)

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

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.contributions(auto_facet=True)

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

        factory = MMMPlotlyFactory(summary=mock_summary)

        factory.contributions(component="control", auto_facet=False)

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

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.posterior_predictive(auto_facet=False)

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

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.posterior_predictive(auto_facet=False)

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

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.posterior_predictive(hdi_prob=0.94, auto_facet=False)

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

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.posterior_predictive(hdi_prob=None, auto_facet=False)

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

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.posterior_predictive(auto_facet=True)

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

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.roas(auto_facet=False)

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

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.roas(hdi_prob=0.94, auto_facet=False)

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

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.roas(auto_facet=False)

        # Assert
        assert isinstance(fig, go.Figure), "Should return Figure for Polars input"

    def test_roas_handles_nan_values(self):
        """Test that roas() handles NaN values in data without crashing."""
        # Arrange - ROAS data with NaN values (e.g., channel with zero spend)
        df = pd.DataFrame(
            {
                "channel": ["TV", "Radio", "Social"],
                "mean": [2.5, np.nan, 1.8],  # NaN for Radio (zero spend)
                "median": [2.4, np.nan, 1.9],
                "abs_error_94_lower": [2.0, np.nan, 1.5],
                "abs_error_94_upper": [3.0, np.nan, 2.1],
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.roas.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act & Assert - Should not raise IntCastingNaNError
        fig = factory.roas(hdi_prob=0.94, auto_facet=False)

        # Verify figure is returned
        assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"


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
        mock_summary = _create_saturation_mock_summary(df, custom_dims=[])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(auto_facet=False)

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
        mock_summary = _create_saturation_mock_summary(df, custom_dims=[])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(auto_facet=False)

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
        mock_summary = _create_saturation_mock_summary(df, custom_dims=[])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(hdi_prob=0.94, auto_facet=False)

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
        mock_summary = _create_saturation_mock_summary(df, custom_dims=["country"])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(auto_facet=True)

        # Assert
        # Should have faceted layout
        assert len(fig.layout.annotations) >= 2, (
            f"Should have annotations for 2 countries, got {len(fig.layout.annotations)}"
        )

    def test_saturation_curves_no_auto_facet_with_custom_dim_has_correct_line_count(
        self,
    ):
        """Test that saturation_curves(auto_facet=False) plots all curves when custom_dim exists.

        When auto_facet=False and data has custom dimensions (e.g., country),
        the figure should contain one line per (channel, custom_dim_coord) combination.
        For 2 channels and 2 countries, there should be 4 lines (excluding HDI bands).
        """
        # Arrange
        x_vals = np.linspace(0, 1, 10)
        n_channels = 2
        n_countries = 2
        n_points = len(x_vals)

        # Create data with 2 channels (TV, Radio) x 2 countries (US, UK) = 4 curves
        mean_values = np.random.rand(n_channels * n_countries * n_points)
        df = pd.DataFrame(
            {
                "x": np.tile(x_vals, n_channels * n_countries),
                "channel": (["TV"] * n_points + ["Radio"] * n_points) * n_countries,
                "country": ["US"] * (n_points * n_channels)
                + ["UK"] * (n_points * n_channels),
                "mean": mean_values,
                "median": mean_values,
                "abs_error_94_lower": mean_values * 0.8,
                "abs_error_94_upper": mean_values * 1.2,
            }
        )
        mock_summary = _create_saturation_mock_summary(df, custom_dims=["country"])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(auto_facet=False, hdi_prob=None)

        # Assert
        # Count line traces (exclude HDI bands which have fill="toself")
        line_traces = [
            t
            for t in fig.data
            if getattr(t, "mode", None) == "lines" and t.fill is None
        ]
        expected_lines = n_channels * n_countries  # 2 channels * 2 countries = 4 lines
        assert len(line_traces) == expected_lines, (
            f"Expected {expected_lines} line traces for {n_channels} channels x "
            f"{n_countries} countries, got {len(line_traces)}"
        )

    def test_saturation_curves_facet_col_with_two_custom_dims(self):
        """Test saturation_curves with facet_col='brand' and two custom dimensions (geo, brand).

        When facet_col='brand' with hdi_prob=None and data has two custom dimensions
        (geo, brand), the figure should:
        1. Create a subplot for each brand (facet columns)
        2. Within each subplot, have a curve for each (channel, geo) combination

        For 2 channels, 2 geos, and 2 brands, there should be:
        - 2 subplots (one per brand)
        - 4 curves per subplot (2 channels x 2 geos)
        - 8 total line traces
        """
        # Arrange
        x_vals = np.linspace(0, 1, 10)
        n_channels = 2
        n_geos = 2
        n_brands = 2
        n_points = len(x_vals)
        total_combinations = n_channels * n_geos * n_brands

        # Create data with all combinations
        # Structure: for each brand, for each geo, for each channel
        channels = []
        geos = []
        brands = []
        x_all = []
        for brand in ["BrandA", "BrandB"]:
            for geo in ["US", "UK"]:
                for channel in ["TV", "Radio"]:
                    channels.extend([channel] * n_points)
                    geos.extend([geo] * n_points)
                    brands.extend([brand] * n_points)
                    x_all.extend(x_vals.tolist())

        mean_values = np.random.rand(total_combinations * n_points)
        df = pd.DataFrame(
            {
                "x": x_all,
                "channel": channels,
                "geo": geos,
                "brand": brands,
                "mean": mean_values,
                "median": mean_values,
            }
        )
        mock_summary = _create_saturation_mock_summary(df, custom_dims=["geo", "brand"])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(
            facet_col="brand", hdi_prob=None, auto_facet=False
        )

        # Assert - Check faceted layout (should have annotations for brand values)
        assert len(fig.layout.annotations) >= n_brands, (
            f"Should have at least {n_brands} subplot annotations for brands, "
            f"got {len(fig.layout.annotations)}"
        )

        # Verify the annotation texts contain the brand names
        annotation_texts = [ann.text for ann in fig.layout.annotations]
        assert any("BrandA" in text for text in annotation_texts), (
            f"Expected 'BrandA' in subplot annotations, got: {annotation_texts}"
        )
        assert any("BrandB" in text for text in annotation_texts), (
            f"Expected 'BrandB' in subplot annotations, got: {annotation_texts}"
        )

        # Count line traces (exclude HDI bands which have fill="toself")
        # Note: px.line creates traces with mode="lines" or "lines+markers"
        line_traces = [
            t
            for t in fig.data
            if getattr(t, "mode", None) in ("lines", "lines+markers") and t.fill is None
        ]

        # Total curves should be n_channels * n_geos * n_brands = 2 * 2 * 2 = 8
        # Each (channel, geo) combination appears in each brand facet
        expected_total_lines = n_channels * n_geos * n_brands
        assert len(line_traces) == expected_total_lines, (
            f"Expected {expected_total_lines} total line traces for "
            f"{n_channels} channels x {n_geos} geos x {n_brands} brands, "
            f"got {len(line_traces)}"
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

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.adstock_curves(auto_facet=False)

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

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.adstock_curves(hdi_prob=0.94, auto_facet=False)

        # Assert
        hdi_traces = [t for t in fig.data if t.fill == "toself"]
        assert len(hdi_traces) >= 2, (
            f"Should have HDI bands for both channels, got {len(hdi_traces)}"
        )


# ============================================================================
# Phase 2 Tests: Saturation Curves Faceting with Custom Dimensions
# ============================================================================


def _count_line_traces(fig: go.Figure) -> int:
    """Count line traces excluding HDI bands."""
    return len(
        [
            t
            for t in fig.data
            if getattr(t, "mode", None) in ("lines", "lines+markers") and t.fill is None
        ]
    )


def _count_subplots(fig: go.Figure) -> int:
    """Count the number of subplots from layout annotations."""
    if not fig.layout.annotations:
        return 1
    return len(fig.layout.annotations)


def _get_subplot_dimensions(fig: go.Figure) -> tuple[int, int]:
    """Get the (rows, cols) dimensions of a faceted figure.

    Returns (1, 1) for non-faceted figures.
    """
    # Count annotations which correspond to subplot titles
    n_subplots = max(1, len(fig.layout.annotations))

    # Look at the annotation positions to determine grid dimensions
    if n_subplots == 1:
        return (1, 1)

    # Check if it's row-based or col-based by looking at annotation y positions
    if fig.layout.annotations:
        y_positions = sorted(set(ann.y for ann in fig.layout.annotations), reverse=True)
        x_positions = sorted(set(ann.x for ann in fig.layout.annotations))
        rows = len(y_positions)
        cols = len(x_positions)
        return (rows, cols)

    return (1, n_subplots)


def _create_saturation_df_one_custom_dim(
    n_channels: int = 2,
    n_coords: int = 2,
    n_points: int = 10,
    custom_dim: str = "geo",
) -> pd.DataFrame:
    """Create saturation curves data with 1 custom dimension.

    Args:
        n_channels: Number of channels (default 2)
        n_coords: Number of coordinates in the custom dimension (default 2)
        n_points: Number of x points per curve
        custom_dim: Name of the custom dimension

    Returns:
        DataFrame with columns: x, channel, {custom_dim}, mean, median
    """
    x_vals = np.linspace(0, 1, n_points)
    channels = ["TV", "Radio"][:n_channels]
    coords = ["North", "South"][:n_coords]

    rows = []
    for coord in coords:
        for channel in channels:
            for x in x_vals:
                rows.append(
                    {
                        "x": x,
                        "channel": channel,
                        custom_dim: coord,
                        "mean": np.random.rand(),
                        "median": np.random.rand(),
                    }
                )

    return pd.DataFrame(rows)


def _create_saturation_df_two_custom_dims(
    n_channels: int = 2,
    n_coords_dim1: int = 2,
    n_coords_dim2: int = 2,
    n_points: int = 10,
    custom_dim1: str = "geo",
    custom_dim2: str = "brand",
) -> pd.DataFrame:
    """Create saturation curves data with 2 custom dimensions.

    Args:
        n_channels: Number of channels (default 2)
        n_coords_dim1: Number of coordinates in first dimension (default 2)
        n_coords_dim2: Number of coordinates in second dimension (default 2)
        n_points: Number of x points per curve
        custom_dim1: Name of first custom dimension
        custom_dim2: Name of second custom dimension

    Returns:
        DataFrame with columns: x, channel, {custom_dim1}, {custom_dim2}, mean, median
    """
    x_vals = np.linspace(0, 1, n_points)
    channels = ["TV", "Radio"][:n_channels]
    coords1 = ["North", "South"][:n_coords_dim1]
    coords2 = ["BrandA", "BrandB"][:n_coords_dim2]

    rows = []
    for coord1 in coords1:
        for coord2 in coords2:
            for channel in channels:
                for x in x_vals:
                    rows.append(
                        {
                            "x": x,
                            "channel": channel,
                            custom_dim1: coord1,
                            custom_dim2: coord2,
                            "mean": np.random.rand(),
                            "median": np.random.rand(),
                        }
                    )

    return pd.DataFrame(rows)


class TestSaturationCurvesOneCustomDim:
    """Tests for saturation_curves with 1 custom dimension (2 coordinates).

    With 2 channels and 2 coordinates, there should always be 4 lines total.
    """

    def test_auto_facet_creates_two_subplots_with_two_lines_each(self):
        """saturation_curves(hdi_prob=None) → 2 subplots, 2 lines per subplot."""
        # Arrange
        df = _create_saturation_df_one_custom_dim(
            n_channels=2, n_coords=2, custom_dim="geo"
        )
        mock_summary = _create_saturation_mock_summary(df, custom_dims=["geo"])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(hdi_prob=None)

        # Assert
        n_subplots = _count_subplots(fig)
        n_lines = _count_line_traces(fig)

        assert n_subplots == 2, (
            f"Expected 2 subplots (one per geo coordinate), got {n_subplots}"
        )
        assert n_lines == 4, (
            f"Expected 4 total lines (2 channels × 2 geos), got {n_lines}"
        )

    def test_auto_facet_false_creates_one_plot_with_four_lines(self):
        """saturation_curves(hdi_prob=None, auto_facet=False) → 1 plot, 4 lines."""
        # Arrange
        df = _create_saturation_df_one_custom_dim(
            n_channels=2, n_coords=2, custom_dim="geo"
        )
        mock_summary = _create_saturation_mock_summary(df, custom_dims=["geo"])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(hdi_prob=None, auto_facet=False)

        # Assert
        n_subplots = _count_subplots(fig)
        n_lines = _count_line_traces(fig)

        assert n_subplots == 1, (
            f"Expected 1 subplot when auto_facet=False, got {n_subplots}"
        )
        assert n_lines == 4, (
            f"Expected 4 lines (2 channels × 2 geos) on single plot, got {n_lines}"
        )

    def test_facet_col_creates_two_column_subplots_with_two_lines_each(self):
        """saturation_curves(facet_col='geo', hdi_prob=None, auto_facet=False) →
        2 subplots (1 row, 2 columns), 2 lines per subplot.
        """
        # Arrange
        df = _create_saturation_df_one_custom_dim(
            n_channels=2, n_coords=2, custom_dim="geo"
        )
        mock_summary = _create_saturation_mock_summary(df, custom_dims=["geo"])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(
            facet_col="geo", hdi_prob=None, auto_facet=False
        )

        # Assert
        rows, cols = _get_subplot_dimensions(fig)
        n_lines = _count_line_traces(fig)

        assert rows == 1, f"Expected 1 row with facet_col, got {rows}"
        assert cols == 2, f"Expected 2 columns (one per geo), got {cols}"
        assert n_lines == 4, f"Expected 4 total lines (2 per subplot), got {n_lines}"

    def test_facet_col_with_auto_facet_true_same_as_auto_facet_false(self):
        """saturation_curves(facet_col='geo', hdi_prob=None, auto_facet=True) →
        same as with auto_facet=False because manual faceting takes precedence.
        """
        # Arrange
        df = _create_saturation_df_one_custom_dim(
            n_channels=2, n_coords=2, custom_dim="geo"
        )
        mock_summary = _create_saturation_mock_summary(df, custom_dims=["geo"])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(facet_col="geo", hdi_prob=None, auto_facet=True)

        # Assert
        rows, cols = _get_subplot_dimensions(fig)
        n_lines = _count_line_traces(fig)

        assert rows == 1, f"Expected 1 row with facet_col, got {rows}"
        assert cols == 2, f"Expected 2 columns (one per geo), got {cols}"
        assert n_lines == 4, f"Expected 4 total lines, got {n_lines}"

    def test_facet_row_creates_two_row_subplots_with_two_lines_each(self):
        """saturation_curves(facet_row='geo', hdi_prob=None, auto_facet=False) →
        2 subplots (2 rows, 1 column), 2 lines per subplot.
        """
        # Arrange
        df = _create_saturation_df_one_custom_dim(
            n_channels=2, n_coords=2, custom_dim="geo"
        )
        mock_summary = _create_saturation_mock_summary(df, custom_dims=["geo"])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(
            facet_row="geo", hdi_prob=None, auto_facet=False
        )

        # Assert
        rows, cols = _get_subplot_dimensions(fig)
        n_lines = _count_line_traces(fig)

        assert rows == 2, f"Expected 2 rows (one per geo), got {rows}"
        assert cols == 1, f"Expected 1 column with facet_row, got {cols}"
        assert n_lines == 4, f"Expected 4 total lines (2 per subplot), got {n_lines}"


class TestSaturationCurvesTwoCustomDims:
    """Tests for saturation_curves with 2 custom dimensions (2 coordinates each).

    With 2 channels × 2 coords_dim1 × 2 coords_dim2 = 8 lines total.
    """

    def test_auto_facet_creates_four_subplots_with_two_lines_each(self):
        """saturation_curves(hdi_prob=None) → 4 subplots (2×2), 2 lines per subplot."""
        # Arrange
        df = _create_saturation_df_two_custom_dims(
            n_channels=2,
            n_coords_dim1=2,
            n_coords_dim2=2,
            custom_dim1="geo",
            custom_dim2="brand",
        )
        mock_summary = _create_saturation_mock_summary(df, custom_dims=["geo", "brand"])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(hdi_prob=None)

        # Assert
        n_subplots = _count_subplots(fig)
        n_lines = _count_line_traces(fig)

        assert n_subplots == 4, (
            f"Expected 4 subplots (2 geo × 2 brand), got {n_subplots}"
        )
        assert n_lines == 8, (
            f"Expected 8 total lines (2 channels × 2 geo × 2 brand), got {n_lines}"
        )

    def test_auto_facet_false_raises_error_for_two_dims(self):
        """saturation_curves(hdi_prob=None, auto_facet=False) → should raise error.

        It's impossible to represent all 8 combinations (2 channels × 2 geo × 2 brand)
        on a single plot in a meaningful way without faceting.
        """
        # Arrange
        df = _create_saturation_df_two_custom_dims(
            n_channels=2,
            n_coords_dim1=2,
            n_coords_dim2=2,
            custom_dim1="geo",
            custom_dim2="brand",
        )
        mock_summary = _create_saturation_mock_summary(df, custom_dims=["geo", "brand"])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act & Assert - Should raise ValueError because too many combinations
        with pytest.raises(ValueError, match="Too many custom dimensions"):
            factory.saturation_curves(hdi_prob=None, auto_facet=False)

    def test_facet_col_geo_creates_two_subplots_with_four_lines_each(self):
        """saturation_curves(facet_col='geo', hdi_prob=None, auto_facet=False) →
        2 subplots (1 row, 2 columns), 4 lines per subplot.

        Each subplot shows 2 channels × 2 brand = 4 lines.
        """
        # Arrange
        df = _create_saturation_df_two_custom_dims(
            n_channels=2,
            n_coords_dim1=2,
            n_coords_dim2=2,
            custom_dim1="geo",
            custom_dim2="brand",
        )
        mock_summary = _create_saturation_mock_summary(df, custom_dims=["geo", "brand"])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(
            facet_col="geo", hdi_prob=None, auto_facet=False
        )

        # Assert
        rows, cols = _get_subplot_dimensions(fig)
        n_lines = _count_line_traces(fig)

        assert rows == 1, f"Expected 1 row with facet_col, got {rows}"
        assert cols == 2, f"Expected 2 columns (one per geo), got {cols}"
        assert n_lines == 8, (
            f"Expected 8 total lines (4 per subplot: 2 channels × 2 brand), got {n_lines}"
        )

    def test_facet_col_with_auto_facet_true_ignored(self):
        """saturation_curves(facet_col='geo', hdi_prob=None, auto_facet=True) →
        same as auto_facet=False because manual facet_col takes precedence.
        """
        # Arrange
        df = _create_saturation_df_two_custom_dims(
            n_channels=2,
            n_coords_dim1=2,
            n_coords_dim2=2,
            custom_dim1="geo",
            custom_dim2="brand",
        )
        mock_summary = _create_saturation_mock_summary(df, custom_dims=["geo", "brand"])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(facet_col="geo", hdi_prob=None, auto_facet=True)

        # Assert
        rows, cols = _get_subplot_dimensions(fig)
        n_lines = _count_line_traces(fig)

        assert rows == 1, f"Expected 1 row, got {rows}"
        assert cols == 2, f"Expected 2 columns (one per geo), got {cols}"
        assert n_lines == 8, f"Expected 8 total lines, got {n_lines}"

    def test_facet_row_geo_creates_two_row_subplots_with_four_lines_each(self):
        """saturation_curves(facet_row='geo', hdi_prob=None, auto_facet=False) →
        2 subplots (2 rows, 1 column), 4 lines per subplot.

        Each subplot shows 2 channels × 2 brand = 4 lines.
        """
        # Arrange
        df = _create_saturation_df_two_custom_dims(
            n_channels=2,
            n_coords_dim1=2,
            n_coords_dim2=2,
            custom_dim1="geo",
            custom_dim2="brand",
        )
        mock_summary = _create_saturation_mock_summary(df, custom_dims=["geo", "brand"])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(
            facet_row="geo", hdi_prob=None, auto_facet=False
        )

        # Assert
        rows, cols = _get_subplot_dimensions(fig)
        n_lines = _count_line_traces(fig)

        assert rows == 2, f"Expected 2 rows (one per geo), got {rows}"
        assert cols == 1, f"Expected 1 column with facet_row, got {cols}"
        assert n_lines == 8, (
            f"Expected 8 total lines (4 per subplot: 2 channels × 2 brand), got {n_lines}"
        )


# ============================================================================
# Tests: Saturation Curves Original Scale
# ============================================================================


class TestSaturationCurvesOriginalScale:
    """Tests for saturation_curves original_scale parameter.

    When original_scale=True (default), the x-axis values should be multiplied
    by the channel scale factors to convert from scaled space to original units.
    """

    def test_original_scale_true_multiplies_x_by_channel_scale(self):
        """saturation_curves(original_scale=True) scales x values by channel scale."""
        # Arrange - Create data with known x values for 2 channels x 2 geos
        x_vals = np.array([0.0, 0.5, 1.0])
        df = pd.DataFrame(
            {
                "x": np.tile(x_vals, 4),  # 2 channels x 2 geos = 4 combinations
                "channel": ["TV"] * 3 + ["Radio"] * 3 + ["TV"] * 3 + ["Radio"] * 3,
                "geo": ["US"] * 6 + ["UK"] * 6,
                "mean": [0.0, 0.4, 0.8] * 4,
                "median": [0.0, 0.4, 0.8] * 4,
            }
        )

        # Create channel scale DataArray with different values per (channel, geo)
        # TV,US=1000  TV,UK=800  Radio,US=500  Radio,UK=400
        channel_scale = xr.DataArray(
            [[1000.0, 800.0], [500.0, 400.0]],
            dims=["channel", "geo"],
            coords={"channel": ["TV", "Radio"], "geo": ["US", "UK"]},
        )

        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.saturation_curves.return_value = df
        mock_summary.data = Mock(custom_dims=["geo"])
        mock_summary.data.get_channel_scale.return_value = channel_scale

        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(
            hdi_prob=None, auto_facet=False, original_scale=True
        )

        # Assert - Check that x values are scaled by the correct per-(channel, geo) scale
        traces = [t for t in fig.data if t.mode == "lines"]

        # Find traces for each (channel, geo) combination
        tv_us_trace = next(
            (t for t in traces if "TV" in t.name and "US" in t.name), None
        )
        radio_us_trace = next(
            (t for t in traces if "Radio" in t.name and "US" in t.name), None
        )
        tv_uk_trace = next(
            (t for t in traces if "TV" in t.name and "UK" in t.name), None
        )
        radio_uk_trace = next(
            (t for t in traces if "Radio" in t.name and "UK" in t.name), None
        )

        assert tv_us_trace is not None, "Should have TV, US trace"
        assert radio_us_trace is not None, "Should have Radio, US trace"
        assert tv_uk_trace is not None, "Should have TV, UK trace"
        assert radio_uk_trace is not None, "Should have Radio, UK trace"

        # Check TV, US x values (0 * 1000, 0.5 * 1000, 1.0 * 1000)
        np.testing.assert_array_almost_equal(
            tv_us_trace.x,
            [0.0, 500.0, 1000.0],
            err_msg="TV, US x values should be scaled by 1000",
        )

        # Check TV, UK x values (0 * 800, 0.5 * 800, 1.0 * 800)
        np.testing.assert_array_almost_equal(
            tv_uk_trace.x,
            [0.0, 400.0, 800.0],
            err_msg="TV, UK x values should be scaled by 800",
        )

        # Check Radio, US x values (0 * 500, 0.5 * 500, 1.0 * 500)
        np.testing.assert_array_almost_equal(
            radio_us_trace.x,
            [0.0, 250.0, 500.0],
            err_msg="Radio, US x values should be scaled by 500",
        )

        # Check Radio, UK x values (0 * 400, 0.5 * 400, 1.0 * 400)
        np.testing.assert_array_almost_equal(
            radio_uk_trace.x,
            [0.0, 200.0, 400.0],
            err_msg="Radio, UK x values should be scaled by 400",
        )
