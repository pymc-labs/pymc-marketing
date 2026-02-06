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
