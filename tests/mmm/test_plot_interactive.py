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

import narwhals as nw
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


def _create_simple_mock_summary(
    df: pd.DataFrame | None = None,
    custom_dims: list[str] | None = None,
    method_name: str = "contributions",
) -> Mock:
    """Create a simple mock summary for non-saturation tests.

    Parameters
    ----------
    df : pd.DataFrame, optional
        DataFrame to return from the specified method. If None, creates default contributions data.
    custom_dims : list[str], optional
        Custom dimensions. Defaults to empty list.
    method_name : str, optional
        Name of the method to mock (e.g., "contributions", "roas"). Defaults to "contributions".

    Returns
    -------
    Mock
        Properly configured mock summary
    """
    if custom_dims is None:
        custom_dims = []

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
    getattr(mock_summary, method_name).return_value = df
    mock_summary.data = Mock(custom_dims=custom_dims)
    return mock_summary


def _count_line_traces(fig: go.Figure) -> int:
    """Count line traces excluding HDI bands."""
    return len(
        [
            t
            for t in fig.data
            if getattr(t, "mode", None) in ("lines", "lines+markers") and t.fill is None
        ]
    )


def _get_subplot_dimensions(fig: go.Figure) -> tuple[int, int]:
    """Get the (rows, cols) dimensions of a faceted figure.

    Returns (1, 1) for non-faceted figures.
    """
    return len(fig._grid_ref), len(fig._grid_ref[0])


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


class TestPlotMethodsReturnFigure:
    """Parametrized tests for basic plot method behavior."""

    @pytest.mark.parametrize(
        "method_name,df_setup",
        [
            (
                "contributions",
                {
                    "date": pd.date_range("2024-01-01", periods=3),
                    "channel": ["TV", "Radio", "Social"],
                    "mean": [100.0, 200.0, 300.0],
                    "median": [100.0, 200.0, 300.0],
                    "abs_error_94_lower": [90.0, 190.0, 290.0],
                    "abs_error_94_upper": [110.0, 210.0, 310.0],
                },
            ),
            (
                "roas",
                {
                    "channel": ["TV", "Radio", "Social"],
                    "mean": [2.5, 3.0, 1.8],
                    "median": [2.4, 3.1, 1.9],
                    "abs_error_94_lower": [2.0, 2.5, 1.5],
                    "abs_error_94_upper": [3.0, 3.5, 2.1],
                },
            ),
        ],
    )
    def test_accepts_polars_dataframe(self, method_name, df_setup):
        """Test that method handles Polars DataFrames via Narwhals."""
        df_polars = pl.DataFrame(df_setup)
        mock_summary = Mock(spec=MMMSummaryFactory)
        getattr(mock_summary, method_name).return_value = df_polars
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary)
        method = getattr(factory, method_name)

        fig = method(color="date" if "date" in df_setup else None, auto_facet=False)

        assert isinstance(fig, go.Figure), f"{method_name} should handle Polars input"


class TestRoasAndContributionsBarCharts:
    """Parametrized tests for roas() and contributions() bar chart methods.

    These methods share similar structure and mock data requirements,
    so they are tested together using parametrize.
    """

    @pytest.fixture
    def bar_chart_df(self):
        """Create standard DataFrame for bar chart tests."""
        return pd.DataFrame(
            {
                "channel": ["TV", "Radio", "Social"],
                "mean": [100.0, 200.0, 300.0],
                "median": [100.0, 200.0, 300.0],
                "abs_error_94_lower": [90.0, 190.0, 290.0],
                "abs_error_94_upper": [110.0, 210.0, 310.0],
            }
        )

    @pytest.fixture
    def asymmetric_error_df(self):
        """Create DataFrame with asymmetric error bars for testing conversion."""
        return pd.DataFrame(
            {
                "channel": ["TV", "Radio"],
                "mean": [100.0, 200.0],
                "median": [100.0, 200.0],
                "abs_error_94_lower": [90.0, 190.0],  # 10 below each mean
                "abs_error_94_upper": [115.0, 220.0],  # 15 above TV, 20 above Radio
            }
        )

    @pytest.fixture
    def no_hdi_df(self):
        """Create DataFrame without HDI columns."""
        return pd.DataFrame(
            {
                "channel": ["TV", "Radio"],
                "mean": [100.0, 200.0],
                "median": [100.0, 200.0],
            }
        )

    @pytest.mark.parametrize("method_name", ["contributions", "roas"])
    def test_has_error_bars(self, method_name, bar_chart_df):
        """Test that method returns a Figure with error bars."""
        # Arrange
        mock_summary = _create_simple_mock_summary(
            df=bar_chart_df, method_name=method_name
        )
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        method = getattr(factory, method_name)
        fig = method(auto_facet=False)

        # Assert that error bars are added
        bar_trace = fig.data[0]
        assert bar_trace.error_y is not None, (
            f"{method_name} bar chart should have error_y for HDI upper bounds"
        )
        assert hasattr(bar_trace.error_y, "array"), (
            f"{method_name} error_y should have array attribute"
        )

    @pytest.mark.parametrize("method_name", ["contributions", "roas"])
    def test_converts_absolute_to_relative_errors(
        self, method_name, asymmetric_error_df
    ):
        """Test that absolute HDI bounds are converted to relative errors."""
        # Arrange
        mock_summary = _create_simple_mock_summary(
            df=asymmetric_error_df, method_name=method_name
        )
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        method = getattr(factory, method_name)
        fig = method(hdi_prob=0.94, auto_facet=False)

        # Assert - Check that error bars have correct relative values
        bar_trace = fig.data[0]
        assert bar_trace.error_y is not None, (
            f"{method_name} bar chart should have error_y for HDI upper bounds"
        )

        # Verify upper errors: absolute_upper - mean
        # TV: 115 - 100 = 15, Radio: 220 - 200 = 20
        np.testing.assert_array_almost_equal(
            bar_trace.error_y.array,
            [15.0, 20.0],
            err_msg=f"{method_name}: Upper error bars should be [15.0, 20.0]",
        )

        # Verify lower errors: mean - absolute_lower
        # TV: 100 - 90 = 10, Radio: 200 - 190 = 10
        np.testing.assert_array_almost_equal(
            bar_trace.error_y.arrayminus,
            [10.0, 10.0],
            err_msg=f"{method_name}: Lower error bars should be [10.0, 10.0]",
        )

    @pytest.mark.parametrize("method_name", ["contributions", "roas"])
    def test_no_error_bars_when_hdi_prob_is_none(self, method_name, no_hdi_df):
        """Test that no error bars are added when hdi_prob=None."""
        # Arrange
        mock_summary = _create_simple_mock_summary(
            df=no_hdi_df, method_name=method_name
        )
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        method = getattr(factory, method_name)
        fig = method(hdi_prob=None, auto_facet=False)

        # Assert
        bar_trace = fig.data[0]
        # Plotly always creates an error_y object, but array should be None when no errors
        assert bar_trace.error_y.array is None, (
            f"{method_name} bar chart should not have error bar data when hdi_prob=None"
        )


class TestMMMPlotlyFactoryContributions:
    """Tests specific to MMMPlotlyFactory.contributions() method."""

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

    def test_plot_interactive_property(self, simple_fitted_mmm):
        """Test that plot_interactive property returns configured factory."""
        factory = simple_fitted_mmm.plot_interactive

        assert factory is not None, "plot_interactive should not be None"
        assert isinstance(factory, MMMPlotlyFactory), (
            f"Expected MMMPlotlyFactory, got {type(factory)}"
        )
        assert hasattr(factory, "summary"), "Factory should have summary attribute"
        assert factory.summary is not None, "Factory summary should not be None"


class TestMMMPlotlyFactoryPosteriorPredictive:
    """Tests for MMMPlotlyFactory.posterior_predictive() method."""

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


class TestMMMPlotlyFactoryROAS:
    """Tests specific to MMMPlotlyFactory.roas() method."""

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


class TestMMMPlotlyFactorySaturationCurves:
    """Tests for MMMPlotlyFactory.saturation_curves() method."""

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


class TestMMMPlotlyFactoryAdstockCurves:
    """Tests for MMMPlotlyFactory.adstock_curves() method."""

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


class TestSaturationCurvesFaceting:
    @pytest.mark.parametrize(
        "kwargs, expected_rows, expected_cols",
        [
            pytest.param(
                {},
                1,
                2,
                id="auto_facet_creates_two_subplots",
            ),
            pytest.param(
                {"auto_facet": False},
                1,
                1,
                id="auto_facet_false_creates_one_plot",
            ),
            pytest.param(
                {"facet_col": "geo", "auto_facet": False},
                1,
                2,
                id="facet_col_creates_two_column_subplots",
            ),
            pytest.param(
                {"facet_col": "geo", "auto_facet": True},
                1,
                2,
                id="facet_col_with_auto_facet_true_same_as_auto_facet_false",
            ),
            pytest.param(
                {"facet_row": "geo", "auto_facet": False},
                2,
                1,
                id="facet_row_creates_two_row_subplots",
            ),
        ],
    )
    def test_saturation_curves_faceting_one_custom_dim(
        self, kwargs, expected_rows, expected_cols
    ):
        """Test saturation_curves faceting behavior with 1 custom dimension.

        With 2 channels × 2 geo coordinates = 4 lines total.
        """
        # Arrange
        df = _create_saturation_df_one_custom_dim(
            n_channels=2, n_coords=2, custom_dim="geo"
        )
        mock_summary = _create_saturation_mock_summary(df, custom_dims=["geo"])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        fig = factory.saturation_curves(hdi_prob=None, **kwargs)

        # Assert
        rows, cols = _get_subplot_dimensions(fig)
        n_lines = _count_line_traces(fig)

        assert rows == expected_rows, f"Expected {expected_rows} rows, got {rows}"
        assert cols == expected_cols, f"Expected {expected_cols} columns, got {cols}"
        assert n_lines == 4, (
            f"Expected 4 total lines (2 channels × 2 geos), got {n_lines}"
        )

    @pytest.mark.parametrize(
        "kwargs, expected_rows, expected_cols",
        [
            pytest.param(
                {},
                2,
                2,
                id="auto_facet_true",
            ),
            pytest.param(
                {"facet_col": "geo", "auto_facet": False},
                1,
                2,
                id="facet_col_creates_two_column_subplots",
            ),
            pytest.param(
                {"facet_col": "geo", "auto_facet": True},
                1,
                2,
                id="facet_col_with_auto_facet_true_same_as_auto_facet_false",
            ),
            pytest.param(
                {"facet_row": "geo", "auto_facet": False},
                2,
                1,
                id="facet_row_creates_two_row_subplots",
            ),
        ],
    )
    def test_saturation_curves_faceting_two_custom_dims(
        self, kwargs, expected_rows, expected_cols
    ):
        """Test saturation_curves faceting behavior with 2 custom dimensions.

        With 2 channels × 2 geo × 2 brand coordinates = 8 lines total.
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
        fig = factory.saturation_curves(hdi_prob=None, **kwargs)

        # Assert
        rows, cols = _get_subplot_dimensions(fig)
        n_lines = _count_line_traces(fig)

        assert rows == expected_rows, f"Expected {expected_rows} rows, got {rows}"
        assert cols == expected_cols, f"Expected {expected_cols} columns, got {cols}"
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

        # Expected scales: TV,US=1000  TV,UK=800  Radio,US=500  Radio,UK=400
        expected_scales = {
            ("TV", "US"): 1000.0,
            ("TV", "UK"): 800.0,
            ("Radio", "US"): 500.0,
            ("Radio", "UK"): 400.0,
        }

        for (channel, geo), scale in expected_scales.items():
            trace = next(
                (t for t in traces if channel in t.name and geo in t.name), None
            )
            assert trace is not None, f"Should have {channel}, {geo} trace"

            expected_x = [0.0, 0.5 * scale, 1.0 * scale]
            np.testing.assert_array_almost_equal(
                trace.x,
                expected_x,
                err_msg=f"{channel}, {geo} x values should be scaled by {scale}",
            )


class TestMMMPlotlyFactoryErrorHandling:
    """Tests for error handling in MMMPlotlyFactory."""

    def test_get_hdi_columns_raises_for_empty_dataframe(self):
        """Test error when DataFrame is empty and HDI columns expected."""
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.data = Mock(custom_dims=[])
        factory = MMMPlotlyFactory(summary=mock_summary)

        empty_df = pd.DataFrame(columns=["channel", "mean"])
        nw_df = nw.from_native(empty_df)

        with pytest.raises(ValueError, match="DataFrame is empty"):
            factory._get_hdi_columns(nw_df, hdi_prob=0.94)

    def test_get_hdi_columns_raises_for_missing_columns(self):
        """Test error when HDI columns don't exist."""
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.data = Mock(custom_dims=[])
        factory = MMMPlotlyFactory(summary=mock_summary)

        df = pd.DataFrame({"channel": ["TV"], "mean": [100.0]})
        nw_df = nw.from_native(df)

        with pytest.raises(
            ValueError, match=r"HDI columns for probability 0\.94 not found"
        ):
            factory._get_hdi_columns(nw_df, hdi_prob=0.94)

    def test_contributions_raises_when_date_not_in_x_or_color(self):
        """Test error when date column exists but not assigned to x or color."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "channel": ["TV", "Radio", "Social"],
                "mean": [100.0, 200.0, 300.0],
                "median": [100.0, 200.0, 300.0],
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.contributions.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary)

        with pytest.raises(ValueError, match="choose either x='date' or color='date'"):
            factory.contributions(auto_facet=False)  # date exists but not assigned

    def test_contributions_raises_when_component_not_in_x_or_color(self):
        """Test error when component column not assigned to x or color."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=2),
                "channel": ["TV", "Radio"],
                "mean": [100.0, 200.0],
                "median": [100.0, 200.0],
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.contributions.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary)

        with pytest.raises(
            ValueError, match="choose either x=`channel` or color=`channel`"
        ):
            factory.contributions(x="date", color="date", auto_facet=False)

    def test_roas_raises_when_channel_not_in_x_or_color(self):
        """Test error when channel not assigned to x or color in ROAS."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=2),
                "channel": ["TV", "Radio"],
                "mean": [2.5, 3.0],
                "median": [2.4, 3.1],
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.roas.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary)

        with pytest.raises(
            ValueError, match="choose either x='channel' or color='channel'"
        ):
            factory.roas(x="date", color="date", auto_facet=False)

    def test_posterior_predictive_raises_for_missing_columns(self):
        """Test error when required columns missing from posterior predictive."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "mean": [100.0, 200.0, 300.0],
                # Missing "observed" column
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.posterior_predictive.return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary)

        with pytest.raises(ValueError, match="missing required columns"):
            factory.posterior_predictive(auto_facet=False)

    def test_plot_bar_raises_for_missing_y_column(self):
        """Test error when y column missing from DataFrame."""
        df = pd.DataFrame({"channel": ["TV", "Radio"]})
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary)
        nw_df = nw.from_native(df)

        with pytest.raises(ValueError, match="DataFrame must have 'mean' column"):
            factory._plot_bar(nw_df, x="channel", y="mean")

    def test_plot_bar_raises_for_missing_x_column(self):
        """Test error when x column missing from DataFrame."""
        df = pd.DataFrame({"mean": [100.0, 200.0]})
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary)
        nw_df = nw.from_native(df)

        with pytest.raises(ValueError, match="DataFrame must have 'channel' column"):
            factory._plot_bar(nw_df, x="channel", y="mean")


class TestAutoFacetingEdgeCases:
    """Tests for _apply_auto_faceting edge cases."""

    def test_auto_facet_false_with_two_dims_raises_for_bar_chart(self):
        """Test error when auto_facet=False with 2 custom dims (no line_dash support)."""
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.data = Mock(custom_dims=["geo", "brand"])

        factory = MMMPlotlyFactory(summary=mock_summary)

        with pytest.raises(ValueError, match="Too many custom dimensions"):
            factory._apply_auto_faceting(
                {}, auto_facet=False, supports_line_styling=False
            )

    def test_auto_facet_true_with_three_dims_raises_for_bar_chart(self):
        """Test error when 3 custom dims without line_dash support."""
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.data = Mock(custom_dims=["geo", "brand", "segment"])

        factory = MMMPlotlyFactory(summary=mock_summary)

        with pytest.raises(ValueError, match="Too many custom dimensions"):
            factory._apply_auto_faceting(
                {}, auto_facet=True, supports_line_styling=False
            )

    def test_auto_facet_with_three_dims_uses_line_dash(self):
        """Test that 3 custom dims uses line_dash for line charts."""
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.data = Mock(custom_dims=["geo", "brand", "segment"])

        factory = MMMPlotlyFactory(summary=mock_summary)

        result = factory._apply_auto_faceting(
            {}, auto_facet=True, supports_line_styling=True
        )

        assert result["facet_row"] == "geo"
        assert result["facet_col"] == "brand"
        assert result["line_dash"] == "segment"

    def test_manual_faceting_with_remaining_dim_uses_line_dash(self):
        """Test that manual faceting leaves remaining dim for line_dash."""
        mock_summary = Mock(spec=MMMSummaryFactory)
        mock_summary.data = Mock(custom_dims=["geo", "brand"])

        factory = MMMPlotlyFactory(summary=mock_summary)

        result = factory._apply_auto_faceting(
            {"facet_col": "geo"}, auto_facet=True, supports_line_styling=True
        )

        assert result["facet_col"] == "geo"
        assert result.get("line_dash") == "brand"


class TestDateFormatting:
    """Tests for date formatting functionality."""

    def test_format_date_column_quarterly(self):
        """Test quarterly date formatting (YYYY-QN)."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"]
                ),
                "value": [1, 2, 3, 4],
            }
        )
        nw_df = nw.from_native(df)

        result = MMMPlotlyFactory._format_date_column(
            nw_df, "date", frequency="quarterly"
        )

        expected = ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"]
        assert result.get_column("date").to_list() == expected

    def test_format_date_column_yearly(self):
        """Test yearly date formatting."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2022-06-15", "2023-03-20", "2024-11-01"]),
                "value": [1, 2, 3],
            }
        )
        nw_df = nw.from_native(df)

        result = MMMPlotlyFactory._format_date_column(nw_df, "date", frequency="yearly")

        expected = ["2022", "2023", "2024"]
        assert result.get_column("date").to_list() == expected

    def test_format_date_column_raises_for_missing_column(self):
        """Test error when date column doesn't exist."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        nw_df = nw.from_native(df)

        with pytest.raises(ValueError, match="Column 'date' not found"):
            MMMPlotlyFactory._format_date_column(nw_df, "date")


class TestSaturationCurvesOriginalScaleFalse:
    """Tests for saturation_curves with original_scale=False."""

    def test_saturation_curves_original_scale_false_no_scaling(self):
        """Test that original_scale=False keeps x values unchanged."""
        x_vals = np.array([0.0, 0.5, 1.0])
        df = pd.DataFrame(
            {
                "x": np.tile(x_vals, 2),
                "channel": ["TV"] * 3 + ["Radio"] * 3,
                "mean": [0.0, 0.4, 0.8] * 2,
                "median": [0.0, 0.4, 0.8] * 2,
            }
        )
        mock_summary = _create_saturation_mock_summary(df, custom_dims=[])
        factory = MMMPlotlyFactory(summary=mock_summary)

        fig = factory.saturation_curves(
            hdi_prob=None, auto_facet=False, original_scale=False
        )

        # X values should be unchanged (0, 0.5, 1.0)
        for trace in fig.data:
            if trace.mode == "lines":
                np.testing.assert_array_almost_equal(
                    sorted(trace.x),
                    [0.0, 0.5, 1.0],
                    err_msg="X values should not be scaled when original_scale=False",
                )

        # get_channel_scale should NOT be called
        mock_summary.data.get_channel_scale.assert_not_called()
