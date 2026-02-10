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
        Custom dimensions (e.g., ["geo"], ["geo", "brand"], ["geo", "brand", "segment"])
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

    if channels is None:
        channels = df["channel"].unique().tolist()

    # Build channel_scale DataArray with correct dimensions (works for any # of dims)
    dims = ["channel", *custom_dims]
    coords = {"channel": channels}
    shape = [len(channels)]
    for dim in custom_dims:
        dim_coords = df[dim].unique().tolist()
        coords[dim] = dim_coords
        shape.append(len(dim_coords))

    channel_scale = xr.DataArray(np.ones(shape), dims=dims, coords=coords)
    mock_summary.data.get_channel_scale.return_value = channel_scale
    return mock_summary


def _create_simple_mock_summary(
    df: pd.DataFrame | None = None,
    custom_dims: list[str] | None = None,
    method_name: str = "contributions",
) -> Mock:
    """Create a mock summary for non-saturation tests.

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


def _create_saturation_df(
    custom_dims: list[str],
    n_channels: int = 2,
    n_points: int = 10,
    include_hdi: bool = False,
) -> pd.DataFrame:
    """Create saturation curves data with any number of custom dimensions.

    Args:
        custom_dims: List of custom dimension names (e.g., ["geo"], ["geo", "brand"])
        n_channels: Number of channels (default 2)
        n_points: Number of x points per curve
        include_hdi: Whether to include HDI columns

    Returns:
        DataFrame with columns: x, channel, {custom_dims...}, mean, median
    """
    import itertools

    x_vals = np.linspace(0, 1, n_points)
    channels = ["TV", "Radio"][:n_channels]
    coord_values = [["North", "South"], ["BrandA", "BrandB"], ["Low", "High"]]

    # Build all combinations of custom dimension coordinates
    dim_coords = [coord_values[i][:2] for i in range(len(custom_dims))]
    all_combos = list(itertools.product(*dim_coords)) if dim_coords else [()]

    rows = []
    for combo in all_combos:
        for channel in channels:
            for x in x_vals:
                row = {
                    "x": x,
                    "channel": channel,
                    "mean": np.random.rand(),
                    "median": np.random.rand(),
                }
                for dim_name, coord in zip(custom_dims, combo, strict=True):
                    row[dim_name] = coord
                if include_hdi:
                    row["abs_error_94_lower"] = row["mean"] - 0.1
                    row["abs_error_94_upper"] = row["mean"] + 0.1
                rows.append(row)

    return pd.DataFrame(rows)


class TestRoasAndContributions:
    """Parametrized tests for roas() and contributions() methods.

    These methods share similar structure and mock data requirements,
    so they are tested together using parametrize.
    """

    @pytest.mark.parametrize("method_name", ["contributions", "roas"])
    def test_accepts_polars_dataframe(self, method_name):
        """Test that method handles Polars DataFrames via Narwhals."""
        df_setup = {
            "date": pd.date_range("2024-01-01", periods=3),
            "channel": ["TV", "Radio", "Social"],
            "mean": [100.0, 200.0, 300.0],
            "median": [100.0, 200.0, 300.0],
            "abs_error_94_lower": [90.0, 190.0, 290.0],
            "abs_error_94_upper": [110.0, 210.0, 310.0],
        }
        df_polars = pl.DataFrame(df_setup)
        mock_summary = Mock(spec=MMMSummaryFactory)
        getattr(mock_summary, method_name).return_value = df_polars
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary)
        method = getattr(factory, method_name)

        fig = method(color="date" if "date" in df_setup else None)

        assert isinstance(fig, go.Figure), f"{method_name} should handle Polars input"

    @pytest.mark.parametrize("method_name", ["contributions", "roas"])
    def test_converts_absolute_to_relative_errors(self, method_name):
        """Test that absolute HDI bounds are converted to relative errors."""
        # Arrange - DataFrame with asymmetric error bars
        # 10 below each mean, 15 above TV, 20 above Radio
        df = pd.DataFrame(
            {
                "channel": ["TV", "Radio"],
                "mean": [100.0, 200.0],
                "median": [100.0, 200.0],
                "abs_error_94_lower": [90.0, 190.0],
                "abs_error_94_upper": [115.0, 220.0],
            }
        )
        mock_summary = _create_simple_mock_summary(df=df, method_name=method_name)
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        method = getattr(factory, method_name)
        fig = method(hdi_prob=0.94)

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
    def test_no_error_bars_when_hdi_prob_is_none(self, method_name):
        """Test that no error bars are added when hdi_prob=None."""
        # Arrange - DataFrame without HDI columns
        df = pd.DataFrame(
            {
                "channel": ["TV", "Radio"],
                "mean": [100.0, 200.0],
                "median": [100.0, 200.0],
            }
        )
        mock_summary = _create_simple_mock_summary(df=df, method_name=method_name)
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act
        method = getattr(factory, method_name)
        fig = method(hdi_prob=None)

        # Assert
        bar_trace = fig.data[0]
        # Plotly always creates an error_y object, but array should be None when no errors
        assert bar_trace.error_y.array is None, (
            f"{method_name} bar chart should not have error bar data when hdi_prob=None"
        )

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
        fig = factory.roas(hdi_prob=0.94)

        # Verify figure is returned
        assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"


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
        rows, cols = _get_subplot_dimensions(fig)
        assert rows == 2 and cols == 2, (
            "2D faceted figure should have 2 rows and 2 columns"
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

        factory.contributions(component="control")

        # Assert
        call_kwargs = mock_summary.contributions.call_args[1]
        assert call_kwargs["component"] == "control"


class TestMMMPlotInteractiveProperty:
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

    @pytest.mark.parametrize(
        "hdi_prob, expected_hdi_traces",
        [
            pytest.param(
                None,
                0,
                id="hdi_prob_none_no_hdi_band",
            ),
            pytest.param(
                0.94,
                1,
                id="hdi_prob_94_adds_hdi_band",
            ),
        ],
    )
    def test_posterior_predictive_hdi_behavior(self, hdi_prob, expected_hdi_traces):
        """Test posterior predictive traces and HDI band behavior with various hdi_prob values."""
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
        fig = factory.posterior_predictive(hdi_prob=hdi_prob)

        # Assert - Always check for Predicted and Observed traces
        assert _count_line_traces(fig) == 2, (
            f"Expected 2 traces (Predicted + Observed), got {_count_line_traces(fig)}"
        )

        # Assert - Check HDI band presence based on expected_hdi_traces
        hdi_traces = [t for t in fig.data if t.fill == "toself"]
        assert len(hdi_traces) == expected_hdi_traces, (
            f"Expected {expected_hdi_traces} HDI bands, got {len(hdi_traces)}"
        )

    @pytest.mark.parametrize("df_type", [pd.DataFrame, pl.DataFrame])
    def test_posterior_predictive_with_custom_dimensions(self, df_type):
        """Test posterior predictive with faceting by country."""
        # Arrange
        df = df_type(
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


class TestCurveMethodsHDI:
    """Tests for HDI bands in saturation_curves() and adstock_curves() methods."""

    def test_saturation_curves_hdi_bands(self):
        """Test that HDI bands are added for each channel in saturation curves."""
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
        fig = factory.saturation_curves(hdi_prob=0.94)

        # Assert
        # Should have filled traces for HDI bands
        hdi_traces = [t for t in fig.data if t.fill == "toself"]
        assert len(hdi_traces) >= 2, (
            f"Should have HDI bands for both channels, got {len(hdi_traces)}"
        )

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
        fig = factory.adstock_curves(hdi_prob=0.94)

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
        df = _create_saturation_df(custom_dims=["geo"], n_channels=2)
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
        df = _create_saturation_df(custom_dims=["geo", "brand"], n_channels=2)
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
        df = _create_saturation_df(custom_dims=["geo", "brand"], n_channels=2)
        mock_summary = _create_saturation_mock_summary(df, custom_dims=["geo", "brand"])
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act & Assert - Should raise ValueError because too many combinations
        with pytest.raises(ValueError, match="Too many custom dimensions"):
            factory.saturation_curves(hdi_prob=None, auto_facet=False)


class TestSaturationCurvesOriginalScale:
    """Tests for saturation_curves original_scale parameter.

    When original_scale=True (default), the x-axis values should be multiplied
    by the channel scale factors to convert from scaled space to original units.
    When original_scale=False, x values remain unchanged.
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
            hdi_prob=None, original_scale=True, auto_facet=False
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

    def test_original_scale_false_no_scaling(self):
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

        fig = factory.saturation_curves(hdi_prob=None, original_scale=False)

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


class TestMMMPlotlyFactoryErrorHandling:
    """Tests for error handling in MMMPlotlyFactory."""

    @pytest.mark.parametrize(
        "method_name, column_name, error_pattern",
        [
            pytest.param(
                "contributions",
                None,
                "choose either x='date' or color='date'",
                id="contributions_date_not_assigned",
            ),
            pytest.param(
                "contributions",
                "date",
                "choose either x='channel' or color='channel'",
                id="contributions_component_not_assigned",
            ),
            pytest.param(
                "roas",
                "date",
                "choose either x='channel' or color='channel'",
                id="roas_channel_not_assigned",
            ),
        ],
    )
    def test_raises_when_required_column_not_in_x_or_color(
        self, method_name, column_name, error_pattern
    ):
        """Test error when required column not assigned to x or color."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=2),
                "channel": ["TV", "Radio"],
                "mean": [100.0, 200.0],
                "median": [100.0, 200.0],
            }
        )
        mock_summary = Mock(spec=MMMSummaryFactory)
        getattr(mock_summary, method_name).return_value = df
        mock_summary.data = Mock(custom_dims=[])

        factory = MMMPlotlyFactory(summary=mock_summary)

        kwargs = {}
        if column_name is not None:
            kwargs["x"] = column_name
            kwargs["color"] = column_name

        with pytest.raises(ValueError, match=error_pattern):
            getattr(factory, method_name)(**kwargs)

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
            factory.posterior_predictive()

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


class TestHDIBandsWithLineDash:
    """Tests for HDI bands when line_dash dimension is used (3 custom dimensions)."""

    def test_hdi_bands_filter_by_line_dash(self):
        """HDI bands must filter by line_dash to avoid duplicate x-values.

        BUG: _add_hdi_bands filters by facet_row, facet_col, and color, but NOT
        by line_dash. When 3 custom dimensions are used (dim3 maps to line_dash),
        the filtered data contains multiple rows per x-value, causing malformed
        HDI band polygons that zigzag between curves.
        """
        # Arrange - 2 channels x 2 geo x 2 brand x 2 segment with 10 x-points
        df = _create_saturation_df(
            custom_dims=["geo", "brand", "segment"], n_channels=2, include_hdi=True
        )
        mock_summary = _create_saturation_mock_summary(
            df, custom_dims=["geo", "brand", "segment"]
        )
        factory = MMMPlotlyFactory(summary=mock_summary)

        # Act - auto_facet with 3 dims: facet_row=geo, facet_col=brand, line_dash=segment
        fig = factory.saturation_curves(hdi_prob=0.94, auto_facet=True)

        # Assert - HDI bands should have unique x-values (not duplicated per line_dash)
        hdi_traces = [t for t in fig.data if t.fill == "toself"]
        assert len(hdi_traces) > 0, "Should have HDI band traces"

        for trace in hdi_traces:
            # HDI polygon: [x1..xN, xN..x1], so first half is forward x-values
            forward_x = list(trace.x)[: len(trace.x) // 2]
            unique_x = list(dict.fromkeys(forward_x))

            assert len(unique_x) == len(forward_x), (
                f"HDI band has duplicate x-values (line_dash not filtered). "
                f"Expected {len(forward_x)} unique, got {len(unique_x)}"
            )


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
