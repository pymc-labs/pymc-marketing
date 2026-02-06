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
"""Interactive Plotly plotting factory for MMM.

This module provides `MMMPlotlyFactory`, which creates interactive Plotly
visualizations from MMM summary data produced by `MMMSummaryFactory`.

The factory supports:
- Bar charts with HDI error bars for contributions/ROAS
- Automatic faceting based on custom dimensions
- Both Pandas and Polars DataFrames via Narwhals

Example
-------
>>> # Disable auto-faceting and customize
>>> factory = MMMPlotlyFactory(mmm.summary, auto_facet=False)
>>> fig = factory.contributions(facet_col="country", title="Channel Effects")
>>> fig.show()
"""

from __future__ import annotations

from typing import Literal

import narwhals as nw
import plotly.express as px
import plotly.graph_objects as go
from narwhals.typing import IntoDataFrameT

from pymc_marketing.data.idata.schema import Frequency
from pymc_marketing.mmm.summary import MMMSummaryFactory

# Type aliases matching MMMSummaryFactory for consistency
ComponentType = Literal["channel", "control", "seasonality", "baseline"]


class MMMPlotlyFactory:
    """Factory for creating interactive Plotly plots from MMM summary data.

    This class provides methods for visualizing MMM results using Plotly,
    with automatic support for both Pandas and Polars DataFrames via Narwhals.

    Parameters
    ----------
    summary : MMMSummaryFactory
        Summary factory that provides access to data and model
    auto_facet : bool, default True
        Automatically detect and apply faceting for custom dimensions

    Examples
    --------
    >>> # Access via fitted MMM model
    >>> fig = mmm.plot_interactive.contributions()
    >>> fig.show()

    >>> # Disable auto-faceting
    >>> factory = MMMPlotlyFactory(mmm.summary, auto_facet=False)
    >>> fig = factory.contributions(facet_col="country")
    >>> fig.show()
    """

    def __init__(
        self,
        summary: MMMSummaryFactory,
        auto_facet: bool = True,
    ):
        """Initialize the plotting factory.

        Parameters
        ----------
        summary : MMMSummaryFactory
            Summary factory providing access to MMM data and model
        auto_facet : bool, default True
            Whether to automatically apply faceting based on custom dimensions
        """
        self.summary = summary
        self.auto_facet = auto_facet

    # Date format mapping for different frequencies
    # Note: "quarterly" is a placeholder - actual formatting handled in _format_date_column
    _DATE_FORMATS: dict[str | None, str] = {
        "yearly": "%Y",
        "monthly": "%Y-%m",
        "quarterly": "%Y-Q%q",  # Placeholder, requires special handling
        "all_time": "%Y",  # Fallback for aggregated data
        "original": "%Y-%m-%d",
        "weekly": "%Y-%m-%d",
        None: "%Y-%m-%d",
    }

    @staticmethod
    def _get_date_format(frequency: Frequency | None) -> str:
        """Get date format string based on data frequency.

        Returns the appropriate strftime format string for formatting dates
        based on the aggregation frequency of the data.

        Examples
        --------
        >>> MMMPlotlyFactory._get_date_format("yearly")
        '%Y'

        """
        return MMMPlotlyFactory._DATE_FORMATS.get(frequency, "%Y-%m-%d")

    @staticmethod
    def _is_datetime_column(nw_df: nw.DataFrame, col: str) -> bool:
        """Check if a column is a datetime type."""
        dtype = nw_df.get_column(col).dtype
        return dtype == nw.Datetime or str(dtype).startswith("Datetime")

    @staticmethod
    def _format_date_column(
        nw_df: IntoDataFrameT,
        date_col: str,
        frequency: Frequency | None = None,
    ) -> IntoDataFrameT:
        """Format date column based on data frequency.

        Formats a datetime column in a DataFrame according to the specified
        frequency. The formatted dates are stored in a new column with
        suffix "_formatted".

        Parameters
        ----------
        nw_df : IntoDataFrameT
            Narwhals DataFrame containing the date column
        date_col : str
            Name of the datetime column to format
        frequency : Frequency or None, optional
            Time aggregation frequency. If None, uses "%Y-%m-%d" format.

        Returns
        -------
        IntoDataFrameT
            DataFrame with "{date_col}" column formatted as per the frequency

        """
        if date_col not in nw_df.columns:
            raise ValueError(f"Column '{date_col}' not found in DataFrame")

        date_format = MMMPlotlyFactory._get_date_format(frequency)

        # Handle quarterly separately since strftime doesn't support %q directly
        if frequency == "quarterly":
            # Calculate quarter and format as "YYYY-QN"
            nw_df = nw_df.with_columns(
                **{
                    f"{date_col}": (
                        nw.col(date_col).dt.year().cast(nw.String)
                        + "-Q"
                        + ((nw.col(date_col).dt.month() - 1) // 3 + 1).cast(nw.String)
                    )
                }
            )
        else:
            # Use standard strftime formatting for other frequencies
            nw_df = nw_df.with_columns(
                **{f"{date_col}": nw.col(date_col).dt.to_string(date_format)}
            )

        return nw_df

    def _get_hdi_columns(
        self, nw_df: IntoDataFrameT, hdi_prob: float
    ) -> tuple[str, str]:
        """Get HDI column names for probability level.

        Parameters
        ----------
        nw_df : IntoDataFrameT
            Narwhals DataFrame containing HDI columns
        hdi_prob : float
            HDI probability (e.g., 0.94 for 94% interval)

        Returns
        -------
        tuple[str, str]
            (lower_col, upper_col) names
        """
        prob_str = str(round(hdi_prob * 100))
        lower_col = f"abs_error_{prob_str}_lower"
        upper_col = f"abs_error_{prob_str}_upper"

        if lower_col in nw_df.columns and upper_col in nw_df.columns:
            return lower_col, upper_col

        # Check if DataFrame is empty for more informative error
        if len(nw_df) == 0:
            raise ValueError(
                f"Cannot compute HDI error bars: DataFrame is empty. "
                f"Expected columns: {lower_col}, {upper_col}."
            )

        raise ValueError(
            f"HDI columns for probability {hdi_prob} not found. "
            f"Expected: {lower_col}, {upper_col}. "
            f"Available columns: {nw_df.columns}"
        )

    def _apply_auto_faceting(self, plotly_kwargs: dict) -> dict:
        """Apply automatic faceting based on custom dimensions.

        Strategy:
        - 1 custom dimension → facet_col with facet_col_wrap=3
        - 2+ custom dimensions → facet_row=dims[0], facet_col=dims[1]
        - Manual faceting in plotly_kwargs takes precedence

        Parameters
        ----------
        plotly_kwargs : dict
            Existing Plotly kwargs (may contain manual faceting)

        Returns
        -------
        dict
            Updated kwargs with faceting parameters

        Examples
        --------
        >>> # Single dimension: country
        >>> kwargs = factory._apply_auto_faceting({})
        >>> # Returns: {"facet_col": "country", "facet_col_wrap": 3}

        >>> # Two dimensions: country, region
        >>> kwargs = factory._apply_auto_faceting({})
        >>> # Returns: {"facet_row": "country", "facet_col": "region"}

        >>> # Manual override
        >>> kwargs = factory._apply_auto_faceting({"facet_row": "brand"})
        >>> # Returns: {"facet_row": "brand"} (auto-faceting skipped)
        """
        plotly_kwargs = plotly_kwargs.copy()

        # Skip if auto_facet disabled
        if not self.auto_facet:
            return plotly_kwargs

        # Don't override explicit faceting
        if plotly_kwargs.get("facet_row") or plotly_kwargs.get("facet_col"):
            return plotly_kwargs

        # Get custom dimensions
        custom_dims = self.summary.data.custom_dims

        # Apply faceting based on number of dimensions
        if len(custom_dims) == 1:
            plotly_kwargs["facet_col"] = custom_dims[0]
            plotly_kwargs.setdefault("facet_col_wrap", 3)
        elif len(custom_dims) >= 2:
            plotly_kwargs["facet_row"] = custom_dims[0]
            plotly_kwargs["facet_col"] = custom_dims[1]

        return plotly_kwargs

    def _plot_bar(
        self,
        df: IntoDataFrameT,
        x: str = "channel",
        y: str = "mean",
        color: str | None = None,
        hdi_prob: float | None = None,
        yaxis_title: str | None = None,
        frequency: Frequency | None = None,
        **plotly_kwargs,
    ) -> go.Figure:
        """Create a bar chart with optional error bars.

        This is a private helper used by contributions() and roas() methods.

        Parameters
        ----------
        df : IntoDataFrameT
            DataFrame (Pandas or Polars) from summary factory
        x : str, default "channel"
            Column for x-axis (typically channel or component name)
        y : str, default "mean"
            Column for y-axis (mean values)
        color : str, optional
            Column for color encoding
        hdi_prob : float, optional
            If provided, adds error bars using HDI columns
        yaxis_title : str, optional
            Y-axis label
        **plotly_kwargs
            Additional Plotly Express arguments including:
            - title: Figure title
            - facet_row, facet_col: Faceting parameters
            - barmode: "group" or "stack"

        Returns
        -------
        go.Figure
            Plotly figure with bar chart
        """
        # Convert to Narwhals for unified API
        nw_df = nw.from_native(df, eager_only=True)

        # Validate required columns
        if y not in nw_df.columns:
            raise ValueError(f"DataFrame must have '{y}' column.")

        if x not in nw_df.columns:
            raise ValueError(f"DataFrame must have '{x}' column")

        # Prepare error bars if requested
        error_y = None
        error_y_minus = None
        if hdi_prob is not None:
            lower_col, upper_col = self._get_hdi_columns(nw_df, hdi_prob)
            # Convert absolute to relative errors using Narwhals
            nw_df = nw_df.with_columns(
                error_upper=(nw.col(upper_col) - nw.col(y)),
                error_lower=(nw.col(y) - nw.col(lower_col)),
                y_string=nw.concat_str(
                    nw.col(y).cast(nw.Int64).cast(nw.String),
                    nw.lit(" HDI: ["),
                    nw.col(lower_col).cast(nw.Int64).cast(nw.String),
                    nw.lit(", "),
                    nw.col(upper_col).cast(nw.Int64).cast(nw.String),
                    nw.lit("]"),
                ),
            )
            error_y = "error_upper"
            error_y_minus = "error_lower"

        # Handle datetime columns in color dimension
        if color and color in nw_df.columns and self._is_datetime_column(nw_df, color):
            # Format dates based on frequency for better legends
            nw_df = self._format_date_column(nw_df, color, frequency=frequency)

        # handle datetime columns in x dimension
        if x and x in nw_df.columns and self._is_datetime_column(nw_df, x):
            nw_df = self._format_date_column(nw_df, x, frequency=frequency)

        # set default barmode if not provided
        plotly_kwargs.setdefault("barmode", "group")

        # set hover data
        hover_data = {x: False}
        if "y_string" in nw_df.columns:
            hover_data[yaxis_title] = nw_df.get_column("y_string").to_list()  # type: ignore[index]
        for facet in ["facet_row", "facet_col"]:
            if plotly_kwargs.get(facet):
                hover_data[plotly_kwargs.get(facet)] = False  # type: ignore[index]

        # Create bar chart (pass native DataFrame to Plotly)
        fig = px.bar(
            nw_df.to_native(),
            x=x,
            y=y,
            color=color,
            error_y=error_y,
            error_y_minus=error_y_minus,
            labels={y: yaxis_title},  # type: ignore[index]
            hover_data=hover_data,
            **plotly_kwargs,
        )

        # Clean facet titles: remove "column=" prefix, show only value
        if plotly_kwargs.get("facet_row") or plotly_kwargs.get("facet_col"):
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        return fig

    def contributions(
        self,
        hdi_prob: float | None = 0.94,
        component: ComponentType = "channel",
        frequency: Frequency | None = None,
        **plotly_kwargs,
    ) -> go.Figure:
        """Plot contributions bar chart with optional error bars and faceting.

        Creates an interactive Plotly bar chart showing contributions from
        channels, controls, seasonality, or baseline. Automatically applies
        faceting for multi-dimensional MMM models.

        Parameters
        ----------
        hdi_prob : float, optional
            HDI probability for error bars (default: 0.94). If None, no error bars.
        component : {"channel", "control", "seasonality", "baseline"}
            Which contribution component to plot (default: "channel")
        frequency : str, optional
            Time aggregation (e.g., "monthly", "all_time"). None = no aggregation.
        **plotly_kwargs
            Additional Plotly Express arguments including:
            - title: Figure title (default: "{Component} Contributions")
            - facet_row: Column for row facets (e.g., "country")
            - facet_col: Column for column facets (e.g., "brand")
            - facet_col_wrap: Max columns before wrapping
            - barmode: "group" (side-by-side) or "stack"

        Returns
        -------
        go.Figure
            Interactive Plotly figure

        Examples
        --------
        >>> # Basic channel contributions
        >>> fig = mmm.plot_interactive.contributions()
        >>> fig.show()

        >>> # Contributions by country (auto-faceted)
        >>> fig = mmm.plot_interactive.contributions(facet_col="country")
        >>> fig.show()

        >>> # Control contributions with custom title
        >>> fig = mmm.plot_interactive.contributions(
        ...     component="control", title="Control Variable Effects"
        ... )
        >>> fig.show()
        """
        # Get data from summary factory
        hdi_probs = [hdi_prob] if hdi_prob else []
        df = self.summary.contributions(
            hdi_probs=hdi_probs,
            component=component,
            frequency=frequency,
        )

        # Auto-detect faceting from custom dimensions
        plotly_kwargs = self._apply_auto_faceting(plotly_kwargs)

        # Set default values if not provided
        plotly_kwargs.setdefault("title", f"{component.capitalize()} Contributions")
        plotly_kwargs.setdefault("x", component)

        nw_df = nw.from_native(df)
        # if `date` column exist then it should either be x or color.
        x = plotly_kwargs.get("x")
        color = plotly_kwargs.get("color")

        if "date" in nw_df.columns and len(nw_df["date"].unique()) > 1:
            if "date" not in [x, color]:
                raise ValueError("choose either x='date' or color='date'")

        if component not in [x, color]:
            raise ValueError(f"choose either x=`{component}` or color=`{component}`")

        return self._plot_bar(
            df=df,
            y="mean",
            hdi_prob=hdi_prob,
            yaxis_title="Contribution",
            frequency=frequency,
            **plotly_kwargs,
        )
