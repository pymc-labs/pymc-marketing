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
import numpy as np
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
        nw_df: nw.DataFrame,
        x: str = "channel",
        y: str = "mean",
        color: str | None = None,
        hdi_prob: float | None = None,
        yaxis_title: str | None = None,
        frequency: Frequency | None = None,
        round_digits: int = 0,
        **plotly_kwargs,
    ) -> go.Figure:
        """Create a bar chart with optional error bars.

        This is a private helper used by contributions() and roas() methods.

        Parameters
        ----------
        nw_df : nw.DataFrame
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
        round_digits : int, default 0
            Number of decimal places for rounding values in hover text
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

            # Helper to format numbers with specified decimal places, handling NaN
            # Round, cast to string, then clean up trailing zeros for integers
            def _fmt_num(col: str) -> nw.Expr:
                expr = nw.col(col).round(round_digits).cast(nw.String)
                # Only remove trailing ".0" for integer rounding
                if round_digits == 0:
                    expr = expr.str.replace(r"\.0$", "")
                return expr

            # Convert absolute to relative errors using Narwhals
            nw_df = nw_df.with_columns(
                error_upper=(nw.col(upper_col) - nw.col(y)),
                error_lower=(nw.col(y) - nw.col(lower_col)),
                y_string=nw.concat_str(
                    _fmt_num(y),
                    nw.lit(" HDI: ["),
                    _fmt_num(lower_col),
                    nw.lit(", "),
                    _fmt_num(upper_col),
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
        round_digits: int = 0,
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
        round_digits : int, default 0
            Number of decimal places for rounding values in hover text.
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
            nw_df=nw_df,
            y="mean",
            hdi_prob=hdi_prob,
            yaxis_title="Contribution",
            frequency=frequency,
            round_digits=round_digits,
            **plotly_kwargs,
        )

    def roas(
        self,
        hdi_prob: float | None = 0.94,
        frequency: Frequency | None = "all_time",
        round_digits: int = 3,
        **plotly_kwargs,
    ) -> go.Figure:
        """Plot ROAS (Return on Ad Spend) bar chart.

        Creates an interactive Plotly bar chart showing ROAS for each channel,
        with optional HDI error bars and faceting for multi-dimensional models.

        Parameters
        ----------
        hdi_prob : float, optional
            HDI probability for error bars (default: 0.94). If None, no error bars.
        frequency : str, optional
            Time aggregation (default: "all_time"). Options: "original", "weekly",
            "monthly", "quarterly", "yearly", "all_time".
        round_digits : int, default 3
            Number of decimal places for rounding values in hover text.
        **plotly_kwargs
            Additional Plotly Express arguments including:
            - title: Figure title (default: "Return on Ad Spend")
            - facet_row: Column for row facets (e.g., "country")
            - facet_col: Column for column facets (e.g., "brand")
            - facet_col_wrap: Max columns before wrapping
            - barmode: "group" or "stack"

        Returns
        -------
        go.Figure
            Interactive Plotly figure

        Examples
        --------
        >>> # Basic ROAS plot
        >>> fig = mmm.plot_interactive.roas()
        >>> fig.show()

        >>> # ROAS by country (auto-faceted)
        >>> fig = mmm.plot_interactive.roas(facet_col="country")
        >>> fig.show()

        >>> # Custom frequency and HDI
        >>> fig = mmm.plot_interactive.roas(frequency="monthly", hdi_prob=0.80)
        >>> fig.show()
        """
        # Get data from summary factory
        hdi_probs = [hdi_prob] if hdi_prob else []
        df = self.summary.roas(
            hdi_probs=hdi_probs,
            frequency=frequency,
        )

        # Auto-detect faceting from custom dimensions
        plotly_kwargs = self._apply_auto_faceting(plotly_kwargs)

        # Set default values if not provided
        plotly_kwargs.setdefault("title", "Return on Ad Spend")
        plotly_kwargs.setdefault("x", "channel")

        nw_df = nw.from_native(df)
        # if `date` column exist then it should either be x or color.
        x = plotly_kwargs.get("x")
        color = plotly_kwargs.get("color")

        if "date" in nw_df.columns and len(nw_df["date"].unique()) > 1:
            if "date" not in [x, color]:
                raise ValueError("choose either x='date' or color='date'")

        if "channel" not in [x, color]:
            raise ValueError("choose either x='channel' or color='channel'")

        return self._plot_bar(
            nw_df=nw_df,
            y="mean",
            hdi_prob=hdi_prob,
            yaxis_title="ROAS",
            frequency=frequency,
            round_digits=round_digits,
            **plotly_kwargs,
        )

    def posterior_predictive(
        self,
        hdi_prob: float | None = 0.94,
        frequency: Frequency | None = None,
        **plotly_kwargs,
    ) -> go.Figure:
        """Plot posterior predictive with HDI band.

        Creates an interactive Plotly line chart showing model predictions vs
        observations, with optional HDI uncertainty band and faceting for
        multi-dimensional models.

        Parameters
        ----------
        hdi_prob : float, optional
            HDI probability for uncertainty band (default: 0.94). If None, no band.
        frequency : str, optional
            Time aggregation (e.g., "monthly", "weekly"). None = no aggregation.
        **plotly_kwargs
            Additional Plotly Express arguments including:
            - title: Figure title (default: "Posterior Predictive")
            - facet_row: Column for row facets (e.g., "country")
            - facet_col: Column for column facets (e.g., "region")
            - facet_col_wrap: Max columns before wrapping

        Returns
        -------
        go.Figure
            Interactive Plotly figure

        Examples
        --------
        >>> # Basic posterior predictive plot
        >>> fig = mmm.plot_interactive.posterior_predictive()
        >>> fig.show()

        >>> # With faceting by country
        >>> fig = mmm.plot_interactive.posterior_predictive(
        ...     facet_col="country", facet_col_wrap=3
        ... )
        >>> fig.show()

        >>> # Without HDI band
        >>> fig = mmm.plot_interactive.posterior_predictive(hdi_prob=None)
        >>> fig.show()
        """
        # Get data from Component 2 with HDI columns
        hdi_probs = [hdi_prob] if hdi_prob else []
        df = self.summary.posterior_predictive(
            hdi_probs=hdi_probs,
            frequency=frequency,
        )

        # Auto-detect faceting from custom dimensions
        plotly_kwargs = self._apply_auto_faceting(plotly_kwargs)

        # Convert to Narwhals for unified API
        nw_df = nw.from_native(df, eager_only=True)

        # Validate required columns
        required_cols = {"date", "mean", "observed"}
        if not required_cols.issubset(set(nw_df.columns)):
            raise ValueError(
                f"DataFrame missing required columns: {required_cols - set(nw_df.columns)}"
            )

        # Sort by date for proper line plotting
        nw_df = nw_df.sort("date")

        # Extract facet params for HDI band logic
        facet_row = plotly_kwargs.get("facet_row")
        facet_col = plotly_kwargs.get("facet_col")

        # Identify columns to preserve (date, facet columns)
        id_cols = ["date"]
        if facet_row:
            id_cols.append(facet_row)
        if facet_col:
            id_cols.append(facet_col)

        # Create long-format DataFrame with both predicted and observed
        plot_df = nw_df.select(*id_cols, "mean", "observed").unpivot(
            on=["mean", "observed"],
            index=id_cols,
            variable_name="series",
            value_name="value",
        )

        # Rename series values for nicer legend
        plot_df = plot_df.with_columns(
            nw.when(nw.col("series") == "mean")
            .then(nw.lit("Predicted"))
            .otherwise(nw.lit("Observed"))
            .alias("series")
        )

        # Set default title
        plotly_kwargs.setdefault("title", "Posterior Predictive")

        # Create figure with single px.line call
        fig = px.line(
            plot_df.to_native(),
            x="date",
            y="value",
            color="series",
            labels={"value": "Value", "date": "Date", "series": ""},
            **plotly_kwargs,
        )

        # Add HDI band if requested
        if hdi_prob is not None:
            lower_col, upper_col = self._get_hdi_columns(nw_df, hdi_prob)

            if facet_row is None and facet_col is None:
                # Simple case: single plot
                date_values = nw_df.get_column("date").to_list()
                lower_values = nw_df.get_column(lower_col).to_list()
                upper_values = nw_df.get_column(upper_col).to_list()
                self._add_hdi_band(
                    fig,
                    date_values,
                    lower_values,
                    upper_values,
                    name=f"{int(hdi_prob * 100)}% HDI",
                )
            else:
                # Faceted case: add band to each facet
                self._add_hdi_bands_to_facets(
                    fig,
                    nw_df,
                    facet_row,
                    facet_col,
                    lower_col,
                    upper_col,
                    hdi_prob,
                )

        # Clean up facet titles
        if facet_row or facet_col:
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        fig.update_layout(hovermode="x")
        return fig

    def _add_hdi_band(
        self,
        fig: go.Figure,
        x: list,
        lower: list,
        upper: list,
        name: str = "HDI",
        fillcolor: str | None = None,
        opacity: float = 0.2,
        showlegend: bool = True,
        row: int | None = None,
        col: int | None = None,
    ) -> None:
        """Add HDI band to figure as filled area.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure to add band to
        x : list
            X-axis values (e.g., dates)
        lower : list
            Lower bound values
        upper : list
            Upper bound values
        name : str
            Legend name
        fillcolor : str, optional
            Fill color (RGBA or hex)
        opacity : float
            Fill opacity (0-1)
        showlegend : bool
            Whether to show in legend
        row : int, optional
            Subplot row (for faceted plots)
        col : int, optional
            Subplot column (for faceted plots)
        """
        # Convert to numpy for array operations
        x_arr = np.asarray(x)
        lower_arr = np.asarray(lower)
        upper_arr = np.asarray(upper)

        # Create closed polygon: lower forward + upper backward
        x_concat = np.concatenate([x_arr, x_arr[::-1]])
        y_concat = np.concatenate([lower_arr, upper_arr[::-1]])

        trace = go.Scatter(
            x=x_concat,
            y=y_concat,
            mode="lines",
            fill="toself",
            fillcolor=fillcolor or f"rgba(65,105,225,{opacity})",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="none",
            name=name,
            showlegend=showlegend,
        )

        if row is not None and col is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)

    def _add_hdi_bands_to_facets(
        self,
        fig: go.Figure,
        nw_df,
        facet_row: str | None,
        facet_col: str | None,
        lower_col: str,
        upper_col: str,
        hdi_prob: float,
    ) -> None:
        """Add HDI bands to each facet in a faceted plot.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure with facets
        nw_df : nw.DataFrame
            Narwhals DataFrame with data
        facet_row : str or None
            Column name used for row facets
        facet_col : str or None
            Column name used for column facets
        lower_col : str
            Name of lower bound column
        upper_col : str
            Name of upper bound column
        hdi_prob : float
            HDI probability for legend name
        """
        # Get unique facet combinations
        facet_dims = []
        if facet_row:
            facet_dims.append(facet_row)
        if facet_col:
            facet_dims.append(facet_col)

        if not facet_dims:
            return

        # Get unique combinations using Narwhals
        facet_df = nw_df.select(*facet_dims).unique()

        for i, row_dict in enumerate(facet_df.to_native().to_dict("records")):
            # Build filter expression
            filter_expr = nw.lit(True)
            for dim, val in row_dict.items():
                filter_expr = filter_expr & (nw.col(dim) == val)

            # Filter data for this facet
            facet_data = nw_df.filter(filter_expr)

            # Extract values
            x_values = facet_data.get_column("date").to_list()
            lower_values = facet_data.get_column(lower_col).to_list()
            upper_values = facet_data.get_column(upper_col).to_list()

            # Determine subplot indices (1-based for Plotly)
            if facet_row and facet_col:
                row_val = row_dict[facet_row]
                col_val = row_dict[facet_col]
                row_vals = sorted(nw_df.get_column(facet_row).unique().to_list())
                col_vals = sorted(nw_df.get_column(facet_col).unique().to_list())
                row_idx = row_vals.index(row_val) + 1
                col_idx = col_vals.index(col_val) + 1
            elif facet_col:
                row_idx = 1
                col_val = row_dict[facet_col]
                col_vals = sorted(nw_df.get_column(facet_col).unique().to_list())
                col_idx = col_vals.index(col_val) + 1
            else:  # facet_row
                row_val = row_dict[facet_row]
                row_vals = sorted(nw_df.get_column(facet_row).unique().to_list())
                row_idx = row_vals.index(row_val) + 1
                col_idx = 1

            self._add_hdi_band(
                fig,
                x_values,
                lower_values,
                upper_values,
                name=f"{int(hdi_prob * 100)}% HDI",
                showlegend=(i == 0),  # Only show once in legend
                row=row_idx,
                col=col_idx,
            )

    def saturation_curves(
        self,
        hdi_prob: float | None = 0.94,
        max_value: float = 1.0,
        num_points: int = 100,
        **plotly_kwargs,
    ) -> go.Figure:
        """Plot saturation curves by channel.

        Creates an interactive Plotly line chart showing saturation response
        curves for each channel, with optional HDI uncertainty bands and faceting
        for multi-dimensional models.

        Parameters
        ----------
        hdi_prob : float, optional
            HDI probability for uncertainty bands (default: 0.94). If None, no bands.
        max_value : float, default 1.0
            Maximum value for curve x-axis (in scaled space)
        num_points : int, default 100
            Number of points to evaluate curves at
        **plotly_kwargs
            Additional Plotly Express arguments including:
            - title: Figure title (default: "Saturation Curves")
            - facet_row: Column for row facets
            - facet_col: Column for column facets
            - facet_col_wrap: Max columns before wrapping

        Returns
        -------
        go.Figure
            Interactive Plotly figure

        Examples
        --------
        >>> # Basic saturation curves
        >>> fig = mmm.plot_interactive.saturation_curves()
        >>> fig.show()

        >>> # With faceting by country
        >>> fig = mmm.plot_interactive.saturation_curves(
        ...     facet_col="country", facet_col_wrap=3
        ... )
        >>> fig.show()

        >>> # Custom x-axis range
        >>> fig = mmm.plot_interactive.saturation_curves(max_value=2.0, num_points=50)
        >>> fig.show()
        """
        # Get data from Component 2
        hdi_probs = [hdi_prob] if hdi_prob else []
        df = self.summary.saturation_curves(
            hdi_probs=hdi_probs,
            max_value=max_value,
            num_points=num_points,
        )

        # Auto-detect faceting from custom dimensions
        plotly_kwargs = self._apply_auto_faceting(plotly_kwargs)

        return self._plot_curves(
            df=df,
            x="x",
            hdi_prob=hdi_prob,
            title="Saturation Curves",
            xaxis_title="Spend (scaled)",
            yaxis_title="Response",
            **plotly_kwargs,
        )

    def adstock_curves(
        self,
        hdi_prob: float | None = 0.94,
        amount: float = 1.0,
        **plotly_kwargs,
    ) -> go.Figure:
        """Plot adstock/decay curves by channel.

        Creates an interactive Plotly line chart showing adstock decay curves
        for each channel, with optional HDI uncertainty bands and faceting for
        multi-dimensional models.

        Parameters
        ----------
        hdi_prob : float, optional
            HDI probability for uncertainty bands (default: 0.94). If None, no bands.
        amount : float, default 1.0
            Impulse amount at time 0
        **plotly_kwargs
            Additional Plotly Express arguments including:
            - title: Figure title (default: "Adstock Curves")
            - facet_row: Column for row facets
            - facet_col: Column for column facets
            - facet_col_wrap: Max columns before wrapping

        Returns
        -------
        go.Figure
            Interactive Plotly figure

        Examples
        --------
        >>> # Basic adstock curves
        >>> fig = mmm.plot_interactive.adstock_curves()
        >>> fig.show()

        >>> # With faceting by country
        >>> fig = mmm.plot_interactive.adstock_curves(facet_col="country")
        >>> fig.show()

        >>> # Custom impulse amount
        >>> fig = mmm.plot_interactive.adstock_curves(amount=100.0)
        >>> fig.show()
        """
        # Get data from Component 2
        hdi_probs = [hdi_prob] if hdi_prob else []
        df = self.summary.adstock_curves(
            hdi_probs=hdi_probs,
            amount=amount,
        )

        # Auto-detect faceting from custom dimensions
        plotly_kwargs = self._apply_auto_faceting(plotly_kwargs)

        return self._plot_curves(
            df=df,
            x="time since exposure",
            hdi_prob=hdi_prob,
            title="Adstock Curves",
            xaxis_title="Time Since Exposure",
            yaxis_title="Effect Weight",
            **plotly_kwargs,
        )

    def _plot_curves(
        self,
        df: IntoDataFrameT,
        x: str,
        y: str = "mean",
        color: str = "channel",
        hdi_prob: float | None = None,
        title: str | None = None,
        xaxis_title: str | None = None,
        yaxis_title: str | None = None,
        **plotly_kwargs,
    ) -> go.Figure:
        """Private helper: Generic curve plotting.

        Parameters
        ----------
        df : IntoDataFrameT
            DataFrame from summary factory
        x : str
            Column for x-axis (e.g., "x", "time since exposure")
        y : str, default "mean"
            Column for y-axis
        color : str, default "channel"
            Column for color encoding
        hdi_prob : float, optional
            If provided, adds HDI bands per color value
        title : str, optional
            Figure title
        xaxis_title : str, optional
            X-axis label
        yaxis_title : str, optional
            Y-axis label
        **plotly_kwargs
            Additional Plotly Express arguments

        Returns
        -------
        go.Figure
            Plotly figure with curves
        """
        # Convert to Narwhals
        nw_df = nw.from_native(df, eager_only=True)

        # Sort by x column for proper line plotting
        nw_df = nw_df.sort(x)

        # Extract facet params for HDI band logic
        facet_row = plotly_kwargs.get("facet_row")
        facet_col = plotly_kwargs.get("facet_col")

        # Set default title
        plotly_kwargs.setdefault("title", title or "Curves")

        # Create line chart (pass native DataFrame to Plotly)
        fig = px.line(
            nw_df.to_native(),
            x=x,
            y=y,
            color=color,
            labels={
                x: xaxis_title or x.capitalize(),
                y: yaxis_title or y.capitalize(),
            },
            **plotly_kwargs,
        )

        # Add HDI bands per color value if requested
        if hdi_prob is not None:
            lower_col, upper_col = self._get_hdi_columns(nw_df, hdi_prob)

            # Get unique values for color column using Narwhals
            color_values = nw_df.get_column(color).unique().to_list()

            # Add band for each color value (e.g., each channel)
            for var_val in color_values:
                # Filter using Narwhals
                nw_var = nw_df.filter(nw.col(color) == var_val)

                # Extract values for HDI band
                x_values = nw_var.get_column(x).to_list()
                lower_values = nw_var.get_column(lower_col).to_list()
                upper_values = nw_var.get_column(upper_col).to_list()

                # Add band to main plot (faceting handled by px.line)
                self._add_hdi_band(
                    fig,
                    x_values,
                    lower_values,
                    upper_values,
                    name=f"{var_val} HDI",
                    showlegend=False,
                )

        # Clean facet titles
        if facet_row or facet_col:
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        fig.update_layout(hovermode="x")
        return fig
