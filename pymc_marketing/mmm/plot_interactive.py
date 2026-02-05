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

- **Contributions**: Bar charts showing channel/control/seasonality contributions
- **ROAS**: Return on Ad Spend analysis with confidence intervals
- **Posterior Predictive**: Time series with HDI bands comparing actual vs predicted
- **Saturation Curves**: Visualize diminishing returns per channel
- **Adstock Curves**: Show carryover effects over time
- Automatic faceting based on custom dimensions (e.g., geo, brand)
- Both Pandas and Polars DataFrames via Narwhals

Examples
--------
**Basic Usage via MMM Model**

Access the plotting factory directly from a fitted MMM model:

>>> # Posterior predictive with actual vs predicted
>>> fig = mmm.plot_interactive.posterior_predictive()
>>> fig.show()

>>> # Channel contributions over time
>>> fig = mmm.plot_interactive.contributions()
>>> fig.show()

>>> # ROAS analysis aggregated by year
>>> fig = mmm.plot_interactive.roas(frequency="yearly")
>>> fig.show()

>>> # Saturation curves showing diminishing returns
>>> fig = mmm.plot_interactive.saturation_curves()
>>> fig.show()

>>> # Adstock curves showing carryover effects
>>> fig = mmm.plot_interactive.adstock_curves()
>>> fig.show()

**Customizing Plots**

Control faceting and styling with kwargs:

>>> # ROAS colored by date, grouped by channel
>>> fig = mmm.plot_interactive.roas(frequency="yearly", color="date", x="channel")
>>> fig.show()

>>> # Disable auto-faceting and manually set facet column
>>> fig = mmm.plot_interactive.contributions(
...     facet_col="country", title="Channel Effects by Country"
... )
>>> fig.show()

>>> # Saturation curves faceted by brand
>>> fig = mmm.plot_interactive.saturation_curves(
...     facet_row="brand",
... )
>>> fig.show()

**Working with Filtered/Aggregated Data**

Create custom factories with filtered or aggregated data:

>>> from pymc_marketing.mmm.summary import MMMSummaryFactory
>>> from pymc_marketing.mmm.plot_interactive import MMMPlotlyFactory

>>> # Aggregate multiple geos into one
>>> agg_data = mmm.data.aggregate_dims(
...     dim="geo", values=["geo_a", "geo_b"], new_label="all_geos"
... )
>>> agg_summary = MMMSummaryFactory(agg_data, mmm)
>>> agg_factory = MMMPlotlyFactory(summary=agg_summary)
>>> fig = agg_factory.roas(frequency="yearly", color="channel", x="date")
>>> fig.show()

>>> # Filter to specific geo
>>> filtered_data = mmm.data.filter_dims(geo="geo_a")
>>> filtered_summary = MMMSummaryFactory(filtered_data, mmm, validate_data=False)
>>> filtered_factory = MMMPlotlyFactory(summary=filtered_summary)
>>> fig = filtered_factory.roas(frequency="yearly", color="channel", x="date")
>>> fig.show()

>>> # Filter by date range
>>> filtered_data = mmm.data.filter_dates(start_date="2024-01-01")
>>> filtered_summary = MMMSummaryFactory(filtered_data, mmm)
>>> filtered_factory = MMMPlotlyFactory(summary=filtered_summary)
>>> fig = filtered_factory.roas(frequency="quarterly", color="channel", x="date")
>>> fig.show()
"""

from __future__ import annotations

from typing import Literal

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrameT

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as e:
    raise ImportError(
        "Plotly is required for interactive plotting. "
        "Install it with: pip install pymc-marketing[plotly]"
    ) from e

from pymc_marketing.data.idata.schema import Frequency
from pymc_marketing.mmm.summary import MMMSummaryFactory

# Type aliases matching MMMSummaryFactory for consistency
ComponentType = Literal["channel", "control", "seasonality", "baseline"]

# Default Plotly color sequence
PLOTLY_COLORS = px.colors.qualitative.Plotly


def _hex_to_rgba(color: str, opacity: float) -> str:
    """Convert a hex color to rgba string.

    Parameters
    ----------
    color : str
        Hex color string (e.g., "#636EFA")
    opacity : float
        Opacity value between 0 and 1

    Returns
    -------
    str
        RGBA color string (e.g., "rgba(99,110,250,0.3)")
    """
    hex_color = color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


class MMMPlotlyFactory:
    """Factory for creating interactive Plotly plots from MMM summary data.

    This class provides methods for visualizing MMM results using Plotly,
    with automatic support for both Pandas and Polars DataFrames via Narwhals.

    Parameters
    ----------
    summary : MMMSummaryFactory
        Summary factory that provides access to data and model

    Attributes
    ----------
    custom_dims : list[str]
        Custom dimensions available for faceting (e.g., ["geo", "brand"])

    Methods
    -------
    contributions(hdi_prob, component, frequency, ...)
        Bar chart of channel/control/seasonality contributions
    roas(hdi_prob, frequency, ...)
        Bar chart of Return on Ad Spend metrics
    posterior_predictive(hdi_prob, frequency, ...)
        Time series comparing actual vs predicted with HDI bands
    saturation_curves(hdi_prob, max_value, ...)
        Line plots showing diminishing returns per channel
    adstock_curves(hdi_prob, amount, ...)
        Line plots showing carryover effects over time

    Examples
    --------
    **Access via fitted MMM model**

    >>> # Basic posterior predictive plot
    >>> fig = mmm.plot_interactive.posterior_predictive()
    >>> fig.show()

    >>> # Channel contributions with default settings
    >>> fig = mmm.plot_interactive.contributions()
    >>> fig.show()

    >>> # ROAS aggregated yearly, colored by date
    >>> fig = mmm.plot_interactive.roas(frequency="yearly", color="date", x="channel")
    >>> fig.show()

    **Controlling faceting behavior**

    For models with custom dimensions (geo, brand, etc.), auto_facet=True
    (default) automatically creates subplots for each dimension combination.

    >>> # Auto-facet enabled (default): creates subplots automatically
    >>> fig = mmm.plot_interactive.saturation_curves()
    >>> fig.show()

    >>> # Disable auto-facet and manually control faceting
    >>> fig = mmm.plot_interactive.saturation_curves(
    ...     facet_row="brand", auto_facet=False
    ... )
    >>> fig.show()

    >>> # Use facet_col instead of facet_row
    >>> fig = mmm.plot_interactive.saturation_curves(facet_col="brand", auto_facet=True)
    >>> fig.show()

    **Customizing appearance with Plotly kwargs**

    Any additional keyword arguments are passed directly to Plotly Express:

    >>> fig = mmm.plot_interactive.contributions(
    ...     title="My Custom Title",
    ...     height=600,
    ...     width=1000,
    ...     color_discrete_sequence=["red", "blue", "green"],
    ... )
    >>> fig.show()

    See Also
    --------
    MMMSummaryFactory : Data factory providing summary statistics
    MMMPlotSuite : Static matplotlib plotting functionality
    """

    def __init__(
        self,
        summary: MMMSummaryFactory,
    ):
        """Initialize the plotting factory.

        Parameters
        ----------
        summary : MMMSummaryFactory
            Summary factory providing access to MMM data and model
        """
        self.summary = summary

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

    @property
    def custom_dims(self) -> list[str]:
        """Get custom dimensions from summary factory."""
        return self.summary.data.custom_dims

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

    def _apply_auto_faceting(
        self,
        plotly_kwargs: dict,
        auto_facet: bool = True,
        single_dim_facet: Literal["col", "row"] = "col",
        supports_line_styling: bool = False,
    ) -> dict:
        """Apply automatic faceting based on custom dimensions.

        Strategy:
        - auto_facet=True:
          - 1 custom dimension → facet_col or facet_row with wrap=3
          - 2 custom dimensions → facet_row=dims[0], facet_col=dims[1]
          - 3 custom dimensions (line charts only) → adds line_dash=dims[2]
          - More dimensions than supported → raises ValueError
        - auto_facet=False with supports_line_styling=True:
          - 1 custom dimension → line_dash=dims[0] (visual differentiation without faceting)
          - 2+ custom dimensions → raises ValueError
        - auto_facet=False with supports_line_styling=False:
          - No automatic styling applied (for bar charts etc.)
        - Manual faceting/styling in plotly_kwargs:
          - Remaining custom dimensions are applied via line_dash if supported

        Parameters
        ----------
        plotly_kwargs : dict
            Existing Plotly kwargs (may contain manual faceting)
        auto_facet : bool, default True
            Whether to automatically apply faceting based on custom dimensions
        single_dim_facet : {"col", "row"}, default "col"
            When there is exactly one custom dimension, this controls
            whether it is applied as facet_col or facet_row.
        supports_line_styling : bool, default False
            Whether the plot type supports line_dash parameter.
            Set to True for line charts (px.line), False for bar charts (px.bar).

        Returns
        -------
        dict
            Updated kwargs with faceting parameters

        Examples
        --------
        >>> # Single dimension: country (default uses facet_col)
        >>> kwargs = factory._apply_auto_faceting({})
        >>> # Returns: {"facet_col": "country", "facet_col_wrap": 3}

        >>> # Single dimension with single_dim_facet="row"
        >>> kwargs = factory._apply_auto_faceting({}, single_dim_facet="row")
        >>> # Returns: {"facet_row": "country"} (no wrap for row faceting)

        >>> # Two dimensions: country, region
        >>> kwargs = factory._apply_auto_faceting({})
        >>> # Returns: {"facet_row": "country", "facet_col": "region"}

        >>> # Three dimensions on line chart: country, region, segment
        >>> kwargs = factory._apply_auto_faceting({}, supports_line_styling=True)
        >>> # Returns: {"facet_row": "country", "facet_col": "region",
        >>> #          "line_dash": "segment"}

        >>> # Manual override
        >>> kwargs = factory._apply_auto_faceting({"facet_row": "brand"})
        >>> # Returns: {"facet_row": "brand"} (auto-faceting skipped)

        >>> # auto_facet=False with custom dimension: country (line chart)
        >>> kwargs = factory._apply_auto_faceting(
        ...     {}, auto_facet=False, supports_line_styling=True
        ... )
        >>> # Returns: {"line_dash": "country"} (uses line dash for differentiation)
        """
        plotly_kwargs = plotly_kwargs.copy()

        # Get custom dimensions
        custom_dims = self.custom_dims

        # No custom dimensions, no faceting needed
        if len(custom_dims) == 0:
            return plotly_kwargs

        # Get manually specified faceting
        manual_facet_row = plotly_kwargs.get("facet_row")
        manual_facet_col = plotly_kwargs.get("facet_col")
        has_manual_faceting = manual_facet_row or manual_facet_col

        # Calculate dimensions already used for faceting
        faceted_dims = [
            dim for dim in (manual_facet_row, manual_facet_col) if dim in custom_dims
        ]

        # Get remaining custom dimensions not used for faceting
        remaining_dims = [d for d in custom_dims if d not in faceted_dims]

        # When auto_facet is disabled, use line_dash to differentiate custom dims
        # (only for line charts)
        if not auto_facet:
            if has_manual_faceting:
                # Manual faceting provided: apply remaining dims via line styling
                if supports_line_styling and remaining_dims:
                    if "line_dash" not in plotly_kwargs:
                        plotly_kwargs["line_dash"] = remaining_dims.pop()

                    # Check if there are still unrepresentable dimensions
                    if len(remaining_dims) > 0:
                        raise ValueError(
                            f"Too many custom dimensions ({len(custom_dims)}) for this plot type. "
                            f"Faceted dimensions: {faceted_dims}. "
                            f"Remaining dimensions that cannot be represented: {remaining_dims}. "
                            "Please filter the data or use additional faceting if available."
                        )
            else:
                # No manual faceting, auto_facet=False
                if supports_line_styling:
                    # With color and line_dash we can handle channel + 1 custom dim
                    # If there are 2+ custom dims, raise error
                    if len(custom_dims) >= 2:
                        raise ValueError(
                            f"Too many custom dimensions ({len(custom_dims)}) to display "
                            "without faceting. Please use facet_row or facet_col,"
                            " or enable auto_facet=True."
                        )
                    # if there is only one custom dimension, use line_dash
                    elif "line_dash" not in plotly_kwargs:
                        plotly_kwargs["line_dash"] = custom_dims[0]
                else:
                    # Bar charts: can only handle 1 custom dim via color
                    # If there are 2+ custom dims, raise error
                    if len(custom_dims) >= 2:
                        raise ValueError(
                            f"Too many custom dimensions ({len(custom_dims)}) to display "
                            "without faceting. Please use facet_row or facet_col,"
                            " or enable auto_facet=True."
                        )

            return plotly_kwargs

        # Don't override explicit faceting, but still apply line_dash for remaining dims
        if has_manual_faceting:
            # Apply line_dash for remaining custom dimensions if supported
            if (
                supports_line_styling
                and remaining_dims
                and "line_dash" not in plotly_kwargs
            ):
                plotly_kwargs["line_dash"] = remaining_dims[0]
            return plotly_kwargs

        # Apply faceting based on number of dimensions
        if len(custom_dims) == 1:
            facet_key = f"facet_{single_dim_facet}"
            plotly_kwargs[facet_key] = custom_dims[0]
            # Only facet_col_wrap is supported by Plotly (not facet_row_wrap)
            if single_dim_facet == "col":
                plotly_kwargs.setdefault("facet_col_wrap", 3)
        elif len(custom_dims) >= 2:
            plotly_kwargs["facet_row"] = custom_dims[0]
            plotly_kwargs["facet_col"] = custom_dims[1]

            # For line charts, we can use line_dash for a 3rd dimension
            if len(custom_dims) >= 3 and supports_line_styling:
                if "line_dash" not in plotly_kwargs:
                    plotly_kwargs["line_dash"] = custom_dims[2]

            # Determine max supported dimensions based on plot type
            max_supported = 3 if supports_line_styling else 2

            if len(custom_dims) > max_supported:
                raise ValueError(
                    f"Too many custom dimensions ({len(custom_dims)}) for this plot type. "
                    f"Maximum supported: {max_supported} "
                    f"(facet_row, facet_col{', line_dash' if supports_line_styling else ''}). "
                    "Please filter the data to reduce dimensions."
                )

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

    def _prepare_summaries_for_bar_plot(
        self,
        df: IntoDataFrameT,
        *,
        required_column: str,
        default_title: str,
        auto_facet: bool,
        single_dim_facet: Literal["col", "row"],
        plotly_kwargs: dict,
    ) -> tuple[nw.DataFrame, dict]:
        """Prepare summary data for bar plot.

        This helper method handles common preparation logic shared between
        contributions() and roas() plotting methods.

        Parameters
        ----------
        df : IntoDataFrameT
            Summary data from summary factory
        required_column : str
            Column that must be present as x or color (e.g., "channel", component)
        default_title : str
            Default title if not provided in plotly_kwargs
        auto_facet : bool
            Whether to automatically detect and apply faceting
        single_dim_facet : {"col", "row"}
            Direction for single dimension auto-faceting
        plotly_kwargs : dict
            Plotly keyword arguments (will be modified with defaults)

        Returns
        -------
        tuple[nw.DataFrame, dict]
            Prepared narwhals DataFrame and updated plotly_kwargs

        Raises
        ------
        ValueError
            If date column exists with >1 unique values but not in x/color,
            or if required_column not in x/color.
        """
        # Auto-detect faceting from custom dimensions
        plotly_kwargs = self._apply_auto_faceting(
            plotly_kwargs, auto_facet, single_dim_facet
        )

        # Set default values if not provided
        plotly_kwargs.setdefault("title", default_title)
        plotly_kwargs.setdefault("x", required_column)

        nw_df = nw.from_native(df)

        # Validate date column usage if it exists
        x = plotly_kwargs.get("x")
        color = plotly_kwargs.get("color")

        if "date" in nw_df.columns and len(nw_df["date"].unique()) > 1:
            if "date" not in [x, color]:
                raise ValueError("choose either x='date' or color='date'")

        if required_column not in [x, color]:
            raise ValueError(
                f"choose either x='{required_column}' or color='{required_column}'"
            )

        return nw_df, plotly_kwargs

    def contributions(
        self,
        hdi_prob: float | None = 0.94,
        component: ComponentType = "channel",
        frequency: Frequency | None = None,
        round_digits: int = 0,
        auto_facet: bool = True,
        single_dim_facet: Literal["col", "row"] = "col",
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
        auto_facet : bool, default True
            Automatically detect and apply faceting for custom dimensions.
        single_dim_facet : {"col", "row"}, default "col"
            When auto_facet is enabled and there is exactly one custom dimension,
            this controls whether it is applied as facet_col or facet_row.
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

        nw_df, plotly_kwargs = self._prepare_summaries_for_bar_plot(
            df=df,
            required_column=component,
            default_title=f"{component.capitalize()} Contributions",
            auto_facet=auto_facet,
            single_dim_facet=single_dim_facet,
            plotly_kwargs=plotly_kwargs,
        )

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
        auto_facet: bool = True,
        single_dim_facet: Literal["col", "row"] = "col",
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
        auto_facet : bool, default True
            Automatically detect and apply faceting for custom dimensions.
        single_dim_facet : {"col", "row"}, default "col"
            When auto_facet is enabled and there is exactly one custom dimension,
            this controls whether it is applied as facet_col or facet_row.
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

        nw_df, plotly_kwargs = self._prepare_summaries_for_bar_plot(
            df=df,
            required_column="channel",
            default_title="Return on Ad Spend",
            auto_facet=auto_facet,
            single_dim_facet=single_dim_facet,
            plotly_kwargs=plotly_kwargs,
        )

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
        hdi_opacity: float = 0.2,
        frequency: Frequency | None = None,
        auto_facet: bool = True,
        single_dim_facet: Literal["col", "row"] = "row",
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
        hdi_opacity : float, default 0.2
            Opacity for HDI band fill (0-1).
        frequency : str, optional
            Time aggregation (e.g., "monthly", "weekly"). None = no aggregation.
        auto_facet : bool, default True
            Automatically detect and apply faceting for custom dimensions.
        single_dim_facet : {"col", "row"}, default "row"
            When auto_facet is enabled and there is exactly one custom dimension,
            this controls whether it is applied as facet_col or facet_row.
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
        plotly_kwargs = self._apply_auto_faceting(
            plotly_kwargs, auto_facet, single_dim_facet, supports_line_styling=True
        )

        # Convert to Narwhals for unified API
        nw_df = nw.from_native(df)

        # Validate required columns
        required_cols = {"date", "mean", "observed"}
        if not required_cols.issubset(set(nw_df.columns)):
            raise ValueError(
                f"DataFrame missing required columns: {required_cols - set(nw_df.columns)}"
            )

        # Sort by date for proper line plotting
        nw_df = nw_df.sort("date")

        # Create long-format DataFrame with both predicted and observed
        id_cols = ["date", *self.custom_dims]
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

        # set hover data
        hover_data = {}
        for facet in ["facet_row", "facet_col"]:
            if plotly_kwargs.get(facet):
                hover_data[plotly_kwargs.get(facet)] = False  # type: ignore[index]

        # plot observed and predicted lines
        fig = px.line(
            plot_df.to_native(),
            x="date",
            y="value",
            color="series",
            labels={"value": "Value", "date": "Date", "series": ""},
            hover_data=hover_data,
            **plotly_kwargs,
        )

        # Extract facet params for HDI band logic
        facet_row = plotly_kwargs.get("facet_row")
        facet_col = plotly_kwargs.get("facet_col")

        # Add HDI band if requested
        if hdi_prob is not None:
            lower_col, upper_col = self._get_hdi_columns(nw_df, hdi_prob)
            self._add_hdi_bands(
                fig,
                nw_df,
                x="date",
                lower_col=lower_col,
                upper_col=upper_col,
                facet_row=facet_row,
                facet_col=facet_col,
                hdi_prob=hdi_prob,
                hdi_opacity=hdi_opacity,
            )

        # Clean up facet titles
        if facet_row or facet_col:
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        fig.update_layout(hovermode="x")
        return fig

    def _add_single_hdi_band(
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
        """Add a single HDI band to figure as filled area.

        This is a low-level helper used internally by `_add_hdi_bands`.

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
            Subplot row (for faceted plots).
            For some reason, in Plotly Express, row 1 is the bottom row.
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

        # Determine fill color with opacity
        if fillcolor:
            # Convert hex color to rgba with opacity
            if fillcolor.startswith("#"):
                fill = _hex_to_rgba(fillcolor, opacity)
            else:
                # Assume it's already an rgba or other format
                fill = fillcolor
        else:
            # Default blue color
            fill = f"rgba(65,105,225,{opacity})"

        trace = go.Scatter(
            x=x_concat,
            y=y_concat,
            mode="lines",
            fill="toself",
            fillcolor=fill,
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="none",
            name=name,
            showlegend=showlegend,
        )

        if row is not None and col is not None:
            # Note: Passing row=1, col=1 works even for non-faceted plots because
            # Plotly Express (px.line, px.bar) creates figures with an internal
            # _grid_ref structure, unlike plain go.Figure() which would raise an error.
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)

    def _add_hdi_bands(
        self,
        fig: go.Figure,
        nw_df,
        x: str,
        lower_col: str,
        upper_col: str,
        facet_row: str | None = None,
        facet_col: str | None = None,
        hdi_prob: float | None = None,
        color: str | None = None,
        color_values: list | None = None,
        color_map: dict[str, str] | None = None,
        line_dash: str | None = None,
        line_dash_values: list | None = None,
        hdi_opacity: float = 0.2,
    ) -> None:
        """Add HDI bands to a plot, handling both faceted and non-faceted cases.

        This method supports two modes:
        1. Without color: Adds a single HDI band (per facet if faceted) with legend
           showing HDI probability
        2. With color: Adds HDI bands for each color value (per facet if faceted)
           with matching colors

        When line_dash is provided, separate HDI bands are created for each
        line_dash value to avoid mixing data from different line styles.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure
        nw_df : nw.DataFrame
            Narwhals DataFrame with data
        x : str
            Column name for x-axis
        lower_col : str
            Name of lower bound column
        upper_col : str
            Name of upper bound column
        facet_row : str or None, optional
            Column name used for row facets
        facet_col : str or None, optional
            Column name used for column facets
        hdi_prob : float, optional
            HDI probability for legend name (used when color is None)
        color : str, optional
            Column name used for color encoding
        color_values : list, optional
            List of unique color values (required if color is provided)
        color_map : dict[str, str], optional
            Mapping from color values to hex color codes (required if color is provided)
        line_dash : str, optional
            Column name used for line_dash encoding (3rd custom dimension)
        line_dash_values : list, optional
            List of unique line_dash values (required if line_dash is provided)
        hdi_opacity : float, default 0.2
            Opacity for HDI band fill (0-1)
        """
        # Get unique facet combinations
        facet_dims = []
        if facet_row:
            facet_dims.append(facet_row)
        if facet_col:
            facet_dims.append(facet_col)

        # Track if we've shown the legend (for non-color mode)
        legend_shown = False

        if not facet_dims:
            # Non-faceted case: single iteration with row=1, col=1
            facet_combinations: list[dict] = [{}]
        else:
            # Faceted case: get unique combinations using Narwhals
            facet_df = nw_df.select(*facet_dims).unique()
            facet_combinations = list(facet_df.iter_rows(named=True))

        for row_dict in facet_combinations:
            # Build filter expression for facet (only if there are facet dimensions)
            filter_conditions = [nw.col(dim) == val for dim, val in row_dict.items()]

            # Filter data for this facet (or use full data if no facets)
            if filter_conditions:
                filter_expr = filter_conditions[0]
                for cond in filter_conditions[1:]:
                    filter_expr = filter_expr & cond
                facet_data = nw_df.filter(filter_expr)
            else:
                facet_data = nw_df

            # Determine subplot indices (1-based for Plotly)
            # Handle row index
            if facet_row:
                row_val = row_dict[facet_row]
                # For some reason, when faceting in Plotly Express row 1 is the bottom row,
                # so we need to reverse the index.
                row_vals = sorted(
                    nw_df.get_column(facet_row).unique().to_list(), reverse=True
                )
                row_idx = row_vals.index(row_val) + 1
            else:
                row_idx = 1

            # Handle column index
            if facet_col:
                col_val = row_dict[facet_col]
                col_vals = sorted(nw_df.get_column(facet_col).unique().to_list())
                col_idx = col_vals.index(col_val) + 1
            else:
                col_idx = 1

            if color is not None and color_values is not None and color_map is not None:
                # Color mode: add HDI band for each color value in this facet
                for color_val in color_values:
                    # Filter data for this color within this facet
                    facet_color_data = facet_data.filter(nw.col(color) == color_val)

                    # Skip if no data for this combination
                    if len(facet_color_data) == 0:
                        continue

                    # Determine data subsets to add HDI bands for
                    if line_dash is not None and line_dash_values is not None:
                        data_subsets = [
                            facet_color_data.filter(nw.col(line_dash) == ld_val)
                            for ld_val in line_dash_values
                        ]
                    else:
                        data_subsets = [facet_color_data]

                    for band_data in data_subsets:
                        if len(band_data) == 0:
                            continue

                        self._add_single_hdi_band(
                            fig,
                            x=band_data.get_column(x).to_list(),
                            lower=band_data.get_column(lower_col).to_list(),
                            upper=band_data.get_column(upper_col).to_list(),
                            name=f"{color_val} HDI",
                            fillcolor=color_map[color_val],
                            opacity=hdi_opacity,
                            showlegend=False,
                            row=row_idx,
                            col=col_idx,
                        )
            else:
                # Non-color mode: add single HDI band per facet
                self._add_single_hdi_band(
                    fig,
                    x=facet_data.get_column(x).to_list(),
                    lower=facet_data.get_column(lower_col).to_list(),
                    upper=facet_data.get_column(upper_col).to_list(),
                    name=f"{int(hdi_prob * 100)}% HDI" if hdi_prob else "HDI",
                    opacity=hdi_opacity,
                    showlegend=not legend_shown,  # Only show once in legend
                    row=row_idx,
                    col=col_idx,
                )
                legend_shown = True

    def saturation_curves(
        self,
        hdi_prob: float | None = None,
        hdi_opacity: float = 0.2,
        max_value: float = 1.0,
        num_points: int = 100,
        num_samples: int | None = 500,
        random_state: int | None = None,
        auto_facet: bool = True,
        single_dim_facet: Literal["col", "row"] = "col",
        original_scale: bool = True,
        **plotly_kwargs,
    ) -> go.Figure:
        """Plot saturation curves by channel.

        Creates an interactive Plotly line chart showing saturation response
        curves for each channel, with optional HDI uncertainty bands and faceting
        for multi-dimensional models.

        Parameters
        ----------
        hdi_prob : float or None, optional
            HDI probability for uncertainty bands. If None (default), no bands.
        hdi_opacity : float, default 0.2
            Opacity for HDI band fill (0-1).
        max_value : float, default 1.0
            Maximum value for curve x-axis (in scaled space)
        num_points : int, default 100
            Number of points along the x-axis to evaluate curves at
        num_samples : int or None, optional
            Number of posterior samples to use for generating curves. By default 500.
            Using fewer samples speeds up computation and reduces memory usage while
            still capturing posterior uncertainty.
            If None, all posterior samples are used without subsampling.
        random_state : int, np.random.Generator, or None, optional
            Random state for reproducible subsampling. Can be an integer seed,
            a numpy Generator instance, or None for non-reproducible sampling.
            Only used when num_samples is not None and less than total available
            samples.
        auto_facet : bool, default True
            Automatically detect and apply faceting for custom dimensions.
        single_dim_facet : {"col", "row"}, default "col"
            When auto_facet is enabled and there is exactly one custom dimension,
            this controls whether it is applied as facet_col or facet_row.
        original_scale : bool, default True
            Whether to plot x-axis in original scale. If True (default), the x-axis
            values are multiplied by the channel scale factor to show spend in
            original units (e.g., dollars). If False, x-axis shows scaled values.
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
            num_samples=num_samples,
            random_state=random_state,
        )
        nw_df = nw.from_native(df)

        # Convert x-axis to original scale if requested
        if original_scale:
            # Get channel scale factors and convert to DataFrame
            channel_scale = self.summary.data.get_channel_scale()
            scale_df = channel_scale.to_dataframe(name="channel_scale").reset_index()
            nw_scale = nw.from_native(scale_df)

            # Determine join columns: "channel" plus any custom dimensions
            # that are present in both the scale DataFrame and the data
            join_cols = ["channel"]
            for dim in self.custom_dims:
                if dim in nw_scale.columns and dim in nw_df.columns:
                    join_cols.append(dim)

            # Join scale factors with data and multiply x by scale
            nw_df = nw_df.join(nw_scale, on=join_cols, how="left")
            nw_df = nw_df.with_columns(x=nw.col("x") * nw.col("channel_scale"))
            nw_df = nw_df.drop("channel_scale")

            xaxis_title = "Spend"
        else:
            xaxis_title = "Spend (scaled)"

        # Auto-detect faceting from custom dimensions
        plotly_kwargs = self._apply_auto_faceting(
            plotly_kwargs, auto_facet, single_dim_facet, supports_line_styling=True
        )

        return self._plot_curves(
            df=nw_df.to_native(),
            x="x",
            hdi_prob=hdi_prob,
            hdi_opacity=hdi_opacity,
            title="Saturation Curves",
            xaxis_title=xaxis_title,
            yaxis_title="Response",
            **plotly_kwargs,
        )

    def adstock_curves(
        self,
        hdi_prob: float | None = None,
        hdi_opacity: float = 0.2,
        amount: float = 1.0,
        num_samples: int | None = 500,
        random_state: int | None = None,
        auto_facet: bool = True,
        single_dim_facet: Literal["col", "row"] = "col",
        **plotly_kwargs,
    ) -> go.Figure:
        """Plot adstock/decay curves by channel.

        Creates an interactive Plotly line chart showing adstock decay curves
        for each channel, with optional HDI uncertainty bands and faceting for
        multi-dimensional models.

        Parameters
        ----------
        hdi_prob : float or None, optional
            HDI probability for uncertainty bands. If None (default), no bands.
        hdi_opacity : float, default 0.2
            Opacity for HDI band fill (0-1).
        amount : float, default 1.0
            Impulse amount at time 0
        num_samples : int or None, optional
            Number of posterior samples to use for generating curves. By default 500.
            Using fewer samples speeds up computation and reduces memory usage while
            still capturing posterior uncertainty.
            If None, all posterior samples are used without subsampling.
        random_state : int, np.random.Generator, or None, optional
            Random state for reproducible subsampling. Can be an integer seed,
            a numpy Generator instance, or None for non-reproducible sampling.
            Only used when num_samples is not None and less than total available
            samples.
        auto_facet : bool, default True
            Automatically detect and apply faceting for custom dimensions.
        single_dim_facet : {"col", "row"}, default "col"
            When auto_facet is enabled and there is exactly one custom dimension,
            this controls whether it is applied as facet_col or facet_row.
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
            num_samples=num_samples,
            random_state=random_state,
        )

        # Auto-detect faceting from custom dimensions
        plotly_kwargs = self._apply_auto_faceting(
            plotly_kwargs, auto_facet, single_dim_facet, supports_line_styling=True
        )

        return self._plot_curves(
            df=df,
            x="time since exposure",
            hdi_prob=hdi_prob,
            hdi_opacity=hdi_opacity,
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
        hdi_opacity: float = 0.2,
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
        hdi_opacity : float, default 0.2
            Opacity for HDI band fill (0-1)
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
        nw_df = nw.from_native(df)

        # Sort by x column for proper line plotting
        nw_df = nw_df.sort(x)

        # Extract facet params for HDI band logic
        facet_row = plotly_kwargs.get("facet_row")
        facet_col = plotly_kwargs.get("facet_col")
        line_dash = plotly_kwargs.get("line_dash")

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
            # Sort to ensure consistent ordering with Plotly's color assignment
            color_values = sorted(nw_df.get_column(color).unique().to_list())

            # Create color map matching Plotly's default color assignment
            color_map = {
                val: PLOTLY_COLORS[i % len(PLOTLY_COLORS)]
                for i, val in enumerate(color_values)
            }

            # Get line_dash values if line_dash is used
            line_dash_values = None
            if line_dash is not None:
                line_dash_values = sorted(
                    nw_df.get_column(line_dash).unique().to_list()
                )

            self._add_hdi_bands(
                fig,
                nw_df,
                x=x,
                lower_col=lower_col,
                upper_col=upper_col,
                facet_row=facet_row,
                facet_col=facet_col,
                color=color,
                color_values=color_values,
                color_map=color_map,
                line_dash=line_dash,
                line_dash_values=line_dash_values,
                hdi_opacity=hdi_opacity,
            )

        # Clean facet titles
        if facet_row or facet_col:
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        fig.update_layout(hovermode="x")
        return fig
