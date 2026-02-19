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
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

COLORS = px.colors.qualitative.Plotly


def _get_hdi_columns(df: pl.DataFrame, hdi_prob: float) -> tuple[str, str]:
    """Get the HDI column names for a given probability level.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing HDI columns.
    hdi_prob : float
        HDI probability level (e.g., 0.80, 0.90, 0.95).

    Returns
    -------
    tuple[str, str]
        Tuple of (lower_col, upper_col) containing absolute HDI bounds.

    Raises
    ------
    ValueError
        If the HDI columns for the given probability level are not found.
    """
    prob_str = str(int(hdi_prob * 100))
    lower_col = f"abs_error_{prob_str}_lower"
    upper_col = f"abs_error_{prob_str}_upper"

    if lower_col in df.columns and upper_col in df.columns:
        return lower_col, upper_col
    raise ValueError(
        f"HDI columns for probability {hdi_prob} not found. Expected columns: {lower_col}, {upper_col}"
    )


def _plot_hdi_band(
    fig: go.Figure,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    name: str = "HDI",
    fillcolor: str = COLORS[0],
    opacity: float = 0.2,
    showlegend: bool = True,
) -> None:
    """Add an HDI band to a plotly figure.

    Parameters
    ----------
    fig : go.Figure
        The plotly figure to add the band to.
    x : np.ndarray
        X-axis values.
    lower : np.ndarray
        Lower bound values.
    upper : np.ndarray
        Upper bound values.
    name : str
        Name for the legend.
    fillcolor : str
        Fill color for the band.
    opacity : float
        Opacity of the fill.
    showlegend : bool
        Whether to show in legend.
    """
    x_concat = np.concatenate([x, x[::-1]])
    y_concat = np.concatenate([lower, upper[::-1]])
    fig.add_trace(
        go.Scatter(
            x=x_concat,
            y=y_concat,
            mode="lines",
            fill="toself",
            fillcolor=f"rgba({_hex_to_rgb(fillcolor)},{opacity})"
            if fillcolor.startswith("#") or fillcolor in COLORS
            else fillcolor,
            line_color="rgba(255,255,255,0)",
            hoverinfo="none",
            name=name,
            showlegend=showlegend,
        )
    )


def _hex_to_rgb(color: str) -> str:
    """Convert a color name or hex to RGB string for rgba()."""
    # Map common color names to RGB
    color_map = {
        "royalblue": "65,105,225",
        "red": "255,0,0",
        "green": "0,128,0",
        "blue": "0,0,255",
    }
    if color.lower() in color_map:
        return color_map[color.lower()]

    # Handle hex colors (e.g., "#FF5733" or "#abc")
    if color.startswith("#"):
        hex_color = color.lstrip("#")
        # Handle both 3-digit and 6-digit hex
        if len(hex_color) == 3:
            # Expand 3-digit hex to 6-digit (e.g., "abc" -> "aabbcc")
            hex_color = "".join([c * 2 for c in hex_color])
        # Convert hex to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"{r},{g},{b}"

    # For plotly colors, just return a default
    return "65,105,225"


def _determine_date_format(dates: pd.Series) -> str:
    """Determine the most appropriate date format based on the date series.

    Analyzes the date series to determine if dates vary by day, month, quarter, or year,
    and returns the appropriate format string.

    Parameters
    ----------
    dates : pd.Series
        Series of datetime values.

    Returns
    -------
    str
        Format string: '%Y-%m-%d', '%Y-%m', 'quarter', or '%Y'
    """
    if len(dates) < 2:
        return "%Y-%m-%d"

    # Get unique dates sorted
    unique_dates = pd.Series(dates.unique()).sort_values()

    # Calculate differences between consecutive dates
    diffs = unique_dates.diff().dropna()

    # Get the most common difference (mode)
    if len(diffs) == 0:
        return "%Y-%m-%d"

    # Check if all dates are on the same day of month (monthly pattern)
    if len(unique_dates) > 1:
        median_diff_days = diffs.median().total_seconds() / (24 * 3600)

        day_of_month = unique_dates.dt.day
        if day_of_month.nunique() == 1:
            # Check if it's monthly (approximately 28-31 days apart)
            if 28 <= median_diff_days <= 31:
                return "%Y-%m"

        # Check if it's quarterly (approximately 90-92 days apart)
        if 90 <= median_diff_days <= 92:
            # For quarterly, we'll format as YYYY-QQ manually
            return "quarter"

        # Check if it's yearly (approximately 365 days apart)
        if 365 <= median_diff_days <= 366:
            return "%Y"

    # Default to daily format
    return "%Y-%m-%d"


def _format_date_for_legend(date_val, format_str: str) -> str:
    """Format a single date value according to the format string.

    Parameters
    ----------
    date_val : datetime-like
        The date value to format.
    format_str : str
        Format string or 'quarter' for quarterly formatting.

    Returns
    -------
    str
        Formatted date string.
    """
    if format_str == "quarter":
        # Format as YYYY-QQ
        year = pd.Timestamp(date_val).year
        quarter = (pd.Timestamp(date_val).month - 1) // 3 + 1
        return f"{year}-Q{quarter}"
    else:
        return pd.Timestamp(date_val).strftime(format_str)


def format_date_column_for_legend(df: pd.DataFrame, date_column: str) -> pd.Series:
    """Format a date column for use in plot legends.

    Determines the most appropriate date format based on the date series granularity
    and returns a formatted series suitable for legend labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the date column.
    date_column : str
        Name of the date column to format.

    Returns
    -------
    pd.Series
        Series with formatted date strings (YYYY-MM-DD, YYYY-MM, YYYY-QQ, or YYYY).

    Raises
    ------
    ValueError
        If the column is not found or is not a datetime type.
    """
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame")

    date_col = df[date_column]
    if not pd.api.types.is_datetime64_any_dtype(date_col):
        raise ValueError(f"Column '{date_column}' is not a datetime type")

    # Determine appropriate format
    date_format = _determine_date_format(date_col)

    # Format dates
    return date_col.apply(lambda d: _format_date_for_legend(d, date_format))


def _prepare_color_column_for_plot(df: pd.DataFrame, color: str | None) -> str | None:
    """Prepare a color column for plotting by formatting dates if needed.

    If the color column is a datetime type, formats it appropriately for legend display.
    Otherwise, returns the original color column name.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the color column.
    color : str | None
        Name of the color column, or None if no color column.

    Returns
    -------
    str | None
        Name of the color column to use for plotting (may be formatted version),
        or None if color was None.
    """
    if color is None or color not in df.columns:
        return color

    color_col = df[color]
    # Check if it's a datetime column
    if pd.api.types.is_datetime64_any_dtype(color_col):
        # Format dates for legend
        df[f"{color}_formatted"] = format_date_column_for_legend(df, color)
        return f"{color}_formatted"

    return color


def plot_posterior_predictive(
    posterior_predictive_df: pl.DataFrame,
    hdi_prob: float | None = 0.90,
) -> go.Figure:
    """Plot posterior predictive distributions over time.

    Parameters
    ----------
    posterior_predictive_df : pl.DataFrame
        DataFrame with columns: date, mean, observed, and HDI columns.
        can be MMMPlotData.posterior_predictive
    hdi_prob : float | None
        HDI probability level to plot. If None, no HDI band is shown.

    Returns
    -------
    go.Figure
    """
    # Convert to pandas for plotly compatibility
    pdf = posterior_predictive_df.to_pandas()

    # Melt data for px.line
    melted = pdf.melt(
        id_vars=["date"],
        value_vars=["mean", "observed"],
        var_name="variable",
        value_name="value",
    )
    # Rename for better legend labels
    melted["variable"] = melted["variable"].replace(
        {"mean": "Predicted", "observed": "Observed"}
    )

    # Create color mapping
    color_map = {"Predicted": COLORS[0], "Observed": "rgba(0,0,0, 0.7)"}

    fig = px.line(
        melted,
        x="date",
        y="value",
        color="variable",
        color_discrete_map=color_map,
    )

    # Add HDI band
    if hdi_prob is not None:
        lower_col, upper_col = _get_hdi_columns(posterior_predictive_df, hdi_prob)
        _plot_hdi_band(
            fig,
            pdf["date"].values,
            pdf[lower_col].values,
            pdf[upper_col].values,
            name=f"HDI {hdi_prob}",
            fillcolor=COLORS[0],
            opacity=0.6,
        )

    fig.update_layout(hovermode="x")
    return fig


def plot_curves(
    curves_df: pl.DataFrame,
    x: str,
    yaxis_title: str,
    hdi_prob: float | None = None,
) -> go.Figure:
    """Plot saturation curves.

    Parameters
    ----------
    curves_df : pl.DataFrame
        DataFrame with columns: x, channel, saturation_curve, and optionally HDI columns.
    x : str
        Column to use for x-axis.
    hdi_prob : float | None
        HDI probability level to plot. If None, no HDI bands are shown.


    Returns
    -------
    go.Figure
    """
    pdf = curves_df.to_pandas()

    # Get unique channels in consistent order and create color mapping
    channels = pdf["channel"].unique()
    color_map = {channel: COLORS[i % len(COLORS)] for i, channel in enumerate(channels)}

    fig = px.line(pdf, x=x, y="mean", color="channel", color_discrete_map=color_map)

    fig.update_layout(
        hovermode="x",
        yaxis_title=yaxis_title,
    )

    if hdi_prob is not None:
        lower_col, upper_col = _get_hdi_columns(curves_df, hdi_prob)
        for channel_key, channel_data in curves_df.group_by("channel"):
            channel_name = channel_key[0]
            _plot_hdi_band(
                fig,
                channel_data[x],
                channel_data[lower_col],
                channel_data[upper_col],
                name=f"HDI {hdi_prob}",
                fillcolor=color_map[channel_name],
                opacity=0.2,
                showlegend=False,
            )

    return fig


def plot_saturation_curves(
    curves_df: pl.DataFrame,
    hdi_prob: float | None = None,
) -> go.Figure:
    """Plot curves."""
    return plot_curves(curves_df, x="x", yaxis_title="Revenue", hdi_prob=hdi_prob)


def plot_decay_curves(
    decay_df: pl.DataFrame,
    hdi_prob: float | None = None,
) -> go.Figure:
    """Plot decay curves."""
    return plot_curves(decay_df, x="time", yaxis_title="Revenue", hdi_prob=hdi_prob)


def plot_bar(
    df: pl.DataFrame,
    hdi_prob: float | None = None,
    x: str | None = "channel",
    color: str | None = None,
    yaxis_title: str | None = None,
    **kwargs,
) -> go.Figure:
    """Plot bar chart.

    Can be used to plot ROAS, contribution, etc.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with columns: channel, ROAs.
        can be MMMPlotData.roas or MMMPlotData.contribution
    hdi_prob : float | None
        HDI probability level for error bars. If None, no error bars are shown.
    x : str | None
        Column to use for x-axis. Defaults to "channel".
    color : str | None
        Column to use for coloring bars.
    yaxis_title : str | None
        Title of the plot.

    Returns
    -------
    go.Figure
    """
    pdf = df.to_pandas()

    error_y = None
    error_y_minus = None

    if hdi_prob is not None:
        lower_col, upper_col = _get_hdi_columns(df, hdi_prob)
        # Compute relative errors: abs_error - mean_value
        pdf["error_upper"] = pdf[upper_col] - pdf["mean"]
        pdf["error_lower"] = pdf["mean"] - pdf[lower_col]
        error_y = "error_upper"
        error_y_minus = "error_lower"

    # Handle date formatting for color column
    color_col_formatted = _prepare_color_column_for_plot(pdf, color)

    fig = px.bar(
        pdf,
        x=x,
        y="mean",
        color=color_col_formatted,
        barmode="group",
        error_y=error_y,
        error_y_minus=error_y_minus,
        **kwargs,
    )

    # fig.update_layout(yaxis_title=yaxis_title)
    return fig


def plot_total_contribution_bar(
    contribution_df: pl.DataFrame,
) -> go.Figure:
    """Plot total contribution as a bar chart."""
    pdf = contribution_df.to_pandas()

    # Handle date formatting for color column
    color_col_formatted = _prepare_color_column_for_plot(pdf, "channel")

    fig = px.bar(
        pdf,
        x="date",
        y="mean",
        color=color_col_formatted,
        barmode="stack",
    )

    fig.update_layout(yaxis_title="Total Contribution")
    return fig


def plot_channel_spend_bar(
    df: pl.DataFrame,
    x: str | None = "channel",
    color: str | None = None,
) -> go.Figure:
    """Plot channel spend as a bar chart.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with columns: date, channel, channel_data.
    x : str | None
        Column to use for x-axis. Defaults to "channel".
    color : str | None
        Column to use for coloring bars.

    Returns
    -------
    go.Figure
    """
    pdf = df.to_pandas()

    # Handle date formatting for color column
    color_col_formatted = _prepare_color_column_for_plot(pdf, color)

    fig = px.bar(
        pdf,
        x=x,
        y="channel_data",
        color=color_col_formatted,
        barmode="group",
    )
    fig.update_layout(yaxis_title="Channel Spend")
    return fig


def plot_contribution_vs_roas(
    contribution_df: pl.DataFrame,
    roas_df: pl.DataFrame,
    hdi_prob: float | None = None,
) -> go.Figure:
    """Plot contribution vs ROAS scatter plot.

    Parameters
    ----------
    contribution_df : pl.DataFrame
        DataFrame with columns: channel, contribution, and optionally HDI columns.
    roas_df : pl.DataFrame
        DataFrame with columns: channel, ROAs, and optionally HDI columns.
    hdi_prob : float | None
        HDI probability level for error bars. If None, no error bars are shown.

    Returns
    -------
    go.Figure
    """
    contrib_pdf = contribution_df.to_pandas()
    roas_pdf = roas_df.to_pandas()

    # Prepare error columns for contribution
    contrib_error_y = None
    contrib_error_y_minus = None

    if hdi_prob is not None:
        lower_col, upper_col = _get_hdi_columns(contribution_df, hdi_prob)
        # Compute relative errors: abs_error - mean_value
        contrib_pdf["contribution_error_upper"] = (
            contrib_pdf[upper_col] - contrib_pdf["mean"]
        )
        contrib_pdf["contribution_error_lower"] = (
            contrib_pdf["mean"] - contrib_pdf[lower_col]
        )
        contrib_pdf.rename(columns={"mean": "contribution"}, inplace=True)
        contrib_pdf = contrib_pdf[
            [
                "channel",
                "contribution",
                "contribution_error_upper",
                "contribution_error_lower",
            ]
        ]
        contrib_error_y = "contribution_error_upper"
        contrib_error_y_minus = "contribution_error_lower"

    # Prepare error columns for ROAS
    roas_error_x = None
    roas_error_x_minus = None

    if hdi_prob is not None:
        lower_col, upper_col = _get_hdi_columns(roas_df, hdi_prob)
        # Compute relative errors: abs_error - mean_value
        roas_pdf["roas_error_upper"] = roas_pdf[upper_col] - roas_pdf["mean"]
        roas_pdf["roas_error_lower"] = roas_pdf["mean"] - roas_pdf[lower_col]
        roas_pdf.rename(columns={"mean": "ROAs"}, inplace=True)
        roas_pdf = roas_pdf[["channel", "ROAs", "roas_error_upper", "roas_error_lower"]]
        roas_error_x = "roas_error_upper"
        roas_error_x_minus = "roas_error_lower"

    # Merge dataframes
    merged_pdf = contrib_pdf.merge(roas_pdf, on="channel")

    fig = px.scatter(
        merged_pdf,
        x="ROAs",
        y="contribution",
        color="channel",
        error_y=contrib_error_y,
        error_y_minus=contrib_error_y_minus,
        error_x=roas_error_x,
        error_x_minus=roas_error_x_minus,
    )
    return fig
