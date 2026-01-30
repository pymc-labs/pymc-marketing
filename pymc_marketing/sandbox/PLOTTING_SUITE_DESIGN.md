# MMM Plotting Suite Design Document

## Overview

This document describes the design of the `MMMStakeholderPlotSuite` class, which provides interactive Plotly-based visualization methods for Media Mix Model (MMM) results. The suite transforms standalone plotting functions from `plotly_utils.py` into a cohesive class-based API.

## Class Structure

### Class: `MMMStakeholderPlotSuite`

A plotting suite class that provides methods for visualizing MMM inference results using Plotly. The suite accepts processed Polars DataFrames (typically from `MMMSummaryData`) and generates interactive visualizations.

**Initialization:**

```python
class MMMStakeholderPlotSuite:
    def __init__(self, plot_data: MMMSummaryData | None = None):
        """Initialize the plotting suite.

        Parameters
        ----------
        plot_data : MMMSummaryData | None, optional
            Pre-processed plot data container. If None, individual plot methods
            must receive DataFrames directly.
        """
```

**Class Attributes:**

- `plot_data: MMMSummaryData | None` - Optional pre-processed plot data container
- `COLORS: list[str]` - Default color palette (Plotly qualitative colors)

---

## Public Plotting Methods

### Summary

The `MMMStakeholderPlotSuite` class provides the following public plotting methods:

1. **`posterior_predictive()`** - Plot predicted vs observed values over time with optional HDI bands
2. **`saturation_curves()`** - Plot saturation curves for marketing channels
3. **`decay_curves()`** - Plot adstock decay curves over time for each channel
4. **`marginal_curves()`** - Plot marginal curves for marketing channels
5. **`roas()`** - Bar chart for ROAS by channel with optional error bars
6. **`contribution()`** - Bar chart for contribution by channel with optional error bars
7. **`roas_change_over_time()`** - Bar chart for period-over-period ROAS changes with optional error bars
8. **`total_contribution()`** - Stacked bar chart showing total contribution by channel over time
9. **`channel_spend()`** - Bar chart visualizing marketing spend by channel
10. **`contribution_vs_roas()`** - Scatter plot comparing contribution vs ROAS with error bars

---

### 1. `posterior_predictive`

Plot posterior predictive distributions over time, comparing predicted vs observed values.

**Signature:**

```python
def posterior_predictive(
    self,
    posterior_predictive_df: pl.DataFrame | None = None,
    hdi_prob: float | None = 0.90,
    **kwargs,
) -> go.Figure:
```

**Parameters:**

- `posterior_predictive_df : pl.DataFrame | None`
  - DataFrame with columns: `date`, `mean`, `observed`, and optionally HDI columns.
  - Can be `MMMSummaryData.posterior_predictive` if `plot_data` is set.
  - If `None` and `plot_data` is set, uses `plot_data.posterior_predictive`.

- `hdi_prob : float | None`
  - HDI probability level to plot (e.g., 0.80, 0.90, 0.95).
  - If `None`, no HDI band is shown.
  - Default: `0.90`

- `**kwargs`
  - Additional keyword arguments passed to `fig.update_layout()` for customizing the plot layout.

**Returns:**

- `go.Figure` - Plotly figure with predicted line, observed line, and optional HDI band

**Description:**

Creates a time series plot showing:
- Predicted values (mean posterior predictive) as a line
- Observed values as a line
- Optional HDI uncertainty band around predictions

---

### 2. `saturation_curves`

Plot saturation curves for marketing channels.

**Signature:**

```python
def saturation_curves(
    self,
    curves_df: pl.DataFrame | None = None,
    hdi_prob: float | None = None,
    **kwargs,
) -> go.Figure:
```

**Parameters:**

- `curves_df : pl.DataFrame | None`
  - DataFrame with columns: `x`, `channel`, `mean` (saturation curve values), and optionally HDI columns.
  - If `None` and `plot_data` is set, attempts to use saturation curve data from `plot_data`.

- `hdi_prob : float | None`
  - HDI probability level to plot. If `None`, no HDI bands are shown.
  - Default: `None`

- `**kwargs`
  - Additional keyword arguments passed to `plotly.express.line()` for customizing the plot.

**Returns:**

- `go.Figure` - Plotly figure with saturation curves per channel

**Description:**

Plots saturation curves (revenue vs spend) for each channel. Each channel is displayed in a different color with optional HDI uncertainty bands.

---

### 3. `decay_curves`

Plot adstock decay curves over time.

**Signature:**

```python
def decay_curves(
    self,
    decay_df: pl.DataFrame | None = None,
    hdi_prob: float | None = None,
    **kwargs,
) -> go.Figure:
```

**Parameters:**

- `decay_df : pl.DataFrame | None`
  - DataFrame with columns: `time`, `channel`, `mean` (decay curve values), and optionally HDI columns.
  - If `None` and `plot_data` is set, attempts to use decay curve data from `plot_data`.

- `hdi_prob : float | None`
  - HDI probability level to plot. If `None`, no HDI bands are shown.
  - Default: `None`

- `**kwargs`
  - Additional keyword arguments passed to `plotly.express.line()` for customizing the plot.

**Returns:**

- `go.Figure` - Plotly figure with decay curves per channel

**Description:**

Plots adstock decay curves showing how the effect of marketing spend decays over time for each channel. Each channel is displayed in a different color with optional HDI uncertainty bands.

---

### 4. `marginal_curves`

Plot marginal curves for marketing channels.

**Signature:**

```python
def marginal_curves(
    self,
    curves_df: pl.DataFrame | None = None,
    hdi_prob: float | None = None,
    **kwargs,
) -> go.Figure:
```

**Parameters:**

- `curves_df : pl.DataFrame | None`
  - DataFrame with columns: `x`, `channel`, `mean` (marginal curve values), and optionally HDI columns.
  - If `None` and `plot_data` is set, attempts to use marginal curve data from `plot_data`.

- `hdi_prob : float | None`
  - HDI probability level to plot. If `None`, no HDI bands are shown.
  - Default: `None`

- `**kwargs`
  - Additional keyword arguments passed to `plotly.express.line()` for customizing the plot.

**Returns:**

- `go.Figure` - Plotly figure with marginal curves per channel

**Description:**

Plots marginal curves (marginal effect vs spend) for each channel. Each channel is displayed in a different color with optional HDI uncertainty bands.

---

### 5. `roas`

Bar chart for ROAS by channel.

**Signature:**

```python
def roas(
    self,
    roas_df: pl.DataFrame | None = None,
    hdi_prob: float | None = None,
    x: str | None = "channel",
    color: str | None = None,
    **kwargs,
) -> go.Figure:
```

**Parameters:**

- `roas_df : pl.DataFrame | None`
  - DataFrame with columns: `mean` (ROAS values), and optionally HDI columns.
  - Can be `MMMSummaryData.roas` if `plot_data` is set.
  - If `None` and `plot_data` is set, uses `plot_data.roas`.

- `hdi_prob : float | None`
  - HDI probability level for error bars. If `None`, no error bars are shown.
  - Default: `None`

- `x : str | None`
  - Column to use for x-axis (typically channel names or dates).
  - Default: `"channel"`

- `color : str | None`
  - Column to use for coloring bars (e.g., for grouping by date or other dimension).
  - If the column is datetime type, it will be automatically formatted for legend display.
  - Default: `None`

- `**kwargs`
  - Additional keyword arguments passed to `plotly.express.bar()` for customizing the plot.

**Returns:**

- `go.Figure` - Plotly bar chart figure with ROAS values

**Description:**

Creates a grouped bar chart showing ROAS by channel with optional error bars based on HDI intervals. Supports grouping by a color dimension (e.g., by date or geo).

---

### 6. `contribution`

Bar chart for contribution by channel.

**Signature:**

```python
def contribution(
    self,
    contribution_df: pl.DataFrame | None = None,
    hdi_prob: float | None = None,
    x: str | None = "channel",
    color: str | None = None,
    **kwargs,
) -> go.Figure:
```

**Parameters:**

- `contribution_df : pl.DataFrame | None`
  - DataFrame with columns: `mean` (contribution values), and optionally HDI columns.
  - Can be `MMMSummaryData.contribution` if `plot_data` is set.
  - If `None` and `plot_data` is set, uses `plot_data.contribution`.

- `hdi_prob : float | None`
  - HDI probability level for error bars. If `None`, no error bars are shown.
  - Default: `None`

- `x : str | None`
  - Column to use for x-axis (typically channel names or dates).
  - Default: `"channel"`

- `color : str | None`
  - Column to use for coloring bars (e.g., for grouping by date or other dimension).
  - If the column is datetime type, it will be automatically formatted for legend display.
  - Default: `None`

- `**kwargs`
  - Additional keyword arguments passed to `plotly.express.bar()` for customizing the plot.

**Returns:**

- `go.Figure` - Plotly bar chart figure with contribution values

**Description:**

Creates a grouped bar chart showing contribution by channel with optional error bars based on HDI intervals. Supports grouping by a color dimension (e.g., by date or geo).

---

### 7. `roas_change_over_time`

Bar chart for period-over-period ROAS changes.

**Signature:**

```python
def roas_change_over_time(
    self,
    roas_pop_df: pl.DataFrame | None = None,
    hdi_prob: float | None = None,
    x: str | None = "channel",
    color: str | None = None,
    **kwargs,
) -> go.Figure:
```

**Parameters:**

- `roas_pop_df : pl.DataFrame | None`
  - DataFrame with columns: `mean` (period-over-period ROAS change values), and optionally HDI columns.
  - Can be `MMMSummaryData.roas_change_over_time` if `plot_data` is set.
  - If `None` and `plot_data` is set, uses `plot_data.roas_change_over_time`.

- `hdi_prob : float | None`
  - HDI probability level for error bars. If `None`, no error bars are shown.
  - Default: `None`

- `x : str | None`
  - Column to use for x-axis (typically channel names or dates).
  - Default: `"channel"`

- `color : str | None`
  - Column to use for coloring bars (e.g., for grouping by date or other dimension).
  - If the column is datetime type, it will be automatically formatted for legend display.
  - Default: `None`

- `**kwargs`
  - Additional keyword arguments passed to `plotly.express.bar()` for customizing the plot.

**Returns:**

- `go.Figure` - Plotly bar chart figure with period-over-period ROAS changes

**Description:**

Creates a grouped bar chart showing period-over-period ROAS changes (e.g., year-over-year) by channel with optional error bars based on HDI intervals. Supports grouping by a color dimension (e.g., by date or geo).

---

### 8. `total_contribution`

Plot total contribution as a stacked bar chart over time.

**Signature:**

```python
def total_contribution(
    self,
    contribution_df: pl.DataFrame | None = None,
    **kwargs,
) -> go.Figure:
```

**Parameters:**

- `contribution_df : pl.DataFrame | None`
  - DataFrame with columns: `date`, `channel`, `mean` (contribution values).
  - Can be `MMMSummaryData.contribution` if `plot_data` is set.
  - If `None` and `plot_data` is set, uses `plot_data.contribution`.

- `**kwargs`
  - Additional keyword arguments passed to `plotly.express.bar()` for customizing the plot.

**Returns:**

- `go.Figure` - Plotly stacked bar chart figure

**Description:**

Creates a stacked bar chart showing total contribution by channel over time. Each bar represents a time period, with segments colored by channel to show the breakdown of contributions.

---

### 9. `channel_spend`

Plot channel spend as a bar chart.

**Signature:**

```python
def channel_spend(
    self,
    df: pl.DataFrame | None = None,
    x: str | None = "channel",
    color: str | None = None,
    **kwargs,
) -> go.Figure:
```

**Parameters:**

- `df : pl.DataFrame | None`
  - DataFrame with columns: `date`, `channel`, `channel_data` (spend values).
  - Can be `MMMSummaryData.channel_spend` if `plot_data` is set.
  - If `None` and `plot_data` is set, uses `plot_data.channel_spend`.

- `x : str | None`
  - Column to use for x-axis. Defaults to `"channel"`.
  - Default: `"channel"`

- `color : str | None`
  - Column to use for coloring bars (e.g., for grouping by date).
  - If the column is datetime type, it will be automatically formatted for legend display.
  - Default: `None`

- `**kwargs`
  - Additional keyword arguments passed to `plotly.express.bar()` for customizing the plot.

**Returns:**

- `go.Figure` - Plotly bar chart figure

**Description:**

Creates a bar chart showing marketing spend by channel. Useful for visualizing budget allocation across channels, optionally grouped by time period or other dimensions.

---

### 10. `contribution_vs_roas`

Plot contribution vs ROAS as a scatter plot with error bars.

**Signature:**

```python
def contribution_vs_roas(
    self,
    contribution_df: pl.DataFrame | None = None,
    roas_df: pl.DataFrame | None = None,
    hdi_prob: float | None = None,
    **kwargs,
) -> go.Figure:
```

**Parameters:**

- `contribution_df : pl.DataFrame | None`
  - DataFrame with columns: `channel`, `mean` (contribution), and optionally HDI columns.
  - Can be `MMMSummaryData.contribution` if `plot_data` is set.
  - If `None` and `plot_data` is set, uses `plot_data.contribution`.

- `roas_df : pl.DataFrame | None`
  - DataFrame with columns: `channel`, `mean` (ROAS), and optionally HDI columns.
  - Can be `MMMSummaryData.roas` if `plot_data` is set.
  - If `None` and `plot_data` is set, uses `plot_data.roas`.

- `hdi_prob : float | None`
  - HDI probability level for error bars. If `None`, no error bars are shown.
  - Default: `None`

- `**kwargs`
  - Additional keyword arguments passed to `plotly.express.scatter()` for customizing the plot.

**Returns:**

- `go.Figure` - Plotly scatter plot figure

**Description:**

Creates a scatter plot comparing contribution (y-axis) vs ROAS (x-axis) for each channel. Each point represents a channel, colored by channel name. Optional error bars show uncertainty in both dimensions based on HDI intervals.

---

## Public Utility Methods

### Summary

The `MMMStakeholderPlotSuite` class provides the following public utility methods:

1. **`format_date_column_for_legend()`** - Format date columns for use in plot legends with automatic granularity detection

---

### 8. `format_date_column_for_legend`

Format a date column for use in plot legends.

**Signature:**

```python
@staticmethod
def format_date_column_for_legend(
    df: pd.DataFrame,
    date_column: str,
) -> pd.Series:
```

**Parameters:**

- `df : pd.DataFrame`
  - DataFrame containing the date column.

- `date_column : str`
  - Name of the date column to format.

**Returns:**

- `pd.Series` - Series with formatted date strings (YYYY-MM-DD, YYYY-MM, YYYY-QQ, or YYYY).

**Raises:**

- `ValueError` - If the column is not found or is not a datetime type.

**Description:**

Determines the most appropriate date format based on the date series granularity (daily, monthly, quarterly, or yearly) and returns a formatted series suitable for legend labels.

---

## Private Helper Methods

### Summary

The `MMMStakeholderPlotSuite` class uses the following private helper methods (not intended for direct use):

1. **`_get_hdi_columns()`** - Extract HDI column names for a given probability level
2. **`_plot_hdi_band()`** - Add HDI uncertainty bands to plotly figures
3. **`_hex_to_rgb()`** - Convert color names or hex codes to RGB strings
4. **`_determine_date_format()`** - Detect appropriate date format based on data granularity
5. **`_format_date_for_legend()`** - Format a single date value according to format string
6. **`_prepare_color_column_for_plot()`** - Prepare color columns for plotting with date formatting

---

### `_get_hdi_columns`

Get the HDI column names for a given probability level.

**Signature:**

```python
@staticmethod
def _get_hdi_columns(
    df: pl.DataFrame,
    hdi_prob: float,
) -> tuple[str, str]:
```

**Parameters:**

- `df : pl.DataFrame`
  - DataFrame containing HDI columns.

- `hdi_prob : float`
  - HDI probability level (e.g., 0.80, 0.90, 0.95).

**Returns:**

- `tuple[str, str]` - Tuple of (lower_col, upper_col) containing absolute HDI bounds.

**Raises:**

- `ValueError` - If the HDI columns for the given probability level are not found.

**Description:**

Extracts the column names for HDI bounds based on the probability level. Expected column format: `abs_error_{prob}_lower` and `abs_error_{prob}_upper` where `prob` is the integer percentage (e.g., 90 for 0.90).

---

### `_plot_hdi_band`

Add an HDI band to a plotly figure.

**Signature:**

```python
@staticmethod
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
```

**Parameters:**

- `fig : go.Figure`
  - The plotly figure to add the band to.

- `x : np.ndarray`
  - X-axis values.

- `lower : np.ndarray`
  - Lower bound values.

- `upper : np.ndarray`
  - Upper bound values.

- `name : str`
  - Name for the legend.
  - Default: `"HDI"`

- `fillcolor : str`
  - Fill color for the band (hex color, color name, or Plotly color).
  - Default: `COLORS[0]`

- `opacity : float`
  - Opacity of the fill (0.0 to 1.0).
  - Default: `0.2`

- `showlegend : bool`
  - Whether to show in legend.
  - Default: `True`

**Returns:**

- `None` - Modifies the figure in place.

**Description:**

Adds a filled uncertainty band to a plotly figure using the lower and upper bounds. The band is created by concatenating the x values with their reverse and the lower/upper bounds to create a closed polygon.

---

### `_hex_to_rgb`

Convert a color name or hex to RGB string for rgba().

**Signature:**

```python
@staticmethod
def _hex_to_rgb(color: str) -> str:
```

**Parameters:**

- `color : str`
  - Color name (e.g., "royalblue", "red") or hex color (e.g., "#FF5733" or "#abc").

**Returns:**

- `str` - RGB string in format "r,g,b" for use in rgba().

**Description:**

Converts color names or hex colors to RGB strings. Supports common color names and both 3-digit and 6-digit hex colors.

---

### `_determine_date_format`

Determine the most appropriate date format based on the date series.

**Signature:**

```python
@staticmethod
def _determine_date_format(dates: pd.Series) -> str:
```

**Parameters:**

- `dates : pd.Series`
  - Series of datetime values.

**Returns:**

- `str` - Format string: `'%Y-%m-%d'`, `'%Y-%m'`, `'quarter'`, or `'%Y'`.

**Description:**

Analyzes the date series to determine if dates vary by day, month, quarter, or year, and returns the appropriate format string. Uses median time differences between consecutive dates to infer granularity.

---

### `_format_date_for_legend`

Format a single date value according to the format string.

**Signature:**

```python
@staticmethod
def _format_date_for_legend(
    date_val: Any,
    format_str: str,
) -> str:
```

**Parameters:**

- `date_val : Any`
  - The date value to format (datetime-like).

- `format_str : str`
  - Format string or `'quarter'` for quarterly formatting.

**Returns:**

- `str` - Formatted date string.

**Description:**

Formats a single date value according to the provided format string. Special handling for `'quarter'` format which produces YYYY-QQ format.

---

### `_prepare_color_column_for_plot`

Prepare a color column for plotting by formatting dates if needed.

**Signature:**

```python
@staticmethod
def _prepare_color_column_for_plot(
    df: pd.DataFrame,
    color: str | None,
) -> str | None:
```

**Parameters:**

- `df : pd.DataFrame`
  - DataFrame containing the color column.

- `color : str | None`
  - Name of the color column, or `None` if no color column.

**Returns:**

- `str | None` - Name of the color column to use for plotting (may be formatted version), or `None` if color was `None`.

**Description:**

If the color column is a datetime type, formats it appropriately for legend display by creating a `{color}_formatted` column. Otherwise, returns the original color column name.

---

## Usage Examples

### Basic Usage with MMMSummaryData

```python
from pymc_marketing.mmm.plot import MMMStakeholderPlotSuite
from pymc_marketing.sandbox.xarray_processing_utils import process_idata_for_plotting

# Process idata into plot data
plot_data = process_idata_for_plotting(idata, hdi_probs=[0.80, 0.90, 0.95])

# Initialize plotting suite
plot_suite = MMMStakeholderPlotSuite(plot_data=plot_data)

# Create visualizations
fig1 = plot_suite.posterior_predictive(hdi_prob=0.90)
fig2 = plot_suite.roas(hdi_prob=0.90)
fig3 = plot_suite.contribution(hdi_prob=0.90)
fig4 = plot_suite.roas_change_over_time(hdi_prob=0.90)
fig5 = plot_suite.saturation_curves(hdi_prob=0.90)
fig6 = plot_suite.marginal_curves(hdi_prob=0.90)
fig7 = plot_suite.contribution_vs_roas(hdi_prob=0.90)
fig8 = plot_suite.total_contribution()
fig9 = plot_suite.channel_spend()
```

### Usage without MMMSummaryData

```python
from pymc_marketing.mmm.plot import MMMStakeholderPlotSuite

# Initialize without plot_data
plot_suite = MMMStakeholderPlotSuite()

# Provide DataFrames directly
fig1 = plot_suite.posterior_predictive(
    posterior_predictive_df=my_posterior_predictive_df,
    hdi_prob=0.90
)
fig2 = plot_suite.roas(
    roas_df=my_roas_df,
    hdi_prob=0.90,
    x="channel"
)
fig3 = plot_suite.contribution(
    contribution_df=my_contribution_df,
    hdi_prob=0.90,
    x="channel"
)
```

### Custom Plotting

```python
# Plot ROAS by channel and year
fig = plot_suite.roas(
    roas_df=roas_by_year_df,
    hdi_prob=0.90,
    x="channel",
    color="year",  # Will be automatically formatted if datetime
)

# Plot contribution by channel
fig = plot_suite.contribution(
    contribution_df=contribution_by_channel_df,
    hdi_prob=0.90,
    x="channel"
)

# Plot period-over-period ROAS changes
fig = plot_suite.roas_change_over_time(
    roas_pop_df=roas_pop_df,
    hdi_prob=0.90,
    x="channel",
    color="date"
)

# Plot total contribution over time
fig = plot_suite.total_contribution(contribution_df=contribution_over_time_df)

# Plot channel spend
fig = plot_suite.channel_spend(df=channel_spend_df, x="channel")

# Customize plots using kwargs (passed to plotly.express functions)
fig = plot_suite.roas(
    roas_df=roas_df,
    hdi_prob=0.90,
    title="Custom ROAS Plot",
    width=800,
    height=600,
    labels={"mean": "ROAS Value", "channel": "Marketing Channel"}
)
```

---

## Design Notes

### Keyword Arguments (kwargs) Support

All plotting methods accept `**kwargs` that are passed directly to the underlying Plotly Express functions:

- **Line plots** (`saturation_curves`, `decay_curves`, `marginal_curves`): kwargs passed to `plotly.express.line()`
- **Bar plots** (`roas`, `contribution`, `roas_change_over_time`, `total_contribution`, `channel_spend`): kwargs passed to `plotly.express.bar()`
- **Scatter plots** (`contribution_vs_roas`): kwargs passed to `plotly.express.scatter()`
- **Posterior predictive** (`posterior_predictive`): kwargs passed to `fig.update_layout()` (uses graph_objects directly)

This allows users to customize plots extensively without modifying the plotting suite code. Common kwargs include:
- `title`: Plot title
- `width`, `height`: Figure dimensions
- `labels`: Dictionary mapping column names to display labels
- `template`: Plotly template (e.g., `"plotly_dark"`, `"ggplot2"`)
- `color_discrete_map`: Custom color mapping
- Any other valid Plotly Express parameter

**Example:**

```python
fig = plot_suite.roas(
    roas_df=roas_df,
    hdi_prob=0.90,
    title="Marketing Channel ROAS Analysis",
    width=1000,
    height=600,
    template="plotly_white",
    labels={"mean": "ROAS", "channel": "Channel Name"}
)
```

### Data Format Requirements

All plotting methods expect Polars DataFrames with specific column structures:

- **Time series data**: Must include a `date` column (datetime type)
- **HDI columns**: Format `abs_error_{prob}_lower` and `abs_error_{prob}_upper` where `prob` is integer percentage
- **Channel data**: Must include a `channel` column for channel-specific plots
- **Mean values**: Must include a `mean` column containing the metric to plot

### Color Handling

The suite automatically handles date formatting for color columns:
- Datetime columns used for coloring are automatically formatted based on granularity
- Formatting creates a new `{column}_formatted` column in the DataFrame
- Supports daily, monthly, quarterly, and yearly date formats

### HDI Visualization

- HDI bands are shown as filled areas with configurable opacity
- Error bars on bar charts show relative errors (HDI bounds - mean)
- Scatter plots support error bars in both x and y dimensions

### Backward Compatibility

The suite is designed to work with existing `MMMSummaryData` structures from `xarray_processing_utils.py`, ensuring seamless integration with the current data processing pipeline.

---

## Integration Plan

1. **Create the class structure** in `pymc_marketing/mmm/plot.py` (alongside or replacing existing `MMMPlotSuite`)
2. **Migrate functions** from `plotly_utils.py` to class methods
3. **Update imports** and ensure compatibility with existing code
4. **Add tests** for the new class-based API
5. **Update documentation** with new usage patterns
