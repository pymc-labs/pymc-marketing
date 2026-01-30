# MMM Plotting Utilities

This directory contains utilities for processing MMM (Marketing Mix Model) inference data (`MMM.idata`) and creating interactive plots for the stakeholder mode of the insight agents system. The utilities provide a pipeline for:

1. processing `MMM.idata` (xarray DataTree format) into structured Polars DataFrames.
2. visualizing MMM results using Plotly for interactive stakeholder dashboards.

These utilities are currently standalone and experimental, developed in the `feature/experimental-plots` branch.

## Dependencies

The utilities require installing two packages (see the new `pyproject.toml`):
- **plotly** >=6.5.0 - Interactive plotting and visualization
- **polars** >=1.36.1 - Fast DataFrame operations

## Files

### Core Utilities

#### `xarray_processing_utils.py`
Main processing utilities for converting xarray DataTree structures (from `MMM.idata`) into Polars DataFrames ready for plotting.

**Key Functions:**

*Data Conversion (idata: az.InferenceData → idata: xr.DataTree):*
- `idata_to_datatree()` - Converts ArviZ InferenceData to xarray DataTree format. Created because newer ArviZ versions use DataTree format directly (idata is already `xr.DataTree`). Handles coordinate renaming (e.g., 'index' → 'date') for proper date handling in the fit_data group.

*Data Wrangling (idata: xr.DataTree → idata: xr.DataTree):*
Wrangling tools for processing the whole idata structure. These can be easily adapted as agent tools to enable user-driven data manipulation. Common use cases include filtering to specific date ranges, aggregating channels (e.g., grouping all social channels together), or selecting specific coordinates.

- `select_by_data_range()` - Filters data by date range
- `select_by_coords()` / `select_from_dataset_by_coords()` / `select_from_datatree_by_coords()` - Selects specific coordinate values (e.g., channels, geos)
- `drop_by_coords()` / `drop_from_dataset_by_coords()` - Removes specific coordinate values
- `aggregate_by_period()` / `aggregate_dataset_by_period()` - Aggregates data by time periods (daily, monthly, quarterly, yearly, or total)
- `aggregate_by_coords()` / `aggregate_dataset_by_coords()` - Aggregates data by coordinate mappings (e.g., grouping channels)
- `period_over_period()` - Computes period-over-period percentage changes

*Processing idata (xr.DataTree) into Polars' DataFrames for Plotting:*
- `process_idata_for_plotting()` - Main entry point that converts `MMM.idata` into `MMMPlotData` dataclass containing pre-processed DataFrames
- `idata_var_to_summary_df_with_hdi()` - Converts xarray DataArrays to Polars DataFrames with mean, median, and HDI (Highest Density Interval) columns
- `process_adstock_decay_df()` - Processes adstock decay curves for visualization

**Data Structures:**

`MMMPlotData` is currently a convenient way to group processed dataframes together, created by `process_idata_for_plotting()`.It's current design is very simple:

- `MMMPlotData` - Dataclass containing:
  - `posterior_predictive`: Predicted vs observed values over time
  - `roas`: Return on Ad Spend by channel
  - `roas_pop`: Period-over-period ROAS changes
  - `contribution`: Channel contribution to revenue
  - `channel_spend`: Marketing spend by channel
  - `channels`: List of channel names
  - `hdi_probs`: List of HDI probability levels computed

**Current unaddressed issues with MMMPlotData**

Several design issues need to be addressed for production use:
1. **Lazy Evaluation**: Currently, all dataframes are created at once when `process_idata_for_plotting()` is called. For large models or when only specific plots are needed, this can be inefficient. Should we implement lazy evaluation where dataframes are created on-demand only when accessed?

2. **Multiple Aggregation Levels**: Users often need plots at different time aggregation levels (yearly, quarterly, monthly, or total). Currently, this requires creating separate `MMMPlotData` instances for each level. Should we:
   - Create separate `MMMPlotData` instances for each aggregation level?
   - Store multiple aggregation levels within a single `MMMPlotData` structure?
   - Use a factory pattern that generates `MMMPlotData` on-demand for requested aggregation levels?

3. **Extra Dimensions**: Models may have additional dimensions beyond `date` and `channel` (e.g., `geo` for geographic regions). The plan is to plot only one coordinate at at time, so the plots are fine, but what is the appropriate data structure and processing? Should we:
   - Create a separate `MMMPlotData` for each dimension combination?
   - Store dimension-filtered data within a single `MMMPlotData`?
   - Use a more flexible structure that can handle multi-dimensional data?

These considerations will need to be resolved during integration to ensure the utilities scale effectively for production use.

#### `plotly_utils.py`
Plotting utilities for creating interactive Plotly visualizations from processed MMM data.

**Key Functions:**
- `plot_posterior_predictive()` - Plots predicted vs observed values with optional HDI bands
- `plot_saturation_curves()` - Plots saturation curves showing diminishing returns
- `plot_decay_curves()` - Plots adstock decay curves over time
- `plot_bar()` - Generic bar chart function for ROAS, contribution, etc.
- `plot_total_contribution_bar()` - Stacked bar chart for total contribution by channel
- `plot_channel_spend_bar()` - Bar chart for channel spend
- `plot_contribution_vs_roas()` - Scatter plot comparing contribution vs ROAS

**Helper Functions:**
- `_get_hdi_columns()` - Retrieves HDI column names for a given probability level
- `_plot_hdi_band()` - Adds HDI uncertainty bands to plots
- `format_date_column_for_legend()` - Formats date columns appropriately for legends (handles daily, monthly, quarterly, yearly)
- `_prepare_color_column_for_plot()` - Prepares color columns (especially dates) for plotting

**Features:**
- Automatic HDI (uncertainty) band visualization
- Smart date formatting based on data granularity
- Consistent color schemes across plots
- Error bars for bar charts using HDI intervals

### Test Files

#### `test_xarray_processing_utils.py`
Comprehensive test suite for `xarray_processing_utils.py`.

#### `test_plotly_utils.py`
Comprehensive test suite for `plotly_utils.py`.

### Demo Notebooks

#### `demo_notebook.ipynb`
Interactive Jupyter notebook demonstrating the full workflow:
1. Loading a fitted MMM model (created by https://github.com/pymc-labs/pymc-marketing/blob/main/docs/source/notebooks/mmm/mmm_case_study.ipynb)
2. Converting `MMM.idata` to DataTree format
3. Processing data for different time aggregations (weekly, monthly, yearly)
4. Creating various visualizations:
   - Posterior predictive plots
   - ROAS by channel and year
   - Year-over-year ROAS performance
   - Channel spend visualizations
   - Contribution analysis
   - Saturation curves

#### `demo_marimo_notebook.py`
A Marimo notebook that is almost identical to the Jupyter notebook above, demonstrating the same full workflow using Marimo's reactive notebook format.

## Usage Example

```python
import xarray_processing_utils as xpu
import plotly_utils as pu
from pymc_marketing.mmm import MMM

# Load fitted MMM model
mmm = MMM.load("path/to/model.pm")

# Convert idata to DataTree and process for plotting
dt = xpu.idata_to_datatree(mmm.idata)
plot_data = xpu.process_idata_for_plotting(dt, hdi_probs=[0.80, 0.90, 0.95])

# Create visualizations
fig1 = pu.plot_posterior_predictive(plot_data.posterior_predictive, hdi_prob=0.90)
fig2 = pu.plot_bar(plot_data.roas, hdi_prob=0.90, x='channel', yaxis_title='ROAs')
fig3 = pu.plot_saturation_curves(saturation_curves_df, hdi_prob=0.90)
```


## Testing

Both utility modules have comprehensive test suites. Run tests with:

```bash
pytest src/sandbox/test_xarray_processing_utils.py
pytest src/sandbox/test_plotly_utils.py
```
