---
date: 2026-01-09T17:16:52Z
researcher: Claude Sonnet 4.5
git_commit: c1a8a3828e0e3928572c9da6553c0b7bac9705f1
branch: isofer/plotting-design
repository: pymc-marketing
topic: "Component 3 (Plotting Suite) Implementation Design"
tags: [research, codebase, mmm, component3, plotting, plotly, narwhals, faceting, custom-dimensions, interactive-plots]
status: complete
last_updated: 2026-01-09
last_updated_by: Claude Sonnet 4.5
related_research: ["2026-01-07_21-08-47_mmm-data-plotting-framework-architecture.md", "2026-01-07_22-19-51_codifying-idata-conventions-component1-wrapper.md", "2026-01-08_20-09-03_component2-mmm-summary-object-implementation.md"]
---

# Research: Component 3 (Plotting Suite) Implementation Design

**Date**: 2026-01-09T17:16:52Z
**Researcher**: Claude Sonnet 4.5
**Git Commit**: c1a8a3828e0e3928572c9da6553c0b7bac9705f1
**Branch**: isofer/plotting-design
**Repository**: pymc-marketing

---

## Table of Contents

1. [Research Question](#research-question)
2. [Summary](#summary)
3. [Detailed Findings](#detailed-findings)
   - [Finding 1: Current Sandbox Implementation Analysis](#finding-1-current-sandbox-implementation-analysis)
   - [Finding 2: Narwhals Integration Requirements](#finding-2-narwhals-integration-requirements)
   - [Finding 3: Plotly Express Faceting for Multi-Dimensional Support](#finding-3-plotly-express-faceting-for-multi-dimensional-support)
   - [Finding 4: Custom Dimensions in MMM Models](#finding-4-custom-dimensions-in-mmm-models)
   - [Finding 5: DataFrame Contracts from Components 1 & 2](#finding-5-dataframe-contracts-from-components-1--2)
   - [Finding 6: Existing Plotting Architecture Patterns](#finding-6-existing-plotting-architecture-patterns)
4. [Component 3 Architecture Design](#component-3-architecture-design)
5. [Implementation Recommendations](#implementation-recommendations)
6. [Code References](#code-references)
7. [Open Questions](#open-questions)
8. [Conclusion](#conclusion)

---

## Research Question

How should we implement Component 3 (Plotting Suite) from the MMM Data & Plotting Framework Architecture, with specific requirements to:
1. Migrate from current Polars-only implementation to use **Narwhals** for DataFrame-agnostic plotting
2. Add support for **Plotly Express faceting** (facet_row, facet_col) to handle MMM models with custom dimensions (country, brand, region, etc.)

### Context

The three-component architecture has been designed:
- **Component 1**: Codified Data Wrapper (MMMIDataWrapper) - data access, filtering, aggregation
- **Component 2**: MMM Summary Object (MMMSummaryFactory) - transforms data into structured DataFrames with summary statistics
- **Component 3**: Plotting Suite (to be implemented) - thin Plotly functions consuming summary DataFrames

**Key Requirements**:
- Must integrate cleanly with Component 2's DataFrame output (both Pandas and Polars)
- Must support multi-dimensional faceting for models with custom dimensions
- Must replace matplotlib-based MMMPlotSuite with interactive Plotly plots
- Must maintain backward compatibility where feasible

---

## Summary

**Component 3 should be implemented as standalone Plotly functions using Narwhals for DataFrame abstraction and Plotly Express faceting for multi-dimensional support.**

### Key Findings

1. **Current sandbox implementation uses Polars directly** - needs migration to Narwhals
2. **Narwhals provides zero-cost DataFrame abstraction** - perfect for plotting libraries
3. **Plotly Express faceting handles custom dimensions naturally** - ideal for MMM use cases
4. **Component 2 produces standardized DataFrame schemas** - clear integration contracts
5. **Existing MMMPlotSuite provides proven patterns** - architectural guidance
6. **Custom dimensions become faceting columns** - straightforward implementation

### Recommended Architecture

**File Structure**:
```
pymc_marketing/mmm/
├── plot_interactive.py          # Component 3 (NEW - public API)
├── _plotly_helpers.py           # Internal Plotly utilities (NEW)
├── plot.py                      # Existing matplotlib suite (KEEP)
└── plotly_utils.py (sandbox)    # To be promoted/refactored (MOVE)
```

**Data Flow**:
```
Component 2: MMMSummaryFactory
  ↓ (produces DataFrames with schema)
Component 3: Plotly Functions (accept via Narwhals)
  ↓ (IntoDataFrame type hint)
nw.from_native(df) → unified Narwhals API
  ↓
Process using Narwhals (filtering, column ops, etc.)
  ↓
nw_df.to_native() → original DataFrame type preserved
  ↓
Plotly Express (with facet_row/facet_col for custom dims)
  ↓
go.Figure (interactive plot)
```

---

## Detailed Findings

### Finding 1: Current Sandbox Implementation Analysis

**Location**: [pymc_marketing/sandbox/plotly_utils.py](pymc_marketing/sandbox/plotly_utils.py)

#### Current Implementation Characteristics

**Line 18**: Imports Polars directly
```python
import polars as pl
```

**Line 23-51**: Type hints use `pl.DataFrame`
```python
def _get_hdi_columns(df: pl.DataFrame, hdi_prob: float) -> tuple[str, str]:
```

**Line 278-337**: Functions accept Polars, convert to Pandas internally
```python
def plot_posterior_predictive(
    posterior_predictive_df: pl.DataFrame,
    hdi_prob: float | None = 0.90,
) -> go.Figure:
    # Convert to pandas for plotly compatibility
    pdf = posterior_predictive_df.to_pandas()
```

#### Issues with Current Implementation

1. **No Narwhals**: Uses Polars directly, limiting to a single DataFrame type
2. **No faceting support**: No `facet_row` or `facet_col` parameters
3. **Manual subplot logic**: Would need custom code for each dimension combination
4. **Type restrictions**: Only accepts `pl.DataFrame`, not Pandas

#### What Works Well

1. **HDI column validation** (line 23-51): Clean helper for finding HDI columns
2. **HDI band plotting** (line 54-101): Reusable function for uncertainty visualization
3. **Date formatting** (line 133-243): Smart date format detection
4. **Color preparation** (line 246-275): Handles datetime columns in color dimension

---

### Finding 2: Narwhals Integration Requirements

**Source**: Web research on Narwhals library

#### What is Narwhals

Narwhals is a **zero-dependency, zero-cost DataFrame abstraction layer** that provides a unified API for Pandas, Polars, PyArrow, cuDF, Modin, and other DataFrame libraries.

**Key Quote**:
> "If you started with pandas, you'll get pandas back; if you started with Polars, you'll get Polars back... Narwhals doesn't copy or convert data—it wraps the original DataFrame."

#### Already Adopted By Major Visualization Libraries

- **Plotly** (v6+) - 3-10x faster Polars plotting, no Pandas overhead
- **Marimo** - DataFrame-agnostic plotting
- **Altair** - Declarative visualization
- **Bokeh** - Interactive plotting

#### Type Hints for Component 3

**Recommended Pattern** (preserving native DataFrame type):
```python
from narwhals.typing import IntoDataFrameT
import narwhals as nw

def plot_function(
    df_native: IntoDataFrameT,
    x_col: str,
    y_col: str,
) -> go.Figure:
    """Accept both Pandas and Polars DataFrames."""
    # Convert to narwhals (eager only - we need concrete data)
    nw_df = nw.from_native(df_native, eager_only=True)

    # All processing done via Narwhals API
    nw_df = nw_df.filter(nw.col(x_col).is_not_null())
    nw_df = nw_df.sort(x_col)

    # Pass native DataFrame to Plotly (preserves Pandas/Polars type)
    fig = px.line(nw_df.to_native(), x=x_col, y=y_col)
    return fig
```

**Key Pattern**: `nw.from_native(df)` → process with Narwhals → `nw_df.to_native()` to Plotly

#### Key Narwhals Methods for Plotting

**Column Access**:
```python
series = df.get_column('column_name')  # Get single column
subset = df.select('col1', 'col2')     # Select multiple
col_names = df.columns                  # List of column names
```

**Data Extraction**:
```python
data_list = series.to_list()           # Convert Series to Python list
data_array = series.to_numpy()         # Convert to NumPy (if supported)
```

**Filtering and Sorting**:
```python
filtered = df.filter(nw.col('value') > 100)
sorted_df = df.sort('date_column')
```

**Datetime Operations**:
```python
df = df.with_columns(
    year=nw.col('date').dt.year(),
    month=nw.col('date').dt.month(),
    date_str=nw.col('date').dt.to_string('%Y-%m-%d')
)
```

#### Performance Characteristics

- **Zero overhead**: Benchmarks show no detectable difference vs direct Pandas
- **Sometimes faster**: 20% faster in some cases due to avoiding unnecessary copies
- **Lazy execution preserved**: Polars LazyFrames stay lazy (but use `eager_only=True` for plotting)

#### Best Practices for Component 3

1. **Always use `eager_only=True`** - Plotting needs concrete data, not lazy queries
2. **Process with Narwhals, pass native to Plotly** - Use `nw_df.to_native()` for Plotly functions
3. **Use `IntoDataFrameT` for type hints** - Enables proper type checking
4. **Prefer `from_native/to_native`** - Better type preservation than `@narwhalify` decorator
5. **Select only needed columns** - More efficient with large DataFrames
6. **Never convert to Pandas explicitly** - Let `to_native()` preserve the original DataFrame type
7. **Extract to lists only when needed** - For custom trace construction (e.g., HDI bands)

---

### Finding 3: Plotly Express Faceting for Multi-Dimensional Support

**Source**: Web research on Plotly Express documentation and examples

#### Core Faceting Concepts

**Facet plots** (trellis plots, small multiples) are figures made up of multiple subplots showing the same axes, where each subplot shows a subset of the data.

**Parameters**:
- `facet_row`: Assigns marks to subplots stacked **vertically**
- `facet_col`: Assigns marks to subplots arranged **horizontally**
- `facet_col_wrap`: Maximum columns before wrapping to new row

#### Basic Faceting Examples

**Column Facets** (side-by-side):
```python
fig = px.scatter(df, x="total_bill", y="tip",
                 color="smoker",
                 facet_col="sex")  # One subplot per sex value
```

**Row Facets** (stacked):
```python
fig = px.bar(df, x="size", y="total_bill",
             color="sex",
             facet_row="smoker")  # One subplot per smoker value
```

**Grid Layout** (both row and column):
```python
fig = px.scatter(df, x="total_bill", y="tip",
                 facet_col="day",      # Columns: days of week
                 facet_row="time")     # Rows: lunch/dinner
```

**Wrapped Layout** (for many categories):
```python
fig = px.scatter(df, x='gdpPercap', y='lifeExp',
                 facet_col='year',           # 12 year panels
                 facet_col_wrap=4)           # Wrap to 4 columns
```

#### Faceting with Custom Dimensions (MMM Use Cases)

**Example 1: Channel Contributions by Country**
```python
# DataFrame: date, country, channel, mean, hdi_lower, hdi_upper
fig = px.bar(df, x="channel", y="mean",
             facet_col="country",         # Separate subplot per country
             facet_col_wrap=3,            # 3 countries per row
             color="channel",
             error_y="error_upper",       # HDI bars
             error_y_minus="error_lower")
```

**Example 2: ROAS Time Series by Brand**
```python
# DataFrame: date, brand, channel, mean
fig = px.line(df, x="date", y="mean",
              facet_row="brand",          # Stack brands vertically
              color="channel")             # Color by channel
```

**Example 3: Multi-Dimensional Faceting**
```python
# DataFrame: date, country, region, channel, mean
fig = px.line(df, x="date", y="mean",
              facet_row="country",        # Rows: countries
              facet_col="region",         # Columns: regions
              color="channel")             # Color: channels
```

#### Customizing Facet Plots

**Spacing**:
```python
fig = px.line(df, ...,
              facet_col_wrap=4,
              facet_row_spacing=0.04,     # Vertical spacing (0-1)
              facet_col_spacing=0.04)     # Horizontal spacing (0-1)
```

**Titles**:
```python
# Remove prefix from subplot titles
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
```

**Axes**:
```python
# Independent y-axes per facet
fig.update_yaxes(matches=None)

# Show tick labels on all subplots
fig.update_yaxes(showticklabels=True)
```

#### Adding Custom Traces to Facets

**Same trace to all facets**:
```python
# Add horizontal reference line to all facets
fig.add_hline(y=1, line_dash="dot",
              annotation_text="baseline",
              row="all", col="all")
```

**Different traces per facet**:
```python
# Must iterate and add to specific subplots
channels = df['channel'].unique()
for i, channel in enumerate(channels, 1):
    df_channel = df[df['channel'] == channel]

    # Add HDI band to specific facet
    fig.add_trace(
        go.Scatter(x=..., y=..., fill='toself', ...),
        row=1, col=i  # Specific subplot
    )
```

#### Limitations of Plotly Express Faceting

1. **No per-facet custom traces**: Different HDI bands per facet requires iteration
2. **Same chart type**: Cannot mix bar charts in one facet, line charts in another
3. **Axis sharing limitations**: `matches=None` affects all subplots globally
4. **Trace order issues**: Updates may not preserve zoom/selection state

#### When to Use Faceting vs `make_subplots`

**Use Plotly Express Faceting When**:
- ✅ Splitting data by 1-2 categorical dimensions
- ✅ All subplots show the same type of visualization
- ✅ Want automatic layout and consistent styling
- ✅ Data is in tidy/long format

**Use `make_subplots` When**:
- ✅ Need arbitrary subplot layouts
- ✅ Want different plot types (scatter + bar + heatmap)
- ✅ Need fine-grained control per subplot
- ✅ Want custom axis sharing

**Key Insight**: "Plotly Express faceting uses `make_subplots` internally," so you can combine both approaches:
```python
# Start with PX for easy faceting
fig = px.scatter(df, x="x", y="y", facet_col="category")

# Add custom traces using make_subplots syntax
fig.add_trace(go.Scatter(x=[...], y=[...]), row=1, col=2)
```

---

### Finding 4: Custom Dimensions in MMM Models

**Source**: Codebase analysis from [pymc_marketing/mmm/multidimensional.py](pymc_marketing/mmm/multidimensional.py)

#### How Custom Dimensions Are Defined

**Model Initialization** (multidimensional.py:262-350):
```python
mmm = MMM(
    date_column="date",
    channel_columns=["C1", "C2"],
    target_column="y",
    dims=("country",),  # Custom dimension tuple
    adstock=GeometricAdstock(l_max=10),
    saturation=LogisticSaturation(),
)

# Multiple dimensions
mmm_multi = MMM(
    dims=("country", "region"),  # Tuple of dimension names
    ...
)
```

**Key aspects**:
- `dims` is always a tuple of strings, even for single dimension: `("country",)`
- Empty tuple `()` or `None` means no custom dimensions (marginal model)
- Dimensions are stored as `self.dims`

#### DataFrame Schema with Dimensions

**Required Structure**:
```python
# Long format: one row per date-dimension-channel combination
df = pd.DataFrame({
    'date': [...],
    'country': [...],      # Dimension column
    'region': [...],       # Another dimension
    'channel_1': [...],    # Channel data
    'channel_2': [...],
    'target': [...]
})
```

**Key aspects**:
- Dimension columns must be present in DataFrame
- Long format required: one row per date-dimension combination
- Order typically: `[date, dim1, dim2, ..., channels..., target]`

#### Dimensions in InferenceData

**Stored as Coordinates** (multidimensional.py:977-982):
```python
# After fitting
countries = mmm.model.coords["country"]  # xarray coordinate
country_values = mmm.idata.posterior.coords["country"].values  # numpy array

# All dimensions
all_dims = list(mmm.model.coords.keys())  # ["date", "country", "channel", ...]
```

**Accessing Coordinates**:
- Available in `model.coords`, `idata.posterior.coords`, `idata.constant_data.coords`
- Values extracted with `.values` for numpy arrays

#### Iterating Over Dimension Values

**Pattern 1: Simple Iteration**:
```python
for geo in mmm.model.coords["geo"]:
    data_subset = mmm.idata.posterior.sel(geo=geo)
    # Process data for this geo...
```

**Pattern 2: Multiple Dimensions** (using itertools.product):
```python
import itertools

additional_coords = [
    mmm.idata.posterior.coords[dim].values
    for dim in additional_dims
]
dim_combinations = list(itertools.product(*additional_coords))

# Example: dims=("country", "region")
# countries = ["US", "UK"], regions = ["North", "South"]
# dim_combinations = [("US", "North"), ("US", "South"), ("UK", "North"), ("UK", "South")]
```

#### Filtering by Dimension Values

**Using xarray .sel()** (plot.py:759-760):
```python
# Select specific dimension value
geo_a_data = mmm.idata.posterior.sel(geo="geo_a")

# Select multiple values
selected = mmm.idata.posterior.sel(country=["US", "UK"])

# Select across multiple dimensions
subset = mmm.idata.posterior.sel(geo="geo_a", channel="C1")
```

#### Custom Dimensions in DataFrames (Component 2 Output)

**Dimensions become columns** (component2-doc:1546-1568):
```python
# Model with dims=("country",)
df = create_contribution_summary(data)

# DataFrame structure:
#     date       | country | channel | mean  | median | abs_error_94_lower | ...
# 0   2025-01-01 |   US    |   C1    | 1000  |   998  |        950         | ...
# 1   2025-01-01 |   US    |   C2    |  500  |   502  |        475         | ...
# 2   2025-01-01 |   UK    |   C1    |  800  |   798  |        750         | ...
```

**Standard vs Custom Dimensions**:

| Dimension | Type | Always Present |
|-----------|------|----------------|
| `date` | Standard | ✅ Yes |
| `channel` | Standard | ✅ Yes |
| `control` | Standard | ❌ Conditional |
| `country`, `region`, `brand`, etc. | Custom | ❌ Model-dependent |

#### Identifying Custom Dimensions

**Component 1 provides helper** (component1-doc:1638-1648):
```python
@property
def custom_dims(self) -> list[str]:
    """Get all custom dimension names."""
    standard_dims = {"date", "channel", "control", "fourier_mode", "chain", "draw"}

    return [
        dim for dim in self.idata.constant_data.dims
        if dim not in standard_dims
    ]

# Usage:
# mmm.data.custom_dims → ["country", "region"]
```

---

### Finding 5: DataFrame Contracts from Components 1 & 2

**Source**: Research documents on Component 1 and Component 2

#### DataFrame Schemas Component 2 Produces

**Schema 1: Posterior Predictive**
```python
{
    'date': datetime64[ns],            # REQUIRED
    'mean': float64,                   # REQUIRED
    'median': float64,                 # Always present
    'observed': float64,               # REQUIRED
    'abs_error_94_lower': float64,     # OPTIONAL (if hdi_probs specified)
    'abs_error_94_upper': float64,     # OPTIONAL
}
```

**Schema 2: Contributions**
```python
{
    'date': datetime64[ns],            # Optional (can be aggregated out)
    'channel': str,                    # REQUIRED
    'mean': float64,                   # REQUIRED
    'median': float64,                 # Always present
    'abs_error_94_lower': float64,     # OPTIONAL
    'abs_error_94_upper': float64,     # OPTIONAL
}
```

**Schema 3: ROAS**
```python
{
    'date': datetime64[ns],            # Optional
    'channel': str,                    # REQUIRED
    'mean': float64,                   # REQUIRED (contribution / spend)
    'median': float64,                 # Always present
    'abs_error_94_lower': float64,     # OPTIONAL
    'abs_error_94_upper': float64,     # OPTIONAL
}
```

**Schema 4: Channel Spend** (no uncertainty)
```python
{
    'date': datetime64[ns],            # REQUIRED
    'channel': str,                    # REQUIRED
    'channel_data': float64            # Raw spend values
}
```

#### HDI Column Naming Convention

**Pattern**: `abs_error_{prob}_lower`, `abs_error_{prob}_upper`

**Examples**:
- `hdi_probs=[0.80]` → `abs_error_80_lower`, `abs_error_80_upper`
- `hdi_probs=[0.90, 0.95]` → `abs_error_90_lower`, `abs_error_90_upper`, `abs_error_95_lower`, `abs_error_95_upper`

**Validation Function** (needed in Component 3):
```python
def _get_hdi_columns(df, hdi_prob: float) -> tuple[str, str]:
    """Get HDI column names for probability level."""
    prob_str = str(int(hdi_prob * 100))  # 0.80 → "80"
    lower_col = f"abs_error_{prob_str}_lower"
    upper_col = f"abs_error_{prob_str}_upper"

    if lower_col in df.columns and upper_col in df.columns:
        return lower_col, upper_col
    raise ValueError(f"HDI columns for {hdi_prob} not found")
```

#### Custom Dimensions in DataFrames

**Additional columns for each dimension**:
```python
# Model: dims=("country", "region")
# DataFrame columns: ['date', 'country', 'region', 'channel', 'mean', 'median', ...]
```

**For faceting**: Custom dimension columns become `facet_row` or `facet_col` parameters

#### Output Format (Pandas vs Polars)

Component 2 produces either format based on `output_format` parameter:
```python
df_pandas = mmm.summary.contributions()  # Default: Pandas
df_polars = mmm.summary.contributions(output_format="polars")
```

**Component 3 must accept both** via Narwhals `IntoDataFrameT` type hint.

#### Absolute vs Relative Errors

**DataFrames store absolute bounds**:
```python
df['abs_error_94_lower'] = 950  # Absolute lower bound
df['mean'] = 1000
df['abs_error_94_upper'] = 1050  # Absolute upper bound
```

**Plotly expects relative errors**:
```python
error_upper = df['abs_error_94_upper'] - df['mean']  # 50
error_lower = df['mean'] - df['abs_error_94_lower']  # 50
```

**Component 3 must convert** before passing to Plotly's `error_y` parameter.

---

### Finding 6: Existing Plotting Architecture Patterns

**Source**: Codebase analysis from [pymc_marketing/mmm/plot.py](pymc_marketing/mmm/plot.py)

#### Class-Based Architecture (MMMPlotSuite)

**Structure** (plot.py:208-219):
```python
class MMMPlotSuite:
    """Media Mix Model Plot Suite."""

    def __init__(self, idata: xr.Dataset | az.InferenceData):
        self.idata = idata

    # Public plotting methods
    def posterior_predictive(...) -> tuple[Figure, NDArray[Axes]]:
    def contributions_over_time(...) -> tuple[Figure, NDArray[Axes]]:
    def saturation_scatterplot(...) -> tuple[Figure, NDArray[Axes]]:
```

**Characteristics**:
- Stateless design - all data from idata
- Returns tuple of (Figure, NDArray[Axes])
- Internal helpers prefixed with `_`

#### Property-Based Integration

**Model Classes** (multidimensional.py:618-623):
```python
@property
def plot(self) -> MMMPlotSuite:
    """Use the MMMPlotSuite to plot the results."""
    self._validate_model_was_built()
    self._validate_idata_exists()
    return MMMPlotSuite(idata=self.idata)

# Usage:
mmm.plot.posterior_predictive()
```

**Pattern for Component 3**:
```python
@property
def plot_interactive(self) -> MMMPlotlyFactory:
    """Interactive Plotly plots."""
    return MMMPlotlyFactory(summary=self.summary)
```

#### Standard Method Signatures

**Common Parameters**:
- `var`: Variable name(s) - `str | list[str] | None`
- `hdi_prob`: Credible interval - `float` (default 0.85)
- `dims`: Dimension filters - `dict[str, str | int | list] | None`
- Return: `tuple[Figure, NDArray[Axes]]` (matplotlib) or `go.Figure` (plotly)

#### Dimension Handling Patterns

**Pattern 1: Identify Custom Dimensions** (plot.py:268-288):
```python
def _get_additional_dim_combinations(
    data: xr.Dataset,
    variable: str,
    ignored_dims: set[str],
) -> tuple[list[str], list[tuple]]:
    """Get non-standard dimensions and their combinations."""
    all_dims = list(data[variable].dims)
    additional_dims = [d for d in all_dims if d not in ignored_dims]

    if additional_dims:
        additional_coords = [data.coords[d].values for d in additional_dims]
        dim_combinations = list(itertools.product(*additional_coords))
    else:
        dim_combinations = [()]

    return additional_dims, dim_combinations
```

**Standard Ignored Dimensions**:
```python
ignored_dims = {"date", "chain", "draw", "sample", "channel", "control"}
```

**Pattern 2: Create Subplots for Each Combination** (plot.py:485-518):
```python
for row_idx, combo in enumerate(dim_combinations):
    ax = axes[row_idx][0]

    # Build indexers for this combination
    indexers = dict(zip(additional_dims, combo)) if additional_dims else {}

    # Select data for this subplot
    data = pp_data[var].sel(**indexers)

    # Plot on specific axis
    ax = self._add_median_and_hdi(ax, data, var, hdi_prob)

    # Title
    title = ", ".join(f"{dim}={val}" for dim, val in zip(additional_dims, combo))
    ax.set_title(title)
```

#### Validation Patterns

**Pattern 1: Check Data Group Exists** (plot.py:316-324):
```python
if not hasattr(self.idata, "posterior_predictive"):
    raise ValueError(
        "No posterior_predictive data found. "
        "Please run 'MMM.sample_posterior_predictive()'"
    )
```

**Pattern 2: Check Variable Exists** (plot.py:1139-1143):
```python
if var not in self.idata.posterior:
    raise ValueError(
        f"Variable '{var}' not found. "
        f"Available: {list(self.idata.posterior.data_vars)}"
    )
```

**Pattern 3: Validate Dimensions** (plot.py:366-387):
```python
def _validate_dims(dims: dict, all_dims: list[str]) -> None:
    """Validate dims exist and values are valid."""
    for key, val in dims.items():
        if key not in all_dims:
            raise ValueError(f"Dimension '{key}' not found")
        valid_values = idata.posterior.coords[key].values
        if val not in valid_values:
            raise ValueError(f"Value '{val}' not in dimension '{key}'")
```

#### Code Organization

**Three-Layer Structure**:
1. **Core Utilities** (`pymc_marketing/plot.py`) - Generic xarray plotting functions
2. **Domain Suite** (`pymc_marketing/mmm/plot.py`) - MMM-specific plotting class
3. **Sandbox/Experimental** (`pymc_marketing/sandbox/plotly_utils.py`) - New Plotly functions

**Component 3 should follow this pattern**:
- Public API in `pymc_marketing/mmm/plot_interactive.py`
- Internal helpers in `pymc_marketing/mmm/_plotly_helpers.py`
- Integrate with models via property

---

## Component 3 Architecture Design

### Design Principles

1. **DataFrame-agnostic via Narwhals** - Accept both Pandas and Polars
2. **Faceting for custom dimensions** - Use Plotly Express facet_row/facet_col
3. **Thin plotting functions** - Minimal logic, delegate to Plotly
4. **Clear contracts** - Explicit DataFrame schemas required
5. **Composable API** - Functions work independently
6. **Backward compatible** - Don't break existing matplotlib plots

### File Structure

```
pymc_marketing/mmm/
├── plot_interactive.py          # Component 3 public API (NEW)
│   └── class MMMPlotlyFactory   # Similar to MMMPlotSuite pattern
│
├── _plotly_helpers.py           # Internal utilities (NEW)
│   ├── _get_hdi_columns()       # HDI column validation (Narwhals)
│   ├── _add_hdi_band()          # Add HDI band to figure
│   ├── _is_datetime_column()    # Check datetime type (Narwhals)
│   ├── _format_date_column_nw() # Format dates as strings (Narwhals)
│   └── _detect_date_granularity() # Detect date formatting (Narwhals)
│
├── plot.py                      # Existing matplotlib suite (UNCHANGED)
│   └── class MMMPlotSuite       # Keep for backward compatibility
│
└── sandbox/plotly_utils.py      # To be refactored/promoted (DEPRECATED)
```

**Note**: Custom dimensions for auto-faceting come from `MMMIDataWrapper.custom_dims` (Component 1),
not detected from DataFrames.

### Core Functions Design

#### Function 1: `plot_posterior_predictive()`

**Purpose**: Plot model predictions vs observations with HDI band

```python
from narwhals.typing import IntoDataFrameT
import narwhals as nw
import plotly.express as px
import plotly.graph_objects as go

def plot_posterior_predictive(
    df: IntoDataFrameT,
    hdi_prob: float | None = 0.94,
    title: str | None = None,
    **plotly_kwargs,
) -> go.Figure:
    """
    Plot posterior predictive with HDI band.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Summary DataFrame from Component 2 with schema:
        - date: datetime64[ns] (REQUIRED)
        - mean: float64 (REQUIRED)
        - observed: float64 (REQUIRED)
        - abs_error_{prob}_lower: float64 (optional)
        - abs_error_{prob}_upper: float64 (optional)
    hdi_prob : float, optional
        HDI probability level (e.g., 0.94). If None, no uncertainty band.
    title : str, optional
        Figure title
    **plotly_kwargs
        Additional Plotly Express arguments including:
        - facet_row: Column name for row facets (e.g., "country")
        - facet_col: Column name for column facets (e.g., "region")
        - facet_col_wrap: Maximum columns before wrapping

    Returns
    -------
    go.Figure
        Interactive Plotly figure

    Examples
    --------
    >>> # Simple plot
    >>> df = mmm.summary.posterior_predictive()
    >>> fig = plot_posterior_predictive(df, hdi_prob=0.94)
    >>> fig.show()

    >>> # With faceting by country
    >>> fig = plot_posterior_predictive(df, facet_col="country", facet_col_wrap=3)
    >>> fig.show()
    """
    # Convert to Narwhals for unified API
    nw_df = nw.from_native(df, eager_only=True)

    # Validate required columns
    required_cols = {'date', 'mean', 'observed'}
    if not required_cols.issubset(set(nw_df.columns)):
        raise ValueError(f"DataFrame missing required columns: {required_cols}")

    # Sort by date for proper line plotting
    nw_df = nw_df.sort('date')

    # Extract facet params for HDI band logic
    facet_row = plotly_kwargs.get('facet_row')
    facet_col = plotly_kwargs.get('facet_col')

    # Identify columns to preserve (date, facet columns, HDI columns)
    id_cols = ['date']
    if facet_row:
        id_cols.append(facet_row)
    if facet_col:
        id_cols.append(facet_col)

    # Create long-format DataFrame with both predicted and observed
    # Using Narwhals unpivot (melt) operation
    plot_df = nw_df.select(
        *id_cols, 'mean', 'observed'
    ).unpivot(
        on=['mean', 'observed'],
        index=id_cols,
        variable_name='series',
        value_name='value'
    )

    # Rename series values for nicer legend
    plot_df = plot_df.with_columns(
        nw.when(nw.col('series') == 'mean')
        .then(nw.lit('Predicted'))
        .otherwise(nw.lit('Observed'))
        .alias('series')
    )

    # Create figure with single px.line call
    fig = px.line(
        plot_df.to_native(),
        x='date',
        y='value',
        color='series',
        title=title or "Posterior Predictive",
        labels={'value': 'Value', 'date': 'Date', 'series': ''},
        **plotly_kwargs
    )

    # Add HDI band if requested (still needs custom traces)
    if hdi_prob is not None:
        lower_col, upper_col = _get_hdi_columns(nw_df, hdi_prob)

        # Extract values for HDI band (custom trace needs lists)
        date_values = nw_df.get_column('date').to_list()
        lower_values = nw_df.get_column(lower_col).to_list()
        upper_values = nw_df.get_column(upper_col).to_list()

        # Add HDI band as filled area
        # Note: For faceted plots, would need to iterate per facet
        if facet_row is None and facet_col is None:
            # Simple case: single plot
            _add_hdi_band(fig, date_values, lower_values, upper_values,
                         name=f"{int(hdi_prob*100)}% HDI")
        else:
            # Faceted case: add band to each facet
            _add_hdi_bands_to_facets(fig, nw_df, facet_row, facet_col,
                                    lower_col, upper_col, hdi_prob)

    # Clean up facet titles
    if facet_row or facet_col:
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    fig.update_layout(hovermode="x")
    return fig
```

#### Function 2: `plot_bar()`

**Purpose**: Bar chart for contributions, ROAS, or any metric by channel/category

```python
def plot_bar(
    df: IntoDataFrameT,
    x: str = "channel",
    y: str = "mean",
    color: str | None = None,
    hdi_prob: float | None = None,
    title: str | None = None,
    yaxis_title: str | None = None,
    **plotly_kwargs,
) -> go.Figure:
    """
    Plot bar chart with optional error bars and faceting.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Summary DataFrame with schema:
        - x column (e.g., "channel"): str (REQUIRED)
        - y column (e.g., "mean"): float64 (REQUIRED)
        - abs_error_{prob}_lower: float64 (optional, for error bars)
        - abs_error_{prob}_upper: float64 (optional)
    x : str
        Column name for x-axis (default: "channel")
    y : str
        Column name for y-axis (default: "mean")
    color : str, optional
        Column name for bar coloring (e.g., "date")
    hdi_prob : float, optional
        HDI probability for error bars
    title : str, optional
        Figure title
    yaxis_title : str, optional
        Y-axis label
    **plotly_kwargs
        Additional Plotly Express arguments including:
        - facet_row: Column for row facets (e.g., "country")
        - facet_col: Column for column facets (e.g., "brand")
        - facet_col_wrap: Max columns before wrapping
        - barmode: "group" (side-by-side) or "stack" (stacked bars)

    Returns
    -------
    go.Figure
        Interactive Plotly figure

    Examples
    --------
    >>> # ROAS by channel
    >>> df = mmm.summary.roas(frequency='all_time')
    >>> fig = plot_bar(df, x='channel', y='mean', hdi_prob=0.94,
    ...                title='ROAS by Channel', yaxis_title='Return on Ad Spend')
    >>> fig.show()

    >>> # Contributions by country and channel
    >>> df = mmm.summary.contributions()
    >>> fig = plot_bar(df, x='channel', color='date', facet_col='country',
    ...                facet_col_wrap=3, title='Contributions by Country')
    >>> fig.show()
    """
    # Convert to Narwhals
    nw_df = nw.from_native(df, eager_only=True)

    # Validate required columns
    if x not in nw_df.columns or y not in nw_df.columns:
        raise ValueError(f"DataFrame must have '{x}' and '{y}' columns")

    # Prepare error bars if requested (add computed columns via Narwhals)
    error_y = None
    error_y_minus = None

    if hdi_prob is not None:
        lower_col, upper_col = _get_hdi_columns(nw_df, hdi_prob)
        # Convert absolute to relative errors using Narwhals
        nw_df = nw_df.with_columns(
            error_upper=(nw.col(upper_col) - nw.col(y)),
            error_lower=(nw.col(y) - nw.col(lower_col)),
        )
        error_y = 'error_upper'
        error_y_minus = 'error_lower'

    # Handle datetime columns in color (format via Narwhals)
    if color and _is_datetime_column(nw_df, color):
        nw_df = nw_df.with_columns(
            **{f'{color}_formatted': _format_date_column_nw(nw.col(color))}
        )
        color = f'{color}_formatted'

    # Create bar chart (pass native DataFrame)
    fig = px.bar(
        nw_df.to_native(),
        x=x,
        y=y,
        color=color,
        error_y=error_y,
        error_y_minus=error_y_minus,
        title=title,
        labels={y: yaxis_title or y.capitalize()},
        **plotly_kwargs
    )

    # Clean facet titles if faceting was used
    if plotly_kwargs.get('facet_row') or plotly_kwargs.get('facet_col'):
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # Rotate x-axis labels if needed
    fig.update_xaxes(tickangle=45)

    return fig
```

#### Function 3: `plot_curves()`

**Purpose**: Plot saturation or decay curves by channel

```python
def plot_curves(
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
    """
    Plot curves (saturation, decay) by channel.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Summary DataFrame with schema:
        - x column: float64 or int64 (REQUIRED)
        - y column (e.g., "mean"): float64 (REQUIRED)
        - color column (e.g., "channel"): str (REQUIRED)
        - abs_error_{prob}_lower: float64 (optional)
        - abs_error_{prob}_upper: float64 (optional)
    x : str
        Column name for x-axis (e.g., "x" for saturation, "time" for decay)
    y : str
        Column name for y-axis (default: "mean")
    color : str
        Column name for line coloring (default: "channel")
    hdi_prob : float, optional
        HDI probability for uncertainty bands
    title : str, optional
        Figure title
    xaxis_title : str, optional
        X-axis label
    yaxis_title : str, optional
        Y-axis label
    **plotly_kwargs
        Additional Plotly Express arguments including:
        - facet_row: Column for row facets
        - facet_col: Column for column facets

    Returns
    -------
    go.Figure
        Interactive Plotly figure

    Examples
    --------
    >>> # Saturation curves
    >>> df = mmm.summary.saturation_curves()
    >>> fig = plot_curves(df, x='x', title='Saturation Curves',
    ...                   xaxis_title='Spend', yaxis_title='Response')
    >>> fig.show()

    >>> # Decay curves by country
    >>> df = mmm.summary.decay_curves()
    >>> fig = plot_curves(df, x='time', facet_col='country',
    ...                   title='Decay Curves by Country')
    >>> fig.show()
    """
    # Convert to Narwhals
    nw_df = nw.from_native(df, eager_only=True)

    # Sort by x column for proper line plotting
    nw_df = nw_df.sort(x)

    # Extract facet params for HDI band logic
    facet_row = plotly_kwargs.get('facet_row')
    facet_col = plotly_kwargs.get('facet_col')

    # Create line chart (pass native DataFrame)
    fig = px.line(
        nw_df.to_native(),
        x=x,
        y=y,
        color=color,
        title=title,
        labels={
            x: xaxis_title or x.capitalize(),
            y: yaxis_title or y.capitalize()
        },
        **plotly_kwargs
    )

    # Add HDI bands per variable if requested
    if hdi_prob is not None:
        lower_col, upper_col = _get_hdi_columns(nw_df, hdi_prob)

        # Get unique values for color column using Narwhals
        color_values = nw_df.get_column(color).unique().to_list()

        # Add band for each variable value
        for var_val in color_values:
            # Filter using Narwhals
            nw_var = nw_df.filter(nw.col(color) == var_val)

            # Extract values for HDI band (custom trace needs lists)
            x_values = nw_var.get_column(x).to_list()
            lower_values = nw_var.get_column(lower_col).to_list()
            upper_values = nw_var.get_column(upper_col).to_list()

            # Determine which subplot this goes to
            # If no faceting, all go to main plot
            if facet_row is None and facet_col is None:
                _add_hdi_band(fig, x_values, lower_values, upper_values,
                             name=f"{var_val} HDI", showlegend=False)

    # Clean facet titles
    if facet_row or facet_col:
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    fig.update_layout(hovermode="x")
    return fig
```

### Helper Functions Design

#### Helper 1: `_get_hdi_columns()`

```python
def _get_hdi_columns(nw_df: nw.DataFrame, hdi_prob: float) -> tuple[str, str]:
    """
    Get HDI column names for probability level.

    Parameters
    ----------
    nw_df : nw.DataFrame
        Narwhals DataFrame containing HDI columns
    hdi_prob : float
        HDI probability (e.g., 0.94)

    Returns
    -------
    tuple[str, str]
        (lower_col, upper_col) names

    Raises
    ------
    ValueError
        If HDI columns not found
    """
    prob_str = str(int(hdi_prob * 100))
    lower_col = f"abs_error_{prob_str}_lower"
    upper_col = f"abs_error_{prob_str}_upper"

    if lower_col in nw_df.columns and upper_col in nw_df.columns:
        return lower_col, upper_col

    raise ValueError(
        f"HDI columns for probability {hdi_prob} not found. "
        f"Expected: {lower_col}, {upper_col}. "
        f"Available columns: {nw_df.columns}"
    )
```

#### Helper 2: `_add_hdi_band()`

```python
def _add_hdi_band(
    fig: go.Figure,
    x: list | np.ndarray,
    lower: list | np.ndarray,
    upper: list | np.ndarray,
    name: str = "HDI",
    fillcolor: str | None = None,
    opacity: float = 0.2,
    showlegend: bool = True,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """
    Add HDI band to figure as filled area.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to add band to
    x : list or array-like
        X-axis values (extracted from Narwhals via .to_list())
    lower : list or array-like
        Lower bound values (extracted from Narwhals via .to_list())
    upper : list or array-like
        Upper bound values (extracted from Narwhals via .to_list())
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

    x_concat = np.concatenate([x_arr, x_arr[::-1]])
    y_concat = np.concatenate([lower_arr, upper_arr[::-1]])

    trace = go.Scatter(
        x=x_concat,
        y=y_concat,
        mode='lines',
        fill='toself',
        fillcolor=fillcolor or f'rgba(65,105,225,{opacity})',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='none',
        name=name,
        showlegend=showlegend,
    )

    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)
```

#### Note: Custom Dimensions from Component 1

Instead of detecting custom dimensions from DataFrames, the `MMMPlotlyFactory` accesses
the `custom_dims` property from `MMMIDataWrapper` (Component 1) via the summary factory:

```python
# Component 1 provides custom dimension names
custom_dims = self.summary.data.custom_dims
# Returns: ["country", "region"]
```

This is more reliable than heuristic detection from DataFrames and maintains
proper separation of concerns - the model knows its dimensions.

#### Helper 4: `_is_datetime_column()`

```python
def _is_datetime_column(nw_df: nw.DataFrame, col: str) -> bool:
    """
    Check if a column is a datetime type.

    Parameters
    ----------
    nw_df : nw.DataFrame
        Narwhals DataFrame
    col : str
        Column name to check

    Returns
    -------
    bool
        True if column is datetime type
    """
    dtype = nw_df.get_column(col).dtype
    return dtype == nw.Datetime or str(dtype).startswith('Datetime')
```

#### Helper 5: `_format_date_column_nw()`

```python
def _format_date_column_nw(col_expr: nw.Expr) -> nw.Expr:
    """
    Format date column expression for plotting legends.

    Uses YYYY-MM-DD format (can be enhanced to detect granularity).

    Parameters
    ----------
    col_expr : nw.Expr
        Narwhals column expression

    Returns
    -------
    nw.Expr
        Expression that produces formatted date strings
    """
    # Default to ISO format; can be enhanced to detect granularity
    return col_expr.dt.to_string('%Y-%m-%d')
```

#### Helper 6: `_detect_date_granularity()`

```python
def _detect_date_granularity(nw_df: nw.DataFrame, col: str) -> str:
    """
    Detect date granularity to determine formatting.

    Parameters
    ----------
    nw_df : nw.DataFrame
        Narwhals DataFrame
    col : str
        Date column name

    Returns
    -------
    str
        Format string: '%Y-%m-%d', '%Y-%m', '%Y-Q%q', or '%Y'
    """
    dates = nw_df.get_column(col).unique().sort().to_list()

    if len(dates) < 2:
        return '%Y-%m-%d'

    # Calculate median difference in days
    diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
    median_diff = sorted(diffs)[len(diffs) // 2]

    if 28 <= median_diff <= 31:
        return '%Y-%m'
    elif 90 <= median_diff <= 92:
        return '%Y-Q'  # Quarterly needs special handling
    elif 365 <= median_diff <= 366:
        return '%Y'
    else:
        return '%Y-%m-%d'
```

### Factory Class Integration

```python
class MMMPlotlyFactory:
    """
    Factory for creating interactive Plotly plots from MMM summary data.

    Provides convenient access to plotting functions with smart defaults
    for faceting based on custom dimensions.

    Parameters
    ----------
    summary : MMMSummaryFactory
        Summary factory from Component 2
    auto_facet : bool, default True
        Automatically detect and apply faceting for custom dimensions

    Examples
    --------
    >>> # Access via model
    >>> factory = mmm.plot_interactive
    >>> fig = factory.posterior_predictive()
    >>> fig.show()

    >>> # With custom dimensions
    >>> fig = factory.contributions(facet_col='country', facet_col_wrap=3)
    >>> fig.show()
    """

    def __init__(
        self,
        summary: 'MMMSummaryFactory',
        auto_facet: bool = True,
    ):
        self.summary = summary
        self.auto_facet = auto_facet

    def posterior_predictive(
        self,
        hdi_prob: float | None = 0.94,
        **kwargs,
    ) -> go.Figure:
        """
        Plot posterior predictive with automatic faceting.

        If auto_facet=True and model has custom dimensions,
        automatically applies faceting (unless facet params already in kwargs).
        """
        df = self.summary.posterior_predictive()

        # Auto-facet using custom dimensions from Component 1
        facet_row = kwargs.get('facet_row')
        facet_col = kwargs.get('facet_col')

        if self.auto_facet and facet_row is None and facet_col is None:
            custom_dims = self.summary.data.custom_dims
            if len(custom_dims) == 1:
                kwargs['facet_col'] = custom_dims[0]
                kwargs.setdefault('facet_col_wrap', 3)
            elif len(custom_dims) >= 2:
                kwargs['facet_row'] = custom_dims[0]
                kwargs['facet_col'] = custom_dims[1]

        return plot_posterior_predictive(
            df,
            hdi_prob=hdi_prob,
            **kwargs
        )

    def contributions(
        self,
        hdi_prob: float | None = 0.94,
        component: str = "channel",
        frequency: str | None = None,
        **kwargs,
    ) -> go.Figure:
        """Plot contributions with automatic faceting."""
        df = self.summary.contributions(component=component, frequency=frequency)

        # Auto-facet using custom dimensions from Component 1
        facet_row = kwargs.get('facet_row')
        facet_col = kwargs.get('facet_col')

        if self.auto_facet and facet_row is None and facet_col is None:
            custom_dims = self.summary.data.custom_dims
            if custom_dims:
                kwargs['facet_col'] = custom_dims[0]
                kwargs.setdefault('facet_col_wrap', 3)

        return plot_bar(
            df,
            x='channel',
            y='mean',
            hdi_prob=hdi_prob,
            title='Channel Contributions',
            yaxis_title='Contribution',
            **kwargs
        )

    def roas(
        self,
        hdi_prob: float | None = 0.94,
        frequency: str | None = 'all_time',
        **kwargs,
    ) -> go.Figure:
        """Plot ROAS with automatic faceting."""
        df = self.summary.roas(frequency=frequency)

        # Auto-facet using custom dimensions from Component 1
        facet_row = kwargs.get('facet_row')
        facet_col = kwargs.get('facet_col')

        if self.auto_facet and facet_row is None and facet_col is None:
            custom_dims = self.summary.data.custom_dims
            if custom_dims:
                kwargs['facet_col'] = custom_dims[0]

        return plot_bar(
            df,
            x='channel',
            y='mean',
            hdi_prob=hdi_prob,
            title='Return on Ad Spend',
            yaxis_title='ROAS',
            **kwargs
        )

    def saturation_curves(
        self,
        hdi_prob: float | None = 0.94,
        **kwargs,
    ) -> go.Figure:
        """Plot saturation curves."""
        # Note: Need to add create_saturation_curves to Component 2
        df = self.summary.saturation_curves()

        return plot_curves(
            df,
            x='x',
            hdi_prob=hdi_prob,
            title='Saturation Curves',
            xaxis_title='Spend',
            yaxis_title='Response',
            **kwargs
        )
```

### Model Integration

```python
# In pymc_marketing/mmm/multidimensional.py

@property
def plot_interactive(self) -> MMMPlotlyFactory:
    """
    Interactive Plotly plotting suite.

    Returns
    -------
    MMMPlotlyFactory
        Factory for creating interactive plots

    Examples
    --------
    >>> mmm.fit(X, y)
    >>> mmm.sample_posterior_predictive(X)
    >>>
    >>> # Interactive posterior predictive plot
    >>> fig = mmm.plot_interactive.posterior_predictive()
    >>> fig.show()
    >>>
    >>> # Contributions with faceting
    >>> fig = mmm.plot_interactive.contributions(facet_col='country')
    >>> fig.show()
    """
    self._validate_model_was_built()
    self._validate_idata_exists()

    from pymc_marketing.mmm.plot_interactive import MMMPlotlyFactory

    return MMMPlotlyFactory(summary=self.summary)
```

---

## Implementation Recommendations

### Phase 1: Core Implementation (Immediate)

**Priority**: High | **Effort**: High | **Impact**: Very High

#### Tasks

1. **Create `pymc_marketing/mmm/_plotly_helpers.py`**
   - [ ] Implement `_get_hdi_columns(nw_df, hdi_prob)` - Validate HDI columns exist (Narwhals)
   - [ ] Implement `_add_hdi_band(fig, x, lower, upper, ...)` - Add uncertainty bands
   - [ ] Implement `_is_datetime_column(nw_df, col)` - Check datetime type (Narwhals)
   - [ ] Implement `_format_date_column_nw(col_expr)` - Smart date formatting (Narwhals expr)
   - [ ] Implement `_detect_date_granularity(nw_df, col)` - Detect date format (Narwhals)
   - [ ] Implement `_add_hdi_bands_to_facets(fig, nw_df, ...)` - Add bands to each facet
   - [ ] Add comprehensive docstrings and type hints
   - [ ] Write unit tests for each helper
   - Note: Custom dimensions come from `MMMIDataWrapper.custom_dims`, not detected from DataFrames

2. **Create `pymc_marketing/mmm/plot_interactive.py`**
   - [ ] Implement `plot_posterior_predictive(df, hdi_prob, facet_row, facet_col, ...)`
   - [ ] Implement `plot_bar(df, x, y, hdi_prob, facet_row, facet_col, ...)`
   - [ ] Implement `plot_curves(df, x, hdi_prob, facet_row, facet_col, ...)`
   - [ ] Implement `plot_saturation_curves(df, hdi_prob, ...)`
   - [ ] Implement `plot_decay_curves(df, hdi_prob, ...)`
   - [ ] All functions use Narwhals `IntoDataFrameT` type hint
   - [ ] All functions process via Narwhals, pass `nw_df.to_native()` to Plotly
   - [ ] All functions support faceting parameters
   - [ ] Add comprehensive docstrings with examples
   - [ ] Write integration tests with Component 2

3. **Implement `MMMPlotlyFactory` class**
   - [ ] Constructor accepts `MMMSummaryFactory` (Component 2)
   - [ ] Add `auto_facet` parameter for automatic dimension detection
   - [ ] Implement convenience methods calling standalone functions:
     - [ ] `posterior_predictive()`
     - [ ] `contributions()`
     - [ ] `roas()`
     - [ ] `saturation_curves()`
     - [ ] `decay_curves()`
   - [ ] Add smart defaults (e.g., `frequency='all_time'` for ROAS)
   - [ ] Write integration tests

4. **Update dependencies**
   - [ ] Add `narwhals>=1.0` to `pyproject.toml` (required dependency)
   - [ ] Confirm `plotly` is already a dependency
   - [ ] Update documentation on optional dependencies

5. **Integration with MMM models**
   - [ ] Add `plot_interactive` property to `MMM` class (multidimensional.py)
   - [ ] Add `plot_interactive` property to legacy MMM class (mmm.py)
   - [ ] Add validation before returning factory
   - [ ] Write integration tests

### Phase 2: Extended Functionality (Next 1-2 Releases)

**Priority**: Medium | **Effort**: Medium | **Impact**: High

#### Tasks

1. **Additional plot types**
   - [ ] `plot_contribution_over_time()` - Stacked area chart for time series contributions
   - [ ] `plot_waterfall()` - Waterfall chart for contribution decomposition
   - [ ] `plot_channel_spend()` - Bar chart for channel spend
   - [ ] `plot_contribution_vs_roas()` - Scatter plot with error bars

2. **Enhanced faceting support**
   - [ ] Support for `facet_row_spacing` and `facet_col_spacing` parameters
   - [ ] Better handling of many facets (e.g., scrollable subplots)
   - [ ] Support for shared vs independent axes per facet
   - [ ] Custom subplot titles beyond dimension values

3. **Improved HDI visualization**
   - [ ] Support multiple HDI levels in same plot (e.g., 50%, 90%, 95%)
   - [ ] Different styles for different HDI levels
   - [ ] Option to show HDI as error bars vs filled bands

4. **Component 2 additions**
   - [ ] Implement `create_saturation_curves()` in Component 2
   - [ ] Implement `create_decay_curves()` in Component 2
   - [ ] Implement `create_total_contribution_summary()` in Component 2
   - [ ] Implement `create_period_over_period_summary()` in Component 2

### Phase 3: Advanced Features (Future)

**Priority**: Low | **Effort**: Medium | **Impact**: Medium

#### Tasks

1. **Interactive dashboard support**
   - [ ] Callbacks for linked plots (click on bar → update time series)
   - [ ] Date range selectors
   - [ ] Channel selection widgets
   - [ ] Export to Dash app

2. **Animation support**
   - [ ] Animate contributions over time
   - [ ] Animate by dimension values
   - [ ] Playback controls

3. **Custom styling**
   - [ ] PyMC-Marketing theme for Plotly
   - [ ] Color palette customization
   - [ ] Layout templates

4. **Performance optimization**
   - [ ] Lazy rendering for large datasets
   - [ ] WebGL rendering for many data points
   - [ ] Streaming updates for real-time dashboards

### Phase 4: Documentation & Testing

**Priority**: High | **Effort**: Medium | **Impact**: Very High

#### Tasks

1. **Documentation**
   - [ ] User guide: "Interactive Plotting with Component 3"
   - [ ] Migration guide: matplotlib → Plotly
   - [ ] API reference for all functions
   - [ ] Gallery of examples with code
   - [ ] Jupyter notebook examples
   - [ ] Add to main documentation site

2. **Testing**
   - [ ] Unit tests for all helper functions
   - [ ] Integration tests with Component 2
   - [ ] Tests for both Pandas and Polars inputs
   - [ ] Tests for faceting with custom dimensions
   - [ ] Tests for HDI visualization
   - [ ] Visual regression tests (plotly snapshots)
   - [ ] Performance benchmarks

3. **Examples and tutorials**
   - [ ] Basic plotting tutorial
   - [ ] Multi-dimensional MMM example
   - [ ] Custom styling example
   - [ ] Dashboard integration example
   - [ ] Comparison with matplotlib plots

### Deprecation Strategy

**Goal**: Maintain backward compatibility while encouraging migration to interactive plots

1. **Phase 1: Parallel Implementation** (Current Release)
   - Add Component 3 (interactive plots) alongside existing matplotlib plots
   - No deprecation warnings
   - Document both approaches in user guide

2. **Phase 2: Soft Deprecation** (Next Release)
   - Add soft deprecation notices in matplotlib plot docstrings
   - Point users to interactive equivalents
   - "Note: Consider using `mmm.plot_interactive.posterior_predictive()` for interactive plots"

3. **Phase 3: Hard Deprecation** (2-3 Releases Later)
   - Add `@deprecated` decorator to matplotlib plot methods
   - Show warnings when called
   - Still functional but discouraged

4. **Phase 4: Removal** (Major Version Bump)
   - Remove matplotlib plot methods (or move to legacy module)
   - Interactive plots become default

---

## Code References

### Current Sandbox Implementation
- [pymc_marketing/sandbox/plotly_utils.py:1-603](pymc_marketing/sandbox/plotly_utils.py) - Current Polars-only implementation
- [pymc_marketing/sandbox/plotly_utils.py:23-51](pymc_marketing/sandbox/plotly_utils.py#L23-L51) - HDI column validation
- [pymc_marketing/sandbox/plotly_utils.py:54-101](pymc_marketing/sandbox/plotly_utils.py#L54-L101) - HDI band plotting
- [pymc_marketing/sandbox/plotly_utils.py:133-243](pymc_marketing/sandbox/plotly_utils.py#L133-L243) - Date formatting utilities

### Existing Plotting Architecture
- [pymc_marketing/mmm/plot.py:208-219](pymc_marketing/mmm/plot.py#L208-L219) - MMMPlotSuite class structure
- [pymc_marketing/mmm/plot.py:268-288](pymc_marketing/mmm/plot.py#L268-L288) - Dimension combination logic
- [pymc_marketing/mmm/plot.py:366-387](pymc_marketing/mmm/plot.py#L366-L387) - Dimension validation
- [pymc_marketing/mmm/plot.py:485-518](pymc_marketing/mmm/plot.py#L485-L518) - Subplot creation pattern

### Model Integration
- [pymc_marketing/mmm/multidimensional.py:618-623](pymc_marketing/mmm/multidimensional.py#L618-L623) - Property-based integration pattern

### Custom Dimensions
- [pymc_marketing/mmm/multidimensional.py:262-350](pymc_marketing/mmm/multidimensional.py#L262-L350) - Dimension definition
- [pymc_marketing/mmm/multidimensional.py:977-982](pymc_marketing/mmm/multidimensional.py#L977-L982) - Coordinate storage

### Component 1 & 2 Contracts
- [thoughts/shared/research/2026-01-07_22-19-51_codifying-idata-conventions-component1-wrapper.md](thoughts/shared/research/2026-01-07_22-19-51_codifying-idata-conventions-component1-wrapper.md) - Component 1 design
- [thoughts/shared/research/2026-01-08_20-09-03_component2-mmm-summary-object-implementation.md](thoughts/shared/research/2026-01-08_20-09-03_component2-mmm-summary-object-implementation.md) - Component 2 design and DataFrame schemas

---

## Open Questions

### 1. Plotly Express Limitations with Complex HDI Bands

**Question**: How should we handle adding different HDI bands to each facet when using Plotly Express faceting?

**Options**:
A. **Iterate and add manually** (recommended for Phase 1)
   - Use PX faceting for base plot
   - Loop through facets and add HDI bands with `fig.add_trace(..., row=R, col=C)`
   - Pros: Works with PX faceting
   - Cons: More complex code

B. **Use `make_subplots` instead of PX**
   - Create subplots manually
   - Full control over each subplot
   - Pros: More flexible
   - Cons: Loses PX automatic styling and layout

C. **Compute HDI bands as separate traces in long format**
   - Pre-compute band coordinates for each facet
   - Pass as additional traces to PX
   - Pros: Cleaner code
   - Cons: May not work with all PX functions

**Recommendation**: Start with Option A for Phase 1, explore Option C in Phase 2.

---

### 2. Auto-Faceting Heuristics

**Question**: How should `auto_facet=True` decide which custom dimensions to facet by?

**Current Proposal**:
- 1 custom dimension → `facet_col` with `facet_col_wrap=3`
- 2+ custom dimensions → `facet_row=dims[0]`, `facet_col=dims[1]`
- User can always override by specifying explicitly

**Concerns**:
- May not match user expectations
- Order of dimensions matters (which should be row vs col?)
- Large number of facets can be overwhelming

**Alternative**:
- Default `auto_facet=False`, require explicit faceting
- Provide helper function `suggest_faceting(df)` that prints recommendation
- Document faceting patterns in user guide

**Recommendation**: Start with `auto_facet=False` (explicit), add `suggest_faceting()` helper.

---

### 3. Performance with Many Facets

**Question**: What happens when a model has many dimension values (e.g., 50 countries)?

**Concerns**:
- 50 facets in one figure is overwhelming
- Plotly performance may degrade
- Hard to see individual plots

**Options**:
A. **Limit facets and warn**
   - Max 12 facets, warn if more
   - Suggest filtering dimensions first

B. **Pagination support**
   - Create multiple figures, one per "page"
   - `plot_bar(df, facet_col='country', max_facets_per_page=12)`

C. **Scrollable subplots**
   - Create figure with scrollbars
   - Use Plotly's `updatemenus` for navigation

**Recommendation**: Phase 1 - Option A (warn), Phase 2 - explore Option B.

---

### 4. Component 2 Dependencies

**Question**: Which additional summary functions does Component 2 need to implement?

**Required for Component 3**:
- ✅ `create_posterior_predictive_summary()` - Already designed
- ✅ `create_contribution_summary()` - Already designed
- ✅ `create_roas_summary()` - Already designed
- ❌ `create_saturation_curves()` - NOT YET DESIGNED
- ❌ `create_decay_curves()` - NOT YET DESIGNED
- ❌ `create_total_contribution_summary()` - NOT YET DESIGNED

**Recommendation**: Design these functions as part of Component 2 follow-up work before implementing corresponding Component 3 plots.

---

### 5. Backward Compatibility Strategy

**Question**: How aggressively should we deprecate matplotlib plots?

**Options**:
A. **Conservative** (recommended)
   - Keep matplotlib plots indefinitely
   - Add interactive plots as alternative
   - Let users choose based on use case

B. **Moderate**
   - Soft deprecation warnings in next release
   - Hard deprecation in 2-3 releases
   - Remove in major version bump

C. **Aggressive**
   - Deprecate immediately
   - Remove in next major version

**Recommendation**: Option A for now - both plotting systems coexist. Revisit based on user feedback.

---

## Conclusion

Component 3 (Plotting Suite) should be implemented as **DataFrame-agnostic Plotly functions using Narwhals and Plotly Express faceting**, following these key design decisions:

### Architecture Summary

**✅ Standalone Functions + Factory Pattern**
- Core functions: `plot_posterior_predictive()`, `plot_bar()`, `plot_curves()`
- Factory class: `MMMPlotlyFactory` for convenient access
- Integration: `mmm.plot_interactive` property

**✅ Narwhals for DataFrame Abstraction**
- Accept both Pandas and Polars via `IntoDataFrameT` type hint
- Process data using Narwhals API, pass native DataFrame to Plotly via `to_native()`
- Zero-cost abstraction, no performance overhead, preserves original DataFrame type

**✅ Plotly Express Faceting for Custom Dimensions**
- Use `facet_row`, `facet_col`, `facet_col_wrap` parameters
- Automatic layout and styling
- Support for multi-dimensional MMM models

**✅ HDI Visualization**
- Validate HDI columns exist with `_get_hdi_columns()`
- Add uncertainty bands with `_add_hdi_band()`
- Handle per-facet bands by iteration

**✅ Clear Integration Contracts**
- Component 2 produces standardized DataFrame schemas
- Component 3 validates required columns
- Custom dimensions become faceting columns

### Design Benefits

1. **✅ DataFrame-agnostic** - Works with Pandas and Polars via Narwhals
2. **✅ Multi-dimensional support** - Plotly Express faceting handles custom dims naturally
3. **✅ Interactive plots** - Plotly provides hover, zoom, pan out of the box
4. **✅ Clean separation** - Component 3 only does plotting, no data wrangling
5. **✅ Backward compatible** - Doesn't break existing matplotlib plots
6. **✅ Extensible** - Easy to add new plot types
7. **✅ Testable** - Pure functions with clear inputs/outputs

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| DataFrame abstraction | **Narwhals** | Zero-cost, already used by Plotly, supports 10+ libraries |
| Faceting approach | **Plotly Express** | Automatic layout, native support, easy to use |
| Function vs Class | **Both** | Functions for flexibility, Factory for convenience |
| HDI bands per facet | **Manual iteration** | PX limitation, but manageable with helper functions |
| Auto-faceting | **Opt-in with smart defaults** | Balance convenience vs explicit control |
| Backward compatibility | **Parallel implementation** | Keep matplotlib, add interactive as alternative |

### Implementation Priority

**Phase 1** (Immediate):
- Core plotting functions with Narwhals
- MMMPlotlyFactory class
- Integration with MMM models
- Basic documentation and tests

**Phase 2** (1-2 releases):
- Extended plot types
- Enhanced faceting support
- Component 2 additions (saturation/decay curves)

**Phase 3** (Future):
- Interactive dashboards
- Animation support
- Performance optimization

### Next Steps

1. **Implement `_plotly_helpers.py`** - Core helper functions
2. **Implement `plot_interactive.py`** - Public plotting API
3. **Add `plot_interactive` property** - Model integration
4. **Write comprehensive tests** - Unit and integration tests
5. **Update documentation** - User guide and API reference

This architecture provides a solid foundation for interactive plotting in pymc-marketing while maintaining clean separation of concerns across the three-component framework.
