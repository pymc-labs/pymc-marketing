---
date: 2026-01-07T21:08:47Z
researcher: Claude Sonnet 4.5
git_commit: ceb76cd374e458ce0302d800b60e71cafe41c997
branch: main
repository: pymc-marketing
topic: "MMM Data & Plotting Framework Architecture Evaluation"
tags: [research, codebase, mmm, plotting, data-handling, architecture, idata]
status: complete
last_updated: 2026-01-07
last_updated_by: Claude Sonnet 4.5
---

# Research: MMM Data & Plotting Framework Architecture Evaluation

**Date**: 2026-01-07T21:08:47Z
**Researcher**: Claude Sonnet 4.5
**Git Commit**: ceb76cd374e458ce0302d800b60e71cafe41c997
**Branch**: main
**Repository**: pymc-marketing

## Research Question

Evaluate the proposed three-component architecture for MMM Data & Plotting Framework:
1. Does the solution answer the stated requirements?
2. Provide feedback on the solution and suggest improvements
3. If there is a better solution, provide it

### Context

The current PyMC-Marketing implementation lacks a formal data-handling layer and interactive plotting suite, leading to three key problems:

**A. The "Trapped Data" Problem**: Users cannot access the underlying summary statistics used to create plots. Data wrangling happens internally within plotting functions.

**B. Rigidity of InferenceData**: No codified way to perform time aggregation (monthly/yearly rollups) or dimension operations (filtering, aggregating channels).

**C. Complex Visualization Requirements**: Need to support interactive Plotly plots with arbitrary dimensions (Date, Channel, Brand, Geo) and flexible grouping/filtering.

### Proposed Solution

A three-tier decoupled system:
1. **Codified Data Wrapper**: Wraps iData with API for time resampling, dimension filtering, and grouping
2. **MMM Summary Object**: Transforms wrapped data into structured DataFrames with summary statistics
3. **Plotting Suite**: Thin Plotly functions consuming summary objects

## Summary

**Yes, the proposed architecture addresses the stated requirements**, but it needs refinement based on existing codebase patterns. The current implementation has significant gaps:
- No standardized time aggregation utilities
- Summary statistics are computed ad-hoc within each plot method
- Data wrangling logic is duplicated across 30+ plotting methods
- Users have no way to access summary DataFrames without plotting

The sandbox WIP shows good progress toward the solution with `xarray_processing_utils.py` (data wrangling) and `plotly_utils.py` (thin plotting). However, the architecture needs clearer component interfaces and backward compatibility considerations.

## Detailed Findings

### 1. Current MMM Plotting Implementation

**Primary Location**: [pymc_marketing/mmm/plot.py](pymc_marketing/mmm/plot.py)

The `MMMPlotSuite` class (line 208) contains 30+ visualization methods organized into categories:

**Predictive Analysis**:
- `posterior_predictive()` (line 430)
- `prior_predictive()` (line 520)

**Contribution Analysis**:
- `contributions_over_time()` (line 942)
- `posterior_distribution()` (line 1067)

**Saturation Curves**:
- `saturation_scatterplot()` (line 1252)
- `saturation_curves()` (line 1406)

**Additional Methods**: Budget allocation, sensitivity analysis, uplift curves, waterfall decomposition, cross-validation, etc.

**Model Class Methods**: [pymc_marketing/mmm/base.py:573-1371](pymc_marketing/mmm/base.py#L573-L1371) has 8 plot_* methods, [pymc_marketing/mmm/mmm.py:1531-3114](pymc_marketing/mmm/mmm.py#L1531-L3114) has 7 plot_* methods.

**Key Issues**:
1. All methods use **matplotlib/seaborn**, not Plotly
2. Data wrangling happens **inside each plot method**
3. No way for users to get the underlying DataFrame
4. Logic duplication across similar plot types

**Example of Data Trapped Inside Plotting** ([base.py:635-760](pymc_marketing/mmm/base.py#L635-L760)):
```python
def plot_posterior_predictive(self, ...):
    # Data extraction happens internally
    posterior_predictive = az.extract(
        self.idata, group="posterior_predictive", combined=True
    )
    mean_predictions = posterior_predictive.mean(dim="sample")

    # Summary stats computed but not returned
    hdi_94 = az.hdi(posterior_predictive, hdi_prob=0.94)

    # Only the plot is returned, not the data
    return fig, ax
```

Users have no access to `mean_predictions` or `hdi_94` DataFrames.

### 2. InferenceData Usage Patterns

**Storage Location**: [pymc_marketing/model_builder.py:591](pymc_marketing/model_builder.py#L591)

InferenceData (idata) is the central data structure:
```python
self.idata: az.InferenceData | None = None  # Generated during fitting
```

**Access Patterns**:

1. **Property Accessors** ([model_builder.py:765-784](pymc_marketing/model_builder.py#L765-L784)):
   - `mmm.posterior` - posterior samples
   - `mmm.posterior_predictive` - predictions
   - `mmm.prior` - prior samples

2. **Direct Group Access** ([plot.py:317-318](pymc_marketing/mmm/plot.py#L317-L318)):
   ```python
   if not hasattr(self.idata, "posterior_predictive"):
       raise ValueError("No posterior_predictive data found")
   ```

3. **Dimension-Based Selection** ([plot.py:378](pymc_marketing/mmm/plot.py#L378)):
   ```python
   valid_values = self.idata.posterior.coords[key].values
   ```

**iData Groups**:
- `posterior` - Model parameters (sampled during fit)
- `posterior_predictive` - Predictions from posterior
- `constant_data` - Input features (channel_data, scaling factors)
- `fit_data` - Training data as xarray Dataset
- `observed_data` - Observed target values
- `sensitivity_analysis` - Counterfactual sweep results
- `cv_metadata` - Cross-validation metadata

**Common Operations**:

**Slicing** ([budget_optimizer.py:452-454](pymc_marketing/mmm/budget_optimizer.py#L452-L454)):
```python
idata = self.models[0].idata.isel(
    draw=slice(None, None, self.use_every_n_draw)
)
```

**Extending** ([mmm.py:2517](pymc_marketing/mmm/mmm.py#L2517)):
```python
self.idata.extend(post_pred, join="right")
```

**Adding Groups** ([sensitivity_analysis.py:287](pymc_marketing/mmm/sensitivity_analysis.py#L287)):
```python
self.idata.add_groups({"sensitivity_analysis": dataset})
```

**Key Finding**: There is **NO codified wrapper** around iData. All access is direct via properties and xarray operations.

### 3. Existing Data Summarization Patterns

**Pattern 1: Hierarchical Summary Statistics** ([evaluation.py:101-141](pymc_marketing/mmm/evaluation.py#L101-L141))

```python
def summarize_metric_distributions(
    metric_distributions: dict[str, npt.NDArray],
    hdi_prob: float = 0.94,
) -> dict[str, dict[str, float]]:
    """Returns nested dict with mean, median, std, hdi_lower, hdi_upper."""
    metric_summaries = {}
    for metric, distribution in metric_distributions.items():
        hdi = az.hdi(distribution, hdi_prob=hdi_prob)
        metric_summaries[metric] = {
            "mean": np.mean(distribution),
            "median": np.median(distribution),
            "std": np.std(distribution),
            "min": np.min(distribution),
            "max": np.max(distribution),
            f"{hdi_prob:.0%}_hdi_lower": hdi[0],
            f"{hdi_prob:.0%}_hdi_upper": hdi[1],
        }
    return metric_summaries
```

**Pattern 2: Build Contributions Function** ([utils.py:471-649](pymc_marketing/mmm/utils.py#L471-L649))

Extracts and aggregates posterior variables into wide DataFrame:
- Filters variables present in `idata.posterior`
- Reduces over sampling dimensions ("chain", "draw")
- Supports custom aggregation: "mean", "median", "sum", or callable
- Broadcasts to common grid
- Returns pandas DataFrame with channels as columns

**Pattern 3: Arviz Integration**

Common integration points throughout codebase:
- `az.summary()` - Convert InferenceData to DataFrame with stats
- `az.hdi()` - Compute highest density intervals
- `az.extract()` - Extract posterior samples
- `az.r2_score()` - Compute R-squared on posterior samples

**Key Finding**: Summary statistics are computed ad-hoc per method. There is **NO unified "summary object"** that users can access.

### 4. Time Aggregation and Dimension Handling

**Time-Based Aggregation**:

**Key Finding**: There are **NO explicit monthly/yearly rollup utilities** in the production codebase.

- Time aggregation happens at data preparation, not post-hoc
- Frequency is specified upfront (typically `freq="W-MON"` for weekly)
- `_convert_frequency_to_timedelta()` ([utils.py:181-223](pymc_marketing/mmm/utils.py#L181-L223)) converts frequency strings ('D', 'W', 'M', 'Y')
- Scaling configuration ([scaling.py](pymc_marketing/mmm/scaling.py)) includes date dimension implicitly

**Dimension Filtering**:

Implemented through xarray's `.sel()` method ([plot.py](pymc_marketing/mmm/plot.py)):
- `_filter_df_by_indexer()` (line 412-424) filters DataFrames by dimension values
- Example: `data.sel(**indexers)` where indexers is `{"country": "US", "channel": "C1"}`
- `_get_additional_dim_combinations()` (line 377-410) generates all dimension value combinations

**Multi-Dimensional Models** ([multidimensional.py](pymc_marketing/mmm/multidimensional.py)):

Supports arbitrary panel dimensions via `dims` parameter:
```python
mmm = MMM(
    date_column="date",
    channel_columns=["channel_1", "channel_2"],
    dims=("country",),  # Panel dimension
    target_column="target",
)
```

**Data Organization**:
- Uses xarray Dataset internally with dims: `("date", *self.dims, "channel")`
- `_create_xarray_from_pandas()` (lines 935-977) converts flat DataFrames to multi-indexed xarray
- Filtering happens at prediction/plotting time via `.sel()`

**Key Finding**: Dimension filtering exists but is **not codified as a wrapper API**. Users must use xarray operations directly.

### 5. Sandbox Work-in-Progress Analysis

**Location**: [pymc_marketing/sandbox/](pymc_marketing/sandbox/)

**Implemented Files**:

1. **[xarray_processing_utils.py](pymc_marketing/sandbox/xarray_processing_utils.py)** - Data wrangling utilities:
   - `process_idata_for_plotting()` - Main entry point returning `MMMPlotData` dataclass
   - `idata_var_to_summary_df_with_hdi()` - Converts xarray to Polars DataFrames with HDI
   - `select_by_data_range()` - Date filtering
   - `aggregate_by_period()` - Time aggregation (monthly, yearly)
   - `aggregate_by_coords()` - Dimension aggregation
   - `period_over_period()` - Comparison between time periods

2. **[plotly_utils.py](pymc_marketing/sandbox/plotly_utils.py)** - Thin plotting functions:
   - `plot_posterior_predictive()` - Predicted vs observed with HDI bands
   - `plot_curves()` - Generic curve plotting with HDI
   - `plot_saturation_curves()` / `plot_decay_curves()` - Specific curve types
   - `plot_bar()` - Bar charts with error bars (ROAS, contribution)
   - `plot_contribution_vs_roas()` - Scatter plot with error bars
   - Helper functions: `_get_hdi_columns()`, `_plot_hdi_band()`, date formatting utilities

**Data Structure**: `MMMPlotData` dataclass (eager evaluation):
```python
@dataclass
class MMMPlotData:
    posterior_predictive: pl.DataFrame
    saturation_curves: pl.DataFrame
    decay_curves: pl.DataFrame
    roas: pl.DataFrame
    contribution: pl.DataFrame
    # ... etc
```

**Design Documents**:

1. **[MMMPLOTDATA_DESIGN.md](pymc_marketing/sandbox/MMMPLOTDATA_DESIGN.md)** - Proposes `MMMSummaryData` class with:
   - **Lazy evaluation** - Compute DataFrames on-demand with caching
   - Built-in caching mechanism
   - Optional MMM model dependency for curve generation
   - Supports both `xr.DataTree` and `az.InferenceData` inputs
   - 8+ lazy properties for different summary views

2. **[PLOTTING_SUITE_DESIGN.md](pymc_marketing/sandbox/PLOTTING_SUITE_DESIGN.md)** - Proposes `MMMStakeholderPlotSuite` class:
   - 10+ plotting methods consuming MMMSummaryData
   - Methods: `posterior_predictive()`, `saturation_curves()`, `roas()`, `contribution()`, etc.
   - Thin wrappers around plotly_utils functions

**Key Finding**: Sandbox has **functional approach implemented**, class-based approach is **design-only**. The functional utilities already solve the time aggregation gap!

### 6. Architectural Patterns in Existing Code

**Pattern 1: Functional Utilities with DataFrame Returns**

The CLV models show a precedent for returning structured data ([clv/models/basic.py:347-358](pymc_marketing/clv/models/basic.py#L347-L358)):
```python
def fit_summary(self, **kwargs):
    """Returns pd.DataFrame or pd.Series with summary statistics."""
    if res.chain.size == 1 and res.draw.size == 1:
        return az.summary(self.fit_result, **kwargs, kind="stats")["mean"].rename("value")
    else:
        return az.summary(self.fit_result, **kwargs)
```

Users can call `mmm.fit_summary()` to get a DataFrame.

**Pattern 2: Xarray as Return Type**

Posterior predictive methods return xarray DataArrays ([clv/models/beta_geo.py:436-500](pymc_marketing/clv/models/beta_geo.py#L436-L500)):
```python
def expected_purchases(self, ...) -> xarray.DataArray:
    """Returns xarray.DataArray shaped as (chain, draw, customer_id)."""
    # ... computation ...
    return (numerator / denominator).transpose(
        "chain", "draw", "customer_id", missing_dims="ignore"
    )
```

This preserves dimensions for further slicing/aggregation by users.

**Pattern 3: Wrapper Classes for Specialized Operations**

`BudgetOptimizer` wraps multiple MMM models ([budget_optimizer.py:289-510](pymc_marketing/mmm/budget_optimizer.py#L289-L510)):
- `OptimizerCompatibleModelWrapper` Protocol defines interface requiring `idata` attribute
- `MultiDimensionalBudgetOptimizerWrapper` merges multiple model idatas
- Provides `_merge_idata()` and `_prefix_idata()` helper methods

**Key Finding**: There is precedent for both **functional utilities** (evaluation.py) and **wrapper classes** (BudgetOptimizer).

## Evaluation of Proposed Architecture

### Question 1: Does the solution answer the requirements?

**YES**, the proposed three-component architecture addresses all stated requirements:

#### A. "Trapped Data" Problem ✅ SOLVED

**Component 2 (MMM Summary Object)** explicitly returns DataFrames of summary statistics:
- Users can call methods to get DataFrames independently of plotting
- Plotting suite consumes these DataFrames, making the data flow transparent

**Current Gap**: Sandbox `xarray_processing_utils.py` already implements this functionally via `process_idata_for_plotting()`, but users must know to call it.

#### B. Rigidity of InferenceData ✅ SOLVED

**Component 1 (Codified Data Wrapper)** provides standardized operations:
- Time aggregation: Sandbox has `aggregate_by_period(period="monthly"|"yearly")`
- Field operations: Sandbox has `aggregate_by_coords(coords=["channel1", "channel2"], new_label="Social")`
- Date filtering: Sandbox has `select_by_data_range(start_date, end_date)`

**Current Gap**: These utilities exist but are not wrapped in a reusable object. No production equivalent.

#### C. Complex Visualization Requirements ✅ SOLVED

**Component 3 (Plotting Suite)** with thin Plotly functions:
- Sandbox `plotly_utils.py` demonstrates this approach
- Functions accept DataFrames with specific schema (date, channel, mean, HDI columns)
- Multi-dimensional support via `color` parameter for grouping
- Layout logic can be added as subplot wrappers

**Current Gap**: Production uses matplotlib with thick plot methods. Sandbox Plotly is WIP.

### Question 2: Feedback and Suggested Improvements

#### Strengths of Proposed Architecture

1. **Clear Separation of Concerns**: Data wrangling → Summarization → Visualization is a clean pipeline
2. **Addresses Real Pain Points**: The "trapped data" problem is real and pervasive
3. **Aligns with Sandbox Progress**: The functional utilities in sandbox validate the approach
4. **Extensibility**: Users can create custom plots from summary DataFrames
5. **Testability**: Each component can be unit tested independently

#### Weaknesses and Gaps

1. **Lazy vs Eager Evaluation Ambiguity**:
   - Design doc proposes lazy `MMMSummaryData` class
   - Sandbox implements eager `MMMPlotData` dataclass
   - **Recommendation**: Start with **eager evaluation** for simplicity. Add lazy evaluation if performance issues arise with large models.

2. **No Clear Interface Contract**:
   - What DataFrame schema does Component 2 produce?
   - What schema does Component 3 expect?
   - **Recommendation**: Document the DataFrame schema explicitly (columns, types, index structure)

3. **Backward Compatibility Not Addressed**:
   - Existing 30+ matplotlib plot methods in production
   - Users depend on these methods
   - **Recommendation**: Implement new architecture in parallel, mark old methods as deprecated, provide migration guide

4. **Multi-Dimensional Handling Unclear**:
   - How does aggregation work with extra dimensions (geo, brand)?
   - Should wrapper support `.groupby()` style operations?
   - **Recommendation**: Support dimension parameter in wrapper methods: `summary.contribution(groupby="brand", aggregate_over=["geo"])`

5. **Missing Component Interactions**:
   - Does Component 3 access Component 1 or only Component 2?
   - Can users skip Component 2 and plot directly from Component 1?
   - **Recommendation**: Make Component 2 the **single source of truth**. Component 3 should only consume Component 2 outputs.

6. **No Time Granularity Validation**:
   - What happens when aggregating weekly data to monthly when dates don't align?
   - How to handle partial months?
   - **Recommendation**: Add validation and warnings in aggregation utilities

7. **HDI Probability Not Configurable Enough**:
   - Sandbox uses fixed column names like `abs_error_90_lower`
   - What if user wants multiple HDI levels (50%, 90%, 95%)?
   - **Recommendation**: Support multiple HDI levels in summary object, let plotting functions select which to display

8. **Polars vs Pandas DataFrame Return Type**:
   - Sandbox functions return Polars DataFrames (`pl.DataFrame`)
   - PyMC-Marketing currently only supports Pandas throughout the codebase
   - Users expect Pandas DataFrames from existing methods (e.g., `fit_summary()`, `az.summary()`)
   - This creates API inconsistency and potential user friction
   - **Decision Required**: Choose one of:
     - **(a) Pandas only**: Convert sandbox to return Pandas for consistency with existing codebase
     - **(b) Polars only**: Migrate entire codebase to Polars (breaking change, significant effort)
     - **(c) Dual support**: Return Polars internally, provide `.to_pandas()` convenience or `return_type` parameter
     - **(d) User configurable**: Global config setting for default return type
   - **Recommendation**: Start with **(a) Pandas only** for backward compatibility. Polars can be used internally for performance if needed, but public API should return Pandas until a broader Polars migration strategy is decided.

#### Specific Implementation Suggestions

**Component 1: Codified Data Wrapper**

```python
class MMMDataWrapper:
    """Wrapper around InferenceData with MMM-specific operations."""

    def __init__(self, idata: az.InferenceData, mmm: Optional[MMM] = None):
        self.idata = idata
        self.mmm = mmm  # Optional, for generating curves

    def filter_dates(self, start_date, end_date) -> "MMMDataWrapper":
        """Return new wrapper with filtered date range."""
        # Uses xarray .sel()

    def aggregate_time(self, period: Literal["weekly", "monthly", "yearly"]) -> "MMMDataWrapper":
        """Return new wrapper with aggregated time dimension."""
        # Uses pandas resample or xarray groupby

    def filter_dimensions(self, **kwargs) -> "MMMDataWrapper":
        """Return new wrapper with filtered dimensions (channel, geo, etc)."""
        # e.g., filter_dimensions(channel=["TV", "Radio"], geo="US")

    def aggregate_dimensions(self, dim: str, values: list, new_label: str) -> "MMMDataWrapper":
        """Combine multiple dimension values into one."""
        # e.g., aggregate_dimensions("channel", ["FB", "IG"], "Social")
```

**Component 2: MMM Summary Object**

```python
class MMMSummary:
    """Produces summary DataFrames from MMMDataWrapper."""

    def __init__(self, data: MMMDataWrapper, hdi_prob: float = 0.94):
        self.data = data
        self.hdi_prob = hdi_prob

    def posterior_predictive(self) -> pl.DataFrame:
        """Return DataFrame with columns: date, mean, hdi_lower, hdi_upper, observed."""

    def contributions(self, groupby: Optional[str] = None) -> pl.DataFrame:
        """Return DataFrame with columns: date, channel, mean, hdi_lower, hdi_upper.

        If groupby specified (e.g., "brand"), adds that dimension as a column.
        """

    def roas(self, groupby: Optional[str] = None) -> pl.DataFrame:
        """Return DataFrame with columns: channel, mean, hdi_lower, hdi_upper."""

    # ... other summary methods ...
```

**Component 3: Plotting Suite**

```python
# Thin functions consuming MMMSummary DataFrames

def plot_posterior_predictive(
    df: pl.DataFrame,
    hdi_prob: float = 0.94,
    **plotly_kwargs
) -> go.Figure:
    """Plot posterior predictive with HDI band.

    Expected schema:
    - date: datetime column
    - mean: float (predicted mean)
    - observed: float (actual values)
    - abs_error_{prob}_lower: float (HDI lower bound)
    - abs_error_{prob}_upper: float (HDI upper bound)
    """
    # Implementation in sandbox/plotly_utils.py already exists!

def plot_contribution_bar(
    df: pl.DataFrame,
    hdi_prob: Optional[float] = None,
    x: str = "channel",
    color: Optional[str] = None,
    **plotly_kwargs
) -> go.Figure:
    """Plot contribution bar chart with optional grouping."""
    # Implementation in sandbox/plotly_utils.py already exists!
```

#### Critical Design Decision: Lazy vs Eager

**Eager Evaluation (Current Sandbox)**:
```python
summary = MMMSummary(data)
summary.contributions()  # Always recomputes
summary.contributions()  # Always recomputes again
```

**Lazy Evaluation (Proposed in Design Doc)**:
```python
summary = MMMSummary(data)
summary.contributions  # Property, computed on first access, cached
summary.contributions  # Returns cached result
```

**Recommendation**:
- **Start with eager** (functional approach) for simplicity
- Measure performance on large models (1000+ days, 20+ channels, 10+ dimensions)
- Add lazy evaluation + caching **only if** needed

Reasoning: Premature optimization. The functional approach in sandbox is simpler and easier to test.

#### Handling Backward Compatibility

**Phase 1**: Implement new architecture in parallel
- Add `MMMDataWrapper`, `MMMSummary` classes to `pymc_marketing.mmm.summary` module
- Add Plotly plotting functions to `pymc_marketing.mmm.plot_plotly` module
- Keep existing matplotlib `MMMPlotSuite` unchanged

**Phase 2**: Add deprecation warnings
- Mark old plot methods with `@deprecated` decorator
- Point users to new equivalents in docstrings

**Phase 3**: Migration guide
- Document migration path in release notes
- Provide code examples showing old vs new approach

**Phase 4**: Remove deprecated methods (2-3 releases later)

### Question 3: Is there a better solution?

The proposed architecture is **fundamentally sound**, but I recommend a **hybrid approach** combining the best of current codebase patterns and sandbox WIP:

#### Recommended Architecture

**Tier 1: Functional Data Utilities** (Based on sandbox `xarray_processing_utils.py`)

Keep as pure functions in `pymc_marketing.mmm.data_transforms` module:
```python
# Time operations
select_by_data_range(idata, start, end) -> az.InferenceData
aggregate_by_period(idata, period) -> az.InferenceData

# Dimension operations
filter_by_coords(idata, **filters) -> az.InferenceData
aggregate_coords(idata, coord, values, new_label) -> az.InferenceData

# Summary statistics
compute_summary_stats(data: xr.DataArray, hdi_prob: float) -> pl.DataFrame
```

**Tier 2: Convenience Wrapper (Optional)**

For users who want method chaining:
```python
class MMMDataView:
    """Lightweight view for chaining operations."""
    def __init__(self, idata): ...
    def filter_dates(self, start, end): return MMMDataView(filter_dates(self.idata, start, end))
    def aggregate_time(self, period): return MMMDataView(aggregate_time(self.idata, period))
    # ... etc
```

Usage: `view.filter_dates("2024-01-01", "2024-12-31").aggregate_time("monthly")`

**Tier 3: Summary Factory Functions**

Replace "summary object" class with factory functions:
```python
def create_posterior_predictive_summary(
    idata: az.InferenceData,
    hdi_prob: float = 0.94,
) -> pl.DataFrame:
    """Extract posterior predictive with HDI into DataFrame."""

def create_contribution_summary(
    idata: az.InferenceData,
    hdi_prob: float = 0.94,
    groupby: Optional[str] = None,
) -> pl.DataFrame:
    """Extract channel contributions with HDI into DataFrame."""
```

**Tier 4: Thin Plotting Functions** (Already in sandbox)

Keep as-is from `plotly_utils.py`.

#### Why This Is Better

1. **Simpler**: No complex class hierarchies, just functions
2. **Testable**: Pure functions are easier to unit test
3. **Flexible**: Users can compose operations as needed
4. **Familiar**: Aligns with pandas/xarray functional style
5. **Incremental**: Can add optional wrapper later without breaking changes
6. **Existing Code**: Sandbox already implements this approach!

#### Integration with Existing Model Classes

Add convenience methods to `MMM` class that use the new utilities:

```python
class MMM(BaseValidateMMM):
    # ... existing methods ...

    def get_posterior_predictive_summary(self, hdi_prob: float = 0.94) -> pl.DataFrame:
        """Get posterior predictive summary as DataFrame."""
        return create_posterior_predictive_summary(self.idata, hdi_prob)

    def get_contribution_summary(self, hdi_prob: float = 0.94, **filters) -> pl.DataFrame:
        """Get channel contribution summary as DataFrame."""
        filtered_idata = self.idata
        if filters:
            filtered_idata = filter_by_coords(self.idata, **filters)
        return create_contribution_summary(filtered_idata, hdi_prob)

    def plot_contribution_interactive(self, hdi_prob: float = 0.94, **kwargs) -> go.Figure:
        """Plot contributions using Plotly (new interactive version)."""
        df = self.get_contribution_summary(hdi_prob)
        return plot_bar(df, hdi_prob=hdi_prob, **kwargs)
```

Users can now:
```python
# Get data only
df = mmm.get_contribution_summary()

# Get data and plot
df = mmm.get_contribution_summary()
fig = plot_bar(df, hdi_prob=0.94)

# Or do it in one step
fig = mmm.plot_contribution_interactive()
```

## Code References

### Current Implementation
- [pymc_marketing/mmm/plot.py:208-3453](pymc_marketing/mmm/plot.py#L208-L3453) - MMMPlotSuite class with 30+ methods
- [pymc_marketing/mmm/base.py:573-1371](pymc_marketing/mmm/base.py#L573-L1371) - Base model plot methods
- [pymc_marketing/mmm/utils.py:471-649](pymc_marketing/mmm/utils.py#L471-L649) - build_contributions() utility
- [pymc_marketing/mmm/evaluation.py:101-141](pymc_marketing/mmm/evaluation.py#L101-L141) - Summary statistics pattern
- [pymc_marketing/mmm/multidimensional.py:935-1452](pymc_marketing/mmm/multidimensional.py#L935-L1452) - Multi-dimensional data handling
- [pymc_marketing/model_builder.py:591-784](pymc_marketing/model_builder.py#L591-L784) - InferenceData storage and accessors

### Sandbox WIP
- [pymc_marketing/sandbox/xarray_processing_utils.py](pymc_marketing/sandbox/xarray_processing_utils.py) - Data wrangling utilities (IMPLEMENTED)
- [pymc_marketing/sandbox/plotly_utils.py:261-572](pymc_marketing/sandbox/plotly_utils.py#L261-L572) - Thin Plotly functions (IMPLEMENTED)
- [pymc_marketing/sandbox/MMMPLOTDATA_DESIGN.md](pymc_marketing/sandbox/MMMPLOTDATA_DESIGN.md) - Lazy evaluation design (DESIGN ONLY)
- [pymc_marketing/sandbox/PLOTTING_SUITE_DESIGN.md](pymc_marketing/sandbox/PLOTTING_SUITE_DESIGN.md) - Class-based suite design (DESIGN ONLY)

## Architecture Insights

### Current Pain Points
1. **Data Duplication**: Same data wrangling logic repeated in 30+ plot methods
2. **No User Access**: Users cannot get summary DataFrames without plotting
3. **Matplotlib Lock-in**: Hard to support Plotly or other visualization libraries
4. **No Time Aggregation**: Must prepare data at desired granularity upfront
5. **Ad-hoc Filtering**: Each plot method implements dimension filtering differently

### Design Patterns Observed
1. **Functional Utilities**: evaluation.py, utils.py use pure functions returning DataFrames
2. **Wrapper Classes**: BudgetOptimizer, SensitivityAnalysis wrap idata with specialized methods
3. **Property Accessors**: ModelBuilder provides `.posterior`, `.prior` convenience properties
4. **Xarray as Return Type**: CLV models return xarray DataArrays preserving dimensions

### Architectural Trade-offs

**Class-based (Proposed in Design Docs)**:
- ✅ Method chaining, fluent interface
- ✅ State management (caching)
- ❌ More complex, harder to test
- ❌ Tight coupling between components

**Functional (Implemented in Sandbox)**:
- ✅ Simple, composable
- ✅ Easy to test and reason about
- ✅ Aligns with xarray/pandas style
- ❌ No built-in caching
- ❌ More verbose for complex chains

**Recommendation**: Start with functional approach, add optional class wrapper for convenience.

### Multi-Dimensional Complexity

The codebase already handles multi-dimensional models well:
- Dimensions declared via `dims=("country", "geo", ...)`
- Xarray coordinates preserve dimension metadata
- Filtering via `.sel()` is standard

The proposed wrapper should **leverage existing xarray operations** rather than reimplementing:
```python
def filter_by_coords(idata, **filters):
    """Thin wrapper around xarray .sel()."""
    filtered_posterior = idata.posterior.sel(**filters)
    # Create new InferenceData with filtered group
    return az.InferenceData(posterior=filtered_posterior, ...)
```

## Related Research

This research connects to several ongoing architectural themes:
- **Component-based MMM models** - Each component (adstock, saturation) has transformation logic
- **Budget optimization multi-model** - Already demonstrates idata merging patterns
- **Sensitivity analysis storage** - Shows how to extend idata with custom groups
- **Cross-validation fold combination** - Demonstrates concat along new dimensions

## Open Questions

1. **Performance**: What is the performance impact of eager DataFrame computation for large models (1000+ days, 50+ channels, 10 dims)? Need benchmarks.

2. **Schema Evolution**: How do we version the DataFrame schemas returned by summary functions? What if we add new columns?

3. **Interoperability**: Should summary DataFrames use Polars (as in sandbox) or Pandas (as in current code)? Polars is faster but Pandas is more familiar.

4. **Plotly Theme**: Should we create a standard PyMC-Marketing Plotly theme for consistency?

5. **Matplotlib Deprecation Timeline**: How long should we support old matplotlib methods before removing?

6. **Documentation Migration**: How do we update 100+ documentation examples that use old plot methods?

## Recommendations

### Immediate Actions

1. **Promote sandbox utilities to production**:
   - Move `xarray_processing_utils.py` → `pymc_marketing/mmm/data_transforms.py`
   - Move `plotly_utils.py` → `pymc_marketing/mmm/plot_interactive.py`
   - Add comprehensive tests and documentation

2. **Add convenience methods to MMM class**:
   - `get_posterior_predictive_summary()` → Returns DataFrame
   - `get_contribution_summary()` → Returns DataFrame
   - `get_roas_summary()` → Returns DataFrame
   - Each returns structured Polars/Pandas DataFrame

3. **Create migration guide**:
   - Document old vs new approach
   - Provide code examples
   - List feature parity matrix

4. **Benchmark performance**:
   - Test on large models to validate eager evaluation is sufficient
   - If slow, add caching layer as separate concern

### Medium-Term (Next 2-3 Releases)

1. **Add optional method-chaining wrapper** (`MMMDataView` or similar)
2. **Deprecate matplotlib plot methods** in favor of new interactive methods
3. **Extend to other model types** (CLV, time-varying parameter models)
4. **Add multi-HDI support** (e.g., show both 50% and 90% intervals)

### Long-Term

1. **Remove deprecated matplotlib methods**
2. **Standardize DataFrame schemas** across all summary functions
3. **Create comprehensive plotting gallery** with Plotly examples
4. **Publish design patterns guide** for extending the framework

## Conclusion

The proposed three-component architecture **correctly identifies the problems** and provides a **sound solution framework**. However, the implementation should favor **simplicity over abstraction**:

- ✅ **Use functional utilities** from sandbox (already implemented and working)
- ✅ **Add convenience methods** to MMM class for common use cases
- ✅ **Keep plotting functions thin** consuming DataFrames
- ⚠️ **Defer class-based wrappers** until proven necessary
- ⚠️ **Start with eager evaluation** and optimize later if needed

The sandbox work validates this approach and provides a strong foundation. The main remaining work is:
1. Productionizing the sandbox utilities with tests and docs
2. Integrating with existing MMM class via convenience methods
3. Managing backward compatibility with existing matplotlib plots

This will solve all three stated problems while maintaining simplicity and alignment with existing codebase patterns.
