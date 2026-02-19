---
date: 2026-01-08T20:09:03Z
researcher: Claude Sonnet 4.5
git_commit: c1a8a3828e0e3928572c9da6553c0b7bac9705f1
branch: isofer/plotting-design
repository: pymc-marketing
topic: "Component 2 (MMM Summary Object) Implementation Design"
tags: [research, codebase, mmm, component2, summary-object, plotting, dataframe-schemas, integration-patterns, narwhals, polars, pandas]
status: complete
last_updated: 2026-01-09
last_updated_by: Claude Opus 4.5
last_updated_note: "Added Narwhals support for dual Pandas/Polars DataFrame output via output_format parameter"
related_research: ["2026-01-07_21-08-47_mmm-data-plotting-framework-architecture.md", "2026-01-07_22-19-51_codifying-idata-conventions-component1-data.md"]
---

# Research: Component 2 (MMM Summary Object) Implementation Design

**Date**: 2026-01-08T20:09:03Z
**Researcher**: Claude Sonnet 4.5
**Git Commit**: c1a8a3828e0e3928572c9da6553c0b7bac9705f1
**Branch**: isofer/plotting-design
**Repository**: pymc-marketing

---

## Table of Contents

1. [Research Question](#research-question)
   - [Context](#context)
2. [Summary](#summary)
   - [Key Findings](#key-findings)
   - [Recommended Architecture](#recommended-architecture)
3. [Detailed Findings](#detailed-findings)
   - [Finding 1: Sandbox Implementation Analysis](#finding-1-sandbox-implementation-analysis)
   - [Finding 2: Existing Summary Statistics Patterns](#finding-2-existing-summary-statistics-patterns)
   - [Finding 3: DataFrame Validation Patterns](#finding-3-dataframe-validation-patterns)
   - [Finding 4: Polars vs Pandas Decision](#finding-4-polars-vs-pandas-decision)
   - [Finding 5: Integration Patterns with MMM Models](#finding-5-integration-patterns-with-mmm-models)
   - [Finding 6: Plotly Function Requirements](#finding-6-plotly-function-requirements)
4. [Implementation](#implementation)
   - [Data Flow](#data-flow)
   - [Benefits of This Architecture](#benefits-of-this-architecture)
   - [Factory Functions](#factory-functions)
   - [MMM Integration](#mmm-integration)
   - [Complete Usage Example](#complete-usage-example)
5. [Code References](#code-references)
6. [Open Questions](#open-questions)
7. [Conclusion](#conclusion)
   - [Architecture Benefits](#architecture-benefits)
   - [Design Decisions](#design-decisions)
   - [Implementation Checklist](#implementation-checklist)

---

## Research Question

How should we implement Component 2 (MMM Summary Object) from the three-component MMM Data & Plotting Framework Architecture?

### Context

The [MMM Data & Plotting Framework Architecture](2026-01-07_21-08-47_mmm-data-plotting-framework-architecture.md) identified three key components:
1. **Component 1**: Codified Data Wrapper (completed research in [Component 1 document](2026-01-07_22-19-51_codifying-idata-conventions-component1-data.md))
2. **Component 2**: MMM Summary Object - Transforms wrapped data into structured DataFrames with summary statistics
3. **Component 3**: Plotting Suite - Thin Plotly functions consuming summary objects

**Key Constraints**:
- Component 2 will **only support Plotly plotting functions** (new interactive plots)
- **MMMPlotSuite (matplotlib) will NOT be deprecated** - it will continue to exist alongside the new Plotly plotting suite. Users can choose which to use based on their needs.
- Must integrate cleanly with Component 1 (MMMIDataWrapper)
- Should follow existing codebase patterns

## Summary

**Component 2 should be implemented as factory functions consuming MMMIDataWrapper (Component 1)**, returning **either Pandas or Polars DataFrames** controlled by an `output_format` parameter.

**Note on Narwhals**: Narwhals is used by **Component 3 (Plotting)**, not Component 2. Component 3 uses Narwhals to accept either Pandas or Polars DataFrames as input. Component 2 simply converts output format using `pl.from_pandas()` when Polars is requested.

### Key Findings

1. **Component 2 must consume Component 1** - Takes `MMMIDataWrapper` as input, not raw `InferenceData`
2. **Output format is configurable** - `output_format` parameter controls Pandas vs Polars output
3. **Functional approach is validated** - sandbox demonstrates this works well
4. **DataFrame schemas are well-defined** - clear conventions for HDI columns and summary stats
5. **Integration pattern identified** - property-based access (like `mmm.plot`) is preferred
6. **Component 3 uses Narwhals** - Plotting functions accept either DataFrame type via Narwhals

### Recommended Architecture

**Data Flow**:
```
az.InferenceData
  ↓
Component 1: MMMIDataWrapper (filtering, scaling, aggregation)
  ↓
Component 2: Summary Functions (data → DataFrames with stats)
  ↓                    ↓ (output_format="pandas" | "polars")
  ↓              pd.DataFrame or pl.DataFrame
  ↓
Component 3: Plotly Functions (visualize DataFrames)
```

**Tier 1: Factory Functions** (Core Implementation)
```python
# Module: pymc_marketing/mmm/summary.py
from typing import Literal
from narwhals.typing import IntoDataFrame

# Type alias for output format
OutputFormat = Literal["pandas", "polars"]

def create_posterior_predictive_summary(
    data: MMMIDataWrapper,  # ← Takes Component 1, not idata
    hdi_probs: list[float] = [0.94],
    output_format: OutputFormat = "pandas",  # ← Configurable output
) -> IntoDataFrame:
    """Extract posterior predictive with HDI into DataFrame."""

def create_contribution_summary(
    data: MMMIDataWrapper,  # ← Takes Component 1, not idata
    hdi_probs: list[float] = [0.94],
    component: Literal["channel", "control", "seasonality"] = "channel",
    output_format: OutputFormat = "pandas",  # ← Configurable output
) -> IntoDataFrame:
    """Extract contributions with HDI into DataFrame."""

def create_roas_summary(
    data: MMMIDataWrapper,  # ← Takes Component 1, not idata
    hdi_probs: list[float] = [0.94],
    output_format: OutputFormat = "pandas",  # ← Configurable output
) -> IntoDataFrame:
    """Compute ROAS (contribution / spend) with HDI."""
```

**Tier 2: Convenience Property** (MMM Integration)
```python
# In MMM class
@property
def summary(self) -> MMMSummaryFactory:
    """Access summary data generation functions."""
    # Pass the data data (Component 1), not raw idata
    return MMMSummaryFactory(data=self.data)

# Usage with Component 1 filtering/aggregation
data = mmm.data.filter_dates("2025-01", "2025-06").aggregate_time("monthly")
df = MMMSummaryFactory(data).contributions()  # Returns Pandas by default

# Get Polars DataFrame instead
df_polars = MMMSummaryFactory(data, output_format="polars").contributions()

# Or via convenience property
df = mmm.summary.contributions()  # Uses mmm.data implicitly, returns Pandas
```

## Detailed Findings

### Finding 1: Sandbox Implementation Analysis

**Location**: [pymc_marketing/sandbox/xarray_processing_utils.py](pymc_marketing/sandbox/xarray_processing_utils.py)

#### Core Summary Function

The sandbox implementation provides a proven pattern for converting xarray posterior samples to summary DataFrames:

```python
def idata_var_to_summary_df_with_hdi(
    data: xr.DataArray,
    hdi_probs: float | list[float],
) -> pl.DataFrame:  # Currently Polars, should be Pandas
    """Create DataFrame from xarray data with HDI columns."""

    # 1. Compute point estimates
    mean_ = data.mean(dim=['chain', 'draw'])
    median_ = data.median(dim=['chain', 'draw'])

    # 2. Compute HDI for each probability level
    for hdi_prob in hdi_probs:
        hdi = az.stats.hdi(data, hdi_prob)
        # HDI returns DataArray with 'hdi' dimension: [lower, upper]

    # 3. Combine into DataFrame with standard column names
    return pl.DataFrame({
        'mean': mean_.values,
        'median': median_.values,
        'abs_error_80_lower': hdi.isel(hdi=0).values,
        'abs_error_80_upper': hdi.isel(hdi=1).values,
        # ... additional HDI levels
    })
```

**Key Insights**:
- Uses `az.stats.hdi()` for highest density intervals
- Computes both mean and median (not just one)
- Supports multiple HDI probability levels in same DataFrame
- Uses absolute bounds (not relative errors) - conversion happens at plot time
- Collapses MCMC dimensions (chain, draw) but preserves data dimensions (date, channel)

#### DataFrame Schema Conventions

**Column Naming Convention** (from [sandbox/xarray_processing_utils.py:15-17](pymc_marketing/sandbox/xarray_processing_utils.py#L15-L17)):
```
abs_error_{prob}_lower, abs_error_{prob}_upper
```

**Example**: For `hdi_probs=[0.80, 0.90, 0.95]`:
- `abs_error_80_lower`, `abs_error_80_upper`
- `abs_error_90_lower`, `abs_error_90_upper`
- `abs_error_95_lower`, `abs_error_95_upper`

**Posterior Predictive DataFrame**:
```python
{
    'date': datetime64[ns],            # Time dimension
    'mean': float64,                   # Predicted mean
    'median': float64,                 # Predicted median
    'observed': float64,               # Actual values (from fit_data)
    'abs_error_94_lower': float64,    # HDI lower bound
    'abs_error_94_upper': float64,    # HDI upper bound
}
```

**Contribution DataFrame**:
```python
{
    'date': datetime64[ns],            # Optional - can aggregate
    'channel': str,                    # Channel dimension
    'mean': float64,                   # Contribution mean
    'median': float64,                 # Contribution median
    'abs_error_94_lower': float64,    # HDI bounds
    'abs_error_94_upper': float64,
}
```

**ROAS DataFrame**:
```python
{
    'date': datetime64[ns],            # Optional
    'channel': str,                    # Channel dimension
    'mean': float64,                   # ROAS mean (contribution / spend)
    'median': float64,                 # ROAS median
    'abs_error_94_lower': float64,    # HDI bounds
    'abs_error_94_upper': float64,
}
```

**Channel Spend DataFrame** (no HDI - raw data):
```python
{
    'date': datetime64[ns],
    'channel': str,
    'channel_data': float64            # Raw spend values
}
```

---

### Finding 2: Existing Summary Statistics Patterns

The codebase has multiple patterns for generating summary statistics from InferenceData:

#### Pattern A: Wide DataFrame with Aggregated Statistics
**Location**: [pymc_marketing/mmm/utils.py:471-649](pymc_marketing/mmm/utils.py#L471-L649)

```python
def build_contributions(
    idata,
    var: list[str] | tuple[str, ...],
    agg: str | Callable = "mean",
    *,
    agg_dims: list[str] | tuple[str, ...] | None = None,
    index_dims: list[str] | tuple[str, ...] | None = None,
    expand_dims: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build a wide contributions DataFrame from idata.posterior variables.

    Returns wide-format DataFrame with channels as columns:

        date | channel__C1 | channel__C2 | control__x1 | intercept
        -----|-------------|-------------|-------------|----------
        2024 |    1234.5   |    567.8    |    89.2     |   100.0
    """
```

**Key aspects**:
- Pivots channels to wide format (channel__C1, channel__C2 columns)
- Aggregates over sampling dimensions using mean/median/custom function
- Broadcasts and merges multiple variables
- Memory efficient with category dtypes

#### Pattern B: Nested Dict with HDI
**Location**: [pymc_marketing/mmm/evaluation.py:101-141](pymc_marketing/mmm/evaluation.py#L101-L141)

```python
def summarize_metric_distributions(
    metric_distributions: dict[str, npt.NDArray],
    hdi_prob: float = 0.94,
) -> dict[str, dict[str, float]]:
    """Returns nested dict with summary statistics.

    Returns:
        {
            "r_squared": {
                "mean": 0.9055,
                "median": 0.9061,
                "std": 0.0098,
                "min": 0.8669,
                "max": 0.9371,
                "94%_hdi_lower": 0.8891,
                "94%_hdi_upper": 0.9198,
            },
            # ... more metrics
        }
    """
```

**Key aspects**:
- Computes 7 summary statistics per metric
- Dynamic HDI label formatting
- Used for evaluation metrics, not time series

#### Pattern C: Direct xarray → DataFrame Conversion
**Location**: Multiple locations (e.g., [mmm/base.py:970-1066](pymc_marketing/mmm/base.py#L970-L1066))

```python
# Pattern used in compute_mean_contributions_over_time()
contributions = (
    az.extract(idata.posterior, var_names=[var])
    .mean("sample")           # Collapse MCMC samples
    .to_dataframe()           # xarray → pandas
    .unstack()                # Pivot to wide format
)
```

**Key Pattern**: xarray → aggregate → pandas → reshape

---

### Finding 3: DataFrame Validation Patterns

The codebase uses multiple approaches for validating DataFrame schemas:

#### Pattern A: Pydantic BaseModel for Configuration
**Location**: [pymc_marketing/mmm/scaling.py:21-79](pymc_marketing/mmm/scaling.py#L21-L79)

```python
from pydantic import BaseModel, Field, model_validator

class VariableScaling(BaseModel):
    """How to scale a variable."""
    method: Literal["max", "mean"]
    dims: str | tuple[str, ...]

    @model_validator(mode="after")
    def _validate_dims(self) -> Self:
        if "date" in self.dims:
            raise ValueError("dim of 'date' is already assumed")
        return self
```

**Use case**: Configuration objects that need validation and serialization

#### Pattern B: Mixin Classes for DataFrame Validation
**Location**: [pymc_marketing/mmm/validating.py:47-159](pymc_marketing/mmm/validating.py#L47-L159)

```python
class ValidateChannelColumns:
    """Validate the channel columns."""

    @validation_method_X
    def validate_channel_columns(self, data: pd.DataFrame) -> None:
        if not set(self.channel_columns).issubset(data.columns):
            raise ValueError(f"channel_columns {self.channel_columns} not in data")
```

**Use case**: Composable validation logic shared across model classes

#### Pattern C: TypedDict for Result Structures
**Location**: [pymc_marketing/mmm/causal.py:54-65](pymc_marketing/mmm/causal.py#L54-L65)

```python
from typing import TypedDict, NotRequired

class TestResult(TypedDict):
    """Conditional independence test statistics."""
    bic0: float
    bic1: float
    delta_bic: float
    independent: bool
    conditioning_set: list[str]
    forced: NotRequired[bool]
```

**Use case**: Documenting dictionary return types with known structure

#### Pattern D: Docstring Examples
**Location**: [pymc_marketing/mmm/multidimensional.py:25-103](pymc_marketing/mmm/multidimensional.py#L25-L103)

```python
"""
Examples
--------
.. code-block:: python

    X = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=8, freq="W-MON"),
        "C1": [100, 120, 90, 110, 105, 115, 98, 102],
        "C2": [80, 70, 95, 85, 90, 88, 92, 94],
    })
"""
```

**Use case**: Showing users expected DataFrame structure through examples

**Recommendation for Component 2**: Use **Docstring Examples** + **TypedDict** for schema documentation. Pydantic is overkill for return types.

---

### Finding 4: Pandas and Polars Output Support

**Analysis Location**: [Task output: "Investigate Polars vs Pandas"](#detailed-findings)

#### Current State

| Aspect | Pandas | Polars |
|--------|--------|--------|
| **Production dependency** | ✅ Required | ❌ Not installed (optional) |
| **File usage** | 218 occurrences | 47 occurrences (sandbox only) |
| **Public APIs** | All return `pd.DataFrame` | None |
| **xarray integration** | `.to_dataframe()` returns Pandas | Requires manual conversion |
| **PyMC ecosystem** | ArviZ returns Pandas | Not used |
| **Plotly integration** | Native support | Requires `.to_pandas()` |

#### Decision: Support Both via `output_format` Parameter

**Approach**: Component 2 outputs either Pandas or Polars via simple conversion.

**Rationale**:
1. **User Choice**: Users can choose Pandas or Polars based on their workflow
2. **Ecosystem Alignment**: Pandas is standard in PyMC/ArviZ ecosystem (xarray returns Pandas)
3. **Future-Proof**: Ready for growing Polars adoption in data science
4. **Polars Optional**: Polars remains an optional dependency
5. **No Forced Conversion**: Users stay in their preferred DataFrame ecosystem

**Component 2 Implementation** (simple conversion, no Narwhals):
```python
from typing import Literal, Union
import pandas as pd

OutputFormat = Literal["pandas", "polars"]

# Union type for return values
try:
    import polars as pl
    DataFrameType = Union[pd.DataFrame, pl.DataFrame]
except ImportError:
    DataFrameType = pd.DataFrame


def create_contribution_summary(
    data: MMMIDataWrapper,
    hdi_probs: list[float] | None = None,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Create contribution summary from data.

    Parameters
    ----------
    output_format : {"pandas", "polars"}
        Output DataFrame format. Defaults to "pandas".
        - "pandas": Returns pd.DataFrame (default)
        - "polars": Returns pl.DataFrame (requires polars to be installed)
    """
    # Internal computation using Pandas (from xarray)
    df_pandas = _compute_summary_internal(data, hdi_probs)

    # Convert to requested format
    return _convert_output(df_pandas, output_format)


def _convert_output(df: pd.DataFrame, output_format: OutputFormat) -> DataFrameType:
    """Convert Pandas DataFrame to requested output format."""
    if output_format == "pandas":
        return df
    elif output_format == "polars":
        try:
            import polars as pl
            return pl.from_pandas(df)
        except ImportError:
            raise ImportError(
                "Polars is required for output_format='polars'. "
                "Install it with: pip install pymc-marketing[polars]"
            )
    else:
        raise ValueError(f"Unknown output_format: {output_format}. Use 'pandas' or 'polars'.")
```

#### Narwhals: Used by Component 3 (Plotting), NOT Component 2

[Narwhals](https://narwhals-dev.github.io/narwhals/) is used by **Component 3 (Plotting functions)** to accept either Pandas or Polars DataFrames as input. This allows users to pass their preferred DataFrame type to plotting functions.

**Component 3 uses Narwhals** because:
- Plotting functions need to **accept** either DataFrame type
- Narwhals provides a unified API to work with both without code duplication
- Operations like filtering, sorting, column access work the same way

**Component 2 does NOT need Narwhals** because:
- It always computes internally with Pandas (xarray returns Pandas)
- It only needs to **convert output** to Polars when requested
- Simple `pl.from_pandas()` is sufficient

```python
# Component 3 (Plotting) - uses Narwhals to ACCEPT either type
import narwhals as nw
from narwhals.typing import IntoDataFrame

def plot_bar(df: IntoDataFrame, ...) -> go.Figure:
    """Plot bar chart. Accepts both Pandas and Polars DataFrames."""
    nw_df = nw.from_native(df)
    # Use Narwhals API for DataFrame operations
    x_values = nw_df[x_col].to_list()
    ...
```

#### Dependency Configuration

**pyproject.toml**:
```toml
[project]
dependencies = [
    "pandas>=2.0",
    "narwhals>=1.0",  # Required for Component 3 (plotting)
    # ... other deps
]

[project.optional-dependencies]
polars = ["polars>=0.20"]

# Users who want Polars output from Component 2:
# pip install pymc-marketing[polars]
```

---

### Finding 5: Integration Patterns with MMM Models

The codebase shows three primary integration patterns:

#### Pattern A: Property-Based Lazy Initialization (Preferred)
**Location**: [pymc_marketing/mmm/multidimensional.py:618-623](pymc_marketing/mmm/multidimensional.py#L618-L623)

```python
class MMM(BaseValidateMMM):

    @property
    def plot(self) -> MMMPlotSuite:
        """Use the MMMPlotSuite to plot the results."""
        self._validate_model_was_built()
        self._validate_idata_exists()
        return MMMPlotSuite(idata=self.idata)

    @property
    def sensitivity(self) -> SensitivityAnalysis:
        """Access sensitivity analysis functionality."""
        return SensitivityAnalysis(
            pymc_model=self.model,
            idata=self.idata,
            dims=self.dims
        )
```

**Usage**:
```python
mmm.fit(X, y)
_ = mmm.plot.contributions_over_time(var=["channel_contribution"])
_ = mmm.sensitivity.run_sweep(...)
```

**Key aspects**:
- Returns fully-configured instance on-demand
- No state stored on MMM instance
- Validation happens before instantiation
- Clean fluent API
- Each access creates new instance with current model state

**Recommendation**: **Use this pattern for Component 2**

#### Pattern B: Wrapper Classes with Protocol
**Location**: [pymc_marketing/mmm/budget_optimizer.py:288-297](pymc_marketing/mmm/budget_optimizer.py#L288-L297)

```python
@runtime_checkable
class OptimizerCompatibleModelWrapper(Protocol):
    """Protocol for models compatible with BudgetOptimizer."""
    adstock: Any
    _channel_scales: Any
    idata: InferenceData

    def _set_predictors_for_optimization(self, num_periods: int) -> Model:
        ...
```

**Use case**: When external tools need to integrate with multiple model types

#### Pattern C: Factory Functions
**Location**: Various (e.g., [mmm/utils.py:226-229](pymc_marketing/mmm/utils.py#L226-L229))

```python
def create_zero_dataset(
    model: Any,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    include_carryover: bool = True,
) -> pd.DataFrame:
    """Create a dataset with zero channel values."""
```

**Use case**: Constructing complex objects with validated configuration

---

### Finding 6: Plotly Function Requirements

**Location**: [pymc_marketing/sandbox/plotly_utils.py](pymc_marketing/sandbox/plotly_utils.py)

#### Required DataFrame Schema for Plotting

**For `plot_posterior_predictive()`**:
```python
# Expected columns
{
    'date': datetime64[ns],            # REQUIRED
    'mean': float64,                   # REQUIRED
    'observed': float64,               # REQUIRED
    'abs_error_{prob}_lower': float64, # OPTIONAL (if hdi_prob specified)
    'abs_error_{prob}_upper': float64  # OPTIONAL
}
```

**For `plot_bar()` (contributions, ROAS)**:
```python
{
    'channel': str,                    # REQUIRED (x-axis)
    'date': datetime64[ns],            # Optional (for grouping)
    'mean': float64,                   # REQUIRED (y-axis)
    'abs_error_{prob}_lower': float64, # OPTIONAL
    'abs_error_{prob}_upper': float64  # OPTIONAL
}
```

**For `plot_curves()` (saturation, decay)**:
```python
{
    'x': float64 | 'time': int64,      # x-axis (varies by curve type)
    'channel': str,                    # REQUIRED (for grouping)
    'mean': float64,                   # REQUIRED (y-axis)
    'abs_error_{prob}_lower': float64, # OPTIONAL
    'abs_error_{prob}_upper': float64  # OPTIONAL
}
```

#### HDI Column Validation

**Helper function** ([plotly_utils.py:23-51](pymc_marketing/sandbox/plotly_utils.py#L23-L51)):
```python
def _get_hdi_columns(df: pl.DataFrame, hdi_prob: float) -> tuple[str, str]:
    """Get the HDI column names for a given probability level.

    Raises ValueError if columns don't exist.
    """
    prob_str = str(int(hdi_prob * 100))  # 0.80 → "80"
    lower_col = f"abs_error_{prob_str}_lower"
    upper_col = f"abs_error_{prob_str}_upper"

    if lower_col in df.columns and upper_col in df.columns:
        return lower_col, upper_col
    raise ValueError(f"HDI columns for probability {hdi_prob} not found")
```

**Key Insight**: Plotting functions validate HDI columns exist before use.

#### Absolute vs Relative Errors

**Storage** (in DataFrames):
- Store **absolute bounds**: `abs_error_80_lower`, `abs_error_80_upper`
- More flexible (can compute different error types)

**Conversion for Plotly** ([plotly_utils.py:442-448](pymc_marketing/sandbox/plotly_utils.py#L442-L448)):
```python
# Plotly expects relative errors (distance from mean)
if hdi_prob is not None:
    lower_col, upper_col = _get_hdi_columns(df, hdi_prob)
    pdf["error_upper"] = pdf[upper_col] - pdf["mean"]  # Relative
    pdf["error_lower"] = pdf["mean"] - pdf[lower_col]  # Relative
```

---

## Implementation

### Data Flow

```
User Request
  ↓
mmm.fit(X, y)  # Creates az.InferenceData
  ↓
Component 1: mmm.data → MMMIDataWrapper
  ├─ .filter_dates("2025-01", "2025-06")  # Filtering
  ├─ .filter_dims(channel=["TV", "Radio"])  # Dimension filtering
  ├─ .aggregate_time("monthly")  # Time aggregation
  └─ Returns: filtered/aggregated MMMIDataWrapper
  ↓
Component 2: MMMSummaryFactory(data)
  ├─ .posterior_predictive() → DataFrame
  ├─ .contributions() → DataFrame
  └─ .roas() → DataFrame
  ↓
Component 3: Plotly plotting functions
  └─ plot_bar(df, hdi_prob=0.94) → go.Figure
```

### Benefits of This Architecture

1. **✅ Proper separation**: Component 1 handles data access/transformation, Component 2 handles summarization
2. **✅ No duplicate logic**: Scaling, filtering, aggregation live in one place (Component 1)
3. **✅ Component 2 is simpler**: Just xarray → DataFrame + HDI computation
4. **✅ Composable**: `mmm.data.filter(...).aggregate(...) → factory → plot`
5. **✅ Wrapper encapsulates complexity**: All idata conventions hidden in Component 1

### Factory Functions

**File**: `pymc_marketing/mmm/summary.py`

```python
"""Summary DataFrame generation for MMM models (Component 2).

This module transforms MMMIDataWrapper (Component 1) into structured
DataFrames with summary statistics for plotting (Component 3).

Key Features:
- `output_format` parameter: Choose between Pandas and Polars DataFrames
- `frequency` parameter: View data at different aggregation levels

Note: This module does NOT use Narwhals. Narwhals is used by Component 3
(plotting functions) to accept either DataFrame type as input.

Examples
--------
>>> # Get summary as Pandas (default)
>>> df = create_contribution_summary(mmm.data)
>>> type(df)
<class 'pandas.DataFrame'>

>>> # Get summary as Polars
>>> df = create_contribution_summary(mmm.data, output_format="polars")
>>> type(df)
<class 'polars.DataFrame'>
"""

from typing import Literal, Union
import pandas as pd
import xarray as xr
import arviz as az
from pymc_marketing.mmm.idata_wrapper import MMMIDataWrapper

# Type aliases
Frequency = Literal["original", "yearly", "monthly", "weekly", "all_time"] | None
OutputFormat = Literal["pandas", "polars"]

# Union type for return values (actual type depends on output_format)
try:
    import polars as pl
    DataFrameType = Union[pd.DataFrame, pl.DataFrame]
except ImportError:
    DataFrameType = pd.DataFrame


# ==================== Output Format Conversion ====================

def _convert_output(df: pd.DataFrame, output_format: OutputFormat) -> DataFrameType:
    """Convert Pandas DataFrame to requested output format.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (always Pandas from internal computation)
    output_format : {"pandas", "polars"}
        Desired output format

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        DataFrame in requested format

    Raises
    ------
    ImportError
        If output_format="polars" but polars is not installed
    ValueError
        If output_format is not recognized
    """
    if output_format == "pandas":
        return df
    elif output_format == "polars":
        try:
            import polars as pl
            return pl.from_pandas(df)
        except ImportError:
            raise ImportError(
                "Polars is required for output_format='polars'. "
                "Install it with: pip install pymc-marketing[polars]"
            )
    else:
        raise ValueError(
            f"Unknown output_format: {output_format!r}. Use 'pandas' or 'polars'."
        )


# ==================== Factory Functions ====================

def create_posterior_predictive_summary(
    data: MMMIDataWrapper,
    hdi_probs: list[float] | None = None,
    frequency: Frequency = None,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Create posterior predictive summary from data.

    Parameters
    ----------
    data : MMMIDataWrapper
        Data wrapper (Component 1) with filtered/aggregated data
    hdi_probs : list of float, optional
        HDI probability levels. Defaults to [0.94].
    frequency : {"original", "yearly", "monthly", "weekly", "all_time"}, optional
        Time aggregation frequency. If None, uses original data frequency.
        - "original": No aggregation (default)
        - "yearly": Aggregate to yearly totals
        - "monthly": Aggregate to monthly totals
        - "weekly": Aggregate to weekly totals
        - "all_time": Aggregate to single total (no time dimension)
    output_format : {"pandas", "polars"}, optional
        Output DataFrame format. Defaults to "pandas".

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Schema:
        - date : datetime64[ns]
        - mean : float64 (predicted mean)
        - median : float64 (predicted median)
        - observed : float64 (actual values)
        - abs_error_{prob}_lower : float64 (HDI bounds)
        - abs_error_{prob}_upper : float64

    Examples
    --------
    >>> # Get Pandas DataFrame (default)
    >>> df = create_posterior_predictive_summary(mmm.data, hdi_probs=[0.94])
    >>>
    >>> # Get Polars DataFrame
    >>> df = create_posterior_predictive_summary(
    ...     mmm.data, hdi_probs=[0.94], output_format="polars"
    ... )
    >>>
    >>> # Plot (Component 3 - works with both)
    >>> from pymc_marketing.mmm.plotly_utils import plot_posterior_predictive
    >>> fig = plot_posterior_predictive(df, hdi_prob=0.94)
    """
    if hdi_probs is None:
        hdi_probs = [0.94]

    # Apply time aggregation if requested
    if frequency is not None and frequency != "original":
        data = data.aggregate_time(frequency)

    # Access posterior predictive via data
    # TODO: We assume the variable name is "y". Need to ask devs about the
    # best way to find/configure the posterior predictive variable name.
    pp_data = data.idata.posterior_predictive["y"]  # xr.DataArray

    # Compute summary stats with HDI (always returns Pandas internally)
    df = _compute_summary_stats_with_hdi(pp_data, hdi_probs)

    # Add observed data via data (handles multiple sources and scaling)
    observed = data.get_target(original_scale=True)
    obs_df = observed.to_dataframe(name='observed').reset_index()

    df = df.merge(obs_df, on='date', how='left')

    return _convert_output(df, output_format)


def create_contribution_summary(
    data: MMMIDataWrapper,
    hdi_probs: list[float] | None = None,
    component: Literal["channel", "control", "seasonality", "baseline"] = "channel",
    frequency: Frequency = None,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Create contribution summary from data.

    Parameters
    ----------
    data : MMMIDataWrapper
        Data wrapper with filtered/aggregated data
    hdi_probs : list of float, optional
        HDI probability levels. Defaults to [0.94].
    component : {"channel", "control", "seasonality", "baseline"}
        Which contribution component to summarize
    frequency : {"original", "yearly", "monthly", "weekly", "all_time"}, optional
        Time aggregation frequency. If None, uses original data frequency.
    output_format : {"pandas", "polars"}, optional
        Output DataFrame format. Defaults to "pandas".

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Schema:
        - date : datetime64[ns] (optional - may be aggregated out)
        - channel : str (or control/other dimension)
        - mean : float64
        - median : float64
        - abs_error_{prob}_lower : float64
        - abs_error_{prob}_upper : float64

    Examples
    --------
    >>> # Filter to specific channels (Component 1)
    >>> data = mmm.data.filter_dims(channel=["TV", "Radio"])
    >>>
    >>> # Generate summary as Pandas (default)
    >>> df = create_contribution_summary(data, hdi_probs=[0.80, 0.94])
    >>>
    >>> # Generate summary as Polars
    >>> df_polars = create_contribution_summary(
    ...     data, hdi_probs=[0.80, 0.94], output_format="polars"
    ... )
    >>>
    >>> # Plot (Component 3)
    >>> from pymc_marketing.mmm.plotly_utils import plot_bar
    >>> fig = plot_bar(df, hdi_prob=0.80, yaxis_title="Contribution")
    """
    if hdi_probs is None:
        hdi_probs = [0.94]

    # Apply time aggregation if requested
    if frequency is not None and frequency != "original":
        data = data.aggregate_time(frequency)

    # Get contributions via data (already handles scaling, filtering)
    contributions = data.get_contributions(
        original_scale=True,
        include_baseline=(component == "baseline"),
        include_controls=(component == "control"),
        include_seasonality=(component == "seasonality"),
    )

    # Extract the requested component
    if component == "channel":
        component_data = contributions["channel"]
    elif component == "control":
        component_data = contributions.get("control")
        if component_data is None:
            raise ValueError("No control contributions in model")
    elif component == "seasonality":
        component_data = contributions.get("seasonality")
        if component_data is None:
            raise ValueError("No seasonality contributions in model")
    elif component == "baseline":
        component_data = contributions.get("baseline")
        if component_data is None:
            raise ValueError("No baseline contributions in model")

    # Compute summary stats with HDI
    df = _compute_summary_stats_with_hdi(component_data, hdi_probs)

    return _convert_output(df, output_format)


def create_roas_summary(
    data: MMMIDataWrapper,
    hdi_probs: list[float] | None = None,
    frequency: Frequency = None,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Create ROAS (contribution / spend) summary from data.

    Parameters
    ----------
    data : MMMIDataWrapper
        Data wrapper with filtered/aggregated data
    hdi_probs : list of float, optional
        HDI probability levels. Defaults to [0.94].
    frequency : {"original", "yearly", "monthly", "weekly", "all_time"}, optional
        Time aggregation frequency. If None, uses original data frequency.
    output_format : {"pandas", "polars"}, optional
        Output DataFrame format. Defaults to "pandas".

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Schema: date, channel, mean, median, abs_error_{prob}_lower, abs_error_{prob}_upper

    Examples
    --------
    >>> # Generate ROAS at different frequencies
    >>> df_roas = create_roas_summary(mmm.data, hdi_probs=[0.94])
    >>> df_roas_yearly = create_roas_summary(mmm.data, frequency="yearly")
    >>> df_roas_all_time = create_roas_summary(mmm.data, frequency="all_time")
    >>>
    >>> # Get Polars DataFrame
    >>> df_roas_polars = create_roas_summary(mmm.data, output_format="polars")
    >>>
    >>> # Plot (Component 3)
    >>> fig = plot_bar(df_roas_all_time, x="channel", hdi_prob=0.94, yaxis_title="ROAS")
    """
    if hdi_probs is None:
        hdi_probs = [0.94]

    # Apply time aggregation if requested
    if frequency is not None and frequency != "original":
        data = data.aggregate_time(frequency)

    # Get contributions in original scale (data handles scaling)
    contributions = data.get_contributions(original_scale=True)
    channel_contrib = contributions["channel"]

    # Get spend data via wrapper method
    channel_spend = data.get_channel_spend()

    # Compute ROAS (avoid division by zero)
    roas_data = xr.where(
        channel_spend != 0,
        channel_contrib / channel_spend,
        0.0
    )

    df = _compute_summary_stats_with_hdi(roas_data, hdi_probs)

    return _convert_output(df, output_format)


def create_channel_spend_dataframe(
    data: MMMIDataWrapper,
    output_format: OutputFormat = "pandas",
) -> DataFrameType:
    """Extract channel spend data from data.

    Raw data (no HDI) - no MCMC samples involved.

    Parameters
    ----------
    data : MMMIDataWrapper
        Data data
    output_format : {"pandas", "polars"}, optional
        Output DataFrame format. Defaults to "pandas".

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Schema: date, channel, channel_data
    """
    channel_spend = data.get_channel_spend()
    df = channel_spend.to_dataframe(name='channel_data').reset_index()

    return _convert_output(df, output_format)


# ==================== Convenience Wrapper ====================

class MMMSummaryFactory:
    """Factory for creating summary DataFrames from MMMIDataWrapper.

    Provides convenient access to summary functions with defaults.

    Key Features:
    - `output_format` parameter: Set once at factory level, applies to all methods
    - `frequency` parameter: Available on all summary methods for aggregation

    Parameters
    ----------
    data : MMMIDataWrapper
        Data wrapper (Component 1)
    hdi_probs : list of float, optional
        Default HDI probability levels. Defaults to [0.94].
    output_format : {"pandas", "polars"}, optional
        Default output DataFrame format. Defaults to "pandas".

    Examples
    --------
    >>> # Access via mmm.summary property (returns Pandas by default)
    >>> df_pp = mmm.summary.posterior_predictive()
    >>>
    >>> # Create factory with Polars output
    >>> factory = MMMSummaryFactory(mmm.data, output_format="polars")
    >>> df_polars = factory.contributions()
    >>> type(df_polars)
    <class 'polars.DataFrame'>
    >>>
    >>> # Override output_format per method call
    >>> df_pandas = factory.roas(output_format="pandas")  # Override factory default
    >>>
    >>> # View ROAS at different aggregation levels
    >>> df_roas = mmm.summary.roas()                          # Original frequency
    >>> df_roas_yearly = mmm.summary.roas(frequency='yearly') # Yearly totals
    """

    def __init__(
        self,
        data: MMMIDataWrapper,
        hdi_probs: list[float] | None = None,
        output_format: OutputFormat = "pandas",
    ):
        self.data = data
        self.hdi_probs = hdi_probs if hdi_probs is not None else [0.94]
        self.output_format = output_format

    def posterior_predictive(
        self,
        hdi_probs: list[float] | None = None,
        frequency: Frequency = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Get posterior predictive summary.

        Parameters
        ----------
        output_format : {"pandas", "polars"}, optional
            Override factory default output format.
        """
        return create_posterior_predictive_summary(
            self.data,
            hdi_probs=hdi_probs or self.hdi_probs,
            frequency=frequency,
            output_format=output_format or self.output_format,
        )

    def contributions(
        self,
        hdi_probs: list[float] | None = None,
        component: Literal["channel", "control", "seasonality", "baseline"] = "channel",
        frequency: Frequency = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Get contribution summary.

        Parameters
        ----------
        output_format : {"pandas", "polars"}, optional
            Override factory default output format.
        """
        return create_contribution_summary(
            self.data,
            hdi_probs=hdi_probs or self.hdi_probs,
            component=component,
            frequency=frequency,
            output_format=output_format or self.output_format,
        )

    def roas(
        self,
        hdi_probs: list[float] | None = None,
        frequency: Frequency = None,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Get ROAS summary.

        Parameters
        ----------
        output_format : {"pandas", "polars"}, optional
            Override factory default output format.
        """
        return create_roas_summary(
            self.data,
            hdi_probs=hdi_probs or self.hdi_probs,
            frequency=frequency,
            output_format=output_format or self.output_format,
        )

    def channel_spend(
        self,
        output_format: OutputFormat | None = None,
    ) -> DataFrameType:
        """Get channel spend data.

        Parameters
        ----------
        output_format : {"pandas", "polars"}, optional
            Override factory default output format.
        """
        return create_channel_spend_dataframe(
            self.data,
            output_format=output_format or self.output_format,
        )


# ==================== Internal Helper ====================

def _compute_summary_stats_with_hdi(
    data: xr.DataArray,
    hdi_probs: list[float],
) -> pd.DataFrame:
    """Convert xarray to DataFrame with summary stats and HDI.

    This is the core transformation function that:
    1. Computes mean and median across MCMC samples
    2. Computes HDI bounds for each probability level
    3. Returns structured DataFrame with absolute HDI bounds

    Note: Always returns Pandas internally. Conversion to other formats
    happens at the public API boundary via _convert_output().

    Parameters
    ----------
    data : xr.DataArray
        Must have 'chain' and 'draw' dimensions
    hdi_probs : list of float
        HDI probability levels (e.g., [0.80, 0.94])

    Returns
    -------
    pd.DataFrame
        With columns: <dimensions>, mean, median, abs_error_{prob}_lower, abs_error_{prob}_upper
    """
    # Compute point estimates
    mean_ = data.mean(dim=['chain', 'draw'])
    median_ = data.median(dim=['chain', 'draw'])

    df = xr.Dataset({
        'mean': mean_,
        'median': median_,
    }).to_dataframe().reset_index()

    # Compute HDI for each probability level
    for hdi_prob in hdi_probs:
        hdi = az.stats.hdi(data, hdi_prob=hdi_prob)
        hdi_df = hdi.to_dataframe().unstack()
        hdi_df.columns = ['abs_error_lower', 'abs_error_upper']

        prob_str = str(int(hdi_prob * 100))
        hdi_df = hdi_df.rename(columns={
            'abs_error_lower': f'abs_error_{prob_str}_lower',
            'abs_error_upper': f'abs_error_{prob_str}_upper',
        })

        # Merge on all index columns
        df = df.merge(hdi_df.reset_index(), on=list(hdi_df.index.names), how='left')

    return df
```

### MMM Integration

**File**: `pymc_marketing/mmm/mmm.py` and `pymc_marketing/mmm/multidimensional.py`

```python
class MMM(BaseValidateMMM):
    # ... existing methods ...

    @property
    def summary(self) -> MMMSummaryFactory:
        """Access summary data generation functions.

        Returns factory that consumes the data data (Component 1)
        and produces summary DataFrames (Component 2).

        Returns
        -------
        MMMSummaryFactory
            Factory with methods to create summary DataFrames.
            Default output_format is "pandas" (aligns with PyMC/ArviZ ecosystem).

        Examples
        --------
        >>> mmm.fit(X, y)
        >>> mmm.sample_posterior_predictive(X)
        >>>
        >>> # Get summary DataFrames (Pandas by default)
        >>> df_pp = mmm.summary.posterior_predictive()
        >>> df_contrib = mmm.summary.contributions()
        >>> df_roas = mmm.summary.roas()
        >>>
        >>> # Get Polars DataFrames instead
        >>> df_contrib_polars = mmm.summary.contributions(output_format="polars")
        >>>
        >>> # Or create a factory that returns Polars by default
        >>> from pymc_marketing.mmm.summary import MMMSummaryFactory
        >>> polars_factory = MMMSummaryFactory(mmm.data, output_format="polars")
        >>> df = polars_factory.contributions()  # Returns Polars
        >>>
        >>> # View data at different aggregation levels (common use case)
        >>> df_roas_yearly = mmm.summary.roas(frequency='yearly')
        >>> df_roas_all_time = mmm.summary.roas(frequency='all_time')
        >>>
        >>> # With filtering (Component 1) + output format
        >>> data = mmm.data.filter_dates("2025-01", "2025-06")
        >>> factory = MMMSummaryFactory(data, output_format="polars")
        >>> df = factory.contributions(frequency='monthly')  # Returns Polars
        >>>
        >>> # Use with Plotly plotting (Component 3 - works with both formats)
        >>> from pymc_marketing.mmm.plotly_utils import plot_bar
        >>> fig = plot_bar(df_roas_all_time, hdi_prob=0.94)
        >>> fig.show()
        """
        self._validate_model_was_built()
        self._validate_idata_exists()

        from pymc_marketing.mmm.summary import MMMSummaryFactory

        # Pass the data data (Component 1), not raw idata
        # Default to Pandas (aligns with PyMC/ArviZ ecosystem)
        return MMMSummaryFactory(data=self.data, output_format="pandas")
```

### Complete Usage Example

```python
# ============================================================================
# Step 1: Fit model (creates InferenceData)
# ============================================================================
mmm = MMM(
    date_column="date",
    channel_columns=["TV", "Radio", "Social"],
    target_column="sales",
)
mmm.fit(X, y)
mmm.sample_posterior_predictive(X)

# ============================================================================
# Step 2: Component 1 - Filter and Aggregate Data
# ============================================================================
data = mmm.data  # Returns MMMIDataWrapper
data = data.filter_dates("2025-01-01", "2025-06-30")  # Filter time
data = data.filter_dims(channel=["TV", "Radio"])      # Filter channels
data = data.aggregate_time("monthly", method="sum")   # Monthly rollup

# ============================================================================
# Step 3: Component 2 - Generate Summary DataFrames
# ============================================================================
from pymc_marketing.mmm.summary import MMMSummaryFactory

# Option A: Get Pandas DataFrames (default)
factory = MMMSummaryFactory(data, hdi_probs=[0.80, 0.94])
df_pp = factory.posterior_predictive()
df_contrib = factory.contributions()
df_roas = factory.roas()

# Option B: Get Polars DataFrames
polars_factory = MMMSummaryFactory(data, hdi_probs=[0.80, 0.94], output_format="polars")
df_pp_polars = polars_factory.posterior_predictive()
df_contrib_polars = polars_factory.contributions()

# Option C: Override output format per method call
df_roas_polars = factory.roas(output_format="polars")  # Override Pandas default

# Option D: Use factory functions directly
from pymc_marketing.mmm.summary import create_contribution_summary
df = create_contribution_summary(data, hdi_probs=[0.94])  # Pandas
df_polars = create_contribution_summary(data, hdi_probs=[0.94], output_format="polars")

# Option E: Use convenience property (uses unfiltered mmm.data, Pandas default)
df = mmm.summary.contributions()  # No filtering applied, returns Pandas
df_polars = mmm.summary.contributions(output_format="polars")  # Returns Polars

# ============================================================================
# Step 4: Component 3 - Plot with Plotly (works with both Pandas and Polars)
# ============================================================================
from pymc_marketing.mmm.plotly_utils import plot_bar, plot_posterior_predictive

# Plotting functions accept both Pandas and Polars DataFrames
fig1 = plot_posterior_predictive(df_pp, hdi_prob=0.94)  # Pandas
fig2 = plot_bar(df_contrib_polars, hdi_prob=0.80, yaxis_title="Contribution")  # Polars
fig3 = plot_bar(df_roas, x="channel", hdi_prob=0.94, yaxis_title="ROAS")

fig1.show()
fig2.show()
fig3.show()
```

### Narwhals for DataFrame-Agnostic Plotting Functions

Plotting functions (Component 3) should use Narwhals to accept both Pandas and Polars:

```python
# In pymc_marketing/mmm/plotly_utils.py
import narwhals as nw
from narwhals.typing import IntoDataFrame

def plot_bar(
    df: IntoDataFrame,  # Accepts both Pandas and Polars
    x: str = "channel",
    y: str = "mean",
    hdi_prob: float | None = None,
    yaxis_title: str | None = None,
) -> go.Figure:
    """Create bar chart from summary DataFrame.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Summary DataFrame (Pandas or Polars)

    Examples
    --------
    >>> # Works with Pandas
    >>> df_pandas = mmm.summary.contributions()
    >>> fig = plot_bar(df_pandas, hdi_prob=0.94)
    >>>
    >>> # Works with Polars too
    >>> df_polars = mmm.summary.contributions(output_format="polars")
    >>> fig = plot_bar(df_polars, hdi_prob=0.94)
    """
    # Wrap in Narwhals for unified API
    nw_df = nw.from_native(df)

    # Get column values using Narwhals API
    x_values = nw_df[x].to_list()
    y_values = nw_df[y].to_list()

    # ... rest of plotting logic ...
```

---

## Code References

### Sandbox Implementation
- [pymc_marketing/sandbox/xarray_processing_utils.py:48-89](pymc_marketing/sandbox/xarray_processing_utils.py#L48-L89) - Core summary function
- [pymc_marketing/sandbox/xarray_processing_utils.py:11-37](pymc_marketing/sandbox/xarray_processing_utils.py#L11-L37) - MMMPlotData dataclass
- [pymc_marketing/sandbox/xarray_processing_utils.py:138-220](pymc_marketing/sandbox/xarray_processing_utils.py#L138-L220) - Main processing pipeline
- [pymc_marketing/sandbox/plotly_utils.py:23-51](pymc_marketing/sandbox/plotly_utils.py#L23-L51) - HDI column validation
- [pymc_marketing/sandbox/plotly_utils.py:278-337](pymc_marketing/sandbox/plotly_utils.py#L278-L337) - Posterior predictive plot
- [pymc_marketing/sandbox/plotly_utils.py:409-464](pymc_marketing/sandbox/plotly_utils.py#L409-L464) - Bar chart plot

### Existing Patterns
- [pymc_marketing/mmm/utils.py:471-649](pymc_marketing/mmm/utils.py#L471-L649) - build_contributions() pattern
- [pymc_marketing/mmm/evaluation.py:101-141](pymc_marketing/mmm/evaluation.py#L101-L141) - summarize_metric_distributions() pattern
- [pymc_marketing/mmm/base.py:970-1066](pymc_marketing/mmm/base.py#L970-L1066) - compute_mean_contributions_over_time()

### Validation Patterns
- [pymc_marketing/mmm/scaling.py:21-79](pymc_marketing/mmm/scaling.py#L21-L79) - Pydantic BaseModel pattern
- [pymc_marketing/mmm/validating.py:47-159](pymc_marketing/mmm/validating.py#L47-L159) - Mixin validation pattern
- [pymc_marketing/mmm/causal.py:54-65](pymc_marketing/mmm/causal.py#L54-L65) - TypedDict pattern

### Integration Patterns
- [pymc_marketing/mmm/multidimensional.py:618-623](pymc_marketing/mmm/multidimensional.py#L618-L623) - Property-based access
- [pymc_marketing/mmm/budget_optimizer.py:288-297](pymc_marketing/mmm/budget_optimizer.py#L288-L297) - Protocol pattern
- [pymc_marketing/mmm/plot.py:208-220](pymc_marketing/mmm/plot.py#L208-L220) - MMMPlotSuite integration

### Pandas vs Polars
- `pyproject.toml:31` - Pandas is required dependency
- All production methods return `pd.DataFrame`
- Sandbox uses Polars (47 occurrences in 4 files)
- No Polars in dependencies or production code

---

## Open Questions

### 1. Polars Version Compatibility

**Question**: What minimum version of Polars should we support?

**Decision**: Use `polars>=0.20` (optional dependency).

**Rationale**:
- Polars 0.20+ has mature DataFrame API
- Polars remains optional via `[polars]` extra to keep base install small
- Component 2 only needs `pl.from_pandas()` which is stable

**Implementation**:
```toml
# pyproject.toml
[project.optional-dependencies]
polars = ["polars>=0.20"]
```

**Note**: Narwhals (`narwhals>=1.0`) is a dependency for **Component 3 (Plotting)**, not Component 2. It will be added when implementing Component 3.

---

### 2. Performance Benchmarking

**Question**: What is the performance impact of using Pandas vs Polars for summary generation?

**Decision**: Let users choose via `output_format` parameter.

**Rationale**:
- Summary DataFrames are typically small (aggregated statistics), so performance difference is minimal
- Users who work primarily with Polars can avoid Pandas→Polars conversion overhead
- Internal computation still uses Pandas (from xarray), so main benefit is for downstream workflows
- No forced conversion for users who want to stay in their preferred DataFrame ecosystem

---

### 2. Additional Summary Functions

**Question**: What other summary functions should Component 2 provide?

**Required** (to be implemented):
- `create_saturation_curves()` - Sample saturation curves for each channel
- `create_decay_curves()` - Sample adstock decay curves for each channel
- `create_total_contribution_summary()` - All effects combined (baseline, controls, seasonality, media)
- `create_period_over_period_summary()` - Percentage change comparisons between time periods

**Future candidates** (based on user feedback):
- `create_waterfall_data()` - Waterfall chart data
- `create_lift_test_comparison()` - Compare model predictions to lift tests

---

### 3. Time Aggregation Integration

**Question**: Should Component 2 handle time aggregation, or should that stay in Component 1?

**Decision**: Both. Time aggregation is very common, so Component 2 provides a `frequency` parameter.

**Rationale**: It is common in MMM analysis to want to see the data at multiple aggregation levels:
- **Original frequency**: Weekly/daily data as fitted
- **Yearly**: Annual totals for year-over-year comparisons
- **Total**: Overall totals across all time for summary metrics like total ROAS

This is so common that it warrants a convenience parameter rather than forcing users to always go through Component 1.

**Approach**:
- Component 1 (MMMIDataWrapper) provides full control via `.aggregate_time()` method
- Component 2 summary functions accept optional `frequency` parameter for convenience
- Component 2 delegates to Component 1 internally

**Example API**:
```python
>>> mmm.fit(X, y)
>>> mmm.sample_posterior_predictive(X)
>>>
>>> # Get summary DataFrames at different frequencies
>>> df_pp = mmm.summary.posterior_predictive()
>>> df_roas_yearly = mmm.summary.roas(frequency='yearly')
>>> df_roas_all_time = mmm.summary.roas(frequency='all_time')
```

---

### 4. Multi-Dimensional Data Handling

**Question**: How should Component 2 handle custom dimensions (geo, brand, etc.)?

**Current approach**: Preserve all dimensions in DataFrames (they become columns).

**Example**:
```python
# Input: idata with dims (date, country, channel)
df = create_contribution_summary(idata)

# Output DataFrame:
#     date | country | channel | mean | median | ...
# 0   2025 |   US    |   C1    | 1000 |   998  | ...
# 1   2025 |   US    |   C2    |  500 |   502  | ...
# 2   2025 |   UK    |   C1    |  800 |   798  | ...
```

**Question**: Should there be helpers for aggregating over dimensions?
```python
df_total = aggregate_dimension(df, dim='country', method='sum')
# Sums across countries, keeps date and channel
```

**Decision**: Keep Component 2 simple (preserve dimensions). Aggregation helpers can be added later if needed.

---

### 5. Caching Strategy

**Question**: Should Component 2 cache computed DataFrames?

**Current design**: No caching - recompute on each call.

**Pros of caching**:
- Faster repeated access
- Avoid redundant computation

**Cons of caching**:
- Increased memory usage
- Cache invalidation complexity
- Property-based access creates new instance each time (no state to cache)

**Recommendation**: Start without caching. Add if users report performance issues with repeated access.

---

### 6. Error Handling

**Question**: How should Component 2 handle missing variables in idata?

**Current approach**:
- Check if `_original_scale` variable exists
- If not, compute on-the-fly by multiplying by `target_scale`
- Raise clear error if neither exists

**Alternative**: Provide `strict=True/False` parameter
- `strict=True`: Raise error if exact variable not found
- `strict=False`: Try fallback computation

**Recommendation**: Use fallback computation (current approach) but add warnings when using fallback.

---

## Conclusion

Component 2 should be implemented as **factory functions consuming MMMIDataWrapper (Component 1)**, returning **either Pandas or Polars DataFrames** controlled by an `output_format` parameter.

**Important**: Component 2 does NOT use Narwhals. Narwhals is used by Component 3 (plotting functions) to accept either DataFrame type as input.

### Architecture Benefits

✅ **Proper separation of concerns** - Component 1 handles data access, Component 2 handles summarization, Component 3 handles plotting
✅ **No duplicate logic** - Scaling, filtering, aggregation centralized in Component 1
✅ **Component 2 is simpler** - Just xarray → DataFrame + HDI (50% less code)
✅ **Composable workflow** - Chain operations: `filter → aggregate → summarize → plot`
✅ **Follows existing patterns** - Consistent with property-based access (`mmm.plot`, `mmm.sensitivity`)
✅ **DataFrame flexibility** - Users choose Pandas or Polars based on their workflow

### Design Decisions

✅ **Takes MMMIDataWrapper, not idata** - Critical architectural decision
✅ **Supports both Pandas and Polars** - Via `output_format` parameter (simple `pl.from_pandas()` conversion)
✅ **Defaults to Pandas** - Aligns with PyMC/ArviZ ecosystem conventions
✅ **Functional approach** - Proven in sandbox, simple and testable
✅ **Property-based access** - Clean fluent API via `mmm.summary`
✅ **Wrapper encapsulates complexity** - All idata conventions hidden in Component 1

### Implementation Checklist

**Phase 1: Core Implementation (Component 2)**
- [ ] Add `polars` to `pyproject.toml` optional dependencies (`[polars]` extra)
- [ ] Create `pymc_marketing/mmm/summary.py` module
- [ ] Implement `_convert_output()` helper for Pandas/Polars conversion (uses `pl.from_pandas()`)
- [ ] Implement factory functions with `output_format` parameter:
  - [ ] `create_posterior_predictive_summary(data, ..., output_format="pandas")`
  - [ ] `create_contribution_summary(data, ..., output_format="pandas")`
  - [ ] `create_roas_summary(data, ..., output_format="pandas")`
  - [ ] `create_channel_spend_dataframe(data, output_format="pandas")`
  - [ ] `create_saturation_curves(data, ..., output_format="pandas")`
  - [ ] `create_decay_curves(data, ..., output_format="pandas")`
  - [ ] `create_total_contribution_summary(data, ..., output_format="pandas")`
  - [ ] `create_period_over_period_summary(data, ..., output_format="pandas")`
- [ ] Implement `MMMSummaryFactory` class with `output_format` parameter
- [ ] Implement `_compute_summary_stats_with_hdi()` helper

**Phase 2: Integration**
- [ ] Ensure Component 1 (`MMMIDataWrapper`) is implemented first
- [ ] Add `MMM.summary` property returning `MMMSummaryFactory(data=self.data, output_format="pandas")`

**Phase 3: Component 3 Integration (Plotting - uses Narwhals)**
- [ ] Add `narwhals` to `pyproject.toml` dependencies (for Component 3)
- [ ] Update Plotly utils to accept both Pandas and Polars via Narwhals
- [ ] Add Narwhals-based helper functions for DataFrame-agnostic operations

**Phase 4: Testing & Documentation**
- [ ] Write comprehensive unit tests (test both Pandas and Polars output)
- [ ] Write integration tests with Component 1 filtering/aggregation
- [ ] Test Polars-not-installed graceful error handling
- [ ] Document DataFrame schemas in docstrings
- [ ] Document `output_format` parameter usage
- [ ] Create user guide with examples showing both formats

**Next step**: Implement Component 1 (`MMMIDataWrapper`), then implement Component 2 consuming it.
