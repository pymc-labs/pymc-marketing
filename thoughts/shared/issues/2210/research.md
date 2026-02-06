---
date: 2026-02-06T18:55:44+00:00
researcher: Claude Sonnet 4.5
git_commit: 3a358967271a3ade6aeffaa21769feef9570c69c
branch: work-issue-2210
repository: pymc-marketing
topic: "Converting between impressions and spend in MMM models"
tags: [research, codebase, mmm, budget-optimization, saturation-curves, channel-metadata, cost-per-unit, time-varying]
status: complete
last_updated: 2026-02-06
last_updated_by: Claude Sonnet 4.5
issue_number: 2210
---

# Research: Converting between impressions and spend in MMM models (Updated)

**Date**: 2026-02-06T18:55:44+00:00
**Researcher**: Claude Sonnet 4.5
**Git Commit**: 3a358967271a3ade6aeffaa21769feef9570c69c
**Branch**: work-issue-2210
**Repository**: pymc-marketing
**Issue**: #2210

## Research Question

How should pymc-marketing support cost_per_unit conversion for channels with non-spend data (impressions, clicks, etc.)? Based on feedback from @isofer:

1. Use unit-agnostic naming (not "impression" - just cost_per_unit)
2. Leverage MMMIDataWrapper.get_channel_spend() as the conversion point
3. Support time-varying cost_per_unit from the start
4. Conversion only affects plots and optimization (NOT modeling)
5. Support both initialization-time and post-hoc specification

## Summary

The codebase has a strategic architecture point for implementing cost_per_unit conversion: **MMMIDataWrapper.get_channel_spend()**. This method is used by:
- ROAS calculations (contribution / spend)
- Summary reporting (channel spend tables)
- But NOT by budget optimization or saturation plotting (they access data directly)

Key findings:
1. **get_channel_spend() is the ideal conversion point** - centralized accessor in `pymc_marketing/data/idata/mmm_wrapper.py:165-189`
2. **Budget optimization doesn't use get_channel_spend()** - accesses channel_data through PyMC model graph
3. **Saturation plotting doesn't use get_channel_spend()** - accesses idata.constant_data.channel_data directly
4. **InferenceData schema supports time-varying metadata** - can store cost_per_unit with dims ("date", "channel")
5. **Need conversion at multiple touch points** - wrapper, optimizer, and plotting all need awareness

## Detailed Findings

### 1. MMMIDataWrapper.get_channel_spend() - The Strategic Conversion Point

#### Current Implementation
**File**: `pymc_marketing/data/idata/mmm_wrapper.py:165-189`

```python
def get_channel_spend(self) -> xr.DataArray:
    """Get channel spend data with consistent access pattern.

    Returns raw channel spend data (not MCMC samples).

    Returns
    -------
    xr.DataArray
        Channel spend values with dims (date, channel)

    Raises
    ------
    ValueError
        If channel_data not found in constant_data
    """
    if not (
        hasattr(self.idata, "constant_data")
        and "channel_data" in self.idata.constant_data
    ):
        raise ValueError(
            "Channel data not found in constant_data. "
            "Expected 'channel_data' variable in idata.constant_data."
        )

    return self.idata.constant_data.channel_data
```

**Current behavior**: Simple passthrough to raw channel data, no conversion applied.

#### Usage Analysis

**Direct Calls** (2 locations):

1. **ROAS Calculation** (`mmm_wrapper.py:342`):
   ```python
   def get_roas(self, original_scale: bool = True) -> xr.DataArray:
       contributions = self.get_channel_contributions(original_scale=original_scale)
       spend = self.get_channel_spend()  # <-- Used as denominator
       spend_safe = xr.where(spend == 0, np.nan, spend)
       return contributions / spend_safe
   ```
   **Impact**: If channel_data is in impressions, ROAS will be wrong (contribution per impression, not per dollar)

2. **Channel Spend Summary** (`summary.py:561`):
   ```python
   def channel_spend(self, output_format: OutputFormat | None = None) -> DataFrameType:
       spend = self.data.get_channel_spend()  # <-- Raw data for reporting
       df = spend.to_dataframe(name="channel_data").reset_index()
       return self._convert_output(df, effective_output_format)
   ```
   **Impact**: Reports will show raw units (impressions or spend) without conversion

**Indirect Calls** (via get_roas):
- `summary.roas()` - User-facing ROAS summary tables
- All ROAS tests in `tests/mmm/test_summary.py`

#### Proposed Enhancement

```python
def get_channel_spend(self, apply_cost_per_unit: bool = True) -> xr.DataArray:
    """Get channel spend data, optionally converting non-spend units.

    Parameters
    ----------
    apply_cost_per_unit : bool, default True
        If True and cost_per_unit metadata exists, converts non-spend
        channel data to spend units by multiplying by cost_per_unit.
        If False, returns raw channel_data without conversion.

    Returns
    -------
    xr.DataArray
        Channel spend values with dims (date, channel) or
        (date, *custom_dims, channel). Values are in spend units
        if apply_cost_per_unit=True and conversion metadata exists.
    """
    if not (
        hasattr(self.idata, "constant_data")
        and "channel_data" in self.idata.constant_data
    ):
        raise ValueError(
            "Channel data not found in constant_data. "
            "Expected 'channel_data' variable in idata.constant_data."
        )

    channel_data = self.idata.constant_data.channel_data

    # Apply cost_per_unit conversion if requested and available
    if (
        apply_cost_per_unit
        and hasattr(self.idata, "constant_data")
        and "cost_per_unit" in self.idata.constant_data
    ):
        cost_per_unit = self.idata.constant_data.cost_per_unit
        # Multiply: impressions * cost_per_impression = spend
        # Broadcasting handles both scalar and time-varying cases
        channel_data = channel_data * cost_per_unit

    return channel_data
```

**Key change**: Default to `apply_cost_per_unit=True`, making conversion automatic. Users can opt out with `apply_cost_per_unit=False` to get raw data.

### 2. Budget Optimization - Direct Channel Data Access

#### Current Implementation
**File**: `pymc_marketing/mmm/budget_optimizer.py:896-952`

**Critical finding**: BudgetOptimizer does NOT call `get_channel_spend()`. Instead:

```python
def _replace_channel_data_by_optimization_variable(self, model: Model) -> Model:
    """Replace `channel_data` in model graph with budgets."""
    # Line 900: Extract channel_data dims from PyMC model
    channel_data_dims = model.named_vars_to_dims["channel_data"]

    # Line 902: Access channel scales
    channel_scales = self.mmm_model._channel_scales

    # Line 905-906: Apply scaling
    budgets = self._budgets
    budgets /= channel_scales

    # Line 952: Replace in model graph
    modified_model = pm.do(model, {model.named_vars["channel_data"]: budgets})
    return modified_model
```

**Data flow**: User provides `total_budget` → Optimizer creates `_budgets` variables → Divided by `channel_scales` → Inserted into PyMC model graph

**Problem**: If channel_data is in impressions but budgets are in dollars, the scaling is wrong.

#### Proposed Enhancement

**New wrapper method** (`mmm_wrapper.py`, add around line 190):

```python
def get_cost_per_unit(self) -> xr.DataArray | None:
    """Get cost per unit conversion factors.

    Returns
    -------
    xr.DataArray or None
        Cost per unit values with dims ("channel",) for scalar
        or ("date", "channel") for time-varying. Returns None
        if cost_per_unit metadata not present.
    """
    if (
        hasattr(self.idata, "constant_data")
        and "cost_per_unit" in self.idata.constant_data
    ):
        return self.idata.constant_data.cost_per_unit
    return None
```

**Modified optimizer** (`budget_optimizer.py:905-906`):

```python
# Current:
budgets = self._budgets
budgets /= channel_scales

# Proposed:
budgets = self._budgets

# Convert budget from spend to original units (if cost_per_unit exists)
cost_per_unit = self._get_cost_per_unit()  # New helper method
if cost_per_unit is not None:
    # For scalar cost_per_unit: shape (channel,)
    # For time-varying: shape (date, channel) - need to average or use first period
    if cost_per_unit.ndim > 1:
        # Use mean cost_per_unit for optimization period
        cost_per_unit = cost_per_unit.mean(dim="date")
    # Divide budget by cost to get original units (e.g., $ / $-per-impression = impressions)
    budgets = budgets / cost_per_unit

budgets /= channel_scales
```

**New helper in BudgetOptimizer**:

```python
def _get_cost_per_unit(self) -> xr.DataArray | None:
    """Extract cost_per_unit from wrapped model's InferenceData."""
    if hasattr(self.mmm_model, "idata"):
        wrapper = MMMIDataWrapper(self.mmm_model.idata, validate_on_init=False)
        return wrapper.get_cost_per_unit()
    return None
```

### 3. Saturation Curve Plotting - Direct Channel Data Access

#### Current Implementation
**File**: `pymc_marketing/mmm/plot.py:2273-2333`

**Critical finding**: Plotting does NOT call `get_channel_spend()`. It accesses data directly:

```python
# Line 2273-2281: Convert curve x-coords to original scale
if original_scale:
    channel_scale = self.idata.constant_data.channel_scale.sel(**valid_idx)
    x_original = subplot_curve.coords["x"] * channel_scale
    subplot_curve = subplot_curve.assign_coords(x=x_original)

# Line 2315: Get scatter plot x-data directly
x_data = self.idata.constant_data.channel_data.sel(**indexers)

# Line 2316-2326: Get scatter plot y-data
y = (
    self.idata.posterior[contrib_var]
    .sel(**indexers)
    .mean(dim=[d for d in ... if d in ("chain", "draw")])
)
```

**Data flow**:
1. Curve x-coords: `np.linspace(0, max_value, 100)` → scaled by `channel_scale`
2. Scatter x-data: `idata.constant_data.channel_data` (raw)
3. Scatter y-data: `idata.posterior.channel_contribution` (mean across samples)

**Problem**: Scatter data and curve x-axis both need cost_per_unit conversion for spend-based plots.

#### Proposed Enhancement

Add new parameter and conversion logic:

```python
def saturation_curves(
    self,
    curve: xr.DataArray,
    channels: Sequence[str] | None = None,
    original_scale: bool = True,
    plot_as_spend: bool = True,  # NEW PARAMETER
    # ... other params
) -> tuple[plt.Figure, np.ndarray]:
    """Plot saturation curves.

    Parameters
    ----------
    # ... existing params
    plot_as_spend : bool, default True
        If True, convert x-axis to spend units using cost_per_unit.
        If False, plot in original channel data units.
        Only applies if cost_per_unit metadata exists.
    """
    # ... existing setup code

    # Line 2273-2281: Convert curve x-coords
    if original_scale:
        channel_scale = self.idata.constant_data.channel_scale.sel(**valid_idx)
        x_original = subplot_curve.coords["x"] * channel_scale
        subplot_curve = subplot_curve.assign_coords(x=x_original)

        # NEW: Apply cost_per_unit conversion
        if plot_as_spend and "cost_per_unit" in self.idata.constant_data:
            cost_per_unit = self.idata.constant_data.cost_per_unit.sel(**valid_idx)
            # For time-varying cost_per_unit, use mean or specific date
            if "date" in cost_per_unit.dims:
                cost_per_unit = cost_per_unit.mean(dim="date")
            x_spend = subplot_curve.coords["x"] * cost_per_unit
            subplot_curve = subplot_curve.assign_coords(x=x_spend)

    # ... plotting code

    # Line 2315-2333: Scatter plot
    x_data = self.idata.constant_data.channel_data.sel(**indexers)

    # NEW: Apply cost_per_unit conversion to scatter data
    if plot_as_spend and "cost_per_unit" in self.idata.constant_data:
        cost_per_unit_scatter = self.idata.constant_data.cost_per_unit.sel(**indexers)
        x_data = x_data * cost_per_unit_scatter

    y = self.idata.posterior[contrib_var].sel(**indexers).mean(dim=[...])

    ax.scatter(x_data, y, alpha=0.5, s=10, label="Historical data")

    # Update axis label
    x_label = "Spend" if plot_as_spend else "Channel Data"
    ax.set_xlabel(x_label)
```

**Other affected plotting functions**:
1. `saturation_scatterplot()` (`plot.py:2060`) - needs same conversion
2. `budget_allocation()` (`plot.py:2447-2450`) - already in spend units
3. Sensitivity analysis plots (`plot.py:3410-3611`) - needs conversion

### 4. InferenceData Schema for cost_per_unit Storage

#### Recommended Schema Addition
**File**: `pymc_marketing/data/idata/schema.py:293` (add after target_scale)

```python
constant_data_vars["cost_per_unit"] = VariableSchema(
    name="cost_per_unit",
    dims="*",  # Flexible: ("channel",) or ("date", "channel")
    dtype=("float64", "float32"),
    description="Cost per unit for converting non-spend data to spend units",
    required=False,  # Optional, not all models need this
)
```

**Why wildcard dims="*"**:
- Supports scalar per channel: `dims=("channel",)` → shape (n_channels,)
- Supports time-varying: `dims=("date", "channel")` → shape (n_dates, n_channels)
- Follows pattern of `channel_scale` which also uses `dims="*"` (line 281)
- Validation at line 84-89 accepts any dimensional structure for wildcards

#### Physical Variable Creation
**File**: `pymc_marketing/mmm/multidimensional.py:1351` (add near other pm.Data calls)

```python
# After target_scale creation
if self.cost_per_unit is not None:
    if isinstance(self.cost_per_unit, dict):
        # Convert dict to DataArray
        cost_per_unit_array = xr.DataArray(
            [self.cost_per_unit.get(ch, 1.0) for ch in self.channel_columns],
            dims="channel",
            coords={"channel": self.channel_columns},
        )
    elif isinstance(self.cost_per_unit, xr.DataArray):
        cost_per_unit_array = self.cost_per_unit
    elif isinstance(self.cost_per_unit, pd.DataFrame):
        # Time-varying from DataFrame
        cost_per_unit_array = self.cost_per_unit.to_xarray()
    else:
        raise TypeError(f"Unsupported cost_per_unit type: {type(self.cost_per_unit)}")

    _cost_per_unit = pm.Data(
        name="cost_per_unit",
        value=cost_per_unit_array.values,
        dims=cost_per_unit_array.dims,
    )
```

#### Model Initialization Parameter
**File**: `pymc_marketing/mmm/base.py:104-120` (add to MMMModelBuilder)

```python
cost_per_unit: dict[str, float] | xr.DataArray | pd.DataFrame | None = Field(
    default=None,
    description=(
        "Cost per unit conversion factors for non-spend channels. "
        "Provide as dict for scalar (e.g., {'channel1': 0.01}), "
        "xarray.DataArray with dims ('channel',) or ('date', 'channel'), "
        "or pandas.DataFrame with 'date' and channel columns for time-varying."
    ),
)
```

### 5. Time-Varying Support

#### Data Structure Options

**Option A: xarray.DataArray with dims ("date", "channel")**
```python
cost_per_unit = xr.DataArray(
    [[0.01, 0.02], [0.012, 0.021], [0.011, 0.019]],  # 3 dates, 2 channels
    dims=("date", "channel"),
    coords={
        "date": pd.date_range("2024-01-01", periods=3),
        "channel": ["channel1", "channel2"],
    },
)
```

**Option B: pandas.DataFrame**
```python
cost_per_unit = pd.DataFrame({
    "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "channel": ["channel1", "channel2", "channel1", "channel2"],
    "cost_per_unit": [0.01, 0.02, 0.012, 0.021],
})
```

**Recommendation**: Accept both formats in initialization, convert to xarray internally for storage in InferenceData.

#### Broadcasting Behavior

**Scalar case** (dims=("channel",)):
```python
channel_data.shape  # (n_dates, n_channels)
cost_per_unit.shape  # (n_channels,)
result = channel_data * cost_per_unit  # Broadcasts over dates
```

**Time-varying case** (dims=("date", "channel")):
```python
channel_data.shape  # (n_dates, n_channels)
cost_per_unit.shape  # (n_dates, n_channels)
result = channel_data * cost_per_unit  # Element-wise multiplication
```

xarray handles broadcasting automatically, no special logic needed.

#### Aggregation for Optimization

Budget optimization operates on a single time period, so time-varying cost_per_unit needs aggregation:

```python
if cost_per_unit.ndim > 1 and "date" in cost_per_unit.dims:
    # Option 1: Mean across time
    cost_per_unit_scalar = cost_per_unit.mean(dim="date")

    # Option 2: Use specific period
    cost_per_unit_scalar = cost_per_unit.sel(date=optimization_period).mean(dim="date")

    # Option 3: User-specified aggregation
    cost_per_unit_scalar = cost_per_unit.sel(date=slice(start, end)).mean(dim="date")
```

Recommend: Use mean across the optimization time window.

### 6. Post-Hoc Specification Pattern

#### Adding cost_per_unit to Existing Models

**New method in MMMIDataWrapper** (`mmm_wrapper.py`, add around line 125):

```python
def set_cost_per_unit(
    self,
    cost_per_unit: dict[str, float] | xr.DataArray | pd.DataFrame,
    overwrite: bool = False,
) -> None:
    """Add or update cost_per_unit metadata in InferenceData.

    Parameters
    ----------
    cost_per_unit : dict, xr.DataArray, or pd.DataFrame
        Cost per unit conversion factors. Can be:
        - dict mapping channel names to scalars: {'channel1': 0.01}
        - xr.DataArray with dims ('channel',) for scalar
        - xr.DataArray with dims ('date', 'channel') for time-varying
        - pd.DataFrame with columns 'date' and channel names
    overwrite : bool, default False
        If True, overwrite existing cost_per_unit. If False and
        cost_per_unit exists, raise ValueError.

    Raises
    ------
    ValueError
        If cost_per_unit exists and overwrite=False
    """
    if (
        "cost_per_unit" in self.idata.constant_data
        and not overwrite
    ):
        raise ValueError(
            "cost_per_unit already exists in InferenceData. "
            "Use overwrite=True to replace it."
        )

    # Convert to xarray
    if isinstance(cost_per_unit, dict):
        channels = list(self.idata.constant_data.coords["channel"].values)
        values = [cost_per_unit.get(ch, 1.0) for ch in channels]
        cost_per_unit_array = xr.DataArray(
            values,
            dims="channel",
            coords={"channel": channels},
        )
    elif isinstance(cost_per_unit, pd.DataFrame):
        cost_per_unit_array = cost_per_unit.set_index("date").to_xarray()
    elif isinstance(cost_per_unit, xr.DataArray):
        cost_per_unit_array = cost_per_unit
    else:
        raise TypeError(f"Unsupported type: {type(cost_per_unit)}")

    # Add to InferenceData
    self.idata.constant_data["cost_per_unit"] = cost_per_unit_array
```

**Usage example**:
```python
# Fit model with impression data
mmm = MMM(channel_columns=["ch1", "ch2"], ...)
mmm.fit(data_with_impressions)

# Add cost_per_unit later
mmm.data.set_cost_per_unit({"ch1": 0.01, "ch2": 0.02})

# Now get_channel_spend() returns converted values
spend = mmm.data.get_channel_spend()  # In dollars, not impressions
roas = mmm.data.get_roas()  # Correct ROAS ($/$ not $/impression)
```

## Code References

### Key Files to Modify

1. **MMMIDataWrapper** (`pymc_marketing/data/idata/mmm_wrapper.py`)
   - Line 165-189: `get_channel_spend()` - add conversion logic
   - Line 190+: Add `get_cost_per_unit()` method
   - Line 125+: Add `set_cost_per_unit()` method

2. **BudgetOptimizer** (`pymc_marketing/mmm/budget_optimizer.py`)
   - Line 896-952: `_replace_channel_data_by_optimization_variable()` - apply conversion
   - Add `_get_cost_per_unit()` helper method
   - Line 1010-1209: `allocate_budget()` - document unit assumptions

3. **Saturation Plotting** (`pymc_marketing/mmm/plot.py`)
   - Line 2099-2351: `saturation_curves()` - add plot_as_spend parameter
   - Line 2273-2281: Convert curve x-coords
   - Line 2315-2333: Convert scatter x-data
   - Line 1945-2097: `saturation_scatterplot()` - same changes

4. **InferenceData Schema** (`pymc_marketing/data/idata/schema.py`)
   - Line 293+: Add cost_per_unit variable schema

5. **MMM Base Model** (`pymc_marketing/mmm/base.py`)
   - Line 104-120: Add cost_per_unit initialization parameter

6. **Multidimensional MMM** (`pymc_marketing/mmm/multidimensional.py`)
   - Line 1351+: Create pm.Data for cost_per_unit
   - Initialization: Accept and validate cost_per_unit parameter

### Test Files to Update

- `tests/mmm/test_idata_wrapper.py` - Add tests for get_channel_spend conversion
- `tests/mmm/test_budget_optimizer.py` - Test optimization with cost_per_unit
- `tests/mmm/test_plot.py` - Test plotting with cost_per_unit
- `tests/mmm/test_summary.py` - Test ROAS with conversion
- `tests/data/test_idata_schema.py` - Test schema validation for cost_per_unit

## Architecture Insights

### 1. Strategic Conversion Point: get_channel_spend()

The method `get_channel_spend()` is perfectly positioned for applying cost_per_unit conversion:
- Used by ROAS calculation (most critical use case)
- Used by summary reporting (user-facing output)
- NOT used by model fitting (preserves modeling invariance)
- Centralized location (single point of change)

**Design principle**: "Convert on read, not on write"
- Store raw channel_data in original units (impressions, clicks, etc.)
- Store cost_per_unit metadata separately
- Apply conversion when reading via `get_channel_spend()`

### 2. Multi-Touch Conversion Architecture

Cost_per_unit conversion needs to happen at multiple points:

```
┌─────────────────────────────────────────────────────────┐
│                   InferenceData Storage                  │
│  constant_data:                                          │
│    - channel_data (raw units: impressions/spend/clicks)  │
│    - cost_per_unit (conversion factors)                  │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌────────────────┐  ┌──────────────┐  ┌──────────────┐
│ get_channel_   │  │ Budget       │  │ Saturation   │
│ spend()        │  │ Optimizer    │  │ Plotting     │
│                │  │              │  │              │
│ Apply          │  │ Apply        │  │ Apply        │
│ conversion     │  │ conversion   │  │ conversion   │
│ (automatic)    │  │ (explicit)   │  │ (opt-in)     │
└────────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
    ROAS calcs      Budget allocation   Saturation curves
```

**Key insight**: Not all code paths use `get_channel_spend()`, so conversion must be implemented redundantly at each touch point.

### 3. Time-Varying Flexibility

Supporting time-varying cost_per_unit from the start provides:
- **Realism**: Ad costs change over time (seasonality, market conditions)
- **Forward compatibility**: No breaking changes when users need time-varying later
- **Implementation simplicity**: xarray broadcasting handles both cases uniformly

**Storage pattern**:
- Scalar: dims=("channel",), shape=(n_channels,)
- Time-varying: dims=("date", "channel"), shape=(n_dates, n_channels)
- Validation: Use wildcard `dims="*"` in schema

### 4. Backward Compatibility Strategy

**For existing models** (no cost_per_unit):
- `get_channel_spend()` returns raw channel_data (unchanged behavior)
- ROAS calculations work as before (assumes data already in spend units)
- Budget optimization works as before (assumes data already in spend units)

**For new models** (with cost_per_unit):
- `get_channel_spend()` automatically converts to spend units (default behavior)
- Users can opt out with `get_channel_spend(apply_cost_per_unit=False)`
- Breaking change is acceptable per @isofer's comment: "It's ok that it's not backward compatible"

**Migration path**:
```python
# Old code (still works)
mmm = MMM(channel_columns=["ch1", "ch2"])
mmm.fit(data_with_spend)
spend = mmm.data.get_channel_spend()  # Returns spend

# New code (impression data)
mmm = MMM(
    channel_columns=["ch1", "ch2"],
    cost_per_unit={"ch1": 0.01, "ch2": 0.02},  # NEW
)
mmm.fit(data_with_impressions)
spend = mmm.data.get_channel_spend()  # Returns converted spend

# Post-hoc addition
mmm.data.set_cost_per_unit({"ch1": 0.01, "ch2": 0.02})
```

### 5. Unit Agnostic Naming

Following @isofer's guidance:
- **Not**: "impression_to_spend", "impression_cost", "cpi" (cost per impression)
- **Yes**: "cost_per_unit" - agnostic to actual data type (impressions, clicks, views, etc.)

**Rationale**:
- channel_data could be impressions, clicks, video views, email opens, etc.
- We don't need to know or care what the unit is
- cost_per_unit universally means "multiply this by channel_data to get spend"

**Variable naming**:
- `cost_per_unit` (not `cost_per_impression`)
- `apply_cost_per_unit` (not `convert_impressions`)
- `plot_as_spend` (not `plot_as_cost` or `convert_to_dollars`)

## Open Questions Resolved

### 1. ✅ Where should conversion happen?

**Answer**: Multiple points, as conversions don't propagate:
- `MMMIDataWrapper.get_channel_spend()` - automatic conversion for ROAS/summaries
- `BudgetOptimizer._replace_channel_data_by_optimization_variable()` - explicit conversion
- `MMMPlotSuite.saturation_curves()` - opt-in conversion via `plot_as_spend` parameter

### 2. ✅ Should we support time-varying?

**Answer**: Yes, from the start. Implementation complexity is minimal:
- xarray broadcasting handles both scalar and time-varying uniformly
- Schema wildcard `dims="*"` accepts both
- Storage cost is negligible (small metadata array)

### 3. ✅ How should users provide cost_per_unit?

**Answer**: Both initialization and post-hoc:
```python
# Option A: During initialization
mmm = MMM(..., cost_per_unit={"ch1": 0.01, "ch2": 0.02})

# Option B: After fitting (post-hoc)
mmm.fit(data)
mmm.data.set_cost_per_unit({"ch1": 0.01, "ch2": 0.02})
```

### 4. ✅ Should conversion affect modeling?

**Answer**: No. Per issue description and @isofer's comment:
> "The conversion should only affect Only affects plots and optimization. it should not affect modeling."

Store raw data in InferenceData, apply conversion during post-processing (plots, optimization, summaries).

### 5. ✅ How to handle mixed channel types?

**Answer**: Use cost_per_unit=1.0 as identity (no conversion):
```python
cost_per_unit = {
    "channel1": 0.01,    # Impressions → spend
    "channel2": 1.0,     # Already in spend units
    "channel3": 0.005,   # Clicks → spend
}
```

Default missing channels to 1.0 in implementation.

## Recommended Implementation Approach

### Phase 1: Core Infrastructure (Breaking Changes OK)

1. **Add cost_per_unit to InferenceData schema**
   - File: `pymc_marketing/data/idata/schema.py:293`
   - Add wildcard VariableSchema with `required=False`

2. **Add cost_per_unit parameter to model initialization**
   - File: `pymc_marketing/mmm/base.py:120`
   - Accept dict, xarray, or DataFrame
   - Validate and convert to xarray internally

3. **Store cost_per_unit in InferenceData**
   - File: `pymc_marketing/mmm/multidimensional.py:1351`
   - Create pm.Data with appropriate dims

4. **Modify get_channel_spend() to apply conversion**
   - File: `pymc_marketing/data/idata/mmm_wrapper.py:165-189`
   - Add `apply_cost_per_unit=True` parameter
   - Multiply by cost_per_unit if present
   - Default to True (automatic conversion)

5. **Add get_cost_per_unit() accessor**
   - File: `pymc_marketing/data/idata/mmm_wrapper.py:190`
   - Return cost_per_unit or None

6. **Add set_cost_per_unit() method**
   - File: `pymc_marketing/data/idata/mmm_wrapper.py:125`
   - Support post-hoc addition
   - Validate and convert input formats

### Phase 2: Budget Optimization

7. **Update BudgetOptimizer for cost_per_unit**
   - File: `pymc_marketing/mmm/budget_optimizer.py:896-952`
   - Extract cost_per_unit from InferenceData
   - Convert budgets from spend to original units before scaling
   - Handle time-varying by averaging

8. **Document budget unit assumptions**
   - File: `pymc_marketing/mmm/budget_optimizer.py:1010-1209`
   - Clarify that `total_budget` is in spend units
   - Document conversion behavior

### Phase 3: Plotting

9. **Update saturation_curves() plotting**
   - File: `pymc_marketing/mmm/plot.py:2099-2351`
   - Add `plot_as_spend=True` parameter
   - Convert curve x-coords (line 2273-2281)
   - Convert scatter x-data (line 2315-2333)
   - Update axis labels

10. **Update saturation_scatterplot()**
    - File: `pymc_marketing/mmm/plot.py:1945-2097`
    - Apply same conversion logic

11. **Update other relevant plots**
    - Sensitivity analysis plots (line 3410-3611)
    - Any other plots showing channel data vs contributions

### Phase 4: Testing and Documentation

12. **Add comprehensive tests**
    - Test scalar cost_per_unit
    - Test time-varying cost_per_unit
    - Test mixed channel types (some with conversion, some without)
    - Test backward compatibility (models without cost_per_unit)
    - Test post-hoc addition via set_cost_per_unit()

13. **Update documentation**
    - Add user guide section on cost_per_unit
    - Update examples with impression data
    - Document migration path for existing models
    - Add FAQ about units and conversion

14. **Add notebook example**
    - Show full workflow with impression data
    - Demonstrate time-varying cost_per_unit
    - Compare results with/without conversion

## Related Research

This supersedes the previous research conducted on 2026-01-30 in the same file, incorporating feedback from @isofer's comment on 2026-02-06.

## Implementation Checklist

- [ ] Add cost_per_unit to schema (schema.py:293)
- [ ] Add cost_per_unit parameter to MMMModelBuilder (base.py:120)
- [ ] Create pm.Data for cost_per_unit (multidimensional.py:1351)
- [ ] Modify get_channel_spend() (mmm_wrapper.py:165-189)
- [ ] Add get_cost_per_unit() (mmm_wrapper.py:190)
- [ ] Add set_cost_per_unit() (mmm_wrapper.py:125)
- [ ] Update BudgetOptimizer (budget_optimizer.py:896-952)
- [ ] Update saturation_curves() (plot.py:2273-2333)
- [ ] Update saturation_scatterplot() (plot.py:2060)
- [ ] Add tests for scalar cost_per_unit
- [ ] Add tests for time-varying cost_per_unit
- [ ] Add tests for mixed channel types
- [ ] Update documentation
- [ ] Add example notebook
