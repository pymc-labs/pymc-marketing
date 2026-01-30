---
date: 2026-01-30T20:00:00Z
researcher: Claude Sonnet 4.5
git_commit: 8c056f8553c7c092f1e89c18db37323fe765c825
branch: work-issue-2210
repository: pymc-marketing
topic: "Converting between impressions and spend in MMM models"
tags: [research, codebase, mmm, budget-optimization, saturation-curves, channel-metadata, impressions, spend]
status: complete
last_updated: 2026-01-30
last_updated_by: Claude Sonnet 4.5
issue_number: 2210
---

# Research: Converting between impressions and spend in MMM models

**Date**: 2026-01-30T20:00:00Z
**Researcher**: Claude Sonnet 4.5
**Git Commit**: 8c056f8553c7c092f1e89c18db37323fe765c825
**Branch**: work-issue-2210
**Repository**: pymc-marketing
**Issue**: #2210

## Research Question

How does pymc-marketing currently handle channel data (impressions vs spend), and what architecture changes are needed to support:
1. Mixed channel types (some channels in impressions, others in spend)
2. Conversion between impressions and spend for specific operations (saturation curves, budget optimization)
3. User-provided conversion factors (scalar or time-varying)

## Summary

pymc-marketing currently treats all channel data uniformly without distinguishing between impressions and spend. The codebase has no built-in mechanism for:
- Storing channel metadata (type, conversion factors)
- Converting impressions to spend for budget optimization or saturation curves
- Supporting mixed data types across channels

However, the architecture has several extension points where impression-to-spend conversion could be integrated:

1. **Channel metadata system**: No dedicated structure exists; channels are identified only by string names in `channel_columns`
2. **Data processing**: `process_fivetran_ad_reporting()` already handles both impressions and spend via `value_columns` parameter
3. **Calibration system**: `add_cost_per_target_potentials()` provides a pattern for adding conversion constraints
4. **InferenceData schema**: `channel_data` is documented as "Raw channel spend/impressions data" but no type distinction exists

## Detailed Findings

### 1. Current Channel Data Handling

#### Channel Identification and Storage
**File**: `pymc_marketing/mmm/base.py:104-120`

Channels are defined minimally:
```python
channel_columns: list[str] = Field(min_length=1, description="Column names of the media channel variables.")
self.channel_columns: list[str] | tuple[str] = channel_columns
self.n_channel: int = len(channel_columns)
```

**Key limitation**: Channel metadata is non-existent. Only channel names are stored.

#### InferenceData Storage Pattern
**File**: `pymc_marketing/data/idata/schema.py:265-269`

```python
"channel_data": VariableSchema(
    name="channel_data",
    dims=("date", *custom_dims, "channel"),
    dtype=("float64", "float32", "int64", "int32"),
    description="Raw channel spend/impressions data",  # <- Note: mentions both!
    required=True,
),
```

The schema already acknowledges both spend and impressions but provides no way to distinguish them.

#### Fivetran Data Processing
**File**: `pymc_marketing/data/fivetran.py`

The `process_fivetran_ad_reporting()` function supports both:
```python
value_columns: str | list[str] = "spend"  # Can be "impressions", "clicks", or ["spend", "impressions"]
```

This demonstrates the codebase already encounters mixed data types in practice.

### 2. Saturation Curve Implementation

#### Core Saturation Logic
**Files**:
- `pymc_marketing/mmm/components/saturation.py:104-496`
- `pymc_marketing/mmm/multidimensional.py:1888`

Saturation transformations operate on raw channel data without any awareness of units:
```python
def sample_saturation_curve(self, ...) -> xr.DataArray:
    # Samples saturation response curves from posterior
    # No conversion logic present
```

**Key insight**: Saturation curves are computed on whatever units the channel data uses. Issue #2210 requests they be computed based on **cost**, not impressions.

#### Saturation Curve Plotting
**File**: `pymc_marketing/mmm/plot.py:2099`

```python
def saturation_curves(self, ...) -> plt.Figure:
    # Plots saturation curves overlaid with HDI bands
    # Uses raw channel_data values without conversion
```

### 3. Budget Optimization Implementation

#### Budget Allocation Logic
**File**: `pymc_marketing/mmm/budget_optimizer.py:1-200`

The budget optimizer assumes all channel data is in cost units:
```python
def allocate_budget(
    self,
    total_budget: float,  # <- Assumes monetary units
    budget_bounds: dict[str, tuple[float, float]] | xr.DataArray | None = None,
    ...
) -> tuple[xr.DataArray, Any]:
```

**Critical finding**: Budget optimization currently cannot work with impression-based channels unless they're pre-converted to spend.

#### Constraint System
**File**: `pymc_marketing/mmm/constraints.py`

```python
def build_default_sum_constraint() -> Constraint:
    # Sum of allocated budgets must equal total_budget
    # No conversion logic
```

### 4. Cost-Per-Target Calibration (Existing Conversion Pattern)

#### add_cost_per_target_potentials Function
**File**: `pymc_marketing/mmm/lift_test.py:784-891`

This function provides a model for how conversion factors could work:

```python
def add_cost_per_target_potentials(
    calibration_df: pd.DataFrame,
    *,
    model: pm.Model | None = None,
    cpt_value: TensorVariable,
    target_column: str = "cost_per_target",
    name_prefix: str = "cpt_calibration",
    ...
) -> None:
    """Add ``pm.Potential`` penalties to calibrate cost-per-target.

    For each row, we compute the mean of ``cpt_variable_name`` across the date
    dimension for the specified (dims, channel) slice and add a soft quadratic
    penalty:

    ``penalty = - |cpt_mean - target|^2 / (2 * sigma^2)``.
    """
```

**Key pattern**: This shows how channel-specific conversion factors could be:
1. Provided via a DataFrame with channel column
2. Applied as soft constraints during model fitting
3. Support both scalar (averaged over time) and time-varying approaches

### 5. Scaling Infrastructure

#### Current Scaling System
**File**: `pymc_marketing/mmm/scaling.py`

```python
class VariableScaling(BaseModel):
    scaling_target: Literal["mean", "max"] = "mean"

class Scaling(BaseModel):
    target: VariableScaling = Field(default_factory=VariableScaling)
    channel: VariableScaling = Field(default_factory=VariableScaling)
```

**Observation**: Scaling is applied uniformly across all channels. This could be extended to handle per-channel conversion factors.

#### Channel Scaling in Practice
**File**: `pymc_marketing/mmm/preprocessing.py`

```python
class MaxAbsScaleChannels:
    # Scales each channel independently
    # Could be extended to apply conversion factors
```

## Code References

### Key Implementation Files

1. **MMM Base Classes**
   - `pymc_marketing/mmm/base.py:104-120` - Channel columns definition
   - `pymc_marketing/mmm/mmm.py` - Main MMM class
   - `pymc_marketing/mmm/multidimensional.py` - Multi-dimensional MMM

2. **Budget Optimization**
   - `pymc_marketing/mmm/budget_optimizer.py:1-200` - Core optimization logic
   - `pymc_marketing/mmm/constraints.py` - Constraint system
   - `pymc_marketing/mmm/utility.py` - Objective functions

3. **Saturation Curves**
   - `pymc_marketing/mmm/components/saturation.py:104-496` - Saturation transformations
   - `pymc_marketing/mmm/plot.py:2099` - Saturation curve plotting
   - `pymc_marketing/mmm/summary.py:566` - Saturation curve summaries

4. **Data Processing**
   - `pymc_marketing/data/fivetran.py` - Handles impressions/spend conversion
   - `pymc_marketing/mmm/preprocessing.py` - Channel scaling
   - `pymc_marketing/data/idata/schema.py:265-269` - InferenceData schema

5. **Calibration Pattern**
   - `pymc_marketing/mmm/lift_test.py:784-891` - Cost-per-target calibration

### Test Files for Reference

- `tests/mmm/test_budget_optimizer.py` - Budget optimization tests
- `tests/mmm/test_multidimensional.py` - Multi-dimensional calibration tests
- `tests/data/test_fivetran.py` - Impression/spend data processing tests
- `tests/mmm/test_lift_test.py` - Calibration tests

### Example Notebooks

- `docs/source/notebooks/mmm/mmm_fivetran_connectors.ipynb` - Shows `value_columns="spend"` or `"impressions"`
- `docs/source/notebooks/mmm/mmm_budget_allocation_example.ipynb` - Budget optimization examples
- `docs/source/notebooks/mmm/mmm_lift_test.ipynb` - Cost-per-target calibration examples

## Architecture Insights

### 1. No Channel Metadata Infrastructure

Currently, channels are "stringly typed" - identified only by their names in `channel_columns`. There's no structure for storing:
- Channel type (impression, spend, clicks, etc.)
- Conversion factors (cost per impression)
- Channel category or grouping
- Any other channel-level attributes

### 2. Uniform Data Treatment

All channel data flows through the same pipeline:
```
DataFrame → Validation → Scaling → PyMC Model Coords → Transformations → Optimization
```

No step distinguishes between data types or applies conversions.

### 3. Extension Points Identified

Based on the cost-per-target calibration pattern, conversion factors could be integrated at:

**Option A: During data preparation** (before model fitting)
- Pro: Simple, works with existing code
- Con: Loses flexibility for time-varying conversion, harder to propagate uncertainty

**Option B: During model building** (as calibration constraints)
- Pro: Flexible, can handle time-varying factors, propagates uncertainty
- Con: More complex, affects model structure

**Option C: During post-processing** (budget optimization, plotting)
- Pro: Minimal changes to core model
- Con: Conversion logic scattered across codebase

### 4. InferenceData Schema Flexibility

The `channel_data` variable in InferenceData already accepts dims like `("date", *custom_dims, "channel")`. Additional metadata could be stored as:
- `constant_data.channel_type` - dims: `("channel",)`, values: "impression"/"spend"
- `constant_data.cost_per_impression` - dims: `("channel",)` or `("date", "channel")` for time-varying

## Historical Context (from thoughts/)

No existing research documents were found in `thoughts/shared/research/` or `thoughts/searchable/` related to impression-to-spend conversion.

## Related Research

This is the first research document for issue #2210.

## Open Questions

1. **Conversion factor specification**: How should users provide conversion factors?
   - Via initialization parameter: `MMM(..., cost_per_impression={'channel1': 0.01, 'channel2': 0.02})`
   - Via separate DataFrame: `mmm.add_conversion_factors(df)`
   - Via calibration constraints: `mmm.add_cost_per_impression_calibration(df)`

2. **Time-varying vs scalar**: Should we support both?
   - Scalar: One conversion factor per channel (simpler)
   - Time-varying: Array with dims `("date", "channel")` (more flexible)

3. **Modeling vs post-processing**: Where should conversion happen?
   - During modeling: Affects saturation/adstock calculations
   - After modeling: Only affects plots and optimization
   - Issue description suggests: "This would not affect modeling, but plots and budget optimization"

4. **Backward compatibility**: How to handle existing models?
   - Default: All channels assumed to be spend (current behavior)
   - Require explicit opt-in for conversion features

5. **Uncertainty propagation**: Should conversion factors have uncertainty?
   - If scalar: Probably not needed
   - If time-varying or estimated: Should propagate to budget optimization

## Recommended Implementation Approach

Based on the research, here's a proposed architecture:

### Phase 1: Channel Metadata Infrastructure

1. **Add channel metadata storage** in `MMMModelBuilder`:
```python
class ChannelMetadata(BaseModel):
    channel: str
    data_type: Literal["spend", "impression", "clicks"]
    cost_per_unit: float | None = None  # For impression → spend conversion

channel_metadata: dict[str, ChannelMetadata] | None = None
```

2. **Store in InferenceData** `constant_data` group:
```python
# New variables in InferenceData
constant_data.channel_data_type: dims=("channel",), dtype=object
constant_data.cost_per_unit: dims=("channel",), dtype=float64
```

### Phase 2: Conversion in Budget Optimization

3. **Update `BudgetOptimizer`** to apply conversions:
```python
def allocate_budget(self, ...):
    # Convert impression channels to spend before optimization
    spend_data = self._convert_to_spend(channel_data, channel_metadata)
    # Run optimization on spend_data
    # Convert back to original units if needed
```

### Phase 3: Conversion in Plotting

4. **Update `saturation_curves()`** plotting:
```python
def saturation_curves(self, ...):
    # Convert x-axis to spend units if channel is impression-based
    x_spend = self._convert_to_spend(x_impression, cost_per_unit)
    # Plot saturation curve in spend space
```

### Phase 4: Time-Varying Support

5. **Extend to time-varying conversion factors**:
```python
cost_per_unit: xr.DataArray | dict[str, float]  # Support both scalar and time-varying
```

This approach:
- Preserves backward compatibility (metadata optional)
- Follows the calibration pattern from `lift_test.py`
- Doesn't affect core modeling (saturation/adstock applied to original data)
- Enables conversion only where needed (optimization, plotting)
