---
date: 2026-01-14 11:21:01 UTC
researcher: Claude Sonnet 4.5
git_commit: c0f88f7e707cfddf21da7763f4da2786c1eebf19
branch: work-issue-2188
repository: pymc-marketing
topic: "Add a method for registering data needed for predictions by a MuEffect"
tags: [research, codebase, mmm, mueffect, data-registration, extensibility, optional-columns, event-effect]
status: updated
last_updated: 2026-01-14 13:02:06 UTC
last_updated_by: Claude Sonnet 4.5
issue_number: 2188
---

# Research: Add a method for registering data needed for predictions by a MuEffect

**Date**: 2026-01-14 11:21:01 UTC
**Researcher**: Claude Sonnet 4.5
**Git Commit**: c0f88f7e707cfddf21da7763f4da2786c1eebf19
**Branch**: work-issue-2188
**Repository**: pymc-marketing
**Issue**: #2188

## Research Question

How can we add a method for registering additional data columns that need to be included in the xr.Dataset passed to `MuEffect.set_data()` during posterior predictive sampling in the Multidimensional MMM?

## Summary

The MuEffect protocol currently requires effects to handle their own data via three methods: `create_data()`, `create_effect()`, and `set_data()`. The issue arises when a MuEffect needs additional columns from the input DataFrame `X` that aren't already captured by the standard `channel_columns`, `control_columns`, or `target_column`.

Current architecture supports extension via the `mu_effects` list using the MuEffect protocol, but there's no built-in mechanism for effects to declare which additional columns from `X` they need included in the xr.Dataset passed to `set_data()`. The issue suggests creating a `data_vars` dictionary pattern similar to how channel, control, and target columns are currently registered.

**Key Finding**: The current implementation already has the infrastructure needed through `_create_xarray_from_pandas()` and the dataarrays merge pattern in `_posterior_predictive_data_transformation()`. The suggested solution of a `data_vars` registry is architecturally sound and aligns with existing patterns.

**Important Extension** (from community discussion): Adding an `optional` flag to the data registration would enable EventEffect to handle out-of-sample predictions with future events dynamically, rather than requiring all future events to be specified at model creation time. This would avoid the current workaround of adding zero-filled columns for non-existing future events.

## Detailed Findings

### 1. MuEffect Protocol Architecture

**Location**: `pymc_marketing/mmm/additive_effect.py:134-145`

The MuEffect protocol defines three required methods:

```python
class MuEffect(Protocol):
    def create_data(self, mmm: Model) -> None:
        """Create the required data in the model."""

    def create_effect(self, mmm: Model) -> pt.TensorVariable:
        """Create the additive effect in the model."""

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
        """Set the data for new predictions."""
```

**Current Built-in Implementations**:
1. `FourierEffect` (line 147): Seasonal/cyclic patterns - derives data from dates only
2. `LinearTrendEffect` (line 259): Time trends - derives data from dates only
3. `EventAdditiveEffect` (line 442): Event-based effects - uses dates and event DataFrame

**Observation**: All existing built-in effects derive their data from **dates** or **separate event DataFrames**, not from additional columns in the main input DataFrame `X`. This is why the gap exists.

### 2. Current Data Registration Pattern in MMM

**Location**: `pymc_marketing/mmm/multidimensional.py:340-393`

Columns are currently stored as simple instance attributes:

```python
self.control_columns = control_columns          # Line 340
self.date_column = date_column                  # Line 343, 390
self.target_column = target_column              # Line 391
self.channel_columns = channel_columns          # Line 392
```

These are used in two key transformation methods:

#### During Model Building
**Method**: `_generate_and_preprocess_model_data()` (lines 935-990)
**Purpose**: Creates training dataset `self.xarray_dataset`

```python
# Channel data
X_dataarray = self._create_xarray_from_pandas(
    data=X,
    date_column=self.date_column,
    dims=self.dims,
    metric_list=self.channel_columns,
    metric_coordinate_name="channel",
)
dataarrays.append(X_dataarray)

# Control data (if present)
if self.control_columns is not None:
    control_dataarray = self._create_xarray_from_pandas(
        data=X,
        date_column=self.date_column,
        dims=self.dims,
        metric_list=self.control_columns,
        metric_coordinate_name="control",
    )
    dataarrays.append(control_dataarray)

# Target data
y_dataarray = self._create_xarray_from_pandas(
    data=temp_y_df.set_index([self.date_column, *self.dims])[self.target_column],
    date_column=self.date_column,
    dims=self.dims,
    metric_list=[self.target_column],
    metric_coordinate_name="target",
).sum("target")
dataarrays.append(y_dataarray)

self.xarray_dataset = xr.merge(dataarrays).fillna(0)
```

#### During Posterior Predictive Sampling
**Method**: `_posterior_predictive_data_transformation()` (lines 1497-1585)
**Purpose**: Creates prediction dataset passed to `mu_effect.set_data()`

The same pattern repeats (lines 1530-1581):
- Channel data: lines 1530-1537
- Control data: lines 1539-1547
- Target data: lines 1549-1581

**Critical Line**: `multidimensional.py:1704-1705`
```python
for mu_effect in self.mu_effects:
    mu_effect.set_data(self, model, dataset_xarray)
```

The `dataset_xarray` contains only `_channel`, `_control`, and `_target` data variables.

### 3. How _create_xarray_from_pandas Works

**Location**: `pymc_marketing/mmm/multidimensional.py:870-933`

This is the core utility that transforms pandas data to xarray format:

```python
def _create_xarray_from_pandas(
    self,
    data: pd.DataFrame | pd.Series,
    date_column: str,
    dims: tuple[str, ...],
    metric_list: list[str],
    metric_coordinate_name: str,
) -> xr.Dataset:
```

**Process**:
1. Validates that metrics exist in the data (lines 748-768)
2. For DataFrames, uses `pd.melt()` to convert wide → long format (line 825)
3. Creates MultiIndex with `[date, *dims, metric_coordinate_name]` (line 858)
4. Converts to xarray, creating:
   - Coordinate: `metric_coordinate_name` with values from `metric_list`
   - Data variable: `_<metric_coordinate_name>` with dims `(date, *dims, metric_coordinate_name)`

**Example Output Structure**:
```python
# Input
metric_list = ["tv", "radio", "digital"]
metric_coordinate_name = "channel"

# Output Dataset
<xr.Dataset>
Coordinates:
  * channel  (channel) object 'tv' 'radio' 'digital'
Data variables:
    _channel (date, geo, channel) float64 ...
```

### 4. The Issue: Missing Extension Point

**Problem Statement**: A MuEffect that needs additional columns from `X` (e.g., campaign dates, holiday indicators, weather data) has no way to request that these columns be included in the xr.Dataset passed to `set_data()`.

**Current Workaround** (from issue description):
```python
# In multidimensional.MMM._posterior_predictive_data_transformation
if hasattr(self, "additional_vars"):
    for model_coord, (col_names, var_dims) in self.additional_vars.items():
        var_dataarray = self._create_xarray_from_pandas(
            data=X,
            date_column=self.date_column,
            dims=var_dims,
            metric_list=col_names,
            metric_coordinate_name=model_coord,
        ).transpose("date", *var_dims, model_coord)
        dataarrays.append(var_dataarray)

# In MuEffect.create_data()
if not hasattr(mmm, 'additional_vars'):
    mmm.additional_vars = {}
mmm.additional_vars[effect_coord_name] = effect_required_columns
```

**Issues with Current Workaround**:
- Uses `hasattr()` and manual attribute creation (not type-safe)
- Bypasses validation and initialization patterns
- Not documented or part of public API
- Requires effects to directly mutate MMM state

### 5. Suggested Solution Analysis

The issue proposes storing column specifications in a unified dictionary:

```python
mmm.data_vars = {
    'target': target_column,
    'channel': channel_columns,
    'control': control_columns,
}
```

Then iterating in `_posterior_predictive_data_transformation`:

```python
for var_coord, col_names in self.data_vars.items():
    var_dataarray = self._create_xarray_from_pandas(
        data=X,
        date_column=self.date_column,
        dims=self.dims,
        metric_list=col_names,
        metric_coordinate_name=var_coord,
    ).transpose("date", *self.dims, var_coord)
    dataarrays.append(var_dataarray)
```

**Advantages**:
1. **Unified Pattern**: All data sources follow same registration pattern
2. **Extensible**: MuEffects can add entries to `data_vars` in `create_data()`
3. **Type-Safe**: Can use proper initialization and validation
4. **Consistent**: Aligns with existing `_create_xarray_from_pandas` usage
5. **Clear API**: Explicit registration method makes intent clear

**Considerations**:
1. **dims Parameter**: The suggestion shows `dims=self.dims`, but MuEffects might need different dims (see note below)
2. **Initialization**: Need to populate initial `data_vars` from existing attributes
3. **Backward Compatibility**: Must maintain existing `channel_columns`, `control_columns`, `target_column` attributes
4. **Validation**: Need to validate that registered columns exist in input data

**Important Note on dims**: The issue's suggested solution uses `self.dims` universally, but the workaround shows `var_dims` as a parameter. Looking at the actual code:
- Channel data: `dims=self.dims` (line 948)
- Control data: `dims=self.dims` (line 971)
- Target data: `dims=self.dims` (line 962)

All current data uses `self.dims`, so the suggestion is consistent. However, for flexibility, the registration could include dims specification.

### 6. Existing Extension Patterns

The codebase shows several established patterns for extending MMM:

#### Pattern 1: The mu_effects List
**Location**: `multidimensional.py:443`
```python
self.mu_effects: list[MuEffect] = []
```

**Usage**: `multidimensional.py:1275-1276` (during build_model)
```python
for mu_effect in self.mu_effects:
    mu_effect.create_data(self)
```

**Usage**: `multidimensional.py:1704-1705` (during sample_posterior_predictive)
```python
for mu_effect in self.mu_effects:
    mu_effect.set_data(self, model, dataset_xarray)
```

#### Pattern 2: Event Registration via add_events()
**Location**: `multidimensional.py:465-503`

```python
def add_events(self, df_events: pd.DataFrame, prefix: str, effect: EventEffect) -> None:
    # Validate effect dims match model dims
    if not set(effect.dims).issubset((prefix, *self.dims)):
        raise ValueError(...)

    # Create and append event effect
    event_effect = EventAdditiveEffect(
        df_events=df_events,
        prefix=prefix,
        effect=effect,
    )
    self.mu_effects.append(event_effect)
```

**Observation**: This provides a higher-level API for a specific effect type. A similar pattern could be used for data registration.

#### Pattern 3: Coordinate Registration
**Location**: `multidimensional.py:979-982`

```python
self.model_coords = {
    dim: self.xarray_dataset.coords[dim].values
    for dim in self.xarray_dataset.coords.dims
}
```

Coordinates from xarray dataset become PyMC model coordinates (line 1226):
```python
with pm.Model(coords=self.model_coords) as self.model:
```

### 7. Example Custom MuEffect Needing Additional Data

**Location**: `docs/source/notebooks/mmm/mmm_gam_options.ipynb:7644-7722`

The `TransformedControlsEffect` example shows the need:

```python
class TransformedControlsEffect:
    def __init__(self, name: str, control_columns: Sequence[str], transformer, dim_suffix: str = "control_tf"):
        self.name = name
        self.control_columns = list(control_columns)
        self.transformer = deepcopy(transformer)
        self.dim_name = f"{name}_{dim_suffix}"
        self.data_name = f"{name}_data"

    def set_data(self, mmm, model: pm.Model, X: xr.Dataset) -> None:
        # Must convert xr.Dataset to DataFrame to extract custom columns
        df = (
            X.to_dataframe()
            .reset_index()
            .loc[:, [mmm.date_column, *mmm.dims, *self.control_columns]]
        )
        da = self._build_dataset(mmm, df).reindex(...)
        pm.set_data({self.data_name: da.values}, model=model)
```

**Problem**: The effect needs `self.control_columns` from `X`, but they're not in the xr.Dataset. It has to convert back to DataFrame and re-extract them. If those columns were registered in `data_vars`, they'd be available in the xr.Dataset directly.

### 8. Proposed Implementation Approach

Based on the research, here's a refined approach:

#### Step 1: Add data_vars Dictionary
**Location**: `multidimensional.py:__init__` (around line 443)

```python
# Initialize data_vars registry
self.data_vars: dict[str, list[str]] = {
    "channel": self.channel_columns,
    "target": [self.target_column],
}
if self.control_columns is not None:
    self.data_vars["control"] = self.control_columns
```

#### Step 2: Add Registration Method
```python
def register_data_columns(
    self,
    coordinate_name: str,
    column_names: list[str],
    dims: tuple[str, ...] | None = None,
) -> None:
    """Register additional data columns to include in predictions.

    Parameters
    ----------
    coordinate_name : str
        Name for the coordinate dimension in the xarray Dataset
    column_names : list[str]
        List of column names from input DataFrame to include
    dims : tuple[str, ...], optional
        Dimensions for this data. If None, uses self.dims

    Examples
    --------
    # In a MuEffect's create_data method:
    mmm.register_data_columns(
        coordinate_name="campaign_feature",
        column_names=["campaign_intensity", "campaign_reach"],
    )
    """
    if coordinate_name in self.data_vars:
        raise ValueError(f"Coordinate '{coordinate_name}' already registered")

    self.data_vars[coordinate_name] = column_names
    # If dims support is needed, store in separate dict:
    # self.data_vars_dims[coordinate_name] = dims or self.dims
```

#### Step 3: Update _posterior_predictive_data_transformation
**Location**: Lines 1530-1581 (replace hardcoded sections)

```python
def _posterior_predictive_data_transformation(
    self,
    X: pd.DataFrame,
    y: pd.Series | None = None,
    include_last_observations: bool = False,
) -> xr.Dataset:
    dataarrays = []

    if include_last_observations:
        last_obs = self.xarray_dataset.isel(date=slice(-self.adstock.l_max, None))
        dataarrays.append(last_obs)

    # Iterate through registered data sources
    for coord_name, col_names in self.data_vars.items():
        if coord_name == "target":
            # Special handling for target (can be None, needs sum)
            if y is not None:
                y_xarray = (
                    self._create_xarray_from_pandas(
                        data=y,
                        date_column=self.date_column,
                        dims=self.dims,
                        metric_list=col_names,
                        metric_coordinate_name=coord_name,
                    )
                    .sum(coord_name)
                    .transpose("date", *self.dims)
                )
            else:
                # Create zeros
                target_dtype = self.xarray_dataset._target.dtype
                y_xarray = xr.DataArray(
                    np.zeros(..., dtype=target_dtype),
                    dims=("date", *self.dims),
                    coords={...},
                    name="_target",
                ).to_dataset()
            dataarrays.append(y_xarray)
        else:
            # Standard handling for all other data sources
            data_xarray = self._create_xarray_from_pandas(
                data=X,
                date_column=self.date_column,
                dims=self.dims,  # or self.data_vars_dims[coord_name] if supporting custom dims
                metric_list=col_names,
                metric_coordinate_name=coord_name,
            ).transpose("date", *self.dims, coord_name)
            dataarrays.append(data_xarray)

    self.dataarrays = dataarrays
    self._new_internal_xarray = xr.merge(dataarrays).fillna(0)
    return xr.merge(dataarrays).fillna(0)
```

#### Step 4: Update _generate_and_preprocess_model_data Similarly
**Location**: Lines 943-977

Apply the same loop pattern to ensure training and prediction use the same data structure.

#### Step 5: Usage in MuEffect
```python
class CustomCampaignEffect:
    def __init__(self, campaign_columns: list[str]):
        self.campaign_columns = campaign_columns

    def create_data(self, mmm):
        # Register columns needed from X
        mmm.register_data_columns(
            coordinate_name="campaign",
            column_names=self.campaign_columns,
        )

        # Then use them to create PyMC Data
        model = mmm.model
        dates = pd.to_datetime(model.coords["date"])
        # ... create pm.Data variables

    def create_effect(self, mmm):
        # Create effect using registered data
        ...

    def set_data(self, mmm, model, X):
        # X now contains _campaign data variable!
        campaign_data = X["_campaign"]
        # Use it directly without DataFrame conversion
        pm.set_data({...}, model=model)
```

### 9. Alternative: Protocol Extension

Another approach would be to extend the MuEffect protocol:

```python
class MuEffect(Protocol):
    def create_data(self, mmm: Model) -> None:
        """Create the required data in the model."""

    def create_effect(self, mmm: Model) -> pt.TensorVariable:
        """Create the additive effect in the model."""

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
        """Set the data for new predictions."""

    def required_columns(self) -> dict[str, list[str]]:
        """Return required columns from input DataFrame.

        Returns
        -------
        dict[str, list[str]]
            Mapping of coordinate_name to list of column names

        Examples
        --------
        {"campaign_vars": ["campaign_start", "campaign_end"]}
        """
        return {}
```

Then in `build_model()` or during effect registration:
```python
for mu_effect in self.mu_effects:
    if hasattr(mu_effect, "required_columns"):
        for coord_name, col_names in mu_effect.required_columns().items():
            self.register_data_columns(coord_name, col_names)
```

**Advantages**:
- More declarative - effects specify their needs
- Automatic registration
- Type-checkable via Protocol

**Disadvantages**:
- Requires all effects to implement required_columns (breaks existing effects unless made optional)
- Less flexible for effects that compute column needs dynamically

## Code References

### Core Files
- `pymc_marketing/mmm/multidimensional.py:340-393` - Column attribute storage
- `pymc_marketing/mmm/multidimensional.py:443` - mu_effects list initialization
- `pymc_marketing/mmm/multidimensional.py:870-933` - _create_xarray_from_pandas method
- `pymc_marketing/mmm/multidimensional.py:935-990` - _generate_and_preprocess_model_data method
- `pymc_marketing/mmm/multidimensional.py:1497-1585` - _posterior_predictive_data_transformation method
- `pymc_marketing/mmm/multidimensional.py:1704-1705` - mu_effect.set_data() calls
- `pymc_marketing/mmm/additive_effect.py:134-145` - MuEffect Protocol definition

### Example Implementations
- `pymc_marketing/mmm/additive_effect.py:147` - FourierEffect class
- `pymc_marketing/mmm/additive_effect.py:259` - LinearTrendEffect class
- `pymc_marketing/mmm/additive_effect.py:442` - EventAdditiveEffect class
- `pymc_marketing/mmm/additive_effect.py:31-83` - PenaltyEffect example (docstring)
- `docs/source/notebooks/mmm/mmm_gam_options.ipynb:7644-7722` - TransformedControlsEffect example

### Tests
- `tests/mmm/test_additive_effect.py:94-146` - FourierEffect tests
- `tests/mmm/test_multidimensional.py:2453` - LinearTrendEffect usage
- `tests/mmm/test_multidimensional.py:1062-1077` - EventAdditiveEffect usage

## Architecture Insights

1. **xarray as Data Interchange Format**: The architecture uses xarray.Dataset as the standard format for passing data between MMM and MuEffects, providing explicit coordinate systems and type safety.

2. **Factory Pattern**: `_create_xarray_from_pandas()` acts as a factory that normalizes different pandas structures (DataFrame, MultiIndex Series) into consistent xarray format.

3. **Builder Pattern**: Both transformation methods build xarray Datasets incrementally via the `dataarrays` list before merging, making it easy to add new data sources.

4. **Protocol-Based Extension**: The MuEffect protocol allows arbitrary effects without modifying core MMM code, following the Open/Closed Principle.

5. **Separation of Concerns**:
   - MMM handles data transformation and model structure
   - MuEffects handle specific mathematical transformations
   - Clear interface via xr.Dataset prevents tight coupling

6. **Dual Transformation Path**: Both `_generate_and_preprocess_model_data()` (training) and `_posterior_predictive_data_transformation()` (prediction) use the same underlying `_create_xarray_from_pandas()` utility, ensuring consistency.

7. **Coordinate-Driven Architecture**: The system is fundamentally organized around named coordinates ("channel", "control", "target", "date", custom dims), making dimensionality explicit and reducing broadcasting errors.

## Implementation Recommendations

Based on this research, the recommended implementation approach is:

1. **Add `data_vars` dictionary** initialized with existing channel/control/target columns in `__init__`
2. **Create `register_data_columns()` method** for MuEffects to register their data needs
3. **Refactor transformation methods** to iterate over `data_vars` instead of hardcoded channel/control/target
4. **Maintain backward compatibility** by keeping existing column attributes
5. **Add validation** to ensure registered columns exist in input data
6. **Document the pattern** with examples in MuEffect docstring

The suggested solution from the issue is architecturally sound and aligns well with existing patterns. The main enhancement would be adding the `register_data_columns()` method to provide a clean API.

## Additional Considerations from Community Discussion

### Optional Columns for Out-of-Sample Predictions (Issue Comment by @TeemuSailynoja)

**Key Insight**: The proposed data registration approach would enable EventEffect to make true out-of-sample predictions with future events.

**Current EventEffect Limitation** (`pymc_marketing/mmm/additive_effect.py:442-567`):
- EventAdditiveEffect receives `df_events` (with start_date, end_date, name) during initialization
- In `create_data()`, these events are stored as PyMC Data in the model (lines 511-520)
- In `set_data()`, only the "days" reference is updated (lines 560-567)
- **Problem**: To make predictions including future events, you must include those events in the original `df_events` at model creation time
- **Consequence**: Cannot add new events discovered after model fitting for out-of-sample predictions

**Proposed Enhancement**: Add `optional` flag to `register_data_columns()`

```python
def register_data_columns(
    self,
    coordinate_name: str,
    column_names: list[str],
    dims: tuple[str, ...] | None = None,
    optional: bool = False,
) -> None:
    """Register additional data columns to include in predictions.

    Parameters
    ----------
    ...
    optional : bool, default False
        If True, columns are not required to exist in input DataFrame.
        Missing optional columns will be filled with zeros.
        Useful for event data where future events may not be known.
    """
```

**Benefits for EventEffect**:
1. Register event-related columns as optional during `create_data()`
2. If future event columns present in prediction DataFrame → include them
3. If not present → use zeros (no events) rather than failing
4. Eliminates need to pre-specify all future events at model creation
5. Enables truly dynamic out-of-sample event handling

**Example Usage**:
```python
class EventAdditiveEffect:
    def create_data(self, mmm):
        # Register event columns as optional
        mmm.register_data_columns(
            coordinate_name="future_events",
            column_names=["event_start", "event_end", "event_intensity"],
            optional=True,  # Don't fail if not present
        )
```

**Note**: This enhancement would **not require changes** to current MuEffect implementations except EventEffect itself. Existing effects (FourierEffect, LinearTrendEffect) don't need additional columns, so they remain unchanged.

## Open Questions

1. **dims Flexibility**: Should registered data sources support custom dims (different from `self.dims`), or always use `self.dims`? The issue's workaround shows `var_dims` as a parameter.

2. **Registration Timing**: Should registration happen in `create_data()` or earlier (e.g., in effect `__init__` via a protocol method)?

3. **Validation Timing**: When should column existence be validated - at registration time or during transformation?

4. **Backward Compatibility**: How to handle effects that might try to use the old `hasattr(mmm, 'additional_vars')` pattern?

5. **Coordinate Name Conflicts**: How to handle if an effect tries to register a coordinate_name that conflicts with existing coordinates?

6. **Target Special Handling**: Should target remain special-cased (due to `y` parameter and `.sum()` operation), or should it fully conform to the loop pattern?

7. **Optional Columns Support**: Should `register_data_columns()` support an `optional` flag? If so:
   - How should missing optional columns be handled? (zeros, NaN, custom fill value?)
   - Should there be a callback mechanism for effects to handle missing data differently?
   - How does this interact with the `.fillna(0)` currently applied to all xarray data?
   - Should optional columns create a separate tracking mechanism to inform effects which columns were actually present?

## Related Research

No previous research documents found in thoughts/shared/research/ or thoughts/shared/issues/ directories related to this specific topic.

## Next Steps

### Core Implementation
1. Decide on dims flexibility (use `self.dims` universally or allow custom dims per data source)
2. Implement `data_vars` dictionary initialization
3. Implement `register_data_columns()` method with validation
4. **Decide on optional columns support** and implementation strategy:
   - Should `optional` parameter be included in initial implementation?
   - Define behavior for missing optional columns (zeros vs. NaN vs. custom fill)
   - Consider impact on EventEffect out-of-sample predictions
5. Refactor `_posterior_predictive_data_transformation()` to use loop over `data_vars`
6. Refactor `_generate_and_preprocess_model_data()` to match

### Testing & Documentation
7. Add tests demonstrating custom MuEffect registering additional columns
8. Add tests for optional columns (if implemented)
9. Add tests for EventEffect with out-of-sample event predictions (if optional columns implemented)
10. Update MuEffect protocol documentation with data registration pattern
11. Add examples showing the data registration pattern in docstrings

### Backward Compatibility
12. Consider deprecation path for old `hasattr(mmm, 'additional_vars')` pattern if it exists in the wild
13. Ensure existing MuEffect implementations continue to work without modification (except EventEffect if enhanced)
