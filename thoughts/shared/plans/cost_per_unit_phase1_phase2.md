# cost_per_unit Implementation Plan - Phase 1 & Phase 2

**Date**: 2026-02-16
**Author**: Claude Sonnet 4.5
**Issue**: #2210
**Branch**: TBD
**Related Research**: [thoughts/shared/issues/2210/research.md](https://github.com/pymc-labs/pymc-marketing/blob/work-issue-2210/thoughts/shared/issues/2210/research.md)

## Table of Contents

- [Overview](#overview)
- [Changes from Original Research](#changes-from-original-research)
- [Current State Analysis](#current-state-analysis)
  - [Existing Architecture](#existing-architecture)
  - [Key Constraint](#key-constraint)
- [Desired End State](#desired-end-state)
  - [Verification](#verification)
- [What We're NOT Doing](#what-were-not-doing)
- [Phase 1: Core Infrastructure](#phase-1-core-infrastructure)
  - [Overview](#overview-1)
  - [Changes Required](#changes-required)
    - [1.1 Update InferenceData Schema](#11-update-inferencedata-schema)
    - [1.2 Add Model Initialization Parameter](#12-add-model-initialization-parameter)
    - [1.3 Shared parsing utility + post-fit injection](#13-shared-parsing-utility--post-fit-injection)
    - [1.4 Modify get_channel_spend() for Conversion](#14-modify-get_channel_spend-for-conversion)
    - [1.5 Add cost_per_unit Property](#15-add-cost_per_unit-property)
    - [1.6 Add set_cost_per_unit() Method to MMM Class](#16-add-set_cost_per_unit-method-to-mmm-class)
  - [Phase 1 Success Criteria](#phase-1-success-criteria)
- [Phase 2: Budget Optimization Support](#phase-2-budget-optimization-support)
  - [Overview](#overview-2)
  - [Changes Required](#changes-required-1)
    - [2.1 Add cost_per_unit Parameter to BudgetOptimizer](#21-add-cost_per_unit-parameter-to-budgetoptimizer)
    - [2.2 Update Budget Replacement Method](#22-update-budget-replacement-method)
    - [2.3 Add cost_per_unit to optimize_budget](#23-add-cost_per_unit-to-optimize_budget)
    - [2.4 Document Budget Units in allocate_budget](#24-document-budget-units-in-allocate_budget)
    - [2.5 Add Unit Conversion Documentation Comment](#25-add-unit-conversion-documentation-comment)
  - [Phase 2 Success Criteria](#phase-2-success-criteria)
- [Testing Strategy](#testing-strategy)
  - [Unit Tests](#unit-tests)
  - [Integration Tests](#integration-tests)
  - [Performance Tests](#performance-tests)
- [Migration Notes](#migration-notes)
  - [For Existing Users](#for-existing-users)
  - [For Model Serialization](#for-model-serialization)
- [Performance Considerations](#performance-considerations)
  - [Memory](#memory)
  - [Computation](#computation)
  - [Broadcasting Efficiency](#broadcasting-efficiency)
- [Open Questions (Resolved)](#open-questions-resolved)
- [Implementation Order](#implementation-order)
- [References](#references)
- [Changelog](#changelog)

## Overview

This plan implements support for `cost_per_unit` conversion in PyMC Marketing MMM models, enabling users to provide channel data in non-monetary units (impressions, clicks, etc.) and convert to spend units for ROAS calculations, budget optimization, and reporting. This addresses issue #2210.

**Key Design Principle**: Store raw channel data in original units, store `cost_per_unit` metadata separately, and apply conversion "on read" at strategic access points.

## Changes from Original Research

Based on user feedback, this plan includes:

1. **Single Input Format — `pd.DataFrame`**:
   - Wide-format DataFrame where rows are `(date, *custom_dims)` combinations
     and columns are channel names containing cost-per-unit values.
   - Not all model channels need to appear; missing channels default to 1.0
     (assumed to already be in spend units).
   - Naturally supports scalar, time-varying, per-channel, and
     per-custom-dim cost_per_unit without separate code paths.

2. **Post-hoc Setter Method**:
   - Add `mmm.set_cost_per_unit()` method on the MMM class
   - Enable setting cost_per_unit after model fitting

3. **All logic in the MMM class (no separate file)**:
   - A single `_parse_cost_per_unit_df()` static method on the MMM class
     converts the DataFrame to `xr.DataArray`; used by Phase 1 (fit-time
     injection and `mmm.set_cost_per_unit()`) **and** Phase 2 (optimizer
     wrapper's `_parse_cost_per_unit_for_optimizer()` delegates to it).
   - This eliminates a DRY violation: one parser, two call sites.

## Current State Analysis

### Existing Architecture

**Data Storage**:
- [pymc_marketing/data/idata/mmm_wrapper.py:227-251](pymc_marketing/data/idata/mmm_wrapper.py#L227-L251) - `get_channel_spend()` currently returns raw `channel_data` without conversion
- [pymc_marketing/data/idata/schema.py:267-295](pymc_marketing/data/idata/schema.py#L267-L295) - InferenceData schema defines `constant_data` variables

**ROAS Calculation**:
- [pymc_marketing/data/idata/mmm_wrapper.py:382-409](pymc_marketing/data/idata/mmm_wrapper.py#L382-L409) - `get_roas()` divides contributions by spend from `get_channel_spend()`
- [pymc_marketing/mmm/summary.py:481](pymc_marketing/mmm/summary.py#L481) - `MMMSummaryFactory.roas()` uses wrapper's `get_roas()`

**Budget Optimization**:
- [pymc_marketing/mmm/budget_optimizer.py:896-952](pymc_marketing/mmm/budget_optimizer.py#L896-L952) - `BudgetOptimizer._replace_channel_data_by_optimization_variable()` accesses `channel_data` directly through PyMC model graph
- [pymc_marketing/mmm/budget_optimizer.py:1010-1209](pymc_marketing/mmm/budget_optimizer.py#L1010-L1209) - `allocate_budget()` expects budgets in monetary units

**Model Initialization**:
- [pymc_marketing/mmm/base.py:55-85](pymc_marketing/mmm/base.py#L55-L85) - `MMMModelBuilder.__init__()` accepts model configuration
- [pymc_marketing/mmm/multidimensional.py:1662-1687](pymc_marketing/mmm/multidimensional.py#L1662-L1687) - Creates `pm.Data` variables that populate `constant_data`

### Key Constraint

**Conversion only affects reporting and optimization, NOT modeling**:
- The model is fit on raw channel data (impressions, clicks, etc.)
- Saturation and adstock curves operate on raw units
- Only ROAS calculations, summaries, and budget optimization need spend units

## Desired End State

After implementing Phase 1 and Phase 2:

1. **Users can initialize models with cost_per_unit**:
   ```python
   # Example: 2 channels (TV, Radio), 3 geos, 4 dates
   # Only TV is in impressions; Radio is already in dollars
   cost_per_unit = pd.DataFrame({
       "date":  dates.repeat(3),            # 12 rows = 4 dates × 3 geos
       "geo":   ["US", "UK", "DE"] * 4,
       "TV":    [0.01, 0.02, 0.015] * 4,    # cost per impression
   })

   mmm = MMM(
       channel_columns=["TV", "Radio"],
       dims=("geo",),
       cost_per_unit=cost_per_unit,
   )
   ```

   ```python
   # Simpler case: no custom dims, constant cost per channel
   cost_per_unit = pd.DataFrame({
       "date": dates,
       "TV":   [0.01] * len(dates),
       "Radio": [0.02] * len(dates),
   })
   mmm = MMM(
       channel_columns=["TV", "Radio"],
       cost_per_unit=cost_per_unit,
   )
   ```

2. **Users can add cost_per_unit post-hoc**:
   ```python
   # Fit model first
   mmm.fit(data_with_impressions)

   # Add conversion factors later (same DataFrame format)
   mmm.set_cost_per_unit(cost_per_unit_df)
   ```

3. **ROAS calculations use converted spend**:
   ```python
   roas = mmm.summary.get_roas()  # Automatically converts to spend units
   spend = mmm.data.get_channel_spend()  # Returns spend, not impressions
   ```

4. **Budget optimization works with cost_per_unit**:
   ```python
   # User provides future cost_per_unit for the optimization period
   future_dates = pd.date_range("2025-01-01", "2025-03-31", freq="W-MON")
   future_cpu = pd.DataFrame({
       "date": future_dates,
       "TV": [0.012] * len(future_dates),   # expected future CPM
       "Radio": [0.025] * len(future_dates),
   })

   wrapper = MultiDimensionalBudgetOptimizerWrapper(
       model=mmm,
       start_date="2025-01-01",
       end_date="2025-03-31",
   )
   result = wrapper.optimize_budget(
       budget=100000,  # In dollars
       cost_per_unit=future_cpu,  # Date-specific conversion rates
   )
   # Returns optimal_budgets in dollars
   ```

5. **Backward compatibility maintained**:
   - Models without `cost_per_unit` work exactly as before
   - No breaking changes to existing API

### Verification

**Automated**:
- [ ] Unit tests pass: `pytest tests/mmm/test_idata_wrapper.py -v`
- [ ] Budget optimizer tests pass: `pytest tests/mmm/test_budget_optimizer.py -v`
- [ ] Schema validation tests pass: `pytest tests/data/test_idata_schema.py -v`
- [ ] Integration tests pass: `pytest tests/mmm/test_multidimensional.py -v`

**Manual**:
- [ ] Fit MMM with impression data + cost_per_unit
- [ ] Verify ROAS values are sensible ($/$ not $/impression)
- [ ] Run budget optimization and check results
- [ ] Compare results with/without cost_per_unit on same data

## What We're NOT Doing

To prevent scope creep:

1. **NOT modifying saturation/adstock curves** - They continue to operate on raw units
2. **NOT changing plotting defaults** - Saturation curves plot in original units (Phase 3 feature)
3. **NOT supporting non-DataFrame inputs** - Only `pd.DataFrame` is accepted; no scalar, dict, or array shortcuts
4. **NOT supporting currency conversion** - Only unit→spend conversion, not EUR→USD
5. **NOT modifying model fitting** - Conversion happens post-training only

---

## Phase 1: Core Infrastructure

**Goal**: Add `cost_per_unit` storage to InferenceData and enable conversion in `get_channel_spend()`

### Overview

Phase 1 establishes the foundational infrastructure for storing and applying `cost_per_unit` conversion factors. This phase focuses on:
- Extending the InferenceData schema to support optional `cost_per_unit` metadata
- Adding model initialization parameters to accept various input formats
- Implementing conversion logic in the wrapper's `get_channel_spend()` method
- Creating setter methods for post-hoc specification

### Changes Required

#### 1.1 Update InferenceData Schema

**File**: [pymc_marketing/data/idata/schema.py](pymc_marketing/data/idata/schema.py#L293)
**Location**: After `target_scale` definition (around line 293)

**Add to `constant_data_vars` dictionary in `from_model_config()`**:

```python
constant_data_vars["cost_per_unit"] = VariableSchema(
    name="cost_per_unit",
    dims=("date", *custom_dims, "channel"),
    dtype=("float64", "float32", "int64", "int32"),
    description=(
        "Cost per unit conversion factors for non-spend channels. "
        "Shape: (date, *custom_dims, channel)"
    ),
    required=False,  # Optional metadata
)
```

**Rationale**:
- `dims=("date", *custom_dims, "channel")` matches the existing `channel_data` dim convention (channel last), ensuring consistency within `constant_data` and predictable xarray broadcasting
- `required=False` maintains backward compatibility
- Multiple dtypes support numpy, pandas, and xarray inputs

**Testing**:
- Add test in `tests/data/test_idata_schema.py` verifying optional validation
- Test with cost_per_unit: `dims=("date", *custom_dims, "channel")`
- Test with constant values broadcast across dates
- Test absence doesn't cause validation errors

---

#### 1.2 Add Model Initialization Parameter

**File**: [pymc_marketing/mmm/multidimensional.py](pymc_marketing/mmm/multidimensional.py#L402)
**Location**: Add to `MMM.__init__()` signature (around line 402)

> **Note**: `MMM` inherits from `RegressionModelBuilder`, **not** from
> `MMMModelBuilder` in `base.py` (they are sibling classes).  The
> `cost_per_unit` parameter belongs on `MMM` because it is the only class
> that has `dims`, channel metadata, and the parsing/injection logic.

**Add `cost_per_unit` parameter** to the existing `MMM.__init__` signature
(after `adstock_first` or another appropriate position):

```python
    cost_per_unit: pd.DataFrame | None = Field(
        default=None,
        description=(
            "Cost per unit conversion factors for non-spend channels. "
            "Wide-format DataFrame where rows are (date, *custom_dims) "
            "combinations and columns are channel names containing cost "
            "values. Not all model channels need to appear; missing "
            "channels default to 1.0 (already in spend units)."
        ),
    ),
```

**Store in `__init__` body** (near the other attribute assignments):

```python
    self._cost_per_unit_input = cost_per_unit  # Raw DataFrame, parsed at fit time
```

> **Naming convention**: The raw DataFrame is stored as a **private**
> attribute `_cost_per_unit_input` (not `self.cost_per_unit`) to avoid
> confusion with the public read API `mmm.data.cost_per_unit` (section
> 1.5), which returns the parsed `xr.DataArray` from idata.  The two
> represent the same information at different lifecycle stages:
>
> | | `mmm._cost_per_unit_input` | `mmm.data.cost_per_unit` |
> |---|---|---|
> | **Type** | `pd.DataFrame \| None` | `xr.DataArray \| None` |
> | **Available** | At init (before fit) | After fit (from idata) |
> | **Purpose** | Stage raw user input | Provide processed data for computation |
> | **Visibility** | Private (internal) | Public read API |
>
> The `__init__` parameter is still called `cost_per_unit` for
> user-friendliness, but internally stored with a private name.

**Update `__eq__`** (lines 590–707): The MMM `__eq__` method compares all
configuration attributes.  Add a `_cost_per_unit_input` comparison before the
final `return True` (around line 706):

```python
# cost_per_unit
if (self._cost_per_unit_input is None) != (other._cost_per_unit_input is None):
    return False
if self._cost_per_unit_input is not None and not self._cost_per_unit_input.equals(other._cost_per_unit_input):
    return False
```

Without this, two models with different `cost_per_unit` DataFrames would
incorrectly compare as equal.

**Rationale**:
- Single input format (DataFrame) eliminates ambiguity and multi-format code paths
- DataFrame naturally supports all dimensionality combinations:
  scalar, time-varying, per-channel, per-custom-dim
- Stores raw input as private `_cost_per_unit_input` to prevent confusion
  with the public `mmm.data.cost_per_unit` property (section 1.5)
- Conversion deferred to fit-time injection
- Maintains backward compatibility with `default=None`
- Lives on `MMM` (not `MMMModelBuilder`) because `MMM` owns `dims`,
  channel metadata, and all cost_per_unit logic

---

#### 1.3 Shared parsing utility + post-fit injection

**Why NOT `pm.Data`**: `cost_per_unit` is never referenced by any model
variable -- no saturation, adstock, or likelihood depends on it.  It is
purely reporting/optimization metadata (see [Key Constraint](#key-constraint)).
Using `pm.Data` would add an unused node to the PyMC graph, which is
misleading and inconsistent with the post-hoc setter path (section 1.6)
that injects xarray directly.  Storing as xarray only keeps the model
graph clean and uses a single code path regardless of *when*
`cost_per_unit` is provided.

##### 1.3a Shared utility: `_parse_cost_per_unit_df()` (static method on MMM)

**File**: `pymc_marketing/mmm/multidimensional.py`

A private **static method** on the MMM class that takes explicit
coordinate parameters.  This makes it reusable by both Phase 1
(fit-time injection (1.3b) and `mmm.set_cost_per_unit()` (1.6)) and
Phase 2 (`_parse_cost_per_unit_for_optimizer()` in section 2.3), which
simply calls this static method with its own coordinates.

Instance-method wrappers on `MMM` and
`MultiDimensionalBudgetOptimizerWrapper` extract the appropriate
coordinates from their respective contexts and delegate to this static
method.

```python
@staticmethod
def _parse_cost_per_unit_df(
    df: pd.DataFrame,
    channels: list[str],
    dates: pd.DatetimeIndex | np.ndarray,
    custom_dims: tuple[str, ...] = (),
    custom_dim_coords: dict[str, np.ndarray] | None = None,
) -> xr.DataArray:
    """Convert a cost_per_unit DataFrame to an xr.DataArray.

    The DataFrame is wide-format: rows are ``(date, *custom_dims)``
    combinations, and columns include channel names with cost values.
    Channels not present in the DataFrame default to 1.0 (assumed
    already in spend units).

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame.  Must contain a ``"date"`` column and
        one column per custom dimension.  Remaining columns are
        interpreted as channel names.
    channels : list of str
        All channel names the model knows about.
    dates : pd.DatetimeIndex or np.ndarray
        Expected date coordinates.
    custom_dims : tuple of str, optional
        Names of custom dimensions (e.g. ``("geo",)``).
    custom_dim_coords : dict[str, np.ndarray] or None, optional
        Coordinate values for each custom dimension.

    Returns
    -------
    xr.DataArray
        Dims ``("date", "channel")`` when no custom dims, or
        ``("date", *custom_dims, "channel")`` otherwise.
        Matches ``channel_data``'s dim convention (channel last).
        Missing channels are filled with 1.0.

    Raises
    ------
    ValueError
        If ``"date"`` column is missing, if custom dim columns are
        missing, if unknown channel columns are present, if no channel
        columns are present, if any values are non-positive, or if
        ``reindex`` against the model's coordinates produces NaN values
        (indicating date/dim mismatch — e.g. timezone differences,
        missing dates, or a subset of dates that doesn't cover the
        model's full date range).

    Examples
    --------
    >>> mmm.fit(data_with_impressions)
    >>> # 2 channels, 3 geos, 4 dates — cost_per_unit for TV only
    >>> df = pd.DataFrame({
    ...     "date": dates.repeat(3),
    ...     "geo": ["US", "UK", "DE"] * 4,
    ...     "TV": [0.01, 0.02, 0.015] * 4,
    ... })
    >>> result = MMM._parse_cost_per_unit_df(
    ...     df,
    ...     channels=["TV", "Radio"],
    ...     dates=dates,
    ...     custom_dims=("geo",),
    ...     custom_dim_coords={"geo": np.array(["US", "UK", "DE"])},
    ... )
    >>> result.dims  # ("date", "geo", "channel")
    >>> result.sel(channel="Radio")  # all 1.0
    """
    if "date" not in df.columns:
        raise ValueError(
            "cost_per_unit DataFrame must contain a 'date' column."
        )

    # Identify dim columns vs channel value columns
    dim_cols = ["date", *custom_dims]
    missing_dims = set(dim_cols) - set(df.columns)
    if missing_dims:
        raise ValueError(
            f"cost_per_unit DataFrame missing dim columns: {missing_dims}"
        )

    value_cols = [c for c in df.columns if c not in dim_cols]
    unknown_channels = set(value_cols) - set(channels)
    if unknown_channels:
        raise ValueError(
            f"cost_per_unit DataFrame contains unknown channels: "
            f"{unknown_channels}. Model channels are: {channels}"
        )

    if not value_cols:
        raise ValueError(
            "cost_per_unit DataFrame has no channel columns. "
            f"Expected at least one of: {channels}"
        )

    # Validate positive values
    numeric_values = df[value_cols]
    if (numeric_values <= 0).any().any():
        raise ValueError(
            "cost_per_unit values must be positive (> 0). "
            "Zero or negative values would cause division-by-zero "
            "or negative spend in downstream calculations."
        )

    # Set multi-index and convert to xarray Dataset
    df_indexed = df.set_index(dim_cols)
    ds = df_indexed.to_xarray()

    # Build full DataArray with all channels (missing → 1.0)
    dim_order = ["date", *custom_dims]
    coord_dict = {"date": dates}
    if custom_dim_coords:
        coord_dict.update(custom_dim_coords)

    full_shape = tuple(len(coord_dict[d]) for d in dim_order)
    channel_arrays = []
    for ch in channels:
        if ch in ds:
            reindexed = ds[ch].reindex(coord_dict)
            if reindexed.isnull().any():
                # Identify which coordinates caused the mismatch
                nan_positions = reindexed.where(reindexed.isnull(), drop=True)
                raise ValueError(
                    f"cost_per_unit reindex produced NaN values for channel "
                    f"'{ch}'. This typically means the DataFrame's date (or "
                    f"custom dim) values don't exactly match the model's "
                    f"coordinates (e.g. timezone mismatch, missing dates, or "
                    f"subset of dates). NaN positions:\n{nan_positions.coords}"
                )
            channel_arrays.append(reindexed.values)
        else:
            channel_arrays.append(np.ones(full_shape))

    result = xr.DataArray(
        np.stack(channel_arrays, axis=len(dim_order)),
        dims=(*dim_order, "channel"),
        coords={**coord_dict, "channel": channels},
    )

    # Ensure standard dim order: (date, *custom_dims, channel)
    # Matches channel_data's convention (channel last)
    return result.transpose("date", *custom_dims, "channel")
```

**Instance-method convenience wrapper** (used by 1.3b and 1.6 — extracts
coordinates from the fitted model's idata and delegates to the static
method):

**Design note — static/instance split**:  The parsing logic is a
`@staticmethod` with explicit coordinate parameters so that (a) it is
unit-testable in isolation without constructing a full MMM object, and
(b) both Phase 1 (MMM) and Phase 2 (optimizer wrapper) can reuse it
with their own coordinates (training dates vs. optimization-window
dates).  The instance method below is a thin convenience wrapper that
extracts coordinates from `self` and delegates — it exists only to
avoid repeating the coordinate-extraction boilerplate at each call site.

```python
def _build_cost_per_unit_array(self, df: pd.DataFrame) -> xr.DataArray:
    """Parse cost_per_unit DataFrame using coordinates from the fitted model."""
    custom_dims = tuple(self.data.custom_dims)
    return MMM._parse_cost_per_unit_df(
        df=df,
        channels=self.data.channels,
        dates=self.data.dates,
        custom_dims=custom_dims,
        custom_dim_coords={
            dim: self.data.idata.constant_data.coords[dim].values
            for dim in custom_dims
        } if custom_dims else None,
    )
```

##### 1.3b Fit-time injection: Override `fit()` in MMM

**File**: [pymc_marketing/mmm/multidimensional.py](pymc_marketing/mmm/multidimensional.py)
**Approach**: Override `fit()` in MMM, call `super().fit()`, then inject
`cost_per_unit` into idata.

**Implementation**:

```python
def fit(
    self,
    X: pd.DataFrame | xr.Dataset | xr.DataArray,
    y: pd.Series | xr.DataArray | np.ndarray | None = None,
    progressbar: bool | None = None,
    random_seed: RandomState | None = None,
    **kwargs: Any,
) -> az.InferenceData:
    idata = super().fit(
        X, y, progressbar=progressbar, random_seed=random_seed, **kwargs
    )
    if self._cost_per_unit_input is not None:
        cpu_array = self._build_cost_per_unit_array(self._cost_per_unit_input)
        self.idata.constant_data["cost_per_unit"] = cpu_array
    return idata
```

> **Note**: The override must match the full parent signature
> (`RegressionModelBuilder.fit` at model_builder.py line 960).
> A minimal `(self, X, y, **kwargs)` would incorrectly bind
> `progressbar` or `random_seed` when passed positionally.

**Rationale**:
- Zero changes to the base class — no risk of breaking other model types
  (CLV, choice models)
- Explicit and self-contained: all cost_per_unit logic lives in `MMM`
- Easy to understand: "fit, then inject metadata"
- Keeps the PyMC model graph clean — no unused `pm.Data` node
- Same code path as the post-hoc setter (section 1.6), both calling
  `self._build_cost_per_unit_array()` which delegates to the shared
  static `_parse_cost_per_unit_df()`
- DataFrame-only input eliminates multi-format complexity
- Missing channels default to 1.0 — channels already in spend units
  simply don't appear in the DataFrame
- Serialization: `cost_per_unit` is persisted when saving idata (xarray
  variables in `constant_data` are serialized by ArviZ automatically)

**Error Handling**:
- Missing `date` or custom dim in DataFrame → clear `ValueError`
- Unknown channel names (not in model) → `ValueError`
- Non-positive values (≤ 0) → `ValueError` (prevents division-by-zero
  and negative spend in downstream calculations like `get_roas()`)
- Reindex produces NaN (date/dim coordinate mismatch) → `ValueError`
  with diagnostic info showing which coordinates didn't match
- Missing model channels → silently filled with 1.0

---

#### 1.4 Modify get_channel_spend() for Conversion

**File**: [pymc_marketing/data/idata/mmm_wrapper.py](pymc_marketing/data/idata/mmm_wrapper.py#L227)
**Location**: Replace existing `get_channel_spend()` method (lines 227-251)

**Replace with**:

```python
def get_channel_spend(
    self,
    apply_cost_per_unit: bool = True
) -> xr.DataArray:
    """Get channel spend data, optionally converting non-spend units.

    Returns raw channel spend data (not MCMC samples). If cost_per_unit
    metadata exists and apply_cost_per_unit=True, multiplies channel_data
    by cost_per_unit to convert from original units to spend units.

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

    Raises
    ------
    ValueError
        If channel_data not found in constant_data

    Examples
    --------
    >>> # Get converted spend (default)
    >>> spend = mmm.data.get_channel_spend()
    >>>
    >>> # Get raw data without conversion
    >>> raw = mmm.data.get_channel_spend(apply_cost_per_unit=False)
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
    if apply_cost_per_unit and self.cost_per_unit is not None:
        channel_data = channel_data * self.cost_per_unit

    return channel_data
```

**Rationale**:
- Default behavior applies conversion (opt-out with `apply_cost_per_unit=False`)
- xarray broadcasting handles both scalar and time-varying automatically
- Backward compatible: models without cost_per_unit work unchanged
- Clear documentation of behavior

**Impact**:
- `get_roas()` automatically uses converted spend (calls `get_channel_spend()`)
- Summary methods automatically report correct ROAS
- `Incrementality.compute_incremental_contribution` (incrementality.py line 377)
  must be updated to pass `apply_cost_per_unit=False` — this caller feeds raw
  channel data into the compiled PyTensor model graph, which expects values in
  the original units the model was trained on (impressions, clicks, etc.).
  Without this opt-out, cost_per_unit conversion would silently feed
  dollar-converted values into saturation curves calibrated for raw units,
  producing incorrect counterfactual results.
- The dim-checking call at line 669 (`self.data.get_channel_spend().dims`) is
  unaffected — dims are the same regardless of conversion.
- `_aggregate_channel_spend` (line 1474) correctly uses the default
  `apply_cost_per_unit=True` since it provides the ROAS/CAC denominator.

**Required change in `incrementality.py`**:

```python
# incrementality.py line 377 — BEFORE:
baseline_array = self.data.get_channel_spend().values

# AFTER (opt out of cost_per_unit conversion for model-graph input):
baseline_array = self.data.get_channel_spend(apply_cost_per_unit=False).values
```

---

#### 1.5 Add cost_per_unit Property

**File**: [pymc_marketing/data/idata/mmm_wrapper.py](pymc_marketing/data/idata/mmm_wrapper.py#L252)
**Location**: After `get_channel_spend()` method (around line 252)

**Add new property**:

```python
@property
def cost_per_unit(self) -> xr.DataArray | None:
    """Cost per unit conversion factors, or None if not set.

    Returns
    -------
    xr.DataArray or None
        Cost per unit values with dims ("date", *custom_dims, "channel").
        Returns None if cost_per_unit metadata not present.

    Examples
    --------
    >>> if mmm.data.cost_per_unit is not None:
    ...     print(f"Using conversion: {mmm.data.cost_per_unit}")
    """
    if (
        hasattr(self.idata, "constant_data")
        and "cost_per_unit" in self.idata.constant_data
    ):
        return self.idata.constant_data.cost_per_unit
    return None
```

**Rationale**:
- **This is the public read API** for cost_per_unit.  Users and internal
  callers should use `mmm.data.cost_per_unit` (not the private
  `mmm._cost_per_unit_input`, which is the raw init-time DataFrame).
  This mirrors the pattern of `mmm.channel_columns` (init param) vs
  `mmm.data.channels` (wrapper accessor from idata).
- Property provides clean, idiomatic access (read like an attribute, not a method call)
- Used internally by `get_channel_spend()` (section 1.4) for conversion
- Follows pattern of other data attributes

---

#### 1.6 Add set_cost_per_unit() Method to MMM Class

**File**: [pymc_marketing/mmm/multidimensional.py](pymc_marketing/mmm/multidimensional.py#L3200)
**Location**: After `optimize_budget()` method (around line 3200)

**Add new method**:

```python
def set_cost_per_unit(
    self,
    cost_per_unit: pd.DataFrame,
    overwrite: bool = False,
) -> None:
    """Set or update cost_per_unit metadata for the fitted model.

    Allows post-hoc specification of conversion factors after model
    fitting.  The cost_per_unit metadata is stored in
    ``idata.constant_data`` and used by ``get_channel_spend()`` to
    convert from original units to spend units.

    Parameters
    ----------
    cost_per_unit : pd.DataFrame
        Wide-format DataFrame.  Rows are ``(date, *custom_dims)``
        combinations; columns are channel names with cost values.
        Not all model channels need to appear; missing channels
        default to 1.0 (assumed already in spend units).
    overwrite : bool, default False
        If True, overwrite existing cost_per_unit. If False and
        cost_per_unit already exists, raise ValueError.

    Raises
    ------
    RuntimeError
        If model has not been fitted yet (no idata available).
    ValueError
        If cost_per_unit exists and overwrite=False, or if
        date/dim values don't match the fitted data.

    Examples
    --------
    >>> mmm.fit(data_with_impressions)
    >>> cpu_df = pd.DataFrame({
    ...     "date": dates, "TV": [0.01] * len(dates),
    ... })
    >>> mmm.set_cost_per_unit(cpu_df)
    >>> roas = mmm.summary.roas()
    """
    if not hasattr(self, "idata") or self.idata is None:
        raise RuntimeError(
            "Model must be fitted before setting cost_per_unit. "
            "Call mmm.fit() first."
        )

    if not hasattr(self.idata, "constant_data"):
        raise ValueError("InferenceData missing constant_data group")

    # Check if already exists
    if "cost_per_unit" in self.idata.constant_data and not overwrite:
        raise ValueError(
            "cost_per_unit already exists in InferenceData. "
            "Use overwrite=True to replace it."
        )

    cost_per_unit_array = self._build_cost_per_unit_array(cost_per_unit)
    self.idata.constant_data["cost_per_unit"] = cost_per_unit_array
```

**Rationale**:
- Single entry point for setting cost_per_unit (`mmm.set_cost_per_unit()`)
- All logic lives in the MMM class — no delegation to wrapper, no separate file
- Uses the shared `_build_cost_per_unit_array()` convenience wrapper (section 1.3a),
  which delegates to the static `_parse_cost_per_unit_df()`
- Checks that model has been fitted first
- Protects against accidental overwrites

---

### Phase 1 Success Criteria

#### Automated Verification:
- [ ] Schema validation tests pass: `pytest tests/data/test_idata_schema.py::test_cost_per_unit_schema -v`
- [ ] Wrapper tests pass: `pytest tests/mmm/test_idata_wrapper.py::test_get_channel_spend_with_cost_per_unit -v`
- [ ] MMM setter tests pass: `pytest tests/mmm/test_cost_per_unit.py::test_set_cost_per_unit -v`
- [ ] ROAS tests pass: `pytest tests/mmm/test_summary.py::test_roas_with_cost_per_unit -v`
- [ ] Model initialization tests pass: `pytest tests/mmm/test_multidimensional.py::test_init_with_cost_per_unit -v`
- [ ] Type checking passes: `mypy pymc_marketing/data/idata/mmm_wrapper.py`

#### Manual Verification:
- [ ] Create MMM with cost_per_unit (constant values broadcast across dates), verify ROAS calculations
- [ ] Create MMM with time-varying cost_per_unit (values that change per date), verify conversion
- [ ] Fit MMM, then call `mmm.set_cost_per_unit(df)`, verify ROAS updates
- [ ] Verify backward compatibility: existing tests without cost_per_unit still pass

---

## Phase 2: Budget Optimization Support

**Goal**: Enable the budget optimizer to convert dollar budgets to original
units using **user-provided future `cost_per_unit`** values, preserving
date-varying rates across the optimization window.

### Overview

Phase 2 adds `cost_per_unit` support to the budget optimizer.  Unlike
Phase 1 (which stores *historical* cost_per_unit in InferenceData for
reporting), Phase 2 accepts *forward-looking* cost_per_unit values
directly from the user.  This design reflects the reality that future
media costs (contracted CPMs, seasonal pricing, etc.) are typically
different from historical averages.

**Key design decisions**:

1. **User-provided, not model-derived** — The optimizer accepts
   `cost_per_unit` as an explicit parameter rather than reading from the
   model's historical data.  This makes Phase 2 **independent of Phase 1**.
2. **Date-varying** — `cost_per_unit` has a date dimension matching the
   optimization window (`num_periods`), so each time step converts using
   its own rate.  No averaging over dates.
3. **Applied after time distribution** — The conversion happens *after*
   the budget is distributed over time (gaining a date dimension) and
   *before* channel scaling, ensuring correct per-period conversion.

### Changes Required

#### 2.1 Add `cost_per_unit` Parameter to BudgetOptimizer

**File**: [pymc_marketing/mmm/budget_optimizer.py](pymc_marketing/mmm/budget_optimizer.py#L562)
**Location**: `BudgetOptimizer` class, add new Field (around line 636)

**Add after `budget_distribution_over_period` field**:

```python
cost_per_unit: DataArray | None = Field(
    default=None,
    description=(
        "Cost per unit conversion factors for converting budgets from "
        "monetary units (dollars) to original units (impressions, clicks). "
        "Must have dims (date, *budget_dims) where date has length "
        "num_periods. If None, budgets are assumed to already be in "
        "the model's native units (no conversion applied)."
    ),
)
```

**In `__init__`** (around line 735), validate and convert to a PyTensor constant:

```python
# 5b. Validate and process cost_per_unit
self._cost_per_unit_tensor = self._validate_and_process_cost_per_unit(
    cost_per_unit=self.cost_per_unit,
    num_periods=self.num_periods,
    budget_dims=self._budget_dims,
)
```

**Add validation method** (after `_validate_and_process_budget_distribution`):

```python
def _validate_and_process_cost_per_unit(
    self,
    cost_per_unit: DataArray | None,
    num_periods: int,
    budget_dims: list[str],
) -> pt.TensorConstant | None:
    """Validate and convert cost_per_unit to a PyTensor constant.

    Parameters
    ----------
    cost_per_unit : DataArray or None
        Cost per unit with dims (date, *budget_dims).
    num_periods : int
        Number of optimization periods.
    budget_dims : list[str]
        Budget dimension names (excluding 'date').

    Returns
    -------
    pt.TensorConstant or None
        Constant tensor with shape (num_periods, *budget_shape), or
        None if no cost_per_unit provided.

    Raises
    ------
    ValueError
        If dimensions or date length don't match expectations.
    """
    if cost_per_unit is None:
        return None

    expected_dims = ("date", *budget_dims)
    if set(cost_per_unit.dims) != set(expected_dims):
        raise ValueError(
            f"cost_per_unit must have dims {expected_dims}, "
            f"but got {cost_per_unit.dims}"
        )

    if len(cost_per_unit.coords["date"]) != num_periods:
        raise ValueError(
            f"cost_per_unit date dimension must have length {num_periods}, "
            f"but got {len(cost_per_unit.coords['date'])}"
        )

    if (cost_per_unit <= 0).any():
        raise ValueError("cost_per_unit values must be positive.")

    values = cost_per_unit.transpose(*expected_dims).values
    return pt.constant(values, name="cost_per_unit")
```

**Rationale**:
- Explicit parameter makes the API self-documenting
- Validation catches shape mismatches early (before graph compilation)
- Converting to `pt.constant` once avoids repeated conversions
- Positive-value check prevents division by zero in the graph

---

#### 2.2 Update Budget Replacement Method

**File**: [pymc_marketing/mmm/budget_optimizer.py](pymc_marketing/mmm/budget_optimizer.py#L896)
**Location**: Modify `_replace_channel_data_by_optimization_variable()` method

The key change: move channel_scales division and insert cost_per_unit
conversion **after** time distribution, so the budget has a date dimension
when cost_per_unit (which varies by date) is applied.

**Find this section** (around lines 904-923):

```python
# Scale budgets by channel_scales
budgets = self._budgets
budgets /= channel_scales

# Repeat budgets over num_periods
repeated_budgets_shape = list(tuple(budgets.shape))
repeated_budgets_shape.insert(date_dim_idx, num_periods)

if self._budget_distribution_over_period_tensor is not None:
    # Apply time distribution factors
    repeated_budgets = self._apply_budget_distribution_over_period(
        budgets, num_periods, date_dim_idx
    )
else:
    # Default behavior: distribute evenly across periods
    repeated_budgets = pt.broadcast_to(
        pt.expand_dims(budgets, date_dim_idx),
        shape=repeated_budgets_shape,
    )
```

**Replace with**:

```python
budgets = self._budgets

# Repeat budgets over num_periods (still in monetary units)
repeated_budgets_shape = list(tuple(budgets.shape))
repeated_budgets_shape.insert(date_dim_idx, num_periods)

if self._budget_distribution_over_period_tensor is not None:
    repeated_budgets = self._apply_budget_distribution_over_period(
        budgets, num_periods, date_dim_idx
    )
else:
    repeated_budgets = pt.broadcast_to(
        pt.expand_dims(budgets, date_dim_idx),
        shape=repeated_budgets_shape,
    )

# Convert from monetary units to original units using date-specific rates.
# Applied AFTER time distribution so each period uses its own cost rate.
# Example: $1000 at week 1 / ($0.01/impression) = 100,000 impressions
if self._cost_per_unit_tensor is not None:
    repeated_budgets = repeated_budgets / self._cost_per_unit_tensor

# Apply model's channel scaling (original units → model scale)
repeated_budgets /= channel_scales
```

**Note on Order of Operations**:
1. Start with per-channel budgets in $ (from optimizer)
2. Distribute over time → $ per period per channel
3. Convert to original units: `budgets / cost_per_unit[t]` ($ → impressions)
4. Apply scaling: `budgets / channel_scales` (impressions → model scale)
5. Result flows through saturation/adstock/contribution

Moving `channel_scales` after time distribution is mathematically
equivalent to before (it has no date dimension and broadcasts), but keeps
the conversion pipeline readable: dollars → units → model scale.

**Broadcasting assumption — `date_dim_idx == 0`**:
The division `repeated_budgets / self._cost_per_unit_tensor` requires both
tensors to have compatible shapes.  `_cost_per_unit_tensor` is created
with shape `(num_periods, *budget_shape)` via
`cost_per_unit.transpose("date", *budget_dims).values`, which places the
date dimension first.  After time distribution, `repeated_budgets` also
has date at position `date_dim_idx`.  This works because
`channel_data_dims` is always `("date", *custom_dims, "channel")` — date
is the **first** dimension — so `date_dim_idx` is always 0.  If a future
refactoring were to change `channel_data_dims` to place date elsewhere
(e.g. channel-first ordering), the `_cost_per_unit_tensor` transpose
would need updating to match.  An assertion should be added during
implementation:

```python
assert date_dim_idx == 0, (
    "cost_per_unit conversion assumes date is the first dimension "
    f"in channel_data_dims, but date_dim_idx={date_dim_idx}. "
    "If channel_data_dims ordering has changed, update "
    "_cost_per_unit_tensor transpose accordingly."
)
```

---

#### 2.3 Add `cost_per_unit` to `optimize_budget`

**File**: [pymc_marketing/mmm/multidimensional.py](pymc_marketing/mmm/multidimensional.py#L3186)
**Location**: `MultiDimensionalBudgetOptimizerWrapper.optimize_budget()` method

**Add `cost_per_unit` parameter** to the method signature:

```python
def optimize_budget(
    self,
    budget: float | int,
    budget_bounds: xr.DataArray | None = None,
    response_variable: str = "total_media_contribution_original_scale",
    utility_function: UtilityFunctionType = average_response,
    constraints: Sequence[dict[str, Any]] = (),
    default_constraints: bool = True,
    budgets_to_optimize: xr.DataArray | None = None,
    budget_distribution_over_period: xr.DataArray | None = None,
    cost_per_unit: pd.DataFrame | xr.DataArray | None = None,
    callback: bool = False,
    **minimize_kwargs,
) -> ...:
    """Optimize budget allocation across channels.

    Parameters
    ----------
    ...
    cost_per_unit : pd.DataFrame or xr.DataArray or None, optional
        Cost per unit conversion factors for the **optimization period**.
        Converts budgets from monetary units (e.g., dollars) to the
        model's native channel units (e.g., impressions).

        - pd.DataFrame: Wide-format with a ``"date"`` column matching the
          optimization window dates, plus one column per channel.
          Missing channels default to 1.0 (no conversion).
        - xr.DataArray: Must have dims ``("date", *budget_dims)`` where
          ``date`` has length ``num_periods``.

        If None, no conversion is applied (budgets are assumed to be in
        the model's native units).

        **This is independent of Phase 1's historical cost_per_unit.**
        Use this to provide forward-looking cost rates (contracted CPMs,
        expected future pricing, etc.) for the optimization window.
    ...
    """
```

**Parse DataFrame to DataArray before passing to BudgetOptimizer**:

```python
# Parse cost_per_unit if provided as DataFrame
cost_per_unit_da = None
if cost_per_unit is not None:
    if isinstance(cost_per_unit, pd.DataFrame):
        cost_per_unit_da = self._parse_cost_per_unit_for_optimizer(cost_per_unit)
    elif isinstance(cost_per_unit, xr.DataArray):
        cost_per_unit_da = cost_per_unit
    else:
        raise TypeError(
            "cost_per_unit must be a pd.DataFrame or xr.DataArray, "
            f"got {type(cost_per_unit)}"
        )

allocator = BudgetOptimizer(
    num_periods=self.num_periods,
    ...,
    cost_per_unit=cost_per_unit_da,
)
```

**Add helper method on the wrapper** — a thin delegation to the shared
static `MMM._parse_cost_per_unit_df()` (section 1.3a), extracting
optimization-window coordinates from the wrapper's context:

```python
def _parse_cost_per_unit_for_optimizer(
    self,
    df: pd.DataFrame,
) -> xr.DataArray:
    """Parse a cost_per_unit DataFrame for the optimization window.

    Delegates to ``MMM._parse_cost_per_unit_df()`` (the shared static
    utility from Phase 1, section 1.3a) with coordinates appropriate
    for the optimization window rather than the training data.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame with ``"date"`` column and channel columns.
        Must have ``num_periods`` unique dates matching the optimization
        window.

    Returns
    -------
    xr.DataArray
        Dims ``("date", *custom_dims, "channel")`` with ``date`` length
        ``num_periods``.
    """
    channels = self.model_class.channel_columns
    custom_dims = tuple(self.model_class.dims)
    dates = sorted(df["date"].unique())

    custom_dim_coords = None
    if custom_dims:
        idata = self.model_class.idata
        custom_dim_coords = {
            dim: idata.constant_data.coords[dim].values
            for dim in custom_dims
        }

    return MMM._parse_cost_per_unit_df(
        df=df,
        channels=channels,
        dates=dates,
        custom_dims=custom_dims,
        custom_dim_coords=custom_dim_coords,
    )
```

**Rationale**:
- **No DRY violation**: Delegates to the same static parsing utility used
  by Phase 1 (section 1.3a), so all parsing logic — column validation,
  channel defaulting, positive-value checks — lives in one place.
  A bug fix to the parser automatically applies to both Phase 1 and Phase 2.
- DataFrame input at the wrapper level provides a user-friendly API
- DataArray input at the `BudgetOptimizer` level provides a clean low-level API
- Missing channels default to 1.0 (consistent with Phase 1 convention)
- Positive-value validation inherited from the shared utility

---

#### 2.4 Document Budget Units in allocate_budget

**File**: [pymc_marketing/mmm/budget_optimizer.py](pymc_marketing/mmm/budget_optimizer.py#L1010)
**Location**: Update `allocate_budget()` docstring

**Update Notes section**:

```python
    Notes
    -----
    **Units and cost_per_unit**:
    - All budget inputs (total_budget, budget_bounds) are in monetary units
    - If cost_per_unit is provided, the optimizer converts internally:
      budget_in_original_units[t] = budget_in_dollars[t] / cost_per_unit[t]
    - Each time period uses its own cost_per_unit value (no averaging)
    - Output optimal_budgets are in monetary units for user convenience
    - This ensures API consistency regardless of channel data units

    **cost_per_unit is forward-looking**:
    - cost_per_unit represents expected future costs for the optimization
      window, NOT historical costs from the training data
    - This is independent of any historical cost_per_unit stored in the
      model's InferenceData (Phase 1)
    - Typical sources: media buying contracts, seasonal rate cards,
      projected CPM/CPC rates
```

**Rationale**:
- Clarifies that budget inputs/outputs are always in monetary units
- Emphasizes the forward-looking nature and Phase 1 independence
- Documents per-period (not averaged) conversion behavior

---

#### 2.5 Add Unit Conversion Documentation Comment

**File**: [pymc_marketing/mmm/budget_optimizer.py](pymc_marketing/mmm/budget_optimizer.py#L902)
**Location**: Before cost_per_unit conversion code (added in 2.2)

**Add comment block**:

```python
# Budget Unit Conversion Pipeline
# --------------------------------
# The budget optimizer converts user-facing dollar budgets into model-scale
# channel data through the following steps:
#
# 1. User Input: Budgets in monetary units (e.g., dollars)
#    - total_budget = $100,000
#    - budget_bounds = {"TV": ($0, $50,000), ...}
#    - cost_per_unit = DataArray with dims (date, channel, ...)
#
# 2. Time Distribution: Spread per-channel budget over optimization periods
#    - Shape: (*budget_dims) → (date, *budget_dims)
#    - Each period gets a fraction of the total channel budget
#
# 3. cost_per_unit Conversion: Convert to original units (if provided)
#    - Applied per time period using date-specific rates:
#      budgets_in_units[t] = budgets_in_dollars[t] / cost_per_unit[t]
#    - Example: $5,000 at week 3 / ($0.012/impression) = 416,667 impressions
#    - No averaging: each period uses its own rate
#
# 4. Channel Scaling: Apply model's normalization
#    - budgets_scaled = budgets_in_units / channel_scales
#    - Matches the scaling applied to channel_data during training
#
# 5. Model Propagation: Scaled budgets flow through:
#    - Saturation transformation
#    - Adstock transformation
#    - Contribution calculation
```

**Rationale**:
- Documents the updated conversion flow with per-period cost_per_unit
- Helps future maintainers understand the transformation pipeline
- Clarifies the new order: time distribution → cost conversion → scaling

---

### Phase 2 Success Criteria

#### Automated Verification:
- [ ] Budget optimizer tests pass: `pytest tests/mmm/test_budget_optimizer.py -v`
- [ ] Multidimensional budget tests pass: `pytest tests/mmm/test_budget_optimizer_multidimensional.py -v`
- [ ] Test `_validate_and_process_cost_per_unit()` rejects wrong dims, wrong date length, non-positive values
- [ ] Test with constant cost_per_unit (same rate every period): verify budget conversion
- [ ] Test with date-varying cost_per_unit: verify each period uses its own rate
- [ ] Test with `budget_distribution_over_period` + `cost_per_unit` combined
- [ ] Test `optimize_budget()` with DataFrame input (parsing)
- [ ] Test `optimize_budget()` with DataArray input (pass-through)
- [ ] Test backward compatibility: `cost_per_unit=None` optimizes correctly (no conversion)

#### Manual Verification:
- [ ] Create MMM with impression data, fit it
- [ ] Run `optimize_budget(budget=100000, cost_per_unit=future_cpu_df)`
- [ ] Verify optimal_budgets are in dollars (not impressions)
- [ ] Check that budget / cost_per_unit matches expected impression volumes per period
- [ ] Compare optimization results with/without cost_per_unit on same data

---

## Testing Strategy

### Unit Tests

**New test file**: `tests/mmm/test_cost_per_unit.py`

Key test cases:

**`_parse_cost_per_unit_df()` unit tests** (in `tests/mmm/test_cost_per_unit.py`):

1. **test_parse_single_channel_no_custom_dims()**
   - DataFrame with `date` + one channel column
   - Verify output has dims `(date, channel)` with missing channel = 1.0

2. **test_parse_all_channels_no_custom_dims()**
   - DataFrame with `date` + all channel columns
   - Verify element-wise values match

3. **test_parse_single_channel_with_custom_dims()**
   - DataFrame with `date, geo, TV` (3 geos × 4 dates = 12 rows)
   - Verify output has dims `(date, geo, channel)`
   - Missing channel (Radio) filled with 1.0

4. **test_parse_all_channels_with_custom_dims()**
   - DataFrame with `date, geo, TV, Radio`
   - Verify all values correct across `(date, geo, channel)`

5. **test_parse_missing_date_column_raises()**
   - DataFrame without `date` → `ValueError`

6. **test_parse_missing_custom_dim_column_raises()**
   - DataFrame without required custom dim column → `ValueError`

7. **test_parse_unknown_channel_raises()**
   - DataFrame with column not in model channels → `ValueError`

8. **test_parse_no_channel_columns_raises()**
   - DataFrame with only index columns → `ValueError`

9. **test_parse_reindex_nan_raises()**
   - DataFrame with dates that don't match model dates (e.g. subset,
     timezone-aware vs naive, off-by-one) → `ValueError` with diagnostic
     message indicating which coordinates caused the NaN

**Wrapper / integration tests** (in `tests/mmm/test_cost_per_unit.py`):

10. **test_get_channel_spend_without_cost_per_unit()**
    - Backward compatibility: returns raw channel_data

11. **test_get_channel_spend_with_cost_per_unit()**
    - Verify conversion applied, opt-out with `apply_cost_per_unit=False`

12. **test_set_cost_per_unit_post_hoc()**
    - Fit model, call `mmm.set_cost_per_unit(df)`, verify cost_per_unit stored and ROAS updates

13. **test_set_cost_per_unit_overwrite_protection()**
    - Set once, attempt again without `overwrite=True` → `ValueError`

14. **test_roas_with_cost_per_unit()**
    - Verify ROAS = contribution / (channel_data × cost_per_unit)

15. **test_budget_optimizer_with_constant_cost_per_unit()**
    - Provide constant cost_per_unit (same rate every period)
    - Run optimization, verify budget conversion applied correctly

16. **test_budget_optimizer_with_date_varying_cost_per_unit()**
    - Provide date-varying cost_per_unit (different rate per period)
    - Verify each period converts using its own rate (no averaging)

17. **test_budget_optimizer_cost_per_unit_validation()**
    - Wrong dims → `ValueError`
    - Wrong date length → `ValueError`
    - Non-positive values → `ValueError`
    - None → no conversion (backward compatible)

18. **test_optimize_budget_cost_per_unit_dataframe_input()**
    - Pass `pd.DataFrame` to `optimize_budget()`, verify parsing
    - Missing channels default to 1.0

19. **test_budget_optimizer_cost_per_unit_with_distribution()**
    - Combine `budget_distribution_over_period` with `cost_per_unit`
    - Verify both are applied correctly (distribution first, then conversion)

**Incrementality integration tests** (in `tests/mmm/test_incrementality.py`):

20a. **test_incremental_contribution_uses_raw_data_with_cost_per_unit()**
    - Fit model with `cost_per_unit`, run `compute_incremental_contribution()`
    - Verify the baseline array fed into the model graph uses raw channel data
      (not cost_per_unit-converted spend)

20b. **test_roas_via_incrementality_with_cost_per_unit()**
    - Fit model with `cost_per_unit`, run `contribution_over_spend()`
    - Verify ROAS denominators use converted spend (via `_aggregate_channel_spend`)
    - Confirm ROAS values are sensible ($/$ not $/impression)

### Serialization Roundtrip Tests

**New test cases** (in `tests/mmm/test_cost_per_unit.py`):

These tests verify the claims made in the [Migration Notes](#for-model-serialization)
section ("Models saved with cost_per_unit will load correctly").

20. **test_save_load_with_cost_per_unit()**
    - Fit model with `cost_per_unit`, save to disk (e.g. `mmm.save("model.nc")`)
    - Load model (`MMM.load("model.nc")`)
    - Verify `cost_per_unit` is preserved in `loaded_mmm.idata.constant_data`
    - Verify `loaded_mmm.data.cost_per_unit` returns the same DataArray
    - Verify `loaded_mmm.data.get_channel_spend()` applies conversion correctly
    - Verify `loaded_mmm.summary.get_roas()` returns correct values

21. **test_load_old_model_without_cost_per_unit()**
    - Save a model **without** `cost_per_unit` to disk
    - Load it back, verify backward compatibility:
      - `loaded_mmm.data.cost_per_unit` returns `None`
      - `loaded_mmm.data.get_channel_spend()` returns raw channel_data
      - `loaded_mmm.summary.get_roas()` works (no conversion applied)
    - Then call `loaded_mmm.set_cost_per_unit(df)` to add it post-hoc
    - Verify conversion now works correctly

22. **test_save_load_set_cost_per_unit_posthoc()**
    - Fit model without `cost_per_unit`, save, load
    - Call `loaded_mmm.set_cost_per_unit(df)` on the loaded model
    - Save again, load again
    - Verify `cost_per_unit` is now preserved across the second roundtrip

### Integration Tests

**Update existing tests**: `tests/mmm/test_multidimensional.py`

Add fixture with cost_per_unit:
```python
@pytest.fixture
def mmm_with_cost_per_unit(toy_X, toy_y, dates):
    """MMM model with cost_per_unit conversion."""
    cpu_df = pd.DataFrame({
        "date": dates,
        "x1": [0.01] * len(dates),
        "x2": [0.02] * len(dates),
    })
    mmm = MMM(
        date_column="date_week",
        channel_columns=["x1", "x2"],
        adstock="geometric",
        saturation="logistic",
        cost_per_unit=cpu_df,
    )
    mmm.fit(toy_X, toy_y, chains=1, draws=100)
    return mmm
```

Test cases:
1. **test_fit_predict_with_cost_per_unit()**
   - Verify model fits successfully
   - Check predictions work

2. **test_optimize_budget_with_cost_per_unit()**
   - Run budget optimization
   - Verify results are sensible

### Performance Tests

Verify cost_per_unit doesn't significantly impact:
- Model fitting time (should be negligible)
- get_channel_spend() performance (broadcasting is fast)
- Budget optimization time (one extra division operation)

---

## Migration Notes

### For Existing Users

**No action required** - Models without cost_per_unit work exactly as before.

**To adopt cost_per_unit**:

1. **During Model Creation**:
   ```python
   # Old: channel data assumed to be in spend units
   mmm = MMM(channel_columns=["TV", "Radio"])
   mmm.fit(data_with_spend)

   # New: channel data in impressions, specify conversion
   cpu_df = pd.DataFrame({
       "date": dates,
       "TV": [0.01] * len(dates),
       "Radio": [0.02] * len(dates),
   })
   mmm = MMM(
       channel_columns=["TV", "Radio"],
       cost_per_unit=cpu_df,
   )
   mmm.fit(data_with_impressions)
   ```

2. **After Model Fitting** (Post-hoc via `mmm.set_cost_per_unit()`):
   ```python
   # Fit with impression data (forgot to add cost_per_unit)
   mmm.fit(data_with_impressions)

   # Realize ROAS is wrong, add conversion
   cpu_df = pd.DataFrame({
       "date": dates,
       "TV": [0.01] * len(dates),
   })
   mmm.set_cost_per_unit(cpu_df)

   # Now ROAS is correct (Radio defaults to 1.0)
   roas = mmm.summary.roas()
   ```

### For Model Serialization

**InferenceData Compatibility**:
- cost_per_unit is stored in `constant_data` group
- Models saved with cost_per_unit will load correctly
- Models saved without cost_per_unit work unchanged

**Loading Old Models**:
```python
# Load model saved before cost_per_unit feature
mmm_loaded = MMM.load("old_model.nc")

# Add cost_per_unit to loaded model
cpu_df = pd.DataFrame({"date": dates, "TV": [0.01] * len(dates)})
mmm_loaded.set_cost_per_unit(cpu_df)
```

---

## Performance Considerations

### Memory

- Storage: `n_dates * n_channels * n_custom_dim_product * 8 bytes`
  (e.g., 52 weeks × 5 channels × 3 geos = 6KB)
- Missing channels stored as 1.0 (not sparse), but still negligible
- Overall impact: minimal

### Computation

- get_channel_spend(): one multiplication (xarray broadcasting)
- Budget optimizer: one division + potential mean aggregation
- Expected overhead: <1% of total runtime

### Broadcasting Efficiency

xarray broadcasting is highly optimized:
- Element-wise multiplication (vectorized)
- Handles `(date, channel)` and `(date, channel, *custom_dims)` uniformly
- No loops or manual broadcasting needed

---

## Open Questions (Resolved)

### 1. ✅ Input Format Support
**Question**: Which input formats to support?
**Answer**: DataFrame only.  Wide-format with rows = `(date, *custom_dims)`,
columns = channel names.  Missing channels default to 1.0.  Single format
eliminates multi-path complexity and naturally supports all dimensionality
combinations (scalar, time-varying, per-channel, per-custom-dim).

### 2. ✅ Date-dim Handling for Optimization
**Question**: How to handle cost_per_unit's date dimension for optimization?
**Answer**: User provides forward-looking cost_per_unit with a date dimension
matching the optimization window (`num_periods`).  Each period uses its own
rate — no averaging.  This preserves date-varying effects (seasonal pricing,
contract changes) and decouples Phase 2 from Phase 1's historical data.

### 3. ✅ Setter Method Location
**Question**: Where to expose `set_cost_per_unit()`?
**Answer**: Only `mmm.set_cost_per_unit()` on the MMM class.  No wrapper-level
setter needed — the MMM class owns all cost_per_unit logic (parsing,
fit-time injection, post-hoc setter).

### 4. ✅ Default Behavior
**Question**: Should `get_channel_spend()` apply conversion by default?
**Answer**: Yes, with opt-out parameter `apply_cost_per_unit=False` for flexibility

---

## Implementation Order

**Phase 1** (Core Infrastructure):
1. Update schema (1.1)
2. Add model parameter (1.2)
3. Shared parsing utility + post-fit injection (1.3)
4. Modify get_channel_spend (1.4)
5. Add cost_per_unit property on wrapper (1.5)
6. Add MMM setter (1.6)
7. Write Phase 1 tests

**Phase 2** (Budget Optimization):
1. Add `cost_per_unit` parameter + validation to `BudgetOptimizer` (2.1)
2. Update `_replace_channel_data_by_optimization_variable` (2.2)
3. Add `cost_per_unit` to `optimize_budget()` with DataFrame parsing (2.3)
4. Document units (2.4, 2.5)
5. Write Phase 2 tests

**Phase dependency**: Phase 2 depends on Phase 1's static
`_parse_cost_per_unit_df()` method (section 1.3a) for DataFrame parsing.
Phase 1 should be implemented first (or at minimum, the static method
must exist before Phase 2's wrapper parsing can work).  Phase 1 stores
*historical* cost_per_unit in InferenceData for reporting; Phase 2 accepts
*forward-looking* cost_per_unit as an explicit optimizer parameter.

---

## References

### Original Research
- Research document: [thoughts/shared/issues/2210/research.md](https://github.com/pymc-labs/pymc-marketing/blob/work-issue-2210/thoughts/shared/issues/2210/research.md)
- Issue: [pymc-labs/pymc-marketing#2210](https://github.com/pymc-labs/pymc-marketing/issues/2210)

### Key Files Modified

**Phase 1** (reporting/historical):
- [pymc_marketing/data/idata/schema.py](pymc_marketing/data/idata/schema.py) (schema update)
- [pymc_marketing/data/idata/mmm_wrapper.py](pymc_marketing/data/idata/mmm_wrapper.py) (cost_per_unit property + get_channel_spend conversion)
- [pymc_marketing/mmm/multidimensional.py](pymc_marketing/mmm/multidimensional.py) (init parameter, `_parse_cost_per_unit_df`, `set_cost_per_unit`, fit-time injection)
- [pymc_marketing/mmm/incrementality.py](pymc_marketing/mmm/incrementality.py) (line 377: opt out of cost_per_unit conversion for model-graph input via `apply_cost_per_unit=False`)

**Phase 2** (optimization/forward-looking):
- [pymc_marketing/mmm/budget_optimizer.py](pymc_marketing/mmm/budget_optimizer.py) (`cost_per_unit` Field, validation, conversion in graph)
- [pymc_marketing/mmm/multidimensional.py](pymc_marketing/mmm/multidimensional.py) (`optimize_budget` parameter, `_parse_cost_per_unit_for_optimizer`)

### Test Files
- `tests/mmm/test_cost_per_unit.py` (new — parsing, MMM setter, integration tests)
- `tests/mmm/test_idata_wrapper.py` (updated — get_channel_spend with cost_per_unit)
- `tests/mmm/test_budget_optimizer.py` (updated)
- `tests/data/test_idata_schema.py` (updated)
- `tests/mmm/test_multidimensional.py` (updated)
- `tests/mmm/test_incrementality.py` (updated — verify raw data usage with cost_per_unit)

---

## Changelog

- **2026-02-20 (k)**: Fix test cases 3 & 4 dim order (review Issue 7)
  - Testing Strategy: Changed dims from `(date, channel, geo)` to
    `(date, geo, channel)` in test cases 3 and 4, matching the plan's
    channel-last convention `(date, *custom_dims, channel)`.
- **2026-02-20 (j)**: Rename MMM-level `self.cost_per_unit` to `self._cost_per_unit_input`
  - Problem: The plan had both `mmm.cost_per_unit` (a `pd.DataFrame` on
    the MMM instance, set at init) and `mmm.data.cost_per_unit` (an
    `xr.DataArray` property on the wrapper, reading from idata).  Both
    are needed (they serve different lifecycle stages — pre-fit vs
    post-fit), but sharing the same public name caused confusion about
    which is the canonical accessor.
  - Fix: Renamed the MMM-level attribute from `self.cost_per_unit` to
    `self._cost_per_unit_input` (private).  The `__init__` parameter
    remains `cost_per_unit` for user-friendliness, but is stored
    internally with a private name.  The public read API is
    `mmm.data.cost_per_unit` (the wrapper property from section 1.5).
  - Updated sections: 1.2 (storage + `__eq__`), 1.3b (`fit()` override),
    1.5 (rationale clarifying public API role).
  - Added comparison table in section 1.2 documenting the two attributes,
    their types, availability, purpose, and visibility.
- **2026-02-20 (i)**: Update `__eq__` for `cost_per_unit` (review Issue 5)
  - Section 1.2: Added `cost_per_unit` comparison to `MMM.__eq__` (lines
    590–707).  Without this, two models with different `cost_per_unit`
    DataFrames would incorrectly compare as equal.  Uses `DataFrame.equals()`
    for comparison, with a `None`-vs-non-`None` guard.
- **2026-02-20 (h)**: Fix `fit()` override signature mismatch (review Issue 3)
  - Section 1.3b: Replaced minimal `fit(self, X, y, **kwargs)` with the full
    parent signature from `RegressionModelBuilder.fit` (model_builder.py:960):
    `fit(self, X, y=None, progressbar=None, random_seed=None, **kwargs)`.
    The minimal version would incorrectly bind `progressbar` or `random_seed`
    when passed positionally.  Forward all named parameters explicitly to
    `super().fit()`.
- **2026-02-20 (g)**: Fix critical `get_channel_spend()` default breaking `Incrementality` (review Issue 1, Option B)
  - Problem: Section 1.4 proposes `get_channel_spend(apply_cost_per_unit=True)` as default.
    The `Incrementality` module calls `get_channel_spend()` at line 377 to obtain raw
    channel data for the compiled PyTensor model graph.  With the default conversion,
    this would silently feed dollar-converted values into saturation curves calibrated
    for raw units (impressions, clicks), producing incorrect counterfactual results.
  - Fix (Option B): Keep `apply_cost_per_unit=True` as default (correct for ROAS/reporting
    callers), but update `incrementality.py` line 377 to explicitly pass
    `apply_cost_per_unit=False`.  This is the minimal fix — only the one caller that
    needs raw data opts out.
  - Added `incrementality.py` to Key Files Modified (Phase 1) in References section.
  - Added incrementality integration test cases (20a, 20b) to Testing Strategy:
    verify raw data usage in `compute_incremental_contribution()` and correct ROAS
    via `contribution_over_spend()`.
  - Added `tests/mmm/test_incrementality.py` to Test Files in References.
- **2026-02-19 (f)**: Fix redundant hasattr, misleading section title, and document static/instance split
  - Problem 10: Removed redundant `hasattr(self.idata, "constant_data")`
    check in `set_cost_per_unit()` (section 1.6).  Line 705 already
    validates this and raises `ValueError` if missing, so the second
    check on line 710 was guaranteed to be true — dead code that added
    noise.  Simplified to a direct `"cost_per_unit" in self.idata.constant_data`
    check.
  - Problem 11: Renamed "Init-time injection" to "Fit-time injection" in
    section 1.3b title and all 6 references throughout the document.  The
    `cost_per_unit` DataFrame is stored at `__init__` time, but the
    xarray injection into `idata.constant_data` happens during `fit()`
    (after `super().fit()` returns).  The old title was misleading about
    *when* the injection actually executes.
  - Problem 12: Added explicit "Design note — static/instance split"
    to section 1.3a documenting *why* `_parse_cost_per_unit_df()` is a
    `@staticmethod` with explicit parameters while
    `_build_cost_per_unit_array()` is an instance method wrapper.  The
    core refactoring was already done in changelog (d) (problem 4), but
    the design rationale (testability in isolation, reusability across
    Phase 1 and Phase 2 with different coordinates) was not documented.
- **2026-02-19 (e)**: Fix reindex NaN, broadcasting assumption, and serialization tests
  - Problem 6: Added explicit NaN validation after `reindex()` in
    `_parse_cost_per_unit_df()`.  Previously, if a user's dates didn't
    exactly match the model's dates (timezone mismatch, subset of dates,
    off-by-one), `reindex` would silently fill positions with NaN,
    leading to silent corruption downstream.  Now raises `ValueError`
    with diagnostic info showing which coordinates caused the mismatch.
    Added corresponding test case `test_parse_reindex_nan_raises()`.
  - Problem 8: Added explicit documentation of the `date_dim_idx == 0`
    broadcasting assumption in section 2.2.  The division
    `repeated_budgets / _cost_per_unit_tensor` relies on both tensors
    having date as their first dimension.  This is guaranteed by
    `channel_data_dims = ("date", *custom_dims, "channel")`, but was
    previously implicit.  Added an assertion guard and documented the
    assumption so future refactorings don't silently break it.
  - Problem 9: Added serialization roundtrip tests (tests 20-22) to the
    Testing Strategy.  The Migration Notes claimed "Models saved with
    cost_per_unit will load correctly" but there were no tests verifying
    this.  New tests cover: save/load with cost_per_unit preserved,
    load old model without cost_per_unit (backward compat), and
    post-hoc set → save → load roundtrip.
- **2026-02-19 (d)**: Fix DRY violation and add positive-value validation
  - Problem 4: Refactored `_parse_cost_per_unit_df()` from instance method
    to **static method** with explicit coordinate parameters (`channels`,
    `dates`, `custom_dims`, `custom_dim_coords`).  Phase 2's
    `_parse_cost_per_unit_for_optimizer()` now delegates to this shared
    static method instead of duplicating the parsing logic.  One parser,
    two call sites — bug fixes apply to both phases automatically.
  - Added `_build_cost_per_unit_array()` convenience wrapper on MMM that
    extracts coordinates from the fitted model and delegates to the static
    method (used by 1.3b `fit()` override and 1.6 `set_cost_per_unit()`).
  - Problem 5: Added positive-value validation to `_parse_cost_per_unit_df()`.
    Phase 2 already rejected non-positive values in
    `_validate_and_process_cost_per_unit()`, but Phase 1's parser had no
    validation — a user could set `cost_per_unit` to 0 or negative via
    `mmm.set_cost_per_unit()`, causing silent division-by-zero in
    `get_roas()`.  Now both phases validate consistently via the shared
    parser.
- **2026-02-19 (c)**: Concretized fit-time injection hook (section 1.3b)
  - Replaced vague "end of `fit()`, after `pm.sample()`" with explicit
    "Option A: Override `fit()` in MMM" — call `super().fit()`, then inject
  - Documented the full `fit()` flow from the base class (8 steps) and why
    injection must happen after `super().fit()` returns
  - Added concrete `fit()` override implementation
  - Documented alternatives considered and rejected (Options B and C)
- **2026-02-19 (b)**: Review fixes — correct class target and dim ordering
  - Section 1.2: Retargeted from `MMMModelBuilder` in `base.py` to `MMM` in
    `multidimensional.py` (`MMM` inherits from `RegressionModelBuilder`, not
    `MMMModelBuilder` — they are sibling classes)
  - All dim orderings changed from `("date", "channel", *custom_dims)` to
    `("date", *custom_dims, "channel")` to match `channel_data`'s convention
    (channel last): sections 1.1, 1.3a, 1.5, 2.3
  - Removed `base.py` from Key Files Modified (Phase 1)
- **2026-02-19 (a)**: Redesigned Phase 2 — user-provided forward-looking cost_per_unit
  - Phase 2 no longer reads historical cost_per_unit from the model's InferenceData
  - User provides future cost_per_unit as a parameter to `optimize_budget()` / `BudgetOptimizer`
  - Date-varying rates: each optimization period uses its own cost_per_unit (no averaging)
  - Conversion applied AFTER time distribution (budget has date dim), BEFORE channel scaling
  - Phases 1 and 2 are now fully independent (can be implemented in any order)
  - Removed `_get_cost_per_unit()` helper and `data` property on Protocol (no longer needed)
  - Added `_validate_and_process_cost_per_unit()` and `_parse_cost_per_unit_for_optimizer()`
  - Updated success criteria, tests, and documentation sections
- **2026-02-18 (b)**: Simplified setter design — single `mmm.set_cost_per_unit()`, no separate file
  - Removed `mmm.data.set_cost_per_unit()` — only `mmm.set_cost_per_unit()` needed
  - Moved `_parse_cost_per_unit_df()` from `utils.py` to a `@staticmethod` on the MMM class
  - All cost_per_unit logic (parsing, fit-time injection, post-hoc setter) lives in MMM class
  - Wrapper retains read-only `cost_per_unit` property (1.5) for `get_channel_spend()` conversion
- **2026-02-18 (a)**: Major revision — DataFrame-only input, shared parsing, no duplication
  - Replaced 5 input formats with single `pd.DataFrame` (wide-format)
  - DataFrame rows = `(date, *custom_dims)`, columns = channel names
  - Missing channels default to 1.0 (not all channels need cost_per_unit)
  - Extracted shared `_parse_cost_per_unit_df()` utility used by both
    fit-time injection (1.3b) and post-hoc setter (1.6)
  - Replaced `pm.Data` with direct xarray injection into `idata.constant_data`
  - `cost_per_unit` is pure metadata, never used in the model graph
- **2026-02-16**: Initial plan created based on research document
  - Focused on Phase 1 (Core Infrastructure) and Phase 2 (Budget Optimization)
  - Added support for 5 input formats including numpy arrays
  - Included `mmm.set_cost_per_unit()` convenience method
  - Documented time-varying aggregation strategy for budget optimization
