# MMMSummaryData Design Document

## Overview

This document describes the design of the `MMMSummaryData` class, which provides lazy-evaluated access to pre-processed Polars DataFrames for Media Mix Model (MMM) visualization. The class serves as a data container for the `MMMStakeholderPlotSuite` plotting methods, computing DataFrames on-demand only when accessed.

## Design Goals

1. **Lazy Evaluation**: DataFrames are computed only when accessed, improving performance for large models when only specific plots are needed
2. **Class-Based API**: Replace the current dataclass with a proper class that provides computed properties
3. **Caching**: Once computed, DataFrames are cached to avoid redundant computation
4. **Flexible Input**: Support both `xr.DataTree` and `az.InferenceData` inputs
5. **Optional Model Dependency**: Some dataframes (saturation curves, marginal curves, decay curves) require the `MMM` model instance, which should be optional

---

## Class Structure

### Class: `MMMSummaryData`

A class that provides lazy-evaluated access to MMM plotting dataframes.

**Initialization:**

```python
class MMMSummaryData:
    def __init__(
        self,
        idata: xr.DataTree | az.InferenceData,
        mmm: MMM | None = None,
        hdi_probs: list[float] | None = None,
        target_var_posterior: str = "y_original_scale",
        target_var_observed: str = "y",
    ):
        """Initialize the MMMSummaryData container.

        Parameters
        ----------
        idata : xr.DataTree | az.InferenceData
            Inference data containing posterior, posterior_predictive, fit_data, and constant_data.
            If az.InferenceData is provided, it will be converted to xr.DataTree.
        mmm : MMM | None, optional
            The MMM model instance. Required for saturation_curves, marginal_curves, and decay_curves.
            If None, these properties will raise an error when accessed.
        hdi_probs : list[float] | None, optional
            List of HDI probability levels to compute. Defaults to [0.80, 0.90, 0.95].
        target_var_posterior : str, optional
            Variable name for posterior predictive target. Defaults to "y_original_scale".
        target_var_observed : str, optional
            Variable name for observed target. Defaults to "y".
        """
```

**Class Attributes:**

- `_idata: xr.DataTree` - Internal inference data tree (converted from az.InferenceData if needed)
- `_mmm: MMM | None` - Optional MMM model instance
- `_hdi_probs: list[float]` - List of HDI probability levels
- `_target_var_posterior: str` - Variable name for posterior predictive
- `_target_var_observed: str` - Variable name for observed target
- `_cache: dict[str, pl.DataFrame]` - Internal cache for computed DataFrames

---

## Lazy-Evaluated Properties

All DataFrames are accessed via properties that compute them on-demand and cache the results.

### Core DataFrames (Always Available)

#### 1. `posterior_predictive`

Posterior predictive distributions over time with observed values.

**Property:**

```python
@property
def posterior_predictive(self) -> pl.DataFrame:
    """Posterior predictive DataFrame.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: `date`, `mean`, `observed`, `median`,
        and HDI columns (`abs_error_{prob}_lower`, `abs_error_{prob}_upper`).
    """
```

**Computation:**
- Extracts `posterior_predictive[target_var_posterior]` from `idata`
- Computes mean, median, and HDI intervals for all `hdi_probs`
- Merges with observed values from `fit_data[target_var_observed]`
- Returns empty DataFrame if `date` dimension is not present

**Required Data:**
- `idata.posterior_predictive[target_var_posterior]` (with `date` dimension)
- `idata.fit_data[target_var_observed]` (with `date` dimension)

---

#### 2. `roas`

Return on Ad Spend (ROAS) by channel and optionally by date.

**Property:**

```python
@property
def roas(self) -> pl.DataFrame:
    """ROAS DataFrame.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: `channel`, `mean`, `median`,
        optionally `date` (if time series), and HDI columns.
    """
```

**Computation:**
- Extracts `posterior.channel_contribution_original_scale` and `constant_data.channel_data`
- Computes ROAS as `contribution / spend` (handling zero spend)
- Computes mean, median, and HDI intervals for all `hdi_probs`
- May include `date` dimension if contribution has time dimension

**Required Data:**
- `idata.posterior.channel_contribution_original_scale`
- `idata.constant_data.channel_data`

---

#### 3. `roas_change_over_time`

Period-over-period ROAS changes.

**Property:**

```python
@property
def roas_change_over_time(self) -> pl.DataFrame:
    """Period-over-period ROAS changes DataFrame.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: `date`, `channel`, `mean`, `median`,
        and HDI columns. Empty DataFrame if no date dimension.
    """
```

**Computation:**
- Computes period-over-period changes: `(current - previous) / previous * 100`
- Handles infinite values (from division by zero) by converting to NaN
- Computes mean, median, and HDI intervals
- Returns empty DataFrame if ROAS has no `date` dimension

**Required Data:**
- `roas` property (computed internally)

---

#### 4. `contribution`

Channel contribution by channel and optionally by date.

**Property:**

```python
@property
def contribution(self) -> pl.DataFrame:
    """Channel contribution DataFrame.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: `channel`, `mean`, `median`,
        optionally `date` (if time series), and HDI columns.
    """
```

**Computation:**
- Extracts `posterior.channel_contribution_original_scale`
- Computes mean, median, and HDI intervals for all `hdi_probs`
- May include `date` dimension if contribution has time dimension

**Required Data:**
- `idata.posterior.channel_contribution_original_scale`

---

#### 5. `channel_spend`

Marketing spend by channel and date.

**Property:**

```python
@property
def channel_spend(self) -> pl.DataFrame:
    """Channel spend DataFrame.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: `date`, `channel`, `channel_data`.
    """
```

**Computation:**
- Extracts `constant_data.channel_data`
- Converts to Polars DataFrame with index reset

**Required Data:**
- `idata.constant_data.channel_data`

---

### Optional DataFrames (Require MMM Model)

#### 6. `saturation_curves`

Saturation curves for marketing channels.

**Property:**

```python
@property
def saturation_curves(self) -> pl.DataFrame:
    """Saturation curves DataFrame.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: `x`, `channel`, `mean`, `median`,
        and HDI columns.

    Raises
    ------
    ValueError
        If `mmm` model instance is not provided.
    """
```

**Computation:**
- Uses `mmm.saturation.sample_curve()` to generate saturation curves
- Scales curves by `channel_scale` and `target_scale` from `constant_data`
- Computes mean, median, and HDI intervals
- Returns DataFrame with `x` (spend), `channel`, and statistics

**Required Data:**
- `mmm` model instance (required)
- `idata.posterior.saturation_beta` and `idata.posterior.saturation_lam`
- `idata.constant_data.channel_scale` and `idata.constant_data.target_scale`

---

#### 7. `marginal_curves`

Marginal curves for marketing channels.

**Property:**

```python
@property
def marginal_curves(self) -> pl.DataFrame:
    """Marginal curves DataFrame.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: `x`, `channel`, `mean`, `median`,
        and HDI columns.

    Raises
    ------
    ValueError
        If `mmm` model instance is not provided.
    """
```

**Computation:**
- Computes marginal curves as the derivative of saturation curves
- Uses `saturation_curves.diff(dim='x')` to compute derivatives
- Computes mean, median, and HDI intervals
- Returns DataFrame with `x` (spend), `channel`, and statistics

**Required Data:**
- `saturation_curves` property (computed internally)
- `mmm` model instance (required)

---

#### 8. `decay_curves`

Adstock decay curves over time for each channel.

**Property:**

```python
@property
def decay_curves(self) -> pl.DataFrame:
    """Adstock decay curves DataFrame.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: `time`, `channel`, `mean`,
        and optionally HDI columns.

    Raises
    ------
    ValueError
        If `mmm` model instance is not provided.
    """
```

**Computation:**
- Uses `mmm.adstock.function()` to compute decay curves
- Creates unit impulse input (spend[0] = 1, rest = 0)
- Evaluates adstock function for each channel's alpha parameter
- Computes mean and optionally HDI intervals across posterior samples
- Returns DataFrame with `time`, `channel`, and statistics

**Required Data:**
- `mmm` model instance (required)
- `idata.posterior.adstock_alpha`
- `mmm.adstock.l_max` and `mmm.adstock.function`

---

## Metadata Properties

### `channels`

List of channel names.

**Property:**

```python
@property
def channels(self) -> list[str]:
    """List of channel names.

    Returns
    -------
    list[str]
        List of channel names from constant_data.
    """
```

**Computation:**
- Extracts channel names from `constant_data.channel_data.channel.values`

---

### `hdi_probs`

List of HDI probability levels.

**Property:**

```python
@property
def hdi_probs(self) -> list[float]:
    """List of HDI probability levels.

    Returns
    -------
    list[float]
        List of HDI probability levels used for computation.
    """
```

---

## Implementation Details

### Lazy Evaluation Pattern

Each DataFrame property follows this pattern:

```python
@property
def property_name(self) -> pl.DataFrame:
    """Property docstring."""
    cache_key = "property_name"

    # Check cache first
    if cache_key in self._cache:
        return self._cache[cache_key]

    # Compute DataFrame
    df = self._compute_property_name()

    # Cache and return
    self._cache[cache_key] = df
    return df
```

### Caching Strategy

- **Cache Storage**: Internal `_cache` dictionary stores computed DataFrames
- **Cache Key**: Property name (e.g., `"posterior_predictive"`, `"roas"`)
- **Cache Invalidation**: Cache persists for the lifetime of the instance
- **Memory Management**: Users can manually clear cache via `clear_cache()` method if needed

### Data Conversion

- **Input Conversion**: `az.InferenceData` is automatically converted to `xr.DataTree` using `az.to_datatree()`
- **Index Handling**: Date index is preserved and renamed to `date` coordinate if needed
- **Output Format**: All DataFrames are Polars DataFrames with consistent column naming

### Error Handling

- **Missing Data**: Properties return empty DataFrames if required data is missing (e.g., no `date` dimension)
- **Missing Model**: Optional properties (`saturation_curves`, `marginal_curves`, `decay_curves`) raise `ValueError` if `mmm` is `None`
- **Missing Variables**: Raises appropriate errors if required variables are missing from `idata`

---

## Public Utility Methods

### `clear_cache()`

Clear the internal cache of computed DataFrames.

**Signature:**

```python
def clear_cache(self) -> None:
    """Clear the internal cache of computed DataFrames.

    This forces recomputation of all DataFrames on next access.
    """
```

---

## Integration with MMMStakeholderPlotSuite

The `MMMSummaryData` class is designed to work seamlessly with `MMMStakeholderPlotSuite`:

```python
from pymc_marketing.mmm.plot import MMMStakeholderPlotSuite
from pymc_marketing.mmm.plot import MMMSummaryData

# Create plot data container
plot_data = MMMSummaryData(
    idata=idata,
    mmm=mmm,  # Optional, needed for curves
    hdi_probs=[0.80, 0.90, 0.95]
)

# Initialize plotting suite with plot data
plot_suite = MMMStakeholderPlotSuite(plot_data=plot_data)

# Access dataframes lazily - only computed when needed
fig1 = plot_suite.posterior_predictive(hdi_prob=0.90)  # Computes posterior_predictive
fig2 = plot_suite.roas(hdi_prob=0.90)  # Computes roas
fig3 = plot_suite.saturation_curves(hdi_prob=0.90)  # Computes saturation_curves (requires mmm)
```

---

## Usage Examples

### Basic Usage

```python
from pymc_marketing.mmm.plot import MMMSummaryData

# Initialize with inference data
plot_data = MMMSummaryData(
    idata=idata,
    hdi_probs=[0.80, 0.90, 0.95]
)

# Access dataframes (computed on-demand)
posterior_df = plot_data.posterior_predictive  # Computed now
roas_df = plot_data.roas  # Computed now
contribution_df = plot_data.contribution  # Computed now

# Access again - uses cache
posterior_df_cached = plot_data.posterior_predictive  # From cache
```

### With MMM Model (for curves)

```python
from pymc_marketing.mmm.plot import MMMSummaryData

# Initialize with model instance
plot_data = MMMSummaryData(
    idata=idata,
    mmm=mmm,
    hdi_probs=[0.80, 0.90, 0.95]
)

# Access curve dataframes
saturation_df = plot_data.saturation_curves  # Computed now
marginal_df = plot_data.marginal_curves  # Computed now (uses saturation_curves cache)
decay_df = plot_data.decay_curves  # Computed now
```

### Without MMM Model (curves unavailable)

```python
from pymc_marketing.mmm.plot import MMMSummaryData

# Initialize without model
plot_data = MMMSummaryData(
    idata=idata,
    hdi_probs=[0.80, 0.90, 0.95]
)

# Core dataframes work fine
posterior_df = plot_data.posterior_predictive
roas_df = plot_data.roas

# Curves will raise error
try:
    saturation_df = plot_data.saturation_curves  # Raises ValueError
except ValueError as e:
    print(f"Error: {e}")  # "MMM model instance required for saturation_curves"
```

### Cache Management

```python
# Clear cache to force recomputation
plot_data.clear_cache()

# Or access specific property to recompute just that one
del plot_data._cache["roas"]  # Force recompute roas on next access
roas_df = plot_data.roas  # Recomputed now
```

---

## Migration from Current Implementation

### Current Usage (Eager Evaluation)

```python
from pymc_marketing.sandbox.xarray_processing_utils import process_idata_for_plotting

# All dataframes computed immediately
plot_data = process_idata_for_plotting(idata, hdi_probs=[0.80, 0.90, 0.95])

# Access pre-computed dataframes
posterior_df = plot_data.posterior_predictive
roas_df = plot_data.roas
```

### New Usage (Lazy Evaluation)

```python
from pymc_marketing.mmm.plot import MMMSummaryData

# No computation on initialization
plot_data = MMMSummaryData(idata, hdi_probs=[0.80, 0.90, 0.95])

# Dataframes computed on first access
posterior_df = plot_data.posterior_predictive  # Computed now
roas_df = plot_data.roas  # Computed now
```


---

## Future Enhancements

### Potential Additions

1. **Selective HDI Computation**: Only compute HDI intervals for requested probability levels
2. **Lazy HDI Computation**: Compute HDI intervals on-demand per probability level
3. **Filtering/Subsetting**: Methods to filter dataframes by date range or channels before computation
4. **Validation**: Validate `idata` structure on initialization to catch errors early
5. **Serialization**: Support for saving/loading computed DataFrames to disk

---

## Testing Considerations

### Unit Tests

- Test each property computes correct DataFrame structure
- Test caching behavior (first access computes, second access uses cache)
- Test error handling (missing model, missing variables)
- Test empty DataFrame returns for missing dimensions

### Integration Tests

- Test with `MMMStakeholderPlotSuite` to ensure compatibility
- Test with real `idata` from MMM models
- Test performance with large datasets

### Edge Cases

- Empty `idata` (no posterior samples)
- Missing optional groups (e.g., no `posterior_predictive`)
- Single channel models
- Single time period models
- Models without date dimension

---

## Open Questions

### 1. Target Variable Arguments (`target_var_posterior` and `target_var_observed`)

**Question**: Do we need `target_var_posterior` and `target_var_observed` as initialization arguments, or is there a better way to extract these from the MMM model or `idata`?

**Current Approach:**
- `target_var_posterior` defaults to `"y_original_scale"` (deterministic variable in model)
- `target_var_observed` defaults to `"y"` (observed variable)

**Considerations:**
- The MMM model has an `output_var` property that returns `"y"`, which could be used for `target_var_observed`
- `y_original_scale` is a deterministic variable created during model building, but may not always exist:
  - It's created in `_add_original_scale_deterministics()` method
  - May not exist if the model wasn't built with original scale transformations
  - Could potentially be detected by checking if it exists in `idata.posterior_predictive`
- For custom models or models without original scale variables, users might need different variable names

**Potential Solutions:**
1. **Auto-detect from `idata`**: Check `idata.posterior_predictive` for available variables and use heuristics (e.g., look for `y_original_scale`, fallback to `y` or `output_var`)
2. **Extract from MMM model**: If `mmm` is provided, use `mmm.output_var` for observed and check for `y_original_scale` in posterior_predictive
3. **Keep as arguments but improve defaults**: Keep the arguments for flexibility, but improve default detection logic
4. **Hybrid approach**: Auto-detect when possible, allow override via arguments

**Recommendation**: Investigate whether we can reliably auto-detect these variables from `idata` structure, potentially using the MMM model's `output_var` property when available. This would reduce the API surface while maintaining flexibility through optional arguments.
