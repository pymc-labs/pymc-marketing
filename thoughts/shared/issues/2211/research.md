---
date: 2026-01-30T20:59:08+00:00
researcher: Claude Sonnet 4.5
git_commit: 3aa9b3b6180a23d4880c3d3e3041c85757a90806
branch: work-issue-2211
repository: pymc-labs/pymc-marketing
topic: "Compute ROAS the correct way (like in the Google paper)"
tags: [research, codebase, roas, adstock, carryover, mmm, counterfactual, multidimensional, vectorization, hierarchical-dimensions, xarray]
status: complete
last_updated: 2026-01-30
last_updated_by: Claude Sonnet 4.5
issue_number: 2211
---

# Research: Compute ROAS the Correct Way (Google MMM Paper Formula 10)

**Date**: 2026-01-30T20:59:08+00:00
**Researcher**: Claude Sonnet 4.5
**Git Commit**: 3aa9b3b6180a23d4880c3d3e3041c85757a90806
**Branch**: work-issue-2211
**Repository**: pymc-labs/pymc-marketing
**Issue**: #2211

## Research Question

How is ROAS currently computed in the codebase, and what changes are needed to implement formula (10) from the Google MMM paper using a **counterfactual approach** that accounts for carryover effects? Specifically, the implementation must:
1. Work with MultidimensionalMMM class and hierarchical dimensions (geo, country, region, etc.)
2. Use xarray operations on `fit_data` instead of pandas
3. Handle arbitrary time frequencies
4. Process all channels simultaneously with vectorized operations
5. Support separate flags for carry-in and carry-out effects
6. Output shape: `(chain, draw, date, channel, *custom_dims)`
7. Use period-end dates for labeling

## Summary

### Google Paper Formula (10) - Counterfactual Approach

```
ROAS_m =
  Σ(t=t0 to t1+L-1) [Y_hat_t^m(x_{t-L+1,m}, ..., x_{t,m}; Ω) - Y_hat_t^m(x̃_{t-L+1,m}, ..., x̃_{t,m}; Ω)]
  ────────────────────────────────────────────────────────────────────────────────────────────────
  Σ(t=t0 to t1) x_{t,m}
```

Where:
- `Y_hat_t^m(...)` = Predicted KPI at time t as a function of media inputs and model parameters Ω
- `x_{t,m}` = Actual observed spend for channel m at time t
- `x̃_{t,m}` = Counterfactual spend (zero for channel m during [t0, t1], actual otherwise)
- `t0, t1` = Start and end of the evaluation period
- `L` = Carryover window length (l_max in code)
- `Ω` = All Bayesian model parameters

### Key Implementation Requirements

1. **Target Class**: `MultidimensionalMMM` in `pymc_marketing/mmm/multidimensional.py`

2. **Data Access**: Use `idata.fit_data` xarray Dataset which preserves hierarchical structure:
   ```python
   # For model with dims=("country",):
   fit_data structure:
     Dimensions: (date: n_dates, country: n_countries)
     Data variables: channel_1, channel_2, ..., target
   ```

3. **Hierarchical Dimensions**: Models can have custom dims (geo, country, region, etc.):
   - Each date×channel combination has separate values for each dimension value
   - ROAS must be computed for each (period, channel, *dim) combination
   - Use xarray broadcasting and operations to handle all dimensions simultaneously

4. **Vectorized Processing**: Compute counterfactuals for ALL channels simultaneously using xarray operations

5. **Frequency-Agnostic**: Use `pd.infer_freq()` - never assume weekly

6. **Carryover Flags**:
   - `include_carryin` (default True): Include pre-period channel contribution impact
   - `include_carryout` (default True): Include post-period channel contribution impact

7. **Output Shape**: `(chain, draw, date, channel, *custom_dims)` where:
   - `date` represents periods (but dimension is still called "date")
   - Order doesn't matter (xarray is dimension-aware)

8. **Period Labeling**: Use **period-end dates** following codebase convention
   - Example: Monthly period "2024-01-01 to 2024-01-31" → labeled as `2024-01-31`

9. **No Backwards Compatibility**: Remove old ROAS behavior entirely

10. **Required Frequency Parameter**: No default - force users to think about computational cost

## Problem Statement

### Current Implementation (Incorrect)
**File**: `pymc_marketing/data/idata/mmm_wrapper.py:315-342`

```python
ROAS[t] = contribution[t] / spend[t]
```

**Problems**:
- Element-wise division ignoring carryover effects
- Treats each period independently
- Does not use counterfactual approach
- Does not properly handle hierarchical dimensions

### Required Implementation (Counterfactual with Hierarchical Dimensions)

```python
# For all channels and all dimension combinations over period [t0, t1]:

# 1. Extract data from fit_data (xarray Dataset with hierarchical dims)
fit_dataset = idata.fit_data  # Shape: (date, *custom_dims) for each channel

# 2. Predict KPI with actual spend on all channels
Y_actual = model.sample_posterior_predictive(fit_dataset_actual)
# Shape: (chain, draw, date, *custom_dims)

# 3. Create counterfactual: zero spend on all channels during [t0, t1]
fit_dataset_counter = fit_dataset.copy()
fit_dataset_counter[channel_vars].loc[dict(date=slice(t0, t1))] = 0

# 4. Predict KPI with counterfactual (zero) spend
Y_counterfactual = model.sample_posterior_predictive(fit_dataset_counter)
# Shape: (chain, draw, date, *custom_dims)

# 5. Compute incremental contribution with xarray operations
incremental = Y_actual - Y_counterfactual
# Shape: (chain, draw, date, *custom_dims)

# 6. Sum over evaluation window and divide by spend
# xarray broadcasting handles all dimensions automatically
total_incremental = incremental.sum(dim='date')  # (chain, draw, *custom_dims)
total_spend = fit_dataset[channel_vars].sum(dim='date')  # (*custom_dims,)
ROAS = total_incremental / total_spend  # (chain, draw, channel, *custom_dims)
```

## Detailed Findings

### 1. MultidimensionalMMM Hierarchical Dimensions

**File**: `pymc_marketing/mmm/multidimensional.py:213-3000+`

#### Hierarchical Dimension Support

The MultidimensionalMMM class supports arbitrary custom dimensions via the `dims` parameter:

```python
mmm = MMM(
    date_column="date",
    channel_columns=["tv", "radio"],
    target_column="sales",
    dims=("country", "region"),  # Custom hierarchical dimensions
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
)
```

**Key Attributes**:
- `self.dims`: Tuple of custom dimension names (e.g., `("country",)` or `("country", "region")`)
- `self.date_column`: Time dimension name
- `self.channel_columns`: List of channel names
- `self.target_column`: Target variable name

#### Data Storage with Custom Dimensions (lines 1034-1089)

```python
def _generate_and_preprocess_model_data(self, X: pd.DataFrame, y: pd.Series):
    # Convert to xarray preserving hierarchical structure
    X_dataarray = self._create_xarray_from_pandas(
        data=X,
        date_column=self.date_column,
        dims=self.dims,  # e.g., ("country", "region")
        metric_list=self.channel_columns,
        metric_coordinate_name="channel",
    )
    # Result: xarray with coords (date, country, region, channel)

    # Merge into single Dataset
    self.xarray_dataset = xr.merge(dataarrays).fillna(0)
```

### 2. fit_data Structure with Hierarchical Dimensions

**File**: `pymc_marketing/model_builder.py:905-922, 1014-1025`
**File**: `pymc_marketing/mmm/multidimensional.py:2552-2663`

#### fit_data Creation (MultidimensionalMMM override at lines 2552-2663)

```python
def create_fit_data(self, X, y) -> xr.Dataset:
    """Create fit_data preserving hierarchical dimension structure."""

    # Extract custom dimensions present in data
    dims_in_X = [d for d in self.dims if d in X_df.columns]
    coord_cols = [self.date_column, *dims_in_X]

    # Align data with MultiIndex structure
    # ... alignment logic ...

    # Convert to xarray with hierarchical coordinates
    ds = X_df.sort_values(coord_cols).set_index(coord_cols).to_xarray()
    return ds
```

#### Example fit_data Structures

**Single Custom Dimension** (`dims=("country",)`):
```python
<xarray.Dataset>
Dimensions:  (date: 52, country: 3)
Coordinates:
  * date      (date) datetime64[ns] 2024-01-07 2024-01-14 ... 2024-12-29
  * country   (country) object 'US' 'UK' 'DE'
Data variables:
    tv         (date, country) float64 100.0 120.0 95.0 ...
    radio      (date, country) float64 80.0 75.0 90.0 ...
    sales      (date, country) float64 450.0 520.0 480.0 ...
```

**Multiple Custom Dimensions** (`dims=("country", "region")`):
```python
<xarray.Dataset>
Dimensions:  (date: 52, country: 3, region: 4)
Coordinates:
  * date      (date) datetime64[ns] ...
  * country   (country) object 'US' 'UK' 'DE'
  * region    (region) object 'North' 'South' 'East' 'West'
Data variables:
    tv         (date, country, region) float64 ...
    radio      (date, country, region) float64 ...
    sales      (date, country, region) float64 ...
```

**No Custom Dimensions** (base case):
```python
<xarray.Dataset>
Dimensions:  (date: 52)
Coordinates:
  * date      (date) datetime64[ns] ...
Data variables:
    tv         (date) float64 100.0 120.0 ...
    radio      (date) float64 80.0 75.0 ...
    sales      (date) float64 450.0 520.0 ...
```

#### Key Pattern: Dimensions Preserved Throughout

**Internal xarray_dataset** (during model building):
```python
Dimensions:  (date: n_dates, country: n_countries, channel: n_channels)
Data variables:
    _channel   (date, country, channel) float64
    _target    (date, country) float64
```

**External fit_data** (stored in idata):
```python
Dimensions:  (date: n_dates, country: n_countries)
Data variables:
    tv         (date, country) float64
    radio      (date, country) float64
    sales      (date, country) float64
```

**Difference**: Channels become separate data variables in fit_data (no "channel" coordinate), but custom dimensions are preserved.

### 3. Working with fit_data in ROAS Computation

#### Pattern 1: Extract Data Preserving Dimensions

```python
# Access fit_data
fit_dataset = idata.fit_data

# Get channel columns (all except target)
channel_cols = [col for col in fit_dataset.data_vars if col != self.target_column]

# Extract specific channel (preserves all dimensions)
tv_data = fit_dataset["tv"]  # Shape: (date, country, region)

# Select specific dimension values
us_data = fit_dataset.sel(country="US")  # All channels for US
```

#### Pattern 2: Aggregate Across Dimensions for Period Totals

```python
# Sum spend across time for each channel and dimension combination
total_spend_per_dim = fit_dataset[channel_cols].sum(dim="date")
# Result: (country, region) for each channel

# Example: If dims=("country", "region"):
# tv: (country, region) array with total spend per country-region
# radio: (country, region) array with total spend per country-region
```

#### Pattern 3: Broadcasting Operations

```python
# Incremental contribution: (chain, draw, date, country, region)
# Total spend: (country, region) for each channel

# xarray automatically broadcasts for division:
roas = incremental.sum(dim="date") / total_spend
# Result: (chain, draw, country, region) per channel
```

### 4. sample_posterior_predictive with Custom Dimensions

**File**: `pymc_marketing/mmm/multidimensional.py:1800-1871`

```python
def sample_posterior_predictive(
    self,
    X: pd.DataFrame | xr.Dataset | None = None,
    extend_idata: bool = True,
    include_last_observations: bool = False,
    original_scale: bool = True,
    **sample_posterior_predictive_kwargs,
) -> xr.DataArray:
    """Sample from posterior predictive distribution.

    Returns DataArray with dimensions matching input structure.
    For models with custom dims, output has shape:
    (chain, draw, date, *custom_dims)
    """
```

**Key Features**:
- Accepts `xr.Dataset` directly (can pass fit_data or modified version)
- Preserves dimension structure in output
- `include_last_observations=True` prepends last `l_max` observations
- Returns shape: `(chain, draw, date, *custom_dims)`

**Usage for Counterfactual**:
```python
# Get actual predictions
Y_actual = self.sample_posterior_predictive(
    X=fit_dataset_window,  # xarray Dataset
    include_last_observations=True,
    original_scale=True,
)
# Shape: (chain, draw, date, country, region)

# Create counterfactual
fit_counter = fit_dataset_window.copy(deep=True)
for channel in self.channel_columns:
    fit_counter[channel].loc[dict(date=slice(t0, t1))] = 0

# Get counterfactual predictions
Y_counter = self.sample_posterior_predictive(
    X=fit_counter,
    include_last_observations=True,
    original_scale=True,
)
# Shape: (chain, draw, date, country, region)
```

### 5. Time Frequency Handling (Frequency-Agnostic)

**File**: `pymc_marketing/mmm/utils.py:268-277`

```python
# Infer date frequency from data
date_series = pd.to_datetime(original_data[date_col])
inferred_freq = pd.infer_freq(date_series.unique())

if inferred_freq is None:
    warnings.warn("Could not infer frequency. Using weekly ('W').")
    inferred_freq = "W"
```

**File**: `pymc_marketing/mmm/utils.py:181-224`

```python
def _convert_frequency_to_timedelta(periods: int, freq: str) -> pd.Timedelta:
    """Convert frequency string and periods to Timedelta."""
    base_freq = freq[0] if len(freq) > 1 else freq

    if base_freq == "D":
        return pd.Timedelta(days=periods)
    elif base_freq == "W":
        return pd.Timedelta(weeks=periods)
    elif base_freq == "M":
        return pd.Timedelta(days=periods * 30)  # Approximate
    # ... more frequencies ...
```

**Key Insight**: Never assume weekly data. All time operations use detected frequency.

### 6. Period Labeling Convention: Period-End Dates

**File**: `pymc_marketing/data/idata/utils.py:187-193`

```python
# Map user-friendly names to Pandas frequency strings
period_map = {
    "weekly": "W",        # Week ending Sunday
    "monthly": "ME",      # Month End (explicit)
    "quarterly": "QE",    # Quarter End (explicit)
    "yearly": "YE",       # Year End (explicit)
}
freq = period_map[period]

# Aggregate using xarray resample
if method == "sum":
    aggregated = group.resample(date=freq).sum(dim="date")
```

**Convention**: The codebase uses **period-end dates** for all time aggregations:
- Monthly: `"ME"` → last day of month (e.g., `2024-01-31`)
- Quarterly: `"QE"` → last day of quarter (e.g., `2024-03-31`)
- Yearly: `"YE"` → last day of year (e.g., `2024-12-31`)

**Example Period Labels**:
```python
# Weekly data: 2024-01-01 through 2024-12-31
# Monthly aggregation labels:
["2024-01-31", "2024-02-29", "2024-03-31", ..., "2024-12-31"]

# Quarterly aggregation labels:
["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]

# Yearly aggregation label:
["2024-12-31"]
```

### 7. Adstock and Carryover Window

**File**: `pymc_marketing/mmm/components/adstock.py:84-176`

```python
class AdstockTransformation(BaseModel, ABC):
    def __init__(self, l_max: int = Field(..., gt=0), ...):
        self.l_max = l_max  # Maximum carryover duration
```

All adstock types have `l_max` accessible via `model.adstock.l_max`.

**File**: `pymc_marketing/mmm/transformers.py:212-297`

```python
def geometric_adstock(x, alpha: float = 0.0, l_max: int = 12, ...):
    """Geometric adstock transformation.

    The cumulative media effect is a weighted average of current and
    previous l_max - 1 periods. l_max is maximum duration of carryover.
    """
```

With `ConvMode.After` (default), spend at time `t` affects contributions at times `t, t+1, ..., t+l_max-1`.

### 8. Carryover Handling Patterns

#### Pattern 1: include_last_observations (Carry-In)

**File**: `pymc_marketing/mmm/multidimensional.py:1665-1683`

```python
def _posterior_predictive_data_transformation(
    self, X, include_last_observations: bool = False
) -> xr.Dataset:
    """Transform data for posterior predictive sampling."""

    dataarrays = []
    if include_last_observations:
        # Prepend last l_max observations for historical carryover
        last_obs = self.xarray_dataset.isel(date=slice(-self.adstock.l_max, None))
        dataarrays.append(last_obs)

    # Add new prediction data
    X_xarray = self._create_xarray_from_pandas(...)
    dataarrays.append(X_xarray)

    return xr.concat(dataarrays, dim='date')
```

**Used For**: Capturing carryover from historical spend into evaluation period.

#### Pattern 2: Budget Optimizer Carryover Padding (Carry-Out)

**File**: `pymc_marketing/mmm/budget_optimizer.py:926-952`

```python
# Pad optimization horizon with l_max extra periods to capture carryover
repeated_budgets_with_carry_over_shape.insert(
    date_dim_idx, num_periods + max_lag
)
```

**Used For**: Capturing carryover from spend period into post-period.

## Implementation Recommendations

### Required Method Signature

**Location**: Add to `pymc_marketing/mmm/multidimensional.py` (MultidimensionalMMM class)

```python
def compute_roas(
    self,
    frequency: Literal["weekly", "monthly", "quarterly", "yearly", "all_time"],
    period_start: str | pd.Timestamp | None = None,
    period_end: str | pd.Timestamp | None = None,
    include_carryin: bool = True,
    include_carryout: bool = True,
    original_scale: bool = True,
) -> xr.DataArray:
    """Compute ROAS using counterfactual approach (Google MMM paper formula 10).

    Implements correct ROAS computation by comparing predicted KPI with actual
    spend vs. counterfactual (zero) spend for all channels, accounting for adstock
    carryover effects. Properly handles hierarchical dimensions (geo, country, etc.).

    Parameters
    ----------
    frequency : {"weekly", "monthly", "quarterly", "yearly", "all_time"}
        Time aggregation frequency for ROAS computation. REQUIRED parameter.
        Lower frequencies are more efficient computationally.
        - "all_time": Single ROAS across entire period
        - "yearly", "quarterly", "monthly": Aggregate by period
        - "weekly": Per-week ROAS
    period_start : str or Timestamp, optional
        Start date for ROAS evaluation. If None, uses start of fitted data period.
    period_end : str or Timestamp, optional
        End date for ROAS evaluation. If None, uses end of fitted data period.
    include_carryin : bool, default=True
        Include impact of pre-period channel spend via adstock carryover.
        When True, prepends last l_max observations to capture historical
        spending effects that carry into the evaluation period.
    include_carryout : bool, default=True
        Include impact of evaluation period spend that carries into post-period.
        When True, extends evaluation window by l_max periods to capture trailing
        adstock effects from spend during [period_start, period_end].
    original_scale : bool, default=True
        Return ROAS in original scale of target variable.

    Returns
    -------
    xr.DataArray
        ROAS with dimensions (chain, draw, date, channel, *custom_dims) where:
        - chain, draw: Posterior samples
        - date: Time periods (labeled with period-end dates)
        - channel: Marketing channels
        - *custom_dims: Any hierarchical dimensions (country, region, etc.)

        For models with dims=("country",), shape is (chain, draw, date, channel, country).
        For models with dims=("country", "region"), shape is (chain, draw, date, channel, country, region).

    Raises
    ------
    ValueError
        If frequency is not one of the allowed values.
        If period dates are outside fitted data range.

    Notes
    -----
    **Hierarchical Dimensions**: For models with custom dimensions like
    dims=("country",), ROAS is computed for each combination of (period, channel, country).
    The output preserves all dimensions using xarray's broadcasting.

    **Computational Cost**: Computing counterfactual ROAS requires 2 posterior
    predictive evaluations (actual vs counterfactual) per time period.
    Lower frequencies = fewer periods = faster computation.

    **Carryover Logic**:
    - include_carryin=True: Prepends last l_max observations before evaluation period
    - include_carryout=True: Extends evaluation window to period_end + l_max
    - Both default to True for accurate ROAS capturing full adstock effects

    **Period Labeling**: Periods are labeled with **period-end dates**:
    - Monthly: Last day of month (e.g., "2024-01-31")
    - Quarterly: Last day of quarter (e.g., "2024-03-31")
    - Yearly: Last day of year (e.g., "2024-12-31")

    **Edge Cases**:
    - Zero spend: Returns NaN for that channel/period/dimension combination
    - Evaluation window extends beyond data: Truncates to available data with warning
    - Time-varying media effects: Handled automatically in counterfactual predictions

    Examples
    --------
    >>> # Compute overall ROAS (no hierarchical dims)
    >>> roas = mmm.compute_roas(frequency="all_time")
    >>> roas.mean(dim=["chain", "draw"])
    <xarray.DataArray (channel: 2)>
    array([2.34, 1.87])
    Coordinates:
      * channel  (channel) object 'tv' 'radio'

    >>> # Compute quarterly ROAS with hierarchical dimensions
    >>> roas_q = mmm.compute_roas(
    ...     frequency="quarterly",
    ...     period_start="2024-01-01",
    ...     period_end="2024-12-31"
    ... )
    >>> # For model with dims=("country",):
    >>> roas_q.mean(dim=["chain", "draw"])
    <xarray.DataArray (date: 4, channel: 2, country: 3)>
    ...
    Coordinates:
      * date     (date) datetime64[ns] 2024-03-31 2024-06-30 2024-09-30 2024-12-31
      * channel  (channel) object 'tv' 'radio'
      * country  (country) object 'US' 'UK' 'DE'

    >>> # Compute ROAS without carryover (for comparison)
    >>> roas_no_carry = mmm.compute_roas(
    ...     frequency="all_time",
    ...     include_carryin=False,
    ...     include_carryout=False
    ... )

    References
    ----------
    Google MMM Paper: https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf
    Formula (10), Section 3.2.2
    """
```

### Implementation Pseudocode

```python
def compute_roas(
    self,
    frequency: Literal["weekly", "monthly", "quarterly", "yearly", "all_time"],
    period_start: str | pd.Timestamp | None = None,
    period_end: str | pd.Timestamp | None = None,
    include_carryin: bool = True,
    include_carryout: bool = True,
    original_scale: bool = True,
) -> xr.DataArray:
    """Compute ROAS using counterfactual approach with hierarchical dimensions."""

    # 1. Access fit_data (xarray Dataset with hierarchical structure)
    fit_dataset = self.idata.fit_data
    # Shape: (date, *custom_dims) for each channel variable

    # 2. Validate and set period bounds
    if period_start is None:
        period_start = fit_dataset.coords["date"].values[0]
    if period_end is None:
        period_end = fit_dataset.coords["date"].values[-1]

    period_start = pd.to_datetime(period_start)
    period_end = pd.to_datetime(period_end)

    # 3. Infer data frequency
    dates = pd.to_datetime(fit_dataset.coords["date"].values)
    inferred_freq = pd.infer_freq(dates)
    if inferred_freq is None:
        warnings.warn("Could not infer frequency. Using weekly ('W').")
        inferred_freq = "W"

    # 4. Create period groups based on requested frequency
    periods = self._create_period_groups(period_start, period_end, frequency)
    # Returns list of (period_start, period_end) tuples

    # 5. Get l_max for carryover calculations
    l_max = self.adstock.l_max

    # 6. Get channel columns (all except target)
    channel_cols = [col for col in fit_dataset.data_vars
                    if col != self.target_column]

    # 7. Compute ROAS for each period (vectorized across all channels and dims)
    roas_results = []

    for period_idx, (t0, t1) in enumerate(periods):
        # 7a. Determine data window based on carryover flags
        if include_carryin:
            data_start = t0 - _convert_frequency_to_timedelta(l_max, inferred_freq)
        else:
            data_start = t0

        if include_carryout:
            data_end = t1 + _convert_frequency_to_timedelta(l_max, inferred_freq)
        else:
            data_end = t1

        # 7b. Extract data window using xarray slicing (preserves all dimensions)
        fit_window = fit_dataset.sel(date=slice(data_start, data_end))
        # Shape: (date_window, *custom_dims) for each channel

        # 7c. Predict with actual spend (all channels active)
        # Note: sample_posterior_predictive can accept xarray Dataset directly
        Y_actual = self.sample_posterior_predictive(
            X=fit_window,
            extend_idata=False,
            include_last_observations=include_carryin,
            original_scale=original_scale,
            var_names=["y"],
        )
        # Shape: (chain, draw, date_extended, *custom_dims)

        # 7d. Create counterfactual: zero spend on ALL channels during [t0, t1]
        fit_counter = fit_window.copy(deep=True)
        for channel in channel_cols:
            # Use xarray's loc with dict selector to set period to zero
            # This preserves all custom dimensions
            fit_counter[channel].loc[dict(date=slice(t0, t1))] = 0

        # 7e. Predict with counterfactual (zero spend)
        Y_counter = self.sample_posterior_predictive(
            X=fit_counter,
            extend_idata=False,
            include_last_observations=include_carryin,
            original_scale=original_scale,
            var_names=["y"],
        )
        # Shape: (chain, draw, date_extended, *custom_dims)

        # 7f. Compute incremental contribution
        # Extract evaluation window: [t0, data_end] (includes carryout if enabled)
        eval_dates = fit_window.sel(date=slice(t0, data_end)).coords["date"]

        incremental = Y_actual.sel(date=eval_dates) - Y_counter.sel(date=eval_dates)
        # Shape: (chain, draw, date_eval, *custom_dims)

        # 7g. Sum over evaluation window
        total_incremental = incremental.sum(dim="date")
        # Shape: (chain, draw, *custom_dims)

        # 7h. Compute total spend in period [t0, t1] (NOT extended window)
        # Use xarray operations to preserve all dimensions
        spend_window = fit_dataset.sel(date=slice(t0, t1))

        # Sum spend over time for each channel and dimension combination
        total_spend = xr.Dataset()
        for channel in channel_cols:
            total_spend[channel] = spend_window[channel].sum(dim="date")
        # Shape per channel: (*custom_dims,)

        # 7i. Compute ROAS for each channel using xarray broadcasting
        # Handle zero spend
        roas_period = xr.Dataset()
        for channel in channel_cols:
            roas_period[channel] = xr.where(
                total_spend[channel] == 0,
                np.nan,
                total_incremental / total_spend[channel]
            )
        # Shape per channel: (chain, draw, *custom_dims)

        # 7j. Stack channels into a new dimension
        roas_period_da = xr.concat(
            [roas_period[ch] for ch in channel_cols],
            dim=pd.Index(channel_cols, name="channel")
        )
        # Shape: (chain, draw, channel, *custom_dims)

        # 7k. Add period coordinate (use period-end date)
        period_label = self._format_period_label(t0, t1, frequency)
        roas_period_da = roas_period_da.assign_coords(date=period_label).expand_dims("date")

        roas_results.append(roas_period_da)

    # 8. Concatenate all periods
    roas_da = xr.concat(roas_results, dim="date")
    # Final shape: (chain, draw, date, channel, *custom_dims)

    # 9. Ensure dimension order (chain, draw, date, channel, *custom_dims)
    dim_order = ["chain", "draw", "date", "channel", *self.dims]
    roas_da = roas_da.transpose(*dim_order)

    return roas_da


def _create_period_groups(
    self,
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Create list of (period_start, period_end) tuples for given frequency."""

    if frequency == "all_time":
        return [(start, end)]

    # Create period index using pandas to_period
    dates = pd.date_range(start, end, freq='D')

    freq_map = {
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "Y"
    }

    periods = dates.to_period(freq_map[frequency])
    unique_periods = periods.unique()

    # Convert periods to timestamp ranges
    period_ranges = []
    for period in unique_periods:
        period_start = period.to_timestamp()
        period_end = period.to_timestamp(how='end')  # Period-end date

        # Clip to requested range
        period_start = max(period_start, start)
        period_end = min(period_end, end)

        period_ranges.append((period_start, period_end))

    return period_ranges


def _format_period_label(
    self,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    frequency: str
) -> pd.Timestamp:
    """Format period label using period-end date convention.

    Returns the period-end date as a Timestamp.
    """
    # Return period-end date (t1) directly
    # This matches the codebase convention of using period-end dates
    return t1
```

### Summary Interface Integration

**File**: `pymc_marketing/mmm/summary.py`

Remove the old `roas()` method entirely and replace with:

```python
def roas(
    self,
    frequency: Frequency,  # Required, no default
    period_start: str | pd.Timestamp | None = None,
    period_end: str | pd.Timestamp | None = None,
    include_carryin: bool = True,
    include_carryout: bool = True,
    hdi_probs: Sequence[float] = (0.025, 0.5, 0.975),
    output_format: OutputFormatType = "dataframe",
) -> pd.DataFrame | xr.Dataset:
    """Compute ROAS with summary statistics using counterfactual approach.

    Parameters
    ----------
    frequency : {"weekly", "monthly", "quarterly", "yearly", "all_time"}
        Time aggregation frequency. REQUIRED parameter.
    period_start, period_end : str or Timestamp, optional
        Period bounds for ROAS evaluation.
    include_carryin : bool, default=True
        Include pre-period adstock carryover effects.
    include_carryout : bool, default=True
        Include post-period adstock carryover effects.
    hdi_probs : sequence of float, default=(0.025, 0.5, 0.975)
        HDI probability levels for credible intervals.
    output_format : {"dataframe", "dataset"}, default="dataframe"
        Output format.

    Returns
    -------
    DataFrame or Dataset
        ROAS summary with mean, HDI intervals, and other statistics.
        For models with hierarchical dimensions, DataFrame includes columns
        for each dimension (e.g., 'country', 'region').

    Examples
    --------
    >>> # Get overall ROAS summary
    >>> summary.roas(frequency="all_time")

    >>> # Get quarterly ROAS with custom HDI levels
    >>> summary.roas(
    ...     frequency="quarterly",
    ...     hdi_probs=(0.05, 0.5, 0.95)
    ... )
    """
    # Delegate to model's compute_roas method
    roas_data = self.model.compute_roas(
        frequency=frequency,
        period_start=period_start,
        period_end=period_end,
        include_carryin=include_carryin,
        include_carryout=include_carryout,
        original_scale=True,
    )
    # Shape: (chain, draw, date, channel, *custom_dims)

    # Compute summary statistics with HDI
    df = self._compute_summary_stats_with_hdi(roas_data, hdi_probs)
    # DataFrame includes columns: date, channel, *custom_dims, mean, hdi_X%, ...

    return self._convert_output(df, output_format)
```

### Documentation Requirements

#### Updated Docstrings

The method docstring should include:
- Formula reference to Google MMM paper formula (10)
- Clear explanation of counterfactual approach
- Examples for common use cases (all_time, quarterly, with/without carryover)
- Examples showing hierarchical dimension handling
- Performance notes explaining computational cost by frequency
- Period-end date labeling convention

## Architecture Insights

### Why Simple Division is Wrong

The stored `channel_contribution[t, m]` includes effects from multiple spend periods due to adstock convolution:

```
channel_contribution[t, m] = saturation(adstock(spend[:, m]))[t]
adstock(spend)[t] = w[0]*spend[t] + w[1]*spend[t-1] + ... + w[l_max-1]*spend[t-l_max+1]
```

Therefore:
1. `contribution[t]` includes effects from past spend (t-l_max+1 through t)
2. `spend[t]` creates effects extending into future (t through t+l_max-1)

Simple division `contribution[t] / spend[t]` incorrectly assumes:
- All of `contribution[t]` comes from `spend[t]` ❌
- `spend[t]` only affects `contribution[t]` ❌

### The Counterfactual Solution

To correctly measure incremental effect of spend during [t0, t1]:

1. **Predict with channel active**: Full model prediction with actual spend
2. **Predict with channel removed**: Counterfactual with zero spend in [t0, t1]
3. **Compute difference**: `incremental = Y_actual - Y_counterfactual`
4. **Account for carryover**:
   - Carry-in: Prepend l_max historical observations
   - Carry-out: Extend evaluation window by l_max periods
5. **Aggregate**: Sum incremental over extended window, divide by period spend
6. **Handle dimensions**: xarray broadcasting handles all hierarchical dimensions automatically

### Hierarchical Dimensions in ROAS Computation

For a model with `dims=("country", "region")`:

**Input fit_data**:
```
Dimensions: (date: 52, country: 3, region: 4)
Data variables:
  tv: (date, country, region)
  radio: (date, country, region)
```

**After counterfactual computation**:
```
Y_actual: (chain, draw, date, country, region)
Y_counter: (chain, draw, date, country, region)
incremental: (chain, draw, date, country, region)
```

**After aggregation over time**:
```
total_incremental: (chain, draw, country, region)
total_spend[tv]: (country, region)
total_spend[radio]: (country, region)
```

**Final ROAS**:
```
roas: (chain, draw, date, channel, country, region)
```

**Key Insight**: xarray's dimension-aware operations automatically broadcast and align data across all hierarchical dimensions without explicit loops.

### Efficiency Through Frequency Selection

Computing counterfactual ROAS requires posterior predictive evaluations. The number of evaluations scales with the number of periods:

| Frequency | Periods (104 weeks) | Evaluations per Period | Total Evaluations |
|-----------|---------------------|------------------------|-------------------|
| all_time  | 1                   | 2 (actual + counter)   | 2                 |
| yearly    | 2                   | 2                      | 4                 |
| quarterly | 8                   | 2                      | 16                |
| monthly   | 24                  | 2                      | 48                |
| weekly    | 104                 | 2                      | 208               |

**Recommendation**: Use "all_time" or "yearly" for typical use cases unless temporal variation is critical.

### Data Flow Summary

```
Input: frequency, period_start, period_end, carryover flags

1. Access fit_data (xarray Dataset with hierarchical structure)
   └─> Dimensions: (date, *custom_dims) for each channel

2. Create period groups based on frequency
   └─> [(t0_1, t1_1), (t0_2, t1_2), ...]

3. For each period:
   a. Extract data window from fit_data (preserves all dimensions)
      ├─ if include_carryin: start = t0 - l_max*freq
      └─ if include_carryout: end = t1 + l_max*freq

   b. Sample with actual spend
      └─> Y_actual(chain, draw, date_extended, *custom_dims)

   c. Create counterfactual (zero channels in [t0, t1])
      └─> fit_counter with channels=0 during [t0, t1]

   d. Sample with counterfactual
      └─> Y_counter(chain, draw, date_extended, *custom_dims)

   e. Compute incremental
      └─> incremental = Y_actual - Y_counter

   f. Aggregate over time
      └─> total_incremental(chain, draw, *custom_dims)

   g. Compute spend totals
      └─> total_spend(*custom_dims) per channel

   h. Compute ROAS with xarray broadcasting
      └─> roas_period(chain, draw, channel, *custom_dims)

4. Concatenate all periods with period-end date labels
   └─> roas(chain, draw, date, channel, *custom_dims)
```

## Code References

### Primary Files to Modify

- **`pymc_marketing/mmm/multidimensional.py`** - Add `compute_roas()` method
- **`pymc_marketing/data/idata/mmm_wrapper.py:315-342`** - Remove old `get_roas()` method
- **`pymc_marketing/mmm/summary.py:470-520`** - Replace `roas()` with new implementation

### Key Infrastructure to Use

- **`pymc_marketing/mmm/multidimensional.py:1800`** - `sample_posterior_predictive()` for predictions
- **`pymc_marketing/mmm/multidimensional.py:2552`** - `create_fit_data()` for understanding structure
- **`pymc_marketing/mmm/utils.py:181`** - `_convert_frequency_to_timedelta()` for frequency handling
- **`pymc_marketing/mmm/utils.py:268`** - Frequency inference from data
- **`pymc_marketing/data/idata/utils.py:187`** - Period mapping (for reference on conventions)

### Reference Implementations

- **`pymc_marketing/mmm/sensitivity_analysis.py:289-438`** - Vectorized counterfactual pattern
- **`pymc_marketing/mmm/budget_optimizer.py:926-952`** - Carryover padding pattern
- **`pymc_marketing/data/idata/utils.py:136-213`** - Time aggregation with period-end labeling

### Test Files to Update/Create

- **`tests/mmm/test_multidimensional.py`** - Add comprehensive ROAS tests
- **`tests/mmm/test_summary.py:976-980`** - Update `test_division_by_zero_in_roas_handled()`
- **`tests/mmm/test_summary.py:380-382`** - Update `test_roas_summary_schema()`
- **New**: `tests/mmm/test_roas_counterfactual.py` - Dedicated ROAS test suite

## Implementation Checklist

- [ ] Implement `compute_roas()` method in MultidimensionalMMM class
  - [ ] Use xarray operations on fit_data (not pandas)
  - [ ] Handle hierarchical dimensions with broadcasting
  - [ ] Implement period-end date labeling
  - [ ] Support carryin/carryout flags separately
- [ ] Add `_create_period_groups()` helper for frequency-based grouping
- [ ] Add `_format_period_label()` helper returning period-end Timestamp
- [ ] Remove old `get_roas()` from MMMIDataWrapper
- [ ] Update `roas()` in MMMSummaryFactory
- [ ] Update docstrings with formula reference and hierarchical dimension examples
- [ ] Validate with models having multiple custom dimensions
- [ ] Performance profiling with different frequencies

## Open Questions (Resolved)

### 1. Backward Compatibility
**Decision**: Remove old behavior entirely per user feedback.

### 2. Default Frequency
**Decision**: Make `frequency` parameter required (no default).

### 3. Edge Period Handling
**Decision**: Truncate evaluation window at data boundaries with warning.

### 4. Data Access
**Decision**: Use `idata.fit_data` xarray Dataset (not self.X pandas).

### 5. Hierarchical Dimensions
**Decision**: Use xarray operations to handle all dimensions via broadcasting.

### 6. Output Shape
**Decision**: `(chain, draw, date, channel, *custom_dims)` where date represents periods.

### 7. Period Labeling
**Decision**: Use period-end dates following codebase convention (e.g., "2024-01-31" for January).

### 8. Carryover Flags
**Decision**: Split into `include_carryin` and `include_carryout` (both default True).

## Related Research

- **Google MMM Paper**: https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf (Formula 10)
- **Jin et al. (2017)**: "Bayesian methods for media mix modeling with carryover and shape effects"
- **Budget optimizer implementation**: Demonstrates correct carryover handling pattern

## Next Steps

1. **Implement core `compute_roas()` method** in MultidimensionalMMM class
   - Use counterfactual approach with `sample_posterior_predictive()`
   - Work with xarray operations on fit_data
   - Support hierarchical dimensions with broadcasting
   - Implement carryin/carryout flags
   - Use period-end date labeling

2. **Update summary interface**
   - Remove old ROAS method
   - Add new ROAS method with required frequency parameter
   - Ensure DataFrame output includes hierarchical dimension columns

3. **Documentation**
   - Update method docstrings with formula reference
   - Include examples with hierarchical dimensions
   - Explain period-end date labeling convention

4. **Validation**
   - Test with models having various hierarchical dimension configurations
   - Validate xarray broadcasting works correctly across all dimensions
   - Confirm period-end date labels match convention
