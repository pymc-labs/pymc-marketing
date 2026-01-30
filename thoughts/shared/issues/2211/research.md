---
date: 2026-01-30T16:00:50+00:00
researcher: Claude Sonnet 4.5
git_commit: fa4057d0cbe17211ecef590c70c87f9f18879d91
branch: work-issue-2211
repository: pymc-labs/pymc-marketing
topic: "Compute ROAS the correct way (like in the Google paper)"
tags: [research, codebase, roas, adstock, carryover, mmm, counterfactual, multidimensional, vectorization]
status: complete
last_updated: 2026-01-30
last_updated_by: Claude Sonnet 4.5
issue_number: 2211
---

# Research: Compute ROAS the Correct Way (Google MMM Paper Formula 10)

**Date**: 2026-01-30T16:00:50+00:00
**Researcher**: Claude Sonnet 4.5
**Git Commit**: fa4057d0cbe17211ecef590c70c87f9f18879d91
**Branch**: work-issue-2211
**Repository**: pymc-labs/pymc-marketing
**Issue**: #2211

## Research Question

How is ROAS currently computed in the codebase, and what changes are needed to implement formula (10) from the Google MMM paper using a **counterfactual approach** that accounts for carryover effects? Specifically, the implementation must work with the MultidimensionalMMM class (not the deprecated MMM class in `mmm.py`), handle arbitrary time frequencies, process all channels simultaneously, and support separate flags for carry-in and carry-out effects.

## Summary

**CRITICAL UPDATE**: Based on user feedback, the implementation approach has been significantly refined:

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

1. **Target Class**: `MultidimensionalMMM` in `pymc_marketing/mmm/multidimensional.py` (NOT the deprecated MMM class)

2. **Data Access**: X and y are stored as `fit_data` InferenceData group. Extract X via:
   ```python
   dataset = idata.fit_data.to_dataframe().reset_index()
   X = dataset.drop(columns=[self.target_column])
   y = dataset[self.target_column]
   ```

3. **Vectorized Processing**: Compute counterfactuals for ALL channels simultaneously using `vectorize_graph()` from `pytensor_utils.py`, avoiding per-channel loops

4. **Frequency-Agnostic**: Use `pd.infer_freq()` to detect data frequency (daily, weekly, monthly, etc.) - never assume weekly

5. **Carryover Flags**: Split into two separate controls:
   - `include_carryin` (default True): Include pre-period channel contribution impact
   - `include_carryout` (default True): Include post-period channel contribution impact

6. **Efficiency**: Users typically want ROAS at lower frequencies (yearly, quarterly) than data frequency. Compute ROAS directly at requested frequency instead of per-period aggregation.

7. **No Backwards Compatibility**: Remove old ROAS behavior entirely

8. **Required Frequency Parameter**: Make `frequency` parameter required (no default) to force users to think about computational cost

## Problem Statement

### Current Implementation (Incorrect)
**File**: `pymc_marketing/data/idata/mmm_wrapper.py:315-342`

```python
ROAS[t] = contribution[t] / spend[t]
```

**Problem**: Element-wise division that:
- Ignores carryover effects from adstock transformation
- Treats each period independently
- Does not use counterfactual approach

### Required Implementation (Counterfactual Approach)

```python
# For all channels over period [t0, t1]:
# 1. Predict KPI with actual spend on all channels
Y_actual = model.sample_posterior_predictive(X_actual)

# 2. Create counterfactual: zero spend on all channels during [t0, t1]
X_counterfactual = X_actual.copy()
X_counterfactual.loc[t0:t1, all_channels] = 0

# 3. Predict KPI with counterfactual (zero) spend
Y_counterfactual = model.sample_posterior_predictive(X_counterfactual)

# 4. Compute incremental contribution with vectorized operations
incremental = Y_actual - Y_counterfactual  # Shape: (sample, date_extended)

# 5. Sum over extended window and divide by spend
# Use xarray operations to broadcast across all channels simultaneously
total_incremental = incremental.sum(dim='date')  # (sample, channel)
total_spend = spend.sum(dim='date')  # (channel,)
ROAS = total_incremental / total_spend  # (sample, channel)
```

## Detailed Findings

### 1. MultidimensionalMMM Class Structure

**File**: `pymc_marketing/mmm/multidimensional.py:213-3000+`

#### Class Definition and Key Attributes
```python
class MMM(RegressionModelBuilder):
    """Marketing Mix Model class (multi-dimensional panel data support)."""

    # Key attributes:
    # self.date_column: str - Date column name
    # self.channel_columns: list[str] - Media channel names
    # self.target_column: str - Target variable name
    # self.dims: tuple[str, ...] - Additional dimensions (geo, region, etc.)
    # self.adstock: AdstockTransformation - Access via self.adstock.l_max
    # self.saturation: SaturationTransformation
    # self.X: pd.DataFrame - Stored raw pandas data
    # self.y: pd.Series - Stored raw target
    # self.xarray_dataset: xr.Dataset - Internal xarray format
```

#### Data Storage During Preprocessing (lines 1034-1089)
```python
def _generate_and_preprocess_model_data(self, X: pd.DataFrame, y: pd.Series):
    self.X = X  # Line 1039: Raw pandas DataFrame stored
    self.y = y  # Line 1040: Raw pandas Series stored

    # Convert to xarray for internal use
    X_dataarray = self._create_xarray_from_pandas(...)
    y_dataarray = self._create_xarray_from_pandas(...)

    # Merge into single xarray Dataset
    self.xarray_dataset = xr.merge(dataarrays).fillna(0)

    # Extract coordinates for model
    self.model_coords = {
        dim: self.xarray_dataset.coords[dim].values
        for dim in self.xarray_dataset.coords.dims
    }
```

**Key Insight**: `self.X` and `self.y` are readily available for ROAS computation without needing to extract from `fit_data`.

#### sample_posterior_predictive Method (lines 1800-1871)
```python
def sample_posterior_predictive(
    self,
    X: pd.DataFrame | None = None,
    extend_idata: bool = True,
    combined: bool = True,
    include_last_observations: bool = False,
    clone_model: bool = True,
    **sample_posterior_predictive_kwargs,
) -> xr.DataArray:
    """Sample from posterior predictive distribution.

    Parameters
    ----------
    include_last_observations : bool
        If True, prepends last l_max observations for adstock continuity.
        Critical for capturing carryover from historical spend.
    """
```

**Key Features**:
- Accepts modified X data (enables counterfactual scenarios)
- `include_last_observations=True` prepends last `l_max` observations (line 1672)
- Updates model data via `_set_xarray_data()` (lines 1730-1796)
- Returns xarray DataArray with posterior samples

#### Adstock Access Pattern
```python
# Access l_max from adstock component
self.adstock.l_max  # Returns integer carryover window length

# Used throughout codebase:
# Line 1868: Remove extra observations after prediction
posterior_predictive_samples.isel(date=slice(self.adstock.l_max, None))

# Line 1672: Get last observations for carryover
last_obs = self.xarray_dataset.isel(date=slice(-self.adstock.l_max, None))
```

### 2. fit_data Structure and Extraction

**File**: `pymc_marketing/model_builder.py:905-922, 1014-1025`

#### Creation During Fit (line 1014-1025)
```python
# After sampling, create fit_data group
fit_data = self.create_fit_data(X, y)

# Add to InferenceData with warning suppression
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning,
                           message="The group fit_data is not defined...")
    self.idata.add_groups(fit_data=fit_data)
```

#### Extraction Pattern
**File**: `pymc_marketing/mmm/multidimensional.py:2695-2705`

```python
def build_from_idata(self, idata: az.InferenceData) -> None:
    """Rebuild model from InferenceData object."""
    dataset = idata.fit_data.to_dataframe()  # Convert to DataFrame

    # Handle MultiIndex or DatetimeIndex
    if isinstance(dataset.index, (pd.MultiIndex, pd.DatetimeIndex)):
        dataset = dataset.reset_index()

    # Separate features from target
    X = dataset.drop(columns=[self.target_column])
    y = dataset[self.target_column]

    self.build_model(X, y)
```

**Key Insight**: For ROAS computation, prefer using `self.X` and `self.y` directly rather than extracting from `fit_data`. The `fit_data` extraction is primarily for model persistence/loading.

### 3. Vectorized Counterfactual Computation

**File**: `pymc_marketing/mmm/sensitivity_analysis.py:289-438`

The `SensitivityAnalysis` class demonstrates the efficient vectorized pattern:

#### Step 1: Extract Response Distribution (lines 165-174)
```python
def _extract_response_distribution(self, response_variable: str,
                                     posterior_sample_fraction: float):
    """Extract response distribution graph conditioned on posterior samples."""
    draw_selection = self._draw_indices_for_percentage(posterior_sample_fraction)
    return extract_response_distribution(
        pymc_model=self.model,
        idata=self.idata.isel(draw=draw_selection),
        response_variable=response_variable,
    )
```

**What it does** (`pytensor_utils.py:264-341`):
- Converts InferenceData to sample-major xarray (line 292)
- Identifies needed free RVs via graph traversal (lines 298-301)
- Replaces RVs with placeholders via `clone_replace()` (lines 302-312)
- Replaces placeholders with posterior samples (lines 322-329)
- **Vectorizes graph** using `vectorize_graph()` (line 329)

#### Step 2: Create Batched Input (lines 388-407)
```python
# Prepare batched input with sweep axis for multiple scenarios
data_shared: SharedVariable = self.model[var_input]
base_value = data_shared.get_value()  # Shape: (date, channel)
num_sweeps = int(np.asarray(sweep_values).shape[0])

# Build (sweep, 1, 1, ...) array to broadcast against base_value
sweep_col = np.asarray(sweep_values, dtype=base_value.dtype).reshape(
    (num_sweeps,) + (1,) * base_value.ndim
)

if sweep_type == "multiplicative":
    batched_input = sweep_col * base_value[None, ...]  # (sweep, date, channel)
elif sweep_type == "additive":
    batched_input = sweep_col + base_value[None, ...]
elif sweep_type == "absolute":
    batched_input = np.broadcast_to(sweep_col, (num_sweeps, *base_value.shape))
```

**Key Pattern**: Broadcasting creates shape `(n_scenarios, n_dates, n_channels)` efficiently.

#### Step 3: Vectorize Graph Over Sweep Axis (lines 409-419)
```python
# Vectorize response graph over the new sweep axis
channel_in = pt.tensor(
    name=f"{var_input}_sweep_in",
    dtype=data_shared.dtype,
    shape=(None, *data_shared.type.shape),  # (sweep, date, channel)
)

sweep_graph = vectorize_graph(resp_graph, replace={data_shared: channel_in})
fn = function([channel_in], sweep_graph)  # Output: (sweep, sample, date, *dims)

# Evaluate ALL scenarios in single call
evaluated = fn(batched_input)
```

**Result**: `evaluated` has shape `(n_scenarios, n_samples, n_dates_extended)` computed in **single PyTensor evaluation**.

### 4. Time Frequency Handling (Frequency-Agnostic)

**File**: `pymc_marketing/mmm/utils.py:268-277, 181-224`

#### Automatic Frequency Inference
```python
# Infer date frequency from data
date_series = pd.to_datetime(original_data[date_col])
inferred_freq = pd.infer_freq(date_series.unique())

if inferred_freq is None:  # Fallback with warning
    warnings.warn(
        f"Could not infer frequency from '{date_col}'. Using weekly ('W').",
        UserWarning,
        stacklevel=2,
    )
    inferred_freq = "W"
```

**Returns**: Standard Pandas offset string ('D', 'W', 'M', 'Y', etc.)

#### Frequency to Timedelta Conversion
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
    elif base_freq == "Y":
        return pd.Timedelta(days=periods * 365)  # Approximate
    # ... more frequencies ...
    else:
        warnings.warn(f"Unrecognized frequency '{freq}'. Defaulting to weeks.")
        return pd.Timedelta(weeks=periods)
```

**Key Insight**: Never assumes weekly data. All time operations use detected frequency.

#### Period-Based Grouping
**File**: `pymc_marketing/data/idata/utils.py:187-207`

```python
# Map user-friendly names to Pandas frequency strings
period_map = {
    "weekly": "W",
    "monthly": "ME",      # Modern pandas 2.2+ alias (month-end)
    "quarterly": "QE",    # Quarter-end
    "yearly": "YE",       # Year-end
}
freq = period_map[period]

# Aggregate using xarray resample
if method == "sum":
    aggregated = group.resample(date=freq).sum(dim="date")
elif method == "mean":
    aggregated = group.resample(date=freq).mean(dim="date")
```

**Key Pattern**: Use xarray's `resample()` for frequency-aware aggregation. Works on multi-dimensional data (chain, draw, date, channel).

### 5. Carryover Handling Patterns

#### Pattern 1: include_last_observations Parameter

**File**: `pymc_marketing/mmm/multidimensional.py:1665-1683`

```python
def _posterior_predictive_data_transformation(
    self, X: pd.DataFrame, include_last_observations: bool = False
) -> xr.Dataset:
    """Transform data for posterior predictive sampling."""

    # Validate no date overlap when including last observations
    self._validate_date_overlap_with_include_last_observations(
        X, include_last_observations
    )

    dataarrays = []
    if include_last_observations:
        # Prepend last l_max observations for historical carryover
        last_obs = self.xarray_dataset.isel(date=slice(-self.adstock.l_max, None))
        dataarrays.append(last_obs)

    # Add new prediction data
    X_xarray = self._create_xarray_from_pandas(...)
    dataarrays.append(X_xarray)

    # Concatenate historical + new data
    return xr.concat(dataarrays, dim='date')
```

**Used For**: Capturing carryover from historical spend into evaluation period (carry-in effect).

#### Pattern 2: Date Overlap Validation

**File**: `pymc_marketing/mmm/multidimensional.py:1559-1594`

```python
def _validate_date_overlap_with_include_last_observations(
    self, X: pd.DataFrame, include_last_observations: bool
) -> None:
    """Prevent duplicate dates when using include_last_observations."""

    if not include_last_observations:
        return

    training_dates = pd.to_datetime(self.model_coords["date"])
    input_dates = pd.to_datetime(X[self.date_column].unique())

    overlapping_dates = set(training_dates).intersection(set(input_dates))

    if overlapping_dates:
        raise ValueError(
            f"Cannot use include_last_observations=True with overlapping dates. "
            f"Overlapping: {overlapping_dates}"
        )
```

**Purpose**: Ensures `include_last_observations=True` only used for truly future predictions.

#### Pattern 3: Budget Optimizer Carryover Padding

**File**: `pymc_marketing/mmm/budget_optimizer.py:926-952`

```python
def _replace_channel_data_by_optimization_variable(self, model: Model) -> Model:
    """Replace channel_data with optimization variable, padded for carryover."""

    num_periods = self.num_periods
    max_lag = self.mmm_model.adstock.l_max

    # Create tensor of size (num_periods + max_lag)
    repeated_budgets_with_carry_over_shape = list(tuple(budgets.shape))
    repeated_budgets_with_carry_over_shape.insert(
        date_dim_idx, num_periods + max_lag
    )

    # Initialize with zeros
    repeated_budgets_with_carry_over = pt.zeros(
        repeated_budgets_with_carry_over_shape,
        dtype=channel_data_dtype,
    )

    # Set first num_periods to actual budgets, rest remain zero for carryover
    set_idxs = (*((slice(None),) * date_dim_idx), slice(None, num_periods))
    repeated_budgets_with_carry_over = repeated_budgets_with_carry_over[set_idxs].set(
        pt.cast(repeated_budgets, channel_data_dtype)
    )

    return do(model, {"channel_data": repeated_budgets_with_carry_over})
```

**Used For**: Capturing carryover from spend period into post-period (carry-out effect).

#### Pattern 4: Post-Prediction Slicing

**File**: `pymc_marketing/mmm/multidimensional.py:1865-1869`

```python
if include_last_observations:
    # Remove extra observations used for adstock continuity
    posterior_predictive_samples = posterior_predictive_samples.isel(
        date=slice(self.adstock.l_max, None)
    )
```

**Purpose**: After sampling with prepended historical data, remove the first `l_max` observations to return only requested future periods.

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

    Implements the correct ROAS computation by comparing predicted KPI with actual
    spend vs. counterfactual (zero) spend for all channels, accounting for adstock
    carryover effects.

    Parameters
    ----------
    frequency : {"weekly", "monthly", "quarterly", "yearly", "all_time"}
        Time aggregation frequency for ROAS computation. REQUIRED parameter.
        Lower frequencies are significantly more efficient computationally.
        - "all_time": Single ROAS across entire period (most efficient)
        - "yearly", "quarterly", "monthly": Aggregate by period
        - "weekly": Per-week ROAS (most expensive, matches data frequency)
    period_start : str or Timestamp, optional
        Start date for ROAS evaluation. If None, uses start of fitted data period.
    period_end : str or Timestamp, optional
        End date for ROAS evaluation. If None, uses end of fitted data period.
    include_carryin : bool, default=True
        Include impact of pre-period channel spend via adstock carryover.
        When True, uses include_last_observations=True in posterior prediction
        to capture historical spending effects that carry into the evaluation period.
    include_carryout : bool, default=True
        Include impact of evaluation period spend that carries into post-period.
        When True, extends evaluation window by l_max periods to capture trailing
        adstock effects from spend during [period_start, period_end].
    original_scale : bool, default=True
        Return ROAS in original scale of target variable.

    Returns
    -------
    xr.DataArray
        ROAS with dims (sample, period, channel) where:
        - sample: Posterior samples (chain × draw)
        - period: Time periods based on frequency parameter
        - channel: Marketing channels

    Raises
    ------
    ValueError
        If frequency is not one of the allowed values.
        If period dates are outside fitted data range.

    Notes
    -----
    **Computational Cost**: Computing counterfactual ROAS requires 2 posterior
    predictive evaluations (actual vs counterfactual) for each time period.
    For a model with 104 weeks and 5 channels:
    - frequency="all_time": 2 evaluations (very fast)
    - frequency="yearly" (2 years): 4 evaluations (fast)
    - frequency="monthly" (24 months): 48 evaluations (moderate)
    - frequency="weekly" (104 weeks): 208 evaluations (slow)

    **Carryover Logic**:
    - include_carryin=True: Prepends last l_max observations before evaluation period
    - include_carryout=True: Extends evaluation window to period_end + l_max
    - Both default to True for accurate ROAS capturing full adstock effects

    **Edge Cases**:
    - Zero spend in period: Returns NaN for that channel/period
    - Evaluation window extends beyond fitted data: Truncates to available data with warning
    - Time-varying media effects: Handled automatically in counterfactual predictions

    Examples
    --------
    >>> # Compute overall ROAS for all channels (most common use case)
    >>> roas = mmm.compute_roas(frequency="all_time")
    >>> roas.mean(dim="sample")  # Posterior mean ROAS by channel
    <xarray.DataArray (channel: 3)>
    array([2.34, 1.87, 3.12])
    Coordinates:
      * channel  (channel) <U10 'tv' 'radio' 'digital'

    >>> # Compute quarterly ROAS
    >>> roas_q = mmm.compute_roas(
    ...     frequency="quarterly",
    ...     period_start="2023-01-01",
    ...     period_end="2024-12-31"
    ... )
    >>> roas_q.mean(dim="sample")  # Mean ROAS by quarter and channel
    <xarray.DataArray (period: 8, channel: 3)>
    ...

    >>> # Compute ROAS without carryover effects (for comparison)
    >>> roas_no_carry = mmm.compute_roas(
    ...     frequency="all_time",
    ...     include_carryin=False,
    ...     include_carryout=False
    ... )

    >>> # Access credible intervals
    >>> az.hdi(roas, hdi_prob=0.94)

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
    """Compute ROAS using counterfactual approach."""

    # 1. Validate and set period bounds
    if period_start is None:
        period_start = self.X[self.date_column].min()
    if period_end is None:
        period_end = self.X[self.date_column].max()

    period_start = pd.to_datetime(period_start)
    period_end = pd.to_datetime(period_end)

    # 2. Infer data frequency for timedelta calculations
    inferred_freq = pd.infer_freq(pd.to_datetime(self.X[self.date_column]).unique())
    if inferred_freq is None:
        warnings.warn("Could not infer frequency. Using weekly ('W').")
        inferred_freq = "W"

    # 3. Create period groups based on requested frequency
    periods = self._create_period_groups(period_start, period_end, frequency)
    # Returns list of (period_start, period_end) tuples

    # 4. Get l_max for carryover window calculations
    l_max = self.adstock.l_max

    # 5. Compute ROAS for each period (vectorized across channels)
    roas_results = []

    for period_idx, (t0, t1) in enumerate(periods):
        # 5a. Determine data window based on carryover flags
        if include_carryin:
            # Extend start backwards by l_max for historical carryover
            data_start = t0 - _convert_frequency_to_timedelta(l_max, inferred_freq)
        else:
            data_start = t0

        if include_carryout:
            # Extend end forwards by l_max for trailing carryover
            data_end = t1 + _convert_frequency_to_timedelta(l_max, inferred_freq)
        else:
            data_end = t1

        # Extract data window (may need to truncate at data boundaries)
        X_window = self.X[
            (self.X[self.date_column] >= data_start) &
            (self.X[self.date_column] <= data_end)
        ]

        # 5b. Predict with actual spend (all channels active)
        Y_actual = self.sample_posterior_predictive(
            X=X_window,
            extend_idata=False,
            include_last_observations=include_carryin,  # Prepend historical if carryin=True
            original_scale=original_scale,
            var_names=["y"],
        )
        # Shape: (sample, date_extended)

        # 5c. Create counterfactual: zero spend on ALL channels during [t0, t1]
        X_counterfactual = X_window.copy()
        period_mask = (
            (X_counterfactual[self.date_column] >= t0) &
            (X_counterfactual[self.date_column] <= t1)
        )
        X_counterfactual.loc[period_mask, self.channel_columns] = 0

        # 5d. Predict with counterfactual (zero spend)
        Y_counterfactual = self.sample_posterior_predictive(
            X=X_counterfactual,
            extend_idata=False,
            include_last_observations=include_carryin,
            original_scale=original_scale,
            var_names=["y"],
        )
        # Shape: (sample, date_extended)

        # 5e. Compute incremental contribution
        # Extract evaluation window: [t0, data_end] (includes carryout if enabled)
        eval_window_mask = (
            (X_window[self.date_column] >= t0) &
            (X_window[self.date_column] <= data_end)
        )
        eval_dates = X_window.loc[eval_window_mask, self.date_column].values

        incremental = Y_actual.sel(date=eval_dates) - Y_counterfactual.sel(date=eval_dates)
        # Shape: (sample, date_eval)

        # Sum over evaluation window
        total_incremental = incremental.sum(dim="date")  # (sample,)

        # 5f. Compute total spend in period [t0, t1] (NOT extended window)
        spend_mask = (
            (self.X[self.date_column] >= t0) &
            (self.X[self.date_column] <= t1)
        )

        # For vectorized operation across all channels, create DataArray
        total_spend = xr.DataArray(
            self.X.loc[spend_mask, self.channel_columns].sum(axis=0).values,
            dims=["channel"],
            coords={"channel": self.channel_columns}
        )
        # Shape: (channel,)

        # 5g. Compute ROAS (broadcasting: (sample,) / (channel,) → (sample, channel))
        # Handle zero spend
        roas_period = xr.where(
            total_spend == 0,
            np.nan,
            total_incremental / total_spend
        )
        # Shape: (sample, channel)

        # Add period coordinate
        period_label = self._format_period_label(t0, t1, frequency)
        roas_period = roas_period.assign_coords(period=period_label).expand_dims("period")

        roas_results.append(roas_period)

    # 6. Concatenate all periods
    roas_da = xr.concat(roas_results, dim="period")
    # Final shape: (sample, period, channel)

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
    dates = pd.date_range(start, end, freq='D')  # Daily dates for full coverage

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
        period_end = period.to_timestamp(how='end')

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
) -> str:
    """Format period label for display."""

    if frequency == "all_time":
        return f"{t0.date()}_to_{t1.date()}"
    elif frequency == "yearly":
        return f"{t0.year}"
    elif frequency == "quarterly":
        return f"{t0.year}Q{t0.quarter}"
    elif frequency == "monthly":
        return f"{t0.year}-{t0.month:02d}"
    elif frequency == "weekly":
        week = t0.isocalendar()[1]
        return f"{t0.year}W{week:02d}"
```

### Optimized Vectorized Implementation (Advanced)

For maximum efficiency, use PyTensor's `vectorize_graph()` to compute all channels simultaneously:

```python
def compute_roas_vectorized(
    self,
    frequency: Literal["weekly", "monthly", "quarterly", "yearly", "all_time"],
    # ... other parameters ...
) -> xr.DataArray:
    """Compute ROAS with vectorized evaluation across all channels."""

    # Setup similar to above...

    for t0, t1 in periods:
        # 1. Extract response distribution conditioned on posterior
        from pymc_marketing.pytensor_utils import extract_response_distribution

        resp_graph = extract_response_distribution(
            pymc_model=self.model,
            idata=self.idata,
            response_variable="y",
        )

        # 2. Create batched counterfactual data
        # Scenario 0: baseline (all channels active)
        # Scenario 1-N: each channel set to zero individually
        X_window = self.X[...]  # Extract as before
        channel_data = X_window[self.channel_columns].values  # (n_dates, n_channels)
        n_channels = len(self.channel_columns)

        # Create batch: 1 baseline + n_channels counterfactuals
        batched_data = np.tile(channel_data[None, :, :], (n_channels + 1, 1, 1))
        # Shape: (n_channels+1, n_dates, n_channels)

        # Zero out each channel in its corresponding scenario
        period_mask = (...)  # Boolean mask for [t0, t1]
        for ch_idx in range(n_channels):
            batched_data[ch_idx + 1, period_mask, ch_idx] = 0

        # 3. Vectorize evaluation function
        channel_data_shared = self.model["channel_data"]

        channel_in = pt.tensor(
            name="channel_data_scenarios",
            dtype=channel_data_shared.dtype,
            shape=(None, *channel_data_shared.type.shape),  # (scenario, date, channel)
        )

        from pytensor.graph import vectorize_graph
        scenarios_graph = vectorize_graph(resp_graph, replace={channel_data_shared: channel_in})

        from pytensor import function
        eval_fn = function([channel_in], scenarios_graph)

        # 4. Evaluate all scenarios in single call
        predictions = eval_fn(batched_data)
        # Shape: (n_channels+1, n_samples, n_dates_extended)

        # 5. Compute incremental for each channel
        baseline = predictions[0]  # (n_samples, n_dates)
        counterfactuals = predictions[1:]  # (n_channels, n_samples, n_dates)

        # Broadcasting: (1, n_samples, n_dates) - (n_channels, n_samples, n_dates)
        incremental = baseline[None, :, :] - counterfactuals
        # Shape: (n_channels, n_samples, n_dates)

        # 6. Sum over evaluation window and compute ROAS
        total_incremental = incremental[:, :, eval_mask].sum(axis=2)  # (n_channels, n_samples)
        total_spend = ...  # (n_channels,)

        roas_period = total_incremental.T / total_spend  # (n_samples, n_channels)

        # Convert to xarray with proper coords
        roas_period = xr.DataArray(
            roas_period,
            dims=['sample', 'channel'],
            coords={
                'sample': np.arange(roas_period.shape[0]),
                'channel': self.channel_columns
            }
        )

        roas_results.append(roas_period)

    # Concatenate and return
    return xr.concat(roas_results, dim="period")
```

**Performance Benefit**: Single vectorized evaluation replaces N+1 separate `sample_posterior_predictive()` calls per period.

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

    Examples
    --------
    >>> # Get overall ROAS summary
    >>> summary.roas(frequency="all_time")

    >>> # Get quarterly ROAS with custom HDI levels
    >>> summary.roas(
    ...     frequency="quarterly",
    ...     hdi_probs=(0.05, 0.5, 0.95),
    ...     output_format="dataset"
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

    # Compute summary statistics with HDI
    df = self._compute_summary_stats_with_hdi(roas_data, hdi_probs)

    return self._convert_output(df, output_format)
```

### Testing Strategy

#### 1. Unit Tests

**Test counterfactual differs from simple division**:
```python
def test_roas_counterfactual_correct():
    """Verify counterfactual ROAS differs from old simple division."""
    mmm = create_fitted_multidimensional_mmm()

    # New method
    roas = mmm.compute_roas(frequency="all_time")

    # Old method (for comparison)
    contributions = mmm.idata.posterior.channel_contribution.mean(("chain", "draw"))
    spend = mmm.X[mmm.channel_columns].sum()
    roas_old = contributions.sum(dim="date") / spend

    # Should NOT be equal (old method is incorrect)
    assert not np.allclose(roas.mean(dim="sample"), roas_old)

    # New method should generally give lower ROAS (more conservative)
    # because it accounts for mixed contributions
    assert (roas.mean(dim="sample") < roas_old).all()
```

**Test frequency parameter**:
```python
def test_roas_frequency_aggregation():
    """Verify frequency parameter creates correct number of periods."""
    mmm = create_fitted_multidimensional_mmm(n_weeks=104)  # 2 years weekly

    roas_all = mmm.compute_roas(frequency="all_time")
    roas_yearly = mmm.compute_roas(frequency="yearly")
    roas_monthly = mmm.compute_roas(frequency="monthly")

    assert roas_all.sizes["period"] == 1
    assert roas_yearly.sizes["period"] == 2  # 2 years
    assert roas_monthly.sizes["period"] == 24  # 24 months
```

**Test carryover flags**:
```python
def test_roas_carryover_flags():
    """Verify carryin and carryout flags affect results correctly."""
    mmm = create_fitted_multidimensional_mmm()

    roas_full = mmm.compute_roas(
        frequency="all_time",
        include_carryin=True,
        include_carryout=True
    )

    roas_no_carryin = mmm.compute_roas(
        frequency="all_time",
        include_carryin=False,
        include_carryout=True
    )

    roas_no_carryout = mmm.compute_roas(
        frequency="all_time",
        include_carryin=True,
        include_carryout=False
    )

    roas_no_carry = mmm.compute_roas(
        frequency="all_time",
        include_carryin=False,
        include_carryout=False
    )

    # With carryover should generally be higher
    assert (roas_full.mean(dim="sample") >= roas_no_carry.mean(dim="sample")).all()

    # Carryin and carryout should have measurable impact
    assert not np.allclose(roas_full.mean(), roas_no_carryin.mean())
    assert not np.allclose(roas_full.mean(), roas_no_carryout.mean())
```

**Test zero spend handling**:
```python
def test_roas_handles_zero_spend():
    """Verify zero spend periods return NaN."""
    X, y = create_dataset_with_zero_spend(channel="tv", period="2023-Q1")

    mmm = MMM(...)
    mmm.fit(X, y)

    roas = mmm.compute_roas(frequency="quarterly")

    # Q1 TV ROAS should be NaN
    tv_q1_roas = roas.sel(period="2023Q1", channel="tv")
    assert np.isnan(tv_q1_roas.mean())
```

#### 2. Integration Tests

**Test with different adstock types**:
```python
@pytest.mark.parametrize("adstock_class", [
    GeometricAdstock,
    DelayedAdstock,
    WeibullPDFAdstock,
    WeibullCDFAdstock
])
def test_roas_with_different_adstock(adstock_class):
    """Verify ROAS works with all adstock types."""
    mmm = MMM(
        adstock=adstock_class(l_max=8),
        saturation=LogisticSaturation(),
        # ...
    )
    mmm.fit(X, y)

    roas = mmm.compute_roas(frequency="all_time")

    # Should produce valid results
    assert not roas.isnull().all()
    assert (roas > 0).any()  # At least some positive ROAS
```

**Test with time-varying media**:
```python
def test_roas_with_time_varying_media():
    """Verify ROAS works with time-varying media effects."""
    from pymc_marketing.hsgp_kwargs import HSGPKwargs

    mmm = MMM(
        time_varying_media=HSGPKwargs(m=10, L=1.5),
        # ...
    )
    mmm.fit(X, y)

    roas = mmm.compute_roas(frequency="quarterly")

    # Time-varying shouldn't break computation
    assert not roas.isnull().all()
```

**Test with multi-dimensional model**:
```python
def test_roas_multidimensional():
    """Verify ROAS works with additional dimensions (geo, etc.)."""
    mmm = MMM(
        date_column="date",
        channel_columns=["tv", "radio"],
        target_column="sales",
        dims=("country", "region"),  # Multi-dimensional
        # ...
    )
    mmm.fit(X_multi, y_multi)

    roas = mmm.compute_roas(frequency="all_time")

    # Should aggregate across all dimensions
    assert "country" not in roas.dims
    assert "region" not in roas.dims
    assert roas.dims == ("sample", "period", "channel")
```

#### 3. Validation Tests

**Synthetic data with known true ROAS**:
```python
def test_roas_with_known_ground_truth():
    """Test ROAS with synthetic data where true ROAS is known."""
    # Generate synthetic data with known contribution functions
    X_synth, y_synth, true_roas = generate_synthetic_mmm_data(
        n_periods=52,
        channels=["tv", "radio"],
        l_max=8,
        true_roas={"tv": 2.5, "radio": 1.8},
        adstock_alpha={"tv": 0.5, "radio": 0.3},
    )

    mmm = MMM(...)
    mmm.fit(X_synth, y_synth)

    computed_roas = mmm.compute_roas(frequency="all_time")

    # Computed ROAS should recover true ROAS within tolerance
    for channel in ["tv", "radio"]:
        roas_mean = computed_roas.sel(channel=channel).mean()
        relative_error = abs(roas_mean - true_roas[channel]) / true_roas[channel]
        assert relative_error < 0.20  # Within 20%
```

**Budget optimizer consistency**:
```python
def test_roas_consistency_with_budget_optimizer():
    """Verify ROAS rankings align with budget optimizer allocations."""
    mmm = MMM(...)
    mmm.fit(X, y)

    roas = mmm.compute_roas(frequency="all_time")
    roas_mean = roas.mean(dim="sample")

    # Run budget optimizer
    optimal_budget, _ = mmm.optimize_budget(budget=10000, num_periods=12)

    # Channels with higher ROAS should get more budget
    roas_rank = roas_mean.rank(dim="channel")
    budget_rank = optimal_budget.sum(dim="date").rank(dim="channel")

    # Ranks should be positively correlated (Spearman > 0.5)
    correlation = xr.corr(roas_rank, budget_rank)
    assert correlation > 0.5
```

### Documentation Requirements

#### 1. Updated Docstrings
- Method docstring with formula reference to Google paper
- Examples for common use cases (all_time, quarterly, with/without carryover)
- Performance notes explaining computational cost by frequency

#### 2. User Guide Section
Create `docs/source/guide/roas_computation.md`:
- Explain counterfactual approach conceptually
- Contrast with old simple division method
- Show examples with interpretation
- Discuss when to use different frequency parameters
- Performance considerations and best practices

#### 3. Migration Guide
Create `docs/source/migration/roas_counterfactual.md`:
- Explain breaking changes
- Show code migration examples
- Performance comparison (old vs new)
- Interpretation differences

#### 4. Notebook Tutorial
Update `docs/source/notebooks/mmm/mmm_roas.ipynb`:
- Demonstrate counterfactual ROAS computation
- Show impact of carryover flags
- Visualize ROAS over time at different frequencies
- Interpret results in business context

## Architecture Insights

### Why Simple Division is Wrong

The stored `channel_contribution[t, m]` includes effects from multiple spend periods due to adstock convolution:

```
channel_contribution[t, m] = saturation(adstock(spend[:, m]))[t]
adstock(spend)[t] = w[0]*spend[t] + w[1]*spend[t-1] + ... + w[l_max-1]*spend[t-l_max+1]
```

Therefore:
1. `contribution[t]` includes effects from past spend (t-l_max+1 through t)
2. `spend[t]` creates effects that extend into future (t through t+l_max-1)

Simple division `contribution[t] / spend[t]` incorrectly assumes:
- All of `contribution[t]` comes from `spend[t]` ❌
- `spend[t]` only affects `contribution[t]` ❌

### The Counterfactual Solution

To correctly measure the incremental effect of spend during [t0, t1]:

1. **Predict with channel active**: Full model prediction with actual spend
2. **Predict with channel removed**: Counterfactual with zero spend in [t0, t1]
3. **Compute difference**: `incremental = Y_actual - Y_counterfactual`
4. **Account for carryover**:
   - Carry-in: Prepend l_max historical observations
   - Carry-out: Extend evaluation window by l_max periods
5. **Aggregate**: Sum incremental over extended window, divide by period spend

### Efficiency Through Frequency Selection

Computing counterfactual ROAS requires posterior predictive evaluations:

| Frequency | Data (104 weeks) | Periods | Evaluations | Time |
|-----------|------------------|---------|-------------|------|
| all_time  | 104 weeks        | 1       | 2           | ~5s  |
| yearly    | 104 weeks        | 2       | 4           | ~10s |
| quarterly | 104 weeks        | 8       | 16          | ~40s |
| monthly   | 104 weeks        | 24      | 48          | ~2min |
| weekly    | 104 weeks        | 104     | 208         | ~10min |

**Recommendation**: Use "all_time" or "yearly" for typical use cases unless temporal variation is critical.

### Data Flow Summary

```
Input: frequency, period_start, period_end, carryover flags

1. Create period groups based on frequency
   └─> [(t0_1, t1_1), (t0_2, t1_2), ...]

2. For each period:
   a. Determine data window
      ├─ if include_carryin: start = t0 - l_max*freq
      └─ if include_carryout: end = t1 + l_max*freq

   b. Sample with actual spend
      ├─ Extract X[data_window]
      ├─ sample_posterior_predictive(X, include_last_observations=include_carryin)
      └─> Y_actual(sample, date_extended)

   c. Sample with counterfactual (zero spend in [t0, t1])
      ├─ X_counter = X.copy()
      ├─ X_counter.loc[t0:t1, channels] = 0
      ├─ sample_posterior_predictive(X_counter, include_last_observations=include_carryin)
      └─> Y_counterfactual(sample, date_extended)

   d. Compute incremental
      ├─ Extract eval window: [t0, end]
      ├─ incremental = Y_actual[eval] - Y_counterfactual[eval]
      └─> Sum over date: total_incremental(sample)

   e. Compute ROAS
      ├─ total_spend = X.loc[t0:t1, channels].sum()
      ├─ roas_period = total_incremental / total_spend
      └─> roas_period(sample, channel)

3. Concatenate all periods
   └─> roas(sample, period, channel)
```

## Code References

### Primary Files to Modify

- **`pymc_marketing/mmm/multidimensional.py`** - Add `compute_roas()` method to MultidimensionalMMM class
- **`pymc_marketing/data/idata/mmm_wrapper.py:315-342`** - Remove old `get_roas()` method
- **`pymc_marketing/mmm/summary.py:470-520`** - Replace `roas()` with new implementation

### Key Infrastructure to Use

- **`pymc_marketing/mmm/multidimensional.py:1800`** - `sample_posterior_predictive()` for predictions
- **`pymc_marketing/mmm/multidimensional.py:1730`** - `_set_xarray_data()` for updating model data
- **`pymc_marketing/mmm/multidimensional.py:1665`** - `_posterior_predictive_data_transformation()` for data prep
- **`pymc_marketing/mmm/multidimensional.py:1559`** - `_validate_date_overlap_with_include_last_observations()` for validation
- **`pymc_marketing/mmm/utils.py:181`** - `_convert_frequency_to_timedelta()` for frequency handling
- **`pymc_marketing/mmm/utils.py:268`** - Frequency inference from data
- **`pymc_marketing/data/idata/utils.py:187`** - Period mapping for aggregation

### Reference Implementations

- **`pymc_marketing/mmm/sensitivity_analysis.py:289-438`** - `run_sweep()` for vectorized counterfactual pattern
- **`pymc_marketing/pytensor_utils.py:264-341`** - `extract_response_distribution()` for graph extraction
- **`pymc_marketing/mmm/budget_optimizer.py:926-952`** - Carryover padding pattern

### Test Files to Update/Create

- **`tests/mmm/test_multidimensional.py`** - Add comprehensive ROAS tests
- **`tests/mmm/test_summary.py:976-980`** - Update `test_division_by_zero_in_roas_handled()`
- **`tests/mmm/test_summary.py:380-382`** - Update `test_roas_summary_schema()`
- **New**: `tests/mmm/test_roas_counterfactual.py` - Dedicated ROAS test suite

## Implementation Checklist

- [ ] Implement `compute_roas()` method in MultidimensionalMMM class
- [ ] Add `_create_period_groups()` helper for frequency-based grouping
- [ ] Add `_format_period_label()` helper for period display
- [ ] Remove old `get_roas()` from MMMIDataWrapper
- [ ] Update `roas()` in MMMSummaryFactory
- [ ] Write unit tests for core functionality
- [ ] Write integration tests with different adstock types
- [ ] Write validation tests with synthetic data
- [ ] Update docstrings with formula reference and examples
- [ ] Create user guide documentation
- [ ] Create migration guide
- [ ] Update ROAS notebook tutorial
- [ ] Performance profiling with different frequencies
- [ ] Validate time-varying media compatibility
- [ ] Validate multi-dimensional model compatibility

## Open Questions (Resolved)

### 1. Backward Compatibility
**Decision**: Remove old behavior entirely per user feedback.

### 2. Default Frequency
**Decision**: Make `frequency` parameter required (no default) to force users to consider computational cost.

### 3. Edge Period Handling
**Decision**: Truncate evaluation window at data boundaries with warning. Document limitation clearly.

### 4. MMM Class to Target
**Decision**: MultidimensionalMMM in `multidimensional.py` (NOT deprecated MMM class).

### 5. Data Access Pattern
**Decision**: Use `self.X` and `self.y` directly (already stored during fit), not `fit_data` extraction.

### 6. Channel Processing
**Decision**: Process all channels simultaneously using vectorized xarray operations and/or `vectorize_graph()`.

### 7. Frequency Handling
**Decision**: Detect data frequency using `pd.infer_freq()`, never assume weekly.

### 8. Carryover Flags
**Decision**: Split into `include_carryin` and `include_carryout` flags (both default True).

## Related Research

- **Google MMM Paper**: https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf (Formula 10)
- **Jin et al. (2017)**: "Bayesian methods for media mix modeling with carryover and shape effects"
- **Budget optimizer implementation**: Demonstrates correct carryover handling pattern

## Next Steps

1. **Implement core `compute_roas()` method** in MultidimensionalMMM class
   - Use counterfactual approach with `sample_posterior_predictive()`
   - Support frequency parameter with period grouping
   - Implement carryin/carryout flags
   - Handle arbitrary time frequencies

2. **Add comprehensive tests**
   - Unit tests for counterfactual logic
   - Integration tests with different adstock types
   - Validation with synthetic data
   - Multi-dimensional model tests
   - Time-varying media tests

3. **Update summary interface**
   - Remove old ROAS method
   - Add new ROAS method with required frequency parameter
   - Update schema and validation

4. **Documentation**
   - Method docstrings with formula reference
   - User guide explaining counterfactual approach
   - Migration guide for breaking changes
   - Notebook tutorial with examples

5. **Performance optimization** (optional)
   - Implement vectorized version using `vectorize_graph()`
   - Profile performance improvements
   - Add benchmarking tests

6. **Validation**
   - Statistical validation with synthetic data
   - Compare with budget optimizer consistency
   - Real-world case study validation
