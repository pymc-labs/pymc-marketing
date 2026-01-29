---
date: 2026-01-29T22:46:24+00:00
researcher: Claude Sonnet 4.5
git_commit: 75f94001de1854af87bc0c8a98813bec37f75621
branch: work-issue-2211
repository: pymc-labs/pymc-marketing
topic: "Compute ROAs the correct way (like in the Google paper)"
tags: [research, codebase, roas, adstock, carryover, mmm, budget-optimizer, counterfactual]
status: complete
last_updated: 2026-01-29
last_updated_by: Claude Sonnet 4.5
issue_number: 2211
---

# Research: Compute ROAs the correct way (like in the Google paper)

**Date**: 2026-01-29T22:46:24+00:00
**Researcher**: Claude Sonnet 4.5
**Git Commit**: 75f94001de1854af87bc0c8a98813bec37f75621
**Branch**: work-issue-2211
**Repository**: pymc-labs/pymc-marketing
**Issue**: #2211

## Research Question

How is ROAS currently computed in the codebase, and what changes are needed to implement formula (10) from the Google MMM paper (https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf) which uses a **counterfactual approach** considering the carryover window L (l_max in the code)?

## Summary

**CRITICAL CORRECTION**: The current ROAS implementation `ROAS[t] = contribution[t] / spend[t]` is incorrect, but **NOT** for the reason initially documented. The Google paper formula (10) requires a **counterfactual approach**, not simple summation over carryover windows.

### Google Paper Formula (10) Explained

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

**Key Insight**: ROAS for a channel over a period is the **difference** between:
1. Predicted KPI with actual spend on that channel
2. Predicted KPI with **zero** spend on that channel (counterfactual)

This difference captures the **incremental contribution** of the channel, accounting for carryover effects that extend L-1 periods beyond the spend window.

### Efficiency Considerations

Computing counterfactual ROAS for every individual time period would be computationally expensive. The user suggests:
- **Users typically want ROAS at lower frequencies** (yearly, quarterly) than data frequency (weekly, daily)
- **Optimization**: Compute ROAS directly at the requested frequency rather than computing per-period then aggregating
- Example: For yearly ROAS with weekly data, compute once per year instead of 52 times

The codebase already has robust time aggregation patterns (xarray resample, pandas groupby) that can support this.

## Problem Statement

### Current Implementation (Incorrect)
```python
ROAS[t] = contribution[t] / spend[t]
```
**Problem**: Treats each period independently, ignoring that contributions are mixtures of effects from multiple spend periods.

### Initially Proposed (Still Incorrect)
```python
ROAS[t] = sum(contribution[t:t+l_max]) / spend[t]
```
**Problem**: Still doesn't isolate the incremental effect of spend[t]. The contribution[t+i] includes effects from multiple spend periods, not just spend[t].

### Required Implementation (Counterfactual Approach)
```python
# For channel m over period [t0, t1]:
# 1. Predict KPI with actual spend on channel m
Y_actual = model.predict(X_actual)

# 2. Create counterfactual data: zero spend on channel m during [t0, t1]
X_counterfactual = X_actual.copy()
X_counterfactual.loc[t0:t1, channel_m] = 0

# 3. Predict KPI with counterfactual (zero) spend
Y_counterfactual = model.predict(X_counterfactual)

# 4. Sum differences over extended window [t0, t1+L-1] to capture carryover
incremental_contribution = sum(Y_actual[t0:t1+L] - Y_counterfactual[t0:t1+L])

# 5. Compute ROAS
ROAS_m = incremental_contribution / sum(spend[t0:t1])
```

## Detailed Findings

### 1. Current ROAS Implementation

#### Primary ROAS Computation
**File**: `pymc_marketing/data/idata/mmm_wrapper.py:315-342`

```python
def get_roas(self, original_scale: bool = True) -> xr.DataArray:
    """Compute ROAS (Return on Ad Spend) for each channel.

    ROAS = contribution / spend for each channel.
    """
    contributions = self.get_channel_contributions(original_scale=original_scale)
    spend = self.get_channel_spend()

    # Handle zero spend - use xr.where to avoid division by zero
    spend_safe = xr.where(spend == 0, np.nan, spend)

    return contributions / spend_safe  # Line 342: Element-wise division
```

**Critical Issue**:
- Line 342 performs element-wise division: `contribution[t] / spend[t]`
- Does **not** use counterfactual approach
- Does **not** account for carryover effects extending beyond spend period
- Incorrectly attributes mixed contributions to individual spend periods

#### ROAS Summary Interface
**File**: `pymc_marketing/mmm/summary.py:470-520`

```python
def roas(
    self,
    hdi_probs: Sequence[float] = (0.025, 0.5, 0.975),
    output_format: OutputFormatType = "dataframe",
) -> pd.DataFrame | xr.Dataset:
    """Computes ROAS = contribution / spend"""
    data = self._ensure_data_is_loaded()
    roas = data.get_roas(original_scale=True)  # Line 515: Delegates to wrapper
    df = self._compute_summary_stats_with_hdi(roas, hdi_probs)
    return self._convert_output(df, output_format)
```

This inherits the same issue from the wrapper.

### 2. How Channel Contributions are Actually Computed

#### Model Structure
**File**: `pymc_marketing/mmm/mmm.py:779-914`

The MMM computes the total KPI prediction additively:

```python
mu (scaled) = intercept + Σ(channel_contributions) + Σ(control_contributions) + seasonality
```

Each `channel_contribution[t, m]` is computed via:
1. **Adstock transformation** (geometric, delayed, Weibull, etc.) creates temporal dependencies
2. **Saturation transformation** (logistic, Hill, etc.) creates diminishing returns
3. **Time-varying effects** (optional) modulate contributions temporally

**Critical point**: The stored `channel_contribution` is NOT the marginal/incremental contribution of that channel. It's the contribution computed WITH all channels active. To get the incremental effect, we need counterfactual comparison.

#### Forward Pass
**File**: `pymc_marketing/mmm/mmm.py:492-518`

```python
def forward_pass(self, x: pt.TensorVariable | npt.NDArray) -> pt.TensorVariable:
    """Transform channel input into target contributions.

    Handles the ordering of adstock and saturation transformations.
    """
    first, second = (
        (self.adstock, self.saturation)
        if self.adstock_first
        else (self.saturation, self.adstock)
    )

    return second.apply(x=first.apply(x=x, dims="channel"), dims="channel")
```

The forward pass can be evaluated with **modified inputs** (e.g., zero spend for specific channels), enabling counterfactual predictions.

### 3. Counterfactual Prediction Capabilities

The codebase has extensive infrastructure for counterfactual analysis:

#### Method 1: `sample_posterior_predictive()` with Modified Data
**File**: `pymc_marketing/mmm/mmm.py:2467-2546`

```python
def sample_posterior_predictive(
    self,
    X=None,
    extend_idata: bool = True,
    combined: bool = True,
    include_last_observations: bool = False,
    original_scale: bool = True,
    **sample_posterior_predictive_kwargs,
) -> DataArray:
    """Sample from the model's posterior predictive distribution.

    Can accept modified X data with channels set to zero for counterfactuals.
    """
    if include_last_observations:
        # Prepend last l_max observations to capture carryover from historical spend
        X = pd.concat(
            [self.X.iloc[-self.adstock.l_max :, :], X], axis=0
        ).sort_values(by=self.date_column)

    self._data_setter(X)  # Updates model data via pm.set_data()

    with self.model:
        post_pred = pm.sample_posterior_predictive(
            self.idata, **sample_posterior_predictive_kwargs
        )
```

**Usage for counterfactual ROAS**:
```python
# 1. Predict with actual spend
Y_actual = mmm.sample_posterior_predictive(
    X_actual,
    var_names=["y"],
    include_last_observations=True,  # Important for carryover!
    original_scale=True
)

# 2. Create counterfactual: zero spend on target channel
X_counterfactual = X_actual.copy()
X_counterfactual["tv"] = 0

# 3. Predict with counterfactual
Y_counterfactual = mmm.sample_posterior_predictive(
    X_counterfactual,
    var_names=["y"],
    include_last_observations=True,
    original_scale=True
)

# 4. Compute incremental contribution
incremental = Y_actual - Y_counterfactual

# 5. Aggregate over period and compute ROAS
```

#### Method 2: `channel_contribution_forward_pass()`
**File**: `pymc_marketing/mmm/mmm.py:956-1025, 1439-1476`

```python
def channel_contribution_forward_pass(
    self,
    channel_data: npt.NDArray,
    disable_logger_stdout: bool | None = False,
) -> npt.NDArray:
    """Evaluate channel contribution for given channel data (forward pass).

    Can be used with zero spend for specific channels to compute counterfactuals.
    """
```

This creates a new PyMC model context with custom channel data and samples contributions directly.

#### Method 3: `new_spend_contributions()`
**File**: `pymc_marketing/mmm/mmm.py:1876-1992`

```python
def new_spend_contributions(
    self,
    spend: np.ndarray | pd.Series | pd.DataFrame,
    one_time: bool = True,
    spend_leading_up: np.ndarray | pd.DataFrame | None = None,
    prior: bool = False,
    original_scale: bool = True,
    **sample_posterior_predictive_kwargs,
) -> DataArray:
    """Compute contributions for hypothetical new spend scenarios."""
```

Useful for scenario analysis, though designed for new spend rather than counterfactuals.

#### Related: Sensitivity Analysis
**File**: `pymc_marketing/mmm/sensitivity_analysis.py:105-497`

The `SensitivityAnalysis` class provides systematic intervention analysis:
- `run_sweep()`: Evaluate model with systematic parameter changes
- `compute_uplift_curve_respect_to_base()`: Compute uplift vs baseline
- `compute_marginal_effects()`: Differentiate to get marginal ROI

### 4. Time Aggregation Infrastructure

The codebase has robust support for time aggregation:

#### XArray Resample for InferenceData
**File**: `pymc_marketing/data/idata/utils.py:136-213`

```python
def aggregate_idata_time(
    idata: az.InferenceData,
    period: Literal["weekly", "monthly", "quarterly", "yearly", "all_time"],
    method: Literal["sum", "mean"] = "sum",
) -> az.InferenceData:
    """Aggregate InferenceData over time periods using xarray resample."""
```

#### Wrapper Method
**File**: `pymc_marketing/data/idata/mmm_wrapper.py:548-576`

```python
def aggregate_time(
    self,
    period: Literal["weekly", "monthly", "quarterly", "yearly", "all_time"],
    method: Literal["sum", "mean"] = "sum",
) -> "MMMIDataWrapper":
    """Aggregate data over time periods."""
```

#### Summary Factory Integration
**File**: `pymc_marketing/mmm/summary.py:287-327, 399-468`

```python
def contributions(
    self,
    hdi_probs: Sequence[float] | None = None,
    component: Literal["channel", "control", "seasonality", "baseline"] = "channel",
    frequency: Frequency | None = None,  # "weekly", "monthly", "quarterly", "yearly"
    output_format: OutputFormat | None = None,
) -> DataFrameType:
    """Create contribution summary at specified frequency."""
```

**Key insight**: The `frequency` parameter already exists in the summary API and aggregates data before computing statistics. This pattern should be extended to ROAS computation.

### 5. Adstock and Carryover Window

#### Adstock Base Class
**File**: `pymc_marketing/mmm/components/adstock.py:84-176`

```python
class AdstockTransformation(BaseModel, ABC):
    """Base class for adstock transformations."""

    def __init__(
        self,
        l_max: int = Field(..., gt=0, description="Maximum lag for adstock"),
        # ...
    ) -> None:
        self.l_max = l_max  # Carryover window length
```

All adstock types (Geometric, Delayed, Weibull) have `l_max` parameter accessible via `model.adstock.l_max`.

#### Convolution Creates Carryover
**File**: `pymc_marketing/mmm/transformers.py:212-297`

```python
def geometric_adstock(
    x,
    alpha: float = 0.0,
    l_max: int = 12,
    normalize: bool = False,
    axis: int = 0,
    mode: ConvMode = ConvMode.After,
) -> TensorVariable:
    """Geometric adstock transformation.

    The cumulative media effect is a weighted average of current and past spend.
    l_max is the maximum duration of carryover effect.
    """
```

With `ConvMode.After` (default), spend at time `t` affects contributions at times `t, t+1, ..., t+l_max-1`.

### 6. Budget Optimizer: Carryover Handling Reference

**File**: `pymc_marketing/mmm/budget_optimizer.py:896-952`

```python
def _replace_channel_data_by_optimization_variable(self, model: Model) -> Model:
    """Replace channel_data with optimization variable."""
    num_periods = self.num_periods
    max_lag = self.mmm_model.adstock.l_max  # Line 899: Access l_max

    # Pad optimization horizon with l_max extra periods to capture carryover
    repeated_budgets_with_carry_over_shape.insert(
        date_dim_idx, num_periods + max_lag  # Line 930
    )
```

This demonstrates the correct pattern: extend the evaluation window by `l_max` periods to capture trailing carryover effects.

## Architecture Insights

### Why Current Implementation is Wrong

The stored `channel_contribution[t, m]` is computed as:
```
channel_contribution[t, m] = saturation(adstock(spend[:, m]))[t]
```

Due to adstock convolution:
```
adstock(spend)[t] = w[0]*spend[t] + w[1]*spend[t-1] + ... + w[l_max-1]*spend[t-l_max+1]
```

Therefore, `channel_contribution[t, m]` includes effects from `spend[t-l_max+1:t+1, m]`.

Simply dividing `contribution[t] / spend[t]` incorrectly assumes:
1. All of `contribution[t]` comes from `spend[t]` (ignores past spend effects)
2. `spend[t]` only affects `contribution[t]` (ignores future effects)

### The Counterfactual Solution

To correctly attribute the effect of `spend[t0:t1, m]`:

1. **Recognize that spend effects extend beyond spend period**: Spend during `[t0, t1]` creates contributions through `[t0, t1+l_max-1]`

2. **Use incremental contribution**: Compare predicted KPI with vs. without the channel active

3. **Account for carryover**: Extend evaluation window by `l_max-1` periods

4. **Aggregate properly**: Sum the incremental effects and divide by total spend

### Data Flow for Counterfactual ROAS

```
Input: Time period [t0, t1], target channel m

Step 1: Prepare data for extended window [t0, t1+l_max-1]
  - If t1+l_max-1 > T (end of data), use available data up to T
  - Include last l_max observations before t0 for historical carryover

Step 2: Predict with actual spend
  Y_actual[t0:t1+l_max] = model.sample_posterior_predictive(
      X[t0-l_max:t1+l_max],
      var_names=["y"]
  )

Step 3: Create counterfactual
  X_counter = X.copy()
  X_counter[t0:t1, channel_m] = 0

Step 4: Predict with counterfactual
  Y_counter[t0:t1+l_max] = model.sample_posterior_predictive(
      X_counter[t0-l_max:t1+l_max],
      var_names=["y"]
  )

Step 5: Compute incremental contribution (across posterior samples)
  incremental[t0:t1+l_max] = Y_actual[t0:t1+l_max] - Y_counter[t0:t1+l_max]

Step 6: Aggregate and compute ROAS
  ROAS_m = sum(incremental[t0:t1+l_max]) / sum(spend[t0:t1, m])
```

### Efficient Implementation Strategy

#### Challenge: Computational Cost
Computing counterfactual ROAS for every time period requires:
- 2 × T posterior predictive samplings (actual + counterfactual)
- For C channels: 2 × T × C samplings
- With weekly data over 2 years: 2 × 104 × C samplings

#### Solution: Frequency-Aware Computation

**Observation**: Users typically request ROAS at lower frequencies than data frequency.

**Strategy**:
1. Accept a `frequency` parameter: "weekly", "monthly", "quarterly", "yearly", "all_time"
2. Compute ROAS directly at requested frequency instead of per-period
3. Use vectorization where possible

**Example**: Yearly ROAS with weekly data
- **Naive**: Compute 52 weekly ROAS values, then aggregate → 2 × 52 samplings per channel
- **Efficient**: Compute 1 yearly ROAS directly → 2 samplings per channel

**Implementation**:
```python
def get_roas(
    self,
    original_scale: bool = True,
    frequency: Literal["original", "weekly", "monthly", "quarterly", "yearly", "all_time"] = "original",
) -> xr.DataArray:
    """Compute ROAS using counterfactual approach at specified frequency."""

    if frequency == "original":
        # Compute at original data frequency (most expensive)
        periods = range(len(self.dates))
    else:
        # Group dates by requested frequency
        periods = self._get_period_groups(frequency)

    roas_results = []

    for period_dates in periods:
        t0, t1 = period_dates[0], period_dates[-1]
        l_max = self._get_l_max()

        # Extend window for carryover
        eval_window = self._extend_window(t0, t1, l_max)

        for channel in self.channels:
            # Compute incremental contribution for this channel and period
            incremental = self._compute_incremental_contribution(
                channel, t0, t1, eval_window
            )

            # Compute ROAS
            total_spend = self.get_channel_spend().sel(
                date=slice(t0, t1), channel=channel
            ).sum()

            roas = incremental.sum(dim="date") / total_spend
            roas_results.append(roas)

    return xr.concat(roas_results, dim="date")
```

### Vectorization Opportunities

**Challenge**: Computing counterfactuals for all channels separately is inefficient.

**Optimization**: Compute all counterfactuals in parallel by setting multiple channels to zero simultaneously and using xarray's broadcasting.

**Not recommended**: The marginal cost of separate channel predictions may not justify the complexity of vectorized implementation, especially given frequency-based optimization above.

## Implementation Recommendations

### Required Changes

#### 1. Add Counterfactual ROAS Method to MMMIDataWrapper
**File**: `pymc_marketing/data/idata/mmm_wrapper.py`

Add new method `get_roas_counterfactual()`:

```python
def get_roas_counterfactual(
    self,
    frequency: Literal["original", "weekly", "monthly", "quarterly", "yearly", "all_time"] = "original",
    include_carryover: bool = True,
) -> xr.DataArray:
    """Compute ROAS using counterfactual approach from Google MMM paper.

    Parameters
    ----------
    frequency : str
        Time frequency for ROAS computation. Computing at lower frequencies
        (e.g., "yearly" vs "original") is much more efficient.
    include_carryover : bool
        If True, extends evaluation window by l_max periods to capture
        carryover effects. Should generally be True for accurate ROAS.

    Returns
    -------
    xr.DataArray
        ROAS with dims (chain, draw, period, channel)
    """
```

This requires access to:
- The fitted MMM model instance (to call `sample_posterior_predictive`)
- Original input data X
- Model's adstock l_max parameter

**Problem**: `MMMIDataWrapper` only has access to `InferenceData`, not the original model. Need to either:
- Store model reference in wrapper during initialization
- Move ROAS computation to the MMM class itself
- Store necessary info (l_max, X) in InferenceData attributes

#### 2. Add ROAS Method to MMM Class
**File**: `pymc_marketing/mmm/mmm.py`

Add method to MMM class (preferred location):

```python
def compute_roas(
    self,
    period_start: str | pd.Timestamp | None = None,
    period_end: str | pd.Timestamp | None = None,
    frequency: Literal["original", "weekly", "monthly", "quarterly", "yearly", "all_time"] = "all_time",
    channels: list[str] | None = None,
    include_carryover: bool = True,
    original_scale: bool = True,
) -> xr.DataArray:
    """Compute ROAS using counterfactual approach.

    Implements Google MMM paper formula (10): compares predicted KPI with
    actual spend vs. zero spend for each channel, accounting for carryover.

    Parameters
    ----------
    period_start, period_end : str or Timestamp, optional
        Period for ROAS computation. If None, uses entire fitted data period.
    frequency : str
        Aggregation frequency. Lower frequencies are more efficient.
        - "all_time": Single ROAS across entire period
        - "yearly", "quarterly", "monthly": Aggregate by period
        - "original": Per-period ROAS (most expensive)
    channels : list of str, optional
        Channels to compute ROAS for. If None, computes for all channels.
    include_carryover : bool
        If True, extends evaluation window by l_max periods to capture
        trailing carryover effects from spend in the period.
    original_scale : bool
        If True, returns ROAS in original target scale.

    Returns
    -------
    xr.DataArray
        ROAS with dims (chain, draw, period, channel)

    Examples
    --------
    >>> # Overall ROAS for all channels
    >>> roas = mmm.compute_roas(frequency="all_time")
    >>>
    >>> # Quarterly ROAS for TV and Radio
    >>> roas = mmm.compute_roas(
    ...     frequency="quarterly",
    ...     channels=["tv", "radio"]
    ... )
    """
```

#### 3. Update Summary Interface
**File**: `pymc_marketing/mmm/summary.py`

Update the `roas()` method to use counterfactual approach:

```python
def roas(
    self,
    hdi_probs: Sequence[float] | None = None,
    frequency: Frequency | None = "all_time",  # Add frequency parameter
    output_format: OutputFormat | None = None,
    method: Literal["simple", "counterfactual"] = "counterfactual",  # Deprecation path
) -> DataFrameType:
    """Compute ROAS with summary statistics.

    Parameters
    ----------
    hdi_probs : sequence of float, optional
        HDI probability levels for credible intervals
    frequency : str, optional
        Time aggregation frequency. Defaults to "all_time" for overall ROAS.
        Options: "original", "weekly", "monthly", "quarterly", "yearly", "all_time"
    output_format : str, optional
        Output format ("dataframe" or "dataset")
    method : str, optional
        ROAS computation method:
        - "counterfactual" (default): Google paper formula with counterfactuals
        - "simple" (deprecated): Old element-wise division method

    Returns
    -------
    DataFrame or Dataset
        ROAS summary with mean, HDI intervals, and other statistics
    """
```

#### 4. Deprecate Old Method

Add deprecation warning to old `get_roas()`:

```python
def get_roas(self, original_scale: bool = True) -> xr.DataArray:
    """Compute ROAS (Return on Ad Spend) for each channel.

    .. deprecated:: X.X.X
        This method uses element-wise division which doesn't account for
        adstock carryover effects. Use `get_roas_counterfactual()` instead
        for correct ROAS computation following the Google MMM paper.

    ROAS = contribution / spend for each channel.
    """
    warnings.warn(
        "This ROAS computation method is deprecated and may give incorrect "
        "results due to ignoring adstock carryover effects. Use "
        "`get_roas_counterfactual()` for accurate ROAS computation.",
        DeprecationWarning,
        stacklevel=2,
    )
    # ... existing implementation ...
```

### Implementation Pseudocode

```python
def compute_roas(
    self,
    period_start=None,
    period_end=None,
    frequency="all_time",
    channels=None,
    include_carryover=True,
    original_scale=True,
) -> xr.DataArray:
    """Compute ROAS using counterfactual approach."""

    # 1. Determine period groups based on frequency
    if period_start is None:
        period_start = self.X[self.date_column].min()
    if period_end is None:
        period_end = self.X[self.date_column].max()

    periods = self._create_period_groups(
        period_start, period_end, frequency
    )  # Returns list of (t0, t1) tuples

    # 2. Determine channels to compute
    if channels is None:
        channels = self.channel_columns

    # 3. Get l_max for carryover window
    l_max = self.adstock.l_max if include_carryover else 0

    # 4. Compute ROAS for each period and channel
    roas_results = []

    for t0, t1 in periods:
        for channel in channels:
            # 4a. Prepare data windows
            # Include l_max before t0 for historical carryover into period
            # Include l_max after t1 for carryover from period spend
            data_start = t0 - pd.Timedelta(weeks=l_max)
            data_end = t1 + pd.Timedelta(weeks=l_max)

            X_window = self.X[
                (self.X[self.date_column] >= data_start) &
                (self.X[self.date_column] <= data_end)
            ]

            # 4b. Predict with actual spend
            Y_actual = self.sample_posterior_predictive(
                X_window,
                var_names=["y"],
                original_scale=original_scale,
                extend_idata=False,
            )

            # 4c. Create counterfactual: zero spend on target channel in [t0, t1]
            X_counter = X_window.copy()
            counter_mask = (
                (X_counter[self.date_column] >= t0) &
                (X_counter[self.date_column] <= t1)
            )
            X_counter.loc[counter_mask, channel] = 0

            # 4d. Predict with counterfactual
            Y_counter = self.sample_posterior_predictive(
                X_counter,
                var_names=["y"],
                original_scale=original_scale,
                extend_idata=False,
            )

            # 4e. Compute incremental contribution
            # Only sum over [t0, t1+l_max] (not before t0)
            eval_mask = (
                (X_window[self.date_column] >= t0) &
                (X_window[self.date_column] <= data_end)
            )

            incremental = (Y_actual - Y_counter).sel(date=eval_mask)
            total_incremental = incremental.sum(dim="date")

            # 4f. Compute total spend in period [t0, t1]
            spend_mask = (
                (self.X[self.date_column] >= t0) &
                (self.X[self.date_column] <= t1)
            )
            total_spend = self.X.loc[spend_mask, channel].sum()

            # 4g. Compute ROAS
            if total_spend == 0:
                roas = xr.full_like(total_incremental, np.nan)
            else:
                roas = total_incremental / total_spend

            # Store result
            roas = roas.assign_coords(
                period=f"{t0.date()}_to_{t1.date()}",
                channel=channel
            )
            roas_results.append(roas)

    # 5. Concatenate all results
    roas_da = xr.concat(roas_results, dim="period_channel")
    # Reshape to (chain, draw, period, channel)
    # ... dimensional manipulation ...

    return roas_da
```

### Testing Strategy

#### 1. Unit Tests

**Test counterfactual computation**:
```python
def test_roas_counterfactual_vs_simple():
    """Verify counterfactual ROAS differs from simple division."""
    mmm = # ... fitted model ...

    roas_simple = mmm.data.get_roas(original_scale=True)
    roas_counter = mmm.compute_roas(frequency="all_time")

    # Should NOT be equal (simple is wrong)
    assert not np.allclose(
        roas_simple.mean(),
        roas_counter.mean()
    )
```

**Test frequency parameter**:
```python
def test_roas_frequency_aggregation():
    """Verify frequency parameter reduces computation."""
    mmm = # ... fitted model with 104 weeks ...

    roas_weekly = mmm.compute_roas(frequency="original")
    roas_yearly = mmm.compute_roas(frequency="yearly")

    # Yearly should have fewer periods
    assert roas_yearly.sizes["period"] < roas_weekly.sizes["period"]
    assert roas_yearly.sizes["period"] == 2  # 2 years
```

**Test carryover inclusion**:
```python
def test_roas_includes_carryover():
    """Verify carryover effects are captured."""
    mmm = # ... fitted model ...

    roas_with_carry = mmm.compute_roas(include_carryover=True)
    roas_without_carry = mmm.compute_roas(include_carryover=False)

    # With carryover should generally be higher (captures trailing effects)
    assert roas_with_carry.mean() > roas_without_carry.mean()
```

**Test zero spend handling**:
```python
def test_roas_handles_zero_spend():
    """Verify zero spend periods return NaN."""
    X_with_zero = X.copy()
    X_with_zero.loc["2023-Q1", "tv"] = 0

    mmm.fit(X_with_zero, y)
    roas = mmm.compute_roas(frequency="quarterly")

    # Q1 TV ROAS should be NaN
    assert np.isnan(roas.sel(period="2023-Q1", channel="tv").mean())
```

#### 2. Integration Tests

**Test with different adstock types**:
```python
@pytest.mark.parametrize("adstock_type", [
    "geometric",
    "delayed",
    "weibull_pdf",
    "weibull_cdf"
])
def test_roas_with_different_adstock(adstock_type):
    """Verify ROAS works with all adstock types."""
    mmm = create_mmm_with_adstock(adstock_type)
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
    mmm = MMM(time_varying_media=True, ...)
    mmm.fit(X, y)
    roas = mmm.compute_roas(frequency="quarterly")

    # Time-varying shouldn't break computation
    assert not roas.isnull().all()
```

#### 3. Validation Tests

**Compare with budget optimizer**:
```python
def test_roas_consistency_with_budget_optimizer():
    """Verify ROAS is consistent with budget optimizer results."""
    # Budget optimizer correctly handles carryover
    # Its optimal allocations should align with ROAS rankings

    mmm.fit(X, y)
    roas = mmm.compute_roas(frequency="all_time")

    # Channels with higher ROAS should get more budget in optimizer
    optimal_budget = mmm.optimize_budget(...)

    roas_rank = roas.mean(dim=["chain", "draw"]).rank(dim="channel")
    budget_rank = optimal_budget.rank(dim="channel")

    # Ranks should be positively correlated
    assert correlation(roas_rank, budget_rank) > 0.5
```

**Synthetic data test**:
```python
def test_roas_with_known_ground_truth():
    """Test ROAS computation with synthetic data with known true ROAS."""
    # Generate synthetic data with known contribution functions
    X_synth, y_synth, true_roas = generate_synthetic_mmm_data(
        n_periods=52,
        channels=["tv", "radio"],
        l_max=8,
        true_roas={"tv": 2.5, "radio": 1.8}
    )

    mmm.fit(X_synth, y_synth)
    computed_roas = mmm.compute_roas(frequency="all_time")

    # Computed ROAS should be close to true ROAS
    for channel in ["tv", "radio"]:
        roas_mean = computed_roas.sel(channel=channel).mean()
        assert abs(roas_mean - true_roas[channel]) / true_roas[channel] < 0.2  # Within 20%
```

### Documentation Updates

#### 1. Docstrings
- Update `MMM.compute_roas()` with detailed docstring
- Include mathematical formula reference to Google paper
- Provide examples for common use cases
- Explain computational cost implications of frequency parameter

#### 2. User Guide
Create new documentation section:
- `docs/source/guide/roas_computation.md`
- Explain counterfactual approach conceptually
- Show examples with interpretation
- Discuss when to use different frequency parameters
- Performance considerations

#### 3. Notebooks
Update or create notebooks:
- `docs/source/notebooks/mmm/mmm_roas.ipynb` - ROAS computation tutorial
- Show comparison between old and new method
- Demonstrate frequency parameter usage
- Interpret results in business context

#### 4. API Reference
- Update API reference for `MMMSummaryFactory.roas()`
- Add migration guide from old to new method
- Document backward compatibility approach

## Open Questions

### 1. Backward Compatibility Strategy

**Question**: How to handle breaking change?

**Options**:
- **Option A**: Keep old method with deprecation warning, add new method
  - Pro: No breaking changes immediately
  - Con: API confusion, two methods for same concept
- **Option B**: Make counterfactual the default, add flag for old behavior
  - Pro: Cleaner API
  - Con: Breaking change
- **Option C**: Add `method` parameter to existing ROAS functions
  - Pro: Single API, explicit choice
  - Con: More complex implementation

**Recommendation**: Option A for next minor release, Option B for next major release.

### 2. Default Frequency

**Question**: What should be the default frequency for ROAS computation?

**Options**:
- `"original"`: Matches current behavior (per-period ROAS)
  - Pro: Backward compatible
  - Con: Expensive, often not what users want
- `"all_time"`: Single ROAS across entire dataset
  - Pro: Fast, often sufficient
  - Con: May be too aggregated for some use cases
- Make it required (no default)
  - Pro: Forces users to think about what they want
  - Con: More friction in API

**Recommendation**: Default to `"all_time"` for compute_roas(), maintain `"original"` for summary.roas() for backward compatibility.

### 3. Edge Period Handling

**Question**: How to handle periods where `t1 + l_max > T` (end of data)?

**Options**:
- **Option A**: Truncate evaluation window at T
  - Pro: Simple
  - Con: Underestimates ROAS for late periods (missing carryover)
- **Option B**: Set ROAS to NaN for periods with incomplete carryover
  - Pro: Conservative, honest about uncertainty
  - Con: Loses information
- **Option C**: Extrapolate or pad with zeros
  - Pro: Complete information
  - Con: Introduces assumptions

**Recommendation**: Option A (truncate) with a warning if carryover window extends beyond data. Document this limitation.

### 4. Time-Varying Effects Interaction

**Question**: Do time-varying media effects affect counterfactual computation?

**Analysis**:
- Time-varying effects multiply baseline contribution: `contribution[t] = baseline[t] * temporal_multiplier[t]`
- In counterfactual, we set channel spend to zero
- The temporal multiplier is still applied to the (now zero) baseline contribution
- Result: Temporal multipliers correctly carry through to counterfactual

**Conclusion**: No special handling needed, works correctly as-is.

### 5. Multidimensional MMM Support

**Question**: Does the implementation work with multidimensional MMM?

**Analysis**:
- Multidimensional MMM has same structure, just additional dimensions
- `sample_posterior_predictive()` works the same way
- Channel dimension may be hierarchical (e.g., platform × creative)

**Recommendation**: Test with multidimensional models, may need dimension-aware grouping for frequency aggregation.

### 6. Computational Optimization

**Question**: Can we optimize counterfactual computation further?

**Possible optimizations**:
- **Batch all channels**: Set all channels to zero in one pass, use matrix operations
  - Requires careful dimension handling
  - May not save much time vs frequency optimization
- **Caching**: Cache actual predictions, only compute counterfactuals
  - Useful if computing ROAS for multiple channels separately
- **Approximations**: Use point estimates instead of full posterior
  - Loses uncertainty quantification
  - Not recommended for Bayesian workflow

**Recommendation**: Start with straightforward implementation using frequency optimization. Profile and optimize if needed.

## Related Research

- Google MMM Paper: https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf (Formula 10)
- Jin et al. (2017): "Bayesian methods for media mix modeling with carryover and shape effects"
- Budget optimizer implementation: `pymc_marketing/mmm/budget_optimizer.py` (carryover handling reference)

## Code References

### Primary Files to Modify
- `pymc_marketing/mmm/mmm.py` - Add `compute_roas()` method (MAIN LOCATION)
- `pymc_marketing/data/idata/mmm_wrapper.py:315-342` - Deprecate old `get_roas()`
- `pymc_marketing/mmm/summary.py:470-520` - Update `roas()` to use counterfactual

### Key Infrastructure to Use
- `pymc_marketing/mmm/mmm.py:2467` - `sample_posterior_predictive()` for predictions
- `pymc_marketing/mmm/mmm.py:1176` - `_data_setter()` for updating model data
- `pymc_marketing/data/idata/utils.py:136` - `aggregate_idata_time()` for frequency aggregation
- `pymc_marketing/mmm/components/adstock.py:84` - Access `l_max` from adstock

### Reference Implementations
- `pymc_marketing/mmm/budget_optimizer.py:896-952` - Carryover padding pattern
- `pymc_marketing/mmm/sensitivity_analysis.py:289` - Intervention analysis pattern
- `pymc_marketing/mmm/summary.py:399-468` - Frequency parameter integration

### Test Files to Update
- `tests/mmm/test_summary.py:976-980` - `test_division_by_zero_in_roas_handled()`
- `tests/mmm/test_summary.py:380-382` - `test_roas_summary_schema()`
- Add new test file: `tests/mmm/test_roas_counterfactual.py`

## Next Steps

1. **Implement `compute_roas()` method in MMM class**
   - Use counterfactual approach with `sample_posterior_predictive()`
   - Support frequency parameter for efficient computation
   - Handle carryover window extension

2. **Add unit tests**
   - Test counterfactual vs simple division (should differ)
   - Test frequency parameter (fewer periods at lower frequencies)
   - Test carryover inclusion
   - Test zero spend handling
   - Test edge cases (incomplete carryover windows)

3. **Add integration tests**
   - Test with different adstock types
   - Test with time-varying media effects
   - Test with multidimensional models
   - Validate against budget optimizer consistency

4. **Update documentation**
   - Docstrings with mathematical formula
   - User guide explaining counterfactual approach
   - Notebook tutorial with examples
   - Migration guide from old method

5. **Deprecate old method**
   - Add deprecation warning to `get_roas()`
   - Update summary factory to use new method
   - Document breaking change in release notes

6. **Performance testing**
   - Profile with different frequencies
   - Validate that frequency optimization provides expected speedup
   - Test with large datasets (multiple years of weekly data)

7. **Review and validation**
   - Code review focusing on correctness of counterfactual logic
   - Statistical validation with synthetic data
   - Comparison with budget optimizer results for consistency
