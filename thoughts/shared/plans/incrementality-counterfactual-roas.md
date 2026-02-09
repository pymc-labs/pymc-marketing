# Incrementality Module: Counterfactual ROAS Implementation Plan

## Overview

Implement a new `Incrementality` module that computes Return on Ad Spend (ROAS), Customer Acquisition Cost (CAC), and incremental contributions using proper counterfactual analysis. This replaces the current element-wise division approach with a method that accounts for adstock carryover effects, following the Google MMM paper's Formula (10).

**Reference Research**: `thoughts/shared/research/2026-02-06_14-21-06_channel-attribution-architecture.md`

## Current State Analysis

### What Exists Now

**Current ROAS Implementation** ([mmm_wrapper.py:315-343](pymc_marketing/data/idata/mmm_wrapper.py#L315-L343)):
```python
def get_roas(self, original_scale: bool = True) -> xr.DataArray:
    contributions = self.get_channel_contributions(original_scale=original_scale)
    spend = self.get_channel_spend()
    spend_safe = xr.where(spend == 0, np.nan, spend)
    return contributions / spend_safe
```

**Problem**: Element-wise division `ROAS[t] = contribution[t] / spend[t]` treats each time period independently, ignoring that:
- Spend at time t affects outcomes at t, t+1, ..., t+L via adstock
- To measure full impact, we must sum contributions over the carryover window
- Need counterfactual: "What if channel was off during evaluation period?"

**Affected Methods**:
- [mmm_wrapper.py:315-343](pymc_marketing/data/idata/mmm_wrapper.py#L315-L343) - `MMMIDataWrapper.get_roas()`
- [summary.py:470-520](pymc_marketing/mmm/summary.py#L470-L520) - `MMMSummaryFactory.roas()`

### Key Discoveries

1. **Vectorization Infrastructure Exists** ([pytensor_utils.py:264-341](pymc_marketing/pytensor_utils.py#L264-L341)):
   - `extract_response_distribution()` is battle-tested, used by `SensitivityAnalysis` and `BudgetOptimizer`
   - Can efficiently evaluate multiple scenarios across posterior samples

2. **Property Pattern Established** ([multidimensional.py:2159-2185](pymc_marketing/mmm/multidimensional.py#L2159-L2185)):
   - MMM exposes functionality via properties: `.plot`, `.summary`, `.sensitivity`
   - Each returns a specialized class with focused responsibilities

3. **Frequency Handling Well-Defined** ([summary.py:45-47](pymc_marketing/mmm/summary.py#L45-L47)):
   - `Frequency` type alias: `"original" | "weekly" | "monthly" | "quarterly" | "yearly" | "all_time"`
   - Uses modern pandas 2.2+ offset aliases (`"ME"`, `"QE"`, `"YE"`)
   - Period-end labeling convention

4. **SensitivityAnalysis Provides Reusable Patterns** ([sensitivity_analysis.py:289-438](pymc_marketing/mmm/sensitivity_analysis.py#L289-L438)):
   - Dimension management utilities
   - Posterior subsampling with quadratic formula
   - xarray transformation helpers

## Desired End State

### Functional Requirements

After implementation, users can:

```python
# Compute quarterly ROAS with carryover effects
roas = mmm.incrementality.contribution_over_spend(
    frequency="quarterly",
    period_start="2024-01-01",
    period_end="2024-12-31",
    include_carryin=True,   # Account for pre-period carryover
    include_carryout=True,  # Account for post-period carryover
)

# Compute monthly CAC
cac = mmm.incrementality.spend_over_contribution(
    frequency="monthly"
)

# Core incrementality function for custom analysis
incremental = mmm.incrementality.compute_incremental_contribution(
    frequency="weekly",
    period_start="2024-Q1",
    period_end="2024-Q2",
)

# Existing summary interface works correctly
df = mmm.summary.roas(frequency="monthly")  # Now uses counterfactual approach
```

### Technical Requirements

1. **Output Shape**: `(chain, draw, date, channel, *custom_dims)` or `(chain, draw, channel, *custom_dims)` for `frequency="all_time"`
2. **Performance**: Vectorized evaluation of all channels and scenarios simultaneously
3. **Carryover Logic**:
   - `include_carryin=True`: Prepend last `l_max` periods to capture historical effects
   - `include_carryout=True`: Extend window by `l_max` periods to capture trailing effects
4. **Frequency Support**: All standard frequencies via established `Frequency` type alias
5. **Dimension Handling**: Works with `MultidimensionalMMM` and hierarchical dimensions (geo, country, region)

### Verification

**Automated**:
- [ ] Type checking passes: `mypy pymc_marketing/mmm/incrementality.py`
- [ ] Linting passes: `ruff check pymc_marketing/mmm/incrementality.py`
- [ ] Unit tests pass: `pytest tests/mmm/test_incrementality.py -v`
- [ ] Integration tests pass
- [ ] Existing tests still pass: `pytest tests/mmm/test_summary.py -k roas`

**Manual**:
- [ ] ROAS values are higher than element-wise approach (carryover captured)
- [ ] Quarterly ROAS with carryout > without carryout (validates carryover logic)
- [ ] `frequency="all_time"` removes date dimension
- [ ] Works with hierarchical models (test with `dims=("country",)`)

## What We're NOT Doing

**Out of Scope** (future work):

1. **Memory Optimization**: No chunking/batching for very large models initially
   - Add if users report OOM errors
   - Would implement `max_scenarios_per_batch` parameter

2. **Shared Utility Extraction**: Not creating `counterfactual_core.py` initially
   - Assess duplication after Phase 2
   - Refactor if >100 lines of similar code with `SensitivityAnalysis`

3. **Advanced Features**:
   - Interaction effects isolation
   - Synergy quantification
   - Competitive effects modeling
   - Uncertainty propagation in spend data

4. **Backward Compatibility Layer**: No deprecation warnings, just replace implementation
   - Clean break, not gradual migration

5. **Time-Varying Media Special Handling**: Assume `extract_response_distribution()` handles correctly
   - Add tests to verify, but no special code paths initially

## Implementation Approach

**Strategy**: Implement core incrementality logic using proven vectorization patterns from `SensitivityAnalysis`, then replace existing ROAS computation.

**Key Technical Approach**:

1. **Leverage `extract_response_distribution()`**: Get posterior-conditioned response graph
2. **Vectorize over scenarios**: Use `vectorize_graph()` to evaluate all counterfactuals at once
3. **Per-Period Evaluation**: For each period, compare baseline (actual spend) vs counterfactual (zero spend)
4. **Temporal Aggregation**: Sum incrementals over carryover window, accounting for carry-in/out

**Pattern** (from research):
```python
# 1. Extract response graph (batched over posterior samples)
response_graph = extract_response_distribution(model, idata, "y")

# 2. Create scenarios: [baseline_period1, counterfactual_period1, baseline_period2, ...]
scenario_array = create_all_scenarios(periods, channel_data)

# 3. Vectorize and compile
evaluator = compile_scenario_evaluator(response_graph, "channel_data", model)

# 4. Evaluate all scenarios in single call
predictions = evaluator(scenario_array)

# 5. Compute incrementals: baseline - counterfactual per period
incrementals = predictions[::2] - predictions[1::2]

# 6. Aggregate and transform to xarray
result = aggregate_by_frequency(incrementals, periods)
```

---

## Phase 1: Core Incrementality Implementation

### Overview

Create the `Incrementality` class with three methods and integrate into MMM.

### Changes Required

#### 1. Create `pymc_marketing/mmm/incrementality.py`

**New file**: Complete implementation of `Incrementality` class

**Key Components**:

**A. Class Structure**:
```python
"""Incrementality and counterfactual analysis for Marketing Mix Models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from pytensor import function
from pytensor.graph.basic import vectorize_graph
import pytensor.tensor as pt

if TYPE_CHECKING:
    from pymc_marketing.mmm.multidimensional import MMM

# Import from existing modules
from pymc_marketing.mmm.summary import Frequency
from pymc_marketing.pytensor_utils import extract_response_distribution


class Incrementality:
    """Incrementality and counterfactual analysis for MMM models.

    Computes incremental channel contributions by comparing predictions with
    actual spend vs. counterfactual (zero) spend, accounting for
    adstock carryover effects.

    Parameters
    ----------
    model : MMM
        Fitted MMM model instance
    idata : az.InferenceData
        InferenceData containing posterior samples and fit data
    """

    def __init__(self, model: MMM, idata: az.InferenceData):
        self.model = model
        self.idata = idata
        self.pymc_model = model.model

    def compute_incremental_contribution(
        self,
        frequency: Frequency,
        period_start: str | pd.Timestamp | None = None,
        period_end: str | pd.Timestamp | None = None,
        include_carryin: bool = True,
        include_carryout: bool = True,
        original_scale: bool = True,
    ) -> xr.DataArray:
        """Compute incremental channel contributions using counterfactual analysis.

        [Full docstring from research doc lines 606-714]
        """
        # Implementation here (see detailed pseudocode below)
        pass

    def contribution_over_spend(
        self,
        frequency: Frequency,
        period_start: str | pd.Timestamp | None = None,
        period_end: str | pd.Timestamp | None = None,
        include_carryin: bool = True,
        include_carryout: bool = True,
    ) -> xr.DataArray:
        """Compute contribution per unit of spend (ROAS, customers per dollar, etc.).

        [Full docstring from research doc lines 719-781]
        """
        # Get incremental contributions
        incremental = self.compute_incremental_contribution(
            frequency=frequency,
            period_start=period_start,
            period_end=period_end,
            include_carryin=include_carryin,
            include_carryout=include_carryout,
            original_scale=True,
        )

        # Get spend aggregated by same frequency
        spend = self._get_spend_by_frequency(
            frequency=frequency,
            period_start=period_start,
            period_end=period_end,
        )

        # Divide, handling zero spend
        spend_safe = xr.where(spend == 0, np.nan, spend)
        return incremental / spend_safe

    def spend_over_contribution(
        self,
        frequency: Frequency,
        period_start: str | pd.Timestamp | None = None,
        period_end: str | pd.Timestamp | None = None,
        include_carryin: bool = True,
        include_carryout: bool = True,
    ) -> xr.DataArray:
        """Compute spend per unit of contribution (CAC, cost per unit, etc.).

        [Full docstring from research doc lines 830-903]
        """
        ratio = self.contribution_over_spend(
            frequency=frequency,
            period_start=period_start,
            period_end=period_end,
            include_carryin=include_carryin,
            include_carryout=include_carryout,
        )
        return 1.0 / ratio

    # Helper methods
    def _get_spend_by_frequency(
        self,
        frequency: Frequency,
        period_start: str | pd.Timestamp | None,
        period_end: str | pd.Timestamp | None,
    ) -> xr.DataArray:
        """Get spend data aggregated by frequency."""
        # Implementation here
        pass

    def _create_period_groups(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        frequency: Frequency,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Create list of (period_start, period_end) tuples for given frequency."""
        # Implementation here (see research doc lines 1059-1097)
        pass
```

**B. Core Implementation - `compute_incremental_contribution()`**:

Follow the detailed algorithm from research doc lines 913-1058:

```python
def compute_incremental_contribution(self, ...) -> xr.DataArray:
    # 1. Get fit_data and date range
    fit_data = self.idata.fit_data
    dates = pd.to_datetime(fit_data.coords["date"].values)

    # Resolve period bounds
    if period_start is None:
        period_start = dates[0]
    else:
        period_start = pd.to_datetime(period_start)

    if period_end is None:
        period_end = dates[-1]
    else:
        period_end = pd.to_datetime(period_end)

    # 2. Extract response distribution
    response_graph = extract_response_distribution(
        pymc_model=self.pymc_model,
        idata=self.idata,
        response_variable="y",  # Or "mu" depending on model
    )
    # Shape: (chain*draw, date, *custom_dims)

    # 3. Get l_max for carryover calculations
    l_max = self.model.adstock.l_max
    inferred_freq = pd.infer_freq(dates)

    # 4. Create period groups
    periods = self._create_period_groups(period_start, period_end, frequency)
    # Returns: [(t0_1, t1_1), (t0_2, t1_2), ...] or [(t0, t1)] for "all_time"

    # 5. Build all scenarios for vectorized evaluation
    all_scenarios = []
    period_labels = []

    for period_idx, (t0, t1) in enumerate(periods):
        # Determine data window with carryover padding
        if include_carryin:
            data_start = t0 - pd.Timedelta(l_max * inferred_freq)
        else:
            data_start = t0

        if include_carryout:
            data_end = t1 + pd.Timedelta(l_max * inferred_freq)
        else:
            data_end = t1

        # Extract data window
        fit_window = fit_data.sel(date=slice(data_start, data_end))

        # Get channel data: (date_window, channel, *custom_dims)
        channel_data = fit_window[self.model.channel_columns].to_array(dim="channel")
        channel_data = channel_data.transpose("date", "channel", ...)

        # Create counterfactual: zero out channels during [t0, t1]
        counterfactual = channel_data.copy(deep=True)
        eval_mask = (fit_window.date >= t0) & (fit_window.date <= t1)
        counterfactual.loc[{"date": eval_mask}] = 0  # Zero all channels in eval period

        # Add both scenarios
        all_scenarios.append(channel_data.values)      # Baseline
        all_scenarios.append(counterfactual.values)    # Counterfactual

        period_labels.append(t1)  # Period-end date convention

    # 6. Stack all scenarios: (n_scenarios, date_window, channel, *custom_dims)
    scenario_array = np.stack(all_scenarios, axis=0)
    n_periods = len(periods)

    # 7. Create vectorized evaluator
    data_shared = self.pymc_model["channel_data"]

    channel_in = pt.tensor(
        name="channel_data_scenarios",
        dtype=data_shared.dtype,
        shape=(None, *data_shared.type.shape),  # (scenario, date, channel, *dims)
    )

    scenario_graph = vectorize_graph(response_graph, replace={data_shared: channel_in})
    evaluator = function([channel_in], scenario_graph)

    # 8. Evaluate all scenarios at once
    all_predictions = evaluator(scenario_array)
    # Shape: (n_scenarios, n_samples, date_window, *custom_dims)

    # 9. Compute incrementals per period
    results = []

    for period_idx in range(n_periods):
        baseline_idx = period_idx * 2
        counter_idx = period_idx * 2 + 1

        baseline_pred = all_predictions[baseline_idx]  # (n_samples, date_window, *dims)
        counter_pred = all_predictions[counter_idx]

        # Incremental contribution
        incremental = baseline_pred - counter_pred

        # Sum over evaluation window (including carryout if enabled)
        t0, t1 = periods[period_idx]
        if include_carryout:
            carryout_end = t1 + pd.Timedelta(l_max * inferred_freq)
            eval_dates = (dates >= t0) & (dates <= carryout_end)
        else:
            eval_dates = (dates >= t0) & (dates <= t1)

        # Get indices for eval_dates in the window
        window_dates = fit_data.sel(date=slice(data_start, data_end)).date.values
        eval_indices = np.isin(window_dates, dates[eval_dates])

        total_incremental = incremental[:, eval_indices].sum(axis=1)
        # Shape: (n_samples, *custom_dims)

        # Convert to xarray with coordinates
        total_incremental_da = xr.DataArray(
            total_incremental,
            dims=("sample", *self.model.dims),
            coords={
                "sample": np.arange(total_incremental.shape[0]),
                **{dim: fit_data.coords[dim].values for dim in self.model.dims},
            },
        )

        # Add period coordinate and expand date dimension
        total_incremental_da = total_incremental_da.assign_coords(date=period_labels[period_idx])
        total_incremental_da = total_incremental_da.expand_dims("date")

        results.append(total_incremental_da)

    # 10. Concatenate all periods
    if frequency == "all_time":
        # Single period, remove date dimension
        result = results[0].squeeze("date", drop=True)
    else:
        result = xr.concat(results, dim="date")

    # 11. Add channel dimension
    result = result.expand_dims({"channel": self.model.channel_columns})

    # 12. Ensure standard dimension order: (sample, date, channel, *custom_dims)
    dim_order = ["sample", "date", "channel", *self.model.dims]
    if frequency == "all_time":
        dim_order.remove("date")
    result = result.transpose(*dim_order)

    # 13. Scale to original if needed
    if original_scale:
        # Get target scale from model or idata
        if hasattr(self.idata, "constant_data") and "target_scale" in self.idata.constant_data:
            target_scale = float(self.idata.constant_data["target_scale"].values)
            result = result * target_scale

    return result
```

**C. Helper Method - `_get_spend_by_frequency()`**:

```python
def _get_spend_by_frequency(
    self,
    frequency: Frequency,
    period_start: str | pd.Timestamp | None,
    period_end: str | pd.Timestamp | None,
) -> xr.DataArray:
    """Get spend data aggregated by frequency.

    Returns xarray with dims matching incremental contribution output.
    """
    fit_data = self.idata.fit_data

    # Filter date range if specified
    if period_start is not None or period_end is not None:
        fit_data = fit_data.sel(date=slice(period_start, period_end))

    # Get channel spend data
    channel_data_list = [fit_data[ch] for ch in self.model.channel_columns]

    # Stack channels: (date, channel, *custom_dims)
    spend = xr.concat(
        channel_data_list,
        dim=pd.Index(self.model.channel_columns, name="channel"),
    )

    # Aggregate by frequency
    if frequency == "all_time":
        spend = spend.sum(dim="date")
    elif frequency != "original":
        # Use pandas resampling
        period_map = {
            "weekly": "W",
            "monthly": "ME",
            "quarterly": "QE",
            "yearly": "YE",
        }
        freq = period_map[frequency]
        spend = spend.resample(date=freq).sum()

    return spend
```

**D. Helper Method - `_create_period_groups()`**:

```python
def _create_period_groups(
    self,
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: Frequency,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Create list of (period_start, period_end) tuples for given frequency.

    Returns period boundaries for aggregation. Periods are labeled with
    period-end dates following established convention.
    """
    if frequency == "all_time":
        return [(start, end)]

    if frequency == "original":
        # Use original data frequency - one period per observation
        dates = pd.date_range(
            start,
            end,
            freq=pd.infer_freq(self.idata.fit_data.date.values)
        )
        return [(d, d) for d in dates]

    # Map frequency to pandas period
    freq_map = {
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "Y",
    }

    # Create date range and convert to periods
    dates = pd.date_range(start, end, freq="D")
    periods = dates.to_period(freq_map[frequency])
    unique_periods = periods.unique()

    period_ranges = []
    for period in unique_periods:
        period_start = period.to_timestamp()
        period_end = period.to_timestamp(how="end")  # Period-end date

        # Clip to requested range
        period_start = max(period_start, start)
        period_end = min(period_end, end)

        period_ranges.append((period_start, period_end))

    return period_ranges
```

**Success Criteria**:
- [ ] File exists: `pymc_marketing/mmm/incrementality.py`
- [ ] All three public methods implemented with full docstrings
- [ ] Helper methods implemented
- [ ] Type hints are complete
- [ ] Imports are correct
- [ ] No syntax errors: `python -m py_compile pymc_marketing/mmm/incrementality.py`

#### 2. Add Property to MMM Class

**File**: [pymc_marketing/mmm/multidimensional.py](pymc_marketing/mmm/multidimensional.py)

**Location**: After the `.sensitivity` property (around line 2186)

**Add**:
```python
@property
def incrementality(self) -> Incrementality:
    """Access incrementality and counterfactual analysis functionality.

    Returns an Incrementality instance that can be used to compute
    incremental contributions, ROAS, and CAC using proper counterfactual
    analysis that accounts for adstock carryover effects.

    Returns
    -------
    Incrementality
        An instance configured with this MMM model for computing
        incremental contributions and efficiency metrics.

    Examples
    --------
    >>> # Compute quarterly ROAS with carryover effects
    >>> roas = mmm.incrementality.contribution_over_spend(
    ...     frequency="quarterly",
    ...     include_carryin=True,
    ...     include_carryout=True,
    ... )
    >>>
    >>> # Compute monthly CAC (Cost of Acquisition)
    >>> cac = mmm.incrementality.spend_over_contribution(
    ...     frequency="monthly",
    ... )
    >>>
    >>> # Core incrementality function for custom analysis
    >>> incremental = mmm.incrementality.compute_incremental_contribution(
    ...     frequency="weekly",
    ...     period_start="2024-01-01",
    ...     period_end="2024-03-31",
    ... )

    Notes
    -----
    This uses a counterfactual approach that correctly handles temporal
    incrementality: spend at time t affects outcomes at t, t+1, ..., t+L
    via adstock effects. The incrementality is computed by comparing
    predictions with actual spend vs. zero spend.

    See Also
    --------
    Incrementality : The incrementality analysis class
    """
    from pymc_marketing.mmm.incrementality import Incrementality

    self._validate_idata_exists()

    return Incrementality(model=self, idata=self.idata)
```

**Success Criteria**:
- [ ] Property added after `.sensitivity` property
- [ ] Import statement uses lazy import (inside method)
- [ ] Validation call included
- [ ] Type hint added to return type
- [ ] Docstring includes examples

#### 3. Update `summary.roas()` Method

**File**: [pymc_marketing/mmm/summary.py](pymc_marketing/mmm/summary.py)

**Location**: Lines 470-520

**Replace implementation**:

```python
def roas(
    self,
    hdi_probs: Sequence[float] | None = None,
    frequency: Frequency | None = None,
    output_format: OutputFormat | None = None,
    include_carryin: bool = True,
    include_carryout: bool = True,
) -> DataFrameType:
    """Create ROAS summary DataFrame using counterfactual analysis.

    Computes Return on Ad Spend (ROAS) using proper counterfactual approach
    that accounts for adstock carryover effects. This replaces the previous
    element-wise division method.

    Parameters
    ----------
    hdi_probs : Sequence[float], optional
        HDI probability levels. Defaults to factory's configured value.
    frequency : Frequency, optional
        Time aggregation frequency. One of:
        - "original": No aggregation (default)
        - "weekly", "monthly", "quarterly", "yearly": Aggregate to period
        - "all_time": Single value across entire period
    output_format : OutputFormat, optional
        Output format ("pandas" or "polars"). Defaults to factory's value.
    include_carryin : bool, default=True
        Include impact of pre-period channel spend via adstock carryover.
    include_carryout : bool, default=True
        Include impact of evaluation period spend carrying into post-period.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Summary statistics with columns:
        - Dimension coordinates (date, channel, custom dims)
        - mean: Mean ROAS across posterior
        - median: Median ROAS across posterior
        - abs_error_{prob}_lower: HDI lower bound
        - abs_error_{prob}_upper: HDI upper bound

    Notes
    -----
    **New in this version**: This method now uses counterfactual analysis
    via the Incrementality module. The computation compares actual predictions
    vs. predictions with zero channel spend, properly accounting for temporal
    carryover effects.

    **Previous behavior**: Used element-wise division (contribution[t] / spend[t])
    which ignored carryover. New values will typically be slightly higher.

    Examples
    --------
    >>> # Monthly ROAS with default carryover handling
    >>> df = mmm.summary.roas(frequency="monthly")
    >>>
    >>> # Quarterly ROAS without carryover effects (for comparison)
    >>> df = mmm.summary.roas(
    ...     frequency="quarterly",
    ...     include_carryin=False,
    ...     include_carryout=False,
    ... )
    """
    # Prepare data and resolve defaults
    data, hdi_probs, output_format = self._prepare_data_and_hdi(
        hdi_probs, frequency, output_format
    )

    # Use incrementality module for correct ROAS computation
    roas_data = self.model.incrementality.contribution_over_spend(
        frequency=frequency or "original",
        include_carryin=include_carryin,
        include_carryout=include_carryout,
    )

    # Compute summary stats with HDI
    df = self._compute_summary_stats_with_hdi(roas_data, hdi_probs)

    return self._convert_output(df, output_format)
```

**Success Criteria**:
- [ ] Method signature updated with new parameters
- [ ] Docstring updated explaining counterfactual approach
- [ ] Implementation delegates to `self.model.incrementality.contribution_over_spend()`
- [ ] Existing return type and structure preserved

#### 4. Deprecate or Update `mmm_wrapper.get_roas()`

**File**: [pymc_marketing/data/idata/mmm_wrapper.py](pymc_marketing/data/idata/mmm_wrapper.py)

**Location**: Lines 315-343

**Option A - Add Deprecation Notice** (Recommended):

```python
def get_roas(self, original_scale: bool = True) -> xr.DataArray:
    """Compute ROAS using element-wise division (deprecated).

    .. deprecated:: X.Y.Z
        This method uses element-wise division (contribution[t] / spend[t])
        which does not account for adstock carryover effects. Use
        ``model.incrementality.contribution_over_spend()`` for proper
        counterfactual ROAS computation.

    For backward compatibility, this method remains available but is
    not recommended for new analyses.

    See Also
    --------
    mmm.incrementality.contribution_over_spend : Recommended method
    """
    import warnings

    warnings.warn(
        "MMMIDataWrapper.get_roas() uses element-wise division which ignores "
        "adstock carryover effects. Use model.incrementality.contribution_over_spend() "
        "for proper counterfactual ROAS computation.",
        DeprecationWarning,
        stacklevel=2,
    )

    contributions = self.get_channel_contributions(original_scale=original_scale)
    spend = self.get_channel_spend()
    spend_safe = xr.where(spend == 0, np.nan, spend)
    return contributions / spend_safe
```

**Option B - Remove Method Entirely**:

Comment out or delete the method, update any internal usages.

**Recommendation**: Use Option A for Phase 1 (add deprecation), then remove in Phase 3 cleanup.

**Success Criteria**:
- [ ] Deprecation warning added
- [ ] Docstring updated with deprecation notice
- [ ] See Also section points to new method

---

## Phase 2: Testing & Validation

### Overview

Comprehensive test coverage for the new incrementality module.

### Test File Structure

**Create**: `tests/mmm/test_incrementality.py`

```python
"""Tests for incrementality and counterfactual ROAS computation."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.incrementality import Incrementality


class TestIncrementality:
    """Test suite for Incrementality class."""

    # Fixtures
    @pytest.fixture
    def simple_mmm(self):
        """Create a simple fitted MMM for testing."""
        # Build and fit a minimal MMM
        pass

    @pytest.fixture
    def hierarchical_mmm(self):
        """Create MMM with hierarchical dimensions."""
        # Build MMM with dims=("country",)
        pass

    # Unit Tests
    def test_initialization(self, simple_mmm):
        """Test Incrementality class initialization."""
        incr = Incrementality(model=simple_mmm, idata=simple_mmm.idata)
        assert incr.model is simple_mmm
        assert incr.idata is simple_mmm.idata
        assert incr.pymc_model is simple_mmm.model

    def test_compute_incremental_contribution_original_frequency(self, simple_mmm):
        """Test compute_incremental_contribution with original frequency."""
        incr = simple_mmm.incrementality
        result = incr.compute_incremental_contribution(frequency="original")

        # Check output shape
        assert "sample" in result.dims
        assert "date" in result.dims
        assert "channel" in result.dims

        # Check all channels present
        assert len(result.channel) == len(simple_mmm.channel_columns)

    def test_compute_incremental_contribution_monthly(self, simple_mmm):
        """Test with monthly aggregation."""
        incr = simple_mmm.incrementality
        result = incr.compute_incremental_contribution(frequency="monthly")

        # Monthly should have fewer time points than daily
        n_months = len(pd.date_range(
            simple_mmm.idata.fit_data.date[0].values,
            simple_mmm.idata.fit_data.date[-1].values,
            freq="ME"
        ))
        assert len(result.date) == n_months

    def test_compute_incremental_contribution_all_time(self, simple_mmm):
        """Test with all_time frequency (no date dimension)."""
        incr = simple_mmm.incrementality
        result = incr.compute_incremental_contribution(frequency="all_time")

        # Should not have date dimension
        assert "date" not in result.dims
        assert "sample" in result.dims
        assert "channel" in result.dims

    def test_carryover_flags(self, simple_mmm):
        """Test include_carryin and include_carryout flags."""
        incr = simple_mmm.incrementality

        # With carryover
        with_carryover = incr.compute_incremental_contribution(
            frequency="quarterly",
            include_carryin=True,
            include_carryout=True,
        )

        # Without carryover
        no_carryover = incr.compute_incremental_contribution(
            frequency="quarterly",
            include_carryin=False,
            include_carryout=False,
        )

        # With carryout should be >= without (captures trailing effects)
        # Mean across samples for comparison
        mean_with = with_carryover.mean(dim="sample")
        mean_no = no_carryover.mean(dim="sample")

        # At least some channels should show difference
        assert not np.allclose(mean_with.values, mean_no.values)

    def test_period_filtering(self, simple_mmm):
        """Test period_start and period_end filtering."""
        incr = simple_mmm.incrementality

        dates = pd.to_datetime(simple_mmm.idata.fit_data.date.values)
        mid_point = dates[len(dates) // 2]

        result = incr.compute_incremental_contribution(
            frequency="original",
            period_start=dates[0],
            period_end=mid_point,
        )

        # Should only include first half of dates
        assert len(result.date) <= len(dates) // 2

    def test_contribution_over_spend(self, simple_mmm):
        """Test contribution_over_spend (ROAS) computation."""
        incr = simple_mmm.incrementality
        roas = incr.contribution_over_spend(frequency="monthly")

        # Check shape
        assert "sample" in roas.dims
        assert "channel" in roas.dims

        # ROAS should be positive for most samples (assuming positive contributions)
        mean_roas = roas.mean(dim="sample")
        assert (mean_roas > 0).any()

        # Should handle zero spend gracefully (NaN)
        # (Test if any zeros exist in spend)

    def test_spend_over_contribution(self, simple_mmm):
        """Test spend_over_contribution (CAC) computation."""
        incr = simple_mmm.incrementality
        cac = incr.spend_over_contribution(frequency="monthly")

        # CAC should be reciprocal of ROAS
        roas = incr.contribution_over_spend(frequency="monthly")

        # Check reciprocal relationship (where both are finite)
        finite_mask = np.isfinite(roas) & np.isfinite(cac)
        if finite_mask.any():
            np.testing.assert_allclose(
                (1.0 / roas).where(finite_mask),
                cac.where(finite_mask),
                rtol=1e-6,
            )

    def test_hierarchical_dimensions(self, hierarchical_mmm):
        """Test with hierarchical model (country dimension)."""
        incr = hierarchical_mmm.incrementality
        result = incr.compute_incremental_contribution(frequency="monthly")

        # Should have country dimension
        assert "country" in result.dims
        assert len(result.country) > 1

    def test_zero_spend_handling(self, simple_mmm):
        """Test handling of periods with zero spend."""
        incr = simple_mmm.incrementality
        roas = incr.contribution_over_spend(frequency="monthly")

        # Any NaN values should correspond to zero spend
        # (Implementation detail: NaN where spend == 0)
        if np.isnan(roas.values).any():
            # Verify NaN handling is correct
            assert True  # Detailed check depends on test data

    def test_output_scale(self, simple_mmm):
        """Test original_scale parameter."""
        incr = simple_mmm.incrementality

        scaled = incr.compute_incremental_contribution(
            frequency="monthly",
            original_scale=False,
        )

        original = incr.compute_incremental_contribution(
            frequency="monthly",
            original_scale=True,
        )

        # Original scale should be larger if target was scaled
        # (Only true if target_scale != 1)
        # Check that they differ if scaling was applied
        if hasattr(simple_mmm.idata.constant_data, "target_scale"):
            target_scale = float(simple_mmm.idata.constant_data.target_scale.values)
            if not np.isclose(target_scale, 1.0):
                ratio = (original / scaled).mean()
                assert np.isclose(ratio, target_scale, rtol=0.01)


class TestIncrementalityIntegration:
    """Integration tests comparing to known results."""

    def test_comparison_to_manual_loop(self, simple_mmm):
        """Compare vectorized implementation to manual loop."""
        # Implement manual loop version for comparison
        # This tests that vectorization produces same results as sequential
        pass

    def test_higher_than_element_wise(self, simple_mmm):
        """Verify counterfactual ROAS >= element-wise ROAS."""
        # Due to carryover, counterfactual approach should capture more value

        # Get new counterfactual ROAS
        new_roas = simple_mmm.incrementality.contribution_over_spend(
            frequency="all_time",
            include_carryin=True,
            include_carryout=True,
        )

        # Get old element-wise ROAS
        contributions = simple_mmm.data.get_channel_contributions(original_scale=True)
        spend = simple_mmm.data.get_channel_spend()
        old_roas = (contributions / spend).sum(dim="date")

        # New should be >= old (captures carryover effects)
        # Mean across samples for comparison
        new_mean = new_roas.mean(dim="sample")
        old_mean = old_roas.mean(dim=("chain", "draw"))

        # At least some channels should show improvement
        # (May not be true for all channels depending on adstock)
        improvements = new_mean > old_mean
        assert improvements.any()

    def test_summary_integration(self, simple_mmm):
        """Test that summary.roas() works with new implementation."""
        df = simple_mmm.summary.roas(frequency="monthly")

        # Should return DataFrame
        assert hasattr(df, "columns")

        # Should have expected columns
        assert "mean" in df.columns
        assert "median" in df.columns
        assert "channel" in df.columns or df.index.name == "channel"


class TestPeriodGrouping:
    """Test helper method _create_period_groups."""

    def test_weekly_grouping(self, simple_mmm):
        """Test weekly period grouping."""
        incr = simple_mmm.incrementality

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-31")

        periods = incr._create_period_groups(start, end, "weekly")

        # Should have ~4 weeks
        assert len(periods) >= 4
        assert len(periods) <= 5

        # Each period should be a tuple of (start, end)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in periods)

    def test_all_time_grouping(self, simple_mmm):
        """Test all_time returns single period."""
        incr = simple_mmm.incrementality

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-12-31")

        periods = incr._create_period_groups(start, end, "all_time")

        assert len(periods) == 1
        assert periods[0] == (start, end)
```

### Integration Test Specifications

**Test 1: Validate Against Known Example**

Create a simple synthetic model where ROAS can be computed analytically:
- Single channel
- No saturation (linear)
- Simple geometric adstock with known decay
- Compute expected ROAS manually
- Verify implementation matches

**Test 2: Compare to Sequential Loop**

Implement a reference implementation using sequential evaluation:
- Loop over each period
- For each period, create baseline and counterfactual scenarios
- Call model prediction separately for each
- Compare results to vectorized implementation

**Test 3: End-to-End User Flow**

```python
def test_end_to_end_user_flow():
    # Build and fit MMM
    mmm = MMM(...)
    mmm.build_model(...)
    mmm.fit(...)

    # Compute ROAS via incrementality
    roas = mmm.incrementality.contribution_over_spend(frequency="quarterly")

    # Compute via summary interface
    df = mmm.summary.roas(frequency="quarterly")

    # Verify consistency
    assert ...
```

### Success Criteria

#### Automated Verification:
- [ ] All unit tests pass: `pytest tests/mmm/test_incrementality.py -v`
- [ ] Test coverage >90%: `pytest --cov=pymc_marketing.mmm.incrementality`
- [ ] Integration tests pass
- [ ] Existing ROAS tests updated and passing: `pytest tests/mmm/test_summary.py -k roas`
- [ ] No regressions: `pytest tests/mmm/`

#### Manual Verification:
- [ ] ROAS values from `mmm.incrementality` are reasonable
- [ ] Carryout effects captured (values with carryout > without)
- [ ] Hierarchical dimensions work correctly
- [ ] Summary interface produces expected output

---

## Decision Point: Assess Code Duplication

**After Phase 2 tests pass**, assess duplication with `SensitivityAnalysis`:

### Assessment Criteria

Run the following analysis:

```bash
# Compare implementations
diff -u \
  pymc_marketing/mmm/incrementality.py \
  pymc_marketing/mmm/sensitivity_analysis.py \
  | grep "^[+-]" | wc -l
```

**Count duplicated patterns**:
1. Dimension management (e.g., `_compute_dims_order_from_varinput`)
2. Posterior subsampling logic
3. xarray transformation helpers
4. Scenario compilation patterns

**Threshold**: If >100 lines of similar code exist, proceed to **Phase 2.5: Extract Shared Utilities**

### Phase 2.5: Extract Shared Utilities (Conditional)

**Only if duplication threshold is met.**

#### Create `counterfactual_core.py`

**File**: `pymc_marketing/mmm/counterfactual_core.py`

Extract the following utilities:

```python
"""Core utilities for counterfactual analysis (sensitivity, incrementality, optimization).

This module provides shared infrastructure for:
- Vectorized scenario evaluation
- Posterior subsampling
- Dimension management
- Response filtering
"""

from pytensor_utils import extract_response_distribution  # Re-export

def compile_scenario_evaluator(
    response_graph: pt.TensorVariable,
    var_input: str,
    model: Model,
) -> Callable[[np.ndarray], np.ndarray]:
    """Compile a function to evaluate batched scenarios."""
    # Implementation from research doc lines 218-253
    pass

def compute_dims_order(var_input: str, model: Model, custom_dims: tuple) -> list[str]:
    """Compute dimension order from model variable."""
    # Extracted from SensitivityAnalysis._compute_dims_order_from_varinput
    pass

def transform_to_xarray(
    result: np.ndarray,
    dims: list[str],
    coords: dict[str, np.ndarray],
) -> xr.DataArray:
    """Transform numpy result to xarray with proper dimensions."""
    # Extracted from SensitivityAnalysis._transform_output_to_xarray
    pass

def subsample_posterior(
    idata: az.InferenceData,
    fraction: float,
    random_state: RandomState | None = None,
) -> az.InferenceData:
    """Subsample posterior for memory management."""
    # Extracted from SensitivityAnalysis._draw_indices_for_percentage
    pass
```

#### Update Both Classes

Refactor `Incrementality` and `SensitivityAnalysis` to use shared utilities.

**Success Criteria**:
- [ ] `counterfactual_core.py` created with shared utilities
- [ ] Both classes updated to use shared code
- [ ] All tests still pass
- [ ] Code duplication reduced by >70%

---

## Phase 3: Documentation & Cleanup

### Overview

Polish the implementation with documentation and cleanup.

### Changes Required

#### 1. Update Docstrings

Ensure all methods have comprehensive docstrings with:
- Clear parameter descriptions
- Return value specifications
- Detailed examples
- Notes on carryover handling
- References to Google MMM paper

**Files**:
- `pymc_marketing/mmm/incrementality.py` - All public methods
- `pymc_marketing/mmm/multidimensional.py` - `.incrementality` property

#### 2. Add User Guide Section

**File**: `docs/source/notebooks/mmm/incrementality_guide.ipynb`

Create tutorial notebook demonstrating:
1. Basic ROAS computation
2. Understanding carryover effects
3. CAC computation
4. Comparing different frequencies
5. Interpreting `include_carryin` and `include_carryout`
6. Visualizing incrementality over time

#### 3. Update API Reference

**File**: `docs/source/api/mmm.rst`

Add section for `Incrementality` class with autogenerated docs.

#### 4. Code Cleanup

**Remove deprecated code** (if appropriate):
- Remove `MMMIDataWrapper.get_roas()` if deprecated in Phase 1
- Clean up any temporary debug code
- Remove any commented-out code

**Add TODO comments** for future work:
```python
# TODO: Add memory optimization for large models (chunking)
# TODO: Consider supporting custom counterfactual scenarios (not just zero)
# TODO: Add support for interaction effects isolation
```

#### 5. Update CHANGELOG

**File**: `CHANGELOG.md`

Add entry:
```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New `Incrementality` module for counterfactual ROAS computation
- `mmm.incrementality` property for accessing incrementality analysis
- `compute_incremental_contribution()` method with proper carryover handling
- `contribution_over_spend()` convenience method (ROAS, customers per dollar)
- `spend_over_contribution()` convenience method (CAC, cost per unit)

### Changed
- **Breaking**: `summary.roas()` now uses counterfactual analysis instead of element-wise division
- ROAS values will be slightly higher due to carryover effects being captured

### Deprecated
- `MMMIDataWrapper.get_roas()` deprecated in favor of `mmm.incrementality.contribution_over_spend()`
```

### Success Criteria

#### Automated Verification:
- [ ] Documentation builds successfully: `make docs`
- [ ] No broken links in docs
- [ ] All docstring examples run correctly (doctest)
- [ ] API reference includes new module

#### Manual Verification:
- [ ] Tutorial notebook runs end-to-end
- [ ] Docstrings are clear and helpful
- [ ] Examples are realistic and useful
- [ ] CHANGELOG accurately reflects changes

---

## Testing Strategy

### Unit Tests

**Focus**: Individual functions in isolation

**Tests**:
1. `compute_incremental_contribution()`:
   - Each frequency type
   - Carryover flags combinations
   - Period filtering
   - Output shapes and dimensions
   - Scale handling

2. `contribution_over_spend()`:
   - Zero spend handling (NaN)
   - Positive ROAS values
   - Dimension preservation

3. `spend_over_contribution()`:
   - Reciprocal relationship with ROAS
   - Inf handling for zero contribution

4. Helper methods:
   - `_create_period_groups()` for all frequencies
   - `_get_spend_by_frequency()` aggregation

### Integration Tests

**Focus**: End-to-end flows and comparisons

**Tests**:
1. **Vectorized vs Sequential**:
   - Implement reference loop-based version
   - Compare outputs (should match exactly)

2. **Counterfactual vs Element-wise**:
   - Compare new method to old method
   - Verify new values >= old (carryover captured)

3. **Summary Interface**:
   - Call `mmm.summary.roas()`
   - Verify DataFrame structure
   - Check consistency with direct incrementality calls

4. **Hierarchical Models**:
   - Test with `dims=("country",)`
   - Verify dimension handling
   - Check aggregation across custom dims

### Manual Testing Checklist

**After automated tests pass**:

1. **Spot-check ROAS values**:
   - [ ] Values are in reasonable range (e.g., 0.5 to 10 for typical marketing)
   - [ ] No unexpected NaN or Inf values
   - [ ] Posterior uncertainty is reasonable (HDI not too wide)

2. **Validate carryover effects**:
   - [ ] ROAS with `include_carryout=True` > `include_carryout=False`
   - [ ] Difference is proportional to adstock decay rate
   - [ ] Effect size makes sense given l_max

3. **Test different model types**:
   - [ ] Simple MMM (single country, few channels)
   - [ ] Hierarchical MMM (multiple countries)
   - [ ] Large model (many channels, long time series)

4. **User experience**:
   - [ ] API is intuitive
   - [ ] Error messages are helpful
   - [ ] Examples in docstrings work

---

## Performance Considerations

**Current approach** (Phase 1):
- All scenarios evaluated in single vectorized call
- Memory usage: O(n_periods × n_channels × n_samples × n_dates)
- Efficient for most models

**Future optimizations** (if needed):
- Chunking scenarios for very large models
- Lazy evaluation for memory-constrained environments
- Parallel evaluation across periods

**When to optimize**:
- User reports OOM errors
- Models with >100 channels or >10 custom dimension levels

---

## Migration Notes

**For users upgrading**:

1. **ROAS values will change**: New method captures carryover, so values will typically be slightly higher
2. **API is compatible**: Existing `mmm.summary.roas()` calls work, just use new implementation
3. **Deprecation**: `mmm.data.get_roas()` will show deprecation warning

**Breaking changes**:
- None (API compatible, just different computation)

**Recommended actions**:
- Re-run ROAS analyses to get updated values
- Update any hardcoded ROAS thresholds
- Document that values changed due to methodology improvement

---

## References

### Original Sources
- **Research Document**: `thoughts/shared/research/2026-02-06_14-21-06_channel-attribution-architecture.md`
- **Issue**: GitHub issue #2211 (ROAS computation improvement)
- **Google MMM Paper**: Formula (10), Section 3.2.2 - https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf

### Key Files
- [pymc_marketing/pytensor_utils.py:264-341](pymc_marketing/pytensor_utils.py#L264-L341) - `extract_response_distribution()`
- [pymc_marketing/mmm/sensitivity_analysis.py:289-438](pymc_marketing/mmm/sensitivity_analysis.py#L289-L438) - Vectorization patterns
- [pymc_marketing/mmm/multidimensional.py:2159-2185](pymc_marketing/mmm/multidimensional.py#L2159-L2185) - Property pattern
- [pymc_marketing/mmm/summary.py:45-47](pymc_marketing/mmm/summary.py#L45-L47) - Frequency type alias
- [pymc_marketing/data/idata/mmm_wrapper.py:315-343](pymc_marketing/data/idata/mmm_wrapper.py#L315-L343) - Current ROAS implementation

### Similar Implementations
- `SensitivityAnalysis.run_sweep()` - Vectorized scenario evaluation
- `BudgetOptimizer` - Uses `extract_response_distribution()` for optimization
