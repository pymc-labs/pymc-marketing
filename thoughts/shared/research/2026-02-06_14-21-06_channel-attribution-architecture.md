---
date: 2026-02-06T14:21:06+0000
researcher: Claude Sonnet 4.5
git_commit: 7c7b8326c319d80f998f4e5bdffa356160e3f131
branch: work-issue-2211
repository: pymc-marketing
topic: "Channel Incrementality Architecture: Unified Framework for ROAS, CAC, and Counterfactual Analysis"
tags: [research, codebase, mmm, incrementality, roas, cac, vectorization, architecture, counterfactual]
status: complete
last_updated: 2026-02-06
last_updated_by: Claude Sonnet 4.5
---

# Research: Channel Incrementality Architecture for ROAS, CAC, and Counterfactual Analysis

**Date**: 2026-02-06T14:21:06+0000
**Researcher**: Claude Sonnet 4.5
**Git Commit**: 7c7b8326c319d80f998f4e5bdffa356160e3f131
**Branch**: work-issue-2211
**Repository**: pymc-marketing

## Table of Contents

1. [Research Question](#research-question)
2. [Summary](#summary)
3. [Background: Previous Research](#background-previous-research)
4. [Detailed Findings](#detailed-findings)
   - [1. Why Incrementality Shouldn't Live in SensitivityAnalysis](#1-why-incrementality-shouldnt-live-in-sensitivityanalysis)
   - [2. Shared Vectorization Infrastructure](#2-shared-vectorization-infrastructure-what-to-extract)
   - [3. Proposed Shared Utilities Module](#3-proposed-shared-utilities-module)
   - [4. Current Incrementality and Contribution Patterns](#4-current-incrementality-and-contribution-patterns)
   - [5. Frequency Argument Patterns in Codebase](#5-frequency-argument-patterns-in-codebase)
   - [6. Module Organization and Where Incrementality Should Live](#6-module-organization-and-where-incrementality-should-live)
5. [Proposed Architecture](#proposed-architecture)
   - [Core Incrementality Module Structure](#core-incrementality-module-structure)
   - [Implementation Details for `compute_incremental_contribution()`](#implementation-details-for-compute_incremental_contribution)
   - [Integration with Summary Interface](#integration-with-summary-interface)
6. [Code References](#code-references)
7. [Open Questions](#open-questions)
8. [Implementation Checklist](#implementation-checklist)

## Research Question

Building on previous research on ROAS computation and vectorization optimizations, how should we architect a unified incrementality framework that:

1. Implements efficient counterfactual analysis similar to SensitivityAnalysis but in a more appropriate module (incrementality is not sensitivity)
2. Provides a general `compute_incremental_contribution()` function that solves the core temporal incrementality problem
3. Offers two convenience functions: `contribution_over_spend()` (ROAS, customers per dollar) and `spend_over_contribution()` (CAC)
4. Supports frequency-based aggregation (weekly, monthly, quarterly, yearly, all_time)

## Summary

**A new `Incrementality` module is recommended** that leverages the vectorization patterns from `SensitivityAnalysis` while maintaining architectural clarity.

> **Why "Incrementality" and not "Attribution"?** The entire MMM project is fundamentally a channel attribution model — it attributes outcomes to marketing channels. Naming a sub-module "attribution" would be confusing and overloaded. What this module specifically does is decompose a single channel's contribution into its **incremental** effects over time periods. "Incrementality" precisely captures this: measuring the incremental lift each channel creates, accounting for adstock carryover.

The core insight is that **incrementality and sensitivity are distinct analysis types** that share counterfactual evaluation infrastructure but serve different purposes:

- **Sensitivity Analysis**: "What if we change X by Y%?" (continuous sweeps, marginal effects)
- **Incrementality Analysis**: "What incremental value did X create?" (discrete on/off scenarios, ROAS/CAC)

**Key Design Decisions:**

1. **New Module**: Create `pymc_marketing/mmm/incrementality.py` with `Incrementality` class
2. **Exposure**: Via property `mmm.incrementality` (follows `.sensitivity`, `.plot`, `.summary` pattern)
3. **Core Function**: `compute_incremental_contribution()` implements counterfactual incrementality with carryover handling
4. **Convenience Functions**: `contribution_over_spend()` and `spend_over_contribution()` wrap core for specific use cases
5. **Shared Infrastructure**: Extract reusable vectorization utilities to `counterfactual_core.py`
6. **Frequency Support**: Use established `Frequency` type alias from `summary.py`

**Expected Performance**: 30-100x speedup vs. naive loop-based implementation using vectorized graph evaluation.

## Background: Previous Research

### Research Document 1: ROAS Computation (Issue #2211)

**Source**: `thoughts/shared/issues/2211/research.md`

**Key Findings:**
- Current ROAS implementation (`mmm_wrapper.py:315-342`) uses simple element-wise division: `ROAS[t] = contribution[t] / spend[t]`
- **Problem**: Ignores adstock carryover effects, treating each period independently
- **Solution**: Implement Google MMM paper Formula (10) - counterfactual approach
  ```
  ROAS_m = Σ(t=t0..t1+L-1) [Y_actual - Y_counterfactual] / Σ(t=t0..t1) spend
  ```
- **Requirements**:
  - Work with `MultidimensionalMMM` and hierarchical dimensions (geo, country, region)
  - Use xarray operations on `fit_data` instead of pandas
  - Handle arbitrary time frequencies
  - Process all channels simultaneously with vectorized operations
  - Support separate `include_carryin` and `include_carryout` flags
  - Output shape: `(chain, draw, date, channel, *custom_dims)`
  - Use period-end dates for labeling

### Research Document 2: Vectorization Optimization

**Source**: `.cursor/research_roas_vectorization.md`

**Key Findings:**
- `sample_posterior_predictive()` cannot batch scenarios due to `pm.set_data()` limitations
- **Solution**: Use `vectorize_graph()` pattern from `SensitivityAnalysis.run_sweep()`
- **Performance**: O(n_scenarios) sequential → O(1) parallel evaluation
- **Pattern**:
  1. Extract response distribution with `extract_response_distribution()`
  2. Create batched scenario inputs (all counterfactuals at once)
  3. Vectorize graph with `vectorize_graph()`
  4. Compile and evaluate all scenarios in single function call
  5. Aggregate and convert to xarray

**Speedup Example**: 5 channels × 12 months = 60 scenarios
- Loop-based: ~300 seconds
- Vectorized: ~10 seconds
- **30x faster**

## Detailed Findings

### 1. Why Incrementality Shouldn't Live in SensitivityAnalysis

**Analysis of SensitivityAnalysis Class** (`sensitivity_analysis.py:105+`)

**Current Responsibilities:**
- Continuous parameter sweeps (multiplicative, additive, absolute)
- Marginal effects computation via differentiation (`compute_marginal_effects()`)
- Uplift curves relative to baseline (`compute_uplift_curve_respect_to_base()`)
- Exploring "what if we change spending by X%?" scenarios

**Why Incrementality Is Different:**

| Aspect | Sensitivity Analysis | Incrementality Analysis |
|--------|---------------------|---------------------|
| **Purpose** | Explore parameter space | Measure causal impact |
| **Scenarios** | Continuous sweeps (0.5x, 1x, 1.5x, 2x) | Discrete on/off (baseline vs counterfactual) |
| **Question** | "What if we change X by Y%?" | "What did X contribute?" |
| **Output** | Sweep curves, marginal effects | ROAS, CAC, incremental contribution |
| **Aggregation** | Often keep time dimension | Often sum over time (by period) |
| **Use Cases** | Scenario planning, sensitivity | Performance measurement, budgeting |

**Architectural Concern**: Mixing incrementality into `SensitivityAnalysis` would:
- Blur conceptual boundaries (sensitivity vs. incrementality are distinct analyses)
- Overload a single class with two different analysis paradigms
- Reduce discoverability (users looking for ROAS wouldn't think "sensitivity")
- Complicate the API (different parameter patterns for different use cases)

**Recommendation**: **Create separate `Incrementality` class** that reuses shared vectorization infrastructure.

---

### 2. Shared Vectorization Infrastructure (What to Extract)

**Core Reusable Patterns** (from `sensitivity_analysis.py` and `pytensor_utils.py`):

#### Pattern A: Extract Response Distribution
**Location**: `pytensor_utils.py:264-341`
**Function**: `extract_response_distribution(model, idata, response_variable)`

**What it does:**
- Converts PyMC model + posterior into computational graph batched over MCMC samples
- Replaces random variables with posterior constants
- Uses `vectorize_graph()` to add sample dimension
- Returns symbolic graph (shape: `(sample, date, *dims)`)

**Reusability**: ✅ **Already shared** - Used by `SensitivityAnalysis`, `BudgetOptimizer`, and future `Incrementality`

#### Pattern B: Vectorize Over Scenarios
**Location**: `sensitivity_analysis.py:410-419`

**What it does:**
- Creates batched input tensor with scenario dimension
- Vectorizes graph using `vectorize_graph(resp_graph, replace={data_shared: batched_input})`
- Compiles once with `function([input], graph)`
- Evaluates all scenarios in single call

**Reusability**: ✅ **Core pattern** - Can be extracted to shared utility

#### Pattern C: Dimension Management
**Location**: `sensitivity_analysis.py:115-127, 234-263`

**Functions:**
- `_compute_dims_order_from_varinput()` - Extract dimensions from model variable
- `_transform_output_to_xarray()` - Convert numpy to xarray with proper dims/coords

**Reusability**: ✅ **Highly reusable** - Incrementality needs identical functionality

#### Pattern D: Posterior Subsampling
**Location**: `sensitivity_analysis.py:138-163`

**Function**: `_draw_indices_for_percentage(posterior_sample_fraction)`

**What it does:**
- Subsamples posterior draws for memory management
- Uses quadratic formula: `retained = total * fraction^2`

**Reusability**: ✅ **Useful for large models** - Both sensitivity and incrementality benefit

#### Pattern E: Response Masking
**Location**: `sensitivity_analysis.py:176-232, 366-386`

**Function**: `_prepare_response_mask()` + masking logic

**What it does:**
- Validates mask dimensions
- Converts to PyTensor tensor
- Broadcasts to match graph shape
- Selectively zeros responses

**Reusability**: ✅ **Useful for focused incrementality** - Can isolate specific channels/regions

---

### 3. Proposed Shared Utilities Module

**New File**: `pymc_marketing/mmm/counterfactual_core.py`

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
    """Compile a function to evaluate batched scenarios.

    Parameters
    ----------
    response_graph : pt.TensorVariable
        Response graph from extract_response_distribution()
    var_input : str
        Name of input variable to replace
    model : Model
        PyMC model containing var_input

    Returns
    -------
    Callable
        Compiled function: scenarios -> predictions
        Output shape: (n_scenarios, n_samples, *dims)
    """
    data_shared = model[var_input]

    # Create batched input placeholder
    batched_input = pt.tensor(
        name=f"{var_input}_batched",
        dtype=data_shared.dtype,
        shape=(None, *data_shared.type.shape),
    )

    # Vectorize over scenario dimension
    batched_graph = vectorize_graph(response_graph, replace={data_shared: batched_input})

    # Compile
    return function([batched_input], batched_graph)

def compute_dims_order(var_input: str, model: Model, custom_dims: tuple) -> list[str]:
    """Compute dimension order from model variable.

    Extracted from SensitivityAnalysis._compute_dims_order_from_varinput.
    """
    var_dims = tuple(model.named_vars_to_dims.get(var_input, ()))
    if not var_dims:
        return list(custom_dims)
    return [d for d in var_dims if d != "date"]

def transform_to_xarray(
    result: np.ndarray,
    dims: list[str],
    coords: dict[str, np.ndarray],
) -> xr.DataArray:
    """Transform numpy result to xarray with proper dimensions.

    Extracted from SensitivityAnalysis._transform_output_to_xarray.
    """
    return xr.DataArray(result, coords=coords, dims=dims)

def subsample_posterior(
    idata: az.InferenceData,
    fraction: float,
    random_state: RandomState | None = None,
) -> az.InferenceData:
    """Subsample posterior for memory management.

    Parameters
    ----------
    idata : az.InferenceData
        Full posterior
    fraction : float
        Fraction of samples to retain (0.0 to 1.0)
    random_state : RandomState, optional
        Random state for reproducibility

    Returns
    -------
    az.InferenceData
        Subsampled posterior
    """
    if fraction >= 1.0:
        return idata

    n_draws = int(idata.posterior.sizes["draw"] * fraction ** 2)
    rng = np.random.default_rng(random_state)
    draw_indices = rng.choice(idata.posterior.sizes["draw"], size=n_draws, replace=False)

    return idata.isel(draw=draw_indices)

class ResponseFilter(Protocol):
    """Protocol for response filtering."""

    def apply(self, response: pt.TensorVariable, mask: xr.DataArray) -> pt.TensorVariable:
        """Apply filter to response graph."""
        ...

class ZeroOutFilter:
    """Filter that zeros out non-masked values."""

    def apply(self, response: pt.TensorVariable, mask: xr.DataArray) -> pt.TensorVariable:
        mask_tensor = pt.constant(mask.to_numpy(), dtype="bool")
        mask_tensor = pt.broadcast_to(mask_tensor, response.shape)
        zeros = pt.zeros_like(response)
        return pt.set_subtensor(zeros[mask_tensor], response[mask_tensor])
```

**Benefits:**
- Single source of truth for vectorization patterns
- Easier to maintain and test
- Clear separation between infrastructure and analysis logic
- Can evolve independently

---

### 4. Current Incrementality and Contribution Patterns

**Research Summary** (from codebase analysis):

#### Pattern A: Channel Contribution Computation
**Location**: `mmm.py:611-654`

**Current Flow:**
1. `channel_data` → scaling → `channel_data_scaled`
2. `channel_data_scaled` → adstock (temporal attribution) → `adstocked_data`
3. `adstocked_data` → saturation (diminishing returns) → `contribution`
4. `contribution` → `* target_scale` → `contribution_original_scale`

**Key Insight**: The **adstock transformation is the temporal attribution mechanism**:
- Spend at t-L affects outcomes at t-L, t-L+1, ..., t
- Uses convolution with decay weights
- `l_max` parameter controls attribution window

#### Pattern B: Current ROAS Computation
**Location**: `mmm_wrapper.py:315-343`

```python
def get_roas(self, original_scale: bool = True) -> xr.DataArray:
    """Compute ROAS = contribution / spend."""
    contributions = self.get_channel_contributions(original_scale=original_scale)
    spend = self.get_channel_spend()
    spend_safe = xr.where(spend == 0, np.nan, spend)
    return contributions / spend_safe
```

**Problem**: Element-wise division doesn't account for:
- Spend at t affecting contribution at t+1, t+2, ..., t+l_max
- Need to sum incremental contribution over carryover window
- Need counterfactual: "What if channel was off?"

#### Pattern C: Forward Pass for Counterfactuals
**Location**: `mmm.py:956-1025`

**Function**: `channel_contribution_forward_pass(channel_data)`

**What it does:**
- Applies same transformations (scaling → adstock → saturation) to new data
- Uses fitted posterior parameters
- Returns full posterior distribution of contributions

**Usage**: Already used for scenario analysis and budget optimization

---

### 5. Frequency Argument Patterns in Codebase

**Research Summary** (from frequency analysis):

#### Established Pattern: `Frequency` Type Alias
**Location**: `summary.py:45-48`

```python
Frequency = Literal["original", "weekly", "monthly", "quarterly", "yearly", "all_time"]
```

**Usage in Summary Methods:**
- `contributions(frequency=...)`
- `roas(frequency=...)`
- `posterior_predictive(frequency=...)`

**Frequency to Pandas Mapping** (`data/idata/utils.py:187-192`):
```python
period_map = {
    "weekly": "W",
    "monthly": "ME",      # Month End
    "quarterly": "QE",    # Quarter End
    "yearly": "YE",       # Year End
}
```

**Special Case**: `"all_time"` removes date dimension entirely

**Period Labeling Convention**: **Period-end dates**
- Monthly: Last day of month (e.g., `2024-01-31`)
- Quarterly: Last day of quarter (e.g., `2024-03-31`)
- Yearly: Last day of year (e.g., `2024-12-31`)

**Validation**: Implicit via `Literal` type hints (no separate validation function)

**Recommendation**: Use `Frequency` type alias from `summary.py` for consistency.

---

### 6. Module Organization and Where Incrementality Should Live

**Research Summary** (from architecture analysis):

#### Current Module Structure

```
pymc_marketing/mmm/
├── multidimensional.py  # Main MMM class with property-based interface
│   ├── .plot → MMMPlotSuite
│   ├── .summary → MMMSummaryFactory
│   ├── .data → MMMIDataWrapper
│   └── .sensitivity → SensitivityAnalysis
├── sensitivity_analysis.py  # Counterfactual sweeps
├── summary.py              # DataFrame generation with HDI
├── plot.py                 # Visualization methods
├── budget_optimizer.py     # Budget allocation
└── data/idata/
    └── mmm_wrapper.py      # Data access layer
```

#### Established Interface Pattern

**Property-Based Facade** (lines from `multidimensional.py`):

```python
@property
def plot(self) -> MMMPlotSuite:
    """Access plotting functionality."""
    return MMMPlotSuite(self.idata)

@property
def data(self) -> MMMIDataWrapper:
    """Access data wrapper functionality."""
    return MMMIDataWrapper(self.idata, schema=self._schema)

@property
def summary(self) -> MMMSummaryFactory:
    """Access summary factory functionality."""
    return MMMSummaryFactory(data=self.data, model=self)

@property
def sensitivity(self) -> SensitivityAnalysis:
    """Access sensitivity analysis functionality."""
    return SensitivityAnalysis(pymc_model=self.model, idata=self.idata, dims=self.dims)
```

**Key Pattern**: Each property returns a specialized class with focused responsibilities.

#### Recommendation: New `incrementality.py` Module

**Location**: `pymc_marketing/mmm/incrementality.py`

**Exposure**: Via property on `MMM` class (around line 2186 in `multidimensional.py`):

```python
@property
def incrementality(self) -> Incrementality:
    """Access incrementality and counterfactual analysis functionality.

    Returns
    -------
    Incrementality
        An instance configured with this MMM model for computing
        incremental contributions, ROAS, and CAC.

    Examples
    --------
    >>> # Compute ROAS with carryover effects
    >>> roas = mmm.incrementality.contribution_over_spend(
    ...     frequency="quarterly",
    ...     include_carryin=True,
    ...     include_carryout=True,
    ... )
    >>>
    >>> # Compute CAC (Cost of Acquisition)
    >>> cac = mmm.incrementality.spend_over_contribution(
    ...     frequency="monthly",
    ... )
    >>>
    >>> # Core incrementality function
    >>> incremental = mmm.incrementality.compute_incremental_contribution(
    ...     period_start="2024-01-01",
    ...     period_end="2024-03-31",
    ...     frequency="weekly",
    ... )
    """
    return Incrementality(model=self, idata=self.idata)
```

**Rationale**:
- Follows established property pattern (`.plot`, `.summary`, `.data`, `.sensitivity`)
- Keeps incrementality logic separate and focused
- Easy to discover: `mmm.incrementality.contribution_over_spend()`
- Can leverage `self.data` wrapper internally
- Can access model transformations (adstock, saturation) via `self.model`

---

## Proposed Architecture

### Core Incrementality Module Structure

**File**: `pymc_marketing/mmm/incrementality.py`

```python
"""Incrementality and counterfactual analysis for Marketing Mix Models.

This module provides functionality to compute incremental channel contributions
using counterfactual analysis, accounting for adstock carryover effects.
"""

from __future__ import annotations

import warnings
from typing import Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from pytensor import function
from pytensor.graph.basic import vectorize_graph
import pytensor.tensor as pt

from pymc_marketing.mmm.counterfactual_core import (
    compile_scenario_evaluator,
    compute_dims_order,
    extract_response_distribution,
    subsample_posterior,
    transform_to_xarray,
)
from pymc_marketing.mmm.summary import Frequency
from pymc_marketing.mmm.utils import _convert_frequency_to_timedelta
from pymc_marketing.mmm.multidimensional import MMM


class Incrementality:
    """Incrementality and counterfactual analysis for MMM models.

    Computes incremental channel contributions by comparing predictions with
    actual spend vs. counterfactual (zero) spend, accounting for
    adstock carryover effects.

    This class uses vectorized graph evaluation to efficiently compute
    incrementality across all channels and posterior samples simultaneously.

    Parameters
    ----------
    model : MMM
        Fitted MMM model instance
    idata : az.InferenceData
        InferenceData containing posterior samples and fit data

    Examples
    --------
    >>> # Access via model property
    >>> incr = mmm.incrementality
    >>>
    >>> # Compute quarterly ROAS
    >>> roas = incr.contribution_over_spend(
    ...     frequency="quarterly",
    ...     period_start="2024-01-01",
    ...     period_end="2024-12-31",
    ... )
    >>>
    >>> # Compute monthly CAC
    >>> cac = incr.spend_over_contribution(
    ...     frequency="monthly",
    ... )
    """

    def __init__(self, model: MMM, idata: az.InferenceData):
        self.model = model
        self.idata = idata
        self.data = model.data  # MMMIDataWrapper

    def compute_incremental_contribution(
        self,
        frequency: Frequency,
        period_start: str | pd.Timestamp | None = None,
        period_end: str | pd.Timestamp | None = None,
        include_carryin: bool = True,
        include_carryout: bool = True,
        original_scale: bool = True,
        posterior_sample_fraction: float = 1.0,
    ) -> xr.DataArray:
        """Compute incremental channel contributions using counterfactual analysis.

        This is the core incrementality function that solves the temporal
        incrementality problem: determining what incremental value each channel created during a
        time period, accounting for adstock carryover effects.

        The computation compares:
        - **Actual scenario**: Model predictions with actual channel spend
        - **Counterfactual scenario**: Model predictions with zero channel spend

        The difference represents the incremental contribution attributable to
        the channels during the evaluation period.

        Parameters
        ----------
        frequency : Frequency
            Time aggregation frequency. One of:
            - "original": No aggregation (daily/weekly as in data)
            - "weekly": Aggregate to weeks
            - "monthly": Aggregate to months
            - "quarterly": Aggregate to quarters
            - "yearly": Aggregate to years
            - "all_time": Single value across entire period
        period_start : str or pd.Timestamp, optional
            Start date for evaluation window. If None, uses start of fitted data.
        period_end : str or pd.Timestamp, optional
            End date for evaluation window. If None, uses end of fitted data.
        include_carryin : bool, default=True
            Include impact of pre-period channel spend via adstock carryover.
            When True, prepends last l_max observations to capture historical
            effects that carry into the evaluation period.
        include_carryout : bool, default=True
            Include impact of evaluation period spend that carries into post-period.
            When True, extends evaluation window by l_max periods to capture
            trailing adstock effects.
        original_scale : bool, default=True
            Return contributions in original scale of target variable.
        posterior_sample_fraction : float, default=1.0
            Fraction of posterior samples to use (0.0 to 1.0).
            Reduce for large models to manage memory. Uses quadratic sampling:
            actual_samples = total_samples * fraction^2

        Returns
        -------
        xr.DataArray
            Incremental contributions with dimensions:
            - (sample, date, channel, *custom_dims) when frequency != "all_time"
            - (sample, channel, *custom_dims) when frequency == "all_time"

            For models with hierarchical dimensions like dims=("country",),
            output has shape (sample, date, channel, country).

        Raises
        ------
        ValueError
            If frequency is invalid or period dates are outside fitted data range.

        Notes
        -----
        **Temporal Incrementality**: This function correctly handles the fact that:
        - Spend at time t0 affects outcomes at t0, t0+1, ..., t0+l_max-1 (via adstock)
        - To measure full impact, we must sum contributions over carryover window

        **Carryover Logic**:
        - `include_carryin=True`: Accounts for pre-period spend still affecting evaluation period
        - `include_carryout=True`: Accounts for evaluation period spend affecting post-period
        - Both default to True for accurate incrementality measurement

        **Performance**: Uses vectorized graph evaluation to compute all channels
        and posterior samples simultaneously. Typical speedup: 30-100x vs. naive loops.

        **Period Labeling**: Periods are labeled with period-end dates:
        - Monthly: Last day of month (e.g., "2024-01-31")
        - Quarterly: Last day of quarter (e.g., "2024-03-31")
        - Yearly: Last day of year (e.g., "2024-12-31")

        References
        ----------
        Google MMM Paper: https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf
        Formula (10), Section 3.2.2

        Examples
        --------
        >>> # Compute quarterly incremental contributions
        >>> incremental = mmm.incrementality.compute_incremental_contribution(
        ...     frequency="quarterly",
        ...     period_start="2024-01-01",
        ...     period_end="2024-12-31",
        ... )
        >>>
        >>> # Mean contribution per channel per quarter
        >>> incremental.mean(dim="sample")
        <xarray.DataArray (date: 4, channel: 3)>
        ...
        >>>
        >>> # Total annual contribution (all_time)
        >>> annual = mmm.incrementality.compute_incremental_contribution(
        ...     frequency="all_time",
        ...     period_start="2024-01-01",
        ...     period_end="2024-12-31",
        ... )
        >>>
        >>> # Without carryover (for comparison)
        >>> no_carryover = mmm.incrementality.compute_incremental_contribution(
        ...     frequency="monthly",
        ...     include_carryin=False,
        ...     include_carryout=False,
        ... )
        """
        # Implementation will follow vectorized pattern from research
        # See implementation pseudocode in "Implementation Details" section below
        pass

    def contribution_over_spend(
        self,
        frequency: Frequency,
        period_start: str | pd.Timestamp | None = None,
        period_end: str | pd.Timestamp | None = None,
        include_carryin: bool = True,
        include_carryout: bool = True,
        posterior_sample_fraction: float = 1.0,
    ) -> xr.DataArray:
        """Compute contribution per unit of spend (ROAS, customers per dollar, etc.).

        This convenience function wraps `compute_incremental_contribution()` and
        divides by spend to compute efficiency metrics:

        - **ROAS** (Return on Ad Spend): When target is revenue
        - **Customers per dollar**: When target is customer count
        - **Units per dollar**: When target is sales volume

        Formula:
        ```
        contribution_over_spend = incremental_contribution / total_spend
        ```

        Parameters
        ----------
        frequency : Frequency
            Time aggregation frequency (weekly, monthly, quarterly, yearly, all_time)
        period_start, period_end : str or pd.Timestamp, optional
            Date range for computation
        include_carryin : bool, default=True
            Include pre-period carryover effects
        include_carryout : bool, default=True
            Include post-period carryover effects
        posterior_sample_fraction : float, default=1.0
            Fraction of posterior samples to use

        Returns
        -------
        xr.DataArray
            Contribution per unit spend with dimensions (sample, date, channel, *custom_dims).
            Zero spend results in NaN for that channel/period.

        Examples
        --------
        >>> # Compute quarterly ROAS (revenue per dollar spent)
        >>> roas = mmm.incrementality.contribution_over_spend(
        ...     frequency="quarterly",
        ...     period_start="2024-01-01",
        ...     period_end="2024-12-31",
        ... )
        >>>
        >>> # Mean ROAS per channel
        >>> roas.mean(dim=["sample", "date"])
        <xarray.DataArray (channel: 3)>
        array([2.34, 1.87, 3.12])
        Coordinates:
          * channel  (channel) object 'tv' 'radio' 'digital'
        >>>
        >>> # For customer acquisition model, this gives customers per dollar
        >>> customers_per_dollar = mmm.incrementality.contribution_over_spend(
        ...     frequency="monthly",
        ... )
        """
        # Get incremental contributions
        incremental = self.compute_incremental_contribution(
            frequency=frequency,
            period_start=period_start,
            period_end=period_end,
            include_carryin=include_carryin,
            include_carryout=include_carryout,
            original_scale=True,
            posterior_sample_fraction=posterior_sample_fraction,
        )

        # Get total spend per period
        # Need to aggregate fit_data by frequency first
        fit_data = self.idata.fit_data

        if period_start is not None:
            fit_data = fit_data.sel(date=slice(period_start, period_end))

        # Aggregate spend by frequency
        if frequency != "original":
            # Use established aggregation pattern
            if frequency == "all_time":
                spend = fit_data[self.model.channel_columns].sum(dim="date")
            else:
                period_map = {
                    "weekly": "W",
                    "monthly": "ME",
                    "quarterly": "QE",
                    "yearly": "YE",
                }
                freq = period_map[frequency]
                spend = fit_data[self.model.channel_columns].resample(date=freq).sum()
        else:
            spend = fit_data[self.model.channel_columns]

        # Stack channels into single dimension for easier division
        spend_stacked = xr.concat(
            [spend[ch] for ch in self.model.channel_columns],
            dim=pd.Index(self.model.channel_columns, name="channel"),
        )

        # Divide contribution by spend (xarray handles broadcasting)
        # Use xr.where to handle zero spend
        spend_safe = xr.where(spend_stacked == 0, np.nan, spend_stacked)
        ratio = incremental / spend_safe

        return ratio

    def spend_over_contribution(
        self,
        frequency: Frequency,
        period_start: str | pd.Timestamp | None = None,
        period_end: str | pd.Timestamp | None = None,
        include_carryin: bool = True,
        include_carryout: bool = True,
        posterior_sample_fraction: float = 1.0,
    ) -> xr.DataArray:
        """Compute spend per unit of contribution (CAC, cost per unit, etc.).

        This convenience function wraps `compute_incremental_contribution()` and
        computes the reciprocal of `contribution_over_spend()`:

        - **CAC** (Customer Acquisition Cost): When target is customer count
        - **Cost per sale**: When target is sales volume
        - **Cost per revenue unit**: When target is revenue (1/ROAS)

        Formula:
        ```
        spend_over_contribution = total_spend / incremental_contribution
        ```

        Parameters
        ----------
        frequency : Frequency
            Time aggregation frequency (weekly, monthly, quarterly, yearly, all_time)
        period_start, period_end : str or pd.Timestamp, optional
            Date range for computation
        include_carryin : bool, default=True
            Include pre-period carryover effects
        include_carryout : bool, default=True
            Include post-period carryover effects
        posterior_sample_fraction : float, default=1.0
            Fraction of posterior samples to use

        Returns
        -------
        xr.DataArray
            Spend per unit contribution with dimensions (sample, date, channel, *custom_dims).
            Zero contribution results in Inf for that channel/period.

        Examples
        --------
        >>> # Compute monthly CAC (cost to acquire one customer)
        >>> cac = mmm.incrementality.spend_over_contribution(
        ...     frequency="monthly",
        ... )
        >>>
        >>> # Mean CAC per channel
        >>> cac.mean(dim=["sample", "date"])
        <xarray.DataArray (channel: 3)>
        array([12.50, 18.75, 8.33])
        Coordinates:
          * channel  (channel) object 'tv' 'radio' 'digital'
        >>>
        >>> # For revenue model, this gives cost per dollar of revenue (1/ROAS)
        >>> cost_per_revenue = mmm.incrementality.spend_over_contribution(
        ...     frequency="quarterly",
        ... )
        """
        # Simply compute reciprocal of contribution_over_spend
        ratio = self.contribution_over_spend(
            frequency=frequency,
            period_start=period_start,
            period_end=period_end,
            include_carryin=include_carryin,
            include_carryout=include_carryout,
            posterior_sample_fraction=posterior_sample_fraction,
        )

        # Reciprocal: spend / contribution
        # Handle NaN (from zero spend) and Inf (from zero contribution) appropriately
        return 1.0 / ratio
```

---

### Implementation Details for `compute_incremental_contribution()`

**High-Level Algorithm** (vectorized, following SensitivityAnalysis pattern):

```python
def compute_incremental_contribution(self, ...) -> xr.DataArray:
    # 1. Subsample posterior if needed
    if posterior_sample_fraction < 1.0:
        idata = subsample_posterior(self.idata, posterior_sample_fraction)
    else:
        idata = self.idata

    # 2. Extract response distribution (batched over samples)
    response_graph = extract_response_distribution(
        pymc_model=self.model.model,
        idata=idata,
        response_variable="y",  # or "mu"
    )
    # Shape: (sample, date, *custom_dims)

    # 3. Get fit_data and determine period bounds
    fit_data = idata.fit_data
    dates = pd.to_datetime(fit_data.coords["date"].values)

    if period_start is None:
        period_start = dates[0]
    else:
        period_start = pd.to_datetime(period_start)

    if period_end is None:
        period_end = dates[-1]
    else:
        period_end = pd.to_datetime(period_end)

    # 4. Create period groups based on frequency
    periods = self._create_period_groups(period_start, period_end, frequency)
    # Returns: [(t0_1, t1_1), (t0_2, t1_2), ...] or [(t0, t1)] for "all_time"

    # 5. Get l_max for carryover calculations
    l_max = self.model.adstock.l_max
    inferred_freq = pd.infer_freq(dates)

    # 6. Create batched scenarios for ALL periods at once
    all_scenarios = []
    period_labels = []

    for period_idx, (t0, t1) in enumerate(periods):
        # Determine data window with carryover padding
        if include_carryin:
            data_start = t0 - _convert_frequency_to_timedelta(l_max, inferred_freq)
        else:
            data_start = t0

        if include_carryout:
            data_end = t1 + _convert_frequency_to_timedelta(l_max, inferred_freq)
        else:
            data_end = t1

        # Extract data window
        fit_window = fit_data.sel(date=slice(data_start, data_end))
        base_data = fit_window[self.model.channel_columns].to_array(dim="channel")
        # Shape: (channel, date_window, *custom_dims)

        # Create counterfactual: zero out ALL channels during [t0, t1]
        counterfactual = base_data.copy(deep=True)
        eval_mask = (fit_window.date >= t0) & (fit_window.date <= t1)
        counterfactual[:, eval_mask] = 0  # Zero all channels in evaluation period

        # Stack: [baseline, counterfactual] → 2 scenarios per period
        all_scenarios.append(base_data.values)
        all_scenarios.append(counterfactual.values)

        period_labels.append(t1)  # Period-end date convention

    # 7. Batch all scenarios: (n_periods * 2, n_channels, date_window, *custom_dims)
    scenario_array = np.stack(all_scenarios, axis=0)
    n_periods = len(periods)

    # 8. Compile vectorized evaluator (once)
    evaluator = compile_scenario_evaluator(
        response_graph=response_graph,
        var_input="channel_data",
        model=self.model.model,
    )

    # 9. Evaluate ALL scenarios at once
    all_predictions = evaluator(scenario_array)
    # Shape: (n_periods * 2, n_samples, date_window, *custom_dims)

    # 10. Compute incremental contributions per period
    results = []

    for period_idx in range(n_periods):
        baseline_idx = period_idx * 2
        counter_idx = period_idx * 2 + 1

        baseline_pred = all_predictions[baseline_idx]  # (n_samples, date_window, *dims)
        counter_pred = all_predictions[counter_idx]

        # Incremental contribution
        incremental = baseline_pred - counter_pred
        # Shape: (n_samples, date_window, *custom_dims)

        # Sum over evaluation window (including carryout if enabled)
        t0, t1 = periods[period_idx]
        if include_carryout:
            carryout_end = t1 + _convert_frequency_to_timedelta(l_max, inferred_freq)
            eval_dates = (dates >= t0) & (dates <= carryout_end)
        else:
            eval_dates = (dates >= t0) & (dates <= t1)

        total_incremental = incremental[:, eval_dates].sum(axis=1)
        # Shape: (n_samples, *custom_dims)

        # Add period coordinate
        period_label = period_labels[period_idx]
        total_incremental_da = xr.DataArray(
            total_incremental,
            dims=("sample", *self.model.dims),
            coords={
                "sample": np.arange(total_incremental.shape[0]),
                **{dim: fit_data.coords[dim].values for dim in self.model.dims},
            },
        )
        total_incremental_da = total_incremental_da.assign_coords(date=period_label).expand_dims("date")

        results.append(total_incremental_da)

    # 11. Concatenate all periods
    if frequency == "all_time":
        # Single period, no date dimension
        result = results[0].squeeze("date", drop=True)
    else:
        result = xr.concat(results, dim="date")

    # 12. Add channel dimension
    result = result.expand_dims({"channel": self.model.channel_columns})

    # 13. Ensure standard dimension order
    dim_order = ["sample", "date", "channel", *self.model.dims]
    if frequency == "all_time":
        dim_order.remove("date")
    result = result.transpose(*dim_order)

    # 14. Scale to original if needed
    if original_scale:
        target_scale = self.data.get_target_scale()
        result = result * target_scale

    return result

def _create_period_groups(
    self,
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: Frequency,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Create list of (period_start, period_end) tuples for given frequency."""
    if frequency == "all_time":
        return [(start, end)]

    if frequency == "original":
        # Use original data frequency
        dates = pd.date_range(start, end, freq=pd.infer_freq(self.idata.fit_data.date.values))
        return [(d, d) for d in dates]

    # Map frequency to pandas period
    freq_map = {
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "Y",
    }

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

---

### Integration with Summary Interface

**Keep existing `summary.roas()` for backward compatibility**

Update `mmm/summary.py` to delegate to incrementality:

```python
def roas(
    self,
    hdi_probs: Sequence[float] | None = None,
    frequency: Frequency | None = None,
    output_format: OutputFormat | None = None,
    include_carryin: bool = True,
    include_carryout: bool = True,
) -> DataFrameType:
    """Create ROAS summary DataFrame.

    Delegates to Incrementality.contribution_over_spend() for computation.
    """
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

---

## Code References

### Files to Create

- **`pymc_marketing/mmm/incrementality.py`** - New Incrementality class
- **`pymc_marketing/mmm/counterfactual_core.py`** - Shared vectorization utilities

### Files to Modify

- **`pymc_marketing/mmm/multidimensional.py`** (line ~2186)
  - Add `.incrementality` property to MMM class

- **`pymc_marketing/mmm/summary.py`** (line ~470)
  - Update `roas()` to delegate to Incrementality or add deprecation warning

- **`pymc_marketing/mmm/sensitivity_analysis.py`** (lines 115-263)
  - Refactor to use shared utilities from `counterfactual_core.py`

### Files for Reference

- **`pymc_marketing/pytensor_utils.py:264-341`**
  - `extract_response_distribution()` - Core infrastructure (already shared)

- **`pymc_marketing/mmm/sensitivity_analysis.py:289-438`**
  - `run_sweep()` - Pattern for vectorized scenario evaluation

- **`pymc_marketing/mmm/budget_optimizer.py:896-972`**
  - Graph replacement patterns with `do()`

- **`pymc_marketing/data/idata/mmm_wrapper.py:315-343`**
  - Current `get_roas()` implementation (to be deprecated/replaced)

- **`pymc_marketing/mmm/summary.py:45-48`**
  - `Frequency` type alias definition

---

## Open Questions

### 1. How to handle time-varying media effects?

**Context**: Models with `time_varying_media=True` have multiplicative GP processes.

**Question**: Does vectorization handle this correctly?

**Answer**: Yes, if response_variable includes time-varying effects. Need to verify with test.

### 2. Memory management for large models?

**Context**: Vectorizing all scenarios requires (n_periods * n_channels * 2) × (n_samples) × (n_dates) × (n_dims) memory.

**Question**: When does this become prohibitive?

**Potential solutions**:
- Implement chunking across scenarios
- Add `max_scenarios_per_batch` parameter
- Automatic fallback to sequential evaluation for very large models

**Recommendation**: Implement basic version first, add chunking if users report OOM errors.

---

## Implementation Checklist

### Phase 1: Shared Infrastructure
- [ ] Create `counterfactual_core.py` module
- [ ] Move dimension utilities from `sensitivity_analysis.py`:
  - [ ] `compute_dims_order()`
  - [ ] `transform_to_xarray()`
  - [ ] `subsample_posterior()`
- [ ] Create `compile_scenario_evaluator()` function
- [ ] Create `ResponseFilter` protocol and implementations
- [ ] Write unit tests for shared utilities

### Phase 2: Incrementality Module
- [ ] Create `incrementality.py` with `Incrementality` class
- [ ] Implement `compute_incremental_contribution()`:
  - [ ] Period grouping logic
  - [ ] Carryover window calculation
  - [ ] Vectorized scenario creation
  - [ ] Graph evaluation
  - [ ] Result aggregation
- [ ] Implement `contribution_over_spend()`
- [ ] Implement `spend_over_contribution()`
- [ ] Write comprehensive docstrings with examples

### Phase 3: Integration
- [ ] Add `.incrementality` property to `MMM` class (`multidimensional.py`)
- [ ] Update `summary.roas()` to delegate or deprecate
- [ ] Update `MMMIDataWrapper.get_roas()` deprecation

### Phase 4: Testing
- [ ] Unit tests for `compute_incremental_contribution()`:
  - [ ] Test with different frequencies
  - [ ] Test with hierarchical dimensions
  - [ ] Test carryover flags

- [ ] Unit tests for convenience functions
- [ ] Integration tests comparing to manual loop implementation
- [ ] Performance benchmarks (vectorized vs. loop)
- [ ] Test with time-varying media effects

### Phase 5: Documentation
- [ ] API reference for Incrementality class

### Phase 6: Refactor SensitivityAnalysis
- [ ] Update `SensitivityAnalysis` to use shared `counterfactual_core` utilities
- [ ] Remove duplicate dimension management code
- [ ] Verify all sensitivity tests still pass
- [ ] Document shared infrastructure in both modules
