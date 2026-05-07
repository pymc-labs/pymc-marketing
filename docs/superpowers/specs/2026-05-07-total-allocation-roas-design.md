# Design: Add `total_allocation` to `sample_response_distribution` output

**Date:** 2026-05-07
**Branch:** `isofer/2525-migrate-budget-allocation-plots`

## Problem

`allocation_roas` in `BudgetPlots` computes the ROAS denominator as:

```python
n_periods = len(samples.date)
roas_da = samples["channel_contribution_original_scale"].sum("date") / (
    samples["allocation"] * n_periods
)
```

`allocation` is the per-period (weekly) spend per channel. When `include_carryover=True`,
`sample_response_distribution` appends `l_max` extra zeroed-out dates to the dataset for
adstock tail computation. This means `len(samples.date)` equals `num_budget_periods + l_max`,
overcounting the denominator and producing an artificially low ROAS.

## Solution

Add a `total_allocation` data variable to the output of `sample_response_distribution`.
This is the ground-truth total spend per channel (and any extra dims like geo) over the
budget period, independent of carryover.

`total_allocation = allocation_strategy * self.num_periods`

`self.num_periods` is already correctly computed in `MultiDimensionalBudgetOptimizerWrapper.__init__`
as `total_dataset_dates - adstock.l_max`, so it represents the actual budget period count
regardless of `include_carryover`.

Update `allocation_roas` to use `total_allocation` directly as the denominator.

## Files Changed

| File | Change |
|------|--------|
| `pymc_marketing/mmm/multidimensional.py` | Add `total_allocation` to `constant_data` in `sample_response_distribution` |
| `pymc_marketing/mmm/plotting/budget.py` | Update `allocation_roas` to validate and use `total_allocation` |
| `tests/mmm/plotting/test_budget.py` | Add `total_allocation` to fixtures; add missing-var test |

## Detailed Changes

### 1. `sample_response_distribution` — multidimensional.py:3987

Replace:
```python
constant_data = allocation_strategy.to_dataset(name="allocation")
```

With:
```python
constant_data = xr.merge([
    allocation_strategy.to_dataset(name="allocation"),
    (allocation_strategy * self.num_periods).to_dataset(name="total_allocation"),
])
```

`total_allocation` inherits the same dims as `allocation` — `(channel,)` or `(channel, geo, ...)`.
It is unaffected by `include_carryover` and `budget_distribution_over_period` because
`allocation * num_periods` equals the total budget regardless of how it is distributed
across periods (distribution sums to 1).

### 2. `allocation_roas` — budget.py

**Validation:** Add a check for `total_allocation` alongside the existing checks.

**ROAS computation:** Replace:
```python
n_periods = len(samples.date)
roas_da = samples["channel_contribution_original_scale"].sum("date") / (
    samples["allocation"] * n_periods
)
```

With:
```python
roas_da = samples["channel_contribution_original_scale"].sum("date") / samples["total_allocation"]
```

### 3. Tests — test_budget.py

- Add `total_allocation` to `simple_allocation_samples` and `panel_allocation_samples` fixtures
  (same value as `allocation * n_date` to preserve existing test semantics).
- Add a test `test_missing_total_allocation_raises` for the new validation.

## Invariants

- `include_carryover=True/False` does not change `total_allocation`.
- `budget_distribution_over_period` does not change `total_allocation`.
- `total_allocation.dims == allocation.dims` (no `date` dim).
