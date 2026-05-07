# total_allocation in sample_response_distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `total_allocation` data variable to `sample_response_distribution` output, and update `allocation_roas` to use it as the ROAS denominator instead of `allocation * len(samples.date)`.

**Architecture:** Three focused changes — update test fixtures first so they include `total_allocation`, then update `allocation_roas` validation + denominator, then add `total_allocation` to `sample_response_distribution` output. TDD throughout.

**Tech Stack:** Python, xarray, PyMC-Marketing `MultiDimensionalBudgetOptimizerWrapper`, `BudgetPlots`

---

## File Map

| File | Change |
|------|--------|
| `tests/mmm/plotting/test_budget.py` | Add `total_allocation` to two fixtures; add `test_missing_total_allocation_raises` |
| `pymc_marketing/mmm/plotting/budget.py` | Add `total_allocation` validation; replace denominator in `allocation_roas` |
| `tests/mmm/test_budget_optimizer_multidimensional.py` | Add `test_sample_response_distribution_includes_total_allocation` |
| `pymc_marketing/mmm/multidimensional.py` | Add `total_allocation` to `constant_data` in `sample_response_distribution` |

---

## Task 1: Update `allocation_roas` (test first)

**Files:**
- Test: `tests/mmm/plotting/test_budget.py`
- Modify: `pymc_marketing/mmm/plotting/budget.py`

### Step 1a: Update `simple_allocation_samples` fixture to include `total_allocation`

In `tests/mmm/plotting/test_budget.py`, replace the `simple_allocation_samples` fixture (lines 43–65) with:

```python
@pytest.fixture(scope="module")
def simple_allocation_samples(channels) -> xr.Dataset:
    """xr.Dataset with channel_contribution_original_scale + allocation, no extra dims."""
    rng = np.random.default_rng(SEED)
    n_sample, n_date = 80, 20
    dates = np.arange(n_date)
    contrib = rng.uniform(100, 500, (n_sample, n_date, len(channels)))
    alloc = rng.uniform(1000, 5000, len(channels))
    return xr.Dataset(
        {
            "channel_contribution_original_scale": xr.DataArray(
                contrib,
                dims=("sample", "date", "channel"),
                coords={
                    "sample": np.arange(n_sample),
                    "date": dates,
                    "channel": channels,
                },
            ),
            "allocation": xr.DataArray(
                alloc,
                dims=("channel",),
                coords={"channel": channels},
            ),
            "total_allocation": xr.DataArray(
                alloc * n_date,
                dims=("channel",),
                coords={"channel": channels},
            ),
        }
    )
```

### Step 1b: Update `panel_allocation_samples` fixture to include `total_allocation`

Replace the `panel_allocation_samples` fixture (lines 68–93) with:

```python
@pytest.fixture(scope="module")
def panel_allocation_samples(channels) -> xr.Dataset:
    """xr.Dataset with geo extra dim for panel tests."""
    rng = np.random.default_rng(SEED + 1)
    n_sample, n_date = 80, 20
    geos = ["CA", "NY"]
    dates = np.arange(n_date)
    contrib = rng.uniform(100, 500, (n_sample, n_date, len(geos), len(channels)))
    alloc = rng.uniform(1000, 5000, (len(geos), len(channels)))
    return xr.Dataset(
        {
            "channel_contribution_original_scale": xr.DataArray(
                contrib,
                dims=("sample", "date", "geo", "channel"),
                coords={
                    "sample": np.arange(n_sample),
                    "date": dates,
                    "geo": geos,
                    "channel": channels,
                },
            ),
            "allocation": xr.DataArray(
                alloc,
                dims=("geo", "channel"),
                coords={"geo": geos, "channel": channels},
            ),
            "total_allocation": xr.DataArray(
                alloc * n_date,
                dims=("geo", "channel"),
                coords={"geo": geos, "channel": channels},
            ),
        }
    )
```

- [ ] **Step 1c: Write the failing test for missing `total_allocation`**

Add this test inside `class TestAllocationRoas` in `tests/mmm/plotting/test_budget.py`, after the existing `test_missing_channel_dim_raises` test:

```python
def test_missing_total_allocation_raises(self):
    from pymc_marketing.mmm.plotting.budget import BudgetPlots

    rng = np.random.default_rng(SEED)
    channels = ["tv", "radio"]
    bad_samples = xr.Dataset(
        {
            "channel_contribution_original_scale": xr.DataArray(
                rng.uniform(0, 1, (10, 5, 2)),
                dims=("sample", "date", "channel"),
                coords={"channel": channels},
            ),
            "allocation": xr.DataArray(
                [1000.0, 2000.0],
                dims=("channel",),
                coords={"channel": channels},
            ),
        }
    )
    with pytest.raises(ValueError, match="total_allocation"):
        BudgetPlots().allocation_roas(bad_samples)
```

- [ ] **Step 1d: Run the new test to confirm it fails**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_budget.py::TestAllocationRoas::test_missing_total_allocation_raises -v
```

Expected: FAIL — `ValueError` is not raised because the validation doesn't exist yet.

- [ ] **Step 1e: Implement the `total_allocation` validation and new denominator in `allocation_roas`**

In `pymc_marketing/mmm/plotting/budget.py`, inside `allocation_roas`:

1. After the `"channel" not in samples.dims` check (line 100–101), add:

```python
if "total_allocation" not in samples:
    raise ValueError(
        "Expected 'total_allocation' variable in samples, but none found."
    )
```

2. Replace lines 110–113:

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

- [ ] **Step 1f: Run the full `TestAllocationRoas` test class**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_budget.py::TestAllocationRoas -v
```

Expected: ALL PASS (fixtures now include `total_allocation`, new validation test passes).

- [ ] **Step 1g: Run pre-commit and commit**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/plotting/budget.py tests/mmm/plotting/test_budget.py
git add pymc_marketing/mmm/plotting/budget.py tests/mmm/plotting/test_budget.py
git commit -m "feat(budget): use total_allocation as ROAS denominator"
```

---

## Task 2: Add `total_allocation` to `sample_response_distribution` (test first)

**Files:**
- Test: `tests/mmm/test_budget_optimizer_multidimensional.py`
- Modify: `pymc_marketing/mmm/multidimensional.py`

- [ ] **Step 2a: Write the failing test**

Add this standalone test function to `tests/mmm/test_budget_optimizer_multidimensional.py` (after the existing `test_budget_distribution_carryover_interaction_issue` function):

```python
@compile_kwargs
def test_sample_response_distribution_includes_total_allocation(
    dummy_df, fitted_mmm, compile_kwargs
):
    """total_allocation == allocation * num_periods, regardless of include_carryover."""
    _df_kwargs, X_dummy, _y_dummy = dummy_df

    fitted_mmm.add_original_scale_contribution_variable(["channel_contribution"])

    num_periods = 4
    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
        model=fitted_mmm,
        start_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=1),
        end_date=X_dummy["date_week"].max() + pd.Timedelta(weeks=num_periods),
        compile_kwargs=compile_kwargs,
    )

    allocation_strategy = xr.DataArray(
        np.full((2, 2), 10.0),
        dims=["channel", "geo"],
        coords={
            "channel": ["channel_1", "channel_2"],
            "geo": ["A", "B"],
        },
    )

    for include_carryover in [False, True]:
        result = optimizable_model.sample_response_distribution(
            allocation_strategy=allocation_strategy,
            include_carryover=include_carryover,
        )
        assert "total_allocation" in result, (
            f"total_allocation missing from output with include_carryover={include_carryover}"
        )
        expected = allocation_strategy * optimizable_model.num_periods
        xr.testing.assert_allclose(result["total_allocation"], expected)
```

- [ ] **Step 2b: Run the test to confirm it fails**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/test_budget_optimizer_multidimensional.py::test_sample_response_distribution_includes_total_allocation -v
```

Expected: FAIL — `AssertionError: total_allocation missing from output`.

- [ ] **Step 2c: Implement `total_allocation` in `sample_response_distribution`**

In `pymc_marketing/mmm/multidimensional.py`, replace line 3987:

```python
constant_data = allocation_strategy.to_dataset(name="allocation")
```

With:

```python
constant_data = xr.merge(
    [
        allocation_strategy.to_dataset(name="allocation"),
        (allocation_strategy * self.num_periods).to_dataset(name="total_allocation"),
    ]
)
```

- [ ] **Step 2d: Run the new test to confirm it passes**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/test_budget_optimizer_multidimensional.py::test_sample_response_distribution_includes_total_allocation -v
```

Expected: PASS.

- [ ] **Step 2e: Run the broader test suite to check for regressions**

```bash
conda run -n pymc-marketing-dev pytest tests/mmm/plotting/test_budget.py tests/mmm/test_budget_optimizer_multidimensional.py -v --tb=short 2>&1 | tail -30
```

Expected: ALL PASS.

- [ ] **Step 2f: Run pre-commit and commit**

```bash
conda run -n pymc-marketing-dev pre-commit run --files pymc_marketing/mmm/multidimensional.py tests/mmm/test_budget_optimizer_multidimensional.py
git add pymc_marketing/mmm/multidimensional.py tests/mmm/test_budget_optimizer_multidimensional.py
git commit -m "feat(mmm): add total_allocation to sample_response_distribution output"
```
