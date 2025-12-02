---
date: 2025-12-02T00:00:00Z
researcher: Claude (Sonnet 4.5)
git_commit: ffb3756a98f076470fbc822582a79e0c15b97ff7
branch: work-issue-2114
repository: pymc-labs/pymc-marketing
topic: "BUG: ShiftedBetaGeoModel predictive method parametrization with cohorts and covariates"
tags: [research, codebase, shifted-beta-geo, cohorts, covariates, indexing, bug-analysis]
status: complete
last_updated: 2025-12-02
last_updated_by: Claude (Sonnet 4.5)
issue_number: 2114
---

# Research: BUG: ShiftedBetaGeoModel Predictive Method Parametrization

**Date**: 2025-12-02T00:00:00Z
**Researcher**: Claude (Sonnet 4.5)
**Git Commit**: ffb3756a98f076470fbc822582a79e0c15b97ff7
**Branch**: work-issue-2114
**Repository**: pymc-labs/pymc-marketing
**Issue**: #2114

## Research Question

Why does `ShiftedBetaGeoModel` raise an `IndexError: index 1 is out of bounds for axis 0 with size 1` when calling predictive methods with:
- Models configured with multiple cohorts AND at least one covariate
- Predictive methods called with the `data` parameter (external prediction data)

## Summary

The bug is caused by an **index mismatch** in the `_extract_predictive_variables` method. The code filters cohort-level parameters to only include cohorts present in the prediction data, but then computes cohort indices relative to ALL training cohorts instead of the filtered cohorts.

**Root cause location**: `pymc_marketing/clv/models/shifted_beta_geo.py:423-434`

**Fix**: Change the categorical index computation from `categories=self.cohorts` to `categories=cohorts_present` at line 424.

## Detailed Findings

### Core Bug Analysis

The bug occurs in the covariate branch of `_extract_predictive_variables` when reconstructing customer-level parameters:

**File**: `pymc_marketing/clv/models/shifted_beta_geo.py:403-444`

```python
if self.dropout_covariate_cols:
    # Get alpha and beta scale parameters for each cohort
    alpha_cohort = self.fit_result["alpha_scale"].sel(cohort=pred_cohorts)  # Line 405
    beta_cohort = self.fit_result["beta_scale"].sel(cohort=pred_cohorts)    # Line 406
    # ... get coefficients ...

    # Map cohort indices for each customer
    pred_cohort_idx = pd.Categorical(
        customer_cohort_map.values, categories=self.cohorts  # Line 424 - BUG!
    ).codes

    # Reconstruct customer-level parameters
    alpha_pred = alpha_cohort.isel(
        cohort=xarray.DataArray(pred_cohort_idx, dims="customer_id")  # Line 428
    ) * np.exp(...)
```

**The Problem:**

1. **Line 405**: `alpha_cohort` is filtered to only cohorts in `pred_cohorts` (intersection of prediction data cohorts with training cohorts)
2. **Line 424**: `pred_cohort_idx` is computed using `self.cohorts` (ALL training cohorts)
3. **Line 428**: Code tries to use `pred_cohort_idx` to index into the filtered `alpha_cohort`

**Example that triggers the bug:**

```python
Training cohorts: ["2025-01", "2025-02", "2025-03"]  # self.cohorts
Prediction cohorts: ["2025-02"]                       # cohorts_present

# What happens:
pred_cohort_idx = [1, 1, 1, ...]  # Index 1 because "2025-02" is at position 1 in training
alpha_cohort.shape = (chains, draws, 1)  # Only 1 cohort after filtering

# Result: IndexError when trying to access index 1 in axis with size 1
```

### Proposed Fix

**Location**: `pymc_marketing/clv/models/shifted_beta_geo.py:423-425`

**Change from:**
```python
pred_cohort_idx = pd.Categorical(
    customer_cohort_map.values, categories=self.cohorts
).codes
```

**Change to:**
```python
pred_cohort_idx = pd.Categorical(
    customer_cohort_map.values, categories=cohorts_present
).codes
```

This ensures indices are relative to the filtered cohorts, not the full training cohorts.

**Same fix needed at**: Line 437-439 for `beta_pred` (though it references the same `pred_cohort_idx` variable, so fixing it once fixes both uses).

---

### Why the Bug Wasn't Caught by Tests

**Test Coverage Gap**: The test suite has comprehensive coverage for cohorts + covariates, but a critical scenario is missing:

**Existing test**: `test_predictions_with_covariates` (`tests/clv/models/test_shifted_beta_geo.py:860-899`)
- Tests predictions WITH covariates
- Tests with multiple cohorts in training data
- **BUT**: Prediction data includes ALL cohorts from training

**Missing test scenario**: Prediction data with SUBSET of training cohorts
- Training cohorts: ["A", "B", "C"]
- Prediction cohorts: ["B"] or ["C"] (not starting at index 0)

The bug only manifests when:
1. Prediction data has fewer cohorts than training data
2. Prediction cohorts are NOT at the beginning of the training cohort list

---

### Comparison with Non-Covariate Code Path

The non-covariate branch (lines 446-460) handles this correctly by using xarray selection instead of integer indexing:

```python
else:
    # WITHOUT COVARIATES
    alpha_cohort = self.fit_result["alpha"].sel(cohort=pred_cohorts)
    beta_cohort = self.fit_result["beta"].sel(cohort=pred_cohorts)

    # Uses cohort labels for selection, not integer indices
    customer_cohort_mapping = xarray.DataArray(
        customer_cohort_map.values,  # Cohort labels, not indices
        dims=("customer_id",),
        coords={"customer_id": customer_cohort_map.index},
        name="customer_cohort_mapping",
    )
    alpha_pred = alpha_cohort.sel(cohort=customer_cohort_mapping)  # Label-based selection
    beta_pred = beta_cohort.sel(cohort=customer_cohort_mapping)
```

**Why it works:**
- Uses `.sel()` with cohort labels instead of `.isel()` with integer indices
- No need to compute categorical codes
- xarray handles the mapping automatically

**Potential refactor consideration**: The covariate branch could potentially use a similar label-based approach, though it would require restructuring the covariate application logic.

---

## Code References

### Bug Location
- `pymc_marketing/clv/models/shifted_beta_geo.py:423-434` - Incorrect index computation and usage

### Related Code Sections
- `pymc_marketing/clv/models/shifted_beta_geo.py:363-494` - Full `_extract_predictive_variables` method
- `pymc_marketing/clv/models/shifted_beta_geo.py:212-215` - Training cohort index initialization
- `pymc_marketing/clv/models/shifted_beta_geo.py:391-398` - Prediction cohort filtering
- `pymc_marketing/clv/models/shifted_beta_geo.py:548-602` - `expected_probability_alive` (calls `_extract_predictive_variables`)
- `pymc_marketing/clv/models/shifted_beta_geo.py:496-546` - `expected_retention_rate` (also affected)
- `pymc_marketing/clv/models/shifted_beta_geo.py:604-655` - `expected_residual_lifetime` (also affected)
- `pymc_marketing/clv/models/shifted_beta_geo.py:657-708` - `expected_retention_elasticity` (also affected)

### Test Files
- `tests/clv/models/test_shifted_beta_geo.py:860-899` - Existing covariate test (incomplete coverage)
- `tests/clv/models/test_shifted_beta_geo.py:581-635` - Multi-cohort prediction tests (no covariates)

---

## Architecture Insights

### Cohort + Covariate Interaction Pattern

The ShiftedBetaGeoModel uses a two-level parameter structure:

1. **Cohort-level scale parameters** (`alpha_scale`, `beta_scale`)
   - Dims: `(chain, draw, cohort)`
   - Learned during training
   - Represent baseline behavior per cohort

2. **Customer-level parameters** (`alpha`, `beta`)
   - Dims: `(chain, draw, customer_id)`
   - Computed as: `scale[cohort_idx] * exp(-dot(covariates, coefficients))`
   - Combine cohort baseline + individual covariate effects

**Key insight**: The covariate branch must handle three mappings simultaneously:
1. Cohort filtering (training cohorts → prediction cohorts)
2. Cohort-to-customer mapping (cohort → customer_id)
3. Covariate application (customer covariates → parameter adjustment)

The bug occurs in step 2, where the mapping uses incorrect indices after step 1's filtering.

### fit_result Structure

The `fit_result` property returns an xarray Dataset from the posterior group of the InferenceData object:

**With covariates:**
```python
fit_result["alpha_scale"]  # Shape: (chain, draw, cohort)
fit_result["beta_scale"]   # Shape: (chain, draw, cohort)
fit_result["dropout_coefficient_alpha"]  # Shape: (chain, draw, dropout_covariate)
fit_result["dropout_coefficient_beta"]   # Shape: (chain, draw, dropout_covariate)
```

**Without covariates:**
```python
fit_result["alpha"]  # Shape: (chain, draw, cohort)
fit_result["beta"]   # Shape: (chain, draw, cohort)
```

The covariate branch reconstructs customer-level parameters from scale + coefficients during prediction, while the non-covariate branch directly uses cohort-level parameters.

---

## Recommended Test Addition

To prevent regression, add a test case covering the bug scenario:

**Test location**: `tests/clv/models/test_shifted_beta_geo.py`

**Test name**: `test_predictions_with_covariates_subset_cohorts`

**Key aspects**:
- Training data with cohorts: ["A", "B", "C"]
- Prediction data with cohorts: ["B", "C"] (missing "A")
- Prediction data with cohorts: ["C"] (only last cohort)
- Model configured with covariates
- Call all predictive methods with external data parameter

**Expected behavior**: No IndexError, predictions returned successfully

**Example structure**:
```python
def test_predictions_with_covariates_subset_cohorts(self):
    # Training data with 3 cohorts
    train_data = pd.DataFrame({
        "customer_id": range(30),
        "recency": [3, 4, 5] * 10,
        "T": [5] * 30,
        "cohort": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        "channel": [0, 1, 0, 1, 0] * 6,
    })

    model = ShiftedBetaGeoModel(
        data=train_data,
        model_config={"dropout_covariate_cols": ["channel"]}
    )
    model.fit(method="map", maxeval=10)

    # Prediction data with subset of cohorts (NOT starting at index 0)
    pred_data = pd.DataFrame({
        "customer_id": [100, 101, 102],
        "T": [3, 3, 3],
        "cohort": ["B", "C", "C"],  # Missing cohort "A"
        "channel": [1, 0, 1],
    })

    # Should not raise IndexError
    prob_alive = model.expected_probability_alive(data=pred_data, future_t=1)
    assert prob_alive.shape[-1] == 3  # 3 customers

    retention = model.expected_retention_rate(data=pred_data, future_t=1)
    assert retention.shape[-1] == 3
```

---

## Open Questions

1. **Should the covariate branch be refactored to use label-based selection like the non-covariate branch?**
   - Pros: More consistent, less error-prone
   - Cons: May require restructuring covariate application logic

2. **Are there similar bugs in other CLV models with cohorts and covariates?**
   - Quick check: No other CLV models use both cohorts and covariates together
   - ShiftedBetaGeoModel is unique in combining both features

3. **Should `_validate_cohorts(check_pred_data=True)` provide better error messages?**
   - Current: Generic "cohorts don't match" error
   - Potential: Show which cohorts are missing/extra

---

## Historical Context (from thoughts/)

No previous research or documentation found in the thoughts/ directory related to:
- ShiftedBetaGeoModel bugs
- Cohort handling issues
- xarray indexing problems with CLV models

This appears to be a newly discovered edge case in the covariate + cohort interaction.

---

## Related Research

None yet - this is the first research document for this issue.
