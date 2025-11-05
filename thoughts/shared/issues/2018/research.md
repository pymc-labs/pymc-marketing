---
date: 2025-11-05T00:51:50+00:00
researcher: Claude
git_commit: 9537b9a08837a3c5dabcdee6244a0cd1c4688ea0
branch: work-issue-2018
repository: pymc-marketing
topic: "Return floats instead of integers in clv.distributions.ShiftedBetaGeometricRV.rng_fn"
tags: [research, codebase, clv, distributions, shifted-beta-geometric, dtype, issue-2018]
status: complete
last_updated: 2025-11-05
last_updated_by: Claude
issue_number: 2018
---

# Research: Return floats instead of integers in `clv.distributions.ShiftedBetaGeometricRV.rng_fn`

**Date**: 2025-11-05T00:51:50+00:00
**Researcher**: Claude
**Git Commit**: 9537b9a08837a3c5dabcdee6244a0cd1c4688ea0
**Branch**: work-issue-2018
**Repository**: pymc-marketing
**Issue**: #2018

## Research Question

Should `ShiftedBetaGeometricRV.rng_fn` return floats instead of integers? This issue stems from a PR discussion comment suggesting that using the inverse transform method (`geom = np.ceil(np.log(u) / np.log1p(-p))`) would provide better numerical behavior and support for larger values or infinity, rather than the current approach using NumPy's geometric distribution which performs aggressive truncation to return integers.

**PR Discussion Reference**: https://github.com/pymc-labs/pymc-marketing/pull/2010#discussion_r2444116986

## Summary

The `ShiftedBetaGeometricRV` distribution currently returns `int64` values via `rng.geometric()`, with a TODO comment explicitly suggesting consideration of returning `float64` instead. Analysis reveals:

1. **Current Implementation**: Uses `rng.geometric(p, size=size)` which returns integer values truncated by NumPy's internal limits
2. **Proposed Alternative**: Use inverse transform method `np.ceil(np.log(u) / np.log1p(-p))` which could return floats and support infinity
3. **Impact Scope**: The change would affect 1 distribution class out of 7 RandomVariable implementations in the CLV module
4. **Test Dependencies**: Multiple test assertions explicitly check for `int64` dtype and integer values
5. **Unique Status**: `ShiftedBetaGeometricRV` is the ONLY distribution in the CLV module using `dtype = "int64"` - all others use `dtype = "floatX"`

## Detailed Findings

### Current Implementation

**File**: `pymc_marketing/clv/distributions.py:896-920`

```python
class ShiftedBetaGeometricRV(RandomVariable):
    name = "sbg"
    signature = "(),()->()"

    dtype = "int64"  # Currently specified as integer output
    _print_name = ("ShiftedBetaGeometric", "\\operatorname{ShiftedBetaGeometric}")

    @classmethod
    def rng_fn(cls, rng, alpha, beta, size):
        if size is None:
            size = np.broadcast_shapes(alpha.shape, beta.shape)

        alpha = np.broadcast_to(alpha, size)
        beta = np.broadcast_to(beta, size)

        p_samples = rng.beta(a=alpha, b=beta, size=size)

        # prevent log(0) by clipping small p samples
        p = np.clip(p_samples, 1e-100, 1)
        # TODO: Consider returning np.float64 types instead of np.int64
        #       See relevant PR comment: https://github.com/pymc-labs/pymc-marketing/pull/2010#discussion_r2444116986
        return rng.geometric(p, size=size)
```

**Key characteristics**:
- Returns integer samples from `rng.geometric()`
- Clips probability samples to `[1e-100, 1]` to prevent `log(0)` errors
- Has explicit TODO comment referencing this exact issue

### Comparison with Other RandomVariable Implementations

**Analysis of `pymc_marketing/clv/distributions.py`**:

| RandomVariable Class | dtype | Output Shape | Geometric Usage |
|---------------------|-------|--------------|-----------------|
| `ContNonContractRV` | `floatX` | `(2,)` | Uses `rng.geometric(p)` as intermediate value |
| `ContContractRV` | `floatX` | `(3,)` | Uses `rng.geometric(p)` as intermediate value |
| `ParetoNBDRV` | `floatX` | `(2,)` | No geometric distribution |
| `BetaGeoBetaBinomRV` | `floatX` | `(2,)` | No geometric distribution |
| `BetaGeoNBDRV` | `floatX` | `(2,)` | No geometric distribution |
| `ModifiedBetaGeoNBDRV` | `floatX` | `(2,)` | No geometric distribution |
| **`ShiftedBetaGeometricRV`** | **`int64`** | `()` | **Returns `rng.geometric()` directly** |

**Key observations**:
1. `ShiftedBetaGeometricRV` is the ONLY distribution returning `int64`
2. Two other distributions (`ContNonContractRV`, `ContContractRV`) use `rng.geometric()` but only as an intermediate calculation, not as the final return value
3. All other CLV distributions use `floatX` dtype for continuous outputs

### Usage in Models

**File**: `pymc_marketing/clv/models/shifted_beta_geo.py:243-253`

```python
dropout = ShiftedBetaGeometric.dist(
    alpha[self.cohort_idx],
    beta[self.cohort_idx],
)

pm.Censored(
    "dropout",
    dropout,
    lower=None,
    upper=self.data["T"],
    observed=self.data["recency"],
)
```

**Usage context**:
- Used in `ShiftedBetaGeoModel` (cohort-based) and `ShiftedBetaGeoModelIndividual` classes
- Wrapped in `pm.Censored()` for censored observations
- Models customer dropout time (time until churn)
- The "recency" observed data represents time periods (could theoretically be fractional)

### Test Dependencies on int64 dtype

**File**: `tests/clv/test_distributions.py:784-904`

Multiple test assertions explicitly validate integer behavior:

```python
# test_random_basic_properties (lines 794-809)
def test_random_basic_properties(self):
    for alpha in alpha_vals:
        for beta in beta_vals:
            dist = self.pymc_dist.dist(alpha=alpha, beta=beta, size=1000)
            draws = dist.eval()

            # Check basic properties
            assert np.all(draws > 0)
            assert np.all(draws.astype(int) == draws)  # Validates integer values
            assert np.mean(draws) > 0
            assert np.var(draws) >= 0
```

```python
# test_random_edge_cases (lines 811-825)
def test_random_edge_cases(self):
    # ...
    assert np.all(draws.astype(int) == draws)  # Validates integer values
```

```python
# test_logp (line 830)
value = pt.vector(dtype="int64")
```

```python
# test_logp_matches_paper (line 860, 866)
t_vec = np.array([1, 2, 3, 4, 5, 6, 7], dtype="int64")
value = pt.vector(dtype="int64")
```

**Test modifications required**:
- Lines 807-808, 823: Remove or modify `assert np.all(draws.astype(int) == draws)` checks
- Lines 830, 843, 847, 860, 866: Change `dtype="int64"` specifications to `dtype="float64"` or remove dtype specifications
- Potentially add new tests to validate float behavior and support for larger values

## Code References

- `pymc_marketing/clv/distributions.py:900` - dtype specification (`dtype = "int64"`)
- `pymc_marketing/clv/distributions.py:904-917` - `rng_fn` implementation with TODO comment
- `pymc_marketing/clv/distributions.py:915-916` - Explicit TODO comment about this issue
- `pymc_marketing/clv/models/shifted_beta_geo.py:243-253` - Model usage
- `tests/clv/test_distributions.py:784-904` - Test suite with int64 dependencies
- `tests/clv/test_distributions.py:807-808` - Integer value assertions
- `tests/clv/test_distributions.py:830,843,847,860,866` - int64 dtype specifications

## Architecture Insights

### Mathematical Background

From Fader & Hardie (2007), the Shifted Beta-Geometric distribution has:
- **Support**: `t ∈ N_{>0}` (positive integers)
- **PMF**: `P(T=t|α,β) = B(α+1,β+t-1)/B(α,β)` for `t=1,2,...`
- **Survival**: `S(t|α,β) = B(α,β+t)/B(α,β)` for `t=1,2,...`

The mathematical definition uses discrete time periods (natural numbers), which historically justified the integer return type.

### Inverse Transform Method Benefits

According to the PR discussion (ricardoV94's comment):

**Current approach**:
- `rng.geometric(p)` performs aggressive truncation to return integers
- Cannot represent infinity
- NumPy has internal cutoffs for maximum returnable values

**Proposed inverse transform method**:
```python
u = rng.uniform(0, 1, size=size)
geom = np.ceil(np.log(u) / np.log1p(-p))
```

**Advantages**:
1. Returns float values that can represent larger numbers
2. Can naturally support infinity for extreme parameter values
3. Better numerical stability for edge cases
4. More mathematically faithful to the continuous formulation

### Numerical Stability Analysis

The current implementation clips p-values to prevent numerical issues:
```python
p = np.clip(p_samples, 1e-100, 1)
```

This clipping ensures:
- No `log(0)` errors
- Threshold (1e-100) is higher than NumPy's internal truncation cutoff
- Mathematically valid behavior maintained

The inverse transform method would still require similar safeguards but could naturally handle extreme values better.

## Implementation Considerations

### Required Changes

1. **Distribution class** (`pymc_marketing/clv/distributions.py`):
   - Line 900: Change `dtype = "int64"` to `dtype = "floatX"` or `dtype = "float64"`
   - Lines 904-917: Replace `rng_fn` implementation with inverse transform method
   - Remove or update TODO comment

2. **Test suite** (`tests/clv/test_distributions.py`):
   - Lines 807-808, 823: Modify or remove integer validation assertions
   - Lines 830, 843, 847, 860, 866: Update dtype specifications
   - Add new tests for float behavior and edge cases (infinity, very large values)

3. **Documentation**:
   - Update any documentation that specifies integer returns
   - Add notes about floating-point representation of discrete values

### Backward Compatibility Considerations

**Potential breaking changes**:
- Code that explicitly checks for integer dtypes will fail
- Code that relies on integer arithmetic may have precision issues
- Serialized models may have dtype mismatches

**Mitigation strategies**:
- Version the change appropriately (major version bump?)
- Provide clear migration guide
- Consider adding deprecation warnings first
- Ensure `logp`, `logcdf`, and other methods handle float inputs correctly

### Alternative Approaches

1. **Keep int64 but increase max value**: Modify NumPy's geometric implementation (not feasible)
2. **Add a parameter**: Allow users to choose between int64 and float64 (adds complexity)
3. **Automatic conversion**: Return floats but round to integers for compatibility (defeats the purpose)
4. **New distribution class**: Create `ShiftedBetaGeometricFloat` alongside existing (maintains compatibility but duplicates code)

## Recommendation

Based on the analysis, **implementing the inverse transform method with float64 return type is recommended** because:

1. **Consistency**: Aligns with all other CLV distributions using `floatX`
2. **Numerical superiority**: Better handling of extreme values and edge cases
3. **Mathematical validity**: Discrete values can be represented as floats (1.0, 2.0, 3.0, etc.)
4. **Limited impact**: Only affects 1 distribution class and its tests
5. **Explicit TODO**: The code already acknowledges this as a known improvement

**Implementation priority**: Medium (numerical improvement with manageable scope)

## Related Research

- PR #2010 discussion: https://github.com/pymc-labs/pymc-marketing/pull/2010#discussion_r2444116986
- Fader & Hardie (2007): "How to project customer retention" - Mathematical foundation
- NumPy geometric distribution limitations: Internal truncation for integer returns

## Open Questions

1. **Performance**: How does the inverse transform method compare to `rng.geometric()` in terms of speed?
2. **Precision**: Are there any cases where float representation causes precision issues for large discrete values?
3. **PyMC compatibility**: Does PyMC have any requirements or conventions for discrete vs continuous distributions and their dtypes?
4. **User impact**: Are there existing user models that would break with this change?
5. **Testing**: What additional test cases should be added to validate float behavior (infinity, very large values, numerical stability)?
