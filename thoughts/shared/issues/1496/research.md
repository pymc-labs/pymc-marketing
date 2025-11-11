---
date: 2025-11-11T06:13:22+00:00
researcher: Claude
git_commit: 708a1b6d30dc71383f0898becac2d307f98913ee
branch: work-issue-1496
repository: pymc-marketing
topic: "Introduce different a and b priors in (Modified)BetaGeoModel"
tags: [research, codebase, beta-geo-model, modified-beta-geo-model, priors, dropout-probability, issue-1496]
status: complete
last_updated: 2025-11-11
last_updated_by: Claude
issue_number: 1496
---

# Research: Introduce different a and b priors in (Modified)BetaGeoModel

**Date**: 2025-11-11T06:13:22+00:00
**Researcher**: Claude
**Git Commit**: 708a1b6d30dc71383f0898becac2d307f98913ee
**Branch**: work-issue-1496
**Repository**: pymc-marketing
**Issue**: #1496

## Research Question

Following insights from issue #1390, should we introduce different priors for parameters `a` and `b` in the `(Modified)BetaGeoModel`? The main concern is that when `a=b`, the Beta distribution Beta(a,b) has mean 0.5, which translates to `dropout_probability = 0.5` regardless of covariates or the absolute values of `a` and `b`.

## Executive Summary

**Yes, introducing different priors for `a` and `b` is necessary and beneficial.** The current default hierarchical parameterization forces a symmetric relationship through the constraint:
- `a = phi_dropout × kappa_dropout`
- `b = (1 - phi_dropout) × kappa_dropout`

When covariates are introduced with equal coefficients (`dropout_coefficient_a = dropout_coefficient_b`), this creates a situation where:

1. **The mean dropout probability remains fixed at `phi_dropout`** regardless of covariate values
2. **Covariates only affect the variance/concentration** of the Beta distribution, not the expected dropout rate
3. **This is mathematically problematic** because E[Beta(a,b)] = a/(a+b), and when the ratio a:b is constant (enforced by shared scaling), the mean doesn't change

This issue was identified in the research notebook `docs/source/notebooks/clv/dev/bg_nbg_covariates_test_issues.ipynb` during the implementation of issue #1390 (static covariates support).

## Detailed Findings

### 1. Current Prior Structure

**Location**: `pymc_marketing/clv/models/beta_geo.py:182-194`

The default model configuration uses hierarchical priors that enforce a dependent relationship:

```python
@property
def default_model_config(self) -> ModelConfig:
    """Default model configuration."""
    return {
        "alpha": Prior("Weibull", alpha=2, beta=10),
        "r": Prior("Weibull", alpha=2, beta=1),
        "phi_dropout": Prior("Uniform", lower=0, upper=1),
        "kappa_dropout": Prior("Pareto", alpha=1, m=1),
        "purchase_coefficient": Prior("Normal", mu=0, sigma=1),
        "dropout_coefficient": Prior("Normal", mu=0, sigma=1),
        "purchase_covariate_cols": [],
        "dropout_covariate_cols": [],
    }
```

**Transformation**: `pymc_marketing/clv/models/beta_geo.py:311-320`

```python
phi_dropout = self.model_config["phi_dropout"].create_variable("phi_dropout")
kappa_dropout = self.model_config["kappa_dropout"].create_variable("kappa_dropout")

a = pm.Deterministic("a", phi_dropout * kappa_dropout)
b = pm.Deterministic("b", (1.0 - phi_dropout) * kappa_dropout)
```

**Mathematical Constraint**:
- `a + b = kappa_dropout` (fixed sum)
- `a / (a + b) = phi_dropout` (fixed ratio, hence fixed mean)

### 2. The Problem with Covariates

**Location**: `pymc_marketing/clv/models/beta_geo.py:267-309`

When dropout covariates are introduced:

```python
dropout_coefficient_a = self.model_config["dropout_coefficient"].create_variable(
    "dropout_coefficient_a"
)
dropout_coefficient_b = self.model_config["dropout_coefficient"].create_variable(
    "dropout_coefficient_b"
)

a_scale = pm.Deterministic("a_scale", phi_dropout * kappa_dropout)
b_scale = pm.Deterministic("b_scale", (1.0 - phi_dropout) * kappa_dropout)

a = pm.Deterministic(
    "a",
    a_scale * pm.math.exp(pm.math.dot(dropout_data, dropout_coefficient_a)),
    dims="customer_id",
)
b = pm.Deterministic(
    "b",
    b_scale * pm.math.exp(pm.math.dot(dropout_data, dropout_coefficient_b)),
    dims="customer_id",
)
```

**The Issue**:
- Both `a` and `b` are scaled from a common hierarchical baseline (`a_scale`, `b_scale`)
- If `dropout_coefficient_a ≈ dropout_coefficient_b`, then for customer `i`:
  ```
  a_i / b_i = (a_scale × exp(X_i^T β_a)) / (b_scale × exp(X_i^T β_b))
              = (a_scale / b_scale) × exp(X_i^T (β_a - β_b))
  ```
- When `β_a = β_b`, the ratio `a_i / b_i = a_scale / b_scale = phi_dropout / (1 - phi_dropout)` is constant
- Therefore: **E[dropout_i] = a_i/(a_i + b_i) = constant** (doesn't vary with covariates!)

### 3. Evidence from Research Notebook

**Location**: `docs/source/notebooks/clv/dev/bg_nbg_covariates_test_issues.ipynb`

The notebook demonstrates the issue empirically:

**Setup** (cells 12-13):
```python
true_params = dict(
    a_scale=1,
    b_scale=1,
    alpha_scale=1,
    r=1,
    purchase_coefficient_alpha=np.array([1.0, -2.0]),
    dropout_coefficient_a=np.array([3.0]),
    dropout_coefficient_b=np.array([3.0]),  # Equal coefficients!
)
```

**Finding** (cell 28):
> "The introduction of covariates is not changing the value of the dropout probability. We will investigate this below."

**Mathematical Explanation** (cells 33-42):

The notebook provides visual proof that:

1. **When a = b** (symmetric Beta): E[X] = 0.5 regardless of the absolute value
   - Beta(0.5, 0.5), Beta(1, 1), Beta(2, 2), Beta(3, 3), Beta(4, 4) all have mean 0.5
   - "Introducing covariates only narrows our distribution"

2. **When a > b**: E[X] > 0.5 (shifted towards higher dropout)
   - Example: Beta(1, 0.5), Beta(2, 1), Beta(3, 1.5), etc.

3. **When a < b**: E[X] < 0.5 (shifted towards lower dropout)
   - Example: Beta(0.5, 1), Beta(1, 2), Beta(1.5, 3), etc.

**Contrast with ParetoNBD** (cells 43-45):
- ParetoNBD uses a Gamma distribution for the dropout process
- For Gamma distribution, E[X] is monotonically increasing with the shape parameter
- This allows covariates to properly shift dropout probability

**Corrected Assertion** (cell 47):
```python
res_zero["dropout"].std("customer_id") > res_high["dropout"].std("customer_id")
```
The test should check variance, not mean!

### 4. Current Workaround: Direct Priors

Users can already specify independent priors for `a` and `b`:

**Location**: `pymc_marketing/clv/models/beta_geo.py:103-104` (documentation example)

```python
model = BetaGeoModel(
    data=data,
    model_config={
        "r": Prior("Weibull", alpha=2, beta=1),
        "alpha": Prior("HalfFlat"),
        "a": Prior("Beta", alpha=2, beta=3),  # Different shape!
        "b": Prior("Beta", alpha=3, beta=2),  # Different shape!
    },
)
```

**Location**: `pymc_marketing/clv/models/beta_geo.py:232-264`

When `a` and `b` are explicitly provided, the hierarchical constraint is bypassed:

```python
if "a" in self.model_config and "b" in self.model_config:
    if self.dropout_covariate_cols:
        a_scale = self.model_config["a"].create_variable("a_scale")
        b_scale = self.model_config["b"].create_variable("b_scale")
        # Independent scaling with separate coefficients
        a = pm.Deterministic(
            "a",
            a_scale * pm.math.exp(pm.math.dot(dropout_data, dropout_coefficient_a)),
            dims="customer_id",
        )
        b = pm.Deterministic(
            "b",
            b_scale * pm.math.exp(pm.math.dot(dropout_data, dropout_coefficient_b)),
            dims="customer_id",
        )
    else:
        a = self.model_config["a"].create_variable("a")
        b = self.model_config["b"].create_variable("b")
```

**This allows**:
- `a_scale` and `b_scale` to be independently specified
- The ratio `a_scale / b_scale` is not constrained to `phi_dropout / (1 - phi_dropout)`
- Covariates can now shift the mean dropout probability

### 5. Test Coverage

**Location**: `tests/clv/models/test_beta_geo.py:82-98`

Tests already use independent priors:

```python
@pytest.fixture(scope="class")
def model_config(self):
    return {
        "a": Prior("HalfNormal"),
        "b": Prior("HalfStudentT", nu=4),  # Different distribution!
        "alpha": Prior("HalfCauchy", beta=2),
        "r": Prior("Gamma", alpha=1, beta=1),
    }

@pytest.fixture(scope="class")
def default_model_config(self):
    return {
        "a": Prior("HalfFlat"),
        "b": Prior("HalfFlat"),
        "alpha": Prior("HalfFlat"),
        "r": Prior("HalfFlat"),
    }
```

**Location**: `tests/clv/conftest.py:110-115`

Fitted model fixtures use point mass priors (DiracDelta) at MLE values:

```python
model_config = {
    # Narrow Gaussian centered at MLE params from lifetimes BetaGeoFitter
    "a": Prior("DiracDelta", c=1.85034151),
    "alpha": Prior("DiracDelta", c=1.86428187),
    "b": Prior("DiracDelta", c=3.18105431),  # b ≠ a
    "r": Prior("DiracDelta", c=0.16385072),
}
```

**Note**: `a ≈ 1.85` and `b ≈ 3.18` (different values), which allows proper parameter recovery.

### 6. ModifiedBetaGeoModel Has Same Issue

**Location**: `pymc_marketing/clv/models/modified_beta_geo.py:180-268`

The ModifiedBetaGeoModel inherits from BetaGeoModel and has identical prior structure:

```python
class ModifiedBetaGeoModel(BetaGeoModel):
    """Modified Beta-Geometric Negative Binomial Distribution (MBG/NBD) model."""
    # ... same default_model_config ...
    # ... same hierarchical phi/kappa parameterization ...
```

The only difference is the likelihood distribution (`ModifiedBetaGeoNBD` vs `BetaGeoNBD`), but the prior structure is identical.

## Code References

### Key Implementation Files
- `pymc_marketing/clv/models/beta_geo.py:182-194` - Default model configuration
- `pymc_marketing/clv/models/beta_geo.py:232-264` - Direct priors for a and b
- `pymc_marketing/clv/models/beta_geo.py:267-309` - Hierarchical priors with covariates
- `pymc_marketing/clv/models/beta_geo.py:311-320` - Hierarchical transformation (no covariates)
- `pymc_marketing/clv/models/modified_beta_geo.py:180-268` - ModifiedBetaGeoModel (same structure)

### Research & Testing
- `docs/source/notebooks/clv/dev/bg_nbg_covariates_test_issues.ipynb` - Research notebook documenting the issue
- `tests/clv/models/test_beta_geo.py:82-98` - Test fixtures with independent priors
- `tests/clv/conftest.py:110-115` - Fitted model with a ≠ b

### Distribution Implementations
- `pymc_marketing/clv/distributions.py:698-745` - BetaGeoNBD likelihood
- `pymc_marketing/clv/distributions.py:847-893` - ModifiedBetaGeoNBD likelihood

## Architecture Insights

### Current Design Philosophy

The hierarchical parameterization (`phi_dropout`, `kappa_dropout`) was designed for:
1. **Interpretability**: `phi` represents mean dropout rate, `kappa` represents heterogeneity
2. **Efficiency**: Reduces dimensionality from 2 parameters (a, b) to 2 parameters (φ, κ) with clearer interpretation
3. **Regularization**: Partial pooling through hierarchy

### The Mathematical Problem

The Beta distribution has mean:
```
E[Beta(a, b)] = a / (a + b)
```

In the hierarchical parameterization:
```
a = φ × κ
b = (1 - φ) × κ
E[Beta(a, b)] = φ × κ / (φ × κ + (1-φ) × κ) = φ × κ / κ = φ
```

This is elegant and interpretable! **However**, when covariates are introduced with shared coefficients:
```
a_i = (φ × κ) × exp(X_i^T β_a)
b_i = ((1-φ) × κ) × exp(X_i^T β_b)

If β_a = β_b = β, then:
E[Beta(a_i, b_i)] = (φ × κ × exp(X_i^T β)) / (κ × exp(X_i^T β)) = φ
```

The exponential scaling cancels out in the ratio! The mean remains `φ` regardless of covariate values.

### The Fix Required

Two separate coefficient vectors `β_a` and `β_b` are already implemented, but they need to diverge for covariates to affect mean dropout probability. The model should:

1. **Allow different default priors** for `a` and `b` in the hierarchical parameterization
2. **Better document** that equal coefficients only affect variance
3. **Consider alternative parameterizations** that naturally allow mean-shifting with covariates

## Historical Context

### Issue #1390: Introduction of Static Covariates

**Commit**: 5f599190 - "Allow static covariates in BGNBDModel (#1390)"
**Date**: February 14, 2025
**Author**: Pablo de Roque

This issue introduced covariate support for both purchase and dropout processes. The implementation:
- Added `purchase_coefficient` and `dropout_coefficient` priors
- Added `purchase_covariate_cols` and `dropout_covariate_cols` configuration
- Implemented exponential link function: `parameter = scale × exp(X^T β)`
- Created separate coefficients for `alpha`, `a`, and `b` when covariates are present

The research notebook `bg_nbg_covariates_test_issues.ipynb` was created during this work to investigate parameter recovery issues, which led to discovering the `a=b` problem.

### Issue #1815: Extension to ModifiedBetaGeoModel

Extended static covariate support to ModifiedBetaGeoModel, maintaining the same prior structure.

### Related Academic Work

The implementation references Fader and Hardie (2007): "Incorporating Time-Invariant Covariates into the Pareto/NBD and BG/NBD Models"

## Recommendations

### Short-term (Immediate Changes)

1. **Update Default Priors**: Change `default_model_config` to use independent priors for `a` and `b` instead of hierarchical `phi_dropout`/`kappa_dropout`:
   ```python
   "a": Prior("HalfNormal", sigma=2),
   "b": Prior("HalfNormal", sigma=2),
   ```

2. **Document the Constraint**: Add clear documentation warning that when using `phi_dropout`/`kappa_dropout`:
   - Equal covariate coefficients only affect variance, not mean
   - For mean-shifting effects, use independent priors for `a` and `b`

3. **Update Tests**: Modify the covariate tests in `bg_nbg_covariates_test_issues.ipynb` to:
   - Check variance reduction instead of mean shift when `β_a = β_b`
   - Add test cases with `β_a ≠ β_b` to demonstrate mean-shifting

### Medium-term (Redesign)

4. **Alternative Hierarchical Parameterization**: Consider a parameterization that allows covariate effects on the mean:
   ```python
   # Mean dropout probability (affected by covariates)
   phi_i = logit^(-1)(logit(phi_base) + X_i^T β_phi)

   # Concentration (affected by covariates)
   kappa_i = kappa_base × exp(X_i^T β_kappa)

   # Beta parameters
   a_i = phi_i × kappa_i
   b_i = (1 - phi_i) × kappa_i
   ```

   This allows covariates to independently affect mean and concentration.

5. **Add Helper Function**: Create a utility to diagnose whether covariates will affect mean or just variance:
   ```python
   def check_covariate_effects(model):
       """Check if dropout covariates affect mean or just variance."""
       if uses_phi_kappa and beta_a == beta_b:
           warn("Covariates only affect variance, not mean dropout probability")
   ```

### Long-term (Research & Development)

6. **Compare with ParetoNBD**: Investigate why ParetoNBD doesn't have this issue (uses Gamma instead of Beta) and document trade-offs

7. **Benchmark Different Parameterizations**: Compare:
   - Independent a, b priors
   - Hierarchical φ, κ priors
   - Alternative logit(φ) + log(κ) parameterization
   - Mixed parameterizations

8. **User Study**: Survey users to understand:
   - How often they use covariates
   - Whether they expect covariates to shift mean or just variance
   - Preference for interpretability vs. flexibility

## Open Questions

1. **Should the default change?**
   - Pro: Independent priors are more flexible and avoid the constraint
   - Con: Less interpretable, breaks backward compatibility
   - Compromise: Keep hierarchical as default, but warn when covariates are used

2. **Should we deprecate φ/κ parameterization?**
   - Pro: Simplifies implementation, avoids confusion
   - Con: Loses interpretability benefit for models without covariates
   - Compromise: Keep both, document clearly when to use each

3. **How do other packages handle this?**
   - lifetimes package (no covariates)
   - BTYD R package (limited covariate support)
   - Need to research best practices

4. **What about ShiftedBetaGeoModel?**
   - Uses same `phi`/`kappa` parameterization (but called `alpha`/`beta` in the code)
   - Location: `pymc_marketing/clv/models/shifted_beta_geo.py:232-241`
   - Should be updated consistently

## Related Research

- Issue #1390: Static covariates introduction
- Issue #1815: ModifiedBetaGeoModel covariates
- Research notebook: `docs/source/notebooks/clv/dev/bg_nbg_covariates_test_issues.ipynb`

## Conclusion

**The necessity of introducing different priors for `a` and `b` is clear.** The current hierarchical constraint through `phi_dropout` and `kappa_dropout` is elegant and interpretable for models without covariates, but creates a mathematical limitation when dropout covariates are introduced with similar coefficients.

**Recommended immediate action**:
1. Change the default configuration to use independent priors for `a` and `b`
2. Update documentation to explain the constraint and when to use each approach
3. Add deprecation warning when using `phi_dropout`/`kappa_dropout` with dropout covariates

**The fix is straightforward**: The model already supports independent priors through the direct specification path (`pymc_marketing/clv/models/beta_geo.py:232-264`). We just need to make this the default instead of the hierarchical path.

This change will:
- ✅ Allow covariates to affect mean dropout probability
- ✅ Maintain backward compatibility (users can still specify `phi_dropout`/`kappa_dropout`)
- ✅ Align with test fixtures which already use independent priors
- ✅ Resolve the issue identified in #1390 research
