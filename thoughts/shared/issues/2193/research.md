---
date: 2026-01-15T00:00:00Z
researcher: Claude (Sonnet 4.5)
git_commit: d954cef509d81e03bb8c29d0e6456f356430ff53
branch: work-issue-2193
repository: pymc-marketing
topic: "sample_curve sampling from prior with hierarchical priors"
tags: [research, codebase, sample_curve, saturation, hierarchical_priors, posterior_predictive]
status: complete
last_updated: 2026-01-15
last_updated_by: Claude (Sonnet 4.5)
issue_number: 2193
---

# Research: sample_curve Sampling from Prior Even When Posterior is Provided

**Date**: 2026-01-15T00:00:00Z
**Researcher**: Claude (Sonnet 4.5)
**Git Commit**: d954cef509d81e03bb8c29d0e6456f356430ff53
**Branch**: work-issue-2193
**Repository**: pymc-marketing
**Issue**: #2193

## Research Question

Why does `mmm.saturation.sample_curve` produce different results when called multiple times with the same parameters? The function should be deterministic given fixed parameter values, but it's sampling from priors for hierarchical parameters (`saturation_beta_log_offset`, `saturation_beta_mean`, `saturation_beta_std`) even when the realized parameters (`saturation_beta`, `saturation_lam`) are provided.

## Summary

The root cause is a **mismatch between the variables provided in the parameters dataset and the variables created when the model is rebuilt**. When `sample_curve` is called:

1. It calls `_sample_curve`, which creates a **brand new PyMC model** with all the transformation's priors
2. When using `LogNormalPrior` with `centered=False`, this recreates the **full hierarchical structure** including intermediate variables like `saturation_beta_mean`, `saturation_beta_std`, and `saturation_beta_log_offset`
3. The `parameters` dataset typically only contains the **final realized values** (`saturation_beta`, `saturation_lam`) from `variable_mapping`
4. When `pm.sample_posterior_predictive` runs, it finds the final variables in the parameters but **not the hierarchical intermediates**
5. PyMC **samples the missing hierarchical variables from their priors**, introducing randomness

This is the expected behavior of PyMC's `sample_posterior_predictive` when given an incomplete parameter set, but it's surprising in this context because users expect deterministic curves when providing what they believe are "all the parameters."

**Solution**: Pass the **entire posterior/prior dataset** to `sample_curve` instead of selecting specific variables. This was recently fixed in commit d954cef5.

## Detailed Findings

### Component 1: The _sample_curve Method

**Location**: `pymc_marketing/mmm/components/base.py:495-534`

This is the core private method that all transformation classes use to generate curves.

**Key Implementation Details**:

1. **Creates a New PyMC Model** (line 524):
   ```python
   with pm.Model(coords=coords):
       pm.Deterministic(
           var_name,
           self.apply(x, dims=output_core_dims),
           dims=(x_dim, *output_core_dims),
       )
   ```
   - Every call creates a completely new model context
   - No connection to the original model used for inference

2. **Calls self.apply()** (line 527):
   - Defined at `base.py:616-655`
   - Internally calls `self._create_distributions(dims=dims)` at line 654
   - This method recreates all PyMC random variables from the transformation's priors

3. **_create_distributions Implementation** (`base.py:370-403`):
   ```python
   def create_variable(parameter_name: str, variable_name: str) -> TensorVariable:
       dist = self.function_priors[parameter_name]
       if not hasattr(dist, "create_variable"):
           return dist

       var = dist.create_variable(variable_name)  # Line 389
       # ... dimension handling ...
       return dim_handler(var, dist_dims)

   return {
       parameter_name: create_variable(parameter_name, variable_name)
       for parameter_name, variable_name in self.variable_mapping.items()
   }
   ```
   - For each parameter in `variable_mapping`, calls `dist.create_variable(variable_name)`
   - This is where the hierarchical structure gets recreated

4. **Posterior Predictive Sampling** (lines 531-534):
   ```python
   return pm.sample_posterior_predictive(
       parameters,
       var_names=[var_name],
   ).posterior_predictive[var_name]
   ```
   - Uses the provided `parameters` dataset
   - PyMC matches variable names in the dataset to variables in the model
   - For matched variables: uses values from dataset
   - For unmatched variables: **samples from their priors**

### Component 2: LogNormalPrior with Non-Centered Parameterization

**Location**: `pymc_marketing/special_priors.py:216-245`

The `LogNormalPrior` class creates multiple intermediate random variables when using hierarchical priors.

**Variables Created in Non-Centered Mode** (`centered=False`):

When `create_variable("saturation_beta")` is called:

1. **Hierarchical Mean Parameter** (if `mean` is a Prior):
   - Created at line 59 via `_create_parameter`
   - Name: `saturation_beta_mean`
   - Type: Whatever Prior is specified (e.g., `pm.Gamma`)

2. **Hierarchical Std Parameter** (if `std` is a Prior):
   - Created at line 59 via `_create_parameter`
   - Name: `saturation_beta_std`
   - Type: Whatever Prior is specified (e.g., `pm.Exponential`)

3. **Standard Normal Offset**:
   - Created at line 237-239:
   ```python
   log_phi_z = pm.Normal(
       name + "_log" + "_offset", mu=0, sigma=1, dims=self.dims
   )
   ```
   - Name: `saturation_beta_log_offset`
   - Type: `pm.Normal(mu=0, sigma=1)` (standard normal)

4. **Final Deterministic**:
   - Created at line 243:
   ```python
   phi = pm.Deterministic(name, phi, dims=self.dims)
   ```
   - Name: `saturation_beta`
   - Value: `exp(mu_log + saturation_beta_log_offset * sigma_log)`

**The _create_parameter Method** (`special_priors.py:55-60`):
```python
def _create_parameter(self, param, value, name):
    if not hasattr(value, "create_variable"):
        return value

    child_name = f"{name}_{param}"
    return self.dim_handler(value.create_variable(child_name), value.dims)
```
- Recursively creates variables for hierarchical Prior objects
- Constructs child names by appending `_{param}` to parent name
- Example: `saturation_beta` + `_mean` → `saturation_beta_mean`

### Component 3: Variable Mapping

**Location**: `pymc_marketing/mmm/components/base.py:346-351`

```python
@property
def variable_mapping(self) -> dict[str, str]:
    """Mapping from parameter name to variable name in the model."""
    return {
        parameter: f"{self.prefix}_{parameter}"
        for parameter in self.default_priors.keys()
    }
```

**Key Limitation**:
- Only maps **function parameter names** to **prefixed variable names**
- Example for `LogisticSaturation`: `{"lam": "saturation_lam", "beta": "saturation_beta"}`
- Does **NOT** include hierarchical intermediate variables like:
  - `saturation_beta_mean`
  - `saturation_beta_std`
  - `saturation_beta_log_offset`

This is by design—`variable_mapping` represents the transformation's functional interface, not its internal hierarchical structure.

### Component 4: Recent Fix in Commit d954cef5

**Commit**: d954cef509d81e03bb8c29d0e6456f356430ff53
**Author**: Daniel Saunders
**Date**: 2026-01-15
**Message**: "correct call to .sample_curves. Passing the whole posterior is the safest bet because then you don't have to ensure you have listed all the necessary variables."

**Change in `mmm_multidimensional_example.ipynb`**:

**Before (problematic)**:
```python
curve = mmm.saturation.sample_curve(
    mmm.idata.posterior[["saturation_beta", "saturation_lam"]],
    max_value=2
)
```

**After (fixed)**:
```python
curve = mmm.saturation.sample_curve(mmm.idata.posterior, max_value=2)
```

**Why This Works**:
- Passing the entire `idata.posterior` includes **all** variables sampled during MCMC
- This includes both the final variables (`saturation_beta`, `saturation_lam`) **and** the hierarchical intermediates (`saturation_beta_mean`, `saturation_beta_std`, `saturation_beta_log_offset`)
- When `pm.sample_posterior_predictive` runs, it finds all necessary variables in the dataset
- No variables are missing, so nothing is sampled from priors
- Results are deterministic (modulo floating-point arithmetic)

## Code References

### Core Implementation Files
- `pymc_marketing/mmm/components/base.py:495-534` - `_sample_curve` method
- `pymc_marketing/mmm/components/base.py:370-403` - `_create_distributions` method
- `pymc_marketing/mmm/components/base.py:346-351` - `variable_mapping` property
- `pymc_marketing/mmm/components/saturation.py:155-193` - `SaturationTransformation.sample_curve` method
- `pymc_marketing/special_priors.py:216-245` - `LogNormalPrior.create_variable` method
- `pymc_marketing/special_priors.py:55-60` - `SpecialPrior._create_parameter` method

### Test Files
- `tests/mmm/components/test_saturation.py` - Tests for saturation curves
- `tests/mmm/components/test_adstock.py` - Tests for adstock curves
- `tests/test_special_priors.py` - Tests for special priors

### Documentation
- `docs/source/notebooks/mmm/mmm_multidimensional_example.ipynb` - Example using the fixed approach
- `docs/source/notebooks/mmm/mmm_lift_test.ipynb` - Contains old approach (may need updating)
- `docs/source/notebooks/general/prior_predictive.ipynb` - Shows `sample_curve` with priors

## Architecture Insights

### Design Pattern: Model Recreation
The `_sample_curve` method follows a **model recreation pattern**:
1. Creates a fresh PyMC model in a new context
2. Rebuilds all transformation components from priors
3. Uses `pm.sample_posterior_predictive` to evaluate with provided parameters

**Trade-offs**:
- **Pro**: Decouples curve sampling from the original model
- **Pro**: Works consistently for both prior and posterior sampling
- **Pro**: Handles arbitrary input ranges (not limited to observed data)
- **Con**: Requires **complete** parameter sets including hierarchical structure
- **Con**: User confusion when "obvious" parameters aren't sufficient

### Hierarchical Prior Structure
When using hierarchical priors like `LogNormalPrior` with nested `Prior` objects:

```
LogNormalPrior(
    mean=Prior("Gamma", ...),      ← Creates saturation_beta_mean
    std=Prior("Exponential", ...),  ← Creates saturation_beta_std
    centered=False                  ← Creates saturation_beta_log_offset
)
```

The full variable tree is:
```
saturation_beta_mean         ~ Gamma(...)
saturation_beta_std          ~ Exponential(...)
saturation_beta_log_offset   ~ Normal(0, 1)
saturation_beta              = Deterministic(exp(mu_log + log_offset * sigma_log))
```

Users typically only see `saturation_beta` in `variable_mapping`, but **all four variables** must be in the parameters dataset for deterministic evaluation.

### PyMC sample_posterior_predictive Behavior
From PyMC documentation and observed behavior:

When you provide a parameters dataset to `sample_posterior_predictive`:
1. PyMC iterates through each (chain, draw) in the dataset
2. For each variable in the model:
   - If the variable name exists in the dataset: use the dataset value
   - If the variable name is missing: **sample from the prior**
3. Evaluate all downstream deterministic variables
4. Return the requested variables

This is **working as designed** from PyMC's perspective—it's a convenience feature for partial parameter specification. However, in the MMM context, it creates surprising behavior.

## Historical Context

### Evolution of the API
Based on git history and code comments:

1. **Original Design**: Users were expected to select specific parameters using `variable_mapping`
2. **Hierarchical Priors Added**: When hierarchical priors were introduced, the parameter set became more complex
3. **User Confusion**: Multiple users likely encountered non-deterministic curves
4. **Recent Fix (d954cef5)**: Standardized on passing the complete posterior/prior dataset

### Related Issues
No explicit GitHub issues found in the repository discussing this problem prior to #2193, but the commit message suggests it was a known pain point.

## Open Questions

1. **Should variable_mapping be expanded?**
   - Should `variable_mapping` include hierarchical intermediates?
   - This would make the interface more explicit but potentially confusing

2. **Should _sample_curve validate parameters?**
   - Could check if all required variables are present in the parameters dataset
   - Raise an error or warning if hierarchical variables are missing
   - Trade-off: more rigid API vs. clearer error messages

3. **Should there be separate methods?**
   - `sample_curve_from_posterior(idata)` - expects complete posterior
   - `sample_curve_from_parameters(params_dict)` - expects only final params
   - Different implementations for different use cases

4. **Documentation improvements needed?**
   - Document that the entire posterior/prior should be passed
   - Add warnings to docstrings about hierarchical variables
   - Include examples in method docstrings

5. **Test coverage**:
   - Are there tests that verify deterministic behavior?
   - Should tests explicitly check with hierarchical priors?
   - Should tests verify that partial parameter sets raise errors?

## Recommendations

### For Users (Immediate)
1. **Always pass the complete dataset**:
   ```python
   # Good
   curve = mmm.saturation.sample_curve(mmm.idata.posterior, max_value=2)

   # Bad - will sample hierarchical parameters from priors
   curve = mmm.saturation.sample_curve(
       mmm.idata.posterior[["saturation_beta", "saturation_lam"]],
       max_value=2
   )
   ```

2. **If you must subset**:
   - Use `.sel()` or `.isel()` to select along chain/draw dimensions
   - Never use variable selection (e.g., `[["var1", "var2"]]`)
   - Example:
   ```python
   # Good - selects first draw but keeps all variables
   curve = mmm.saturation.sample_curve(
       mmm.idata.posterior.isel(draw=[0]),
       max_value=2
   )
   ```

### For Developers (Long-term)

1. **Add Parameter Validation**:
   - Check if all variables from the rebuilt model exist in the parameters dataset
   - Raise an informative error if hierarchical variables are missing
   - Example implementation location: `base.py:_sample_curve` around line 531

2. **Improve Documentation**:
   - Update `sample_curve` docstrings to explicitly state "pass the entire posterior"
   - Add a "Common Mistakes" section
   - Include examples with hierarchical priors

3. **Add Tests**:
   - Test that `sample_curve` is deterministic with the same full parameters
   - Test that `sample_curve` raises or warns with incomplete parameters
   - Specifically test with `LogNormalPrior(centered=False)` and nested Priors

4. **Consider API Changes** (Breaking):
   - Rename to `sample_curve_from_inference_data(idata)`
   - Require `InferenceData` object instead of `Dataset`
   - Automatically extract the appropriate group (posterior/prior)

## Related Research

No existing research documents found in `thoughts/shared/research/` directory for this topic.

## Conclusion

The issue is **not a bug** but rather a **documentation and UX problem**. PyMC's `sample_posterior_predictive` is working as designed, but the interaction between:
- Model recreation in `_sample_curve`
- Hierarchical priors in `LogNormalPrior`
- User expectations about "complete" parameters

creates a confusing user experience. The fix is simple: pass the complete dataset. However, better validation and documentation would prevent this confusion in the future.
