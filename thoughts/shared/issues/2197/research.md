---
date: 2026-01-16T15:41:00+00:00
researcher: Claude Sonnet 4.5
git_commit: d19500dcfc23d24e347883ad11b4257e29ba9722
branch: work-issue-2197
repository: pymc-labs/pymc-marketing
topic: "Add sample_adstock_curve() Method to MMM"
tags: [research, codebase, mmm, adstock, saturation, transformations, issue-2197]
status: complete
last_updated: 2026-01-16
last_updated_by: Claude Sonnet 4.5
issue_number: 2197
---

# Research: Add sample_adstock_curve() Method to MMM

**Date**: 2026-01-16T15:41:00+00:00
**Researcher**: Claude Sonnet 4.5
**Git Commit**: d19500dcfc23d24e347883ad11b4257e29ba9722
**Branch**: work-issue-2197
**Repository**: pymc-labs/pymc-marketing
**Issue**: #2197

## Research Question

How should we implement the `sample_adstock_curve()` method for the MMM class, following the pattern established by PR #2195 which added `sample_saturation_curve()`? The method should use the signature:

```python
def sample_adstock_curve(
    self,
    amount: float = 1.0,        # Amount to apply the adstock transformation to
    num_samples: int | None = 500,  # Number of posterior samples (None = all)
    random_state: RandomState | None = None,  # For reproducible subsampling
) -> xr.DataArray:
```

It should NOT have an `original_scale` argument.

## Summary

The implementation should follow the established pattern from `sample_saturation_curve()` (PR #2195) while accounting for the key differences between saturation and adstock transformations:

1. **Existing Infrastructure**: The `AdstockTransformation.sample_curve()` method already exists at `pymc_marketing/mmm/components/adstock.py:142-174` and takes `parameters` and `amount` arguments
2. **Pattern to Follow**: The `MMM.sample_saturation_curve()` method at `pymc_marketing/mmm/multidimensional.py:1780-1945` provides the template for posterior sampling, subsampling, and result formatting
3. **Key Difference**: Adstock curves represent decay over time (not diminishing returns like saturation), so they should NOT include `original_scale` parameter as specified in the issue
4. **Return Dimensions**: The adstock curve returns dimensions `(time since exposure, channel, sample)` vs saturation's `(x, channel, sample)`

## Detailed Findings

### 1. Reference Implementation: sample_saturation_curve() (PR #2195)

**Location**: `pymc_marketing/mmm/multidimensional.py:1780-1945`

The `sample_saturation_curve()` method provides the architectural pattern to follow:

**Key Components**:
1. **Validation**: Checks that model is fitted and posterior exists (lines 1886-1895)
2. **Posterior Subsampling**: Optionally subsamples posterior if `num_samples < total_samples` (lines 1897-1921)
3. **Delegation**: Calls the transformation's `sample_curve()` method (lines 1926-1930)
4. **Result Formatting**: Flattens chain/draw dimensions to single "sample" dimension (line 1933)
5. **Optional Scaling**: Applies target_scale when `original_scale=True` (lines 1936-1943)

**Subsampling Logic** (lines 1906-1919):
```python
rng = np.random.default_rng(random_state)
flat_indices = rng.choice(total_samples, size=num_samples, replace=False)

# Stack chain/draw into single dimension, select samples, reshape to chain=1
stacked = posterior.stack(sample=("chain", "draw"))
selected = stacked.isel(sample=flat_indices)
params = (
    selected.drop_vars(["chain", "draw"])
    .rename({"sample": "draw"})
    .expand_dims("chain")
)
```

This approach:
- Randomly samples across all chains/draws without replacement
- Maintains chain/draw structure for compatibility with `sample_curve()`
- Reshapes to chain=1 after sampling

### 2. Underlying Transformation: AdstockTransformation.sample_curve()

**Location**: `pymc_marketing/mmm/components/adstock.py:142-174`

The adstock transformation already implements a `sample_curve()` method:

**Signature**:
```python
def sample_curve(
    self,
    parameters: xr.Dataset,
    amount: float = 1.0,
) -> xr.DataArray:
```

**Implementation Details**:
- Creates time array: `np.arange(0, self.l_max)` where `l_max` is the maximum lag
- Creates coordinate dict with dimension "time since exposure"
- Initializes impulse response: zeros with spike at time 0 (`x[0] = amount`)
- Delegates to base `_sample_curve()` with var_name "adstock"

**Key Difference from Saturation**: Adstock curves show the effect of a single impulse (`amount`) decaying over time, while saturation curves show diminishing returns across spending levels.

### 3. Base Transformation Architecture

**Base Class**: `pymc_marketing/mmm/components/base.py:117-656`

The `Transformation` class provides the `_sample_curve()` method (lines 495-534) that both saturation and adstock use:

**Process**:
1. Infers output dimensions from priors
2. Expands x dimensions for broadcasting
3. Creates PyMC model context with coordinates
4. Defines deterministic using `self.apply()`
5. Samples via `pm.sample_posterior_predictive()`

### 4. Adstock Transformation Classes

**Available Adstock Types** (all in `pymc_marketing/mmm/components/adstock.py`):

1. **GeometricAdstock** (lines 210-241) - `lookup_name = "geometric"`
   - Most commonly used in examples
   - Default prior: `alpha`: Beta(alpha=1, beta=3)

2. **DelayedAdstock** (lines 243-282) - `lookup_name = "delayed"`
   - Default in BaseMMM class
   - Default priors: `alpha`: Beta(alpha=1, beta=3), `theta`: HalfNormal(sigma=1)

3. **WeibullPDFAdstock** (lines 284-324) - `lookup_name = "weibull_pdf"`
   - Default priors: `lam`: Gamma(mu=2, sigma=1), `k`: Gamma(mu=3, sigma=1)

4. **WeibullCDFAdstock** (lines 327-367) - `lookup_name = "weibull_cdf"`
   - Same priors as WeibullPDFAdstock

5. **BinomialAdstock** (lines 177-208) - `lookup_name = "binomial"`
   - Default prior: `alpha`: Beta(alpha=1, beta=3)

6. **NoAdstock** (lines 368-382) - `lookup_name = "no_adstock"`
   - Identity transformation, no parameters

### 5. MMM Class Integration

**Location**: `pymc_marketing/mmm/mmm.py:84-266`

The MMM class stores transformations as instance attributes:
- `self.adstock`: AdstockTransformation instance (line 191)
- `self.saturation`: SaturationTransformation instance (line 192)
- `self.adstock_first`: Boolean for ordering (line 193)

The `sample_adstock_curve()` method should be added to the MMM class following the same pattern as `sample_saturation_curve()`.

### 6. Test Infrastructure

**Primary Test File**: `tests/mmm/test_multidimensional_saturation_curve.py`

This file provides comprehensive test coverage for `sample_saturation_curve()` and serves as a template for adstock curve tests:

**Test Categories**:
1. **Basic Functionality** (lines 29-233):
   - Return type validation
   - Dimension checking (simple and panel models)
   - Parameter control (num_points, num_samples)
   - Edge cases (num_samples > total, num_samples = None)
   - Reproducibility (random_state with int and Generator)
   - Coordinate range validation

2. **Scaling Tests** (lines 235-304):
   - Original vs scaled comparison
   - X coordinate behavior
   - Target scale usage

3. **Validation Tests** (lines 306-386):
   - Invalid parameter handling
   - Unfitted model detection
   - Missing posterior detection

4. **Integration Tests** (lines 388-496):
   - Curve properties (monotonic increasing for saturation)
   - Numerical stability
   - Panel model support
   - Default parameters
   - Plotting compatibility

**Test Fixtures** (`tests/mmm/conftest.py`):
- `simple_mmm_data`: Simple single-dimension MMM data
- `panel_mmm_data`: Panel/multidimensional MMM data
- `simple_fitted_mmm`: Fitted simple MMM instance
- `panel_fitted_mmm`: Fitted panel MMM instance

**Existing Adstock Tests**: `tests/mmm/components/test_adstock.py` contains tests for the underlying `AdstockTransformation.sample_curve()` method, including `test_adstock_sample_curve()`.

## Code References

- `pymc_marketing/mmm/multidimensional.py:1780-1945` - Reference `sample_saturation_curve()` implementation
- `pymc_marketing/mmm/components/adstock.py:142-174` - Existing `AdstockTransformation.sample_curve()` method
- `pymc_marketing/mmm/components/saturation.py:154-193` - Comparable `SaturationTransformation.sample_curve()` method
- `pymc_marketing/mmm/components/base.py:495-534` - Base `_sample_curve()` implementation
- `pymc_marketing/mmm/mmm.py:191` - MMM stores `self.adstock` attribute
- `tests/mmm/test_multidimensional_saturation_curve.py` - Comprehensive test suite to use as template

## Implementation Plan

### Step 1: Add Method to MMM Class

Add the following method to `pymc_marketing/mmm/multidimensional.py` (likely after `sample_saturation_curve()`):

```python
def sample_adstock_curve(
    self,
    amount: float = Field(
        1.0, gt=0, description="Amount to apply the adstock transformation to."
    ),
    num_samples: int | None = Field(
        500, gt=0, description="Number of posterior samples to use."
    ),
    random_state: RandomState | None = None,
) -> xr.DataArray:
    """Sample adstock curves from posterior parameters.

    This method samples the adstock transformation curves using posterior
    parameters from the fitted model. It allows visualization of the
    carryover effect of media exposure over time.

    Parameters
    ----------
    amount : float, optional
        Amount to apply the adstock transformation to. By default 1.0.
        This represents an impulse of spend at time 0, and the curve
        shows how this effect decays over subsequent time periods.
    num_samples : int or None, optional
        Number of posterior samples to use for generating curves. By default 500.
        Samples are drawn randomly from the full posterior (across all chains
        and draws). Using fewer samples speeds up computation and reduces memory
        usage while still capturing posterior uncertainty. If None, all posterior
        samples are used without subsampling.
    random_state : int, np.random.Generator, or None, optional
        Random state for reproducible subsampling. Can be an integer seed,
        a numpy Generator instance, or None for non-reproducible sampling.
        Only used when num_samples is not None and less than total available
        samples.

    Returns
    -------
    xr.DataArray
        Sampled adstock curves with dimensions:
        - Simple model: (time since exposure, channel, sample)
        - Panel model: (time since exposure, *custom_dims, channel, sample)

        The "sample" dimension indexes the posterior samples used.
        The "time since exposure" coordinate represents time periods from 0
        to l_max (the maximum lag for the adstock transformation).

    Raises
    ------
    ValueError
        If called before model is fitted (idata doesn't exist)
    ValueError
        If idata exists but no posterior (model not fitted)

    Examples
    --------
    Sample curves with default parameters:

    >>> curves = mmm.sample_adstock_curve()
    >>> curves.dims
    ('sample', 'time since exposure', 'channel')

    Sample curves using all posterior samples:

    >>> curves_all = mmm.sample_adstock_curve(num_samples=None)

    Sample curves with custom amount and reproducible sampling:

    >>> curves = mmm.sample_adstock_curve(
    ...     amount=100.0, num_samples=1000, random_state=42
    ... )

    Notes
    -----
    - The adstock curve shows the carryover effect of a single impulse of
      media exposure over time, unlike saturation curves which show
      diminishing returns.
    - For panel models, curves are generated for each combination of custom
      dimensions (e.g., each country) and channel.
    - The returned array includes a "sample" dimension for uncertainty
      quantification. Use `.mean(dim='sample')` for point estimates and
      `.quantile()` for credible intervals.
    - Posterior samples are drawn randomly without replacement when num_samples
      is less than the total available samples.
    """
    self._validate_idata_exists()

    # Validate that posterior exists
    if (
        not hasattr(self.idata, "posterior") or self.idata.posterior is None
    ):
        raise ValueError(
            "posterior not found in idata. "
            "The model must be fitted (call .fit()) before sampling adstock curves."
        )

    # Step 1: Subsample posterior
    posterior = self.idata.posterior

    n_chains = posterior.sizes["chain"]
    n_draws = posterior.sizes["draw"]
    total_samples = n_chains * n_draws

    # Subsample from posterior if needed
    if num_samples is not None and num_samples < total_samples:
        rng = np.random.default_rng(random_state)
        # Randomly select samples across all chains/draws
        flat_indices = rng.choice(total_samples, size=num_samples, replace=False)

        # Stack chain/draw into single dimension, select samples, reshape to chain=1
        stacked = posterior.stack(sample=("chain", "draw"))
        selected = stacked.isel(sample=flat_indices)
        # Drop the multi-index coords before renaming to avoid conflicts
        params = (
            selected.drop_vars(["chain", "draw"])
            .rename({"sample": "draw"})
            .expand_dims("chain")
        )
    else:
        params = posterior

    # Step 2: Sample curve using transformation's method
    # This automatically handles channel dimensions
    curve = self.adstock.sample_curve(
        parameters=params,
        amount=amount,
    )

    # Flatten chain/draw to 'sample' dimension for consistent output
    curve = curve.stack(sample=("chain", "draw"))

    # Note: No scaling step - adstock curves represent time decay,
    # not contribution to target variable

    return curve
```

### Step 2: Create Comprehensive Test File

Create `tests/mmm/test_multidimensional_adstock_curve.py` following the structure of `test_multidimensional_saturation_curve.py`:

**Test Categories to Include**:

1. **Basic Functionality**:
   - `test_sample_adstock_curve_returns_dataarray`
   - `test_sample_adstock_curve_has_correct_dims_simple_model`
   - `test_sample_adstock_curve_has_correct_dims_panel_model`
   - `test_sample_adstock_curve_num_samples_controls_shape`
   - `test_sample_adstock_curve_uses_all_samples_when_num_samples_exceeds_total`
   - `test_sample_adstock_curve_uses_all_samples_when_num_samples_is_none`
   - `test_sample_adstock_curve_random_state_reproducibility`
   - `test_sample_adstock_curve_random_state_different_seeds_differ`
   - `test_sample_adstock_curve_random_state_with_generator`
   - `test_sample_adstock_curve_time_coordinate_range`

2. **Validation Tests**:
   - `test_sample_adstock_curve_raises_on_invalid_amount`
   - `test_sample_adstock_curve_raises_on_invalid_num_samples`
   - `test_sample_adstock_curve_raises_on_unfitted_model`
   - `test_sample_adstock_curve_raises_when_no_posterior`

3. **Integration Tests**:
   - `test_sample_adstock_curve_curves_decay_over_time` (adstock-specific property)
   - `test_sample_adstock_curve_with_panel_model_works`
   - `test_sample_adstock_curve_default_parameters`
   - `test_sample_adstock_curve_can_be_used_for_plotting`
   - `test_sample_adstock_curve_amount_scales_linearly` (adstock-specific)

4. **Different Adstock Types**:
   - Test with GeometricAdstock, DelayedAdstock, WeibullPDFAdstock, etc.

### Step 3: Update Documentation

Add usage examples to:
- Docstring (already included above)
- User guide documentation if applicable
- Notebook examples showing adstock curve visualization

### Step 4: Key Differences from Saturation Implementation

**Removed Components**:
- No `original_scale` parameter (as specified in issue)
- No `max_value` parameter (adstock uses fixed time range 0 to l_max)
- No `num_points` parameter (time steps are discrete: 0, 1, 2, ..., l_max-1)
- No target_scale scaling step (adstock is about time decay, not contribution scaling)

**Modified Components**:
- Different coordinate name: "time since exposure" instead of "x"
- Different curve property tests: decay over time instead of monotonic increasing
- Different interpretation: carryover effect vs diminishing returns

## Architecture Insights

### Transformation System Design

The MMM transformation architecture uses a clean separation of concerns:

1. **Base Transformation Class** (`components/base.py`): Provides common infrastructure for all transformations including:
   - Prior management and validation
   - PyMC integration via `apply()` and `_create_distributions()`
   - Serialization/deserialization
   - Curve sampling via `_sample_curve()`

2. **Specialized Transformation Classes**:
   - `AdstockTransformation` for temporal carryover effects
   - `SaturationTransformation` for diminishing returns
   - Both implement `sample_curve()` with appropriate parameters

3. **MMM-Level Methods**: High-level convenience methods like `sample_saturation_curve()` and (to be added) `sample_adstock_curve()` that:
   - Handle posterior sampling and subsampling
   - Provide user-friendly interfaces with validation
   - Format results consistently
   - Optionally apply scaling transformations

This three-layer architecture allows for:
- Easy addition of new transformation types
- Consistent interface across transformations
- Separation between mathematical transformations and statistical inference
- Reusable components across different MMM variants

### Design Pattern: Template Method

Both `sample_saturation_curve()` and `sample_adstock_curve()` follow the Template Method pattern:

1. **Validate** model state (fitted, has posterior)
2. **Subsample** posterior if requested
3. **Delegate** to transformation's `sample_curve()` method
4. **Format** result (flatten chain/draw to sample dimension)
5. **Transform** if needed (scale to original units - only for saturation)

This consistent structure makes the codebase maintainable and predictable.

## Open Questions

1. **Should adstock curves include confidence intervals in plotting helpers?** The sample dimension enables uncertainty quantification via `.mean()` and `.quantile()`, but plot helpers may need to be created or updated.

2. **Should we add a convenience method for comparing saturation and adstock curves side-by-side?** This could help users understand the combined effect of both transformations.

3. **Do we need additional validation for the `amount` parameter?** The current spec uses `gt=0` (must be positive), which makes sense for impulse responses.

4. **Should we add support for sampling curves at different amounts simultaneously?** E.g., `amounts=[0.5, 1.0, 2.0]` to see how the adstock effect scales. (This could be a follow-up enhancement.)

## Related Research

This research builds on the recently added `sample_saturation_curve()` functionality from PR #2195 (commit d19500dc). The implementation should maintain consistency with that pattern while accounting for the specific characteristics of adstock transformations.

## Next Steps

1. Implement `sample_adstock_curve()` method following the pattern above
2. Create comprehensive test file `test_multidimensional_adstock_curve.py`
3. Verify all tests pass, especially:
   - Dimension checking
   - Reproducibility
   - Panel model support
   - Different adstock types
4. Update any relevant documentation or examples
5. Consider adding plotting utilities specifically for adstock curves
