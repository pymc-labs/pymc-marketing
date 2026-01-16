# Add sample_adstock_curve() Method to MMM - Implementation Plan

**Issue**: #2197
**Date**: 2026-01-16
**Branch**: work-issue-2197
**Related Research**: `thoughts/shared/issues/2197/research.md`

## Overview

This plan implements a new `sample_adstock_curve()` method for the MMM class that allows users to visualize adstock (carryover) effects from posterior samples. This follows the pattern established by PR #2195 which added `sample_saturation_curve()`, adapting it for the specific characteristics of adstock transformations.

The method will enable users to visualize how media exposure effects decay over time, which is essential for understanding the carryover dynamics of their marketing activities.

## Current State Analysis

### Existing Infrastructure

1. **AdstockTransformation.sample_curve()** already exists at `pymc_marketing/mmm/components/adstock.py:142-174`
   - Takes `parameters` (xr.Dataset) and `amount` (float) arguments
   - Creates time array from 0 to l_max
   - Uses "time since exposure" coordinate
   - Returns adstock decay curve

2. **MMM.sample_saturation_curve()** at `pymc_marketing/mmm/multidimensional.py:1780-1945`
   - Provides the architectural template to follow
   - Implements validation, posterior subsampling, delegation, and formatting
   - Uses Pydantic Field validation for parameters

3. **Test Infrastructure** at `tests/mmm/test_multidimensional_saturation_curve.py`
   - Comprehensive 501-line test suite
   - Organized into Basic Functionality, Scaling, Validation, and Integration tests
   - Uses fixtures: `simple_fitted_mmm`, `panel_fitted_mmm`, `simple_mmm_data`, `panel_mmm_data`

### Key Differences from Saturation

- **No original_scale parameter**: Adstock curves represent time decay, not contribution to target variable
- **No max_value or num_points parameters**: Time range is fixed (0 to l_max), discrete time steps
- **Different coordinate name**: "time since exposure" vs "x"
- **Different mathematical property**: Decay over time vs monotonic increasing
- **Single input amount**: Impulse response at time 0 vs range of spending levels

## Desired End State

After implementation:

1. **Method Available**: Users can call `mmm.sample_adstock_curve()` on fitted MMM instances
2. **Return Value**: xr.DataArray with dimensions (time since exposure, channel, sample) for simple models, (time since exposure, *custom_dims, channel, sample) for panel models
3. **Visualization Ready**: Returned arrays support `.mean()`, `.quantile()`, and other operations for plotting
4. **Well Tested**: Comprehensive test coverage matching saturation curve tests (adapted for adstock specifics)
5. **Consistent API**: Follows same validation and error handling patterns as `sample_saturation_curve()`

### Verification

Success is verified when:
- Method can be called on fitted MMM instances without errors
- Returns correct xr.DataArray structure with expected dimensions
- Test suite passes with ~18 tests covering all scenarios
- Curves show decay over time (adstock-specific behavior)
- Documentation accurately describes parameters and return values

## What We're NOT Doing

- **No original_scale parameter**: As specified in issue, adstock curves don't need scaling to original units
- **No plotting utilities**: Focus on data generation; plotting can be added in follow-up if needed
- **No multi-amount support**: Feature to test multiple amounts simultaneously (e.g., [0.5, 1.0, 2.0]) is deferred to future enhancement
- **No parameterized test structure**: Tests will be in separate file, not parameterized with saturation tests (as per research recommendation)
- **No documentation updates beyond docstring**: User guide or notebook examples are out of scope for this PR

## Implementation Approach

Follow the Template Method pattern established by `sample_saturation_curve()`:

1. **Validate**: Check model is fitted and has posterior
2. **Subsample**: Optionally subsample posterior if num_samples < total
3. **Delegate**: Call `AdstockTransformation.sample_curve()`
4. **Format**: Flatten chain/draw to single "sample" dimension
5. **Return**: No scaling step (unlike saturation)

The implementation will be simpler than saturation because:
- Fewer parameters to validate (no max_value, num_points, original_scale)
- No scaling logic needed
- Fixed time range determined by l_max

## Phase 1: Implement Core Method

### Overview
Add `sample_adstock_curve()` method to MMM class with full validation, posterior sampling, and result formatting.

### Changes Required

#### 1. MMM Class Method
**File**: `pymc_marketing/mmm/multidimensional.py`
**Location**: After `sample_saturation_curve()` method (after line 1945)
**Changes**: Add new method

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
        not hasattr(self.idata, "posterior") or self.idata.posterior is None  # type: ignore[union-attr]
    ):
        raise ValueError(
            "posterior not found in idata. "
            "The model must be fitted (call .fit()) before sampling adstock curves."
        )

    # Step 1: Subsample posterior
    posterior = self.idata.posterior  # type: ignore[union-attr]

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

### Success Criteria

#### Automated Verification:
- [ ] Code passes linting: `make lint` or equivalent
- [ ] Type checking passes: `mypy pymc_marketing/mmm/multidimensional.py`
- [ ] Method can be imported: `from pymc_marketing.mmm.multidimensional import MMM; hasattr(MMM, 'sample_adstock_curve')`
- [ ] No syntax errors when loading module

#### Manual Verification:
- [ ] Method signature matches specification exactly
- [ ] Docstring is complete with all sections (Parameters, Returns, Raises, Examples, Notes)
- [ ] Code follows same structure as `sample_saturation_curve()`
- [ ] Type hints are correct (especially RandomState and xr.DataArray)
- [ ] Comments explain key steps clearly

---

## Phase 2: Create Comprehensive Test Suite

### Overview
Create new test file `tests/mmm/test_multidimensional_adstock_curve.py` with ~18 tests covering basic functionality, validation, and integration scenarios.

### Changes Required

#### 1. Test File
**File**: `tests/mmm/test_multidimensional_adstock_curve.py` (NEW)
**Changes**: Create complete test suite

**Structure** (following saturation test organization):

```python
#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Tests for MMM.sample_adstock_curve() method.

Note: Fixtures `simple_mmm_data`, `panel_mmm_data`, `simple_fitted_mmm`, and
`panel_fitted_mmm` are defined in tests/mmm/conftest.py and automatically
available to all tests in this module.
"""

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM

# ============================================================================
# Basic Functionality Tests
# ============================================================================


def test_sample_adstock_curve_returns_dataarray(simple_fitted_mmm):
    """Test that sample_adstock_curve returns xr.DataArray."""
    # Act
    curves = simple_fitted_mmm.sample_adstock_curve()

    # Assert
    assert isinstance(curves, xr.DataArray)


def test_sample_adstock_curve_has_correct_dims_simple_model(simple_fitted_mmm):
    """Test that curves have correct dimensions for simple model.

    Note: The dimensions depend on how the adstock transformation's
    priors are configured. With default priors without channel dims,
    the output will be (time since exposure, sample). With channel-specific
    priors, it would include a channel dimension.
    """
    # Act
    curves = simple_fitted_mmm.sample_adstock_curve()

    # Assert - should have sample and time since exposure dims at minimum
    assert "sample" in curves.dims
    assert "time since exposure" in curves.dims


def test_sample_adstock_curve_has_correct_dims_panel_model(panel_fitted_mmm):
    """Test that curves have correct dimensions for panel model.

    Note: The dimensions depend on how the adstock transformation's
    priors are configured. Default priors result in (time since exposure, sample).
    """
    # Act
    curves = panel_fitted_mmm.sample_adstock_curve()

    # Assert - should have sample and time since exposure dims at minimum
    assert "sample" in curves.dims
    assert "time since exposure" in curves.dims


def test_sample_adstock_curve_num_samples_controls_shape(simple_fitted_mmm):
    """Test that num_samples parameter controls number of posterior samples.

    Note: With mock_pymc_sample, we get a small number of samples.
    This test verifies that when num_samples < total, we get num_samples.
    """
    # Arrange - use a smaller num_samples than available
    total_available = (
        simple_fitted_mmm.idata.posterior.sizes["chain"]
        * simple_fitted_mmm.idata.posterior.sizes["draw"]
    )
    num_samples = min(5, total_available - 1)  # Request fewer than available

    # Skip if we don't have enough samples to test subsampling
    if total_available <= 2:
        pytest.skip("Not enough posterior samples to test subsampling")

    # Act
    curves = simple_fitted_mmm.sample_adstock_curve(num_samples=num_samples)

    # Assert - should have exactly num_samples
    assert curves.sizes["sample"] == num_samples


def test_sample_adstock_curve_uses_all_samples_when_num_samples_exceeds_total(
    simple_fitted_mmm,
):
    """Test that all samples are used when num_samples > total available."""
    # Arrange - Request more samples than available
    num_samples = 10000

    # Act
    curves = simple_fitted_mmm.sample_adstock_curve(num_samples=num_samples)

    # Assert - Should get all available samples, not num_samples
    total_available = (
        simple_fitted_mmm.idata.posterior.sizes["chain"]
        * simple_fitted_mmm.idata.posterior.sizes["draw"]
    )
    assert curves.sizes["sample"] == total_available


def test_sample_adstock_curve_uses_all_samples_when_num_samples_is_none(
    simple_fitted_mmm,
):
    """Test that all samples are used when num_samples is None."""
    # Act
    curves = simple_fitted_mmm.sample_adstock_curve(num_samples=None)

    # Assert - Should get all available samples
    total_available = (
        simple_fitted_mmm.idata.posterior.sizes["chain"]
        * simple_fitted_mmm.idata.posterior.sizes["draw"]
    )
    assert curves.sizes["sample"] == total_available


def test_sample_adstock_curve_random_state_reproducibility(simple_fitted_mmm):
    """Test that random_state produces reproducible results."""
    # Arrange
    num_samples = 50
    random_state = 42

    # Act - Sample twice with same random_state
    curves1 = simple_fitted_mmm.sample_adstock_curve(
        num_samples=num_samples, random_state=random_state
    )
    curves2 = simple_fitted_mmm.sample_adstock_curve(
        num_samples=num_samples, random_state=random_state
    )

    # Assert - Results should be identical
    xr.testing.assert_equal(curves1, curves2)


def test_sample_adstock_curve_random_state_different_seeds_differ(simple_fitted_mmm):
    """Test that different random_state values produce different results.

    Note: This only works when subsampling (num_samples < total available).
    With mock_pymc_sample, we may have limited samples.
    """
    # Arrange - use smaller num_samples to ensure subsampling happens
    total_available = (
        simple_fitted_mmm.idata.posterior.sizes["chain"]
        * simple_fitted_mmm.idata.posterior.sizes["draw"]
    )
    num_samples = min(5, total_available - 1)

    # Skip if we don't have enough samples to test different subsampling
    if total_available <= 5:
        pytest.skip("Not enough posterior samples to test different subsampling")

    # Act - Sample with different random states
    curves1 = simple_fitted_mmm.sample_adstock_curve(
        num_samples=num_samples, random_state=42
    )
    curves2 = simple_fitted_mmm.sample_adstock_curve(
        num_samples=num_samples, random_state=123
    )

    # Assert - Results should differ (different posterior samples selected)
    assert not np.allclose(curves1.values, curves2.values)


def test_sample_adstock_curve_random_state_with_generator(simple_fitted_mmm):
    """Test that random_state accepts numpy Generator."""
    # Arrange - use smaller num_samples to ensure subsampling happens
    total_available = (
        simple_fitted_mmm.idata.posterior.sizes["chain"]
        * simple_fitted_mmm.idata.posterior.sizes["draw"]
    )
    num_samples = min(5, total_available - 1)

    # Skip if we don't have enough samples to test subsampling
    if total_available <= 2:
        pytest.skip("Not enough posterior samples to test subsampling")

    rng = np.random.default_rng(42)

    # Act - Should not raise
    curves = simple_fitted_mmm.sample_adstock_curve(
        num_samples=num_samples, random_state=rng
    )

    # Assert
    assert curves.sizes["sample"] == num_samples


def test_sample_adstock_curve_time_coordinate_range(simple_fitted_mmm):
    """Test that time coordinate spans from 0 to l_max-1."""
    # Act
    curves = simple_fitted_mmm.sample_adstock_curve()

    # Assert
    time_coords = curves.coords["time since exposure"].values
    assert time_coords[0] == pytest.approx(0.0)
    # Maximum time should be l_max - 1 (0-indexed)
    l_max = simple_fitted_mmm.adstock.l_max
    assert np.max(time_coords) == pytest.approx(l_max - 1)


# ============================================================================
# Parameter Validation Tests
# ============================================================================


@pytest.mark.parametrize("amount", [0, -1])
def test_sample_adstock_curve_raises_on_invalid_amount(
    simple_fitted_mmm, amount
):
    """Test that invalid amount raises ValidationError."""
    # Act & Assert
    with pytest.raises(ValidationError):
        simple_fitted_mmm.sample_adstock_curve(amount=amount)


@pytest.mark.parametrize("num_samples", [0, -1])
def test_sample_adstock_curve_raises_on_invalid_num_samples(
    simple_fitted_mmm, num_samples
):
    """Test that invalid num_samples raises ValidationError.

    Note: None is valid (uses all samples), but 0 and negative are not.
    """
    # Act & Assert
    with pytest.raises(ValidationError):
        simple_fitted_mmm.sample_adstock_curve(num_samples=num_samples)


def test_sample_adstock_curve_raises_on_unfitted_model():
    """Test that calling on unfitted model raises ValueError."""
    # Arrange - Create unfitted MMM
    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="target",
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )

    # Act & Assert
    with pytest.raises(ValueError, match="idata does not exist"):
        mmm.sample_adstock_curve()


def test_sample_adstock_curve_raises_when_no_posterior(simple_mmm_data):
    """Test that calling raises ValueError when idata exists but has no posterior.

    This can happen if only sample_prior_predictive() was called but not fit().
    """
    import arviz as az

    # Arrange - Create MMM with idata but no posterior
    X = simple_mmm_data["X"]
    y = simple_mmm_data["y"]

    mmm = MMM(
        channel_columns=["channel_1", "channel_2", "channel_3"],
        date_column="date",
        target_column="target",
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )

    # Build model and create empty idata (simulating prior-only sampling)
    mmm.build_model(X, y)
    mmm.idata = az.InferenceData()  # Empty idata without posterior

    # Act & Assert
    with pytest.raises(ValueError, match="posterior not found in idata"):
        mmm.sample_adstock_curve()


# ============================================================================
# Integration Tests
# ============================================================================


def test_sample_adstock_curve_curves_decay_over_time(simple_fitted_mmm):
    """Test that sampled curves decay over time (adstock-specific behavior).

    Adstock curves should generally decrease as time since exposure increases,
    showing the carryover effect diminishing over time.

    Note: We check that the curve starts higher and ends lower, which is the
    characteristic behavior of adstock transformations.
    """
    # Act
    curves = simple_fitted_mmm.sample_adstock_curve()

    # Assert - Check that mean curve shows decay behavior
    mean_curve = curves.mean(dim="sample")

    # First value (time 0) should be higher than last value (time l_max-1)
    # This is the characteristic decay of adstock effects
    first_val = float(mean_curve.isel({"time since exposure": 0}).values)
    last_val = float(mean_curve.isel({"time since exposure": -1}).values)

    assert first_val > last_val, "Adstock curve should decay over time"


def test_sample_adstock_curve_amount_scales_linearly(simple_fitted_mmm):
    """Test that doubling amount approximately doubles curve values (linearity).

    Adstock transformations should be linear with respect to the input amount.
    """
    # Act
    curves_1x = simple_fitted_mmm.sample_adstock_curve(amount=1.0)
    curves_2x = simple_fitted_mmm.sample_adstock_curve(amount=2.0)

    # Assert - 2x amount should give approximately 2x values
    # Use mean across samples for simpler comparison
    mean_1x = curves_1x.mean(dim="sample")
    mean_2x = curves_2x.mean(dim="sample")

    # Check approximate 2x relationship (within 10% tolerance for numerical stability)
    ratio = mean_2x / mean_1x
    assert np.allclose(ratio.values, 2.0, rtol=0.1)


def test_sample_adstock_curve_with_panel_model_works(panel_fitted_mmm):
    """Test that panel model curves can be sampled.

    Note: The adstock transformation's sample_curve returns
    (time since exposure, sample) dimensions. Custom dimensions
    (like country) would only appear if the adstock priors were
    configured with those dimensions.
    """
    # Act
    curves = panel_fitted_mmm.sample_adstock_curve()

    # Assert - Should have time since exposure and sample dimensions
    assert "time since exposure" in curves.dims
    assert "sample" in curves.dims
    # Should be a valid DataArray with values
    assert curves.sizes["time since exposure"] > 0
    assert curves.sizes["sample"] > 0


def test_sample_adstock_curve_default_parameters(simple_fitted_mmm):
    """Test that default parameters produce expected output."""
    # Act
    curves = simple_fitted_mmm.sample_adstock_curve()

    # Assert - time coordinate spans from 0 to l_max-1
    l_max = simple_fitted_mmm.adstock.l_max
    assert curves.sizes["time since exposure"] == l_max
    # Time values start at 0
    assert curves.coords["time since exposure"].values[0] == pytest.approx(0.0)


def test_sample_adstock_curve_can_be_used_for_plotting(simple_fitted_mmm):
    """Test that the returned array can be used for common plotting operations."""
    # Act
    curves = simple_fitted_mmm.sample_adstock_curve()

    # Assert - These operations should work without error
    mean_curves = curves.mean(dim="sample")
    assert isinstance(mean_curves, xr.DataArray)

    lower = curves.quantile(0.05, dim="sample")
    upper = curves.quantile(0.95, dim="sample")
    assert isinstance(lower, xr.DataArray)
    assert isinstance(upper, xr.DataArray)

    # time coordinate should be usable for plotting
    time_values = curves.coords["time since exposure"].values
    assert len(time_values) > 0
    assert time_values[0] == 0.0  # Starts at 0
```

**Test Count**: ~18 tests
- Basic Functionality: 10 tests
- Validation: 4 tests
- Integration: 4 tests

### Success Criteria

#### Automated Verification:
- [ ] All tests pass: `pytest tests/mmm/test_multidimensional_adstock_curve.py -v`
- [ ] Test coverage for new method: `pytest --cov=pymc_marketing.mmm.multidimensional --cov-report=term tests/mmm/test_multidimensional_adstock_curve.py`
- [ ] No test failures in CI pipeline
- [ ] Linting passes: `make lint` or `ruff check tests/mmm/test_multidimensional_adstock_curve.py`

#### Manual Verification:
- [ ] Test file structure matches saturation test organization
- [ ] Test names are descriptive and follow naming conventions
- [ ] Docstrings explain what each test verifies
- [ ] Edge cases are covered (unfitted model, no posterior, invalid params)
- [ ] Adstock-specific behaviors are tested (decay over time, linear scaling)

---

## Phase 3: Verify Integration

### Overview
Ensure the new method integrates properly with existing MMM functionality and doesn't break anything.

### Changes Required

#### 1. Run Full Test Suite
**Command**: `pytest tests/mmm/ -v`
**Purpose**: Ensure no regressions in existing tests

#### 2. Test with Different Adstock Types
**Manual Testing**: Verify method works with all adstock transformations:
- GeometricAdstock
- DelayedAdstock
- WeibullPDFAdstock
- WeibullCDFAdstock
- BinomialAdstock

**Test Script**:
```python
from pymc_marketing.mmm import (
    GeometricAdstock, DelayedAdstock,
    WeibullPDFAdstock, LogisticSaturation
)
from pymc_marketing.mmm.multidimensional import MMM
import pandas as pd

# Create sample data
X = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=20, freq="W-MON"),
    "C1": np.random.randint(50, 150, 20),
    "C2": np.random.randint(50, 150, 20),
})
y = pd.Series(np.random.randint(200, 300, 20), name="y")

# Test each adstock type
for adstock_cls in [GeometricAdstock, DelayedAdstock, WeibullPDFAdstock]:
    mmm = MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        target_column="y",
        adstock=adstock_cls(l_max=10),
        saturation=LogisticSaturation(),
    )
    mmm.fit(X, y, chains=1, draws=50)
    curves = mmm.sample_adstock_curve(num_samples=10)
    assert "time since exposure" in curves.dims
    assert "sample" in curves.dims
    print(f"✓ {adstock_cls.__name__} works")
```

### Success Criteria

#### Automated Verification:
- [ ] Full MMM test suite passes: `pytest tests/mmm/ -v`
- [ ] No new warnings or errors in test output
- [ ] Type checking passes: `mypy pymc_marketing/mmm/`
- [ ] All linting checks pass: `make lint`

#### Manual Verification:
- [ ] Method works with GeometricAdstock
- [ ] Method works with DelayedAdstock
- [ ] Method works with WeibullPDFAdstock
- [ ] Method works with simple (non-panel) models
- [ ] Method works with panel models
- [ ] Error messages are clear and helpful

---

## Testing Strategy

### Unit Tests (Phase 2)

**Basic Functionality** (10 tests):
- Return type validation (xr.DataArray)
- Dimension checking (simple and panel models)
- num_samples parameter control
- Edge cases (num_samples > total, num_samples = None)
- Reproducibility with random_state (int and Generator)
- Time coordinate range validation

**Validation Tests** (4 tests):
- Invalid amount (≤ 0) raises ValidationError
- Invalid num_samples (≤ 0) raises ValidationError
- Unfitted model raises ValueError
- Missing posterior raises ValueError

**Integration Tests** (4 tests):
- Curves decay over time (adstock-specific)
- Amount scales linearly (adstock property)
- Panel model support
- Default parameters work correctly
- Plotting operations (mean, quantile)

### Integration Testing (Phase 3)

**Different Adstock Types**:
- Test with each adstock transformation class
- Verify dimensions and values are reasonable
- Check l_max affects time dimension correctly

**Regression Testing**:
- Run full MMM test suite
- Verify no existing functionality breaks

### Manual Testing Steps

1. **Basic Usage**:
   ```python
   curves = mmm.sample_adstock_curve()
   assert "time since exposure" in curves.dims
   ```

2. **With Custom Amount**:
   ```python
   curves = mmm.sample_adstock_curve(amount=100.0)
   # Should scale linearly
   ```

3. **Reproducibility**:
   ```python
   c1 = mmm.sample_adstock_curve(random_state=42)
   c2 = mmm.sample_adstock_curve(random_state=42)
   assert xr.testing.assert_equal(c1, c2)
   ```

4. **Plotting**:
   ```python
   curves = mmm.sample_adstock_curve()
   mean = curves.mean(dim="sample")
   lower = curves.quantile(0.05, dim="sample")
   upper = curves.quantile(0.95, dim="sample")
   # Plot time vs mean with uncertainty bands
   ```

## Performance Considerations

1. **Memory Usage**:
   - Subsampling reduces memory: 500 samples (default) vs potentially 1000s
   - Time dimension is typically small (l_max ≈ 10-20)
   - Panel models may have large custom dimensions

2. **Computation Time**:
   - Faster than saturation curves (no num_points loop)
   - Fixed time array (0 to l_max)
   - Bottleneck is PyMC posterior_predictive sampling

3. **Optimization Opportunities** (future):
   - Cache results for same random_state
   - Parallelize across channels if needed
   - Not critical for initial implementation

## Migration Notes

Not applicable - this is a new feature addition with no breaking changes.

## References

- **Original Issue**: #2197
- **Research Document**: `thoughts/shared/issues/2197/research.md`
- **Reference PR**: #2195 (sample_saturation_curve implementation)
- **Reference Implementation**: `pymc_marketing/mmm/multidimensional.py:1780-1945` (sample_saturation_curve)
- **Underlying Method**: `pymc_marketing/mmm/components/adstock.py:142-174` (AdstockTransformation.sample_curve)
- **Test Template**: `tests/mmm/test_multidimensional_saturation_curve.py`
- **Test Fixtures**: `tests/mmm/conftest.py`

## Technical Decisions Made

1. **Separate Test File**: Create `test_multidimensional_adstock_curve.py` instead of parameterizing with saturation tests (per research recommendation)

2. **No Scaling Parameter**: Exclude `original_scale` as specified in issue - adstock curves represent time decay, not target contribution

3. **Fixed Time Range**: Use l_max to determine time range (0 to l_max-1) - no num_points parameter needed

4. **Pydantic Validation**: Use Field() for parameter validation following saturation pattern

5. **Type Hints**: Use RandomState union type for random_state parameter compatibility

6. **Error Messages**: Match existing pattern from sample_saturation_curve for consistency

7. **Test Coverage**: Target ~18 tests (60% of saturation tests) - exclude scaling-specific tests

## Assumptions

1. **l_max is available**: AdstockTransformation always has l_max attribute
2. **Posterior structure**: Assumes chain/draw dimensions exist in posterior
3. **Test fixtures work**: Existing conftest.py fixtures are compatible
4. **No plotting utilities needed**: Users can create plots from returned DataArray
5. **Linear scaling is expected**: Adstock transformations are linear (testable property)
