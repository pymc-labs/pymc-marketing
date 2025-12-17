---
date: 2025-12-17
issue_number: 1981
repository: pymc-marketing
branch: work-issue-1981
status: ready_for_implementation
title: "Fix Basis Functions with Multidimensional Events"
tags: [implementation-plan, events, dimensions, broadcasting, issue-1981]
---

# Fix Basis Functions with Multidimensional Events - Implementation Plan

## Overview

The EventEffect system currently fails when users attempt to create events with multidimensional parameters where effect_size varies by multiple dimensions (e.g., `dims=("sales", "country")`). This is caused by a dimension mismatch during broadcasting - the basis matrix `X` has implicit dimensions that are never communicated to the transformation system, leading to incompatible shapes when basis function parameters have multiple dimensions.

**Technical Root Cause**: The `Transformation.apply()` method only handles parameter dimensions, not data dimensions. When `x` has shape `(n_dates, n_sales)` and `sigma` has shape `(n_sales, n_countries)`, PyTensor's broadcasting cannot properly align them.

## Current State Analysis

### Key Discoveries:

1. **Data dimensions are not tracked** (`pymc_marketing/mmm/events.py:217-223`): The basis matrix `X` has implicit dimensions (e.g., "date", "sales") that are never communicated to the transformation system

2. **Parameter-only broadcasting** (`pymc_marketing/mmm/components/base.py:611-650`): The `Transformation.apply()` method's docstring explicitly states: "The dims of the parameters. Defaults to None. **Not the dims of the data!**"

3. **Single-dimension pattern throughout codebase**: All 36+ tests in `tests/mmm/test_events.py` use single-dimension patterns like `dims=("event",)` or `dims=("campaign",)`. No working examples of multidimensional events exist.

4. **Broadcasting mismatch in GaussianBasis.function** (`pymc_marketing/mmm/events.py:262-266`): The error manifests when `pm.logp(rv, x)` tries to broadcast incompatible shapes

5. **EventEffect.dims validation is permissive** (`pymc_marketing/mmm/events.py:204-215`): The validation allows multiple dimensions but doesn't check for data dimension compatibility

### Current Limitations:

- No data dimension tracking in the transformation system
- Parameter broadcasting only in `_create_distributions`
- All examples and tests use single dimension patterns
- Error manifests deep in basis function, not at validation

## Desired End State

After implementation:

1. **Clear error messages**: Users attempting multidimensional events receive a clear, actionable error message explaining the limitation
2. **Validation at construction**: The error occurs during `EventEffect` initialization, not deep in model building
3. **Documentation updated**: Users understand that only single-dimension events are currently supported
4. **Tests enforce limitation**: New tests verify that multidimensional attempts are properly rejected
5. **Future-ready**: The restriction is implemented in a way that can be removed when full multidimensional support is added

### Verification Criteria:

**Automated Verification:**
- [ ] Test suite passes: `pytest tests/mmm/test_events.py -v`
- [ ] New validation test passes: `pytest tests/mmm/test_events.py::test_multidimensional_event_validation -v`
- [ ] All existing tests continue to pass: `pytest tests/mmm/test_multidimensional.py -v`
- [ ] Type checking passes: `mypy pymc_marketing/mmm/events.py`

**Manual Verification:**
- [ ] Error message is clear and actionable when creating `EventEffect` with `dims=("sales", "country")`
- [ ] Error message includes suggestion to use single dimension
- [ ] Existing single-dimension examples in module docstring still work
- [ ] No regressions in MMM event integration

## What We're NOT Doing

- **NOT implementing full multidimensional support**: This would require significant architectural changes to the Transformation base class and dimension tracking system
- **NOT modifying Transformation base class**: Changes are scoped to EventEffect only
- **NOT changing existing APIs**: All current working code remains compatible
- **NOT adding new features**: This is purely a bug fix to prevent silent failures
- **NOT modifying basis function implementations**: GaussianBasis, HalfGaussianBasis, etc. remain unchanged

## Implementation Approach

**Strategy**: Add validation to EventEffect that restricts it to single dimensions until proper multidimensional support can be implemented. This is a pragmatic short-term fix that:

1. Prevents users from hitting cryptic broadcasting errors
2. Makes the current limitation explicit and documented
3. Maintains backward compatibility (all existing code uses single dimensions)
4. Can be easily removed when full support is added

**Key Decision**: We're implementing validation restrictions rather than attempting full multidimensional support because:
- Research shows all current tests use single dimensions
- Full support requires changes to the Transformation base class (affects all transformations)
- Data dimension tracking would need to be added throughout the system
- This provides immediate value while being easily reversible

## Phase 1: Add Validation and Error Messages

### Overview
Add validation to EventEffect that restricts dims to single dimension with clear error messages.

### Changes Required:

#### 1. EventEffect Validation in `pymc_marketing/mmm/events.py`

**File**: `pymc_marketing/mmm/events.py`
**Location**: Lines 204-215 (modify existing `_validate_dims` method)
**Changes**: Add validation for single dimension

```python
@model_validator(mode="after")
def _validate_dims(self):
    if not self.dims:
        raise ValueError("The dims must not be empty.")

    # NEW: Restrict to single dimension
    if len(self.dims) > 1:
        raise ValueError(
            f"EventEffect currently only supports single dimension. "
            f"Got dims={self.dims}. "
            f"Please use a single dimension like dims=('{self.dims[0]}',). "
            f"Multi-dimensional event effects where parameters vary across "
            f"multiple dimensions (e.g., events varying by country) are not "
            f"yet supported due to dimension broadcasting limitations in the "
            f"underlying transformation system. "
            f"See https://github.com/pymc-labs/pymc-marketing/issues/1981"
        )

    if not set(self.basis.combined_dims).issubset(set(self.dims)):
        raise ValueError("The dims must contain all dimensions of the basis.")

    if not set(self.effect_size.dims).issubset(set(self.dims)):
        raise ValueError("The dims must contain all dimensions of the effect size.")

    return self
```

**Rationale**:
- Error occurs at EventEffect construction, not during model building
- Clear explanation of the limitation and workaround
- Link to tracking issue for users who need this feature
- Validation happens in "after" mode so dims are already normalized to tuple

#### 2. Update Module Docstring in `pymc_marketing/mmm/events.py`

**File**: `pymc_marketing/mmm/events.py`
**Location**: Lines 14-92 (module docstring)
**Changes**: Add note about single dimension limitation

Add after line 91 (before the closing triple quotes):

```python
Notes
-----
Event effects currently support only single dimensions (e.g., dims=("event",)).
Multi-dimensional event effects where parameters vary across multiple dimensions
(e.g., dims=("event", "country")) are not supported due to dimension broadcasting
limitations. Each EventEffect should use a single dimension to identify events.

If you need event effects that vary by multiple dimensions, consider creating
separate EventEffect instances for each dimension and combining them in your model.
```

**Rationale**:
- Documents limitation at the module level
- Provides workaround guidance
- Sets clear expectations for users

### Success Criteria:

#### Automated Verification:
- [ ] Validation rejects multidimensional dims: `pytest tests/mmm/test_events.py::test_event_effect_rejects_multiple_dims -v`
- [ ] Error message is clear and contains issue link
- [ ] All existing tests pass: `pytest tests/mmm/test_events.py -v`
- [ ] No regressions in multidimensional MMM tests: `pytest tests/mmm/test_multidimensional.py::test_mmm_with_events -v`

#### Manual Verification:
- [ ] Creating `EventEffect(basis=..., effect_size=..., dims=("sales", "country"))` raises ValueError with clear message
- [ ] Error message includes link to issue #1981
- [ ] Error message suggests using single dimension
- [ ] Single-dimension examples in module docstring still execute correctly

---

## Phase 2: Add Test Coverage

### Overview
Add comprehensive tests for the new validation and ensure error messages are helpful.

### Changes Required:

#### 1. Test for Multiple Dimension Rejection in `tests/mmm/test_events.py`

**File**: `tests/mmm/test_events.py`
**Location**: Add after existing validation tests (around line 500)
**Changes**: Add new test function

```python
def test_event_effect_rejects_multiple_dims():
    """Test that EventEffect rejects multiple dimensions with clear error message."""
    gaussian = GaussianBasis(
        priors={
            "sigma": Prior("Gamma", mu=2, sigma=1, dims="sales"),
        }
    )
    effect_size = Prior("Normal", mu=0, sigma=1, dims=("sales", "country"))

    # Should raise ValueError during construction
    with pytest.raises(ValueError) as excinfo:
        EventEffect(
            basis=gaussian,
            effect_size=effect_size,
            dims=("sales", "country"),
        )

    error_msg = str(excinfo.value)

    # Verify error message content
    assert "currently only supports single dimension" in error_msg
    assert "dims=('sales', 'country')" in error_msg
    assert "1981" in error_msg  # Issue link
    assert "Please use a single dimension" in error_msg


def test_event_effect_rejects_multiple_dims_with_string_input():
    """Test that multiple dimensions are rejected even when dims passed as string initially."""
    # This tests the normalization flow: string -> tuple -> validation
    gaussian = GaussianBasis(
        priors={"sigma": Prior("Gamma", mu=2, sigma=1, dims="event")}
    )
    effect_size = Prior("Normal", mu=0, sigma=1, dims="event")

    # Single dimension as string should work (gets normalized to tuple)
    event_effect = EventEffect(
        basis=gaussian,
        effect_size=effect_size,
        dims="event",
    )
    assert event_effect.dims == ("event",)


def test_event_effect_accepts_single_dim_tuple():
    """Test that single dimension as tuple still works."""
    gaussian = GaussianBasis(
        priors={"sigma": Prior("Gamma", mu=2, sigma=1, dims="campaign")}
    )
    effect_size = Prior("Normal", mu=0, sigma=1, dims="campaign")

    # Single dimension as tuple should work
    event_effect = EventEffect(
        basis=gaussian,
        effect_size=effect_size,
        dims=("campaign",),
    )
    assert event_effect.dims == ("campaign",)


def test_event_effect_validation_explains_workaround():
    """Test that error message provides actionable workaround."""
    gaussian = GaussianBasis()
    effect_size = Prior("Normal", mu=0, sigma=1)

    with pytest.raises(ValueError) as excinfo:
        EventEffect(
            basis=gaussian,
            effect_size=effect_size,
            dims=("event", "country", "channel"),
        )

    error_msg = str(excinfo.value).lower()

    # Should mention the architectural limitation
    assert "broadcasting" in error_msg or "dimension" in error_msg

    # Should provide link to tracking issue
    assert "1981" in str(excinfo.value)
```

**Rationale**:
- Tests both tuple and string input paths
- Verifies error message quality and content
- Ensures single dimensions still work
- Tests edge case with three dimensions

#### 2. Update Existing Multidimensional Test in `tests/mmm/test_multidimensional.py`

**File**: `tests/mmm/test_multidimensional.py`
**Location**: Lines 1053-1119 (test_mmm_with_events function)
**Changes**: Verify the test at line 1074 and document its behavior

After reviewing the test at line 1074, we need to understand what `dims=("country", "another_event_type")` actually means in context. Looking at the fixture (lines 993-1007), it creates an EventEffect with default Prior that has no dims specified.

This test will fail after our changes if it's truly using multiple dims. We need to check if this is the actual behavior:

```python
# Option A: If test currently fails (dims validation issue)
# Update line 1073-1076 to use single dimension:
effect=create_event_effect(
    prefix="another_event_type",
    dims=("another_event_type",)  # Changed: single dimension only
),

# Option B: If test currently passes (dims parameter might not be working as expected)
# Add a comment explaining the limitation:
# Note: EventEffect only supports single dimension.
# Previously this test used dims=("country", "another_event_type") but
# this was not actually creating a multidimensional effect due to
# the fixture implementation. Now using single dimension explicitly.
```

**Decision Rule**: We'll test the current behavior first, then update based on findings. If the test passes currently, we document the change. If it fails currently, we fix it as part of this implementation.

### Success Criteria:

#### Automated Verification:
- [ ] New validation tests pass: `pytest tests/mmm/test_events.py::test_event_effect_rejects_multiple_dims -v`
- [ ] All validation tests pass: `pytest tests/mmm/test_events.py -k validation -v`
- [ ] Full test suite passes: `pytest tests/mmm/ -v`
- [ ] Coverage for validation code is 100%: `pytest tests/mmm/test_events.py --cov=pymc_marketing.mmm.events --cov-report=term-missing`

#### Manual Verification:
- [ ] Error messages are readable and actionable
- [ ] Tests fail appropriately when validation is commented out
- [ ] Test names clearly describe what they verify

---

## Phase 3: Documentation Updates

### Overview
Update documentation to clearly communicate the current single-dimension limitation.

### Changes Required:

#### 1. Update EventEffect Class Docstring in `pymc_marketing/mmm/events.py`

**File**: `pymc_marketing/mmm/events.py`
**Location**: Lines 189-243 (EventEffect class)
**Changes**: Add comprehensive docstring with limitations

```python
class EventEffect(BaseModel):
    """Event effect associated with an event model.

    An EventEffect combines a basis function (describing the temporal shape of an
    event's impact) with an effect size (describing the magnitude) to model how
    events influence an outcome over time.

    Parameters
    ----------
    basis : Basis
        The basis transformation that defines the temporal shape of the event effect.
        Common choices: GaussianBasis, HalfGaussianBasis, AsymmetricGaussianBasis.
    effect_size : Prior
        Prior distribution for the effect size (magnitude) of the event.
    dims : str | tuple[str, ...]
        Dimension name(s) for the event effect. Currently only single dimensions
        are supported (e.g., dims="event" or dims=("campaign",)).

    Attributes
    ----------
    basis : Basis
        The basis transformation.
    effect_size : Prior
        The effect size prior.
    dims : tuple[str, ...]
        The dimension tuple (normalized from string input).

    Notes
    -----
    **Single Dimension Limitation**: EventEffect currently supports only single
    dimensions. Attempting to create an EventEffect with multiple dimensions
    (e.g., dims=("event", "country")) will raise a ValueError.

    This limitation exists because the underlying Transformation system only
    tracks parameter dimensions, not data dimensions, leading to broadcasting
    mismatches when basis function parameters have multiple dimensions.

    For more details, see: https://github.com/pymc-labs/pymc-marketing/issues/1981

    Examples
    --------
    Create a simple Gaussian event effect:

    >>> from pymc_marketing.mmm.events import EventEffect, GaussianBasis
    >>> from pymc_extras.prior import Prior
    >>> gaussian = GaussianBasis(
    ...     priors={"sigma": Prior("Gamma", mu=7, sigma=1, dims="event")}
    ... )
    >>> effect = EventEffect(
    ...     basis=gaussian,
    ...     effect_size=Prior("Normal", mu=1, sigma=1, dims="event"),
    ...     dims="event",  # Single dimension
    ... )

    Create event effect with custom dimension name:

    >>> campaign_effect = EventEffect(
    ...     basis=GaussianBasis(),
    ...     effect_size=Prior("Normal", mu=2, sigma=0.5, dims="campaign"),
    ...     dims="campaign",  # Single dimension with custom name
    ... )

    **Invalid usage** (multiple dimensions not supported):

    >>> # This will raise ValueError:
    >>> try:
    ...     EventEffect(
    ...         basis=GaussianBasis(),
    ...         effect_size=Prior("Normal", dims=("event", "country")),
    ...         dims=("event", "country"),  # Multiple dimensions
    ...     )
    ... except ValueError as e:
    ...     print("Error:", str(e)[:50])
    Error: EventEffect currently only supports single dimen
    """

    basis: InstanceOf[Basis]
    effect_size: InstanceOf[Prior]
    dims: str | tuple[str, ...]
    model_config = ConfigDict(extra="forbid")
    # ... rest of implementation
```

**Rationale**:
- Clear parameter documentation
- Prominent "Notes" section explaining limitation
- Both valid and invalid examples
- Link to tracking issue

#### 2. Add Migration Note for Users

**File**: `pymc_marketing/mmm/events.py`
**Location**: Module docstring (around line 91)
**Changes**: Add troubleshooting section

```python
Troubleshooting
---------------
**ValueError: "EventEffect currently only supports single dimension"**

This error occurs when trying to create an EventEffect with multiple dimensions.

Before (not supported)::

    EventEffect(
        basis=GaussianBasis(
            priors={"sigma": Prior("Gamma", mu=2, sigma=1, dims="sales")}
        ),
        effect_size=Prior("Normal", mu=0, sigma=1, dims=("sales", "country")),
        dims=("sales", "country"),  # ❌ Multiple dimensions
    )

After (supported)::

    EventEffect(
        basis=GaussianBasis(
            priors={"sigma": Prior("Gamma", mu=2, sigma=1, dims="sales")}
        ),
        effect_size=Prior("Normal", mu=0, sigma=1, dims="sales"),
        dims=("sales",),  # ✓ Single dimension
    )

If you need effects that vary by multiple dimensions, consider modeling them
separately and combining in your MMM structure, or follow issue #1981 for
updates on multi-dimensional support.
```

**Rationale**:
- Provides before/after comparison
- Shows users what to change in their code
- Links to tracking issue for future updates

### Success Criteria:

#### Automated Verification:
- [ ] Docstring examples are valid Python: `python -m doctest pymc_marketing/mmm/events.py`
- [ ] Documentation builds without warnings: `cd docs && make html`
- [ ] All cross-references resolve correctly

#### Manual Verification:
- [ ] EventEffect docstring appears in built documentation
- [ ] Examples in docstring are clear and accurate
- [ ] Troubleshooting guide is easy to find and follow
- [ ] Links to issue #1981 work correctly

---

## Testing Strategy

### Unit Tests

**File**: `tests/mmm/test_events.py`

1. **Test validation rejects multiple dimensions**:
   - Input: EventEffect with dims=("sales", "country")
   - Expected: ValueError with specific message content
   - Verifies: Core validation logic

2. **Test error message quality**:
   - Input: Various multidimensional dims
   - Expected: Error messages contain key phrases and issue link
   - Verifies: User experience of error messages

3. **Test single dimension still works**:
   - Input: EventEffect with dims=("event",) and dims="event"
   - Expected: No errors, correct behavior
   - Verifies: No regression for valid use cases

4. **Test dimension normalization**:
   - Input: String dims are converted to tuples before validation
   - Expected: Validation sees normalized tuples
   - Verifies: Correct validation order

### Integration Tests

**File**: `tests/mmm/test_multidimensional.py`

1. **Test MMM with events still works**:
   - Input: MMM with single-dimension events
   - Expected: Model builds and fits successfully
   - Verifies: No regression in MMM integration

2. **Test existing multidimensional test behavior**:
   - Input: Test at line 1074 with potentially problematic dims
   - Expected: Update test if needed, document behavior
   - Verifies: No breaking changes to existing tests

### Manual Testing Steps

1. **Create minimal reproduction case from issue**:
   ```python
   from pymc_marketing.mmm.events import EventEffect, GaussianBasis
   from pymc_extras.prior import Prior

   # This should raise clear ValueError
   try:
       EventEffect(
           basis=GaussianBasis(
               priors={"sigma": Prior("Gamma", mu=2, sigma=1, dims="sales")}
           ),
           effect_size=Prior("Normal", mu=0, sigma=1, dims=("sales", "country")),
           dims=("sales", "country"),
       )
   except ValueError as e:
       print(f"Error message:\n{e}")
   ```
   - Verify error message is clear and helpful
   - Verify issue link is present

2. **Verify single-dimension examples still work**:
   ```python
   # Run the example from module docstring
   # Should execute without errors
   ```

3. **Check documentation rendering**:
   - Build docs locally: `cd docs && make html`
   - Navigate to events module documentation
   - Verify docstrings render correctly
   - Verify examples are formatted properly

## Performance Considerations

**No performance impact expected**. The validation happens once at EventEffect construction time, which is negligible compared to model building and sampling. The validation is a simple length check on a tuple.

**Benchmark**: Construction of EventEffect objects should remain under 1ms, which is already the case.

## Migration Notes

### For Users

**Who is affected**: Users attempting to create EventEffect with multiple dimensions in `dims` parameter.

**What changes**: EventEffect will now raise a clear ValueError at construction if multiple dimensions are provided.

**Migration path**:
1. Change `dims=("event", "country")` to `dims=("event",)`
2. If you need effects varying by country, use separate EventEffect instances
3. Follow issue #1981 for updates on when full multi-dimensional support will be available

**Example migration**:

```python
# Before (will now raise error)
EventEffect(
    basis=GaussianBasis(),
    effect_size=Prior("Normal", dims=("sales", "country")),
    dims=("sales", "country"),
)

# After (works)
EventEffect(
    basis=GaussianBasis(),
    effect_size=Prior("Normal", dims="sales"),
    dims="sales",
)
```

### For Contributors

**What changed**:
- EventEffect._validate_dims now enforces single dimension restriction
- New tests in test_events.py verify the validation
- Documentation updated to explain limitation

**Testing**:
- Run full test suite: `pytest tests/mmm/`
- Verify error messages: `pytest tests/mmm/test_events.py::test_event_effect_rejects_multiple_dims -v`
- Check docs: `cd docs && make html`

**Future work**: When implementing full multi-dimensional support:
1. Remove the length check in `_validate_dims`
2. Update tests to verify multi-dimensional cases work
3. Remove "currently only supports" language from documentation
4. Keep issue #1981 open until full support is verified

## References

- **Original Issue**: https://github.com/pymc-labs/pymc-marketing/issues/1981
- **Research Document**: `thoughts/shared/issues/1981/research.md`
- **Related Files**:
  - `pymc_marketing/mmm/events.py:189-243` - EventEffect class
  - `pymc_marketing/mmm/events.py:217-223` - EventEffect.apply method
  - `pymc_marketing/mmm/events.py:262-266` - GaussianBasis.function (where error occurs)
  - `pymc_marketing/mmm/components/base.py:611-650` - Transformation.apply
  - `pymc_marketing/mmm/components/base.py:365-398` - Transformation._create_distributions
  - `tests/mmm/test_events.py` - Event test suite
  - `tests/mmm/test_multidimensional.py:1053-1136` - MMM with events test

## Implementation Checklist

### Phase 1: Validation
- [ ] Update `_validate_dims` method with single-dimension check
- [ ] Add clear error message with issue link
- [ ] Update module docstring with limitation note
- [ ] Run existing tests to verify no regressions

### Phase 2: Testing
- [ ] Add `test_event_effect_rejects_multiple_dims`
- [ ] Add `test_event_effect_rejects_multiple_dims_with_string_input`
- [ ] Add `test_event_effect_accepts_single_dim_tuple`
- [ ] Add `test_event_effect_validation_explains_workaround`
- [ ] Review and update `test_mmm_with_events` if needed
- [ ] Verify all tests pass

### Phase 3: Documentation
- [ ] Add comprehensive docstring to EventEffect class
- [ ] Add troubleshooting section to module docstring
- [ ] Verify docstring examples are valid
- [ ] Build documentation and verify rendering
- [ ] Check all links and cross-references

### Final Verification
- [ ] Run full test suite: `pytest tests/mmm/`
- [ ] Run type checking: `mypy pymc_marketing/mmm/events.py`
- [ ] Build docs: `cd docs && make html`
- [ ] Manually test reproduction case from issue
- [ ] Verify error messages are helpful
- [ ] Review all changes for consistency
