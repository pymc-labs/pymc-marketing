---
date: 2025-11-04T22:26:04+00:00
researcher: Claude
git_commit: 9537b9a08837a3c5dabcdee6244a0cd1c4688ea0
branch: work-issue-1981
repository: pymc-marketing
topic: "Basis functions fail with multidimensional events"
tags: [research, codebase, events, dimensions, broadcasting, issue-1981]
status: complete
last_updated: 2025-11-04
last_updated_by: Claude
issue_number: 1981
---

# Research: Basis functions fail with multidimensional events

**Date**: 2025-11-04T22:26:04+00:00
**Researcher**: Claude
**Git Commit**: 9537b9a08837a3c5dabcdee6244a0cd1c4688ea0
**Branch**: work-issue-1981
**Repository**: pymc-marketing
**Issue**: #1981

## Research Question

When trying to create an EventEffect with multidimensional parameters where the effect_size varies by multiple dimensions (e.g., sales and country):

```python
EventEffect(
    basis=GaussianBasis(
        priors={"sigma": Prior("Gamma", mu=2, sigma=1, dims=("sales"))}
    ),
    effect_size=Prior("Normal", mu=0, sigma=1, dims=("sales", "country")),
    dims=("sales", "country"),
)
```

The model build fails when calling the basis function at `pymc_marketing/mmm/events.py:262-266`. The issue is that `x` has dimensions `('date', 'sales')` while `sigma` has dimensions `('sales', 'country')`, causing a dimension mismatch during broadcasting.

## Summary

The EventEffect system was designed for **single-dimension** events (e.g., `dims=("event",)`) and does not currently support **multidimensional** events where effect parameters vary across multiple dimensions simultaneously. The root cause is that:

1. **Data dimensions are not tracked**: The basis matrix `X` has implicit dimensions (e.g., "date", "sales") that are never communicated to the transformation system
2. **Parameter dimensions only**: The `Transformation.apply()` method only handles parameter dimensions, not data dimensions
3. **Broadcasting mismatch**: When `x` has shape `(date, sales)` and `sigma` has shape `(sales, country)`, PyTensor's broadcasting cannot properly align them because it doesn't know which axes represent "sales"

**Current state**: All existing tests and examples use single-dimension patterns like `dims=("event",)` or `dims=("campaign",)`. No working examples of multidimensional events exist in the codebase.

## Detailed Findings

### 1. EventEffect Architecture

#### EventEffect.apply Method
**Location**: `pymc_marketing/mmm/events.py:217-223`

```python
def apply(self, X: pt.TensorLike, name: str = "event") -> TensorVariable:
    """Apply the event effect to the data."""
    dim_handler = create_dim_handler(("x", *self.dims))
    return self.basis.apply(X, dims=self.dims) * dim_handler(
        self.effect_size.create_variable(f"{name}_effect_size"),
        self.effect_size.dims,
    )
```

**Key observations**:
- Line 219: Creates `dim_handler` with dimensions `("x", *self.dims)`
- Line 220: Calls `self.basis.apply(X, dims=self.dims)` - passes only parameter dims
- The "x" dimension is assumed to be the first axis of the data
- No information about the actual data dimensions is passed to the basis

#### Dimension Validation
**Location**: `pymc_marketing/mmm/events.py:204-215`

```python
@model_validator(mode="after")
def _validate_dims(self):
    if not self.dims:
        raise ValueError("The dims must not be empty.")

    if not set(self.basis.combined_dims).issubset(set(self.dims)):
        raise ValueError("The dims must contain all dimensions of the basis.")

    if not set(self.effect_size.dims).issubset(set(self.dims)):
        raise ValueError("The dims must contain all dimensions of the effect size.")

    return self
```

**Validation ensures**:
- All basis parameter dims must be in `EventEffect.dims`
- All effect_size dims must be in `EventEffect.dims`
- Does NOT validate data dimensions

### 2. Transformation Base Class

#### Transformation.apply Method
**Location**: `pymc_marketing/mmm/components/base.py:611-650`

```python
def apply(
    self,
    x: pt.TensorLike,
    dims: Dims | None = None,
    idx: dict[str, pt.TensorLike] | None = None,
) -> TensorVariable:
    """Call within a model context.

    Parameters
    ----------
    x : pt.TensorLike
        The data to be transformed.
    dims : str, sequence[str], optional
        The dims of the parameters. Defaults to None. Not the dims of the
        data!
    """
    kwargs = self._create_distributions(dims=dims, idx=idx)
    return self.function(x, **kwargs)
```

**Critical limitation documented in docstring**:
- `dims` parameter is "The dims of the parameters. Defaults to None. **Not the dims of the data!**"
- Data `x` is treated as a raw tensor without dimension metadata

#### _create_distributions Method
**Location**: `pymc_marketing/mmm/components/base.py:365-398`

```python
def _create_distributions(
    self,
    dims: Dims | None = None,
    idx: dict[str, pt.TensorLike] | None = None,
) -> dict[str, TensorVariable]:
    if isinstance(dims, str):
        dims = (dims,)

    dims = dims or self.combined_dims
    if idx is not None:
        dims = ("N", *dims)

    dim_handler = create_dim_handler(dims)  # Line 377

    def create_variable(parameter_name: str, variable_name: str) -> TensorVariable:
        dist = self.function_priors[parameter_name]
        if not hasattr(dist, "create_variable"):
            return dist

        var = dist.create_variable(variable_name)

        dist_dims = dist.dims
        if idx is not None and any(dim in idx for dim in dist_dims):
            var = index_variable(var, dist.dims, idx)

            dist_dims = [dim for dim in dist_dims if dim not in idx]
            dist_dims = ("N", *dist_dims)

        return dim_handler(var, dist_dims)

    return {
        parameter_name: create_variable(parameter_name, variable_name)
        for parameter_name, variable_name in self.variable_mapping.items()
    }
```

**Key behavior**:
- Line 377: Creates `dim_handler` with **parameter dimensions only**
- Line 393: Broadcasts parameters to parameter dimensions
- No knowledge of data dimensions

### 3. Basis Function Implementation

#### GaussianBasis.function
**Location**: `pymc_marketing/mmm/events.py:262-266`

```python
def function(self, x: pt.TensorLike, sigma: pt.TensorLike) -> TensorVariable:
    """Gaussian bump function."""
    rv = pm.Normal.dist(mu=0.0, sigma=sigma)
    out = pm.math.exp(pm.logp(rv, x))
    return out
```

**Where the error occurs**:
- Line 265: `pm.logp(rv, x)` attempts to compute log probability
- This requires broadcasting `x` against `sigma` (embedded in `rv`)
- Without dimension metadata, PyTensor uses numpy-style broadcasting rules
- **Fails when shapes don't align properly**

### 4. The Dimension Mismatch Problem

#### Scenario Walkthrough

Given:
```python
EventEffect(
    basis=GaussianBasis(
        priors={"sigma": Prior("Gamma", mu=2, sigma=1, dims=("sales",))}
    ),
    effect_size=Prior("Normal", mu=0, sigma=1, dims=("sales", "country")),
    dims=("sales", "country"),
)
```

With basis matrix `X` of shape `(n_dates, n_sales)`:

**Step 1**: `EventEffect.apply(X)` is called
- Creates `dim_handler` with dims `("x", "sales", "country")`
- Calls `basis.apply(X, dims=("sales", "country"))`

**Step 2**: `Transformation.apply(X, dims=("sales", "country"))`
- Calls `_create_distributions(dims=("sales", "country"))`
- Creates `dim_handler` with dims `("sales", "country")` (no "x"!)

**Step 3**: `_create_distributions` creates sigma
- `sigma` Prior has `dims=("sales",)`
- But `dim_handler` expects output dims `("sales", "country")`
- `dim_handler(sigma_var, ("sales",))` broadcasts sigma to shape `(n_sales, n_countries)`

**Step 4**: `GaussianBasis.function(x, sigma)` is called
- `x` has shape `(n_dates, n_sales)` - implicit dims `("date", "sales")`
- `sigma` has shape `(n_sales, n_countries)` - dims `("sales", "country")`
- `pm.logp(rv, x)` tries to broadcast:
  - `x`: shape `(n_dates, n_sales)`
  - `sigma`: shape `(n_sales, n_countries)`
- **Broadcasting alignment from right**:
  - Position -1: `n_sales` vs `n_countries` - **MISMATCH**
  - Position -2: `n_dates` vs `n_sales` - **MISMATCH**
- **Result**: Broadcasting error or incorrect result

#### Why Single Dimensions Work

With `dims=("event",)` and `X` of shape `(n_dates, n_events)`:

**Step 4 becomes**:
- `x` has shape `(n_dates, n_events)` - implicit dims `("date", "event")`
- `sigma` has shape `(n_events,)` - dims `("event",)`
- `pm.logp(rv, x)` broadcasts:
  - `x`: shape `(n_dates, n_events)`
  - `sigma`: shape `(n_events,)` - broadcasts to `(1, n_events)` then `(n_dates, n_events)`
- **Broadcasting works**: Last axis aligns, first axis broadcasts
- **Result**: Output shape `(n_dates, n_events)` ✓

### 5. Test Coverage Analysis

#### Primary Test File
**Location**: `tests/mmm/test_events.py`

**Key tests**:
- `test_gaussian_basis_multiple_events()` (line 183): Tests multiple events with `dims="event"`
- `test_event_effect_different_dims()` (line 199): Tests `dims="campaign"` (still single dimension)
- `test_event_effect_dim_validation()` (line 486): Tests dimension validation
- `test_half_gaussian_in_event_effect_apply()` (line 540): Tests HalfGaussianBasis with `dims=("event",)`
- `test_asymmetric_gaussian_basis_multiple_events()` (line 687): Multiple events with single dimension

**Pattern observed**: All 36 test functions use single-dimension patterns:
- `dims=("event",)` - most common
- `dims=("campaign",)` - alternative name
- `dims=("holiday",)` - in multidimensional MMM tests

#### Multidimensional MMM Test
**Location**: `tests/mmm/test_multidimensional.py:1053-1136`

```python
def test_mmm_with_events():
    # ...
    effect=create_event_effect(
        prefix="another_event_type",
        dims=("country", "another_event_type")  # Line 1074
    )
```

**Important note**: This test uses `dims=("country", "another_event_type")` but:
- Both dimensions are event-type dimensions, not data dimensions
- The basis matrix would still be 2D: `(dates, events)`
- This is NOT the same as having effect parameters vary by country for each event
- Need to verify if this test actually passes and what it's testing

### 6. Usage Examples

#### Canonical Example (Single Dimension)
**Location**: `pymc_marketing/mmm/events.py:66-91` (module docstring)

```python
gaussian = GaussianBasis(
    priors={
        "sigma": Prior("Gamma", mu=7, sigma=1, dims="event"),
    }
)
effect_size = Prior("Normal", mu=1, sigma=1, dims="event")
effect = EventEffect(basis=gaussian, effect_size=effect_size, dims=("event",))

dates = pd.date_range("2024-12-01", periods=3 * 31, freq="D")
X = create_basis_matrix(df_events, model_dates=dates)

coords = {"date": dates, "event": df_events["event"].to_numpy()}
with pm.Model(coords=coords) as model:
    pm.Deterministic("effect", effect.apply(X), dims=("date", "event"))
```

**Key aspects**:
- Single dimension: `dims=("event",)`
- Basis matrix shape: `(n_dates, n_events)`
- All priors have same dimension: `dims="event"`

#### EventAdditiveEffect Integration
**Location**: `pymc_marketing/mmm/additive_effect.py:522-558`

```python
def create_effect(self, mmm: Model) -> pt.TensorVariable:
    # Create basis matrix
    X = create_basis_matrix(start_ref, end_ref)

    # Apply event effect
    event_effect = self.effect.apply(X, name=self.prefix)  # Shape: (date, event)

    # Sum across events
    total_effect = pm.Deterministic(
        f"{self.prefix}_total_effect",
        event_effect.sum(axis=1),  # Line 553: Collapses event dimension
        dims=self.date_dim_name,
    )

    # Broadcast to MMM dimensions
    dim_handler = create_dim_handler((self.date_dim_name, *mmm.dims))
    return dim_handler(total_effect, self.date_dim_name)
```

**Critical observation**:
- Line 553: **Events are summed** before broadcasting to MMM dims
- This means all events are aggregated to a single time series
- No way to have event effects vary by MMM dimensions (like market/country)

### 7. Dimension Handling Patterns

#### create_dim_handler Usage Pattern
From `pymc_extras.prior`, `create_dim_handler` is used throughout:

```python
dim_handler = create_dim_handler(desired_dims)
result = dim_handler(variable, variable_dims)
```

**Behavior**:
1. Takes tuple of desired output dimensions
2. Returns function that broadcasts variables to those dimensions
3. Adds singleton dimensions where needed
4. Works when variable_dims is a subset of desired_dims

**Example**:
```python
dim_handler = create_dim_handler(("date", "event", "market"))
# variable with dims ("event",) gets broadcast to ("date", "event", "market")
# by adding singleton dims at positions for "date" and "market"
```

### 8. Related Code Patterns

#### Broadcasting Comments in Codebase
- `pymc_marketing/mmm/events.py:327`: "Build boolean mask(s) in x's shape and broadcast to out's shape"
- `pymc_marketing/mmm/additive_effect.py:49`: "Negative-only coefficient per extra dims, broadcast over date"
- `docs/source/notebooks/mmm/mmm_gam_options.ipynb:2685`: "handle broadcasting automatically"

#### Similar Dimension Systems
Other transformations handle dimensions similarly:
- `pymc_marketing/mmm/components/saturation.py`: Uses same `Transformation.apply` base
- `pymc_marketing/mmm/components/adstock.py`: Uses same pattern
- All face same limitation: data dims not tracked

## Code References

### Primary Implementation
- `pymc_marketing/mmm/events.py:189-243` - EventEffect class
- `pymc_marketing/mmm/events.py:217-223` - EventEffect.apply method (where dims mismatch manifests)
- `pymc_marketing/mmm/events.py:262-266` - GaussianBasis.function (where error occurs)

### Base Class System
- `pymc_marketing/mmm/components/base.py:611-650` - Transformation.apply
- `pymc_marketing/mmm/components/base.py:365-398` - Transformation._create_distributions
- `pymc_marketing/mmm/components/base.py:349-363` - Transformation.combined_dims

### Integration Points
- `pymc_marketing/mmm/additive_effect.py:442-567` - EventAdditiveEffect
- `pymc_marketing/mmm/additive_effect.py:522-558` - EventAdditiveEffect.create_effect
- `pymc_marketing/mmm/additive_effect.py:553` - Event aggregation (sums across events)

### Test Coverage
- `tests/mmm/test_events.py:183-197` - Multiple events test
- `tests/mmm/test_events.py:199-216` - Different dims test
- `tests/mmm/test_events.py:486-497` - Dimension validation test
- `tests/mmm/test_multidimensional.py:1053-1136` - MMM with events test

## Architecture Insights

### Design Philosophy
The EventEffect system follows these principles:
1. **Separation of concerns**: Basis functions compute temporal effects, effect_size scales them
2. **Dimension flexibility**: Any dimension name can be used (not just "event")
3. **Parameter independence**: Each event can have different parameters
4. **Integration via summation**: Events are aggregated before MMM broadcasting

### Current Limitations
1. **No data dimension tracking**: The system doesn't track what dimensions the data has
2. **Parameter-only broadcasting**: Transformation.apply only handles parameter dimensions
3. **Single-dimension assumption**: All examples and tests use single dimension
4. **Event aggregation**: Events are summed before MMM integration, preventing multi-dimensional effects

### Why This Wasn't Caught Earlier
1. **All tests use single dimensions**: No test exercises multi-dimensional case
2. **Validation is permissive**: Allows multiple dims in EventEffect.dims
3. **Error manifests deep**: Fails in basis function, not at validation
4. **Documentation doesn't specify**: No explicit statement that multi-dimensional events aren't supported

## Open Questions

1. **Is multidimensional support intended?**
   - Is `dims=("event", "market")` supposed to work?
   - Or is it a future feature request?

2. **What would multidimensional events mean semantically?**
   - Different events for different markets?
   - Events that affect markets differently?
   - Events with market-varying parameters?

3. **How should data dimensions be communicated?**
   - Should `X` have named dimensions?
   - Should `EventEffect.apply` receive data dimension info?
   - Should basis functions accept dimension metadata?

4. **How should broadcasting work?**
   - If `x` is `(date, event)` and `sigma` is `(event, market)`, what should output be?
   - `(date, event, market)` with outer product behavior?
   - Or something else?

5. **Test at test_multidimensional.py:1074**
   - Does the test `dims=("country", "another_event_type")` actually work?
   - What shape is the basis matrix in that case?
   - How are the dimensions being used?

## Potential Solutions

### Option 1: Explicit Data Dimension Tracking
Modify `EventEffect.apply` to accept data dimensions:

```python
def apply(
    self,
    X: pt.TensorLike,
    data_dims: tuple[str, ...],  # NEW
    name: str = "event"
) -> TensorVariable:
    dim_handler = create_dim_handler((*data_dims, *self.dims))
    return self.basis.apply(X, dims=self.dims, data_dims=data_dims) * dim_handler(
        self.effect_size.create_variable(f"{name}_effect_size"),
        self.effect_size.dims,
    )
```

### Option 2: Restrict to Single Dimension
Add validation that only allows single dimension:

```python
@model_validator(mode="after")
def _validate_dims(self):
    if len(self.dims) > 1:
        raise ValueError(
            "EventEffect currently only supports single dimension. "
            f"Got dims={self.dims}"
        )
    # ... rest of validation
```

### Option 3: Separate Data and Parameter Dimensions
Make dims explicitly separate data dims from parameter dims:

```python
class EventEffect(BaseModel):
    basis: InstanceOf[Basis]
    effect_size: InstanceOf[Prior]
    data_dims: tuple[str, ...]  # e.g., ("date", "event")
    param_dims: tuple[str, ...]  # e.g., ("market",)
```

### Option 4: Use Named Tensors / Xarray
Leverage PyMC's named dimension support more explicitly:

```python
def apply(self, X: pt.TensorLike, name: str = "event") -> TensorVariable:
    # X should be a named tensor with dims specified
    data_dims = X.dims  # Get dims from X directly
    # ... rest of implementation
```

## Related Research

No related research documents found in `thoughts/shared/research/` for multidimensional events or dimension broadcasting issues in EventEffect.

## Recommendations

1. **Short-term**: Add validation to prevent multidimensional dims until properly supported
2. **Medium-term**: Investigate the test at `test_multidimensional.py:1074` to understand intent
3. **Long-term**: Design proper multidimensional event support with clear semantics
4. **Documentation**: Clarify that only single dimensions are currently supported

## Next Steps for Issue #1981

1. Verify the error with a minimal reproduction case
2. Determine if multidimensional support is a feature request or bug
3. If bug: Implement proper dimension tracking
4. If feature request: Design multidimensional semantics
5. Add tests for either the restriction or the new feature
6. Update documentation with usage guidelines
