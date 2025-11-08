<!-- ab88a3d6-e3c5-4007-b85d-d1cbf13d175f 2e9fa816-2d0c-4658-90cd-6b9555c48b58 -->
# Product Requirement Document: Time-Invariant Covariates for ShiftedBetaGeoModel

## 1. Executive Summary

Add support for time-invariant customer-level covariates to `ShiftedBetaGeoModel` to allow modeling heterogeneity in customer dropout rates based on customer characteristics (e.g., demographics, acquisition channel, customer tier). This feature will follow the implementation pattern established in `BetaGeoModel` while extending it to work with the cohort-based structure of the sBG model.

## 2. Goals & Objectives

### Primary Goals

- Enable users to incorporate customer-level features (covariates) into dropout rate estimation
- Maintain backward compatibility with existing cohort-only models
- Follow established patterns from `BetaGeoModel` for consistency
- Design modular code that can be moved to `CLVModel` base class in future

### Success Criteria

- Users can specify `dropout_covariate_cols` in model initialization
- Covariates modify customer-specific `alpha` and `beta` parameters within cohorts
- All existing prediction methods work correctly with covariates
- Code is modular and ready for future refactoring to `basic.py`

## 3. User Stories

**Story 1**: As a data scientist, I want to model how customer acquisition channel affects retention rates within each cohort, so I can better target high-value acquisition sources.

**Story 2**: As an analyst, I want to incorporate customer demographics into my retention model while maintaining cohort-level baseline effects, so I can understand both cohort trends and individual variation.

**Story 3**: As a researcher, I want the flexibility to use either hierarchical (`phi`/`kappa`) or direct (`alpha`/`beta`) parameterization with covariates, depending on my modeling needs.

## 4. Functional Requirements

### 4.1 Model Configuration

- Add `dropout_covariate_cols` parameter to `default_model_config` (default: empty list)
- Add `dropout_coefficient` parameter for covariate coefficients (default: `Prior("Normal", mu=0, sigma=1)`)
- Support both hierarchical (`phi`/`kappa`) and direct (`alpha`/`beta`) parameterizations with covariates

### 4.2 Data Requirements

- Covariate columns must be present in input DataFrame when specified in `dropout_covariate_cols`
- Covariates must be numeric and time-invariant
- Validate covariate columns exist during initialization

### 4.3 Mathematical Structure

**Without covariates (current behavior)**:

- Hierarchical: `alpha[cohort] = phi[cohort] * kappa[cohort]`, `beta[cohort] = (1-phi[cohort]) * kappa[cohort]`
- Direct: `alpha[cohort]`, `beta[cohort]` specified directly

**With covariates**:

- Hierarchical:
  - `alpha_scale[cohort] = phi[cohort] * kappa[cohort]`
  - `beta_scale[cohort] = (1-phi[cohort]) * kappa[cohort]`
  - `alpha[customer_id] = alpha_scale[cohort[customer_id]] * exp(-X[customer_id] @ dropout_coefficient_alpha)`
  - `beta[customer_id] = beta_scale[cohort[customer_id]] * exp(-X[customer_id] @ dropout_coefficient_beta)`
- Direct with covariates:
  - `alpha_scale[cohort]` and `beta_scale[cohort]` from priors
  - `alpha[customer_id] = alpha_scale[cohort[customer_id]] * exp(-X[customer_id] @ dropout_coefficient_alpha)`
  - `beta[customer_id] = beta_scale[cohort[customer_id]] * exp(-X[customer_id] @ dropout_coefficient_beta)`

### 4.4 Parameter Dimensions

- `dropout_covariate_cols`: list of column names
- `dropout_coefficient_alpha`: dims=`("dropout_covariate",)`
- `dropout_coefficient_beta`: dims=`("dropout_covariate",)`
- `alpha_scale`/`beta_scale`: dims=`("cohort",)` when covariates present
- `alpha`/`beta`: dims=`("customer_id",)` when covariates present, dims=`("cohort",)` otherwise

## 5. Technical Design

### 5.1 Code Structure Changes

**File**: `pymc_marketing/clv/models/shifted_beta_geo.py`

### 5.2 Key Methods to Modify

#### `__init__` method

- Extract and store `dropout_covariate_cols` from model_config
- Validate covariate columns exist in data
- Update `_validate_cols` call to include covariate columns

#### `default_model_config` property

```python
{
    "phi": Prior("Uniform", lower=0, upper=1, dims="cohort"),
    "kappa": Prior("Pareto", alpha=1, m=1, dims="cohort"),
    "dropout_coefficient": Prior("Normal", mu=0, sigma=1),
    "dropout_covariate_cols": [],
}
```

#### `build_model` method

- Add `dropout_covariate` coordinate when covariates present
- Create `pm.Data` for dropout covariates
- Call `_build_covariate_effects` helper method
- Apply covariate effects to `alpha_scale` and `beta_scale` to create customer-level parameters

**Pseudocode structure**:

```python
if self.dropout_covariate_cols:
    # Add coordinate
    coords["dropout_covariate"] = self.dropout_covariate_cols

    # Create data container
    dropout_data = pm.Data("dropout_data", self.data[self.dropout_covariate_cols],
                          dims=["customer_id", "dropout_covariate"])

    # Get scale parameters (cohort-level)
    if "alpha" in self.model_config and "beta" in self.model_config:
        alpha_scale = self.model_config["alpha"].create_variable("alpha_scale")
        beta_scale = self.model_config["beta"].create_variable("beta_scale")
    else:
        phi = self.model_config["phi"].create_variable("phi")
        kappa = self.model_config["kappa"].create_variable("kappa")
        alpha_scale = pm.Deterministic("alpha_scale", phi * kappa, dims="cohort")
        beta_scale = pm.Deterministic("beta_scale", (1.0 - phi) * kappa, dims="cohort")

    # Get covariate coefficients
    dropout_coefficient_alpha = self.model_config["dropout_coefficient"].create_variable(
        "dropout_coefficient_alpha", dims="dropout_covariate"
    )
    dropout_coefficient_beta = self.model_config["dropout_coefficient"].create_variable(
        "dropout_coefficient_beta", dims="dropout_covariate"
    )

    # Apply covariate effects
    alpha = pm.Deterministic(
        "alpha",
        alpha_scale[self.cohort_idx] * pm.math.exp(
            -pm.math.dot(dropout_data, dropout_coefficient_alpha)
        ),
        dims="customer_id"
    )
    beta = pm.Deterministic(
        "beta",
        beta_scale[self.cohort_idx] * pm.math.exp(
            -pm.math.dot(dropout_data, dropout_coefficient_beta)
        ),
        dims="customer_id"
    )
else:
    # Current behavior (cohort-level only)
```

#### New method: `_build_covariate_effects`

- Encapsulates covariate effect logic for reuse
- Called by `build_model`
- Returns modified parameters with covariate effects applied
- Signature: `_build_covariate_effects(self, base_param, covariate_data, coefficient, cohort_idx) -> pm.Deterministic`

#### `_extract_predictive_variables` method

- Call `_extract_covariate_parameters` helper method
- Reconstruct customer-level `alpha` and `beta` from scale parameters and covariates
- Validate covariate columns present in prediction data

**Pseudocode structure**:

```python
if self.dropout_covariate_cols:
    # Extract scale parameters
    if "alpha_scale" in self.fit_result:
        alpha_scale = self.fit_result["alpha_scale"]
        beta_scale = self.fit_result["beta_scale"]
    else:
        # Compute from phi/kappa if needed
        ...

    # Create covariate xarray
    dropout_xarray = xarray.DataArray(
        pred_data[self.dropout_covariate_cols],
        dims=["customer_id", "dropout_covariate"],
        coords=[customer_id, self.dropout_covariate_cols]
    )

    # Get coefficients
    dropout_coefficient_alpha = self.fit_result["dropout_coefficient_alpha"]
    dropout_coefficient_beta = self.fit_result["dropout_coefficient_beta"]

    # Map cohort indices
    pred_cohort_idx = pd.Categorical(
        pred_data["cohort"], categories=self.cohorts
    ).codes

    # Reconstruct parameters
    alpha = alpha_scale.isel(cohort=pred_cohort_idx) * np.exp(
        -xarray.dot(dropout_xarray, dropout_coefficient_alpha, dim="dropout_covariate")
    )
    beta = beta_scale.isel(cohort=pred_cohort_idx) * np.exp(
        -xarray.dot(dropout_xarray, dropout_coefficient_beta, dim="dropout_covariate")
    )
else:
    # Current behavior
```

#### New method: `_extract_covariate_parameters`

- Encapsulates covariate extraction logic for reuse
- Called by `_extract_predictive_variables`
- Returns reconstructed parameters with covariate effects
- Signature: `_extract_covariate_parameters(self, pred_data, base_param_name, coefficient_name, scale_param) -> xarray.DataArray`

### 5.3 Reference Implementation Locations

**BetaGeoModel patterns to follow**:

- Lines 165-168: Storing covariate columns in `__init__`
- Lines 170-180: Validating covariate columns
- Lines 232-261: Building model with dropout covariates (hierarchical case)
- Lines 267-309: Building model with dropout covariates (direct a/b case)
- Lines 390-412: Extracting covariate parameters for predictions

## 6. API Design

### 6.1 User-Facing API

**Example usage**:

```python
from pymc_marketing.clv import ShiftedBetaGeoModel
from pymc_extras.prior import Prior

model = ShiftedBetaGeoModel(
    data=pd.DataFrame({
        "customer_id": [1, 2, 3, ...],
        "recency": [8, 1, 4, ...],
        "T": [8, 5, 5, ...],
        "cohort": ["2025-02", "2025-04", "2025-04", ...],
        "channel": [1, 0, 1, ...],  # acquisition channel
        "tier": [2, 1, 2, ...],     # customer tier
    }),
    model_config={
        # Use hierarchical parameterization with covariates
        "phi": Prior("Uniform", lower=0, upper=1, dims="cohort"),
        "kappa": Prior("Pareto", alpha=1, m=1, dims="cohort"),
        "dropout_coefficient": Prior("Normal", mu=0, sigma=2),
        "dropout_covariate_cols": ["channel", "tier"],
    }
)

# OR use direct parameterization with covariates
model = ShiftedBetaGeoModel(
    data=...,
    model_config={
        "alpha": Prior("HalfNormal", sigma=10, dims="cohort"),
        "beta": Prior("HalfNormal", sigma=10, dims="cohort"),
        "dropout_coefficient": Prior("Normal", mu=0, sigma=2),
        "dropout_covariate_cols": ["channel", "tier"],
    }
)

model.fit()

# All existing prediction methods work with covariates
# Must include covariate columns in prediction data
pred_data = pd.DataFrame({
    "customer_id": [...],
    "T": [...],
    "cohort": [...],
    "channel": [...],
    "tier": [...],
})

retention = model.expected_retention_rate(data=pred_data, future_t=1)
prob_alive = model.expected_probability_alive(data=pred_data, future_t=0)
lifetime = model.expected_residual_lifetime(data=pred_data, discount_rate=0.05)
```

### 6.2 Backward Compatibility

- All existing code without covariates continues to work unchanged
- `dropout_covariate_cols` defaults to empty list
- No breaking changes to existing API

## 7. Testing Requirements

### 7.1 Unit Tests

- Test model initialization with covariates
- Test validation of covariate columns
- Test both hierarchical and direct parameterization with covariates
- Test parameter dimensions are correct
- Test covariate coefficient creation

### 7.2 Integration Tests

- Test model fitting with covariates (MCMC and MAP)
- Test all prediction methods with covariates
- Test predictions on out-of-sample data with covariates
- Test backward compatibility (no covariates)
- Test error handling for missing covariate columns

### 7.3 Numerical Tests

- Compare results with known solutions (if available)
- Test that covariates have expected directional effects
- Verify hierarchical structure correctly pools information

### 7.4 Edge Cases

- Empty covariate list (backward compatibility)
- Single covariate
- Multiple covariates
- Covariate values of zero
- Missing covariate columns in prediction data

## 8. Documentation Requirements

### 8.1 Docstring Updates

**ShiftedBetaGeoModel class docstring**:

- Add `dropout_coefficient` to model_config parameters
- Add `dropout_covariate_cols` to model_config parameters
- Add example showing covariate usage
- Update references section if needed

**Prediction method docstrings**:

- Update to mention covariate columns required in prediction data
- Add examples with covariates

### 8.2 User Guide

- Add section on using covariates with sBG model
- Explain interpretation of covariate coefficients
- Provide guidance on covariate selection and preprocessing
- Show comparison with and without covariates

### 8.3 API Reference

- Document new model_config parameters
- Update parameter tables

## 9. Implementation Considerations

### 9.1 Modularization Strategy

The two helper methods should be designed for future extraction:

- `_build_covariate_effects`: Pure logic, minimal sBG-specific code
- `_extract_covariate_parameters`: Pure logic, minimal sBG-specific code
- Both methods should be self-contained and well-documented
- Use clear parameter names that work across models

### 9.2 Future Refactoring Path

In a subsequent PR, these methods can be:

1. Moved to `CLVModel` base class in `basic.py`
2. Made generic to handle any parameter (not just alpha/beta)
3. Reused by BetaGeoModel and other CLV models
4. Extended to support additional covariate patterns

### 9.3 Code Quality

- Follow existing code style and patterns
- Add type hints where appropriate
- Include comprehensive docstrings
- Use descriptive variable names
- Add inline comments for complex logic

## 10. Open Questions / Future Work

### Out of Scope for This PR

- Moving methods to `basic.py` (separate PR)
- Adding covariates to `ShiftedBetaGeoModelIndividual`
- Time-varying covariates
- Interaction effects between covariates and cohorts
- Covariate selection/regularization features

### Future Enhancements

- Add support for standardization/normalization of covariates
- Add diagnostics for covariate effects
- Add plotting functions for covariate effects
- Add methods to compute marginal effects
