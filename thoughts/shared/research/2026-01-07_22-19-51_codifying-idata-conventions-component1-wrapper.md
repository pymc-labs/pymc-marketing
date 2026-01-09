---
date: 2026-01-07T22:19:51Z
researcher: Claude Sonnet 4.5
git_commit: c1a8a3828e0e3928572c9da6553c0b7bac9705f1
branch: isofer/plotting-design
repository: pymc-marketing
topic: "Codifying InferenceData Conventions and Component 1 Data Wrapper Design"
tags: [research, codebase, mmm, idata, schema, pydantic, data-wrapper, conventions, component1]
status: complete
last_updated: 2026-01-07
last_updated_by: Claude Sonnet 4.5
related_research: ["2026-01-07_21-08-47_mmm-data-plotting-framework-architecture.md"]
---

# Research: Codifying InferenceData Conventions and Component 1 Data Wrapper Design

**Date**: 2026-01-07T22:19:51Z
**Researcher**: Claude Sonnet 4.5
**Git Commit**: c1a8a3828e0e3928572c9da6553c0b7bac9705f1
**Branch**: isofer/plotting-design
**Repository**: pymc-marketing

---

## Table of Contents

1. [Research Question](#research-question)
2. [Executive Summary](#executive-summary)
   - [Problems with Current Approach](#problems-with-current-approach)
   - [Proposed Solution: Two-Tier Codification](#proposed-solution-two-tier-codification)
   - [Key Benefits](#key-benefits)
3. [Detailed Findings](#detailed-findings)
   - [Finding 1: InferenceData Structure in Multidimensional MMM](#finding-1-inferencedata-structure-in-multidimensional-mmm)
   - [Finding 2: The `_original_scale` Naming Convention](#finding-2-the-_original_scale-naming-convention)
   - [Finding 3: Multiple Access Patterns for Observed Data](#finding-3-multiple-access-patterns-for-observed-data)
   - [Finding 4: Comprehensive Naming Conventions Catalog](#finding-4-comprehensive-naming-conventions-catalog)
   - [Finding 5: Existing Validation Patterns in Codebase](#finding-5-existing-validation-patterns-in-codebase)
   - [Finding 6: Historical Context from Previous Research](#finding-6-historical-context-from-previous-research)
4. [Proposed Solution: Codifying idata with Pydantic](#proposed-solution-codifying-idata-with-pydantic)
   - [Design Philosophy](#design-philosophy)
   - [Tier 1: Pydantic Schemas for idata Structure](#tier-1-pydantic-schemas-for-idata-structure)
   - [Tier 2: Component 1 - Codified Data Wrapper](#tier-2-component-1---codified-data-wrapper)
5. [Implementation Recommendations](#implementation-recommendations)
   - [Phase 1: Foundational Schemas (Immediate)](#phase-1-foundational-schemas-immediate)
   - [Phase 2: Data Wrapper (Next 1-2 releases)](#phase-2-data-wrapper-next-1-2-releases)
   - [Phase 3: Extended Schemas (Future)](#phase-3-extended-schemas-future)
6. [Code References](#code-references)
   - [Key Implementation Files](#key-implementation-files)
   - [Related Files](#related-files)
7. [Architecture Insights](#architecture-insights)
   - [Design Patterns Observed](#design-patterns-observed)
   - [Design Patterns Proposed](#design-patterns-proposed)
   - [Key Trade-offs](#key-trade-offs)
   - [Recommended Approach](#recommended-approach)
8. [Related Research](#related-research)
9. [Recommendations](#recommendations)
    - [Immediate Actions (This Release)](#immediate-actions-this-release)
    - [Short-Term (Next 1-2 Releases)](#short-term-next-1-2-releases)
    - [Long-Term](#long-term)

---

## Research Question

One of the things I noticed working on plotting tools for pymc-marketing is that there are many undocumented assumptions and implicit conventions built into the structure of InferenceData. These are used during post-processing, but never explicitly declared. For instance:

- Scaling predictors or outcomes to original space is done in ad-hoc operations
- It's not clear where to get the value of the observed outcome
- Naming conventions are not documented (like the `_original_scale` suffix)

**Goals:**
1. Document all implicit conventions in idata structure
2. Evaluate whether we should codify these (possibly using Pydantic)
3. Design Component 1 (Codified Data Wrapper) from the MMM Data & Plotting Framework Architecture
4. Focus on multidimensional MMM as the reference implementation

## Executive Summary

**YES, we absolutely should codify InferenceData conventions** for the following reasons:

### Problems with Current Approach

1. **Implicit Conventions Are Widespread**
   - 50+ naming patterns across groups, variables, dimensions, and suffixes
   - Multiple access patterns for the same data (observed values in 4 different locations)
   - No single source of truth for "what should be in idata"

2. **High Cognitive Load for Users**
   - Users must know that `channel_contribution_original_scale` must be manually added before plotting
   - No documentation on which idata groups are required for which operations
   - Error messages often unhelpful ("Variable X not found in posterior")

3. **Fragile Post-Processing**
   - Plotting functions assume specific variable names and structures
   - Breaking changes when variables renamed or moved between groups
   - No validation that idata has required structure before operations

4. **Scaling Logic Scattered**
   - `_original_scale` suffix convention not enforced
   - Scaling factors in `constant_data` but no schema defining their structure
   - Multiple ways to compute original scale (multiply by `target_scale` vs use `_original_scale` variable)

### Proposed Solution: Two-Tier Codification

**Tier 1: Pydantic Schemas for InferenceData Structure**
- Define expected groups, variables, dimensions, and coordinates
- Validate idata structure before operations
- Provide helpful error messages when structure is incomplete

**Tier 2: Component 1 Data Wrapper**
- Wraps idata with high-level API for common operations
- Handles scaling, dimension filtering, time aggregation
- Provides consistent access patterns for observed data, predictions, contributions

### Key Benefits

1. **Type Safety**: Catch errors at validation time, not during plotting
2. **Discoverability**: Users can inspect schema to understand idata structure
3. **Documentation**: Schema serves as living documentation
4. **Extensibility**: New schemas for new model types without breaking existing code
5. **Testing**: Validate test fixtures match expected structure

## Detailed Findings

### Finding 1: InferenceData Structure in Multidimensional MMM

InferenceData is populated across three stages of the modeling workflow, with **6 distinct groups** containing different types of data.

#### Group Lifecycle

**Stage 1: Model Building** ([multidimensional.py:1148-1452](pymc_marketing/mmm/multidimensional.py#L1148-L1452))
- Creates PyMC `pm.Data` variables that will populate `constant_data` during sampling
- Defines model structure, priors, and deterministic relationships

**Stage 2: Model Fitting** ([model_builder.py:928-1027](pymc_marketing/model_builder.py#L928-L1027))
- `pm.sample()` creates `posterior`, `observed_data`, `constant_data`, `sample_stats`
- `compute_deterministics()` adds deterministic variables to posterior
- Manually adds `fit_data` group with training data

**Stage 3: Prediction** ([multidimensional.py:1651-1722](pymc_marketing/mmm/multidimensional.py#L1651-L1722))
- `sample_posterior_predictive()` creates `posterior_predictive` group
- Adds `posterior_predictive_constant_data` with prediction inputs

#### Group 1: `constant_data`

**Purpose**: Stores input data, scaling factors, and time indices used during modeling.

**Variables** (7 total, 4 required + 3 conditional):

| Variable | Dims | Content | When Added |
|----------|------|---------|------------|
| `channel_data` | `("date", *dims, "channel")` | Raw channel spend/impressions | Always |
| `target_data` | `("date", *dims)` | Raw target variable | Always |
| `channel_scale` | Varies by config | Scaling factors for channels | Always |
| `target_scale` | Varies by config | Scaling factor for target | Always |
| `control_data_` | `("date", *dims, "control")` | Control variable data | If `control_columns` provided |
| `time_index` | `("date",)` | Integer time index | If time-varying effects enabled |
| `dayofyear` | `("date",)` | Day of year (1-365) | If yearly seasonality enabled |

**Key Insight**: `constant_data` stores **unscaled** raw data. Scaling factors are separate variables.

**Code Reference**: [multidimensional.py:1222-1277](pymc_marketing/mmm/multidimensional.py#L1222-L1277)

#### Group 2: `observed_data`

**Purpose**: Stores observed target values used in the likelihood function.

**Variables**:
- `y` (or `self.output_var`): Observed target values in **scaled** space
- Dims: `("date", *dims)`

**Key Insight**: Contains **scaled** observations, not raw values. For raw values, use `constant_data.target_data` or `self.y` attribute.

**Code Reference**: [multidimensional.py:1448-1452](pymc_marketing/mmm/multidimensional.py#L1448-L1452)

#### Group 3: `posterior`

**Purpose**: Stores MCMC samples of model parameters and deterministic quantities.

**Variable Categories** (20+ variables total):

1. **Adstock/Saturation Parameters**
   - `adstock_alpha`: Decay rate for adstock transformation
   - `saturation_lambda`, `saturation_beta`: Saturation curve parameters
   - Dims: Typically `("channel",)` or `(*dims, "channel")`

2. **Core Model Parameters**
   - `intercept_baseline` or `intercept_contribution`: Model baseline
   - `gamma_control`: Control variable coefficients (if applicable)
   - `gamma_fourier`: Fourier seasonality coefficients (if applicable)
   - `likelihood_sigma`: Observation noise parameter

3. **Contribution Decomposition** (Deterministic Variables)
   - `channel_contribution`: Media effects per channel over time
   - `baseline_channel_contribution`: Pre-time-varying channel effects (conditional)
   - `control_contribution`: Control variable effects (conditional)
   - `fourier_contribution`: Individual Fourier mode effects (conditional)
   - `yearly_seasonality_contribution`: Total seasonality (conditional)
   - `mu`: Total predicted mean (sum of all contributions)

4. **Time-Varying Effects** (Conditional on config)
   - `intercept_latent_process`: Time-varying intercept multiplier from HSGP
   - `media_temporal_latent_multiplier`: Time-varying media multiplier from HSGP

**Key Insight**: All variables are in **scaled space**. To get original scale, must multiply by `target_scale` or use `_original_scale` variables if manually added.

**Code References**:
- Parameters: [multidimensional.py:1284-1438](pymc_marketing/mmm/multidimensional.py#L1284-L1438)
- Deterministics: [multidimensional.py:1323-1445](pymc_marketing/mmm/multidimensional.py#L1323-L1445)

#### Group 4: `fit_data`

**Purpose**: Stores original training dataset (X + y combined) for model rebuilding and reference.

**Structure**: xarray.Dataset with all input columns

**Variables**:
- All channel columns (e.g., "C1", "C2", ...)
- All control columns (if provided)
- Target variable column (name from `self.target_column`)

**Coordinates**:
- `date`: Date index
- `*dims`: Custom dimensions (e.g., "country", "region")

**Key Insight**: This is the **only group** that stores both features and target together. Useful for model retraining or data inspection.

**Code Reference**: [multidimensional.py:2117-2228](pymc_marketing/mmm/multidimensional.py#L2117-L2228)

#### Group 5: `posterior_predictive`

**Purpose**: Stores posterior predictive samples (predictions with uncertainty).

**Variables**:
- `y` (or `self.output_var`): Predicted target values
- Dims: `("chain", "draw", "date", *dims)`

**Key Insight**: Contains **predictions**, not observations. In scaled space unless `y_original_scale` manually added.

**Code Reference**: [multidimensional.py:1703-1711](pymc_marketing/mmm/multidimensional.py#L1703-L1711)

#### Group 6: `posterior_predictive_constant_data`

**Purpose**: Stores input data used for posterior predictive sampling (may differ from training data).

**Structure**: Same as `constant_data` but with coordinates matching prediction dataset.

**Key Insight**: When predicting on new dates/dimensions, this group contains the new input data, while original `constant_data` retains training inputs.

### Finding 2: The `_original_scale` Naming Convention

The most pervasive implicit convention is the `_original_scale` suffix for variables transformed back to original data scale.

#### How It Works

**Creation Pattern 1: Automatic (Legacy MMM)**
- Automatically created in `build_model()` via `_add_original_scale_deterministics()`
- Multiplies scaled contributions by `target_scale`
- [mmm.py:725-777](pymc_marketing/mmm/mmm.py#L725-L777)

```python
pm.Deterministic(
    name="channel_contribution_original_scale",
    var=channel_contribution * target_scale,
    dims=("date", "channel"),
)
```

**Creation Pattern 2: Manual (Multidimensional MMM)**
- Must be called **after** `build_model()` but **before** `fit()`
- [multidimensional.py:1097-1146](pymc_marketing/mmm/multidimensional.py#L1097-L1146)

```python
mmm.add_original_scale_contribution_variable(
    var=["channel_contribution", "total_media_contribution", "y"]
)
```

#### Variables with `_original_scale` Suffix

| Variable | Original (Scaled) | Original Scale Version |
|----------|------------------|------------------------|
| `channel_contribution` | Scaled contributions | `channel_contribution_original_scale` |
| `total_contribution` | Scaled total | `total_contribution_original_scale` |
| `control_contribution` | Scaled controls | `control_contribution_original_scale` |
| `y` (predictions) | Scaled predictions | `y_original_scale` |
| `yearly_seasonality_contribution` | Scaled seasonality | `yearly_seasonality_contribution_original_scale` |

#### The Problem

1. **Not Automatically Created**: Multidimensional MMM requires manual call to `add_original_scale_contribution_variable()`
2. **No Validation**: Plotting functions fail with cryptic errors if `_original_scale` variables missing
3. **Inconsistent Access Patterns**: Some functions check for `_original_scale` variable, others apply `* target_scale` transformation inline

**Example Error from Plotting** ([plot.py:2838-2847](pymc_marketing/mmm/plot.py#L2838-L2847)):

```python
if "channel_contribution_original_scale" not in self.idata.posterior:
    raise ValueError(
        "Variable 'channel_contribution_original_scale' not found in posterior. "
        "Add it using:\n"
        "    mmm.add_original_scale_contribution_variable(\n"
        "        var=['channel_contribution']\n"
        "    )"
    )
```

### Finding 3: Multiple Access Patterns for Observed Data

Users need observed data for residual plots, evaluation metrics, and comparisons. However, **4 different access patterns** exist:

#### Pattern 1: Instance Attribute `self.y`
- **Location**: Model instance attribute
- **Type**: `pd.Series | np.ndarray`
- **Scale**: Original (unscaled)
- **Usage**: Evaluation, plotting
- **Code**: [base.py:78](pymc_marketing/mmm/base.py#L78)

```python
# Access
observed = mmm.y

# Limitation: Must manually update for out-of-sample predictions
```

#### Pattern 2: `constant_data.target_data`
- **Location**: `idata.constant_data` group
- **Type**: `xr.DataArray`
- **Scale**: Original (unscaled)
- **Usage**: Residual plots
- **Code**: [plot.py:637-645](pymc_marketing/mmm/plot.py#L637-L645)

```python
# Access
observed = mmm.idata.constant_data.target_data

# Limitation: Not always present! Must be explicitly added
```

#### Pattern 3: `fit_data` Group
- **Location**: `idata.fit_data` dataset
- **Type**: `xr.Dataset`
- **Scale**: Original (unscaled)
- **Usage**: Model rebuilding
- **Code**: [model_builder.py:899-901](pymc_marketing/model_builder.py#L899-L901)

```python
# Access
dataset = mmm.idata.fit_data.to_dataframe()
observed = dataset[mmm.output_var]

# Limitation: Only training data, not extended predictions
```

#### Pattern 4: `observed_data` Group
- **Location**: `idata.observed_data` group
- **Type**: `xr.DataArray`
- **Scale**: **Scaled** (not original)
- **Usage**: Internal model checks
- **Code**: [multidimensional.py:1448-1452](pymc_marketing/mmm/multidimensional.py#L1448-L1452)

```python
# Access
observed_scaled = mmm.idata.observed_data.y

# Limitation: In scaled space, requires transformation to original scale
```

#### The Problem

- **No canonical source**: Different functions use different approaches
- **Inconsistent naming**: `y` vs `target_data` vs `self.output_var`
- **Scale confusion**: Only `observed_data.y` is scaled; others are original scale
- **Missing data handling**: Residual plots fail if `constant_data.target_data` missing

### Finding 4: Comprehensive Naming Conventions Catalog

The codebase uses **50+ naming patterns** across variables, dimensions, and suffixes.

#### Standard Suffixes

| Suffix | Meaning | Example Variables |
|--------|---------|------------------|
| `_original_scale` | Unscaled (original data scale) | `channel_contribution_original_scale`, `y_original_scale` |
| `_scaled` | Scaled/normalized for modeling | `channel_data_scaled`, `target_scaled` |
| `_contribution` | Contribution to predicted outcome | `channel_contribution`, `control_contribution` |
| `_baseline` | Pre-time-varying baseline value | `intercept_baseline`, `baseline_channel_contribution` |
| `_latent_process` | Latent Gaussian process | `intercept_latent_process` |
| `_temporal_latent_multiplier` | Time-varying multiplier | `media_temporal_latent_multiplier` |
| `_coef` / `_coefs` | Coefficients | `gamma_control`, `hsgp_coefs` |
| `_effect` / `_effect_size` | Effect magnitude | `holiday_effect`, `event_effect_size` |
| `_penalty` | Regularization penalty | `event_penalty` |
| `_components` | Individual components | `fourier_components` |

#### Standard Prefixes

| Prefix | Meaning | Example Variables |
|--------|---------|------------------|
| `total_` | Aggregated across channels/components | `total_contribution`, `total_media_contribution_original_scale` |
| `baseline_` | See suffix above | `baseline_channel_contribution` |
| `channel_` | Channel-specific | `channel_contribution`, `channel_data`, `channel_scale` |
| `control_` | Control variable-related | `control_contribution`, `control_data` |
| `yearly_` / `fourier_` | Seasonality-related | `yearly_seasonality_contribution`, `fourier_contribution` |
| `saturation_` / `adstock_` | Transformation parameters | `saturation_lambda`, `adstock_alpha` |
| `media_` | Media effects | `media_temporal_latent_multiplier` |

#### Core Dimension Names

| Dimension | Meaning | Values | Always Present |
|-----------|---------|--------|----------------|
| `date` | Time dimension | datetime64 array | ✅ Yes |
| `channel` | Media channels | Channel names | ✅ Yes |
| `control` | Control variables | Control names | ❌ Conditional |
| `fourier_mode` | Fourier basis functions | `["fourier_mode_1", ...]` | ❌ Conditional |
| `chain` | MCMC chain | [0, 1, 2, ...] | ✅ Yes (after sampling) |
| `draw` | MCMC sample | [0, 1, ..., n-1] | ✅ Yes (after sampling) |
| `*dims` | Custom dimensions | User-defined (e.g., "country", "region") | ❌ Optional |

#### Key Patterns

1. **Contribution Variables**: Always end with `_contribution`, optionally prefixed by component name
2. **Scale Transformation**: Original scale variables have `_original_scale` suffix
3. **Time Dimension**: Always named `"date"` (never `"time"` in MMM context)
4. **Hierarchical Dims**: Order is `("date", *custom_dims, "channel")` or `("date", *custom_dims)`
5. **Scalar Variables**: Use `dims=()` for single-value parameters

### Finding 5: Existing Validation Patterns in Codebase

PyMC-Marketing already uses **Pydantic extensively** for configuration and validation. We should leverage existing patterns.

#### Pattern 1: Pydantic BaseModel for Configuration

**Example: Scaling Configuration** ([scaling.py:21-79](pymc_marketing/mmm/scaling.py#L21-L79))

```python
from pydantic import BaseModel, Field, model_validator

class VariableScaling(BaseModel):
    """How to scale a variable."""

    method: Literal["max", "mean"] = Field(..., description="The scaling method.")
    dims: str | tuple[str, ...] = Field(..., description="The dimensions to scale over.")

    @model_validator(mode="after")
    def _validate_dims(self) -> Self:
        if isinstance(self.dims, str):
            self.dims = (self.dims,)
        if "date" in self.dims:
            raise ValueError("dim of 'date' is already assumed in the model.")
        return self

class Scaling(BaseModel):
    """Scaling configuration for the MMM."""
    target: VariableScaling
    channel: VariableScaling
```

**Other Examples**:
- `HSGPKwargs`: HSGP configuration ([hsgp_kwargs.py:23](pymc_marketing/hsgp_kwargs.py#L23))
- `EventEffect`: Event modeling ([events.py:189](pymc_marketing/mmm/events.py#L189))
- `FourierBase`: Seasonality config ([fourier.py:300](pymc_marketing/mmm/fourier.py#L300))
- `BudgetOptimizer`: Budget allocation ([budget_optimizer.py:562](pymc_marketing/mmm/budget_optimizer.py#L562))

#### Pattern 2: Mixin-Based Validation Classes

**Example: Input Data Validation** ([validating.py:47-159](pymc_marketing/mmm/validating.py#L47-L159))

```python
def validation_method_X(method: Callable) -> Callable:
    """Tag a method as a validation method for features."""
    if not hasattr(method, "_tags"):
        method._tags = {}
    method._tags["validation_X"] = True
    return method

class ValidateChannelColumns:
    """Validate the channel columns."""
    channel_columns: list[str] | tuple[str]

    @validation_method_X
    def validate_channel_columns(self, data: pd.DataFrame) -> None:
        if not set(self.channel_columns).issubset(data.columns):
            raise ValueError(f"channel_columns {self.channel_columns} not in data")
```

**Usage via Multiple Inheritance**:
```python
class BaseValidateMMM(
    MMMModelBuilder,
    ValidateTargetColumn,
    ValidateDateColumn,
    ValidateChannelColumns,
):
    """Base class with input validation."""
```

#### Pattern 3: Protocol-Based Type Checking

**Example: MMM Builder Protocol** ([types.py:25-90](pymc_marketing/mmm/types.py#L25-L90))

```python
from typing import Protocol

class MMMBuilder(Protocol):
    """Protocol for objects that can build MMM models."""

    def build_model(self, X: pd.DataFrame, y: pd.Series) -> Any: ...
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Self: ...
    @property
    def idata(self) -> az.InferenceData: ...
```

**Key Insight**: Protocols enable structural typing (duck typing with type checking) without requiring inheritance.

### Finding 6: Historical Context from Previous Research

The [MMM Data & Plotting Framework Architecture](2026-01-07_21-08-47_mmm-data-plotting-framework-architecture.md) research identified three key problems that codifying idata addresses:

#### Problem A: "Trapped Data" Problem
- Users cannot access summary statistics used in plots
- Data wrangling happens internally within plotting functions
- No way to get DataFrames without plotting

**Solution via Codified Wrapper**:
```python
# After codification
wrapper = MMMIDataWrapper(mmm.idata)
df = wrapper.get_contribution_summary()  # Returns DataFrame
```

#### Problem B: Rigidity of InferenceData
- No codified way to perform time aggregation (monthly/yearly rollups)
- No dimension operations (filtering, aggregating channels)

**Solution via Codified Wrapper**:
```python
# After codification
filtered = wrapper.filter_dates("2024-01-01", "2024-12-31")
monthly = filtered.aggregate_time("monthly")
```

#### Problem C: Complex Visualization Requirements
- Need interactive Plotly plots with arbitrary dimensions
- Flexible grouping/filtering

**Solution via Summary Objects**:
```python
# After codification
summary = wrapper.get_summary(original_scale=True)
fig = plot_contributions(summary.contributions)
```

## Proposed Solution: Codifying idata with Pydantic

### Design Philosophy

1. **Layered Validation**: Start simple, add complexity as needed
2. **Backward Compatible**: Existing code continues to work
3. **Opt-In**: Validation is optional but encouraged
4. **Extensible**: Easy to add new schemas for new model types

### Tier 1: Pydantic Schemas for idata Structure

Define schemas that **describe** what should be in idata, then validate actual idata against schemas.

#### Schema 1: Group-Level Schema

```python
from pydantic import BaseModel, Field
from typing import Literal
import arviz as az
import xarray as xr

class InferenceDataGroupSchema(BaseModel):
    """Schema for a single idata group."""

    name: Literal["posterior", "prior", "constant_data", "observed_data",
                  "fit_data", "posterior_predictive", "prior_predictive",
                  "sample_stats", "posterior_predictive_constant_data"]
    required: bool = True
    variables: dict[str, "VariableSchema"] = Field(default_factory=dict)

    def validate_group(self, idata: az.InferenceData) -> list[str]:
        """Validate group exists and contains expected variables.

        Returns list of validation errors (empty if valid).
        """
        errors = []

        # Check group exists
        if self.required and not hasattr(idata, self.name):
            errors.append(f"Required group '{self.name}' not found in InferenceData")
            return errors

        if not hasattr(idata, self.name):
            return errors  # Optional group not present

        group = getattr(idata, self.name)

        # Check variables
        for var_name, var_schema in self.variables.items():
            if var_schema.required and var_name not in group:
                errors.append(
                    f"Required variable '{var_name}' not found in group '{self.name}'"
                )
            elif var_name in group:
                # Validate variable structure
                errors.extend(var_schema.validate_variable(group[var_name]))

        return errors
```

#### Schema 2: Variable-Level Schema

```python
class VariableSchema(BaseModel):
    """Schema for a single variable in idata."""

    name: str
    required: bool = True
    dims: tuple[str, ...] | Literal["*"]  # "*" means any dims acceptable
    dtype: str | None = None  # e.g., "float64", "datetime64[ns]"
    description: str = ""

    def validate_variable(self, data_array: xr.DataArray) -> list[str]:
        """Validate variable structure.

        Returns list of validation errors.
        """
        errors = []

        # Check dimensions
        if self.dims != "*":
            if set(data_array.dims) != set(self.dims):
                errors.append(
                    f"Variable '{self.name}' has dims {data_array.dims}, "
                    f"expected {self.dims}"
                )

        # Check dtype
        if self.dtype and str(data_array.dtype) != self.dtype:
            errors.append(
                f"Variable '{self.name}' has dtype {data_array.dtype}, "
                f"expected {self.dtype}"
            )

        return errors
```

#### Schema 3: Complete idata Schema

```python
class MMMIdataSchema(BaseModel):
    """Complete schema for multidimensional MMM InferenceData."""

    model_type: Literal["mmm"] = "mmm"
    groups: dict[str, InferenceDataGroupSchema]
    custom_dims: tuple[str, ...] = Field(default=(), description="Custom dimensions beyond standard (date, channel)")

    @classmethod
    def from_model_config(cls,
                         custom_dims: tuple[str, ...] = (),
                         has_controls: bool = False,
                         has_seasonality: bool = False,
                         time_varying: bool = False,
                         ) -> "MMMIdataSchema":
        """Create schema based on model configuration."""

        groups = {}

        # Constant data group
        constant_data_vars = {
            "channel_data": VariableSchema(
                name="channel_data",
                dims=("date", *custom_dims, "channel"),
                dtype="float64",
                description="Raw channel spend/impressions data"
            ),
            "target_data": VariableSchema(
                name="target_data",
                dims=("date", *custom_dims),
                dtype="float64",
                description="Raw target variable"
            ),
            "channel_scale": VariableSchema(
                name="channel_scale",
                dims="*",  # Varies by scaling config
                dtype="float64",
                description="Scaling factors for channels"
            ),
            "target_scale": VariableSchema(
                name="target_scale",
                dims="*",  # Varies by scaling config
                dtype="float64",
                description="Scaling factor for target"
            ),
        }

        if has_controls:
            constant_data_vars["control_data_"] = VariableSchema(
                name="control_data_",
                dims=("date", *custom_dims, "control"),
                dtype="float64",
                description="Control variable data"
            )

        if time_varying:
            constant_data_vars["time_index"] = VariableSchema(
                name="time_index",
                dims=("date",),
                dtype="int64",
                description="Integer time index"
            )

        if has_seasonality:
            constant_data_vars["dayofyear"] = VariableSchema(
                name="dayofyear",
                dims=("date",),
                dtype="int64",
                description="Day of year (1-365)"
            )

        groups["constant_data"] = InferenceDataGroupSchema(
            name="constant_data",
            required=True,
            variables=constant_data_vars
        )

        # Posterior group
        posterior_vars = {
            "channel_contribution": VariableSchema(
                name="channel_contribution",
                dims=("date", *custom_dims, "channel"),
                dtype="float64",
                description="Channel contributions (scaled)",
                required=True
            ),
            "mu": VariableSchema(
                name="mu",
                dims=("date", *custom_dims),
                dtype="float64",
                description="Total predicted mean (scaled)",
                required=True
            ),
        }

        if has_controls:
            posterior_vars["control_contribution"] = VariableSchema(
                name="control_contribution",
                dims=("date", *custom_dims, "control"),
                dtype="float64",
                description="Control variable contributions"
            )

        if has_seasonality:
            posterior_vars["yearly_seasonality_contribution"] = VariableSchema(
                name="yearly_seasonality_contribution",
                dims=("date", *custom_dims),
                dtype="float64",
                description="Yearly seasonality contribution"
            )

        groups["posterior"] = InferenceDataGroupSchema(
            name="posterior",
            required=True,
            variables=posterior_vars
        )

        # Fit data group
        groups["fit_data"] = InferenceDataGroupSchema(
            name="fit_data",
            required=True,
            variables={}  # Dynamic based on input columns
        )

        # Posterior predictive group
        groups["posterior_predictive"] = InferenceDataGroupSchema(
            name="posterior_predictive",
            required=False,  # Only after prediction
            variables={
                "y": VariableSchema(
                    name="y",
                    dims=("chain", "draw", "date", *custom_dims),
                    dtype="float64",
                    description="Posterior predictive samples"
                )
            }
        )

        return cls(groups=groups, custom_dims=custom_dims)

    def validate(self, idata: az.InferenceData) -> list[str]:
        """Validate InferenceData against schema.

        Returns list of all validation errors.
        """
        all_errors = []

        for group_name, group_schema in self.groups.items():
            errors = group_schema.validate_group(idata)
            all_errors.extend(errors)

        return all_errors

    def validate_or_raise(self, idata: az.InferenceData) -> None:
        """Validate InferenceData, raising detailed exception if invalid."""
        errors = self.validate(idata)

        if errors:
            error_msg = "InferenceData validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)
```

#### Usage Example

```python
# Create schema based on model configuration
schema = MMMIdataSchema.from_model_config(
    custom_dims=("country",),
    has_controls=True,
    has_seasonality=True,
    time_varying=False
)

# Validate after fitting
try:
    schema.validate_or_raise(mmm.idata)
    print("✓ InferenceData structure is valid")
except ValueError as e:
    print(f"✗ Validation failed:\n{e}")

# Before plotting, check for specific variables
errors = schema.groups["posterior"].validate_group(mmm.idata)
if errors:
    print(f"Warning: Some expected variables missing:\n{errors}")
```

### Tier 2: Component 1 - Codified Data Wrapper

A high-level wrapper around InferenceData that provides:
1. Validated access to data
2. Common transformations (scaling, filtering, aggregation)
3. Consistent API across model types

#### Standalone Utility Functions

These utility functions operate on `az.InferenceData` objects directly and can be used independently of the wrapper class. The wrapper methods call these functions internally.

```python
from typing import Literal
import pandas as pd
import xarray as xr
import arviz as az


# ==================== Filtering Utilities ====================

def filter_idata_by_dates(
    idata: az.InferenceData,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> az.InferenceData:
    """Filter InferenceData to a date range.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object to filter
    start_date : str or pd.Timestamp, optional
        Start date (inclusive)
    end_date : str or pd.Timestamp, optional
        End date (inclusive)

    Returns
    -------
    az.InferenceData
        New InferenceData with filtered groups

    Examples
    --------
    >>> filtered = filter_idata_by_dates(idata, "2024-01-01", "2024-12-31")
    >>> filtered = filter_idata_by_dates(idata, start_date="2024-06-01")
    """
    if start_date is None and end_date is None:
        return idata  # No filtering needed

    date_slice = {"date": slice(start_date, end_date)}

    filtered_groups = {}
    for group_name in idata.groups():
        group = getattr(idata, group_name)
        if "date" in group.dims:
            filtered_groups[group_name] = group.sel(**date_slice)
        else:
            filtered_groups[group_name] = group

    return az.InferenceData(**filtered_groups)


def filter_idata_by_dims(
    idata: az.InferenceData,
    **dim_filters,
) -> az.InferenceData:
    """Filter InferenceData by dimension values.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object to filter
    **dim_filters
        Dimension filters, e.g., country="US", channel=["TV", "Radio"]

    Returns
    -------
    az.InferenceData
        New InferenceData with filtered groups

    Raises
    ------
    ValueError
        If a dimension doesn't exist in any group (likely a typo)

    Examples
    --------
    >>> filtered = filter_idata_by_dims(idata, country="US")
    >>> filtered = filter_idata_by_dims(idata, channel=["TV", "Radio", "Social"])
    """
    if not dim_filters:
        return idata

    # Collect all available dimensions across all groups
    all_dims = set()
    for group_name in idata.groups():
        group = getattr(idata, group_name)
        all_dims.update(group.dims)

    # Check that all requested dimensions exist in at least one group
    for dim in dim_filters:
        if dim not in all_dims:
            raise ValueError(
                f"Dimension '{dim}' not found in any group. "
                f"Available dimensions: {sorted(all_dims)}. "
                f"Check for typos in dimension name."
            )

    # Filter all groups
    filtered_groups = {}
    for group_name in idata.groups():
        group = getattr(idata, group_name)

        # Build selection for this group's dimensions
        group_sel = {}
        for dim, values in dim_filters.items():
            if dim in group.dims:
                group_sel[dim] = values

        if group_sel:
            filtered_groups[group_name] = group.sel(**group_sel)
        else:
            filtered_groups[group_name] = group

    return az.InferenceData(**filtered_groups)


# ==================== Aggregation Utilities ====================

def aggregate_idata_time(
    idata: az.InferenceData,
    period: Literal["weekly", "monthly", "quarterly", "yearly", "all_time"],
    method: Literal["sum", "mean"] = "sum",
) -> az.InferenceData:
    """Aggregate InferenceData over time periods.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object to aggregate
    period : {"weekly", "monthly", "quarterly", "yearly", "all_time"}
        Time period to aggregate to. Use "all_time" to aggregate over
        the entire time dimension (removes the date dimension).
    method : {"sum", "mean"}, default "sum"
        Aggregation method

    Returns
    -------
    az.InferenceData
        New InferenceData with aggregated groups. Note: when using "all_time",
        the date dimension is removed.

    Examples
    --------
    >>> monthly = aggregate_idata_time(idata, "monthly", method="sum")
    >>> total = aggregate_idata_time(idata, "all_time", method="sum")
    """
    # Handle "all_time" aggregation (removes date dimension entirely)
    if period == "all_time":
        aggregated_groups = {}
        for group_name in idata.groups():
            group = getattr(idata, group_name)

            if "date" not in group.dims:
                aggregated_groups[group_name] = group
                continue

            if method == "sum":
                aggregated = group.sum(dim="date")
            elif method == "mean":
                aggregated = group.mean(dim="date")
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

            aggregated_groups[group_name] = aggregated

        return az.InferenceData(**aggregated_groups)

    # Map period to pandas offset for periodic aggregation
    period_map = {
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "Y",
    }
    freq = period_map[period]

    # Aggregate all groups with date dimension
    aggregated_groups = {}
    for group_name in idata.groups():
        group = getattr(idata, group_name)

        if "date" not in group.dims:
            aggregated_groups[group_name] = group
            continue

        if method == "sum":
            aggregated = group.resample(date=freq).sum(dim="date")
        elif method == "mean":
            aggregated = group.resample(date=freq).mean(dim="date")
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        aggregated_groups[group_name] = aggregated

    return az.InferenceData(**aggregated_groups)


def aggregate_idata_dims(
    idata: az.InferenceData,
    dim: str,
    values: list[str],
    new_label: str,
    method: Literal["sum", "mean"] = "sum",
) -> az.InferenceData:
    """Aggregate multiple dimension values into one.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object to aggregate
    dim : str
        Dimension to aggregate (e.g., "channel", "country")
    values : list of str
        Values to aggregate
    new_label : str
        Label for aggregated value
    method : {"sum", "mean"}, default "sum"
        Aggregation method

    Returns
    -------
    az.InferenceData
        New InferenceData with aggregated dimension values

    Examples
    --------
    >>> # Combine social channels into one
    >>> combined = aggregate_idata_dims(
    ...     idata,
    ...     dim="channel",
    ...     values=["Facebook", "Instagram", "TikTok"],
    ...     new_label="Social",
    ...     method="sum"
    ... )
    """
    aggregated_groups = {}
    for group_name in idata.groups():
        group = getattr(idata, group_name)

        if dim not in group.dims:
            aggregated_groups[group_name] = group
            continue

        # Select values to aggregate
        selected = group.sel({dim: values})

        # Aggregate
        if method == "sum":
            aggregated_values = selected.sum(dim=dim)
        elif method == "mean":
            aggregated_values = selected.mean(dim=dim)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # Get other values (not aggregated)
        all_coords = set(group[dim].values)
        other_values = list(all_coords - set(values))

        if other_values:
            other_data = group.sel({dim: other_values})

            # Assign new label to aggregated
            aggregated_values = aggregated_values.expand_dims({dim: [new_label]})

            # Concatenate
            combined = xr.concat([other_data, aggregated_values], dim=dim)
        else:
            # All values aggregated
            combined = aggregated_values.expand_dims({dim: [new_label]})

        aggregated_groups[group_name] = combined

    return az.InferenceData(**aggregated_groups)
```

#### Core Wrapper Class

```python
from typing import Literal, Optional
import pandas as pd
import xarray as xr
import arviz as az

class MMMIDataWrapper:
    """Codified wrapper around InferenceData for MMM models.

    Provides validated access to data and common transformations.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object from fitted MMM model
    schema : MMMIdataSchema, optional
        Schema to validate against. If None, validation skipped.
    validate_on_init : bool, default True
        Whether to validate idata structure on initialization

    Examples
    --------
    >>> wrapper = MMMIDataWrapper(mmm.idata)
    >>>
    >>> # Access observed data (automatically finds correct source)
    >>> observed = wrapper.get_target()
    >>>
    >>> # Get channel spend data
    >>> channel_spend = wrapper.get_channel_spend()
    >>>
    >>> # Get contributions in original scale
    >>> contributions = wrapper.get_contributions(original_scale=True)
    >>>
    >>> # Filter and aggregate
    >>> filtered = wrapper.filter_dates("2024-01-01", "2024-12-31")
    >>> monthly = filtered.aggregate_time("monthly")
    """

    def __init__(
        self,
        idata: az.InferenceData,
        schema: Optional[MMMIdataSchema] = None,
        validate_on_init: bool = True,
    ):
        self.idata = idata
        self.schema = schema

        if validate_on_init and schema is not None:
            schema.validate_or_raise(idata)

        # Cache for expensive operations
        self._cache: dict[str, Any] = {}

    # ==================== Observed Data Access ====================

    def get_target(self, original_scale: bool = True) -> xr.DataArray:
        """Get observed target data with consistent access pattern.

        Parameters
        ----------
        original_scale : bool, default True
            Whether to return data in original scale

        Returns
        -------
        xr.DataArray
            Observed target values

        Raises
        ------
        ValueError
            If target data not found in constant_data
        """
        # TODO: Verify with developers that constant_data.target_data is the
        # canonical source for target data. Other potential sources exist
        # (fit_data, observed_data.y) but we need to confirm the best approach.
        if not (hasattr(self.idata, "constant_data") and
                "target_data" in self.idata.constant_data):
            raise ValueError(
                "Target data not found in constant_data. "
                "Expected 'target_data' variable in idata.constant_data."
            )

        data = self.idata.constant_data.target_data
        if original_scale:
            return data
        else:
            # Scale down using target_scale
            target_scale = self.idata.constant_data.target_scale
            return data / target_scale

    def get_channel_spend(self) -> xr.DataArray:
        """Get channel spend data with consistent access pattern.

        Returns raw channel spend data (not MCMC samples).

        Returns
        -------
        xr.DataArray
            Channel spend values with dims (date, channel)

        Raises
        ------
        ValueError
            If channel_data not found in constant_data
        """
        if not (hasattr(self.idata, "constant_data") and
                "channel_data" in self.idata.constant_data):
            raise ValueError(
                "Channel data not found in constant_data. "
                "Expected 'channel_data' variable in idata.constant_data."
            )

        return self.idata.constant_data.channel_data

    # ==================== Contribution Access ====================

    def get_contributions(
        self,
        original_scale: bool = True,
        include_baseline: bool = True,
        include_controls: bool = True,
        include_seasonality: bool = True,
    ) -> xr.Dataset:
        """Get all contribution variables in a single dataset.

        Parameters
        ----------
        original_scale : bool, default True
            Whether to return contributions in original scale
        include_baseline : bool, default True
            Include intercept/baseline contribution
        include_controls : bool, default True
            Include control variable contributions (if present)
        include_seasonality : bool, default True
            Include seasonality contributions (if present)

        Returns
        -------
        xr.Dataset
            Dataset with all contribution variables
        """
        contributions = {}

        # Channel contributions
        if original_scale:
            if "channel_contribution_original_scale" in self.idata.posterior:
                contributions["channel"] = self.idata.posterior.channel_contribution_original_scale
            else:
                # Compute on-the-fly
                channel_contrib = self.idata.posterior.channel_contribution
                target_scale = self.idata.constant_data.target_scale
                contributions["channel"] = channel_contrib * target_scale
        else:
            contributions["channel"] = self.idata.posterior.channel_contribution

        # Baseline/intercept
        if include_baseline:
            for var in ["intercept_contribution", "intercept_baseline"]:
                if var in self.idata.posterior:
                    baseline = self.idata.posterior[var]
                    if original_scale:
                        target_scale = self.idata.constant_data.target_scale
                        contributions["baseline"] = baseline * target_scale
                    else:
                        contributions["baseline"] = baseline
                    break

        # Control variables
        if include_controls and "control_contribution" in self.idata.posterior:
            control = self.idata.posterior.control_contribution
            if original_scale:
                if "control_contribution_original_scale" in self.idata.posterior:
                    contributions["control"] = self.idata.posterior.control_contribution_original_scale
                else:
                    target_scale = self.idata.constant_data.target_scale
                    contributions["control"] = control * target_scale
            else:
                contributions["control"] = control

        # Seasonality
        if include_seasonality and "yearly_seasonality_contribution" in self.idata.posterior:
            seasonality = self.idata.posterior.yearly_seasonality_contribution
            if original_scale:
                if "yearly_seasonality_contribution_original_scale" in self.idata.posterior:
                    contributions["seasonality"] = self.idata.posterior.yearly_seasonality_contribution_original_scale
                else:
                    target_scale = self.idata.constant_data.target_scale
                    contributions["seasonality"] = seasonality * target_scale
            else:
                contributions["seasonality"] = seasonality

        return xr.Dataset(contributions)

    # ==================== Scaling Operations ====================

    def to_original_scale(self, var: str | xr.DataArray) -> xr.DataArray:
        """Transform variable from scaled to original scale.

        Parameters
        ----------
        var : str or xr.DataArray
            Variable name in posterior or DataArray in scaled space

        Returns
        -------
        xr.DataArray
            Variable in original scale
        """
        if isinstance(var, str):
            if f"{var}_original_scale" in self.idata.posterior:
                # Already exists
                return self.idata.posterior[f"{var}_original_scale"]
            elif var in self.idata.posterior:
                # Compute on-the-fly
                data = self.idata.posterior[var]
            else:
                raise ValueError(f"Variable '{var}' not found in posterior")
        else:
            data = var

        target_scale = self.idata.constant_data.target_scale
        return data * target_scale

    def to_scaled(self, var: str | xr.DataArray) -> xr.DataArray:
        """Transform variable from original to scaled space.

        Parameters
        ----------
        var : str or xr.DataArray
            Variable name ending with '_original_scale' or DataArray in original space

        Returns
        -------
        xr.DataArray
            Variable in scaled space
        """
        if isinstance(var, str):
            if var.endswith("_original_scale"):
                # Get base variable name
                base_name = var.replace("_original_scale", "")
                if base_name in self.idata.posterior:
                    return self.idata.posterior[base_name]

            if var in self.idata.posterior:
                data = self.idata.posterior[var]
            else:
                raise ValueError(f"Variable '{var}' not found in posterior")
        else:
            data = var

        target_scale = self.idata.constant_data.target_scale
        return data / target_scale

    # ==================== Filtering Operations ====================

    def filter_dates(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> "MMMIDataWrapper":
        """Filter to date range, returning new wrapper.

        Delegates to standalone `filter_idata_by_dates` utility function.

        Parameters
        ----------
        start_date : str or pd.Timestamp, optional
            Start date (inclusive)
        end_date : str or pd.Timestamp, optional
            End date (inclusive)

        Returns
        -------
        MMMIDataWrapper
            New wrapper with filtered idata
        """
        if start_date is None and end_date is None:
            return self

        filtered_idata = filter_idata_by_dates(self.idata, start_date, end_date)

        return MMMIDataWrapper(
            filtered_idata,
            schema=self.schema,
            validate_on_init=False
        )

    def filter_dims(self, **dim_filters) -> "MMMIDataWrapper":
        """Filter by custom dimensions, returning new wrapper.

        Delegates to standalone `filter_idata_by_dims` utility function.

        Parameters
        ----------
        **dim_filters
            Dimension filters, e.g., country="US", channel=["TV", "Radio"]

        Returns
        -------
        MMMIDataWrapper
            New wrapper with filtered idata

        Raises
        ------
        ValueError
            If a dimension doesn't exist in any group (likely a typo)

        Examples
        --------
        >>> # Filter to single country
        >>> wrapper.filter_dims(country="US")
        >>>
        >>> # Filter to multiple channels
        >>> wrapper.filter_dims(channel=["TV", "Radio", "Social"])
        """
        if not dim_filters:
            return self

        filtered_idata = filter_idata_by_dims(self.idata, **dim_filters)

        return MMMIDataWrapper(
            filtered_idata,
            schema=self.schema,
            validate_on_init=False
        )

    # ==================== Aggregation Operations ====================

    def aggregate_time(
        self,
        period: Literal["weekly", "monthly", "quarterly", "yearly", "all_time"],
        method: Literal["sum", "mean"] = "sum",
    ) -> "MMMIDataWrapper":
        """Aggregate data over time periods.

        Delegates to standalone `aggregate_idata_time` utility function.

        Parameters
        ----------
        period : {"weekly", "monthly", "quarterly", "yearly", "all_time"}
            Time period to aggregate to. Use "all_time" to aggregate over
            the entire time dimension (removes the date dimension).
        method : {"sum", "mean"}, default "sum"
            Aggregation method

        Returns
        -------
        MMMIDataWrapper
            New wrapper with aggregated idata. Note: when using "all_time",
            the returned wrapper has schema=None since the date dimension
            is removed and the original schema no longer applies.

        Examples
        --------
        >>> # Aggregate daily data to monthly
        >>> monthly_wrapper = wrapper.aggregate_time("monthly", method="sum")
        >>>
        >>> # Aggregate over entire time dimension (for total ROAS, etc.)
        >>> total_wrapper = wrapper.aggregate_time("all_time", method="sum")
        """
        aggregated_idata = aggregate_idata_time(self.idata, period, method)

        # For "all_time", schema no longer applies (date dimension removed)
        schema = None if period == "all_time" else self.schema

        return MMMIDataWrapper(
            aggregated_idata,
            schema=schema,
            validate_on_init=False
        )

    def aggregate_dims(
        self,
        dim: str,
        values: list[str],
        new_label: str,
        method: Literal["sum", "mean"] = "sum",
    ) -> "MMMIDataWrapper":
        """Aggregate multiple dimension values into one.

        Delegates to standalone `aggregate_idata_dims` utility function.

        Parameters
        ----------
        dim : str
            Dimension to aggregate (e.g., "channel", "country")
        values : list of str
            Values to aggregate
        new_label : str
            Label for aggregated value
        method : {"sum", "mean"}, default "sum"
            Aggregation method

        Returns
        -------
        MMMIDataWrapper
            New wrapper with aggregated idata

        Examples
        --------
        >>> # Combine social channels
        >>> wrapper.aggregate_dims(
        ...     dim="channel",
        ...     values=["Facebook", "Instagram", "TikTok"],
        ...     new_label="Social",
        ...     method="sum"
        ... )
        """
        aggregated_idata = aggregate_idata_dims(self.idata, dim, values, new_label, method)

        return MMMIDataWrapper(
            aggregated_idata,
            schema=self.schema,
            validate_on_init=False
        )

    # ==================== Summary Statistics ====================

    def compute_posterior_summary(
        self,
        var: str,
        hdi_prob: float = 0.94,
        original_scale: bool = True,
    ) -> pd.DataFrame:
        """Compute summary statistics for a variable.

        Parameters
        ----------
        var : str
            Variable name in posterior
        hdi_prob : float, default 0.94
            Probability for HDI computation
        original_scale : bool, default True
            Whether to compute in original scale

        Returns
        -------
        pd.DataFrame
            Summary statistics (mean, median, std, HDI)

        Examples
        --------
        >>> # Summary of channel contributions
        >>> wrapper.compute_posterior_summary("channel_contribution", original_scale=True)
        """
        # Get variable
        if original_scale:
            data = self.to_original_scale(var)
        else:
            if var in self.idata.posterior:
                data = self.idata.posterior[var]
            else:
                raise ValueError(f"Variable '{var}' not found in posterior")

        # Use arviz for summary
        summary = az.summary(
            data,
            hdi_prob=hdi_prob,
            kind="stats"
        )

        return summary

    # ==================== Validation ====================

    def validate(self) -> list[str]:
        """Validate idata structure against schema.

        Returns
        -------
        list of str
            Validation errors (empty if valid)
        """
        if self.schema is None:
            raise ValueError("No schema provided for validation")

        return self.schema.validate(self.idata)

    def is_valid(self) -> bool:
        """Check if idata structure is valid."""
        if self.schema is None:
            return True  # No schema, assume valid

        errors = self.validate()
        return len(errors) == 0

    # ==================== Convenience Properties ====================

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Get date coordinate."""
        if hasattr(self.idata, "constant_data"):
            return pd.DatetimeIndex(self.idata.constant_data.coords["date"].values)
        elif hasattr(self.idata, "posterior"):
            return pd.DatetimeIndex(self.idata.posterior.coords["date"].values)
        else:
            raise ValueError("Could not find date coordinate in InferenceData")

    @property
    def channels(self) -> list[str]:
        """Get channel coordinate."""
        if hasattr(self.idata, "constant_data"):
            return self.idata.constant_data.coords["channel"].values.tolist()
        elif hasattr(self.idata, "posterior"):
            return self.idata.posterior.coords["channel"].values.tolist()
        else:
            raise ValueError("Could not find channel coordinate in InferenceData")

    @property
    def custom_dims(self) -> list[str]:
        """Get all custom dimension names."""
        standard_dims = {"date", "channel", "control", "fourier_mode", "chain", "draw"}

        if hasattr(self.idata, "constant_data"):
            return [
                dim for dim in self.idata.constant_data.dims
                if dim not in standard_dims
            ]

        return []
```

#### Integration with Existing MMM Class

Add convenience method to MMM class:

```python
# In multidimensional.py or mmm.py

class MMM(BaseValidateMMM):
    # ... existing methods ...

    @property
    def data(self) -> MMMIDataWrapper:
        """Get codified data wrapper for this model's InferenceData.

        Returns
        -------
        MMMIDataWrapper
            Wrapper providing validated access and transformations

        Examples
        --------
        >>> # Access observed data
        >>> observed = mmm.data.get_target()
        >>>
        >>> # Get channel spend data
        >>> channel_spend = mmm.data.get_channel_spend()
        >>>
        >>> # Get contributions in original scale
        >>> contributions = mmm.data.get_contributions(original_scale=True)
        >>>
        >>> # Filter and aggregate
        >>> monthly = mmm.data.filter_dates("2024-01-01", "2024-12-31").aggregate_time("monthly")
        """
        if not hasattr(self, "_data_wrapper") or self._data_wrapper is None:
            # Create schema from model configuration
            schema = MMMIdataSchema.from_model_config(
                custom_dims=self.dims if hasattr(self, "dims") else (),
                has_controls=self.control_columns is not None,
                has_seasonality=self.yearly_seasonality is not None,
                time_varying=(
                    getattr(self, "time_varying_intercept", False) or
                    getattr(self, "time_varying_media", False)
                ),
            )

            self._data_wrapper = MMMIDataWrapper(
                self.idata,
                schema=schema,
                validate_on_init=False  # Don't validate on every access
            )

        return self._data_wrapper
```

## Implementation Recommendations

### Phase 1: Foundational Schemas (Immediate)

**Priority**: High
**Effort**: Medium
**Impact**: High

1. **Create `pymc_marketing/mmm/idata_schema.py`**
   - Implement `VariableSchema`, `InferenceDataGroupSchema`, `MMMIdataSchema`
   - Add comprehensive docstrings and examples
   - Include type hints for all methods

2. **Add schema creation to MMM classes**
   - Add class method `get_idata_schema()` to `MultidimensionalMMM`
   - Returns appropriate schema based on model configuration

3. **Add validation helpers**
   - `validate_idata_for_plotting()` - Check variables needed for plotting
   - `validate_idata_for_contributions()` - Check contribution variables present
   - `validate_idata_for_prediction()` - Check prediction groups present

4. **Update error messages**
   - Replace generic "Variable X not found" with helpful messages
   - Suggest specific fixes (e.g., "Call `add_original_scale_contribution_variable()`")
   - Show what's expected vs what's present

### Phase 2: Data Wrapper (Next 1-2 releases)

**Priority**: High
**Effort**: High
**Impact**: Very High

1. **Implement `MMMIDataWrapper` class**
   - Start with core methods: `get_target()`, `get_channel_spend()`, `get_contributions()`, scaling operations
   - Add filtering and aggregation methods
   - Comprehensive test coverage (unit + integration tests)

2. **Add `.data` property to MMM classes**
   - Returns `MMMIDataWrapper` instance
   - Lazy initialization with caching

3. **Update plotting functions to use wrapper**
   - Refactor plotting to accept wrapper or idata
   - Deprecate direct idata access patterns
   - Provide migration guide

4. **Documentation**
   - User guide on using data wrapper
   - Migration guide from old to new patterns
   - API reference for all methods

### Phase 3: Extended Schemas (Future)

**Priority**: Low
**Effort**: Medium
**Impact**: Medium

1. **Create schemas for other model types**
   - `LegacyMMMIdataSchema` - For non-multidimensional MMM
   - `CLVIdataSchema` - For CLV models
   - `TimeVaryingMMMIdataSchema` - For time-varying parameter models

2. **Schema registry**
   - Auto-detect appropriate schema based on model type
   - Allow users to register custom schemas


## Code References

### Key Implementation Files

**InferenceData Population**:
- [multidimensional.py:1148-1452](pymc_marketing/mmm/multidimensional.py#L1148-L1452) - `build_model()` creates pm.Data variables
- [multidimensional.py:1222-1277](pymc_marketing/mmm/multidimensional.py#L1222-L1277) - constant_data variables
- [multidimensional.py:1323-1445](pymc_marketing/mmm/multidimensional.py#L1323-L1445) - posterior deterministics
- [model_builder.py:928-1027](pymc_marketing/model_builder.py#L928-L1027) - `fit()` samples posterior, adds fit_data
- [multidimensional.py:1651-1722](pymc_marketing/mmm/multidimensional.py#L1651-L1722) - `sample_posterior_predictive()`

**Scaling Conventions**:
- [mmm.py:725-777](pymc_marketing/mmm/mmm.py#L725-L777) - `_add_original_scale_deterministics()` (legacy MMM)
- [multidimensional.py:1097-1146](pymc_marketing/mmm/multidimensional.py#L1097-L1146) - `add_original_scale_contribution_variable()` (multidimensional)
- [mmm.py:565-572](pymc_marketing/mmm/mmm.py#L565-L572) - Scaling operations (division)

**Observed Data Access**:
- [base.py:78](pymc_marketing/mmm/base.py#L78) - `self.y` attribute
- [plot.py:637-645](pymc_marketing/mmm/plot.py#L637-L645) - `constant_data.target_data` access
- [model_builder.py:899-901](pymc_marketing/model_builder.py#L899-L901) - `fit_data` access
- [multidimensional.py:1448-1452](pymc_marketing/mmm/multidimensional.py#L1448-L1452) - `observed_data` creation

**Naming Conventions**:
- [mmm.py:544-903](pymc_marketing/mmm/mmm.py#L544-L903) - Variable naming patterns
- [plot.py:2838-2875](pymc_marketing/mmm/plot.py#L2838-L2875) - Contribution variable access
- [multidimensional.py:1284-1445](pymc_marketing/mmm/multidimensional.py#L1284-L1445) - Multidimensional variable names

**Existing Validation**:
- [scaling.py:21-79](pymc_marketing/mmm/scaling.py#L21-L79) - Pydantic configuration example
- [validating.py:47-159](pymc_marketing/mmm/validating.py#L47-L159) - Mixin validation pattern
- [types.py:25-90](pymc_marketing/mmm/types.py#L25-L90) - Protocol definitions
- [hsgp_kwargs.py:23-84](pymc_marketing/hsgp_kwargs.py#L23-L84) - Field constraints example

### Related Files

**Model Building**:
- [multidimensional.py:935-990](pymc_marketing/mmm/multidimensional.py#L935-L990) - Data preprocessing (DataFrame → xarray)
- [multidimensional.py:1033-1049](pymc_marketing/mmm/multidimensional.py#L1033-L1049) - Scale computation
- [multidimensional.py:991-1031](pymc_marketing/mmm/multidimensional.py#L991-L1031) - Forward pass (adstock + saturation)

**Plotting**:
- [plot.py:208-3453](pymc_marketing/mmm/plot.py#L208-L3453) - MMMPlotSuite (relies on implicit conventions)
- [plot.py:1270-1295](pymc_marketing/mmm/plot.py#L1270-L1295) - Dimension validation example

**Budget Optimization**:
- [budget_optimizer.py:289-510](pymc_marketing/mmm/budget_optimizer.py#L289-L510) - Requires specific idata structure
- [budget_optimizer.py:111-183](pymc_marketing/mmm/budget_optimizer.py#L111-L183) - Documents required variables

## Architecture Insights

### Design Patterns Observed

1. **Implicit Contracts**: Most code relies on undocumented assumptions about idata structure
2. **Ad-hoc Validation**: Validation scattered across codebase, no central validation
3. **Multiple Access Paths**: Same data accessible via multiple routes, causing confusion
4. **Naming Conventions**: Strong conventions but not enforced or validated

### Design Patterns Proposed

1. **Explicit Schemas**: Pydantic models document and validate idata structure
2. **Centralized Validation**: Single source of truth for validation logic
3. **Wrapper Abstraction**: Hide complexity, provide high-level API
4. **Convention over Configuration**: Smart defaults, but allow customization
5. **Standalone Utility Functions**: Core operations (filtering, aggregation) are implemented as standalone functions that operate on `az.InferenceData`. The wrapper class methods delegate to these utilities. This provides:
   - **Reusability**: Functions can be used independently without instantiating a wrapper
   - **Testability**: Easier to unit test individual functions in isolation
   - **Composability**: Functions can be chained or combined in custom workflows
   - **Separation of Concerns**: Business logic is separate from wrapper state management

### Key Trade-offs

**Schema Validation**:
- ✅ Pro: Early error detection, clear contracts, better error messages
- ❌ Con: Performance overhead, complexity, potential false positives

**Data Wrapper**:
- ✅ Pro: Unified API, easier testing, better UX
- ❌ Con: Additional layer of abstraction, learning curve, maintenance burden

**Pydantic vs Custom Validation**:
- ✅ Pro: Familiar to Python users, well-tested, good error messages
- ❌ Con: Dependency, may be overkill for simple cases

### Recommended Approach

**Start Simple, Add Complexity As Needed**:
1. Begin with basic schema validation (groups and key variables)
2. Add wrapper with core functionality (observed data, contributions, scaling)
3. Incrementally add filtering, aggregation, summary methods

**Measure Impact**:
- Track validation overhead (should be <1% of total runtime)
- Monitor error message quality (user feedback)
- Assess developer experience (survey, interviews)

## Related Research

This research builds upon and complements:
- [2026-01-07_21-08-47_mmm-data-plotting-framework-architecture.md](2026-01-07_21-08-47_mmm-data-plotting-framework-architecture.md) - Identified need for codified data wrapper (Component 1)
- Previous work on MMM plotting suite migration

### Resolved: `all_time` Aggregation and Schema Validation

**Question**: How should `aggregate_time("all_time")` work when it removes the `date` dimension, which would fail schema validation?

**Decision**: When aggregating to `all_time`, the returned wrapper sets `schema=None` because the original schema no longer applies (date dimension is removed).

**Rationale**:
- Users explicitly requesting `all_time` aggregation understand the dimension is removed
- Component 2 (Summary Object) needs this for total ROAS and other aggregate metrics
- Setting `schema=None` prevents false validation errors
- The wrapper enters a "transformed state" where the original schema doesn't apply
- Plotting functions (Component 3) handle both timeseries and aggregated data
