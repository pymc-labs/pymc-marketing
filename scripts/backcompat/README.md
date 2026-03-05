# Backwards Compatibility Testing

This directory contains scripts for testing backwards compatibility of PyMC Marketing models by comparing model structure and posterior shapes between the `main` branch and the current branch.

## Overview

The backwards compatibility system:
1. **Captures** a baseline model from the `main` branch
2. **Captures** a candidate model from the current branch
3. **Compares** both models to ensure compatibility

Models are fitted using mock sampling (`pymc.testing.mock_sample`) for speed, which replaces `pm.sample` with prior predictive sampling.

## Quick Start

### Test All Passing Models
```bash
./scripts/backcompat/local_backcompat.sh
```

### Test a Single Model (Fast Iteration)
```bash
./scripts/backcompat/local_backcompat.sh gamma_gamma
# OR use the convenience wrapper
./scripts/backcompat/test_single_model.sh gamma_gamma
```

### Test Multiple Specific Models
```bash
./scripts/backcompat/local_backcompat.sh pareto_nbd shifted_beta_geo
```

### Get Help
```bash
./scripts/backcompat/local_backcompat.sh --help
```

### Manually Test a Specific Model
```bash
# Capture baseline from main branch
python -m scripts.backcompat.capture basic_mmm /tmp/backcompat/main/basic_mmm

# Capture candidate from current branch
python -m scripts.backcompat.capture basic_mmm /tmp/backcompat/head/basic_mmm

# Compare
python -m scripts.backcompat.compare /tmp/backcompat/main/basic_mmm/manifest.json
```

## Supported Models

### ✅ Active Models (Tested in CI/CD)

The following models pass all backwards compatibility checks:

| Model | Description | Status |
|-------|-------------|--------|
| `basic_mmm` | Basic Marketing Mix Model | ✅ Passing |
| `beta_geo` | Beta Geometric Model | ✅ Passing |
| `beta_geo_beta_binom` | Beta Geometric Beta Binomial Model | ✅ Passing |
| `gamma_gamma` | Gamma-Gamma Model | ✅ Passing |
| `modified_beta_geo` | Modified Beta Geometric Model | ✅ Passing |
| `pareto_nbd` | Pareto NBD Model | ✅ Passing (fixed 2026-02-09) |
| `shifted_beta_geo` | Shifted Beta Geometric Model | ✅ Passing (fixed 2026-02-09) |

### ❌ Excluded Models (Known Issues)

These models have **pre-existing issues** and are excluded from CI/CD:

| Model | Issue | Root Cause |
|-------|-------|------------|
| `mixed_logit` | Serialization not supported | `MixedLogit.__init__()` requires 4 positional arguments (`choice_df`, `utility_equations`, `depvar`, `covariates`) which are not saved/restored by the base `ModelBuilder.load_from_idata()` method. Requires implementing custom serialization. |

**Note**: This is **not** a bug in the backwards compatibility system. This is a limitation of the `MixedLogit` class that needs to be fixed separately (requires implementing `_save_input_params()` and modifying load logic).

## How It Works

### 1. Capture Phase (`capture.py`)

```bash
python -m scripts.backcompat.capture <model_name> <output_dir>
```

For each model:
1. Builds the model using the definition in `scripts/backcompat/models/<model_name>.py`
2. Fits the model using mock sampling (fast, no actual MCMC)
3. Saves the fitted model to `<output_dir>/<model_name>.nc`
4. Creates `manifest.json` with metadata (version, sampler config, etc.)

### 2. Compare Phase (`compare.py`)

```bash
python -m scripts.backcompat.compare <path_to_manifest.json>
```

For each model:
1. Loads the baseline model from the manifest
2. Builds and fits a new candidate model with the same configuration
3. Compares:
   - Model version strings (must match)
   - Posterior variable names (must match)
   - Posterior shapes and dimensions (must match)
4. Raises `CompatibilityError` if any check fails

### 3. Mock Sampling

To enable fast testing without running expensive MCMC sampling, the system uses `pymc.testing.mock_sample` which:
- Replaces `pm.sample()` with `pm.sample_prior_predictive()`
- Replaces `pm.HalfFlat` with `pm.HalfNormal(sigma=10)`
- Replaces `pm.Flat` with `pm.Normal(mu=0, sigma=10)`

This is implemented in `scripts/backcompat/mock_pymc.py` via the `mock_sampling()` context manager.

**Important**: Because `pymc_extras.Prior` objects cache distribution classes, the compare script also explicitly replaces `HalfFlat` and `Flat` priors in model configs after loading (see `replace_incompatible_priors_in_config()` in `compare.py`).

### 4. Auto-Discovery

Models are **automatically discovered** by scanning `scripts/backcompat/models/` for Python files that define a `get_model_definition()` function. No manual registration in `__init__.py` is needed!

The discovery system:
1. Scans all `*.py` files in the models directory (except `_*.py` files)
2. Imports each module dynamically
3. Validates that `get_model_definition()` exists and is callable
4. Validates that it takes no arguments
5. Validates that it returns a `ModelDefinition` instance
6. Registers the model using the `name` field from the `ModelDefinition`

If a module is missing `get_model_definition()` or returns an invalid type, a `RuntimeError` is raised with a clear error message.

**Benefits**:
- ✅ No manual edits to `__init__.py` needed
- ✅ Impossible to forget to register a model
- ✅ Clear error messages if requirements aren't met
- ✅ Models are discovered at import time

## Adding New Models

To add a new model to backwards compatibility testing:

### 1. Create Model Definition

Create `scripts/backcompat/models/<model_name>.py`:

```python
from __future__ import annotations

import pandas as pd
from pymc_marketing.mmm import MMM
from ..model_definition import ModelDefinition


def _make_args() -> dict:
    """Create test data for the model."""
    data = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10),
        "channel_1": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
        "sales": [50, 75, 100, 125, 150, 175, 200, 225, 250, 275],
    })
    return {"data": data, "date_column": "date", "channel_columns": ["channel_1"]}


def _build_model(data, date_column, channel_columns) -> MMM:
    """Build the model instance."""
    return MMM(
        data=data,
        date_column=date_column,
        channel_columns=channel_columns,
    )


def get_model_definition() -> ModelDefinition:
    """
    Return the complete model definition.

    REQUIRED: Every model file must define this function with this exact signature.
    The system will auto-discover and register all models that define this function.
    """
    return ModelDefinition(
        name="my_new_model",
        builder_cls=MMM,
        builder_fn=_build_model,
        build_args_fn=_make_args,
        fit_args_fn=lambda: {},  # Optional fit kwargs
        sampler_kwargs={"chains": 1, "tune": 2, "draws": 2},  # Fast for testing
        fit_seed=42,
    )
```

**Important**: The `get_model_definition()` function is **required** and must:
- Be named exactly `get_model_definition`
- Take no arguments
- Return a `ModelDefinition` instance
- Be defined at the module level (not inside a class)

The model will be **automatically discovered and registered** - no need to manually edit `__init__.py`!

### 2. Update CI/CD Configuration (Optional)

If you want your model tested in CI/CD, add it to the passing models list.

Add to `.github/workflows/backcompat.yml`:

```yaml
strategy:
  matrix:
    model:
      - basic_mmm
      - beta_geo
      # ... other models ...
      - my_new_model  # Add here
```

Add to `scripts/backcompat/local_backcompat.sh`:

```bash
DEFAULT_MODELS=("basic_mmm" "beta_geo" "..." "my_new_model")
```

**Note**: Only add models that you've verified pass locally! Models that fail will block CI/CD.

### 3. Test Locally

```bash
# Test your new model only (fast iteration)
./scripts/backcompat/local_backcompat.sh my_new_model

# If it passes, run full suite to check for regressions
./scripts/backcompat/local_backcompat.sh
```

## Compatibility Checks

The system verifies:

1. **Model Version**: `model.version` must match between baseline and candidate
   - If incompatible changes are made, increment the version in the model class

2. **Posterior Variables**: Same set of random variables in the posterior
   - Adding/removing variables = breaking change

3. **Posterior Shapes**: Each variable must have identical dimensions
   - Changing parameter dimensionality = breaking change

4. **Posterior Dims**: Named dimensions must match (e.g., `chain`, `draw`, `channel`)
   - Renaming dimensions = breaking change

### When to Increment Model Version

Increment the model's version string when making **breaking changes**:
- Adding or removing model parameters
- Changing parameter dimensions or shapes
- Changing coordinate/dimension names
- Changing the model's mathematical structure

Example:
```python
class MMM(BaseValidateMMM):
    _model_type = "MMM"
    version = "0.0.3"  # ← Increment this for breaking changes
```

## Troubleshooting

### Issue: `RuntimeError: Model module 'my_model' must define a 'get_model_definition()' function`

**Cause**: Your model file is missing the required `get_model_definition()` function.

**Solution**: Add the function to your model file:
```python
def get_model_definition() -> ModelDefinition:
    return ModelDefinition(
        name="my_model",
        # ... other fields ...
    )
```

### Issue: `RuntimeError: 'get_model_definition()' should take no arguments`

**Cause**: Your `get_model_definition()` function has parameters.

**Solution**: Remove all parameters from the function signature:
```python
# ❌ Wrong
def get_model_definition(some_arg):
    ...

# ✅ Correct
def get_model_definition() -> ModelDefinition:
    ...
```

### Issue: `RuntimeError: must return a ModelDefinition, got <class 'dict'>`

**Cause**: Your `get_model_definition()` returns the wrong type.

**Solution**: Ensure it returns a `ModelDefinition` instance:
```python
from ..model_definition import ModelDefinition

def get_model_definition() -> ModelDefinition:
    return ModelDefinition(...)  # ← Must return this type
```

### Issue: `NotImplementedError: Cannot sample from half_flat variable`

**Cause**: Model uses `HalfFlat` or `Flat` priors, which don't support sampling.

**Solution**: Already handled! The `mock_sampling()` context manager automatically replaces these distributions. If you still see this error, ensure:
1. The model is being built inside a `mock_sampling()` context
2. The model config priors are replaced using `replace_incompatible_priors_in_config()`

### Issue: `ValueError: All arrays must be of the same length`

**Cause**: Test data in `_make_args()` has mismatched array lengths.

**Solution**: Fix the test data to ensure all DataFrame columns have the same length.

### Issue: `Model incompatible, increase the model version`

**Cause**: Posterior structure changed between baseline and candidate.

**Solution**:
1. If the change is intentional (breaking change), increment the model version
2. If unintentional, investigate what changed in the model structure

### Issue: GitHub Actions fails with "Undefined error: 0"

**Cause**: Known issue with `micromamba run` command.

**Solution**: The local script uses direct python paths instead. For CI/CD, ensure the workflow uses the pattern shown in `.github/workflows/backcompat.yml`.

## File Structure

```
scripts/backcompat/
├── README.md                  # This file
├── local_backcompat.sh        # Main script - test all or specific models
├── test_single_model.sh       # Convenience wrapper (forwards to local_backcompat.sh)
├── capture.py                 # Capture baseline models
├── compare.py                 # Compare models against baseline
├── mock_pymc.py              # Mock sampling context manager
├── utils.py                   # Utility functions (deprecated priors replacement)
├── model_definition.py        # ModelDefinition dataclass
└── models/
    ├── __init__.py           # Model auto-discovery and registry
    ├── basic_mmm.py          # Basic MMM model definition
    ├── beta_geo.py           # Beta Geo model definition
    ├── gamma_gamma.py        # Gamma-Gamma model definition
    └── ...                    # Other model definitions
```

**Note**: `local_backcompat.sh` is the main entry point. It can:
- Test all passing models (no arguments)
- Test specific models (pass model names as arguments)
- Show help (`--help` flag)

The `test_single_model.sh` is a thin wrapper for convenience (e.g., easier to remember for single model testing).

## CI/CD Integration

The backwards compatibility tests run automatically via GitHub Actions on:
- Every push to branches with backcompat changes
- Every pull request modifying backcompat files
- Manual workflow dispatch

See `.github/workflows/backcompat.yml` for the complete CI/CD configuration.

## Contributing

When contributing changes that affect model structure:

1. **Run local tests first**:
   ```bash
   ./scripts/backcompat/local_backcompat.sh
   ```

2. **If tests fail**:
   - If the change is breaking and intentional → increment model version
   - If the change is a bug → fix the code
   - If the test data is wrong → fix the test data

3. **Document changes**: Update this README if adding new models or changing behavior

4. **PR review**: Backwards compatibility test results are shown in the PR checks

---

**Last Updated**: 2026-02-09
**Maintained by**: PyMC Marketing Team
