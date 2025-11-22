# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyMC-Marketing is a Bayesian marketing analytics library built on PyMC, providing three main modeling capabilities:

- **Marketing Mix Modeling (MMM)**: Measure marketing channel effectiveness with adstock, saturation, and budget optimization
- **Customer Lifetime Value (CLV)**: Predict customer value using probabilistic models (BG/NBD, Pareto/NBD, Gamma-Gamma, etc.)
- **Customer Choice Analysis**: Understand product selection with Multivariate Interrupted Time Series (MVITS) and discrete choice models

## Development Commands

### Environment Setup
```bash
# Create and activate conda environment (recommended)
conda env create -f environment.yml
conda activate pymc-marketing-dev

# Install package in editable mode
make init
```

### Testing and Quality
To use pytest you first need to activate the enviroment:
```bash
# Try to initialize conda (works if conda is in PATH or common locations)
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate pymc-marketing-dev || \
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" && conda activate pymc-marketing-dev
```

Running tests:
```bash
# first need to activate the enviorment:

# Run all tests with coverage
make test

# Run specific test file (you first need to activate the conda env: conda activate pymc-marketing-dev)
pytest tests/path/to/test_file.py

# Run specific test function (you first need to activate the conda env: conda activate pymc-marketing-dev)
pytest tests/path/to/test_file.py::test_function_name

# Check linting (ruff + mypy)
make check_lint

# Auto-fix linting issues
make lint

# Check code formatting
make check_format

# Auto-format code
make format
```

### Documentation
```bash
# Build HTML documentation
make html

# Clean docs and rebuild from scratch
make cleandocs && make html

# Run notebooks to verify examples
make run_notebooks           # All notebooks
make run_notebooks_mmm       # MMM notebooks only
make run_notebooks_other     # Non-MMM notebooks
```

### Other Utilities
```bash
# Generate UML diagrams for architecture
make uml

# Start MLflow tracking server
make mlflow_server
```

## High-Level Architecture

### Core Base Classes

**ModelBuilder** ([pymc_marketing/model_builder.py](pymc_marketing/model_builder.py))
- Abstract base class for all PyMC-Marketing models
- Defines the model lifecycle: `build_model()` → `fit()` → `predict()`
- Provides save/load functionality via NetCDF and InferenceData
- Manages `model_config` (priors) and `sampler_config` (MCMC settings)

**RegressionModelBuilder** (extends ModelBuilder)
- Adds scikit-learn-like API: `fit(X, y)`, `predict(X)`
- Base class for MMM and some customer choice models
- Handles prior/posterior predictive sampling

**CLVModel** ([pymc_marketing/clv/models/basic.py](pymc_marketing/clv/models/basic.py))
- Base class for CLV models (BetaGeo, ParetoNBD, GammaGamma, etc.)
- Takes data in constructor, not fit method: `model = BetaGeoModel(data=df)`
- Supports multiple inference methods: `method="mcmc"` (default), `"map"`, `"advi"`, etc.

### Module 1: MMM Architecture

**Class Hierarchy:**
```
RegressionModelBuilder
  └── MMMModelBuilder (mmm/base.py)
        ├── BaseMMM/MMM (mmm/mmm.py) - Single market
        └── MMM (mmm/multidimensional.py) - Panel/hierarchical data
```

**Component-Based Design:**

MMM uses composable transformation components:

1. **Adstock Transformations** ([pymc_marketing/mmm/components/adstock.py](pymc_marketing/mmm/components/adstock.py))
   - Model carryover effects of advertising
   - Built-in: GeometricAdstock, DelayedAdstock, WeibullCDFAdstock, WeibullPDFAdstock
   - All extend `AdstockTransformation` base class

2. **Saturation Transformations** ([pymc_marketing/mmm/components/saturation.py](pymc_marketing/mmm/components/saturation.py))
   - Model diminishing returns
   - Built-in: LogisticSaturation, HillSaturation, MichaelisMentenSaturation, TanhSaturation
   - All extend `SaturationTransformation` base class

3. **Transformation Protocol** ([pymc_marketing/mmm/components/base.py](pymc_marketing/mmm/components/base.py))
   - Base class defining transformation interface
   - Requires: `function()`, `prefix`, `default_priors`
   - Custom transformations should extend this

**Validation and Preprocessing System:**

MMM models use a decorator-based system:
- Methods tagged with `_tags = {"validation_X": True}` run during `fit(X, y)`
- Methods tagged with `_tags = {"preprocessing_y": True}` transform data before modeling
- Built-in validators in [pymc_marketing/mmm/validating.py](pymc_marketing/mmm/validating.py)
- Built-in preprocessors in [pymc_marketing/mmm/preprocessing.py](pymc_marketing/mmm/preprocessing.py)

**Key MMM Features:**
- Time-varying parameters via HSGP (Hilbert Space Gaussian Process)
- Lift test calibration for experiments
- Budget optimization ([pymc_marketing/mmm/budget_optimizer.py](pymc_marketing/mmm/budget_optimizer.py))
- Causal DAG support ([pymc_marketing/mmm/causal.py](pymc_marketing/mmm/causal.py))
- Additive effects system ([pymc_marketing/mmm/additive_effect.py](pymc_marketing/mmm/additive_effect.py)) for custom components

**Multidimensional MMM vs Base MMM:**
- Base MMM ([pymc_marketing/mmm/mmm.py](pymc_marketing/mmm/mmm.py)): Single market, simpler API
- Multidimensional MMM ([pymc_marketing/mmm/multidimensional.py](pymc_marketing/mmm/multidimensional.py)): Panel data, per-channel transformations via `MediaConfigList`, more flexible

### Module 2: CLV Architecture

**Available Models:**
- BetaGeoModel: Beta-Geometric/NBD for continuous non-contractual settings
- ParetoNBDModel: Pareto/NBD alternative formulation
- GammaGammaModel: Monetary value prediction
- ShiftedBetaGeoModel, ModifiedBetaGeoModel: Variants
- BetaGeoBetaBinomModel: Discrete time variant

**CLV Pattern:**
```python
# Data passed to constructor, not fit()
model = clv.BetaGeoModel(data=df)

# Fit with various inference methods
model.fit(method="mcmc")  # or "map", "advi", "fullrank_advi"

# Predict for known customers
model.expected_purchases(customer_id, t)
model.probability_alive(customer_id)
```

**Custom Distributions:**
CLV models use custom distributions in [pymc_marketing/clv/distributions.py](pymc_marketing/clv/distributions.py)

### Module 3: Customer Choice

- **MVITS** ([pymc_marketing/customer_choice/mv_its.py](pymc_marketing/customer_choice/mv_its.py)): Multivariate Interrupted Time Series for product launch incrementality
- **Discrete Choice Models**: Logit models in [pymc_marketing/customer_choice/](pymc_marketing/customer_choice/)

### Cross-Cutting Systems

**Prior Configuration System** ([pymc_marketing/prior.py](pymc_marketing/prior.py), now in pymc_extras)
- Declarative prior specification outside PyMC context
- Example: `Prior("Normal", mu=0, sigma=1)`
- Supports hierarchical priors, non-centered parameterization, transformations
- Used in all `model_config` dictionaries

**Model Configuration** ([pymc_marketing/model_config.py](pymc_marketing/model_config.py))
- `parse_model_config()` converts dicts to Prior objects
- Handles nested priors for hierarchical models
- Supports HSGP kwargs for Gaussian processes

**Save/Load Infrastructure**
- Models save to NetCDF via ArviZ InferenceData
- `model.save("filename.nc")` serializes model + data + config
- `Model.load("filename.nc")` reconstructs from file
- Training data stored in `idata.fit_data` group

**MLflow Integration** ([pymc_marketing/mlflow.py](pymc_marketing/mlflow.py))
- `autolog()` patches PyMC and PyMC-Marketing functions
- Automatically logs: model structure, diagnostics (r_hat, ESS, divergences), MMM/CLV configs
- Start server with: `make mlflow_server`

## Code Style and Testing

**Linting:**
- Uses Ruff for linting and formatting
- Uses mypy for type checking
- Config in [pyproject.toml](pyproject.toml) under `[tool.ruff]` and `[tool.mypy]`
- Docstrings follow NumPy style guide

**Testing:**
- pytest with coverage reporting
- Config in [pyproject.toml](pyproject.toml) under `[tool.pytest.ini_options]`
- Test files mirror package structure in [tests/](tests/)

**Pre-commit Hooks:**
```bash
pre-commit install  # Set up hooks
pre-commit run --all-files  # Run manually
```

## Important Patterns and Conventions

### Adding a New MMM Transformation

1. Extend `AdstockTransformation` or `SaturationTransformation` from [pymc_marketing/mmm/components/base.py](pymc_marketing/mmm/components/base.py)
2. Implement: `function()`, `prefix` property, `default_priors` property
3. Add to [pymc_marketing/mmm/components/adstock.py](pymc_marketing/mmm/components/adstock.py) or [saturation.py](pymc_marketing/mmm/components/saturation.py)
4. Export in [pymc_marketing/mmm/__init__.py](pymc_marketing/mmm/__init__.py)

### Adding a New CLV Model

1. Extend `CLVModel` from [pymc_marketing/clv/models/basic.py](pymc_marketing/clv/models/basic.py)
2. Implement: `build_model()`, prediction methods (e.g., `expected_purchases()`)
3. Define required data columns in `__init__`
4. Add tests in [tests/clv/models/](tests/clv/models/)

### Adding a New Additive Effect (MMM)

1. Implement `MuEffect` protocol from [pymc_marketing/mmm/additive_effect.py](pymc_marketing/mmm/additive_effect.py)
2. Required methods: `create_data()`, `create_effect()`, `set_data()`
3. See FourierEffect, LinearTrendEffect as examples

### Model Lifecycle

All models follow this pattern:
1. **Configuration**: Store data and config in `__init__`
2. **Build**: `build_model()` creates PyMC model, attaches to `self.model`
3. **Fit**: `fit()` calls `pm.sample()` or alternative inference
4. **Store**: Results stored in `self.idata` (ArviZ InferenceData)
5. **Predict**: `sample_posterior_predictive()` with new data

## Documentation and Examples

**Notebooks:**
- MMM examples: [docs/source/notebooks/mmm/](docs/source/notebooks/mmm/)
- CLV examples: [docs/source/notebooks/clv/](docs/source/notebooks/clv/)
- Customer choice: [docs/source/notebooks/customer_choice/](docs/source/notebooks/customer_choice/)

**Gallery Generation:**
- [scripts/generate_gallery.py](scripts/generate_gallery.py) creates notebook gallery for docs
- Run with `make html`

**UML Diagrams:**
- Architecture diagrams in [docs/source/uml/](docs/source/uml/)
- Generate with `make uml`
- See [CONTRIBUTING.md](CONTRIBUTING.md) for package/class diagrams

## Community and Support

- [GitHub Issues](https://github.com/pymc-labs/pymc-marketing/issues) for bugs/features
- [PyMC Discourse](https://discourse.pymc.io/) for general discussion
- [PyMC-Marketing Discussions](https://github.com/pymc-labs/pymc-marketing/discussions) for Q&A
