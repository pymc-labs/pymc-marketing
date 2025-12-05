# Copilot Instructions for PyMC-Marketing

This document provides instructions for GitHub Copilot when working in the PyMC-Marketing repository.

## Project Overview

PyMC-Marketing is a Python library for Bayesian marketing analytics built on top of PyMC. The library provides tools for:

- **Marketing Mix Modeling (MMM)**: Bayesian models for measuring marketing effectiveness, including adstock transformations, saturation effects, and budget optimization
- **Customer Lifetime Value (CLV)**: Models for predicting customer value, including BG/NBD, Pareto/NBD, and Gamma-Gamma models
- **Customer Choice Analysis**: Multivariate Interrupted Time Series (MVITS) models for analyzing product launches
- **Bass Diffusion Models**: For predicting adoption of new products

## Project Structure

```
pymc_marketing/
├── mmm/           # Marketing Mix Modeling components
├── clv/           # Customer Lifetime Value models
├── customer_choice/  # Customer choice analysis (MVITS)
├── bass/          # Bass diffusion models
├── model_builder.py  # Base model building utilities
└── prior.py       # Prior distribution utilities
```

## Development Setup

### Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate pymc-marketing-dev

# Install package in editable mode with development dependencies
make init
```

### Pre-commit Hooks

```bash
pre-commit install
```

## Code Style and Conventions

### Formatting and Linting

- **Formatter**: Ruff (configured in `pyproject.toml`)
- **Linter**: Ruff and mypy
- **Max line length**: 120 characters
- **Docstring format**: NumPy style

```bash
# Check linting
make check_lint

# Auto-fix linting issues
make lint

# Check formatting
make check_format

# Auto-fix formatting
make format
```

### Documentation Style

All public methods must have informative docstrings following the [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html):

```python
def function_name(param1: type, param2: type) -> return_type:
    """Short description of the function.

    Longer description if needed.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.

    Returns
    -------
    return_type
        Description of return value.

    Examples
    --------
    Doctest style:

    >>> function_name(value1, value2)
    expected_output

    Or Sphinx code block style:

    .. code-block:: python

        result = function_name(value1, value2)

    """
```

### Type Hints

- Use type hints for all function parameters and return values
- Import types from `typing` module when needed
- The codebase uses `py.typed` marker for PEP 561 compliance

### Testing

Ensure the conda environment is activated before running tests:

```bash
# Activate environment first
conda activate pymc-marketing-dev

# Run all tests
make test

# Run specific test file
pytest tests/path/to/test_file.py

# Run with verbose output
pytest -v tests/
```

Tests are located in the `tests/` directory and mirror the structure of `pymc_marketing/`.

## Key Dependencies

See `pyproject.toml` for version requirements. Main dependencies include:

- **PyMC**: Core probabilistic programming framework
- **PyTensor**: Tensor computation library
- **ArviZ**: Bayesian model diagnostics and visualization
- **pandas**: Data manipulation
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **xarray**: Multi-dimensional arrays with labels
- **Pydantic**: Data validation
- **pymc-extras**: Extended PyMC utilities (priors, etc.)

## Common Patterns

### Model Building

Models in PyMC-Marketing typically inherit from base classes in `model_builder.py`:

```python
from pymc_marketing.model_builder import ModelBuilder

class MyModel(ModelBuilder):
    # Model implementation
    pass
```

### Prior Distributions

Use the `Prior` class from `pymc_extras.prior` for defining priors:

```python
from pymc_extras.prior import Prior

prior = Prior("Normal", mu=0, sigma=1)
```

### Data Loading

Use paths from `pymc_marketing.paths` for accessing example data:

```python
from pymc_marketing.paths import data_dir

file_path = data_dir / "example_data.csv"
```

## Building Documentation

```bash
# Build HTML documentation
make html

# Clean and rebuild
make cleandocs
make html
```

Documentation is in `docs/source/` and built with Sphinx.

## Pull Request Guidelines

- Reference the related issue in the PR description
- Include tests for new functionality
- Update documentation for API changes
- Ensure all linting and tests pass
- Follow the PR checklist in `.github/pull_request_template.md`

## Additional Resources

- [CONTRIBUTING.md](../CONTRIBUTING.md): Full contribution guidelines
- [PyMC-Marketing Documentation](https://www.pymc-marketing.io)
- [PyMC Documentation](https://www.pymc.io)
