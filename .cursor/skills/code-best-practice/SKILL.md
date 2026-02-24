---
name: code-best-practice
description: PyMC-Marketing coding conventions, preferred implementations, and style guidelines.
disable-model-invocation: true
---

# PyMC-Marketing Best Practices Guide

This document outlines the coding conventions, preferred implementations, and style guidelines for contributing to `pymc-marketing`.

## 1. Coding Style & Standards

We follow strict linting and formatting rules enforced by **Ruff** and **MyPy**.

### Python Conventions
- **Type Hints**: All function arguments and return types must be typed.
  ```python
  # Good
  def calculate_metric(data: pd.DataFrame, col: str) -> xarray.DataArray: ...

  # Bad
  def calculate_metric(data, col): ...
  ```
- **Line Length**: Maximum 120 characters.
- **Imports**: Sorted automatically by Ruff.
- **Validation**: Use `pydantic` for runtime validation of user inputs in `__init__` methods.

### Docstrings (NumPy Style)
All public classes and functions must have docstrings following the **NumPy style**.
- **Sections**: Summary, Parameters, Returns, References (if applicable), Examples.
- **Math**: Include citations to relevant papers for statistical models.
- **Examples**: Use `.. code-block:: python` directive for code examples.

```python
def expected_purchases(self, future_t: int) -> xarray.DataArray:
    """
    Compute expected number of future purchases.

    Parameters
    ----------
    future_t : int
        Number of time periods to predict.

    Returns
    -------
    xarray.DataArray
        The expected number of purchases.

    Examples
    --------
    .. code-block:: python

        model = MyModel(data)
        model.fit()
        model.expected_purchases(future_t=12)

    References
    ----------
    .. [1] Fader, P. S., et al. (2005). "Counting Your Customers..."
    """
```

## 2. Preferred Implementations for Speed & Scalability

### Vectorization over Loops
Avoid Python loops for mathematical operations. Use `numpy`, `xarray`, or `pytensor` (via `pymc`) broadcasting.

- **Why**: Python loops are slow; vectorized operations in C/Fortran backends are orders of magnitude faster.

```python
# Bad: Iterating over customers
results = []
for customer in customers:
    results.append(calculate_val(customer))

# Good: Vectorized operation
results = alpha * np.exp(-beta * data)
```

### Efficient PyMC Modeling
1. **Use `pm.Data` for Mutable Inputs**:
   Allows you to change data (e.g., for out-of-sample predictions) without rebuilding the model graph.
   ```python
   # In build_model
   self.model_coords = {"customer_id": unique_ids}
   with pm.Model(coords=self.model_coords) as self.model:
       # Mutable data container
       x_data = pm.Data("x_data", data[cols], dims="customer_id")
       ...
   ```

2. **Batch Dimensions (Coords)**:
   Use named dimensions (`dims`) instead of raw shapes. This integrates with `xarray` for post-processing.
   ```python
   # Good
   alpha = pm.Normal("alpha", mu=0, sigma=1, dims="channel")
   ```

3. **HSGP for Gaussian Processes**:
   For time-varying parameters (like in MMM), prefer **Hilbert Space Gaussian Processes (HSGP)** over standard GPs. HSGP approximates the GP using basis functions, reducing complexity from $O(n^3)$ to $O(n \cdot m)$.

### PyTensor for Optimization & Analysis
When implementing functionality like sensitivity analysis, optimization routines, or complex transformations, **prefer `pytensor` operations** over pure Python/NumPy.

- **Backend Capabilities**: PyTensor graphs can be compiled to C, JAX, or MLX (Apple Silicon), enabling hardware acceleration and automatic differentiation.
- **Automatic Differentiation**: Essential for gradient-based optimization and sensitivity analysis.

```python
import pytensor.tensor as pt

# Good: PyTensor implementation
def saturation(x, alpha):
    return 1 - pt.exp(-alpha * x)

# This graph can now be differentiated with respect to alpha
```

## 3. Class Structure & Design Patterns

### Model Class Architecture
Models should inherit from base classes (`BaseMMM`, `CLVModel`) and implement specific lifecycle methods:

1.  **`__init__`**:
    -   Validate inputs using `validate_call` or `pydantic`.
    -   Store configuration (priors) in a `model_config` dictionary.
    -   **Do not** build the PyMC model here.

2.  **`build_model`**:
    -   Constructs the PyMC model context.
    -   Defines `pm.Data` containers and random variables.

3.  **`_extract_predictive_variables`**:
    -   Helper to prepare input data for predictions, converting `pandas` to `xarray`.

### Configuration Management & Priors
Use a `default_model_config` property to define default priors. This allows users to easily override specific priors without rewriting the whole model.

**Always use `pymc_extras.prior.Prior` for defining distributions.** This provides a dictionary-based specification that is serializable and easy for users to modify.

```python
from pymc_extras.prior import Prior

@property
def default_model_config(self) -> dict:
    return {
        "alpha": Prior("Weibull", alpha=2, beta=10),
        "beta": Prior("Normal", mu=0, sigma=1),
    }
```

## 4. Testing Best Practices

- **Framework**: `pytest`.
- **Coverage**: High test coverage is required.
- **Parametrization**: Use `@pytest.mark.parametrize` to test multiple scenarios efficiently.
- **Synthetic Data**: Use helper functions to generate synthetic data for testing model recovery.

```python
@pytest.mark.parametrize("future_t", [1, 10])
def test_expected_purchases(model, future_t):
    pred = model.expected_purchases(future_t=future_t)
    assert pred.shape == (4000, 100)  # (chains*draws, customers)
```

## 5. Workflow Checklist

Before submitting a PR:
1.  **Lint**: `make lint` (runs Ruff).
2.  **Type Check**: `pre-commit run mypy --all-files`.
3.  **Test**: `make test`.
4.  **Docs**: Ensure new public methods have docstrings.
