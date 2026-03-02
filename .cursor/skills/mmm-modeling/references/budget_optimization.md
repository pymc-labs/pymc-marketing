# Budget Optimization

## Table of Contents
- [Setup](#setup)
- [Budget Bounds](#budget-bounds)
- [Running Optimization](#running-optimization)
- [Response Sampling](#response-sampling)
- [Plotting Results](#plotting-results)
- [Advanced Patterns](#advanced-patterns)

## Setup

Budget optimization uses `MultiDimensionalBudgetOptimizerWrapper`, which wraps a fitted MMM and optimizes spend allocation over a future date range:

```python
import pandas as pd
from pymc_marketing.mmm.multidimensional import MultiDimensionalBudgetOptimizerWrapper

# Define optimization window (typically future periods)
last_date = pd.Timestamp(X[date_column].max())
start_date = last_date + pd.Timedelta(weeks=1)
end_date = start_date + pd.Timedelta(weeks=12)

optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
    model=mmm,
    start_date=str(start_date),
    end_date=str(end_date),
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `MMM` | Fitted MMM instance |
| `start_date` | `str` | Start of optimization window (ISO format) |
| `end_date` | `str` | End of optimization window (ISO format) |
| `compile_kwargs` | `dict \| None` | Optional compilation arguments |

## Budget Bounds

Budget bounds constrain per-channel (and optionally per-geo) allocation. They are specified as `xr.DataArray` objects:

### Single-Geo Bounds

```python
import numpy as np
import xarray as xr

budget_bounds = xr.DataArray(
    data=np.array([
        [0.5, 1.5],   # tv: 50%-150% of equal share
        [0.3, 2.0],   # radio: 30%-200%
        [0.5, 1.5],   # social
    ]) * equal_share_per_channel,
    dims=["channel", "bound"],
    coords={
        "channel": channel_columns,
        "bound": ["lower", "upper"],
    },
)
```

### Multidimensional Bounds (Channel x Geo)

```python
n_channels = len(channel_columns)
n_geos = len(geos)

budget_bounds = xr.DataArray(
    data=np.stack([
        np.full((n_channels, n_geos), 0.0),       # lower bounds
        np.full((n_channels, n_geos), max_budget),  # upper bounds
    ], axis=-1),
    dims=["channel", "geo", "bound"],
    coords={
        "channel": channel_columns,
        "geo": geos,
        "bound": ["lower", "upper"],
    },
)
```

### Using the Builder Helper

```python
import numpy as np
from pymc_marketing.mmm.budget_optimizer import optimizer_xarray_builder

budget_bounds = optimizer_xarray_builder(
    value=np.array([
        [0.5 * equal_share, 1.5 * equal_share],   # tv
        [0.3 * equal_share, 2.0 * equal_share],   # radio
        [0.5 * equal_share, 1.5 * equal_share],   # social
    ]),
    channel=channel_columns,
    bound=["lower", "upper"],
)
```

## Running Optimization

```python
allocation, result = optimizable_model.optimize_budget(
    budget=budget_per_period,
    budget_bounds=budget_bounds,
    minimize_kwargs={
        "method": "SLSQP",
        "options": {"ftol": 1e-4, "maxiter": 10_000},
    },
)
```

| Parameter | Description |
|-----------|-------------|
| `budget` | Total budget per period to allocate |
| `budget_bounds` | `xr.DataArray` with lower/upper bounds |
| `minimize_kwargs` | Arguments passed to `scipy.optimize.minimize` |

The `allocation` is an `xr.DataArray` with the optimal spend per channel (and per geo if multidimensional).

## Response Sampling

After optimization, sample the posterior predictive response under the optimal allocation:

```python
response = optimizable_model.sample_response_distribution(
    allocation_strategy=allocation,
    additional_var_names=[
        "channel_contribution_original_scale",
        "intercept_contribution_original_scale",
    ],
    include_last_observations=True,
    include_carryover=True,
    noise_level=0.05,
)
```

| Parameter | Description |
|-----------|-------------|
| `allocation_strategy` | Output from `optimize_budget()` |
| `additional_var_names` | Extra variables to include in the response |
| `include_last_observations` | Include adstock carryover from historical data |
| `include_carryover` | Include inter-period carryover in the optimization window |
| `noise_level` | Scale of observation noise (0 = deterministic) |

## Plotting Results

### Contribution by Channel Over Time

```python
mmm.plot.allocated_contribution_by_channel_over_time(response)
```

### Budget Allocation Summary

```python
optimizable_model.plot.budget_allocation(samples=response)
```

This produces a multi-panel summary showing:
- Optimal allocation per channel
- Expected response distribution
- Channel-level contribution breakdown

## Advanced Patterns

### Fixing Certain Channels

Optimize only a subset of channels while keeping others at fixed budgets:

```python
budgets_to_optimize = xr.DataArray(
    data=[True, True, False],  # only optimize tv and radio; fix social
    dims=["channel"],
    coords={"channel": channel_columns},
)

allocation, result = optimizable_model.optimize_budget(
    budget=budget_per_period,
    budget_bounds=budget_bounds,
    budgets_to_optimize=budgets_to_optimize,
    minimize_kwargs={"method": "SLSQP", "options": {"ftol": 1e-4, "maxiter": 10_000}},
)
```

### Custom Budget Distribution Over Time

By default, the budget is distributed equally across periods. Use `budget_distribution_over_period` for custom temporal patterns (flighting):

```python
import xarray as xr

n_periods = 12
time_weights = np.linspace(1.5, 0.5, n_periods)
time_weights /= time_weights.sum()

# For single-geo models: dims=("date", "channel")
budget_distribution = xr.DataArray(
    data=np.tile(time_weights[:, None], (1, len(channel_columns))),
    dims=["date", "channel"],
    coords={"channel": channel_columns},
)

allocation, result = optimizable_model.optimize_budget(
    budget=budget_per_period,
    budget_bounds=budget_bounds,
    budget_distribution_over_period=budget_distribution,
    minimize_kwargs={"method": "SLSQP", "options": {"ftol": 1e-4, "maxiter": 10_000}},
)

# Also pass to sample_response_distribution for consistency
response = optimizable_model.sample_response_distribution(
    allocation_strategy=allocation,
    budget_distribution_over_period=budget_distribution,
    include_carryover=True,
)
```

Values along the `date` dimension must sum to 1 for each combination of other dimensions.

### Budget Sweep

Evaluate how optimal response changes across different total budgets:

```python
import numpy as np

budget_levels = np.linspace(50_000, 500_000, 10)
sweep_results = []

for budget in budget_levels:
    alloc, res = optimizable_model.optimize_budget(
        budget=budget,
        budget_bounds=budget_bounds,
        minimize_kwargs={"method": "SLSQP", "options": {"ftol": 1e-4, "maxiter": 10_000}},
    )
    sweep_results.append({"budget": budget, "allocation": alloc, "result": res})
```

Plot the efficient frontier (budget vs. expected response) to identify the point of diminishing returns.

### Custom Constraints

For constraints beyond simple bounds (e.g., minimum spend ratios between channels), pass `Constraint` objects to `optimize_budget`:

```python
import pytensor.tensor as pt
from pymc_marketing.mmm.constraints import Constraint

# tv must be at least 2x radio: tv - 2*radio >= 0
def tv_ge_2x_radio(budgets_sym, total_budget_sym, optimizer):
    tv_idx = list(channel_columns).index("tv")
    radio_idx = list(channel_columns).index("radio")
    return budgets_sym[tv_idx] - 2 * budgets_sym[radio_idx]

constraint = Constraint(
    key="tv_ge_2x_radio",
    constraint_type="ineq",
    constraint_fun=tv_ge_2x_radio,
)

allocation, result = optimizable_model.optimize_budget(
    budget=budget_per_period,
    budget_bounds=budget_bounds,
    constraints=[constraint],
    minimize_kwargs={"method": "SLSQP", "options": {"ftol": 1e-4, "maxiter": 10_000}},
)
```

### In-Sample Uplift Estimation

Estimate what would have happened with a different historical allocation by reweighting spend:

```python
X_counterfactual = X.copy()
X_counterfactual["tv"] = X_counterfactual["tv"] * 1.2   # 20% more TV
X_counterfactual["radio"] = X_counterfactual["radio"] * 0.8  # 20% less radio

response_cf = mmm.sample_posterior_predictive(
    X_counterfactual, extend_idata=False, random_seed=42,
)

uplift = (
    response_cf["posterior_predictive"]["y"].mean(dim=("chain", "draw"))
    - mmm.idata["posterior_predictive"]["y"].mean(dim=("chain", "draw"))
)
```

---

## See Also

- [media_deep_dive.md](media_deep_dive.md) -- ROAS and saturation analysis (informs budget decisions)
- [model_fit.md](model_fit.md) -- Model must be well-diagnosed before optimization
- [liftest_calibration.md](liftest_calibration.md) -- Calibration improves optimization reliability
