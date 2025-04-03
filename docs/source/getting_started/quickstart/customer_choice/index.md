# Customer Choice Quickstart

```python
from pymc_marketing.customer_choice import (
    MVITS,
    generate_saturated_data,
    plot_product,
)

# Generate simulated data
scenario = {
    "total_sales_mu": 1000,
    "total_sales_sigma": 5,
    "treatment_time": 40,
    "n_observations": 100,
    "market_shares_before": [[0.7, 0.3, 0]],
    "market_shares_after": [[0.65, 0.25, 0.1]],
    "market_share_labels": ["competitor", "own", "new"],
    "random_seed": rng,
}

data = generate_saturated_data(**scenario)

# Build a multivariate interrupted time series model
model = MVITS(
    existing_sales=["competitor", "own"],
    saturated_market=True,
)

model.inform_default_prior(
    data=data.loc[: scenario1["treatment_time"], ["competitor", "own"]]
)

# Parameter estimation
model.sample(data[["competitor", "own"]], data["new"])

# Visualize the results
model.plot_fit();
```

See the {ref}`gallery` section for more information about using the customer choice module.
