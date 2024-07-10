#   Copyright 2024 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pymc_marketing.product_incrementality.mv_its import MVITS

rng = np.random.default_rng(123)

scenario = {
    "total_sales_mu": 1000,
    "total_sales_sigma": 5,
    "treatment_time": 40,
    "n_observations": 100,
    "market_shares_before": [[0.7, 0.3, 0]],
    "market_shares_after": [[0.65, 0.25, 0.1]],
    "market_share_labels": ["competitor", "own", "new"],
}

sample_kwargs = {"tune": 100, "draws": 100}


def generate_data(
    total_sales_mu: int,
    total_sales_sigma: float,
    treatment_time: int,
    n_observations: int,
    market_shares_before,
    market_shares_after,
    market_share_labels,
):
    rates = np.array(
        treatment_time * market_shares_before
        + (n_observations - treatment_time) * market_shares_after
    )

    # Generate total demand (sales) as normally distributed around some average level of sales
    total = (
        rng.normal(loc=total_sales_mu, scale=total_sales_sigma, size=n_observations)
    ).astype(int)

    # Ensure total sales are never negative
    total[total < 0] = 0

    # Generate sales counts
    counts = rng.multinomial(total, rates)

    # Convert to DataFrame
    data = pd.DataFrame(counts)
    data.columns = market_share_labels
    data.columns.name = "product"
    data.index.name = "day"
    data["pre"] = data.index < treatment_time
    return data


def test_plot_data():
    data = generate_data(**scenario)
    ax = MVITS.plot_data(data)
    assert ax is not None


def test_MVITS():
    data = generate_data(**scenario)
    result = MVITS(
        data,
        treatment_time=scenario["treatment_time"],
        background_sales=["competitor", "own"],
        innovation_sales="new",
        rng=rng,
        sample_kwargs=sample_kwargs,
    )
    assert isinstance(result, MVITS)

    ax = result.plot_fit()
    assert isinstance(ax, plt.Axes)

    ax = result.plot_counterfactual()
    assert isinstance(ax, plt.Axes)

    ax = result.plot_causal_impact()
    assert isinstance(ax, plt.Axes)

    result.plot_causal_impact(type="sales")
    assert isinstance(ax, plt.Axes)

    result.plot_causal_impact(type="market_share")
    assert isinstance(ax, plt.Axes)
