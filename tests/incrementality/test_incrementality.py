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
import pytest
from matplotlib import pyplot as plt

from pymc_marketing.product_incrementality.mv_its import (
    MVITS,
    generate_saturated_data,
    generate_unsaturated_data,
)

rng = np.random.default_rng(123)

scenario_saturated = {
    "total_sales_mu": 1000,
    "total_sales_sigma": 5,
    "treatment_time": 40,
    "n_observations": 100,
    "market_shares_before": [[0.7, 0.3, 0]],
    "market_shares_after": [[0.65, 0.25, 0.1]],
    "market_share_labels": ["competitor", "own", "new"],
    "rng": rng,
}

scenario_unsaturated_bad = {
    "total_sales_before": [1000],
    "total_sales_after": [1400],
    "total_sales_sigma": 20,
    "treatment_time": 40,
    "n_observations": 100,
    "market_shares_before": [[0.7, 0.3, 0]],
    "market_shares_after": [[0.65, 0.25, 0.1]],
    "market_share_labels": ["competitor", "own", "new"],
    "rng": rng,
}

scenario_unsaturated_good = {
    "total_sales_before": [800],
    "total_sales_after": [950],
    "total_sales_sigma": 10,
    "treatment_time": 40,
    "n_observations": 100,
    "market_shares_before": [[500 / 800, 300 / 800, 0]],
    "market_shares_after": [[400 / 950, 200 / 950, 350 / 950]],
    "market_share_labels": ["competitor", "own", "new"],
    "rng": rng,
}

sample_kwargs = {"tune": 100, "draws": 100}


@pytest.fixture
def saturated_data_fixture():
    return generate_saturated_data(**scenario_saturated)


def test_plot_data(saturated_data_fixture):
    ax = MVITS.plot_data(saturated_data_fixture)
    assert isinstance(ax, plt.Axes)


def test_MVITS_saturated(saturated_data_fixture):
    result = MVITS(
        saturated_data_fixture,
        treatment_time=scenario_saturated["treatment_time"],
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

    ax = result.plot_causal_impact_sales()
    assert isinstance(ax, plt.Axes)

    ax = result.plot_causal_impact_market_share()
    assert isinstance(ax, plt.Axes)


@pytest.mark.parametrize(
    "scenario", [scenario_unsaturated_bad, scenario_unsaturated_good]
)
def test_MVITS_unsaturated(scenario):
    """We will test the `unsaturated` version of the MVITS model. And we will do this
    with multiple scenarios."""

    data = generate_unsaturated_data(**scenario)
    assert isinstance(data, pd.DataFrame)

    result = MVITS(
        data,
        treatment_time=scenario_saturated["treatment_time"],
        background_sales=["competitor", "own"],
        market_saturated=False,
        innovation_sales="new",
        rng=rng,
        sample_kwargs=sample_kwargs,
    )
    assert isinstance(result, MVITS)

    ax = result.plot_fit()
    assert isinstance(ax, plt.Axes)

    ax = result.plot_counterfactual()
    assert isinstance(ax, plt.Axes)

    ax = result.plot_causal_impact_sales()
    assert isinstance(ax, plt.Axes)

    ax = result.plot_causal_impact_market_share()
    assert isinstance(ax, plt.Axes)
