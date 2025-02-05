#   Copyright 2022 - 2025 The PyMC Labs Developers
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

import re

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from xarray import DataArray

from pymc_marketing.customer_choice import (
    MVITS,
    generate_saturated_data,
    generate_unsaturated_data,
)
from pymc_marketing.prior import Prior

seed = sum(map(ord, "CustomerChoice"))
rng = np.random.default_rng(seed)


scenario_saturated = {
    "total_sales_mu": 1000,
    "total_sales_sigma": 5,
    "treatment_time": 40,
    "n_observations": 100,
    "market_shares_before": [[0.7, 0.3, 0]],
    "market_shares_after": [[0.65, 0.25, 0.1]],
    "market_share_labels": ["competitor", "own", "new"],
    "random_seed": rng,
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
    "random_seed": rng,
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
    "random_seed": rng,
}


@pytest.fixture(scope="module")
def saturated_data():
    return generate_saturated_data(**scenario_saturated)


def test_plot_data(saturated_data):
    model = MVITS(existing_sales=["competitor", "own"])
    model.X = saturated_data.loc[:, ["competitor", "own"]]
    model.y = saturated_data["new"]

    ax = model.plot_data()
    assert isinstance(ax, Axes)
    plt.close()


@pytest.fixture(scope="module")
def fit_model(saturated_data, mock_pymc_sample):
    model = MVITS(existing_sales=["competitor", "own"], saturated_market=True)

    model.sample(
        saturated_data.loc[:, ["competitor", "own"]],
        saturated_data["new"],
        random_seed=rng,
        sample_prior_predictive_kwargs={"samples": 10},
    )
    return model


@pytest.mark.parametrize(
    "plot_method",
    [
        "plot_fit",
        "plot_counterfactual",
        "plot_causal_impact_sales",
        "plot_causal_impact_market_share",
    ],
)
def test_MVITS_saturated(fit_model, plot_method):
    ax = getattr(fit_model, plot_method)()
    assert isinstance(ax, Axes)
    plt.close()


@pytest.fixture(scope="module")
def unsaturated_data_bad():
    return generate_unsaturated_data(**scenario_unsaturated_bad)


@pytest.fixture(scope="module")
def unsaturated_data_good():
    return generate_unsaturated_data(**scenario_unsaturated_good)


@pytest.fixture(scope="module")
def unsaturated_model_bad(unsaturated_data_bad, mock_pymc_sample):
    model = MVITS(existing_sales=["competitor", "own"], saturated_market=False)

    model.sample(
        unsaturated_data_bad.loc[:, ["competitor", "own"]],
        unsaturated_data_bad["new"],
        random_seed=rng,
        sample_prior_predictive_kwargs={"samples": 10},
    )
    return model


@pytest.fixture(scope="module")
def unsaturated_model_good(unsaturated_data_good, mock_pymc_sample):
    model = MVITS(existing_sales=["competitor", "own"], saturated_market=False)

    model.sample(
        unsaturated_data_good.loc[:, ["competitor", "own"]],
        unsaturated_data_good["new"],
        random_seed=rng,
        sample_prior_predictive_kwargs={"samples": 10},
    )
    return model


@pytest.mark.parametrize(
    "model_name", ["unsaturated_model_bad", "unsaturated_model_good"]
)
@pytest.mark.parametrize(
    "plot_method",
    [
        "plot_fit",
        "plot_counterfactual",
        "plot_causal_impact_sales",
        "plot_causal_impact_market_share",
    ],
)
def test_MVITS_unsaturated(request, model_name, plot_method):
    """We will test the `unsaturated` version of the MVITS model. And we will do this
    with multiple scenarios."""

    model = request.getfixturevalue(model_name)

    ax = getattr(model, plot_method)()
    assert isinstance(ax, Axes)
    plt.close()


def test_save_load(fit_model, saturated_data) -> None:
    test_file = "test-mvits.nc"
    fit_model.save(test_file)

    loaded = MVITS.load(test_file)

    assert loaded.model_config == fit_model.model_config
    assert loaded.existing_sales == fit_model.existing_sales
    assert loaded.saturated_market == fit_model.saturated_market
    assert loaded.X.columns.name is None
    pd.testing.assert_frame_equal(loaded.X, fit_model.X, check_names=False)
    assert loaded.y.name == fit_model.y.name
    pd.testing.assert_series_equal(
        loaded.y,
        saturated_data["new"].rename(fit_model.output_var),
    )


@pytest.mark.parametrize("variable", ["y", "mu"])
def test_causal_impact(fit_model, variable) -> None:
    causal_impact = fit_model.causal_impact(variable=variable)

    assert isinstance(causal_impact, DataArray)
    assert causal_impact.dims == ("chain", "draw", "time", "existing_product")
    assert causal_impact.name == variable


def test_distribution_checks_wrong_market_distribution() -> None:
    priors = {
        "market_distribution": Prior("HalfNormal"),
    }

    match = "market_distribution must be a Dirichlet distribution"
    with pytest.raises(ValueError, match=match):
        MVITS(existing_sales=["competitor", "own"], model_config=priors)


def test_distribution_checks_wrong_dims() -> None:
    priors = {
        "market_distribution": Prior("Dirichlet", dims="wrong"),
    }

    match = re.escape(
        "market_distribution must have dims='existing_product', not ('wrong',)"
    )
    with pytest.raises(ValueError, match=match):
        MVITS(existing_sales=["competitor", "own"], model_config=priors)


@pytest.fixture
def data_to_inform_prior() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "competitor": [60, 80, 100],
            "own": [30, 20, 10],
        },
    )


def test_inform_default_prior(data_to_inform_prior) -> None:
    model = MVITS(existing_sales=["competitor", "own"])
    model.inform_default_prior(data_to_inform_prior)

    expected = {
        "market_distribution": Prior(
            "Dirichlet",
            a=[0.5, 0.5],
            dims="existing_product",
        ),
        "intercept": Prior(
            "Normal",
            mu=[80, 20],
            sigma=[20, 10],
            dims="existing_product",
        ),
        "likelihood": Prior(
            "TruncatedNormal",
            lower=0,
            sigma=Prior("HalfNormal", sigma=15, dims="existing_product"),
            dims=("time", "existing_product"),
        ),
    }
    assert model.model_config == expected


@pytest.mark.parametrize(
    "priors, match",
    [
        (
            {"intercept": Prior("HalfNormal", sigma=15, dims="existing_product")},
            "intercept must be a Normal distribution",
        ),
        (
            {
                "likelihood": Prior(
                    "Normal",
                    sigma=Prior("Gamma", alpha=1, beta=1, dims="existing_product"),
                    dims=("time", "existing_product"),
                )
            },
            "likelihood sigma must be a HalfNormal distribution",
        ),
    ],
    ids=["intercept", "likelihood"],
)
def test_inform_default_prior_raises(priors, match, saturated_data) -> None:
    model = MVITS(existing_sales=["competitor", "own"], model_config=priors)
    with pytest.raises(ValueError, match=match):
        model.inform_default_prior(saturated_data.loc[:, model.existing_sales])


def test_calculate_counterfactual_raises() -> None:
    model = MVITS(existing_sales=["competitor", "own"])
    match = "Call the 'fit' method first."
    with pytest.raises(RuntimeError, match=match):
        model.calculate_counterfactual()
