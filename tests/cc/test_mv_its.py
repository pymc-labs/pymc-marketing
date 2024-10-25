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

import warnings

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from pymc_marketing.cc.mv_its import (
    MVITS,
    generate_saturated_data,
    generate_unsaturated_data,
)

seed = sum(map(ord, "Product Incrementality"))
rng = np.random.default_rng(seed)


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


@pytest.fixture(scope="module")
def saturated_data():
    return generate_saturated_data(**scenario_saturated)


def test_plot_data(saturated_data):
    model = MVITS(existing_sales=["competitor", "own"])
    model.X = saturated_data.loc[:, ["competitor", "own"]]
    model.y = saturated_data["new"]

    ax = model.plot_data()
    assert isinstance(ax, plt.Axes)
    plt.close()


def mock_fit(self, X, y, **kwargs):
    self.idata.add_groups(
        {
            "posterior": self.idata.prior,
        }
    )

    combined_data = pd.concat([X, y.rename(self.output_var)], axis=1)

    if "fit_data" in self.idata:
        del self.idata.fit_data

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The group fit_data is not defined in the InferenceData scheme",
        )
        self.idata.add_groups(fit_data=combined_data.to_xarray())  # type: ignore

    return self


@pytest.fixture(scope="module")
def fit_model(module_mocker, saturated_data):
    model = MVITS(existing_sales=["competitor", "own"], market_saturated=True)

    module_mocker.patch(
        "pymc_marketing.cc.mv_its.MVITS.fit",
        mock_fit,
    )

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
    assert isinstance(ax, plt.Axes)
    plt.close()


@pytest.fixture(scope="module")
def unsaturated_data_bad():
    return generate_unsaturated_data(**scenario_unsaturated_bad)


@pytest.fixture(scope="module")
def unsaturated_data_good():
    return generate_unsaturated_data(**scenario_unsaturated_good)


@pytest.fixture(scope="module")
def unsaturated_model_bad(module_mocker, unsaturated_data_bad):
    model = MVITS(existing_sales=["competitor", "own"], market_saturated=False)

    module_mocker.patch(
        "pymc_marketing.cc.mv_its.MVITS.fit",
        mock_fit,
    )

    model.sample(
        unsaturated_data_bad.loc[:, ["competitor", "own"]],
        unsaturated_data_bad["new"],
        random_seed=rng,
        sample_prior_predictive_kwargs={"samples": 10},
    )
    return model


@pytest.fixture(scope="module")
def unsaturated_model_good(module_mocker, unsaturated_data_good):
    model = MVITS(existing_sales=["competitor", "own"], market_saturated=False)

    module_mocker.patch(
        "pymc_marketing.cc.mv_its.MVITS.fit",
        mock_fit,
    )

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
    assert isinstance(ax, plt.Axes)
    plt.close()


def test_save_load(fit_model, saturated_data) -> None:
    test_file = "test-mvits.nc"
    fit_model.save(test_file)

    loaded = MVITS.load(test_file)

    assert loaded.model_config == fit_model.model_config
    assert loaded.existing_sales == fit_model.existing_sales
    assert loaded.market_saturated == fit_model.market_saturated
    assert loaded.X.columns.name is None
    pd.testing.assert_frame_equal(loaded.X, fit_model.X, check_names=False)
    assert loaded.y.name == fit_model.output_var
    pd.testing.assert_series_equal(loaded.y.rename("new"), saturated_data["new"])
