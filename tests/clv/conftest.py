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
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
from arviz import InferenceData
from pymc.testing import mock_sample
from pymc_extras.prior import Prior
from xarray import Dataset

from pymc_marketing.clv.models import (
    BetaGeoModel,
    CLVModel,
    ModifiedBetaGeoModel,
    ParetoNBDModel,
)


@pytest.fixture(scope="module")
def cdnow_trans() -> pd.DataFrame:
    """Load CDNOW_ample transaction data into a Pandas dataframe.

    Data source: https://www.brucehardie.com/datasets/
    """
    return pd.read_csv("data/cdnow_transactions.csv")


@pytest.fixture(scope="module")
def test_summary_data() -> pd.DataFrame:
    df = pd.read_csv("data/clv_quickstart.csv")
    df["customer_id"] = df.index
    df["future_spend"] = df["monetary_value"]
    return df


def set_model_fit(model: CLVModel, fit: InferenceData | Dataset):
    if isinstance(fit, InferenceData):
        assert "posterior" in fit.groups()
    else:
        fit = InferenceData(posterior=fit)
    if not hasattr(model, "model"):
        model.build_model()
    model.idata = fit

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The group fit_data is not defined in the InferenceData scheme",
        )
        model.idata.add_groups(fit_data=model.data.to_xarray())
    model.set_idata_attrs(fit)


def set_idata(model):
    """Part of basic fit method for CLVModel."""
    model.set_idata_attrs(model.idata)
    if model.data is not None:
        model._add_fit_data_group(model.data)


def create_mock_fit(params: dict[str, float]):
    """This is a mock of the fit method for the CLVModel.

    It create a fake InferenceData object that is centered around the given parameters.

    """

    def mock_fit(model, chains, draws, rng):
        model.idata = az.from_dict(
            {
                param: rng.normal(value, 1e-3, size=(chains, draws))
                for param, value in params.items()
            }
        )
        set_idata(model)

    return mock_fit


def mock_fit_MAP(self, *args, **kwargs):
    draws = 1
    chains = 1
    idata = mock_sample(*args, **kwargs, chains=chains, draws=draws, model=self.model)

    return idata.sel(chain=[0], draw=[0])


# TODO: This fixture is used in the plotting and utils test modules.
#       Consider creating a MockModel class to replace this and other fitted model fixtures.
@pytest.fixture(scope="module")
def fitted_bg(test_summary_data) -> BetaGeoModel:
    rng = np.random.default_rng(13)

    model_config = {
        # Narrow Gaussian centered at MLE params from lifetimes BetaGeoFitter
        "a": Prior("DiracDelta", c=1.85034151),
        "alpha": Prior("DiracDelta", c=1.86428187),
        "b": Prior("DiracDelta", c=3.18105431),
        "r": Prior("DiracDelta", c=0.16385072),
    }
    model = BetaGeoModel(
        data=test_summary_data,
        model_config=model_config,
    )
    model.build_model()
    fake_fit = pm.sample_prior_predictive(draws=50, model=model.model, random_seed=rng)
    # posterior group required to pass L80 assert check
    fake_fit.add_groups(posterior=fake_fit.prior)
    set_model_fit(model, fake_fit)

    return model


# TODO: This fixture is used in the plotting and utils test modules.
#       Consider creating a MockModel class to replace this and other fitted model fixtures.
@pytest.fixture(scope="module")
def fitted_mbg(test_summary_data) -> ModifiedBetaGeoModel:
    rng = np.random.default_rng(13)

    model_config = {
        # Narrow Gaussian centered at MLE params from lifetimes BetaGeoFitter
        "a": Prior("DiracDelta", c=1.85034151),
        "alpha": Prior("DiracDelta", c=1.86428187),
        "b": Prior("DiracDelta", c=3.18105431),
        "r": Prior("DiracDelta", c=0.16385072),
    }
    model = ModifiedBetaGeoModel(
        data=test_summary_data,
        model_config=model_config,
    )
    model.build_model()
    fake_fit = pm.sample_prior_predictive(draws=50, model=model.model, random_seed=rng)
    # posterior group required to pass L80 assert check
    fake_fit.add_groups(posterior=fake_fit.prior)
    set_model_fit(model, fake_fit)

    return model


# TODO: This fixture is used in the plotting and utils test modules.
#       Consider creating a MockModel class to replace this and other fitted model fixtures.
@pytest.fixture(scope="module")
def fitted_pnbd(test_summary_data) -> ParetoNBDModel:
    rng = np.random.default_rng(45)

    model_config = {
        # Narrow Gaussian centered at MLE params from lifetimes ParetoNBDFitter
        "r": Prior("DiracDelta", c=0.560),
        "alpha": Prior("DiracDelta", c=10.591),
        "s": Prior("DiracDelta", c=0.550),
        "beta": Prior("DiracDelta", c=9.756),
    }
    pnbd_model = ParetoNBDModel(
        data=test_summary_data,
        model_config=model_config,
    )
    pnbd_model.build_model()

    # Mock an idata object for tests requiring a fitted model
    # TODO: This is quite slow. Check similar fixtures in the model tests to speed this up.
    fake_fit = pm.sample_prior_predictive(
        draws=50,
        model=pnbd_model.model,
        random_seed=rng,
    )
    # posterior group required to pass L80 assert check
    fake_fit.add_groups(posterior=fake_fit.prior)
    set_model_fit(pnbd_model, fake_fit)

    return pnbd_model


@pytest.fixture(params=["bg_model", "mbg_model", "pnbd_model"])
def fitted_model(request, fitted_bg, fitted_mbg, fitted_pnbd):
    fitted_models = {
        "bg_model": fitted_bg,
        "mbg_model": fitted_mbg,
        "pnbd_model": fitted_pnbd,
    }
    return fitted_models[request.param]
