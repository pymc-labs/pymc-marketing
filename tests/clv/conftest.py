#   Copyright 2022 - 2026 The PyMC Labs Developers
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
import pymc as pm
import pytest
import xarray as xr
from pymc.testing import mock_sample
from pymc_extras.prior import Prior
from xarray import Dataset

from pymc_marketing.clv.models import (
    BetaGeoModel,
    CLVModel,
    ModifiedBetaGeoModel,
    ParetoNBDModel,
)


def pytest_collection_modifyitems(items):
    """Escalate warnings to errors for the CLV suite (deprecation regression guard).

    ``pytestmark`` in a conftest.py is a no-op, so the marker is applied to each
    collected test here instead. pytensor/numba warnings are OS-dependent and
    outside the scope of this guard.
    """
    for item in items:
        # Prepend the error filter so per-test ``filterwarnings`` ignore
        # markers (and the appended ignores below) keep precedence over it.
        item.add_marker(pytest.mark.filterwarnings("error"), append=False)
        item.add_marker(pytest.mark.filterwarnings("ignore::Warning:pytensor.*"))
        item.add_marker(
            pytest.mark.filterwarnings("ignore::numba.core.errors.NumbaWarning")
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


def sample_prior_predictive_ignoring_potentials(**kwargs):
    """Sample prior predictive, ignoring the expected pymc Potentials warning.

    CLV likelihoods are registered as ``pm.Potential``, so pymc warns that
    potentials are ignored during prior predictive sampling. That is expected
    for these mock fits and must not trip the suite-wide ``error`` filter.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The effect of Potentials on other parameters is ignored",
        )
        return pm.sample_prior_predictive(**kwargs)


def set_model_fit(model: CLVModel, fit: xr.DataTree | Dataset):
    if isinstance(fit, xr.DataTree):
        if "/posterior" in fit.groups:
            pass
        elif "/prior" in fit.groups:
            ds = fit["/prior"].to_dataset()
            fit = xr.DataTree.from_dict({"/posterior": ds})
        else:
            raise ValueError(
                f"Cannot fit model. Expected /posterior or /prior group, "
                f"got {fit.groups}"
            )
    else:
        fit = xr.DataTree.from_dict({"/posterior": fit})
    if not hasattr(model, "model"):
        model.build_model()
    model.idata = fit

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The group fit_data is not defined in the InferenceData scheme",
        )
        model.idata["/fit_data"] = model.data.to_xarray()
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
        posterior_ds = xr.Dataset(
            {
                param: (
                    ["chain", "draw"],
                    rng.normal(value, 1e-3, size=(chains, draws)),
                )
                for param, value in params.items()
            }
        )
        model.idata = xr.DataTree.from_dict({"/posterior": posterior_ds})
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
        model_config=model_config,
    )
    model.build_model(data=test_summary_data)
    fake_fit = sample_prior_predictive_ignoring_potentials(
        draws=50, model=model.model, random_seed=rng
    )
    # posterior group required to pass L80 assert check
    fake_fit["/posterior"] = fake_fit["/prior"].to_dataset()
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
        model_config=model_config,
    )
    model.build_model(data=test_summary_data)
    fake_fit = sample_prior_predictive_ignoring_potentials(
        draws=50, model=model.model, random_seed=rng
    )
    # posterior group required to pass L80 assert check
    fake_fit["/posterior"] = fake_fit["/prior"].to_dataset()
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
        model_config=model_config,
    )
    pnbd_model.build_model(data=test_summary_data)

    # Mock an idata object for tests requiring a fitted model
    # TODO: This is quite slow. Check similar fixtures in the model tests to speed this up.
    fake_fit = sample_prior_predictive_ignoring_potentials(
        draws=50,
        model=pnbd_model.model,
        random_seed=rng,
    )
    # posterior group required to pass L80 assert check
    fake_fit["/posterior"] = fake_fit["/prior"].to_dataset()
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
