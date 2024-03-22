import numpy as np
import pandas as pd
import pymc as pm
import pytest

from pymc_marketing.clv.models import BetaGeoModel, ParetoNBDModel
from tests.clv.utils import set_model_fit


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="also run slow tests"
    )
    parser.addoption(
        "--only-slow", action="store_true", default=False, help="only run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not need to skip any tests
        return

    elif config.getoption("--only-slow"):
        # --only-slow given in cli: need to skip non-slow tests
        skip_fast = pytest.mark.skip(reason="Fast test")
        for item in items:
            if "slow" not in item.keywords:
                item.add_marker(skip_fast)

    else:
        # Default: skip slow tests
        skip_slow = pytest.mark.skip(reason="Slow test, use --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(scope="module")
def cdnow_trans() -> pd.DataFrame:
    """
    Load CDNOW_sample transaction data into a Pandas dataframe.

    Data source: https://www.brucehardie.com/datasets/
    """
    return pd.read_csv("datasets/cdnow_transactions.csv")


@pytest.fixture(scope="module")
def test_summary_data() -> pd.DataFrame:
    rng = np.random.default_rng(14)
    df = pd.read_csv("tests/clv/datasets/test_summary_data.csv", index_col=0)
    df["monetary_value"] = rng.lognormal(size=(len(df)))
    df["customer_id"] = df.index
    return df


@pytest.fixture(scope="module")
def fitted_bg(test_summary_data) -> BetaGeoModel:
    rng = np.random.default_rng(13)
    data = pd.DataFrame(
        {
            "customer_id": test_summary_data.index,
            "frequency": test_summary_data["frequency"],
            "recency": test_summary_data["recency"],
            "T": test_summary_data["T"],
        }
    )
    model_config = {
        # Narrow Gaussian centered at MLE params from lifetimes BetaGeoFitter
        "a_prior": {"dist": "DiracDelta", "kwargs": {"c": 1.85034151}},
        "alpha_prior": {"dist": "DiracDelta", "kwargs": {"c": 1.86428187}},
        "b_prior": {"dist": "DiracDelta", "kwargs": {"c": 3.18105431}},
        "r_prior": {"dist": "DiracDelta", "kwargs": {"c": 0.16385072}},
    }
    model = BetaGeoModel(
        data=data,
        model_config=model_config,
    )
    model.build_model()
    fake_fit = pm.sample_prior_predictive(
        samples=50, model=model.model, random_seed=rng
    ).prior
    set_model_fit(model, fake_fit)

    return model


@pytest.fixture(scope="module")
def fitted_pnbd(test_summary_data) -> ParetoNBDModel:
    rng = np.random.default_rng(45)

    model_config = {
        # Narrow Gaussian centered at MLE params from lifetimes ParetoNBDFitter
        "r_prior": {"dist": "DiracDelta", "kwargs": {"c": 0.5534}},
        "alpha_prior": {"dist": "DiracDelta", "kwargs": {"c": 10.5802}},
        "s_prior": {"dist": "DiracDelta", "kwargs": {"c": 0.6061}},
        "beta_prior": {"dist": "DiracDelta", "kwargs": {"c": 11.6562}},
    }
    pnbd_model = ParetoNBDModel(
        data=test_summary_data,
        model_config=model_config,
    )
    pnbd_model.build_model()

    # Mock an idata object for tests requiring a fitted model
    # TODO: This is quite slow. Check similar fixtures in the model tests to speed this up.
    fake_fit = pm.sample_prior_predictive(
        samples=50, model=pnbd_model.model, random_seed=rng
    ).prior
    set_model_fit(pnbd_model, fake_fit)

    return pnbd_model
