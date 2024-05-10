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
from arviz import InferenceData
from xarray import Dataset

from pymc_marketing.clv.models import CLVModel


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
    return pd.read_csv("data/cdnow_transactions.csv")


@pytest.fixture(scope="module")
def test_summary_data() -> pd.DataFrame:
    rng = np.random.default_rng(14)
    df = pd.read_csv("tests/clv/datasets/test_summary_data.csv", index_col=0)
    df["monetary_value"] = rng.lognormal(size=(len(df)))
    df["customer_id"] = df.index
    return df


def set_model_fit(model: CLVModel, fit: InferenceData | Dataset):
    if isinstance(fit, InferenceData):
        assert "posterior" in fit.groups()
    else:
        fit = InferenceData(posterior=fit)
    if model.model is None:
        model.build_model()
    model.idata = fit
    model.idata.add_groups(fit_data=model.data.to_xarray())
    model.set_idata_attrs(fit)
