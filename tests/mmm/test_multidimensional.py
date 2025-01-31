#   Copyright 2025 - 2025 The PyMC Labs Developers
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
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM
from tests.conftest import mock_sample


@pytest.fixture
def mmm():
    return MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        dims=("country",),
        target_column="y",
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )


@pytest.fixture
def df() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=3, freq="W-MON").rename("date")
    df = pd.DataFrame(
        {
            ("A", "C1"): [1, 2, 3],
            ("B", "C1"): [4, 5, 6],
            ("A", "C2"): [7, 8, 9],
            ("B", "C2"): [10, 11, 12],
        },
        index=dates,
    )
    df.columns.names = ["country", "channel"]

    y = pd.DataFrame(
        {
            ("A", "y"): [1, 2, 3],
            ("B", "y"): [4, 5, 6],
        },
        index=dates,
    )
    y.columns.names = ["country", "channel"]

    return pd.concat(
        [
            df.stack("country"),
            y.stack("country"),
        ],
        axis=1,
    ).reset_index()


@pytest.fixture
def mock_pymc_sample() -> None:
    original_sample = pm.sample
    pm.sample = mock_sample

    yield

    pm.sample = original_sample


@pytest.fixture
def fit_mmm(df, mmm, mock_pymc_sample):
    X = df.drop(columns=["y"])
    y = df["y"]

    mmm.fit(X, y)

    return mmm


def test_fit(fit_mmm):
    assert isinstance(fit_mmm.posterior, xr.Dataset)
    assert isinstance(fit_mmm.idata.fit_data, xr.Dataset)


def test_sample_prior_predictive(mmm: MMM, df: pd.DataFrame):
    X = df.drop(columns=["y"])
    y = df["y"]
    mmm.sample_prior_predictive(X, y)

    assert isinstance(mmm.prior, xr.Dataset)
    assert isinstance(mmm.prior_predictive, xr.Dataset)


def test_save_load(fit_mmm: MMM):
    file = "test.nc"
    fit_mmm.save(file)

    loaded = MMM.load(file)
    assert isinstance(loaded, MMM)
