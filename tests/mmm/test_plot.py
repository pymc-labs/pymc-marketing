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
import pytest
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from pymc_marketing.mmm.multidimensional import MMM

from pymc_marketing.mmm.plot import MMMPlotSuite


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
            df.stack("country", future_stack=True),
            y.stack("country", future_stack=True),
        ],
        axis=1,
    ).reset_index()


@pytest.fixture
def fit_mmm_with_channel_original_scale(df, mmm, mock_pymc_sample):
    X = df.drop(columns=["y"])
    y = df["y"]

    mmm.build_model(X, y)
    mmm.add_original_scale_contribution_variable(
        var=[
            "channel_contribution",
        ]
    )

    mmm.fit(X, y)

    return mmm


@pytest.fixture
def fit_mmm_without_channel_original_scale(df, mmm, mock_pymc_sample):
    X = df.drop(columns=["y"])
    y = df["y"]

    mmm.fit(X, y)

    return mmm


def test_saturation_curves_scatter_original_scale(fit_mmm_with_channel_original_scale):
    fig, ax = fit_mmm_with_channel_original_scale.plot.saturation_curves_scatter(
        original_scale=True
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    assert all(isinstance(a, Axes) for a in ax.flat)


def test_saturation_curves_scatter_original_scale_fails_if_no_deterministic(
    fit_mmm_without_channel_original_scale,
):
    with pytest.raises(ValueError):
        fit_mmm_without_channel_original_scale.plot.saturation_curves_scatter(
            original_scale=True
        )


def test_contributions_over_time(fit_mmm_with_channel_original_scale):
    fig, ax = fit_mmm_with_channel_original_scale.plot.contributions_over_time(
        var=["channel_contribution"],
        hdi_prob=0.95,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    assert all(isinstance(a, Axes) for a in ax.flat)


def test_posterior_predictive(fit_mmm_with_channel_original_scale, df):
    fit_mmm_with_channel_original_scale.sample_posterior_predictive(
        df.drop(columns=["y"])
    )
    fig, ax = fit_mmm_with_channel_original_scale.plot.posterior_predictive(
        hdi_prob=0.95,
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    assert all(isinstance(a, Axes) for a in ax.flat)


@pytest.fixture(scope="module")
def mock_idata() -> az.InferenceData:
    seed = sum(map(ord, "Fake posterior"))
    rng = np.random.default_rng(seed)
    normal = rng.normal

    dates = pd.date_range("2025-01-01", periods=52, freq="W-MON")
    return az.InferenceData(
        posterior=xr.Dataset(
            {
                "intercept": xr.DataArray(
                    normal(size=(4, 100, 52, 3)),
                    dims=("chain", "draw", "date", "country"),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(100),
                        "date": dates,
                        "country": ["A", "B", "C"],
                    },
                ),
                "linear_trend": xr.DataArray(
                    normal(size=(4, 100, 52)),
                    dims=("chain", "draw", "date"),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(100),
                        "date": dates,
                    },
                ),
            }
        )
    )


@pytest.fixture(scope="module")
def mock_suite(mock_idata):
    """Fixture to create a mock MMMPlotSuite with a mocked posterior."""
    return MMMPlotSuite(idata=mock_idata)


def test_contributions_over_time_expand_dims(mock_suite: MMMPlotSuite):
    fig, ax = mock_suite.contributions_over_time(
        var=[
            "intercept",
            "linear_trend",
        ]
    )

    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    assert all(isinstance(a, Axes) for a in ax.flat)
