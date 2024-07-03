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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.plot import (
    plot_hdi,
    plot_samples,
    random_samples,
    selections,
)


@pytest.mark.parametrize(
    "coords, expected",
    [
        ({}, [{}]),
        ({"channel": [1, 2, 3]}, [{"channel": 1}, {"channel": 2}, {"channel": 3}]),
        (
            {"channel": [1, 2], "country": ["A", "B"]},
            [
                {"channel": 1, "country": "A"},
                {"channel": 1, "country": "B"},
                {"channel": 2, "country": "A"},
                {"channel": 2, "country": "B"},
            ],
        ),
    ],
)
def test_selections(coords, expected) -> None:
    assert list(selections(coords)) == expected


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [
            range(1),
            range(10),
        ]
    )
    return pd.DataFrame(
        np.random.randn(10, 1),
        index=index,
        columns=["y"],
    )


def test_random_samples(sample_frame) -> None:
    rng = np.random.default_rng(0)
    n = 5
    idx = random_samples(rng, n=n, n_chains=1, n_draws=10)

    assert len(idx) == n
    assert idx == [
        (0, 4),
        (0, 7),
        (0, 2),
        (0, 3),
        (0, 5),
    ]
    df_sub = sample_frame.loc[idx, :]
    assert len(df_sub) == n


@pytest.fixture
def mock_curve() -> xr.DataArray:
    coords = {
        "chain": np.arange(1),
        "draw": np.arange(15),
        "geo": np.arange(5),
        "day": np.arange(31),
    }
    return xr.DataArray(
        np.ones((1, 15, 5, 31)),
        coords=coords,
    )


def test_plot_samples(mock_curve) -> None:
    fig, axes = plot_samples(mock_curve, non_grid_names={"chain", "draw", "day"})

    assert axes.size == 5
    assert isinstance(fig, plt.Figure)


def test_plot_hdi(mock_curve) -> None:
    fig, axes = plot_hdi(mock_curve, non_grid_names={"day"})

    assert axes.size == 5
    assert isinstance(fig, plt.Figure)
