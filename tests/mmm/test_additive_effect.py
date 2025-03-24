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
import pandas as pd
import pymc as pm
import pytest

from pymc_marketing.mmm.additive_effect import FourierEffect, LinearTrendEffect
from pymc_marketing.mmm.fourier import MonthlyFourier, WeeklyFourier, YearlyFourier
from pymc_marketing.mmm.linear_trend import LinearTrend


@pytest.fixture(scope="function")
def create_mock_mmm():
    class MMM:
        pass

    def func(dims, model):
        mmm = MMM()

        mmm.dims = dims
        mmm.model = model

        return mmm

    return func


@pytest.fixture(scope="function")
def fourier_model() -> pm.Model:
    dates = pd.date_range("2025-01-01", periods=52, freq="W-MON")

    coords = {"date": dates}
    return pm.Model(coords=coords)


@pytest.mark.parametrize(
    "fourier",
    [
        WeeklyFourier(n_order=10, prefix="weekly"),
        MonthlyFourier(n_order=10, prefix="monthly"),
        YearlyFourier(n_order=10, prefix="yearly"),
    ],
    ids=["weekly", "monthly", "yearly"],
)
def test_fourier_effect(create_mock_mmm, fourier_model, fourier) -> None:
    effect = FourierEffect(fourier)

    mmm = create_mock_mmm(dims=(), model=fourier_model)

    with mmm.model:
        effect.create_data(mmm)

    assert set(mmm.model.named_vars) == set([f"{fourier.prefix}_day"])
    assert set(mmm.model.coords) == {"date"}

    with mmm.model:
        effect.create_effect(mmm)

    assert set(mmm.model.named_vars) == set(
        [f"{fourier.prefix}_day", f"{fourier.prefix}_beta", f"{fourier.prefix}_effect"]
    )
    assert set(mmm.model.coords) == {"date", fourier.prefix}


@pytest.fixture(scope="function")
def linear_trend_model() -> pm.Model:
    dates = pd.date_range("2025-01-01", periods=52, freq="W-MON")

    coords = {"date": dates}
    return pm.Model(coords=coords)


def test_linear_trend_effect(create_mock_mmm, linear_trend_model) -> None:
    prefix = "linear_trend"
    effect = LinearTrendEffect(LinearTrend(), prefix=prefix)

    mmm = create_mock_mmm(dims=(), model=linear_trend_model)

    with mmm.model:
        effect.create_data(mmm)

    assert set(mmm.model.named_vars) == {f"{prefix}_t"}
    assert set(mmm.model.coords) == {"date"}

    with mmm.model:
        effect.create_effect(mmm)

    assert set(mmm.model.named_vars) == {"delta", f"{prefix}_effect", f"{prefix}_t"}
    assert set(mmm.model.coords) == {"date", "changepoint"}
