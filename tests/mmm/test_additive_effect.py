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
import numpy as np
import pandas as pd
import pymc as pm
import pytest

from pymc_marketing.mmm.additive_effect import (
    FourierEffect,
    LinearTrendEffect,
)
from pymc_marketing.mmm.fourier import MonthlyFourier, WeeklyFourier, YearlyFourier
from pymc_marketing.mmm.linear_trend import LinearTrend
from pymc_marketing.prior import Prior


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
def dates() -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=52, freq="W-MON", name="date")


@pytest.fixture(scope="function")
def new_dates(dates) -> pd.DatetimeIndex:
    last_date = dates.max()

    return pd.date_range(
        last_date + pd.Timedelta(days=7),
        periods=26,
        freq="W-MON",
        name="date",
    )


def set_new_model_dates(dates):
    # Just changing the coordinates of the model
    model = pm.modelcontext(None)
    model.set_dim("date", len(dates), coord_values=dates)


@pytest.fixture(scope="function")
def create_fourier_model(dates):
    def create_model(coords) -> pm.Model:
        coords = coords | {"date": dates}
        return pm.Model(coords=coords)

    return create_model


@pytest.mark.parametrize(
    "fourier",
    [
        WeeklyFourier(n_order=10, prefix="weekly"),
        MonthlyFourier(n_order=10, prefix="monthly"),
        YearlyFourier(n_order=10, prefix="yearly"),
    ],
    ids=["weekly", "monthly", "yearly"],
)
@pytest.mark.parametrize(
    "dims, coords",
    [
        ((), {}),
        (("geo",), {"geo": ["A", "B"]}),
    ],
    ids=["no_dims", "with_dims"],
)
def test_fourier_effect(
    create_mock_mmm,
    new_dates,
    create_fourier_model,
    fourier,
    dims,
    coords,
) -> None:
    effect = FourierEffect(fourier)

    mmm = create_mock_mmm(
        dims=dims,
        model=create_fourier_model(coords=coords),
    )

    with mmm.model:
        effect.create_data(mmm)

    assert set(mmm.model.named_vars) == set([f"{fourier.prefix}_day"])
    assert set(mmm.model.coords) == {"date", *dims}

    with mmm.model:
        # Should just be broadcastable with target.
        # Not necessarily the same shape
        created_variable = effect.create_effect(mmm)

    assert created_variable.ndim == len(mmm.dims) + 1

    assert set(mmm.model.named_vars) == set(
        [
            f"{fourier.prefix}_day",
            f"{fourier.prefix}_beta",
            f"{fourier.prefix}_effect",
        ]
    )
    assert set(mmm.model.coords) == {"date", *dims, fourier.prefix}

    with mmm.model:
        idata = pm.sample_prior_predictive()
        set_new_model_dates(new_dates)
        effect.set_data(mmm, mmm.model, None)

        idata.extend(
            pm.sample_posterior_predictive(
                idata.prior,
                var_names=[f"{fourier.prefix}_effect"],
            )
        )

    effect_predictions = idata.posterior_predictive[f"{fourier.prefix}_effect"]
    np.testing.assert_allclose(effect_predictions.notnull().mean().item(), 1.0)
    pd.testing.assert_index_equal(effect_predictions.date.to_index(), new_dates)


@pytest.mark.parametrize(
    "prior_dims",
    [
        (),
        ("weekly",),
        ("weekly", "geo"),
        ("geo", "weekly"),
    ],
    ids=["no-dims", "exclude", "include", "include_reverse"],
)
def test_fourier_effect_multidimensional(
    create_mock_mmm,
    create_fourier_model,
    prior_dims,
) -> None:
    mmm = create_mock_mmm(
        dims=("geo",),
        model=create_fourier_model(coords={"geo": ["A", "B"]}),
    )

    prefix = "weekly"
    prior = Prior("Laplace", mu=0, b=0.1, dims=prior_dims)
    fourier = WeeklyFourier(n_order=10, prefix=prefix, prior=prior)
    fourier_effect = FourierEffect(fourier)

    with mmm.model:
        fourier_effect.create_data(mmm)
        effect = fourier_effect.create_effect(mmm)
        pm.sample_prior_predictive()

    assert effect.ndim == 2


@pytest.fixture(scope="function")
def linear_trend_model(dates) -> pm.Model:
    coords = {"date": dates}
    return pm.Model(coords=coords)


@pytest.mark.parametrize(
    "mmm_dims, priors, linear_trend_dims, deterministic_dims",
    [
        pytest.param((), {}, (), ("date",), id="scalar"),
        pytest.param(
            ("geo", "product"),
            {},
            ("geo", "product"),
            ("date", None, None),
            id="2d",
        ),
        pytest.param(
            ("geo", "product"),
            {"delta": Prior("Normal", dims=("geo", "changepoint"))},
            ("geo", "product"),
            ("date", None, None),
            id="missing-product-dim-in-delta",
        ),
    ],
)
def test_linear_trend_effect(
    create_mock_mmm,
    new_dates,
    linear_trend_model,
    mmm_dims,
    priors,
    linear_trend_dims,
    deterministic_dims,
) -> None:
    prefix = "linear_trend"
    effect = LinearTrendEffect(
        LinearTrend(priors=priors, dims=linear_trend_dims),
        prefix=prefix,
    )

    mmm = create_mock_mmm(dims=mmm_dims, model=linear_trend_model)

    mmm.model.add_coords({dim: ["dummy1", "dummy2"] for dim in linear_trend_dims})

    with mmm.model:
        effect.create_data(mmm)

    assert set(mmm.model.named_vars) == {f"{prefix}_t"}
    assert set(mmm.model.coords) == {"date"}.union(linear_trend_dims)
    assert effect.linear_trend_first_date == mmm.model.coords["date"][0]

    with mmm.model:
        pm.Deterministic(
            "effect",
            effect.create_effect(mmm),
            dims=deterministic_dims,
        )

    assert set(mmm.model.named_vars) == {
        "delta",
        "effect",
        f"{prefix}_effect_contribution",
        f"{prefix}_t",
    }
    assert set(mmm.model.coords) == {"date", "changepoint"}.union(linear_trend_dims)

    with mmm.model:
        idata = pm.sample_prior_predictive()
        set_new_model_dates(new_dates)
        effect.set_data(mmm, mmm.model, None)

        idata.extend(
            pm.sample_posterior_predictive(
                idata.prior,
                var_names=["effect"],
            )
        )

    effect_predictions = idata.posterior_predictive.effect
    np.testing.assert_allclose(effect_predictions.notnull().mean().item(), 1.0)
    pd.testing.assert_index_equal(effect_predictions.date.to_index(), new_dates)
