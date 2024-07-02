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
import pytest
import xarray as xr

from pymc_marketing.mmm.fourier import YearlyFourier
from pymc_marketing.prior import Prior


def test_prior_without_dims() -> None:
    prior = Prior("Normal")
    yearly = YearlyFourier(n_order=2, prior=prior)

    assert yearly.prior.dims == (yearly.prefix,)
    assert prior.dims == ()


def test_prior_doesnt_have_prefix() -> None:
    prior = Prior("Normal", dims="hierarchy")
    with pytest.raises(ValueError, match="Prior distribution must have"):
        YearlyFourier(n_order=2, prior=prior)


def test_nodes() -> None:
    yearly = YearlyFourier(n_order=2)

    assert yearly.nodes == ["sin_1", "sin_2", "cos_1", "cos_2"]


def test_sample_prior() -> None:
    n_order = 2
    yearly = YearlyFourier(n_order=n_order)
    prior = yearly.sample_prior(samples=10)

    assert prior.sizes == {
        "chain": 1,
        "draw": 10,
        yearly.prefix: n_order * 2,
    }


def test_sample_full_period() -> None:
    n_order = 2
    yearly = YearlyFourier(n_order=n_order)
    prior = yearly.sample_prior(samples=10)
    curve = yearly.sample_full_period(prior)

    assert curve.sizes == {
        "chain": 1,
        "draw": 10,
        "day": 367,
    }


def create_mock_variable(coords):
    shape = [len(values) for values in coords.values()]

    return xr.DataArray(
        np.ones(shape),
        coords=coords,
    )


@pytest.fixture
def mock_parameters() -> xr.Dataset:
    n_chains = 1
    n_draws = 250

    return xr.Dataset(
        {
            "fourier_beta": create_mock_variable(
                coords={
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "fourier": ["sin_1", "sin_2", "cos_1", "cos_2"],
                }
            ).rename("fourier_beta"),
            "another_larger_variable": create_mock_variable(
                coords={
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "additional_dim": np.arange(10),
                }
            ).rename("another_larger_variable"),
        },
    )


def test_sample_full_period_additional_dims(mock_parameters) -> None:
    yearly = YearlyFourier(n_order=2)
    curve = yearly.sample_full_period(mock_parameters)

    assert curve.sizes == {
        "chain": 1,
        "draw": 250,
        "day": 367,
    }


def test_additional_dimension() -> None:
    prior = Prior("Normal", dims=("fourier", "additional_dim", "yet_another_dim"))
    yearly = YearlyFourier(n_order=2, prior=prior)

    coords = {
        "additional_dim": range(2),
        "yet_another_dim": range(3),
    }
    prior = yearly.sample_prior(samples=10, coords=coords)
    curve = yearly.sample_full_period(prior)

    assert curve.sizes == {
        "chain": 1,
        "draw": 10,
        "additional_dim": 2,
        "yet_another_dim": 3,
        "day": 367,
    }


def test_plot_full_period() -> None:
    prior = Prior("Normal", dims=("fourier", "additional_dim"))
    yearly = YearlyFourier(n_order=2, prior=prior)

    coords = {"additional_dim": range(4)}
    prior = yearly.sample_prior(samples=10, coords=coords)
    curve = yearly.sample_full_period(prior)

    subplot_kwargs = {"ncols": 2}
    fig, axes = yearly.plot_full_period(curve, subplot_kwargs=subplot_kwargs)

    assert isinstance(fig, plt.Figure)
    assert axes.shape == (2, 2)
