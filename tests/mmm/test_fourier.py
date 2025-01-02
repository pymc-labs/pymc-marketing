#   Copyright 2025 The PyMC Labs Developers
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
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytest
import xarray as xr

from pymc_marketing.deserialize import (
    DESERIALIZERS,
    deserialize,
    register_deserialization,
)
from pymc_marketing.mmm.fourier import (
    FourierBase,
    MonthlyFourier,
    YearlyFourier,
    generate_fourier_modes,
)
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


def test_sample_curve() -> None:
    n_order = 2
    yearly = YearlyFourier(n_order=n_order)
    prior = yearly.sample_prior(samples=10)
    curve = yearly.sample_curve(prior)

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


def test_sample_curve_additional_dims(mock_parameters) -> None:
    yearly = YearlyFourier(n_order=2)
    curve = yearly.sample_curve(mock_parameters)

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
    curve = yearly.sample_curve(prior)

    assert curve.sizes == {
        "chain": 1,
        "draw": 10,
        "additional_dim": 2,
        "yet_another_dim": 3,
        "day": 367,
    }


def test_plot_curve() -> None:
    prior = Prior("Normal", dims=("fourier", "additional_dim"))
    yearly = YearlyFourier(n_order=2, prior=prior)

    coords = {"additional_dim": range(4)}
    prior = yearly.sample_prior(samples=10, coords=coords)
    curve = yearly.sample_curve(prior)

    subplot_kwargs = {"ncols": 2}
    fig, axes = yearly.plot_curve(curve, subplot_kwargs=subplot_kwargs)

    assert isinstance(fig, plt.Figure)
    assert axes.shape == (2, 2)


@pytest.mark.parametrize("n_order", [0, -1, -100])
def test_bad_negative_order(n_order) -> None:
    with pytest.raises(
        ValueError,
        match="1 validation error for YearlyFourier\\nn_order\\n  Input should be greater than 0",
    ):
        YearlyFourier(n_order=n_order)


@pytest.mark.parametrize(
    argnames="n_order",
    argvalues=[2.5, 100.001, "m", None],
    ids=["neg_float", "neg_float_2", "str", "None"],
)
def test_bad_non_integer_order(n_order) -> None:
    with pytest.raises(
        ValueError,
        match="1 validation error for YearlyFourier\nn_order\n  Input should be a valid integer",
    ):
        YearlyFourier(n_order=n_order)


@pytest.mark.parametrize(
    "periods, n_order, expected_shape",
    [
        (np.linspace(start=0.0, stop=1.0, num=50), 10, (50, 10 * 2)),
        (np.linspace(start=-1.0, stop=1.0, num=70), 9, (70, 9 * 2)),
        (np.ones(shape=1), 1, (1, 1 * 2)),
    ],
)
def test_fourier_modes_shape(periods, n_order, expected_shape) -> None:
    result = generate_fourier_modes(periods, n_order)
    assert result.eval().shape == expected_shape


@pytest.mark.parametrize(
    "periods, n_order",
    [
        (np.linspace(start=0.0, stop=1.0, num=50), 10),
        (np.linspace(start=-1.0, stop=1.0, num=70), 9),
        (np.ones(shape=1), 1),
    ],
)
def test_fourier_modes_range(periods, n_order):
    fourier_modes = generate_fourier_modes(periods=periods, n_order=n_order).eval()

    assert fourier_modes.min() >= -1.0
    assert fourier_modes.max() <= 1.0


@pytest.mark.parametrize(
    "periods, n_order",
    [
        (np.linspace(start=-1.0, stop=1.0, num=100), 10),
        (np.linspace(start=-10.0, stop=2.0, num=170), 60),
        (np.linspace(start=-15, stop=5.0, num=160), 20),
    ],
)
def test_fourier_modes_frequency_integer_range(periods, n_order):
    fourier_modes = generate_fourier_modes(periods=periods, n_order=n_order).eval()

    assert (fourier_modes[:, :n_order].mean(axis=0) < 1e-10).all()
    assert (fourier_modes[:-1, n_order:].mean(axis=0) < 1e-10).all()

    assert fourier_modes[fourier_modes > 0].shape
    assert fourier_modes[fourier_modes < 0].shape
    assert fourier_modes[fourier_modes == 0].shape
    assert fourier_modes[fourier_modes == 1].shape


@pytest.mark.parametrize(
    "periods, n_order",
    [
        (np.linspace(start=0.0, stop=1.0, num=100), 10),
        (np.linspace(start=0.0, stop=2.0, num=170), 60),
        (np.linspace(start=0.0, stop=5.0, num=160), 20),
        (np.linspace(start=-9.0, stop=1.0, num=100), 10),
        (np.linspace(start=-80.0, stop=2.0, num=170), 60),
        (np.linspace(start=-100.0, stop=-5.0, num=160), 20),
    ],
)
def test_fourier_modes_pythagoras(periods, n_order):
    fourier_modes = generate_fourier_modes(periods=periods, n_order=n_order).eval()
    norm = fourier_modes[:, :n_order] ** 2 + fourier_modes[:, n_order:] ** 2

    assert (abs(norm - 1) < 1e-10).all()


def test_apply_result_callback() -> None:
    n_order = 3
    fourier = YearlyFourier(n_order=n_order)

    def result_callback(x):
        pm.Deterministic(
            "components",
            x,
            dims=("dayofyear", *fourier.prior.dims),
        )

    dayofyear = np.arange(365)
    coords = {
        "dayofyear": dayofyear,
    }
    with pm.Model(coords=coords) as model:
        fourier.apply(dayofyear, result_callback=result_callback)

    assert "components" in model
    assert model["components"].eval().shape == (365, n_order * 2)


def test_error_with_prefix_and_variable_name() -> None:
    name = "variable_name"
    with pytest.raises(ValueError, match="Variable name cannot"):
        YearlyFourier(n_order=2, prefix=name, variable_name=name)


def test_change_name() -> None:
    variable_name = "variable_name"
    fourier = YearlyFourier(n_order=2, variable_name=variable_name)
    prior = fourier.sample_prior(samples=10)
    assert variable_name in prior


def test_serialization_to_json() -> None:
    fourier = YearlyFourier(n_order=2)
    fourier.model_dump_json()


@pytest.fixture
def yearly_fourier() -> YearlyFourier:
    prior = Prior("Laplace", mu=0, b=1, dims="fourier")
    return YearlyFourier(n_order=2, prior=prior)


@pytest.fixture
def monthly_fourier() -> MonthlyFourier:
    prior = Prior("Laplace", mu=0, b=1, dims="fourier")
    return MonthlyFourier(n_order=2, prior=prior)


def test_get_default_start_date_none_yearly(yearly_fourier: YearlyFourier):
    current_year = datetime.datetime.now().year
    expected_start_date = datetime.datetime(year=current_year, month=1, day=1)
    actual_start_date = yearly_fourier.get_default_start_date()
    assert actual_start_date == expected_start_date


def test_get_default_start_date_none_monthly(monthly_fourier: MonthlyFourier):
    now = datetime.datetime.now()
    expected_start_date = datetime.datetime(year=now.year, month=now.month, day=1)
    actual_start_date = monthly_fourier.get_default_start_date()
    assert actual_start_date == expected_start_date


def test_get_default_start_date_str_yearly(yearly_fourier: YearlyFourier):
    start_date_str = "2023-02-01"
    actual_start_date = yearly_fourier.get_default_start_date(start_date=start_date_str)
    assert actual_start_date == start_date_str


def test_get_default_start_date_datetime_yearly(yearly_fourier: YearlyFourier):
    start_date_dt = datetime.datetime(2023, 3, 1)
    actual_start_date = yearly_fourier.get_default_start_date(start_date=start_date_dt)
    assert actual_start_date == start_date_dt


def test_get_default_start_date_invalid_type_yearly(yearly_fourier: YearlyFourier):
    invalid_start_date = 12345  # Invalid type again
    with pytest.raises(TypeError) as exc_info:
        yearly_fourier.get_default_start_date(start_date=invalid_start_date)
    assert "start_date must be a datetime.datetime object, a string, or None" in str(
        exc_info.value
    )


def test_get_default_start_date_str_monthly(monthly_fourier: MonthlyFourier):
    start_date_str = "2023-06-15"
    actual_start_date = monthly_fourier.get_default_start_date(
        start_date=start_date_str
    )
    assert actual_start_date == start_date_str


def test_get_default_start_date_datetime_monthly(monthly_fourier: MonthlyFourier):
    start_date_dt = datetime.datetime(2023, 7, 1)
    actual_start_date = monthly_fourier.get_default_start_date(start_date=start_date_dt)
    assert actual_start_date == start_date_dt


def test_get_default_start_date_invalid_type_monthly(monthly_fourier: MonthlyFourier):
    invalid_start_date = [2023, 1, 1]
    with pytest.raises(TypeError) as exc_info:
        monthly_fourier.get_default_start_date(start_date=invalid_start_date)
    assert "start_date must be a datetime.datetime object, a string, or None" in str(
        exc_info.value
    )


def test_fourier_base_instantiation():
    with pytest.raises(TypeError) as exc_info:
        FourierBase(
            n_order=2,
            prior=Prior("Laplace", mu=0, b=1, dims="fourier"),
        )
    assert "Can't instantiate abstract class FourierBase" in str(exc_info.value)


class ArbitraryCode:
    def __init__(self, dims: tuple[str, ...]) -> None:
        self.dims = dims

    def create_variable(self, name: str):
        return pm.Normal(name, dims=self.dims)


def test_fourier_arbitrary_prior() -> None:
    prior = ArbitraryCode(dims=("fourier",))
    fourier = YearlyFourier(n_order=4, prior=prior)

    x = np.arange(10)
    with pm.Model():
        y = fourier.apply(x)

    assert y.eval().shape == (10,)


def test_fourier_dims_modified() -> None:
    prior = ArbitraryCode(dims=())
    YearlyFourier(n_order=4, prior=prior)
    assert prior.dims == "fourier"


class SerializableArbitraryCode(ArbitraryCode):
    def to_dict(self):
        return {"dims": self.dims, "msg": "Hello, World!"}


def test_fourier_serializable_arbitrary_prior() -> None:
    prior = SerializableArbitraryCode(dims=("fourier",))
    fourier = YearlyFourier(n_order=4, prior=prior)

    assert fourier.model_dump(mode="json") == {
        "n_order": 4,
        "days_in_period": 365.25,
        "prefix": "fourier",
        "prior": {"dims": ["fourier"], "msg": "Hello, World!"},
        "variable_name": "fourier_beta",
    }


@pytest.mark.parametrize(
    "name, cls, days_in_period",
    [
        ("YearlyFourier", YearlyFourier, 365.25),
        ("MonthlyFourier", MonthlyFourier, 30.4375),
    ],
    ids=["YearlyFourier", "MonthlyFourier"],
)
def test_fourier_to_dict(name, cls, days_in_period) -> None:
    fourier = cls(n_order=4)
    assert fourier.to_dict() == {
        "class": name,
        "data": {
            "n_order": 4,
            "days_in_period": days_in_period,
            "prefix": "fourier",
            "prior": {
                "dist": "Laplace",
                "kwargs": {"b": 1, "mu": 0},
                "dims": ["fourier"],
            },
            "variable_name": "fourier_beta",
        },
    }


@pytest.fixture
def serialization() -> None:
    register_deserialization(
        is_type=lambda data: data.keys() == {"dims", "msg"},
        deserialize=lambda data: SerializableArbitraryCode(dims=data["dims"]),
    )

    yield

    DESERIALIZERS.pop()


@pytest.mark.parametrize(
    "name, cls",
    [
        ("YearlyFourier", YearlyFourier),
        ("MonthlyFourier", MonthlyFourier),
    ],
    ids=["YearlyFourier", "MonthlyFourier"],
)
def test_fourier_deserialization(serialization, name, cls) -> None:
    data = {
        "class": name,
        "data": {
            "n_order": 4,
            "prior": {"dims": ["fourier"], "msg": "Hello, World!"},
        },
    }
    fourier = deserialize(data)

    assert isinstance(fourier, cls)
    assert fourier.n_order == 4
    assert fourier.prefix == "fourier"
    assert fourier.variable_name == "fourier_beta"
    assert isinstance(fourier.prior, SerializableArbitraryCode)
