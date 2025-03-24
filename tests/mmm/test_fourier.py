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
    WeeklyFourier,
    YearlyFourier,
    generate_fourier_modes,
)
from pymc_marketing.prior import Prior


@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_prior_without_dims(seasonality) -> None:
    prior = Prior("Normal")
    periodicity = seasonality(n_order=2, prior=prior)

    assert periodicity.prior.dims == (periodicity.prefix,)
    assert prior.dims == ()


@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_prior_doesnt_have_prefix(seasonality) -> None:
    prior = Prior("Normal", dims="hierarchy")
    with pytest.raises(ValueError, match="Prior distribution must have"):
        seasonality(n_order=2, prior=prior)


@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_nodes(seasonality) -> None:
    periodicity = seasonality(n_order=2)

    assert periodicity.nodes == ["sin_1", "sin_2", "cos_1", "cos_2"]


@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_sample_prior(seasonality) -> None:
    n_order = 2
    periodicity = seasonality(n_order=n_order)
    prior = periodicity.sample_prior(samples=10)

    assert prior.sizes == {
        "chain": 1,
        "draw": 10,
        periodicity.prefix: n_order * 2,
    }


@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_sample_curve(seasonality) -> None:
    n_order = 2
    periodicity = seasonality(n_order=n_order)
    prior = periodicity.sample_prior(samples=10)
    curve = periodicity.sample_curve(prior)

    assert curve.sizes == {
        "chain": 1,
        "draw": 10,
        "day": np.ceil(periodicity.days_in_period) + 1,
    }


@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_sample_curve_use_dates(seasonality) -> None:
    n_order = 2
    periodicity = seasonality(n_order=n_order)
    prior = periodicity.sample_prior(samples=10)
    curve = periodicity.sample_curve(prior, use_dates=True)

    assert curve.sizes == {
        "chain": 1,
        "draw": 10,
        "date": np.ceil(periodicity.days_in_period) + 1,
    }


@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_sample_curve_same_size(seasonality) -> None:
    n_order = 2
    periodicity = seasonality(n_order=n_order)
    prior = periodicity.sample_prior(samples=10)
    curve_without_dates = periodicity.sample_curve(prior, use_dates=False)
    curve_with_dates = periodicity.sample_curve(prior, use_dates=True)

    assert curve_without_dates.shape == curve_with_dates.shape


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


@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_sample_curve_additional_dims(mock_parameters, seasonality) -> None:
    periodicity = seasonality(n_order=2)
    curve = periodicity.sample_curve(mock_parameters)

    assert curve.sizes == {
        "chain": 1,
        "draw": 250,
        "day": np.ceil(periodicity.days_in_period) + 1,
    }


@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_additional_dimension(seasonality) -> None:
    prior = Prior("Normal", dims=("fourier", "additional_dim", "yet_another_dim"))
    periodicity = YearlyFourier(n_order=2, prior=prior)

    coords = {
        "additional_dim": range(2),
        "yet_another_dim": range(3),
    }
    prior = periodicity.sample_prior(samples=10, coords=coords)
    curve = periodicity.sample_curve(prior)

    assert curve.sizes == {
        "chain": 1,
        "draw": 10,
        "additional_dim": 2,
        "yet_another_dim": 3,
        "day": np.ceil(periodicity.days_in_period) + 1,
    }


@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_plot_curve(seasonality) -> None:
    prior = Prior("Normal", dims=("fourier", "additional_dim"))
    periodicity = seasonality(n_order=2, prior=prior)

    coords = {"additional_dim": range(4)}
    prior = periodicity.sample_prior(samples=10, coords=coords)
    curve = periodicity.sample_curve(prior)

    subplot_kwargs = {"ncols": 2}
    fig, axes = periodicity.plot_curve(curve, subplot_kwargs=subplot_kwargs)

    assert isinstance(fig, plt.Figure)
    assert axes.shape == (2, 2)


@pytest.mark.parametrize("n_order", [0, -1, -100])
@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_bad_negative_order(n_order, seasonality) -> None:
    with pytest.raises(
        ValueError,
        match=f"1 validation error for {seasonality.__name__}\\nn_order\\n  Input should be greater than 0",
    ):
        seasonality(n_order=n_order)


@pytest.mark.parametrize(
    argnames="n_order",
    argvalues=[2.5, 100.001, "m", None],
    ids=["neg_float", "neg_float_2", "str", "None"],
)
@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_bad_non_integer_order(n_order, seasonality) -> None:
    with pytest.raises(
        ValueError,
        match=f"1 validation error for {seasonality.__name__}\nn_order\n  Input should be a valid integer",
    ):
        seasonality(n_order=n_order)


@pytest.mark.parametrize(
    "periods, n_order, expected_shape",
    [
        (np.linspace(start=0.0, stop=1.0, num=50), 10, (50, 10 * 2)),
        (np.linspace(start=-1.0, stop=1.0, num=70), 9, (70, 9 * 2)),
        (np.ones(shape=1), 1, (1, 1 * 2)),
    ],
)
@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_fourier_modes_shape(periods, n_order, expected_shape, seasonality) -> None:
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


@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_apply_result_callback(seasonality) -> None:
    n_order = 3
    fourier = seasonality(n_order=n_order)

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


@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_error_with_prefix_and_variable_name(seasonality) -> None:
    name = "variable_name"
    with pytest.raises(ValueError, match="Variable name cannot"):
        seasonality(n_order=2, prefix=name, variable_name=name)


@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_change_name(seasonality) -> None:
    variable_name = "variable_name"
    fourier = seasonality(n_order=2, variable_name=variable_name)
    prior = fourier.sample_prior(samples=10)
    assert variable_name in prior


@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_serialization_to_json(seasonality) -> None:
    fourier = seasonality(n_order=2)
    fourier.model_dump_json()


@pytest.fixture
def yearly_fourier() -> YearlyFourier:
    prior = Prior("Laplace", mu=0, b=1, dims="fourier")
    return YearlyFourier(n_order=2, prior=prior)


@pytest.fixture
def monthly_fourier() -> MonthlyFourier:
    prior = Prior("Laplace", mu=0, b=1, dims="fourier")
    return MonthlyFourier(n_order=2, prior=prior)


@pytest.fixture
def weekly_fourier() -> WeeklyFourier:
    prior = Prior("Laplace", mu=0, b=1, dims="fourier")
    return WeeklyFourier(n_order=2, prior=prior)


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


def test_get_default_start_date_none_weekly(weekly_fourier: WeeklyFourier):
    now = datetime.datetime.now()
    expected_start_date = datetime.datetime.fromisocalendar(
        year=now.year, week=now.isocalendar().week, day=1
    )
    actual_start_date = weekly_fourier.get_default_start_date()
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


def test_get_default_start_date_str_weekly(weekly_fourier: WeeklyFourier):
    start_date_str = "2023-06-15"
    actual_start_date = weekly_fourier.get_default_start_date(start_date=start_date_str)
    assert actual_start_date == start_date_str


def test_get_default_start_date_datetime_weekly(weekly_fourier: WeeklyFourier):
    start_date_dt = datetime.datetime(2023, 7, 1)
    actual_start_date = weekly_fourier.get_default_start_date(start_date=start_date_dt)
    assert actual_start_date == start_date_dt


def test_get_default_start_date_invalid_type_weekly(weekly_fourier: WeeklyFourier):
    invalid_start_date = [2023, 1, 1]
    with pytest.raises(TypeError) as exc_info:
        weekly_fourier.get_default_start_date(start_date=invalid_start_date)
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


@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_fourier_arbitrary_prior(seasonality) -> None:
    prior = ArbitraryCode(dims=("fourier",))
    fourier = seasonality(n_order=4, prior=prior)

    x = np.arange(10)
    with pm.Model():
        y = fourier.apply(x)

    assert y.eval().shape == (10,)


@pytest.mark.parametrize(
    argnames="seasonality",
    argvalues=[YearlyFourier, MonthlyFourier, WeeklyFourier],
    ids=[
        "yearly",
        "monthly",
        "weekly",
    ],
)
def test_fourier_dims_modified(seasonality) -> None:
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
        ("WeeklyFourier", WeeklyFourier, 7),
    ],
    ids=["YearlyFourier", "MonthlyFourier", "WeeklyFourier"],
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
        ("WeeklyFourier", WeeklyFourier),
    ],
    ids=["YearlyFourier", "MonthlyFourier", "WeeklyFourier"],
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
