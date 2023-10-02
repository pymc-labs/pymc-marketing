import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.utils import (
    compute_sigmoid_second_derivative,
    estimate_menten_parameters,
    estimate_sigmoid_parameters,
    extense_sigmoid,
    find_sigmoid_inflection_point,
    generate_fourier_modes,
    michaelis_menten,
)


@pytest.mark.parametrize(
    "periods, n_order, expected_shape",
    [
        (np.linspace(start=0.0, stop=1.0, num=50), 10, (50, 10 * 2)),
        (np.linspace(start=-1.0, stop=1.0, num=70), 9, (70, 9 * 2)),
        (np.ones(shape=1), 1, (1, 1 * 2)),
    ],
)
def test_fourier_modes_shape(periods, n_order, expected_shape):
    fourier_modes_df = generate_fourier_modes(periods=periods, n_order=n_order)
    assert fourier_modes_df.shape == expected_shape


@pytest.mark.parametrize(
    "periods, n_order",
    [
        (np.linspace(start=0.0, stop=1.0, num=50), 10),
        (np.linspace(start=-1.0, stop=1.0, num=70), 9),
        (np.ones(shape=1), 1),
    ],
)
def test_fourier_modes_range(periods, n_order):
    fourier_modes_df = generate_fourier_modes(periods=periods, n_order=n_order)
    assert fourier_modes_df.min().min() >= -1.0
    assert fourier_modes_df.max().max() <= 1.0


@pytest.mark.parametrize(
    "periods, n_order",
    [
        (np.linspace(start=-1.0, stop=1.0, num=100), 10),
        (np.linspace(start=-10.0, stop=2.0, num=170), 60),
        (np.linspace(start=-15, stop=5.0, num=160), 20),
    ],
)
def test_fourier_modes_frequency_integer_range(periods, n_order):
    fourier_modes_df = generate_fourier_modes(periods=periods, n_order=n_order)
    assert (fourier_modes_df.filter(regex="sin").mean(axis=0) < 1e-10).all()
    assert (fourier_modes_df.filter(regex="cos").iloc[:-1].mean(axis=0) < 1e-10).all()
    assert not fourier_modes_df[fourier_modes_df > 0].empty
    assert not fourier_modes_df[fourier_modes_df < 0].empty
    assert not fourier_modes_df[fourier_modes_df == 0].empty
    assert not fourier_modes_df[fourier_modes_df == 1].empty


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
    fourier_modes_df = generate_fourier_modes(periods=periods, n_order=n_order)
    for i in range(1, n_order + 1):
        norm = (
            fourier_modes_df[f"sin_order_{i}"] ** 2
            + fourier_modes_df[f"cos_order_{i}"] ** 2
        )
        assert abs(norm - 1).all() < 1e-10


@pytest.mark.parametrize("n_order", [0, -1, -100])
def test_bad_order(n_order):
    with pytest.raises(ValueError, match="n_order must be greater than or equal to 1"):
        generate_fourier_modes(
            periods=np.linspace(start=0.0, stop=1.0, num=50), n_order=n_order
        )


@pytest.mark.parametrize(
    "x, alpha, lam, expected",
    [
        (10, 100, 5, 66.67),
        (20, 100, 5, 80),
    ],
)
def test_michaelis_menten(x, alpha, lam, expected):
    assert np.isclose(michaelis_menten(x, alpha, lam), expected, atol=0.01)


@pytest.mark.parametrize(
    "x, alpha, lam, expected",
    [
        (0, 1, 1, 0),
        (1, 1, 1, 0.4621),
    ],
)
def test_extense_sigmoid(x, alpha, lam, expected):
    assert np.isclose(extense_sigmoid(x, alpha, lam), expected, atol=0.01)


@pytest.mark.parametrize(
    "x, alpha, lam",
    [
        (0, 0, 1),
        (1, -1, 1),
        (1, 1, 0),
    ],
)
def test_extense_sigmoid_value_errors(x, alpha, lam):
    with pytest.raises(ValueError):
        extense_sigmoid(x, alpha, lam)


# Test estimate_menten_parameters with valid inputs
@pytest.mark.parametrize(
    "channel,original_dataframe,contributions,expected",
    [
        (
            "channel1",
            pd.DataFrame({"channel1": [0, 1, 2, 3, 4], "channel2": [0, 2, 4, 6, 8]}),
            xr.DataArray(
                np.array([[0, 2.85714286, 5, 6.66666667, 8]]),
                coords=[("channel", ["channel1"]), ("observation", list(range(5)))],
            ),
            [20, 6],
        ),
        # Add more test cases as needed
    ],
)
def test_estimate_menten_parameters_valid(
    channel, original_dataframe, contributions, expected
):
    result = estimate_menten_parameters(channel, original_dataframe, contributions)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


# Test estimate_sigmoid_parameters with valid inputs
@pytest.mark.parametrize(
    "channel,original_dataframe,contributions,expected",
    [
        (
            "channel1",
            pd.DataFrame({"channel1": [0, 1, 2, 3, 4], "channel2": [0, 2, 4, 6, 8]}),
            xr.DataArray(
                np.array([[0, 1, 2, 2.8, 2.95]]),
                coords=[("channel", ["channel1"]), ("observation", list(range(5)))],
            ),
            [3.53, 0.648],
        ),
        # Add more test cases as needed
    ],
)
def test_estimate_sigmoid_parameters_valid(
    channel, original_dataframe, contributions, expected
):
    result = estimate_sigmoid_parameters(channel, original_dataframe, contributions)
    np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-4)


# Test compute_sigmoid_second_derivative with valid inputs
@pytest.mark.parametrize(
    "x,alpha,lam,expected",
    [
        (
            np.array([0, 1, 2, 3, 4]),
            3.53,
            0.648,
            np.array([-0, 0.04411199, -0.00336529, -0.04266177, -0.04798905]),
        ),
        # Add more test cases as needed
    ],
)
def test_compute_sigmoid_second_derivative_valid(x, alpha, lam, expected):
    result = compute_sigmoid_second_derivative(x, alpha, lam)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


# Test find_sigmoid_inflection_point with valid inputs
@pytest.mark.parametrize(
    "alpha,lam,expected",
    [
        (3.53, 0.648, (0.8041700751856726, 0.8994825718533391)),
        # Add more test cases as needed
    ],
)
def test_find_sigmoid_inflection_point_valid(alpha, lam, expected):
    result = find_sigmoid_inflection_point(alpha, lam)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)
