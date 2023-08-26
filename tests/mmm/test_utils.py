import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.utils import (
    estimate_menten_parameters,
    extended_sigmoid,
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
    "L, k, s, expected",
    [
        (1, 2, 3, 0.6),
        (0, 2, 3, 0),
    ],
)
def test_michaelis_menten(L, k, s, expected):
    assert michaelis_menten(L, k, s) == expected


@pytest.mark.parametrize(
    "channel, original_dataframe, contributions, expected_output",
    [
        # Test case 1: single data point
        (
            "channel1",
            pd.DataFrame({"channel1": [1]}),
            xr.DataArray(
                np.array([1]),
                dims=["quantile", "channel"],
                coords={"quantile": [0.5], "channel": ["channel1"]},
            ),
            [1, 0.001],
        ),
        # Test case 2: multiple data points, same values
        (
            "channel1",
            pd.DataFrame({"channel1": [1, 1, 1, 1]}),
            xr.DataArray(
                np.array([1, 1, 1, 1]),
                dims=["quantile", "channel"],
                coords={"quantile": [0.5], "channel": ["channel1"]},
            ),
            [1, 0.001],
        ),
    ],
)
def test_estimate_menten_parameters(
    channel, original_dataframe, contributions, expected_output
):
    assert np.allclose(
        estimate_menten_parameters(channel, original_dataframe, contributions),
        expected_output,
    )


@pytest.mark.parametrize(
    "x, alpha, lam, expected",
    [
        # Test case 1: Test with x=0, alpha=1, lam=1, expected output is 0.5
        (0, 1, 1, 0.5),
        # Test case 2: Test with x=1, alpha=1, lam=1, expected output is approximately 0.731
        (1, 1, 1, 0.731),
        # Test case 3: Test with x=-1, alpha=1, lam=1, expected output is approximately 0.269
        (-1, 1, 1, 0.269),
        # Test case 4: Test with x=0, alpha=2, lam=1, expected output is 1
        (0, 2, 1, 1),
        # Test case 5: Test with x=0, alpha=1, lam=2, expected output is 0.5
        (0, 1, 2, 0.5),
    ],
)
def test_extended_sigmoid(x, alpha, lam, expected):
    assert np.isclose(extended_sigmoid(x, alpha, lam), expected, atol=0.01)


@pytest.mark.parametrize(
    "x, alpha, lam",
    [
        # Test case 1: Test with x=0, alpha=0, lam=0, expected output is a ValueError
        (0, 0, 0),
        # Test case 2: Test with x=1, alpha=-1, lam=1, expected output is a ValueError
        (1, -1, 1),
        # Test case 3: Test with x=-1, alpha=1, lam=-1, expected output is a ValueError
        (-1, 1, -1),
    ],
)
def test_extended_sigmoid_value_errors(x, alpha, lam):
    with pytest.raises(ValueError):
        extended_sigmoid(x, alpha, lam)
