import numpy as np
import pytest

from pymc_marketing.mmm.utils import generate_fourier_modes


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
