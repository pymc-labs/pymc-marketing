import numpy as np
import pytest

from pymc_marketing.mmm.utils import calculate_curve, find_elbow, generate_fourier_modes


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


# New tests


@pytest.mark.parametrize(
    "x, y, expected_index",
    [
        (
            np.array([0, 1, 2, 3, 4]),
            np.array([0, 1, 2, 1, 0]),
            2,
        ),  # simple quadratic curve, elbow at index 2
        (
            np.array([0, 1, 2, 3, 4]),
            np.array([0, 1, 3, 1, 0]),
            2,
        ),  # same curve with steeper climb, elbow still at index 2
    ],
)
def test_find_elbow(x, y, expected_index):
    assert find_elbow(x, y) == expected_index


@pytest.mark.parametrize(
    "x, y",
    [
        (
            np.array([0, 1, 2, 3, 4]),
            np.array([0, 1, 2, 1, 0]),
        ),  # simple quadratic curve
        (
            np.array([0, 1, 2, 3, 4]),
            np.array([0, 1, 3, 1, 0]),
        ),  # same curve with steeper climb
    ],
)
def test_calculate_curve(x, y):
    (
        polynomial,
        x_space_actual,
        y_space_actual,
        x_space_projected,
        y_space_projected,
        roots,
    ) = calculate_curve(x, y)

    # Check if polynomial is of degree 2
    assert len(polynomial.coefficients) == 3

    # Check if y_space_actual and y_space_projected are calculated correctly
    assert np.allclose(y_space_actual, polynomial(x_space_actual))
    assert np.allclose(y_space_projected, polynomial(x_space_projected))

    # Check if roots are real and located within x_space_projected
    assert all(isinstance(root, float) for root in roots)
    assert all(
        min(x_space_projected) <= root <= max(x_space_projected) for root in roots
    )
