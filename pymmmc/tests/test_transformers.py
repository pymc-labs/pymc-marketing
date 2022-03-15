import aesara.tensor as at
import numpy as np
import pytest
from aesara.tensor.var import TensorVariable

from pymmmc.transformers import (
    delayed_adstock,
    geometric_adstock,
    logistic_saturation,
    tanh_saturation,
)


def test_geometric_adstock_output_type():
    x = np.ones(shape=(100))
    y = geometric_adstock(x=x, alpha=0.5)
    assert isinstance(y, TensorVariable)
    assert isinstance(y.eval(), np.ndarray)


@pytest.mark.parametrize(
    "x, alpha",
    [
        (np.ones(shape=(100)), 1.0),
        (np.ones(shape=(100)), 0.0),
        (np.zeros(shape=(100)), -0.2),
        (np.ones(shape=(100)), -0.5),
        (np.linspace(start=0.0, stop=1.0, num=50), 10),
        (np.linspace(start=0.0, stop=1.0, num=50), -10),
    ],
)
def test_geometric_adstock_bad_alpha(x, alpha):
    with pytest.raises(ValueError, match=f"alpha must be between 0 and 1. Got {alpha}"):
        geometric_adstock(x=x, alpha=alpha)


def test_geometric_adstock_x_zero():
    x = np.zeros(shape=(100))
    y = geometric_adstock(x=x, alpha=0.2)
    np.testing.assert_array_equal(x=x, y=y.eval())


@pytest.mark.parametrize(
    "x, alpha, l_max",
    [
        (np.ones(shape=(100)), 0.3, 10),
        (np.ones(shape=(100)), 0.7, 100),
        (np.zeros(shape=(100)), 0.2, 5),
        (np.ones(shape=(100)), 0.5, 7),
        (np.linspace(start=0.0, stop=1.0, num=50), 0.8, 3),
        (np.linspace(start=0.0, stop=1.0, num=50), 0.8, 50),
    ],
)
def test_geometric_adstock_good_alpha(x, alpha, l_max):
    y = geometric_adstock(x=x, alpha=alpha, l_max=l_max)
    y_np = y.eval()
    assert y_np[0] == x[0]
    assert y_np[1] == x[1] + alpha * x[0]
    assert y_np[2] == x[2] + alpha * x[1] + (alpha ** 2) * x[0]


def test_delayed_adstock_output_type():
    x = np.ones(shape=(100))
    y = delayed_adstock(x=x, alpha=0.5, theta=6, l_max=7)
    assert isinstance(y, TensorVariable)
    assert isinstance(y.eval(), np.ndarray)


@pytest.mark.parametrize(
    "x, alpha",
    [
        (np.ones(shape=(100)), 1.0),
        (np.ones(shape=(100)), 0.0),
        (np.zeros(shape=(100)), -0.2),
        (np.ones(shape=(100)), -0.5),
        (np.linspace(start=0.0, stop=1.0, num=50), 10),
        (np.linspace(start=0.0, stop=1.0, num=50), -10),
    ],
)
def test_delayed_adstock_bad_alpha(x, alpha):
    with pytest.raises(ValueError, match=f"alpha must be between 0 and 1. Got {alpha}"):
        delayed_adstock(x=x, alpha=alpha)


@pytest.mark.parametrize(
    "x, alpha, theta, l_max",
    [
        (np.ones(shape=(100)), 0.5, -10, 10),
        (np.ones(shape=(100)), 0.5, 10, 10),
        (np.zeros(shape=(100)), 0.2, 8, 5),
        (np.ones(shape=(100)), 0.5, -7, -5),
        (np.linspace(start=0.0, stop=1.0, num=50), 0.3, 3, 1),
    ],
)
def test_delayed_adstock_bad_theta(x, alpha, theta, l_max):
    with pytest.raises(
        ValueError, match=f"theta must be between 0 and l_max - 1. Got {theta}"
    ):
        delayed_adstock(x=x, alpha=alpha, theta=theta, l_max=l_max)


def test_delayed_adstock_x_zero():
    x = np.zeros(shape=(100))
    y = delayed_adstock(x=x, alpha=0.2, theta=2, l_max=4)
    np.testing.assert_array_equal(x=x, y=y.eval())


def test_logistic_saturation_output_type():
    x = np.ones(shape=(100))
    y = logistic_saturation(x=x, lam=1.0)
    assert isinstance(y, TensorVariable)
    assert isinstance(y.eval(), np.ndarray)


def test_logistic_saturation_lam_zero():
    x = np.ones(shape=(100))
    y = logistic_saturation(x=x, lam=0.0)
    np.testing.assert_array_equal(x=np.zeros(shape=(100)), y=y.eval())


def test_logistic_saturation_lam_one():
    x = np.ones(shape=(100))
    y = logistic_saturation(x=x, lam=1.0)
    np.testing.assert_array_equal(
        x=((1 - np.e ** (-1)) / (1 + np.e ** (-1))) * x, y=y.eval()
    )


@pytest.mark.parametrize(
    "x, lam",
    [
        (np.ones(shape=(100)), -1.0),
        (np.ones(shape=(100)), -19.0),
        (np.zeros(shape=(100)), -5.0),
        (np.ones(shape=(100)), -10.0),
        (np.linspace(start=0.0, stop=1.0, num=50), -0.01),
    ],
)
def test_logistic_saturation_bad_lam(x, lam):
    with pytest.raises(ValueError, match=f"lam must be non-negative. Got {lam}"):
        logistic_saturation(x=x, lam=lam)


@pytest.mark.parametrize(
    "x",
    [
        np.ones(shape=(100)),
        np.linspace(start=0.0, stop=1.0, num=50),
        np.linspace(start=200, stop=1000, num=50),
    ],
)
def test_logistic_saturation_lam_large(x):
    y = logistic_saturation(x=x, lam=1e6)
    assert abs(y.eval()).mean() == pytest.approx(1.0, 1e-1)


@pytest.mark.parametrize(
    "x, lam",
    [
        (np.ones(shape=(100)), 30),
        (np.linspace(start=0.0, stop=1.0, num=50), 90),
        (np.linspace(start=200, stop=1000, num=50), 17),
        (np.zeros(shape=(100)), 200),
    ],
)
def test_logistic_saturation_min_max_value(x, lam):
    y = logistic_saturation(x=x, lam=lam)
    assert y.eval().max() <= 1
    assert y.eval().min() >= 0


@pytest.mark.parametrize(
    "x, alpha, lam",
    [
        (np.ones(shape=(100)), 0.5, 1.0),
        (np.ones(shape=(100)), 0.2, 19.0),
        (np.zeros(shape=(100)), 0.6, 5.0),
        (np.ones(shape=(100)), 0.99, 10.0),
        (np.linspace(start=0.0, stop=1.0, num=50), 0.001, 0.01),
    ],
)
def test_logistic_saturation_geometric_adstock_composition(x, alpha, lam):
    y1 = logistic_saturation(x=x, lam=lam)
    z1 = geometric_adstock(x=y1, alpha=alpha, l_max=1)
    y2 = geometric_adstock(x=x, alpha=alpha, l_max=1)
    z2 = logistic_saturation(x=y2, lam=lam)
    assert isinstance(z1, TensorVariable)
    assert isinstance(z1.eval(), np.ndarray)
    assert isinstance(z2, TensorVariable)
    assert isinstance(z2.eval(), np.ndarray)
    assert z2.eval().max() <= 1
    assert z2.eval().min() >= 0


@pytest.mark.parametrize(
    "x, alpha, lam, theta, l_max",
    [
        (np.ones(shape=(100)), 0.5, 1.0, 0, 1),
        (np.ones(shape=(100)), 0.2, 19.0, 1, 2),
        (np.zeros(shape=(100)), 0.6, 5.0, 3, 4),
        (np.ones(shape=(100)), 0.99, 10.0, 0, 5),
        (np.linspace(start=0.0, stop=1.0, num=50), 0.001, 0.01, 4, 5),
    ],
)
def test_logistic_saturation_delayed_adstock_composition(x, alpha, lam, theta, l_max):
    y1 = logistic_saturation(x=x, lam=lam)
    z1 = delayed_adstock(x=y1, alpha=alpha, theta=theta, l_max=l_max)
    y2 = delayed_adstock(x=x, alpha=alpha, theta=theta, l_max=l_max)
    z2 = logistic_saturation(x=y2, lam=lam)
    assert isinstance(z1, TensorVariable)
    assert isinstance(z1.eval(), np.ndarray)
    assert isinstance(z2, TensorVariable)
    assert isinstance(z2.eval(), np.ndarray)
    assert z2.eval().max() <= 1
    assert z2.eval().min() >= 0


@pytest.mark.parametrize(
    "x, b, c",
    [
        (np.ones(shape=(100)), 0.5, 1.0),
        (np.zeros(shape=(100)), 0.6, 5.0),
        (np.linspace(start=0.0, stop=100.0, num=50), 0.001, 0.01),
        (np.linspace(start=-2.0, stop=1.0, num=50), 0.1, 0.01),
        (np.linspace(start=-80.0, stop=1.0, num=50), 1, 1),
    ],
)
def test_tanh_saturation_range(x, b, c):
    assert tanh_saturation(x=x, b=b, c=c).eval().max() <= b
    assert tanh_saturation(x=x, b=b, c=c).eval().min() >= -b


@pytest.mark.parametrize(
    "x, b, c",
    [
        (np.ones(shape=(100)), 0.5, 1.0),
        (np.zeros(shape=(100)), 0.6, 5.0),
        (np.linspace(start=0.0, stop=1.0, num=50), 1, 1),
        (np.linspace(start=-2.0, stop=1.0, num=50), 1, 2),
        (np.linspace(start=-1.0, stop=1.0, num=50), 1, 2),
    ],
)
def test_tanh_saturation_inverse(x, b, c):
    y = tanh_saturation(x=x, b=b, c=c)
    y_inv = (b * c) * at.arctanh(y / b)
    np.testing.assert_array_almost_equal(x=x, y=y_inv.eval(), decimal=6)


@pytest.mark.parametrize(
    "x, b, c",
    [
        (np.ones(shape=(100)), -0.5, 1.0),
        (np.zeros(shape=(100)), -0.6, 5.0),
        (np.linspace(start=0.0, stop=100.0, num=50), -0.001, 0.01),
        (np.linspace(start=-2.0, stop=1.0, num=50), -0.1, 0.01),
        (np.linspace(start=-80.0, stop=1.0, num=50), -1, 1),
    ],
)
def test_tanh_saturation_bad_b(x, b, c):
    with pytest.raises(ValueError, match=f"b must be non-negative. Got {b}"):
        tanh_saturation(x=x, b=b, c=c)


@pytest.mark.parametrize(
    "x, b, c",
    [
        (np.ones(shape=(100)), 0.5, 0),
        (np.zeros(shape=(100)), 0.6, 0.0),
        (np.linspace(start=0.0, stop=100.0, num=50), 0.001, 0.00),
        (np.linspace(start=-2.0, stop=1.0, num=50), 0.1, 0.0),
        (np.linspace(start=-80.0, stop=1.0, num=50), 1, 0),
    ],
)
def test_tanh_saturation_bad_c(x, b, c):
    with pytest.raises(ValueError, match="c must be non-zero."):
        tanh_saturation(x=x, b=b, c=c)
