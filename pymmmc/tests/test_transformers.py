import aesara.tensor as at
import numpy as np
import pytest
from aesara.tensor.var import TensorVariable

from pymmmc.transformers import (
    delayed_adstock,
    delayed_adstock_vectorized,
    geometric_adstock,
    geometric_adstock_scan,
    geometric_adstock_vectorized,
    logistic_saturation,
    tanh_saturation,
)


@pytest.fixture
def dummy_design_matrix():
    return np.concatenate(
        (
            np.ones(shape=(100, 1)),
            0.5 * np.ones(shape=(100, 1)),
            np.zeros(shape=(100, 1)),
            np.linspace(start=0.0, stop=1.0, num=(100))[..., None],
            np.linspace(start=0.0, stop=3.0, num=(100))[..., None],
        ),
        axis=1,
    )


def test_geometric_adstock_output_type():
    x = np.ones(shape=(100))
    y = geometric_adstock(x=x, alpha=0.5)
    assert isinstance(y, TensorVariable)
    assert isinstance(y.eval(), np.ndarray)


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
    assert y_np[2] == x[2] + alpha * x[1] + (alpha**2) * x[0]


def test_delayed_adstock_output_type():
    x = np.ones(shape=(100))
    y = delayed_adstock(x=x, alpha=0.5, theta=6, l_max=7)
    assert isinstance(y, TensorVariable)
    assert isinstance(y.eval(), np.ndarray)


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


def test_geometric_adstock_vactorized(dummy_design_matrix):
    x = dummy_design_matrix.copy()
    x_tensor = at.as_tensor_variable(x)
    alpha = [0.9, 0.33, 0.5, 0.1, 0.0]
    alpha_tensor = at.as_tensor_variable(alpha)
    y_tensor = geometric_adstock_vectorized(x=x_tensor, alpha=alpha_tensor, l_max=12)
    y = y_tensor.eval()

    y_tensors = [
        geometric_adstock(x=x[:, i], alpha=alpha[i], l_max=12)
        for i in range(x.shape[1])
    ]
    ys = np.concatenate([y_t.eval()[..., None] for y_t in y_tensors], axis=1)
    assert y.shape == x.shape
    np.testing.assert_almost_equal(actual=y, desired=ys, decimal=12)


def test_delayed_adstock_vactorized(dummy_design_matrix):
    x = dummy_design_matrix
    x_tensor = at.as_tensor_variable(x)
    alpha = [0.9, 0.33, 0.5, 0.1, 0.0]
    alpha_tensor = at.as_tensor_variable(alpha)
    theta = [0, 1, 2, 3, 4]
    theta_tensor = at.as_tensor_variable(theta)
    y_tensor = delayed_adstock_vectorized(
        x=x_tensor, alpha=alpha_tensor, theta=theta_tensor, l_max=12
    )
    y = y_tensor.eval()

    y_tensors = [
        delayed_adstock(x=x[:, i], alpha=alpha[i], theta=theta[i], l_max=12)
        for i in range(x.shape[1])
    ]
    ys = np.concatenate([y_t.eval()[..., None] for y_t in y_tensors], axis=1)
    assert y.shape == x.shape
    np.testing.assert_almost_equal(actual=y, desired=ys, decimal=12)


def test_geometric_adstock_vactorized_logistic_saturation(dummy_design_matrix):
    x = dummy_design_matrix.copy()
    x_tensor = at.as_tensor_variable(x)
    alpha = [0.9, 0.33, 0.5, 0.1, 0.0]
    alpha_tensor = at.as_tensor_variable(alpha)
    lam = [0.5, 1.0, 2.0, 3.0, 4.0]
    lam_tensor = at.as_tensor_variable(lam)
    y_tensor = geometric_adstock_vectorized(x=x_tensor, alpha=alpha_tensor, l_max=12)
    z_tensor = logistic_saturation(x=y_tensor, lam=lam_tensor)
    z = z_tensor.eval()

    y_tensors = [
        geometric_adstock(x=x[:, i], alpha=alpha[i], l_max=12)
        for i in range(x.shape[1])
    ]
    z_tensors = [
        logistic_saturation(x=y_t, lam=lam[i]) for i, y_t in enumerate(y_tensors)
    ]
    zs = np.concatenate([z_t.eval()[..., None] for z_t in z_tensors], axis=1)
    assert zs.shape == x.shape
    np.testing.assert_almost_equal(actual=z, desired=zs, decimal=12)


def test_delayed_adstock_vactorized_logistic_saturation(dummy_design_matrix):
    x = dummy_design_matrix.copy()
    x_tensor = at.as_tensor_variable(x)
    alpha = [0.9, 0.33, 0.5, 0.1, 0.0]
    alpha_tensor = at.as_tensor_variable(alpha)
    theta = [0, 1, 2, 3, 4]
    theta_tensor = at.as_tensor_variable(theta)
    lam = [0.5, 1.0, 2.0, 3.0, 4.0]
    lam_tensor = at.as_tensor_variable(lam)
    y_tensor = delayed_adstock_vectorized(
        x=x_tensor, alpha=alpha_tensor, theta=theta_tensor, l_max=12
    )
    z_tensor = logistic_saturation(x=y_tensor, lam=lam_tensor)
    z = z_tensor.eval()

    y_tensors = [
        delayed_adstock(x=x[:, i], alpha=alpha[i], theta=theta[i], l_max=12)
        for i in range(x.shape[1])
    ]
    z_tensors = [
        logistic_saturation(x=y_t, lam=lam[i]) for i, y_t in enumerate(y_tensors)
    ]
    zs = np.concatenate([z_t.eval()[..., None] for z_t in z_tensors], axis=1)
    assert zs.shape == x.shape
    np.testing.assert_almost_equal(actual=z, desired=zs, decimal=12)


# TODO: Fix case when l_max == x.shape[0]
@pytest.mark.parametrize(
    "x, alpha, l_max",
    [
        (at.as_tensor_variable(np.ones(shape=(100))), at.as_tensor_variable(0.3), 10),
        # (at.as_tensor_variable(np.ones(shape=(100))), tt.as_tensor_variable(0.7), 100), # noqa: E501
        (at.as_tensor_variable(np.zeros(shape=(100))), at.as_tensor_variable(0.2), 5),
        (at.as_tensor_variable(np.ones(shape=(100))), at.as_tensor_variable(0.5), 7),
        (
            at.as_tensor_variable(np.linspace(start=0.0, stop=1.0, num=50)),
            at.as_tensor_variable(0.8),
            3,
        ),
        # (
        #     at.as_tensor_variable(np.linspace(start=0.0, stop=1.0, num=50)),
        #     at.as_tensor_variable(0.8),
        #     50,
        # ),
    ],
)
def test_geometric_adstock_scan_scalar(x, alpha, l_max):
    y_tensor = geometric_adstock(x=x, alpha=alpha, l_max=l_max)
    y_tensor_scan = geometric_adstock_scan(x=x, alpha=alpha, l_max=l_max)
    y = y_tensor.eval()
    y_scan = y_tensor_scan.eval()
    assert y_scan.shape == x.eval().shape
    assert y_scan.shape == y.shape
    np.testing.assert_almost_equal(actual=y, desired=y_scan, decimal=12)


# TODO: Fix case when l_max == x.shape[0]
@pytest.mark.parametrize(
    "x, alpha, l_max",
    [
        (at.as_tensor_variable(np.ones(shape=(100))), at.as_tensor_variable(0.3), 10),
        # (tt.as_tensor_variable(np.ones(shape=(100))), tt.as_tensor_variable(0.7), 100), # noqa: E501
        (at.as_tensor_variable(np.zeros(shape=(100))), at.as_tensor_variable(0.2), 5),
        (at.as_tensor_variable(np.ones(shape=(100))), at.as_tensor_variable(0.5), 7),
        (
            at.as_tensor_variable(np.linspace(start=0.0, stop=1.0, num=50)),
            at.as_tensor_variable(0.8),
            3,
        ),
        # (
        #     tt.as_tensor_variable(np.linspace(start=0.0, stop=1.0, num=50)),
        #     tt.as_tensor_variable(0.8),
        #     50,
        # ),
    ],
)
def test_geometric_adstock_scan_scalar_normalized(x, alpha, l_max):
    y_tensor = geometric_adstock(x=x, alpha=alpha, l_max=l_max, normalize=True)
    y_tensor_scan = geometric_adstock_scan(
        x=x, alpha=alpha, l_max=l_max, normalize=True
    )
    y = y_tensor.eval()
    y_scan = y_tensor_scan.eval()
    assert y_scan.shape == x.eval().shape
    assert y_scan.shape == y.shape
    np.testing.assert_almost_equal(actual=y, desired=y_scan, decimal=6)


def test_geometric_adstock_scan_vector(dummy_design_matrix):
    x = dummy_design_matrix.copy()
    x_tensor = at.as_tensor_variable(x)
    alpha = [0.9, 0.33, 0.5, 0.1, 0.0]
    alpha_tensor = at.as_tensor_variable(alpha)
    y_tensor = geometric_adstock_scan(x=x_tensor, alpha=alpha_tensor, l_max=12)
    y = y_tensor.eval()

    y_tensors = [
        geometric_adstock(x=x[:, i], alpha=alpha[i], l_max=12)
        for i in range(x.shape[1])
    ]
    ys = np.concatenate([y_t.eval()[..., None] for y_t in y_tensors], axis=1)
    assert y.shape == x.shape
    np.testing.assert_almost_equal(actual=y, desired=ys, decimal=12)


def test_geometric_adstock_scan_vector_normalized(dummy_design_matrix):
    x = dummy_design_matrix.copy()
    x_tensor = at.as_tensor_variable(x)
    alpha = [0.9, 0.33, 0.5, 0.1, 0.0]
    alpha_tensor = at.as_tensor_variable(alpha)
    y_tensor = geometric_adstock_scan(
        x=x_tensor, alpha=alpha_tensor, l_max=12, normalize=True
    )
    y = y_tensor.eval()

    y_tensors = [
        geometric_adstock(x=x[:, i], alpha=alpha[i], l_max=12, normalize=True)
        for i in range(x.shape[1])
    ]
    ys = np.concatenate([y_t.eval()[..., None] for y_t in y_tensors], axis=1)
    assert y.shape == x.shape
    np.testing.assert_almost_equal(actual=y, desired=ys, decimal=12)
