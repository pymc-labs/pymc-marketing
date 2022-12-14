import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.tensor.var import TensorVariable

from pymc_marketing.mmm.transformers import (
    batched_convolution,
    delayed_adstock,
    geometric_adstock,
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


@pytest.fixture(
    scope="module", params=["ndarray", "TensorConstant", "TensorVariable"], ids=str
)
def convolution_inputs(request):
    x_val = np.ones((3, 4, 5))
    w_val = np.ones((2))
    if request.param == "ndarray":
        return x_val, w_val, None, None
    elif request.param == "TensorConstant":
        return pt.as_tensor_variable(x_val), pt.as_tensor_variable(w_val), None, None
    elif request.param == "TensorVariable":
        return (
            pt.dtensor3("x"),
            pt.specify_shape(pt.dvector("w"), w_val.shape),
            x_val,
            w_val,
        )


@pytest.fixture(scope="module", params=[0, 1, -1])
def convolution_axis(request):
    return request.param


def test_batched_convolution(convolution_inputs, convolution_axis):
    x, w, x_val, w_val = convolution_inputs
    y = batched_convolution(x, w, convolution_axis)
    if x_val is None:
        y_val = y.eval()
        expected_shape = getattr(x, "value", x).shape
    else:
        y_val = pytensor.function([x, w], y)(x_val, w_val)
        expected_shape = x_val.shape
    assert y_val.shape == expected_shape
    y_val = np.moveaxis(y_val, convolution_axis, 0)
    x_val = np.moveaxis(
        x_val if x_val is not None else getattr(x, "value", x), convolution_axis, 0
    )
    assert np.allclose(y_val[0], x_val[0])
    assert np.allclose(y_val[1:], x_val[1:] + x_val[:-1])


def test_batched_convolution_broadcasting():
    x_val = np.random.default_rng(42).normal(size=(3, 1, 5))
    x = pt.as_tensor_variable(x_val)
    w = pt.as_tensor_variable(np.ones((1, 1, 4, 2)))
    y = batched_convolution(x, w, axis=-1).eval()
    assert y.shape == (1, 3, 4, 5)
    assert np.allclose(y[..., 0], x_val[..., 0])
    assert np.allclose(y[..., 1:], x_val[..., 1:] + x_val[..., :-1])


class TestsAdstockTransformers:
    def test_geometric_adstock_x_zero(self):
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
    def test_geometric_adstock_good_alpha(self, x, alpha, l_max):
        y = geometric_adstock(x=x, alpha=alpha, l_max=l_max)
        y_np = y.eval()
        assert y_np[0] == x[0]
        assert y_np[1] == x[1] + alpha * x[0]
        assert y_np[2] == x[2] + alpha * x[1] + (alpha**2) * x[0]

    def test_delayed_adstock_output_type(self):
        x = np.ones(shape=(100))
        y = delayed_adstock(x=x, alpha=0.5, theta=6, l_max=7)
        assert isinstance(y, TensorVariable)
        assert isinstance(y.eval(), np.ndarray)

    def test_delayed_adstock_x_zero(self):
        x = np.zeros(shape=(100))
        y = delayed_adstock(x=x, alpha=0.2, theta=2, l_max=4)
        np.testing.assert_array_equal(x=x, y=y.eval())

    def test_geometric_adstock_vectorized(self, dummy_design_matrix):
        x = dummy_design_matrix.copy()
        x_tensor = pt.as_tensor_variable(x)
        alpha = [0.9, 0.33, 0.5, 0.1, 0.0]
        alpha_tensor = pt.as_tensor_variable(alpha)
        y_tensor = geometric_adstock(x=x_tensor, alpha=alpha_tensor, l_max=12, axis=0)
        y = y_tensor.eval()

        y_tensors = [
            geometric_adstock(x=x[:, i], alpha=alpha[i], l_max=12)
            for i in range(x.shape[1])
        ]
        ys = np.concatenate([y_t.eval()[..., None] for y_t in y_tensors], axis=1)
        assert y.shape == x.shape
        np.testing.assert_almost_equal(actual=y, desired=ys, decimal=12)

    def test_delayed_adstock_vectorized(self, dummy_design_matrix):
        x = dummy_design_matrix
        x_tensor = pt.as_tensor_variable(x)
        alpha = [0.9, 0.33, 0.5, 0.1, 0.0]
        alpha_tensor = pt.as_tensor_variable(alpha)
        theta = [0, 1, 2, 3, 4]
        theta_tensor = pt.as_tensor_variable(theta)
        y_tensor = delayed_adstock(
            x=x_tensor, alpha=alpha_tensor, theta=theta_tensor, l_max=12, axis=0
        )
        y = y_tensor.eval()

        y_tensors = [
            delayed_adstock(x=x[:, i], alpha=alpha[i], theta=theta[i], l_max=12)
            for i in range(x.shape[1])
        ]
        ys = np.concatenate([y_t.eval()[..., None] for y_t in y_tensors], axis=1)
        assert y.shape == x.shape
        np.testing.assert_almost_equal(actual=y, desired=ys, decimal=12)


class TestSaturationTransformers:
    def test_logistic_saturation_lam_zero(self):
        x = np.ones(shape=(100))
        y = logistic_saturation(x=x, lam=0.0)
        np.testing.assert_array_equal(x=np.zeros(shape=(100)), y=y.eval())

    def test_logistic_saturation_lam_one(self):
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
    def test_logistic_saturation_lam_large(self, x):
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
    def test_logistic_saturation_min_max_value(self, x, lam):
        y = logistic_saturation(x=x, lam=lam)
        y_eval = y.eval()
        assert y_eval.max() <= 1
        assert y_eval.min() >= 0

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
    def test_tanh_saturation_range(self, x, b, c):
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
    def test_tanh_saturation_inverse(self, x, b, c):
        y = tanh_saturation(x=x, b=b, c=c)
        y_inv = (b * c) * pt.arctanh(y / b)
        np.testing.assert_array_almost_equal(x=x, y=y_inv.eval(), decimal=6)


class TestTransformersComposition:
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
    def test_logistic_saturation_geometric_adstock_composition(self, x, alpha, lam):
        y1 = logistic_saturation(x=x, lam=lam)
        z1 = geometric_adstock(x=y1, alpha=alpha, l_max=1)
        y2 = geometric_adstock(x=x, alpha=alpha, l_max=1)
        z2 = logistic_saturation(x=y2, lam=lam)
        z2_eval = z2.eval()
        assert isinstance(z1, TensorVariable)
        assert isinstance(z1.eval(), np.ndarray)
        assert isinstance(z2, TensorVariable)
        assert isinstance(z2_eval, np.ndarray)
        assert z2_eval.max() <= 1
        assert z2_eval.min() >= 0

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
    def test_logistic_saturation_delayed_adstock_composition(
        self, x, alpha, lam, theta, l_max
    ):
        y1 = logistic_saturation(x=x, lam=lam)
        z1 = delayed_adstock(x=y1, alpha=alpha, theta=theta, l_max=l_max)
        y2 = delayed_adstock(x=x, alpha=alpha, theta=theta, l_max=l_max)
        z2 = logistic_saturation(x=y2, lam=lam)
        z2_eval = z2.eval()
        assert isinstance(z1, TensorVariable)
        assert isinstance(z1.eval(), np.ndarray)
        assert isinstance(z2, TensorVariable)
        assert isinstance(z2_eval, np.ndarray)
        assert z2_eval.max() <= 1
        assert z2_eval.min() >= 0

    def test_geometric_adstock_vectorized_logistic_saturation(
        self, dummy_design_matrix
    ):
        x = dummy_design_matrix.copy()
        x_tensor = pt.as_tensor_variable(x)
        alpha = [0.9, 0.33, 0.5, 0.1, 0.0]
        alpha_tensor = pt.as_tensor_variable(alpha)
        lam = [0.5, 1.0, 2.0, 3.0, 4.0]
        lam_tensor = pt.as_tensor_variable(lam)
        y_tensor = geometric_adstock(x=x_tensor, alpha=alpha_tensor, l_max=12, axis=0)
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

    def test_delayed_adstock_vectorized_logistic_saturation(self, dummy_design_matrix):
        x = dummy_design_matrix.copy()
        x_tensor = pt.as_tensor_variable(x)
        alpha = [0.9, 0.33, 0.5, 0.1, 0.0]
        alpha_tensor = pt.as_tensor_variable(alpha)
        theta = [0, 1, 2, 3, 4]
        theta_tensor = pt.as_tensor_variable(theta)
        lam = [0.5, 1.0, 2.0, 3.0, 4.0]
        lam_tensor = pt.as_tensor_variable(lam)
        y_tensor = delayed_adstock(
            x=x_tensor, alpha=alpha_tensor, theta=theta_tensor, l_max=12, axis=0
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
