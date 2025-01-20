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
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import scipy as sp
from pymc.logprob.utils import ParameterValueError
from pytensor.tensor.variable import TensorVariable

from pymc_marketing.mmm.transformers import (
    ConvMode,
    TanhSaturationParameters,
    WeibullType,
    batched_convolution,
    delayed_adstock,
    geometric_adstock,
    hill_function,
    hill_saturation_sigmoid,
    inverse_scaled_logistic_saturation,
    logistic_saturation,
    michaelis_menten,
    tanh_saturation,
    tanh_saturation_baselined,
    weibull_adstock,
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
    w_val = np.ones(2)
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


@pytest.mark.parametrize("mode", [ConvMode.After, ConvMode.Before, ConvMode.Overlap])
def test_batched_convolution(convolution_inputs, convolution_axis, mode):
    x, w, x_val, w_val = convolution_inputs
    y = batched_convolution(x, w, convolution_axis, mode)
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
    if mode == ConvMode.After:
        np.testing.assert_allclose(y_val[0], x_val[0])
        np.testing.assert_allclose(y_val[1:], x_val[1:] + x_val[:-1])
    elif mode == ConvMode.Before:
        np.testing.assert_allclose(y_val[-1], x_val[-1])
        np.testing.assert_allclose(y_val[:-1], x_val[1:] + x_val[:-1])
    elif mode == ConvMode.Overlap:
        np.testing.assert_allclose(y_val[0], x_val[0])
        np.testing.assert_allclose(y_val[1:-1], x_val[1:-1] + x_val[:-2])


def test_batched_convolution_invalid_mode(convolution_inputs, convolution_axis):
    x, w, x_val, w_val = convolution_inputs
    invalid_mode = "InvalidMode"
    with pytest.raises(ValueError):
        batched_convolution(x, w, convolution_axis, invalid_mode)


def test_batched_convolution_broadcasting():
    x_val = np.random.default_rng(42).normal(size=(3, 1, 5))
    x = pt.as_tensor_variable(x_val)
    w = pt.as_tensor_variable(np.ones((1, 1, 4, 2)))
    y = batched_convolution(x, w, axis=-1).eval()
    assert y.shape == (1, 3, 4, 5)
    assert np.allclose(y[..., 0], x_val[..., 0])
    assert np.allclose(y[..., 1:], x_val[..., 1:] + x_val[..., :-1])


class TestsAdstockTransformers:
    @pytest.mark.parametrize(
        argnames="mode",
        argvalues=[ConvMode.After, ConvMode.Before, ConvMode.Overlap],
        ids=["After", "Before", "Overlap"],
    )
    def test_geometric_adstock_x_zero(self, mode):
        x = np.zeros(shape=(100))
        y = geometric_adstock(x=x, alpha=0.2, mode=mode)
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

    @pytest.mark.parametrize(
        "alpha",
        [-0.3, -2, 22.5, 2],
        ids=[
            "less_than_zero_0",
            "less_than_zero_1",
            "greater_than_one_0",
            "greater_than_one_1",
        ],
    )
    def test_geometric_adstock_bad_alpha(self, alpha):
        l_max = 10
        x = np.ones(shape=100)
        y = geometric_adstock(x=x, alpha=alpha, l_max=l_max)
        with pytest.raises(ParameterValueError):
            y.eval()

    @pytest.mark.parametrize(
        argnames="mode",
        argvalues=[ConvMode.After, ConvMode.Before, ConvMode.Overlap],
        ids=["After", "Before", "Overlap"],
    )
    def test_delayed_adstock_output_type(self, mode):
        x = np.ones(shape=(100))
        y = delayed_adstock(x=x, alpha=0.5, theta=6, l_max=7, mode=mode)
        assert isinstance(y, TensorVariable)
        assert isinstance(y.eval(), np.ndarray)

    def test_delayed_adstock_x_zero(self):
        x = np.zeros(shape=(100))
        y = delayed_adstock(x=x, alpha=0.2, theta=2, l_max=4)
        np.testing.assert_array_equal(x=x, y=y.eval())

    @pytest.mark.parametrize(
        argnames="mode",
        argvalues=[ConvMode.After, ConvMode.Before, ConvMode.Overlap],
        ids=["After", "Before", "Overlap"],
    )
    def test_geometric_adstock_vectorized(self, dummy_design_matrix, mode):
        x = dummy_design_matrix.copy()
        x_tensor = pt.as_tensor_variable(x)
        alpha = [0.9, 0.33, 0.5, 0.1, 0.0]
        alpha_tensor = pt.as_tensor_variable(alpha)
        y_tensor = geometric_adstock(
            x=x_tensor, alpha=alpha_tensor, l_max=12, axis=0, mode=mode
        )
        y = y_tensor.eval()

        y_tensors = [
            geometric_adstock(x=x[:, i], alpha=alpha[i], l_max=12, mode=mode)
            for i in range(x.shape[1])
        ]
        ys = np.concatenate([y_t.eval()[..., None] for y_t in y_tensors], axis=1)
        assert y.shape == x.shape
        np.testing.assert_almost_equal(actual=y, desired=ys, decimal=12)

    @pytest.mark.parametrize(
        argnames="mode",
        argvalues=[ConvMode.After, ConvMode.Before, ConvMode.Overlap],
        ids=["After", "Before", "Overlap"],
    )
    def test_delayed_adstock_vectorized(self, dummy_design_matrix, mode):
        x = dummy_design_matrix
        x_tensor = pt.as_tensor_variable(x)
        alpha = [0.9, 0.33, 0.5, 0.1, 0.0]
        alpha_tensor = pt.as_tensor_variable(alpha)
        theta = [0, 1, 2, 3, 4]
        theta_tensor = pt.as_tensor_variable(theta)
        y_tensor = delayed_adstock(
            x=x_tensor,
            alpha=alpha_tensor,
            theta=theta_tensor,
            l_max=12,
            axis=0,
            mode=mode,
        )
        y = y_tensor.eval()

        y_tensors = [
            delayed_adstock(
                x=x[:, i], alpha=alpha[i], theta=theta[i], l_max=12, mode=mode
            )
            for i in range(x.shape[1])
        ]
        ys = np.concatenate([y_t.eval()[..., None] for y_t in y_tensors], axis=1)
        assert y.shape == x.shape
        np.testing.assert_almost_equal(actual=y, desired=ys, decimal=12)

    @pytest.mark.parametrize(
        "x, lam, k, l_max",
        [
            (np.zeros(shape=(100)), 1, 1, 4),
            (np.ones(shape=(100)), 0.3, 0.5, 10),
            (np.ones(shape=(100)), 0.7, 1, 100),
            (np.zeros(shape=(100)), 0.2, 0.2, 5),
            (np.ones(shape=(100)), 0.5, 0.8, 7),
            (np.linspace(start=0.0, stop=1.0, num=50), 0.8, 1.5, 3),
            (np.linspace(start=0.0, stop=1.0, num=50), 0.8, 1, 50),
        ],
    )
    def test_weibull_pdf_adstock(self, x, lam, k, l_max):
        y = weibull_adstock(x=x, lam=lam, k=k, l_max=l_max, type=WeibullType.PDF).eval()

        assert np.all(np.isfinite(y))
        w = sp.stats.weibull_min.pdf(np.arange(l_max) + 1, c=k, scale=lam)
        w = (w - np.min(w)) / (np.max(w) - np.min(w))
        sp_y = batched_convolution(x, w).eval()

        np.testing.assert_almost_equal(y, sp_y)

    @pytest.mark.parametrize(
        "x, lam, k, l_max",
        [
            (np.zeros(shape=(100)), 1, 1, 4),
            (np.ones(shape=(100)), 0.3, 0.5, 10),
            (np.ones(shape=(100)), 0.7, 1, 100),
            (np.zeros(shape=(100)), 0.2, 0.2, 5),
            (np.ones(shape=(100)), 0.5, 0.8, 7),
            (np.linspace(start=0.0, stop=1.0, num=50), 0.8, 1.5, 3),
            (np.linspace(start=0.0, stop=1.0, num=50), 0.8, 1, 50),
        ],
    )
    def test_weibull_cdf_adsotck(self, x, lam, k, l_max):
        y = weibull_adstock(x=x, lam=lam, k=k, l_max=l_max, type=WeibullType.CDF).eval()

        assert np.all(np.isfinite(y))
        w = 1 - sp.stats.weibull_min.cdf(np.arange(l_max) + 1, c=k, scale=lam)
        w = np.cumprod(np.concatenate([[1], w]))
        sp_y = batched_convolution(x, w).eval()
        np.testing.assert_almost_equal(y, sp_y)

    @pytest.mark.parametrize(
        "type",
        [
            WeibullType.PDF,
            WeibullType.CDF,
        ],
    )
    def test_weibull_adstock_vectorized(self, type, dummy_design_matrix):
        x = dummy_design_matrix.copy()
        x_tensor = pt.as_tensor_variable(x)
        lam = [0.9, 0.33, 0.5, 0.1, 1.0]
        lam_tensor = pt.as_tensor_variable(lam)
        k = [0.8, 0.2, 0.6, 0.4, 1.0]
        k_tensor = pt.as_tensor_variable(k)
        y = weibull_adstock(
            x=x_tensor, lam=lam_tensor, k=k_tensor, l_max=12, type=type
        ).eval()

        y_tensors = [
            weibull_adstock(
                x=x_tensor[:, i], lam=lam_tensor[i], k=k_tensor[i], l_max=12, type=type
            )
            for i in range(x.shape[1])
        ]
        ys = np.concatenate([y_t.eval()[..., None] for y_t in y_tensors], axis=1)
        assert y.shape == x.shape
        np.testing.assert_almost_equal(actual=y, desired=ys, decimal=12)

    @pytest.mark.parametrize(
        "type, expectation",
        [
            ("PDF", does_not_raise()),
            ("CDF", does_not_raise()),
            ("PMF", pytest.raises(ValueError)),
            (WeibullType.PDF, does_not_raise()),
            (WeibullType.CDF, does_not_raise()),
        ],
    )
    def test_weibull_adstock_type(self, type, expectation):
        with expectation:
            weibull_adstock(x=np.ones(shape=(100)), lam=0.5, k=0.5, l_max=10, type=type)


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

    def test_inverse_scaled_logistic_saturation_lam_half(self):
        x = np.array([0.01, 0.1, 0.5, 1, 100])
        y = inverse_scaled_logistic_saturation(x=x, lam=x)
        expected = np.array([0.5] * len(x))
        np.testing.assert_almost_equal(
            y.eval(),
            expected,
            decimal=5,
            err_msg="The function does not behave as expected at the default value for eps",
        )

    def test_inverse_scaled_logistic_saturation_min_max_value(self):
        x = np.array([0, 1, 100, 500, 5000])
        lam = np.array([0.01, 0.25, 0.75, 1.5, 5.0, 10.0, 15.0])[:, None]

        y = inverse_scaled_logistic_saturation(x=x, lam=lam)
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

    @pytest.mark.parametrize(
        "x, x0, gain, r",
        [
            (np.ones(shape=(100)), 10, 0.5, 0.5),
            (np.zeros(shape=(100)), 10, 0.6, 0.3),
            (np.linspace(start=0.0, stop=100.0, num=50), 10, 0.001, 0.01),
            (np.linspace(start=0.0, stop=100.0, num=50), 10, 0.1, 0.01),
            (np.linspace(start=0.0, stop=100.0, num=50), 10, 1, 0.25),
        ],
    )
    def test_tanh_saturation_baselined_range(self, x, x0, gain, r):
        b = (gain * x0) / r
        assert tanh_saturation_baselined(x=x, x0=x0, gain=gain, r=r).eval().max() <= b
        assert tanh_saturation_baselined(x=x, x0=x0, gain=gain, r=r).eval().min() >= -b

    @pytest.mark.parametrize(
        "x, x0, gain, r",
        [
            (np.ones(shape=(100)), 10, 0.5, 0.5),
            (np.zeros(shape=(100)), 10, 0.6, 0.3),
            (np.linspace(start=0.0, stop=100.0, num=50), 10, 0.001, 0.1),
            (np.linspace(start=0.0, stop=100.0, num=50), 10, 0.1, 0.01),
            (np.linspace(start=0.0, stop=100.0, num=50), 10, 1, 0.25),
        ],
    )
    def test_tanh_saturation_baselined_inverse(self, x, x0, gain, r):
        y = tanh_saturation_baselined(x=x, x0=x0, gain=gain, r=r)
        b = (gain * x0) / r
        c = r / (gain * pt.arctanh(r))
        y_inv = (b * c) * pt.arctanh(y / b)
        np.testing.assert_array_almost_equal(x=x, y=y_inv.eval(), decimal=6)

    @pytest.mark.parametrize(
        "x, b, c",
        [
            (np.linspace(start=0.0, stop=10.0, num=50), 20, 0.5),
            (np.linspace(start=0.0, stop=10.0, num=50), 100, 0.5),
            (np.linspace(start=0.0, stop=10.0, num=50), 100, 1),
        ],
    )
    def test_tanh_saturation_parameterization_transformation(self, x, b, c):
        param_classic = TanhSaturationParameters(b, c)
        param_x0 = param_classic.baseline(5)
        param_x1 = param_x0.rebaseline(6)
        param_classic1 = param_x1.debaseline()
        y1 = tanh_saturation(x, *param_classic).eval()
        y2 = tanh_saturation_baselined(x, *param_x0).eval()
        y3 = tanh_saturation_baselined(x, *param_x1).eval()
        y4 = tanh_saturation(x, *param_classic1).eval()
        np.testing.assert_allclose(y1, y2)
        np.testing.assert_allclose(y2, y3)
        np.testing.assert_allclose(y3, y4)
        np.testing.assert_allclose(param_classic1.b.eval(), b)
        np.testing.assert_allclose(param_classic1.c.eval(), c, rtol=1e-06)

    @pytest.mark.parametrize(
        "x, alpha, lam, expected",
        [
            (10, 100, 5, 66.67),
            (20, 100, 5, 80),
        ],
    )
    def test_michaelis_menten(self, x, alpha, lam, expected):
        assert np.isclose(michaelis_menten(x, alpha, lam), expected, atol=0.01)

    @pytest.mark.parametrize(
        "sigma, beta, lam",
        [
            (1, 1, 0),
            (2, 0.5, 1),
            (3, 2, -1),
        ],
    )
    def test_hill_sigmoid_monotonicity(self, sigma, beta, lam):
        x = np.linspace(-10, 10, 100)
        y = hill_saturation_sigmoid(x, sigma, beta, lam).eval()
        assert np.all(np.diff(y) >= 0), "The function is not monotonic."

    @pytest.mark.parametrize(
        "sigma, beta, lam",
        [
            (1, 1, 0),
            (2, 0.5, 1),
            (3, 2, -1),
        ],
    )
    def test_hill_sigmoid_zero(self, sigma, beta, lam):
        y = hill_saturation_sigmoid(0, sigma, beta, lam).eval()
        assert y == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "x, sigma, beta, lam",
        [
            (0, 1, 1, 0),
            (5, 2, 0.5, 1),
            (-3, 3, 2, -1),
        ],
    )
    def test_hill_sigmoid_sigma_upper_bound(self, x, sigma, beta, lam):
        y = hill_saturation_sigmoid(x, sigma, beta, lam).eval()
        assert y <= sigma, f"The output {y} exceeds the upper bound sigma {sigma}."

    @pytest.mark.parametrize(
        "x, sigma, beta, lam, expected",
        [
            (0, 1, 1, 0, 0.5),
            (1, 2, 0.5, 1, 1),
            (-1, 3, 2, -1, 1.5),
        ],
    )
    def test_hill_sigmoid_behavior_at_lambda(self, x, sigma, beta, lam, expected):
        y = hill_saturation_sigmoid(x, sigma, beta, lam).eval()
        offset = sigma / (1 + np.exp(beta * lam))
        expected_with_offset = expected - offset
        np.testing.assert_almost_equal(
            y,
            expected_with_offset,
            decimal=5,
            err_msg="The function does not behave as expected at lambda.",
        )

    @pytest.mark.parametrize(
        "x, sigma, beta, lam",
        [
            (np.array([0, 1, 2]), 1, 1, 1),
            (np.array([-1, 0, 1]), 2, 0.5, 0),
            (np.array([1, 2, 3]), 3, 2, 2),
        ],
    )
    def test_hill_sigmoid_vectorized_input(self, x, sigma, beta, lam):
        y = hill_saturation_sigmoid(x, sigma, beta, lam).eval()
        assert y.shape == x.shape, (
            "The function did not return the correct shape for vectorized input."
        )

    @pytest.mark.parametrize(
        "sigma, beta, lam",
        [
            (1, 1, 0),
            (2, 0.5, 1),
            (3, 2, -1),
        ],
    )
    def test_hill_sigmoid_asymptotic_behavior(self, sigma, beta, lam):
        x = 1e6  # A very large value to approximate infinity
        y = hill_saturation_sigmoid(x, sigma, beta, lam).eval()
        offset = sigma / (1 + np.exp(beta * lam))
        expected = sigma - offset
        np.testing.assert_almost_equal(
            y,
            expected,
            decimal=5,
            err_msg="The function does not approach sigma as x approaches infinity.",
        )

    @pytest.mark.parametrize(
        argnames=["slope", "kappa"],
        argvalues=[
            (1, 1),
            (2, 0.5),
            (3, 2),
        ],
        ids=["slope=1, kappa=1", "slope=2, kappa=0.5", "slope=3, kappa=2"],
    )
    def test_hill_monotonicity(self, slope, kappa):
        x = np.linspace(0, 10, 100)
        y = hill_function(x, slope, kappa).eval()
        assert np.all(np.diff(y) >= 0), "The function is not monotonic."

    @pytest.mark.parametrize(
        argnames=["slope", "kappa"],
        argvalues=[
            (1, 1),
            (2, 0.5),
            (3, 2),
        ],
        ids=["slope=1, kappa=1", "slope=2, kappa=0.5", "slope=3, kappa=2"],
    )
    def test_hill_zero(self, slope, kappa):
        y = hill_function(0, slope, kappa).eval()
        assert y == pytest.approx(0.0)

    @pytest.mark.parametrize(
        argnames=["x", "slope", "kappa"],
        argvalues=[
            (1, 1, 1),
            (2, 0.5, 0.5),
            (3, 2, 2),
        ],
        ids=[
            "x=1, slope=1, kappa=1",
            "x=2, slope=0.5, kappa=0.5",
            "x=3, slope=2, kappa=2",
        ],
    )
    def test_hill_upper_bound(self, x, slope, kappa):
        y = hill_function(x, slope, kappa).eval()
        assert y <= 1, f"The output {y} exceeds the upper bound 1."

    @pytest.mark.parametrize(
        argnames=["slope", "kappa"],
        argvalues=[
            (1, 1),
            (2, 0.5),
            (3, 2),
        ],
        ids=["slope=1, kappa=1", "slope=2, kappa=0.5", "slope=3, kappa=2"],
    )
    def test_hill_behavior_at_midpoint(self, slope, kappa):
        y = hill_function(kappa, slope, kappa).eval()
        expected = 0.5
        np.testing.assert_almost_equal(
            y,
            expected,
            decimal=5,
            err_msg="The function does not behave as expected at the midpoint.",
        )

    @pytest.mark.parametrize(
        "x, slope, kappa",
        [
            (np.array([0, 1, 2]), 1, 1),
            (np.array([-1, 0, 1]), 2, 0.5),
            (np.array([1, 2, 3]), 3, 2),
        ],
    )
    def test_hill_vectorized_input(self, x, slope, kappa):
        y = hill_function(x, slope, kappa).eval()
        assert y.shape == x.shape, (
            "The function did not return the correct shape for vectorized input."
        )

    @pytest.mark.parametrize(
        "slope, kappa",
        [
            (1, 1),
            (2, 0.5),
            (3, 2),
        ],
    )
    def test_hill_asymptotic_behavior(self, slope, kappa):
        x = 1e6  # A very large value to approximate infinity
        y = hill_function(x, slope, kappa).eval()
        expected = 1
        np.testing.assert_almost_equal(
            y,
            expected,
            decimal=5,
            err_msg="The function does not approach sigma as x approaches infinity.",
        )


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
