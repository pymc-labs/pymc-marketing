#   Copyright 2024 The PyMC Labs Developers
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
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.preprocessing import MaxAbsScaler

from pymc_marketing.mmm.utils import (
    apply_sklearn_transformer_across_dim,
    compute_sigmoid_second_derivative,
    create_new_spend_data,
    drop_scalar_coords,
    estimate_menten_parameters,
    estimate_sigmoid_parameters,
    find_sigmoid_inflection_point,
    sigmoid_saturation,
    transform_1d_array,
)


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


@pytest.fixture
def mock_method():
    def _mock_method(x):
        if x.ndim != 2:
            raise ValueError("x must be 2-dimensional")

        return x * 2

    return _mock_method


@pytest.fixture
def create_mock_mmm_return_data():
    def _create_mock_mm_return_data(combined: bool) -> xr.DataArray:
        dates = pd.date_range(start="2020-01-01", end="2020-01-31", freq="W-MON")
        data = xr.DataArray(
            np.ones(shape=(1, 3, len(dates))),
            coords={
                "chain": [1],
                "draw": [1, 2, 3],
                "date": dates,
            },
        )

        if combined:
            data = data.stack(sample=("chain", "draw"))

        return data

    return _create_mock_mm_return_data


@pytest.mark.parametrize("combined", [True, False])
def test_apply_sklearn_function_across_dim(
    mock_method, create_mock_mmm_return_data, combined: bool
) -> None:
    # Data that would be returned from a MMM model
    data = create_mock_mmm_return_data(combined=combined)
    result = apply_sklearn_transformer_across_dim(
        data,
        mock_method,
        dim_name="date",
        combined=combined,
    )

    xr.testing.assert_allclose(result, data * 2)


def test_apply_sklearn_function_across_dim_error(
    mock_method,
    create_mock_mmm_return_data,
) -> None:
    data = create_mock_mmm_return_data(combined=False)

    with pytest.raises(ValueError, match="x must be 2-dimensional"):
        apply_sklearn_transformer_across_dim(
            data,
            mock_method,
            dim_name="date",
            combined=True,
        )


@pytest.mark.parametrize("constructor", [pd.Series, np.array])
def test_transform_1d_array(constructor):
    transform = MaxAbsScaler()
    y = constructor([1, 2, 3, 4, 5])
    transform.fit(np.array(y)[:, None])
    expected = np.array([1, 2, 3, 4, 5]) / 5
    result = transform_1d_array(transform.transform, y)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "x, alpha, lam, expected",
    [
        (0, 1, 1, 0),
        (1, 1, 1, 0.4621),
    ],
)
def test_sigmoid_saturation(x, alpha, lam, expected):
    assert np.isclose(sigmoid_saturation(x, alpha, lam), expected, atol=0.01)


@pytest.mark.parametrize(
    "x, alpha, lam",
    [
        (0, 0, 1),
        (1, -1, 1),
        (1, 1, 0),
    ],
)
def test_sigmoid_saturation_value_errors(x, alpha, lam):
    with pytest.raises(ValueError):
        sigmoid_saturation(x, alpha, lam)
    (
        "spend, adstock_max_lag, one_time, spend_leading_up, expected_result",
        [
            (
                [1, 2],
                2,
                True,
                None,
                [[0, 0], [0, 0], [1, 2], [0, 0], [0, 0]],
            ),
            (
                [1, 2],
                2,
                False,
                None,
                [[0, 0], [0, 0], [1, 2], [1, 2], [1, 2]],
            ),
            (
                [1, 2],
                2,
                True,
                [3, 4],
                [[3, 4], [3, 4], [1, 2], [0, 0], [0, 0]],
            ),
        ],
    )


@pytest.mark.parametrize(
    "spend, adstock_max_lag, one_time, spend_leading_up, expected_result",
    [
        (
            [1, 2],
            2,
            True,
            None,
            [[0, 0], [0, 0], [1, 2], [0, 0], [0, 0]],
        ),
        (
            [1, 2],
            2,
            False,
            None,
            [[0, 0], [0, 0], [1, 2], [1, 2], [1, 2]],
        ),
        (
            [1, 2],
            2,
            True,
            [3, 4],
            [[3, 4], [3, 4], [1, 2], [0, 0], [0, 0]],
        ),
    ],
)
def test_create_new_spend_data(
    spend, adstock_max_lag, one_time, spend_leading_up, expected_result
) -> None:
    spend = np.array(spend)
    if spend_leading_up is not None:
        spend_leading_up = np.array(spend_leading_up)
    new_spend_data = create_new_spend_data(
        spend, adstock_max_lag, one_time, spend_leading_up
    )

    np.testing.assert_allclose(
        new_spend_data,
        np.array(expected_result),
    )


def test_create_new_spend_data_value_errors() -> None:
    with pytest.raises(
        ValueError, match="spend_leading_up must be the same length as the spend"
    ):
        create_new_spend_data(
            spend=np.array([1, 2]),
            adstock_max_lag=2,
            one_time=True,
            spend_leading_up=np.array([3, 4, 5]),
        )


@pytest.fixture
def mock_curve_with_scalars() -> xr.DataArray:
    coords = {
        "x": [1, 2, 3],
        "y": [10, 20, 30],
        "scalar1": 42,  # Scalar coordinate
        "scalar2": 3.14,  # Another scalar coordinate
    }
    data = np.random.rand(3, 3)
    return xr.DataArray(data, coords=coords, dims=["x", "y"])


def test_drop_scalar_coords(mock_curve_with_scalars) -> None:
    original_curve = mock_curve_with_scalars.copy(deep=True)  # Make a deep copy
    curve = drop_scalar_coords(mock_curve_with_scalars)

    # Ensure scalar coordinates are removed
    assert "scalar1" not in curve.coords
    assert "scalar2" not in curve.coords

    # Ensure other coordinates are still present
    assert "x" in curve.coords
    assert "y" in curve.coords

    # Ensure data shape is unchanged
    assert curve.shape == (3, 3)

    # Ensure the original DataArray was not modified
    xr.testing.assert_identical(mock_curve_with_scalars, original_curve)
