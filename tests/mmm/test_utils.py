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
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.preprocessing import MaxAbsScaler

from pymc_marketing.mmm.utils import (
    apply_sklearn_transformer_across_dim,
    create_new_spend_data,
    sigmoid_saturation,
    transform_1d_array,
)


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
            np.ones(shape=(1, 3, len(dates), 2)),
            coords={
                "chain": [1],
                "draw": [1, 2, 3],
                "date": dates,
                "channel": ["channel1", "channel2"],
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
    )

    xr.testing.assert_allclose(result, data * 2)


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
