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
import pytest

from pymc_marketing.metrics import crps, per_observation_crps


@pytest.mark.parametrize(
    argnames="y_true, y_pred, expected",
    argvalues=[
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3])[None, ...],
            np.array([0.0, 0.0, 0.0]),
        ),
        (
            np.ones(shape=(3, 1)),
            np.ones(shape=(3, 1))[None, ...],
            np.zeros(shape=(3, 1)),
        ),
        (
            np.ones(shape=(3, 3)),
            np.ones(shape=(3, 3))[None, ...],
            np.zeros(shape=(3, 3)),
        ),
    ],
    ids=["scalar", "vector", "tensor"],
)
def test_per_observation_crps_is_zero(y_true, y_pred, expected) -> None:
    per_obs_crps = per_observation_crps(y_true, y_pred)
    np.testing.assert_allclose(per_obs_crps, expected)


@pytest.mark.parametrize(
    argnames="y_true, y_pred, expected",
    argvalues=[
        (
            np.array([1, 0, 1]),
            np.array([0, 1, 0])[None, ...],
            np.array([1.0, 1.0, 1.0]),
        ),
        (
            np.ones(shape=(3, 1)),
            2 * np.ones(shape=(3, 1))[None, ...],
            np.ones(shape=(3, 1)),
        ),
        (
            2 * np.ones(shape=(3, 3)),
            np.ones(shape=(3, 3))[None, ...],
            np.ones(shape=(3, 3)),
        ),
    ],
    ids=["scalar", "vector", "tensor"],
)
def test_per_observation_crps_is_one(y_true, y_pred, expected) -> None:
    per_obs_crps = per_observation_crps(y_true, y_pred)
    np.testing.assert_allclose(per_obs_crps, expected)


@pytest.mark.parametrize(
    argnames="y_true, y_pred, expected",
    argvalues=[
        (
            np.array([1, 2, 3]),
            np.vstack([np.array([1, 2, 3])[None, ...], np.array([1, 2, 3])[None, ...]]),
            0.0,
        ),
        (
            np.ones(shape=(3, 1)),
            np.vstack(
                [np.ones(shape=(3, 1))[None, ...], np.ones(shape=(3, 1))[None, ...]]
            ),
            0.0,
        ),
        (
            np.ones(shape=(3, 3)),
            np.vstack(
                [np.ones(shape=(3, 3))[None, ...], np.ones(shape=(3, 3))[None, ...]]
            ),
            0.0,
        ),
    ],
    ids=["scalar", "vector", "tensor"],
)
def test_crps_is_zero(y_true, y_pred, expected) -> None:
    assert crps(y_true, y_pred) == expected


@pytest.mark.parametrize(
    argnames="y_true, y_pred, expected",
    argvalues=[
        (
            np.array([0, 1, 1]),
            np.vstack([np.array([1, 0, 0])[None, ...], np.array([1, 0, 0])[None, ...]]),
            1.0,
        ),
        (
            3 * np.ones(shape=(3, 1)),
            4
            * np.vstack(
                [
                    np.ones(shape=(3, 1))[None, ...],
                    np.ones(shape=(3, 1))[None, ...],
                ]
            ),
            1.0,
        ),
        (
            18 * np.ones(shape=(3, 3)),
            17
            * np.vstack(
                [
                    np.ones(shape=(3, 3))[None, ...],
                    np.ones(shape=(3, 3))[None, ...],
                ]
            ),
            1.0,
        ),
    ],
    ids=["scalar", "vector", "tensor"],
)
def test_crps_is_one(y_true, y_pred, expected) -> None:
    assert crps(y_true, y_pred) == expected


def test_weighted_crps_is_zero() -> None:
    y_true = np.array([1, 2, 3])
    y_pred = np.vstack([np.array([1, 0, 0])[None, ...], np.array([1, 0, 0])[None, ...]])
    sample_weight = np.array([1, 0, 0])
    result = crps(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    assert result == pytest.approx(0.0)


def test_weighted_crps_is_one() -> None:
    y_true = np.array([1, 2, 3])
    y_pred = np.vstack([np.array([1, 1, 2])[None, ...], np.array([1, 1, 2])[None, ...]])
    sample_weight = np.array([0, 0.5, 0.5])
    result = crps(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    assert result == pytest.approx(1.0)


def test_weighted_crps() -> None:
    y_true = np.array([1, 2, 3])
    y_pred = np.vstack([np.array([0, 0, 0])[None, ...], np.array([0, 0, 0])[None, ...]])
    sample_weight = np.array([1, 2, 3])
    result = crps(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    expected = (1 / 6) * (1 + 2 * 2 + 3 * 3)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    argnames="y_true, y_pred, expected",
    argvalues=[
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([0.0, 0.0, 0.0]),
        ),
        (
            np.ones(shape=(3, 1)),
            np.ones(shape=(3, 1)),
            np.zeros(shape=(3, 1)),
        ),
        (
            np.ones(shape=(3, 3)),
            np.ones(shape=(3, 3)),
            np.zeros(shape=(3, 3)),
        ),
    ],
    ids=["scalar", "vector", "tensor"],
)
def test_per_observation_crps_bad_shape_missing_sample_dim(
    y_true, y_pred, expected
) -> None:
    with pytest.raises(
        ValueError, match="Expected y_pred to have one extra sample dim on left"
    ):
        per_observation_crps(y_true, y_pred)
