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
import pymc as pm
import pytest
from pymc.distributions.distribution import DistributionMeta
from pymc.model_graph import fast_eval

from pymc_marketing.model_config import (
    ModelConfigError,
    UnsupportedShapeError,
    create_dim_handler,
    create_distribution_from_config,
    get_distribution,
)


def test_dim_handler_too_large():
    with pytest.raises(UnsupportedShapeError, match="At most two dims"):
        create_dim_handler(desired_dims=("dim_1", "dim_2", "dim_3"))


@pytest.mark.parametrize(
    "var, dims, expected",
    [
        (1, None, 1),
        (np.array([1, 2, 3]), "dim_1", np.array([1, 2, 3])),
    ],
)
def test_dim_handler_1d(var, dims, expected) -> None:
    handle_shape = create_dim_handler(desired_dims="dim_1")

    np.testing.assert_array_equal(handle_shape(var, dims=dims), expected)


@pytest.mark.parametrize(
    "var, dims, expected",
    [
        (1, None, 1),
        (np.array([1, 2, 3]), "dim_1", np.array([1, 2, 3])[:, None]),
        (np.array([1, 2, 3]), "dim_2", np.array([1, 2, 3])),
        (
            np.array(
                [
                    [1, 2],
                    [3, 4],
                ]
            ),
            ("dim_1", "dim_2"),
            np.array(
                [
                    [1, 2],
                    [3, 4],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 2],
                    [3, 4],
                ]
            ),
            ("dim_2", "dim_1"),
            np.array(
                [
                    [1, 3],
                    [2, 4],
                ]
            ),
        ),
    ],
)
def test_dim_handler_2d(var, dims, expected) -> None:
    handle_shape = create_dim_handler(desired_dims=("dim_1", "dim_2"))

    np.testing.assert_array_equal(handle_shape(var, dims=dims), expected)


def test_dim_handler_nonsubset():
    handle_shape = create_dim_handler(desired_dims=("dim_1", "dim_2"))

    with pytest.raises(
        UnsupportedShapeError, match="The dims of the variable are not supported"
    ):
        handle_shape(var=1, dims="dim_3")


@pytest.mark.parametrize(
    "name",
    [
        "Normal",
        "StudentT",
        "Laplace",
        "Logistic",
    ],
)
def test_get_distribution(name: str):
    distribution = get_distribution(name=name)

    assert isinstance(distribution, DistributionMeta)
    assert distribution.__name__ == name


def test_get_distribution_unknown():
    with pytest.raises(ValueError):
        get_distribution(name="Unknown")


@pytest.fixture
def model_config():
    return {
        # Non-nested distribution
        "beta": {
            "dist": "Normal",
            "kwargs": {
                "mu": 0.0,
                "sigma": 1.0,
            },
        },
        # Nested distribution
        "alpha": {
            "dist": "Normal",
            "kwargs": {
                "mu": {
                    "dist": "Normal",
                    "kwargs": {
                        "mu": 0.0,
                        "sigma": 1.0,
                    },
                },
                "sigma": {
                    "dist": "HalfNormal",
                    "kwargs": {
                        "sigma": 1.0,
                    },
                },
            },
            "dims": "channel",
        },
        # 2D nested distribution
        "gamma": {
            "dist": "Normal",
            "kwargs": {
                "mu": {
                    "dist": "Normal",
                    "kwargs": {
                        "mu": 0.0,
                        "sigma": 1.0,
                    },
                    "dims": "channel",
                },
                "sigma": {
                    "dist": "HalfNormal",
                    "kwargs": {
                        "sigma": 1.0,
                    },
                    "dims": "geo",
                },
            },
            "dims": ("channel", "geo"),
        },
        # 2D explicit kwargs
        "delta": {
            "dist": "Normal",
            "kwargs": {
                "mu": np.array([1.0]),
                "sigma": np.array([1.0, 2.0, 3.0])[:, None],
            },
            "dims": ("channel", "control"),
        },
        # Incorrect config
        "error": {
            "dist": "Normal",
        },
        "error_later": {
            "dist": "Normal",
            "kwargs": {"mu": "wrong"},
        },
    }


@pytest.fixture
def coords() -> dict[str, list[str]]:
    return {
        "channel": ["A", "B", "C"],
        "geo": ["X", "Y"],
        "control": ["X1"],
    }


@pytest.mark.parametrize(
    "name, expected_names, expected_shapes",
    [
        ("beta", ["beta"], [()]),
        ("alpha", ["alpha", "alpha_mu", "alpha_sigma"], [(3,), (), ()]),
        ("gamma", ["gamma", "gamma_mu", "gamma_sigma"], [(3, 2), (3,), (2,)]),
        ("delta", ["delta"], [(3, 1)]),
    ],
)
def test_create_distribution(
    coords, model_config, name, expected_names, expected_shapes
) -> None:
    with pm.Model(coords=coords) as model:
        create_distribution_from_config(name, model_config)

    assert len(model.named_vars) == len(expected_names)
    assert all(name in model.named_vars for name in expected_names)

    for name, expected_shape in zip(expected_names, expected_shapes, strict=True):
        assert fast_eval(model[name]).shape == expected_shape


@pytest.mark.parametrize(
    "name, param_name",
    [("error", "error"), ("error_later", "error_later_mu")],
)
def test_create_distribution_error(model_config, name, param_name) -> None:
    with pm.Model():
        msg = f"Invalid parameter configuration for '{param_name}'"
        with pytest.raises(ModelConfigError, match=msg):
            create_distribution_from_config(name, model_config)
