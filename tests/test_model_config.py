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

import warnings

import numpy as np
import pytest

from pymc_marketing.deserialize import (
    DESERIALIZERS,
    register_deserialization,
)
from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.model_config import ModelConfigError, parse_model_config
from pymc_marketing.prior import Prior


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
        # Hierarchical centered distribution
        "hierarchical_centered": {
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
            "centered": True,
        },
        # Hierarchical non-centered distribution
        "hierarchical_non_centered": {
            "dist": "Normal",
            "kwargs": {
                "mu": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
                "sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
            },
            "dims": "channel",
            "centered": False,
        },
        # 2D Hierarchical non-centered distribution
        "hierarchical_non_centered_2d": {
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
            "centered": False,
        },
        # TVP Intercept
        "intercept_tvp_config": {
            "m": 200,
            "L": 119.17,
            "eta_lam": 1.0,
            "ls_mu": 5.0,
            "ls_sigma": 10.0,
            "cov_func": None,
        },
        # Incorrect config
        "error": {
            "dist": "Normal",
            "kwargs": {"mu": "wrong"},
        },
        # Non distribution
        "non_distribution": {
            "key": "This is not a distribution",
        },
    }


def test_parse_model_config(model_config) -> None:
    ignore_keys = ["delta"]
    non_distributions = ["non_distribution", "error"]
    to_parse = {
        name: value for name, value in model_config.items() if name not in ignore_keys
    }
    # Ignore deprecation warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result = parse_model_config(
            to_parse,
            hsgp_kwargs_fields=["intercept_tvp_config"],
            non_distributions=non_distributions,
        )

    assert result == {
        "beta": Prior("Normal", mu=0.0, sigma=1.0),
        "alpha": Prior(
            "Normal",
            mu=Prior("Normal", mu=0.0, sigma=1.0),
            sigma=Prior("HalfNormal", sigma=1.0),
            dims="channel",
        ),
        "gamma": Prior(
            "Normal",
            mu=Prior("Normal", mu=0.0, sigma=1.0, dims="channel"),
            sigma=Prior("HalfNormal", sigma=1.0, dims="geo"),
            dims=("channel", "geo"),
        ),
        "hierarchical_centered": Prior(
            "Normal",
            mu=Prior("Normal", mu=0.0, sigma=1.0, dims="channel"),
            sigma=Prior("HalfNormal", sigma=1.0, dims="geo"),
            dims=("channel", "geo"),
        ),
        "hierarchical_non_centered": Prior(
            "Normal",
            mu=Prior("HalfNormal", sigma=2),
            sigma=Prior("HalfNormal", sigma=1),
            dims="channel",
            centered=False,
        ),
        "hierarchical_non_centered_2d": Prior(
            "Normal",
            mu=Prior("Normal", mu=0.0, sigma=1.0, dims="channel"),
            sigma=Prior("HalfNormal", sigma=1.0, dims="geo"),
            dims=("channel", "geo"),
            centered=False,
        ),
        "intercept_tvp_config": HSGPKwargs(
            m=200,
            L=119.17,
            eta_lam=1.0,
            ls_mu=5.0,
            ls_sigma=10.0,
            cov_func=None,
        ),
        "error": {
            "dist": "Normal",
            "kwargs": {"mu": "wrong"},
        },
        "non_distribution": {
            "key": "This is not a distribution",
        },
    }


def test_parse_model_config_warns() -> None:
    model_config = {
        "alpha": {
            "dist": "Normal",
            "kwargs": {"mu": 0, "sigma": 1},
        },
    }

    with pytest.warns(DeprecationWarning, match="alpha is automatically"):
        result = parse_model_config(model_config)

    assert result == {
        "alpha": Prior("Normal", mu=0, sigma=1),
    }


def test_parse_model_config_catches_errors() -> None:
    model_config = {
        "alpha": "Normal",
        "beta": {"dist": "Beta", "kwargs": {"lam": 1}},
        "lam": {"dist": "IncorrectDistribution"},
        "gamma": Prior("Normal"),
    }

    msg = "3 errors"
    with pytest.raises(ModelConfigError, match=msg):
        parse_model_config(model_config)


class AribraryPriorClass:
    def __init__(self, msg: str, value: int):
        self.msg = msg
        self.value = value
        self.dims = ()

    def create_variable(self, name: str):
        return 1

    def __eq__(self, other):
        return self.msg == other.msg and self.value == other.value


@pytest.fixture
def register_arbitrary_prior_class():
    register_deserialization(
        is_type=lambda data: data.keys() == {"msg", "value"},
        deserialize=lambda data: AribraryPriorClass(
            msg=data["msg"], value=data["value"]
        ),
    )

    yield

    DESERIALIZERS.pop()


def test_parse_model_config_custom_class(register_arbitrary_prior_class) -> None:
    model_config = {
        "alpha": {"msg": "Hello", "value": 42},
    }

    with pytest.warns(DeprecationWarning, match="alpha is automatically"):
        result = parse_model_config(model_config)

    assert result == {
        "alpha": AribraryPriorClass(msg="Hello", value=42),
    }
