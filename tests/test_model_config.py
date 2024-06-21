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

from pymc_marketing.model_config import (
    parse_model_config,
)
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
        # Incorrect config
        "error": {
            "dist": "Normal",
            "kwargs": {"mu": "wrong"},
        },
        # Non distribution
        "non_distribution": {
            "key": "This is not a distribution",
    }


def test_parse_model_config(model_config) -> None:
    ignore_keys = ["delta"]
    non_distributions = ["non_distribution", "error"]
    result = parse_model_config(
        {
            name: value
            for name, value in model_config.items()
            if name not in ignore_keys
        },
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
        "error": {
            "dist": "Normal",
            "kwargs": {"mu": "wrong"},
        },
        "non_distribution": {
            "key": "This is not a distribution",
        },
    }
