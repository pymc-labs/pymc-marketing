#   Copyright 2022 - 2026 The PyMC Labs Developers
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
from pymc_extras.prior import Prior

from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.model_config import ModelConfigError, parse_model_config


@pytest.fixture
def model_config():
    return {
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
        "delta": Prior(
            "Normal",
            mu=np.array([1.0]),
            sigma=np.array([1.0, 2.0, 3.0])[:, None],
            dims=("channel", "control"),
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
        "intercept_tvp_config": {
            "m": 200,
            "L": 119.17,
            "eta_lam": 1.0,
            "ls_mu": 5.0,
            "ls_sigma": 10.0,
            "cov_func": None,
        },
        "non_distribution": {
            "key": "This is not a distribution",
        },
    }


def test_parse_model_config(model_config) -> None:
    ignore_keys = ["delta"]
    to_parse = {
        name: value for name, value in model_config.items() if name not in ignore_keys
    }

    result = parse_model_config(
        to_parse,
        hsgp_kwargs_fields=["intercept_tvp_config"],
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
        "non_distribution": {
            "key": "This is not a distribution",
        },
    }


def test_parse_model_config_passes_lists_through() -> None:
    """Test that list values pass through unchanged."""
    model_config = {
        "dropout_covariate_cols": ["channel", "tier"],
        "alpha": Prior("Normal", mu=0, sigma=1),
    }

    result = parse_model_config(model_config)

    assert result["dropout_covariate_cols"] == ["channel", "tier"]
    assert result["alpha"] == Prior("Normal", mu=0, sigma=1)


@pytest.mark.parametrize("legacy_key", ["dist", "distribution"])
def test_parse_model_config_rejects_legacy_prior_spec(legacy_key) -> None:
    """Legacy dict-format prior specs raise a clear migration error."""
    model_config = {
        "alpha": {legacy_key: "Normal", "kwargs": {"mu": 0, "sigma": 1}},
    }

    with pytest.raises(ModelConfigError, match=r"use pymc_extras\.prior\.Prior"):
        parse_model_config(model_config)
