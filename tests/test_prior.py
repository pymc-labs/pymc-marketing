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

import pytest
from pymc_extras.deserialize import deserialize
from pymc_extras.prior import Prior

from pymc_marketing import prior


@pytest.mark.parametrize(
    "data, expected",
    [
        pytest.param(
            {
                "distribution": "Normal",
                "mu": 10,
                "sigma": 1,
            },
            Prior("Normal", mu=10, sigma=1),
            id="another Prior",
        ),
        pytest.param(
            {
                "distribution": "Laplace",
                "mu": 1,
                "b": 2,
                "dims": ("x", "y"),
                "transform": "sigmoid",
            },
            Prior("Laplace", mu=1, b=2, dims=("x", "y"), transform="sigmoid"),
            id="Prior",
        ),
        pytest.param(
            {"distribution": "Normal", "mu": {"distribution": "Normal"}},
            Prior("Normal", mu=Prior("Normal")),
            id="Prior with nested distribution",
        ),
    ],
)
def test_alternative_prior_deserialize(data, expected) -> None:
    assert deserialize(data) == expected


@pytest.mark.parametrize(
    "obj, args, kwargs, match",
    [
        (prior.Prior, ["Normal"], {}, "The Prior class"),
        (prior.Censored, [Prior("Normal")], dict(lower=0), "The Censored"),
        (prior.Scaled, [Prior("Normal")], dict(factor=2), "The Scaled"),
        (prior.create_dim_handler, [("date", "channel")], {}, "The create_dim_handler"),
    ],
)
def test_deprecation_warnings(obj, args, kwargs, match):
    with pytest.warns(DeprecationWarning, match=match):
        obj(*args, **kwargs)
