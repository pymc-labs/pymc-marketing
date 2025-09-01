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
# Test arguments - numpy arrays, scalars, and nested priors
# Test round trip dict to class to dict to class
# Test exception raising around mu and sigma
# Test sample_prior

import numpy as np
import pymc as pm
import pytest
from pymc_extras.prior import Prior

from pymc_marketing.special_priors import LogNormalExp, _is_log_normal_exp_type


@pytest.mark.parametrize(
    "mu, sigma, centered",
    [
        (Prior("Gamma", mu=1.0, sigma=1.0), Prior("Gamma", mu=1.0, sigma=1.0), True),
        (np.array([1, 2, 3]), np.array([4, 5, 6]), True),
        (1.0, 2.0, False),
    ],
)
def test_lognormalexp_args(mu, sigma, centered):
    """
    Checks:
    - sample_prior
    - create_variable
    - round trip dict to class to dict to class
    """
    lognormalexp = LogNormalExp(
        mu=mu, sigma=sigma, centered=centered, dims=("channel",)
    )

    coords = {"channel": ["C1", "C2", "C3"]}
    prior = lognormalexp.sample_prior(coords=coords)
    assert prior.channel.shape == (3,)
    if centered is False:
        assert "test_log_offset" in prior.data_vars

    with pm.Model(coords=coords):
        lognormalexp.create_variable("test")

    assert (
        lognormalexp.to_dict()
        == lognormalexp.from_dict(lognormalexp.to_dict()).to_dict()
    )


def test_lognormalexp_args_invalid():
    with pytest.raises(ValueError):
        LogNormalExp(alpha=1.0, beta=1.0)


def test_the_deserializer_can_distinguish_between_types_of_prior_classes():
    assert _is_log_normal_exp_type(LogNormalExp(mu=1.0, sigma=1.0).to_dict())
    assert not _is_log_normal_exp_type(Prior("Normal", mu=1.0, sigma=1.0).to_dict())
