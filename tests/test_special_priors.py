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
import pymc as pm
import pytest
import xarray as xr
from pymc_extras.prior import Prior

from pymc_marketing.special_priors import (
    LogNormalPrior,
    _is_LogNormalPrior_type,
)


@pytest.mark.parametrize(
    "mean, std, centered, dims",
    [
        (
            Prior("Gamma", mu=1.0, sigma=1.0),
            Prior("Gamma", mu=1.0, sigma=1.0),
            True,
            ("channel",),
        ),
        (1.0, 2.0, False, ("channel",)),
        (1.0, 2.0, True, ("channel",)),
        (np.array([1, 2, 3]), np.array([4, 5, 6]), True, ("channel",)),
        (np.array([1, 2, 3]), np.array([4, 5, 6]), False, ("channel",)),
        (1.0, 2.0, True, ()),
    ],
)
def test_LogNormalPrior_args(mean, std, centered, dims):
    """
    Checks:
    - sample_prior runs
    - create_variable runs
    - round trip: dict to class to dict to class, doesn't lose any information
    """
    rv = LogNormalPrior(mean=mean, std=std, centered=centered, dims=dims)

    coords = {"channel": ["C1", "C2", "C3"]}

    if dims:
        prior = rv.sample_prior(coords=coords)
        assert prior.channel.shape == (len(coords["channel"]),)
    else:
        prior = rv.sample_prior()
        assert isinstance(prior, xr.Dataset)

    if centered is False:
        assert "variable_log_offset" in prior.data_vars

    with pm.Model(coords=coords):
        rv.create_variable("test")

    assert rv.to_dict() == rv.from_dict(rv.to_dict()).to_dict()


def test_LogNormalPrior_args_invalid():
    with pytest.raises(ValueError):
        LogNormalPrior(alpha=1.0, beta=1.0)


def test_the_deserializer_can_distinguish_between_types_of_prior_classes():
    assert _is_LogNormalPrior_type(LogNormalPrior(mean=1.0, std=1.0).to_dict())
    assert not _is_LogNormalPrior_type(Prior("Normal", mu=1.0, sigma=1.0).to_dict())
