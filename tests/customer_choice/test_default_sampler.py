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
"""The customer-choice models only default to NumPyro when it is installed."""

import importlib.metadata

import pytest

from pymc_marketing.customer_choice.bayesian_blp import BayesianBLP
from pymc_marketing.customer_choice.maxdiff import MaxDiffMixedLogit
from pymc_marketing.customer_choice.mixed_logit import MixedLogit


@pytest.mark.parametrize("model_cls", [BayesianBLP, MaxDiffMixedLogit, MixedLogit])
@pytest.mark.parametrize(
    "numpyro_installed, expected", [(True, "numpyro"), (False, None)]
)
def test_default_nuts_sampler(monkeypatch, model_cls, numpyro_installed, expected):
    real_version = importlib.metadata.version

    def version(name):
        if name == "numpyro" and not numpyro_installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", version)

    # ``default_sampler_config`` does not depend on the instance's data.
    config = model_cls.default_sampler_config.fget(None)

    assert config["nuts_sampler"] == expected
