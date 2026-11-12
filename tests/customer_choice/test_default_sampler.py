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
from pymc_marketing.customer_choice.nested_logit import NestedLogit

NUMPYRO_CASES = pytest.mark.parametrize(
    "numpyro_installed, expected", [(True, "numpyro"), (False, None)]
)


def _patch_version(monkeypatch, **overrides):
    """Patch ``importlib.metadata.version``: a value overrides the real version, ``None`` means missing."""
    real_version = importlib.metadata.version

    def version(name):
        if name in overrides:
            if overrides[name] is None:
                raise importlib.metadata.PackageNotFoundError(name)
            return overrides[name]
        return real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", version)


def _patch_numpyro(monkeypatch, numpyro_installed):
    _patch_version(monkeypatch, **({} if numpyro_installed else {"numpyro": None}))


@pytest.mark.parametrize("model_cls", [BayesianBLP, MaxDiffMixedLogit, MixedLogit])
@NUMPYRO_CASES
def test_default_nuts_sampler(monkeypatch, model_cls, numpyro_installed, expected):
    _patch_numpyro(monkeypatch, numpyro_installed)

    # ``default_sampler_config`` does not depend on the instance's data.
    config = model_cls.default_sampler_config.fget(None)

    assert config["nuts_sampler"] == expected


@NUMPYRO_CASES
def test_nested_logit_sample_default_nuts_sampler(
    monkeypatch, numpyro_installed, expected
):
    _patch_numpyro(monkeypatch, numpyro_installed)
    fit_kwargs = {}
    monkeypatch.setattr(
        NestedLogit, "sample_prior_predictive", lambda self, **kwargs: None
    )
    monkeypatch.setattr(
        NestedLogit, "fit", lambda self, **kwargs: fit_kwargs.update(kwargs)
    )
    monkeypatch.setattr(
        NestedLogit, "sample_posterior_predictive", lambda self, **kwargs: None
    )

    # ``sample`` only needs choice data to be present before it calls the stubs above.
    model = object.__new__(NestedLogit)
    model.choice_df = object()
    model.sample()

    assert fit_kwargs == {
        "nuts_sampler": expected,
        "idata_kwargs": {"log_likelihood": True},
    }


@pytest.mark.parametrize("model_cls", [BayesianBLP, MaxDiffMixedLogit, MixedLogit])
def test_default_nuts_sampler_with_damaged_nutpie_metadata(monkeypatch, model_cls):
    # An empty METADATA file makes ``version("nutpie")`` return None. The models
    # only care about NumPyro, so reading their defaults must not fail.
    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: None if name == "nutpie" else real_version(name),
    )

    config = model_cls.default_sampler_config.fget(None)

    assert config["nuts_sampler"] == "numpyro"
