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

import pandas as pd
import pymc as pm
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


@pytest.mark.parametrize(
    "model_cls", [BayesianBLP, MaxDiffMixedLogit, MixedLogit, NestedLogit]
)
@NUMPYRO_CASES
def test_default_nuts_sampler(monkeypatch, model_cls, numpyro_installed, expected):
    _patch_numpyro(monkeypatch, numpyro_installed)

    # ``default_sampler_config`` does not depend on the instance's data.
    config = model_cls.default_sampler_config.fget(None)

    assert config["nuts_sampler"] == expected


def _mixed_logit(sampler_config):
    choice_df = pd.DataFrame(
        {
            "choice": ["bus", "car", "bus", "train", "car"],
            "bus_price": [2.0, 2.5, 2.0, 2.2, 2.3],
            "bus_time": [45, 50, 45, 48, 52],
            "car_price": [5.0, 4.8, 5.2, 5.1, 4.9],
            "car_time": [30, 28, 32, 29, 31],
            "train_price": [3.5, 3.8, 3.6, 3.7, 3.9],
            "train_time": [35, 38, 36, 37, 39],
            "income": [50000, 60000, 55000, 70000, 65000],
        }
    )
    utility_equations = [
        "bus ~ bus_price + bus_time | income | bus_price",
        "car ~ car_price + car_time | income | car_price",
        "train ~ train_price + train_time | income | train_price",
    ]
    return MixedLogit(
        choice_df,
        utility_equations,
        "choice",
        ["price", "time"],
        sampler_config=sampler_config,
    )


def _nested_logit(sampler_config):
    choice_df = pd.DataFrame(
        {
            "choice": ["alt", "another", "other", "option"],
            "alt_X1": [1, 2, 3, 7],
            "alt_X2": [4, 5, 6, 5],
            "other_X1": [1, 2, 4, 7],
            "other_X2": [5, 6, 8, 0],
            "option_X1": [4, 6, 7, 1],
            "option_X2": [6, 6, 7, 4],
            "another_X1": [5, 3, 7, 8],
            "another_X2": [7, 4, 2, 8],
        }
    )
    utility_equations = [
        "alt ~ alt_X1 + alt_X2",
        "other ~ other_X1 + other_X2",
        "option ~ option_X1 + option_X2",
        "another ~ another_X1 + another_X2",
    ]
    return NestedLogit(
        choice_df=choice_df,
        utility_equations=utility_equations,
        depvar="choice",
        covariates=["X1", "X2"],
        nesting_structure={"nest1": ["alt", "other"], "nest2": ["option", "another"]},
        sampler_config=sampler_config,
    )


class _SampleCalled(Exception):
    """Raised by the patched ``pm.sample`` once it has recorded its arguments."""


@pytest.mark.parametrize(
    "build_model, model_cls",
    [(_mixed_logit, MixedLogit), (_nested_logit, NestedLogit)],
)
@pytest.mark.parametrize("numpyro_installed", [True, False])
def test_sample_keeps_user_nuts_sampler(
    monkeypatch, build_model, model_cls, numpyro_installed
):
    _patch_numpyro(monkeypatch, numpyro_installed)
    sampled_with = {}

    def fake_sample(**kwargs):
        sampled_with.update(kwargs)
        raise _SampleCalled

    monkeypatch.setattr(
        model_cls, "sample_prior_predictive", lambda self, **kwargs: None
    )
    monkeypatch.setattr(pm, "sample", fake_sample)
    model = build_model({"nuts_sampler": "blackjax"})

    with pytest.raises(_SampleCalled):
        model.sample()

    assert sampled_with["nuts_sampler"] == "blackjax"
