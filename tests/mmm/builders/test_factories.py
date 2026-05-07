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
import warnings

import pytest
from pymc_extras.prior import Prior

from pymc_marketing.mmm.builders.factories import build, locate, resolve
from pymc_marketing.mmm.mmm import MMM
from pymc_marketing.special_priors import LogNormalPrior


@pytest.mark.parametrize(
    "qualname, expected",
    [
        pytest.param("pymc_marketing.mmm.MMM", MMM, id="alternative-import"),
        pytest.param("pymc_extras.prior.Prior", Prior, id="full-import"),
    ],
)
def test_locate(qualname, expected) -> None:
    assert locate(qualname) is expected


def test_build_warns_on_unknown_spec_keys():
    """build() should warn when spec contains keys besides class/kwargs/args."""
    spec = {
        "class": "pymc_marketing.mmm.GeometricAdstock",
        "kwargs": {"l_max": 4},
        "original_scale_vars": ["channel_contribution"],
        "calibration": [],
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = build(spec)

    assert obj is not None
    warning_messages = [str(w.message) for w in caught]
    assert any("Unknown keys" in msg for msg in warning_messages)
    assert any("original_scale_vars" in msg for msg in warning_messages)


def test_build_no_warning_for_clean_spec():
    """build() should not warn for a spec with only class/kwargs/args."""
    spec = {
        "class": "pymc_marketing.mmm.GeometricAdstock",
        "kwargs": {"l_max": 4},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build(spec)

    warning_messages = [str(w.message) for w in caught]
    assert not any("Unknown keys" in msg for msg in warning_messages)


class TestBuildClassKeyedPriorInPriors:
    """Regression tests for GitHub issues #2071 and #2439.

    When a prior inside ``priors`` uses the ``class:`` BuildSpec format
    (e.g. LogNormalPrior), ``build()`` must route it through the factory
    instead of ``pymc_extras.deserialize()``, which cannot handle
    ``class``-keyed dicts.
    """

    def test_build_lognormal_prior_via_class_key_in_priors(self):
        """LogNormalPrior specified with class: key inside priors dict."""
        spec = {
            "class": "pymc_marketing.mmm.LogisticSaturation",
            "kwargs": {
                "priors": {
                    "lam": {
                        "distribution": "Gamma",
                        "mu": 0.5,
                        "sigma": 1.5,
                    },
                    "beta": {
                        "class": "pymc_marketing.special_priors.LogNormalPrior",
                        "kwargs": {
                            "mean": 1.0,
                            "std": 0.5,
                        },
                    },
                }
            },
        }
        result = build(spec)
        assert isinstance(result.priors["lam"], Prior)
        assert isinstance(result.priors["beta"], LogNormalPrior)

    def test_build_lognormal_prior_with_nested_distribution_params(self):
        """LogNormalPrior with nested distribution-keyed Prior parameters."""
        spec = {
            "class": "pymc_marketing.special_priors.LogNormalPrior",
            "kwargs": {
                "mean": {
                    "distribution": "Gamma",
                    "mu": 0.25,
                    "sigma": 1.0,
                },
                "std": {
                    "distribution": "HalfNormal",
                    "sigma": 1.0,
                },
            },
        }
        result = build(spec)
        assert isinstance(result, LogNormalPrior)
        assert isinstance(result.parameters["mean"], Prior)
        assert isinstance(result.parameters["std"], Prior)

    def test_build_lognormal_prior_with_dims_and_centered(self):
        """LogNormalPrior with dims and centered=False inside kwargs."""
        spec = {
            "class": "pymc_marketing.special_priors.LogNormalPrior",
            "kwargs": {
                "mean": {
                    "distribution": "Gamma",
                    "mu": 0.25,
                    "sigma": 1.0,
                    "dims": "channel",
                },
                "std": {
                    "distribution": "HalfNormal",
                    "sigma": 1.0,
                    "dims": "channel",
                },
                "centered": False,
                "dims": ["geo", "channel"],
            },
        }
        result = build(spec)
        assert isinstance(result, LogNormalPrior)
        assert result.centered is False
        assert result.dims == ("geo", "channel")
        assert isinstance(result.parameters["mean"], Prior)
        assert isinstance(result.parameters["std"], Prior)


class TestResolveDistributionDicts:
    """resolve() must handle distribution-keyed and special_prior-keyed dicts."""

    def test_resolve_distribution_dict(self):
        data = {"distribution": "HalfNormal", "sigma": 1.0}
        result = resolve(data)
        assert isinstance(result, Prior)

    def test_resolve_special_prior_dict(self):
        data = {"special_prior": "LogNormalPrior", "mean": 1.0, "std": 0.5}
        result = resolve(data)
        assert isinstance(result, LogNormalPrior)

    def test_resolve_plain_dict_unchanged(self):
        data = {"some_key": "some_value", "number": 42}
        result = resolve(data)
        assert result == data

    def test_resolve_class_dict_builds(self):
        data = {
            "class": "pymc_marketing.mmm.GeometricAdstock",
            "kwargs": {"l_max": 4},
        }
        result = resolve(data)
        from pymc_marketing.mmm.components.adstock import GeometricAdstock

        assert isinstance(result, GeometricAdstock)
