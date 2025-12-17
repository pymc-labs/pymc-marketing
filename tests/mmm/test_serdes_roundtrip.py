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
"""Serialization round-trip tests for MMM components after Pydantic v2 migration."""

import json

from pymc_extras.prior import Prior

from pymc_marketing.mmm import (
    GeometricAdstock,
    LogisticSaturation,
)
from pymc_marketing.mmm.events import EventEffect, GaussianBasis


def test_adstock_serialization_roundtrip() -> None:
    """Test that AdstockTransformation serializes and deserializes correctly."""
    # Create adstock with custom priors
    adstock = GeometricAdstock(
        l_max=10,
        priors={"alpha": Prior("Beta", alpha=1.5, beta=3)},
    )

    # Serialize to dict
    adstock_dict = adstock.to_dict()
    assert isinstance(adstock_dict, dict)
    assert "lookup_name" in adstock_dict
    assert adstock_dict["lookup_name"] == "geometric"

    # Serialize to JSON
    json_str = json.dumps(adstock_dict)
    assert isinstance(json_str, str)

    # Deserialize from JSON
    json_dict = json.loads(json_str)
    adstock_restored = GeometricAdstock.from_dict(json_dict)

    # Verify properties are preserved
    assert adstock_restored.l_max == adstock.l_max
    assert adstock_restored.lookup_name == adstock.lookup_name


def test_saturation_serialization_roundtrip() -> None:
    """Test that SaturationTransformation serializes and deserializes correctly."""
    # Create saturation with custom priors
    saturation = LogisticSaturation(
        priors={
            "lam": Prior("Gamma", mu=2, sigma=1),
            "beta": Prior("Gamma", mu=1.5, sigma=1),
        }
    )

    # Serialize to dict
    saturation_dict = saturation.to_dict()
    assert isinstance(saturation_dict, dict)
    assert "lookup_name" in saturation_dict
    assert saturation_dict["lookup_name"] == "logistic"

    # Serialize to JSON
    json_str = json.dumps(saturation_dict)
    assert isinstance(json_str, str)

    # Deserialize from JSON
    json_dict = json.loads(json_str)
    saturation_restored = LogisticSaturation.from_dict(json_dict)

    # Verify properties are preserved
    assert saturation_restored.lookup_name == saturation.lookup_name


def test_gaussian_basis_serialization_roundtrip() -> None:
    """Test that GaussianBasis serializes and deserializes correctly."""
    # Create Gaussian basis with custom priors
    basis = GaussianBasis(
        priors={"sigma": Prior("Gamma", mu=7, sigma=1)},
    )

    # Serialize to dict
    basis_dict = basis.to_dict()
    assert isinstance(basis_dict, dict)
    assert "lookup_name" in basis_dict
    assert basis_dict["lookup_name"] == "gaussian"

    # Serialize to JSON
    json_str = json.dumps(basis_dict)
    assert isinstance(json_str, str)

    # Deserialize from JSON
    json_dict = json.loads(json_str)
    basis_restored = GaussianBasis.from_dict(json_dict)

    # Verify properties are preserved
    assert basis_restored.lookup_name == basis.lookup_name


def test_event_effect_serialization_roundtrip() -> None:
    """Test that EventEffect serializes and deserializes correctly."""
    # Create Gaussian basis
    basis = GaussianBasis(
        priors={"sigma": Prior("Gamma", mu=7, sigma=1)},
    )

    # Create event effect
    effect_size = Prior("Normal", mu=1, sigma=1)
    event_effect = EventEffect(
        basis=basis,
        effect_size=effect_size,
        dims=("event",),
    )

    # Serialize to dict
    effect_dict = event_effect.to_dict()
    assert isinstance(effect_dict, dict)
    assert "data" in effect_dict
    assert "basis" in effect_dict["data"]
    assert "effect_size" in effect_dict["data"]

    # Serialize to JSON
    json_str = json.dumps(effect_dict)
    assert isinstance(json_str, str)

    # Deserialize from JSON
    json_dict = json.loads(json_str)
    effect_restored = EventEffect.from_dict(json_dict["data"])

    # Verify properties are preserved
    assert effect_restored.dims == event_effect.dims
    assert effect_restored.basis.lookup_name == event_effect.basis.lookup_name


def test_multiple_components_serialization_integrity() -> None:
    """Test that multiple components can be serialized together without conflicts."""
    # Create multiple components
    adstock = GeometricAdstock(l_max=10)
    saturation = LogisticSaturation()
    basis = GaussianBasis()

    # Serialize all components
    adstock_dict = adstock.to_dict()
    saturation_dict = saturation.to_dict()
    basis_dict = basis.to_dict()

    # Create combined JSON
    combined = {
        "adstock": adstock_dict,
        "saturation": saturation_dict,
        "basis": basis_dict,
    }
    json_str = json.dumps(combined)

    # Deserialize and verify integrity
    loaded = json.loads(json_str)
    assert loaded["adstock"]["lookup_name"] == "geometric"
    assert loaded["saturation"]["lookup_name"] == "logistic"
    assert loaded["basis"]["lookup_name"] == "gaussian"

    # Verify each component can be restored independently
    adstock_restored = GeometricAdstock.from_dict(loaded["adstock"])
    saturation_restored = LogisticSaturation.from_dict(loaded["saturation"])
    basis_restored = GaussianBasis.from_dict(loaded["basis"])

    assert adstock_restored.lookup_name == "geometric"
    assert saturation_restored.lookup_name == "logistic"
    assert basis_restored.lookup_name == "gaussian"
