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
import pytest

from pymc_marketing.mmm.scaling import Scaling, VariableScaling
from pymc_marketing.serialization import serialization


class TestScalingRoundtrips:
    def test_variable_scaling_roundtrip_all_parameters(self):
        original = VariableScaling(method="mean", dims=("geo", "channel"))
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is VariableScaling
        assert restored.method == "mean"
        assert restored.dims == ("geo", "channel")
        assert restored == original

    def test_scaling_roundtrip_all_parameters(self):
        original = Scaling(
            target=VariableScaling(method="max", dims="geo"),
            channel=VariableScaling(method="mean", dims=("geo", "channel")),
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is Scaling
        assert type(restored.target) is VariableScaling
        assert type(restored.channel) is VariableScaling
        assert restored.target.method == "max"
        assert restored.target.dims == ("geo",)
        assert restored.channel.method == "mean"
        assert restored.channel.dims == ("geo", "channel")
        assert restored == original


class TestFixedScaling:
    def test_fixed_scalar_construction(self):
        vs = VariableScaling(method="fixed", dims=(), value=1000.0)
        assert vs.method == "fixed"
        assert vs.value == 1000.0
        assert vs.dims == ()

    def test_fixed_dict_construction(self):
        vs = VariableScaling(
            method="fixed",
            dims=("country",),
            value={"US": 50_000, "UK": 30_000},
        )
        assert vs.method == "fixed"
        assert vs.value == {"US": 50_000, "UK": 30_000}

    def test_fixed_missing_value_raises(self):
        with pytest.raises(ValueError, match="value is required"):
            VariableScaling(method="fixed", dims=())

    def test_fixed_zero_value_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            VariableScaling(method="fixed", dims=(), value=0.0)

    def test_fixed_negative_value_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            VariableScaling(method="fixed", dims=(), value=-5.0)

    def test_fixed_dict_negative_value_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            VariableScaling(method="fixed", dims=("geo",), value={"A": 100, "B": -1})

    def test_value_forbidden_for_max(self):
        with pytest.raises(ValueError, match="must be None"):
            VariableScaling(method="max", dims=(), value=100.0)

    def test_value_forbidden_for_mean(self):
        with pytest.raises(ValueError, match="must be None"):
            VariableScaling(method="mean", dims=(), value=100.0)

    def test_roundtrip_fixed_scalar(self):
        original = VariableScaling(method="fixed", dims=(), value=42.0)
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)
        assert restored == original
        assert restored.value == 42.0

    def test_roundtrip_fixed_dict(self):
        original = VariableScaling(
            method="fixed",
            dims=("geo",),
            value={"US": 100, "UK": 200},
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)
        assert restored == original
        assert restored.value == {"US": 100, "UK": 200}

    def test_roundtrip_scaling_with_fixed(self):
        original = Scaling(
            target=VariableScaling(method="fixed", dims=(), value=50_000.0),
            channel=VariableScaling(method="fixed", dims=(), value=10_000.0),
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)
        assert restored == original


@pytest.mark.parametrize(
    "type_key",
    [
        "pymc_marketing.mmm.scaling.Scaling",
        "pymc_marketing.mmm.scaling.VariableScaling",
    ],
    ids=lambda s: s.rsplit(".", 1)[-1],
)
def test_scaling_type_registered(type_key):
    assert type_key in serialization._registry, f"{type_key} not registered"
