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
