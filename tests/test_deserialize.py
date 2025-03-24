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

from pymc_marketing.deserialize import (
    DESERIALIZERS,
    DeserializableError,
    deserialize,
    register_deserialization,
)


@pytest.mark.parametrize(
    "unknown_data",
    [
        {"unknown": 1},
        {"dist": "Normal", "kwargs": {"something": "else"}},
        1,
    ],
    ids=["unknown_structure", "prior_like", "non_dict"],
)
def test_unknown_type_raises(unknown_data) -> None:
    match = "Couldn't deserialize"
    with pytest.raises(DeserializableError, match=match):
        deserialize(unknown_data)


class ArbitraryObject:
    def __init__(self, code: str):
        self.code = code
        self.value = 1


@pytest.fixture
def register_arbitrary_object():
    register_deserialization(
        is_type=lambda data: data.keys() == {"code"},
        deserialize=lambda data: ArbitraryObject(code=data["code"]),
    )

    yield

    DESERIALIZERS.pop()


def test_registration(register_arbitrary_object) -> None:
    instance = deserialize({"code": "test"})

    assert isinstance(instance, ArbitraryObject)
    assert instance.code == "test"


def test_registeration_mixup() -> None:
    data_that_looks_like_prior = {
        "dist": "Normal",
        "kwargs": {"something": "else"},
    }

    match = "Couldn't deserialize"
    with pytest.raises(DeserializableError, match=match):
        deserialize(data_that_looks_like_prior)
