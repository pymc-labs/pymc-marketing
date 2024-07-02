#   Copyright 2024 The PyMC Labs Developers
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

from pymc_marketing.mmm.components.base import (
    selections,
)


@pytest.mark.parametrize(
    "coords, expected",
    [
        ({}, [{}]),
        ({"channel": [1, 2, 3]}, [{"channel": 1}, {"channel": 2}, {"channel": 3}]),
        (
            {"channel": [1, 2], "country": ["A", "B"]},
            [
                {"channel": 1, "country": "A"},
                {"channel": 1, "country": "B"},
                {"channel": 2, "country": "A"},
                {"channel": 2, "country": "B"},
            ],
        ),
    ],
)
def test_selections(coords, expected) -> None:
    assert list(selections(coords)) == expected
