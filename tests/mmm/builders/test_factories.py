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

from pymc_marketing.mmm.builders.factories import build, locate
from pymc_marketing.mmm.mmm import MMM


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
