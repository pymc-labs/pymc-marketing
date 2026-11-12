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
"""Tests for the validate-names pre-commit hook."""

import sys
from pathlib import Path

import pytest

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from validate_names import check_source, read_names

NAMES = read_names()


def problems(expression: str, relative: str = "mmm/example.py") -> list[str]:
    source = f"def check(obj: object) -> object:\n    return {expression}\n"
    return check_source(relative, source, NAMES)[0]


@pytest.mark.parametrize(
    "expression, relative",
    [
        ('hasattr(obj, "sensitivity_analysis")', "mmm/example.py"),
        ('hasattr(obj, "cv_metadata")', "mmm/example.py"),
        ('getattr(obj, "cv_metadata", None)', "mmm/example.py"),
        ('hasattr(obj, "fit_data")', "bass/example.py"),
        ('hasattr(obj, "channel_contribution")', "mmm/example.py"),
        ('hasattr(obj, "customer_id")', "clv/example.py"),
    ],
)
def test_group_variable_or_column_as_an_attribute_name_is_reported(
    expression, relative
) -> None:
    assert len(problems(expression, relative)) == 1


@pytest.mark.parametrize(
    "expression",
    [
        'getattr(obj, "mu_effects", [])',
        'hasattr(obj, "adstock")',
        'getattr(obj, "model_config")',
    ],
)
def test_attrs_key_as_an_attribute_name_is_an_object_attribute(expression) -> None:
    assert problems(expression) == []


def test_value_that_is_also_a_dim_gets_no_attribute_exemption() -> None:
    # "covariates" is both Attr.COVARIATES and Dim.COVARIATES of customer choice.
    expression = 'hasattr(obj, "covariates")'
    assert len(problems(expression, "customer_choice/example.py")) == 1


@pytest.mark.parametrize(
    "expression", ['hasattr(obj, "date")', 'getattr(obj, "date", None)']
)
def test_mmm_dim_in_a_silent_attribute_lookup_is_reported(expression) -> None:
    assert len(problems(expression)) == 1


def test_mmm_dim_in_getattr_without_default_raises_on_a_typo() -> None:
    assert problems('getattr(obj, "date")') == []
