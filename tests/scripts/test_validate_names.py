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


@pytest.mark.parametrize(
    "expression",
    [
        'obj.drop(columns=["date"], errors="ignore")',
        'obj.drop(columns="date", errors="ignore")',
        'obj.transpose(dims=("date", "channel"), missing_dims="ignore")',
        'obj.rename(columns={"date": "ds"})',
        'obj.rename(index={"channel": "media"})',
        'obj.setdefault("channel", [])',
    ],
)
def test_mmm_dim_where_pandas_or_a_dict_ignores_a_typo_is_reported(expression) -> None:
    assert problems(expression)


@pytest.mark.parametrize(
    "expression",
    [
        'obj.drop(columns=["date"])',
        'obj.rename(columns={"date": "ds"}, errors="raise")',
        'obj.rename({"date": "ds"})',
        'obj.sum(dim="date")',
    ],
)
def test_mmm_dim_where_a_typo_raises_keeps_the_literal(expression) -> None:
    assert problems(expression) == []


@pytest.mark.parametrize(
    "import_line, constant",
    [
        ("import pymc_marketing.names.mmm as nm", "nm.Var.MU"),
        ("import pymc_marketing.names.mmm", "pymc_marketing.names.mmm.Var.MU"),
        ("import pymc_marketing.names as shared", "shared.Attr.ID"),
    ],
)
def test_constant_reached_through_a_module_import_needs_annotations(
    import_line, constant
) -> None:
    source = f"{import_line}\n\n\ndef check(obj):\n    return {constant} in obj\n"
    assert len(check_source("example.py", source, NAMES)[0]) == 1


@pytest.mark.parametrize(
    "import_line",
    ["import pymc_marketing.names.mmm as nm", "import pymc_marketing.names.mmm"],
)
def test_module_import_makes_the_names_module_own_the_file(import_line) -> None:
    source = (
        f"{import_line}\n\n\n"
        "def check(obj: object) -> bool:\n"
        '    return "channel_contribution" in obj\n'
    )
    assert len(check_source("example.py", source, NAMES)[0]) == 1
