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
"""Tests for the backward-compat shim at `pymc_marketing.mmm.multidimensional`."""

import importlib
import sys

import pytest


def _fresh_import_shim():
    sys.modules.pop("pymc_marketing.mmm.multidimensional", None)
    return importlib.import_module("pymc_marketing.mmm.multidimensional")


def test_multidimensional_shim_emits_future_warning():
    sys.modules.pop("pymc_marketing.mmm.multidimensional", None)
    with pytest.warns(FutureWarning, match="multidimensional"):
        importlib.import_module("pymc_marketing.mmm.multidimensional")


def test_multidimensional_shim_reexports_match_mmm_module():
    with pytest.warns(FutureWarning):
        shim = _fresh_import_shim()
    from pymc_marketing.mmm import mmm as mmm_module

    assert shim.MMM is mmm_module.MMM
    assert shim.BudgetOptimizerWrapper is mmm_module.BudgetOptimizerWrapper
    assert (
        shim.MultiDimensionalBudgetOptimizerWrapper is mmm_module.BudgetOptimizerWrapper
    )
    assert shim.create_sample_kwargs is mmm_module.create_sample_kwargs
