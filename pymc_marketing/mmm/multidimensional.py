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
"""Deprecated module path; use :mod:`pymc_marketing.mmm.mmm` instead.

This module is a thin backward-compatibility shim. It re-exports the public
symbols from :mod:`pymc_marketing.mmm.mmm` and emits a single
:class:`FutureWarning` on import. It will be removed in a future release.

The class previously named ``MultiDimensionalBudgetOptimizerWrapper`` has been
renamed to :class:`~pymc_marketing.mmm.mmm.BudgetOptimizerWrapper`. The old
name is preserved as an alias inside this module only.
"""

import warnings

from pymc_marketing.mmm.mmm import (
    MMM,
    BudgetOptimizerWrapper,
    create_sample_kwargs,
)

MultiDimensionalBudgetOptimizerWrapper = BudgetOptimizerWrapper

warnings.warn(
    "`pymc_marketing.mmm.multidimensional` is deprecated and will be "
    "removed in a future release. Import from `pymc_marketing.mmm.mmm` "
    "(or directly from `pymc_marketing.mmm`) instead. The class "
    "`MultiDimensionalBudgetOptimizerWrapper` has been renamed to "
    "`BudgetOptimizerWrapper`; the old name is still available from this "
    "module as a temporary alias.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "MMM",
    "BudgetOptimizerWrapper",
    "MultiDimensionalBudgetOptimizerWrapper",
    "create_sample_kwargs",
]
