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
"""Configuration I/O for PyMC-Marketing."""

import warnings

from pymc_marketing.mmm.builders.factories import build
from pymc_marketing.mmm.builders.yaml import build_mmm_from_yaml

__all__ = ["build", "build_mmm_from_yaml"]

warnings.warn(
    "The pymc_marketing.mmm.builders module is experimental and its API may change without warning.",
    UserWarning,
    stacklevel=2,
)
