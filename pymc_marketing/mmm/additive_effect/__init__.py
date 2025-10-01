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
"""
Additive effects submodule.

This submodule provides classes for modeling additive effects within the PyMC Marketing Mix Modeling (MMM) framework.

It exposes the following components:
    - Model: The main class for specifying additive effect models.
    - MuEffect: A class representing the mean effect in additive models.

Typical usage involves importing these classes to construct and analyze additive effect models in marketing analytics.
"""

from pymc_marketing.mmm.additive_effect.additive_effect import Model, MuEffect

__all__ = ["Model", "MuEffect"]
