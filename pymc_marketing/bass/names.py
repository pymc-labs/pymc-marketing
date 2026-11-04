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
"""Names of the Bass diffusion model: its time dimension, input columns and model variables.

Each name is a member of a :class:`~pymc_marketing.names.Name` enum, so ``mypy``
reports a typo as ``attr-defined``; see :class:`~pymc_marketing.names.Name` for
how members behave when saved and loaded. The values are public API: users read
these variables and saved models store them, so they must not change.

Parameter names that are math symbols (``m``, ``p``, ``q``) are not listed:
``default_model_config`` is their single source of truth.
"""

from enum import unique

from pymc_marketing.names import Name


@unique
class Dim(Name):
    """Dimension names of the Bass model."""

    T = "T"


@unique
class Col(Name):
    """DataFrame columns that :func:`~pymc_marketing.bass.data.to_bass_dataset` reads."""

    OBSERVED = "observed"
    T = "T"


@unique
class Var(Name):
    """Data and model variable names that the Bass model creates and reads back."""

    OBSERVED = "observed"
    ADOPTERS = "adopters"
    INNOVATORS = "innovators"
    IMITATORS = "imitators"
    PEAK = "peak"
    Y = "y"
    Y_OBS = "y_obs"
