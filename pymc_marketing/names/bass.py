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

Each name is a ``Final`` string attribute, so ``mypy`` reports a typo as
``attr-defined`` and the runtime value stays a plain ``str``. The values are
public API: users read these variables and saved models store them, so they
must not change.

Parameter names that are math symbols (``m``, ``p``, ``q``) are not listed:
``default_model_config`` is their single source of truth.
"""

from typing import Final


class Dim:
    """Dimension names of the Bass model."""

    T: Final = "T"


class Col:
    """DataFrame columns that :func:`~pymc_marketing.bass.data.to_bass_dataset` reads."""

    OBSERVED: Final = "observed"
    T: Final = "T"


class Var:
    """Data and model variable names that the Bass model creates and reads back."""

    OBSERVED: Final = "observed"
    ADOPTERS: Final = "adopters"
    INNOVATORS: Final = "innovators"
    IMITATORS: Final = "imitators"
    PEAK: Final = "peak"
    Y: Final = "y"
    Y_OBS: Final = "y_obs"
