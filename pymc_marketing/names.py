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
"""Names shared by every model in the package.

Each name is a ``Final`` string attribute of a plain class, so ``mypy`` reports
a typo as ``attr-defined`` and the runtime value stays a plain ``str`` (safe for
PyMC, xarray, pickle and YAML). The values are part of the public API: saved
models store them, so they must not change.

Each model package keeps its own names in a ``names.py`` (the MMM's live in
:mod:`pymc_marketing.data.idata.names`), and its ``Attr`` and ``Group`` extend
the ones here when it adds keys of its own. This module imports nothing from
``pymc_marketing``.
"""

from typing import Final


class Group:
    """InferenceData groups that :class:`~pymc_marketing.model_builder.ModelBuilder` writes."""

    FIT_DATA: Final = "fit_data"


class Attr:
    """``idata.attrs`` keys shared across models.

    :class:`~pymc_marketing.model_builder.ModelBuilder` writes all of them except
    ``SERIALIZATION_VERSION``, which the MMM and
    :mod:`pymc_marketing.serialization_migration` write.
    """

    ID: Final = "id"
    MODEL_TYPE: Final = "model_type"
    VERSION: Final = "version"
    SAMPLER_CONFIG: Final = "sampler_config"
    MODEL_CONFIG: Final = "model_config"
    SERIALIZATION_VERSION: Final = "__serialization_version__"


class Key:
    """Keys of the dictionaries that :mod:`pymc_marketing.serialization` reads and writes."""

    TYPE: Final = "__type__"
    DEFERRED: Final = "__deferred__"
