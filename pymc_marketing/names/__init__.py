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

Each model package has its own module next to this one
(:mod:`pymc_marketing.names.mmm`, :mod:`pymc_marketing.names.clv`,
:mod:`pymc_marketing.names.customer_choice`, :mod:`pymc_marketing.names.bass`),
and its ``Attr`` and ``Group`` extend the ones here when it adds keys of its own.
The package imports nothing else from ``pymc_marketing``, so any module can import
it without a cycle.
"""

from typing import Final


class Group:
    """InferenceData groups that more than one model writes.

    :class:`~pymc_marketing.model_builder.ModelBuilder` and the experimental
    ``SCM`` both store their training data in ``FIT_DATA``.
    """

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
    """Marker keys of the serialized dictionaries.

    :mod:`pymc_marketing.serialization` writes ``TYPE`` and ``DEFERRED``, and the
    experimental terms serializer uses ``TYPE`` for its own nodes.
    :mod:`pymc_marketing.terms` wraps plain values under ``LITERAL``.
    """

    TYPE: Final = "__type__"
    DEFERRED: Final = "__deferred__"
    LITERAL: Final = "__literal__"
