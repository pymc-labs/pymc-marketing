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
"""Deserialize into a PyMC-Marketing object.

.. note::

    This module is deprecated and will be removed in a future release. Use
    :mod:`pymc_extras.deserialize` instead.

"""

import warnings

from pymc_extras.deserialize import (  # noqa: F401
    DESERIALIZERS,
    DeserializableError,
    Deserialize,
    Deserializer,
    IsType,
    deserialize,
    register_deserialization,
)

warnings.warn(
    "The pymc_marketing.deserialize module is deprecated and will be removed in a future release. "
    "Please use pymc_extras.deserialize instead.",
    DeprecationWarning,
    stacklevel=2,
)
