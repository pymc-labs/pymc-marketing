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
"""Names shared by every model in the package, and the :class:`Name` base class.

Every library-owned name (model variable, dimension, fixed column, InferenceData
group, ``idata.attrs`` key, serialization key) is a member of a :class:`Name`
subclass, so ``mypy`` reports a typo as ``attr-defined``. The values are part of
the public API: saved models store them, so they must not change.

Each model package keeps its own names in a ``names.py`` (the MMM's live in
:mod:`pymc_marketing.data.idata.names`). This module imports nothing from
``pymc_marketing``.
"""

from collections.abc import Iterator
from enum import StrEnum, unique
from typing import Any

import yaml


class Name(StrEnum):
    """Base class of every library-owned name.

    A member is a ``str``: it compares, hashes and formats like its value, so it
    can be passed anywhere a name is expected (PyMC variable names, dims and
    coords, xarray keys, DataFrame columns, ``idata.attrs`` keys).

    Save and load contract:

    * ``repr`` is the ``repr`` of the value, so error messages, ``list(ds)`` and
      the text of PyMC and xarray stay exactly as with plain strings.
    * ``pickle`` and ``cloudpickle`` store the plain value: unpickling gives a
      ``str`` and does not need this class (MLflow artifacts, multiprocessing).
      The pickled bytes still differ from those of the plain string.
    * YAML: every PyYAML dumper, pure Python and libyaml, writes the plain value.
      Importing this module registers the representer process-wide, on every
      PyYAML representer class that exists at that time.
    * JSON, Zarr and netCDF names, dims and attribute keys are written as plain
      strings. netCDF attribute *values* must be exact ``str`` for h5py, so
      :func:`~pymc_marketing.data.idata.io.save_idata` converts members there.
      After any reload, every name is a plain ``str``.

    Differences from a plain ``str`` while a session lives:

    * ``type(member) is str`` is ``False`` and ``isinstance(member, Enum)`` is
      ``True``. PyMC models and InferenceData built in the session keep the
      member as the key. Keys created by pydantic validation are plain ``str``;
      pandas keeps the member in labels unless it stores strings with pyarrow.
    * ``member.name`` is the attribute name (``"CHANNEL_CONTRIBUTION"``) and
      ``member.value`` is the string.
    * ``copy.deepcopy`` keeps the member, ``pickle`` returns the ``str``.
    * ``sys.intern(member)`` raises ``TypeError``.
    * xarray cannot write an object array that mixes members with other strings
      ("mixed native types"), for example DataFrame labels turned into values by
      ``melt`` or ``stack`` without pyarrow. Convert it with ``.astype(str)``.
    """

    def __repr__(self) -> str:
        """Return the ``repr`` of the value, as for a plain ``str``."""
        return repr(self._value_)

    def __reduce_ex__(self, protocol: Any) -> tuple[type[str], tuple[str]]:
        """Pickle as the plain value, so unpickling does not need this class."""
        return str, (self._value_,)


def _represent_name(dumper: yaml.BaseDumper, data: Name) -> yaml.ScalarNode:
    return dumper.represent_str(data._value_)


def _representer_classes(
    cls: type[yaml.representer.SafeRepresenter],
) -> Iterator[type[yaml.representer.SafeRepresenter]]:
    """Yield every subclass of ``cls`` that keeps its own representer table."""
    subclass: type[yaml.representer.SafeRepresenter]
    for subclass in cls.__subclasses__():
        if "yaml_multi_representers" in vars(subclass):
            yield subclass
        yield from _representer_classes(subclass)


# ``add_multi_representer`` copies the table into the class it is called on.
# Registering on ``SafeRepresenter`` covers every dumper that inherits its table;
# the classes that already hold their own table (``Representer`` and the
# dumpers built on it, and any subclass created before this import) are
# registered one by one. Classes created later inherit or copy the registration.
yaml.representer.SafeRepresenter.add_multi_representer(Name, _represent_name)
for _representer in set(_representer_classes(yaml.representer.SafeRepresenter)):
    _representer.add_multi_representer(Name, _represent_name)


@unique
class Group(Name):
    """InferenceData groups that :class:`~pymc_marketing.model_builder.ModelBuilder` writes."""

    FIT_DATA = "fit_data"


@unique
class Attr(Name):
    """``idata.attrs`` keys shared across models.

    :class:`~pymc_marketing.model_builder.ModelBuilder` writes all of them except
    ``SERIALIZATION_VERSION``, which the MMM and
    :mod:`pymc_marketing.serialization_migration` write.
    """

    ID = "id"
    MODEL_TYPE = "model_type"
    VERSION = "version"
    SAMPLER_CONFIG = "sampler_config"
    MODEL_CONFIG = "model_config"
    SERIALIZATION_VERSION = "__serialization_version__"


@unique
class Key(Name):
    """Keys of the dictionaries that :mod:`pymc_marketing.serialization` reads and writes."""

    TYPE = "__type__"
    DEFERRED = "__deferred__"
