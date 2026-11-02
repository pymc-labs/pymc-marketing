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
"""Vector autoregression (VAR) baseline support, built on Impulso.

`Impulso <https://github.com/QuantClimate/Impulso>`_ is an optional dependency of
PyMC-Marketing, and the VAR baseline needs ``impulso>=0.1.3``. Install it with
``pip install 'pymc-marketing[var]'``. This module imports Impulso only when it is
used, so PyMC-Marketing imports without it.
"""

from importlib import metadata
from types import ModuleType

from packaging.version import Version

# The lower bound of the impulso pins in pyproject.toml; a test keeps them in step.
_MIN_IMPULSO = "0.1.3"


def _import_impulso() -> ModuleType:
    """Import Impulso, the optional dependency of the VAR baseline.

    Impulso's version is read from its installed distribution. An Impulso without
    one, such as a source checkout put on ``sys.path``, is used unchecked, since
    its version cannot be known. A pre-release of the minimum version, such as a
    development build made before the release was tagged, is older than it and is
    refused, since it may lack what the release adds.

    Returns
    -------
    ModuleType
        The ``impulso`` module.

    Raises
    ------
    ImportError
        If Impulso is not installed, or is older than the minimum version.
    """
    try:
        import impulso
    except ImportError as exc:
        raise ImportError(
            "impulso is required for the VAR baseline. "
            "Install it with: pip install 'pymc-marketing[var]'"
        ) from exc
    try:
        installed = metadata.version("impulso")
    except metadata.PackageNotFoundError:
        return impulso
    version = Version(installed)
    if version < Version(_MIN_IMPULSO):
        prerelease = (
            f"Pre-releases of {_MIN_IMPULSO} are refused. "
            if version.is_prerelease
            and Version(version.base_version) == Version(_MIN_IMPULSO)
            else ""
        )
        raise ImportError(
            f"impulso>={_MIN_IMPULSO} is required for the VAR baseline, but "
            f"impulso {installed} is installed. {prerelease}"
            "Upgrade it with: pip install -U 'pymc-marketing[var]'"
        )
    return impulso
