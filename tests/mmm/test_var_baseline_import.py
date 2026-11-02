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
"""Tests for importing Impulso, the VAR baseline's optional dependency.

They run without Impulso installed, so nothing here imports it.
"""

import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path
from types import ModuleType

import pytest
from packaging.requirements import Requirement


def test_import_impulso_missing_raises(monkeypatch):
    """Without Impulso, the error names the extra and the install command."""
    from pymc_marketing.mmm.var_baseline import _import_impulso

    monkeypatch.setitem(sys.modules, "impulso", None)

    with pytest.raises(ImportError, match=r"pip install 'pymc-marketing\[var\]'"):
        _import_impulso()


def test_import_impulso_too_old_raises(monkeypatch):
    """An Impulso older than the minimum is refused with the upgrade command."""
    from pymc_marketing.mmm.var_baseline import _import_impulso

    monkeypatch.setitem(sys.modules, "impulso", ModuleType("impulso"))
    monkeypatch.setattr(metadata, "version", lambda name: "0.0.14")

    with pytest.raises(ImportError) as excinfo:
        _import_impulso()

    message = str(excinfo.value)
    assert "impulso>=0.1.3" in message
    assert "0.0.14" in message
    assert "pip install -U 'pymc-marketing[var]'" in message
    assert "Pre-releases" not in message


@pytest.mark.parametrize("installed", ["0.1.3.dev4", "0.1.3rc1"])
def test_import_impulso_prerelease_of_the_minimum_raises(monkeypatch, installed):
    """A pre-release of the minimum is refused, and the message says why.

    An editable Impulso checkout versioned before the release was tagged is one.
    """
    from pymc_marketing.mmm.var_baseline import _import_impulso

    monkeypatch.setitem(sys.modules, "impulso", ModuleType("impulso"))
    monkeypatch.setattr(metadata, "version", lambda name: installed)

    with pytest.raises(ImportError) as excinfo:
        _import_impulso()

    message = str(excinfo.value)
    assert f"impulso {installed} is installed" in message
    assert "Pre-releases of 0.1.3 are refused." in message
    assert "pip install -U 'pymc-marketing[var]'" in message


@pytest.mark.parametrize("extra", ["var", "test", "docs"])
def test_min_impulso_matches_the_pins(extra):
    """The runtime minimum is the lower bound every extra pins Impulso to."""
    from pymc_marketing.mmm.var_baseline import _MIN_IMPULSO

    pyproject = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text()
    )
    requirements = map(
        Requirement, pyproject["project"]["optional-dependencies"][extra]
    )
    (specifier,) = [r.specifier for r in requirements if r.name == "impulso"]

    assert str(specifier) == f">={_MIN_IMPULSO}"


def test_import_impulso_without_metadata_is_unchecked(monkeypatch):
    """An Impulso with no installed distribution, such as a source tree, is used."""
    from pymc_marketing.mmm.var_baseline import _import_impulso

    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    impulso = ModuleType("impulso")
    monkeypatch.setitem(sys.modules, "impulso", impulso)
    monkeypatch.setattr(metadata, "version", missing)

    assert _import_impulso() is impulso


def test_pymc_marketing_imports_without_impulso():
    """Neither the package nor the effect's module needs Impulso at import time.

    This guards two things that run on import: the eager import of the effect in
    ``pymc_marketing/mmm/__init__.py``, and the ``serialization.register`` call in
    ``var_baseline.py``. Keep it a subprocess. A module's top level runs once per
    interpreter, and the test session has already imported both modules, so an
    in-process monkeypatch of ``sys.modules`` would miss a module-level
    ``import impulso``.
    """
    code = (
        "import sys; sys.modules['impulso'] = None; "
        "import pymc_marketing.mmm; import pymc_marketing.mmm.var_baseline"
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
