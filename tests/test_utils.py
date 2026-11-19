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
import importlib.metadata
import importlib.util
import sys
import warnings

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError
from pymc.sampling import mcmc

from pymc_marketing.utils import InstalledSamplers, from_netcdf


class TestFromNetcdf:
    def test_loads_inference_data(self, tmp_path):
        filepath = tmp_path / "test.nc"
        ds = xr.Dataset({"x": ("draw", np.random.randn(100))})
        idata = xr.DataTree.from_dict({"/posterior": ds})
        idata.to_netcdf(str(filepath))

        with pytest.warns(FutureWarning, match="deprecated"):
            result = from_netcdf(filepath)

        assert isinstance(result, xr.DataTree)
        assert "/posterior" in result.groups

    def test_emits_deprecation_warning(self, tmp_path):
        filepath = tmp_path / "test.nc"
        ds = xr.Dataset({"x": ("draw", np.random.randn(100))})
        idata = xr.DataTree.from_dict({"/posterior": ds})
        idata.to_netcdf(str(filepath))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from_netcdf(filepath)

        future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(future_warnings) == 1
        assert "xr.open_datatree" in str(future_warnings[0].message)


INSTALLED = {
    "nutpie": "0.16.10",
    "numpyro": "0.19.0",
    "blackjax": "1.2.5",
    "jax": "0.7.0",
}


def _fake_versions(monkeypatch, versions, nutpie_detected=True):
    """Make ``importlib.metadata.version`` report ``versions`` (``None`` = missing).

    ``nutpie_detected`` stands in for PyMC's import-time ``NUTPIE_INSTALLED``.
    """

    def version(name):
        if versions.get(name) is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(mcmc, "NUTPIE_INSTALLED", nutpie_detected)


class TestInstalledSamplers:
    @pytest.mark.parametrize("missing", ["nutpie", "numpyro", "blackjax"])
    def test_check_reports_missing_sampler(self, monkeypatch, missing):
        _fake_versions(monkeypatch, {**INSTALLED, missing: None})

        result = InstalledSamplers.check()

        expected = {name: name != missing for name in ("nutpie", "numpyro", "blackjax")}
        assert result.model_dump() == expected

    def test_check_requires_jax_for_numpyro_and_blackjax(self, monkeypatch):
        _fake_versions(monkeypatch, {**INSTALLED, "jax": None})

        assert InstalledSamplers.check().model_dump() == {
            "nutpie": True,
            "numpyro": False,
            "blackjax": False,
        }

    @pytest.mark.parametrize(
        "version, expected",
        [
            ("0.16.9", False),
            ("0.16.10", True),
            ("0.17.0rc1", True),
        ],
    )
    def test_check_requires_pymc_nutpie_version(self, monkeypatch, version, expected):
        _fake_versions(monkeypatch, {**INSTALLED, "nutpie": version})

        assert InstalledSamplers.check().nutpie is expected

    def test_check_requires_pymc_to_have_detected_nutpie(self, monkeypatch):
        # nutpie installed after PyMC was imported: the metadata is there, but
        # PyMC computed NUTPIE_INSTALLED at import time and won't use it.
        _fake_versions(monkeypatch, INSTALLED, nutpie_detected=False)

        assert InstalledSamplers.check().nutpie is False

    @pytest.mark.parametrize("name", ["nutpie", "numpyro", "blackjax", "jax"])
    @pytest.mark.parametrize("bad_version", [None, "", "dev"])
    def test_check_treats_unreadable_version_as_missing(
        self, monkeypatch, name, bad_version
    ):
        # A dist-info folder with empty METADATA makes ``version()`` return None.
        _fake_versions(monkeypatch, INSTALLED)
        fake = importlib.metadata.version

        def version(dist):
            return bad_version if dist == name else fake(dist)

        monkeypatch.setattr(importlib.metadata, "version", version)

        result = InstalledSamplers.check()

        missing = {"jax": ("numpyro", "blackjax")}.get(name, (name,))
        assert result.model_dump() == {
            s: s not in missing for s in ("nutpie", "numpyro", "blackjax")
        }

    def test_check_reports_installed_samplers(self):
        # The test extra installs all three optional samplers.
        assert InstalledSamplers.check() == InstalledSamplers(
            nutpie=True, numpyro=True, blackjax=True
        )

    def test_check_does_not_materialize_lazy_modules(self, monkeypatch, tmp_path):
        # arviz-base registers lazy placeholders for jax and numpyro in
        # ``sys.modules``. ``check()`` must leave them unloaded.
        pkg = tmp_path / "numpyro"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("raise AssertionError('numpyro was imported')")
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "numpyro", raising=False)

        spec = importlib.util.find_spec("numpyro")
        loader = importlib.util.LazyLoader(spec.loader)
        spec.loader = loader
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, "numpyro", module)
        loader.exec_module(module)

        InstalledSamplers.check()

        assert type(sys.modules["numpyro"]).__name__ == "_LazyModule"

    def test_repr(self):
        samplers = InstalledSamplers(nutpie=True, numpyro=False, blackjax=True)

        assert (
            repr(samplers)
            == "InstalledSamplers(nutpie=True, numpyro=False, blackjax=True)"
        )

    @pytest.mark.parametrize("field", ["nutpie", "numpyro", "blackjax"])
    @pytest.mark.parametrize("value", ["yes", 0, "false"])
    def test_rejects_non_bool_values(self, field, value):
        kwargs = {"nutpie": True, "numpyro": True, "blackjax": True, field: value}

        with pytest.raises(ValidationError):
            InstalledSamplers(**kwargs)
