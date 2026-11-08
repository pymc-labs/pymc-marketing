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
import sys
import warnings

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from pymc_marketing.utils import InstalledSamplers, from_netcdf, installed_samplers


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


class TestInstalledSamplers:
    @pytest.mark.parametrize("missing", ["nutpie", "numpyro", "blackjax"])
    def test_check_reports_missing_sampler(self, monkeypatch, missing):
        # A ``None`` entry in ``sys.modules`` makes the package unimportable.
        monkeypatch.setitem(sys.modules, missing, None)

        result = InstalledSamplers.check()

        assert getattr(result, missing) is False

    def test_check_reports_installed_samplers(self):
        # The test extra installs all three optional samplers.
        assert InstalledSamplers.check() == InstalledSamplers(
            nutpie=True, numpyro=True, blackjax=True
        )

    def test_check_does_not_import_samplers(self, monkeypatch):
        for name in ("nutpie", "numpyro", "blackjax"):
            monkeypatch.delitem(sys.modules, name, raising=False)

        InstalledSamplers.check()

        assert not {"nutpie", "numpyro", "blackjax"} & sys.modules.keys()

    def test_module_level_instance(self):
        assert isinstance(installed_samplers, InstalledSamplers)
        assert installed_samplers == InstalledSamplers.check()

    def test_repr(self):
        samplers = InstalledSamplers(nutpie=True, numpyro=False, blackjax=True)

        assert (
            repr(samplers)
            == "InstalledSamplers(nutpie=True, numpyro=False, blackjax=True)"
        )

    def test_is_frozen(self):
        with pytest.raises(ValidationError, match="frozen"):
            installed_samplers.numpyro = not installed_samplers.numpyro
