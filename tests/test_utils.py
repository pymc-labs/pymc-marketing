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
import warnings

import arviz as az
import numpy as np
import pytest

from pymc_marketing.utils import from_netcdf


class TestFromNetcdf:
    def test_loads_inference_data(self, tmp_path):
        filepath = tmp_path / "test.nc"
        idata = az.from_dict(posterior={"x": np.random.randn(2, 100)})
        idata.to_netcdf(str(filepath))

        with pytest.warns(FutureWarning, match="deprecated"):
            result = from_netcdf(filepath)

        assert isinstance(result, az.InferenceData)
        assert "posterior" in result.groups()

    def test_emits_deprecation_warning(self, tmp_path):
        filepath = tmp_path / "test.nc"
        idata = az.from_dict(posterior={"x": np.random.randn(2, 100)})
        idata.to_netcdf(str(filepath))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from_netcdf(filepath)

        future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(future_warnings) == 1
        assert "arviz.from_netcdf" in str(future_warnings[0].message)
