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
"""Base-install check that the default NetCDF save works without test extras.

Run with the environment Python directly so dev dependencies are not synchronized.
"""

import tempfile
from pathlib import Path

import xarray as xr

from pymc_marketing.clv.models.beta_geo import BetaGeoModel


def main() -> None:
    posterior = xr.Dataset({"alpha": ("draw", [0.5])})
    model = BetaGeoModel()
    model.idata = xr.DataTree.from_dict({"posterior": posterior})

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.nc"
        # No engine override: a base install must supply the NetCDF backend.
        model.save(str(path))
        loaded = xr.open_datatree(path)
        try:
            assert loaded["posterior"].to_dataset().equals(posterior)
        finally:
            loaded.close()


if __name__ == "__main__":
    main()
