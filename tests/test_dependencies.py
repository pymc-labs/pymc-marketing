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

from packaging.requirements import Requirement


def _base_requirements() -> set[str]:
    """Distribution names a plain ``pip install pymc-marketing`` pulls in (no extras)."""
    return {
        req.name
        for req in map(Requirement, importlib.metadata.requires("pymc-marketing"))
        if req.marker is None or req.marker.evaluate({"extra": ""})
    }


def test_netcdf_backend_is_a_base_dependency():
    """``ModelIO.save`` writes NetCDF4 through xarray, which needs h5netcdf and h5py (#3102).

    ArviZ 1.x stopped shipping a NetCDF backend and h5netcdf does not depend on h5py,
    so a base install only works if this package declares both itself.
    """
    missing = {"h5netcdf", "h5py"} - _base_requirements()
    assert not missing, f"NetCDF backend not in base dependencies: {sorted(missing)}"
