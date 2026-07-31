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
"""Utility functions for PyMC-Marketing."""

import warnings
from pathlib import Path

import xarray as xr

from pymc_marketing.data.idata.utils import idata_from_zarr, idata_to_zarr

__all__ = ["from_netcdf", "idata_from_zarr", "idata_to_zarr"]


def from_netcdf(filepath: str | Path) -> xr.DataTree:
    """Load inference data from a netcdf file.

    .. deprecated::
        ``from_netcdf`` will be removed in a future release.
        Use ``xr.open_datatree`` directly instead.

    Parameters
    ----------
    filepath : str or Path
        The path to the netcdf file.

    Returns
    -------
    xr.DataTree
        The inference data.
    """
    warnings.warn(
        "pymc_marketing.utils.from_netcdf is deprecated and will be removed "
        "in a future release. Use xr.open_datatree directly instead.",
        FutureWarning,
        stacklevel=2,
    )
    return xr.open_datatree(filepath)
