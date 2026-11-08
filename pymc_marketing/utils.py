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

import importlib.util
import warnings
from pathlib import Path

import xarray as xr
from pydantic import BaseModel, ConfigDict

from pymc_marketing.data.idata.io import idata_from_zarr, idata_to_zarr

__all__ = [
    "InstalledSamplers",
    "from_netcdf",
    "idata_from_zarr",
    "idata_to_zarr",
    "installed_samplers",
]


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


def _is_installed(name: str) -> bool:
    """Return whether the top-level package ``name`` can be imported, without importing it."""
    return importlib.util.find_spec(name) is not None


class InstalledSamplers(BaseModel):
    """Optional NUTS samplers that are installed in the current environment.

    PyMC's own NUTS sampler (``nuts_sampler="pymc"``) is always available. The
    samplers listed here are optional dependencies of PyMC-Marketing, so code
    that wants to use one of them, such as an example notebook, can check for
    it first and fall back to ``"pymc"``.

    Use :meth:`check` to inspect the current environment, or the module-level
    ``installed_samplers`` instance, which is computed once on import.

    Parameters
    ----------
    nutpie : bool
        Whether `nutpie <https://github.com/pymc-devs/nutpie>`_ is installed.
    numpyro : bool
        Whether `NumPyro <https://num.pyro.ai>`_ is installed.
    blackjax : bool
        Whether `BlackJAX <https://blackjax-devs.github.io/blackjax/>`_ is installed.

    Examples
    --------
    Use NumPyro when it is installed and PyMC's sampler otherwise:

    .. code-block:: python

        from pymc_marketing.utils import installed_samplers

        installed_samplers
        # InstalledSamplers(nutpie=True, numpyro=False, blackjax=True)

        mmm.fit(
            X=X,
            y=y,
            nuts_sampler="numpyro" if installed_samplers.numpyro else "pymc",
        )
    """

    model_config = ConfigDict(frozen=True)

    nutpie: bool
    numpyro: bool
    blackjax: bool

    @classmethod
    def check(cls) -> "InstalledSamplers":
        """Check which of the optional samplers are installed.

        The check looks each package up without importing it, so it is cheap
        and does not pull in JAX.

        Returns
        -------
        InstalledSamplers
            The installed status of each optional sampler.
        """
        return cls(
            nutpie=_is_installed("nutpie"),
            numpyro=_is_installed("numpyro"),
            blackjax=_is_installed("blackjax"),
        )


installed_samplers = InstalledSamplers.check()
