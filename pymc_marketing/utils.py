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

import importlib.metadata
import re
import warnings
from pathlib import Path

import xarray as xr
from pydantic import BaseModel, ConfigDict, StrictBool
from pymc.sampling import mcmc

from pymc_marketing.data.idata.io import idata_from_zarr, idata_to_zarr

__all__ = [
    "InstalledSamplers",
    "from_netcdf",
    "idata_from_zarr",
    "idata_to_zarr",
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


def _installed_version(name: str) -> tuple[int, ...] | None:
    """Return the installed version of distribution ``name``, or ``None`` if it is missing.

    This reads the package metadata instead of looking the module up, so it
    never imports the package. That matters because arviz-base registers lazy
    placeholders for ``jax`` and ``numpyro`` in ``sys.modules``, and
    ``importlib.util.find_spec`` would load them.

    The version is parsed like PyMC parses nutpie's: the leading digits of the
    first three parts, so "0.17.0rc1" becomes (0, 17, 0). A version that can't
    be read that way, such as the ``None`` returned for a damaged install with
    empty metadata, counts as missing instead of raising.
    """
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    if not isinstance(version, str):
        return None
    parts: list[int] = []
    for part in version.split(".")[:3]:
        match = re.match(r"\d+", part)
        if match is None:
            return None
        parts.append(int(match.group()))
    return tuple(parts)


class InstalledSamplers(BaseModel):
    """Optional NUTS samplers that are installed in the current environment.

    PyMC's own NUTS sampler is always available. The samplers listed here are
    optional dependencies of PyMC-Marketing, so code that wants one of them,
    such as an example notebook, can check for it first.

    When a sampler is missing, fall back to ``nuts_sampler=None`` (the
    ``pm.sample`` default) rather than ``"pymc"``: with ``None``, PyMC picks
    nutpie when it is usable and its own sampler otherwise. Because PyMC already
    does that for nutpie, this class is mostly useful for NumPyro and BlackJAX.

    Use :meth:`check` to inspect the environment. It reads package metadata
    only, so it is cheap and never imports the samplers. It sees NumPyro and
    BlackJAX when they are installed after PyMC-Marketing was imported (for
    example with ``%pip install`` in a running notebook). nutpie needs a kernel
    restart after such an install, because PyMC checks for nutpie once, when it
    is imported, and ``nutpie`` is only ``True`` once PyMC can use it.

    Parameters
    ----------
    nutpie : bool
        Whether `nutpie <https://github.com/pymc-devs/nutpie>`_ is installed at
        the minimum version PyMC requires, and PyMC has detected it.
    numpyro : bool
        Whether `NumPyro <https://num.pyro.ai>`_ and JAX are installed.
    blackjax : bool
        Whether `BlackJAX <https://blackjax-devs.github.io/blackjax/>`_ and JAX are installed.

    Examples
    --------
    Use NumPyro when it is installed and PyMC's default choice otherwise:

    .. code-block:: python

        from pymc_marketing.utils import InstalledSamplers

        installed_samplers = InstalledSamplers.check()
        installed_samplers
        # InstalledSamplers(nutpie=True, numpyro=False, blackjax=True)

        mmm.fit(
            X=X,
            y=y,
            nuts_sampler="numpyro" if installed_samplers.numpyro else None,
        )
    """

    model_config = ConfigDict(frozen=True)

    nutpie: StrictBool
    numpyro: StrictBool
    blackjax: StrictBool

    @classmethod
    def check(cls) -> "InstalledSamplers":
        """Check which of the optional samplers are installed.

        The check reads package metadata and does not import the samplers, so
        it does not load JAX. nutpie only counts as installed when PyMC has
        detected it and its version meets PyMC's minimum, since
        ``nuts_sampler="nutpie"`` raises otherwise. NumPyro and BlackJAX run on
        JAX, so they only count as installed together with JAX.

        Returns
        -------
        InstalledSamplers
            The installed status of each optional sampler.
        """
        nutpie_version = _installed_version("nutpie")
        jax_installed = _installed_version("jax") is not None
        return cls(
            nutpie=mcmc.NUTPIE_INSTALLED
            and nutpie_version is not None
            and nutpie_version >= mcmc.NUTPIE_MIN_VERSION,
            numpyro=jax_installed and _installed_version("numpyro") is not None,
            blackjax=jax_installed and _installed_version("blackjax") is not None,
        )
