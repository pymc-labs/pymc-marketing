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
"""Read and write ``DataTree`` inference data as netCDF or Zarr.

The format of a path follows its suffix: ``.zarr`` (or an existing directory) is
a Zarr store, anything else is netCDF. :func:`save_idata` and :func:`load_idata`
are the only places that dispatch on the format; ``ModelIO`` and the MLflow
integration call them.
"""

from __future__ import annotations

import importlib
import os
import shutil
import tempfile
import warnings
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

import xarray as xr
from packaging.version import Version


class IdataFormat(StrEnum):
    """Inference-data formats :func:`save_idata` writes.

    The value is the format name accepted by ``autolog(idata_format=...)``;
    ``suffix`` is the file suffix that selects the format in a path.
    """

    NETCDF = "netcdf"
    ZARR = "zarr"

    @property
    def suffix(self) -> str:
        """File suffix of the format, for example ``".zarr"``."""
        return {IdataFormat.NETCDF: ".nc", IdataFormat.ZARR: ".zarr"}[self]


# The xarray floor of the ``zarr`` extra in pyproject.toml; a test keeps the two equal.
# DataTree.to_zarr needs it to work with zarr-python 3; reading works on older releases.
_MIN_XARRAY_FOR_WRITING_ZARR = Version("2025.6.0")


def require_zarr(*, writing: bool = False) -> None:
    """Raise ``ImportError`` with an install hint when Zarr stores cannot be used.

    Parameters
    ----------
    writing : bool, default False
        Also require the xarray release that can write Zarr 3 stores.

    Raises
    ------
    ImportError
        If ``zarr`` cannot be imported, or ``writing`` is True and xarray is
        older than the floor of the ``zarr`` extra.
    """
    try:
        importlib.import_module("zarr")
    except ImportError as err:
        raise ImportError(
            "zarr is required for Zarr stores. Install it with: pip install 'pymc-marketing[zarr]'"
        ) from err
    if writing and Version(xr.__version__) < _MIN_XARRAY_FOR_WRITING_ZARR:
        raise ImportError(
            f"Writing Zarr stores needs xarray>={_MIN_XARRAY_FOR_WRITING_ZARR}, found {xr.__version__}. "
            "Install it with: pip install 'pymc-marketing[zarr]'"
        )


def idata_format_from_path(path: str | Path) -> IdataFormat:
    """Return the inference-data format implied by ``path``.

    An existing directory or a name ending in ``.zarr`` is ``"zarr"``. Any
    other path, including one without a suffix, is ``"netcdf"``.

    Parameters
    ----------
    path : str or Path
        File or store path.

    Returns
    -------
    IdataFormat
        ``"netcdf"`` or ``"zarr"``.
    """
    path = Path(path)
    if path.is_dir():
        return IdataFormat.ZARR
    for fmt in IdataFormat:
        if path.name.endswith(fmt.suffix):
            return fmt
    return IdataFormat.NETCDF


def _select_groups(idata: xr.DataTree, groups: list[str]) -> xr.DataTree:
    """Return a copy of ``idata`` with only the requested top-level groups.

    Group names may be bare (``"posterior"``) or slash-prefixed
    (``"/posterior"``). Names not present in the tree are ignored. Root
    attributes are kept, so a filtered tree still loads through ``ModelIO.load``.

    Parameters
    ----------
    idata : xr.DataTree
        Tree to filter. Not modified.
    groups : list of str
        Groups to keep.

    Returns
    -------
    xr.DataTree
        Tree holding only ``groups``.
    """
    # DataTree paths are POSIX-style ("/posterior"), so compare paths, not names.
    wanted = {str(PurePosixPath("/", group)) for group in groups}
    return idata.drop_nodes(
        [name for name, child in idata.children.items() if child.path not in wanted]
    )


def _case_colliding_paths(idata: xr.DataTree, inherit: bool = False) -> list[str]:
    """Return the paths of variables or child groups whose names differ only by letter case.

    A Zarr directory store maps each of them to a directory, which collides on
    case-insensitive filesystems such as APFS and NTFS. ``inherit`` mirrors
    ``DataTree.to_zarr(write_inherited_coords=...)``: inherited coordinates
    only count when they are written into the child groups.
    """
    offenders: list[str] = []
    for node in idata.subtree:
        by_fold: dict[str, list[str]] = {}
        for name in (
            *map(str, node.to_dataset(inherit=inherit).variables),
            *node.children,
        ):
            by_fold.setdefault(name.casefold(), []).append(name)
        for names in by_fold.values():
            if len(names) > 1:
                offenders.extend(str(PurePosixPath(node.path) / name) for name in names)
    return offenders


def remove_idata(path: str | Path) -> None:
    """Delete a netCDF file or a Zarr store. A missing path is not an error.

    Parameters
    ----------
    path : str or Path
        File or store directory to delete.
    """
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def idata_to_zarr(
    idata: xr.DataTree,
    store: str | Path,
    groups: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """Write a ``DataTree`` to a Zarr directory store.

    Parameters
    ----------
    idata : xr.DataTree
        Tree to write.
    store : str or Path
        Store directory. It must not exist yet unless ``mode`` says otherwise.
    groups : list of str, optional
        Groups to keep. All groups are written when ``None``.
    **kwargs : Any
        Passed to :meth:`xarray.DataTree.to_zarr`.

    Raises
    ------
    ImportError
        If ``zarr`` is not installed.
    ValueError
        If the filesystem is case-insensitive and the tree holds names that
        differ only by letter case. zarr's ``ContainsArrayError`` and
        ``ContainsGroupError`` are ``ValueError`` subclasses too; this one
        names the offending variables.

    Warns
    -----
    UserWarning
        After a successful write, if any node holds names that differ only by
        letter case: the store will not load on a case-insensitive filesystem.
    """
    require_zarr(writing=True)
    from zarr.errors import ContainsArrayError, ContainsGroupError

    if groups is not None:
        idata = _select_groups(idata, groups)
    offenders = _case_colliding_paths(
        idata, inherit=kwargs.get("write_inherited_coords", False)
    )
    try:
        idata.to_zarr(store, **kwargs)
    except (ContainsArrayError, ContainsGroupError) as err:
        if not offenders:
            raise
        raise ValueError(
            "This filesystem is case-insensitive, so a Zarr store cannot hold names that differ "
            f"only by letter case: {', '.join(offenders)}. Rename one of each pair or save as netCDF."
        ) from err
    if offenders:
        warnings.warn(
            "The store will not load on a case-insensitive filesystem (macOS, Windows) because "
            f"these names differ only by letter case: {', '.join(offenders)}. Rename one of each pair.",
            UserWarning,
            skip_file_prefixes=(os.path.dirname(__file__) + os.sep,),
        )


def check_zarr_writer() -> None:
    """Write a two-value tree with a string coordinate to a temporary Zarr store.

    Version floors cannot capture every xarray, zarr and pandas combination, so
    the ``autolog`` preflight writes a tiny store instead of trusting them.

    Raises
    ------
    ImportError
        If ``zarr`` is missing or xarray is older than the floor of the ``zarr`` extra.
    RuntimeError
        If the installed xarray, zarr and pandas cannot write a Zarr store together.
    """
    # Outside the try, so a missing zarr or an old xarray stays an ImportError.
    require_zarr(writing=True)
    probe = xr.DataTree.from_dict(
        {"/posterior": xr.Dataset({"x": ("c", [0.0, 1.0])}, coords={"c": ["a", "b"]})}
    )
    with tempfile.TemporaryDirectory(
        prefix="pymc_marketing_zarr_probe_", ignore_cleanup_errors=True
    ) as tmp:
        try:
            # The probe's own zarr warnings are noise; the real write emits them anyway.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                idata_to_zarr(probe, Path(tmp) / "probe.zarr")
        except OSError:
            raise
        except Exception as err:
            raise RuntimeError(
                f"The installed xarray, zarr and pandas cannot write a Zarr store ({err}). "
                "Upgrade them with: pip install -U xarray zarr pandas"
            ) from err


def idata_from_zarr(store: str | Path) -> xr.DataTree:
    """Open a Zarr store written by :func:`idata_to_zarr` as a lazy ``DataTree``.

    Parameters
    ----------
    store : str or Path
        Store directory.

    Returns
    -------
    xr.DataTree
        Lazily loaded tree. Call ``.load()`` before deleting the store.

    Raises
    ------
    ImportError
        If ``zarr`` is not installed.
    """
    require_zarr()
    return xr.open_datatree(store, engine="zarr")


def save_idata(
    idata: xr.DataTree,
    path: str | Path,
    *,
    groups: list[str] | None = None,
    **kwargs: Any,
) -> Path:
    """Write ``idata`` in the format implied by ``path``.

    Parameters
    ----------
    idata : xr.DataTree
        Tree to write.
    path : str or Path
        ``.zarr`` or an existing directory for a Zarr store, anything else for netCDF.
    groups : list of str, optional
        Groups to keep. All groups are written when ``None``.
    **kwargs : Any
        Passed to :meth:`xarray.DataTree.to_netcdf` or :meth:`xarray.DataTree.to_zarr`.

    Returns
    -------
    Path
        ``path`` as a :class:`pathlib.Path`.

    Raises
    ------
    ImportError
        If ``path`` is a Zarr store and ``zarr`` is not installed.
    ValueError
        If a Zarr store cannot hold names that differ only by letter case on
        this filesystem; see :func:`idata_to_zarr`.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.data.idata import save_idata

        save_idata(mmm.idata, "model.nc")
        save_idata(mmm.idata, "model.zarr", groups=["posterior"])
    """
    path = Path(path)
    if groups is not None:
        idata = _select_groups(idata, groups)
    if idata_format_from_path(path) is IdataFormat.ZARR:
        idata_to_zarr(idata, path, **kwargs)
    else:
        idata.to_netcdf(str(path), **kwargs)
    return path


def load_idata(path: str | Path) -> xr.DataTree:
    """Open inference data in the format implied by ``path``.

    Parameters
    ----------
    path : str or Path
        ``.zarr`` or an existing directory for a Zarr store, anything else for netCDF.

    Returns
    -------
    xr.DataTree
        Lazily loaded tree for both backends.

    Raises
    ------
    ImportError
        If ``path`` is a Zarr store and ``zarr`` is not installed.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.data.idata import load_idata

        idata = load_idata("model.zarr").load()
    """
    path = Path(path)
    if idata_format_from_path(path) is IdataFormat.ZARR:
        return idata_from_zarr(path)
    return xr.open_datatree(str(path))
