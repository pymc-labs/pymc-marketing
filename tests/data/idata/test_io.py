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
"""Tests for netCDF/Zarr inference-data I/O."""

import sys
import tempfile
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from packaging.requirements import Requirement

from pymc_marketing.data.idata.io import (
    _MIN_XARRAY_FOR_WRITING_ZARR,
    IdataFormat,
    _select_groups,
    check_zarr_writer,
    idata_format_from_path,
    idata_to_zarr,
    load_idata,
    remove_idata,
    require_zarr,
    save_idata,
)


def _filesystem_is_case_insensitive() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "CaseProbe").touch()
        return (Path(tmp) / "caseprobe").exists()


CASE_INSENSITIVE_FS = _filesystem_is_case_insensitive()


@pytest.fixture
def idata() -> xr.DataTree:
    rng = np.random.default_rng(42)
    posterior = xr.Dataset(
        {"mu": (("chain", "draw"), rng.normal(size=(2, 5)))},
        coords={"chain": [0, 1], "draw": np.arange(5)},
    )
    sample_stats = xr.Dataset({"lp": (("chain", "draw"), rng.normal(size=(2, 5)))})
    tree = xr.DataTree.from_dict(
        {"/posterior": posterior, "/sample_stats": sample_stats}
    )
    tree.attrs["model_id"] = "abc123"
    return tree


@pytest.mark.parametrize("path_factory", [str, Path], ids=["str", "path"])
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.nc", "netcdf"),
        ("model.zarr", "zarr"),
        ("model", "netcdf"),
        ("model.h5", "netcdf"),
    ],
)
def test_idata_format_from_path_uses_suffix(
    tmp_path, path_factory, name, expected
) -> None:
    assert idata_format_from_path(path_factory(tmp_path / name)) == expected


def test_idata_format_from_path_treats_existing_directory_as_zarr(tmp_path) -> None:
    store = tmp_path / "no_suffix"
    store.mkdir()
    assert idata_format_from_path(store) == "zarr"


def test_idata_format_members_carry_their_suffix() -> None:
    assert IdataFormat("zarr") is IdataFormat.ZARR
    assert IdataFormat.ZARR.suffix == ".zarr"
    assert IdataFormat.NETCDF.suffix == ".nc"
    assert f"idata{IdataFormat.ZARR.suffix}" == "idata.zarr"
    with pytest.raises(ValueError, match="parquet"):
        IdataFormat("parquet")


def test_select_groups_ignores_nested_or_unknown_paths(idata) -> None:
    selected = _select_groups(idata, ["posterior/nested", "missing"])

    assert set(selected.children) == set()
    assert selected.attrs == {"model_id": "abc123"}


def test_require_zarr_names_the_extra_when_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "zarr", None)
    with pytest.raises(ImportError, match=r"pymc-marketing\[zarr\]"):
        require_zarr()


@pytest.mark.parametrize(
    "groups",
    [["posterior"], ["/posterior"], ["posterior/"]],
    ids=["bare", "slash", "trailing-slash"],
)
def test_select_groups_keeps_requested_group_and_root_attrs(idata, groups) -> None:
    selected = _select_groups(idata, groups)

    assert set(selected.children) == {"posterior"}
    assert selected.attrs == {"model_id": "abc123"}
    assert set(idata.children) == {"posterior", "sample_stats"}  # input untouched


@pytest.mark.parametrize("suffix", [".nc", ".zarr"], ids=["netcdf", "zarr"])
def test_save_and_load_round_trip(idata, tmp_path, suffix) -> None:
    path = save_idata(idata, tmp_path / f"idata{suffix}")

    loaded = load_idata(path).load()

    assert path == tmp_path / f"idata{suffix}"
    xr.testing.assert_identical(loaded, idata)


@pytest.mark.parametrize("suffix", [".nc", ".zarr"], ids=["netcdf", "zarr"])
def test_save_idata_with_groups_keeps_root_attrs(idata, tmp_path, suffix) -> None:
    path = save_idata(idata, tmp_path / f"idata{suffix}", groups=["posterior"])

    loaded = load_idata(path)

    assert set(loaded.children) == {"posterior"}
    assert loaded.attrs["model_id"] == "abc123"


def test_save_idata_forwards_netcdf_kwargs(idata, tmp_path) -> None:
    with pytest.raises(ValueError, match="engine"):
        save_idata(idata, tmp_path / "idata.nc", engine="not-a-backend")


def test_save_idata_forwards_zarr_kwargs(idata, tmp_path) -> None:
    path = save_idata(idata, tmp_path / "idata.zarr")

    with pytest.raises(FileExistsError):
        save_idata(idata, path)

    save_idata(idata, path, mode="w")
    xr.testing.assert_identical(load_idata(path).load(), idata)


def test_idata_to_zarr_with_groups_keeps_root_attrs(idata, tmp_path) -> None:
    store = tmp_path / "store"  # no suffix: a bare store name must still be Zarr
    idata_to_zarr(idata, store, groups=["/posterior"])

    loaded = load_idata(store)

    assert set(loaded.children) == {"posterior"}
    assert loaded.attrs["model_id"] == "abc123"


def test_idata_to_zarr_reports_case_colliding_names(tmp_path) -> None:
    """Case-insensitive filesystems get an error naming the pair; others get a portability warning."""
    constant_data = xr.Dataset({"t": ("T", np.arange(3))}, coords={"T": np.arange(3)})
    tree = xr.DataTree.from_dict({"/constant_data": constant_data})
    store = tmp_path / "idata.zarr"

    if CASE_INSENSITIVE_FS:
        with pytest.raises(ValueError, match=r"letter case.*constant_data/T"):
            idata_to_zarr(tree, store)
    else:
        with pytest.warns(UserWarning, match=r"letter case.*constant_data/T"):
            idata_to_zarr(tree, store)
        assert set(load_idata(store)["constant_data"].variables) == {"t", "T"}


def test_idata_to_zarr_ignores_inherited_coordinates(tmp_path) -> None:
    """A root coordinate is not written into child groups, so it cannot collide there."""
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"T": np.arange(3)}),
            "/constant_data": xr.Dataset({"t": ("T", np.arange(3))}),
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        idata_to_zarr(tree, tmp_path / "idata.zarr")

    assert not [w for w in caught if "letter case" in str(w.message)]


def test_idata_to_zarr_checks_inherited_coordinates_when_written(tmp_path) -> None:
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"T": np.arange(3)}),
            "/constant_data": xr.Dataset({"t": ("T", np.arange(3))}),
        }
    )
    store = tmp_path / "idata.zarr"

    if CASE_INSENSITIVE_FS:
        with pytest.raises(ValueError, match=r"letter case.*constant_data/T"):
            idata_to_zarr(tree, store, write_inherited_coords=True)
    else:
        with pytest.warns(UserWarning, match=r"letter case.*constant_data/T"):
            idata_to_zarr(tree, store, write_inherited_coords=True)


@pytest.mark.skipif(CASE_INSENSITIVE_FS, reason="the write fails before it can warn")
def test_case_collision_warning_points_at_the_caller(idata, tmp_path) -> None:
    tree = xr.DataTree.from_dict(
        {
            "/constant_data": xr.Dataset(
                {"t": ("T", np.arange(3))}, coords={"T": np.arange(3)}
            )
        }
    )

    with pytest.warns(UserWarning, match="letter case") as record:
        save_idata(tree, tmp_path / "idata.zarr")

    [collision] = [w for w in record if "letter case" in str(w.message)]
    assert collision.filename == __file__


def test_require_zarr_writing_rejects_old_xarray(monkeypatch) -> None:
    monkeypatch.setattr(xr, "__version__", "2025.4.0")
    with pytest.raises(ImportError, match=r"xarray>=2025\.6\.0.*2025\.4\.0"):
        require_zarr(writing=True)


def test_require_zarr_reading_accepts_old_xarray(monkeypatch) -> None:
    monkeypatch.setattr(xr, "__version__", "2025.4.0")
    require_zarr()


def test_xarray_writer_floor_matches_the_zarr_extra() -> None:
    """The constant in io.py and the ``zarr`` extra in pyproject.toml must agree."""
    pyproject = tomllib.loads(
        (Path(__file__).parents[3] / "pyproject.toml").read_text()
    )
    requirements = [
        Requirement(entry)
        for entry in pyproject["project"]["optional-dependencies"]["zarr"]
    ]
    [xarray_requirement] = [r for r in requirements if r.name == "xarray"]
    [floor] = [s for s in xarray_requirement.specifier if s.operator == ">="]

    assert floor.version == str(_MIN_XARRAY_FOR_WRITING_ZARR)


def test_check_zarr_writer_passes_quietly() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_zarr_writer()


def test_check_zarr_writer_names_the_failing_write(monkeypatch) -> None:
    def broken_to_zarr(self, *args, **kwargs):
        raise TypeError("Cannot interpret StringDtype")

    monkeypatch.setattr(xr.DataTree, "to_zarr", broken_to_zarr)
    with pytest.raises(RuntimeError, match=r"cannot write a Zarr store.*StringDtype"):
        check_zarr_writer()


def test_check_zarr_writer_reports_old_xarray_as_import_error(monkeypatch) -> None:
    monkeypatch.setattr(xr, "__version__", "2025.4.0")
    with pytest.raises(ImportError, match=r"xarray>=2025\.6\.0"):
        check_zarr_writer()


def test_check_zarr_writer_does_not_relabel_os_errors(monkeypatch) -> None:
    def disk_full(self, *args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(xr.DataTree, "to_zarr", disk_full)
    with pytest.raises(OSError, match="No space left"):
        check_zarr_writer()


@pytest.mark.parametrize("suffix", [".nc", ".zarr"], ids=["netcdf", "zarr"])
def test_remove_idata_deletes_file_or_store_and_tolerates_absence(
    idata, tmp_path, suffix
) -> None:
    path = save_idata(idata, tmp_path / f"idata{suffix}")

    remove_idata(path)
    remove_idata(path)

    assert not path.exists()


def test_moved_helpers_still_importable_from_utils_with_a_warning() -> None:
    from pymc_marketing.data.idata import io, utils

    with pytest.warns(FutureWarning, match="pymc_marketing.data.idata.io"):
        assert utils.idata_to_zarr is io.idata_to_zarr
    with pytest.raises(AttributeError):
        utils.not_a_helper
