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
"""Save/load contract of the library-owned names in the ``names`` modules.

Every name crosses the same boundaries a plain string does: reprs in messages,
pickle (MLflow artifacts, multiprocessing), YAML, netCDF and Zarr. Each
parametrized test runs over every name or class of every ``names`` module, so a
new name or a new class is covered without editing this file.
"""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path

import cloudpickle
import numpy as np
import pytest
import xarray as xr
import yaml

import pymc_marketing
from pymc_marketing.data.idata import load_idata, save_idata
from pymc_marketing.names import Name

PACKAGE_ROOT = Path(pymc_marketing.__file__).parent


def _names() -> list[tuple[str, type, str]]:
    """Return ``(qualified attribute, class, name)`` for every name of every ``names`` module."""
    found = []
    for path in sorted(PACKAGE_ROOT.rglob("names.py")):
        module_name = ".".join(
            ("pymc_marketing", *path.relative_to(PACKAGE_ROOT).with_suffix("").parts)
        )
        module = importlib.import_module(module_name)
        for cls_name, cls in vars(module).items():
            if not isinstance(cls, type) or cls.__module__ != module_name:
                continue
            for attr in dir(cls):
                value = getattr(cls, attr)
                if attr.isupper() and isinstance(value, str):
                    found.append((f"{module_name}.{cls_name}.{attr}", cls, value))
    return found


NAMES = _names()
NAME_IDS = [qualified for qualified, _, _ in NAMES]
NAME_VALUES = [name for _, _, name in NAMES]
NAME_CLASSES = sorted(
    {cls for _, cls, _ in NAMES}, key=lambda cls: f"{cls.__module__}.{cls.__name__}"
)

YAML_DUMPERS = [yaml.SafeDumper, yaml.Dumper]
if yaml.__with_libyaml__:
    YAML_DUMPERS += [yaml.CSafeDumper, yaml.CDumper]


def test_names_are_found():
    assert "pymc_marketing.names.Group.FIT_DATA" in NAME_IDS
    assert "pymc_marketing.data.idata.names.Var.CHANNEL_CONTRIBUTION" in NAME_IDS


@pytest.mark.parametrize(
    "cls",
    NAME_CLASSES,
    ids=[f"{cls.__module__}.{cls.__name__}" for cls in NAME_CLASSES],
)
def test_name_classes_are_unique_name_enums(cls):
    """A plain ``StrEnum`` would lose the save/load contract, and without ``@unique``
    a repeated value silently becomes an alias of the first name."""
    assert issubclass(cls, Name)
    assert len(cls.__members__) == len(cls)


@pytest.mark.parametrize("name", NAME_VALUES, ids=NAME_IDS)
def test_repr_is_the_plain_string_repr(name):
    assert repr(name) == repr(f"{name}")
    assert repr([name]) == repr([f"{name}"])
    assert f"{name!r}" == repr(f"{name}")


@pytest.mark.parametrize("name", NAME_VALUES, ids=NAME_IDS)
@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_stores_the_plain_string(name, protocol):
    payload = pickle.dumps(name, protocol=protocol)

    restored = pickle.loads(payload)  # noqa: S301

    assert type(restored) is str
    assert restored == name
    assert b"pymc_marketing" not in payload


@pytest.mark.parametrize("name", NAME_VALUES, ids=NAME_IDS)
def test_cloudpickle_stores_the_plain_string(name):
    payload = cloudpickle.dumps({name: [name]})

    restored = cloudpickle.loads(payload)

    assert [type(key) for key in restored] == [str]
    assert type(restored[name][0]) is str
    assert b"pymc_marketing" not in payload


@pytest.mark.parametrize("dumper", YAML_DUMPERS, ids=lambda d: d.__name__)
@pytest.mark.parametrize("name", NAME_VALUES, ids=NAME_IDS)
def test_yaml_writes_the_plain_string(name, dumper):
    plain = f"{name}"

    text = yaml.dump({name: [name]}, Dumper=dumper)

    assert text == yaml.dump({plain: [plain]}, Dumper=dumper)
    assert yaml.safe_load(text) == {plain: [plain]}


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_saved_idata_holds_plain_strings(tmp_path, suffix):
    """Names as variable names, dims, coords, attribute keys and attribute values."""
    from pymc_marketing.data.idata.names import Attr, Dim, Group, Var

    if suffix == ".zarr":
        pytest.importorskip("zarr")
    dataset = xr.Dataset(
        {
            Var.CHANNEL_CONTRIBUTION: (
                (Dim.DATE, Dim.CHANNEL),
                np.ones((2, 2)),
                {"var": Var.Y},
            )
        },
        coords={Dim.DATE: [0, 1], Dim.CHANNEL: [Var.CHANNEL_DATA, Var.TARGET_DATA]},
        attrs={Attr.LINK: Var.MU, "names": [Var.Y, Var.MU]},
    )
    idata = xr.DataTree.from_dict({f"/{Group.FIT_DATA}": dataset})
    idata.attrs[Attr.MODEL_CONFIG] = Var.Y

    loaded = load_idata(save_idata(idata, tmp_path / f"model{suffix}"))

    group = loaded[Group.FIT_DATA]
    variable = group[Var.CHANNEL_CONTRIBUTION]
    assert {
        type(key)
        for key in [*group.data_vars, *variable.dims, *group.attrs, *loaded.attrs]
    } == {str}
    assert loaded.attrs == {"model_config": "y"}
    assert type(loaded.attrs["model_config"]) is str
    assert group.attrs["link"] == "mu"
    assert list(group.attrs["names"]) == ["y", "mu"]
    assert variable.attrs == {"var": "y"}
    assert list(group["channel"].values) == ["channel_data", "target_data"]
    # The tree in memory is not modified by saving it.
    assert idata[Group.FIT_DATA].attrs[Attr.LINK] is Var.MU
