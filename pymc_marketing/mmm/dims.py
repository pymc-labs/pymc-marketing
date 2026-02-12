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
"""Dims related functionality."""

from __future__ import annotations

from typing import TypeAlias

import pymc as pm
from pytensor.tensor import TensorLike
from pytensor.xtensor.type import XTensorSharedVariable, XTensorVariable, xtensor_shared
from xarray import DataArray

XTensorLike: TypeAlias = TensorLike | XTensorVariable | DataArray


def XData(name: str, value, *, dims=None, shape=None) -> XTensorSharedVariable:
    """Register a shared_xtensor as model Data.

    This will eventually be what `pymc.dims.XData` does
    """
    if dims is not None and isinstance(dims, str):
        dims = (dims,)
    model = pm.modelcontext(None)
    x = xtensor_shared(value, dims=dims, shape=shape, name=name)
    model.register_data_var(x, dims=x.dims)
    return x
