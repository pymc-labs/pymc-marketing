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

from typing import TypeAlias

from pytensor.tensor import TensorLike
from pytensor.xtensor.type import XTensorVariable
from xarray import DataArray

# TODO: This will eventually exist in PyTensor or PyMC, remove then
XTensorLike: TypeAlias = TensorLike | XTensorVariable | DataArray
