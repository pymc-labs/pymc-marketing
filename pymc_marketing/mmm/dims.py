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

from pytensor.graph.replace import _vectorize_node
from pytensor.tensor import TensorLike
from pytensor.tensor.shape import Shape_i, shape_i
from pytensor.xtensor.type import XTensorVariable
from xarray import DataArray

# TODO: This will eventually exist in PyTensor or PyMC, remove then
XTensorLike: TypeAlias = TensorLike | XTensorVariable | DataArray


@_vectorize_node.register(Shape_i)
def _vectorize_shape_i(op: Shape_i, node, batched_x):
    from pytensor.tensor.extra_ops import broadcast_to

    [old_x] = node.inputs
    core_ndims = old_x.type.ndim
    batch_ndims = batched_x.type.ndim - core_ndims
    batched_x_shape_i = shape_i(batched_x, op.i + batch_ndims)
    if not batch_ndims:
        return [batched_x_shape_i]
    else:
        batch_shape = batched_x.shape[:batch_ndims]
        return [broadcast_to(batched_x_shape_i, batch_shape)]
