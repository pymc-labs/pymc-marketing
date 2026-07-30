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

"""PyTensor rewrites for MMM optimization.

Provides a rewrite that converts
``Sum(ElemwiseMul(ExpandDims(w), X))`` to ``Dot(X, w)``
for improved performance during MCMC sampling.

The rewrite is registered with ``use_db_name_as_tag=False``, so it is
**not** included in the default ``fast_run`` optimization pipeline.
It must be explicitly enabled — see :func:`create_sampling_mode`.
"""

import pytensor.tensor as pt
from pytensor.compile import Mode, get_default_mode, optdb
from pytensor.graph.rewriting.basic import copy_stack_trace, node_rewriter
from pytensor.scalar import Mul as ScalarMul
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import Sum


@node_rewriter([Sum])
def local_sum_mul_to_dot(fgraph, node):
    """Rewrite ``Sum(ElemwiseMul(ExpandDims(w), X), axis=last) -> Dot(X, w)``.

    Detects the pattern ``(X * expand_dims(w)).sum(axis=-1)`` and replaces
    it with ``pt.dot(X, w)``. This eliminates the intermediate ``(..., C)``
    tensor and replaces the element-wise multiply + reduce with a fused
    BLAS-level operation.

    The rewrite targets the common MMM pattern where channel-level
    coefficients are broadcast via DimShuffle then summed:

        channel_contribution = (channel_data * beta).sum(dim="channel")

    becomes:

        mu = pt.dot(channel_data, beta)

    Parameters
    ----------
    fgraph : FunctionGraph
        The function graph being rewritten.
    node : Apply
        A ``Sum`` node to potentially rewrite.

    Returns
    -------
    list[Variable] | None
        A list with the new output variable, or ``None`` if the rewrite
        does not apply.
    """
    if node.op.axis is None:
        return None

    [inp] = node.inputs
    if inp.owner is None:
        return None
    if not isinstance(inp.owner.op, Elemwise):
        return None
    if not isinstance(inp.owner.op.scalar_op, ScalarMul):
        return None

    mul_inputs = inp.owner.inputs
    if len(mul_inputs) != 2:
        return None

    a, b = mul_inputs

    def _unwrap_weight(var):
        """Return the inner 1D vector if *var* is a DimShuffle of one."""
        if var.owner is not None and isinstance(var.owner.op, DimShuffle):
            inner = var.owner.inputs[0]
            if inner.type.ndim == 1:
                return inner
        return None

    weight_a = _unwrap_weight(a)
    if weight_a is not None:
        weight = weight_a
        data = b
    else:
        weight = _unwrap_weight(b)
        if weight is None:
            return None
        data = a

    # Only handle single-axis sums over the last dimension of data.
    # The weight vector (C,) maps to the trailing axis after broadcasting
    # and pt.dot(data, weight) always contracts the last axis of data
    # with the only axis of weight.
    if len(node.op.axis) != 1:
        return None
    if max(node.op.axis) != data.type.ndim - 1:
        return None

    result = pt.dot(data, weight)
    copy_stack_trace(node.outputs, result)
    return [result]


# Register explicitly — NOT included in "fast_run" by default.
# Users must include "local_sum_mul_to_dot" via Mode customization.
optdb["specialize"].register(
    "local_sum_mul_to_dot",
    local_sum_mul_to_dot,
    use_db_name_as_tag=False,
)


def create_sampling_mode(base_mode=None):
    """Create a :class:`pytensor.compile.Mode` with MMM rewrites enabled.

    Parameters
    ----------
    base_mode : Mode, optional
        Base mode to extend. Defaults to the current default mode.

    Returns
    -------
    Mode
        A mode with the rewriter ``"local_sum_mul_to_dot"`` included.

    Examples
    --------
    .. code-block:: python

        import pytensor
        from pytensor.compile import Mode, get_default_mode
        from pymc_marketing.mmm.rewrites import create_sampling_mode

        custom_mode = create_sampling_mode()
        with pytensor.config.change_flags(mode=custom_mode):
            idata = pm.sample(...)
    """
    if base_mode is None:
        base_mode = get_default_mode()
    opt_qry = base_mode.provided_optimizer.including("local_sum_mul_to_dot")
    return Mode(linker=base_mode.linker, optimizer=opt_qry)
