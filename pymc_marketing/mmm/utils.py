from itertools import zip_longest

import numpy as np
import numpy.typing as npt
import pandas as pd
from pytensor import tensor as pt
from pytensor.raise_op import Assert
from pytensor.tensor.basic import get_scalar_constant_value
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.var import TensorConstant


def generate_fourier_modes(
    periods: npt.NDArray[np.float_], n_order: int
) -> pd.DataFrame:
    """Generate Fourier modes.

    Parameters
    ----------
    periods : array-like of float
        Input array denoting the period range.
    n_order : int
        Maximum order of Fourier modes.

    Returns
    -------
    pd.DataFrame
        Fourier modes (sin and cos with different frequencies) as columns in a dataframe.

    References
    ----------
    See :ref:`examples:Air_passengers-Prophet_with_Bayesian_workflow` in PyMC examples collection.
    """
    if n_order < 1:
        raise ValueError("n_order must be greater than or equal to 1")
    return pd.DataFrame(
        {
            f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
            for order in range(1, n_order + 1)
            for func in ("sin", "cos")
        }
    )


def params_broadcast_shapes(param_shapes, ndims_params, broadcastable_patterns):
    """Broadcast shape tuples except for some number of support dimensions.

    Parameters
    ==========
    param_shapes : list of ndarray or Variable
        The shapes to broadcast.
    ndims_params : list of int
        The number of dimensions for each shape that are to be considered support dimensions that
        need not broadcast together.
    broadcastable_patterns : list of bool
        The broadcastable pattern of each input shape.

    Returns
    =======
    bcast_shapes : list of ndarray
        The broadcasted values of `params`.
    """

    rev_extra_dims = tuple()
    rev_extra_broadcastable = tuple()
    for param_ind, (ndim_param, param_shape, broadcastable) in enumerate(
        zip(ndims_params, param_shapes, broadcastable_patterns)
    ):
        # Try to get concrete shapes from the start
        if isinstance(param_shape, TensorConstant):
            param_shape = param_shape.value
        # We need this in order to use `len`
        param_shape = tuple(param_shape)
        extras = tuple(param_shape[: (len(param_shape) - ndim_param)])
        extra_broadcastable = tuple(broadcastable[: (len(param_shape) - ndim_param)])

        def bcast(x, y, bcast_x, bcast_y, i):
            try:
                concrete_x = get_scalar_constant_value(x)
            except NotScalarConstantError:
                concrete_x = None
            try:
                concrete_y = get_scalar_constant_value(y)
            except NotScalarConstantError:
                concrete_y = None
            if not bcast_x and not bcast_y:
                if concrete_x is not None and concrete_y is not None:
                    if (
                        (concrete_x == 1 and not bcast_x)
                        or (concrete_y == 1 and not bcast_y)
                    ) and concrete_x != concrete_y:
                        raise ValueError(
                            f"Shape along axis {i} in the {param_ind} supplied param_shape was "
                            "tagged as not broadcastable and it was not exactly equal to the other "
                            "supplied param_shapes."
                        )
                    elif concrete_x != concrete_y:
                        raise ValueError(
                            f"Cannot broadcast shapes {concrete_x} and {concrete_y} together"
                        )
                    return (x, False)
                else:
                    assert_op = Assert(
                        f"Shape along axis {i} in the {param_ind} supplied param_shape was tagged as "
                        f"not broadcastable and it was not exactly equal to the other supplied "
                        "param_shapes."
                    )
                    return (assert_op(x, pt.eq(x, y)), False)
            elif bcast_x:
                return (y, bcast_y)
            elif bcast_y:
                return (x, bcast_x)
            else:
                return (x, bcast_x)

        temp = [
            bcast(a, b, broadcastable_a, broadcastable_b, i)
            for i, (b, a, (broadcastable_b, broadcastable_a)) in enumerate(
                zip_longest(
                    reversed(extras),
                    rev_extra_dims,
                    zip_longest(
                        reversed(extra_broadcastable),
                        rev_extra_broadcastable,
                        fillvalue=True,
                    ),
                    fillvalue=1,
                )
            )
        ]
        if len(temp) > 0:
            rev_extra_dims, rev_extra_broadcastable = list(zip(*temp))

    extra_dims = tuple(reversed(rev_extra_dims))
    extra_broadcastable = tuple(reversed(rev_extra_broadcastable))

    bcast_shapes = [
        (extra_dims + tuple(getattr(param_shape, "value", param_shape))[-ndim_param:])
        if ndim_param > 0
        else extra_dims
        for ndim_param, param_shape in zip(ndims_params, param_shapes)
    ]
    new_broadcastable_patterns = [
        (extra_broadcastable + broadcastable[-ndim_param:])
        if ndim_param > 0
        else extra_broadcastable
        for ndim_param, broadcastable in zip(ndims_params, broadcastable_patterns)
    ]

    return list(bcast_shapes), list(new_broadcastable_patterns)
