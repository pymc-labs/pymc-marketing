from itertools import zip_longest

import numpy as np
import numpy.typing as npt
import pandas as pd
from pytensor import tensor as pt
from pytensor.raise_op import Assert
from pytensor.tensor.basic import get_scalar_constant_value
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.math import maximum
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


def params_broadcast_shapes(param_shapes, ndims_params, use_pytensor=True):
    """Broadcast shape tuples except for some number of support dimensions.

    Parameters
    ==========
    param_shapes : list of ndarray or Variable
        The shapes to broadcast.
    ndims_params : list of int
        The number of dimensions for each shape that are to be considered support dimensions that
        need not broadcast together.
    use_pytensor : bool
        If ``True``, use PyTensor maximum Op; otherwise, use NumPy.

    Returns
    =======
    bcast_shapes : list of ndarray
        The broadcasted values of `params`.
    """
    max_fn = maximum if use_pytensor else np.maximum

    rev_extra_dims = []
    for param_ind, (ndim_param, param_shape) in enumerate(
        zip(ndims_params, param_shapes)
    ):
        # Try to get concrete shapes from the start
        if isinstance(param_shape, TensorConstant):
            param_shape = param_shape.value
        # We need this in order to use `len`
        param_shape = tuple(param_shape)
        extras = tuple(param_shape[: (len(param_shape) - ndim_param)])

        def max_bcast(x, y, i):
            assert_op = Assert(
                f"Failed to broadcast dynamically set shapes along axis {i} "
                f"in the {param_ind} supplied param_shape"
            )
            try:
                concrete_x = get_scalar_constant_value(x)
            except NotScalarConstantError:
                concrete_x = None
            try:
                concrete_y = get_scalar_constant_value(y)
            except NotScalarConstantError:
                concrete_y = None
            if concrete_x == 1:
                return y
            if concrete_y == 1:
                return x
            if concrete_x is not None and concrete_y is not None:
                if concrete_x == concrete_y:
                    return x
                raise ValueError(
                    f"Cannot broadcast shapes {concrete_x} and {concrete_y} together"
                )
            return assert_op(
                max_fn(x, y),
                pt.or_(
                    pt.eq(x, 1),
                    pt.or_(
                        pt.eq(y, 1),
                        pt.eq(x, y),
                    ),
                ),
            )

        rev_extra_dims = [
            max_bcast(a, b, i)
            for i, (b, a) in enumerate(
                zip_longest(reversed(extras), rev_extra_dims, fillvalue=1)
            )
        ]

    extra_dims = tuple(reversed(rev_extra_dims))

    bcast_shapes = [
        (extra_dims + tuple(getattr(param_shape, "value", param_shape))[-ndim_param:])
        if ndim_param > 0
        else extra_dims
        for ndim_param, param_shape in zip(ndims_params, param_shapes)
    ]

    return bcast_shapes
