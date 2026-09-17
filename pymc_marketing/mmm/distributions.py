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
"""Probability distributions for marketing mix models.

:class:`WeightedZeroSumNormal` is a normal distribution whose last axis is
constrained to a weighted zero sum, ``sum(weights * x) = 0``. It generalises
:class:`pymc.ZeroSumNormal` to unequal weights, such as spend shares, and is
provided for the tensor API (:class:`WeightedZeroSumNormal`) and the
:mod:`pymc.dims` API (:class:`DimWeightedZeroSumNormal`), following the same
split pymc uses for its own distributions.

The distribution is incubating here before a proposed move to pymc-extras or
pymc; its import path may change.
"""

from collections.abc import Sequence

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytensor.xtensor as ptx
from numpy.typing import (
    ArrayLike,  # noqa: F401  # resolves pt.TensorLike's ForwardRef('ArrayLike') for sphinx_autodoc_typehints (#1197)
)
from pymc.dims.distributions.core import VectorDimDistribution
from pymc.dims.distributions.transforms import DimTransform
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import (
    Distribution,
    SymbolicRandomVariable,
    _support_point,
)
from pymc.distributions.shape_utils import (
    Dims,
    get_support_shape_1d,
    rv_size_is_none,
    to_tuple,
)
from pymc.distributions.transforms import _default_transform
from pymc.exceptions import NotConstantValueError
from pymc.logprob.abstract import _logprob
from pymc.logprob.transforms import Transform
from pymc.pytensorf import constant_fold, normalize_rng_param
from pymc.util import UNSET
from pytensor.graph.basic import Constant, Variable
from pytensor.raise_op import Assert
from pytensor.tensor import TensorConstant, TensorLike, TensorVariable
from pytensor.tensor.random.utils import normalize_size_param
from pytensor.xtensor import as_xtensor
from pytensor.xtensor import random as pxr
from pytensor.xtensor.type import XTensorConstant, XTensorVariable

__all__ = [
    "DimWeightedZeroSumNormal",
    "DimWeightedZeroSumTransform",
    "WeightedZeroSumNormal",
    "WeightedZeroSumTransform",
]


def _unit_direction(weights: TensorLike) -> TensorVariable:
    """``u = weights / |weights|`` along the last axis.

    Shared by the random graph, the logp and the transforms so that all of
    them agree on ``u`` to machine precision: the logp checks that values lie
    on the constraint hyperplane up to a tight relative tolerance.
    """
    weights = pt.as_tensor(weights)
    return weights / pt.sqrt(pt.sum(weights**2, axis=-1, keepdims=True))


def _validate_weights(weights: TensorLike) -> TensorVariable:
    """Cast weights to a 1-d floatX tensor and reject non-positive constants."""
    weights = pt.as_tensor(weights).astype("floatX")
    if weights.type.ndim != 1:
        raise ValueError("weights must be a 1-d vector")
    if isinstance(weights, TensorConstant) and np.any(weights.data <= 0):
        raise ValueError("weights must be strictly positive")
    return weights


def _check_length(weights: TensorVariable, expected: TensorLike) -> TensorVariable:
    """Check ``len(weights)`` against an expected length.

    Statically when both are known now, otherwise with a runtime ``Assert``
    so a length that changes later (a resized shared coordinate, new data)
    is still caught.
    """
    expected = pt.as_tensor(expected)
    static_n = weights.type.shape[0]
    try:
        (expected_n,) = constant_fold([expected])
    except NotConstantValueError:
        expected_n = None
    msg = "length of weights does not match the constrained dimension"
    if static_n is not None and expected_n is not None:
        if static_n != int(expected_n):
            raise ValueError(
                f"{msg}: got {static_n} weights for a length of {int(expected_n)}"
            )
        return weights
    return Assert(msg=msg)(weights, pt.eq(weights.shape[0], expected))


# ---------------------------------------------------------------------------
# Tensor API
# ---------------------------------------------------------------------------


class WeightedZeroSumTransform(Transform):
    """Map the hyperplane ``sum(weights * value) = 0`` to ``n - 1`` free coordinates.

    The map is the restriction of the Householder reflection sending
    ``u = weights / |weights|`` to ``-e_n``. It is an isometry, so the log
    Jacobian determinant is zero, and with equal weights it coincides with
    :class:`pymc.distributions.transforms.ZeroSumTransform` on one axis.

    As the default transform of :class:`WeightedZeroSumNormal` it reads the
    weights from the random variable's inputs, so they may be data or another
    random variable and the transform follows the graph through model cloning.
    Pass ``weights`` only to use the transform on its own.

    Parameters
    ----------
    weights : tensor_like, optional
        1-d vector of strictly positive weights, for standalone use.
    """

    name = "weighted_zerosum"
    ndim_supp = 1

    def __init__(self, weights: TensorLike | None = None) -> None:
        self.weights = None if weights is None else _validate_weights(weights)

    def _direction(self, rv_inputs: Sequence[Variable]) -> TensorVariable:
        if rv_inputs:
            # WeightedZeroSumNormalRV inputs are (rng, size, sigma, weights)
            return _unit_direction(rv_inputs[-1])
        if self.weights is not None:
            return _unit_direction(self.weights)
        raise ValueError(
            "WeightedZeroSumTransform needs weights: pass them at construction "
            "or use it as the default transform of WeightedZeroSumNormal"
        )

    def forward(self, value: TensorVariable, *rv_inputs: Variable) -> TensorVariable:
        """Map a value on the constraint hyperplane to unconstrained coordinates."""
        u = self._direction(rv_inputs)
        # explicit length-1 axes keep this broadcastable when the value's
        # static shape is unknown (symbolic weights)
        coef = pt.expand_dims(value[..., -1], -1) / (1 + pt.expand_dims(u[..., -1], -1))
        return value[..., :-1] - coef * u[..., :-1]

    def backward(self, value: TensorVariable, *rv_inputs: Variable) -> TensorVariable:
        """Map unconstrained coordinates onto the constraint hyperplane."""
        u = self._direction(rv_inputs)
        u_head, u_last = u[..., :-1], pt.expand_dims(u[..., -1], -1)
        proj = pt.sum(value * u_head, axis=-1, keepdims=True)
        return pt.concatenate([value - proj / (1 + u_last) * u_head, -proj], axis=-1)

    def log_jac_det(
        self, value: TensorVariable, *rv_inputs: Variable
    ) -> TensorVariable:
        """Zero: the map is an isometry."""
        return pt.zeros(value.shape[:-1], dtype=value.dtype)


class WeightedZeroSumNormalRV(SymbolicRandomVariable):
    """WeightedZeroSumNormal random variable."""

    name = "WeightedZeroSumNormal"
    extended_signature = "[rng],[size],(),(n)->[rng],(n)"
    _print_name = ("WeightedZeroSumNormal", "\\operatorname{WeightedZeroSumNormal}")

    @classmethod
    def rv_op(cls, sigma, weights, *, size=None, rng=None):
        """Draw an isotropic normal and project it onto the constraint hyperplane."""
        sigma = pt.as_tensor(sigma)
        weights = pt.as_tensor(weights)
        rng = normalize_rng_param(rng)
        size = normalize_size_param(size)
        if rv_size_is_none(size):
            size = sigma.shape  # sigma carries the batch shape

        shape = (*tuple(size), weights.shape[-1])
        next_rng, normal = pm.Normal.dist(
            sigma=pt.shape_padright(sigma), shape=shape, rng=rng, return_next_rng=True
        )
        u = _unit_direction(weights)
        draw = normal - pt.sum(normal * u, axis=-1, keepdims=True) * u
        return cls(inputs=[rng, size, sigma, weights], outputs=[next_rng, draw])(
            rng, size, sigma, weights
        )


class WeightedZeroSumNormal(Distribution):
    r"""Normal distribution whose last axis satisfies ``sum(weights * x) = 0``.

    Generalises :class:`pymc.ZeroSumNormal` to unequal weights. Writing
    :math:`u = w / \|w\|`,

    .. math::

        \text{WZSN}(\sigma, w) = N\Big(0, \sigma^2 (I_n - u u^T)\Big),

    the orthogonal projection of an isotropic normal onto the constraint
    hyperplane. The distribution is isotropic within the hyperplane and its
    density depends on the weights only through the constraint. It is not the
    same as subtracting the weighted mean, an oblique projection with a
    different, non-isotropic covariance. The marginal variances are
    :math:`\sigma^2 (1 - u_i^2)`: the component with the dominant weight has
    the smallest variance (with ``weights = [0.9, 0.05, 0.03, 0.02]`` the first
    component has variance :math:`\approx 0.004\,\sigma^2`).

    With equal weights this is exactly ``ZeroSumNormal`` with one zero-sum
    axis. Only a single constrained axis, the last, is supported.

    Parameters
    ----------
    sigma : tensor_like of float
        Scale of the underlying isotropic normal, ``sigma > 0``. Its shape is
        the batch shape of the variable; it cannot vary along the constrained
        axis. Defaults to 1.
    weights : tensor_like
        1-d vector of strictly positive weights. Its length is the length of
        the constrained axis. May be symbolic, e.g. :func:`pymc.Data` or
        another random variable.
    support_shape : int, optional
        Length of the constrained axis, only to check it against ``weights``;
        otherwise inferred from ``shape``, ``dims`` or ``observed``.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import pymc as pm
        from pymc_marketing.mmm import WeightedZeroSumNormal

        share = np.array([0.7, 0.2, 0.1])
        with pm.Model(coords={"channel": ["tv", "radio", "web"]}):
            x = WeightedZeroSumNormal("x", weights=share, dims="channel")
        assert np.allclose(pm.draw(x, draws=100) @ share, 0.0)
    """

    rv_type = WeightedZeroSumNormalRV
    rv_op = WeightedZeroSumNormalRV.rv_op

    def __new__(
        cls, *args, support_shape: int | None = None, dims: Dims | None = None, **kwargs
    ):
        """Create the named variable; ``dims`` or ``observed`` fix the constrained length."""
        if dims is not None or kwargs.get("observed") is not None:
            support_shape = get_support_shape_1d(
                support_shape=support_shape,
                shape=None,  # checked in `dist`
                dims=dims,
                observed=kwargs.get("observed", None),
            )
        return super().__new__(
            cls, *args, support_shape=support_shape, dims=dims, **kwargs
        )

    @classmethod
    def dist(
        cls,
        sigma: TensorLike = 1.0,
        *,
        weights: TensorLike,
        support_shape: int | TensorVariable | None = None,
        **kwargs,
    ) -> TensorVariable:
        """Create an unnamed variable, checking ``weights`` against any declared length."""
        weights = _validate_weights(weights)
        shape = kwargs.get("shape")
        for expected in (support_shape, None if shape is None else to_tuple(shape)[-1]):
            if expected is not None:
                weights = _check_length(weights, expected)
        return super().dist([pt.as_tensor(sigma), weights], **kwargs)


@_support_point.register(WeightedZeroSumNormalRV)
def _support_point_(op, rv, *rv_inputs):
    return pt.zeros_like(rv)


@_default_transform.register(WeightedZeroSumNormalRV)
def _default_transform_(op, rv):
    return WeightedZeroSumTransform()


@_logprob.register(WeightedZeroSumNormalRV)
def _logp(op, values, rng, size, sigma, weights, **kwargs):
    (value,) = values
    n = value.shape[-1].astype("floatX")
    sigma = pt.shape_padright(sigma)  # batch-only: align against the batch axes

    # the constraint check is relative to the value's scale, so it holds for
    # large sigma and in float32
    rtol = 1e-9 if value.dtype == "float64" else 1e-6
    tol = rtol * (1 + pt.sqrt(pt.sum(value**2, axis=-1)))
    on_hyperplane = pt.abs(pt.sum(value * _unit_direction(weights), axis=-1)) <= tol

    logp = pt.sum(
        -0.5 * pt.pow(value / sigma, 2)
        - (pt.log(pt.sqrt(2.0 * np.pi)) + pt.log(sigma)) * (n - 1) / n,
        axis=-1,
    )
    return check_parameters(
        logp,
        pt.all(on_hyperplane),
        pt.all(weights > 0),
        pt.all(sigma > 0),
        msg="sum(weights * value, axis=-1) = 0, weights > 0, sigma > 0",
    )


# ---------------------------------------------------------------------------
# pymc.dims API
# ---------------------------------------------------------------------------


def _constant_weights(
    weights: TensorLike | XTensorVariable, dim: str
) -> XTensorConstant:
    """Coerce weights to a constant xtensor over ``dim``.

    Like :class:`pymc.dims.distributions.transforms.IntervalTransform`, the
    dims transform keeps its parameters as constants rather than reading them
    from the random variable's inputs. Use the tensor API for symbolic
    weights.
    """
    not_constant = NotImplementedError(
        "DimWeightedZeroSumNormal needs constant weights; use "
        "WeightedZeroSumNormal for symbolic weights"
    )
    if isinstance(weights, XTensorVariable) and weights.type.dims != (dim,):
        raise ValueError(f"weights must have dims ({dim!r},), got {weights.type.dims}")
    if isinstance(weights, Constant):
        values = weights.data  # a constant of either API
    elif isinstance(weights, Variable):
        raise not_constant
    else:
        values = weights
    values = np.asarray(values, dtype=pytensor.config.floatX)
    if values.ndim != 1:
        raise ValueError("weights must be a 1-d vector")
    if np.any(values <= 0):
        raise ValueError("weights must be strictly positive")
    return as_xtensor(values, dims=(dim,))


class DimWeightedZeroSumTransform(DimTransform):
    """Map the hyperplane ``(weights * value).sum(dim) = 0`` to ``n - 1`` free coordinates.

    The same Householder map as :class:`WeightedZeroSumTransform`, applied
    along ``dim``. The weights are constants fixed at construction, as for
    :class:`pymc.dims.distributions.transforms.IntervalTransform`.

    Parameters
    ----------
    dim : str
        The constrained dimension.
    weights : array_like or constant xtensor
        1-d vector of strictly positive weights over ``dim``.
    """

    name = "weighted_zerosum"

    def __init__(self, dim: str, weights: TensorLike | XTensorVariable) -> None:
        self.dim = dim
        self.weights = _constant_weights(weights, dim)
        u = as_xtensor(_unit_direction(self.weights.values), dims=(dim,))
        self.u_head = u.isel({dim: slice(None, -1)})
        self.u_last = u.isel({dim: -1})

    def forward(self, value: XTensorVariable, *rv_inputs: Variable) -> XTensorVariable:
        """Map a value on the constraint hyperplane to unconstrained coordinates."""
        coef = value.isel({self.dim: -1}) / (1 + self.u_last)
        return value.isel({self.dim: slice(None, -1)}) - coef * self.u_head

    def backward(self, value: XTensorVariable, *rv_inputs: Variable) -> XTensorVariable:
        """Map unconstrained coordinates onto the constraint hyperplane."""
        proj = (value * self.u_head).sum(self.dim)
        head = value - proj / (1 + self.u_last) * self.u_head
        return ptx.concat([head, -proj], dim=self.dim)

    def log_jac_det(
        self, value: XTensorVariable, *rv_inputs: Variable
    ) -> XTensorVariable:
        """Zero: the map is an isometry."""
        return as_xtensor(0.0).broadcast_like(value, exclude=(self.dim,))


class DimWeightedZeroSumNormal(VectorDimDistribution):
    """The :mod:`pymc.dims` flavour of :class:`WeightedZeroSumNormal`.

    Draws satisfy ``(weights * value).sum(core_dim) = 0``. Weights must be
    constants: pymc's dims transforms keep their parameters fixed at
    construction, so symbolic weights are only supported by the tensor API.

    Parameters
    ----------
    sigma : xtensor_like, optional
        Scale of the underlying isotropic normal; cannot have the core
        dimension. Defaults to 1.
    weights : array_like or constant xtensor
        Strictly positive weights along the single core dimension.
    core_dims : str or sequence of one str
        The constrained dimension.
    **kwargs
        Forwarded to :class:`pymc.dims.distributions.core.DimDistribution`.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import pymc as pm
        from pymc_marketing.mmm import DimWeightedZeroSumNormal

        share = np.array([0.7, 0.2, 0.1])
        with pm.Model(coords={"channel": ["tv", "radio", "web"]}):
            x = DimWeightedZeroSumNormal("x", weights=share, core_dims="channel")
        assert np.allclose(pm.draw(x, draws=100) @ share, 0.0)
    """

    @classmethod
    def __new__(
        cls,
        *args,
        core_dims=None,
        dims=None,
        default_transform=UNSET,
        observed=None,
        **kwargs,
    ):
        """Create the named variable with the weighted zero-sum transform on the core dim."""
        if core_dims is not None:
            if isinstance(core_dims, str):
                core_dims = (core_dims,)
            if observed is None and default_transform is UNSET:
                default_transform = DimWeightedZeroSumTransform(
                    core_dims[-1], kwargs["weights"]
                )
        if dims is None and core_dims is not None:
            dims = (..., *core_dims)  # forwarded to `dist` through `dim_lengths`
        return super().__new__(
            *args,
            core_dims=core_dims,
            dims=dims,
            default_transform=default_transform,
            observed=observed,
            **kwargs,
        )

    @classmethod
    def dist(
        cls,
        sigma: TensorLike | XTensorVariable = 1.0,
        *,
        weights: TensorLike | XTensorVariable | None = None,
        core_dims: str | Sequence[str] | None = None,
        dim_lengths: dict[str, Variable],
        **kwargs,
    ) -> XTensorVariable:
        """Create an unnamed variable, checking ``weights`` against the core dim's length."""
        if isinstance(core_dims, str):
            core_dims = (core_dims,)
        if core_dims is None or len(core_dims) != 1:
            raise ValueError("DimWeightedZeroSumNormal requires exactly one core_dims")
        if weights is None:
            raise ValueError("DimWeightedZeroSumNormal requires weights")
        (core_dim,) = core_dims
        weights = _constant_weights(weights, core_dim)
        if core_dim in dim_lengths:
            checked = _check_length(weights.values, dim_lengths[core_dim])
            weights = as_xtensor(checked, dims=(core_dim,))
        sigma = cls._as_xtensor(sigma)
        return super().dist(
            [sigma, weights], core_dims=core_dims, dim_lengths=dim_lengths, **kwargs
        )

    @classmethod
    def xrv_op(cls, sigma, weights, core_dims, extra_dims=None, rng=None, **kwargs):
        """Vectorise the tensor random variable over the batch dims."""
        sigma = cls._as_xtensor(sigma)
        weights = as_xtensor(weights)
        # a throwaway core op with scalar sigma: `as_xrv` aligns batch and core
        # dims by name when the xtensors are passed below
        core_op = WeightedZeroSumNormalRV.rv_op(
            sigma=pt.scalar(dtype=sigma.type.dtype), weights=weights.values
        ).owner.op
        xop = pxr.as_xrv(core_op, core_inps_dims_map=[(), (0,)], core_out_dims_map=(0,))
        return xop(
            sigma,
            weights,
            core_dims=core_dims,
            extra_dims=extra_dims,
            rng=rng,
            **kwargs,
        )
