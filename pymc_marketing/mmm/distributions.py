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

:class:`WeightedZeroSumNormal` is :class:`pymc.ZeroSumNormal` in reflected
coordinates. A weighted zero-sum hyperplane ``sum(weights * x) = 0`` is the
plain zero-sum hyperplane reflected by the Householder map that sends the
unit weight vector onto the ones direction, so the distribution draws from
``ZeroSumNormal`` and reflects, and its transform composes that reflection
with pymc's ``ZeroSumTransform``. It is provided for the tensor API
(:class:`WeightedZeroSumNormal`) and the :mod:`pymc.dims` API
(:class:`DimWeightedZeroSumNormal`), following the same split pymc uses for
its own distributions.

The distribution is incubating here before a proposed move to pymc-extras or
pymc; its import path may change.
"""

from collections.abc import Sequence

import numpy as np
import pytensor
import pytensor.tensor as pt
from numpy.typing import (
    ArrayLike,  # noqa: F401  # resolves pt.TensorLike's ForwardRef('ArrayLike') for sphinx_autodoc_typehints (#1197)
)
from pymc.dims.distributions.core import VectorDimDistribution
from pymc.dims.distributions.transforms import DimTransform
from pymc.dims.distributions.transforms import ZeroSumTransform as DimZeroSumTransform
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import Distribution
from pymc.distributions.multivariate import ZeroSumNormal, ZeroSumNormalRV
from pymc.distributions.shape_utils import (
    Dims,
    get_support_shape_1d,
    to_tuple,
)
from pymc.distributions.transforms import ZeroSumTransform, _default_transform
from pymc.exceptions import NotConstantValueError
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import logp
from pymc.logprob.transforms import ChainedTransform, Transform
from pymc.pytensorf import constant_fold
from pymc.util import UNSET
from pytensor.graph.basic import Constant, Variable
from pytensor.raise_op import Assert
from pytensor.tensor import TensorConstant, TensorLike, TensorVariable
from pytensor.xtensor import as_xtensor
from pytensor.xtensor import random as pxr
from pytensor.xtensor.type import XTensorConstant, XTensorVariable

__all__ = [
    "DimWeightedZeroSumNormal",
    "DimWeightedZeroSumTransform",
    "WeightedZeroSumNormal",
    "WeightedZeroSumTransform",
]


def _reflect(value: TensorLike, weights: TensorLike) -> TensorVariable:
    """Householder reflection swapping the weighted and the plain zero-sum planes.

    With ``u = weights / |weights|`` and ``v = u + 1/sqrt(n)`` the map
    ``x - 2 v (v . x) / (v . v)`` sends ``u`` to ``-1/sqrt(n)``, so it carries
    the plane ``u . x = 0`` onto the plane ``sum(x) = 0`` and back: it is its
    own inverse and an isometry. For strictly positive weights ``|v|^2 >= 2``,
    so the map is never degenerate; with equal weights it is the identity on
    the plane.
    """
    value = pt.as_tensor(value)
    weights = pt.as_tensor(weights)
    u = weights / pt.sqrt(pt.sum(weights**2, axis=-1, keepdims=True))
    v = u + 1 / pt.sqrt(pt.cast(weights.shape[-1], weights.dtype))
    proj = pt.sum(value * v, axis=-1, keepdims=True) / pt.sum(
        v * v, axis=-1, keepdims=True
    )
    return value - 2 * v * proj


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


class _Reflect(Transform):
    """The reflection of :func:`_reflect` as a transform: self-inverse, zero Jacobian."""

    name = "reflect"
    ndim_supp = 1

    def __init__(self, weights: TensorVariable | None) -> None:
        self.weights = weights

    def _weights(self, rv_inputs: Sequence[Variable]) -> Variable:
        if rv_inputs:
            # WeightedZeroSumNormalRV inputs are (rng, size, sigma, weights)
            return rv_inputs[-1]
        if self.weights is not None:
            return self.weights
        raise ValueError(
            "WeightedZeroSumTransform needs weights: pass them at construction "
            "or use it as the default transform of WeightedZeroSumNormal"
        )

    def forward(self, value: TensorVariable, *rv_inputs: Variable) -> TensorVariable:
        return _reflect(value, self._weights(rv_inputs))

    def backward(self, value: TensorVariable, *rv_inputs: Variable) -> TensorVariable:
        return _reflect(value, self._weights(rv_inputs))

    def log_jac_det(
        self, value: TensorVariable, *rv_inputs: Variable
    ) -> TensorVariable:
        return pt.zeros(value.shape[:-1], dtype=value.dtype)


class WeightedZeroSumTransform(ChainedTransform):
    """Map the hyperplane ``sum(weights * value) = 0`` to ``n - 1`` free coordinates.

    Reflects the weighted zero-sum plane onto the plain one and then applies
    :class:`pymc.distributions.transforms.ZeroSumTransform`. Both steps are
    isometries, so the log Jacobian determinant is zero; with equal weights
    the reflection is the identity on the plane and the transform *is*
    ``ZeroSumTransform``.

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

    def __init__(self, weights: TensorLike | None = None) -> None:
        self.weights = None if weights is None else _validate_weights(weights)
        super().__init__([_Reflect(self.weights), ZeroSumTransform(zerosum_axes=(-1,))])

    def log_jac_det(
        self, value: TensorVariable, *rv_inputs: Variable
    ) -> TensorVariable:
        """Zero: both steps are isometries."""
        return pt.zeros(value.shape[:-1], dtype=value.dtype)


class WeightedZeroSumNormalRV(ZeroSumNormalRV):
    """WeightedZeroSumNormal random variable: a reflected ZeroSumNormal draw."""

    name = "WeightedZeroSumNormal"
    _print_name = ("WeightedZeroSumNormal", "\\operatorname{WeightedZeroSumNormal}")

    @classmethod
    def rv_op(cls, sigma, weights, *, size=None, rng=None):
        """Draw a ZeroSumNormal on the plain plane and reflect it onto the weighted one."""
        sigma = pt.as_tensor(sigma)
        weights = pt.as_tensor(weights)
        zerosum = ZeroSumNormalRV.rv_op(
            sigma=pt.shape_padright(sigma),
            support_shape=weights.shape[-1:],
            size=size,
            rng=rng,
        )
        next_rng = zerosum.owner.outputs[0]
        rng, size = zerosum.owner.inputs[:2]
        return cls(
            inputs=[rng, size, sigma, weights],
            outputs=[next_rng, _reflect(zerosum, weights)],
            extended_signature="[rng],[size],(),(n)->[rng],(n)",
        )(rng, size, sigma, weights)


class WeightedZeroSumNormal(Distribution):
    r"""Normal distribution whose last axis satisfies ``sum(weights * x) = 0``.

    :class:`pymc.ZeroSumNormal` in reflected coordinates. Writing
    :math:`u = w / \|w\|`,

    .. math::

        \text{WZSN}(\sigma, w) = N\Big(0, \sigma^2 (I_n - u u^T)\Big),

    the orthogonal projection of an isotropic normal onto the constraint
    hyperplane, which is what a ``ZeroSumNormal`` is for ``u = 1/\sqrt{n}``.
    The distribution is isotropic within the hyperplane and its density
    depends on the weights only through the constraint. It is not the same as
    subtracting the weighted mean, an oblique projection with a different,
    non-isotropic covariance. The marginal variances are
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


# The support point (zeros) is inherited from ZeroSumNormalRV's registration.


@_default_transform.register(WeightedZeroSumNormalRV)
def _default_transform_(op, rv):
    return WeightedZeroSumTransform()


@_logprob.register(WeightedZeroSumNormalRV)
def _logp(op, values, rng, size, sigma, weights, **kwargs):
    """ZeroSumNormal's density, evaluated in the reflected coordinates."""
    (value,) = values
    zerosum = ZeroSumNormal.dist(
        sigma=pt.shape_padright(sigma),
        n_zerosum_axes=1,
        support_shape=(value.shape[-1],),
    )
    return check_parameters(
        logp(zerosum, _reflect(value, weights)), pt.all(weights > 0), msg="weights > 0"
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

    The dims counterpart of :class:`WeightedZeroSumTransform`: reflect the
    weighted plane onto the plain one along ``dim``, then apply pymc's dims
    :class:`~pymc.dims.distributions.transforms.ZeroSumTransform`. The weights
    are constants fixed at construction, as for
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
        w = self.weights.data
        self.v = as_xtensor(w / np.linalg.norm(w) + 1 / np.sqrt(len(w)), dims=(dim,))
        self.zerosum = DimZeroSumTransform(dims=(dim,))

    def _reflect(self, value: XTensorVariable) -> XTensorVariable:
        return value - 2 * self.v * (value * self.v).sum(self.dim) / (
            self.v * self.v
        ).sum(self.dim)

    def forward(self, value: XTensorVariable, *rv_inputs: Variable) -> XTensorVariable:
        """Map a value on the constraint hyperplane to unconstrained coordinates."""
        return self.zerosum.forward(self._reflect(value))

    def backward(self, value: XTensorVariable, *rv_inputs: Variable) -> XTensorVariable:
        """Map unconstrained coordinates onto the constraint hyperplane."""
        return self._reflect(self.zerosum.backward(value))

    def log_jac_det(
        self, value: XTensorVariable, *rv_inputs: Variable
    ) -> XTensorVariable:
        """Zero: both steps are isometries."""
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
