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

Currently provides :class:`WeightedZeroSumNormal`, a normal distribution
constrained to a spend-share-weighted zero sum, in both the tensor API
(:class:`WeightedZeroSumNormal`) and the :mod:`pymc.dims` API
(:class:`DimWeightedZeroSumNormal`).

The constrained multipliers of :class:`~pymc_marketing.mmm.campaign_media.NestedCampaignMedia`
use it to preserve the spend-weighted channel mean exactly.
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
from pytensor.xtensor.type import XTensorVariable

__all__ = [
    "DimWeightedZeroSumNormal",
    "DimWeightedZeroSumTransform",
    "WeightedZeroSumNormal",
    "WeightedZeroSumTransform",
]


def weighted_zerosum_direction(weights: TensorLike) -> TensorVariable:
    """Return the unit vector ``u = weights / |weights|`` along the last axis.

    Shared by :class:`WeightedZeroSumNormal`'s random graph, its logp
    and :class:`WeightedZeroSumTransform`, so that all three agree on ``u``
    to machine precision. The logp checks that values lie on the constraint hyperplane
    up to a tight tolerance, which relies on this agreement.
    """
    weights = pt.as_tensor(weights)
    return weights / pt.sqrt(pt.sum(weights**2, axis=-1, keepdims=True))


class WeightedZeroSumTransform(Transform):
    """
    Constrains random samples to satisfy ``sum(weights * value) = 0`` along the last axis.

    The map is the restriction of the Householder reflection sending the
    normalized weight vector ``u = weights / |weights|`` to ``-e_n``. It is an
    isometry between the unconstrained space and the constraint hyperplane, so
    the log Jacobian determinant is zero. With equal weights it reproduces
    ``ZeroSumTransform(zerosum_axes=(-1,))`` exactly.

    When used as the default transform of :class:`WeightedZeroSumNormal`
    the weights are read from the random variable's inputs, so they may be
    data or other random variables and the transform follows the graph through
    model cloning. The ``weights`` argument is only needed when the transform
    is used standalone.

    Parameters
    ----------
    weights : tensor_like, optional
        1-d vector of strictly positive weights defining the constraint
        ``sum(weights * value) = 0``. Only a single constrained axis (the
        last) is supported.
    """

    name = "weighted_zerosum"
    ndim_supp = 1

    def __init__(self, weights: TensorLike | None = None) -> None:
        if weights is not None:
            weights = pt.as_tensor(weights)
            if weights.type.ndim != 1:
                raise ValueError("weights must be a 1-d vector")
        self.weights = weights

    def _get_weights(self, rv_inputs: Sequence[Variable]) -> TensorVariable:
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
        """Map a value on the constraint hyperplane to unconstrained coordinates."""
        u = weighted_zerosum_direction(self._get_weights(rv_inputs))
        # Keep static length-1 axes so this broadcasts even when the value
        # has unknown static shape (symbolic weights)
        coef = pt.expand_dims(value[..., -1], -1) / (1 + pt.expand_dims(u[..., -1], -1))
        return value[..., :-1] - coef * u[..., :-1]

    def backward(self, value: TensorVariable, *rv_inputs: Variable) -> TensorVariable:
        """Map unconstrained coordinates onto the constraint hyperplane."""
        u = weighted_zerosum_direction(self._get_weights(rv_inputs))
        u_head, u_last = u[..., :-1], pt.expand_dims(u[..., -1], -1)
        sum_vals = pt.sum(value * u_head, axis=-1, keepdims=True)
        head = value - sum_vals / (1 + u_last) * u_head
        return pt.concatenate([head, -sum_vals], axis=-1)

    def log_jac_det(
        self, value: TensorVariable, *rv_inputs: Variable
    ) -> TensorVariable:
        """Log determinant of the Jacobian of ``backward``; zero, the map is an isometry."""
        return pt.zeros(value.shape[:-1], dtype=value.dtype)


def _static_length(length: Variable) -> int | None:
    """Value of a dimension length if it can be constant folded, else None."""
    try:
        (folded,) = constant_fold([length])
    except NotConstantValueError:
        return None
    return int(folded)


class WeightedZeroSumNormalRV(SymbolicRandomVariable):
    """WeightedZeroSumNormal random variable."""

    name = "WeightedZeroSumNormal"
    extended_signature = "[rng],[size],(),(n)->[rng],(n)"
    _print_name = ("WeightedZeroSumNormal", "\\operatorname{WeightedZeroSumNormal}")

    @classmethod
    def rv_op(cls, sigma, weights, *, size=None, rng=None):
        """Build the symbolic random variable graph."""
        sigma = pt.as_tensor(sigma)
        weights = pt.as_tensor(weights)
        rng = normalize_rng_param(rng)
        size = normalize_size_param(size)

        if rv_size_is_none(size):
            # Size is implied by the batch shape of sigma
            size = sigma.shape

        shape = (*tuple(size), weights.shape[-1])
        next_rng, normal_dist = pm.Normal.dist(
            sigma=pt.shape_padright(sigma), shape=shape, rng=rng, return_next_rng=True
        )

        # Project onto the hyperplane orthogonal to u = weights / |weights|
        u = weighted_zerosum_direction(weights)
        weighted_zerosum_rv = (
            normal_dist - pt.sum(normal_dist * u, axis=-1, keepdims=True) * u
        )

        return cls(
            inputs=[rng, size, sigma, weights],
            outputs=[next_rng, weighted_zerosum_rv],
        )(rng, size, sigma, weights)


class WeightedZeroSumNormal(Distribution):
    r"""
    Normal distribution where the last axis is constrained to sum to zero under weights.

    Generalizes :class:`~pymc.ZeroSumNormal`: draws satisfy
    ``sum(weights * value) = 0`` along the last axis instead of
    ``sum(value) = 0``. Writing :math:`u = w / \|w\|`,

    .. math::

        WZSN(\sigma, w) = N\Big(0, \sigma^2 (I_n - u u^T)\Big)

    This is the *orthogonal projection* of an isotropic normal onto the
    constraint hyperplane: the distribution is isotropic within the hyperplane
    and its density does not depend on the weights except through the
    constraint itself. It is not the same as subtracting the weighted mean,
    :math:`x = z - (w^T z / \sum w) \mathbf{1}`, which is an oblique projection
    with a different, non-isotropic covariance.

    A consequence is that the marginal variances are
    :math:`\operatorname{Var}(x_i) = \sigma^2 (1 - u_i^2)`, so the component
    with the dominant weight has the smallest variance. For example with
    ``weights = [0.9, 0.05, 0.03, 0.02]`` the first component has variance
    :math:`\approx 0.004\,\sigma^2`.

    With equal weights this is exactly ``ZeroSumNormal`` with one zero-sum
    axis. Only a single constrained axis (the last) is supported.

    Parameters
    ----------
    sigma : tensor_like of float
        Scale parameter (sigma > 0), the standard deviation of the underlying
        unconstrained Normal distribution. Defaults to 1. It cannot vary along
        the constrained axis: its shape is the batch shape of the variable.
    weights : tensor_like
        1-d vector of strictly positive weights defining the constraint.
        Its length defines the support shape. May be symbolic, e.g. a
        :func:`~pymc.Data` variable or another random variable.
    support_shape : int, optional
        Length of the constrained axis. Only needed to check consistency with
        ``weights``; it is otherwise inferred from ``shape``, ``dims`` or
        ``observed``.

    Examples
    --------
    .. code-block:: python

        with pm.Model(coords={"channel": ["tv", "radio", "web"]}) as model:
            spend_share = pm.Data("spend_share", [0.7, 0.2, 0.1], dims="channel")
            x = WeightedZeroSumNormal("x", weights=spend_share, dims="channel")
    """

    rv_type = WeightedZeroSumNormalRV
    rv_op = WeightedZeroSumNormalRV.rv_op

    def __new__(
        cls, *args, support_shape: int | None = None, dims: Dims | None = None, **kwargs
    ):
        """Create the variable in the current model and set up its default transform."""
        if dims is not None or kwargs.get("observed") is not None:
            support_shape = get_support_shape_1d(
                support_shape=support_shape,
                shape=None,  # Shape will be checked in `cls.dist`
                dims=dims,
                observed=kwargs.get("observed", None),
            )

        return super().__new__(
            cls,
            *args,
            support_shape=support_shape,
            dims=dims,
            **kwargs,
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
        """Create an unnamed random variable with validated weights."""
        weights = pt.as_tensor(weights).astype("floatX")
        if weights.type.ndim != 1:
            raise ValueError("weights must be a 1-d vector")
        if isinstance(weights, TensorConstant) and np.any(weights.data <= 0):
            raise ValueError("weights must be strictly positive")

        # The weights define the length of the constrained axis. Check it agrees
        # with whatever else the user told us: eagerly when the length is known
        # now, and at runtime otherwise.
        shape = kwargs.get("shape")
        expected_lengths = [
            support_shape,
            None if shape is None else to_tuple(shape)[-1],
        ]
        for expected in expected_lengths:
            if expected is not None:
                cls._check_weights_length_static(weights, pt.as_tensor(expected))
        support_shape = get_support_shape_1d(support_shape=support_shape, shape=shape)
        if support_shape is not None and not isinstance(support_shape, Constant):
            weights = Assert(msg=cls._length_mismatch_msg)(
                weights, pt.eq(weights.shape[0], support_shape)
            )

        sigma = pt.as_tensor(sigma)

        return super().dist([sigma, weights], **kwargs)

    _length_mismatch_msg = "length of weights does not match the constrained dimension"

    @classmethod
    def _check_weights_length_static(
        cls, weights: TensorVariable, expected: TensorVariable
    ) -> None:
        static_n = weights.type.shape[0]
        expected_n = _static_length(expected)
        if static_n is not None and expected_n is not None and static_n != expected_n:
            raise ValueError(
                f"{cls._length_mismatch_msg}: got {static_n} weights "
                f"for a dimension of length {expected_n}"
            )


@_support_point.register(WeightedZeroSumNormalRV)
def weighted_zerosumnormal_support_point(
    op: WeightedZeroSumNormalRV, rv: TensorVariable, *rv_inputs: Variable
) -> TensorVariable:
    return pt.zeros_like(rv)


@_default_transform.register(WeightedZeroSumNormalRV)
def weighted_zerosum_default_transform(
    op: WeightedZeroSumNormalRV, rv: TensorVariable
) -> WeightedZeroSumTransform:
    # The transform reads the weights from the RV inputs it is handed at
    # logp / initial point time, so it is not bound to this graph
    return WeightedZeroSumTransform()


@_logprob.register(WeightedZeroSumNormalRV)
def weighted_zerosumnormal_logp(
    op: WeightedZeroSumNormalRV,
    values: Sequence[TensorVariable],
    rng: Variable,
    size: TensorVariable,
    sigma: TensorVariable,
    weights: TensorVariable,
    **kwargs,
) -> TensorVariable:
    (value,) = values
    n = value.shape[-1].astype("floatX")

    # sigma is batch-only; align it against the batch axes, not the constrained one
    sigma = pt.shape_padright(sigma)

    u = weighted_zerosum_direction(weights)
    # The constraint is checked relative to the scale of the value, so it
    # holds for large sigma and in float32
    rtol = 1e-9 if value.dtype == "float64" else 1e-6
    tol = rtol * (1 + pt.sqrt(pt.sum(value**2, axis=-1)))
    weighted_zerosum = pt.all(pt.abs(pt.sum(value * u, axis=-1)) <= tol)

    out = pt.sum(
        -0.5 * pt.pow(value / sigma, 2)
        - (pt.log(pt.sqrt(2.0 * np.pi)) + pt.log(sigma)) * (n - 1) / n,
        axis=-1,
    )

    return check_parameters(
        out,
        weighted_zerosum,
        pt.all(weights > 0),
        pt.all(sigma > 0),
        msg="sum(weights * value, axis=-1) = 0, weights > 0, sigma > 0",
    )


def _weights_as_xtensor(
    weights: TensorLike | XTensorVariable, dim: str
) -> XTensorVariable:
    """Coerce constant, tensor or xtensor weights to a 1-d xtensor over ``dim``."""
    if isinstance(weights, XTensorVariable):
        if weights.type.dims != (dim,):
            raise ValueError(
                f"weights must have dims ({dim!r},), got {weights.type.dims}"
            )
        return weights
    weights = pt.as_tensor(weights)
    if weights.type.ndim != 1:
        raise ValueError("weights must be a 1-d vector")
    if isinstance(weights, pt.TensorConstant):
        if np.any(weights.data <= 0):
            raise ValueError("weights must be strictly positive")
        weights = weights.astype(pytensor.config.floatX)
    return as_xtensor(weights, dims=(dim,))


def _weights_from_rv_inputs(rv_inputs: Sequence[Variable]) -> Variable:
    """Find the weights among the inputs of a WeightedZeroSumNormal node.

    Before lowering (e.g. at initial point time), the node is the XRV with
    inputs ``(rng, *extra_dim_lengths, sigma, weights)``. In the logp graph
    the XRV has already been lowered to
    ``XTensorFromTensor(WeightedZeroSumNormalRV(...))``, so the single input
    is the tensor RV, and its op knows from its signature which inputs are
    the distribution parameters.
    """
    if len(rv_inputs) == 1:
        (tensor_rv,) = rv_inputs
        node = tensor_rv.owner
        if node is None or not isinstance(node.op, WeightedZeroSumNormalRV):
            raise ValueError("Could not find weights among the random variable inputs")
        _sigma, weights = node.op.dist_params(node)
        # Drop the broadcastable batch axes the vectorized RV prepends
        return weights[(0,) * (weights.type.ndim - 1)]
    return rv_inputs[-1]


class DimWeightedZeroSumTransform(DimTransform):
    """Constrains samples to satisfy ``(weights * value).sum(dim) = 0``.

    Restriction of the Householder reflection sending ``u = w / |w|`` to
    ``-e_n`` along ``dim``; an isometry, so the log Jacobian determinant is
    zero. With equal weights it reproduces ``pymc.dims.distributions.transforms.ZeroSumTransform`` on a single
    dim exactly.

    When used as the default transform of :class:`DimWeightedZeroSumNormal`
    the weights are read from the random variable's inputs. The ``weights``
    argument is only needed when the transform is used standalone.
    """

    name = "weighted_zerosum"

    def __init__(
        self, dim: str, weights: TensorLike | XTensorVariable | None = None
    ) -> None:
        self.dim = dim
        self.weights = None if weights is None else _weights_as_xtensor(weights, dim)

    def _get_weights(self, rv_inputs: Sequence[Variable]) -> XTensorVariable:
        if rv_inputs:
            return _weights_as_xtensor(_weights_from_rv_inputs(rv_inputs), self.dim)
        if self.weights is not None:
            return self.weights
        raise ValueError(
            "DimWeightedZeroSumTransform needs weights: pass them at construction "
            "or use it as the default transform of DimWeightedZeroSumNormal"
        )

    def _weight_direction(
        self, rv_inputs: Sequence[Variable]
    ) -> tuple[XTensorVariable, XTensorVariable]:
        weights = self._get_weights(rv_inputs)
        # Same computation as the tensor API, so both agree on u to machine precision
        u = as_xtensor(weighted_zerosum_direction(weights.values), dims=(self.dim,))
        return u.isel({self.dim: slice(None, -1)}), u.isel({self.dim: -1})

    def forward(self, value: XTensorVariable, *rv_inputs: Variable) -> XTensorVariable:
        """Map a value on the constraint hyperplane to unconstrained coordinates."""
        u_head, u_last = self._weight_direction(rv_inputs)
        coef = value.isel({self.dim: -1}) / (1 + u_last)
        return value.isel({self.dim: slice(None, -1)}) - coef * u_head

    def backward(self, value: XTensorVariable, *rv_inputs: Variable) -> XTensorVariable:
        """Map unconstrained coordinates onto the constraint hyperplane."""
        u_head, u_last = self._weight_direction(rv_inputs)
        sum_vals = (value * u_head).sum(self.dim)
        head = value - sum_vals / (1 + u_last) * u_head
        return ptx.concat([head, -sum_vals], dim=self.dim)

    def log_jac_det(
        self, value: XTensorVariable, *rv_inputs: Variable
    ) -> XTensorVariable:
        """Log determinant of the Jacobian of ``backward``; zero, the map is an isometry."""
        return as_xtensor(0.0).broadcast_like(value, exclude=(self.dim,))


class DimWeightedZeroSumNormal(VectorDimDistribution):
    """Weighted zero-sum multivariate normal distribution for the ``pymc.dims`` API.

    Draws satisfy ``(weights * value).sum(core_dim) = 0``. Generalization of
    :class:`pymc.dims.ZeroSumNormal`; with equal weights the two coincide. Exactly one
    core dimension is supported.

    Parameters
    ----------
    sigma : xtensor_like, optional
        The standard deviation of the underlying unconstrained normal
        distribution. Defaults to 1.0. It cannot have core dimensions.
    weights : xtensor_like
        Strictly positive weights along the single core dimension.
    core_dims : str or Sequence of str
        The single dimension along which the constraint is applied.
    **kwargs
        Additional keyword arguments used to define the distribution.

    Returns
    -------
    XTensorVariable
        An xtensor variable representing the weighted zero-sum normal
        distribution.
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
        """Create the variable in the current model and set up its default transform."""
        if core_dims is not None:
            if isinstance(core_dims, str):
                core_dims = (core_dims,)

            # The transform reads the weights from the RV inputs, so it only needs the dim
            if observed is None and default_transform is UNSET:
                default_transform = DimWeightedZeroSumTransform(dim=core_dims[-1])

        # If the user didn't specify dims, take it from core_dims
        # We need them to be forwarded to dist in the `dim_lengths` argument
        if dims is None and core_dims is not None:
            dims = (..., *core_dims)

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
        """Create an unnamed random variable with validated weights."""
        if isinstance(core_dims, str):
            core_dims = (core_dims,)
        if core_dims is None or len(core_dims) != 1:
            raise ValueError("DimWeightedZeroSumNormal requires exactly one core_dims")
        if weights is None:
            raise ValueError("DimWeightedZeroSumNormal requires weights")
        (core_dim,) = core_dims

        weights = _weights_as_xtensor(weights, core_dim)

        sigma = cls._as_xtensor(sigma)

        # The weights define the length of the core dim; check it agrees with the model
        expected_n = dim_lengths.get(core_dim)
        if expected_n is not None:
            weights = cls._check_weights_length(weights, expected_n, core_dim)

        return super().dist(
            [sigma, weights], core_dims=core_dims, dim_lengths=dim_lengths, **kwargs
        )

    @staticmethod
    def _check_weights_length(
        weights: XTensorVariable, expected_n: Variable, core_dim: str
    ) -> XTensorVariable:
        msg = f"length of weights does not match the length of dim {core_dim!r}"
        static_n = weights.type.shape[0]
        expected_n_static = _static_length(expected_n)
        if (
            static_n is not None
            and expected_n_static is not None
            and static_n != expected_n_static
        ):
            raise ValueError(
                f"{msg}: got {static_n} weights for a length of {expected_n_static}"
            )
        if isinstance(expected_n, Constant):
            return weights
        # Shared or symbolic length: check again at runtime in case it changes
        checked = Assert(msg=msg)(
            weights.values, pt.eq(weights.values.shape[0], expected_n)
        )
        return as_xtensor(checked, dims=(core_dim,))

    @classmethod
    def xrv_op(
        cls,
        sigma: TensorLike | XTensorVariable,
        weights: TensorLike | XTensorVariable,
        core_dims: Sequence[str],
        extra_dims: dict[str, Variable] | None = None,
        rng: Variable | None = None,
        **kwargs,
    ) -> XTensorVariable:
        """Build the xtensor random variable from the tensor core op."""
        sigma = cls._as_xtensor(sigma)
        weights = as_xtensor(weights)
        # Only the core op is needed here; alignment of batch and core dims is
        # done by `as_xrv` when calling `xop` below
        core_rv = WeightedZeroSumNormalRV.rv_op(
            sigma=pt.scalar(dtype=sigma.type.dtype), weights=weights.values
        ).owner.op
        xop = pxr.as_xrv(
            core_rv,
            core_inps_dims_map=[(), (0,)],
            core_out_dims_map=(0,),
        )
        return xop(
            sigma,
            weights,
            core_dims=core_dims,
            extra_dims=extra_dims,
            rng=rng,
            **kwargs,
        )
