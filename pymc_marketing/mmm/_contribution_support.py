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
"""Conservative sign checks at the historical contribution-factor boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from pymc.distributions.transforms import _default_transform
from pymc.logprob.transforms import IntervalTransform, LogOddsTransform, LogTransform
from pytensor.compile.ops import ViewOp
from pytensor.graph.basic import Constant
from pytensor.graph.traversal import graph_inputs
from pytensor.scalar.basic import Exp
from pytensor.scalar.math import Sigmoid, Softplus
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.vectorization import XRV

from pymc_marketing.mmm.components.adstock import (
    BinomialAdstock,
    DelayedAdstock,
    GeometricAdstock,
    NoAdstock,
    WeibullCDFAdstock,
    WeibullPDFAdstock,
)
from pymc_marketing.mmm.components.saturation import (
    HillSaturation,
    HillSaturationSigmoid,
    InverseScaledLogisticSaturation,
    LogisticSaturation,
    LogSaturation,
    MichaelisMentenSaturation,
    NoSaturation,
    RootSaturation,
    TanhSaturation,
    TanhSaturationBaselined,
)
from pymc_marketing.mmm.hsgp import HSGPBase, SoftPlusHSGP
from pymc_marketing.mmm.transformers import ConvMode
from pymc_marketing.names.mmm import Dim, Var

if TYPE_CHECKING:
    from pymc_marketing.mmm.mmm import MMM


_ADSTOCKS = {
    BinomialAdstock,
    GeometricAdstock,
    DelayedAdstock,
    NoAdstock,
    WeibullPDFAdstock,
    WeibullCDFAdstock,
}
_SATURATIONS = {
    LogisticSaturation,
    InverseScaledLogisticSaturation,
    MichaelisMentenSaturation,
    TanhSaturation,
    TanhSaturationBaselined,
    HillSaturation,
    HillSaturationSigmoid,
    RootSaturation,
    LogSaturation,
    NoSaturation,
}


def _constant_bound(value: Any, fallback: float) -> float | None:
    """Read a distribution bound only when it has no mutable/stochastic inputs."""
    if value is None:
        return fallback
    tensor = value.values if hasattr(value, "values") else value
    if not all(isinstance(v, Constant) for v in graph_inputs([tensor])):
        return None
    values = np.asarray(tensor.eval())
    return float(values.min() if fallback < 0 else values.max())


def _support_bounds(variable: Any) -> tuple[float, float] | None:
    """Read support from the actual parameter graph, never configuration mirrors."""
    while variable.owner and isinstance(variable.owner.op, ViewOp):
        variable = variable.owner.inputs[0]
    if isinstance(variable, Constant):
        values = np.asarray(variable.data)
        return float(values.min()), float(values.max())
    if variable.owner is None:
        return None
    op = variable.owner.op
    # Includes transformed Normal priors and the special LogNormalPrior factory.
    scalar = getattr(op, "scalar_op", None)
    if isinstance(scalar, (Exp, Softplus)):
        return 0.0, np.inf
    if isinstance(scalar, Sigmoid):
        return 0.0, 1.0
    if not isinstance(op, XRV):
        return None
    # Use the distribution's canonical support transform, not a user-selected
    # value-variable transform (which need not describe distribution support).
    transform = _default_transform(op.core_op, None)
    if isinstance(transform, LogTransform):
        return 0.0, np.inf
    if isinstance(transform, LogOddsTransform):
        return 0.0, 1.0
    if isinstance(transform, IntervalTransform):
        params = [v.values for v in op.dist_params(variable.owner)]
        lower, upper = transform.args_fn(None, None, *params)
        lo, hi = _constant_bound(lower, -np.inf), _constant_bound(upper, np.inf)
        if lo is not None and hi is not None:
            return lo, hi
    return None


def _parameter_error(mmm: MMM) -> str | None:
    """Certify built-in response families using their constructed parameters."""
    for transformation, supported in (
        (mmm.adstock, _ADSTOCKS),
        (mmm.saturation, _SATURATIONS),
    ):
        if type(transformation) not in supported:
            return f"the sign of {type(transformation).__name__} is not established"
        for parameter, variable in transformation.variable_mapping.items():
            built = mmm.model.named_vars.get(variable)
            if built is None:
                # Fixed transformation parameters are passed straight to the
                # response function and create no named graph variable.
                fixed = transformation.function_priors[parameter]
                if not isinstance(fixed, (int, float, np.number, np.ndarray)):
                    return f"the sign of {variable} is not established"
                built = as_xtensor(np.array(fixed, copy=True))
            bounds = _support_bounds(built)
            if bounds is None or bounds[0] < 0:
                return f"{variable} must have nonnegative support (its sign is unknown or signed)"
            if (
                type(transformation) is BinomialAdstock
                or (
                    type(transformation) is TanhSaturationBaselined
                    and variable == transformation.variable_mapping["r"]
                )
            ) and bounds[1] > 1:
                return f"{variable} must have support within [0, 1]"
    multiplier = mmm.time_varying_media
    if isinstance(multiplier, HSGPBase) and type(multiplier) is not SoftPlusHSGP:
        return "the media multiplier must be the nonnegative SoftPlusHSGP"
    return None


@dataclass(frozen=True)
class ContributionSupport:
    """Build-time sign and temporal-reach facts for one historical graph.

    Reach is conservative: every lag in a built-in kernel may carry spend.
    It establishes impossible zero targets without claiming that every parameter
    value has positive density. Neither prediction nor later factory edits can
    alter these facts.
    """

    parameter_error: str | None
    negative_inputs: xr.DataArray
    positive_reach: xr.DataArray

    @classmethod
    def capture(cls, mmm: MMM) -> ContributionSupport:
        """Capture facts after constructing the graph and before any prediction."""
        data = mmm.model[Var.CHANNEL_DATA]
        channels = xr.DataArray(
            np.array(data.get_value(), copy=True),
            dims=data.dims,
            coords={dim: mmm.model_coords[dim] for dim in data.dims},
        )
        scale = mmm.model[Var.CHANNEL_SCALE]
        scales = xr.DataArray(
            np.array(scale.get_value(), copy=True),
            dims=scale.dims,
            coords={dim: mmm.model_coords[dim] for dim in scale.dims},
        )
        negative = ((channels < 0) | (scales < 0)).any(
            dim=[dim for dim in channels.dims if dim != Dim.CHANNEL]
        )
        positive = (channels > 0) & (scales > 0)
        reach = positive.copy()
        if type(mmm.adstock) in _ADSTOCKS and type(mmm.adstock) is not NoAdstock:
            width = mmm.adstock.l_max
            left = {
                ConvMode.After: width - 1,
                ConvMode.Before: 0,
                ConvMode.Overlap: width // 2,
            }[mmm.adstock.mode]
            for shift in range(left - width + 1, left + 1):
                reach = reach | positive.shift({Dim.DATE: shift}, fill_value=False)
        return cls(_parameter_error(mmm), negative, reach)

    def nonnegative_error(self, channels: list[str]) -> str | None:
        """Explain why a selected direct contribution is not proven nonnegative."""
        if self.parameter_error is not None:
            return self.parameter_error
        if bool(self.negative_inputs.sel({Dim.CHANNEL: channels}).any()):
            return "historical channel inputs and channel scales must be nonnegative"
        return None

    def has_positive_target(self, channels: list[str], mask: xr.DataArray) -> bool:
        """Use the same labeled observed cells as the contribution aggregate."""
        return bool((self.positive_reach.sel({Dim.CHANNEL: channels}) & mask).any())
