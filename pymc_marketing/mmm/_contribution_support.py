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

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from pymc_extras.prior import Prior

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
from pymc_marketing.names.mmm import Var
from pymc_marketing.special_priors import LogNormalPrior

if TYPE_CHECKING:
    from pymc_marketing.mmm.mmm import MMM


# These families have parameter-independent support, even when their scale or
# other parameters have hierarchical priors that PreliZ cannot instantiate.
_NONNEGATIVE_FAMILIES = {
    "HalfNormal",
    "LogNormal",
    "Gamma",
    "InverseGamma",
    "Exponential",
    "Weibull",
    "HalfStudentT",
    "ChiSquared",
}
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


def _support_bounds(factory: Any) -> tuple[float, float] | None:
    """Return proven bounds, leaving custom factories/transforms unknown."""
    if isinstance(factory, LogNormalPrior):
        return 0.0, np.inf
    if isinstance(factory, Prior):
        if factory.transform is not None:
            return None
        if factory.distribution in _NONNEGATIVE_FAMILIES:
            return 0.0, np.inf
        if factory.distribution == "Beta":
            return 0.0, 1.0
        try:
            lower, upper = factory.preliz.support
            return float(np.min(lower)), float(np.max(upper))
        except (TypeError, ValueError, AttributeError):
            return None
    if isinstance(factory, (int, float, np.number, np.ndarray)):
        values = np.asarray(factory)
        if np.isfinite(values).all():
            return float(values.min()), float(values.max())
    return None


def nonnegative_contribution_error(
    mmm: MMM, channels: list[str], inputs: dict[str, np.ndarray]
) -> str | None:
    """Explain why the direct contribution cannot be proven nonnegative.

    Exact built-in types prevent custom subclasses from inheriting a sign claim
    after overriding the response function. This proves sign, not feasibility of
    every parameter combination or the density's requested concentration.
    """
    for transformation, supported in (
        (mmm.adstock, _ADSTOCKS),
        (mmm.saturation, _SATURATIONS),
    ):
        if type(transformation) not in supported:
            return f"the sign of {type(transformation).__name__} is not established"
        for variable in transformation.variable_mapping.values():
            bounds = _support_bounds(mmm.model_config[variable])
            if bounds is None or bounds[0] < 0:
                return f"{variable} must have nonnegative support (its sign is unknown or signed)"
            # Binomial kernels involve (1 - alpha), and baselined tanh uses arctanh(r).
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
    channel_data = xr.DataArray(
        inputs[Var.CHANNEL_DATA],
        dims=mmm.model[Var.CHANNEL_DATA].dims,
        coords={dim: mmm.model_coords[dim] for dim in mmm.model[Var.CHANNEL_DATA].dims},
    ).sel(channel=channels)
    if bool((channel_data < 0).any()) or np.any(inputs[Var.CHANNEL_SCALE] < 0):
        return "historical channel inputs and channel scales must be nonnegative"
    return None
