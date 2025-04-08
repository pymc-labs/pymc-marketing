#   Copyright 2022 - 2025 The PyMC Labs Developers
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
"""Bass diffusion model for product adoption.

Adapted from Wiki: https://en.wikipedia.org/wiki/Bass_diffusion_model

"""

from typing import Any

import pymc as pm
import pytensor.tensor as pt
from pymc.model import Model

from pymc_marketing.prior import Censored, Prior, VariableFactory, create_dim_handler


def F(p, q, t):
    """Installed base fraction."""
    return (1 - pt.exp(-(p + q) * t)) / (1 + (q / p) * pt.exp(-(p + q) * t))


def f(p, q, t):
    """Installed base fraction rate of change."""
    return (p + q) * pt.exp(-(p + q) * t) / (1 + (q / p) * pt.exp(-(p + q) * t)) ** 2


def create_bass_model(
    t: pt.TensorLike,
    observed: pt.TensorLike | None,
    priors: dict[str, Prior | Censored | VariableFactory],
    coords: dict[str, Any],
) -> Model:
    """Define a Bass diffusion model."""
    with pm.Model(coords=coords) as model:
        combined_dims = (
            "date",
            *set(priors["p"].dims).union(priors["q"].dims).union(priors["m"].dims),
        )
        dim_handler = create_dim_handler(combined_dims)

        m = dim_handler(priors["m"].create_variable("m"), priors["m"].dims)
        p = dim_handler(priors["p"].create_variable("p"), priors["p"].dims)
        q = dim_handler(priors["q"].create_variable("q"), priors["q"].dims)

        time = dim_handler(t, "date")

        adopters = pm.Deterministic("adopters", m * f(p, q, time), dims=combined_dims)

        pm.Deterministic(
            "innovators",
            m * p * (1 - F(p, q, time)),
            dims=combined_dims,
        )
        pm.Deterministic(
            "imitators",
            m * q * F(p, q, time) * (1 - F(p, q, time)),
            dims=combined_dims,
        )

        pm.Deterministic(
            "peak",
            (pt.log(q) - pt.log(p)) / (p + q),
            dims=combined_dims[1:],
        )

        priors["likelihood"].dims = combined_dims
        priors["likelihood"].create_likelihood_variable(  # type: ignore
            "y",
            mu=adopters,
            observed=observed,
        )

    return model
