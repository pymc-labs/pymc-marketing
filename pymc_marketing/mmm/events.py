#   Copyright 2025 - 2025 The PyMC Labs Developers
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
"""Event transformations."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pydantic import BaseModel, Field, InstanceOf, validate_call
from pytensor.tensor.variable import TensorVariable

from pymc_marketing.deserialize import deserialize, register_deserialization
from pymc_marketing.mmm.components.base import Transformation, create_registration_meta
from pymc_marketing.prior import Prior, create_dim_handler

BASIS_TRANSFORMATIONS: dict = {}
BasisMeta = create_registration_meta(BASIS_TRANSFORMATIONS)


class Basis(Transformation, metaclass=BasisMeta):  # type: ignore[misc]
    """Basis transformation associated with an event model."""

    prefix: str = "basis"
    lookup_name: str

    @validate_call
    def sample_curve(
        self,
        parameters: InstanceOf[xr.Dataset] = Field(
            ..., description="Parameters of the saturation transformation."
        ),
        days: int = Field(0, ge=0, description="Minimum number of days."),
    ) -> xr.DataArray:
        """Sample the curve of the saturation transformation given parameters.

        Parameters
        ----------
        parameters : xr.Dataset
            Dataset with the parameters of the saturation transformation.

        Returns
        -------
        xr.DataArray
            Curve of the saturation transformation.

        """
        x = np.linspace(-days, days, 100)

        coords = {"x": x}

        return self._sample_curve(
            var_name="saturation",
            parameters=parameters,
            x=x,
            coords=coords,
        )


def basis_from_dict(data: dict) -> Basis:
    """Create a basis transformation from a dictionary."""
    data = data.copy()
    lookup_name = data.pop("lookup_name")
    cls = BASIS_TRANSFORMATIONS[lookup_name]

    if "priors" in data:
        data["priors"] = {k: deserialize(v) for k, v in data["priors"].items()}

    return cls(**data)


def _is_basis(data):
    return "lookup_name" in data and data["lookup_name"] in BASIS_TRANSFORMATIONS


register_deserialization(
    is_type=_is_basis,
    deserialize=basis_from_dict,
)


class EventEffect(BaseModel):
    """Event effect associated with an event model."""

    basis: Basis
    effect_size: Prior
    dims: tuple[str, ...]

    def apply(self, X: pt.TensorLike, name: str = "event") -> TensorVariable:
        """Apply the event effect to the data."""
        dim_handler = create_dim_handler(("x", *self.dims))
        return self.basis.apply(X, dims=self.dims) * dim_handler(
            self.effect_size.create_variable(f"{name}_effect_size"),
            self.effect_size.dims,
        )

    def to_dict(self) -> dict:
        """Convert the event effect to a dictionary."""
        return {
            "class": "EventEffect",
            "data": {
                "basis": self.basis.to_dict(),
                "effect_size": self.effect_size.to_dict(),
                "dims": self.dims,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EventEffect":
        """Create an event effect from a dictionary."""
        return cls(
            basis=deserialize(data["basis"]),
            effect_size=deserialize(data["effect_size"]),
            dims=data["dims"],
        )


def _is_event_effect(data: dict) -> bool:
    """Check if the data is an event effect."""
    return data["class"] == "EventEffect"


register_deserialization(
    is_type=_is_event_effect,
    deserialize=lambda data: EventEffect.from_dict(data["data"]),
)


class GaussianBasis(Basis):
    """Gaussian basis transformation."""

    lookup_name = "gaussian"

    def function(self, x: pt.TensorLike, sigma: pt.TensorLike) -> TensorVariable:
        """Gaussian bump function."""
        return pm.math.exp(-0.5 * (x / sigma) ** 2)

    default_priors = {
        "sigma": Prior("Gamma", mu=7, sigma=1),
    }
