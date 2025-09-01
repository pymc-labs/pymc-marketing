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
"""
Specialized priors that behave like the Prior class.

The Prior class has certain design constraints that make it difficult
to adapt to some cases. So this module contains a set of collection of
priors that deviate slightly from the Prior class.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc_extras.deserialize import deserialize, register_deserialization
from pymc_extras.prior import Prior, create_dim_handler, sample_prior
from pytensor.tensor import TensorVariable


class LogNormalExp:
    """
    A specialized implementation of a log normal distribution.

    Like the LogNormal distribution, this distribution covers the positive numbers.
    However, unlike the lognormal, the parameters are also specified in the positive
    domain. This is very useful in modeling MMM parameters because they are also
    constrained to be positive and you get specify priors on a intuitive scale.

    The other advantage of this prior is in constructing hierarchical models. It allows users to toggle
    between centered and non-centered parameterization. This facilitates rapid iteration when searching
    for a parameterization that enables efficient sampling.

    Parameters
    ----------
    mu : Prior, float, int, array-like
        The prior for the mean of the distribution.
    sigma : Prior, float, int, array-like
        The prior for the standard deviation of the distribution.
    dims : tuple[str, ...], optional
        The dimensions of the distribution, by default None.
    centered : bool, optional
        Whether the distribution is centered, by default True.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.special_priors import LogNormalExp

        normal = LogNormalExp(mu=Prior("Normal"), sigma=Prior("HalfNormal"))
    """

    def __init__(self, dims: tuple | None = None, centered: bool = True, **parameters):
        self.parameters = parameters
        self.dims = dims
        self.centered = centered

        self._checks()

    def _checks(self) -> None:
        self._parameters_are_correct_set()

    def _parameters_are_correct_set(self) -> None:
        if set(self.parameters.keys()) != {"mu", "sigma"}:
            raise ValueError("Parameters must be mu and sigma")

    def _create_parameter(self, param, value, name):
        if not hasattr(value, "create_variable"):
            return value

        child_name = f"{name}_{param}"
        return self.dim_handler(value.create_variable(child_name), value.dims)

    def create_variable(self, name: str) -> TensorVariable:
        """Create a variable from the prior distribution."""
        self.dim_handler = create_dim_handler(self.dims)
        parameters = {
            param: self._create_parameter(param, value, name)
            for param, value in self.parameters.items()
        }
        mu_log = pt.log(
            parameters["mu"] ** 2
            / pt.sqrt(parameters["mu"] ** 2 + parameters["sigma"] ** 2)
        )
        sigma_log = pt.sqrt(
            pt.log(1 + (parameters["sigma"] ** 2 / parameters["mu"] ** 2))
        )

        if self.centered:
            log_phi = pm.Normal(
                name + "_log", mu=mu_log, sigma=sigma_log, dims=self.dims
            )

        else:
            log_phi_z = pm.Normal(
                name + "_log" + "_offset", mu=0, sigma=1, dims=self.dims
            )
            log_phi = mu_log + log_phi_z * sigma_log

        phi = pm.math.exp(log_phi)
        phi = pm.Deterministic(name, phi, dims=self.dims)

        return phi

    def to_dict(self):
        """Convert the prior distribution to a dictionary."""
        data = {
            "special_prior": "LogNormalExp",
        }
        if self.parameters:

            def handle_value(value):
                if isinstance(value, Prior):
                    return value.to_dict()

                if isinstance(value, pt.TensorVariable):
                    value = value.eval()

                if isinstance(value, np.ndarray):
                    return value.tolist()

                if hasattr(value, "to_dict"):
                    return value.to_dict()

                return value

            data["kwargs"] = {
                param: handle_value(value) for param, value in self.parameters.items()
            }
        if not self.centered:
            data["centered"] = False

        if self.dims:
            data["dims"] = self.dims

        return data

    @classmethod
    def from_dict(cls, data) -> Prior:
        """Create a LogNormalExp prior from a dictionary."""
        if not isinstance(data, dict):
            msg = (
                "Must be a dictionary representation of a prior distribution. "
                f"Not of type: {type(data)}"
            )
            raise ValueError(msg)

        kwargs = data.get("kwargs", {})

        def handle_value(value):
            if isinstance(value, dict):
                return deserialize(value)

            if isinstance(value, list):
                return np.array(value)

            return value

        kwargs = {param: handle_value(value) for param, value in kwargs.items()}
        centered = data.get("centered", True)
        dims = data.get("dims")
        if isinstance(dims, list):
            dims = tuple(dims)

        return cls(dims=dims, centered=centered, **kwargs)

    def sample_prior(
        self,
        coords=None,
        name: str = "variable",
        **sample_prior_predictive_kwargs,
    ) -> xr.Dataset:
        """Sample from the prior distribution."""
        return sample_prior(
            factory=self,
            coords=coords,
            name=name,
            **sample_prior_predictive_kwargs,
        )


def _is_log_normal_exp_type(data: dict) -> bool:
    if "special_prior" in data:
        return data["special_prior"] == "LogNormalExp"
    else:
        return False


register_deserialization(
    is_type=_is_log_normal_exp_type, deserialize=LogNormalExp.from_dict
)
