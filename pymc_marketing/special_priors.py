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

The Prior class has certain design constraints that prevent it from
covering all cases. So this module contains a collection of
priors that do not inherit from the Prior class but have many
of the same methods.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc_extras.deserialize import deserialize, register_deserialization
from pymc_extras.prior import Prior, create_dim_handler, sample_prior
from pytensor.tensor import TensorVariable


class LogNormalPrior:
    r"""Lognormal prior parameterized by positive-scale mean and std.

    A lognormal prior parameterized by mean and standard deviation
    on the positive domain, with optional centered or non-centered
    parameterization.

    This prior differs from the standard ``LogNormal`` distribution,
    which takes log-scale parameters (``mu_log``, ``sigma_log``).
    Instead, it is parameterized directly in terms of the mean and
    standard deviation (``mean``, ``std``) on the positive scale, making it more intuitive
    and suitable for hierarchical modeling.

    To achieve this, the lognormal parameters are computed internally
    from the positive-domain parameters:

    .. math::

        \mu_{\log} &= \ln \left( \frac{\mean^2}{\sqrt{\mean^2 + \std^2}} \right) \\
        \sigma_{\log} &= \sqrt{ \ln \left( 1 + \frac{\std^2}{\mean^2} \right) }

    where :math:`\\mean > 0` and :math:`\\std > 0`.

    The prior is then defined as:

    .. math::

        \\phi &\\sim \text{LogNormal}(\\mu_{\\log}, \\sigma_{\\log})

    This construction ensures that the resulting random variable
    has approximately the intended mean and variance on the positive scale,
    even when :math:`\\mean` and :math:`\\std` are themselves random variables.

    Parameters
    ----------
    mean : Prior, float, int, array-like
        The mean of the distribution on the positive scale.
    std : Prior, float, int, array-like
        The standard deviation of the distribution on the positive scale.
    dims : tuple[str, ...], optional
        The dimensions of the distribution, by default None.
    centered : bool, optional
        Whether to use the centered parameterization, by default True.

    Examples
    --------
    Build a non-centered hierarchical model where information is shared across groups:

    .. code-block:: python

        from pymc_marketing.special_priors import LogNormalPrior

        prior = LogNormalPrior(
            mean=Prior("Gamma", mu=1.0, sigma=1.0),
            std=Prior("HalfNormal", sigma=1.0),
            dims=("geo",),
            centered=False,
        )

    References
    ----------
    - D. Saunders, *A positive constrained non-centered prior that sparks joy*.
    - Wikipedia, *Log-normal distribution — Definitions*.
    """

    def __init__(self, dims: tuple | None = None, centered: bool = True, **parameters):
        self.parameters = parameters
        self.dims = dims
        self.centered = centered

        self._checks()

    def _checks(self) -> None:
        self._parameters_are_correct_set()

    def _parameters_are_correct_set(self) -> None:
        if set(self.parameters.keys()) != {"mean", "std"}:
            raise ValueError("Parameters must be mean and std")

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
            parameters["mean"] ** 2
            / pt.sqrt(parameters["mean"] ** 2 + parameters["std"] ** 2)
        )
        sigma_log = pt.sqrt(
            pt.log(1 + (parameters["std"] ** 2 / parameters["mean"] ** 2))
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
            "special_prior": "LogNormalPrior",
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
        """Create a LogNormalPrior prior from a dictionary."""
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


def _is_LogNormalPrior_type(data: dict) -> bool:
    if "special_prior" in data:
        return data["special_prior"] == "LogNormalPrior"
    else:
        return False


register_deserialization(
    is_type=_is_LogNormalPrior_type,
    deserialize=LogNormalPrior.from_dict,
)
