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
"""Dims related functionality."""

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import pymc as pm
from pydantic import InstanceOf
from pymc.dims.distributions.core import DimDistribution
from pymc_extras.deserialize import deserialize
from pymc_extras.prior import (
    MuAlreadyExistsError,
    Prior,
    UnsupportedDistributionError,
    _get_pymc_parameters,
    _remove_random_variable,
    sample_prior,
)
from pytensor import Variable
from pytensor.tensor import TensorLike, TensorVariable
from pytensor.xtensor.shape import Transpose
from pytensor.xtensor.type import XTensorVariable, as_xtensor
from pytensor.xtensor.vectorization import XRV
from xarray import DataArray, Dataset

# TODO: This will eventually exist in PyTensor or PyMC, remove then
XTensorLike: TypeAlias = TensorLike | XTensorVariable | DataArray


# All this Censored code belongs in https://github.com/pymc-devs/pymc/pull/8133
def expand_dist_dims(dist, extra_dims: dict[str, Any]):
    """Add extra dims to a dist variable."""
    dist = as_xtensor(dist)

    if overlap := (set(extra_dims) & set(dist.dims)):
        raise ValueError(
            f"extra_dims already present in distribution: {sorted(overlap)}"
        )

    op = None if dist.owner is None else dist.owner.op
    match op:
        case XRV():
            # Recreate dist with new extra dims
            dist_props = dist.owner.op._props_dict()
            dist_props["extra_dims"] = (*(extra_dims.keys()), *dist_props["extra_dims"])
            new_dist_op = type(dist.owner.op)(**dist_props)
            _old_rng, *params_and_dim_lengths = dist.owner.inputs
            new_rng = None  # We don't propaget the old RNG, because we don't want the dists to be correlated
            return new_dist_op(new_rng, *extra_dims.values(), *params_and_dim_lengths)
        case Transpose():
            return expand_dist_dims(
                dist.owner.inputs[0], extra_dims=extra_dims
            ).transpose(..., *dist.dims)
        case _:
            raise NotImplementedError(
                f"expand_dist_dims not implemented for {dist} with op {op}"
            )


class _Censored(DimDistribution):
    @classmethod
    def dist(cls, dist, *, lower=None, upper=None, dim_lengths, **kwargs):
        if lower is None:
            lower = -np.inf
        if upper is None:
            upper = np.inf
        return super().dist([dist, lower, upper], dim_lengths=dim_lengths, **kwargs)

    @classmethod
    def xrv_op(cls, dist, lower, upper, core_dims=None, extra_dims=None, rng=None):
        if extra_dims is None:
            extra_dims = {}

        dist = cls._as_xtensor(dist)
        lower = cls._as_xtensor(lower)
        upper = cls._as_xtensor(upper)

        # Any dimensions in extra_dims, or only present in lower, upper,
        # must propagate back to the dist as `extra_dims`
        bounds_sizes = lower.sizes | upper.sizes
        dist_dims_set = set(dist.dims)
        extra_dist_dims = extra_dims | {
            dim: size for dim, size in bounds_sizes.items() if dim not in dist_dims_set
        }
        if extra_dist_dims:
            dist = expand_dist_dims(dist, extra_dist_dims)

        # Probability is inferred from the clip operation
        # TODO: Make this a SymbolicRandomVariable that can itself be resized
        return dist.clip(lower, upper)


@dataclass
class Censored:
    """Create censored random variable.

    Examples
    --------
    Create a censored Normal distribution:

    .. code-block:: python

        from pymc_extras.prior import Prior, Censored

        normal = Prior("Normal")
        censored_normal = Censored(normal, lower=0)

    Create hierarchical censored Normal distribution:

    .. code-block:: python

        from pymc_extras.prior import Prior, Censored

        normal = Prior(
            "Normal",
            mu=Prior("Normal"),
            sigma=Prior("HalfNormal"),
            dims="channel",
        )
        censored_normal = Censored(normal, lower=0)

        coords = {"channel": range(3)}
        samples = censored_normal.sample_prior(coords=coords)

    """

    distribution: InstanceOf[Prior]
    lower: float | InstanceOf[Variable] = -np.inf
    upper: float | InstanceOf[Variable] = np.inf

    def __post_init__(self) -> None:
        """Check validity at initialization."""
        if not self.distribution.centered:
            raise ValueError(
                "Censored distribution must be centered so that .dist() API can be used on distribution."
            )

        if self.distribution.transform is not None:
            raise ValueError(
                "Censored distribution can't have a transform so that .dist() API can be used on distribution."
            )

    @property
    def dims(self) -> tuple[str, ...] | None:
        """The dims from the distribution to censor."""
        return self.distribution.dims

    @dims.setter
    def dims(self, dims) -> None:
        self.distribution.dims = dims

    def create_variable(
        self, name: str, xdist: bool = False
    ) -> TensorVariable | XTensorVariable:
        """Create censored random variable."""
        dist = self.distribution.create_variable(name, xdist=xdist)
        _remove_random_variable(var=dist)
        if xdist:
            censored_constructor = _Censored
        else:
            censored_constructor = pm.Censored
        return censored_constructor(
            name, dist, lower=self.lower, upper=self.upper, dims=self.dims
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the censored distribution to a dictionary."""

        def handle_value(value):
            if isinstance(value, TensorVariable):
                return value.eval().tolist()

            return value

        return {
            "class": "Censored",
            "data": {
                "dist": self.distribution.to_dict(),
                "lower": handle_value(self.lower),
                "upper": handle_value(self.upper),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Censored":
        """Create a censored distribution from a dictionary."""
        data = data["data"]
        return cls(  # type: ignore
            distribution=deserialize(data["dist"]),
            lower=data["lower"],
            upper=data["upper"],
        )

    def sample_prior(
        self,
        coords=None,
        name: str = "variable",
        xdist: bool = False,
        **sample_prior_predictive_kwargs,
    ) -> Dataset:
        """Sample the prior distribution for the variable.

        Parameters
        ----------
        coords : dict[str, list[str]], optional
            The coordinates for the variable, by default None.
            Only required if the dims are specified.
        name : str, optional
            The name of the variable, by default "var".
        sample_prior_predictive_kwargs : dict
            Additional arguments to pass to `pm.sample_prior_predictive`.

        Returns
        -------
        Dataset
            The dataset of the prior samples.

        Example
        -------
        Sample from a censored Gamma distribution.

        .. code-block:: python

            gamma = Prior("Gamma", mu=1, sigma=1, dims="channel")
            dist = Censored(gamma, lower=0.5)

            coords = {"channel": ["C1", "C2", "C3"]}
            prior = dist.sample_prior(coords=coords)

        """
        return sample_prior(
            factory=self,
            coords=coords,
            name=name,
            xdist=xdist,
            **sample_prior_predictive_kwargs,
        )

    def to_graph(self):
        """Generate a graph of the variables.

        Examples
        --------
        Create graph for a censored Normal distribution

        .. code-block:: python

            from pymc_extras.prior import Prior, Censored

            normal = Prior("Normal")
            censored_normal = Censored(normal, lower=0)

            censored_normal.to_graph()

        """
        coords = {name: ["DUMMY"] for name in self.dims or ()}
        with pm.Model(coords=coords) as model:
            self.create_variable("var")

        return pm.model_to_graphviz(model)

    def create_likelihood_variable(
        self,
        name: str,
        mu: XTensorLike,
        observed: XTensorLike,
        xdist: bool = False,
    ) -> TensorVariable | XTensorVariable:
        """Create observed censored variable.

        Will require that the distribution has a `mu` parameter
        and that it has not been set in the parameters.

        Parameters
        ----------
        name : str
            The name of the variable.
        mu : pt.TensorLike
            The mu parameter for the likelihood.
        observed : pt.TensorLike
            The observed data.
        xdist: bool, default False
            Whether to create a variable from pymc.dims or regular pymc distributions

        Returns
        -------
        TensorVariable or XTensorVariable
            The PyMC variable.

        Examples
        --------
        Create a censored likelihood variable in a larger PyMC model.

        .. code-block:: python

            import pymc as pm
            from pymc_extras.prior import Prior, Censored

            normal = Prior("Normal", sigma=Prior("HalfNormal"))
            dist = Censored(normal, lower=0)

            observed = 1

            with pm.Model():
                # Create the likelihood variable
                mu = pm.HalfNormal("mu", sigma=1)
                dist.create_likelihood_variable("y", mu=mu, observed=observed)

        """
        if "mu" not in _get_pymc_parameters(self.distribution.pymc_distribution):
            raise UnsupportedDistributionError(
                f"Likelihood distribution {self.distribution.distribution!r} is not supported."
            )

        if "mu" in self.distribution.parameters:
            raise MuAlreadyExistsError(self.distribution)

        distribution = self.distribution.deepcopy()
        distribution.parameters["mu"] = mu
        dist = distribution.create_variable(name, xdist=xdist)
        _remove_random_variable(var=dist)

        if xdist:
            censored_constructor = _Censored
        else:
            censored_constructor = pm.Censored
        return censored_constructor(
            name,
            dist,
            lower=self.lower,
            upper=self.upper,
            dims=self.dims,
            observed=observed,
        )
