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

"""R2-D2-M2 prior for variance decomposition in regression models.

The R2-D2-M2 prior (Zhang et al., 2022) provides automatic shrinkage and
variable selection by placing a Dirichlet distribution on the R-squared of a
regression model, then allocating explained variance across coefficients.

Reference: https://arxiv.org/abs/2208.07132

Example
-------
>>> from pymc_marketing.r2d2 import R2D2Decomposition
>>> from pymc_extras.prior import Prior
>>>
>>> r2d2 = R2D2Decomposition(
...     r2=Prior("Beta", mu=0.8, sigma=0.4),
...     total_sigma=Prior("LogNormal", mu=0, sigma=1),
...     dims={"control": "control", "fourier": "fourier"},
... )
>>>
>>> # Use splits as priors in model config
>>> model_config = {
...     "gamma_control": r2d2.split("control"),
...     "gamma_fourier": r2d2.split("fourier"),
...     "likelihood": Prior("Normal", sigma=r2d2.error_sigma),
... }
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytensor.tensor as pt
import pytensor.xtensor.math as ptx
from pymc_extras.deserialize import deserialize
from pymc_extras.prior import Prior

from pymc_marketing.serialization import serialization


@serialization.register
@dataclass
class R2D2Split:
    """Lazy reference to a component's split variable.

    Implements VariableFactory protocol so it can be used
    wherever a Prior is accepted (e.g., in model_config priors).

    When create_variable() is called, auto-builds the decomposition
    if not already built, then creates a pmd.Normal coefficient with
    mu=0, sigma=split, dims=component_dim.

    Parameters
    ----------
    decomposition : R2D2Decomposition
        The parent decomposition.
    component_name : str
        Name of the component.
    """

    decomposition: "R2D2Decomposition"
    component_name: str
    _dims: tuple[str, ...] | None = field(default=None, repr=False)

    def __deepcopy__(self, memo):
        """Preserve decomposition reference across copies.

        This ensures that when the MMM deepcopies priors,
        all R2D2Split copies share the SAME decomposition.
        """
        return R2D2Split(self.decomposition, self.component_name, self._dims)

    @property
    def dims(self) -> tuple[str, ...] | None:
        """Dimension of the split variable."""
        if self._dims is not None:
            return self._dims
        # Look up the dim name for this component
        for comp, dim in self.decomposition.dims.items():
            if comp == self.component_name:
                return (dim,)
        return None

    @dims.setter
    def dims(self, value: tuple[str, ...] | None) -> None:
        self._dims = value

    def create_variable(self, name: str, xdist: bool = False) -> pt.TensorVariable:
        """Create coefficient variable. Auto-builds decomposition if needed.

        Parameters
        ----------
        name : str
            Name for the PyMC variable.
        xdist : bool
            Whether to use xdist mode (ignored, kept for protocol compatibility).

        Returns
        -------
        pt.TensorVariable
            The created coefficient variable.
        """
        # Lazy auto-build: create decomposition if not yet built
        if not self.decomposition._built:
            self.decomposition.create_variable("r2d2")

        # Get the variance split for this component
        split = self.decomposition._splits[self.component_name]
        dims = self.decomposition.dims[self.component_name]

        # Create coefficient: beta ~ pmd.Normal(mu=0, sigma=split, dims=dims)
        return pmd.Normal(name, mu=0, sigma=split, dims=dims)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "decomposition": serialization.serialize(self.decomposition),
            "component_name": self.component_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "R2D2Split":
        """Deserialize from dictionary."""
        return cls(
            decomposition=serialization.deserialize(data["decomposition"]),
            component_name=data["component_name"],
        )


@serialization.register
@dataclass
class R2D2Sigma:
    """Lazy reference to the error sigma variable.

    Implements VariableFactory protocol so it can be used
    wherever a Prior is accepted (e.g., in likelihood sigma).

    Returns the scalar error_sigma variable directly (no coefficient creation).
    """

    decomposition: "R2D2Decomposition"

    def __deepcopy__(self, memo):
        """Preserve decomposition reference across copies."""
        return R2D2Sigma(self.decomposition)

    @property
    def dims(self) -> tuple[str, ...]:
        """Error sigma is scalar - no dimensions."""
        return ()

    def create_variable(self, name: str, xdist: bool = False) -> pt.TensorVariable:
        """Return the error sigma variable. Auto-builds decomposition if needed.

        Parameters
        ----------
        name : str
            Name for the PyMC variable (ignored for error_sigma).
        xdist : bool
            Whether to use xdist mode (ignored, kept for protocol compatibility).

        Returns
        -------
        pt.TensorVariable
            The error sigma variable.
        """
        # Lazy auto-build: create decomposition if not yet built
        if not self.decomposition._built:
            self.decomposition.create_variable("r2d2")

        return self.decomposition._error_sigma

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "decomposition": serialization.serialize(self.decomposition),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "R2D2Sigma":
        """Deserialize from dictionary."""
        return cls(
            decomposition=serialization.deserialize(data["decomposition"]),
        )


@serialization.register
@dataclass
class R2D2Decomposition:
    """R2-D2-M2 variance decomposition.

    Splits total variance between model and error, then splits
    model variance across components via Dirichlet.

    Variables are created once via create_variable(), then
    referenced by split() and error_sigma.

    .. note::
        The decomposition allocates variance ONLY across the components
        defined in ``dims``. Any covariates not included in ``dims`` will
        not be covered by the variance decomposition. Ensure all relevant
        covariates are included in ``dims``.

    Parameters
    ----------
    r2 : Prior
        R² prior (how much variance the model explains).
        Typically Beta(mu=0.8, sigma=0.4).
    total_sigma : Prior
        Total scale of the data.
        Typically LogNormal(mu=np.log(std(y)), sigma=0.1).
    dims : dict[str, str]
        Maps component name to dim name.
        E.g., {"control": "control", "fourier": "fourier"}.

    Example
    -------
    >>> r2d2 = R2D2Decomposition(
    ...     r2=Prior("Beta", mu=0.8, sigma=0.4),
    ...     total_sigma=Prior("LogNormal", mu=0, sigma=1),
    ...     dims={"control": "control", "fourier": "fourier"},
    ... )
    >>> # Use in model config
    >>> config = {
    ...     "gamma_control": r2d2.split("control"),
    ...     "gamma_fourier": r2d2.split("fourier"),
    ...     "likelihood": Prior("Normal", sigma=r2d2.error_sigma),
    ... }
    """

    r2: Prior
    total_sigma: Prior
    dims: dict[str, str]  # component_name -> dim_name

    def __post_init__(self) -> None:
        """Validate parameters and initialize state."""
        # Type hints already enforce Prior, but add runtime check for clarity
        if not isinstance(self.r2, Prior):
            raise TypeError(
                f"r2 must be a Prior instance, got {type(self.r2).__name__}. "
                f"Example: Prior('Beta', mu=0.8, sigma=0.4)"
            )

        if not isinstance(self.total_sigma, Prior):
            raise TypeError(
                f"total_sigma must be a Prior instance, got {type(self.total_sigma).__name__}. "
                f"Example: Prior('LogNormal', mu=0, sigma=1)"
            )

        # total_sigma must be scalar (no dims) per the R2D2M2 paper
        if self.total_sigma.dims is not None:
            raise ValueError(
                f"total_sigma must be a scalar Prior (no dims), got dims={self.total_sigma.dims}. "
                f"The R2D2M2 paper defines total_sigma as a scalar hyperparameter. "
                f"If you need hierarchical total_sigma, please open an issue."
            )

        # Validate dims values are strings
        for comp_name, dim_name in self.dims.items():
            if not isinstance(dim_name, str):
                raise TypeError(
                    f"dim value for '{comp_name}' must be a string, got {type(dim_name).__name__}. "
                    f"Example: dims={{'control': 'control'}}"
                )

        self._splits: dict[str, pt.TensorVariable] = {}
        self._error_sigma: pt.TensorVariable | None = None
        self._built: bool = False

    def create_variable(self, name: str) -> pt.TensorVariable:
        """Create all decomposition variables ONCE inside pm.Model.

        Parameters
        ----------
        name : str
            Prefix for variable names.

        Returns
        -------
        pt.TensorVariable
            The error sigma variable.
        """
        if self._built:
            return self._error_sigma

        model = pm.modelcontext(None)

        # Create hyperpriors
        r2 = self.r2.create_variable(f"{name}_r2")
        total_sigma = self.total_sigma.create_variable(f"{name}_total_sigma")

        # Variance split
        model_sigma = r2**0.5 * total_sigma
        self._error_sigma = (1 - r2) ** 0.5 * total_sigma

        # Look up dim sizes from model coords
        dim_sizes = {}
        for comp, dim in self.dims.items():
            dim_sizes[comp] = int(model.dim_lengths[dim].eval())
        K = sum(dim_sizes.values())

        # Dirichlet with named dimension
        model.add_coord(f"{name}_component", list(self.dims.keys()))
        split = pm.Dirichlet(f"{name}_split", np.ones(K), dims=f"{name}_component")

        # Slice Dirichlet into per-component groups
        # Store as xtensor for compatibility with pmd.Data
        idx = 0
        for comp_name, dim_name in self.dims.items():
            dim_size = dim_sizes[comp_name]
            comp_split = split[idx : idx + dim_size]
            # Convert to xtensor for use with pmd.Data
            self._splits[comp_name] = ptx.as_xtensor(
                model_sigma * comp_split**0.5,
                dims=(dim_name,),
            )
            idx += dim_size

        self._built = True
        return self._error_sigma

    def split(self, component_name: str) -> R2D2Split:
        """Get lazy reference to a component's split variable.

        Parameters
        ----------
        component_name : str
            Name of the component (must be a key in dims).

        Returns
        -------
        R2D2Split
            Lazy reference to the split variable.

        Raises
        ------
        ValueError
            If component_name is not a key in dims.
        """
        if component_name not in self.dims:
            available = list(self.dims.keys())
            raise ValueError(
                f"Component '{component_name}' not found in dims. "
                f"Available components: {available}. "
                f"Use r2d2.split(component_name) where component_name is a key in dims."
            )
        return R2D2Split(self, component_name)

    @property
    def error_sigma(self) -> R2D2Sigma:
        """Get lazy reference to error sigma."""
        return R2D2Sigma(self)

    @property
    def splits(self) -> dict[str, pt.TensorVariable]:
        """Access created splits (after create_variable)."""
        return self._splits.copy()

    @property
    def built(self) -> bool:
        """Whether create_variable has been called."""
        return self._built

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "r2": self.r2.to_dict(),
            "total_sigma": self.total_sigma.to_dict(),
            "dims": self.dims,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "R2D2Decomposition":
        """Deserialize from dictionary."""

        def _deserialize_prior(val: Any) -> Prior:
            if isinstance(val, dict):
                return deserialize(val)
            return val

        return cls(
            r2=_deserialize_prior(data["r2"]),
            total_sigma=_deserialize_prior(data["total_sigma"]),
            dims=data["dims"],
        )
