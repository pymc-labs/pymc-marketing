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
"""
Specialized priors that behave like the Prior class.

The Prior class has certain design constraints that prevent it from
covering all cases. So this module contains a collection of
priors that do not inherit from the Prior class but have many
of the same methods.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytensor.tensor as pt
import xarray as xr
from pymc_extras.deserialize import deserialize, register_deserialization
from pymc_extras.prior import (
    Prior,
    VariableFactory,
    _param_value_with_dims,
    sample_prior,
)
from pytensor.tensor import TensorVariable, as_tensor
from pytensor.xtensor.type import XTensorVariable, as_xtensor


class SpecialPrior(ABC):
    """A base class for specialized priors."""

    def __init__(self, dims: tuple | None = None, centered: bool = True, **parameters):
        self.dims = dims
        self.centered = centered
        self.parameters = parameters
        self._checks()

    def __eq__(self, other):
        """Check equality based on class, dims, centered, and parameters."""
        if not isinstance(other, self.__class__):
            return False
        if self.dims != other.dims or self.centered != other.centered:
            return False

        # Compare parameters, handling numpy arrays
        if set(self.parameters.keys()) != set(other.parameters.keys()):
            return False

        for key in self.parameters:
            val1 = self.parameters[key]
            val2 = other.parameters[key]

            # Handle numpy arrays
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if not np.array_equal(val1, val2):
                    return False
            elif isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
                return False  # One is array, other isn't
            elif val1 != val2:
                return False

        return True

    def __hash__(self):
        """Compute hash based on class, dims, centered, and parameters."""
        # Convert parameters to a hashable tuple
        param_items = []
        for key in sorted(self.parameters.keys()):
            value = self.parameters[key]
            # Convert numpy arrays to tuples for hashing
            if isinstance(value, np.ndarray):
                value = tuple(value.flat)
            # Convert lists to tuples
            elif isinstance(value, list):
                value = tuple(value)
            # For unhashable types, use their string representation
            try:
                hash(value)
            except TypeError:
                value = str(value)
            param_items.append((key, value))

        return hash(
            (self.__class__.__name__, self.dims, self.centered, tuple(param_items))
        )

    @abstractmethod
    def _checks(self) -> None:  # pragma: no cover
        """Check that the parameters are correct."""
        pass

    @abstractmethod
    def create_variable(self, name: str) -> TensorVariable:  # pragma: no cover
        """Create a variable from the prior distribution."""
        pass

    def _create_parameter(self, param, value, name, xdist: bool = False):
        if not hasattr(value, "create_variable"):
            if xdist:
                return _param_value_with_dims(param, value, dims=self.dims)
            else:
                return value

        child_name = f"{name}_{param}"
        return value.create_variable(child_name, xdist=xdist)

    def to_dict(self):
        """Convert the SpecialPrior to a dictionary."""
        class_name = self.__class__.__name__
        data = {
            "special_prior": class_name,
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
    def from_dict(cls, data) -> "SpecialPrior":
        """Create a SpecialPrior prior from a dictionary."""
        if not isinstance(data, dict):
            msg = (
                "Must be a dictionary representation of a prior distribution. "
                f"Not of type: {type(data)}"
            )
            raise ValueError(msg)

        # Extract special keys
        centered = data.get("centered", True)
        dims = data.get("dims")
        # Convert dims to tuple if it's a list (e.g., from YAML)
        if isinstance(dims, list):
            dims = tuple(dims)

        # Check if parameters are in kwargs or at top level
        if "kwargs" in data:
            # Parameters are in kwargs subdictionary
            kwargs = data.get("kwargs", {})
        else:
            # Parameters are at top level - extract everything except special keys
            special_keys = {"special_prior", "centered", "dims"}
            kwargs = {k: v for k, v in data.items() if k not in special_keys}

        def handle_value(value):
            if isinstance(value, dict):
                return deserialize(value)

            if isinstance(value, list):
                return np.array(value)

            return value

        kwargs = {param: handle_value(value) for param, value in kwargs.items()}

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
            xdist=True,
            **sample_prior_predictive_kwargs,
        )


class LogNormalPrior(SpecialPrior):
    r"""Lognormal prior parameterized by positive-scale mean and std.

    This prior differs from the standard ``LogNormal`` distribution,
    which takes log-scale parameters (``mu_log``, ``sigma_log``).
    Instead, it is parameterized directly in terms of the mean and
    standard deviation (``mean``, ``std``) on the positive scale, making it more intuitive
    and suitable for hierarchical modeling.

    To achieve this, the lognormal parameters are computed internally
    from the positive-domain parameters:

    .. math::

        \mu_{\text{log}} &= \ln \left( \frac{\text{mean}^2}{\sqrt{\text{mean}^2 + \text{std}^2}} \right) \\
        \sigma_{\text{log}} &= \sqrt{ \ln \left( 1 + \frac{\text{std}^2}{\text{mean}^2} \right) }

    where :math:`\text{mean} > 0` and :math:`\text{std} > 0`.

    The prior is then defined as:

    .. math::

        \phi \sim \text{LogNormal}(\mu_{\text{log}}, \sigma_{\text{log}})

    This construction ensures that the resulting random variable
    has approximately the intended mean and variance on the positive scale,
    even when :math:`\text{mean}` and :math:`\text{std}` are themselves random variables.

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
    - D. Saunders, `A positive constrained non-centered prior that sparks joy <https://daniel-saunders-phil.github.io/imagination_machine/posts/a-positive-constrained-non-centered-prior-that-sparks-joy/>`_.
    - Wikipedia, `Log-normal distribution — Definitions <https://en.wikipedia.org/wiki/Log-normal_distribution#Definitions>`_.
    """

    def __init__(self, dims: tuple | None = None, centered: bool = True, **parameters):
        # Accept aliases mu->mean and sigma->std for convenience/compatibility
        if "mean" not in parameters and "mu" in parameters:
            parameters["mean"] = parameters.pop("mu")
        if "std" not in parameters and "sigma" in parameters:
            parameters["std"] = parameters.pop("sigma")

        super().__init__(dims=dims, centered=centered, **parameters)

    def _checks(self) -> None:
        self._parameters_are_correct_set()

    def _parameters_are_correct_set(self) -> None:
        # Only allow exactly these keys after alias normalization
        if set(self.parameters.keys()) != {"mean", "std"}:
            raise ValueError("Parameters must be mean and std")

    def create_variable(
        self, name: str, xdist: bool = False
    ) -> TensorVariable | XTensorVariable:
        """Create a variable from the prior distribution."""
        if not xdist:
            raise NotImplementedError(f"{self!r} only supports xdist=True")

        parameters = {
            param: as_xtensor(self._create_parameter(param, value, name, xdist=True))
            for param, value in self.parameters.items()
        }
        mu_log = pmd.math.log(
            parameters["mean"] ** 2
            / pmd.math.sqrt(parameters["mean"] ** 2 + parameters["std"] ** 2)
        )
        sigma_log = pmd.math.sqrt(
            pmd.math.log1p(parameters["std"] ** 2 / parameters["mean"] ** 2)
        )

        if self.centered:
            log_phi = pmd.Normal(
                name + "_log", mu=mu_log, sigma=sigma_log, dims=self.dims
            )

        else:
            log_phi_z = pmd.Normal(
                name + "_log" + "_offset", mu=0, sigma=1, dims=self.dims
            )
            log_phi = mu_log + log_phi_z * sigma_log

        phi = pmd.math.exp(log_phi)
        phi = pmd.Deterministic(name, phi)

        return phi


def _is_LogNormalPrior_type(data: dict) -> bool:
    if "special_prior" in data:
        return data["special_prior"] == "LogNormalPrior"
    else:
        return False


register_deserialization(
    is_type=_is_LogNormalPrior_type,
    deserialize=LogNormalPrior.from_dict,
)


class LaplacePrior(SpecialPrior):
    """A Laplace prior parameterized by a location and a scale parameter.

    Unlike the standard Laplace distribution available through the Prior class,
    this distribution can optionally be centered or non-centered. A non-centered parameterization
    can sometimes eliminate sampling problems in hierarchical models.

    Parameters
    ----------
    mu : Prior, float, int, array-like
        The location parameter of the distribution.
    b : Prior, float, int, array-like
        The scale parameter of the distribution.
    dims : tuple[str, ...], optional
        The dimensions of the distribution, by default None.
    centered : bool, optional
        Whether to use the centered parameterization, by default True.

    References
    ----------
    - A.C. Jones, `Scale mixtures of unbounded continuous distributions <https://andrewcharlesjones.github.io/journal/scale-mixtures.html>`_.
    - Stan Documentation, `Unbounded continuous distributions <https://mc-stan.org/docs/functions-reference/unbounded_continuous_distributions.html#double-exponential-laplace-distribution>`_.
    """

    def _checks(self) -> None:
        self._parameters_are_correct_set()

    def _parameters_are_correct_set(self) -> None:
        # Only allow exactly these keys after alias normalization
        if set(self.parameters.keys()) != {"mu", "b"}:
            raise ValueError("Parameters must be mu and b")

    def create_variable(self, name: str, xdist: bool = False) -> TensorVariable:
        """Create a variable from the prior distribution."""
        if not xdist:
            raise NotImplementedError(f"{self!r} only supports xdist=True")

        parameters = {
            param: self._create_parameter(param, value, name, xdist=True)
            for param, value in self.parameters.items()
        }
        if self.centered:
            phi = pmd.Laplace(
                name, mu=parameters["mu"], b=parameters["b"], dims=self.dims
            )

        else:
            sigma = pmd.Exponential(name + "_sigma", scale=2 * parameters["b"] ** 2)
            phi = pmd.Normal(name + "_offset", mu=0, sigma=1, dims=self.dims)
            phi = pmd.Deterministic(
                name,
                phi * pmd.math.sqrt(sigma) + parameters["mu"],
            )

        return phi


def _is_LaplacePrior_type(data: dict) -> bool:
    if "special_prior" in data:
        return data["special_prior"] == "LaplacePrior"
    else:
        return False


register_deserialization(
    is_type=_is_LaplacePrior_type,
    deserialize=LaplacePrior.from_dict,
)


class MaskedPrior:
    """Create variables from a prior over only the active entries of a boolean mask.

    .. warning::
        This class is experimental and its API may change in future versions.

    Parameters
    ----------
    prior : Prior
        Base prior whose variable is defined over `prior.dims`. Internally, the
        variable is created only for the active entries given by `mask` and
        then expanded back to the full shape with zeros at inactive positions.
    mask : xarray.DataArray
        Boolean array with the same dims and shape as `prior.dims` marking active
        (True) and inactive (False) entries.
    active_dim : str, optional
        Name of the coordinate indexing the active subset. If not provided, a
        name is generated as ``"non_null_dims:<dim1>_<dim2>_..."``. If an existing
        coordinate with the same name has a different length, a suffix with the
        active length is appended.

    Examples
    --------
    Simple 1D masking.

    .. code-block:: python

        import numpy as np
        import xarray as xr
        import pymc as pm
        from pymc_extras.prior import Prior
        from pymc_marketing.special_priors import MaskedPrior

        coords = {"country": ["Venezuela", "Colombia"]}
        mask = xr.DataArray(
            [True, False],
            dims=["country"],
            coords={"country": coords["country"]},
        )
        intercept = Prior("Normal", mu=0, sigma=10, dims=("country",))
        with pm.Model(coords=coords):
            masked = MaskedPrior(intercept, mask)
            intercept_full = masked.create_variable("intercept")

    Nested parameter priors with dims remapped to the active subset.

    .. code-block:: python

        import numpy as np
        import xarray as xr
        import pymc as pm
        from pymc_extras.prior import Prior
        from pymc_marketing.special_priors import MaskedPrior

        coords = {"country": ["Venezuela", "Colombia"]}
        mask = xr.DataArray(
            [True, False],
            dims=["country"],
            coords={"country": coords["country"]},
        )
        intercept = Prior(
            "Normal",
            mu=Prior("HalfNormal", sigma=1, dims=("country",)),
            sigma=10,
            dims=("country",),
        )
        with pm.Model(coords=coords):
            masked = MaskedPrior(intercept, mask)
            intercept_full = masked.create_variable("intercept")

    All entries masked (returns deterministic zeros with original dims).

    .. code-block:: python

        import numpy as np
        import xarray as xr
        import pymc as pm
        from pymc_extras.prior import Prior
        from pymc_marketing.special_priors import MaskedPrior

        coords = {"country": ["Venezuela", "Colombia"]}
        mask = xr.DataArray(
            [False, False],
            dims=["country"],
            coords={"country": coords["country"]},
        )
        prior = Prior("Normal", mu=0, sigma=10, dims=("country",))
        with pm.Model(coords=coords):
            masked = MaskedPrior(prior, mask)
            zeros = masked.create_variable("intercept")

    Apply over a saturation function priors:

    .. code-block:: python

        from pymc_marketing.mmm import LogisticSaturation
        from pymc_marketing.special_priors import MaskedPrior

        coords = {
            "country": ["Colombia", "Venezuela"],
            "channel": ["x1", "x2", "x3", "x4"],
        }

        mask_excluded_x4_colombia = xr.DataArray(
            [[True, False, True, False], [True, True, True, True]],
            dims=["country", "channel"],
            coords=coords,
        )

        saturation = LogisticSaturation(
            priors={
                "lam": MaskedPrior(
                    Prior(
                        "Gamma",
                        mu=2,
                        sigma=0.5,
                        dims=("country", "channel"),
                    ),
                    mask=mask_excluded_x4_colombia,
                ),
                "beta": Prior(
                    "Gamma",
                    mu=3,
                    sigma=0.5,
                    dims=("country", "channel"),
                ),
            }
        )

        prior = saturation.sample_prior(coords=coords, random_seed=10)
        curve = saturation.sample_curve(prior)
        saturation.plot_curve(
            curve,
            subplot_kwargs={
                "ncols": 4,
                "figsize": (12, 18),
            },
        )

    Masked likelihood over an arbitrary subset of entries (2D example over (date, country)):

    .. code-block:: python

        import numpy as np
        import xarray as xr
        import pymc as pm
        from pymc_extras.prior import Prior
        from pymc_marketing.special_priors import MaskedPrior

        coords = {
            "date": np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[D]"),
            "country": ["Venezuela", "Colombia"],
        }

        mask = xr.DataArray(
            [[True, False], [True, False]],
            dims=["date", "country"],
            coords={"date": coords["date"], "country": coords["country"]},
        )

        intercept = Prior("Normal", mu=0, sigma=10, dims=("country",))
        likelihood = Prior(
            "Normal", sigma=Prior("HalfNormal", sigma=1), dims=("date", "country")
        )
        observed = np.random.normal(0, 1, size=(2, 2))

        with pm.Model(coords=coords):
            mu = intercept.create_variable("intercept")
            masked = MaskedPrior(likelihood, mask)
            y = masked.create_likelihood_variable("y", mu=mu, observed=observed)
    """

    def __init__(
        self, prior: Prior, mask: xr.DataArray, active_dim: str | None = None
    ) -> None:
        self.prior = prior
        self.mask = mask
        self.dims = prior.dims
        self.active_dim = active_dim or f"non_null_dims:{'_'.join(self.dims)}"
        self._validate_mask()

        warnings.warn(
            "This class is experimental and its API may change in future versions.",
            stacklevel=2,
        )

    def _validate_mask(self) -> None:
        if tuple(self.mask.dims) != tuple(self.dims):
            raise ValueError("mask dims must match prior.dims order")

    def _remap_dims(self, factory: VariableFactory) -> VariableFactory:
        # Depth-first remap of any nested VariableFactory with dims == parent dims
        # This keeps internal subset checks (_param_dims_work) satisfied.
        if hasattr(factory, "parameters"):
            # Recurse on child parameters first
            for key, value in list(factory.parameters.items()):
                if hasattr(value, "create_variable") and hasattr(value, "dims"):
                    factory.parameters[key] = self._remap_dims(value)  # type: ignore[arg-type]

        # Now remap this object's dims if they exactly match the masked dims
        if hasattr(factory, "dims"):
            dims = factory.dims or ()
            if isinstance(dims, str):
                dims = (dims,)
            if tuple(dims) == tuple(self.dims):
                factory.dims = (self.active_dim,)

        return factory

    def create_variable(
        self, name: str, xdist: bool = False
    ) -> TensorVariable | XTensorVariable:
        """Create a deterministic variable with full dims using the active subset.

        Creates an underlying variable over the active entries only and expands
        it back to the full masked shape, filling inactive entries with zeros.

        Parameters
        ----------
        name : str
            Base name for the created variables.

        Returns
        -------
        TensorVariable or XTensorVariable
            Deterministic variable with the original dims, zeros on inactive entries.
        """
        model = pm.modelcontext(None)
        flat_mask = self.mask.values.astype(bool).ravel()
        n_active = int(flat_mask.sum())

        if n_active == 0:
            return pm.Deterministic(name, pt.zeros(self.mask.shape), dims=self.dims)

        # Ensure the coord exists and has the right length
        if (
            self.active_dim in model.coords
            and len(model.coords[self.active_dim]) != n_active
        ):
            self.active_dim = f"{self.active_dim}__{n_active}"
        model.add_coords({self.active_dim: np.arange(n_active)})

        # Make a deep copy and remap dims depth-first before creating the RV
        reduced = self._remap_dims(self.prior.deepcopy())

        active_rv = reduced.create_variable(
            f"{name}_active", xdist=xdist
        )  # shape: (active_dim,)
        flat_zeros = pt.zeros((self.mask.size,), dtype=active_rv.dtype)
        flat_full = flat_zeros[flat_mask].set(
            as_tensor(active_rv, allow_xtensor_conversion=True)
        )
        full = flat_full.reshape(self.mask.shape)
        if xdist:
            return pmd.Deterministic(name, as_xtensor(full, dims=self.dims))
        else:
            return pm.Deterministic(name, full, dims=self.dims)

    def to_dict(self) -> dict[str, Any]:
        """Serialize MaskedPrior to a JSON-serializable dictionary.

        Returns
        -------
        dict
            Dictionary containing the prior, mask, and active_dim.
        """
        # Store mask as a plain nested list of bools to avoid datetime coords serialization
        mask_list = (
            self.mask.values.astype(bool).tolist()
            if hasattr(self.mask, "values")
            else np.asarray(self.mask, dtype=bool).tolist()
        )
        return {
            "class": "MaskedPrior",
            "data": {
                "prior": self.prior.to_dict()
                if hasattr(self.prior, "to_dict")
                else None,
                "mask": mask_list,
                "mask_dims": list(self.dims),
                "active_dim": self.active_dim,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaskedPrior":
        """Deserialize MaskedPrior from dictionary created by ``to_dict``.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict`.

        Returns
        -------
        MaskedPrior
            Reconstructed instance.
        """
        payload = data["data"] if "data" in data else data
        prior = (
            deserialize(payload["prior"])
            if isinstance(payload.get("prior"), dict)
            else payload.get("prior")
        )
        mask_vals = payload.get("mask")
        # Fallback to provided dims or infer from prior if available
        mask_dims = payload.get("mask_dims") or (getattr(prior, "dims", None) or ())
        mask_da = xr.DataArray(np.asarray(mask_vals, dtype=bool), dims=tuple(mask_dims))
        active_dim = payload.get("active_dim")
        return cls(prior=prior, mask=mask_da, active_dim=active_dim)

    def create_likelihood_variable(
        self,
        name: str,
        *,
        mu: pt.TensorLike,
        observed: pt.TensorLike,
        xdist: bool = False,
    ) -> TensorVariable:
        """Create an observed variable over the active subset and expand to full dims.

        Parameters
        ----------
        name : str
            Base name for the created variables.
        mu : XTensorLike
            Mean/location parameter broadcastable to the masked shape.
        observed : XTensorLike
            Observations broadcastable to the masked shape.

        Returns
        -------
        XTensorVariable
            Deterministic variable over the full dims with observed RV on active entries.
        """
        if xdist:
            raise NotImplementedError(f"{self!r} only supports xdist=False")

        model = pm.modelcontext(None)
        flat_mask = self.mask.values.ravel().astype(bool)
        n_active = int(flat_mask.sum())

        if n_active == 0:
            return pm.Deterministic(name, pt.zeros(self.mask.shape), dims=self.dims)

        # Ensure the coord exists and has the right length
        if (
            self.active_dim in model.coords
            and len(model.coords[self.active_dim]) != n_active
        ):
            self.active_dim = f"{self.active_dim}__{n_active}"
        model.add_coords({self.active_dim: np.arange(n_active)})

        # Remap dims on a deep copy so nested parameter priors match the active subset
        reduced = self._remap_dims(self.prior.deepcopy())

        # Broadcast mu/observed to full mask shape via arithmetic broadcasting, then select active entries
        mu_tensor = pt.as_tensor_variable(mu)
        mu_full = mu_tensor + pt.zeros(self.mask.shape, dtype=mu_tensor.dtype)
        mu_active = mu_full.reshape((self.mask.size,))[flat_mask]

        obs = observed.values if hasattr(observed, "values") else observed
        obs_tensor = pt.as_tensor_variable(obs)
        obs_full = obs_tensor + pt.zeros(self.mask.shape, dtype=obs_tensor.dtype)
        obs_active = obs_full.reshape((self.mask.size,))[flat_mask]

        # Create the masked observed RV over the active subset
        active_name = f"{name}_active"
        active_rv = reduced.create_likelihood_variable(
            active_name, mu=mu_active, observed=obs_active
        )

        # Expand back to full shape for user-friendly access
        flat_full = pt.zeros((self.mask.size,), dtype=active_rv.dtype)
        full = flat_full[flat_mask].set(active_rv).reshape(self.mask.shape)
        return pm.Deterministic(name, full, dims=self.dims)


def _is_masked_prior_type(data: dict) -> bool:
    return data.keys() == {"class", "data"} and data.get("class") == "MaskedPrior"


register_deserialization(
    is_type=_is_masked_prior_type, deserialize=MaskedPrior.from_dict
)


def _is_special_prior_type(data: dict) -> bool:
    """Check if data represents a SpecialPrior subclass."""
    return isinstance(data, dict) and "special_prior" in data


def _get_all_special_prior_subclasses(
    base_class: type[SpecialPrior],
) -> dict[str, type[SpecialPrior]]:
    """Recursively get all subclasses of a base class.

    Returns a dict mapping class name to class object.
    """
    subclasses: dict[str, type[SpecialPrior]] = {}
    for subclass in base_class.__subclasses__():
        subclasses[subclass.__name__] = subclass
        # Recursively get subclasses of subclasses
        subclasses.update(_get_all_special_prior_subclasses(subclass))
    return subclasses


def _deserialize_special_prior(data: dict) -> SpecialPrior:
    """Deserialize any SpecialPrior subclass by looking up the class dynamically.

    This function automatically discovers all SpecialPrior subclasses using __subclasses__(),
    so new SpecialPrior subclasses don't need explicit registration.
    """
    class_name = data.get("special_prior")
    if not isinstance(class_name, str):
        raise ValueError(
            f"Expected 'special_prior' to be a string, got {type(class_name)}"
        )

    # Get all SpecialPrior subclasses recursively
    special_prior_classes = _get_all_special_prior_subclasses(SpecialPrior)  # type: ignore[type-abstract]

    cls = special_prior_classes.get(class_name)
    if cls is None:
        raise ValueError(
            f"Unknown SpecialPrior class: {class_name}. "
            f"Available classes: {list(special_prior_classes.keys())}"
        )

    return cls.from_dict(data)


register_deserialization(
    is_type=_is_special_prior_type, deserialize=_deserialize_special_prior
)
