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

import warnings
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc_extras.deserialize import deserialize, register_deserialization
from pymc_extras.prior import Prior, VariableFactory, create_dim_handler, sample_prior
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
        # Accept aliases mu->mean and sigma->std for convenience/compatibility
        if "mean" not in parameters and "mu" in parameters:
            parameters["mean"] = parameters.pop("mu")
        if "std" not in parameters and "sigma" in parameters:
            parameters["std"] = parameters.pop("sigma")

        self.parameters = parameters
        self.dims = dims
        self.centered = centered

        self._checks()

    def _checks(self) -> None:
        self._parameters_are_correct_set()

    def _parameters_are_correct_set(self) -> None:
        # Only allow exactly these keys after alias normalization
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
            dims = factory.dims
            if isinstance(dims, str):
                dims = (dims,)
            if tuple(dims) == tuple(self.dims):
                factory.dims = (self.active_dim,)

        return factory

    def create_variable(self, name: str) -> TensorVariable:
        """Create a deterministic variable with full dims using the active subset.

        Creates an underlying variable over the active entries only and expands
        it back to the full masked shape, filling inactive entries with zeros.

        Parameters
        ----------
        name : str
            Base name for the created variables.

        Returns
        -------
        pt.TensorVariable
            Deterministic variable with the original dims, zeros on inactive entries.
        """
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

        # Make a deep copy and remap dims depth-first before creating the RV
        reduced = self._remap_dims(self.prior.deepcopy())

        active_rv = reduced.create_variable(f"{name}_active")  # shape: (active_dim,)
        flat_full = pt.zeros((self.mask.size,), dtype=active_rv.dtype)
        full = flat_full[flat_mask].set(active_rv).reshape(self.mask.shape)
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
        self, name: str, *, mu: pt.TensorLike, observed: pt.TensorLike
    ) -> TensorVariable:
        """Create an observed variable over the active subset and expand to full dims.

        Parameters
        ----------
        name : str
            Base name for the created variables.
        mu : pt.TensorLike
            Mean/location parameter broadcastable to the masked shape.
        observed : pt.TensorLike
            Observations broadcastable to the masked shape.

        Returns
        -------
        pt.TensorVariable
            Deterministic variable over the full dims with observed RV on active entries.
        """
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
