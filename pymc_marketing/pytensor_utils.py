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

"""PyTensor utility functions."""

import arviz as az
import pytensor.tensor as pt
from arviz import InferenceData
from pymc import Model
from pytensor import clone_replace
from pytensor.graph import rewrite_graph, vectorize_graph
from pytensor.graph.basic import ancestors

try:
    from pymc.pytensorf import rvs_in_graph
except ImportError:
    from pymc.logprob.utils import rvs_in_graph

import warnings

import numpy as np
import pymc as pm
import xarray as xr
from pymc_extras.deserialize import deserialize, register_deserialization
from pymc_extras.prior import Prior


def extract_response_distribution(
    pymc_model: Model,
    idata: InferenceData,
    response_variable: str,
) -> pt.TensorVariable:
    """Extract the response distribution graph, conditioned on posterior parameters.

    Parameters
    ----------
    pymc_model : Model
        The PyMC model to extract the response distribution from.
    idata : InferenceData
        The inference data containing posterior samples.
    response_variable : str
        The name of the response variable to extract.

    Returns
    -------
    pt.TensorVariable
        The response distribution graph.

    Example
    -------
    `extract_response_distribution(model, idata, "channel_contribution")`
    returns a graph that computes `"channel_contribution"` as a function of both
    the newly introduced budgets and the posterior of model parameters.
    """
    # Convert InferenceData to a sample-major xarray
    posterior = az.extract(idata).transpose("sample", ...)  # type: ignore

    # The PyMC variable to extract
    response_var = pymc_model[response_variable]

    # Identify which free RVs are needed to compute `response_var`
    free_rvs = set(pymc_model.free_RVs)
    needed_rvs = [
        rv for rv in ancestors([response_var], blockers=free_rvs) if rv in free_rvs
    ]
    placeholder_replace_dict = {
        pymc_model[rv.name]: pt.tensor(
            name=rv.name, shape=rv.type.shape, dtype=rv.dtype
        )
        for rv in needed_rvs
    }

    [response_var] = clone_replace(
        [response_var],
        replace=placeholder_replace_dict,
    )

    if rvs_in_graph([response_var]):
        raise RuntimeError("RVs found in the extracted graph, this is likely a bug")

    # Cleanup graph
    response_var = rewrite_graph(response_var, include=("canonicalize", "ShapeOpt"))

    # Replace placeholders with actual posterior samples
    replace_dict = {}
    for placeholder in placeholder_replace_dict.values():
        replace_dict[placeholder] = pt.constant(
            posterior[placeholder.name].astype(placeholder.dtype),
            name=placeholder.name,
        )

    # Vectorize across samples
    response_distribution = vectorize_graph(response_var, replace=replace_dict)

    # Final cleanup
    response_distribution = rewrite_graph(
        response_distribution,
        include=(
            "useless",
            "local_eager_useless_unbatched_blockwise",
            "local_useless_unbatched_blockwise",
        ),
    )

    return response_distribution


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
        from pymc_marketing.pytensor_utils import MaskedPrior

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
        from pymc_marketing.pytensor_utils import MaskedPrior

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
        from pymc_marketing.pytensor_utils import MaskedPrior

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
        from pymc_marketing.pytensor_utils import MaskedPrior

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
        from pymc_marketing.pytensor_utils import MaskedPrior

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

    def __init__(self, prior: Prior, mask: xr.DataArray, active_dim: str | None = None):
        self.prior = prior
        self.mask = mask
        self.dims = prior.dims
        self.active_dim = active_dim or f"non_null_dims:{'_'.join(self.dims)}"
        self._validate_mask()

        warnings.warn(
            "This class is experimental and its API may change in future versions.",
            stacklevel=2,
        )

    def _validate_mask(self):
        if tuple(self.mask.dims) != tuple(self.dims):
            raise ValueError("mask dims must match prior.dims order")

    def _remap_dims(self, factory):
        # Depth-first remap of any nested VariableFactory with dims == parent dims
        # This keeps internal subset checks (_param_dims_work) satisfied.
        if hasattr(factory, "parameters"):
            # Recurse on child parameters first
            for key, value in list(factory.parameters.items()):
                if hasattr(value, "create_variable") and hasattr(value, "dims"):
                    factory.parameters[key] = self._remap_dims(value)

        # Now remap this object's dims if they exactly match the masked dims
        if hasattr(factory, "dims"):
            dims = factory.dims
            if isinstance(dims, str):
                dims = (dims,)
            if tuple(dims) == tuple(self.dims):
                factory.dims = (self.active_dim,)

        return factory

    def create_variable(self, name: str):
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

    def to_dict(self) -> dict:
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
    def from_dict(cls, data: dict) -> "MaskedPrior":
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
    ):
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
