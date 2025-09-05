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

from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
from arviz import InferenceData
from pymc import Model
from pymc.model.fgraph import (
    ModelVar,
    extract_dims,
    fgraph_from_model,
    model_from_fgraph,
)
from pytensor import clone_replace
from pytensor.graph import FunctionGraph, rewrite_graph, vectorize_graph
from pytensor.graph.basic import ancestors

try:
    from pymc.pytensorf import rvs_in_graph
except ImportError:
    from pymc.logprob.utils import rvs_in_graph

from pymc_extras.deserialize import deserialize, register_deserialization
from pymc_extras.prior import Prior, handle_dims


def extract_response_distribution(
    pymc_model: Model,
    idata: InferenceData,
    response_variable: str,
) -> pt.TensorVariable:
    """Extract the response distribution graph conditioned on posterior parameters.

    Parameters
    ----------
    pymc_model : Model
        The PyMC model to extract the response distribution from.
    idata : InferenceData
        Inference data containing posterior samples.
    response_variable : str
        Name of the response variable to extract from the model.

    Returns
    -------
    pt.TensorVariable
        A graph that computes the requested response variable as a function of
        newly introduced inputs (e.g., budgets) and the posterior of model parameters.

    Examples
    --------
    Build a graph for a response variable, evaluated under the posterior:

    .. code-block:: python

        from pymc_marketing.pytensor_utils import extract_response_distribution

        # response_graph can be compiled with pytensor.function and evaluated by
        # providing the required inputs (e.g., new budget allocations)
        response_graph = extract_response_distribution(
            model, idata, "channel_contribution"
        )
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


class MaskedDist(Prior):
    """Create a masked deterministic from a Prior over full dims.

    The foal is to reduce the number of parameters in the model by creating a masked deterministic
    and only active the parameters that are needed.

    The active RV is created without labeled dims; its shape is inferred from its parameters and equals
    the number of active positions. The mask is treated as a NumPy boolean array at model build time. It must be
    fully specified (non-symbolic) and match the product of the coordinate lengths of the distribution dims.

    Parameters
    ----------
    distribution : Prior
        A Prior object whose dims define the full grid structure.
    mask : array-like
        Boolean mask with same total number of positions as the distribution dims.
        Can be same shape as dims or 1D flattened.
    active_dim_name : str, default "active"
        Name for the active dimension in the reduced distribution.

    Examples
    --------
    Create a masked normal distribution over a 2D grid:

    .. code-block:: python

        import numpy as np
        import pymc as pm
        from pymc_extras.prior import Prior
        from pymc_marketing.pytensor_utils import MaskedDist

        # Define a 3x4 grid
        coords = {"row": [0, 1, 2], "col": [0, 1, 2, 3]}

        # Create mask - only activate positions (0,0), (1,2), (2,3)
        mask = np.array(
            [
                [True, False, False, False],
                [False, False, True, False],
                [False, False, False, True],
            ]
        )

        # Define prior over full grid
        prior = Prior("Normal", mu=0, sigma=1, dims=("row", "col"))

        with pm.Model(coords=coords):
            # Create masked distribution
            masked_dist = MaskedDist(prior, mask)
            coeff = masked_dist.create_variable("coeff")
    """

    def __init__(
        self, distribution: Prior, mask: Any, active_dim_name: str = "active"
    ) -> None:
        # Initialize as a Prior by copying fields from the wrapped prior
        super().__init__(
            distribution=distribution.distribution,
            dims=distribution.dims,
            centered=distribution.centered,
            transform=distribution.transform,
            **distribution.parameters,
        )
        self.mask = mask
        self.active_dim_name = active_dim_name

    # Inherit dims property and behavior from Prior

    def _aligned_param_full(
        self,
        *,
        model: pm.Model,
        name: str,
        param_name: str,
        value: Any,
        full_dims: tuple[str, ...],
        full_sizes: tuple[int, ...],
    ) -> Any:
        """Create the full-grid parameter aligned to ``full_dims``.

        Parameters
        ----------
        model : pm.Model
            Active PyMC model context.
        name : str
            Base name used when creating variables from nested ``Prior`` objects.
        param_name : str
            Name of the parameter in the wrapped ``Prior``.
        value : Any
            Parameter value. Can be a ``Prior``-like object, NumPy array, PyTensor
            variable, scalar, or other broadcastable values.
        full_dims : tuple of str
            Target named dimensions (from the full grid).
        full_sizes : tuple of int
            Sizes corresponding to ``full_dims``.

        Returns
        -------
        Any
            Aligned parameter on the full grid. If ``value`` is a ``Prior`` it returns
            its created variable aligned to ``full_dims``; arrays/tensors are reshaped
            or broadcast when feasible; scalars are returned unchanged.
        """
        if hasattr(value, "create_variable"):
            var_full = value.create_variable(f"{name}_{param_name}")
            return handle_dims(var_full, value.dims, full_dims)

        if isinstance(value, np.ndarray | pt.TensorVariable):
            x = pt.as_tensor_variable(value)
            if x.ndim == 0:
                return x

            # Try to reshape or broadcast to the full grid
            if x.ndim != len(full_sizes):
                total = int(np.prod(full_sizes))
                if x.ndim == 1 and int(x.shape[0]) == total:
                    return pt.reshape(x, full_sizes)
                raise ValueError(
                    "Parameter array must be scalar, 1D flat over full grid, or have same ndim as distribution dims",
                )

            cur_shape = tuple(x.shape)
            if cur_shape == full_sizes:
                return x

            # Broadcast dimensions that are 1 to the target size
            can_broadcast = all(
                cs in (1, s) for cs, s in zip(cur_shape, full_sizes, strict=False)
            )
            if can_broadcast:
                return pt.broadcast_to(x, full_sizes)

            raise ValueError(
                "Parameter array shape incompatible with distribution dims"
            )

        return value

    def create_variable(self, name: str) -> pt.TensorVariable:
        """Create the masked deterministic and the underlying active RV.

        Parameters
        ----------
        name : str
            Base name for the created variables. The active RV will be named
            ``"{name}_dist"`` and the returned deterministic will be named ``"{name}"``.

        Returns
        -------
        pt.TensorVariable
            A deterministic tensor with the same shape as the full distribution grid,
            with active positions filled by the learned parameters and zeros elsewhere.

        Examples
        --------
        Create a masked distribution over a 2D grid and materialize the deterministic:

        .. code-block:: python

            prior = Prior("Normal", mu=0, sigma=1, dims=("country", "channel"))
            mask = np.array([[True, False, False], [False, False, True]])
            masked = MaskedDist(prior, mask)
            with pm.Model(
                coords={"country": ["A", "B"], "channel": ["C1", "C2", "C3"]}
            ):
                out = masked.create_variable("coeff")
        """
        model = pm.modelcontext(None)
        full_dims: tuple[str, ...] = (
            (self.dims if isinstance(self.dims, tuple) else (self.dims,))
            if self.dims
            else ()
        )

        if not full_dims:
            raise ValueError(
                "MaskedDist requires the wrapped Prior to define named dims"
            )

        # Prepare mask as a concrete NumPy boolean array
        if isinstance(self.mask, pt.TensorVariable):
            mask_np = np.asarray(self.mask.eval()).astype(bool)
        else:
            mask_np = np.asarray(self.mask).astype(bool)
        # Require coords for all dims to be present; do not auto-create
        try:
            full_sizes = tuple(len(model.coords[dim]) for dim in full_dims)
        except KeyError as err:
            raise KeyError(
                f"Missing coords for dim {err.args[0]!r} in the current model. "
                "Pass coords when building the model or when calling sample_prior."
            ) from err

        if mask_np.ndim == 1:
            total_size = int(np.prod(full_sizes))
            if mask_np.size != total_size:
                raise ValueError(
                    "1D mask length must equal the product of coordinates across dims",
                )
            mask_np = mask_np.reshape(full_sizes)
        elif mask_np.shape != full_sizes:
            raise ValueError(
                "Mask must have the same shape as the distribution dims or be a 1D vector of matching length",
            )

        n_active = int(mask_np.sum())

        # Build active parameters by indexing the full-grid parameters
        params_active: dict[str, Any] = {}
        for param_name, value in self.parameters.items():
            full_param = self._aligned_param_full(
                model=model,
                name=name,
                param_name=param_name,
                value=value,
                full_dims=full_dims,
                full_sizes=full_sizes,
            )

            if isinstance(full_param, int | float):
                params_active[param_name] = full_param
            elif isinstance(full_param, pt.TensorVariable):
                if full_param.ndim == 0:
                    params_active[param_name] = full_param
                else:
                    params_active[param_name] = full_param[mask_np]
            elif isinstance(full_param, np.ndarray):
                if full_param.ndim == 0:
                    params_active[param_name] = float(full_param)
                else:
                    params_active[param_name] = full_param[mask_np]
            else:
                params_active[param_name] = full_param

        # Ensure resulting active distribution has vector shape (n_active,)
        has_vector_param = False
        for val in params_active.values():
            if isinstance(val, pt.TensorVariable):
                if val.ndim >= 1:
                    has_vector_param = True
                    break
            elif isinstance(val, np.ndarray):
                if val.ndim >= 1:
                    has_vector_param = True
                    break

        if not has_vector_param:
            # Broadcast the first parameter to length n_active
            for key in self.parameters.keys():
                val = params_active[key]
                if isinstance(val, int | float):
                    params_active[key] = np.full((n_active,), val)
                    break
                if isinstance(val, np.ndarray) and val.ndim == 0:
                    params_active[key] = np.full((n_active,), float(val))
                    break
                if isinstance(val, pt.TensorVariable) and val.ndim == 0:
                    params_active[key] = pt.broadcast_to(val, (n_active,))
                    break

        # Ensure an active coordinate exists (and matches length)
        active_dim = self.active_dim_name
        if active_dim in model.coords and len(model.coords[active_dim]) != n_active:
            active_dim = f"{active_dim}_{name}"
        if active_dim not in model.coords:
            # Create a simple integer coordinate
            model.add_coord(active_dim, np.arange(n_active))

        # Create the active RV with labeled active dim
        active_prior = Prior(
            self.distribution,
            dims=(active_dim,),
            centered=self.centered,
            transform=self.transform,
            **params_active,
        )
        active_rv = active_prior.create_variable(f"{name}_dist")

        # Scatter active RV back into a full tensor and wrap as Deterministic
        zeros_full = pt.zeros(full_sizes, dtype=active_rv.dtype)
        filled = zeros_full[mask_np].set(active_rv)

        return pm.Deterministic(name, filled, dims=full_dims)

    def create_likelihood_variable(
        self,
        name: str,
        *,
        mu: pt.TensorLike,
        observed: Any,
        **kwargs: Any,
    ):
        """Create a masked likelihood by applying the mask over the likelihood dims.

        Parameters
        ----------
        name : str
            Name of the likelihood variable.
        mu : pt.TensorLike
            Mean/location parameter of the likelihood over the full grid dims.
        observed : array-like or TensorLike
            Observations over the full grid dims.
        **kwargs : Any
            Additional keyword arguments forwarded to the underlying likelihood
            creation (e.g., transform, tags, etc.).

        Notes
        -----
        This method selects the active positions defined by ``mask`` across the
        likelihood dims and creates the likelihood only for those positions under
        a new 1D active dimension. This effectively excludes masked-out positions
        from the log-likelihood.
        """
        model = pm.modelcontext(None)

        full_dims: tuple[str, ...] = (
            (self.dims if isinstance(self.dims, tuple) else (self.dims,))
            if self.dims
            else ()
        )
        if not full_dims:
            raise ValueError(
                "MaskedDist requires the wrapped Prior to define named dims"
            )

        # Prepare mask as a concrete NumPy boolean array
        if isinstance(self.mask, pt.TensorVariable):
            mask_np = np.asarray(self.mask.eval()).astype(bool)
        else:
            mask_np = np.asarray(self.mask).astype(bool)

        # Validate dims exist in model coords
        try:
            full_sizes = tuple(len(model.coords[dim]) for dim in full_dims)
        except KeyError as err:
            raise KeyError(
                f"Missing coords for dim {err.args[0]!r} in the current model. "
                "Pass coords when building the model or when calling sample_prior."
            ) from err

        if mask_np.ndim == 1:
            total_size = int(np.prod(full_sizes))
            if mask_np.size != total_size:
                raise ValueError(
                    "1D mask length must equal the product of coordinates across dims",
                )
            mask_np = mask_np.reshape(full_sizes)
        elif mask_np.shape != full_sizes:
            raise ValueError(
                "Mask must have the same shape as the distribution dims or be a 1D vector of matching length",
            )

        n_active = int(mask_np.sum())

        # Build active parameters by indexing the full-grid parameters
        params_active: dict[str, Any] = {}
        for param_name, value in self.parameters.items():
            full_param = self._aligned_param_full(
                model=model,
                name=name,
                param_name=param_name,
                value=value,
                full_dims=full_dims,
                full_sizes=full_sizes,
            )

            if isinstance(full_param, int | float):
                params_active[param_name] = full_param
            elif isinstance(full_param, pt.TensorVariable):
                if full_param.ndim == 0:
                    params_active[param_name] = full_param
                else:
                    params_active[param_name] = full_param[mask_np]
            elif isinstance(full_param, np.ndarray):
                if full_param.ndim == 0:
                    params_active[param_name] = float(full_param)
                else:
                    params_active[param_name] = full_param[mask_np]
            else:
                params_active[param_name] = full_param

        # Ensure active coordinate exists
        active_dim = self.active_dim_name
        if active_dim in model.coords and len(model.coords[active_dim]) != n_active:
            active_dim = f"{active_dim}_{name}"
        if active_dim not in model.coords:
            model.add_coord(active_dim, np.arange(n_active))

        # Build masked mu and observed
        if isinstance(mu, pt.TensorVariable | np.ndarray):
            mu_active = mu[mask_np]
        else:
            mu_active = mu

        if isinstance(observed, pt.TensorVariable | np.ndarray):
            observed_active = observed[mask_np]
        else:
            observed_active = observed

        # Create active likelihood prior and variable
        active_prior = Prior(
            self.distribution,
            dims=(active_dim,),
            centered=self.centered,
            transform=self.transform,
            **params_active,
        )

        return active_prior.create_likelihood_variable(
            name=name,
            mu=mu_active,
            observed=observed_active,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the masked distribution.

        Returns
        -------
        dict
            A dictionary with the wrapped distribution config and boolean mask.
        """
        base = super().to_dict()
        if isinstance(self.mask, pt.TensorVariable):
            mask_val = np.asarray(self.mask.eval()).astype(bool).tolist()
        else:
            mask_val = np.asarray(self.mask).astype(bool).tolist()
        return {
            "class": "MaskedDist",
            "data": {
                "dist": base,
                "active_dim_name": self.active_dim_name,
                "mask": mask_val,
            },
        }


def _is_masked_dist(data: dict) -> bool:
    """Check if the data is a wrapped MaskedDist dictionary."""
    return data.keys() == {"class", "data"} and data["class"] == "MaskedDist"


def _masked_dist_from_dict(data: dict) -> MaskedDist:  # type: ignore
    """Deserialize a wrapped MaskedDist dictionary.

    Parameters
    ----------
    data : dict
        Wrapped schema: {"class": "MaskedDist", "data": {"dist": prior_dict,
        "mask": list[bool] | list[list[bool]], "active_dim_name": str}}

    Returns
    -------
    MaskedDist
        Reconstructed masked distribution instance.
    """
    payload = data["data"]
    base = deserialize(payload["dist"])  # type: ignore
    mask = np.array(payload["mask"]).astype(bool)
    active_dim_name = payload.get("active_dim_name", "active")
    return MaskedDist(distribution=base, mask=mask, active_dim_name=active_dim_name)


register_deserialization(is_type=_is_masked_dist, deserialize=_masked_dist_from_dict)


def _prefix_model(f2, prefix: str, exclude_vars: set | None = None):
    """Prefix variable and dimension names in a FunctionGraph.

    Variables listed in ``exclude_vars`` (e.g., a shared variable like ``channel_data``)
    are kept unprefixed, and their dims/coords are also preserved without prefix.
    """
    if exclude_vars is None:
        exclude_vars = set()

    # First, collect dimensions that belong to excluded variables
    exclude_dims = set()
    for v in f2.outputs:
        if v.name in exclude_vars:
            v_dims = extract_dims(v)
            for dim in v_dims:
                exclude_dims.add(dim.data)

    dims = set()
    for v in f2.outputs:
        # Only prefix if not in exclude_vars
        if v.name not in exclude_vars:
            new_name = f"{prefix}_{v.name}"
            v.name = new_name
            if isinstance(v.owner.op, ModelVar):
                rv = v.owner.inputs[0]
                rv.name = new_name
        dims.update(extract_dims(v))

    # Don't rename dimensions that belong to excluded variables
    dims_rename = {
        dim: pytensor.as_symbolic(f"{prefix}_{dim.data}")
        for dim in dims
        if dim.data not in exclude_dims
    }
    if dims_rename:
        f2.replace_all(tuple(dims_rename.items()))

    # Don't prefix coordinates for excluded dimensions
    new_coords = {}
    for k, v in f2._coords.items():  # type: ignore[attr-defined]
        if k not in exclude_dims:
            new_coords[f"{prefix}_{k}"] = v
        else:
            new_coords[k] = v
    f2._coords = new_coords  # type: ignore[attr-defined]

    return f2


def merge_models(
    models: list[Model],
    *,
    prefixes: list[str] | None = None,
    merge_on: str | None = None,
) -> Model:
    """Merge multiple PyMC models into a single model.

    Parameters
    ----------
    models : list of pm.Model
        List of models to merge.
    prefixes : list of str or None
        List of prefixes for each model. If None, will auto-generate as 'model1', 'model2', ...
    merge_on : str or None
        Variable name to merge on (shared across all models) - this variable will NOT be prefixed.

    Returns
    -------
    pm.Model
        Merged model.
    """
    if len(models) < 2:
        raise ValueError("Need at least 2 models to merge")

    # Auto-generate prefixes if not provided
    if prefixes is None:
        prefixes = [f"model{i + 1}" for i in range(len(models))]
    elif len(prefixes) != len(models):
        raise ValueError(
            f"Number of prefixes ({len(prefixes)}) must match number of models ({len(models)})"
        )

    # Variables to exclude from prefixing
    exclude_vars = {merge_on} if merge_on else set()

    # Convert all models to FunctionGraphs and apply prefixes
    fgraphs: list[FunctionGraph] = []
    for model, prefix in zip(models, prefixes, strict=False):
        fg, _ = fgraph_from_model(model, inlined_views=True)
        if prefix is not None:
            fg = _prefix_model(fg, prefix=prefix, exclude_vars=exclude_vars)
        fgraphs.append(fg)

    # Handle merge_on logic
    if merge_on is not None:
        # Find the merge variable in the first model (unprefixed)
        first_merge_vars = [out for out in fgraphs[0].outputs if out.name == merge_on]
        if not first_merge_vars:
            raise ValueError(f"Variable '{merge_on}' not found in first model")
        first_merge_var = first_merge_vars[0]

        # Replace the merge variable in all other models with the one from the first model
        for i in range(1, len(fgraphs)):
            merge_vars = [out for out in fgraphs[i].outputs if out.name == merge_on]
            if not merge_vars:
                raise ValueError(f"Variable '{merge_on}' not found in model {i + 1}")
            fgraphs[i].replace(merge_vars[0], first_merge_var, import_missing=True)

    # Combine all outputs
    all_outputs = []
    for fg in fgraphs:
        all_outputs.extend(fg.outputs)

    # Create merged FunctionGraph
    f = FunctionGraph(outputs=all_outputs, clone=False)

    # Merge coordinates from all models
    merged_coords: dict = {}
    for fg in fgraphs:
        merged_coords.update(fg._coords)  # type: ignore[attr-defined]
    f._coords = merged_coords  # type: ignore[attr-defined]

    return model_from_fgraph(f, mutate_fgraph=True)


class ModelSamplerEstimator:
    """Estimate computational characteristics of a PyMC model using JAX/NumPyro.

    This utility measures the average evaluation time of the model's logp and gradients
    and estimates the number of integrator steps taken by NUTS during warmup + sampling.
    It then compiles the information into a single-row pandas DataFrame with helpful
    metadata to guide planning and benchmarking.

    Parameters
    ----------
    tune : int, default 1000
        Number of warmup iterations to use when estimating NUTS steps.
    draws : int, default 1000
        Number of sampling iterations to use when estimating NUTS steps.
    chains : int, default 1
        Intended number of chains (metadata only; not used in JAX runs here).
    sequential_chains : int, default 1
        Number of chains expected to run sequentially on the target environment.
        Used to scale the wall-clock time estimate.
    seed : int | None, default None
        Random seed used for the step estimation runs.

    Examples
    --------
    .. code-block:: python

        est = ModelSamplerEstimator(
            tune=1000, draws=1000, chains=4, sequential_chains=1, seed=1
        )
        df = est.run(model)
        print(df)
    """

    def __init__(
        self,
        *,
        tune: int = 1000,
        draws: int = 1000,
        chains: int = 1,
        sequential_chains: int = 1,
        seed: int | None = None,
    ) -> None:
        self.tune = int(tune)
        self.draws = int(draws)
        self.chains = int(chains)
        self.sequential_chains = int(sequential_chains)
        self.seed = seed

    def estimate_model_eval_time(self, model: Model, n: int | None = None) -> float:
        """Estimate average evaluation time (seconds) of logp+dlogp using JAX.

        Parameters
        ----------
        model : Model
            PyMC model whose logp and gradients are jitted and evaluated.
        n : int | None, optional
            Number of repeated evaluations to average over. If ``None``, a value
            is chosen to take roughly 5 seconds in total for a stable estimate.

        Returns
        -------
        float
            Average evaluation time in seconds.
        """
        from time import perf_counter_ns

        import numpy as np

        try:
            import jax
            from pymc.sampling.jax import get_jaxified_logp
        except Exception as err:  # pragma: no cover - environment specific
            raise RuntimeError(
                "JAX backend is required for ModelSamplerEstimator."
            ) from err

        initial_point = list(model.initial_point().values())
        logp_fn = get_jaxified_logp(model)
        logp_dlogp_fn = jax.jit(jax.value_and_grad(logp_fn, argnums=0))
        logp_res, grad_res = logp_dlogp_fn(initial_point)
        for val in (logp_res, *grad_res):
            if not np.isfinite(val).all():
                raise RuntimeError(
                    "logp or gradients are not finite at the model initial point; the model may be misspecified"
                )

        if n is None:
            start = perf_counter_ns()
            jax.block_until_ready(logp_dlogp_fn(initial_point))
            end = perf_counter_ns()
            n = max(5, int(5e9 / max(end - start, 1)))

        start = perf_counter_ns()
        for _ in range(n):
            jax.block_until_ready(logp_dlogp_fn(initial_point))
        end = perf_counter_ns()
        eval_time = (end - start) / n * 1e-9
        return float(eval_time)

    def estimate_num_steps_sampling(
        self,
        model: Model,
        *,
        tune: int | None = None,
        draws: int | None = None,
        seed: int | None = None,
    ) -> int:
        """Estimate total number of NUTS steps during warmup + sampling using NumPyro.

        Parameters
        ----------
        model : Model
            PyMC model to estimate steps for using a JAX/NumPyro NUTS kernel.
        tune : int | None, optional
            Warmup iterations. Defaults to the estimator setting if ``None``.
        draws : int | None, optional
            Sampling iterations. Defaults to the estimator setting if ``None``.
        seed : int | None, optional
            Random seed for the JAX run. Defaults to the estimator setting if ``None``.

        Returns
        -------
        int
            Total number of leapfrog steps across warmup + sampling.
        """
        import numpy as np

        try:
            import jax
            from numpyro.infer import MCMC, NUTS
            from pymc.sampling.jax import get_jaxified_logp
        except Exception as err:  # pragma: no cover - environment specific
            raise RuntimeError(
                "JAX and NumPyro are required for ModelSamplerEstimator."
            ) from err

        num_warmup = int(self.tune if tune is None else tune)
        num_samples = int(self.draws if draws is None else draws)

        initial_point = list(model.initial_point().values())
        logp_fn = get_jaxified_logp(model, negative_logp=False)
        nuts_kernel = NUTS(
            potential_fn=logp_fn,
            target_accept_prob=0.8,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            dense_mass=False,
        )

        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=1,
            postprocess_fn=None,
            chain_method="sequential",
            progress_bar=False,
        )

        if seed is None:
            rng_seed = int(np.random.default_rng().integers(2**32))
        else:
            rng_seed = int(seed)

        tune_rng, sample_rng = jax.random.split(jax.random.PRNGKey(int(rng_seed)), 2)
        mcmc.warmup(
            tune_rng,
            init_params=initial_point,
            extra_fields=("num_steps",),
            collect_warmup=True,
        )
        warmup_steps = int(mcmc.get_extra_fields()["num_steps"].sum())
        mcmc.run(sample_rng, extra_fields=("num_steps",))
        sample_steps = int(mcmc.get_extra_fields()["num_steps"].sum())
        return int(warmup_steps + sample_steps)

    def run(self, model: Model) -> pd.DataFrame:
        """Execute the estimation pipeline and return a single-row DataFrame.

        Parameters
        ----------
        model : Model
            PyMC model to evaluate.

        Returns
        -------
        pandas.DataFrame
            Single-row DataFrame with columns including ``num_steps``, ``eval_time_seconds``,
            ``sequential_chains``, and estimated sampling wall-clock time in seconds,
            minutes, and hours, along with metadata such as ``tune``, ``draws``, ``chains``,
            ``seed``, ``timestamp``, and ``model_name``.

        Examples
        --------
        .. code-block:: python

            df = ModelSamplerEstimator().run(model)
            df[
                [
                    "num_steps",
                    "eval_time_seconds",
                    "estimated_sampling_time_minutes",
                ]
            ]
        """
        import time

        steps = self.estimate_num_steps_sampling(
            model, tune=self.tune, draws=self.draws, seed=self.seed
        )
        eval_time_s = self.estimate_model_eval_time(model)

        sampling_time_seconds = float(
            eval_time_s * steps * max(self.sequential_chains, 1)
        )
        data = {
            "model_name": getattr(model, "name", "PyMCModel"),
            "num_steps": int(steps),
            "eval_time_seconds": float(eval_time_s),
            "sequential_chains": int(self.sequential_chains),
            "estimated_sampling_time_seconds": sampling_time_seconds,
            "estimated_sampling_time_minutes": sampling_time_seconds / 60.0,
            "estimated_sampling_time_hours": sampling_time_seconds / 3600.0,
            "tune": int(self.tune),
            "draws": int(self.draws),
            "chains": int(self.chains),
            "seed": int(self.seed) if self.seed is not None else None,
            "timestamp": pd.Timestamp.utcfromtimestamp(int(time.time())),
        }
        df = pd.DataFrame([data])
        return df
