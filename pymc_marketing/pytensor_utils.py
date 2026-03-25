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

"""PyTensor utility functions."""

from collections import Counter

import arviz as az
import pandas as pd
import pytensor.tensor as pt
from arviz import InferenceData
from pymc.model.core import Model
from pymc.model.fgraph import (
    ModelVar,
    extract_dims,
    fgraph_from_model,
    model_from_fgraph,
)
from pymc.pytensorf import rvs_in_graph
from pytensor import as_symbolic
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting import rewrite_graph
from pytensor.graph.traversal import ancestors
from pytensor.xtensor import xtensor_constant
from pytensor.xtensor.vectorization import vectorize_graph


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

    # Track dims and build a mapping from base variable names to prefixed names
    dims = set()
    base_to_prefixed: dict[str, str] = {}
    for v in f2.outputs:
        # Only prefix if not in exclude_vars and has a valid name
        old_name = getattr(v, "name", None)
        if old_name and (old_name not in exclude_vars):
            new_name = f"{prefix}_{old_name}"
            v.name = new_name
            if isinstance(v.owner.op, ModelVar):
                rv = v.owner.inputs[0]
                rv.name = new_name
            # Record base to prefixed mapping for subsequent value-var renaming
            base_to_prefixed[old_name] = new_name
        dims.update(extract_dims(v))

    # Also collect ModelVar outputs that may not be listed among f2.outputs
    # (e.g., observed RVs or deterministics created internally)
    for var in list(f2.variables):
        if (
            (owner := getattr(var, "owner", None)) is not None
            and isinstance(owner.op, ModelVar)
            and isinstance(name := getattr(var, "name", None), str)
            and name
            and name not in exclude_vars
            and name not in base_to_prefixed
            and not name.startswith(prefix + "_")
        ):
            new_name = f"{prefix}_{name}"
            var.name = new_name
            owner.inputs[0].name = new_name
            base_to_prefixed[name] = new_name

    # Don't rename dimensions that belong to excluded variables
    dims_rename = {
        dim: as_symbolic(f"{prefix}_{dim.data}")
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

    # Also rename associated transformed/value variables to keep names unique across merged graphs.
    # Example patterns include: "<base>", "<base>_log__", "<base>_logodds__", etc.
    # We only attempt renames for bases we actually prefixed above.
    if base_to_prefixed:
        for var in list(f2.variables):
            if (
                isinstance(name := getattr(var, "name", None), str)
                and name
                and name not in exclude_vars
                and (
                    match := next(
                        (
                            (base, prefixed)
                            for base, prefixed in base_to_prefixed.items()
                            if isinstance(base, str)
                            and base
                            and (
                                name == base
                                or name.startswith(base + "_")
                                or name.startswith(base + "__")
                            )
                        ),
                        None,
                    )
                )
            ):
                base, prefixed = match
                var.name = name.replace(base, prefixed, 1)

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


def validate_unique_value_vars(model: Model) -> None:
    """Validate that a model has unique, non-null value var names and 1:1 mappings.

    This checks that:
    - All entries in ``model.value_vars`` have unique, non-empty names
    - Keys of ``model.values_to_rvs`` (value vars) also have unique names
    - ``model.rvs_to_values`` mapping is consistent (bijection by names)
    """
    # Check value_vars names are unique and non-empty
    value_vars = list(getattr(model, "value_vars", []))
    value_var_names = [getattr(v, "name", None) for v in value_vars]
    if any(n is None or n == "" for n in value_var_names):
        raise ValueError("Found unnamed value variables in model.value_vars")
    dup_vnames = [n for n, c in Counter(value_var_names).items() if c > 1]
    if dup_vnames:
        raise ValueError(f"Duplicate value variable names: {dup_vnames}")

    # Check values_to_rvs keys are unique by name
    v2r = getattr(model, "values_to_rvs", {})
    v2r_value_vars = list(v2r.keys())
    v2r_value_names = [getattr(v, "name", None) for v in v2r_value_vars]
    if any(n is None or n == "" for n in v2r_value_names):
        raise ValueError("Found unnamed value variables in values_to_rvs")
    # Some observed/deterministic value-vars may legitimately share names across merged models
    # if they were intentionally merged on (e.g., merge_on) or are non-free and identical.
    # Only enforce uniqueness among value vars that correspond to free RVs.
    _ = {
        getattr(v2r[v], "name", None)
        for v in v2r_value_vars
        if v in getattr(model, "value_vars", [])
        and v2r.get(v) in getattr(model, "free_RVs", [])
    }
    # Map back to the value-var names for those free RVs
    free_value_var_names = [
        getattr(model.rvs_to_values[rv], "name", None) for rv in model.free_RVs
    ]
    dup_map_names = [n for n, c in Counter(free_value_var_names).items() if n and c > 1]
    if dup_map_names:
        raise ValueError("Duplicate value variable names for free RVs: {dup_map_names}")

    # Check consistency of reverse mapping by names
    r2v = getattr(model, "rvs_to_values", {})
    # Names on the value side of both dicts should align set-wise
    r2v_value_names = [getattr(v, "name", None) for v in r2v.values()]
    if set(r2v_value_names) != set(v2r_value_names):
        raise ValueError(
            "Mismatch between values_to_rvs and rvs_to_values by value var names"
        )


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
    placeholder_replace_dict = {pymc_model[rv.name]: rv.clone() for rv in needed_rvs}

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
        replace_dict[placeholder] = xtensor_constant(
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

    @property
    def default_nuts_kwargs(self) -> dict:
        """Default keyword arguments for a NumPyro NUTS kernel.

        Mirrors the current hard-coded defaults used in this estimator.
        """
        return {
            "target_accept_prob": 0.8,
            "adapt_step_size": True,
            "adapt_mass_matrix": True,
            "dense_mass": False,
        }

    @property
    def default_mcmc_kwargs(self) -> dict:
        """Default keyword arguments for a NumPyro MCMC runner.

        Parameters that depend on the run size (``num_warmup`` and ``num_samples``)
        are intentionally excluded and provided explicitly by the estimator.
        """
        return {
            "num_chains": 1,
            "postprocess_fn": None,
            "chain_method": "sequential",
            "progress_bar": False,
        }

    def estimate_model_eval_time(
        self, model: Model, num_evaluations: int | None = None
    ) -> float:
        """Estimate average evaluation time (seconds) of logp+dlogp using JAX.

        Parameters
        ----------
        model : Model
            PyMC model whose logp and gradients are jitted and evaluated.
        num_evaluations : int | None, optional
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

        if num_evaluations is None:
            start = perf_counter_ns()
            jax.block_until_ready(logp_dlogp_fn(initial_point))
            end = perf_counter_ns()
            num_evaluations = max(5, int(5e9 / max(end - start, 1)))

        start = perf_counter_ns()
        for _ in range(num_evaluations):
            jax.block_until_ready(logp_dlogp_fn(initial_point))
        end = perf_counter_ns()
        eval_time = (end - start) / num_evaluations * 1e-9
        return float(eval_time)

    def estimate_num_steps_sampling(
        self,
        model: Model,
        *,
        tune: int | None = None,
        draws: int | None = None,
        seed: int | None = None,
        nuts_kwargs: dict | None = None,
        mcmc_kwargs: dict | None = None,
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
        nuts_kwargs : dict | None, optional
            Additional keyword arguments passed to ``numpyro.infer.NUTS``. If not provided,
            the estimator's ``default_nuts_kwargs`` are used. Provided values override
            the defaults.
        mcmc_kwargs : dict | None, optional
            Additional keyword arguments passed to ``numpyro.infer.MCMC`` (excluding
            ``num_warmup`` and ``num_samples``, which are set by ``tune``/``draws``). If not
            provided, the estimator's ``default_mcmc_kwargs`` are used. Provided values
            override the defaults.

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
        merged_nuts_kwargs = {**self.default_nuts_kwargs, **(nuts_kwargs or {})}
        nuts_kernel = NUTS(
            potential_fn=logp_fn,
            **merged_nuts_kwargs,
        )

        merged_mcmc_kwargs = {**self.default_mcmc_kwargs, **(mcmc_kwargs or {})}
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            **merged_mcmc_kwargs,
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
        return pd.DataFrame([data])
