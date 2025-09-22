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
import pytensor
import pytensor.tensor as pt
from arviz import InferenceData
from pymc import Model
from pymc.model.fgraph import (
    FunctionGraph,
    ModelVar,
    extract_dims,
    fgraph_from_model,
    model_from_fgraph,
)
from pytensor import clone_replace
from pytensor.graph import rewrite_graph, vectorize_graph
from pytensor.graph.basic import ancestors

try:
    from pymc.pytensorf import rvs_in_graph
except ImportError:
    from pymc.logprob.utils import rvs_in_graph


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
