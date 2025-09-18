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
