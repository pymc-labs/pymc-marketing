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
"""Functions to manipulate PyMC models as graphs."""

import pymc as pm
from pymc.model.fgraph import (
    extract_dims,
    fgraph_from_model,
    model_free_rv,
    model_from_fgraph,
)
from pymc.pytensorf import toposort_replace
from pytensor.graph import rewrite_graph
from pytensor.tensor.basic import infer_shape_db


def deterministics_to_flat(model, names):
    """Replace all Deterministics in a model with Flat.

    Parameters
    ----------
    model : pm.Model
        PyMC model to be transformed
    names : list[str]
        Names of the deterministic variables to be replaced by flat

    Returns
    -------
    new_model : pm.Model
        New model with all priors replaced by flat priors
    """
    fg, memo = fgraph_from_model(model, inlined_views=True)

    model_variables = [x for x in set(model.deterministics) if x.name in names]
    replacements = {}

    for variable in model_variables:
        model_var = memo[variable]
        dims = extract_dims(model_var)

        new_rv = pm.Flat.dist(shape=model_var.shape)
        new_rv.name = model_var.name

        replacements[model_var] = model_free_rv(
            new_rv,
            new_rv.type(name=model_var.name),
            None,
            *dims,
        )

    toposort_replace(fg, replacements=tuple(replacements.items()))
    fg = rewrite_graph(
        fg,
        include=("ShapeOpt",),
        custom_rewrite=infer_shape_db.default_query,
        clone=False,
    )

    new_model = model_from_fgraph(fg, mutate_fgraph=True)

    return new_model
