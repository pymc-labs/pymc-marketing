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
"""Functions to manipulate PyMC models as graphs."""

import pymc as pm
import pymc.dims as pmd
from pymc.model.fgraph import (
    extract_dims,
    fgraph_from_model,
    model_free_rv,
    model_from_fgraph,
)
from pytensor import config
from pytensor.graph import FunctionGraph, Variable, ancestors
from pytensor.tensor import TensorVariable
from pytensor.tensor.basic import infer_shape_db
from pytensor.tensor.rewriting.shape import ShapeFeature
from pytensor.xtensor.type import XTensorVariable


def get_symbolic_rv_shape(
    rv: Variable, raise_if_rv_in_graph: bool = True
) -> tuple[TensorVariable]:
    """Get the symbolic shape of the random variables in the graph.

    Parameters
    ----------
    rv : Variable
        The random variable to get the shape of.
    raise_if_rv_in_graph : bool, optional
        Whether to raise an error if the random variable is still in the graph.

    Returns
    -------
    tuple[TensorVariable]
        The shape of the random variables.
    """
    # TODO: Remove this when utility is available in pymc.pytensorf
    shape_fg = FunctionGraph(
        outputs=list(rv.shape), features=[ShapeFeature()], clone=True
    )
    with config.change_flags(optdb__max_use_ratio=10, cxx=""):
        infer_shape_db.default_query.rewrite(shape_fg)
    rv_shape = tuple(shape_fg.outputs)

    if raise_if_rv_in_graph and rv in (ancestors(rv_shape)):
        raise ValueError("rv_shape still depends the original rv")

    return rv_shape


def deterministics_to_flat(model: pm.Model, names: list[str]) -> pm.Model:
    """Replace all specified Deterministic nodes in a pm.Model with Flat.

    This is useful to capture some state from a model and to then sample from
    the model using that state. For example, capturing the mean of a distribution
    or a value of a deterministic variable.

    See :class:`pymc_marketing.mmm.hsgp.SoftPlusHSGP` for an example of how this
    is used to keep a variable centered around 1.0 during sampling but stay continuous
    with new values.

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

    Examples
    --------
    Replace single Deterministic with Flat and sample as if it were zeros.

    .. code-block:: python

        import pymc as pm
        import numpy as np
        import xarray as xr
        from pymc_marketing.model_graph import deterministics_to_flat

        with pm.Model() as model:
            x = pm.Normal("x", mu=0, sigma=1)
            y = pm.Deterministic("y", x**2)
            z = pm.Deterministic("z", x + y)

        new_model = deterministics_to_flat(model, ["y"])

        chains, draws = 2, 100
        mock_posterior = xr.Dataset(
            {
                "y": (("chain", "draw"), np.zeros((chains, draws))),
            },
            coords={"chain": np.arange(chains), "draw": np.arange(draws)},
        )

        x_z_given_y = pm.sample_posterior_predictive(
            mock_posterior,
            model=new_model,
            var_names=["x", "z"],
        ).posterior_predictive

        np.testing.assert_allclose(
            x_z_given_y["x"],
            x_z_given_y["z"],
        )

    """
    fg, memo = fgraph_from_model(model, inlined_views=True)

    model_variables = [x for x in set(model.deterministics) if x.name in names]
    replacements = {}

    for variable in model_variables:
        model_var = memo[variable]
        dims = extract_dims(model_var)
        underlying_var = model_var.owner.inputs[0]
        underlying_shape = get_symbolic_rv_shape(underlying_var)
        if isinstance(underlying_var, XTensorVariable):
            new_rv = pmd.Flat.dist(
                dim_lengths=dict(
                    zip(underlying_var.dims, underlying_shape, strict=True)
                )
            )
        else:
            new_rv = pm.Flat.dist(shape=underlying_shape)
        new_rv.name = model_var.name

        new_free_rv = model_free_rv(
            new_rv,
            new_rv.type(name=model_var.name),
            None,
            *dims,
        )

        replacements[model_var] = new_free_rv

    fg.replace_all(
        replacements.items(),
        reason="deterministics_to_flat",
        import_missing=True,
    )

    new_model = model_from_fgraph(fg, mutate_fgraph=True)
    return new_model
