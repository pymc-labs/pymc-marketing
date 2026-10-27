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
"""Experimental, graph-first Structural Causal Models (SCMs).

Compose the mean of each observed ``Equation`` from ``pymc_marketing.terms``
building blocks, ``Data`` references, and the graph terms in this namespace, then
hand the equations to ``SCM``. Data enter only through ``fit``,
``sample_prior_predictive``, and ``sample_posterior_predictive`` as an
``xarray.Dataset`` in which each variable keeps its own labeled dimensions. There
is no model-wide dimension list, no privileged outcome, and no hidden scaling.

``Data("spend") >> adstock >> saturation`` applies configured transformations
along ``date``; their priors without ``dims`` take the input's other dimensions.
Fourier seasonality components such as ``YearlyFourier`` enter an expression
directly and are evaluated on the dataset dates. ``expression.named(name)`` records
a deterministic and ``expression.sum(dim)`` reduces a dimension.

This namespace does not change the stable MMM. Its interfaces are experimental.
``expression.named(name)`` uses the shared ``Named(name, expr=expression)`` term.
Older drafts using ``Named.inner``, MMM graph identifiers, or the ``GAM`` format are incompatible.
The format marker is ``pymc_marketing.experimental.SCM/1``.
Recreate the model with ``SCM``, refit it, and save a new store; ``check=False`` does not convert older specifications.

Two targets sharing one likelihood family form one ``target`` dimension; different
families or observation layouts are separate equations sharing terms by identity:

.. code-block:: python

    import xarray as xr
    from pymc_extras.prior import Prior
    from pymc_marketing.experimental.terms import SCM, Data, Equation
    from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
    from pymc_marketing.terms import Intercept, Parameter

    response = (
        Data("spend")
        >> GeometricAdstock(l_max=8)
        >> LogisticSaturation(
            priors={"beta": Prior("HalfNormal", dims=("channel", "target"))}
        )
    ).named("channel_contribution")
    price_beta = Parameter("price_beta", Prior("Normal", dims="target"))
    mu = (
        Intercept(prior=Prior("Normal", dims=("product", "target")))
        + response.sum("channel")
        + Data("price") * price_beta
    )
    sales = Equation(
        observed="sales",
        mu=mu,
        likelihood=Prior("Normal", sigma=Prior("HalfNormal", dims="target")),
    )
    scm = SCM(sales)
    # train: xr.Dataset with spend(date, channel), price(date, product),
    # and sales(date, product, target); future omits sales.
    # scm.sample_prior_predictive(train)
    # scm.fit(train)
    # scm.save("model.zarr"); scm = SCM.load("model.zarr")
    # scm.sample_posterior_predictive(future)

Causal interpretation depends on the assumptions encoded by the equations.
Prediction is forward simulation, not automatic causal identification, intervention, or counterfactual inference.

``Problem`` binds selected fitted deterministic terms and infers their decision inputs.
Construct it once with decided dates and explicit bounds, fill its labeled ``decisions``
template, score plans with ``evaluate``, and reuse ``solve`` across initial plans and
declared-parameter overrides without rebinding or recompiling.
Callbacks write every reduction, trade-off, and unit conversion.
Inputs, bounds, objectives, residuals, and allocations remain in original units.
``OptimizationResult`` reports fixed per-solve scaling and untouched SciPy diagnostics.
Measured history and carryover rows are explicit and fixed; outputs exclude history,
while allocations contain only decided rows.
"""

from pymc_marketing.experimental.terms._graph import Data, Equation
from pymc_marketing.experimental.terms._optimizer import OptimizationResult, Problem
from pymc_marketing.experimental.terms._scm import SCM
from pymc_marketing.experimental.terms._terms import MediaTransform

__all__ = [
    "SCM",
    "Data",
    "Equation",
    "MediaTransform",
    "OptimizationResult",
    "Problem",
]
