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
"""Experimental, graph-first generalized additive models.

Compose the mean of each observed ``Equation`` from ``pymc_marketing.terms``
building blocks, ``Data`` references, and the graph terms in this namespace, then
hand the equations to ``GAM``. Data enter only through ``fit``,
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
Older draft GAM stores containing ``Named`` terms with an ``inner`` field are incompatible with this revision.
Recreate the model, refit it, and save a new store; ``check=False`` does not convert older specifications.

Two targets sharing one likelihood family form one ``target`` dimension; different
families or observation layouts are separate equations sharing terms by identity:

.. code-block:: python

    import xarray as xr
    from pymc_extras.prior import Prior
    from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
    from pymc_marketing.mmm.experimental import GAM, Data, Equation
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
    gam = GAM(sales)
    # train: xr.Dataset with spend(date, channel), price(date, product),
    # and sales(date, product, target); future omits sales.
    # gam.sample_prior_predictive(train)
    # gam.fit(train)
    # gam.save("model.zarr"); gam = GAM.load("model.zarr")
    # gam.sample_posterior_predictive(future)

Prediction is forward simulation with fitted parameters, not a causal-identification procedure.
"""

from pymc_marketing.mmm.experimental._gam import GAM
from pymc_marketing.mmm.experimental._graph import Data, Equation
from pymc_marketing.mmm.experimental._terms import MediaTransform

__all__ = ["GAM", "Data", "Equation", "MediaTransform"]
