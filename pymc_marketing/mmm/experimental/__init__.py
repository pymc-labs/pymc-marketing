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
"""Experimental, graph-first media mix models.

Compose the mean of each observed ``Equation`` from ``pymc_marketing.terms``
building blocks, ``Data`` references, and the graph terms in this namespace, then
hand the equations to ``MMM``. Data enter only through ``fit`` and
``sample_posterior_predictive`` as an ``xarray.Dataset`` in which each variable
keeps its own labeled dimensions. There is no model-wide dimension list, no
privileged outcome, and no hidden scaling.

This namespace does not change the stable MMM. Its interfaces are experimental.

Two targets sharing one likelihood family form one ``target`` dimension; different
families or observation layouts are separate equations sharing terms by identity:

.. code-block:: python

    import xarray as xr
    from pymc_extras.prior import Prior
    from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
    from pymc_marketing.mmm.experimental import MMM, Data, Equation, MediaTransform
    from pymc_marketing.terms import Intercept, Parameter, Transform

    response = MediaTransform(
        Data("spend"),
        GeometricAdstock(l_max=8),
        LogisticSaturation(
            priors={"beta": Prior("HalfNormal", dims=("channel", "target"))}
        ),
    )
    price_beta = Parameter("price_beta", Prior("Normal", dims="target"))
    mu = (
        Intercept(prior=Prior("Normal", dims=("product", "target")))
        + Transform(response, lambda value: value.sum(dim="channel"))
        + Data("price") * price_beta
    )
    sales = Equation(
        observed="sales",
        mu=mu,
        likelihood=Prior("Normal", sigma=Prior("HalfNormal", dims="target")),
    )
    mmm = MMM(sales)
    # train: xr.Dataset with spend(date, channel), price(date, product),
    # and sales(date, product, target); future omits sales.
    # mmm.fit(train)
    # mmm.sample_posterior_predictive(future)

Prediction is forward simulation with fitted parameters, not a causal-identification procedure.
"""

from pymc_marketing.mmm.experimental._graph import Data, Equation
from pymc_marketing.mmm.experimental._mmm import MMM
from pymc_marketing.mmm.experimental._terms import MediaTransform, Seasonality

__all__ = ["MMM", "Data", "Equation", "MediaTransform", "Seasonality"]
