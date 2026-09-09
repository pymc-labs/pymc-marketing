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
"""Experimental, terms-based media mix models.

``MMM`` creates a conventional outcome equation using configured transformation
instances, or accepts a complete replacement through ``mmm.y``. ``Equation``
introduces a stochastic mechanism; ``Data`` references a named input column.
Compose expressions with the existing ``pymc_marketing.terms`` building blocks.

This namespace does not change the stable MMM. Its interfaces are experimental.

Attach a raw-scale intermediate mechanism and explicitly choose its prediction policy:

.. code-block:: python

    from pymc_extras.prior import Prior
    from pymc_marketing.mmm import GeometricAdstock, MichaelisMentenSaturation
    from pymc_marketing.mmm.experimental import MMM, Equation

    mmm = MMM(
        target_column="sales",
        channel_columns=["tv", "search"],
        yearly_seasonality=2,
        media_transform=(
            GeometricAdstock(l_max=8),
            MichaelisMentenSaturation(),
        ),
    )
    mmm.media["search"].equation = Equation(
        name="search_demand",
        mu=2 * mmm.media["tv"].value,
        likelihood=Prior("Normal", sigma=1),
    )
    # Fit with measured TV, search, and sales columns.
    # mmm.fit(training_data)
    # Hold supplied search fixed (the default).
    # mmm.sample_posterior_predictive(future_data, condition_on=["search"])
    # Generate search under its equation.
    # mmm.sample_posterior_predictive(future_data, condition_on=())

Replacing ``mmm.y`` replaces the whole outcome recipe and disables its default target scaling.
Channel contributions retain their fitted scales.
Unbound equations with no observation column are latent, while outcome and channel slots supply observation bindings.
Prediction is forward simulation with fitted parameters, not a causal-identification procedure.
"""

from pymc_marketing.mmm.experimental._graph import Data, Equation
from pymc_marketing.mmm.experimental._mmm import MMM

__all__ = ["MMM", "Data", "Equation"]
