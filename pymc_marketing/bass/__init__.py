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
"""Bass diffusion model for product adoption forecasting.

The recommended entry point is :class:`BassModel`, which wraps the model in a
:class:`~pymc_marketing.model_builder.ModelBuilder` interface with standard
``.fit()``, ``.save()``, and ``.load()`` methods.

The lower-level :func:`create_bass_model` and :class:`BassPriors` are still
available for users who need the raw ``pm.Model`` without the class wrapper.

Examples
--------
Fit a single-product model from an array of adoption counts:

.. code-block:: python

    import numpy as np
    from pymc_marketing.bass import BassModel

    model = BassModel()
    idata = model.fit(data=np.random.poisson(lam=100, size=50))

Generate synthetic data from the prior, then fit the model:

.. code-block:: python

    import xarray as xr
    import pymc as pm

    ds = xr.Dataset({"T": np.arange(50)})
    model = BassModel()
    model.build_model(data=ds)

    with model.model:
        prior = pm.sample_prior_predictive(draws=50, random_seed=42)
        y_sim = prior.prior["y"].sel(draw=0, chain=0)

    idata = model.fit(data=y_sim.values)
"""

from pymc_marketing.bass.data import to_bass_dataset
from pymc_marketing.bass.model import BassModel, BassPriors, create_bass_model

__all__ = [
    "BassModel",
    "BassPriors",
    "create_bass_model",
    "to_bass_dataset",
]
