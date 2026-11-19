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
"""Conversion models.

This package is in alpha: the API and defaults may change between releases.

The entry point is :class:`~pymc_marketing.conversion.funnel.FunnelModel`, a
Bayesian model of the conversion through the steps of a funnel. It reads
counts by step and by any number of groups, such as the variants of an
experiment or customer segments, and returns posterior draws of the step
conversions and of the quantities derived from them.
:mod:`pymc_marketing.decision` turns those draws into comparisons between
variants.

Examples
--------
Fit the stage counts of an A/B test and compare the overall conversion of the
two variants:

.. code-block:: python

    import pandas as pd

    from pymc_marketing.conversion import FunnelModel, to_funnel_dataset
    from pymc_marketing.decision import prob_best

    stages = ["visit", "signup", "paid"]
    df = pd.DataFrame(
        {
            "stage": stages * 2,
            "variant": ["control"] * 3 + ["treatment"] * 3,
            "count": [1000, 300, 60, 1000, 330, 75],
        }
    )

    model = FunnelModel()
    model.fit(to_funnel_dataset(df, stage_order=stages))
    prob_best(model.overall_conversion(), "variant")
"""

from pymc_marketing.conversion.data import to_funnel_dataset
from pymc_marketing.conversion.funnel import FunnelModel, pooled_p
from pymc_marketing.conversion.plotting import plot_funnel

__all__ = ["FunnelModel", "plot_funnel", "pooled_p", "to_funnel_dataset"]
