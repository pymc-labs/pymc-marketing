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
"""Components for media transformation in the MMM model.

Examples
--------
Use custom transformations for media in the MMM model:

.. code-block:: python

    from pymc_marketing.prior import Prior
    from pymc_marketing.mmm import (
        SaturationTransformation,
        MMM,
        WeibullPDFAdstock,
    )


    class InfiniteReturns(SaturationTransformation):
        def function(self, x, b):
            return b * x

        default_priors = {"b": Prior("HalfNormal")}


    saturation = InfiniteReturns()
    adstock = WeibullPDFAdstock(l_max=15)

    mmm = MMM(
        ...,
        saturation=saturation,
        adstock=adstock,
        adstock_first=True,
    )

"""
