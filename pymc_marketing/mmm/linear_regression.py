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
"""Linear regression model implemented using the MMM class."""

from pymc_marketing.mmm.components.adstock import NoAdstock
from pymc_marketing.mmm.components.saturation import NoSaturation
from pymc_marketing.mmm.mmm import MMM


def FancyLinearRegression(
    **mmm_kwargs,
) -> MMM:
    """Create wrapper around MMM for a linear regression model.

    See :func:`pymc_marketing.mmm.mmm.MMM` for more details.

    Parameters
    ----------
    mmm_kwargs
        Keyword arguments to pass to the MMM constructor.

    Returns
    -------
    MMM
        An instance of the MMM class with linear regression settings.

    Examples
    --------
    Load a saved MMM model with linear regression settings:

    .. code-block:: python

        from pymc_marketing.mmm import MMM

        linear_regression = MMM.load("linear_regression_model.nc")

    """
    return MMM(
        adstock=NoAdstock(l_max=1),
        saturation=NoSaturation(),
        **mmm_kwargs,
    )
