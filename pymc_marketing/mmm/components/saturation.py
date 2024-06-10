#   Copyright 2024 The PyMC Labs Developers
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
"""Saturation transformations for the MMM model.

Each of these transformations is a subclass of
:class:`pymc_marketing.mmm.components.saturation.SaturationTransformation` and defines a function
that takes media and return the saturated media. The parameters of the function
are the parameters of the saturation transformation.

Examples
--------
Create a new saturation transformation:

from pymc_marketing.mmm.components.saturation import SaturationTransformation

.. code-block:: python

    class InfiniteReturns(SaturationTransformation):
        def function(self, x, b):
            return b * x

        default_priors = {"b": {"dist": "HalfNormal", "kwargs": {"sigma": 1}}}

"""

from pymc_marketing.mmm.components.base import Transformation
from pymc_marketing.mmm.transformers import (
    hill_saturation,
    logistic_saturation,
    michaelis_menten,
    tanh_saturation,
    tanh_saturation_baselined,
)


class SaturationTransformation(Transformation):
    """Subclass for all saturation transformations.

    In order to use a custom saturation transformation, subclass and define:

    - `function`: function to take x to contributions
    - `default_priors`: default distributions for each parameter in function

    By subclassing from this method, lift test integration will come for free!

    Examples
    ----------
    Make a non-saturating saturation transformation

    .. code-block:: python
    
        from pymc_marketing.mmm import SaturationTransformation

        def infinite_returns(x, b):
            return b * x

        class InfiniteReturns(SaturationTransformation):
            function = infinite_returns
            default_priors = {"b": {"dist": "HalfNormal", "kwargs": {"sigma": 1}}}

    """

    prefix: str = "saturation"


class LogisticSaturation(SaturationTransformation):
    """Wrapper around logistic saturation function.

    For more information, see :func:`pymc_marketing.mmm.transformers.logistic_saturation`.

    """

    lookup_name = "logistic"

    def function(self, x, lam, beta):
        return beta * logistic_saturation(x, lam)

    default_priors = {
        "lam": {"dist": "Gamma", "kwargs": {"alpha": 3, "beta": 1}},
        "beta": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
    }


class TanhSaturation(SaturationTransformation):
    """Wrapper around tanh saturation function.

    For more information, see :func:`pymc_marketing.mmm.transformers.tanh_saturation`.

    """

    lookup_name = "tanh"

    def function(self, x, b, c, beta):
        return beta * tanh_saturation(x, b, c)

    default_priors = {
        "b": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "c": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "beta": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
    }


class TanhSaturationBaselined(SaturationTransformation):
    """Wrapper around tanh saturation function.

    For more information, see :func:`pymc_marketing.mmm.transformers.tanh_saturation_baselined`.

    """

    lookup_name = "tanh_baselined"

    def function(self, x, x0, gain, r, beta):
        return beta * tanh_saturation_baselined(x, x0, gain, r)

    default_priors = {
        "x0": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "gain": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "r": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "beta": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
    }


class MichaelisMentenSaturation(SaturationTransformation):
    """Wrapper around Michaelis-Menten saturation function.

    For more information, see :func:`pymc_marketing.mmm.transformers.michaelis_menten`.

    """

    lookup_name = "michaelis_menten"

    function = michaelis_menten

    default_priors = {
        "alpha": {"dist": "Gamma", "kwargs": {"mu": 2, "sigma": 1}},
        "lam": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
    }


class HillSaturation(SaturationTransformation):
    """Wrapper around Hill saturation function.

    For more information, see :func:`pymc_marketing.mmm.transformers.hill_saturation`.

    """

    lookup_name = "hill"

    function = hill_saturation

    default_priors = {
        "sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
        "beta": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
        "lam": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
    }


SATURATION_TRANSFORMATIONS: dict[str, type[SaturationTransformation]] = {
    cls.lookup_name: cls
    for cls in [
        LogisticSaturation,
        TanhSaturation,
        TanhSaturationBaselined,
        MichaelisMentenSaturation,
        HillSaturation,
    ]
}


def _get_saturation_function(
    function: str | SaturationTransformation,
) -> SaturationTransformation:
    """Helper for use in the MMM to get a saturation function."""
    if isinstance(function, SaturationTransformation):
        return function

    if function not in SATURATION_TRANSFORMATIONS:
        raise ValueError(
            f"Unknown saturation function: {function}. Choose from {list(SATURATION_TRANSFORMATIONS.keys())}"
        )

    return SATURATION_TRANSFORMATIONS[function]()
