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

.. code-block:: python

    from pymc_marketing.mmm import SaturationTransformation

    class InfiniteReturns(SaturationTransformation):
        def function(self, x, b):
            return b * x

        default_priors = {"b": {"dist": "HalfNormal", "kwargs": {"sigma": 1}}}

Plot the default priors for a saturation transformation:

.. code-block:: python

    from pymc_marketing.mmm import HillSaturation

    import matplotlib.pyplot as plt

    saturation = HillSaturation()
    prior = saturation.sample_prior()
    curve = saturation.sample_curve(prior)
    saturation.plot_curve(curve)
    plt.show()

Define a hierarchical saturation function with only hierarchical parameters
for saturation parameter of logistic saturation.

.. code-block:: python

    from pymc_marketing.mmm import LogisticSaturation

    priors = {
        "lam": {
            "dist": "Gamma",
            "kwargs": {
                "alpha": {
                    "dist": "HalfNormal",
                    "kwargs": {"sigma": 1},
                },
                "beta": {
                    "dist": "HalfNormal",
                    "kwargs": {"sigma": 1},
                },
            },
            "dims": "channel",
        },
        "beta": {
            "dist": "HalfNormal",
            "kwargs": {"sigma": 1},
            "dims": "channel",
        },
    }
    saturation = LogisticSaturation(priors=priors)

"""

import numpy as np
import xarray as xr

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

    Make use of plotting capabilities to understand the transformation and its
    priors

    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np

        saturation = InfiniteReturns()

        rng = np.random.default_rng(0)

        prior = saturation.sample_prior(random_seed=rng)
        curve = saturation.sample_curve(prior)
        saturation.plot_curve(curve, sample_kwargs={"rng": rng})
        plt.show()

    """

    prefix: str = "saturation"

    def sample_curve(
        self,
        parameters: xr.Dataset,
        max_value: float = 1.0,
    ) -> xr.DataArray:
        """Sample the curve of the saturation transformation given parameters.

        Parameters
        ----------
        parameters : xr.Dataset
            Dataset with the parameters of the saturation transformation.
        max_value : float, optional
            Maximum value of the curve, by default 1.0.

        Returns
        -------
        xr.DataArray
            Curve of the saturation transformation.

        """
        x = np.linspace(0, max_value, 100)

        coords = {
            "x": x,
        }

        return self._sample_curve(
            var_name="saturation",
            parameters=parameters,
            x=x,
            coords=coords,
        )


class LogisticSaturation(SaturationTransformation):
    """Wrapper around logistic saturation function.

    For more information, see :func:`pymc_marketing.mmm.transformers.logistic_saturation`.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from pymc_marketing.mmm import LogisticSaturation

        rng = np.random.default_rng(0)

        adstock = LogisticSaturation()
        prior = adstock.sample_prior(random_seed=rng)
        curve = adstock.sample_curve(prior)
        adstock.plot_curve(curve, sample_kwargs={"rng": rng})
        plt.show()

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

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from pymc_marketing.mmm import TanhSaturation

        rng = np.random.default_rng(0)

        adstock = TanhSaturation()
        prior = adstock.sample_prior(random_seed=rng)
        curve = adstock.sample_curve(prior)
        adstock.plot_curve(curve, sample_kwargs={"rng": rng})
        plt.show()

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

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from pymc_marketing.mmm import TanhSaturationBaselined

        rng = np.random.default_rng(0)

        adstock = TanhSaturationBaselined()
        prior = adstock.sample_prior(random_seed=rng)
        curve = adstock.sample_curve(prior)
        adstock.plot_curve(curve, sample_kwargs={"rng": rng})
        plt.show()

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

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from pymc_marketing.mmm import MichaelisMentenSaturation

        rng = np.random.default_rng(0)

        adstock = MichaelisMentenSaturation()
        prior = adstock.sample_prior(random_seed=rng)
        curve = adstock.sample_curve(prior)
        adstock.plot_curve(curve, sample_kwargs={"rng": rng})
        plt.show()

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

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from pymc_marketing.mmm import HillSaturation

        rng = np.random.default_rng(0)

        adstock = HillSaturation()
        prior = adstock.sample_prior(random_seed=rng)
        curve = adstock.sample_curve(prior)
        adstock.plot_curve(curve, sample_kwargs={"rng": rng})
        plt.show()

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
