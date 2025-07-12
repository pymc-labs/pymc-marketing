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
    from pymc_marketing.prior import Prior


    class InfiniteReturns(SaturationTransformation):
        lookup_name: str = "infinite_returns"

        def function(self, x, b):
            return b * x

        default_priors = {"b": Prior("HalfNormal", sigma=1)}

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

    from pymc_marketing.prior import Prior
    from pymc_marketing.mmm import LogisticSaturation

    hierarchical_lam = Prior(
        "Gamma",
        alpha=Prior("HalfNormal"),
        beta=Prior("HalfNormal"),
        dims="channel",
    )
    priors = {
        "lam": hierarchical_lam,
        "beta": Prior("HalfNormal", dims="channel"),
    }
    saturation = LogisticSaturation(priors=priors)

"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import xarray as xr
from pydantic import Field, InstanceOf, validate_call

from pymc_marketing.deserialize import deserialize, register_deserialization
from pymc_marketing.mmm.components.base import (
    Transformation,
    create_registration_meta,
)
from pymc_marketing.mmm.transformers import (
    hill_function,
    hill_saturation_sigmoid,
    inverse_scaled_logistic_saturation,
    logistic_saturation,
    michaelis_menten,
    root_saturation,
    tanh_saturation,
    tanh_saturation_baselined,
)
from pymc_marketing.prior import Prior

SATURATION_TRANSFORMATIONS: dict[str, type[SaturationTransformation]] = {}

SaturationRegistrationMeta = create_registration_meta(SATURATION_TRANSFORMATIONS)


class SaturationTransformation(Transformation, metaclass=SaturationRegistrationMeta):  # type: ignore
    """Subclass for all saturation transformations.

    In order to use a custom saturation transformation, subclass and define:

    - `function`: function to take x to contributions
    - `default_priors`: default distributions for each parameter in function

    By subclassing from this method, lift test integration will come for free!

    Examples
    --------
    Make a non-saturating saturation transformation

    .. code-block:: python

        from pymc_marketing.mmm import SaturationTransformation
        from pymc_marketing.prior import Prior


        def infinite_returns(x, b):
            return b * x


        class InfiniteReturns(SaturationTransformation):
            lookup_name = "infinite_returns"
            function = infinite_returns
            default_priors = {"b": Prior("HalfNormal")}

    Make use of plotting capabilities to understand the transformation and its
    priors

    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np

        saturation = InfiniteReturns()

        rng = np.random.default_rng(0)

        prior = saturation.sample_prior(random_seed=rng)
        curve = saturation.sample_curve(prior)
        saturation.plot_curve(curve, random_seed=rng)
        plt.show()

    """

    prefix: str = "saturation"

    @validate_call
    def sample_curve(
        self,
        parameters: InstanceOf[xr.Dataset] = Field(
            ..., description="Parameters of the saturation transformation."
        ),
        max_value: float = Field(1.0, gt=0, description="Maximum range value."),
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
        adstock.plot_curve(curve, random_seed=rng)
        plt.show()

    """

    lookup_name = "logistic"

    def function(self, x, lam, beta):
        """Logistic saturation function."""
        return beta * logistic_saturation(x, lam)

    default_priors = {
        "lam": Prior("Gamma", alpha=3, beta=1),
        "beta": Prior("HalfNormal", sigma=2),
    }


class InverseScaledLogisticSaturation(SaturationTransformation):
    """Wrapper around inverse scaled logistic saturation function.

    For more information, see :func:`pymc_marketing.mmm.transformers.inverse_scaled_logistic_saturation`.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from pymc_marketing.mmm import InverseScaledLogisticSaturation

        rng = np.random.default_rng(0)

        adstock = InverseScaledLogisticSaturation()
        prior = adstock.sample_prior(random_seed=rng)
        curve = adstock.sample_curve(prior)
        adstock.plot_curve(curve, random_seed=rng)
        plt.show()

    """

    lookup_name = "inverse_scaled_logistic"

    def function(self, x, lam, beta):
        """Inverse scaled logistic saturation function."""
        return beta * inverse_scaled_logistic_saturation(x, lam)

    default_priors = {
        "lam": Prior("Gamma", alpha=0.5, beta=1),
        "beta": Prior("HalfNormal", sigma=2),
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
        adstock.plot_curve(curve, random_seed=rng)
        plt.show()

    """

    lookup_name = "tanh"

    def function(self, x, b, c):
        """Tanh saturation function."""
        return tanh_saturation(x, b, c)

    default_priors = {
        "b": Prior("HalfNormal", sigma=1),
        "c": Prior("HalfNormal", sigma=1),
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
        adstock.plot_curve(curve, random_seed=rng)
        plt.show()

    """

    lookup_name = "tanh_baselined"

    def function(self, x, x0, gain, r, beta):
        """Tanh saturation function."""
        return beta * tanh_saturation_baselined(x, x0, gain, r)

    default_priors = {
        "x0": Prior("HalfNormal", sigma=1),
        "gain": Prior("HalfNormal", sigma=1),
        "r": Prior("HalfNormal", sigma=1),
        "beta": Prior("HalfNormal", sigma=1),
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
        adstock.plot_curve(curve, random_seed=rng)
        plt.show()

    """

    lookup_name = "michaelis_menten"

    def function(self, x, alpha, lam):
        """Michaelis-Menten saturation function."""
        return pt.as_tensor_variable(michaelis_menten(x, alpha, lam))

    default_priors = {
        "alpha": Prior("Gamma", mu=2, sigma=1),
        "lam": Prior("HalfNormal", sigma=1),
    }


class HillSaturation(SaturationTransformation):
    """Wrapper around Hill saturation function.

    For more information, see :func:`pymc_marketing.mmm.transformers.hill_function`.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from pymc_marketing.mmm import HillSaturation

        rng = np.random.default_rng(0)

        adstock = HillSaturation()
        prior = adstock.sample_prior(random_seed=rng)
        curve = adstock.sample_curve(prior)
        adstock.plot_curve(curve, random_seed=rng)
        plt.show()

    """

    lookup_name = "hill"

    def function(self, x, slope, kappa, beta):
        """Hill saturation function."""
        return beta * hill_function(x, slope, kappa)

    default_priors = {
        "slope": Prior("HalfNormal", sigma=1.5),
        "kappa": Prior("HalfNormal", sigma=1.5),
        "beta": Prior("HalfNormal", sigma=1.5),
    }


class HillSaturationSigmoid(SaturationTransformation):
    """Wrapper around Hill saturation sigmoid function.

    For more information, see :func:`pymc_marketing.mmm.transformers.hill_saturation_sigmoid`.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from pymc_marketing.mmm import HillSaturationSigmoid

        rng = np.random.default_rng(0)

        adstock = HillSaturationSigmoid()
        prior = adstock.sample_prior(random_seed=rng)
        curve = adstock.sample_curve(prior)
        adstock.plot_curve(curve, random_seed=rng)
        plt.show()

    """

    lookup_name = "hill_sigmoid"

    function = hill_saturation_sigmoid

    default_priors = {
        "sigma": Prior("HalfNormal", sigma=1.5),
        "beta": Prior("HalfNormal", sigma=1.5),
        "lam": Prior("HalfNormal", sigma=1.5),
    }


class RootSaturation(SaturationTransformation):
    """Wrapper around Root saturation function.

    For more information, see :func:`pymc_marketing.mmm.transformers.root_saturation`.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from pymc_marketing.mmm import RootSaturation

        rng = np.random.default_rng(0)

        saturation = RootSaturation()
        prior = saturation.sample_prior(random_seed=rng)
        curve = saturation.sample_curve(prior)
        saturation.plot_curve(curve, random_seed=rng)
        plt.show()

    """

    lookup_name = "root"

    def function(self, x, alpha, beta):
        """Root saturation function."""
        return beta * root_saturation(x, alpha)

    default_priors = {
        "alpha": Prior("Beta", alpha=1, beta=2),
        "beta": Prior("Gamma", mu=1, sigma=1),
    }


class NoSaturation(SaturationTransformation):
    """Wrapper around linear saturation function.

    For more information, see :func:`pymc_marketing.mmm.transformers.linear_saturation`.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from pymc_marketing.mmm import NoSaturation

        rng = np.random.default_rng(0)

        saturation = NoSaturation()
        prior = saturation.sample_prior(random_seed=rng)
        curve = saturation.sample_curve(prior)
        saturation.plot_curve(curve, random_seed=rng)
        plt.show()

    """

    lookup_name = "no_saturation"

    def function(self, x, beta):
        """Linear saturation function."""
        return pt.as_tensor_variable(beta * x)

    default_priors = {"beta": Prior("HalfNormal", sigma=1)}


def saturation_from_dict(data: dict) -> SaturationTransformation:
    """Get a saturation function from a dictionary."""
    data = data.copy()
    cls = SATURATION_TRANSFORMATIONS[data.pop("lookup_name")]

    if "priors" in data:
        data["priors"] = {
            key: deserialize(value) for key, value in data["priors"].items()
        }
    return cls(**data)


def _is_saturation(data):
    return "lookup_name" in data and data["lookup_name"] in SATURATION_TRANSFORMATIONS


register_deserialization(_is_saturation, saturation_from_dict)
