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
"""Adstock transformations for MMM.

Each of these transformations is a subclass of
:class:`pymc_marketing.mmm.components.adstock.AdstockTransformation`
and defines a function that takes a time series and returns the adstocked
version of it. The parameters of the function are the parameters
of the adstock transformation.

Examples
--------
Create a new adstock transformation:

.. code-block:: python

    from pymc_marketing.mmm import AdstockTransformation
    from pymc_extras.prior import Prior


    from pymc_marketing.serialization import registry


    @registry.register
    class MyAdstock(AdstockTransformation):
        def function(self, x, alpha):
            return x * alpha

        default_priors = {"alpha": Prior("HalfNormal", sigma=1)}

Plot the default priors for an adstock transformation:

.. code-block:: python

    from pymc_marketing.mmm import GeometricAdstock

    import matplotlib.pyplot as plt

    adstock = GeometricAdstock(l_max=15)
    prior = adstock.sample_prior()
    curve = adstock.sample_curve(prior)
    adstock.plot_curve(curve)
    plt.show()

"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import xarray as xr
from pydantic import Field, validate_call
from pymc_extras.deserialize import deserialize
from pymc_extras.prior import Prior
from pytensor.xtensor import as_xtensor

from pymc_marketing.mmm.components.base import (
    SupportedPrior,
    Transformation,
)
from pymc_marketing.mmm.transformers import (
    ConvMode,
    WeibullType,
    binomial_adstock,
    delayed_adstock,
    geometric_adstock,
    weibull_adstock,
)
from pymc_marketing.serialization import registry


class AdstockTransformation(Transformation):
    """Subclass for all adstock functions.

    In order to use a custom saturation function, inherit from this class and define:

    - `function`: a function that takes x to adstock x, along a given `dim`
    - `default_priors`: dictionary with priors for every parameter in function

    Consider the predefined subclasses as examples.

    """

    prefix: str = "adstock"

    @validate_call
    def __init__(
        self,
        l_max: int = Field(
            ..., gt=0, description="Maximum lag for the adstock transformation."
        ),
        normalize: bool = Field(
            True, description="Whether to normalize the adstock values."
        ),
        mode: ConvMode = Field(ConvMode.After, description="Convolution mode."),
        priors: dict[str, SupportedPrior] | None = Field(
            default=None, description="Priors for the parameters."
        ),
        prefix: str | None = Field(None, description="Prefix for the parameters."),
    ) -> None:
        self.l_max = l_max
        self.normalize = normalize
        self.mode = mode

        super().__init__(priors=priors, prefix=prefix)

    def __repr__(self) -> str:
        """Representation of the adstock transformation."""
        return (
            f"{self.__class__.__name__}("
            f"prefix={self.prefix!r}, "
            f"l_max={self.l_max}, "
            f"normalize={self.normalize}, "
            f"mode={self.mode.name!r}, "
            f"priors={self.function_priors}"
            ")"
        )

    def to_dict(self) -> dict:
        """Convert the adstock transformation to a dictionary."""
        data = super().to_dict()

        data["l_max"] = self.l_max
        data["normalize"] = self.normalize
        data["mode"] = self.mode.name

        return data

    @classmethod
    def from_dict(cls, data: dict) -> AdstockTransformation:
        """Reconstruct an adstock transformation from a dict."""
        data = data.copy()
        data.pop("__type__", None)
        data.pop(
            "lookup_name", None
        )  # TODO(1.0): Remove once Legacy MMM is removed (#2430)

        if "priors" in data:
            data["priors"] = {k: deserialize(v) for k, v in data["priors"].items()}

        return cls(**data)

    def sample_curve(
        self,
        parameters: xr.Dataset,
        amount: float = 1.0,
        **sample_prior_predictive_kwargs: Any,
    ) -> xr.DataArray:
        """Sample the adstock transformation given parameters.

        Parameters
        ----------
        parameters : xr.Dataset
            Dataset with parameter values.
        amount : float, optional
            Amount to apply the adstock transformation to, by default 1.0.
        sample_prior_predictive_kwargs : Any
            Pass kwargs to pm.sample_prior_predictive

        Returns
        -------
        xr.DataArray
            Adstocked version of the amount.

        """
        time_since = np.arange(0, self.l_max)
        coords = {
            "time since exposure": time_since,
        }
        x = np.zeros(self.l_max)
        x[0] = amount

        return self._sample_curve(
            var_name="adstock",
            parameters=parameters,
            x=x,
            coords=coords,
            **sample_prior_predictive_kwargs,
        )


@registry.register
class BinomialAdstock(AdstockTransformation):
    """Wrapper around the binomial adstock function.

    For more information, see :func:`pymc_marketing.mmm.transformers.binomial_adstock`.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from pymc_marketing.mmm import BinomialAdstock

        rng = np.random.default_rng(0)

        adstock = BinomialAdstock(l_max=10)
        prior = adstock.sample_prior(random_seed=rng)
        curve = adstock.sample_curve(prior)
        adstock.plot_curve(curve, random_seed=rng)
        plt.show()

    """

    def function(self, x, alpha, *, dim: str):
        """Binomial adstock function."""
        return binomial_adstock(
            x,
            alpha=alpha,
            l_max=self.l_max,
            normalize=self.normalize,
            mode=self.mode,
            dim=dim,
        )

    default_priors = {"alpha": Prior("Beta", alpha=1, beta=3)}


@registry.register
class GeometricAdstock(AdstockTransformation):
    """Wrapper around geometric adstock function.

    For more information, see :func:`pymc_marketing.mmm.transformers.geometric_adstock`.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from pymc_marketing.mmm import GeometricAdstock

        rng = np.random.default_rng(0)

        adstock = GeometricAdstock(l_max=10)
        prior = adstock.sample_prior(random_seed=rng)
        curve = adstock.sample_curve(prior)
        adstock.plot_curve(curve, random_seed=rng)
        plt.show()

    """

    def function(self, x, alpha, *, dim: str):
        """Geometric adstock function."""
        return geometric_adstock(
            x,
            alpha=alpha,
            l_max=self.l_max,
            normalize=self.normalize,
            mode=self.mode,
            dim=dim,
        )

    default_priors = {"alpha": Prior("Beta", alpha=1, beta=3)}


@registry.register
class DelayedAdstock(AdstockTransformation):
    """Wrapper around delayed adstock function.

    For more information, see :func:`pymc_marketing.mmm.transformers.delayed_adstock`.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from pymc_marketing.mmm import DelayedAdstock

        rng = np.random.default_rng(0)

        adstock = DelayedAdstock(l_max=10)
        prior = adstock.sample_prior(random_seed=rng)
        curve = adstock.sample_curve(prior)
        adstock.plot_curve(curve, random_seed=rng)
        plt.show()

    """

    def function(self, x, alpha, theta, *, dim: str):
        """Delayed adstock function."""
        return delayed_adstock(
            x,
            alpha=alpha,
            theta=theta,
            l_max=self.l_max,
            normalize=self.normalize,
            mode=self.mode,
            dim=dim,
        )

    default_priors = {
        "alpha": Prior("Beta", alpha=1, beta=3),
        "theta": Prior("HalfNormal", sigma=1),
    }


@registry.register
class WeibullPDFAdstock(AdstockTransformation):
    """Wrapper around weibull adstock with PDF function.

    For more information, see :func:`pymc_marketing.mmm.transformers.weibull_adstock`.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from pymc_marketing.mmm import WeibullPDFAdstock

        rng = np.random.default_rng(0)

        adstock = WeibullPDFAdstock(l_max=10)
        prior = adstock.sample_prior(random_seed=rng)
        curve = adstock.sample_curve(prior)
        adstock.plot_curve(curve, random_seed=rng)
        plt.show()

    """

    def function(self, x, lam, k, *, dim: str):
        """Weibull adstock function."""
        return weibull_adstock(
            x=x,
            lam=lam,
            k=k,
            l_max=self.l_max,
            mode=self.mode,
            type=WeibullType.PDF,
            normalize=self.normalize,
            dim=dim,
        )

    default_priors = {
        "lam": Prior("Gamma", mu=2, sigma=1),
        "k": Prior("Gamma", mu=3, sigma=1),
    }


@registry.register
class WeibullCDFAdstock(AdstockTransformation):
    """Wrapper around weibull adstock with CDF function.

    For more information, see :func:`pymc_marketing.mmm.transformers.weibull_adstock`.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from pymc_marketing.mmm import WeibullCDFAdstock

        rng = np.random.default_rng(0)

        adstock = WeibullCDFAdstock(l_max=10)
        prior = adstock.sample_prior(random_seed=rng)
        curve = adstock.sample_curve(prior)
        adstock.plot_curve(curve, random_seed=rng)
        plt.show()

    """

    def function(self, x, lam, k, *, dim: str):
        """Weibull adstock function."""
        return weibull_adstock(
            x=x,
            lam=lam,
            k=k,
            l_max=self.l_max,
            mode=self.mode,
            type=WeibullType.CDF,
            normalize=self.normalize,
            dim=dim,
        )

    default_priors = {
        "lam": Prior("Gamma", mu=2, sigma=2.5),
        "k": Prior("Gamma", mu=2, sigma=2.5),
    }


@registry.register
class NoAdstock(AdstockTransformation):
    """Wrapper around no adstock transformation."""

    def function(self, x, *, dim: str | None = None):
        """No adstock function."""
        x = as_xtensor(x)
        return x

    default_priors = {}

    def update_priors(self, priors):
        """Update priors for the no adstock transformation."""
        return


# TODO(1.0): Remove this dict once Legacy MMM is removed (see #2430)
ADSTOCK_TRANSFORMATIONS: dict[str, type[AdstockTransformation]] = {
    "geometric": GeometricAdstock,
    "delayed": DelayedAdstock,
    "weibull_cdf": WeibullCDFAdstock,
    "weibull_pdf": WeibullPDFAdstock,
    "binomial": BinomialAdstock,
    "no_adstock": NoAdstock,
}


def adstock_from_dict(data: dict) -> AdstockTransformation:
    """Create an adstock transformation from a dictionary.

    .. deprecated:: 0.18.2
        `adstock_from_dict` is deprecated and will be removed in 0.20.0.
        Use ``from pymc_marketing.serialization import registry; registry.deserialize(data)`` instead.
    """
    warnings.warn(
        "adstock_from_dict is deprecated and will be removed in 0.20.0. "
        "Use `from pymc_marketing.serialization import registry; "
        "registry.deserialize(data)` instead.",
        FutureWarning,
        stacklevel=2,
    )
    data = data.copy()
    type_key = data.pop("__type__", None)
    lookup_name = data.pop("lookup_name", None)

    if lookup_name:
        cls = ADSTOCK_TRANSFORMATIONS[lookup_name]
    elif type_key:
        return registry.deserialize({**data, "__type__": type_key})
    else:
        raise ValueError(
            "Cannot deserialize adstock: missing both 'lookup_name' and '__type__'"
        )

    if "priors" in data:
        data["priors"] = {k: deserialize(v) for k, v in data["priors"].items()}

    return cls(**data)
