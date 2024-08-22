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
    from pymc_marketing.prior import Prior

    class MyAdstock(AdstockTransformation):
        lookup_name: str = "my_adstock"

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

import warnings

import numpy as np
import xarray as xr
from pydantic import Field, InstanceOf, validate_call

from pymc_marketing.mmm.components.base import Transformation
from pymc_marketing.mmm.transformers import (
    ConvMode,
    WeibullType,
    delayed_adstock,
    geometric_adstock,
    weibull_adstock,
)
from pymc_marketing.prior import Prior


class AdstockTransformation(Transformation):
    """Subclass for all adstock functions.

    In order to use a custom saturation function, inherit from this class and define:

    - `function`: a function that takes x to adstock x
    - `default_priors`: dictionary with priors for every parameter in function

    Consider the predefined subclasses as examples.

    """

    prefix: str = "adstock"
    lookup_name: str

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
        priors: dict[str, InstanceOf[Prior]] | None = Field(
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

    def sample_curve(
        self,
        parameters: xr.Dataset,
        amount: float = 1.0,
    ) -> xr.DataArray:
        """Sample the adstock transformation given parameters.

        Parameters
        ----------
        parameters : xr.Dataset
            Dataset with parameter values.
        amount : float, optional
            Amount to apply the adstock transformation to, by default 1.0.

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
        )


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
        adstock.plot_curve(curve, sample_kwargs={"rng": rng})
        plt.show()

    """

    lookup_name = "geometric"

    def function(self, x, alpha):
        """Geometric adstock function."""
        return geometric_adstock(
            x, alpha=alpha, l_max=self.l_max, normalize=self.normalize, mode=self.mode
        )

    default_priors = {"alpha": Prior("Beta", alpha=1, beta=3)}


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
        adstock.plot_curve(curve, sample_kwargs={"rng": rng})
        plt.show()

    """

    lookup_name = "delayed"

    def function(self, x, alpha, theta):
        """Delayed adstock function."""
        return delayed_adstock(
            x,
            alpha=alpha,
            theta=theta,
            l_max=self.l_max,
            normalize=self.normalize,
            mode=self.mode,
        )

    default_priors = {
        "alpha": Prior("Beta", alpha=1, beta=3),
        "theta": Prior("HalfNormal", sigma=1),
    }


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
        adstock.plot_curve(curve, sample_kwargs={"rng": rng})
        plt.show()

    """

    lookup_name = "weibull_pdf"

    def function(self, x, lam, k):
        """Weibull adstock function."""
        return weibull_adstock(
            x=x,
            lam=lam,
            k=k,
            l_max=self.l_max,
            mode=self.mode,
            type=WeibullType.PDF,
            normalize=self.normalize,
        )

    default_priors = {
        "lam": Prior("Gamma", mu=2, sigma=1),
        "k": Prior("Gamma", mu=3, sigma=1),
    }


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
        adstock.plot_curve(curve, sample_kwargs={"rng": rng})
        plt.show()

    """

    lookup_name = "weibull_cdf"

    def function(self, x, lam, k):
        """Weibull adstock function."""
        return weibull_adstock(
            x=x,
            lam=lam,
            k=k,
            l_max=self.l_max,
            mode=self.mode,
            type=WeibullType.CDF,
            normalize=self.normalize,
        )

    default_priors = {
        "lam": Prior("Gamma", mu=2, sigma=2.5),
        "k": Prior("Gamma", mu=2, sigma=2.5),
    }


class WeibullAdstock(AdstockTransformation):
    """Wrapper around weibull adstock function.

    For more information, see :func:`pymc_marketing.mmm.transformers.weibull_adstock`.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from pymc_marketing.mmm import WeibullAdstock

        rng = np.random.default_rng(0)

        adstock = WeibullAdstock(l_max=10, kind="CDF")
        prior = adstock.sample_prior(random_seed=rng)
        curve = adstock.sample_curve(prior)
        adstock.plot_curve(curve, sample_kwargs={"rng": rng})
        plt.show()

    """

    lookup_name = "weibull"

    def __init__(
        self,
        l_max: int,
        normalize: bool = True,
        kind=WeibullType.PDF,
        mode: ConvMode = ConvMode.After,
        priors: dict | None = None,
        prefix: str | None = None,
    ) -> None:
        self.kind = kind

        super().__init__(
            l_max=l_max, normalize=normalize, mode=mode, priors=priors, prefix=prefix
        )

        msg = (
            f"Use the Weibull{kind}Adstock class instead for better default priors. "
            "This class will deprecate in 0.9.0."
        )
        warnings.warn(
            msg,
            UserWarning,
            stacklevel=1,
        )

    def function(self, x, lam, k):
        """Weibull adstock function."""
        return weibull_adstock(
            x=x,
            lam=lam,
            k=k,
            l_max=self.l_max,
            mode=self.mode,
            type=self.kind,
            normalize=self.normalize,
        )

    default_priors = {
        "lam": Prior("HalfNormal", sigma=1),
        "k": Prior("HalfNormal", sigma=1),
    }


ADSTOCK_TRANSFORMATIONS: dict[str, type[AdstockTransformation]] = {
    cls.lookup_name: cls  # type: ignore
    for cls in [
        GeometricAdstock,
        DelayedAdstock,
        WeibullAdstock,
        WeibullPDFAdstock,
        WeibullCDFAdstock,
    ]
}


def register_adstock_transformation(cls: type[AdstockTransformation]) -> None:
    """Register a new adstock transformation."""
    ADSTOCK_TRANSFORMATIONS[cls.lookup_name] = cls


def adstock_from_dict(data: dict) -> AdstockTransformation:
    """Create an adstock transformation from a dictionary."""
    data = data.copy()
    lookup_name = data.pop("lookup_name")
    cls = ADSTOCK_TRANSFORMATIONS[lookup_name]

    if "priors" in data:
        data["priors"] = {k: Prior.from_json(v) for k, v in data["priors"].items()}
    return cls(**data)


def _get_adstock_function(
    function: str | AdstockTransformation,
    **kwargs,
) -> AdstockTransformation:
    """Get an adstock function.

    Helper for use in the MMM to get an adstock function from the if registered.
    """
    if isinstance(function, AdstockTransformation):
        return function

    elif isinstance(function, str):
        if function not in ADSTOCK_TRANSFORMATIONS:
            raise ValueError(
                f"Unknown adstock function: {function}. Choose from {list(ADSTOCK_TRANSFORMATIONS.keys())}"
            )

        if kwargs:
            msg = (
                "The preferred method of initializing a "
                "lagging function is to use the class directly. "
                "String support will deprecate in 0.9.0."
            )
            warnings.warn(
                msg,
                DeprecationWarning,
                stacklevel=1,
            )

        return ADSTOCK_TRANSFORMATIONS[function](**kwargs)

    else:
        raise ValueError(
            f"Unknown adstock function: {function}. Choose from {list(ADSTOCK_TRANSFORMATIONS.keys())}"
        )
