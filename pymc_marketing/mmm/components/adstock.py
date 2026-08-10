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


    from pymc_marketing.serialization import serialization


    @serialization.register
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

from inspect import signature
from typing import Any, Literal

import numpy as np
import xarray as xr
from pydantic import Field, validate_call
from pymc_extras.deserialize import deserialize
from pymc_extras.prior import Prior
from pytensor.xtensor import as_xtensor

from pymc_marketing.mmm.components.base import (
    ParameterPriorException,
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
from pymc_marketing.serialization import serialization


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


@serialization.register
class BinomialAdstock(AdstockTransformation):
    """Wrapper around the binomial adstock function.

    Calls :func:`pymc_marketing.mmm.transformers.binomial_adstock` with the wrapper's
    ``l_max``, ``normalize`` and ``mode`` settings.

    Parameters
    ----------
    alpha : tensor
        Retention rate of the ad effect; must be between 0 and 1. Default prior:
        ``Prior("Beta", alpha=1, beta=3)``.

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


@serialization.register
class GeometricAdstock(AdstockTransformation):
    r"""Wrapper around geometric adstock function.

    Calls :func:`pymc_marketing.mmm.transformers.geometric_adstock` with the wrapper's
    ``l_max``, ``normalize`` and ``mode`` settings.

    The decay can be parametrised either by the retention rate ``alpha`` (the
    default) or by the half-life of the ad effect. Since the weight at lag
    :math:`t` is :math:`\alpha^{t}`, a half-life :math:`h` corresponds to

    .. math::

        \alpha = 0.5^{1 / h}

    which maps any positive half-life into :math:`(0, 1)` and holds exactly for
    every ``l_max`` and either setting of ``normalize``. Under the half-life
    parametrisation the trace will contain ``adstock_halflife`` instead of
    ``adstock_alpha``.

    The two defaults are matched, implying a median ``alpha`` of 0.207 against
    0.206. The priors imply a median half-life of 0.44 periods. A custom
    half-life prior should keep its mass away from zero, where the likelihood
    goes numerically flat. The
    :ref:`adstock functions guide <adstock_functions_guide>` covers both points.

    Parameters
    ----------
    alpha : tensor
        Retention rate of the ad effect; must be between 0 and 1. Default prior:
        ``Prior("Beta", alpha=1, beta=3)``. Only used when
        ``parametrization="alpha"``.
    halflife : tensor
        Number of time periods after which the ad effect has decayed by half;
        must be positive. Default prior:
        ``Prior("InverseGamma", alpha=2.6, beta=1)``. Only used when
        ``parametrization="halflife"``.
    parametrization : str
        Either ``"alpha"`` or ``"halflife"``. When left unset it is inferred
        from the priors, defaulting to ``"alpha"``. Passing a prior for the
        parameter of the other parametrisation raises a ``ValueError``.

    Examples
    --------
    Parametrise the decay by its half-life instead of the retention rate:

    .. code-block:: python

        from pymc_extras.prior import Prior
        from pymc_marketing.mmm import GeometricAdstock

        adstock = GeometricAdstock(
            l_max=10,
            priors={"halflife": Prior("InverseGamma", alpha=4, beta=2)},
        )

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
        parametrization: Literal["alpha", "halflife"] | None = Field(
            None, description="Whether to parametrize the decay by alpha or half-life."
        ),
    ) -> None:
        prior_names = set(priors or {})

        if parametrization is None:
            parametrization = "halflife" if "halflife" in prior_names else "alpha"

        if parametrization == "halflife":
            # Shadows the class attribute so that variable_mapping, model_config
            # and the function_priors setter all key off halflife.
            self.default_priors = self.halflife_priors

        self.parametrization = parametrization

        # Priors for the inactive parametrization are rejected by the
        # function_priors setter, which this call assigns through.
        super().__init__(
            l_max=l_max,
            normalize=normalize,
            mode=mode,
            priors=priors,
            prefix=prefix,
        )

    def function(self, x, alpha=None, halflife=None, *, dim: str):
        """Geometric adstock function."""
        if (alpha is None) == (halflife is None):
            raise ValueError("Provide exactly one of 'alpha' and 'halflife'.")

        if halflife is not None:
            alpha = 0.5 ** (1.0 / halflife)

        return geometric_adstock(
            x,
            alpha=alpha,
            l_max=self.l_max,
            normalize=self.normalize,
            mode=self.mode,
            dim=dim,
        )

    @AdstockTransformation.function_priors.setter  # type: ignore[attr-defined]
    def function_priors(self, priors: dict[str, Any | Prior] | None) -> None:
        """Reject priors for the inactive parametrization before storing them.

        The base setter merges what it is given into ``default_priors``, so a
        prior for the inactive parameter would be kept, ignored when building
        the model, and then serialised alongside the parametrization into
        something that ``from_dict`` refuses to load.
        """
        if conflicting := (
            set(self._alternative_parameters) - {self.parametrization}
        ) & set(priors or {}):
            raise ValueError(
                f"Priors for {conflicting.pop()!r} are not used when"
                f" parametrization={self.parametrization!r}."
                f" 'alpha' and 'halflife' are alternatives, so pass a prior for"
                f" {self.parametrization!r} only."
            )

        AdstockTransformation.function_priors.fset(self, priors)  # type: ignore[attr-defined]

    def _has_defaults_for_all_arguments(self) -> None:
        """Check the priors of the active parametrization.

        ``alpha`` and ``halflife`` are alternatives, so only the one of the
        active parametrization needs a prior. Every other argument of
        ``function`` still does, which keeps the check meaningful for subclasses
        that override it.
        """
        function_signature = signature(self.function)

        # Drop the first one as assumed to be the data, and the dim kwarg
        parameters = set(list(function_signature.parameters.keys())[1:]) - {"dim"}
        parameters_that_need_priors = (
            parameters - set(self._alternative_parameters)
        ) | {self.parametrization}
        parameters_with_priors = set(self.default_priors)

        missing_priors = parameters_that_need_priors - parameters_with_priors
        missing_parameters = parameters_with_priors - parameters_that_need_priors

        if missing_priors or missing_parameters:
            raise ParameterPriorException(missing_priors, missing_parameters)

    def to_dict(self) -> dict:
        """Convert the adstock transformation to a dictionary."""
        data = super().to_dict()

        if self.parametrization != "alpha":
            data["parametrization"] = self.parametrization

        return data

    _alternative_parameters = ("alpha", "halflife")
    default_priors = {"alpha": Prior("Beta", alpha=1, beta=3)}
    # Implies approximately the default alpha prior under alpha = 0.5 ** (1 / h)
    halflife_priors = {"halflife": Prior("InverseGamma", alpha=2.6, beta=1)}


@serialization.register
class DelayedAdstock(AdstockTransformation):
    """Wrapper around delayed adstock function.

    Calls :func:`pymc_marketing.mmm.transformers.delayed_adstock` with the wrapper's
    ``l_max``, ``normalize`` and ``mode`` settings.

    Parameters
    ----------
    alpha : tensor
        Retention rate of the ad effect; must be between 0 and 1. Default prior:
        ``Prior("Beta", alpha=1, beta=3)``.
    theta : tensor
        Delay of the peak effect; must be between 0 and ``l_max - 1``. Default prior:
        ``Prior("HalfNormal", sigma=1)``.

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


@serialization.register
class WeibullPDFAdstock(AdstockTransformation):
    """Wrapper around weibull adstock with PDF function.

    Calls :func:`pymc_marketing.mmm.transformers.weibull_adstock` with
    ``type=WeibullType.PDF`` and the wrapper's ``l_max``, ``normalize`` and ``mode``
    settings.

    Parameters
    ----------
    lam : tensor
        Scale parameter of the Weibull distribution; must be positive. Default prior:
        ``Prior("Gamma", mu=2, sigma=1)``.
    k : tensor
        Shape parameter of the Weibull distribution; must be positive. Default prior:
        ``Prior("Gamma", mu=3, sigma=1)``.

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


@serialization.register
class WeibullCDFAdstock(AdstockTransformation):
    """Wrapper around weibull adstock with CDF function.

    Calls :func:`pymc_marketing.mmm.transformers.weibull_adstock` with
    ``type=WeibullType.CDF`` and the wrapper's ``l_max``, ``normalize`` and ``mode``
    settings.

    Parameters
    ----------
    lam : tensor
        Scale parameter of the Weibull distribution; must be positive. Default prior:
        ``Prior("Gamma", mu=2, sigma=2.5)``.
    k : tensor
        Shape parameter of the Weibull distribution; must be positive. Default prior:
        ``Prior("Gamma", mu=2, sigma=2.5)``.

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


@serialization.register
class NoAdstock(AdstockTransformation):
    """Wrapper around no adstock transformation.

    Identity transformation that returns the input unchanged. Useful as a no-op
    placeholder when carryover is not modelled. Takes no priors.
    """

    def function(self, x, *, dim: str | None = None):
        """No adstock function."""
        x = as_xtensor(x)
        return x

    default_priors = {}

    def update_priors(self, priors):
        """Update priors for the no adstock transformation."""
        return
