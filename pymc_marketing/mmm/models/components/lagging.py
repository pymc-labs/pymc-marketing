import warnings
from typing import Optional, Type

from pymc_marketing.mmm.models.components.base import Transformation
from pymc_marketing.mmm.transformers import (
    WeibullType,
    delayed_adstock,
    geometric_adstock,
    weibull_adstock,
)


class AdstockTransformation(Transformation):
    """Subclass for all adstock functions."""

    prefix: str = "adstock"

    def __init__(
        self, l_max: int = 10, normalize: bool = False, priors: Optional[dict] = None
    ) -> None:
        self.l_max = l_max
        self.normalize = normalize

        super().__init__(priors)


class GeometricAdstock(AdstockTransformation):
    def function(self, x, alpha):
        return geometric_adstock(x, alpha, self.l_max, self.normalize)

    default_priors = {"alpha": {"dist": "HalfNormal", "kwargs": {"sigma": 1}}}


class DelayedAdstock(AdstockTransformation):
    def function(self, x, alpha, theta):
        return delayed_adstock(x, alpha, theta, self.l_max, self.normalize)

    default_priors = {
        "alpha": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "theta": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
    }


class WeibullAdstock(AdstockTransformation):
    def __init__(
        self,
        l_max: int = 10,
        normalize: bool = False,
        kind=WeibullType.PDF,
        priors: Optional[dict] = None,
    ) -> None:
        self.kind = kind

        super().__init__(l_max, normalize, priors)

    def function(self, x, lam, k):
        return weibull_adstock(
            x,
            lam,
            k,
            self.l_max,
            type=self.kind,
        )

    default_priors = {
        "lam": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "k": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
    }


ADSTOCK_TRANSFORMATIONS: dict[str, Type[AdstockTransformation]] = {  # type: ignore
    "geometric": GeometricAdstock,
    "delayed": DelayedAdstock,
    "weibull": WeibullAdstock,
}


def _get_lagging_function(
    function: str | AdstockTransformation,
    **kwargs,
) -> AdstockTransformation:
    if kwargs:
        warnings.warn(
            "The preferred method of initializing a lagging function is to use the class directly.",
            DeprecationWarning,
            stacklevel=1,
        )

    if isinstance(function, str):
        return ADSTOCK_TRANSFORMATIONS[function](**kwargs)  # type: ignore

    return function
