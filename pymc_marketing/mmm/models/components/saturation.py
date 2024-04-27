from pymc_marketing.mmm.models.components.base import Transformation
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

        def infinite_returns(x, b):
            return b * x


        class InfiniteReturns(SaturationTransformation):
            function = infinite_returns

            default_priors = {"b": {"dist": "HalfNormal", "kwargs": {"sigma": 1}}}

    """

    prefix: str = "saturation"


class LogisticSaturation(SaturationTransformation):
    def function(self, x, lam, beta):
        return beta * logistic_saturation(x, lam)

    default_priors = {
        "lam": {"dist": "Gamma", "kwargs": {"alpha": 3, "beta": 1}},
        "beta": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
    }


class TanhSaturation(SaturationTransformation):
    def function(self, x, b, c, beta):
        return beta * tanh_saturation(x, b, c)

    default_priors = {
        "b": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "c": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "beta": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
    }


class TanhSaturationBaselined(SaturationTransformation):
    def function(self, x, x0, gain, r, beta):
        return beta * tanh_saturation_baselined(x, x0, gain, r)

    default_priors = {
        "x0": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "gain": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "r": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "beta": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
    }


class MichaelisMentenSaturation(SaturationTransformation):
    function = michaelis_menten

    default_priors = {
        "alpha": {"dist": "Gamma", "kwargs": {"mu": 2, "sigma": 1}},
        "lam": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
    }


class HillSaturation(SaturationTransformation):
    function = hill_saturation

    default_priors = {
        "sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
        "beta": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
        "lam": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
    }


SATURATION_TRANSFORMATIONS: dict[str, type[SaturationTransformation]] = {
    "logistic": LogisticSaturation,
    "tanh": TanhSaturation,
    "hill": HillSaturation,
    "tanh_baselined": TanhSaturationBaselined,
    "michaelis_menten": MichaelisMentenSaturation,
}


def _get_saturation_function(
    function: str | SaturationTransformation,
) -> SaturationTransformation:
    if isinstance(function, str):
        return SATURATION_TRANSFORMATIONS[function]()

    return function
