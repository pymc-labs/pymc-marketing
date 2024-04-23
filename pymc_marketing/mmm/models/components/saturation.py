from typing import Dict, Type

from pymc_marketing.mmm.models.components.base import Transformation
from pymc_marketing.mmm.transformers import (
    hill_saturation,
    logistic_saturation,
    michaelis_menten,
    tanh_saturation,
    tanh_saturation_baselined,
)


class SaturationTransformation(Transformation):
    prefix: str = "saturation"


class LogisticSaturation(SaturationTransformation):
    def function(self, x, lam, beta):
        return beta * logistic_saturation(x, lam)

    default_priors = {
        "lam": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "beta": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
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
        "alpha": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "lam": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
    }


class HillSaturation(SaturationTransformation):
    function = hill_saturation

    default_priors = {
        "sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "beta": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
        "lam": {"dist": "HalfNormal", "kwargs": {"sigma": 1}},
    }


SATURATION_TRANSFORMATIONS: Dict[str, Type[SaturationTransformation]] = {
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
