from functools import wraps
from typing import Callable, Union

import pymc as pm

from pymc_marketing.mmm.transformers import (
    WeibullType,
    geometric_adstock,
    logistic_saturation,
    scale_preserving_logistic_saturation,
    tanh_saturation,
    tanh_saturation_baselined,
    weibull_adstock,
)


def wrap_as_deterministic(name, dims):
    """Create a decorator that wraps a function as a PyMC Deterministic."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return pm.Deterministic(name=name, var=func(*args, **kwargs), dims=dims)

        return wrapper

    return decorator


def create_geometric_adstock_handler(l_max: int, normalize: bool, axis: int):
    def handler(x, params):
        return geometric_adstock(
            x,
            alpha=params["alpha"],
            l_max=l_max,
            normalize=normalize,
            axis=axis,
        )

    return handler


def logistic_saturation_handler(x, params):
    return logistic_saturation(
        x,
        lam=params["lam"],
    )


def create_weibull_adstock_handler(
    l_max: int, axis: int, type: Union[WeibullType, str]
):
    def handler(x, params):
        return weibull_adstock(
            x,
            lam=params["lam"],
            k=params["k"],
            l_max=l_max,
            axis=axis,
            type=type,
        )

    return handler


def scale_preserving_logistic_saturation_handler(x, params):
    return scale_preserving_logistic_saturation(
        x,
        m=params["m"],
    )


def tanh_saturation_handler(x, params):
    return tanh_saturation(
        x,
        b=params["b"],
        c=params["c"],
    )


def create_tanh_saturation_baselined_handler(x0):
    def handler(x, params):
        return tanh_saturation_baselined(
            x,
            x0=x0,
            gain0=params["gain0"],
            r=params["r"],
        )

    return handler


def create_multiply_handler(name: str):
    def multiply(x, params):
        return x * params[name]

    return multiply


def apply_forward_pass(x, steps, params):
    """Apply a sequence of transformations to variable x."""
    for step in steps:
        x = step(x, params)

    return x


def create_forward_pass(forward_pass_steps, params) -> Callable:
    def forward_pass(x):
        return apply_forward_pass(x, forward_pass_steps, params)

    return forward_pass
