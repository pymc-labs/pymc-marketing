from __future__ import annotations

import pandas as pd

from pymc_marketing.customer_choice.mixed_logit import MixedLogit

from ..model_definition import ModelDefinition


def _make_args() -> dict[str, pd.DataFrame | list[str]]:
    choice_df = pd.DataFrame(
        {
            "choice": ["bus", "car", "bus", "train", "car"],
            "bus_price": [2.0, 2.5, 2.0, 2.2, 2.3],
            "bus_time": [45, 50, 45, 48, 52],
            "car_price": [5.0, 4.8, 5.2, 5.1, 4.9],
            "car_time": [30, 28, 32, 29, 31],
            "train_price": [3.5, 3.8, 3.6, 3.7, 3.9],
            "train_time": [35, 38, 36, 37, 39],
            "income": [50000, 60000, 55000, 70000, 65000],
        }
    )
    utility_equations = [
        "bus ~ bus_price + bus_time | income | bus_price",
        "car ~ car_price + car_time | income | car_price",
        "train ~ train_price + train_time | income | train_price",
    ]
    depvar = "choice"
    covariates = ["price", "time"]
    return {
        "choice_df": choice_df,
        "utility_equations": utility_equations,
        "depvar": depvar,
        "covariates": covariates,
    }


def _build_model(
    choice_df: pd.DataFrame,
    utility_equations: list[str],
    depvar: str,
    covariates: list[str],
) -> MixedLogit:
    return MixedLogit(choice_df, utility_equations, depvar, covariates)


def get_model_definition() -> ModelDefinition:
    return ModelDefinition(
        name="mixed_logit",
        builder_cls=MixedLogit,
        builder_fn=_build_model,
        build_args_fn=_make_args,
        fit_args_fn=lambda: {},
        fit_data_fn=lambda build_kwargs: {
            "choice_df": build_kwargs["choice_df"],
            "utility_equations": build_kwargs["utility_equations"],
        },
        sampler_kwargs={"chains": 1, "tune": 2, "draws": 2},
        fit_seed=42,
    )
