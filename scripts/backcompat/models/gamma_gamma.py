from __future__ import annotations

import pandas as pd

from pymc_marketing.clv.models.gamma_gamma import GammaGammaModel

from ..model_definition import ModelDefinition


def _make_args() -> dict[str, pd.DataFrame]:
    df = pd.DataFrame(
        {
            "customer_id": [0, 1, 2],
            "frequency": [3, 1, 4],
            "monetary_value": [50.0, 75.5, 20.0],
        }
    )
    return {"data": df}


def _build_model(data: pd.DataFrame) -> GammaGammaModel:
    return GammaGammaModel(data=data)


def get_model_definition() -> ModelDefinition:
    return ModelDefinition(
        name="gamma_gamma",
        builder_cls=GammaGammaModel,
        builder_fn=_build_model,
        build_args_fn=_make_args,
        fit_args_fn=lambda: {"method": "mcmc"},
        sampler_kwargs={"chains": 1, "tune": 2, "draws": 2},
        fit_seed=42,
    )
