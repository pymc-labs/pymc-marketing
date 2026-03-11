from __future__ import annotations

import pandas as pd

from pymc_marketing.clv.models.modified_beta_geo import ModifiedBetaGeoModel

from ..model_definition import ModelDefinition


def _make_args() -> dict[str, pd.DataFrame]:
    df = pd.DataFrame(
        {
            "customer_id": [0, 1, 2],
            "frequency": [2, 1, 0],
            "recency": [0, 1, 0],
            "T": [5, 5, 5],
        }
    )
    return {"data": df}


def _build_model(data: pd.DataFrame) -> ModifiedBetaGeoModel:
    return ModifiedBetaGeoModel(data=data)


def get_model_definition() -> ModelDefinition:
    return ModelDefinition(
        name="modified_beta_geo",
        builder_cls=ModifiedBetaGeoModel,
        builder_fn=_build_model,
        build_args_fn=_make_args,
        fit_args_fn=lambda: {"method": "mcmc"},
        sampler_kwargs={"chains": 1, "tune": 2, "draws": 2},
        fit_seed=42,
    )
