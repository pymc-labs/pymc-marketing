from __future__ import annotations

import pandas as pd

from pymc_marketing.clv.models.beta_geo import BetaGeoModel

from ..model_definition import ModelDefinition


def _make_args() -> dict[str, pd.DataFrame]:
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "frequency": [1, 2, 3],
            "recency": [0, 1, 2],
            "T": [1, 2, 3],
        }
    )
    return {"data": df}


def _build_model(data: pd.DataFrame) -> BetaGeoModel:
    return BetaGeoModel(data=data)


def get_model_definition() -> ModelDefinition:
    return ModelDefinition(
        name="beta_geo",
        builder_cls=BetaGeoModel,
        builder_fn=_build_model,
        build_args_fn=_make_args,
        fit_args_fn=lambda: {"method": "mcmc"},
        sampler_kwargs={"chains": 1, "tune": 2, "draws": 2},
        fit_seed=42,
    )
