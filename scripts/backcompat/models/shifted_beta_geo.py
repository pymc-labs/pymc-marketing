from __future__ import annotations

import pandas as pd

from pymc_marketing.clv.models.shifted_beta_geo import ShiftedBetaGeoModel

from ..model_definition import ModelDefinition


def _make_args() -> dict[str, pd.DataFrame]:
    df = pd.DataFrame(
        {
            "customer_id": [0, 1, 2, 3],
            "recency": [1, 2, 3, 4],
            "T": [5, 5, 10, 10],  # T must be homogeneous within each cohort
            "cohort": ["A", "A", "B", "B"],
        }
    )
    return {"data": df}


def _build_model(data: pd.DataFrame) -> ShiftedBetaGeoModel:
    return ShiftedBetaGeoModel(data=data)


def get_model_definition() -> ModelDefinition:
    return ModelDefinition(
        name="shifted_beta_geo",
        builder_cls=ShiftedBetaGeoModel,
        builder_fn=_build_model,
        build_args_fn=_make_args,
        fit_args_fn=lambda: {"method": "mcmc"},
        sampler_kwargs={"chains": 1, "tune": 2, "draws": 2},
        fit_seed=42,
    )
