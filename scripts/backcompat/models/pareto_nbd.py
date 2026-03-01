from __future__ import annotations

import pandas as pd

from pymc_marketing.clv.models.pareto_nbd import ParetoNBDModel
from pymc_marketing.clv.utils import rfm_summary

from ..model_definition import ModelDefinition


def _make_args() -> dict[str, pd.DataFrame]:
    transactions = pd.DataFrame(
        {
            "customer_id": [0, 0, 1, 1, 2, 2, 2],
            "date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-15",
                    "2023-01-01",
                    "2023-01-10",
                    "2023-01-20",
                    "2023-01-05",
                    "2023-01-01",
                ]
            ),
        }
    )
    df = rfm_summary(transactions, "customer_id", "date")
    return {"data": df}


def _build_model(data: pd.DataFrame) -> ParetoNBDModel:
    return ParetoNBDModel(data=data)


def get_model_definition() -> ModelDefinition:
    return ModelDefinition(
        name="pareto_nbd",
        builder_cls=ParetoNBDModel,
        builder_fn=_build_model,
        build_args_fn=_make_args,
        fit_args_fn=lambda: {"method": "mcmc"},
        sampler_kwargs={"chains": 1, "tune": 2, "draws": 2},
        fit_seed=42,
    )
