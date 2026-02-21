from __future__ import annotations

import pandas as pd

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM

from ..model_definition import ModelDefinition


def _make_args() -> dict[str, pd.DataFrame | pd.Series]:
    dates = pd.date_range("2025-01-01", periods=16, freq="W-MON")
    df = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "C1": [100, 120, 90, 110, 105, 115, 98, 102] * 4,
            "C2": [80, 70, 95, 85, 90, 88, 92, 94] * 4,
            "geo": ["A"] * len(dates) + ["B"] * len(dates),
        }
    ).reset_index(drop=True)

    y = pd.Series([230, 260, 220, 240, 245, 255, 235, 238] * 4, name="y")
    return {"df": df, "y": y}


def _build_model(df: pd.DataFrame, y: pd.Series) -> MMM:
    adstock = GeometricAdstock(l_max=3)
    saturation = LogisticSaturation()
    mmm = MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        target_column="y",
        dims=("geo",),
        adstock=adstock,
        saturation=saturation,
    )
    mmm.build_model(X=df, y=y)
    return mmm


def get_model_definition() -> ModelDefinition:
    return ModelDefinition(
        name="basic_mmm",
        builder_cls=MMM,
        builder_fn=_build_model,
        build_args_fn=_make_args,
        fit_args_fn=lambda: {},
        fit_data_fn=lambda build_kwargs: {
            "X": build_kwargs["df"],
            "y": build_kwargs["y"],
        },
        sampler_kwargs={"chains": 1, "tune": 2, "draws": 2},
        fit_seed=42,
    )
