#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Common models and labeled data for contribution-factor contract tests."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation


def make_model(dims=(), **kwargs):
    return MMM(
        date_column="date",
        channel_columns=["search", "tiktok"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        dims=dims,
        **kwargs,
    )


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5),
            "search": [1.0, 3.0, 2.0, 4.0, 5.0],
            "tiktok": [2.0, 4.0, 3.0, 5.0, 6.0],
        }
    )
    return X, pd.Series([10.0, 20.0, np.nan, 30.0, 40.0])


def evaluate(model, expressions, point=None):
    expressions = [v.values if hasattr(v, "values") else v for v in expressions]
    fn = model.compile_fn(
        model.replace_rvs_by_values(expressions),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    return fn(model.initial_point() if point is None else point)


@pytest.fixture
def labeled_panel():
    dates = pd.date_range("2025-01-01", periods=5)
    coords = {
        "date": dates,
        "region": ["A", "B"],
        "product": ["p", "q"],
        "channel": ["tiktok", "search"],
    }
    ds = xr.Dataset(
        {
            "_channel": xr.DataArray(
                np.arange(40).reshape(2, 2, 5, 2) + 1.0,
                dims=("channel", "product", "date", "region"),
                coords=coords,
            ),
            "_target": xr.DataArray(
                np.arange(20).reshape(2, 5, 2) + 20.0,
                dims=("region", "date", "product"),
                coords={d: coords[d] for d in ["region", "date", "product"]},
            ),
        }
    )
    ds["_target"].loc[{"region": "B", "product": "q", "date": dates[2]}] = np.nan
    channels = ds["_channel"].to_series().unstack("channel").reset_index()
    targets = ds["_target"].to_series().rename("y").reset_index()
    frame = channels.merge(targets, on=["date", "region", "product"])
    return ds, frame.drop(columns="y"), frame["y"]
