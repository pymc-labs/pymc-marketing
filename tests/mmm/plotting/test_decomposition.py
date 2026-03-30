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
from __future__ import annotations

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from pymc_marketing.data.idata import MMMIDataWrapper
from pymc_marketing.mmm.plotting.decomposition import DecompositionPlots

matplotlib.use("Agg")

SEED = sum(map(ord, "DecompositionPlots tests"))


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def simple_idata() -> az.InferenceData:
    """Minimal idata with channels + baseline contributions, no extra dims.

    posterior:
      channel_contribution   (chain, draw, date, channel)
      intercept_contribution (chain, draw, date)
    constant_data:
      target_data  (date,)
      target_scale scalar
    """
    rng = np.random.default_rng(SEED)
    n_chain, n_draw, n_date = 2, 40, 20
    channels = ["tv", "radio", "social"]
    dates = np.arange(n_date)

    posterior = xr.Dataset(
        {
            "channel_contribution": xr.DataArray(
                rng.uniform(0, 100, size=(n_chain, n_draw, n_date, len(channels))),
                dims=("chain", "draw", "date", "channel"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "date": dates,
                    "channel": channels,
                },
            ),
            "intercept_contribution": xr.DataArray(
                rng.uniform(50, 150, size=(n_chain, n_draw, n_date)),
                dims=("chain", "draw", "date"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "date": dates,
                },
            ),
        }
    )
    const = xr.Dataset(
        {
            "target_data": xr.DataArray(
                rng.normal(500, 50, size=(n_date,)),
                dims=("date",),
                coords={"date": dates},
            ),
            "target_scale": xr.DataArray(1000.0),
        }
    )
    return az.InferenceData(posterior=posterior, constant_data=const)


@pytest.fixture(scope="module")
def panel_idata() -> az.InferenceData:
    """idata with geo extra dim — (chain, draw, date, channel, geo) for channels.

    posterior:
      channel_contribution   (chain, draw, date, channel, geo)
      intercept_contribution (chain, draw, date, geo)
    constant_data:
      target_data  (date, geo)
      target_scale scalar
    """
    rng = np.random.default_rng(SEED + 1)
    n_chain, n_draw, n_date = 2, 30, 15
    channels = ["tv", "radio"]
    geos = ["CA", "NY"]
    dates = np.arange(n_date)

    posterior = xr.Dataset(
        {
            "channel_contribution": xr.DataArray(
                rng.uniform(
                    0, 100, size=(n_chain, n_draw, n_date, len(channels), len(geos))
                ),
                dims=("chain", "draw", "date", "channel", "geo"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "date": dates,
                    "channel": channels,
                    "geo": geos,
                },
            ),
            "intercept_contribution": xr.DataArray(
                rng.uniform(50, 150, size=(n_chain, n_draw, n_date, len(geos))),
                dims=("chain", "draw", "date", "geo"),
                coords={
                    "chain": np.arange(n_chain),
                    "draw": np.arange(n_draw),
                    "date": dates,
                    "geo": geos,
                },
            ),
        }
    )
    const = xr.Dataset(
        {
            "target_data": xr.DataArray(
                rng.normal(500, 50, size=(n_date, len(geos))),
                dims=("date", "geo"),
                coords={"date": dates, "geo": geos},
            ),
            "target_scale": xr.DataArray(1000.0),
        }
    )
    return az.InferenceData(posterior=posterior, constant_data=const)


@pytest.fixture(scope="module")
def simple_data(simple_idata) -> MMMIDataWrapper:
    return MMMIDataWrapper(simple_idata, validate_on_init=False)


@pytest.fixture(scope="module")
def panel_data(panel_idata) -> MMMIDataWrapper:
    return MMMIDataWrapper(panel_idata, validate_on_init=False)


@pytest.fixture(scope="module")
def simple_plots(simple_data) -> DecompositionPlots:
    return DecompositionPlots(simple_data)


@pytest.fixture(scope="module")
def panel_plots(panel_data) -> DecompositionPlots:
    return DecompositionPlots(panel_data)
