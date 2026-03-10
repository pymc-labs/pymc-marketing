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
from pathlib import Path

import pandas as pd
import xarray as xr

from benchmark.ground_truth import (
    compute_true_parameters_from_roas_dataset,
    compute_true_roas_from_multidimensional_data,
    compute_true_roas_from_roas_dataset,
    load_true_parameters_from_netcdf,
)


def test_compute_true_roas_from_roas_dataset(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "y": [120.0, 130.0],
            "y01": [100.0, 110.0],
            "y02": [115.0, 120.0],
            "x1": [10.0, 10.0],
            "x2": [5.0, 5.0],
        }
    )
    csv_path = tmp_path / "roas.csv"
    df.to_csv(csv_path, index=False)

    roas = compute_true_roas_from_roas_dataset(csv_path, channel_columns=["x1", "x2"])
    assert roas["x1"] == 2.0
    assert roas["x2"] == 1.5


def test_compute_true_parameters_from_roas_dataset(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "x1_effect": [20.0, 40.0],
            "x1_adstock_saturated": [10.0, 20.0],
            "x2_effect": [9.0, 18.0],
            "x2_adstock_saturated": [3.0, 6.0],
        }
    )
    csv_path = tmp_path / "roas.csv"
    df.to_csv(csv_path, index=False)

    params = compute_true_parameters_from_roas_dataset(
        csv_path, channel_columns=["x1", "x2"]
    )
    assert params["beta_x1_adstock_saturated"] == 2.0
    assert params["beta_x2_adstock_saturated"] == 3.0


def test_load_true_parameters_from_netcdf(tmp_path: Path) -> None:
    ds = xr.Dataset(
        data_vars={
            "saturation_lam": ("channel", [0.4, 0.5]),
        },
        coords={"channel": ["x1", "x2"]},
    )
    nc_path = tmp_path / "truth.nc"
    ds.to_netcdf(nc_path)

    loaded = load_true_parameters_from_netcdf(nc_path)
    assert loaded["saturation_lam"]["x1"] == 0.4


def test_compute_true_roas_from_multidimensional_data(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-08", "2024-01-15"],
            "geo": ["geo_a", "geo_a", "geo_a"],
            "x1": [10.0, 0.0, 0.0],
            "x2": [0.0, 5.0, 0.0],
        }
    )
    csv_path = tmp_path / "multi.csv"
    data.to_csv(csv_path, index=False)

    params = xr.Dataset(
        data_vars={
            "adstock_alpha": (("geo", "channel"), [[0.0, 0.0]]),
            "saturation_lam": ("channel", [1.0, 1.0]),
            "saturation_beta": (("channel", "geo"), [[2.0], [3.0]]),
        },
        coords={"geo": ["geo_a"], "channel": ["x1", "x2"]},
    )
    nc_path = tmp_path / "multi.nc"
    params.to_netcdf(nc_path)

    roas = compute_true_roas_from_multidimensional_data(
        dataset_path=csv_path,
        parameters_netcdf_path=nc_path,
        channel_columns=["x1", "x2"],
    )
    assert roas["x1"] > 0.0
    assert roas["x2"] > 0.0
