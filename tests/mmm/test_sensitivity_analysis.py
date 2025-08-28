#   Copyright 2022 - 2025 The PyMC Labs Developers
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
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.mmm.sensitivity_analysis import SensitivityAnalysis


@pytest.fixture
def simple_model_and_idata():
    """A minimal PyMC model and an InferenceData posterior compatible with SensitivityAnalysis.

    Model dims convention: (date, channel). Response is a deterministic depending on pm.Data and two RVs.
    """
    rng = np.random.default_rng(123)
    n_dates, n_channels, n_draws = 6, 4, 5

    coords = {"date": np.arange(n_dates), "channel": list("abcd")}
    with pm.Model(coords=coords) as model:
        X = pm.Data(
            "channel_data",
            rng.gamma(2.0, 1.0, size=(n_dates, n_channels)).astype("float64"),
            dims=("date", "channel"),
        )
        alpha = pm.Gamma("alpha", 1.0, 1.0, dims=("channel",))
        lam = pm.Gamma("lam", 1.0, 1.0, dims=("channel",))

        def saturation(x, alpha_param, lam_param):
            return (alpha_param * x) / (x + lam_param)

        pm.Deterministic(
            "channel_contribution",
            saturation(X, alpha, lam),
            dims=("date", "channel"),
        )

    # Build a tiny posterior with shapes matching free RVs: (chain=1, draw=n_draws, channel)
    alpha_draws = np.full((1, n_draws, n_channels), 0.8, dtype="float64")
    lam_draws = np.full((1, n_draws, n_channels), 1.2, dtype="float64")

    idata = az.from_dict(posterior={"alpha": alpha_draws, "lam": lam_draws})
    return model, idata


@pytest.fixture
def sensitivity(simple_model_and_idata):
    model, idata = simple_model_and_idata
    return SensitivityAnalysis(model, idata)


@pytest.mark.parametrize("sweep_type", ["multiplicative", "additive", "absolute"])
def test_run_sweep_basic(sensitivity, sweep_type):
    sweep_values = np.linspace(0.5, 1.5, 3)
    results = sensitivity.run_sweep(
        varinput="channel_data",
        var_names="channel_contribution",
        sweep_values=sweep_values,
        sweep_type=sweep_type,
    )

    assert isinstance(results, xr.DataArray)
    assert list(results.dims)[:2] == ["sample", "sweep"]
    assert "channel" in results.dims
    assert results.sizes["sweep"] == len(sweep_values)


def test_run_sweep_invalid_var_name(sensitivity):
    with pytest.raises(KeyError):
        sensitivity.run_sweep(
            varinput="channel_data",
            var_names="invalid_variable",
            sweep_values=np.linspace(0.5, 1.5, 3),
        )


def test_compute_marginal_effects(sensitivity):
    sweeps = np.linspace(0.5, 1.5, 5)
    results = sensitivity.run_sweep(
        varinput="channel_data",
        var_names="channel_contribution",
        sweep_values=sweeps,
        sweep_type="multiplicative",
    )
    me = SensitivityAnalysis.compute_marginal_effects(results, sweeps)
    assert isinstance(me, xr.DataArray)
    # same dims, same sizes except along sweep reduced by differentiation (xarray keeps same length)
    assert list(me.dims) == list(results.dims)
    assert me.sizes == results.sizes


def test_run_sweep_with_filter(sensitivity):
    sweeps = np.linspace(0.5, 1.5, 4)
    results = sensitivity.run_sweep(
        varinput="channel_data",
        var_names="channel_contribution",
        sweep_values=sweeps,
        var_names_filter={"channel": ["a", "c"]},
    )
    assert results.sizes["channel"] == 2


def test_extend_idata_stores_results(simple_model_and_idata):
    model, idata = simple_model_and_idata
    sa = SensitivityAnalysis(model, idata)
    sweeps = np.linspace(0.5, 1.5, 3)
    out = sa.run_sweep(
        varinput="channel_data",
        var_names="channel_contribution",
        sweep_values=sweeps,
        extend_idata=True,
    )
    assert out is None
    # Ensure results are attached; InferenceData stores arbitrary groups as attributes
    assert hasattr(idata, "sensitivity_analysis")


@pytest.fixture
def df() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=3, freq="W-MON").rename("date")
    df = pd.DataFrame(
        {
            ("A", "C1"): [1, 2, 3],
            ("B", "C1"): [4, 5, 6],
            ("A", "C2"): [7, 8, 9],
            ("B", "C2"): [10, 11, 12],
        },
        index=dates,
    )
    df.columns.names = ["country", "channel"]
    df = df.astype(float)

    y = pd.DataFrame(
        {
            ("A", "y"): [1, 2, 3],
            ("B", "y"): [4, 5, 6],
        },
        index=dates,
    )
    y.columns.names = ["country", "channel"]
    y = y.astype(float)

    return pd.concat(
        [
            df.stack("country", future_stack=True),
            y.stack("country", future_stack=True),
        ],
        axis=1,
    ).reset_index()


@pytest.fixture
def multidim_mmm(df, mock_pymc_sample):
    mmm = MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        dims=("country",),
        target_column="y",
        adstock=GeometricAdstock(l_max=3),
        saturation=LogisticSaturation(),
    )
    X = df.drop(columns=["y"]).copy()
    y = df["y"].copy()
    # Fit with mocked sampling (conftest provides mock_pymc_sample)
    mmm.fit(X, y)
    return mmm


@pytest.mark.parametrize("sweep_type", ["multiplicative", "additive", "absolute"])
def test_mmm_sensitivity_dims_and_filter(multidim_mmm, sweep_type):
    sweeps = np.linspace(0.5, 1.5, 4)
    # Run sweep via the MMM.sensitivity property using the real model/idata
    result = multidim_mmm.sensitivity.run_sweep(
        varinput="channel_data",
        var_names="channel_contribution",
        sweep_values=sweeps,
        sweep_type=sweep_type,
    )

    # Dims order should follow (sample, sweep, *dims, channel)
    assert list(result.dims)[:2] == ["sample", "sweep"]
    assert list(result.dims[2:]) == ["country", "channel"]

    # Coords should match model coords
    assert list(result.coords["country"].values) == ["A", "B"]
    assert list(result.coords["channel"].values) == ["C1", "C2"]

    # Filtering by labels should preserve order and sizes
    filtered = multidim_mmm.sensitivity.run_sweep(
        varinput="channel_data",
        var_names="channel_contribution",
        sweep_values=sweeps,
        sweep_type=sweep_type,
        var_names_filter={"country": ["B"], "channel": ["C2"]},
    )
    assert filtered.sizes["country"] == 1
    assert filtered.sizes["channel"] == 1
    assert list(filtered.coords["country"].values) == ["B"]
    assert list(filtered.coords["channel"].values) == ["C2"]
