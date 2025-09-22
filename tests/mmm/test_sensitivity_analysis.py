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
    me = sensitivity.compute_marginal_effects(results)
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
    )
    # Apply filtering post hoc per new API
    filtered = results.sel(channel=["a", "c"])
    assert filtered.sizes["channel"] == 2


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
    filtered = result.sel(country=["B"], channel=["C2"])  # filter post hoc per new API
    assert filtered.sizes["country"] == 1
    assert filtered.sizes["channel"] == 1
    assert list(filtered.coords["country"].values) == ["B"]
    assert list(filtered.coords["channel"].values) == ["C2"]


def test_compute_uplift_curve_respect_to_base(sensitivity):
    sweeps = np.linspace(0.5, 1.5, 5)
    # Run without extending idata to get the results in-memory
    results = sensitivity.run_sweep(
        varinput="channel_data",
        var_names="channel_contribution",
        sweep_values=sweeps,
        sweep_type="multiplicative",
    )
    # Choose a scalar reference (e.g., overall mean at baseline factor 1.0)
    ref_value = float(results.sel(sweep=1.0).mean().item())

    uplift = sensitivity.compute_uplift_curve_respect_to_base(
        results=results, ref=ref_value
    )

    # Dimensions and sizes unchanged
    assert list(uplift.dims) == list(results.dims)
    assert uplift.sizes == results.sizes

    # Uplift equals results minus scalar reference (broadcasted)
    xr.testing.assert_allclose(uplift, results - ref_value)

    # Now also test persistence to idata
    # First ensure the group exists
    _ = sensitivity.run_sweep(
        varinput="channel_data",
        var_names="channel_contribution",
        sweep_values=sweeps,
        extend_idata=True,
    )
    persisted = sensitivity.compute_uplift_curve_respect_to_base(
        results=results, ref=ref_value, extend_idata=True
    )
    assert hasattr(sensitivity.idata, "sensitivity_analysis")
    sa_group = sensitivity.idata.sensitivity_analysis
    assert isinstance(sa_group, xr.Dataset)
    assert "uplift_curve" in sa_group
    xr.testing.assert_allclose(sa_group["uplift_curve"], persisted)


def test_compute_dims_order_from_varinput_internal(sensitivity):
    # Drops 'date' and preserves remaining order
    dims_order = sensitivity._compute_dims_order_from_varinput("channel_data")
    assert dims_order == ["channel"]


def test_add_to_idata_internal_updates_dataset(simple_model_and_idata):
    model, idata = simple_model_and_idata
    sa = SensitivityAnalysis(model, idata)
    sweeps = np.linspace(0.5, 1.5, 3)

    # Create a tiny result DataArray consistent with model coords
    result1 = xr.DataArray(
        np.ones((2, len(sweeps), len(model.coords["channel"]))),
        dims=["sample", "sweep", "channel"],
        coords={
            "sample": np.arange(2),
            "sweep": sweeps,
            "channel": list(model.coords["channel"]),
        },
    )

    # First add: creates the group
    sa._add_to_idata(result1)
    assert hasattr(idata, "sensitivity_analysis")
    assert isinstance(idata.sensitivity_analysis, xr.Dataset)
    xr.testing.assert_allclose(idata.sensitivity_analysis["x"], result1)

    # Second add: updates the variable and warns
    result2 = result1 * 2.0
    with pytest.warns(UserWarning):
        sa._add_to_idata(result2)
    xr.testing.assert_allclose(idata.sensitivity_analysis["x"], result2)
