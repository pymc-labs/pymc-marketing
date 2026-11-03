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
import copy
import gc
import weakref

import cloudpickle
import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytest
import xarray as xr
from pytensor.compile.maker import FunctionMaker

from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_marketing.mmm.leased_cache import Lease
from pymc_marketing.mmm.mmm import MMM
from pymc_marketing.mmm.sensitivity_analysis import SensitivityAnalysis
from pymc_marketing.mmm.sweep_cache import SweepCache
from pymc_marketing.mmm.sweep_graph import SweepEntry
from pymc_marketing.pytensor_utils import ConditionedModel, ModelCloneError

_SENSITIVITY_KWARGS = {
    "var_names": "channel_contribution",
    "sweep_type": "multiplicative",
}


def _make_saturation_model_and_idata(channel_data, n_draws, alpha_val=0.8, lam_val=1.2):
    """Build a minimal saturation model (alpha*X)/(X+lam) with posterior draws."""
    n_dates, n_channels = channel_data.shape
    channel_names = [chr(ord("a") + i) for i in range(n_channels)]
    coords = {"date": np.arange(n_dates), "channel": channel_names}

    with pm.Model(coords=coords) as model:
        X = pmd.Data("channel_data", channel_data, dims=("date", "channel"))
        alpha = pmd.Gamma("alpha", 1.0, 1.0, dims=("channel",))
        lam = pmd.Gamma("lam", 1.0, 1.0, dims=("channel",))
        pmd.Deterministic(
            "channel_contribution",
            (alpha * X) / (X + lam),
            dims=("date", "channel"),
        )

    alpha_draws = np.full((1, n_draws, n_channels), alpha_val, dtype=np.float64)
    lam_draws = np.full((1, n_draws, n_channels), lam_val, dtype=np.float64)
    idata = xr.DataTree.from_dict(
        {
            "/posterior": xr.Dataset(
                {
                    "alpha": xr.DataArray(
                        alpha_draws, dims=["chain", "draw", "channel"]
                    ),
                    "lam": xr.DataArray(lam_draws, dims=["chain", "draw", "channel"]),
                }
            ),
        }
    )
    assert set(idata.posterior.dims) == {"chain", "draw", "channel"}
    return model, idata


@pytest.fixture
def simple_model_and_idata():
    """A minimal PyMC model and an InferenceData posterior compatible with SensitivityAnalysis.

    Model dims convention: (date, channel). Response is a deterministic depending on pm.Data and two RVs.
    """
    rng = np.random.default_rng(123)
    channel_data = rng.gamma(2.0, 1.0, size=(6, 4)).astype(np.float64)
    return _make_saturation_model_and_idata(channel_data, n_draws=5)


@pytest.fixture
def sensitivity(simple_model_and_idata):
    model, idata = simple_model_and_idata
    return SensitivityAnalysis(model, idata)


@pytest.mark.parametrize("sweep_type", ["multiplicative", "additive", "absolute"])
@pytest.mark.parametrize("use_mask", [False, True])
def test_run_sweep_basic(sensitivity, sweep_type, use_mask):
    sweep_values = np.linspace(0.5, 1.5, 3)

    if use_mask:
        # Create a mask that keeps first and last channels for all dates
        # channel_contribution has dims (date, channel) with 6 dates and 4 channels
        mask_2d = np.zeros((6, 4), dtype=bool)
        mask_2d[:, [0, 3]] = True  # Keep channels 'a' and 'd' for all dates
        response_mask = xr.DataArray(mask_2d, dims=("date", "channel"))
    else:
        response_mask = None

    results = sensitivity.run_sweep(
        var_input="channel_data",
        var_names="channel_contribution",
        sweep_values=sweep_values,
        sweep_type=sweep_type,
        response_mask=response_mask,
    )

    assert isinstance(results, xr.DataArray)
    assert list(results.dims)[:2] == ["sample", "sweep"]
    assert "channel" in results.dims
    assert results.sizes["sweep"] == len(sweep_values)

    if use_mask:
        # Check that masked channels are zeroed
        assert np.all(results.sel(channel="b").values == 0.0)
        assert np.all(results.sel(channel="c").values == 0.0)


def test_run_sweep_with_response_mask(sensitivity):
    sweep_values = np.linspace(0.5, 1.5, 3)
    full = sensitivity.run_sweep(
        var_input="channel_data",
        var_names="channel_contribution",
        sweep_values=sweep_values,
    )

    # Create 2D mask for (date, channel) dimensions
    mask_2d = np.zeros((6, 4), dtype=bool)
    mask_2d[:, [0, 3]] = True  # Keep channels 'a' and 'd' for all dates
    masked = sensitivity.run_sweep(
        var_input="channel_data",
        var_names="channel_contribution",
        sweep_values=sweep_values,
        response_mask=xr.DataArray(mask_2d, dims=("date", "channel")),
    )

    assert masked.shape == full.shape
    kept = full.sel(channel=["a", "d"]).sum("channel")
    xr.testing.assert_allclose(masked.sum("channel"), kept)
    assert np.all(masked.sel(channel="b").values == 0.0)


def test_run_sweep_invalid_var_name(sensitivity):
    with pytest.raises(KeyError):
        sensitivity.run_sweep(
            var_input="channel_data",
            var_names="invalid_variable",
            sweep_values=np.linspace(0.5, 1.5, 3),
        )


def test_compute_marginal_effects(sensitivity):
    sweeps = np.linspace(0.5, 1.5, 5)
    results = sensitivity.run_sweep(
        var_input="channel_data",
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
        var_input="channel_data",
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
        var_input="channel_data",
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
        var_input="channel_data",
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
        var_input="channel_data",
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
        var_input="channel_data",
        var_names="channel_contribution",
        sweep_values=sweeps,
        extend_idata=True,
    )
    persisted = sensitivity.compute_uplift_curve_respect_to_base(
        results=results, ref=ref_value, extend_idata=True
    )
    assert hasattr(sensitivity.idata, "sensitivity_analysis")
    sa_group = sensitivity.idata.sensitivity_analysis
    if hasattr(sa_group, "dataset"):
        sa_group = sa_group.dataset
    assert "uplift_curve" in sa_group
    xr.testing.assert_allclose(sa_group["uplift_curve"], persisted)


def test_compute_uplift_curve_respect_to_base_array_ref(sensitivity):
    sweeps = np.linspace(0.5, 1.5, 5)
    results = sensitivity.run_sweep(
        var_input="channel_data",
        var_names="channel_contribution",
        sweep_values=sweeps,
        sweep_type="multiplicative",
    )

    baseline = results.sel(sweep=1.0).mean(dim="sample")

    uplift = sensitivity.compute_uplift_curve_respect_to_base(
        results=results, ref=baseline
    )

    broadcasted = results - baseline
    xr.testing.assert_allclose(uplift, broadcasted)


def test_posterior_sample_percentage_controls_draws(sensitivity):
    sweeps = np.linspace(0.5, 1.5, 4)
    full = sensitivity.run_sweep(
        var_input="channel_data",
        var_names="channel_contribution",
        sweep_values=sweeps,
        posterior_sample_fraction=1.0,
    )
    limited = sensitivity.run_sweep(
        var_input="channel_data",
        var_names="channel_contribution",
        sweep_values=sweeps,
        posterior_sample_fraction=0.6,
    )

    assert full.sizes["sample"] == 5
    assert limited.sizes["sample"] == 2


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
    sa_group = idata.sensitivity_analysis
    if hasattr(sa_group, "dataset"):
        sa_group = sa_group.dataset
    xr.testing.assert_allclose(sa_group["x"], result1)

    # Second add: updates the variable and warns
    result2 = result1 * 2.0
    with pytest.warns(UserWarning):
        sa._add_to_idata(result2)
    sa_group = idata.sensitivity_analysis
    if hasattr(sa_group, "dataset"):
        sa_group = sa_group.dataset
    xr.testing.assert_allclose(sa_group["x"], result2)


def test_prepare_response_mask_numpy_array(sensitivity):
    """Test _prepare_response_mask with numpy array input."""
    response_dims = ("date", "channel")

    # Test boolean array
    mask = np.array(
        [
            [True, False, True, False],
            [False, True, False, True],
            [True, True, False, False],
            [False, False, True, True],
            [True, False, True, False],
            [False, True, False, True],
        ]
    )
    result = sensitivity._prepare_response_mask(mask, response_dims, var_names="test")
    assert result.shape == (6, 4)
    assert result.dtype == bool
    np.testing.assert_array_equal(result, mask)

    # Test integer array (should be cast to bool)
    int_mask = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ]
    )
    result = sensitivity._prepare_response_mask(
        int_mask, response_dims, var_names="test"
    )
    assert result.dtype == bool
    np.testing.assert_array_equal(result, mask)


def test_prepare_response_mask_xarray(sensitivity):
    """Test _prepare_response_mask with xarray input."""
    response_dims = ("date", "channel")

    # Test with correct dims
    mask_data = np.array(
        [
            [True, False, True, False],
            [False, True, False, True],
            [True, True, False, False],
            [False, False, True, True],
            [True, False, True, False],
            [False, True, False, True],
        ]
    )
    mask_xr = xr.DataArray(mask_data, dims=("date", "channel"))
    result = sensitivity._prepare_response_mask(
        mask_xr, response_dims, var_names="test"
    )
    np.testing.assert_array_equal(result, mask_data)

    # Test with reordered dims (should be transposed)
    mask_xr_transposed = xr.DataArray(mask_data.T, dims=("channel", "date"))
    result = sensitivity._prepare_response_mask(
        mask_xr_transposed, response_dims, var_names="test"
    )
    np.testing.assert_array_equal(result, mask_data)


def test_prepare_response_mask_errors(sensitivity):
    """Test _prepare_response_mask error cases."""
    response_dims = ("date", "channel")

    # Test missing dims in xarray
    mask_xr = xr.DataArray(np.ones((6,)), dims=("date",))
    with pytest.raises(ValueError, match="response_mask is missing required dims"):
        sensitivity._prepare_response_mask(mask_xr, response_dims, var_names="test")

    # Test wrong number of dimensions
    mask = np.ones((6,))  # 1D instead of 2D
    with pytest.raises(
        ValueError, match="response_mask must have the same number of dims"
    ):
        sensitivity._prepare_response_mask(mask, response_dims, var_names="test")

    # Test wrong shape
    mask = np.ones((5, 3))  # Wrong shape
    with pytest.raises(ValueError, match="response_mask shape does not match"):
        sensitivity._prepare_response_mask(mask, response_dims, var_names="test")

    # Test non-castable type
    mask = np.array([["yes", "no"], ["no", "yes"]])
    with pytest.raises(TypeError, match="response_mask must be boolean"):
        sensitivity._prepare_response_mask(mask, response_dims, var_names="test")


def test_run_sweep_with_2d_mask(simple_model_and_idata):
    """Test run_sweep with a 2D mask matching date x channel dims."""
    model, idata = simple_model_and_idata
    sa = SensitivityAnalysis(model, idata)

    # Create a 2D mask that only keeps some dates for channel a and d, none for b and c
    mask_2d = np.zeros((6, 4), dtype=bool)
    # Keep first 3 dates for channel a
    mask_2d[:3, 0] = True
    # Keep last 3 dates for channel d
    mask_2d[3:, 3] = True
    # Channels b and c are completely masked

    mask_xr = xr.DataArray(mask_2d, dims=("date", "channel"))

    sweep_values = np.linspace(0.5, 1.5, 3)
    masked = sa.run_sweep(
        var_input="channel_data",
        var_names="channel_contribution",
        sweep_values=sweep_values,
        response_mask=mask_xr,
    )

    # Get the full result for comparison
    full = sa.run_sweep(
        var_input="channel_data",
        var_names="channel_contribution",
        sweep_values=sweep_values,
    )

    # Channels 'b' and 'c' should be completely zero since no dates were unmasked
    assert np.all(masked.sel(channel="b").values == 0.0)
    assert np.all(masked.sel(channel="c").values == 0.0)

    # Channels 'a' and 'd' should have non-zero values
    assert np.any(masked.sel(channel="a").values > 0.0)
    assert np.any(masked.sel(channel="d").values > 0.0)

    # The masked result should be smaller than the full result
    assert masked.sum().item() < full.sum().item()


def test_run_sweep_integer_channel_data_no_truncation():
    """Fractional sweep_values must not be truncated when channel_data has integer dtype."""
    rng = np.random.default_rng(42)
    channel_data_int = rng.integers(50, 200, size=(8, 3)).astype(np.int32)
    model, idata = _make_saturation_model_and_idata(channel_data_int, n_draws=10)

    sa = SensitivityAnalysis(model, idata)
    sweep_values = np.array([0.5, 1.0, 1.5])

    result = sa.run_sweep(
        var_input="channel_data",
        var_names="channel_contribution",
        sweep_values=sweep_values,
        sweep_type="multiplicative",
    )

    # The channel contribution (alpha*X)/(X+lam) is strictly monotone in X for
    # positive alpha and lam.  So mean contribution must be strictly ordered with
    # sweep index — any truncation of fractional values breaks this ordering.
    mean_by_sweep = result.mean(dim=["sample", "channel"])  # shape: (sweep,)
    sweep_vals = mean_by_sweep.values

    assert sweep_vals[0] > 0
    assert sweep_vals[0] < sweep_vals[1] < sweep_vals[2], (
        f"expected strictly increasing contributions across sweep values "
        f"{sweep_values.tolist()}, got {sweep_vals.tolist()}. "
        "Integer truncation of fractional sweep_values may be the cause."
    )


# ---------------------------------------------------------------------------
# Closed-form and cold-instance references for the sweep-cache behaviour.
# ---------------------------------------------------------------------------

_SATURATION_ALPHA = 0.8
_SATURATION_LAM = 1.2


def _swept_input(
    channel_data: np.ndarray,
    sweep_values: np.ndarray,
    sweep_type: str,
) -> np.ndarray:
    """Return the swept input with shape ``(sweep, date, channel)``."""
    sweep = np.asarray(sweep_values)[:, None, None]
    match sweep_type:
        case "multiplicative":
            return sweep * channel_data
        case "additive":
            return sweep + channel_data
        case "absolute":
            return np.broadcast_to(sweep, (len(sweep_values), *channel_data.shape))
        case _:
            raise ValueError(f"Unknown sweep_type {sweep_type!r}")


def _saturation_values(
    channel_data: np.ndarray,
    alpha: np.ndarray,
    lam: np.ndarray,
    response_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``alpha*x/(x+lam)`` with shape ``(sample, date, channel)``."""
    response = (
        alpha[:, None, :] * channel_data[None] / (channel_data[None] + lam[:, None, :])
    )
    if response_mask is not None:
        response = np.where(response_mask[None, ...], response, 0.0)
    return response


def _saturation_response(
    channel_data: np.ndarray,
    sweep_values: np.ndarray,
    sweep_type: str,
    alpha: np.ndarray,
    lam: np.ndarray,
    response_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return the saturation response summed over date for every sweep value."""
    swept = _swept_input(channel_data, sweep_values, sweep_type)
    response = (
        alpha[:, None, None, :] * swept[None] / (swept[None] + lam[:, None, None, :])
    )
    if response_mask is not None:
        response = np.where(response_mask[None, None, ...], response, 0.0)
    return response.sum(axis=-2)


def _as_draws(
    values: np.ndarray | float, *, sample_size: int, n_channels: int
) -> np.ndarray:
    """Broadcast a scalar or a per-channel value to ``(sample, channel)``."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return np.full((sample_size, n_channels), float(values))
    if values.ndim == 1:
        values = values[None, :]
    return np.broadcast_to(values, (sample_size, n_channels))


def _expected_saturation_sweep(
    model: pm.Model,
    channel_data: np.ndarray,
    sweep_values: np.ndarray,
    *,
    sweep_type: str = "multiplicative",
    alpha: np.ndarray | float = _SATURATION_ALPHA,
    lam: np.ndarray | float = _SATURATION_LAM,
    response_mask: xr.DataArray | np.ndarray | None = None,
    sample_size: int = 5,
) -> xr.DataArray:
    """Return the closed-form sweep of ``alpha*x/(x+lam)`` for the synthetic model."""
    n_channels = channel_data.shape[1]
    alpha = _as_draws(alpha, sample_size=sample_size, n_channels=n_channels)
    lam = _as_draws(lam, sample_size=sample_size, n_channels=n_channels)
    mask = None if response_mask is None else np.asarray(response_mask)
    values = _saturation_response(
        np.asarray(channel_data),
        np.asarray(sweep_values),
        sweep_type,
        alpha,
        lam,
        mask,
    )
    return xr.DataArray(
        values,
        dims=("sample", "sweep", "channel"),
        coords={
            "sample": np.arange(sample_size),
            "sweep": np.asarray(sweep_values),
            "channel": list(model.coords["channel"]),
        },
    )


def _expected_two_input_sweep(
    model: pm.Model,
    sweep_values: np.ndarray,
    *,
    var_input: str,
    var_names: str,
    sweep_type: str,
    response_mask: xr.DataArray | np.ndarray | None = None,
    sample_size: int = 5,
) -> xr.DataArray:
    """Return the closed-form sweep of the two-input synthetic model."""
    first = np.asarray(model["channel_data"].get_value())
    second = np.asarray(model["channel_data_2"].get_value())
    n_channels = first.shape[1]
    alpha = _as_draws(_SATURATION_ALPHA, sample_size=sample_size, n_channels=n_channels)
    lam = _as_draws(_SATURATION_LAM, sample_size=sample_size, n_channels=n_channels)

    def contribution(data: np.ndarray, swept: bool) -> np.ndarray:
        if not swept:
            values = _saturation_values(data, alpha, lam)
            return values[:, None, :, :]
        swept_data = _swept_input(data, sweep_values, sweep_type)
        return (
            alpha[:, None, None, :]
            * swept_data[None]
            / (swept_data[None] + lam[:, None, None, :])
        )

    response = contribution(first, var_input == "channel_data") + contribution(
        second, var_input == "channel_data_2"
    )
    if var_names == "date_contribution":
        response = response.sum(axis=-1)
        dims = ("sample", "sweep")
        date_axis = -1
    else:
        dims = ("sample", "sweep", "channel")
        date_axis = -2

    if response_mask is not None:
        response = np.where(np.asarray(response_mask)[None, None, ...], response, 0.0)
    response = response.sum(axis=date_axis)

    coords = {"sample": np.arange(sample_size), "sweep": np.asarray(sweep_values)}
    if var_names == "channel_contribution":
        coords["channel"] = list(model.coords["channel"])
    return xr.DataArray(response, dims=dims, coords=coords)


def _synthetic_response_mask() -> xr.DataArray:
    """Keep channels ``a`` and ``d`` on every date of ``simple_model_and_idata``."""
    mask = np.zeros((6, 4), dtype=bool)
    mask[:, [0, 3]] = True
    return xr.DataArray(mask, dims=("date", "channel"))


def _multidim_response_mask() -> xr.DataArray:
    """Keep channel ``C1`` of country ``B`` on every date of ``multidim_mmm``."""
    mask = np.zeros((3, 2, 2), dtype=bool)
    mask[:, 1, 0] = True
    return xr.DataArray(mask, dims=("date", "country", "channel"))


@pytest.mark.parametrize(
    ("posterior_sample_fraction", "sample_size"), [(1.0, 5), (0.5, 1)]
)
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("sweep_type", ["multiplicative", "additive", "absolute"])
def test_run_sweep_matches_the_closed_form(
    simple_model_and_idata,
    sweep_type,
    use_mask,
    posterior_sample_fraction,
    sample_size,
):
    """The sweep grid on the synthetic model matches its closed-form values."""
    model, idata = simple_model_and_idata
    sweep_values = np.linspace(0.5, 1.5, 4)
    response_mask = _synthetic_response_mask() if use_mask else None

    result = SensitivityAnalysis(model, idata).run_sweep(
        "channel_data",
        sweep_values,
        var_names="channel_contribution",
        sweep_type=sweep_type,
        posterior_sample_fraction=posterior_sample_fraction,
        response_mask=response_mask,
    )
    expected = _expected_saturation_sweep(
        model,
        np.asarray(model["channel_data"].get_value()),
        sweep_values,
        sweep_type=sweep_type,
        response_mask=response_mask,
        sample_size=sample_size,
    )

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("sweep_type", ["multiplicative", "additive", "absolute"])
def test_multidim_run_sweep_matches_a_cold_instance(multidim_mmm, sweep_type, use_mask):
    """The sweep grid keeps its dims, labels and values on a multidim MMM.

    A fresh instance has its own empty cache, so the warm shared cache cannot
    influence the expected result.  The mask covers one country/channel pair,
    so the grid exercises the labeled ``country`` and ``channel`` axes as well
    as the masked-overlay path.  A changed draw count is covered, for the
    multiplicative masked case, by the interleaved-fraction test below.
    """
    sweep_values = np.linspace(0.5, 1.5, 4)
    response_mask = _multidim_response_mask() if use_mask else None
    kwargs = {
        **_SENSITIVITY_KWARGS,
        "sweep_type": sweep_type,
        "response_mask": response_mask,
    }

    actual = multidim_mmm.sensitivity.run_sweep("channel_data", sweep_values, **kwargs)
    expected = SensitivityAnalysis(multidim_mmm.model, multidim_mmm.idata).run_sweep(
        "channel_data", sweep_values, **kwargs
    )

    xr.testing.assert_allclose(actual, expected)


def test_multidim_interleaved_fractions_match_their_own_cold_runs(multidim_mmm):
    """One shared cache serves both draw fractions without mixing their draws.

    ``posterior_sample_fraction`` is not part of the cache key, so a call at
    another fraction rebinds the entry the previous call warmed.  Each result
    must still equal an empty cache's result for its own fraction, mask
    included.
    """
    sweeps = np.linspace(0.5, 1.5, 4)
    kwargs = {
        **_SENSITIVITY_KWARGS,
        "response_mask": _multidim_response_mask(),
    }

    for fraction in (1.0, 0.5, 1.0, 0.5):
        result = multidim_mmm.sensitivity.run_sweep(
            "channel_data", sweeps, posterior_sample_fraction=fraction, **kwargs
        )
        expected = SensitivityAnalysis(
            multidim_mmm.model, multidim_mmm.idata
        ).run_sweep(
            "channel_data",
            sweeps,
            posterior_sample_fraction=fraction,
            **kwargs,
        )
        xr.testing.assert_allclose(result, expected)


def _data_only_response_model_and_idata() -> tuple[pm.Model, xr.DataTree]:
    """A model whose response reads no posterior variable, only ``pm.Data``."""
    rng = np.random.default_rng(7)
    coords = {"date": np.arange(6), "channel": list("abcd")}
    with pm.Model(coords=coords) as model:
        X = pmd.Data(
            "channel_data", rng.gamma(2.0, 1.0, (6, 4)), dims=("date", "channel")
        )
        # A free RV the response does not read, so the posterior has a draw set.
        pmd.Gamma("alpha", 1.0, 1.0, dims=("channel",))
        pmd.Deterministic("channel_contribution", 2.0 * X, dims=("date", "channel"))

    idata = xr.DataTree.from_dict(
        {
            "/posterior": xr.Dataset(
                {
                    "alpha": xr.DataArray(
                        np.full((1, 5, 4), 0.8),
                        dims=["chain", "draw", "channel"],
                        coords={"channel": list("abcd")},
                    )
                }
            )
        }
    )
    return model, idata


def test_run_sweep_a_data_only_response_keeps_its_failure_mode():
    """A response that reads no draw keeps the whole tree and fails at the sum.

    Such a response has no ``sample`` dim to transpose.  Narrowing it to an
    empty posterior would fail earlier, about a missing ``draw`` dim, so the
    tree is used whole and the error stays the one the uncached path raises.
    """
    model, idata = _data_only_response_model_and_idata()
    analysis = SensitivityAnalysis(model, idata)
    sweeps = np.linspace(0.5, 1.5, 4)

    with pytest.raises(ValueError, match=r"Dimensions \{'sample'\}"):
        analysis.run_sweep("channel_data", sweeps, var_names="channel_contribution")


def _constant_saturation_idata(n_channels: int = 4, n_draws: int = 5) -> xr.DataTree:
    """Posterior draws of ``alpha`` and ``lam`` constant across draws and channels."""
    channel = list("abcd")[:n_channels]
    draws = {
        name: xr.DataArray(
            np.full((1, n_draws, n_channels), value),
            dims=["chain", "draw", "channel"],
            coords={"channel": channel},
        )
        for name, value in (("alpha", _SATURATION_ALPHA), ("lam", _SATURATION_LAM))
    }
    return xr.DataTree.from_dict({"/posterior": xr.Dataset(draws)})


def _plain_deterministic_model_and_idata() -> tuple[pm.Model, xr.DataTree]:
    """A model whose response is a plain ``pm.Deterministic`` without dims."""
    rng = np.random.default_rng(3)
    coords = {"date": np.arange(6), "channel": list("abcd")}
    with pm.Model(coords=coords) as model:
        X = pm.Data(
            "channel_data", rng.gamma(2.0, 1.0, (6, 4)), dims=("date", "channel")
        )
        alpha = pm.Gamma("alpha", 1.0, 1.0, dims=("channel",))
        lam = pm.Gamma("lam", 1.0, 1.0, dims=("channel",))
        pm.Deterministic("channel_contribution", (alpha * X) / (X + lam))

    return model, _constant_saturation_idata()


def _scalar_deterministic_model_and_idata() -> tuple[pm.Model, xr.DataTree]:
    """A model whose response is a scalar ``pmd.Deterministic``."""
    rng = np.random.default_rng(4)
    coords = {"date": np.arange(6), "channel": list("abcd")}
    with pm.Model(coords=coords) as model:
        X = pmd.Data(
            "channel_data", rng.gamma(2.0, 1.0, (6, 4)), dims=("date", "channel")
        )
        alpha = pmd.Gamma("alpha", 1.0, 1.0, dims=("channel",))
        lam = pmd.Gamma("lam", 1.0, 1.0, dims=("channel",))
        pmd.Deterministic("channel_contribution", ((alpha * X) / (X + lam)).sum())

    return model, _constant_saturation_idata()


def _date_less_deterministic_model_and_idata() -> tuple[pm.Model, xr.DataTree]:
    """A model whose response has named dimensions but no ``date`` dimension."""
    rng = np.random.default_rng(5)
    coords = {"date": np.arange(6), "channel": list("abcd")}
    with pm.Model(coords=coords) as model:
        X = pmd.Data(
            "channel_data", rng.gamma(2.0, 1.0, (6, 4)), dims=("date", "channel")
        )
        alpha = pmd.Gamma("alpha", 1.0, 1.0, dims=("channel",))
        lam = pmd.Gamma("lam", 1.0, 1.0, dims=("channel",))
        pmd.Deterministic(
            "channel_contribution",
            ((alpha * X) / (X + lam)).sum(dim="date"),
            dims=("channel",),
        )

    return model, _constant_saturation_idata()


@pytest.mark.parametrize(
    ("builder", "use_mask", "match"),
    [
        pytest.param(
            _plain_deterministic_model_and_idata,
            False,
            "has no dimensions",
            id="plain",
        ),
        pytest.param(
            _scalar_deterministic_model_and_idata,
            False,
            "no 'date' dimension",
            id="scalar",
        ),
        pytest.param(
            _scalar_deterministic_model_and_idata,
            True,
            "no 'date' dimension",
            id="scalar-masked",
        ),
        pytest.param(
            _date_less_deterministic_model_and_idata,
            False,
            "no 'date' dimension",
            id="date-less",
        ),
    ],
)
def test_run_sweep_dimensionless_response_raises_a_clear_error(
    builder, use_mask, match
):
    """A response without dims explains what it needs instead of an internal error."""
    model, idata = builder()
    response_mask = np.ones((6, 4), dtype=bool) if use_mask else None

    with pytest.raises(ValueError, match=match):
        SensitivityAnalysis(model, idata).run_sweep(
            "channel_data",
            np.linspace(0.5, 1.5, 4),
            var_names="channel_contribution",
            response_mask=response_mask,
        )


def test_run_sweep_frees_draws_the_graph_does_not_read(simple_model_and_idata):
    """A cached entry pins only the posterior draws its graph reads.

    The warmed entry is never rebound: the second call asks a different
    structure, so it builds its own entry and the first one stays cached.  A
    posterior variable no graph reads must be collectable once the tree that
    carried it is gone, which a full-posterior entry would prevent.
    """
    model, _ = simple_model_and_idata
    posterior = _channel_varying_posterior(model).posterior.to_dataset()
    posterior["unused"] = xr.DataArray(
        np.full((1, 5, 4), 3.0),
        dims=["chain", "draw", "channel"],
        coords={"channel": list(model.coords["channel"])},
    )
    idata = xr.DataTree.from_dict({"/posterior": posterior})
    freed = weakref.ref(idata.posterior["unused"].values)
    sweeps = np.linspace(0.5, 1.5, 4)

    analysis = SensitivityAnalysis(model, idata)
    analysis.run_sweep(
        "channel_data",
        sweeps,
        var_names="channel_contribution",
        sweep_type="multiplicative",
    )
    assert freed() is not None

    analysis.idata = xr.DataTree.from_dict(
        {"/posterior": posterior.drop_vars("unused")}
    )
    analysis.run_sweep(
        "channel_data", sweeps, var_names="channel_contribution", sweep_type="additive"
    )
    del posterior, idata
    gc.collect()

    assert freed() is None


def test_run_sweep_reflects_set_data_between_calls(sensitivity):
    """``pm.set_data`` between calls must change the sweep, as it does today."""
    sweeps = np.linspace(0.5, 1.5, 4)
    kwargs = dict(_SENSITIVITY_KWARGS)
    before = sensitivity.run_sweep("channel_data", sweeps, **kwargs)

    channel_data = np.asarray(sensitivity.model["channel_data"].get_value()) * 1.5
    pm.set_data({"channel_data": channel_data}, model=sensitivity.model)
    after = sensitivity.run_sweep("channel_data", sweeps, **kwargs)

    xr.testing.assert_allclose(
        after,
        _expected_saturation_sweep(
            sensitivity.model, channel_data, sweeps, sweep_type="multiplicative"
        ),
    )
    assert not np.allclose(after.values, before.values)


@pytest.mark.parametrize("target", ["mmm", "analysis"])
def test_run_sweep_replacing_idata_between_calls_matches_an_empty_cache(
    multidim_mmm, target
):
    """A posterior replaced between calls reaches the next sweep.

    ``target="mmm"`` assigns ``mmm.idata``, which the next ``mmm.sensitivity``
    access reads; ``target="analysis"`` assigns the idata of the analysis
    instance holding the warm cache.  Either way the call must answer what an
    empty cache answers for the replacement, and differ from the warm result.
    """
    sweeps = np.linspace(0.5, 1.5, 4)
    analysis = multidim_mmm.sensitivity
    before = analysis.run_sweep("channel_data", sweeps, **_SENSITIVITY_KWARGS)

    replacement = multidim_mmm.idata.copy(deep=True)
    replacement.posterior["saturation_beta"] = (
        replacement.posterior["saturation_beta"] * 0.5
    )
    if target == "mmm":
        multidim_mmm.idata = replacement
        analysis = multidim_mmm.sensitivity
    else:
        analysis.idata = replacement
    after = analysis.run_sweep("channel_data", sweeps, **_SENSITIVITY_KWARGS)

    xr.testing.assert_allclose(
        after,
        SensitivityAnalysis(multidim_mmm.model, replacement).run_sweep(
            "channel_data", sweeps, **_SENSITIVITY_KWARGS
        ),
    )
    assert not np.allclose(after.values, before.values)


def _channel_varying_posterior(model: pm.Model, n_draws: int = 5) -> xr.DataTree:
    """Posterior draws whose ``alpha`` and ``lam`` differ per channel label."""
    channel = list(model.coords["channel"])
    alpha = np.linspace(0.4, 1.6, len(channel))
    lam = np.linspace(0.6, 2.4, len(channel))

    def draws(values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(
            np.broadcast_to(values, (1, n_draws, len(channel))).copy(),
            dims=["chain", "draw", "channel"],
            coords={"channel": channel},
        )

    return xr.DataTree.from_dict(
        {"/posterior": xr.Dataset({"alpha": draws(alpha), "lam": draws(lam)})}
    )


def _reverse_channel_coords(idata: xr.DataTree) -> xr.DataTree:
    """Reverse the stored channel order while keeping each label's values."""
    reversed_vars = {}
    for name, posterior_da in idata.posterior.data_vars.items():
        reversed_vars[name] = posterior_da.isel(channel=slice(None, None, -1))
    return xr.DataTree.from_dict({"/posterior": xr.Dataset(reversed_vars)})


def _with_renamed_channel_labels(idata: xr.DataTree) -> xr.DataTree:
    """Give the ``channel`` dim fresh labels, keeping every value in place."""
    labels = [f"renamed-{index}" for index in range(idata.posterior.sizes["channel"])]
    posterior = idata.posterior.to_dataset().assign_coords(channel=labels)
    return xr.DataTree.from_dict({"/posterior": posterior})


def _without_channel_labels(idata: xr.DataTree) -> xr.DataTree:
    """Drop the ``channel`` labels, keeping every value in place."""
    posterior = idata.posterior.to_dataset().drop_vars("channel")
    return xr.DataTree.from_dict({"/posterior": posterior})


def _as_float32(idata: xr.DataTree) -> xr.DataTree:
    """Cast every posterior variable to ``float32``."""
    posterior = idata.posterior.to_dataset().astype("float32")
    return xr.DataTree.from_dict({"/posterior": posterior})


def _with_draws(idata: xr.DataTree, chains: int, draws: int) -> xr.DataTree:
    """Resample the sample dims to ``chains`` x ``draws``, cycling the originals."""
    posterior = idata.posterior.to_dataset()
    posterior = posterior.isel(
        chain=np.arange(chains) % posterior.sizes["chain"],
        draw=np.arange(draws) % posterior.sizes["draw"],
    ).assign_coords(chain=np.arange(chains), draw=np.arange(draws))
    return xr.DataTree.from_dict({"/posterior": posterior})


@pytest.mark.parametrize(
    "replace",
    [
        _with_renamed_channel_labels,
        _without_channel_labels,
        _as_float32,
        lambda idata: _with_draws(idata, 1, 1),
        lambda idata: _with_draws(idata, 4, 9),
    ],
    ids=["relabelled", "unlabeled", "float32", "one-draw", "4x9-draws"],
)
def test_run_sweep_replacement_posterior_matches_an_empty_cache(
    simple_model_and_idata, replace
):
    """A replacement posterior gives an empty cache's result.

    A renamed or unlabeled channel axis cannot be checked against the cached
    entry's construction layout, so the entry is dropped and rebuilt for this
    posterior; float32 values and another number of draws rebind the entry in
    place.  A later call with the original posterior must still work.
    """
    model, _ = simple_model_and_idata
    original = _channel_varying_posterior(model)
    replacement = replace(original)
    sweeps = np.linspace(0.5, 1.5, 4)
    kwargs = dict(_SENSITIVITY_KWARGS)

    analysis = SensitivityAnalysis(model, original)
    first = analysis.run_sweep("channel_data", sweeps, **kwargs)

    analysis.idata = replacement
    warm = analysis.run_sweep("channel_data", sweeps, **kwargs)
    empty = SensitivityAnalysis(model, replacement).run_sweep(
        "channel_data", sweeps, **kwargs
    )

    xr.testing.assert_allclose(warm, empty)

    analysis.idata = original
    again = analysis.run_sweep("channel_data", sweeps, **kwargs)

    xr.testing.assert_allclose(again, first)


def test_run_sweep_renamed_then_reversed_posterior_matches_an_empty_cache(
    simple_model_and_idata,
):
    """Renamed labels stored in another order give the empty cache's result.

    A cached entry built on renamed labels would realign a permutation of them
    by label, while an empty cache reads the stored order by position.  The
    entry must refuse the rebind and rebuild instead.
    """
    model, _ = simple_model_and_idata
    renamed = _with_renamed_channel_labels(_channel_varying_posterior(model))
    renamed_reversed = _reverse_channel_coords(renamed)
    sweeps = np.linspace(0.5, 1.5, 4)
    kwargs = dict(_SENSITIVITY_KWARGS)

    analysis = SensitivityAnalysis(model, renamed)
    analysis.run_sweep("channel_data", sweeps, **kwargs)
    analysis.idata = renamed_reversed
    warm = analysis.run_sweep("channel_data", sweeps, **kwargs)
    empty = SensitivityAnalysis(model, renamed_reversed).run_sweep(
        "channel_data", sweeps, **kwargs
    )

    xr.testing.assert_allclose(warm, empty)


def test_run_sweep_an_entry_released_after_a_discard_is_not_reused(
    simple_model_and_idata, monkeypatch
):
    """A discard is not followed by a second look in the cache.

    Another call can release an entry for the key between the discard and the
    rebuild.  Consulting it again would answer with its layout or fail, while
    an empty cache builds the entry this call asked for.
    """
    model, _ = simple_model_and_idata
    original = _channel_varying_posterior(model)
    unlabeled = _without_channel_labels(original)
    sweeps = np.linspace(0.5, 1.5, 4)
    kwargs = dict(_SENSITIVITY_KWARGS)
    cache = SweepCache()
    analysis = SensitivityAnalysis(model, original, _cache=cache)
    analysis.run_sweep("channel_data", sweeps, **kwargs)
    key = next(iter(cache._entries))
    discard = Lease.discard

    def discard_then_another_call_releases(lease: Lease[SweepEntry]) -> None:
        discard(lease)
        # A concurrent call on the fitted posterior finishes right here.
        cache.put(key, SweepEntry.build(model, original, key))

    monkeypatch.setattr(Lease, "discard", discard_then_another_call_releases)
    analysis.idata = unlabeled
    warm = analysis.run_sweep("channel_data", sweeps, **kwargs)
    monkeypatch.setattr(Lease, "discard", discard)
    empty = SensitivityAnalysis(model, unlabeled).run_sweep(
        "channel_data", sweeps, **kwargs
    )

    xr.testing.assert_allclose(warm, empty)


def test_run_sweep_a_failed_rebind_is_not_kept_in_the_cache(
    simple_model_and_idata, sweep_compile_counter
):
    """The entry rebuilt after a failed rebind is cached, not rebuilt every call."""
    model, _ = simple_model_and_idata
    original = _channel_varying_posterior(model)
    replacement = _with_renamed_channel_labels(original)
    sweeps = np.linspace(0.5, 1.5, 4)
    kwargs = dict(_SENSITIVITY_KWARGS)

    analysis = SensitivityAnalysis(model, original)
    compiles_before = len(sweep_compile_counter)
    analysis.run_sweep("channel_data", sweeps, **kwargs)
    analysis.idata = replacement
    analysis.run_sweep("channel_data", sweeps, **kwargs)

    # the first build and the rebuild
    assert len(sweep_compile_counter) - compiles_before == 2

    analysis.run_sweep("channel_data", sweeps, **kwargs)

    assert len(sweep_compile_counter) - compiles_before == 2


def test_run_sweep_a_posterior_missing_a_variable_keeps_the_cached_entry(
    simple_model_and_idata, sweep_compile_counter
):
    """A missing variable fails as on a fresh build and leaves the entry cached."""
    model, _ = simple_model_and_idata
    original = _channel_varying_posterior(model)
    sweeps = np.linspace(0.5, 1.5, 4)
    kwargs = dict(_SENSITIVITY_KWARGS)

    analysis = SensitivityAnalysis(model, original)
    compiles_before = len(sweep_compile_counter)
    first = analysis.run_sweep("channel_data", sweeps, **kwargs)
    analysis.idata = xr.DataTree.from_dict(
        {"posterior": original.posterior.to_dataset().drop_vars("alpha")}
    )
    with pytest.raises(KeyError, match=r"missing variables.*alpha"):
        analysis.run_sweep("channel_data", sweeps, **kwargs)
    analysis.idata = original

    xr.testing.assert_allclose(
        analysis.run_sweep("channel_data", sweeps, **kwargs), first
    )
    assert len(sweep_compile_counter) - compiles_before == 1


def test_run_sweep_a_posterior_missing_a_variable_names_it_warm_and_cold(
    simple_model_and_idata,
):
    """A missing variable raises the same exact message on either cache.

    An empty cache builds the entry and a warm one rebinds a cached entry: both
    name the variable the same way.
    """
    model, _ = simple_model_and_idata
    original = _channel_varying_posterior(model)
    missing = xr.DataTree.from_dict(
        {"posterior": original.posterior.to_dataset().drop_vars("alpha")}
    )
    sweeps = np.linspace(0.5, 1.5, 4)
    expected = "Posterior is missing variables this graph reads: ['alpha']"

    warm = SensitivityAnalysis(model, original)
    warm.run_sweep("channel_data", sweeps, **_SENSITIVITY_KWARGS)
    warm.idata = missing

    with pytest.raises(KeyError) as cold:
        SensitivityAnalysis(model, missing).run_sweep(
            "channel_data", sweeps, **_SENSITIVITY_KWARGS
        )
    with pytest.raises(KeyError) as warm_error:
        warm.run_sweep("channel_data", sweeps, **_SENSITIVITY_KWARGS)

    assert cold.value.args == (expected,)
    assert warm_error.value.args == (expected,)


def test_run_sweep_fallback_realigns_a_permuted_posterior_by_label():
    """Both paths realign a permuted posterior by label, not by position.

    The permuted posterior carries the same label-to-draw mapping as the
    unpermuted one, only stored in reverse order.  The cached path's
    ``ConditionedModel`` and the per-call path's explicit reordering both give
    the closed-form label-correct result, which the positional read does not.
    """
    unclonable, _ = _initval_model_and_idata()
    clonable, _ = _initval_model_and_idata(with_initval=False)
    idata = _channel_varying_posterior(clonable)
    permuted_idata = _reverse_channel_coords(idata)
    sweeps = np.linspace(0.5, 1.5, 4)
    channel_data = np.asarray(clonable["channel_data"].get_value())

    label_correct = _expected_saturation_sweep(
        clonable,
        channel_data,
        sweeps,
        alpha=idata.posterior["alpha"].isel(chain=0).values,
        lam=idata.posterior["lam"].isel(chain=0).values,
    )
    cached_permuted = SensitivityAnalysis(clonable, permuted_idata).run_sweep(
        "channel_data", sweeps, **_SENSITIVITY_KWARGS
    )
    fallback_permuted = SensitivityAnalysis(unclonable, permuted_idata).run_sweep(
        "channel_data", sweeps, **_SENSITIVITY_KWARGS
    )

    xr.testing.assert_allclose(cached_permuted, label_correct)
    xr.testing.assert_allclose(fallback_permuted, label_correct)

    # Reading the stored order by position swaps each channel's alpha and lam.
    positional = _expected_saturation_sweep(
        clonable,
        channel_data,
        sweeps,
        alpha=permuted_idata.posterior["alpha"].isel(chain=0).values,
        lam=permuted_idata.posterior["lam"].isel(chain=0).values,
    )

    assert not np.allclose(positional.values, label_correct.values)


@pytest.fixture
def sweep_compile_counter(monkeypatch):
    """Record every PyTensor compilation made while the test runs.

    ``pytensor.function`` builds exactly one
    :class:`~pytensor.compile.maker.FunctionMaker`, so counting the maker's
    constructions counts compiles without patching each module that imports
    ``function``.
    """
    compiles: list[int] = []
    real_init = FunctionMaker.__init__

    def counting_init(self, *args, **kwargs):
        compiles.append(1)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(FunctionMaker, "__init__", counting_init)
    return compiles


def test_run_sweep_compiles_once_per_call(sensitivity, sweep_compile_counter):
    """The first call compiles one sweep and a repeat call reuses it."""
    sweeps = np.linspace(0.5, 1.5, 4)
    kwargs = dict(_SENSITIVITY_KWARGS)

    compiles_before = len(sweep_compile_counter)
    sensitivity.run_sweep("channel_data", sweeps, **kwargs)
    first_call_compiles = len(sweep_compile_counter) - compiles_before
    sensitivity.run_sweep("channel_data", sweeps, **kwargs)
    repeat_compiles = len(sweep_compile_counter) - compiles_before - first_call_compiles

    assert first_call_compiles == 1, f"first call compiled {first_call_compiles} times"
    assert repeat_compiles == 0, f"repeat call compiled {repeat_compiles} times"


# ---------------------------------------------------------------------------
# Cache key and fallback coverage for the sweep cache.
# ---------------------------------------------------------------------------

_KEY_CHANGE_COMPILES = (
    ("sweep_length", 0),
    ("mask_values", 0),
    ("posterior_sample_fraction", 0),
    ("var_names", 1),
    ("var_input", 1),
    ("sweep_type", 1),
    ("masked_response_shape", 1),
)


def _two_input_model_and_idata() -> tuple[pm.Model, xr.DataTree]:
    """A model whose response reads two inputs and has two response shapes.

    Two inputs let the ``var_input`` key field change independently, and the
    date-only deterministic lets the response dims, and with them the mask
    shape, change.
    """
    rng = np.random.default_rng(2024)
    coords = {"date": np.arange(6), "channel": list("abcd")}
    channel_data = rng.gamma(2.0, 1.0, size=(6, 4)).astype(np.float64)
    channel_data_2 = rng.gamma(2.0, 1.0, size=(6, 4)).astype(np.float64)

    with pm.Model(coords=coords) as model:
        first = pmd.Data("channel_data", channel_data, dims=("date", "channel"))
        second = pmd.Data("channel_data_2", channel_data_2, dims=("date", "channel"))
        alpha = pmd.Gamma("alpha", 1.0, 1.0, dims=("channel",))
        lam = pmd.Gamma("lam", 1.0, 1.0, dims=("channel",))
        response = (alpha * first) / (first + lam) + (alpha * second) / (second + lam)
        pmd.Deterministic("channel_contribution", response, dims=("date", "channel"))
        pmd.Deterministic(
            "date_contribution", response.sum(dim="channel"), dims=("date",)
        )

    idata = xr.DataTree.from_dict(
        {
            "/posterior": xr.Dataset(
                {
                    "alpha": xr.DataArray(
                        np.full((1, 5, 4), 0.8),
                        dims=["chain", "draw", "channel"],
                        coords={"channel": list("abcd")},
                    ),
                    "lam": xr.DataArray(
                        np.full((1, 5, 4), 1.2),
                        dims=["chain", "draw", "channel"],
                        coords={"channel": list("abcd")},
                    ),
                }
            ),
        }
    )
    return model, idata


def _key_change_calls(case: str) -> tuple[dict, dict]:
    """Return the two ``run_sweep`` calls of one cache-key change case."""
    sweeps = np.linspace(0.5, 1.5, 4)
    base = {
        "var_input": "channel_data",
        "var_names": "channel_contribution",
        "sweep_values": sweeps,
        "sweep_type": "multiplicative",
    }

    match case:
        case "sweep_length":
            return dict(base), {**base, "sweep_values": np.linspace(0.5, 1.5, 6)}
        case "mask_values":
            first_mask = np.zeros((6, 4), dtype=bool)
            first_mask[:, [0, 1]] = True
            second_mask = np.zeros((6, 4), dtype=bool)
            second_mask[:, [2, 3]] = True
            return (
                {**base, "response_mask": first_mask},
                {**base, "response_mask": second_mask},
            )
        case "posterior_sample_fraction":
            return dict(base), {**base, "posterior_sample_fraction": 0.6}
        case "var_names":
            return dict(base), {**base, "var_names": "date_contribution"}
        case "var_input":
            return dict(base), {**base, "var_input": "channel_data_2"}
        case "sweep_type":
            return dict(base), {**base, "sweep_type": "additive"}
        case "masked_response_shape":
            channel_mask = np.zeros((6, 4), dtype=bool)
            channel_mask[:, [0]] = True
            date_mask = np.zeros((6,), dtype=bool)
            date_mask[:3] = True
            return (
                {**base, "response_mask": channel_mask},
                {
                    **base,
                    "var_names": "date_contribution",
                    "response_mask": date_mask,
                },
            )
        case _:
            raise ValueError(f"Unknown key change case {case!r}")


@pytest.mark.parametrize(("case", "expected_compiles"), _KEY_CHANGE_COMPILES)
def test_run_sweep_recompiles_only_on_key_change(
    case, expected_compiles, sweep_compile_counter
):
    """A new grid, mask values or fraction reuses the compiled sweep.

    A different variable, input, sweep type or mask shape is a different
    compiled graph, so it compiles once more.
    """
    model, idata = _two_input_model_and_idata()
    analysis = SensitivityAnalysis(model, idata)
    first_call, second_call = _key_change_calls(case)

    compiles_before = len(sweep_compile_counter)
    first = analysis.run_sweep(**first_call)
    compiles_after_first = len(sweep_compile_counter) - compiles_before
    second = analysis.run_sweep(**second_call)

    assert compiles_after_first == 1
    assert len(sweep_compile_counter) - compiles_before - compiles_after_first == (
        expected_compiles
    )
    assert isinstance(first, xr.DataArray)
    assert isinstance(second, xr.DataArray)

    # A reused or a rebuilt entry must honour this call's grid, mask and draws.
    fraction = second_call.get("posterior_sample_fraction", 1.0)
    expected = _expected_two_input_sweep(
        model,
        second_call["sweep_values"],
        var_input=second_call["var_input"],
        var_names=second_call["var_names"],
        sweep_type=second_call["sweep_type"],
        response_mask=second_call.get("response_mask"),
        sample_size=5 if fraction == 1.0 else 2,
    )
    xr.testing.assert_allclose(second, expected)


def _initval_model_and_idata(
    *, with_initval: bool = True
) -> tuple[pm.Model, xr.DataTree]:
    """A model with a non-default ``initval``, which ``clone_model`` refuses.

    With ``with_initval=False`` the same model is clonable, so a test can
    compare the per-call path against the cached path on the same graph.
    """
    rng = np.random.default_rng(11)
    coords = {"date": np.arange(6), "channel": list("abcd")}
    channel_data = rng.gamma(2.0, 1.0, size=(6, 4)).astype(np.float64)

    with pm.Model(coords=coords) as model:
        X = pmd.Data("channel_data", channel_data, dims=("date", "channel"))
        alpha = pmd.Gamma(
            "alpha",
            1.0,
            1.0,
            dims=("channel",),
            initval=np.full(4, 0.5) if with_initval else None,
        )
        lam = pmd.Gamma("lam", 1.0, 1.0, dims=("channel",))
        pmd.Deterministic(
            "channel_contribution",
            (alpha * X) / (X + lam),
            dims=("date", "channel"),
        )

    idata = xr.DataTree.from_dict(
        {
            "/posterior": xr.Dataset(
                {
                    "alpha": xr.DataArray(
                        np.full((1, 5, 4), 0.8),
                        dims=["chain", "draw", "channel"],
                        coords={"channel": list("abcd")},
                    ),
                    "lam": xr.DataArray(
                        np.full((1, 5, 4), 1.2),
                        dims=["chain", "draw", "channel"],
                        coords={"channel": list("abcd")},
                    ),
                }
            ),
        }
    )
    return model, idata


def test_run_sweep_initval_model_falls_back(sweep_compile_counter):
    """A model ``ConditionedModel`` cannot clone still sweeps, per call.

    The sweep compiles on every call on this path, and the result equals the
    closed form, so the fallback is behaviour-preserving.
    """
    model, idata = _initval_model_and_idata()
    with pytest.raises(ModelCloneError):
        ConditionedModel(model, idata)
    analysis = SensitivityAnalysis(model, idata)
    sweeps = np.linspace(0.5, 1.5, 4)
    kwargs = dict(_SENSITIVITY_KWARGS)

    compiles_before = len(sweep_compile_counter)
    first = analysis.run_sweep("channel_data", sweeps, **kwargs)
    second = analysis.run_sweep("channel_data", sweeps, **kwargs)

    expected = _expected_saturation_sweep(
        model,
        np.asarray(model["channel_data"].get_value()),
        sweeps,
        sweep_type="multiplicative",
    )
    xr.testing.assert_allclose(first, expected)
    xr.testing.assert_allclose(second, first)
    assert len(sweep_compile_counter) - compiles_before == 2


def _three_channel_posterior() -> xr.DataTree:
    """Draws over three channels for a four-channel model.

    The entry builds fine and the compiled function then refuses the response's
    channel length when it is executed.
    """
    channel = ["a", "b", "c"]
    return xr.DataTree.from_dict(
        {
            "/posterior": xr.Dataset(
                {
                    "alpha": xr.DataArray(
                        np.full((1, 5, 3), 0.8),
                        dims=["chain", "draw", "channel"],
                        coords={"channel": channel},
                    ),
                    "lam": xr.DataArray(
                        np.full((1, 5, 3), 1.2),
                        dims=["chain", "draw", "channel"],
                        coords={"channel": channel},
                    ),
                }
            ),
        }
    )


def test_run_sweep_a_fresh_entry_that_fails_to_evaluate_is_not_cached(
    simple_model_and_idata, sweep_compile_counter
):
    """A broken fresh entry is dropped and leaves another key's entry cached."""
    model, _ = simple_model_and_idata
    sweeps = np.linspace(0.5, 1.5, 4)
    cache = SweepCache()
    good = SensitivityAnalysis(model, _channel_varying_posterior(model), _cache=cache)
    good.run_sweep("channel_data", sweeps, sweep_type="multiplicative")
    assert len(cache) == 1
    compiles = len(sweep_compile_counter)

    broken = SensitivityAnalysis(model, _three_channel_posterior(), _cache=cache)
    with pytest.raises(ValueError, match="incompatible shape"):
        broken.run_sweep("channel_data", sweeps, sweep_type="additive")

    assert len(cache) == 1
    assert len(sweep_compile_counter) == compiles + 1
    # The good entry was not evicted: the warm call reuses it.
    good.run_sweep("channel_data", sweeps, sweep_type="multiplicative")
    assert len(sweep_compile_counter) == compiles + 1


def test_run_sweep_a_cached_entry_that_fails_to_evaluate_stays_cached(
    simple_model_and_idata, monkeypatch, sweep_compile_counter
):
    """A cached entry whose evaluation fails is kept: the call's arguments failed."""
    model, _ = simple_model_and_idata
    sweeps = np.linspace(0.5, 1.5, 4)
    kwargs = dict(_SENSITIVITY_KWARGS)
    analysis = SensitivityAnalysis(model, _channel_varying_posterior(model))
    analysis.run_sweep("channel_data", sweeps, **kwargs)
    entry = next(iter(analysis._cache._entries.values()))
    compiles = len(sweep_compile_counter)

    def failing_evaluate(sweep_values: np.ndarray, mask_values: np.ndarray | None):
        raise ValueError("evaluation failed")

    monkeypatch.setattr(entry, "evaluate", failing_evaluate)

    with pytest.raises(ValueError, match="evaluation failed"):
        analysis.run_sweep("channel_data", sweeps, **kwargs)

    assert len(analysis._cache) == 1
    # The second call leases the same entry again rather than compiling.
    with pytest.raises(ValueError, match="evaluation failed"):
        analysis.run_sweep("channel_data", sweeps, **kwargs)
    assert len(sweep_compile_counter) == compiles


def test_run_sweep_a_clone_failure_during_the_rebuild_falls_back(
    simple_model_and_idata, monkeypatch
):
    """A model that becomes unclonable while rebuilding answers per call.

    The cached entry cannot take the new labels, so it is dropped; the rebuild
    then cannot clone the model, which must fall back to the per-call path as an
    empty cache does, not surface the clone failure.
    """
    model, _ = simple_model_and_idata
    original = _channel_varying_posterior(model)
    replacement = _with_renamed_channel_labels(original)
    sweeps = np.linspace(0.5, 1.5, 4)
    kwargs = dict(_SENSITIVITY_KWARGS)
    analysis = SensitivityAnalysis(model, original)

    real_conditioned = ConditionedModel
    builds: list[int] = []

    def clone_once_then_refuse(*args: object, **kwargs: object):
        builds.append(1)
        if len(builds) > 1:
            raise ModelCloneError("cannot clone")
        return real_conditioned(*args, **kwargs)

    monkeypatch.setattr(
        "pymc_marketing.mmm.sweep_graph.ConditionedModel", clone_once_then_refuse
    )
    analysis.run_sweep("channel_data", sweeps, **kwargs)
    analysis.idata = replacement
    fallback = analysis.run_sweep("channel_data", sweeps, **kwargs)

    xr.testing.assert_allclose(
        fallback,
        _expected_saturation_sweep(
            model,
            np.asarray(model["channel_data"].get_value()),
            sweeps,
            alpha=replacement.posterior["alpha"].isel(chain=0).values,
            lam=replacement.posterior["lam"].isel(chain=0).values,
        ),
    )
    assert len(builds) == 2
    assert len(analysis._cache) == 0


def _draw_varying_posterior(model: pm.Model, n_draws: int = 5) -> xr.DataTree:
    """Posterior draws whose ``alpha`` and ``lam`` differ on every draw index."""
    n_channels = len(model.coords["channel"])
    steps = np.arange(n_draws, dtype=np.float64)[None, :, None]
    ones = np.ones((1, n_draws, n_channels))
    return xr.DataTree.from_dict(
        {
            "/posterior": xr.Dataset(
                {
                    "alpha": xr.DataArray(
                        (0.4 + 0.3 * steps) * ones, dims=["chain", "draw", "channel"]
                    ),
                    "lam": xr.DataArray(
                        (0.6 + 0.5 * steps) * ones, dims=["chain", "draw", "channel"]
                    ),
                }
            ),
        }
    )


def test_run_sweep_posterior_sample_fraction_selects_the_expected_draws(
    simple_model_and_idata,
):
    """Five draws at fraction 0.6 keep the first and last, and use their values."""
    model, _ = simple_model_and_idata
    idata = _draw_varying_posterior(model)
    sweeps = np.linspace(0.5, 1.5, 4)
    selected = [0, 4]

    result = SensitivityAnalysis(model, idata).run_sweep(
        "channel_data",
        sweeps,
        var_names="channel_contribution",
        sweep_type="multiplicative",
        posterior_sample_fraction=0.6,
    )
    expected = _expected_saturation_sweep(
        model,
        np.asarray(model["channel_data"].get_value()),
        sweeps,
        alpha=idata.posterior["alpha"].isel(chain=0, draw=selected).values,
        lam=idata.posterior["lam"].isel(chain=0, draw=selected).values,
        sample_size=len(selected),
    )

    assert result.sizes["sample"] == len(selected)
    xr.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# MMM-owned shared cache and lifecycle.
# ---------------------------------------------------------------------------


def test_mmm_sensitivity_run_sweep_compiles_once_across_accesses(
    multidim_mmm, sweep_compile_counter
):
    """Two accesses to ``mmm.sensitivity`` share one compiled sweep."""
    sweeps = np.linspace(0.5, 1.5, 4)

    compiles_before = len(sweep_compile_counter)
    first = multidim_mmm.sensitivity.run_sweep(
        "channel_data", sweeps, **_SENSITIVITY_KWARGS
    )
    first_call_compiles = len(sweep_compile_counter) - compiles_before
    second = multidim_mmm.sensitivity.run_sweep(
        "channel_data", sweeps, **_SENSITIVITY_KWARGS
    )
    repeat_compiles = len(sweep_compile_counter) - compiles_before - first_call_compiles

    assert first_call_compiles == 1, f"first call compiled {first_call_compiles} times"
    assert repeat_compiles == 0, f"repeat call compiled {repeat_compiles} times"
    xr.testing.assert_allclose(second, first)


_LIFECYCLE_SITES = ("fit", "build_model", "build_from_idata")


def _invoke_lifecycle_site(mmm: MMM, site: str, X: pd.DataFrame, y: pd.Series) -> None:
    """Rebuild or refit ``mmm`` at one of the sites that must clear the cache."""
    match site:
        case "fit":
            # The seed keeps the refit posterior fixed, so the results stay comparable.
            mmm.fit(X, y, random_seed=42)
        case "build_model":
            mmm.build_model(X, y)
        case "build_from_idata":
            mmm.build_from_idata(mmm.idata)
        case _:
            raise ValueError(f"Unknown lifecycle site {site!r}")


@pytest.mark.parametrize("site", _LIFECYCLE_SITES)
def test_mmm_lifecycle_site_clears_both_caches(
    multidim_mmm, df, site, sweep_compile_counter, monkeypatch
):
    """``fit``, ``build_model`` and ``build_from_idata`` each drop both caches."""
    mmm = multidim_mmm
    if site == "build_from_idata":
        # build_model clears both caches too; stub it so only this site's own
        # up-front clear can empty them.
        monkeypatch.setattr(mmm, "build_model", lambda X, y: None)
    X = df.drop(columns=["y"]).copy()
    y = df["y"].copy()
    if site == "fit":
        # Put both sides of the site on the same seeded posterior.
        mmm.fit(X, y, random_seed=42)

    sweeps = np.linspace(0.5, 1.5, 4)
    before = mmm.sensitivity.run_sweep("channel_data", sweeps, **_SENSITIVITY_KWARGS)
    mmm._incrementality_evaluator_cache().put(object(), object())
    assert len(mmm._sensitivity_sweep_cache()) == 1
    assert len(mmm._incrementality_evaluator_cache()) == 1

    _invoke_lifecycle_site(mmm, site, X, y)

    assert len(mmm._sensitivity_sweep_cache()) == 0
    assert len(mmm._incrementality_evaluator_cache()) == 0
    compiles_before_sweep = len(sweep_compile_counter)
    after = mmm.sensitivity.run_sweep("channel_data", sweeps, **_SENSITIVITY_KWARGS)

    assert len(sweep_compile_counter) - compiles_before_sweep == 1
    xr.testing.assert_allclose(after, before)


_SWEEP_CACHE_COPY_METHODS = (
    pytest.param(copy.deepcopy, id="deepcopy"),
    pytest.param(
        lambda mmm: cloudpickle.loads(cloudpickle.dumps(mmm)),
        id="pickle",
    ),
)


@pytest.mark.parametrize("copy_mmm", _SWEEP_CACHE_COPY_METHODS)
def test_mmm_deepcopy_and_pickle_drop_the_sweep_cache(multidim_mmm, copy_mmm):
    """A copy of a warm model carries no compiled sweep but still sweeps.

    Plain ``pickle`` cannot serialize an MMM at all -- a local function inside
    ``create_dim_handler`` is unpicklable, unrelated to this cache -- so the
    round trip uses ``cloudpickle``, which PyMC itself depends on.
    """
    sweeps = np.linspace(0.5, 1.5, 4)
    warm = multidim_mmm.sensitivity.run_sweep(
        "channel_data", sweeps, **_SENSITIVITY_KWARGS
    )
    assert len(multidim_mmm._sensitivity_sweep_cache()) == 1

    clone = copy_mmm(multidim_mmm)

    assert len(clone._sensitivity_sweep_cache()) == 0
    assert len(multidim_mmm._sensitivity_sweep_cache()) == 1
    copied = clone.sensitivity.run_sweep("channel_data", sweeps, **_SENSITIVITY_KWARGS)
    xr.testing.assert_allclose(copied, warm)


def test_sweep_cache_evicts_the_oldest_entry_and_recompiles(sweep_compile_counter):
    """A fifth key evicts the oldest, which recompiles to the same result."""
    model, idata = _two_input_model_and_idata()
    cache = SweepCache()
    analysis = SensitivityAnalysis(model, idata, _cache=cache)
    sweeps = np.linspace(0.5, 1.5, 4)
    calls = [
        {"var_input": "channel_data", "sweep_type": "multiplicative"},
        {"var_input": "channel_data", "sweep_type": "additive"},
        {"var_input": "channel_data", "sweep_type": "absolute"},
        {"var_input": "channel_data_2", "sweep_type": "multiplicative"},
        {"var_input": "channel_data_2", "sweep_type": "additive"},
    ]

    def run(call: dict) -> xr.DataArray:
        return analysis.run_sweep(
            call["var_input"],
            sweeps,
            var_names="channel_contribution",
            sweep_type=call["sweep_type"],
        )

    results = []
    for call in calls:
        results.append(run(call))
        assert len(cache) <= 4
    assert len(cache) == 4

    compiles_before = len(sweep_compile_counter)
    first_again = run(calls[0])
    assert len(sweep_compile_counter) - compiles_before == 1
    xr.testing.assert_allclose(first_again, results[0])

    compiles_before = len(sweep_compile_counter)
    last_again = run(calls[-1])
    assert len(sweep_compile_counter) == compiles_before
    xr.testing.assert_allclose(last_again, results[-1])
    assert len(cache) == 4
