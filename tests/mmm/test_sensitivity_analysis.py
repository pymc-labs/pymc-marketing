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
import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.tensor as pt
import pytest
import xarray as xr
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.xtensor import as_xtensor

from pymc_marketing.mmm import sensitivity_analysis
from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_marketing.mmm.mmm import MMM
from pymc_marketing.mmm.sensitivity_analysis import SensitivityAnalysis

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


def test_run_sweep_takes_frozen_deterministics_from_the_posterior():
    """A latent path over ``date`` in the response is taken from its posterior draws.

    The path is a random walk built from plain PyTensor variables and wrapped as an
    xtensor, as a latent path from a library outside ``pymc.dims`` is, and the sweep
    cannot recompute it. Frozen, it keeps its posterior draws in every sweep, so the
    swept response is the path summed over dates plus the swept saturation term.
    The path's draws are not the cumulative sum of the steps' draws, so a recomputed
    path would differ.
    """
    rng = np.random.default_rng(123)
    channel_data = rng.gamma(2.0, 1.0, size=(6, 4))
    model, idata = _make_saturation_model_and_idata(channel_data, n_draws=5)
    with model:
        steps = pm.Normal("path_steps", sigma=0.1, dims="date")
        path = pmd.Deterministic("path", as_xtensor(pt.cumsum(steps), dims=("date",)))
        contribution = as_xtensor(
            model["channel_contribution"], dims=("date", "channel")
        )
        pmd.Deterministic("mu", path + contribution.sum("channel"))
    path_draws = rng.normal(0.0, 0.1, size=(1, 5, 6)).cumsum(axis=-1)
    sample_dims = ["chain", "draw", "date"]
    idata = xr.DataTree.from_dict(
        {
            "/posterior": idata.posterior.to_dataset().assign(
                path_steps=(sample_dims, rng.normal(0.0, 0.1, size=(1, 5, 6))),
                path=(sample_dims, path_draws),
            )
        }
    )
    sweep_values = np.linspace(0.5, 1.5, 3)
    sweep = {
        "var_input": "channel_data",
        "var_names": "mu",
        "sweep_values": sweep_values,
    }

    with pytest.raises(AttributeError):
        SensitivityAnalysis(model, idata).run_sweep(**sweep)

    results = SensitivityAnalysis(
        model, idata, frozen_deterministics=["path"]
    ).run_sweep(**sweep)

    swept = sweep_values[:, None, None] * channel_data
    saturation = (0.8 * swept / (swept + 1.2)).sum(axis=(1, 2))
    expected = path_draws[0].sum(axis=-1)[:, None] + saturation
    np.testing.assert_allclose(results.transpose("sample", "sweep"), expected)


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
# Closed-form references for the sweep behaviour.
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


@pytest.mark.parametrize("sweep_type", ["multiplicative", "additive", "absolute"])
def test_multidim_response_mask_keeps_only_the_masked_cells(multidim_mmm, sweep_type):
    """A mask over ``(date, country, channel)`` zeroes every other cell of a sweep.

    The mask keeps channel ``C1`` of country ``B`` on every date, so the masked
    sweep equals the unmasked one in that cell and is zero in the other three.
    """
    sweep_values = np.linspace(0.5, 1.5, 4)
    kwargs = {**_SENSITIVITY_KWARGS, "sweep_type": sweep_type}
    kept = {"country": "B", "channel": "C1"}

    full = multidim_mmm.sensitivity.run_sweep("channel_data", sweep_values, **kwargs)
    masked = multidim_mmm.sensitivity.run_sweep(
        "channel_data", sweep_values, response_mask=_multidim_response_mask(), **kwargs
    )

    assert np.all(full.sel(kept).values != 0.0)
    xr.testing.assert_allclose(masked.sel(kept), full.sel(kept))
    np.testing.assert_array_equal(masked.sel(country="A").values, 0.0)
    np.testing.assert_array_equal(masked.sel(country="B", channel="C2").values, 0.0)


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


def test_run_sweep_a_data_only_response_raises_a_clear_error():
    """A response that reads no posterior variable has no draws to sweep over."""
    model, idata = _data_only_response_model_and_idata()
    analysis = SensitivityAnalysis(model, idata)
    sweeps = np.linspace(0.5, 1.5, 4)

    with pytest.raises(ValueError, match="reads no posterior variable"):
        analysis.run_sweep("channel_data", sweeps, var_names="channel_contribution")


@pytest.mark.parametrize(
    "var_input", ["other_data", "alpha"], ids=["unread-data", "free-variable"]
)
def test_run_sweep_an_input_the_response_graph_does_not_read_raises_a_clear_error(
    simple_model_and_idata, var_input
):
    """Sweeping a variable the conditioned response does not read names it.

    That is a data variable the response ignores, and also a free variable: its
    draws replace it in the graph, so there is nothing left to sweep.
    """
    model, idata = simple_model_and_idata
    with model:
        pmd.Data("other_data", np.ones((6, 4)), dims=("date", "channel"))

    with pytest.raises(ValueError, match=f"does not read '{var_input}'"):
        SensitivityAnalysis(model, idata).run_sweep(
            var_input, np.linspace(0.5, 1.5, 4), var_names="channel_contribution"
        )


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


def _plain_deterministic_model_and_idata(
    dims: tuple[str, ...] | None = None,
) -> tuple[pm.Model, xr.DataTree]:
    """A model whose response is a plain ``pm.Deterministic``, with or without dims."""
    rng = np.random.default_rng(3)
    coords = {"date": np.arange(6), "channel": list("abcd")}
    with pm.Model(coords=coords) as model:
        X = pm.Data(
            "channel_data", rng.gamma(2.0, 1.0, (6, 4)), dims=("date", "channel")
        )
        alpha = pm.Gamma("alpha", 1.0, 1.0, dims=("channel",))
        lam = pm.Gamma("lam", 1.0, 1.0, dims=("channel",))
        pm.Deterministic("channel_contribution", (alpha * X) / (X + lam), dims=dims)

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
            "not a pymc.dims variable",
            id="plain",
        ),
        pytest.param(
            lambda: _plain_deterministic_model_and_idata(dims=("date", "channel")),
            False,
            "not a pymc.dims variable",
            id="plain-with-dims",
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
    """A response the sweep cannot use says what it needs, not an internal error."""
    model, idata = builder()
    response_mask = np.ones((6, 4), dtype=bool) if use_mask else None

    with pytest.raises(ValueError, match=match):
        SensitivityAnalysis(model, idata).run_sweep(
            "channel_data",
            np.linspace(0.5, 1.5, 4),
            var_names="channel_contribution",
            response_mask=response_mask,
        )


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
def test_run_sweep_reads_the_posterior_assigned_between_calls(multidim_mmm, target):
    """A posterior assigned between calls is the one the next sweep reads.

    ``target="mmm"`` assigns ``mmm.idata``, which the next ``mmm.sensitivity``
    access reads; ``target="analysis"`` assigns the idata of the analysis
    instance.  The channel contribution is linear in ``saturation_beta``, so
    halving its draws halves the sweep.
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

    assert np.all(before.values != 0.0)
    xr.testing.assert_allclose(after, 0.5 * before)


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
    ("layout", "sample_size"),
    [
        (_with_renamed_channel_labels, 5),
        (_without_channel_labels, 5),
        (_as_float32, 5),
        (lambda idata: _with_draws(idata, 1, 1), 1),
        (lambda idata: _with_draws(idata, 4, 9), 36),
    ],
    ids=["relabelled", "unlabeled", "float32", "one-draw", "4x9-draws"],
)
def test_run_sweep_posterior_layouts_match_the_closed_form(
    simple_model_and_idata, layout, sample_size
):
    """A posterior stored in another layout gives the closed form of its draws.

    A renamed or unlabeled channel axis is read by position, float32 values are
    cast to the dtype the model expects, and another number of chains and draws
    is bound as given.
    """
    model, _ = simple_model_and_idata
    original = _channel_varying_posterior(model)
    sweeps = np.linspace(0.5, 1.5, 4)

    result = SensitivityAnalysis(model, layout(original)).run_sweep(
        "channel_data", sweeps, **_SENSITIVITY_KWARGS
    )

    expected = _expected_saturation_sweep(
        model,
        np.asarray(model["channel_data"].get_value()),
        sweeps,
        alpha=original.posterior["alpha"].isel(chain=0, draw=0).values,
        lam=original.posterior["lam"].isel(chain=0, draw=0).values,
        sample_size=sample_size,
    )
    xr.testing.assert_allclose(result, expected)


def test_run_sweep_realigns_a_permuted_posterior_by_label():
    """A permuted posterior is realigned by label, not read by position.

    The permuted posterior carries the same label-to-draw mapping as the
    unpermuted one, only stored in reverse order.  The sweep gives the
    closed-form label-correct result, which the positional read does not.
    """
    model, _ = _initval_model_and_idata()
    idata = _channel_varying_posterior(model)
    permuted_idata = _reverse_channel_coords(idata)
    sweeps = np.linspace(0.5, 1.5, 4)
    channel_data = np.asarray(model["channel_data"].get_value())

    label_correct = _expected_saturation_sweep(
        model,
        channel_data,
        sweeps,
        alpha=idata.posterior["alpha"].isel(chain=0).values,
        lam=idata.posterior["lam"].isel(chain=0).values,
    )
    permuted = SensitivityAnalysis(model, permuted_idata).run_sweep(
        "channel_data", sweeps, **_SENSITIVITY_KWARGS
    )

    xr.testing.assert_allclose(permuted, label_correct)

    # Reading the stored order by position swaps each channel's alpha and lam.
    positional = _expected_saturation_sweep(
        model,
        channel_data,
        sweeps,
        alpha=permuted_idata.posterior["alpha"].isel(chain=0).values,
        lam=permuted_idata.posterior["lam"].isel(chain=0).values,
    )

    assert not np.allclose(positional.values, label_correct.values)


def test_run_sweep_reads_duplicated_labels_by_position():
    """A dim whose labels repeat is read by position.

    Two events of the same name give such a dim.  Its labels cannot be
    realigned, so the draws are bound in the order they are stored, and the sweep
    gives the closed form for that order.
    """
    rng = np.random.default_rng(13)
    channel_data = rng.gamma(2.0, 1.0, size=(6, 4))
    coords = {"date": np.arange(6), "channel": ["a", "a", "b", "c"]}
    with pm.Model(coords=coords) as model:
        X = pmd.Data("channel_data", channel_data, dims=("date", "channel"))
        alpha = pmd.Gamma("alpha", 1.0, 1.0, dims=("channel",))
        lam = pmd.Gamma("lam", 1.0, 1.0, dims=("channel",))
        pmd.Deterministic(
            "channel_contribution", (alpha * X) / (X + lam), dims=("date", "channel")
        )
    idata = _channel_varying_posterior(model)
    sweeps = np.linspace(0.5, 1.5, 4)

    result = SensitivityAnalysis(model, idata).run_sweep(
        "channel_data", sweeps, **_SENSITIVITY_KWARGS
    )

    expected = _expected_saturation_sweep(
        model,
        channel_data,
        sweeps,
        alpha=idata.posterior["alpha"].isel(chain=0).values,
        lam=idata.posterior["lam"].isel(chain=0).values,
    )
    xr.testing.assert_allclose(result, expected)


def test_run_sweep_a_posterior_missing_a_variable_names_it(simple_model_and_idata):
    """A posterior without a variable the response reads names it in the error."""
    model, _ = simple_model_and_idata
    original = _channel_varying_posterior(model)
    missing = xr.DataTree.from_dict(
        {"posterior": original.posterior.to_dataset().drop_vars("alpha")}
    )

    with pytest.raises(KeyError) as error:
        SensitivityAnalysis(model, missing).run_sweep(
            "channel_data", np.linspace(0.5, 1.5, 4), **_SENSITIVITY_KWARGS
        )

    assert error.value.args == (
        "Posterior is missing variables this graph reads: ['alpha']",
    )


@pytest.mark.parametrize(
    ("arguments", "error", "match"),
    [
        ({"sweep_type": "quadratic"}, ValueError, "Unknown sweep_type 'quadratic'"),
        ({"sweep_type": ["additive"]}, ValueError, "Unknown sweep_type"),
        ({"var_input": "not_in_the_model"}, KeyError, "not_in_the_model"),
    ],
    ids=["sweep-type", "unhashable-sweep-type", "var-input"],
)
def test_run_sweep_checks_its_arguments_before_it_reads_the_posterior(
    simple_model_and_idata, arguments, error, match
):
    """A wrong argument is reported before the posterior is read.

    The posterior here lacks a variable the response reads, which raises once
    the posterior is read.  The argument's own error comes first.
    """
    model, _ = simple_model_and_idata
    original = _channel_varying_posterior(model)
    missing = xr.DataTree.from_dict(
        {"posterior": original.posterior.to_dataset().drop_vars("alpha")}
    )
    kwargs = {
        "var_input": "channel_data",
        "sweep_values": np.linspace(0.5, 1.5, 4),
        **_SENSITIVITY_KWARGS,
        **arguments,
    }

    with pytest.raises(error, match=match):
        SensitivityAnalysis(model, missing).run_sweep(**kwargs)


@pytest.mark.parametrize("n_channels", [3, 5], ids=["fewer-labels", "more-labels"])
def test_run_sweep_names_the_posterior_variable_that_does_not_fit_the_model(
    simple_model_and_idata, n_channels
):
    """A posterior with another number of labels names the variable and the dim.

    Four channels of data cannot be evaluated against three or five channels of
    draws.  The error says which posterior variable and dim do not fit, not which
    input of the compiled function had the wrong shape.
    """
    model, _ = simple_model_and_idata
    draws = xr.DataArray(
        np.full((1, 5, n_channels), 0.8),
        dims=["chain", "draw", "channel"],
        coords={"channel": list("abcde")[:n_channels]},
    )
    idata = xr.DataTree.from_dict(
        {"/posterior": xr.Dataset({"alpha": draws, "lam": draws + 0.4})}
    )

    with pytest.raises(
        ValueError,
        match=f"'alpha' has length {n_channels} on 'channel', the model has 4",
    ) as error:
        SensitivityAnalysis(model, idata).run_sweep(
            "channel_data", np.linspace(0.5, 1.5, 4), **_SENSITIVITY_KWARGS
        )

    # The error the compiled sweep raised stays attached.
    assert error.value.__cause__ is not None


def test_run_sweep_keeps_an_evaluation_error_a_fitting_posterior_did_not_cause(
    sensitivity,
):
    """An error of the compiled sweep keeps its type when the posterior fits.

    The compiled sweep refuses a two-dimensional sweep grid.  The posterior has
    the model's sizes, so that error is not turned into the one for a posterior
    that does not fit.
    """
    with pytest.raises(TypeError):
        sensitivity.run_sweep(
            "channel_data", np.ones((2, 2)), var_names="channel_contribution"
        )


def test_run_sweep_binds_the_draws_as_shared_variables(
    simple_model_and_idata, monkeypatch
):
    """Two calls agree, and the draws are shared inputs of the compiled sweep.

    Without a ``SharedPosterior`` the draws are baked into the graph as
    constants, and numba cannot cache a compiled graph that holds large
    constants, so every call would pay the JIT again.  Here they must arrive as
    shared variables.
    """
    model, idata = simple_model_and_idata
    sweeps = np.linspace(0.5, 1.5, 4)
    compiled: list = []
    real_function = sensitivity_analysis.function

    def recording_function(inputs, outputs):
        compiled.append(real_function(inputs, outputs))
        return compiled[-1]

    monkeypatch.setattr(sensitivity_analysis, "function", recording_function)

    analysis = SensitivityAnalysis(model, idata)
    first = analysis.run_sweep("channel_data", sweeps, **_SENSITIVITY_KWARGS)
    second = analysis.run_sweep("channel_data", sweeps, **_SENSITIVITY_KWARGS)

    xr.testing.assert_identical(second, first)

    assert len(compiled) == 2
    for function in compiled:
        shared_names = {
            input.name
            for input in function.maker.fgraph.inputs
            if isinstance(input, SharedVariable)
        }
        assert {"alpha", "lam"} <= shared_names


def _initval_model_and_idata() -> tuple[pm.Model, xr.DataTree]:
    """A model whose ``alpha`` carries a non-default ``initval``.

    PyMC cannot clone such a model.  ``run_sweep`` never clones the model it is
    given, and this model guards that.
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
            initval=np.full(4, 0.5),
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


def test_run_sweep_works_with_a_non_default_initval():
    """Regression guard: a model PyMC cannot clone still sweeps.

    ``run_sweep`` binds the draws on the model it is given and never clones it,
    so a non-default ``initval`` does not get in the way and the result equals
    the closed form.
    """
    model, idata = _initval_model_and_idata()
    sweeps = np.linspace(0.5, 1.5, 4)

    result = SensitivityAnalysis(model, idata).run_sweep(
        "channel_data", sweeps, **_SENSITIVITY_KWARGS
    )

    expected = _expected_saturation_sweep(
        model,
        np.asarray(model["channel_data"].get_value()),
        sweeps,
        sweep_type="multiplicative",
    )
    xr.testing.assert_allclose(result, expected)


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
