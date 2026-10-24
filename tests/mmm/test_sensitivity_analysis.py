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
import importlib
import math

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import pytest
import xarray as xr
from pytensor import function as pytensor_function
from pytensor.xtensor.type import as_xtensor
from pytensor.xtensor.vectorization import vectorize_graph

from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_marketing.mmm.mmm import MMM
from pymc_marketing.mmm.sensitivity_analysis import SensitivityAnalysis
from pymc_marketing.pytensor_utils import extract_response_distribution

# Modules whose ``pytensor.function`` calls count as sweep compiles.  Ticket 02
# appends the module it moves compilation into; the counter fixture patches the
# bare name ``function`` in every module listed here, so a listed module must
# import it as ``from pytensor import function``.
_SWEEP_COMPILE_MODULES = ("pymc_marketing.mmm.sensitivity_analysis",)


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
# Parity oracle and compile counter for the sweep-cache migration
# (sandbox/issue-3007-followup, ticket 01).
# ---------------------------------------------------------------------------


def _reference_draw_selection(
    idata: xr.DataTree, posterior_sample_fraction: float
) -> slice | np.ndarray:
    """Frozen copy of ``SensitivityAnalysis._draw_indices_for_percentage``."""
    if not (0 < posterior_sample_fraction <= 1):
        raise ValueError("'posterior_sample_fraction' must be in the interval (0, 1].")

    draws = int(idata.posterior.sizes.get("draw", 0))
    if draws <= 1:
        return slice(None)

    retained_fraction = posterior_sample_fraction**2
    if math.isclose(retained_fraction, 1.0, rel_tol=1e-9):
        return slice(None)

    target_draws = max(1, round(draws * retained_fraction))
    if target_draws >= draws:
        return slice(None)

    indices = np.linspace(0, draws - 1, target_draws, dtype=int)
    indices = np.unique(indices)
    if indices.size == 0:
        return np.array([0], dtype=int)
    indices[-1] = draws - 1
    return indices


def _reference_mask_values(
    response_mask: xr.DataArray | np.ndarray,
    response_dims: tuple[str, ...],
    *,
    model: pm.Model,
    var_names: str,
) -> np.ndarray:
    """Frozen copy of ``SensitivityAnalysis._prepare_response_mask``."""
    if isinstance(response_mask, xr.DataArray):
        mask_xr = response_mask
        if response_dims:
            missing_dims = set(response_dims) - set(mask_xr.dims)
            if missing_dims:
                raise ValueError(
                    "response_mask is missing required dims for "
                    f"{var_names!r}: {sorted(missing_dims)}"
                )
            try:
                mask_xr = mask_xr.transpose(*response_dims)
            except ValueError as err:
                raise ValueError(
                    "response_mask dims must match the order of the "
                    f"{var_names!r} deterministic"
                ) from err
        mask_values = mask_xr.to_numpy()
    else:
        mask_values = np.asarray(response_mask)

    if mask_values.dtype != bool:
        if np.issubdtype(mask_values.dtype, np.integer) or np.issubdtype(
            mask_values.dtype, np.floating
        ):
            mask_values = mask_values.astype(bool)
        else:
            raise TypeError("response_mask must be boolean or castable to boolean")

    if response_dims and mask_values.ndim != len(response_dims):
        raise ValueError(
            "response_mask must have the same number of dims as the "
            f"{var_names!r} deterministic (expected {len(response_dims)}, "
            f"got {mask_values.ndim})"
        )

    expected_shape = [
        len(model.coords[dim]) if dim in model.coords else mask_values.shape[axis]
        for axis, dim in enumerate(response_dims)
    ]
    if expected_shape and tuple(expected_shape) != mask_values.shape:
        raise ValueError(
            "response_mask shape does not match deterministic dims: "
            f"expected {tuple(expected_shape)}, got {mask_values.shape}"
        )

    return mask_values


def _reference_sweep_dataarray(
    result: np.ndarray,
    dims: tuple[str, ...],
    sweep_values: np.ndarray,
    model: pm.Model,
) -> xr.DataArray:
    """Frozen copy of the coordinate assembly at the end of ``run_sweep``."""
    coords = {}
    for dim, shape in zip(dims, result.shape, strict=True):
        if dim == "sweep":
            dim_coords = np.asarray(sweep_values)
        else:
            dim_coords = model.coords.get(dim)
            if dim_coords is None:
                dim_coords = np.arange(shape)
            else:
                dim_coords = np.asarray(dim_coords)
        coords[dim] = dim_coords
    return xr.DataArray(result, dims=dims, coords=coords)


def _reference_run_sweep(
    analysis: SensitivityAnalysis,
    var_input: str,
    sweep_values: np.ndarray,
    *,
    var_names: str = "channel_contribution",
    sweep_type: str = "multiplicative",
    posterior_sample_fraction: float = 1.0,
    response_mask: xr.DataArray | np.ndarray | None = None,
) -> xr.DataArray:
    """Frozen copy of ``SensitivityAnalysis.run_sweep`` as of ticket 01.

    The sweep-cache migration rewrites the production method; this copy is the
    oracle every rewritten version must still match.  The duplication is
    deliberate and must not be collapsed back into calls to
    ``SensitivityAnalysis.run_sweep`` or its helpers.
    """
    sweep = ptx.xtensor("sweep", dims=("sweep",))
    response_dims = analysis.model[var_names].dims

    draw_selection = _reference_draw_selection(
        analysis.idata, posterior_sample_fraction
    )
    resp_graph = extract_response_distribution(
        pymc_model=analysis.model,
        idata=analysis.idata.isel(draw=draw_selection),
        response_variable=var_names,
    )

    mask_array = None
    mask = None
    if response_mask is not None:
        mask_array = _reference_mask_values(
            response_mask,
            response_dims,
            model=analysis.model,
            var_names=var_names,
        )
        mask = pt.tensor("mask", shape=mask_array.shape, dtype=bool)
        mask_bcast, _ = pt.broadcast_arrays(mask, resp_graph.values)
        resp_graph = as_xtensor(
            resp_graph.values[~mask_bcast].set(0), dims=resp_graph.dims
        )

    base_value = analysis.model[var_input]
    if sweep_type == "multiplicative":
        sweep_combinations = sweep * base_value
    elif sweep_type == "additive":
        sweep_combinations = sweep + base_value
    elif sweep_type == "absolute":
        sweep_combinations = sweep.broadcast_like(base_value)
    else:
        raise ValueError(f"Unknown sweep_type {sweep_type!r}")

    sweep_response = vectorize_graph(
        resp_graph, replace={base_value: sweep_combinations}
    )
    sweep_response_summed = sweep_response.sum(dim="date").transpose(
        "sample", "sweep", ...
    )

    inputs = [sweep] if mask is None else [sweep, mask]
    # Imported straight from pytensor so oracle compiles never reach the compile counter.
    compiled = pytensor_function(inputs, sweep_response_summed)
    if mask_array is None:
        result = compiled(sweep_values)
    else:
        result = compiled(sweep_values, mask_array)

    return _reference_sweep_dataarray(
        result, sweep_response_summed.dims, sweep_values, analysis.model
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


def _assert_matches_frozen_reference(
    analysis: SensitivityAnalysis,
    sweep_values: np.ndarray,
    *,
    sweep_type: str,
    posterior_sample_fraction: float,
    response_mask: xr.DataArray | None,
) -> None:
    expected = _reference_run_sweep(
        analysis,
        "channel_data",
        sweep_values,
        var_names="channel_contribution",
        sweep_type=sweep_type,
        posterior_sample_fraction=posterior_sample_fraction,
        response_mask=response_mask,
    )
    actual = analysis.run_sweep(
        "channel_data",
        sweep_values,
        var_names="channel_contribution",
        sweep_type=sweep_type,
        posterior_sample_fraction=posterior_sample_fraction,
        response_mask=response_mask,
    )

    xr.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("posterior_sample_fraction", [1.0, 0.5])
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("sweep_type", ["multiplicative", "additive", "absolute"])
def test_run_sweep_matches_frozen_reference(
    sensitivity, sweep_type, use_mask, posterior_sample_fraction
):
    """The sweep grid on a single-dimension model matches the frozen oracle."""
    _assert_matches_frozen_reference(
        sensitivity,
        np.linspace(0.5, 1.5, 4),
        sweep_type=sweep_type,
        posterior_sample_fraction=posterior_sample_fraction,
        response_mask=_synthetic_response_mask() if use_mask else None,
    )


@pytest.mark.parametrize("posterior_sample_fraction", [1.0, 0.5])
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("sweep_type", ["multiplicative", "additive", "absolute"])
def test_multidim_run_sweep_matches_frozen_reference(
    multidim_mmm, sweep_type, use_mask, posterior_sample_fraction
):
    """The sweep grid keeps its dims, labels and values on a multidim MMM.

    The mask covers one country/channel pair, so the grid exercises the labeled
    ``country`` and ``channel`` axes as well as the masked-overlay path.
    """
    _assert_matches_frozen_reference(
        multidim_mmm.sensitivity,
        np.linspace(0.5, 1.5, 4),
        sweep_type=sweep_type,
        posterior_sample_fraction=posterior_sample_fraction,
        response_mask=_multidim_response_mask() if use_mask else None,
    )


def test_run_sweep_reflects_set_data_between_calls(sensitivity):
    """``pm.set_data`` between calls must change the sweep, as it does today."""
    sweeps = np.linspace(0.5, 1.5, 4)
    kwargs = {
        "var_names": "channel_contribution",
        "sweep_type": "multiplicative",
    }
    before = sensitivity.run_sweep("channel_data", sweeps, **kwargs)

    channel_data = np.asarray(sensitivity.model["channel_data"].get_value())
    pm.set_data({"channel_data": channel_data * 1.5}, model=sensitivity.model)
    after = sensitivity.run_sweep("channel_data", sweeps, **kwargs)

    xr.testing.assert_allclose(
        after, _reference_run_sweep(sensitivity, "channel_data", sweeps, **kwargs)
    )
    assert not np.allclose(after.values, before.values)


def test_run_sweep_reflects_replaced_idata(multidim_mmm):
    """Replacing ``mmm.idata`` between calls must change the sweep."""
    sweeps = np.linspace(0.5, 1.5, 4)
    kwargs = {
        "var_names": "channel_contribution",
        "sweep_type": "multiplicative",
    }
    before = multidim_mmm.sensitivity.run_sweep("channel_data", sweeps, **kwargs)

    replacement = multidim_mmm.idata.copy(deep=True)
    replacement.posterior["saturation_beta"] = (
        replacement.posterior["saturation_beta"] * 0.5
    )
    multidim_mmm.idata = replacement
    after = multidim_mmm.sensitivity.run_sweep("channel_data", sweeps, **kwargs)

    xr.testing.assert_allclose(
        after,
        _reference_run_sweep(
            multidim_mmm.sensitivity, "channel_data", sweeps, **kwargs
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


def test_run_sweep_permuted_posterior_coords_pin_positional_reference(
    simple_model_and_idata,
):
    """Pin today's positional posterior read until ticket 02 realigns by label.

    The permuted posterior carries the same label-to-draw mapping as the
    unpermuted one, only stored in reverse order.  ``ConditionedModel`` will
    realign it by label, so the sweep should then equal ``label_correct``.
    Until then ``run_sweep`` binds the draws positionally, and this test records
    that reference value: ticket 02 replaces the two assertions below with
    ``xr.testing.assert_allclose(actual, label_correct)``.
    """
    model, _ = simple_model_and_idata
    sweeps = np.linspace(0.5, 1.5, 4)
    kwargs = {
        "var_names": "channel_contribution",
        "sweep_type": "multiplicative",
    }
    idata = _channel_varying_posterior(model)
    permuted_idata = _reverse_channel_coords(idata)

    label_correct = SensitivityAnalysis(model, idata).run_sweep(
        "channel_data", sweeps, **kwargs
    )
    permuted_analysis = SensitivityAnalysis(model, permuted_idata)
    actual = permuted_analysis.run_sweep("channel_data", sweeps, **kwargs)

    xr.testing.assert_allclose(
        actual,
        _reference_run_sweep(permuted_analysis, "channel_data", sweeps, **kwargs),
    )
    assert not np.allclose(actual.values, label_correct.values)


@pytest.fixture
def sweep_compile_counter(monkeypatch):
    """Record every ``pytensor.function`` call made through a listed module."""
    compiled_at: list[str] = []

    for module_name in _SWEEP_COMPILE_MODULES:
        module = importlib.import_module(module_name)
        real_function = getattr(module, "function", None)
        if real_function is None:
            # The module no longer compiles; the ``>= 1`` total still catches a silent undercount.
            continue

        def counting_function(
            *args, _module_name=module_name, _real=real_function, **kwargs
        ):
            compiled_at.append(_module_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(module, "function", counting_function)

    return compiled_at


def test_run_sweep_compiles_once_per_call(sensitivity, sweep_compile_counter):
    """Every call compiles through the sensitivity module today.

    Ticket 02 tightens these bounds to ``== 1`` for the first call and ``== 0``
    for the repeat call, once a warm cache serves it.
    """
    sweeps = np.linspace(0.5, 1.5, 4)
    kwargs = {
        "var_names": "channel_contribution",
        "sweep_type": "multiplicative",
    }

    sensitivity.run_sweep("channel_data", sweeps, **kwargs)
    first_call_compiles = len(sweep_compile_counter)
    sensitivity.run_sweep("channel_data", sweeps, **kwargs)
    repeat_compiles = len(sweep_compile_counter) - first_call_compiles

    assert first_call_compiles >= 1, f"first call compiled {first_call_compiles} times"
    assert repeat_compiles >= 1, f"repeat call compiled {repeat_compiles} times"
