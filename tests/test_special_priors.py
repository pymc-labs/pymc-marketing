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
import json
import re
import textwrap

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr
import yaml
from pymc.exceptions import SamplingError
from pymc.logprob.utils import ParameterValueError
from pymc_extras.deserialize import deserialize
from pymc_extras.prior import (
    MuAlreadyExistsError,
    Prior,
    UnknownTransformError,
    VariableFactory,
    register_tensor_transform,
    sample_prior,
)
from pytensor import function
from pytensor.xtensor.type import as_xtensor

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.builders.yaml import build_mmm_from_yaml
from pymc_marketing.mmm.mmm import MMM
from pymc_marketing.serialization import serialization
from pymc_marketing.special_priors import (
    IndependentSigmas,
    Interaction,
    LaplacePrior,
    LogNormalPrior,
    MaskedPrior,
    NestedZeroSumPrior,
    Term,
    VarianceBudget,
    _is_LaplacePrior_type,
    _is_LogNormalPrior_type,
)


@pytest.mark.parametrize(
    "mean, std, centered, dims",
    [
        (
            Prior("Gamma", mu=1.0, sigma=1.0),
            Prior("Gamma", mu=1.0, sigma=1.0),
            True,
            ("channel",),
        ),
        (1.0, 2.0, False, ("channel",)),
        (1.0, 2.0, True, ("channel",)),
        (np.array([1, 2, 3]), np.array([4, 5, 6]), True, ("channel",)),
        (np.array([1, 2, 3]), np.array([4, 5, 6]), False, ("channel",)),
        (1.0, 2.0, True, ()),
    ],
)
def test_LogNormalPrior_args(mean, std, centered, dims):
    """
    Checks:
    - sample_prior runs
    - create_variable runs
    - round trip: dict to class to dict to class, doesn't lose any information
    """
    rv = LogNormalPrior(mean=mean, std=std, centered=centered, dims=dims)

    coords = {"channel": ["C1", "C2", "C3"]}

    if dims:
        prior = rv.sample_prior(coords=coords)
        assert prior.channel.shape == (len(coords["channel"]),)
    else:
        prior = rv.sample_prior()
        assert isinstance(prior, xr.Dataset)

    if centered is False:
        assert "variable_log_offset" in prior.data_vars

    with pm.Model(coords=coords):
        rv.create_variable("test", xdist=True)

    assert rv.to_dict() == rv.from_dict(rv.to_dict()).to_dict()


def test_LogNormalPrior_args_invalid():
    with pytest.raises(ValueError):
        LogNormalPrior(alpha=1.0, beta=1.0)


def test_LogNormalPrior_from_dict_invalid():
    with pytest.raises(
        ValueError,
        match=r"Must be a dictionary representation of a prior distribution.",
    ):
        LogNormalPrior.from_dict(LogNormalPrior(mean=0.0, std=1.0))


def test_the_deserializer_can_distinguish_between_types_of_prior_classes():
    assert _is_LogNormalPrior_type(LogNormalPrior(mu=1.0, sigma=1.0).to_dict())
    assert not _is_LogNormalPrior_type(Prior("Normal", mu=1.0, sigma=1.0).to_dict())
    assert not _is_LogNormalPrior_type(LaplacePrior(mu=1.0, b=1.0).to_dict())


@pytest.mark.parametrize(
    "mu, b, centered, dims",
    [
        (
            Prior("Normal", mu=0.0, sigma=1.0),
            Prior("Gamma", alpha=1.0, beta=1.0),
            True,
            ("geo",),
        ),
        (1.0, 2.0, False, ("geo",)),
        (1.0, 2.0, True, ("geo",)),
        (np.array([1, 2, 3]), np.array([4, 5, 6]), True, ("geo",)),
        (np.array([1, 2, 3]), np.array([4, 5, 6]), False, ("geo",)),
        (1.0, 2.0, True, ()),
    ],
)
def test_LaplacePrior_args(mu, b, centered, dims):
    """
    Checks:
    - sample_prior runs
    - create_variable runs
    - round trip: dict to class to dict to class, doesn't lose any information
    """
    rv = LaplacePrior(mu=mu, b=b, centered=centered, dims=dims)

    coords = {"geo": ["G1", "G2", "G3"]}

    if dims:
        prior = rv.sample_prior(coords=coords)
        assert prior.geo.shape == (len(coords["geo"]),)
    else:
        prior = rv.sample_prior()
        assert isinstance(prior, xr.Dataset)

    if centered is False:
        assert "variable_sigma" in prior.data_vars

    with pm.Model(coords=coords):
        rv.create_variable("test", xdist=True)

    assert rv.to_dict() == rv.from_dict(rv.to_dict()).to_dict()


def test_LaplacePrior_args_invalid():
    with pytest.raises(ValueError):
        LaplacePrior(alpha=1.0, beta=1.0)


def test_the_deserializer_can_distinguish_LaplacePrior_from_other_types():
    assert _is_LaplacePrior_type(LaplacePrior(mu=1.0, b=1.0).to_dict())
    assert not _is_LaplacePrior_type(LogNormalPrior(mu=1.0, sigma=1.0).to_dict())
    assert not _is_LaplacePrior_type(Prior("Normal", mu=1.0, sigma=1.0).to_dict())


def test_masked_prior_simple_1d():
    """MaskedPrior creates zeros on inactive entries and preserves dims."""
    coords = {"country": ["Venezuela", "Colombia"]}
    mask = xr.DataArray([True, False], dims=["country"], coords=coords)
    base = Prior("Normal", mu=0, sigma=1, dims=("country",))

    with pm.Model(coords=coords):
        mp = MaskedPrior(base, mask)
        mp.create_variable("intercept")
        idata = pm.sample_prior_predictive()

    samples = idata.prior["intercept"].values  # (chain, draw, country)
    # All draws at the inactive entry must be exactly zero
    assert np.all(samples[..., 1] == 0)


@pytest.mark.parametrize(
    "mu_value, param_id",
    [
        ([0, 1], "list"),
        (np.array([0, 1]), "numpy_array"),
        (xr.DataArray([0, 1], dims="country"), "dataarray_1d"),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_masked_prior_non_scalar_params_regression_2463(mu_value, param_id):
    """Regression test for #2463: MaskedPrior must work with non-scalar Prior params."""
    coords = {"country": ["Venezuela", "Colombia"]}
    mask = xr.DataArray([True, False], dims=["country"], coords=coords)
    prior = Prior("Normal", mu=mu_value, sigma=10, dims=("country",))

    with pm.Model(coords=coords):
        mp = MaskedPrior(prior, mask)
        result = mp.create_variable("test")

    vals = result.eval()
    assert vals.shape == (2,)
    assert vals[1] == 0.0  # Colombia is masked out
    assert np.isfinite(vals[0])  # Venezuela is active


def test_subset_raw_parameter_dataarray_dim_order():
    """Verify xr.DataArray params are subset in mask dim order, not broadcast dim order."""
    coords = {"country": ["Venezuela", "Colombia"], "channel": ["TV", "Radio"]}
    mask = xr.DataArray(
        [[True, False], [True, True]],
        dims=["country", "channel"],
        coords=coords,
    )
    base = Prior("Normal", mu=0, sigma=1, dims=("country", "channel"))
    mp = MaskedPrior(base, mask)
    flat_mask = mask.values.astype(bool).ravel()

    value = xr.DataArray([0, 1], dims="channel")
    result = mp._subset_raw_parameter(value, flat_mask)
    # Active positions in (country, channel) order: (Ven,TV), (Col,TV), (Col,Radio)
    # TV=0, Radio=1 → expected [0, 0, 1]
    np.testing.assert_array_equal(result, [0, 0, 1])

    value_country = xr.DataArray([5, 10], dims="country")
    result_country = mp._subset_raw_parameter(value_country, flat_mask)
    # Venezuela=5, Colombia=10 → expected [5, 10, 10]
    np.testing.assert_array_equal(result_country, [5, 10, 10])


def test_masked_prior_dataarray_per_channel_2d_mask():
    """DataArray param along rightmost dim with a 2D mask (migration guide pattern)."""
    coords = {"country": ["Venezuela", "Colombia"], "channel": ["TV", "Radio"]}
    mask = xr.DataArray(
        [[True, False], [True, True]],
        dims=["country", "channel"],
        coords=coords,
    )
    prior = Prior(
        "Normal",
        mu=xr.DataArray([0, 100], dims="channel"),
        sigma=0.001,
        dims=("country", "channel"),
    )

    with pm.Model(coords=coords):
        mp = MaskedPrior(prior, mask)
        result = mp.create_variable("test")

    vals = result.eval()
    assert vals.shape == (2, 2)
    assert vals[0, 1] == 0.0  # (Venezuela, Radio) masked out
    assert abs(vals[0, 0] - 0) < 1  # (Ven, TV) ≈ mu_TV = 0
    assert abs(vals[1, 0] - 0) < 1  # (Col, TV) ≈ mu_TV = 0
    assert abs(vals[1, 1] - 100) < 1  # (Col, Radio) ≈ mu_Radio = 100


def test_masked_prior_dataarray_non_rightmost_dim_2d_mask():
    """DataArray param along non-rightmost dim -- only xarray broadcasting gets this right."""
    coords = {
        "country": ["Venezuela", "Colombia"],
        "channel": ["TV", "Radio", "Digital"],
    }
    mask = xr.DataArray(
        [[True, False, True], [True, True, False]],
        dims=["country", "channel"],
        coords=coords,
    )
    prior = Prior(
        "Normal",
        mu=xr.DataArray([5, 10], dims="country"),
        sigma=1,
        dims=("country", "channel"),
    )

    with pm.Model(coords=coords):
        mp = MaskedPrior(prior, mask)
        result = mp.create_variable("test")

    vals = result.eval()
    assert vals.shape == (2, 3)
    assert vals[0, 1] == 0.0  # (Venezuela, Radio) masked
    assert vals[1, 2] == 0.0  # (Colombia, Digital) masked
    active_mask = mask.values
    assert np.all(np.isfinite(vals[active_mask]))


def test_masked_prior_2d_mask_full_shape_list():
    """Plain list matching full mask shape with 2D dims."""
    coords = {"country": ["Venezuela", "Colombia"], "channel": ["TV", "Radio"]}
    mask = xr.DataArray(
        [[True, False], [True, True]],
        dims=["country", "channel"],
        coords=coords,
    )
    prior = Prior(
        "Normal",
        mu=[[0, 1], [2, 3]],
        sigma=1,
        dims=("country", "channel"),
    )

    with pm.Model(coords=coords):
        mp = MaskedPrior(prior, mask)
        result = mp.create_variable("test")

    vals = result.eval()
    assert vals.shape == (2, 2)
    assert vals[0, 1] == 0.0


def test_masked_prior_nested_prior_with_array_params():
    """Nested Prior whose inner parameters are lists must be subset via recursion."""
    coords = {"country": ["Venezuela", "Colombia"]}
    mask = xr.DataArray([True, False], dims=["country"], coords=coords)
    prior = Prior(
        "Normal",
        mu=Prior("Normal", mu=[0, 1], sigma=1, dims=("country",)),
        sigma=1,
        dims=("country",),
    )

    with pm.Model(coords=coords):
        mp = MaskedPrior(prior, mask)
        result = mp.create_variable("test")

    vals = result.eval()
    assert vals.shape == (2,)
    assert vals[1] == 0.0


def test_masked_prior_all_active_preserves_array_param():
    """All-True mask with array params: no subsetting, full shape preserved."""
    coords = {"country": ["Venezuela", "Colombia"]}
    mask = xr.DataArray([True, True], dims=["country"], coords=coords)
    prior = Prior("Normal", mu=[0, 1], sigma=10, dims=("country",))

    with pm.Model(coords=coords):
        mp = MaskedPrior(prior, mask)
        result = mp.create_variable("test")

    vals = result.eval()
    assert vals.shape == (2,)
    assert np.all(np.isfinite(vals))


def test_masked_prior_with_logistic_saturation_prior_sampling():
    """Mask a saturation parameter and check zeros at masked positions in prior sampling."""
    coords = {
        "country": ["Colombia", "Venezuela"],
        "channel": ["x1", "x2", "x3", "x4"],
    }
    mask_excluded_x4_colombia = xr.DataArray(
        [[True, False, True, False], [True, True, True, True]],
        dims=["country", "channel"],
        coords=coords,
    )

    saturation = LogisticSaturation(
        priors={
            "lam": MaskedPrior(
                Prior("Gamma", mu=2, sigma=0.5, dims=("country", "channel")),
                mask=mask_excluded_x4_colombia,
            ),
            "beta": Prior("Gamma", mu=3, sigma=0.5, dims=("country", "channel")),
        }
    )

    prior = saturation.sample_prior(coords=coords, random_seed=0)
    lam = prior["saturation_lam"].transpose("chain", "draw", "country", "channel")

    # The masked position (Colombia, x4) should be exactly zero across draws
    colombia_idx = 0
    x4_idx = 3
    assert np.all(lam.values[..., colombia_idx, x4_idx] == 0)


def test_mmm_fit_with_masked_saturation_param_small(mock_pymc_sample):
    """Tiny MMM fit where one saturation parameter is masked to zero across dims."""
    rng = np.random.default_rng(0)
    # Small panel with 2 countries, 2 channels, 10 weekly obs per country
    n = 10
    dates = pd.date_range("2024-01-01", periods=n, freq="W-MON")
    countries = ["Colombia", "Venezuela"]
    df_list = []
    for c in countries:
        df_list.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "country": c,
                    "C1": rng.integers(10, 30, n),
                    "C2": rng.integers(5, 25, n),
                    "control": rng.normal(0, 1, n),
                }
            )
        )
    X = pd.concat(df_list, ignore_index=True)
    # Create a synthetic target
    y = (
        0.4 * X["C1"].values
        + 0.2 * X["C2"].values
        + 1.5 * X["control"].values
        + (X["country"].values == "Colombia").astype(float) * 5
        + rng.normal(0, 1.0, len(X))
    )
    y = pd.Series(y, name="y")

    coords = {
        "country": countries,
        "channel": ["C1", "C2"],
    }
    # Mask lam for (Colombia, C2)
    mask = xr.DataArray(
        [[True, False], [True, True]], dims=["country", "channel"], coords=coords
    )

    mmm = MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        control_columns=["control"],
        target_column="y",
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(
            priors={
                "lam": MaskedPrior(
                    Prior("Gamma", mu=2, sigma=0.5, dims=("country", "channel")),
                    mask=mask,
                ),
                # keep beta unmasked to allow learning
                "beta": Prior("Gamma", mu=3, sigma=0.5, dims=("country", "channel")),
            }
        ),
    )

    # Super small fit to keep runtime minimal
    mmm.fit(X, y, draws=20, tune=20, chains=1, random_seed=1, target_accept=0.8)

    lam = (
        mmm.idata.posterior["saturation_lam"]
        .transpose("chain", "draw", "country", "channel")
        .values
    )
    colombia_idx = 0
    c2_idx = 1
    # All posterior draws at masked position are exactly zero
    assert np.all(lam[..., colombia_idx, c2_idx] == 0)


@pytest.mark.parametrize("mode", ["constructor", "from_dict"])
def test_masked_prior_raises_on_mismatched_mask_dims_order_param(mode):
    """MaskedPrior should raise if mask dims order differs from prior.dims (both paths)."""
    coords = {
        "country": ["Colombia", "Venezuela"],
        "channel": ["x1", "x2", "x3", "x4"],
    }
    prior = Prior("Normal", mu=0, sigma=1, dims=("country", "channel"))

    with pm.Model(coords=coords):
        if mode == "constructor":
            # Correct shape but dims in reverse order
            mask_rev = xr.DataArray(
                np.ones((len(coords["channel"]), len(coords["country"])), dtype=bool),
                dims=["channel", "country"],
                coords=coords,
            )
            with pytest.raises(
                ValueError, match=r"mask dims must match prior\.dims order"
            ):
                MaskedPrior(prior, mask_rev)
        elif mode == "from_dict":
            payload = {
                "class": "MaskedPrior",
                "data": {
                    "prior": prior.to_dict(),
                    "mask": np.ones(
                        (len(coords["channel"]), len(coords["country"])), dtype=bool
                    ).tolist(),
                    "mask_dims": ["channel", "country"],
                    "active_dim": None,
                },
            }
            with pytest.raises(
                ValueError, match=r"mask dims must match prior\.dims order"
            ):
                MaskedPrior.from_dict(payload)
        else:
            pytest.fail(f"Unknown mode: {mode}")


@pytest.mark.parametrize(
    "prior_as, include_mask_dims, active_dim, with_data_wrapper",
    [
        ("dict", True, "my_active", True),
        ("object", False, None, True),
        ("dict", True, None, False),  # no outer "data" key
    ],
)
def test_masked_prior_from_dict_success_covers_dims_and_active(
    prior_as, include_mask_dims, active_dim, with_data_wrapper
):
    """Cover from_dict success: explicit mask_dims and fallback to prior.dims, with/without active_dim."""
    coords = {
        "country": ["Colombia", "Venezuela"],
        "channel": ["x1", "x2", "x3", "x4"],
    }
    prior = Prior("Normal", mu=0, sigma=1, dims=("country", "channel"))

    mask_vals = np.ones((len(coords["country"]), len(coords["channel"])), dtype=bool)

    prior_payload = prior.to_dict() if prior_as == "dict" else prior

    data_payload = {
        "prior": prior_payload,
        "mask": mask_vals.tolist(),
        "active_dim": active_dim,
    }
    if include_mask_dims:
        data_payload["mask_dims"] = ["country", "channel"]

    payload = {"class": "MaskedPrior", "data": data_payload}
    if not with_data_wrapper:
        payload = data_payload  # exercise the branch without outer "data"

    mp = MaskedPrior.from_dict(payload)

    # Check dims and active_dim handling
    assert tuple(mp.dims) == ("country", "channel")
    expected_active = active_dim or "non_null_dims:country_channel"
    assert mp.active_dim == expected_active


def test_masked_prior_round_trip_to_from_dict():
    """Round-trip serialization for MaskedPrior preserves payload contents."""
    coords = {
        "country": ["Venezuela", "Colombia"],
        "channel": ["x1", "x2"],
    }
    mask = xr.DataArray(
        [[True, False], [True, True]],
        dims=["country", "channel"],
        coords=coords,
    )
    prior = Prior("Normal", mu=0, sigma=1, dims=("country", "channel"))

    mp = MaskedPrior(prior, mask, active_dim="my_active")
    payload = mp.to_dict()

    mp2 = MaskedPrior.from_dict(payload)
    assert mp2.to_dict() == payload


def _compile(expr):
    # Helper to compile a pytensor expression to a numpy callable
    return function([], expr)


def test_masked_prior_create_likelihood_all_masked_returns_zeros():
    """When all entries are masked out, return deterministic zeros with original dims."""
    coords = {
        "date": pd.date_range("2024-01-01", periods=3, freq="D"),
        "country": ["CO", "VE"],
    }
    mask = xr.DataArray(
        np.zeros((3, 2), dtype=bool), dims=["date", "country"], coords=coords
    )
    like = Prior("Normal", sigma=1, dims=("date", "country"))

    with pm.Model(coords=coords) as model:
        mp = MaskedPrior(like, mask)
        y = mp.create_likelihood_variable(
            "y", mu=0.0, observed=np.zeros((3, 2), dtype=float)
        )

        # Deterministic exists; verify shape and values instead of internal dims attribute
        assert y.name == "y"
        assert "y" in model.named_vars
        # No active subset variable should be created when all masked
        assert "y_active" not in model.named_vars

        # Evaluate and ensure exact zeros
        eval_y = _compile(model["y"])()
        assert eval_y.shape == (3, 2)
        assert np.all(eval_y == 0.0)


@pytest.mark.parametrize(
    "mu_kind, observed_kind",
    [
        (
            "vector_country",
            "xarray_full",
        ),  # broadcast mu over date, observed has .values
        ("scalar", "xarray_full"),  # scalar mu path
    ],
)
def test_masked_prior_create_likelihood_active_branch_suffix_and_broadcast(
    mu_kind, observed_kind
):
    """Active subset path: coord suffixing and broadcasting of mu/observed are exercised."""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    coords = {
        "date": dates,
        "country": ["CO", "VE"],
    }
    # True at positions: (0,CO), (2,VE), (3,CO) -> 3 active
    mask_vals = np.array(
        [
            [True, False],
            [False, False],
            [False, True],
            [True, False],
        ]
    )
    mask = xr.DataArray(mask_vals, dims=["date", "country"], coords=coords)

    like = Prior(
        "Normal",
        sigma=Prior("HalfNormal", sigma=1, dims=("date", "country")),  # nested param
        dims=("date", "country"),
    )

    # mu will be created as a PyMC variable inside the model context below

    # Observed as xarray with full dims to hit `.values` branch
    if observed_kind == "xarray_full":
        observed = xr.DataArray(
            np.arange(mask_vals.size, dtype=float).reshape(mask_vals.shape),
            dims=["date", "country"],
            coords=coords,
        )
    else:
        raise AssertionError("Unknown observed_kind")

    with pm.Model(coords=coords) as model:
        # Create a conflicting coord name to trigger suffixing
        preexisting = "my_active"
        model.add_coords({preexisting: np.arange(99)})

        mp = MaskedPrior(like, mask, active_dim=preexisting)
        # Create mu as a PyMC variable (distribution) per parametrization
        if mu_kind == "vector_country":
            mu_var = Prior("Normal", mu=0, sigma=1, dims=("country",)).create_variable(
                "mu"
            )
        elif mu_kind == "scalar":
            mu_var = Prior("Normal", mu=0, sigma=1).create_variable("mu")
        else:
            raise AssertionError("Unknown mu_kind")

        mp.create_likelihood_variable("y", mu=mu_var, observed=observed)

        # Suffix must have been applied because lengths mismatch (99 vs 3)
        assert mp.active_dim.startswith(preexisting + "__")
        assert len(model.coords[mp.active_dim]) == int(mask_vals.sum()) == 3

        # Observed RV exists over the active subset
        assert "y_active" in model.named_vars

        # The Deterministic wrapper preserves full shape and values: zeros at inactive
        eval_y = _compile(model["y"])()
        assert eval_y.shape == mask_vals.shape
        # Inactive positions are zeros
        assert np.allclose(eval_y[~mask_vals], 0.0)
        # Active positions are finite and not all zeros (given our observed has non-zeros)
        assert np.all(np.isfinite(eval_y[mask_vals]))
        assert np.any(eval_y[mask_vals] != 0.0)


def test_special_prior_serialization_round_trip():
    """Test that SpecialPrior subclasses can be serialized and deserialized."""
    from pymc_extras.deserialize import deserialize

    # Test LogNormalPrior
    lognormal = LogNormalPrior(mu=1.0, sigma=2.0, dims=("channel",))
    lognormal_dict = lognormal.to_dict()
    lognormal_deserialized = deserialize(lognormal_dict)
    assert lognormal == lognormal_deserialized
    assert lognormal_deserialized.to_dict() == lognormal_dict

    # Test LaplacePrior
    laplace = LaplacePrior(mu=0.5, b=1.5, dims=("geo",))
    laplace_dict = laplace.to_dict()
    laplace_deserialized = deserialize(laplace_dict)
    assert laplace == laplace_deserialized
    assert laplace_deserialized.to_dict() == laplace_dict


def test_special_prior_equality():
    """Test that SpecialPrior __eq__ method works correctly."""
    # Same parameters
    prior1 = LogNormalPrior(mu=1.0, sigma=2.0, dims=("channel",))
    prior2 = LogNormalPrior(mu=1.0, sigma=2.0, dims=("channel",))
    assert prior1 == prior2

    # Different parameters
    prior3 = LogNormalPrior(mu=1.0, sigma=3.0, dims=("channel",))
    assert prior1 != prior3

    # Different dims
    prior4 = LogNormalPrior(mu=1.0, sigma=2.0, dims=("geo",))
    assert prior1 != prior4

    # Different centered
    prior5 = LogNormalPrior(mu=1.0, sigma=2.0, centered=False, dims=("channel",))
    assert prior1 != prior5

    # Different class
    laplace = LaplacePrior(mu=1.0, b=2.0, dims=("channel",))
    assert prior1 != laplace

    # Numpy arrays
    prior_array1 = LogNormalPrior(
        mu=np.array([1.0, 2.0]), sigma=np.array([0.5, 0.5]), dims=("channel",)
    )
    prior_array2 = LogNormalPrior(
        mu=np.array([1.0, 2.0]), sigma=np.array([0.5, 0.5]), dims=("channel",)
    )
    assert prior_array1 == prior_array2

    # Different numpy arrays
    prior_array3 = LogNormalPrior(
        mu=np.array([1.0, 3.0]), sigma=np.array([0.5, 0.5]), dims=("channel",)
    )
    assert prior_array1 != prior_array3


def test_special_prior_hashable():
    """Test that SpecialPrior instances are hashable and can be used in sets/dicts."""
    # Test basic hashability
    prior1 = LogNormalPrior(mu=1.0, sigma=2.0, dims=("channel",))
    prior2 = LogNormalPrior(mu=1.0, sigma=2.0, dims=("channel",))
    prior3 = LogNormalPrior(mu=1.0, sigma=3.0, dims=("channel",))

    # Equal objects should have the same hash
    assert hash(prior1) == hash(prior2)

    # Different objects typically have different hashes (not guaranteed, but expected)
    # We just verify they're hashable
    hash(prior3)

    # Test usage in set
    prior_set = {prior1, prior2, prior3}
    assert len(prior_set) == 2  # prior1 and prior2 are equal, so only 2 unique

    # Test usage as dict key
    prior_dict = {prior1: "value1", prior3: "value3"}
    assert len(prior_dict) == 2
    assert prior_dict[prior2] == "value1"  # prior2 equals prior1

    # Test with numpy arrays
    prior_array1 = LogNormalPrior(
        mu=np.array([1.0, 2.0]), sigma=np.array([0.5, 0.5]), dims=("channel",)
    )
    prior_array2 = LogNormalPrior(
        mu=np.array([1.0, 2.0]), sigma=np.array([0.5, 0.5]), dims=("channel",)
    )

    # Equal arrays should have same hash
    assert hash(prior_array1) == hash(prior_array2)

    # Can be used in sets
    array_set = {prior_array1, prior_array2}
    assert len(array_set) == 1  # They're equal

    # Test LaplacePrior
    laplace1 = LaplacePrior(mu=0.5, b=1.5, dims=("geo",))
    laplace2 = LaplacePrior(mu=0.5, b=1.5, dims=("geo",))
    assert hash(laplace1) == hash(laplace2)

    # Can mix different SpecialPrior types in a set
    mixed_set = {prior1, laplace1}
    assert len(mixed_set) == 2


class TestSpecialPriorTypeRegistry:
    """Tests for TypeRegistry-based round-trip serialization of SpecialPrior classes."""

    @pytest.mark.parametrize(
        "cls_name",
        ["LogNormalPrior", "LaplacePrior", "MaskedPrior"],
    )
    def test_registered_in_type_registry(self, cls_name):
        from pymc_marketing import special_priors
        from pymc_marketing.serialization import serialization

        cls = getattr(special_priors, cls_name)
        type_key = f"{cls.__module__}.{cls.__qualname__}"
        assert type_key in serialization._registry, f"{cls_name} not registered"

    def test_log_normal_roundtrip_all_parameters(self):
        from pymc_marketing.serialization import serialization

        original = LogNormalPrior(mu=2.0, sigma=0.8, dims=("channel",), centered=False)
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is LogNormalPrior
        assert restored.parameters["mean"] == 2.0
        assert restored.parameters["std"] == 0.8
        assert restored.dims == ("channel",)
        assert restored.centered is False

    def test_laplace_prior_roundtrip_all_parameters(self):
        from pymc_marketing.serialization import serialization

        original = LaplacePrior(mu=1.5, b=0.3, dims=("geo",), centered=False)
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is LaplacePrior
        assert restored.parameters["mu"] == 1.5
        assert restored.parameters["b"] == 0.3
        assert restored.dims == ("geo",)
        assert restored.centered is False

    def test_masked_prior_roundtrip_all_parameters(self):
        from pymc_marketing.serialization import serialization

        mask = xr.DataArray(
            np.array([[True, False], [True, True]]),
            dims=("geo", "channel"),
            coords={"geo": ["a", "b"], "channel": ["tv", "radio"]},
        )
        original = MaskedPrior(
            prior=Prior("Normal", mu=0, sigma=1, dims=("geo", "channel")),
            mask=mask,
        )
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is MaskedPrior
        assert restored.dims == original.dims
        assert tuple(restored.mask.dims) == tuple(original.mask.dims)
        np.testing.assert_array_equal(restored.mask.values, original.mask.values)


def test_mmm_with_special_prior_save_load_round_trip(tmp_path, mock_pymc_sample):
    """Test that MMM with SpecialPrior in saturation can be saved and loaded with check=True."""
    # Create minimal data
    rng = np.random.default_rng(42)
    n = 20
    dates = pd.date_range("2024-01-01", periods=n, freq="W-MON")

    X = pd.DataFrame(
        {
            "date": dates,
            "channel_1": rng.integers(10, 100, n),
            "channel_2": rng.integers(10, 100, n),
        }
    )
    y = pd.Series(rng.normal(100, 10, n), name="y")

    # Create MMM with LogisticSaturation using SpecialPrior
    mmm = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="y",
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(
            priors={
                "lam": LogNormalPrior(mu=1.0, sigma=1.0, dims=("channel",)),
                "beta": LogNormalPrior(mu=2.0, sigma=0.5, dims=("channel",)),
            }
        ),
    )

    # Fit the model
    mmm.fit(X, y, chains=1, draws=10, tune=10, random_seed=42)

    # Get the original ID
    original_id = mmm.id

    # Save the model
    save_path = tmp_path / "mmm_with_special_prior.nc"
    mmm.save(str(save_path))

    # Load the model with check=True (should work!)
    mmm_loaded = MMM.load(str(save_path), check=True)

    # Verify IDs match
    assert mmm_loaded.id == original_id

    # Verify models are equal
    assert mmm == mmm_loaded

    # Verify saturation priors are SpecialPrior instances
    assert isinstance(mmm_loaded.saturation.priors["lam"], LogNormalPrior)
    assert isinstance(mmm_loaded.saturation.priors["beta"], LogNormalPrior)

    # Verify prior parameters match
    assert mmm_loaded.saturation.priors["lam"] == mmm.saturation.priors["lam"]
    assert mmm_loaded.saturation.priors["beta"] == mmm.saturation.priors["beta"]


def test_xarray_dataarray_deserializer_registered():
    da = xr.DataArray([1.0, 2.0], dims=["channel"], coords={"channel": ["a", "b"]})
    result = deserialize(da.to_dict())
    assert isinstance(result, xr.DataArray)
    xr.testing.assert_equal(result, da)


def test_lognormal_prior_roundtrip_with_dataarray_params():
    coords = {"country": ["ES", "IT"], "channel": ["search", "social"]}
    mean = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]], dims=["country", "channel"], coords=coords
    )
    std = xr.DataArray(
        [[0.5, 0.5], [0.5, 0.5]], dims=["country", "channel"], coords=coords
    )
    prior = LogNormalPrior(mean=mean, std=std, dims=("country", "channel"))
    restored = LogNormalPrior.from_dict(prior.to_dict())
    assert isinstance(restored.parameters["mean"], xr.DataArray)
    assert isinstance(restored.parameters["std"], xr.DataArray)
    xr.testing.assert_equal(restored.parameters["mean"], mean)
    xr.testing.assert_equal(restored.parameters["std"], std)
    assert restored.dims == prior.dims


def test_lognormal_prior_roundtrip_non_centered():
    coords = {"country": ["ES", "IT"], "channel": ["search", "social"]}
    mean = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]], dims=["country", "channel"], coords=coords
    )
    std = xr.DataArray(
        [[0.5, 0.5], [0.5, 0.5]], dims=["country", "channel"], coords=coords
    )
    prior = LogNormalPrior(
        mean=mean, std=std, dims=("country", "channel"), centered=False
    )
    restored = LogNormalPrior.from_dict(prior.to_dict())
    assert not restored.centered
    xr.testing.assert_equal(restored.parameters["mean"], mean)
    xr.testing.assert_equal(restored.parameters["std"], std)


def test_lognormal_prior_roundtrip_via_deserialize():
    coords = {"country": ["ES", "IT"], "channel": ["search", "social"]}
    mean = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]], dims=["country", "channel"], coords=coords
    )
    std = xr.DataArray(
        [[0.5, 0.5], [0.5, 0.5]], dims=["country", "channel"], coords=coords
    )
    prior = LogNormalPrior(mean=mean, std=std, dims=("country", "channel"))
    restored = deserialize(prior.to_dict())
    assert isinstance(restored, LogNormalPrior)
    xr.testing.assert_equal(restored.parameters["mean"], mean)
    xr.testing.assert_equal(restored.parameters["std"], std)


def test_lognormal_prior_std_only_construction():
    prior = LogNormalPrior(std=1.0)
    assert prior.parameters == {"std": 1.0}
    # The sigma alias normalizes to std, like mu normalizes to mean.
    assert LogNormalPrior(sigma=1.0).parameters == {"std": 1.0}


@pytest.mark.parametrize(
    "parameters",
    [
        {"alpha": 1.0, "beta": 1.0},
        {"mean": 1.0},
        {"mean": 1.0, "std": 1.0, "extra": 1.0},
    ],
)
def test_lognormal_prior_still_rejects_bad_parameter_sets(parameters):
    with pytest.raises(ValueError, match="Parameters must be mean and std"):
        LogNormalPrior(**parameters)


def test_lognormal_prior_std_only_create_variable_raises():
    prior = LogNormalPrior(std=1.0)
    with pm.Model():
        with pytest.raises(ValueError, match="Provide 'mean'"):
            prior.create_variable("x", xdist=True)


def _likelihood_model(mu_offset: float = 1.0):
    """Model with a free ``beta`` shifting the response-scale mu of ``y``."""
    coords = {"date": np.arange(5)}
    observed = np.array([1.0, 2.0, 1.5, 0.8, 1.2])
    likelihood = LogNormalPrior(std=Prior("HalfNormal", sigma=0.5), dims=("date",))
    with pm.Model(coords=coords) as model:
        beta = pmd.Normal("beta")
        mu = beta + as_xtensor(np.full(5, mu_offset), dims=("date",))
        likelihood.create_likelihood_variable(
            "y", mu=mu, observed=as_xtensor(observed, dims=("date",)), xdist=True
        )
    return model, observed


def test_lognormal_prior_create_likelihood_variable():
    model, _ = _likelihood_model()
    assert "y" in model.named_vars
    assert "y_std" in model.named_vars
    assert model["y"] in model.observed_RVs
    assert model["y_std"] in model.free_RVs
    assert all(np.isfinite(v) for v in model.point_logps().values())


def test_lognormal_prior_create_likelihood_matches_conversion():
    m, s = 1.5, 0.5
    coords = {"date": np.arange(4)}
    observed = np.array([1.0, 2.0, 1.5, 0.8])
    likelihood = LogNormalPrior(std=s, dims=("date",))
    with pm.Model(coords=coords) as model:
        mu = as_xtensor(np.full(4, m), dims=("date",))
        likelihood.create_likelihood_variable(
            "y", mu=mu, observed=as_xtensor(observed, dims=("date",)), xdist=True
        )

    # The docstring conversion, written with the squared form.
    mu_log = np.log(m**2 / np.sqrt(m**2 + s**2))
    sigma_log = np.sqrt(np.log1p(s**2 / m**2))
    with pm.Model(coords=coords) as reference:
        pmd.LogNormal(
            "y",
            mu=as_xtensor(np.full(4, mu_log), dims=("date",)),
            sigma=sigma_log,
            observed=as_xtensor(observed, dims=("date",)),
            dims=("date",),
        )

    np.testing.assert_allclose(model.compile_logp()({}), reference.compile_logp()({}))


@pytest.mark.parametrize("beta", [-2.0, 0.0, -1e-300], ids=["neg", "zero", "tiny"])
def test_lognormal_prior_create_likelihood_negative_mu(beta):
    model, _ = _likelihood_model(mu_offset=0.0)
    bad_point = model.initial_point()
    bad_point["beta"] = np.array(beta)

    # pymc-compiled functions rewrite the check to -inf. The conversion is
    # evaluated on a positive stand-in where mu <= 0, so the logp is -inf
    # rather than NaN even where (std / mu) ** 2 would overflow. The
    # rejection is safe but not clean: the -inf log-mean still leaves the
    # gradient non-finite, which NUTS treats as a divergence.
    assert np.isneginf(model.point_logps(point=bad_point)["y"])
    with pytest.raises(SamplingError, match="Initial evaluation"):
        model.check_start_vals([bad_point])

    # A raw evaluation raises the informative error instead.
    with pytest.raises(ParameterValueError, match="requires mu > 0"):
        model.logp().eval(bad_point)


def test_lognormal_prior_create_likelihood_forward_draws_zero_for_negative_mu():
    """Forward sampling is unprotected: pymc rewrites the mu > 0 check in any
    compiled function, so a non-positive mu becomes a -inf log-mean and the
    draw is exactly exp(-inf) = 0 rather than an error."""
    model, _ = _likelihood_model(mu_offset=-5.0)
    draws = pm.draw(model["y"], draws=10, random_seed=42)
    assert np.all(draws == 0)


def test_lognormal_prior_create_likelihood_negative_std():
    # softplus(2 * (log std - log mu)) needs std > 0, where the former
    # log1p((std / mu) ** 2) silently folded the sign, so std is guarded too.
    coords = {"date": np.arange(3)}
    observed = as_xtensor(np.array([1.0, 2.0, 1.5]), dims=("date",))
    likelihood = LogNormalPrior(std=Prior("Normal", mu=0, sigma=1), dims=("date",))
    with pm.Model(coords=coords) as model:
        likelihood.create_likelihood_variable(
            "y",
            mu=as_xtensor(np.full(3, 2.0), dims=("date",)),
            observed=observed,
            xdist=True,
        )
    bad_point = {"y_std": np.array(-0.5)}

    assert np.isneginf(model.point_logps(point=bad_point)["y"])
    with pytest.raises(ParameterValueError, match="std > 0"):
        model.logp().eval(bad_point)


def test_lognormal_prior_create_likelihood_mean_and_std_match_mu():
    """E[y] = mu and sd(y) = std, the claim the parameterization exists for."""
    m, s = 2.0, 0.3
    coords = {"date": np.arange(1)}
    likelihood = LogNormalPrior(std=s, dims=("date",))
    with pm.Model(coords=coords) as model:
        likelihood.create_likelihood_variable(
            "y",
            mu=as_xtensor(np.array([m]), dims=("date",)),
            observed=as_xtensor(np.array([1.0]), dims=("date",)),
            xdist=True,
        )
    draws = pm.draw(model["y"], draws=20_000, random_seed=42)

    np.testing.assert_allclose(draws.mean(), m, rtol=0.02)
    np.testing.assert_allclose(draws.std(), s, rtol=0.05)


def test_lognormal_prior_create_likelihood_mean_already_set_raises():
    prior = LogNormalPrior(mean=1.0, std=1.0)
    # The subclass keeps ``except MuAlreadyExistsError`` working while naming
    # the key that actually collides, which is ``mean``.
    with pytest.raises(MuAlreadyExistsError, match="The mean parameter"):
        prior.create_likelihood_variable(
            "y", mu=as_xtensor(np.ones(2), dims=("date",)), observed=None, xdist=True
        )


def test_lognormal_prior_create_likelihood_does_not_mutate():
    likelihood = LogNormalPrior(std=Prior("HalfNormal", sigma=0.5), dims=("date",))
    with pm.Model(coords={"date": np.arange(3)}):
        likelihood.create_likelihood_variable(
            "y",
            mu=as_xtensor(np.ones(3), dims=("date",)),
            observed=as_xtensor(np.ones(3), dims=("date",)),
            xdist=True,
        )
    assert "mean" not in likelihood.parameters


def test_lognormal_prior_create_likelihood_xdist_false_raises():
    prior = LogNormalPrior(std=1.0)
    with pytest.raises(NotImplementedError, match="only supports xdist=True"):
        prior.create_likelihood_variable("y", mu=None, observed=None)


@pytest.mark.parametrize(
    "y",
    [
        pytest.param(np.array([1.0, 0.0, 2.0]), id="zero"),
        pytest.param(np.array([1.0, -1.0, 2.0]), id="negative"),
        pytest.param(np.array([1.0, np.nan, 2.0]), id="nan"),
        pytest.param(np.array([1.0, np.inf, 2.0]), id="inf"),
    ],
)
def test_lognormal_prior_validate_observed_raises(y):
    likelihood = LogNormalPrior(std=1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        likelihood.validate_observed(y)


def test_lognormal_prior_validate_observed_passes():
    LogNormalPrior(std=1.0).validate_observed(np.array([[0.5, 1.0], [2.0, 3.0]]))


def test_lognormal_prior_std_only_round_trip():
    prior = LogNormalPrior(std=Prior("HalfNormal", sigma=0.5), dims=("date",))
    restored = deserialize(prior.to_dict())
    assert isinstance(restored, LogNormalPrior)
    assert restored == prior


def test_mmm_lognormal_prior_likelihood_save_load_round_trip(
    tmp_path, mock_pymc_sample
):
    """The std-only likelihood survives the attrs-serialization load path."""
    rng = np.random.default_rng(42)
    n = 20
    dates = pd.date_range("2024-01-01", periods=n, freq="W-MON")

    X = pd.DataFrame(
        {
            "date": dates,
            "channel_1": rng.integers(10, 100, n),
            "channel_2": rng.integers(10, 100, n),
        }
    )
    y = pd.Series(rng.uniform(50, 500, n), name="y")

    mmm = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="y",
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        model_config={
            "likelihood": LogNormalPrior(
                std=Prior("HalfNormal", sigma=0.5), dims=("date",)
            ),
        },
    )

    mmm.fit(X, y, chains=1, draws=10, tune=10, random_seed=42)

    save_path = tmp_path / "mmm_lognormal_prior_likelihood.nc"
    mmm.save(str(save_path))
    mmm_loaded = MMM.load(str(save_path), check=True)

    assert mmm == mmm_loaded
    assert isinstance(mmm_loaded.model_config["likelihood"], LogNormalPrior)
    assert mmm_loaded.model_config["likelihood"] == mmm.model_config["likelihood"]


@pytest.mark.parametrize(
    "data, expected",
    [
        pytest.param(
            {
                "distribution": "Normal",
                "mu": 10,
                "sigma": 1,
            },
            Prior("Normal", mu=10, sigma=1),
            id="another Prior",
        ),
        pytest.param(
            {
                "distribution": "Laplace",
                "mu": 1,
                "b": 2,
                "dims": ("x", "y"),
                "transform": "sigmoid",
            },
            Prior("Laplace", mu=1, b=2, dims=("x", "y"), transform="sigmoid"),
            id="Prior",
        ),
        pytest.param(
            {"distribution": "Normal", "mu": {"distribution": "Normal"}},
            Prior("Normal", mu=Prior("Normal")),
            id="Prior with nested distribution",
        ),
    ],
)
def test_alternative_prior_deserialize(data, expected) -> None:
    """The flat ``{"distribution": ...}`` YAML prior format round-trips."""
    assert deserialize(data) == expected


# --- NestedZeroSumPrior and its sigma options IndependentSigmas and VarianceBudget ---

G = Prior("Gamma", alpha=2, beta=20)
MU = Prior("Normal", dims="channel")
FAM = {"b0": 0, "b1": 0, "b2": 1, "b3": 1}  # brand -> family
TIER = {f"d{i}": t for i, t in enumerate([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])}  # dma -> tier
COORDS = {"channel": ["c0", "c1", "c2"], "brand": list(FAM), "dma": list(TIER)}
NEST = {"family": {}, "brand": {"within": "family", "map": FAM}}
LEVEL_HN = Prior("HalfNormal", sigma=0.1)  # dimless: it gets the dim beta_sigma_level
LEVEL_GAMMA = Prior(  # hierarchical: the levels share an unknown mean scale
    "Gamma", mu=Prior("HalfNormal", sigma=0.2), sigma=0.05
)


class ThreeScales:
    """A factory that is given the level dim but ignores it."""

    dims = ()

    def create_variable(self, name, xdist=False):
        return pm.HalfNormal(name, shape=3)


class XScales:
    """A factory that returns its scales as an XTensorVariable."""

    def __init__(self, dims=()):
        self.dims = dims

    def create_variable(self, name, xdist=False):
        return pmd.HalfNormal(name, sigma=0.2, dims=self.dims)


def hierarchy_dims(terms, interactions=None):
    """The dims an MMM over exactly the hierarchy would set: model-coord terms, interaction dims, channel."""
    parents = {t.get("within") for t in terms.values()}
    inner = [d for t in (interactions or {}).values() for d in t.get("dims", [])]
    return (*dict.fromkeys([k for k in terms if k not in parents] + inner), "channel")


def nzs(terms, interactions=None, **kwargs):
    """A NestedZeroSumPrior with the same sigma prior on every level and the hierarchy dims, unless given."""
    levels = [*terms, *(interactions or {})]
    kwargs.setdefault("sigma", IndependentSigmas({lv: G for lv in levels}))
    kwargs.setdefault("mu", MU)
    kwargs.setdefault("dims", hierarchy_dims(terms, interactions))
    return NestedZeroSumPrior(terms=terms, interactions=interactions, **kwargs)


def build(prior, coords=COORDS):
    with pm.Model(coords=coords) as model:
        beta = prior.create_variable("beta", xdist=True)
    return model, beta


def nested_over(n_parents, children=4, label=lambda i: f"f{i}"):
    """A prior with a brand term within a family term, `children` brands per family, and its coords."""
    brands = [f"b{i}" for i in range(n_parents * children)]
    fam = {b: label(i // children) for i, b in enumerate(brands)}
    terms = {"family": {}, "brand": {"within": "family", "map": fam}}
    return nzs(terms), {**COORDS, "brand": brands}


@pytest.mark.parametrize(
    ("terms", "interactions", "message"),
    [
        (  # YAML `brand: family`
            {"family": {}, "brand": "family"},
            None,
            "term brand: expected a mapping with the keys ['centered', 'within', 'map', 'parent_effect'], "
            "got the str 'family'",
        ),
        (
            {"family": {}, "brand": ["family"]},
            None,
            "term brand: expected a mapping with the keys ['centered', 'within', 'map', 'parent_effect'], "
            "got the list ['family']",
        ),
        (
            {"brand": {}, "dma": {}},
            {"brand_dma": "brand"},
            "interaction brand_dma: expected a mapping with the keys ['dims', 'centered'], got the str 'brand'",
        ),
    ],
    ids=["str_term", "list_term", "str_interaction"],
)
def test_a_spec_that_is_not_a_mapping_is_named(terms, interactions, message):
    # a str was read as a collection of keys, its characters
    levels = [*terms, *(interactions or {})]
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        NestedZeroSumPrior(
            mu=MU,
            terms=terms,
            interactions=interactions,
            sigma=IndependentSigmas({lv: G for lv in levels}),
        )


@pytest.mark.parametrize(
    ("terms", "interactions", "message"),
    [
        (
            {"a/b": {}, "c": {}},
            None,
            "term names ['a/b'] contain '/', which is not allowed in a variable name",
        ),
        (
            {"a": {}, "c": {}},
            {"a/x/c": {"dims": ["a", "c"]}},
            "interaction names ['a/x/c'] contain '/', which is not allowed in a variable name",
        ),
    ],
    ids=["term", "interaction"],
)
def test_a_slash_in_a_term_or_interaction_name_fails_at_construction(
    terms, interactions, message
):
    # these names are part of the variable names; pymc only fails on them once the model is partly built
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        nzs(terms, interactions)


def test_parent_labels_are_not_used_in_variable_names():
    # a '/' in a parent label (`Food/Bev`, `N/A`) is not allowed in a pymc variable name
    prior, coords = nested_over(2, label=lambda i: ["Food/Bev", "N/A"][i])
    model, _ = build(prior, coords)
    assert sorted(model.named_vars) == [
        "beta",
        "beta_brand",
        "beta_brand_raw",
        "beta_brand_raw_offset",
        "beta_family",
        "beta_family_offset",
        "beta_mu",
        "beta_sigma",
        "beta_sigma_brand",
        "beta_sigma_family",
    ]
    assert model.named_vars_to_dims["beta_brand"] == ("channel", "brand")
    # the parent labels are only labels of the `family` coord
    assert model.coords["family"] == ("Food/Bev", "N/A")


def test_the_number_of_free_rvs_does_not_grow_with_the_number_of_parents():
    def n_free_rvs(n_parents):
        model, _ = build(*nested_over(n_parents))
        return len(model.free_RVs), len(model.coords)

    # beta_mu, the two level scales, the family offset and the one Normal of the brands; the coords are the
    # model's channel, brand and dma, the parents `family` and the level dim
    assert n_free_rvs(5) == n_free_rvs(40) == (5, 5)


def grouped_offset(scale=0.5, centered=True):
    from pymc_marketing.special_priors import _GroupedZeroSumOffset

    brands = [f"b{i}" for i in range(7)]
    codes = np.array([0, 0, 0, 0, 1, 1, 2])  # parents with 4, 2 and 1 children
    offset = _GroupedZeroSumOffset(
        scale=scale,
        centered=centered,
        dims=("channel", "brand"),
        child_dim="brand",
        codes=codes,
    )
    return offset, {"channel": ["c0", "c1"], "brand": brands}, codes


@pytest.mark.parametrize("centered", [True, False])
def test_a_grouped_offset_is_zero_sum_in_each_group_with_the_zero_sum_scale(centered):
    offset, coords, codes = grouped_offset(scale=0.5, centered=centered)
    assert isinstance(offset, VariableFactory)
    draws = sample_prior(offset, coords=coords, draws=20_000, random_seed=0)[
        "variable"
    ].values.squeeze(0)  # (draw, channel, brand)
    for g in np.unique(codes):
        assert np.abs(draws[..., codes == g].sum(-1)).max() < 1e-10
    n = np.bincount(codes)[codes]
    np.testing.assert_allclose(
        draws.std(axis=(0, 1)), 0.5 * np.sqrt(1 - 1 / n), atol=0.01
    )


FAM_STR = {"b0": "z", "b1": "a", "b2": "a", "b3": "m"}  # str parents, one lone child


@pytest.mark.parametrize(
    ("terms", "term", "parent", "expected"),
    [
        (NEST, "brand", 0, ["b0", "b1"]),  # int parents
        (NEST, "brand", 1, ["b2", "b3"]),
        (
            {"family": {}, "brand": {"within": "family", "map": FAM_STR}},
            "brand",
            "z",
            ["b0"],  # a lone child
        ),
        (
            {"family": {}, "brand": {"within": "family", "map": FAM_STR}},
            "brand",
            "a",
            ["b1", "b2"],
        ),
    ],
    ids=["int_parent_0", "int_parent_1", "lone_child", "str_parent"],
)
def test_children_are_the_labels_of_a_term_under_one_parent(
    terms, term, parent, expected
):
    assert nzs(terms).children(term, parent) == expected


@pytest.mark.parametrize(
    ("term", "parent", "match"),
    [
        ("dma", 0, r"unknown term 'dma'; terms: \['family', 'brand'\]"),
        ("family", 0, r"term family has no `within`, so it has no parents"),
        ("brand", 7, r"term brand has no parent 7; parents: \[0, 1\]"),
    ],
    ids=["unknown_term", "term_without_parents", "unknown_parent"],
)
def test_children_name_the_problem(term, parent, match):
    with pytest.raises(ValueError, match=match):
        nzs(NEST).children(term, parent)


def test_children_select_one_parents_offsets_from_the_draws():
    prior = nzs(NEST, transform=None)
    with pm.Model(coords=COORDS):
        prior.create_variable("beta", xdist=True)
        prior_draws = pm.sample_prior_predictive(draws=20, random_seed=0).prior
    brands = prior.children("brand", 0)
    offsets = prior_draws["beta_brand"].sel(brand=brands)
    assert offsets.sizes["brand"] == len(brands) == 2
    assert abs(offsets.sum("brand")).max() < 1e-12  # zero-sum within the parent


def test_the_level_order_does_not_depend_on_the_order_of_terms():
    # stored draws of a vector on the level dim (a VarianceBudget share, a Prior sigma) are replayed by
    # position, so priors that compare equal need the same level order
    sigma = VarianceBudget(G, {"family": 0.5, "brand": 0.5})
    family, brand = NEST["family"], NEST["brand"]
    a = nzs({"family": family, "brand": brand}, sigma=sigma)
    b = nzs({"brand": brand, "family": family}, sigma=sigma)
    c = NestedZeroSumPrior.from_dict(yaml.safe_load(yaml.safe_dump(a.to_dict())))
    assert a == b == c
    orders = [list(build(p)[0].coords["beta_sigma_level"]) for p in (a, b, c)]
    assert orders == [["brand", "family"]] * 3


def test_the_coefficient_keeps_floatx():
    with pytensor.config.change_flags(floatX="float32"):
        model, _ = build(nzs(NEST))
        dtypes = {v: model[v].dtype for v in ("beta", "beta_brand", "beta_family")}
    assert dtypes == {
        "beta": "float32",
        "beta_brand": "float32",
        "beta_family": "float32",
    }


@pytest.mark.parametrize(
    ("terms", "interactions", "duplicate"),
    [
        ({"brand": {}}, {"brand_offset": {"dims": ["brand"]}}, "beta_brand_offset"),
        (NEST, {"brand_raw_offset": {"dims": ["brand"]}}, "beta_brand_raw_offset"),
    ],
    ids=["crossed_term", "child_term"],
)
def test_a_non_centered_offset_name_is_checked_before_any_variable_is_created(
    terms, interactions, duplicate
):
    with pm.Model(coords=COORDS) as model:
        with pytest.raises(
            ValueError,
            match=re.escape(f"RV or coord names ['{duplicate}'] are generated twice"),
        ):
            nzs(terms, interactions).create_variable("beta", xdist=True)
    assert not model.named_vars


def test_a_centered_offset_creates_no_offset_name_to_collide_with():
    model, _ = build(
        nzs({"brand": {}}, {"brand_offset": {"dims": ["brand"]}}, centered=True)
    )
    assert sorted(model.named_vars) == [
        "beta",
        "beta_brand",
        "beta_brand_offset",
        "beta_mu",
        "beta_sigma",
        "beta_sigma_brand",
        "beta_sigma_brand_offset",
    ]


INT_MAP = {
    "0": "f0",
    "1": "f0",
    "2": "f1",
    "3": "f1",
}  # map keys are str, the coord labels are ints


def test_children_can_return_the_labels_of_the_coord():
    prior = nzs(
        {"family": {}, "brand": {"within": "family", "map": INT_MAP}}, transform=None
    )
    with pm.Model(coords=COORDS | {"brand": [0, 1, 2, 3]}):
        prior.create_variable("beta", xdist=True)
        draws = pm.sample_prior_predictive(draws=5, random_seed=0).prior
    brands = prior.children("brand", "f0", labels=draws["brand"].values)
    assert brands == [0, 1]
    assert draws["beta_brand"].sel(brand=brands).sizes["brand"] == 2


def test_children_name_labels_missing_from_the_map():
    prior = nzs({"family": {}, "brand": {"within": "family", "map": INT_MAP}})
    with pytest.raises(
        ValueError, match=re.escape("term brand: `map` has no labels ['9']")
    ):
        prior.children("brand", "f0", labels=[0, 9])


def test_a_grouped_offset_is_distributed_as_a_zero_sum_normal_per_group():
    # the same distribution as one ZeroSumNormal per group; the variables and the graph differ
    offset, coords, codes = grouped_offset(scale=0.5)
    draws = sample_prior(offset, coords=coords, draws=40_000, random_seed=0)[
        "variable"
    ].values.squeeze(0)[:, 0, :]  # (draw, child), one channel
    for group, n in ((0, 4), (1, 2)):  # the group of one child is 0
        reference = pm.draw(
            pm.ZeroSumNormal.dist(sigma=0.5, shape=n), 40_000, random_seed=1
        )
        np.testing.assert_allclose(
            np.cov(draws[:, codes == group].T), np.cov(reference.T), atol=0.01
        )
    covariance = np.cov(draws.T)  # the groups are independent
    assert np.abs(covariance[np.ix_(codes == 0, codes == 1)]).max() < 0.01


@pytest.mark.parametrize(
    "sigma",
    [
        Prior("HalfNormal", sigma=[1.0, 0.01]),
        Prior("HalfNormal", sigma=np.array([1.0, 0.01])),
        Prior("HalfNormal", sigma=Prior("HalfNormal", sigma=[1.0, 0.01])),
        Prior("HalfNormal", sigma=xr.DataArray([1.0, 0.01], dims="x")),
    ],
    ids=["list", "array", "nested_prior", "data_array"],
)
def test_a_positional_sigma_is_rejected(sigma):
    # one value per level would be matched to the levels by position, i.e. by the order of `terms`
    with pytest.raises(ValueError, match=r"one value per level.*IndependentSigmas"):
        nzs(NEST, sigma=sigma)


def test_the_order_of_terms_does_not_change_the_model():
    # equality ignores the order of terms, so the order must not matter: sigma is keyed by label or exchangeable
    sigma = IndependentSigmas({"family": G, "brand": Prior("Gamma", alpha=2, beta=2)})
    family, brand = NEST["family"], NEST["brand"]
    a = nzs({"family": family, "brand": brand}, sigma=sigma)
    b = nzs({"brand": brand, "family": family}, sigma=sigma)
    assert a == b
    assert hash(a) == hash(b)
    (ma, beta_a), (mb, beta_b) = build(a), build(b)
    rng = np.random.default_rng(0)
    point = {
        v.name: rng.normal(size=np.shape(ma.initial_point()[v.name])) * 0.3
        for v in ma.value_vars
    }
    np.testing.assert_allclose(beta_at(ma, beta_a, point), beta_at(mb, beta_b, point))
    # yaml.safe_dump sorts the keys, so the terms come back reordered: still the same model
    c = NestedZeroSumPrior.from_dict(yaml.safe_load(yaml.safe_dump(a.to_dict())))
    assert list(c.terms) == ["brand", "family"]
    assert c == a
    mc, beta_c = build(c)
    np.testing.assert_allclose(beta_at(ma, beta_a, point), beta_at(mc, beta_c, point))


@pytest.mark.parametrize(
    "sigma",
    [
        IndependentSigmas(
            {lv: LogNormalPrior(mean=0.2, std=0.1) for lv in ("family", "brand")}
        ),
        VarianceBudget(
            LogNormalPrior(mean=0.2, std=0.1), {"family": 0.5, "brand": 0.5}
        ),
    ],
    ids=["independent", "budget"],
)
def test_xdist_only_priors_work_inside_the_sigma_options(sigma):
    # LogNormalPrior only supports xdist=True, so the sigma options must create their levels that way
    model, _ = build(nzs(NEST, sigma=sigma))
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def beta_at(model, beta, point):
    """Beta as a function of the model's value variables."""
    fn = model.compile_fn(
        model.replace_rvs_by_values([beta.values])[0],
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    return fn(point)


# --- Constructor checks -------------------------------------------------------


def test_dims_default_to_none_and_are_accepted_in_any_order():
    # the MMM sets dims (`with_default_prior_dims`), so nothing is hard coded
    interactions = {"brand_dma": {"dims": ["brand", "dma"]}}
    prior = nzs({**NEST, "dma": {}}, interactions, dims=None)
    assert prior.dims is None
    assert prior.interactions == {"brand_dma": Interaction(dims=["brand", "dma"])}
    with pytest.raises(ValueError, match="beta: `dims` is unset"):
        build(prior)
    prior.dims = ("brand", "dma", "channel")
    model, beta = build(prior)
    prior.dims = ["channel", "dma", "brand"]  # a list, as YAML gives it
    model_t, beta_t = build(prior)
    assert beta_t.dims == ("channel", "dma", "brand")
    point = model.initial_point() | {"beta_mu": np.array([0.1, -0.2, 0.3])}
    np.testing.assert_array_equal(
        beta_at(model, beta, point).transpose(2, 1, 0), beta_at(model_t, beta_t, point)
    )


def test_terms_and_interactions_are_objects_or_plain_dicts():
    objects = NestedZeroSumPrior(
        mu=MU,
        terms={
            "family": Term(),
            "brand": Term(within="family", map=FAM),
            "dma": Term(),
        },
        interactions={"bd": Interaction(dims=["brand", "dma"], centered=True)},
        sigma=IndependentSigmas({"family": G, "brand": G, "dma": G, "bd": G}),
        dims=("brand", "dma", "channel"),
    )
    dicts = nzs(
        {**NEST, "dma": {}}, {"bd": {"dims": ["brand", "dma"], "centered": True}}
    )
    assert objects == dicts and hash(objects) == hash(dicts)
    assert dicts.terms["brand"] == Term(within="family", map=FAM)
    # written back as plain dicts, defaults left out, so YAML stays readable
    assert dicts.to_dict()["terms"] == {
        "family": {},
        "brand": {"within": "family", "map": FAM},
        "dma": {},
    }
    assert dicts.to_dict()["interactions"] == {
        "bd": {"dims": ["brand", "dma"], "centered": True}
    }


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: Term(map=FAM), "`map` and `parent_effect` need `within`"),
        (lambda: Term(parent_effect=False), "`map` and `parent_effect` need `within`"),
        (lambda: Term(within="family"), "is `within: family` but has no `map`"),
        (lambda: Interaction(), r"`dims` must be non-empty, each dim once; got \[\]"),
        (
            lambda: Interaction(dims=None),
            "`dims` must be non-empty",
        ),  # was a bare TypeError
        (lambda: Interaction(dims=["dma", "dma"]), "each dim once"),  # failed at build
        (lambda: Interaction(dims=["dma"], centered=0), "`centered` must be a bool"),
        # a quoted YAML "no" was truthy, so the level was silently centered
        (lambda: Term(centered="no"), "centered='no'"),
        (lambda: Term(within=1, map=FAM), "within=1"),  # failed at build
        (
            lambda: Term(within="family", map=FAM, parent_effect="no"),
            "parent_effect='no'",
        ),
        (
            lambda: Interaction.coerce("i", None),
            "interaction i: `dims` must be non-empty",
        ),
    ],
)
def test_term_and_interaction_validate_themselves(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_a_term_with_no_value_is_an_empty_term():  # YAML `brand:` gave a bare TypeError
    assert Term.coerce("brand", None) == Term()


def test_an_interaction_with_a_str_dims_is_one_dim():
    assert Interaction(dims="brand").dims == ["brand"]


def test_raw_dicts_are_deserialized():  # YAML resolve() hands nested priors over as plain dicts
    sigma = {
        "__type__": "pymc_marketing.special_priors.IndependentSigmas",
        "priors": {lv: G.to_dict() for lv in NEST},
    }
    prior = NestedZeroSumPrior(
        mu={"distribution": "Normal", "dims": "channel"}, terms=NEST, sigma=sigma
    )
    assert isinstance(prior.mu, Prior)
    assert isinstance(prior.sigma, IndependentSigmas)
    assert prior.sigma.priors["brand"] == G
    assert all(isinstance(t, Term) for t in prior.terms.values())


@pytest.mark.parametrize(
    "mu", [0.5, 1, [0.1, -0.2, 0.3]], ids=["float", "int", "per_channel"]
)
def test_constant_mu_fixes_the_channel_mean_and_round_trips(mu):
    prior = nzs({"brand": {}}, mu=mu)
    model, beta = build(prior)
    assert "beta_mu" not in model.named_vars
    # brand sums to zero, so the brand mean of log beta is mu
    np.testing.assert_allclose(
        np.log(pm.draw(beta.values)).mean(axis=0), np.broadcast_to(mu, 3), atol=1e-6
    )
    data = json.loads(json.dumps(prior.to_dict()))
    assert NestedZeroSumPrior.from_dict(data).to_dict() == prior.to_dict()


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (
            lambda: IndependentSigmas({"brand": 0.5}),
            r"IndependentSigmas priors\['brand'\] must be a Prior",
        ),
        (
            lambda: VarianceBudget(0.5, {"brand": 1.0}),
            "VarianceBudget total must be a Prior",
        ),
        (lambda: nzs({"brand": {}}, sigma=0.5), "`sigma` must be IndependentSigmas"),
        (  # a dimmed scale broadcast onto the wrong axis and broke the zero sum
            lambda: IndependentSigmas({"brand": Prior("HalfNormal", dims="channel")}),
            r"IndependentSigmas priors\['brand'\] must be scalar \(no dims\)",
        ),
        (
            lambda: VarianceBudget(Prior("HalfNormal", dims="channel"), {"brand": 1.0}),
            r"VarianceBudget total must be scalar \(no dims\)",
        ),
        (lambda: VarianceBudget(G, {"brand": 1.0}, -1), "concentration > 0"),
        (lambda: VarianceBudget(G, {}), "non-empty share"),
    ],
    ids=[
        "independent",
        "budget",
        "sigma",
        "independent_dims",
        "budget_total_dims",
        "budget_concentration",
        "budget_empty_share",
    ],
)
def test_invalid_sigma_raises_naming_the_field(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_the_callers_spec_is_not_mutated():
    terms = {"family": {}, "brand": {"within": "family", "map": {101: 0, 102: 1}}}
    nzs(terms)
    assert list(terms["brand"]["map"]) == [101, 102]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (  # sigma priors go in the top-level `sigma` only
            {"terms": {**NEST, "family": {"sigma": G}}},
            r"term family: unknown keys \['sigma'\]",
        ),
        ({"terms": NEST, "sigma": None}, "keyed by exactly"),
        (
            {"terms": NEST, "sigma": IndependentSigmas({"family": G})},
            "keyed by exactly",
        ),
        (
            {
                "terms": NEST,
                "sigma": VarianceBudget(G, {"family": 0.5, "brand": 0.5}),
                "interactions": {"bd": {"dims": ["brand", "dma"]}},
            },
            "keyed by exactly",
        ),
        (
            {
                "terms": {"brand": {"within": "family", "map": {}}},
                "sigma": IndependentSigmas({"brand": G}),
            },
            r"within: family`, which is not a term.*parent_effect: false",
        ),
        (
            {"terms": {"family": {}, "brand": {"within": "family"}}},
            "has no `map`",
        ),
        (
            {"terms": NEST, "interactions": {"bd": {"dims": ["brand", "dma"]}}},
            r"interaction dims \['dma'\] are not terms that are model coords",
        ),
        (
            {"terms": NEST, "interactions": {"bf": {"dims": ["brand", "family"]}}},
            r"interaction dims \['family'\] are not terms that are model coords",
        ),
        (  # a `within` parent is not a coord, so not a dim
            {"terms": NEST, "dims": ("family", "brand", "channel")},
            r"must be the hierarchy dims \['brand'\]",
        ),
        (
            {"terms": NEST, "dims": ("channel",)},
            r"must be the hierarchy dims \['brand'\]",
        ),
        (  # e.g. an MMM over (brand, dma) with a brand-only hierarchy
            {"terms": NEST, "dims": ("brand", "channel", "control")},
            r"got base dims \['channel', 'control'\]\. .*pass `dims` explicitly",
        ),
        ({"terms": NEST, "dims": ("brand",)}, r"mu must have the base dims \(\)"),
        (
            {
                "terms": {
                    **NEST,
                    "region": {},
                    "family": {"within": "region", "map": {0: "r0"}},
                }
            },
            "nesting is at most two levels per tree",
        ),
        ({"terms": NEST, "centered": "no"}, "`centered` must be a bool, got 'no'"),
        ({"terms": NEST, "centered": None}, "`centered` must be a bool, got None"),
        ({"terms": NEST, "mu": "abc"}, "constant mu must be a number"),
        ({"terms": NEST, "mu": [[0.1], [0.2]]}, "constant mu must be a number"),
        (
            {"terms": NEST, "mu": [0.1, 0.2, 0.3, 0.4], "dims": ("brand",)},
            r"mu must have the base dims \(\)",
        ),
    ],
)
def test_invalid_arguments_raise(kwargs, match):
    terms = kwargs.pop("terms")
    with pytest.raises(ValueError, match=match):
        nzs(terms, **kwargs)


register_tensor_transform("nzs_test_cube", lambda x: x**3)


@pytest.mark.parametrize(
    ("transform", "fn"),
    [
        ("exp", np.exp),  # the default
        ("sigmoid", lambda x: 1 / (1 + np.exp(-x))),
        ("nzs_test_cube", lambda x: x**3),  # registered like a Prior transform
    ],
)
def test_transform_is_applied_to_the_untransformed_coefficient(transform, fn):
    beta, beta_id = (
        pm.draw(build(nzs(NEST, transform=t))[1].values, 20, random_seed=1)
        for t in (transform, None)
    )
    assert (beta_id < 0).any()
    np.testing.assert_allclose(fn(beta_id), beta, rtol=1e-6)


def test_an_unknown_transform_fails_at_construction():
    with pytest.raises(UnknownTransformError, match="'logexp' not present"):
        nzs(NEST, transform="logexp")


CROSS = {"brand": {}, "dma": {}}


@pytest.mark.parametrize(
    ("terms", "interactions", "dims", "mu", "base"),
    [
        (NEST, None, ("brand",), Prior("Normal"), ()),
        (NEST, None, ("brand",), 0.5, ()),
        (CROSS, {"brand_dma": {"dims": ["brand", "dma"]}}, ("brand", "dma"), 0.5, ()),
        (
            NEST,
            None,
            ("brand", "control"),
            Prior("Normal", dims="control"),
            ("control",),
        ),
        (NEST, None, ("brand", "control"), [0.1, -0.2], ("control",)),
    ],
    ids=["none", "none_constant", "none_crossed", "control", "control_constant"],
)
def test_base_dim_is_optional_and_need_not_be_channel(
    terms, interactions, dims, mu, base
):
    prior = nzs(terms, interactions, mu=mu, dims=dims, transform=None)
    coords = {"brand": list(FAM), "dma": list(TIER), "control": ["x0", "x1"]}
    model, beta = build(prior, coords)
    assert beta.dims == dims
    n_base = tuple(len(coords[d]) for d in base)
    offsets = [
        v for v in model.free_RVs if "sigma" not in v.name and v.name != "beta_mu"
    ]
    # every offset spans the base dims, first
    assert {model.named_vars_to_dims[v.name][: len(base)] for v in offsets} == {base}
    # every offset is zero-sum over a hierarchy dim, so the hierarchy mean of beta is mu
    mu_var = model["beta_mu"] if "beta_mu" in model.named_vars else pt.as_tensor(mu)
    beta_draw, mu_draw = pm.draw([beta.values, mu_var], random_seed=2)
    hierarchy_axes = tuple(range(len(dims) - len(base)))
    np.testing.assert_allclose(
        beta_draw.mean(axis=hierarchy_axes), np.broadcast_to(mu_draw, n_base), atol=1e-6
    )


# --- Regressions: one test per fixed bug --------------------------------------


def test_parent_effect_false_with_a_parent_term_raises():
    # was silently ignored: beta still depended on beta_family
    terms = {
        "family": {},
        "brand": {"within": "family", "map": FAM, "parent_effect": False},
    }
    with pytest.raises(
        ValueError, match="brand sets `parent_effect: false` but family is a term"
    ):
        nzs(terms)


def test_mixed_str_and_int_parent_labels_raise():  # np.unique merged 0 and "0" into one parent
    brand_map = {"b0": 0, "b1": 0, "b2": "0", "b3": "0"}
    with pytest.raises(ValueError, match="mixes str and non-str family labels"):
        nzs({"family": {}, "brand": {"within": "family", "map": brand_map}})


@pytest.mark.parametrize(
    "share",
    [
        {"family": 0.5, "brand": 0.6},
        {"family": 0.0, "brand": 1.0},
        {"family": -0.1, "brand": 1.1},
    ],
)
def test_variance_budget_shares_off_the_simplex_raise(share):
    # sum != 1 rescaled the concentration; a share <= 0 built an invalid Dirichlet
    with pytest.raises(ValueError, match="must be > 0 for every level and sum to 1"):
        VarianceBudget(G, share)


@pytest.mark.parametrize(
    ("terms", "interactions", "match"),
    [
        (
            # the pre-rename British spelling of the key is now rejected, not ignored
            {"brand": {"centred": True}},
            None,
            r"brand: unknown keys \['centred'\]; "
            r"allowed: \['centered', 'within', 'map', 'parent_effect'\]",
        ),
        (
            {
                "family": {},
                "brand": {"within": "family", "map": FAM, "parent_efect": 0},
            },
            None,
            r"unknown keys \['parent_efect'\]",
        ),
        (
            {"family": {}, "brand": {"whithin": "family", "map": FAM}},
            None,
            r"unknown keys \['whithin'\]",
        ),
        (
            {"brand": {}},
            {"i": {"dims": ["brand"], "centred": True}},
            r"i: unknown keys \['centred'\]; allowed: \['dims', 'centered'\]",
        ),
    ],
)
def test_unknown_spec_keys_raise(terms, interactions, match):
    # a misspelled key was ignored (e.g. the level was built non-centered)
    with pytest.raises(ValueError, match=match):
        nzs(terms, interactions)


@pytest.mark.parametrize(
    ("terms", "dims"),
    [
        ({"brand": {}}, ("brand", "channel")),
        (
            {"brand": {}, "tier": {}, "dma": {"within": "tier", "map": TIER}},
            ("brand", "dma", "channel"),
        ),
    ],
)
def test_crossed_term_without_within_is_in_the_dims_and_builds(terms, dims):
    # was left out of dims, so the transpose crashed
    prior = nzs(terms)
    model, beta = build(prior)
    assert prior.dims == beta.dims == dims
    assert "beta_brand" in model.named_vars


def test_int_child_labels_survive_the_json_round_trip():
    # JSON made the map keys str, so the reloaded prior missed every int label
    brand_map = {101: 0, 102: 0, 103: 1, 104: 1}
    prior = nzs({"family": {}, "brand": {"within": "family", "map": brand_map}})
    loaded = deserialize(json.loads(json.dumps(prior.to_dict())))
    coords = COORDS | {"brand": list(brand_map)}
    (model_p, beta_p), (model_l, beta_l) = build(prior, coords), build(loaded, coords)
    point = {
        k: v + 0.1 * np.arange(v.size).reshape(v.shape)
        for k, v in model_p.initial_point().items()
    }
    assert loaded == prior
    assert model_p.compile_logp()(point) == model_l.compile_logp()(point)
    np.testing.assert_array_equal(
        beta_at(model_p, beta_p, point), beta_at(model_l, beta_l, point)
    )


GROUPED = {  # one parent term, two children within it
    "grp": {},
    "brand": {"within": "grp", "map": {b: f"g{i // 2}" for i, b in enumerate(FAM)}},
    "dma": {"within": "grp", "map": {d: f"g{i // 5}" for i, d in enumerate(TIER)}},
}


@pytest.mark.parametrize(
    ("make", "coords", "match"),
    [
        (
            lambda: nzs(GROUPED),
            COORDS,
            r"terms \['grp'\] have more than one child `within` them",
        ),
        (
            lambda: nzs(
                NEST,
                {"family": {"dims": ["brand"]}},
                sigma=IndependentSigmas({"family": G, "brand": G}),
            ),
            COORDS,
            r"^interactions \['family'\] are named like terms",  # was "self.interactions"
        ),
        (
            lambda: nzs({"tier": {}, "dma": {"within": "tier", "map": TIER}}),
            COORDS | {"tier": ["t0", "t1", "t2"]},
            r"model coord 'tier' has labels \['t0', 't1', 't2'\], but beta needs the labels \[0, 1, 2\]",
        ),
        (
            lambda: nzs(NEST, {"brand_raw": {"dims": ["brand"]}}),
            COORDS,
            # both the raw Normal of `brand` and the non-centered interaction `brand_raw` create these
            r"RV or coord names \['beta_brand_raw', 'beta_brand_raw_offset'\] are generated twice",
        ),
        (  # beta_sigma_brand is made inside the IndependentSigmas: pymc's own error
            lambda: nzs({"sigma_brand": {}, "brand": {}}),
            COORDS | {"sigma_brand": ["x", "y"]},
            "Variable name beta_sigma_brand already exists",
        ),
        (
            lambda: nzs(NEST, sigma=0.1),
            COORDS,
            "`sigma` must be IndependentSigmas or VarianceBudget",
        ),
        (  # the class sets the level dim; a spelled one is not accepted
            lambda: nzs(NEST, sigma=Prior("HalfNormal", dims="beta_sigma_level")),
            COORDS,
            r"or a prior with no dims, which gets the dim `<name>_sigma_level` \(got dims \('beta_sigma_level',\)\)",
        ),
        (
            lambda: nzs(NEST, sigma=XScales(dims="beta_sigma_level")),
            COORDS,
            r"\(got dims \('beta_sigma_level',\)\)",  # a str dims is one dim, not its characters
        ),
        (  # dims are set by the prior, for every sigma option
            lambda: nzs(NEST, sigma=IndependentSigmas({"family": G, "brand": G}, "x")),
            COORDS,
            r"\(got dims \('x',\)\)",
        ),
        (
            lambda: nzs(NEST, sigma=Prior("HalfNormal", dims=("channel",))),
            COORDS,
            r"or a prior with no dims, .* \(got dims \('channel',\)\)",
        ),
        (  # a term `sigma` creates beta_sigma, the factory's own variable
            lambda: nzs({"sigma": {}, "brand": {}}, sigma=LEVEL_HN),
            COORDS | {"sigma": ["x", "y"]},
            r"RV or coord names \['beta_sigma'\] are generated twice",
        ),
        (
            lambda: nzs(NEST, sigma=ThreeScales()),
            COORDS,
            r"`sigma` must return an xtensor on the dims \('beta_sigma_level',\).*got dims None",
        ),
        (
            lambda: nzs(NEST, sigma=LEVEL_HN),
            COORDS | {"beta_sigma_level": ["family", "brand"]},  # not sorted
            r"coord 'beta_sigma_level' has labels \['family', 'brand'\], "
            r"but beta needs the labels \['brand', 'family'\] \(the sigma levels, sorted",
        ),
        (
            lambda: nzs(NEST),
            {k: v for k, v in COORDS.items() if k != "channel"},
            "the model has no `channel` coord",
        ),
        (
            lambda: nzs({"brand": {}}, {"i": {}}),
            COORDS,
            "interaction i: `dims` must be non-empty",
        ),
        (
            lambda: nzs(
                {"family": {}, "brand": {"within": "family", "map": FAM | {"b3": None}}}
            ),
            COORDS,
            "`map` has a None family label",
        ),
        (
            lambda: nzs({"region": {}, "brand": {}}),
            COORDS,
            r"term region: not a model coord",
        ),
        (
            lambda: nzs(
                {
                    "family": {},
                    "brand": {"within": "family", "map": FAM},
                    "tier": {},
                    "dma": {"within": "tier", "map": {"d0": 0}},
                }
            ),
            COORDS,
            r"term dma: `map` has no tier for data labels \['d1'",
        ),
        (
            lambda: nzs(
                {"family": {}, "brand": {"within": "family", "map": FAM | {"b9": 1}}}
            ),
            COORDS,
            r"term brand: `map` keys \['b9'\] are not brand labels in the data",
        ),
        (  # was pytensor's Alloc error
            lambda: nzs(NEST, mu=[0.1, 0.2]),
            COORDS,
            r"constant mu has shape \(2,\); expected \(\) or \(3,\)",
        ),
        (  # hyperprior names are not checked: pymc's own error
            lambda: nzs(
                {"sigma_sigma": {}},
                sigma=Prior("HalfNormal", sigma=G),
            ),
            COORDS | {"sigma_sigma": ["x", "y"]},
            "Variable name beta_sigma_sigma already exists",
        ),
        (
            lambda: nzs({"mu_sigma": {}}, mu=Prior("Normal", sigma=G, dims="channel")),
            COORDS | {"mu_sigma": ["x", "y"]},
            "Variable name beta_mu_sigma already exists",
        ),
    ],
    ids=[
        "shared_parent",
        "interaction_named_like_term",
        "parent_is_coord",
        "rv_name_collision",
        "sigma_name_collision",
        "sigma_is_not_a_prior",
        "sigma_with_the_level_dim",
        "sigma_factory_with_a_str_dim",
        "sigma_option_with_dims",
        "sigma_with_another_dim",
        "factory_sigma_name_collision",
        "sigma_of_the_wrong_length",
        "level_coord_with_other_labels",
        "no_channel",
        "interaction_without_dims",
        "none_parent",
        "orphan_term",
        "map_misses_a_label",
        "map_has_an_extra_label",
        "constant_mu_of_the_wrong_length",
        "sigma_hyperprior_name_collision",
        "mu_hyperprior_name_collision",
    ],
)
def test_spec_problems_raise_a_valueerror_naming_them(make, coords, match):
    # were: duplicate RV, AttributeError, KeyError, TypeError
    with pytest.raises(ValueError, match=match):
        build(make(), coords)


def test_variance_budget_level_coord_is_named_per_prior():
    # a plain `level` clashed with a user `level` coord (or a second budget)
    prior = nzs(NEST, sigma=VarianceBudget(G, {"family": 0.5, "brand": 0.5}))
    model, _ = build(prior, COORDS | {"level": ["a", "b"]})
    assert list(model.coords["beta_sigma_level"]) == ["brand", "family"]  # sorted
    assert model.coords["level"] == ("a", "b")


@pytest.mark.parametrize("centered", [True, False])
def test_from_dict_defaults_centered_to_false_like_init(centered):
    # and centered=True still survives the JSON round trip
    prior = nzs(NEST, centered=centered)
    data = {"mu": MU, "terms": NEST, "sigma": prior.sigma}
    assert NestedZeroSumPrior.from_dict(data).centered is False
    assert deserialize(json.loads(json.dumps(prior.to_dict()))).centered is centered


@pytest.mark.parametrize(
    ("terms", "interactions", "coords", "match"),
    [
        (
            {"family": {}, "brand": {"within": "family", "map": {b: b for b in FAM}}},
            None,
            COORDS,
            "every family has one brand",
        ),
        (
            {"family": {}, "brand": {"within": "family", "map": {b: 0 for b in FAM}}},
            None,
            COORDS,
            "term family has one label",
        ),
        ({"brand": {}}, None, COORDS | {"brand": ["b0"]}, "term brand has one label"),
        (
            {"brand": {}, "dma": {}},
            {"bd": {"dims": ["brand", "dma"]}},
            COORDS | {"dma": ["d0"]},
            r"interaction bd: dims \['dma'\] have one label",
        ),
    ],
    ids=[
        "singleton_children",
        "one_parent",
        "one_crossed_label",
        "one_interaction_label",
    ],
)
def test_unused_sigma_warns(terms, interactions, coords, match):
    # that level's offsets are identically 0
    with pytest.warns(UserWarning, match=match):
        build(nzs(terms, interactions), coords)


def test_a_term_named_mu_builds_with_a_constant_mu():
    # no beta_mu is created, so there is no collision (was a false positive)
    model, _ = build(nzs({"mu": {}}, mu=0.0), COORDS | {"mu": ["x", "y"]})
    assert "beta_mu" in model.named_vars and "beta_sigma_mu" in model.named_vars


@pytest.mark.parametrize(
    ("mu", "sigma"),
    [
        (LogNormalPrior(mean=1, std=0.5, dims="channel"), LEVEL_HN),
        (LaplacePrior(mu=0, b=1, dims="channel"), LEVEL_HN),
        (MU, LogNormalPrior(mean=0.2, std=0.1)),
    ],
    ids=["lognormal_mu", "laplace_mu", "lognormal_sigma"],
)
def test_xdist_only_special_priors_build_as_mu_or_sigma(mu, sigma):
    # were "NotImplementedError: ... only supports xdist=True" at build
    model, beta = build(nzs(NEST, mu=mu, sigma=sigma))
    assert model.named_vars_to_dims["beta_sigma"] == ("beta_sigma_level",)
    assert beta.dims == ("brand", "channel")
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_a_str_dims_of_a_mu_factory_is_one_dim():
    # a non-Prior factory keeps a str dims as is; it was split into characters
    model, _ = build(nzs(NEST, mu=XScales(dims="channel"), sigma=XScales()))
    assert model.named_vars_to_dims["beta_mu"] == ("channel",)


def test_offsets_have_named_dims():
    # were `*_dim_0`: parents, crossed terms and interactions are on model coords, children on their own coord
    model, _ = build(nzs(NEST | {"dma": {}}, {"brand_dma": {"dims": ["brand", "dma"]}}))
    assert model.named_vars_to_dims["beta_family"] == ("channel", "family")
    assert model.named_vars_to_dims["beta_brand"] == ("channel", "brand")
    assert model.named_vars_to_dims["beta_brand_dma"] == ("channel", "brand", "dma")
    assert model.coords["family"] == (0, 1)
    assert not [
        c for c in model.coords if c.startswith("brand_")
    ]  # no coord per parent
    with model:
        idata = pm.sample_prior_predictive(draws=2, random_seed=1)
    assert all(
        v.ndim == 0 or v.name in model.named_vars_to_dims for v in model.free_RVs
    )
    assert [d for d in idata.prior.dims if "_dim_" in d] == []


def test_two_priors_share_the_parent_and_children_coords():
    # the second finds `family` and `brand_0` already in the model; `family` is not taken for a crossed term
    with pm.Model(coords=COORDS) as model:
        for name in ("beta", "gamma"):
            nzs(NEST | {"dma": {}}).create_variable(name)
    assert {
        v.name.replace("gamma", "beta") for v in model.free_RVs if "gamma" in v.name
    } == {v.name for v in model.free_RVs if "beta" in v.name}


@pytest.mark.parametrize(
    "sigma",
    [
        LEVEL_GAMMA,
        IndependentSigmas({"family": G, "brand": G}),
        VarianceBudget(G, {"family": 0.5, "brand": 0.5}),
    ],
    ids=lambda o: type(o).__name__,
)
def test_every_sigma_option_is_put_on_the_level_dim_of_a_copy(sigma):
    prior = nzs(NEST, sigma=sigma)
    model, _ = build(prior)
    assert model.named_vars_to_dims["beta_sigma"] == ("beta_sigma_level",)
    if sigma is LEVEL_GAMMA:
        assert model["beta_sigma_mu"].ndim == 0  # the hyperprior stays shared
    assert prior.sigma.dims is None and json_round_trip(prior) == prior


def test_a_str_dims_is_one_dim():  # was split into characters, also by from_dict
    prior = nzs({"brand": {}}, mu=Prior("Normal"), dims="brand")
    assert prior.dims == ("brand",)
    assert NestedZeroSumPrior.from_dict(prior.to_dict() | {"dims": "brand"}) == prior


def json_round_trip(prior):
    return NestedZeroSumPrior.from_dict(json.loads(json.dumps(prior.to_dict())))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (
            nzs(NEST),
            nzs(
                {
                    "family": {},
                    "brand": {"within": "family", "map": dict(reversed(FAM.items()))},
                }
            ),
        ),
        (
            nzs(
                NEST,
                sigma=IndependentSigmas(
                    {"family": G, "brand": Prior("HalfNormal", sigma=0.3)}
                ),
            ),
            nzs(
                NEST,
                sigma=IndependentSigmas(
                    {"brand": Prior("HalfNormal", sigma=0.3), "family": G}
                ),
            ),
        ),
        (nzs(NEST, transform="exp"), nzs(NEST)),
        (json_round_trip(p := nzs(NEST, mu=np.array([0.1, -0.2, 0.3]))), p),
        (json_round_trip(p := nzs({}, mu=Prior("Normal"), dims=())), p),
        (json_round_trip(p := nzs(NEST, dims=None)), p),
    ],
    ids=[
        "map_order",
        "sigma_order",
        "explicit_exp_transform",
        "ndarray_mu_round_trip",
        "no_dims_round_trip",
        "unset_dims_round_trip",
    ],
)
def test_equivalent_priors_are_equal_and_hash_equal(a, b):
    assert a == b and hash(a) == hash(b) and a.to_dict() == b.to_dict()


def test_parent_effect_false_builds_nesting_without_the_parent_term():
    terms = {"dma": {"within": "tier", "map": TIER, "parent_effect": False}}
    model, _ = build(nzs(terms))
    assert sorted(model.named_vars) == [
        "beta",
        "beta_dma",
        "beta_dma_raw",
        "beta_dma_raw_offset",
        "beta_mu",
        "beta_sigma",
        "beta_sigma_dma",
    ]  # no beta_tier


# --- Properties of the built prior --------------------------------------------

HN = Prior("HalfNormal", sigma=0.5)
F3 = {
    "b0": "z",
    "b1": "a",
    "b2": "a",
    "b3": "m",
}  # str parents, unsorted, two singletons
CROSSED = {"brand": {}, "tier": {}, "dma": {"within": "tier", "map": TIER}}


def nested(fam=FAM):
    return {
        "family": {},
        "brand": {"within": "family", "map": fam},
        "tier": {},
        "dma": {"within": "tier", "map": TIER},
    }


def build_full(
    terms,
    centered=True,
    sigma=None,
    interaction_dims=("brand", "dma"),
    brands=tuple(FAM),
    dmas=tuple(TIER),
    tier=TIER,
):
    """Beta over (brand, dma, channel) with the given terms plus a brand x dma interaction."""
    terms = {
        k: {**t, "centered": centered} | ({"map": tier} if k == "dma" else {})
        for k, t in terms.items()
    }
    interactions = {"brand_dma": {"dims": list(interaction_dims), "centered": centered}}
    prior = NestedZeroSumPrior(
        mu=MU,
        terms=terms,
        interactions=interactions,
        sigma=sigma or IndependentSigmas({lv: HN for lv in [*terms, "brand_dma"]}),
        dims=hierarchy_dims(terms, interactions),
    )
    coords = {"channel": ["c0", "c1", "c2"], "brand": list(brands), "dma": list(dmas)}
    return build(prior, coords)


def log_beta_fn(model, beta):
    """Log beta as a function of {free RV name: value}."""
    fn = pytensor.function(
        model.free_RVs, pt.log(beta.values), on_unused_input="ignore"
    )
    return lambda values: fn(*(values[rv.name] for rv in model.free_RVs))


def level_rv(level, rv_name):
    """Whether the free RV is one of `level`'s: its offset, its raw Normal, or their standardized forms."""
    return re.fullmatch(rf"beta_{level}(_raw)?(_offset)?", rv_name) is not None


def term_only(model, beta, level, n=20):
    """Level `level`'s contribution to log beta alone, on n prior draws: (draw, brand, dma, channel).

    Mu and every zero-sum variable but the level's are set to 0.
    """
    fn = log_beta_fn(model, beta)
    names = [rv.name for rv in model.free_RVs]
    draws = dict(zip(names, pm.draw(model.free_RVs, n, random_seed=1), strict=True))

    def keep(
        rv_name,
    ):  # beta_brand, beta_brand_raw, beta_brand_raw_offset; not beta_brand_dma
        return "sigma" in rv_name or level_rv(level, rv_name)

    return np.stack(
        [
            fn({k: v[i] if keep(k) else 0 * v[i] for k, v in draws.items()})
            for i in range(n)
        ]
    )


@pytest.mark.parametrize("centered", [True, False])
@pytest.mark.parametrize(
    ("terms", "level", "axis", "groups"),
    [
        (
            nested(),
            "family",
            1,
            [[0, 2]],
        ),  # one brand per family: the families sum to 0
        (nested(), "brand", 1, [[0, 1], [2, 3]]),  # within each family
        (nested(), "tier", 2, [[0, 4, 7]]),
        (nested(), "dma", 2, [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]),  # within each tier
        (nested(), "brand_dma", 1, [range(4)]),  # interaction: both axes
        (nested(), "brand_dma", 2, [range(10)]),
        (CROSSED, "brand", 1, [range(4)]),  # crossed term: over the whole coord
    ],
    ids=[
        "family",
        "brand_in_family",
        "tier",
        "dma_in_tier",
        "interaction_brand_axis",
        "interaction_dma_axis",
        "crossed_brand",
    ],
)
@pytest.mark.parametrize("sigma", [None, LEVEL_HN], ids=["independent", "prior"])
def test_zero_sum_on_every_intended_axis(terms, level, axis, groups, centered, sigma):
    x = term_only(*build_full(terms, centered, sigma), level)
    assert np.abs(x).max() > 1e-3
    for g in groups:
        assert np.abs(x.take(list(g), axis).sum(axis)).max() < 1e-12


@pytest.mark.parametrize("centered", [True, False])
@pytest.mark.parametrize(
    ("brands", "dmas", "fam", "tier"),
    [
        (list(FAM), list(TIER), FAM, TIER),
        (  # permuted coords and map key order
            ["b3", "b0", "b2", "b1"],
            ["d9", "d0", "d5", "d1", "d7", "d2", "d3", "d4", "d6", "d8"],
            dict(reversed(FAM.items())),
            dict(sorted(TIER.items(), key=lambda kv: -int(kv[0][1:]))),
        ),
        (list(FAM), list(TIER), F3, TIER),
    ],
    ids=["base", "permuted", "str_parents_singletons"],
)
def test_each_cell_gets_its_own_parent_and_child_offsets(
    brands, dmas, fam, tier, centered
):
    model, beta = build_full(nested(fam), centered, brands=brands, dmas=dmas, tier=tier)
    rng = np.random.default_rng(0)
    # any values, not only zero-sum ones
    shapes = model.eval_rv_shapes()
    v = {rv.name: rng.normal(size=shapes[rv.name]) for rv in model.free_RVs}

    def scale(level):
        return 1.0 if centered else v[f"beta_sigma_{level}"]

    def rv(level, raw=False):  # the free RV a level's offset is made from
        return (
            f"beta_{level}" + ("_raw" if raw else "") + ("" if centered else "_offset")
        )

    families = list(np.unique(list(fam.values())))
    tiers = list(np.unique(list(tier.values())))
    raw_brand = scale("brand") * v[rv("brand", raw=True)]  # (channel, brand)
    raw_dma = scale("dma") * v[rv("dma", raw=True)]  # (channel, dma)
    expected = np.zeros((len(brands), len(dmas), 3))
    for i, b in enumerate(brands):
        for j, d in enumerate(dmas):
            # siblings, in coord order, itself included: a child is its raw value less its siblings' mean
            sib_b = [k for k, x in enumerate(brands) if fam[x] == fam[b]]
            sib_d = [k for k, x in enumerate(dmas) if tier[x] == tier[d]]
            expected[i, j] = (
                v["beta_mu"]
                + scale("family") * v[rv("family")][:, families.index(fam[b])]
                + scale("tier") * v[rv("tier")][:, tiers.index(tier[d])]
                + scale("brand_dma") * v[rv("brand_dma")][:, i, j]
                + raw_brand[:, i]
                - raw_brand[:, sib_b].mean(axis=1)
                + raw_dma[:, j]
                - raw_dma[:, sib_d].mean(axis=1)
            )
    assert np.abs(log_beta_fn(model, beta)(v) - expected).max() < 1e-12


@pytest.mark.parametrize("centered", [True, False])
def test_singleton_children_get_no_within_offset_but_the_parent_effect(centered):
    model, beta = build_full(nested(F3), centered)
    brand, family = term_only(model, beta, "brand"), term_only(model, beta, "family")
    assert np.all(brand[:, [0, 3]] == 0)  # b0 and b3 are alone in their family
    assert np.abs(brand[:, [1, 2]]).min() > 0
    assert np.abs(family[:, [0, 3]]).min() > 0


def test_variance_budget_sigma_is_total_times_sqrt_share():
    share = {"family": 0.3, "brand": 0.3, "tier": 0.2, "dma": 0.1, "brand_dma": 0.1}
    model, _ = build_full(nested(), sigma=VarianceBudget(HN, share))
    sigma, total, shares = pm.draw(
        [model["beta_sigma"], model["beta_sigma_total"], model["beta_sigma_share"]],
        200,
        random_seed=0,
    )
    np.testing.assert_allclose(sigma, total[:, None] * np.sqrt(shares), rtol=1e-12)


@pytest.mark.parametrize("centered", [True, False])
def test_per_cell_variance_around_mu_matches_the_docstring(centered):
    # Var(log beta - mu) per cell = E[total^2] * sum_l c_l share_l, c_l = 1 - 1/J per zero-sum axis
    share = {
        "family": 0.49261084,
        "brand": 0.49261084,
        "tier": 0.0049261084,
        "dma": 0.0049261084,
        "brand_dma": 0.0049261084,
    }
    total = Prior("Gamma", mu=0.43, sigma=0.11)
    model, beta = build_full(nested(), centered, VarianceBudget(total, share))
    log_beta, mu, t = pm.draw(
        [pt.log(beta.values), model["beta_mu"], model["beta_sigma_total"]],
        4000,
        random_seed=2,
    )
    c_dma = np.array([3 / 4] * 4 + [2 / 3] * 6)  # tiers of 4, 3 and 3 DMAs
    predicted = (t**2).mean() * (
        share["family"] / 2
        + share["brand"] / 2
        + share["tier"] * 2 / 3
        + share["dma"] * c_dma
        + share["brand_dma"] * 3 / 4 * 9 / 10
    )
    ratio = (log_beta - mu[:, None, None, :]).var(0) / predicted[None, :, None]
    assert abs(ratio.mean() - 1) < 0.05
    assert 0.85 < ratio.min() and ratio.max() < 1.15


@pytest.mark.parametrize("centered", [True, False])
def test_reordered_spec_gives_identical_beta_and_logp(centered):
    reordered = {k: nested()[k] for k in ("tier", "dma", "family", "brand")}
    model_a, beta_a = build_full(nested(), centered)
    model_b, beta_b = build_full(reordered, centered, interaction_dims=("dma", "brand"))
    assert beta_a.dims == ("brand", "dma", "channel")
    assert beta_b.dims == ("dma", "brand", "channel")
    rng = np.random.default_rng(1)
    point_a = {
        k: x + 0.5 * rng.normal(size=np.shape(x))
        for k, x in sorted(model_a.initial_point().items())
    }
    # the interaction's axes are swapped
    point_b = point_a | {
        k: np.swapaxes(point_a[k], 1, 2)
        for k in point_a
        if k.startswith("beta_brand_dma")
    }
    np.testing.assert_allclose(
        beta_at(model_a, beta_a, point_a),
        np.swapaxes(beta_at(model_b, beta_b, point_b), 0, 1),
        rtol=1e-13,
        atol=0,
    )
    np.testing.assert_allclose(
        model_a.compile_logp()(point_a),
        model_b.compile_logp()(point_b),
        rtol=1e-12,
        atol=0,
    )


@pytest.mark.parametrize("prior", [LEVEL_HN, LEVEL_GAMMA], ids=["flat", "hierarchical"])
def test_single_variable_sigma_entry_i_scales_level_i(prior):
    # Non-centered, sigma = e_i leaves only level i in log beta, and zeroing level i's offsets removes it
    model, beta = build_full(nested(), centered=False, sigma=prior)
    levels = ["brand", "brand_dma", "dma", "family", "tier"]  # sorted
    assert model.named_vars_to_dims["beta_sigma"] == ("beta_sigma_level",)
    assert list(model.coords["beta_sigma_level"]) == levels
    fn = log_beta_fn(model, beta)
    names = [rv.name for rv in model.free_RVs]
    draws = dict(zip(names, pm.draw(model.free_RVs, random_seed=1), strict=True))
    draws["beta_mu"] = 0 * draws["beta_mu"]
    for i, lv in enumerate(levels):
        draws["beta_sigma"] = np.eye(len(levels))[i]
        assert np.abs(fn(draws)).max() > 1e-3
        own_zeroed = {k: 0 * v if level_rv(lv, k) else v for k, v in draws.items()}
        assert np.abs(fn(own_zeroed)).max() == 0


@pytest.mark.parametrize(
    "coords",
    [COORDS, COORDS | {"beta_sigma_level": ["brand", "family"]}],
    ids=["coord_added_here", "coord_already_there"],
)
@pytest.mark.parametrize("centered", [True, False])
def test_xtensor_sigma_is_unwrapped(coords, centered):
    model, beta = build(nzs(NEST, centered=centered, sigma=XScales()), coords)
    assert list(model.coords["beta_sigma_level"]) == ["brand", "family"]  # sorted
    assert model.named_vars_to_dims["beta_sigma"] == ("beta_sigma_level",)
    assert np.isfinite(model.compile_logp()(model.initial_point()))
    assert np.all(pm.draw(beta, 10, random_seed=1) > 0)


# --- Serialization ------------------------------------------------------------

IND = IndependentSigmas({"family": G, "brand": Prior("HalfNormal", sigma=0.3)})
BUD = VarianceBudget(
    total=Prior("TruncatedNormal", mu=0.4, sigma=0.1, lower=0.0),
    share={"family": 0.3, "brand": 0.7},
    concentration=100,
)
SP = NestedZeroSumPrior(
    mu=Prior("Normal", mu=[0.0, 1.0, 2.0], sigma=0.3, dims="channel"),
    sigma=BUD,
    terms={
        "family": {"centered": True},
        "brand": {"within": "family", "map": {"b0": 0, "b1": 1}},
    },
)
SP_PRIOR = NestedZeroSumPrior(
    mu=MU, terms=NEST, sigma=LEVEL_GAMMA
)  # sigma goes through Prior.to_dict and deserialize
OBJS = [IND, BUD, SP_PRIOR, VarianceBudget(G, {"brand": 1.0}, dims="s_level"), SP]


@pytest.mark.parametrize("obj", OBJS, ids=lambda o: type(o).__name__)
class TestRoundTrip:
    def test_to_dict_from_dict(self, obj):
        data = obj.to_dict()
        assert type(obj).from_dict(data).to_dict() == data

    def test_pymc_marketing_registry(self, obj):  # what fit() and save() use
        data = serialization.serialize(obj)
        assert data["__type__"] == f"pymc_marketing.special_priors.{type(obj).__name__}"
        loaded = serialization.deserialize(data)
        assert type(loaded) is type(obj)
        assert loaded.to_dict() == obj.to_dict()

    def test_pymc_extras_deserialize(
        self, obj
    ):  # what MMM.load uses for the saturation
        loaded = deserialize(serialization.serialize(obj))
        assert type(loaded) is type(obj)
        assert loaded.to_dict() == obj.to_dict()


def test_the_deserializer_can_distinguish_between_the_classes():
    loaded = [deserialize(serialization.serialize(o)) for o in OBJS]
    assert [type(o) for o in loaded] == [type(o) for o in OBJS]
    assert loaded[-1] == SP
    assert loaded[2] == SP_PRIOR
    assert (
        SP.to_dict()["__type__"] == "pymc_marketing.special_priors.NestedZeroSumPrior"
    )
    assert isinstance(deserialize(G.to_dict()), Prior)  # plain Priors still go to Prior


@pytest.mark.parametrize("dumps", [json.dumps, yaml.safe_dump])
@pytest.mark.parametrize("transform", [None, "sigmoid"])
def test_transform_and_base_dims_round_trip(dumps, transform):
    prior = nzs(
        NEST,
        mu=Prior("Normal", dims="control"),
        dims=("brand", "control"),
        transform=transform,
    )
    loaded = NestedZeroSumPrior.from_dict(yaml.safe_load(dumps(prior.to_dict())))
    assert loaded == prior and loaded.transform == transform
    assert deserialize(serialization.serialize(prior)) == prior


def test_nested_sigma_survives_round_trip():
    loaded = deserialize(serialization.serialize(SP))
    assert isinstance(loaded.sigma, VarianceBudget)
    assert loaded.sigma.total == BUD.total
    assert loaded.mu == SP.mu


@pytest.mark.parametrize(
    "obj", [*OBJS, LEVEL_HN, nzs(NEST)], ids=lambda o: type(o).__name__
)
def test_follows_the_pymc_extras_variable_factory_protocol(obj):
    assert isinstance(obj, VariableFactory)


@pytest.mark.parametrize("xdist", [False, True])  # xdist was a TypeError
@pytest.mark.parametrize("sigma", [IND, BUD], ids=lambda o: type(o).__name__)
def test_sigma_options_sample_alone_as_one_vector_over_the_level_dim(sigma, xdist):
    # labels are matched by label, not position: the coord order differs from the spec order
    sigma = copy.deepcopy(sigma)
    sigma.dims = ("s_level",)
    draws = sample_prior(
        sigma, coords={"s_level": ["brand", "family"]}, name="s", draws=5, xdist=xdist
    )
    assert draws["s"].dims == ("chain", "draw", "s_level")
    assert list(draws["s"].coords["s_level"].values) == ["brand", "family"]


@pytest.mark.parametrize("xdist", [False, True])
@pytest.mark.parametrize("sigma", [IND, BUD], ids=lambda o: type(o).__name__)
def test_sigma_options_are_a_prior_parameter_on_their_dims(sigma, xdist):
    # dims was None, read as a scalar: "attempt to get argmax of an empty sequence"
    sigma = type(sigma).from_dict(sigma.to_dict() | {"dims": "x_sigma_level"})
    assert sigma.dims == ("x_sigma_level",)
    draws = sample_prior(
        Prior("Normal", mu=0, sigma=sigma, dims="x_sigma_level"),
        coords={"x_sigma_level": ["brand", "family"]},
        name="x",
        draws=5,
        xdist=xdist,
    )
    assert (
        draws["x"].dims == draws["x_sigma"].dims == ("chain", "draw", "x_sigma_level")
    )


@pytest.mark.parametrize("sigma", [IND, BUD], ids=lambda o: type(o).__name__)
def test_sigma_options_without_dims_raise_at_build(sigma):
    with pytest.raises(ValueError, match=r"s: `dims` must be one level dim, got None"):
        sample_prior(sigma, coords={"s_level": ["brand", "family"]}, name="s")


# --- MMM ----------------------------------------------------------------------

MMM_FAM = {"b0": "f0", "b1": "f0", "b2": "f0", "b3": "f1"}


@pytest.fixture(scope="module")
def panel():
    """A small (date, brand, dma) panel with two channels and one control."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=12, freq="W-MON")
    index = pd.MultiIndex.from_product(
        [dates, list(MMM_FAM), ["d0", "d1"]], names=["date", "brand", "dma"]
    )
    X = index.to_frame(index=False)
    X["c0"] = rng.uniform(0, 10, len(X))
    X["c1"] = rng.uniform(0, 5, len(X))
    X["x0"] = rng.normal(size=len(X))
    y = pd.Series(
        1 + 0.1 * X["c0"] + 0.2 * X["c1"] + rng.normal(0, 0.1, len(X)), name="y"
    )
    return X, y


def make_mmm(sigma, dims=("brand", "dma"), terms=None, beta_dims=None):
    beta = NestedZeroSumPrior(
        mu=Prior("Normal", mu=0, sigma=0.5, dims="channel"),
        terms=terms
        or {"family": {}, "brand": {"within": "family", "map": MMM_FAM}, "dma": {}},
        interactions=None if terms else {"brand_dma": {"dims": ["brand", "dma"]}},
        sigma=sigma,
        dims=beta_dims,
    )
    return MMM(
        date_column="date",
        channel_columns=["c0", "c1"],
        control_columns=["x0"],
        target_column="y",
        dims=dims,
        adstock=GeometricAdstock(
            l_max=2, priors={"alpha": Prior("Beta", alpha=2, beta=3, dims="channel")}
        ),
        saturation=LogisticSaturation(
            priors={
                "beta": beta,
                "lam": Prior("Gamma", alpha=3, beta=2, dims="channel"),
            }
        ),
    )


LEVELS = ["family", "brand", "dma", "brand_dma"]
SIGMAS = {
    "independent": IndependentSigmas(
        {lv: Prior("HalfNormal", sigma=0.2) for lv in LEVELS}
    ),
    "budget": VarianceBudget(
        Prior("HalfNormal", sigma=0.4), dict.fromkeys(LEVELS, 0.25)
    ),
    "prior": Prior("HalfNormal", sigma=Prior("HalfNormal", sigma=0.2)),
}


@pytest.mark.parametrize("dims", [("brand", "dma"), ("dma", "brand")])
def test_mmm_builds_with_the_prior_on_saturation_beta(panel, dims):
    # the MMM sets the prior's dims, in its own order, which need not be the terms order
    X, y = panel
    mmm = make_mmm(SIGMAS["independent"], dims)
    mmm.build_model(X, y)
    beta = mmm.model["saturation_beta"]
    assert mmm.model.named_vars_to_dims["saturation_beta"] == (*dims, "channel")
    assert mmm.model.named_vars_to_dims["saturation_beta_sigma"] == (
        "saturation_beta_sigma_level",
    )
    # one offset per term, whatever the number of parents, and no parent label in a name
    assert sorted(
        v for v in mmm.model.named_vars if v.startswith("saturation_beta")
    ) == [
        "saturation_beta",
        "saturation_beta_brand",
        "saturation_beta_brand_dma",
        "saturation_beta_brand_dma_offset",
        "saturation_beta_brand_raw",
        "saturation_beta_brand_raw_offset",
        "saturation_beta_dma",
        "saturation_beta_dma_offset",
        "saturation_beta_family",
        "saturation_beta_family_offset",
        "saturation_beta_mu",
        "saturation_beta_sigma",
        "saturation_beta_sigma_brand",
        "saturation_beta_sigma_brand_dma",
        "saturation_beta_sigma_dma",
        "saturation_beta_sigma_family",
    ]
    assert np.all(pm.draw(beta, 10, random_seed=1) > 0)


def test_mmm_with_a_hierarchy_over_some_of_its_dims_needs_explicit_dims(panel):
    terms = {"brand": {}}
    with pytest.raises(
        ValueError, match=r"base dims \['dma', 'channel'\].*pass `dims`"
    ):
        make_mmm(IndependentSigmas({"brand": G}), terms=terms)
    mmm = make_mmm(
        IndependentSigmas({"brand": G}), terms=terms, beta_dims=("brand", "channel")
    )
    mmm.build_model(*panel)  # beta broadcasts over dma
    assert mmm.model.named_vars_to_dims["saturation_beta"] == ("brand", "channel")


@pytest.mark.parametrize("sigma", ["independent", "budget", "prior"])
def test_mmm_save_load_round_trip(sigma, panel, tmp_path, mock_pymc_sample):
    X, y = panel
    mmm = make_mmm(SIGMAS[sigma])
    mmm.fit(X, y, chains=1, draws=10, tune=10, random_seed=42)
    assert "saturation_beta" in mmm.idata.posterior
    mmm.save(str(tmp_path / "mmm.nc"))
    loaded = MMM.load(str(tmp_path / "mmm.nc"), check=True)
    original_beta = mmm.saturation.function_priors["beta"]
    loaded_beta = loaded.saturation.function_priors["beta"]
    assert loaded.id == mmm.id
    assert type(loaded_beta) is NestedZeroSumPrior
    assert loaded_beta == original_beta
    assert type(loaded_beta.sigma) is type(SIGMAS[sigma])
    assert loaded_beta.dims == ("brand", "dma", "channel")  # as the MMM set them
    point = mmm.model.initial_point()
    assert loaded.model.compile_logp()(point) == mmm.model.compile_logp()(point)


@pytest.mark.parametrize("sigma", ["independent", "budget", "prior"])
def test_mmm_budget_optimizer_with_the_prior_on_saturation_beta(
    sigma, panel, mock_pymc_sample
):
    # the optimizer replaces every free RV by an xtensor constant, so all of the prior's RVs must be xtensor RVs
    X, y = panel
    mmm = make_mmm(SIGMAS[sigma])
    mmm.fit(X, y, chains=1, draws=10, tune=10, random_seed=42)
    last = X["date"].max()
    optimizer = mmm.budget_optimizer(
        start_date=last + pd.Timedelta(weeks=1),
        end_date=last + pd.Timedelta(weeks=4),
    )
    cells = optimizer.budgets_to_optimize.astype(float)
    bounds = xr.concat(
        [xr.zeros_like(cells), xr.full_like(cells, 100.0)], dim="bound"
    ).transpose(*cells.dims, "bound")
    optimal_budgets, result = optimizer.allocate_budget(
        total_budget=100, budget_bounds=bounds
    )
    assert optimal_budgets.sum().item() == pytest.approx(100)
    assert np.isfinite(result.fun)


def docstring_blocks(cls, language):
    """The ``.. code-block:: <language>`` blocks of a class docstring, dedented."""
    pattern = rf"^( *)\.\. code-block:: {language}\n\n((?:\s*\n|\1 +.*\n)+)"
    return [textwrap.dedent(b) for _, b in re.findall(pattern, cls.__doc__, re.M)]


@pytest.mark.parametrize(
    ("cls", "dims"),
    [
        (NestedZeroSumPrior, ["brand", "dma"]),
        (IndependentSigmas, ["brand"]),  # the examples' MMM dims
        (VarianceBudget, ["brand"]),
    ],
)
def test_docstring_examples_build(cls, dims, panel, tmp_path):
    namespace = {}
    for python_example in docstring_blocks(
        cls, "python"
    ):  # later blocks reuse the first's imports
        exec(python_example, namespace)  # noqa: S102
    assert isinstance(namespace["mmm"], MMM)

    (yaml_example,) = docstring_blocks(cls, "yaml")
    config = {
        "model": {
            "class": "pymc_marketing.mmm.mmm.MMM",
            "kwargs": {
                "date_column": "date",
                "channel_columns": ["c0", "c1"],
                "target_column": "y",
                "dims": dims,
                "adstock": {
                    "class": "pymc_marketing.mmm.GeometricAdstock",
                    "kwargs": {"l_max": 2},
                },
                **yaml.safe_load(yaml_example),
            },
        }
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    X, y = panel
    keep = X["dma"] == "d0" if dims == ["brand"] else slice(None)
    X, y = X[keep].drop(columns=[] if "dma" in dims else ["dma"]), y[keep]
    mmm = build_mmm_from_yaml(path, X=X, y=y)
    beta = mmm.saturation.function_priors["beta"]
    assert isinstance(beta, NestedZeroSumPrior)
    assert isinstance(beta.sigma, IndependentSigmas | VarianceBudget)
    assert "saturation_beta" in mmm.model.named_vars
