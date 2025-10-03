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

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pymc_extras.prior import Prior
from pytensor import function

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.special_priors import (
    LogNormalPrior,
    MaskedPrior,
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
        rv.create_variable("test")

    assert rv.to_dict() == rv.from_dict(rv.to_dict()).to_dict()


def test_LogNormalPrior_args_invalid():
    with pytest.raises(ValueError):
        LogNormalPrior(alpha=1.0, beta=1.0)


def test_the_deserializer_can_distinguish_between_types_of_prior_classes():
    assert _is_LogNormalPrior_type(LogNormalPrior(mu=1.0, sigma=1.0).to_dict())
    assert not _is_LogNormalPrior_type(Prior("Normal", mu=1.0, sigma=1.0).to_dict())


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
