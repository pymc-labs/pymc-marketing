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
"""Tests for the VAR baseline effect and its optional Impulso dependency."""

import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pydantic import ValidationError
from pymc.testing import mock_sample

from pymc_marketing.mmm import (
    MMM,
    GeometricAdstock,
    LogisticSaturation,
    VARBaselineEffect,
)

pytest.importorskip("impulso")

from impulso import VAR, MinnesotaPrior, ar1_residual_sd

seed: int = sum(map(ord, "VARBaselineEffect"))
PREFIX = "brand_var"
ENDOG_NAMES = ["baseline", "awareness", "consideration"]


def make_brand_mmm_data(n_weeks: int = 30) -> dict:
    """Weekly sales whose baseline moves with consideration, and the brand data.

    Awareness responds to brand spend, consideration to awareness, and the sales
    baseline to consideration, each as an AR(1). The true baseline is returned too.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")
    brand_spend = np.zeros(n_weeks)
    brand_spend[5:10] = brand_spend[18:23] = 1.0
    awareness, consideration, baseline = np.zeros((3, n_weeks))
    for t in range(1, n_weeks):
        awareness[t] = (
            0.9 * awareness[t - 1] + 0.25 * brand_spend[t] + rng.normal(0, 0.04)
        )
        consideration[t] = (
            0.8 * consideration[t - 1] + 0.4 * awareness[t - 1] + rng.normal(0, 0.03)
        )
        baseline[t] = (
            0.5 * baseline[t - 1] + 0.6 * consideration[t - 1] + rng.normal(0, 0.04)
        )

    X = pd.DataFrame(
        {
            "date": dates,
            "x1": rng.uniform(0, 1, n_weeks),
            "x2": rng.uniform(0, 1, n_weeks),
        }
    )
    y = pd.Series(
        3.0 + baseline + 0.5 * X["x1"] + 0.3 * X["x2"] + rng.normal(0, 0.1, n_weeks),
        name="sales",
    )
    brand_data = pd.DataFrame(
        {
            "date": dates,
            "awareness": 1.0 + awareness,
            "consideration": 0.5 + consideration,
            "brand_spend": brand_spend,
        }
    )
    return {"X": X, "y": y, "brand_data": brand_data, "baseline": baseline}


@pytest.fixture(scope="module")
def brand_mmm_data() -> dict:
    return make_brand_mmm_data()


def make_effect(brand_data: pd.DataFrame, **kwargs) -> VARBaselineEffect:
    params = {
        "brand_data": brand_data,
        "baseline_name": "baseline",
        "endog_names": ENDOG_NAMES,
        "exog_names": ["brand_spend"],
        "prefix": PREFIX,
        "var": VAR(lags=1),
    }
    return VARBaselineEffect(**(params | kwargs))


def make_mmm(effect) -> MMM:
    return MMM(
        date_column="date",
        channel_columns=["x1", "x2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    ).add_mu_effect(effect)


@pytest.fixture(scope="module")
def fitted_mmm(brand_mmm_data) -> MMM:
    mmm = make_mmm(make_effect(brand_mmm_data["brand_data"]))
    # Patched for this fit only: the module-scoped `mock_pymc_sample` would stay
    # active for the rest of the module and turn the slow NUTS test into prior
    # draws.
    with patch.object(pm, "sample", mock_sample):
        mmm.fit(brand_mmm_data["X"], brand_mmm_data["y"], draws=20, random_seed=seed)
    return mmm


class OwnLagPrior:
    """A lag-coefficient prior that is not a ``MinnesotaPrior``.

    Every own first lag has prior mean 0.9.
    """

    def build_priors(
        self, n_vars: int, n_lags: int, *, sigma: np.ndarray
    ) -> dict[str, np.ndarray]:
        B_mu = np.zeros((n_vars, n_vars * n_lags))
        B_mu[:, :n_vars] = 0.9 * np.eye(n_vars)
        return {"B_mu": B_mu, "B_sigma": np.full_like(B_mu, 0.1)}


def test_constructing_the_effect_without_impulso_raises(brand_mmm_data, monkeypatch):
    spec = VAR(lags=1)
    monkeypatch.setitem(sys.modules, "impulso", None)

    with pytest.raises(ImportError, match=r"pip install 'pymc-marketing\[var\]'"):
        make_effect(brand_mmm_data["brand_data"], var=spec)


def test_var_must_be_an_impulso_var_spec(brand_mmm_data):
    with pytest.raises(TypeError, match=r"impulso\.VAR"):
        make_effect(brand_mmm_data["brand_data"], var=MinnesotaPrior())


@pytest.mark.parametrize("baseline_own_lag_mean", [np.nan, 1.0, -1.0])
def test_baseline_own_lag_mean_must_keep_the_baseline_stationary(
    brand_mmm_data, baseline_own_lag_mean
):
    with pytest.raises(ValidationError, match="baseline_own_lag_mean"):
        make_effect(
            brand_mmm_data["brand_data"], baseline_own_lag_mean=baseline_own_lag_mean
        )


def test_fit_adds_the_var_and_the_baseline_to_the_posterior(fitted_mmm):
    posterior = fitted_mmm.idata.posterior
    var_names = [
        "B",
        "B_exog",
        "intercept",
        "L",
        "Sigma",
        "tril_offdiag",
        "latent_init",
        "latent_innovations",
        "latent",
    ]

    assert {f"{PREFIX}::{name}" for name in var_names} <= set(posterior.data_vars)
    contribution = posterior[f"{PREFIX}_effect_contribution"]
    assert contribution.dims == ("chain", "draw", "date")


def test_baseline_equation_has_no_intercept(fitted_mmm):
    intercept = fitted_mmm.idata.posterior[f"{PREFIX}::intercept"]

    assert intercept.coords["var_intercept"].values.tolist() == ENDOG_NAMES[1:]


def test_baseline_appears_in_the_contribution_decomposition(fitted_mmm):
    """The decomposition shows the baseline, in sales units, under the effect's name."""
    contributions = fitted_mmm.compute_mean_contributions_over_time()
    baseline = fitted_mmm.idata.posterior[f"{PREFIX}::latent"].mean(("chain", "draw"))

    assert f"{PREFIX}_effect" in contributions.columns
    np.testing.assert_allclose(contributions[f"{PREFIX}_effect"], baseline.values[:, 0])


def test_in_sample_posterior_predictive(fitted_mmm, brand_mmm_data):
    """The default ``clone_model=True`` clones the MMM's model, VAR included."""
    X = brand_mmm_data["X"]

    draws = fitted_mmm.sample_posterior_predictive(
        X, extend_idata=False, progressbar=False, random_seed=seed
    )

    assert draws.indexes["date"].equals(pd.DatetimeIndex(X["date"]))
    assert draws[fitted_mmm.output_var].dims == ("date", "sample")
    assert np.isfinite(draws[fitted_mmm.output_var]).all()


@pytest.mark.parametrize(
    "prior, effect_kwargs, expected",
    [
        pytest.param("minnesota", {}, [0.0, 1.0, 1.0], id="shorthand"),
        pytest.param(
            MinnesotaPrior(own_lag_mean=0.5),
            {"baseline_own_lag_mean": 0.3},
            [0.3, 0.5, 0.5],
            id="scalar",
        ),
        pytest.param(
            MinnesotaPrior(own_lag_mean=(0.3, 0.9, 0.8)),
            {"baseline_own_lag_mean": 0.3},
            [0.3, 0.9, 0.8],
            id="per-variable",
        ),
        pytest.param(
            OwnLagPrior(),
            {"baseline_own_lag_mean": 0.3},
            [0.9, 0.9, 0.9],
            id="not-minnesota",
        ),
    ],
)
def test_own_lag_prior_means(brand_mmm_data, prior, effect_kwargs, expected):
    """The effect sets the baseline's own-lag mean and keeps the analyst's spec."""
    spec = VAR(lags=1, prior=prior)
    effect = make_effect(brand_mmm_data["brand_data"], var=spec, **effect_kwargs)
    mmm = make_mmm(effect)
    mmm.build_model(brand_mmm_data["X"], brand_mmm_data["y"])

    B = pm.sample_prior_predictive(
        draws=2000,
        var_names=[f"{PREFIX}::B"],
        model=mmm.model,
        random_seed=seed,
    ).prior[f"{PREFIX}::B"]
    own_lag_means = [
        float(B.sel(var=name, coeff=f"L1.{name}").mean()) for name in ENDOG_NAMES
    ]

    np.testing.assert_allclose(own_lag_means, expected, atol=0.02)
    assert effect.var is spec


@pytest.mark.parametrize(
    "own_lag_mean, baseline_own_lag_mean",
    [
        pytest.param((0.5, 0.9, 0.8), 0.0, id="default-baseline-mean"),
        pytest.param((0.0, 0.9, 0.8), 0.3, id="set-baseline-mean"),
    ],
)
def test_own_lag_mean_for_the_baseline_raises_at_construction(
    brand_mmm_data, own_lag_mean, baseline_own_lag_mean
):
    """The effect sets the baseline's entry, so the analyst's must not differ."""
    spec = VAR(lags=1, prior=MinnesotaPrior(own_lag_mean=own_lag_mean))

    with pytest.raises(
        ValueError,
        match=rf"own_lag_mean \({own_lag_mean[0]}, 0\.9, 0\.8\).*sets the "
        rf"baseline's entry.*baseline_own_lag_mean, which is {baseline_own_lag_mean}",
    ):
        make_effect(
            brand_mmm_data["brand_data"],
            var=spec,
            baseline_own_lag_mean=baseline_own_lag_mean,
        )


def sample_path_start(data: dict, effect) -> np.ndarray:
    """Prior draws of the baseline's first two values, for an effect with two lags."""
    mmm = make_mmm(effect)
    mmm.build_model(data["X"], data["y"])
    latent = pm.sample_prior_predictive(
        draws=10_000,
        var_names=[f"{PREFIX}::latent"],
        model=mmm.model,
        random_seed=seed,
    ).prior[f"{PREFIX}::latent"]
    return latent.to_numpy()[:, :, :2, 0]


@pytest.mark.parametrize(
    "effect_kwargs",
    [{}, {"baseline_own_lag_mean": 0.6}],
    ids=["default", "own-lag-mean"],
)
def test_baseline_starts_at_the_targets_sd(brand_mmm_data, effect_kwargs):
    """The baseline's first ``n_lags`` values have the target's sd.

    That is the sd in the target's original units, with ``ddof=0``, whatever
    ``baseline_own_lag_mean`` is.
    """
    effect = make_effect(brand_mmm_data["brand_data"], var=VAR(lags=2), **effect_kwargs)
    start = sample_path_start(brand_mmm_data, effect)

    np.testing.assert_allclose(
        start.std(axis=(0, 1)), brand_mmm_data["y"].std(ddof=0), rtol=0.05
    )


@pytest.fixture(scope="module")
def prior_draws(brand_mmm_data) -> xr.DataTree:
    """Prior draws of ``B_exog`` from an MMM whose effect has two lags."""
    effect = make_effect(brand_mmm_data["brand_data"], var=VAR(lags=2))
    mmm = make_mmm(effect)
    mmm.build_model(brand_mmm_data["X"], brand_mmm_data["y"])
    return pm.sample_prior_predictive(
        draws=4000,
        var_names=[f"{PREFIX}::B_exog"],
        model=mmm.model,
        random_seed=seed,
    ).prior


def test_exog_prior_scales_with_each_series(brand_mmm_data, prior_draws):
    """Each equation's exog prior sd is proportional to its series' scale.

    The baseline's scale is the AR(1) residual sd of the target, and each brand
    metric's is its own, so the ratios pin the scales the VAR is built with.
    """
    brand_data = brand_mmm_data["brand_data"]
    scales = ar1_residual_sd(
        np.column_stack([brand_mmm_data["y"], brand_data[ENDOG_NAMES[1:]]])
    )
    B_exog = prior_draws[f"{PREFIX}::B_exog"].sel(exog="brand_spend")
    sd = B_exog.std(("chain", "draw"))

    np.testing.assert_allclose(
        sd.sel(var="baseline") / sd.sel(var=ENDOG_NAMES[1:]),
        scales[0] / scales[1:],
        rtol=0.1,
    )


def test_brand_data_needs_one_row_per_mmm_date(brand_mmm_data):
    effect = make_effect(brand_mmm_data["brand_data"].iloc[:-1])
    mmm = make_mmm(effect)

    with pytest.raises(ValueError, match=r"29 rows of brand_data for 30 MMM dates"):
        mmm.build_model(brand_mmm_data["X"], brand_mmm_data["y"])


def test_building_without_a_target_raises(brand_mmm_data):
    """Without ``y`` the MMM builds on an all-zero target, which cannot scale it."""
    mmm = make_mmm(make_effect(brand_mmm_data["brand_data"]))

    with pytest.raises(ValueError, match=rf"{PREFIX}.*target"):
        mmm.sample_prior_predictive(brand_mmm_data["X"])


@pytest.mark.slow
def test_pymc_nuts_moves_the_baseline(brand_mmm_data):
    """PyMC's own NUTS sampler runs on the effect's gradient.

    Only this sampler compiles the gradient with the configured PyTensor linker;
    nutpie compiles its own, and its fit is checked for recovery below.
    """
    mmm = make_mmm(make_effect(brand_mmm_data["brand_data"]))
    mmm.fit(
        brand_mmm_data["X"],
        brand_mmm_data["y"],
        draws=20,
        tune=50,
        chains=1,
        nuts_sampler="pymc",
        random_seed=seed,
        progressbar=False,
    )

    latent = mmm.idata.posterior[f"{PREFIX}::latent"]
    assert (latent.std("draw") > 0).all()


@pytest.fixture(scope="module")
def long_brand_mmm_data() -> dict:
    return make_brand_mmm_data(n_weeks=150)


@pytest.fixture(scope="module")
def nuts_fitted_mmm(long_brand_mmm_data) -> MMM:
    """An MMM with the effect, fitted by nutpie to enough weeks to recover the truth."""
    mmm = make_mmm(make_effect(long_brand_mmm_data["brand_data"]))
    mmm.fit(
        long_brand_mmm_data["X"],
        long_brand_mmm_data["y"],
        draws=500,
        tune=500,
        chains=4,
        cores=4,
        nuts_sampler="nutpie",
        random_seed=seed,
        progressbar=False,
    )
    return mmm


@pytest.mark.slow
def test_nuts_recovers_the_baseline(nuts_fitted_mmm, long_brand_mmm_data):
    """The posterior mean baseline tracks the true one, with few divergences.

    The correlation and the error ignore the baseline's level, which the MMM
    intercept carries. The true path starts well below its mean, since the brand
    flights lift it later, so a start-of-path prior that holds the start near zero
    shows as a larger error over the first weeks than over the rest.
    """
    latent = nuts_fitted_mmm.idata.posterior[f"{PREFIX}::latent"]
    baseline = latent.mean(("chain", "draw")).to_numpy()[:, 0]
    truth = long_brand_mmm_data["baseline"]

    assert int(nuts_fitted_mmm.idata.sample_stats["diverging"].sum()) <= 5
    correlation = np.corrcoef(baseline, truth)[0, 1]
    assert correlation > 0.95
    error = np.abs((baseline - baseline.mean()) - (truth - truth.mean()))
    assert error[:4].mean() < 2 * error[26:].mean()
