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

import json
import sys
from functools import partial
from pathlib import Path
from unittest.mock import patch

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pydantic import BaseModel, ValidationError
from pymc.testing import mock_sample
from scipy import stats

from pymc_marketing.mmm import (
    MMM,
    GeometricAdstock,
    LogisticSaturation,
    SoftPlusHSGP,
    TimeSliceCrossValidator,
    VARBaselineEffect,
)
from pymc_marketing.mmm.scaling import FixedScaling
from pymc_marketing.serialization import DeserializationContext, serialization

pytest.importorskip("impulso")

from impulso import (
    VAR,
    Constant,
    Gaussian,
    InnovationScalePrior,
    MinnesotaPrior,
    NUTSSampler,
    SVDefaultPrior,
    VARData,
    ar1_residual_sd,
)

seed: int = sum(map(ord, "VARBaselineEffect"))
PREFIX = "brand_var"
ENDOG_NAMES = ["baseline", "awareness", "consideration"]
QUARTILES = [0.25, 0.5, 0.75]
FIXED_TARGET_SCALING = {"target": FixedScaling(dims=(), value=6.0)}


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


def make_mmm(effect, date_column: str = "date", **kwargs) -> MMM:
    return MMM(
        date_column=date_column,
        channel_columns=["x1", "x2"],
        target_column="sales",
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        **kwargs,
    ).add_mu_effect(effect)


def fit_mmm(data: dict, effect: VARBaselineEffect) -> MMM:
    """An MMM with ``effect``, fitted to ``data`` with prior draws in place of NUTS.

    The fit has a ``diverging`` sampler statistic, as a NUTS fit would.
    """
    mmm = make_mmm(effect)
    # Patched for this fit only: the module-scoped `mock_pymc_sample` would stay
    # active for the rest of the module and turn the slow NUTS test into prior
    # draws.
    sample = partial(mock_sample, sample_stats={"diverging": np.zeros})
    with patch.object(pm, "sample", sample):
        mmm.fit(data["X"], data["y"], draws=20, random_seed=seed)
    return mmm


@pytest.fixture(scope="module")
def fitted_mmm(brand_mmm_data) -> MMM:
    return fit_mmm(brand_mmm_data, make_effect(brand_mmm_data["brand_data"]))


class PlainPrior:
    """A lag-coefficient prior from outside Impulso that pydantic cannot serialize.

    Constructing the effect refuses it, so ``build_priors`` is never called.
    """

    def build_priors(self, *args, **kwargs):
        raise NotImplementedError


class PydanticPrior(PlainPrior, BaseModel):
    """``PlainPrior`` as a pydantic model, which serializes to an empty dict.

    Impulso loads the empty dict back as a default ``MinnesotaPrior``.
    """


class PlainErrors:
    """Observation errors from outside Impulso that pydantic cannot serialize.

    Constructing the effect refuses them, so no method is ever called.
    """

    name = "plain"
    is_heavy_tailed = False

    def build_likelihood(self, *args, **kwargs):
        raise NotImplementedError

    def logp(self, *args, **kwargs):
        raise NotImplementedError

    def draw_standardised_innovations(self, *args, **kwargs):
        raise NotImplementedError

    def variance_inflation(self, *args, **kwargs):
        raise NotImplementedError


class ForeignGaussian(Gaussian):
    """Impulso's Gaussian errors, subclassed outside Impulso.

    It serializes as a ``Gaussian`` does, so it loads back as one.
    """


def test_constructing_the_effect_without_impulso_raises(brand_mmm_data, monkeypatch):
    spec = VAR(lags=1)
    monkeypatch.setitem(sys.modules, "impulso", None)

    with pytest.raises(ImportError, match=r"pip install 'pymc-marketing\[var\]'"):
        make_effect(brand_mmm_data["brand_data"], var=spec)


def test_var_must_be_an_impulso_var_spec(brand_mmm_data):
    with pytest.raises(TypeError, match=r"impulso\.VAR"):
        make_effect(brand_mmm_data["brand_data"], var=MinnesotaPrior())


@pytest.mark.parametrize(
    "prior",
    [PlainPrior(), PydanticPrior(), SVDefaultPrior()],
    ids=lambda prior: type(prior).__name__,
)
def test_prior_other_than_minnesota_raises_at_construction(brand_mmm_data, prior):
    """No other prior loads back as it was saved, and the fitted MMM stores the spec."""
    spec = VAR(lags=1, prior=prior)

    with pytest.raises(
        TypeError,
        match=rf"prior must be a MinnesotaPrior.*got {type(prior).__name__}.*only a "
        r"MinnesotaPrior is supported for now.*QuantClimate/Impulso/issues/379",
    ):
        make_effect(brand_mmm_data["brand_data"], var=spec)


@pytest.mark.parametrize(
    "error_dist, match",
    [
        pytest.param(
            PlainErrors(), "Unable to serialize.*PlainErrors", id="not-serializable"
        ),
        pytest.param(
            ForeignGaussian(),
            "loads back as a different spec",
            id="loads-back-different",
        ),
    ],
)
def test_var_that_does_not_survive_saving_raises_at_construction(
    brand_mmm_data, error_dist, match
):
    """Fitting serializes the spec after sampling, and loading validates it back.

    Were loading to build a different spec, the reloaded MMM would silently be
    another model.
    """
    spec = VAR(lags=1, error_dist=error_dist)

    with pytest.raises(
        TypeError, match=rf"var must survive saving and loading.*{match}"
    ):
        make_effect(brand_mmm_data["brand_data"], var=spec)


@pytest.mark.parametrize("baseline_own_lag_mean", [np.nan, 1.0, -1.0])
def test_baseline_own_lag_mean_must_keep_the_baseline_stationary(
    brand_mmm_data, baseline_own_lag_mean
):
    with pytest.raises(ValidationError, match="baseline_own_lag_mean"):
        make_effect(
            brand_mmm_data["brand_data"], baseline_own_lag_mean=baseline_own_lag_mean
        )


@pytest.mark.parametrize("baseline_innovation_sd", [0.0, -0.1, np.nan, np.inf])
def test_baseline_innovation_sd_must_be_positive_and_finite(
    brand_mmm_data, baseline_innovation_sd
):
    with pytest.raises(ValidationError, match="baseline_innovation_sd"):
        make_effect(
            brand_mmm_data["brand_data"], baseline_innovation_sd=baseline_innovation_sd
        )


@pytest.mark.parametrize(
    "brand_columns, effect_kwargs, match",
    [
        pytest.param(
            {},
            {"endog_names": ["baseline", "awareness", "intent"]},
            r"\['intent'\] of endog_names are missing in brand_data",
            id="missing-column",
        ),
        pytest.param(
            {"awareness": lambda df: df["awareness"].map("{:.0%}".format)},
            {},
            r"\['awareness'\] of brand_data are not numeric",
            id="text",
        ),
        pytest.param(
            {"baseline": 0.0},
            {},
            "baseline_name 'baseline' must not be a column of brand_data",
            id="baseline-is-a-column",
        ),
        pytest.param(
            {},
            {"endog_names": ["awareness", "baseline", "consideration"]},
            "endog_names must start with baseline_name 'baseline'",
            id="baseline-not-first",
        ),
        pytest.param(
            {},
            {"endog_names": ["baseline"]},
            "endog_names must name at least one column of brand_data",
            id="no-observed-endog",
        ),
        pytest.param(
            {},
            {"endog_names": [*ENDOG_NAMES, "awareness"]},
            r"\['awareness'\] appear more than once in endog_names and exog_names",
            id="repeated-endog",
        ),
        pytest.param(
            {},
            {"exog_names": ["brand_spend", "brand_spend"]},
            r"\['brand_spend'\] appear more than once in endog_names and exog_names",
            id="repeated-exog",
        ),
        pytest.param(
            {},
            {"exog_names": ["brand_spend", "consideration"]},
            r"\['consideration'\] appear more than once in endog_names and exog_names",
            id="endog-and-exog",
        ),
    ],
)
def test_bad_brand_data_raises_at_construction(
    brand_mmm_data, brand_columns, effect_kwargs, match
):
    """Construction checks the names and columns; the values are checked at build."""
    brand_data = brand_mmm_data["brand_data"].assign(**brand_columns)

    with pytest.raises(ValueError, match=match):
        make_effect(brand_data, **effect_kwargs)


def test_brand_data_is_copied_at_construction(brand_mmm_data):
    """Later edits to the caller's frame leave the effect's copy unchanged."""
    brand_data = brand_mmm_data["brand_data"].copy()
    effect = make_effect(brand_data)

    brand_data.loc[3, "awareness"] = np.nan

    pd.testing.assert_frame_equal(effect.brand_data, brand_mmm_data["brand_data"])


def test_fit_adds_the_var_and_the_baseline_to_the_posterior(fitted_mmm):
    posterior = fitted_mmm.idata.posterior
    var_names = [
        "B",
        "B_exog",
        "intercept",
        "sigma_sd_0",
        "sigma_sd_1",
        "sigma_sd_2",
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


@pytest.mark.parametrize("exog_names", [["tv_spend"], []], ids=["exog", "no-exog"])
def test_fit_with_any_brand_column_names(brand_mmm_data, exog_names):
    """No column name is assumed, and exogenous columns are optional."""
    brand_data = brand_mmm_data["brand_data"].rename(
        columns={
            "awareness": "Aided awareness (%)",
            "consideration": "purchase_intent",
            "brand_spend": "tv_spend",
        }
    )
    endog_names = ["demand", "Aided awareness (%)", "purchase_intent"]
    effect = make_effect(
        brand_data,
        baseline_name="demand",
        endog_names=endog_names,
        exog_names=exog_names,
    )

    posterior = fit_mmm(brand_mmm_data, effect).idata.posterior

    assert posterior[f"{PREFIX}::B"].coords["var"].values.tolist() == endog_names
    assert list(posterior.indexes.get("exog", [])) == exog_names


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


def sample_path_start(
    data: dict, effect: VARBaselineEffect, **mmm_kwargs
) -> np.ndarray:
    """Prior draws of the baseline's first two values, for an effect with two lags."""
    latent = sample_prior(data, effect, [f"{PREFIX}::latent"], **mmm_kwargs)[
        f"{PREFIX}::latent"
    ]
    return latent.to_numpy()[:, :, :2, 0]


@pytest.mark.parametrize(
    "effect_kwargs",
    [{}, {"baseline_own_lag_mean": 0.6, "baseline_innovation_sd": 0.1}],
    ids=["default", "own-lag-mean-and-u"],
)
def test_baseline_starts_at_the_targets_sd(brand_mmm_data, effect_kwargs):
    """The baseline's first ``n_lags`` values have the target's sd.

    That is the sd in the target's original units, with ``ddof=0``, whatever
    ``baseline_own_lag_mean`` and U are.
    """
    effect = make_effect(brand_mmm_data["brand_data"], var=VAR(lags=2), **effect_kwargs)
    start = sample_path_start(brand_mmm_data, effect)

    np.testing.assert_allclose(
        start.std(axis=(0, 1)), brand_mmm_data["y"].std(ddof=0), rtol=0.05
    )


@pytest.mark.parametrize(
    "baseline_own_lag_mean", [0.0, 0.6], ids=["default", "own-lag-mean"]
)
def test_baseline_starts_at_its_stationary_sd_on_a_target_that_does_not_vary(
    brand_mmm_data, baseline_own_lag_mean
):
    """On the all-zero target, the start's sd falls back to the stationary sd.

    That is the sd of an AR(1) with the baseline's own-lag prior mean and innovation
    sd U: U itself at the default mean of 0.
    """
    effect = make_effect(
        brand_mmm_data["brand_data"],
        var=VAR(lags=2),
        baseline_own_lag_mean=baseline_own_lag_mean,
        baseline_innovation_sd=0.1,
    )
    zero_target = brand_mmm_data | {"y": 0.0 * brand_mmm_data["y"]}
    start = sample_path_start(zero_target, effect, scaling=FIXED_TARGET_SCALING)

    np.testing.assert_allclose(
        start.std(axis=(0, 1)), 0.1 / np.sqrt(1 - baseline_own_lag_mean**2), rtol=0.05
    )


def sample_prior(
    data: dict, effect: VARBaselineEffect, var_names: list[str], **mmm_kwargs
) -> xr.DataTree:
    """Prior draws of ``var_names`` from an MMM with ``effect``, built on ``data``."""
    mmm = make_mmm(effect, **mmm_kwargs)
    mmm.build_model(data["X"], data["y"])
    return pm.sample_prior_predictive(
        draws=10_000, var_names=var_names, model=mmm.model, random_seed=seed
    ).prior


def assert_draws_follow(draws: xr.DataArray, dist) -> None:
    """The draws have the quartiles of the frozen scipy distribution ``dist``."""
    np.testing.assert_allclose(draws.quantile(QUARTILES), dist.ppf(QUARTILES), rtol=0.1)


@pytest.mark.parametrize(
    "baseline_innovation_sd", [None, 0.1], ids=["from-target", "override"]
)
def test_baseline_innovation_sd_prior_is_half_normal_below_u(
    brand_mmm_data, baseline_innovation_sd
):
    """The baseline's innovation sd is HalfNormal(U / 1.96): P(sd > U) = 0.05.

    U is ``baseline_innovation_sd`` or, by default, the AR(1) residual sd of the
    target in its original units.
    """
    u = baseline_innovation_sd or ar1_residual_sd(brand_mmm_data["y"].to_frame())[0]
    effect = make_effect(
        brand_mmm_data["brand_data"], baseline_innovation_sd=baseline_innovation_sd
    )
    sigma = sample_prior(brand_mmm_data, effect, [f"{PREFIX}::sigma_sd_0"])[
        f"{PREFIX}::sigma_sd_0"
    ]

    np.testing.assert_allclose((sigma > u).mean(), 0.05, atol=0.01)
    assert_draws_follow(sigma, stats.halfnorm(scale=u / 1.96))


@pytest.mark.parametrize(
    "baseline_innovation_sd", [None, 0.1], ids=["from-target", "override"]
)
def test_exog_prior_scales_with_each_series(brand_mmm_data, baseline_innovation_sd):
    """Each equation's exog prior sd is proportional to its series' scale.

    Each brand metric's scale is its AR(1) residual sd, and the baseline's is the
    target's, whatever ``baseline_innovation_sd`` is: U sets only the baseline's
    innovation prior, ``HalfNormal(U / 1.96)``. So the ratios pin the scales the VAR
    is built with, and overriding U moves the innovation prior but neither the
    baseline's exog prior nor its Minnesota cross-lag prior, whose sd on brand metric
    ``j``'s lag is its own-lag sd times the cross shrinkage times the baseline's scale
    over ``j``'s.
    """
    brand_data = brand_mmm_data["brand_data"]
    target_scale = ar1_residual_sd(brand_mmm_data["y"].to_frame())[0]
    u = baseline_innovation_sd or target_scale
    observed_names = ENDOG_NAMES[1:]
    observed_scales = ar1_residual_sd(brand_data[observed_names])
    effect = make_effect(brand_data, baseline_innovation_sd=baseline_innovation_sd)
    draws = sample_prior(
        brand_mmm_data,
        effect,
        [f"{PREFIX}::sigma_sd_0", f"{PREFIX}::B", f"{PREFIX}::B_exog"],
    )
    prior = draws.std(("chain", "draw"))
    sd = prior[f"{PREFIX}::B_exog"].sel(exog="brand_spend")
    cross_lag_sd = prior[f"{PREFIX}::B"].sel(
        var="baseline", coeff=[f"L1.{name}" for name in observed_names]
    )
    own_lag_sd = prior[f"{PREFIX}::B"].sel(var="baseline", coeff="L1.baseline")

    assert_draws_follow(draws[f"{PREFIX}::sigma_sd_0"], stats.halfnorm(scale=u / 1.96))
    np.testing.assert_allclose(
        sd.sel(var="baseline") / sd.sel(var=observed_names),
        target_scale / observed_scales,
        rtol=0.1,
    )
    np.testing.assert_allclose(
        cross_lag_sd / own_lag_sd,
        MinnesotaPrior().cross_shrinkage * target_scale / observed_scales,
        rtol=0.1,
    )


@pytest.mark.parametrize(
    "volatility, observed_dists, tril_offdiag_sigma",
    [
        pytest.param(
            "constant",
            [stats.halfcauchy(scale=2.5)] * 2,
            0.5,
            id="shorthand",
        ),
        pytest.param(
            Constant(sigma_sd_beta=0.5, tril_offdiag_sigma=0.2),
            [stats.halfcauchy(scale=0.5)] * 2,
            0.2,
            id="sigma-sd-beta",
        ),
        pytest.param(
            Constant(
                tril_offdiag_sigma=0.2,
                innovation_scale_priors=[
                    InnovationScalePrior(family="halfcauchy", scale=2.5),
                    InnovationScalePrior(family="halfnormal", scale=0.3),
                    InnovationScalePrior(family="exponential", scale=0.1),
                ],
            ),
            [stats.halfnorm(scale=0.3), stats.expon(scale=0.1)],
            0.2,
            id="per-variable",
        ),
    ],
)
def test_brand_metrics_keep_the_analysts_volatility_prior(
    brand_mmm_data, volatility, observed_dists, tril_offdiag_sigma
):
    """Only the baseline's innovation-scale prior is the effect's own."""
    u = ar1_residual_sd(brand_mmm_data["y"].to_frame())[0]
    effect = make_effect(
        brand_mmm_data["brand_data"], var=VAR(lags=1, volatility=volatility)
    )
    var_names = [f"{PREFIX}::sigma_sd_{i}" for i in range(3)]
    prior = sample_prior(
        brand_mmm_data, effect, [*var_names, f"{PREFIX}::tril_offdiag"]
    )

    assert_draws_follow(prior[var_names[0]], stats.halfnorm(scale=u / 1.96))
    for name, dist in zip(var_names[1:], observed_dists, strict=True):
        assert_draws_follow(prior[name], dist)
    np.testing.assert_allclose(
        prior[f"{PREFIX}::tril_offdiag"].std(), tril_offdiag_sigma, rtol=0.1
    )


@pytest.mark.parametrize(
    "sigma_sd_beta, baseline_prior",
    [
        pytest.param(
            2.5,
            InnovationScalePrior(family="exponential", scale=5.0),
            id="other-family",
        ),
        pytest.param(
            0.5,
            InnovationScalePrior(family="halfcauchy", scale=2.5),
            id="other-sigma-sd-beta",
        ),
    ],
)
def test_innovation_scale_prior_for_the_baseline_raises_at_construction(
    brand_mmm_data, sigma_sd_beta, baseline_prior
):
    """The effect sets the baseline's entry from U, so the analyst's is the default.

    Impulso's default, ``HalfCauchy(sigma_sd_beta)``, is the one the effect replaces.
    """
    volatility = Constant(
        sigma_sd_beta=sigma_sd_beta,
        innovation_scale_priors=[
            baseline_prior,
            *[InnovationScalePrior(family="halfnormal", scale=0.3)] * 2,
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"innovation_scale_priors.*sets the baseline's entry.*from "
        r"baseline_innovation_sd.*Impulso's default, InnovationScalePrior\("
        rf"family='halfcauchy', scale={sigma_sd_beta}\)",
    ):
        make_effect(
            brand_mmm_data["brand_data"], var=VAR(lags=1, volatility=volatility)
        )


def test_volatility_other_than_constant_is_refused_by_impulso(brand_mmm_data):
    """The effect leaves any other volatility to Impulso, which names the reason."""
    effect = make_effect(brand_mmm_data["brand_data"], var=VAR(lags=1, volatility="sv"))
    mmm = make_mmm(effect)

    with pytest.raises(
        ValueError, match=r"volatility=StochasticVolatility.*latent series"
    ):
        mmm.build_model(brand_mmm_data["X"], brand_mmm_data["y"])


@pytest.mark.parametrize(
    "mmm_kwargs, baseline_innovation_sd, match",
    [
        pytest.param({}, None, r"target scale.*pass y", id="default"),
        pytest.param({}, 0.1, r"target scale.*pass y", id="baseline-innovation-sd"),
        pytest.param(
            {"scaling": FIXED_TARGET_SCALING},
            None,
            r"AR\(1\).*set baseline_innovation_sd",
            id="fixed-target-scale",
        ),
    ],
)
def test_building_without_a_target_raises(
    brand_mmm_data, mmm_kwargs, baseline_innovation_sd, match
):
    """Without ``y`` the MMM builds on an all-zero target.

    Its data-derived scale is 0, and its AR(1) residual sd cannot set U.
    """
    effect = make_effect(
        brand_mmm_data["brand_data"], baseline_innovation_sd=baseline_innovation_sd
    )
    mmm = make_mmm(effect, **mmm_kwargs)

    with pytest.raises(ValueError, match=rf"'{PREFIX}'.*{match}"):
        mmm.sample_prior_predictive(brand_mmm_data["X"])


def test_prior_predictive_without_a_target(brand_mmm_data):
    """A fixed target scale and ``baseline_innovation_sd`` replace ``y``.

    The all-zero target does not vary, so the baseline starts at sd U, in the
    target's original units, and its contribution is in units of the fixed target
    scale.
    """
    effect = make_effect(brand_mmm_data["brand_data"], baseline_innovation_sd=0.1)
    mmm = make_mmm(effect, scaling=FIXED_TARGET_SCALING)
    mmm.sample_prior_predictive(brand_mmm_data["X"], samples=4000, random_seed=seed)
    prior = mmm.idata.prior

    start = prior[f"{PREFIX}_effect_contribution"].isel(date=0)
    np.testing.assert_allclose(
        start.std(), 0.1 / FIXED_TARGET_SCALING["target"].value, rtol=0.05
    )
    sigma = prior[f"{PREFIX}::sigma_sd_0"]
    np.testing.assert_allclose((sigma > 0.1).mean(), 0.05, atol=0.02)


@pytest.fixture(scope="module")
def shorter_mmm_data(brand_mmm_data) -> dict:
    """The brand data's 30 weeks, and ``X`` and ``y`` on weeks 4 to 25."""
    mmm_weeks = slice(4, 26)
    return {
        "X": brand_mmm_data["X"].iloc[mmm_weeks],
        "y": brand_mmm_data["y"].iloc[mmm_weeks],
        "brand_data": brand_mmm_data["brand_data"],
    }


def initial_logp(mmm: MMM, X: pd.DataFrame, y: pd.Series) -> float:
    mmm.build_model(X, y)
    return mmm.model.compile_logp()(mmm.model.initial_point())


@pytest.fixture(scope="module")
def logp_on_mmm_dates(shorter_mmm_data) -> float:
    """The initial logp of an MMM whose brand data hold exactly its dates, in order."""
    X, y = shorter_mmm_data["X"], shorter_mmm_data["y"]
    brand_data = shorter_mmm_data["brand_data"]
    on_mmm_dates = brand_data[brand_data["date"].isin(X["date"])]
    return initial_logp(make_mmm(make_effect(on_mmm_dates)), X, y)


@pytest.mark.parametrize(
    "date_column, arrange",
    [
        pytest.param("date", lambda df: df, id="longer-period"),
        pytest.param(
            "date", lambda df: df.sample(frac=1, random_state=seed), id="any-row-order"
        ),
        pytest.param(
            "date",
            lambda df: df.assign(date=df["date"].dt.strftime("%Y-%m-%d")),
            id="string-dates",
        ),
        pytest.param("week", lambda df: df, id="date-column-name"),
    ],
)
def test_brand_data_is_matched_to_the_mmm_on_dates(
    shorter_mmm_data, logp_on_mmm_dates, date_column, arrange
):
    """Rows of brand_data are matched to the MMM by date, not by position."""
    rename = {"date": date_column}
    effect = make_effect(arrange(shorter_mmm_data["brand_data"]).rename(columns=rename))
    mmm = make_mmm(effect, date_column=date_column)
    X = shorter_mmm_data["X"].rename(columns=rename)

    np.testing.assert_allclose(
        initial_logp(mmm, X, shorter_mmm_data["y"]), logp_on_mmm_dates
    )


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(
            lambda df: df.assign(awareness=df["awareness"].mask(df.index == 2)),
            id="nan",
        ),
        pytest.param(
            lambda df: df.assign(
                consideration=df["consideration"].mask(df.index == 28, np.inf)
            ),
            id="inf",
        ),
        pytest.param(lambda df: pd.concat([df, df.iloc[[0, 29]]]), id="repeated-date"),
    ],
)
def test_brand_data_off_the_mmm_dates_is_not_checked(
    shorter_mmm_data, logp_on_mmm_dates, arrange
):
    """Rows on dates the MMM does not have are not used, so they may hold anything."""
    effect = make_effect(arrange(shorter_mmm_data["brand_data"]))
    X, y = shorter_mmm_data["X"], shorter_mmm_data["y"]

    np.testing.assert_allclose(initial_logp(make_mmm(effect), X, y), logp_on_mmm_dates)


@pytest.mark.parametrize(
    "date_column, arrange, match",
    [
        pytest.param(
            "week",
            lambda df: df,
            rf"'{PREFIX}'.*date column 'week'",
            id="no-date-column",
        ),
        pytest.param(
            "date",
            lambda df: df.drop(index=[6, 13]),
            rf"'{PREFIX}'.*MMM dates 2024-02-12, 2024-04-01\.$",
            id="missing-dates",
        ),
        pytest.param(
            "date",
            lambda df: df.assign(date=df["date"] + pd.Timedelta(days=6)),
            rf"'{PREFIX}'.*MMM dates 2024-01-29, 2024-02-05, 2024-02-12, 2024-02-19, "
            r"2024-02-26 and 17 more\. None of the MMM's dates is in brand_data: "
            "check that brand_data uses the same frequency and weekly anchor",
            id="every-date-missing",
        ),
        pytest.param(
            "date",
            lambda df: df.assign(date=df["date"].dt.tz_localize("UTC")),
            rf"'{PREFIX}'.*brand_data's dates are time-zone aware and the MMM's are "
            r"naive.*\.dt\.tz_localize\(None\)",
            id="time-zone-aware",
        ),
        pytest.param(
            "date",
            lambda df: pd.concat([df, df.iloc[[8]]]),
            rf"'{PREFIX}'.*dates 2024-02-26 appear more than once",
            id="repeated-date",
        ),
        pytest.param(
            "date",
            lambda df: pd.concat([df, df]),
            rf"'{PREFIX}'.*dates 2024-01-29, 2024-02-05, 2024-02-12, 2024-02-19, "
            "2024-02-26 and 17 more appear more than once",
            id="many-repeated-dates",
        ),
        pytest.param(
            "date",
            lambda df: df.assign(awareness=df["awareness"].mask(df.index == 10)),
            rf"'{PREFIX}'.*\['awareness'\] of brand_data contain NaN or infinite "
            "values on the MMM's dates",
            id="nan-on-mmm-dates",
        ),
        pytest.param(
            "date",
            lambda df: df.assign(
                consideration=df["consideration"].mask(df.index == 12, np.inf)
            ),
            rf"'{PREFIX}'.*\['consideration'\] of brand_data contain NaN or infinite "
            "values on the MMM's dates",
            id="inf-on-mmm-dates",
        ),
        pytest.param(
            "date",
            lambda df: df.assign(
                brand_spend=df["brand_spend"].mask(df.index == 20, -np.inf)
            ),
            rf"'{PREFIX}'.*\['brand_spend'\] of brand_data contain NaN or infinite "
            "values on the MMM's dates",
            id="minus-inf-on-mmm-dates",
        ),
        pytest.param(
            "date",
            lambda df: df.assign(brand_spend=(df.index < 4).astype(float)),
            rf"'{PREFIX}'.*\['brand_spend'\].*constant",
            id="constant-on-mmm-dates",
        ),
    ],
)
def test_bad_brand_data_raises_at_build(shorter_mmm_data, date_column, arrange, match):
    rename = {"date": date_column}
    effect = make_effect(arrange(shorter_mmm_data["brand_data"]))
    mmm = make_mmm(effect, date_column=date_column)

    with pytest.raises(ValueError, match=match):
        mmm.build_model(
            shorter_mmm_data["X"].rename(columns=rename), shorter_mmm_data["y"]
        )


def as_panel(data: dict, geos: tuple[str, ...] = ("north", "south")) -> dict:
    """The same data for each geo, stacked as a panel MMM takes it."""
    return {
        "X": pd.concat([data["X"].assign(geo=geo) for geo in geos], ignore_index=True),
        "y": pd.concat([data["y"]] * len(geos), ignore_index=True),
        "brand_data": pd.concat(
            [data["brand_data"].assign(geo=geo) for geo in geos], ignore_index=True
        ),
    }


@pytest.mark.parametrize(
    "mmm_kwargs, arrange, match",
    [
        pytest.param(
            {"dims": ("geo",)},
            as_panel,
            rf"'{PREFIX}'.*dims=\('geo',\).*cross-sectional",
            id="extra-dims",
        ),
        pytest.param(
            {"time_varying_intercept": True},
            lambda data: data,
            rf"'{PREFIX}'.*time_varying_intercept.*compete",
            id="time-varying-intercept",
        ),
        pytest.param(
            {
                "time_varying_intercept": SoftPlusHSGP.parameterize_from_data(
                    X=np.arange(30), dims=("date",)
                )
            },
            lambda data: data,
            rf"'{PREFIX}'.*time_varying_intercept.*compete",
            id="hsgp-intercept",
        ),
        pytest.param(
            {"link": "log"},
            lambda data: data,
            rf"'{PREFIX}'.*link='log'.*additively",
            id="log-link",
            marks=pytest.mark.filterwarnings(
                "ignore:The 'log' link is experimental", "ignore:With link='log'"
            ),
        ),
    ],
)
def test_unsupported_mmm_raises_at_build(brand_mmm_data, mmm_kwargs, arrange, match):
    """The MMM's configuration is refused before its data is matched."""
    data = arrange(brand_mmm_data)
    mmm = make_mmm(make_effect(data["brand_data"]), **mmm_kwargs)

    with pytest.raises(ValueError, match=match):
        mmm.build_model(data["X"], data["y"])


@pytest.mark.parametrize(
    "make_other, match",
    [
        pytest.param(
            lambda effect, data: make_effect(data, prefix="tv_var"),
            r"others with prefixes \['tv_var'\]",
            id="same-series",
        ),
        pytest.param(
            lambda effect, data: make_effect(data, prefix="tv_var", exog_names=[]),
            r"others with prefixes \['tv_var'\]",
            id="no-exog",
        ),
        pytest.param(
            lambda effect, data: make_effect(
                data, prefix="tv_var", endog_names=["baseline", "awareness", "intent"]
            ),
            r"others with prefixes \['tv_var'\]",
            id="other-endog",
        ),
        pytest.param(
            lambda effect, data: make_effect(
                data, prefix="tv_var", exog_names=["tv_spend"]
            ),
            r"others with prefixes \['tv_var'\]",
            id="other-exog",
        ),
        pytest.param(
            lambda effect, data: effect, "added more than once", id="same-instance"
        ),
    ],
)
def test_second_var_baseline_effect_raises(brand_mmm_data, make_other, match):
    """Over the same series, a second VAR would count the brand data twice.

    Over any series, only the sum of the two baselines would be identified. The
    same effect added twice is refused too.
    """
    brand_data = brand_mmm_data["brand_data"].assign(
        intent=lambda df: 2 * df["consideration"],
        tv_spend=lambda df: 1 - df["brand_spend"],
    )
    effect = make_effect(brand_data)
    mmm = make_mmm(effect).add_mu_effect(make_other(effect, brand_data))

    with pytest.raises(
        ValueError, match=rf"'{PREFIX}'.*at most one VARBaselineEffect.*{match}"
    ):
        mmm.build_model(brand_mmm_data["X"], brand_mmm_data["y"])


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


@pytest.mark.parametrize(
    "prepare",
    [
        pytest.param(lambda mmm, data: None, id="not-built"),
        pytest.param(
            lambda mmm, data: mmm.sample_prior_predictive(
                data["X"], data["y"], samples=10, random_seed=seed
            ),
            id="prior-only",
        ),
    ],
)
def test_fitted_var_before_fit_raises(brand_mmm_data, prepare):
    effect = make_effect(brand_mmm_data["brand_data"])
    mmm = make_mmm(effect)
    prepare(mmm, brand_mmm_data)

    with pytest.raises(
        RuntimeError, match=rf"'{PREFIX}'.*hasn't been fit yet, call \.fit\(\) first"
    ):
        effect.fitted_var(mmm)


@pytest.mark.parametrize(
    "effect_kwargs, match",
    [
        pytest.param(
            {"prefix": "tv_var"},
            r"'tv_var'.*no 'tv_var::' variables",
            id="other-prefix",
        ),
        pytest.param(
            {"endog_names": ["baseline", "consideration", "awareness"]},
            rf"'{PREFIX}'.*\['baseline', 'awareness', 'consideration'\]",
            id="other-endog-order",
        ),
        pytest.param(
            {"exog_names": []},
            rf"'{PREFIX}'.*\['brand_spend'\]",
            id="exog-dropped",
        ),
    ],
)
def test_fitted_var_on_an_mmm_fitted_with_another_effect_raises(
    fitted_mmm, brand_mmm_data, effect_kwargs, match
):
    """The effect must be the one the MMM was fitted with, with the same series."""
    effect = make_effect(brand_mmm_data["brand_data"], **effect_kwargs)

    with pytest.raises(ValueError, match=match):
        effect.fitted_var(fitted_mmm)


def test_fitted_var_holds_the_var_parameters(fitted_mmm):
    """The VAR's parameters and the sampler statistics, without the baseline path.

    Everything but the intercept passes through unchanged, under Impulso's names.
    """
    fitted = fitted_mmm.mu_effects[0].fitted_var(fitted_mmm)
    posterior = fitted.idata.posterior
    passed_through = [
        "B",
        "B_exog",
        "L",
        "Sigma",
        "tril_offdiag",
        "sigma_sd_0",
        "sigma_sd_1",
        "sigma_sd_2",
    ]

    assert set(fitted.idata.children) == {"posterior", "sample_stats"}
    assert set(posterior.data_vars) == {"intercept", *passed_through}
    for name in passed_through:
        np.testing.assert_array_equal(
            posterior[name], fitted_mmm.idata.posterior[f"{PREFIX}::{name}"]
        )
    assert posterior["intercept"].coords["var"].values.tolist() == ENDOG_NAMES
    assert "diverging" in fitted.idata.sample_stats


def test_fitted_var_data_are_the_raw_brand_data_on_the_mmm_dates(shorter_mmm_data):
    """The baseline is its posterior mean path; the brand columns are not centered."""
    effect = make_effect(shorter_mmm_data["brand_data"])
    mmm = fit_mmm(shorter_mmm_data, effect)
    mmm_dates = shorter_mmm_data["X"]["date"]
    brand_data = shorter_mmm_data["brand_data"]
    brand_data = brand_data[brand_data["date"].isin(mmm_dates)]
    baseline = mmm.idata.posterior[f"{PREFIX}::latent"].mean(("chain", "draw"))

    data = effect.fitted_var(mmm).data

    assert data.index.equals(pd.DatetimeIndex(mmm_dates))
    np.testing.assert_array_equal(
        data.endog, np.column_stack([baseline, brand_data[ENDOG_NAMES[1:]]])
    )
    np.testing.assert_array_equal(data.exog, brand_data[["brand_spend"]])


def one_step_mean(
    intercept: np.ndarray,
    B: np.ndarray,
    B_exog: np.ndarray | None,
    endog: np.ndarray,
    exog: np.ndarray | None,
) -> np.ndarray:
    """A VAR's mean at each date from ``n_lags`` on, given the data before it.

    The coefficients have dims ``(chain, draw, var, ...)``, with ``B``'s columns
    lag-major. ``endog`` is ``(date, var)``, or ``(chain, draw, date, var)`` for
    data that differ by draw. The mean has dims ``(chain, draw, time, var)``.
    """
    n_dates, n_vars = endog.shape[-2:]
    n_lags = B.shape[-1] // n_vars
    mean = intercept[..., None, :]
    for lag in range(1, n_lags + 1):
        A = B[..., (lag - 1) * n_vars : lag * n_vars]
        lagged = endog[..., n_lags - lag : n_dates - lag, :]
        mean = mean + np.einsum("...ij,...tj->...ti", A, lagged)
    if B_exog is not None:
        mean = mean + np.einsum("...ij,tj->...ti", B_exog, exog[n_lags:])
    return mean


def obs_potential(mmm: MMM) -> np.ndarray:
    """The graph's ``{prefix}::obs`` potential at each posterior draw.

    It is the log-likelihood of the brand metrics given the baseline's innovations.
    """
    model = mmm.model
    names = [rv.name for rv in model.free_RVs if rv.name.startswith(f"{PREFIX}::")]
    potential = model.compile_fn(
        model[f"{PREFIX}::obs"],
        inputs=[model[name] for name in names],
        on_unused_input="ignore",
    )
    draws = mmm.idata.posterior.to_dataset()[names]
    sizes = (draws.sizes["chain"], draws.sizes["draw"])
    values = [
        potential({name: draws[name].to_numpy()[index] for name in names})
        for index in np.ndindex(sizes)
    ]
    return np.reshape(values, sizes)


@pytest.mark.parametrize("lags", [1, 2])
@pytest.mark.parametrize("exog_names", [["brand_spend"], []], ids=["exog", "no-exog"])
def test_fitted_var_reproduces_the_mmm_graph(shorter_mmm_data, lags, exog_names):
    """At each draw, the ``FittedVAR`` on the raw data gives what the graph gave.

    The raw data take each draw's own baseline path. The baseline's one-step-ahead
    mean plus its scaled innovation is the path the graph generated, and the brand
    metrics' residuals from their one-step-ahead means, less the baseline
    innovation's share, have the log-likelihood the graph's potential gives them.
    """
    effect = make_effect(
        shorter_mmm_data["brand_data"], var=VAR(lags=lags), exog_names=exog_names
    )
    mmm = fit_mmm(shorter_mmm_data, effect)
    posterior = mmm.idata.posterior
    brand_data = shorter_mmm_data["brand_data"]
    brand_data = brand_data[brand_data["date"].isin(shorter_mmm_data["X"]["date"])]
    latent = posterior[f"{PREFIX}::latent"].to_numpy()
    innovations = posterior[f"{PREFIX}::latent_innovations"].to_numpy()[..., 0]
    observed = brand_data[ENDOG_NAMES[1:]].to_numpy()
    endog = np.concatenate(
        [latent, np.broadcast_to(observed, (*latent.shape[:2], *observed.shape))],
        axis=-1,
    )

    fitted = effect.fitted_var(mmm).idata.posterior
    mean = one_step_mean(
        fitted["intercept"].to_numpy(),
        fitted["B"].to_numpy(),
        fitted["B_exog"].to_numpy() if exog_names else None,
        endog,
        brand_data[exog_names].to_numpy() if exog_names else None,
    )
    L = fitted["L"].to_numpy()
    baseline = mean[..., 0] + L[:, :, None, 0, 0] * innovations
    residuals = (
        endog[..., lags:, 1:]
        - mean[..., 1:]
        - L[:, :, None, 1:, 0] * innovations[..., None]
    )
    loglik = [
        stats.multivariate_normal(cov=chol[1:, 1:] @ chol[1:, 1:].T).logpdf(resid).sum()
        for chol, resid in zip(
            L.reshape(-1, *L.shape[2:]),
            residuals.reshape(-1, *residuals.shape[2:]),
            strict=True,
        )
    ]

    np.testing.assert_allclose(baseline, latent[:, :, lags:, 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.reshape(loglik, L.shape[:2]), obs_potential(mmm), rtol=1e-12
    )


def test_dynamic_multiplier_runs_on_the_fitted_var(fitted_mmm):
    """On impact, the multiplier is ``B_exog``, and a week later ``A_1 B_exog``."""
    posterior = fitted_mmm.idata.posterior
    A_1 = posterior[f"{PREFIX}::B"].to_numpy()[..., : len(ENDOG_NAMES)]
    B_exog = posterior[f"{PREFIX}::B_exog"].to_numpy()
    fitted = fitted_mmm.mu_effects[0].fitted_var(fitted_mmm)

    multiplier = fitted.dynamic_multiplier(horizon=4)

    draws = multiplier.idata.posterior_predictive["dynamic_multiplier"]
    assert draws.sizes["horizon"] == 5
    np.testing.assert_allclose(draws.sel(horizon=0), B_exog)
    np.testing.assert_allclose(draws.sel(horizon=1), A_1 @ B_exog)


def true_cumulative_multiplier(horizon: int) -> float:
    """The true baseline's cumulative response to a unit of brand spend.

    It sums ``A^h b`` over ``h = 0, ..., horizon``, where ``A`` is the lag matrix
    of ``make_brand_mmm_data`` and ``b`` its brand spend loadings, in the order of
    ``ENDOG_NAMES``.
    """
    A = np.array([[0.5, 0.0, 0.6], [0.0, 0.9, 0.0], [0.0, 0.4, 0.8]])
    b = np.array([0.0, 0.25, 0.0])
    response = [np.linalg.matrix_power(A, h) @ b for h in range(horizon + 1)]
    return float(np.sum(response, axis=0)[0])


@pytest.mark.slow
def test_cumulative_multiplier_recovers_the_truth(nuts_fitted_mmm):
    """The 26-week cumulative multiplier of brand spend on the baseline is recovered.

    Its 94% HDI holds the true multiplier and excludes zero.
    """
    fitted = nuts_fitted_mmm.mu_effects[0].fitted_var(nuts_fitted_mmm)

    multiplier = fitted.dynamic_multiplier(horizon=26)

    cumulative = (
        multiplier.idata.posterior_predictive["dynamic_multiplier"]
        .sel(response="baseline", exog="brand_spend")
        .sum("horizon")
    )
    lower, upper = az.hdi(cumulative, prob=0.94).to_numpy()
    assert 0 < lower < true_cumulative_multiplier(26) < upper


@pytest.mark.slow
def test_uncentering_maps_var_fit_on_centered_data_onto_var_fit_on_raw_data(
    brand_mmm_data,
):
    """Uncentering ``VAR.fit`` on centered data gives ``VAR.fit`` on the raw data.

    This checks the uncentering against Impulso's own estimator, with the true
    baseline observed, and does not run the effect's model. The effect's baseline
    equation has no free intercept and its own priors, so the effect's posterior is
    not expected to match ``VAR.fit``; ``test_fitted_var_reproduces_the_mmm_graph``
    checks ``fitted_var`` against the effect's graph instead.

    The brand columns are centered as the effect centers them and the baseline not
    at all. The two fits' priors differ only in where the standard normal intercept
    prior sits, on the centered or the raw intercept, which at these posterior scales
    is far below Monte Carlo error. With over 1,000 effective draws per fit, the
    Monte Carlo error of a difference of means is about 0.04 posterior sd, so the
    means must agree within 0.2 posterior sd and the sds within 10%. Uncentering
    moves the intercepts by more than a posterior sd, far outside that tolerance.
    """
    from pymc_marketing.mmm.var_baseline import _uncenter

    brand_data = brand_mmm_data["brand_data"]
    endog = np.column_stack([brand_mmm_data["baseline"], brand_data[ENDOG_NAMES[1:]]])
    exog = brand_data[["brand_spend"]].to_numpy()
    endog_mean = np.concatenate([[0.0], endog[:, 1:].mean(axis=0)])
    exog_mean = exog.mean(axis=0)
    sampler = NUTSSampler(
        chains=4,
        cores=4,
        target_accept=0.95,
        random_seed=seed,
        nuts_sampler="pymc",
        progressbar=False,
    )

    def fit(endog: np.ndarray, exog: np.ndarray) -> xr.Dataset:
        data = VARData(
            endog=endog,
            endog_names=ENDOG_NAMES,
            exog=exog,
            exog_names=["brand_spend"],
            index=pd.DatetimeIndex(brand_data["date"]),
        )
        return VAR(lags=2).fit(data, sampler).idata.posterior.to_dataset()

    uncentered = _uncenter(
        fit(endog - endog_mean, exog - exog_mean), endog_mean, exog_mean
    )
    raw = fit(endog, exog)

    for name in ["intercept", "B", "B_exog"]:
        mean, sd = raw[name].mean(("chain", "draw")), raw[name].std(("chain", "draw"))
        np.testing.assert_array_less(
            abs(uncentered[name].mean(("chain", "draw")) - mean), 0.2 * sd
        )
        np.testing.assert_allclose(
            uncentered[name].std(("chain", "draw")), sd, rtol=0.1
        )


def save_and_load_effect(
    effect: VARBaselineEffect, path: Path, **changes
) -> VARBaselineEffect:
    """``effect`` through the JSON and netCDF that saving and loading an MMM use.

    ``changes`` replace entries of the saved dict.
    """
    xr.DataTree.from_dict(effect.idata_groups()).to_netcdf(path)
    data = json.loads(json.dumps(serialization.serialize(effect))) | changes
    with xr.open_datatree(path) as idata:
        return serialization.deserialize(data, DeserializationContext(idata=idata))


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(lambda df: df, id="dates"),
        pytest.param(
            lambda df: df.assign(date=df["date"].dt.strftime("%Y-%m-%d")),
            id="string-dates",
        ),
        pytest.param(
            lambda df: df.assign(source="survey", wave=np.arange(len(df)), panel=True),
            id="other-columns",
        ),
        pytest.param(
            lambda df: df.assign(note=pd.Series("n/a", index=df.index, dtype=object)),
            id="object-column",
        ),
        pytest.param(lambda df: df.set_axis(df.index * 2 + 100), id="index"),
        pytest.param(lambda df: df.reset_index(), id="index-column"),
    ],
)
def test_brand_data_is_saved_and_loaded(brand_mmm_data, tmp_path, arrange):
    """Its columns, dtypes and rows come back in order, but not its index."""
    brand_data = arrange(brand_mmm_data["brand_data"])
    effect = make_effect(brand_data)

    loaded = save_and_load_effect(effect, tmp_path / "effect.nc")

    pd.testing.assert_frame_equal(loaded.brand_data, brand_data.reset_index(drop=True))
    assert loaded == effect


def test_missing_text_loads_as_an_empty_string(brand_mmm_data, tmp_path):
    """netCDF has no missing text value."""
    brand_data = brand_mmm_data["brand_data"].assign(source="survey")
    brand_data.loc[3, "source"] = None
    effect = make_effect(brand_data)

    loaded = save_and_load_effect(effect, tmp_path / "effect.nc")

    pd.testing.assert_frame_equal(loaded.brand_data, brand_data.fillna({"source": ""}))


def test_saved_effect_records_its_format_version(brand_mmm_data):
    effect = make_effect(brand_mmm_data["brand_data"])

    assert serialization.serialize(effect)["format_version"] == 1


def test_loading_a_newer_format_version_raises(brand_mmm_data, tmp_path):
    """A layout this version does not read is refused rather than misread."""
    from pymc_marketing.serialization import SerializationError

    effect = make_effect(brand_mmm_data["brand_data"])

    with pytest.raises(
        SerializationError, match=r"format version 2: .* reads only format version 1"
    ):
        save_and_load_effect(effect, tmp_path / "effect.nc", format_version=2)


@pytest.fixture(scope="module")
def saved_mmm(brand_mmm_data, tmp_path_factory) -> tuple[MMM, Path]:
    """A fitted MMM carrying the effect, and the file it is saved to.

    The effect sets every field but ``baseline_innovation_sd``, so U comes from the
    target. Its brand data have a text column the effect does not use and a
    non-default index.
    """
    spec = VAR(
        lags=2,
        prior=MinnesotaPrior(own_lag_mean=(0.3, 0.9, 0.8)),
        volatility=Constant(
            tril_offdiag_sigma=0.2,
            innovation_scale_priors=[
                InnovationScalePrior(family="halfcauchy", scale=2.5),
                InnovationScalePrior(family="halfnormal", scale=0.3),
                InnovationScalePrior(family="exponential", scale=0.1),
            ],
        ),
        error_dist=Gaussian(),
    )
    brand_data = brand_mmm_data["brand_data"].assign(source="survey")
    effect = make_effect(
        brand_data.set_axis(brand_data.index + 100),
        var=spec,
        baseline_own_lag_mean=0.3,
    )
    mmm = fit_mmm(brand_mmm_data, effect)
    path = tmp_path_factory.mktemp("var_baseline") / "mmm.nc"
    mmm.save(str(path))
    return mmm, path


@pytest.fixture(scope="module")
def loaded_mmm(saved_mmm) -> MMM:
    return MMM.load(str(saved_mmm[1]))


def test_loaded_effect_equals_the_saved_one(saved_mmm, loaded_mmm):
    """``==`` compares every field, the VAR spec included, and brand_data's rows."""
    effect = saved_mmm[0].mu_effects[0]
    loaded = loaded_mmm.mu_effects[0]

    assert isinstance(loaded, VARBaselineEffect)
    assert loaded == effect


def test_fitted_var_is_unchanged_by_loading(saved_mmm, loaded_mmm):
    """It recomputes the centering means from brand_data and U from the target."""
    mmm = saved_mmm[0]
    fitted = mmm.mu_effects[0].fitted_var(mmm)

    loaded = loaded_mmm.mu_effects[0].fitted_var(loaded_mmm)

    xr.testing.assert_equal(
        loaded.idata.posterior.to_dataset(), fitted.idata.posterior.to_dataset()
    )
    np.testing.assert_array_equal(loaded.data.endog, fitted.data.endog)
    np.testing.assert_array_equal(loaded.data.exog, fitted.data.exog)
    assert loaded.data.index.equals(fitted.data.index)
    assert loaded.n_lags == fitted.n_lags
    assert loaded.volatility == fitted.volatility
    assert loaded.error_dist == fitted.error_dist


def test_loaded_mmm_equals_the_saved_one(saved_mmm, loaded_mmm):
    assert loaded_mmm == saved_mmm[0]


def test_posterior_predictive_is_unchanged_by_loading(
    brand_mmm_data, saved_mmm, loaded_mmm
):
    """On the training dates, through the graph that loading rebuilds and clones."""
    X = brand_mmm_data["X"]
    kwargs = {"extend_idata": False, "random_seed": seed, "progressbar": False}

    loaded = loaded_mmm.sample_posterior_predictive(X, **kwargs)

    saved = saved_mmm[0].sample_posterior_predictive(X, **kwargs)
    assert loaded["date"].to_index().equals(pd.DatetimeIndex(X["date"]))
    xr.testing.assert_equal(loaded, saved)


def test_loading_without_impulso_raises(saved_mmm, monkeypatch):
    monkeypatch.setitem(sys.modules, "impulso", None)

    with pytest.raises(ImportError, match=r"pip install 'pymc-marketing\[var\]'"):
        MMM.load(str(saved_mmm[1]))


def test_mmm_with_events_of_the_same_prefix_saves_and_loads(brand_mmm_data, tmp_path):
    """The brand data and the events are saved in different idata groups."""
    from pymc_extras.prior import Prior

    from pymc_marketing.mmm.events import EventEffect, GaussianBasis

    df_events = pd.DataFrame(
        {
            "name": ["launch", "summer"],
            "start_date": ["2024-02-05", "2024-06-03"],
            "end_date": ["2024-02-19", "2024-06-17"],
        }
    )
    effect = make_effect(brand_mmm_data["brand_data"])
    events = EventEffect(
        basis=GaussianBasis(), effect_size=Prior("Normal"), dims=(PREFIX,)
    )
    mmm = make_mmm(effect).add_events(df_events, prefix=PREFIX, effect=events)
    with patch.object(pm, "sample", mock_sample):
        mmm.fit(brand_mmm_data["X"], brand_mmm_data["y"], draws=20, random_seed=seed)
    mmm.save(str(tmp_path / "mmm.nc"))

    loaded_effect, loaded_events = MMM.load(str(tmp_path / "mmm.nc")).mu_effects

    assert loaded_effect == effect
    # MMM == raises: EventAdditiveEffect reloads df_events with an extra `index` column.
    pd.testing.assert_frame_equal(loaded_events.df_events[df_events.columns], df_events)


@pytest.mark.parametrize(
    "arrange, effect_kwargs, equal",
    [
        pytest.param(lambda df: df.copy(), {}, True, id="copy"),
        pytest.param(
            lambda df: df.set_axis(df.index + 100), {}, True, id="other-index"
        ),
        pytest.param(
            lambda df: df.assign(awareness=df["awareness"] + 1.0),
            {},
            False,
            id="other-values",
        ),
        pytest.param(lambda df: df, {"var": VAR(lags=2)}, False, id="other-var"),
    ],
)
def test_effects_compare_brand_data_with_pandas(
    brand_mmm_data, arrange, effect_kwargs, equal
):
    """Its index is ignored. MMMs carrying the effects compare as the effects do."""
    brand_data = brand_mmm_data["brand_data"]
    effect = make_effect(brand_data)
    other = make_effect(arrange(brand_data), **effect_kwargs)

    assert (effect == other) is equal
    assert (make_mmm(effect) == make_mmm(other)) is equal


MEASUREMENT_ONLY = (
    rf"VARBaselineEffect '{PREFIX}' takes only the 30 dates the MMM was fitted on, "
    "2024-01-01 to 2024-07-22, so prediction on any other set of dates is not "
    "supported"
)


def later_dates(X: pd.DataFrame) -> pd.DataFrame:
    """``X`` moved to as many dates right after the MMM's."""
    return X.assign(date=X["date"] + pd.Timedelta(weeks=len(X)))


@pytest.mark.parametrize("method", ["sample_posterior_predictive", "predict"])
@pytest.mark.parametrize(
    "arrange, kwargs",
    [
        pytest.param(later_dates, {}, id="later-dates"),
        pytest.param(lambda X: X.iloc[10:20], {}, id="subset-of-training-dates"),
        pytest.param(
            later_dates,
            {"include_last_observations": True},
            id="include-last-observations",
        ),
        pytest.param(later_dates, {"clone_model": False}, id="no-clone"),
    ],
)
def test_prediction_on_new_dates_raises(
    fitted_mmm, brand_mmm_data, method, arrange, kwargs
):
    """The MMM refuses before it sets the new data, so its model keeps its dates."""
    X = brand_mmm_data["X"]

    with pytest.raises(NotImplementedError, match=MEASUREMENT_ONLY):
        getattr(fitted_mmm, method)(arrange(X), progressbar=False, **kwargs)

    np.testing.assert_array_equal(
        np.asarray(fitted_mmm.model.coords["date"]), X["date"].to_numpy()
    )


BUDGET_WINDOWS = [
    pytest.param("2024-07-29", "2024-09-16", id="later-dates"),
    pytest.param("2024-03-11", "2024-04-29", id="fitted-dates"),
]


@pytest.mark.parametrize("start_date, end_date", BUDGET_WINDOWS)
def test_budget_optimizer_ignores_the_baseline(fitted_mmm, start_date, end_date):
    """The optimizer sets its own window, but the default objective is the channels'.

    The baseline does not enter their contributions, so the window's dates need no
    brand data.
    """
    optimizer = fitted_mmm.budget_optimizer(start_date=start_date, end_date=end_date)

    result = optimizer.allocate_budget(total_budget=10.0)

    assert result.scipy_result.success
    np.testing.assert_allclose(result.budgets.sum(), 10.0)


@pytest.mark.parametrize("start_date, end_date", BUDGET_WINDOWS)
def test_budget_objective_with_the_baseline_raises(fitted_mmm, start_date, end_date):
    """The baseline is a path over all of the MMM's dates, so it fits no window.

    Not even one within those dates.
    """
    with pytest.raises(
        NotImplementedError,
        match="'total_response_original_scale' is not supported, because it "
        rf"includes the contribution of VARBaselineEffect '{PREFIX}', which exists "
        r"only on the dates the MMM was fitted on\. The default objective",
    ):
        fitted_mmm.budget_optimizer(
            start_date=start_date,
            end_date=end_date,
            response_variable="total_response_original_scale",
        )


def test_time_slice_cross_validation_raises_before_fitting(brand_mmm_data):
    """Each fold predicts on its test dates, so the first fold is not fitted at all."""
    mmm = make_mmm(make_effect(brand_mmm_data["brand_data"]))
    cv = TimeSliceCrossValidator(n_init=20, forecast_horizon=4, date_column="date")

    with (
        patch.object(pm, "sample", wraps=mock_sample) as sample,
        pytest.raises(
            NotImplementedError,
            match="each fold's test dates, but the contribution of "
            rf"VARBaselineEffect '{PREFIX}' exists only",
        ),
    ):
        cv.run(brand_mmm_data["X"], brand_mmm_data["y"], mmm=mmm)

    sample.assert_not_called()


def test_do_on_the_channel_data(fitted_mmm):
    """``pm.do`` clones the MMM's model, VAR included."""
    channel_data = fitted_mmm.xarray_dataset["_channel"]

    model = pm.do(fitted_mmm.model, {"channel_data": np.zeros(channel_data.shape)})

    assert f"{PREFIX}_effect_contribution" in model.named_vars


def contribution_lost(mmm: MMM, X: pd.DataFrame, period: pd.Series) -> xr.DataArray:
    """Each channel's contribution lost without spend over ``period``.

    In the target's units, summed over all of the MMM's dates, so it carries the
    adstock's carryover past the period.
    """
    without_spend = X.assign(
        **{channel: X[channel].where(~period, 0.0) for channel in mmm.channel_columns}
    )
    contribution = mmm.sample_posterior_predictive(
        without_spend,
        var_names=["channel_contribution"],
        extend_idata=False,
        combined=False,
        progressbar=False,
    )["channel_contribution"]
    lost = mmm.idata.posterior["channel_contribution"] - contribution
    return (lost * mmm.idata.constant_data["target_scale"]).sum("date")


@pytest.mark.parametrize("mmm_name", ["fitted_mmm", "loaded_mmm"])
@pytest.mark.parametrize(
    "kwargs, frequency",
    [
        pytest.param({"frequency": "all_time"}, None, id="all-time"),
        pytest.param({"frequency": "monthly"}, "M", id="monthly"),
        pytest.param({"frequency": "quarterly"}, "Q", id="quarterly"),
        pytest.param(
            {
                "frequency": "all_time",
                "start_date": "2024-03-04",
                "end_date": "2024-05-27",
            },
            None,
            id="window",
        ),
    ],
)
def test_incremental_contribution_is_the_channel_contribution(
    request, brand_mmm_data, mmm_name, kwargs, frequency
):
    """The baseline does not depend on spend, so it adds nothing to the increment.

    Without spend over a period, each channel's increment on it is the contribution
    it loses, in the target's units. Periods are calendar periods, labeled by their
    last day.
    """
    mmm = request.getfixturevalue(mmm_name)
    X = brand_mmm_data["X"]
    dates = X["date"]
    in_range = dates.between(
        kwargs.get("start_date", dates.min()), kwargs.get("end_date", dates.max())
    )

    incremental = mmm.incrementality.compute_incremental_contribution(**kwargs)

    if frequency is None:
        expected = contribution_lost(mmm, X, in_range)
    else:
        ends = dates.dt.to_period(frequency).dt.end_time.dt.normalize()
        expected = xr.concat(
            [
                contribution_lost(mmm, X, in_range & (ends == end)).assign_coords(
                    date=end
                )
                for end in ends.unique()
            ],
            dim="date",
        )
    xr.testing.assert_allclose(incremental, expected.transpose(*incremental.dims))


@pytest.mark.parametrize("mmm_name", ["fitted_mmm", "loaded_mmm"])
def test_monthly_contribution_over_spend(request, mmm_name):
    """Each channel's contribution per unit of spend is finite in each of 7 months."""
    mmm = request.getfixturevalue(mmm_name)

    over_spend = mmm.incrementality.contribution_over_spend(frequency="monthly")

    assert over_spend.dims == ("chain", "draw", "date", "channel")
    assert over_spend.sizes["date"] == 7
    assert np.isfinite(over_spend).all()
