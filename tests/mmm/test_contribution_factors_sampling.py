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
"""Statistical and lifecycle contract for fixed-training contribution factors."""

import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pymc_extras.prior import Prior
from scipy.stats import norm

from tests.mmm.contribution_factors_helpers import data as data
from tests.mmm.contribution_factors_helpers import evaluate, make_model
from tests.mmm.contribution_factors_helpers import labeled_panel as labeled_panel


def test_effective_prior_removes_sales_likelihood_retains_factors(data):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor(
        "belief", ["search"], "share", Prior("Normal", mu=0.3, sigma=0.1)
    )
    from unittest.mock import patch

    def inspect_model(**kwargs):
        prior_model = pm.modelcontext(None)
        assert not prior_model.observed_RVs
        point = prior_model.initial_point()
        assert np.isfinite(prior_model.compile_dlogp()(point)).all()
        actual = prior_model.compile_logp()(point)
        parameter_logp = evaluate(
            prior_model, [prior_model.logp(vars=prior_model.free_RVs)], point
        )[0]
        share = evaluate(prior_model, [prior_model["belief"]], point)[0]
        np.testing.assert_allclose(
            actual - parameter_logp, norm.logpdf(share, 0.3, 0.1)
        )
        return xr.DataTree.from_dict(
            {"posterior": xr.Dataset({"belief": (("chain", "draw"), [[share]])})}
        )

    with patch("pymc.sample", side_effect=inspect_model):
        result = m.sample_effective_prior()
    assert "prior" in result and "posterior" not in result
    with pytest.warns(UserWarning, match="not the effective prior"):
        m.sample_prior_predictive(*data, samples=2, random_seed=123)


def test_public_prediction_optimizer_and_effective_prior(data, mock_pymc_sample):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor(
        "belief", ["search"], "share", Prior("Normal", mu=0.3, sigma=0.1)
    )
    m.fit(*data)
    point = m.model.initial_point()
    before = evaluate(
        m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)], point
    )
    posterior = m.idata.posterior.dataset.copy(deep=True)
    future = data[0].assign(
        date=pd.date_range("2026-01-01", periods=5), search=100.0, tiktok=200.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m.sample_posterior_predictive(future, extend_idata=False, random_seed=123)
        result = m.budget_optimizer("2026-01-01", "2026-01-03").allocate_budget(
            total_budget=30.0
        )
    assert result is not None
    np.testing.assert_allclose(
        evaluate(
            m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)], point
        ),
        before,
    )
    posterior = m.idata.posterior.dataset.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        draws = m.sample_effective_prior(
            draws=3,
            tune=3,
            chains=1,
            cores=1,
            random_seed=123,
            progressbar=False,
            compute_convergence_checks=False,
        )
    assert "prior" in draws
    xr.testing.assert_equal(
        m.idata.posterior.dataset[list(posterior.data_vars)], posterior
    )


def test_effective_prior_panel_target_and_predictive_parameter_preservation(
    labeled_panel,
):
    from unittest.mock import patch

    ds, _, _ = labeled_panel
    m = make_model(dims=("region", "product"))
    m.build_model(ds)
    m.add_original_scale_contribution_variable(["channel_contribution"])
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    point = m.model.initial_point()
    before = evaluate(
        m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)], point
    )

    def check_sampling_target(**kwargs):
        sampling = pm.modelcontext(None)
        assert not sampling.observed_RVs
        np.testing.assert_allclose(
            evaluate(
                sampling,
                [sampling["belief"], sampling.logp(vars=sampling.potentials)],
                point,
            ),
            before,
        )
        # Forward draws here supply test values, not an effective-prior approximation.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            prior = pm.sample_prior_predictive(draws=4, random_seed=123).prior.dataset
        return xr.DataTree.from_dict({"posterior": prior})

    with patch("pymc.sample", side_effect=check_sampling_target):
        result = m.sample_effective_prior()
    with m.model, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictive = pm.sample_posterior_predictive(
            result.prior.dataset, var_names=["y", "belief"], random_seed=123
        )
    xr.testing.assert_allclose(
        predictive.posterior_predictive.belief, result.prior.belief
    )


def test_effective_prior_rejects_target_override(data):
    m = make_model()
    m.build_model(*data)
    with pytest.raises(ValueError, match="controls its model"):
        m.sample_effective_prior(model=m.model)


def test_effective_prior_preserves_user_parameters_and_deterministics(data):
    from unittest.mock import patch

    import pymc.dims as pmd

    m = make_model()
    m.build_model(*data)
    with m.model:
        extra = pmd.Normal("extra_parameter", mu=1, sigma=2)
        pmd.Deterministic("extra_squared", extra**2)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))

    def inspect(**kwargs):
        sampling = pm.modelcontext(None)
        assert "extra_parameter" in {v.name for v in sampling.free_RVs}
        assert "extra_squared" in sampling.named_vars
        assert not sampling.observed_RVs
        return xr.DataTree.from_dict({"posterior": xr.Dataset()})

    with patch("pymc.sample", side_effect=inspect):
        m.sample_effective_prior()


@pytest.mark.parametrize("extra_term", ["potential", "observation"])
def test_effective_prior_rejects_additional_calibration_terms(data, extra_term):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    with m.model:
        if extra_term == "potential":
            pm.Potential("external_calibration", pm.math.constant(0.0))
        else:
            pm.Normal("external_observation", observed=0.0)
    model = m.model
    names = set(model.named_vars)
    with pytest.raises(ValueError, match="additional likelihoods/Potentials"):
        m.sample_effective_prior()
    assert m.model is model
    assert set(model.named_vars) == names
    assert m.idata is None


@pytest.mark.parametrize(
    "prior",
    [
        Prior("Normal"),
        Prior("LogNormal", mu=-1, sigma=0.5),
        Prior("HalfNormal", sigma=0.5),
    ],
)
def test_effective_prior_returns_prior_sampling_diagnostics(data, monkeypatch, prior):
    from pymc.sampling.mcmc import sample

    # Other tests use a module-scoped mock; this contract needs actual MCMC stats.
    monkeypatch.setattr(pm, "sample", sample)
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", prior)
    # A short real chain checks the sampling result contract, not convergence.
    result = m.sample_effective_prior(
        draws=8,
        tune=8,
        chains=1,
        cores=1,
        progressbar=False,
        compute_convergence_checks=False,
        random_seed=123,
    )
    assert "posterior" not in result
    # The positive channel-effect priors also exclude negative shares when
    # the additional weighting density itself has real-line support.
    assert bool((result.prior.belief >= 0).all())
    assert "sample_stats" not in result
    assert result.prior.sizes["draw"] == 8
    assert result.sample_stats_prior.sizes["draw"] == 8
    assert "diverging" in result.sample_stats_prior
    assert np.isfinite(result.prior.belief).all()
    assert m.idata is None


def test_effective_prior_merges_sampler_defaults(data):
    from unittest.mock import patch

    config = {"nuts_sampler": "numpyro", "draws": 777, "target_accept": 0.99}
    m = make_model(sampler_config=config)
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    with patch(
        "pymc.sample", return_value=xr.DataTree.from_dict({"posterior": xr.Dataset()})
    ) as sample:
        m.sample_effective_prior(draws=12, random_seed=42)
    assert sample.call_args.kwargs == {**config, "draws": 12, "random_seed": 42}
    assert m.sampler_config == config


@pytest.mark.slow
def test_effective_prior_matches_independent_importance_weighting(data, monkeypatch):
    """Prior-only MCMC agrees with forward draws weighted by the supplied density."""
    from pymc.sampling.mcmc import sample

    monkeypatch.setattr(pm, "sample", sample)
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor(
        "belief", ["search"], "share", Prior("Normal", mu=0.1, sigma=0.015)
    )
    with m.model, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The effect of Potentials.*")
        forward = pm.sample_prior_predictive(
            draws=200_000, var_names=["belief"], random_seed=731
        )
    shares = forward.prior.belief.values.ravel()
    order = np.argsort(shares)
    weights = norm.pdf(shares[order], loc=0.1, scale=0.015)
    cdf = (weights.cumsum() - 0.5 * weights) / weights.sum()
    probabilities = [0.05, 0.25, 0.5, 0.75, 0.95]
    expected = np.interp(probabilities, cdf, shares[order])
    effective = m.sample_effective_prior(
        draws=1000,
        tune=1000,
        chains=2,
        cores=1,
        target_accept=0.95,
        random_seed=732,
        progressbar=False,
    )
    actual = np.quantile(effective.prior.belief.values, probabilities)
    # This tolerance accommodates Monte Carlo error while excluding unweighted
    # parameter-prior draws and sales-conditioned posterior draws.
    np.testing.assert_allclose(actual, expected, atol=0.01, rtol=0)
    assert not effective.sample_stats_prior.diverging.any()


def test_effective_prior_restores_historical_inputs_after_public_prediction(
    data, mock_pymc_sample
):
    from unittest.mock import patch

    import pymc.dims as pmd

    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    with m.model:
        extra_data = pmd.Data("user_data", np.array(3.0))
        pmd.Deterministic(
            "historical_user_sum", m.model["channel_data"].sum() + extra_data
        )
    expected = data[0][["search", "tiktok"]].to_numpy().sum() + 3.0
    m.fit(*data)
    future = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8),
            "search": 100.0,
            "tiktok": 200.0,
        }
    )
    m.sample_posterior_predictive(
        future, clone_model=False, extend_idata=False, random_seed=123
    )

    def inspect(**kwargs):
        sampling = pm.modelcontext(None)
        assert len(sampling.coords["date"]) == len(data[0])
        np.testing.assert_allclose(
            evaluate(sampling, [sampling["historical_user_sum"]])[0], expected
        )
        return xr.DataTree.from_dict({"posterior": xr.Dataset()})

    with patch("pymc.sample", side_effect=inspect):
        m.sample_effective_prior()
    # Sampling does not undo the caller's prediction data update.
    assert len(m.model.coords["date"]) == 8


def test_effective_prior_with_permuted_control_axes(labeled_panel):
    from unittest.mock import patch

    ds, _, _ = labeled_panel
    ds["_control"] = xr.DataArray(
        np.ones((2, 2, 5, 1)),
        dims=("product", "region", "date", "control"),
        coords={
            **{d: ds.coords[d] for d in ("product", "region", "date")},
            "control": ["promo"],
        },
    )
    m = make_model(dims=("region", "product"), control_columns=["promo"])
    m.build_model(ds)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    expected = evaluate(m.model, [m.model["belief"]])[0]

    def sample(**kwargs):
        model = pm.modelcontext(None)
        np.testing.assert_allclose(evaluate(model, [model["belief"]])[0], expected)
        return xr.DataTree.from_dict({"posterior": xr.Dataset()})

    with patch("pymc.sample", side_effect=sample):
        m.sample_effective_prior()
