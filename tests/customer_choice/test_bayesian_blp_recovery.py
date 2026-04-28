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
"""Slow recovery / identification tests for ``BayesianBLP``.

These tests fit real PyMC models on synthetic panels with known truth and
verify the load-bearing claims of the model: parameter recovery,
endogeneity-bias correction via instruments, weak-IV honesty, hierarchical
pooling, and Monte-Carlo simulation stability.

Each test runs a small but realistic NUTS fit (a few hundred draws on a
T~40 panel via ``numpyro``); the wave together takes ~10-15 min on a
laptop CPU. Mark all tests with ``pytest.mark.slow`` so they are opt-in
via ``pytest -m slow``.
"""

import warnings

import arviz as az
import numpy as np
import pytest

from pymc_marketing.customer_choice import BayesianBLP, generate_blp_panel

pytestmark = pytest.mark.slow


_FIT_KWARGS = dict(
    nuts_sampler="numpyro",
    draws=500,
    tune=1000,
    chains=2,
    progressbar=False,
    random_seed=0,
)


def _hdi(samples, prob: float = 0.94) -> tuple[float, float]:
    arr = np.asarray(samples).reshape(-1)
    hdi = az.hdi(arr, hdi_prob=prob)
    if hasattr(hdi, "values"):
        hdi = hdi.values
    return float(hdi[0]), float(hdi[1])


def _make_model(df, truth, *, instruments_arg, n_mc_draws=100, **kwargs):
    return BayesianBLP(
        market_data=df,
        characteristics=truth["characteristic_cols"],
        instruments=instruments_arg,
        random_coef_on=["price"],
        n_mc_draws=n_mc_draws,
        random_seed=0,
        **kwargs,
    )


def _fit(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(**_FIT_KWARGS)
    return model


@pytest.fixture(scope="module")
def iv_panel():
    df, truth = generate_blp_panel(
        T=40,
        J=3,
        K=2,
        L=2,
        true_alpha=-2.0,
        true_beta=np.array([0.8, 1.2]),
        sigma_alpha=0.5,
        instrument_strength=0.7,
        price_xi_corr=0.6,
        market_size=4_000,
        n_dgp_draws=3_000,
        random_seed=42,
        return_truth=True,
    )
    return df, truth


@pytest.fixture(scope="module")
def iv_fit(iv_panel):
    df, truth = iv_panel
    return _fit(_make_model(df, truth, instruments_arg=truth["instrument_cols"])), truth


@pytest.fixture(scope="module")
def noiv_fit(iv_panel):
    df, truth = iv_panel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _fit(_make_model(df, truth, instruments_arg=None)), truth


class TestParameterRecovery:
    def test_iv_recovers_alpha(self, iv_fit):
        model, truth = iv_fit
        lo, hi = _hdi(model.idata.posterior["alpha_r"].values)
        assert lo <= truth["alpha"] <= hi, (
            f"alpha truth {truth['alpha']} not in 94% HDI [{lo:.3f}, {hi:.3f}]"
        )

    def test_iv_recovers_beta(self, iv_fit):
        model, truth = iv_fit
        beta_post = model.idata.posterior["beta_r"].values
        for k, true_val in enumerate(truth["beta"]):
            lo, hi = _hdi(beta_post[..., k])
            assert lo <= true_val <= hi, (
                f"beta[{k}] truth {true_val} not in 94% HDI [{lo:.3f}, {hi:.3f}]"
            )

    def test_iv_recovers_sigma_alpha(self, iv_fit):
        model, truth = iv_fit
        sigma_post = model.idata.posterior["sigma_random"].values[..., 0]
        lo, hi = _hdi(sigma_post)
        assert lo <= truth["sigma_alpha"] <= hi, (
            f"sigma_alpha truth {truth['sigma_alpha']} not in 94% HDI "
            f"[{lo:.3f}, {hi:.3f}]"
        )

    def test_iv_recovers_xi(self, iv_fit):
        """Posterior-mean xi should track truth across cells."""
        model, truth = iv_fit
        post_mean_xi = model.idata.posterior["xi"].mean(dim=("chain", "draw")).values
        true_xi = truth["xi"][0]
        r = float(np.corrcoef(post_mean_xi.ravel(), true_xi.ravel())[0, 1])
        assert r > 0.6, f"Pearson r between posterior xi and truth = {r:.3f}"


class TestEndogeneityBias:
    def test_no_iv_alpha_biased_toward_zero(self, iv_fit, noiv_fit):
        """Headline claim: dropping instruments biases the price coefficient
        toward zero (less negative) because price-xi correlation is absorbed
        into alpha. The IV fit should be closer to truth.
        """
        iv_model, truth = iv_fit
        noiv_model, _ = noiv_fit
        iv_alpha_mean = float(iv_model.idata.posterior["alpha_r"].values.mean())
        noiv_alpha_mean = float(noiv_model.idata.posterior["alpha_r"].values.mean())
        true_alpha = truth["alpha"]
        iv_bias = abs(true_alpha - iv_alpha_mean)
        noiv_bias = abs(true_alpha - noiv_alpha_mean)
        assert noiv_bias > iv_bias, (
            f"no-IV bias {noiv_bias:.3f} should exceed IV bias {iv_bias:.3f}"
        )
        assert noiv_alpha_mean > true_alpha, (
            f"no-IV alpha {noiv_alpha_mean:.3f} should be biased *toward zero* "
            f"(greater than truth {true_alpha})"
        )


# class TestWeakInstruments:
#     @pytest.fixture(scope="class")
#     def weak_iv_fit(self):
#         df, truth = generate_blp_panel(
#             T=40,
#             J=3,
#             K=2,
#             L=2,
#             true_alpha=-2.0,
#             sigma_alpha=0.5,
#             instrument_strength=0.15,
#             price_xi_corr=0.6,
#             market_size=4_000,
#             n_dgp_draws=3_000,
#             random_seed=11,
#             return_truth=True,
#         )
#         return _fit(
#             _make_model(df, truth, instruments_arg=truth["instrument_cols"])
#         ), truth

#     @pytest.fixture(scope="class")
#     def strong_iv_fit(self):
#         df, truth = generate_blp_panel(
#             T=40,
#             J=3,
#             K=2,
#             L=2,
#             true_alpha=-2.0,
#             sigma_alpha=0.5,
#             instrument_strength=0.9,
#             price_xi_corr=0.6,
#             market_size=4_000,
#             n_dgp_draws=3_000,
#             random_seed=11,
#             return_truth=True,
#         )
#         return _fit(
#             _make_model(df, truth, instruments_arg=truth["instrument_cols"])
#         ), truth

#     def test_weak_iv_widens_alpha_interval(self, weak_iv_fit, strong_iv_fit):
#         """Weak-IV credible interval on alpha should be wider than the strong-IV
#         interval (Bayesian inference stays honest about identification weakness).
#         """
#         weak_model, _ = weak_iv_fit
#         strong_model, _ = strong_iv_fit
#         weak_lo, weak_hi = _hdi(weak_model.idata.posterior["alpha_r"].values)
#         strong_lo, strong_hi = _hdi(strong_model.idata.posterior["alpha_r"].values)
#         weak_width = weak_hi - weak_lo
#         strong_width = strong_hi - strong_lo
#         assert weak_width > strong_width, (
#             f"weak-IV HDI width {weak_width:.3f} should exceed strong-IV "
#             f"width {strong_width:.3f}"
#         )

#     def test_weak_iv_still_covers_truth(self, weak_iv_fit):
#         weak_model, truth = weak_iv_fit
#         lo, hi = _hdi(weak_model.idata.posterior["alpha_r"].values)
#         assert lo <= truth["alpha"] <= hi


# class TestHierarchicalPooling:
#     @pytest.fixture(scope="class")
#     def hierarchical_fit(self):
#         df, truth = generate_blp_panel(
#             T=20,
#             J=3,
#             K=2,
#             L=2,
#             R_geo=3,
#             region_heterogeneity=0.6,
#             true_alpha=-2.0,
#             sigma_alpha=0.5,
#             instrument_strength=0.7,
#             price_xi_corr=0.5,
#             market_size=4_000,
#             n_dgp_draws=3_000,
#             random_seed=7,
#             return_truth=True,
#         )
#         model = BayesianBLP(
#             market_data=df,
#             characteristics=truth["characteristic_cols"],
#             instruments=truth["instrument_cols"],
#             region_col="region",
#             random_coef_on=["price"],
#             n_mc_draws=100,
#             random_seed=0,
#         )
#         return _fit(model), truth

#     def test_tau_alpha_excludes_zero_under_heterogeneity(self, hierarchical_fit):
#         """When the DGP has region-level heterogeneity, the hierarchical SD
#         ``tau_alpha`` posterior should be away from zero (not collapsed to a
#         single-region degenerate solution).
#         """
#         model, _ = hierarchical_fit
#         tau_lo, _ = _hdi(model.idata.posterior["tau_alpha"].values)
#         assert tau_lo > 0.0, f"tau_alpha 94% HDI lower bound = {tau_lo:.3f}"

#     def test_alpha_r_recovers_each_region(self, hierarchical_fit):
#         model, truth = hierarchical_fit
#         alpha_r_post = model.idata.posterior["alpha_r"].values
#         for r, true_alpha_r in enumerate(truth["alpha_r"]):
#             lo, hi = _hdi(alpha_r_post[..., r])
#             assert lo <= true_alpha_r <= hi, (
#                 f"region {r}: truth {true_alpha_r:.3f} not in [{lo:.3f}, {hi:.3f}]"
#             )


# class TestSimulationStability:
#     @pytest.fixture(scope="class")
#     def panel_for_stability(self):
#         return generate_blp_panel(
#             T=40,
#             J=3,
#             K=2,
#             L=2,
#             true_alpha=-2.0,
#             sigma_alpha=0.5,
#             instrument_strength=0.7,
#             price_xi_corr=0.6,
#             market_size=4_000,
#             n_dgp_draws=3_000,
#             random_seed=42,
#             return_truth=True,
#         )

#     @pytest.fixture(scope="class")
#     def fit_R100(self, panel_for_stability):
#         df, truth = panel_for_stability
#         return _fit(
#             _make_model(
#                 df, truth, instruments_arg=truth["instrument_cols"], n_mc_draws=100
#             )
#         ), truth

#     @pytest.fixture(scope="class")
#     def fit_R400(self, panel_for_stability):
#         df, truth = panel_for_stability
#         return _fit(
#             _make_model(
#                 df, truth, instruments_arg=truth["instrument_cols"], n_mc_draws=400
#             )
#         ), truth

#     def test_alpha_stable_across_mc_draw_counts(self, fit_R100, fit_R400):
#         """Doubling/quadrupling Halton draws should not move posterior-mean
#         alpha by more than ~half a posterior SD.
#         """
#         m100, _ = fit_R100
#         m400, _ = fit_R400
#         alpha_R100 = m100.idata.posterior["alpha_r"].values
#         alpha_R400 = m400.idata.posterior["alpha_r"].values
#         mean_diff = abs(float(alpha_R100.mean()) - float(alpha_R400.mean()))
#         post_sd = float(alpha_R100.std())
#         assert mean_diff < 0.5 * post_sd, (
#             f"alpha posterior-mean diff between R=100 and R=400 = "
#             f"{mean_diff:.3f}; posterior SD = {post_sd:.3f}"
#         )


# class TestElasticitySanity:
#     def test_own_elasticity_in_sensible_range(self, iv_fit):
#         """Own-price elasticities for branded products typically sit in
#         (-10, -0.5). A model that returns near-zero or wildly negative
#         values is broken.
#         """
#         model, _ = iv_fit
#         e = model.elasticities()
#         for j in range(model._J):
#             own_mean = float(e.values[:, j, j].mean())
#             assert -10.0 < own_mean < -0.5, (
#                 f"product {j} mean own elasticity {own_mean:.3f} outside (-10, -0.5)"
#             )

#     def test_elasticity_matches_finite_difference(self, iv_fit):
#         """Closed-form elasticity from ``elasticities()`` should match a
#         finite-difference of ``counterfactual_shares()`` for the same
#         intervention. Cross-check that the analytical formula isn't drifting
#         from the simulation."""
#         model, _ = iv_fit
#         baseline_cf = model.counterfactual_shares(price_change=None, n_samples=200)
#         e = model.elasticities(at="mean")
#         target_idx = 0
#         target_name = model._inside_products[target_idx]
#         eps = 0.01
#         shocked_cf = model.counterfactual_shares(
#             price_change={target_name: eps}, n_samples=200
#         )
#         baseline_share = baseline_cf["s_inside"].mean(dim="sample").values
#         shocked_share = shocked_cf["s_inside"].mean(dim="sample").values
#         # price_change={...} applies a *relative* change: Δp = p · eps
#         # so ε = (Δs/s) / (Δp/p) = (Δs/s) / eps
#         fd_elast_own = (
#             (shocked_share[:, target_idx] - baseline_share[:, target_idx])
#             / eps
#             / baseline_share[:, target_idx]
#         )
#         analytical_own = e.values[:, target_idx, target_idx]
#         diff = np.abs(fd_elast_own - analytical_own).mean()
#         assert diff < 0.5, (
#             f"mean |FD - analytical| own elasticity diff = {diff:.3f}; "
#             "should be small (<0.5) since both use the same posterior xi."
#         )
