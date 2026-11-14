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
"""Unit tests for PSIS reweighting (no MCMC)."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from arviz_stats.base import array_stats
from scipy import stats

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.reweighting import (
    PSISReweighting,
    psis_reweight,
    relative_efficiency,
)


@pytest.fixture(scope="module")
def theta() -> np.ndarray:
    """Draws from a standard normal reference posterior."""
    return np.random.default_rng(0).normal(size=8_000)


def log_ratio_to(theta: np.ndarray, loc: float = 0.0, scale: float = 1.0):
    """Log ratio of Normal(loc, scale) to the Normal(0, 1) the draws come from."""
    return stats.norm(loc, scale).logpdf(theta) - stats.norm(0, 1).logpdf(theta)


def test_mild_shift_recovers_the_target_mean(theta):
    result = psis_reweight(log_ratio_to(theta, loc=0.5))

    assert isinstance(result, PSISReweighting)
    assert result.weights @ theta == pytest.approx(0.5, abs=0.05)
    assert result.pareto_k < 0.5
    assert result.verdict == "use it"
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.ess == pytest.approx(result.ess_fraction * theta.size)


def test_sign_convention(theta):
    """A log ratio of the wrong sign moves the mean to the wrong side."""
    result = psis_reweight(-log_ratio_to(theta, loc=0.5))

    assert result.weights @ theta < 0


def test_pareto_k_grows_with_the_distance(theta):
    near = psis_reweight(log_ratio_to(theta, loc=0.5))
    far = psis_reweight(log_ratio_to(theta, loc=2.0))

    assert far.pareto_k > near.pareto_k
    assert far.ess_fraction < near.ess_fraction


@pytest.mark.parametrize(
    "pareto_k, verdict",
    [(0.3, "use it"), (0.6, "check"), (0.8, "refit"), (1.2, "refit")],
)
def test_verdict_follows_the_pareto_k_ranges(monkeypatch, theta, pareto_k, verdict):
    def fake_psislw(x, r_eff=1.0):
        return np.zeros_like(x), pareto_k

    monkeypatch.setattr(array_stats, "psislw", fake_psislw)

    assert psis_reweight(log_ratio_to(theta, loc=0.5)).verdict == verdict


def test_non_finite_pareto_k_is_a_refit(monkeypatch, theta):
    monkeypatch.setattr(
        array_stats, "psislw", lambda x, r_eff=1.0: (np.zeros_like(x), np.nan)
    )

    assert psis_reweight(log_ratio_to(theta, loc=0.5)).verdict == "refit"


def test_r_eff_is_passed_to_psis(monkeypatch, theta):
    seen = {}

    def fake_psislw(x, r_eff=1.0):
        seen["r_eff"] = r_eff
        return np.zeros_like(x), 0.1

    monkeypatch.setattr(array_stats, "psislw", fake_psislw)
    psis_reweight(log_ratio_to(theta, loc=0.5), r_eff=0.4)

    assert seen["r_eff"] == 0.4


def test_ess_is_scaled_by_r_eff():
    """With uniform weights the ESS is r_eff times the number of draws."""
    result = psis_reweight(np.zeros(1_000), r_eff=0.6)

    assert result.ess == pytest.approx(600.0)
    assert result.ess_fraction == pytest.approx(0.6)


def test_matches_arviz_psislw(theta):
    """The weights are ArviZ's smoothed weights of exp(log ratio), normalized."""
    log_ratio = log_ratio_to(theta, loc=0.5)

    log_w, k = array_stats.psislw(-log_ratio, r_eff=0.8)
    expected = np.exp(log_w - log_w.max())
    expected /= expected.sum()
    result = psis_reweight(log_ratio, r_eff=0.8)

    np.testing.assert_allclose(result.weights, expected, rtol=1e-12)
    assert result.pareto_k == pytest.approx(float(k))


def test_accepts_chain_draw_arrays(theta):
    flat = log_ratio_to(theta, loc=0.5)

    result = psis_reweight(flat.reshape(4, -1))

    np.testing.assert_allclose(result.weights, psis_reweight(flat).weights)


def test_identical_model_has_uniform_weights():
    result = psis_reweight(np.zeros(100))

    np.testing.assert_allclose(result.weights, np.full(100, 0.01))
    assert np.isnan(result.pareto_k)
    assert result.verdict == "use it"
    assert result.ess_fraction == pytest.approx(1.0)


def test_truncation_keeps_the_allowed_draws(theta):
    allowed = theta > 0
    result = psis_reweight(np.where(allowed, 0.0, -np.inf))

    assert np.all(result.weights[~allowed] == 0)
    np.testing.assert_allclose(result.weights[allowed], 1 / allowed.sum())


def test_zero_density_draws_get_zero_weight(theta):
    log_ratio = log_ratio_to(theta, loc=0.5)
    log_ratio[:50] = -np.inf

    result = psis_reweight(log_ratio)

    assert np.all(result.weights[:50] == 0)
    assert result.weights.sum() == pytest.approx(1.0)
    assert np.isfinite(result.pareto_k)


def test_truncation_that_keeps_few_draws_says_refit(theta):
    """k-hat is nan for a pure truncation, so only the ESS floor can flag it."""
    allowed = theta > 3.0  # about 11 of 8,000 draws

    result = psis_reweight(np.where(allowed, 0.0, -np.inf))

    assert np.isnan(result.pareto_k)
    assert result.ess == pytest.approx(allowed.sum())
    assert result.verdict == "refit"


def test_low_ess_overrides_a_good_pareto_k(monkeypatch, theta):
    def concentrated_psislw(x, r_eff=1.0):
        log_w = np.full_like(x, -50.0)
        log_w[:20] = 0.0  # 20 draws carry all the weight
        return log_w, 0.1

    monkeypatch.setattr(array_stats, "psislw", concentrated_psislw)

    result = psis_reweight(log_ratio_to(theta, loc=0.5))

    assert result.pareto_k == pytest.approx(0.1)
    assert result.ess == pytest.approx(20.0)
    assert result.verdict == "refit"


@pytest.mark.parametrize("min_ess, verdict", [(10.0, "use it"), (12_000.0, "refit")])
def test_min_ess_sets_the_floor(theta, min_ess, verdict):
    result = psis_reweight(log_ratio_to(theta, loc=0.5), min_ess=min_ess)

    assert result.pareto_k < 0.5
    assert result.verdict == verdict


@pytest.mark.parametrize(
    "log_ratio, match",
    [
        ([0.0], "at least 2 draws"),
        ([0.0, 1.0, np.nan], "NaN or"),
        ([0.0, 1.0, np.inf], "NaN or"),
        ([-np.inf, -np.inf], "zero density on every draw"),
    ],
)
def test_invalid_log_ratio_raises(log_ratio, match):
    with pytest.raises(ValueError, match=match):
        psis_reweight(log_ratio)


@pytest.fixture(scope="module")
def posterior() -> xr.Dataset:
    rng = np.random.default_rng(1)
    return xr.Dataset(
        {
            "a": (("chain", "draw"), rng.normal(size=(4, 500))),
            "b": (("chain", "draw", "channel"), rng.normal(size=(4, 500, 2))),
        },
        coords={"chain": range(4), "draw": range(500), "channel": ["x1", "x2"]},
    )


def test_relative_efficiency_of_independent_draws(posterior):
    value = relative_efficiency(posterior)

    assert 0.5 < value <= 1.0


def test_relative_efficiency_penalizes_autocorrelation(posterior):
    rng = np.random.default_rng(2)
    sticky = np.cumsum(rng.normal(size=(4, 500)), axis=1)
    slow = xr.Dataset({"a": (("chain", "draw"), sticky)}, coords=posterior.coords)

    assert relative_efficiency(slow) < 0.2 < relative_efficiency(posterior[["a"]])


def test_relative_efficiency_selects_variables_and_accepts_datatree(posterior):
    tree = xr.DataTree.from_dict({"posterior": posterior})

    from_tree = relative_efficiency(tree["posterior"], var_names=["a"])

    assert from_tree == pytest.approx(relative_efficiency(posterior[["a"]]))


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    return float(np.interp(q, np.cumsum(w[order]) - w[order] / 2, x[order]))


class TestConjugateNormal:
    """Normal data with known noise and a Normal prior: every posterior is exact.

    Draw from the posterior under the reference prior, reweight to another
    prior, and compare with the exact posterior under that prior.
    """

    NOISE_SD = 2.0
    N_DATA = 25
    N_DRAWS = 20_000

    @staticmethod
    def posterior(prior_mean, prior_sd, data_mean, noise_sd, n):
        precision = 1 / prior_sd**2 + n / noise_sd**2
        mean = (prior_mean / prior_sd**2 + n * data_mean / noise_sd**2) / precision
        return stats.norm(mean, precision**-0.5)

    @pytest.fixture(scope="class")
    @classmethod
    def reference(cls):
        rng = np.random.default_rng(7)
        data = rng.normal(1.0, cls.NOISE_SD, size=cls.N_DATA)
        prior = stats.norm(0.0, 3.0)
        exact = cls.posterior(0.0, 3.0, data.mean(), cls.NOISE_SD, cls.N_DATA)
        theta = exact.rvs(cls.N_DRAWS, random_state=rng)
        return data, prior, theta

    def reweight(self, reference, new_mean, new_sd):
        data, prior, theta = reference
        result = psis_reweight(
            stats.norm(new_mean, new_sd).logpdf(theta) - prior.logpdf(theta)
        )
        exact = self.posterior(
            new_mean, new_sd, data.mean(), self.NOISE_SD, self.N_DATA
        )
        return result, exact, theta

    @pytest.mark.parametrize("new_mean, new_sd", [(1.0, 3.0), (2.0, 1.5), (-1.0, 4.0)])
    def test_matches_the_exact_posterior(self, reference, new_mean, new_sd):
        result, exact, theta = self.reweight(reference, new_mean, new_sd)

        assert result.verdict == "use it"
        mean = result.weights @ theta
        sd = np.sqrt(result.weights @ (theta - mean) ** 2)
        assert mean == pytest.approx(exact.mean(), abs=0.05 * exact.std())
        assert sd == pytest.approx(exact.std(), rel=0.05)
        for q in (0.05, 0.5, 0.95):
            assert weighted_quantile(theta, result.weights, q) == pytest.approx(
                exact.ppf(q), abs=0.1 * exact.std()
            )

    def test_a_prior_the_draws_barely_visit_says_refit(self, reference):
        result, _, _ = self.reweight(reference, 8.0, 0.2)

        assert result.verdict == "refit"
        assert result.ess_fraction < 0.05

    def test_k_hat_flags_the_unreliable_case_before_the_error_shows(self, reference):
        """Fewer draws near the new posterior means a larger k-hat and a larger error."""
        errors, ks = [], []
        for new_mean in (1.0, 2.5, 5.0):
            result, exact, theta = self.reweight(reference, new_mean, 1.0)
            errors.append(abs(result.weights @ theta - exact.mean()) / exact.std())
            ks.append(result.pareto_k)

        assert ks == sorted(ks)
        assert errors[0] < 0.1


class TestStoredNodesRebuildTheLogDensities:
    """The notebooks rebuild log densities from nodes stored in the idata.

    ``log p_new - log p_ref`` is then plain NumPy, with no refit. That works
    only while PyMC's own log densities equal what the notebooks recompute
    from ``mu``, ``y_sigma``, the lift estimate and the ROAS. These tests pin
    that. The identities hold for any draws, so mocked sampling is enough.
    """

    CHANNELS = ["x1", "x2"]
    ROAS_TARGET = [3.0, 2.0]
    ROAS_SIGMA = [1.0, 1.0]
    LIFT_SIGMA = 0.5

    @pytest.fixture(scope="class")
    @classmethod
    def mmm(cls, mock_pymc_sample):
        rng = np.random.default_rng(0)
        n = 40
        X = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-02", periods=n, freq="W-MON"),
                "x1": rng.uniform(1, 10, n),
                "x2": rng.uniform(1, 10, n),
            }
        )
        y = pd.Series(50 + 5 * X["x1"] + rng.normal(0, 2, n), name="y")
        mmm = MMM(
            date_column="date",
            channel_columns=cls.CHANNELS,
            target_column="y",
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        mmm.build_model(X, y)
        mmm.add_lift_test_measurements(
            pd.DataFrame(
                {
                    "channel": ["x1", "x1"],
                    "x": [0.25, 0.8],
                    "delta_x": [0.25, 0.8],
                    "delta_y": [1.0, 3.0],
                    "sigma": [cls.LIFT_SIGMA] * 2,
                    "date": X["date"].iloc[[10, 30]].to_numpy(),
                }
            )
        )
        mmm.add_original_scale_contribution_variable(var=["channel_contribution"])
        mmm.add_cost_per_target_calibration(
            data=X,
            calibration_data=pd.DataFrame(
                {
                    "channel": cls.CHANNELS,
                    "roas": cls.ROAS_TARGET,
                    "sigma": cls.ROAS_SIGMA,
                }
            ),
            name_prefix="business_prior",
            target_column="roas",
            target_per_cost=True,
        )
        mmm.fit(X, y, draws=40, chains=2, random_seed=1, nuts_sampler="pymc")
        with mmm.model:
            pm.compute_log_likelihood(mmm.idata)
        return mmm

    @staticmethod
    def draws(mmm, name):
        """Posterior draws of ``name`` with chains and draws merged."""
        return (
            mmm.idata["posterior"][name]
            .stack(s=("chain", "draw"))
            .transpose("s", ...)
            .to_numpy()
        )

    @staticmethod
    def pymc_log_likelihood(mmm, name):
        return (
            mmm.idata["log_likelihood"][name]
            .stack(s=("chain", "draw"))
            .transpose("s", ...)
            .to_numpy()
        )

    def test_sales_likelihood(self, mmm):
        sales = mmm.idata["observed_data"]["y"].to_numpy()
        mu = self.draws(mmm, "mu")
        y_sigma = self.draws(mmm, "y_sigma")[:, None]

        rebuilt = stats.norm.logpdf(sales, mu, y_sigma)

        np.testing.assert_allclose(
            rebuilt, self.pymc_log_likelihood(mmm, "y"), rtol=1e-9
        )

    def test_lift_likelihood_lives_on_the_scaled_target(self, mmm):
        scale = float(mmm.idata["constant_data"]["target_scale"])
        observed = mmm.idata["observed_data"]["lift_measurements"].to_numpy()
        estimate = self.draws(mmm, "lift_measurements_model_estimated_lift")

        rebuilt = stats.norm.logpdf(observed, estimate, self.LIFT_SIGMA / scale)

        np.testing.assert_allclose(
            rebuilt, self.pymc_log_likelihood(mmm, "lift_measurements"), rtol=1e-9
        )

    def test_business_prior_is_a_normal_on_the_roas(self, mmm):
        roas = mmm.incrementality.compute_incremental_contribution(
            "all_time"
        ) / mmm.idata["constant_data"]["channel_data"].sum("date")
        roas = roas.stack(s=("chain", "draw")).transpose("s", ...).to_numpy()

        rebuilt = stats.norm.logpdf(self.ROAS_TARGET, roas, self.ROAS_SIGMA)

        np.testing.assert_allclose(
            rebuilt, self.pymc_log_likelihood(mmm, "business_prior"), rtol=1e-6
        )


class TestScipyMatchesPyMCParametrization:
    """The notebooks write the changed densities with SciPy; PyMC refits use ``Prior``."""

    X = np.array([0.3, 1.0, 2.5])

    @staticmethod
    def pymc_logp(dist, x):
        return pm.logp(dist, x).eval()

    def test_student_t(self):
        nu, mu, sigma = 10.0, 0.5, 1.7

        rebuilt = stats.t.logpdf(self.X, nu, mu, sigma)

        np.testing.assert_allclose(
            rebuilt, self.pymc_logp(pm.StudentT.dist(nu=nu, mu=mu, sigma=sigma), self.X)
        )

    def test_half_normal(self):
        sigma = 0.7

        rebuilt = stats.halfnorm(scale=sigma).logpdf(self.X)

        np.testing.assert_allclose(
            rebuilt, self.pymc_logp(pm.HalfNormal.dist(sigma=sigma), self.X)
        )

    def test_beta(self):
        x = np.array([0.1, 0.4, 0.8])

        rebuilt = stats.beta(3.0, 2.0).logpdf(x)

        # PyMC evaluates the Beta normalizer in single-precision-like arithmetic.
        np.testing.assert_allclose(
            rebuilt, self.pymc_logp(pm.Beta.dist(alpha=3.0, beta=2.0), x), rtol=1e-6
        )

    def test_normal(self):
        rebuilt = stats.norm.logpdf(self.X, 0.5, 1.7)

        np.testing.assert_allclose(
            rebuilt, self.pymc_logp(pm.Normal.dist(mu=0.5, sigma=1.7), self.X)
        )
