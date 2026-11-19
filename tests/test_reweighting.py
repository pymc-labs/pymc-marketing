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

import arviz as az
import numpy as np
import pytest
import xarray as xr
from arviz_stats.base import array_stats
from scipy import stats

from pymc_marketing.reweighting import (
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
    [
        (0.3, "use it"),
        (0.5, "check"),
        (0.6, "check"),
        (0.7, "refit"),
        (0.8, "refit"),
        (1.2, "refit"),
    ],
)
def test_verdict_follows_the_pareto_k_ranges(monkeypatch, theta, pareto_k, verdict):
    def fake_psislw(x, r_eff=1.0):
        return np.zeros_like(x), pareto_k

    monkeypatch.setattr(array_stats, "psislw", fake_psislw)

    result = psis_reweight(log_ratio_to(theta, loc=0.5))

    assert result.pareto_k_threshold == pytest.approx(0.7)
    assert result.verdict == verdict


@pytest.mark.parametrize("n_draws", [400, 1_000])
def test_refit_threshold_shrinks_with_the_number_of_draws(monkeypatch, n_draws):
    """Below about 2,150 draws the bound min(1 - 1/log10(S), 0.7) is under 0.7."""
    threshold = 1 - 1 / np.log10(n_draws)
    monkeypatch.setattr(
        array_stats, "psislw", lambda x, r_eff=1.0: (np.zeros_like(x), threshold)
    )
    theta = np.random.default_rng(3).normal(size=n_draws)

    result = psis_reweight(log_ratio_to(theta, loc=0.5))

    assert result.pareto_k_threshold == pytest.approx(threshold)
    assert threshold < 0.7
    assert result.verdict == "refit"


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


def test_accepts_chain_draw_arrays(theta):
    flat = log_ratio_to(theta, loc=0.5)

    result = psis_reweight(flat.reshape(4, -1))

    np.testing.assert_allclose(result.weights, psis_reweight(flat).weights)


def test_identical_model_has_uniform_weights():
    result = psis_reweight(np.zeros(1_000))

    np.testing.assert_allclose(result.weights, np.full(1_000, 0.001))
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


def test_default_floor_is_400_effective_draws():
    """A uniform truncation keeping 300 draws has ESS 300: refit by default."""
    log_ratio = np.r_[np.zeros(300), np.full(2_700, -np.inf)]

    assert psis_reweight(log_ratio).verdict == "refit"
    assert psis_reweight(log_ratio, min_ess=250, min_ess_fraction=0).verdict == (
        "use it"
    )


@pytest.fixture(scope="module")
def tight_log_ratio() -> np.ndarray:
    """A much tighter target: bounded weights (low k-hat) that rest on few draws.

    Like a lift test that counts much more: k-hat stays low while the ESS drops.
    """
    theta = np.random.default_rng(8).normal(size=12_000)
    return stats.norm(0, 0.03).logpdf(theta) - stats.norm(0, 1).logpdf(theta)


def test_low_ess_fraction_turns_use_it_into_check(tight_log_ratio):
    result = psis_reweight(tight_log_ratio)

    assert result.pareto_k < 0.5
    assert result.ess >= 400
    assert result.ess_fraction < 0.05
    assert result.verdict == "check"


def test_min_ess_fraction_zero_keeps_use_it(tight_log_ratio):
    assert psis_reweight(tight_log_ratio, min_ess_fraction=0).verdict == "use it"


def test_low_ess_fraction_does_not_soften_refit(monkeypatch, theta):
    """The fraction rule only downgrades "use it"; a k-hat refit stays refit."""
    monkeypatch.setattr(
        array_stats, "psislw", lambda x, r_eff=1.0: (np.zeros_like(x), 0.9)
    )

    result = psis_reweight(log_ratio_to(theta, loc=0.5), min_ess_fraction=1.0)

    assert result.verdict == "refit"


@pytest.mark.parametrize("n_finite", [3, 15, 24])
def test_too_few_finite_draws_to_fit_a_tail_says_refit(n_finite):
    rng = np.random.default_rng(4)
    log_ratio = np.r_[rng.normal(size=n_finite), np.full(200, -np.inf)]

    result = psis_reweight(log_ratio, min_ess=0)

    assert np.isnan(result.pareto_k)
    assert np.isnan(result.pareto_k_threshold)
    assert result.verdict == "refit"
    assert result.weights.sum() == pytest.approx(1.0)
    assert np.all(result.weights[n_finite:] == 0)


@pytest.mark.parametrize(
    "make_log_ratio",
    [
        lambda theta: np.where(theta > 0.5, np.log(2), 0.0),  # two-valued step
        lambda theta: np.where(theta > 0, 0.0, -5.0),  # soft truncation
        lambda theta: np.minimum(theta, np.quantile(theta, 0.9)),  # clipped
    ],
    ids=["step", "soft-truncation", "clipped"],
)
def test_tied_upper_tail_uses_the_bounded_raw_weights(theta, make_log_ratio):
    """The largest ratios are equal, so there is no tail to fit: the ESS decides."""
    log_ratio = make_log_ratio(theta)

    result = psis_reweight(log_ratio)

    expected = np.exp(log_ratio - log_ratio.max())
    np.testing.assert_allclose(result.weights, expected / expected.sum())
    assert np.isnan(result.pareto_k)
    assert result.verdict == "use it"


def test_rounding_noise_of_a_large_log_ratio_counts_as_constant():
    rng = np.random.default_rng(5)
    log_ratio = 1e8 + np.spacing(1e8) * rng.integers(0, 2, 12_000)

    result = psis_reweight(log_ratio)

    np.testing.assert_allclose(result.weights, 1 / 12_000)
    assert np.isnan(result.pareto_k)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"r_eff": np.nan}, "r_eff"),
        ({"r_eff": np.inf}, "r_eff"),
        ({"r_eff": 0.0}, "r_eff"),
        ({"r_eff": -0.5}, "r_eff"),
        ({"r_eff": 5.0}, "r_eff"),
        ({"min_ess": np.nan}, "min_ess"),
        ({"min_ess": -1.0}, "min_ess"),
        ({"min_ess_fraction": np.nan}, "min_ess_fraction"),
        ({"min_ess_fraction": -0.1}, "min_ess_fraction"),
        ({"min_ess_fraction": 1.5}, "min_ess_fraction"),
    ],
)
def test_invalid_r_eff_or_min_ess_raises(theta, kwargs, match):
    with pytest.raises(ValueError, match=match):
        psis_reweight(log_ratio_to(theta, loc=0.5), **kwargs)


def test_result_compares_by_identity_and_is_hashable(theta):
    log_ratio = log_ratio_to(theta, loc=0.5)
    a, b = psis_reweight(log_ratio), psis_reweight(log_ratio)

    assert a != b
    assert a == a
    assert len({a, b}) == 2


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


def test_relative_efficiency_accepts_a_single_name(posterior):
    assert relative_efficiency(posterior, "a") == relative_efficiency(posterior, ["a"])


@pytest.mark.parametrize(
    "make_input, kwargs",
    [
        (lambda p: p, {"var_names": []}),
        (lambda p: xr.DataTree.from_dict({"posterior": p}), {}),  # whole tree
        (lambda p: p.isel(chain=0), {}),  # no chain dimension
    ],
    ids=["empty-var-names", "whole-tree", "no-chain"],
)
def test_relative_efficiency_rejects_unusable_input(posterior, make_input, kwargs):
    with pytest.raises(ValueError):
        relative_efficiency(make_input(posterior), **kwargs)


def test_relative_efficiency_is_capped_at_one():
    """Anticorrelated draws have an ESS above the number of draws."""
    rng = np.random.default_rng(6)
    x = np.zeros((4, 1_000))
    for t in range(1, 1_000):
        x[:, t] = -0.6 * x[:, t - 1] + rng.normal(size=4)
    anti = xr.Dataset(
        {"a": (("chain", "draw"), x)}, coords={"chain": range(4), "draw": range(1_000)}
    )

    assert float(az.ess(anti, method="mean")["a"]) > 4_000
    assert relative_efficiency(anti) == 1.0


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


@pytest.mark.parametrize("loc, verdict", [(0.2, "use it"), (1.0, "refit")])
def test_short_runs_refit_from_k_hat_one_half(loc, verdict):
    """At 100 draws the tail is n / 5 long and the refit threshold is 0.5.

    The shift to 1.0 has k-hat about 0.58: "check" with many draws, "refit" here.
    """
    theta = np.random.default_rng(7).normal(size=100)

    result = psis_reweight(log_ratio_to(theta, loc=loc), min_ess=0)

    assert result.pareto_k_threshold == pytest.approx(0.5)
    assert result.pareto_k < 0.5 if verdict == "use it" else 0.5 < result.pareto_k < 0.7
    assert result.verdict == verdict


def test_non_finite_psis_weights_raise(monkeypatch, theta):
    monkeypatch.setattr(
        array_stats, "psislw", lambda x, r_eff=1.0: (np.full_like(x, np.nan), 0.1)
    )

    with pytest.raises(ValueError, match="PSIS produced"):
        psis_reweight(log_ratio_to(theta, loc=0.5))
