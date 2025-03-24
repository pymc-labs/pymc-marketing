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
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from lifetimes import BetaGeoBetaBinomFitter as BGBBF
from lifetimes import BetaGeoFitter as BG
from lifetimes import ModifiedBetaGeoFitter as MBG
from lifetimes import ParetoNBDFitter as PF
from lifetimes.generate_data import beta_geometric_beta_binom_model
from numpy.testing import assert_almost_equal
from pymc import Model

from pymc_marketing.clv.distributions import (
    BetaGeoBetaBinom,
    BetaGeoNBD,
    ContContract,
    ContNonContract,
    ModifiedBetaGeoNBD,
    ParetoNBD,
)


class TestContNonContract:
    @pytest.mark.parametrize(
        "value, lam, p, T, logp",
        [
            (np.array([4.3, 5]), 0.4, 0.15, 10, -8.39147106159807),
            (
                np.array([4.3, 5]),
                np.array([0.3, 0.2]),
                0.15,
                10,
                np.array([-9.15153637, -10.42037984]),
            ),
            (
                np.array([[4.3, 5], [3.3, 4]]),
                np.array([0.3, 0.2]),
                0.15,
                10,
                np.array([-9.15153637, -8.57264195]),
            ),
            (
                np.array([4.3, 5]),
                0.3,
                np.full((5, 3), 0.15),
                10,
                np.full(shape=(5, 3), fill_value=-9.15153637),
            ),
        ],
    )
    def test_continuous_non_contractual(self, value, lam, p, T, logp):
        with Model():
            cnc = ContNonContract("cnc", lam=lam, p=p, T=T)
        pt = {"cnc": value}

        assert_almost_equal(
            pm.logp(cnc, value).eval(),
            logp,
            decimal=6,
            err_msg=str(pt),
        )

    def test_continuous_non_contractual_invalid(self):
        cnc = ContNonContract.dist(lam=0.8, p=0.15, T=10)
        assert pm.logp(cnc, np.array([-1, 3])).eval() == -np.inf
        assert pm.logp(cnc, np.array([1.5, -1])).eval() == -np.inf
        assert pm.logp(cnc, np.array([1.5, 0])).eval() == -np.inf
        assert pm.logp(cnc, np.array([11, 3])).eval() == -np.inf

    # TODO: test broadcasting of parameters, including T
    @pytest.mark.parametrize(
        "lam_size, p_size, cnc_size, expected_size",
        [
            (None, None, None, (2,)),
            ((5,), None, None, (5, 2)),
            ((5,), None, (5,), (5, 2)),
            ((5, 1), (1, 3), (5, 3), (5, 3, 2)),
            (None, None, (5, 3), (5, 3, 2)),
        ],
    )
    def test_continuous_non_contractual_sample_prior(
        self, lam_size, p_size, cnc_size, expected_size
    ):
        with Model():
            lam = pm.Gamma(name="lam", alpha=1, beta=1, size=lam_size)
            p = pm.Beta(name="p", alpha=1.0, beta=1.0, size=p_size)
            ContNonContract(name="cnc", lam=lam, p=p, T=10, size=cnc_size)
            prior = pm.sample_prior_predictive(draws=100)

        assert prior["prior"]["cnc"][0].shape == (100,) + expected_size  # noqa: RUF005


class TestContContract:
    @pytest.mark.parametrize(
        "value, lam, p, T, logp",
        [
            (np.array([6.3, 5, 1]), 0.3, 0.15, 10, -10.45705972),
            (
                np.array([6.3, 5, 1]),
                np.array([0.3, 0.2]),
                0.15,
                10,
                np.array([-10.45705972, -11.85438527]),
            ),
            (
                np.array([[6.3, 5, 1], [5.3, 4, 0]]),
                np.array([0.3, 0.2]),
                0.15,
                10,
                np.array([-10.45705972, -9.08782737]),
            ),
            (
                np.array([6.3, 5, 0]),
                0.3,
                np.full((5, 3), 0.15),
                10,
                np.full(shape=(5, 3), fill_value=-9.83245867),
            ),
        ],
    )
    def test_continuous_contractual(self, value, lam, p, T, logp):
        with Model():
            cc = ContContract("cc", lam=lam, p=p, T=T)
        pt = {"cc": value}

        assert_almost_equal(
            pm.logp(cc, value).eval(),
            logp,
            decimal=6,
            err_msg=str(pt),
        )

    def test_continuous_contractual_invalid(self):
        cc = ContContract.dist(lam=0.8, p=0.15, T=10)
        assert pm.logp(cc, np.array([-1, 3, 1])).eval() == -np.inf
        assert pm.logp(cc, np.array([1.5, -1, 1])).eval() == -np.inf
        assert pm.logp(cc, np.array([1.5, 0, 1])).eval() == -np.inf
        assert pm.logp(cc, np.array([11, 3, 1])).eval() == -np.inf
        assert pm.logp(cc, np.array([1.5, 3, 0.5])).eval() == -np.inf
        assert pm.logp(cc, np.array([1.5, 3, -1])).eval() == -np.inf

    # TODO: test broadcasting of parameters, including T
    @pytest.mark.parametrize(
        "lam_size, p_size, cc_size, expected_size",
        [
            (None, None, None, (3,)),
            ((7,), None, None, (7, 3)),
            ((7,), None, (7,), (7, 3)),
            ((7, 1), (1, 5), (7, 5), (7, 5, 3)),
            (None, None, (7, 5), (7, 5, 3)),
        ],
    )
    def test_continuous_contractual_sample_prior(
        self, lam_size, p_size, cc_size, expected_size
    ):
        with Model():
            lam = pm.Gamma(name="lam", alpha=1, beta=1, size=lam_size)
            p = pm.Beta(name="p", alpha=1.0, beta=1.0, size=p_size)
            ContContract(name="cc", lam=lam, p=p, T=10, size=cc_size)
            prior = pm.sample_prior_predictive(draws=100)

        assert prior["prior"]["cc"][0].shape == (100,) + expected_size  # noqa: RUF005


class TestParetoNBD:
    @pytest.mark.parametrize(
        "value, r, alpha, s, beta, T",
        [
            (
                np.array([1.5, 1]),
                0.55,
                10.58,
                0.61,
                11.67,
                12,
            ),
            (
                np.array([1.5, 1]),
                [0.45, 0.55],
                10.58,
                0.61,
                11.67,
                12,
            ),
            (
                np.array([1.5, 1]),
                [0.45, 0.55],
                10.58,
                [0.71, 0.61],
                11.67,
                12,
            ),
            (
                np.array([[1.5, 1], [5.3, 4], [6, 2]]),
                0.55,
                11.67,
                0.61,
                10.58,
                [12, 10, 8],
            ),
            (
                np.array([1.5, 1]),
                0.55,
                10.58,
                0.61,
                np.full((5, 3), 11.67),
                12,
            ),
        ],
    )
    def test_pareto_nbd(self, value, r, alpha, s, beta, T):
        def lifetimes_wrapper(r, alpha, s, beta, freq, rec, T):
            """Simple wrapper for Vectorizing the lifetimes likelihood function."""
            return PF._conditional_log_likelihood((r, alpha, s, beta), freq, rec, T)

        vectorized_logp = np.vectorize(lifetimes_wrapper)

        with Model():
            pareto_nbd = ParetoNBD("pareto_nbd", r=r, alpha=alpha, s=s, beta=beta, T=T)
        pt = {"pareto_nbd": value}

        assert_almost_equal(
            pm.logp(pareto_nbd, value).eval(),
            vectorized_logp(r, alpha, s, beta, value[..., 1], value[..., 0], T),
            decimal=6,
            err_msg=str(pt),
        )

    def test_pareto_nbd_invalid(self):
        pareto_nbd = ParetoNBD.dist(r=0.55, alpha=10.58, s=0.61, beta=11.67, T=10)
        assert pm.logp(pareto_nbd, np.array([3, -1])).eval() == -np.inf
        assert pm.logp(pareto_nbd, np.array([-1, 1.5])).eval() == -np.inf
        assert pm.logp(pareto_nbd, np.array([11, 1.5])).eval() == -np.inf

    @pytest.mark.parametrize(
        "r_size, alpha_size, s_size, beta_size, pareto_nbd_size, expected_size",
        [
            (None, None, None, None, None, (2,)),
            ((5,), None, None, None, None, (5, 2)),
            (None, (5,), None, None, (5,), (5, 2)),
            (None, None, (5, 1), (1, 3), (5, 3), (5, 3, 2)),
            (None, None, None, None, (5, 3), (5, 3, 2)),
        ],
    )
    def test_pareto_nbd_sample_prior(
        self, r_size, alpha_size, s_size, beta_size, pareto_nbd_size, expected_size
    ):
        with Model():
            r = pm.Gamma(name="r", alpha=5, beta=1, size=r_size)
            alpha = pm.Gamma(name="alpha", alpha=5, beta=1, size=alpha_size)
            s = pm.Gamma(name="s", alpha=5, beta=1, size=s_size)
            beta = pm.Gamma(name="beta", alpha=5, beta=1, size=beta_size)

            T = pm.Data(name="T", value=np.array(10))

            ParetoNBD(
                name="pareto_nbd",
                r=r,
                alpha=alpha,
                s=s,
                beta=beta,
                T=T,
                size=pareto_nbd_size,
            )
            prior = pm.sample_prior_predictive(draws=100)

        assert prior["prior"]["pareto_nbd"][0].shape == (100, *expected_size)


class TestBetaGeoBetaBinom:
    @pytest.mark.parametrize("batch_shape", [(), (5,)])
    def test_logp_matches_lifetimes(self, batch_shape):
        rng = np.random.default_rng(269)

        alpha = pm.draw(
            pm.Gamma.dist(mu=1.2, sigma=3, shape=batch_shape), random_seed=rng
        )
        beta = pm.draw(
            pm.Gamma.dist(mu=0.75, sigma=3, shape=batch_shape), random_seed=rng
        )
        gamma = pm.draw(
            pm.Gamma.dist(mu=0.657, sigma=3, shape=(1,) * len(batch_shape)),
            random_seed=rng,
        )
        delta = pm.draw(pm.Gamma.dist(mu=2.783, sigma=3), random_seed=rng)
        T = pm.draw(pm.DiscreteUniform.dist(1, 10, shape=batch_shape), random_seed=rng)

        t_x = pm.draw(pm.DiscreteUniform.dist(0, T, shape=batch_shape), random_seed=rng)
        x = pm.draw(pm.DiscreteUniform.dist(0, t_x, shape=batch_shape), random_seed=rng)
        value = np.concatenate([t_x[..., None], x[..., None]], axis=-1)

        dist = BetaGeoBetaBinom.dist(alpha, beta, gamma, delta, T)
        np.testing.assert_allclose(
            pm.logp(dist, value).eval(),
            BGBBF._loglikelihood((alpha, beta, gamma, delta), x, t_x, T),
        )

    def test_logp_matches_excel(self):
        # Expected logp values can be found in excel file in http://brucehardie.com/notes/010/
        # Spreadsheet: Parameter estimate

        alpha = 1.204
        beta = 0.750
        gamma = 0.657
        delta = 2.783
        T = 6

        x = np.array([6, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 2, 1, 1, 0])
        t_x = np.array(
            [6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 1, 0]
        )
        expected_logp = np.array(
            [
                -2.18167018824,
                -4.29485034929,
                -5.38473334360,
                -5.80915881601,
                -5.65172964525,
                -4.88370164695,
                -3.71682127437,
                -5.09558227343,
                -5.61576884108,
                -5.50636893346,
                -4.76723821904,
                -3.84829625138,
                -5.05936147828,
                -5.19562191019,
                -4.57070931973,
                -3.52745257839,
                -4.51620272962,
                -4.22465969453,
                -3.01199924784,
                -3.58817880928,
                -2.28882847451,
                -1.16751622367,
            ]
        )

        value = np.concatenate([t_x[:, None], x[:, None]], axis=-1)
        dist = BetaGeoBetaBinom.dist(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            T=T,
        )
        np.testing.assert_allclose(
            pm.logp(dist, value).eval(),
            expected_logp,
            rtol=1e-3,
        )

    def test_invalid_value_logp(self):
        beta_geo_beta_binom = BetaGeoBetaBinom.dist(
            alpha=1.20, beta=0.75, gamma=0.66, delta=2.78, T=6
        )
        value = pt.vector("value", shape=(2,))
        logp = pm.logp(beta_geo_beta_binom, value)

        logp_fn = pytensor.function([value], logp)
        assert logp_fn(np.array([3, -1])) == -np.inf
        assert logp_fn(np.array([-1, 1.5])) == -np.inf
        assert logp_fn(np.array([11, 1.5])) == -np.inf

    def test_notimplemented_logp(self):
        dist = BetaGeoBetaBinom.dist(alpha=1, beta=1, gamma=2, delta=2, T=10)
        invalid_value = np.broadcast_to([1, 3], (4, 3, 2))
        with pytest.raises(NotImplementedError):
            pm.logp(dist, invalid_value)

    @pytest.mark.parametrize(
        "alpha_size, beta_size, gamma_size, delta_size, beta_geo_beta_binom_size, expected_size",
        [
            (None, None, None, None, None, (2,)),
            ((5,), None, None, None, None, (5, 2)),
            (None, (5,), None, None, (5,), (5, 2)),
            (None, None, (5, 1), (1, 3), (5, 3), (5, 3, 2)),
            (None, None, None, None, (5, 3), (5, 3, 2)),
        ],
    )
    def test_beta_geo_beta_binom_sample_prior(
        self,
        alpha_size,
        beta_size,
        gamma_size,
        delta_size,
        beta_geo_beta_binom_size,
        expected_size,
    ):
        # Declare simulation params
        T_true = 60
        alpha_true = 1.204
        beta_true = 0.750
        gamma_true = 0.657
        delta_true = 2.783

        # Generate simulated data from lifetimes
        # this does not have a random seed, so rtol must be higher
        lt_bgbb = beta_geometric_beta_binom_model(
            N=T_true,
            alpha=alpha_true,
            beta=beta_true,
            gamma=gamma_true,
            delta=delta_true,
            size=1000,
        )
        lt_frequency = lt_bgbb["frequency"].values
        lt_recency = lt_bgbb["recency"].values

        with Model():
            alpha = pm.Normal(name="alpha", mu=alpha_true, sigma=1e-4, size=alpha_size)
            beta = pm.Normal(name="beta", mu=beta_true, sigma=1e-4, size=beta_size)
            gamma = pm.Normal(name="gamma", mu=gamma_true, sigma=1e-4, size=gamma_size)
            delta = pm.Normal(name="delta", mu=delta_true, sigma=1e-4, size=delta_size)

            T = pm.Data(name="T", value=np.array(T_true))

            BetaGeoBetaBinom(
                name="beta_geo_beta_binom",
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                T=T,
                size=beta_geo_beta_binom_size,
            )
            prior = pm.sample_prior_predictive(draws=1000)
            prior = prior["prior"]["beta_geo_beta_binom"][0]
            recency = prior[:, 0]
            frequency = prior[:, 1]

        assert prior.shape == (1000, *expected_size)

        np.testing.assert_allclose(lt_frequency.mean(), recency.mean(), rtol=0.84)
        np.testing.assert_allclose(lt_recency.mean(), frequency.mean(), rtol=0.84)


class TestBetaGeoNBD:
    @pytest.mark.parametrize(
        "value, r, alpha, a, b, T",
        [
            (
                np.array([1.5, 1]),
                0.55,
                10.58,
                0.61,
                11.67,
                12,
            ),
            (
                np.array([1.5, 1]),
                [0.45, 0.55],
                10.58,
                0.61,
                11.67,
                12,
            ),
            (
                np.array([1.5, 1]),
                [0.45, 0.55],
                10.58,
                [0.71, 0.61],
                11.67,
                12,
            ),
            (
                np.array([[1.5, 1], [5.3, 4], [6, 2]]),
                0.55,
                11.67,
                0.61,
                10.58,
                [12, 10, 8],
            ),
            (
                np.array([1.5, 1]),
                0.55,
                10.58,
                0.61,
                np.full((1), 11.67),
                12,
            ),
        ],
    )
    def test_bg_nbd(self, value, r, alpha, a, b, T):
        def lifetimes_wrapper(
            r, alpha, a, b, freq, rec, T, weights=np.array(1), penalizer_coef=0.0
        ):
            log_r = np.log(r)
            log_alpha = np.log(alpha)
            log_a = np.log(a)
            log_b = np.log(b)

            """Simple wrapper for Vectorizing the lifetimes likelihood function.
            Lifetimes uses the negative log likelihood, so we need to negate it to match PyMC3's logp.
            """
            return -1.0 * BG._negative_log_likelihood(
                (log_r, log_alpha, log_a, log_b), freq, rec, T, weights, penalizer_coef
            )

        vectorized_logp = np.vectorize(lifetimes_wrapper)

        with Model():
            bg_nbd = BetaGeoNBD("bg_nbd", a=a, b=b, r=r, alpha=alpha, T=T)
        pt = {"bg_nbd": value}

        assert_almost_equal(
            pm.logp(bg_nbd, value).eval(),
            vectorized_logp(r, alpha, a, b, value[..., 1], value[..., 0], T),
            decimal=6,
            err_msg=str(pt),
        )

    def test_logp_matches_excel(self):
        # Expected logp values can be found in excel file in http://brucehardie.com/notes/004/
        # Spreadsheet: BGNBD Estimation
        a = 0.793
        b = 2.426
        r = 0.243
        alpha = 4.414
        T = 38.86

        x = np.array([2, 1, 0, 0, 0, 7, 1, 0, 2, 0, 5, 0, 0, 0, 0, 0, 10, 1])
        t_x = np.array(
            [
                30.43,
                1.71,
                0.00,
                0.00,
                0.00,
                29.43,
                5.00,
                0.00,
                35.71,
                0.00,
                24.43,
                0.00,
                0.00,
                0.00,
                0.00,
                0.00,
                34.14,
                4.86,
            ]
        )
        expected_logp = np.array(
            [
                -9.4596,
                -4.4711,
                -0.5538,
                -0.5538,
                -0.5538,
                -21.8644,
                -4.8651,
                -0.5538,
                -9.5367,
                -0.5538,
                -17.3593,
                -0.5538,
                -0.5538,
                -0.5538,
                -0.5538,
                -0.5538,
                -27.3144,
                -4.8520,
            ]
        )

        value = np.concatenate([t_x[:, None], x[:, None]], axis=-1)
        dist = BetaGeoNBD.dist(
            a=a,
            b=b,
            r=r,
            alpha=alpha,
            T=T,
        )
        np.testing.assert_allclose(
            pm.logp(dist, value).eval(),
            expected_logp,
            rtol=2e-3,
        )

    def test_invalid_value_logp(self):
        bg_nbd = BetaGeoNBD.dist(a=1.20, b=0.75, r=0.66, alpha=2.78, T=6)
        value = pt.vector("value", shape=(2,))
        logp = pm.logp(bg_nbd, value)

        logp_fn = pytensor.function([value], logp)
        assert logp_fn(np.array([3, -1])) == -np.inf
        assert logp_fn(np.array([-1, 1.5])) == -np.inf
        assert logp_fn(np.array([11, 1.5])) == -np.inf

    def test_notimplemented_logp(self):
        dist = BetaGeoNBD.dist(a=1, b=1, r=2, alpha=2, T=10)
        invalid_value = np.broadcast_to([1, 3], (4, 3, 2))

        with pytest.raises(NotImplementedError):
            pm.logp(dist, invalid_value)

    @pytest.mark.parametrize(
        "a_size, b_size, r_size, alpha_size, bg_nbd_size, expected_size",
        [
            (None, None, None, None, None, (2,)),
            ((5,), None, None, None, None, (5, 2)),
            (None, (5,), None, None, (5,), (5, 2)),
            (None, None, (5, 1), (1, 3), (5, 3), (5, 3, 2)),
            (None, None, None, None, (5, 3), (5, 3, 2)),
        ],
    )
    def test_bg_nbd_sample_prior(
        self, a_size, b_size, r_size, alpha_size, bg_nbd_size, expected_size
    ):
        with Model():
            a = pm.HalfNormal(name="a", sigma=10, size=a_size)
            b = pm.HalfNormal(name="b", sigma=10, size=b_size)
            r = pm.HalfNormal(name="r", sigma=10, size=r_size)
            alpha = pm.HalfNormal(name="alpha", sigma=10, size=alpha_size)

            T = pm.Data(name="T", value=np.array(10))

            BetaGeoNBD(
                name="bg_nbd",
                a=a,
                b=b,
                r=r,
                alpha=alpha,
                T=T,
                size=bg_nbd_size,
            )
            prior = pm.sample_prior_predictive(draws=100)

        assert prior["prior"]["bg_nbd"][0].shape == (100, *expected_size)


class TestModifiedBetaGeoNBD:
    @pytest.mark.parametrize(
        "value, r, alpha, a, b, T",
        [
            (
                np.array([1.5, 1]),
                0.55,
                10.58,
                0.61,
                11.67,
                12,
            ),
            (
                np.array([1.5, 1]),
                [0.45, 0.55],
                10.58,
                0.61,
                11.67,
                12,
            ),
            (
                np.array([1.5, 1]),
                [0.45, 0.55],
                10.58,
                [0.71, 0.61],
                11.67,
                12,
            ),
            (
                np.array([[1.5, 1], [5.3, 4], [6, 2]]),
                0.55,
                11.67,
                0.61,
                10.58,
                [12, 10, 8],
            ),
            (
                np.array([1.5, 1]),
                0.55,
                10.58,
                0.61,
                np.full((1), 11.67),
                12,
            ),
        ],
    )
    def test_mbg_nbd(self, value, r, alpha, a, b, T):
        def lifetimes_wrapper(
            r, alpha, a, b, freq, rec, T, weights=np.array(1), penalizer_coef=0.0
        ):
            log_r = np.log(r)
            log_alpha = np.log(alpha)
            log_a = np.log(a)
            log_b = np.log(b)

            """Simple wrapper for Vectorizing the lifetimes likelihood function.
            Lifetimes uses the negative log likelihood, so we need to negate it to match PyMC3's logp.
            """
            return -1.0 * MBG._negative_log_likelihood(
                (log_r, log_alpha, log_a, log_b), freq, rec, T, weights, penalizer_coef
            )

        vectorized_logp = np.vectorize(lifetimes_wrapper)

        with Model():
            mbg_nbd = ModifiedBetaGeoNBD("mbg_nbd", a=a, b=b, r=r, alpha=alpha, T=T)
        pt = {"mbg_nbd": value}

        assert_almost_equal(
            pm.logp(mbg_nbd, value).eval(),
            vectorized_logp(r, alpha, a, b, value[..., 1], value[..., 0], T),
            decimal=6,
            err_msg=str(pt),
        )

    @pytest.mark.parametrize(
        "a_size, b_size, r_size, alpha_size, mbg_nbd_size, expected_size",
        [
            (None, None, None, None, None, (2,)),
            ((5,), None, None, None, None, (5, 2)),
            (None, (5,), None, None, (5,), (5, 2)),
            (None, None, (5, 1), (1, 3), (5, 3), (5, 3, 2)),
            (None, None, None, None, (5, 3), (5, 3, 2)),
        ],
    )
    def test_mbg_nbd_sample_prior(
        self, a_size, b_size, r_size, alpha_size, mbg_nbd_size, expected_size
    ):
        with Model():
            a = pm.HalfNormal(name="a", sigma=10, size=a_size)
            b = pm.HalfNormal(name="b", sigma=10, size=b_size)
            r = pm.HalfNormal(name="r", sigma=10, size=r_size)
            alpha = pm.HalfNormal(name="alpha", sigma=10, size=alpha_size)

            T = pm.Data(name="T", value=np.array(10))

            ModifiedBetaGeoNBD(
                name="mbg_nbd",
                a=a,
                b=b,
                r=r,
                alpha=alpha,
                T=T,
                size=mbg_nbd_size,
            )
            prior = pm.sample_prior_predictive(draws=100)

        assert prior["prior"]["mbg_nbd"][0].shape == (100, *expected_size)

    def test_invalid_value_logp(self):
        mbg_nbd = ModifiedBetaGeoNBD.dist(a=1.20, b=0.75, r=0.66, alpha=2.78, T=6)
        value = pt.vector("value", shape=(2,))
        logp = pm.logp(mbg_nbd, value)

        logp_fn = pytensor.function([value], logp)
        assert logp_fn(np.array([3, -1])) == -np.inf
        assert logp_fn(np.array([-1, 1.5])) == -np.inf
        assert logp_fn(np.array([11, 1.5])) == -np.inf

    def test_notimplemented_logp(self):
        dist = ModifiedBetaGeoNBD.dist(a=1, b=1, r=2, alpha=2, T=10)
        invalid_value = np.broadcast_to([1, 3], (4, 3, 2))

        with pytest.raises(NotImplementedError):
            pm.logp(dist, invalid_value)
