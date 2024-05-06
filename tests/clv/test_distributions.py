#   Copyright 2024 The PyMC Labs Developers
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
import pytest
from lifetimes import ParetoNBDFitter as PF
from numpy.testing import assert_almost_equal
from pymc import Model

from pymc_marketing.clv.distributions import ContContract, ContNonContract, ParetoNBD


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
            prior = pm.sample_prior_predictive(samples=100)

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
            prior = pm.sample_prior_predictive(samples=100)

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

            T = pm.MutableData(name="T", value=np.array(10))

            ParetoNBD(
                name="pareto_nbd",
                r=r,
                alpha=alpha,
                s=s,
                beta=beta,
                T=T,
                size=pareto_nbd_size,
            )
            prior = pm.sample_prior_predictive(samples=100)

        assert prior["prior"]["pareto_nbd"][0].shape == (100,) + expected_size  # noqa: RUF005
