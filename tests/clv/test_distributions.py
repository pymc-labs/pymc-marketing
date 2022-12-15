import numpy as np
import pymc as pm
import pytest
from numpy.testing import assert_almost_equal
from pymc import Model
from pymc.tests.helpers import select_by_precision

from pymc_marketing.clv.distributions import ContContract, ContNonContract


class TestContNonContract:
    @pytest.mark.parametrize(
        "value, lam, p, T, T0, logp",
        [
            (np.array([6.3, 5]), 0.4, 0.15, 12, 2, -8.39147106159807),
            (
                np.array([6.3, 5]),
                np.array([0.3, 0.2]),
                0.15,
                12,
                2,
                np.array([-9.15153637, -10.42037984]),
            ),
            (
                np.array([[6.3, 5], [5.3, 4]]),
                np.array([0.3, 0.2]),
                0.15,
                12,
                2,
                np.array([-9.15153637, -8.57264195]),
            ),
            (
                np.array([6.3, 5]),
                0.3,
                np.full((5, 3), 0.15),
                12,
                2,
                np.full(shape=(5, 3), fill_value=-9.15153637),
            ),
        ],
    )
    def test_continuous_non_contractual(self, value, lam, p, T, T0, logp):
        with Model():
            cnc = ContNonContract("cnc", lam=lam, p=p, T=T, T0=T0)
        pt = {"cnc": value}

        assert_almost_equal(
            pm.logp(cnc, value).eval(),
            logp,
            decimal=select_by_precision(float64=6, float32=2),
            err_msg=str(pt),
        )

    def test_continuous_non_contractual_invalid(self):
        cnc = ContNonContract.dist(lam=0.8, p=0.15, T=10, T0=2)
        assert pm.logp(cnc, np.array([-1, 3])).eval() == -np.inf
        assert pm.logp(cnc, np.array([1.5, -1])).eval() == -np.inf
        assert pm.logp(cnc, np.array([1.5, 0])).eval() == -np.inf
        assert pm.logp(cnc, np.array([11, 3])).eval() == -np.inf

    # TODO: test broadcasting of parameters, including T and T0
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
            ContNonContract(name="cnc", lam=lam, p=p, T=10, T0=2, size=cnc_size)
            prior = pm.sample_prior_predictive(samples=100)

        assert prior["prior"]["cnc"][0].shape == (100,) + expected_size


class TestContContract:
    @pytest.mark.parametrize(
        "value, lam, p, T, T0, logp",
        [
            (np.array([6.3, 5, 1]), 0.3, 0.15, 12, 2, -10.45705972),
            (
                np.array([6.3, 5, 1]),
                np.array([0.3, 0.2]),
                0.15,
                12,
                2,
                np.array([-10.45705972, -11.85438527]),
            ),
            (
                np.array([[6.3, 5, 1], [5.3, 4, 0]]),
                np.array([0.3, 0.2]),
                0.15,
                12,
                2,
                np.array([-10.45705972, -9.08782737]),
            ),
            (
                np.array([6.3, 5, 0]),
                0.3,
                np.full((5, 3), 0.15),
                12,
                2,
                np.full(shape=(5, 3), fill_value=-9.83245867),
            ),
        ],
    )
    def test_continuous_contractual(self, value, lam, p, T, T0, logp):
        with Model():
            cc = ContContract("cc", lam=lam, p=p, T=T, T0=T0)
        pt = {"cc": value}

        assert_almost_equal(
            pm.logp(cc, value).eval(),
            logp,
            decimal=select_by_precision(float64=6, float32=2),
            err_msg=str(pt),
        )

    def test_continuous_contractual_invalid(self):
        cc = ContContract.dist(lam=0.8, p=0.15, T=10, T0=2)
        assert pm.logp(cc, np.array([-1, 3, 1])).eval() == -np.inf
        assert pm.logp(cc, np.array([1.5, -1, 1])).eval() == -np.inf
        assert pm.logp(cc, np.array([1.5, 0, 1])).eval() == -np.inf
        assert pm.logp(cc, np.array([11, 3, 1])).eval() == -np.inf
        assert pm.logp(cc, np.array([1.5, 3, 0.5])).eval() == -np.inf
        assert pm.logp(cc, np.array([1.5, 3, -1])).eval() == -np.inf

    # TODO: test broadcasting of parameters, including T and T0
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
            ContContract(name="cc", lam=lam, p=p, T=10, T0=2, size=cc_size)
            prior = pm.sample_prior_predictive(samples=100)

        assert prior["prior"]["cc"][0].shape == (100,) + expected_size
