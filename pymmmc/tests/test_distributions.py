import pytest

from pymmmc.distributions import ContNonContract, BetaGeoFitter
from pymc import Model

import pymc as pm

import numpy as np

from numpy.testing import assert_almost_equal
from pymc.tests.helpers import select_by_precision


class TestContNonContract:
    @pytest.mark.parametrize(
        "value, lam, p, T, T0, logp",
        [
            (np.array([6.3, 5]), 0.4, 0.15, 12, 2, -19.842697669107405),
            (np.array([6.3, 5]), np.array([0.3, 0.2]), 0.15, 12, 2, np.array([[-20.88951839, -23.11416947]])),
            (np.array([[6.3, 5], [5.3, 4]]), np.array([0.3, 0.2]), 0.15, 12, 2, np.array([[-20.88951839 -19.37025579]])),
            (np.array([6.3, 5]), 0.3, np.full((5, 3), 0.15), 12, 2, np.full(shape=(5, 3), fill_value=-20.88951839)),
        ]
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

