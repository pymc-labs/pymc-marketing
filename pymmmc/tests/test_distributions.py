import pytest

from pymmmc.clv import IndividualLevelCLV, BetaGeoFitter
from pymc import Model

import pymc as pm

import numpy as np

from numpy.testing import assert_almost_equal
from pymc.tests.helpers import select_by_precision


# class TestIndividualLevelCLV:
    # @pytest.parametrize(
    #     "value, lam, p, T, T0, logp",
    #     [
    #         ()
    #     ]
    # )
    # def test_individual_level_clv_logp(self, value, lam, p, T, T0, logp):
    #     with Model() as model:
    #         il_clv = IndividualLevelCLV("il-clv", lam=lam, p=p, T=T, T0=T0)
    #     pt = {"il_clv": value}


    #     assert_almost_equal(
    #         pm.logp(il_clv, value).eval(),
    #         logp,
    #         decimal=select_by_precision(float64=6, float32=2),
    #         err_msg=str(pt),
    #     )

    # def test_individual_level_clv_invalid(self): # not working...
    #     il_clv = IndividualLevelCLV.dist(lam=0.8, p=0.15, T=10, T0=2)
    #     assert pm.logp(il_clv, np.array([-1, 3])).eval() == -np.inf
    #     assert pm.logp(il_clv, np.array([1.5, -1])).eval() == -np.inf
    #     assert pm.logp(il_clv, np.array([1.5, 0])).eval() == -np.inf
