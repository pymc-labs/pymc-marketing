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
"""Tests for the Beta-Discrete-Weibull distribution and models."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pymc_extras.prior import Prior
from scipy.special import betaln

from pymc_marketing.clv import (
    BetaDiscreteWeibullModel,
    BetaDiscreteWeibullModelIndividual,
)
from pymc_marketing.clv.distributions import BetaDiscreteWeibull, DiscreteWeibull


def _bdw_pmf_reference(t, alpha, beta, c):
    """Closed-form BdW pmf, independent of the pytensor implementation."""
    s_prev = np.exp(
        betaln(alpha, beta + np.where(t > 1, (t - 1), 0.0) ** c) - betaln(alpha, beta)
    )
    s_curr = np.exp(betaln(alpha, beta + t**c) - betaln(alpha, beta))
    return s_prev - s_curr


class TestDiscreteWeibullDistributions:
    @pytest.mark.parametrize("theta, c", [(0.1, 1.0), (0.3, 0.7), (0.05, 1.8)])
    def test_discrete_weibull_logp_matches_reference(self, theta, c):
        value = np.arange(1, 50)
        logp = pm.logp(DiscreteWeibull.dist(theta=theta, c=c), value).eval()
        ref = (1 - theta) ** (np.where(value > 1, (value - 1), 0.0) ** c) - (
            1 - theta
        ) ** (value**c)
        np.testing.assert_allclose(np.exp(logp), ref, rtol=1e-6, atol=1e-12)

    @pytest.mark.parametrize(
        "alpha, beta, c", [(1.0, 1.0, 1.0), (0.8, 2.0, 0.7), (2.0, 3.0, 1.8)]
    )
    def test_bdw_logp_matches_reference(self, alpha, beta, c):
        value = np.arange(1, 50)
        logp = pm.logp(
            BetaDiscreteWeibull.dist(alpha=alpha, beta=beta, c=c), value
        ).eval()
        np.testing.assert_allclose(
            np.exp(logp),
            _bdw_pmf_reference(value, alpha, beta, c),
            rtol=1e-6,
            atol=1e-12,
        )

    def test_bdw_reduces_to_sbg_survivor_at_c_equals_one(self):
        """At c == 1 the BdW survivor equals the shifted-beta-geometric survivor."""
        alpha, beta = 0.8, 2.0
        value = np.arange(1, 30)
        logp = pm.logp(
            BetaDiscreteWeibull.dist(alpha=alpha, beta=beta, c=1.0), value
        ).eval()
        survivor_bdw = 1 - np.cumsum(np.exp(logp))[:12]
        t = np.arange(1, 13, dtype=float)
        survivor_sbg = np.exp(betaln(alpha, beta + t) - betaln(alpha, beta))
        np.testing.assert_allclose(survivor_bdw, survivor_sbg, rtol=1e-5, atol=1e-8)

    def test_bdw_logp_invalid_params(self):
        # support starts at 1; value 0 and out-of-range params are -inf
        assert np.isneginf(
            pm.logp(BetaDiscreteWeibull.dist(alpha=1.0, beta=1.0, c=1.0), 0).eval()
        )

    def test_bdw_draw_shape_and_support(self):
        draws = pm.draw(
            BetaDiscreteWeibull.dist(alpha=2.0, beta=3.0, c=1.2, size=500),
            random_seed=42,
        )
        assert draws.shape == (500,)
        assert draws.min() >= 1


@pytest.fixture(scope="module")
def bdw_cohort_data():
    rng = np.random.default_rng(123)
    rows = []
    cid = 0
    for cohort, (alpha, beta, c, n) in {
        "2025-01": (1.2, 2.0, 1.0, 120),
        "2025-02": (0.9, 1.5, 1.3, 120),
    }.items():
        T = 8
        theta = rng.beta(alpha, beta, size=n)
        # inverse-cdf discrete-Weibull lifetimes, right-censored at T
        u = rng.uniform(size=n)
        raw = np.log1p(-u) / np.log1p(-np.clip(theta, 1e-9, 1 - 1e-9))
        churn = np.maximum(np.ceil(raw ** (1.0 / c)), 1).astype(int)
        recency = np.minimum(churn, T)
        for r in recency:
            rows.append((cid, int(r), T, cohort))
            cid += 1
    return pd.DataFrame(rows, columns=["customer_id", "recency", "T", "cohort"])


class TestBetaDiscreteWeibullModel:
    def test_default_model_config(self, bdw_cohort_data):
        model = BetaDiscreteWeibullModel(data=bdw_cohort_data)
        cfg = model.default_model_config
        assert set(cfg) == {"phi", "kappa", "c"}

    def test_build_model_requires_data(self):
        model = BetaDiscreteWeibullModel()
        with pytest.raises(ValueError, match="requires data parameter"):
            model.build_model()

    def test_build_and_fit_map(self, bdw_cohort_data):
        model = BetaDiscreteWeibullModel(data=bdw_cohort_data)
        model.build_model(bdw_cohort_data)
        assert "recency" in model.model.named_vars
        assert set(model.model.coords["cohort"]) == {"2025-01", "2025-02"}
        model.fit(method="map")
        for var in ("alpha", "beta", "c"):
            assert var in model.fit_result

    def test_validate_data_rejects_bad_recency(self, bdw_cohort_data):
        bad = bdw_cohort_data.copy()
        bad.loc[bad.index[0], "recency"] = 0
        model = BetaDiscreteWeibullModel()
        with pytest.raises(ValueError, match="1 <= recency <= T"):
            model.build_model(bad)

    def test_expected_retention_rate_shape(self, bdw_cohort_data):
        model = BetaDiscreteWeibullModel(data=bdw_cohort_data)
        model.fit(method="map")
        rr = model.expected_retention_rate(future_t=1)
        assert isinstance(rr, xr.DataArray)

    def test_save_load_roundtrip_without_fit_data(self, bdw_cohort_data, tmp_path):
        """Round-trip exercises build_from_idata -> build_model(data)."""
        model = BetaDiscreteWeibullModel(data=bdw_cohort_data)
        model.fit(method="map")
        fp = tmp_path / "bdw.nc"
        model.save(str(fp))
        loaded = BetaDiscreteWeibullModel.load(str(fp))
        assert loaded.fit_result is not None
        np.testing.assert_allclose(
            loaded.fit_result["c"].values, model.fit_result["c"].values
        )


class TestBetaDiscreteWeibullModelIndividual:
    @pytest.fixture(scope="class")
    def individual_data(self):
        rng = np.random.default_rng(7)
        n, T = 200, 10
        theta = rng.beta(1.0, 1.5, size=n)
        u = rng.uniform(size=n)
        raw = np.log1p(-u) / np.log1p(-np.clip(theta, 1e-9, 1 - 1e-9))
        churn = np.maximum(np.ceil(raw ** (1.0 / 1.2)), 1).astype(int)
        t_churn = np.minimum(churn, T)
        return pd.DataFrame({"customer_id": np.arange(n), "t_churn": t_churn, "T": T})

    def test_default_config_keys(self):
        model = BetaDiscreteWeibullModelIndividual()
        assert set(model.default_model_config) == {"alpha", "beta", "c"}

    def test_build_and_fit_map(self, individual_data):
        model = BetaDiscreteWeibullModelIndividual(
            model_config={
                "alpha": Prior("HalfFlat"),
                "beta": Prior("HalfFlat"),
                "c": Prior("HalfNormal", sigma=1.0),
            }
        )
        model.build_model(individual_data)
        assert "churn_censored" in model.model.named_vars
        model.fit(method="map")
        for var in ("alpha", "beta", "c"):
            assert var in model.fit_result
