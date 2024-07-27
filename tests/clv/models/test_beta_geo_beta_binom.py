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
import os

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from lifetimes.fitters.beta_geo_beta_binom_fitter import BetaGeoBetaBinomFitter

from pymc_marketing.clv.models import BetaGeoBetaBinomModel
from pymc_marketing.prior import Prior


class TestBetaGeoBetaBinomModel:
    @classmethod
    def setup_class(cls):
        # Set random seed
        cls.rng = np.random.default_rng(34)

        # parameters
        cls.alpha_true = 0.793
        cls.beta_true = 2.426
        cls.delta_true = 4.414
        cls.gamma_true = 0.243

        # Use Quickstart dataset (the CDNOW_sample research data) for testing
        test_data = pd.read_csv("data/bgbb_donations.csv")

        cls.data = test_data
        cls.customer_id = test_data["customer_id"]
        cls.frequency = test_data["frequency"]
        cls.recency = test_data["recency"]
        cls.T = test_data["T"]

        # Instantiate model with CDNOW data for testing
        cls.model = BetaGeoBetaBinomModel(cls.data)

        # Also instantiate lifetimes model for comparison
        cls.lifetimes_model = BetaGeoBetaBinomFitter()
        cls.lifetimes_model.params_ = {
            "alpha": cls.alpha_true,
            "beta": cls.beta_true,
            "delta": cls.delta_true,
            "gamma": cls.gamma_true,
        }

        # Mock an idata object for tests requiring a fitted model
        cls.N = len(cls.customer_id)
        cls.chains = 2
        cls.draws = 50
        cls.mock_fit = az.from_dict(
            {
                "a": cls.rng.normal(cls.alpha_true, 1e-3, size=(cls.chains, cls.draws)),
                "b": cls.rng.normal(cls.beta_true, 1e-3, size=(cls.chains, cls.draws)),
                "alpha": cls.rng.normal(
                    cls.delta_true, 1e-3, size=(cls.chains, cls.draws)
                ),
                "r": cls.rng.normal(cls.gamma_true, 1e-3, size=(cls.chains, cls.draws)),
            }
        )

        cls.model.idata = cls.mock_fit

    @pytest.fixture(scope="class")
    def model_config(self):
        return {
            "alpha_prior": Prior("HalfNormal"),
            "beta_prior": Prior("HalfStudentT", nu=4),
            "delta_prior": Prior("HalfCauchy", beta=2),
            "gamma_prior": Prior("Gamma", alpha=1, beta=1),
        }

    @pytest.fixture(scope="class")
    def default_model_config(self):
        return {
            "alpha_prior": Prior("HalfFlat"),
            "beta_prior": Prior("HalfFlat"),
            "delta_prior": Prior("HalfFlat"),
            "gamma_prior": Prior("HalfFlat"),
        }

    def test_model(self, model_config, default_model_config):
        for config in (model_config, default_model_config):
            model = BetaGeoBetaBinomModel(
                data=self.data,
                model_config=config,
            )
            model.build_model()
            assert isinstance(
                model.model["alpha"].owner.op,
                pm.HalfFlat
                if config["alpha_prior"].distribution == "HalfFlat"
                else config["alpha_prior"].pymc_distribution,
            )
            assert isinstance(
                model.model["beta"].owner.op,
                pm.HalfFlat
                if config["beta_prior"].distribution == "HalfFlat"
                else config["beta_prior"].pymc_distribution,
            )
            assert isinstance(
                model.model["delta"].owner.op,
                pm.HalfFlat
                if config["delta_prior"].distribution == "HalfFlat"
                else config["delta_prior"].pymc_distribution,
            )
            assert isinstance(
                model.model["gamma"].owner.op,
                pm.HalfFlat
                if config["gamma_prior"].distribution == "HalfFlat"
                else config["gamma_prior"].pymc_distribution,
            )
            assert model.model.eval_rv_shapes() == {
                "alpha": (),
                "alpha_log__": (),
                "beta": (),
                "beta_log__": (),
                "delta": (),
                "delta_log__": (),
                "gamma": (),
                "gamma_log__": (),
            }

    def test_missing_cols(self):
        data_invalid = self.data.drop(columns="customer_id")

        with pytest.raises(ValueError, match="Required column customer_id missing"):
            BetaGeoBetaBinomModel(data=data_invalid)

        data_invalid = self.data.drop(columns="frequency")

        with pytest.raises(ValueError, match="Required column frequency missing"):
            BetaGeoBetaBinomModel(data=data_invalid)

        data_invalid = self.data.drop(columns="recency")

        with pytest.raises(ValueError, match="Required column recency missing"):
            BetaGeoBetaBinomModel(data=data_invalid)

        data_invalid = self.data.drop(columns="T")

        with pytest.raises(ValueError, match="Required column T missing"):
            BetaGeoBetaBinomModel(data=data_invalid)

    def test_customer_id_duplicate(self):
        with pytest.raises(
            ValueError, match="Column customer_id has duplicate entries"
        ):
            data = pd.DataFrame(
                {
                    "customer_id": np.asarray([1, 1]),
                    "frequency": np.asarray([1, 1]),
                    "recency": np.asarray([1, 1]),
                    "T": np.asarray([1, 1]),
                }
            )

            BetaGeoBetaBinomModel(
                data=data,
            )

    def test_T_homogeneity(self):
        with pytest.raises(ValueError, match="Column T has  non-homogeneous entries"):
            data = pd.DataFrame(
                {
                    "customer_id": np.asarray([1, 2]),
                    "frequency": np.asarray([1, 2]),
                    "recency": np.asarray([1, 2]),
                    "T": np.asarray([1, 2]),
                }
            )

            BetaGeoBetaBinomModel(
                data=data,
            )

    def test_model_repr(self):
        model_config = {
            "alpha_prior": Prior("HalfFlat"),
            "beta_prior": Prior("HalfFlat"),
            "delta_prior": Prior("HalfFlat"),
            "gamma_prior": Prior("HalfNormal", sigma=10),
        }
        model = BetaGeoBetaBinomModel(
            data=self.data,
            model_config=model_config,
        )
        model.build_model()
        assert model.__repr__().replace(" ", "") == (
            "BG/BB"
            "\nalpha~HalfFlat()"
            "\nbeta~HalfFlat()"
            "\ngamma~HalfNormal(0,10)"
            "\ndelta~HalfFlat()"
            "\nrecency_frequency~BetaGeoBetaBinom(alpha,beta,gamma,delta,<constant>)"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "fit_method, rtol",
        [
            ("mcmc", 0.1),
            ("map", 0.2),
        ],
    )
    def test_model_convergence(self, fit_method, rtol, model_config):
        model = BetaGeoBetaBinomModel(
            data=self.data,
            model_config=model_config,
        )
        model.build_model()

        sample_kwargs = (
            dict(random_seed=self.rng, chains=2) if fit_method == "mcmc" else {}
        )
        model.fit(fit_method=fit_method, progressbar=False, **sample_kwargs)

        fit = model.idata.posterior
        np.testing.assert_allclose(
            [
                fit["alpha"].mean(),
                fit["beta"].mean(),
                fit["delta"].mean(),
                fit["gamma"].mean(),
            ],
            [self.alpha_true, self.beta_true, self.delta_true, self.gamma_true],
            rtol=rtol,
        )

    def test_fit_result_without_fit(self, model_config):
        model = BetaGeoBetaBinomModel(data=self.data, model_config=model_config)
        with pytest.raises(RuntimeError, match="The model hasn't been fit yet"):
            model.fit_result

        idata = model.fit(
            tune=5,
            chains=2,
            draws=10,
            compute_convergence_checks=False,
        )
        assert isinstance(idata, az.InferenceData)
        assert len(idata.posterior.chain) == 2
        assert len(idata.posterior.draw) == 10
        assert model.idata is idata

    @pytest.mark.parametrize("test_t", [1, 3, 6])
    def test_expected_purchases(self, test_t):
        pass

    @pytest.mark.parametrize("test_t", [1, 3, 6])
    def test_expected_purchases_new_customer(self, test_t):
        pass

    def test_expected_probability_alive(self):
        pass

    # TODO: Add a test for recency_frequency
    def test_distribution_new_customer(self) -> None:
        mock_model = BetaGeoBetaBinomModel(
            data=self.data,
        )
        mock_model.idata = az.from_dict(
            {
                "alpha": [self.alpha_true],
                "beta": [self.beta_true],
                "delta": [self.delta_true],
                "gamma": [self.gamma_true],
            }
        )

        rng = np.random.default_rng(42)
        new_customer_dropout = mock_model.distribution_new_customer_dropout(
            random_seed=rng
        )
        new_customer_purchase_rate = mock_model.distribution_new_customer_purchase_rate(
            random_seed=rng
        )

        assert isinstance(new_customer_dropout, xr.DataArray)
        assert isinstance(new_customer_purchase_rate, xr.DataArray)

        N = 1000
        # TODO: do these match the greek letters in the research?
        p = pm.Beta.dist(self.alpha_true, self.beta_true, size=N)
        lam = pm.Beta.dist(self.delta_true, self.gamma_true, size=N)

        rtol = 0.15
        np.testing.assert_allclose(
            new_customer_dropout.mean(), pm.draw(p.mean(), random_seed=rng), rtol=rtol
        )
        np.testing.assert_allclose(
            new_customer_dropout.var(), pm.draw(p.var(), random_seed=rng), rtol=rtol
        )
        np.testing.assert_allclose(
            new_customer_purchase_rate.mean(),
            pm.draw(lam.mean(), random_seed=rng),
            rtol=rtol,
        )
        np.testing.assert_allclose(
            new_customer_purchase_rate.var(),
            pm.draw(lam.var(), random_seed=rng),
            rtol=rtol,
        )

    def test_save_load(self):
        self.model.build_model()
        self.model.fit("map", maxeval=1)
        self.model.save("test_model")
        # Testing the valid case.

        model2 = BetaGeoBetaBinomModel.load("test_model")

        # Check if the loaded model is indeed an instance of the class
        assert isinstance(self.model, BetaGeoBetaBinomModel)
        # Check if the loaded data matches with the model data
        pd.testing.assert_frame_equal(self.model.data, model2.data, check_names=False)
        assert self.model.model_config == model2.model_config
        assert self.model.sampler_config == model2.sampler_config
        assert self.model.idata == model2.idata
        os.remove("test_model")
