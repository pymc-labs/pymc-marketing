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
import os

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from lifetimes.fitters.beta_geo_fitter import BetaGeoFitter

from pymc_marketing.clv.distributions import BetaGeoNBD
from pymc_marketing.clv.models.beta_geo import BetaGeoModel
from pymc_marketing.prior import Prior
from tests.conftest import create_mock_fit, mock_sample, set_model_fit


class TestBetaGeoModel:
    @classmethod
    def setup_class(cls):
        # Set random seed
        cls.seed = 42
        cls.rng = np.random.default_rng(cls.seed)

        # parameters
        cls.a_true = 0.793
        cls.b_true = 2.426
        cls.alpha_true = 4.414
        cls.r_true = 0.243

        # Use Quickstart dataset (the CDNOW_sample research data) for testing
        test_data = pd.read_csv("data/clv_quickstart.csv")
        test_data["customer_id"] = test_data.index

        cls.data = test_data
        cls.customer_id = test_data["customer_id"]
        cls.frequency = test_data["frequency"]
        cls.recency = test_data["recency"]
        cls.T = test_data["T"]

        # Instantiate model with CDNOW data for testing
        cls.model = BetaGeoModel(cls.data)
        cls.model.build_model()

        # Also instantiate lifetimes model for comparison
        cls.lifetimes_model = BetaGeoFitter()
        cls.lifetimes_model.params_ = {
            "a": cls.a_true,
            "b": cls.b_true,
            "alpha": cls.alpha_true,
            "r": cls.r_true,
        }

        # Mock an idata object for tests requiring a fitted model
        cls.N = len(cls.customer_id)
        cls.chains = 2
        cls.draws = 50

        mock_fit = create_mock_fit(
            {
                "a": cls.a_true,
                "b": cls.b_true,
                "alpha": cls.alpha_true,
                "r": cls.r_true,
            }
        )

        mock_fit(cls.model, cls.chains, cls.draws, cls.rng)

    @pytest.fixture(scope="class")
    def model_config(self):
        return {
            "a": Prior("HalfNormal"),
            "b": Prior("HalfStudentT", nu=4),
            "alpha": Prior("HalfCauchy", beta=2),
            "r": Prior("Gamma", alpha=1, beta=1),
        }

    @pytest.fixture(scope="class")
    def default_model_config(self):
        return {
            "a": Prior("HalfFlat"),
            "b": Prior("HalfFlat"),
            "alpha": Prior("HalfFlat"),
            "r": Prior("HalfFlat"),
        }

    def test_model(self, model_config, default_model_config):
        for config in (model_config, default_model_config):
            model = BetaGeoModel(
                data=self.data,
                model_config=config,
            )
            model.build_model()
            assert isinstance(
                model.model["a"].owner.op,
                pm.HalfFlat
                if config["a"].distribution == "HalfFlat"
                else config["a"].pymc_distribution,
            )
            assert isinstance(
                model.model["b"].owner.op,
                pm.HalfFlat
                if config["b"].distribution == "HalfFlat"
                else config["b"].pymc_distribution,
            )
            assert isinstance(
                model.model["alpha"].owner.op,
                pm.HalfFlat
                if config["alpha"].distribution == "HalfFlat"
                else config["alpha"].pymc_distribution,
            )
            assert isinstance(
                model.model["r"].owner.op,
                pm.HalfFlat
                if config["r"].distribution == "HalfFlat"
                else config["r"].pymc_distribution,
            )
            assert model.model.eval_rv_shapes() == {
                "a": (),
                "a_log__": (),
                "b": (),
                "b_log__": (),
                "alpha": (),
                "alpha_log__": (),
                "r": (),
                "r_log__": (),
            }

    def test_missing_cols(self):
        data_invalid = self.data.drop(columns="customer_id")

        with pytest.raises(ValueError, match="Required column customer_id missing"):
            BetaGeoModel(data=data_invalid)

        data_invalid = self.data.drop(columns="frequency")

        with pytest.raises(ValueError, match="Required column frequency missing"):
            BetaGeoModel(data=data_invalid)

        data_invalid = self.data.drop(columns="recency")

        with pytest.raises(ValueError, match="Required column recency missing"):
            BetaGeoModel(data=data_invalid)

        data_invalid = self.data.drop(columns="T")

        with pytest.raises(ValueError, match="Required column T missing"):
            BetaGeoModel(data=data_invalid)

    def test_customer_id_duplicate(self):
        data = pd.DataFrame(
            {
                "customer_id": np.asarray([1, 1]),
                "frequency": np.asarray([1, 1]),
                "recency": np.asarray([1, 1]),
                "T": np.asarray([1, 1]),
            }
        )

        with pytest.raises(
            ValueError, match="Column customer_id has duplicate entries"
        ):
            BetaGeoModel(
                data=data,
            )

    @pytest.mark.parametrize(
        "frequency, recency, logp_value",
        [
            (0, 0, -0.59947382),
            (200, 38, 100.7957),
        ],
    )
    def test_numerically_stable_logp(
        self, frequency, recency, logp_value, model_config
    ):
        """See Solution #2 on pages 3 and 4 of http://brucehardie.com/notes/027/bgnbd_num_error.pdf"""
        model_config = {
            "a": Prior("Flat"),
            "b": Prior("Flat"),
            "alpha": Prior("Flat"),
            "r": Prior("Flat"),
        }
        data = pd.DataFrame(
            {
                "customer_id": np.asarray([1]),
                "frequency": np.asarray([frequency]),
                "recency": np.asarray([recency]),
                "T": np.asarray([40]),
            }
        )

        model = BetaGeoModel(
            data=data,
            model_config=model_config,
        )

        model.build_model()
        pymc_model = model.model

        logp = pymc_model.compile_logp()

        np.testing.assert_almost_equal(
            logp({"a": 0.80, "b": 2.50, "r": 0.25, "alpha": 4.00}),
            logp_value,
            decimal=5,
        )

    @pytest.mark.parametrize("fit_type", ("map", "mcmc"))
    def test_posterior_distributions(self, fit_type) -> None:
        rng = np.random.default_rng(42)
        dim_T = 2357

        if fit_type == "map":
            map_idata = self.model.idata.copy()
            map_idata.posterior = map_idata.posterior.isel(
                chain=slice(None, 1), draw=slice(None, 1)
            )
            model = self.model._build_with_idata(map_idata)
            # We expect 1000 draws to be sampled with MAP
            expected_shape = (1, 1000)
            expected_pop_dims = (1, 1000, dim_T, 2)
        else:
            model = self.model
            expected_shape = (self.chains, self.draws)
            expected_pop_dims = (self.chains, self.draws, dim_T, 2)

        data = model.data
        customer_dropout = model.distribution_new_customer_dropout(
            data, random_seed=rng
        )
        customer_purchase_rate = model.distribution_new_customer_purchase_rate(
            data, random_seed=rng
        )
        customer_rec_freq = model.distribution_new_customer_recency_frequency(
            data, random_seed=rng
        )
        customer_rec = customer_rec_freq.sel(obs_var="recency")
        customer_freq = customer_rec_freq.sel(obs_var="frequency")

        assert customer_dropout.shape == expected_shape
        assert customer_purchase_rate.shape == expected_shape
        assert customer_rec_freq.shape == expected_pop_dims

        N = 1000
        p = pm.Beta.dist(self.a_true, self.b_true, size=N)
        lam = pm.Gamma.dist(self.r_true, self.alpha_true, size=N)

        ref_rec, ref_freq = pm.draw(
            BetaGeoNBD.dist(
                a=self.a_true,
                b=self.b_true,
                r=self.r_true,
                alpha=self.alpha_true,
                T=self.T,
            ),
            random_seed=rng,
        ).T

        np.testing.assert_allclose(
            customer_purchase_rate.mean(),
            pm.draw(lam.mean(), random_seed=rng),
            rtol=0.5,
        )
        np.testing.assert_allclose(
            customer_purchase_rate.std(),
            pm.draw(lam.std(), random_seed=rng),
            rtol=0.5,
        )
        np.testing.assert_allclose(
            customer_dropout.mean(), pm.draw(p.mean(), random_seed=rng), rtol=0.5
        )
        np.testing.assert_allclose(
            customer_dropout.std(), pm.draw(p.std(), random_seed=rng), rtol=0.5
        )

        np.testing.assert_allclose(customer_rec.mean(), ref_rec.mean(), rtol=0.5)
        np.testing.assert_allclose(customer_rec.std(), ref_rec.std(), rtol=0.5)

        np.testing.assert_allclose(customer_freq.mean(), ref_freq.mean(), rtol=0.5)
        np.testing.assert_allclose(customer_freq.std(), ref_freq.std(), rtol=0.5)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "method, rtol",
        [
            ("mcmc", 0.1),
            ("map", 0.2),
            ("advi", 0.25),
        ],
    )
    def test_model_convergence(self, method, rtol, model_config):
        # b parameter has the largest mismatch of the four parameters
        model = BetaGeoModel(
            data=self.data,
            model_config=model_config,
        )
        model.build_model()

        if method == "advi":
            sample_kwargs = dict(random_seed=self.seed)
        if method == "mcmc":
            sample_kwargs = dict(random_seed=self.seed, chains=2)
        elif method == "map":
            sample_kwargs = dict(seed=self.seed)

        model.fit(method=method, progressbar=False, **sample_kwargs)

        fit = model.idata.posterior
        np.testing.assert_allclose(
            [fit["a"].mean(), fit["b"].mean(), fit["alpha"].mean(), fit["r"].mean()],
            [self.a_true, self.b_true, self.alpha_true, self.r_true],
            rtol=rtol,
        )

    def test_fit_result_without_fit(self, mocker, model_config):
        model = BetaGeoModel(data=self.data, model_config=model_config)
        with pytest.raises(RuntimeError, match="The model hasn't been fit yet"):
            model.fit_result

        mocker.patch("pymc.sample", mock_sample)

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

    def test_expected_probability_no_purchases_infrequent_customers(self):
        atol = 10e-3
        customer_id = np.arange(5)
        test_frequency = [3, 30, 5, 70, 9]
        test_recency = [100, 30, 500, 70, 900]
        test_T = [500, 300, 1000, 700, 1800]
        test_t = 3
        data = pd.DataFrame(
            {
                "customer_id": customer_id,
                "frequency": test_frequency,
                "recency": test_recency,
                "T": test_T,
            }
        )

        bg_model = BetaGeoModel(data=data)
        bg_model.build_model()
        bg_model.idata = az.from_dict(
            {
                "a": np.full((2, 5), self.a_true),
                "b": np.full((2, 5), self.b_true),
                "alpha": np.full((2, 5), self.alpha_true),
                "r": np.full((2, 5), self.r_true),
            }
        )

        res_prob_no_purchases = bg_model.expected_probability_no_purchase(
            t=test_t, data=data
        )
        assert np.all(np.isclose(res_prob_no_purchases.to_numpy(), 1, atol=atol))

    @pytest.mark.parametrize("test_t", [30, 90, 120])
    def test_expected_probability_no_purchases_frequent_customers(self, test_t):
        atol = 10e-3
        customer_id = np.arange(5)
        test_frequency = [100, 300, 500, 700, 900]
        test_recency = [100, 300, 500, 700, 900]
        test_T = [100, 300, 500, 700, 900]
        data = pd.DataFrame(
            {
                "customer_id": customer_id,
                "frequency": test_frequency,
                "recency": test_recency,
                "T": test_T,
            }
        )

        bg_model = BetaGeoModel(data=data)
        bg_model.build_model()
        bg_model.idata = az.from_dict(
            {
                "a": np.full((2, 5), self.a_true),
                "b": np.full((2, 5), self.b_true),
                "alpha": np.full((2, 5), self.alpha_true),
                "r": np.full((2, 5), self.r_true),
            }
        )

        res_prob_no_purchases = bg_model.expected_probability_no_purchase(
            t=test_t, data=data
        )
        assert np.all(np.isclose(res_prob_no_purchases.to_numpy(), 0, atol=atol))

    def test_expected_probability_no_purchases_now(self):
        customer_id = np.arange(10)
        test_frequency = np.tile([1, 3, 5, 7, 9], 2)
        test_recency = np.tile([20, 30], 5)
        test_T = np.tile([25, 35], 5)
        test_t = 0
        data = pd.DataFrame(
            {
                "customer_id": customer_id,
                "frequency": test_frequency,
                "recency": test_recency,
                "T": test_T,
            }
        )

        bg_model = BetaGeoModel(data=data)
        bg_model.build_model()
        bg_model.idata = az.from_dict(
            {
                "a": np.full((2, 5), self.a_true),
                "b": np.full((2, 5), self.b_true),
                "alpha": np.full((2, 5), self.alpha_true),
                "r": np.full((2, 5), self.r_true),
            }
        )

        res_prob_no_purchases = bg_model.expected_probability_no_purchase(
            t=test_t, data=data
        )
        assert np.all(np.isclose(res_prob_no_purchases.to_numpy(), 1))

    @pytest.mark.parametrize("test_t", [0, 30, 90])
    def test_expected_probability_no_purchases(self, test_t):
        customer_id = np.arange(10)
        test_frequency = np.tile([1, 3, 5, 7, 9], 2)
        test_recency = np.tile([20, 30], 5)
        test_T = np.tile([25, 35], 5)
        data = pd.DataFrame(
            {
                "customer_id": customer_id,
                "frequency": test_frequency,
                "recency": test_recency,
                "T": test_T,
            }
        )

        bg_model = BetaGeoModel(data=data)
        bg_model.build_model()
        bg_model.idata = az.from_dict(
            {
                "a": np.full((2, 5), self.a_true),
                "b": np.full((2, 5), self.b_true),
                "alpha": np.full((2, 5), self.alpha_true),
                "r": np.full((2, 5), self.r_true),
            }
        )

        res_prob_no_purchases = bg_model.expected_probability_no_purchase(
            t=test_t, data=data
        )

        assert res_prob_no_purchases.shape == (2, 5, 10)
        assert res_prob_no_purchases.dims == ("chain", "draw", "customer_id")

    @pytest.mark.parametrize("test_t", [1, 3, 6])
    def test_expected_purchases(self, test_t):
        true_purchases = (
            self.lifetimes_model.conditional_expected_number_of_purchases_up_to_time(
                t=test_t,
                frequency=self.frequency,
                recency=self.recency,
                T=self.T,
            )
        )
        est_num_purchases = self.model.expected_purchases(future_t=test_t)

        assert est_num_purchases.shape == (self.chains, self.draws, self.N)
        assert est_num_purchases.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_purchases,
            est_num_purchases.mean(("chain", "draw")),
            rtol=0.001,
        )

    @pytest.mark.parametrize("test_t", [1, 3, 6])
    def test_expected_purchases_new_customer(self, test_t):
        true_purchases_new = (
            self.lifetimes_model.expected_number_of_purchases_up_to_time(
                t=test_t,
            )
        )

        est_purchases_new = self.model.expected_purchases_new_customer(t=test_t)

        assert est_purchases_new.shape == (self.chains, self.draws, 2357)
        assert est_purchases_new.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_purchases_new,
            est_purchases_new.mean(("chain", "draw", "customer_id")),
            rtol=0.001,
        )

    def test_expected_probability_alive(self):
        true_prob_alive = self.lifetimes_model.conditional_probability_alive(
            frequency=self.frequency,
            recency=self.recency,
            T=self.T,
        )

        est_prob_alive = self.model.expected_probability_alive()

        assert est_prob_alive.shape == (self.chains, self.draws, self.N)
        assert est_prob_alive.dims == ("chain", "draw", "customer_id")
        np.testing.assert_allclose(
            true_prob_alive,
            est_prob_alive.mean(("chain", "draw")),
            rtol=0.001,
        )

    def test_model_repr(self):
        model_config = {
            "alpha": Prior("HalfFlat"),
            "r": Prior("HalfFlat"),
            "a": Prior("HalfFlat"),
            "b": Prior("HalfNormal", sigma=10),
        }
        model = BetaGeoModel(
            data=self.data,
            model_config=model_config,
        )
        model.build_model()
        assert model.__repr__().replace(" ", "") == (
            "BG/NBD"
            "\nalpha~HalfFlat()"
            "\na~HalfFlat()"
            "\nb~HalfNormal(0,10)"
            "\nr~HalfFlat()"
            "\nrecency_frequency~BetaGeoNBD(a,b,r,alpha,<constant>)"
        )

    def test_distribution_new_customer(self) -> None:
        mock_model = BetaGeoModel(
            data=self.data,
        )
        mock_model.build_model()
        mock_model.idata = az.from_dict(
            {
                "a": [self.a_true],
                "b": [self.b_true],
                "alpha": [self.alpha_true],
                "r": [self.r_true],
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
        p = pm.Beta.dist(self.a_true, self.b_true, size=N)
        lam = pm.Gamma.dist(self.r_true, self.alpha_true, size=N)

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
        self.model.save("test_model")
        # Testing the valid case.

        model2 = BetaGeoModel.load("test_model")

        # Check if the loaded model is indeed an instance of the class
        assert isinstance(self.model, BetaGeoModel)
        # Check if the loaded data matches with the model data
        pd.testing.assert_frame_equal(self.model.data, model2.data, check_names=False)
        assert self.model.model_config == model2.model_config
        assert self.model.sampler_config == model2.sampler_config
        assert self.model.idata == model2.idata
        os.remove("test_model")


class TestBetaGeoModelWithCovariates:
    @classmethod
    def setup_class(cls):
        # Set random seed
        cls.rng = rng = np.random.default_rng(34)

        # parameters
        cls.true_params = dict(
            a_scale=5,
            b_scale=5,
            alpha_scale=10,
            r=5,
            purchase_coefficient_alpha=np.array([1.0, -2.0]),
            dropout_coefficient_a=np.array([2.0]),
            dropout_coefficient_b=np.array([2.0]),
        )

        # Use Quickstart dataset (the CDNOW_sample research data) for testing
        cls.data = data = pd.read_csv("data/clv_quickstart.csv")
        data["customer_id"] = data.index

        # Create two purchase covariates and one dropout covariate
        # We standardize so that the coefficient * covariates have similar variance
        N = data.shape[0]
        data["purchase_cov1"] = rng.normal(size=N) / 2
        data["purchase_cov2"] = rng.normal(size=N) / 4
        data["dropout_cov"] = rng.normal(size=N) / 6

        purchase_covariate_cols = ["purchase_cov1", "purchase_cov2"]
        dropout_covariate_cols = ["dropout_cov"]
        non_nested_priors = dict(
            a_prior=Prior("Beta", alpha=20, beta=20),
            b_prior=Prior("Beta", alpha=20, beta=20),
        )
        covariate_config = dict(
            purchase_covariate_cols=purchase_covariate_cols,
            dropout_covariate_cols=dropout_covariate_cols,
        )
        cls.model_with_covariates = BetaGeoModel(
            cls.data,
            model_config={**non_nested_priors, **covariate_config},
        )

        # Mock an idata object for tests requiring a fitted model
        chains = 2
        draws = 200
        n_purchase_covariates = len(purchase_covariate_cols)
        n_dropout_covariates = len(dropout_covariate_cols)
        mock_fit_dict = {
            "r": rng.normal(cls.true_params["r"], 1e-3, size=(chains, draws)),
            "alpha_scale": rng.normal(
                cls.true_params["alpha_scale"], 1e-3, size=(chains, draws)
            ),
            "a_scale": rng.normal(
                cls.true_params["a_scale"], 1e-3, size=(chains, draws)
            ),
            "b_scale": rng.normal(
                cls.true_params["b_scale"], 1e-3, size=(chains, draws)
            ),
            "purchase_coefficient_alpha": rng.normal(
                cls.true_params["purchase_coefficient_alpha"],
                1e-3,
                size=(chains, draws, n_purchase_covariates),
            ),
            "dropout_coefficient_a": rng.normal(
                cls.true_params["dropout_coefficient_a"],
                1e-3,
                size=(chains, draws, n_dropout_covariates),
            ),
            "dropout_coefficient_b": rng.normal(
                cls.true_params["dropout_coefficient_b"],
                1e-3,
                size=(chains, draws, n_dropout_covariates),
            ),
        }
        mock_fit_with_covariates = az.from_dict(
            mock_fit_dict,
            dims={
                "purchase_coefficient_alpha": ["purchase_covariate"],
                "dropout_coefficient_a": ["dropout_covariate"],
                "dropout_coefficient_b": ["dropout_covariate"],
            },
            coords={
                "purchase_covariate": purchase_covariate_cols,
                "dropout_covariate": dropout_covariate_cols,
            },
        )
        set_model_fit(cls.model_with_covariates, mock_fit_with_covariates)

        cls.model_with_covariates_phi_kappa = BetaGeoModel(
            cls.data,
            model_config=covariate_config,
        )
        # set_model_fit(cls.model_with_covariates_phi_kappa, mock_fit_with_covariates)

        # Create a reference model without covariates
        cls.model_without_covariates = BetaGeoModel(
            data, model_config=non_nested_priors
        )
        mock_fit_without_covariates = az.from_dict(
            {
                "r": mock_fit_dict["r"],
                "alpha": mock_fit_dict["alpha_scale"],
                "a": mock_fit_dict["a_scale"],
                "b": mock_fit_dict["b_scale"],
            }
        )
        set_model_fit(cls.model_without_covariates, mock_fit_without_covariates)

    def test_extract_predictive_covariates(self):
        """Test that alpha/beta computed from the model and helper match."""
        model = self.model_with_covariates
        with model.model:
            trace = pm.sample_posterior_predictive(
                model.idata, var_names=["alpha", "a", "b"]
            ).posterior_predictive
            alpha_model = trace["alpha"]
            a_model = trace["a"]
            b_model = trace["b"]

        variables = model._extract_predictive_variables(data=self.data)
        alpha_helper = variables["alpha"]
        a_helper = variables["a"]
        b_helper = variables["b"]

        np.testing.assert_allclose(alpha_model, alpha_helper)
        np.testing.assert_allclose(a_model, a_helper)
        np.testing.assert_allclose(b_model, b_helper)

        new_data = self.data.assign(
            purchase_cov1=1.0,
            dropout_cov=1.0,
            customer_id=self.data["customer_id"] + 1,
        )
        different_vars = model._extract_predictive_variables(data=new_data)

        different_alpha = different_vars["alpha"]
        assert np.all(
            different_alpha.customer_id.values == alpha_model.customer_id.values + 1
        )
        assert not np.allclose(alpha_model, different_alpha)

        different_a = different_vars["a"]
        assert np.all(different_a.customer_id.values == a_model.customer_id.values + 1)
        assert not np.allclose(a_model, different_a)

        different_b = different_vars["b"]
        assert np.all(different_b.customer_id.values == b_model.customer_id.values + 1)
        assert not np.allclose(b_model, different_b)

    def test_logp(self):
        """Compare logp matches model without covariates when coefficients are zero, and does not otherwise"""
        model_with_covariates = self.model_with_covariates
        model_likelihood_fn = model_with_covariates.model.compile_logp(
            vars=model_with_covariates.model.observed_RVs
        )
        ip = model_with_covariates.model.initial_point()

        model_without_covariates = self.model_without_covariates
        ref_model_likelihood_fn = model_without_covariates.model.compile_logp(
            vars=model_without_covariates.model.observed_RVs
        )
        ref_ip = model_without_covariates.model.initial_point()

        # alpha coefficient: likelihood should go up if purchase covariate1 goes up (coefficient is positive)
        assert model_likelihood_fn(
            ip | dict(purchase_coefficient_alpha=np.array([1.0, 2.0]))
        ) < ref_model_likelihood_fn(ref_ip)

        # a coefficient: likelihood should go up if purchase covariate1 goes up (coefficient is positive)
        assert model_likelihood_fn(
            ip | dict(dropout_coefficient_a=np.array([3.0]))
        ) < ref_model_likelihood_fn(ref_ip)

        # b coefficient: likelihood should go up if purchase covariate1 goes up (coefficient is positive)
        assert model_likelihood_fn(
            ip | dict(dropout_coefficient_b=np.array([3.0]))
        ) < ref_model_likelihood_fn(ref_ip)

        np.testing.assert_allclose(
            model_likelihood_fn(ip), ref_model_likelihood_fn(ref_ip), rtol=5e-4
        )

    def test_expectation_method(self):
        """Test that predictive methods work with covariates"""
        # Higher covariates with positive coefficients -> higher change of death and vice-versa
        # Zero-d covariates should match the vanilla model
        model = self.model_with_covariates

        # Use patterns that are compatible with customer still being alive
        test_data_zero = pd.DataFrame(
            {
                "customer_id": [0, 1, 2],
                "frequency": [12, 14, 10],
                "recency": [19, 18, 18],
                "purchase_cov1": [0, 0, 0],
                "purchase_cov2": [0, 0, 0],
                "dropout_cov": [0, 0, 0],
                "T": [20, 19, 20],
                "future_t": [10, 13, 9],
            }
        )

        # Probability should match model without covariates, when covariates are all zero
        res_zero = model.expected_purchases(test_data_zero).mean(
            ("chain", "draw", "customer_id")
        )
        res_zero_ref = self.model_without_covariates.expected_purchases(
            test_data_zero
        ).mean(("chain", "draw", "customer_id"))
        np.testing.assert_allclose(res_zero, res_zero_ref, rtol=1e-3)

        # Probability should go up if purchase covariate1 goes up (coefficient is positive)
        test_data_high = test_data_zero.assign(purchase_cov1=2.0)
        res_high_purchase1 = model.expected_purchases(test_data_high).mean(
            ("chain", "draw", "customer_id")
        )
        assert res_zero < res_high_purchase1

        # Probability should go down if purchase covariate2 goes up (coefficient is negative)
        test_data_low = test_data_zero.assign(purchase_cov2=1.0)
        res_high_purchase2 = model.expected_purchases(test_data_low).mean(
            ("chain", "draw", "customer_id")
        )
        assert res_zero > res_high_purchase2

        # Probability should go down if dropout covariate goes up (coefficient is positive)
        test_data_low = test_data_zero.assign(dropout_cov=1.0)
        res_high_drop = model.expected_purchases(test_data_low).mean(
            ("chain", "draw", "customer_id")
        )
        assert res_zero > res_high_drop

    def test_distribution_method(self):
        model = self.model_with_covariates

        reps = 30
        test_data_zero = pd.DataFrame(
            {
                "customer_id": range(3 * reps),
                "frequency": [1, 2, 0] * reps,
                "recency": [7, 5, 2] * reps,
                "purchase_cov1": [0, 0, 0] * reps,
                "purchase_cov2": [0, 0, 0] * reps,
                "dropout_cov": [0, 0, 0] * reps,
                "T": [20, 20, 20] * reps,
                "future_t": [2, 3, 4] * reps,
                "n_purchases": [2, 1, 4] * reps,
            }
        )

        # Probability should match model without covariates, when covariates are all zero
        res_zero = model.distribution_new_customer(test_data_zero).mean(
            ("chain", "draw")
        )
        res_zero_ref = self.model_without_covariates.distribution_new_customer(
            test_data_zero
        ).mean(("chain", "draw"))
        np.testing.assert_allclose(
            res_zero["dropout"].mean("customer_id"), res_zero_ref["dropout"], rtol=0.1
        )
        np.testing.assert_allclose(
            res_zero["purchase_rate"].mean("customer_id"),
            res_zero_ref["purchase_rate"],
            rtol=0.1,
        )
        np.testing.assert_allclose(
            res_zero["recency_frequency"].sel(obs_var="recency").mean("customer_id"),
            res_zero_ref["recency_frequency"]
            .sel(obs_var="recency")
            .mean("customer_id"),
            rtol=0.1,
        )
        np.testing.assert_allclose(
            res_zero["recency_frequency"].sel(obs_var="frequency").mean("customer_id"),
            res_zero_ref["recency_frequency"]
            .sel(obs_var="frequency")
            .mean("customer_id"),
            rtol=0.1,
        )

        # Test case where transaction behavior should increase
        test_data_alt = test_data_zero.assign(
            purchase_cov=1.0,  # positive coefficient
            purchase_cov2=-1,  # negative coefficient
            dropout_cov=3,  # positive coefficient
        )
        res_high = model.distribution_new_customer(test_data_alt).mean(
            ("chain", "draw")
        )
        assert (res_zero["purchase_rate"] < res_high["purchase_rate"]).all()
        # Higher dropout covar -> higher dropout proba -> less purchases
        assert (
            res_zero["recency_frequency"].sel(obs_var="frequency").mean()
            > res_high["recency_frequency"].sel(obs_var="frequency").mean()
        )
        assert (
            res_zero["recency_frequency"].sel(obs_var="recency").mean()
            > res_high["recency_frequency"].sel(obs_var="recency").mean()
        )

        assert res_zero["dropout"].std("customer_id") > res_high["dropout"].std(
            "customer_id"
        )

    def test_covariate_model_convergence_a_b(self):
        """Test that we can recover the true parameters with MAP fitting"""
        rng = np.random.default_rng(627)

        # Create synthetic data from "true" params
        default_model = self.model_with_covariates.model
        with pm.do(default_model, self.true_params):
            prior_pred = pm.sample_prior_predictive(
                draws=1, random_seed=rng
            ).prior_predictive
        synthetic_obs = prior_pred["recency_frequency"].squeeze()

        synthetic_data = self.data.assign(
            recency=synthetic_obs.sel(obs_var="recency"),
            frequency=synthetic_obs.sel(obs_var="frequency"),
        )
        # The default parameter priors are very informative. We use something broader here
        custom_priors = {
            "r_prior": Prior("Exponential", scale=10),
            "alpha_prior": Prior("Exponential", scale=10),
            "a_prior": Prior("Exponential", scale=10),
            "b_prior": Prior("Exponential", scale=10),
            "purchase_coefficient_prior": Prior("Normal", mu=0, sigma=4),
            "dropout_coefficient_prior": Prior("Normal", mu=0, sigma=4),
        }
        new_model = BetaGeoModel(
            synthetic_data,
            model_config=self.model_with_covariates.model_config | custom_priors,
        )
        new_model.fit(fit_method="map")

        result = new_model.fit_result
        for var in default_model.free_RVs:
            var_name = var.name
            np.testing.assert_allclose(
                result[var_name].squeeze(("chain", "draw")),
                self.true_params[var_name],
                err_msg=f"Tolerance exceeded for variable {var_name}",
                rtol=0.25,
            )

    def test_covariate_model_convergence_phi_kappa(self):
        """Test that we can recover the true parameters with MAP fitting"""
        rng = np.random.default_rng(627)

        # Create synthetic data from "true" params
        self.model_with_covariates_phi_kappa.build_model()
        default_model = self.model_with_covariates_phi_kappa.model
        with pm.do(default_model, self.true_params):
            prior_pred = pm.sample_prior_predictive(
                draws=1, random_seed=rng
            ).prior_predictive
        synthetic_obs = prior_pred["recency_frequency"].squeeze()

        synthetic_data = self.data.assign(
            recency=synthetic_obs.sel(obs_var="recency"),
            frequency=synthetic_obs.sel(obs_var="frequency"),
        )
        # The default parameter priors are very informative. We use something broader here
        custom_priors = {
            "r_prior": Prior("Exponential", scale=10),
            "alpha_prior": Prior("Exponential", scale=10),
            "phi_dropout_prior": Prior("Uniform", lower=0, upper=1),
            "kappa_dropout_prior": Prior("Pareto", alpha=1, m=1),
            "purchase_coefficient_prior": Prior("Normal", mu=0, sigma=5),
            "dropout_coefficient_prior": Prior("Normal", mu=0, sigma=5),
        }
        new_model = BetaGeoModel(
            synthetic_data,
            model_config=self.model_with_covariates_phi_kappa.model_config
            | custom_priors,
        )
        new_model.fit(method="map")

        result = new_model.fit_result
        for var in default_model.free_RVs:
            # We remove the checks on "phi_dropout", "kappa_droput"
            # because those paramenters are not part of self.true_values in the current setting
            if var.name not in ["phi_dropout", "kappa_dropout"]:
                var_name = var.name
                np.testing.assert_allclose(
                    result[var_name].squeeze(("chain", "draw")),
                    self.true_params[var_name],
                    err_msg=f"Tolerance exceeded for variable {var_name}",
                    rtol=0.2,
                )
