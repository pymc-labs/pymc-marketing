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
from lifetimes import ParetoNBDFitter

from pymc_marketing.clv import ParetoNBDModel
from pymc_marketing.clv.distributions import ParetoNBD
from pymc_marketing.prior import Prior
from tests.conftest import create_mock_fit, set_model_fit


class TestParetoNBDModel:
    @classmethod
    def setup_class(cls):
        # Set random seed
        cls.rng = np.random.default_rng(34)

        # Parameters
        cls.r_true = 0.5534
        cls.alpha_true = 10.5802
        cls.s_true = 0.6061
        cls.beta_true = 11.6562

        # Use Quickstart dataset (the CDNOW_sample research data) for testing
        # TODO: Create a pytest fixture for this
        test_data = pd.read_csv("data/clv_quickstart.csv")
        test_data["customer_id"] = test_data.index

        cls.data = test_data
        cls.customer_id = test_data["customer_id"]
        cls.frequency = test_data["frequency"]
        cls.recency = test_data["recency"]
        cls.T = test_data["T"]

        # Instantiate model with CDNOW data for testing
        cls.model = ParetoNBDModel(cls.data)

        # Also instantiate lifetimes model for comparison
        cls.lifetimes_model = ParetoNBDFitter()
        cls.lifetimes_model.params_ = {
            "r": cls.r_true,
            "alpha": cls.alpha_true,
            "s": cls.s_true,
            "beta": cls.beta_true,
        }

        # Mock an idata object for tests requiring a fitted model
        cls.N = len(cls.customer_id)
        cls.chains = 2
        cls.draws = 50
        mock_fit = create_mock_fit(
            {
                "r": cls.r_true,
                "alpha": cls.alpha_true,
                "s": cls.s_true,
                "beta": cls.beta_true,
            }
        )

        mock_fit(cls.model, chains=cls.chains, draws=cls.draws, rng=cls.rng)
        cls.mock_fit = az.from_dict(
            {
                "r": cls.rng.normal(cls.r_true, 1e-3, size=(cls.chains, cls.draws)),
                "alpha": cls.rng.normal(
                    cls.alpha_true, 1e-3, size=(cls.chains, cls.draws)
                ),
                "s": cls.rng.normal(cls.s_true, 1e-3, size=(cls.chains, cls.draws)),
                "beta": cls.rng.normal(
                    cls.beta_true, 1e-3, size=(cls.chains, cls.draws)
                ),
            }
        )
        set_model_fit(cls.model, cls.mock_fit)

    @pytest.fixture(scope="class")
    def model_config(self):
        return {
            "r": Prior("HalfNormal"),
            "alpha": Prior("HalfStudentT", nu=4),
            "s": Prior("HalfCauchy", beta=2),
            "beta": Prior("Gamma", alpha=1, beta=1),
        }

    @pytest.fixture(scope="class")
    def default_model_config(self):
        return {
            "r": Prior("Weibull", alpha=2, beta=1),
            "alpha": Prior("Weibull", alpha=2, beta=10),
            "s": Prior("Weibull", alpha=2, beta=1),
            "beta": Prior("Weibull", alpha=2, beta=10),
        }

    def test_model(self, model_config, default_model_config):
        for config in (model_config, default_model_config):
            model = ParetoNBDModel(self.data, model_config=config)

            # TODO: This can be removed after build_model() is called internally with __init__
            model.build_model()

            assert isinstance(
                model.model["r"].owner.op,
                pm.Weibull
                if config["r"].distribution == "Weibull"
                else config["r"].pymc_distribution,
            )
            assert isinstance(
                model.model["alpha"].owner.op,
                pm.Weibull
                if config["alpha"].distribution == "Weibull"
                else config["alpha"].pymc_distribution,
            )
            assert isinstance(
                model.model["s"].owner.op,
                pm.Weibull
                if config["s"].distribution == "Weibull"
                else config["s"].pymc_distribution,
            )
            assert isinstance(
                model.model["beta"].owner.op,
                pm.Weibull
                if config["beta"].distribution == "Weibull"
                else config["beta"].pymc_distribution,
            )

            assert model.model.eval_rv_shapes() == {
                "alpha": (),
                "alpha_log__": (),
                "beta": (),
                "beta_log__": (),
                "r": (),
                "r_log__": (),
                "s": (),
                "s_log__": (),
            }

    def test_missing_cols(self):
        data_invalid = self.data.drop(columns="customer_id")

        with pytest.raises(ValueError, match="Required column customer_id missing"):
            ParetoNBDModel(data=data_invalid)

        data_invalid = self.data.drop(columns="frequency")

        with pytest.raises(ValueError, match="Required column frequency missing"):
            ParetoNBDModel(data=data_invalid)

        data_invalid = self.data.drop(columns="recency")

        with pytest.raises(ValueError, match="Required column recency missing"):
            ParetoNBDModel(data=data_invalid)

        data_invalid = self.data.drop(columns="T")

        with pytest.raises(ValueError, match="Required column T missing"):
            ParetoNBDModel(data=data_invalid)

    def test_customer_id_error(self):
        with pytest.raises(
            ValueError, match="Column customer_id has duplicate entries"
        ):
            test_data = pd.DataFrame(
                {
                    "customer_id": np.array([1, 2, 2]),
                    "frequency": np.array([3, 4, 7]),
                    "recency": np.array([10, 20, 30]),
                    "T": np.array([20, 30, 40]),
                }
            )
            ParetoNBDModel(test_data)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "method, rtol",
        [("mcmc", 0.1), ("map", 0.2), ("demz", 0.2)],
    )
    def test_model_convergence(self, method, rtol):
        model = ParetoNBDModel(
            data=self.data,
        )

        model.fit(method=method, progressbar=False)

        fit = model.idata.posterior
        np.testing.assert_allclose(
            [fit["r"].mean(), fit["alpha"].mean(), fit["s"].mean(), fit["beta"].mean()],
            [self.r_true, self.alpha_true, self.s_true, self.beta_true],
            rtol=rtol,
        )

    def test_model_repr(self):
        assert self.model.__repr__().replace(" ", "") == (
            "Pareto/NBD"
            "\nalpha~Weibull(2,10)"
            "\nbeta~Weibull(2,10)"
            "\nr~Weibull(2,1)"
            "\ns~Weibull(2,1)"
            "\nrecency_frequency~ParetoNBD(r,alpha,s,beta,<constant>)"
        )

    @pytest.mark.parametrize("future_t", [1, 3, 6])
    def test_expected_purchases(self, future_t):
        true_purchases = (
            self.lifetimes_model.conditional_expected_number_of_purchases_up_to_time(
                t=future_t,
                frequency=self.frequency,
                recency=self.recency,
                T=self.T,
            )
        )

        data = self.model.data.assign(future_t=future_t)
        est_num_purchases = self.model.expected_purchases(data)

        assert est_num_purchases.shape == (self.chains, self.draws, self.N)
        assert est_num_purchases.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_purchases,
            est_num_purchases.mean(("chain", "draw")),
            rtol=0.001,
        )

    @pytest.mark.parametrize("t", [1, 3, 6])
    def test_expected_purchases_new_customer(self, t):
        true_purchases_new = (
            self.lifetimes_model.expected_number_of_purchases_up_to_time(
                t=t,
            )
        )

        data = pd.DataFrame({"customer_id": [0], "t": [t]})
        est_purchases_new = self.model.expected_purchases_new_customer(data)

        assert est_purchases_new.shape == (self.chains, self.draws, 1)
        assert est_purchases_new.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_purchases_new,
            est_purchases_new.mean(("chain", "draw")),
            rtol=0.001,
        )

    def test_expected_probability_alive(self):
        true_prob_alive = self.lifetimes_model.conditional_probability_alive(
            frequency=self.frequency,
            recency=self.recency,
            T=self.T,
        )

        data = self.model.data
        est_prob_alive = self.model.expected_probability_alive(data)

        assert est_prob_alive.shape == (self.chains, self.draws, self.N)
        assert est_prob_alive.dims == ("chain", "draw", "customer_id")
        np.testing.assert_allclose(
            true_prob_alive,
            est_prob_alive.mean(("chain", "draw")),
            rtol=0.001,
        )

        alt_data = data.assign(future_t=4.5)
        est_prob_alive_t = self.model.expected_probability_alive(alt_data)
        assert est_prob_alive.mean() > est_prob_alive_t.mean()

    @pytest.mark.parametrize("n_purchases, future_t", [(0, 0), (1, 1), (2, 2)])
    def test_expected_purchase_probability(self, n_purchases, future_t):
        true_prob_purchase = (
            self.lifetimes_model.conditional_probability_of_n_purchases_up_to_time(
                n_purchases,
                future_t,
                frequency=self.frequency,
                recency=self.recency,
                T=self.T,
            )
        )

        data = self.model.data.assign(n_purchases=n_purchases, future_t=future_t)
        est_purchases_new_customer = self.model.expected_purchase_probability(data)

        assert est_purchases_new_customer.shape == (self.chains, self.draws, self.N)
        assert est_purchases_new_customer.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_prob_purchase,
            est_purchases_new_customer.mean(("chain", "draw")),
            rtol=0.001,
        )

    @pytest.mark.parametrize("fit_type", ("map", "mcmc", "advi"))
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

        lam_mean = self.r_true / self.alpha_true
        lam_std = np.sqrt(self.r_true) / self.alpha_true
        mu_mean = self.s_true / self.beta_true
        mu_std = np.sqrt(self.s_true) / self.beta_true
        ref_rec, ref_freq = pm.draw(
            ParetoNBD.dist(
                r=self.r_true,
                alpha=self.alpha_true,
                s=self.s_true,
                beta=self.beta_true,
                T=self.T,
            ),
            random_seed=rng,
        ).T

        np.testing.assert_allclose(
            customer_purchase_rate.mean(),
            lam_mean,
            rtol=0.5,
        )
        np.testing.assert_allclose(
            customer_purchase_rate.std(),
            lam_std,
            rtol=0.5,
        )
        np.testing.assert_allclose(customer_dropout.mean(), mu_mean, rtol=0.5)
        np.testing.assert_allclose(customer_dropout.std(), mu_std, rtol=0.5)

        np.testing.assert_allclose(customer_rec.mean(), ref_rec.mean(), rtol=0.5)
        np.testing.assert_allclose(customer_rec.std(), ref_rec.std(), rtol=0.5)

        np.testing.assert_allclose(customer_freq.mean(), ref_freq.mean(), rtol=0.5)
        np.testing.assert_allclose(customer_freq.std(), ref_freq.std(), rtol=0.5)

    def test_save_load_pareto_nbd(self):
        self.model.save("test_model")
        # Testing the valid case.

        loaded_model = ParetoNBDModel.load("test_model")

        # Check if the loaded model is indeed an instance of the class
        assert isinstance(loaded_model, ParetoNBDModel)
        # Check if the loaded data matches with the model data
        pd.testing.assert_frame_equal(
            self.model.data,
            loaded_model.data,
            check_names=False,
        )
        assert self.model.model_config == loaded_model.model_config
        assert self.model.sampler_config == loaded_model.sampler_config
        assert self.model.idata == loaded_model.idata
        os.remove("test_model")

    def test_fit_exception(self, mock_pymc_sample):
        with pytest.warns(
            DeprecationWarning,
            match=(
                "'fit_method' is deprecated and will be removed in a future release. "
                "Use 'method' instead."
            ),
        ):
            self.model.fit(fit_method="mcmc")


class TestParetoNBDModelWithCovariates:
    @classmethod
    def setup_class(cls):
        rng = np.random.default_rng(34)

        cls.true_params = dict(
            r=5.0,
            alpha_scale=10.0,
            s=1.0,
            beta_scale=10.0,
            purchase_coefficient=np.array([1.0, -2.0]),
            dropout_coefficient=np.array([3.0]),
        )

        cls.data = data = pd.read_csv("data/clv_quickstart.csv").iloc[:500]
        data["customer_id"] = data.index

        # Create two purchase covariates and one dropout covariate
        # We standardize so that the coefficient * covariates have similar variance
        N = data.shape[0]
        data["purchase_cov1"] = rng.normal(size=N) / 2
        data["purchase_cov2"] = rng.normal(size=N) / 4
        data["dropout_cov"] = rng.normal(size=N) / 6

        purchase_covariate_cols = ["purchase_cov1", "purchase_cov2"]
        dropout_covariate_cols = ["dropout_cov"]
        covariate_config = dict(
            purchase_covariate_cols=purchase_covariate_cols,
            dropout_covariate_cols=dropout_covariate_cols,
        )
        cls.model_with_covariates = ParetoNBDModel(
            data,
            model_config=covariate_config,
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
            "s": rng.normal(cls.true_params["s"], 1e-3, size=(chains, draws)),
            "beta_scale": rng.normal(
                cls.true_params["beta_scale"], 1e-3, size=(chains, draws)
            ),
            "purchase_coefficient": rng.normal(
                cls.true_params["purchase_coefficient"],
                1e-3,
                size=(chains, draws, n_purchase_covariates),
            ),
            "dropout_coefficient": rng.normal(
                cls.true_params["dropout_coefficient"],
                1e-3,
                size=(chains, draws, n_dropout_covariates),
            ),
        }
        mock_fit_with_covariates = az.from_dict(
            mock_fit_dict,
            dims={
                "purchase_coefficient": ["purchase_covariate"],
                "dropout_coefficient": ["dropout_covariate"],
            },
            coords={
                "purchase_covariate": purchase_covariate_cols,
                "dropout_covariate": dropout_covariate_cols,
            },
        )
        set_model_fit(cls.model_with_covariates, mock_fit_with_covariates)

        # Create a reference model without covariates
        cls.model_without_covariates = ParetoNBDModel(data)
        mock_fit_without_covariates = az.from_dict(
            {
                "r": mock_fit_dict["r"],
                "alpha": mock_fit_dict["alpha_scale"],
                "s": mock_fit_dict["s"],
                "beta": mock_fit_dict["beta_scale"],
            }
        )
        set_model_fit(cls.model_without_covariates, mock_fit_without_covariates)

    def test_extract_predictive_covariates(self):
        """Test that alpha/beta computed from the model and helper match."""
        model = self.model_with_covariates
        with model.model:
            trace = pm.sample_posterior_predictive(
                model.idata, var_names=["alpha", "beta"]
            ).posterior_predictive
            alpha_model = trace["alpha"]
            beta_model = trace["beta"]

        variables = model._extract_predictive_variables(data=self.data)
        alpha_helper = variables["alpha"]
        beta_helper = variables["beta"]

        np.testing.assert_allclose(alpha_model, alpha_helper)
        np.testing.assert_allclose(beta_model, beta_helper)

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

        different_beta = different_vars["beta"]
        assert np.all(
            different_beta.customer_id.values == beta_model.customer_id.values + 1
        )
        assert not np.allclose(beta_model, different_beta)

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

        ip["purchase_coefficient"] = np.array([1.0, 2.0])
        ip["dropout_coefficient"] = np.array([3.0])
        assert model_likelihood_fn(ip) < ref_model_likelihood_fn(ref_ip)

        ip["purchase_coefficient"] = np.array([0.0, 0.0])
        ip["dropout_coefficient"] = np.array([0.0])
        np.testing.assert_allclose(
            model_likelihood_fn(ip),
            ref_model_likelihood_fn(ref_ip),
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
                "recency": [19, 18, 16],
                "purchase_cov1": [0, 0, 0],
                "purchase_cov2": [0, 0, 0],
                "dropout_cov": [0, 0, 0],
                "T": [20, 19, 20],
                "future_t": [10, 13, 15],
            }
        )

        # Probability should match model without covariates, when covariates are all zero
        res_zero = model.expected_purchases(test_data_zero).mean(("chain", "draw"))
        res_zero_ref = self.model_without_covariates.expected_purchases(
            test_data_zero
        ).mean(("chain", "draw"))
        np.testing.assert_allclose(res_zero, res_zero_ref, rtol=1e-3)

        # Probability should go up if purchase covariate1 goes up (coefficient is positive)
        test_data_high = test_data_zero.assign(purchase_cov1=1.0)
        res_high_purchase1 = model.expected_purchases(test_data_high).mean(
            ("chain", "draw")
        )
        assert (res_zero < res_high_purchase1).all()

        # Probability should go down if purchase covariate2 goes up (coefficient is negative)
        test_data_low = test_data_zero.assign(purchase_cov2=1.0)
        res_high_purchase2 = model.expected_purchases(test_data_low).mean(
            ("chain", "draw")
        )
        assert (res_zero > res_high_purchase2).all()

        # Probability should go down if dropout covariate goes up (coefficient is positive)
        test_data_low = test_data_zero.assign(dropout_cov=1.0)
        res_high_drop = model.expected_purchases(test_data_low).mean(("chain", "draw"))
        assert (res_zero > res_high_drop).all()

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
            res_zero["dropout"].mean("customer_id"), res_zero_ref["dropout"], rtol=0.3
        )
        np.testing.assert_allclose(
            res_zero["purchase_rate"].mean("customer_id"),
            res_zero_ref["purchase_rate"],
            rtol=0.3,
        )
        np.testing.assert_allclose(
            res_zero["recency_frequency"].sel(obs_var="recency").mean("customer_id"),
            res_zero_ref["recency_frequency"]
            .sel(obs_var="recency")
            .mean("customer_id"),
            rtol=0.3,
        )
        np.testing.assert_allclose(
            res_zero["recency_frequency"].sel(obs_var="frequency").mean("customer_id"),
            res_zero_ref["recency_frequency"]
            .sel(obs_var="frequency")
            .mean("customer_id"),
            rtol=0.3,
        )

        # Test case where transaction behavior should increase
        test_data_alt = test_data_zero.assign(
            purchase_cov=1.0,  # positive coefficient
            purchase_cov2=-1,  # negative coefficient
            dropout_cov=-1,  # positive coefficient
        )
        res_high = model.distribution_new_customer(test_data_alt).mean(
            ("chain", "draw")
        )
        assert (res_zero["purchase_rate"] < res_high["purchase_rate"]).all()
        assert (res_zero["dropout"] > res_high["dropout"]).all()
        assert (
            res_zero["recency_frequency"].sel(obs_var="frequency")
            < res_high["recency_frequency"].sel(obs_var="frequency")
        ).all()
        assert (
            res_zero["recency_frequency"].sel(obs_var="recency")
            < res_high["recency_frequency"].sel(obs_var="recency")
        ).all()

    def test_covariate_model_convergence(self):
        """Test that we can recover the true parameters with MAP fitting"""
        rng = np.random.default_rng(627)

        # Create synthetic data from "true" params
        default_model = self.model_with_covariates.model
        with pm.do(default_model, self.true_params):
            prior_pred = pm.sample_prior_predictive(
                samples=1, random_seed=rng
            ).prior_predictive
        synthetic_obs = prior_pred["recency_frequency"].squeeze()

        synthetic_data = self.data.assign(
            recency=synthetic_obs.sel(obs_var="recency"),
            frequency=synthetic_obs.sel(obs_var="frequency"),
        )
        # The default parameter priors are very informative. We use something more broad here
        custom_priors = {
            "r": Prior("Exponential", scale=10),
            "alpha": Prior("Exponential", scale=10),
            "s": Prior("Exponential", scale=10),
            "beta": Prior("Exponential", scale=10),
            "purchase_coefficient": Prior("Normal", mu=6, sigma=6),
            "dropout_coefficient": Prior("Normal", mu=3, sigma=3),
        }
        new_model = ParetoNBDModel(
            synthetic_data,
            model_config=self.model_with_covariates.model_config | custom_priors,
        )
        new_model.fit(method="map")

        result = new_model.fit_result
        for var in default_model.free_RVs:
            var_name = var.name
            np.testing.assert_allclose(
                result[var_name].squeeze(("chain", "draw")),
                self.true_params[var_name],
                err_msg=f"Tolerance exceeded for variable {var_name}",
                rtol=0.2,
            )
