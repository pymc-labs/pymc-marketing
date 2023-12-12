import os

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
from lifetimes import ParetoNBDFitter

from pymc_marketing.clv import ParetoNBDModel
from pymc_marketing.clv.distributions import ParetoNBD

from pytensor.tensor.elemwise import Elemwise


class TestParetoNBDModel:
    @classmethod
    def setup_class(cls):
        # Set random seed
        cls.rng = np.random.default_rng(34)

        # Parameters
        cls.r_true = 0.4655
        cls.alpha_true = 62.4981
        cls.s_true = 0.2499
        cls.beta_true = 25.1449

        # Use Quickstart dataset (the CDNOW_sample research data) for testing
        # TODO: Create a pytest fixture for this
        test_data = pd.read_csv("tests/clv/datasets/test_clv_covar.csv")

        cls.data = test_data
        cls.customer_id = test_data["customer_id"]
        cls.frequency = test_data["frequency"]
        cls.recency = test_data["recency"]
        cls.T = test_data["T"]

        # Instantiate model with CDNOW data for testing
        cls.model = ParetoNBDModel(cls.data)
        # TODO: This can be removed after build_model() is called internally with __init__
        cls.model.build_model()

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

        cls.model.idata = cls.mock_fit

    @pytest.mark.parametrize(
        "model_config, covar_columns",
        [
            (None, None),
            ("custom", None),
            (None, ["cds_bought", "spent"]),
            ("custom", ["cds_bought", "spent"]),
        ],
    )
    def test_model(self, model_config, covar_columns):
        if model_config == "custom":
            config = {
                "r_prior": {"dist": "HalfNormal", "kwargs": {}},
                "alpha_prior": {"dist": "Weibull", "kwargs": {"alpha": 1, "beta": 1}},
                "s_prior": {"dist": "HalfCauchy", "kwargs": {"beta": 2}},
                "beta_prior": {"dist": "Weibull", "kwargs": {"alpha": 1, "beta": 1}},
                "alpha0_prior": {"dist": "Weibull", "kwargs": {"alpha": 2, "beta": 10}},
                "beta0_prior": {"dist": "Weibull", "kwargs": {"alpha": 2, "beta": 10}},
                "dr_coeff": {"dist": "StudentT", "kwargs": {"nu": 1, "shape": (2,)}},
                "pr_coeff": {"dist": "StudentT", "kwargs": {"nu": 1, "shape": (2,)}},
            }
        else:
            config = model_config

        model = ParetoNBDModel(self.data, covar_columns, covar_columns, config)
        # TODO: This can be removed after build_model() is called internally with __init__
        model.build_model()

        assert isinstance(
            model.model["r"].owner.op,
            pm.HalfNormal if model_config == "custom" else pm.Weibull,
        )
        assert isinstance(
            model.model["alpha"].owner.op,
            Elemwise if covar_columns is not None else pm.Weibull,
        )
        assert isinstance(
            model.model["s"].owner.op,
            pm.HalfCauchy if model_config == "custom" else pm.Weibull,
        )
        assert isinstance(
            model.model["beta"].owner.op,
            Elemwise if covar_columns is not None else pm.Weibull,
        )

        if covar_columns is None:
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
        else:
            assert model.model.eval_rv_shapes() == {
                "alpha0": (),
                "alpha0_log__": (),
                "beta0": (),
                "beta0_log__": (),
                "r": (),
                "r_log__": (),
                "s": (),
                "s_log__": (),
                "dr_coeff": (2,),
                "pr_coeff": (2,),
            }

    def test_missing_customer_id(self):
        # Create a version of the data that's missing the 'customer_id' column
        data_invalid = self.data.drop(columns="customer_id")

        with pytest.raises(KeyError, match="customer_id column is missing from data"):
            ParetoNBDModel(data=data_invalid)

    def test_missing_frequency(self):
        # Create a version of the data that's missing the 'frequency' column
        data_invalid = self.data.drop(columns="frequency")

        with pytest.raises(KeyError, match="frequency column is missing from data"):
            ParetoNBDModel(data=data_invalid)

    def test_missing_recency(self):
        # Create a version of the data that's missing the 'recency' column
        data_invalid = self.data.drop(columns="recency")

        with pytest.raises(KeyError, match="recency column is missing from data"):
            ParetoNBDModel(data=data_invalid)

    def test_missing_T(self):
        # Create a version of the data that's missing the 'T' column
        data_invalid = self.data.drop(columns="T")

        with pytest.raises(KeyError, match="T column is missing from data"):
            ParetoNBDModel(data=data_invalid)

    def test_customer_id_warning(self):
        with pytest.raises(ValueError, match="Customers must have unique ID labels."):
            test_data = pd.DataFrame(
                {
                    "customer_id": np.array([1, 2, 2]),
                    "frequency": np.array([3, 4, 7]),
                    "recency": np.array([10, 20, 30]),
                    "T": np.array([20, 30, 40]),
                }
            )
            ParetoNBDModel(test_data)

    def test_missing_covariates(self):
        with pytest.raises(
            KeyError, match="purchase_channel column is missing from data"
        ):
            ParetoNBDModel(
                data=self.data, pr_covar_columns=["spent", "purchase_channel"]
            )
        with pytest.raises(KeyError, match="discounts column is missing from data"):
            ParetoNBDModel(data=self.data, dr_covar_columns=["spent", "discounts"])

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "fit_method, rtol, covar_columns",
        [
            (None, 0.2, None),
            ("map", 0.2, ["cds_bought", "spent"]),
            ("mcmc", 0.1, None),
        ],
    )
    def test_model_convergence(self, fit_method, rtol, covar_columns):
        model = ParetoNBDModel(
            data=self.data,
            pr_covar_columns=covar_columns,
            dr_covar_columns=covar_columns,
        )
        # TODO: This can be removed after build_model() is called internally with __init__
        model.build_model()

        model.fit(fit_method=fit_method, progressbar=False)

        fit = model.idata.posterior
        np.testing.assert_allclose(
            [fit["r"].mean(), fit["alpha"].mean(), fit["s"].mean(), fit["beta"].mean()],
            [self.r_true, self.alpha_true, self.s_true, self.beta_true],
            rtol=rtol,
        )

    # TODO: This should not be passing as currently written
    @pytest.mark.parametrize(
        "covar_columns",
        [None, ["cds_bought", "spent"]],
    )
    def test_model_repr(self, covar_columns):
        model = ParetoNBDModel(self.data, covar_columns, covar_columns)
        # TODO: This can be removed after build_model() is called internally with __init__
        model.build_model()

        if covar_columns is None:
            assert self.model.__repr__().replace(" ", "") == (
                "Pareto/NBD"
                "\nr~Weibull(2,1)"
                "\nalpha~Weibull(2,10)"
                "\ns~Weibull(2,1)"
                "\nbeta~Weibull(2,10)"
                "\nlikelihood~ParetoNBD(r,alpha,s,beta,<constant>)"
            )
        else:
            assert self.model.__repr__().replace(" ", "") == (
                "Pareto/NBD"
                "\nr~Weibull(2,1)"
                "\nalpha~Weibull(2,10)"
                "\ns~Weibull(2,1)"
                "\nbeta~Weibull(2,10)"
                "\nlikelihood~ParetoNBD(r,alpha,s,beta,<constant>)"
            )

    @pytest.mark.parametrize("test_t", [1, 2, 3, 4, 5, 6])
    def test_expected_purchases(self, test_t):
        true_purchases = (
            self.lifetimes_model.conditional_expected_number_of_purchases_up_to_time(
                t=test_t,
                frequency=self.frequency,
                recency=self.recency,
                T=self.T,
            )
        )

        est_num_purchases = self.model.expected_purchases(test_t)

        assert est_num_purchases.shape == (self.chains, self.draws, self.N)
        assert est_num_purchases.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_purchases,
            est_num_purchases.mean(("chain", "draw")),
            rtol=0.001,
        )

    @pytest.mark.parametrize("test_t", [1, 2, 3, 4, 5, 6])
    def test_expected_purchases_new_customer(self, test_t):
        true_purchases_new = (
            self.lifetimes_model.expected_number_of_purchases_up_to_time(
                t=test_t,
            )
        )

        est_purchases_new = self.model.expected_purchases_new_customer(test_t)

        assert est_purchases_new.shape == (self.chains, self.draws)
        assert est_purchases_new.dims == ("chain", "draw")

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

        est_prob_alive = self.model.expected_probability_alive()

        assert est_prob_alive.shape == (self.chains, self.draws, self.N)
        assert est_prob_alive.dims == ("chain", "draw", "customer_id")
        np.testing.assert_allclose(
            true_prob_alive,
            est_prob_alive.mean(("chain", "draw")),
            rtol=0.001,
        )

        est_prob_alive_t = self.model.expected_probability_alive(future_t=4.5)
        assert est_prob_alive.mean() > est_prob_alive_t.mean()

    @pytest.mark.parametrize("test_n, test_t", [(0, 0), (1, 1), (2, 2)])
    def test_expected_purchase_probability(self, test_n, test_t):
        true_prob_purchase = (
            self.lifetimes_model.conditional_probability_of_n_purchases_up_to_time(
                test_n,
                test_t,
                frequency=self.frequency,
                recency=self.recency,
                T=self.T,
            )
        )

        est_purchases_new_customer = self.model.expected_purchase_probability(
            test_n, test_t, self.data
        )

        assert est_purchases_new_customer.shape == (self.chains, self.draws, self.N)
        assert est_purchases_new_customer.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_prob_purchase,
            est_purchases_new_customer.mean(("chain", "draw")),
            rtol=0.001,
        )

    @pytest.mark.parametrize(
        "fake_fit, T",
        [
            ("map", None),
            ("mcmc", None),
            ("map", np.tile(100, 1000)),
            ("mcmc", np.tile(100, 1000)),
        ],
    )
    def test_posterior_distributions(self, fake_fit, T) -> None:
        rng = np.random.default_rng(42)
        rtol = 0.5
        dim_T = 2357 if T is None else len(T)
        N = 1000

        if T is None:
            T = self.T

        # Reset fit result and expected shapes between tests
        if fake_fit == "map":
            map_fit = az.from_dict(
                {
                    "r": [self.r_true],
                    "alpha": [self.alpha_true],
                    "s": [self.s_true],
                    "beta": [self.beta_true],
                }
            )
            self.model.idata = map_fit
            expected_shape = (1, 1, N)
            expected_pop_dims = (1, 1, dim_T, 2)
        else:
            self.model.idata = self.mock_fit
            expected_shape = (self.chains, self.draws)
            expected_pop_dims = (self.chains, self.draws, dim_T, 2)

        customer_dropout = self.model.distribution_new_customer_dropout(random_seed=rng)
        customer_purchase_rate = self.model.distribution_new_customer_purchase_rate(
            random_seed=rng
        )
        customer_rec_freq = self.model.distribution_customer_population(
            random_seed=rng, T=T
        )

        assert customer_dropout.shape == expected_shape
        assert customer_purchase_rate.shape == expected_shape
        assert customer_rec_freq.shape == expected_pop_dims

        lam = pm.Gamma.dist(alpha=self.r_true, beta=self.alpha_true, size=N)
        mu = pm.Gamma.dist(alpha=self.s_true, beta=self.beta_true, size=N)

        rec_freq = ParetoNBD.dist(
            r=self.r_true,
            alpha=self.alpha_true,
            s=self.s_true,
            beta=self.beta_true,
            T=T,
        )

        np.testing.assert_allclose(
            customer_purchase_rate.mean(),
            pm.draw(lam.mean(), random_seed=rng),
            rtol=rtol,
        )
        np.testing.assert_allclose(
            customer_purchase_rate.var(),
            pm.draw(lam.var(), random_seed=rng),
            rtol=rtol,
        )
        np.testing.assert_allclose(
            customer_dropout.mean(), pm.draw(mu.mean(), random_seed=rng), rtol=rtol
        )
        np.testing.assert_allclose(
            customer_dropout.var(), pm.draw(mu.var(), random_seed=rng), rtol=rtol
        )

        np.testing.assert_allclose(
            customer_rec_freq.mean(),
            pm.draw(rec_freq.mean(), random_seed=rng),
            rtol=rtol,
        )
        np.testing.assert_allclose(
            customer_rec_freq.var(),
            pm.draw(rec_freq.var(), random_seed=self.rng),
            rtol=rtol,
        )

    def test_save_load_pareto_nbd(self):
        # TODO: Create a pytest fixture for this
        test_data = pd.read_csv("datasets/clv_quickstart.csv")
        test_data["customer_id"] = test_data.index
        model = ParetoNBDModel(
            data=test_data,
        )
        model.fit("map")
        model.save("test_model")
        # Testing the valid case.

        loaded_model = ParetoNBDModel.load("test_model")

        # Check if the loaded model is indeed an instance of the class
        assert isinstance(loaded_model, ParetoNBDModel)
        # Check if the loaded data matches with the model data
        np.testing.assert_array_equal(
            loaded_model.data["customer_id"].values, model.data["customer_id"].values
        )
        np.testing.assert_array_equal(
            loaded_model.data["frequency"].values, model.data["frequency"].values
        )
        np.testing.assert_array_equal(
            loaded_model.data["T"].values, model.data["T"].values
        )
        np.testing.assert_array_equal(
            loaded_model.data["recency"].values, model.data["recency"].values
        )
        assert model.model_config == loaded_model.model_config
        assert model.sampler_config == loaded_model.sampler_config
        assert model.idata == loaded_model.idata
        os.remove("test_model")
