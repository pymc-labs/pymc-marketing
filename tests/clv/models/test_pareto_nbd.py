import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
from lifetimes import ParetoNBDFitter

from pymc_marketing.clv import ParetoNBDModel


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

        # Quickstart dataset is the CDNOW_sample research dataset
        test_data = pd.read_csv("datasets/clv_quickstart.csv")

        cls.customer_id = test_data.index
        cls.frequency = test_data["frequency"]
        cls.recency = test_data["recency"]
        cls.T = test_data["T"]

        # Instantiate model with CDNOW data for testing
        cls.model = ParetoNBDModel(
            customer_id=cls.customer_id,
            frequency=cls.frequency,
            recency=cls.recency,
            T=cls.T,
        )

        # Also instantiate lifetimes model for comparison
        cls.lifetimes_model = ParetoNBDFitter()
        cls.lifetimes_model.params_ = {
            "r": cls.r_true,
            "alpha": cls.alpha_true,
            "s": cls.s_true,
            "beta": cls.beta_true,
        }

    def test_experimental(self):
        with pytest.warns(
            UserWarning,
            match="The Pareto/NBD model is still experimental. Please see code examples in documentation if model fitting issues are encountered.",
        ):
            ParetoNBDModel(
                customer_id=np.array([1, 2, 3]),
                frequency=np.array([3, 4, 7]),
                recency=np.array([10, 20, 30]),
                T=np.array([20, 30, 40]),
            )

    def test_inputs(self):
        with pytest.raises(ValueError, match="Customers must have unique ID labels."):
            ParetoNBDModel(
                customer_id=np.array([1, 2, 2]),
                frequency=np.array([3, 4, 7]),
                recency=np.array([10, 20, 30]),
                T=np.array([20, 30, 40]),
            )

    @pytest.mark.parametrize(
        "r_prior, alpha_prior, s_prior, beta_prior",
        [
            (None, None, None, None),
            (
                pm.Gamma.dist(1, 1),
                pm.Gamma.dist(10, 1),
                pm.Weibull.dist(5, 1),
                pm.Gamma.dist(10, 10),
            ),
        ],
    )
    def test_model(self, r_prior, alpha_prior, s_prior, beta_prior):
        model = ParetoNBDModel(
            customer_id=self.customer_id,
            frequency=self.frequency,
            recency=self.recency,
            T=self.T,
            r_prior=r_prior,
            alpha_prior=alpha_prior,
            s_prior=s_prior,
            beta_prior=beta_prior,
        )

        assert isinstance(
            model.model["r"].owner.op,
            pm.Weibull if r_prior is None else type(r_prior.owner.op),
        )
        assert isinstance(
            model.model["alpha"].owner.op,
            pm.Weibull if alpha_prior is None else type(alpha_prior.owner.op),
        )
        assert isinstance(
            model.model["s"].owner.op,
            pm.Weibull if s_prior is None else type(s_prior.owner.op),
        )
        assert isinstance(
            model.model["beta"].owner.op,
            pm.Weibull if beta_prior is None else type(beta_prior.owner.op),
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

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "fit_method, rtol",
        [
            ("mcmc", 0.1),
            ("map", 0.2),
        ],
    )
    def test_model_convergence(self, fit_method, rtol):
        # Edit priors here for convergence testing
        # Note that None/pm.HalfFlat is extremely slow to converge
        r_prior = pm.Weibull.dist(alpha=2, beta=1)
        alpha_prior = pm.Weibull.dist(alpha=2, beta=10)
        s_prior = pm.Weibull.dist(alpha=2, beta=1)
        beta_prior = pm.Weibull.dist(alpha=2, beta=10)

        model = ParetoNBDModel(
            customer_id=self.customer_id,
            frequency=self.frequency,
            recency=self.recency,
            T=self.T,
            r_prior=r_prior,
            alpha_prior=alpha_prior,
            s_prior=s_prior,
            beta_prior=beta_prior,
        )

        if fit_method == "mcmc":
            with model.model:
                model.fit(
                    fit_method=fit_method,
                    random_seed=self.rng,
                    chains=2,
                    step=pm.Slice(),
                    draws=2000,
                    progressbar=False,
                )
        else:
            model.fit(fit_method=fit_method)

        fit = model.fit_result.posterior
        np.testing.assert_allclose(
            [fit["r"].mean(), fit["alpha"].mean(), fit["s"].mean(), fit["beta"].mean()],
            [self.r_true, self.alpha_true, self.s_true, self.beta_true],
            rtol=rtol,
        )

    def test_model_repr(self):
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
                frequency=self.frequency.values,
                recency=self.recency.values,
                T=self.T.values,
            )
        )

        N = len(self.customer_id)
        chains = 2
        draws = 50
        fake_fit = az.from_dict(
            {
                "r": self.rng.normal(self.r_true, 1e-3, size=(chains, draws)),
                "alpha": self.rng.normal(self.alpha_true, 1e-3, size=(chains, draws)),
                "s": self.rng.normal(self.s_true, 1e-3, size=(chains, draws)),
                "beta": self.rng.normal(self.beta_true, 1e-3, size=(chains, draws)),
            }
        )
        self.model._fit_result = fake_fit

        est_num_purchases = self.model.expected_purchases(test_t)

        assert est_num_purchases.shape == (chains, draws, N)
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

        chains = 2
        draws = 50
        fake_fit = az.from_dict(
            {
                "r": self.rng.normal(self.r_true, 1e-3, size=(chains, draws)),
                "alpha": self.rng.normal(self.alpha_true, 1e-3, size=(chains, draws)),
                "s": self.rng.normal(self.s_true, 1e-3, size=(chains, draws)),
                "beta": self.rng.normal(self.beta_true, 1e-3, size=(chains, draws)),
            }
        )
        self.model._fit_result = fake_fit

        est_purchases_new = self.model.expected_purchases_new_customer(test_t)

        assert est_purchases_new.shape == (chains, draws)
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

        N = len(self.customer_id)
        chains = 2
        draws = 50
        fake_fit = az.from_dict(
            {
                "r": self.rng.normal(self.r_true, 1e-3, size=(chains, draws)),
                "alpha": self.rng.normal(self.alpha_true, 1e-3, size=(chains, draws)),
                "s": self.rng.normal(self.s_true, 1e-3, size=(chains, draws)),
                "beta": self.rng.normal(self.beta_true, 1e-3, size=(chains, draws)),
            }
        )
        self.model._fit_result = fake_fit

        est_prob_alive = self.model.expected_probability_alive()

        assert est_prob_alive.shape == (chains, draws, N)
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

        N = len(self.customer_id)
        chains = 2
        draws = 50
        fake_fit = az.from_dict(
            {
                "r": self.rng.normal(self.r_true, 1e-3, size=(chains, draws)),
                "alpha": self.rng.normal(self.alpha_true, 1e-3, size=(chains, draws)),
                "s": self.rng.normal(self.s_true, 1e-3, size=(chains, draws)),
                "beta": self.rng.normal(self.beta_true, 1e-3, size=(chains, draws)),
            }
        )
        self.model._fit_result = fake_fit

        est_purchases_new_customer = self.model.expected_purchase_probability(
            test_n, test_t
        )

        assert est_purchases_new_customer.shape == (chains, draws, N)
        assert est_purchases_new_customer.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_prob_purchase,
            est_purchases_new_customer.mean(("chain", "draw")),
            rtol=0.001,
        )
