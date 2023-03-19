import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
from lifetimes import ParetoNBDFitter

from pymc_marketing.clv.models.pareto_nbd import ParetoNBDModel


class TestParetoNBDModel:
    @classmethod
    def setup_class(cls):
        # Set random seed
        cls.rng = np.random.default_rng(34)

        # Parameters
        cls.r_true = 0.55
        cls.alpha_true = 10.58
        cls.s_true = 0.61
        cls.beta_true = 11.67

        test_data = pd.read_csv("tests/clv/datasets/cdnow_sample.csv")
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

        # Also fit the same equivalent lifetimes model to the same dataset for comparison
        cls.lifetimes_model = ParetoNBDFitter().fit(
            frequency=cls.frequency.values, recency=cls.recency.values, T=cls.T.values
        )

    @pytest.mark.parametrize(
        "r_prior, alpha_prior, s_prior, beta_prior",
        [
            (None, None, None, None),
            (
                pm.Gamma.dist(1, 1),
                pm.Gamma.dist(1, 1),
                pm.Gamma.dist(1, 1),
                pm.Gamma.dist(1, 1),
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
            pm.HalfFlat if r_prior is None else type(r_prior.owner.op),
        )
        assert isinstance(
            model.model["alpha"].owner.op,
            pm.HalfFlat if alpha_prior is None else type(alpha_prior.owner.op),
        )
        assert isinstance(
            model.model["s"].owner.op,
            pm.HalfFlat if s_prior is None else type(s_prior.owner.op),
        )
        assert isinstance(
            model.model["beta"].owner.op,
            pm.HalfFlat if beta_prior is None else type(beta_prior.owner.op),
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
            "churn": (),
            "churn_log__": (),
            "purchase_rate": (),
            "purchase_rate_log__": (),
        }

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "fit_method, rtol",
        [
            ("mcmc", 0.3),
            ("map", 0.1),
        ],
    )
    def test_model_convergence(self, cdnow_sample, fit_method, rtol):
        # # Specify priors here for convergence testing
        # # Note that None/pm.HalfFlat is extremely slow to converge
        r_prior = pm.Weibull.dist(alpha=10, beta=1)
        alpha_prior = pm.Weibull.dist(alpha=10, beta=10)
        s_prior = pm.Weibull.dist(alpha=10, beta=1)
        beta_prior = pm.Weibull.dist(alpha=10, beta=10)

        # TODO: casting these Pandas Series to numpy arrays with the .values suffix
        #      was recommended due to bugs in lifetimes. try also testing with PD series because
        #      data preprocessing is now handled by pymc_marketing.clv.utils.clv_summary
        model = ParetoNBDModel(
            customer_id=cdnow_sample["customer_id"].values,
            frequency=cdnow_sample["frequency"].values,
            recency=cdnow_sample["recency"].values,
            T=cdnow_sample["T"].values,
            r_prior=r_prior,
            alpha_prior=alpha_prior,
            s_prior=s_prior,
            beta_prior=beta_prior,
        )
        sample_kwargs = (
            dict(random_seed=self.rng, chains=2, step=pm.Slice())
            if fit_method == "mcmc"
            else {}
        )
        model.fit(fit_method=fit_method, progressbar=False, **sample_kwargs)
        fit = model.fit_result.posterior
        np.testing.assert_allclose(
            [fit["r"].mean(), fit["alpha"].mean(), fit["s"].mean(), fit["beta"].mean()],
            [self.r_true, self.alpha_true, self.s_true, self.beta_true],
            rtol=rtol,
        )

    def test_model_repr(self):
        assert self.model.__repr__().replace(" ", "") == (
            "Pareto/NBD"
            "\nr~Weibull(10,1)"
            "\nalpha~Weibull(10,10)"
            "\ns~Weibull(10,1)"
            "\nbeta~Weibull(10,10)"
            "\npurchase_rate~Gamma(r,f(alpha))"
            "\nchurn~Gamma(s,f(beta))"
            "\nllike~ParetoNBD(r,alpha,s,beta,<constant>)"
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

        est_num_purchases = self.model.expected_purchases(test_t)

        assert est_num_purchases.shape == (1, 1000, len(self.customer_id))
        assert est_num_purchases.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_purchases,
            est_num_purchases.mean(("chain", "draw", "customer_id")),
            rtol=0.1,
        )

    @pytest.mark.parametrize("test_t", [1, 2, 3, 4, 5, 6])
    def test_expected_purchases_new_customer(self, test_t):
        true_purchases_new = (
            self.lifetimes_model.expected_number_of_purchases_up_to_time(
                t=test_t,
            )
        )

        est_purchases_new = self.model.expected_purchases_new_customer(test_t)

        assert est_purchases_new.shape == (1, 1000, 1)
        assert est_purchases_new.dims == ("chain", "draw", "t")

        np.testing.assert_allclose(
            true_purchases_new,
            est_purchases_new.mean(("chain", "draw")),
            rtol=1,
        )

    def test_probability_alive(self):
        true_prob_alive = self.lifetimes_model.conditional_probability_alive(
            frequency=self.frequency.values,
            recency=self.recency.values,
            T=self.T.values,
        )

        fake_fit = az.from_dict(
            {
                "r": self.rng.normal(self.r_true, 1e-3, size=(2, 25)),
                "alpha": self.rng.normal(self.alpha_true, 1e-3, size=(2, 25)),
                "s": self.rng.normal(self.s_true, 1e-3, size=(2, 25)),
                "beta": self.rng.normal(self.beta_true, 1e-3, size=(2, 25)),
            }
        )
        self.model._fit_result = fake_fit

        est_prob_alive = self.model.probability_alive()
        est_prob_alive_t = self.model.probability_alive(t=5)

        assert est_prob_alive.shape == (2, 25, len(self.customer_id))
        assert est_prob_alive.dims == ("chain", "draw", "customer_id")
        assert est_prob_alive.mean() > est_prob_alive_t.mean()

        np.testing.assert_allclose(
            true_prob_alive.mean(),
            est_prob_alive.mean(),
            rtol=0.05,
        )

    @pytest.mark.parametrize("test_n, test_t", [(0, 0), (1, 1), (2, 2)])
    def test_purchase_probability(self, test_n, test_t):
        true_prob_purchase = (
            self.lifetimes_model.conditional_probability_of_n_purchases_up_to_time(
                test_n,
                test_t,
                frequency=self.frequency.values,
                recency=self.recency.values,
                T=self.T.values,
            )
        )
        est_purchases_new_customer = self.model.purchase_probability(test_n, test_t)

        assert est_purchases_new_customer.shape == (1, 1000, len(self.customer_id))
        assert est_purchases_new_customer.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_prob_purchase.mean(),
            est_purchases_new_customer.mean(),
            rtol=0.05,
        )
