import numpy as np
import pandas as pd
import pymc as pm
import pytest
from lifetimes import ParetoNBDFitter

from pymc_marketing.clv.models.pareto_nbd import ParetoNBDModel


class TestParetoNBDModel:
    # TODO: Pytest fixtures cannot be called from within setup_class
    def load_data(self, cdnow_sample):
        return cdnow_sample

    # TODO: lifetimes predictive methods have different formulations
    #       compared to their counterparts in ParetoNBDModel
    @classmethod
    def setup_class(cls):
        # Set random seed
        cls.rng = np.random.default_rng(34)

        # Parameters
        cls.r_true = 0.55
        cls.alpha_true = 10.58
        cls.s_true = 0.61
        cls.beta_true = 11.67

        # Specify priors here for convergence testing
        cls.r_prior = None
        cls.alpha_prior = None
        cls.s_prior = None
        cls.beta_prior = None

        # TODO: To use the fixtures in conftest.py, refactor setup_class into several fixtures
        test_data = pd.read_csv("tests/clv/datasets/cdnow_sample.csv")
        cls.customer_id = test_data.index
        cls.frequency = test_data["frequency"]
        cls.recency = test_data["recency"]
        cls.T = test_data["T"]

        # fit a model with find_MAP() for testing
        cls.model = ParetoNBDModel(
            customer_id=cls.customer_id,
            frequency=cls.frequency,
            recency=cls.recency,
            T=cls.T,
            r_prior=cls.r_prior,
            alpha_prior=cls.alpha_prior,
            s_prior=cls.s_prior,
            beta_prior=cls.beta_prior,
        )
        # model.fit(fitting_method="map", random_seed=self.rng)

        cls.lifetimes_model = ParetoNBDFitter

        cls.lifetimes_model._unload_params = (
            cls.r_true,
            cls.alpha_true,
            cls.s_true,
            cls.beta_true,
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
    def test_model_convergence(self, cdnow_sample):
        # TODO: casting these Pandas Series to numpy arrays with the .values suffix
        #      was recommended due to bugs in lifetimes. try also testing with PD series because
        #      data preprocessing is now handled by pymc_marketing.clv.utils.clv_summary

        model = ParetoNBDModel(
            customer_id=cdnow_sample["customer_id"].values,
            frequency=cdnow_sample["frequency"].values,
            recency=cdnow_sample["recency"].values,
            T=cdnow_sample["T"].values,
        )
        model.fit(chains=1, progressbar=False, random_seed=self.rng)
        fit = model.fit_result.posterior
        np.testing.assert_allclose(
            [fit["r"].mean(), fit["alpha"].mean(), fit["s"].mean(), fit["beta"].mean()],
            [self.r_true, self.alpha_true, self.s_true, self.beta_true],
            rtol=1,
        )

    def test_model_repr(self):
        assert self.model.__repr__().replace(" ", "") == (
            "Pareto/NBD"
            "\nr~HalfFlat()"
            "\nalpha~HalfFlat()"
            "\ns~HalfFlat()"
            "\nbeta~HalfFlat()"
            "\npurchase_rate~Gamma(r,f(alpha))"
            "\nchurn~Gamma(s,f(beta))"
            "\nllike~ParetoNBD(r,alpha,s,beta,<constant>)"
        )

    @pytest.mark.skip(reason="Still a WIP")
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

        assert est_num_purchases.shape == (1, 1000, len(self.customer_id))
        assert est_num_purchases.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_purchases,
            est_num_purchases.mean(("chain", "draw")),
            rtol=0.1,
        )

    @pytest.mark.skip(reason="Still a WIP")
    @pytest.mark.parametrize("test_t", [1, 2, 3, 4, 5, 6])
    def test_expected_purchases_new_customer(self, test_t):
        true_purchases_new = (
            self.lifetimes_model.expected_number_of_purchases_up_to_time(
                t=test_t,
            )
        )

        est_purchases_new = self.model.expected_num_purchases_new_customer(self.test_t)

        assert est_purchases_new.shape == (1, 1000, 1)
        assert est_purchases_new.dims == ("chain", "draw", "t")

        np.testing.assert_allclose(
            true_purchases_new,
            est_purchases_new.mean(("chain", "draw")),
            rtol=1,
        )

    @pytest.mark.skip(reason="Still a WIP")
    @pytest.mark.parametrize("test_t", [1, 2, 3, 4, 5, 6])
    def test_expected_probability_alive(self, test_t):
        true_prob_alive = self.lifetimes_model.expected_number_of_purchases_up_to_time(
            frequency=self.frequency,
            recency=self.recency,
            T=self.T,
        )

        est_prob_alive = self.model.probability_alive(test_t)

        assert est_prob_alive.shape == (1, 1000, len(self.customer_id))
        assert est_prob_alive.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_prob_alive.mean(),
            est_prob_alive.mean(),
            rtol=0.05,
        )

    @pytest.mark.skip(reason="Still a WIP")
    @pytest.mark.parametrize("test_n, test_t", [(0, 0), (1, 1), (2, 2)])
    def test_expected_purchases_probability(self, test_n, test_t):
        true_prob_purchase = self.lifetimes_model.expected_purchases_probability(
            frequency=self.frequency,
            recency=self.recency,
            T=self.T,
        )
        est_purchases_new_customer = (
            self.model.expected_purchases_probability_new_customer(n=test_n, t=test_t)
        )

        assert est_purchases_new_customer.shape == (1, 1000, len(self.customer_id))
        assert est_purchases_new_customer.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_prob_purchase.mean(),
            est_purchases_new_customer.mean(),
            rtol=0.05,
        )
