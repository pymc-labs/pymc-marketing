import numpy as np
import pandas as pd
import pymc as pm
import pytest
from lifetimes import BetaGeoBetaBinomFitter

from pymc_marketing.clv.models.beta_geo_beta_binom import BetaGeoBetaBinomModel


@pytest.fixture(scope="module")
def donations() -> pd.DataFrame:
    """
    Load donations benchmark dataset into a Pandas dataframe.
    This dataset aggregates identical customers by count,
    and should be exploded into one customer per row for testing.

    Data source: https://www.brucehardie.com/datasets/
    """

    count_df = pd.read_csv("tests/clv/datasets/donations.csv")

    agg_df = count_df.drop("count", axis=1)

    for row in zip(agg_df.values, count_df["count"]):
        array = np.tile(row[0], (row[1], 1))
        try:
            concat_array = np.concatenate((concat_array, array), axis=0)
        except NameError:
            concat_array = array

    exploded_df = pd.DataFrame(concat_array, columns=["frequency", "recency", "T"])

    return exploded_df


@pytest.mark.skip(reason="Still a WIP")
class TestBetaGeoBetaBinomModel:
    @classmethod
    def setup_class(cls, donations):
        # Set random seed
        cls.rng = np.random.default_rng(34)

        # Parameters
        alpha_true = 1.204
        beta_true = 0.750
        gamma_true = 0.657
        delta_true = 2.783

        # Define testing data
        cls.customer_id = donations.index
        cls.recency = donations["recency"]
        cls.frequency = donations["frequency"]
        cls.T = donations["T"]

        # Initialize model for testing
        cls.model = BetaGeoBetaBinomModel(
            customer_id=cls.customer_id,
            frequency=cls.frequency,
            recency=cls.recency,
            T=cls.T,
        )

        cls.lifetimes_model = BetaGeoBetaBinomFitter

        # Instead of fitting BetaGeoBetaBinomFitter,
        # Load the true model parameters directly
        cls.lifetimes_model._unload_params = (
            alpha_true,
            beta_true,
            gamma_true,
            delta_true,
        )

    def test_unload_params(self):
        pass

    @pytest.mark.parametrize(
        "purchase_hyperprior, churn_hyperprior",
        [
            (None, None),
            (pm.Pareto.dist(alpha=1, m=1.5), pm.Pareto.dist(alpha=1, m=1.5)),
        ],
    )
    def test_model(self, purchase_hyperprior, churn_hyperprior):
        model = BetaGeoBetaBinomModel(
            customer_id=self.customer_id,
            frequency=self.frequency,
            recency=self.recency,
            T=self.T,
            purchase_hyperprior=purchase_hyperprior,
            churn_hyperprior=churn_hyperprior,
        )

        assert isinstance(
            model.model["purchase_hyperprior"].owner.op,
            pm.HalfFlat
            if purchase_hyperprior is None
            else type(purchase_hyperprior.owner.op),
        )
        assert isinstance(
            model.model["churn_hyperprior"].owner.op,
            pm.HalfFlat
            if churn_hyperprior is None
            else type(churn_hyperprior.owner.op),
        )
        assert model.model.eval_rv_shapes() == {
            "purchase_hyperprior": (),
            "purchase_hyperprior_log__": (),
            "churn_hyperprior": (),
            "churn_hyperprior_log__": (),
            "purchase_pool": (),
            "purchase_pool_log__": (),
            "churn_pool": (),
            "churn_pool_log__": (),
            "alpha": (),
            "alpha_log__": (),
            "beta": (),
            "beta_log__": (),
            "gamma": (),
            "gamma_log__": (),
            "delta": (),
            "delta_log__": (),
            "purchase_heterogeneity": (),
            "purchase_heterogeneity_log__": (),
            "churn_heterogeneity": (),
            "churn_heterogeneity_log__": (),
        }

    @pytest.mark.skip(reason="logp tests failing")
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "fit_method, rtol",
        [
            ("mcmc", 0.1),
            ("map", 0.3),
        ],
    )
    def test_model_convergence(self, fit_method, rtol):
        self.model.fit(
            fitting_method=fit_method, chains=1, progressbar=False, random_seed=self.rng
        )

        fit = self.model.fit_result.posterior
        np.testing.assert_allclose(
            [
                fit["alpha"].mean(),
                fit["beta"].mean(),
                fit["gamma"].mean(),
                fit["delta"].mean(),
            ],
            [self.alpha_true, self.beta_true, self.gamma_true, self.delta_true],
            rtol=rtol,
        )

    @pytest.mark.skip(reason="logp tests failing")
    @pytest.mark.parametrize("test_t", [1, 2, 3, 4, 5, 6])
    def test_probability_alive(self, test_t):
        # TODO: Will need to hardcode expected values from http://brucehardie.com/notes/010/
        true_alive = self.lifetimes_model.conditional_probability_alive(
            m_periods_in_future=test_t,
            frequency=self.frequency,
            recency=self.recency,
            n_periods=self.T,
        )

        est_prob_alive = self.model.probability_alive(t=test_t)

        assert est_prob_alive.shape == (1, 1000, len(self.customer_id))
        assert est_prob_alive.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_alive.mean(),
            est_prob_alive.mean(),
            rtol=0.05,
        )

    @pytest.mark.skip(reason="logp tests failing")
    @pytest.mark.parametrize("test_t", [1, 2, 3, 4, 5, 6])
    def test_expected_purchases(self, test_t):
        # TODO: Will need to hardcode expected values from http://brucehardie.com/notes/010/
        true_purchases = (
            self.lifetimes_model.conditional_expected_number_of_purchases_up_to_time(
                m_periods_in_future=test_t,
                frequency=self.frequency,
                recency=self.recency,
                n_periods=self.T,
            )
        )

        est_purchases = self.model.expected_purchases(test_t)

        assert est_purchases.shape == (1, 1000, len(self.customer_id))
        assert est_purchases.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_purchases,
            est_purchases.mean(("chain", "draw")),
            rtol=0.1,
        )

    @pytest.mark.skip(reason="logp tests failing")
    @pytest.mark.parametrize("test_t", [1, 2, 3, 4, 5, 6])
    def test_expected_purchases_new_customer(self, test_t):
        # TODO: Parametrize expected values from http://brucehardie.com/notes/010/
        true_purchases_new_customer = []

        est_purchases_new_customer = self.model.expected_purchases_new_customer(test_t)

        assert est_purchases_new_customer.shape == (1, 1000, 1)
        assert est_purchases_new_customer.dims == ("chain", "draw", "t")

        np.testing.assert_allclose(
            true_purchases_new_customer,
            est_purchases_new_customer.mean(("chain", "draw")),
            rtol=1,
        )

    def test_model_repr(self):
        assert self.fixed_model.__repr__().replace(" ", "") == (
            "BG/BB"
            "\na~HalfFlat()"
            "\nb~HalfFlat()"
            "\nalpha~HalfFlat()"
            "\nr~HalfFlat()"
            "\nlikelihood~Potential(f(r,alpha,b,a))"
        )
