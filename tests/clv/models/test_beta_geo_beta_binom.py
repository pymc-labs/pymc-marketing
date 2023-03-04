import numpy as np
import pandas as pd
import pymc as pm
import pytest
from lifetimes import BetaGeoFitter as BGF

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
    # Hyperparameters
    alpha_true = 0.8
    beta_true = 2.5
    gamma_true = 3
    delta_true = 4
    rng = np.random.default_rng(34)

    N = 2000

    @classmethod
    def setup_class(cls):
        cls.customer_id = list(range(cls.N))
        cls.recency, cls.frequency, cls.alive, cls.T = cls.generate_data(cls.N)

        # fit the model once for some tests
        cls.fixed_model = BetaGeoBetaBinomModel(
            customer_id=cls.customer_id,
            frequency=cls.frequency,
            recency=cls.recency,
            T=cls.T,
        )
        cls.fixed_model.fit(chains=1, progressbar=False, random_seed=cls.rng)

        cls.test_t = np.linspace(20, 38, 10)
        cls.test_frequency = np.tile([1, 3, 5, 7, 9], 2)
        cls.test_recency = np.tile([20, 30], 5)
        cls.test_T = np.tile([25, 35], 5)

        def overwrite_bgf_unload_params(self, *args, **kwargs):
            """
            The methods from BetaGeoFitter rely on a fitted model, i.e. estimates
            for a, b, alpha and r. This function circumvents the need to use a
            fitted model and uses the data-generating parameters for this test
            instead.
            """
            return cls.r_true, cls.alpha_true, cls.a_true, cls.b_true

        BGF._unload_params = overwrite_bgf_unload_params

        BGF.conditional_expected_number_of_purchases_up_to_time = classmethod(
            BGF.conditional_expected_number_of_purchases_up_to_time
        )

        cls.expected_test_num_purchases = (
            BGF.conditional_expected_number_of_purchases_up_to_time(
                t=cls.test_t,
                frequency=cls.test_frequency,
                recency=cls.test_recency,
                T=cls.test_T,
            )
        )

        BGF.expected_number_of_purchases_up_to_time = classmethod(
            BGF.expected_number_of_purchases_up_to_time
        )

        cls.expected_test_num_purchases_new_customer = (
            BGF.expected_number_of_purchases_up_to_time(t=cls.test_t)
        )

    def test_unload_params(self):
        pass

    @pytest.mark.parametrize(
        "purchase_hyperprior", (None, pm.Pareto.dist(alpha=1, m=1.5))
    )
    @pytest.mark.parametrize("churn_hyperprior", (None, pm.Pareto.dist(alpha=1, m=1.5)))
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
            pm.Pareto if a_prior is None else type(purchase_hyperprior.owner.op),
        )
        assert isinstance(
            model.model["churn_hyperprior"].owner.op,
            pm.Pareto if b_prior is None else type(churn_hyperprior.owner.op),
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

    @pytest.mark.parametrize(
        "N, rtol",
        [
            (500, 0.3),
            (2000, 0.1),
            (10000, 0.055),
        ],
    )
    def test_model_convergence(self, donations):
        recency, frequency, _, T = self.generate_data(N)

        # b parameter has the largest mismatch of the four parameters
        model = BetaGeoBetaBinomModel(
            customer_id=list(range(len(frequency))),
            frequency=frequency,
            recency=recency,
            T=T,
        )
        model.fit(chains=1, progressbar=False, random_seed=self.rng)
        fit = model.fit_result.posterior
        np.testing.assert_allclose(
            [fit["a"].mean(), fit["b"].mean(), fit["alpha"].mean(), fit["r"].mean()],
            [self.a_true, self.b_true, self.alpha_true, self.r_true],
            rtol=rtol,
        )

    def test_expected_probability_alive(self):
        """
        The "true" prefix refers to the value obtained using 1) the closed form
        solution and 2) the data-generating parameter values.
        """
        true_prob_alive = self.alive.mean()  # scalar
        est_prob_alive = self.fixed_model.expected_probability_alive(
            self.customer_id,
            self.frequency,
            self.recency,
            self.T,
        )

        assert est_prob_alive.shape == (1, 1000, self.N)
        assert est_prob_alive.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_prob_alive.mean(),
            est_prob_alive.mean(),
            rtol=0.05,
        )

    def test_expected_num_purchases(self):
        """
        TODO: should we combine this test and the one below?
        """
        est_num_purchases = self.fixed_model.expected_num_purchases(
            list(range(20, 40, 2)),
            self.test_t,
            self.test_frequency,
            self.test_recency,
            self.test_T,
        )

        assert est_num_purchases.shape == (1, 1000, 10)
        assert est_num_purchases.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            self.expected_test_num_purchases,
            est_num_purchases.mean(("chain", "draw")),
            rtol=0.1,
        )

    def test_expected_num_purchases_new_customer(self):
        est_num_purchases = self.fixed_model.expected_num_purchases_new_customer(
            self.test_t
        )

        assert est_num_purchases.shape == (1, 1000, 10)
        assert est_num_purchases.dims == ("chain", "draw", "t")

        np.testing.assert_allclose(
            self.expected_test_num_purchases,
            est_num_purchases.mean(("chain", "draw")),
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
