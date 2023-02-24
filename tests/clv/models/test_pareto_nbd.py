import numpy as np
import pymc as pm
import pytest
from lifetimes import ParetoNBDFitter as PNF

from pymc_marketing.clv.distributions import ParetoNBD
from pymc_marketing.clv.models.pareto_nbd import ParetoNBDModel


@pytest.mark.skip(reason="Still a WIP")
class TestParetoNBDModel:
    # Parameters
    # TODO: True params will change depending on usage of cdnow_sample or cdnow_full
    r_true = 0.55
    alpha_true = 10.58
    s_true = 0.61
    beta_true = 11.67

    rng = np.random.default_rng(34)

    test_t = 30

    # TODO: lifetimes.PNF predictive methods have different formulations
    #       compared to their counterparts in ParetoNBDModel
    @classmethod
    def setup_class(cls, cdnow_sample, cdnow_full):
        # TODO: Is this wrapper even needed, or can the params just be set directly?
        def pnf_params_wrapper(self, *args, **kwargs):
            """
            The methods from ParetoNBDFitter rely on a fitted model, i.e. estimates
            for r, alpha, s, and beta parameters. This function circumvents model fitting
            by assigning the fitted parameters from CDNOW_sample.csv directly.
            """
            return cls.r_true, cls.alpha_true, cls.s_true, cls.beta_true

        PNF._unload_params = pnf_params_wrapper

        PNF.conditional_expected_number_of_purchases_up_to_time = classmethod(
            PNF.conditional_expected_number_of_purchases_up_to_time
        )

        cls.expected_test_num_purchases = (
            PNF.conditional_expected_number_of_purchases_up_to_time(
                t=cdnow_sample.test_t,
                frequency=cdnow_sample.test_frequency,
                recency=cdnow_sample.test_recency,
                T=cdnow_sample.test_T,
            )
        )

        PNF.expected_number_of_purchases_up_to_time = classmethod(
            PNF.expected_number_of_purchases_up_to_time
        )

        cls.expected_test_num_purchases_new_customer = (
            PNF.expected_number_of_purchases_up_to_time(t=cls.test_t)
        )

    @pytest.mark.parametrize("r_prior", (None, pm.Gamma.dist(1, 1)))
    @pytest.mark.parametrize("alpha_prior", (None, pm.Gamma.dist(1, 1)))
    @pytest.mark.parametrize("s_prior", (None, pm.Gamma.dist(1, 1)))
    @pytest.mark.parametrize("beta_prior", (None, pm.Gamma.dist(1, 1)))
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
            "r": (),
            "r_log__": (),
            "alpha": (),
            "alpha_log__": (),
            "s": (),
            "s_log__": (),
            "beta": (),
            "beta_log__": (),
        }

    # TODO: Instead of fitting both datasets, create a class property instead to switch between datasets?
    # TODO: Why can't the cdnow_sample and cdnow_full fixtures be called in this parametrize?
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "dataset, rtol",
        (
            [None, 0.3],
            [None, 0.1],
        ),
    )
    def test_model_convergence(self, dataset, rtol):
        # fit the model to CDNOW_sample with N=2,357 customers
        # fit the model to CDNOW_full with N=23,570 customers
        # TODO: casting these Pandas Series to numpy arrays with the .values suffix
        #      was recommended due to bugs in lifetimes. try also testing with PD series because
        #      data preprocessing is now handled by pymc_marketing.clv.utils.clv_summary
        model = ParetoNBDModel(
            customer_id=dataset["customer_id"].values,
            frequency=dataset["frequency"].values,
            recency=dataset["recency"].values,
            T=dataset["T"].values,
        )
        model.fit(chains=1, progressbar=False, random_seed=self.rng)
        fit = model.fit_result.posterior
        np.testing.assert_allclose(
            [fit["r"].mean(), fit["alpha"].mean(), fit["s"].mean(), fit["beta"].mean()],
            [self.r_true, self.alpha_true, self.s_true, self.beta_true],
            rtol=rtol,
        )

    def test_model_repr(self):
        assert self.fixed_model.__repr__().replace(" ", "") == (
            "BG/NBD"
            "\nr~HalfFlat()"
            "\nalpha~HalfFlat()"
            "\ns~HalfFlat()"
            "\nbeta~HalfFlat()"
            "\nlikelihood~Potential(f(r,alpha,b,a))"
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
        est_num_purchases_new = self.fixed_model.expected_num_purchases_new_customer(
            self.test_t
        )

        assert est_num_purchases_new.shape == (1, 1000, 10)
        assert est_num_purchases_new.dims == ("chain", "draw", "t")

        np.testing.assert_allclose(
            self.expected_test_num_purchases,
            est_num_purchases.mean(("chain", "draw")),
            rtol=1,
        )

    def test_expected_probability_alive(self):
        """
        The "true" prefix refers to the value obtained using 1) the closed form
        solution and 2) the data-generating parameter values.
        """

        est_prob_alive = self.model.expected_probability_alive(
            self.customer_id,
            self.frequency,
            self.recency,
            self.T,
        )

        assert est_prob_alive.shape == (1, 1000, len(cls.dataset))
        assert est_prob_alive.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            self.true_prob_alive.mean(),
            est_prob_alive.mean(),
            rtol=0.05,
        )

    def test_expected_purchases_probability_new_customer(self):
        est_purchases_new_customer = (
            self.model.expected_purchases_probability_new_customer(self.n, self.test_t)
        )

        assert est_purchases_new_customer.shape == (1, 1000, len(cls.dataset))
        assert est_purchases_new_customer.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            self.expected_purchases_new_customer.mean(),
            est_purchases_new_customer.mean(),
            rtol=0.05,
        )

    def test_expected_purchases_probability(self):
        est_purchase_probability = self.model.expected_purchases_probability(
            self.n,
            self.t,
            self.customer_id,
            self.frequency,
            self.recency,
            self.T,
        )

        assert est_purchase_probability.shape == (1, 1000, len(cls.dataset))
        assert est_purchase_probability.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            self.expected_purchase_probability.mean(),
            est_purchase_probability.mean(),
            rtol=0.05,
        )

    def test_expected_future_probability_alive(self):
        est_prob_alive_future = self.model.expected_future_probability_alive(
            self.t,
            self.customer_id,
            self.frequency,
            self.recency,
            self.T,
        )

        assert est_prob_alive_future.shape == (1, 1000, len(cls.dataset))
        assert est_prob_alive_future.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            self.true_prob_alive.mean(),
            est_prob_alive_future.mean(),
            rtol=0.05,
        )
