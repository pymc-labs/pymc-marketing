import numpy as np
import pymc as pm
import pytest

# shorter name for black code style formatter
from lifetimes.fitters.beta_geo_fitter import BetaGeoFitter as BGF

from pymc_marketing.clv.distributions import continuous_contractual
from pymc_marketing.clv.models.beta_geo import BetaGeoModel


class TestBetaGeoModel:
    # Hyperparameters
    a_true = 0.8
    b_true = 2.5
    alpha_true = 3
    r_true = 4
    rng = np.random.default_rng(34)

    N = 2000

    @classmethod
    def generate_data(cls, N):
        # Subject level parameters
        p = pm.draw(pm.Beta.dist(cls.a_true, cls.b_true, size=N), random_seed=cls.rng)
        lam = pm.draw(
            pm.Gamma.dist(cls.r_true, cls.alpha_true, size=N), random_seed=cls.rng
        )

        T = pm.draw(
            pm.DiscreteUniform.dist(lower=20, upper=40, size=N), random_seed=cls.rng
        )

        data = continuous_contractual.rng_fn(cls.rng, lam, p, T, 0, size=N)

        return data[..., 0], data[..., 1], 1 - data[..., 2], T

    @classmethod
    def setup_class(cls):
        cls.customer_id = list(range(cls.N))
        cls.recency, cls.frequency, cls.alive, cls.T = cls.generate_data(cls.N)

        # fit the model once for some tests
        cls.fixed_model = BetaGeoModel(
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

    @pytest.mark.parametrize("a_prior", (None, pm.HalfNormal.dist()))
    @pytest.mark.parametrize("b_prior", (None, pm.HalfStudentT.dist(nu=4)))
    @pytest.mark.parametrize("alpha_prior", (None, pm.HalfCauchy.dist(2)))
    @pytest.mark.parametrize("r_prior", (None, pm.Gamma.dist(1, 1)))
    def test_model(self, a_prior, b_prior, alpha_prior, r_prior):
        model = BetaGeoModel(
            customer_id=self.customer_id,
            frequency=self.frequency,
            recency=self.recency,
            T=self.T,
            a_prior=a_prior,
            b_prior=b_prior,
            alpha_prior=alpha_prior,
            r_prior=r_prior,
        )

        assert isinstance(
            model.model["a"].owner.op,
            pm.HalfFlat if a_prior is None else type(a_prior.owner.op),
        )
        assert isinstance(
            model.model["b"].owner.op,
            pm.HalfFlat if b_prior is None else type(b_prior.owner.op),
        )
        assert isinstance(
            model.model["alpha"].owner.op,
            pm.HalfFlat if alpha_prior is None else type(alpha_prior.owner.op),
        )
        assert isinstance(
            model.model["r"].owner.op,
            pm.HalfFlat if r_prior is None else type(r_prior.owner.op),
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

    @pytest.mark.parametrize(
        "N, rtol",
        [
            (500, 0.3),
            (2000, 0.1),
            (10000, 0.055),
        ],
    )
    def test_model_convergence(self, N, rtol):
        recency, frequency, _, T = self.generate_data(N)

        # b parameter has the largest mismatch of the four parameters
        model = BetaGeoModel(
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
            "BG/NBD"
            "\na~HalfFlat()"
            "\nb~HalfFlat()"
            "\nalpha~HalfFlat()"
            "\nr~HalfFlat()"
            "\nlikelihood~Potential(f(r,alpha,b,a))"
        )
