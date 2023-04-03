import arviz as az
import numpy as np
import pymc as pm
import pytest
from lifetimes.fitters.beta_geo_fitter import BetaGeoFitter

from pymc_marketing.clv.distributions import continuous_contractual
from pymc_marketing.clv.models.beta_geo import BetaGeoModel


class TestBetaGeoModel:
    @staticmethod
    def generate_data(a, b, alpha, r, N, rng):
        # Subject level parameters
        p = pm.Beta.dist(a, b, size=N)
        lam = pm.Gamma.dist(r, alpha, size=N)
        T = pm.DiscreteUniform.dist(lower=20, upper=40, size=N)

        # Observations
        data, T = pm.draw(
            [
                continuous_contractual(p=p, lam=lam, T=T, size=N),
                T,
            ],
            random_seed=rng,
        )
        return data[..., 0], data[..., 1], 1 - data[..., 2], T

    @classmethod
    def setup_class(cls):
        cls.N = 500
        cls.a_true = 0.8
        cls.b_true = 2.5
        cls.alpha_true = 3
        cls.r_true = 4
        rng = np.random.default_rng(34)

        cls.customer_id = list(range(cls.N))
        cls.recency, cls.frequency, cls.alive, cls.T = cls.generate_data(
            cls.a_true, cls.b_true, cls.alpha_true, cls.r_true, cls.N, rng=rng
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

    def test_customer_id_warning(self):
        with pytest.raises(
            ValueError,
            match="The BetaGeoModel expects exactly one entry per customer. More than one entry is currently provided per customer id.",
        ):
            BetaGeoModel(
                customer_id=np.asarray([1, 1]),
                frequency=np.asarray([1, 2]),
                recency=np.asarray([1, 2]),
                T=np.asarray([5, 8]),
            )

    @pytest.mark.parametrize(
        "frequency, recency, logp_value",
        [
            (0, 0, -0.59947382),
            (200, 38, 100.7957),
        ],
    )
    def test_numerically_stable_logp(self, frequency, recency, logp_value):
        """
        See Solution #2 on pages 3 and 4 of http://brucehardie.com/notes/027/bgnbd_num_error.pdf
        """
        model = BetaGeoModel(
            customer_id=np.asarray([1]),
            frequency=np.asarray([frequency]),
            recency=np.asarray([recency]),
            T=np.asarray([40]),
            a_prior=pm.Flat.dist(),
            b_prior=pm.Flat.dist(),
            alpha_prior=pm.Flat.dist(),
            r_prior=pm.Flat.dist(),
        )
        pymc_model = model.model
        logp = pymc_model.compile_fn(pymc_model.potentiallogp)

        np.testing.assert_almost_equal(
            logp({"a": 0.80, "b": 2.50, "r": 0.25, "alpha": 4.00}),
            logp_value,
            decimal=5,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "N, fit_method, rtol",
        [
            (500, "mcmc", 0.3),
            (2000, "mcmc", 0.1),
            (10000, "mcmc", 0.055),
            (2000, "map", 0.1),
        ],
    )
    def test_model_convergence(self, N, fit_method, rtol):
        rng = np.random.default_rng(146)
        recency, frequency, _, T = self.generate_data(
            self.a_true, self.b_true, self.alpha_true, self.r_true, N, rng=rng
        )

        # b parameter has the largest mismatch of the four parameters
        model = BetaGeoModel(
            customer_id=list(range(len(frequency))),
            frequency=frequency,
            recency=recency,
            T=T,
        )
        sample_kwargs = dict(random_seed=rng, chains=2) if fit_method == "mcmc" else {}
        model.fit(fit_method=fit_method, progressbar=False, **sample_kwargs)
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
        rng = np.random.default_rng(152)

        N = 100
        # Almost deterministic p = .02, which yield a p(alive) ~ 0.5
        a = 0.02 * 10_000
        b = 0.98 * 10_000
        alpha = 3
        r = 4

        recency, frequency, alive, T = self.generate_data(a, b, alpha, r, N, rng=rng)
        customer_id = list(range(N))

        bg_model = BetaGeoModel(
            customer_id=customer_id,
            frequency=frequency,
            recency=recency,
            T=T,
        )

        fake_fit = az.from_dict(
            {
                "a": rng.normal(a, 1e-3, size=(2, 25)),
                "b": rng.normal(b, 1e-3, size=(2, 25)),
                "alpha": rng.normal(alpha, 1e-3, size=(2, 25)),
                "r": rng.normal(r, 1e-3, size=(2, 25)),
            }
        )
        bg_model._fit_result = fake_fit

        est_prob_alive = bg_model.expected_probability_alive(
            customer_id,
            frequency,
            recency,
            T,
        )

        assert est_prob_alive.shape == (2, 25, N)
        assert est_prob_alive.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            alive.mean(),
            est_prob_alive.mean(),
            rtol=0.05,
        )

    def test_expected_num_purchases(self):
        customer_id = np.arange(10)
        test_t = np.linspace(20, 38, 10)
        test_frequency = np.tile([1, 3, 5, 7, 9], 2)
        test_recency = np.tile([20, 30], 5)
        test_T = np.tile([25, 35], 5)

        bg_model = BetaGeoModel(
            customer_id=customer_id,
            frequency=test_frequency,
            recency=test_recency,
            T=test_T,
        )
        bg_model._fit_result = az.from_dict(
            {
                "a": np.full((2, 5), self.a_true),
                "b": np.full((2, 5), self.b_true),
                "alpha": np.full((2, 5), self.alpha_true),
                "r": np.full((2, 5), self.r_true),
            }
        )

        res_num_purchases = bg_model.expected_num_purchases(
            customer_id,
            test_t,
            test_frequency,
            test_recency,
            test_T,
        )
        assert res_num_purchases.shape == (2, 5, 10)
        assert res_num_purchases.dims == ("chain", "draw", "customer_id")

        # Compare with lifetimes
        lifetimes_bg_model = BetaGeoFitter()
        lifetimes_bg_model.params_ = {
            "a": self.a_true,
            "b": self.b_true,
            "alpha": self.alpha_true,
            "r": self.r_true,
        }
        lifetimes_res_num_purchases = (
            lifetimes_bg_model.conditional_expected_number_of_purchases_up_to_time(
                t=test_t,
                frequency=test_frequency,
                recency=test_recency,
                T=test_T,
            )
        )
        np.testing.assert_allclose(
            res_num_purchases.mean(("chain", "draw")),
            lifetimes_res_num_purchases,
            rtol=0.1,
        )

    def test_expected_num_purchases_new_customer(self):
        customer_id = np.arange(10)
        test_t = np.linspace(20, 38, 10)
        test_frequency = np.tile([1, 3, 5, 7, 9], 2)
        test_recency = np.tile([20, 30], 5)
        test_T = np.tile([25, 35], 5)

        bg_model = BetaGeoModel(
            customer_id=customer_id,
            frequency=test_frequency,
            recency=test_recency,
            T=test_T,
        )
        bg_model._fit_result = az.from_dict(
            {
                "a": np.full((2, 5), self.a_true),
                "b": np.full((2, 5), self.b_true),
                "alpha": np.full((2, 5), self.alpha_true),
                "r": np.full((2, 5), self.r_true),
            }
        )

        res_num_purchases_new_customer = bg_model.expected_num_purchases_new_customer(
            test_t
        )
        assert res_num_purchases_new_customer.shape == (2, 5, 10)
        assert res_num_purchases_new_customer.dims == ("chain", "draw", "t")

        # Compare with lifetimes
        lifetimes_bg_model = BetaGeoFitter()
        lifetimes_bg_model.params_ = {
            "a": self.a_true,
            "b": self.b_true,
            "alpha": self.alpha_true,
            "r": self.r_true,
        }
        lifetimes_res_num_purchases_new_customer = (
            lifetimes_bg_model.expected_number_of_purchases_up_to_time(t=test_t)
        )

        np.testing.assert_allclose(
            res_num_purchases_new_customer.mean(("chain", "draw")),
            lifetimes_res_num_purchases_new_customer,
            rtol=1,
        )

    def test_model_repr(self):
        model = BetaGeoModel(
            customer_id=self.customer_id,
            frequency=self.frequency,
            recency=self.recency,
            T=self.T,
            b_prior=pm.HalfNormal.dist(10),
        )

        assert model.__repr__().replace(" ", "") == (
            "BG/NBD"
            "\na~HalfFlat()"
            "\nb~HalfNormal(0,10)"
            "\nalpha~HalfFlat()"
            "\nr~HalfFlat()"
            "\nlikelihood~Potential(f(r,alpha,b,a))"
        )
