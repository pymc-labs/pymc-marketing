from unittest.mock import patch

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from pymc_marketing.clv.models.gamma_gamma import (
    GammaGammaModel,
    GammaGammaModelIndividual,
)


class BaseTestGammaGammaModel:
    @classmethod
    def setup_class(cls):
        rng = np.random.default_rng(42)

        # Hyperparameters
        p_true = 6.0
        q_true = 4.0
        v_true = 15.0

        # Number of subjects
        N = 500
        # Subject level parameters
        nu_true = pm.draw(pm.Gamma.dist(q_true, v_true, size=N), random_seed=rng)

        # Number of observations per subject
        x = rng.poisson(lam=2, size=N) + 1
        idx = np.repeat(np.arange(0, N), x)

        # Observations
        z = pm.draw(pm.Gamma.dist(p_true, nu_true[idx]), random_seed=rng)

        # Aggregate per subject
        z_mean = pd.DataFrame(data={"z": z, "id": idx}).groupby("id").mean()["z"].values

        cls.rng = rng
        cls.p_true = p_true
        cls.q_true = q_true
        cls.v_true = v_true
        cls.N = N
        cls.z = z
        cls.z_idx = idx
        cls.z_mean = z_mean
        cls.z_mean_idx = list(range(N))
        cls.z_mean_nobs = x


class TestGammaGammaModel(BaseTestGammaGammaModel):
    @pytest.mark.parametrize("p_prior", (None, pm.HalfNormal.dist()))
    @pytest.mark.parametrize("q_prior", (None, pm.HalfStudentT.dist(nu=4)))
    @pytest.mark.parametrize("v_prior", (None, pm.HalfCauchy.dist(2)))
    def test_model(self, p_prior, q_prior, v_prior):
        model = GammaGammaModel(
            customer_id=self.z_mean_idx,
            mean_transaction_value=self.z_mean,
            frequency=self.z_mean_nobs,
            p_prior=p_prior,
            q_prior=q_prior,
            v_prior=v_prior,
        )

        assert isinstance(
            model.model["p"].owner.op,
            pm.HalfFlat if p_prior is None else type(p_prior.owner.op),
        )
        assert isinstance(
            model.model["q"].owner.op,
            pm.HalfFlat if q_prior is None else type(q_prior.owner.op),
        )
        assert isinstance(
            model.model["v"].owner.op,
            pm.HalfFlat if v_prior is None else type(v_prior.owner.op),
        )
        assert model.model.eval_rv_shapes() == {
            "p": (),
            "p_log__": (),
            "q": (),
            "q_log__": (),
            "v": (),
            "v_log__": (),
        }
        assert len(model.model.potentials) == 1
        assert model.model.coords == {
            "customer_id": tuple(range(self.N)),
        }

    @pytest.mark.slow
    def test_model_convergence(self):
        model = GammaGammaModel(
            customer_id=self.z_mean_idx,
            mean_transaction_value=self.z_mean,
            frequency=self.z_mean_nobs,
        )
        model.fit(chains=2, progressbar=False, random_seed=self.rng)
        fit = model.fit_result.posterior
        np.testing.assert_allclose(
            [fit["p"].mean(), fit["q"].mean(), fit["v"].mean()],
            [self.p_true, self.q_true, self.v_true],
            rtol=0.3,
        )

    def test_expected_spend(self):
        p_mean = self.p_true
        q_mean = self.q_true
        v_mean = self.v_true

        model = GammaGammaModel(
            customer_id=self.z_mean_idx,
            mean_transaction_value=self.z_mean,
            frequency=self.z_mean_nobs,
            # Narrow values
            p_prior=pm.Normal.dist(p_mean, 0.01),
            q_prior=pm.Normal.dist(q_mean, 0.01),
            v_prior=pm.Normal.dist(v_mean, 0.01),
        )

        fake_fit = pm.sample_prior_predictive(
            samples=1000, model=model.model, random_seed=self.rng
        )
        fake_fit.add_groups(dict(posterior=fake_fit.prior))
        model._fit_result = fake_fit

        # Force posterior close to empirical mean with many observations
        preds = model.expected_customer_spend(
            customer_id=self.z_mean_idx,
            mean_transaction_value=self.z_mean,
            frequency=1000,
            random_seed=self.rng,
        )
        assert preds.shape == (1, 1000, len(self.z_mean_idx))

        np.testing.assert_allclose(
            preds.mean(("draw", "chain")), self.z_mean, rtol=0.05
        )

        # Force posterior close to group mean with zero observations
        preds = model.expected_customer_spend(
            customer_id=self.z_mean_idx[:10],
            mean_transaction_value=self.z_mean[:10],
            # Force the posterior to be centered around the empirical mean
            frequency=0,
            random_seed=self.rng,
        )
        assert preds.shape == (1, 1000, 10)

        # Recover nu distribution from mean spend
        nu = fake_fit.posterior["p"] / preds

        # nu_mean = q / v
        expected_nu_mean = q_mean / v_mean
        # nu_std = sqrt(q) / v
        expected_nu_std = np.sqrt(q_mean) / v_mean

        np.testing.assert_allclose(
            nu.mean(("draw", "chain")), expected_nu_mean, rtol=0.05
        )
        np.testing.assert_allclose(nu.std(("draw", "chain")), expected_nu_std, rtol=0.1)

    def test_predict_new_customer_spend(self):
        p_mean = 35
        q_mean = 15
        v_mean = 3

        model = GammaGammaModel(
            customer_id=self.z_mean_idx,
            mean_transaction_value=self.z_mean,
            frequency=self.z_mean_nobs,
            # Narrow values
            p_prior=pm.Normal.dist(p_mean, 0.01),
            q_prior=pm.Normal.dist(q_mean, 0.01),
            v_prior=pm.Normal.dist(v_mean, 0.01),
        )

        fake_fit = pm.sample_prior_predictive(
            samples=1000, model=model.model, random_seed=self.rng
        )
        fake_fit.add_groups(dict(posterior=fake_fit.prior))
        model._fit_result = fake_fit

        preds = model.expected_new_customer_spend(n=5, random_seed=self.rng)
        assert preds.shape == (1, 1000, 5)

        # Recover nu distribution from mean spend
        nu = fake_fit.posterior["p"] / preds

        # nu_mean = q / v
        expected_nu_mean = q_mean / v_mean
        # nu_std = sqrt(q) / v
        expected_nu_std = np.sqrt(q_mean) / v_mean

        np.testing.assert_allclose(
            nu.mean(("draw", "chain")), expected_nu_mean, rtol=0.05
        )
        np.testing.assert_allclose(nu.std(("draw", "chain")), expected_nu_std, rtol=0.1)

    def test_model_repr(self):
        model = GammaGammaModel(
            customer_id=self.z_mean_idx,
            mean_transaction_value=self.z_mean,
            frequency=self.z_mean_nobs,
            p_prior=pm.HalfNormal.dist(10),
        )

        assert model.__repr__().replace(" ", "") == (
            "Gamma-GammaModel(MeanTransactions)"
            "\np~N**+(0,10)"
            "\nq~HalfFlat()"
            "\nv~HalfFlat()"
            "\nlikelihood~Potential(f(q,p,v))"
        )


class TestGammaGammaModelIndividual(BaseTestGammaGammaModel):
    @pytest.mark.parametrize("p_prior", (None, pm.HalfNormal.dist(sigma=10)))
    @pytest.mark.parametrize("q_prior", (None, pm.HalfStudentT.dist(nu=4, sigma=10)))
    @pytest.mark.parametrize("v_prior", (None, pm.HalfCauchy.dist(10)))
    def test_model(self, p_prior, q_prior, v_prior):
        model = GammaGammaModelIndividual(
            customer_id=self.z_idx,
            individual_transaction_value=self.z,
            p_prior=p_prior,
            q_prior=q_prior,
            v_prior=v_prior,
        )

        assert isinstance(
            model.model["p"].owner.op,
            pm.HalfFlat if p_prior is None else type(p_prior.owner.op),
        )
        assert isinstance(
            model.model["q"].owner.op,
            pm.HalfFlat if q_prior is None else type(q_prior.owner.op),
        )
        assert isinstance(
            model.model["v"].owner.op,
            pm.HalfFlat if v_prior is None else type(v_prior.owner.op),
        )
        assert isinstance(model.model["nu"].owner.op, pm.Gamma)
        assert model.model.eval_rv_shapes() == {
            "p": (),
            "p_log__": (),
            "q": (),
            "q_log__": (),
            "v": (),
            "v_log__": (),
            "nu": (self.N,),
            "nu_log__": (self.N,),
        }
        assert model.model.coords == {
            "customer_id": tuple(range(self.N)),
            "obs": tuple(range(len(self.z))),
        }

    @pytest.mark.slow
    def test_model_convergence(self):
        model = GammaGammaModelIndividual(
            customer_id=self.z_idx,
            individual_transaction_value=self.z,
        )
        model.fit(chains=2, progressbar=False, random_seed=self.rng)
        fit = model.fit_result.posterior
        np.testing.assert_allclose(
            [fit["p"].mean(), fit["q"].mean(), fit["v"].mean()],
            [self.p_true, self.q_true, self.v_true],
            rtol=0.3,
        )

    @patch(
        "pymc_marketing.clv.models.gamma_gamma.BaseGammaGammaModel.expected_customer_spend"
    )
    def test_expected_spend(self, dummy_method):
        model = GammaGammaModelIndividual(
            customer_id=self.z_idx,
            individual_transaction_value=self.z,
        )

        model.expected_customer_spend(
            customer_id=self.z_idx, individual_transaction_value=self.z, random_seed=123
        )

        dummy_method.assert_called_once()

        kwargs = dummy_method.call_args[1]
        np.testing.assert_array_equal(kwargs["customer_id"].values, self.z_mean_idx)
        np.testing.assert_array_equal(
            kwargs["mean_transaction_value"].values, self.z_mean
        )
        np.testing.assert_array_equal(kwargs["frequency"].values, self.z_mean_nobs)
        assert kwargs["random_seed"] == 123

    def test_model_repr(self):
        model = GammaGammaModelIndividual(
            customer_id=self.z_idx,
            individual_transaction_value=self.z,
            q_prior=pm.HalfNormal.dist(10),
        )

        assert model.__repr__().replace(" ", "") == (
            "Gamma-GammaModel(IndividualTransactions)"
            "\np~HalfFlat()"
            "\nq~N**+(0,10)"
            "\nv~HalfFlat()"
            "\nnu~Gamma(q,f(v))"
            "\nspend~Gamma(p,f(nu))"
        )
