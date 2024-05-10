#   Copyright 2024 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from pymc_marketing.clv.models.gamma_gamma import (
    GammaGammaModel,
    GammaGammaModelIndividual,
)
from tests.conftest import set_model_fit


class BaseTestGammaGammaModel:
    @classmethod
    def setup_class(cls):
        rng = np.random.default_rng(18)

        # Hyperparameters
        p_true = 6.0
        q_true = 4.0
        v_true = 15.0

        # Number of subjects
        N = 1000
        # Subject level parameters
        nu_true = pm.draw(pm.Gamma.dist(q_true, v_true, size=N), random_seed=rng)

        # Number of observations per subject
        x = rng.poisson(lam=5, size=N) + 1
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

        cls.data = pd.DataFrame(
            {
                "customer_id": cls.z_mean_idx,
                "mean_transaction_value": cls.z_mean,
                "frequency": cls.z_mean_nobs,
            }
        )

        cls.individual_data = pd.DataFrame(
            {
                "customer_id": cls.z_idx,
                "individual_transaction_value": cls.z,
            }
        )


class TestGammaGammaModel(BaseTestGammaGammaModel):
    def test_missing_columns(self):
        data_invalid = self.data.drop(columns="customer_id")
        with pytest.raises(ValueError, match="Required column customer_id missing"):
            GammaGammaModel(data=data_invalid)

        data_invalid = self.data.drop(columns="frequency")

        with pytest.raises(ValueError, match="Required column frequency missing"):
            GammaGammaModel(data=data_invalid)

        data_invalid = self.data.drop(columns="mean_transaction_value")

        with pytest.raises(
            ValueError, match="Required column mean_transaction_value missing"
        ):
            GammaGammaModel(data=data_invalid)

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {
                "p_prior": {"dist": "HalfNormal"},
                "q_prior": {"dist": "HalfStudentT", "kwargs": {"nu": 4}},
            },
        ],
    )
    def test_model(self, config):
        model = GammaGammaModel(
            data=self.data,
            model_config=config,
        )
        model.build_model()
        assert isinstance(
            model.model["p"].owner.op,
            (pm.HalfFlat if config is None else pm.HalfNormal),
        )
        assert isinstance(
            model.model["q"].owner.op,
            (pm.HalfFlat if config is None else pm.HalfStudentT),
        )
        assert isinstance(
            model.model["v"].owner.op,
            pm.HalfFlat,
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
        rng = np.random.default_rng(13)
        model_config = {
            "p_prior": {"dist": "HalfNormal", "kwargs": {"sigma": 10}},
            "q_prior": {"dist": "HalfNormal", "kwargs": {"sigma": 10}},
            "v_prior": {"dist": "HalfNormal", "kwargs": {"sigma": 10}},
        }
        model = GammaGammaModel(data=self.data, model_config=model_config)
        model.fit(chains=2, progressbar=False, random_seed=rng)
        fit = model.idata.posterior
        np.testing.assert_allclose(
            [fit["p"].mean(), fit["q"].mean(), fit["v"].mean()],
            [self.p_true, self.q_true, self.v_true],
            rtol=0.3,
        )

    @pytest.mark.parametrize("distribution", (True, False))
    def test_spend(self, distribution):
        p_mean = self.p_true
        q_mean = self.q_true
        v_mean = self.v_true
        custom_model_config = {
            # Narrow values
            "p_prior": {"dist": "Normal", "kwargs": {"mu": p_mean, "sigma": 0.01}},
            "q_prior": {"dist": "Normal", "kwargs": {"mu": q_mean, "sigma": 0.01}},
            "v_prior": {"dist": "Normal", "kwargs": {"mu": v_mean, "sigma": 0.01}},
        }
        model = GammaGammaModel(
            data=self.data,
            model_config=custom_model_config,
        )
        model.build_model()
        model.fit(chains=1, progressbar=False, random_seed=self.rng)

        # Force posterior close to empirical mean with many observations
        if distribution:
            preds = model.distribution_customer_spend(
                customer_id=self.z_mean_idx,
                mean_transaction_value=self.z_mean,
                frequency=1000,
                random_seed=self.rng,
            )
        else:
            preds = model.expected_customer_spend(
                customer_id=self.z_mean_idx,
                mean_transaction_value=self.z_mean,
                frequency=1000,
            )
        assert preds.shape == (1, 1000, len(self.z_mean_idx))
        np.testing.assert_allclose(
            preds.mean(("draw", "chain")), self.z_mean, rtol=0.05
        )

        # Closed formula solution for the mean and var of the population spend (eqs 3, 4 from [1])
        expected_preds_mean = p_mean * v_mean / (q_mean - 1)
        expected_preds_std = np.sqrt(
            (p_mean**2 * v_mean**2) / ((q_mean - 1) ** 2 * (q_mean - 2))
        )

        # Force posterior close to group mean with zero observations
        if distribution:
            preds = model.distribution_customer_spend(
                customer_id=self.z_mean_idx[:10],
                mean_transaction_value=self.z_mean[:10],
                frequency=0,
                random_seed=self.rng,
            )
            assert preds.shape == (1, 1000, 10)
            np.testing.assert_allclose(
                preds.mean(("draw", "chain")), expected_preds_mean, rtol=0.1
            )
            np.testing.assert_allclose(
                preds.std(("draw", "chain")), expected_preds_std, rtol=0.25
            )

        else:
            # Force posterior close to group mean with zero observations
            preds = model.expected_customer_spend(
                customer_id=self.z_mean_idx[:10],
                mean_transaction_value=self.z_mean[:10],
                # Force the posterior to be centered around the empirical mean
                frequency=0,
            )
            assert preds.shape == (1, 1000, 10)
            np.testing.assert_allclose(
                preds.mean(("draw", "chain")), expected_preds_mean, rtol=0.025
            )

    @pytest.mark.parametrize("distribution", (True, False))
    def test_new_customer_spend(self, distribution):
        p_mean = 35
        q_mean = 15
        v_mean = 3
        test_seed = np.random.default_rng(1234)
        custom_model_config = {
            # Narrow values
            "p_prior": {"dist": "Normal", "kwargs": {"mu": p_mean, "sigma": 0.01}},
            "q_prior": {"dist": "Normal", "kwargs": {"mu": q_mean, "sigma": 0.01}},
            "v_prior": {"dist": "Normal", "kwargs": {"mu": v_mean, "sigma": 0.01}},
        }
        model = GammaGammaModel(
            data=self.data,
            model_config=custom_model_config,
        )
        model.build_model()
        fake_fit = pm.sample_prior_predictive(
            samples=1000, model=model.model, random_seed=self.rng
        )
        set_model_fit(model, fake_fit.prior)
        # Closed formula solution for the mean and var of the population spend (eqs 3, 4 from [1])
        expected_preds_mean = p_mean * v_mean / (q_mean - 1)
        expected_preds_std = np.sqrt(
            (p_mean**2 * v_mean**2) / ((q_mean - 1) ** 2 * (q_mean - 2))
        )

        if distribution:
            preds = model.distribution_new_customer_spend(n=5, random_seed=test_seed)
            assert preds.shape == (1, 1000, 5)
            np.testing.assert_allclose(
                preds.mean(("draw", "chain")), expected_preds_mean, rtol=0.1
            )
            np.testing.assert_allclose(
                preds.std(("draw", "chain")), expected_preds_std, rtol=0.25
            )
        else:
            preds = model.expected_new_customer_spend()
            assert preds.shape == (1, 1000)
            np.testing.assert_allclose(
                preds.mean(("draw", "chain")), expected_preds_mean, rtol=0.05
            )

    def test_model_repr(self):
        custom_model_config = {
            "p_prior": {"dist": "HalfNormal", "kwargs": {"sigma": 10}}
        }
        model = GammaGammaModel(data=self.data, model_config=custom_model_config)
        model.build_model()

        assert model.__repr__().replace(" ", "") == (
            "Gamma-GammaModel(MeanTransactions)"
            "\np~HalfNormal(0,10)"
            "\nq~HalfFlat()"
            "\nv~HalfFlat()"
            "\nlikelihood~Potential(f(q,p,v))"
        )

    def test_save_load(self):
        model = GammaGammaModel(
            data=self.data,
        )
        model.build_model()
        model.fit("map", maxeval=1)
        model.save("test_model")
        # Testing the valid case.

        model2 = GammaGammaModel.load("test_model")

        # Check if the loaded model is indeed an instance of the class
        assert isinstance(model, GammaGammaModel)
        # Check if the loaded data matches with the model data
        pd.testing.assert_frame_equal(model.data, model2.data, check_names=False)
        assert model.model_config == model2.model_config
        assert model.sampler_config == model2.sampler_config
        assert model.idata == model2.idata
        os.remove("test_model")


class TestGammaGammaModelIndividual(BaseTestGammaGammaModel):
    def test_missing_columns(self):
        # Create a version of the data that's missing the 'customer_id' column
        data_invalid = self.individual_data.drop(columns="customer_id")
        with pytest.raises(ValueError, match="Required column customer_id missing"):
            GammaGammaModelIndividual(data=data_invalid)

        data_invalid = self.individual_data.drop(columns="individual_transaction_value")

        with pytest.raises(
            ValueError, match="Required column individual_transaction_value missing"
        ):
            GammaGammaModelIndividual(data=data_invalid)

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {
                "p_prior": {"dist": "HalfNormal"},
                "q_prior": {"dist": "HalfStudentT", "kwargs": {"nu": 4}},
            },
        ],
    )
    def test_model(self, config):
        model = GammaGammaModelIndividual(
            data=self.individual_data,
            model_config=config,
        )
        model.build_model()
        assert isinstance(
            model.model["p"].owner.op,
            pm.HalfFlat if config is None else pm.HalfNormal,
        )
        assert isinstance(
            model.model["q"].owner.op,
            pm.HalfFlat if config is None else pm.HalfStudentT,
        )
        assert isinstance(model.model["v"].owner.op, pm.HalfFlat)
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
        rng = np.random.default_rng(13)
        model = GammaGammaModelIndividual(data=self.individual_data)
        model.fit(chains=2, progressbar=False, random_seed=rng)
        fit = model.idata.posterior
        np.testing.assert_allclose(
            [fit["p"].mean(), fit["q"].mean(), fit["v"].mean()],
            [self.p_true, self.q_true, self.v_true],
            rtol=0.3,
        )

    @patch(
        "pymc_marketing.clv.models.gamma_gamma.BaseGammaGammaModel.distribution_customer_spend"
    )
    def test_distribution_spend(self, dummy_method):
        model = GammaGammaModelIndividual(
            data=self.individual_data,
        )
        model.build_model()
        model.distribution_customer_spend(
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

    @patch(
        "pymc_marketing.clv.models.gamma_gamma.BaseGammaGammaModel.expected_customer_spend"
    )
    def test_expected_spend(self, dummy_method):
        model = GammaGammaModelIndividual(self.individual_data)

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
        custom_model_config = {
            "q_prior": {"dist": "HalfNormal", "kwargs": {"sigma": 10}}
        }
        model = GammaGammaModelIndividual(
            data=self.individual_data,
            model_config=custom_model_config,
        )
        model.build_model()

        assert model.__repr__().replace(" ", "") == (
            "Gamma-GammaModel(IndividualTransactions)"
            "\np~HalfFlat()"
            "\nq~HalfNormal(0,10)"
            "\nv~HalfFlat()"
            "\nnu~Gamma(q,f(v))"
            "\nspend~Gamma(p,f(nu))"
        )

    def test_save_load(self):
        model = GammaGammaModelIndividual(
            data=self.individual_data,
        )
        model.build_model()
        model.fit("map", maxeval=1)
        model.save("test_model")
        # Testing the valid case.

        model2 = GammaGammaModelIndividual.load("test_model")

        # Check if the loaded model is indeed an instance of the class
        assert isinstance(model, GammaGammaModelIndividual)
        # Check if the loaded data matches with the model data
        pd.testing.assert_frame_equal(model.data, model2.data, check_names=False)
        assert model.model_config == model2.model_config
        assert model.sampler_config == model2.sampler_config
        assert model.idata == model2.idata
        os.remove("test_model")
