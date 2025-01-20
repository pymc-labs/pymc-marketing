#   Copyright 2022 - 2025 The PyMC Labs Developers
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

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from lifetimes.fitters.modified_beta_geo_fitter import ModifiedBetaGeoFitter

from pymc_marketing.clv.models.modified_beta_geo import ModifiedBetaGeoModel
from pymc_marketing.prior import Prior
from tests.conftest import create_mock_fit, mock_sample


class TestModifiedBetaGeoModel:
    @classmethod
    def setup_class(cls):
        # Set random seed
        cls.rng = np.random.default_rng(34)

        # parameters
        cls.a_true = 0.891
        cls.b_true = 1.614
        cls.alpha_true = 6.183
        cls.r_true = 0.525

        # Use Quickstart dataset (the CDNOW_sample research data) for testing
        test_data = pd.read_csv("data/clv_quickstart.csv")
        test_data["customer_id"] = test_data.index

        cls.data = test_data
        cls.customer_id = test_data["customer_id"]
        cls.frequency = test_data["frequency"]
        cls.recency = test_data["recency"]
        cls.T = test_data["T"]

        # Instantiate model with CDNOW data for testing
        cls.model = ModifiedBetaGeoModel(cls.data)
        cls.model.build_model()

        # Also instantiate lifetimes model for comparison
        cls.lifetimes_model = ModifiedBetaGeoFitter()
        cls.lifetimes_model.params_ = {
            "a": cls.a_true,
            "b": cls.b_true,
            "alpha": cls.alpha_true,
            "r": cls.r_true,
        }

        # Mock an idata object for tests requiring a fitted model
        cls.N = len(cls.customer_id)
        cls.chains = 2
        cls.draws = 50

        mock_fit = create_mock_fit(
            {
                "a": cls.a_true,
                "b": cls.b_true,
                "alpha": cls.alpha_true,
                "r": cls.r_true,
            }
        )

        mock_fit(cls.model, cls.chains, cls.draws, cls.rng)

    @pytest.fixture(scope="class")
    def model_config(self):
        return {
            "a_prior": Prior("HalfNormal"),
            "b_prior": Prior("HalfStudentT", nu=4),
            "alpha_prior": Prior("HalfCauchy", beta=2),
            "r_prior": Prior("Gamma", alpha=1, beta=1),
        }

    @pytest.fixture(scope="class")
    def default_model_config(self):
        return {
            "a_prior": Prior("HalfFlat"),
            "b_prior": Prior("HalfFlat"),
            "alpha_prior": Prior("HalfFlat"),
            "r_prior": Prior("HalfFlat"),
        }

    def test_model(self, model_config, default_model_config):
        for config in (model_config, default_model_config):
            model = ModifiedBetaGeoModel(
                data=self.data,
                model_config=config,
            )
            model.build_model()
            assert isinstance(
                model.model["a"].owner.op,
                pm.HalfFlat
                if config["a_prior"].distribution == "HalfFlat"
                else config["a_prior"].pymc_distribution,
            )
            assert isinstance(
                model.model["b"].owner.op,
                pm.HalfFlat
                if config["b_prior"].distribution == "HalfFlat"
                else config["b_prior"].pymc_distribution,
            )
            assert isinstance(
                model.model["alpha"].owner.op,
                pm.HalfFlat
                if config["alpha_prior"].distribution == "HalfFlat"
                else config["alpha_prior"].pymc_distribution,
            )
            assert isinstance(
                model.model["r"].owner.op,
                pm.HalfFlat
                if config["r_prior"].distribution == "HalfFlat"
                else config["r_prior"].pymc_distribution,
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
        "missing_column",
        ["customer_id", "frequency", "recency", "T"],
    )
    def test_missing_cols(self, missing_column):
        data_invalid = self.data.drop(columns=missing_column)

        with pytest.raises(
            ValueError, match=f"Required column {missing_column} missing"
        ):
            ModifiedBetaGeoModel(data=data_invalid)

    def test_customer_id_duplicate(self):
        with pytest.raises(
            ValueError, match="Column customer_id has duplicate entries"
        ):
            data = pd.DataFrame(
                {
                    "customer_id": np.asarray([1, 1]),
                    "frequency": np.asarray([1, 1]),
                    "recency": np.asarray([1, 1]),
                    "T": np.asarray([1, 1]),
                }
            )

            ModifiedBetaGeoModel(
                data=data,
            )

    @pytest.mark.parametrize(
        "frequency, recency, logp_value",
        [
            (0, 0, -0.41792826),
            (200, 38, 100.7869),
        ],
    )
    def test_numerically_stable_logp(
        self, frequency, recency, logp_value, model_config
    ):
        """See Solution #2 on pages 3 and 4 of http://brucehardie.com/notes/027/bgnbd_num_error.pdf"""
        model_config = {
            "a_prior": Prior("Flat"),
            "b_prior": Prior("Flat"),
            "alpha_prior": Prior("Flat"),
            "r_prior": Prior("Flat"),
        }
        data = pd.DataFrame(
            {
                "customer_id": np.asarray([1]),
                "frequency": np.asarray([frequency]),
                "recency": np.asarray([recency]),
                "T": np.asarray([40]),
            }
        )
        model = ModifiedBetaGeoModel(
            data=data,
            model_config=model_config,
        )
        model.build_model()
        pymc_model = model.model
        logp = pymc_model.compile_logp()

        np.testing.assert_almost_equal(
            logp({"a": 0.80, "b": 2.50, "r": 0.25, "alpha": 4.00}),
            logp_value,
            decimal=5,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "fit_method, rtol",
        [
            ("mcmc", 0.075),
            ("map", 0.15),
        ],
    )
    def test_model_convergence(self, fit_method, rtol, model_config):
        # b parameter has the largest mismatch of the four parameters
        model = ModifiedBetaGeoModel(
            data=self.data,
            model_config=model_config,
        )
        model.build_model()

        sample_kwargs = (
            dict(random_seed=self.rng, chains=2, target_accept=0.90)
            if fit_method == "mcmc"
            else {}
        )
        model.fit(fit_method=fit_method, progressbar=False, **sample_kwargs)

        fit = model.idata.posterior
        np.testing.assert_allclose(
            [fit["a"].mean(), fit["b"].mean(), fit["alpha"].mean(), fit["r"].mean()],
            [self.a_true, self.b_true, self.alpha_true, self.r_true],
            rtol=rtol,
        )

    def test_fit_result_without_fit(self, mocker, model_config):
        model = ModifiedBetaGeoModel(data=self.data, model_config=model_config)
        with pytest.raises(RuntimeError, match="The model hasn't been fit yet"):
            model.fit_result

        mocker.patch("pymc.sample", mock_sample)

        idata = model.fit(
            tune=5,
            chains=2,
            draws=10,
            compute_convergence_checks=False,
        )
        assert isinstance(idata, az.InferenceData)
        assert len(idata.posterior.chain) == 2
        assert len(idata.posterior.draw) == 10
        assert model.idata is idata

    def tests_expected_probability_no_purchases_raises_exception(self):
        customer_id = np.arange(10)
        test_frequency = np.tile([1, 3, 5, 7, 9], 2)
        test_recency = np.tile([20, 30], 5)
        test_T = np.tile([25, 35], 5)
        test_t = 0
        data = pd.DataFrame(
            {
                "customer_id": customer_id,
                "frequency": test_frequency,
                "recency": test_recency,
                "T": test_T,
            }
        )

        mbg_model = ModifiedBetaGeoModel(data=data)
        mbg_model.build_model()
        mbg_model.idata = az.from_dict(
            {
                "a": np.full((2, 5), self.a_true),
                "b": np.full((2, 5), self.b_true),
                "alpha": np.full((2, 5), self.alpha_true),
                "r": np.full((2, 5), self.r_true),
            }
        )

        with pytest.raises(
            NotImplementedError,
            match="The MBG/NBD model does not support this method.",
        ):
            mbg_model.expected_probability_no_purchase(t=test_t, data=data)

    def test_expected_num_purchases(self):
        customer_id = np.arange(10)
        test_t = np.linspace(20, 38, 10)
        test_frequency = np.tile([1, 3, 5, 7, 9], 2)
        test_recency = np.tile([20, 30], 5)
        test_T = np.tile([25, 35], 5)
        data = pd.DataFrame(
            {
                "customer_id": customer_id,
                "frequency": test_frequency,
                "recency": test_recency,
                "T": test_T,
            }
        )

        mbg_model = ModifiedBetaGeoModel(data=data)
        mbg_model.build_model()
        mbg_model.idata = az.from_dict(
            {
                "a": np.full((2, 5), self.a_true),
                "b": np.full((2, 5), self.b_true),
                "alpha": np.full((2, 5), self.alpha_true),
                "r": np.full((2, 5), self.r_true),
            }
        )

        # TODO: Give this its own test after API revisions completed.
        with pytest.warns(
            FutureWarning,
            match="Deprecated method. Use 'expected_purchases' instead.",
        ):
            res_num_purchases = mbg_model.expected_num_purchases(
                customer_id,
                test_t,
                test_frequency,
                test_recency,
                test_T,
            )
        assert res_num_purchases.shape == (2, 5, 10)
        assert res_num_purchases.dims == ("chain", "draw", "customer_id")

        # Compare with lifetimes
        lifetimes_mbg_model = ModifiedBetaGeoFitter()
        lifetimes_mbg_model.params_ = {
            "a": self.a_true,
            "b": self.b_true,
            "alpha": self.alpha_true,
            "r": self.r_true,
        }
        lifetimes_res_num_purchases = (
            lifetimes_mbg_model.conditional_expected_number_of_purchases_up_to_time(
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

    @pytest.mark.parametrize("test_t", [1, 3, 6])
    def test_expected_purchases(self, test_t):
        true_purchases = (
            self.lifetimes_model.conditional_expected_number_of_purchases_up_to_time(
                t=test_t,
                frequency=self.frequency,
                recency=self.recency,
                T=self.T,
            )
        )
        est_num_purchases = self.model.expected_purchases(future_t=test_t)

        assert est_num_purchases.shape == (self.chains, self.draws, self.N)
        assert est_num_purchases.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_purchases,
            est_num_purchases.mean(("chain", "draw")),
            rtol=0.001,
        )

    @pytest.mark.parametrize("test_t", [1, 3, 6])
    def test_expected_purchases_new_customer(self, test_t):
        true_purchases_new = (
            self.lifetimes_model.expected_number_of_purchases_up_to_time(
                t=test_t,
            )
        )

        est_purchases_new = self.model.expected_purchases_new_customer(t=test_t)

        assert est_purchases_new.shape == (self.chains, self.draws, 2357)
        assert est_purchases_new.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_purchases_new,
            est_purchases_new.mean(("chain", "draw", "customer_id")),
            rtol=0.001,
        )

    def test_expected_num_purchases_new_customer_warning(self):
        with pytest.warns(
            FutureWarning,
            match="Deprecated method. Use 'expected_purchases_new_customer' instead.",
        ):
            self.model.expected_num_purchases_new_customer(t=10)

    def test_expected_probability_alive(self):
        true_prob_alive = self.lifetimes_model.conditional_probability_alive(
            frequency=self.frequency,
            recency=self.recency,
            T=self.T,
        )

        est_prob_alive = self.model.expected_probability_alive()

        assert est_prob_alive.shape == (self.chains, self.draws, self.N)
        assert est_prob_alive.dims == ("chain", "draw", "customer_id")
        np.testing.assert_allclose(
            true_prob_alive,
            est_prob_alive.mean(("chain", "draw")),
            rtol=0.1,
        )

    def test_model_repr(self):
        model_config = {
            "alpha_prior": Prior("HalfFlat"),
            "r_prior": Prior("HalfFlat"),
            "a_prior": Prior("HalfFlat"),
            "b_prior": Prior("HalfNormal", sigma=10),
        }
        model = ModifiedBetaGeoModel(
            data=self.data,
            model_config=model_config,
        )
        model.build_model()
        assert model.__repr__().replace(" ", "") == (
            "MBG/NBD"
            "\nalpha~HalfFlat()"
            "\nr~HalfFlat()"
            "\na~HalfFlat()"
            "\nb~HalfNormal(0,10)"
            "\nrecency_frequency~ModifiedBetaGeoNBD(a,b,r,alpha,<constant>)"
        )

    def test_distribution_new_customer(self) -> None:
        mock_model = ModifiedBetaGeoModel(
            data=self.data,
        )
        mock_model.build_model()
        mock_model.idata = az.from_dict(
            {
                "a": [self.a_true],
                "b": [self.b_true],
                "alpha": [self.alpha_true],
                "r": [self.r_true],
            }
        )

        rng = np.random.default_rng(42)
        new_customer_dropout = mock_model.distribution_new_customer_dropout(
            random_seed=rng
        )
        new_customer_purchase_rate = mock_model.distribution_new_customer_purchase_rate(
            random_seed=rng
        )

        assert isinstance(new_customer_dropout, xr.DataArray)
        assert isinstance(new_customer_purchase_rate, xr.DataArray)

        N = 1000
        p = pm.Beta.dist(self.a_true, self.b_true, size=N)
        lam = pm.Gamma.dist(self.r_true, self.alpha_true, size=N)

        rtol = 0.15
        np.testing.assert_allclose(
            new_customer_dropout.mean(), pm.draw(p.mean(), random_seed=rng), rtol=rtol
        )
        np.testing.assert_allclose(
            new_customer_dropout.var(), pm.draw(p.var(), random_seed=rng), rtol=rtol
        )
        np.testing.assert_allclose(
            new_customer_purchase_rate.mean(),
            pm.draw(lam.mean(), random_seed=rng),
            rtol=rtol,
        )
        np.testing.assert_allclose(
            new_customer_purchase_rate.var(),
            pm.draw(lam.var(), random_seed=rng),
            rtol=rtol,
        )

    def test_save_load(self):
        self.model.save("test_model")
        # Testing the valid case.

        model2 = ModifiedBetaGeoModel.load("test_model")

        # Check if the loaded model is indeed an instance of the class
        assert isinstance(self.model, ModifiedBetaGeoModel)
        # Check if the loaded data matches with the model data
        pd.testing.assert_frame_equal(self.model.data, model2.data, check_names=False)
        assert self.model.model_config == model2.model_config
        assert self.model.sampler_config == model2.sampler_config
        assert self.model.idata == model2.idata
        os.remove("test_model")
