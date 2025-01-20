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
from lifetimes.fitters.beta_geo_fitter import BetaGeoFitter

from pymc_marketing.clv.distributions import BetaGeoNBD
from pymc_marketing.clv.models.beta_geo import BetaGeoModel
from pymc_marketing.prior import Prior
from tests.conftest import create_mock_fit, mock_sample


class TestBetaGeoModel:
    @classmethod
    def setup_class(cls):
        # Set random seed
        cls.rng = np.random.default_rng(34)

        # parameters
        cls.a_true = 0.793
        cls.b_true = 2.426
        cls.alpha_true = 4.414
        cls.r_true = 0.243

        # Use Quickstart dataset (the CDNOW_sample research data) for testing
        test_data = pd.read_csv("data/clv_quickstart.csv")
        test_data["customer_id"] = test_data.index

        cls.data = test_data
        cls.customer_id = test_data["customer_id"]
        cls.frequency = test_data["frequency"]
        cls.recency = test_data["recency"]
        cls.T = test_data["T"]

        # Instantiate model with CDNOW data for testing
        cls.model = BetaGeoModel(cls.data)
        cls.model.build_model()

        # Also instantiate lifetimes model for comparison
        cls.lifetimes_model = BetaGeoFitter()
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
            model = BetaGeoModel(
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

    def test_missing_cols(self):
        data_invalid = self.data.drop(columns="customer_id")

        with pytest.raises(ValueError, match="Required column customer_id missing"):
            BetaGeoModel(data=data_invalid)

        data_invalid = self.data.drop(columns="frequency")

        with pytest.raises(ValueError, match="Required column frequency missing"):
            BetaGeoModel(data=data_invalid)

        data_invalid = self.data.drop(columns="recency")

        with pytest.raises(ValueError, match="Required column recency missing"):
            BetaGeoModel(data=data_invalid)

        data_invalid = self.data.drop(columns="T")

        with pytest.raises(ValueError, match="Required column T missing"):
            BetaGeoModel(data=data_invalid)

    def test_customer_id_duplicate(self):
        data = pd.DataFrame(
            {
                "customer_id": np.asarray([1, 1]),
                "frequency": np.asarray([1, 1]),
                "recency": np.asarray([1, 1]),
                "T": np.asarray([1, 1]),
            }
        )

        with pytest.raises(
            ValueError, match="Column customer_id has duplicate entries"
        ):
            BetaGeoModel(
                data=data,
            )

    @pytest.mark.parametrize(
        "frequency, recency, logp_value",
        [
            (0, 0, -0.59947382),
            (200, 38, 100.7957),
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

        model = BetaGeoModel(
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

    @pytest.mark.parametrize("fit_type", ("map", "mcmc"))
    def test_posterior_distributions(self, fit_type) -> None:
        rng = np.random.default_rng(42)
        dim_T = 2357

        if fit_type == "map":
            map_idata = self.model.idata.copy()
            map_idata.posterior = map_idata.posterior.isel(
                chain=slice(None, 1), draw=slice(None, 1)
            )
            model = self.model._build_with_idata(map_idata)
            # We expect 1000 draws to be sampled with MAP
            expected_shape = (1, 1000)
            expected_pop_dims = (1, 1000, dim_T, 2)
        else:
            model = self.model
            expected_shape = (self.chains, self.draws)
            expected_pop_dims = (self.chains, self.draws, dim_T, 2)

        data = model.data
        customer_dropout = model.distribution_new_customer_dropout(
            data, random_seed=rng
        )
        customer_purchase_rate = model.distribution_new_customer_purchase_rate(
            data, random_seed=rng
        )
        customer_rec_freq = model.distribution_new_customer_recency_frequency(
            data, random_seed=rng
        )
        customer_rec = customer_rec_freq.sel(obs_var="recency")
        customer_freq = customer_rec_freq.sel(obs_var="frequency")

        assert customer_dropout.shape == expected_shape
        assert customer_purchase_rate.shape == expected_shape
        assert customer_rec_freq.shape == expected_pop_dims

        N = 1000
        p = pm.Beta.dist(self.a_true, self.b_true, size=N)
        lam = pm.Gamma.dist(self.r_true, self.alpha_true, size=N)

        ref_rec, ref_freq = pm.draw(
            BetaGeoNBD.dist(
                a=self.a_true,
                b=self.b_true,
                r=self.r_true,
                alpha=self.alpha_true,
                T=self.T,
            ),
            random_seed=rng,
        ).T

        np.testing.assert_allclose(
            customer_purchase_rate.mean(),
            pm.draw(lam.mean(), random_seed=rng),
            rtol=0.5,
        )
        np.testing.assert_allclose(
            customer_purchase_rate.std(),
            pm.draw(lam.std(), random_seed=rng),
            rtol=0.5,
        )
        np.testing.assert_allclose(
            customer_dropout.mean(), pm.draw(p.mean(), random_seed=rng), rtol=0.5
        )
        np.testing.assert_allclose(
            customer_dropout.std(), pm.draw(p.std(), random_seed=rng), rtol=0.5
        )

        np.testing.assert_allclose(customer_rec.mean(), ref_rec.mean(), rtol=0.5)
        np.testing.assert_allclose(customer_rec.std(), ref_rec.std(), rtol=0.5)

        np.testing.assert_allclose(customer_freq.mean(), ref_freq.mean(), rtol=0.5)
        np.testing.assert_allclose(customer_freq.std(), ref_freq.std(), rtol=0.5)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "fit_method, rtol",
        [
            ("mcmc", 0.1),
            ("map", 0.2),
        ],
    )
    def test_model_convergence(self, fit_method, rtol, model_config):
        # b parameter has the largest mismatch of the four parameters
        model = BetaGeoModel(
            data=self.data,
            model_config=model_config,
        )
        model.build_model()

        sample_kwargs = (
            dict(random_seed=self.rng, chains=2) if fit_method == "mcmc" else {}
        )
        model.fit(fit_method=fit_method, progressbar=False, **sample_kwargs)

        fit = model.idata.posterior
        np.testing.assert_allclose(
            [fit["a"].mean(), fit["b"].mean(), fit["alpha"].mean(), fit["r"].mean()],
            [self.a_true, self.b_true, self.alpha_true, self.r_true],
            rtol=rtol,
        )

    def test_fit_result_without_fit(self, mocker, model_config):
        model = BetaGeoModel(data=self.data, model_config=model_config)
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

    def test_expected_probability_no_purchases_infrequent_customers(self):
        atol = 10e-3
        customer_id = np.arange(5)
        test_frequency = [3, 30, 5, 70, 9]
        test_recency = [100, 30, 500, 70, 900]
        test_T = [500, 300, 1000, 700, 1800]
        test_t = 3
        data = pd.DataFrame(
            {
                "customer_id": customer_id,
                "frequency": test_frequency,
                "recency": test_recency,
                "T": test_T,
            }
        )

        bg_model = BetaGeoModel(data=data)
        bg_model.build_model()
        bg_model.idata = az.from_dict(
            {
                "a": np.full((2, 5), self.a_true),
                "b": np.full((2, 5), self.b_true),
                "alpha": np.full((2, 5), self.alpha_true),
                "r": np.full((2, 5), self.r_true),
            }
        )

        res_prob_no_purchases = bg_model.expected_probability_no_purchase(
            t=test_t, data=data
        )
        assert np.all(np.isclose(res_prob_no_purchases.to_numpy(), 1, atol=atol))

    @pytest.mark.parametrize("test_t", [30, 90, 120])
    def test_expected_probability_no_purchases_frequent_customers(self, test_t):
        atol = 10e-3
        customer_id = np.arange(5)
        test_frequency = [100, 300, 500, 700, 900]
        test_recency = [100, 300, 500, 700, 900]
        test_T = [100, 300, 500, 700, 900]
        data = pd.DataFrame(
            {
                "customer_id": customer_id,
                "frequency": test_frequency,
                "recency": test_recency,
                "T": test_T,
            }
        )

        bg_model = BetaGeoModel(data=data)
        bg_model.build_model()
        bg_model.idata = az.from_dict(
            {
                "a": np.full((2, 5), self.a_true),
                "b": np.full((2, 5), self.b_true),
                "alpha": np.full((2, 5), self.alpha_true),
                "r": np.full((2, 5), self.r_true),
            }
        )

        res_prob_no_purchases = bg_model.expected_probability_no_purchase(
            t=test_t, data=data
        )
        assert np.all(np.isclose(res_prob_no_purchases.to_numpy(), 0, atol=atol))

    def test_expected_probability_no_purchases_now(self):
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

        bg_model = BetaGeoModel(data=data)
        bg_model.build_model()
        bg_model.idata = az.from_dict(
            {
                "a": np.full((2, 5), self.a_true),
                "b": np.full((2, 5), self.b_true),
                "alpha": np.full((2, 5), self.alpha_true),
                "r": np.full((2, 5), self.r_true),
            }
        )

        res_prob_no_purchases = bg_model.expected_probability_no_purchase(
            t=test_t, data=data
        )
        assert np.all(np.isclose(res_prob_no_purchases.to_numpy(), 1))

    @pytest.mark.parametrize("test_t", [0, 30, 90])
    def test_expected_probability_no_purchases(self, test_t):
        customer_id = np.arange(10)
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

        bg_model = BetaGeoModel(data=data)
        bg_model.build_model()
        bg_model.idata = az.from_dict(
            {
                "a": np.full((2, 5), self.a_true),
                "b": np.full((2, 5), self.b_true),
                "alpha": np.full((2, 5), self.alpha_true),
                "r": np.full((2, 5), self.r_true),
            }
        )

        res_prob_no_purchases = bg_model.expected_probability_no_purchase(
            t=test_t, data=data
        )

        assert res_prob_no_purchases.shape == (2, 5, 10)
        assert res_prob_no_purchases.dims == ("chain", "draw", "customer_id")

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

        bg_model = BetaGeoModel(data=data)
        bg_model.build_model()
        bg_model.idata = az.from_dict(
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
            rtol=0.001,
        )

    def test_model_repr(self):
        model_config = {
            "alpha_prior": Prior("HalfFlat"),
            "r_prior": Prior("HalfFlat"),
            "a_prior": Prior("HalfFlat"),
            "b_prior": Prior("HalfNormal", sigma=10),
        }
        model = BetaGeoModel(
            data=self.data,
            model_config=model_config,
        )
        model.build_model()
        assert model.__repr__().replace(" ", "") == (
            "BG/NBD"
            "\nalpha~HalfFlat()"
            "\nr~HalfFlat()"
            "\na~HalfFlat()"
            "\nb~HalfNormal(0,10)"
            "\nrecency_frequency~BetaGeoNBD(a,b,r,alpha,<constant>)"
        )

    def test_distribution_new_customer(self) -> None:
        mock_model = BetaGeoModel(
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

        model2 = BetaGeoModel.load("test_model")

        # Check if the loaded model is indeed an instance of the class
        assert isinstance(self.model, BetaGeoModel)
        # Check if the loaded data matches with the model data
        pd.testing.assert_frame_equal(self.model.data, model2.data, check_names=False)
        assert self.model.model_config == model2.model_config
        assert self.model.sampler_config == model2.sampler_config
        assert self.model.idata == model2.idata
        os.remove("test_model")
