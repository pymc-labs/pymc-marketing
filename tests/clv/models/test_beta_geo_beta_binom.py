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
import pytensor as pt
import pytest
import xarray as xr
from lifetimes.fitters.beta_geo_beta_binom_fitter import BetaGeoBetaBinomFitter

from pymc_marketing.clv.distributions import BetaGeoBetaBinom
from pymc_marketing.clv.models import BetaGeoBetaBinomModel
from pymc_marketing.prior import Prior
from tests.conftest import create_mock_fit, mock_sample


class TestBetaGeoBetaBinomModel:
    @classmethod
    def setup_class(cls):
        # Set random seed
        cls.rng = np.random.default_rng(34)

        # parameters
        cls.alpha_true = 1.2035
        cls.beta_true = 0.7497
        cls.delta_true = 2.7834
        cls.gamma_true = 0.6567

        # Use Quickstart dataset (the CDNOW_sample research data) for testing
        cls.data = pd.read_csv("data/bgbb_donations.csv")

        # sample from full dataset for tests involving model fits
        cls.sample_data = cls.data.sample(n=1000, random_state=45)

        # take sample of all unique recency/frequency/T combinations to test predictive methods
        test_customer_ids = [
            3463,
            4554,
            4831,
            4960,
            5038,
            5159,
            5286,
            5899,
            6154,
            6309,
            6482,
            6716,
            7038,
            7219,
            7444,
            7801,
            8041,
            8235,
            8837,
            9172,
            9900,
            11103,
        ]

        cls.pred_data = cls.data.query("customer_id.isin(@test_customer_ids)")
        cls.pred_data_N = len(test_customer_ids)

        # Instantiate model with CDNOW data for testing
        cls.model = BetaGeoBetaBinomModel(cls.data)
        cls.model.build_model()

        # Also instantiate lifetimes model for comparison
        cls.lifetimes_model = BetaGeoBetaBinomFitter()
        cls.lifetimes_model.params_ = {
            "alpha": cls.alpha_true,
            "beta": cls.beta_true,
            "delta": cls.delta_true,
            "gamma": cls.gamma_true,
        }

        # Mock an idata object for tests requiring a fitted model
        cls.N = len(cls.data)

        mock_fit = create_mock_fit(
            {
                "alpha": cls.alpha_true,
                "beta": cls.beta_true,
                "delta": cls.delta_true,
                "gamma": cls.gamma_true,
            }
        )

        cls.chains = 2
        cls.draws = 50
        mock_fit(cls.model, chains=cls.chains, draws=cls.draws, rng=cls.rng)

    @pytest.fixture(scope="class")
    def model_config(self):
        return {
            "alpha": Prior("HalfNormal"),
            "beta": Prior("HalfStudentT", nu=4),
            "delta": Prior("HalfCauchy", beta=2),
            "gamma": Prior("Gamma", alpha=1, beta=1),
        }

    def test_model(self, model_config):
        # this test requires a different setup from other models due to default_model_config containing NoneTypes
        default_model = BetaGeoBetaBinomModel(
            data=self.data,
            model_config=None,
        )
        custom_model = BetaGeoBetaBinomModel(
            data=self.data,
            model_config=model_config,
        )

        for model in (default_model, custom_model):
            model.build_model()
            assert isinstance(
                model.model["alpha"].owner.op,
                pt.tensor.elemwise.Elemwise
                if "alpha" not in model.model_config
                else model.model_config["alpha"].pymc_distribution,
            )
            assert isinstance(
                model.model["beta"].owner.op,
                pt.tensor.elemwise.Elemwise
                if "beta" not in model.model_config
                else model.model_config["beta"].pymc_distribution,
            )
            assert isinstance(
                model.model["delta"].owner.op,
                pt.tensor.elemwise.Elemwise
                if "delta" not in model.model_config
                else model.model_config["delta"].pymc_distribution,
            )
            assert isinstance(
                model.model["gamma"].owner.op,
                pt.tensor.elemwise.Elemwise
                if "gamma" not in model.model_config
                else model.model_config["gamma"].pymc_distribution,
            )

        assert default_model.model.eval_rv_shapes() == {
            "kappa_dropout": (),
            "kappa_dropout_interval__": (),
            "kappa_purchase": (),
            "kappa_purchase_interval__": (),
            "phi_dropout": (),
            "phi_dropout_interval__": (),
            "phi_purchase": (),
            "phi_purchase_interval__": (),
        }

        assert custom_model.model.eval_rv_shapes() == {
            "alpha": (),
            "alpha_log__": (),
            "beta": (),
            "beta_log__": (),
            "delta": (),
            "delta_log__": (),
            "gamma": (),
            "gamma_log__": (),
        }

    def test_missing_cols(self):
        data_invalid = self.data.drop(columns="customer_id")

        with pytest.raises(ValueError, match="Required column customer_id missing"):
            BetaGeoBetaBinomModel(data=data_invalid)

        data_invalid = self.data.drop(columns="frequency")

        with pytest.raises(ValueError, match="Required column frequency missing"):
            BetaGeoBetaBinomModel(data=data_invalid)

        data_invalid = self.data.drop(columns="recency")

        with pytest.raises(ValueError, match="Required column recency missing"):
            BetaGeoBetaBinomModel(data=data_invalid)

        data_invalid = self.data.drop(columns="T")

        with pytest.raises(ValueError, match="Required column T missing"):
            BetaGeoBetaBinomModel(data=data_invalid)

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

            BetaGeoBetaBinomModel(
                data=data,
            )

    def test_T_homogeneity(self):
        with pytest.raises(ValueError, match="Column T has  non-homogeneous entries"):
            data = pd.DataFrame(
                {
                    "customer_id": np.asarray([1, 2]),
                    "frequency": np.asarray([1, 2]),
                    "recency": np.asarray([1, 2]),
                    "T": np.asarray([1, 2]),
                }
            )

            BetaGeoBetaBinomModel(
                data=data,
            )

    @pytest.mark.parametrize("custom_config", (True, False))
    def test_model_repr(self, custom_config):
        if custom_config:
            model_config = {
                "alpha": Prior("HalfFlat"),
                "beta": Prior("HalfFlat"),
                "delta": Prior("HalfFlat"),
                "gamma": Prior("HalfNormal", sigma=10),
            }
            repr = (
                "BG/BB"
                "\nalpha~HalfFlat()"
                "\nbeta~HalfFlat()"
                "\ngamma~HalfNormal(0,10)"
                "\ndelta~HalfFlat()"
                "\nrecency_frequency~BetaGeoBetaBinom(alpha,beta,gamma,delta,<constant>)"
            )
        else:
            model_config = None
            repr = (
                "BG/BB"
                "\nphi_purchase~Uniform(0,1)"
                "\nkappa_purchase~Pareto(1,1)"
                "\nphi_dropout~Uniform(0,1)"
                "\nkappa_dropout~Pareto(1,1)"
                "\nalpha~Deterministic(f(kappa_purchase,phi_purchase))"
                "\nbeta~Deterministic(f(kappa_purchase,phi_purchase))"
                "\ngamma~Deterministic(f(kappa_dropout,phi_dropout))"
                "\ndelta~Deterministic(f(kappa_dropout,phi_dropout))"
                "\nrecency_frequency~BetaGeoBetaBinom(alpha,beta,gamma,delta,<constant>)"
            )
        model = BetaGeoBetaBinomModel(
            data=self.data,
            model_config=model_config,
        )
        model.build_model()

        assert model.__repr__().replace(" ", "") == repr

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "method, rtol",
        [
            (
                "mcmc",
                0.3,
            ),  # higher rtol required for sample_data; within .1 tolerance for full dataset;
            ("map", 0.2),
        ],
    )
    def test_model_convergence(self, method, rtol, model_config):
        model = BetaGeoBetaBinomModel(
            data=self.sample_data,
            model_config=model_config,
        )
        model.build_model()

        sample_kwargs = dict(random_seed=self.rng, chains=2) if method == "mcmc" else {}
        model.fit(method=method, progressbar=False, **sample_kwargs)

        fit = model.idata.posterior
        np.testing.assert_allclose(
            [
                fit["alpha"].mean(),
                fit["beta"].mean(),
                fit["delta"].mean(),
                fit["gamma"].mean(),
            ],
            [self.alpha_true, self.beta_true, self.delta_true, self.gamma_true],
            rtol=rtol,
        )

    def test_fit_result_without_fit(self, mocker, model_config):
        model = BetaGeoBetaBinomModel(data=self.pred_data, model_config=model_config)
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

    @pytest.mark.parametrize("test_t", [1, 3, 6])
    def test_expected_purchases(self, test_t):
        true_purchases = (
            self.lifetimes_model.conditional_expected_number_of_purchases_up_to_time(
                m_periods_in_future=test_t,
                frequency=self.pred_data["frequency"],
                recency=self.pred_data["recency"],
                n_periods=self.pred_data["T"],
            )
        )

        # test parametrization with default data has different dims
        est_num_purchases = self.model.expected_purchases(future_t=test_t)
        assert est_num_purchases.shape == (self.chains, self.draws, self.N)

        data = self.pred_data.assign(future_t=test_t)
        est_num_purchases = self.model.expected_purchases(data)

        assert est_num_purchases.shape == (self.chains, self.draws, self.pred_data_N)
        assert est_num_purchases.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_purchases,
            est_num_purchases.mean(("chain", "draw")),
            rtol=0.01,
        )

    def test_expected_purchases_new_customer(self):
        # values obtained from cells B7:17 from 'Tracking Plot" sheet in https://www.brucehardie.com/notes/010/
        true_purchases_new = np.array(
            [
                0.4985,
                0.9233,
                1.2969,
                1.6323,
                1.9381,
                2.2202,
                2.4826,
                2.7285,
                2.9603,
                3.1798,
                3.3887,
            ]
        )
        time_periods = np.arange(1, 12)

        # test dimensions for a single prediction
        data = pd.DataFrame({"customer_id": [0], "t": [5]})
        est_purchase_new = self.model.expected_purchases_new_customer(data)

        assert est_purchase_new.shape == (self.chains, self.draws, 1)
        assert est_purchase_new.dims == ("chain", "draw", "customer_id")

        # compare against array of true values
        est_purchases_new = (
            xr.concat(
                objs=[
                    self.model.expected_purchases_new_customer(None, t=t).mean()
                    for t in time_periods
                ],
                dim="t",
            )
            .transpose(..., "t")
            .values
        )

        np.testing.assert_allclose(
            true_purchases_new,
            est_purchases_new,
            rtol=0.001,
        )

    @pytest.mark.parametrize("test_t", [1, 3, 6])
    def test_expected_probability_alive(self, test_t):
        true_prob_alive = self.lifetimes_model.conditional_probability_alive(
            m_periods_in_future=test_t,
            frequency=self.pred_data["frequency"],
            recency=self.pred_data["recency"],
            n_periods=self.pred_data["T"],
        )

        # test parametrization with default data has different dims
        est_prob_alive = self.model.expected_probability_alive(future_t=test_t)
        assert est_prob_alive.shape == (self.chains, self.draws, self.N)

        pred_data = self.pred_data.assign(future_t=test_t)
        est_prob_alive = self.model.expected_probability_alive(pred_data)

        assert est_prob_alive.shape == (self.chains, self.draws, self.pred_data_N)
        assert est_prob_alive.dims == ("chain", "draw", "customer_id")
        np.testing.assert_allclose(
            true_prob_alive,
            est_prob_alive.mean(("chain", "draw")),
            rtol=0.01,
        )

        alt_data = self.pred_data.assign(future_t=7.5)
        est_prob_alive_t = self.model.expected_probability_alive(alt_data)
        assert est_prob_alive.mean() > est_prob_alive_t.mean()

    def test_distribution_new_customer(self) -> None:
        mock_model = BetaGeoBetaBinomModel(
            data=self.sample_data,
        )
        mock_model.build_model()
        mock_model.idata = az.from_dict(
            {
                "alpha": [self.alpha_true],
                "beta": [self.beta_true],
                "delta": [self.delta_true],
                "gamma": [self.gamma_true],
            }
        )

        rng = np.random.default_rng(42)
        new_customer_dropout = mock_model.distribution_new_customer_dropout(
            random_seed=rng
        )
        new_customer_purchase_rate = mock_model.distribution_new_customer_purchase_rate(
            random_seed=rng
        )
        customer_rec_freq = mock_model.distribution_new_customer_recency_frequency(
            self.sample_data, T=self.sample_data["T"], random_seed=rng
        )
        customer_rec = customer_rec_freq.sel(obs_var="recency")
        customer_freq = customer_rec_freq.sel(obs_var="frequency")

        assert isinstance(new_customer_dropout, xr.DataArray)
        assert isinstance(new_customer_purchase_rate, xr.DataArray)
        assert isinstance(customer_rec, xr.DataArray)
        assert isinstance(customer_freq, xr.DataArray)

        N = 1000
        p = pm.Beta.dist(self.alpha_true, self.beta_true, size=N)
        theta = pm.Beta.dist(self.gamma_true, self.delta_true, size=N)
        ref_rec, ref_freq = pm.draw(
            BetaGeoBetaBinom.dist(
                alpha=self.alpha_true,
                beta=self.beta_true,
                delta=self.delta_true,
                gamma=self.gamma_true,
                T=self.sample_data["T"],
            ),
            random_seed=rng,
        ).T

        rtol = 0.15
        np.testing.assert_allclose(
            new_customer_dropout.mean(),
            pm.draw(theta.mean(), random_seed=rng),
            rtol=rtol,
        )
        np.testing.assert_allclose(
            new_customer_dropout.var(), pm.draw(theta.var(), random_seed=rng), rtol=rtol
        )
        np.testing.assert_allclose(
            new_customer_purchase_rate.mean(),
            pm.draw(p.mean(), random_seed=rng),
            rtol=rtol,
        )
        np.testing.assert_allclose(
            new_customer_purchase_rate.var(),
            pm.draw(p.var(), random_seed=rng),
            rtol=rtol,
        )
        np.testing.assert_allclose(
            customer_rec.mean(),
            ref_rec.mean(),
            rtol=rtol,
        )
        np.testing.assert_allclose(
            customer_rec.var(),
            ref_rec.var(),
            rtol=rtol,
        )
        np.testing.assert_allclose(
            customer_freq.mean(),
            ref_freq.mean(),
            rtol=rtol,
        )
        np.testing.assert_allclose(
            customer_freq.var(),
            ref_freq.var(),
            rtol=rtol,
        )

    def test_save_load(self):
        self.model.save("test_model")
        # Testing the valid case.

        model2 = BetaGeoBetaBinomModel.load("test_model")

        # Check if the loaded model is indeed an instance of the class
        assert isinstance(self.model, BetaGeoBetaBinomModel)
        # Check if the loaded data matches with the model data
        pd.testing.assert_frame_equal(self.model.data, model2.data, check_names=False)
        assert self.model.model_config == model2.model_config
        assert self.model.sampler_config == model2.sampler_config
        assert self.model.idata == model2.idata
        os.remove("test_model")
