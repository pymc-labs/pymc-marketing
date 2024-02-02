import os

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from lifetimes.fitters.beta_geo_fitter import BetaGeoFitter

from pymc_marketing.clv.models.beta_geo import BetaGeoModel


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
        # TODO: Create a pytest fixture for this
        test_data = pd.read_csv("datasets/clv_quickstart.csv")
        test_data["customer_id"] = test_data.index

        cls.data = test_data
        cls.customer_id = test_data["customer_id"]
        cls.frequency = test_data["frequency"]
        cls.recency = test_data["recency"]
        cls.T = test_data["T"]

        # Instantiate model with CDNOW data for testing
        cls.model = BetaGeoModel(cls.data)
        # TODO: This can be removed after build_model() is called internally with __init__
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
        cls.mock_fit = az.from_dict(
            {
                "a": cls.rng.normal(cls.a_true, 1e-3, size=(cls.chains, cls.draws)),
                "b": cls.rng.normal(cls.b_true, 1e-3, size=(cls.chains, cls.draws)),
                "alpha": cls.rng.normal(
                    cls.alpha_true, 1e-3, size=(cls.chains, cls.draws)
                ),
                "r": cls.rng.normal(cls.r_true, 1e-3, size=(cls.chains, cls.draws)),
            }
        )

        cls.model.idata = cls.mock_fit

    @pytest.fixture(scope="class")
    def data(self):
        return pd.DataFrame(
            {
                "customer_id": self.customer_id,
                "frequency": self.frequency,
                "recency": self.recency,
                "T": self.T,
            }
        )

    @pytest.fixture(scope="class")
    def model_config(self):
        return {
            "a_prior": {"dist": "HalfNormal", "kwargs": {}},
            "b_prior": {"dist": "HalfStudentT", "kwargs": {"nu": 4}},
            "alpha_prior": {"dist": "HalfCauchy", "kwargs": {"beta": 2}},
            "r_prior": {"dist": "Gamma", "kwargs": {"alpha": 1, "beta": 1}},
        }

    @pytest.fixture(scope="class")
    def default_model_config(self):
        return {
            "a_prior": {"dist": "HalfFlat", "kwargs": {}},
            "b_prior": {"dist": "HalfFlat", "kwargs": {}},
            "alpha_prior": {"dist": "HalfFlat", "kwargs": {}},
            "r_prior": {"dist": "HalfFlat", "kwargs": {}},
        }

    def test_model(self, model_config, default_model_config, data):
        for config in (model_config, default_model_config):
            model = BetaGeoModel(
                data=data,
                model_config=config,
            )
            model.build_model()
            assert isinstance(
                model.model["a"].owner.op,
                pm.HalfFlat
                if config["a_prior"]["dist"] == "HalfFlat"
                else getattr(pm, config["a_prior"]["dist"]),
            )
            assert isinstance(
                model.model["b"].owner.op,
                pm.HalfFlat
                if config["b_prior"]["dist"] == "HalfFlat"
                else getattr(pm, config["b_prior"]["dist"]),
            )
            assert isinstance(
                model.model["alpha"].owner.op,
                pm.HalfFlat
                if config["alpha_prior"]["dist"] == "HalfFlat"
                else getattr(pm, config["alpha_prior"]["dist"]),
            )
            assert isinstance(
                model.model["r"].owner.op,
                pm.HalfFlat
                if config["r_prior"]["dist"] == "HalfFlat"
                else getattr(pm, config["r_prior"]["dist"]),
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

    def test_missing_customer_id(self, data):
        # Create a version of the data that's missing the 'customer_id' column
        data_invalid = data.drop(columns="customer_id")

        with pytest.raises(KeyError, match="customer_id column is missing from data"):
            BetaGeoModel(data=data_invalid)

    def test_missing_frequency(self, data):
        # Create a version of the data that's missing the 'frequency' column
        data_invalid = data.drop(columns="frequency")

        with pytest.raises(KeyError, match="frequency column is missing from data"):
            BetaGeoModel(data=data_invalid)

    def test_missing_recency(self, data):
        # Create a version of the data that's missing the 'recency' column
        data_invalid = data.drop(columns="recency")

        with pytest.raises(KeyError, match="recency column is missing from data"):
            BetaGeoModel(data=data_invalid)

    def test_missing_T(self, data):
        # Create a version of the data that's missing the 'T' column
        data_invalid = data.drop(columns="T")

        with pytest.raises(KeyError, match="T column is missing from data"):
            BetaGeoModel(data=data_invalid)

    def test_customer_id_warning(self):
        with pytest.raises(
            ValueError,
            match="The BetaGeoModel expects exactly one entry per customer. More than one entry is currently provided per customer id.",
        ):
            data = pd.DataFrame(
                {
                    "customer_id": np.asarray([1, 1]),
                    "frequency": np.asarray([1, 1]),
                    "recency": np.asarray([1, 1]),
                    "T": np.asarray([1, 1]),
                }
            )

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
        """
        See Solution #2 on pages 3 and 4 of http://brucehardie.com/notes/027/bgnbd_num_error.pdf
        """
        model_config = {
            "a_prior": {"dist": "Flat", "kwargs": {}},
            "b_prior": {"dist": "Flat", "kwargs": {}},
            "alpha_prior": {"dist": "Flat", "kwargs": {}},
            "r_prior": {"dist": "Flat", "kwargs": {}},
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
            (
                500,
                "mcmc",
                0.3,
            ),
            (2000, "mcmc", 0.1),
            (10000, "mcmc", 0.055),
            (2000, "map", 0.1035),
        ],
    )
    def test_model_convergence(self, N, fit_method, rtol, model_config):
        rng = np.random.default_rng(146)
        recency, frequency, _, T = self.generate_data(
            self.a_true, self.b_true, self.alpha_true, self.r_true, N, rng=rng
        )
        data = pd.DataFrame(
            {
                "customer_id": list(range(len(frequency))),
                "frequency": frequency,
                "recency": recency,
                "T": T,
            }
        )
        # b parameter has the largest mismatch of the four parameters
        model = BetaGeoModel(
            data=data,
            model_config=model_config,
        )
        model.build_model()

        sample_kwargs = dict(random_seed=rng, chains=2) if fit_method == "mcmc" else {}
        model.fit(fit_method=fit_method, progressbar=False, **sample_kwargs)

        fit = model.idata.posterior
        np.testing.assert_allclose(
            [fit["a"].mean(), fit["b"].mean(), fit["alpha"].mean(), fit["r"].mean()],
            [self.a_true, self.b_true, self.alpha_true, self.r_true],
            rtol=rtol,
        )

    def test_fit_result_without_fit(self, data, model_config):
        model = BetaGeoModel(data=data, model_config=model_config)
        with pytest.raises(RuntimeError, match="The model hasn't been fit yet"):
            model.fit_result

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

    def test_model_repr(self, data):
        model_config = {
            "alpha_prior": {"dist": "HalfFlat", "kwargs": {}},
            "r_prior": {"dist": "HalfFlat", "kwargs": {}},
            "a_prior": {"dist": "HalfFlat", "kwargs": {}},
            "b_prior": {"dist": "HalfNormal", "kwargs": {"sigma": 10}},
        }
        model = BetaGeoModel(
            data=data,
            model_config=model_config,
        )
        model.build_model()
        assert model.__repr__().replace(" ", "") == (
            "BG/NBD"
            "\na~HalfFlat()"
            "\nb~HalfNormal(0,10)"
            "\nalpha~HalfFlat()"
            "\nr~HalfFlat()"
            "\nlikelihood~Potential(f(r,alpha,b,a))"
        )

    @pytest.mark.parametrize("test_t", [1, 2, 3, 4, 5, 6])
    def test_expected_purchases(self, test_t):
        true_purchases = (
            self.lifetimes_model.conditional_expected_number_of_purchases_up_to_time(
                t=test_t,
                frequency=self.frequency,
                recency=self.recency,
                T=self.T,
            )
        )

        est_num_purchases = self.model.expected_purchases(test_t)

        assert est_num_purchases.shape == (self.chains, self.draws, self.N)
        assert est_num_purchases.dims == ("chain", "draw", "customer_id")

        np.testing.assert_allclose(
            true_purchases,
            est_num_purchases.mean(("chain", "draw")),
            rtol=0.001,
        )

    @pytest.mark.parametrize("test_t", [1, 2, 3, 4, 5, 6])
    def test_expected_purchases_new_customer(self, test_t):
        true_purchases_new = (
            self.lifetimes_model.expected_number_of_purchases_up_to_time(
                t=test_t,
            )
        )

        est_purchases_new = self.model.expected_purchases_new_customer(test_t)

        assert est_purchases_new.shape == (self.chains, self.draws)
        assert est_purchases_new.dims == ("chain", "draw")

        np.testing.assert_allclose(
            true_purchases_new,
            est_purchases_new.mean(("chain", "draw")),
            rtol=0.001,
        )

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

    def test_distribution_new_customer(self, data) -> None:
        # mock_model = BetaGeoModel(
        #     data=data,
        # )
        # mock_model.idata = az.from_dict(
        #     {
        #         "a": [self.a_true],
        #         "b": [self.b_true],
        #         "alpha": [self.alpha_true],
        #         "r": [self.r_true],
        #     }
        # )

        rng = np.random.default_rng(42)
        new_customer_dropout = self.model.distribution_new_customer_dropout(
            random_seed=rng
        )
        new_customer_purchase_rate = self.model.distribution_new_customer_purchase_rate(
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

    def test_save_load_beta_geo(self, data):
        model = BetaGeoModel(
            data=data,
        )
        model.build_model()
        model.fit("map")
        model.save("test_model")
        # Testing the valid case.

        model2 = BetaGeoModel.load("test_model")

        # Check if the loaded model is indeed an instance of the class
        assert isinstance(model, BetaGeoModel)
        # Check if the loaded data matches with the model data
        np.testing.assert_array_equal(
            model2.customer_id.values, model.customer_id.values
        )
        np.testing.assert_array_equal(model2.frequency.values, model.frequency.values)
        np.testing.assert_array_equal(model2.T.values, model.T.values)
        np.testing.assert_array_equal(model2.recency.values, model.recency.values)
        assert model.model_config == model2.model_config
        assert model.sampler_config == model2.sampler_config
        assert model.idata == model2.idata
        os.remove("test_model")

    def test_expected_num_purchases_warning(self, data):
        with pytest.warns(
            FutureWarning,
            match="Method was renamed to 'expected_purchases'. Old method will be removed in a future release.",
        ):
            self.model.expected_num_purchases(10)

    def test_expected_num_purchases_new_customer_warning(self, data):
        with pytest.warns(
            FutureWarning,
            match="Method was renamed to 'expected_purchases_new_customer'. Old method will be removed in a future release.",
        ):
            self.model.expected_num_purchases_new_customer(t=10)
