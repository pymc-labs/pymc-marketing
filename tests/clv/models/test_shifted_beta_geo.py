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
from pymc.distributions.censored import CensoredRV
from pymc_extras.prior import Prior
from scipy import stats

from pymc_marketing.clv import ShiftedBetaGeoModel, ShiftedBetaGeoModelIndividual
from tests.clv.conftest import mock_sample


class TestShiftedBetaGeoModel:
    @classmethod
    def setup_class(cls):
        # Set random seed
        cls.seed = 42
        cls.rng = np.random.default_rng(cls.seed)

        # Test parameters for dual cohort MCMC fit of expected_probability_alive, expected_retention_rate
        # Highend and regular parameters from pg(7) of paper: https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf
        # TODO: might need to be 0.668 instead
        cls.alpha_hi_reg = [0.688, 0.704]
        cls.beta_hi_reg = [3.806, 1.182]

        # TODO: These are needed to test predictive methods in an upcoming PR.
        # Test parameters for single cohort MAP fit of expected_lifetime_purchases, expected_retention_elasticity
        # Both parameter sets from from pg(4) of paper: https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_contractual_mksc_10.pdf
        cls.alpha_case1 = 3.80
        cls.beta_case1 = 15.20

        cls.alpha_case2 = 0.067
        cls.beta_case2 = 0.267

        # Instantiate model with research paper data
        cls.data = pd.read_csv("data/sbg_reg_hi_cohorts.csv").query("T <= 8")
        cls.model = ShiftedBetaGeoModel(cls.data)
        cls.model.build_model()

        # Mock an idata object for tests requiring a fitted model
        cls.N = len(cls.data)
        cls.chains = 2
        cls.draws = 50
        cls.cohorts = ["highend", "regular"]

        # Mock a fitted model with multi-dim parameters
        cls.model = cls.mock_cohort_fit(
            cls.model,
            [cls.alpha_hi_reg, cls.beta_hi_reg],
            cls.chains,
            cls.draws,
            cls.cohorts,
        )

    # TODO: Generalize and move into conftest? create_mock_fit doesn't support multi-dim parameters
    @classmethod
    def mock_cohort_fit(cls, model, params, chains, draws, cohorts):
        """Mock a fitted model with multi-dim parameters."""
        # Generate arrays from true parameters for alpha and beta
        alpha_beta_sim = [
            cls.rng.normal(
                param,
                1e-3,
                size=(chains, draws, len(cohorts)),
            )
            for param in params
        ]
        # Mock posterior
        param_arrays = [
            xr.DataArray(
                param[0],
                dims=("chains", "draws", "cohort"),
                coords={
                    "chains": np.arange(chains),
                    "draws": np.arange(draws),
                    "cohort": cohorts,
                },
                name=param[1],
            )
            for param in zip(alpha_beta_sim, ["alpha", "beta"], strict=False)
        ]
        posterior = az.convert_to_inference_data(xr.merge(param_arrays))
        # Set idata and add fit data group
        model.idata = posterior
        model.set_idata_attrs(model.idata)
        if model.data is not None:
            model._add_fit_data_group(model.data)

        return model

    @pytest.fixture(scope="class")
    def custom_model_config(self):
        return {
            "alpha": Prior("HalfNormal", sigma=10, dims="cohort"),
            "beta": Prior("HalfStudentT", nu=4, sigma=10, dims="cohort"),
        }

    def test_model(self, custom_model_config):
        default_model = ShiftedBetaGeoModel(
            data=self.data,
        )
        custom_model = ShiftedBetaGeoModel(
            data=self.data,
            model_config=custom_model_config,
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
            assert model.model.coords == {
                "customer_id": tuple(range(1, self.N + 1)),
                "cohort": ("highend", "regular"),
            }

        assert default_model.model.eval_rv_shapes() == {
            "kappa": (np.int64(2),),
            "kappa_interval__": (np.int64(2),),
            "phi": (np.int64(2),),
            "phi_interval__": (np.int64(2),),
        }
        assert custom_model.model.eval_rv_shapes() == {
            "alpha": (np.int64(2),),
            "alpha_log__": (np.int64(2),),
            "beta": (np.int64(2),),
            "beta_log__": (np.int64(2),),
        }

    @pytest.fixture(scope="class")
    def prediction_targets(self):
        """Validation data for testing expected_probability_alive and expected_retention_rate."""
        # Data found in Table 1 on pg(3) of paper: https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf
        df = pd.DataFrame(
            {
                "regular": [
                    100.0,
                    63.1,
                    46.8,
                    38.2,
                    32.6,
                    28.9,
                    26.2,
                    24.1,
                    22.3,
                    20.7,
                    19.4,
                    18.3,
                    17.3,
                ],
                "highend": [
                    100.0,
                    86.9,
                    74.3,
                    65.3,
                    59.3,
                    55.1,
                    51.7,
                    49.1,
                    46.8,
                    44.5,
                    42.7,
                    40.9,
                    39.4,
                ],
            }
        )
        alive_prob_highend_obs = df["highend"].values / 100
        alive_prob_regular_obs = df["regular"].values / 100
        retention_rate_highend_obs = (
            alive_prob_highend_obs[1:] / alive_prob_highend_obs[:-1]
        )
        retention_rate_regular_obs = (
            alive_prob_regular_obs[1:] / alive_prob_regular_obs[:-1]
        )

        return (
            alive_prob_highend_obs,
            alive_prob_regular_obs,
            retention_rate_highend_obs,
            retention_rate_regular_obs,
        )

    def test_missing_cols(self):
        data_invalid = self.data.drop(columns="customer_id")

        with pytest.raises(
            ValueError,
            match=r"The following required columns are missing from the input data: \['customer_id'\]",
        ):
            ShiftedBetaGeoModel(data=data_invalid)

        data_invalid = self.data.drop(columns="recency")

        with pytest.raises(
            ValueError,
            match=r"The following required columns are missing from the input data: \['recency'\]",
        ):
            ShiftedBetaGeoModel(data=data_invalid)

        data_invalid = self.data.drop(columns="T")

        with pytest.raises(
            ValueError,
            match=r"The following required columns are missing from the input data: \['T'\]",
        ):
            ShiftedBetaGeoModel(data=data_invalid)

    def test_customer_id_duplicate(self):
        with pytest.raises(
            ValueError, match=r"Column customer_id has duplicate entries"
        ):
            data = pd.DataFrame(
                {
                    "customer_id": np.asarray([1, 1]),
                    "recency": np.asarray([1, 1]),
                    "T": np.asarray([1, 1]),
                    "cohort": np.asarray(["A", "A"]),
                }
            )
            ShiftedBetaGeoModel(data=data)

    def test_invalid_recency(self):
        data = pd.DataFrame(
            {
                "customer_id": np.asarray([1, 2]),
                "recency": np.asarray([1, 3]),
                "T": np.asarray([2, 2]),
                "cohort": np.asarray(["A", "A"]),
            }
        )
        with pytest.raises(
            ValueError, match=r"Model fitting requires 1 <= recency <= T, and T >= 2."
        ):
            ShiftedBetaGeoModel(data=data)

    def test_invalid_T(self):
        data = pd.DataFrame(
            {
                "customer_id": np.asarray([1, 2]),
                "recency": np.asarray([1, 1]),
                "T": np.asarray([1, 1]),
                "cohort": np.asarray(["A", "A"]),
            }
        )
        with pytest.raises(
            ValueError, match=r"Model fitting requires 1 <= recency <= T, and T >= 2."
        ):
            ShiftedBetaGeoModel(data=data)

    def test_cohort_T_homogeneity(self):
        data = pd.DataFrame(
            {
                "customer_id": np.asarray([1, 2]),
                "recency": np.asarray([1, 1]),
                "T": np.asarray([2, 3]),
                "cohort": np.asarray(["A", "A"]),
            }
        )
        with pytest.raises(
            ValueError, match=r"T must be homogeneous within each cohort."
        ):
            ShiftedBetaGeoModel(data=data)

    def test_model_repr(self, custom_model_config):
        default_repr = (
            "ShiftedBeta-Geometric"
            "\nphi~Uniform(0,1)"
            "\nkappa~Pareto(1,1)"
            "\nalpha~Deterministic(f(kappa,phi))"
            "\nbeta~Deterministic(f(kappa,phi))"
            "\ndropout~Censored(ShiftedBetaGeometric(f(kappa,phi),f(kappa,phi)),-inf,<constant>)"
        )

        custom_repr = (
            "ShiftedBeta-Geometric"
            "\nalpha~HalfNormal(0,10)"
            "\nbeta~HalfStudentT(4,10)"
            "\ndropout~Censored(ShiftedBetaGeometric(f(alpha),f(beta)),-inf,<constant>)"
        )

        for repr in zip(
            [custom_model_config, None], [custom_repr, default_repr], strict=False
        ):
            model = ShiftedBetaGeoModel(
                data=self.data,
                model_config=repr[0],
            )
            model.build_model()
            assert model.__repr__().replace(" ", "") == repr[1]

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "method, rtol",
        [
            ("mcmc", 0.1),
            ("map", 0.2),
            ("advi", 0.25),
        ],
    )
    def test_model_convergence(self, method, rtol, custom_model_config):
        model = ShiftedBetaGeoModel(
            data=self.data,
            model_config=custom_model_config,
        )
        model.build_model()

        sample_kwargs = dict(random_seed=self.seed) if method == "mcmc" else {}
        model.fit(method=method, progressbar=False, **sample_kwargs)

        fit = model.idata.posterior
        np.testing.assert_allclose(
            [fit["alpha"].mean(), fit["beta"].mean()],
            [self.alpha_true, self.beta_true],
            rtol=rtol,
        )

    def test_fit_result_without_fit(self, mocker, custom_model_config):
        model = ShiftedBetaGeoModel(data=self.data, model_config=custom_model_config)
        with pytest.raises(RuntimeError, match=r"The model hasn't been fit yet"):
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

    def test_expected_probability_alive(self, prediction_targets):
        alive_prob_highend_obs, alive_prob_regular_obs, _, _ = prediction_targets
        # TODO: need full array comparison
        expected_alive_prob_highend = (
            self.model.expected_probability_alive(future_t=0)
            .sel(cohort="highend")
            .mean()
        )
        expected_alive_prob_regular = (
            self.model.expected_probability_alive(future_t=0)
            .sel(cohort="regular")
            .mean()
        )
        np.testing.assert_allclose(
            expected_alive_prob_highend, alive_prob_highend_obs, rtol=0.01
        )
        np.testing.assert_allclose(
            expected_alive_prob_regular, alive_prob_regular_obs, rtol=0.01
        )

    def test_expected_retention_rate(self, prediction_targets):
        _, _, retention_rate_highend_obs, retention_rate_regular_obs = (
            prediction_targets
        )
        # TODO: need full array comparison
        expected_retention_rate_highend = (
            self.model.expected_retention_rate(future_t=0).sel(cohort="highend").mean()
        )
        expected_retention_rate_regular = (
            self.model.expected_retention_rate(self.data, future_t=0)
            .sel(cohort="regular")
            .mean()
        )
        np.testing.assert_allclose(
            expected_retention_rate_highend, retention_rate_highend_obs, rtol=0.01
        )
        np.testing.assert_allclose(
            expected_retention_rate_regular, retention_rate_regular_obs, rtol=0.01
        )

    def test_save_load(self):
        model = ShiftedBetaGeoModel(data=self.data)
        model.build_model()
        model.fit(method="map")
        model.save("test_model")
        model2 = ShiftedBetaGeoModel.load("test_model")
        assert model.model_config == model2.model_config
        assert model.sampler_config == model2.sampler_config
        assert model.idata == model2.idata
        os.remove("test_model")

    def test_requires_cohort_dims_on_alpha_beta_missing_raises(self):
        config_missing_dims = {
            "alpha": Prior("HalfNormal", sigma=10),  # missing dims
            "beta": Prior("HalfStudentT", nu=4, sigma=10, dims="cohort"),
        }
        with pytest.raises(ValueError, match=r'dims="cohort"'):
            ShiftedBetaGeoModel(
                data=self.data,
                model_config=config_missing_dims,
            )

    def test_requires_cohort_dims_on_alpha_beta_incorrect_raises(self):
        config_incorrect_dims = {
            "alpha": Prior("HalfNormal", sigma=10, dims="customer_id"),
            "beta": Prior("HalfStudentT", nu=4, sigma=10, dims="cohort"),
        }
        with pytest.raises(ValueError, match=r'dims="cohort"'):
            ShiftedBetaGeoModel(
                data=self.data,
                model_config=config_incorrect_dims,
            )

    def test_accepts_alpha_beta_with_cohort_dims(self):
        config_ok = {
            "alpha": Prior("HalfNormal", sigma=10, dims="cohort"),
            "beta": Prior("HalfStudentT", nu=4, sigma=10, dims="dim"),
        }
        with pytest.raises(ValueError, match=r'dims="cohort"'):
            ShiftedBetaGeoModel(
                data=self.data,
                model_config=config_ok,
            )

    def test_extract_predictive_variables_invalid(self):
        hetero_T_data = pd.DataFrame(
            {
                "customer_id": np.asarray([1, 2]),
                "recency": np.asarray([1, 1]),
                "T": np.asarray([2, 3]),
                "cohort": np.asarray(["A", "A"]),
            }
        )
        with pytest.raises(
            ValueError,
            match=r"T must be homogeneous within each cohort. The following cohorts have multiple T values: A",
        ):
            self.model._extract_predictive_variables(
                hetero_T_data,
                customer_varnames=["customer_id", "recency", "T", "cohort"],
            )

        T_lt_2_data = pd.DataFrame(
            {
                "customer_id": np.asarray([1, 2]),
                "recency": np.asarray([1, 1]),
                "T": np.asarray([1, 1]),
                "cohort": np.asarray(["A", "A"]),
            }
        )
        with pytest.raises(ValueError, match=r"T must be a positive integer >= 2."):
            self.model._extract_predictive_variables(
                T_lt_2_data, customer_varnames=["customer_id", "recency", "T", "cohort"]
            )


class TestShiftedBetaGeoModelIndividual:
    @classmethod
    def setup_class(cls):
        def churned_data_from_percentage_alive(percentage_alive, initial_customers):
            n_alive = (np.asarray(percentage_alive) / 100 * initial_customers).astype(
                int
            )

            churned_at = np.zeros((initial_customers,), dtype=int)
            counter = 0
            for t, diff in enumerate((n_alive[:-1] - n_alive[1:]), start=1):
                churned_at[counter : counter + diff] = t
                counter += diff

            censoring_t = t + 1
            churned_at[counter:] = censoring_t

            return churned_at

        # Regular dataset from Fader, P. S., & Hardie, B. G. (2007). How to project customer retention.
        # Journal of Interactive Marketing, 21(1), 76-90. https://journals.sagepub.com/doi/pdf/10.1002/dir.20074
        cls.N = 1000
        cls.T = 8
        cls.customer_id = np.arange(cls.N)
        cls.churn_time = churned_data_from_percentage_alive(
            percentage_alive=[100.0, 63.1, 46.8, 38.2, 32.6, 28.9, 26.2, 24.1],
            initial_customers=cls.N,
        )
        cls.ref_MLE_estimates = {"alpha": 0.688, "beta": 1.182}

    @pytest.fixture(scope="class")
    def model_config(self):
        return {
            "alpha": Prior("HalfNormal", sigma=10, dims="cohort"),
            "beta": Prior("HalfStudentT", nu=4, sigma=10, dims="cohort"),
        }

    @pytest.fixture(scope="class")
    def default_model_config(self):
        return {
            "alpha": Prior("HalfFlat", dims="cohort"),
            "beta": Prior("HalfFlat", dims="cohort"),
        }

    @pytest.fixture(scope="class")
    def data(self):
        return pd.DataFrame(
            {
                "customer_id": self.customer_id,
                "t_churn": self.churn_time,
                "T": self.T,
            }
        )

    def test_missing_cols(self, data):
        # Create a version of the data that's missing the 'customer_id' column
        data_invalid = data.drop(columns="customer_id")

        with pytest.raises(
            ValueError,
            match=r"The following required columns are missing from the input data: \['customer_id'\]",
        ):
            ShiftedBetaGeoModelIndividual(data=data_invalid)

        data_invalid = data.drop(columns="t_churn")

        with pytest.raises(
            ValueError,
            match=r"The following required columns are missing from the input data: \['t_churn'\]",
        ):
            ShiftedBetaGeoModelIndividual(data=data_invalid)

        data_invalid = data.drop(columns="T")

        with pytest.raises(
            ValueError,
            match=r"The following required columns are missing from the input data: \['T'\]",
        ):
            ShiftedBetaGeoModelIndividual(data=data_invalid)

    def test_model_repr(self, default_model_config):
        custom_model_config = default_model_config.copy()
        custom_model_config["alpha"] = Prior("HalfNormal", sigma=10, dims="cohort")
        dataset = pd.DataFrame(
            {"customer_id": self.customer_id, "t_churn": self.churn_time, "T": self.T}
        )
        model = ShiftedBetaGeoModelIndividual(
            data=dataset,
            model_config=custom_model_config,
        )
        model.build_model()
        assert model.__repr__().replace(" ", "") == (
            "Shifted-Beta-GeometricModel(IndividualCustomers)"
            "\nalpha~HalfNormal(0,10)"
            "\nbeta~HalfFlat()"
            "\ntheta~Beta(alpha,beta)"
            "\nchurn_censored~Censored(Geometric(theta),-inf,<constant>)"
        )

    def test_model(self, model_config, default_model_config, data):
        for config in (model_config, default_model_config):
            model = ShiftedBetaGeoModelIndividual(
                data=data,
                model_config=config,
            )
            model.build_model()
            assert isinstance(
                model.model["alpha"].owner.op,
                pm.HalfFlat
                if config["alpha"].distribution == "HalfFlat"
                else config["alpha"].pymc_distribution,
            )
            assert isinstance(
                model.model["beta"].owner.op,
                pm.HalfFlat
                if config["beta"].distribution == "HalfFlat"
                else config["beta"].pymc_distribution,
            )
            assert isinstance(model.model["theta"].owner.op, pm.Beta)
            assert isinstance(model.model["churn_censored"].owner.op, CensoredRV)
            assert isinstance(
                model.model["churn_censored"].owner.inputs[0].owner.op, pm.Geometric
            )
            assert model.model.eval_rv_shapes() == {
                "alpha": (),
                "alpha_log__": (),
                "beta": (),
                "beta_log__": (),
                "theta": (self.N,),
                "theta_logodds__": (self.N,),
            }
            assert model.model.coords == {
                "customer_id": tuple(range(self.N)),
            }

    def test_invalid_t_churn(self, default_model_config):
        match_msg = "t_churn must respect 0 < t_churn <= T"
        dataset = {
            "customer_id": range(3),
            "t_churn": [10, 10, np.nan],
            "T": 10,
        }

        dataset["t_churn"] = [10, 10, np.nan]
        with pytest.raises(ValueError, match=match_msg):
            ShiftedBetaGeoModelIndividual(
                data=pd.DataFrame(dataset), model_config=default_model_config
            )
        dataset["t_churn"] = [10, 10, 11]
        with pytest.raises(ValueError, match=match_msg):
            ShiftedBetaGeoModelIndividual(
                data=pd.DataFrame(dataset), model_config=default_model_config
            )
        dataset["t_churn"] = [-1, 8, 9]
        dataset["T"] = [8, 9, 10]
        with pytest.raises(ValueError, match=match_msg):
            ShiftedBetaGeoModelIndividual(
                data=pd.DataFrame(dataset),
            )

    @pytest.mark.slow
    def test_model_convergence(self, data, model_config):
        model = ShiftedBetaGeoModelIndividual(
            data=data,
            model_config=model_config,
        )
        model.build_model()
        model.fit(chains=2, progressbar=False, random_seed=100)
        fit = model.idata.posterior
        np.testing.assert_allclose(
            [fit["alpha"].mean(), fit["beta"].mean()],
            [self.ref_MLE_estimates["alpha"], self.ref_MLE_estimates["beta"]],
            rtol=0.1,
        )

    def test_distribution_cohort_churn(self):
        dataset = pd.DataFrame(
            {
                "customer_id": [0, 1, 2],
                "t_churn": [10, 10, 10],
                "T": 10,
            }
        )
        model = ShiftedBetaGeoModelIndividual(
            data=dataset,
        )
        model.build_model()
        model.fit(method="map")
        customer_thetas = np.array([0.1, 0.5, 0.9])
        model.idata = az.from_dict(
            posterior={
                "alpha": np.ones((2, 500)),  # Two chains, 500 draws each
                "beta": np.ones((2, 500)),
                "theta": np.full((2, 500, 3), customer_thetas),
            },
            coords={"customer_id": [0, 1, 2]},
            dims={"theta": ["customer_id"]},
        )

        res = model.distribution_cohort_churn(customer_id=[0, 1, 2], random_seed=116)
        np.testing.assert_allclose(
            res.mean(("chain", "draw")),
            stats.geom(customer_thetas).mean(),
            rtol=0.05,
        )

    def test_distribution_new_customer(self):
        dataset = pd.DataFrame(
            {
                "customer_id": [1],
                "t_churn": [10],
                "T": [10],
            }
        )
        model = ShiftedBetaGeoModelIndividual(
            data=dataset,
        )
        model.build_model()
        model.fit(method="map")
        # theta ~ beta(7000, 3000) ~ 0.7
        model.idata = az.from_dict(
            {
                "alpha": np.full((2, 500), 7000),  # Two chains, 500 draws each
                "beta": np.full((2, 500), 3000),
            }
        )

        res = model.distribution_new_customer_theta(random_seed=141)
        np.testing.assert_allclose(res.mean(("chain", "draw")), 0.7, rtol=0.001)

        res = model.distribution_new_customer_churn_time(n=2, random_seed=146)
        np.testing.assert_allclose(
            res.mean(("chain", "draw", "new_customer_id")),
            stats.geom(0.7).mean(),
            rtol=0.05,
        )

    def test_save_load(self, data):
        model = ShiftedBetaGeoModelIndividual(
            data=data,
        )
        model.build_model()
        model.fit("map", maxeval=1)
        model.save("test_model")
        # Testing the valid case.
        model2 = ShiftedBetaGeoModelIndividual.load("test_model")
        # Check if the loaded model is indeed an instance of the class
        assert isinstance(model, ShiftedBetaGeoModelIndividual)
        # Check if the loaded data matches with the model data
        pd.testing.assert_frame_equal(model.data, model2.data, check_names=False)
        assert model.model_config == model2.model_config
        assert model.sampler_config == model2.sampler_config
        assert model.idata == model2.idata
        os.remove("test_model")
