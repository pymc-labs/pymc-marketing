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
from pymc.distributions.censored import CensoredRV
from scipy import stats

from pymc_marketing.clv import ShiftedBetaGeoModelIndividual
from pymc_marketing.prior import Prior


class TestShiftedBetaGeoModel:
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
            "alpha": Prior("HalfNormal", sigma=10),
            "beta": Prior("HalfStudentT", nu=4, sigma=10),
        }

    @pytest.fixture(scope="class")
    def default_model_config(self):
        return {
            "alpha": Prior("HalfFlat"),
            "beta": Prior("HalfFlat"),
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

        with pytest.raises(ValueError, match="Required column customer_id missing"):
            ShiftedBetaGeoModelIndividual(data=data_invalid)

        data_invalid = data.drop(columns="t_churn")

        with pytest.raises(ValueError, match="Required column t_churn missing"):
            ShiftedBetaGeoModelIndividual(data=data_invalid)

        data_invalid = data.drop(columns="T")

        with pytest.raises(ValueError, match="Required column T missing"):
            ShiftedBetaGeoModelIndividual(data=data_invalid)

    def test_model_repr(self, default_model_config):
        custom_model_config = default_model_config.copy()
        custom_model_config["alpha"] = Prior("HalfNormal", sigma=10)
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

    def test_distribution_customer_churn_time(self):
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

        res = model.distribution_customer_churn_time(
            customer_id=[0, 1, 2], random_seed=116
        )
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
