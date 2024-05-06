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
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray
from pymc.util import RandomState
from pytensor.tensor import TensorVariable

from pymc_marketing.clv.models.basic import CLVModel
from pymc_marketing.clv.utils import customer_lifetime_value, to_xarray


class BaseGammaGammaModel(CLVModel):
    def distribution_customer_spend(
        self,
        customer_id: np.ndarray | pd.Series,
        mean_transaction_value: np.ndarray | pd.Series | TensorVariable,
        frequency: np.ndarray | pd.Series | TensorVariable,
        random_seed: RandomState | None = None,
    ) -> xarray.DataArray:
        """Posterior distribution of transaction value per customer"""

        x = frequency
        z_mean = mean_transaction_value

        coords = {"customer_id": np.unique(customer_id)}
        with pm.Model(coords=coords):
            p = pm.HalfFlat("p")
            q = pm.HalfFlat("q")
            v = pm.HalfFlat("v")

            # Closed form solution to the posterior of nu
            # Eq 5 from [1], p.3
            nu = pm.Gamma("nu", p * x + q, v + x * z_mean, dims=("customer_id",))
            pm.Deterministic("mean_spend", p / nu, dims=("customer_id",))

            return pm.sample_posterior_predictive(
                self.idata,
                var_names=["nu", "mean_spend"],
                random_seed=random_seed,
            ).posterior_predictive["mean_spend"]

    def expected_customer_spend(
        self,
        customer_id: np.ndarray | pd.Series,
        mean_transaction_value: np.ndarray | pd.Series,
        frequency: np.ndarray | pd.Series,
    ) -> xarray.DataArray:
        """Expected transaction value per customer

        Eq 5 from [1], p.3

        Adapted from: https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/lifetimes/fitters/gamma_gamma_fitter.py#L117
        """

        mean_transaction_value, frequency = to_xarray(
            customer_id, mean_transaction_value, frequency
        )
        posterior = self.fit_result

        p = posterior["p"]
        q = posterior["q"]
        v = posterior["v"]

        individual_weight = p * frequency / (p * frequency + q - 1)
        population_mean = v * p / (q - 1)
        return (
            1 - individual_weight
        ) * population_mean + individual_weight * mean_transaction_value

    def distribution_new_customer_spend(
        self, n=1, random_seed=None
    ) -> xarray.DataArray:
        """Posterior distribution of transaction value for new customers"""
        coords = {"new_customer_id": range(n)}
        with pm.Model(coords=coords):
            p = pm.HalfFlat("p")
            q = pm.HalfFlat("q")
            v = pm.HalfFlat("v")

            nu = pm.Gamma("nu", q, v, dims=("new_customer_id",))
            pm.Deterministic("mean_spend", p / nu, dims=("new_customer_id",))

            return pm.sample_posterior_predictive(
                self.idata,
                var_names=["nu", "mean_spend"],
                random_seed=random_seed,
            ).posterior_predictive["mean_spend"]

    def expected_new_customer_spend(self) -> xarray.DataArray:
        """Expected transaction value for a new customer"""

        posterior = self.fit_result
        p_mean = posterior["p"]
        q_mean = posterior["q"]
        v_mean = posterior["v"]

        # Closed form solution to the posterior of nu
        # Eq 3 from [1], p.3
        mean_spend = p_mean * v_mean / (q_mean - 1)
        # We could also provide the variance
        # var_spend = (p_mean ** 2 * v_mean ** 2) / ((q_mean - 1) ** 2 * (q_mean - 2))

        return mean_spend

    def expected_customer_lifetime_value(
        self,
        transaction_model: CLVModel,
        customer_id: np.ndarray | pd.Series,
        mean_transaction_value: np.ndarray | pd.Series,
        frequency: np.ndarray | pd.Series,
        recency: np.ndarray | pd.Series,
        T: np.ndarray | pd.Series,
        time: int = 12,
        discount_rate: float = 0.01,
        freq: str = "D",
    ) -> xarray.DataArray:
        """Expected customer lifetime value.

        See clv.utils.customer_lifetime_value for details on the meaning of each parameter
        """

        # Use the Gamma-Gamma estimates for the monetary_values
        adjusted_monetary_value = self.expected_customer_spend(
            customer_id=customer_id,
            mean_transaction_value=mean_transaction_value,
            frequency=frequency,
        )

        return customer_lifetime_value(
            transaction_model=transaction_model,
            customer_id=customer_id,
            frequency=frequency,
            recency=recency,
            T=T,
            monetary_value=adjusted_monetary_value,
            time=time,
            discount_rate=discount_rate,
            freq=freq,
        )


class GammaGammaModel(BaseGammaGammaModel):
    """Gamma-Gamma model

    Estimate the average monetary value of customer transactions.

    The model is conditioned on the mean transaction value of each user, and is based
    on [1]_, [2]_.

    TODO: Explain assumptions of model

    Check `GammaGammaModelIndividual` for an equivalent model conditioned
    on individual transaction values.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing the following columns:
            - customer_id: Customer labels. Must not repeat.
            - mean_transaction_value: Mean transaction value of each customer.
            - frequency: Number of transactions observed for each customer.
    model_config: dict, optional
        Dictionary of model prior parameters. If not provided, the model will use default priors specified in the
        `default_model_config` class attribute.
    sampler_config: dict, optional
        Dictionary of sampler parameters. Defaults to None.

    Examples
    --------
        Gamma-Gamma model condioned on mean transaction value

        .. code-block:: python

            import pymc as pm
            from pymc_marketing.clv import GammaGammaModel

            model = GammaGammaModel(
                data=pd.DataFrame({
                    "customer_id": [0, 1, 2, 3, ...],
                    "mean_transaction_value" :[23.5, 19.3, 11.2, 100.5, ...],
                    "frequency": [6, 8, 2, 1, ...],
                }),
                model_config={
                    "p_prior": {'dist': 'HalfNormal', kwargs: {}},
                    "q_prior": {'dist': 'HalfStudentT', kwargs: {"nu": 4, "sigma": 10}},
                    "v_prior": {'dist': 'HalfCauchy', kwargs: {"beta":1}},
                },
                sampler_config={
                    "draws": 1000,
                    "tune": 1000,
                    "chains": 2,
                    "cores": 2,
                    "nuts_kwargs": {"target_accept": 0.95},
                },
            )
            model.build_model()
            model.fit()
            print(model.fit_summary())

            # Predict spend of customers for which we know transaction history, conditioned on data.
            expected_customer_spend = model.expected_customer_spend(
                customer_id=[0, 1, 2, 3, ...],
                mean_transaction_value=[23.5, 19.3, 11.2, 100.5, ...],
                frequency=[6, 8, 2, 1, ...],
            )
            print(expected_customer_spend.mean("customer_id"))

            # Predict spend of 10 new customers, conditioned on data
            new_customer_spend = model.expected_new_customer_spend(n=10)
            print(new_customer_spend.mean("new_customer_id"))

    References
    ----------
    .. [1] Fader, P. S., & Hardie, B. G. (2013). The Gamma-Gamma model of monetary
           value. February, 2, 1-9. http://www.brucehardie.com/notes/025/gamma_gamma.pdf
    .. [2] Peter S. Fader, Bruce G. S. Hardie, and Ka Lok Lee (2005), “RFM and CLV:
           Using iso-value curves for customer base analysis”, Journal of Marketing
           Research, 42 (November), 415-430.
           https://journals.sagepub.com/doi/pdf/10.1509/jmkr.2005.42.4.415
    """

    _model_type = "Gamma-Gamma Model (Mean Transactions)"

    def __init__(
        self,
        data: pd.DataFrame,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ):
        self._validate_cols(
            data,
            required_cols=["customer_id", "mean_transaction_value", "frequency"],
            must_be_unique=["customer_id"],
        )
        super().__init__(
            data=data, model_config=model_config, sampler_config=sampler_config
        )

    @property
    def default_model_config(self) -> dict:
        return {
            "p_prior": {"dist": "HalfFlat", "kwargs": {}},
            "q_prior": {"dist": "HalfFlat", "kwargs": {}},
            "v_prior": {"dist": "HalfFlat", "kwargs": {}},
        }

    def build_model(self):
        z_mean = pt.as_tensor_variable(self.data["mean_transaction_value"])
        x = pt.as_tensor_variable(self.data["frequency"])

        p_prior = self._create_distribution(self.model_config["p_prior"])
        q_prior = self._create_distribution(self.model_config["q_prior"])
        v_prior = self._create_distribution(self.model_config["v_prior"])

        coords = {"customer_id": self.data["customer_id"]}
        with pm.Model(coords=coords) as self.model:
            p = self.model.register_rv(p_prior, name="p")
            q = self.model.register_rv(q_prior, name="q")
            v = self.model.register_rv(v_prior, name="v")
            # Likelihood for mean_spend, marginalizing over nu
            # Eq 1a from [1], p.2
            pm.Potential(
                "likelihood",
                (
                    pt.gammaln(p * x + q)
                    - pt.gammaln(p * x)
                    - pt.gammaln(q)
                    + q * pt.log(v)
                    + (p * x - 1) * pt.log(z_mean)
                    + (p * x) * pt.log(x)
                    - (p * x + q) * pt.log(x * z_mean + v)
                ),
            )


class GammaGammaModelIndividual(BaseGammaGammaModel):
    """Gamma-Gamma model

    Estimate the average monetary value of customer transactions.

    The model is conditioned on the individual transaction values per user, and is based
    on [1]_, [2]_.

    TODO: Explain assumptions of model

    Check `GammaGammaModel` for an equivalent model conditioned on the average
    transaction value of each user.

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe containing the following columns:
            - customer_id: Customer labels. The same value should be used for each observation
        coming from the same customer.
            - individual_transaction_value: Value of individual transactions.
    model_config: dict, optional
        Dictionary of model prior parameters. If not provided, the model will use default priors specified in the
        `default_model_config` class attribute.
    sampler_config: dict, optional
        Dictionary of sampler parameters. Defaults to None.


    Examples
    --------

        Gamma-Gamma model conditioned on individual customer spend

        .. code-block:: python

            import pymc as pm
            from pymc_marketing.clv import GammaGammaModelIndividual

            model = GammaGammaModelIndividual(
                data=pd.DataFrame({
                    "customer_id": [0, 0, 0, 1, 1, 2, ...],
                    "individual_transaction_value": [5.3. 5.7, 6.9, 13.5, 0.3, 19.2 ...],
                }),
                model_config={
                    "p_prior": {dist: 'HalfNorm', kwargs: {}},
                    "q_prior": {dist: 'HalfStudentT', kwargs: {"nu": 4, "sigma": 10}},
                    "v_prior": {dist: 'HalfCauchy', kwargs: {}},
                },
                sampler_config={
                    "draws": 1000,
                    "tune": 1000,
                    "chains": 2,
                    "cores": 2,
                    "nuts_kwargs": {"target_accept": 0.95},
                },
            )
            model.build_model()
            model.fit()
            print(model.fit_summary())

            # Predict spend of customers for which we know transaction history,
            # conditioned on data. May include customers not included in fitting
            expected_customer_spend = model.expected_customer_spend(
                customer_id=[0, 0, 0, 1, 1, 2, ...],
                individual_transaction_value=[5.3. 5.7, 6.9, 13.5, 0.3, 19.2 ...],
            )
            print(expected_customer_spend.mean("customer_id"))

            # Predict spend of 10 new customers, conditioned on data
            new_customer_spend = model.expected_new_customer_spend(n=10)
            print(new_customer_spend.mean("new_customer_id"))


    References
    ----------
    .. [1] Fader, P. S., & Hardie, B. G. (2013). The Gamma-Gamma model of monetary
           value. February, 2, 1-9. http://www.brucehardie.com/notes/025/gamma_gamma.pdf
    .. [2] Peter S. Fader, Bruce G. S. Hardie, and Ka Lok Lee (2005), “RFM and CLV:
           Using iso-value curves for customer base analysis”, Journal of Marketing
           Research, 42 (November), 415-430.
           https://journals.sagepub.com/doi/pdf/10.1509/jmkr.2005.42.4.415
    """

    _model_type = "Gamma-Gamma Model (Individual Transactions)"

    def __init__(
        self,
        data: pd.DataFrame,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ):
        self._validate_cols(
            data, required_cols=["customer_id", "individual_transaction_value"]
        )
        super().__init__(
            data=data, model_config=model_config, sampler_config=sampler_config
        )

    @property
    def default_model_config(self) -> dict:
        return {
            "p_prior": {"dist": "HalfFlat", "kwargs": {}},
            "q_prior": {"dist": "HalfFlat", "kwargs": {}},
            "v_prior": {"dist": "HalfFlat", "kwargs": {}},
        }

    def build_model(self):
        z = self.data["individual_transaction_value"]

        p_prior = self._create_distribution(self.model_config["p_prior"])
        q_prior = self._create_distribution(self.model_config["q_prior"])
        v_prior = self._create_distribution(self.model_config["v_prior"])

        coords = {
            "customer_id": np.unique(self.data["customer_id"]),
            "obs": range(self.data.shape[0]),
        }
        with pm.Model(coords=coords) as self.model:
            p = self.model.register_rv(p_prior, name="p")
            q = self.model.register_rv(q_prior, name="q")
            v = self.model.register_rv(v_prior, name="v")

            nu = pm.Gamma("nu", q, v, dims=("customer_id",))
            pm.Gamma(
                "spend", p, nu[self.data["customer_id"]], observed=z, dims=("obs",)
            )

    def _summarize_mean_data(self, customer_id, individual_transaction_value):
        df = pd.DataFrame(
            {
                "customer_id": customer_id,
                "individual_transaction_value": individual_transaction_value,
            }
        )
        gdf = df.groupby("customer_id")["individual_transaction_value"].aggregate(
            ("count", "mean")
        )
        customer_id = gdf.index
        x = gdf["count"]
        z_mean = gdf["mean"]

        return customer_id, z_mean, x

    def distribution_customer_spend(  # type: ignore [override]
        self,
        customer_id: np.ndarray | pd.Series,
        individual_transaction_value: np.ndarray | pd.Series | TensorVariable,
        random_seed: RandomState | None = None,
    ) -> xarray.DataArray:
        """Return distribution of transaction value per customer"""

        customer_id, z_mean, x = self._summarize_mean_data(
            customer_id, individual_transaction_value
        )

        return super().distribution_customer_spend(
            customer_id=customer_id,
            mean_transaction_value=z_mean,
            frequency=x,
            random_seed=random_seed,
        )

    def expected_customer_spend(
        self,
        customer_id: np.ndarray | pd.Series,
        individual_transaction_value: np.ndarray | pd.Series | TensorVariable,
        random_seed: RandomState | None = None,
    ) -> xarray.DataArray:
        """Return expected transaction value per customer"""

        customer_id, z_mean, x = self._summarize_mean_data(
            customer_id, individual_transaction_value
        )

        return super().expected_customer_spend(
            customer_id=customer_id,
            mean_transaction_value=z_mean,
            frequency=x,
            random_seed=random_seed,  # type: ignore [call-arg]
        )

    def expected_customer_lifetime_value(  # type: ignore [override]
        self,
        transaction_model: CLVModel,
        customer_id: np.ndarray | pd.Series,
        individual_transaction_value: np.ndarray | pd.Series | TensorVariable,
        recency: np.ndarray | pd.Series,
        T: np.ndarray | pd.Series,
        time: int = 12,
        discount_rate: float = 0.01,
        freq: str = "D",
    ) -> xarray.DataArray:
        """Return expected customer lifetime value.

        See clv.utils.customer_lifetime_value for details on the meaning of each parameter
        """

        customer_id, z_mean, x = self._summarize_mean_data(
            customer_id, individual_transaction_value
        )

        return super().expected_customer_lifetime_value(
            transaction_model=transaction_model,
            customer_id=customer_id,
            mean_transaction_value=z_mean,
            frequency=x,
            recency=recency,
            T=T,
            time=time,
            discount_rate=discount_rate,
            freq=freq,
        )
