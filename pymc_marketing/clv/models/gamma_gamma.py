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
"""Gamma-Gamma Model for expected future monetary value."""

import numpy as np
import pandas
import pymc as pm
import pytensor.tensor as pt
import xarray
from pymc.util import RandomState

from pymc_marketing.clv.models import CLVModel
from pymc_marketing.clv.utils import customer_lifetime_value, to_xarray
from pymc_marketing.model_config import ModelConfig
from pymc_marketing.prior import Prior


class BaseGammaGammaModel(CLVModel):
    """Base class for Gamma-Gamma models."""

    def distribution_customer_spend(
        self,
        data: pandas.DataFrame,
        random_seed: RandomState | None = None,
    ) -> xarray.DataArray:
        """Posterior distribution of mean spend values for each customer.

        Parameters
        ----------
        data : ~pandas.DataFrame
            DataFrame containing the following columns:

            * `customer_id`: Unique customer identifier
            * `frequency`: Number of purchases
            * `monetary_value`: Mean spend values for each customer

        random_seed : ~RandomState, optional
            Optional random seed to fix sampling results.

        """
        x = data["frequency"]
        z_mean = data["monetary_value"]

        coords = {"customer_id": np.unique(data["customer_id"])}
        with pm.Model(coords=coords):
            p = pm.HalfFlat("p")
            q = pm.HalfFlat("q")
            v = pm.HalfFlat("v")

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
        data: pandas.DataFrame,
    ) -> xarray.DataArray:
        """Compute the expected future mean spend value per customer.

        The computations are based on Eq 5 from [1], p.3.

        Adapted from: https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/lifetimes/fitters/gamma_gamma_fitter.py#L117

        data : ~pandas.DataFrame
            DataFrame containing the following columns:

            * `customer_id`: Unique customer identifier
            * `frequency`: Number of transactions observed for each customer
            * `monetary_value`: Mean transaction value of repeat purchases for each customer

        References
        ----------
        .. [1] Fader, P. S., & Hardie, B. G. (2013). "The Gamma-Gamma model of monetary
               value". February, 2, 1-9. https://www.brucehardie.com/notes/025/gamma_gamma.pdf

        """
        mean_transaction_value, frequency = to_xarray(
            data["customer_id"],
            data["monetary_value"],
            data["frequency"],
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
        self, n: int = 1, random_seed: RandomState | None = None
    ) -> xarray.DataArray:
        """Posterior distribution of mean spend values for new customers.

        Parameters
        ----------
        n : int, optional
            Number of posterior distributions to generate. This can usually be left at the default value of 1.

        random_seed : ~RandomState, optional
            Optional random seed to fix sampling results.

        """
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
        """Compute the expected mean spend value for a new customer."""
        posterior = self.fit_result
        p_mean = posterior["p"]
        q_mean = posterior["q"]
        v_mean = posterior["v"]

        # Closed form solution to the posterior of nu
        # Eq 3 from [1], p.3
        mean_spend = p_mean * v_mean / (q_mean - 1)
        # TODO: We could also provide the variance
        # var_spend = (p_mean ** 2 * v_mean ** 2) / ((q_mean - 1) ** 2 * (q_mean - 2))

        return mean_spend

    def expected_customer_lifetime_value(
        self,
        transaction_model: CLVModel,
        data: pandas.DataFrame,
        future_t: int = 12,
        discount_rate: float = 0.00,
        time_unit: str = "D",
    ) -> xarray.DataArray:
        """Compute the average lifetime value for a group of one or more customers.

        In addition, it applies a discount rate for net present value estimations.

        Note `future_t` is measured in months regardless of `time_unit` specified.

        Adapted from lifetimes package
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/gamma_gamma_fitter.py#L246

        Parameters
        ----------
        transaction_model : ~CLVModel
            Predictive model for future transactions. `BetaGeoModel` and `ParetoNBDModel` are currently supported.
        data : ~pandas.DataFrame
            DataFrame containing the following columns:

            * `customer_id`: Unique customer identifier
            * `frequency`: Number of repeat purchases observed for each customer
            * `recency`: Time between the first and the last purchase
            * `T`: Time between the first purchase and the end of the observation period
            * `monetary_value`: Mean spend values of repeat purchases for each customer
        future_t : int, optional
            The lifetime expected for the user in months. Default: 12
        discount_rate : float, optional
            The monthly adjusted discount rate. Default: 0.00
        time_unit : string, optional
            Unit of time of the purchase history. Defaults to "D" for daily.
            Other options are "W" (weekly), "M" (monthly), and "H" (hourly).
            Example: If your dataset contains information about weekly purchases,
            you should use "W".

        Returns
        -------
        xarray
            DataArray containing estimated customer lifetime values

        """
        # Use Gamma-Gamma estimates for the expected_spend values
        predicted_monetary_value = self.expected_customer_spend(data=data)
        data.loc[:, "future_spend"] = predicted_monetary_value.mean(
            ("chain", "draw")
        ).copy()

        return customer_lifetime_value(
            transaction_model=transaction_model,
            data=data,
            future_t=future_t,
            discount_rate=discount_rate,
            time_unit=time_unit,
        )


class GammaGammaModel(BaseGammaGammaModel):
    """Gamma-Gamma Model for expected future monetary value.

    The Gamma-Gamma model assumes expected future spend follows a Gamma distribution,
    and the scale of this distribution is also Gamma-distributed.

    This model is conditioned on the mean value of repeat transactions for each customer, and is based
    on [1]_, [2]_. Data must be summarized by *frequency* and *monetary_value* for each customer,
    using `clv.rfm_summary()` or equivalent.

    See `GammaGammaModelIndividual` for an equivalent model conditioned
    on individual transaction values.

    Parameters
    ----------
    data : ~pandas.DataFrame
        DataFrame containing the following columns:

        * `customer_id`: Unique customer identifier
        * `monetary_value`: Mean transaction value of repeat purchases for each customer
        * `frequency`: Number of repeat transactions observed for each customer
    model_config : dict, optional
        Dictionary of model prior parameters. If not provided, the model will use default priors specified in the
        `default_model_config` class attribute.
    sampler_config : dict, optional
        Dictionary of sampler parameters. Defaults to *None*.

    Examples
    --------

    .. code-block:: python

        import pymc as pm
        from pymc_marketing.clv import GammaGammaModel

        model = GammaGammaModel(
            data=pandas.DataFrame(
                {
                    "customer_id": [0, 1, 2, 3, ...],
                    "monetary_value": [23.5, 19.3, 11.2, 100.5, ...],
                    "frequency": [6, 8, 2, 1, ...],
                }
            ),
            model_config={
                "p": {"dist": "HalfNormal", kwargs: {}},
                "q": {"dist": "HalfStudentT", kwargs: {"nu": 4, "sigma": 10}},
                "v": {"dist": "HalfCauchy", kwargs: {"beta": 1}},
            },
            sampler_config={
                "draws": 1000,
                "tune": 1000,
                "chains": 2,
                "cores": 2,
                "nuts_kwargs": {"target_accept": 0.95},
            },
        )

        model.fit()
        print(model.fit_summary())

        # Predict spend of customers for which we know transaction history, conditioned on data.
        expected_customer_spend = (
            model.expected_customer_spend(
                data=pandas.DataFrame(
                    {
                        "customer_id": [0, 1, 2, 3, ...],
                        "monetary_value": [23.5, 19.3, 11.2, 100.5, ...],
                        "frequency": [6, 8, 2, 1, ...],
                    }
                ),
            ),
        )
        print(expected_customer_spend.mean("customer_id"))

        # Predict spend of 10 new customers, conditioned on data
        new_customer_spend = model.expected_new_customer_spend(n=10)
        print(new_customer_spend.mean("new_customer_id"))

    References
    ----------
    .. [1] Fader, P. S., & Hardie, B. G. (2013). "The Gamma-Gamma model of monetary
           value". https://www.brucehardie.com/notes/025/gamma_gamma.pdf
    .. [2] Peter S. Fader, Bruce G. S. Hardie, and Ka Lok Lee (2005), “RFM and CLV:
           Using iso-value curves for customer base analysis”, Journal of Marketing
           Research, 42 (November), 415-430.
           https://journals.sagepub.com/doi/pdf/10.1509/jmkr.2005.42.4.415

    """

    _model_type = "Gamma-Gamma Model (Mean Transactions)"

    def __init__(
        self,
        data: pandas.DataFrame,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ):
        self._validate_cols(
            data,
            required_cols=["customer_id", "monetary_value", "frequency"],
            must_be_unique=["customer_id"],
        )
        super().__init__(
            data=data, model_config=model_config, sampler_config=sampler_config
        )

    @property
    def default_model_config(self) -> ModelConfig:
        """Default model configuration."""
        return {
            "p": Prior("HalfFlat"),
            "q": Prior("HalfFlat"),
            "v": Prior("HalfFlat"),
        }

    def build_model(self) -> None:  # type: ignore[override]
        """Build the model."""
        z_mean = pt.as_tensor_variable(self.data["monetary_value"])
        x = pt.as_tensor_variable(self.data["frequency"])

        coords = {"customer_id": self.data["customer_id"]}
        with pm.Model(coords=coords) as self.model:
            p = self.model_config["p"].create_variable("p")
            q = self.model_config["q"].create_variable("q")
            v = self.model_config["v"].create_variable("v")

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


# TODO: This model requires further evaluation and reference in a notebook
class GammaGammaModelIndividual(BaseGammaGammaModel):
    """Gamma-Gamma Model for expected future monetary value.

    The Gamma-Gamma model assumes expected future spend follows a Gamma distribution,
    and the scale of this distribution is also Gamma-distributed.

    This model is conditioned on the spend values of each purchase for each customer,
    and is based on [1]_, [2]_.

    See `GammaGammaModel` for an equivalent model conditioned on mean transaction values
    of repeat purchases for the customer population.

    Parameters
    ----------
    data : ~pandas.DataFrame
        Dataframe containing the following columns:

        * `customer_id`: Unique customer identifier
        * `individual_transaction_value`: Monetary value of each purchase for each customer
    model_config : dict, optional
        Dictionary of model prior parameters. If not provided, the model will use default priors specified in the
        `default_model_config` class attribute.
    sampler_config : dict, optional
        Dictionary of sampler parameters. Defaults to *None*.


    Examples
    --------

    .. code-block:: python

        import pymc as pm
        from pymc_marketing.clv import GammaGammaModelIndividual

        model = GammaGammaModelIndividual(
            data=pandas.DataFrame(
                {
                "customer_id": [0, 0, 0, 1, 1, 2, ...],
                "individual_transaction_value": [5.3. 5.7, 6.9, 13.5, 0.3, 19.2 ...],
                }
            ),
            model_config={
                "p": {dist: 'HalfNorm', kwargs: {}},
                "q": {dist: 'HalfStudentT', kwargs: {"nu": 4, "sigma": 10}},
                "v": {dist: 'HalfCauchy', kwargs: {}},
            },
            sampler_config={
                "draws": 1000,
                "tune": 1000,
                "chains": 2,
                "cores": 2,
                "nuts_kwargs": {"target_accept": 0.95},
            },
        )

        model.fit()
        print(model.fit_summary())

        # Predict spend of customers for which we know transaction history,
        # conditioned on data. May include customers not included in fitting
        expected_customer_spend = model.expected_customer_spend(
            data=pandas.DataFrame(
                {
                "customer_id": [0, 0, 0, 1, 1, 2, ...],
                "individual_transaction_value": [5.3. 5.7, 6.9, 13.5, 0.3, 19.2 ...],
                }
            ),
        )
        print(expected_customer_spend.mean("customer_id"))

        # Predict spend of 10 new customers, conditioned on data
        new_customer_spend = model.expected_new_customer_spend(n=10)
        print(new_customer_spend.mean("new_customer_id"))

    References
    ----------
    .. [1] Fader, P. S., & Hardie, B. G. (2013). "The Gamma-Gamma model of monetary
           value". http://www.brucehardie.com/notes/025/gamma_gamma.pdf
    .. [2] Peter S. Fader, Bruce G. S. Hardie, and Ka Lok Lee (2005), “RFM and CLV:
           Using iso-value curves for customer base analysis”, Journal of Marketing
           Research, 42 (November), 415-430.
           https://journals.sagepub.com/doi/pdf/10.1509/jmkr.2005.42.4.415

    """

    _model_type = "Gamma-Gamma Model (Individual Transactions)"

    def __init__(
        self,
        data: pandas.DataFrame,
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
        """Default model configuration."""
        return {
            "p": Prior("HalfFlat"),
            "q": Prior("HalfFlat"),
            "v": Prior("HalfFlat"),
        }

    def build_model(self) -> None:  # type: ignore[override]
        """Build the model."""
        z = self.data["individual_transaction_value"]

        coords = {
            "customer_id": np.unique(self.data["customer_id"]),
            "obs": range(self.data.shape[0]),
        }
        with pm.Model(coords=coords) as self.model:
            p = self.model_config["p"].create_variable("p")
            q = self.model_config["q"].create_variable("q")
            v = self.model_config["v"].create_variable("v")

            nu = pm.Gamma("nu", q, v, dims=("customer_id",))
            pm.Gamma(
                "spend", p, nu[self.data["customer_id"]], observed=z, dims=("obs",)
            )
