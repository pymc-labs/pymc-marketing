from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pymc as pm
from pymc.util import RandomState
from xarray import DataArray, Dataset

from pymc_marketing.clv.models import CLVModel


class ShiftedBetaGeoModelIndividual(CLVModel):
    """Shifted Beta Geometric model

    Model for customer behavior in a discrete contractual setting. It assumes that:
      * At the end of each period, a customer has a probability `theta` of renewing the contract
        and `1-theta` of cancelling
      * The probability `theta` does not change over time for a given customer
      * The probability `theta` varies across customers according to a Beta prior distribution
        with hyperparameters `alpha` and `beta`.

    based on [1]_.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing the following columns:
            * `customer_id`: Customer labels. There should be one unique label for each customer
            * `t_churn`: Time at which the customer cancelled the contract (starting at 0).
        It should  equal T for users that have not cancelled by the end of the
        observation period
            * `T`: Maximum observed time period (starting at 0)
    model_config: dict, optional
        Dictionary of model prior parameters. If not provided, the model will use default priors specified in the `default_model_config` class attribute.
    sampler_config: dict, optional
        Dictionary of sampler parameters. Defaults to None.


    Examples
    --------
        .. code-block:: python

            import pymc as pm
            from pymc_marketing.clv import ShiftedBetaGeoModelIndividual

            model = ShiftedBetaGeoModelIndividual(
                data=pd.DataFrame({
                    customer_id=[0, 1, 2, 3, ...],
                    t_churn=[1, 2, 8, 4, 8 ...],
                    T=[8 for x in range(len(customer_id))],
                }),
                model_config={
                    "alpha_prior": {"dist": "HalfNormal", "kwargs": {"sigma": 10}},
                    "beta_prior": {"dist": "HalfStudentT", "kwargs": {"nu": 4, "sigma": 10}},
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

            # Predict how many periods in the future are existing customers
            likely to cancel (ignoring that some may already have cancelled)
            expected_churn_time = model.distribution_customer_churn_time(
                customer_id=[0, 1, 2, 3, ...],
            )
            print(expected_churn_time.mean("customer_id"))

            # Predict churn time for 10 new customers, conditioned on data
            new_customers_churn_time = model.distribution_new_customer_churn_time(n=10)
            print(new_customers_churn_time.mean("new_customer_id"))


    References
    ----------
    .. [1] Fader, P. S., & Hardie, B. G. (2007). How to project customer retention.
           Journal of Interactive Marketing, 21(1), 76-90.
           https://journals.sagepub.com/doi/pdf/10.1002/dir.20074
    """

    _model_type = "Shifted-Beta-Geometric Model (Individual Customers)"

    def __init__(
        self,
        data: pd.DataFrame,
        model_config: Optional[Dict] = None,
        sampler_config: Optional[Dict] = None,
    ):
        try:
            self.customer_id = data["customer_id"]
        except KeyError:
            raise KeyError("data must contain a 'customer_id' column")
        try:
            self.t_churn: np.ndarray = np.asarray(data["t_churn"])
        except KeyError:
            raise KeyError("data must contain a 't_churn' column")
        try:
            self.T: np.ndarray = np.asarray(data["T"])
        except KeyError:
            raise KeyError("data must contain a 'T' column")
        super().__init__(model_config=model_config, sampler_config=sampler_config)
        self.data = data
        self.alpha_prior = self._create_distribution(self.model_config["alpha_prior"])
        self.beta_prior = self._create_distribution(self.model_config["beta_prior"])
        self._process_priors(self.alpha_prior, self.beta_prior)

        if np.any(
            (self.t_churn < 0) | (self.t_churn > self.T) | np.isnan(self.t_churn)
        ):
            raise ValueError(
                "t_churn must respect 0 < t_churn <= T.\n",
                "Customers that are still alive should have t_churn = T",
            )
        self.coords = {"customer_id": np.asarray(self.customer_id)}

    @property
    def default_model_config(self) -> Dict:
        return {
            "alpha_prior": {"dist": "HalfFlat", "kwargs": {}},
            "beta_prior": {"dist": "HalfFlat", "kwargs": {}},
        }

    def build_model(  # type: ignore
        self,
    ) -> None:
        with pm.Model(coords=self.coords) as self.model:
            alpha = self.model.register_rv(self.alpha_prior, name="alpha")
            beta = self.model.register_rv(self.beta_prior, name="beta")

            theta = pm.Beta("theta", alpha, beta, dims=("customer_id",))

            churn_raw = pm.Geometric.dist(theta)
            pm.Censored(
                "churn_censored",
                churn_raw,
                lower=None,
                upper=self.T,
                observed=self.t_churn,
                dims=("customer_id",),
            )

    def distribution_customer_churn_time(
        self, customer_id: Union[np.ndarray, pd.Series], random_seed: RandomState = None
    ) -> DataArray:
        """Sample distribution of churn time for existing customers.

        The draws represent the number of periods into the future after which
        a customer cancels their contract.

        It ignores that some customers may have already cancelled.
        """

        coords = {"customer_id": np.asarray(customer_id)}
        with pm.Model(coords=coords):
            alpha = pm.HalfFlat("alpha")
            beta = pm.HalfFlat("beta")

            theta = pm.Beta("theta", alpha, beta, dims=("customer_id",))
            pm.Geometric("churn", theta, dims=("customer_id",))

            return pm.sample_posterior_predictive(
                self.idata,
                var_names=["churn"],
                random_seed=random_seed,
            ).posterior_predictive["churn"]

    def _distribution_new_customer(
        self,
        n: int = 1,
        random_seed: RandomState = None,
        var_names: Sequence[str] = ("theta", "churn"),
    ) -> Dataset:
        coords = {"new_customer_id": np.arange(n)}
        with pm.Model(coords=coords):
            alpha = pm.HalfFlat("alpha")
            beta = pm.HalfFlat("beta")

            theta = pm.Beta("theta", alpha, beta, dims=("new_customer_id",))
            pm.Geometric("churn", theta, dims=("new_customer_id",))

            return pm.sample_posterior_predictive(
                self.idata,
                var_names=var_names,
                random_seed=random_seed,
            ).posterior_predictive

    def distribution_new_customer_churn_time(
        self, n: int = 1, random_seed: RandomState = None
    ) -> DataArray:
        """Sample distribution of churn time for new customers.

        The draws represent the number of periods into the future after which
        a customer cancels their contract.

        Use `n > 1` to simulate multiple identically distributed users.
        """
        return self._distribution_new_customer(
            n=n, random_seed=random_seed, var_names=["churn"]
        )["churn"]

    def distribution_new_customer_theta(
        self, n: int = 1, random_seed: RandomState = None
    ) -> DataArray:
        """Sample distribution of theta parameter for new customers.

        Use `n > 1` to simulate multiple identically distributed users.
        """
        return self._distribution_new_customer(
            n=n, random_seed=random_seed, var_names=["theta"]
        )["theta"]
