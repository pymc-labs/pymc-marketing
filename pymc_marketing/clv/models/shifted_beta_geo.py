from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import pymc as pm
from pymc.util import RandomState
from pytensor.tensor import TensorVariable
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
    customer_id: array_like
        Customer labels. There should be one unique label for each customer
    t_churn: array_like
        Time at which the customer cancelled the contract (starting at 0).
        It should  equal T for users that have not cancelled by the end of the
        observation period
    T: array_like
        Maximum observed time period (starting at 0)
    alpha_prior: scalar PyMC distribution, optional
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.HalfFlat.dist()`
    beta_prior: scalar PyMC distribution, optional
        PyMC prior distribution, created via `.dist()` API. Defaults to
        `pm.HalfFlat.dist()`


    Examples
    --------
        .. code-block:: python

            import pymc as pm
            from pymc_marketing.clv import ShiftedBetaGeoModelIndividual

            model = ShiftedBetaGeoModelIndividual(
                customer_id=[0, 1, 2, 3, ...],
                t_churn=[1, 2, 8, 4, 8 ...],
                T=8,  # Can also be an array with one value per customer
                alpha_prior=pm.HalfNormal.dist(10),
                beta_prior=pm.HalfNormal.dist(10),
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

    _model_name = "Shifted-Beta-Geometric Model (Individual Customers)"

    def __init__(
        self,
        customer_id: Union[np.ndarray, pd.Series],
        t_churn: Union[np.ndarray, pd.Series],
        T: Union[np.ndarray, pd.Series],
        alpha_prior: Optional[TensorVariable] = None,
        beta_prior: Optional[TensorVariable] = None,
    ):
        super().__init__()

        t_churn = np.asarray(t_churn)
        T = np.asarray(T)

        if np.any((t_churn < 0) | (t_churn > T) | np.isnan(t_churn)):
            raise ValueError(
                "t_churn must respect 0 < t_churn <= T.\n",
                "Customers that are still alive should have t_churn = T",
            )

        alpha_prior, beta_prior = self._process_priors(alpha_prior, beta_prior)

        coords = {"customer_id": np.asarray(customer_id)}
        with pm.Model(coords=coords) as self.model:
            alpha = self.model.register_rv(alpha_prior, name="alpha")
            beta = self.model.register_rv(beta_prior, name="beta")

            theta = pm.Beta("theta", alpha, beta, dims=("customer_id",))

            churn_raw = pm.Geometric.dist(theta)

            pm.Censored(
                "churn_censored",
                churn_raw,
                lower=None,
                upper=T,
                observed=t_churn,
                dims=("customer_id",),
            )

    def _process_priors(self, alpha_prior, beta_prior):
        if alpha_prior is None:
            alpha_prior = pm.HalfFlat.dist()
        else:
            self._check_prior_ndim(alpha_prior)
        if beta_prior is None:
            beta_prior = pm.HalfFlat.dist()
        else:
            self._check_prior_ndim(beta_prior)

        return super()._process_priors(alpha_prior, beta_prior)

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
                self.fit_result,
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
                self.fit_result,
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
